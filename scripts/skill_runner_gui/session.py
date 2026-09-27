#!/usr/bin/env python3
"""Habitat skill-runner session owned by a dedicated worker thread."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import omegaconf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import open_dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GUI_DIR))

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.examples.example_utils import execute_skill
from habitat_llm.utils import fix_config, setup_config
from habitat_llm.utils.sim import init_agents

import inspectors

SKILLS = {
    "Navigate": "Navigate <agent_index> <entity_name>",
    "Explore": "Explore <agent_index> <room_name>",
    "Open": "Open <agent_index> <entity_name>",
    "Close": "Close <agent_index> <entity_name>",
    "Pick": "Pick <agent_index> <entity_name>",
    "Place": "Place <agent_index> <entity,relation,furniture,constraint,reference>",
    "Fill": "Fill <agent_index> <entity_name>",
    "Pour": "Pour <agent_index> <entity_name>",
    "Clean": "Clean <agent_index> <entity_name>",
    "PowerOn": "PowerOn <agent_index> <entity_name>",
    "PowerOff": "PowerOff <agent_index> <entity_name>",
}

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "skill_runner_gui"
SEED = 47668090


def _graph_for_ui(graph: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop the bulky text dump so status polls stay small."""
    if not graph:
        return graph
    return {k: v for k, v in graph.items() if k != "text"}


def _json_safe(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _action_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_action: Dict[str, Dict[str, Any]] = {}
    total_steps = 0
    ok_count = 0
    for entry in history:
        skill = (entry.get("command") or "").split()[0] or "unknown"
        bucket = per_action.setdefault(
            skill, {"call_count": 0, "ok_count": 0, "sim_steps": []}
        )
        bucket["call_count"] += 1
        steps = int(entry.get("skill_steps") or 0)
        bucket["sim_steps"].append(steps)
        total_steps += steps
        if entry.get("ok"):
            bucket["ok_count"] += 1
            ok_count += 1
    for bucket in per_action.values():
        bucket["total_sim_steps"] = sum(bucket["sim_steps"])
    return {
        "total_actions": len(history),
        "ok_count": ok_count,
        "failed_count": len(history) - ok_count,
        "total_sim_steps": total_steps,
        "per_action": per_action,
        "path": [e.get("command") for e in history],
    }


def _combine_videos(clips: List[Path], out_path: Path) -> Optional[str]:
    """Concatenate MP4 clips into one file. Prefer ffmpeg, fall back to imageio."""
    existing = [p for p in clips if p.exists()]
    if not existing:
        return None
    if len(existing) == 1:
        shutil.copy2(existing[0], out_path)
        return str(out_path)

    list_file = out_path.parent / "_concat.txt"
    list_file.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in existing)
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        return str(out_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        import imageio

        writer = imageio.get_writer(str(out_path), fps=30, quality=4)
        try:
            for clip in existing:
                reader = imageio.get_reader(str(clip))
                try:
                    for frame in reader:
                        writer.append_data(frame)
                finally:
                    reader.close()
        finally:
            writer.close()
        return str(out_path)
    finally:
        if list_file.exists():
            list_file.unlink()


class SkillRunnerSession:
    """
    Owns Habitat + planner on a single worker thread.

    Flask request threads submit work via call_on_worker() and wait for results.
    Frame bytes are published lock-free for the MJPEG streamer.
    """

    def __init__(self, results_dir: Optional[Path] = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "videos").mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._cmd_cond = threading.Condition(self._lock)
        self._pending: Optional[Tuple[str, Dict[str, Any]]] = None
        self._result: Any = None
        self._error: Optional[str] = None
        self._done = False

        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        self.status = "idle"  # idle | loading | running | error
        self.status_message = ""
        self.loaded = False
        self.episode_info: Dict[str, Any] = {}
        self.command_history: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.command_index = 0
        self._cached_entities: Optional[Dict[str, Any]] = None
        self._cached_graph: Optional[Dict[str, Any]] = None
        self._cached_gt_graph: Optional[Dict[str, Any]] = None
        self._cached_robot_graph: Optional[Dict[str, Any]] = None

        self.config = None
        self.env_interface = None
        self.planner = None
        self.active_world_graph = None

        self._stop = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ logging
    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.logs.append(line)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]
        print(line, flush=True)

    # ------------------------------------------------------------------ frames
    def set_frame_rgb(self, frame_rgb: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        if not ok:
            return
        with self._frame_lock:
            self._latest_jpeg = buf.tobytes()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    # ------------------------------------------------------------------ worker
    def call_on_worker(self, op: str, **kwargs) -> Any:
        """Queue work on the Habitat thread and block until finished."""
        with self._cmd_cond:
            while self._pending is not None and not self._stop:
                self._cmd_cond.wait(timeout=0.1)
            self._pending = (op, kwargs)
            self._result = None
            self._error = None
            self._done = False
            self._cmd_cond.notify_all()
            while not self._done and not self._stop:
                self._cmd_cond.wait(timeout=0.5)
            if self._error:
                raise RuntimeError(self._error)
            return self._result

    def _worker_loop(self) -> None:
        while not self._stop:
            with self._cmd_cond:
                while self._pending is None and not self._stop:
                    self._cmd_cond.wait(timeout=0.2)
                if self._stop:
                    break
                op, kwargs = self._pending
            try:
                if op == "load":
                    result = self._do_load(**kwargs)
                elif op == "run_skill":
                    result = self._do_run_skill(**kwargs)
                elif op == "inspect":
                    result = self._do_inspect(**kwargs)
                elif op == "snapshot":
                    result = self._do_snapshot()
                elif op == "exit":
                    result = self._do_exit()
                else:
                    raise ValueError(f"Unknown op: {op}")
                err = None
            except Exception as exc:
                result = None
                err = f"{exc}\n{traceback.format_exc()}"
                self.status = "error"
                self.status_message = str(exc)
                self._log(f"ERROR: {exc}")

            with self._cmd_cond:
                self._result = result
                self._error = err
                self._pending = None
                self._done = True
                self._cmd_cond.notify_all()

    def shutdown(self) -> None:
        self._stop = True
        with self._cmd_cond:
            self._cmd_cond.notify_all()
        self._close_env()

    def _close_env(self) -> None:
        if self.env_interface is None:
            return
        try:
            if hasattr(self.env_interface, "env") and self.env_interface.env is not None:
                self.env_interface.env.close()
        except Exception as exc:
            self._log(f"Warning closing env: {exc}")
        self.env_interface = None
        self.planner = None
        self.active_world_graph = None
        self.loaded = False
        self._cached_entities = None
        self._cached_graph = None
        self._cached_gt_graph = None
        self._cached_robot_graph = None

    # ------------------------------------------------------------------ config
    def _normalize_data_path(
        self, config: omegaconf.DictConfig, data_path: str
    ) -> omegaconf.DictConfig:
        data_path_str = str(data_path)
        path = Path(data_path_str)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
            data_path_str = str(path)

        runtime_config_path = None
        if path.is_dir():
            folder = path
            json_gz = sorted(folder.glob("*.json.gz"))
            json_files = sorted(folder.glob("*.json"))
            chosen = json_gz[0] if json_gz else (json_files[0] if json_files else None)
            if chosen is None:
                raise FileNotFoundError(f"No episode JSON in {folder}")
            data_path_str = str(chosen)
            yaml_files = sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")))
            if yaml_files:
                runtime_config_path = str(yaml_files[0])
        elif data_path_str.endswith(".json") and not data_path_str.endswith(".json.gz"):
            gz = data_path_str + ".gz"
            if Path(gz).exists():
                data_path_str = gz
            parent = Path(data_path_str).parent
            yaml_files = sorted(list(parent.glob("*.yaml")) + list(parent.glob("*.yml")))
            if yaml_files:
                runtime_config_path = str(yaml_files[0])
        else:
            parent = Path(data_path_str).parent
            if parent.is_dir():
                yaml_files = sorted(
                    list(parent.glob("*.yaml")) + list(parent.glob("*.yml"))
                )
                if yaml_files:
                    runtime_config_path = str(yaml_files[0])

        with open_dict(config):
            config.habitat.dataset.data_path = data_path_str
            if runtime_config_path and (
                not hasattr(config, "runtime_config_path") or not config.runtime_config_path
            ):
                config.runtime_config_path = runtime_config_path
            config.paths.results_dir = str(self.results_dir)
        return config

    def _compose_config(
        self,
        data_path: str,
        episode_id: Optional[str] = None,
        episode_index: Optional[int] = None,
        runtime_config_path: Optional[str] = None,
    ) -> omegaconf.DictConfig:
        conf_dir = str(PROJECT_ROOT / "habitat_llm" / "conf")
        overrides = [
            "hydra.run.dir=.",
            "+skill_runner_show_videos=False",
            "evaluation.save_video=True",
            f"evaluation.output_dir={self.results_dir}",
            f"paths.results_dir={self.results_dir}",
            f"habitat.dataset.data_path={data_path}",
        ]
        if episode_id is not None:
            overrides.append(f"+skill_runner_episode_id={episode_id}")
        if episode_index is not None:
            overrides.append(f"+skill_runner_episode_index={episode_index}")
        if runtime_config_path:
            overrides.append(f"runtime_config_path={runtime_config_path}")

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            config = compose(
                config_name="examples/skill_runner_default_config.yaml",
                overrides=overrides,
            )

        from hydra.core.hydra_config import HydraConfig

        # Emulate @hydra.main so any remaining ${hydra:...} interpolations resolve.
        with open_dict(config):
            config.hydra = omegaconf.OmegaConf.create(
                {"runtime": {"output_dir": str(self.results_dir)}}
            )
        HydraConfig().cfg = config

        fix_config(config)
        with open_dict(config):
            config_dict = omegaconf.OmegaConf.create(
                omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
            )
            config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
            config.habitat = config_dict
        config = setup_config(config, SEED)
        config = self._normalize_data_path(config, data_path)
        if runtime_config_path:
            with open_dict(config):
                config.runtime_config_path = runtime_config_path
        return config

    def _apply_runtime_config(self) -> None:
        if self.config is None or self.active_world_graph is None:
            return
        runtime_config = None
        if hasattr(self.config, "runtime_config_path") and self.config.runtime_config_path:
            import yaml

            runtime_config_path = Path(self.config.runtime_config_path)
            if runtime_config_path.exists():
                self._log(f"Loading runtime config: {runtime_config_path}")
                with open(runtime_config_path, "r") as f:
                    runtime_config = yaml.safe_load(f)
            else:
                self._log(f"Runtime config not found: {runtime_config_path}")
                return

        if not (
            self.config.world_model.partial_obs
            and runtime_config
            and hasattr(self.active_world_graph, "add_object_to_graph")
        ):
            return

        from habitat_llm.world_model import SpotRobot

        sim = self.env_interface.sim
        if (
            "runtime_robot_location" in runtime_config
            and "room" in runtime_config["runtime_robot_location"]
        ):
            target_room = runtime_config["runtime_robot_location"]["room"]
            self._log(f"Moving robot to {target_room}")
            self.active_world_graph.move_robot_to_room(
                target_room, verbose=True, sim=sim
            )

        if "runtime_objects" in runtime_config:
            runtime_objs = runtime_config["runtime_objects"]
            if runtime_objs.get("enabled", False) and "objects" in runtime_objs:
                for obj_config in runtime_objs["objects"]:
                    obj_class = obj_config.get("class")
                    furniture_name = obj_config.get("furniture_name")
                    position = obj_config.get("position")
                    rotation = obj_config.get("rotation")
                    if not obj_class:
                        continue
                    try:
                        if furniture_name:
                            self.active_world_graph.add_object_to_graph(
                                object_class=obj_class,
                                furniture_name=furniture_name,
                                connect_to_entities=True,
                                verbose=True,
                            )
                        elif position:
                            self.active_world_graph.add_object_to_graph(
                                object_class=obj_class,
                                position=position,
                                rotation=rotation if rotation else [0, 0, 0, 1],
                                connect_to_entities=True,
                                verbose=True,
                            )
                    except Exception as exc:
                        self._log(f"Failed to add {obj_class}: {exc}")

    def _do_load(
        self,
        data_path: str,
        episode_id: Optional[str] = None,
        episode_index: Optional[int] = None,
        runtime_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.status = "loading"
        self.status_message = f"Loading {data_path}..."
        self._log(self.status_message)
        self._close_env()
        with self._frame_lock:
            self._latest_jpeg = None
        self.command_history = []
        self.command_index = 0

        os.chdir(PROJECT_ROOT)
        config = self._compose_config(
            data_path=data_path,
            episode_id=episode_id,
            episode_index=episode_index,
            runtime_config_path=runtime_config_path,
        )
        self.config = config

        register_sensors(config)
        register_actions(config)
        register_measures(config)

        dataset = CollaborationDatasetV0(config.habitat.dataset)
        self._log(f"Dataset path: {config.habitat.dataset.data_path}")
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        if hasattr(config, "skill_runner_episode_index"):
            env_interface.env.habitat_env.episode_iterator.set_next_episode_by_index(
                config.skill_runner_episode_index
            )
        elif hasattr(config, "skill_runner_episode_id"):
            env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
                str(config.skill_runner_episode_id)
            )
        env_interface.reset_environment()

        planner_conf = config.evaluation.planner
        planner = instantiate(planner_conf)
        planner = planner(env_interface=env_interface)
        planner.agents = init_agents(config.evaluation.agents, env_interface)
        planner.reset()

        self.env_interface = env_interface
        self.planner = planner
        agent_uid = config.robot_agent_uid
        self.active_world_graph = env_interface.world_graph[agent_uid]

        self._apply_runtime_config()

        # Capture an initial third-person frame if available
        try:
            from habitat_llm.examples.example_utils import DebugVideoUtil

            obs = env_interface.get_observations()
            dvu = DebugVideoUtil(env_interface, str(self.results_dir))
            frame = dvu.get_combined_frames(obs)
            self.set_frame_rgb(np.ascontiguousarray(frame))
        except Exception as exc:
            self._log(f"Initial frame unavailable: {exc}")

        ep = env_interface.sim.ep_info
        instruction = getattr(ep, "instruction", "") or ""
        self.episode_info = {
            "episode_id": str(ep.episode_id),
            "scene_id": str(ep.scene_id),
            "instruction": instruction,
            "data_path": str(config.habitat.dataset.data_path),
            "runtime_config_path": getattr(config, "runtime_config_path", None),
            "info": dict(ep.info) if getattr(ep, "info", None) else {},
        }
        self.loaded = True
        self.status = "idle"
        self.status_message = f"Loaded episode {ep.episode_id}"
        self._log(self.status_message)
        self._refresh_world_caches()
        return {
            "ok": True,
            "episode": self.episode_info,
            "graph": self._cached_gt_graph,
            "gt_graph": self._cached_gt_graph,
            "robot_graph": self._cached_robot_graph,
        }

    def _do_run_skill(
        self,
        skill: str,
        agent_index: int,
        target: str,
    ) -> Dict[str, Any]:
        if not self.loaded or self.planner is None:
            raise RuntimeError("No episode loaded")
        if skill not in SKILLS:
            raise ValueError(f"Unknown skill: {skill}")
        if agent_index not in (0, 1):
            raise ValueError("agent_index must be 0 or 1")

        self.status = "running"
        command = f"{skill} {agent_index} {target}"
        self.status_message = f"Running: {command}"
        self._log(self.status_message)

        high_level = {int(agent_index): (skill, target, None)}
        try:
            responses, step_info, _frames = execute_skill(
                high_level,
                self.planner,
                make_video=True,
                vid_postfix=f"{self.command_index}_",
                play_video=False,
                frame_callback=self.set_frame_rgb,
            )
            response = responses.get(int(agent_index), "")
            steps = int(step_info.get("skill_steps", 0))
            video_rel = f"videos/video-{self.command_index}_.mp4"
            video_path = self.results_dir / video_rel
            entry = {
                "command": command,
                "skill": skill,
                "agent_index": int(agent_index),
                "target": target,
                "response": response,
                "skill_steps": steps,
                "ok": True,
                "video": video_rel if video_path.exists() else None,
            }
            self.command_history.append(entry)
            self.command_index += 1
            self.status = "idle"
            self.status_message = f"{skill} done: {response}"
            self._log(self.status_message)
            self._refresh_world_caches()
            return entry
        except Exception as exc:
            entry = {
                "command": command,
                "skill": skill,
                "agent_index": int(agent_index),
                "target": target,
                "response": str(exc),
                "skill_steps": 0,
                "ok": False,
                "video": None,
            }
            self.command_history.append(entry)
            self.command_index += 1
            self.status = "idle"
            self.status_message = f"Skill failed: {exc}"
            self._log(self.status_message)
            return entry

    def _do_inspect(self, kind: str, name: Optional[str] = None) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("No episode loaded")
        result = inspectors.inspect(
            kind, self.active_world_graph, self.env_interface, name
        )
        if kind == "entities":
            self._cached_entities = result
        if kind == "graph":
            self._cached_robot_graph = _graph_for_ui(result)
        if kind == "gt":
            self._cached_gt_graph = _graph_for_ui(result)
            self._cached_graph = self._cached_gt_graph
        return result

    def _refresh_world_caches(self) -> None:
        if self.active_world_graph is None:
            self._cached_entities = None
            self._cached_graph = None
            self._cached_gt_graph = None
            self._cached_robot_graph = None
            return
        try:
            self._cached_gt_graph = _graph_for_ui(
                inspectors.get_gt_graph(self.env_interface)
            )
        except Exception as exc:
            self._log(f"GT graph cache failed: {exc}")
            self._cached_gt_graph = None
        try:
            self._cached_robot_graph = _graph_for_ui(
                inspectors.get_robot_graph(self.active_world_graph, self.env_interface)
            )
        except Exception as exc:
            self._log(f"Robot graph cache failed: {exc}")
            self._cached_robot_graph = None
        self._cached_graph = self._cached_gt_graph
        try:
            entity_graph = None
            if (
                self.env_interface
                and getattr(self.env_interface, "perception", None)
            ):
                entity_graph = self.env_interface.perception.gt_graph
            self._cached_entities = inspectors.get_entities(
                entity_graph or self.active_world_graph
            )
        except Exception as exc:
            self._log(f"Entity cache failed: {exc}")
            self._cached_entities = None

    def _do_exit(self) -> Dict[str, Any]:
        stats = _action_stats(self.command_history)
        history = list(self.command_history)
        episode = dict(self.episode_info)
        self._log("Exiting. Command History:")
        for ix, entry in enumerate(history):
            self._log(
                f" [{ix}]: '{entry.get('command')}' -> '{entry.get('response')}'"
            )
        self._log(f"Total actions: {stats['total_actions']}")
        self._log(f"Total sim steps: {stats['total_sim_steps']}")
        for action_name, action_stats in stats.get("per_action", {}).items():
            self._log(f"  {action_name!r}: {action_stats}")
        self._close_env()
        self.status = "idle"
        self.status_message = "exited"
        return {
            "ok": True,
            "exited": True,
            "history": history,
            "stats": stats,
            "episode": episode,
        }

    def _do_snapshot(self) -> Dict[str, Any]:
        return self._status_dict(include_entities=True)

    def _status_dict(self, include_entities: bool = True) -> Dict[str, Any]:
        return {
            "status": self.status,
            "status_message": self.status_message,
            "loaded": self.loaded,
            "episode": self.episode_info,
            "history": list(self.command_history),
            "logs": list(self.logs[-100:]),
            "skills": SKILLS,
            "entities": self._cached_entities if include_entities else None,
            "graph": self._cached_gt_graph if include_entities else None,
            "gt_graph": self._cached_gt_graph if include_entities else None,
            "robot_graph": self._cached_robot_graph if include_entities else None,
            "has_frame": self.get_latest_jpeg() is not None,
            "results_dir": str(self.results_dir),
        }

    # ------------------------------------------------------------------ public
    def get_status(self) -> Dict[str, Any]:
        # Never block the Habitat worker for status polls.
        return self._status_dict(include_entities=True)

    def load_episode(self, **kwargs) -> Dict[str, Any]:
        return self.call_on_worker("load", **kwargs)

    def run_skill(self, **kwargs) -> Dict[str, Any]:
        return self.call_on_worker("run_skill", **kwargs)

    def inspect(self, kind: str, name: Optional[str] = None) -> Dict[str, Any]:
        return self.call_on_worker("inspect", kind=kind, name=name)

    def exit_session(self) -> Dict[str, Any]:
        return self.call_on_worker("exit")

    def list_videos(self) -> List[Dict[str, Any]]:
        videos_dir = self.results_dir / "videos"
        if not videos_dir.exists():
            return []
        items = []
        for path in sorted(videos_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            items.append(
                {
                    "name": path.name,
                    "path": str(path.relative_to(self.results_dir)),
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                }
            )
        return items

    def save_run(self) -> Dict[str, Any]:
        """Write a snapshot of this session: combined video, path, logs, graph, stats."""
        if not self.loaded:
            raise RuntimeError("Load an episode first")
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        episode_id = str(self.episode_info.get("episode_id") or "unknown")
        save_dir = self.results_dir / "saves" / f"{stamp}_ep{episode_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "clips").mkdir(exist_ok=True)

        history = list(self.command_history)
        stats = _action_stats(history)
        bundle = {
            "saved_at": stamp,
            "episode": _json_safe(self.episode_info),
            "history": _json_safe(history),
            "stats": stats,
            "logs": list(self.logs),
            "entities": _json_safe(self._cached_entities),
            "graph": _json_safe(self._cached_gt_graph),
            "gt_graph": _json_safe(self._cached_gt_graph),
            "robot_graph": _json_safe(self._cached_robot_graph),
        }

        (save_dir / "run.json").write_text(json.dumps(bundle, indent=2))
        (save_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
        (save_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        (save_dir / "logs.txt").write_text("\n".join(self.logs) + ("\n" if self.logs else ""))
        path_lines = [
            f"# episode {episode_id}",
            f"# scene {self.episode_info.get('scene_id', '')}",
            f"# {self.episode_info.get('instruction', '')}",
            "",
        ]
        path_lines.extend(e.get("command", "") for e in history)
        (save_dir / "path.txt").write_text("\n".join(path_lines) + "\n")
        if self._cached_gt_graph is not None:
            (save_dir / "gt_graph.json").write_text(
                json.dumps(self._cached_gt_graph, indent=2, default=str)
            )
        if self._cached_robot_graph is not None:
            (save_dir / "robot_graph.json").write_text(
                json.dumps(self._cached_robot_graph, indent=2, default=str)
            )
        if self._cached_graph is not None:
            (save_dir / "graph.json").write_text(
                json.dumps(self._cached_graph, indent=2, default=str)
            )
        if self._cached_entities is not None:
            (save_dir / "entities.json").write_text(
                json.dumps(self._cached_entities, indent=2, default=str)
            )

        clips: List[Path] = []
        for entry in history:
            rel = entry.get("video")
            if not rel:
                continue
            src = self.results_dir / rel
            if src.exists():
                dest = save_dir / "clips" / src.name
                shutil.copy2(src, dest)
                clips.append(src)

        combined_rel = None
        if clips:
            combined = save_dir / "combined.mp4"
            try:
                _combine_videos(clips, combined)
            except Exception as exc:
                self._log(f"Combined video failed: {exc}")
            if combined.exists():
                combined_rel = str(combined.relative_to(self.results_dir))
                public = self.results_dir / "videos" / f"combined-{stamp}.mp4"
                shutil.copy2(combined, public)

        files = sorted(
            str(p.relative_to(self.results_dir))
            for p in save_dir.rglob("*")
            if p.is_file()
        )
        self._log(f"Saved run to {save_dir}")
        return {
            "ok": True,
            "save_dir": str(save_dir),
            "relative_dir": str(save_dir.relative_to(self.results_dir)),
            "combined_video": combined_rel,
            "files": files,
            "stats": stats,
        }
