#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import ast
import csv
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import hydra
import yaml
from omegaconf import OmegaConf, open_dict
from torch import multiprocessing as mp

from scripts import view_trace_logs
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
)
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
    DecentralizedEvaluationRunner,
    EvaluationRunner,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import cprint, fix_config, setup_config
from habitat_llm.utils.episode_cost import combined_time_breakdown, planner_cost_metrics
from habitat_llm.utils.llm_usage import fmt_usd
from habitat_llm.custom_approach import render_custom_subgoal_log_html
from habitat_llm.vlm_tamp.output_paths import aggregate_outputs_dir_from_results_dir
from habitat_llm.vlm_tamp.render_pddl_baseline_html import (
    EPISODE_METRICS_JSON,
    render_pddl_baseline_log_dir_to_html,
)
from habitat_llm.world_model import SpotRobot
from habitat_baselines.utils.info_dict import extract_scalars_from_info


# append the path of the
# parent directory
sys.path.append("..")


class _TeeStream:
    """Write to a real stream while copying into a file (for terminal + log.txt)."""

    def __init__(self, real_stream, mirror_file):
        self._real = real_stream
        self._mirror = mirror_file

    def write(self, data):
        self._real.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self):
        self._real.flush()
        self._mirror.flush()


def _run_root_log_txt_path(config) -> str:
    """Per-run `log.txt` next to other episode artifacts (uses `paths.results_dir`)."""
    return os.path.join(os.path.abspath(str(config.paths.results_dir)), "log.txt")


@contextlib.contextmanager
def _tee_stdout_stderr_to_run_log(config, header: str):
    """
    Append all stdout/stderr during the block to `<results_dir>/log.txt`, with markers.
    Still prints to the original terminal streams.
    """
    log_path = _run_root_log_txt_path(config)
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        logf.write(f"\n{'='*80}\n{header}\n{ts}\n{'='*80}\n")
        logf.flush()
        old_out, old_err = sys.stdout, sys.stderr
        tee_out = _TeeStream(old_out, logf)
        tee_err = _TeeStream(old_err, logf)
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                yield
        finally:
            end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            logf.write(f"\n--- end: {header} @ {end_ts} ---\n")
            logf.flush()


def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.paths.results_dir,
        dataset_file,
        "stats",
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    # Sort the dictionary by keys
    # Needed to ensure sanity in multi-process operation
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


def save_exception_message(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def save_success_message(config, env_interface, info):
    output_file = get_output_file(config, env_interface)
    failure_dict = {"success": True, "stats": json.dumps(info)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def _is_pddl_run(config) -> bool:
    """Best-effort check for PDDL baseline mode from config flags."""
    try:
        ev = getattr(config, "evaluation", None)
        if ev is not None and bool(getattr(ev, "pddl_baseline", False)):
            return True
    except Exception:
        pass

    try:
        agents = getattr(config.evaluation, "agents", None)
        if agents:
            for _agent_name, agent_cfg in agents.items():
                planner_cfg = getattr(agent_cfg, "planner", None)
                if planner_cfg is None:
                    continue
                plan_cfg = getattr(planner_cfg, "plan_config", None)
                if plan_cfg is not None and bool(getattr(plan_cfg, "pddl_baseline", False)):
                    return True
    except Exception:
        pass
    return False


def _get_vlm_tamp_pddl_log_dir(config, env_interface) -> Optional[str]:
    """Log dir for VLM-TAMP PDDL JSONL / index.html (matches VlmTampPddlPlanner._get_log_dir)."""
    try:
        results_dir = config.paths.results_dir
        log_dir_name = "vlm_tamp_pddl"
        try:
            pc = config.evaluation.agents.agent_0.planner.plan_config
            ld = getattr(pc, "log_dir", None)
            if ld:
                log_dir_name = str(ld)
        except Exception:
            pass
        return os.path.join(results_dir, log_dir_name)
    except Exception:
        return None


def _count_custom_vlm_requests(log_dir: Optional[str]) -> Optional[Dict[int, int]]:
    if not log_dir:
        return None
    jsonl_path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
    if not os.path.isfile(jsonl_path):
        return None

    custom_vlm_events = {
        "custom_task_to_subgoals",
        "custom_subgoal_exploration",
        "custom_subgoal_to_pddl_goals",
    }
    request_count = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                event = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if isinstance(event, dict) and event.get("event") in custom_vlm_events:
                request_count += 1

    return {0: request_count} if request_count else None


def _copy_pddl_interactive_html_to_outputs_dir(
    pddl_html_path: str, config
) -> None:
    """
    Copy all *_pddl.html files under the current run folder into the aggregate
    outputs folder derived from the active experiment results root.

    Example:
      paths.results_dir = results/vlm_tamp_pddl_v5/Task_5_Sg/Task_5_Sg_1
      aggregate dir     = results/outputs_vlm_tamp_pddl_v5
    """
    try:
        rd = os.path.normpath(str(getattr(config.paths, "results_dir", "") or ""))
        if not rd:
            return
        if not pddl_html_path or not os.path.isfile(pddl_html_path):
            return
        results_parent = Path(rd).resolve()
        if not results_parent.exists():
            return

        out_dir = aggregate_outputs_dir_from_results_dir(str(results_parent))
        out_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src in sorted(results_parent.rglob("*_pddl.html")):
            dest = out_dir / src.name
            shutil.copy2(src, dest)
            copied += 1

        cprint("✓ Copied PDDL HTML files to aggregate folder:", "green")
        cprint(f"  {out_dir} ({copied} file(s))", "green")
    except Exception as e:
        cprint(f"⚠ Failed to copy PDDL HTML to outputs folder: {e}", "yellow")


def _find_planning_tree_image(trace_file_path: str) -> Optional[str]:
    """
    Find planning_tree.png near a trace file for PDDL runs.
    Typical layout:
      <run_root>/<task>/traces/0/trace-...txt
      <run_root>/vlm_tamp_pddl/media/planning_tree.png
    """
    try:
        t = Path(trace_file_path).resolve()
        # .../<task>/traces/0/trace.txt -> task_root = .../<task>
        task_root = t.parent.parent.parent
        candidates = [task_root.parent, task_root]
        for base in candidates:
            if not base.exists():
                continue
            for img in base.rglob("planning_tree.png"):
                p = str(img)
                if "/vlm_tamp_pddl/" in p and "/media/" in p:
                    return p
    except Exception:
        return None
    return None


def _load_runtime_config(config) -> Optional[Dict[str, Any]]:
    """Load per-task runtime YAML when present."""
    if not hasattr(config, "runtime_config_path") or not config.runtime_config_path:
        return None

    runtime_config_path = Path(config.runtime_config_path)
    if not runtime_config_path.exists():
        cprint(f"⚠ Runtime config file not found: {runtime_config_path}", "yellow")
        return None

    cprint(f"\nLoading runtime configuration from: {runtime_config_path}", "cyan")
    with open(runtime_config_path, "r", encoding="utf-8") as f:
        runtime_config = yaml.safe_load(f)
    cprint("Runtime config loaded successfully", "green")
    return runtime_config


def _extract_runtime_subgoals(
    runtime_config: Optional[Dict[str, Any]],
    use_runtime_subgoals: bool = True,
) -> List[str]:
    """Return a sanitized list of runtime subgoals."""
    if not runtime_config:
        return []
    if not use_runtime_subgoals:
        return []

    raw_subgoals = runtime_config.get("subgoals")
    if raw_subgoals is None:
        return []

    if not isinstance(raw_subgoals, list):
        cprint("⚠ Ignoring runtime_config 'subgoals': expected a YAML list.", "yellow")
        return []

    subgoals = [
        str(subgoal).strip()
        for subgoal in raw_subgoals
        if isinstance(subgoal, str) and str(subgoal).strip()
    ]
    dropped_entries = len(raw_subgoals) - len(subgoals)
    if dropped_entries:
        cprint(
            f"⚠ Ignored {dropped_entries} empty/non-string runtime subgoal entry(s).",
            "yellow",
        )
    return subgoals


def _print_action_summary(info: Dict[str, Any], label: Optional[str] = None) -> None:
    """Print action count / step summaries for a completed instruction."""
    title_suffix = f" for {label}" if label else ""
    if "action_counts" in info and info["action_counts"]:
        cprint("\n---------------------------------", "cyan")
        cprint(f"Action Counts (times selected){title_suffix}:", "cyan")
        for action_name, count in sorted(info["action_counts"].items()):
            cprint(f"  {action_name}: {count}", "cyan")
        cprint("---------------------------------", "cyan")
    if "action_sim_steps" in info and info["action_sim_steps"]:
        cprint("\n---------------------------------", "cyan")
        cprint(f"Action Simulation Steps (total steps per action){title_suffix}:", "cyan")
        for action_name, steps in sorted(info["action_sim_steps"].items()):
            cprint(f"  {action_name}: {steps}", "cyan")
        cprint("---------------------------------", "cyan")
    cost_metrics = planner_cost_metrics(info)
    breakdown = combined_time_breakdown(
        llm_planning_time_s=info.get(
            "llm_planning_time_s", cost_metrics.get("llm_planning_time_s")
        ),
        action_sim_steps=info.get("action_sim_steps"),
        explore_approx_sim_time_s=info.get(
            "explore_approx_sim_time_s",
            cost_metrics.get("explore_approx_sim_time_s"),
        ),
        limit_s=info.get("combined_time_limit_s", 0.0),
        physical_explore=bool(info.get("physical_explore", cost_metrics.get("physical_explore", False))),
        exclude_explore_from_time_budget=bool(info.get("exclude_explore_from_time_budget", False)),
    )
    limit_s = info.get("combined_time_limit_s", breakdown["limit_s"])
    limit_str = f"{float(limit_s):.2f}s" if limit_s else "disabled"
    cprint("\n---------------------------------", "yellow")
    cprint(f"Combined time budget{title_suffix}:", "yellow")
    cprint(
        f"  used {breakdown['used_s']:.2f}s / {limit_str} "
        f"(llm {breakdown['llm_s']:.2f} + actions {breakdown['action_sim_s']:.2f} "
        f"+ explore~ {breakdown['explore_approx_s']:.2f})"
        + (f"; Explore excluded: {breakdown['explore_excluded_s']:.2f}s"
           if breakdown['exclude_explore_from_time_budget'] else ""),
        "yellow",
    )
    if info.get("combined_time_limit_hit"):
        cprint("  LIMIT HIT — episode stopped", "red")
    model = info.get("llm_model", cost_metrics.get("llm_model"))
    usd = info.get("llm_usd", cost_metrics.get("llm_usd"))
    prompt_tokens = info.get("prompt_tokens", cost_metrics.get("prompt_tokens"))
    completion_tokens = info.get(
        "completion_tokens", cost_metrics.get("completion_tokens")
    )
    if model or usd is not None or prompt_tokens or completion_tokens:
        cprint(
            f"  model {model or 'N/A'}  "
            f"tokens {int(prompt_tokens or 0)} in / {int(completion_tokens or 0)} out  "
            f"API {fmt_usd(usd)}",
            "yellow",
        )
    cprint("---------------------------------", "yellow")


def _trace_episode_filename(trace_file_path: str) -> Optional[str]:
    """Extract episode_filename from trace path `trace-<episode_filename>-<uid>.txt`."""
    stem = Path(trace_file_path).stem
    if not stem.startswith("trace-"):
        return None
    stem_without_prefix = stem[len("trace-") :]
    if "-" not in stem_without_prefix:
        return stem_without_prefix
    return stem_without_prefix.rsplit("-", 1)[0]


def _reset_instruction_histories(env_interface: EnvironmentInterface) -> None:
    """Clear per-instruction action/state histories without resetting the episode."""
    env_interface.agent_state_history = defaultdict(list)
    env_interface.agent_action_history = defaultdict(list)


_PLANNER_DEMO_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_dataset_file(data_path: str) -> Optional[str]:
    path = Path(data_path)
    if path.is_file():
        return str(path)
    for name in ("dataset.json.gz", "dataset.json"):
        candidate = path / name
        if candidate.is_file():
            return str(candidate)
    return None


def _prediviz_scene_cache_dir(config) -> str:
    custom = _PLANNER_DEMO_REPO_ROOT / "data" / "datasets" / "custom" / "scene_info"
    if custom.is_dir() and any(custom.iterdir()):
        return str(custom)
    cache = Path(config.paths.results_dir) / "prediviz_scene_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


def _prediviz_step_index(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _snapshot_object_placements(env_interface) -> Dict[str, Dict[str, Optional[str]]]:
    gt = getattr(getattr(env_interface, "perception", None), "gt_graph", None)
    if gt is None or not hasattr(gt, "get_all_objects"):
        return {}
    placements: Dict[str, Dict[str, Optional[str]]] = {}
    for obj in gt.get_all_objects():
        furn = gt.find_furniture_for_object(obj)
        room_name = None
        try:
            rooms = gt.get_room_for_entity(obj)
            if rooms:
                first = rooms[0] if isinstance(rooms, (list, tuple)) else rooms
                room_name = getattr(first, "name", None)
        except Exception:
            room_name = None
        placements[obj.name] = {
            "object_handle": getattr(obj, "sim_handle", None),
            "furniture": None if furn is None else furn.name,
            "furniture_handle": None
            if furn is None
            else getattr(furn, "sim_handle", None),
            "room": room_name,
        }
    return placements


def _placements_to_prediviz(
    snapshot: Dict[str, Any], metadata: Dict[str, Any]
) -> Dict[str, Dict[str, str]]:
    handle_to_obj = {v: k for k, v in (metadata.get("object_to_handle") or {}).items()}
    handle_to_recep = {v: k for k, v in (metadata.get("recep_to_handle") or {}).items()}
    recep_to_room = metadata.get("recep_to_room") or {}
    object_to_recep: Dict[str, str] = {}
    object_to_room: Dict[str, str] = {}
    for habitat_name, info in snapshot.items():
        if not isinstance(info, dict):
            continue
        obj_id = handle_to_obj.get(info.get("object_handle"))
        if obj_id is None and habitat_name in (metadata.get("object_to_handle") or {}):
            obj_id = habitat_name
        if obj_id is None:
            obj_id = habitat_name
        furniture_handle = info.get("furniture_handle")
        recep = handle_to_recep.get(furniture_handle) if furniture_handle else None
        if recep is None:
            if info.get("furniture") in (None, "floor"):
                recep = "floor"
            else:
                continue
        object_to_recep[obj_id] = recep
        object_to_room[obj_id] = recep_to_room.get(recep) or info.get("room") or ""
    return {"object_to_recep": object_to_recep, "object_to_room": object_to_room}


def _write_prediviz_initial_final(viz_episode_dir: Path) -> Optional[Path]:
    steps = sorted(viz_episode_dir.glob("step_*.png"), key=_prediviz_step_index)
    if not steps:
        return None
    dest = viz_episode_dir / "initial_and_final.png"
    shutil.copy2(steps[0], dest)
    shutil.copy2(steps[0], viz_episode_dir / "initial.png")
    leftover_final = viz_episode_dir / "final.png"
    if leftover_final.exists():
        leftover_final.unlink()
    return dest


def _generate_prediviz_after_run(
    config,
    episode_ids: Sequence[Any],
    initial_snapshot: Optional[Dict[str, Any]] = None,
    final_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    if not getattr(config.evaluation, "generate_prediviz", True):
        return
    unique_ids: List[str] = []
    seen = set()
    for episode_id in episode_ids:
        key = str(episode_id)
        if key in seen:
            continue
        seen.add(key)
        unique_ids.append(key)
    if not unique_ids:
        cprint("⚠ PrediViz skipped: no episode ids from this run", "yellow")
        return

    dataset_path = _resolve_dataset_file(config.habitat.dataset.data_path)
    if dataset_path is None:
        cprint(
            f"⚠ PrediViz skipped: dataset file not found at {config.habitat.dataset.data_path}",
            "yellow",
        )
        return

    results_dir = Path(config.paths.results_dir)
    meta_dir = results_dir / "prediviz_metadata"
    viz_dir = results_dir / "prediviz"
    meta_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    missing_meta = [
        eid for eid in unique_ids if not (meta_dir / f"{eid}.json").is_file()
    ]
    scene_cache = _prediviz_scene_cache_dir(config)
    try:
        if missing_meta:
            cprint(
                f"\n🖼 Extracting PrediViz metadata for {len(missing_meta)} episode(s)...",
                "cyan",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dataset_generation.benchmark_generation.metadata_extractor",
                    "--dataset-path",
                    dataset_path,
                    "--scene-metadata-cache",
                    scene_cache,
                    "--save-dir",
                    str(meta_dir),
                ],
                check=True,
                cwd=str(_PLANNER_DEMO_REPO_ROOT),
            )
        placements_path = results_dir / "prediviz_object_locations.json"
        placements_by_eid: Dict[str, Any] = {}
        for eid in unique_ids:
            meta_path = meta_dir / f"{eid}.json"
            metadata = {}
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            placements_by_eid[eid] = {
                "habitat_initial": initial_snapshot or {},
                "habitat_final": final_snapshot or {},
                "initial": _placements_to_prediviz(initial_snapshot or {}, metadata),
                "final": _placements_to_prediviz(final_snapshot or {}, metadata),
            }
        with open(placements_path, "w", encoding="utf-8") as handle:
            json.dump(placements_by_eid, handle, indent=2)
        for eid in unique_ids:
            cprint(f"🖼 Generating PrediViz for episode_id={eid}...", "cyan")
            viz_cmd = [
                sys.executable,
                str(_PLANNER_DEMO_REPO_ROOT / "scripts" / "prediviz" / "viz.py"),
                "--dataset",
                dataset_path,
                "--metadata-dir",
                str(meta_dir),
                "--save-path",
                str(viz_dir),
                "--episode-id",
                str(int(eid)),
                "--keep-initial-layout",
                "--placements-json",
                str(placements_path),
            ]
            subprocess.run(
                viz_cmd,
                check=True,
                cwd=str(_PLANNER_DEMO_REPO_ROOT),
            )
            combined = _write_prediviz_initial_final(viz_dir / f"viz_{int(eid)}")
            if combined is not None:
                cprint(f"✓ PrediViz: {combined}", "green")
                cprint(f"✓ Object locations: {placements_path}", "green")
            else:
                cprint(
                    f"⚠ PrediViz finished but no step_*.png under {viz_dir / f'viz_{int(eid)}'}",
                    "yellow",
                )
    except Exception as prediviz_error:
        cprint(f"⚠ PrediViz generation failed: {prediviz_error}", "yellow")


# Write the config file into the results folders
def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))

    # Copy over the RLM config
    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
        # Centralized
        if "plan_config" in config.evaluation.planner is not None:
            planner_configs = [config.evaluation.planner.plan_config]
            suffixes = [""]
    else:
        for agent_name in config.evaluation.agents:
            suffixes.append(f"_{agent_name}")
            planner_configs.append(
                config.evaluation.agents[agent_name].planner.plan_config
            )

    for plan_config, suffix_rlm in zip(planner_configs, suffixes):
        if "llm" in plan_config and "serverdir" in plan_config.llm:
            yaml_rlm_path = plan_config.llm.serverdir
            if len(yaml_rlm_path) > 0:
                yaml_rlm_file = f"{yaml_rlm_path}/config.yaml"
                if os.path.isfile(yaml_rlm_file):
                    shutil.copy(
                        yaml_rlm_file, f"{output_file}/config_rlm{suffix_rlm}.yaml"
                    )


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def run_eval(config):
    fix_config(config)
    # Setup a seed
    # seed = 48212516
    seed = 47668090
    t0 = time.time()
    # Setup config
    config = setup_config(config, seed)

    # Normalize dataset path: if folder, pick .json.gz / .json and optional .yaml
    if hasattr(config, "habitat") and hasattr(config.habitat, "dataset"):
        data_path = getattr(config.habitat.dataset, "data_path", None)
        if data_path is not None:
            data_path_str = str(data_path)
            if os.path.isdir(data_path_str):
                folder = Path(data_path_str)
                json_gz_files = sorted(folder.glob("*.json.gz"))
                json_files = sorted(folder.glob("*.json"))
                chosen_dataset = None
                if json_gz_files:
                    chosen_dataset = json_gz_files[0]
                elif json_files:
                    chosen_dataset = json_files[0]
                if chosen_dataset is not None:
                    with open_dict(config):
                        config.habitat.dataset.data_path = str(chosen_dataset)
                    data_path_str = str(chosen_dataset)
                if (not hasattr(config, "runtime_config_path") or not config.runtime_config_path):
                    yaml_files = sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")))
                    if yaml_files:
                        with open_dict(config):
                            config.runtime_config_path = str(yaml_files[0])
            if data_path_str.endswith(".json") and not data_path_str.endswith(".json.gz"):
                with open_dict(config):
                    config.habitat.dataset.data_path = data_path_str + ".gz"

    dataset = CollaborationDatasetV0(config.habitat.dataset)

    write_config(config)
    if config.get("resume", False):
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # stats_dir = os.path.join(config.paths.results_dir, dataset_file, "stats")
        plan_log_dir = os.path.join(
            config.paths.results_dir, dataset_file, "planner-log"
        )

        # Find incomplete episodes
        incomplete_episodes = []
        for episode in dataset.episodes:
            episode_id = episode.episode_id
            # stats_file = os.path.join(stats_dir, f"{episode_id}.json")
            planlog_file = os.path.join(
                plan_log_dir, f"planner-log-episode_{episode_id}_0.json"
            )
            if not os.path.exists(planlog_file):
                incomplete_episodes.append(episode)
        print(
            f"Resuming with {len(incomplete_episodes)} incomplete episodes: {[e.episode_id for e in incomplete_episodes]}"
        )
        # Update dataset with only incomplete episodes
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=incomplete_episodes
        )

    # filter episodes by mod for running on multiple nodes
    if config.get("episode_mod_filter", None) is not None:
        rem, mod = config.episode_mod_filter
        episode_subset = [x for x in dataset.episodes if int(x.episode_id) % mod == rem]
        print(f"Mod filter: {rem}, {mod}")
        print(f"Episodes: {[e.episode_id for e in episode_subset]}")
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )

    num_episodes = len(dataset.episodes)
    if config.num_proc == 1:
        if config.get("episode_indices", None) is not None:
            if config.get("resume", False):
                raise ValueError("episode_indices and resume cannot be used together")
            episode_subset = [dataset.episodes[x] for x in config.episode_indices]
            dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )
        run_planner(config, dataset)
    else:
        # Process episodes in parallel
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []
        config.num_proc = min(config.num_proc, num_episodes)
        ochunk_size = num_episodes // config.num_proc
        # Prepare chunked datasets
        chunked_datasets = []
        # TODO: we may want to chunk by scene
        start = 0
        for i in range(config.num_proc):
            chunk_size = ochunk_size
            if i < (num_episodes % config.num_proc):
                chunk_size += 1
            end = min(start + chunk_size, num_episodes)
            indices = slice(start, end)
            chunked_datasets.append(indices)
            start += chunk_size

        for episode_index_chunk in chunked_datasets:
            episode_subset = dataset.episodes[episode_index_chunk]
            new_dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )

            parent_conn, child_conn = mp_ctx.Pipe()
            proc_args = (config, new_dataset, child_conn)
            p = mp_ctx.Process(target=run_planner, args=proc_args)
            p.start()
            proc_infos.append((parent_conn, p))
            print("START PROCESS")

        # Get back info
        all_stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }
        for conn, proc in proc_infos:
            stats_episodes = conn.recv()
            for run_id, stats_run in stats_episodes.items():
                all_stats_episodes[str(run_id)].update(stats_run)
            proc.join()

        all_metrics = aggregate_measures(
            {run_id: aggregate_measures(v) for run_id, v in all_stats_episodes.items()}
        )
        cprint("\n---------------------------------", "blue")
        cprint("Metrics Across All Runs:", "blue")
        for k, v in all_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("\n---------------------------------", "blue")
        # Write aggregated results across experiment
        write_to_csv(config.paths.end_result_file_path, all_metrics)

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")


def run_planner(config, dataset: CollaborationDatasetV0 = None, conn=None):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # Setup interface with the simulator if the planner depends on it
    if config.env == "habitat":
        # RGB sensors are stripped unless video, evaluation.use_rgb, or action images.
        keep_rgb = bool(getattr(config.evaluation, "use_rgb", False))
        try:
            agents_cfg = getattr(config.evaluation, "agents", None)
            if agents_cfg is not None:
                for agent_cfg in agents_cfg.values():
                    plan_cfg = getattr(
                        getattr(agent_cfg, "planner", None), "plan_config", None
                    )
                    if plan_cfg is not None and bool(
                        getattr(plan_cfg, "send_action_image", False)
                    ):
                        keep_rgb = True
                        break
        except Exception:
            pass
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)

        # TODO: Can we move this inside the EnvironmentInterface?
        # We register the dynamic habitat sensors
        register_sensors(config)
        # We register custom actions
        register_actions(config)
        # We register custom measures
        register_measures(config)

        # Initialize the environment interface for the agent
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)
        initial_object_snapshot: Dict[str, Any] = {}

        try:
            env_interface.initialize_perception_and_world_graph()
            initial_object_snapshot = _snapshot_object_placements(env_interface)
        except Exception:
            print("Error initializing the environment")
            if config.evaluation.log_data:
                save_exception_message(config, env_interface)
    else:
        env_interface = None
        initial_object_snapshot = {}

    # Instantiate the agent planner
    eval_runner: EvaluationRunner = None
    if config.evaluation.type == "centralized":
        eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)
    elif config.evaluation.type == "decentralized":
        eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    else:
        cprint(
            "Invalid planner type. Please select between 'centralized' or 'decentralized'. Exiting",
            "red",
        )
        return

    # Print the planner
    cprint(f"Successfully constructed the '{config.evaluation.type}' planner!", "green")
    print(eval_runner)

    # Declare observability mode
    cprint(
        f"Partial observability is set to: '{config.world_model.partial_obs}'", "green"
    )

    # Print the agent list
    print("\nAgent List:")
    print(eval_runner.agent_list)

    # Print the agent description
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # Try to get model info from different config structures
    try:
        if hasattr(config, "planner") and hasattr(config.planner, "llm"):
            model_target = config.planner.llm.llm._target_
        elif hasattr(config, "planner") and hasattr(config.planner, "vlm"):
            model_target = config.planner.vlm.vlm._target_
        else:
            # Decentralized config structure
            agent_config = config.evaluation.agents.agent_0.planner.plan_config
            if hasattr(agent_config, "vlm"):
                model_target = agent_config.vlm.vlm._target_
            elif hasattr(agent_config, "llm"):
                model_target = agent_config.llm.llm._target_
            else:
                model_target = "Unknown"
        cprint(f"Model: {model_target}", "blue")
    except Exception:
        cprint("Model: Unable to determine planner model", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    # Print whether GT object locations or RGB-D observations are being used
    use_gt_locs = getattr(config.world_model, 'use_gt_object_locations', False)
    if use_gt_locs:
        cprint("Perception: Using GT (ground truth) object locations from simulator", "green")
    else:
        cprint("Perception: Using RGB-D observations for object detection and localization", "green")
    cprint("---------------------------------------\n", "blue")

    os.makedirs(config.paths.results_dir, exist_ok=True)

    # Set up image save directory for VLM planners
    vlm_images_dir = os.path.join(config.paths.results_dir, "vlm_images")
    # Handle both centralized (single planner) and decentralized (dict of planners)
    planners_to_check = []
    if hasattr(eval_runner, "planner"):
        if isinstance(eval_runner.planner, dict):
            planners_to_check = list(eval_runner.planner.values())
        else:
            planners_to_check = [eval_runner.planner]
    for planner in planners_to_check:
        if hasattr(planner, "set_image_save_dir"):
            planner_cfg = getattr(planner, "planner_config", None)
            send_action_image = bool(
                getattr(planner_cfg, "send_action_image", False)
            )
            if send_action_image:
                save_dir = os.path.join(
                    eval_runner.output_dir, "traces", "0", "images"
                )
            else:
                save_dir = vlm_images_dir
            planner.set_image_save_dir(save_dir)

    # Run the planner
    # Initialize stats_episodes for both CLI and non-CLI modes
    stats_episodes: Dict[str, Dict] = {
        str(i): {} for i in range(config.num_runs_per_episode)
    }

    trace_run_infos: List[Dict[str, Any]] = []
    ran_episode_ids: List[str] = []

    if config.mode == "cli":
        # Get instruction from config if provided, otherwise use episode instruction
        if config.instruction:
            instruction = config.instruction
        else:
            # Get instruction from current episode
            instruction = env_interface.env.env.env._env.current_episode.instruction

        # Ensure videos are saved in CLI mode, even if task fails
        # if not config.evaluation.save_video:
        #     config.evaluation.save_video = True
        #     cprint("Enabling video saving for CLI mode", "yellow")

        # Load runtime configuration if provided (for robot location and runtime objects)
        runtime_config = _load_runtime_config(config)

        # Override instruction with task from runtime config if present
        if runtime_config and "task" in runtime_config and runtime_config["task"]:
            instruction = runtime_config["task"]
            cprint("Instruction overridden from runtime config 'task'", "green")

        cprint(f'\nExecuting instruction: "{instruction}"', "blue")

        # Apply runtime modifications in partial observability mode
        if config.world_model.partial_obs and runtime_config:
            agent_graph = env_interface.world_graph.get(config.robot_agent_uid)
            if agent_graph is not None:
                # 1. Move robot to specified room if provided
                if 'runtime_robot_location' in runtime_config and 'room' in runtime_config['runtime_robot_location']:
                    target_room = runtime_config['runtime_robot_location']['room']
                    cprint(f"\n🤖 Moving robot to {target_room}...", "cyan")
                    success = agent_graph.move_robot_to_room(target_room, verbose=True, sim=env_interface.sim)
                    if success:
                        robot_nodes = agent_graph.get_all_nodes_of_type(SpotRobot)
                        if robot_nodes:
                            robot_pos = robot_nodes[0].properties.get('translation', [0, 0, 0])
                            cprint(f"✓ Robot moved to {target_room} at position {robot_pos}", "green")
                    else:
                        cprint(f"⚠ Failed to move robot to {target_room}", "yellow")

                # 2. Add runtime objects if provided
                if 'runtime_objects' in runtime_config:
                    runtime_objs = runtime_config['runtime_objects']
                    if runtime_objs.get('enabled', False) and 'objects' in runtime_objs:
                        cprint(f"\n🍾 Adding {len(runtime_objs['objects'])} runtime object(s) to scene graph...", "cyan")
                        for obj_config in runtime_objs['objects']:
                            obj_handle = obj_config.get('handle')
                            obj_class = obj_config.get('class')
                            furniture_name = obj_config.get('furniture_name')
                            position = obj_config.get('position')
                            rotation = obj_config.get('rotation')

                            if not obj_class:
                                cprint(f"⚠ Skipping object: 'class' not specified", "yellow")
                                continue

                            try:
                                if furniture_name:
                                    # Place on furniture
                                    added_obj = agent_graph.add_object_to_graph(
                                        object_class=obj_class,
                                        furniture_name=furniture_name,
                                        sim_handle=obj_handle,
                                        connect_to_entities=True,
                                        verbose=True
                                    )
                                    cprint(f"✓ Added {added_obj.name} (class: {obj_class}) on {furniture_name} at {added_obj.properties.get('translation')}", "green")
                                elif position:
                                    # Place at absolute position
                                    added_obj = agent_graph.add_object_to_graph(
                                        object_class=obj_class,
                                        position=position,
                                        rotation=rotation if rotation else [0, 0, 0, 1],
                                        sim_handle=obj_handle,
                                        connect_to_entities=True,
                                        verbose=True
                                    )
                                    cprint(f"✓ Added {added_obj.name} (class: {obj_class}) at position {position}", "green")
                                else:
                                    cprint(f"⚠ Skipping {obj_class}: neither 'furniture_name' nor 'position' specified", "yellow")
                            except Exception as e:
                                cprint(f"✗ Failed to add {obj_class}: {e}", "red")
            else:
                cprint("⚠ Agent world graph not available, skipping runtime modifications", "yellow")

        # Print the robot's scene graph before episode starts
        cprint("\nRobot Scene Graph:", "magenta")
        cprint("=" * 80, "magenta")
        agent_graph = env_interface.world_graph.get(config.robot_agent_uid)
        if agent_graph is not None:
            agent_graph.display_hierarchy()
        else:
            cprint("Could not access robot world graph", "yellow")
        cprint("=" * 80 + "\n", "magenta")

        subgoals = _extract_runtime_subgoals(
            runtime_config,
            use_runtime_subgoals=bool(getattr(config, "use_runtime_subgoals", True)),
        )
        instruction_runs = []
        if subgoals:
            shared_output_name = "subgoals_sequence"
            cprint(
                f"Executing {len(subgoals)} runtime subgoal(s) on the same episode state.",
                "blue",
            )
            for idx, subgoal in enumerate(subgoals, start=1):
                cprint(f"  {idx}. {subgoal}", "blue")
                instruction_runs.append(
                    {
                        "instruction": subgoal,
                        "output_name": shared_output_name,
                        "label": f"subgoal {idx}/{len(subgoals)}",
                    }
                )
        else:
            instruction_runs.append(
                {
                    "instruction": instruction,
                    "output_name": "",
                    "label": "instruction",
                }
            )

        last_info = None
        _cli_ep = env_interface.env.env.env._env.current_episode.episode_id
        ran_episode_ids.append(str(_cli_ep))
        _cli_header = f"planner_demo cli mode | episode_id={_cli_ep}"
        with _tee_stdout_stderr_to_run_log(config, _cli_header):
            try:
                for idx, run_spec in enumerate(instruction_runs):
                    run_instruction_text = run_spec["instruction"]
                    run_output_name = run_spec["output_name"]
                    run_label = run_spec["label"]
                    cprint("\n" + "=" * 80, "blue")
                    cprint(
                        f"Starting {run_label}: {run_instruction_text}",
                        "blue",
                    )
                    cprint("=" * 80, "blue")

                    # Important call to eval method that runs the planner.
                    info = eval_runner.run_instruction(
                        run_instruction_text,
                        output_name=run_output_name,
                        preserve_planner_state=idx > 0,
                    )

                    _print_action_summary(info, label=run_label)

                    trace_run_infos.append(
                        {
                            "episode_filename": eval_runner.episode_filename,
                            "info": info.copy(),
                        }
                    )
                    last_info = info.copy()
            except Exception as e:
                print("An error occurred inside of this method:", e)
                # Ensure video is saved even if exception occurs
                if config.evaluation.save_video and hasattr(eval_runner, 'dvu') and len(eval_runner.dvu.frames) > 0:
                    try:
                        eval_runner.dvu._make_video(play=False, postfix=eval_runner.episode_filename)
                    except Exception as video_error:
                        print(f"Warning: Failed to save video after exception: {video_error}")

    else:
        num_episodes = len(env_interface.env.episodes)
        last_info = None  # Track last episode's info for HTML generation
        for run_id in range(config.num_runs_per_episode):
            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id
                ran_episode_ids.append(str(episode_id))

                # Get instruction
                instruction = env_interface.env.env.env._env.current_episode.instruction
                _batch_header = (
                    f"planner_demo batch | episode_id={episode_id} run_id={run_id} "
                    f"output_name=episode_{episode_id}_{run_id}"
                )
                with _tee_stdout_stderr_to_run_log(config, _batch_header):
                    print("\n\nEpisode", episode_id)
                    try:
                        _reset_instruction_histories(env_interface)
                        info = eval_runner.run_instruction(
                            output_name=f"episode_{episode_id}_{run_id}"
                        )

                        # Store last info for HTML generation
                        last_info = info
                        trace_run_infos.append(
                            {
                                "episode_filename": eval_runner.episode_filename,
                                "info": info.copy(),
                            }
                        )

                        _print_action_summary(
                            info,
                            label=f"run {run_id} episode {episode_id}",
                        )

                        info_episode = {
                            "run_id": run_id,
                            "episode_id": episode_id,
                            "instruction": instruction,
                        }
                        stats_keys = {
                            "task_percent_complete",
                            "task_state_success",
                            "sim_step_count",
                            "replanning_count",
                            "runtime",
                            "llm_planning_time_s",
                            "explore_approx_sim_steps",
                            "explore_approx_sim_time_s",
                            "combined_time_used_s",
                            "combined_time_limit_s",
                            "combined_time_limit_hit",
                        }

                        # add replanning counts to stats_keys as scalars if replanning_count is a dict
                        if "replanning_count" in info and isinstance(
                            info["replanning_count"], dict
                        ):
                            for agent_id, replan_count in info["replanning_count"].items():
                                stats_keys.add(f"replanning_count_{agent_id}")
                                info[f"replanning_count_{agent_id}"] = replan_count

                        stats_episode = extract_scalars_from_info(
                            info, ignore_keys=info.keys() - stats_keys
                        )
                        stats_episodes[str(run_id)][episode_id] = stats_episode

                        cprint("\n---------------------------------", "blue")
                        cprint(f"Metrics For Run {run_id} Episode {episode_id}:", "blue")
                        for k, v in stats_episodes[str(run_id)][episode_id].items():
                            cprint(f"{k}: {v:.3f}", "blue")
                        cprint("\n---------------------------------", "blue")
                        # Log results onto a CSV
                        epi_metrics = stats_episodes[str(run_id)][episode_id] | info_episode
                        if config.evaluation.log_data:
                            save_success_message(config, env_interface, stats_episode)
                        write_to_csv(config.paths.epi_result_file_path, epi_metrics)
                    except Exception as e:
                        # print exception and trace
                        traceback.print_exc()
                        print("An error occurred while running the episode:", e)
                        print(f"Skipping evaluating episode: {episode_id}")
                        if config.evaluation.log_data:
                            save_exception_message(config, env_interface)

                try:
                    # Reset env_interface (moves onto the next episode in the dataset)
                    # env_interface.reset_environment()
                    # ! We never call this because we are only running one episode at a time
                    pass
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while resetting the env_interface:", e)
                    print("Skipping evaluating episode.")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                # Reset evaluation runner
                eval_runner.reset()

            # aggregate metrics across the current run.
            run_metrics = aggregate_measures(stats_episodes[str(run_id)])
            cprint("\n---------------------------------", "blue")
            cprint(f"Metrics For Run {run_id}:", "blue")
            for k, v in run_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)

    # Generate HTML visualization of trace logs
    try:
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # Remove .json.gz extension to match output_dir structure
        if dataset_file.endswith('.json.gz'):
            dataset_file = dataset_file[:-8]
        elif dataset_file.endswith('.json'):
            dataset_file = dataset_file[:-5]

        # Construct the traces directory path
        traces_dir = os.path.join(
            config.paths.results_dir,
            dataset_file,
            "traces",
            "0"
        )

        trace_info_by_episode_filename = {
            entry["episode_filename"]: entry["info"] for entry in trace_run_infos
        }

        trace_file_paths: List[str] = []
        if os.path.exists(traces_dir):
            trace_file_paths = sorted(
                str(Path(traces_dir) / f)
                for f in os.listdir(traces_dir)
                if f.endswith(".txt")
            )

        if trace_file_paths:
            cprint(f"\n📊 Generating HTML trace visualization...", "cyan")
            is_pddl_run = _is_pddl_run(config)
            for trace_file_path in trace_file_paths:
                planning_tree_image = (
                    _find_planning_tree_image(trace_file_path) if is_pddl_run else None
                )
                trace_data = view_trace_logs.parse_trace_file(
                    trace_file_path,
                    is_pddl_run=is_pddl_run,
                )
                html_output = str(Path(trace_file_path).with_suffix(".html"))

                trace_episode_filename = _trace_episode_filename(trace_file_path)
                trace_info = (
                    trace_info_by_episode_filename.get(trace_episode_filename)
                    or last_info
                )

                action_counts = None
                action_sim_steps = None
                runtime = None
                llm_requests = None
                llm_planning_time_s = None
                explore_approx_sim_steps = None
                explore_approx_sim_time_s = None
                explore_approx_meters = None
                combined_time_used_s = None
                combined_time_limit_s = None
                combined_time_limit_hit = None
                combined_time_breakdown_info = None
                sim_step_count = None
                task_percent_complete = None
                task_state_success = None
                llm_model = None
                prompt_tokens = None
                completion_tokens = None
                cached_tokens = None
                llm_usd = None
                llm_usd_source = None
                llm_reasoning_effort = None
                if trace_info is not None:
                    action_counts = trace_info.get("action_counts")
                    action_sim_steps = trace_info.get("action_sim_steps")
                    runtime = trace_info.get("runtime")
                    llm_requests = trace_info.get("replanning_count")
                    cost_metrics = trace_info.get("cost_metrics") or {}
                    llm_planning_time_s = trace_info.get("llm_planning_time_s")
                    if llm_planning_time_s is None:
                        llm_planning_time_s = cost_metrics.get("llm_planning_time_s")
                    explore_approx_sim_steps = trace_info.get(
                        "explore_approx_sim_steps"
                    )
                    if explore_approx_sim_steps is None:
                        explore_approx_sim_steps = cost_metrics.get(
                            "explore_approx_sim_steps"
                        )
                    explore_approx_sim_time_s = trace_info.get(
                        "explore_approx_sim_time_s"
                    )
                    if explore_approx_sim_time_s is None:
                        explore_approx_sim_time_s = cost_metrics.get(
                            "explore_approx_sim_time_s"
                        )
                    explore_approx_meters = trace_info.get(
                        "explore_approx_meters",
                        cost_metrics.get("explore_approx_meters"),
                    )
                    combined_time_used_s = trace_info.get("combined_time_used_s")
                    combined_time_limit_s = trace_info.get("combined_time_limit_s")
                    combined_time_limit_hit = trace_info.get(
                        "combined_time_limit_hit"
                    )
                    combined_time_breakdown_info = trace_info.get(
                        "combined_time_breakdown"
                    )
                    sim_step_count = trace_info.get("sim_step_count")
                    if sim_step_count is None:
                        sim_step_count = trace_info.get("num_steps")
                    task_percent_complete = trace_info.get("task_percent_complete")
                    task_state_success = trace_info.get("task_state_success")
                    llm_model = trace_info.get("llm_model", cost_metrics.get("llm_model"))
                    prompt_tokens = trace_info.get(
                        "prompt_tokens", cost_metrics.get("prompt_tokens")
                    )
                    completion_tokens = trace_info.get(
                        "completion_tokens", cost_metrics.get("completion_tokens")
                    )
                    cached_tokens = trace_info.get(
                        "cached_tokens", cost_metrics.get("cached_tokens")
                    )
                    llm_usd = trace_info.get("llm_usd", cost_metrics.get("llm_usd"))
                    llm_usd_source = trace_info.get(
                        "llm_usd_source", cost_metrics.get("llm_usd_source")
                    )
                    llm_reasoning_effort = trace_info.get(
                        "llm_reasoning_effort",
                        cost_metrics.get("llm_reasoning_effort"),
                    )
                pddl_log_dir = (
                    _get_vlm_tamp_pddl_log_dir(config, env_interface)
                    if is_pddl_run and env_interface is not None
                    else None
                )
                custom_llm_requests = _count_custom_vlm_requests(pddl_log_dir)
                if custom_llm_requests is not None:
                    llm_requests = custom_llm_requests

                html_kwargs = {
                    "action_counts": action_counts,
                    "action_sim_steps": action_sim_steps,
                    "runtime": runtime,
                    "llm_requests": llm_requests,
                    "is_pddl_run": is_pddl_run,
                    "planning_tree_image": planning_tree_image,
                    "llm_planning_time_s": llm_planning_time_s,
                    "explore_approx_sim_steps": explore_approx_sim_steps,
                    "explore_approx_sim_time_s": explore_approx_sim_time_s,
                    "explore_approx_meters": explore_approx_meters,
                    "sim_step_count": sim_step_count,
                    "task_percent_complete": task_percent_complete,
                    "task_state_success": task_state_success,
                    "combined_time_used_s": combined_time_used_s,
                    "combined_time_limit_s": combined_time_limit_s,
                    "combined_time_limit_hit": combined_time_limit_hit,
                    "combined_time_breakdown": combined_time_breakdown_info,
                    "llm_model": llm_model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "llm_usd": llm_usd,
                    "llm_usd_source": llm_usd_source,
                    "llm_reasoning_effort": llm_reasoning_effort,
                }
                html_params = inspect.signature(
                    view_trace_logs.generate_html
                ).parameters
                view_trace_logs.generate_html(
                    trace_data,
                    html_output,
                    **{
                        key: value
                        for key, value in html_kwargs.items()
                        if key in html_params
                    },
                )
                cprint("✓ HTML trace file generated at:", "green")
                cprint(f"  {html_output}", "green")

                # Match PDDL interactive index.html runtime to trace HTML (evaluation info["runtime"]).
                if is_pddl_run and runtime is not None and env_interface is not None:
                    try:
                        if pddl_log_dir and os.path.isdir(pddl_log_dir):
                            metrics_path = os.path.join(
                                pddl_log_dir, EPISODE_METRICS_JSON
                            )
                            metrics_payload = {
                                "episode_runtime_sec": float(runtime),
                                "task_percent_complete": task_percent_complete,
                                "task_state_success": task_state_success,
                                "llm_model": llm_model,
                                "llm_reasoning_effort": llm_reasoning_effort,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "cached_tokens": cached_tokens,
                                "llm_usd": llm_usd,
                                "llm_usd_source": llm_usd_source,
                                "llm_planning_time_s": llm_planning_time_s,
                                "llm_requests": llm_requests,
                                "action_counts": action_counts,
                                "action_sim_steps": action_sim_steps,
                                "sim_step_count": sim_step_count,
                                "explore_approx_sim_steps": explore_approx_sim_steps,
                                "explore_approx_sim_time_s": explore_approx_sim_time_s,
                                "explore_approx_meters": explore_approx_meters,
                                "combined_time_used_s": combined_time_used_s,
                                "combined_time_limit_s": combined_time_limit_s,
                                "combined_time_limit_hit": combined_time_limit_hit,
                                "combined_time_breakdown": combined_time_breakdown_info,
                            }
                            with open(metrics_path, "w", encoding="utf-8") as mf:
                                json.dump(metrics_payload, mf, indent=2)
                            p_html = render_custom_subgoal_log_html(pddl_log_dir)
                            if not p_html:
                                p_html = render_pddl_baseline_log_dir_to_html(
                                    pddl_log_dir,
                                    episode_runtime_sec=float(runtime),
                                    episode_metrics=metrics_payload,
                                )
                            if p_html:
                                cprint(
                                    "✓ PDDL interactive HTML refreshed with episode runtime:",
                                    "green",
                                )
                                cprint(f"  {p_html}", "green")
                                _copy_pddl_interactive_html_to_outputs_dir(
                                    p_html, config
                                )
                    except Exception as pddl_html_err:
                        cprint(
                            f"⚠ Failed to refresh PDDL interactive HTML runtime: {pddl_html_err}",
                            "yellow",
                        )
        else:
            cprint(f"⚠ Trace file not found in directory: {traces_dir}", "yellow")
    except Exception as trace_error:
        cprint(f"⚠ Failed to generate trace HTML: {trace_error}", "yellow")

    # aggregate metrics across all runs.
    if conn is None:
        all_metrics = aggregate_measures(
            {run_id: aggregate_measures(v) for run_id, v in stats_episodes.items()}
        )
        cprint("\n---------------------------------", "blue")
        cprint("Metrics Across All Runs:", "blue")
        for k, v in all_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("\n---------------------------------", "blue")
        # Write aggregated results across experiment
        write_to_csv(config.paths.end_result_file_path, all_metrics)
    else:
        conn.send(stats_episodes)

    final_object_snapshot = _snapshot_object_placements(env_interface)
    env_interface.env.close()
    del env_interface

    _generate_prediviz_after_run(
        config,
        ran_episode_ids,
        initial_snapshot=initial_object_snapshot,
        final_snapshot=final_object_snapshot,
    )

    if conn is not None:
        # Potentially we may want to send something

        conn.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run planner
    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )
