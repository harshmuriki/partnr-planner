#!/usr/bin/env python3

import csv
import json
import os
import os.path as osp
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# dataset_generation.benchmark_generation.generate_episodes imports habitat_sim. In the
# "habitat" conda env this repo's README calls for, habitat_sim must be imported before
# pandas -- pandas' bundled LLVM loader otherwise shadows the native lib habitat_sim/llvmlite
# needs, and the import fails with "OSError: Could not find/load shared object file" deep in
# numba/llvmlite. Keep these dataset_generation/habitat_llm imports ahead of `import pandas`.
from dataset_generation.benchmark_generation.evaluation_generation.goal_state_propositions import (
    evaluation_payload,
)
from dataset_generation.benchmark_generation.evaluation_generation.metadata_mapping import (
    generate_hash_to_text,
)
from dataset_generation.benchmark_generation.generate_episodes import (
    default_gen_config,
    generate_episode,
    get_generator_state_semantic_debug_info,
    initialize_generator,
    save_ep_dataset,
)
from dataset_generation.benchmark_generation.parse_generated_instructions import (
    InstructionParser,
)
from habitat_llm.llm import instantiate_llm
from habitat_llm.sims.metadata_interface import default_metadata_dict

import pandas as pd

PARTNR_SCENE_IDS = [
    "102817140",
    "106366386_174226770",
    "106366410_174226806",
    "107734176_176000019",
    "107733960_175999701",
    "103997895_171031182",
    "102344529",
    "102816756",
    "106878915_174887025",
    "104348361_171513414",
    "102815835",
    "104348010_171512832",
    "106878960_174887073",
    "108736824_177263559",
    "108736872_177263607",
    "108294870_176710551",
    "108736737_177263406",
    "107734449_176000403",
    "108736851_177263586",
    "108736635_177263256",
    "107734479_176000442",
    "106879044_174887172",
    "106878945_174887058",
    "106366353_174226695",
    "106366173_174226431",
    "105515448_173104512",
    "104348463_171513588",
    "104348082_171512994",
    "103997424_171030444",
    "102817200",
    "102344049",
    "102344193",
    "102344403",
    "102344457",
    "102344022",
    "102344250",
    "102816216",
    "102816009",
    "103997460_171030507",
    "104862681_172226874",
    "102344280",
    "108294897_176710602",
    "108294573_176710113",
    "105515211_173104179",
    "104862639_172226823",
    "102815859",
    "104862669_172226853",
    "103997919_171031233",
    "104862621_172226772",
    "104862660_172226844",
]

SCENES_DIR = "data/hssd-hab/scenes-partnr-filtered"
SCENE_INFO_CACHE_DIR = "data/datasets/custom/scene_info"
OUTPUT_ROOT = "data/datasets/custom"
FPMODELS_CSV = "data/hssd-hab/metadata/fpmodels-with-decomposed.csv"
PROMPT_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "prompt.txt")
MAX_SPAWN_TRIES = 3
def slugify_folder_label(text: str, fallback: str = "episode") -> str:
    raw = (text or "").strip().lower()
    raw = raw.replace("_", "-")
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw).strip("-")
    if not raw:
        raw = fallback
    return raw[:80]


def list_available_scenes() -> List[str]:
    scenes = []
    for scene_id in PARTNR_SCENE_IDS:
        path = osp.join(SCENES_DIR, f"{scene_id}.scene_instance.json")
        if osp.exists(path):
            scenes.append(scene_id)
    return scenes


def json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(x) for x in obj]
        return str(obj)


def write_prediviz(dataset_path: str, out_dir: str, logs: List[str]) -> Optional[str]:
    meta_dir = osp.join(out_dir, "prediviz_metadata")
    viz_dir = osp.join(out_dir, "prediviz")
    repo = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dataset_generation.benchmark_generation.metadata_extractor",
                "--dataset-path",
                dataset_path,
                "--scene-metadata-cache",
                SCENE_INFO_CACHE_DIR,
                "--save-dir",
                meta_dir,
            ],
            check=True,
            cwd=repo,
        )
        subprocess.run(
            [
                sys.executable,
                osp.join("scripts", "prediviz", "viz.py"),
                "--dataset",
                dataset_path,
                "--metadata-dir",
                meta_dir,
                "--save-path",
                viz_dir,
                "--episode-id",
                "0",
            ],
            check=True,
            cwd=repo,
        )
    except Exception as exc:
        logs.append(f"PrediViz failed: {exc}")
        return None
    step = osp.join(viz_dir, "viz_0", "step_0.png")
    if osp.exists(step):
        logs.append(f"Wrote PrediViz {step}")
        return step
    logs.append("PrediViz finished but step_0.png was missing")
    return None


def extract_init_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    marker = cleaned.find("JSON_OUTPUT:")
    if marker != -1:
        cleaned = cleaned[marker + len("JSON_OUTPUT:") :].strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_one_item_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def normalize_region(name: str, valid_rooms: List[str]) -> str:
    if name in valid_rooms:
        return name
    alt = name.replace(" ", "_")
    if alt in valid_rooms:
        return alt
    indexed = f"{alt}_0"
    if indexed in valid_rooms:
        return indexed
    return name


def enrich_scene_info(scene_info: Dict[str, Any]) -> Dict[str, Any]:
    scene_info["all_furniture"] = []
    scene_info["all_rooms"] = []
    for room, furniture_room in scene_info.get("furniture", {}).items():
        scene_info["all_furniture"].extend(furniture_room)
        if room not in scene_info["all_rooms"]:
            scene_info["all_rooms"].append(room)
    scene_info["all_furniture"] = list(set(scene_info["all_furniture"]))
    if "recep_to_description" not in scene_info:
        scene_info["recep_to_description"] = generate_hash_to_text(
            FPMODELS_CSV,
            scene_info.get("receptacle_to_handle", {}),
        )
    return scene_info


def format_house_furniture(scene_info: Dict[str, Any]) -> str:
    descriptions = scene_info.get("recep_to_description", {})
    lines: List[str] = []
    for room, furniture_ids in scene_info.get("furniture", {}).items():
        if not furniture_ids:
            continue
        lines.append(f"{room}:")
        seen = set()
        for furn_id in furniture_ids:
            if furn_id in seen:
                continue
            seen.add(furn_id)
            desc = descriptions.get(furn_id, "")
            if desc:
                lines.append(f"  - {furn_id}: {desc}")
            else:
                lines.append(f"  - {furn_id}")
    return "\n".join(lines)


class EpisodeGenerationService:
    def __init__(self) -> None:
        self._parser = InstructionParser()
        self._llm = None
        self._generator = None
        self._generator_scene_id: Optional[str] = None
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        with open(PROMPT_PATH, "r") as f:
            return f.read()

    def _get_llm(self):
        if self._llm is None:
            self._llm = instantiate_llm(
                "openai_chat",
                generation_params={
                    "model": "gpt-6-astra",
                    "max_tokens": 4000,
                    "stop": "",
                    "request_timeout": 120,
                    "temperature": 0,
                },
            )
        return self._llm

    def _get_generator(self):
        if self._generator is None:
            gen_config = dict(default_gen_config)
            gen_config["ep_dataset_output"] = "test.json.gz"
            self._generator = initialize_generator(gen_config, default_metadata_dict)
        return self._generator

    def _build_scene_info(self, generator) -> Dict[str, Any]:
        mi = generator.metadata_interface
        metadata_dict = default_metadata_dict
        ovmm_metadata = pd.read_csv(
            osp.join(metadata_dict["metadata_folder"], metadata_dict["obj_metadata"])
        )
        object_names = list(set(ovmm_metadata["clean_category"].dropna().tolist()))

        room_to_id = {}
        for k, v in mi.region_semname_to_id.items():
            room_to_id[k] = generator.sim.semantic_scene.regions[v].id

        affordances_csv = osp.join(
            metadata_dict["metadata_folder"], metadata_dict["object_affordances"]
        )
        object_affordances: List[List[str]] = [[], [], [], []]
        if osp.exists(affordances_csv):
            with open(affordances_csv, "r") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx < len(object_affordances):
                        object_affordances[idx] = row

        scene_info = {
            "objects": object_names,
            "furniture": mi.get_region_rec_contents(generator.sim),
            "receptacle_to_handle": dict(mi.recobj_semname_to_handle),
            "room_to_id": room_to_id,
            "object_affordances": object_affordances,
        }
        return enrich_scene_info(scene_info)

    def get_scene_info(self, scene_id: str) -> Dict[str, Any]:
        os.makedirs(SCENE_INFO_CACHE_DIR, exist_ok=True)
        cache_path = osp.join(SCENE_INFO_CACHE_DIR, f"{scene_id}.json")
        if osp.exists(cache_path):
            with open(cache_path, "r") as f:
                scene_info = json.load(f)
            return enrich_scene_info(scene_info)

        generator = self._get_generator()
        generator.initialize_fresh_scene(scene_id)
        self._generator_scene_id = scene_id
        scene_info = self._build_scene_info(generator)
        with open(cache_path, "w") as f:
            json.dump(scene_info, f, indent=2)
        return scene_info

    def _normalize_state_list(
        self, entries: List[Dict[str, Any]], scene_info: Dict[str, Any]
    ) -> None:
        valid_rooms = scene_info.get("all_rooms", [])
        for entry in entries:
            if "object_classes" in entry:
                entry["object_classes"] = _as_one_item_list(entry["object_classes"])
            if "furniture_names" in entry:
                entry["furniture_names"] = _as_one_item_list(entry["furniture_names"])
            if "allowed_regions" in entry:
                regions = _as_one_item_list(entry["allowed_regions"])
                entry["allowed_regions"] = [
                    normalize_region(str(region), valid_rooms) for region in regions
                ]
            if "next_to" in entry:
                entry["next_to"] = _as_one_item_list(entry["next_to"])
            if "location" not in entry:
                entry["location"] = "on"

    def _normalize_init(self, init_episode: Dict[str, Any], scene_info: Dict[str, Any]):
        self._normalize_state_list(init_episode.get("initial_state", []), scene_info)
        self._normalize_state_list(init_episode.get("goal_state", []), scene_info)

    def _call_llm(
        self, scene_info: Dict[str, Any], user_prompt: str, goal_prompt: str = ""
    ) -> str:
        objects_list = "\n".join(sorted(scene_info.get("objects", [])))
        prompt = (
            self._prompt_template.replace(
                "{house_furniture}", format_house_furniture(scene_info)
            )
            .replace("{objects_list}", objects_list)
            .replace("{user_prompt}", user_prompt)
            .replace("{goal_prompt}", goal_prompt.strip() or "(empty — copy initial_state into goal_state)")
        )
        llm = self._get_llm()
        return llm.generate(prompt, max_length=4000) or ""

    def generate(
        self,
        scene_id: str,
        user_prompt: str,
        goal_prompt: str = "",
        folder_label: Optional[str] = None,
        instruction: Optional[str] = None,
        skip_prediviz: bool = False,
    ) -> Dict[str, Any]:
        if not user_prompt.strip():
            return {"ok": False, "error": "Start state is empty."}
        if scene_id not in list_available_scenes():
            return {"ok": False, "error": f"Unknown or missing scene: {scene_id}"}

        logs: List[str] = []
        logs.append(f"Loading scene info for {scene_id}")
        scene_info = self.get_scene_info(scene_id)

        logs.append("Calling LLM")
        try:
            raw_llm = self._call_llm(scene_info, user_prompt, goal_prompt)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"LLM call failed: {exc}",
                "logs": logs,
            }
        init_episode = extract_init_json(raw_llm)
        if init_episode is None:
            return {
                "ok": False,
                "error": "LLM did not return valid JSON.",
                "raw_llm": raw_llm,
                "logs": logs,
            }

        if "instruction" not in init_episode or not init_episode["instruction"]:
            if goal_prompt.strip():
                init_episode["instruction"] = (
                    user_prompt.strip().rstrip(".") + ", then " + goal_prompt.strip()
                )
            else:
                init_episode["instruction"] = user_prompt
        self._normalize_init(init_episode, scene_info)

        (
            is_valid,
            parsed_init,
            missing_objects,
            missing_furniture,
            missing_rooms,
        ) = self._parser.episode_init_valid(
            init_episode, scene_info, add_clutter=False
        )
        logs.append(f"Validation valid={is_valid}")

        parsed_state = parsed_init.get("initial_state", []) if parsed_init else []
        if not parsed_state:
            is_valid = False

        parsed_goal = parsed_init.get("goal_state", []) if parsed_init else []

        if not is_valid or missing_objects or missing_furniture or missing_rooms:
            return {
                "ok": False,
                "error": "LLM JSON failed scene validation.",
                "raw_llm": raw_llm,
                "parsed": parsed_init or init_episode,
                "missing_objects": missing_objects,
                "missing_furniture": missing_furniture,
                "missing_rooms": missing_rooms,
                "logs": logs,
            }

        for entry in parsed_state:
            if "location" not in entry:
                entry["location"] = "on"
            entry["smart_placement"] = True
        for entry in parsed_goal:
            if "location" not in entry:
                entry["location"] = "on"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_label = slugify_folder_label(
            str(
                folder_label
                or parsed_init.get("folder_label")
                or init_episode.get("folder_label")
                or parsed_init.get("instruction")
                or user_prompt
            )
        )
        if instruction:
            parsed_init["instruction"] = instruction.strip().strip('"')
            init_episode["instruction"] = parsed_init["instruction"]
        run_dir_name = f"{folder_label}_{timestamp}"
        parsed_init["scene_id"] = scene_id
        parsed_init["episode_id"] = f"gui|{scene_id}|{run_dir_name}"
        parsed_init["instruction"] = parsed_init.get("instruction") or user_prompt
        parsed_init["folder_label"] = folder_label

        logs.append("Spawning objects in Habitat")
        generator = self._get_generator()
        ep = None
        details_log: Dict[str, Any] = {}
        for try_idx in range(MAX_SPAWN_TRIES):
            ep, details_log = generate_episode(generator, parsed_init)
            if ep is not None:
                break
            failure = details_log.get("failure_mode")
            logs.append(f"Spawn try {try_idx + 1} failed: {failure}")
            if failure != "unstable state":
                break

        if ep is None:
            return {
                "ok": False,
                "error": "Episode spawn failed.",
                "raw_llm": raw_llm,
                "parsed": parsed_init,
                "spawn_log": json_safe(details_log),
                "debug": get_generator_state_semantic_debug_info(generator),
                "logs": logs,
            }

        out_dir = osp.join(OUTPUT_ROOT, scene_id, run_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        dataset_path = osp.join(out_dir, "dataset.json.gz")
        save_ep_dataset([ep], dataset_path)
        scene_info_out = osp.join(out_dir, "scene_info.json")
        with open(scene_info_out, "w") as f:
            json.dump(scene_info, f, indent=2)

        rigid_objs = []
        for item in ep.rigid_objs:
            rigid_objs.append(item[0] if isinstance(item, (list, tuple)) else str(item))

        logs.append(f"Wrote {dataset_path}")
        prediviz_path = None
        if not skip_prediviz:
            prediviz_path = write_prediviz(dataset_path, out_dir, logs)
        return {
            "ok": True,
            "instruction": parsed_init.get("instruction", ""),
            "parsed": parsed_init,
            "raw_llm": raw_llm,
            "dataset_path": dataset_path,
            "prediviz_path": prediviz_path,
            "name_to_receptacle": dict(ep.name_to_receptacle),
            "rigid_objs": rigid_objs,
            "evaluation_propositions": evaluation_payload(
                ep.evaluation_propositions, ep.evaluation_constraints
            ),
            "debug": get_generator_state_semantic_debug_info(generator),
            "spawn_log": json_safe(details_log),
            "logs": logs,
        }
