"""Discover ReAct evaluation runs and persist per-run annotations."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

FAILURE_LABELS = [
    {
        "id": "insufficient_planning",
        "label": "Insufficient planning — Stopped searching or planning too early",
    },
    {
        "id": "no_action_vlm",
        "label": "Missing action — VLM omitted a required action for an object",
    },
    {
        "id": "placement",
        "label": "Placement error — Placed objects in the wrong location",
    },
    {"id": "other_object", "label": "Wrong object — Used an incorrect object"},
    {
        "id": "never_gave_up",
        "label": "Failure to stop — Kept trying when it should have given up",
    },
]

EFFICIENCY_LABELS = [
    {
        "id": "hallucinated_object",
        "label": "Hallucinated object — Tried to use an object that does not exist",
    },
    {
        "id": "extra_subgoals",
        "label": "Extra subgoals — Completed unnecessary subgoals",
    },
    {
        "id": "repeated_actions",
        "label": "Repeated actions — Repeated the same action unnecessarily",
    },
    {
        "id": "precondition_failures",
        "label": "Precondition error — Tried an action before its requirements were met",
    },
]

FAILURE_IDS = {item["id"] for item in FAILURE_LABELS}
EFFICIENCY_IDS = {item["id"] for item in EFFICIENCY_LABELS}

AXIS_ORDER = {"ACC": 0, "INC": 1, "OUT": 2}
KIND_ORDER = {
    "BASE": 0,
    "SUB": 1,
    "ABS": 2,
    "CON": 3,
    "DIS": 4,
    "AMB": 5,
    "ROOM": 6,
    "CAND": 7,
}
AXIS_SUMMARY = {
    "ACC": "Robot memory: Accurate",
    "INC": "Robot memory: Incomplete",
    "OUT": "Robot memory: Outdated",
}
KIND_SUMMARY = {
    "SUB": "Object availability: Substitute Available",
    "ABS": "Object availability: No Suitable Object",
    "CON": "Object containment: Inside Closed Receptacle",
    "DIS": "Distractors: Present",
    "AMB": "Instruction: Underspecified",
    "ROOM": "Localization: Room Known",
    "CAND": "Localization: Candidate Rooms",
}

VARIANT_FOLDER_RE = re.compile(r"^t(\d+)-([a-z]+)-(.+)$", re.I)
TASK_DIR_RE = re.compile(r"^task_(\d+)$", re.I)
HEADING_RE = re.compile(r"^##\s+")
TRACE_NAME_RE = re.compile(r"^trace-episode_(.+)_(\d+)-(\d+)\.txt$")
REACT_ACTION_RE = re.compile(r"(?m)^([A-Za-z]+)\[([^\]]*)\]")
PDDL_FAILURE_MARKERS = (
    "unexpected failure",
    "action failed",
    "failed",
    "error",
    "✗ action result: failed",
    "action result: failed",
)
PDDL_SUCCESS_MARKERS = (
    "successful execution",
    "✓ action result: success",
    "action result: success",
    "success",
)
PDDL_ACTION_RE = re.compile(r"^Action:\s*([A-Za-z_]+)\[(.*)\]\s*$")
PDDL_EXPLORE_RE = re.compile(
    r"^High-level special-case:\s*([A-Za-z_]+)\[(.*)\]\s*$"
)
STEP_IMAGE_RE = re.compile(r"^step_\d+\.png$")
UNEXPECTED_FAILURE_RE = re.compile(r"^Unexpected failure!\s*-\s*", re.I)

PRECONDITION_PATTERNS = [
    re.compile(pat, re.I)
    for pat in (
        r"Agent too far from object",
        r"Not close enough",
        r"occluded or too far",
        r"inside closed",
        r"requires faucet",
        r"not close enough to a water source",
        r"not holding any object",
        r"already held",
        r"Skill took too long",
        r"Could not find a suitable nav target",
    )
]
WRONG_OBJECT_PATTERNS = [
    re.compile(pat, re.I)
    for pat in (r"is not articulated", r"cannot be Opened", r"does not afford")
]
HALLUCINATION_PATTERNS = [
    re.compile(pat, re.I)
    for pat in (
        r"not present in the graph",
        r"isn't valid",
        r"does not have a simulator handle",
    )
]
PLACEMENT_PATTERNS = [
    re.compile(pat, re.I)
    for pat in (
        r"No valid placements found",
        r"Failed to place!",
        r"has no receptacle for proposition",
        r"Destination must be furniture",
    )
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def infer_react_success(result: str) -> bool:
    lowered = result.lower()
    if any(marker in lowered for marker in ("fail", "error", "✗")):
        return False
    return "success" in lowered


def _normalize_error_message(result: str) -> str:
    text = re.sub(r"\s+", " ", str(result or "")).strip()
    return UNEXPECTED_FAILURE_RE.sub("", text).strip()


def _matches_any(message: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(message) for pattern in patterns)


def _failed_step_message(step: Dict[str, Any]) -> str:
    result = str(step.get("result") or "").strip()
    if not result:
        return ""
    success = step.get("success")
    if success is None:
        success = infer_react_success(result)
    if success:
        return ""
    return _normalize_error_message(result)


def _cluster_messages(messages: Sequence[str], limit: int = 4) -> str:
    counts = Counter(item for item in messages if item)
    parts = []
    for message, count in counts.most_common(limit):
        clipped = message if len(message) <= 140 else message[:137] + "..."
        parts.append(f"{clipped} ×{count}" if count > 1 else clipped)
    leftover = len(counts) - min(len(counts), limit)
    if leftover > 0:
        parts.append(f"+{leftover} more")
    return "; ".join(parts)


def infer_trace_issues(
    steps: Sequence[Dict[str, Any]],
    succeeded: Optional[bool] = None,
) -> Dict[str, Any]:
    """Map recurring skill errors onto annotation failure/efficiency tags."""
    failed_messages: List[str] = []
    precondition_msgs: List[str] = []
    hallucination_msgs: List[str] = []
    placement_msgs: List[str] = []
    wrong_object_msgs: List[str] = []
    action_keys: List[str] = []
    explore_args: List[str] = []
    wait_count = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        args = str(step.get("args") or "").strip()
        key = f"{action}[{args}]"
        action_keys.append(key)
        if action == "Explore" and args:
            explore_args.append(args)
        if action == "Wait":
            wait_count += 1
        message = _failed_step_message(step)
        if not message:
            continue
        failed_messages.append(message)
        if _matches_any(message, HALLUCINATION_PATTERNS):
            hallucination_msgs.append(message)
        elif _matches_any(message, PLACEMENT_PATTERNS):
            placement_msgs.append(message)
        elif _matches_any(message, WRONG_OBJECT_PATTERNS):
            wrong_object_msgs.append(message)
        elif _matches_any(message, PRECONDITION_PATTERNS):
            precondition_msgs.append(message)
        else:
            precondition_msgs.append(message)

    repeats = [
        f"{key} ×{count}"
        for key, count in Counter(action_keys).items()
        if count >= 2 and not key.startswith("Wait[")
    ]
    extra_explores = any(count >= 3 for count in Counter(explore_args).values())
    has_precondition = bool(precondition_msgs or wrong_object_msgs)
    run_succeeded = bool(succeeded)

    efficiency: List[str] = []
    failure: List[str] = []
    if has_precondition:
        efficiency.append("precondition_failures")
        failure.append("insufficient_planning")
    if hallucination_msgs:
        efficiency.append("hallucinated_object")
        failure.append("no_action_vlm")
    if placement_msgs:
        failure.append("placement")
        if run_succeeded:
            efficiency.append("precondition_failures")
    if wrong_object_msgs:
        failure.append("other_object")
    if repeats:
        efficiency.append("repeated_actions")
    if run_succeeded and extra_explores:
        efficiency.append("extra_subgoals")
    if wait_count >= 3:
        failure.append("no_action_vlm")

    unique_efficiency = [item for item in dict.fromkeys(efficiency) if item in EFFICIENCY_IDS]
    unique_failure = [item for item in dict.fromkeys(failure) if item in FAILURE_IDS]

    note_lines: List[str] = []
    if precondition_msgs:
        note_lines.append(f"precondition_failures: {_cluster_messages(precondition_msgs)}")
    if wrong_object_msgs:
        note_lines.append(f"other_object / wrong affordance: {_cluster_messages(wrong_object_msgs)}")
    if hallucination_msgs:
        note_lines.append(f"hallucinated_object: {_cluster_messages(hallucination_msgs)}")
    if placement_msgs:
        note_lines.append(f"placement: {_cluster_messages(placement_msgs)}")
    if repeats:
        note_lines.append("repeated_actions: " + "; ".join(repeats[:6]))
    if run_succeeded and extra_explores:
        note_lines.append("extra_subgoals: same room explored 3+ times")
    notes = ""
    if note_lines:
        notes = "Auto-detected from failed steps:\n" + "\n".join(f"- {line}" for line in note_lines)

    return {
        "failure_reasons": unique_failure,
        "efficiency_errors": unique_efficiency,
        "notes": notes,
        "failed_step_count": len(failed_messages),
    }


def parse_react_trace_file(trace_path: str) -> Dict[str, Any]:
    """Parse a ReAct trace text file.

    Line-anchored action matching is based on
    ``scripts/view_trace_logs._parse_trace_file_react`` so this app does not
    import habitat_llm. CamelCase skills such as PowerOff are kept intact.
    """
    content = Path(trace_path).read_text(encoding="utf-8")
    task = ""
    lines = content.split("\n")
    if lines and lines[0].startswith("Task:"):
        task = lines[0].replace("Task:", "", 1).strip()

    steps: List[Dict[str, Any]] = []
    action_matches = list(REACT_ACTION_RE.finditer(content))
    for i, action_match in enumerate(action_matches):
        action_name = action_match.group(1)
        action_args = action_match.group(2)
        step: Dict[str, Any] = {"action": action_name, "args": action_args}

        lookback_start = action_matches[i - 1].end() if i > 0 else 0
        lookback_section = content[lookback_start : action_match.start()]
        thoughts = re.findall(r"^Thought:[ \t]*(.*)$", lookback_section, re.M)
        thought = next((item.strip() for item in reversed(thoughts) if item.strip()), "")
        if thought:
            step["thought"] = thought

        forward_end = (
            action_matches[i + 1].start()
            if i < len(action_matches) - 1
            else len(content)
        )
        forward_section = content[action_match.end() : forward_end]
        result_match = re.search(
            r"Assigned!Result:\s*(.*?)(?=\n(?:Objects:|Thought:|[A-Z][a-z]+\[|$))",
            forward_section,
            re.DOTALL,
        )
        if result_match:
            result = result_match.group(1).strip()
            step["result"] = result
            if action_name == "Done":
                step["success"] = True
            else:
                step["success"] = infer_react_success(result)
        elif action_name == "Done":
            step["success"] = True

        objects_match = re.search(
            r"Objects:\s*(.*?)(?=\n(?:Thought:|[A-Z][a-z]+\[|$))",
            forward_section,
            re.DOTALL,
        )
        if objects_match:
            objects_text = objects_match.group(1).strip()
            if objects_text and objects_text != "No objects found yet":
                step["objects"] = objects_text
        steps.append(step)

    return {"task": task, "steps": steps, "total_steps": len(steps)}


def infer_pddl_step_success(block_lines: Sequence[str], action_name: str) -> bool:
    block_text = "\n".join(block_lines).lower()
    if any(marker in block_text for marker in PDDL_FAILURE_MARKERS):
        return False
    if any(marker in block_text for marker in PDDL_SUCCESS_MARKERS):
        return True
    if action_name.lower() == "explore":
        return True
    return action_name.lower() == "done"


def parse_pddl_trace_file(trace_path: str) -> Dict[str, Any]:
    """Parse VLM-TAMP PDDL traces (Action/Observation and Explore special-cases)."""
    content = Path(trace_path).read_text(encoding="utf-8")
    lines = content.split("\n")
    task = ""
    if lines and lines[0].startswith("Task:"):
        task = lines[0].replace("Task:", "", 1).strip()

    steps: List[Dict[str, Any]] = []
    current_thought = ""
    current_subgoal = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("Thought:"):
            current_thought = line.replace("Thought:", "", 1).strip()
            i += 1
            continue
        if line.startswith("VLM English plan:"):
            plan_lines = [line.replace("VLM English plan:", "", 1).strip()]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt.startswith(
                    (
                        "Subgoal:",
                        "Action:",
                        "High-level special-case:",
                        "Thought:",
                        "VLM English plan:",
                    )
                ):
                    break
                if nxt:
                    plan_lines.append(nxt)
                i += 1
            current_thought = "\n".join(item for item in plan_lines if item).strip()
            continue
        if line.startswith("Subgoal:"):
            current_subgoal = line.replace("Subgoal:", "", 1).strip()
            i += 1
            continue
        action_match = PDDL_ACTION_RE.match(line) or PDDL_EXPLORE_RE.match(line)
        if not action_match:
            i += 1
            continue
        action_name = action_match.group(1).strip()
        action_args = action_match.group(2).strip()
        step: Dict[str, Any] = {
            "action": action_name,
            "args": action_args,
        }
        if current_thought:
            step["thought"] = current_thought
        if current_subgoal:
            step["subgoal"] = current_subgoal
            if not step.get("thought"):
                step["thought"] = f"Subgoal: {current_subgoal}"
        j = i + 1
        block_lines: List[str] = []
        result_lines: List[str] = []
        while j < len(lines):
            nxt = lines[j].strip()
            if (
                nxt.startswith("Action:")
                or nxt.startswith("Subgoal:")
                or nxt.startswith("High-level special-case:")
            ):
                break
            if nxt:
                block_lines.append(nxt)
                if nxt.startswith("Observation:"):
                    result_lines.append(nxt.replace("Observation:", "", 1).strip())
                elif nxt.startswith("Obs:"):
                    result_lines.append(nxt.replace("Obs:", "", 1).strip())
                elif "ACTION RESULT:" in nxt.upper() or nxt.startswith("Action failed:"):
                    result_lines.append(nxt)
            j += 1
        step["result"] = "\n".join(result_lines).strip() or "No result"
        step["success"] = infer_pddl_step_success(block_lines, action_name)
        if block_lines:
            step["pddl_log"] = "\n".join(block_lines)
        steps.append(step)
        i = j

    return {"task": task, "steps": steps, "total_steps": len(steps)}


def parse_eval_trace_file(trace_path: str, planner: str = "react") -> Dict[str, Any]:
    if planner == "vlm_tamp_pddl":
        return parse_pddl_trace_file(trace_path)
    return parse_react_trace_file(trace_path)


def prompt_path_for_trace(trace_path: Path) -> Optional[Path]:
    path = Path(trace_path)
    if path.parent.parent.name != "traces":
        return None
    candidate = (
        path.parent.parent.parent
        / "prompts"
        / path.parent.name
        / path.name.replace("trace-", "prompt-", 1)
    )
    return candidate if candidate.is_file() else None


def parse_object_states(text: str) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        rest = rest.strip()
        if not name:
            continue
        states_text = ""
        if ". States:" in rest:
            rest, states_text = rest.split(". States:", 1)
            rest = rest.strip()
            states_text = states_text.strip()
        held = rest.lower().startswith("held by")
        location = ""
        room = ""
        if not held:
            if " in " in rest:
                location, room = rest.rsplit(" in ", 1)
                location = location.strip()
                room = room.strip().rstrip(".")
            else:
                location = rest.rstrip(".")
        states: Dict[str, str] = {}
        for part in states_text.split(","):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            states[key.strip()] = value.strip()
        objects.append(
            {
                "name": name,
                "held": held,
                "location": location,
                "room": room,
                "states": states,
            }
        )
    return objects


CRITERION_RE = re.compile(r"^([A-Za-z_]+)\((.*)\)$")


def _object_by_name(objects: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("name") or ""): item
        for item in objects
        if isinstance(item, dict) and item.get("name")
    }


def _state_flag(obj: Dict[str, Any], *keys: str) -> Optional[bool]:
    states = obj.get("states") or {}
    lowered = {str(key).lower(): str(value).strip().lower() for key, value in states.items()}
    for key in keys:
        if key in lowered:
            return lowered[key] in {"true", "1", "yes"}
    return None


def infer_criteria_from_trace(
    criteria: Sequence[str],
    objects: Sequence[Dict[str, Any]],
) -> Dict[str, Optional[bool]]:
    by_name = _object_by_name(objects)
    inferred: Dict[str, Optional[bool]] = {item: None for item in criteria}
    for item in criteria:
        match = CRITERION_RE.match(str(item).strip())
        if not match:
            continue
        function_name = match.group(1)
        args = [part.strip() for part in match.group(2).split(",") if part.strip()]
        if function_name in {"is_on_top", "is_inside"} and len(args) >= 2:
            obj = by_name.get(args[0])
            inferred[item] = bool(
                obj and not obj.get("held") and obj.get("location") == args[1]
            )
        elif function_name == "is_on_floor" and args:
            obj = by_name.get(args[0])
            location = str(obj.get("location") or "") if obj else ""
            inferred[item] = bool(obj and not obj.get("held") and location.startswith("floor_"))
        elif function_name == "is_in_room" and len(args) >= 2:
            obj = by_name.get(args[0])
            inferred[item] = bool(obj and obj.get("room") == args[1])
        elif function_name == "is_next_to" and len(args) >= 2:
            left = by_name.get(args[0])
            right = by_name.get(args[1])
            inferred[item] = bool(
                left
                and right
                and not left.get("held")
                and not right.get("held")
                and left.get("location")
                and left.get("location") == right.get("location")
            )
        elif function_name == "is_clean" and args:
            obj = by_name.get(args[0])
            inferred[item] = _state_flag(obj, "clean") if obj else False
        elif function_name == "is_filled" and args:
            obj = by_name.get(args[0])
            inferred[item] = _state_flag(obj, "filled") if obj else False
        elif function_name == "is_powered_on" and args:
            obj = by_name.get(args[0])
            inferred[item] = _state_flag(obj, "powered on", "powered_on") if obj else False
        elif function_name == "is_powered_off" and args:
            obj = by_name.get(args[0])
            flag = _state_flag(obj, "powered on", "powered_on") if obj else None
            inferred[item] = (not flag) if flag is not None else False
    return inferred


def final_object_states(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for step in reversed(list(steps)):
        states = step.get("object_states")
        if isinstance(states, list) and states:
            return [item for item in states if isinstance(item, dict)]
        text = step.get("objects")
        if text:
            parsed = parse_object_states(str(text))
            if parsed:
                return parsed
    return []


def merge_criteria_from_trace(
    auto_criteria: Dict[str, bool],
    trace_criteria: Dict[str, Optional[bool]],
) -> Dict[str, bool]:
    merged = dict(auto_criteria)
    for key, value in trace_criteria.items():
        if value is not None:
            merged[key] = bool(value)
    return merged


def load_world_graph(trace_path: str) -> Dict[str, Any]:
    prompt_path = prompt_path_for_trace(Path(trace_path))
    rooms: List[Dict[str, Any]] = []
    faucets: List[str] = []
    objects: List[Dict[str, Any]] = []
    if prompt_path is None:
        return {"rooms": rooms, "faucets": faucets, "objects": objects}
    text = prompt_path.read_text(encoding="utf-8", errors="replace")
    furniture_match = re.search(
        r"(?ms)Furniture:\s*\n(.*?)(?=^The following furnitures have a faucet:|^Objects:|^Possible Actions:)",
        text,
    )
    if furniture_match:
        for raw in furniture_match.group(1).splitlines():
            line = raw.strip()
            if not line or ":" not in line:
                continue
            room, rest = line.split(":", 1)
            furniture = [
                item.strip()
                for item in rest.split(",")
                if item.strip() and not item.strip().startswith("floor_")
            ]
            rooms.append({"name": room.strip(), "furniture": furniture})
    faucet_match = re.search(
        r"(?m)^The following furnitures have a faucet:\s*(.+)$",
        text,
    )
    if faucet_match:
        faucets = [
            item.strip()
            for item in faucet_match.group(1).split(",")
            if item.strip()
        ]
    objects_match = re.search(
        r"(?ms)^Objects:\s*\n(.*?)(?=^Possible Actions:)",
        text,
    )
    if objects_match:
        objects = parse_object_states(objects_match.group(1))
    return {"rooms": rooms, "faucets": faucets, "objects": objects}


def parse_variant_folder(name: str) -> Dict[str, Any]:
    match = VARIANT_FOLDER_RE.match(name.strip())
    if not match:
        return {
            "task_number": None,
            "axis": None,
            "kind": None,
            "variant_folder": name,
        }
    return {
        "task_number": int(match.group(1)),
        "axis": match.group(2).upper(),
        "kind": match.group(3).upper(),
        "variant_folder": name,
    }


BATCH_LABELS = {
    "vlm_tamp_pddl": "VLM-TAMP PDDL",
}


def batch_from_run_id(run_id: str) -> str:
    parts = Path(run_id).parts
    if len(parts) >= 3 and TASK_DIR_RE.match(parts[1]):
        return parts[0]
    if len(parts) >= 2:
        return parts[0]
    return "(root)"


def batch_label(batch_id: str) -> str:
    return BATCH_LABELS.get(batch_id, batch_id)


def _skip_trace_path(trace_path: Path) -> bool:
    return any(part.startswith("outputs_") for part in trace_path.parts)


def load_pddl_episode_metrics(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / "vlm_tamp_pddl" / "unknown" / "episode_metrics.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "llm_model": str(data.get("llm_model") or "").strip(),
        "llm_reasoning_effort": str(data.get("llm_reasoning_effort") or "").strip(),
        "prompt_tokens": data.get("prompt_tokens"),
        "completion_tokens": data.get("completion_tokens"),
        "cached_tokens": data.get("cached_tokens"),
        "llm_usd": _float_or_none(data.get("llm_usd")),
        "sim_step_count": _float_or_none(data.get("sim_step_count")),
        "episode_runtime_sec": _float_or_none(data.get("episode_runtime_sec")),
        "task_percent_complete": _float_or_none(data.get("task_percent_complete")),
        "task_state_success": _float_or_none(data.get("task_state_success")),
        "combined_time_used_s": _float_or_none(data.get("combined_time_used_s")),
        "combined_time_limit_s": _float_or_none(data.get("combined_time_limit_s")),
        "combined_time_limit_hit": data.get("combined_time_limit_hit"),
    }


def find_pddl_artifacts(
    run_dir: Path, files_root: Path
) -> Optional[Dict[str, Any]]:
    unknown = Path(run_dir) / "vlm_tamp_pddl" / "unknown"
    if not unknown.is_dir():
        return None
    try:
        rel_dir = unknown.resolve().relative_to(Path(files_root).resolve()).as_posix()
    except ValueError:
        return None
    htmls = sorted(unknown.glob("*_pddl.html"))
    graph_name = htmls[0].name if htmls else None
    if graph_name is None and (unknown / "index.html").is_file():
        graph_name = "index.html"
    image_dir = unknown / "vlm_images"
    image_names = []
    if image_dir.is_dir():
        image_names = sorted(
            path.name
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
    return {
        "rel_dir": rel_dir,
        "graph_name": graph_name,
        "prompts_html": (unknown / "vlm_prompts.html").is_file(),
        "prompts_txt": (unknown / "vlm_prompts.txt").is_file(),
        "image_names": image_names,
        "metrics": load_pddl_episode_metrics(run_dir),
    }


def _episode_id_from_trace(path: Path) -> str:
    match = TRACE_NAME_RE.match(path.name)
    if match:
        return match.group(1)
    stem = path.stem
    if stem.startswith("trace-"):
        rest = stem[len("trace-") :]
        if rest.startswith("episode_"):
            rest = rest[len("episode_") :]
        return rest.rsplit("-", 1)[0]
    return path.stem


def _run_dir_from_trace(trace_path: Path) -> Path:
    for parent in trace_path.parents:
        if parent.name == "dataset":
            return parent.parent
    return trace_path.parent.parent.parent


def images_dir_for_trace(trace_path: Path) -> Optional[Path]:
    images_dir = Path(trace_path).parent / "images"
    if images_dir.is_dir() and any(images_dir.glob("step_*.png")):
        return images_dir
    return None


def list_step_images(images_dir: Optional[Path]) -> List[Path]:
    if images_dir is None:
        return []
    directory = Path(images_dir)
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.glob("step_*.png") if STEP_IMAGE_RE.match(path.name)
    )


def step_image_path(
    images_dir: Optional[Path],
    index: int,
    allowed_roots: Sequence[Path],
) -> Optional[Path]:
    images = list_step_images(images_dir)
    if index < 1 or index > len(images):
        return None
    path = images[index - 1].resolve()
    roots = [Path(root).resolve() for root in allowed_roots]
    if not any(path.is_relative_to(root) for root in roots):
        return None
    return path


def parse_success_criteria_from_spec(spec_text: str) -> List[str]:
    lines = spec_text.splitlines()
    collecting = False
    criteria: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## success criteria":
            collecting = True
            continue
        if collecting and HEADING_RE.match(stripped):
            break
        if collecting and stripped.startswith("- "):
            criterion = stripped[2:].strip()
            if criterion.startswith("Distractors ("):
                continue
            criteria.append(criterion)
    return criteria


def parse_uncertainty_from_spec(spec_text: str) -> List[str]:
    collecting = False
    items: List[str] = []
    for line in spec_text.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## uncertainty being tested":
            collecting = True
            continue
        if collecting and HEADING_RE.match(stripped):
            break
        if collecting and stripped.startswith("- "):
            text = stripped[2:].strip()
            if text.lower().startswith("memory in this version"):
                continue
            if text:
                items.append(text)
    return items


def _rewrite_uncertainty_item(item: str) -> str:
    lowered = item.lower()
    if lowered.startswith("internal robot memory"):
        return "Robot memory" + item[len("Internal robot memory") :]
    if lowered.startswith("containment:") and not lowered.startswith("object containment"):
        return "Object containment" + item[len("Containment") :]
    return item


def format_variant_summary(
    axis: Optional[str],
    kind: Optional[str],
    spec_items: Optional[Sequence[str]] = None,
) -> str:
    items = [_rewrite_uncertainty_item(item) for item in (spec_items or [])]
    if items:
        return " and ".join(items)
    parts: List[str] = []
    axis_key = str(axis or "").upper()
    kind_key = str(kind or "").upper()
    if axis_key in AXIS_SUMMARY:
        parts.append(AXIS_SUMMARY[axis_key])
    extra = KIND_SUMMARY.get(kind_key)
    if extra:
        parts.append(extra)
    return " and ".join(parts)


def _add_handle_name(mapping: Dict[str, str], handle: Any, name: Any) -> None:
    handle_s = str(handle or "").strip()
    name_s = str(name or "").strip()
    if not handle_s or not name_s:
        return
    existing = mapping.get(handle_s)
    if (
        existing
        and _looks_like_entity_name(existing)
        and not _looks_like_entity_name(name_s)
    ):
        return
    mapping[handle_s] = name_s
    if "_:" in handle_s:
        short = handle_s.split("_:", 1)[0]
        existing_short = mapping.get(short)
        if not (
            existing_short
            and _looks_like_entity_name(existing_short)
            and not _looks_like_entity_name(name_s)
        ):
            mapping[short] = name_s


def _handle_to_name(episode: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    info = episode.get("info") or {}
    variant = info.get("variant_spec") or {}
    entity_handles = variant.get("entity_handles") or {}
    if isinstance(entity_handles, dict):
        for name, handle in entity_handles.items():
            _add_handle_name(mapping, handle, name)
    extra = (info.get("extra_info") or {}).get("obj_info") or {}
    if isinstance(extra, dict):
        for name, handle in extra.items():
            _add_handle_name(mapping, handle, name)
    return mapping


def _name_to_class(episode: Dict[str, Any]) -> Dict[str, str]:
    classes: Dict[str, str] = {}
    info = episode.get("info") or {}
    extra = info.get("extra_info") or {}
    states = extra.get("initial_state") or []
    if isinstance(states, list):
        for item in states:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            object_classes = item.get("object_classes") or []
            if name and object_classes:
                classes[name] = str(object_classes[0])
    sample_configs = extra.get("sample_configs") or info.get("sample_configs") or {}
    if isinstance(sample_configs, dict):
        for item in sample_configs.values():
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            object_classes = item.get("object_classes") or []
            if name and object_classes and name not in classes:
                classes[name] = str(object_classes[0])
    return classes


def _prop_arg_values(prop: Dict[str, Any]) -> List[str]:
    args = prop.get("args") or {}
    values: List[str] = []
    for key in (
        "object_handles",
        "receptacle_handles",
        "room_ids",
        "entity_handles_a",
        "entity_handles_b",
    ):
        raw = args.get(key) or []
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
    starred = args.get("*args") or []
    if isinstance(starred, list):
        for group in starred:
            if isinstance(group, list):
                values.extend(str(item) for item in group)
    return values


def _prop_entity_names(prop: Dict[str, Any], handle_to_name: Dict[str, str]) -> List[str]:
    names: List[str] = []
    for value in _prop_arg_values(prop):
        mapped = handle_to_name.get(value)
        if mapped is None and "_:" in value:
            mapped = handle_to_name.get(value.split("_:", 1)[0])
        names.append(mapped or value)
    return names


def _format_proposition(prop: Dict[str, Any], handle_to_name: Dict[str, str]) -> str:
    function_name = str(prop.get("function_name") or "unknown")
    parts = _prop_entity_names(prop, handle_to_name)
    readable = [part for part in parts if part and "_:" not in part and len(part) < 80]
    if readable:
        return f"{function_name}({', '.join(readable)})"
    if parts:
        return f"{function_name}({', '.join(parts)})"
    return function_name


def _looks_like_entity_name(value: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9_]*_\d+$", value, re.I)) or bool(
        re.match(r"^[a-z]+_[a-z]+_\d+$", value, re.I)
    )


def _map_proposition_to_criterion(
    prop: Dict[str, Any],
    handle_to_name: Dict[str, str],
    criteria: Sequence[str],
) -> Optional[str]:
    formatted = _format_proposition(prop, handle_to_name)
    if formatted in criteria:
        return formatted
    function_name = str(prop.get("function_name") or "")
    names = [
        name
        for name in _prop_entity_names(prop, handle_to_name)
        if _looks_like_entity_name(name)
    ]
    candidates = [
        item for item in criteria if item.startswith(f"{function_name}(")
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scored: List[tuple[int, str]] = []
    for item in candidates:
        hits = sum(1 for name in names if name in item)
        if hits:
            scored.append((hits, item))
    if not scored:
        return None
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    best_hits = scored[0][0]
    best = [item for hits, item in scored if hits == best_hits]
    if len(best) == 1:
        return best[0]
    return None


def _function_phrases(function_name: str) -> List[str]:
    special = {
        "is_on_top": ["placed on top", "on top of"],
        "is_inside": ["placed inside", "inside the"],
        "is_next_to": ["placed next to", "next to"],
        "is_in_room": ["moved to the", "moved to"],
        "is_on_floor": ["placed on the floor", "on the floor"],
        "is_clustered": ["clustered"],
        "is_powered_off": ["powered off"],
        "is_powered_on": ["powered on"],
    }
    phrases = list(special.get(function_name, []))
    fallback = " ".join(function_name.split("_")[1:]).strip()
    if fallback and fallback not in phrases:
        phrases.append(fallback)
    return phrases


def _entity_tokens(name: str, class_name: Optional[str] = None) -> List[str]:
    tokens: List[str] = []
    raw = str(name or "").strip().lower()
    if raw:
        tokens.append(raw)
        tokens.append(raw.replace("_", " "))
        tokens.append(re.sub(r"_\d+$", "", raw).replace("_", " "))
        tokens.append(raw.split("_")[0])
    if class_name:
        cls = str(class_name).strip().lower().replace("_", " ")
        if cls:
            tokens.append(cls)
    seen = set()
    unique: List[str] = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique


def _failed_explanation_lines(explanation: str) -> List[str]:
    text = str(explanation or "")
    lines: List[str] = []
    for marker in (
        "Missing steps:",
        "Completed steps were later undone:",
        "Steps were completed out of order:",
    ):
        if marker not in text:
            continue
        rest = text.split(marker, 1)[1]
        for other in (
            "Missing steps:",
            "Completed steps were later undone:",
            "Steps were completed out of order:",
            "The same object should have been used",
            "Different objects should have been used",
            "Placements should have been made",
        ):
            if other == marker:
                continue
            cut = rest.find(other)
            if cut >= 0:
                rest = rest[:cut]
        for raw in rest.splitlines():
            stripped = raw.strip()
            if stripped.startswith("-") and not stripped.startswith("-   "):
                item = stripped[1:].strip()
                if item and "should have been completed after" not in item.lower():
                    lines.append(item)
    return lines


def _line_match_score(
    line: str,
    prop: Dict[str, Any],
    names: Sequence[str],
    name_to_class: Dict[str, str],
) -> int:
    lowered = line.lower()
    if not any(phrase in lowered for phrase in _function_phrases(str(prop.get("function_name") or ""))):
        return 0
    tokens: List[str] = []
    for name in names:
        tokens.extend(_entity_tokens(name, name_to_class.get(name)))
    quoted = [item.lower() for item in re.findall(r'"([^"]+)"', line)]
    if quoted:
        return sum(
            1
            for quote in quoted
            if any(quote == token or quote in token or token in quote for token in tokens)
        )
    return 1 if any(token in lowered for token in tokens if len(token) > 2) else 0


def load_planner_eval_stats(run_dir: Path) -> Dict[str, Any]:
    logs = sorted(Path(run_dir).glob("dataset/planner-log/planner-log-*.json"))
    if not logs:
        return {}
    path = logs[0]
    try:
        size = path.stat().st_size
        tail_size = min(size, 400_000)
        with path.open("rb") as handle:
            handle.seek(max(0, size - tail_size))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return {}
    idx = tail.rfind('"stats"')
    if idx < 0:
        return {}
    chunk = tail[idx:]
    result: Dict[str, Any] = {}
    percent = re.search(r'"task_percent_complete"\s*:\s*([0-9.]+)', chunk)
    success = re.search(r'"task_state_success"\s*:\s*([0-9.]+)', chunk)
    explanation = re.search(r'"task_explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
    if percent:
        result["task_percent_complete"] = float(percent.group(1))
    if success:
        result["task_state_success"] = float(success.group(1))
    if explanation:
        raw = explanation.group(1)
        result["task_explanation"] = (
            raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        )
    return result


def infer_criteria_from_eval(
    criteria: Sequence[str],
    propositions: Sequence[Dict[str, Any]],
    handle_to_name: Dict[str, str],
    name_to_class: Dict[str, str],
    stats: Dict[str, Any],
) -> Dict[str, bool]:
    scored: List[tuple[str, Dict[str, Any]]] = []
    used_criteria: set[str] = set()
    for prop in propositions:
        if not isinstance(prop, dict):
            continue
        criterion = _map_proposition_to_criterion(prop, handle_to_name, criteria)
        if criterion is None or criterion in used_criteria:
            continue
        used_criteria.add(criterion)
        scored.append((criterion, prop))

    inferred = {item: False for item in criteria}
    if not scored:
        return inferred

    percent = _float_or_none(stats.get("task_percent_complete"))
    success = _float_or_none(stats.get("task_state_success"))
    explanation = str(stats.get("task_explanation") or "")
    all_done = (success is not None and success >= 1.0) or (
        percent is not None and percent >= 0.999
    )
    if all_done:
        for criterion, _prop in scored:
            inferred[criterion] = True
        return inferred

    failed_lines = _failed_explanation_lines(explanation)
    if not failed_lines:
        return inferred

    for criterion, _prop in scored:
        inferred[criterion] = True
    used_props: set[int] = set()
    for line in failed_lines:
        ranked: List[tuple[int, int, str]] = []
        for index, (criterion, prop) in enumerate(scored):
            if index in used_props:
                continue
            names = [
                name
                for name in _prop_entity_names(prop, handle_to_name)
                if _looks_like_entity_name(name) or name in name_to_class
            ]
            if not names:
                names = _prop_entity_names(prop, handle_to_name)
            score = _line_match_score(line, prop, names, name_to_class)
            if score:
                ranked.append((score, index, criterion))
        if not ranked:
            continue
        ranked.sort(key=lambda item: (-item[0], item[1]))
        _score, index, criterion = ranked[0]
        inferred[criterion] = False
        used_props.add(index)
    return inferred


def parse_success_criteria_from_dataset(dataset: Dict[str, Any]) -> List[str]:
    episodes = dataset.get("episodes") or []
    if not episodes:
        return []
    episode = episodes[0]
    handle_to_name = _handle_to_name(episode)
    criteria: List[str] = []
    for prop in episode.get("evaluation_propositions") or []:
        if isinstance(prop, dict):
            criteria.append(_format_proposition(prop, handle_to_name))
    variant = (episode.get("info") or {}).get("variant_spec") or {}
    for line in variant.get("unscored_success_text") or []:
        text = str(line).strip()
        if text.startswith("- "):
            text = text[2:].strip()
        if text.startswith("Distractors ("):
            continue
        if text and text not in criteria:
            criteria.append(text)
    return criteria


def _load_episode_dataset(
    variant_folder: str,
    episodes_root: Path,
) -> tuple[Optional[Path], Dict[str, Any]]:
    parsed = parse_variant_folder(variant_folder)
    task_number = parsed["task_number"]
    if task_number is None:
        return None, {}
    dataset_path = episodes_root / f"task_{task_number}" / variant_folder / "dataset.json"
    if not dataset_path.is_file():
        return dataset_path, {}
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dataset_path, {}
    if not isinstance(dataset, dict):
        return dataset_path, {}
    return dataset_path, dataset


def load_episode_eval_bundle(
    variant_folder: str,
    episodes_root: Path,
) -> Dict[str, Any]:
    criteria_info = load_success_criteria(variant_folder, episodes_root)
    _dataset_path, dataset = _load_episode_dataset(variant_folder, episodes_root)
    episodes = dataset.get("episodes") or []
    episode = episodes[0] if episodes else {}
    propositions = [
        prop
        for prop in (episode.get("evaluation_propositions") or [])
        if isinstance(prop, dict)
    ]
    return {
        **criteria_info,
        "propositions": propositions,
        "handle_to_name": _handle_to_name(episode) if episode else {},
        "name_to_class": _name_to_class(episode) if episode else {},
    }


def load_success_criteria(
    variant_folder: str,
    episodes_root: Path,
) -> Dict[str, Any]:
    parsed = parse_variant_folder(variant_folder)
    task_number = parsed["task_number"]
    spec_path: Optional[Path] = None
    dataset_path: Optional[Path] = None
    spec_text = ""
    if task_number is not None:
        episode_dir = episodes_root / f"task_{task_number}" / variant_folder
        spec_path = episode_dir / "spec.md"
        dataset_path = episode_dir / "dataset.json"
        if spec_path.is_file():
            spec_text = spec_path.read_text(encoding="utf-8")
    summary = format_variant_summary(
        parsed["axis"],
        parsed["kind"],
        parse_uncertainty_from_spec(spec_text) if spec_text else [],
    )

    if spec_text:
        criteria = parse_success_criteria_from_spec(spec_text)
        if criteria:
            return {
                "criteria": criteria,
                "source": "spec",
                "spec_path": str(spec_path) if spec_path else None,
                "variant_summary": summary,
            }

    if dataset_path and dataset_path.is_file():
        try:
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dataset = {}
        criteria = parse_success_criteria_from_dataset(dataset)
        if criteria:
            return {
                "criteria": criteria,
                "source": "dataset",
                "spec_path": str(dataset_path),
                "variant_summary": summary,
            }

    return {
        "criteria": [],
        "source": None,
        "spec_path": None,
        "variant_summary": summary,
    }


def _float_or_none(raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def load_episode_metrics(csv_path: Path, episode_id: str) -> Dict[str, Any]:
    if not csv_path.is_file():
        return {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return {}
    match = None
    for row in reversed(rows):
        if str(row.get("episode_id", "")).strip() == episode_id:
            match = row
            break
    if match is None and rows:
        match = rows[-1]
    if not match:
        return {}
    percent = _float_or_none(match.get("task_percent_complete"))
    success = _float_or_none(match.get("task_state_success"))
    return {
        "episode_id": str(match.get("episode_id") or episode_id),
        "instruction": str(match.get("instruction") or "").strip(),
        "runtime": _float_or_none(match.get("runtime")),
        "task_percent_complete": percent,
        "task_state_success": success,
        "combined_time_used_s": _float_or_none(match.get("combined_time_used_s")),
        "combined_time_limit_s": _float_or_none(match.get("combined_time_limit_s")),
        "combined_time_limit_hit": _float_or_none(match.get("combined_time_limit_hit")),
        "replanning_count_0": _float_or_none(match.get("replanning_count_0")),
        "sim_step_count": _float_or_none(match.get("sim_step_count")),
    }


def normalize_reason_counts(
    selected: Sequence[str],
    raw: Any,
    allowed: set,
) -> Dict[str, int]:
    raw_counts = raw if isinstance(raw, dict) else {}
    counts: Dict[str, int] = {}
    for item in selected:
        if item not in allowed:
            continue
        value = raw_counts.get(item, 1)
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 1
        counts[item] = max(1, min(number, 99))
    return counts


def attach_issue_summary(run: Dict[str, Any], annotation: Any) -> None:
    if isinstance(annotation, dict) and annotation.get("updated_at"):
        failures = [
            item
            for item in (annotation.get("failure_reasons") or [])
            if item in FAILURE_IDS
        ]
        efficiency = [
            item
            for item in (annotation.get("efficiency_errors") or [])
            if item in EFFICIENCY_IDS
        ]
        run["failure_reasons"] = failures
        run["failure_reason_counts"] = normalize_reason_counts(
            failures,
            annotation.get("failure_reason_counts"),
            FAILURE_IDS,
        )
        run["efficiency_errors"] = efficiency
        run["efficiency_error_counts"] = normalize_reason_counts(
            efficiency,
            annotation.get("efficiency_error_counts"),
            EFFICIENCY_IDS,
        )
        return
    failures = [
        item for item in (run.get("auto_failure_reasons") or []) if item in FAILURE_IDS
    ]
    efficiency = [
        item
        for item in (run.get("auto_efficiency_errors") or [])
        if item in EFFICIENCY_IDS
    ]
    run["failure_reasons"] = failures
    run["failure_reason_counts"] = {item: 1 for item in failures}
    run["efficiency_errors"] = efficiency
    run["efficiency_error_counts"] = {item: 1 for item in efficiency}


def empty_annotation(criteria: Sequence[str]) -> Dict[str, Any]:
    return {
        "episode_id": None,
        "criteria": {item: False for item in criteria},
        "subtasks_done": 0,
        "subtasks_total": len(criteria),
        "failure_reasons": [],
        "failure_reason_counts": {},
        "failure_notes": "",
        "comments": "",
        "efficiency_errors": [],
        "efficiency_error_counts": {},
        "updated_at": None,
        "annotated": False,
        "auto_filled": False,
        "auto_error_filled": False,
    }


def annotation_is_complete(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    return bool(record.get("updated_at"))


def validate_annotation(
    payload: Dict[str, Any],
    criteria: Sequence[str],
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, dict):
        errors.append(
            {
                "field": "criteria",
                "message": "Select which success criteria were completed.",
            }
        )
        raw_criteria = {}

    checked: Dict[str, bool] = {}
    for item in criteria:
        checked[item] = bool(raw_criteria.get(item))
    unknown = [key for key in raw_criteria if key not in checked]
    if unknown:
        errors.append(
            {
                "field": "criteria",
                "message": "Annotation includes unknown success criteria.",
            }
        )

    done = sum(1 for value in checked.values() if value)
    total = len(criteria)

    failure_reasons = [
        item
        for item in (payload.get("failure_reasons") or [])
        if isinstance(item, str)
    ]
    efficiency_errors = [
        item
        for item in (payload.get("efficiency_errors") or [])
        if isinstance(item, str)
    ]
    failure_reason_counts = normalize_reason_counts(
        failure_reasons,
        payload.get("failure_reason_counts"),
        FAILURE_IDS,
    )
    efficiency_error_counts = normalize_reason_counts(
        efficiency_errors,
        payload.get("efficiency_error_counts"),
        EFFICIENCY_IDS,
    )
    comments = str(payload.get("comments") or payload.get("failure_notes") or "").strip()
    failure_notes = comments

    invalid_failure = [item for item in failure_reasons if item not in FAILURE_IDS]
    invalid_efficiency = [
        item for item in efficiency_errors if item not in EFFICIENCY_IDS
    ]
    if invalid_failure:
        errors.append(
            {
                "field": "failure_reasons",
                "message": "Choose a listed failure reason.",
            }
        )
    if invalid_efficiency:
        errors.append(
            {
                "field": "efficiency_errors",
                "message": "Choose a listed inefficiency.",
            }
        )

    if errors:
        return None, errors

    record = {
        "episode_id": payload.get("episode_id"),
        "criteria": checked,
        "subtasks_done": done,
        "subtasks_total": total,
        "failure_reasons": failure_reasons,
        "failure_reason_counts": failure_reason_counts,
        "failure_notes": failure_notes,
        "comments": comments,
        "efficiency_errors": efficiency_errors,
        "efficiency_error_counts": efficiency_error_counts,
        "updated_at": utc_now_iso(),
        "annotated": True,
    }
    return record, []


class RunCatalog:
    def __init__(
        self,
        logs_root: Path,
        episodes_root: Path,
        extra_sources: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.logs_root = logs_root.resolve()
        self.episodes_root = episodes_root.resolve()
        self.extra_sources = [dict(item) for item in (extra_sources or [])]
        self.annotations_path = self.logs_root / "annotations.json"
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._annotations: Dict[str, Any] = {"version": 1, "runs": {}}
        self.refresh()

    def image_roots(self) -> List[Path]:
        roots = [self.logs_root]
        for source in self.extra_sources:
            root = source.get("root")
            if root is None:
                continue
            resolved = Path(root).resolve()
            if resolved not in roots:
                roots.append(resolved)
            files_root = source.get("files_root")
            if files_root is None:
                continue
            resolved_files = Path(files_root).resolve()
            if resolved_files not in roots:
                roots.append(resolved_files)
        return roots

    def refresh(self) -> None:
        self._annotations = self._load_annotations()
        self._runs = self._discover_runs()

    def _load_annotations(self) -> Dict[str, Any]:
        if not self.annotations_path.is_file():
            return {"version": 1, "runs": {}}
        try:
            data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "runs": {}}
        if not isinstance(data, dict):
            return {"version": 1, "runs": {}}
        runs = data.get("runs")
        if not isinstance(runs, dict):
            runs = {}
        return {"version": 1, "runs": runs}

    def save_annotations(self) -> None:
        self.logs_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.annotations_path.with_suffix(".json.tmp")
        payload = json.dumps(self._annotations, indent=2, ensure_ascii=True)
        tmp_path.write_text(payload + "\n", encoding="utf-8")
        tmp_path.replace(self.annotations_path)

    def _discover_runs(self) -> Dict[str, Dict[str, Any]]:
        discovered: Dict[str, Dict[str, Any]] = {}
        sources: List[Dict[str, Any]] = [
            {
                "root": self.logs_root,
                "prefix": "",
                "planner": "react",
                "batch": None,
                "files_root": self.logs_root,
            }
        ]
        sources.extend(self.extra_sources)
        for source in sources:
            self._discover_source(discovered, source)
        for run in discovered.values():
            local_images = images_dir_for_trace(Path(run["trace_path"]))
            if local_images is not None:
                run["images_dir"] = str(local_images)
                run["image_count"] = len(list_step_images(local_images))
        by_variant: Dict[tuple, List[Dict[str, Any]]] = {}
        for run in discovered.values():
            if run.get("images_dir"):
                key = (
                    str(run.get("planner") or "react"),
                    str(run.get("variant_folder") or ""),
                )
                by_variant.setdefault(key, []).append(run)
        for run in discovered.values():
            if run.get("images_dir"):
                continue
            key = (
                str(run.get("planner") or "react"),
                str(run.get("variant_folder") or ""),
            )
            options = by_variant.get(key, [])
            if not options:
                continue
            options = sorted(
                options,
                key=lambda item: (
                    0 if item.get("batch") == "luna_high_states_images" else 1,
                    str(item.get("run_id") or ""),
                ),
            )
            donor = options[0]
            run["images_dir"] = donor["images_dir"]
            run["image_count"] = donor["image_count"]
        for run in discovered.values():
            try:
                parsed = parse_eval_trace_file(
                    run["trace_path"], str(run.get("planner") or "react")
                )
            except OSError:
                continue
            if run.get("subtasks_source") != "annotation":
                inferred = infer_criteria_from_trace(
                    run.get("criteria") or [],
                    final_object_states(parsed.get("steps") or []),
                )
                run["subtasks_done"] = sum(
                    1 for item in (run.get("criteria") or []) if inferred.get(item)
                )
                run["subtasks_source"] = "trace"
            criteria_success = bool(run.get("subtasks_total")) and int(
                run.get("subtasks_done") or 0
            ) >= int(run.get("subtasks_total") or 0)
            issues = infer_trace_issues(
                parsed.get("steps") or [],
                criteria_success,
            )
            run["auto_failure_reasons"] = issues["failure_reasons"]
            run["auto_efficiency_errors"] = issues["efficiency_errors"]
            attach_issue_summary(run, self._annotations["runs"].get(run["run_id"]))
        return discovered

    def _discover_source(
        self, discovered: Dict[str, Dict[str, Any]], source: Dict[str, Any]
    ) -> None:
        root = Path(source["root"]).resolve()
        if not root.is_dir():
            return
        prefix = str(source.get("prefix") or "")
        planner = str(source.get("planner") or "react")
        files_root = Path(source.get("files_root") or root).resolve()
        traces = sorted(root.glob("**/dataset/traces/**/trace-episode_*.txt"))
        for trace_path in traces:
            if _skip_trace_path(trace_path):
                continue
            run_dir = _run_dir_from_trace(trace_path)
            try:
                relative = run_dir.relative_to(root).as_posix()
            except ValueError:
                continue
            run_id = f"{prefix}{relative}"
            variant_folder = run_dir.name
            episode_id = _episode_id_from_trace(trace_path)
            existing = discovered.get(run_id)
            if existing:
                if trace_path.name.endswith("-0.txt"):
                    existing["trace_path"] = str(trace_path)
                    existing["episode_id"] = episode_id
                continue
            parsed = parse_variant_folder(variant_folder)
            criteria_info = load_success_criteria(variant_folder, self.episodes_root)
            metrics = load_episode_metrics(
                run_dir / "episode_result_log.csv",
                episode_id,
            )
            html_path = trace_path.with_suffix(".html")
            annotation = self._annotations["runs"].get(run_id)
            task_number = parsed["task_number"]
            if task_number is None:
                task_match = TASK_DIR_RE.match(run_dir.parent.name)
                if task_match:
                    task_number = int(task_match.group(1))
            criteria = criteria_info["criteria"]
            subtasks_total = len(criteria)
            stored_criteria = (
                annotation.get("criteria") if isinstance(annotation, dict) else None
            )
            if isinstance(stored_criteria, dict) and criteria:
                subtasks_done = sum(1 for item in criteria if stored_criteria.get(item))
                subtasks_source = "annotation"
            else:
                subtasks_done = 0
                subtasks_source = "pending"
            pddl_bundle = None
            if planner == "vlm_tamp_pddl":
                pddl_bundle = find_pddl_artifacts(run_dir, files_root)
                pddl_metrics = (pddl_bundle or {}).get("metrics") or {}
                if metrics.get("runtime") is None:
                    metrics["runtime"] = pddl_metrics.get("episode_runtime_sec")
                if metrics.get("task_percent_complete") is None:
                    metrics["task_percent_complete"] = pddl_metrics.get(
                        "task_percent_complete"
                    )
                if metrics.get("task_state_success") is None:
                    metrics["task_state_success"] = pddl_metrics.get(
                        "task_state_success"
                    )
                if metrics.get("sim_step_count") is None:
                    metrics["sim_step_count"] = pddl_metrics.get("sim_step_count")
            discovered[run_id] = {
                "run_id": run_id,
                "batch": source.get("batch") or batch_from_run_id(run_id),
                "planner": planner,
                "episode_id": metrics.get("episode_id") or episode_id,
                "variant_folder": variant_folder,
                "task_number": task_number,
                "axis": parsed["axis"],
                "kind": parsed["kind"],
                "variant_summary": criteria_info.get("variant_summary")
                or format_variant_summary(parsed["axis"], parsed["kind"]),
                "instruction": metrics.get("instruction") or "",
                "trace_path": str(trace_path),
                "html_path": str(html_path) if html_path.is_file() else None,
                "criteria": criteria,
                "criteria_source": criteria_info["source"],
                "runtime": metrics.get("runtime"),
                "task_percent_complete": metrics.get("task_percent_complete"),
                "task_state_success": metrics.get("task_state_success"),
                "combined_time_used_s": metrics.get("combined_time_used_s"),
                "combined_time_limit_s": metrics.get("combined_time_limit_s"),
                "combined_time_limit_hit": metrics.get("combined_time_limit_hit"),
                "replanning_count_0": metrics.get("replanning_count_0"),
                "sim_step_count": metrics.get("sim_step_count"),
                "subtasks_done": subtasks_done,
                "subtasks_total": subtasks_total,
                "subtasks_source": subtasks_source,
                "annotated": annotation_is_complete(annotation),
                "images_dir": None,
                "image_count": 0,
                "auto_failure_reasons": [],
                "auto_efficiency_errors": [],
                "failure_reasons": [],
                "failure_reason_counts": {},
                "efficiency_errors": [],
                "efficiency_error_counts": {},
                "pddl_bundle": pddl_bundle,
            }

    def list_runs(self) -> List[Dict[str, Any]]:
        def sort_key(run: Dict[str, Any]) -> tuple:
            task = run.get("task_number")
            axis = AXIS_ORDER.get(str(run.get("axis") or ""), 99)
            kind = KIND_ORDER.get(str(run.get("kind") or ""), 99)
            return (
                str(run.get("batch") or ""),
                task if isinstance(task, int) else 99,
                axis,
                kind,
                str(run.get("run_id") or ""),
            )

        return sorted(self._runs.values(), key=sort_key)

    def list_batches(self) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        tasks: Dict[str, set] = {}
        for run in self._runs.values():
            batch = str(run.get("batch") or "(root)")
            counts[batch] = counts.get(batch, 0) + 1
            task = run.get("task_number")
            if isinstance(task, int):
                tasks.setdefault(batch, set()).add(task)
        batches = []
        for name in sorted(counts, key=lambda item: (item == "(root)", item)):
            batches.append(
                {
                    "id": name,
                    "label": batch_label(name),
                    "run_count": counts[name],
                    "tasks": sorted(tasks.get(name, set())),
                }
            )
        return batches

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._runs.get(run_id)

    def annotation_for(
        self,
        run_id: str,
        criteria: Sequence[str],
        auto_criteria: Optional[Dict[str, bool]] = None,
        auto_issues: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stored = self._annotations["runs"].get(run_id)
        if not isinstance(stored, dict):
            merged = empty_annotation(criteria)
            if auto_criteria:
                for item in criteria:
                    if item in auto_criteria:
                        merged["criteria"][item] = bool(auto_criteria[item])
                merged["subtasks_done"] = sum(
                    1 for value in merged["criteria"].values() if value
                )
                merged["auto_filled"] = True
            issues = auto_issues or {}
            merged["efficiency_errors"] = [
                item
                for item in (issues.get("efficiency_errors") or [])
                if item in EFFICIENCY_IDS
            ]
            merged["efficiency_error_counts"] = normalize_reason_counts(
                merged["efficiency_errors"],
                {},
                EFFICIENCY_IDS,
            )
            merged["failure_reasons"] = [
                item
                for item in (issues.get("failure_reasons") or [])
                if item in FAILURE_IDS
            ]
            merged["failure_reason_counts"] = normalize_reason_counts(
                merged["failure_reasons"],
                {},
                FAILURE_IDS,
            )
            notes = str(issues.get("notes") or "").strip()
            if notes:
                merged["comments"] = notes
                merged["failure_notes"] = notes
                merged["auto_error_filled"] = True
            return merged
        merged = empty_annotation(criteria)
        stored_criteria = stored.get("criteria") or {}
        for item in criteria:
            if item in stored_criteria:
                merged["criteria"][item] = bool(stored_criteria[item])
            elif auto_criteria and item in auto_criteria:
                merged["criteria"][item] = bool(auto_criteria[item])
        merged["subtasks_done"] = sum(
            1 for value in merged["criteria"].values() if value
        )
        merged["subtasks_total"] = len(criteria)
        merged["episode_id"] = stored.get("episode_id")
        merged["failure_reasons"] = [
            item
            for item in (stored.get("failure_reasons") or [])
            if item in FAILURE_IDS
        ]
        merged["failure_reason_counts"] = normalize_reason_counts(
            merged["failure_reasons"],
            stored.get("failure_reason_counts"),
            FAILURE_IDS,
        )
        merged["failure_notes"] = str(stored.get("failure_notes") or "")
        merged["comments"] = str(
            stored.get("comments") or stored.get("failure_notes") or ""
        )
        merged["efficiency_errors"] = [
            item
            for item in (stored.get("efficiency_errors") or [])
            if item in EFFICIENCY_IDS
        ]
        merged["efficiency_error_counts"] = normalize_reason_counts(
            merged["efficiency_errors"],
            stored.get("efficiency_error_counts"),
            EFFICIENCY_IDS,
        )
        merged["updated_at"] = stored.get("updated_at")
        merged["annotated"] = annotation_is_complete(stored)
        merged["auto_filled"] = False
        merged["auto_error_filled"] = False
        return merged

    def save_run_annotation(
        self, run_id: str, payload: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
        run = self.get_run(run_id)
        if run is None:
            return None, [{"field": "run_id", "message": "Unknown run."}]
        record, errors = validate_annotation(payload, run["criteria"])
        if record is None:
            return None, errors
        record["episode_id"] = run["episode_id"]
        self._annotations["runs"][run_id] = {
            key: value
            for key, value in record.items()
            if key != "annotated"
        }
        self.save_annotations()
        run["annotated"] = True
        run["subtasks_done"] = record["subtasks_done"]
        run["subtasks_total"] = record["subtasks_total"]
        run["subtasks_source"] = "annotation"
        attach_issue_summary(run, record)
        return record, []
