#!/usr/bin/env python3
"""
HTML viewer for PARTNR trace logs.
Converts trace text files to an interactive HTML page.
"""

import argparse
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from habitat_llm.utils.episode_cost import (
    DEFAULT_MAX_COMBINED_TIME_S,
    DEFAULT_SIM_FREQ,
    build_episode_cost,
    combined_time_breakdown as compute_combined_time_breakdown,
    fmt_meters,
    fmt_seconds,
    fmt_steps,
)
from habitat_llm.utils.llm_usage import fmt_usd


_PDDL_FAILURE_MARKERS = (
    "unexpected failure",
    "action failed",
    "failed",
    "error",
    "✗ action result: failed",
    "action result: failed",
)
_PDDL_SUCCESS_MARKERS = (
    "successful execution",
    "✓ action result: success",
    "action result: success",
    "success",
)


def _expanded_action_counts(action_name: str) -> Dict[str, int]:
    """Expand composite actions into the low-level count summary we display."""
    if action_name == "Rearrange":
        return {
            "Navigate": 2,
            "Pick": 1,
            "Place": 1,
        }
    return {action_name: 1}


def _normalize_action_counts(action_counts: Dict[str, int]) -> Dict[str, int]:
    """Normalize action counts so Rearrange always renders as low-level skills."""
    normalized_counts: Dict[str, int] = {}
    for action_name, count in action_counts.items():
        if not isinstance(count, int):
            continue
        for normalized_name, increment in _expanded_action_counts(action_name).items():
            normalized_counts[normalized_name] = (
                normalized_counts.get(normalized_name, 0) + (increment * count)
            )
    return normalized_counts


def _parse_trace_file_react(content: str) -> Dict[str, Any]:
    """Parse legacy ReAct-style trace format."""
    # Extract task (first line)
    task = ""
    lines = content.split('\n')
    if lines and lines[0].startswith('Task:'):
        task = lines[0].replace('Task:', '').strip()

    steps = []

    # Find all actions first - this ensures we don't miss any
    action_pattern = r'([A-Z][a-z]+)\[([^\]]*)\]'
    action_matches = list(re.finditer(action_pattern, content))

    for i, action_match in enumerate(action_matches):
        action_start = action_match.start()
        action_end = action_match.end()
        action_name = action_match.group(1)
        action_args = action_match.group(2)

        step = {'action': action_name, 'args': action_args}

        # Find thought before this action (look backwards up to 2000 chars or to previous action)
        lookback_start = action_matches[i-1].end() if i > 0 else 0
        lookback_section = content[lookback_start:action_start]
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n(?:[A-Z][a-z]+\[|Assigned!|$))', lookback_section, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            thought = re.sub(r'^Thought:\s*', '', thought).strip()
            step['thought'] = thought

        # Find result after this action (look forward to next action or end)
        forward_end = action_matches[i+1].start() if i < len(action_matches) - 1 else len(content)
        forward_section = content[action_end:forward_end]

        result_match = re.search(r'Assigned!Result:\s*(.*?)(?=\n(?:Objects:|Thought:|[A-Z][a-z]+\[|$))', forward_section, re.DOTALL)
        if result_match:
            result = result_match.group(1).strip()
            step['result'] = result
            # Done action is always successful
            if action_name == "Done":
                step['success'] = True
            else:
                step['success'] = 'Successful' in result or 'success' in result.lower()
        elif action_name == "Done":
            # Done action without result is still successful
            step['success'] = True

        # Find objects after result
        objects_match = re.search(r'Objects:\s*(.*?)(?=\n(?:Thought:|[A-Z][a-z]+\[|$))', forward_section, re.DOTALL)
        if objects_match:
            objects_text = objects_match.group(1).strip()
            if objects_text and objects_text != 'No objects found yet':
                step['objects'] = objects_text

        steps.append(step)

    return {
        'task': task,
        'steps': steps,
        'total_steps': len(steps)
    }


def _infer_pddl_step_success(block_lines: List[str], action_name: str) -> bool:
    block_text = "\n".join(block_lines).lower()
    has_failure = any(marker in block_text for marker in _PDDL_FAILURE_MARKERS)
    has_success = any(marker in block_text for marker in _PDDL_SUCCESS_MARKERS)
    if has_failure:
        return False
    if has_success:
        return True
    return action_name.lower() == "done"


def _parse_trace_file_pddl(content: str) -> Dict[str, Any]:
    """Parse PDDL-style trace format with Action/Observation blocks."""
    task = ""
    lines = content.split('\n')
    if lines and lines[0].startswith('Task:'):
        task = lines[0].replace('Task:', '').strip()

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

        if line.startswith("Subgoal:"):
            current_subgoal = line.replace("Subgoal:", "", 1).strip()
            i += 1
            continue

        action_match = re.match(r"^Action:\s*([A-Za-z_]+)\[(.*)\]\s*$", line)
        if not action_match:
            action_match = re.match(
                r"^High-level special-case:\s*([A-Za-z_]+)\[(.*)\]\s*$",
                line,
            )
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

        j = i + 1
        block_lines: List[str] = []
        result_lines: List[str] = []
        while j < len(lines):
            nxt = lines[j].strip()
            if nxt.startswith("Action:") or nxt.startswith("Subgoal:"):
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
        step["success"] = _infer_pddl_step_success(block_lines, action_name)
        # Keep raw per-action block so the HTML action card can show detailed logs.
        if block_lines:
            step["pddl_log"] = "\n".join(block_lines)
        steps.append(step)
        i = j

    return {
        "task": task,
        "steps": steps,
        "total_steps": len(steps),
    }


def parse_trace_file(trace_file: str, is_pddl_run: bool = False) -> Dict[str, Any]:
    """Parse a trace text file into structured data."""
    with open(trace_file, "r") as f:
        content = f.read()
    if is_pddl_run:
        parsed = _parse_trace_file_pddl(content)
    else:
        parsed = _parse_trace_file_react(content)
    parsed.update(_explore_metrics_from_text(content))
    return parsed


def parse_trace_file(trace_file: str, is_pddl_run: bool = False) -> Dict[str, Any]:
    """Parse a trace text file into structured data."""
    with open(trace_file, "r") as f:
        content = f.read()
    if is_pddl_run:
        parsed = _parse_trace_file_pddl(content)
    else:
        parsed = _parse_trace_file_react(content)
    parsed.update(_explore_metrics_from_text(content))
    return parsed


_EXPLORE_TOUR_RE = re.compile(
    r"\[fast_explore\] approx tour[^:]*:\s*\d+ furniture,\s*"
    r"([\d.]+) m,\s*~(\d+) sim steps,\s*~([\d.]+) s sim time"
)


def _episode_filename_from_trace_path(path: str) -> Optional[str]:
    stem = Path(path).stem
    if not stem.startswith("trace-"):
        return None
    rest = stem[len("trace-") :]
    if "-" not in rest:
        return rest
    return rest.rsplit("-", 1)[0]


def _explore_metrics_from_text(content: str) -> Dict[str, Any]:
    tours = _EXPLORE_TOUR_RE.findall(content or "")
    if not tours:
        return {}
    return {
        "explore_approx_meters": sum(float(row[0]) for row in tours),
        "explore_approx_sim_steps": sum(int(row[1]) for row in tours),
        "explore_approx_sim_time_s": sum(float(row[2]) for row in tours),
    }


def _action_sim_steps_from_planner_log(steps: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    prev_sim = None
    for step in steps:
        sim = step.get("sim_step_count")
        names = []
        high_level = step.get("high_level_actions") or {}
        if isinstance(high_level, dict):
            for value in high_level.values():
                if isinstance(value, (list, tuple)) and value:
                    names.append(str(value[0]))
        if prev_sim is not None and sim is not None:
            try:
                delta = int(sim) - int(prev_sim)
            except (TypeError, ValueError):
                delta = 0
            if delta > 0:
                for name in names or ["Unknown"]:
                    if name == "Explore":
                        continue
                    counts[name] = counts.get(name, 0) + delta
        prev_sim = sim
    return counts


def _metrics_from_planner_log(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list) or not steps:
        return {}
    last = steps[-1] if isinstance(steps[-1], dict) else {}
    metrics: Dict[str, Any] = {}
    cost = last.get("cost_metrics")
    if isinstance(cost, dict):
        metrics.update(cost)
    if last.get("sim_step_count") is not None:
        metrics["sim_step_count"] = last.get("sim_step_count")
    if last.get("replanning_count") is not None:
        metrics["llm_requests"] = last.get("replanning_count")
    for key in (
        "llm_model",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "total_tokens",
        "llm_usd",
        "llm_usd_source",
        "llm_reasoning_effort",
    ):
        if key in last:
            metrics[key] = last[key]
        elif isinstance(cost, dict) and key in cost:
            metrics[key] = cost[key]
    stats = last.get("stats") if isinstance(last.get("stats"), dict) else {}
    if "task_percent_complete" in stats:
        metrics["task_percent_complete"] = stats["task_percent_complete"]
    if "task_state_success" in stats:
        metrics["task_state_success"] = stats["task_state_success"]
    action_sim_steps = _action_sim_steps_from_planner_log(steps)
    if action_sim_steps:
        metrics["action_sim_steps"] = action_sim_steps
    return metrics


def _metrics_from_episode_csv(csv_path: Path, episode_filename: str) -> Dict[str, Any]:
    if not csv_path.is_file():
        return {}

    episode_id = None
    parts = episode_filename.split("_")
    if len(parts) >= 2 and parts[0] == "episode":
        episode_id = parts[1]
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return {}
    match = None
    if episode_id is not None:
        for row in reversed(rows):
            if str(row.get("episode_id", "")).strip() == str(episode_id):
                match = row
                break
    if match is None and rows:
        match = rows[-1]
    if not match:
        return {}
    out: Dict[str, Any] = {}
    mapping = {
        "runtime": "runtime",
        "llm_planning_time_s": "llm_planning_time_s",
        "explore_approx_sim_steps": "explore_approx_sim_steps",
        "explore_approx_sim_time_s": "explore_approx_sim_time_s",
        "sim_step_count": "sim_step_count",
        "combined_time_used_s": "combined_time_used_s",
        "combined_time_limit_s": "combined_time_limit_s",
        "combined_time_limit_hit": "combined_time_limit_hit",
        "task_percent_complete": "task_percent_complete",
        "task_state_success": "task_state_success",
    }
    for src, dest in mapping.items():
        raw = match.get(src)
        if raw in (None, ""):
            continue
        try:
            if src == "combined_time_limit_hit":
                out[dest] = float(raw) != 0.0
            elif src in ("explore_approx_sim_steps", "sim_step_count"):
                out[dest] = int(float(raw))
            else:
                out[dest] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _discovered_metrics(output_file: str, trace_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fill LLM / explore / sim metrics from planner-log, CSV, or the trace text."""
    metrics: Dict[str, Any] = {}
    html_path = Path(output_file).resolve()
    episode = _episode_filename_from_trace_path(str(html_path))
    txt_path = html_path.with_suffix(".txt")
    if txt_path.is_file():
        try:
            metrics.update(_explore_metrics_from_text(txt_path.read_text(encoding="utf-8")))
        except OSError:
            pass
    if isinstance(trace_data, dict):
        for key in (
            "explore_approx_meters",
            "explore_approx_sim_steps",
            "explore_approx_sim_time_s",
        ):
            if key in trace_data and trace_data[key] is not None:
                metrics[key] = trace_data[key]

    # traces/<uid>/file.html -> dataset/
    dataset_dir = html_path.parent.parent.parent if html_path.parent.name.isdigit() else html_path.parent.parent
    if episode:
        planner_metrics = _metrics_from_planner_log(
            dataset_dir / "planner-log" / f"planner-log-{episode}.json"
        )
        metrics.update({k: v for k, v in planner_metrics.items() if v is not None})
        csv_metrics = _metrics_from_episode_csv(
            dataset_dir.parent / "episode_result_log.csv",
            episode,
        )
        for key, value in csv_metrics.items():
            if key not in metrics or metrics[key] in (None, 0, 0.0):
                metrics[key] = value
    return metrics


def _fill_if_none(current: Any, discovered: Dict[str, Any], key: str) -> Any:
    if current is not None:
        return current
    return discovered.get(key)


def _format_task_score(value: Optional[float]) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    number = float(value)
    if 0.0 <= number <= 1.0:
        return f"{number:.3f} ({number:.1%})"
    return f"{number:.3f}"


def _resolve_combined_breakdown(
    provided: Optional[Dict[str, Any]],
    *,
    llm_planning_time_s: Any,
    action_sim_steps: Any,
    explore_approx_sim_time_s: Any,
    sim_freq: float,
    combined_time_used_s: Any,
    combined_time_limit_s: Any,
    combined_time_limit_hit: Any,
) -> Dict[str, Any]:
    limit_arg = (
        combined_time_limit_s
        if combined_time_limit_s is not None
        else DEFAULT_MAX_COMBINED_TIME_S
    )
    computed = compute_combined_time_breakdown(
        llm_planning_time_s=llm_planning_time_s,
        action_sim_steps=action_sim_steps,
        explore_approx_sim_time_s=explore_approx_sim_time_s,
        sim_freq=sim_freq,
        limit_s=limit_arg,
    )
    if isinstance(provided, dict):
        for key in ("llm_s", "action_sim_s", "explore_approx_s"):
            raw = provided.get(key)
            if raw is not None:
                try:
                    computed[key] = float(raw)
                except (TypeError, ValueError):
                    pass
        computed["used_s"] = (
            computed["llm_s"] + computed["action_sim_s"] + computed["explore_approx_s"]
        )
        raw_used = provided.get("used_s")
        if raw_used is not None:
            try:
                computed["used_s"] = float(raw_used)
            except (TypeError, ValueError):
                pass
    if combined_time_used_s is not None:
        try:
            computed["used_s"] = float(combined_time_used_s)
        except (TypeError, ValueError):
            pass
    if combined_time_limit_s is not None:
        try:
            computed["limit_s"] = float(combined_time_limit_s)
        except (TypeError, ValueError):
            pass
    if combined_time_limit_hit is not None:
        computed["exceeded"] = bool(combined_time_limit_hit)
    else:
        limit = computed["limit_s"]
        computed["exceeded"] = limit > 0 and computed["used_s"] >= limit
    return computed


def collect_step_camera_images(output_file: str) -> List[str]:
    """Relative paths of per-replan robot camera PNGs next to the HTML file."""
    html_dir = os.path.dirname(os.path.abspath(output_file))
    images_dir = os.path.join(html_dir, "images")
    if not os.path.isdir(images_dir):
        return []
    names = [
        name
        for name in os.listdir(images_dir)
        if re.fullmatch(r"step_\d+\.png", name)
    ]
    names.sort()
    return [os.path.join("images", name) for name in names]


def generate_html(
    trace_data: Dict[str, Any],
    output_file: str,
    action_counts: Dict[str, int] = None,
    action_sim_steps: Dict[str, int] = None,
    runtime: float = None,
    llm_requests: Dict[int, int] = None,
    is_pddl_run: bool = False,
    planning_tree_image: Optional[str] = None,
    llm_planning_time_s: float = None,
    explore_approx_sim_steps: int = None,
    explore_approx_sim_time_s: float = None,
    sim_freq: float = DEFAULT_SIM_FREQ,
    explore_approx_meters: float = None,
    sim_step_count: int = None,
    task_percent_complete: float = None,
    task_state_success: float = None,
    combined_time_used_s: float = None,
    combined_time_limit_s: float = None,
    combined_time_limit_hit: bool = None,
    combined_time_breakdown: Dict[str, Any] = None,
    llm_model: str = None,
    prompt_tokens: int = None,
    completion_tokens: int = None,
    cached_tokens: int = None,
    llm_usd: float = None,
    llm_usd_source: str = None,
    llm_reasoning_effort: str = None,
) -> None:
    """Generate an HTML file from parsed trace data.

    Args:
        trace_data: Parsed trace data dictionary
        output_file: Path to output HTML file
        action_counts: Optional dict mapping action names to counts (from evaluation runner)
        action_sim_steps: Optional dict mapping action names to simulation step counts (from evaluation runner)
        runtime: Optional total wall-clock runtime in seconds (from evaluation runner)
        llm_requests: Optional dict mapping agent IDs to LLM request counts (replanning_count from evaluation runner)
        llm_planning_time_s: Optional wall-clock LLM/VLM planning time in seconds
        explore_approx_sim_steps: Optional Explore sim steps (first walked + later geodesic x ratio)
        explore_approx_sim_time_s: Optional approximate Explore sim time in seconds
        sim_freq: Simulator control frequency used to convert exact steps to sim time
        explore_approx_meters: Optional geodesic tour length in meters
        sim_step_count: Optional env-reported exact sim step count
        task_percent_complete: Optional Habitat task percent_complete
        task_state_success: Optional Habitat task_state_success
        combined_time_used_s: Optional combined budget used (LLM wall + action sim + explore approx)
        combined_time_limit_s: Optional combined budget limit (default 600s / 10 min)
        combined_time_limit_hit: Optional flag that the episode stopped on combined budget
        combined_time_breakdown: Optional dict with llm_s, action_sim_s, explore_approx_s
        llm_model: Optional OpenAI / VLM model name
        prompt_tokens: Optional prompt token total from the API
        completion_tokens: Optional completion token total from the API
        cached_tokens: Optional cached prompt token total
        llm_usd: Optional estimated USD cost for those tokens
        llm_usd_source: api or estimated
    """

    task = html.escape(str(trace_data.get("task", "No task")))
    steps = trace_data.get("steps", [])
    total_steps = trace_data.get("total_steps", 0)

    if not isinstance(steps, list):
        steps = []
    if not isinstance(total_steps, (int, float)) or total_steps < 0:
        total_steps = len(steps)

    file_path = html.escape(os.path.abspath(str(output_file)))

    discovered = _discovered_metrics(output_file, trace_data)
    action_counts = _fill_if_none(action_counts, discovered, "action_counts")
    action_sim_steps = _fill_if_none(action_sim_steps, discovered, "action_sim_steps")
    runtime = _fill_if_none(runtime, discovered, "runtime")
    llm_requests = _fill_if_none(llm_requests, discovered, "llm_requests")
    llm_planning_time_s = _fill_if_none(
        llm_planning_time_s, discovered, "llm_planning_time_s"
    )
    explore_approx_sim_steps = _fill_if_none(
        explore_approx_sim_steps, discovered, "explore_approx_sim_steps"
    )
    explore_approx_sim_time_s = _fill_if_none(
        explore_approx_sim_time_s, discovered, "explore_approx_sim_time_s"
    )
    explore_approx_meters = _fill_if_none(
        explore_approx_meters, discovered, "explore_approx_meters"
    )
    sim_step_count = _fill_if_none(sim_step_count, discovered, "sim_step_count")
    task_percent_complete = _fill_if_none(
        task_percent_complete, discovered, "task_percent_complete"
    )
    task_state_success = _fill_if_none(
        task_state_success, discovered, "task_state_success"
    )
    combined_time_used_s = _fill_if_none(
        combined_time_used_s, discovered, "combined_time_used_s"
    )
    combined_time_limit_s = _fill_if_none(
        combined_time_limit_s, discovered, "combined_time_limit_s"
    )
    combined_time_limit_hit = _fill_if_none(
        combined_time_limit_hit, discovered, "combined_time_limit_hit"
    )
    llm_model = _fill_if_none(llm_model, discovered, "llm_model")
    prompt_tokens = _fill_if_none(prompt_tokens, discovered, "prompt_tokens")
    completion_tokens = _fill_if_none(completion_tokens, discovered, "completion_tokens")
    cached_tokens = _fill_if_none(cached_tokens, discovered, "cached_tokens")
    llm_usd = _fill_if_none(llm_usd, discovered, "llm_usd")
    llm_usd_source = _fill_if_none(llm_usd_source, discovered, "llm_usd_source")
    llm_reasoning_effort = _fill_if_none(
        llm_reasoning_effort, discovered, "llm_reasoning_effort"
    )

    successes = sum(1 for s in steps if isinstance(s, dict) and s.get("success", False))
    success_rate = (successes / total_steps * 100) if total_steps > 0 else 0

    freq = sim_freq if isinstance(sim_freq, (int, float)) and sim_freq > 0 else DEFAULT_SIM_FREQ

    cost = build_episode_cost(
        action_sim_steps=action_sim_steps,
        action_counts=action_counts,
        llm_planning_time_s=llm_planning_time_s,
        llm_requests=llm_requests,
        explore_approx_sim_steps=explore_approx_sim_steps,
        explore_approx_sim_time_s=explore_approx_sim_time_s,
        explore_approx_meters=explore_approx_meters,
        runtime=runtime,
        sim_step_count=sim_step_count,
        sim_freq=freq,
        task_percent_complete=task_percent_complete,
        task_state_success=task_state_success,
    )
    budget = _resolve_combined_breakdown(
        combined_time_breakdown,
        llm_planning_time_s=llm_planning_time_s,
        action_sim_steps=action_sim_steps,
        explore_approx_sim_time_s=explore_approx_sim_time_s,
        sim_freq=freq,
        combined_time_used_s=combined_time_used_s,
        combined_time_limit_s=combined_time_limit_s,
        combined_time_limit_hit=combined_time_limit_hit,
    )

    llm_requests_str = "N/A"
    total_llm_requests = cost.get("llm_requests") or 0
    if llm_requests is not None and isinstance(llm_requests, dict):
        if total_llm_requests > 0:
            if len(llm_requests) == 1:
                llm_requests_str = str(total_llm_requests)
            else:
                parts = [f"Agent {aid}: {count}" for aid, count in sorted(llm_requests.items())]
                llm_requests_str = f"{total_llm_requests} ({', '.join(parts)})"
    elif total_llm_requests > 0:
        llm_requests_str = str(total_llm_requests)

    trace_action_counts = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        action_name = step.get("action", "Unknown")
        for normalized_name, increment in _expanded_action_counts(action_name).items():
            trace_action_counts[normalized_name] = (
                trace_action_counts.get(normalized_name, 0) + increment
            )

    if action_counts is None or not isinstance(action_counts, dict):
        action_counts = trace_action_counts
    else:
        action_counts = _normalize_action_counts(action_counts)

    action_counts_dict = action_counts if isinstance(action_counts, dict) else {}
    action_sim_steps_dict = action_sim_steps if isinstance(action_sim_steps, dict) else {}
    action_count_names_sorted = sorted(
        set(action_counts_dict.keys()) | set(trace_action_counts.keys())
    )
    display_action_counts = {}
    for action_name in action_count_names_sorted:
        info_count = action_counts_dict.get(action_name, 0)
        trace_count = trace_action_counts.get(action_name, 0)
        if is_pddl_run and info_count == 0 and trace_count > 0:
            display_action_counts[action_name] = trace_count
        else:
            display_action_counts[action_name] = info_count

    display_total_actions = sum(display_action_counts.values())

    total_sim_steps = cost.get("exact_sim_steps") or 0
    exact_sim_time_str = fmt_seconds(cost.get("exact_sim_time_s"))
    planning_time_str = fmt_seconds(cost.get("llm_planning_time_s"))
    explore_approx_steps_str = fmt_steps(cost.get("explore_approx_sim_steps"))
    explore_approx_time_str = fmt_seconds(cost.get("explore_approx_sim_time_s"))
    explore_meters_str = fmt_meters(cost.get("explore_approx_meters"))
    llm_avg_str = fmt_seconds(cost.get("llm_avg_s"))
    runtime_str = fmt_seconds(cost.get("runtime_s"))
    non_llm_wall_str = fmt_seconds(cost.get("non_llm_wall_s"))
    env_steps_str = fmt_steps(cost.get("sim_step_count"))
    task_pct_str = _format_task_score(cost.get("task_percent_complete"))
    task_ok_str = _format_task_score(cost.get("task_state_success"))
    model_str = html.escape(str(llm_model).strip()) if llm_model else "N/A"
    effort_str = (
        html.escape(str(llm_reasoning_effort).strip())
        if llm_reasoning_effort
        else "N/A"
    )

    def _fmt_tokens(value: Any) -> str:
        if value is None or not isinstance(value, (int, float)):
            return "N/A"
        return f"{int(value):,}"

    prompt_str = _fmt_tokens(prompt_tokens)
    completion_str = _fmt_tokens(completion_tokens)
    cached_str = _fmt_tokens(cached_tokens)
    if llm_usd is not None:
        usd_str = fmt_usd(llm_usd)
        if llm_usd_source == "estimated":
            usd_str = f"{usd_str} est."
    elif prompt_tokens or completion_tokens:
        usd_str = "N/A (unknown model price)"
    else:
        usd_str = "N/A (usage not recorded)"

    used_s = float(budget.get("used_s") or 0.0)
    limit_s = float(budget.get("limit_s") or 0.0)
    remaining_s = max(0.0, limit_s - used_s) if limit_s > 0 else None
    budget_pct = min(100.0, (used_s / limit_s) * 100.0) if limit_s > 0 else 0.0
    budget_over = bool(budget.get("exceeded"))
    llm_s = float(budget.get("llm_s") or 0.0)
    action_s = float(budget.get("action_sim_s") or 0.0)
    explore_s = float(budget.get("explore_approx_s") or 0.0)

    def _share(part: float) -> float:
        if used_s <= 0:
            return 0.0
        return max(0.0, min(100.0, (part / used_s) * 100.0))

    banner_html = ""
    if budget_over:
        banner_html = (
            '<div class="budget-banner">'
            "Episode stopped because combined budget was exceeded."
            "</div>"
        )

    used_str = fmt_seconds(used_s)
    limit_str = fmt_seconds(limit_s) if limit_s > 0 else "disabled"
    remaining_str = fmt_seconds(remaining_s)
    fill_class = "over" if budget_over else ""

    exact_table_rows = ""
    explore_count = display_action_counts.get("Explore", 0)
    explore_steps_val = cost.get("explore_approx_sim_steps")
    if explore_steps_val is None:
        explore_steps_val = action_sim_steps_dict.get("Explore", 0)
    try:
        explore_steps_val = int(explore_steps_val or 0)
    except (TypeError, ValueError):
        explore_steps_val = 0
    action_names = sorted(
        name
        for name in set(action_count_names_sorted)
        | set(action_sim_steps_dict.keys())
        | ({"Explore"} if explore_count or explore_steps_val else set())
    )
    if not isinstance(trace_action_counts, dict):
        trace_action_counts = {}
    for action_name in action_names:
        info_count = action_counts_dict.get(action_name, 0)
        trace_count = trace_action_counts.get(action_name, 0)
        display_count = display_action_counts.get(action_name, 0)
        count_color = "#ef4444" if info_count != trace_count else "#1f2937"
        if action_name == "Explore":
            sim_steps_val = explore_steps_val
        else:
            sim_steps_val = action_sim_steps_dict.get(action_name, 0)
            try:
                sim_steps_val = int(sim_steps_val)
            except (TypeError, ValueError):
                sim_steps_val = 0
        action_time_s = sim_steps_val / freq if freq else 0.0
        exact_table_rows += f"""
                    <tr>
                        <td>{html.escape(str(action_name))}</td>
                        <td class="num">
                            <div style="font-weight: 700; color: {count_color};" title="Trace file: {trace_count}, Info: {info_count}">{display_count}</div>
                        </td>
                        <td class="num">{sim_steps_val}</td>
                        <td class="num">{action_time_s:.2f}s sim</td>
                    </tr>
"""
    if not exact_table_rows:
        exact_table_rows = (
            '<tr><td colspan="4" class="empty">No actions recorded.</td></tr>'
        )

    planning_tree_html = ""
    if is_pddl_run and planning_tree_image and os.path.isfile(planning_tree_image):
        rel_tree = os.path.relpath(
            os.path.abspath(planning_tree_image),
            start=os.path.dirname(os.path.abspath(output_file)),
        )
        rel_tree_esc = html.escape(rel_tree)
        planning_tree_html = f"""
            <section class="panel">
                <h2>Planning Tree</h2>
                <div class="tree-wrap">
                    <img src="{rel_tree_esc}" alt="Planning tree" />
                </div>
            </section>
"""

    step_images = collect_step_camera_images(output_file)
    steps_html = ""
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_num = idx + 1
        thought = step.get("thought", "No thought")
        action = step.get("action", "Unknown")
        args = step.get("args", "")
        result = step.get("result", "No result")
        objects = step.get("objects", "")
        success = step.get("success", False)
        camera_src = step.get("image") or (
            step_images[idx] if idx < len(step_images) else ""
        )

        thought_esc = html.escape(thought)
        result_esc = html.escape(result)
        objects_esc = html.escape(objects) if objects else ""
        pddl_log = step.get("pddl_log", "") if is_pddl_run else ""
        pddl_log_esc = html.escape(pddl_log) if pddl_log else ""

        status_class = "status-success" if success else "status-failure"
        status_text = "Success" if success else "Failed"
        step_class = "success" if success else "failure"

        action_display = f"{action}[{args}]" if args else action
        action_display_esc = html.escape(action_display)

        extra_sections = ""
        camera_thumb_html = ""
        if camera_src:
            camera_src_esc = html.escape(str(camera_src))
            camera_thumb_html = (
                f'<img class="step-cam-thumb" src="{camera_src_esc}" '
                'alt="Robot camera" />'
            )
            extra_sections += f"""
                        <div class="section">
                            <div class="section-label">Robot camera</div>
                            <div class="section-content">
                                <img class="step-cam" src="{camera_src_esc}" alt="Robot overhead camera" />
                            </div>
                        </div>
"""
        if objects:
            extra_sections += f"""
                        <div class="section">
                            <div class="section-label">Objects Discovered</div>
                            <div class="section-content">
                                <div class="objects">{objects_esc}</div>
                            </div>
                        </div>
"""
        if pddl_log:
            extra_sections += f"""
                        <div class="section">
                            <div class="section-label">Execution Log</div>
                            <div class="section-content">
                                <div class="objects">{pddl_log_esc}</div>
                            </div>
                        </div>
"""

        result_preview = result.replace("\n", " ").strip()
        if len(result_preview) > 72:
            result_preview = result_preview[:69] + "..."
        result_preview_esc = html.escape(result_preview)

        steps_html += f"""
                <div class="step {step_class}" data-status="{step_class}">
                    <div class="step-action-row" onclick="toggleStep(this.parentElement)">
                        <span class="step-number">{step_num}</span>
                        {camera_thumb_html}
                        <span class="step-action">{action_display_esc}</span>
                        <span class="step-preview">{result_preview_esc}</span>
                        <span class="step-status {status_class}">{status_text}</span>
                        <span class="step-chevron">&#9660;</span>
                    </div>
                    <div class="step-expand">
                        <div class="section">
                            <div class="section-label">Thought</div>
                            <div class="section-content">{thought_esc}</div>
                        </div>
                        <div class="section">
                            <div class="section-label">Result</div>
                            <div class="section-content">{result_esc}</div>
                        </div>
                        {extra_sections}
                    </div>
                </div>
"""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trace Log Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #5b4fc9;
            min-height: 100vh;
            padding: 8px;
            color: #111827;
            font-size: 13px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #f4f3f8;
            border-radius: 10px;
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 12px;
        }}

        .header-top {{
            display: flex;
            gap: 12px;
            align-items: baseline;
            justify-content: space-between;
            flex-wrap: wrap;
        }}

        .header h1 {{
            font-size: 14px;
            font-weight: 650;
            line-height: 1.3;
            flex: 1;
            min-width: 220px;
        }}

        .header-path {{
            font-size: 10px;
            opacity: 0.9;
            font-family: ui-monospace, Menlo, monospace;
            word-break: break-all;
            line-height: 1.35;
            margin-top: 4px;
        }}

        .outcome-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 6px;
        }}

        .chip {{
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 12px;
            font-variant-numeric: tabular-nums;
        }}

        .chip-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            opacity: 0.8;
            margin-right: 4px;
        }}

        .chip-value {{
            font-weight: 700;
        }}

        .budget-banner {{
            margin-top: 6px;
            background: #7f1d1d;
            color: #fff;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }}

        .content {{
            padding: 8px 10px 12px;
            display: grid;
            gap: 8px;
        }}

        .panel {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 8px 10px;
        }}

        .panel h2 {{
            font-size: 12px;
            color: #4c1d95;
            margin-bottom: 4px;
        }}

        .panel-sub {{
            font-size: 11px;
            color: #6b7280;
            font-weight: 400;
            margin-left: 6px;
        }}

        .hint {{
            font-size: 11px;
            color: #6b7280;
            line-height: 1.3;
            margin-bottom: 6px;
        }}

        .dash {{
            display: grid;
            grid-template-columns: minmax(280px, 1.1fr) minmax(320px, 1.3fr);
            gap: 8px;
        }}

        @media (max-width: 840px) {{
            .dash {{ grid-template-columns: 1fr; }}
        }}

        .kv {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px 10px;
            font-variant-numeric: tabular-nums;
            font-size: 12px;
        }}

        .kv span {{
            white-space: nowrap;
        }}

        .kv b {{
            font-weight: 700;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 4px;
            margin-bottom: 6px;
        }}

        .metric {{
            background: #f5f3ff;
            border-radius: 4px;
            padding: 4px 6px;
        }}

        .metric-label {{
            font-size: 10px;
            color: #6d28d9;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        .metric-value {{
            font-size: 13px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: #111827;
        }}

        .budget-head {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 8px;
            margin-bottom: 3px;
            font-variant-numeric: tabular-nums;
            font-size: 12px;
        }}

        .budget-used {{
            font-size: 14px;
            font-weight: 700;
        }}

        .budget-bar {{
            height: 6px;
            background: #ede9fe;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 4px;
        }}

        .budget-bar-fill {{
            height: 100%;
            width: {budget_pct:.2f}%;
            background: #667eea;
        }}

        .budget-bar-fill.over {{
            background: #dc2626;
        }}

        .breakdown {{
            display: grid;
            gap: 2px;
        }}

        .breakdown-row {{
            display: grid;
            grid-template-columns: 1fr 64px 72px;
            gap: 6px;
            align-items: center;
            font-size: 11px;
        }}

        .breakdown-track {{
            height: 5px;
            background: #f3f4f6;
            border-radius: 999px;
            overflow: hidden;
        }}

        .breakdown-fill {{
            height: 100%;
            background: #667eea;
        }}

        .num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-variant-numeric: tabular-nums;
        }}

        .data-table th {{
            text-align: left;
            font-size: 10px;
            text-transform: uppercase;
            color: #6d28d9;
            padding: 2px 6px;
            border-bottom: 1px solid #ede9fe;
        }}

        .data-table th.num {{
            text-align: right;
        }}

        .data-table td {{
            padding: 2px 6px;
            border-bottom: 1px solid #f3f4f6;
            font-size: 12px;
        }}

        .data-table td.empty {{
            text-align: center;
            color: #6b7280;
        }}

        .tree-wrap img {{
            max-width: 100%;
            max-height: 280px;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }}

        .controls {{
            display: flex;
            gap: 4px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            font-size: 12px;
        }}

        .btn:hover {{ background: #5568d3; }}

        .filter-btn {{
            background: #e5e7eb;
            color: #374151;
        }}

        .filter-btn:hover {{ background: #d1d5db; }}

        .filter-btn.active {{
            background: #10b981;
            color: white;
        }}

        .steps {{
            display: grid;
            gap: 3px;
        }}

        .step {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
        }}

        .step.success {{ border-left: 3px solid #10b981; }}
        .step.failure {{ border-left: 3px solid #ef4444; }}

        .step-action-row {{
            padding: 3px 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            user-select: none;
            min-height: 28px;
        }}

        .step-action-row:hover {{ background: #f5f3ff; }}

        .step-number {{
            background: #667eea;
            color: white;
            min-width: 22px;
            height: 18px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 11px;
            flex-shrink: 0;
            font-variant-numeric: tabular-nums;
        }}

        .step-action {{
            font-family: ui-monospace, Menlo, monospace;
            font-weight: 700;
            color: #111827;
            font-size: 12px;
            flex: 0 1 auto;
        }}

        .step-preview {{
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #6b7280;
            font-size: 11px;
        }}

        .step-status {{
            padding: 1px 6px;
            border-radius: 999px;
            font-size: 10px;
            font-weight: 700;
            flex-shrink: 0;
        }}

        .step-cam-thumb {{
            width: 48px;
            height: 36px;
            object-fit: cover;
            border-radius: 3px;
            flex-shrink: 0;
            background: #111827;
        }}

        .step-cam {{
            max-width: 100%;
            max-height: 320px;
            border-radius: 4px;
            display: block;
            background: #111827;
        }}

        .step-chevron {{
            font-size: 9px;
            color: #6b7280;
        }}

        .step.expanded .step-chevron {{ transform: rotate(180deg); }}

        .status-success {{ background: #d1fae5; color: #065f46; }}
        .status-failure {{ background: #fee2e2; color: #991b1b; }}

        .step-expand {{
            padding: 6px 8px;
            display: none;
            border-top: 1px solid #e5e7eb;
            background: #fafafa;
        }}

        .step.expanded .step-expand {{ display: block; }}

        .section {{ margin-bottom: 4px; }}

        .section-label {{
            font-size: 10px;
            text-transform: uppercase;
            color: #667eea;
            font-weight: 600;
        }}

        .section-content {{
            padding: 4px 6px;
            background: white;
            border-radius: 3px;
            border: 1px solid #e5e7eb;
            font-size: 12px;
            line-height: 1.35;
            white-space: pre-wrap;
        }}

        .objects {{
            max-height: 120px;
            overflow-y: auto;
            font-size: 11px;
            line-height: 1.35;
            color: #4b5563;
            font-family: ui-monospace, Menlo, monospace;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-top">
                <h1>{task}</h1>
            </div>
            <div class="header-path">{file_path}</div>
            <div class="outcome-row">
                <div class="chip"><span class="chip-label">Success</span><span class="chip-value">{success_rate:.1f}% ({successes}/{total_steps})</span></div>
                <div class="chip"><span class="chip-label">task_percent_complete</span><span class="chip-value">{task_pct_str}</span></div>
                <div class="chip"><span class="chip-label">task_state_success</span><span class="chip-value">{task_ok_str}</span></div>
                <div class="chip"><span class="chip-label">Model</span><span class="chip-value">{model_str}</span></div>
                <div class="chip"><span class="chip-label">Effort</span><span class="chip-value">{effort_str}</span></div>
                <div class="chip"><span class="chip-label">API cost</span><span class="chip-value">{usd_str}</span></div>
            </div>
            {banner_html}
        </div>

        <div class="content">
            <div class="dash">
            <section class="panel">
                <h2>Combined budget <span class="panel-sub">10-minute clock / limit</span></h2>
                <p class="hint">Combined = LLM wall + action sim + explore sim. Combined is not wall-clock runtime.</p>
                <div class="budget-head">
                    <div class="budget-used">{used_str} / {limit_str}</div>
                    <div>left {remaining_str} · {budget_pct:.0f}%</div>
                </div>
                <div class="budget-bar"><div class="budget-bar-fill {fill_class}"></div></div>
                <div class="breakdown">
                    <div class="breakdown-row">
                        <div>LLM/VLM planning (wall-clock)</div>
                        <div class="num">{fmt_seconds(llm_s)}</div>
                        <div class="breakdown-track"><div class="breakdown-fill" style="width:{_share(llm_s):.1f}%"></div></div>
                    </div>
                    <div class="breakdown-row">
                        <div>Actions (sim / {freq:g} Hz)</div>
                        <div class="num">{fmt_seconds(action_s)}</div>
                        <div class="breakdown-track"><div class="breakdown-fill" style="width:{_share(action_s):.1f}%"></div></div>
                    </div>
                    <div class="breakdown-row">
                        <div>Explore approx (~)</div>
                        <div class="num">~{fmt_seconds(explore_s)}</div>
                        <div class="breakdown-track"><div class="breakdown-fill" style="width:{_share(explore_s):.1f}%"></div></div>
                    </div>
                </div>
                <div class="kv" style="margin-top:6px;">
                    <span>LLM <b>{planning_time_str}</b> · {llm_requests_str} req · {llm_avg_str}/avg</span>
                    <span>Tokens <b>{prompt_str}</b> in / <b>{completion_str}</b> out · cached {cached_str}</span>
                    <span>API cost <b>{usd_str}</b></span>
                    <span>Wall-clock runtime <b>{runtime_str}</b> (non-LLM {non_llm_wall_str})</span>
                </div>
            </section>

            <section class="panel">
                <h2>Actions</h2>
                <p class="hint">Explore times are how often it was selected. Explore steps are walked + later geodesic × ratio.</p>
                <div class="metric-grid">
                    <div class="metric"><div class="metric-label">Action steps</div><div class="metric-value">{total_sim_steps}</div></div>
                    <div class="metric"><div class="metric-label">Action time</div><div class="metric-value">{exact_sim_time_str}</div></div>
                    <div class="metric"><div class="metric-label">Env steps</div><div class="metric-value">{env_steps_str}</div></div>
                    <div class="metric"><div class="metric-label">Selected</div><div class="metric-value">{display_total_actions}</div></div>
                </div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Action</th>
                            <th class="num">Times</th>
                            <th class="num">Steps</th>
                            <th class="num">Time</th>
                        </tr>
                    </thead>
                    <tbody>
{exact_table_rows}
                    </tbody>
                </table>
                <h2 style="margin-top:8px;">Explore</h2>
                <p class="hint">First Explore is walked. Later tours use geodesic × walked/geodesic ratio at {freq:g} Hz.</p>
                <div class="kv">
                    <span>times <b>{explore_count}</b></span>
                    <span>steps <b>{explore_approx_steps_str}</b></span>
                    <span>sim <b>{explore_approx_time_str}</b></span>
                    <span>meters <b>{explore_meters_str}</b></span>
                </div>
            </section>
            </div>
{planning_tree_html}
            <section class="panel">
                <div class="controls" style="margin-bottom: 6px;">
                    <h2 style="margin:0; margin-right:auto;">Step log</h2>
                    <button class="btn" onclick="toggleAll()">Expand/Collapse All</button>
                    <button class="btn filter-btn" onclick="filterSteps('all')">All</button>
                    <button class="btn filter-btn" onclick="filterSteps('success')">Success</button>
                    <button class="btn filter-btn" onclick="filterSteps('failure')">Failures</button>
                </div>
                <div class="steps">
{steps_html}
                </div>
            </section>
        </div>
    </div>

    <script>
        function toggleStep(element) {{
            element.classList.toggle('expanded');
        }}

        let allExpanded = false;
        function toggleAll() {{
            const steps = document.querySelectorAll('.step');
            allExpanded = !allExpanded;
            steps.forEach(step => {{
                if (allExpanded) {{
                    step.classList.add('expanded');
                }} else {{
                    step.classList.remove('expanded');
                }}
            }});
        }}

        function filterSteps(filter) {{
            const steps = document.querySelectorAll('.step');
            const buttons = document.querySelectorAll('.filter-btn');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            steps.forEach(step => {{
                if (filter === 'all') {{
                    step.style.display = 'block';
                }} else {{
                    const status = step.getAttribute('data-status');
                    if (status === filter) {{
                        step.style.display = 'block';
                    }} else {{
                        step.style.display = 'none';
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"HTML file generated: {output_file}")
    print(f"  Total steps: {total_steps}")
    print(f"  Success rate: {success_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Convert PARTNR trace logs to HTML')
    parser.add_argument('trace_file', help='Path to the trace text file')
    parser.add_argument('--output', '-o', help='Output HTML file (default: same name as input with .html extension)')
    parser.add_argument('--open', action='store_true', help='Open the HTML file in browser after generation')
    parser.add_argument('--pddl-run', action='store_true', help='Use PDDL-specific trace parsing and rendering mode')
    parser.add_argument('--planning-tree-image', default=None, help='Optional planning_tree.png path to embed for PDDL runs')

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = Path(args.trace_file).with_suffix('.html')

    # Parse and convert
    print(f"Parsing trace file: {args.trace_file}")
    trace_data = parse_trace_file(args.trace_file, is_pddl_run=args.pddl_run)

    print(f"Generating HTML...")
    generate_html(
        trace_data,
        str(output_file),
        is_pddl_run=args.pddl_run,
        planning_tree_image=args.planning_tree_image,
    )

    # Open in browser if requested
    if args.open:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        print(f"✓ Opened in browser")


if __name__ == '__main__':
    main()
