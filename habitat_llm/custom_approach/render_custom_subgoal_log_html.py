"""Render custom-approach runs as a subgoal-centered HTML report."""

from __future__ import annotations

import ast
import html
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _safe_literal_eval(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        value = ast.literal_eval(line)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _parse_action_calls(plan_src: str) -> List[str]:
    actions: List[str] = []
    for match in re.finditer(r"Action\(name='([^']+)',\s*args=\((.*?)\)\)", plan_src):
        actions.append(f"{match.group(1)}({match.group(2).strip()})")
    return actions


def _parse_pddl_plan_line(line: str) -> Optional[Dict[str, Any]]:
    if "'event': 'pddl_plan'" not in line:
        return None

    def int_field(key: str) -> Optional[int]:
        match = re.search(rf"'{key}':\s*(-?\d+)", line)
        return int(match.group(1)) if match else None

    def str_field(key: str) -> Optional[str]:
        match = re.search(rf"'{key}':\s*'([^']*)'", line)
        return match.group(1) if match else None

    plan_src = ""
    match = re.search(r"'plan':\s*(.*?)(?:,\s*'wall_time':|\}\s*$)", line.strip())
    if match:
        plan_src = match.group(1).strip()

    reprompt_round = int_field("reprompt_round")
    branch = int_field("branch")
    subgoal_idx = int_field("subgoal_idx")
    subgoal = str_field("subgoal")
    if reprompt_round is None or branch is None or subgoal_idx is None:
        return None

    return {
        "event": "pddl_plan",
        "reprompt_round": reprompt_round,
        "branch": branch,
        "subgoal_idx": subgoal_idx,
        "subgoal": subgoal,
        "custom_object_subgoal": str_field("custom_object_subgoal"),
        "status": str_field("status"),
        "failure_type": str_field("failure_type"),
        "plan": _parse_action_calls(plan_src),
        "plan_raw": plan_src,
    }


def _read_events(log_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
    if not os.path.isfile(path):
        return []

    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            event = _safe_literal_eval(line)
            if event is None:
                event = _parse_pddl_plan_line(line)
            if event is not None:
                events.append(event)
    return events


def _is_custom_run(events: List[Dict[str, Any]]) -> bool:
    return any(str(event.get("event", "")).startswith("custom_") for event in events)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pretty(value: Any) -> str:
    if value in (None, "", [], {}):
        return "(none)"
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) if value else "(none)"
    if isinstance(value, dict):
        return "\n".join(f"{key}: {val}" for key, val in value.items()) or "(none)"
    return str(value)


def _details(summary: str, body_lines: List[str], idx: int, *, failed: bool = False) -> str:
    body = "\n\n".join(line for line in body_lines if line)
    css_class = "step failed" if failed else "step"
    return (
        f'<details class="{css_class}">'
        f"<summary>{_escape(summary)}</summary>"
        f"<pre>{_escape(body)}</pre>"
        "</details>"
    )


def _event_sort_key(event: Dict[str, Any]) -> Tuple[float, int]:
    wall_time = event.get("wall_time")
    try:
        wall = float(wall_time) if wall_time is not None else 0.0
    except (TypeError, ValueError):
        wall = 0.0
    seq = event.get("seq_idx")
    try:
        seq_idx = int(seq) if seq is not None else 0
    except (TypeError, ValueError):
        seq_idx = 0
    return wall, seq_idx


def _raw_subgoal_key(event: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    branch = event.get("branch")
    subgoal_idx = event.get("subgoal_idx")
    if branch is None or subgoal_idx is None:
        return None
    try:
        return int(branch), int(subgoal_idx)
    except (TypeError, ValueError):
        return None


def _custom_subgoal_lookup(
    events: List[Dict[str, Any]],
) -> Dict[Tuple[int, str], Tuple[int, int]]:
    all_matches: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "subgoal_status":
            continue
        key = _raw_subgoal_key(event)
        subgoal = event.get("subgoal")
        if key is None or not subgoal:
            continue
        match_key = (key[0], str(subgoal))
        if key not in all_matches[match_key]:
            all_matches[match_key].append(key)
    return {
        match_key: keys[0]
        for match_key, keys in all_matches.items()
        if len(keys) == 1
    }


def _subgoal_key(
    event: Dict[str, Any],
    custom_subgoals: Dict[Tuple[int, str], Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    raw_key = _raw_subgoal_key(event)
    if raw_key is None:
        return None

    custom_object_subgoal = event.get("custom_object_subgoal")
    if event.get("event") == "pddl_plan" and custom_object_subgoal:
        return custom_subgoals.get((raw_key[0], str(custom_object_subgoal)), raw_key)

    return raw_key


def _group_events(events: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    custom_subgoals = _custom_subgoal_lookup(events)
    for event in events:
        key = _subgoal_key(event, custom_subgoals)
        if key is not None:
            grouped[key].append(event)
    for key in grouped:
        grouped[key].sort(key=_event_sort_key)
    return dict(sorted(grouped.items()))


def _subgoal_title(key: Tuple[int, int], events: List[Dict[str, Any]]) -> str:
    for event in events:
        subgoal = event.get("subgoal")
        if subgoal and event.get("event") == "subgoal_status":
            return str(subgoal)
    for event in events:
        subgoal = event.get("custom_object_subgoal") or event.get("subgoal")
        if subgoal:
            return str(subgoal)
    return f"branch {key[0]} subgoal {key[1]}"


def _subgoal_status(events: List[Dict[str, Any]]) -> str:
    statuses = [
        str(event.get("status"))
        for event in events
        if event.get("event") == "subgoal_status" and event.get("status")
    ]
    return statuses[-1] if statuses else "unknown"


def _render_exploration(events: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    idx = 0
    event_idx = 0
    while event_idx < len(events):
        event = events[event_idx]
        kind = event.get("event")
        if kind == "custom_subgoal_exploration_parsed":
            actions_or_objects = event.get("actions_or_objects") or []
            reasons = event.get("reasons") or []
            summary = (
                "Exploration VLM output: "
                + (", ".join(str(item) for item in actions_or_objects) or "(empty)")
            )
            body = [
                "Raw parsed output:\n" + _pretty(event.get("parsed")),
                "Actions or objects:\n" + _pretty(actions_or_objects),
                "Why:\n" + _pretty(reasons),
            ]
            next_event = events[event_idx + 1] if event_idx + 1 < len(events) else {}
            if (
                next_event.get("event") == "custom_fast_explore"
                and actions_or_objects
                and str(actions_or_objects[0]).startswith("Explore")
            ):
                body.extend(
                    [
                        "Fast-explore result:\n"
                        + str(next_event.get("summary") or ""),
                        "Found relations:\n"
                        + _pretty(next_event.get("relation_lines")),
                    ]
                )
                event_idx += 1
            rows.append(_details(summary, body, idx))
            idx += 1
        elif kind == "custom_fast_explore":
            room = event.get("room", "unknown_room")
            body = [
                str(event.get("summary") or ""),
                "Found relations:\n" + _pretty(event.get("relation_lines")),
            ]
            rows.append(_details(f"Fast Explore[{room}]", body, idx))
            idx += 1
        elif kind == "custom_exploration_skipped":
            needed = event.get("needed_objects") or []
            reason = event.get("reason") or "unknown"
            body = [
                f"Reason: {reason}",
                "Needed objects:\n" + _pretty(needed),
            ]
            rows.append(_details("Skipped exploration: needed objects found", body, idx))
            idx += 1
        elif kind == "custom_exploration_rejected_completed_objects":
            body = [
                "Rejected objects:\n" + _pretty(event.get("rejected_objects")),
                "Completed objects:\n" + _pretty(event.get("completed_objects")),
            ]
            rows.append(_details("Rejected completed object candidate", body, idx, failed=True))
            idx += 1
        event_idx += 1

    if not rows:
        return '<p class="muted">(no exploration events)</p>'
    return "\n".join(rows)


def _render_pddl(events: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    idx = 0
    for event in events:
        kind = event.get("event")
        if kind == "custom_subgoal_to_pddl_goals_parsed":
            body = [
                "Needed objects:\n" + _pretty(event.get("needed_objects")),
                "Parsed predicates:\n" + _pretty(event.get("parsed")),
                f"Reprompt count: {event.get('reprompt_count', 0)}",
            ]
            rows.append(_details("VLM predicate translation", body, idx))
            idx += 1
        elif kind == "custom_pddl_predicates_rejected_completed_objects":
            body = [
                "Rejected predicates:\n" + _pretty(event.get("rejected_predicates")),
                "Completed objects:\n" + _pretty(event.get("completed_objects")),
            ]
            rows.append(_details("Rejected PDDL predicates using completed objects", body, idx, failed=True))
            idx += 1
        elif kind == "pddl_plan":
            plan = event.get("plan") or []
            status = event.get("status") or ("failed" if not plan else "planned")
            predicate = event.get("subgoal") or "unknown_predicate"
            failed = status == "failed" or event.get("failure_type") is not None
            body = [
                f"Status: {status}",
                f"Plan length: {len(plan) if isinstance(plan, list) else 0}",
                "Plan steps:\n" + _pretty(plan),
                "Raw plan:\n" + _pretty(event.get("plan_raw")),
                "Failure type: " + _pretty(event.get("failure_type")),
            ]
            rows.append(_details(f"Predicate: {predicate}", body, idx, failed=failed))
            idx += 1

    if not rows:
        return '<p class="muted">(no PDDL events)</p>'
    return "\n".join(rows)


def render_custom_subgoal_log_html(
    log_dir: str,
    out_path: Optional[str] = None,
) -> Optional[str]:
    events = _read_events(log_dir)
    if not events or not _is_custom_run(events):
        return None

    grouped = _group_events(events)
    out_path = out_path or os.path.join(log_dir, "index.html")

    prompt_links: List[str] = []
    if os.path.isfile(os.path.join(log_dir, "vlm_prompts.html")):
        prompt_links.append('<a href="vlm_prompts.html" target="_blank" rel="noopener">VLM chat log</a>')
    if os.path.isfile(os.path.join(log_dir, "vlm_prompts.txt")):
        prompt_links.append('<a href="vlm_prompts.txt" target="_blank" rel="noopener">raw prompts</a>')

    sections: List[str] = []
    for key, subgoal_events in grouped.items():
        title = _subgoal_title(key, subgoal_events)
        status = _subgoal_status(subgoal_events)
        sections.append(
            "<section>"
            f"<h2>Subgoal {key[1]} <span class=\"status\">{_escape(status)}</span></h2>"
            f"<p class=\"subgoal\">{_escape(title)}</p>"
            "<h3>Exploration</h3>"
            f"{_render_exploration(subgoal_events)}"
            "<h3>PDDL Branch</h3>"
            f"{_render_pddl(subgoal_events)}"
            "</section>"
        )

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Custom Approach Subgoal Log</title>
    <style>
body {{ font-family: system-ui, sans-serif; margin: 24px; line-height: 1.45; background: #f8f9fa; color: #202124; }}
a {{ color: #1a73e8; }}
section {{ background: white; border: 1px solid #dadce0; border-radius: 10px; margin: 18px 0; padding: 16px; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ margin-bottom: 4px; }}
h3 {{ margin-top: 18px; }}
pre {{ white-space: pre-wrap; overflow-x: auto; margin: 8px 0 0; font-size: 12px; }}
details.step {{ border: 1px solid #dadce0; border-radius: 8px; margin: 8px 0; padding: 8px 10px; background: #fff; }}
details.failed {{ border-color: #f4b4ae; background: #fff7f6; }}
summary {{ cursor: pointer; font-weight: 600; }}
.status {{ font-size: 12px; font-weight: 600; color: #5f6368; border: 1px solid #dadce0; border-radius: 999px; padding: 2px 8px; }}
.subgoal {{ margin-top: 0; color: #3c4043; }}
.muted {{ color: #777; font-style: italic; }}
.links {{ margin: 10px 0 20px; }}
    </style>
</head>
<body>
    <h1>Custom Approach Subgoal Log</h1>
    <p><code>{_escape(os.path.abspath(log_dir))}</code></p>
    <p class="links">{" · ".join(prompt_links)}</p>
    {"".join(sections)}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(html_doc)

    return out_path
