"""
Kitchen-worlds-style planning tree renderer for VLM-TAMP PDDL logs.

This module intentionally stays standalone and PNG-focused:
- initial branches are merged with kitchen-worlds prefix-tree semantics
- reprompt branches are appended as fresh children without re-merging
- rendering is performed by Graphviz `dot`
"""

from __future__ import annotations

import ast
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


END = "end"
CH = "_"

STATUS_TO_COLOR: Dict[str, str] = {
    "started": "purple",
    "already": "dodgerblue3",
    "skipped": "goldenrod3",
    "blocked": "gray55",
    "success": "green",
    "solved": "green",
    "failed": "red",
    "restart": "purple",
    "ungrounded": "orange",
    "current": "orange",
    "planned": "gray40",
    "pending": "gray40",
    "default": "black",
    "succeed": "yellow",
    "end": "yellow",
}

STATUS_TO_FONT_COLOR: Dict[str, str] = {
    "already": "dodgerblue4",
    "skipped": "goldenrod4",
    "blocked": "gray45",
}


@dataclass
class KWNode:
    node_id: str
    name: str
    action: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    color: Optional[str] = None
    status: str = "default"
    failure_type: str = ""
    failure_msg: str = ""
    seq_idx: int = -1
    pair_refs: List[Tuple[int, int]] = field(default_factory=list)


def _safe_literal_eval(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        value = ast.literal_eval(line)
        return value if isinstance(value, dict) else None
    except (SyntaxError, ValueError):
        return None


def _parse_action_calls(plan_src: str) -> List[str]:
    actions: List[str] = []
    for m in re.finditer(r"Action\(name='([^']+)',\s*args=\((.*?)\)\)", plan_src):
        name = m.group(1)
        args = m.group(2).strip()
        actions.append(f"{name}({args})")
    return actions


def _parse_pddl_plan_line(line: str) -> Optional[Dict[str, Any]]:
    if "'event': 'pddl_plan'" not in line:
        return None

    def _int_field(key: str) -> Optional[int]:
        m = re.search(rf"'{key}':\s*(-?\d+)", line)
        return int(m.group(1)) if m else None

    def _str_field(key: str) -> Optional[str]:
        m = re.search(rf"'{key}':\s*'([^']*)'", line)
        return m.group(1) if m else None

    plan_src = ""
    m_plan = re.search(r"'plan':\s*(.*)\}\s*$", line.strip())
    if m_plan:
        plan_src = m_plan.group(1).strip()

    reprompt_round = _int_field("reprompt_round")
    branch = _int_field("branch")
    subgoal = _str_field("subgoal")
    subgoal_idx = _int_field("subgoal_idx")
    if reprompt_round is None or branch is None or subgoal is None:
        return None

    status = "failed" if plan_src == "None" else "already" if plan_src == "[]" else "planned"
    return {
        "event": "pddl_plan",
        "reprompt_round": reprompt_round,
        "branch": branch,
        "subgoal_idx": subgoal_idx,
        "subgoal": subgoal,
        "status": status,
        "failure_type": "pddl_no_plan" if status == "failed" else None,
        "plan": _parse_action_calls(plan_src),
    }


def _read_events(log_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
    if not os.path.isfile(path):
        return []
    events: List[Dict[str, Any]] = []
    seq = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = _safe_literal_eval(line)
            if ev is None:
                ev = _parse_pddl_plan_line(line)
            if ev is None:
                continue
            if "seq_idx" not in ev:
                ev["seq_idx"] = seq
            seq += 1
            events.append(ev)
    return events


def _get_initial_letter(n: int) -> str:
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    if n < 26:
        return alphabets[n]
    mod = n // 26
    rem = n % 26
    return alphabets[mod - 1] + alphabets[rem]


def _get_reprompt_letter(n: int) -> str:
    alphabets = "xyz"
    if n < len(alphabets):
        return alphabets[n]
    return f"z{n - (len(alphabets) - 1)}"


def _normalize_branches(value: Any) -> List[List[str]]:
    out: List[List[str]] = []
    if not isinstance(value, list):
        return out
    for branch in value:
        if not isinstance(branch, list):
            continue
        parsed = [str(x) for x in branch if str(x).strip()]
        if parsed:
            out.append(parsed)
    return out


def _extract_initial_branch_event(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for ev in events:
        if ev.get("event") == "reprompt_branches_added" and not bool(ev.get("append")):
            branches = _normalize_branches(ev.get("added_branches"))
            if branches:
                return ev

    for ev in events:
        if ev.get("event") == "vlm_predicate_subgoals":
            branches = _normalize_branches(ev.get("branches"))
            if branches:
                return {
                    "event": "reprompt_branches_added",
                    "append": False,
                    "added_indices": list(range(len(branches))),
                    "added_branches": branches,
                    "reprompt_round": int(ev.get("reprompt_round", 0) or 0),
                }
    return None


def _make_node(
    nodes: Dict[str, KWNode],
    node_id: str,
    name: str,
    action: str,
    parent_id: Optional[str],
) -> str:
    nodes[node_id] = KWNode(
        node_id=node_id,
        name=name,
        action=action,
        parent_id=parent_id,
    )
    if parent_id is not None:
        nodes[parent_id].children.append(node_id)
    return node_id


def _find_context_node(
    nodes: Dict[str, KWNode],
    pair_to_node: Dict[Tuple[int, int], str],
    branch_idx: Optional[int],
    subgoal_idx: Optional[int],
    root_id: str,
    include_current: bool,
) -> str:
    if not isinstance(branch_idx, int) or not isinstance(subgoal_idx, int):
        return root_id
    start = subgoal_idx if include_current else subgoal_idx - 1
    for idx in range(start, -1, -1):
        node_id = pair_to_node.get((branch_idx, idx))
        if node_id is not None:
            return node_id
    if not include_current and subgoal_idx == 0:
        failed_node_id = pair_to_node.get((branch_idx, 0))
        if failed_node_id is not None:
            return nodes[failed_node_id].parent_id or root_id
    return root_id


def _choose_reprompt_branch(ev: Dict[str, Any]) -> Optional[Tuple[int, List[str]]]:
    branches = _normalize_branches(ev.get("added_branches"))
    if not branches:
        return None

    added_indices = ev.get("added_indices")
    indices: List[int] = []
    if isinstance(added_indices, list):
        for idx in added_indices:
            if isinstance(idx, int):
                indices.append(idx)

    if len(indices) != len(branches):
        indices = list(range(len(branches)))

    active_branch = ev.get("active_branch")
    if isinstance(active_branch, int) and active_branch in indices:
        pos = indices.index(active_branch)
        return indices[pos], branches[pos]
    return indices[0], branches[0]


def _infer_missing_started_event(
    events: List[Dict[str, Any]],
    event_idx: int,
    reprompt_round: Optional[int],
) -> Optional[Dict[str, Any]]:
    for prev_idx in range(event_idx - 1, -1, -1):
        prev = events[prev_idx]
        prev_rr = prev.get("reprompt_round")
        if (
            isinstance(reprompt_round, int)
            and isinstance(prev_rr, int)
            and prev_rr < reprompt_round
        ):
            break
        if prev.get("event") != "subgoal_status":
            continue
        status = str(prev.get("status") or "")
        if status not in {"success", "solved"}:
            continue
        subgoal = str(prev.get("subgoal") or "")
        if not subgoal.startswith("explore("):
            continue
        branch_idx = prev.get("branch")
        subgoal_idx = prev.get("subgoal_idx")
        if not isinstance(branch_idx, int) or not isinstance(subgoal_idx, int):
            continue
        return {
            "reason": "explore_refresh",
            "branch": branch_idx,
            "subgoal_idx": subgoal_idx,
            "subgoal": subgoal,
            "reprompt_round": reprompt_round,
        }
    return None


def _build_tree_from_events(
    events: List[Dict[str, Any]],
) -> Tuple[Dict[str, KWNode], str, Dict[Tuple[int, int], str]]:
    nodes: Dict[str, KWNode] = {}
    root_id = _make_node(nodes, "root", "start", "start", None)
    pair_to_node: Dict[Tuple[int, int], str] = {}
    name_to_node_id: Dict[str, str] = {"start": root_id}

    initial_ev = _extract_initial_branch_event(events)
    if initial_ev is None:
        return nodes, root_id, pair_to_node

    initial_branches = _normalize_branches(initial_ev.get("added_branches"))
    initial_indices_raw = initial_ev.get("added_indices")
    initial_indices: List[int] = []
    if isinstance(initial_indices_raw, list):
        for idx in initial_indices_raw:
            if isinstance(idx, int):
                initial_indices.append(idx)
    if len(initial_indices) != len(initial_branches):
        initial_indices = list(range(len(initial_branches)))

    for order_idx, (branch_idx, plan) in enumerate(zip(initial_indices, initial_branches)):
        current = root_id
        node_name_by_key = name_to_node_id
        for subgoal_idx, action in enumerate(plan + [END]):
            if subgoal_idx == len(plan):
                key = f"{_get_initial_letter(order_idx)}{CH}end"
            else:
                key = f"{subgoal_idx + 1}{CH}{_get_initial_letter(order_idx)}{CH}{action}"
                for prev_idx in range(order_idx):
                    prev_key = f"{subgoal_idx + 1}{CH}{_get_initial_letter(prev_idx)}{CH}{action}"
                    prev_node_id = node_name_by_key.get(prev_key)
                    if (
                        prev_node_id is not None
                        and nodes[prev_node_id].parent_id == current
                    ):
                        key = prev_key
                        break
            node_id = node_name_by_key.get(key)
            if node_id is None:
                action_text = END if subgoal_idx == len(plan) else action
                node_id = _make_node(nodes, key, key, action_text, current)
                node_name_by_key[key] = node_id
            current = node_id
            if subgoal_idx < len(plan):
                pair_to_node[(branch_idx, subgoal_idx)] = node_id
                ref = (branch_idx, subgoal_idx)
                if ref not in nodes[node_id].pair_refs:
                    nodes[node_id].pair_refs.append(ref)

    reprompt_count = 0
    initial_seen = False
    pending_started_ev: Optional[Dict[str, Any]] = None
    for event_idx, ev in enumerate(events):
        if ev.get("event") == "reprompt_started":
            pending_started_ev = ev
            continue
        if ev.get("event") != "reprompt_branches_added":
            continue
        if not bool(ev.get("append")):
            if initial_seen:
                continue
            initial_seen = True
            continue

        chosen = _choose_reprompt_branch(ev)
        if chosen is None:
            continue
        branch_idx, plan = chosen
        rr = ev.get("reprompt_round")
        started_ev = pending_started_ev
        pending_started_ev = None
        if started_ev is None:
            started_ev = _infer_missing_started_event(events, event_idx, rr)

        parent_id = root_id
        if started_ev is not None:
            reason = str(started_ev.get("reason") or "")
            if reason == "explore_refresh":
                parent_id = _find_context_node(
                    nodes,
                    pair_to_node,
                    started_ev.get("branch"),
                    started_ev.get("subgoal_idx"),
                    root_id,
                    include_current=True,
                )
            else:
                parent_id = _find_context_node(
                    nodes,
                    pair_to_node,
                    started_ev.get("failed_branch"),
                    started_ev.get("failed_subgoal_idx"),
                    root_id,
                    include_current=False,
                )
                if parent_id == root_id:
                    parent_id = _find_context_node(
                        nodes,
                        pair_to_node,
                        started_ev.get("branch"),
                        started_ev.get("subgoal_idx"),
                        root_id,
                        include_current=True,
                    )

        current = parent_id
        reprompt_letter = _get_reprompt_letter(reprompt_count)
        reprompt_count += 1
        for subgoal_idx, action in enumerate(plan + [END]):
            key = f"{subgoal_idx + 1}{CH}{reprompt_letter}{CH}{action}"
            action_text = END if subgoal_idx == len(plan) else action
            node_id = _make_node(nodes, key, key, action_text, current)
            current = node_id
            if subgoal_idx < len(plan):
                pair_to_node[(branch_idx, subgoal_idx)] = node_id
                ref = (branch_idx, subgoal_idx)
                if ref not in nodes[node_id].pair_refs:
                    nodes[node_id].pair_refs.append(ref)

    return nodes, root_id, pair_to_node


def _apply_status_colors(
    nodes: Dict[str, KWNode],
    pair_to_node: Dict[Tuple[int, int], str],
    events: List[Dict[str, Any]],
) -> None:
    latest_status: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for ev in events:
        if ev.get("event") != "subgoal_status":
            continue
        branch_idx = ev.get("branch")
        subgoal_idx = ev.get("subgoal_idx")
        if not isinstance(branch_idx, int) or not isinstance(subgoal_idx, int):
            continue
        latest_status[(branch_idx, subgoal_idx)] = ev

    for ev in events:
        if ev.get("event") != "pddl_plan":
            continue
        branch_idx = ev.get("branch")
        subgoal_idx = ev.get("subgoal_idx")
        if not isinstance(branch_idx, int) or not isinstance(subgoal_idx, int):
            continue
        if (branch_idx, subgoal_idx) in latest_status:
            continue
        latest_status[(branch_idx, subgoal_idx)] = {
            "status": ev.get("status", "pending"),
            "failure_type": ev.get("failure_type"),
            "failure_msg": "",
            "seq_idx": ev.get("seq_idx", -1),
        }

    for pair, status_ev in latest_status.items():
        node_id = pair_to_node.get(pair)
        if node_id is None:
            continue
        node = nodes[node_id]
        seq_idx = int(status_ev.get("seq_idx", -1))
        if seq_idx < node.seq_idx:
            continue
        raw_status = str(status_ev.get("status", "pending"))
        status = "solved" if raw_status == "success" else raw_status
        node.seq_idx = seq_idx
        node.status = status
        node.color = STATUS_TO_COLOR.get(status, STATUS_TO_COLOR["default"])
        node.failure_type = str(status_ev.get("failure_type") or "")
        node.failure_msg = str(status_ev.get("failure_msg") or "")

    def _mark_branch_suffix(
        branch_idx: Optional[int],
        subgoal_idx: Optional[int],
        *,
        status: str,
        color: str,
        seq_idx: int,
    ) -> None:
        if not isinstance(branch_idx, int) or not isinstance(subgoal_idx, int):
            return
        next_idx = subgoal_idx + 1
        while True:
            node_id = pair_to_node.get((branch_idx, next_idx))
            if node_id is None:
                break
            node = nodes[node_id]
            if node.status == "default":
                node.seq_idx = seq_idx
                node.status = status
                node.color = color
            next_idx += 1

    # Suffixes below a failed subgoal are structurally unreachable in that branch.
    for pair, status_ev in latest_status.items():
        if str(status_ev.get("status", "")) != "failed":
            continue
        branch_idx, subgoal_idx = pair
        _mark_branch_suffix(
            branch_idx,
            subgoal_idx,
            status="blocked",
            color=STATUS_TO_COLOR["blocked"],
            seq_idx=int(status_ev.get("seq_idx", -1)),
        )

    # Branch suffixes abandoned after an explore refresh were skipped structurally,
    # even though they never receive explicit runtime status events.
    for ev in events:
        if ev.get("event") != "reprompt_started":
            continue
        if str(ev.get("reason") or "") != "explore_refresh":
            continue
        _mark_branch_suffix(
            ev.get("branch"),
            ev.get("subgoal_idx"),
            status="skipped",
            color=STATUS_TO_COLOR["skipped"],
            seq_idx=int(ev.get("seq_idx", -1)),
        )

    pending_started_ev: Optional[Dict[str, Any]] = None
    for event_idx, ev in enumerate(events):
        if ev.get("event") == "reprompt_started":
            pending_started_ev = ev
            continue
        if ev.get("event") != "reprompt_branches_added" or not bool(ev.get("append")):
            continue
        started_ev = pending_started_ev
        pending_started_ev = None
        if started_ev is None:
            started_ev = _infer_missing_started_event(
                events,
                event_idx,
                ev.get("reprompt_round"),
            )
        if started_ev is None:
            continue
        if str(started_ev.get("reason") or "") != "explore_refresh":
            continue
        _mark_branch_suffix(
            started_ev.get("branch"),
            started_ev.get("subgoal_idx"),
            status="skipped",
            color=STATUS_TO_COLOR["skipped"],
            seq_idx=int(ev.get("seq_idx", -1)),
        )


def _node_label(
    node: KWNode,
    show_failure_labels: bool,
    failure_label_mode: str,
) -> str:
    if node.action == "start":
        label = "start"
    elif node.action == END:
        if node.pair_refs:
            refs = ", ".join(f"r{b}:end" for b, _ in sorted(node.pair_refs))
            label = refs
        else:
            label = "end"
    else:
        refs = ", ".join(f"r{b}:s{s}" for b, s in sorted(node.pair_refs))
        label = f"{refs}\n{node.action}" if refs else node.action
    if show_failure_labels and node.status == "failed":
        if failure_label_mode == "flag":
            label = f"{label}\nfailed"
        elif failure_label_mode == "type":
            detail = node.failure_type or "failed"
            label = f"{label}\nfailed: {detail}"
        else:
            detail = node.failure_type or "failed"
            if node.failure_msg:
                msg = node.failure_msg.strip().split("\n")[0]
                if len(msg) > 60:
                    msg = msg[:57] + "..."
                detail = f"{detail}: {msg}"
            label = f"{label}\nfailed: {detail}"
    return label


def _node_label_attr(
    node: KWNode,
    show_failure_labels: bool,
    failure_label_mode: str,
) -> str:
    label = _node_label(node, show_failure_labels, failure_label_mode)
    if node.status != "blocked":
        return f'label="{_dot_escape(label)}"'
    lines = label.split("\n")
    html_lines = "<BR/>".join(
        f"<S>{html.escape(line, quote=False)}</S>" for line in lines
    )
    return f"label=<{html_lines}>"


def _legend_label() -> str:
    rows = [
        ("green", "Solved"),
        ("red", "Failed"),
        ("purple", "Current / started"),
        ("dodgerblue3", "Skipped by PDDL"),
        ("goldenrod3", "Skipped after explore"),
        ("gray55", "Blocked after failure"),
    ]
    table_rows = [
        '<TR><TD COLSPAN="2"><B>Legend</B></TD></TR>',
    ]
    for color, label in rows:
        table_rows.append(
            f'<TR><TD WIDTH="18" HEIGHT="14" BGCOLOR="{color}"></TD>'
            f'<TD ALIGN="LEFT">{html.escape(label, quote=False)}</TD></TR>'
        )
    return (
        '<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" '
        'BGCOLOR="white">'
        + "".join(table_rows)
        + "</TABLE>>"
    )


def _dot_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _tree_to_dot(
    nodes: Dict[str, KWNode],
    root_id: str,
    show_failure_labels: bool = True,
    failure_label_mode: str = "flag",
) -> str:
    lines = [
        "digraph planning_tree {",
        '  graph [rankdir=TB, outputorder="edgesfirst"];',
        '  node [shape=ellipse, fontname="Helvetica"];',
        '  edge [arrowsize=0.7];',
        f'  "__legend__" [shape=plain, margin=0, label={_legend_label()}];',
    ]

    def visit(node_id: str) -> None:
        node = nodes[node_id]
        attrs = [_node_label_attr(node, show_failure_labels, failure_label_mode)]
        if node.color:
            attrs.append(f'color="{node.color}"')
            attrs.append("penwidth=2")
        font_color = STATUS_TO_FONT_COLOR.get(node.status)
        if font_color:
            attrs.append(f'fontcolor="{font_color}"')
        lines.append(f'  "{node_id}" [{", ".join(attrs)}];')
        for child_id in node.children:
            lines.append(f'  "{node_id}" -> "{child_id}";')
            visit(child_id)

    visit(root_id)
    lines.append('  { rank=min; "__legend__"; "root"; }')
    lines.append('  "__legend__" -> "root" [style=invis, weight=100];')
    lines.append("}")
    return "\n".join(lines)


def _compute_tree_layout(
    nodes: Dict[str, KWNode],
    root_id: str,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    depths: Dict[str, int] = {}

    def assign_depth(node_id: str, depth: int) -> None:
        depths[node_id] = depth
        for child_id in nodes[node_id].children:
            assign_depth(child_id, depth + 1)

    assign_depth(root_id, 0)

    x_positions: Dict[str, float] = {}
    next_leaf_x = 0.0

    def assign_x(node_id: str) -> float:
        nonlocal next_leaf_x
        children = nodes[node_id].children
        if not children:
            x_positions[node_id] = next_leaf_x
            next_leaf_x += 1.0
            return x_positions[node_id]
        child_xs = [assign_x(child_id) for child_id in children]
        x_positions[node_id] = sum(child_xs) / len(child_xs)
        return x_positions[node_id]

    assign_x(root_id)
    return depths, x_positions


def _serialize_layout_data(
    nodes: Dict[str, KWNode],
    root_id: str,
    pair_to_node: Dict[Tuple[int, int], str],
    show_failure_labels: bool,
    failure_label_mode: str,
) -> Dict[str, Any]:
    depths, x_positions = _compute_tree_layout(nodes, root_id)

    node_rows: List[Dict[str, Any]] = []
    for node_id, node in sorted(
        nodes.items(),
        key=lambda item: (
            depths.get(item[0], 0),
            x_positions.get(item[0], 0.0),
            item[0],
        ),
    ):
        row: Dict[str, Any] = {
            "id": node_id,
            "parent_id": node.parent_id,
            "label": _node_label(node, show_failure_labels, failure_label_mode),
            "status": node.status,
            "failure_type": node.failure_type,
            "failure_msg": node.failure_msg,
            "depth": depths.get(node_id, 0),
            "x": x_positions.get(node_id, 0.0),
            "pairs": [
                {"branch": branch, "subgoal_idx": subgoal_idx}
                for branch, subgoal_idx in sorted(node.pair_refs)
            ],
        }
        if node.pair_refs:
            row["branch"] = node.pair_refs[0][0]
            row["subgoal_idx"] = node.pair_refs[0][1]
        node_rows.append(row)

    pair_rows = [
        {
            "branch": branch,
            "subgoal_idx": subgoal_idx,
            "node_id": node_id,
        }
        for (branch, subgoal_idx), node_id in sorted(pair_to_node.items())
    ]

    return {
        "nodes": node_rows,
        "pair_to_node": pair_rows,
        "layout": {
            "x_spacing": 280,
            "y_spacing": 150,
            "margin_x": 120,
            "margin_y": 60,
        },
    }


def export_planning_tree_layout_json(
    log_dir: str,
    nodes: Dict[str, KWNode],
    root_id: str,
    pair_to_node: Dict[Tuple[int, int], str],
    show_failure_labels: bool = True,
    failure_label_mode: str = "flag",
) -> str:
    media_dir = os.path.join(log_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    out_path = os.path.join(media_dir, "planning_tree_layout.json")
    layout_data = _serialize_layout_data(
        nodes,
        root_id,
        pair_to_node,
        show_failure_labels=show_failure_labels,
        failure_label_mode=failure_label_mode,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(layout_data, f, indent=2)
    return out_path


def render_planning_tree_from_log_dir(
    log_dir: str,
    out_path: Optional[str] = None,
    show_failure_labels: bool = True,
    failure_label_mode: str = "flag",
    write_layout_json: bool = False,
) -> Optional[str]:
    events = _read_events(log_dir)
    if not events:
        return None

    nodes, root_id, pair_to_node = _build_tree_from_events(events)
    if len(nodes) <= 1:
        return None
    _apply_status_colors(nodes, pair_to_node, events)
    if write_layout_json:
        export_planning_tree_layout_json(
            log_dir,
            nodes,
            root_id,
            pair_to_node,
            show_failure_labels=show_failure_labels,
            failure_label_mode=failure_label_mode,
        )

    dot_bin = shutil.which("dot")
    if not dot_bin:
        return None

    out_path = out_path or os.path.join(log_dir, "media", "planning_tree.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dot_text = _tree_to_dot(
        nodes,
        root_id,
        show_failure_labels=show_failure_labels,
        failure_label_mode=failure_label_mode,
    )

    fd, dot_path = tempfile.mkstemp(prefix="planning_tree_", suffix=".dot", dir=log_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(dot_text)
        result = subprocess.run(
            [dot_bin, "-Tpng", dot_path, "-o", out_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
    finally:
        try:
            os.remove(dot_path)
        except OSError:
            pass

    return out_path
