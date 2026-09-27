"""
Expandable planning and execution trace for PDDL baselines.

Reads: vlm_tamp_pddl_log.jsonl, episode_metrics.json
Writes: log_dir/{run_id}_pddl.html (default; see _default_interactive_pddl_html_basename)
"""

from __future__ import annotations

import ast
from collections import Counter
import html
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from habitat_llm.utils.episode_cost import fmt_seconds
from habitat_llm.utils.llm_usage import fmt_usd

# Match render_planning_tree.py (RGB)
_STATUS_HEX: Dict[str, Tuple[str, str]] = {
    "started": ("#800080", "#f5e6f5"),
    "planned": ("#505050", "#ebebeb"),
    "already": ("#1f78d1", "#d9ebfb"),
    "skipped": ("#b8860b", "#fff7cc"),
    "blocked": ("#6b7280", "#efefef"),
    "solved": ("#27ae60", "#c8ebdc"),
    "failed": ("#c0392b", "#fcdcd6"),
    "current": ("#f39c12", "#fff0d2"),
    "pending": ("#5a5a5a", "#ffffff"),
    "default": ("#3c3c3c", "#f8f8f8"),
}

_STATUS_FONT_HEX: Dict[str, str] = {
    "already": "#145a9c",
    "skipped": "#7a5a00",
    "blocked": "#4b5563",
}


def _safe_filename_segment(segment: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (segment or "").strip()).strip("_")
    return s or "x"


def _default_interactive_pddl_html_basename(log_dir: str) -> str:
    """
    Default output filename: {run_id}_pddl.html

    Expects log_dir like: .../<task_name>/<run_id>/vlm_tamp_pddl/
    Also accepts legacy episode subdirectories.
    where run_id is typically like Task_3_Obj_1.
    """
    try:
        log_dir = os.path.normpath(os.path.abspath(log_dir))
        vlm_sub = os.path.dirname(log_dir)
        run_dir = vlm_sub if os.path.basename(log_dir) == "vlm_tamp_pddl" else os.path.dirname(vlm_sub)
        run_name = os.path.basename(run_dir)
        if not run_name:
            return "index.html"
        return f"{_safe_filename_segment(run_name)}_pddl.html"
    except Exception:
        return "index.html"


def _svg_escape_text(s: str) -> str:
    return html.escape(s or "", quote=False).replace("&#x27;", "'")


def _svg_text_lines(label: str, *, blocked: bool) -> str:
    raw_lines = str(label or "").split("\n")
    lines = [ln if len(ln) <= 52 else ln[:49] + "..." for ln in raw_lines]
    line_height = 14
    start_dy = 5 - (line_height * (len(lines) - 1) / 2)
    parts: List[str] = []
    for idx, line in enumerate(lines):
        dy = start_dy if idx == 0 else line_height
        strike = ' text-decoration="line-through"' if blocked else ""
        parts.append(
            f'<tspan x="0" dy="{dy}"{strike}>{_svg_escape_text(line)}</tspan>'
        )
    return "".join(parts)


def _svg_legend(x: float = 16, y: float = 16) -> str:
    rows = [
        ("#27ae60", "Solved"),
        ("#c0392b", "Failed"),
        ("#800080", "Current / started"),
        ("#1f78d1", "Skipped by PDDL"),
        ("#b8860b", "Skipped after explore"),
        ("#6b7280", "Blocked after failure"),
    ]
    row_h = 22
    width = 170
    height = 28 + row_h * len(rows)
    parts = [
        f'<g transform="translate({x},{y})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#444" stroke-width="1.5"/>',
        '<text x="85" y="18" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="13" font-weight="700" fill="#111">Legend</text>',
    ]
    for idx, (color, label) in enumerate(rows):
        row_y = 28 + idx * row_h
        parts.append(
            f'<rect x="0" y="{row_y}" width="{width}" height="{row_h}" fill="none" stroke="#444" stroke-width="1"/>'
        )
        parts.append(
            f'<rect x="0" y="{row_y}" width="18" height="{row_h}" fill="{color}" stroke="#444" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="26" y="{row_y + 15}" font-family="DejaVu Sans, sans-serif" font-size="12" fill="#111">{_svg_escape_text(label)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _safe_literal_eval(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        tree = ast.parse(line, mode="eval")
        # PDDLStream records namedtuple reprs. Decode their literal fields,
        # without evaluating any logged function calls.
        class ActionLiteral(ast.NodeTransformer):
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Name) and node.func.id == "Action"
                        and not node.args
                        and {kw.arg for kw in node.keywords} == {"name", "args"}):
                    return ast.Dict(
                        keys=[ast.Constant(kw.arg) for kw in node.keywords],
                        values=[kw.value for kw in node.keywords],
                    )
                return node
        v = ast.literal_eval(ActionLiteral().visit(tree))
        return v if isinstance(v, dict) else None
    except (SyntaxError, ValueError):
        return None


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
    if reprompt_round is None or branch is None or subgoal is None:
        return None

    return {
        "event": "pddl_plan",
        "reprompt_round": reprompt_round,
        "branch": branch,
        "subgoal_idx": _int_field("subgoal_idx"),
        "seq_idx": _int_field("seq_idx"),
        "subgoal": subgoal,
        "plan_raw": plan_src,
    }


EPISODE_METRICS_JSON = "episode_metrics.json"


def _load_episode_metrics(log_dir: str) -> Dict[str, Any]:
    """
    Episode outcome/cost metrics from evaluation (same source as ReAct trace HTML).
    Written by planner_demo as episode_metrics.json after the episode finishes.
    """
    path = os.path.join(log_dir, EPISODE_METRICS_JSON)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _load_episode_runtime_sec(log_dir: str) -> Optional[float]:
    """Episode wall-clock runtime from episode_metrics.json, if present."""
    data = _load_episode_metrics(log_dir)
    v = data.get("episode_runtime_sec")
    if isinstance(v, (int, float)) and float(v) >= 0:
        return float(v)
    return None


def _format_task_score(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    number = float(value)
    if 0.0 <= number <= 1.0:
        return f"{number:.3f} ({number:.1%})"
    return f"{number:.3f}"


def _format_usd(value: Any, source: Any = None) -> str:
    if value is None:
        return "N/A"
    text = fmt_usd(value)
    if source == "estimated":
        return f"{text} est."
    return text


def _format_tokens(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    return f"{int(value):,}"


def _outcome_chips_html(metrics: Dict[str, Any]) -> str:
    """ReAct-style outcome chips from episode_metrics.json fields."""
    if not metrics:
        return ""

    task_ok = metrics.get("task_state_success")
    if isinstance(task_ok, (int, float)):
        success_value = "Success" if float(task_ok) >= 1.0 else "Failed"
    else:
        success_value = "N/A"

    model = metrics.get("llm_model")
    model_str = html.escape(str(model).strip()) if model else "N/A"
    effort = metrics.get("llm_reasoning_effort")
    effort_str = html.escape(str(effort).strip()) if effort else "N/A"
    usd_str = _format_usd(metrics.get("llm_usd"), metrics.get("llm_usd_source"))

    chips = [
        ("Outcome", success_value),
        ("task_percent_complete", _format_task_score(metrics.get("task_percent_complete"))),
        ("task_state_success", _format_task_score(metrics.get("task_state_success"))),
        ("Model", model_str),
        ("Effort", effort_str),
        ("API cost", usd_str),
    ]
    parts = ['<div class="outcome-row">']
    for label, value in chips:
        parts.append(
            f'<div class="chip"><span class="chip-label">{html.escape(label)}</span>'
            f'<span class="chip-value">{value}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _cost_kv_html(metrics: Dict[str, Any]) -> str:
    """Compact token/planning/runtime line under the chips."""
    if not metrics:
        return ""

    prompt = _format_tokens(metrics.get("prompt_tokens"))
    completion = _format_tokens(metrics.get("completion_tokens"))
    cached = _format_tokens(metrics.get("cached_tokens"))
    usd = _format_usd(metrics.get("llm_usd"), metrics.get("llm_usd_source"))
    llm_s = metrics.get("llm_planning_time_s")
    llm_req = metrics.get("llm_requests")
    runtime = metrics.get("episode_runtime_sec")
    bits = [
        f"Tokens <b>{prompt}</b> in / <b>{completion}</b> out · cached {cached}",
        f"API cost <b>{usd}</b>",
    ]
    if isinstance(llm_s, (int, float)):
        req_part = f" · {int(llm_req)} req" if isinstance(llm_req, (int, float)) else ""
        bits.insert(0, f"LLM/VLM planning <b>{fmt_seconds(llm_s)}</b>{req_part}")
    if isinstance(runtime, (int, float)):
        bits.append(f"Wall-clock runtime <b>{fmt_seconds(runtime)}</b>")
    return '<div class="kv">' + "".join(f"<span>{b}</span>" for b in bits) + "</div>"


def _read_jsonl_events(log_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = _safe_literal_eval(line)
            if ev is None:
                ev = _parse_pddl_plan_line(line)
            if ev is not None:
                out.append(ev)
    return out


def _execution_logs_by_key(events: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map 'branch-subgoal_idx' -> latest log_text for subgoal_execution events."""
    latest: Dict[str, Tuple[int, str]] = {}
    for ev in events:
        if ev.get("event") != "subgoal_execution":
            continue
        b = ev.get("branch")
        s = ev.get("subgoal_idx")
        if not isinstance(b, int) or not isinstance(s, int):
            continue
        key = f"{b}-{s}"
        seq = int(ev.get("seq_idx", -1))
        prev = latest.get(key)
        if prev is None or seq >= prev[0]:
            latest[key] = (seq, str(ev.get("log_text", "")))
    return {k: v[1] for k, v in latest.items()}


def _pair_log_key(branch: int, subgoal_idx: int) -> str:
    return f"{int(branch)}-{int(subgoal_idx)}"


def _node_pairs(node: Dict[str, Any]) -> List[Dict[str, int]]:
    pairs: List[Dict[str, int]] = []
    raw_pairs = node.get("pairs")
    if isinstance(raw_pairs, list):
        for row in raw_pairs:
            if not isinstance(row, dict):
                continue
            b = row.get("branch")
            sidx = row.get("subgoal_idx")
            if isinstance(b, int) and isinstance(sidx, int):
                pairs.append({"branch": int(b), "subgoal_idx": int(sidx)})
    if pairs:
        return pairs

    b = node.get("branch")
    sidx = node.get("subgoal_idx")
    if isinstance(b, int) and isinstance(sidx, int):
        return [{"branch": int(b), "subgoal_idx": int(sidx)}]
    return []


def _node_execution_logs_by_id(
    nodes: List[Dict[str, Any]],
    exec_logs: Dict[str, str],
) -> Dict[str, str]:
    node_logs: Dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        pairs = _node_pairs(node)
        if not pairs:
            continue

        sections: List[str] = []
        show_headers = len(pairs) > 1
        for pair in pairs:
            key = _pair_log_key(pair["branch"], pair["subgoal_idx"])
            text = str(exec_logs.get(key) or "").strip()
            if not text:
                continue
            if show_headers:
                text = (
                    f"(branch {pair['branch']}, subgoal {pair['subgoal_idx']})\n{text}"
                )
            sections.append(text)

        if sections:
            node_logs[node_id] = "\n\n".join(sections)

    return node_logs


def _extract_top_prompt_text(events: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract a concise run prompt/instruction for display at the top of index.html.
    Prefers the user task goal embedded in the first English VLM prompt.
    """
    for ev in events:
        if str(ev.get("event", "")) != "vlm_english_subgoals":
            continue
        raw = str(ev.get("prompt", "") or "").strip()
        if not raw:
            continue

        # Typical format:
        # "accomplishes the following goal: ``...''."
        goal_match = re.search(
            r"accomplishes the following goal:\s*``(.*?)''",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if goal_match:
            goal = re.sub(r"\s+", " ", goal_match.group(1)).strip()
            if goal:
                return goal

        # Fallback: compact leading section of the prompt.
        compact = re.sub(r"\s+", " ", raw).strip()
        if compact:
            return compact[:600] + ("..." if len(compact) > 600 else "")
    return None


def _extract_latest_scene_prompt(events: List[Dict[str, Any]]) -> str:
    """Return the latest English prompt text that contains scene observation lines."""
    for ev in reversed(events):
        if str(ev.get("event", "")) == "vlm_english_subgoals":
            raw = str(ev.get("prompt", "") or "").strip()
            if raw:
                return raw
    return ""


def _extract_movable_names_from_prompt(prompt: str) -> List[str]:
    """Extract movable object names from `movable: ...` in prompt text."""
    if not prompt:
        return []
    m = re.search(r"^\s*movable:\s*(.+)$", prompt, flags=re.MULTILINE)
    if not m:
        return []
    names = [x.strip() for x in m.group(1).split(",") if x.strip()]
    return sorted(set(names), key=lambda s: s.lower())


def _extract_currently_visible_lines(prompt: str) -> List[str]:
    """Extract lines inside the `Currently, you can see` fenced block."""
    if not prompt:
        return []
    m = re.search(
        r"Currently,\s*you can see the following:\s*``(.*?)''",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    block = m.group(1).strip()
    return [ln.strip() for ln in block.splitlines() if ln.strip()]


def _build_scene_state_rows(events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build final object-state rows from the latest observed scene block.
    Columns: object, relation, support, room, source.
    """
    prompt = _extract_latest_scene_prompt(events)
    movables = _extract_movable_names_from_prompt(prompt)
    lines = _extract_currently_visible_lines(prompt)

    rows_by_obj: Dict[str, Dict[str, str]] = {}
    on_re = re.compile(r"^([A-Za-z0-9_]+)\s+is\s+on\s+([A-Za-z0-9_]+)\.$")
    in_re = re.compile(r"^([A-Za-z0-9_]+)\s+is\s+in\s+([A-Za-z0-9_]+)\.$")
    holding_re = re.compile(r"^Robot\s+is\s+holding\s+([A-Za-z0-9_]+)\.$")

    for ln in lines:
        hm = holding_re.match(ln)
        if hm:
            obj = hm.group(1)
            rows_by_obj[obj] = {
                "object": obj,
                "relation": "holding",
                "support": "agent_0",
                "room": "unknown",
                "source": ln,
            }
            continue
        om = on_re.match(ln)
        if om:
            obj, support = om.group(1), om.group(2)
            rows_by_obj[obj] = {
                "object": obj,
                "relation": "on",
                "support": support,
                "room": "unknown",
                "source": ln,
            }
            continue
        im = in_re.match(ln)
        if im:
            obj, support = im.group(1), im.group(2)
            rows_by_obj[obj] = {
                "object": obj,
                "relation": "in",
                "support": support,
                "room": "unknown",
                "source": ln,
            }
            continue

    for obj in movables:
        if obj not in rows_by_obj:
            rows_by_obj[obj] = {
                "object": obj,
                "relation": "unknown",
                "support": "unknown",
                "room": "unknown",
                "source": "(not visible in final observation block)",
            }

    return sorted(rows_by_obj.values(), key=lambda r: r["object"].lower())


def _build_scene_graph_svg(state_rows: List[Dict[str, str]]) -> str:
    """Simple static scene-graph SVG: object -> relation -> support."""
    if not state_rows:
        return '<p class="hint">(no final scene-state rows available)</p>'

    row_h = 46
    top = 24
    width = 980
    height = top + row_h * max(1, len(state_rows)) + 24
    x_obj, x_rel, x_sup = 150, 470, 790

    parts: List[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" style="background:#f8f8f8;border:1px solid #bbb;border-radius:6px;">',
        '<defs><marker id="sgArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#333"/></marker></defs>',
    ]
    for i, row in enumerate(state_rows):
        y = top + i * row_h
        obj = html.escape(row["object"])
        rel = html.escape(row["relation"])
        sup = html.escape(row["support"])
        parts.append(
            f'<line x1="{x_obj + 105}" y1="{y}" x2="{x_rel - 75}" y2="{y}" stroke="#333" stroke-width="1.8" marker-end="url(#sgArrow)"/>'
        )
        parts.append(
            f'<line x1="{x_rel + 75}" y1="{y}" x2="{x_sup - 105}" y2="{y}" stroke="#333" stroke-width="1.8" marker-end="url(#sgArrow)"/>'
        )
        parts.append(
            f'<rect x="{x_obj - 105}" y="{y - 16}" width="210" height="32" rx="10" fill="#d3e3fd" stroke="#174ea6" stroke-width="1.5"/>'
            f'<text x="{x_obj}" y="{y + 5}" text-anchor="middle" font-size="12" fill="#111">{obj}</text>'
        )
        parts.append(
            f'<rect x="{x_rel - 75}" y="{y - 14}" width="150" height="28" rx="10" fill="#fff3cd" stroke="#9a6700" stroke-width="1.3"/>'
            f'<text x="{x_rel}" y="{y + 4}" text-anchor="middle" font-size="12" fill="#111">{rel}</text>'
        )
        parts.append(
            f'<rect x="{x_sup - 105}" y="{y - 16}" width="210" height="32" rx="10" fill="#e6f4ea" stroke="#1e8e3e" stroke-width="1.5"/>'
            f'<text x="{x_sup}" y="{y + 5}" text-anchor="middle" font-size="12" fill="#111">{sup}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _compute_summary_stats(
    layout_data: Dict[str, Any],
    events: List[Dict[str, Any]],
    episode_runtime_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute subgoal/action summary stats for the top of HTML page."""
    nodes: List[Dict[str, Any]] = layout_data.get("nodes") or []
    node_by_id: Dict[str, Dict[str, Any]] = {
        str(n.get("id")): n for n in nodes if n.get("id") is not None
    }
    pair_rows: List[Dict[str, Any]] = layout_data.get("pair_to_node") or []

    subgoals_generated = len(pair_rows)
    subgoals_used = 0
    subgoals_failed = 0
    for row in pair_rows:
        nid = str(row.get("node_id", ""))
        node = node_by_id.get(nid)
        if not node:
            continue
        st = str(node.get("status", "pending"))
        if st in ("solved", "already", "failed"):
            subgoals_used += 1
        if st == "failed":
            subgoals_failed += 1

    # Count executed low-level actions from per-subgoal execution logs.
    action_counter: Counter[str] = Counter()
    bullet_action_re = re.compile(r"▶\s*\[\d+/\d+\]\s*([A-Za-z_][A-Za-z0-9_-]*)\[")
    special_action_re = re.compile(r"\bAction:\s*([A-Za-z_][A-Za-z0-9_-]*)\[")
    for ev in events:
        if ev.get("event") != "subgoal_execution":
            continue
        text = str(ev.get("log_text", "") or "")
        bullet_matches = list(bullet_action_re.finditer(text))
        if bullet_matches:
            for m in bullet_matches:
                action_counter[m.group(1)] += 1
        else:
            for m in special_action_re.finditer(text):
                action_counter[m.group(1)] += 1

    actions_run_total = int(sum(action_counter.values()))
    actions_breakdown = dict(sorted(action_counter.items(), key=lambda kv: kv[0].lower()))

    # Replanning count = number of times a reprompt/replan cycle was started.
    replanning_count = 0
    for ev in events:
        if str(ev.get("event", "")) == "reprompt_started":
            replanning_count += 1

    # VLM requests = number of prompt calls logged (both turns).
    vlm_requests = 0
    for ev in events:
        et = str(ev.get("event", ""))
        if et in ("vlm_english_subgoals", "vlm_predicate_subgoals"):
            vlm_requests += 1

    # Prefer evaluation-runner episode runtime (matches trace HTML); else JSONL wall_time span.
    runtime_sec: Optional[float] = None
    if episode_runtime_sec is not None:
        runtime_sec = float(episode_runtime_sec)
    else:
        times: List[float] = []
        for ev in events:
            t = ev.get("wall_time")
            if isinstance(t, (int, float)):
                times.append(float(t))
        if times:
            runtime_sec = max(times) - min(times)
    return {
        "subgoals_generated": subgoals_generated,
        "subgoals_used": subgoals_used,
        "subgoals_failed": subgoals_failed,
        "replanning_count": replanning_count,
        "actions_run_total": actions_run_total,
        "actions_breakdown": actions_breakdown,
        "runtime_sec": runtime_sec,
        "vlm_requests": vlm_requests,
    }


def _ensure_failed_node_logs(
    nodes: List[Dict[str, Any]],
    exec_logs: Dict[str, str],
) -> Dict[str, str]:
    """
    Ensure failed nodes are clickable with useful diagnostics even when
    subgoal_execution log_text is unavailable.
    """
    merged = dict(exec_logs)
    for n in nodes:
        st = str(n.get("status", ""))
        if st != "failed":
            continue
        failure_type = str(n.get("failure_type") or "").strip() or "unknown_failure"
        failure_msg = str(n.get("failure_msg") or "").strip()
        label = str(n.get("label") or n.get("id") or "")
        for pair in _node_pairs(n):
            key = _pair_log_key(pair["branch"], pair["subgoal_idx"])
            if key in merged and str(merged[key]).strip():
                continue
            text = (
                f"Subgoal: {label}\n"
                f"Status: failed\n"
                f"Failure type: {failure_type}\n"
            )
            if failure_msg:
                text += f"Failure message: {failure_msg}\n"
            merged[key] = text
    return merged


def render_pddl_baseline_log_dir_to_html(
    log_dir: str,
    out_path: Optional[str] = None,
    episode_runtime_sec: Optional[float] = None,
    episode_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Render the compact, expandable PARTNR-style trace."""
    from habitat_llm.vlm_tamp.render_trace_html import render_trace_html

    return render_trace_html(
        log_dir, out_path=out_path, episode_runtime_sec=episode_runtime_sec,
        episode_metrics=episode_metrics,
    )


def _svg_escape_label(s: str) -> str:
    s = html.escape(s, quote=True)
    return s.replace("&#x27;", "'")
