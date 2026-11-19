"""Render a TruPOMDP decision/module timeline as self-contained HTML.

Parses the saved text trace (ToH hypotheses, decisions, skills, evidence, belief
updates) and joins DESPOT search stats from the planner-log. No simulator or
model calls.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

DECISION_RE = re.compile(r"^Decision\s+(\d+):\s*(.*)$")
ACTION_RE = re.compile(r"^Action:\s*(.+)$")
RESULT_RE = re.compile(r"^Result:\s*(.*)$")
EVIDENCE_RE = re.compile(r"^Evidence:\s*(.*)$")
BELIEF_RE = re.compile(r"^Belief:\s*(.*)$")
MEMORY_RE = re.compile(r"^Memory:\s*(.*)$")
HYPO_RE = re.compile(r"^Hypotheses\s+\(([^)]+)\):\s*(.*)$")


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _loads(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def _format_goal(goal: Any) -> str:
    if not isinstance(goal, dict):
        return str(goal)
    obj = goal.get("object") or "?"
    relation = goal.get("relation")
    area = goal.get("target_area")
    states = goal.get("states") or []
    parts = [obj]
    if relation and area:
        parts.append(f"{relation} {area}")
    elif relation:
        parts.append(str(relation))
    elif area:
        parts.append(f"at {area}")
    if states:
        parts.append("[" + ", ".join(str(s) for s in states) + "]")
    return " ".join(parts)


def _summarize_hypotheses(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    rows = []
    for particle in raw:
        if not isinstance(particle, dict):
            continue
        goals = particle.get("goals") or []
        placements = particle.get("placements") or {}
        rows.append(
            {
                "weight": particle.get("weight"),
                "goals": [_format_goal(g) for g in goals],
                "placements": placements,
                "hypothesized": particle.get("hypothesized") or [],
            }
        )
    rows.sort(key=lambda r: (-(r["weight"] or 0), str(r["goals"])))
    return rows


def parse_trace(text: str) -> Dict[str, Any]:
    """Split the TruPOMDP text trace into an initial block plus decision cycles."""
    lines = text.splitlines()
    initial: Dict[str, Any] = {"hypotheses": None, "memory": None}
    decisions: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def flush():
        nonlocal current
        if current is not None:
            decisions.append(current)
            current = None

    i = 0
    while i < len(lines):
        line = lines[i]
        hypo = HYPO_RE.match(line)
        if hypo:
            kind, payload = hypo.group(1), hypo.group(2)
            while i + 1 < len(lines) and not re.match(
                r"^(Hypotheses |Memory:|Decision |Action:|Result:|Evidence:|Belief:|Tree of )",
                lines[i + 1],
            ):
                i += 1
                payload += lines[i]
            parsed = _summarize_hypotheses(_loads(payload))
            if current is None and kind == "initial":
                initial["hypotheses"] = parsed
                initial["hypotheses_kind"] = kind
            elif current is not None:
                current["hypotheses"] = parsed
                current["hypotheses_kind"] = kind
            else:
                initial["hypotheses"] = parsed
                initial["hypotheses_kind"] = kind
            i += 1
            continue
        mem = MEMORY_RE.match(line)
        if mem:
            value = _loads(mem.group(1))
            if current is None:
                initial["memory"] = value
            else:
                current["memory"] = value
            i += 1
            continue
        dec = DECISION_RE.match(line)
        if dec:
            flush()
            current = {
                "index": int(dec.group(1)),
                "symbolic": dec.group(2).strip(),
                "skills": [],
                "evidence": None,
                "observation": None,
                "hypotheses": None,
                "hypotheses_kind": None,
                "memory": None,
                "belief": None,
                "despot": None,
            }
            i += 1
            continue
        act = ACTION_RE.match(line)
        if act and current is not None:
            skill = {"action": act.group(1).strip(), "result": None}
            if i + 1 < len(lines) and RESULT_RE.match(lines[i + 1]):
                i += 1
                skill["result"] = RESULT_RE.match(lines[i]).group(1).strip()
            current["skills"].append(skill)
            i += 1
            continue
        ev = EVIDENCE_RE.match(line)
        if ev and current is not None:
            payload = ev.group(1)
            while i + 1 < len(lines) and not re.match(
                r"^(Hypotheses |Memory:|Decision |Action:|Result:|Evidence:|Belief:)",
                lines[i + 1],
            ):
                i += 1
                payload += lines[i]
            # "…; observation={…}"
            obs = None
            match = re.search(r"observation=(\{.*\})\s*$", payload)
            if match:
                obs = _loads(match.group(1))
                payload = payload[: match.start()].rstrip("; ")
            current["evidence"] = payload
            current["observation"] = obs
            i += 1
            continue
        bel = BELIEF_RE.match(line)
        if bel and current is not None:
            current["belief"] = bel.group(1).strip()
            i += 1
            continue
        i += 1
    flush()
    return {"initial": initial, "decisions": decisions}


def load_despot_by_decision(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Map decision index -> tru_pomdp stats from the planner-log replan rows."""
    folder = run_dir / "dataset" / "planner-log"
    if not folder.exists():
        return {}
    files = sorted(folder.glob("*.json"))
    if not files:
        return {}
    steps = json.loads(files[0].read_text()).get("steps") or []
    by_decision: Dict[int, Dict[str, Any]] = {}
    for step in steps:
        if not (step.get("replanned") or {}).get("0"):
            continue
        info = step.get("tru_pomdp") or {}
        index = info.get("num_decisions")
        if not isinstance(index, int):
            continue
        action = (step.get("high_level_actions") or {}).get("0")
        by_decision[index] = {
            "sim_step": step.get("sim_step_count"),
            "skill": action,
            "belief_summary": info.get("belief"),
            "num_particles": info.get("num_particles"),
            "memory_revisions": info.get("memory_revisions"),
            "rejected_hypotheses": info.get("rejected_hypotheses"),
            "inspected_areas": info.get("inspected_areas"),
            "belief_update": info.get("belief_update") or {},
            "search_time_s": info.get("search_time_s"),
            "elapsed_s": info.get("elapsed_s"),
            "llm_time_s": info.get("llm_time_s"),
            "toh_llm_time_s": info.get("toh_llm_time_s"),
            "toh_process_time_s": info.get("toh_process_time_s"),
            "planner_overhead_time_s": info.get("planner_overhead_time_s"),
            "trials": info.get("trials"),
            "nodes": info.get("nodes"),
            "num_actions": info.get("num_actions"),
            "root_lower": info.get("root_lower"),
            "root_upper": info.get("root_upper"),
            "failed_attempts": info.get("failed_attempts"),
            "termination_reason": info.get("termination_reason"),
            "task_percent_complete": (step.get("stats") or {}).get(
                "task_percent_complete"
            ),
        }
    return by_decision


def load_run_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    run_json = run_dir / "run.json"
    if run_json.exists():
        data = json.loads(run_json.read_text())
        metrics["status"] = data.get("status")
        metrics["runtime_s"] = data.get("elapsed_seconds") or data.get("runtime_seconds")
        m = data.get("metrics") or {}
        metrics["task_percent_complete"] = m.get("task_percent_complete")
        metrics["task_state_success"] = m.get("task_state_success")
        cost = m.get("cost_metrics") or {}
        metrics.update(
            {
                "llm_model": cost.get("llm_model"),
                "llm_usd": cost.get("llm_usd"),
                "llm_planning_time_s": cost.get("llm_planning_time_s"),
                "total_tokens": cost.get("total_tokens"),
            }
        )
    return metrics


def build_report(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    traces = sorted((run_dir / "dataset" / "traces").glob("*/trace-*.txt"))
    if not traces:
        raise FileNotFoundError(f"No trace-*.txt under {run_dir}/dataset/traces")
    parsed = parse_trace(traces[0].read_text(errors="replace"))
    from habitat_llm.termination import termination_from_trace
    termination = termination_from_trace(traces[0].read_text(errors="replace"))
    despot = load_despot_by_decision(run_dir)
    for decision in parsed["decisions"]:
        decision["despot"] = despot.get(decision["index"])
    return {
        "run_dir": str(run_dir),
        "trace_path": str(traces[0]),
        "termination": termination,
        "task": _task_from_run(run_dir),
        "metrics": load_run_metrics(run_dir),
        "initial": parsed["initial"],
        "decisions": parsed["decisions"],
    }


def _task_from_run(run_dir: Path) -> str:
    logs = sorted((run_dir / "dataset" / "planner-log").glob("*.json"))
    if logs:
        return json.loads(logs[0].read_text()).get("task") or run_dir.name
    spec = run_dir / "input" / "spec.md"
    if spec.exists():
        for line in spec.read_text(errors="replace").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return run_dir.name


def _chip(label: str, value: Any) -> str:
    if value is None or value == "":
        return ""
    return (
        f'<div class="chip"><span class="chip-label">{_escape(label)}</span> '
        f"{_escape(value)}</div>"
    )


def _kv_table(rows: List[tuple]) -> str:
    cells = "".join(
        f"<tr><th>{_escape(k)}</th><td>{_escape(v)}</td></tr>"
        for k, v in rows
        if v is not None and v != ""
    )
    return f'<table class="kv">{cells}</table>' if cells else ""


def _hypo_table(rows: Optional[List[Dict[str, Any]]], limit: int = 12) -> str:
    if not rows:
        return '<p class="muted">No hypotheses recorded.</p>'
    body = []
    for index, row in enumerate(rows[:limit], 1):
        goals = "<br>".join(_escape(g) for g in row["goals"]) or "—"
        places = ", ".join(
            f"{_escape(k)}@{_escape(v)}" for k, v in sorted((row["placements"] or {}).items())
        ) or "—"
        hypo = ", ".join(_escape(x) for x in row["hypothesized"]) or "—"
        weight = row["weight"]
        weight_s = f"{weight:.3f}" if isinstance(weight, (int, float)) else _escape(weight)
        body.append(
            f"<tr><td>{index}</td><td>{weight_s}</td><td>{goals}</td>"
            f"<td>{places}</td><td>{hypo}</td></tr>"
        )
    extra = (
        f'<p class="muted">Showing {limit} of {len(rows)} particles.</p>'
        if len(rows) > limit
        else ""
    )
    return (
        '<table class="hypo"><thead><tr><th>#</th><th>Weight</th><th>Goal</th>'
        "<th>Placements</th><th>Still hypothesized</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        + extra
    )


def _skill_list(skills: List[Dict[str, str]]) -> str:
    if not skills:
        return '<p class="muted">No skills recorded for this decision.</p>'
    items = []
    for skill in skills:
        ok = skill.get("result") or ""
        cls = "ok" if "success" in ok.lower() else ("bad" if ok else "")
        items.append(
            f'<li><code>{_escape(skill["action"])}</code> '
            f'<span class="result {cls}">{_escape(ok)}</span></li>'
        )
    return "<ol class='skills'>" + "".join(items) + "</ol>"


def _obs_summary(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ""
    bits = []
    if obs.get("robot_area"):
        bits.append(f"robot @ {obs['robot_area']}")
    if obs.get("inspected"):
        bits.append("inspected " + ", ".join(map(str, obs["inspected"])))
    parents = obs.get("parents") or {}
    if parents:
        bits.append(
            "seen "
            + ", ".join(f"{k}@{v}" for k, v in sorted(parents.items()))
        )
    states = obs.get("states") or {}
    if states:
        bits.append(
            "states "
            + "; ".join(
                f"{obj}: "
                + ", ".join(f"{k}={v}" for k, v in sorted(flags.items()))
                for obj, flags in sorted(states.items())
                if isinstance(flags, dict)
            )
        )
    return "<p>" + _escape(" · ".join(bits)) + "</p>" if bits else ""


def render_html(report: Dict[str, Any]) -> str:
    metrics = report["metrics"]
    decisions = report["decisions"]
    initial = report["initial"]
    success = metrics.get("task_state_success")
    outcome = (
        "Success"
        if success == 1 or success is True
        else "Failure"
        if success == 0 or success is False
        else metrics.get("status") or "Unknown"
    )
    particle_counts = [
        (d.get("despot") or {}).get("num_particles")
        for d in decisions
        if (d.get("despot") or {}).get("num_particles") is not None
    ]
    eliminated = sum(
        int((d.get("despot") or {}).get("belief_update", {}).get("eliminated") or 0)
        for d in decisions
    )
    replenished = sum(
        1
        for d in decisions
        if (d.get("despot") or {}).get("belief_update", {}).get("replenished")
    )
    skill_counts = Counter(
        (s["action"].split("[", 1)[0] if "[" in s["action"] else s["action"])
        for d in decisions
        for s in d["skills"]
    )

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>TruPOMDP decision trace</title><style>",
        CSS,
        "</style></head><body><div class='container'>",
        "<header class='header'><h1>",
        _escape(report["task"]),
        "</h1><div class='path'>",
        _escape(report["run_dir"]),
        "</div><div class='chips'>",
        _chip("OUTCOME", outcome),
        _chip(
            "COMPLETION",
            None
            if metrics.get("task_percent_complete") is None
            else f"{100 * float(metrics['task_percent_complete']):.1f}%",
        ),
        _chip("STATUS", metrics.get("status")),
        _chip("TERMINATION", report.get("termination", {}).get("message")),
        _chip(
            "RUNTIME",
            None
            if metrics.get("runtime_s") is None
            else f"{float(metrics['runtime_s']):.1f}s",
        ),
        _chip("MODEL", metrics.get("llm_model")),
        _chip(
            "API COST",
            None if metrics.get("llm_usd") is None else f"${float(metrics['llm_usd']):.4f}",
        ),
        _chip("DECISIONS", len(decisions)),
        _chip("SKILLS", sum(len(d["skills"]) for d in decisions)),
        _chip(
            "MAX PARTICLES",
            max(particle_counts) if particle_counts else None,
        ),
        _chip("ELIMINATED", eliminated or None),
        _chip("REPLENISHED", replenished or None),
        "</div></header><main>",
        "<section class='panel'><h2>Module loop</h2>"
        "<div class='pipeline'>"
        "<span>1. Tree of Hypotheses</span><span>2. Belief / particles</span>"
        "<span>3. DESPOT search</span><span>4. Execute skills</span>"
        "<span>5. Evidence</span><span>6. Hybrid belief update</span>"
        "</div>"
        "<p class='hint'>Each card below is one online decision. Expand it for hypotheses, "
        "DESPOT stats, executed skills, and the observation that updated the belief.</p>",
        _kv_table(
            [
                ("Skill counts", ", ".join(f"{k}×{v}" for k, v in skill_counts.most_common())),
                ("Trace", report["trace_path"]),
            ]
        ),
        "</section>",
        "<section class='panel'><h2>Initial Tree of Hypotheses</h2>",
        f"<p class='muted'>Kind: {_escape(initial.get('hypotheses_kind') or 'initial')} · "
        f"{len(initial.get('hypotheses') or [])} particles</p>",
        _hypo_table(initial.get("hypotheses")),
    ]
    if initial.get("memory"):
        parts.append(
            "<details><summary>Initial memory</summary><pre>"
            + _escape(json.dumps(initial["memory"], indent=2))
            + "</pre></details>"
        )
    parts.append("</section><section class='panel'><h2>Decisions</h2><div class='timeline'>")

    for decision in decisions:
        despot = decision.get("despot") or {}
        update = despot.get("belief_update") or {}
        n_hyp = len(decision.get("hypotheses") or [])
        status = "ok"
        if any(
            s.get("result") and "success" not in (s.get("result") or "").lower()
            for s in decision["skills"]
        ):
            status = "warn"
        if update.get("replenished") or (update.get("eliminated") or 0) > 0:
            status = "replan" if status == "ok" else status
        preview_bits = [
            decision["symbolic"],
            f"{len(decision['skills'])} skills",
        ]
        if despot.get("num_particles") is not None:
            preview_bits.append(f"{despot['num_particles']} particles")
        if update.get("eliminated"):
            preview_bits.append(f"−{update['eliminated']} eliminated")
        if update.get("replenished"):
            preview_bits.append("replenished")
        parts.append(
            f"<details class='decision {status}'>"
            f"<summary><span class='idx'>D{decision['index']}</span>"
            f"<span class='sym'>{_escape(decision['symbolic'])}</span>"
            f"<span class='preview'>{_escape(' · '.join(preview_bits[1:]))}</span>"
            "</summary><div class='body'>"
        )
        # Module strips
        parts.append("<div class='modules'>")
        parts.append(
            "<div class='mod'><h3>DESPOT</h3>"
            + _kv_table(
                [
                    ("Search time", _fmt_s(despot.get("elapsed_s") or despot.get("search_time_s"))),
                    ("Cumulative search", _fmt_s(despot.get("search_time_s"))),
                    ("LLM time", _fmt_s(despot.get("llm_time_s"))),
                    ("TOH model time", _fmt_s(despot.get("toh_llm_time_s"))),
                    ("TOH processing", _fmt_s(despot.get("toh_process_time_s"))),
                    ("Planner overhead", _fmt_s(despot.get("planner_overhead_time_s"))),
                    ("Trials / nodes", f"{despot.get('trials')} / {despot.get('nodes')}"),
                    ("Actions considered", despot.get("num_actions")),
                    (
                        "Root bounds",
                        None
                        if despot.get("root_lower") is None
                        else f"{despot.get('root_lower'):.1f} … {despot.get('root_upper'):.1f}",
                    ),
                    ("Sim step", despot.get("sim_step")),
                    ("Belief summary", despot.get("belief_summary")),
                    ("Task completion then", despot.get("task_percent_complete")),
                ]
            )
            + "</div>"
        )
        parts.append(
            "<div class='mod'><h3>Execute</h3>"
            + _skill_list(decision["skills"])
            + "</div>"
        )
        parts.append(
            "<div class='mod'><h3>Evidence</h3><p>"
            + _escape(decision.get("evidence") or "—")
            + "</p>"
            + _obs_summary(decision.get("observation"))
            + "</div>"
        )
        parts.append(
            "<div class='mod'><h3>Belief update</h3>"
            + _kv_table(
                [
                    ("Trace belief", decision.get("belief")),
                    ("Particles before → after",
                     None
                     if not update
                     else f"{update.get('num_particles_before')} → {update.get('num_particles_after')}"),
                    ("Surviving mass", update.get("surviving_mass")),
                    ("Eliminated", update.get("eliminated")),
                    ("Replenished", update.get("replenished")),
                    ("Replenishment cause", update.get("replenishment_cause")),
                    ("LLM particles added", update.get("num_llm_particles")),
                    ("Memory revisions", despot.get("memory_revisions")),
                    ("Inspected areas", despot.get("inspected_areas")),
                ]
            )
            + "</div>"
        )
        parts.append("</div>")  # modules
        parts.append(
            f"<details class='hypo-block'{' open' if n_hyp and n_hyp <= 8 else ''}>"
            f"<summary>Hypotheses after update ({n_hyp})</summary>"
            + _hypo_table(decision.get("hypotheses"))
            + "</details>"
        )
        if decision.get("memory") not in (None, {}):
            parts.append(
                "<details><summary>Memory</summary><pre>"
                + _escape(json.dumps(decision["memory"], indent=2))
                + "</pre></details>"
            )
        parts.append("</div></details>")

    parts.append(
        "</div></section></main>"
        "<footer class='panel muted'>Generated from the saved TruPOMDP text trace and "
        "planner-log DESPOT stats. Module names follow habitat_llm/planner/tru_pomdp.</footer>"
        "<script>"
        "document.querySelectorAll('[data-expand]').forEach(b=>b.onclick=()=>"
        "document.querySelectorAll('details.decision').forEach(d=>d.open=true));"
        "</script></div></body></html>"
    )
    # Insert expand controls after Decisions heading - simpler: add buttons at top of timeline
    html_doc = "".join(parts)
    html_doc = html_doc.replace(
        "<div class='timeline'>",
        "<div class='controls'><button type='button' onclick=\"document.querySelectorAll('details.decision').forEach(d=>d.open=true)\">Expand all</button>"
        "<button type='button' onclick=\"document.querySelectorAll('details.decision').forEach(d=>d.open=false)\">Collapse all</button></div>"
        "<div class='timeline'>",
        1,
    )
    return html_doc


def _fmt_s(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return str(value)


CSS = """
:root { --bg:#5b4fc9; --card:#f4f3f8; --ink:#111827; --muted:#64748b; --line:#ddd; --ok:#15803d; --warn:#b45309; --replan:#0369a1; }
*{box-sizing:border-box} body{margin:0;font:13px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink);padding:8px}
.container{max-width:1200px;margin:0 auto;background:var(--card);border-radius:10px;overflow:hidden}
.header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:12px 16px}
.header h1{margin:0;font-size:16px}
.path{font:11px ui-monospace,Menlo,monospace;opacity:.9;word-break:break-all;margin-top:4px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25);border-radius:4px;padding:2px 8px}
.chip-label{font-size:10px;opacity:.85}
main{padding:12px 16px}
.panel{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:12px}
.panel h2{margin:0 0 8px;font-size:15px}
.pipeline{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.pipeline span{background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;padding:3px 10px;font-size:12px}
.hint,.muted{color:var(--muted)}
.controls{display:flex;gap:8px;margin-bottom:10px}
.controls button{padding:5px 10px;border:1px solid #cbd5e1;border-radius:6px;background:#fff;cursor:pointer}
.timeline{display:flex;flex-direction:column;gap:8px}
details.decision{border:1px solid var(--line);border-radius:8px;background:#fafafa}
details.decision>summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:56px 1fr auto;gap:10px;align-items:center;padding:10px 12px}
details.decision>summary::-webkit-details-marker{display:none}
details.decision .idx{font-weight:700;color:#4338ca}
details.decision .sym{font-family:ui-monospace,Menlo,monospace}
details.decision .preview{color:var(--muted);font-size:12px}
details.decision.ok{border-left:4px solid var(--ok)}
details.decision.warn{border-left:4px solid var(--warn)}
details.decision.replan{border-left:4px solid var(--replan)}
.body{padding:0 12px 12px;border-top:1px solid #eee}
.modules{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:10px 0}
.mod{background:#fff;border:1px solid #e5e7eb;border-radius:6px;padding:8px 10px}
.mod h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#4338ca}
table.kv,table.hypo{width:100%;border-collapse:collapse;font-size:12px}
table.kv th{text-align:left;color:var(--muted);font-weight:500;padding:2px 8px 2px 0;white-space:nowrap;vertical-align:top}
table.kv td,table.hypo td,table.hypo th{padding:3px 6px;border-bottom:1px solid #f1f5f9;vertical-align:top}
table.hypo th{text-align:left;background:#f8fafc}
.skills{margin:0;padding-left:18px}
.skills code{font-size:12px}
.result{color:var(--muted);margin-left:6px}
.result.ok{color:var(--ok)}
.result.bad{color:#b91c1c}
pre{white-space:pre-wrap;background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:8px;font-size:11px;max-height:280px;overflow:auto}
.hypo-block{margin-top:8px}
footer{margin:0 16px 16px}
"""


def render_run(run_dir: Path, out_path: Optional[Path] = None) -> Path:
    report = build_report(run_dir)
    destination = Path(out_path) if out_path else Path(run_dir) / "tru_pomdp_trace.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(report), encoding="utf-8")
    return destination


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run_dir", type=Path, help="TruPOMDP run folder containing dataset/")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    path = render_run(args.run_dir, args.output)
    print(path.resolve())


if __name__ == "__main__":
    main()
