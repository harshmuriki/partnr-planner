"""
Render a kitchen-worlds-style index.html from a VLM-TAMP PDDL episode log directory.

Expects:
  - vlm_tamp_pddl_log.jsonl  (dict lines via repr, readable with ast.literal_eval)
  - plan_tree.txt            (optional)
  - vlm_images/*.png         (optional, matched to VLM rounds in order)
"""

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
        v = ast.literal_eval(line)
        return v if isinstance(v, dict) else None
    except (SyntaxError, ValueError):
        return None


def _parse_action_calls(plan_src: str) -> List[str]:
    """Extract Action(name='x', args=(...)) calls into readable strings."""
    actions: List[str] = []
    for m in re.finditer(r"Action\(name='([^']+)',\s*args=\((.*?)\)\)", plan_src):
        name = m.group(1)
        args = m.group(2).strip()
        actions.append(f"{name}({args})")
    return actions


def _parse_pddl_plan_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse pddl_plan entries that contain Action(...) objects and therefore
    cannot be read by ast.literal_eval.
    """
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
    plan_list = _parse_action_calls(plan_src)

    reprompt_round = _int_field("reprompt_round")
    branch = _int_field("branch")
    branch_origin = _int_field("branch_origin")
    subgoal = _str_field("subgoal")
    if reprompt_round is None or branch is None or subgoal is None:
        return None

    return {
        "event": "pddl_plan",
        "reprompt_round": reprompt_round,
        "branch": branch,
        "branch_origin": branch_origin,
        "subgoal": subgoal,
        "plan": plan_list,
        "plan_raw": plan_src,
    }


def _fmt_plan(plan: Any) -> str:
    if plan is None:
        return ""
    if isinstance(plan, list):
        lines = []
        for a in plan:
            lines.append(html.escape(str(a)))
        return "\n".join(lines)
    return html.escape(str(plan))


def render_log_dir_to_html(log_dir: str, out_path: Optional[str] = None) -> Optional[str]:
    """
    Write index.html next to vlm_tamp_pddl_log.jsonl.

    :param log_dir: Directory containing vlm_tamp_pddl_log.jsonl
    :param out_path: Optional path for HTML (default: log_dir/index.html)
    :return: Path to written HTML, or None if jsonl missing
    """
    jsonl_path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
    if not os.path.isfile(jsonl_path):
        return None

    events: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = _safe_literal_eval(line)
            if ev is None:
                ev = _parse_pddl_plan_line(line)
            if ev is not None:
                events.append(ev)

    plan_tree_text = ""
    pt_path = os.path.join(log_dir, "plan_tree.txt")
    if os.path.isfile(pt_path):
        with open(pt_path, "r", encoding="utf-8", errors="replace") as f:
            plan_tree_text = f.read()
    planning_tree_rel = None
    if os.path.isfile(os.path.join(log_dir, "media", "planning_tree.png")):
        planning_tree_rel = "media/planning_tree.png"
    elif os.path.isfile(os.path.join(log_dir, "planning_tree.png")):
        planning_tree_rel = "planning_tree.png"
    vlm_prompts_rel = None
    vlm_prompts_chat_rel = None
    if os.path.isfile(os.path.join(log_dir, "vlm_prompts.txt")):
        vlm_prompts_rel = "vlm_prompts.txt"
        try:
            from habitat_llm.vlm_tamp.render_vlm_prompts_html import (
                render_vlm_prompts_chat_html,
            )

            p_chat = render_vlm_prompts_chat_html(log_dir)
            if p_chat:
                vlm_prompts_chat_rel = os.path.basename(p_chat)
        except Exception:
            vlm_prompts_chat_rel = None

    rounds: Dict[int, Dict[str, Any]] = defaultdict(dict)
    pddl_by_round: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    pddl_all: List[Tuple[int, Dict[str, Any]]] = []

    for ev in events:
        r = int(ev.get("reprompt_round", 0))
        et = ev.get("event", "")
        if et == "vlm_english_subgoals":
            rounds[r]["english"] = ev
        elif et == "vlm_predicate_subgoals":
            rounds[r]["predicate"] = ev
        elif et == "pddl_plan":
            pddl_by_round[r].append(ev)
            pddl_all.append((r, ev))

    # VLM images: one per vlm_english_subgoals in chronological order
    vlm_image_idx = 0
    round_to_img: Dict[int, str] = {}
    for ev in events:
        if ev.get("event") == "vlm_english_subgoals":
            r = int(ev.get("reprompt_round", 0))
            rel = f"vlm_images/vlm_input_{vlm_image_idx:04d}.png"
            abs_img = os.path.join(log_dir, rel)
            if os.path.isfile(abs_img):
                round_to_img[r] = rel
            vlm_image_idx += 1

    out_path = out_path or os.path.join(log_dir, "index.html")

    def esc(s: Any) -> str:
        return html.escape("" if s is None else str(s), quote=True)

    rows_pddl = []
    for idx, (r, ev) in enumerate(pddl_all):
        plan = ev.get("plan", [])
        plen = len(plan) if isinstance(plan, list) else 0
        rows_pddl.append(
            f"<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{esc(r)}</td>"
            f"<td>{esc(ev.get('branch'))}</td>"
            f"<td>{esc(ev.get('subgoal'))}</td>"
            f"<td>{plen}</td>"
            f"<td><pre style='margin:0;font-size:10px;'>{_fmt_plan(plan)}</pre></td>"
            f"</tr>"
        )

    body_rounds: List[str] = []
    pddl_toggle_id = 0
    sorted_rounds = sorted(set(rounds.keys()) | set(pddl_by_round.keys()))

    for r in sorted_rounds:
        rd = rounds[r]
        eng = rd.get("english")
        pred = rd.get("predicate")
        body_rounds.append(
            f'<tr><td colspan="2" class="merged">Round {r + 1} (reprompt_round={r})</td></tr>'
        )

        left_parts: List[str] = []
        img_rel = round_to_img.get(r)
        if eng:
            prompt = (eng.get("prompt") or "")[:120000]
            resp = (eng.get("response") or "")[:120000]
            img_html = ""
            if img_rel:
                img_html = f'<p><img src="{esc(img_rel)}" alt="VLM scene"/></p>'
            left_parts.append(
                f'<div class="message user1">'
                f'<p class="username">Prompt (English subgoals):</p>'
                f"{img_html}"
                f'<p style="white-space:pre-wrap;">{esc(prompt)}</p>'
                f"</div>"
            )
            left_parts.append(
                f'<div class="message user2">'
                f'<p class="username">Answer:</p>'
                f'<p style="white-space:pre-wrap;">{esc(resp)}</p>'
                f"</div>"
            )
        if pred:
            pp = (pred.get("prompt") or "")[:120000]
            pr = (pred.get("response") or "")[:120000]
            branches = pred.get("branches", [])
            br_str = esc(repr(branches))
            left_parts.append(
                f'<div class="message user1">'
                f'<p class="username">Prompt (PDDL predicates):</p>'
                f'<p style="white-space:pre-wrap;">{esc(pp)}</p>'
                f"</div>"
            )
            left_parts.append(
                f'<div class="message user2">'
                f'<p class="username">Answer:</p>'
                f'<p style="white-space:pre-wrap;">{esc(pr)}</p>'
                f"</div>"
            )
            left_parts.append(
                f'<div class="message user3">'
                f'<p class="username">Parsed branches:</p>'
                f'<p style="white-space:pre-wrap;">{br_str}</p>'
                f"</div>"
            )

        right_parts: List[str] = []
        if planning_tree_rel:
            right_parts.append(
                "<p><strong>Planning tree</strong></p>"
                f'<p><img src="{esc(planning_tree_rel)}" alt="Planning tree"/></p>'
            )

        pdls = pddl_by_round.get(r, [])
        if pdls:
            right_parts.append("<p><strong>PDDL plans</strong> (this round):</p>")
            for ev in pdls:
                plan = ev.get("plan", [])
                plen = len(plan) if isinstance(plan, list) else 0
                sub = ev.get("subgoal", "")
                br = ev.get("branch", "")
                pid = f"pddl-{pddl_toggle_id}"
                pddl_toggle_id += 1
                body = _fmt_plan(plan)
                btn_cls = "button-solved" if plen > 0 else "button-failed"
                right_parts.append(
                    f'<button type="button" class="{btn_cls}" '
                    f'onclick="toggleText(\'{pid}\')">'
                    f'branch {esc(br)} | {esc(sub)} | len={plen}</button>&nbsp;'
                )
                right_parts.append(
                    f'<div id="{pid}" class="hidden-text"><pre>{body}</pre></div><br/>'
                )

        left_html = "\n".join(left_parts) if left_parts else "<p>(no VLM events)</p>"
        if right_parts:
            right_html = "\n".join(right_parts)
        else:
            fallback = "<p>No PDDL plans this round.</p>"
            if plan_tree_text:
                fallback += (
                    '<pre style="max-height:340px;overflow:auto;font-size:10px;">'
                    f"{esc(plan_tree_text)}"
                    "</pre>"
                )
            right_html = fallback
        body_rounds.append(
            f'<tr><td class="left-column">{left_html}</td>'
            f'<td class="right-column">{right_html}</td></tr>'
        )

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PARTNR VLM-TAMP + PDDL log</title>
    <style>
table {{ width: 100%; border-collapse: collapse; }}
th, td {{
    border: 1px solid black;
    padding: 10px;
    vertical-align: top;
    text-align: left;
    box-sizing: border-box;
}}
.left-column {{ width: 50%; font-size: 12px; }}
.right-column {{ width: 50%; font-size: 12px; }}
.hidden-text {{ display: none; white-space: pre-wrap; overflow-x: auto; font-size: 11px; }}
button {{ font-size: 13px; margin: 4px 2px; display: inline-block; cursor: pointer; }}
.button-solved {{ color: #27ae60; }}
.button-failed {{ color: #c0392b; }}
.message {{ border-radius: 5px; padding: 10px; margin-bottom: 10px; }}
.user1 {{ background-color: #dbedf9; margin-right: 5%; color: #333; }}
.user2 {{ background-color: #fae5d2; margin-left: 5%; color: #333; }}
.user3 {{ background-color: #dbf7e7; margin-left: 5%; margin-right: 5%; color: #333; }}
.merged {{ background-color: #dde0e2; font-weight: bold; }}
.username {{ font-weight: bold; }}
img {{ max-width: 100%; }}
</style>
    <script>
function toggleText(id) {{
    var el = document.getElementById(id);
    if (!el) return;
    el.style.display = (el.style.display === "none" || el.style.display === "") ? "block" : "none";
}}
    </script>
</head>
<body>
    <h1>PARTNR VLM-TAMP + PDDL</h1>
    <p><code>{esc(os.path.abspath(log_dir))}</code></p>
    {(
        f'<p>VLM prompts: <a href="{esc(vlm_prompts_chat_rel)}" target="_blank" rel="noopener">chat view</a>'
        + (f' · <a href="{esc(vlm_prompts_rel)}" target="_blank" rel="noopener">raw .txt</a>' if vlm_prompts_rel else "")
        + "</p>"
    )
    if vlm_prompts_chat_rel
    else (f'<p><a href="{esc(vlm_prompts_rel)}" target="_blank" rel="noopener">VLM prompts (vlm_prompts.txt)</a></p>' if vlm_prompts_rel else "")}

    <h2>PDDL planning steps</h2>
    <table>
    <tr>
        <th>idx</th><th>reprompt_round</th><th>branch</th><th>subgoal</th><th>plan_len</th><th>plan</th>
    </tr>
    {"".join(rows_pddl)}
    </table>

    <h2>VLM rounds &amp; plan tree</h2>
    <table>
    {"".join(body_rounds)}
    </table>

    <h2>Full plan tree file</h2>
    <pre style="max-height:600px;overflow:auto;font-size:11px;border:1px solid #ccc;padding:8px;">{esc(plan_tree_text)}</pre>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    return out_path
