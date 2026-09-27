"""
Render vlm_prompts.txt as a readable chat-style HTML (planner → VLM, then VLM reply when present in jsonl).

Writes: log_dir/vlm_prompts.html
"""

from __future__ import annotations

import ast
import html
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Max bubble body height (px) before collapse + "Expand" is offered.
_BUBBLE_MAX_HEIGHT_PX = 240


def parse_vlm_prompts_txt(content: str) -> List[Dict[str, Any]]:
    """Split the append-only vlm_prompts.txt into blocks (metadata + body)."""
    parts = re.split(r"^={60,}\s*\n", content.strip(), flags=re.MULTILINE)
    blocks: List[Dict[str, Any]] = []
    for chunk in parts:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.split("\n")
        meta = {"reprompt_round": "", "prompt_kind": "", "image_count": "0"}
        body_start = len(lines)
        for i, line in enumerate(lines):
            ls = line.strip()
            if ls.startswith("reprompt_round:"):
                meta["reprompt_round"] = line.split(":", 1)[1].strip()
            elif ls.startswith("prompt_kind:"):
                meta["prompt_kind"] = line.split(":", 1)[1].strip()
            elif ls.startswith("image_count:"):
                meta["image_count"] = line.split(":", 1)[1].strip()
            elif ls == "----------------------------------------------------------------":
                body_start = i + 1
                break
        body = "\n".join(lines[body_start:]).strip()
        blocks.append({**meta, "body": body})
    return blocks


def _format_wall_time(w: Any) -> str:
    """Human-readable local timestamp for jsonl wall_time (epoch seconds)."""
    if w is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(w)).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError, OverflowError, TypeError):
        return str(w)


def _vlm_turn_sort_key(ev: Dict[str, Any]) -> Tuple[float, int, int]:
    """Order turns chronologically: wall_time first (seq_idx may reset across runs)."""
    wt = ev.get("wall_time")
    try:
        wall = float(wt) if wt is not None else 0.0
    except (TypeError, ValueError):
        wall = 0.0
    s = ev.get("seq_idx")
    seq = int(s) if s is not None else 0
    pos = int(ev.get("_file_pos", 0))
    return (wall, seq, pos)


def _empty_response_summary(metadata: Any) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return "(no response text in jsonl for this call)"

    details = []
    status = metadata.get("status")
    if status:
        details.append(f"status={status}")
    finish_reasons = metadata.get("finish_reasons")
    if isinstance(finish_reasons, list):
        reasons = [str(reason) for reason in finish_reasons if reason is not None]
        if reasons:
            details.append(f"finish_reason={','.join(reasons)}")
    for key in (
        "completion_tokens",
        "reasoning_tokens",
        "max_completion_tokens",
        "http_status",
        "request_id",
    ):
        value = metadata.get(key)
        if value is not None:
            details.append(f"{key}={value}")
    error = metadata.get("error")
    if isinstance(error, dict) and error.get("message"):
        details.append(f"error={error['message']}")

    suffix = "; ".join(details)
    if not suffix:
        return "(no response text in jsonl for this call)"
    return f"(empty VLM response; {suffix})"


def _read_vlm_turn_events_from_jsonl(log_dir: str) -> List[Dict[str, Any]]:
    """
    VLM chat turns from vlm_tamp_pddl_log.jsonl, sorted first→last by time/order.
    Each dict: prompt_kind, reprompt_round, image_count, prompt, response, wall_time, seq_idx.
    """
    path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
    if not os.path.isfile(path):
        return []
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for pos, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ev = ast.literal_eval(line)
            except (SyntaxError, ValueError, TypeError):
                continue
            if not isinstance(ev, dict):
                continue
            et = str(ev.get("event") or "")
            if et in ("vlm_english_subgoals", "vlm_predicate_subgoals"):
                kind = (
                    "english_subgoals"
                    if et == "vlm_english_subgoals"
                    else "predicate_subgoals"
                )
            elif et.startswith("custom_") and ev.get("prompt") is not None:
                kind = et
            else:
                continue
            events.append(
                {
                    "prompt_kind": kind,
                    "reprompt_round": ev.get("reprompt_round", ""),
                    "image_count": int(ev.get("image_count", 0) or 0),
                    "prompt": ev.get("prompt") or "",
                    "response": ev.get("response"),
                    "api_response": ev.get("api_response"),
                    "wall_time": ev.get("wall_time"),
                    "seq_idx": ev.get("seq_idx"),
                    "_file_pos": pos,
                }
            )
    events.sort(key=_vlm_turn_sort_key)
    return events


def _read_vlm_responses_in_order(log_dir: str) -> List[str]:
    """Responses for vlm_english_subgoals / vlm_predicate_subgoals in log order."""
    path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
    if not os.path.isfile(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = ast.literal_eval(line)
            except (SyntaxError, ValueError, TypeError):
                continue
            if not isinstance(ev, dict):
                continue
            et = str(ev.get("event") or "")
            if et not in ("vlm_english_subgoals", "vlm_predicate_subgoals") and not (
                et.startswith("custom_") and ev.get("prompt") is not None
            ):
                continue
            r = ev.get("response")
            if r is None:
                out.append("")
            else:
                out.append(str(r))
    return out


def _kind_label(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k == "english_subgoals":
        return "English subgoals"
    if k == "predicate_subgoals":
        return "Predicate subgoals"
    if k == "custom_task_to_subgoals":
        return "Custom task to subgoals"
    if k == "custom_subgoal_exploration":
        return "Custom subgoal exploration"
    if k == "custom_subgoal_to_actions":
        return "Custom subgoal to actions"
    if k == "custom_subgoal_to_pddl_goals":
        return "Custom subgoal to PDDL goals"
    return kind or "prompt"


def render_vlm_prompts_chat_html(
    log_dir: str, out_path: Optional[str] = None
) -> Optional[str]:
    """
    Build chat-style HTML from vlm_tamp_pddl_log.jsonl (preferred: timestamp, chronological order)
    or from vlm_prompts.txt + jsonl responses.
    """
    txt_path = os.path.join(log_dir, "vlm_prompts.txt")
    jsonl_turns = _read_vlm_turn_events_from_jsonl(log_dir)

    rows: List[str] = []
    info_line = ""
    warn = ""

    if jsonl_turns:
        info_line = (
            '<p class="info">Chronological order (first → last): '
            "<code>vlm_tamp_pddl_log.jsonl</code> VLM events, sorted by "
            "<code>wall_time</code> (then <code>seq_idx</code>, then line order). "
            "Timestamps are local time from <code>wall_time</code>.</p>"
        )
        for b in jsonl_turns:
            rr = html.escape(str(b.get("reprompt_round", "")), quote=True)
            pk = html.escape(_kind_label(str(b.get("prompt_kind", ""))), quote=True)
            ic = html.escape(str(b.get("image_count", "0")), quote=True)
            ts = html.escape(_format_wall_time(b.get("wall_time")), quote=True)
            sq = b.get("seq_idx")
            seq_e = html.escape(str(sq) if sq is not None else "—", quote=True)
            body = html.escape(b.get("prompt") or "", quote=False)
            resp = b.get("response")
            parts = [
                '<div class="turn">',
                f'<div class="meta">{ts} · seq {seq_e} · Reprompt {rr} · {pk} · {ic} image(s)</div>',
                '<div class="row planner-row">',
                '<div class="avatar planner-av">Planner</div>',
                '<div class="bubble-col bubble-col-planner">',
                '<div class="bubble-stack">',
                f'<div class="bubble planner"><div class="bubble-text">{body}</div></div>',
                '<button type="button" class="bubble-toggle" hidden aria-expanded="false">Expand</button>',
                "</div></div>",
                "</div>",
            ]
            if resp is not None and str(resp).strip():
                rb = html.escape(str(resp), quote=False)
                parts.extend(
                    [
                        '<div class="row vlm-row">',
                        '<div class="bubble-col bubble-col-vlm">',
                        '<div class="bubble-stack">',
                        f'<div class="bubble vlm"><div class="bubble-text">{rb}</div></div>',
                        '<button type="button" class="bubble-toggle" hidden aria-expanded="false">Expand</button>',
                        "</div></div>",
                        '<div class="avatar vlm-av">VLM</div>',
                        "</div>",
                    ]
                )
            else:
                empty_summary = html.escape(
                    _empty_response_summary(b.get("api_response")), quote=False
                )
                parts.extend(
                    [
                        '<div class="row vlm-row">',
                        '<div class="bubble-col bubble-col-vlm">',
                        '<div class="bubble-stack">',
                        f'<div class="bubble vlm muted"><div class="bubble-text">{empty_summary}</div></div>',
                        '<button type="button" class="bubble-toggle" hidden aria-expanded="false">Expand</button>',
                        "</div></div>",
                        '<div class="avatar vlm-av">VLM</div>',
                        "</div>",
                    ]
                )
            parts.append("</div>")
            rows.append("\n".join(parts))

        warn = info_line

    if not rows and os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        blocks = parse_vlm_prompts_txt(content)
        responses = _read_vlm_responses_in_order(log_dir)
        mismatch = bool(blocks) and bool(responses) and len(blocks) != len(responses)

        for i, b in enumerate(blocks):
            rr = html.escape(str(b.get("reprompt_round", "")), quote=True)
            pk = html.escape(_kind_label(str(b.get("prompt_kind", ""))), quote=True)
            ic = html.escape(str(b.get("image_count", "0")), quote=True)
            body = html.escape(b.get("body") or "", quote=False)
            parts = [
                '<div class="turn">',
                f'<div class="meta">Reprompt {rr} · {pk} · {ic} image(s) · (order: file order in vlm_prompts.txt)</div>',
                '<div class="row planner-row">',
                '<div class="avatar planner-av">Planner</div>',
                '<div class="bubble-col bubble-col-planner">',
                '<div class="bubble-stack">',
                f'<div class="bubble planner"><div class="bubble-text">{body}</div></div>',
                '<button type="button" class="bubble-toggle" hidden aria-expanded="false">Expand</button>',
                "</div></div>",
                "</div>",
            ]
            if i < len(responses):
                if responses[i].strip():
                    rb = html.escape(responses[i], quote=False)
                    parts.extend(
                        [
                            '<div class="row vlm-row">',
                            '<div class="bubble-col bubble-col-vlm">',
                            '<div class="bubble-stack">',
                            f'<div class="bubble vlm"><div class="bubble-text">{rb}</div></div>',
                            '<button type="button" class="bubble-toggle" hidden aria-expanded="false">Expand</button>',
                            "</div></div>",
                            '<div class="avatar vlm-av">VLM</div>',
                            "</div>",
                        ]
                    )
                else:
                    parts.extend(
                        [
                            '<div class="row vlm-row">',
                            '<div class="bubble-col bubble-col-vlm">',
                            '<div class="bubble-stack">',
                            '<div class="bubble vlm muted"><div class="bubble-text">(no response text in log for this call)</div></div>',
                            '<button type="button" class="bubble-toggle" hidden aria-expanded="false">Expand</button>',
                            "</div></div>",
                            '<div class="avatar vlm-av">VLM</div>',
                            "</div>",
                        ]
                    )
            parts.append("</div>")
            rows.append("\n".join(parts))

        if mismatch:
            warn = (
                f'<p class="warn">Note: {len(blocks)} prompt block(s) in vlm_prompts.txt vs '
                f"{len(responses)} VLM event(s) in jsonl — pairing may be partial.</p>"
            )
        elif blocks and not responses:
            warn = (
                '<p class="warn">No matching VLM responses found in vlm_tamp_pddl_log.jsonl '
                "(showing prompts only).</p>"
            )
        else:
            warn = ""

    if not rows:
        return None

    out_path = out_path or os.path.join(log_dir, "vlm_prompts.html")
    esc_dir = html.escape(os.path.abspath(log_dir), quote=True)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VLM prompts — chat view</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  margin: 0;
  background: #e8eaed;
  color: #1a1a1a;
  line-height: 1.45;
}}
header {{
  background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
  color: #fff;
  padding: 16px 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}}
header h1 {{ margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600; }}
header .sub {{ font-size: 0.85rem; opacity: 0.92; }}
header a {{ color: #a8d4ff; }}
.wrap {{ max-width: 920px; margin: 0 auto; padding: 20px 16px 48px; }}
.warn {{
  background: #fff8e6;
  border: 1px solid #e6c200;
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 0.88rem;
  margin-bottom: 16px;
}}
.info {{
  background: #e8f4ea;
  border: 1px solid #81c995;
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 0.88rem;
  margin-bottom: 16px;
}}
.turn {{
  margin-bottom: 20px;
}}
.meta {{
  font-size: 0.75rem;
  color: #5f6368;
  margin-bottom: 8px;
  padding-left: 4px;
}}
.row {{
  display: flex;
  align-items: flex-end;
  gap: 10px;
  margin-bottom: 8px;
}}
.planner-row {{ justify-content: flex-end; }}
.vlm-row {{ justify-content: flex-start; }}
.bubble-col {{
  display: flex;
  flex-direction: column;
  max-width: min(100%, 78%);
  min-width: 0;
}}
.bubble-col-planner {{ align-items: flex-end; }}
.bubble-col-vlm {{ align-items: flex-start; }}
.bubble-stack {{
  display: flex;
  flex-direction: column;
  align-items: inherit;
  max-width: 100%;
}}
.avatar {{
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  font-size: 0.65rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  line-height: 1.1;
  color: #fff;
}}
.planner-av {{ background: #1a73e8; }}
.vlm-av {{ background: #5f6368; }}
.bubble {{
  max-width: 100%;
  padding: 12px 14px;
  border-radius: 16px;
  font-size: 0.82rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}}
.bubble-text {{
  white-space: pre-wrap;
  word-break: break-word;
}}
.bubble-stack:not(.expanded) .bubble-text.is-clamped {{
  max-height: {_BUBBLE_MAX_HEIGHT_PX}px;
  overflow: hidden;
  position: relative;
}}
.bubble-stack:not(.expanded) .bubble-text.is-clamped::after {{
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 36px;
  background: linear-gradient(transparent, rgba(255,255,255,0.92));
  pointer-events: none;
  border-radius: 0 0 12px 12px;
}}
.bubble-stack:not(.expanded) .bubble.planner .bubble-text.is-clamped::after {{
  background: linear-gradient(transparent, rgba(211,227,253,0.95));
}}
.bubble-toggle {{
  margin-top: 6px;
  font-size: 0.75rem;
  padding: 4px 10px;
  cursor: pointer;
  border: 1px solid #dadce0;
  border-radius: 8px;
  background: #fff;
  color: #1a73e8;
  font-weight: 500;
}}
.bubble-toggle:hover {{
  background: #f8f9fa;
}}
.bubble-toggle:focus-visible {{
  outline: 2px solid #1a73e8;
  outline-offset: 2px;
}}
.planner {{
  background: #d3e3fd;
  color: #174ea6;
  border-bottom-right-radius: 4px;
}}
.vlm {{
  background: #fff;
  color: #202124;
  border: 1px solid #dadce0;
  border-bottom-left-radius: 4px;
}}
.vlm.muted {{ color: #80868b; font-style: italic; }}
</style>
</head>
<body>
<header>
  <h1>VLM prompts (chat view)</h1>
  <div class="sub"><code>{esc_dir}</code></div>
  <div class="sub">Also available: <a href="vlm_prompts.txt">vlm_prompts.txt</a> (raw)</div>
</header>
<div class="wrap">
{warn}
{chr(10).join(rows)}
</div>
<script>
(function() {{
  var MAX = {_BUBBLE_MAX_HEIGHT_PX};
  document.querySelectorAll(".bubble-stack").forEach(function(stack) {{
    var textEl = stack.querySelector(".bubble-text");
    var btn = stack.querySelector(".bubble-toggle");
    if (!textEl || !btn) return;
    function applyClamp() {{
      if (stack.classList.contains("expanded")) return;
      textEl.classList.add("is-clamped");
    }}
    function updateButton() {{
      var needs = textEl.scrollHeight > MAX + 2;
      btn.hidden = !needs;
      if (!needs) {{
        textEl.classList.remove("is-clamped");
        stack.classList.remove("expanded");
        btn.textContent = "Expand";
        btn.setAttribute("aria-expanded", "false");
      }} else {{
        if (!stack.classList.contains("expanded")) {{
          applyClamp();
        }}
      }}
    }}
    updateButton();
    btn.addEventListener("click", function() {{
      var ex = !stack.classList.contains("expanded");
      stack.classList.toggle("expanded", ex);
      textEl.classList.toggle("is-clamped", !ex);
      btn.textContent = ex ? "Collapse" : "Expand";
      btn.setAttribute("aria-expanded", ex ? "true" : "false");
    }});
  }});
}})();
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path
