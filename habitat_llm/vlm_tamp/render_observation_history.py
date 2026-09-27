"""Small audit page for observation-driven replanning and exact VLM inputs."""
import ast
import html
import json
from pathlib import Path


def render_observation_history(log_dir):
    root = Path(log_dir)
    events = []
    for line in (root / 'vlm_tamp_pddl_log.jsonl').read_text().splitlines():
        try:
            event = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if event.get('event') in {'reprompt_started', 'memory_update', 'memory_snapshot',
                                 'vlm_english_subgoals', 'planner_decision', 'search_limit'}:
            events.append(event)
    parts = ['<!doctype html><html><meta charset="utf-8"><title>Observation and replanning history</title><style>body{font:14px system-ui;background:#f4f3f8;margin:24px}details{background:white;padding:12px;margin:8px 0;border-radius:8px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:30%;vertical-align:top;margin:6px}</style><h1>Observation and replanning history</h1>']
    for e in events:
        title = f"{e.get('seq_idx', '')} · {e['event']} · {e.get('reason', e.get('image_context', e.get('decision', '')))}"
        parts.append('<details><summary>'+html.escape(title)+'</summary>')
        for path in e.get('image_paths', []):
            # Only files inside this run can be linked from model-independent metadata.
            if Path(path).is_absolute() or '..' in Path(path).parts:
                continue
            parts.append('<a href="'+html.escape(path, quote=True)+'"><img src="'+html.escape(path, quote=True)+'"></a>')
        parts.append('<pre>'+html.escape(json.dumps(e, indent=2))+'</pre></details>')
    (root/'observation_history.html').write_text(''.join(parts)+'</html>')
