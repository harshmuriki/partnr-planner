#!/usr/bin/env python3
"""
Trace log annotator for baseline_evaluation_v3_react runs.

Usage:
    python scripts/trace_annotator/app.py [--host 127.0.0.1] [--port 5055]

Then open http://127.0.0.1:5055
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GUI_DIR))
os.chdir(PROJECT_ROOT)

from catalog import (  # noqa: E402
    EFFICIENCY_LABELS,
    FAILURE_LABELS,
    RunCatalog,
    _run_dir_from_trace,
    infer_criteria_from_eval,
    infer_criteria_from_trace,
    infer_react_success,
    infer_trace_issues,
    load_episode_eval_bundle,
    load_planner_eval_stats,
    final_object_states,
    merge_criteria_from_trace,
    parse_object_states,
    parse_eval_trace_file,
    step_image_path,
    load_world_graph,
)

DEFAULT_LOGS_ROOT = PROJECT_ROOT / "results" / "baseline_evaluation_v3_react"
DEFAULT_PDDL_ROOT = PROJECT_ROOT / "results" / "baseline_evaluation_v3_vlm_tamp_pddl"
EPISODES_ROOT = PROJECT_ROOT / "baseline_evaluation_v3" / "episodes"

app = Flask(
    __name__,
    template_folder=str(GUI_DIR / "templates"),
    static_folder=str(GUI_DIR / "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

PDDL_EMBED_CSS = """
<style id="annotator-embed">
body > p { display: none !important; }
.prompt-box, .summary, .scene-box { display: none !important; }
#wrap { margin: 0; padding: 8px; }
svg { pointer-events: auto; }
.sg-node, .sg-node * { pointer-events: auto; cursor: pointer; }
#node-log-popup { z-index: 10000; }
</style>
"""

PDDL_EMBED_JS = """
<script id="annotator-embed-js">
(function() {
  function bindNode(g) {
    if (!g || !g.querySelector("ellipse")) return;
    g.style.cursor = "pointer";
    g.addEventListener("click", function(evt) {
      evt.stopPropagation();
      evt.preventDefault();
      var k = g.getAttribute("data-log-key");
      var dataEl = document.getElementById("exec-logs");
      var data = {};
      try { data = JSON.parse((dataEl && dataEl.textContent) || "{}"); } catch (err) {}
      var text = (k && data[k]) ? data[k] : "(no log for this node)";
      var pre = document.getElementById("logtext");
      if (pre) pre.textContent = text;
      var popup = document.getElementById("node-log-popup");
      var popupContent = document.getElementById("node-log-content");
      if (!popup || !popupContent) return;
      popupContent.textContent = text;
      popup.style.display = "block";
      popup.style.left = (evt.clientX + 12) + "px";
      popup.style.top = (evt.clientY + 12) + "px";
    }, true);
  }
  document.querySelectorAll("#wrap > svg g").forEach(bindNode);
})();
</script>
"""


catalog: Optional[RunCatalog] = None


def pddl_extra_sources(logs_root: Path) -> list:
    scan_root = (DEFAULT_PDDL_ROOT / "luna_high_all").resolve()
    logs = logs_root.resolve()
    if not scan_root.is_dir():
        return []
    if logs == scan_root or logs == DEFAULT_PDDL_ROOT.resolve():
        return []
    return [
        {
            "prefix": "vlm_tamp_pddl/",
            "root": scan_root,
            "planner": "vlm_tamp_pddl",
            "batch": "vlm_tamp_pddl",
            "files_root": DEFAULT_PDDL_ROOT.resolve(),
        }
    ]


def make_catalog(logs_root: Path) -> RunCatalog:
    return RunCatalog(logs_root, EPISODES_ROOT, pddl_extra_sources(logs_root))


def get_catalog() -> RunCatalog:
    global catalog
    if catalog is None:
        catalog = make_catalog(DEFAULT_LOGS_ROOT)
    return catalog


def _pddl_payload(bundle: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(bundle, dict):
        return None
    rel = str(bundle.get("rel_dir") or "").strip()
    if not rel:
        return None
    graph_name = str(bundle.get("graph_name") or "").strip()
    images = [
        f"/pddl-files/{quote(rel)}/vlm_images/{quote(name)}"
        for name in (bundle.get("image_names") or [])
        if name
    ]
    return {
        "graph_url": f"/pddl-files/{quote(rel)}/{quote(graph_name)}" if graph_name else None,
        "prompts_url": (
            f"/pddl-files/{quote(rel)}/vlm_prompts.html"
            if bundle.get("prompts_html")
            else None
        ),
        "prompts_txt_url": (
            f"/pddl-files/{quote(rel)}/vlm_prompts.txt"
            if bundle.get("prompts_txt")
            else None
        ),
        "metrics": bundle.get("metrics") or {},
        "images": images,
    }


def _public_run(run: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(run)
    payload.pop("images_dir", None)
    bundle = payload.pop("pddl_bundle", None)
    payload["has_images"] = int(payload.get("image_count") or 0) > 0
    payload["pddl"] = _pddl_payload(bundle)
    return payload


def _serialize_step(
    step: Dict[str, Any],
    index: int,
    run_id: str,
    image_count: int,
) -> Dict[str, Any]:
    success = step.get("success")
    if success is None:
        success = infer_react_success(str(step.get("result") or ""))
    image_url = None
    if image_count > 0 and index <= image_count:
        image_url = f"/api/step-image?id={quote(run_id)}&index={index}"
    thought = str(step.get("thought") or "").strip()
    subgoal = str(step.get("subgoal") or "").strip()
    if not thought and subgoal:
        thought = f"Subgoal: {subgoal}"
    result = str(step.get("result") or "").strip()
    if not result and step.get("pddl_log"):
        result = str(step.get("pddl_log") or "").strip()
    return {
        "index": index,
        "action": step.get("action") or "",
        "args": step.get("args") or "",
        "thought": thought,
        "subgoal": subgoal,
        "result": result,
        "objects": step.get("objects") or "",
        "object_states": parse_object_states(str(step.get("objects") or "")),
        "success": bool(success),
        "image_url": image_url,
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/meta")
def api_meta():
    current = get_catalog()
    current.refresh()
    return jsonify(
        {
            "logs_root": str(current.logs_root),
            "annotations_path": str(current.annotations_path),
            "failure_labels": FAILURE_LABELS,
            "efficiency_labels": EFFICIENCY_LABELS,
            "batches": current.list_batches(),
        }
    )


@app.get("/api/runs")
def api_runs():
    current = get_catalog()
    current.refresh()
    return jsonify(
        {
            "runs": [_public_run(run) for run in current.list_runs()],
            "batches": current.list_batches(),
        }
    )


@app.get("/api/run")
def api_run():
    run_id = str(request.args.get("id") or "").strip()
    current = get_catalog()
    run = current.get_run(run_id)
    if run is None:
        return jsonify({"error": "Unknown run.", "field": "run_id"}), 404
    parsed = parse_eval_trace_file(
        run["trace_path"], str(run.get("planner") or "react")
    )
    image_count = int(run.get("image_count") or 0)
    steps = [
        _serialize_step(step, index, run_id, image_count)
        for index, step in enumerate(parsed.get("steps") or [], start=1)
        if isinstance(step, dict)
    ]
    instruction = run.get("instruction") or parsed.get("task") or ""
    eval_bundle = load_episode_eval_bundle(run["variant_folder"], EPISODES_ROOT)
    planner_stats = load_planner_eval_stats(_run_dir_from_trace(Path(run["trace_path"])))
    auto_criteria = merge_criteria_from_trace(
        infer_criteria_from_eval(
            run["criteria"],
            eval_bundle.get("propositions") or [],
            eval_bundle.get("handle_to_name") or {},
            eval_bundle.get("name_to_class") or {},
            planner_stats,
        ),
        infer_criteria_from_trace(run["criteria"], final_object_states(steps)),
    )
    auto_issues = infer_trace_issues(
        steps,
        bool(auto_criteria) and all(auto_criteria.get(item) for item in run["criteria"]),
    )
    annotation = current.annotation_for(run_id, run["criteria"], auto_criteria, auto_issues)
    payload = _public_run(run)
    payload.update(
        {
            "instruction": instruction,
            "task": parsed.get("task") or instruction,
            "steps": steps,
            "total_steps": len(steps),
            "world_graph": load_world_graph(run["trace_path"]),
            "task_explanation": planner_stats.get("task_explanation") or "",
            "annotation": annotation,
        }
    )
    return jsonify(payload)


@app.get("/api/step-image")
def api_step_image():
    run_id = str(request.args.get("id") or "").strip()
    try:
        index = int(request.args.get("index") or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid step index."}), 400
    current = get_catalog()
    run = current.get_run(run_id)
    if run is None:
        return jsonify({"error": "Unknown run."}), 404
    path = step_image_path(run.get("images_dir"), index, current.image_roots())
    if path is None or not path.is_file():
        return jsonify({"error": "No image for this step."}), 404
    response = send_file(path, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.put("/api/annotation")
def api_save_annotation():
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id") or "").strip()
    current = get_catalog()
    record, errors = current.save_run_annotation(run_id, payload)
    if errors:
        status = 404 if errors[0].get("field") == "run_id" else 400
        return jsonify({"ok": False, "errors": errors}), status
    return jsonify({"ok": True, "annotation": record})


@app.get("/pddl-files/<path:rel>")
def pddl_files(rel: str):
    root = DEFAULT_PDDL_ROOT.resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        abort(404)
    if path.suffix.lower() == ".html" and path.name.endswith("_pddl.html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "</head>" in text:
            text = text.replace("</head>", PDDL_EMBED_CSS + "</head>", 1)
        if "</body>" in text:
            text = text.replace("</body>", PDDL_EMBED_JS + "</body>", 1)
        response = Response(text, mimetype="text/html; charset=utf-8")
    else:
        response = send_file(path)
    response.headers["Cache-Control"] = "no-store"
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace log annotator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument(
        "--root",
        default=str(DEFAULT_LOGS_ROOT),
        help="Directory to scan for evaluation runs",
    )
    return parser.parse_args()


def main() -> None:
    global catalog
    args = parse_args()
    catalog = make_catalog(Path(args.root))
    print(f"Scanning {catalog.logs_root}")
    print(f"Found {len(catalog.list_runs())} runs")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
