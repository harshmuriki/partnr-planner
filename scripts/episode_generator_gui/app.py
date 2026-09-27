#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GUI_DIR))
os.chdir(PROJECT_ROOT)

from generation_service import EpisodeGenerationService, list_available_scenes

app = Flask(__name__, template_folder="templates")
service = EpisodeGenerationService()
CUSTOM_ROOT = (PROJECT_ROOT / "data" / "datasets" / "custom").resolve()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scenes")
def scenes():
    return jsonify({"scenes": list_available_scenes()})


@app.route("/api/generate", methods=["POST"])
def generate():
    payload = request.get_json(force=True, silent=True) or {}
    scene_id = str(payload.get("scene_id", "")).strip()
    prompt = str(payload.get("prompt", "")).strip()
    goal = str(payload.get("goal", "")).strip()
    folder_label = str(payload.get("folder_label", "")).strip() or None
    instruction = str(payload.get("instruction", "")).strip() or None
    skip_prediviz = bool(payload.get("skip_prediviz", False))
    if not scene_id or not prompt:
        return jsonify({"ok": False, "error": "scene_id and prompt are required."}), 400
    try:
        result = service.generate(
            scene_id,
            prompt,
            goal,
            folder_label=folder_label,
            instruction=instruction,
            skip_prediviz=skip_prediviz,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/download")
def download():
    rel = request.args.get("path", "")
    full = (PROJECT_ROOT / rel).resolve()
    if not str(full).startswith(str(CUSTOM_ROOT)) or not full.exists():
        abort(404)
    return send_file(full, as_attachment=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
