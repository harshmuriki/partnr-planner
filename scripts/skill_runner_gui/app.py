#!/usr/bin/env python3
"""
Skill Runner localhost sandbox.

Usage:
    python scripts/skill_runner_gui/app.py [--host 127.0.0.1] [--port 5051]

Then open http://127.0.0.1:5051
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GUI_DIR))
os.chdir(PROJECT_ROOT)

from session import SKILLS, SkillRunnerSession

app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
session: Optional[SkillRunnerSession] = None

EPISODES_ROOT = PROJECT_ROOT / "baseline_evaluation_v3" / "episodes"
CUSTOM_DATASETS = PROJECT_ROOT / "data" / "datasets" / "custom"
TASK_FOLDER_RE = re.compile(
    r"^t([1-7])-([a-z0-9-]+)_(\d{8}_\d{6})$",
    re.I,
)
AXIS_ORDER = {"ACC": 0, "INC": 1, "OUT": 2}
KIND_ORDER = {
    "BASE": 0,
    "SUB": 1,
    "ABS": 2,
    "CON": 3,
    "DIS": 4,
    "AMB": 5,
    "ROOM": 6,
    "CAND": 7,
}
PLACEHOLDER_JPEG = None


def get_session() -> SkillRunnerSession:
    global session
    if session is None:
        session = SkillRunnerSession()
    return session


def _placeholder_jpeg() -> bytes:
    global PLACEHOLDER_JPEG
    if PLACEHOLDER_JPEG is not None:
        return PLACEHOLDER_JPEG
    import cv2
    import numpy as np

    img = np.zeros((240, 640, 3), dtype=np.uint8)
    img[:] = (36, 39, 58)  # dark panel BGR-ish after encode from BGR
    cv2.putText(
        img,
        "Waiting for frames...",
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 200, 245),
        2,
    )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    PLACEHOLDER_JPEG = buf.tobytes() if ok else b""
    return PLACEHOLDER_JPEG


def _variant_meta(folder_name: str) -> Optional[Dict[str, str]]:
    m = TASK_FOLDER_RE.match(folder_name)
    if not m:
        return None
    task_n, variant, stamp = m.group(1), m.group(2).upper(), m.group(3)
    return {
        "group": f"T{task_n}",
        "label": f"T{task_n}-{variant}",
        "stamp": stamp,
        "task": int(task_n),
    }


def _episode_sort_key(ep: Dict[str, Any]) -> Any:
    m = re.match(r"^T(\d+)-([A-Z]+)-([A-Z]+)$", ep["label"])
    if not m:
        return (99, 99, 99, ep["label"])
    return (
        int(m.group(1)),
        AXIS_ORDER.get(m.group(2), 99),
        KIND_ORDER.get(m.group(3), 99),
        ep["label"],
    )


def _runtime_yaml(folder: Path) -> Optional[str]:
    yaml_files = sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")))
    return str(yaml_files[0]) if yaml_files else None


def _record_for_dataset(path: Path) -> Dict[str, Any]:
    folder = path.parent
    meta = _variant_meta(folder.name)
    if meta:
        return {
            "label": meta["label"],
            "group": meta["group"],
            "path": str(path),
            "folder": str(folder),
            "runtime_yaml": _runtime_yaml(folder),
            "stamp": meta["stamp"],
            "task": meta["task"],
        }
    return {
        "label": f"{folder.name}/{path.name}",
        "group": folder.parent.name,
        "path": str(path),
        "folder": str(folder),
        "runtime_yaml": _runtime_yaml(folder),
        "stamp": "",
        "task": 99,
    }


def _find_dataset_files(root: Path, variant_named_only: bool) -> List[Path]:
    if not root.exists():
        return []
    found: List[Path] = []
    for path in root.rglob("dataset.json.gz"):
        if "prediviz" in path.parts:
            continue
        if variant_named_only and not TASK_FOLDER_RE.match(path.parent.name):
            continue
        found.append(path)
    if found:
        return found
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(folder.glob("*.json.gz")) + sorted(folder.glob("*.json")):
            if path.name in {"scene_info.json", "run_data.json"}:
                continue
            if path.suffix == ".json" and Path(str(path) + ".gz").exists():
                continue
            found.append(path)
    return found


def _dedupe_latest(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_label: Dict[str, Dict[str, Any]] = {}
    leftovers: List[Dict[str, Any]] = []
    for ep in episodes:
        if TASK_FOLDER_RE.match(Path(ep["folder"]).name):
            prev = by_label.get(ep["label"])
            if prev is None or str(ep.get("stamp") or "") > str(prev.get("stamp") or ""):
                by_label[ep["label"]] = ep
        else:
            leftovers.append(ep)
    return list(by_label.values()) + leftovers


def scan_episodes(
    root: Optional[Path] = None,
    include_custom: bool = False,
) -> List[Dict[str, Any]]:
    if root is not None:
        episodes = [_record_for_dataset(p) for p in _find_dataset_files(root, False)]
    else:
        episodes = [
            _record_for_dataset(p) for p in _find_dataset_files(EPISODES_ROOT, False)
        ]
        if include_custom:
            episodes.extend(
                _record_for_dataset(p)
                for p in _find_dataset_files(CUSTOM_DATASETS, True)
            )
    seen = set()
    unique: List[Dict[str, Any]] = []
    for ep in _dedupe_latest(episodes):
        key = str(Path(ep["path"]).resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(ep)
    unique.sort(key=_episode_sort_key)
    return unique


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/episodes")
def api_episodes():
    custom = request.args.get("root")
    if custom:
        root = Path(custom)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return jsonify({"episodes": scan_episodes(root), "root": str(root)})
    return jsonify(
        {
            "episodes": scan_episodes(),
            "root": str(EPISODES_ROOT),
        }
    )


@app.route("/api/status")
def api_status():
    return jsonify(get_session().get_status())


@app.route("/api/skills")
def api_skills():
    # Preserve declaration order (Flask jsonify may reorder plain dicts).
    return jsonify(
        {
            "skills": [{"name": k, "help": v} for k, v in SKILLS.items()],
            "skills_map": SKILLS,
        }
    )


@app.route("/api/load", methods=["POST"])
def api_load():
    payload = request.get_json(force=True, silent=True) or {}
    data_path = str(payload.get("data_path", "")).strip()
    if not data_path:
        return jsonify({"ok": False, "error": "data_path is required"}), 400
    episode_id = payload.get("episode_id")
    episode_index = payload.get("episode_index")
    runtime_config_path = payload.get("runtime_config_path")
    if episode_id is not None:
        episode_id = str(episode_id)
    if episode_index is not None:
        try:
            episode_index = int(episode_index)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "episode_index must be int"}), 400

    sess = get_session()
    if sess.status in ("loading", "running"):
        return jsonify({"ok": False, "error": f"Busy ({sess.status})"}), 409

    try:
        result = sess.load_episode(
            data_path=data_path,
            episode_id=episode_id,
            episode_index=episode_index,
            runtime_config_path=runtime_config_path or None,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/run_skill", methods=["POST"])
def api_run_skill():
    payload = request.get_json(force=True, silent=True) or {}
    skill = str(payload.get("skill", "")).strip()
    target = str(payload.get("target", "")).strip()
    try:
        agent_index = int(payload.get("agent_index", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "agent_index must be int"}), 400

    if not skill or not target:
        return jsonify({"ok": False, "error": "skill and target are required"}), 400

    sess = get_session()
    if not sess.loaded:
        return jsonify({"ok": False, "error": "Load an episode first"}), 400
    if sess.status in ("loading", "running"):
        return jsonify({"ok": False, "error": f"Busy ({sess.status})"}), 409

    try:
        result = sess.run_skill(skill=skill, agent_index=agent_index, target=target)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/inspect")
def api_inspect():
    kind = request.args.get("kind", "entities")
    name = request.args.get("name") or None
    sess = get_session()
    if not sess.loaded:
        return jsonify({"error": "No episode loaded"}), 400
    if sess.status in ("loading", "running"):
        return jsonify({"error": f"Busy ({sess.status})"}), 409
    try:
        return jsonify(sess.inspect(kind=kind, name=name))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/frame.jpg")
def api_frame():
    jpeg = get_session().get_latest_jpeg()
    if jpeg is None:
        jpeg = _placeholder_jpeg()
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/api/stream")
def api_stream():
    sess = get_session()
    boundary = "frame"

    def generate():
        while True:
            jpeg = sess.get_latest_jpeg() or _placeholder_jpeg()
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1.0 / 15.0)

    return Response(
        generate(),
        mimetype=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.route("/api/videos")
def api_videos():
    return jsonify({"videos": get_session().list_videos()})


@app.route("/api/video/<path:relpath>")
def api_video(relpath: str):
    sess = get_session()
    full = (sess.results_dir / relpath).resolve()
    if not str(full).startswith(str(sess.results_dir.resolve())) or not full.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(full, mimetype="video/mp4")


@app.route("/api/exit", methods=["POST"])
def api_exit():
    sess = get_session()
    if sess.status in ("loading", "running"):
        return jsonify({"ok": False, "error": f"Busy ({sess.status})"}), 409
    try:
        result = sess.exit_session()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    def _stop_server():
        time.sleep(0.4)
        os._exit(0)

    threading.Thread(target=_stop_server, daemon=True).start()
    return jsonify(result)


@app.route("/api/save", methods=["POST"])
def api_save():
    sess = get_session()
    if not sess.loaded:
        return jsonify({"ok": False, "error": "Load an episode first"}), 400
    if sess.status in ("loading", "running"):
        return jsonify({"ok": False, "error": f"Busy ({sess.status})"}), 409
    try:
        return jsonify(sess.save_run())
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def main():
    parser = argparse.ArgumentParser(description="Skill Runner localhost sandbox")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    args = parser.parse_args()
    get_session()
    print(f"Skill Runner GUI → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
