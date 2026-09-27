#!/usr/bin/env python3
"""Spawn Habitat episodes from variant spec markdown via the episode GUI API."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPECS = REPO / "baseline_evaluation_v2" / "specs"
DEFAULT_EPISODES = REPO / "baseline_evaluation_v2" / "episodes"
DEFAULT_LOG = REPO / "baseline_evaluation_v2" / "sims" / "spawn_log.jsonl"


def extract_section(md: str, heading: str) -> str:
    pat = rf"(?ms)^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    m = re.search(pat, md)
    return m.group(1).strip() if m else ""


def parse_spec(path: Path) -> dict:
    md = path.read_text()
    scene = re.search(r"scene_id:\s*(\S+)", md)
    if not scene:
        raise ValueError(f"no scene_id in {path}")
    instr = extract_section(md, "Task instruction / prompt given").strip().strip('"')
    goal = "\n\n".join(
        [
            extract_section(md, "Final expected world state"),
            extract_section(md, "Success criteria"),
            extract_section(md, "Spawn / planner notes"),
        ]
    )
    return {
        "scene_id": scene.group(1).strip(),
        "instruction": instr,
        "prompt": md,
        "goal": goal,
        "folder_label": path.stem.lower(),
        "variant": path.stem,
        "spec_path": str(path),
    }


def spawn_one(client: httpx.Client, url: str, spec: dict, timeout: float, episodes_dir: Path) -> dict:
    payload = {
        "scene_id": spec["scene_id"],
        "prompt": spec["prompt"],
        "goal": spec["goal"],
        "folder_label": spec["folder_label"],
        "instruction": spec["instruction"],
        "skip_prediviz": True,
    }
    r = client.post(url, json=payload, timeout=timeout)
    data = r.json()
    data["_http"] = r.status_code
    data["_variant"] = spec["variant"]
    data["_scene_id"] = spec["scene_id"]
    data["_spec_path"] = spec["spec_path"]
    if data.get("ok") and data.get("dataset_path"):
        # The service always writes under data/datasets/custom/<scene_id>/<run_dir_name>/ --
        # relocate into <episodes_dir>/task_N/<run_dir_name>/ so the skill-runner GUI (which
        # scans <episodes_dir>/task_N/...) can find it.
        src_dir = Path(data["dataset_path"]).resolve().parent
        task_num = spec["variant"][1]  # "T1-ACC-BASE" -> "1"
        dest_dir = (REPO / episodes_dir / f"task_{task_num}" / src_dir.name).resolve()
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(src_dir), str(dest_dir))
        shutil.copy2(spec["spec_path"], dest_dir / "spec.md")
        data["dataset_path"] = str(dest_dir / "dataset.json.gz")
        data["spec_copy"] = str(dest_dir / "spec.md")
        if data.get("prediviz_path"):
            data["prediviz_path"] = str(dest_dir / Path(data["prediviz_path"]).name)
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:5000")
    p.add_argument("--task", action="append", default=[], help="T1 T2 ... (repeatable)")
    p.add_argument("--variant", default="", help="e.g. ACC-BASE; default all")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument(
        "--specs-dir",
        type=Path,
        default=DEFAULT_SPECS,
        help="directory containing T{n}/ spec subfolders (default: baseline_evaluation_v2/specs)",
    )
    p.add_argument(
        "--episodes-dir",
        type=Path,
        default=DEFAULT_EPISODES,
        help="where spawned episodes land, as task_N/ subfolders (default: baseline_evaluation_v2/episodes)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="spawn log path (default: <episodes-dir>/../sims/spawn_log.jsonl)",
    )
    args = p.parse_args()

    specs_dir = args.specs_dir
    episodes_dir = args.episodes_dir
    log_path = args.log or (episodes_dir.parent / "sims" / "spawn_log.jsonl")

    tasks = args.task or ["T1", "T2"]
    specs = []
    for task in tasks:
        d = specs_dir / task
        paths = sorted(d.glob(f"{task}-*.md"))
        if args.variant:
            paths = [d / f"{task}-{args.variant}.md"]
        for path in paths:
            if path.exists():
                specs.append(parse_spec(path))
    if not specs:
        raise SystemExit("no spec files matched")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    url = args.host.rstrip("/") + "/api/generate"
    print(f"spawning {len(specs)} specs via {url}", flush=True)
    with httpx.Client() as client:
        for i, spec in enumerate(specs, 1):
            print(
                f"[{i}/{len(specs)}] {spec['variant']} scene {spec['scene_id']}",
                flush=True,
            )
            t0 = time.time()
            try:
                data = spawn_one(client, url, spec, args.timeout, episodes_dir)
            except Exception as exc:
                data = {
                    "ok": False,
                    "error": str(exc),
                    "_variant": spec["variant"],
                    "_scene_id": spec["scene_id"],
                    "_spec_path": spec["spec_path"],
                }
            data["_elapsed_s"] = round(time.time() - t0, 1)
            with log_path.open("a") as f:
                f.write(json.dumps(data) + "\n")
            if data.get("ok"):
                print(f"  ok {data.get('dataset_path')}", flush=True)
            else:
                print(f"  FAIL {data.get('error')}", flush=True)


if __name__ == "__main__":
    main()
