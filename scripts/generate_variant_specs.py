#!/usr/bin/env python3
"""Generate furniture-grounded spec markdown for each New Tasks × 20-variant cell."""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
import httpx
from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.chdir(REPO)

PROMPT_PATH = REPO / "baseline_evaluation_v2" / "spec_generation_prompt.txt"
OUT_DIR = REPO / "baseline_evaluation_v2" / "specs"
XLSX_PATH = Path("/home/harshmuriki/Downloads/Magic Table.xlsx")
MODEL = "gpt-6-astra"

VARIANT_IDS = [
    "ACC-BASE",
    "ACC-SUB",
    "ACC-ABS",
    "ACC-CON",
    "ACC-DIS",
    "ACC-AMB",
    "ACC-ROOM",
    "ACC-CAND",
    "INC-BASE",
    "INC-SUB",
    "INC-ABS",
    "INC-CON",
    "INC-DIS",
    "INC-AMB",
    "OUT-BASE",
    "OUT-SUB",
    "OUT-ABS",
    "OUT-CON",
    "OUT-DIS",
    "OUT-AMB",
]

TASK_HEADERS = [
    "id",
    "scene_id",
    "task",
    "required_objects_to_spawn",
    "uncertainty_targets",
    "suitable_substitutes",
    "atomic_actions",
    "memory_accurate",
    "memory_missing",
    "memory_outdated",
    "avail_target_exists",
    "avail_substitute",
    "avail_none",
    "containment_on_surface",
    "containment_inside",
    "distractors_none",
    "distractors_present",
    "instruction_full",
    "instruction_underspecified",
    "loc_exact",
    "loc_room",
    "loc_candidate_rooms",
]


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()


def load_tasks(xlsx: Path) -> List[Dict[str, str]]:
    wb = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)
    ws = wb["New Tasks"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    tasks = []
    for row in rows[3:]:
        tid = _cell(row[0] if row else "")
        if not re.fullmatch(r"T[1-7]", tid):
            continue
        rec = {}
        for i, key in enumerate(TASK_HEADERS):
            rec[key] = _cell(row[i]) if i < len(row) else ""
        tasks.append(rec)
    return tasks


def allowed_object_classes() -> List[str]:
    import csv

    path = REPO / "data" / "hssd-hab" / "metadata" / "object_categories_filtered.csv"
    cats = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = (row.get("clean_category") or "").strip()
            if c:
                cats.add(c)
    return sorted(cats)


def format_house_furniture_local(scene_info: Dict[str, Any]) -> str:
    descriptions = scene_info.get("recep_to_description") or {}
    lines: List[str] = []
    for room, furniture_ids in (scene_info.get("furniture") or {}).items():
        lines.append(f"{room}:")
        seen = set()
        for furn_id in furniture_ids:
            if furn_id in seen:
                continue
            seen.add(furn_id)
            desc = descriptions.get(furn_id, "")
            lines.append(f"  - {furn_id}: {desc}" if desc else f"  - {furn_id}")
    rooms = []
    for room in scene_info.get("furniture") or {}:
        if room not in rooms:
            rooms.append(room)
    return "rooms: " + ", ".join(rooms) + "\n" + "\n".join(lines)


_SCENE_SVC = None


def furniture_catalog(scene_id: str) -> str:
    cache = REPO / "data" / "datasets" / "custom" / "scene_info" / f"{scene_id}.json"
    if cache.exists():
        with open(cache) as f:
            info = json.load(f)
        return format_house_furniture_local(info)

    from scripts.episode_generator_gui.generation_service import (
        EpisodeGenerationService,
        format_house_furniture,
    )

    global _SCENE_SVC
    if _SCENE_SVC is None:
        _SCENE_SVC = EpisodeGenerationService()
    info = _SCENE_SVC.get_scene_info(scene_id)
    rooms = ", ".join(info.get("all_rooms") or [])
    furn = format_house_furniture(info)
    return f"rooms: {rooms}\n{furn}"


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def split_markdown_specs(text: str, task_id: str) -> Dict[str, Any]:
    pattern = rf"(?m)^# ({re.escape(task_id)}-[A-Z]+-[A-Z]+)\s*$"
    matches = list(re.finditer(pattern, text))
    files = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        files.append(
            {
                "filename": f"{m.group(1)}.md",
                "content": text[m.start() : end].strip(),
            }
        )
    return {"files": files}


def build_batch_user_message(task: Dict[str, str], catalog: str, classes: str) -> str:
    variants = ", ".join(f"{task['id']}-{v}" for v in VARIANT_IDS)
    return f"""TASK ROW (New Tasks tab)
{json.dumps(task, indent=2)}

Emit ALL 20 variants in one JSON object. Filenames:
{variants}

SCENE FURNITURE CATALOG ({task['scene_id']})
{catalog}

ALLOWED OBJECT CLASSES
{classes}

Return ONLY this JSON (no markdown fences, no extra text):
{{"files":[{{"filename":"{task['id']}-ACC-BASE.md","content":"<full markdown spec>"}}]}}
Exactly 20 objects in files, one per variant above. Each content is a complete spec. Uncertainty being tested lists ONLY the matrix-cell axes.
"""


def build_user_message(
    task: Dict[str, str], catalog: str, classes: str, variant_id: str
) -> str:
    return f"""TASK ROW (New Tasks tab)
{json.dumps(task, indent=2)}

Write ONE spec: {task['id']}-{variant_id}
Output ONLY the markdown spec. No JSON. No fences.

SCENE FURNITURE CATALOG ({task['scene_id']})
{catalog}

ALLOWED OBJECT CLASSES
{classes}
"""


def call_model(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        timeout=600.0,
    )
    return resp.choices[0].message.content or ""


def write_files(task_id: str, payload: Dict[str, Any]) -> List[str]:
    dest = OUT_DIR / task_id
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    files = payload.get("files") or []
    expected = {f"{task_id}-{v}.md" for v in VARIANT_IDS}
    got = set()
    for item in files:
        name = Path(item.get("filename") or "").name
        content = item.get("content") or ""
        if not name.endswith(".md"):
            name = name + ".md"
        got.add(name)
        path = dest / name
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        written.append(str(path))
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        print(f"WARNING {task_id}: missing={missing} extra={extra}", flush=True)
    return written


COL = {
    "BASE": None,
    "SUB": ("Object availability", "Substitute Available"),
    "ABS": ("Object availability", "No Suitable Object"),
    "CON": ("Containment", "Inside Closed Receptacle"),
    "DIS": ("Distractors", "Present"),
    "AMB": ("Instruction", "Underspecified"),
    "ROOM": ("Localization", "Room Known"),
    "CAND": ("Localization", "Candidate Rooms"),
}
MEM = {"ACC": "Accurate", "INC": "Incomplete", "OUT": "Outdated"}
UNCERT_RE = re.compile(r"## Uncertainty being tested\n.*?(?=\n## )", re.S)


def compact_uncertainty(task_id: str) -> None:
    dest = OUT_DIR / task_id
    for p in dest.glob(f"{task_id}-*.md"):
        parts = p.stem.split("-")
        if len(parts) != 3:
            continue
        _, mem, col = parts
        lines = [f"- Internal robot memory: {MEM[mem]}"]
        extra = COL[col]
        if extra:
            lines.append(f"- {extra[0]}: {extra[1]}")
        repl = "## Uncertainty being tested\n" + "\n".join(lines) + "\n"
        text = p.read_text()
        new, n = UNCERT_RE.subn(repl, text, count=1)
        if n == 1:
            p.write_text(new)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing")

    system = PROMPT_PATH.read_text()
    args = [a for a in sys.argv[1:] if a != "--batch"]
    batch = "--batch" in sys.argv[1:]
    if batch:
        system = (
            "You output one JSON object only: "
            '{"files":[{"filename":"T2-ACC-BASE.md","content":"...markdown..."}]}. '
            "The content of each file follows the spec skeleton. "
            "Do not emit markdown outside JSON.\n\n"
            + system.replace(
                "You write one PARTNR/Habitat-LLM evaluation spec file.\n"
                "Output ONLY the markdown spec. No JSON. No fences. No preamble.",
                "Each files[].content is one full markdown spec (no JSON inside content).",
            )
        )
    tasks = load_tasks(XLSX_PATH)
    if len(tasks) != 7:
        raise SystemExit(f"expected 7 tasks, got {len(tasks)}: {[t['id'] for t in tasks]}")

    classes = ", ".join(allowed_object_classes())
    client = OpenAI(
        api_key=api_key,
        http_client=httpx.Client(timeout=600.0, follow_redirects=True),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    args = [a for a in sys.argv[1:] if a != "--batch"]
    batch = "--batch" in sys.argv[1:]
    only = args
    for task in tasks:
        if only and task["id"] not in only:
            continue
        print(f"=== {task['id']} scene {task['scene_id']} furniture ===", flush=True)
        catalog = furniture_catalog(task["scene_id"])
        dest = OUT_DIR / task["id"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "_furniture_catalog.txt").write_text(catalog)
        if batch:
            user = build_batch_user_message(task, catalog, classes)
            (dest / "_gpt_user_message.txt").write_text(user)
            print(f"=== {task['id']} calling {MODEL} for all 20 variants ===", flush=True)
            raw = call_model(client, system, user)
            (dest / "_gpt_raw.json").write_text(raw or "")
            if not (raw or "").strip():
                raise SystemExit(f"empty response for {task['id']}")
            try:
                payload = extract_json(raw)
            except (json.JSONDecodeError, ValueError):
                payload = split_markdown_specs(raw, task["id"])
            n = len(payload.get("files") or [])
            print(f"parsed {n} files", flush=True)
            if n < 20:
                raise SystemExit(f"{task['id']}: expected 20 files, got {n}")
            write_files(task["id"], payload)
            compact_uncertainty(task["id"])
            print(f"wrote 20 files for {task['id']}", flush=True)
            continue
        (dest / "_gpt_user_message.txt").write_text(
            build_user_message(task, catalog, classes, "ACC-BASE")
        )
        raw_path = dest / "_gpt_raw.json"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            recovered = split_markdown_specs(raw_path.read_text(), task["id"])
            if recovered["files"]:
                write_files(task["id"], recovered)
                compact_uncertainty(task["id"])
        written = 0
        for i, variant_id in enumerate(VARIANT_IDS, 1):
            out_path = dest / f"{task['id']}-{variant_id}.md"
            if out_path.exists() and out_path.stat().st_size > 100:
                print(f"skip {out_path.name}", flush=True)
                written += 1
                continue
            print(
                f"=== {task['id']} {i}/20 {MODEL} {task['id']}-{variant_id} ===",
                flush=True,
            )
            user = build_user_message(task, catalog, classes, variant_id)
            raw = call_model(client, system, user)
            (dest / f"_{variant_id}_raw.md").write_text(raw or "")
            if not (raw or "").strip():
                raise SystemExit(f"empty response for {task['id']}-{variant_id}")
            payload = split_markdown_specs(raw, task["id"])
            if not payload["files"]:
                payload = {
                    "files": [
                        {
                            "filename": f"{task['id']}-{variant_id}.md",
                            "content": raw.strip(),
                        }
                    ]
                }
            write_files(task["id"], payload)
            compact_uncertainty(task["id"])
            written += 1
            print(f"wrote {out_path}", flush=True)
        print(f"done {task['id']} ({written}/20)", flush=True)

    print("done")


if __name__ == "__main__":
    main()
