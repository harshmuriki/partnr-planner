#!/usr/bin/env python3
"""Caption object turntable GIFs with GPT-5.6 Luna (name, tags, description)."""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import io
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

_gifs_spec = importlib.util.spec_from_file_location(
    "object_turntable_gifs",
    REPO / "scripts" / "object_turntable_gifs.py",
)
_gifs = importlib.util.module_from_spec(_gifs_spec)
assert _gifs_spec.loader is not None
_gifs_spec.loader.exec_module(_gifs)
CSV_PATH = _gifs.CSV_PATH
DEFAULT_OUT = _gifs.DEFAULT_OUT
gif_path = _gifs.gif_path
load_assets = _gifs.load_assets
write_gallery = _gifs.write_gallery

MODEL = "gpt-5.6-luna"
CAPTION_FIELDS = ["id", "clean_category", "gif", "name", "tags", "description"]
SYSTEM = (
    "You label household 3D assets from a turntable still. "
    "Return JSON only with keys name, tags, description. "
    "name: 2-6 word object name. "
    "tags: 3-6 short lowercase tags (color, material, form, use). "
    "description: one concise sentence of what it looks like. No fluff."
)
USER_TMPL = (
    "Asset id: {aid}\n"
    "Category: {cat}\n"
    "GIF file: {gif}\n"
    "Write JSON: {{\"name\":\"...\",\"tags\":[\"...\"],\"description\":\"...\"}}"
)


def gif_to_data_url(path: Path) -> str:
    im = Image.open(path)
    n = getattr(im, "n_frames", 1)
    im.seek(min(4, max(0, n - 1)))
    frame = im.convert("RGB")
    buf = io.BytesIO()
    frame.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON in model output: {raw[:200]!r}")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


def normalize_row(aid: str, cat: str, gif_rel: str, data: Dict[str, Any]) -> Dict[str, str]:
    name = str(data.get("name") or aid).strip()
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        tag_list = [str(t).strip().lower() for t in tags if str(t).strip()]
    desc = str(data.get("description") or "").strip()
    return {
        "id": aid,
        "clean_category": cat,
        "gif": gif_rel,
        "name": name,
        "tags": ", ".join(tag_list),
        "description": desc,
    }


def caption_one(client: OpenAI, aid: str, cat: str, gif: Path, gif_rel: str) -> Dict[str, str]:
    data_url = gif_to_data_url(gif)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TMPL.format(aid=aid, cat=cat, gif=gif_rel)},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    return normalize_row(aid, cat, gif_rel, parse_json(content))


def load_existing(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            aid = (row.get("id") or "").strip()
            if aid:
                out[aid] = row
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAPTION_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CAPTION_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--gif-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    out_csv = args.out_csv or (args.gif_dir / "object_captions.csv")
    load_dotenv(REPO / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    client = OpenAI()
    assets = load_assets(args.csv)
    if args.limit:
        assets = assets[: args.limit]

    existing = {} if args.no_resume else load_existing(out_csv)
    rows_by_id = dict(existing)
    lock = threading.Lock()
    fail_log = args.gif_dir / "caption_failures.jsonl"

    jobs: List[Tuple[str, str, Path, str]] = []
    skipped = 0
    missing = 0
    for aid, cat in assets:
        gif = gif_path(args.gif_dir, cat, aid)
        gif_rel = gif.relative_to(args.gif_dir).as_posix() if gif.exists() else ""
        if not gif.exists():
            missing += 1
            print(f"NO GIF {cat}/{aid}", flush=True)
            continue
        if aid in rows_by_id and not args.no_resume:
            skipped += 1
            continue
        jobs.append((aid, cat, gif, gif_rel))

    print(
        f"caption {len(jobs)}  skip {skipped}  missing_gif {missing}  model {MODEL}",
        flush=True,
    )

    done = 0
    failed = 0

    def run(job: Tuple[str, str, Path, str]) -> Tuple[str, Optional[Dict[str, str]], Optional[str]]:
        aid, cat, gif, gif_rel = job
        try:
            return aid, caption_one(client, aid, cat, gif, gif_rel), None
        except Exception as exc:
            return aid, None, str(exc)

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run, job) for job in jobs]
        for fut in as_completed(futs):
            aid, row, err = fut.result()
            if row is None:
                failed += 1
                with lock:
                    with fail_log.open("a") as f:
                        f.write(json.dumps({"id": aid, "error": err}) + "\n")
                print(f"FAIL {aid}: {err}", flush=True)
                continue
            with lock:
                rows_by_id[aid] = row
                ordered = [
                    rows_by_id[a]
                    for a, _ in load_assets(args.csv)
                    if a in rows_by_id
                ]
                write_csv(out_csv, ordered)
            done += 1
            print(
                f"[{done}/{len(jobs)}] {row['clean_category']}/{aid}  "
                f"{row['name']}  [{row['tags']}]",
                flush=True,
            )

    write_csv(
        out_csv,
        [rows_by_id[a] for a, _ in load_assets(args.csv) if a in rows_by_id],
    )
    write_gallery(args.gif_dir, load_assets(args.csv))
    print(
        f"done wrote={done} skipped={skipped} failed={failed} missing_gif={missing}",
        flush=True,
    )
    print(f"csv {out_csv}", flush=True)
    print(f"gallery {args.gif_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
