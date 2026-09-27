#!/usr/bin/env python3
"""Spawn each object_categories_filtered.csv asset and write a 360° turntable GIF."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

CSV_PATH = REPO / "data" / "hssd-hab" / "metadata" / "object_categories_filtered.csv"
DEFAULT_OUT = REPO / "data" / "hssd-hab" / "metadata" / "object_gifs"
DEFAULT_SCENE = "107734176_176000019"


def slug(text: str, max_len: int = 80) -> str:
    raw = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-._")
    return (raw or "asset")[:max_len]


def load_assets(csv_path: Path) -> List[Tuple[str, str]]:
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            aid = (row.get("id") or "").strip()
            cat = (row.get("clean_category") or "").strip()
            if aid and cat:
                rows.append((aid, cat))
    return rows


def resolve_handle(otm, asset_id: str) -> Optional[str]:
    candidates = [
        asset_id,
        f"{asset_id}.object_config.json",
        f"{asset_id}.object_config",
    ]
    for handle in candidates:
        if otm.get_library_has_handle(handle):
            return handle
    suffix = f"{asset_id}.object_config.json"
    for handle in otm.get_template_handles():
        if handle.endswith(suffix) or Path(handle).stem == asset_id:
            return handle
    return None


def gif_path(out_dir: Path, category: str, asset_id: str) -> Path:
    return out_dir / slug(category, 40) / f"{slug(asset_id)}.gif"


def load_captions(out_dir: Path) -> Dict[str, Dict[str, str]]:
    path = out_dir / "object_captions.csv"
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            aid = (row.get("id") or "").strip()
            if aid:
                out[aid] = row
    return out


def write_gallery(out_dir: Path, assets: List[Tuple[str, str]]) -> None:
    captions = load_captions(out_dir)
    by_cat: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for aid, cat in assets:
        rel = gif_path(out_dir, cat, aid).relative_to(out_dir).as_posix()
        by_cat[cat].append((aid, rel))
    cats = sorted(by_cat)
    sections = []
    for cat in cats:
        cards = []
        for aid, rel in by_cat[cat]:
            exists = (out_dir / rel).exists()
            img = (
                f'<img src="{html.escape(rel)}" alt="{html.escape(aid)}" loading="lazy">'
                if exists
                else '<div class="missing">missing</div>'
            )
            cap = captions.get(aid, {})
            name = (cap.get("name") or "").strip()
            desc = (cap.get("description") or "").strip()
            tags = (cap.get("tags") or "").strip()
            q = " ".join(
                part for part in (cat, aid, name, tags, desc) if part
            ).lower().replace('"', "")
            extra = ""
            if name:
                extra += f'<div class="name">{html.escape(name)}</div>'
            if desc:
                extra += f'<div class="desc">{html.escape(desc)}</div>'
            if tags:
                extra += f'<div class="tags">{html.escape(tags)}</div>'
            cards.append(
                f'<div class="card" data-q="{html.escape(q, quote=True)}">{img}'
                f'<div class="id">{html.escape(aid)}</div>{extra}</div>'
            )
        sections.append(
            f'<section><h2 id="{slug(cat)}">{html.escape(cat)} '
            f"<span>({len(by_cat[cat])})</span></h2>"
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Object turntable GIFs</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:#111; color:#eee; margin:0; }}
header {{ position:sticky; top:0; background:#1b1b1b; padding:12px 16px; border-bottom:1px solid #333; z-index:1; }}
input {{ width:min(520px,90vw); padding:8px; font:inherit; }}
h2 {{ margin:24px 16px 8px; font-size:1.1rem; }}
h2 span {{ color:#888; font-weight:400; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:10px; padding:0 16px 24px; }}
.card {{ background:#1e1e1e; border:1px solid #333; border-radius:8px; padding:8px; }}
.card img {{ width:100%; height:180px; object-fit:contain; background:#2a2a2a; display:block; }}
.id {{ font-size:11px; word-break:break-all; margin-top:6px; color:#888; }}
.name {{ font-size:12px; margin-top:4px; color:#eee; }}
.desc {{ font-size:11px; margin-top:3px; color:#bbb; line-height:1.3; }}
.tags {{ font-size:10px; margin-top:4px; color:#8ab4f8; }}
.missing {{ height:180px; display:flex; align-items:center; justify-content:center; color:#666; }}
.hidden {{ display:none; }}
</style></head><body>
<header>
  <div>{len(assets)} assets · {len(cats)} classes</div>
  <input id="q" placeholder="Filter class, id, tags, or description"/>
</header>
{"".join(sections)}
<script>
const q = document.getElementById('q');
q.addEventListener('input', () => {{
  const t = q.value.toLowerCase().trim();
  document.querySelectorAll('.card').forEach(c => {{
    c.classList.toggle('hidden', t && !c.dataset.q.includes(t));
  }});
  document.querySelectorAll('section').forEach(s => {{
    const any = [...s.querySelectorAll('.card')].some(c => !c.classList.contains('hidden'));
    s.classList.toggle('hidden', !any);
  }});
}});
</script>
</body></html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html_doc)


def hide_scene_geometry(sim) -> None:
    """Drop house furniture/AOs so the turntable is object-only."""
    rom = sim.get_rigid_object_manager()
    for handle in list(rom.get_object_handles()):
        try:
            rom.remove_object_by_handle(handle)
        except Exception:
            pass
    aom = sim.get_articulated_object_manager()
    for handle in list(aom.get_object_handles()):
        try:
            aom.remove_object_by_handle(handle)
        except Exception:
            pass


def capture_gif(sim, obj, out_file: Path, frames: int, size: int) -> None:
    import magnum as mn
    from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
    from PIL import Image

    dbv = DebugVisualizer(
        sim,
        resolution=(size, size),
        clear_color=mn.Color4(0.62, 0.62, 0.62, 1.0),
    )
    bb = obj.aabb
    xf = obj.transformation
    look_at = xf.transform_point(bb.center())
    bb_size = bb.size()
    extent = max(float(bb_size.x), float(bb_size.y), float(bb_size.z), 0.05)
    distance = (extent / 0.85) / math.tan(math.radians(45.0)) * 1.25
    height = extent * 0.35
    images = []
    for i in range(frames):
        theta = 2.0 * math.pi * i / frames
        look_from = mn.Vector3(
            look_at.x + distance * math.sin(theta),
            look_at.y + height,
            look_at.z + distance * math.cos(theta),
        )
        obs = dbv.get_observation(look_at=look_at, look_from=look_from)
        images.append(obs.get_image().convert("P", palette=Image.ADAPTIVE, colors=256))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_file,
        save_all=True,
        append_images=images[1:],
        duration=220,
        loop=0,
        optimize=True,
    )
    dbv.remove_dbv_agent()


def spawn_pose(sim):
    import magnum as mn

    # High above the house so walls/floor stay out of the orbit.
    return mn.Vector3(0.0, 80.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument(
        "--asset-id", action="append", help="Render only these asset IDs (repeatable)."
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    from habitat_sim.physics import MotionType

    from dataset_generation.benchmark_generation.generate_episodes import (
        default_gen_config,
        initialize_generator,
    )
    from habitat_llm.sims.metadata_interface import default_metadata_dict

    all_assets = load_assets(args.csv)
    assets = all_assets
    if args.asset_id:
        requested = set(args.asset_id)
        unknown = requested - {aid for aid, _ in all_assets}
        if unknown:
            parser.error(f"Unknown asset IDs: {', '.join(sorted(unknown))}")
        assets = [(aid, cat) for aid, cat in all_assets if aid in requested]
    if args.limit:
        assets = assets[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = args.out_dir / "failures.jsonl"

    print(f"loading scene {args.scene}", flush=True)
    gen_config = dict(default_gen_config)
    gen_config["ep_dataset_output"] = "test.json.gz"
    generator = initialize_generator(gen_config, default_metadata_dict)
    generator.initialize_fresh_scene(args.scene)
    sim = generator.sim
    hide_scene_geometry(sim)
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    pose = spawn_pose(sim)
    print(f"spawn pose {pose}  assets {len(assets)}", flush=True)

    done = 0
    skipped = 0
    failed = 0
    for i, (aid, cat) in enumerate(assets, 1):
        dest = gif_path(args.out_dir, cat, aid)
        if dest.exists() and not args.no_resume:
            skipped += 1
            print(f"[{i}/{len(assets)}] skip {cat}/{aid}", flush=True)
            continue
        handle = resolve_handle(otm, aid)
        if handle is None:
            failed += 1
            with fail_log.open("a") as f:
                f.write(json.dumps({"id": aid, "class": cat, "error": "no template"}) + "\n")
            print(f"[{i}/{len(assets)}] NO TEMPLATE {cat}/{aid}", flush=True)
            continue
        obj = None
        try:
            obj = rom.add_object_by_template_handle(handle)
            obj.motion_type = MotionType.KINEMATIC
            obj.translation = pose
            capture_gif(sim, obj, dest, args.frames, args.size)
            done += 1
            print(f"[{i}/{len(assets)}] wrote {dest}", flush=True)
        except Exception as exc:
            failed += 1
            with fail_log.open("a") as f:
                f.write(json.dumps({"id": aid, "class": cat, "error": str(exc)}) + "\n")
            print(f"[{i}/{len(assets)}] FAIL {cat}/{aid}: {exc}", flush=True)
        finally:
            if obj is not None:
                try:
                    rom.remove_object_by_handle(obj.handle)
                except Exception:
                    pass

    write_gallery(args.out_dir, all_assets)
    print(f"done wrote={done} skipped={skipped} failed={failed}", flush=True)
    print(f"gallery {args.out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
