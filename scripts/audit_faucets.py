"""Audit the faucets in every baseline_evaluation_v3 scene.

Filling needs the agent within 1.5 m of an object carrying a "faucets" marker set
(habitat_llm/utils/sim.py: get_faucet_points). Only furniture that keeps at least one
active receptacle becomes a node in the runtime world graph, so only those faucet
objects can be navigated to by name; the rest count for proximity but are unnamed.

Run from the repository root:  python3 scripts/audit_faucets.py
"""
import csv
import gzip
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HSSD = REPO / "data/hssd-hab"
EPISODES = REPO / "baseline_evaluation_v3/episodes"
OVERRIDES = REPO / "habitat_llm/sims/scene_override_data"


def faucet_assets():
    """Asset ids whose object/ao config declares a "faucets" marker set."""
    ids = set()
    for config in list(HSSD.glob("objects/**/*.object_config.json")) + list(
        HSSD.glob("urdf/*/*.ao_config.json")
    ):
        if '"faucets"' in config.read_text():
            ids.add(config.name.split(".")[0])
    return ids


def asset_names():
    with open(HSSD / "metadata/fpmodels-with-decomposed.csv") as f:
        return {row["id"]: row["name"] for row in csv.DictReader(f)}


def scene_id(variant_dir):
    dataset = json.loads(gzip.decompress((variant_dir / "dataset.json.gz").read_bytes()))
    return dataset["episodes"][0]["scene_id"]


def receptacle_filter(scene):
    """The filter the simulator actually applies, project override first."""
    override = OVERRIDES / f"{scene}.rec_filter.json"
    path = override if override.exists() else HSSD / f"scene_filter_files/{scene}.rec_filter.json"
    return json.loads(path.read_text()), "project override" if override.exists() else "dataset"


def regions(scene):
    path = HSSD / f"semantics/scenes/{scene}.semantic_config.json"
    return json.loads(path.read_text())["region_annotations"] if path.exists() else []


def region_of(point, annotations):
    x, y, z = point
    for region in annotations:
        low, high = region["min_bounds"], region["max_bounds"]
        if not low[1] - 0.5 <= y <= high[1] + 0.5:
            continue
        poly = [(p[0], p[2]) for p in region["poly_loop"]]
        inside, previous = False, len(poly) - 1
        for index, (xi, zi) in enumerate(poly):
            xj, zj = poly[previous]
            if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi:
                inside = not inside
            previous = index
        if inside:
            return region["name"]
    return "unassigned"


def instances(scene):
    """Every instance handle in the scene, as the simulator names it."""
    data = json.loads((HSSD / f"scenes-partnr-filtered/{scene}.scene_instance.json").read_text())
    placed, counts = {}, {}
    for key in ("object_instances", "articulated_object_instances"):
        for item in data.get(key, []):
            template = item["template_name"]
            index = counts.get(template, 0)
            counts[template] = index + 1
            placed[f"{template}_:{index:04}"] = (item["translation"], key.split("_")[0])
    return placed


def main():
    faucets, names = faucet_assets(), asset_names()
    scenes = {}
    for scene_info in sorted(EPISODES.glob("task_*/*/scene_info.json")):
        variant = scene_info.parent
        scenes.setdefault(scene_id(variant), []).append(variant)

    for scene, variants in sorted(scenes.items(), key=lambda item: item[1][0].parent.name):
        variants.sort()
        info = json.loads((variants[0] / "scene_info.json").read_text())
        handle_to_name = {handle: name for name, handle in info["receptacle_to_handle"].items()}
        furniture_room = {f: room for room, items in info["furniture"].items() for f in items}
        rec_filter, source = receptacle_filter(scene)
        active = {}
        for rec in rec_filter["active"]:
            handle, receptacle = rec.split("|")
            active.setdefault(handle, []).append(receptacle)
        annotations, placed = regions(scene), instances(scene)
        tasks = sorted({v.parent.name for v in variants})
        print(f"\n=== {scene} — {', '.join(tasks)} ({len(variants)} variants, rec filter: {source}) ===")
        for handle, (translation, kind) in placed.items():
            if re.split(r"_part_|_:", handle)[0] not in faucets:
                continue
            entity = handle_to_name.get(handle)
            room = furniture_room.get(entity) or region_of(translation, annotations)
            receptacles = active.get(handle, [])
            asset = names.get(re.split(r"_part_|_:", handle)[0], "?")
            if entity and receptacles:
                print(f"  USABLE   {entity:11} {room:24} {asset[:44]:46} {kind}")
                for receptacle in receptacles:
                    print(f"           └ {receptacle}")
            else:
                why = "no active receptacle, so no world-graph node" if not receptacles else "no furniture name"
                print(f"  unusable {'—':11} {room:24} {asset[:44]:46} {kind} ({why})")


if __name__ == "__main__":
    main()
