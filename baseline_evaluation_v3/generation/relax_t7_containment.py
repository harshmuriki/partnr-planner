#!/usr/bin/env python3
"""T7: require both toys in wardrobe_2, and allow the plate in any cabinet.

Replaces the two toy propositions with one is_inside targeting wardrobe_2
(number=2, is_same_receptacle=True), in runtime room closet_1 (generation
room closet_0). The plate proposition accepts every cabinet in the apartment.
"""
import argparse
import json
import shutil
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
EPISODES = V3 / "episodes/task_7"
TOYS = ["FIRE_ENGINE_:0000", "b3e8be210978ec373be6eb5fcffec36c6a9712e6_:0000"]
PLATE = "Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring_:0000"
WARDROBES = [
    "f19f4eb60204eef42a4dc3b073ef9006916a2c0b_:0000",
]
CABINETS = [
    "33c1689bc5fb3b6f599f261ad069d9510804deb9_:0000",
    "33c1689bc5fb3b6f599f261ad069d9510804deb9_:0001",
    "424d7e7d258370736e6cfdbad2b253e521f80c79_:0000",
    "42c2fce42ed600de5f0c6919ba47fe97e60ab673_:0000",
    "42c2fce42ed600de5f0c6919ba47fe97e60ab673_:0001",
    "6acafe4f37b17ff0c33d22c8dc3fede1bddc67de_:0000",
    "77af27c6a414131f13d2aac55153720028fd4ed5_:0000",
    "a19b47699883e5841599452274d671a1bb2c775e_:0000",
    "bdab4ca579230f7652590bc19fa00a35701857ef_:0000",
]


def relax(episode):
    props = episode["evaluation_propositions"]
    toy_props = [p for p in props if p["function_name"] == "is_inside" and p["args"]["object_handles"][0] in TOYS]
    plate_props = [p for p in props if p["function_name"] == "is_inside" and p["args"]["object_handles"] == [PLATE]]
    assert len(toy_props) == 2 and len(plate_props) == 1, episode["episode_id"]
    assert sorted(p["args"]["object_handles"][0] for p in toy_props) == sorted(TOYS)
    rest = [p for p in props if p not in toy_props and p not in plate_props]
    new = [
        {"function_name": "is_inside", "args": {
            "object_handles": TOYS, "receptacle_handles": WARDROBES,
            "number": 2, "is_same_receptacle": True}},
        {"function_name": "is_inside", "args": {
            "object_handles": [PLATE], "receptacle_handles": CABINETS, "number": 1}},
        *rest,
    ]
    assert not episode["evaluation_proposition_dependencies"], episode["episode_id"]
    assert all(not c["args"].get("dag_edges") for c in episode["evaluation_constraints"]
               if c["type"] == "TemporalConstraint"), episode["episode_id"]
    n = len(new)
    episode["evaluation_propositions"] = new
    episode["evaluation_constraints"] = [
        {"type": "TemporalConstraint", "args": {"dag_edges": [], "n_propositions": n}},
        {"type": "TerminalSatisfactionConstraint",
         "args": {"proposition_indices": list(range(n)), "n_propositions": n}},
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", type=Path, help="Write edited copies here instead of in place")
    args = parser.parse_args()
    for variant in sorted(EPISODES.iterdir()):
        data = json.loads((variant / "dataset.json").read_text())
        for episode in data["episodes"]:
            relax(episode)
        if args.stage:
            out = args.stage / variant.name
            out.mkdir(parents=True, exist_ok=False)
            for name in ("scene_info.json", "initial_robot_memory.json", "review.json", "spec.md"):
                shutil.copy2(variant / name, out / name)
        else:
            out = variant
            backup = variant / "dataset.pre_relaxed_containment.json"
            assert not backup.exists(), backup
            shutil.copy2(variant / "dataset.json", backup)
        (out / "dataset.json").write_text(json.dumps(data))
        print(variant.name, "->", out / "dataset.json")


if __name__ == "__main__":
    main()
