"""Add Task 6's completed microwave Wait check without resampling scenes.

Run from the repository root with python -m
dataset_generation.benchmark_generation.add_task6_wait_check.
Only current v3 datasets/specifications are updated; saved runs are untouched.
"""

import argparse
import copy
import gzip
import json
from pathlib import Path

from dataset_generation.benchmark_generation.audit_variant_specs import parse_spec


CHECK = "is_waited_inside"
OLD_NOTE = (
    "Heating has no skill or object state, so it is scored as a microwave visit: "
    "bread_0 must be placed inside microwave_0 (kitchen_0) before it is placed "
    "on table_11, and it does not need to remain in the microwave."
)
NEW_NOTE = (
    "Heating has no object state, so evaluation requires bread_0 to be placed "
    "inside microwave_0 (kitchen_0), followed by a completed Wait action while "
    "the bread remains inside throughout, before delivery to table_11. The "
    "existing Wait skill duration is unchanged; microwave power and door state "
    "are not additional requirements. Bread need not remain in the microwave "
    "at episode end."
)


def add_check(episode):
    """Append one intermediate criterion, preserving existing proposition IDs."""
    props = episode["evaluation_propositions"]
    if any(p["function_name"] == CHECK for p in props):
        return
    bread = episode["info"]["variant_spec"]["entity_handles"]["bread_0"]
    inside = [p for p in props if p["function_name"] == "is_inside"
              and p["args"]["object_handles"] == [bread]]
    delivery = [i for i, p in enumerate(props) if p["function_name"] == "is_on_top"
                and p["args"]["object_handles"] == [bread]]
    if len(inside) != 1 or len(delivery) != 1:
        raise ValueError(f"Unexpected bread criteria: {episode['episode_id']}")
    wait_index = len(props)
    props.append({"function_name": CHECK, "args": copy.deepcopy(inside[0]["args"])})
    episode["evaluation_proposition_dependencies"].append({
        "proposition_indices": delivery, "depends_on": [wait_index],
        "relation_type": "after_satisfied", "dependency_mode": "all",
    })
    for constraint in episode["evaluation_constraints"]:
        if "n_propositions" in constraint["args"]:
            constraint["args"]["n_propositions"] = len(props)


def write_json(path, data, compact=False):
    text = json.dumps(data, indent=None if compact else 2) + "\n"
    path.write_text(text)
    compressed = path.with_suffix(path.suffix + ".gz")
    if compressed.exists():
        compressed.write_bytes(gzip.compress(text.encode(), mtime=0))


def migrate(root):
    audit_path = root / "generation/audit.json"
    index_path = root / "generation/episodes.json"
    audit = json.loads(audit_path.read_text())
    index = json.loads(index_path.read_text())
    updated = {}
    for src in sorted((root / "specs/T6").glob("T6-*.md")):
        text = src.read_text()
        if CHECK not in text:
            text = text.replace(
                "- is_inside(bread_0, microwave_0)\n",
                "- is_inside(bread_0, microwave_0)\n"
                "- is_waited_inside(bread_0, microwave_0)\n"
                "- order: is_waited_inside(bread_0, microwave_0) before is_on_top(bread_0, table_11)\n",
                1,
            ).replace(OLD_NOTE, NEW_NOTE)
        src.write_text(text)
        spec = parse_spec(src)
        audit["specs"][src.stem] = spec
        item = index["variants"][src.stem]
        dest = root / item["dataset"].removesuffix(".gz")
        data = json.loads(dest.read_text())
        for episode in data["episodes"]:
            add_check(episode)
            episode["info"]["variant_spec"]["sha256"] = spec["sha256"]
            updated[episode["episode_id"]] = episode
        write_json(dest, data, compact=True)
        (dest.parent / "spec.md").write_text(text)
        review_path = root / item["review"]
        review = json.loads(review_path.read_text())
        for metadata in (review, item):
            metadata["spec_sha256"] = spec["sha256"]
            metadata.pop("runtime_verified", None)
        write_json(review_path, review)
    write_json(audit_path, audit)
    write_json(index_path, index)
    aggregate_path = root / "generation/review_dataset.json"
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
        for episode in aggregate["episodes"]:
            current = updated.get(episode["episode_id"])
            if current is not None:
                for key in ("evaluation_propositions", "evaluation_constraints",
                            "evaluation_proposition_dependencies"):
                    episode[key] = copy.deepcopy(current[key])
                episode["info"]["variant_spec"]["sha256"] = current["info"]["variant_spec"]["sha256"]
        write_json(aggregate_path, aggregate, compact=True)
    return len(updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("baseline_evaluation_v3"))
    args = parser.parse_args()
    print(f"Updated {migrate(args.root)} Task 6 episodes")
