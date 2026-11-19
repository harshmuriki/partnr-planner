"""Recover simulator evaluation results without asking an LLM to score criteria."""

from __future__ import annotations

import csv
import gzip
import itertools
import json
from pathlib import Path

from study import REPO, digest


def metadata_categories():
    categories, sources = {}, {}
    for name, column in (
        ("fpmodels-with-decomposed.csv", "main_category"),
        ("object_categories_filtered.csv", "clean_category"),
    ):
        path = REPO / "data/hssd-hab/metadata" / name
        if path.exists():
            sources[str(path)] = digest(path)
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    if row.get(column):
                        categories[row["id"]] = row[column]
    return categories, sources


def short_handle(handle):
    return str(handle).split("/")[-1].split(".")[0].split("_:")[0]


def describe(prop, categories, should=False):
    """Same class-level wording as failure_explanations.derive_proposition_str."""
    args, function = prop["args"], prop["function_name"]

    def category(key):
        values = {categories.get(short_handle(h)) for h in args.get(key, [])}
        if not values or None in values:
            raise KeyError(key)
        return " or ".join(
            '"' + value.replace("_", " ") + '"' for value in sorted(values)
        )

    modal = " should have been" if should else ""
    if function == "is_next_to":
        return f"{category('entity_handles_a')}{modal} placed next to {category('entity_handles_b')}"
    obj = category("object_handles")
    if function in ("is_inside", "is_on_top"):
        relation = "inside" if function == "is_inside" else "on top of"
        return f"{obj}{modal} placed {relation} the {category('receptacle_handles')}"
    if function == "is_on_floor":
        return f"{obj}{modal} placed on the floor"
    if function == "is_in_room":
        room = " or ".join(
            '"' + r.split(".")[0] + '"' for r in sorted(set(args["room_ids"]))
        )
        return f"{obj}{modal} moved to the {room}"
    return f"Object {obj}{modal} {function.removeprefix('is_').replace('_', ' ')}"


def format_predicate(prop, aliases):
    args = prop["args"]
    names = []
    for key in (
        "object_handles",
        "receptacle_handles",
        "entity_handles_a",
        "entity_handles_b",
        "room_ids",
    ):
        if key in args:
            names.append(" | ".join(aliases.get(h, h) for h in args[key]))
    return prop["function_name"] + "(" + ", ".join(names) + ")"


def recover_from_metrics(propositions, metrics, categories):
    """Enumerate all assignments consistent with exact saved simulator output.

    Repeated class labels are never arbitrarily assigned to object instances.
    """
    n = len(propositions)
    statuses = ["unknown"] * n
    percent = metrics.get("task_percent_complete")
    success = metrics.get("task_state_success")
    explanation = str(metrics.get("task_explanation") or "")
    if success == 1 or percent == 1:
        return ["met"] * n, "simulator_all_satisfied", []
    if percent == 0:
        return ["unmet"] * n, "simulator_none_satisfied", []
    if percent is None or not n:
        return statuses, "unavailable", []
    count = float(percent) * n
    if abs(count - round(count)) > 1e-5:
        return statuses, "inconsistent_saved_count", []
    unmet_count = n - round(count)
    should = explanation.startswith("Episode failed. Missing steps:")
    terminal = "Completed steps were later undone:" in explanation
    temporal = "Steps were completed out of order:" in explanation
    if not should and not (terminal or temporal):
        return statuses, "aggregate_only", []
    # If both constraint types fail, a proposition can be repeated across sections.
    # Keep that case aggregate-only rather than treating duplicates as distinct goals.
    if terminal and temporal:
        return statuses, "aggregate_only", []
    lines = [
        line[3:].strip().removesuffix(" should have been completed after:")
        for line in explanation.splitlines()
        if line.startswith(" - ")
    ]
    if not lines or n > 24:
        return statuses, "aggregate_only", []
    try:
        descriptions = [describe(p, categories, should) for p in propositions]
    except KeyError:
        return statuses, "unmapped_simulator_category", []
    base = [
        set(indices)
        for indices in itertools.combinations(range(n), len(lines))
        if [descriptions[i] for i in indices] == lines
    ]
    if not base or unmet_count < len(lines):
        return statuses, "unmatched_simulator_explanation", []
    assignments = []
    for required in base:
        for extra in itertools.combinations(
            sorted(set(range(n)) - required), unmet_count - len(required)
        ):
            assignments.append(required | set(extra))
    if not assignments:
        return statuses, "inconsistent_simulator_explanation", []
    definite_failures = set.intersection(*assignments)
    possible_failures = set.union(*assignments)
    statuses = [
        (
            "unmet"
            if i in definite_failures
            else "met" if i not in possible_failures else "unknown"
        )
        for i in range(n)
    ]
    ambiguous = (
        [
            {
                "criterion_ids": [
                    f"p{i}" for i in sorted(possible_failures - definite_failures)
                ],
                "unmet_count": unmet_count - len(definite_failures),
            }
        ]
        if possible_failures != definite_failures
        else []
    )
    return statuses, "exact_simulator_explanation_constraints", ambiguous


def structured_results(data, n):
    tracker = (
        data.get("auto_eval_proposition_tracker")
        or data.get("task_evaluation_log")
        or {}
    )
    at = tracker.get("proposition_satisfied_at")
    constraints = data.get(
        "task_constraint_validation", tracker.get("constraint_satisfaction", [])
    )
    if not isinstance(at, list) or len(at) != n or not isinstance(constraints, list):
        return None
    if any(not isinstance(row, list) or len(row) != n for row in constraints):
        return None
    return [
        "met" if time != -1 and all(bool(row[i]) for row in constraints) else "unmet"
        for i, time in enumerate(at)
    ]


def extract(run, categories):
    root = Path(run["path"])
    dataset_path = root / "input/dataset.json.gz"
    ep = json.loads(gzip.decompress(dataset_path.read_bytes()))["episodes"][0]
    props = ep.get("evaluation_propositions", [])
    scene_path = root / "input/scene_info.json"
    scene = json.loads(scene_path.read_text()) if scene_path.exists() else {}
    variant = ep.get("info", {}).get("variant_spec", {})
    aliases = {h: name for name, h in scene.get("receptacle_to_handle", {}).items()}
    aliases.update({h: name for name, h in variant.get("entity_handles", {}).items()})
    metrics = {
        k: v
        for k, v in (run.get("benchmark") or {}).items()
        if k in ("task_percent_complete", "task_state_success", "task_explanation")
    }
    source = {"file": run.get("source_summary"), "locator": run["run_id"] + ".metrics"}
    # A terminal incremental snapshot supersedes an aggregate-only export.
    direct = None
    for path in sorted(root.glob("**/evidence-*.jsonl")):
        for index, line in enumerate(path.read_text(errors="replace").splitlines()):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") != "terminal":
                continue
            data = record.get("evaluator_only_state", {})
            direct = structured_results(data, len(props))
            if direct is not None:
                source = {
                    "file": str(path),
                    "locator": f"line {index+1}",
                    "sha256": digest(path),
                }
                metrics.update(
                    {
                        k: data[k]
                        for k in (
                            "task_percent_complete",
                            "task_state_success",
                            "task_explanation",
                        )
                        if k in data
                    }
                )
    if direct is not None:
        statuses, method, ambiguous = direct, "terminal_simulator_snapshot", []
    else:
        statuses, method, ambiguous = recover_from_metrics(props, metrics, categories)
    percent = metrics.get("task_percent_complete")
    success = metrics.get("task_state_success")
    if (
        percent is not None
        and 0 < percent < 1
        and not ambiguous
        and "unknown" in statuses
    ):
        unknown = [f"p{i}" for i, state in enumerate(statuses) if state == "unknown"]
        unknown_unmet = (
            len(props) - round(percent * len(props)) - statuses.count("unmet")
        )
        if unknown_unmet > 0:
            ambiguous = [{"criterion_ids": unknown, "unmet_count": unknown_unmet}]
    outcome = (
        "complete"
        if success == 1 or percent == 1
        else (
            "partial"
            if percent is not None and percent > 0
            else "no_progress" if percent == 0 else "unavailable"
        )
    )
    criteria = [
        {
            "id": f"p{i}",
            "text": format_predicate(p, aliases),
            "kind": "simulator_proposition",
            "status": statuses[i],
            "proposition": p,
            "source": source,
            "recovery_method": method,
        }
        for i, p in enumerate(props)
    ]
    return {
        "source": source,
        "metrics": metrics,
        "outcome": outcome,
        "criteria": criteria,
        "satisfied_count": round(percent * len(props)) if percent is not None else None,
        "total_count": len(props),
        "recovery_method": method,
        "ambiguous_failure_groups": ambiguous,
        "constraints": ep.get("evaluation_constraints", []),
        "unscored_spec_notes": variant.get("unscored_success_text", []),
        "dataset_sha256": digest(dataset_path),
        "aliases": aliases,
        "availability_note": (
            "No saved terminal simulator evaluation; do not infer final success from action text."
            if outcome == "unavailable"
            else ""
        ),
    }
