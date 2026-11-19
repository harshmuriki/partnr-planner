"""Manifest-based, evidence-linked study annotations, independent of Habitat."""

from __future__ import annotations

import ast
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATES = {"met", "unmet", "unknown", "not_applicable"}
CONDITIONS = {
    "vlm_tamp_no_images": "image_comparison_20260927_232820/vlm_tamp_no_images",
    "react_no_images": "image_comparison_20260927_232820/react_no_images",
    "react_images": "image_comparison_20260927_232820/react_images",
    "tru_pomdp_no_images": "tru_pomdp_no_images_20260928_002709",
    "vlm_tamp_images": "vlm_tamp_images_10min_20260928_111039",
}
# Tasks 6 and 7 were rerun with a 30-minute wall limit; tasks 1-5 remain 600 s.
LONG_TIMEOUT_TASKS = {"task_6": 1800, "task_7": 1800}
LABELS = {
    "insufficient_planning": (
        "failure",
        "Insufficient planning",
        "Premature abandonment despite evidence that further useful search or planning was warranted. A timeout alone is insufficient.",
    ),
    "no_action_vlm": (
        "failure",
        "Missing action",
        "Required action omitted from the plan or execution, not merely unreached when the deadline interrupted a plan.",
    ),
    "placement": (
        "failure",
        "Placement error",
        "Object actually placed at an incorrect destination; a rejected placement attempt is insufficient.",
    ),
    "other_object": (
        "failure",
        "Wrong object",
        "Incorrect object selected or used; approved substitutes are correct.",
    ),
    "never_gave_up": (
        "failure",
        "Failure to stop",
        "Continued trying despite available evidence to abandon a subgoal. Hidden ground-truth absence alone is insufficient.",
    ),
    "hallucinated_object": (
        "inefficiency",
        "Hallucinated object",
        "Attempted action on an invented/nonexistent object. Missing from current memory alone is insufficient.",
    ),
    "extra_subgoals": (
        "inefficiency",
        "Extra subgoals",
        "Completed unnecessary subgoals with no justified task, exploration, or recovery purpose.",
    ),
    "repeated_actions": (
        "inefficiency",
        "Repeated actions",
        "Redundant high-level attempt without relevant new evidence or justified recovery; count each redundant attempt, not simulation ticks.",
    ),
    "precondition_failures": (
        "inefficiency",
        "Precondition error",
        "Attempted action with an identifiable unmet prerequisite; generic navigation/skill failure is insufficient.",
    ),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        temp = f.name
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def criteria_from_spec(text):
    """Keep required propositions/order/absence; substitution notes are rules."""
    criteria, rules, constraints = [], [], []
    section = re.search(r"^## Success criteria\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not section:
        raise ValueError("Spec lacks Success criteria")
    for line in section.group(1).splitlines():
        if not line.strip().startswith("- "):
            continue
        item = line.strip()[2:].strip().replace("`", "")
        if item.startswith("Using "):
            rules.append(item)
        elif item.startswith("Distractors ("):
            constraints.append(
                {"id": f"x{len(constraints)+1}", "text": item, "kind": "constraint"}
            )
        else:
            kind = (
                "order"
                if item.startswith("order:")
                else "predicate" if re.match(r"is_\w+\(", item) else "behavior"
            )
            criteria.append({"id": f"c{len(criteria)+1}", "text": item, "kind": kind})
    if not criteria:
        raise ValueError("Spec has no criteria")
    return {"criteria": criteria, "constraints": constraints, "acceptance_rules": rules}


def choose_pilot(runs):
    """Ten per condition: all seven tasks, then ABS/SUB/CON and timeout coverage."""
    chosen = []
    for condition in sorted({r["condition"] for r in runs}):
        pool = sorted(
            (r for r in runs if r["condition"] == condition),
            key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest(),
        )
        local = []
        for task in range(1, 8):
            local.append(next(r for r in pool if r["task"] == task))
        while len(local) < 10:

            def novelty(r, selected=tuple(local)):
                return (
                    3
                    * (
                        r["status"] == "timeout"
                        and not any(x["status"] == "timeout" for x in selected)
                    )
                    + 2 * (r["uncertainty"] not in {x["uncertainty"] for x in selected})
                    + (r["memory"] not in {x["memory"] for x in selected})
                )

            local.append(max((r for r in pool if r not in local), key=novelty))
        chosen.extend(r["id"] for r in local)
    return chosen


def audit_criteria_attachment(study, attachment):
    """Verify every archived variant against the user's supplied reference table."""
    registry, task, axes, bullets = {}, None, [], []

    def flush():
        if task and axes and bullets:
            criteria = criteria_from_spec("## Success criteria\n" + "\n".join(bullets))
            for axis in axes:
                registry[(task, axis)] = criteria

    for line in Path(attachment).read_text().splitlines() + ["## END"]:
        heading = re.match(r"^## T(\d+)$", line)
        axis = re.match(r"^(BASE(?:, CON)?|ABS|SUB|DIS) \(", line)
        if heading or axis or line == "## END":
            flush()
            bullets = []
            if heading:
                task, axes = int(heading.group(1)), []
            elif axis:
                axes = axis.group(1).split(", ")
            else:
                axes = []
        elif line.startswith("- ") and axes:
            bullets.append(line)
    mismatches = []
    for run in study.runs.values():
        expected = registry.get((run["task"], run["uncertainty"]))
        actual = criteria_from_spec(run["spec_text"])

        # The attachment spells the display noun "hand towel"; archived specs
        # use "hand_towel". Preserve entity IDs such as hand_towel_0 exactly.
        def normalized(value):
            return re.sub(
                r"(?<![\w])hand_towel(?![\w])",
                "hand towel",
                json.dumps(value, sort_keys=True),
            )

        if normalized(actual) != normalized(expected):
            mismatches.append(run["id"])
    report = {
        "attachment_sha256": digest(attachment),
        "checked_runs": len(study.runs),
        "reference_variants": len(registry),
        "mismatches": mismatches,
        "checked_at": now(),
        "display_normalization": "bare hand_towel -> hand towel; entity IDs unchanged",
    }
    atomic_json(study.root / "criteria_audit.json", report)
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} archived specs differ from the criteria attachment"
        )
    return report


def initialize(root, sources=None, results_root=None, criteria_attachment=None):
    root = Path(root).resolve()
    if (root / "study.json").exists():
        raise ValueError("Study already exists; use the existing catalog")
    results_root = Path(results_root or REPO / "results")
    runs = []
    for condition, relative in (sources or CONDITIONS).items():
        batch = results_root / relative
        summaries = sorted(batch.glob("task_*/experiment/summary.json"))
        if not summaries:
            raise ValueError(f"No summaries: {batch}")
        for summary in summaries:
            manifest = read_json(summary.with_name("manifest.json"))
            task_dir = summary.parent.parent.name
            expected = LONG_TIMEOUT_TASKS.get(task_dir, 600)
            if manifest.get("timeout_seconds") != expected:
                raise ValueError(f"Not a {expected}-second cohort: {summary}")
            for job in read_json(summary)["jobs"]:
                if job["status"] in ("pending", "running"):
                    raise ValueError("Import only finished batches")
                episode = job["episode_id"].upper()
                task, memory, uncertainty = episode.split("-")
                run_path = Path(job["output"]).resolve()
                spec_path = run_path / "input/spec.md"
                spec = spec_path.read_text()
                model = ((job.get("metrics") or {}).get("cost_metrics") or {}).get(
                    "llm_model"
                )
                run = {
                    "id": condition + "/" + job["run_id"],
                    "run_id": job["run_id"],
                    "condition": condition,
                    "episode_id": episode,
                    "task": int(task[1:]),
                    "memory": memory,
                    "uncertainty": uncertainty,
                    "repetition": int(re.search(r"_r(\d+)$", job["run_id"]).group(1)),
                    "status": job["status"],
                    "runtime_seconds": job.get("elapsed_seconds"),
                    "benchmark": job.get("metrics"),
                    "model": model,
                    "path": str(run_path),
                    "source_summary": str(summary.resolve()),
                    "spec_text": spec,
                    "spec_sha256": digest(spec_path),
                    "settings": {
                        "planner": manifest["planner"],
                        "overrides": manifest["overrides"],
                        "wall_timeout_seconds": manifest["timeout_seconds"],
                    },
                    **criteria_from_spec(spec),
                }
                runs.append(run)
    if len({r["id"] for r in runs}) != len(runs):
        raise ValueError("Duplicate study identities")
    if sources is None and (
        len(runs) != 1350
        or any(v != 270 for v in Counter(r["condition"] for r in runs).values())
    ):
        raise ValueError("Expected exactly five conditions and 1,350 runs")
    study = {
        "schema_version": 2,
        "created_at": now(),
        "model": "gpt-6-luna",
        "requested_model": "gpt-6-luna-high",
        "reasoning_effort": "high",
        "review_policy": "every_run",
        "runs": runs,
        "pilot_ids": choose_pilot(runs),
        "pilot_approved": False,
    }
    atomic_json(root / "study.json", study)
    atomic_json(
        root / "codebook.json",
        {
            "version": 1,
            "labels": {
                k: {"family": v[0], "label": v[1], "definition": v[2], "approved": True}
                for k, v in LABELS.items()
            },
        },
    )
    if criteria_attachment:
        (root / "attached_criteria.md").write_text(
            Path(criteria_attachment).read_text()
        )
    return study


class Study:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / "study.json")
        if not self.config or self.config.get("schema_version") != 2:
            raise ValueError("Not a version-2 annotation study")
        self.runs = {r["id"]: r for r in self.config["runs"]}

    def path(self, kind, run_id):
        if run_id not in self.runs:
            raise KeyError("Unknown study run")
        return self.root / kind / (run_id + ".json")

    def codebook(self):
        return read_json(self.root / "codebook.json")

    def annotation(self, run_id):
        reviewed = read_json(self.path("reviewed", run_id))
        draft = read_json(self.path("drafts", run_id))
        return {
            "reviewed": reviewed,
            "draft": draft,
            "revision": (reviewed or {}).get("revision", 0),
            "draft_sha256": digest(self.path("drafts", run_id)) if draft else "",
        }

    def evidence(self, run_id, refresh=False):
        cache = self.path("evidence", run_id)
        if cache.exists() and not refresh:
            return read_json(cache)
        run = self.runs[run_id]
        root = Path(run["path"])
        events, files, images, reports = [], {}, [], []

        def add(source, locator, value, kind):
            events.append(
                {
                    "id": f"e{len(events)+1:05d}",
                    "source": source,
                    "locator": locator,
                    "kind": kind,
                    "data": value,
                }
            )

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(root).as_posix()
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                images.append(rel)
            if path.suffix == ".html" and ("traces/" in rel or "vlm_tamp_pddl/" in rel or "rail_planner/" in rel):
                reports.append(rel)
            if rel == "console.log":
                lines = path.read_text(errors="replace").splitlines()
                for i in range(0, len(lines), 60):
                    chunk = re.sub(
                        r"\x1b\[[0-?]*[ -/]*[@-~]", "", "\n".join(lines[i : i + 60])
                    )
                    add(rel, f"lines {i+1}-{min(i+60,len(lines))}", chunk, "console")
            elif rel.startswith("dataset/planner-log/") and path.suffix == ".json":
                for i, step in enumerate(read_json(path).get("steps", [])):
                    add(rel, f"steps[{i}]", step, "planner_step")
            elif rel.endswith("vlm_tamp_pddl_log.jsonl"):
                for i, line in enumerate(path.read_text(errors="replace").splitlines()):
                    try:
                        value = ast.literal_eval(line)
                    except (ValueError, SyntaxError):
                        value = {"raw_record": line}  # Never evaluate logged calls.
                    add(rel, f"line {i+1}", value, "pddl_event")
            elif rel.startswith("dataset/traces/") and path.suffix == ".txt":
                lines = path.read_text(errors="replace").splitlines()
                for i in range(0, len(lines), 80):
                    add(
                        rel,
                        f"lines {i+1}-{min(i+80,len(lines))}",
                        "\n".join(lines[i : i + 80]),
                        "trace",
                    )
            elif path.name.startswith("evidence-") and path.suffix == ".jsonl":
                for i, line in enumerate(path.read_text(errors="replace").splitlines()):
                    try:
                        add(rel, f"line {i+1}", json.loads(line), "incremental")
                    except ValueError:
                        continue  # Interrupted final write does not discard earlier records.
            else:
                continue
            files[rel] = digest(path)
        # Archived evaluator inputs are reference truth, never evidence of robot knowledge.
        references = {}
        for name in ("initial_robot_memory.json", "scene_info.json", "dataset.json.gz"):
            p = root / "input" / name
            if p.exists():
                content = (
                    json.loads(gzip.decompress(p.read_bytes()))
                    if name.endswith(".gz")
                    else read_json(p)
                )
                if name == "dataset.json.gz":
                    ep = content["episodes"][0]
                    content = {
                        k: ep.get(k)
                        for k in (
                            "instruction",
                            "info",
                            "evaluation_propositions",
                            "evaluation_constraints",
                            "object_states",
                        )
                    }
                references[name] = content
                files["input/" + name] = digest(p)
        evidence = {
            "run_id": run_id,
            "events": events,
            "reference_truth": references,
            "files_sha256": files,
            "images": images,
            "reports": reports,
            "partial_run": run["status"] != "completed",
            "has_final_planner_log": any(e["kind"] == "planner_step" for e in events),
        }
        evidence["fingerprint"] = hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest()
        atomic_json(cache, evidence)
        return evidence

    def blank(self, run_id):
        run = self.runs[run_id]
        return {
            "criteria": [
                {**c, "status": "unknown", "evidence_ids": [], "explanation": ""}
                for c in run["criteria"] + run["constraints"]
            ],
            "incidents": [],
            "proposed_categories": [],
            "summary": "",
            "unresolved_cause": "",
            "issues_assessment": "unknown",
        }

    def validate(self, run_id, annotation, reviewed=False):
        run = self.runs[run_id]
        evidence = {e["id"] for e in self.evidence(run_id)["events"]}
        expected = {c["id"] for c in run["criteria"] + run["constraints"]}
        rows = annotation.get("criteria", [])
        if (
            not isinstance(rows, list)
            or len(rows) != len(expected)
            or {r.get("id") for r in rows} != expected
        ):
            raise ValueError("Every criterion/constraint must appear exactly once")

        def refs(item):
            ids = item.get("evidence_ids", [])
            if not isinstance(ids, list) or any(
                not isinstance(x, str) or x not in evidence for x in ids
            ):
                raise ValueError("Evidence reference does not belong to this run")

        for row in rows:
            if row.get("status") not in STATES:
                raise ValueError("Invalid criterion status")
            refs(row)
            if row["status"] in ("met", "unmet") and not row.get("evidence_ids"):
                raise ValueError("Met/unmet criteria require evidence")
            if (
                row["status"] != "unknown"
                and not str(row.get("explanation", "")).strip()
            ):
                raise ValueError("Decisions require an explanation")
        issues = annotation.get("issues_assessment")
        if issues not in ("assessed", "unknown"):
            raise ValueError("Explicit issue assessment required")
        labels = self.codebook()["labels"]
        seen = set()
        for event in annotation.get("incidents", []):
            refs(event)
            event_id = event.get("id")
            if not isinstance(event_id, str) or not event_id or event_id in seen:
                raise ValueError("Incident IDs must be unique")
            seen.add(event_id)
            if (
                not event.get("evidence_ids")
                or not str(event.get("explanation", "")).strip()
            ):
                raise ValueError("Every incident requires evidence and explanation")
            tag_ids = event.get("labels", [])
            if (
                not tag_ids
                or len(tag_ids) != len(set(tag_ids))
                or any(k not in labels for k in tag_ids)
            ):
                raise ValueError(
                    "Use codebook labels; propose new categories separately"
                )
            targets = event.get("criterion_ids", [])
            if any(k not in expected for k in targets):
                raise ValueError("Unknown affected criterion")
            if any(labels[k]["family"] == "failure" for k in tag_ids) and (
                not targets
                or not any(r["id"] in targets and r["status"] == "unmet" for r in rows)
            ):
                raise ValueError(
                    "Failure causes must link to an unmet criterion/constraint"
                )
        proposals = annotation.get("proposed_categories", [])
        if not isinstance(proposals, list):
            raise ValueError("Proposed categories must be a list")
        for proposal in proposals:
            refs(proposal)
            if (
                proposal.get("family") not in ("failure", "inefficiency")
                or not all(
                    isinstance(proposal.get(k), str) and proposal[k].strip()
                    for k in ("id", "label", "definition", "example")
                )
                or not proposal.get("evidence_ids")
            ):
                raise ValueError(
                    "Category proposals require definition, family, example, and evidence"
                )
        if reviewed and annotation.get("proposed_categories"):
            raise ValueError(
                "Approve or reject proposed categories before final review"
            )
        if reviewed and not str(annotation.get("reviewer", "")).strip():
            raise ValueError("Reviewer identity required")
        return annotation

    def save_review(self, run_id, annotation, revision, expected_draft_sha256=None):
        self.validate(run_id, annotation, reviewed=True)
        lock = self.path("locks", run_id)
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            previous = read_json(self.path("reviewed", run_id))
            current = (previous or {}).get("revision", 0)
            if revision != current:
                raise RuntimeError("Stale revision; reload before saving")
            draft_path = self.path("drafts", run_id)
            draft_hash = digest(draft_path) if draft_path.exists() else ""
            if (
                expected_draft_sha256 is not None
                and expected_draft_sha256 != draft_hash
            ):
                raise RuntimeError(
                    "AI draft changed since loading; reload before saving"
                )
            record = {
                **annotation,
                "run_id": run_id,
                "revision": current + 1,
                "reviewed_at": now(),
                "codebook_version": self.codebook()["version"],
                "evidence_fingerprint": self.evidence(run_id)["fingerprint"],
            }
            record["source_draft_sha256"] = draft_hash or None
            atomic_json(
                self.path("history", run_id).with_name(
                    f"{Path(run_id).name}.r{current+1}.json"
                ),
                record,
            )
            atomic_json(self.path("reviewed", run_id), record)
        return record

    def add_category(self, category, reviewer):
        key = category.get("id", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,60}", key) or category.get(
            "family"
        ) not in ("failure", "inefficiency"):
            raise ValueError("Invalid category ID/family")
        if not reviewer or not all(
            str(category.get(k, "")).strip() for k in ("label", "definition", "example")
        ):
            raise ValueError(
                "Reviewer, label, definition, and evidence example required"
            )
        with (self.root / "codebook.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            book = self.codebook()
            if key in book["labels"]:
                raise ValueError("Category already exists")
            book["labels"][key] = {
                **category,
                "approved": True,
                "approved_by": reviewer,
                "approved_at": now(),
            }
            book["version"] += 1
            atomic_json(self.root / "codebook.json", book)
        return book

    def list_runs(self):
        rows = []
        for r in self.runs.values():
            a = self.annotation(r["id"])
            review = a["reviewed"]
            rows.append(
                {k: v for k, v in r.items() if k not in ("spec_text", "settings")}
                | {
                    "review_state": (
                        "reviewed"
                        if review
                        else "draft" if a["draft"] else "unannotated"
                    ),
                    "outcome": outcome(r, review) if review else "unreviewed",
                    "pilot": r["id"] in self.config["pilot_ids"],
                }
            )
        return rows

    def aggregate(self, filters=None):
        runs = self.list_runs()
        for key, value in (filters or {}).items():
            if value and key in (
                "condition",
                "task",
                "memory",
                "uncertainty",
                "repetition",
                "status",
                "review_state",
            ):
                runs = [r for r in runs if str(r.get(key)) == str(value)]
        result = {
            "total_runs": len(runs),
            "reviewed": 0,
            "issue_assessed": 0,
            "issue_free": 0,
            "outcomes": Counter(),
            "execution": Counter(r["status"] for r in runs),
            "prevalence": Counter(),
            "occurrences": Counter(),
            "points": [],
            "by_task": {},
            "by_memory": {},
            "by_uncertainty": {},
            "by_condition": {},
            "criterion_met": 0,
            "criterion_known": 0,
            "criterion_unknown": 0,
            "issue_overlap": Counter(),
            "partial_progress": [],
        }
        labels = self.codebook()["labels"]
        for run in runs:
            review = self.annotation(run["id"])["reviewed"]
            result["outcomes"][run["outcome"]] += 1
            for dimension in ("task", "memory", "uncertainty", "condition"):
                group = result["by_" + dimension].setdefault(
                    str(run[dimension]),
                    {
                        "n": 0,
                        "reviewed": 0,
                        "issue_assessed": 0,
                        "outcomes": Counter(),
                        "occurrences": Counter(),
                    },
                )
                group["n"] += 1
                group["reviewed"] += bool(review)
                group["outcomes"][run["outcome"]] += 1
            if not review:
                continue
            result["reviewed"] += 1
            required = {c["id"] for c in run["criteria"]}
            criteria = [
                r
                for r in review["criteria"]
                if r["id"] in required and r["status"] != "not_applicable"
            ]
            met = sum(c["status"] == "met" for c in criteria)
            known = sum(c["status"] in ("met", "unmet") for c in criteria)
            result["criterion_met"] += met
            result["criterion_known"] += known
            result["criterion_unknown"] += len(criteria) - known
            if run["outcome"] == "partial":
                result["partial_progress"].append(
                    {
                        "run_id": run["id"],
                        "task": run["task"],
                        "fraction": met / len(criteria) if criteria else None,
                    }
                )
            if review["issues_assessment"] != "assessed":
                continue
            result["issue_assessed"] += 1
            counts = Counter(k for i in review["incidents"] for k in i["labels"])
            result["prevalence"].update(counts.keys())
            result["occurrences"].update(counts)
            for dim in ("task", "memory", "uncertainty", "condition"):
                result["by_" + dim][str(run[dim])]["issue_assessed"] += 1
                result["by_" + dim][str(run[dim])]["occurrences"].update(counts)
            failure = sum(
                any(labels[k]["family"] == "failure" for k in i["labels"])
                for i in review["incidents"]
            )
            ineff = sum(
                any(labels[k]["family"] == "inefficiency" for k in i["labels"])
                for i in review["incidents"]
            )
            overlap = (
                "both"
                if failure and ineff
                else (
                    "failure_only"
                    if failure
                    else "inefficiency_only" if ineff else "issue_free"
                )
            )
            result["issue_overlap"][overlap] += 1
            result["issue_free"] += not bool(counts)
            result["points"].append(
                {
                    "run_id": run["id"],
                    "task": run["task"],
                    "condition": run["condition"],
                    "failures": failure,
                    "inefficiencies": ineff,
                }
            )
        # Each variant average retains its three repetitions. Unknown/unreviewed excluded explicitly.
        groups = defaultdict(list)
        for run in runs:
            if run["outcome"] in (
                "complete",
                "partial",
                "no_progress",
                "constraint_violation",
            ):
                groups[(run["condition"], run["episode_id"])].append(run)
        result["variant_success"] = [
            {
                "condition": k[0],
                "episode_id": k[1],
                "n": len(v),
                "success_rate": sum(r["outcome"] == "complete" for r in v) / len(v),
            }
            for k, v in groups.items()
        ]
        result["task_balanced_success"] = {}
        for condition in {r["condition"] for r in runs}:
            rates = []
            for task in range(1, 8):
                subset = [
                    r
                    for r in runs
                    if r["condition"] == condition
                    and r["task"] == task
                    and r["outcome"]
                    in ("complete", "partial", "no_progress", "constraint_violation")
                ]
                if subset:
                    rates.append(
                        sum(r["outcome"] == "complete" for r in subset) / len(subset)
                    )
            result["task_balanced_success"][condition] = {
                "tasks_with_known_outcomes": len(rates),
                "rate": sum(rates) / len(rates) if rates else None,
            }
        result["run_ids"] = [r["id"] for r in runs]
        return result


def outcome(run, review):
    if not review:
        return "unreviewed"
    rows = review["criteria"]
    if any(c["status"] == "unknown" for c in rows):
        return "undetermined"
    requirements = {c["id"] for c in run["criteria"]}
    scored = [
        c for c in rows if c["id"] in requirements and c["status"] != "not_applicable"
    ]
    if not scored:
        return "undetermined"
    if any(c["status"] == "unmet" for c in rows if c["id"] not in requirements):
        return "constraint_violation"
    n = sum(c["status"] == "met" for c in scored)
    return "complete" if n == len(scored) else "partial" if n else "no_progress"


def export_csv(study, kind="runs", filters=None):
    if kind not in ("runs", "criteria", "incidents"):
        raise ValueError("Unknown export kind")
    rows = []
    allowed = set(study.aggregate(filters)["run_ids"])
    for run in study.list_runs():
        if run["id"] not in allowed:
            continue
        review = study.annotation(run["id"])["reviewed"]
        base = {
            k: run[k]
            for k in (
                "id",
                "condition",
                "episode_id",
                "task",
                "memory",
                "uncertainty",
                "repetition",
                "status",
                "review_state",
                "outcome",
            )
        }
        if kind == "runs":
            rows.append(
                base
                | {
                    "benchmark": json.dumps(run["benchmark"]),
                    "runtime_seconds": run["runtime_seconds"],
                }
            )
        elif review:
            for item in review["criteria" if kind == "criteria" else "incidents"]:
                rows.append(base | {"annotation": json.dumps(item, ensure_ascii=False)})
    stream = io.StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        "'" + v
                        if isinstance(v, str) and v.startswith(("=", "+", "-", "@"))
                        else v
                    )
                    for k, v in row.items()
                }
            )
    return stream.getvalue()
