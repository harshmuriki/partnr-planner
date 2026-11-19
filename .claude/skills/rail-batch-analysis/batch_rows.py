"""Write baseline_results.json rows for a rail sweep (results/<batch>/shards/*/attempt_*/runs/*/run.json),
in the row format of results/rail_planner_5reps_20261028/build_rows.py, so scripts/dashboard_tc_spl/metrics.py
and the paper-figure scripts can read it. Each row also carries end-of-plan and replan-trigger stats parsed
from the run's vlm_tamp_pddl_log.jsonl:

    reached_plan_end, sent_back, end_of_plan_replans, recovered      (end_of_plan_check events)
    trigger_checks, trigger_fired, trigger_rules                     (replan_trigger events)

Usage (habitat-llm env):
    python .claude/skills/rail-batch-analysis/batch_rows.py results/<batch> --condition rail_trig_images --prefix rail-trig
"""
import argparse
import ast
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("rail_rows", REPO / "results/rail_planner_5reps_20261028/build_rows.py")
rail_rows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rail_rows)  # row_for(job): the shared row format


def latest_runs(batch):
    latest = {}
    for path in sorted(batch.glob("shards/*/attempt_*/runs/*/run.json")):
        job = json.loads(path.read_text())
        if job.get("status") in ("completed", "timeout"):
            latest[job["run_id"]] = job  # later attempts win
    return latest


def events(out, names):
    found = []
    for p in out.glob("*/vlm_tamp_pddl_log.jsonl"):
        for line in p.open(errors="replace"):
            if not any(f"'{n}'" in line for n in names):
                continue
            try:
                e = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if e.get("event") in names:
                found.append(e)
    return found


def replan_stats(out):
    evs = events(out, ("end_of_plan_check", "replan_trigger"))
    checks = [e for e in evs if e["event"] == "end_of_plan_check"]
    rules = [e.get("rule") for e in evs if e["event"] == "replan_trigger" and e.get("fired")]
    return {"reached_plan_end": bool(checks), "sent_back": bool(checks) and not checks[0]["goal_satisfied"],
            "end_of_plan_replans": sum(bool(e.get("replan")) for e in checks),
            "trigger_checks": sum(e["event"] == "replan_trigger" for e in evs),
            "trigger_fired": len(rules), "trigger_rules": rules}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    p.add_argument("--condition", required=True, help="e.g. rail_trig_images (metrics/figure key)")
    p.add_argument("--prefix", required=True, help="row id prefix, e.g. rail-trig (classifier/heatmap key)")
    p.add_argument("--out", type=Path, help="default: <batch>/baseline_results.json")
    a = p.parse_args()
    batch = a.batch.resolve()

    def build(job):
        row = rail_rows.row_for(job)
        row["id"] = row["id"].replace("rail/", f"{a.prefix}/", 1)
        row["condition"] = a.condition
        row.update(replan_stats(Path(job["output"])))
        row["recovered"] = row["sent_back"] and row["task_state_success"] == 1
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = sorted(pool.map(build, latest_runs(batch).values()), key=lambda r: r["id"])
    assert len({r["id"] for r in rows}) == len(rows)
    out = a.out or batch / "baseline_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} rows, {sum(r['task_state_success'] == 1 for r in rows)} successes -> {out}")


if __name__ == "__main__":
    main()
