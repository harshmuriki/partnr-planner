"""Build the Error Viewer page from the findings CSVs of classify_errors.py and the judge JSONLs of llm-as-a-judge.py.

Usage:
    python3 build_viewer.py [--out viewer/error_viewer.html]

The page is self-contained: run details, actions, findings and judge attributions of every batch in BATCHES are embedded,
with the judge's TAXONOMY from llm-as-a-judge.py.
Trace and run-folder links are relative paths under /coc/flash5, so they open when
the page is served from /coc/flash5 (see README.md).
"""

import argparse
import csv
import gzip
import json
import os
import re
import runpy
from pathlib import Path

import sys
from functools import cache

from classify_errors import FAILURES, INEFFICIENCIES, Run

# SPL_act / TC-SPL_act use the same definitions and human references as the TC-SPL dashboard.
sys.path.insert(0, "/coc/flash5/vmuriki3/tamp/partnr-planner/scripts/dashboard_tc_spl")
import metrics as tc_spl  # noqa: E402

@cache
def gt_references() -> dict:
    return tc_spl.read_gt()

def action_metrics(row: dict, success: bool) -> dict:
    """SPL_act = S * A_ref / max(A, A_ref); TC-SPL_act = C * A_ref / max(A, A_ref).

    A counts recorded skill attempts (Rearrange as 4, Done/ReportAbsence excluded); A_ref is the human ground-truth
    action count of the variant."""
    ref = gt_references()[row["task_id"]]["actions"]
    actions = tc_spl.action_count(row.get("counts") or {}) if row.get("num_actions") is not None else None
    completion = row.get("completion")
    spl = tc_spl.action_efficiency(float(success), actions, ref)
    tcspl = tc_spl.action_efficiency(completion, actions, ref)
    rnd = lambda v: None if v is None else round(v, 4)
    return {"completion": rnd(completion), "spl_act": rnd(spl), "tc_spl_act": rnd(tcspl),
            "action_count": actions, "reference_actions": ref}

HERE = Path(__file__).resolve().parent
RESULTS = Path("/coc/flash5/vmuriki3/tamp/partnr-planner/results")
# Each batch: (name, [(findings CSV, judge JSONL or None, results dir), ...]). A batch can combine several
# results dirs; their baseline_results ids must not collide (each planner/config has its own id prefix).
LUNA_T4 = ("findings_luna_all_planners_t4basket.csv", "judge_luna_all_planners_t4basket.jsonl", RESULTS / "all_baselines_5reps_vlmtamp_t4basket_20261030")
PSK = Path("/coc/pskynet1/vmuriki3/partnr-results")  # T6 reruns with waited_in (flash5 was full on 2026-11-19)
# T1-T5 + T7 from the original sweeps; T6 rerun with waited_in at 1200 s (merged/ under each rerun batch).
RAIL = [
    ("findings_rail_wait_merged.csv", None, PSK / "rail_planner_t6_wait_5reps_20261118" / "merged"),
    ("findings_rail_placement_oracle_wait_merged.csv", None, PSK / "rail_placement_oracle_t6_wait_5reps_20261118" / "merged"),
    ("findings_rail_eop_feedback_wait_merged.csv", None, PSK / "rail_end_of_plan_feedback_t6_wait_5reps_20261118" / "merged"),
    ("findings_rail_replan_trigger_oracle_wait_merged.csv", None, PSK / "rail_replan_trigger_oracle_t6_wait_5reps_20261118" / "merged"),
    # T1-T5 from rail_placement_oracle_eop_feedback_5reps_20261029, T6-T7 rerun with waited_in at 1200 s.
    ("findings_rail_placement_oracle_eop_feedback_wait_merged.csv", None, RESULTS / "rail_placement_oracle_eop_feedback_wait_t6t7_5reps_20261118" / "merged"),
]
BATCHES = [
    ("gpt-5.6-luna · all planners", [("findings_luna_all_planners.csv", "judge_luna_all_planners.jsonl", RESULTS / "all_baselines_5reps_20261005_001408")]),
    ("gpt-5.6-luna · all planners, VLM-TAMP T4 basket fix", [LUNA_T4]),
    ("gpt-5.6-luna · baselines (T4 basket fix) + RailPlanner", [LUNA_T4, RAIL[0]]),
    ("gpt-5.6-luna · RailPlanner configurations (oracles are privileged upper bounds)", RAIL),
    ("gpt-6-astra · ReAct", [("findings_astra_react.csv", "judge_astra_react.jsonl", RESULTS / "react_gpt6astra_medium_3reps_20261110")]),
    ("gpt-6-astra · ReAct placement oracle, T3+T5", [("findings_astra_react_placement_oracle.csv", "judge_astra_react_placement_oracle.jsonl", RESULTS / "react_gpt6astra_placement_oracle_t3t5_3reps_20261111")]),
]

def run_record(row: dict, viewer_dir: Path) -> dict:
    run = Run(row)
    folder = run.folder.resolve()
    baseline, variant, repeat = row["id"].split("/")
    traces = folder / "dataset" / "traces" / "0"
    trace_html = next(traces.glob("trace-*.html"), None)
    trace_txt = next(traces.glob("trace-*.txt"), None)
    termination = (row.get("termination") or {}).get("reason")
    ended = "time limit" if row["execution_status"] == "timeout" else "action limit" if termination == "max_actions" else "stopped"

    def link(path: Path | None) -> str | None:
        return os.path.relpath(path, viewer_dir) if path else None

    success = row["task_state_success"] == 1.0
    return {
        **action_metrics(row, success),
        "id": row["id"],
        "baseline": baseline,
        "variant": variant,
        "task": variant.split("-")[0],
        "memory": row["memory"],
        "uncertainty": row["uncertainty"],
        "repeat": int(re.sub(r"\D", "", repeat) or 0),
        "success": success,
        "ended": ended,
        "runtime": round(row.get("runtime_s") or 0),
        "instruction": json.loads(gzip.open(folder / "input" / "dataset.json.gz").read())["episodes"][0]["instruction"],
        "steps": [[action.id, f"{action.name}[{action.input}]", action.response.startswith("Successful")] for action in run.actions],
        "trace_html": link(trace_html),
        "trace_txt": link(trace_txt),
        "folder": link(folder),
        "findings": [],
        "judge": None,
    }

def batch_record(name: str, sources: list, viewer_dir: Path) -> dict:
    runs = {}
    for findings, judge, results_dir in sources:
        add_source(runs, HERE / findings, HERE / judge if judge else None, results_dir, viewer_dir)
    return {"name": name, "findings_csv": ", ".join(f for f, _, _ in sources),
            "results_dir": ", ".join(str(d) for _, _, d in sources), "runs": list(runs.values())}

def add_source(runs: dict, findings: Path, judge, results_dir: Path, viewer_dir: Path) -> None:
    rows = json.loads((results_dir / "baseline_results.json").read_text())
    new = {row["id"]: run_record(row, viewer_dir) for row in rows}
    assert not set(new) & set(runs), f"duplicate run ids from {results_dir}"
    runs.update(new)
    with findings.open() as f:
        for finding in csv.DictReader(f):
            runs[finding["run_id"]]["findings"].append(
                [finding["kind"], finding["category"], finding["label"], finding["evidence"]]
            )
    if judge is None:
        return
    with judge.open() as f:
        for line in f:
            record = json.loads(line)
            run = runs[record["run_id"]]
            result = record["result"]
            if "failed" in result:
                run["judge"] = {"failed": str(result["failed"])}
                continue
            attributions = []
            for a in result["attributions"]:
                error = None if a["error_id"] == "extra" else int(a["error_id"][1:]) - 1
                assert error is None or 0 <= error < len(run["findings"]), (run["id"], a["error_id"])
                attributions.append([error, a["module"], a["error_type"], a["step"], a["evidence"], a["description"], a["confidence"]])
            run["judge"] = {"reasoning": result["reasoning"], "attributions": attributions}

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Error Viewer page.")
    parser.add_argument("--out", type=Path, default=HERE / "viewer" / "error_viewer.html")
    args = parser.parse_args()

    # Links are relative to the page's home on flash5, where it is served from.
    viewer_dir = Path("/coc/flash5/vmuriki3/tamp/partnr-planner/uncertainty_classifier/viewer")
    data = {
        "rules": [[rule.kind, rule.category, rule.label] for rule in FAILURES + INEFFICIENCIES],
        "taxonomy": {module: list(types) for module, types in runpy.run_path(HERE / "llm-as-a-judge.py")["TAXONOMY"].items()},
        "batches": [batch_record(name, sources, viewer_dir) for name, sources in BATCHES],
    }
    template = (HERE / "viewer" / "template.html").read_text()
    page = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    args.out.write_text(page)
    for batch in data["batches"]:
        judged = [r["judge"] for r in batch["runs"] if r["judge"]]
        print(f"{batch['name']}: {len(batch['runs'])} runs, {sum(len(r['findings']) for r in batch['runs'])} findings, "
              f"{len(judged)} judged runs, {sum(len(j.get('attributions', [])) for j in judged)} attributions")
    print(f"-> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
