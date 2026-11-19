"""Timeouts count as zero-progress failures in dashboard aggregates."""

from build_baseline_dashboard import build, render, summarize


def row(iteration, status, outcome, solved, total, completion, runtime, **extra):
    return {
        "id": f"react_no_images/t1-acc-base_r00{iteration}",
        "condition": "react_no_images",
        "task": 1,
        "task_id": "T1-ACC-BASE",
        "memory": "ACC",
        "uncertainty": "BASE",
        "iteration": iteration,
        "run_id": f"t1-acc-base_r00{iteration}",
        "execution_status": status,
        "simulator_outcome": outcome,
        "sim_steps": extra.get("sim_steps"),
        "num_actions": extra.get("num_actions"),
        "counts": {},
        "runtime_s": runtime,
        "num_subtasks": total,
        "subtasks_succeeded": solved,
        "completion": completion,
        "run_folder": "/tmp/run",
    }


ROWS = [
    row(1, "completed", "complete", 2, 2, 1.0, 100.0, num_actions=10),
    row(2, "completed", "partial", 1, 2, 0.5, 200.0, num_actions=20),
    row(3, "timeout", "unavailable", None, 2, None, 600.0),
]


def test_timeouts_count_as_failures():
    s = summarize(ROWS)
    assert s["n"] == 3 and s["successes"] == 1
    assert s["sr"] == 1 / 3
    assert s["cr"] == 0.5
    assert (s["sub_solved"], s["sub_total"]) == (3, 6)
    assert s["sub_pct"] == 0.5 and s["sub_mean"] == 1.0
    assert s["timeouts"] == 1 and s["timeout_rate"] == 1 / 3
    assert s["runtime_mean"] == 300.0 and s["runtime_median"] == 200.0
    assert s["runtime_success_mean"] == 100.0


def test_secondary_metrics_average_available_runs_only():
    s = summarize(ROWS)
    assert (s["actions_mean"], s["actions_n"]) == (15.0, 2)
    assert (s["usd_mean"], s["usd_n"]) == (None, 0)


def test_build_joins_costs_and_groups_variants():
    costs = {("react_no_images", "t1-acc-base_r001"): {"llm_usd": 0.02, "total_tokens": 500}}
    data = build(ROWS, costs)
    assert [c["key"] for c in data["conditions"]] == ["react_no_images"]
    assert data["overall"]["react_no_images"]["usd_n"] == 1
    cell = data["variants"][0]["cells"]["react_no_images"]
    assert (cell["successes"], cell["n"], cell["cr"]) == (1, 3, 0.5)
    assert data["breakdowns"]["task"]["stats"]["2"]["react_no_images"] is None
    assert "</script>" not in render(data).split("const DATA = ")[1].split(";\n")[0]
