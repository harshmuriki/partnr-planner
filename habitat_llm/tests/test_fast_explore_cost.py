from types import SimpleNamespace

import numpy as np
import pytest

from habitat_llm.utils.fast_explore_cost import (
    approx_steps_from_geodesic,
    estimate_explore_tour,
    sample_explore_furniture_queue,
    scale_explore_steps,
    sim_time_from_steps,
    walk_to_approx_ratio,
)


def test_approx_steps_from_geodesic_uses_velocity_and_freq():
    # 10 m/s at 120 Hz => 0.0833 m/step; 1.0 m => 12 steps
    assert approx_steps_from_geodesic(1.0, forward_velocity=10.0, sim_freq=120.0) == 12


def test_approx_steps_caps_per_hop():
    assert approx_steps_from_geodesic(1000.0, max_steps=300) == 300


def test_sim_time_from_steps():
    assert abs(sim_time_from_steps(120, sim_freq=120.0) - 1.0) < 1e-9


def test_walk_to_approx_ratio_and_scale():
    assert walk_to_approx_ratio(1800, 335) == pytest.approx(1800 / 335)
    assert walk_to_approx_ratio(0, 335) is None
    assert walk_to_approx_ratio(100, 0) is None
    assert scale_explore_steps(335, 2.0) == 670
    assert scale_explore_steps(335, None) == 335
    assert scale_explore_steps(10, 0.0) == 10


def test_sample_queue_keeps_all_when_at_most_max():
    furniture = [SimpleNamespace(name=f"f{i}") for i in range(4)]
    queued = sample_explore_furniture_queue(furniture, max_samples=10)
    assert [f.name for f in queued] == ["f0", "f1", "f2", "f3"]


def test_sample_queue_matches_oracle_explore_random_subset():
    furniture = [SimpleNamespace(name=f"f{i}") for i in range(20)]
    rng = np.random.default_rng(0)
    queued = sample_explore_furniture_queue(furniture, max_samples=10, rng=rng)
    assert len(queued) == 10
    rng2 = np.random.default_rng(0)
    expected_idxs = rng2.choice(20, size=10, replace=False)
    assert [f.name for f in queued] == [f"f{int(i)}" for i in expected_idxs]


def test_nested_cost_metrics_merge_like_evaluation_runner():
    """DecentralizedEvaluationRunner only merges dict/str planner_info values."""
    this_planner_info = {
        "print": "ok",
        "cost_metrics": {
            "llm_planning_time_s": 1.5,
            "explore_approx_sim_steps": 335,
            "explore_approx_sim_time_s": 2.79,
            "explore_approx_meters": 27.5,
        },
    }
    planner_info = {}
    for key, val in this_planner_info.items():
        if type(val) == dict:
            if key not in planner_info:
                planner_info[key] = {}
            planner_info[key].update(val)
        elif type(val) == str:
            if key not in planner_info:
                planner_info[key] = ""
            planner_info[key] += val
        else:
            raise ValueError("Logging entity can only be a dictionary or string!")
    assert planner_info["cost_metrics"]["explore_approx_sim_steps"] == 335
    assert planner_info["print"] == "ok"


def test_estimate_explore_tour_follows_queue_order():
    furniture = [
        SimpleNamespace(name="a", properties={"translation": [1.0, 0.0, 0.0]}),
        SimpleNamespace(name="b", properties={"translation": [2.0, 0.0, 0.0]}),
    ]

    def geodesic_fn(_pathfinder, start, end):
        return float(abs(end[0] - start[0]))

    result = estimate_explore_tour(
        pathfinder=None,
        start_pos=[0.0, 0.0, 0.0],
        furniture=furniture,
        geodesic_fn=geodesic_fn,
        max_samples=10,
    )
    assert result["total_meters"] == 2.0
    assert [leg["to"] for leg in result["legs"]] == ["a", "b"]
    assert result["total_steps"] > 0
    assert "approx tour" in result["summary_lines"][0]
