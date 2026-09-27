"""Independent audit of PARTNR planner timing/cost clocks (no env.step)."""

from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from habitat_llm.utils.fast_explore_cost import (
    FORWARD_VELOCITY,
    MAX_FURNITURE_SAMPLES_PER_ROOM,
    MAX_NAV_STEPS_PER_FURNITURE,
    MAX_SKILL_STEPS,
    SIM_FREQ,
    approx_steps_from_geodesic,
    estimate_explore_tour,
    meters_per_step,
    sim_time_from_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_EXPLORE_YAML = (
    REPO_ROOT / "habitat_llm/conf/tools/motor_skills/oracle_explore.yaml"
)
HABITAT_LAB_DEFAULTS = (
    REPO_ROOT
    / "third_party/habitat-lab/habitat-lab/habitat/config/default_structured_configs.py"
)

# Logged kitchen_1 hop table (meters printed to 1 decimal).
KITCHEN_HOPS = [
    ("robot", "table_24", 10.8, 130),
    ("table_24", "cabinet_27", 2.1, 25),
    ("cabinet_27", "cabinet_34", 0.8, 10),
    ("cabinet_34", "cabinet_40", 0.8, 11),
    ("cabinet_40", "cabinet_28", 0.2, 3),
    ("cabinet_28", "fridge_36", 2.3, 28),
    ("fridge_36", "cabinet_23", 2.3, 28),
    ("cabinet_23", "counter_29", 1.3, 16),
    ("counter_29", "cabinet_30", 3.6, 44),
    ("cabinet_30", "cabinet_25", 3.3, 40),
]


def _ceil_steps(distance_m: float) -> int:
    return int(ceil(distance_m / (FORWARD_VELOCITY / SIM_FREQ)))


def _rounds_to_one_decimal(true_m: float, printed_m: float) -> bool:
    return abs(round(true_m, 1) - printed_m) < 1e-9


def test_meters_per_step_is_velocity_over_freq():
    assert meters_per_step(10.0, 120.0) == pytest.approx(10.0 / 120.0)
    assert FORWARD_VELOCITY == 10.0
    assert SIM_FREQ == 120.0


def test_hop_formula_ceil_geodesic_over_meters_per_step():
    assert approx_steps_from_geodesic(1.0) == 12
    assert _ceil_steps(1.0) == 12
    assert approx_steps_from_geodesic(10.8) == _ceil_steps(10.8) == 130
    assert approx_steps_from_geodesic(0.0) == 0
    assert approx_steps_from_geodesic(-1.0) == 0


def test_hop_and_tour_caps():
    assert MAX_NAV_STEPS_PER_FURNITURE == 300
    assert MAX_SKILL_STEPS == 2400
    assert MAX_FURNITURE_SAMPLES_PER_ROOM == 10
    assert approx_steps_from_geodesic(1000.0, max_steps=300) == 300
    over_tour = 2400 * meters_per_step() + 10.0
    assert approx_steps_from_geodesic(over_tour, max_steps=300) == 300


def test_kitchen_logged_totals_are_internally_consistent():
    printed_m = [m for _, _, m, _ in KITCHEN_HOPS]
    printed_steps = [s for _, _, _, s in KITCHEN_HOPS]
    assert len(KITCHEN_HOPS) == 10
    assert sum(printed_m) == pytest.approx(27.5)
    assert sum(printed_steps) == 335
    assert 335 / 120.0 == pytest.approx(2.7916666667)
    assert round(335 / 120.0, 2) == 2.79


def test_kitchen_hops_match_ceil_formula_allowing_one_decimal_rounding():
    """Printed meters are .1f; 2.1 m / 25 steps is valid if true dist ~2.05 m."""
    mps = meters_per_step()
    mismatches_if_printed_were_exact = []
    for src, dst, printed_m, printed_steps in KITCHEN_HOPS:
        exact_from_print = _ceil_steps(printed_m)
        if exact_from_print == printed_steps:
            continue
        lo = (printed_steps - 1) * mps
        hi = printed_steps * mps
        # Any true geodesic in (lo, hi] yields ceil(d/mps) == printed_steps.
        candidates = np.linspace(lo + 1e-9, hi, 50)
        feasible = [
            float(d)
            for d in candidates
            if _rounds_to_one_decimal(float(d), printed_m)
            and approx_steps_from_geodesic(float(d)) == printed_steps
        ]
        assert feasible, (
            f"{src} -> {dst}: printed {printed_m} m / {printed_steps} steps "
            f"is not ceil(printed/mps)={exact_from_print} and no 1-decimal-"
            f"consistent true distance exists in ({lo:.4f}, {hi:.4f}]"
        )
        mismatches_if_printed_were_exact.append((src, dst, printed_m, printed_steps, exact_from_print))
    # The audit log itself flags 2.1 m -> 25 (not 26) as a rounding case.
    assert any(row[2] == 2.1 and row[3] == 25 for row in mismatches_if_printed_were_exact)


def test_exact_sim_time_704_over_120():
    assert 697 + 3 + 2 + 2 == 704
    assert 704 / 120.0 == pytest.approx(5.8666666667)
    assert round(704 / 120.0, 2) == 5.87
    assert sim_time_from_steps(704) == pytest.approx(704 / 120.0)


def test_estimate_explore_tour_matches_hop_formula():
    furniture = [
        SimpleNamespace(name="a", properties={"translation": [1.0, 0.0, 0.0]}),
        SimpleNamespace(name="b", properties={"translation": [1.0, 0.0, 2.5]}),
        SimpleNamespace(name="c", properties={"translation": [4.0, 0.0, 2.5]}),
    ]
    hops = {"a": 1.0, "b": 2.5, "c": 3.0}

    def geodesic_fn(_pathfinder, start, end):
        # Pathfinder-less snap uses the translation; fake geodesic by target name via x+z identity.
        key = None
        for node in furniture:
            t = np.array(node.properties["translation"], dtype=np.float64)
            if np.allclose(t[:3], end[:3]):
                key = node.name
                break
        assert key is not None
        return hops[key]

    result = estimate_explore_tour(
        pathfinder=None,
        start_pos=[0.0, 0.0, 0.0],
        furniture=furniture,
        geodesic_fn=geodesic_fn,
        max_samples=10,
    )
    expected_steps = sum(approx_steps_from_geodesic(d) for d in hops.values())
    assert result["total_meters"] == pytest.approx(6.5)
    assert result["total_steps"] == expected_steps
    assert result["total_time_s"] == pytest.approx(expected_steps / 120.0)
    assert [leg["steps"] for leg in result["legs"]] == [
        approx_steps_from_geodesic(hops[name]) for name in ("a", "b", "c")
    ]


def test_estimate_explore_tour_respects_skill_step_cap():
    furniture = [
        SimpleNamespace(name=f"f{i}", properties={"translation": [float(i + 1), 0.0, 0.0]})
        for i in range(5)
    ]

    def geodesic_fn(_pathfinder, start, end):
        return 50.0  # 50 m => 600 raw steps, hop-capped at 300

    result = estimate_explore_tour(
        pathfinder=None,
        start_pos=[0.0, 0.0, 0.0],
        furniture=furniture,
        geodesic_fn=geodesic_fn,
        max_nav_steps_per_furniture=300,
        max_skill_steps=2400,
    )
    # 300 * 8 would exceed 2400; with 5 furniture, 5*300=1500 < 2400
    assert result["total_steps"] == 1500
    result_capped = estimate_explore_tour(
        pathfinder=None,
        start_pos=[0.0, 0.0, 0.0],
        furniture=furniture,
        geodesic_fn=geodesic_fn,
        max_nav_steps_per_furniture=300,
        max_skill_steps=700,
    )
    # First hop 300, remaining 400 -> second hop min(300, 400)=300, remaining 100 -> third hop 100, then stop.
    assert result_capped["total_steps"] == 700


def test_action_sim_steps_increment_skips_empty_and_explore():
    """Mirrors evaluation_runner: step (if any) using prev_action, then assign the new action."""
    action_sim_steps = {}
    prev_action_per_agent = {}
    low_level_actions: list = []
    env_steps = 0
    # Planner returns (low_level_actions, high_level_action_name) each iteration.
    planner_returns = [
        ({}, "Explore"),
        ([{"base_vel": 1.0}], "Navigate"),
        ([{"base_vel": 1.0}], "Navigate"),
        ([{"base_vel": 1.0}], "Pick"),
        ([], "Done"),
    ]
    for new_actions, new_action in planner_returns:
        if len(low_level_actions) > 0:
            for prev_action_name in prev_action_per_agent.values():
                if prev_action_name:
                    action_sim_steps[prev_action_name] = (
                        action_sim_steps.get(prev_action_name, 0) + 1
                    )
            env_steps += 1
        low_level_actions = new_actions
        prev_action_per_agent[0] = new_action
    assert "Explore" not in action_sim_steps
    assert action_sim_steps["Navigate"] == 2
    assert action_sim_steps["Pick"] == 1
    assert env_steps == 3
    assert sum(action_sim_steps.values()) == env_steps


def test_oracle_explore_yaml_matches_python_constants():
    with ORACLE_EXPLORE_YAML.open() as handle:
        loaded = yaml.safe_load(handle)
    skill = loaded["oracle_explore"]["skill_config"]
    assert skill["sim_freq"] == 120
    assert skill["max_skill_steps"] == 2400
    assert skill["max_nav_steps_per_furniture"] == 300
    assert skill["max_furniture_samples_per_room"] == 10
    nav = skill["nav_skill_config"]
    assert nav["forward_velocity"] == 10.0
    assert nav["turn_velocity"] == 10.0
    assert nav["sim_freq"] == 120
    assert nav["dist_thresh"] == 0.2


def test_habitat_ctrl_freq_default_is_120():
    text = HABITAT_LAB_DEFAULTS.read_text(encoding="utf-8")
    assert "ctrl_freq: float = 120.0" in text


def test_react_baseline_does_not_override_ctrl_freq():
    conf_dir = REPO_ROOT / "habitat_llm/conf"
    hits = []
    for path in conf_dir.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if "ctrl_freq" in text:
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == []
