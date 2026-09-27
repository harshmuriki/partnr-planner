from types import SimpleNamespace

from habitat_llm.utils.episode_cost import (
    DEFAULT_MAX_COMBINED_TIME_S,
    combined_time_breakdown,
)


def test_combined_budget_adds_three_clocks():
    status = combined_time_breakdown(
        llm_planning_time_s=19.57,
        action_sim_steps={"Navigate": 697, "Pick": 3, "Place": 2, "PowerOff": 2},
        explore_approx_sim_time_s=5.52,
        sim_freq=120.0,
        limit_s=600.0,
    )
    assert DEFAULT_MAX_COMBINED_TIME_S == 600.0
    assert status["llm_s"] == 19.57
    assert status["action_sim_s"] == 704 / 120.0
    assert status["explore_approx_s"] == 5.52
    assert status["used_s"] == 19.57 + 704 / 120.0 + 5.52
    assert status["limit_s"] == 600.0
    assert status["exceeded"] is False


def test_combined_budget_excludes_explore_from_exact_actions():
    status = combined_time_breakdown(
        llm_planning_time_s=0.0,
        action_sim_steps={"Navigate": 120, "Explore": 999},
        explore_approx_sim_time_s=1.0,
        sim_freq=120.0,
        limit_s=600.0,
    )
    assert status["action_sim_s"] == 1.0
    assert status["used_s"] == 2.0


def test_combined_budget_exceeded_at_limit():
    status = combined_time_breakdown(
        llm_planning_time_s=500.0,
        action_sim_steps={"Navigate": 12000},
        explore_approx_sim_time_s=0.0,
        sim_freq=120.0,
        limit_s=600.0,
    )
    # 500 + 100 + 0 = 600
    assert status["used_s"] == 600.0
    assert status["exceeded"] is True


def test_combined_budget_disabled_when_limit_non_positive():
    status = combined_time_breakdown(
        llm_planning_time_s=999.0,
        action_sim_steps={"Navigate": 12000},
        explore_approx_sim_time_s=50.0,
        limit_s=0,
    )
    assert status["exceeded"] is False


def test_store_combined_time_status_sets_info_keys():
    from habitat_llm.evaluation.evaluation_runner import EvaluationRunner

    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.evaluation_runner_config = SimpleNamespace(
        max_combined_time_s=10.0,
        combined_time_sim_freq=120.0,
    )
    runner._last_budget_log_used_s = -1.0
    info: dict = {}
    budget_info: dict = {}
    planner_info = {
        "cost_metrics": {
            "llm_planning_time_s": 6.0,
            "explore_approx_sim_time_s": 3.0,
        }
    }
    exceeded = runner._store_combined_time_status(
        info,
        budget_info,
        runner._combined_time_status(planner_info, {"Navigate": 240}),
        log=False,
    )
    # 6 + 240/120 + 3 = 11 >= 10
    assert exceeded is True
    assert info["combined_time_limit_hit"] is True
    assert info["combined_time_used_s"] == 11.0
    assert info["combined_time_limit_s"] == 10.0
    assert info["combined_time_breakdown"]["action_sim_s"] == 2.0
