"""TRU-POMDP viewer regressions: truthful feedback and belief/episode separation."""

import json

from scripts import view_trace_logs
from scripts.tru_pomdp_trace import load_context, parse_trace, result_status


def make_trace(tmp_path):
    path = tmp_path / "trace.txt"
    particle = {
        "weight": 1.0,
        "goals": [{"object": "cup", "target_area": "sink"}],
        "placements": {"cup": "table"},
        "hypothesized": ["cup"],
    }
    path.write_text(
        "\n".join(
            [
                "Task: Move <cup> & fill it",
                "Hypotheses (initial): " + json.dumps([particle]),
                'Memory: {"cup": "table"}',
                "Decision 1: (PICK cup table)",
                "Action: Navigate[table]",
                "Result: Successful execution!",
                "Action: Pick[cup]",
                "Result: Unexpected failure! Failed to pick <cup>.",
                'Evidence: (PICK cup table): success=False; observation={"parents": {"cup": "table"}}',
                "Hypotheses (updated): " + json.dumps([particle]),
                "Memory: {}",
                "Belief: mass=1.00 eliminated=0 replenished=False particles=1",
                "Decision 2: (POWER_OFF lamp)",
                "Action: PowerOff[lamp]",
                "Result: Successful execution!",
                "Hypotheses (updated): []",
                "Decision 3: (EXPLORE room)",
                "Action: Explore[room]",
            ]
        )
    )
    return path


def test_tru_parser_matches_plain_results_without_metadata_contamination(tmp_path):
    trace = view_trace_logs.parse_trace_file(str(make_trace(tmp_path)))
    assert trace["planner_type"] == "tru_pomdp"
    assert [s["action"] for s in trace["steps"]] == [
        "Navigate",
        "Pick",
        "PowerOff",
        "Explore",
    ]
    assert [s["success"] for s in trace["steps"]] == [True, False, True, None]
    assert trace["steps"][0]["result"] == "Successful execution!"
    first = trace["decisions"][0]
    assert first["before"][0]["weight"] == 1.0
    assert first["after"][0]["placements"] == {"cup": "table"}
    assert first["evidence"] == {"parents": {"cup": "table"}}
    assert first["memory_before"] == {"cup": "table"}
    assert first["memory_after"] == {}
    assert trace["decisions"][2]["before"] == []
    assert trace["decisions"][2]["after"] is None


def test_feedback_unknown_and_negated_success_are_not_success():
    assert result_status("Not successful execution") == "failure"
    assert result_status("Unsuccessful execution") == "failure"
    assert result_status("Successful execution!\nError: dropped object") == "failure"
    assert result_status("Waiting for feedback") == "unknown"
    assert result_status("") == "unknown"


def test_empty_initial_belief_still_selects_tru_viewer(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text("Tree of Hypotheses returned no hypotheses; stopping.\n")
    trace = view_trace_logs.parse_trace_file(str(path))
    assert trace["planner_type"] == "tru_pomdp"
    assert trace["steps"] == []
    assert trace["decisions"] == []


def test_tru_html_separates_skill_outcomes_from_benchmark_and_escapes(tmp_path):
    trace = view_trace_logs.parse_trace_file(str(make_trace(tmp_path)))
    output = tmp_path / "trace.html"
    view_trace_logs.generate_html(
        trace, str(output), task_percent_complete=0.0, task_state_success=0.0
    )
    page = output.read_text()
    assert "TRU-POMDP" in page
    assert "50.0% (2/4)" in page
    assert "0.000 (0.0%)" in page
    assert 'class="step unknown"' in page
    assert 'class="step failure"' in page
    assert "No result recorded" in page
    assert "0 image attachments" in page
    assert "Move &lt;cup&gt; &amp; fill it" in page
    assert "Failed to pick &lt;cup&gt;" in page
    assert "Hypotheses before this decision" in page
    assert "Observed evidence" in page
    assert "No thought" not in page
    assert 'class="dash"' in page
    assert 'class="step-action-row"' in page
    assert 'class="outcome-row"' in page
    assert 'class="belief-strip"' not in page
    assert 'class="decision ' not in page


def test_context_matches_episode_and_run_and_keeps_search_and_update(tmp_path):
    dataset = tmp_path / "dataset"
    source = dataset / "traces/0/trace-episode_TEST_WITH_UNDERSCORES_0-0.txt"
    source.parent.mkdir(parents=True)
    log_path = dataset / "planner-log/planner-log-episode_TEST_WITH_UNDERSCORES_0.json"
    log_path.parent.mkdir()
    log_path.write_text(
        json.dumps(
            {
                "task": "A task",
                "steps": [
                    {
                        "tru_pomdp": {"num_decisions": 1, "root_lower": 3.0},
                        "stats": {"task_state_success": 0},
                    },
                    {
                        "tru_pomdp": {
                            "num_decisions": 1,
                            "termination_reason": "belief_complete",
                            "belief_update": {"replenished": True},
                        },
                        "stats": {"task_state_success": 1},
                    },
                ],
            }
        )
    )
    (tmp_path / "episode_result_log.csv").write_text(
        "episode_id,run_id,runtime,task_percent_complete\nTEST_WITH_UNDERSCORES,0,12.5,1\nTEST_WITH_UNDERSCORES,1,999,0\nOTHER,0,888,0\n"
    )
    context = load_context(source)
    assert context["metrics"]["runtime"] == 12.5
    assert context["metrics"]["task_state_success"] == 1
    assert context["metrics"]["task_percent_complete"] == 1
    assert context["decisions"][1]["search"]["root_lower"] == 3.0
    assert context["decisions"][1]["after"]["belief_update"]["replenished"]
    assert context["diagnostics"]["termination_reason"] == "belief_complete"


def test_malformed_snapshot_is_reported_without_fabricating_empty_belief():
    trace = parse_trace(
        "Hypotheses (initial): not json\nDecision 1: (OPEN cabinet)\nAction: Open[cabinet]\nResult: Successful execution!"
    )
    assert trace["warnings"]
    assert trace["decisions"][0]["before"] is None
    assert trace["decisions"][0]["after"] is None
    assert trace["steps"][0]["status"] == "success"


def test_visible_text_estimate_ignores_simulator_output_and_rejects_partial_logs():
    from scripts.tru_pomdp_trace import estimate_visible_text_usage

    text = (
        "length: 400\n[2026-09-27 12:00:00][httpx] HTTP 200\n"
        "[toh] raw response 1:\n" + "a" * 80 + "\n\n"
        "\x1b[32m" + "simulation output" * 100 + "\n"
    )
    estimate = estimate_visible_text_usage(text, "gpt-5.2", 1)
    assert estimate["prompt_tokens"] == 100
    assert estimate["completion_tokens"] == 20
    assert estimate["llm_usd_source"] == "estimated"
    assert estimate["llm_usd"] > 0
    assert "hidden reasoning" in estimate["llm_usage_note"]
    assert estimate_visible_text_usage(text, "gpt-5.2", 2) == {}


def test_archived_config_fills_model_effort_and_labels_historical_estimate(tmp_path):
    from scripts.tru_pomdp_trace import fill_saved_llm_metadata

    dataset = tmp_path / "dataset"
    source = dataset / "traces/0/trace-episode_TEST_0-0.txt"
    config = tmp_path / "hydra/.hydra/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """evaluation:
  agents:
    agent_0:
      planner:
        plan_config:
          llm:
            keep_message_history: false
            system_message: planner
            generation_params:
              model: gpt-5.2
              reasoning_effort: high
"""
    )
    (tmp_path / "run.log").write_text(
        "Episode TEST\nlength: 400\n[toh] raw response 1:\nSuccess\n"
    )
    metrics = {"llm_call_count": 1}
    fill_saved_llm_metadata(source, dataset, "episode_TEST_0", metrics)
    assert metrics["llm_model"] == "gpt-5.2"
    assert metrics["llm_reasoning_effort"] == "high"
    assert metrics["llm_usd_source"] == "estimated"
    # API-reported usage takes priority over archived config and approximate logs.
    recorded = {
        "llm_call_count": 1,
        "llm_model": "gpt-5.2-2025-12-11",
        "llm_usd": 0.42,
        "llm_usd_source": "api",
    }
    fill_saved_llm_metadata(source, dataset, "episode_TEST_0", recorded)
    assert recorded["llm_model"] == "gpt-5.2-2025-12-11"
    assert recorded["llm_usd"] == 0.42
    assert "llm_usage_note" not in recorded
    # Never borrow another episode's stdout usage.
    wrong = {"llm_call_count": 1}
    fill_saved_llm_metadata(source, dataset, "episode_OTHER_0", wrong)
    assert "llm_usd" not in wrong
