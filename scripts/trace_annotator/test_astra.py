"""Regression tests for simulator authority and the revised user taxonomy."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from astra_study import RUBRIC, full_packet, validate_analysis
from sim_results import recover_from_metrics, structured_results


def prop(function, *handles):
    args = {"object_handles": [handles[0]], "number": 1}
    if function == "is_on_top":
        args["receptacle_handles"] = [handles[1]]
    return {"function_name": function, "args": args}


def test_simulator_success_overrides_trace_uncertainty():
    props = [
        prop("is_on_top", "jug_:0000", "table_:0000"),
        prop("is_filled", "jug_:0000"),
    ]
    states, method, groups = recover_from_metrics(
        props, {"task_state_success": 1, "task_percent_complete": 1}, {}
    )
    assert (
        states == ["met", "met"] and method == "simulator_all_satisfied" and not groups
    )


def test_simulator_explanation_resolves_unique_failures():
    props = [
        prop("is_on_top", "jug_:0000", "table_:0000"),
        prop("is_filled", "jug_:0000"),
        prop("is_powered_off", "lamp_:0000"),
    ]
    metrics = {
        "task_percent_complete": 1 / 3,
        "task_state_success": 0,
        "task_explanation": 'Episode failed. Missing steps:\n - "jug" should have been placed on top of the "table"\n - Object "jug" should have been filled',
    }
    states, _, groups = recover_from_metrics(
        props, metrics, {"jug": "jug", "table": "table", "lamp": "lamp"}
    )
    assert states == ["unmet", "unmet", "met"] and not groups


def test_duplicate_class_failure_is_not_assigned_arbitrarily():
    props = [
        prop("is_on_top", "box_:0000", "table_:0000"),
        prop("is_on_top", "box_:0001", "table_:0000"),
        prop("is_filled", "jug_:0000"),
    ]
    states, _, groups = recover_from_metrics(
        props,
        {
            "task_percent_complete": 2 / 3,
            "task_state_success": 0,
            "task_explanation": 'Episode failed. Missing steps:\n - "box" should have been placed on top of the "table"',
        },
        {"box": "box", "table": "table", "jug": "jug"},
    )
    assert states == ["unknown", "unknown", "met"]
    assert groups == [{"criterion_ids": ["p0", "p1"], "unmet_count": 1}]


def test_temporal_violation_uses_simulator_constraint_result():
    props = [
        prop("is_clean", "glass_:0000"),
        prop("is_clean", "glass_:0001"),
        prop("is_filled", "glass_:0000"),
    ]
    states, _, groups = recover_from_metrics(
        props,
        {
            "task_percent_complete": 2 / 3,
            "task_state_success": 0,
            "task_explanation": 'Episode failed. All steps were completed, but constraints were broken:\nSteps were completed out of order:\n - Object "glass" clean should have been completed after:\n   - "soap dispenser" placed next to "glass"',
        },
        {"glass": "glass"},
    )
    assert states == ["unknown", "unknown", "met"] and groups[0]["unmet_count"] == 1


def test_missing_terminal_metrics_stay_unavailable():
    states, method, groups = recover_from_metrics([prop("is_filled", "jug")], {}, {})
    assert states == ["unknown"] and method == "unavailable" and not groups


def test_structured_scoring_matches_evaluator_formula():
    data = {
        "auto_eval_proposition_tracker": {"proposition_satisfied_at": [1, 2, -1]},
        "task_constraint_validation": [[True, False, True]],
    }
    assert structured_results(data, 3) == ["met", "unmet", "unmet"]


@pytest.fixture
def setup():
    sim = {
        "outcome": "partial",
        "criteria": [{"id": "p0", "status": "met"}, {"id": "p1", "status": "unmet"}],
        "ambiguous_failure_groups": [],
    }
    study = SimpleNamespace(runs={"run": {"simulator": sim}})
    packet = {"events": [{"id": "e00001"}, {"id": "e00002"}, {"id": "e00003"}]}
    analysis = {
        "summary": "Observed run",
        "failures": [],
        "inefficiencies": [],
        "important_notes": [],
        "possible_failure_notes": [],
    }
    return study, packet, analysis


def incident(label):
    return {
        "id": "i1",
        "label": label,
        "criterion_ids": ["p1"],
        "unmet_group_ids": [],
        "evidence_ids": ["e00002"],
        "explanation": "Observed evidence",
        "confidence": "high",
        "object_name": "jug_0",
        "outdated_memory_object": False,
        "prior_failed_pick_evidence_ids": [],
    }


def test_no_failure_labels_on_simulator_success(setup):
    study, packet, a = setup
    study.runs["run"]["simulator"]["outcome"] = "complete"
    a["failures"] = [incident("no_action_vlm")]
    with pytest.raises(ValueError, match="simulator-unsuccessful"):
        validate_analysis(study, "run", a, packet)
    a["failures"] = []
    a["inefficiencies"] = [incident("precondition_failures")]
    validate_analysis(study, "run", a, packet)


def test_only_notes_for_missing_simulator_scores(setup):
    study, packet, a = setup
    study.runs["run"]["simulator"]["outcome"] = "unavailable"
    a["failures"] = [incident("insufficient_planning")]
    with pytest.raises(ValueError, match="simulator-unsuccessful"):
        validate_analysis(study, "run", a, packet)
    a["failures"] = []
    a["possible_failure_notes"] = [
        {
            "text": "Unproductive loop before timeout; no final score",
            "evidence_ids": ["e00002"],
        }
    ]
    validate_analysis(study, "run", a, packet)


def test_stale_memory_requires_prior_pick_and_later_use(setup):
    study, packet, a = setup
    x = incident("hallucinated_object")
    x["outdated_memory_object"] = True
    a["inefficiencies"] = [x]
    with pytest.raises(ValueError, match="exact evidence"):
        validate_analysis(study, "run", a, packet)
    x["prior_failed_pick_evidence_ids"] = ["e00002"]
    with pytest.raises(ValueError, match="later evidence"):
        validate_analysis(study, "run", a, packet)
    x["prior_failed_pick_evidence_ids"] = ["e00001"]
    validate_analysis(study, "run", a, packet)


def test_model_cannot_supply_success_scores(setup):
    study, packet, a = setup
    a["criteria"] = [{"status": "unmet"}]
    with pytest.raises(ValueError, match="cannot be changed"):
        validate_analysis(study, "run", a, packet)


def test_failure_can_reference_explicit_ambiguous_group(setup):
    study, packet, a = setup
    sim = study.runs["run"]["simulator"]
    sim["criteria"][1]["status"] = "unknown"
    sim["ambiguous_failure_groups"] = [
        {"id": "g0", "criterion_ids": ["p1"], "unmet_count": 1}
    ]
    x = incident("placement")
    x["unmet_group_ids"] = ["g0"]
    a["failures"] = [x]
    validate_analysis(study, "run", a, packet)


def test_full_history_retains_late_events_and_saved_prompts(tmp_path):
    source = tmp_path / "source"
    prompts = source / "dataset/prompts/0"
    prompts.mkdir(parents=True)
    (prompts / "prompt.txt").write_text("Complete original planner context.")
    (tmp_path / "rubric.md").write_text(RUBRIC.read_text())
    run = {
        "path": str(source),
        "spec_text": "Task",
        "criteria": [],
        "constraints": [],
        "acceptance_rules": [],
        "status": "timeout",
        "runtime_seconds": 600,
        "episode_id": "T1-OUT-BASE",
        "task": 1,
        "memory": "OUT",
        "uncertainty": "BASE",
        "repetition": 1,
        "simulator": {"outcome": "unavailable"},
    }
    events = [
        {"id": "e00001", "kind": "trace", "locator": "trace:1", "data": "x" * 400000},
        {
            "id": "e00002",
            "kind": "trace",
            "locator": "trace:2",
            "data": "Late failed pick and subsequent retry.",
        },
    ]
    evidence = {
        "events": events,
        "reference_truth": {},
        "has_final_planner_log": False,
        "partial_run": True,
    }
    study = SimpleNamespace(
        root=Path(tmp_path),
        runs={"run": run},
        evidence=lambda _: evidence,
        codebook=lambda: {},
        blank=lambda _: {"old_label": "never send"},
    )
    packet = full_packet(study, "run")
    assert packet["coverage"]["omitted_event_ids"] == []
    assert [e["id"] for e in packet["events"]] == ["e00001", "e00002"]
    assert packet["events"][-1]["data"] == events[-1]["data"]
    assert (
        packet["saved_planner_context"][0]["text"]
        == "Complete original planner context."
    )
    assert "annotation_template" not in packet and "codebook" not in packet
    assert packet["simulator"] == run["simulator"]
