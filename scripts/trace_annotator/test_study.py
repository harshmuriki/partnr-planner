"""Study semantics, provenance, review gate, and safe artifact access."""

import copy
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from judge import approve_pilot, draft_one, make_packet, run_drafts, share_text_blocks
from study import (
    LABELS,
    Study,
    atomic_json,
    criteria_from_spec,
    read_json,
)
from study_web import create_app


@pytest.fixture
def study(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "console.log").write_text(
        "Pick cup_0 failed: gripper occupied.\nPlace cup_0 on table_0.\n"
    )
    (source / "photo.png").write_bytes(b"PNG test")
    spec = "## Success criteria\n- is_on_top(cup_0, table_0)\n- is_filled(cup_0)\n"
    runs = []
    for i in range(3):
        runs.append(
            {
                "id": f"react_images/run{i}",
                "run_id": f"run{i}",
                "condition": "react_images",
                "episode_id": "T1-ACC-BASE",
                "task": 1,
                "memory": "ACC",
                "uncertainty": "BASE",
                "repetition": i + 1,
                "status": "completed",
                "runtime_seconds": 100,
                "benchmark": {},
                "model": "original-model",
                "path": str(source),
                "spec_text": spec,
                "settings": {},
                **criteria_from_spec(spec),
            }
        )
    root = tmp_path / "study"
    atomic_json(
        root / "study.json",
        {
            "schema_version": 2,
            "model": "gpt-6-luna",
            "requested_model": "gpt-6-luna-high",
            "reasoning_effort": "high",
            "runs": runs,
            "pilot_ids": [r["id"] for r in runs],
            "pilot_approved": False,
        },
    )
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
    return Study(root)


def reviewed(study, rid, statuses=("met", "unmet")):
    a = study.blank(rid)
    a["reviewer"] = "test human"
    a["issues_assessment"] = "assessed"
    for c, s in zip(a["criteria"], statuses):
        c.update(
            status=s,
            explanation="Observed in source evidence.",
            evidence_ids=["e00001"],
        )
    return a


def test_criteria_rules_are_not_extra_subtasks():
    c = criteria_from_spec(
        "## Success criteria\n- `is_filled(cup_0)`\n"
        "- order: `a` before `b`\n- The jug and any substitute are absent: report and give up.\n"
        "- Using `cup_0` in place of the missing `jug_0` counts as success.\n"
        "- Distractors (`vase_0`) must not be used.\n## Next\n- not a criterion\n"
    )
    assert len(c["criteria"]) == 3
    assert [x["kind"] for x in c["criteria"]] == ["predicate", "order", "behavior"]
    assert len(c["acceptance_rules"]) == len(c["constraints"]) == 1
    assert c["criteria"][1]["text"] == "order: a before b"


def test_unknown_is_not_failed_or_issue_free(study):
    rid = next(iter(study.runs))
    a = study.blank(rid)
    a["reviewer"] = "test human"
    study.save_review(rid, a, 0)
    stats = study.aggregate()
    assert stats["outcomes"]["undetermined"] == 1
    assert stats["outcomes"]["unreviewed"] == 2
    assert stats["issue_assessed"] == stats["issue_free"] == 0
    assert stats["criterion_unknown"] == 2


def test_review_requires_evidence_and_valid_cause(study):
    rid = next(iter(study.runs))
    a = reviewed(study, rid)
    a["criteria"][0]["evidence_ids"] = ["another_run/e00001"]
    with pytest.raises(ValueError, match="Evidence reference"):
        study.save_review(rid, a, 0)
    a = reviewed(study, rid)
    a["incidents"] = [
        {
            "id": "i1",
            "labels": ["no_action_vlm"],
            "criterion_ids": ["c1"],
            "evidence_ids": ["e00001"],
            "explanation": "Not linked to an unmet criterion",
        }
    ]
    with pytest.raises(ValueError, match="unmet criterion"):
        study.validate(rid, a)
    a["incidents"][0]["criterion_ids"] = ["c2"]
    study.validate(rid, a)


def test_review_revision_history(study):
    rid = next(iter(study.runs))
    a = reviewed(study, rid)
    one = study.save_review(rid, a, 0)
    assert one["revision"] == 1 and one["evidence_fingerprint"]
    with pytest.raises(RuntimeError, match="Stale"):
        study.save_review(rid, a, 0)
    a["summary"] = "Revised after checking evidence"
    study.save_review(rid, a, 1)
    history = list((study.root / "history").rglob("*.json"))
    assert len(history) == 2 and read_json(history[0])["reviewer"] == "test human"


def test_review_detects_replaced_ai_draft(study):
    rid = next(iter(study.runs))
    a = reviewed(study, rid)
    atomic_json(study.path("drafts", rid), a)
    previous = study.annotation(rid)["draft_sha256"]
    atomic_json(study.path("drafts", rid), a | {"summary": "Newer AI draft"})
    with pytest.raises(RuntimeError, match="AI draft changed"):
        study.save_review(rid, a, 0, previous)


def test_counts_unique_run_and_distinct_incident(study):
    ids = list(study.runs)
    a = reviewed(study, ids[0])
    a["incidents"] = [
        {
            "id": "i1",
            "labels": ["no_action_vlm", "insufficient_planning"],
            "criterion_ids": ["c2"],
            "evidence_ids": ["e00001"],
            "explanation": "One causal decision",
        },
        {
            "id": "i2",
            "labels": ["precondition_failures"],
            "criterion_ids": [],
            "evidence_ids": ["e00001"],
            "explanation": "Separate failed attempt",
        },
    ]
    study.save_review(ids[0], a, 0)
    study.save_review(ids[1], reviewed(study, ids[1], ("met", "met")), 0)
    atomic_json(study.path("drafts", ids[2]), reviewed(study, ids[2]))
    s = study.aggregate()
    assert s["reviewed"] == 2 and s["issue_assessed"] == 2 and s["issue_free"] == 1
    assert s["points"][0]["failures"] == 1 and s["points"][0]["inefficiencies"] == 1
    assert s["prevalence"]["no_action_vlm"] == 1
    assert (
        s["outcomes"]["complete"]
        == s["outcomes"]["partial"]
        == s["outcomes"]["unreviewed"]
        == 1
    )
    assert s["variant_success"][0]["n"] == 2
    assert s["variant_success"][0]["success_rate"] == 0.5
    assert s["partial_progress"][0]["fraction"] == 0.5
    assert study.aggregate({"repetition": "3"})["reviewed"] == 0


def test_categories_need_review_and_versions(study):
    rid = next(iter(study.runs))
    a = reviewed(study, rid)
    p = {
        "id": "motor_execution",
        "family": "failure",
        "label": "Motor execution failure",
        "definition": "Verified motor execution prevented completion",
        "example": "Motor failed despite met preconditions",
        "evidence_ids": ["e00001"],
    }
    a["proposed_categories"] = [p]
    with pytest.raises(ValueError, match="Approve or reject"):
        study.save_review(rid, a, 0)
    assert study.add_category(p, "human")["version"] == 2
    assert study.codebook()["labels"]["motor_execution"]["approved_by"] == "human"


def test_pilot_gate(study):
    with pytest.raises(ValueError, match="human review"):
        approve_pilot(study, "human")
    with pytest.raises(ValueError, match="pilot"):
        run_drafts(study, None, pilot=False)
    for rid in study.runs:
        study.save_review(rid, reviewed(study, rid), 0)
    approve_pilot(study, "human")
    assert Study(study.root).config["pilot_approved"]


def test_web_assets_filters_exports_and_csrf(study):
    app = create_app(study.root)
    c = app.test_client()
    rid = next(iter(study.runs))
    assert c.get("/").status_code == 200
    assert c.get("/api/runs").json[0]["review_state"] == "unannotated"
    assert c.get("/api/stats?task=7").json["total_runs"] == 0
    assert (
        c.get(
            "/api/asset", query_string={"id": rid, "path": "../../study/study.json"}
        ).status_code
        == 404
    )
    assert (
        c.get("/api/asset", query_string={"id": rid, "path": "photo.png"}).status_code
        == 200
    )
    body = {"run_id": rid, "annotation": reviewed(study, rid), "revision": 0}
    assert (
        c.put(
            "/api/review", json=body, headers={"Origin": "https://example.com"}
        ).status_code
        == 403
    )
    assert c.put("/api/review", json=body).status_code == 200
    assert c.put("/api/review", json=body).status_code == 409
    assert "c1" in c.get("/api/export?kind=criteria").text
    assert c.get("/api/export?kind=json").json["schema_version"] == 2
    assert c.get("/api/export?kind=unknown").status_code == 400


def test_safe_pddl_parse_and_partial_jsonl(study):
    rid = next(iter(study.runs))
    source = Path(study.runs[rid]["path"])
    (source / "vlm_tamp_pddl_log.jsonl").write_text(
        "{'event':'action', 'value': Action(name='x')}\n"
    )
    (source / "evidence-test.jsonl").write_text('{"event":"boundary"}\n{"broken":')
    e = study.evidence(rid)
    assert any(
        "raw_record" in x["data"] for x in e["events"] if isinstance(x["data"], dict)
    )
    assert sum(x["kind"] == "incremental" for x in e["events"]) == 1


def fake_client(content):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        value = content(len(calls))
        result = {
            "model": "gpt-6-luna",
            "id": "fake-id",
            "usage": {"total_tokens": 5},
            "choices": [{"message": {"content": json.dumps(value)}}],
        }
        return SimpleNamespace(
            model="gpt-6-luna",
            id="fake-id",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 5}),
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))
            ],
            model_dump=lambda **kwargs: result,
        )

    return (
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        calls,
    )


def test_judge_exact_model_resume_and_retains_failed_attempt(study):
    rid = next(iter(study.runs))
    a = study.blank(rid)
    invalid = copy.deepcopy(a)
    invalid["criteria"][0]["evidence_ids"] = ["fake-event"]
    client, calls = fake_client(lambda n: invalid if n == 1 else a)
    assert draft_one(study, rid, client)["status"] == "drafted"
    assert len(calls) == 2
    assert all(
        c["model"] == "gpt-6-luna" and c["reasoning_effort"] == "high" for c in calls
    )
    assert study.annotation(rid)["reviewed"] is None
    assert draft_one(study, rid, client)["status"] == "skipped"
    assert len(calls) == 2
    assert len(list((study.root / "attempts").rglob("*.json"))) == 2
    d = study.annotation(rid)["draft"]
    assert d["provenance"]["usage"]["total_tokens"] == 5


def test_judge_retains_api_error_without_credentials(study):
    class SensitiveError(Exception):
        status_code = 429

    def fail(**kwargs):
        raise SensitiveError("authorization secret should not be logged")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    rid = next(iter(study.runs))
    result = draft_one(study, rid, client)
    assert result["http_status"] == 429
    saved = next((study.root / "attempts").rglob("*.json")).read_text()
    assert "authorization secret" not in saved
    assert not study.path("drafts", rid).exists()


def test_packet_preserves_boundaries_removes_ticks(study):
    rid = next(iter(study.runs))
    root = Path(study.runs[rid]["path"]) / "dataset/planner-log"
    root.mkdir(parents=True)
    base = {
        "high_level_actions": {"0": ["Pick", "cup_0"]},
        "responses": {},
        "curr_graph": {"0": "cup on table"},
    }
    steps = [base | {"total_step_count": i} for i in range(100)]
    steps += [base | {"responses": {"0": "Pick succeeded"}, "total_step_count": 100}]
    atomic_json(root / "steps.json", {"steps": steps})
    packet = make_packet(study, rid)
    assert packet["coverage"]["unchanged_simulation_ticks_removed"] == 99
    assert sum(e["kind"] == "planner_step" for e in packet["events"]) == 2


def test_incremental_writer_survives_partial_run(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[2] / "habitat_llm/utils/annotation_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("annotation_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    writer = module.AnnotationEvidenceWriter(tmp_path, "episode", "task")
    info = {"high_level_actions": {"0": ["Pick", "cup_0"]}, "total_step_count": 0}
    calls = []
    writer.record(
        info, lambda: calls.append(1) or {"ground_truth": "only in evaluator log"}
    )
    writer.record(info | {"total_step_count": 1}, lambda: calls.append(1))
    lines = writer.path.read_text().splitlines()
    assert len(lines) == 2 and len(calls) == 1
    assert "ground_truth" not in info
    writer.record(info, final=True)
    assert json.loads(writer.path.read_text().splitlines()[-1])["event"] == "terminal"
    monkeypatch.setenv("PARTNR_ANNOTATION_EVIDENCE", "0")
    disabled = module.AnnotationEvidenceWriter(tmp_path, "disabled", "task")
    disabled.record(info)
    assert not disabled.path.exists()


def test_evaluator_snapshot_read_only_handles():
    path = (
        Path(__file__).resolve().parents[2] / "habitat_llm/utils/annotation_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("annotation_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Node:
        name = "table_14"
        sim_handle = "table-handle"
        properties = {"type": "table", "region": object()}

    node = Node()
    graph = SimpleNamespace(graph={node: {}})
    measure = SimpleNamespace(get_metric=lambda: {"satisfied": [True, False]})
    state = module.evaluator_snapshot({"auto_eval_proposition_tracker": measure}, graph)
    assert state["reference_world_graph"][0]["sim_handle"] == "table-handle"
    assert "region" not in state["reference_world_graph"][0]["properties"]
    assert "region" in node.properties
    result_type = dataclasses.make_dataclass("Predicate", [("is_satisfied", bool)])
    encoded = json.dumps({"predicate": result_type(True)}, default=module._json_value)
    assert json.loads(encoded)["predicate"]["is_satisfied"] is True


def test_shared_text_is_lossless_and_does_not_merge_actions():
    line = "Common long prompt paragraph. " * 6 + "\n"
    records = [
        {"id": "e1", "data": line * 4 + "first action\n"},
        {"id": "e2", "data": line * 4 + "second action\n"},
    ]
    encoded, blocks = share_text_blocks(records)

    def decode(value):
        if isinstance(value, dict) and "text_block_sequence" in value:
            return "".join(
                blocks[p["ref"]] if isinstance(p, dict) else p
                for p in value["text_block_sequence"]
            )
        if isinstance(value, dict):
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    assert decode(encoded) == records
    assert len(json.dumps(encoded)) + len(json.dumps(blocks)) < len(json.dumps(records))


def test_duplicate_nav_diagnostics_fold_but_keep_later_actions(study):
    rid = next(iter(study.runs))
    source = Path(study.runs[rid]["path"]) / "console.log"
    source.write_text(
        "".join(
            f"[00:15:49:{i:06d}]:[Error]:[Nav] Sampling failed\n" for i in range(600)
        )
        + "PowerOff lamp_0 succeeded\n"
    )
    packet = make_packet(study, rid, max_chars=2000)
    assert packet["coverage"]["repeated_diagnostic_events_folded"] == 9
    assert not packet["coverage"]["omitted_event_ids"]
    assert packet["events"][0]["repeat_count"] == 600
    assert "PowerOff" in packet["events"][-1]["data"]
