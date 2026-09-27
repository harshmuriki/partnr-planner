#!/usr/bin/env python3

from habitat_llm.vlm_tamp.prompts_vlm_tamp import (
    _format_objects,
    build_english_subgoal_prompt,
    build_predicate_translation_prompt,
)
from habitat_llm.vlm_tamp.render_vlm_prompts_html import _empty_response_summary
from habitat_llm.vlm_tamp.vlm_api import GPT4vApi, _openai_sampling_payload


def _sample_objects_by_type():
    return {
        "movable": ["box_4", "knife_7"],
        "surface_furniture": ["bench_18", "chest_of_drawers_56", "table_12"],
        "furniture_by_room": {
            "entryway_1": ["bench_18", "chest_of_drawers_56"],
            "living_room_1": ["table_12"],
        },
        "objects_by_room": {
            "entryway_1": ["box_4"],
            "unassigned": ["knife_7"],
        },
        "container_furniture": ["bench_18", "chest_of_drawers_56"],
        "joint": ["chest_of_drawers_56"],
        "room": ["entryway_1", "living_room_1"],
    }


def test_format_objects_groups_surface_by_room_with_braces():
    formatted = _format_objects(_sample_objects_by_type())
    lines = formatted.splitlines()

    assert lines[0] == "movable: box_4, knife_7"
    assert lines[1] == "surface_furniture_by_room:"
    assert lines[2] == "entryway_1: bench_18, chest_of_drawers_56, box_4"
    assert lines[3] == "living_room_1: table_12"
    assert lines[4] == "container_furniture: bench_18, chest_of_drawers_56"
    assert lines[5] == "joint: chest_of_drawers_56"
    assert lines[6] == "room: entryway_1, living_room_1"


def test_format_objects_includes_faucet_furniture():
    obj = dict(_sample_objects_by_type())
    obj["faucet_furniture"] = ["counter_2", "sink_cabinet_5"]
    formatted = _format_objects(obj)
    assert "furniture_with_faucet: counter_2, sink_cabinet_5" in formatted.splitlines()


def test_prompt_builders_include_room_grouped_surface_format():
    objects = _sample_objects_by_type()

    english_prompt = build_english_subgoal_prompt(
        goal="Move boxes",
        objects_by_type=objects,
        scene_description="Robot hand is empty.",
    )
    predicate_prompt = build_predicate_translation_prompt(objects_by_type=objects)

    expected_surface = "\n".join(
        [
            "surface_furniture_by_room:",
            "entryway_1: bench_18, chest_of_drawers_56, box_4",
            "living_room_1: table_12",
        ]
    )
    assert expected_surface in english_prompt
    assert expected_surface in predicate_prompt
    assert (
        "It can have accurate, missing or outdated information." in english_prompt
    )


def test_openai_sampling_omits_temperature_for_gpt5():
    assert _openai_sampling_payload("gpt-5.6-luna", 0.2, n=1) == {}
    assert _openai_sampling_payload("gpt-4o-mini", 0.2, n=1) == {"temperature": 0.2}
    assert _openai_sampling_payload("gpt-4o-mini", 0.2, n=2) == {
        "temperature": 0.2,
        "n": 2,
    }
    assert _openai_sampling_payload(
        "gpt-5.6-luna", 0.2, n=1, reasoning_effort="high"
    ) == {"reasoning_effort": "high"}
    assert _openai_sampling_payload(
        "gpt-4o-mini", 0.2, n=1, reasoning_effort="high"
    ) == {"temperature": 0.2}


def test_openai_empty_response_records_finish_and_reasoning_tokens(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {"x-request-id": "req_test_123"}

        def json(self):
            return {
                "id": "chatcmpl_test",
                "model": "gpt-5.6-luna",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 1200,
                    "prompt_tokens_details": {"cached_tokens": 100},
                    "completion_tokens_details": {"reasoning_tokens": 1200},
                },
            }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "habitat_llm.vlm_tamp.vlm_api.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    api = GPT4vApi(model_name="gpt-5.6-luna")
    response = api.ask(
        "test prompt",
        max_completion_tokens=1200,
        reasoning_effort="high",
        use_cache=False,
    )
    metadata = api.get_last_response_metadata()

    assert response == ""
    assert metadata["status"] == "ok"
    assert metadata["finish_reasons"] == ["length"]
    assert metadata["completion_tokens"] == 1200
    assert metadata["reasoning_tokens"] == 1200
    assert metadata["empty_response"] is True
    assert metadata["request_id"] == "req_test_123"


def test_empty_response_summary_exposes_recorded_reason():
    summary = _empty_response_summary(
        {
            "status": "ok",
            "finish_reasons": ["length"],
            "completion_tokens": 1200,
            "reasoning_tokens": 1200,
            "max_completion_tokens": 1200,
        }
    )

    assert "finish_reason=length" in summary
    assert "reasoning_tokens=1200" in summary
