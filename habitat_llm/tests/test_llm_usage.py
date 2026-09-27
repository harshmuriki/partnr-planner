from types import SimpleNamespace

from habitat_llm.llm.openai_chat import chat_create_kwargs
from habitat_llm.utils.llm_usage import (
    TokenUsageTracker,
    estimate_tracker_from_react_run_log,
    fmt_usd,
    request_cost_usd,
    usage_from_openai_payload,
)


def test_chat_create_kwargs_adds_high_effort_for_gpt5():
    kwargs = chat_create_kwargs(
        {"model": "gpt-5.6-luna", "reasoning_effort": "high"}
    )
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["reasoning_effort"] == "high"


def test_chat_create_kwargs_skips_effort_for_gpt4o_mini():
    kwargs = chat_create_kwargs(
        {"model": "gpt-4o-mini", "reasoning_effort": "high"}
    )
    assert kwargs == {"model": "gpt-4o-mini"}
    cost = request_cost_usd("gpt-5.6-luna", 100_000, 100_000, 0)
    assert abs(cost - 0.14) < 1e-9


def test_cached_tokens_use_cached_rate():
    cost = request_cost_usd("gpt-5.6-luna", 100_000, 0, 100_000)
    assert abs(cost - 0.002) < 1e-9


def test_long_context_surcharge():
    cost = request_cost_usd("gpt-5.6-luna", 272_001, 1000, 0)
    no_surcharge = (272_001 * 0.20 + 1000 * 1.20) / 1_000_000
    with_surcharge = (272_001 * 0.40 + 1000 * 1.80) / 1_000_000
    assert abs(cost - with_surcharge) < 1e-9
    assert cost > no_surcharge


def test_unknown_model_has_no_price():
    assert request_cost_usd("local-llama", 100, 10) is None


def test_usage_from_openai_dict_and_object():
    payload = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
        }
    }
    parsed = usage_from_openai_payload(payload)
    assert parsed == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cached_tokens": 40,
    }
    obj = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1),
        )
    )
    assert usage_from_openai_payload(obj)["prompt_tokens"] == 10


def test_tracker_snapshot_and_fmt():
    tracker = TokenUsageTracker("gpt-4o-mini")
    tracker.record(
        model="gpt-4o-mini",
        prompt_tokens=100_000,
        completion_tokens=100_000,
        source="api",
    )
    snap = tracker.snapshot()
    assert snap["llm_model"] == "gpt-4o-mini"
    assert abs(snap["llm_usd"] - 0.075) < 1e-9
    assert snap["llm_usd_source"] == "api"
    assert fmt_usd(0.00012) == "$0.0001"


def test_estimate_from_react_log():
    text = (
        "model=gpt-5.6-luna episode=t1-acc-base\n"
        "length: 8\n"
        "LLM RESPONSE\n"
        "====\n"
        "abcd\n"
        "====\n"
    )
    tracker = estimate_tracker_from_react_run_log(text)
    assert tracker.model == "gpt-5.6-luna"
    assert tracker.source == "estimated"
    assert tracker.prompt_tokens == 2
    assert tracker.completion_tokens == 1


def test_length_on_same_line_as_objects():
    text = "lamp_1: chest_of_drawers_32 in bedroom_1length: 6405\n"
    tracker = estimate_tracker_from_react_run_log(text, model="gpt-5.6-luna")
    assert tracker.prompt_tokens == 1601
