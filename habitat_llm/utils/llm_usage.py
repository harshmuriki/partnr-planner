"""Token usage and USD cost for OpenAI / Anthropic LLM and VLM calls."""

import re
from typing import Any, Dict, Optional

# Standard short-context API rates, USD per 1M tokens.
# Source: OpenAI API pricing (2026-09). Long-context surcharge applied per request.
MODEL_PRICES_PER_M: Dict[str, Dict[str, float]] = {
    "gpt-5.6-luna": {"input": 0.20, "cached": 0.02, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gpt-5.6-sol": {"input": 4.00, "cached": 0.40, "output": 20.00},
    "gpt-5.2": {"input": 1.75, "cached": 0.175, "output": 14.00},
    "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "cached": 2.50, "output": 30.00},
}

GPT56_LONG_CONTEXT_TOKENS = 272_000
CHARS_PER_TOKEN_ESTIMATE = 4.0


def normalize_model_name(model: Any) -> str:
    if model is None:
        return ""
    return str(model).strip()


def prices_for_model(model: Any) -> Optional[Dict[str, float]]:
    name = normalize_model_name(model).lower()
    if not name:
        return None
    if name in MODEL_PRICES_PER_M:
        return MODEL_PRICES_PER_M[name]
    for key, prices in MODEL_PRICES_PER_M.items():
        if name.startswith(key):
            return prices
    return None


def _int_or_zero(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def usage_from_openai_payload(payload: Any) -> Dict[str, int]:
    """Parse OpenAI chat.completions usage (SDK object or dict)."""
    if payload is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    usage = getattr(payload, "usage", None)
    if usage is None and isinstance(payload, dict):
        usage = payload.get("usage", payload)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    def _get(obj: Any, key: str, default: Any = 0) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    prompt = _int_or_zero(_get(usage, "prompt_tokens"))
    completion = _int_or_zero(_get(usage, "completion_tokens"))
    details = _get(usage, "prompt_tokens_details", None)
    cached = _int_or_zero(_get(details, "cached_tokens") if details is not None else 0)
    if cached > prompt:
        cached = prompt
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
    }


def request_cost_usd(
    model: Any,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> Optional[float]:
    prices = prices_for_model(model)
    if prices is None:
        return None
    prompt = _int_or_zero(prompt_tokens)
    completion = _int_or_zero(completion_tokens)
    cached = min(_int_or_zero(cached_tokens), prompt)
    uncached = prompt - cached
    in_rate = prices["input"]
    cached_rate = prices.get("cached", in_rate)
    out_rate = prices["output"]
    name = normalize_model_name(model).lower()
    if prompt > GPT56_LONG_CONTEXT_TOKENS and name.startswith("gpt-5.6"):
        in_rate *= 2.0
        cached_rate *= 2.0
        out_rate *= 1.5
    return (uncached * in_rate + cached * cached_rate + completion * out_rate) / 1_000_000.0


def fmt_usd(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number < 0:
        return "N/A"
    if number == 0:
        return "$0.00"
    if number < 0.01:
        return f"${number:.4f}"
    return f"${number:.3f}"


def estimate_tokens_from_chars(n_chars: int) -> int:
    if n_chars <= 0:
        return 0
    return max(1, int(round(n_chars / CHARS_PER_TOKEN_ESTIMATE)))


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_MODEL_HEADER_RE = re.compile(r"model=([A-Za-z0-9._-]+)")
_LENGTH_RE = re.compile(r"length:\s*(\d+)")
_RESPONSE_RE = re.compile(
    r"LLM RESPONSE\s*\n=+\s*\n(.*?)\n=+\s*\n",
    re.S,
)


def model_from_run_log(text: str) -> str:
    clean = _ANSI_RE.sub("", text or "")
    match = _MODEL_HEADER_RE.search(clean)
    return match.group(1) if match else ""


class TokenUsageTracker:
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = normalize_model_name(model)
        self.reasoning_effort = ""
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.usd = 0.0
        self.has_api_usage = False
        self.unknown_price = False
        self.source = ""

    def set_model(self, model: Any) -> None:
        name = normalize_model_name(model)
        if name:
            self.model = name

    def set_reasoning_effort(self, effort: Any) -> None:
        name = normalize_model_name(effort).lower()
        if name:
            self.reasoning_effort = name

    def reset(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.usd = 0.0
        self.has_api_usage = False
        self.unknown_price = False
        self.source = ""

    def set_model(self, model: Any) -> None:
        name = normalize_model_name(model)
        if name:
            self.model = name

    def record(
        self,
        *,
        model: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        source: str = "api",
    ) -> None:
        if model:
            self.set_model(model)
        prompt = _int_or_zero(prompt_tokens)
        completion = _int_or_zero(completion_tokens)
        cached = _int_or_zero(cached_tokens)
        if prompt == 0 and completion == 0:
            return
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached
        if source == "api":
            self.has_api_usage = True
        if not self.source:
            self.source = source
        elif self.source != source:
            self.source = "mixed"
        cost = request_cost_usd(self.model, prompt, completion, cached)
        if cost is None:
            self.unknown_price = True
        else:
            self.usd += cost

    def record_openai(self, payload: Any, model: Any = None) -> None:
        usage = usage_from_openai_payload(payload)
        self.record(model=model, source="api", **usage)

    def snapshot(self) -> Dict[str, Any]:
        total = self.prompt_tokens + self.completion_tokens
        source = self.source or ("api" if self.has_api_usage else "")
        if self.unknown_price:
            usd: Optional[float] = None
        elif total == 0:
            usd = None
        else:
            usd = self.usd
        out: Dict[str, Any] = {
            "llm_model": self.model or "",
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": total,
            "llm_calls_with_usage": self.calls,
        }
        if self.reasoning_effort:
            out["llm_reasoning_effort"] = self.reasoning_effort
        if usd is not None:
            out["llm_usd"] = usd
        if source:
            out["llm_usd_source"] = source
        prices = prices_for_model(self.model)
        if prices is not None:
            out["input_usd_per_m"] = prices["input"]
            out["output_usd_per_m"] = prices["output"]
        return out


def model_name_from_llm(llm: Any) -> str:
    if llm is None:
        return ""
    name = getattr(llm, "model_name", None)
    if name:
        return normalize_model_name(name)
    params = getattr(llm, "generation_params", None)
    if params is None:
        return ""
    if isinstance(params, dict):
        return normalize_model_name(params.get("model"))
    return normalize_model_name(getattr(params, "model", None))


def snapshot_from_llm(llm: Any) -> Dict[str, Any]:
    tracker = getattr(llm, "token_usage", None)
    effort = ""
    params = getattr(llm, "generation_params", None)
    if params is not None:
        if isinstance(params, dict):
            effort = normalize_model_name(params.get("reasoning_effort"))
        else:
            effort = normalize_model_name(getattr(params, "reasoning_effort", None))
    if isinstance(tracker, TokenUsageTracker):
        if not tracker.model:
            tracker.set_model(model_name_from_llm(llm))
        if effort:
            tracker.set_reasoning_effort(effort)
        return tracker.snapshot()
    out = {}
    model = model_name_from_llm(llm)
    if model:
        out["llm_model"] = model
    if effort:
        out["llm_reasoning_effort"] = effort.lower()
    return out


def estimate_tracker_from_react_run_log(
    text: str, model: Optional[str] = None
) -> TokenUsageTracker:
    """Rough token estimate from ReAct planner stdout (chars/4)."""
    clean = _ANSI_RE.sub("", text or "")
    resolved = model or model_from_run_log(clean)
    tracker = TokenUsageTracker(resolved)
    lengths = [int(value) for value in _LENGTH_RE.findall(clean)]
    responses = _RESPONSE_RE.findall(clean)
    count = max(len(lengths), len(responses))
    for index in range(count):
        prompt_chars = lengths[index] if index < len(lengths) else 0
        response = responses[index] if index < len(responses) else ""
        tracker.record(
            model=resolved,
            prompt_tokens=estimate_tokens_from_chars(prompt_chars),
            completion_tokens=estimate_tokens_from_chars(len(response.strip())),
            source="estimated",
        )
    return tracker
