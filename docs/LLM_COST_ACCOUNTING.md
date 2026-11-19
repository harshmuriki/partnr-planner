# LLM cost accounting

`habitat_llm/utils/llm_usage.py` supplies the shared token tracker used by
ReAct, VLM-TAMP, and TRU-POMDP. OpenAI Chat Completions and Responses usage
payloads are accepted as SDK objects or dictionaries.

For each request, input tokens are partitioned into ordinary input, cached
reads, and cache writes. The write count is already part of the input total:

```text
ordinary = prompt_tokens - cached_tokens - cache_write_tokens
USD = (ordinary * input_rate + cached_tokens * cached_rate
       + cache_write_tokens * write_rate + completion_tokens * output_rate) / 1e6
```

For the supported GPT-5.6 models, the cache-write rate is 1.25 times the ordinary
input rate. Requests with more than 272,000 input tokens use twice the ordinary,
cached-read, and cache-write rates, and 1.5 times the output rate. The threshold
is evaluated per request, never on an episode's accumulated tokens. Earlier
models retain their existing prices without a GPT-5.6 cache-write surcharge.

Sources checked October 2, 2026:
- https://developers.openai.com/api/docs/guides/prompt-caching
- https://developers.openai.com/api/docs/models/gpt-5.6-luna

Saved tracker fields:

- `cache_write_tokens`: episode total when every recorded call reported a write
  count; otherwise `null`. An explicit zero is distinct from a missing counter.
- `cache_write_tokens_reported`: sum for calls that supplied the counter.
- `llm_calls_with_cache_write_usage`: number of calls that supplied the counter.
- `cache_write_usd_per_m`: standard short-context write rate.
- `llm_usd_is_estimate`: true if cache-write pricing could not be completed from
  reported usage, or if token usage itself was estimated or mixed.

When a required write counter is missing, the existing base-cost estimate is
retained and marked as an estimate. `llm_usd_source=api` describes the source of
token counts; it is not proof of invoice reconciliation. Unknown model prices
still produce no dollar amount. Recorded costs exclude account-specific credits
and discounts.

Existing result artifacts and frozen experiment snapshots are not rewritten.
Their historical costs cannot be corrected exactly when write-token counts were
not saved. Newly launched experiments using the updated source collect the new
fields automatically; an old frozen snapshot continues using its archived code.
