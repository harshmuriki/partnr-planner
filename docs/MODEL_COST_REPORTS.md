# Total model spend per task

All three baselines now write an episode-wide billing report automatically:

`<results_dir>/<dataset>/llm_costs/<episode_output_name>.json`

`summary.total_usd` is the sum of all priced requests in the episode, across
planner and auxiliary calls through OpenAIChat/VLMApi. `summary.cost_usd` breaks
it into ordinary input, cache reads, cache writes, and output. Each call includes
raw provider usage, model, returned service tier, response/request IDs, rates,
pricing date, token counts, and dollar amounts. No prompt content or credentials
are stored. The report is updated before and after each request and finalized
at normal or exceptional exit. A hard kill leaves the last checkpoint, including
any pending request. Repeated subgoals share one ledger; new episodes start fresh.

For each request, with total input I, cache reads R, cache writes W, and output O:

`USD = ((I - R - W) * input_rate + R * read_rate + W * write_rate + O * output_rate) / 1,000,000`

Rates are applied **per request**, including long-context surcharges. Input usage
already contains billed vision/image tokens; reasoning tokens are part of output
usage and are shown as a subset, never charged twice. Local response-cache hits
make no API call and cost zero. Provider prompt-cache hits are billed cache reads.
Billed empty/malformed generations and reprompts count. OpenAI SDK HTTP hooks
record automatic retry attempts without changing retry behavior.

For the configured GPT-5.6 Luna model, standard short-context rates checked on
2026-10-03 are $0.20 ordinary input, $0.02 cached input, $0.25 cache writes, and
$1.20 output per million tokens. Requests above 272,000 input tokens use 2x input
rates (including cache categories) and 1.5x output. Sources:

- [GPT-5.6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [Cache billing formula](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Reasoning token usage](https://developers.openai.com/api/docs/guides/reasoning)

Missing usage, unpriced models, and unsupported service tiers produce a null
`total_usd` plus explicit issues; they are never treated as zero-cost calls.
Missing cache counts produce lower/upper bounds when possible. A request whose
response was lost can have unknown charges; reconcile its request ID with the
provider for invoice-level precision. These are standard list-price model costs,
not credits, discounts, taxes, GPU costs, or a provider billing export. Anthropic
requests are captured but remain unpriced until verified provider rates are added.
The existing planner-local `llm_usd` metrics remain for compatibility; the new
`model_costs` result and report are the source for episode-wide accounting.

## Compare baselines for a task

```sh
python scripts/summarize_model_costs.py \
  --root results/all_baselines_no_video_20261001_220617 \
  --task T3-ACC-BASE \
  --output results/model_costs_T3_ACC_BASE_20261003
```

The command writes a summary CSV and an auditable JSON with source paths. It
includes all repetitions and attempts under the supplied root, prioritizes new
request ledgers over legacy `run.json` totals, and never sums repeated cumulative
planner snapshots. Omit `--task` to group all discovered tasks by task and baseline.

Older run totals lack cache-write counts and may omit failed requests. Their
reported ranges cover **recorded tokens only**, not a guaranteed full historical
bill. A missing final metric cannot be reconstructed as zero. Episode-total token
counts alone cannot resolve long-context pricing when they exceed the threshold.

## Verification

`python scripts/test_model_costs.py` exercises the real OpenAI SDK against an
HTTP mock transport and VLM API against mocked responses. It checks retries,
cache partitioning, reasoning inclusion, interrupted persistence, sequential
subgoals, planner resets, local cache replay, malformed billed responses, and
unknown usage/prices. No API requests or charges are required.
