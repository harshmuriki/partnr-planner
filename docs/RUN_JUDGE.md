# Run judge: three baselines, failures and inefficiencies

`run_judge.py` is a separate, versioned pipeline for the final 1,350-run collection or
an explicitly supplied portable run manifest. Annotation is restricted to image-enabled
`react`, image-enabled `vlm-tamp`, and `tru-pomdp`; both no-image baselines are
excluded from selection and rejected if explicitly requested. Historical artifacts
are retained. It does not modify simulator outputs,
rerun experiments, or change historical Astra studies. The default judge is
`gpt-6-astra` with high reasoning effort.

## Setup and current batch

```bash
PYTHON=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python
JUDGE=results/run_judge_20261001
SOURCE=results/all_baselines_no_video_20261001_220617

"$PYTHON" -m pip install -r scripts/trace_annotator/requirements.txt
# Catalog all runs and prepare the first five finished records, without API calls.
"$PYTHON" scripts/trace_annotator/run_judge.py --out "$JUDGE" prepare --source "$SOURCE" \
  --overrides results/final_baselines_html/rerun_overrides.json --limit 5
# Build exact structured requests for inspection. This does not send API requests.
"$PYTHON" scripts/trace_annotator/run_judge.py --out "$JUDGE" run --limit 5
# Send requests for all finished runs; resume skips unchanged validated results.
"$PYTHON" scripts/trace_annotator/run_judge.py --out "$JUDGE" run --execute --workers 4
# Refresh JSON/CSV exports and the static review page after a batch.
"$PYTHON" scripts/trace_annotator/run_judge.py --out "$JUDGE" export
```

`--limit 0` means all runs. Both `prepare` and `run` accept `--baseline` and
`--run-id`; for example `--baseline react` or
`--run-id react/T3-ACC-BASE/repeat_1`. A run ID always includes its explicit
repetition, even when the underlying runner names every individual job `_r001`.
Active/pending runs are deferred; rerun the command when they finish. The runner
refreshes the source summary each invocation. It does not continuously poll for
new runs. Annotation calls use **only `OPENAI_MODEL_PERSONAL`** from the environment or
repository `.env`, passed directly to this annotator’s client. There is no fallback
to `OPENAI_API_KEY`, and other experiments’ environment variables are unchanged.
Credentials are never stored in requests or results.

The replacement manifest is saved in the judge's `manifest.json` as `overrides`.
Every later `prepare` or `run` refresh reapplies it by baseline, variant and
repetition, preserving catalog order and the total of 1,350 runs. It replaces the
18 old Task 7 TRU-POMDP paths with the checkpoint reruns. Duplicate, unknown,
unfinished or missing-folder replacements are rejected. Each replacement retains
`supersedes` provenance; no source experiment files are modified.

The source catalog can contain all five canonical baseline names: `react`,
`react-no-image`, `vlm-tamp`, `vlm-tamp-no-image`, and `tru-pomdp`. Only the three
selected baselines are annotated. The existing older names such as
`react_images` and `vlm_tamp_no_images` are also accepted. A portable source is:

```json
{
  "runs": [
    {
      "baseline": "react",
      "episode_id": "T3-ACC-BASE",
      "repetition": 1,
      "status": "completed",
      "path": "/absolute/path/to/run",
      "metrics": {"task_percent_complete": 0.3333333333, "task_state_success": 0}
    }
  ]
}
```

Paths can be relative to this manifest. Finished statuses are completed, timeout,
failed and cancelled. The source must explicitly identify the baseline and
repetition; they are never guessed from trace content.

## Add your examples

Editable files in each output directory are:

- `rubric.md`: the exact ten user-defined categories and exceptions.
- `system_prompt.txt`: judge instructions, probability meaning and evidence rules.
- `examples/*.json`: optional human-reviewed input/output demonstrations.

You can create a disabled example draft from an existing annotation:

```bash
"$PYTHON" scripts/trace_annotator/run_judge.py --out "$JUDGE" example \
  --run-id react/T3-ACC-BASE/repeat_1 --name placement_example
```

The example stays disabled until you correct it, set `reviewed_by`, and enable it.

`examples/template.json` is disabled initially. To add a positive example or a
counterexample (e.g. a justified recovery that is **not** a repeated-action error):

1. Copy a run's packet from `packets/<baseline>/<variant>/repeat_N.json` into the
   example's `input` field. This includes simulator criteria, event IDs and attempts.
2. Copy and **correct** the `analysis` object from its saved annotation into
   `expected_output`. The loader removes stored `confidence_calibrated` and
   `review_status` bookkeeping, which are not model output fields.
3. Populate `reviewed_by`, list the source run in `source_run_ids`, and set
   `enabled: true`. Only enabled examples enter the prompt; they must pass the same
   schema and semantic validator as live judgments.
4. Review negative cases too. Do not set `absent` for labels that cannot be assessed
   from an excerpt. If trimming an example, set `coverage.history_complete=false`,
   retain referenced events/attempts/blocks, and use `insufficient_evidence` for
   unsupported whole-run negatives.

```json
{
  "enabled": false,
  "reviewed_by": "",
  "source_run_ids": ["react/T3-ACC-BASE/repeat_1"],
  "input": {},
  "expected_output": {}
}
```

This skeleton stays disabled until filled. The runner hashes the exact examples,
prompt, schema, rubric and evidence. Changing any of them invalidates cached
judgments for subsequent `run` commands; previous annotations are archived under
`history/`. Runs explicitly listed as example sources are excluded from held-out
judging by an error, avoiding evaluating on the answer supplied in the prompt.
Keep related variants/repetitions together when designing your own held-out split.

## Structured input and scoring

The current request format is `html-judge-v3`. The exact API user message is
saved in `model_inputs/<baseline>/<variant>/repeat_N.json` with eight sections:

1. `scenario`: instruction, variant, memory/availability condition, written requirements.
2. `initial_scene`: a compact `rooms` mapping from room names to comma-separated entries, such as `table_0(is_clean=false), book_0(on table_0, is_clean=true), agent_0`. Object locations and recorded states are retained; unlocated entities use `unknown_room`. Missing initial snapshots retain the declared-state fallback.
3. `initial_robot_memory`: declared memory, kept separate from simulator truth.
4. `trace`: numbered HTML actions, thoughts, feedback and observations. VLM-TAMP's
   visible planning report is a separate sequence, not extra executed actions.
5. `subtask_results`: named criteria and succeeded/failed/unknown results; checkpoint
   scores remain separate from final scores.
6. `definitions`: the editable failure and inefficiency rubric.
7. `examples`: short projections of reviewed examples.
8. `output_format`: incident, confidence, citation and counting requirements.

`step_N` is both the displayed action's attempt ID and its evidence ID. `plan_N`
and the named context sections are evidence only. The validator rejects hidden
raw-log IDs. HTML hashes stay in request provenance; editing an HTML trace
invalidates the cached request. Raw packets, simulator handles, event boundaries,
physics, source-record dumps, CSS, JavaScript and images are not sent. Missing HTML
is marked unavailable, with no substitution from raw logs. Missing initial
snapshots use explicitly labeled declared state, never a later snapshot.
Runs without a readable HTML action trace are skipped before any paid API call.
Exact repeated observation sections are omitted with an explicit encoding note;
actions, feedback, changed observations and termination reasons remain. TRU-POMDP
search timings, tree bounds and counters are excluded.

Numeric counts are derived after semantic validation, including distinct-attempt
counts when one attempt independently receives multiple labels. The complete raw
packets and historical paid requests remain local audit artifacts.

New example drafts retain the exact `model_input` from the archived paid request,
so their `step_N` citations remain valid even after a later prompt refresh. Older
examples using raw event IDs remain supported and are projected into short examples.

`packets/` contains the full audit evidence. **Astra receives a filtered, readable
view**, saved separately at `model_inputs/<baseline>/<variant>/repeat_N.json`.
That file is the exact user-message JSON, including the rubric and enabled
examples. `requests/` wraps the same content in the API request with the effective
system prompt; it is not the best file for reading the evidence manually.

Archived scenario data takes precedence over current viewer data. Generation
notes, replay timing, raw identity maps, and duplicate source dumps are excluded.
Internal receptacle meshes resolve to their named furniture. Full raw events and
checkpoint histories are kept only in the audit packet, not in model input.

This projection is deterministic. Raw files and `packets/` remain available for
inspection; source provenance stays local. Output validation rejects citations to
hidden events, failure labels on successful/unscored runs, and invalid subtask links.
Changed request contents invalidate cached annotations. Written requirements do
not create extra simulator-scored subtasks, and declared memory does not prove all
its statements were supplied to the robot.

The strict JSON response contains incidents, numeric `confidence_raw`, subtask and
attempt links, evidence citations, explanations, category assessments, and notes.
After validation, each saved result gains a deterministic `counts` object with
counts per label, distinct failed criteria, ambiguous groups, distinct inefficient
attempts, and score availability. Multiple labels on one criterion do not become
multiple failed subtasks. Unknown results stay unknown; zero annotated incidents
is not proof of absence when a category has insufficient evidence. Confidence is
model-reported and uncalibrated until separately evaluated against human reviews.

Each raw packet contains task instruction, archived input specification and memory,
reference truth, immutable simulator results, execution events, inferred action
attempts, planner-native supporting logs, source-file hashes and coverage flags.
The model is not given baseline/repetition identifiers in the packet provenance.
Planner logs themselves may still reveal planner-specific behavior.

- `p0`, `p1`, ... are zero-based **runtime simulator proposition** indices. Use the
  terminal tracker's expanded list for quantity requirements, not just the original
  dataset's list. Transient satisfaction and applicable constraints both matter.
- In raw audit packets, `e0_4` identifies sequence 4 in the first evidence JSONL file. Each event carries
  its original file and line locator. It is not timestep 4.
- In raw audit packets, `a0001` identifies an inferred high-level action attempt. Execution of one skill
  across many ticks does not produce many attempts. Supplemental logs are never
  independently counted as attempts.
- Runtime name-to-handle mappings and reference graphs establish identity, but
  reference truth does not prove what the robot knew at the time.

Only terminal snapshots or supplied final simulator metrics establish outcomes.
An externally killed run's last intermediate score is not promoted to a final
result. With only an aggregate partial score, an ambiguous failed group is retained
when its failure count can be established; individual object failures are not
invented. Full structured terminal evaluation takes precedence.

Version 2 packets additionally ingest `dataset/evaluation-checkpoints.jsonl`
and `dataset/evaluation-checkpoint.json`. Every saved record is a separately
citable `evaluator_checkpoint` event (for example `checkpoint0_2`), with its file,
line, timestamp, step and content hash. These records do not create action attempts
or make an interrupted action history complete. The latest valid checkpoint after
execution began is scored under `checkpoint_evaluation`, including its lag before
process termination. A truncated latest record falls back to valid history;
initial-only snapshots do not establish checkpoint outcomes.

**Checkpoint policy:** `simulator` remains the final scoring authority.
`checkpoint_evaluation` never replaces final criteria or enables failure labels.
When final scores are missing, checkpoint evidence can support
`possible_failure_notes` and supported inefficiencies; `failures` stays empty.
Thus the dashboard may display last-saved-state subtask counts while the judge
still identifies final scoring as unavailable. A later, explicit policy change
would be needed to treat timeout-cutoff scores as failure-label eligibility.

Raw checkpoint states use lossless `checkpoint_delta` patches between full anchors
(at most 49 patches per anchor). Each patch names its base event and exact JSON
paths to replace or remove. The local view builder reconstructs these before
selecting symbolic changes; these patch instructions are not sent to Astra.

Version 1 example packets remain valid. Existing editable prompts and examples
are preserved. Readable-view instructions are appended to the effective prompt;
the built-in raw-packet decoding instructions are removed from that effective
copy, without modifying the editable prompt file. Changed evidence, view content,
checkpoint bytes or replacement provenance
invalidate cached annotations on the next requested judging pass.

Repeated text and JSON subtrees are encoded losslessly as `text_block_sequence`
references into `text_blocks`, and `json_ref` references into `json_blocks`.
Per-tick planner position/animation diagnostics are omitted from redundant planner
logs; simulator reference graphs at execution boundaries remain available.
Consecutive identical console diagnostics are folded with line numbers and counts.
Proposition history retains every distinct state transition and its original tick.
Attempt boundaries are conservative inferences, not newly instrumented simulator IDs.
Legacy planner logs can supply attempts when incremental evidence is absent, but
coverage remains incomplete. Lack of evidence must not become a negative label.

The default text budget is 950,000 estimated tokens. `--max-input-tokens` can change
it. Oversized requests fail explicitly; no history is silently truncated. The
estimate covers text only; the judge does not receive images. Inspect request sizes before large jobs.

## Structured output and validation

The API is called with strict JSON-schema output. `output.schema.json` describes:

- `failures` and `inefficiencies`: label, criterion/group links, attempt/evidence
  IDs, explanation, alternative explanation, object identity, raw confidence,
  stale-memory discovery citations and unmet prerequisite where relevant.
- `label_assessments`: exactly one row for each of the ten categories, with
  `present`, `absent`, `insufficient_evidence` or `not_applicable`.
- `unclassified_failed_criterion_ids`: scored failures with no established cause.
- `important_notes`, `possible_failure_notes`, and a brief summary.

The validator checks schema/types, valid references, failure eligibility, all
unclassified criteria, per-attempt inefficiency counting, duplicate incidents,
chronology and object consistency of stale-memory claims, and overlapping planning
failure labels. A stale-memory claim must cite pick-capable action evidence with
absence/removal feedback, followed by a distinct later use. These checks do not
prove semantics or causation. A successful skill response is not proof of achieving
an evaluation criterion, and simulator truth is not proof of planner knowledge.

Model refusals, truncated outputs, API failures, invalid references and semantic
validation errors are saved as errors, never accepted annotations. One retry is
allowed using the same versioned request. Successful results include the returned
model name, token usage and request/evidence hashes. Per-run locks prevent duplicate
concurrent requests in the same output directory. No model substitution occurs.

## Text-only annotation

Astra receives **only text and structured data** for all five baselines, including
image-enabled ReAct and VLM-TAMP. It receives no images or video frames and must
not claim direct visual inspection. Logged visual observations remain textual
evidence with their original provenance.

Image attachment code and the `--image-manifest` CLI option have been removed.
Legacy Python callers supplying images or image manifests receive an explicit
error before any image is read or annotation API call is sent. Enabled
examples must also use text/data evidence. This restriction applies only to the
annotation judge; the evaluated baselines keep their original modalities.

## Review, confidence and exports

`index.html` is a small static review page, served through the existing repository
HTTP server, for example:

<http://localhost:8000/results/run_judge_20261001/index.html>

It provides baseline filtering, raw numeric confidence, criteria/notes, structured
input/output and exact-request links. During a batch (at most once every 15 seconds on completions) and at its end,
`summary.json`, `runs.csv`,
`incidents.csv`, and `review_template.json` are refreshed. They do not overwrite
experiment scores or the experiment status dashboard. No automatic acceptance is
implied by a high confidence value.

Copy `review_template.json` to `human_reviews.json` (export overwrites the template,
not your separate review file). Fill `human_accept` with true/false for whether the
**whole incident claim** is correct and add `reviewer`. Unreviewed rows stay null.

```bash
"$PYTHON" scripts/trace_annotator/run_judge.py --out "$JUDGE" calibration \
  --reviews "$JUDGE/human_reviews.json"
```

The report includes Brier score, ECE, reliability bins and accepted fraction per
label, checked against the exact annotation version. It evaluates confidence in
**emitted incidents**, not recall of missed incidents. Recall needs separately
annotated full traces, including human-found incidents the judge missed. Reviews
must include confident outputs and negative cases, not only uncertain cases.
No probability mapping is fitted automatically; `confidence_calibrated` remains
null until a separately validated mapping is available. Raw confidence is never
presented as calibrated. Aggregate label counts overlap; unique inefficient action
attempt counts deduplicate labels on the same attempt.

## Tests and API reference

```bash
"$PYTHON" -m pytest -q scripts/trace_annotator/test_run_judge.py \
  scripts/trace_annotator/test_judge_*.py scripts/trace_annotator/test_episode_context.py
```

The strict output request follows the official [Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

## Timeout failure category

`timeout` records an actual execution time-limit cutoff linked to final simulator
failures. It does not imply unproductive planning; that needs separate behavior
evidence. A cancelled job, API request timeout, or crash is not automatically an
execution timeout. Successful runs get no failure labels, and runs with missing
final scores keep timeout information in notes. Older reviewed examples mark the
new category unassessed rather than inventing a negative.

## Prompt checks with Astra

`results/run_judge_20261001/astra_smoke_20261002.json` records the small authorized
API trial and its targeted follow-up. Requests, responses, prior annotation
versions, and exact model inputs remain in their normal artifact directories.
The checks cover a scored placement failure, a simulator-successful run, and a
timeout with unavailable final scores. They are evidence checks, not a calibrated
accuracy estimate.

The follow-up prompt explicitly permits independently supported inefficiency
labels on the same attempt, sharing its behavior ID. Label counts overlap while
the distinct-attempt count remains a union. It also checks repeated multi-action
cycles: intervening navigation or exploration does not automatically demonstrate
a useful correction. Initial reasonable recovery and evidenced new information
remain valid reasons to retry.

The evidence review is saved in `astra_smoke_review_20261002.json`. Four API calls
were made. The last API-tested prompt is v3; v4 adds a first-attempt precondition
clarification after that result missed one explicit reach failure. This final
wording change has not received another paid call. Model annotations remain
unchanged and unreviewed; the missed attempt is recorded for review.
