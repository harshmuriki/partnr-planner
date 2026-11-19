# Ordered simulator subtask comparison

The **Final Baseline results dashboard** includes a single-task stacked-bar
comparison of **VLM-TAMP images, ReAct images, and TRU-POMDP**. The other
baseline variants remain available in the rest of the dashboard.

Choose a task, optionally filter memory and uncertainty, and compare succeeded
(green), failed (red), and unknown (gray) counts for each subtask. All three
repetitions contribute separately. Every bar uses the same count scale within
the selected view. Click a bar to inspect repetitions, evaluator provenance,
and trace/run links.

Subtasks follow their order in the archived ACC-BASE evaluation list. Additional
variant-specific predicates follow that list, ordered by their earliest
criterion position. This is evaluation-list order, not inferred execution order.
Matching still uses predicate content: an absent-object variant's shifted index
does not change the order of the corresponding base criterion.

Unknown outcomes are shown separately, never counted as failures. Missing
terminal results do not establish failure, and an omitted criterion contributes
no outcome. Compare bar totals and unknown counts before comparing baselines.

## Provenance and matching

- Read the dataset archived with each run, not the current task specification.
- Prefer the terminal simulator proposition tracker and constraint validation.
- Otherwise use saved final aggregate scores: all met or all unmet when the
  score establishes that, or exact simulator explanations when they uniquely
  constrain criterion assignments. Identical object classes remain ambiguous
  where individual attribution cannot be established.
- Match criteria across runs using task, predicate and all arguments, replacing
  handles with the archived aliases for display. Do not join by proposition
  index: absent-object variants can remove and reorder criteria.
- Preserve allowed alternative targets and quantity thresholds. For example,
  Task 7's two-toy predicate expands to two thresholds. The one-toy threshold
  does not identify a particular toy. Counts are evaluator criteria, not
  necessarily independent semantic subgoals.
- Predicate results incorporate simulator constraint validation; they need not
  describe only the final physical state. A microwave placement criterion is
  displayed as placement, not relabeled as an independently measured heat state.

This chart does not assign causal failure categories such as `placement` or
`insufficient_planning`. Those require the separate trace-annotation pipeline.
No model/API calls or simulator reruns are made to build the chart.

## Refresh

From the repository root:

```bash
python scripts/trace_annotator/enrich_dashboard_subtasks.py
```

This enriches cached final comparison rows, preserving action and cost metrics,
and updates the self-contained dashboard at
`results/run_comparison_20260929/baseline_dashboard.html`. Subsequent executions
reuse the per-criterion cache. The full `build_final_baselines.py` builder also
extracts these fields when rebuilding the collection from source.

The cached rows contain `subtask_results` with original propositions and score
provenance. `dashboard_data.json` contains `subtasks.definitions` plus per-run
observations. Dashboard filters and drill-down run entirely in the browser.

## Interrupted runs and checkpoints

When no final score exists, the chart now uses a valid evaluator checkpoint
recorded after execution began. Bars identify how many outcomes come from a
**last saved state**. Drill-down shows the source, step, saved timestamp and lag
before process termination when available. These are scores *at that checkpoint*,
not claims about a later unsaved state. The **Final scores only** checkbox restores
unknown outcomes for these runs. Final scores always take precedence, and the
other dashboard metrics retain their final-score conventions.

`EnvironmentInterface.step` saves `dataset/evaluation-checkpoint.json` atomically
after every completed simulator step, including steps inside skills. A compact
history is appended to `evaluation-checkpoints.jsonl` every 25 steps. States
include proposition satisfaction, constraints, score metrics and reference world
graph; they never enter planner prompts. The tracker history is reduced to its
length and latest state rather than duplicating the entire trajectory each step.
An initial-only snapshot is not sufficient to recover an interrupted run.

Recover existing archived evidence once:

```bash
python scripts/trace_annotator/refresh_dashboard_checkpoints.py --recover-existing
```

Publish completed checkpoint reruns, preserving the previous run folders:

```bash
python scripts/trace_annotator/refresh_dashboard_checkpoints.py \
  --rerun-root results/tru_pomdp_t7_checkpoints_20261002_020415 --watch 30
```

`rerun_overrides.json` records replacement runs so full dashboard rebuilds retain
them. Rerun HTML gets its own subdirectory inside `final_baselines_html`.
