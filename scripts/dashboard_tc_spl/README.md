# Completion and execution metrics (Svelte)

Build with Node 22 or later:

```
cd scripts/dashboard_tc_spl
npm ci
npm run build
```

From the repository root, refresh the standalone dashboard without rerunning simulations:

```
python scripts/dashboard_tc_spl/metrics.py results/all_baselines_5reps_20261005_001408
```

The compiled Svelte bundle and JSON payload are embedded in the existing HTML,
so no additional development server or CDN is needed. The batch's full dashboard
and Sankey rebuilds also inject this section. `tc_spl_data.json` contains per-run
scores and per-variant reference provenance with source hashes.

GT selection follows `baseline_evaluation_v3/ground_truth_timing/combined.html`,
then reads and validates the referenced replay JSON. Successful replay simulator
steps are used for 88 variants. Failed replay variants T2-ACC-DIS and T4-ACC-ABS
use saved archive step counts (604 and 394). These are human reference executions,
not proven optimal paths. Timeouts retain their saved evaluator results. Missing
inputs remain unavailable and coverage is reported. Zero steps are not missing.

Baseline palette (Okabe–Ito): VLM-TAMP #0072B2, ReAct #D55E00,
TRU-POMDP #CC79A7. Baseline bars have direct text labels and numeric values.
This mapping is also applied to the existing baseline comparison charts when
injected; Sankey outcome colors encode outcomes, not baseline identity.

`verify.py` checks the formulas, reproduces the older chart's five step-based scores (historically labeled TC-SPL),
and exercises the Svelte filters and run drilldowns using Playwright.

## Completion-weighted action efficiency (TC-SPL)

For each run: `TC-SPL_i = C_i * A_ref_i / max(A_i, A_ref_i)`.
The displayed score is the arithmetic mean of available per-run scores.
`C_i` is saved evaluator completion, `A_i` is recorded skill attempts, and
`A_ref_i` is the corresponding human reference skill count. Failed and interrupted
attempts count; Explore and Wait each count once. Exclude Done and ReportAbsence
on both sides. Expand any Rearrange to two Navigate + Pick + Place. Unknown
counts stay unavailable; zero observed attempts are valid. Nonpositive reference
counts cannot be scored. These are demonstration references, not proven optima.

Action references use successful GT replay action lists for 88 variants and
archive actions.csv for the two failed-replay fallback variants. Action source
paths, original counts, and hashes are saved in tc_spl_data.json. The existing
step-based TC-Steps is unchanged. Current 450-run means: VLM-TAMP 0.3779658901,
ReAct 0.5912806327, TRU-POMDP 0.3954877128. These use latest attempts only.

## Copied rule findings

If the batch contains `uncertainty_classifier/dashboard_data.json`, the Svelte
`RuleFindings` section loads that snapshot during dashboard generation. The complete
viewer and original classifier files are copied locally, and all trace links are
relocated into the current user's batch. `import_rules.py SOURCE BATCH` imports
and validates a fresh copy; it refuses to overwrite an existing source snapshot.

`verify_rules.py` checks original hashes, links, CSV/viewer agreement at import,
filter and drilldown behavior, and the relocated viewer's trace fetching over a
local HTTP server. Classifications remain explicitly rule-based and may overlap.

### Failure-combination Sankeys

`FailureSankey.svelte` shows selected runs -> successful/unsuccessful -> exact
sets of failure labels. `failure_groups.js` assigns each unsuccessful run once,
deduplicates labels, and keeps the six most frequent labeled sets separate.
Remaining labeled sets are collected under Other combinations; unlabeled failures
always get an explicit No failure finding branch. Branch fractions use failed
runs as denominator. Outcomes use all selected runs. These describe association,
not causal attribution. Clicking any branch reveals its runs and all evidence.
The category selector switches both bars and Sankeys. Failure mode partitions
unsuccessful runs by failure-label sets. Inefficiency mode partitions all selected
runs into unflagged/flagged and divides flagged runs by inefficiency-label sets;
its branch percentages use flagged runs. Successful runs can be flagged. Both
modes respond to task, memory, scenario, and outcome filters. Other combinations
always appears last; all remaining branches are sorted by descending count.

`node test_failure_groups.mjs` (from this directory) verifies run conservation,
exclusive assignment, duplicate-label handling, missing labels and empty input.
The browser checks in verify_rules.py include branch drilldowns and filters.

Finding rates use two horizontal grouped bar panels (failure and inefficiency),
with a fixed VLM-TAMP / ReAct / TRU-POMDP order for every label. Rows sort descending
by the unweighted mean of the available baseline percentages. Each bar deduplicates
runs, shows a percentage on a common 0–100% scale, and exposes its count/denominator
on hover. Missing denominators render as a dash and are excluded from the sorting
mean. Clicking a bar selects its category, baseline, and label in the evidence
panel. The existing task/memory/scenario/outcome filters apply to both panels;
the Category selector applies to the Sankeys and evidence. No UpSet section is added.

The independent `Task results by scenario and initial memory` section uses the
scenario viewer's seven-column layout, with all three baseline panels per task.
Its task selector supports individual tasks and all seven tasks together. Success
is mean binary evaluator success; completion is mean saved evaluator completion,
with equal weight per repetition and missing metrics excluded. Each cell reports
coverage and opens the saved repetitions. Repeated BASE columns share the same
runs; unavailable variants show No runs, not zero. `verify_scenario_grid.py`
compares all 441 displayed cells against the saved metrics for both selectors.

Dashboard injection removes the legacy Per-variant results and Subtask successes
and failures by task sections, including their event handlers and the old variant
detail panel. The Svelte sections mount before Metric definitions; the new task
scenario grid continues to provide per-variant repetition details. Both direct
metric refreshes and regeneration from the saved batch data preserve this layout.

Scenario and initial-memory line charts are a separate `#scenario-memory` section
below Completion and execution efficiency. Their own task and metric selectors
operate independently of the completion section's filters.

Completion and execution efficiency breakdown line plots have their own metric selector:
Task success, Task completion, SPL, TC-Steps, and TC-SPL (default). Point tooltips and
coverage follow that selection, and its column is highlighted in the breakdown
table. Task/memory/scenario filters and run drilldowns apply to every selection.

The top comparison also includes TC-SPL and labels its Y axis Score (%). Each
line breakdown labels its selected percentage metric on the Y axis; TC-SPL is highlighted
in the summary by default.

Compare by now renders one line per baseline (distinct colors, dashes, and markers),
defaulting to Accurate / Incomplete / Outdated memory on the X axis and TC-SPL on
the Y axis. Task and scenario groupings and all five metric choices remain
available. Points open the same filtered run details; missing scores stay gaps.

The Compare by Scenario line plot follows Target Exists, Substitute Available,
No Suitable Object, On Surface, Inside Closed Receptacle, None, Present. The
three BASE reference columns reuse the same cohort without changing aggregate
run totals. Missing variants retain their labeled positions with gaps.

Display naming: TC-Steps is completion weighted by simulator-step efficiency;
TC-SPL is completion weighted by high-level action efficiency (formerly TC-AE).
Calculations are unchanged. Historical JSON keys remain `tcspl` for TC-Steps
and `tcae` for TC-SPL; `metric_labels` records this mapping in the export.

Completion and execution efficiency now exposes only Task success, Task completion,
SPL, and TC-SPL. TC-Steps is removed from that section's charts, selector, summary,
and run details; it remains available in the separate scenario/memory section and
in the metric export.

`#memory-scenario-heatmap` is a separate 9x5 TC-SPL section. It groups three
memory rows per baseline and shows BASE, SUB, ABS, CON, DIS exactly once, with
one shared 0–100% color scale. It defaults to all tasks; each cell averages finite
`tcae` (action-based TC-SPL) values equally and offers coverage and repetition
links. Missing values are gray dashes. `verify_heatmap.py` validates all 360 cells
across the all-task and seven individual-task selections against saved runs.

Task scenario cells include a Cell findings selector (Failure, Inefficiency, None).
Each selected category has a shared color key and a proportional strip inside
its cells. Failure strips use runs with known unsuccessful evaluator outcomes;
inefficiency strips use all runs. A run with k distinct labels contributes 1/k
to each, so strip shares sum to 100% without duplicating run mass. Repeated events
of the same label do not increase its share. No finding and missing annotation
are retained as explicit gray segments. Tooltips distinguish allocated share
from raw affected-run counts. These allocations describe association, not cause.
Per-repetition cell details show matching rule evidence. `verify_cell_findings.py`
checks 882 cell/category combinations against independent fractional arithmetic.

Finding-rate bars share a fixed, linear 0–100% axis with ticks every 20%.
Both panels retain the full range without breaks or compression.

SPL now uses binary success weighted by human-reference action efficiency (`spl`).
SPL_S preserves the previous simulator-step score (`spl_s`). Both are selectable
in completion comparisons and scenario/memory plots. TC-SPL is unchanged.

The former heatmap section now directly shows Change from BASE: a grouped
TC-SPL delta chart for SUB, ABS, CON, and DIS. Each comparison restricts both
scenario and BASE to their common available tasks, within baseline and memory.
Bars always show the equally weighted mean across the three memories
(requiring all three); values are percentage-point differences.
