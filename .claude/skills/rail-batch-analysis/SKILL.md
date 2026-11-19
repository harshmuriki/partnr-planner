---
name: rail-batch-analysis
description: Analyze a finished RailPlanner sweep - build baseline_results.json rows, compare success/completion per task or variant against the rail planner and baselines, count end-of-plan replans and replan-trigger firings, diagnose a per-task regression, make paper figures (overall metrics bars, per-task success, replanning outcomes) and failure/inefficiency heatmaps with the rule-based classifier. No API calls.
---

# Analyzing a rail sweep

Paths are relative to the repo root; habitat-llm env for matplotlib/row building. Everything here is offline.
Reference batches: rail planner `results/rail_planner_5reps_20261028` (T4 = basket rerun; owned by another
session, read only), oracle end-of-plan `results/rail_end_of_plan_feedback_5reps_20261029`, oracle new-info
`results/rail_replan_trigger_oracle_5reps_20261029`. Baselines with VLM-TAMP T4 basket rerun:
`results/all_baselines_5reps_vlmtamp_t4basket_20261030/baseline_results.json`.

Paper order and colors (Okabe-Ito): ReAct #D55E00, VLM-TAMP #0072B2, TRU-POMDP #CC79A7, Rail #009E73,
Rail + end-of-plan replanning #E69F00, Rail + new-info replanning #56B4E9.

## 1. Rows

```bash
python .claude/skills/rail-batch-analysis/batch_rows.py results/<batch> --condition rail_<x>_images --prefix rail-<x>
```

Writes `results/<batch>/baseline_results.json` (latest attempt per run): metrics fields plus `reached_plan_end`,
`sent_back`, `recovered`, `end_of_plan_replans`, `trigger_checks`, `trigger_fired`, `trigger_rules`. The
`--prefix` is the id's first segment, used by the classifier and heatmaps; `--condition` is the metrics key.

## 2. Compare

```bash
python3 .claude/skills/rail-batch-analysis/compare_tasks.py \
    rail=results/rail_planner_5reps_20261028/baseline_results.json new=results/<batch>/baseline_results.json \
    [--task 5 --variants]
```

Diagnosing a task: each row's `subtask_results.criteria` lists every goal proposition with `status` (met/unmet)
and `label` (e.g. `glass_0 on table_0`); count unmet labels per condition. For what the planner did, parse
`<run_folder>/*/vlm_tamp_pddl_log.jsonl` (Python dict reprs, `ast.literal_eval`): `subgoal_status` (subgoal,
status solved/failed, failure_msg), `vlm_predicate_subgoals`, `planner_decision`, `replan_trigger`
(delta, rule, fired), `end_of_plan_check` (goal_satisfied, replan). Example finding: T5 new-info drop came from
the VLM choosing the wrong table, not from the trigger. Check rate differences against noise (5 reps).

## 3. Figures

Templates in `results/rail_end_of_plan_feedback_5reps_20261029/paper_figure/` - copy into
`results/<batch>/paper_figure/` and edit the inputs/`PLANNERS` list:

- `make_overall_figure.py`: task success, SPL_act, completion, TC-SPL_act bars via
  `scripts/dashboard_tc_spl/metrics.py` `payload(rows, root)`; baselines through
  `results/rail_planner_5reps_20261028/baseline_rows.py` `baseline_rows()`. Writes `overall_combined.*` + CSV.
- `make_replanning_figures.py`: per-task success (hatched = recovered at plan end), end-of-plan outcomes,
  trigger firings by rule and success split by fired / not fired; `by_task*.csv`.
- `make_findings_heatmaps.py`: figures 09/10 (% of all runs; failures counted on failed runs only).

Look at every PNG after writing it (clipped labels, legends missing a series whose first group is empty).

## 4. Failure / inefficiency findings

```bash
mkdir -p results/<batch>/uncertainty_classifier
cp results/rail_planner_5reps_20261028/uncertainty_classifier/classify_uncertainties.py results/<batch>/uncertainty_classifier/
# add the batch's id prefix to RETRYING_EXECUTORS (rail logs each failed skill twice, like VLM-TAMP)
cd results/<batch>/uncertainty_classifier && /usr/bin/python3 classify_uncertainties.py .. --out findings.csv
```

Needs Python 3.10+ (`/usr/bin/python3`; the habitat-llm env is 3.9). Reads `<batch>/baseline_results.json`.
Rail/baseline findings for comparison: `results/rail_planner_5reps_20261028/uncertainty_classifier/combined/`.

## Gotchas

- Importing two different `build_rows.py` modules by name collides in `sys.modules`; load by path (importlib).
- Rail rows outside T4 ran on 28 Oct code; the oracle batches use the 29 Oct snapshot - say so in captions.
- Slurm FAILED shards with all runs present are timeouts only.
- Oracle conditions are upper bounds: only a replan bit / unmet-goal feedback reaches the planner, never where
  things belong. Keep that wording in captions.
