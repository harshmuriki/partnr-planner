# Uncertainty Classifier

> **Copy.** Copied on 2026-11-18 from `/coc/flash5/garavindan3/uncertainty_classifier/` (without `.openai_api_key`,
> `.judge_work/` and `__pycache__/`) and extended with RailPlanner batches; see "RailPlanner additions" at the end.

Finds and explains errors in the robot planner runs (ReAct, VLM-TAMP, TRU-POMDP) of the PARTNR baseline batches.
It has two steps, followed by a viewer:

1. **`classify_errors.py`**: rule-based **observed errors** (failures and inefficiencies) from each run's logs.
2. **`llm-as-a-judge.py`**: an LLM explains the **underlying cause** of each observed error.
3. **Error Viewer**: a web page for browsing both.

Data locations are listed in `DATA_PATHS.md`, and the files in each run folder are described in `DATA_OUTPUTS.md`.

## 1. `classify_errors.py`: observed errors

```bash
python3 classify_errors.py RESULTS_DIR --out findings_<name>.csv
```

- `RESULTS_DIR` is a batch folder containing `baseline_results.json`, e.g.
  `/coc/flash5/vmuriki3/tamp/partnr-planner/results/all_baselines_5reps_20261005_001408`.
- The output CSV has one row per finding: `run_id, kind, category, label, evidence`.
  - `kind` is `failure` (only checked on failed runs) or `inefficiency` (checked on every run).
  - `category` equals `label`, except that Repeated Action is counted under Unnecessary Action.
- **Failures:** Incorrect Placement, Object Not Found, Object Untouched, Wrong Object, Wrong Order, Missing Action,
  Missing Prerequisite, Action Undone, Task Abandoned, Time Limit Reached, Action Limit Reached.
- **Inefficiencies:** Nonexistent Object, Unnecessary Action, Repeated Action, Precondition Error, Affordance Error,
  Excess Search.
- A full run takes about 1.5 minutes per 1,350-run batch.

Current outputs:

| File | Batch |
|---|---|
| `findings_luna_all_planners.csv` | `all_baselines_5reps_20261005_001408` |
| `findings_luna_all_planners_t4basket.csv` | `all_baselines_5reps_vlmtamp_t4basket_20261030` |
| `findings_astra_react.csv` | `react_gpt6astra_medium_3reps_20261110` |
| `findings_astra_react_placement_oracle.csv` | `react_gpt6astra_placement_oracle_t3t5_3reps_20261111` |

## 2. `llm-as-a-judge.py`: underlying causes

```bash
python3 llm-as-a-judge.py RESULTS_DIR findings_<name>.csv --out judge_<name>.jsonl [--limit N] [--runs ID ...] [--dry-run]
```

- The judge is called once per run that has findings. It gets the task, the variant spec, the robot's starting memory,
  the outcome, every action, the planner's trace and the observed errors (`e1..eN`). It returns, for each observed
  error, a cause from AgentErrorTaxonomy (Memory, Reflection, Planning, Action, System), plus our Goal
  Misinterpretation and Framework Limitation types. Each cause has a step, evidence, a description and a confidence.
  The judge may also report genuine errors the rules missed, with `error_id` set to `extra`.
- Answers are checked (`check()`) and re-asked up to 2 times if invalid.
- The model is set by `MODEL` (`gpt-6-astra`), and the API key is read from `.openai_api_key`.
- `--dry-run` prints the first prompt without calling the API. Rerunning the same command skips runs already in `--out`.
- The current `judge_*.jsonl` files (one per findings file) were produced with `.judge_work/run.py`. That runner uses
  the same prompt and checks, but calls the model through the Codex CLI under a ChatGPT account.

## 3. Error Viewer

Build the page (about 20 seconds), which reads all four findings and judge files:

```bash
python3 build_viewer.py          # writes viewer/error_viewer.html
```

Serve it from `/coc/flash5`, so the links to traces and run folders work:

```bash
python3 -m http.server 8010 --bind 127.0.0.1 --directory /coc/flash5
```

Then open **http://localhost:8010/vmuriki3/tamp/partnr-planner/uncertainty_classifier/viewer/error_viewer.html**. In VS Code on a
remote machine, forward port 8010 first (Ports panel → Forward a Port).

The page has a batch selector and filters, prevalence charts for failures and inefficiencies, the judge's causes,
and a runs table. Click a run to see its findings, the judge's explanations and the actions they point to.

## RailPlanner additions (2026-11-18)

- `classify_errors.py`: `Run.vlm_tamp_family` treats ids starting with `rail` like `vlm-tamp` (RailPlanner subclasses
  VlmTampPddlPlanner: same rerun dedup, `memory_snapshot` events, `Planner decision: unsolvable` give-up check), and
  `Run.vlm_log` reads `rail_planner/vlm_tamp_pddl_log.jsonl` for those runs.
- `build_viewer.py`: a batch can combine several results dirs (ids must not collide), and the judge file is optional
  (`None`).
- `viewer/template.html`: Rail baselines with their own colours (`--series-4..8`); baseline order is ReAct, VLM-TAMP,
  TRU-POMDP, then the Rail configurations.
- New findings (rules only, **no LLM judge run**):

| File | Batch | Viewer id prefix |
|---|---|---|
| `findings_rail.csv` | `rail_planner_5reps_20261028` | `rail` |
| `findings_rail_placement_oracle.csv` | `rail_placement_oracle_t4_basketfix_20261029/merged` | `rail-oracle-fix` |
| `findings_rail_eop_feedback.csv` | `rail_end_of_plan_feedback_5reps_20261029` | `rail-eopfb` |
| `findings_rail_replan_trigger_oracle.csv` | `rail_replan_trigger_oracle_5reps_20261029` | `rail-trig` |
| `findings_rail_placement_oracle_eop_feedback_wait_merged.csv` (viewer) | `rail_placement_oracle_eop_feedback_wait_t6t7_5reps_20261118/merged` (T1–T5 old batch, T6–T7 rerun with `waited_in` at 1200 s) | `rail-oracle-eop` |
| `findings_rail_placement_oracle_eop_feedback.csv` (superseded) | `rail_placement_oracle_eop_feedback_5reps_20261029` | `rail-oracle-eop` |

- New viewer batches: "baselines (T4 basket fix) + RailPlanner" (ReAct / VLM-TAMP / TRU-POMDP with judge, plus plain
  Rail without) and "RailPlanner configurations" (all five, no judge). The oracle configurations use privileged ground
  truth: upper bounds, not deployable. Only the placement oracle + EoP feedback batch includes the `waited_in` subgoal (its T6–T7
  rerun); the other Rail batches lack it, so their T6 is 0%.
