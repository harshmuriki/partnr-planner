---
name: baseline-batch-rebuild
description: Replace or add runs in the 5-rep baseline batch (ReAct / VLM-TAMP / TRU-POMDP, results/all_baselines_5reps_*) - e.g. swap in a rerun of one planner's task - and recompute everything derived from it - baseline_results.json, dashboard, TC-SPL/SPL metrics, rule-based failure/inefficiency findings and viewer, and paper figures 01-10 - in a new folder, leaving the original untouched. Also use to just regenerate the metrics or paper figures of a baseline batch.
---

# Rebuilding a baseline batch

Paths relative to the repo root. Offline (no simulator or API calls). Current batch with the VLM-TAMP T4 basket
rerun: `results/all_baselines_5reps_vlmtamp_t4basket_20261030` (its README lists what was done and the
validation numbers). Never modify `results/all_baselines_5reps_20261005_001408`: make a new folder.

Layout the scripts expect: `<planner>/<VARIANT>/repeat_<N>/attempt_*/summary.json` (`{"jobs": [job]}` with
`output` = run dir) + `runs/<run_id>/`; `batch.json` lists allocations; `status.py` picks the newest attempt.

## 1. Copy (background; ~30-40 min)

```bash
.claude/skills/baseline-batch-rebuild/copy_batch.sh results/<source_batch> results/<new_batch> 'vlm-tamp/T4-*'
```

Runs and snapshot are hardlinked, other files copied, the listed variants hardlinked into
`archive/<planner>_replaced/`, old figure folders into `archive/`, and the copy's `status.py` patched to map
the original folder's paths to the copy.

## 2. Put the new runs in place

Write a small import script in the new batch (model: `results/all_baselines_5reps_vlmtamp_t4basket_20261030/import_vlm_tamp_t4_basket.py`).
Per run: hardlink `runs/<run_id>` (+ `outputs_<run_id>`) into
`<planner>/<VARIANT>/repeat_<N>/attempt_<tag>/runs/` (N from the `_rNNN` suffix) and write a one-job
`summary.json` with `output` pointing there. Use a new attempt name (not `attempt_000`): `retry_plan.json`
marks slots pending if all attempts are in its `previous_attempts` list.

## 3. Recompute

```bash
.claude/skills/baseline-batch-rebuild/rebuild_batch.sh results/<new_batch> paper_figures_<YYYYMMDD>
# or a subset of stages: rows | findings | metrics | figures
```

- `rows`: `status.py` + `build_results_dashboard.py` → `summary.json`, `baseline_results.json`, dashboard.
- `findings`: classifier (`/usr/bin/python3`, 3.10+) on all runs + viewer, imported with
  `scripts/dashboard_tc_spl/import_rules.py`; the previous `uncertainty_classifier/` goes to `archive/`.
- `metrics`: `scripts/dashboard_tc_spl/metrics.py` → `tc_spl_data.json`, dashboard TC-SPL section.
- `figures`: `figure_tools/` (batch-agnostic copies of `scripts/dashboard_tc_spl` exporters, paper order
  ReAct, VLM-TAMP, TRU-POMDP) → 01-04, 06-08 from the dashboard (Playwright), 01/02 redrawn in matplotlib from
  `metric_percentages.csv`, 09/10 heatmaps, 11 copied. Replace the figure folder's auto README with one
  saying what changed.

## 4. Validate (do all of these)

- Rows: unchanged runs' rows equal the original's (compare `task_state_success`/`completion`/`counts`/
  `num_actions`/`sim_steps` by id); swapped rows equal the source batch's rows; every `run_folder` is inside the new batch.
- Findings: non-swapped findings (run_id, category, label, evidence) equal the original
  `uncertainty_classifier/source/findings.csv`; swapped ones equal the rerun's own findings.
- Overall metrics vs `results/rail_planner_5reps_20261028/paper_figure/overall_with_rail.csv` (uses the same
  VLM-TAMP T4 swap via `baseline_rows.py`).
- Original untouched: `touch` a marker before starting, then
  `find results/<source_batch> -newer <marker>` - only `summary.json` / `status_watch.log` (a status watcher
  already running there) may appear. A modified hardlinked file would show up here.
- Look at the PNGs.

## Gotchas

- Hardlinked `summary.json` files name the original folder; never edit them (it edits the original) - the
  patched `status.py` remaps instead. `build_results_dashboard.py` fails with "is not in the subpath" without it.
- `scripts/dashboard_tc_spl/export_figures.py` is stale: it selects a "Heatmap task" control that no longer
  exists; the dashboard section `#memory-scenario-heatmap` now holds the change-from-BASE chart (figure 08).
  The old 9x5 TC-SPL heatmap (05) cannot be regenerated. `figure_tools/export_figures.py` has the fix.
- Headless Chromium on sky2 needs `libasound.so.2`; `rebuild_batch.sh` symlinks the one in
  `~/miniconda3/pkgs/alsa-lib-*/lib` into a temp dir on `LD_LIBRARY_PATH`.
- Dashboard-exported figures (03, 04, 06-08) use the compiled `scripts/dashboard_tc_spl/bundle.js` order
  (VLM-TAMP, ReAct, TRU-POMDP; colors assigned by index). Reordering them needs Node (not installed) to rebuild
  the Svelte bundle.
- The repo `scripts/dashboard_tc_spl/*` exporters hardcode `all_baselines_5reps_20261005_001408`; use
  `figure_tools/` (reads `BATCH_DIR` / `FIG_DIR`), don't edit the originals.
- `/coc/flash5` is near full: hardlink, don't copy run folders.
