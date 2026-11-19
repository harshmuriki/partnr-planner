---
name: rail-sweep
description: Launch, resume or check status of a multi-rep RailPlanner sweep on all baseline_evaluation_v3 tasks on Slurm (rail-lab + overcap) with a plan_config override, e.g. replan_trigger=oracle, end_of_plan_check=oracle_feedback, placement_oracle. Use for "run all tasks N times with X", "status of the runs", "rerun task 4 with ...".
---

# RailPlanner sweeps on Slurm

Paths are relative to the repo root. These runs make **paid OpenAI calls** (450 runs for 90 variants x 5 reps);
launch only when the user asked for the run. Conda env: `habitat-llm`. Never print `.env` values.

## Launch

```bash
# Preview: prints shards, accounts, timeouts and run count; writes nothing.
python3 .claude/skills/rail-sweep/launch_rail_sweep.py rail_<what>_5reps_<YYYYMMDD> \
    --override evaluation.agents.agent_0.planner.plan_config.replan_trigger=oracle --dry-run
# Same command without --dry-run submits 21 shards (task x memory).
```

- `--snapshot-from results/<other_batch>/snapshot` copies another batch's frozen code, so two conditions differ
  only in the override (used for the end-of-plan vs new-info comparison). Otherwise the current repo code is
  frozen into `results/<batch>/snapshot/` (habitat_llm, scripts, dataset_generation copied; data, third_party
  symlinked); `snapshot_rail_planner.sha256` records the rail planner files.
- Options: `--tasks 4`, `--memories acc`, `--reps 5`. Shards go to rail-lab, overflowing to overcap only when rail-lab is full (`-p rail-lab,overcap`), `--requeue`,
  A40, 12 CPUs, 128G, 4 h; 6 workers; timeouts 600 s (T1-T5) / 1200 s (T6-T7), matching the baseline batch.
- Overrides used so far (`evaluation.agents.agent_0.planner.plan_config.` prefix): `replan_trigger=oracle`
  (oracle replanning on new task-relevant observations), `end_of_plan_check=oracle_feedback` (oracle check when
  the plan finishes, replans with unmet-goal feedback), `placement_oracle=...`. Config defaults stay
  failure / none / none: do not change `habitat_llm/conf/planner/rail_planner.yaml` defaults.

## Status

```bash
python3 .claude/skills/rail-sweep/sweep_status.py results/<batch>
```

Prints Slurm state per shard and runs done / success / timeout per task. To confirm an override is live, check
`shards/*/attempt_00/manifest.json` for it and the `*/vlm_tamp_pddl_log.jsonl` event logs (Python dict reprs,
parse with `ast.literal_eval`) for the expected events (`replan_trigger`, `end_of_plan_check`).

After it finishes, use the `rail-batch-analysis` skill for rows, comparisons and figures.

## Gotchas

- Slurm FAILED with all runs present just means a run timed out (run_parallel exits 1 on any timeout).
- The snapshot has no `.env`; `shard.sh` exports `OPENAI_API_KEY` from the repo `.env` and stops if missing.
  `run_parallel` itself loads `.env` from its own parent dir, which inside a snapshot does not exist.
- sbatch copies the script at submit time: after editing `shard.sh`, cancel and resubmit pending shards too.
- On requeue a shard makes `attempt_NN+1` and reruns only runs without a completed/timeout `run.json`.
- `run_parallel` refuses an existing output dir; the launcher refuses an existing batch name.
- Other sessions may own batches in `results/` (e.g. `rail_planner_5reps_20261028` reruns); don't modify theirs.
