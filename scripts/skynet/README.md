# Ground-truth timing on Skynet

Replays every saved ground-truth sequence (exact skill/target steps from
`baseline_evaluation_v3/ground_truth.sqlite3`) headlessly on A40s and measures timing with the
same worker timers as the local T6 batch ([docs/ground_truth_timing_handoff.md](../../docs/ground_truth_timing_handoff.md)).
Nothing is saved to the DB or archive. A replay counts only if every action and the final
evaluation succeed.

| File | Role |
| --- | --- |
| `submit_gt_timing.sh` | Entry point on sky1: preflight, build queue, submit the array + summary jobs |
| `gt_timing_shard.sbatch` | One array task = 1 A40 + 15 CPUs, replays one shard sequentially |
| `gt_timing_replay.py` | `queue` / `run` / `summarize` / `publish` |

## 1. Copy the repo and data to Skynet

Sync **after** the local T6 batch finishes. Its re-recordings change T6 run IDs, and a queue
built from an older DB is skipped by `publish`.

```bash
# from the local machine; DEST is project storage, not $HOME (30 GB quota)
DEST=sky1.cc.gatech.edu:/coc/<your-storage>/partnr-planner
rsync -ahP --exclude data --exclude '.git' --exclude 'baseline_evaluation_v3/ground_truth/' ./ "$DEST/"
rsync -ahP --copy-links --exclude scene_datasets data/ "$DEST/data/"   # ~30 GB; resolves local symlinks
```

You also need the `habitat` conda env (habitat-sim headless build, CUDA ≤ 11.x on Skynet) with this
repo installed (`pip install -e .`). The submit script refuses to run if `data/` has broken symlinks.

## 2. Smoke test, then the full run

```bash
ssh sky1.cc.gatech.edu && cd /coc/<your-storage>/partnr-planner
PARTITION=<your-lab> scripts/skynet/submit_gt_timing.sh --smoke     # shortest variant of each task, 1 GPU
cat baseline_evaluation_v3/ground_truth_timing/skynet_*_smoke/logs/summary.out
PARTITION=<your-lab> scripts/skynet/submit_gt_timing.sh             # all 90 variants on 8 A40s
```

Options are environment variables (`QOS`, `ACCOUNT`, `SHARDS`, `REPEATS`, `TIME_LIMIT`,
`HABITAT_PYTHON`, `TAG`) plus `--variants T6-OUT-CON ...`. Run `--help` for the list. `REPEATS=3`
replays each variant three times and reports the median, which is worth it for timing.
Shards are balanced by saved simulator steps. A requeued array task skips attempts already replayed.

## 3. Read the results

In `baseline_evaluation_v3/ground_truth_timing/<TAG>/`:

- `summary_tasks.csv`: per-task mean/median/max elapsed and mean active skill time, plus an `ALL` row
  (each recording weighted once, like the Metrics page)
- `summary_variants.csv`: per variant elapsed, active, env.step, frame callbacks, worker CPU, load, sim steps
- `summary_skills.csv`: time per skill type and **seconds per simulator step**
- `summary_slowest_actions.txt`: the 25 slowest single actions. Start here for the T6 question.
- `results/<VARIANT>.r<N>.json`: full per-action evidence; `gpu/shard_*.csv`: GPU utilization every 2 s;
  `conditions/`: node, CPU, GPU, Slurm settings; `logs/`: Slurm output

`worker_cpu_seconds ≈ elapsed` means the action was CPU-bound. Much lower CPU time with low GPU
utilization means the worker was waiting.

## 4. Publish to the Metrics page (local machine)

```bash
rsync -ahP sky1.cc.gatech.edu:/coc/<your-storage>/partnr-planner/baseline_evaluation_v3/ground_truth_timing/<TAG> \
      baseline_evaluation_v3/ground_truth_timing/
python3 scripts/skynet/gt_timing_replay.py publish --out baseline_evaluation_v3/ground_truth_timing/<TAG>
```

`publish` writes `measured_skynet` rows into `current_recordings.csv` (backup kept in the batch
folder). It only publishes a variant whose saved run ID and spec/dataset hashes still match. It replaces
older timing rows for those variants, so per-task and overall averages all come from one Skynet batch.
Variants that failed or were skipped keep their old rows, and `publish` lists them. The interval is
`first_action_submit_to_final_action_return_headless`. It has no HTTP server or 0.2 s client polling,
so it is slightly tighter than the local sandbox interval. Different hardware also means these numbers
are not directly comparable to the local runs.
