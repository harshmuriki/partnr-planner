# Parallel PARTNR runs on one allocated GPU

`scripts/run_parallel.py` queues single-episode VLM-TAMP or ReAct runs. The default
is six concurrent processes, with two CPU threads per process. More queued tasks
start as workers finish. The existing sequential wrapper is unchanged.

Run from the repository root with the Habitat environment's Python:

```bash
cd /coc/flash5/vmuriki3/tamp/partnr-planner
export PATH=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin:$PATH
```

## Preview and run

The following previews the available v3 `inc-con` episodes with six workers
(currently six matching folders; task 7 has no `inc-con` variant):

```bash
python scripts/run_parallel.py \
  --episodes 'baseline_evaluation_v3/episodes/task_*/t*-inc-con' \
  --workers 6 --job-id 3965858 --dry-run
```

Remove `--dry-run` to execute. Replace `3965858` with your current running Slurm
allocation. The launcher dispatches one `srun` step into that allocation. If
already inside an allocated GPU step, omit `--job-id`. An actual run requires
a visible GPU; running directly on a login node is rejected.

To run six repetitions of the tested episode concurrently:

```bash
python scripts/run_parallel.py \
  --episodes baseline_evaluation_v3/episodes/task_2/t2-inc-con \
  --runs-per-episode 6 --workers 6 --job-id 3965858
```

To queue all v3 episode folders with six workers:

```bash
python scripts/run_parallel.py \
  --episodes 'baseline_evaluation_v3/episodes/task_*/*' \
  --workers 6 --job-id 3965858
```

Each episode file must contain exactly one episode. Directories must have
`dataset.json` or `dataset.json.gz`. JSON inputs are compressed into each run's
private input directory without changing the source. Initial robot memory,
scene info, spec, and review sidecars are copied with them. Custom runtime YAML
files are not automatically applied; this launcher targets the v3 episode files.
Repeated runs use the planner's existing fixed simulator seed; they are separate
model/planner executions, not a simulator-seed sweep.

These commands make API calls. Dry runs and `--probe` do not. No paid parallel
episode batch was launched during the capacity validation below.

## Concurrency and limits

- `--workers 5` or `--workers 6` sets the maximum number of active episodes.
- `--workers auto` uses allocated CPUs divided by `--cpus-per-worker` (default 2),
  capped by the number of queued runs. With 16 CPUs this is at most eight workers.
  This is a CPU limit, not a measured universal GPU/API throughput maximum.
- `--timeout 600` (the default) enforces 10 minutes of actual wall time per run, including exploration and VLM
  latency. On timeout, the entire child process group, including solvers, is killed.
  This does not rely on VLM-TAMP's incomplete combined-time telemetry.
- VLM-TAMP uses normal physical Explore (`explore_fast: false`). Its internal
  budget also includes exploration (`exclude_explore_from_time_budget: false`);
  that simulation-time accounting is separate from the wall-clock timeout above.
- Ctrl-C or SIGTERM stops the active process groups and marks queued runs cancelled.
  The enclosing Slurm allocation is not cancelled.
- All workers share the GPU visibility granted by Slurm. This launcher targets one
  node; it does not distribute workers across multiple GPUs or nodes.
- CPU thread variables are bounded, but RAM and GPU memory are not reserved per
  worker. Scene complexity, API rate limits, recording, and model latency affect
  sustainable concurrency. Check the saved resource samples when changing workloads.
- `--planner react` selects ReAct instead of the default VLM-TAMP. The planner is
  common to one invocation; use distinct output roots for separate planner batches.
- `--planner tru-pomdp` selects Tru-POMDP. Its hypothesis-generation requests are
  text-only by construction, and it uses normal physical Explore. Set its model
  using `--override evaluation.agents.agent_0.planner.plan_config.llm.generation_params.model=gpt-5.6-luna`
  and reasoning using `--override evaluation.agents.agent_0.planner.plan_config.llm.generation_params.reasoning_effort=high`.

VLM-TAMP defaults to the tested per-run override
`plan_config.vlm.max_completion_tokens=8192` because 1,200 tokens produced empty
reasoning-only responses. Override configuration explicitly when needed:

```bash
python scripts/run_parallel.py \
  --episodes baseline_evaluation_v3/episodes/task_2/t2-inc-con \
  --job-id 3965858 --workers 1 \
  --override evaluation.agents.agent_0.planner.plan_config.vlm.max_completion_tokens=8192
```

The scheduler fixes input/output paths, single-process/single-episode execution,
CPU policy device, video selection, and disabled PrediViz after user overrides. Add
`--save-video` to record each episode (default: off). Videos are encoded during
execution as 1280×720 H.264 at 30 fps, using the ultrafast preset, CRF 28 and one
encoder thread. Camera frames are fitted without cropping. Set the third-person
sensor to 1280×720 as well for native 720p rendering. Fragmented MP4 preserves
completed fragments if execution is interrupted; startup failures before the
first captured frame cannot produce a video. Encoding counts toward `--timeout`. Habitat still
renders on the GPU. The OpenAI key is inherited or loaded from the original
repository's `.env`, never copied into the output manifest or worker directories.

## Images sent to the planner

Both baselines already have image switches, enabled by default. With the parallel
launcher, append the appropriate override to disable sending images:

```bash
# VLM-TAMP (--planner vlm-tamp)
--override evaluation.agents.agent_0.planner.plan_config.vlm.use_images=False

# PARTNR / ReAct (--planner react)
--override evaluation.agents.agent_0.planner.plan_config.send_action_image=False
```

Use `True` to enable images. VLM-TAMP attaches annotated scene images to its
English planning requests, initially and on replanning. After physical Explore,
it sends up to three sampled views (first, middle, last). Its predicate-translation
turn adds no new image attachment. PARTNR / ReAct attaches a current robot camera
image at each LLM replan, including initial planning and subsequent action
feedback. Neither switch makes the planner query the model on every simulation
step; replanning rules stay the same. If a camera image is unavailable, requests
can fall back to text only.

`evaluation.use_rgb` controls observation availability, not whether an API request
contains images. Keep it enabled and use the planner switches above for text-only
requests. Trace screenshots are separate from API attachments, so an image in an
HTML report does not by itself prove it was sent to the model.

Both single-agent baselines use normal physical exploration: VLM-TAMP has
`plan_config.explore_fast: false`, and PARTNR / ReAct has
`plan_config.fast_explore: false`. The launcher's 600-second wall-clock timeout
includes exploration regardless of whether images are sent.

## Worker isolation and results

Every invocation creates a timestamped directory under `results/`. You may supply
`--output-dir`, but it must not already exist. No earlier result is overwritten.

```text
results/parallel_<timestamp>/
  manifest.json             # exact sources, planner overrides, concurrency
  summary.json              # running and final states, task metrics, timings
  resources.jsonl           # sampled GPU memory/utilization every ~10 seconds
  workers/0/ ... workers/5/  # separate source copies and relative temp/log files
  runs/<episode>_<hash>_r001/
    input/                  # immutable-for-run dataset and memory sidecars
    log.txt                 # command, startup, and all process stdout/stderr
    console.log -> log.txt   # compatibility alias
    run.json                # command, worker, status, final metrics
    dataset/                # planner logs and ordinary traces
    vlm_tamp_pddl/          # VLM-TAMP interactive HTML and solver logs
```

Workers have separate `habitat_llm`, `scripts`, and `dataset_generation` copies.
Large assets and the installed third-party source/builds are shared by symlink.
Do not edit shared assets or rebuild dependencies while a batch runs.
PDDLStream's working-directory `temp/`, generic logs, and Hydra outputs are
isolated. API prompt logs are additionally directed to each run's output.
The launcher creates no environment clones and downloads no datasets.

The launcher opens `log.txt` before starting the simulator command, so imports,
native simulator messages, and failures before episode setup are captured too.
Episode logging uses this same output stream without duplicating messages.

`completed` means the process exited zero, no caught traceback was found, and
final task metrics were written. **It does not mean task success.** Inspect
`metrics.task_state_success` and `metrics.task_percent_complete` in `summary.json`.
The launcher returns nonzero for setup failures, timeouts, and cancellations;
normal benchmark failure remains a completed run. Empty runs and missing metrics
are failures even if the planner swallowed an exception and exited zero.

Monitor from another terminal:

```bash
cat results/parallel_<timestamp>/summary.json
tail -f results/parallel_<timestamp>/runs/<run-id>/log.txt
```

For unattended work, submit the coordinator as a Slurm batch job rather than
depending on a connected terminal. For example, a fresh 16-CPU/A40 allocation:

```bash
sbatch --partition=rail-lab --gpus=a40:1 --cpus-per-task=16 \
  --time=04:00:00 --job-name=partnr-parallel \
  --chdir=/coc/flash5/vmuriki3/tamp/partnr-planner \
  --wrap="/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python scripts/run_parallel.py --episodes 'baseline_evaluation_v3/episodes/task_*/t*-inc-con' --workers 6"
```

Partition limits/account requirements depend on the cluster. The Slurm batch
step supplies the allocation context; no `--job-id` is needed in this command.
Allow enough allocation time for the whole queue, not just one episode.

## Capacity checks executed on sonny, 2026-09-27

`--probe` uses the same dataset/config loader and full Habitat environment. It
renders RGB, initializes perception/world graphs, and solves a small PDDL problem
in each worker. A barrier keeps the first wave alive together for 20 seconds so
GPU usage reflects concurrent environments. It never constructs an API client or
executes a model call. It does not run long action trajectories or measure API
throughput.

```bash
python scripts/run_parallel.py \
  --episodes baseline_evaluation_v3/episodes/task_2/t2-inc-con \
  --runs-per-episode 6 --workers 6 --job-id 3965858 --probe
```

Six simultaneous workers passed all rendering/PDDL checks on the A40 with 12
allocated CPUs. Peak sampled GPU memory was 3,096 MiB of 46,068 MiB. Individual
peak host RSS was 1,597–1,663 MiB (about 9.7 GiB total). The six-worker evidence
is under `results/parallel_probe_six_20260927/`.

Eight simultaneous workers also passed, spanning tasks 1–7 (the six available
`inc-con` variants, `t2-acc-con`, and `t7-inc-dis`). With 16 CPUs, peak sampled GPU
memory was 3,527 MiB and each worker's peak host RSS was 1,415–1,663 MiB.
Evidence: `results/parallel_probe_max8_20260927/summary.json` and
`resources.jsonl`. An intermediate seven-worker mixed-episode check also passed
under `results/parallel_probe_eight_20260927/`; despite that folder's initial
name, its manifest correctly records seven workers.

Thus six is the default and eight is the tested CPU ceiling at two CPUs per
worker on this allocation. Paid parallel API throughput and full concurrent
episode completion have not been benchmarked. Those may require lower concurrency
than the environment/PDDL capacity test, especially if the API throttles requests.

The scheduler's five automated tests passed: read-only preview, input validation,
existing-output preservation, actual subprocess concurrency/worker isolation/
metric handling, and real subprocess timeouts. Run them without GPU/API work:

```bash
python -m unittest scripts.test_run_parallel
```
