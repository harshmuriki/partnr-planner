# Sonny server setup and ReAct smoke run

Verified on 2026-09-27 using the existing environment, without cloning it.
Repository: `/coc/flash5/vmuriki3/tamp/partnr-planner`.
Environment: `/nethome/vmuriki3/miniconda3/envs/habitat-llm`.
Use the absolute repository path on compute nodes.

## Allocation and storage

The controlling shell was on `sky2`, outside Slurm. Job `3965858` was already
running on `sonny`, with 16 CPUs and one NVIDIA A40 (driver `580.178.04`).
All GPU and simulator checks ran through `srun --jobid=3965858 --overlap`.
Use your current job ID on subsequent runs; this allocation is temporary.

Home initially had 9.1 GiB available on its reported 50 GiB filesystem;
`/coc/flash5` had about 146 GiB shared free space. No `quota`, `lfs`, or
`mmlsquota` utility was available on the compute node, so a separate personal
quota could not be established. No large filesystem scan or environment clone
was needed. Conda's dry run required only 187,378,703 bytes of downloads because
the CUDA PyTorch build was already extracted in the cache.

`conda clean --packages --dry-run --json` reported about 4.1 GB of unused cache,
including the desired CUDA PyTorch build. That cache was retained. After repair,
`conda clean --tarballs -y --json` removed only the two newly downloaded MKL and
NumPy archives (187,378,703 bytes). No environment, dataset, or result was deleted.
Home then reported about 8.7 GiB available.

## Repaired dependency stack

| Package | Verified version/build |
| --- | --- |
| Python | 3.9.20 |
| PyTorch | 2.4.1 `py3.9_cuda12.4_cudnn9.1.0_0`, pytorch channel |
| TorchVision | 0.19.1 `py39_cu124`, pytorch channel |
| TorchAudio | 2.4.1 `py39_cu124`, pytorch channel |
| pytorch-cuda | 12.4 |
| MKL | 2023.1.0 `h213fc3f_46344`, defaults channel |
| llvm-openmp | 14.0.6 `h9e868ea_0` |
| NumPy | 1.26.4 |
| Habitat-Sim / Habitat-Lab / Habitat-Baselines | 0.3.3 |
| TensorBoard | 2.17.1 |
| protobuf | 3.20.1 |
| OpenAI Python SDK | 1.109.1 |
| httpx | 0.28.1 |

The original environment mixed conda-forge CPU PyTorch/libtorch with CUDA
TorchVision/TorchAudio. `torch.version.cuda` was `None`, and importing TorchVision
failed with `operator torchvision::nms does not exist`.

The following transaction was first reviewed with `--dry-run --json`, then
executed with `-y`:

```bash
/nethome/vmuriki3/miniconda3/bin/conda install \
  -p /nethome/vmuriki3/miniconda3/envs/habitat-llm \
  -c pytorch -c nvidia -c defaults \
  'pytorch::pytorch=2.4.1=*cuda12.4*' \
  'pytorch::torchvision=0.19.1=py39_cu124' \
  'pytorch::torchaudio=2.4.1=py39_cu124' \
  'pytorch::pytorch-cuda=12.4' \
  'defaults::mkl=2023.1.0' -y
```

This replaced CPU PyTorch, removed standalone CPU libtorch and nomkl, and selected
compatible MKL/OpenMP dependencies. PyTorch packages remain Conda-managed.

The existing pip-managed SDK also needed updating: version 1.10.0 rejected
`reasoning_effort` before making an HTTP request. The updated version's method
signature accepts it. The model remains `gpt-5.2` with `high` reasoning effort,
which is supported according to the
[official GPT-5.2 documentation](https://developers.openai.com/api/docs/models/gpt-5.2).

```bash
/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python -m pip install \
  --no-cache-dir openai==1.109.1
```

`requirements.txt` now records this SDK version and TensorBoard 2.17.1 while
preserving protobuf 3.20.1. The planner package lazily imports its two
PDDL-dependent planners so ReAct no longer requires a compiled Fast Downward.
Selecting a PDDL planner still requires its normal build/dependencies.

## Target data and execution

The transferred target contained `dataset.json`, but no `dataset.json.gz`.
A gzip copy was created without modifying the JSON and verified by decompressing
it and comparing every byte. The dataset contains exactly one episode,
`T2-INC-CON`, in scene `106878960_174887073`.

API credential presence was checked through dotenv without displaying its value.
The initial two planner attempts failed locally before any API request: first on
the eager PDDL import, then on the old SDK keyword argument.

From inside a valid GPU allocation:

```bash
cd /coc/flash5/vmuriki3/tamp/partnr-planner
export PATH=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin:$PATH
run_dir="results/partnr_v3_t2_inc_con_$(date +%Y%m%d_%H%M%S)"
python -u -m habitat_llm.examples.planner_demo \
  --config-name=baselines/single_agent_zero_shot_react_summary \
  habitat.dataset.data_path=baseline_evaluation_v3/episodes/task_2/t2-inc-con/dataset.json.gz \
  +habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/ \
  evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=api \
  paths.results_dir="$run_dir" \
  evaluation.output_dir="$run_dir" \
  evaluation.save_video=False \
  evaluation.generate_prediviz=False \
  num_runs_per_episode=1 mode=cli device=cpu
```

PrediViz is disabled for this smoke run because its postprocessing converts the
episode ID to an integer and fails on `T2-INC-CON`. This does not affect planner
execution. `device=cpu` still requires the GPU for Habitat rendering.

## Validation evidence

Local logs and verification scripts are in
`results/partnr_setup_20260927_105829/` (ignored experiment artifacts).

- `stack.log`: imports of torch, torchvision, torchaudio, and SummaryWriter pass;
  CUDA 12.4 is available; actual NMS calls return the expected indices on both
  CPU and CUDA.
- `scene.log` and `scene.png`: actual target scene loaded with physics enabled
  and rendered on the NVIDIA A40 at 128x128 RGBA; RGB standard deviation was
  18.31. The missing `pbr` warning was nonfatal. This check used the scene and
  start position; the planner subsequently initialized the complete episode.
- `planner_attempt1.log`, `planner_attempt2.log`: the two pre-API setup failures.
- `planner_imports.log`: Hydra resolves `ZeroShotReactPlanner` without importing
  Fast Downward; requesting `VlmTampPddlPlanner` still reports its missing build.
- `planner_attempt3.log`: API-backed single-episode run, with optional PrediViz
  disabled. Process exited normally with status 0; final outcome is below.
- `conda-explicit.txt` and `pip-freeze.txt`: environment snapshots.

`pip check` also reports existing metadata conflicts: Transformers-CFG requests
protobuf >=4.25.2, while the repository requires 3.20.1; Habitat-Sim requests
Pillow 10.4.0 while pip reports 11.1.0. These are separate from the tested CUDA
repair; the render check passed. The protobuf pin was retained as requested.

## Completed run outcome

Output directory: `results/partnr_v3_t2_inc_con_20260927_1104/`.
The single API-backed episode finished normally after 163.36 seconds wall time,
with 10 GPT-5.2 calls, 25,535 input tokens, 1,954 output tokens, and a tracker-reported
API cost of $0.05349505. Its combined planning/simulation budget usage was
51.68 seconds of the configured 600 seconds; this is not wall-clock runtime.

The planner moved both boxes and the scissors to `table_12` in its world graph
and selected `Done`. However, the final step in
`dataset/planner-log/planner-log-move-both-packages-to-the-living-room-center-table.json`
records `task_percent_complete: 0.0` and `task_state_success: 0.0`.
Its explanation lists all six on-top/next-to conditions as unsatisfied.
The HTML trace's 100% action success rate must not be interpreted as benchmark
task success. `end_result_log.csv` contains no metrics in this CLI run; use the
planner JSON. Investigating the discrepancy between executed placements and
episode predicates is a separate evaluation issue, not a verified setup failure.
No second paid run was launched to change this outcome.

The environment, CUDA operators, rendering, API access, and ReAct execution were
therefore verified end to end. Benchmark success was not achieved. PrediViz's
string-ID issue and the dependency metadata conflicts above remain limitations.

## VLM-TAMP follow-up test

For the separately requested VLM-TAMP test, Fast Downward was built successfully
on sonny in the same allocation, with the environment's CMake 3.14 and system
GCC/G++ 13.3.0:

```bash
cd /coc/flash5/vmuriki3/tamp/partnr-planner/third_party/pddlstream
export PATH=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin:$PATH
python downward/build.py -j16
```

A blocks-world problem passed through the repository's
`solve_pddlstream_problem(..., planner="ff-astar1")` wrapper and returned a
four-action plan with cost 4. Hydra then resolved `VlmTampPddlPlanner` successfully.
The configured `gpt-5.6-luna` model returned HTTP 200 from the model-access check.
Build, solver, model-access, and episode logs are under
`results/vlm_tamp_setup_20260927/`.

The episode command uses the same dataset and flags as the ReAct command above,
but selects `--config-name=baselines/single_agent_vlm_tamp_pddl` and omits the
ReAct-specific `evaluation.agents.agent_0.planner.plan_config.llm.inference_mode`
override. VLM-TAMP configures its API through `plan_config.vlm` instead.
The fresh output directory is
`results/vlm_tamp_v3_t2_inc_con_20260927_1115/`.

The first VLM-TAMP attempt exposed a completion-budget problem: both API calls
returned HTTP 200 with `finish_reasons: ["length"]`, 1,200 completion tokens,
1,200 reasoning tokens, and empty text. It aborted without executing any actions;
the tracker reported $0.0038 for the attempt. This was not a Fast Downward failure.
The follow-up smoke command adds
`evaluation.agents.agent_0.planner.plan_config.vlm.max_completion_tokens=8192`
and writes to `results/vlm_tamp_v3_t2_inc_con_20260927_1117/`, keeping the same
model and high reasoning effort. Its log is `planner_retry.log` in the setup folder.

The retry completed with process exit status 0, taking 635.28 seconds overall
(621.01 seconds episode runtime). It exercised physical exploration, six PDDL
solves, navigation, opening furniture, picking, and placement. There were 12 VLM
calls, 34,071 input tokens and 38,352 output tokens; tracker-reported API cost was
$0.0523506. Two English-subgoal responses still exhausted all 8,192 tokens without
text, but predicate responses and subsequent English responses returned content.
The larger limit enabled this run; it is not a guarantee against truncation and
was applied as a command-line override rather than changing the baseline default.

Final `episode_metrics.json` reports `task_percent_complete: 0.5` and
`task_state_success: 0.0`. Both packages were placed on `table_16` and their
conditions were satisfied; the scissors-on-table and both scissors-next-to-box
conditions remained unsatisfied. The planner exhausted its generated subgoals
without completing the full instruction. Its HTML action-success rate of 100%
does not mean the task succeeded.

The actual VLM-TAMP planning viewer is
`results/vlm_tamp_v3_t2_inc_con_20260927_1117/vlm_tamp_pddl/unknown/vlm_tamp_v3_t2_inc_con_20260927_1117_pddl.html`.
The sibling `episode_metrics.json` holds the task metrics. The `unknown` folder
name is the current planner's episode-log naming behavior for this run.

Remaining observed limitations: VLM completion truncation, incomplete task-level
planning, and timing telemetry. VLM-TAMP reported no LLM planning time, so its
combined budget counted only 6.24 seconds of actions despite over ten minutes
of wall time. The 600-second combined limit therefore did not bound VLM latency.
The simulator/API/PDDL execution path is verified, but these results do not
establish that every planner feature or task-success condition works correctly.

## Parallel launcher

See [Parallel PARTNR runs](PARALLEL_RUNS.md) for the implemented queue, isolated
workers, wall-clock timeouts, Slurm commands, and concurrent capacity-test results.
Use six workers by default; `auto` is bounded by allocated CPUs and is not a
promise about model rate limits or maximum GPU throughput.
