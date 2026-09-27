# Evaluation wrapper

Run commands from the repository root with the Habitat environment activated.
For moving this project to a server, see [Server setup and parallel runs](SERVER_SETUP.md).

## Configuration

The local configs are under `baseline_evaluation_v1/configs/`, not `configs/`.
That directory is ignored by Git: copy it to the server with your episodes.
The minimal example uses the ReAct planner; `baseline_runs.yaml` uses VLM-TAMP PDDL.

A complete config using one existing v1 task looks like this:

```yaml
output_base_dir: results/baseline_test
num_runs_per_task: 1
planner_config: baselines/single_agent_zero_shot_react_summary

global_hydra_overrides:
  - "+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/"
  - "evaluation.agents.agent_0.planner.plan_config.objects_response_include_states=True"
  - "device=cpu"
  - "habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor.normalize_depth=False"
  - "llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat"
  - "evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=api"
  - "evaluation.save_video=True"
  - "num_runs_per_episode=1"
  - "mode=cli"

tasks_root: baseline_evaluation_v1/episodes/
tasks:
  - Task_3_Loc
```

For a string task name, the wrapper expects `<tasks_root>/<task>/<task>.json.gz`
and discovers the first `*.yaml` in that folder as the runtime config. Keep only
one runtime YAML there, or specify it explicitly using the dictionary format:

```yaml
tasks:
  - task_id: example
    task_folder: path/to/episode_folder
    episode_file: path/to/episode_folder/episode.json.gz
    runtime_config: path/to/episode_folder/runtime.yaml
    hydra_overrides:
      - "evaluation.save_video=False"
```

`runtime_config` is optional. The wrapper passes the task folder to `planner_demo`,
which selects the dataset inside it; keep one intended dataset per task folder.
Do not use the old `runtime_objects` config key. Per-task Hydra overrides are
appended after global overrides. `num_runs_per_task` controls wrapper repetitions;
keep `num_runs_per_episode=1` to avoid nested repetitions.

## Run and preview

```bash
python scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs_minimal.yaml \
  --output-dir results/smoke_test --dry-run

python scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs_minimal.yaml \
  --output-dir results/smoke_test
```

`--dry-run` prints commands without launching Habitat or creating, deleting, or
rewriting result files. It does not verify simulator dependencies or scene assets.
`--output-dir` overrides the config's `output_base_dir`.

`--task-ids` accepts task IDs or task folder names already listed in the config:

```bash
python scripts/run_tasks_wrapper.py \
  --config baseline_evaluation_v1/configs/baseline_runs_minimal.yaml \
  --task-ids Task_3_Loc --output-dir results/task_3_loc
```

The shell launcher accepts the same flags and changes to the repository root:

```bash
bash scripts/run_baseline_evaluation.sh --help
bash scripts/run_baseline_evaluation.sh --dry-run
```

Its default is `baseline_evaluation_v1/configs/baseline_runs.yaml`.
A real rerun replaces each selected task's matching run directories. Use a fresh
output directory to retain previous results; rerunning is not checkpoint resume.

## Parallel runs

Each wrapper executes sequentially. Start separate wrapper processes with distinct
`--output-dir` values and split the configured tasks with `--task-ids`. Separate
task IDs alone are insufficient: the experiment config and summary filenames are
shared within an output directory. Task folder basenames must also be unique
within a worker because they determine run directory names.

For VLM-TAMP/PDDLStream, use a separate working copy per worker as described in
[the server guide](SERVER_SETUP.md#parallel-workers). PDDLStream uses a shared
relative `temp/` directory; separate result directories alone do not isolate it.
Other planners may also write working-directory files, so separate working copies
are the default server workflow.

## Outputs and monitoring

For `--output-dir results/smoke_test` and task `Task_3_Loc`:

```text
results/smoke_test/
├── experiment_config.json
├── experiment_summary.json
└── Task_3_Loc/
    ├── task_summary.json
    └── Task_3_Loc_1/
        ├── run_metadata.json
        ├── stdout.log
        ├── stderr.log
        └── ... planner output, traces, stats, and optional videos
```

When found, HTML traces are copied to the sibling directory
`results/outputs_smoke_test/`. The planner's nested output layout depends on its
configuration. Task summaries are saved after each task, and the experiment
summary is saved when the wrapper finishes.

The wrapper has a 30-minute timeout per planner subprocess. Its `success` field
reports process completion, not semantic task success; consult planner stats and
traces for task outcomes. Inspect `stderr.log` and `run_metadata.json` for failures.
Use `evaluation.save_video=False` to reduce recording overhead when appropriate.
