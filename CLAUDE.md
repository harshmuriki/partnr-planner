# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a research fork of Meta's PARTNR benchmark (embodied LLM planning in the Habitat simulator), extended with uncertainty-aware TAMP baselines (VLM-TAMP + PDDLStream, Tru-POMDP, a custom hierarchical-prompting approach) and a hand-specified evaluation set in `baseline_evaluation_v3/`.

## Environment

- Conda env `habitat-llm` (Python 3.9, habitat-sim 0.3.3, editable installs of `third_party/habitat-lab` and `third_party/transformers-CFG`). On the cluster: `export PATH=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin:$PATH`. See `INSTALLATION.md`, `docs/SERVER_SETUP.md`, `docs/SONNY_SETUP.md`.
- Run everything from the repo root; `data/` holds symlinked HSSD scenes, PARTNR episodes, and skill checkpoints.
- Simulator runs need a visible GPU; login nodes have none. Use a Slurm allocation (`srun`/`--job-id`).
- OpenAI-backed planners need `OPENAI_API_KEY` (`.env` exists at the root). Runs make paid API calls; `--dry-run`/`--probe` flags on launchers do not.

## Commands

```bash
# Unit tests (need data/hssd-partnr-ci and data/test_rag symlinks, see INSTALLATION.md)
python -m pytest habitat_llm/tests
python -m pytest habitat_llm/tests/test_foo.py::test_bar -v

# Script-level tests in scripts/test_*.py are unittest-based and mostly run without Habitat/GPU/API
python -m pytest scripts/test_run_parallel.py

# Lint/format: pre-commit runs isort (black profile), black, ruff --fix, autoflake (third_party/ excluded)
pre-commit run --all-files

# Run a planner (Hydra; pick a config from habitat_llm/conf/baselines/)
python -m habitat_llm.examples.planner_demo --config-name baselines/single_agent_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat

# Interactive oracle-skill sandbox for one episode
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir="." \
    habitat.dataset.data_path=<dataset> +skill_runner_episode_id="334"

# Check episodes initialize / step-0 success
python -m habitat_llm.examples.verify_episodes --config-name examples/planner_multi_agent_demo_config.yaml ...  # see README

# Results summary for a run
python scripts/read_results.py <output_dir>/<dataset_name>
```

Batch evaluation:
- `scripts/run_parallel.py`: queues single-episode ReAct / VLM-TAMP runs concurrently inside one GPU allocation (default 6 workers). Example: `--episodes 'baseline_evaluation_v3/episodes/task_*/t*-inc-con' --workers 6 --job-id <slurm_job> --dry-run`. Details in `docs/PARALLEL_RUNS.md`.
- `scripts/run_tasks_wrapper.py` / `scripts/run_baseline_evaluation.sh`: sequential, YAML-config driven (`docs/WRAPPER_SCRIPT_USAGE.md`). A rerun overwrites matching run dirs, and its `success` field means the process exited cleanly, not that the task succeeded.
- PDDLStream writes to a shared relative `temp/` dir. Parallel VLM-TAMP workers need separate working copies.

## Architecture

The core loop, documented in the README: `EnvironmentInterface` (`habitat_llm/agent/env/`) reads per-agent observations, then `perception/perception_sim.py` turns them into detections. Those detections update the `WorldGraph`/`DynamicWorldGraph` (`world_model/`: a hierarchical graph of rooms → furniture → receptacles → objects with object states). A `Planner` (`planner/`) reads the world graph plus the instruction and selects a `Tool` (`tools/`). A tool is either a perception query or a motor skill (oracle or neural-net nav/pick/place/open/close/...). `evaluation/` runners (centralized/decentralized) drive episodes and score them against the dataset's evaluation propositions/constraints.

Everything is composed with Hydra under `habitat_llm/conf/`. `baselines/*.yaml` select a planner (`conf/planner/`), an LLM backend (`conf/llm/`, e.g. `openai_chat`, `hf`), an instruct/prompt config (`conf/instruct/`), agent tool sets (`conf/agent/`), and habitat task configs. Override with `group@path=value` syntax, as in the commands above.

Fork-specific planners (single-agent):
- `planner/vlm_tamp_pddl_planner.py` + `habitat_llm/pddlstream/` (problem/streams/solve, domains in `pddl_domains/`) + `habitat_llm/vlm_tamp/` (prompts, VLM API, HTML trace/tree renderers). Uses `third_party/pddlstream`. Config: `baselines/single_agent_vlm_tamp_pddl.yaml`.
- `planner/tru_pomdp/`: Tru-POMDP reimplementation. `toh.py` (LLM Tree of Hypotheses, the only LLM caller), `belief.py` (particle update), `search.py` (Python DESPOT), `scene.py` (symbolic model), `planner.py` (PARTNR adapter). Read its `README.md` for the contract. Config: `baselines/single_agent_tru_pomdp.yaml`.
- `planner/custom_approach_planner.py` + `habitat_llm/custom_approach/`: hierarchical task→object-scoped subgoals→explore/act prompting (`approach.md`). Config: `baselines/single_agent_custom_approach.yaml`.
- `rail_planner/` (`RailPlanner`): our method. Subclasses `VlmTampPddlPlanner`, keeping its VLM turns, branches, replanning, explore and logging, but expands each symbolic subgoal into oracle skills with hand-written expanders (`subgoal_expanders.py`) instead of PDDLStream, so it needs no `temp/` dir. Adds a `waited_in(obj, appliance)` subgoal (Wait skill). Configs: `baselines/single_agent_rail_planner.yaml`, `baselines/single_agent_rail_planner_placement_oracle.yaml`; options in `conf/planner/rail_planner.yaml`:
  - `replan_trigger` (`replan_trigger.py`): `failure` (default) | `every_step` | `any_change` | `oracle`. Proactive replans count toward `max_vlm_cycles`.
  - `placement_oracle` (`placement_oracle.py`): `none` | `hint` | `place` | `full`. Reads the episode's placement goals.
  - `end_of_plan_check`: `none` | `oracle` | `oracle_feedback` (replan while the sim goal check is unmet, up to `max_end_of_plan_replans`).
  - All `oracle*` options and any `placement_oracle` other than `none` use privileged ground truth: upper bounds, not deployable.
- `planner/zero_shot_react_planner.py` is the ReAct baseline used for comparisons (`planner/react_placement_oracle.py` adds a privileged placement-hint ablation via `plan_config.placement_oracle=hint`).

`dataset_generation/benchmark_generation/` generates episodes (upstream LLM pipeline, plus the fork's `generate_verified_specs`, `audit_variant_specs`, and `verify_dataset`).

## baseline_evaluation_v3

- `episodes/task_N/tN-<memory>-<variant>/`. `<memory>` ∈ `acc` (accurate robot memory), `inc` (incomplete), `out` (outdated). `<variant>` ∈ `base`, `abs`, `con`, `dis`, `sub` (uncertainty types; see each `spec.md`). Each folder holds one-episode `dataset.json(.gz)`, `initial_robot_memory.json`, `scene_info.json`, `spec.md`, and `review.json`. 90 variants in total across T1–T7.
- Source of truth is `specs/` + `generation/source_tasks.xlsx`. The generation flow is documented in `generation/README.md`, with current state in `generation/CURRENT_STATUS.md`. The flow does not edit source specs, and it does not move failed placements or substitute other assets.
- `ground_truth/`, `ground_truth.sqlite3`, `ground_truth_timing/`: human/oracle ground-truth traces and timing (`scripts/scenario_ground_truth.py`, `ground_truth_worker.py`, `docs/ground_truth_metrics.md`).
- `gui/` + `scenario_viewer.html`: static viewer served by `scripts/serve_scenario_viewer.py`.
- `baseline_evaluation_v2/` is the previous version; prefer v3.
