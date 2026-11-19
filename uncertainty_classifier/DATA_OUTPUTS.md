# Data Outputs

Every output available for the baseline runs, what is in it, and how many runs have it. Paths are in [DATA_PATHS.md](DATA_PATHS.md).

```
PARTNR   = /coc/flash5/vmuriki3/tamp/partnr-planner
RESULTS  = $PARTNR/results/all_baselines_5reps_20261005_001408
GT       = $PARTNR/baseline_evaluation_v3
RUN      = $RESULTS/<baseline>/<variant>/repeat_<n>/attempt_<k>/runs/<run_id>/
```

## The Batch

| | |
|---|---|
| Runs | 1,350 = 3 baselines × 90 variants × 5 repeats |
| Baselines | `react` (images), `vlm-tamp` (images), `tru-pomdp` (text); 450 runs each |
| Model | `gpt-5.6-luna`, reasoning effort `high` |
| Time limit | 600 s (Tasks 1–5), 1,200 s (Tasks 6–7), LLM + simulated action time combined |
| Tasks | T1–T7 (runs per task: T1, T2, T5, T6 = 225; T3, T4 = 180; T7 = 90) |
| Memory | `ACC` accurate, `INC` incomplete, `OUT` outdated (450 runs each) |
| Uncertainty | `BASE` 315, `DIS` distractor 315, `ABS` absent 270, `CON` container 270, `SUB` substitute 180 |
| Success (`task_state_success = 1`) | react 204/450, tru-pomdp 147/450, vlm-tamp 131/450 |
| Attempts | 2,118 attempt folders on disk; `baseline_results.json` points to the latest one per run (762 runs were retried after quota errors) |

A variant id is `T<task>-<memory>-<uncertainty>`, e.g. `T1-ACC-ABS`.

## Batch-Level Files (`RESULTS/`)

| File | Contents |
|---|---|
| `baseline_results.json` | **The run index.** One row per run (fields below) |
| `batch.json` | Batch settings: source and snapshot repo, model, baselines, repeats, GPU, SHA-256 of every episode input |
| `summary.json` | Scheduler record per job: planner, episode, partition, repetition, timeout, status, full command line |
| `README.md` | Batch description, retry history, video policy, dashboard notes |
| `snapshot/` | Frozen code the batch ran: `habitat_llm/`, `scripts/`, `baseline_evaluation_v3/`, `third_party/`, `data/` |
| `logs/` | 710 Slurm job logs (`overcap-<job>_<n>.log`) |
| `retry_plan.json`, `retry_submissions.json`, `submissions*.json`, `quota_retry_audit.json` | Retry bookkeeping |
| `video_policy_*.json`, `concurrency_update.json`, `preflight*.log` | Run-policy changes and startup checks |
| `baseline_dashboard.html`, `baseline_dashboard_data.json`, `tc_spl_data.json` | Results dashboard and its aggregates (incl. TC-SPL efficiency metric) |
| `paper_figures_20261006/`, `*.png` | Figures |
| `uncertainty_classifier/` | Copy of the earlier rule viewer (`rule_viewer.html`, `rules_data.json`, `source/`, `revisions/`) |

### `baseline_results.json` Row

| Field | Example / values |
|---|---|
| `id`, `run_id`, `task_id`, `task`, `memory`, `uncertainty`, `iteration`, `attempt` | `react/T1-ACC-ABS/repeat_1`, `t1-acc-abs_ea1cf791_r001`, `T1-ACC-ABS`, `1`, `ACC`, `ABS`, `1`, `attempt_000` |
| `condition` | `react_images`, ... |
| `run_folder`, `folder_url`, `source_run_json`, `html_links` | Paths to the run folder, its `run.json`, trace HTML and video |
| `task_state_success`, `completion` | Evaluator success (0/1) and fraction of requirements met |
| `subtasks_succeeded`, `num_subtasks` | Requirements met / total |
| `subtask_results.criteria[]` | Per requirement: `id`, `label` (`lamp_0: powered off`), `status` (`met`/...), `proposition` (`function_name`, `args` with sim handles) |
| `subtask_results.method`, `.source`, `.score_basis` | How it was scored (`terminal_simulator_snapshot`), the evidence file and line it came from |
| `task_explanation` | Evaluator text, e.g. `Episode failed. Missing steps: ...` |
| `execution_status` | `completed` 1,043, `timeout` 307 |
| `simulator_outcome` | `partial` 723, `complete` 482, `no_progress` 145 |
| `sankey_reason` | `Stopped with unmet goals` 593, `100% successful` 482, `Time limit reached` 261, `Action limit reached` 14 |
| `termination` | `null`, or `{"reason": "max_actions", "message": ...}` (19 runs) |
| `counts` | Actions by skill, e.g. `{"Navigate": 1, "PowerOff": 1}` |
| `num_actions`, `sim_steps`, `runtime_s` | Action count, simulator steps, wall time |
| `llm_requests`, `total_tokens`, `llm_planning_time_s`, `llm_usd`, `llm_usd_upper_bound` | LLM usage and cost |
| `metrics_are_partial` | True when the run timed out and metrics come from the last checkpoint |

## Per-Run Files (`RUN/`)

Counts are runs that have the file, out of 450 per baseline (latest attempt).

### Inputs (all runs)

| File | Contents |
|---|---|
| `input/spec.md` | Variant spec in prose: base task, scene, rooms and furniture used, affected objects, entity registry (targets and distractors), uncertainty tested, initial and expected final world state, initial robot memory |
| `input/dataset.json.gz` | Habitat episode: `instruction`, `scene_id`, start pose, `rigid_objs`, `object_states`, `name_to_receptacle`, `evaluation_propositions` (e.g. `is_powered_off(handles)`), `evaluation_constraints`, `info.variant_spec` (entity handles, memory, unscored success text) |
| `input/initial_robot_memory.json` | Which entities are in the robot's starting memory (`in_initial_robot_memory`), memory lines (e.g. `jug_0: known absent ...`), `initial_memory_source` |
| `input/review.json` | Per object: asset, sim handle, initial and expected relation/furniture/room/states, position; checks run; `unscored_success_text`; spec hash |
| `input/scene_info.json` | Scene layout: furniture per room, all objects (108 in T1), `receptacle_to_handle`, `room_to_id`, `recep_to_description`, `object_affordances`, spec→runtime furniture aliases |

### Run Record and Logs (all runs unless noted)

| File | Contents |
|---|---|
| `run.json` | Worker record: status, exit code, elapsed seconds, command, and `metrics` (`task_percent_complete`, `task_state_success`, `task_explanation`, `cost_metrics`, `model_costs`) |
| `log.txt` | Console log, including the printed starting **Robot Scene Graph** (rooms → furniture → receptacles → known objects) |
| `hydra/habitat_llm.log` | Debug log: skill internals (navigation candidates, world-graph updates) |
| `hydra/.hydra/{config,hydra,overrides}.yaml` | Full resolved config and command-line overrides |
| `end_result_log.csv` | Effectively empty (4 bytes); react 376, vlm-tamp 382, tru-pomdp 287 |

### Actions (all runs)

| File | Contents |
|---|---|
| `dataset/actions/actions-*.json` | Final list of skill calls: `action_id`, `action` (`Navigate`, `Pick`, `Place`, `Open`, `PowerOff`, `Explore`, ...), `action_input`, `status`, `response` (`Successful execution!` or the error text), `started_sim_step`/`finished_sim_step`, wall times; plus `sim_step_count` |
| `dataset/actions/actions-*.jsonl` | Same calls as an event stream (`in_progress` then finished), with wall time |

### Planner Reasoning

| File | Who has it | Contents |
|---|---|---|
| `dataset/traces/0/trace-*.txt` | all | Readable trace. ReAct: `Thought:`, action, `Result:`, and the objects it sees after each step. VLM-TAMP: subgoals, PDDL plans, `Planner decision:` |
| `dataset/traces/0/trace-*.html` | react 376, vlm-tamp 382, tru-pomdp 287 | HTML version of the trace |
| `dataset/traces/0/images/step_*.png` | react only | Image sent to the model at each step |
| `dataset/prompts/0/prompt-*.txt` | react, tru-pomdp | Full prompt text (system prompt, examples, task) |
| `dataset/planner-log/planner-log-*.json` | all | One entry per planner step (`steps[]`, 207 in the example). Common: `high_level_actions`, `responses`, `is_done`, `replanned`, `replan_required`, `cost_metrics`, `stats` (percent complete so far), `sim_step_count`, `curr_graph` (the world-graph text the planner saw). ReAct adds `thought`, `agent_states`, `agent_positions`, `agent_collisions`, `action_image`. TRU-POMDP adds `tru_pomdp` (below) |
| `dataset/planner-log/planner-log-*.jsonl` | all | Same as an event stream |
| `dataset/planner-log/checkpoint-*.json` | react 450, vlm-tamp 407, tru-pomdp 370 | Last saved step count and stats |
| `dataset/detailed_traces/*.pkl` | all | Pickled detailed trace |

**TRU-POMDP `tru_pomdp` block** (per step): `belief` (e.g. `w=1.00 goal=[lamp_0 [is_powered_off], jug_0 on table_0 [is_filled]]`), `termination_reason`, `num_particles`, `num_decisions`, `memory_revisions`, `rejected_hypotheses`, `belief_update`, `failed_attempts`, `inspected_areas`, search stats (`trials`, `nodes`, `root_lower`, `root_upper`, `search_time_s`), LLM and overhead times.
Last-step `termination_reason` across the 450 runs: `belief_complete` 176, none 163, `empty_belief` 85, `budget_exhaustion` 16, `no_useful_action` 10.

### VLM-TAMP Extras (`vlm-tamp` only)

| File | Count | Contents |
|---|---|---|
| `vlm_tamp_pddl/vlm_tamp_pddl_log.jsonl` | 450 | Event log, **written as Python dict reprs, not JSON**. Events (over the 90 repeat-1 runs): `subgoal_status` 2,193, `pddl_plan` 921, `subgoal_execution` 861, `memory_snapshot` 419, `vlm_english_subgoals` 417, `branch_start` 402, `reprompt_branches_added` 402, `vlm_predicate_subgoals` 402, `reprompt_started` 330, `special_subgoal_action` 182, `branch_exhausted` 152, `failure_observation` 151, `planner_decision` 75, `search_limit` 7 |
| `vlm_tamp_pddl/vlm_prompts.txt`, `.html` | 450 | Each VLM prompt per reprompt round (English subgoals, then predicate subgoals) |
| `vlm_api_prompts.txt` | 450 | Raw API prompts with timestamps |
| `vlm_tamp_pddl/plan_tree.txt` | 407 | Branches and subgoals with status (`EXHAUSTED`, `CURRENT`) |
| `vlm_tamp_pddl/media/planning_tree_layout.json` | | Plan tree as nodes with `status`, `failure_type`, `failure_msg` |
| `vlm_tamp_pddl/pddl_dumps/domain.pddl`, `stream.pddl` | 386 | PDDL domain: `navigate`, `pick`, `place_on`, `place_in`, `open`, `close`, `power_on/off`, `power_on/off_object`, `fill_held`, `clean_object`, `clean_furniture`, `pour` (no next-to) |
| `vlm_tamp_pddl/pddl_dumps/problem_b<branch>_s<subgoal>.txt` | | PDDL problem: init facts and goal |
| `vlm_tamp_pddl/episode_metrics.json` | 382 | Runtime, tokens, cost, sim steps per skill, combined time used vs limit and its breakdown |
| `vlm_tamp_pddl/observation_history.html`, `<run_id>_pddl.html` | | HTML views of observations and the PDDL log |
| `vlm_tamp_pddl/{explore_images,subgoal_images,vlm_images}/*.png` | | Exploration frames, subgoal start/solved frames, images sent to the VLM |

### Evaluation and World State (all runs)

| File | Contents |
|---|---|
| `dataset/evaluation-checkpoint.json` | Final evaluator state: `proposition_satisfied_at` (sim step each requirement was first met, -1 if never), `last_state`, `task_constraint_validation`, `task_percent_complete`, `task_state_success`, `task_explanation`, and `reference_world_graph` (full true scene graph, 128 nodes in T1) |
| `dataset/evaluation-checkpoints.jsonl` | The same state at each checkpoint over time |
| `dataset/evidence-*.jsonl` | Evaluator event stream: `start`, `boundary` (per planner step, with `planner_observed` and evaluator state), `terminal` |
| `dataset/object_states/*.json` | Every object and furniture piece, `initial` and `final`: `kind`, `sim_handle`, `location` (room, furniture, held_by, xyz, relations), `states` (`is_clean`, `is_open`, `is_powered_on`, `is_filled`) |
| `dataset/llm_costs/*.json` | Per LLM call: model, status, token usage (incl. reasoning tokens), cost, request id, HTTP status; plus summary |
| `dataset/videos/video-*.mp4` | Run video (repeat 1 only by policy): react 203, vlm-tamp 207, tru-pomdp 210 |

## Ground Truth (`GT/`)

| File | Contents |
|---|---|
| `ground_truth.sqlite3` | Human demonstrations. Open read-only: `sqlite3.connect(f"file:{path}?immutable=1", uri=True)` |
| ↳ `runs` (90) | One per variant: status, action count, sim steps, spec/dataset hash, metadata, snapshot |
| ↳ `actions` (1,734) | `run_id`, `sequence`, `skill`, `target`, `result` |
| ↳ `object_rooms` (24), `object_furniture` (23), `room_choices` (2) | Likely rooms and furniture to search per object |
| ↳ `action_packs` (22), `notes` (4) | Reusable action sequences per task; notes |
| `ground_truth_timing/` | `summary.csv` (per task: min/median/max seconds), `current_recordings.csv`, `attempts.csv`, `http_intervals.csv`, `replay_measurements.json` |
| `ground_truth/T<n>/<variant>/` | `video.mp4`, `initial.jpg`, `clips/` readable; `actions.csv`, `run.json`, choices files owner-only |
| `specs/T<n>/` | Source spec per variant and `_furniture_catalog.txt` (room → furniture) |
| `episodes/task_<n>/t<n>-<memory>-<variant>/` | Same inputs as `RUN/input/` |
| `variant_object_assets.csv` | Per variant and object: label (target/distractor), in initial memory, outdated location, asset id, start and end states |
| `human_verification.json` | Per variant: `verified`, `needs_review`, timestamps |
| `ground_truth_t6_elapsed_audit.csv`, `FAUCETS.md`, `scenario_viewer.html` | T6 timing audit, faucet notes, scenario viewer |

## Notes

- `end_result_log.csv` has no content; use `run.json` or `baseline_results.json` for outcomes.
- `vlm_tamp_pddl_log.jsonl` is not valid JSON (Python reprs with object calls); parse with care.
- Timed-out runs are missing the HTML trace, `checkpoint-*.json`, `episode_metrics.json` and sometimes the PDDL dumps; their metrics come from the last saved checkpoint (`metrics_are_partial`).
- Earlier attempts (quota failures) are still on disk next to the latest one.
