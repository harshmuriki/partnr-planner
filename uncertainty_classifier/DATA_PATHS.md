# Data paths

```
PARTNR   = /coc/flash5/vmuriki3/tamp/partnr-planner
RESULTS  = $PARTNR/results/all_baselines_5reps_20261005_001408
GT       = $PARTNR/baseline_evaluation_v3
```

## Baseline Runs (`RESULTS`)

| Path | What it is |
|---|---|
| `RESULTS/baseline_results.json` | Index: one row per run with id, memory, uncertainty, success, `execution_status`, `termination`, `subtask_results.criteria` (evaluator verdict per requirement) and `run_folder` |
| `RESULTS/batch.json` | Batch settings: model, baselines, repeats, dataset hashes |
| `RESULTS/snapshot/` | The exact code the batch ran with (`habitat_llm/`, `scripts/`, ...) |
| `RESULTS/<baseline>/<variant>/repeat_<n>/attempt_000/runs/<run_id>/` | One run folder (the `run_folder` field) |

Inside a run folder:

| Path | What it is |
|---|---|
| `input/spec.md` | Variant spec: entity registry (distractors), absent objects |
| `input/dataset.json.gz` | Episode: instruction, scene id, evaluation propositions and constraints |
| `input/initial_robot_memory.json` | Robot's starting memory |
| `log.txt` | Run log, including the printed starting "Robot Scene Graph" |
| `dataset/actions/*.jsonl` | Every skill call with its result string and sim step |
| `dataset/traces/0/trace-*.txt`, `.html` | Planner trace (VLM-TAMP: subgoals, PDDL plans, "Planner decision:") |
| `dataset/planner-log/*.json`, `*.jsonl` | Per-decision planner log: world graph text, TRU-POMDP belief and `termination_reason` |
| `dataset/object_states/*.json` | Each object's start and end location and state, with simulator handles |
| `dataset/evaluation-checkpoint.json` | Final evaluator state: when each requirement was first met, reference world graph |
| `dataset/evaluation-checkpoints.jsonl` | Evaluator state over time (which requirements held at each checkpoint) |
| `dataset/videos/*.mp4` | Run video |

## Ground Truth (`GT`)

| Path | Readable | What it is |
|---|---|---|
| `GT/ground_truth.sqlite3` | yes | Human ground-truth recordings: `runs` (90, one per variant, with snapshots), `actions` (1,734), `object_rooms` and `object_furniture` (likely search locations per object), `room_choices`, `action_packs`, `notes`. Open read-only: `sqlite3.connect(f"file:{path}?immutable=1", uri=True)` |
| `GT/ground_truth/T<n>/<variant>/` | partly | Per-variant recording: `video.mp4`, `initial.jpg` and `clips/` are readable; `actions.csv`, `run.json`, `room_choices.json`, `furniture_choices.json` are owner-only |
| `GT/ground_truth/index.*`, `README.md` | no | Owner-only; `assumptions.md` is readable |
| `GT/ground_truth_timing/` | yes | Ground-truth timing: `current_recordings.csv`, `attempts.csv`, `summary.csv`, `README.md` |
| `GT/specs/T<n>/` | yes | Source specs per variant (`T1-ACC-ABS.md`, ...) and `_furniture_catalog.txt` (room → furniture) |
| `GT/episodes/task_<n>/t<n>-<memory>-<variant>/` | yes | Episode inputs: `dataset.json(.gz)`, `initial_robot_memory.json`, `scene_info.json`, `spec.md`, `review.json` |
| `GT/human_verification.json` | yes | Human verification of the variants |
| `GT/variant_object_assets.csv` | yes | Object assets per variant |