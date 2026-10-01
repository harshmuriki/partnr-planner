# Ground-truth timing: Claude handoff

Read this before changing GT timing, the Metrics UI, or the running T6 rerun.
This is repository-local documentation; no chat-history access is needed.

## User authorization and current work

The user authorized rerunning **all 15 saved T6 variants with their exact existing
skill/target sequences**, using only one sandbox, carefully measuring runtime and
updating per-variant times, averages per task, and an overall average in the UI.
The sequences are already approved. Do not change steps, add retries, or change
simulation/rendering settings during this measurement batch.

The batch started on September 30, 2026 in America/New_York (October 1 in UTC).
T6-OUT-CON runs first because its historical elapsed time was 34m 58s. The other
14 variants follow sequentially. **Read the live status files for progress;
do not assume this document's creation time is the latest run status.**

The one-slot server and runner are already launched. Do not launch another runner,
reset a sandbox, restart the server, or close its worker while it is active.

## Important correction to the previous timing discussion

The 34m 58s figure was recovered from an automatic recorder log, from 00:10:52
("loaded 39 steps") to 00:45:50 (success). Four recordings ran concurrently.
It was not calculated from the HTML, video length, or steps divided by 120.
There is no evidence of human pauses in that automatic loop. Describe it as
elapsed recording time including simulator computation, rendering, encoding,
communication, and polling. Do not describe unexplained time as human "lag".

The earlier 12–52 second T6 figures were **accounted simulation time**, using
`sim_steps / 120`. They are a different clock, not evidence that GT completed in
12–52 seconds of wall time. A single simulator step can take substantial real
CPU/GPU time to execute. Do not compare GT simulated seconds against baseline
elapsed runtime as if they were the same metric.

## Files and entry points

Paths below are relative to the repository root:

| Path | Purpose |
| --- | --- |
| `scripts/ground_truth_worker.py` | Worker-side monotonic per-action instrumentation |
| `scripts/rerun_t6_timed.py` | Sequential HTTP runner, verification, timing publication |
| `scripts/scenario_ground_truth.py` | Existing session manager, worker transport, save/DB lifecycle |
| `baseline_evaluation_v3/ground_truth_timing/current_recordings.csv` | Current per-variant timing evidence consumed by the UI; joined on run ID |
| `baseline_evaluation_v3/ground_truth_timing/single_sandbox_20260930/` | New batch evidence, progress, logs, backups |
| `baseline_evaluation_v3/ground_truth/T6/<VARIANT>/run.json` | Saved actions, including each action's `result.timing` |
| `baseline_evaluation_v3/ground_truth.sqlite3` | Source of truth for saved recordings; action `result` JSON includes timing |
| `baseline_evaluation_v3/gui/metrics.html` | Metrics UI structure and definitions |
| `baseline_evaluation_v3/gui/metrics.js` | Run-ID matching, means, filters, per-action evidence, CSV export |
| `docs/ground_truth_metrics.md` | General Metrics-page documentation |
| `memory/baseline_evaluation_v3.md` | User workflow rules and recording history |

The new batch folder contains:

- `queue.json`: fixed original variant IDs, old run IDs, spec/dataset hashes, and
  exact skill/target sequences. This is the replay source; preserve it.
- `before/`: full pre-rerun T6 archive, full SQLite DB backup, review flags, timing
  CSV, index JSON, and the worker source before instrumentation. Never overwrite.
- `status.json`: array of attempt records. `running`, `saved`, or `failed`; includes
  all completed action measurements. It is atomically rewritten after each action.
- `<VARIANT>.json`: detailed evidence for that variant, also atomically written.
- `record.log`: UTC progress messages; one completed-action line per action.
- `server.log`, `server.pid`, `runner.pid`: operational logs and PID hints.
  Verify process command lines before sending signals; PIDs can become stale.
- `conditions.json`: CPU/GPU/platform and batch conditions.

## Worker timers: units and boundaries

All durations are **seconds**, measured with `time.perf_counter()` (monotonic).
UTC timestamps are for audit readability; duration calculations do not subtract
wall-clock timestamps, so clock corrections do not alter measured durations.

For each normal action, `action.result.timing` contains:

| Field | Measured boundary | Interpretation |
| --- | --- | --- |
| `action_wall_seconds` | Entry into `GroundTruthSession._do_run_skill` through skill execution and the live clip writer closing | Active skill wall time, including simulator work, skill control, observations, video work, cache refresh, and instrumentation overhead |
| `worker_command_wall_seconds` | Worker receives the skill branch through pre-action snapshot, alias resolution, skill execution, and post-action snapshot | Contains active skill time plus worker-side command preparation/finalization; excludes request transport before this branch and response serialization afterward |
| `environment_step_wall_seconds` | Sum of time inside the original `env.step(...)` calls | Inclusive environment-interface work, not isolated physics or GPU time; periodic evaluator publication added by the wrapper happens outside this sub-timer |
| `frame_callback_wall_seconds` | Sum of `set_frame_rgb` callbacks | Parent frame handling, live clip `append_data`, and throttled JPEG publication; does not cover every rendering/encoding operation elsewhere in the skill pipeline |
| `clip_close_wall_seconds` | Live clip writer `.close()` | Encoder flush/close for the additional GT clip writer; not all video encoding time |

**Do not add these five fields together.** Command time contains active skill time;
active time contains the diagnostic components. Components do not exhaust all work
and are not a validated exclusive physics/rendering/encoding decomposition.
The timers observe execution without changing the action, control frequency,
step count, video settings, or prescribed sequence.

`ReportAbsence` is handled in the server without a worker skill call. Its observed
request duration is measured, but no worker timing is available. Aggregate active
worker time sums the actual worker calls; do not invent an action timer for reports.

## Runner measurements

Per-variant JSON fields:

| Field | Meaning |
| --- | --- |
| `execution_elapsed_seconds` | From just before the first action submission to the client's observation of final-action completion; final evaluation must be successful before saving |
| `load_wall_seconds` | Sandbox POST through observed load completion, including worker startup; excluded from execution elapsed |
| `save_wall_seconds` | Record POST through observed save completion; excluded from execution elapsed |
| `action_wall_seconds` | Sum of worker active skill times |
| `worker_command_wall_seconds` | Sum of worker skill-command times |
| `environment_step_wall_seconds`, `frame_callback_wall_seconds`, `clip_close_wall_seconds` | Sums of corresponding diagnostic fields |
| `started_at`, `execution_started_at`, `execution_completed_at`, `completed_at` | UTC audit timestamps, not the duration clock |
| `max_sandboxes` | One for this batch |
| `poll_interval_seconds` | Client polling sleep: 0.2 seconds |
| `old_run_id`, `run_id` | Link original and replacement recording identities |
| `sim_steps`, `evaluation` | Saved step total and final task success evidence |

Each `actions[]` entry includes sequence number, original skill/target,
`submitted_at`, `completed_at`, `observed_wall_seconds`, the complete result
(including worker timers), and a resource sample. Observed duration includes
HTTP dispatch, worker transport, execution, state-query time, and polling.
Execution elapsed also includes small automatic bookkeeping gaps between actions.
There are no intentional per-action human pauses in this runner.

Resource snapshots at start, after every action, and after saving contain:

- `MemAvailable` and `SwapFree`, as kernel strings in kB.
- `load_average`: 1/5/15-minute load averages (not CPU percentages).
- Sample UTC timestamp.

There is no continuous GPU-utilization series or per-component CPU-time profile.
Do not infer those from the existing timers. The machine has an i7-7700K (8 logical
CPUs) and GTX 1060 6 GB. At launch no other simulators were active and about 8.4 GB
RAM was available. Desktop apps were retained because their use was unknown and
memory was sufficient; "one sandbox" does not mean an otherwise empty machine.

The existing worker transport also polls its IPC response every 0.1 seconds and
has a 600-second per-command timeout. This is separate from the client's 0.2-second
poll and from a baseline's combined-time budget.

## Save and publication checks

Before each variant: require an empty one-slot server and unchanged original
recording ID, spec hash, and dataset hash. Before/after every action: retain the
session identity and enforce the one-session invariant. Require the expected
request ID, action count, and successful action result.

Before saving: require final evaluation success and an unchanged original saved
recording. After saving: verify a new run ID, ready artifacts, complete step counts,
non-stale spec/dataset hashes, exact skill/target sequence, matching exported run
ID, and video SHA-256. Then publish timing and close that sandbox before the next.

The save path already marks replaced recordings as **re-recorded / needs human
review**. Automated success must not silently restore human verification.

A failure stops the batch and leaves the sandbox open for diagnosis. Existing
recordings are preserved until a successful save. If a check fails after a save,
the backup remains available; inspect `run_id` and the archive before doing more.

## Timing CSV and UI

`current_recordings.csv` retains old evidence for other tasks and receives new
T6 rows with:

- `timing_status = measured_single_sandbox`
- `interval_definition = first_action_request_to_final_success_observed`
- Replacement `run_id`
- `execution_elapsed_seconds`, `execution_elapsed_minutes`
- `action_wall_seconds`, `worker_command_wall_seconds`, `load_wall_seconds`,
  `save_wall_seconds`
- `source = single_sandbox_20260930/<VARIANT>.json`

The UI accepts elapsed values only for a matching current archive run ID and an
allowed status (`recovered_current`, `recovered_http_session`, or
`measured_single_sandbox`). Active times are accepted only for measured rows.
Missing, stale, invalid, and interrupted values are excluded, never zero-filled.

Open `/baseline_evaluation_v3/gui/metrics.html` through the viewer on port 8000.

- Default matrix comparison: elapsed recording time. Active skill time is also
  selectable; BASE-relative deltas use whichever metric is selected.
- Recordings table: elapsed, active skill, video time, and timing provenance per
  variant, alongside action and simulator-step counts.
- Task averages: mean elapsed, active skill, and video time for each task, each
  with its own timed/saved coverage.
- Overall mean: equal weight for each available recording, **not** equal weight
  for each task's average. Repeated BASE matrix cells are counted only once.
- Task/memory/scenario filters apply to summaries, means, rows, action details,
  and export. "Overall" means overall within current filters.
- Measured per-action timings: expandable table per instrumented variant,
  including observed, active, env.step, and frame-callback times, with separate
  loading/saving/worker-command totals.
- CSV export includes elapsed and active seconds plus status/interval definition.
  Full loading/saving/component/resource evidence remains in per-variant JSON.
- Polls index + timing CSV every five seconds while visible; a changed signature
  refreshes the page data. Only saved matching measurements enter averages.

Historical runs and new single-sandbox runs have different load/measurement
conditions. Label this distinction; do not claim a causal speedup from one rerun
alone. Previous timing coverage may shrink when recordings are replaced; this is
intentional protection against displaying timings from a different recording.

## Monitoring and safe recovery

Read-only progress commands from the repository root:

```bash
tail -10 baseline_evaluation_v3/ground_truth_timing/single_sandbox_20260930/record.log
python3 - <<'PY'
import json
from pathlib import Path
p = Path('baseline_evaluation_v3/ground_truth_timing/single_sandbox_20260930/status.json')
for r in json.loads(p.read_text()):
    print(r['variant'], r['status'], len(r['actions']), r.get('execution_elapsed_seconds'), r.get('error'))
PY
```

API base: `http://127.0.0.1:8000/baseline_evaluation_v3/api/ground-truth/<VARIANT>`.
GET includes `sessions`, `max_sessions`, and the current session. Do not repeatedly
fetch video/frame endpoints during a timing run. Normal runner state polling is
already active.

Do not blindly rerun the script after an error. It skips `saved` status entries,
but an interrupted save/publication may require reconciliation: compare status,
current DB/archive run ID, timing CSV, and exact sequence. A failure can leave an
open sandbox, which the script deliberately refuses to discard. Preserve its
logs and steps; recover without changing the approved sequence. Retrying a failed
attempt, if needed, must start a fresh complete sequence with separate evidence,
not silently stitch partial intervals into a complete duration.

Do not change a running server's `--max-sandboxes 1` constraint. Startup validates
all saved videos and may take several minutes; that is not execution time.

## Checks

```bash
python3 -m unittest scripts.test_ground_truth_worker_timing scripts.test_scenario_ground_truth scripts.test_ground_truth_sessions
node scripts/test_ground_truth_metrics.cjs
```

Use a modern Node runtime; this machine's system `node` is too old. A working one
is `/tmp/scenario-viewer-test/playwright/driver/node`. The worker unit test checks
that timing instrumentation preserves original environment calls, frame ordering,
and results without needing Habitat. JS checks cover missing/stale evidence,
measurement statuses, zero/invalid values, coverage and weighted means. A Chrome
fixture test also exercised per-action evidence, filters, active-time matrix,
averages and CSV export; its temporary script is `/tmp/test-t6-timing-ui.py`.
