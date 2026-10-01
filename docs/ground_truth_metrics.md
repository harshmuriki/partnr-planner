# Ground-truth Metrics page

Open `baseline_evaluation_v3/gui/metrics.html` through the scenario viewer server,
 or select the **Metrics** tab in the viewer.

The page compares saved ground truths using one matrix per task. Rows are Accurate,
Incomplete, and Outdated memory. Columns mirror the scenario viewer:

| Group | Columns |
| --- | --- |
| Object Availability | Target Exists (BASE), Substitute Available (SUB), No Suitable Object (ABS) |
| Object Containment | On Surface (BASE), Inside Closed Receptacle (CON) |
| Distractor Presence | None (BASE), Present (DIS) |

BASE is repeated for visual comparison; it is counted only once in summaries and
CSV exports. Filters apply to the matrices, summaries, recordings, and export.
Each cell shows the selected metric and links to its variant in the viewer.

## Metric definitions and limitations

Actions are high-level skill calls; simulator steps count environment steps within
those calls. Navigate, Explore, Open, and Close are action counts, not inferred
search effort. All submitted actions in a saved recording count, including failed
attempts. Means include only known values for saved recordings in the current
filters; coverage can differ between groups. These are descriptive comparisons of
single recordings, not statistical estimates.

This page reads the saved archive only and does not open or modify a sandbox.
Saved recordings are not checked against current spec or episode hashes here;
inspect a recording in the viewer for freshness and criterion evidence.

Video time is the length of each variant's saved ground-truth MP4, read by the
browser from the video file. It excludes human pauses between actions. The
recordings table shows it per variant, and Export CSV includes
`video_duration_sec` (seconds, three decimals).

Incomplete simulator totals are excluded from comparisons and means. A dash means
missing or unknown; N/A means the variant does not exist. Delta compares the
selected metric with the same task and memory's BASE recording. Bars are scaled to
the largest known selected metric within each task's current filters.

Data comes from `variant_object_assets.csv`, `ground_truth/index.json`, and each
saved recording's `actions.csv`. The page checks the archive index every five seconds while visible and on return
to the tab. It updates metrics only when the saved archive changes (including a
replacement recording), preserving filters. Refresh reloads immediately. Successful
background checks show no status banner. If refreshing fails, the previous
snapshot remains visible with an error message.

## Elapsed recording times and task averages

Elapsed recording time is the default comparison metric. The recordings table and
CSV export include `execution_elapsed_seconds`, timing status, and the interval
definition. Values come from `ground_truth_timing/current_recordings.csv` and are
used only when its `run_id` matches the current archive recording ID and its
status is recovered. Missing, interrupted, and replaced-run timings are excluded,
not counted as zero. The timing audit describes the different interval definitions;
these include processing/recording/polling overhead and HTTP intervals may include
human pauses. They are distinct from video playback and simulated physical time.

The task averages table always displays mean elapsed and video time for each task,
plus an overall mean per available recording (not an unweighted mean of task
means). Coverage is shown separately for each time measure. Filters apply to both
task and overall averages. Timing CSV changes also trigger the automatic refresh.

## Instrumented single-sandbox T6 rerun (2026-09-30)

The authorized rerun preserves all 15 saved T6 skill/target sequences verbatim.
`ground_truth_timing/single_sandbox_20260930/before/` preserves the previous DB,
recordings, timing CSV, review flags, and worker code; `queue.json` is the fixed
sequence source. The server permits one sandbox and the runner closes each saved
session before opening the next. Any action or verification failure stops the queue.

Each variant JSON records monotonic per-action request-to-observed-completion
intervals, UTC timestamps, memory/load samples, and worker-side timing:

- `action_wall_seconds`: the sum of active skill execution durations, including
  rendering, world updates, and encoding. Excludes gaps between commands.
- `worker_command_wall_seconds`: also includes before/after snapshots and alias
  resolution. This contains active skill time; do not add the two.
- `execution_elapsed_seconds`: first action request to observation of final
  successful completion, including automatic polling and dispatch overhead.
- `load_wall_seconds` and `save_wall_seconds`: separate from execution elapsed.
- Environment-step, frame-callback, and clip-close durations are diagnostic
  components of active skill time, not additional top-level times. They do not
  separately isolate physics, GPU rendering, and all encoding work.

The runner publishes the elapsed and active values only after verifying success,
unchanged action sequence, current hashes, exported run ID, and video SHA-256.
New values have `measured_single_sandbox` status and match the replacement run ID.
The Metrics page displays this provenance, active-time averages and coverage,
and expandable per-action evidence. Historical elapsed values are still available
for other tasks; their load/interval conditions differ from the new measurements.
