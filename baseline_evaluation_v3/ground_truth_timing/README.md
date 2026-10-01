# Ground-truth elapsed times

## Instrumented T6 reruns (2026-09-30)

The single-sandbox rerun uses all 15 saved T6 action sequences unchanged.
Progress and per-action measurements are in `single_sandbox_20260930/`.
`before/` preserves the original recordings and timing CSV. Successfully saved
reruns update `current_recordings.csv` with `measured_single_sandbox` status and
the new run ID. See `../../docs/ground_truth_metrics.md` for timing definitions.
Active skill time, execution elapsed, loading, and saving are measured separately.
The UI excludes evidence that no longer matches a saved recording's run ID.

## Historical recovery (snapshot at 2026-09-29)

The counts and ranges below describe the original recovery, not live coverage.

Recovered on 2026-09-29 from retained automatic recorder logs and dated viewer HTTP logs. No sandboxes were run and no saved ground truths were changed.

73 of 90 current saved recordings have recoverable full session intervals: 60 from automatic recorder logs and 13 additional intervals from HTTP requests. HTTP-only records establish session duration, but the request log alone does not prove whether each command was submitted automatically or manually. The other 17 remain explicitly missing, partial, or interrupted; blanks are not zero.

| Task | Recovered / saved | Observed elapsed range (m:ss) |
| --- | --- | --- |
| T1 | 13 / 15 | 1:23–5:20 |
| T2 | 5 / 15 | 1:51–7:32 |
| T3 | 12 / 12 | 0:36–4:54 |
| T4 | 12 / 12 | 1:15–4:36 |
| T5 | 15 / 15 | 1:05–2:44 |
| T6 | 13 / 15 | 7:29–34:58 |
| T7 | 3 / 6 | 9:40–11:31 |

## Files

- `current_recordings.csv`: one row for every current saved variant; elapsed seconds/minutes, interval definition, matching evidence reference, and missing-data status.
- `attempts.csv`: 74 retained automatic recording attempts/segments, including failures, interrupted attempts and resumed segments. These are not 74 distinct variants. Only complete successful intervals populate execution duration.
- `http_intervals.csv`: dated first-action to save-request intervals matched to the current recording by full save-request date/time (within 3 seconds) and action count. Includes overlapping evidence for automatic logs. An interval with a request gap over 30 minutes is flagged as interrupted and not used to fill duration. This threshold does not prove shorter intervals are pause-free.
- `summary.csv`: descriptive ranges/medians of recovered current recordings only, mixing interval definitions as described below; not repeated-trial estimates.
- `sources/` and `sources_manifest.json`: preserved recorder logs and extracted HTTP evidence, SHA-256 hashes and original locations. Original log line references in `attempts.csv` refer to the archived identical log. HTTP snippets preserve the dated request lines; manifest references original line locations.

## Timing definitions and limitations

`loaded_to_success`: automatic loop's "loaded N steps" timestamp to its successful final evaluation. Excludes scene loading and saving; includes processing, rendering, encoding during actions, API overhead and two-second polling. Commands were issued by the recorder loop, without per-action human interaction.

`first_action_to_save_request`: automatic re-recording loop's first action submission to the database recording creation time. The code creates that database row only after all actions and successful final evaluation. Excludes scene loading and final save/assembly; includes three-second polling. Matching uses full action sequence (logged target prefixes where truncated), action count and save-completion clock within 120 seconds of the current recording in America/New_York. Batch logs omit dates, so this is corroborated matching, not a stored run-ID join.

`first_action_to_save_request_http`: first accepted action POST to the matching record POST. Excludes loading and final saving, but can include pauses between requests. It is a session elapsed interval, not a sum of measured active skill times.

`recording_to_save_seconds` in attempts is a separate wider interval: for queue logs it includes scene loading and saving. Do not substitute it for execution time.

Some runs shared the machine concurrently. Timing includes simulator/rendering cost and recording overhead and depends on machine load. These measurements cannot be interpreted as robot physical task duration or controlled single-run benchmarks. They are distinct from video duration, steps/120 accounting, and the baseline combined-time budget.

Resumed T6 INC/OUT-SUB recordings do not have a reliable complete uninterrupted interval. Their partial logs remain available without treating overnight pauses as execution time. The HTTP interval for T7-INC-DIS has a gap over 30 minutes and is excluded. Other absent values lack retained complete matching evidence. No durations were imputed from frame counts, file modification times or neighbouring variants.

This audit supersedes the narrower `../ground_truth_t6_elapsed_audit.csv` for coverage; its 11 previously recovered T6 intervals remain unchanged.
