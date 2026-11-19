# Episode logs that survive timeouts

All baselines use shared incremental logging in `EvaluationRunner`, `Planner`,
and `EnvironmentInterface`. Logs are saved during execution, including when
Explore or another skill steps internally without returning to the runner.
No timeout increase or successful episode termination is required.

Under `<results_dir>/<dataset>/`:

| File | Contents |
| --- | --- |
| `actions/actions-<episode>.json` | Readable list of attempted actions, their arguments, responses and status |
| `actions/actions-<episode>.jsonl` | Flushed action-start/finish events, written before/after skill execution |
| `planner-log/planner-log-<episode>.jsonl` | Every returned planner step, written incrementally |
| `planner-log/planner-log-<episode>.json` | Standard planner log, checkpointed at action boundaries and at least every five seconds of returned planner steps |
| `planner-log/checkpoint-<episode>.json` | Latest completed simulator step and evaluator metrics, including internal skill steps |
| `traces/<agent>/trace-<episode>-<agent>.txt` | Most recent returned planner trace |
| `prompts/<agent>/prompt-<episode>-<agent>.txt` | Most recent returned planner prompt, when supplied by that planner |
| `detailed_traces/detailed_trace-<episode>.pkl` | Action/state history checkpoint at returned action boundaries, if enabled |
| `llm_costs/<episode>.json` | Incremental model usage/cost report, including pending requests |

Action status `finished` means the skill returned a response (which can report
failure), not that the task succeeded. `incomplete` means the action was still
in progress when execution stopped. An interrupted action never gets an invented
success response. Action counts use unique attempts: repeated simulator ticks of
the same skill do not create extra actions; a retry after a response is a new attempt.
An action-start event records dispatch, not proof it executed a simulator step.

SIGTERM unwinds the runner and writes its last in-memory checkpoint. For SIGKILL,
`run_parallel.py` reconstructs the standard JSON after the worker exits using
flushed journals. An incomplete final JSONL line is ignored while earlier records
are retained. The launcher marks recovered logs `status: timeout`, `complete:
false` and marks metrics `metrics_are_partial: true`. Partial evaluator metrics
are measurements at the last checkpoint, not a final successful episode result.

For manual/external kills outside `run_parallel.py`, journals and checkpoints
still survive. To materialize the latest standard log and mark the cutoff:

```sh
python scripts/recover_episode_logs.py /absolute/path/to/run_output --status timeout
```

Normal completion flushes once without appending duplicate steps. Sequential
subgoals retain one action/step history; separate runs reset it. Completed simulator
steps persist immediately; a step interrupted within the simulator cannot be
claimed complete. A process crash before episode initialization has no actions
to save. HTML/video rendering is not required to retain these files.

Verification: `python -m unittest scripts.test_episode_progress
scripts.test_run_parallel` exercises actual process termination and launcher
recovery. `scripts/test_timeout_logging_gpu.py` runs a short Wait then blocks
inside another skill on the real simulator and forcibly terminates workers for
all three baseline configs, without LLM calls.

## Action-limit termination

When ReAct reaches `replanning_threshold` or TRU-POMDP reaches `max_decisions`,
the console prints `Reached max actions (50).` (using the configured limit).
The trace records the same text under `Termination:`, and generated HTML shows
it as a notice above the episode metrics. The final planner record and
`run.json` metrics include a `termination` object with the reason `max_actions`,
message, limit, count, and source configuration key.

These remain the existing planner limits: ReAct counts replans and TRU-POMDP
counts symbolic decisions. VLM-TAMP's separate reprompt budget keeps its
`MAX REPROMPTS REACHED` notice. A normal ReAct `Done` response is not labeled as
an action-limit stop. Termination notices do not change evaluator success or
introduce a new action limit, and historical runs are not labeled by guessing
from their action count.
