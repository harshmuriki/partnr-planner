# Episode object-state reports

VLM-TAMP, ReAct and TRU-POMDP automatically write a JSON report at normal episode
termination (including step/time limits handled by the evaluation runner):

`<results_dir>/<dataset_name>/object_states/<episode_output_name>.json`

Each entry under `objects` pairs `initial` and `final` snapshots. Snapshots include
entity kind, simulator handle, room, supporting furniture, holding agent, world
position in meters, spatial relations, and applicable states (for example
`is_clean`, `is_filled`, `is_powered_on`, and furniture `is_open`). All movable
objects and furniture are included, even if the planner never observed them.
Locations and states come from refreshed simulator ground truth, not robot memory.
Missing states and null locations are unknown, not false. `relations` preserves
the simulator graph's labels and includes containment in movable containers.

The initial snapshot is taken before planning. Sequential subgoals using
`preserve_planner_state=True` retain that original snapshot and refresh the final
snapshot in the same report. Separate episode runs use separate output names.
The initial snapshot is written before planning. The final snapshot is written
before video and trace rendering; graceful SIGTERM or an exception also attempts
a cutoff snapshot, marked incomplete with the termination status. A hard kill
can leave only the initial snapshot (`final_snapshot_available: false`); unknown
final values remain null. A cutoff snapshot is not a completed-task result.
These reports are output artifacts and are not passed to the planner.

## API-free GPU verification

`scripts/probe_object_state_report.py` accepts the same Hydra command-line options
as `habitat_llm.examples.planner_demo`. It loads a real scene and renders RGB,
then uses the shared evaluation runner with a deterministic object position/state
change instead of an LLM planner. It checks initial/final values, object coverage,
subgoal continuation and initial-snapshot reset, and writes `probe_result.json`.
This verifies the report lifecycle; it does not evaluate planner performance.
