# Parallel ground-truth recording

Run one viewer server with up to two independent Habitat workers:

```bash
python3 scripts/serve_scenario_viewer.py --port 8000 --max-sandboxes 2
```

Save any recording you want to keep before restarting an older viewer server.
Live simulation state does not survive a server restart. Do not launch another
server against the same database/archive: startup migration and cleanup are
server-wide operations.

Open different variants in separate tabs and select **Open sandbox** in each.
Each open variant owns its simulation, action history, evaluation, camera, and
recorded clips. Navigating away or reloading a tab leaves its sandbox running.
Returning to the variant reattaches to it. Two tabs on the same variant share
that variant's session; use different variants for independent recordings.

The viewer shows the open variants and the number of occupied slots. When all
slots are occupied, explicitly **Close sandbox** on one variant to release its
worker. Closing discards that sandbox's unsaved actions and scratch clips;
saved ground truth is retained. Busy sandboxes cannot be closed or reset.
Saving keeps the sandbox open, so you can inspect or continue it. No session is
automatically evicted. After viewing saved video, **Return to live sandbox**
resumes the existing sandbox without resetting it.

Recording semantics are unchanged: load the exact episode, record every action
(including failed attempts and applied packs), then save the live recording
without replay once the requirements pass. Each variant has one saved ground
truth. Notes, action packs, and room/furniture annotations remain shared.
Durable archive writes are serialized; simulator actions run independently.

## Agent API

All routes remain under `/baseline_evaluation_v3/api/ground-truth/<variant>`.

1. `POST /sandbox` with `{}` returns `{loading: true, session_id: "..."}`.
2. Poll `GET` on the variant's base route until `session.busy` is false;
   check `session.error` and `session.mode` before acting.
3. Include `session_id` in subsequent `/action`, `/packs`, `/steps`, `/record`,
   `/sandbox` (reset), and `/close` JSON requests. Actions still require their
   own unique `request_id`, `skill`, and `target`.
4. Fetch frames with `/frame?session_id=...`.
5. `POST /close` releases the slot after the current operation finishes.

The base state response includes `session.id`, `sessions` (open IDs, variants,
busy flags, phases), and `max_sessions`. A reset issues a new session ID; stale
IDs and IDs belonging to a different variant are rejected. Existing scripts
that omit the ID still route by variant, but should adopt IDs to detect resets.
A full pool rejects a new sandbox without changing existing recordings.

The limit is configurable, but GPU memory and rendering/encoding throughput
still constrain useful concurrency. The default of two is not a measured GPU
capacity guarantee. Validate real Habitat workloads before raising it.

## Validation

```bash
python3 -m unittest scripts.test_ground_truth_sessions scripts.test_scenario_ground_truth scripts.test_scenario_viewer_server scripts.test_ground_truth_evidence
```

The session tests use independent deterministic worker doubles and real video
encoding in temporary storage. They cover overlapping actions, isolated frames
and evaluation, simultaneous saves, capacity races, close/reuse, stale IDs, and
HTTP routing. They do not benchmark concurrent Habitat GPU workloads.
