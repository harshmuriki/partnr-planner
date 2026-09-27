# Scenario viewer

A browser page for inspecting every baseline_evaluation_v3 variant: its objects,
instruction, success criteria, robot memory, the generated episode's rendered
placements, and the spec file.

The viewer reads project files directly and uses a small Python server to save shared
human verification. There is nothing to build; reload after regenerating episodes.

## Start it

Run the viewer server from the repository root (the standard `http.server` cannot
save verification changes):

```bash
cd /path/to/partnr-planner
python3 scripts/serve_scenario_viewer.py --port 8000
```

Open to:

```
http://localhost:8000/baseline_evaluation_v3/gui/scenario_viewer.html
```

Working on a remote machine through VS Code: forward port 8000 in the **Ports** tab
and open the same path on the local address it shows.

Use this single server for the viewer. It serves both IPv4 and IPv6 localhost on
port 8000 in one process, sharing the same simulator, database, and verification
records.

## Page layout

### Verified checklist

Open the **Verified** checklist button at the top right to see every scenario once, grouped
by task, with per-task and overall verification counts. Click a scenario ID to open
its page. Check it off after a human has verified both its spec and episode starting
scene. The checklist opens in a collapsible side drawer.

Checks save in `baseline_evaluation_v3/human_verification.json` on the server. Every
browser connected to this repository sees the same checks, including after browser
or server restarts. Open pages refresh shared state every five seconds and on focus.
Old browser-local checks are imported automatically when that browser next opens the
viewer at its original address; existing shared decisions take precedence. Uncheck
a scenario to revoke verification. Clearing browser data does not remove shared checks.
These manual checks are separate from automated runtime verification and remain set
after regeneration, so re-review and uncheck any scenario whose spec or episode changes.

### Task tabs

T1 to T7 across the top. Click to switch task.

### Variant matrix

- **Rows** are the robot-memory levels: Accurate, Incomplete, Outdated.
- **Columns** are grouped by uncertainty dimension (object availability, containment,
  distractors).
- Each cell is one variant (`BASE`, `SUB`, `ABS`, `CON`, `DIS`)
  with its object count. `N/A` means the task has no such variant.
- `BASE` appears in several columns because it is the "no uncertainty" setting of each
  dimension, so selecting it highlights all of those cells.

### Selected variant

- **Instruction**: the prompt given to the robot.
- **Success criteria**: the scored propositions. `Order:` lines are steps that must
  happen before a later criterion (e.g. soap next to a glass before it is cleaned, bread
  inside the microwave before it is placed on the desk).
- **Initial robot memory**: what the baseline is initialized with at this memory level.
- **Object cards**: the object's preview animation, role (target, substitute,
  distractor), state changes (start → end), start location and goal location, and the
  pinned asset id.

### Generated episode

The saved episode for the variant, rendered from its actual saved placements.

- `Generated` and `Runtime verified` badges show the episode's status.
- Click an object on the left to see its views. Use **View** to change camera angle,
  scroll or `+`/`−` to zoom, drag to pan, **Fit** to reset, **Open image** for full size.
- "Interior" views are taken from inside closed furniture; doors are not opened for
  the picture.
- Below the image: start, expected destination, saved position, surface, asset and
  initial states.
- **Download dataset** gives the episode's `dataset.json.gz`; **Placement validation
  data** opens its `review.json`.

Variants that share a physical scene with another memory level (e.g. `T3-OUT-BASE`
and `T3-ACC-BASE`) show the same images.

### Spec

Collapsed panel at the bottom with the full spec markdown.

- **Spec** shows the file.
- **Diff** compares it with another variant of the same task (default: the same
  scenario at the next memory level). **Only changed sections** hides identical sections.

### Ground truth

Below **Spec**, every variant has a ground-truth recorder and target-room selections.

Each variant has **one** saved ground truth. There is no run history.

**Assumptions**, at the top of the Ground truth panel, holds two free-text notes: one for this
variant and one shared by every variant of the task (e.g. all of T1). They save automatically
about a second after you stop typing, and are shared across browsers. They are stored in
`ground_truth.sqlite3` (table `notes`) and mirrored to `ground_truth/assumptions.md`, which lists every note.
Clearing a box deletes that note.

- **Open sandbox** loads that variant's exact saved dataset into the Skill Runner
  simulator, with the saved robot starting pose. The live cameras, apartment/object tree,
  and skill controls let you Navigate, Explore, Pick, Place, Open, Close, Fill, Pour,
  Clean, PowerOn, and PowerOff as robot agent 0. Click an entity in the tree to set its target.
- The sandbox is practice: **nothing you do there is saved**. Each attempt is listed under
  **Sandbox steps**. Successful steps start checked, and failed ones start unchecked. Check,
  uncheck, or remove (✕) steps to build the sequence you want. **Reset sandbox to the start**
  reloads the episode and clears the list.
- **Record ground truth** resets the episode to its exact start and replays only the
  checked steps, in order. It is saved only if the replay completes the spec. ABS variants
  also need a checked **Report no suitable object** step. A saved recording **replaces**
  the previous ground truth. If the replay does not complete the task, nothing is saved and
  the previous ground truth is unchanged. Afterwards the sandbox continues from the replay's
  end state, with the replayed steps listed.
- **Initial robot memory** and **Completion requirements** show this variant's spec
  sections. **Place** uses `object,on/within,destination,none/next_to,reference/none`,
  for example `jug_0,on,table_0,none,none`. Spec entity names are mapped to runtime entities.
- The saved ground truth counts one action per replayed step, and counts actual simulator
  steps separately. Episodes whose spec hash differs from the current spec cannot be loaded.
- Under **Most likely rooms and furniture for each target**, select rooms worth visiting and rank them using **↑ / ↓**.
  The numbered list is the search order: **1 is most likely and visited first**.
  Adding a room places it last; unchecking it removes it from the search plan.
  The saved array order is the ranking, including in exported `room_choices.json`. The list
  comes from the full apartment metadata, including rooms unused by the task and
  absent targets. Choices and rankings save automatically across browsers and are shared by object
  name: every variant containing `jug_0` shows the same rooms, in every task. Tasks
  use different apartments, so each variant shows only the rooms its own apartment has;
  rooms chosen in an apartment that lacks them stay saved for the variants that do.
- Each target also has a ranked **Likely furniture** list: the furniture it is most likely on or
  inside (useful for containment variants). **Add furniture to check** lists the apartment's
  furniture grouped by room, with your ranked rooms first. Use ↑ / ↓ to order it (1 is checked
  first). Names are the spec's furniture names. Furniture is shared by object name within the
  same apartment, so every T1 variant shares `jug_0`'s list, and T3/T5 share an apartment.
  It is saved in `ground_truth.sqlite3` (`object_furniture`) and exported per variant
  as `furniture_choices.json`. A saved ground truth's `run.json` records it as it was at save time.

The server stores each variant's ground truth (actions, counts, spec/dataset hashes,
video size and SHA-256) and the room choices in `baseline_evaluation_v3/ground_truth.sqlite3`.
Files are in `baseline_evaluation_v3/ground_truth/T<task>/<variant>/`. One simulator runs at a
time. Opening another variant's sandbox discards the current sandbox steps. A server
restart discards the sandbox and any unfinished recording, but never the saved ground truth.

#### Saved data and videos

**All saved ground truths** in the Ground truth panel opens the archive index. Each saved
ground truth has a combined MP4 of the replay at 30 FPS. Expand **Ground-truth video** to
watch it, or download its MP4, run JSON, or actions CSV. A report-only/zero-step
recording saves a short view of the actual initial scene.

```text
baseline_evaluation_v3/
  ground_truth.sqlite3                     # durable source of truth
  ground_truth/
    index.html                            # browse every saved ground truth
    index.csv / index.json                # summary, one row per variant
    T1/T1-ACC-BASE/
      room_choices.json                   # current room annotations
      run.json                            # counts, action results, evaluation, hashes
      actions.csv                         # ordered actions and simulator steps
      video.mp4                           # full replay
      initial.jpg                         # actual starting view
      clips/                              # individual action recordings
```

Run JSON freezes the room selections at the time it was saved. The variant's `room_choices.json`
tracks later edits. On startup the server checks each video against its stored checksum
and rebuilds a missing or changed video from the clips.

A video/export error is shown separately; it does not delete the completed task data.
Keep the archive and SQLite database together when backing up. Stop the server before
copying the database, or use SQLite's backup API while it is running.

The server launches Habitat only when you start an episode. It defaults to
`~/miniconda3/envs/habitat/bin/python`; use `--habitat-python /path/to/python` or
`HABITAT_PYTHON` if your environment lives elsewhere. It needs the same scene assets,
robot models, and GPU/EGL access as the existing Skill Runner GUI.

## Keyboard and links

| Key       | Action                                      |
| --------- | ------------------------------------------- |
| `←` / `→` | Previous / next variant in the current task |
| `↑` / `↓` | Previous / next task                        |

The selected variant is kept in the URL hash, so links such as
`scenario_viewer.html#T7-ACC-BASE` open directly on that variant.

## Where the data comes from

| Shown                                                 | File                                                        |
| ----------------------------------------------------- | ----------------------------------------------------------- |
| Variants, objects, roles, states, asset ids           | `baseline_evaluation_v3/variant_object_assets.csv`          |
| Instruction, criteria, memory, start/goal, spec panel | `baseline_evaluation_v3/specs/T*/T*-*-*.md`                 |
| Episode status and badges                             | `baseline_evaluation_v3/generation/episodes.json`           |
| Per-object placement details                          | `baseline_evaluation_v3/episodes/task_*/*/review.json`      |
| Episode images                                        | `baseline_evaluation_v3/gui/episode_views/`                 |
| Open issues (shown only if any exist)                 | `baseline_evaluation_v3/generation/audit.json`              |
| Object preview animations                             | `data/hssd-hab/metadata/object_gifs/<category>/<asset>.gif` |

## After changing data

- **Edited the CSV or a spec**: reload the page.
- **Regenerated episodes**: reload the page; the generator writes new images to
  `episode_views/` and updates `episodes.json`.
- **Replaced preview GIFs**: change `GIF_VERSION` near the top of the script in
  `scenario_viewer.html` so browsers stop using cached images.

## Troubleshooting

If Ground truth stays at **Loading ground truth…**, reload the page and use
`http://localhost:8000/baseline_evaluation_v3/gui/scenario_viewer.html`. A plain
`python3 -m http.server` cannot load or save ground-truth data. The panel now shows
a connection error with **Retry connection** and a link to port 8000; stalled requests
time out after 12 seconds. Room checkboxes appear before starting the simulator.

| Symptom                                                    | Fix                                                                                                        |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| "Could not load `variant_object_assets.csv`"               | The page was opened as a file or from the wrong folder. Serve the repository root and use the URL above.   |
| "No generated episode available" / "Queued for generation" | The variant has no saved episode yet; check `generation/episodes.json`.                                    |
| "Saved placement image could not be loaded"                | The variant's `episode_views/` folder is missing; rerun the generator with `--render-only --variant <ID>`. |
| Card shows "no image"                                      | No preview GIF exists for that asset under `object_gifs/`.                                                 |
| Old content after regeneration                             | Hard refresh (Ctrl+Shift+R).                                                                               |

The object gallery with every preview animation is at
`http://localhost:8000/data/hssd-hab/metadata/object_gifs/index.html`.
