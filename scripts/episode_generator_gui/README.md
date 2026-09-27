# Episode Generator GUI

Pick a PARTNR scene, enter a start state and an end state, and spawn a `CollaborationEpisode`.

OpenAI maps those boxes onto furniture IDs and object classes. Habitat spawns the start; evaluation propositions are compiled from the end state.

## Setup

```bash
conda activate habitat
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OPENAI_API_KEY=...
```

`LD_LIBRARY_PATH` is required so Habitat/llvmlite can find the conda `libstdc++`.

## Run

From the repo root:

```bash
python scripts/episode_generator_gui/app.py --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Select a scene, fill **Start state** and **End state**, click Generate.

Example: start `apple and orange on the dining table`, end `both inside the fridge`.

If the episode editor is already on 5000, use `--port 5001`.

## Output

Successful runs write:

```text
data/datasets/custom/<scene_id>/<folder-label>_YYYYMMDD_HHMMSS/dataset.json.gz
```

For baseline eval v2, put the same folder under:

```text
baseline_evaluation_v2/episodes/task_<N>/<folder-label>_YYYYMMDD_HHMMSS/dataset.json.gz
```

`folder_label` comes from the mapping LLM (kebab-case task name). Timestamp is when the episode was written. Example: `apple-orange-dining-table-to-fridge_20260903_130612`.

Episode id inside the file is `0`. Scene dropdown is PARTNR train+val IDs that exist under `data/hssd-hab/scenes-partnr-filtered/`.

## Goal state and evaluation

`goal_state` uses the same row schema as `initial_state`. After spawn, each goal row consumes the next spawned instances of that object class and becomes:

| `location` | Predicate |
| ---------- | --------- |
| `on` | `is_on_top` |
| `within` / `in` / `inside` | `is_inside` |
| `floor` | `is_on_floor` (+ `is_in_room` if a room is set) |
| `in_room` | `is_in_room` |

Optional per goal row: `object_states` (`is_clean`, `is_filled`, `is_powered_on`, …), `next_to` (object class list), `phase` (integer; later phases must finish after earlier ones).

All propositions get a `TerminalSatisfactionConstraint`. You can also pass `goal_state` directly into `generate_episode` without the GUI.

If the prompt is only a spawn and has no task, the LLM copies `initial_state` into `goal_state` (already-satisfied eval).

## Placement

After parse, every `initial_state` entry gets `smart_placement: true`.

`location: "on"` (default): largest upright receptacle per furniture parent, clustered on top, navmesh-reachable.

`location: "within"` / `"in"` / `"inside"`: use the scene `within_set` interiors (fridge shelves, cabinet interiors, drawers). Articulated doors/drawers are opened for sampling, then closed so the episode starts shut. Navmesh reachability is not required until the agent Opens the furniture.

If smart placement fails, fall back to Habitat `ObjectSampler.single_sample`.

## Visualize an episode

Skill runner (use a different port from the generator if both are up):

```bash
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir=. \
  +skill_runner_show_topdown=True \
  habitat.dataset.data_path=data/datasets/custom/<scene_id>/<timestamp>/dataset.json.gz \
  +skill_runner_episode_id=0
```

Episode editor:

```bash
python scripts/episode_editor/add_objects_to_scene.py \
  --dataset data/datasets/custom/<scene_id>/<timestamp>/dataset.json.gz \
  --episode-id 0 --port 5001
```

## File structure

```text
scripts/episode_generator_gui/
├── app.py                 # Flask UI + /api/generate
├── generation_service.py  # scene_info cache, OpenAI parse, spawn
├── prompt.txt             # LLM mapping prompt
├── templates/index.html
└── README.md
```

Spawn logic lives in `dataset_generation/benchmark_generation/generate_episodes.py` (`LLMRearrangeEpisodeGenerator.sample_objects`).

## API

| Endpoint         | Method | Description                                      |
| ---------------- | ------ | ------------------------------------------------ |
| `/`              | GET    | GUI                                              |
| `/api/scenes`    | GET    | Scene IDs with filtered HSSD assets              |
| `/api/generate`  | POST   | `{ "scene_id", "prompt" }` → spawn + dataset path |
| `/api/download`  | GET    | `?path=` relative path under `data/datasets/custom/` |

## Notes

- Furniture IDs are scene-specific (`table_4`, `counter_0`). Everyday names like “dining table” are mapped via product descriptions in scene_info. Rooms in this dataset often have no `dining_room`.
- One object class per `initial_state` row. Apple then orange on the same table still cluster because history is keyed by parent handle.
- Requires Habitat data (`hssd-hab`, object configs) and a working OpenAI key.
