---
name: run-partnr-planner
description: Run, launch, drive, smoke-test or screenshot the PARTNR/Habitat planner fork (RailPlanner, VLM-TAMP) on a baseline_evaluation_v3 episode without paid API calls - load an episode, execute oracle skills (Navigate/Pick/Place/Open/PowerOff/Fill), expand RailPlanner subgoals, read task success, save robot camera PNGs; also run the rail planner unit tests and batch runs via run_parallel.
---

# Run partnr-planner

The app is a Habitat simulator episode plus a planner. The agent path is
`.claude/skills/run-partnr-planner/driver.py`. It loads one v3 episode with the
**same Hydra overrides as `scripts/run_parallel.py`** and builds the real `RailPlanner`.
It then reads commands from stdin: oracle skills, RailPlanner subgoal expansion,
goal measures and camera snapshots. It **never calls the VLM**, so it costs
nothing. All paths are relative to the repo root.

## Prerequisites

- Conda env: `export PATH=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin:$PATH`
- A GPU node. The login node (`sky2`) has no GPU and no libOpenGL. Even
  `import habitat_sim`/`magnum` fails there. Wrap everything below in `srun`
  (the `overcap` account works and doesn't use up rail-lab GRES).

## Run (agent path): the driver

```bash
cd /coc/flash5/vmuriki3/tamp/partnr-planner
export PATH=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin:$PATH
OUT=outputs/driver_smoke_$(date +%H%M%S)      # MUST be on shared storage, not /tmp
printf 'snap start\nachieve powered_off(lamp_0)\nachieve filled(jug_0)\nachieve on(jug_0, table_0)\nsnap end\nmeasures\nquit\n' | \
  srun -A overcap -p overcap --gres=gpu:a40:1 --cpus-per-task=4 --mem=32G --time=0:30:00 \
  python .claude/skills/run-partnr-planner/driver.py \
    --episode baseline_evaluation_v3/episodes/task_1/t1-acc-base --out $OUT > $OUT.log 2>&1
grep '\[driver\]' $OUT.log
```

This takes about 1 minute after the node is allocated. It solves T1-acc-base and ends with
`task_state_success=True task_percent_complete=1.0`. The screenshots go to
`$OUT/snap_<name>_<sensor>.png`. `agent_0_third_rgb.png` (third-person, 512²) is
the useful one. `agent_0_head_rgb.png` is 256².

Driver commands (one per line; failed commands print `ERROR ...` and the session keeps going):

| Command | Effect |
|---|---|
| `instruction` / `graph` / `state` | episode text / robot world graph / RailPlanner `SceneState` (holding, placements, closed furniture) |
| `plan <subgoal>` | `planner._parse_subgoal` → `expand_subgoal` on the live state; prints each action and the tool call it maps to |
| `achieve <subgoal>` | same, then executes each tool call; stops on the first failed response |
| `Navigate x` · `Pick x` · `Place obj, on, furn, None, None` · `Open x` · `PowerOff x` · `Fill x` … | one oracle skill |
| `measures` | `task_state_success`, `task_percent_complete` (read-only) |
| `hint` | placement-oracle hint for every object (finals, plus intermediate stops such as `waited_in`) |
| `snap <name>` | saves every RGB sensor as a PNG |

Options: `--planner rail|vlm-tamp` and trailing Hydra overrides, e.g.
`evaluation.agents.agent_0.planner.plan_config.end_of_plan_check=oracle`.
Subgoal syntax is the planner's: `on(a, b)`, `in(a, b)`, `waited_in(a, appliance)`, `powered_off(x)`,
`filled(x)`, `open(x)`, `holding(agent_0, x)`.

For interactive use, run the same `srun ... python driver.py ...` with `--pty`
inside `tmux` and type commands.

## Run with the real VLM (paid, only when asked)

Batch runs go through `scripts/run_parallel.py` (see `docs/PARALLEL_RUNS.md`). Check the
generated commands first, which costs nothing:

```bash
python scripts/run_parallel.py --episodes 'baseline_evaluation_v3/episodes/task_1/t1-acc-base' \
  --planner rail --workers 1 --output-dir outputs/drytest --dry-run
```

Without `--dry-run` it needs a Slurm step with a GPU and `OPENAI_API_KEY`
(read from `.env`), and every episode makes paid OpenAI calls.

## Test

```bash
srun -A overcap -p overcap --cpus-per-task=2 --mem=16G --time=0:20:00 \
  python -m pytest habitat_llm/tests/test_rail_planner.py -q -p no:cacheprovider   # 50 passed
```

## Gotchas

- **`--out` under `/tmp` silently loses the screenshots.** The compute node has
  its own `/tmp`. The driver prints `saved /tmp/...` and the files never reach the
  login node. Use `outputs/...` or `results/...`.
- **The world graph doesn't record placements after a raw skill.** After the
  driver's `Place`, `state` shows `jug_0: None`. The planner's
  `get_next_action` loop does the perception refresh that fixes it; the driver
  bypasses that. Trust `measures` and `snap` over `state` after a Place.
- **Room names in `spec.md` don't match the graph.** The spec says
  `living_room_0`/`bedroom_0`; the robot's graph uses `living_room_1`/`bedroom_1`.
  Object and furniture names (`jug_0`, `table_0`) do match.
- `filled(jug_0)` navigates to `cabinet_6` (the expander's `_nearest_faucet`
  choice in this scene), not a node named "sink". Fill still succeeds there.
- A human agent (`agent_1`) is present in the config even for single-agent
  baselines, so `snap` also writes `agent_1_*` images.
- `Navigate <bad name>` doesn't return a failure string. `execute_skill` raises
  `AssertionError: No low level actions returned` with the tool's message inside.
- `measures` reflects the last env step only. It's read-only on purpose:
  calling `update_metric` mid-episode appends to the scored state sequence.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError ... magnum/__init__.py` when running pytest | You're on the login node; use the `srun` line. |
| PNG paths printed but `ls` shows nothing | `--out` was on node-local `/tmp`; rerun with `outputs/...`. |
| `run_parallel.py: error: Run inside a Slurm step or supply --job-id` | Expected without `--dry-run` on the login node. |
