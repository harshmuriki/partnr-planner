#!/usr/bin/env python3
"""API-free driver: load one v3 episode with a real planner config and drive it.

Run from the repo root on a GPU node:
  printf 'graph\nsnap start\nachieve on(cup_0, table_16)\nmeasures\n' | \
    python .claude/skills/run-partnr-planner/driver.py \
      --episode baseline_evaluation_v3/episodes/task_1/t1-acc-base --out /tmp/drv

Commands (one per line on stdin; '#' comments and blank lines ignored):
  graph                 robot world graph (what the planner sees)
  state                 RailPlanner SceneState (holding, placements, open, ...)
  instruction           episode instruction
  snap <name>           save every RGB sensor to <out>/snap_<name>_<sensor>.png
  measures              task_state_success / task_percent_complete
  hint                  placement-oracle hint for every object (privileged; no VLM)
  plan <subgoal>        RailPlanner expansion of e.g. on(cup_0, table_16); no execution
  achieve <subgoal>     expand, then execute each step with the oracle skills
  <Tool> <args>         one skill, e.g. Navigate table_16 | Pick cup_0 |
                        Place cup_0, on, table_16, None, None | Open cabinet_3
  quit
No VLM/LLM request is ever made: the planner object is constructed but its
get_next_action is never called.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

PLANNERS = {
    "rail": "baselines/single_agent_rail_planner",
    "vlm-tamp": "baselines/single_agent_vlm_tamp_pddl",
}


def say(msg=""):
    print(f"[driver] {msg}", flush=True)


def topdown(env, uid, out, name="x", *entities, zoom=2.5, margin=1.0):
    """Look straight down with the third-person camera over the robot plus `entities`.

    Writes <out>/topdown_<name>.png and topdown_<name>.json (pixel x/y of each
    entity and the robot); the entity `scene` frames the whole navmesh. The camera sits `zoom` times higher than needed and the
    frame is center-cropped, which flattens perspective on tall furniture. Other
    agents (the human) are moved out of view. Raise third_rgb_sensor.width/height
    via Hydra to keep the crop sharp. Only valid until the next env step.
    """
    import json

    import magnum as mn
    import numpy as np
    from PIL import Image

    sim = env.sim
    key = next(k for k in sim._sensors if k.endswith("third_rgb") and str(uid) in k)
    sensor = sim._sensors[key]
    robot = np.array(sim.agents_mgr[int(uid)].articulated_agent.base_pos)
    pts = {"robot": robot}
    frame_pts = []
    for e in entities:
        if e == "scene":  # frame the whole navmesh (the entire apartment)
            frame_pts += [np.array(b) for b in sim.pathfinder.get_bounds()]
            continue
        pts[e] = np.array(env.full_world_graph.get_node_from_name(e).properties["translation"])
    xz = np.array([[p[0], p[2]] for p in [*pts.values(), *frame_pts]])
    lo, hi = xz.min(0) - margin, xz.max(0) + margin
    cx, cz = (lo + hi) / 2
    hfov = np.deg2rad(float(sensor._sensor_object.hfov))
    half = max(hi - lo) / 2
    eye_y = robot[1] + zoom * half / np.tan(hfov / 2)
    for i in range(len(sim.agents_mgr)):
        if i != int(uid):
            other = sim.agents_mgr[i].articulated_agent
            other.base_pos = mn.Vector3(other.base_pos.x, -100.0, other.base_pos.z)
    # Image up = -z (world forward), so the map reads like a floor plan.
    world = mn.Matrix4.look_at(mn.Vector3(cx, eye_y, cz), mn.Vector3(cx, robot[1], cz),
                               mn.Vector3(0, 0, -1))
    node = sensor._sensor_object.node
    node.transformation = node.parent.absolute_transformation().inverted() @ world
    frame = sim.get_sensor_observations()[key][..., :3]
    h, w = frame.shape[:2]
    x0, y0 = int(w * (1 - 1 / zoom) / 2), int(h * (1 - 1 / zoom) / 2)
    frame = np.ascontiguousarray(frame[y0:h - y0, x0:w - x0])
    cam = sensor._sensor_object.render_camera
    proj = cam.projection_matrix @ cam.camera_matrix
    pix = {}
    for k, p in pts.items():
        v = proj.transform_point(mn.Vector3(*map(float, p)))  # NDC, divided by w
        pix[k] = [(v.x + 1) / 2 * w - x0, (1 - v.y) / 2 * h - y0]
    Image.fromarray(frame).save(out / f"topdown_{name}.png")
    (out / f"topdown_{name}.json").write_text(json.dumps(
        {"pixels": pix, "world": {k: p.tolist() for k, p in pts.items()}}, indent=1))
    say(f"saved {out / f'topdown_{name}.png'} eye_y={eye_y:.1f} pixels={pix}")


def drive(config, dataset, conn=None):
    from habitat_llm.agent.env import (
        EnvironmentInterface, register_actions, register_measures, register_sensors)
    from habitat_llm.evaluation import DecentralizedEvaluationRunner
    from habitat_llm.examples.example_utils import execute_skill
    from habitat_llm.rail_planner.replan_trigger import gt_goal_status
    from habitat_llm.rail_planner.subgoal_expanders import (
        SubgoalPlanError, build_scene_state, expand_subgoal)

    register_sensors(config)
    register_actions(config)
    register_measures(config)
    env = EnvironmentInterface(config, dataset=dataset, init_wg=False)
    env.initialize_perception_and_world_graph()
    runner = DecentralizedEvaluationRunner(config.evaluation, env)
    uid = config.robot_agent_uid
    planner = runner.planner[uid]
    planner.reset()
    out = Path(config.paths.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    wg = lambda: env.world_graph[uid]  # noqa: E731

    def scene():
        s = build_scene_state(wg(), uid)
        if hasattr(planner, "_sync_holding_with_gripper"):
            planner._sync_holding_with_gripper(s, wg(), uid)
        s.waited |= getattr(planner, "_waited", set())
        return s

    def run_tool(tool, args):
        responses, info, _ = execute_skill({uid: (tool, args, None)}, planner,
                                           make_video=False, play_video=False)
        say(f"{tool}[{args}] -> {responses[uid]!r} ({info['skill_steps']} steps)")
        return responses[uid]

    def expand(subgoal):
        pred, args = planner._parse_subgoal(subgoal)
        if not pred:
            raise SubgoalPlanError(f"could not parse {subgoal!r}")
        return expand_subgoal(scene(), pred, args)

    say(f"ready: episode {env.env.env.env._env.current_episode.episode_id}, "
        f"planner {type(planner).__name__}, tools {sorted(planner.agents[0].tools)}")
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        say(f"> {line}")
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if cmd == "quit":
                break
            elif cmd == "graph":
                wg().display_hierarchy()
            elif cmd == "instruction":
                say(env.env.env.env._env.current_episode.instruction)
            elif cmd == "state":
                s = scene()
                say(f"agent={s.agent} room={s.room(s.agent)} holding={s.holding}")
                for obj in sorted(s.objects):
                    say(f"  {obj}: {s.placement.get(obj)}")
                say(f"  open={sorted(s.open)} closed={sorted(s.known_closed)}")
            elif cmd == "snap":
                from PIL import Image
                obs = env.sim.get_sensor_observations()
                for key, frame in obs.items():
                    if "rgb" in key:
                        p = out / f"snap_{rest or 'x'}_{key}.png"
                        Image.fromarray(frame[..., :3]).save(p)
                        say(f"saved {p}")
            elif cmd == "topdown":
                topdown(env, uid, out, *rest.split())
            elif cmd == "hint":
                from habitat_llm.rail_planner import placement_oracle
                goals = placement_oracle.placement_goals(env)
                say(placement_oracle.format_hint(goals, set(goals)).strip() or "(no placement goals)")
            elif cmd == "measures":
                ok, pc = gt_goal_status(env)
                say(f"task_state_success={ok} task_percent_complete={pc}")
            elif cmd in ("plan", "achieve"):
                steps = expand(rest)
                tools = [planner._action_to_tool(a) for a in steps]
                for a, t in zip(steps, tools):
                    say(f"  {a.name}{tuple(a.args)} -> {t[0]}[{t[1]}]")
                if not steps:
                    say("  already satisfied")
                if cmd == "achieve":
                    for step, (tool, args) in zip(steps, tools):
                        if planner._response_failed(run_tool(tool, args)):
                            say("step failed; stopping")
                            break
                        planner._on_skill_succeeded(step)
            else:
                run_tool(cmd, rest)
        except Exception as e:  # keep the session alive
            say(f"ERROR {type(e).__name__}: {e}")
    env.env.close()
    say("done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True, help="v3 episode folder or dataset.json")
    p.add_argument("--planner", choices=sorted(PLANNERS), default="rail")
    p.add_argument("--out", required=True, help="results dir (snapshots land here)")
    p.add_argument("overrides", nargs="*", help="extra Hydra overrides")
    a = p.parse_args()
    ep = Path(a.episode).resolve()
    if ep.is_dir():
        ep = ep / "dataset.json.gz"
    out = Path(a.out).resolve()
    # Same overrides as scripts/run_parallel.py; save_video=True keeps the RGB sensors.
    sys.argv = [sys.argv[0], f"--config-name={PLANNERS[a.planner]}",
                "evaluation.agents.agent_0.planner.plan_config.vlm.max_completion_tokens=8192",
                f"habitat.dataset.data_path={ep}",
                "+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/",
                f"paths.results_dir={out}", f"evaluation.output_dir={out}",
                f"hydra.run.dir={out}/hydra", "hydra.job.chdir=False",
                "evaluation.save_video=True", "evaluation.generate_prediviz=False",
                "num_runs_per_episode=1", "num_proc=1", "mode=cli", "device=cpu", *a.overrides]
    from habitat_llm.examples import planner_demo
    planner_demo.run_planner = drive  # replace only the execution stage
    planner_demo.run_eval()


if __name__ == "__main__":
    main()
