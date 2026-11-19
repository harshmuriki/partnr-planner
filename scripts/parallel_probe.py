"""API-free parallel capacity probe using the full episode environment."""

import json
import os
from pathlib import Path
import resource
import time

from habitat_llm.examples import planner_demo


def probe(config, dataset, conn=None):
    from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_measures, register_sensors
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    env = EnvironmentInterface(config, dataset=dataset, init_wg=False)
    try:
        env.initialize_perception_and_world_graph()
        obs = env.sim.get_sensor_observations()
        rgb = [value for name, value in obs.items() if "rgb" in name]
        assert rgb and any(float(frame.std()) > 0 for frame in rgb), "No rendered RGB observations"
        # This uses the same solver wrapper and relative temp directory as VLM-TAMP.
        from habitat_llm.pddlstream.solve import solve_pddlstream_problem
        from pddlstream.language.constants import PDDLProblem
        domain = """(define (domain probe) (:requirements :strips)
          (:predicates (ready) (done))
          (:action finish :parameters () :precondition (ready)
            :effect (and (done) (not (ready)))))"""
        plan, cost, _ = solve_pddlstream_problem(PDDLProblem(domain, {}, None, {}, [("ready",)], ("done",)))
        assert plan is not None and len(plan) == 1
        barrier = Path(os.environ["PARTNR_PROBE_DIR"])
        barrier.mkdir(parents=True, exist_ok=True)
        (barrier / os.environ["PARTNR_PROBE_SLOT"]).write_text("ready")
        deadline = time.monotonic() + 180
        while len(list(barrier.iterdir())) < int(os.environ["PARTNR_PROBE_WORKERS"]):
            if time.monotonic() > deadline:
                raise TimeoutError("Other probe workers did not become ready")
            time.sleep(1)
        # Keep all full environments alive while the coordinator samples GPU usage.
        time.sleep(20)
        result = {"render_pass": True, "pddl_pass": True,
                  "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "pid": os.getpid(), "cwd": os.getcwd(), "rgb_sensors": len(rgb)}
        Path(config.paths.results_dir, "probe.json").write_text(json.dumps(result, indent=2))
        print("PARALLEL PROBE PASS", result, flush=True)
    finally:
        env.env.close()


if __name__ == "__main__":
    # Reuse dataset/config initialization, substituting only the execution stage.
    # No evaluation runner or VLM is constructed by the probe.
    planner_demo.run_planner = probe
    planner_demo.run_eval()
