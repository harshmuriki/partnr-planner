"""API-free GPU integration probe for episode object-state reports.

Run like planner_demo with a baseline config and a single-episode dataset.
Uses the real simulator and EvaluationRunner lifecycle with deterministic
boundary mutations instead of an LLM planner. Writes probe_result.json.
"""

import json
from pathlib import Path

from habitat_llm.examples import planner_demo
from habitat_llm.evaluation.evaluation_runner import EvaluationRunner


class ReportProbeRunner(EvaluationRunner):
    def _initialize_planners(self):
        self.planner = {}

    def reset_planners(self):
        pass

    def _log_planner_data(self, planner_infos):
        pass

    def get_low_level_actions(self, instruction, observations, world_graph):
        from habitat.sims.habitat_simulator.object_state_machine import set_state_of_obj
        sim = self.env_interface.sim
        # Deterministic boundary changes, not a claim of planner/skill success.
        if self.mutate:
            node = self.env_interface.perception.gt_graph.get_node_from_name(self.object_name)
            obj = sim.get_rigid_object_manager().get_object_by_handle(node.sim_handle)
            obj.translation = obj.translation + type(obj.translation)(0.15, 0, 0)
            set_state_of_obj(obj, self.state_name, not self.initial_state)
        return {}, {}, True


def probe(config, dataset, conn=None):
    from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_measures, register_sensors
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    env = EnvironmentInterface(config, dataset=dataset)
    try:
        rgb = [v for k, v in env.sim.get_sensor_observations().items() if 'rgb' in k]
        assert rgb and any(float(frame.std()) > 0 for frame in rgb)
        runner = ReportProbeRunner(config.evaluation, env)
        nodes = env.perception.gt_graph.get_all_objects()
        node = next(n for n in nodes if any(isinstance(v, bool) for v in n.properties.get('states', {}).values()))
        runner.object_name = node.name
        runner.state_name, runner.initial_state = next((k, v) for k, v in node.properties['states'].items() if isinstance(v, bool))
        runner.mutate = True
        runner.run_instruction(output_name='report_probe')
        path = Path(runner.output_dir) / 'object_states/report_probe.json'
        report = json.loads(path.read_text())
        pair = report['objects'][node.name]
        assert pair['initial']['states'][runner.state_name] == runner.initial_state
        assert pair['final']['states'][runner.state_name] != runner.initial_state
        assert pair['initial']['location']['position_xyz'] != pair['final']['location']['position_xyz']
        assert {n.name for n in nodes} <= report['objects'].keys()
        assert all(report['objects'][n.name]['initial']['location']['room'] for n in nodes)
        # A continuation must retain the episode's original state.
        runner.mutate = False
        runner.run_instruction(output_name='report_probe', preserve_planner_state=True)
        continued = json.loads(path.read_text())
        assert continued['objects'][node.name]['initial'] == pair['initial']
        # A new instruction/episode must not reuse an earlier initial snapshot.
        runner.run_instruction(output_name='report_probe_next')
        next_report = json.loads(path.with_name('report_probe_next.json').read_text())
        assert next_report['objects'][node.name]['initial'] == pair['final']
        result = {'passed': True, 'rendered_rgb': True, 'llm_calls': 0,
                  'objects': len(nodes), 'reported_entities': len(report['objects']),
                  'changed_object': node.name, 'changed_state': runner.state_name,
                  'planner_target': config.evaluation.agents.agent_0.planner._target_,
                  'report': str(path.resolve())}
        Path(config.paths.results_dir, 'probe_result.json').write_text(json.dumps(result, indent=2) + '\n')
        print('OBJECT STATE REPORT PROBE PASS', result, flush=True)
    finally:
        env.env.close()


if __name__ == '__main__':
    planner_demo.run_planner = probe
    planner_demo.run_eval()
