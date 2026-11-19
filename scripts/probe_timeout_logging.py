"""API-free real-simulator worker for forced-timeout logging verification.

Runs a short Wait, then blocks inside a second skill after two internal simulator
steps. A controller terminates this process after timeout_probe_ready.json appears.
"""
import json
from pathlib import Path
import time

from habitat_llm.examples import planner_demo
from habitat_llm.evaluation.evaluation_runner import EvaluationRunner
from habitat_llm.planner.planner import Planner


class TimeoutProbeRunner(EvaluationRunner):
    def _initialize_planners(self):
        self.planner = {}
        self.driver = Planner(None, self.env_interface)
        self.driver.agents = list(self.agents.values())
        self.first = True
        self.phase = 0
        self.trace = ''

    def reset_planners(self):
        pass

    def get_low_level_actions(self, instruction, observations, world_graph):
        action = {0: ('Wait', '', None)}
        low, responses = self.driver.process_high_level_actions(action, observations)
        replanned = self.first
        self.first = False
        response = responses.get(0, '')
        if response and self.phase == 0:
            self.phase = 1
            self.first = True
            self.trace = f'Task: {instruction}\nThought: deterministic probe\nWait[]\nResult: {response}\n'
            tool = self.agents[0].get_tool_from_name('Wait')
            original = tool.process_high_level_action
            def blocking_skill(action_input, obs):
                low_action, _ = original(action_input, obs)
                assert low_action is not None
                for _ in range(2):
                    self.env_interface.step({0: low_action})
                marker = Path(self.output_dir).parent / 'timeout_probe_ready.json'
                marker.write_text(json.dumps({'phase': 'blocked_inside_second_action', 'llm_calls': 0}))
                time.sleep(120)
                raise AssertionError('Controller failed to terminate probe')
            tool.process_high_level_action = blocking_skill
        return low, {'high_level_actions': action, 'responses': responses,
                     'replanned': {0: replanned}, 'replan_required': {0: bool(response)},
                     'agent_states': {0: 'Waiting'}, 'traces': {0: self.trace}}, False


def probe(config, dataset, conn=None):
    from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_measures, register_sensors
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    env = EnvironmentInterface(config, dataset=dataset)
    try:
        runner = TimeoutProbeRunner(config.evaluation, env)
        runner.run_instruction(output_name='timeout_probe')
    finally:
        env.env.close()


if __name__ == '__main__':
    planner_demo.run_planner = probe
    planner_demo.run_eval()
