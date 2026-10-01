"""Exercise the real worker's timing wrapper without loading Habitat assets."""
import ast
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest


class WorkerTimingTest(unittest.TestCase):
    def test_instrumentation_preserves_steps_frames_and_result(self):
        tree = ast.parse(Path('scripts/ground_truth_worker.py').read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GroundTruthSession')
        calls = []
        class Parent:
            def _do_run_skill(self, **kwargs):
                for i in range(3):
                    self.env_interface.step(i)
                    self.set_frame_rgb(i)
                return {'ok': True, 'response': 'Successful execution!'}
            def set_frame_rgb(self, frame):
                calls.append(('display', frame))
        class Writer:
            def append_data(self, frame):
                calls.append(('record', frame))
            def close(self):
                calls.append(('close',))
        namespace = {'SkillRunnerSession': Parent, 'time': time,
                     'imageio': SimpleNamespace(get_writer=lambda *a, **k: Writer())}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<worker class>', 'exec'), namespace)
        worker = namespace['GroundTruthSession']()
        step = lambda i: calls.append(('step', i))
        worker.env_interface = SimpleNamespace(step=step)
        worker.command_index = 0
        worker._last_evaluation_publish = worker._last_publish = float('inf')
        with tempfile.TemporaryDirectory() as folder:
            worker.results_dir = Path(folder)
            result = worker._do_run_skill(skill='Navigate', target='table_7')
        self.assertIs(worker.env_interface.step, step)
        self.assertEqual(result['skill_steps'], 3)
        self.assertEqual(result['recorded_frames'], 3)
        self.assertTrue(result['ok'])
        self.assertEqual(calls, [item for i in range(3) for item in [('step', i), ('display', i), ('record', i)]] + [('close',)])
        timing = result['timing']
        for key, value in timing.items():
            self.assertGreaterEqual(value, 0, key)
            self.assertLessEqual(value, timing['action_wall_seconds'], key)
        self.assertIsNone(worker._action_timing)


if __name__ == '__main__':
    unittest.main()
