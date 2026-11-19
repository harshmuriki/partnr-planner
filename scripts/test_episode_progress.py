"""Timeout regression tests with real SIGTERM/SIGKILL, no simulator or APIs."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'habitat_llm/utils/episode_progress.py'
spec = importlib.util.spec_from_file_location('progress', MODULE_PATH)
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_repeated_ticks_retry_and_continued_subgoals(self):
        writer = progress.EpisodeProgress(self.root, 'episode', 'first')
        for _ in range(3):
            writer.start_action(0, 'Pick', 'cup')
            writer.finish_action(0, '')
            writer.simulator_step({'task_percent_complete': .2})
        writer.finish_action(0, 'failed')
        writer.start_action(0, 'Pick', 'cup')
        writer.finish_action(0, 'success')
        writer.record({'high_level_actions': {0: ['Pick', 'cup', None]}, 'responses': {0: 'success'},
                       'traces': {0: 'Task: first\nPick[cup]'}, 'stats': {'task_state_success': 0}})
        writer.finish('completed')
        writer.continue_instruction('second')
        writer.record({'high_level_actions': {0: ['Done', None, None]}})
        writer.finish('completed')
        document = json.loads(writer.path.read_text())
        self.assertEqual(len(document['actions']), 2)
        self.assertEqual([a['response'] for a in document['actions']], ['failed', 'success'])
        self.assertEqual([s['log_index'] for s in document['steps']], [0, 1])
        self.assertEqual(document['subgoal_instructions'], ['first', 'second'])
        self.assertTrue(document['complete'])
        self.assertTrue((self.root / 'traces/0/trace-episode-0.txt').exists())
        fresh = progress.EpisodeProgress(self.root, 'episode', 'new run')
        fresh.finish('completed')
        recovered = progress.recover_episode(fresh.journal)
        self.assertEqual(recovered['steps'], [])
        self.assertEqual(recovered['actions'], [])

    def run_killed(self, signum):
        script = '''
import importlib.util, pathlib, time
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('progress', %r)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class Runner:
    def __init__(self):
        self.env_interface = SimpleNamespace(_episode_progress=None)
    @m.checkpoint_episode
    def run(self):
        p = m.EpisodeProgress(%r, 'episode', 'test task')
        self.env_interface._episode_progress = p
        p.start_action(0, 'Navigate', 'table')
        p.simulator_step({'task_state_success':0, 'task_percent_complete':0.5})
        p.finish_action(0, 'success')
        p.record({'high_level_actions':{0:['Navigate','table',None]}, 'responses':{0:'success'}, 'stats':{'task_state_success':0, 'task_percent_complete':0.5}, 'traces':{0:'Task: test task\\nNavigate[table]'}})
        p.start_action(0, 'Explore', 'kitchen')
        p.simulator_step({'task_state_success':0, 'task_percent_complete':0.75})
        p.record({'high_level_actions':{0:['Explore','kitchen',None]}, 'responses':{0:''}})
        # Same signature avoids full-log rewrites; this latest step is journal-only.
        p.record({'high_level_actions':{0:['Explore','kitchen',None]}, 'responses':{0:''}, 'sim_step_count':2})
        print('READY', flush=True)
        time.sleep(60)
Runner().run()
''' % (str(MODULE_PATH), str(self.root / 'dataset'))
        child = subprocess.Popen([sys.executable, '-u', '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            import select
            self.assertTrue(select.select([child.stdout], [], [], 10)[0], 'Worker did not initialize')
            self.assertEqual(child.stdout.readline().strip(), 'READY')
            # Files exist before termination, independent of cleanup/signal handlers.
            saved = json.loads((self.root / 'dataset/actions/actions-episode.json').read_text())
            self.assertEqual(len(saved['actions']), 2)
            child.send_signal(signum)
            child.communicate(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
        journal = self.root / 'dataset/planner-log/planner-log-episode.jsonl'
        # A partial filesystem write must not discard preceding valid events.
        with journal.open('a') as stream:
            stream.write('{"event": "step", "step":')
        recovered = progress.recover_run(self.root, 'timeout')
        self.assertEqual(len(recovered), 1)
        document = json.loads(Path(recovered[0]).read_text())
        self.assertEqual(document['status'], 'timeout')
        self.assertFalse(document['complete'])
        self.assertEqual(len(document['steps']), 3)
        self.assertEqual(document['steps'][-1]['sim_step_count'], 2)
        self.assertEqual(document['actions'][0]['response'], 'success')
        self.assertEqual(document['actions'][1]['status'], 'incomplete')
        self.assertIsNone(document['actions'][1]['response'])
        self.assertEqual(document['latest_simulator_checkpoint']['stats']['task_percent_complete'], .75)
        self.assertTrue((self.root / 'dataset/traces/0/trace-episode-0.txt').exists())
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.run_parallel import result_metrics
        metrics = result_metrics(self.root)
        self.assertTrue(metrics['metrics_are_partial'])
        self.assertEqual(metrics['task_percent_complete'], .75)
        self.assertEqual(metrics['recorded_actions'], 2)

    def test_sigterm_retains_actions(self):
        self.run_killed(signal.SIGTERM)

    def test_sigkill_retains_actions(self):
        self.run_killed(signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
