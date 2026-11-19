"""CPU-only checks for report serialization and unknown/held locations."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    'object_state_report', Path(__file__).resolve().parents[1] / 'habitat_llm/utils/object_state_report.py'
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Node:
    def __init__(self, name, states=None):
        self.name = name
        self.sim_handle = name + '_handle'
        self.properties = {'translation': [1, 2, 3], 'states': states or {}}


class Graph:
    def __init__(self):
        self.obj = Node('cup', {'is_clean': False})
        self.hidden = Node('hidden_object')
        self.agent = Node('agent_0')
        self.room = Node('kitchen')
        self.table = Node('table', {'is_powered_on': False})
        self.graph = {self.obj: {self.agent: 'on'}, self.agent: {self.room: 'inside'},
                      self.table: {self.room: 'inside'}, self.hidden: {}, self.room: {}}

    def get_all_objects(self):
        return [self.obj, self.hidden]

    def get_all_furnitures(self):
        return [self.table]

    def get_agents(self):
        return [self.agent]

    def find_furniture_for_object(self, obj):
        return None

    def get_room_for_entity(self, node):
        if self.room in self.graph[node]:
            return self.room
        raise ValueError('No room')


class ObjectReportTests(unittest.TestCase):
    def test_held_unobserved_unknown_and_snapshot_independence(self):
        graph = Graph()
        env = SimpleNamespace(perception=SimpleNamespace(get_recent_graph=lambda: graph))
        initial = report.snapshot_objects(env)
        graph.obj.properties['states']['is_clean'] = True
        graph.obj.properties['translation'][0] = 5
        final = report.snapshot_objects(env)
        self.assertEqual(initial['cup']['location']['held_by'], 'agent_0')
        self.assertEqual(initial['cup']['location']['room'], 'kitchen')
        self.assertIsNone(initial['hidden_object']['location']['room'])
        self.assertEqual(initial['hidden_object']['states'], {})
        self.assertFalse(initial['cup']['states']['is_clean'])
        self.assertTrue(final['cup']['states']['is_clean'])
        self.assertEqual(initial['cup']['location']['position_xyz'], [1, 2, 3])
        self.assertEqual(initial['table']['kind'], 'furniture')

    def test_receptacle_joint_resolves_supporting_furniture_room(self):
        graph = Graph()
        receptacle = Node('receptacle')
        graph.graph[graph.obj] = {receptacle: 'on'}
        graph.graph[receptacle] = {graph.table: 'joint'}
        graph.graph[graph.table][receptacle] = 'joint'
        env = SimpleNamespace(perception=SimpleNamespace(get_recent_graph=lambda: graph))
        snapshot = report.snapshot_objects(env)
        self.assertEqual(snapshot['cup']['location']['room'], 'kitchen')

    def test_initial_and_cutoff_report_status(self):
        with TemporaryDirectory() as directory:
            initial = {'cup': {'states': {'is_clean': False}}}
            path = report.write_object_state_report(directory, 'episode', 'task', initial, {}, status='running')
            data = json.loads(path.read_text())
            self.assertFalse(data['complete'])
            self.assertFalse(data['final_snapshot_available'])
            self.assertIsNone(data['objects']['cup']['final'])
            final = {'cup': {'states': {'is_clean': True}}}
            report.write_object_state_report(directory, 'episode', 'task', initial, final, status='terminated')
            data = json.loads(path.read_text())
            self.assertFalse(data['complete'])
            self.assertTrue(data['final_snapshot_available'])
            self.assertFalse(data['objects']['cup']['initial']['states']['is_clean'])
            self.assertTrue(data['objects']['cup']['final']['states']['is_clean'])

    def test_missing_entities_and_separate_episode_files(self):
        with TemporaryDirectory() as directory:
            first = report.write_object_state_report(directory, 'episode_1_0', 'task',
                                                     {'removed': {'states': {}}}, {'added': {'states': {}}})
            second = report.write_object_state_report(directory, 'episode_1_1', 'task', {}, {})
            data = json.loads(first.read_text())
            self.assertIsNone(data['objects']['removed']['final'])
            self.assertIsNone(data['objects']['added']['initial'])
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists() and second.exists())
            self.assertFalse(list(first.parent.glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
