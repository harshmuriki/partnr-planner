"""Ground-truth sandbox/record/persistence contract tests without a simulator."""
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
import unittest
import uuid

from scripts.ground_truth_archive import file_sha256
from scripts.scenario_ground_truth import EpisodeCatalog, GroundTruthService, GroundTruthStore


class FakeCatalog:
    def get(self, variant):
        return dict(variant=variant, spec_hash='spec', dataset_hash='episode', available=True,
                    dataset='episode.json.gz', episode_id=variant, rooms=['kitchen_0', 'bedroom_0'],
                    targets=['jug_0'], reports=['report absence'] if variant.endswith('ABS') else [],
                    unsupported_criteria=[])


FURNITURE = {'kitchen_0': [{'name': 'cabinet_6', 'description': 'Kitchen cabinet with sink'},
                           {'name': 'fridge_0', 'description': 'Fridge'}],
             'bedroom_0': [{'name': 'chest_of_drawers_0', 'description': 'Nightstand'}]}


class FurnitureCatalog(FakeCatalog):
    def get(self, variant):
        scene = 'scene-b' if variant.startswith('T6') else 'scene-a'
        return {**super().get(variant), 'scene_id': scene, 'furniture': FURNITURE}


class OtherApartmentCatalog(FakeCatalog):
    """A second scene that shares kitchen_0 with FakeCatalog but has its own garage_0."""

    def get(self, variant):
        return {**super().get(variant), 'rooms': ['kitchen_0', 'garage_0']}


def make_clip(path, color, frames=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                    '-i', f'color=c={color}:s=64x64:r=30', '-frames:v', str(frames), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    str(path)], check=True)


class FakeWorker:
    """A deterministic world: the task succeeds once the lamp is powered off since the last load."""

    def __init__(self, path, clips=False):
        self.directory = path
        self.session_dir = path / 'session'
        self.calls = []
        self.clips = clips
        self.done = False
        self.index = 0
        self.crash = False

    def call(self, command):
        self.calls.append(command)
        if self.crash:
            raise RuntimeError('Simulator failed')
        if command['op'] == 'load':
            self.done, self.index = False, 0
            for clip in self.session_dir.glob('clips/*.mp4'):
                clip.unlink()
            return dict(loaded=True, evaluation={'success': False, 'percent_complete': 0})
        ok = command['target'] != 'nowhere'
        if ok and command['skill'] == 'PowerOff':
            self.done = True
        result = {'ok': True, 'skill_steps': 3,
                  'response': 'Successful execution!' if ok else 'Unexpected failure! - no such target'}
        if self.clips:
            rel = f'clips/{self.index:04d}.mp4'
            make_clip(self.session_dir / rel, ['red', 'blue', 'green', 'white'][self.index % 4])
            result['ground_truth_video'] = rel
        self.index += 1
        return dict(loaded=True, evaluation={'success': self.done, 'percent_complete': 1 if self.done else .5},
                    action_result=result)


class GroundTruthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = FakeWorker(self.root)
        self.service = self.make_service()

    def make_service(self, catalog=None):
        return GroundTruthService(self.root/'test.sqlite3', worker=self.worker, catalog=catalog or FakeCatalog(),
                                  results_root=self.root/'archive')

    def idle(self, service=None):
        service = service or self.service
        deadline = time.monotonic() + 20
        while service.lock.locked() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(service.lock.locked())

    def sandbox(self, variant='T1-ACC-BASE'):
        self.service.sandbox(variant)
        self.idle()

    def act(self, skill='Navigate', target='lamp_0', variant='T1-ACC-BASE', request_id=None):
        result = self.service.action(variant, dict(request_id=request_id or uuid.uuid4().hex, skill=skill, target=target))
        self.idle()
        return result

    def record(self, variant='T1-ACC-BASE'):
        self.service.record(variant)
        self.idle()
        return self.service.last_result

    def runs(self):
        db = sqlite3.connect(self.root/'test.sqlite3')
        try:
            return db.execute('SELECT variant,status FROM runs').fetchall(), db.execute('SELECT COUNT(*) FROM actions').fetchone()[0]
        finally:
            db.close()

    def test_sandbox_loads_exact_episode_and_saves_nothing(self):
        self.sandbox()
        self.assertEqual(self.worker.calls[0]['episode_id'], 'T1-ACC-BASE')
        self.act()
        self.act(target='nowhere')
        self.act('PowerOff')
        self.assertEqual(self.runs(), ([], 0))
        self.assertEqual([s['keep'] for s in self.service.steps], [True, False, True])
        state = self.service.state('T1-ACC-BASE')
        self.assertIsNone(state['ground_truth'])
        self.assertEqual(len(state['session']['steps']), 3)
        self.assertEqual(self.service.state('T3-ACC-BASE')['session']['steps'], [])

    def test_action_pack_saves_checked_steps_and_applies_in_another_variant(self):
        self.sandbox()
        self.act()
        self.act(target='nowhere')
        self.act('PowerOff')
        self.service.packs('T1-ACC-BASE', {'save': ' Lamp off '})
        packs = self.service.state('T1-INC-BASE')['packs']
        self.assertEqual([(p['name'], p['source_variant']) for p in packs], [('Lamp off', 'T1-ACC-BASE')])
        self.assertEqual(packs[0]['steps'], [{'skill': 'Navigate', 'target': 'lamp_0'}, {'skill': 'PowerOff', 'target': 'lamp_0'}])
        self.assertEqual(self.service.state('T2-ACC-BASE')['packs'], [])
        with self.assertRaises(ValueError):
            self.service.packs('T1-INC-BASE', {'apply': packs[0]['id']})  # Needs this variant's sandbox.
        self.sandbox('T1-INC-BASE')
        self.act('Explore', 'kitchen_0', variant='T1-INC-BASE')
        self.service.packs('T1-INC-BASE', {'apply': packs[0]['id']})
        self.idle()
        self.assertEqual([(s['skill'], s['keep']) for s in self.service.steps], [('Explore', True), ('Navigate', True), ('PowerOff', True)])
        self.assertTrue(self.service.last_result['ok'])
        self.assertEqual(self.runs(), ([], 0))
        with self.assertRaises(ValueError):
            self.service.packs('T2-ACC-BASE', {'apply': packs[0]['id']})  # Packs belong to one task.

    def test_action_pack_stops_at_first_failure_and_can_be_replaced_or_deleted(self):
        self.sandbox()
        self.act()
        self.service.packs('T1-ACC-BASE', {'save': 'route'})
        self.act(target='nowhere')
        self.service.edit_steps('T1-ACC-BASE', {'id': self.service.steps[-1]['id'], 'keep': True})
        self.act('PowerOff')
        self.service.packs('T1-ACC-BASE', {'save': 'route'})  # Same name replaces the pack.
        pack, = self.service.state('T1-ACC-BASE')['packs']
        self.assertEqual(len(pack['steps']), 3)
        self.sandbox()
        self.service.packs('T1-ACC-BASE', {'apply': pack['id']})
        self.idle()
        self.assertEqual([(s['target'], s['keep']) for s in self.service.steps], [('lamp_0', True), ('nowhere', False)])
        self.assertFalse(self.service.last_result['ok'])
        self.assertIn('step 2 of 3', self.service.last_result['message'])
        self.service.packs('T1-ACC-BASE', {'delete': pack['id']})
        self.assertEqual(self.service.state('T1-ACC-BASE')['packs'], [])
        for payload in ({'save': ''}, {'save': 'x' * 81}, {'delete': 'missing'}, {}):
            with self.assertRaises(ValueError):
                self.service.packs('T1-ACC-BASE', payload)

    def test_action_pack_with_absence_report_needs_a_matching_spec(self):
        self.sandbox('T1-ACC-ABS')
        self.act('ReportAbsence', 'report absence', variant='T1-ACC-ABS')
        self.service.packs('T1-ACC-ABS', {'save': 'report'})
        pack, = self.service.state('T1-ACC-ABS')['packs']
        self.sandbox('T1-ACC-BASE')
        with self.assertRaises(ValueError):
            self.service.packs('T1-ACC-BASE', {'apply': pack['id']})
        self.assertFalse(self.service.lock.locked())

    def test_live_evaluation_is_current_episode_only_and_does_not_save(self):
        self.sandbox()
        original = dict(self.service.snapshot)
        path = self.worker.directory / 'evaluation.json'
        self.service.lock.acquire()
        try:
            live = {'success': True, 'percent_complete': 1}
            path.write_text(json.dumps({'load_id': self.service.load_id, 'evaluation': live}))
            state = self.service.state('T1-ACC-BASE')
            self.assertEqual(state['session']['snapshot']['evaluation'], live)
            self.assertIsNone(state['ground_truth'])
            self.assertEqual(self.service.snapshot, original)
            self.assertEqual(self.service.state('T7-ACC-BASE')['session']['snapshot'], {})
            # A previous episode's completion must never leak into a fresh load.
            path.write_text(json.dumps({'load_id': 'old-load', 'evaluation': live}))
            self.assertEqual(self.service.state('T1-ACC-BASE')['session']['snapshot'], original)
            path.write_text('{invalid')
            self.assertEqual(self.service.state('T1-ACC-BASE')['session']['snapshot'], original)
        finally:
            self.service.lock.release()
        self.assertEqual(self.runs(), ([], 0))

    def test_save_keeps_the_live_run_without_replaying(self):
        self.sandbox()
        self.act()
        self.act(target='nowhere')
        self.act('Pick', 'jug_0')
        self.act('PowerOff')
        self.service.edit_steps('T1-ACC-BASE', {'id': self.service.steps[2]['id'], 'keep': False})
        self.worker.calls.clear()
        self.assertTrue(self.record()['ok'])
        self.assertEqual(self.worker.calls, [])  # No reset and no replay.
        gt = self.service.store.ground_truth('T1-ACC-BASE')
        # Every action that ran is saved, in order, including failed and unchecked ones.
        self.assertEqual([(a['skill'], a['target']) for a in gt['actions']],
                         [('Navigate', 'lamp_0'), ('Navigate', 'nowhere'), ('Pick', 'jug_0'), ('PowerOff', 'lamp_0')])
        self.assertEqual((gt['action_count'], gt['sim_steps']), (4, 12))
        self.assertEqual(self.service.mode, 'sandbox')
        self.assertEqual(len(self.service.steps), 4)
        folder = self.root/'archive/T1/T1-ACC-BASE'
        self.assertEqual(json.loads((folder/'run.json').read_text())['id'], gt['id'])
        self.assertIn('PowerOff', (folder/'actions.csv').read_text())

    def test_new_recording_replaces_the_only_ground_truth(self):
        flagged = []
        self.service.on_rerecord = flagged.append
        self.sandbox()
        self.act('PowerOff')
        self.record()
        self.assertEqual(flagged, [])  # A first recording needs no re-verification.
        first = self.service.store.ground_truth('T1-ACC-BASE')['id']
        self.act('Navigate', 'jug_0')
        self.record()
        second = self.service.store.ground_truth('T1-ACC-BASE')
        self.assertNotEqual(second['id'], first)
        self.assertEqual(second['action_count'], 2)
        self.assertEqual(self.runs(), ([('T1-ACC-BASE', 'completed')], 2))
        self.assertIn('replacing', self.service.last_result['message'])
        self.assertEqual(flagged, ['T1-ACC-BASE'])
        reopened = GroundTruthStore(self.root/'test.sqlite3')
        self.assertIsNone(reopened.get_run(first))
        self.assertEqual(len(reopened.all_ground_truths()), 1)
        with self.assertRaises(sqlite3.IntegrityError):
            with reopened.connect() as db:
                db.execute("INSERT INTO runs(id,variant,status,started_at,spec_hash,dataset_hash,metadata) VALUES('x','T1-ACC-BASE','completed','t','s','d','{}')")

    def test_incomplete_run_cannot_be_saved_and_keeps_previous(self):
        self.sandbox()
        self.act('PowerOff')
        self.record()
        saved = self.service.store.ground_truth('T1-ACC-BASE')['id']
        self.service.sandbox('T1-ACC-BASE')  # Reset: the lamp is on again and the recording restarts.
        self.idle()
        self.assertEqual(self.service.steps, [])
        self.act('Navigate')
        with self.assertRaisesRegex(ValueError, '50%'):
            self.service.record('T1-ACC-BASE')
        self.assertFalse(self.service.lock.locked())
        self.assertEqual(self.service.store.ground_truth('T1-ACC-BASE')['id'], saved)
        self.assertEqual(self.runs(), ([('T1-ACC-BASE', 'completed')], 1))

    def test_archive_failure_while_saving_saves_nothing(self):
        self.sandbox()
        self.act('PowerOff')
        original = self.service._save
        self.service._save = lambda *args: (_ for _ in ()).throw(OSError('disk full'))
        try:
            result = self.record()
        finally:
            self.service._save = original
        self.assertFalse(result['ok'])
        self.assertEqual(self.runs(), ([], 0))
        self.assertEqual(len(self.service.steps), 1)  # The live run is still there to save again.

    def test_steps_cannot_be_removed_and_save_needs_an_action(self):
        self.sandbox()
        with self.assertRaises(ValueError):
            self.service.record('T1-ACC-BASE')
        self.assertFalse(self.service.lock.locked())
        self.act('PowerOff')
        for payload in ({'id': self.service.steps[0]['id'], 'remove': True}, {'clear': True}):
            with self.assertRaises(ValueError):
                self.service.edit_steps('T1-ACC-BASE', payload)
        self.assertEqual(len(self.service.steps), 1)

    def test_actions_need_this_variants_sandbox_and_duplicates_run_once(self):
        with self.assertRaises(ValueError):
            self.act()
        self.sandbox('T3-ACC-BASE')
        with self.assertRaises(ValueError):
            self.act()
        key = uuid.uuid4().hex
        self.act(variant='T3-ACC-BASE', request_id=key)
        self.assertTrue(self.act(variant='T3-ACC-BASE', request_id=key)['duplicate'])
        self.assertEqual(len(self.service.steps), 1)

    def test_absence_report_required_and_counted(self):
        self.sandbox('T1-ACC-ABS')
        self.act('PowerOff', variant='T1-ACC-ABS')
        with self.assertRaisesRegex(ValueError, 'absence report'):
            self.service.record('T1-ACC-ABS')
        self.act('ReportAbsence', 'report absence', variant='T1-ACC-ABS')
        self.assertTrue(self.record('T1-ACC-ABS')['ok'])
        gt = self.service.store.ground_truth('T1-ACC-ABS')
        self.assertEqual([a['skill'] for a in gt['actions']], ['PowerOff', 'ReportAbsence'])
        self.assertEqual(gt['sim_steps'], 3)

    def test_video_is_assembled_checksummed_and_rebuilt_if_missing(self):
        self.worker.clips = True
        self.sandbox()
        self.act()
        self.act('PowerOff')
        self.assertTrue(self.record()['ok'])
        gt = self.service.store.ground_truth('T1-ACC-BASE')
        video = self.root/'archive/T1/T1-ACC-BASE/video.mp4'
        self.assertEqual(gt['artifact_status'], 'ready')
        self.assertEqual((gt['video_bytes'], gt['video_sha256']), (video.stat().st_size, file_sha256(video)))
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-show_entries', 'stream=nb_read_frames,avg_frame_rate,duration', '-of', 'json', str(video)]))
        self.assertEqual(int(probe['streams'][0]['nb_read_frames']), 12)
        self.assertTrue((self.root/'archive/T1/T1-ACC-BASE/clips/0001.mp4').exists())
        stream = probe['streams'][0]
        self.assertEqual(stream['avg_frame_rate'], '30/1')
        self.assertAlmostEqual(float(stream['duration']), 12 / 30, places=5)
        indexed = json.loads((self.root/'archive/index.json').read_text())['ground_truths']
        saved = next(run for run in indexed if run['variant'] == 'T1-ACC-BASE')
        self.assertAlmostEqual(saved['video_duration_sec'], 12 / 30, places=2)
        self.assertAlmostEqual(self.service.state('T1-ACC-BASE')['ground_truth']['video_duration_sec'], 12 / 30, places=2)
        self.assertIn('video_duration_sec', (self.root/'archive/index.csv').read_text().splitlines()[0])
        self.assertIn(gt['video_sha256'][:12], self.service.state('T1-ACC-BASE')['ground_truth']['artifacts']['video'])
        video.unlink()
        self.make_service()
        self.assertTrue(video.exists())
        self.assertEqual(GroundTruthStore(self.root/'test.sqlite3').ground_truth('T1-ACC-BASE')['video_sha256'], file_sha256(video))

    def test_short_action_clips_keep_every_frame_in_order_at_constant_rate(self):
        folder = self.root / 'timing'
        actions = []
        expected = []
        for index, (color, rgb, count) in enumerate([
                ('red', (255, 0, 0), 19), ('blue', (0, 0, 255), 2),
                ('green', (0, 128, 0), 3), ('white', (255, 255, 255), 7)]):
            name = f'{index}.mp4'
            make_clip(folder / name, color, count)
            actions.append({'sequence': index + 1, 'result': {'ground_truth_video': name, 'skill_steps': count}})
            expected.extend([rgb] * count)
        self.service.archive.video(folder, actions)
        video = folder / 'video.mp4'
        stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-show_entries', 'stream=nb_read_frames,avg_frame_rate,duration', '-of', 'json', str(video)]))['streams'][0]
        self.assertEqual(int(stream['nb_read_frames']), len(expected))
        self.assertEqual(stream['avg_frame_rate'], '30/1')
        self.assertAlmostEqual(float(stream['duration']), len(expected) / 30, places=5)
        pixels = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video),
            '-vf', 'scale=1:1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
        self.assertEqual(len(pixels), len(expected) * 3)
        for index, rgb in enumerate(expected):
            self.assertTrue(all(abs(actual - channel) < 10 for actual, channel
                in zip(pixels[index * 3:index * 3 + 3], rgb)), f'Frame {index} changed order')

    def test_missing_video_keeps_the_saved_actions(self):
        self.sandbox()
        self.act('PowerOff')
        self.assertTrue(self.record()['ok'])
        gt = self.service.store.ground_truth('T1-ACC-BASE')
        self.assertEqual(gt['artifact_status'], 'error')  # Fake worker made no clip or initial frame.
        self.assertEqual(gt['action_count'], 1)

    def test_restart_drops_unfinished_recording(self):
        meta = FakeCatalog().get('T1-ACC-BASE')
        self.service.store.new_recording(meta)
        staging = self.service.archive.staging('T1-ACC-BASE')
        staging.mkdir(parents=True)
        self.make_service()
        self.assertEqual(self.runs(), ([], 0))
        self.assertFalse(staging.exists())

    def test_legacy_history_is_pruned_to_the_newest_completed_run_and_flattened(self):
        path = self.root/'legacy.sqlite3'
        store = GroundTruthStore(path)
        with store.connect() as db:
            db.execute('DROP INDEX one_ground_truth')
            for run_id, status, completed in [('a'*32, 'completed', '2026-09-16T10'), ('b'*32, 'completed', '2026-09-16T11'),
                                              ('c'*32, 'interrupted', None)]:
                db.execute('INSERT INTO runs(id,variant,status,started_at,completed_at,spec_hash,dataset_hash,metadata) VALUES(?,?,?,?,?,?,?,?)',
                           (run_id, 'T1-ACC-BASE', status, 't', completed, 'spec', 'episode', json.dumps(FakeCatalog().get('T1-ACC-BASE'))))
                db.execute('INSERT INTO actions VALUES(?,?,?,?,?,?,?)', (uuid.uuid4().hex, run_id, 1, 'Navigate', 'x', 't',
                           json.dumps({'skill_steps': 1, 'ok': True})))
        variant_dir = self.root/'legacy_archive/T1/T1-ACC-BASE'
        for run_id in ('a'*32, 'b'*32, 'c'*32):
            (variant_dir/run_id).mkdir(parents=True)
            (variant_dir/run_id/'run.json').write_text(json.dumps({'id': run_id}))
        (variant_dir/'latest.json').write_text('{}')
        (variant_dir/'room_choices.json').write_text('{}')
        service = GroundTruthService(path, worker=self.worker, catalog=FakeCatalog(), results_root=self.root/'legacy_archive')
        self.assertEqual([r['id'] for r in service.store.all_ground_truths()], ['b'*32])
        with service.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM actions').fetchone()[0], 1)
        self.assertEqual(sorted(p.name for p in variant_dir.iterdir()), ['actions.csv', 'furniture_choices.json', 'room_choices.json', 'run.json'])
        self.assertEqual(json.loads((variant_dir/'run.json').read_text())['id'], 'b'*32)
        self.assertEqual(json.loads((self.root/'legacy_archive/index.json').read_text())['ground_truths'][0]['id'], 'b'*32)

    def test_legacy_database_migrates_counts_without_losing_results(self):
        path = self.root/'legacy.sqlite3'
        db = sqlite3.connect(path)
        try:
            with db:
                db.executescript('''CREATE TABLE runs (
                    id TEXT PRIMARY KEY, variant TEXT, status TEXT, started_at TEXT,
                    completed_at TEXT, action_count INTEGER, spec_hash TEXT, dataset_hash TEXT,
                    metadata TEXT, snapshot TEXT, error TEXT);
                    CREATE TABLE actions (request_id TEXT PRIMARY KEY, run_id TEXT, sequence INTEGER,
                    skill TEXT, target TEXT, started_at TEXT, result TEXT);''')
                db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    ('old-run', 'T1-ACC-BASE', 'completed', 'start', 'end', 2, 'spec', 'dataset', '{}', '{}', None))
                for i, result in enumerate([{'skill_steps': 12, 'ok': True}, {'skill_steps': 0, 'ok': False}]):
                    db.execute('INSERT INTO actions VALUES(?,?,?,?,?,?,?)',
                        (str(i), 'old-run', i, 'Navigate', 'jug_0', 'start', json.dumps(result)))
        finally:
            db.close()
        store = GroundTruthStore(path)
        run = store.get_run('old-run')
        self.assertEqual(run['sim_steps'], 12)
        self.assertFalse(run['steps_complete'])
        self.assertEqual(run['status'], 'completed')
        self.assertEqual(len(run['actions']), 2)
        self.assertEqual(GroundTruthStore(path).get_run('old-run'), run)

    def test_room_choices_are_shared_by_object_and_survive_restart(self):
        self.service.rooms('T1-ACC-BASE', {'object': 'jug_0', 'rooms': ['bedroom_0', 'kitchen_0']})
        self.assertEqual(self.service.state('T3-ACC-BASE')['room_choices']['jug_0'], ['bedroom_0', 'kitchen_0'])
        self.assertEqual(GroundTruthStore(self.root/'test.sqlite3').object_rooms()['jug_0'], ['bedroom_0', 'kitchen_0'])
        exported = self.root/'archive/T1/T1-ACC-BASE/room_choices.json'
        self.assertEqual(json.loads(exported.read_text())['jug_0'], ['bedroom_0', 'kitchen_0'])
        with self.assertRaises(ValueError):
            self.service.rooms('T1-ACC-BASE', {'object': 'jug_0', 'rooms': ['imaginary_room']})
        self.service.rooms('T1-ACC-BASE', {'object': 'jug_0', 'rooms': []})
        self.assertNotIn('jug_0', self.service.state('T1-ACC-BASE')['room_choices'])

    def test_rooms_frozen_in_saved_run_and_kept_when_replaced(self):
        self.service.rooms('T1-ACC-BASE', {'object': 'jug_0', 'rooms': ['bedroom_0']})
        self.sandbox()
        self.act('PowerOff')
        self.record()
        self.service.rooms('T1-ACC-BASE', {'object': 'jug_0', 'rooms': ['kitchen_0']})
        folder = self.root/'archive/T1/T1-ACC-BASE'
        self.assertEqual(json.loads((folder/'run.json').read_text())['room_choices']['jug_0'], ['bedroom_0'])
        self.assertEqual(json.loads((folder/'room_choices.json').read_text())['jug_0'], ['kitchen_0'])
        self.record()
        self.assertEqual(json.loads((folder/'run.json').read_text())['room_choices']['jug_0'], ['kitchen_0'])
        self.assertEqual(json.loads((folder/'room_choices.json').read_text())['jug_0'], ['kitchen_0'])

    def test_other_apartments_keep_their_rooms_and_show_only_their_own(self):
        other = self.make_service(OtherApartmentCatalog())
        self.service.rooms('T1-ACC-BASE', {'object': 'jug_0', 'rooms': ['kitchen_0', 'bedroom_0']})
        other.rooms('T6-ACC-BASE', {'object': 'jug_0', 'rooms': ['garage_0']})
        self.assertEqual(self.service.state('T1-ACC-BASE')['room_choices']['jug_0'], ['bedroom_0'])
        self.assertEqual(other.state('T6-ACC-BASE')['room_choices']['jug_0'], ['garage_0'])
        self.assertEqual(self.service.store.object_rooms()['jug_0'], ['garage_0', 'bedroom_0'])

    def test_assumption_notes_per_task_and_all_tasks_persist_and_export(self):
        self.service.save_notes('T1-ACC-BASE', {'scope': 'task', 'text': 'Jug counts as full after one Fill.'})
        self.service.save_notes('T1-INC-SUB', {'scope': 'all', 'text': 'Only the bedroom lamp counts.'})
        state = self.service.state('T1-ACC-BASE')
        self.assertEqual(state['notes']['task']['text'], 'Jug counts as full after one Fill.')
        self.assertEqual(state['notes']['all']['text'], 'Only the bedroom lamp counts.')
        self.assertEqual(self.service.state('T1-ACC-ABS')['notes']['task']['text'], 'Jug counts as full after one Fill.')
        self.assertIsNone(self.service.state('T3-ACC-BASE')['notes']['task'])
        self.assertEqual(self.service.state('T3-ACC-BASE')['notes']['all']['text'], 'Only the bedroom lamp counts.')
        store = GroundTruthStore(self.root/'test.sqlite3')  # Reopening must not fold task notes into All tasks.
        self.assertEqual(sorted(store.notes()), ['ALL', 'T1'])
        exported = (self.root/'archive/assumptions.md').read_text()
        self.assertLess(exported.index('## All tasks (T1-T7)'), exported.index('## T1 (every variant)'))
        self.service.save_notes('T1-INC-SUB', {'scope': 'task', 'text': '  '})
        self.assertIsNone(self.service.state('T1-ACC-BASE')['notes']['task'])
        for payload in ({'scope': 'variant', 'text': 'x'}, {'scope': 'all', 'text': 'x' * 20001}, {'scope': 'all'}):
            with self.assertRaises(ValueError):
                self.service.save_notes('T1-ACC-BASE', payload)

    def test_likely_furniture_ranked_shared_within_apartment_and_exported(self):
        service = self.make_service(FurnitureCatalog())
        service.furniture('T1-ACC-CON', {'object': 'jug_0', 'furniture': ['fridge_0', 'cabinet_6']})
        self.assertEqual(service.state('T1-ACC-BASE')['furniture_choices']['jug_0'], ['fridge_0', 'cabinet_6'])
        self.assertNotIn('jug_0', service.state('T6-ACC-CON')['furniture_choices'])  # Different apartment.
        self.assertEqual(GroundTruthStore(self.root/'test.sqlite3').object_furniture('scene-a')['jug_0'], ['fridge_0', 'cabinet_6'])
        exported = self.root/'archive/T1/T1-ACC-CON/furniture_choices.json'
        self.assertEqual(json.loads(exported.read_text())['jug_0'], ['fridge_0', 'cabinet_6'])
        for chosen in (['imaginary_0'], ['fridge_0', 'fridge_0'], 'fridge_0'):
            with self.assertRaises(ValueError):
                service.furniture('T1-ACC-CON', {'object': 'jug_0', 'furniture': chosen})
        with self.assertRaises(ValueError):
            service.furniture('T1-ACC-CON', {'object': 'lamp_9', 'furniture': []})
        service.furniture('T1-ACC-CON', {'object': 'jug_0', 'furniture': []})
        self.assertNotIn('jug_0', service.state('T1-ACC-CON')['furniture_choices'])

    def test_furniture_choices_frozen_in_saved_run_and_kept_when_replaced(self):
        self.service = self.make_service(FurnitureCatalog())
        self.service.furniture('T1-ACC-CON', {'object': 'jug_0', 'furniture': ['fridge_0']})
        self.sandbox('T1-ACC-CON')
        self.act('PowerOff', variant='T1-ACC-CON')
        self.record('T1-ACC-CON')
        folder = self.root/'archive/T1/T1-ACC-CON'
        self.service.furniture('T1-ACC-CON', {'object': 'jug_0', 'furniture': ['cabinet_6']})
        self.assertEqual(json.loads((folder/'run.json').read_text())['furniture_choices']['jug_0'], ['fridge_0'])
        self.assertEqual(json.loads((folder/'furniture_choices.json').read_text())['jug_0'], ['cabinet_6'])
        self.record('T1-ACC-CON')
        self.assertEqual(json.loads((folder/'furniture_choices.json').read_text())['jug_0'], ['cabinet_6'])
        self.assertEqual(json.loads((folder/'run.json').read_text())['furniture_choices']['jug_0'], ['cabinet_6'])

    def test_real_catalog_furniture_uses_spec_names(self):
        meta = EpisodeCatalog().get('T2-ACC-CON')
        names = [e['name'] for entries in meta['furniture'].values() for e in entries]
        self.assertIn('table_16', names)
        self.assertNotIn('table_2', names)  # Runtime alias of table_16.
        self.assertTrue(meta['scene_id'])
        self.assertTrue(all(e['description'] for entries in EpisodeCatalog().get('T7-ACC-BASE')['furniture'].values() for e in entries))

    def test_all_variant_catalogs_have_all_rooms_and_absent_target(self):
        catalog = EpisodeCatalog()
        index = json.loads((catalog.base/'generation/episodes.json').read_text())['variants']
        for variant in index:
            with self.subTest(variant=variant):
                meta = catalog.get(variant)
                self.assertTrue(meta['rooms'])
                self.assertTrue(meta['targets'])
        self.assertIn('jug_0', catalog.get('T1-ACC-ABS')['targets'])
        self.assertIn('closet_0', catalog.get('T1-ACC-ABS')['rooms'])

    def test_failed_attempt_steps_and_unknown_counts(self):
        total, complete = GroundTruthStore.step_totals([
            {'result': {'skill_steps': 7, 'ok': False, 'steps_source': 'environment_step_calls'}},
            {'result': {'skill_steps': 4, 'ok': True}},
        ])
        self.assertEqual((total, complete), (11, True))
        self.assertEqual(GroundTruthStore.step_totals([{'result': None}]), (0, False))


if __name__ == '__main__':
    unittest.main()
