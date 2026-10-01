"""Concurrent recording isolation, lifecycle, persistence, and HTTP routing."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

from scripts.scenario_ground_truth import GroundTruthSessions
from scripts.serve_scenario_viewer import ViewerHandler
from scripts.test_scenario_ground_truth import FakeCatalog, FakeWorker

A, B, C = 'T1-ACC-BASE', 'T1-INC-BASE', 'T1-OUT-BASE'


class Worker(FakeWorker):
    def __init__(self, path):
        path.mkdir()
        super().__init__(path, clips=True)
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.closed = False

    def call(self, command):
        if command['op'] == 'skill':
            self.started.set()
            if not self.release.wait(10):
                raise RuntimeError('Test worker was not released')
        return super().call(command)

    def close(self):
        self.closed = True

    def frame(self):
        return self.directory.name.encode()


class SessionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workers = []
        def factory():
            worker = Worker(self.root / f'worker-{len(self.workers)}')
            self.workers.append(worker)
            return worker
        self.pool = GroundTruthSessions(worker_factory=factory, database=self.root / 'test.sqlite3',
                                        results_root=self.root / 'archive', catalog=FakeCatalog())
        self.addCleanup(self.pool.close)

    def idle(self, variant):
        service = self.pool.sessions[variant][1]
        deadline = time.monotonic() + 20
        while service.lock.locked() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(service.lock.locked())
        return service

    def open(self, variant):
        result = self.pool.dispatch(variant, 'sandbox', {})
        self.idle(variant)
        return result['session_id']

    def act(self, variant, target='lamp_0'):
        return self.pool.dispatch(variant, 'action', dict(skill='PowerOff', target=target,
                                  request_id=uuid.uuid4().hex, session_id=self.pool.sessions[variant][0]))

    def test_overlapping_actions_keep_frames_evaluation_and_steps_separate(self):
        aid, bid = self.open(A), self.open(B)
        a, b = self.pool.sessions[A][1], self.pool.sessions[B][1]
        a.worker.release.clear()
        b.worker.release.clear()
        self.addCleanup(a.worker.release.set)
        self.addCleanup(b.worker.release.set)
        self.act(A)
        self.act(B, 'nowhere')
        self.assertTrue(a.worker.started.wait(2))
        self.assertTrue(b.worker.started.wait(2))
        self.assertTrue(self.pool.state(A)['session']['busy'])
        self.assertTrue(self.pool.state(B)['session']['busy'])
        self.assertNotEqual(self.pool.frame(A, aid), self.pool.frame(B, bid))
        with self.assertRaisesRegex(ValueError, 'finish'):
            self.pool.dispatch(A, 'close', {'session_id': aid})
        a.worker.release.set()
        b.worker.release.set()
        self.idle(A); self.idle(B)
        self.assertTrue(self.pool.state(A)['session']['snapshot']['evaluation']['success'])
        self.assertFalse(self.pool.state(B)['session']['snapshot']['evaluation']['success'])
        self.assertEqual([s['target'] for s in a.steps], ['lamp_0'])
        self.assertEqual([s['target'] for s in b.steps], ['nowhere'])
        self.assertIsNone(self.pool.state(C)['session']['variant'])

    def test_capacity_close_and_reset_generation(self):
        aid = self.open(A)
        self.open(B)
        with self.assertRaisesRegex(ValueError, 'slots'):
            self.open(C)
        with self.assertRaisesRegex(ValueError, 'changed or closed'):
            self.pool.dispatch(B, 'close', {'session_id': aid})
        new = self.pool.dispatch(A, 'sandbox', {'session_id': aid})['session_id']
        self.assertNotEqual(aid, new)
        self.idle(A)
        with self.assertRaisesRegex(ValueError, 'changed or closed'):
            self.pool.dispatch(A, 'sandbox', {'session_id': aid})
        with self.assertRaises(ValueError):
            self.pool.frame(A, aid)
        worker = self.pool.sessions[A][1].worker
        self.pool.dispatch(A, 'close', {'session_id': new})
        self.assertTrue(worker.closed)
        self.open(C)
        self.assertEqual(set(self.pool.sessions), {B, C})
        with self.assertRaises(ValueError):
            self.pool.dispatch(A, 'action', {'skill':'Navigate', 'target':'lamp_0', 'request_id':uuid.uuid4().hex})

    def test_concurrent_allocation_never_exceeds_capacity(self):
        def attempt(variant):
            try:
                return self.pool.dispatch(variant, 'sandbox', {})
            except ValueError:
                return None
        with ThreadPoolExecutor(3) as executor:
            results = list(executor.map(attempt, [A, B, C]))
        self.assertEqual(sum(item is not None for item in results), 2)
        for variant in self.pool.sessions:
            self.idle(variant)

    def test_new_session_does_not_repeat_startup_cleanup(self):
        self.open(A)
        store = self.pool.shared.store
        pending = store.new_recording(FakeCatalog().get(A))
        staging = self.pool.shared.archive.staging(A)
        staging.mkdir(parents=True)
        (staging / 'marker').write_text('in progress')
        with patch('scripts.scenario_ground_truth.GroundTruthStore', side_effect=AssertionError('reopened database')):
            self.open(B)
        self.assertIsNotNone(store.get_run(pending))
        self.assertEqual((staging / 'marker').read_text(), 'in progress')
        self.assertIs(self.pool.sessions[A][1].archive, self.pool.sessions[B][1].archive)

    def test_two_saves_preserve_separate_archives_without_replay(self):
        self.open(A); self.open(B)
        self.act(A); self.act(B, 'lamp_1')
        a, b = self.idle(A), self.idle(B)
        before = [list(item.worker.calls) for item in (a, b)]
        self.pool.dispatch(A, 'record', {})
        self.pool.dispatch(B, 'record', {})
        self.idle(A); self.idle(B)
        self.assertTrue(a.last_result['ok'], a.last_result)
        self.assertTrue(b.last_result['ok'], b.last_result)
        self.assertEqual(before, [a.worker.calls, b.worker.calls])
        for variant, target in ((A, 'lamp_0'), (B, 'lamp_1')):
            run = self.pool.shared.store.ground_truth(variant)
            self.assertEqual(run['artifact_status'], 'ready')
            self.assertEqual([s['target'] for s in run['actions']], [target])
            folder = self.pool.shared.archive.directory(variant)
            self.assertEqual(json.loads((folder / 'run.json').read_text())['id'], run['id'])
            self.assertTrue((folder / 'video.mp4').stat().st_size)
        index = json.loads((self.root / 'archive/index.json').read_text())
        self.assertEqual({run['variant'] for run in index['ground_truths']}, {A, B})
        self.pool.dispatch(A, 'close', {})
        self.assertIsNotNone(self.pool.state(A)['ground_truth'])
        self.assertEqual(self.pool.state(B)['session']['steps'][0]['target'], 'lamp_1')

    def test_http_routes_sessions_and_rejects_stale_frame_and_action(self):
        class QuietHandler(ViewerHandler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, store=None, ground_truth=self.pool))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def stop():
            server.shutdown(); server.server_close(); thread.join()
        self.addCleanup(stop)
        base = f'http://127.0.0.1:{server.server_port}/baseline_evaluation_v3/api/ground-truth/'
        def post(path, payload):
            request = Request(base + path, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
            with urlopen(request) as response:
                return json.load(response)
        aid = post(A + '/sandbox', {})['session_id']; self.idle(A)
        post(B + '/sandbox', {}); self.idle(B)
        with urlopen(base + A) as response:
            state = json.load(response)
        self.assertEqual(state['session']['id'], aid)
        self.assertEqual(len(state['sessions']), 2)
        with urlopen(base + A + '/frame?session_id=' + aid) as response:
            self.assertEqual(response.read(), self.pool.frame(A, aid))
        for path in (B + '/frame?session_id=' + aid,):
            with self.assertRaises(HTTPError) as error:
                urlopen(base + path)
            self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError):
            post(B + '/action', {'session_id':aid, 'request_id':uuid.uuid4().hex, 'skill':'Navigate', 'target':'lamp_0'})
        post(A + '/close', {'session_id':aid})
        post(C + '/sandbox', {}); self.idle(C)


if __name__ == '__main__':
    unittest.main()
