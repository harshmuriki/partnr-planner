"""Per-variant ground truth: a live sandbox recording, saved as the one ground truth per variant."""
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from scripts.ground_truth_archive import ALL_TASKS_NOTE, GroundTruthArchive, file_sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'baseline_evaluation_v3'
SKILLS = ['Navigate', 'Explore', 'Open', 'Close', 'Pick', 'Place', 'Fill', 'Pour', 'Clean', 'PowerOn', 'PowerOff']
REPORT_RESULT = {'ok': True, 'response': 'Reported no suitable object; abandoning the absent-object subgoal.', 'skill_steps': 0}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_section(text, heading):
    match = re.search(rf'^## {heading}\n(.*?)(?=^## |\Z)', text, re.M | re.S)
    return match.group(1).strip() if match else ''


def furniture_catalog(path):
    """Spec furniture catalog: scene_id and {room: [{name, description}]} in catalog order."""
    scene, rooms, current = None, {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('scene_id:'):
            scene = line.split(':', 1)[1].strip()
        elif line.startswith('rooms:') or not line.strip():
            continue
        elif line.startswith('  - ') and current is not None:
            name, _, description = line[4:].partition(': ')
            rooms[current].append({'name': name.strip(), 'description': description.strip()})
        elif line.endswith(':') and not line.startswith(' '):
            current = line[:-1]
            rooms[current] = []
        elif current is not None and rooms[current]:
            # A description that wrapped onto its own line.
            rooms[current][-1]['description'] = (rooms[current][-1]['description'] + ' ' + line.strip()).strip()
    return scene, rooms


def succeeded(skill, result):
    """Default for whether a sandbox step is checked (included in packs): skills that reported success."""
    if skill == 'ReportAbsence':
        return True
    if not result or result.get('ok') is False or result.get('error'):
        return False
    return 'success' in str(result.get('response', '')).lower()


class GroundTruthStore:
    """SQLite record of exactly one saved ground truth per variant (plus an in-flight recording)."""

    def __init__(self, path):
        self.path = Path(path)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, variant TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TEXT NOT NULL, completed_at TEXT, action_count INTEGER NOT NULL DEFAULT 0,
                    spec_hash TEXT NOT NULL, dataset_hash TEXT NOT NULL, metadata TEXT NOT NULL,
                    snapshot TEXT NOT NULL DEFAULT '{}', error TEXT);
                CREATE TABLE IF NOT EXISTS actions (
                    request_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    skill TEXT NOT NULL, target TEXT NOT NULL, started_at TEXT NOT NULL,
                    result TEXT, UNIQUE(run_id, sequence));
                CREATE TABLE IF NOT EXISTS room_choices (
                    variant TEXT NOT NULL, object TEXT NOT NULL, rooms TEXT NOT NULL,
                    updated_at TEXT NOT NULL, PRIMARY KEY(variant, object));
                CREATE TABLE IF NOT EXISTS object_rooms (
                    object TEXT PRIMARY KEY, rooms TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS object_furniture (
                    object TEXT NOT NULL, scene TEXT NOT NULL, furniture TEXT NOT NULL,
                    updated_at TEXT NOT NULL, PRIMARY KEY(object, scene));
                CREATE TABLE IF NOT EXISTS notes (
                    scope TEXT PRIMARY KEY, text TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS action_packs (
                    id TEXT PRIMARY KEY, task TEXT NOT NULL, name TEXT NOT NULL, steps TEXT NOT NULL,
                    source_variant TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(task, name));
            ''')
            # Room annotations are keyed by object name, shared by every variant holding it.
            # Merge the older per-variant rows once; keep that table as the pre-merge record.
            if not db.execute('SELECT 1 FROM object_rooms LIMIT 1').fetchone():
                merged = {}
                for row in db.execute('SELECT object,rooms FROM room_choices ORDER BY updated_at'):
                    merged.setdefault(row['object'], []).extend(json.loads(row['rooms']))
                for obj, rooms in merged.items():
                    db.execute('INSERT INTO object_rooms VALUES(?,?,?)',
                               (obj, json.dumps(list(dict.fromkeys(rooms))), now()))
            existing = {row['name'] for row in db.execute('PRAGMA table_info(runs)')}
            columns = {'sim_steps': 'INTEGER NOT NULL DEFAULT 0', 'steps_complete': 'INTEGER NOT NULL DEFAULT 1',
                       'artifact_status': "TEXT NOT NULL DEFAULT 'pending'", 'artifact_error': 'TEXT',
                       'video_bytes': 'INTEGER', 'video_sha256': 'TEXT'}
            for column, definition in columns.items():
                if column not in existing:
                    db.execute(f'ALTER TABLE runs ADD COLUMN {column} {definition}')
            if 'sim_steps' not in existing:
                for row in db.execute('SELECT id FROM runs').fetchall():
                    actions = [dict(a) for a in db.execute('SELECT result FROM actions WHERE run_id=?', (row['id'],))]
                    for action in actions:
                        action['result'] = json.loads(action['result']) if action['result'] else None
                    steps, complete = self.step_totals(actions)
                    db.execute('UPDATE runs SET sim_steps=?,steps_complete=? WHERE id=?', (steps, complete, row['id']))
            # No history: keep only the newest completed run per variant, drop unfinished ones, then enforce it.
            db.execute('''DELETE FROM runs WHERE status='completed' AND EXISTS (
                SELECT 1 FROM runs newer WHERE newer.variant=runs.variant AND newer.status='completed'
                AND (newer.completed_at > runs.completed_at OR (newer.completed_at = runs.completed_at AND newer.id > runs.id)))''')
            db.execute("DELETE FROM runs WHERE status!='completed'")
            db.execute('DELETE FROM actions WHERE run_id NOT IN (SELECT id FROM runs)')
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_ground_truth ON runs(variant) WHERE status='completed'")

    @staticmethod
    def step_totals(actions):
        total, complete = 0, True
        for action in actions:
            result = action['result'] or {}
            steps = result.get('skill_steps')
            if not isinstance(steps, int) or steps < 0:
                complete = False
                continue
            total += steps
            if result.get('ok') is False and result.get('steps_source') != 'environment_step_calls':
                complete = False  # Legacy exception path reported zero even after stepping.
        return total, complete

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def new_recording(self, meta):
        run_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO runs(id,variant,status,started_at,spec_hash,dataset_hash,metadata) VALUES(?,?,?,?,?,?,?)',
                       (run_id, meta['variant'], 'recording', now(), meta['spec_hash'], meta['dataset_hash'], json.dumps(meta)))
        return run_id

    def get_run(self, run_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                return None
            data = dict(row)
            data['metadata'] = json.loads(data['metadata'])
            data['snapshot'] = json.loads(data['snapshot'])
            data['actions'] = [dict(a) for a in db.execute('SELECT * FROM actions WHERE run_id=? ORDER BY sequence', (run_id,))]
            for action in data['actions']:
                action['result'] = json.loads(action['result']) if action['result'] else None
            return data

    def add_action(self, run_id, sequence, skill, target, result):
        with self.connect() as db:
            db.execute('INSERT INTO actions VALUES(?,?,?,?,?,?,?)',
                       (uuid.uuid4().hex, run_id, sequence, skill, target, now(), json.dumps(result)))
            actions = [{'result': json.loads(row['result'])} for row in db.execute('SELECT result FROM actions WHERE run_id=?', (run_id,))]
            steps, complete = self.step_totals(actions)
            db.execute('UPDATE runs SET action_count=?,sim_steps=?,steps_complete=? WHERE id=?',
                       (len(actions), steps, complete, run_id))

    def complete(self, run_id, snapshot, artifact_status, artifact_error=None, video_bytes=None, video_sha256=None):
        """Make this recording the variant's only ground truth, deleting the one it replaces."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            variant = db.execute('SELECT variant FROM runs WHERE id=?', (run_id,)).fetchone()['variant']
            db.execute('DELETE FROM actions WHERE run_id IN (SELECT id FROM runs WHERE variant=? AND id!=?)', (variant, run_id))
            db.execute('DELETE FROM runs WHERE variant=? AND id!=?', (variant, run_id))
            db.execute('''UPDATE runs SET status='completed',completed_at=?,snapshot=?,artifact_status=?,artifact_error=?,
                          video_bytes=?,video_sha256=? WHERE id=?''',
                       (now(), json.dumps(snapshot), artifact_status, artifact_error, video_bytes, video_sha256, run_id))

    def discard(self, run_id):
        """Drop an unfinished recording; a saved ground truth is never deleted this way."""
        with self.connect() as db:
            if db.execute("SELECT 1 FROM runs WHERE id=? AND status!='completed'", (run_id,)).fetchone():
                db.execute('DELETE FROM actions WHERE run_id=?', (run_id,))
                db.execute('DELETE FROM runs WHERE id=?', (run_id,))

    def artifacts(self, run_id, status, error=None, video_bytes=None, video_sha256=None):
        with self.connect() as db:
            db.execute('UPDATE runs SET artifact_status=?,artifact_error=?,video_bytes=?,video_sha256=? WHERE id=?',
                       (status, error, video_bytes, video_sha256, run_id))

    def ground_truth(self, variant):
        with self.connect() as db:
            row = db.execute("SELECT id FROM runs WHERE variant=? AND status='completed'", (variant,)).fetchone()
        return self.get_run(row['id']) if row else None

    def all_ground_truths(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('''SELECT id,variant,completed_at,action_count,sim_steps,steps_complete,
                artifact_status,artifact_error,video_bytes,video_sha256,spec_hash,dataset_hash
                FROM runs WHERE status='completed' ORDER BY variant''')]

    def object_rooms(self):
        with self.connect() as db:
            return {row['object']: json.loads(row['rooms']) for row in db.execute('SELECT * FROM object_rooms')}

    def object_furniture(self, scene):
        with self.connect() as db:
            return {row['object']: json.loads(row['furniture'])
                    for row in db.execute('SELECT object,furniture FROM object_furniture WHERE scene=?', (scene,))}

    def save_furniture(self, obj, scene, furniture):
        with self.connect() as db:
            db.execute('''INSERT INTO object_furniture VALUES(?,?,?,?) ON CONFLICT(object,scene)
                          DO UPDATE SET furniture=excluded.furniture,updated_at=excluded.updated_at''',
                       (obj, scene, json.dumps(furniture), now()))

    def notes(self):
        with self.connect() as db:
            return {row['scope']: {'text': row['text'], 'updated_at': row['updated_at']}
                    for row in db.execute('SELECT * FROM notes ORDER BY scope')}

    def save_note(self, scope, text):
        with self.connect() as db:
            if text.strip():
                db.execute('INSERT INTO notes VALUES(?,?,?) ON CONFLICT(scope) DO UPDATE SET text=excluded.text,updated_at=excluded.updated_at',
                           (scope, text, now()))
            else:
                db.execute('DELETE FROM notes WHERE scope=?', (scope,))

    def save_rooms(self, obj, rooms):
        with self.connect() as db:
            db.execute('INSERT INTO object_rooms VALUES(?,?,?) ON CONFLICT(object) DO UPDATE SET rooms=excluded.rooms,updated_at=excluded.updated_at',
                       (obj, json.dumps(rooms), now()))

    def packs(self, task):
        with self.connect() as db:
            return [{**dict(row), 'steps': json.loads(row['steps'])}
                    for row in db.execute('SELECT * FROM action_packs WHERE task=? ORDER BY name COLLATE NOCASE', (task,))]

    def get_pack(self, pack_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM action_packs WHERE id=?', (pack_id,)).fetchone()
        return {**dict(row), 'steps': json.loads(row['steps'])} if row else None

    def save_pack(self, task, name, steps, source_variant):
        """Saving under an existing name in the same task replaces that pack's steps."""
        with self.connect() as db:
            db.execute('''INSERT INTO action_packs VALUES(?,?,?,?,?,?) ON CONFLICT(task,name) DO UPDATE SET
                          steps=excluded.steps,source_variant=excluded.source_variant,updated_at=excluded.updated_at''',
                       (uuid.uuid4().hex, task, name, json.dumps(steps), source_variant, now()))
            return dict(db.execute('SELECT id FROM action_packs WHERE task=? AND name=?', (task, name)).fetchone())['id']

    def delete_pack(self, pack_id):
        with self.connect() as db:
            return db.execute('DELETE FROM action_packs WHERE id=?', (pack_id,)).rowcount


class EpisodeCatalog:
    def __init__(self, base=BASE):
        self.base = base

    def get(self, variant):
        if not re.fullmatch(r'T[1-7]-(ACC|INC|OUT)-[A-Z]+', variant):
            raise ValueError('Invalid variant')
        index = json.loads((self.base / 'generation/episodes.json').read_text())['variants']
        if variant not in index:
            raise ValueError('Unknown variant')
        item = index[variant]
        task = variant.split('-')[0]
        spec = self.base / f'specs/{task}/{variant}.md'
        text = spec.read_text()
        catalog = (self.base / f'specs/{task}/_furniture_catalog.txt').read_text()
        rooms = list(dict.fromkeys(room.strip() for line in catalog.splitlines() if line.startswith('rooms:') for room in line.split(':', 1)[1].split(',')))
        scene, furniture = furniture_catalog(self.base / f'specs/{task}/_furniture_catalog.txt')
        with (self.base / 'variant_object_assets.csv').open() as f:
            objects = [dict(row) for row in csv.DictReader(f) if row['Variant_ID'] == variant]
        # Substitutes are not asked for, so they get no likely rooms or furniture.
        targets = [row['Object'].split(' (')[0] for row in objects if row['Label'] == 'target']
        absent = re.findall(r'^- (\w+) \([^\n]*\): [^\n]*absent from the scene', text, re.M)
        absent += re.findall(r'^- (\w+): (?:known )?absent', text, re.M)
        targets = list(dict.fromkeys(targets + absent))
        dataset = (self.base / item.get('dataset', '')).resolve()
        if not dataset.is_relative_to(self.base.resolve()):
            raise ValueError('Invalid episode path')
        meta = {'variant': variant, 'rooms': rooms, 'targets': targets, 'spec_hash': digest(spec),
                'robot_memory': spec_section(text, 'Initial robot memory'),
                'dataset': str(dataset), 'dataset_hash': '', 'episode_id': str(item.get('episode_id', variant)),
                'reports': [], 'unsupported_criteria': [], 'available': item['status'] == 'generated' and dataset.is_file(),
                'scene_id': scene, 'furniture': furniture}
        if not meta['available']:
            meta['blocked_reason'] = 'Generate this episode before recording ground truth.'
            return meta
        scene_info = json.loads((dataset.parent / 'scene_info.json').read_text())
        meta['rooms'] = list(scene_info.get('room_to_id', {})) or rooms
        # Add furniture the episode knows but the spec catalog omits; skip runtime aliases of catalog names.
        handles = scene_info.get('receptacle_to_handle', {})
        known = {entry['name'] for entries in furniture.values() for entry in entries}
        known_handles = {handles.get(name) for name in known} - {None}
        for room, names in scene_info.get('furniture', {}).items():
            for name in names:
                if name not in known and handles.get(name) not in known_handles:
                    furniture.setdefault(room, []).append(
                        {'name': name, 'description': scene_info.get('recep_to_description', {}).get(name, '')})
                    known.add(name)
        episode = json.loads(gzip.decompress(dataset.read_bytes()))['episodes']
        episode = next(ep for ep in episode if str(ep['episode_id']) == meta['episode_id'])
        meta['dataset_hash'] = digest(dataset)
        provenance = episode.get('info', {}).get('variant_spec', {})
        if provenance.get('sha256') != meta['spec_hash']:
            meta.update(available=False, blocked_reason='The episode does not match the current spec. Regenerate it before recording ground truth.')
        for criterion in provenance.get('unscored_success_text', []):
            if 'should report that no suitable object exists' in criterion:
                meta['reports'].append(criterion)
            elif criterion.startswith('- Using ') or criterion.startswith('- Distractors ('):
                # Exact entity handles in the compiled predicates enforce these choices.
                pass
            else:
                meta['unsupported_criteria'].append(criterion)
        meta['success_criteria'] = spec_section(text, 'Success criteria')
        meta['propositions'] = episode.get('evaluation_propositions', [])
        return meta


class HabitatWorker:
    def __init__(self, python=None):
        self.python = python or os.environ.get('HABITAT_PYTHON') or str(Path.home() / 'miniconda3/envs/habitat/bin/python')
        self.directory = Path(tempfile.mkdtemp(prefix='scenario-ground-truth-'))
        # Clips and the initial frame of the loaded episode; copied out only when a recording is saved.
        self.session_dir = self.directory / 'session'
        self.process = None

    def call(self, command):
        if self.process is None or self.process.poll() is not None:
            env = os.environ.copy()
            lib = str(Path(self.python).resolve().parent.parent / 'lib')
            env['LD_LIBRARY_PATH'] = lib + ':' + env.get('LD_LIBRARY_PATH', '')
            env.setdefault('MAGNUM_LOG', 'quiet')
            env.setdefault('HABITAT_SIM_LOG', 'quiet')
            with (self.directory / 'worker.log').open('ab') as log:
                self.process = subprocess.Popen([self.python, str(ROOT / 'scripts/ground_truth_worker.py'), str(self.directory)],
                    cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=log, stderr=log, text=True)
        command = {**command, 'id': uuid.uuid4().hex}
        self.process.stdin.write(json.dumps(command) + '\n')
        self.process.stdin.flush()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            path = self.directory / 'response.json'
            if path.exists():
                response = json.loads(path.read_text())
                if response['id'] == command['id']:
                    if response.get('error'):
                        raise RuntimeError(response['error'])
                    return response['result']
            if self.process.poll() is not None:
                raise RuntimeError(f'Habitat worker stopped. See {self.directory / "worker.log"}')
            time.sleep(0.1)
        self.process.terminate()
        raise RuntimeError('Habitat action timed out; reset the sandbox.')

    def frame(self):
        path = self.directory / 'frame.jpg'
        return path.read_bytes() if path.exists() else None

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)


def archive_locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.archive.lock:
            return method(self, *args, **kwargs)
    return wrapped


class GroundTruthService:
    """One independent simulator and live recording.

    Sandbox: load the exact episode; everything run from then on is being recorded.
    Save: store that live run (all actions, their clips) as ground truth once it completes the
    spec. A saved recording replaces the variant's previous ground truth. Nothing is replayed.
    """

    def __init__(self, database=None, worker=None, catalog=None, results_root=None, shared=None, on_rerecord=None):
        self.store = shared.store if shared else GroundTruthStore(database or BASE / 'ground_truth.sqlite3')
        self.on_rerecord = shared.on_rerecord if shared else on_rerecord  # Called with a variant whose ground truth was replaced.
        self.catalog = shared.catalog if shared else catalog or EpisodeCatalog()
        self.worker = worker or HabitatWorker()
        self.lock = threading.Lock()
        self.results_root = shared.results_root if shared else Path(results_root or BASE / 'ground_truth')
        self.archive = shared.archive if shared else GroundTruthArchive(self.results_root, self.store)
        self.archive_error = None
        self.variant = None      # Variant loaded in the simulator.
        self.meta = None
        self.mode = None         # 'sandbox' | None
        self.phase = ''
        self.steps = []          # Every action since the sandbox opened; checked ones go into packs.
        self.snapshot = {}
        self.load_id = None
        self.recording = None
        self.last_result = None
        self.error = None
        if shared:
            return  # Never run startup cleanup when allocating another live session.
        self.archive.clear_staging()
        saved = {run['variant']: run['id'] for run in self.store.all_ground_truths()}
        for folder in self.results_root.glob('T*/T*-*-*'):
            if folder.is_dir():
                self.archive.migrate_legacy(folder.name, saved.get(folder.name))
        for variant in saved:
            self._verify(self.store.ground_truth(variant))
        self.archive.index()

    def _verify(self, run):
        """Check the saved video against the database checksum; rebuild it from clips if needed."""
        try:
            folder = self.archive.directory(run['variant'])
            video = folder / 'video.mp4'
            if run['artifact_status'] == 'ready' and video.is_file() and run['video_sha256'] in (None, file_sha256(video)):
                if run['video_sha256'] is None:
                    self.store.artifacts(run['id'], 'ready', None, video.stat().st_size, file_sha256(video))
            else:
                try:
                    size, sha = self.archive.video(folder, run['actions'])
                    self.store.artifacts(run['id'], 'ready', None, size, sha)
                except Exception as error:
                    self.store.artifacts(run['id'], 'error', str(error))
            run = self.store.get_run(run['id'])
            self.archive.export(run, *self.choices(run['variant'], run['metadata']))
        except Exception as error:
            self.archive_error = str(error)

    def variant_furniture(self, meta):
        """Likely furniture per target in this apartment (furniture names are apartment-specific)."""
        names = {entry['name'] for entries in meta.get('furniture', {}).values() for entry in entries}
        stored = self.store.object_furniture(meta['scene_id']) if meta.get('scene_id') else {}
        return {obj: chosen for obj in meta['targets']
                if (chosen := [name for name in stored.get(obj, []) if name in names])}

    def choices(self, variant, meta=None):
        """Current (rooms, furniture) annotations for a variant, using the live catalog when possible."""
        try:
            meta = self.catalog.get(variant)
        except (ValueError, OSError, KeyError):
            meta = meta or {'targets': [], 'rooms': []}
        return self.variant_rooms(meta), self.variant_furniture(meta)

    def variant_rooms(self, meta):
        """This variant's view of the shared annotations: rooms its own apartment has."""
        stored = self.store.object_rooms()
        return {obj: rooms for obj in meta['targets']
                if (rooms := [room for room in stored.get(obj, []) if room in meta['rooms']])}

    def artifact_links(self, run):
        folder = f"../ground_truth/{run['variant'].split('-')[0]}/{run['variant']}/"
        # The URL stays the same when a recording is replaced, so bust the browser cache by checksum.
        version = (run.get('video_sha256') or run['id'])[:12]
        return {'data': folder + 'run.json', 'actions': folder + 'actions.csv', 'files': folder,
                'video': f'{folder}video.mp4?v={version}' if run['artifact_status'] == 'ready' else None}

    def state(self, variant):
        meta = self.catalog.get(variant)
        gt = self.store.ground_truth(variant)
        if gt:
            gt['artifacts'] = self.artifact_links(gt)
            gt['video_duration_sec'] = self.archive.video_duration(gt)
            gt['stale'] = gt['spec_hash'] != meta['spec_hash'] or gt['dataset_hash'] != meta['dataset_hash']
            gt['evaluation'] = gt['snapshot'].get('evaluation')
            gt['criteria_evidence'] = self.archive.evidence(gt)
            del gt['snapshot'], gt['metadata']
        here = self.variant == variant
        snapshot = self.snapshot if here else {}
        if here and self.lock.locked() and self.load_id:
            try:
                live = json.loads((self.worker.directory / 'evaluation.json').read_text())
                if live.get('load_id') == self.load_id:
                    snapshot = {**snapshot, 'evaluation': live['evaluation']}
            except (OSError, ValueError, KeyError):
                pass
        notes = self.store.notes()
        task = variant.split('-')[0]
        return {'meta': meta, 'ground_truth': gt, 'room_choices': self.variant_rooms(meta),
                'furniture_choices': self.variant_furniture(meta), 'skills': SKILLS,
                'notes': {'task': notes.get(task), 'all': notes.get(ALL_TASKS_NOTE)},
                'packs': self.store.packs(task), 'archive_error': self.archive_error,
                'session': {'variant': self.variant, 'mode': self.mode, 'busy': self.lock.locked(),
                            'phase': self.phase if here else '', 'steps': self.steps if here else [],
                            'snapshot': snapshot, 'recording': self.recording if here else None,
                            'last_result': self.last_result if here else None, 'error': self.error if here else None}}

    def _acquire(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError('Wait for the current operation to finish')

    def _background(self, job, *args):
        def run():
            try:
                job(*args)
            finally:
                self.phase = ''
                self.lock.release()
        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def _clean(snapshot):
        for key in ('history', 'logs', 'results_dir'):
            snapshot.pop(key, None)
        return snapshot

    def _load(self, meta):
        self.load_id = uuid.uuid4().hex
        return self._clean(self.worker.call({'op': 'load', 'dataset': meta['dataset'], 'episode_id': meta['episode_id'],
                                             'results_dir': str(self.worker.session_dir), 'load_id': self.load_id}))

    def _execute(self, skill, target):
        if skill == 'ReportAbsence':
            return dict(REPORT_RESULT)
        snapshot = self._clean(self.worker.call({'op': 'skill', 'skill': skill, 'target': target}))
        result = snapshot.pop('action_result', None) or {}
        self.snapshot = snapshot
        return result

    # ------------------------------------------------------------------ sandbox
    def sandbox(self, variant):
        """Load (or reset) the exact episode for free practice."""
        self._acquire()
        try:
            meta = self.catalog.get(variant)
            if not meta['available']:
                raise ValueError(meta['blocked_reason'])
        except Exception:
            self.lock.release()
            raise
        self.variant, self.meta, self.mode = variant, meta, None
        self.steps, self.snapshot, self.last_result, self.error = [], {}, None, None
        self.phase = 'Loading the episode at its exact start…'
        frame = self.worker.directory / 'frame.jpg'
        if frame.exists():
            frame.unlink()
        self._background(self._sandbox_job, meta)
        return {'loading': True}

    def _sandbox_job(self, meta):
        try:
            self.snapshot = self._load(meta)
            self.mode = 'sandbox'
        except Exception as error:
            self.error = f'Could not load the episode: {error}'

    def action(self, variant, payload):
        request_id = payload.get('request_id')
        skill, target = payload.get('skill'), payload.get('target')
        if not isinstance(request_id, str) or not re.fullmatch(r'[a-zA-Z0-9-]{16,64}', request_id):
            raise ValueError('A unique action request ID is required')
        if skill not in SKILLS + ['ReportAbsence'] or not isinstance(target, str) or not 0 < len(target.strip()) <= 4000:
            raise ValueError('Choose a skill and target')
        if variant != self.variant or self.mode != 'sandbox':
            raise ValueError('Open the sandbox for this variant first')
        if any(step['id'] == request_id for step in self.steps):
            return {'duplicate': True}
        if skill == 'ReportAbsence' and target not in self.meta['reports']:
            raise ValueError('No such reporting requirement in this spec')
        self._acquire()
        self.phase = f'Running {skill} {target.strip()}…'
        self._background(self._action_job, request_id, skill, target.strip())
        return {'running': True}

    def _action_job(self, request_id, skill, target):
        try:
            result = self._execute(skill, target)
        except Exception as error:
            result = {'ok': False, 'error': str(error), 'response': str(error)}
        self.steps.append({'id': request_id, 'skill': skill, 'target': target, 'result': result,
                           'keep': succeeded(skill, result)})

    def edit_steps(self, variant, payload):
        """Check/uncheck sandbox steps to choose what a pack saves. Every step that ran stays part
        of the recording, so steps cannot be removed; reset the sandbox to start over."""
        if variant != self.variant or self.mode != 'sandbox' or self.lock.locked():
            raise ValueError('Steps can only be edited in this variant\'s idle sandbox')
        step = next((s for s in self.steps if s['id'] == payload.get('id')), None)
        if step is None:
            raise ValueError('Unknown step')
        if not isinstance(payload.get('keep'), bool):
            raise ValueError('Expected keep')
        step['keep'] = payload['keep']
        return {'steps': self.steps}

    # ------------------------------------------------------------------ action packs
    def packs(self, variant, payload):
        """Named step sequences shared by every variant of a task: {save: name} stores this
        sandbox's checked steps, {apply: id} runs a pack in this sandbox, {delete: id} removes one."""
        self.catalog.get(variant)
        task = variant.split('-')[0]
        if 'save' in payload:
            name = payload['save'].strip() if isinstance(payload['save'], str) else ''
            if not 0 < len(name) <= 80:
                raise ValueError('Give the pack a name of at most 80 characters')
            if variant != self.variant or self.mode != 'sandbox' or self.lock.locked():
                raise ValueError('Open this variant\'s sandbox and wait for it to be idle')
            steps = [{'skill': s['skill'], 'target': s['target']} for s in self.steps if s['keep']]
            if not steps:
                raise ValueError('Check at least one sandbox step to save as a pack')
            return {'saved': True, 'id': self.store.save_pack(task, name, steps, variant)}
        if 'delete' in payload:
            if not self.store.delete_pack(payload['delete']):
                raise ValueError('Unknown pack')
            return {'deleted': True}
        if 'apply' in payload:
            pack = self.store.get_pack(payload['apply'])
            if pack is None or pack['task'] != task:
                raise ValueError(f'Unknown pack for {task}')
            if variant != self.variant or self.mode != 'sandbox':
                raise ValueError('Open the sandbox for this variant first')
            unknown = [s['target'] for s in pack['steps'] if s['skill'] == 'ReportAbsence' and s['target'] not in self.meta['reports']]
            if unknown:
                raise ValueError('This pack reports an absence this spec does not ask for: ' + '; '.join(unknown))
            self._acquire()
            self.last_result = None
            self._background(self._pack_job, pack)
            return {'running': True}
        raise ValueError('Expected save, apply, or delete')

    def _pack_job(self, pack):
        """Run the pack's steps in order as ordinary sandbox steps; stop at the first failure."""
        total = len(pack['steps'])
        for index, step in enumerate(pack['steps'], 1):
            self.phase = f"Applying pack “{pack['name']}”: step {index} of {total} · {step['skill']} {step['target']}"
            try:
                result = self._execute(step['skill'], step['target'])
            except Exception as error:
                result = {'ok': False, 'error': str(error), 'response': str(error)}
            ok = succeeded(step['skill'], result)
            self.steps.append({'id': uuid.uuid4().hex, 'skill': step['skill'], 'target': step['target'],
                               'result': result, 'keep': ok})
            if not ok:
                self.last_result = {'ok': False, 'message': f"Pack “{pack['name']}” stopped at step {index} of {total} "
                                    f"({step['skill']} {step['target']}): {result.get('response') or result.get('error')}"}
                return
        self.last_result = {'ok': True, 'message': f"Applied pack “{pack['name']}”: all {total} steps succeeded."}

    # ------------------------------------------------------------------ recording
    def record(self, variant):
        """Save the live sandbox as ground truth: every action run since the sandbox opened, in
        order, with the clips it already recorded. Nothing is replayed; reset the sandbox to redo."""
        self._acquire()
        try:
            if variant != self.variant or self.mode != 'sandbox':
                raise ValueError('Open the sandbox for this variant first')
            if not self.steps:
                raise ValueError('Run at least one action before saving ground truth')
            meta = self.catalog.get(variant)
            if not meta['available']:
                raise ValueError(meta['blocked_reason'])
            if (meta['spec_hash'], meta['dataset_hash']) != (self.meta['spec_hash'], self.meta['dataset_hash']):
                raise ValueError('The spec or episode changed since the sandbox opened. Reset the sandbox first.')
            evaluation = self.snapshot.get('evaluation', {})
            reported = {s['target'] for s in self.steps if s['skill'] == 'ReportAbsence'}
            if not evaluation.get('success'):
                raise ValueError(f"Not saved: the sandbox has reached {round(evaluation.get('percent_complete', 0) * 100)}% of the spec")
            if [r for r in meta['reports'] if r not in reported]:
                raise ValueError('Not saved: the absence report is missing')
            if meta['unsupported_criteria']:
                raise ValueError('Not saved: this spec has requirements that cannot be evaluated yet')
            run_id = self.store.new_recording(meta)
        except Exception:
            self.lock.release()
            raise
        self.meta, self.last_result = meta, None
        self.phase = 'Saving the ground truth and video…'
        self._background(self._record_job, run_id, meta, [dict(step) for step in self.steps])
        return {'recording': True}

    def _record_job(self, run_id, meta, steps):
        try:
            for index, step in enumerate(steps, 1):
                self.store.add_action(run_id, index, step['skill'], step['target'], step['result'])
            replaced = self.store.ground_truth(meta['variant']) is not None
            self._save(run_id, meta)
            flag_error = None
            if replaced and self.on_rerecord:
                try:
                    self.on_rerecord(meta['variant'])
                except Exception as error:
                    flag_error = error
            self.last_result = {'ok': True, 'message': f'Saved {len(steps)} actions as this variant\'s ground truth'
                                + (', replacing the previous one; it needs human re-verification.' if replaced else '.')
                                + (f' Could not flag it for re-verification: {flag_error}' if flag_error else '')}
        except Exception as error:
            self.store.discard(run_id)
            self.last_result = {'ok': False, 'message': f'Saving failed and nothing was saved: {error}'}

    @archive_locked
    def _save(self, run_id, meta):
        staging = self.archive.staging(meta['variant'])
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        run = self.store.get_run(run_id)
        source = Path(self.worker.session_dir)
        if (source / 'initial.jpg').exists():
            shutil.copy2(source / 'initial.jpg', staging / 'initial.jpg')
        for action in run['actions']:
            rel = (action['result'] or {}).get('ground_truth_video')
            if rel and (source / rel).is_file():
                (staging / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / rel, staging / rel)
        try:
            size, sha = self.archive.video(staging, run['actions'])
            self.store.complete(run_id, self.snapshot, 'ready', None, size, sha)
        except Exception as error:
            # A missing video never discards a completed task; the error is shown with the ground truth.
            self.store.complete(run_id, self.snapshot, 'error', str(error))
        self.archive.promote(meta['variant'], staging)
        try:
            self.archive.export(self.store.get_run(run_id), *self.choices(meta['variant'], meta))
            self.archive_error = None
        except Exception as error:
            self.archive_error = str(error)

    # ------------------------------------------------------------------ assumption notes
    @archive_locked
    def save_notes(self, variant, payload):
        """Free-text assumptions for this variant's task (e.g. T4), or shared by every task (T1-T7)."""
        self.catalog.get(variant)
        scope, text = payload.get('scope'), payload.get('text')
        if scope not in ('task', 'all') or not isinstance(text, str) or len(text) > 20000:
            raise ValueError('Expected scope task/all and at most 20,000 characters of text')
        key = variant.split('-')[0] if scope == 'task' else ALL_TASKS_NOTE
        self.store.save_note(key, text)
        try:
            self.archive.notes(self.store.notes())
        except OSError as error:
            self.archive_error = str(error)
        return {'saved': True, 'note': self.store.notes().get(key)}

    # ------------------------------------------------------------------ room annotations
    @archive_locked
    def rooms(self, variant, payload):
        meta = self.catalog.get(variant)
        obj, rooms = payload.get('object'), payload.get('rooms')
        if obj not in meta['targets'] or not isinstance(rooms, list) or any(room not in meta['rooms'] for room in rooms):
            raise ValueError('Choose a target object and rooms from this apartment')
        # Rooms this object was given in other apartments stay saved; a variant edits only its own.
        elsewhere = [room for room in self.store.object_rooms().get(obj, []) if room not in meta['rooms']]
        self.store.save_rooms(obj, list(dict.fromkeys(list(rooms) + elsewhere)))
        self.archive.refresh_variant(variant, *self.choices(variant, meta))
        self._refresh_shared(obj, variant)
        return {'saved': True}

    @archive_locked
    def furniture(self, variant, payload):
        """Ranked likely furniture for a target, shared by every variant in the same apartment."""
        meta = self.catalog.get(variant)
        obj, chosen = payload.get('object'), payload.get('furniture')
        names = {entry['name'] for entries in meta['furniture'].values() for entry in entries}
        if (obj not in meta['targets'] or not meta.get('scene_id') or not isinstance(chosen, list)
                or any(name not in names for name in chosen) or len(set(chosen)) != len(chosen)):
            raise ValueError('Choose a target object and furniture from this apartment')
        self.store.save_furniture(obj, meta['scene_id'], chosen)
        self.archive.refresh_variant(variant, *self.choices(variant, meta))
        self._refresh_shared(obj, variant)
        return {'saved': True}

    def _refresh_shared(self, obj, edited):
        """Other variants holding this object share the annotation, so re-export their copies."""
        for row in self.store.all_ground_truths():
            if row['variant'] == edited:
                continue
            run = self.store.get_run(row['id'])
            meta = run['metadata'] if run else {}
            if obj in meta.get('targets', []):
                self.archive.refresh_variant(row['variant'], *self.choices(row['variant'], meta))


class GroundTruthSessions:
    """Bounded pool, one explicitly addressed live recording per variant.

    The registry lock covers dispatch, not background simulator work. Session services
    share durable storage and the archive lock, but never workers or recording state.
    """
    def __init__(self, max_sessions=2, worker_factory=HabitatWorker, **kwargs):
        if max_sessions < 1:
            raise ValueError('max_sessions must be at least 1')
        self.max_sessions = max_sessions
        self.worker_factory = worker_factory
        self.shared = GroundTruthService(worker=worker_factory(), **kwargs)
        self.sessions = {}
        self.lock = threading.RLock()

    def _check(self, variant, session_id):
        entry = self.sessions.get(variant)
        if session_id is not None and (not entry or entry[0] != session_id):
            raise ValueError('This sandbox session has changed or closed. Refresh before retrying.')
        return entry

    def state(self, variant):
        with self.lock:
            entry = self.sessions.get(variant)
            service = entry[1] if entry else self.shared
            sessions = [{'id': key, 'variant': name, 'busy': item.lock.locked(), 'phase': item.phase}
                        for name, (key, item) in self.sessions.items()]
            session_id = entry[0] if entry else None
        state = service.state(variant)
        state['archive_error'] = state['archive_error'] or self.shared.archive_error
        state['session']['id'] = session_id
        state['sessions'] = sessions
        state['max_sessions'] = self.max_sessions
        return state

    def frame(self, variant, session_id=None):
        with self.lock:
            entry = self._check(variant, session_id)
            return entry[1].worker.frame() if entry else None

    def dispatch(self, variant, operation, payload):
        # Annotation exports may wait for a video save. Do not hold up unrelated
        # simulator dispatch while they wait for the shared archive lock.
        annotations = {'notes': 'save_notes', 'rooms': 'rooms', 'furniture': 'furniture'}
        if operation in annotations:
            with self.lock:
                self._check(variant, payload.get('session_id'))
            return getattr(self.shared, annotations[operation])(variant, payload)
        with self.lock:
            entry = self._check(variant, payload.get('session_id'))
            if operation == 'sandbox':
                if not entry:
                    meta = self.shared.catalog.get(variant)
                    if not meta['available']:
                        raise ValueError(meta['blocked_reason'])
                    if len(self.sessions) >= self.max_sessions:
                        names = ', '.join(self.sessions)
                        raise ValueError(f'All {self.max_sessions} sandbox slots are in use ({names}). Close one sandbox first.')
                    service = GroundTruthService(worker=self.worker_factory(), shared=self.shared)
                    entry = (uuid.uuid4().hex, service)
                    self.sessions[variant] = entry
                try:
                    result = entry[1].sandbox(variant)
                except Exception:
                    if entry[1].variant is None:
                        entry[1].worker.close()
                        del self.sessions[variant]
                    raise
                # A reset creates a new generation: reject late commands from the old run.
                session_id = uuid.uuid4().hex
                self.sessions[variant] = (session_id, entry[1])
                return {**result, 'session_id': session_id}
            if operation == 'close':
                if not entry:
                    raise ValueError('No sandbox is open for this variant')
                service = entry[1]
                if service.lock.locked():
                    raise ValueError('Wait for this sandbox operation to finish before closing it')
                service.worker.close()
                shutil.rmtree(service.worker.directory, ignore_errors=True)
                del self.sessions[variant]
                return {'closed': True}
            methods = {'record': 'record', 'action': 'action', 'steps': 'edit_steps',
                       'packs': 'packs', 'notes': 'save_notes', 'rooms': 'rooms', 'furniture': 'furniture'}
            if operation not in methods:
                raise ValueError('Unknown sandbox operation')
            needs_session = operation in ('record', 'action', 'steps') or (operation == 'packs' and ('apply' in payload or 'save' in payload))
            if needs_session and not entry:
                raise ValueError('Open the sandbox for this variant first')
            service = entry[1] if entry else self.shared
            method = getattr(service, methods[operation])
            return method(variant) if operation == 'record' else method(variant, payload)

    def close(self):
        # Finish outstanding operations before shutting down their workers.
        with self.lock:
            for _, service in self.sessions.values():
                with service.lock:
                    service.worker.close()
            self.shared.worker.close()
