"""Portable per-variant ground-truth files: one folder per variant, replaced on each new recording."""
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading

RUN_ID = re.compile(r'[0-9a-f]{32}')


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.export-', delete=False) as stream:
            name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def json_write(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class GroundTruthArchive:
    def __init__(self, root, store):
        self.root = Path(root)
        self.store = store
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, variant):
        return self.root / variant.split('-')[0] / variant

    def staging(self, variant):
        return self.root / variant.split('-')[0] / f'.{variant}.recording'

    def clear_staging(self):
        for folder in self.root.glob('T*/.*.recording'):
            shutil.rmtree(folder, ignore_errors=True)

    def migrate_legacy(self, variant, keep_id):
        """Flatten the old T<n>/<variant>/<run-id>/ layout; drop other runs' folders."""
        folder = self.directory(variant)
        if not folder.is_dir():
            return
        for child in list(folder.iterdir()):
            if child.is_dir() and RUN_ID.fullmatch(child.name):
                if child.name == keep_id:
                    for item in child.iterdir():
                        target = folder / item.name
                        if target.is_dir():
                            shutil.rmtree(target)
                        elif target.exists():
                            target.unlink()
                        shutil.move(str(item), target)
                    child.rmdir()
                else:
                    shutil.rmtree(child)
        (folder / 'latest.json').unlink(missing_ok=True)
        # The old skill runner also wrote an unused per-run videos/ folder.
        videos = folder / 'videos'
        if videos.is_dir():
            shutil.rmtree(videos)

    def remove(self, variant):
        """No saved ground truth: keep only the variant's current room annotations."""
        folder = self.directory(variant)
        if not folder.is_dir():
            return
        for child in folder.iterdir():
            if child.name in ('room_choices.json', 'furniture_choices.json'):
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()

    def video(self, folder, actions):
        """Concatenate the recorded clips in action order; return (bytes, sha256)."""
        folder = Path(folder)
        destination = folder / 'video.mp4'
        clips, errors = [], []
        for action in actions:
            result = action['result'] or {}
            rel = result.get('ground_truth_video')
            if result.get('recording_error'):
                errors.append(f"Action {action['sequence']}: {result['recording_error']}")
            if rel:
                clip = (folder / rel).resolve()
                if not clip.is_relative_to(folder.resolve()) or not clip.is_file():
                    errors.append(f"Missing video for action {action['sequence']}")
                else:
                    clips.append(clip)
            elif result.get('skill_steps', 0) > 0:
                errors.append(f"Action {action['sequence']} has simulator steps but no video")
        if errors:
            raise RuntimeError('; '.join(errors))
        temporary = folder / '.video-building.mp4'
        args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
        if clips:
            # ffconcat quoting also supports repository paths containing apostrophes.
            listing = folder / '.video-clips.txt'
            listing.write_text(''.join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in clips))
            args += ['-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy']
        else:
            initial = folder / 'initial.jpg'
            if not initial.exists():
                raise RuntimeError('No initial frame or action video is available for this recording')
            # Zero-step completion / report-only recording: show the actual initial scene.
            args += ['-loop', '1', '-i', str(initial), '-t', '1', '-r', '30', '-c:v', 'libx264',
                     '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p']
        args += ['-movflags', '+faststart', str(temporary)]
        try:
            subprocess.run(args, check=True, capture_output=True, timeout=180)
            if not temporary.exists() or not temporary.stat().st_size:
                raise RuntimeError('Video encoder produced no video')
            os.replace(temporary, destination)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('Video assembly failed: ' + error.stderr.decode(errors='replace')[-1200:]) from error
        finally:
            temporary.unlink(missing_ok=True)
            (folder / '.video-clips.txt').unlink(missing_ok=True)
        return destination.stat().st_size, file_sha256(destination)

    def promote(self, variant, staging):
        """Swap a finished recording in place of the previous ground truth, keeping room annotations."""
        with self.lock:
            folder = self.directory(variant)
            for name in ('room_choices.json', 'furniture_choices.json'):
                if (folder / name).exists():
                    shutil.copy2(folder / name, Path(staging) / name)
            old = folder.with_name(f'.{variant}.replaced')
            shutil.rmtree(old, ignore_errors=True)
            if folder.exists():
                os.replace(folder, old)
            os.replace(staging, folder)
            shutil.rmtree(old, ignore_errors=True)

    def export(self, run, room_choices, furniture_choices=None):
        with self.lock:
            folder = self.directory(run['variant'])
            folder.mkdir(parents=True, exist_ok=True)
            export = dict(run)
            export['schema_version'] = 2
            export['simulator_steps_definition'] = 'Sum of per-action environment step calls; reports add zero.'
            export['video_playback_fps'] = 30
            # Rooms at the moment this ground truth was saved; current choices live in room_choices.json.
            old = folder / 'run.json'
            previous = json.loads(old.read_text()) if old.exists() else {}
            same = previous.get('id') == run['id']
            export['room_choices'] = previous.get('room_choices', room_choices) if same else room_choices
            export['furniture_choices'] = previous.get('furniture_choices', furniture_choices or {}) if same else furniture_choices or {}
            export['video'] = 'video.mp4' if run.get('artifact_status') == 'ready' else None
            json_write(folder / 'run.json', export)
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(['sequence', 'skill', 'target', 'simulator_steps', 'response', 'video'])
            for action in run['actions']:
                result = action['result'] or {}
                writer.writerow([action['sequence'], action['skill'], action['target'], result.get('skill_steps', ''),
                                 result.get('response') or result.get('error', ''), result.get('ground_truth_video') or ''])
            atomic_write(folder / 'actions.csv', stream.getvalue())
            self.refresh_variant(run['variant'], room_choices, furniture_choices)

    def notes(self, notes):
        """Readable copy of every assumption note; SQLite remains the source of truth."""
        with self.lock:
            lines = ['# Assumptions', '', 'Written in the scenario viewer. Task notes apply to every variant of that task.', '']
            for scope, note in sorted(notes.items(), key=lambda item: (item[0].split('-')[0], '-' in item[0], item[0])):
                lines += [f"## {scope}{'' if '-' in scope else ' (whole task)'}", '', f"_Updated {note['updated_at']}_", '', note['text'].rstrip(), '']
            atomic_write(self.root / 'assumptions.md', '\n'.join(lines))

    def refresh_variant(self, variant, room_choices, furniture_choices=None):
        with self.lock:
            json_write(self.directory(variant) / 'room_choices.json', room_choices)
            json_write(self.directory(variant) / 'furniture_choices.json', furniture_choices or {})
            self.index()

    def index(self):
        with self.lock:
            runs = self.store.all_ground_truths()
            columns = ['variant', 'id', 'completed_at', 'action_count', 'sim_steps', 'steps_complete',
                       'artifact_status', 'artifact_error', 'video_bytes', 'video_sha256', 'spec_hash', 'dataset_hash']
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: run[key] for key in columns} for run in runs)
            atomic_write(self.root / 'index.csv', stream.getvalue())
            json_write(self.root / 'index.json', {'schema_version': 2, 'ground_truths': runs})
            rows = []
            for run in runs:
                base = f"{run['variant'].split('-')[0]}/{run['variant']}/"
                video = f'<a href="{base}video.mp4">Watch video</a>' if run['artifact_status'] == 'ready' else html.escape(run['artifact_status'])
                rows.append(f"<tr><td>{html.escape(run['variant'])}</td><td>{html.escape(run['completed_at'] or '')}</td>"
                            f"<td>{run['action_count']}</td><td>{run['sim_steps']}{'' if run['steps_complete'] else ' (partial)'}</td>"
                            f'<td>{video}</td><td><a href="{base}run.json">Run JSON</a> · '
                            f'<a href="{base}actions.csv">Actions CSV</a> · <a href="{base}">Files</a></td></tr>')
            atomic_write(self.root / 'index.html', '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground-truth archive</title><style>body{font:14px system-ui;margin:32px;color:#222}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}a{color:#2563eb}.scroll{overflow:auto}</style>
<h1>Ground-truth archive</h1><p>One saved ground truth per variant. Recording a variant again replaces its previous ground truth.</p>
<p><a href="index.csv">Download all (CSV)</a> · <a href="index.json">All (JSON)</a> · <a href="../gui/scenario_viewer.html">Scenario viewer</a></p>
<div class="scroll"><table><thead><tr><th>Variant</th><th>Saved</th><th>Actions</th><th>Simulator steps</th><th>Video</th><th>Data</th></tr></thead><tbody>''' + ''.join(rows) + '</tbody></table></div>')
            atomic_write(self.root / 'README.md', '''# Ground-truth archive

Open index.html through the scenario viewer server, or use index.csv/index.json for analysis.

Layout: T<task>/<variant>/run.json, actions.csv, video.mp4, initial.jpg, clips/, room_choices.json,
furniture_choices.json (ranked likely furniture per target; shared by variants in the same apartment).
There is exactly one ground truth per variant. Recording again replaces it; no history is kept.
Ground truth is recorded by replaying the checked sandbox steps from the episode's exact start.
Sandbox (practice) actions are never saved.
Run JSON includes action results, evaluation state, spec/dataset hashes, video size/SHA-256,
and room choices at the time it was saved. Reports add zero simulator steps.

The live source of truth is ../ground_truth.sqlite3 (actions, counts, video checksum).
Keep this folder and that database together when backing up. Use sqlite3's backup API
for a live database backup, or stop the viewer before copying the database.
''')
