"""Scheduler tests run without Habitat, GPUs, Slurm, or API credentials."""

import contextlib
import io
import importlib.util
import json
import os
import shlex
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_parallel as runner


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.episode = self.root / "episode" / "dataset.json"
        self.episode.parent.mkdir()
        self.episode.write_text(json.dumps({"episodes": [{"episode_id": "T2-INC-CON"}]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_direct_terminal_mirroring_restores_streams_after_failure(self):
        spec = importlib.util.spec_from_file_location(
            "run_log", runner.REPO / "habitat_llm/utils/run_log.py")
        run_log = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_log)
        log = self.root / "log.txt"
        log.write_text("earlier output\n")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, "episode failed"):
                with run_log.tee_to_run_log(log, "direct episode"):
                    print("terminal stdout")
                    print("terminal stderr", file=sys.stderr)
                    raise RuntimeError("episode failed")
            self.assertIs(sys.stdout, stdout)
            self.assertIs(sys.stderr, stderr)
        self.assertEqual(stdout.getvalue(), "terminal stdout\n")
        self.assertEqual(stderr.getvalue(), "terminal stderr\n")
        captured = log.read_text()
        self.assertTrue(captured.startswith("earlier output\n"))
        self.assertEqual(captured.count("terminal stdout"), 1)
        self.assertEqual(captured.count("terminal stderr"), 1)
        self.assertIn("--- end: direct episode", captured)

    def args(self, *more):
        return ["--episodes", str(self.episode), "--python", sys.executable,
                "--output-dir", str(self.root / "results"), "--cpus-per-worker", "1", *more]

    def test_preview_is_read_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(self.args("--workers", "1", "--dry-run")), 0)
        self.assertFalse((self.root / "results").exists())

    def test_video_flag_controls_final_override(self):
        for enabled in (False, True):
            preview = io.StringIO()
            extra = ["--save-video"] if enabled else []
            with contextlib.redirect_stdout(preview):
                self.assertEqual(runner.main(self.args(
                    "--workers", "1", "--dry-run", "--override",
                    f"evaluation.save_video={not enabled}", *extra)), 0)
            cmd = shlex.split(preview.getvalue().splitlines()[-1])
            video = [a for a in cmd if a.startswith("evaluation.save_video=")]
            self.assertEqual(video[-1], f"evaluation.save_video={enabled}")

    def test_tru_pomdp_uses_its_own_config_without_react_overrides(self):
        preview = io.StringIO()
        with contextlib.redirect_stdout(preview):
            self.assertEqual(runner.main(self.args("--planner", "tru-pomdp", "--workers", "1", "--dry-run")), 0)
        self.assertIn("--config-name=baselines/single_agent_tru_pomdp", preview.getvalue())
        self.assertNotIn("inference_mode", preview.getvalue())
        self.assertNotIn("vlm.max_completion_tokens", preview.getvalue())
        self.assertFalse((self.root / "results").exists())

    def test_multi_episode_file_rejected(self):
        self.episode.write_text(json.dumps({"episodes": [{}, {}]}))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.datasets([str(self.episode)])

    def test_select_repetition_without_rerunning_other_repetitions(self):
        preview = io.StringIO()
        args = self.args("--workers", "1", "--runs-per-episode", "3", "--dry-run")
        with contextlib.redirect_stdout(preview):
            self.assertEqual(runner.main(args), 0)
        ids = [Path(next(arg.split("=", 1)[1] for arg in shlex.split(line)
                         if arg.startswith("paths.results_dir="))).name
               for line in preview.getvalue().splitlines()[2:]]
        selected = io.StringIO()
        with contextlib.redirect_stdout(selected):
            self.assertEqual(runner.main(args + ["--run-ids", ids[1]]), 0)
        self.assertTrue(selected.getvalue().startswith("1 runs, 1 workers"))
        self.assertIn(ids[1], selected.getvalue())
        self.assertNotIn(ids[0], selected.getvalue())
        self.assertNotIn(ids[2], selected.getvalue())
        self.assertFalse((self.root / "results").exists())

    def test_unknown_repetition_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(self.args("--workers", "1", "--dry-run", "--run-ids", "missing"))
        self.assertFalse((self.root / "results").exists())

    def test_existing_output_rejected(self):
        (self.root / "results").mkdir()
        marker = self.root / "results" / "keep"
        marker.write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(self.args("--workers", "1", "--dry-run"))
        self.assertEqual(marker.read_text(), "keep")

    def run_fake(self, sleep, timeout=30, incremental=False, early_failure=False):
        def fake_command(python, planner, output, episode, overrides, probe=False, save_video=False):
            script = """
import json, os, pathlib, time
root = pathlib.Path(%r)
(root/'started.json').write_text(json.dumps({'time':time.time(), 'cwd':os.getcwd()}))
assert (root/'log.txt').exists()
print('startup-stdout', flush=True)
os.write(2, b'native-startup-stderr\x5cn')
if %r:
    raise ModuleNotFoundError('simulator_import_failed_before_episode_setup')
import importlib.util
spec=importlib.util.spec_from_file_location('run_log', %r)
run_log=importlib.util.module_from_spec(spec); spec.loader.exec_module(run_log)
with run_log.tee_to_run_log(str(root/'log.txt'), 'episode header'):
    print('episode-output-once', flush=True)
    os.write(2, b'native-episode-stderr\x5cn')
time.sleep(%r)
p=root/'dataset/planner-log/result.json';p.parent.mkdir(parents=True)
p.write_text(json.dumps({'steps':[{'stats':{'task_state_success':0.0,'task_percent_complete':0.5}}]}))
""" % (str(output), early_failure, str(runner.REPO / 'habitat_llm/utils/run_log.py'), sleep)
            if incremental:
                setup = """
import importlib.util
spec=importlib.util.spec_from_file_location('progress', %r)
progress=importlib.util.module_from_spec(spec); spec.loader.exec_module(progress)
writer=progress.EpisodeProgress(root/'dataset', 'fake_episode', 'test')
writer.start_action(0,'Pick','cup')
writer.finish_action(0,'success')
writer.record({'stats':{'task_state_success':0.0,'task_percent_complete':0.5}})
writer.start_action(0,'Explore','kitchen')
writer.simulator_step({'task_state_success':0.0,'task_percent_complete':0.75})
""" % str(runner.REPO / 'habitat_llm/utils/episode_progress.py')
                script = script.replace("time.sleep(", setup + "\ntime.sleep(", 1)
            return [python, "-c", script]

        with patch.dict(os.environ, {"SLURM_JOB_ID": "test", "SLURM_STEP_ID": "test",
                                     "SLURM_CPUS_PER_TASK": "2", "OPENAI_API_KEY": "test-only"}), \
                patch.object(runner, "gpu_sample", return_value=["fake"]), \
                patch.object(runner, "workspace", side_effect=lambda p: p.mkdir(parents=True)), \
                patch.object(runner, "command", side_effect=fake_command), \
                contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(self.args("--workers", "2", "--runs-per-episode", "3",
                                         "--timeout", str(timeout)))
        return code, json.loads((self.root / "results/summary.json").read_text())["jobs"]

    def test_queue_concurrency_isolation_and_metrics(self):
        code, jobs = self.run_fake(2)
        self.assertEqual(code, 0)
        self.assertEqual([j["status"] for j in jobs], ["completed"] * 3)
        self.assertTrue(all(j["metrics"]["task_state_success"] == 0 for j in jobs))
        starts = [json.loads((Path(j["output"]) / "started.json").read_text()) for j in jobs]
        self.assertNotEqual(starts[0]["cwd"], starts[1]["cwd"])
        self.assertLess(abs(starts[0]["time"] - starts[1]["time"]), 1.5)
        self.assertGreater(starts[2]["time"] - starts[0]["time"], 2)
        self.assertEqual(len({j["output"] for j in jobs}), 3)
        for job in jobs:
            root = Path(job['output'])
            log = (root / 'log.txt').read_text()
            self.assertTrue(log.startswith('Simulator command started:'))
            self.assertIn('Working directory:', log)
            self.assertIn('Command:', log)
            self.assertTrue((root / 'console.log').samefile(root / 'log.txt'))
            for marker in ('startup-stdout', 'native-startup-stderr',
                           'episode-output-once', 'native-episode-stderr'):
                self.assertEqual(log.splitlines().count(marker), 1)

    def test_import_failure_is_captured_before_episode_setup(self):
        code, jobs = self.run_fake(0, early_failure=True)
        self.assertEqual(code, 1)
        for job in jobs:
            self.assertEqual(job['status'], 'failed')
            log = (Path(job['output']) / 'log.txt').read_text()
            self.assertIn('Traceback (most recent call last):', log)
            self.assertIn('ModuleNotFoundError: simulator_import_failed_before_episode_setup', log)

    def test_timeout_recovers_actions_and_latest_partial_metrics(self):
        code, jobs = self.run_fake(60, timeout=0.5, incremental=True)
        self.assertEqual(code, 1)
        for job in jobs:
            self.assertEqual(job['status'], 'timeout')
            self.assertTrue(job['recovered_logs'])
            self.assertTrue(job['metrics']['metrics_are_partial'])
            self.assertEqual(job['metrics']['task_percent_complete'], .75)
            log = json.loads(Path(job['recovered_logs'][0]).read_text())
            self.assertEqual(log['actions'][0]['response'], 'success')
            self.assertEqual(log['actions'][1]['status'], 'incomplete')
            self.assertFalse(log['complete'])

    def test_timeout_is_reported(self):
        code, jobs = self.run_fake(60, timeout=0.1)
        self.assertEqual(code, 1)
        self.assertTrue(all(j["status"] == "timeout" for j in jobs))
        self.assertTrue(all(j["exit_code"] is not None for j in jobs))
        for job in jobs:
            log = (Path(job['output']) / 'log.txt').read_text()
            self.assertIn('startup-stdout', log.splitlines())


if __name__ == "__main__":
    unittest.main()
