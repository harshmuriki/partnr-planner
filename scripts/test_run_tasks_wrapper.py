"""Wrapper filesystem regressions; no Habitat, GPU, or API access required."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

SPEC = importlib.util.spec_from_file_location(
    "run_tasks_wrapper", Path(__file__).with_name("run_tasks_wrapper.py")
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.default_output = self.root / "default_results"
        self.output = self.root / "worker_results"
        self.task = {
            "task_id": "test",
            "task_folder": str(self.root / "task"),
            "episode_file": str(self.root / "task" / "episode.json.gz"),
        }
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump({
            "output_base_dir": str(self.default_output),
            "num_runs_per_task": 1,
            "planner_config": "baselines/single_agent_zero_shot_react_summary",
            "tasks": [self.task],
        }))

    def invoke(self, *args):
        with patch.object(sys, "argv", ["wrapper", "--config", str(self.config),
                                       "--output-dir", str(self.output), *args]):
            with contextlib.redirect_stdout(io.StringIO()):
                wrapper.main()

    def test_dry_run_preserves_existing_results(self):
        run = self.output / "task" / "task_1"
        run.mkdir(parents=True)
        for path in [run / "valuable.txt", self.output / "experiment_config.json",
                     self.output / "experiment_summary.json"]:
            path.write_text("existing results")
        before = {p.relative_to(self.output): p.read_bytes()
                  for p in self.output.rglob("*") if p.is_file()}
        with patch.object(wrapper.subprocess, "Popen") as launch:
            self.invoke("--dry-run")
            launch.assert_not_called()
        after = {p.relative_to(self.output): p.read_bytes()
                 for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(self.default_output.exists())

    def test_dry_run_creates_no_output(self):
        self.invoke("--dry-run")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.default_output.exists())

    def test_output_override_routes_real_summaries(self):
        def fake_run(task, run_num, base_output_dir, *args, **kwargs):
            (Path(base_output_dir) / "task").mkdir(parents=True, exist_ok=True)
            return {"success": True, "run_id": "task_1"}

        with patch.object(wrapper, "run_task", side_effect=fake_run):
            self.invoke()
        self.assertTrue((self.output / "experiment_config.json").is_file())
        self.assertTrue((self.output / "experiment_summary.json").is_file())
        self.assertTrue((self.output / "task" / "task_summary.json").is_file())
        self.assertFalse(self.default_output.exists())


if __name__ == "__main__":
    unittest.main()
