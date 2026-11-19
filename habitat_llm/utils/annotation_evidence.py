"""Incremental, evaluator-only evidence; never added to a planner prompt."""

import json
import os
import time
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class AnnotationEvidenceWriter:
    """Persist boundaries before teardown so killed runs retain useful evidence."""

    def __init__(self, output_dir, episode_name, instruction):
        self.enabled = os.environ.get("PARTNR_ANNOTATION_EVIDENCE", "1") != "0"
        self.path = Path(output_dir) / (
            "evidence-" + Path(episode_name).name + ".jsonl"
        )
        self.previous = None
        self.sequence = 0
        self.started = time.monotonic()
        self.checkpoint_step = 0
        self.checkpoint_path = Path(output_dir) / "evaluation-checkpoint.json"
        self._append(
            {
                "event": "start",
                "schema_version": 1,
                "instruction": instruction,
                "evaluator_only": True,
            }
        )

    def _append(self, record):
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            **record,
                            "sequence": self.sequence,
                            "wall_time": time.time(),
                            "elapsed_seconds": time.monotonic() - self.started,
                        },
                        default=_json_value,
                    )
                    + "\n"
                )
                stream.flush()
            self.sequence += 1
        except (OSError, TypeError, ValueError) as error:
            self.enabled = False
            warnings.warn(
                "Annotation evidence logging disabled: " + type(error).__name__,
                stacklevel=2,
            )

    def record(self, planner_info, evaluator_snapshot=None, final=False):
        if not self.enabled:
            return
        selected = {
            k: planner_info[k]
            for k in (
                "high_level_actions",
                "responses",
                "thought",
                "replanned",
                "replan_required",
                "curr_graph",
                "stats",
            )
            if k in planner_info
        }
        signature = json.dumps(selected, sort_keys=True, default=str)
        if signature == self.previous and not final:
            return
        self.previous = signature
        record = {
            "event": "terminal" if final else "boundary",
            "planner_observed": selected,
            "total_step_count": planner_info.get("total_step_count"),
            "sim_step_count": planner_info.get("sim_step_count"),
        }
        if evaluator_snapshot:
            try:
                record["evaluator_only_state"] = evaluator_snapshot()
            except Exception as error:
                record["evaluator_snapshot_error"] = type(error).__name__
        self._append(record)

    def checkpoint(self, snapshot, initial=False):
        """Atomically retain the latest completed simulator step, even in a skill.

        This is explicitly a checkpoint, never a final result. Keep a compact
        history every 25 steps in addition to replacing the latest state each step.
        """
        if not self.enabled:
            return
        if not initial:
            self.checkpoint_step += 1
        try:
            record = {"event": "checkpoint", "schema_version": 1,
                      "sim_step_count": self.checkpoint_step,
                      "wall_time": time.time(),
                      "elapsed_seconds": time.monotonic() - self.started,
                      "evaluator_only_state": snapshot()}
            payload = json.dumps(record, default=_json_value) + "\n"
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.checkpoint_path.with_suffix('.tmp')
            with temporary.open('w') as stream:
                stream.write(payload)
                stream.flush()
            temporary.replace(self.checkpoint_path)
            if self.checkpoint_step % 25 == 0:
                with self.checkpoint_path.with_name('evaluation-checkpoints.jsonl').open('a') as stream:
                    stream.write(payload)
        except (OSError, TypeError, ValueError) as error:
            warnings.warn('Evaluator checkpoint failed: ' + type(error).__name__, stacklevel=2)


def evaluator_snapshot(measures, full_world_graph=None):
    """Read existing measures only: no simulator stepping or metric updates."""
    names = (
        "auto_eval_proposition_tracker",
        "task_constraint_validation",
        "task_percent_complete",
        "task_state_success",
        "task_explanation",
    )
    snapshot = {name: measures[name].get_metric() for name in names if name in measures}
    # The full trajectory grows on every step. Persist the current state and its
    # index; satisfaction times and constraint validation already encode scoring.
    tracker = snapshot.get("auto_eval_proposition_tracker")
    if isinstance(tracker, dict):
        tracker = dict(tracker)
        history = tracker.pop("state_sequence", [])
        tracker["state_sequence_length"] = len(history)
        tracker["last_state"] = history[-1] if history else []
        snapshot["auto_eval_proposition_tracker"] = tracker
    if full_world_graph is not None:
        # Simulator handles resolve runtime aliases against archived scene inputs.
        # This reference graph must never be passed into planner_info or prompts.
        snapshot["reference_world_graph"] = [
            {
                "name": node.name,
                "sim_handle": node.sim_handle,
                "properties": {
                    k: v for k, v in node.properties.items() if k != "region"
                },
                "relations": {
                    neighbor.name: relation for neighbor, relation in neighbors.items()
                },
            }
            for node, neighbors in full_world_graph.graph.items()
        ]
    return snapshot
