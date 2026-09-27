#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


def aggregate_outputs_dir_from_results_dir(results_dir: str) -> Path:
    """
    Derive the aggregate outputs folder from a run or experiment results dir.

    Examples:
      results/vlm_tamp_pddl_v5/Task_5_Sg/Task_5_Sg_1 -> results/outputs_vlm_tamp_pddl_v5
      results/vlm_tamp_pddl_v5 -> results/outputs_vlm_tamp_pddl_v5
    """
    resolved = Path(results_dir).resolve()
    experiment_root = resolved
    task_dir = resolved.parent
    if task_dir != resolved and resolved.name.startswith(task_dir.name + "_"):
        experiment_root = task_dir.parent
    return experiment_root.parent / f"outputs_{experiment_root.name}"
