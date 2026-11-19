#!/usr/bin/env python3

from __future__ import annotations

import html
import os
from pathlib import Path
import re
from urllib.parse import quote


def export_html_with_relative_assets(source: Path, destination: Path) -> None:
    """Copy a report while resolving its links against the original asset folder.

    The report and its source folder must stay in the same relative layout when
    serving or moving the results. No images need to be duplicated.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    relative_root = os.path.relpath(source.parent, destination.parent)
    base_url = quote(relative_root.replace(os.sep, "/") + "/", safe="/")
    document = source.read_text(encoding="utf-8")
    document, count = re.subn(
        r"(<head\b[^>]*>)",
        lambda match: match.group(1) + f'<base href="{html.escape(base_url, quote=True)}">',
        document,
        count=1,
        flags=re.IGNORECASE,
    )
    if not count:
        raise ValueError(f"HTML report has no head element: {source}")
    destination.write_text(document, encoding="utf-8")


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
