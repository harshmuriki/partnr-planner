#!/usr/bin/env python3
"""Generate index.html for a PARTNR VLM-TAMP PDDL episode log directory."""

import argparse
import os
import sys

# Allow running from repo root without installing the package
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render kitchen-worlds-style index.html from vlm_tamp_pddl_log.jsonl"
    )
    p.add_argument(
        "log_dir",
        help="Episode log directory (contains vlm_tamp_pddl_log.jsonl), e.g. "
        "outputs/vlm_pddl/<run>/vlm_tamp_pddl/<episode_id>",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: <log_dir>/index.html)",
    )
    p.add_argument(
        "--no-tree",
        action="store_true",
        help="Skip planning_tree.png generation.",
    )
    p.add_argument(
        "--failure-label-mode",
        choices=["flag", "type", "detailed"],
        default="flag",
        help="Failure label mode for planning tree nodes.",
    )
    args = p.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        print(f"Not a directory: {log_dir}", file=sys.stderr)
        return 1

    if not args.no_tree:
        from habitat_llm.vlm_tamp.render_planning_tree import (
            render_planning_tree_from_log_dir,
        )

        render_planning_tree_from_log_dir(
            log_dir,
            show_failure_labels=True,
            failure_label_mode=args.failure_label_mode,
        )

    from habitat_llm.vlm_tamp.render_log_html import render_log_dir_to_html

    out = render_log_dir_to_html(log_dir, out_path=args.output)
    if out is None:
        print(f"Missing or empty vlm_tamp_pddl_log.jsonl under {log_dir}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
