"""
Compatibility shim for the active planning tree renderer.

The project now uses the kitchen-worlds-style renderer implemented in
`tree_render_2.py` as the default `render_planning_tree` entrypoint.
"""

from habitat_llm.vlm_tamp.tree_render_2 import render_planning_tree_from_log_dir

__all__ = ["render_planning_tree_from_log_dir"]
