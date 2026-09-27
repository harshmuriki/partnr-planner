#!/usr/bin/env python3

from habitat_llm.vlm_tamp.render_planning_tree import render_planning_tree_from_log_dir
from habitat_llm.vlm_tamp.tree_render_2 import _build_tree_from_events, _tree_to_dot


def _write_jsonl(log_dir, events):
    path = log_dir / "vlm_tamp_pddl_log.jsonl"
    path.write_text("\n".join(repr(ev) for ev in events) + "\n", encoding="utf-8")
    return path


def test_render_planning_tree_wrapper_uses_kitchen_worlds_style_renderer(tmp_path):
    log_dir = tmp_path / "demo_log"
    (log_dir / "media").mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0, 1],
            "added_branches": [
                ["explore(entryway_1)", "opened-drawer(drawer_1)", "picked(box_1)"],
                ["explore(entryway_1)", "opened-drawer(drawer_1)", "picked(box_2)"],
            ],
            "seq_idx": 0,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 1,
            "status": "already",
            "seq_idx": 1,
        },
    ]
    _write_jsonl(log_dir, events)

    out_path = render_planning_tree_from_log_dir(str(log_dir))
    assert out_path == str(log_dir / "media" / "planning_tree.png")
    assert (log_dir / "media" / "planning_tree.png").is_file()


def test_render_planning_tree_pddl_skipped_nodes_use_blue_outline_only():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [["explore(entryway_1)", "opened-drawer(drawer_1)"]],
            "seq_idx": 0,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 1,
            "status": "already",
            "seq_idx": 1,
        },
    ]

    nodes, root_id, pair_to_node = _build_tree_from_events(events)
    from habitat_llm.vlm_tamp.tree_render_2 import _apply_status_colors

    _apply_status_colors(nodes, pair_to_node, events)
    dot_text = _tree_to_dot(nodes, root_id)

    assert 'color="dodgerblue3"' in dot_text
    assert 'fontcolor="dodgerblue4"' in dot_text
    assert 'fillcolor=' not in dot_text
    assert 'style="filled"' not in dot_text
    assert "r0:s1\\nopened-drawer(drawer_1)" in dot_text
    assert "Legend" in dot_text
