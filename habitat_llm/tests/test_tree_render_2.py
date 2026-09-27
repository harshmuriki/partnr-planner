#!/usr/bin/env python3

from habitat_llm.vlm_tamp.tree_render_2 import (
    _build_tree_from_events,
    _tree_to_dot,
    render_planning_tree_from_log_dir,
)


def _write_jsonl(log_dir, events):
    path = log_dir / "vlm_tamp_pddl_log.jsonl"
    path.write_text("\n".join(repr(ev) for ev in events) + "\n", encoding="utf-8")
    return path


def test_tree_render_2_merges_initial_prefixes():
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
        }
    ]

    nodes, root_id, pair_to_node = _build_tree_from_events(events)

    assert root_id == "root"
    assert len(nodes[root_id].children) == 1
    assert pair_to_node[(0, 0)] == pair_to_node[(1, 0)]
    assert pair_to_node[(0, 1)] == pair_to_node[(1, 1)]
    assert pair_to_node[(0, 2)] != pair_to_node[(1, 2)]


def test_tree_render_2_appends_reprompt_branch_without_remerge():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [
                ["explore(entryway_1)", "opened-drawer(drawer_1)", "picked(box_1)"]
            ],
            "seq_idx": 0,
        },
        {
            "event": "reprompt_started",
            "reprompt_round": 1,
            "reason": "pddl_no_plan",
            "failed_branch": 0,
            "failed_subgoal_idx": 2,
            "failed_subgoal": "picked(box_1)",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 1,
            "append": True,
            "active_branch": 1,
            "added_indices": [1],
            "added_branches": [["picked(box_1)", "on(box_1, table_1)"]],
            "seq_idx": 2,
        },
    ]

    nodes, _root_id, pair_to_node = _build_tree_from_events(events)

    initial_pick = pair_to_node[(0, 2)]
    reprompt_pick = pair_to_node[(1, 0)]
    assert initial_pick != reprompt_pick

    shared_parent = pair_to_node[(0, 1)]
    assert nodes[initial_pick].parent_id == shared_parent
    assert nodes[reprompt_pick].parent_id == shared_parent
    assert len(nodes[shared_parent].children) == 2


def test_tree_render_2_uses_first_active_branch_from_multi_branch_reprompt():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [["explore(entryway_1)"]],
            "seq_idx": 0,
        },
        {
            "event": "reprompt_started",
            "reprompt_round": 1,
            "reason": "explore_refresh",
            "branch": 0,
            "subgoal_idx": 0,
            "subgoal": "explore(entryway_1)",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 1,
            "append": True,
            "active_branch": 1,
            "added_indices": [1, 2],
            "added_branches": [
                ["opened-drawer(drawer_1)"],
                ["opened-drawer(drawer_2)"],
            ],
            "seq_idx": 2,
        },
    ]

    nodes, _root_id, pair_to_node = _build_tree_from_events(events)

    assert (1, 0) in pair_to_node
    assert (2, 0) not in pair_to_node
    chosen_node = nodes[pair_to_node[(1, 0)]]
    assert "drawer_1" in chosen_node.name


def test_tree_render_2_reprompt_after_first_subgoal_failure_uses_last_successful_parent():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [["explore(entryway_1)"]],
            "seq_idx": 0,
        },
        {
            "event": "reprompt_started",
            "reprompt_round": 1,
            "reason": "explore_refresh",
            "branch": 0,
            "subgoal_idx": 0,
            "subgoal": "explore(entryway_1)",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 1,
            "append": True,
            "active_branch": 1,
            "added_indices": [1],
            "added_branches": [["on(box_1, table_1)"]],
            "seq_idx": 2,
        },
        {
            "event": "reprompt_started",
            "reprompt_round": 2,
            "reason": "pddl_no_plan",
            "failed_branch": 1,
            "failed_subgoal_idx": 0,
            "failed_subgoal": "on(box_1, table_1)",
            "seq_idx": 3,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 2,
            "append": True,
            "active_branch": 2,
            "added_indices": [2],
            "added_branches": [["picked(box_1)"]],
            "seq_idx": 4,
        },
    ]

    nodes, _root_id, pair_to_node = _build_tree_from_events(events)

    failed_first = nodes[pair_to_node[(1, 0)]]
    replanned_first = nodes[pair_to_node[(2, 0)]]
    assert failed_first.parent_id == pair_to_node[(0, 0)]
    assert replanned_first.parent_id == pair_to_node[(0, 0)]


def test_tree_render_2_infers_missing_reprompt_started_after_fast_explore():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [["explore(entryway_1)", "picked(box_1)"]],
            "seq_idx": 0,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 0,
            "subgoal": "explore(entryway_1)",
            "status": "success",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": True,
            "active_branch": 1,
            "added_indices": [1],
            "added_branches": [["opened-drawer(drawer_1)"]],
            "seq_idx": 2,
        },
    ]

    nodes, _root_id, pair_to_node = _build_tree_from_events(events)

    assert nodes[pair_to_node[(1, 0)]].parent_id == pair_to_node[(0, 0)]


def test_tree_render_2_consumes_reprompt_started_once_before_fast_explore_append():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [["explore(entryway_1)", "picked(box_1)"]],
            "seq_idx": 0,
        },
        {
            "event": "reprompt_started",
            "reprompt_round": 1,
            "reason": "pddl_no_plan",
            "failed_branch": 0,
            "failed_subgoal_idx": 1,
            "failed_subgoal": "picked(box_1)",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 1,
            "append": True,
            "active_branch": 1,
            "added_indices": [1],
            "added_branches": [["explore(kitchen_1)"]],
            "seq_idx": 2,
        },
        {
            "event": "subgoal_status",
            "branch": 1,
            "subgoal_idx": 0,
            "subgoal": "explore(kitchen_1)",
            "status": "success",
            "seq_idx": 3,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 1,
            "append": True,
            "active_branch": 2,
            "added_indices": [2],
            "added_branches": [["opened-drawer(drawer_1)"]],
            "seq_idx": 4,
        },
    ]

    nodes, _root_id, pair_to_node = _build_tree_from_events(events)

    assert nodes[pair_to_node[(1, 0)]].parent_id == pair_to_node[(0, 0)]
    assert nodes[pair_to_node[(2, 0)]].parent_id == pair_to_node[(1, 0)]


def test_tree_render_2_applies_latest_status_color_and_fallback():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0, 1],
            "added_branches": [
                ["explore(entryway_1)", "opened-drawer(drawer_1)"],
                ["explore(entryway_1)", "opened-drawer(drawer_1)"],
            ],
            "seq_idx": 0,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 1,
            "status": "started",
            "seq_idx": 1,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 1,
            "status": "solved",
            "seq_idx": 2,
        },
        {
            "event": "subgoal_status",
            "branch": 1,
            "subgoal_idx": 1,
            "status": "already",
            "seq_idx": 3,
        },
        {
            "event": "pddl_plan",
            "branch": 0,
            "subgoal_idx": 0,
            "subgoal": "explore(entryway_1)",
            "status": "failed",
            "failure_type": "pddl_no_plan",
            "seq_idx": 4,
        },
    ]

    nodes, _root_id, pair_to_node = _build_tree_from_events(events)
    from habitat_llm.vlm_tamp.tree_render_2 import _apply_status_colors

    _apply_status_colors(nodes, pair_to_node, events)

    shared_open = nodes[pair_to_node[(0, 1)]]
    assert pair_to_node[(0, 1)] == pair_to_node[(1, 1)]
    assert shared_open.status == "already"
    assert shared_open.color == "dodgerblue3"

    explore_node = nodes[pair_to_node[(0, 0)]]
    assert explore_node.status == "failed"
    assert explore_node.color == "red"


def test_tree_render_2_treats_fast_explore_success_as_green_solved():
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
            "subgoal_idx": 0,
            "status": "success",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_started",
            "reprompt_round": 1,
            "reason": "explore_refresh",
            "branch": 0,
            "subgoal_idx": 0,
            "subgoal": "explore(entryway_1)",
            "seq_idx": 2,
        },
    ]

    nodes, _root_id, pair_to_node = _build_tree_from_events(events)
    from habitat_llm.vlm_tamp.tree_render_2 import _apply_status_colors

    _apply_status_colors(nodes, pair_to_node, events)

    explore_node = nodes[pair_to_node[(0, 0)]]
    skipped_suffix = nodes[pair_to_node[(0, 1)]]
    assert explore_node.status == "solved"
    assert explore_node.color == "green"
    assert skipped_suffix.status == "skipped"
    assert skipped_suffix.color == "goldenrod3"


def test_tree_render_2_marks_explore_refresh_suffix_as_skipped():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [[
                "explore(entryway_1)",
                "opened-drawer(drawer_1)",
                "picked(box_1)",
            ]],
            "seq_idx": 0,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 0,
            "status": "solved",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_started",
            "reprompt_round": 1,
            "reason": "explore_refresh",
            "branch": 0,
            "subgoal_idx": 0,
            "subgoal": "explore(entryway_1)",
            "seq_idx": 2,
        },
    ]

    nodes, root_id, pair_to_node = _build_tree_from_events(events)
    from habitat_llm.vlm_tamp.tree_render_2 import _apply_status_colors

    _apply_status_colors(nodes, pair_to_node, events)
    skipped_node = nodes[pair_to_node[(0, 1)]]
    dot_text = _tree_to_dot(nodes, root_id)

    assert skipped_node.status == "skipped"
    assert skipped_node.color == "goldenrod3"
    assert 'color="goldenrod3"' in dot_text
    assert 'fontcolor="goldenrod4"' in dot_text
    assert 'fillcolor=' not in dot_text
    assert 'style="filled"' not in dot_text


def test_tree_render_2_marks_inferred_fast_explore_suffix_as_skipped():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [[
                "explore(entryway_1)",
                "opened-drawer(drawer_1)",
                "picked(box_1)",
            ]],
            "seq_idx": 0,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 0,
            "subgoal": "explore(entryway_1)",
            "status": "success",
            "seq_idx": 1,
        },
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": True,
            "active_branch": 1,
            "added_indices": [1],
            "added_branches": [["opened-door(cabinet_1)"]],
            "seq_idx": 2,
        },
    ]

    nodes, root_id, pair_to_node = _build_tree_from_events(events)
    from habitat_llm.vlm_tamp.tree_render_2 import _apply_status_colors

    _apply_status_colors(nodes, pair_to_node, events)
    skipped_node = nodes[pair_to_node[(0, 1)]]
    dot_text = _tree_to_dot(nodes, root_id)

    assert skipped_node.status == "skipped"
    assert skipped_node.color == "goldenrod3"
    assert 'color="goldenrod3"' in dot_text
    assert 'fontcolor="goldenrod4"' in dot_text


def test_tree_render_2_strikes_through_suffix_after_failed_node():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0],
            "added_branches": [[
                "explore(entryway_1)",
                "opened-drawer(drawer_1)",
                "picked(box_1)",
            ]],
            "seq_idx": 0,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 1,
            "status": "failed",
            "failure_type": "execution_failure",
            "seq_idx": 1,
        },
    ]

    nodes, root_id, pair_to_node = _build_tree_from_events(events)
    from habitat_llm.vlm_tamp.tree_render_2 import _apply_status_colors

    _apply_status_colors(nodes, pair_to_node, events)
    blocked_node = nodes[pair_to_node[(0, 2)]]
    dot_text = _tree_to_dot(nodes, root_id)

    assert blocked_node.status == "blocked"
    assert blocked_node.color == "gray55"
    assert "<S>r0:s2</S><BR/><S>picked(box_1)</S>" in dot_text
    assert 'fontcolor="gray45"' in dot_text


def test_tree_render_2_includes_legend_and_blue_pddl_skip_styling():
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
    skipped_by_pddl = nodes[pair_to_node[(0, 1)]]
    dot_text = _tree_to_dot(nodes, root_id)

    assert skipped_by_pddl.status == "already"
    assert skipped_by_pddl.color == "dodgerblue3"
    assert 'color="dodgerblue3"' in dot_text
    assert 'fontcolor="dodgerblue4"' in dot_text
    assert "Legend" in dot_text
    assert "Skipped by PDDL" in dot_text
    assert "Skipped after explore" in dot_text


def test_tree_render_2_renders_png_and_dot_from_synthetic_log(tmp_path):
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
            "subgoal_idx": 0,
            "status": "solved",
            "seq_idx": 1,
        },
        {
            "event": "subgoal_status",
            "branch": 0,
            "subgoal_idx": 1,
            "status": "failed",
            "failure_type": "pddl_no_plan",
            "failure_msg": "synthetic failure",
            "seq_idx": 2,
        },
    ]
    _write_jsonl(log_dir, events)

    nodes, root_id, _pair_to_node = _build_tree_from_events(events)
    dot_text = _tree_to_dot(nodes, root_id, show_failure_labels=True, failure_label_mode="flag")
    assert "digraph planning_tree" in dot_text
    assert "r0:s0, r1:s0\\nexplore(entryway_1)" in dot_text

    out_path = render_planning_tree_from_log_dir(str(log_dir))
    assert out_path == str(log_dir / "media" / "planning_tree.png")
    assert (log_dir / "media" / "planning_tree.png").is_file()
    assert (log_dir / "media" / "planning_tree.png").stat().st_size > 0


def test_tree_render_2_labels_use_branch_and_subgoal_ids():
    events = [
        {
            "event": "reprompt_branches_added",
            "reprompt_round": 0,
            "append": False,
            "added_indices": [0, 1],
            "added_branches": [
                ["explore(entryway_1)", "opened-drawer(drawer_1)"],
                ["explore(entryway_1)", "opened-drawer(drawer_1)"],
            ],
            "seq_idx": 0,
        },
    ]

    nodes, root_id, pair_to_node = _build_tree_from_events(events)
    shared_node = nodes[pair_to_node[(0, 0)]]
    dot_text = _tree_to_dot(nodes, root_id)

    assert shared_node.pair_refs == [(0, 0), (1, 0)]
    assert "r0:s0, r1:s0\\nexplore(entryway_1)" in dot_text


def test_tree_render_2_writes_layout_json_and_interactive_html(tmp_path):
    from habitat_llm.vlm_tamp.render_pddl_baseline_html import (
        render_pddl_baseline_log_dir_to_html,
    )

    log_dir = tmp_path / "Task_1_Dis_2" / "vlm_tamp_pddl" / "unknown"
    (log_dir / "media").mkdir(parents=True, exist_ok=True)
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
            "subgoal_idx": 0,
            "status": "success",
            "subgoal": "explore(entryway_1)",
            "seq_idx": 1,
        },
        {
            "event": "subgoal_execution",
            "branch": 0,
            "subgoal_idx": 0,
            "log_text": "HIGH-LEVEL EXPLORE",
            "subgoal": "explore(entryway_1)",
            "status": "solved",
            "seq_idx": 2,
        },
    ]
    _write_jsonl(log_dir, events)

    png_path = render_planning_tree_from_log_dir(
        str(log_dir),
        write_layout_json=True,
    )
    html_path = render_pddl_baseline_log_dir_to_html(str(log_dir))

    assert png_path == str(log_dir / "media" / "planning_tree.png")
    assert (log_dir / "media" / "planning_tree_layout.json").is_file()
    assert html_path == str(log_dir / "Task_1_Dis_2_pddl.html")
    html_text = (log_dir / "Task_1_Dis_2_pddl.html").read_text(encoding="utf-8")
    assert 'data-node="b0:s0"' in html_text
    assert 'class="step success"' in html_text
    assert "Execution log" in html_text
    assert "HIGH-LEVEL EXPLORE" in html_text
    assert 'id="expand"' in html_text
    assert '<svg' not in html_text
