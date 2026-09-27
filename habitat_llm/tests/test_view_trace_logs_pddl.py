#!/usr/bin/env python3

from scripts import view_trace_logs


def test_parse_trace_file_pddl_success_and_failure(tmp_path):
    trace_path = tmp_path / "trace.txt"
    trace_path.write_text(
        "\n".join(
            [
                "Task: demo task",
                "Subgoal: picked(box_4)",
                "Action: Navigate[box_4]",
                "Observation: Successful execution!",
                "Action: Pick[box_6]",
                "Observation: Unexpected failure! - Failed to pick!",
                "Action failed: hard failure",
            ]
        ),
        encoding="utf-8",
    )

    parsed = view_trace_logs.parse_trace_file(str(trace_path), is_pddl_run=True)
    assert parsed["total_steps"] == 2
    assert parsed["steps"][0]["action"] == "Navigate"
    assert parsed["steps"][0]["success"] is True
    assert "Successful execution!" in parsed["steps"][0].get("pddl_log", "")
    assert parsed["steps"][1]["action"] == "Pick"
    assert parsed["steps"][1]["success"] is False
    assert "Action failed: hard failure" in parsed["steps"][1].get("pddl_log", "")


def test_parse_trace_file_pddl_high_level_special_case_explore(tmp_path):
    trace_path = tmp_path / "trace.txt"
    trace_path.write_text(
        "\n".join(
            [
                "Task: demo task",
                "Subgoal: explore(entryway_1)",
                "High-level special-case: Explore[entryway_1]",
                "Observation: Successful execution!",
                "Recorded formal action: explore(entryway_1) (success)",
            ]
        ),
        encoding="utf-8",
    )

    parsed = view_trace_logs.parse_trace_file(str(trace_path), is_pddl_run=True)
    assert parsed["total_steps"] == 1
    assert parsed["steps"][0]["action"] == "Explore"
    assert parsed["steps"][0]["args"] == "entryway_1"
    assert parsed["steps"][0]["success"] is True


def test_parse_trace_file_pddl_action_result_marker(tmp_path):
    trace_path = tmp_path / "trace.txt"
    trace_path.write_text(
        "\n".join(
            [
                "Task: demo task",
                "Action: Navigate[table_12]",
                "✓ ACTION RESULT: SUCCESS",
                "Action: Pick[box_6]",
                "✗ ACTION RESULT: FAILED",
            ]
        ),
        encoding="utf-8",
    )

    parsed = view_trace_logs.parse_trace_file(str(trace_path), is_pddl_run=True)
    assert parsed["total_steps"] == 2
    assert parsed["steps"][0]["success"] is True
    assert parsed["steps"][1]["success"] is False


def test_generate_html_embeds_planning_tree_for_pddl(tmp_path):
    out_html = tmp_path / "trace.html"
    img_path = tmp_path / "media" / "planning_tree.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"fake")

    trace_data = {
        "task": "demo task",
        "steps": [{"action": "Navigate", "args": "table_12", "result": "Successful execution!", "success": True}],
        "total_steps": 1,
    }
    view_trace_logs.generate_html(
        trace_data,
        str(out_html),
        is_pddl_run=True,
        planning_tree_image=str(img_path),
    )
    html_text = out_html.read_text(encoding="utf-8")
    assert "Planning Tree" in html_text
    assert "media/planning_tree.png" in html_text


def test_generate_html_pddl_uses_trace_count_when_info_zero(tmp_path):
    out_html = tmp_path / "trace.html"
    trace_data = {
        "task": "demo task",
        "steps": [
            {
                "action": "Navigate",
                "args": "table_12",
                "result": "Successful execution!",
                "success": True,
                "pddl_log": "Observation: Successful execution!",
            }
        ],
        "total_steps": 1,
    }
    view_trace_logs.generate_html(
        trace_data,
        str(out_html),
        is_pddl_run=True,
        action_counts={"Navigate": 0},
    )
    html_text = out_html.read_text(encoding="utf-8")
    assert "Execution Log" in html_text
    assert "title=\"Trace file: 1, Info: 0\">1</div>" in html_text


def test_generate_html_includes_explore_approx_and_planning_time(tmp_path):
    out_html = tmp_path / "trace.html"
    trace_data = {
        "task": "demo task",
        "steps": [
            {
                "action": "Navigate",
                "args": "table_12",
                "result": "Successful execution!",
                "success": True,
            }
        ],
        "total_steps": 1,
    }
    view_trace_logs.generate_html(
        trace_data,
        str(out_html),
        action_sim_steps={"Navigate": 120, "Pick": 3},
        action_counts={"Navigate": 1, "Pick": 1, "Explore": 2},
        runtime=12.5,
        llm_planning_time_s=4.25,
        explore_approx_sim_steps=149,
        explore_approx_sim_time_s=1.24,
        sim_freq=120.0,
        combined_time_limit_s=600,
        combined_time_limit_hit=True,
    )
    html_text = out_html.read_text(encoding="utf-8")
    assert "LLM" in html_text
    assert "Explore" in html_text
    assert "times <b>2</b>" in html_text
    assert "Actions" in html_text
    assert "Exact actions" not in html_text
    assert "Combined budget" in html_text
    assert "10-minute clock / limit" in html_text
    assert "Combined is not wall-clock runtime" in html_text
    assert "4.25s" in html_text
    assert "149" in html_text
    assert "1.24s" in html_text
    assert "1.00s sim" in html_text
    assert "walked/geodesic" in html_text
    assert "Wall-clock runtime" in html_text
    assert "Episode stopped because combined budget was exceeded" in html_text


def test_generate_html_includes_model_and_api_cost(tmp_path):
    out_html = tmp_path / "trace.html"
    view_trace_logs.generate_html(
        {
            "task": "demo task",
            "steps": [],
            "total_steps": 0,
        },
        str(out_html),
        llm_model="gpt-5.6-luna",
        llm_reasoning_effort="high",
        prompt_tokens=1500,
        completion_tokens=80,
        cached_tokens=100,
        llm_usd=0.00042,
        llm_usd_source="api",
    )
    html_text = out_html.read_text(encoding="utf-8")
    assert "gpt-5.6-luna" in html_text
    assert ">high<" in html_text
    assert "API cost" in html_text
    assert "$0.0004" in html_text
    assert "1,500" in html_text
    assert "80" in html_text


def test_explore_metrics_from_trace_text():
    text = (
        "[fast_explore] approx tour kitchen_1: 10 furniture, 32.1 m, "
        "~391 sim steps, ~3.26 s sim time\n"
        "[fast_explore] approx tour living_room_1: 7 furniture, 22.5 m, "
        "~272 sim steps, ~2.27 s sim time\n"
    )
    metrics = view_trace_logs._explore_metrics_from_text(text)
    assert metrics["explore_approx_sim_steps"] == 663
    assert abs(metrics["explore_approx_sim_time_s"] - 5.53) < 1e-9


def test_generate_html_loads_planner_log_metrics(tmp_path):
    dataset = tmp_path / "dataset"
    traces = dataset / "traces" / "0"
    traces.mkdir(parents=True)
    planner_dir = dataset / "planner-log"
    planner_dir.mkdir()
    (planner_dir / "planner-log-episode_2_0.json").write_text(
        '{"steps":[{"sim_step_count":0,"high_level_actions":{"0":["Navigate","x",null]}},'
        '{"sim_step_count":120,"high_level_actions":{"0":["Navigate","x",null]},'
        '"cost_metrics":{"llm_planning_time_s":4.25,'
        '"explore_approx_sim_steps":149,"explore_approx_sim_time_s":1.24,'
        '"explore_approx_meters":10.0},"replanning_count":{"0":3}}]}',
        encoding="utf-8",
    )
    out_html = traces / "trace-episode_2_0-0.html"
    view_trace_logs.generate_html(
        {"task": "demo", "steps": [], "total_steps": 0},
        str(out_html),
    )
    html_text = out_html.read_text(encoding="utf-8")
    assert "4.25s" in html_text
    assert "1.24s" in html_text
    assert "149" in html_text
    assert "0.00s / 600.00s" not in html_text


def test_generate_html_embeds_step_camera_images(tmp_path):
    out_html = tmp_path / "trace.html"
    images = tmp_path / "images"
    images.mkdir()
    (images / "step_0000.png").write_bytes(b"fake-png")
    (images / "step_0001.png").write_bytes(b"fake-png")
    trace_data = {
        "task": "demo task",
        "steps": [
            {
                "action": "Explore",
                "args": "living_room_1",
                "result": "Successful execution!",
                "success": True,
            },
            {
                "action": "Navigate",
                "args": "table_11",
                "result": "Successful execution!",
                "success": True,
            },
        ],
        "total_steps": 2,
    }
    view_trace_logs.generate_html(trace_data, str(out_html))
    html_text = out_html.read_text(encoding="utf-8")
    assert 'src="images/step_0000.png"' in html_text
    assert 'src="images/step_0001.png"' in html_text
    assert "Robot camera" in html_text

