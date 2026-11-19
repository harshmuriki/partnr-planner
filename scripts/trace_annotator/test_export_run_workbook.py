"""Distinguish actual skill selections from plans, ticks, and composite internals."""

from export_run_workbook import console_actions, summary_step_count, trace_actions


def test_vlm_trace_does_not_duplicate_explore_diagnostics_or_plans():
    text = """PDDL plan: [Action(name='navigate', args=('agent_0', 'x'))]
Action: Navigate[x]
High-level special-case: Explore[kitchen_1]
  Action: Explore[kitchen_1]
  ▶ [1/2] Navigate[internal_cabinet]
High-level special-case: Explore[kitchen_1]
  Action: Explore[kitchen_1]
Action: Open[cabinet_0]
"""
    assert trace_actions(text, "vlm_tamp_images") == [
        ("Navigate", "x"),
        ("Explore", "kitchen_1"),
        ("Explore", "kitchen_1"),
        ("Open", "cabinet_0"),
    ]


def test_tru_trace_includes_followup_skills_without_replanning():
    text = """Decision 1: (OPEN cabinet_0)
Action: Navigate[cabinet_0]
Result: Successful execution!
Action: Open[cabinet_0]
Result: Successful execution!
"""
    assert len(trace_actions(text, "tru_pomdp_no_images")) == 2


def test_react_console_does_not_count_reprinted_prompt_actions():
    text = """Example: Pick[cup_0]
🤖 LLM RESPONSE
================================================================================
Thought: I should move first.
Navigate[table_0]
================================================================================
Prompt history:
Navigate[table_0]
🤖 LLM RESPONSE
================================================================================
Thought: Finished.
Done[]
================================================================================
"""
    actions, _ = console_actions(text, "react_images")
    assert actions == [("Navigate", "table_0"), ("Done", "")]


def test_vlm_console_ignores_proposed_pddl_plan():
    text = """Plan (2 steps):
  1. navigate(agent_0, cabinet_0, kitchen_1)
  2. open(agent_0, cabinet_0, kitchen_1)
  ▶ [1/2]  Navigate[cabinet_0]
  ▶ [2/2]  Open[cabinet_0]
  Action: Explore[kitchen_1]
✓ Navigation SUCCESS: Moving to target 'table_0'
"""
    actions, _ = console_actions(text, "vlm_tamp_images")
    assert actions == [
        ("Navigate", "cabinet_0"),
        ("Open", "cabinet_0"),
        ("Explore", "kitchen_1"),
    ]


def test_timeout_tru_counts_are_explicit_lower_bounds():
    actions, coverage = console_actions(
        "[DEBUG] - agent_uid=0: high_level_action=('Explore', 'kitchen_1', None), action_response='Successful execution!'",
        "tru_pomdp_no_images",
    )
    assert actions == [("Explore", "kitchen_1")]
    assert coverage.startswith("Lower bound")


def test_sim_steps_use_counter_summary_not_action_counts():
    text = """Action Counts (times selected) for instruction:
  Navigate: 2
---------------------------------
Action Simulation Steps (total steps per action) for instruction:
  Navigate: 301
  Pick: 2
---------------------------------
"""
    assert summary_step_count(text) == 303
    assert summary_step_count("Run killed without final logs") is None
