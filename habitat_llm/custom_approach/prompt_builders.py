"""Prompt builder helpers for the custom approach planner."""

from typing import Any, Dict, List, Optional

from habitat_llm.custom_approach.custom_constants import (  # pyright: ignore[reportMissingImports]
    CUSTOM_ACTION_NAMES,
)
from .prompt_templates_vlm_tamp_react import (
    SUBGOAL_TO_PDDL_GOALS_PROMPT,
    TASK_TO_OBJECT_SUBGOALS_PROMPT,
)


def _format_objects(objects_by_type: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key in (
        "movable",
        "surface_furniture",
        "container_furniture",
        "joint",
        "room",
        "faucet_furniture",
    ):
        vals = objects_by_type.get(key, [])
        if isinstance(vals, list) and vals:
            lines.append(f"{key}: {', '.join(vals)}")
    if not lines:
        return "(none)"
    return "\n".join(lines)


def build_english_subgoal_prompt(
    goal: str,
    objects_by_type: Dict[str, Any],
    scene_description: str,
    history: str = "",
    after_explore: bool = False,
    explored_room: Optional[str] = None,
) -> str:
    prompt = TASK_TO_OBJECT_SUBGOALS_PROMPT.format(high_level_task=goal, action_names=CUSTOM_ACTION_NAMES)
    extra = [
        "",
        "Additional Context:",
        "Objects:",
        _format_objects(objects_by_type),
        "",
        "Scene graph snapshot:",
        scene_description or "(empty)",
    ]
    if history:
        extra.extend(["", "History:", history])
    if after_explore:
        room_text = explored_room if explored_room else "unknown_room"
        extra.extend(
            [
                "",
                f"Exploration update: room `{room_text}` was just explored.",
                "Re-plan subgoals from latest observations and avoid repeating the same exploration unless necessary.",
            ]
        )
    return prompt + "\n" + "\n".join(extra)


def build_predicate_translation_prompt(
    objects_by_type: Dict[str, Any],
    num_branches: int = 1,
    after_explore: bool = False,
    high_level_task: Optional[str] = None,
    sub_goal: Optional[str] = None,
    scene_description: Optional[str] = None,
    action_history: Optional[str] = None,
    needed_objects: Optional[List[str]] = None,
    replan_needed: bool = False,
    latest_failure: Optional[str] = None,
) -> str:
    sub_goal_text = sub_goal or "Translate each current subgoal from the previous turn."
    if needed_objects:
        sub_goal_text += "\nNeeded objects: " + ", ".join(needed_objects)

    prompt = SUBGOAL_TO_PDDL_GOALS_PROMPT.format(
        high_level_task=high_level_task or "Use the high-level task from the previous turn.",
        sub_goal=sub_goal_text,
        objects_by_type=_format_objects(objects_by_type),
        scene_graph=scene_description
        or "Use the latest scene graph context from the previous turn.",
    )
    if action_history:
        prompt += "\n\n# Action History\n" + action_history
    prompt += "\n\n# Replan Context\n"
    if replan_needed:
        prompt += "Replan needed: True"
        if latest_failure:
            prompt += f"\nLatest failure: {latest_failure}"
    else:
        prompt += "Replan needed: False"
    if num_branches > 1:
        prompt += (
            "\n\nIf possible, return "
            f"{num_branches} alternative plans as a Python list of lists."
        )
    if after_explore:
        prompt += (
            "\n\nPost-exploration rule: prioritize non-exploration predicates if "
            "the object is now known in the latest scene graph."
        )
    return prompt


def build_failure_history(
    actions: List[str],
    failure: Optional[str] = None,
    already_succeeded: Optional[List[str]] = None,
    explore_room: Optional[str] = None,
) -> str:
    lines: List[str] = []
    if actions:
        lines.append("Actions so far:")
        lines.extend(actions)
    if explore_room:
        lines.append(f"Exploration completed for room: {explore_room}")
    if failure:
        lines.append(f"Failure: {failure}")
    if already_succeeded:
        lines.append("Already succeeded subgoals:")
        lines.extend(already_succeeded)
    return "\n".join(lines)
