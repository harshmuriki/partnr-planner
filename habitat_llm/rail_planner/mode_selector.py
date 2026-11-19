"""Mode selector for RailPlanner's two_phase mode (plan_config.two_phase=true).

At every replan point the VLM chooses between
  explore:  a ReAct exploration phase (react_explorer.ExploreReactPlanner) that only searches, or
  interact: the usual RailPlanner subgoal loop (VLM Turn 1/2 + expanders), which no longer emits explore().
Uses only the robot's own memory, history and camera: no privileged information.
"""

import json
import re
from typing import List, Tuple

MODES = ("explore", "interact")

SELECTOR_PROMPT = """You control a mobile household robot with one arm. Decide what the robot should do next for this task:
``{goal}''

The robot works in two modes:
- explore: search the house to find objects and furniture the task needs. The robot moves between rooms and furniture and
  opens and closes doors or drawers, step by step, but does not pick, place or change any object.
- interact: plan and execute the manipulation subgoals (pick, place, open, close, power, fill, clean, wait) with the
  objects and furniture the robot already knows about.

Choose explore when the task needs objects or furniture that are not in the robot's memory, when the remembered location
of a needed object may be wrong (a previous attempt could not find or reach it), or when a container should be searched
first. Choose interact when the known objects are enough to make progress on the task. Not seeing an object does not
prove it is absent, but do not search the same places again without a reason.

Robot memory (world graph; facts may be outdated):
``{observed}''

Why a decision is needed now: {reason}
{history}{explorations}Exploration phases left: {phases_left}.

Return exactly one JSON object and nothing else:
{{"mode": "explore" or "interact", "reason": "<one sentence>", "explore_goal": "<if explore: what to find and where to look first; otherwise empty>"}}
"""

MAX_HISTORY_LINES = 15


def build_selector_prompt(
    goal: str,
    scene_description: str,
    reason: str,
    history: List[str],
    explore_summaries: List[str],
    phases_left: int,
) -> str:
    recent = [str(h) for h in history[-MAX_HISTORY_LINES:]]
    history_block = (
        "Recent actions and failures (oldest first):\n" + "".join(f"- {h}\n" for h in recent) + "\n" if recent else ""
    )
    explore_block = (
        "Earlier exploration phases:\n" + "".join(f"- {s}\n" for s in explore_summaries) + "\n"
        if explore_summaries else ""
    )
    return SELECTOR_PROMPT.format(
        goal=goal,
        observed=scene_description,
        reason=reason,
        history=history_block,
        explorations=explore_block,
        phases_left=phases_left,
    )


def parse_selector_response(text) -> Tuple[str, str, str, bool]:
    """(mode, reason, explore_goal, parsed). Anything unparsable falls back to interact."""
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = None
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            data = None
    if not isinstance(data, dict):
        return "interact", "unparsable selector response", "", False
    mode = str(data.get("mode", "")).strip().lower()
    if mode not in MODES:
        return "interact", f"unknown mode {mode!r}", "", False
    return mode, str(data.get("reason", "")), str(data.get("explore_goal", "") or ""), True
