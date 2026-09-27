
# ---------------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------------

from typing import Optional

# Keep observation text short enough for stable prompt sizes.
MAX_OBSERVED_CHARS = 12000

# ---------------------------------------------------------------------------
# Turn 1: English subgoals  (matches prompt_subgoals_english from kitchen-worlds)
# ---------------------------------------------------------------------------

# Base template — identical wording to kitchen-worlds' prompt_planning
_PROMPT_PLANNING = """Plan a short sequence of [OUTPUT] that accomplishes the following goal: 
``{goal}''. 
[RESPOND_WITH]
where <movable>, <surface_furniture>, <container_furniture>, <joint> and <appliance> must be items from the following list:
{objects}.

This observed state is your initial state of the world. You should use it to plan your actions. It can have accurate, missing or outdated information.
World-graph memory (the same scene description available to PARTNR; facts may be outdated and are not necessarily visible in the attached images):
``{observed}''
{history}
You are a mobile robot with one arm. You must obey the following commonsense rules:
1. You must have at least one empty hand before you can pick up an object or open or close a joint.
2. When you sprinkle or pour something into a container, there must not be objects placed on top of the container.
3. You may plan for remembered objects outside the current view. Their location may be outdated; navigate and verify before manipulation. Unknown is not absent.
4. If you don't see a particular object, it may be somewhere in the room on top of a surface or inside a furniture. Use the explore action to search for it.
5. If you cannot see an object, it may be inside a furniture, open that furniture to find it.
6. If you cannot see the inside of a container furniture, you must open its door or drawer before you can pick objects from it or place objects inside it.
7. Never invent new object IDs or names. If the goal mentions a generic object (e.g., "package") and no exact matching name exists in the provided list, add explore goals first and wait for updated observations before naming concrete objects.
8. If you have to fill a container, you must take it next to a faucet and use the fill action.
9. If you have to clean an object, pick it up, take it to a faucet, and use the clean action (similar to filling). For furniture surfaces, navigate to the furniture and use the clean action directly.
10. Search for objects in the usual places and give up if you don't find them.
The accompanying images show a household scene with a robot, including annotated object names and corresponding bounding boxes on the images.

Observation and replanning protocol:
- Explore(room) gathers observations and triggers replanning; remaining subgoals are replaced.
- Opening furniture triggers inspection of newly exposed contents and replanning. End the current plan at that information boundary; do not invent unseen contents.
- Execution failures first receive local PDDL retries. Exhausted retries/branches trigger replanning with a current camera image and failure details.
- Once all subgoals in the active plan are finished, execution stops without another VLM request. Propose goals that cover the full instruction, unless Explore or Open will trigger new observations and replanning.
- At a terminal decision, return exactly one JSON object: {{"decision":"complete","reason":"evidence that every requested condition holds"}} or {{"decision":"unsolvable","reason":"what remains unknown or impossible and searches attempted"}}. These are your judgments, not evaluator verdicts.
- Otherwise return the next intermediate goals. Use only known IDs; keep missing locations unknown. Not seeing an object does not prove absence. Respect reported search limits; do not repeatedly search unchanged places.
"""

# Appended to Turn 1 only after a high-level Explore[room] completes and the planner re-queries the VLM.
_POST_EXPLORE_ENGLISH_APPENDIX = """
--- Context: a room exploration just finished ({room_context}) ---
You now have newer observations from that exploration. Plan the next steps with these rules:
1. Do not propose explore(<room>) again for the same room if the current observation already gives you the objects and relationships you need for the task goal.
2. If you need to see inside cabinets, drawers, or other furniture, do not re-explore the whole room: plan to go to that furniture and use opening steps (open the relevant door/drawer joint) before looking for objects inside.
3. Only use explore(<room>) for a room you already explored if you still believe important objects may be missing there after the last pass.
4. If something is still missing, you may explore a different room instead.
5. If you believe you have all objects and information needed for the goal, do not include any explore(...) steps—continue with pick, place, open, and other task actions only.
6. All subgoals after the explore action will not be run and the planner will re-plan.
"""

# Optional suffix for Turn 2 after explore (reinforces formal subgoals).
_POST_EXPLORE_PREDICATE_SUFFIX = """

Post-exploration rules for subgoals: avoid explore(<room>) for a room that was just searched unless still necessary; prefer opened-door/opened-drawer on specific joints to inspect furniture interiors; omit explore entirely if the observation is sufficient.
"""

# Turn 1 prompt — identical construction to prompt_subgoals_english
SUBGOAL_ENGLISH_PROMPT = (
    _PROMPT_PLANNING
    .replace("[OUTPUT]", "intermediate goals")
    .replace("<movable>, <surface>, <space>, <joint> and <appliance>", "objects mentioned")
    .replace("[RESPOND_WITH]", """
Respond with detailed but simple instructions in English. Each line must consists of only one intermediate goal where objects mentioned must be items from the following list:, """)
)

# ---------------------------------------------------------------------------
# Turn 2: PDDL predicate translation  (matches prompt_english_to_subgoals) THESE ARE THE SUBGOALS (NOT Actions)
# ---------------------------------------------------------------------------

ENGLISH_TO_PREDICATE_PROMPT = """Translate the above intermediate goals into a formal language defined by the following subgoals.

subgoals = 
[
'picked(<movable>)': the result of picking up <movable>, it contains one argument.
'on(<movable>, <surface_furniture>)': the result of picking up <movable> and placing it on <surface_furniture>, it contains two arguments.
'in(<movable>, <container_furniture>)': the result of picking up <movable> and placing it inside <container_furniture>, it contains two arguments.
'explore(<room>)': the result of exploring a room to discover objects and relationships, it contains one argument.
'opened-door(<joint>)': the result of opening <joint>, it contains one argument.
'closed-door(<joint>)': the result of closing <joint>, it contains one argument.
'opened-drawer(<joint>)': the result of opening <joint> (a drawer), it contains one argument.
'closed-drawer(<joint>)': the result of closing <joint> (a drawer), it contains one argument.
'powered_on(<appliance>)': the result of turning on <appliance>, it contains one argument.
'powered_off(<appliance>)': the result of turning off <appliance>, it contains one argument.
'filled(<movable>)': the result of filling <movable> with liquid, it contains one argument.
'poured-into(<movable>)': the result of pouring liquid into <movable>, it contains one argument.
'cleaned(<movable_or_furniture>)': the result of cleaning <movable_or_furniture>, it contains one argument.
], 

The above subgoals include argument types. Please use the objects in the respective types:
``
{objects}
''

IMPORTANT type rules:
- picked(<movable>), on(<movable>, ...), in(<movable>, ...), filled(<movable>), poured-into(<movable>): <movable> MUST be a name from the movable list.
- on(<movable>, <surface_furniture>): the second argument MUST be a name from the surface_furniture list. NEVER use a movable object or receptacle id as the second argument of on().
- in(<movable>, <container_furniture>): the second argument MUST be a name from the container_furniture list. NEVER use a movable object or receptacle id as the second argument of in().
- explore(<room>): <room> MUST be a name from the room list.
- opened-door(<joint>), closed-door(<joint>), opened-drawer(<joint>), closed-drawer(<joint>): the argument MUST be a name from the joint list only.

Return the subgoals in a list and give no explanation. 
Make sure the sub-goals are in the same order as the steps in the intermediate goals. 
Note that the arguments shouldn't include robot parts, e.g., 'arm', 'gripper'.
Never invent object names (e.g., package_1, item_0). If a required object name is unknown, output explore(<room>) for one or more rooms first, and only use concrete names that exist in the provided lists.
If one intermediate goal cannot be translated into a sub-goal, skip that step.
"""
# If you can think of {num_branches} alternative orderings or plans, return them as a Python list of lists (one inner list per alternative plan). Make each alternative genuinely different in subgoal choice or ordering, not just a copy of the first.


# ---------------------------------------------------------------------------
# Failure history block  (matches include_history from kitchen-worlds exactly)
# ---------------------------------------------------------------------------

INCLUDE_HISTORY = """
You have already taken the following actions written in a formal language:
{actions}

You just failed at planning for {failure}.

"""

INCLUDE_HISTORY_EXPLORE = """
You have already taken the following actions written in a formal language:
{actions}

You just finished exploring {room}. Re-plan based on your new observations.

"""

ALREADY_SUCCEEDED = """
You have already succeeded at achieving the following subgoals:
{already_succeeded}

"""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def _format_objects(objects_by_type: dict) -> str:
    """Format the typed object dict with room-wise furniture context."""
    lines = []

    movable_names = objects_by_type.get("movable", [])
    if movable_names:
        lines.append(f"movable: {', '.join(movable_names)}")

    by_room = objects_by_type.get("furniture_by_room")
    room_objects = objects_by_type.get("objects_by_room")
    if isinstance(by_room, dict) and by_room:
        lines.append("surface_furniture_by_room:")
        for room_name in sorted(by_room):
            furniture_names = list(by_room.get(room_name, []))
            object_names = (
                list(room_objects.get(room_name, []))
                if isinstance(room_objects, dict)
                else []
            )
            merged = []
            seen = set()
            for name in furniture_names + object_names:
                if name in seen:
                    continue
                seen.add(name)
                merged.append(name)
            if merged:
                lines.append(f"{room_name}: {', '.join(merged)}")
    else:
        names = objects_by_type.get("surface_furniture", [])
        if names:
            lines.append(f"surface_furniture: {', '.join(names)}")

    for key, label in [
        ("container_furniture", "container_furniture"),
        ("joint", "joint"),
        ("room", "room"),
    ]:
        names = objects_by_type.get(key, [])
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    faucet_names = objects_by_type.get("faucet_furniture", [])
    if isinstance(faucet_names, list) and faucet_names:
        lines.append("furniture_with_faucet: " + ", ".join(faucet_names))
    return "\n".join(lines) if lines else "(none visible)"


def _truncate_observed_text(text: str, max_chars: int = MAX_OBSERVED_CHARS) -> str:
    if text is None:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def build_english_subgoal_prompt(
    goal: str,
    objects_by_type: dict,
    scene_description: str,
    history: str = "",
    after_explore: bool = False,
    explored_room: Optional[str] = None,
) -> str:
    base = SUBGOAL_ENGLISH_PROMPT.format(
        goal=goal,
        objects=_format_objects(objects_by_type),
        observed=_truncate_observed_text(scene_description),
        history=history,
    )
    if not after_explore:
        return base
    if explored_room:
        room_context = f"you explored room `{explored_room}`"
    else:
        room_context = "a room was fully explored"
    return base + _POST_EXPLORE_ENGLISH_APPENDIX.format(room_context=room_context)


def build_predicate_translation_prompt(
    objects_by_type: dict,
    num_branches: int = 1,
    after_explore: bool = False,
) -> str:
    out = ENGLISH_TO_PREDICATE_PROMPT.format(
        objects=_format_objects(objects_by_type),
        num_branches=num_branches,
    )
    if after_explore:
        out += _POST_EXPLORE_PREDICATE_SUFFIX
    return out


def build_failure_history(
    actions: list,
    failure: Optional[str] = None,
    already_succeeded: Optional[list] = None,
    explore_room: Optional[str] = None,
) -> str:
    actions_str = "\n".join(str(a) for a in actions)
    if explore_room is not None:
        history = INCLUDE_HISTORY_EXPLORE.format(actions=actions_str, room=explore_room)
    else:
        history = INCLUDE_HISTORY.format(actions=actions_str, failure=failure)
    if already_succeeded:
        history += ALREADY_SUCCEEDED.format(
            already_succeeded="\n".join(str(s) for s in already_succeeded)
        )
    return history


# ---------------------------------------------------------------------------
# Legacy single-shot prompt (kept for backward compatibility)
# ---------------------------------------------------------------------------

SUBGOAL_PROMPT_TEMPLATE = """
Plan a short sequence of intermediate subgoals that accomplishes the following goal:
"{goal}"

Return the subgoals as a Python list of strings using ONLY the following formal language:
- picked(<object>)
- on(<object>, <furniture>)
- in(<object>, <furniture>)
- opened-door(<furniture>)
- opened-drawer(<furniture>)
- closed-door(<furniture>)
- closed-drawer(<furniture>)

Use ONLY names from this list:
{objects}

Currently observed:
{observed}

{history}
Return ONLY the Python list and no extra text.
"""


def build_subgoal_prompt(goal, objects, observed, history):
    return SUBGOAL_PROMPT_TEMPLATE.format(
        goal=goal,
        objects=objects,
        observed=_truncate_observed_text(observed),
        history=history or "",
    )
