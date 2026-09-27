"""VLM-TAMP and PARTNR ReAct style prompts for the custom approach planner."""

# TASK_TO_OBJECT_SUBGOALS_PROMPT_OLD = """# Role and Objective
# Plan a short sequence of object-scoped intermediate goals that accomplishes a high-level household task.

# # Instructions
# - Output only a Python list of strings.
# - Write each subgoal as detailed with all the actions/states that the object must be in to complete the task but simple English instructions.
# - Each subgoal must focus on one primary manipulated object.
# - Use only objects and furniture visible in the provided object lists or scene graph.
# - Never invent new object IDs or names.
# - If the task says "<object> 1", "<object> 2", etc. with a space before the number, treat that as a task label/count, not an exact scene graph name. Use "a <object>" or "another <object>" unless an exact underscore name such as bottle_3 is already known.
# - Preserve ordering constraints from the original task.
# - Do not output markdown, explanations, or any text outside the list.

# # Input
# High-level task: {high_level_task}

# # Actions

# # Output Format
# ["<subgoal 1>", "<subgoal 2>"]
# """

TASK_TO_OBJECT_SUBGOALS_PROMPT = """# Role and Objective
Break a high-level household task into concise, actionable object-scoped subgoals.

# Instructions
- Output only a Python list of strings.
- Each subgoal must focus on one unique primary manipulated object, that is not already in the list of already completed object instances.
- Include all actions needed for that object in the same subgoal (start with Explore if needed).
- If multiple objects of the same type must be handled separately, create separate subgoals unless the task explicitly gives a count.
- Do not write natural-language object references as "<object> 1", "<object> 2", etc. Use "a <object>" or "another <object>" unless the exact scene graph name with an underscore is known, such as bottle_3.
- Treat names with spaces plus numbers, such as "bottle 1", as task item labels/counts, not exact object names.
- Preserve ordering constraints from the original task.
- Keep each subgoal concise.
- Do not output markdown, explanations, or any text outside the list.

# Input
High-level task: {high_level_task}

# Actions:
    {action_names}

# Output Format
["<subgoal 1>", "<subgoal 2>"]
"""


SUBGOAL_EXPLORATION_PROMPT ="""
You are an agent that solves household planning problems by choosing the next exploration action.
The current task is situated in a house and may require navigating to rooms, inspecting furniture, and finding objects before PDDL planning can manipulate them.

Task: {high_level_task}

Current object-scoped subgoal: {sub_goal}

Scene graph:
{scene_graph}

Possible exploration actions:
{actions}

Last action run and result: {last_action_result}

Already completed object instances from the task: {completed_objects}

Action history:
{exploration_history}

Rules:
1. Rooms do not need to be explored more than once.
2. If you do not find the target object outside, check appropriate articulated furniture.
3. Objects may be inside articulated furniture, so inspect cabinets, drawers, wardrobes, fridges, or similar furniture when likely for the object type.
4. Many calls to the same action in a row are a sign something has gone wrong. Try a different useful action.
5. Pay attention to previous actions to avoid repeating mistakes.
6. If an object is not visible, it may be in a room, on top of a surface, or inside articulated furniture.
7. If container furniture contents are not visible, include opening its door/drawer before picking from it or placing into it.

Custom grounding rules:
1. If any object matching the requested type is already present in the scene graph, stop exploration and return the exact object name.
2. For repeated-object tasks like "another bottle", choose a matching object that is not in Already completed object instances and was not already successfully picked, filled, placed, cleaned, powered, or otherwise completed in Action history.
3. Do not output literal task labels such as 1_bottle, 1_jug, or "bottle 1" unless that exact underscore object name appears in the scene graph.
4. If the subgoal says "<object> 1", "<object> 2", etc. with a space before the number, search for the base object type.
5. If a room was already explored and the scene graph did not change, do not explore that same room again. Explore a different likely room or inspect likely unopened articulated furniture.
6. After exploring most rooms and inside of the most likely furniture, if the exact requested object still is not found, choose the next best available object of the same type from the scene graph.
7. Open only works when the robot is already near the target furniture. If not already near it, output Navigate[target_furniture] first and Open[target_furniture] later.
8. Never navigate to receptacle nodes whose names start with rec_. Use the parent furniture/object name instead.
9. You can only explore rooms with exact room names from the scene graph.
10. Action calls in this custom planner do not include the agent index inside brackets. Use Navigate[fridge_57], not Navigate[0 fridge_57]
11. For fill tasks, move the container near faucet furniture and use fill.
12. For cleaning movable objects, pick the object, move it near faucet furniture, and clean it. For furniture surfaces, navigate to the furniture and clean it directly.

Return your response as a Python list only.
If more exploration is needed, return one action string and one short reason string:
["Explore[kitchen_1]", "Reason: kitchen_1 is a likely room for the target object and has not been explored yet."]

If the needed object is present, return exact object names only:
["bottle_3"]
"""


SUBGOAL_TO_PDDL_GOALS_PROMPT = """Translate the current object-scoped subgoal into a formal language defined by the following PDDLStream subgoals.

Task: {high_level_task}

Current subgoal: {sub_goal}

Typed objects:
{objects_by_type}

Scene graph:
{scene_graph}

subgoals =
[
"picked(<movable>)": the result of picking up <movable>, it contains one argument.
"on(<movable>, <surface_furniture>)": the result of picking up <movable> and placing it on <surface_furniture>, it contains two arguments.
"in(<movable>, <container_furniture>)": the result of picking up <movable> and placing it inside <container_furniture>, it contains two arguments.
"opened-door(<joint>)": the result of opening a door joint, it contains one argument.
"closed-door(<joint>)": the result of closing a door joint, it contains one argument.
"opened-drawer(<joint>)": the result of opening a drawer joint, it contains one argument.
"closed-drawer(<joint>)": the result of closing a drawer joint, it contains one argument.
"powered_on(<appliance>)": the result of turning on <appliance>, it contains one argument.
"powered_off(<appliance>)": the result of turning off <appliance>, it contains one argument.
"filled(<movable>)": the result of filling <movable> with liquid, it contains one argument.
"poured-into(<movable>)": the result of pouring liquid into <movable>, it contains one argument.
"cleaned(<movable_or_furniture>)": the result of cleaning <movable_or_furniture>, it contains one argument.
]

IMPORTANT type rules:
1. picked, on, in, filled, poured-into: the movable argument must be a name from the movable list or an exact movable object visible in the scene graph.
2. on(<movable>, <surface_furniture>): the second argument must be a surface furniture name. Never use a movable object or receptacle id as the second argument.
3. in(<movable>, <container_furniture>): the second argument must be a container furniture name. Never use a movable object or receptacle id as the second argument.
4. opened-door, closed-door, opened-drawer, closed-drawer: the argument must be a joint/articulated furniture name only.
5. Never emit predicates with names starting with rec_.
6. Never invent object names such as package_1, item_0, 1_bottle, 1_jug, or 1_scissors.
7. If the task uses labels like "bottle 1" but that exact object is not in the scene graph, choose a real matching scene graph object found by exploration.

Navigation-only and approach milestones: do NOT respond with [] when typed room names and/or Needed-objects lines clearly identify where to navigate. Prefer at(<ignored_stub>,<exact_room>) for pure room-movement milestones. When the English subgoal is about moving toward furniture or staging near an object listed under "Needed objects" without manipulating it yet (navigate/reposition while that object is still on the floor or on furniture), emit picked(<exact_movable_name>) so PDDL can schedule navigation toward that graspable object.

8. If exploration selected a next-best same-type object because the exact requested object was not found after enough searching, use that substitute object's exact scene graph name.
9. For Fill, emit filled(<movable>) only. Do not add open/close predicates for faucet furniture just because it is used for filling.
10. Keep subgoals in the same order as the current subgoal's intended steps.
11. If one step cannot be translated into a valid predicate, skip that step instead of inventing arguments.

Return [] only when a required movable, room, joint, or furniture name truly cannot be inferred from the scene graph, objects_by_type lists, Needed objects lines, or the current subgoal text.

Return only a Python list of predicate strings. Do not output markdown or explanations.

Output format:
["picked(bread_1)", "in(bread_1,microwave_1)"]
"""


# not used
SUBGOAL_TO_VLM_ACTIONS_PROMPT = """Plan a short sequence of executable high-level actions that completes the current object-scoped subgoal.

    Task: {high_level_task}

    Current subgoal: {sub_goal}

    Available actions:
    {actions}

    Scene graph:
    {scene_graph}

    Action history:
    {action_history}

    Replan needed: {replan_needed}

    You are a mobile robot with one arm. Follow these rules:
    1. Use exact object and furniture names from the scene graph.
    2. Never invent new object IDs or names.
    3. If the subgoal uses a task label such as "bottle 1", search for the base object type and choose an exact scene graph name such as bottle_3.
    4. Do not use receptacle ids whose names start with rec_ as navigation or placement targets. Use parent furniture names.
    5. You can only open articulated furniture listed as joints in the scene context.
    6. If opening furniture, navigate to that furniture before Open.
    7. For Fill, the agent only needs to be near faucet furniture. Do not open or close faucet furniture unless the task separately requires it.
    8. If the required object is missing, return ["Explore[None]"] so the planner returns to exploration.
    9. Prefer the shortest valid action sequence.
    10. Do not include the agent index in action arguments. Use Place[jug_1, on, table_15, None, None], never Place[0 jug_1, on, table_15, None, None].

    Output only a Python list of action-call strings.

    Output format:
    ["Navigate[bread_1]", "Pick[bread_1]", "Navigate[microwave_1]", "Place[jug_1, on, table_15, None, None]"]
    """
