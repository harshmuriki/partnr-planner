"""Code-ready prompt templates for the custom TAMP hierarchy.

Usage:
    prompt = TASK_TO_OBJECT_SUBGOALS_PROMPT.format(
        high_level_task="Heat bread and place it in the bedroom."
    )
"""

TASK_TO_OBJECT_SUBGOALS_PROMPT = """# Role and Objective
Break a high-level household task into concise, actionable object-scoped subgoals.

# Instructions
- Output only a Python list of strings.
- Each subgoal must focus on one primary manipulated object.
- Include all actions needed for that object in the same subgoal.
- If multiple objects of the same type must be handled separately, create separate subgoals unless the task explicitly gives a count.
- Do not write natural-language object references as "<object> 1", "<object> 2", etc. Use "a <object>" or "another <object>" unless the exact scene graph name with an underscore is known, such as bottle_3.
- Treat names with spaces plus numbers, such as "bottle 1", as task item labels/counts, not exact object names.
- Preserve ordering constraints from the original task.
- Keep each subgoal concise.
- Do not output markdown, explanations, or any text outside the list.

# Input
High-level task: {high_level_task}

# Output Format
["<subgoal 1>", "<subgoal 2>"]
"""


SUBGOAL_EXPLORATION_PROMPT = """# Role and Objective
Generate the next exploration action to find the object needed for the current subgoal, or return [] if the object is already present in the scene graph.

# Inputs
High-level task: {high_level_task}

Current subgoal: {sub_goal}

Available actions: {actions}

Scene graph: {scene_graph}

Last action run and result: {last_action_result}

Already completed object instances: {completed_objects}

Action history: {exploration_history}

# Instructions
- First search in likely rooms and furniture items from the scene graph (based on the current subgoal).
- If any object matching the requested type is already visible in the Scene graph, stop exploration immediately and return that exact object name. Do not keep searching for another or better instance.
- For repeated-object tasks like "another bottle", choose a matching visible object that is not in Already completed object instances and has not already been successfully picked/filled/placed according to Action history. Do not choose objects already completed in earlier subgoals.
- Do not repeat an action that already failed or an explore/search action that already ran unless the scene graph changed.
- The scene is static. Do not Explore the same room more than once; repeat room exploration only if the goal is to inspect inside furniture that was not previously opened or inspected.
- If all likely rooms have already been explored and the needed object is still missing, inspect inside the most likely container furniture for that object type. For example, cups/bottles/jugs are likely inside cupboards, cabinets, shelves, or drawers; clothes/towels are likely inside wardrobes, dressers, or drawers. Choose the single most likely unopened furniture from the scene graph.
- After a decent amount of exploration, if the exact requested object still is not found, stop searching for that exact instance and choose the next best available object of the same type from the scene graph. Prefer objects in plausible locations and objects not listed in Already completed object instances.
- If the subgoal says "<object> 1", "<object> 2", etc. with a space before the number, search for the base object type ("<object>"). The number is a natural-language label, not an exact scene graph name.
- Exact scene graph object names use underscores, such as bottle_3. If any matching exact object name is present, return that exact scene graph name.
- This prompt will run repeatedly until the object is found and it will always output only the next most probable action to take.

# Action Instructions
- Use only actions that help locate the target object, such as Explore, Navigate, Open, or Look/Inspect if available.
- Do not output object manipulation actions such as Pick, Place, Clean, Pour, Fill, PowerOn, or PowerOff.
- You can only Open articulated furniture listed as joints in the scene context. Do not Open non-articulated furniture such as normal tables, shelves, counters, beds, couches, or stools.
- Open only works when the robot is already near the target furniture. If the robot is not already next to that furniture, output Navigate[target_furniture] first, then Open[target_furniture] in a later step.
- Never navigate to receptacle nodes whose names start with "rec_". For navigation, use the parent furniture/object name instead; for example use Navigate[table_11], not Navigate[rec_table_11_0].
- You can only explore rooms with the exact name from the scene graph.
- When outputting an exploration action, also include one short reason string explaining why that action is the best next search step. Format it as "Reason: <one sentence>".
- Output only a Python list containing action-call strings and, when an action is returned, one reason string.
- Do not output JSON objects, markdown, explanations, or any text outside the list.

# Important
- If the target object type is already present in the scene graph, return the list of exact object names needed to complete the subgoal. For example, if the subgoal needs a bottle and the scene graph contains bottle_3, return ["bottle_3"] instead of exploring/opening more rooms or furniture.

# Output Format
If more exploration is needed:
["Explore[kitchen]", "Reason: kitchen is a likely room for the target object and has not been explored yet."]

If the needed objects are present:
["<object_name_1>", "<object_name_2>", ...]
"""

# not used
SUBGOAL_TO_VLM_ACTIONS_PROMPT = """# Role and Objective
Generate executable high-level actions for the current object-scoped subgoal.

# Inputs
High-level task: {high_level_task}

Current subgoal: {sub_goal}

Available actions: {actions}

Scene graph: {scene_graph}

Action history:
{action_history}

Replan needed: {replan_needed}

# Instructions
- Output the shortest valid action sequence that completes the entire subgoal.
- Do not invent object names. Use exact scene graph names.
- If the subgoal says "<object> 1", "<object> 2", etc. with a space before the number, treat it as a generic requested object of that type. Choose an exact matching scene graph name such as bottle_3; do not look for a literal object named "bottle 1".
- Do not include exploration if the required object is already in the scene graph.
- If the required object is missing, call "Explore[None] to re-start the exploration phase".
- During cleaning actions, the agent must be near a faucet.
- For Fill actions, the agent only needs to be near faucet furniture; do not open, close, or manipulate the faucet furniture unless the task separately requires opening it.
- If the exact object name isn't found, use the most like object name from the scene graph.

# Action Instructions
- Use only actions from Available actions.
- Include required navigation, open/close, pick/place, state-change, clean, fill, or pour actions in an order.
- You can only Open articulated furniture listed as joints in the scene context. Do not Open non-articulated furniture such as normal tables, shelves, counters, beds, couches, or stools.
- Open only works when the robot is already near the target furniture. If the robot is not already next to that furniture, output Navigate[target_furniture] before Open[target_furniture].
- Never navigate to receptacle nodes whose names start with "rec_". For navigation, use the parent furniture/object name instead; for example use Navigate[table_11], not Navigate[rec_table_11_0].
- For navigate action, use the actual object or furniture name, not a receptacle name.
- Output only a Python list of action-call strings.
- Do not output JSON objects, markdown, explanations, or any text outside the list.

# Output Format
["Navigate[bread_1]", "Pick[bread_1]", "Navigate[microwave_1]", "Place[bread_1,microwave_1]"]
"""


SUBGOAL_TO_PDDL_GOALS_PROMPT = """# Role and Objective
Translate the current object-scoped subgoal into formal predicates for PDDLStream planning.

# Inputs
High-level task: {high_level_task}

Current subgoal: {sub_goal}

Scene graph: {scene_graph}

# Predicate Set
- picked(<movable>)
- on(<movable>, <surface_furniture>)
- in(<movable>, <container_furniture>)
- opened-door(<joint>)
- closed-door(<joint>)
- opened-drawer(<joint>)
- closed-drawer(<joint>)
- powered_on(<appliance>)
- powered_off(<appliance>)
- filled(<movable>)
- poured-into(<movable>)
- cleaned(<movable_or_furniture>)

# Instructions
- Return only a Python list of predicate strings.
- Use exact object and furniture names from the Scene graph.
- If the subgoal says "<object> 1", "<object> 2", etc. with a space before the number, treat it as a generic requested object of that type. Choose an exact matching object from the Scene graph such as bottle_3; do not look for a literal object named "bottle 1".
- Only emit opened-door/closed-door/opened-drawer/closed-drawer predicates for articulated furniture that the Scene graph marks as open or closed.
- Open predicates require the robot to be near that furniture at execution time; do not use open predicates as a substitute for exploration or navigation.
- For filling an object, emit filled(<movable>) only. Do not emit open/close predicates for faucet furniture just because it is used for filling; the planner only needs the agent near faucet furniture to fill.
- If the required object is missing and exploration has not yet identified a substitute, return [] and let the exploration phase run.
- If exploration selected a next-best substitute object because the exact requested object was not found after enough searching, use that substitute's exact scene graph name in all predicates.
- Keep predicates in execution order.
- Do not output markdown, explanations, or any text outside the list.

# Output Format
["picked(bread_1)", "in(bread_1,microwave_1)"]
"""
