# Custom TAMP Approach

This directory captures the planned prompt hierarchy for a custom TAMP method.

## Goal

Build a hierarchical prompting flow that turns a human household instruction into object-scoped subgoals, explores when the active object is missing, and then generates actions once the object is present in the scene graph.

## High-Level Flow

```text
Human task
  -> Task-to-subgoals prompt
  -> For each object-scoped subgoal:
      -> If object is missing or unlocalized:
          -> Exploration prompt
          -> Update scene graph
      -> If object is present:
          -> Subgoal-to-actions prompt
          -> Execute generated actions
```

## Phases

### 1. Task To Object-Scoped Subgoals

Break the human task into concise, actionable subgoals. Each subgoal should focus on one primary manipulated object and include all actions needed for that object.

Example:

```text
Input:
Heat some bread and place it in the bedroom you started at. Wash the bottles (3) and place a towel next to them. And turn off the lights in the room you took the bread to and clean the towel.

Output:
1. Heat bread and place in bedroom.
2. Wash 3 bottles.
3. Place a towel next to the bottles.
4. Turn off lights in the room with bread.
5. Clean towel.
```

### 2. Exploration For Missing Objects

When the object required by the current subgoal is not found in the scene graph, generate only exploration/search actions.

The exploration phase should:

- Search likely rooms, furniture, and containers.
- Use actions such as `Explore`, `Navigate`, `Open`, or `Inspect` if available.
- Stop once the target object is found or localized.
- Avoid manipulation actions like `Pick`, `Place`, `Clean`, `Pour`, `Fill`, `PowerOn`, or `PowerOff`.

### 3. Subgoal To Actions

Once the object for the current subgoal is present in the scene graph, generate actual execution actions.

Inputs:

- High-level task: `{high_level_task}`
- Current subgoal: `{sub_goal}`
- Available actions: `{actions}`
- Scene graph: `{scene_graph}`
- Action history: `{action_history}`

## Backend Flag

Plan for a later implementation flag:

```yaml
subgoal_action_backend: vlm  # vlm | pddlstream
```

Backend behavior:

- `vlm`: task -> object subgoals -> exploration if needed -> direct high-level actions.
- `pddlstream`: task -> object subgoals -> exploration if needed -> formal predicates -> existing PDDLStream planner.

Exploration should remain direct in both modes because it is a discovery phase rather than the final object manipulation planner.

## Save Location

New custom prompt artifacts and design notes should live under `custom_approach/`.

Existing planner files are references for later integration only:

- Existing prompt definitions: `habitat_llm/vlm_tamp/prompts_vlm_tamp.py`
- Existing VLM-TAMP planner: `habitat_llm/planner/vlm_tamp_pddl_planner.py`
- Existing runtime subgoal support: `habitat_llm/examples/planner_demo.py`

See also:

- `custom_approach/prompt_templates.md` for finalized prompt text and parser contract.
- `custom_approach/config_spec.md` for planned backend flag shape.
- `custom_approach/dry_run_results.md` for manual validation runs.
