# T2-INC-ABS

## Base task
Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them.

## Scene
- scene_id: 106878960_174887073
- rooms used: entryway/foyer/lobby_0, living_room_0
- furniture used (id — room — catalog description):
  - chest_of_drawers_2 — entryway/foyer/lobby_0 — Bombay Two Drawer Chest, Black
  - table_2 — living_room_0 — Madison Park Signature Bordeaux Coffee Table

## Task instruction / prompt given
"Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them."

## Affected object(s)
- box: box_0 is the selected uncertainty target omitted from memory; box_1 is the remaining task object retained in memory. Both physically exist.
- scissors: scissors_0 is the absent requested uncertainty target, not spawned.
- No knife substitute, other acceptable cutting tool, or distractors exist.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: No Suitable Object

## Information supplied in instruction
- Mentions two packages, entryway source, living-room center table destination, mutual adjacency, and scissors beside the packages.
- Omits tool unavailability, source furniture, exact floor positions, instance ids, and states. Memory does not supply a reliable cutting-tool availability conclusion.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0 is absent; no suitable cutting tool exists, including on chest_of_drawers_2.

## Final expected world state
- box_0: on table_2 (living_room_0).
- box_1: on table_2 (living_room_0), next to box_0.
- No cutting tool is fabricated or delivered. The tool-delivery subgoal is abandoned as infeasible.

## Initial robot memory
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- No selected-target records or location associations are supplied; cutting-tool availability is unknown rather than known absent.

## Success criteria
- After checking availability, give up on the unavailable cutting-tool subgoal; do not invent tools or claim complete fulfillment.
- Feasible package subgoals: is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- Feasible arrangement: is_next_to(box_0, box_1), with both on table_2.
- No scissors or substitute placement proposition is required.

## Spawn / planner notes
- Spawn only box ×2 on the entryway floor. Spawn no scissors, knife, or other acceptable cutting tool.
- chest_of_drawers_2's top is the mapped entryway-table search location; table_2 is the center table.
- Start in entryway/foyer/lobby_0. Discover the unremembered box, complete feasible moves, and terminate the tool search after establishing unavailability rather than looping on missing memory.
