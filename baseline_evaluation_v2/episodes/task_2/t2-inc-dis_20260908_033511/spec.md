# T2-INC-DIS

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
- box: box_0 is the uncertainty target omitted from memory; box_1 is the remaining task object.
- scissors: scissors_0 is the other uncertainty target omitted from memory.
- pencil_case: pencil_case_0 is a distractor near box_0.
- pen: pen_0 and screwdriver: screwdriver_0 are distractors beside scissors_0, not substitutes. Distractor object records remain in memory.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Distractors: Present

## Information supplied in instruction
- Mentions two packages, entryway source, living-room center table destination, mutual adjacency, and scissors beside the packages.
- Omits distractors, source furniture, exact floor positions, instance ids, and states.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.
- pencil_case_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable, near box_0.
- pen_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable, next to scissors_0.
- screwdriver_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable, next to scissors_0.

## Final expected world state
- box_0 and box_1: on table_2 (living_room_0), next to each other.
- scissors_0: on table_2 (living_room_0), next to both boxes.
- pencil_case_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), unchanged position.
- pen_0 and screwdriver_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), unchanged positions.

## Initial robot memory
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- pencil_case_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- pen_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.
- screwdriver_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.
- Selected-target records and all relations referencing those omitted targets are absent; distractor support locations remain accurate.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(scissors_0, table_2).
- is_next_to(scissors_0, box_0) and is_next_to(scissors_0, box_1), all on table_2.
- is_on_top(pen_0, chest_of_drawers_2) and is_on_top(screwdriver_0, chest_of_drawers_2).
- in_room(pencil_case_0, entryway/foyer/lobby_0); retain its original floor placement.

## Spawn / planner notes
- Spawn box ×2, scissors ×1, pencil_case ×1, pen ×1, screwdriver ×1 at the listed anchors, preserving near-target placement without overlap.
- chest_of_drawers_2's top maps the entryway table; table_2 maps the center table. Start in entryway/foyer/lobby_0.
- Discover the missing targets among remembered distractors; do not deliver a pencil case instead of a box or a pen/screwdriver instead of scissors.
