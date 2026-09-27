# T2-ACC-CAND

## Base task
Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them.

## Scene
- scene_id: 106878960_174887073
- rooms used: entryway/foyer/lobby_0, living_room_0; kitchen_0 is an additional candidate search room
- furniture used (id — room — catalog description):
  - chest_of_drawers_2 — entryway/foyer/lobby_0 — Bombay Two Drawer Chest, Black
  - table_2 — living_room_0 — Madison Park Signature Bordeaux Coffee Table

## Task instruction / prompt given
"Move both packages to the living-room center table and place them next to each other. Bring the scissors beside them. One package is in the entryway; the other is either in the entryway or the living room. The scissors are either in the entryway or the kitchen."

## Affected object(s)
- box: box_0 is the candidate-room uncertainty target; box_1 is the remaining task object with exact memory.
- scissors: scissors_0 is the other candidate-room uncertainty target.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Candidate Rooms

## Information supplied in instruction
- Specifies two packages, the living-room center table, mutual adjacency, scissors beside them, and one package in the entryway.
- Gives entryway/living room as the selected package's candidate rooms and entryway/kitchen as the scissors' candidate rooms.
- Omits which candidate is true, exact target furniture, floor positions, instance ids, and states. The source wording avoids revealing that both packages are actually in the entryway.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.

## Final expected world state
- box_0: on table_2 (living_room_0).
- box_1: on table_2 (living_room_0), next to box_0.
- scissors_0: on table_2 (living_room_0), next to both boxes.

## Initial robot memory
- box_0: in one of {entryway/foyer/lobby_0, living_room_0}, movable; actual room and support unresolved.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: in one of {entryway/foyer/lobby_0, kitchen_0}, movable; actual room and support unresolved.
- Candidate sets are accurate and include the true rooms; no exact target furniture associations are supplied.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(scissors_0, table_2).
- is_next_to(scissors_0, box_0) and is_next_to(scissors_0, box_1), all on table_2.

## Spawn / planner notes
- Spawn box ×2 on the entryway floor and scissors ×1 on chest_of_drawers_2's top, identical to BASE. Do not spawn duplicates in candidate rooms.
- chest_of_drawers_2 maps the entryway table; table_2 maps the center table.
- Start in entryway/foyer/lobby_0. Supply only candidate-room target knowledge; exact world anchors are evaluator-only until observed.
