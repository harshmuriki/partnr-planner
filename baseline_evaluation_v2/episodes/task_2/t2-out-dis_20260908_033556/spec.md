# T2-OUT-DIS

## Base task
Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them.

## Scene
- scene_id: 106878960_174887073
- rooms used: entryway/foyer/lobby_0, living_room_0, dining_room_0
- furniture used (id — room — catalog description):
  - chest_of_drawers_2 — entryway/foyer/lobby_0 — Bombay Two Drawer Chest, Black
  - table_2 — living_room_0 — Madison Park Signature Bordeaux Coffee Table
  - table_0 — dining_room_0 — Livingston Dining Table 62"

## Task instruction / prompt given
"Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them."

## Affected object(s)
- box: box_0 is the uncertainty target with stale memory; box_1 is the remaining task object with accurate memory.
- scissors: scissors_0 is the other uncertainty target with stale memory.
- pencil_case: pencil_case_0 is a distractor near the actual box_0.
- pen: pen_0 and screwdriver: screwdriver_0 are distractors beside the actual scissors_0, not substitutes. Their own support locations remain accurate in memory.

## Uncertainty being tested
- Internal robot memory: Outdated
- Distractors: Present

## Information supplied in instruction
- Mentions two packages, entryway source, living-room center table destination, mutual adjacency, and scissors beside the packages.
- Omits distractors, source furniture, exact floor positions, instance ids, and states. Stale memory mislocates the selected targets, not the remaining box or distractor supports.

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
- box_0: on table_2 (living_room_0), movable; stale location.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on table_0 (dining_room_0), movable; stale location.
- pencil_case_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- pen_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.
- screwdriver_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.
- Near-target relations are not retained because the remembered target locations differ; no distractor itself is relocated.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(scissors_0, table_2).
- is_next_to(scissors_0, box_0) and is_next_to(scissors_0, box_1), all on table_2.
- is_on_top(pen_0, chest_of_drawers_2) and is_on_top(screwdriver_0, chest_of_drawers_2).
- in_room(pencil_case_0, entryway/foyer/lobby_0); retain its original floor placement.

## Spawn / planner notes
- Spawn box ×2, scissors ×1, pencil_case ×1, pen ×1, screwdriver ×1 at the actual listed anchors. Spawn no copies at stale locations.
- chest_of_drawers_2's top maps the entryway table; table_2 maps the center table and stale box support.
- Map the spreadsheet's dining bench, absent from this catalog, to dining-room table_0 for stale scissors memory only.
- Start in entryway/foyer/lobby_0. Correct target localization and discriminate targets from nearby distractors; do not use remembered distractors as replacements.
