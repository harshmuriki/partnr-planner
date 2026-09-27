# T2-ACC-ROOM

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
- box: box_0 is the room-localized uncertainty target; box_1 is the remaining task object with exact memory.
- scissors: scissors_0 is the other room-localized uncertainty target.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Room Known

## Information supplied in instruction
- Mentions two packages, entryway source, living-room center table destination, mutual adjacency, and scissors beside the packages.
- Omits exact source furniture and floor positions, scissors source room, instance ids, and states. Memory supplies the entryway room for both selected targets without furniture anchors.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.

## Final expected world state
- box_0: on table_2 (living_room_0).
- box_1: on table_2 (living_room_0), next to box_0.
- scissors_0: on table_2 (living_room_0), next to both boxes.

## Initial robot memory
- box_0: in room entryway/foyer/lobby_0, movable; exact position and support omitted.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: in room entryway/foyer/lobby_0, movable; exact position and support omitted.
- Target room knowledge is accurate; no target furniture or floor-support association is supplied.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(scissors_0, table_2).
- is_next_to(scissors_0, box_0) and is_next_to(scissors_0, box_1), all on table_2.

## Spawn / planner notes
- Spawn box ×2 on the entryway floor and scissors ×1 on chest_of_drawers_2's top. Physical placements remain exact and identical to BASE.
- chest_of_drawers_2 maps the entryway table; table_2 maps the center table.
- Start in entryway/foyer/lobby_0. Expose only room-level target localization to the planner, not evaluator-only spawn anchors; locate targets by observation.
