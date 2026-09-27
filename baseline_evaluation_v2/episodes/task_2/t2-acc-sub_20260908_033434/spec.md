# T2-ACC-SUB

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
- box: box_0 is the selected uncertainty target; box_1 is the remaining task object. Both remain available.
- scissors: scissors_0 is the requested uncertainty target, absent and not spawned.
- knife: knife_0 is the suitable substitute for scissors_0.
- No distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: Substitute Available

## Information supplied in instruction
- Mentions two packages, entryway source, living-room center table destination, mutual adjacency, and scissors beside the packages.
- Omits scissors unavailability, the substitute, source furniture, exact floor positions, instance ids, and states.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- knife_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable; suitable package-opening tool.
- scissors_0 is absent; no scissors are available.

## Final expected world state
- box_0: on table_2 (living_room_0).
- box_1: on table_2 (living_room_0), next to box_0.
- knife_0: on table_2 (living_room_0), next to both boxes.
- Scissors remain absent.

## Initial robot memory
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- knife_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable; suitable package-opening tool.
- scissors_0 is absent; no scissors are available.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(knife_0, table_2).
- is_next_to(knife_0, box_0) and is_next_to(knife_0, box_1), all on table_2.
- Accept knife_0 as the substitute; do not require or invent scissors.

## Spawn / planner notes
- Spawn box ×2 on the entryway floor and knife ×1 on chest_of_drawers_2. Do not spawn scissors or other cutting tools.
- Use chest_of_drawers_2's top as the catalog equivalent of the entryway table; table_2 is the center table.
- Start in entryway/foyer/lobby_0. Move and place the substitute; opening packages is not a required action.
