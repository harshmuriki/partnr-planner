# T2-OUT-AMB

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
"Move the packages to the center of the living room and bring something to open them."

## Affected object(s)
- box: box_0 is the uncertainty target remembered at a stale living-room location; box_1 is the remaining task object with accurate memory.
- scissors: scissors_0 is the other uncertainty target remembered at a stale dining-room location and is the actual appropriate opening tool.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Instruction: Underspecified

## Information supplied in instruction
- Mentions packages, the center of the living room, and a tool's intended package-opening function.
- Omits explicit count, entryway source, destination furniture, next-to arrangement, scissors identity, source furniture, instance ids, and states. The prompt does not correct stale source locations.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), movable.

## Final expected world state
- box_0: on table_2 (living_room_0).
- box_1: on table_2 (living_room_0), next to box_0.
- scissors_0: on table_2 (living_room_0), next to both boxes.
- No package-opened state is required.

## Initial robot memory
- box_0: on table_2 (living_room_0), movable; stale location.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on table_0 (dining_room_0), movable; stale location.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2), established in the actual world rather than accepted from stale memory.
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(scissors_0, table_2).
- is_next_to(scissors_0, box_0) and is_next_to(scissors_0, box_1), all on table_2.

## Spawn / planner notes
- Spawn box ×2 on the entryway floor and scissors ×1 on chest_of_drawers_2's top, the mapped entryway table. Keep the physical BASE world unchanged; spawn no stale-location copies.
- Resolve the center to table_2 and the opening tool to scissors_0. Adjacency is the canonical benchmark arrangement, not an explicit prompt requirement.
- table_2 also anchors stale box memory. Because no dining bench is cataloged, map the stale dining-bench scissors reference to dining-room table_0.
- Start in entryway/foyer/lobby_0. Verify remembered locations, update target records, and use Navigate, Pick, Place; no cutting or package-opening action is required.
