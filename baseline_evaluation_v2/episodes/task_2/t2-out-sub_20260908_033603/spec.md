# T2-OUT-SUB

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
- box: box_0 is the selected uncertainty target with stale memory; box_1 is the remaining task object with accurate memory.
- scissors: scissors_0 is the absent requested uncertainty target, present only as a stale memory record.
- knife: knife_0 is the physically available suitable substitute, not represented in stale memory.
- No distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
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
- box_0: on table_2 (living_room_0), movable; stale location.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on table_0 (dining_room_0), movable; false existence and stale location.
- No substitute record is supplied; memory retains the missing requested tool instead of the actual knife.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(knife_0, table_2).
- is_next_to(knife_0, box_0) and is_next_to(knife_0, box_1), all on table_2.
- Accept the discovered knife as substitute success; reject the nonexistent remembered scissors as evidence of delivery.

## Spawn / planner notes
- Spawn box ×2 on the entryway floor and knife ×1 on chest_of_drawers_2's top. Spawn no scissors or stale-location duplicates.
- chest_of_drawers_2 maps the entryway table; table_2 is the center table and stale box support.
- Map the unavailable catalog dining bench to table_0 in dining_room_0 for stale scissors memory only.
- Start in entryway/foyer/lobby_0. Correct both stale targets and discover the substitute before planning delivery; no package-opening action is required.
