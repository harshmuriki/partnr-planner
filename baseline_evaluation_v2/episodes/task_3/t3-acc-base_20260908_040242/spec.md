# T3-ACC-BASE

## Base task
Collect all the 2 books from the living room and bedroom, place them in the living room table, and turn off the laptop in the dining room.

## Scene
- scene_id: 106878915_174887025
- rooms used: living_room_0, bedroom_1, kitchen_0
- furniture used (id — room — catalog description):
  - table_0 — living_room_0 — Dip-Dyed Side Table
  - table_3 — living_room_0 — Marrakesh Console Table
  - table_1 — bedroom_1 — Small African Table
  - table_2 — kitchen_0 — SKOGSTA Dining table

## Task instruction / prompt given
"Collect all the 2 books from the living room and bedroom, place them on the living room side table, and turn off the laptop on the kitchen dining table."

## Affected object(s)
- book: book_0 and book_1 are both uncertainty targets and required task objects.
- laptop: laptop_0 is the remaining task object.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate

## Information supplied in instruction
- Specifies two books, their source rooms, the living-room side-table destination, and the laptop's dining-table location and required off state.
- Omits exact book source furniture and instance ids. Accurate memory supplies these.
- Dining room is grounded to the kitchen dining area. The unavailable living-room shelf is grounded to the living-room console table.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Final expected world state
- book_0: on table_0 (living_room_0).
- book_1: on table_0 (living_room_0), separately placed without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.

## Initial robot memory
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Success criteria
- is_on_top(book_0, table_0).
- is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2).
- is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 on the specified surfaces.
- Use table_3 as the closest living-room source surface; do not use bedroom shelves as if they were in the living room.
- Use table_2 for the task's dining area. PowerOff targets the laptop; no lamp is needed.
- Robot starts in living_room_0. No containment or stacking is required.
