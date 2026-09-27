# T3-INC-BASE

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
- book: book_0 and book_1 are both uncertainty targets, physically present but omitted from memory.
- laptop: laptop_0 is the remaining task object, retained in memory.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete

## Information supplied in instruction
- Specifies two books, their source rooms, the side-table destination, and the laptop's location and off state.
- Omits book source furniture and instance ids. The instruction gives a requested count, not a memory record of observed objects.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area at table_2.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Final expected world state
- book_0: on table_0 (living_room_0).
- book_1: on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.

## Initial robot memory
- laptop_0: on table_2 (kitchen_0), powered on.
- Both book records and all their location relations are omitted, not marked unavailable.
- Scene furniture and destination table_0 remain known independently of object observations.

## Success criteria
- is_on_top(book_0, table_0).
- is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2).
- is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 on the listed BASE surfaces.
- Remove only the uncertainty-target book records from initial memory; do not remove the laptop or change the physical world.
- Search and observation must recover book locations. No containment, lamps, or stacking are required.
- Robot starts in living_room_0.
