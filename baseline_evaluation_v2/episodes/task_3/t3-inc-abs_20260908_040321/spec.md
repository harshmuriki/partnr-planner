# T3-INC-ABS

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
- book: book_0 and book_1 are uncertainty targets; book_0 exists and book_1 is absent, not spawned.
- laptop: laptop_0 is the remaining task object.
- No suitable substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies two books, source rooms, the side-table destination, and the laptop's location and off state.
- Omits the absent book and exact source furniture. Missing memory must not be treated as proof of physical absence.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area at table_2.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- book_1: absent; table_1 in bedroom_1 has no task book. No substitute exists.

## Final expected world state
- book_1 remains absent; full two-book completion is impossible.
- Feasible completion: book_0 on table_0 (living_room_0); laptop_0 on table_2 (kitchen_0), powered off.

## Initial robot memory
- laptop_0: on table_2 (kitchen_0), powered on.
- Both book records, their locations, and any book-absence findings are omitted.
- Scene furniture remains known; no book availability conclusion is preloaded.

## Success criteria
- Planner searches as needed, then gives up on the unavailable second-book subgoal without inventing an object or substitute.
- Do not report full completion of the two-book goal.
- Remaining feasible propositions may hold: is_on_top(book_0, table_0), is_on_top(laptop_0, table_2), is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×1 and laptop ×1 at their BASE locations.
- Remove one required book physically, as specified by the task row; omit both uncertainty targets from memory.
- Retain accurate laptop memory. No lamps or containment are needed.
- Robot starts in living_room_0.
