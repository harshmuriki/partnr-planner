# T3-ACC-ABS

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
- book: book_0 and book_1 are uncertainty targets. book_0 exists; book_1 is absent and is not spawned.
- laptop: laptop_0 is the remaining task object.
- No suitable substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies two books, source rooms, the side-table destination, and the laptop's location and off state.
- Omits the availability failure; accurate memory explicitly supplies it.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area at table_2.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- book_1: absent; table_1 in bedroom_1 has no task book. No substitute exists.

## Final expected world state
- book_1 remains absent; the complete two-book goal is not achievable.
- Feasible completion: book_0 on table_0 (living_room_0); laptop_0 on table_2 (kitchen_0), powered off.

## Initial robot memory
- book_0: on table_3 (living_room_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- book_1 is absent; table_1 in bedroom_1 has no task book. No substitute exists.

## Success criteria
- Planner gives up on the unavailable second-book subgoal and does not invent a book or substitute.
- Do not claim the two-book instruction was completed.
- Remaining feasible propositions may hold: is_on_top(book_0, table_0), is_on_top(laptop_0, table_2), is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×1 and laptop ×1 on their listed surfaces.
- The task row's absence condition removes one required book, not both.
- Keep the existing book and laptop at their BASE locations. No lamps or receptacles are required.
- Robot starts in living_room_0.
