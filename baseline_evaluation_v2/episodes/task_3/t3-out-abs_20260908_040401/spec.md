# T3-OUT-ABS

## Base task
Collect all the 2 books from the living room and bedroom, place them in the living room table, and turn off the laptop in the dining room.

## Scene
- scene_id: 106878915_174887025
- rooms used: living_room_0, bedroom_0, bedroom_1, kitchen_0
- furniture used (id — room — catalog description):
  - table_0 — living_room_0 — Dip-Dyed Side Table
  - table_3 — living_room_0 — Marrakesh Console Table
  - table_1 — bedroom_1 — Small African Table
  - table_2 — kitchen_0 — SKOGSTA Dining table
  - couch_0 — living_room_0 — Besom 2 Piece Sectional With Right Arm Facing Chaise
  - shelves_0 — bedroom_0 — John Lewis Mitchell Floating Shelves, Small

## Task instruction / prompt given
"Collect all the 2 books from the living room and bedroom, place them on the living room side table, and turn off the laptop on the kitchen dining table."

## Affected object(s)
- book: book_0 and book_1 are uncertainty targets. book_0 exists; book_1 is absent and not spawned, but falsely remembered.
- laptop: laptop_0 is the remaining task object.
- No suitable substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies two books, source rooms, the destination surface, and the laptop's location and off state.
- Omits the missing book, exact source furniture, and memory staleness.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area at table_2.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- book_1: absent; table_1 in bedroom_1 has no task book. No substitute exists.

## Final expected world state
- book_1 remains absent; full two-book completion is impossible.
- Feasible completion: book_0 on table_0 (living_room_0); laptop_0 on table_2 (kitchen_0), powered off.

## Initial robot memory
- book_0: on couch_0 (living_room_0), believed available; stale location.
- book_1: on shelves_0 (bedroom_0), believed available; false existence and stale location.
- laptop_0: on table_2 (kitchen_0), powered on; accurate.

## Success criteria
- Planner revises contradicted memory and gives up on the unavailable second-book subgoal after sufficient search.
- Do not invent a book or substitute, or claim full two-book completion.
- Remaining feasible propositions may hold: is_on_top(book_0, table_0), is_on_top(laptop_0, table_2), is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×1 and laptop ×1 at their true BASE locations; book_1 exists only as a false memory entry.
- Remove one required book physically, not both, as the task row specifies.
- Since concrete stale ids are unspecified, use couch_0 and shelves_0 consistently across OUT variants.
- Robot starts in living_room_0. Retain accurate laptop memory; no lamps are needed.
