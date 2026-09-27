# T3-OUT-SUB

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
- book: book_0 and book_1 are designated uncertainty targets. book_0 exists; book_1 is absent and not spawned, but falsely remembered.
- laptop: laptop_0 is the remaining task object.
- Suitable substitutes: none authorized; the task row says N/A. No distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies two books, source rooms, the destination surface, and the laptop's location and off state.
- Omits the missing book, substitute authorization, and the fact that memory is stale.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area at table_2.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- book_1: absent; table_1 in bedroom_1 has no task book. No suitable substitute exists.

## Final expected world state
- No achievable two-book completion exists in this diagnostic fallback.
- Feasible completion: book_0 on table_0 (living_room_0); laptop_0 on table_2 (kitchen_0), powered off.
- book_1 remains absent; no replacement is invented.

## Initial robot memory
- book_0: on couch_0 (living_room_0), believed available; stale location.
- book_1: on shelves_0 (bedroom_0), believed available; false existence and stale location.
- laptop_0: on table_2 (kitchen_0), powered on; accurate.
- No substitute is represented.

## Success criteria
- SUB benchmark success is not defined for this unsupported matrix cell; exclude it from SUB scoring.
- Diagnostic behavior: reject contradicted memory and give up on the missing-book subgoal without inventing a substitute.
- Feasible subgoals: is_on_top(book_0, table_0), is_on_top(laptop_0, table_2), is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×1 and laptop ×1 only at true locations. Do not instantiate memory-only book_1.
- This is an explicitly absence-only diagnostic fallback, not a valid SUB trial. An amended task row must authorize a substitute before SUB scoring is possible.
- The row supplies no concrete stale ids; use couch_0 and shelves_0 consistently across OUT variants.
- Robot starts in living_room_0; keep laptop memory accurate.
