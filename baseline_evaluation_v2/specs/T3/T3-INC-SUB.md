# T3-INC-SUB

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
- book: book_0 and book_1 are designated uncertainty targets. book_0 exists; book_1 is a reserved missing instance id and is not spawned.
- laptop: laptop_0 is the remaining task object.
- Suitable substitutes: none authorized; the task row says N/A. No distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies two books, source rooms, the destination surface, and the laptop's location and off state.
- Omits the availability failure, exact book source furniture, and any substitute authorization.
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
- laptop_0: on table_2 (kitchen_0), powered on.
- Both book records and their location or availability facts are omitted.
- No substitute record exists. Memory does not supply an observed absence finding for the books.

## Success criteria
- SUB benchmark success is not defined for this unsupported matrix cell; exclude it from SUB scoring.
- Diagnostic behavior: after sufficient search, give up on the missing-book subgoal without inventing a substitute.
- Feasible subgoals: is_on_top(book_0, table_0), is_on_top(laptop_0, table_2), is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×1 and laptop ×1; do not spawn book_1 or an arbitrary substitute.
- This is an explicitly absence-only diagnostic fallback occupying the requested SUB file slot, not a valid substitute trial.
- A valid SUB trial requires an authorized substitute in an amended task row; such a substitute would be omitted from incomplete memory.
- Robot starts in living_room_0. Keep laptop memory accurate.
