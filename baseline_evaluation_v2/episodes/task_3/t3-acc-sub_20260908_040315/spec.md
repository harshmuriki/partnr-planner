# T3-ACC-SUB

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
- book: book_0 and book_1 are the designated uncertainty targets; only book_0 exists.
- book_1 is a reserved missing instance id, not a spawned object.
- laptop: laptop_0 is the remaining task object.
- Suitable substitutes: none authorized; the task row says N/A. No distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies two books, source rooms, a side-table destination, and the laptop's location and off state.
- Does not disclose the missing book or authorize replacing a book with another class.
- The living-room shelf maps to table_3; the dining area maps to table_2 in kitchen_0.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- book_1: absent; no substitute exists. table_1 in bedroom_1 has no task book.

## Final expected world state
- No achievable two-book completion exists in this diagnostic fallback.
- Feasible completion: book_0 on table_0 (living_room_0); laptop_0 on table_2 (kitchen_0), powered off.
- book_1 remains absent; no replacement is invented.

## Initial robot memory
- book_0: on table_3 (living_room_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- book_1 is unavailable; no suitable substitute exists. table_1 in bedroom_1 has no task book.

## Success criteria
- SUB benchmark success is not defined for this unsupported matrix cell; exclude it from SUB scoring.
- Diagnostic behavior: give up on the missing-book subgoal without inventing a substitute.
- Feasible subgoals: is_on_top(book_0, table_0), is_on_top(laptop_0, table_2), is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×1 and laptop ×1 on their listed surfaces; do not spawn book_1 or an arbitrary substitute.
- This file preserves the requested SUB slot but supplies an explicitly absence-only diagnostic fallback, not a valid substitute trial.
- A valid SUB trial requires a task-row amendment identifying an acceptable substitute.
- Robot starts in living_room_0; no lamps or containment are needed.
