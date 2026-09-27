# T3-ACC-AMB

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
"Place the 2 books in the living room and turn off the laptop"

## Affected object(s)
- book: book_0 and book_1 are both uncertainty targets.
- laptop: laptop_0 is the remaining task object.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Instruction: Underspecified

## Information supplied in instruction
- Specifies two books, the living room as their destination, and powering off the laptop.
- Omits book source rooms, source furniture, a specific destination surface, and the laptop's room and furniture.
- The world retains the canonical table destination as an acceptable choice, not an undisclosed requirement.
- The living-room shelf maps to table_3; the canonical dining area maps to table_2 in kitchen_0.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Final expected world state
- Accepted grounded realization: book_0 on table_0 (living_room_0); book_1 on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.
- Since no destination surface is specified, leaving book_0 on table_3 and placing book_1 on table_0 also satisfies the room-level book goal.

## Initial robot memory
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Success criteria
- in_room(book_0, living_room_0).
- in_room(book_1, living_room_0).
- is_on_top(laptop_0, table_2) and is_powered_off(laptop_0).
- is_on_top(book_0, table_0) and is_on_top(book_1, table_0) are a valid witness, not mandatory hidden surface constraints.

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 on BASE surfaces.
- Use the spreadsheet's underspecified prompt verbatim. Do not infer an obligatory destination table from evaluator-only base-task information.
- The planner may choose table_0 without requiring clarification for a room-level goal.
- Robot starts in living_room_0; no lamps or containment are needed.
