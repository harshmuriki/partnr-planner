# T3-INC-DIS

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
- book: book_0 and book_1 are uncertainty targets, omitted from memory.
- laptop: laptop_0 is the remaining task object.
- folder: folder_0 and folder_1 are distractors, not substitutes; their own surface locations remain in memory.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Distractors: Present

## Information supplied in instruction
- Specifies two books, source rooms, the destination surface, and the laptop's location and off state.
- Omits folders and exact book source furniture.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area at table_2.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- folder_0: on table_3 (living_room_0), next to book_0.
- folder_1: on table_1 (bedroom_1), next to book_1.

## Final expected world state
- book_0: on table_0 (living_room_0).
- book_1: on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.
- folder_0: on table_3 (living_room_0), retained at its source.
- folder_1: on table_1 (bedroom_1), retained at its source.

## Initial robot memory
- laptop_0: on table_2 (kitchen_0), powered on.
- folder_0: on table_3 (living_room_0).
- folder_1: on table_1 (bedroom_1).
- Both book records and all relations referencing them, including folder-to-book adjacency, are omitted.

## Success criteria
- is_on_top(book_0, table_0) and is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2) and is_powered_off(laptop_0).
- is_on_top(folder_0, table_3) and is_on_top(folder_1, table_1).
- Folders do not count as books.

## Spawn / planner notes
- Spawn book ×2, laptop ×1, folder ×2 on the listed surfaces.
- Initially satisfy is_next_to(folder_0, book_0) and is_next_to(folder_1, book_1), with no stacking.
- Keep distractor surface memory accurate but remove adjacency facts that would reintroduce omitted targets.
- Task-object world locations remain BASE. Robot starts in living_room_0.
