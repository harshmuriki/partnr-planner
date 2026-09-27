# T3-ACC-CAND

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

## Task instruction / prompt given
"Collect the 2 books, which may be in the living room or either bedroom, place them on the living room side table, and turn off the laptop on the kitchen dining table."

## Affected object(s)
- book: book_0 and book_1 are uncertainty targets with candidate-room localization.
- laptop: laptop_0 is the remaining task object with its exact location known.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Candidate Rooms

## Information supplied in instruction
- Specifies two books, candidate source rooms, the destination surface, and the laptop's location and off state.
- Does not identify either book's actual room or furniture, nor assert one book per room.
- The spreadsheet's study is absent from this catalog; bedroom_0 is the study-like candidate-room proxy. bedroom_1 is the actual bedroom source.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- bedroom_0 contains no task book; it is a search candidate only.

## Final expected world state
- book_0: on table_0 (living_room_0).
- book_1: on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.

## Initial robot memory
- book_0: available; candidate rooms are living_room_0, bedroom_1, bedroom_0; no actual room or furniture supplied.
- book_1: available; candidate rooms are living_room_0, bedroom_1, bedroom_0; no actual room or furniture supplied.
- laptop_0: on table_2 (kitchen_0), powered on.
- Destination table_0 is known. Memory does not disclose that bedroom_0 is empty of task books.

## Success criteria
- is_on_top(book_0, table_0).
- is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2).
- is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 at BASE locations. Do not randomize the world merely because memory lists candidates.
- Use only the three cataloged candidate rooms, not an invented study room.
- Withhold exact target locations from planner input. Robot starts in living_room_0; no lamps are needed.
