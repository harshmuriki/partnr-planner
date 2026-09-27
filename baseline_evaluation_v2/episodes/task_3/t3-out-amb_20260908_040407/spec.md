# T3-OUT-AMB

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
"Place the 2 books in the living room and turn off the laptop"

## Affected object(s)
- book: book_0 and book_1 are uncertainty targets with stale remembered locations.
- laptop: laptop_0 is the remaining task object with accurate memory.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Instruction: Underspecified

## Information supplied in instruction
- Specifies two books, the living room as their destination, and turning off the laptop.
- Omits book source rooms and furniture, a destination surface, and the laptop's room and furniture.
- Does not disclose stale memory. The canonical table destination is an acceptable choice, not a hidden requirement.
- The living-room shelf maps to table_3; the canonical dining area maps to table_2 in kitchen_0.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Final expected world state
- Accepted grounded realization: book_0 on table_0 (living_room_0); book_1 on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.
- Leaving book_0 on table_3 and placing book_1 on table_0 is also accepted because the prompt only requires the books to be in the living room.

## Initial robot memory
- book_0: on couch_0 (living_room_0), believed available; stale.
- book_1: on shelves_0 (bedroom_0), believed available; stale.
- laptop_0: on table_2 (kitchen_0), powered on; accurate.

## Success criteria
- in_room(book_0, living_room_0).
- in_room(book_1, living_room_0).
- is_on_top(laptop_0, table_2) and is_powered_off(laptop_0).
- is_on_top(book_0, table_0) and is_on_top(book_1, table_0) are an acceptable witness, not mandatory hidden surface constraints.

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 at true BASE locations, not remembered surfaces.
- Use the spreadsheet AMB prompt verbatim. Do not impose the canonical table destination as a mandatory success condition.
- Concrete stale ids are unspecified in the row; use couch_0 and shelves_0 consistently across OUT variants.
- Change only book memory and retain accurate laptop memory. Robot starts in living_room_0; no lamps or containment are needed.
