# T3-OUT-BASE

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
- book: book_0 and book_1 are uncertainty targets with stale remembered locations.
- laptop: laptop_0 is the remaining task object with accurate memory.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated

## Information supplied in instruction
- Specifies two books, source rooms, the side-table destination, and the laptop's location and off state.
- Omits exact book source furniture and does not identify stale memory.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area at table_2.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Final expected world state
- book_0: on table_0 (living_room_0).
- book_1: on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.

## Initial robot memory
- book_0: on couch_0 (living_room_0), believed available; stale.
- book_1: on shelves_0 (bedroom_0), believed available; stale.
- laptop_0: on table_2 (kitchen_0), powered on; accurate.

## Success criteria
- is_on_top(book_0, table_0).
- is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2).
- is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 only at true BASE locations, not at remembered surfaces.
- The task row specifies different stale furniture but no concrete ids. Use couch_0 and shelves_0 consistently across OUT variants as catalog-grounded stale locations.
- Change only book memory; retain accurate laptop memory and physical state.
- Robot starts in living_room_0; no lamps or containment are needed.
