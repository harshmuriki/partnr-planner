# T3-ACC-ROOM

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
"Collect the 2 books, one in the living room and one in bedroom 1, place them on the living room side table, and turn off the laptop on the kitchen dining table."

## Affected object(s)
- book: book_0 and book_1 are uncertainty targets with room-only localization.
- laptop: laptop_0 is the remaining task object with its exact location known.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Room Known

## Information supplied in instruction
- Specifies two books, one source room per book, the exact destination surface, and the laptop's location and off state.
- Omits book source furniture. The bedroom label identifies bedroom_1 rather than the other catalog bedroom.
- The unavailable living-room shelf maps to table_3 in the world only. Dining room maps to the kitchen dining area.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: on table_1 (bedroom_1), available.
- laptop_0: on table_2 (kitchen_0), powered on.

## Final expected world state
- book_0: on table_0 (living_room_0).
- book_1: on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.

## Initial robot memory
- book_0: in living_room_0, available; supporting furniture is not supplied.
- book_1: in bedroom_1, available; supporting furniture is not supplied.
- laptop_0: on table_2 (kitchen_0), powered on.
- Destination table_0 is known; neither book's initial furniture is encoded or implied by an object-surface link.

## Success criteria
- is_on_top(book_0, table_0).
- is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2).
- is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 at exact BASE locations; localization uncertainty affects information, not physical placement.
- Supply only room-level target facts to the planner; do not expose the evaluator's exact book placements.
- Keep the laptop's exact memory unchanged. Robot starts in living_room_0; no lamps or containment are needed.
