# T3-OUT-DIS

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
- book: book_0 and book_1 are uncertainty targets with stale memory.
- laptop: laptop_0 is the remaining task object.
- folder: folder_0 and folder_1 are distractors, not substitutes; their surface locations remain accurately remembered.

## Uncertainty being tested
- Internal robot memory: Outdated
- Distractors: Present

## Information supplied in instruction
- Specifies two books, source rooms, the destination surface, and the laptop's location and off state.
- Omits folders, exact book source furniture, and memory staleness.
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
- book_0: on couch_0 (living_room_0), believed available; stale.
- book_1: on shelves_0 (bedroom_0), believed available; stale.
- laptop_0: on table_2 (kitchen_0), powered on; accurate.
- folder_0: on table_3 (living_room_0); accurate surface location.
- folder_1: on table_1 (bedroom_1); accurate surface location.
- Current folder-to-book adjacency is not remembered; target-related relations reflect the stale book placements rather than leaking true locations.

## Success criteria
- is_on_top(book_0, table_0) and is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2) and is_powered_off(laptop_0).
- is_on_top(folder_0, table_3) and is_on_top(folder_1, table_1).
- Folders do not count toward the two-book goal.

## Spawn / planner notes
- Spawn book ×2, laptop ×1, folder ×2 at true locations. Initially satisfy is_next_to(folder_0, book_0) and is_next_to(folder_1, book_1), without stacking.
- The row gives no concrete stale ids; couch_0 and shelves_0 consistently ground the different-furniture requirement.
- Change only target-related memory, not laptop or folder surface facts. Robot starts in living_room_0.
