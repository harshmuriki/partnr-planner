# T3-OUT-CON

## Base task
Collect all the 2 books from the living room and bedroom, place them in the living room table, and turn off the laptop in the dining room.

## Scene
- scene_id: 106878915_174887025
- rooms used: living_room_0, bedroom_0, kitchen_0
- furniture used (id — room — catalog description):
  - table_0 — living_room_0 — Dip-Dyed Side Table
  - table_3 — living_room_0 — Marrakesh Console Table
  - cabinet_0 — kitchen_0 — Kitchen
  - table_2 — kitchen_0 — SKOGSTA Dining table
  - couch_0 — living_room_0 — Besom 2 Piece Sectional With Right Arm Facing Chaise
  - shelves_0 — bedroom_0 — John Lewis Mitchell Floating Shelves, Small

## Task instruction / prompt given
"Collect the 2 books from the living room and kitchen, place them on the living room side table, and turn off the laptop on the kitchen dining table."

## Affected object(s)
- book: book_0 and book_1 are uncertainty targets with stale on-surface memory. Only book_1 receives the physical containment change.
- laptop: laptop_0 is the remaining task object.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies two books, scene-grounded source rooms, the destination surface, and the laptop's location and off state.
- Omits exact book furniture and containment. The current prompt may contradict stale source-room memory.
- No bedroom cabinet is cataloged; use cabinet_0 in kitchen_0 and name its true room in the prompt.
- The living-room shelf maps to table_3; dining room maps to the kitchen dining area.

## Initial world state
- book_0: on table_3 (living_room_0), available.
- book_1: within cabinet_0 (kitchen_0), available.
- laptop_0: on table_2 (kitchen_0), powered on.
- cabinet_0 (kitchen_0): closed; Open/Close enabled, using a containing compartment.

## Final expected world state
- book_0: on table_0 (living_room_0).
- book_1: on table_0 (living_room_0), without stacking.
- laptop_0: on table_2 (kitchen_0), powered off.
- cabinet_0 may remain open or be reclosed.

## Initial robot memory
- book_0: on couch_0 (living_room_0), believed available; stale.
- book_1: on shelves_0 (bedroom_0), believed available; stale on-surface location, with no current containment relation.
- laptop_0: on table_2 (kitchen_0), powered on; accurate.
- cabinet_0 (kitchen_0): closed; Open/Close enabled. No book is associated with it in memory.

## Success criteria
- is_on_top(book_0, table_0).
- is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2).
- is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×2 and laptop ×1 at true locations. Put book_1 within cabinet_0 and close the containing compartment.
- Open the compartment before retrieval. Do not spawn books at remembered locations.
- Concrete stale ids are unspecified in the row; couch_0 and shelves_0 consistently ground its different-furniture requirement.
- Keep laptop and cabinet articulation memory accurate; alter only book facts. Robot starts in living_room_0.
