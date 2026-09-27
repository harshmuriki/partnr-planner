# T3-INC-CON

## Base task
Collect all the 2 books from the living room and bedroom, place them in the living room table, and turn off the laptop in the dining room.

## Scene
- scene_id: 106878915_174887025
- rooms used: living_room_0, kitchen_0
- furniture used (id — room — catalog description):
  - table_0 — living_room_0 — Dip-Dyed Side Table
  - table_3 — living_room_0 — Marrakesh Console Table
  - cabinet_0 — kitchen_0 — Kitchen
  - table_2 — kitchen_0 — SKOGSTA Dining table

## Task instruction / prompt given
"Collect the 2 books from the living room and kitchen, place them on the living room side table, and turn off the laptop on the kitchen dining table."

## Affected object(s)
- book: book_0 and book_1 are uncertainty targets, both omitted from memory. Only book_1 receives the containment change.
- laptop: laptop_0 is the remaining task object.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies two books, scene-grounded source rooms, the side-table destination, and the laptop's location and off state.
- Omits exact book furniture and containment.
- No bedroom cabinet is cataloged. cabinet_0 in kitchen_0 is the supported containing-furniture proxy; the prompt names its true room.
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
- laptop_0: on table_2 (kitchen_0), powered on.
- cabinet_0 (kitchen_0): closed; Open/Close enabled. Its task-object contents are not supplied.
- Both book records and all book location and containment relations are omitted.

## Success criteria
- is_on_top(book_0, table_0).
- is_on_top(book_1, table_0).
- is_on_top(laptop_0, table_2).
- is_powered_off(laptop_0).

## Spawn / planner notes
- Spawn book ×2 and laptop ×1. Place book_1 within a containing compartment of cabinet_0 and close it.
- The planner must discover the contained book and open the compartment before picking it.
- Preserve book_0 and laptop_0 at BASE locations; retain laptop and cabinet articulation facts in memory without leaking contents.
- Robot starts in living_room_0. No lamps are required.
