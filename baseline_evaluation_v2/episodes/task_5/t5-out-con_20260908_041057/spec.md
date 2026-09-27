# T5-OUT-CON

## Base task
Get the soap to clean 2 bottles, fill them, and place them on the living-room table. Get a plate and place it next to them.

## Scene
- scene_id: 106878915_174887025
- rooms used: kitchen_0, living_room_0, bathroom_1 (stale-memory search)
- furniture used (id — room — catalog description):
  - cabinet_0 — kitchen_0 — Kitchen
  - table_2 — kitchen_0 — SKOGSTA Dining table
  - table_0 — living_room_0 — Dip-Dyed Side Table
  - washer_dryer_0 — bathroom_1 — Washing machine

## Task instruction / prompt given
"Get the soap to clean 2 bottles, fill them, and place them on the living-room table. Get a plate and place it next to them."

## Affected object(s)
- soap_dispenser: soap_dispenser_0, contained uncertainty target with stale surface memory.
- bottle: bottle_0, contained uncertainty target with stale surface memory; bottle_1, remaining task object.
- plate: plate_0, white contained uncertainty target with stale surface memory.
- No substitutes or distractors spawned.

## Uncertainty being tested
- Internal robot memory: Outdated
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, plate, living-room table, and adjacency.
- Omits starting locations, containment, cabinet state, and stale-memory warning. Destination is table_0.

## Initial world state
- soap_dispenser_0: within cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: within cabinet_0 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: within cabinet_0 (kitchen_0), white, usable serving plate.
- cabinet_0: closed; Open/Close enabled on its containing compartment.

## Final expected world state
- bottle_0 and bottle_1: on table_0 (living_room_0), clean, filled.
- plate_0: on table_0 (living_room_0), next to both bottles.
- soap_dispenser_0: on cabinet_0 (kitchen_0) if retrieved, or within cabinet_0 if no relocation is needed.
- cabinet_0 may finish open or closed.

## Initial robot memory
- soap_dispenser_0: on washer_dryer_0 (bathroom_1), usable soap dispenser; stale surface location.
- bottle_0: on cabinet_0 (kitchen_0), dirty, empty; stale surface location, not inside.
- bottle_1: on table_2 (kitchen_0), dirty, empty; accurate.
- plate_0: on table_2 (kitchen_0), white, usable serving plate; stale surface location.
- cabinet_0: closed; Open/Close enabled on its containing compartment. Target containment is not remembered.

## Success criteria
- is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).
- Soap may satisfy either is_inside(soap_dispenser_0, cabinet_0) or is_on_top(soap_dispenser_0, cabinet_0); no mandatory relocation.

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1; uncertainty targets within cabinet_0, bottle_1 on table_2. Never spawn at stale locations.
- No dining room/cabinet exists: dining maps to table_2's kitchen area, and both spreadsheet cabinets map to cabinet_0's containing compartment. Stale bathroom soap maps to washer_dryer_0.
- Validate containment and Open/Close. Open before retrieving; use cabinet_0's counter for sink-side access and the validated kitchen faucet affordance for Clean/Fill.
- Robot starts in living_room_0; no stacking.
