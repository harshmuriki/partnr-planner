# T5-ACC-DIS

## Base task
Get the soap to clean 2 bottles, fill them, and place them on the living-room table. Get a plate and place it next to them.

## Scene
- scene_id: 106878915_174887025
- rooms used: kitchen_0, living_room_0
- furniture used (id — room — catalog description):
  - cabinet_0 — kitchen_0 — Kitchen
  - table_2 — kitchen_0 — SKOGSTA Dining table
  - table_0 — living_room_0 — Dip-Dyed Side Table

## Task instruction / prompt given
"Get the soap to clean 2 bottles, fill them, and place them on the living-room table. Get a plate and place it next to them."

## Affected object(s)
- soap_dispenser: soap_dispenser_0, uncertainty target.
- bottle: bottle_0, uncertainty target; bottle_1, remaining task object.
- plate: plate_0, white appearance, uncertainty target.
- soap_dish: soap_dish_0; spray_bottle: spray_bottle_0; plant_saucer: plant_saucer_0; all distractors, not substitutes.

## Uncertainty being tested
- Internal robot memory: Accurate
- Distractors: Present

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, a plate, living-room table, and adjacency.
- Omits source locations, ids, and distractors. Destination is table_0.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.
- soap_dish_0: on cabinet_0 (kitchen_0), near soap_dispenser_0, empty.
- spray_bottle_0: on table_2 (kitchen_0), near bottle_0, non-drinking spray bottle.
- plant_saucer_0: on cabinet_0 (kitchen_0), near plate_0, unsuitable for serving food.

## Final expected world state
- bottle_0 and bottle_1: on table_0 (living_room_0), clean, filled.
- plate_0: on table_0 (living_room_0), next to both bottles.
- soap_dispenser_0, soap_dish_0, plant_saucer_0: on cabinet_0 (kitchen_0), unchanged otherwise.
- spray_bottle_0: on table_2 (kitchen_0), unchanged.

## Initial robot memory
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.
- soap_dish_0: on cabinet_0 (kitchen_0), near soap_dispenser_0, empty.
- spray_bottle_0: on table_2 (kitchen_0), near bottle_0, non-drinking spray bottle.
- plant_saucer_0: on cabinet_0 (kitchen_0), near plate_0, unsuitable for serving food.

## Success criteria
- is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).
- Preserve is_on_top(soap_dish_0, cabinet_0), is_on_top(plant_saucer_0, cabinet_0), and is_on_top(spray_bottle_0, table_2); distractors do not satisfy task goals.

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1, soap_dish ×1, spray_bottle ×1, plant_saucer ×1, all on surfaces.
- Map dining to table_2 in kitchen_0 and counter/sink-side placement to cabinet_0's surface.
- Validate and use the kitchen faucet affordance for Clean/Fill; soap need not move. Robot starts in living_room_0; no stacking.
