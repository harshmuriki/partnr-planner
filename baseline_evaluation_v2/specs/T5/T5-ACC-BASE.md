# T5-ACC-BASE

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
- No substitutes or distractors spawned.

## Uncertainty being tested
- Internal robot memory: Accurate

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, a plate, a living-room table, and plate adjacency to the bottles.
- Omits starting locations, bottle identities, plate color, and furniture ids. Ground the destination to table_0.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.

## Final expected world state
- bottle_0: on table_0 (living_room_0), clean, filled.
- bottle_1: on table_0 (living_room_0), clean, filled.
- plate_0: on table_0 (living_room_0), next to both bottles.
- soap_dispenser_0: on cabinet_0 (kitchen_0); no relocation required.

## Initial robot memory
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.

## Success criteria
- is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1 at the surfaces listed above.
- The catalog has no dining_room: its dining area is table_2 in kitchen_0. Map kitchen counter and sink-side placement to the surface of cabinet_0.
- Use the kitchen faucet affordance for Clean and Fill; do not invent sink furniture or require soap relocation. Validate that affordance during scene setup.
- Robot starts in living_room_0. Use adjacency, not stacking.
