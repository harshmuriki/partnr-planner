# T5-ACC-ROOM

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
- soap_dispenser: soap_dispenser_0, room-localized uncertainty target.
- bottle: bottle_0, room-localized uncertainty target; bottle_1, remaining task object with exact memory.
- plate: plate_0, white appearance, room-localized uncertainty target.
- No substitutes or distractors spawned.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Room Known

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, plate, living-room table, and adjacency.
- Omits source furniture and rooms. Memory supplies target rooms only; destination is table_0.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.

## Final expected world state
- bottle_0 and bottle_1: on table_0 (living_room_0), clean, filled.
- plate_0: on table_0 (living_room_0), next to both bottles.
- soap_dispenser_0: on cabinet_0 (kitchen_0); no relocation required.

## Initial robot memory
- soap_dispenser_0: in kitchen_0, usable soap dispenser; exact furniture and relation withheld.
- bottle_0: in kitchen_0, dirty, empty; exact furniture and relation withheld.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: in kitchen_0, white, usable serving plate; exact furniture and relation withheld.

## Success criteria
- Initial localization information: in_room(soap_dispenser_0, kitchen_0); in_room(bottle_0, kitchen_0); in_room(plate_0, kitchen_0), without target furniture disclosure.
- Final: is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- Final: is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- Final: is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1 at the exact BASE surface locations; do not expose those target placements through memory.
- The catalog has no dining room. Map the spreadsheet's dining-room localization to kitchen_0, which contains its dining table, table_2. Map counter/sink-side placement to cabinet_0's surface.
- Validate and use the kitchen faucet affordance for Clean/Fill; soap need not move.
- Robot starts in living_room_0. Search within known rooms; no stacking.
