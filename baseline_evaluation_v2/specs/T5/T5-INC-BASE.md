# T5-INC-BASE

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
- soap_dispenser: soap_dispenser_0, uncertainty target omitted from memory.
- bottle: bottle_0, uncertainty target omitted from memory; bottle_1, remaining task object retained in memory.
- plate: plate_0, white appearance, uncertainty target omitted from memory.
- No substitutes or distractors spawned.

## Uncertainty being tested
- Internal robot memory: Incomplete

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, plate, living-room table, and adjacency.
- Omits source locations and object ids. Destination is table_0; the requested objects' mention does not provide their environmental records.

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
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- No environmental records for the three selected uncertainty targets; absence of a record is not evidence of physical absence.

## Success criteria
- is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).
- Discover missing target records through observation rather than fabricating locations.

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1 at the listed BASE surfaces; memory omission does not remove physical objects.
- Map the spreadsheet dining area to table_2 in kitchen_0 and counter/sink-side placement to cabinet_0's surface.
- Validate and use the kitchen faucet affordance for Clean/Fill; no invented sink or mandatory soap relocation.
- Robot starts in living_room_0; no stacking.
