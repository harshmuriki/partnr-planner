# T5-INC-SUB

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
- soap_dispenser: soap_dispenser_0, uncertainty target, present but omitted from memory.
- bottle: bottle_0, uncertainty target, present but omitted from memory; bottle_1, remaining task object.
- plate: plate_0, white uncertainty target, absent; plate_1, black suitable substitute, omitted from memory.
- No distractors spawned.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, a plate, living-room table, and adjacency.
- Omits source locations, ids, plate color, and substitute availability. The black plate satisfies the generic plate request; destination is table_0.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_1: on cabinet_0 (kitchen_0), black, usable serving plate.
- plate_0 is absent.

## Final expected world state
- bottle_0 and bottle_1: on table_0 (living_room_0), clean, filled.
- plate_1: on table_0 (living_room_0), next to both bottles.
- soap_dispenser_0: on cabinet_0 (kitchen_0); no relocation required.
- plate_0 remains absent.

## Initial robot memory
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- Environmental records for the selected targets and the substitute are omitted. Plate availability is unknown, not recorded as impossible.

## Success criteria
- is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- is_on_top(plate_1, table_0); is_next_to(plate_1, bottle_0); is_next_to(plate_1, bottle_1).
- Discovering and using plate_1 is success; do not invent plate_0.

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1, with a black asset for plate_1. Do not spawn white plate_0 or use a black_plate class.
- Keep soap and bottles at BASE locations. Map dining to table_2 in kitchen_0 and counter/sink-side placement to cabinet_0's surface.
- Validate and use the kitchen faucet affordance for Clean/Fill; soap need not move.
- Robot starts in living_room_0; no stacking.
