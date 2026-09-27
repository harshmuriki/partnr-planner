# T5-ACC-ABS

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
- soap_dispenser: soap_dispenser_0, uncertainty target, present unchanged.
- bottle: bottle_0, uncertainty target, present unchanged; bottle_1, remaining task object.
- plate: plate_0, uncertainty target, absent.
- No suitable serving substitutes or distractors exist.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, plate, living-room table, and adjacency.
- Omits source locations, ids, and the plate's unavailability. Destination is table_0.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0 and all acceptable plate substitutes are absent.

## Final expected world state
- plate_0 and acceptable substitutes remain absent; the plate subgoal is abandoned without a false completion claim.
- Feasible partial completion: bottle_0 and bottle_1 on table_0 (living_room_0), clean, filled.
- soap_dispenser_0: on cabinet_0 (kitchen_0); no relocation required.

## Initial robot memory
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0 and all acceptable plate substitutes are known absent.

## Success criteria
- Required ABS behavior: give up on the impossible plate subgoal; do not invent an object or claim full task success.
- Feasible partial propositions: is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- Feasible partial propositions: is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- No plate placement proposition is required or satisfiable.

## Spawn / planner notes
- Spawn soap_dispenser ×1 and bottle ×2 on the listed surfaces; spawn no plate or acceptable serving substitute.
- Map dining table to table_2 in kitchen_0; map counter/sink-side location to cabinet_0's surface.
- Validate and use the kitchen faucet affordance for Clean and Fill; do not invent a sink id.
- Robot starts in living_room_0. Feasible bottle work is permitted, but cannot make the complete request achievable.
