# T5-INC-ABS

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
- plate: plate_0, uncertainty target, absent.
- No acceptable serving substitutes or distractors exist.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, plate, living-room table, and adjacency.
- Omits source locations, ids, and plate unavailability. Destination is table_0.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0 and all acceptable plate substitutes are absent.

## Final expected world state
- Plate and substitutes remain absent; abandon the impossible plate subgoal without claiming full completion.
- Feasible partial completion: bottle_0 and bottle_1 on table_0 (living_room_0), clean, filled.
- soap_dispenser_0: on cabinet_0 (kitchen_0); no relocation required.

## Initial robot memory
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- Selected target records are omitted. There is no authoritative availability record for a plate or substitute; missing memory alone does not establish nonexistence.

## Success criteria
- Required ABS behavior: after bounded search or verification, give up on the unavailable plate goal; do not invent a plate or substitute, or claim full task success.
- Feasible partial propositions: is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- Feasible partial propositions: is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- No plate placement proposition is required or satisfiable.

## Spawn / planner notes
- Spawn soap_dispenser ×1 and bottle ×2 on BASE surfaces; spawn no plate or acceptable serving substitute.
- Map dining to table_2 in kitchen_0 and counter/sink-side placement to cabinet_0's surface.
- Validate and use the kitchen faucet affordance for Clean/Fill; no invented sink id.
- Robot starts in living_room_0. Search can resolve incomplete memory, but cannot make the missing plate physically available.
