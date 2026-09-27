# T5-OUT-BASE

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
- soap_dispenser: soap_dispenser_0, uncertainty target with stale bathroom memory.
- bottle: bottle_0, uncertainty target with stale counter memory; bottle_1, remaining task object with accurate memory.
- plate: plate_0, white uncertainty target with stale dining-table memory.
- No substitutes or distractors spawned.

## Uncertainty being tested
- Internal robot memory: Outdated

## Information supplied in instruction
- Specifies soap, two bottles, cleaning, filling, plate, living-room table, and adjacency.
- Omits starting locations and ids; supplies no warning that remembered locations are stale. Destination is table_0.

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
- soap_dispenser_0: on washer_dryer_0 (bathroom_1), usable soap dispenser; stale location.
- bottle_0: on cabinet_0 (kitchen_0), dirty, empty; stale location.
- bottle_1: on table_2 (kitchen_0), dirty, empty; accurate.
- plate_0: on table_2 (kitchen_0), white, usable serving plate; stale location.

## Success criteria
- is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).
- Verify stale beliefs against observations; do not spawn objects at remembered locations.

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1 only at true BASE surfaces.
- Map dining to table_2 in kitchen_0 and counter/sink-side placement to cabinet_0's surface. Map stale bathroom soap placement to washer_dryer_0, the available bathroom surface.
- Validate and use the kitchen faucet affordance for Clean/Fill; no invented sink or mandatory soap movement.
- Robot starts in living_room_0; no stacking.
