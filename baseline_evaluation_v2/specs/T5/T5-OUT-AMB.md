# T5-OUT-AMB

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
"Clean the two bottles and bring something to serve food on."

## Affected object(s)
- soap_dispenser: soap_dispenser_0, uncertainty target with stale bathroom memory.
- bottle: bottle_0, uncertainty target with stale counter memory; bottle_1, remaining task object.
- plate: plate_0, white uncertainty target with stale dining-table memory, suitable for serving food.
- No substitutes or distractors spawned.

## Uncertainty being tested
- Internal robot memory: Outdated
- Instruction: Underspecified

## Information supplied in instruction
- Mentions two bottles, cleaning, and an unspecified food-serving object to bring.
- Omits explicit soap use, filling, plate identity, source locations, destination room/table, adjacency, and any stale-memory warning.
- Canonical filling and placement goals below are evaluator intent, not supplied instructions; resolve omissions rather than treating them as explicit.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.

## Final expected world state
- Intended canonical completion after clarification: bottle_0 and bottle_1 on table_0 (living_room_0), clean, filled.
- plate_0: on table_0 (living_room_0), next to both bottles.
- soap_dispenser_0: on cabinet_0 (kitchen_0); no relocation required.

## Initial robot memory
- soap_dispenser_0: on washer_dryer_0 (bathroom_1), usable soap dispenser; stale location.
- bottle_0: on cabinet_0 (kitchen_0), dirty, empty; stale location.
- bottle_1: on table_2 (kitchen_0), dirty, empty; accurate.
- plate_0: on table_2 (kitchen_0), white, usable serving plate; stale location.

## Success criteria
- Canonical evaluation: is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- Canonical evaluation: is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- Canonical evaluation: is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).
- Seek clarification for omitted goals and verify stale environmental beliefs; this prompt alone does not uniquely determine canonical completion.

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1 at true BASE surface locations; do not spawn at stale locations.
- Dining maps to table_2 in kitchen_0; counter/sink-side placement to cabinet_0's surface; stale bathroom soap to washer_dryer_0.
- Validate and use the kitchen faucet affordance for Clean/Fill; soap need not move and no sink furniture id is invented.
- Robot starts in living_room_0. Do not silently expand the underspecified prompt; use adjacency, not stacking.
