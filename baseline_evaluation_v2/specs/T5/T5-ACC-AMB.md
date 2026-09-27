# T5-ACC-AMB

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
"Clean the two bottles and bring something to serve food on."

## Affected object(s)
- soap_dispenser: soap_dispenser_0, uncertainty target.
- bottle: bottle_0, uncertainty target; bottle_1, remaining task object.
- plate: plate_0, white appearance, uncertainty target and suitable referent for serving food.
- No substitutes or distractors spawned.

## Uncertainty being tested
- Internal robot memory: Accurate
- Instruction: Underspecified

## Information supplied in instruction
- Mentions two bottles, cleaning, and an unspecified food-serving object to bring.
- Omits explicit soap use, filling, plate identity, source rooms, destination room/table, and adjacency arrangement.
- The canonical destination and filling/arrangement goals below are evaluator intent, not facts communicated by this prompt. Seek clarification rather than claiming they are explicit.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.

## Final expected world state
- Intended canonical completion after resolving omitted goals: bottle_0 and bottle_1 on table_0 (living_room_0), clean, filled.
- plate_0: on table_0 (living_room_0), next to both bottles.
- soap_dispenser_0: on cabinet_0 (kitchen_0); no relocation required.

## Initial robot memory
- soap_dispenser_0: on cabinet_0 (kitchen_0), usable soap dispenser.
- bottle_0: on table_2 (kitchen_0), dirty, empty.
- bottle_1: on table_2 (kitchen_0), dirty, empty.
- plate_0: on cabinet_0 (kitchen_0), white, usable serving plate.

## Success criteria
- Canonical evaluation: is_on_top(bottle_0, table_0); is_clean(bottle_0); is_filled(bottle_0).
- Canonical evaluation: is_on_top(bottle_1, table_0); is_clean(bottle_1); is_filled(bottle_1).
- Canonical evaluation: is_on_top(plate_0, table_0); is_next_to(plate_0, bottle_0); is_next_to(plate_0, bottle_1).
- A clarification request is appropriate for omitted goals; the prompt alone does not uniquely determine the complete canonical state.

## Spawn / planner notes
- Spawn soap_dispenser ×1, bottle ×2, plate ×1 on the listed surfaces; the physical world is unchanged from BASE.
- Map dining to table_2 in kitchen_0 and counter/sink-side placement to cabinet_0's surface.
- Validate and use the kitchen faucet affordance for Clean/Fill; no invented sink or mandatory soap movement.
- Robot starts in living_room_0. Do not silently expand the underspecified prompt; no stacking.
