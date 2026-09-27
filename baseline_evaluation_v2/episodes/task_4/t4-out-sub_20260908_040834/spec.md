# T4-OUT-SUB

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0; living_room_0 for stale dining-table memory
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90
  - table_4 — living_room_0 — "Alameda" Dining Table

## Task instruction / prompt given
"Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter."

## Affected object(s)
- apple: apple_0 is absent but falsely remembered as an uncertainty target; apple_1 remains spawned.
- bottle: bottle_0 is an existing uncertainty target with stale location; bottle_1 remains accurately represented.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- Substitutes: none; the task row specifies N/A. No distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies three fruits, two drink containers, basket, kitchen, and counter.
- Does not disclose absence or stale memory and does not authorize a substitute.
- This file supplies a missing-apple diagnostic fixture, not an evaluable SUB trial.

## Initial world state
- apple_0: absent; no furniture placement.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No substitute exists.

## Final expected world state
- No valid SUB completion can be specified from this task row.
- Diagnostic feasible outcome: apple_1 and orange_0 on counter_0 (kitchen_0), each next_to basket_0; basket_0, bottle_0, and bottle_1 on counter_0 (kitchen_0).
- apple_0 remains absent. Correct the false inventory claim and report the impossible third-fruit requirement.

## Initial robot memory
- apple_0: on table_4 (living_room_0), pickupable; not next_to basket_0. This existence and location claim is false.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on table_4 (living_room_0), pickupable. This location is stale.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No substitute record is supplied.

## Success criteria
- SUB scoring is disabled: no row-authorized substitute exists, so no substitute-success proposition can be instantiated.
- Diagnostic behavior: reject stale existence claims after checking; do not invent apple_0 or claim full completion.
- Feasible diagnostic propositions: is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0).
- Preserve is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Retain this explicitly unsupported SUB file but exclude it from executable SUB benchmark scoring.
- Spawn apple ×1 as apple_1, orange ×1, basket ×1, bottle ×2 on counter_0. Spawn no apple_0 or substitute.
- Map kitchen table to counter_0 and stale dining table to table_4 in living_room_0. Start in kitchen_0.

## Constraints you MUST obey
- Do not invent an N/A substitute or instantiate memory-only objects in the world.
- Preserve non-target BASE placements. Use next_to basket_0 on counter_0 without containment or stacking.
