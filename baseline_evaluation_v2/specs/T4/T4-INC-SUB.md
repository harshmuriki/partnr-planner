# T4-INC-SUB

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90

## Task instruction / prompt given
"Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter."

## Affected object(s)
- apple: apple_0 is the absent uncertainty target; apple_1 remains spawned.
- bottle: bottle_0 is an existing uncertainty target omitted from memory; bottle_1 remains represented.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- Substitutes: none; the task row specifies N/A. No distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies three fruits, two drink containers, the basket, kitchen, and counter.
- Does not disclose missing inventory or identify a substitute.
- This file supplies a missing-apple diagnostic fixture, not a valid SUB trial; the row supplies no substitute to omit from memory.

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
- apple_0 remains absent. After checking available inventory, report the unsatisfied third-fruit requirement.

## Initial robot memory
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No uncertainty-target inventory or location records are provided, nor an absence certificate or substitute record.

## Success criteria
- SUB scoring is disabled: no row-authorized substitute exists and no substitute-success proposition can be instantiated.
- Diagnostic behavior: inspect rather than assuming memory omission means absence, then give up on the missing third fruit without inventing it.
- Feasible diagnostic propositions: is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0).
- Preserve is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Retain this filename as an explicitly unsupported SUB cell; exclude it from executable SUB benchmark scoring.
- Spawn apple ×1 as apple_1, orange ×1, basket ×1, bottle ×2 on counter_0. Do not spawn apple_0 or a substitute.
- Map kitchen table to counter_0, the kitchen island. Both bottle goals hold in the world. Start in kitchen_0.

## Constraints you MUST obey
- Do not invent a substitute to fill an N/A spreadsheet cell or label diagnostic abstention as SUB success.
- Keep existing bottle_0 in the world but out of initial memory. Use next_to basket_0 on counter_0, without stacking.
