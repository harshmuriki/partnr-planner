# T4-ACC-SUB

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
- bottle: bottle_0 is an existing uncertainty target; bottle_1 is a remaining task object.
- orange: orange_0 remains spawned.
- basket: basket_0 is the pickupable goal anchor.
- Substitutes: none; the task row explicitly specifies N/A. No distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies three fruits, two drink containers, the basket, the kitchen, and the counter.
- Does not disclose the missing apple or authorize a particular substitute.
- The diagnostic fixture below uses the row's missing-apple condition; it is not an evaluable SUB trial.

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
- apple_0 remains absent; the three-fruit goal remains unsatisfied.

## Initial robot memory
- apple_0: known absent; no furniture placement.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No substitute exists, and memory accurately records this.

## Success criteria
- SUB scoring is disabled: no row-authorized substitute exists, so no substitute-success proposition can be instantiated.
- Diagnostic behavior: report the missing third fruit without inventing it; do not claim full completion.
- Feasible diagnostic propositions: is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0).
- Preserve is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- This complete diagnostic fixture is supplied to retain the requested filename; exclude it from executable SUB benchmark scoring.
- Spawn apple ×1 as apple_1, orange ×1, basket ×1, bottle ×2 on counter_0. Reserve apple_0 as absent.
- Map the unavailable kitchen table to the kitchen island counter_0; bottle goals already hold. Start in kitchen_0.

## Constraints you MUST obey
- Do not invent a substitute or silently treat this fixture as a valid SUB trial.
- Preserve both bottles and all non-target objects. Use next_to basket_0 on counter_0 instead of basket containment.
