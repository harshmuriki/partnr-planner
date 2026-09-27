# T4-ACC-BASE

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
- apple: apple_0 is an uncertainty target; apple_1 is a remaining task object.
- bottle: bottle_0 is an uncertainty target; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate

## Information supplied in instruction
- Specifies three fruits, two drink containers, the fruit basket, the kitchen, and the kitchen counter.
- Specifies the fruits' source and both destination arrangements, but not fruit species, instance ids, or the drinks' source.
- No cleaning, filling, or power state is requested.

## Initial world state
- apple_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.

## Final expected world state
- apple_0, apple_1, and orange_0: on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1: on counter_0 (kitchen_0).
- Fruit placement beside the basket replaces unsupported containment in a pickupable basket.

## Initial robot memory
- apple_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2, all on counter_0.
- The catalog has no kitchen table. Map the spreadsheet's kitchen-table source to counter_0, the kitchen island, using a separate staging region from the basket.
- The drinks' destination propositions therefore already hold; unnecessary bottle movement is not required.
- Start the robot in kitchen_0. No articulated receptacles or lamps are needed.

## Constraints you MUST obey
- Use only the listed catalog furniture and allowed spawn classes.
- Basket is pickupable, not a receptacle. Do not stack fruits or require is_inside with basket_0.
- Keep these non-target placements unchanged in all variants unless the task itself moves them.
