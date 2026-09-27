# T4-ACC-DIS

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
- toy_fruits: toy_fruits_0 is a distractor near apple_0.
- spray_bottle: spray_bottle_0 is a distractor near bottle_0.
- No substitutes.

## Uncertainty being tested
- Internal robot memory: Accurate
- Distractors: Present

## Information supplied in instruction
- Specifies three fruits, two drink containers, the basket, the kitchen, and the kitchen counter.
- Does not name distractors, instance ids, or drink-source furniture.
- Toy fruit is not edible fruit, and the spray bottle is not a drink container.

## Initial world state
- apple_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- toy_fruits_0: on counter_0 (kitchen_0), pickupable; fruit-side placement, away from basket_0.
- spray_bottle_0: on counter_0 (kitchen_0), pickupable; drink-side placement.

## Final expected world state
- apple_0, apple_1, and orange_0: on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1: on counter_0 (kitchen_0).
- toy_fruits_0 and spray_bottle_0 remain on counter_0, unselected as task objects; toy_fruits_0 remains away from basket_0.

## Initial robot memory
- apple_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- toy_fruits_0: on counter_0 (kitchen_0), pickupable; fruit-side placement, away from basket_0.
- spray_bottle_0: on counter_0 (kitchen_0), pickupable; drink-side placement.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- Preserve is_on_top(toy_fruits_0, counter_0) and is_on_top(spray_bottle_0, counter_0); neither distractor satisfies a required fruit or drink slot.

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 at unchanged BASE positions on counter_0.
- Add toy_fruits ×1 beside apple_0 and spray_bottle ×1 beside bottle_0, without moving targets.
- Map the uncataloged kitchen table to counter_0's drink-side staging area. Bottle goals already hold. Start in kitchen_0.

## Constraints you MUST obey
- Do not count toy fruits or spray bottles toward the requested objects.
- Basket is not a receptacle; use next_to on counter_0 and no stacking.
