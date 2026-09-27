# T4-ACC-ABS

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
- apple: apple_0 is the absent uncertainty target; apple_1 is a remaining task object.
- bottle: bottle_0 is an existing uncertainty target; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors exist. Only the selected apple is removed, as specified by the task row.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies three fruits, two drink containers, the basket, the kitchen, and the kitchen counter.
- Does not mention the absent apple. Counts and destinations remain fully specified.
- Omits instance ids, drink source, and any object-state operations.

## Initial world state
- apple_0: absent; no furniture placement.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No suitable replacement fruit exists.

## Final expected world state
- apple_0 remains absent. The planner gives up on the impossible third-fruit subgoal and reports that full completion is impossible.
- Feasible partial completion: apple_1 and orange_0 on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1 remain on counter_0 (kitchen_0).

## Initial robot memory
- apple_0: known absent; no furniture placement.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No suitable replacement fruit exists, and memory knows this.

## Success criteria
- Required ABS behavior: give up on the unavailable third fruit; do not invent apple_0, a substitute, or full completion.
- Remaining feasible subgoals may hold: is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0).
- Preserve is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Spawn apple ×1 as apple_1, orange ×1, basket ×1, bottle ×2, all on counter_0; do not spawn apple_0.
- Map the spreadsheet's kitchen table to counter_0 because no kitchen table is cataloged. Both drink goals already hold.
- Start in kitchen_0. Do not remove bottle_0 merely because it is also an uncertainty target.

## Constraints you MUST obey
- Missing fruit cannot be replaced by an invented instance or a nonfruit.
- Basket containment is replaced by fruit next_to basket_0 on counter_0; no stacking.
