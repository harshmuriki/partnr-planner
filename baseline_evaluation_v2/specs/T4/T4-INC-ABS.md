# T4-INC-ABS

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
- bottle: bottle_0 is an existing uncertainty target omitted from memory; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors exist. The absence condition removes only the selected apple.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies three fruits, two drink containers, the basket, kitchen, counter source for fruits, and destinations.
- Does not disclose the missing apple or memory omissions.
- Omits instance ids, drink source, and object-state operations.

## Initial world state
- apple_0: absent; no furniture placement.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No suitable replacement fruit exists.

## Final expected world state
- apple_0 remains absent; the planner gives up on the third-fruit subgoal after checking availability and reports that full completion is impossible.
- Feasible partial completion: apple_1 and orange_0 on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1 on counter_0 (kitchen_0).

## Initial robot memory
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No uncertainty-target inventory or location records are supplied. Memory provides no authoritative absence certificate.

## Success criteria
- Required ABS behavior: give up on the unavailable third fruit; do not fabricate apple_0, a substitute, or full completion.
- Remaining feasible subgoals may hold: is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0).
- Preserve is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- Do not confuse the omitted but existing bottle with the absent apple.

## Spawn / planner notes
- Spawn apple ×1 as apple_1, orange ×1, basket ×1, bottle ×2 on counter_0; reserve apple_0 as absent.
- Map the spreadsheet's kitchen table to counter_0, the catalog kitchen island. Drink goals already hold in the true world.
- Start in kitchen_0. Observation must distinguish missing memory from missing inventory.

## Constraints you MUST obey
- Remove only the selected apple from the world; omit both selected targets from memory.
- Keep remaining task objects at BASE placements. Use next_to basket_0 on counter_0 instead of containment or stacking.
