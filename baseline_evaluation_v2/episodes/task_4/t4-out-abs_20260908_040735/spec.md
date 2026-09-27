# T4-OUT-ABS

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
- apple: apple_0 is the absent uncertainty target falsely remembered as present; apple_1 is a remaining task object.
- bottle: bottle_0 exists but has a stale remembered location; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors. The absence condition removes only apple_0.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies three fruits, two drink containers, basket, kitchen counter, fruit source, and destinations.
- Omits the absent apple and the fact that selected-object memory is stale.
- No instance ids, drink-source furniture, or object-state operations are specified.

## Initial world state
- apple_0: absent; no furniture placement.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No suitable replacement fruit exists.

## Final expected world state
- apple_0 remains absent; the planner gives up on the third-fruit subgoal after checking the stale claim and available inventory.
- Feasible partial completion: apple_1 and orange_0 on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1 on counter_0 (kitchen_0). Report partial, not full, completion.

## Initial robot memory
- apple_0: on table_4 (living_room_0), pickupable; not next_to basket_0. This existence and location claim is false.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on table_4 (living_room_0), pickupable. This location is stale.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- Memory falsely suggests that the third required fruit is available.

## Success criteria
- Required ABS behavior: abandon the unavailable third-fruit subgoal; do not invent apple_0, a substitute, or a successful pickup at table_4.
- Remaining feasible subgoals may hold: is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0).
- Preserve is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Spawn apple ×1 as apple_1, orange ×1, basket ×1, bottle ×2 on counter_0; never spawn apple_0 to match memory.
- Map kitchen table to counter_0, the island, and stale dining table to table_4 in living_room_0.
- Start in kitchen_0. Both drink goals already hold in the true world; verify rather than relying on stale source records.

## Constraints you MUST obey
- World absence overrides a remembered existence claim. Do not remove the still-existing selected bottle.
- Use fruit next_to basket_0 on counter_0, not basket containment or stacking.
