# T4-OUT-BASE

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
- apple: apple_0 is an uncertainty target with stale location; apple_1 is a remaining task object.
- bottle: bottle_0 is an uncertainty target with stale location; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated

## Information supplied in instruction
- Specifies three fruits, two drink containers, basket, kitchen counter, fruit source, and destinations.
- Omits instance ids and drink-source furniture.
- The current fruit-source instruction conflicts with stale apple memory; current observation must override stale memory.

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
- Correct target memory from observations; no object must be transported from the falsely remembered table_4.

## Initial robot memory
- apple_0: on table_4 (living_room_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on table_4 (living_room_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- Only the selected apple and bottle have stale records.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 on counter_0 at BASE positions; spawn no task objects on table_4.
- Map kitchen table to counter_0, the kitchen island. Map the stale dining table to table_4 in living_room_0.
- Start in kitchen_0. Drink goals already hold in the true world; do not execute a pick against a memory-only object location.

## Constraints you MUST obey
- Stale placement changes memory only, not the world or remaining object records.
- Use fruits next_to basket_0 on counter_0 instead of unsupported basket containment or stacking.
