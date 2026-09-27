# T4-OUT-CON

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0; living_room_0 for stale dining-table memory
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90
  - cabinet_0 — kitchen_0 — Kitchen cabinet with drawers
  - fridge_0 — kitchen_0 — KEW - Fridge
  - table_4 — living_room_0 — "Alameda" Dining Table

## Task instruction / prompt given
"Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter."

## Affected object(s)
- apple: apple_0 is a contained uncertainty target remembered at an old surface location; apple_1 is a remaining task object.
- bottle: bottle_0 is a contained uncertainty target remembered at an old surface location; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies three fruits, two drink containers, basket, kitchen, and counter.
- Omits actual containment, door states, and stale dining-table records.
- Canonical counter-source wording is retained; current observation determines the selected apple's actual cabinet location.

## Initial world state
- apple_0: within cabinet_0 (kitchen_0), pickupable.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: within fridge_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- cabinet_0: Open/Close affordance; initially closed.
- fridge_0: Open/Close affordance; initially closed.

## Final expected world state
- apple_0, apple_1, and orange_0: on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1: on counter_0 (kitchen_0).
- Retrieve apple_0 from cabinet_0 and bottle_0 from fridge_0, not table_4. Final door positions are unconstrained.

## Initial robot memory
- apple_0: on table_4 (living_room_0), pickupable; stale on-surface location, with no current containment record.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on table_4 (living_room_0), pickupable; stale on-surface location, with no current containment record.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- cabinet_0: Open/Close affordance; initially closed.
- fridge_0: Open/Close affordance; initially closed.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Spawn apple_0 within cabinet_0 and bottle_0 within fridge_0; close both after initialization. Open to retrieve; closing afterward is optional.
- Spawn apple_1, orange_0, basket_0, and bottle_1 on counter_0 at unchanged BASE positions.
- Map kitchen table to counter_0 and stale dining table to table_4 in living_room_0. Start in kitchen_0.

## Constraints you MUST obey
- Only target locations and containment are stale in memory; furniture states and non-target records remain accurate.
- Do not spawn targets on table_4. Basket is pickupable; use fruit next_to basket_0 on counter_0 without stacking.
