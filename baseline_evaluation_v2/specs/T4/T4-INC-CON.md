# T4-INC-CON

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90
  - cabinet_0 — kitchen_0 — Kitchen cabinet with drawers
  - fridge_0 — kitchen_0 — KEW - Fridge

## Task instruction / prompt given
"Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter."

## Affected object(s)
- apple: apple_0 is a contained uncertainty target omitted from memory; apple_1 is a remaining task object.
- bottle: bottle_0 is a contained uncertainty target omitted from memory; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies three fruits, two drink containers, basket, kitchen, and counter.
- Omits actual target containment, receptacle ids, door states, and instance ids.
- Retains canonical fruit-source wording; observation must resolve that the selected apple is actually in a cabinet.

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
- Selected objects are retrieved from cabinet_0 and fridge_0. Final door positions are unconstrained.

## Initial robot memory
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- cabinet_0: Open/Close affordance; initially closed.
- fridge_0: Open/Close affordance; initially closed.
- No uncertainty-target object records or containment links are supplied. Known furniture is not asserted to be empty.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).

## Spawn / planner notes
- Spawn apple_0 within cabinet_0 and bottle_0 within fridge_0. Close both receptacles after initialization.
- Spawn apple_1, orange_0, basket_0, and bottle_1 on counter_0 at BASE positions.
- Open receptacles to inspect and retrieve; optional closing afterward. Start in kitchen_0.
- BASE kitchen-table references map to counter_0, the kitchen island.

## Constraints you MUST obey
- Missing memory includes both targets and their containment, not remaining objects or furniture affordances.
- Only cabinet_0 and fridge_0 provide containment here. Basket is pickupable; place fruits next_to it on counter_0 without stacking.
