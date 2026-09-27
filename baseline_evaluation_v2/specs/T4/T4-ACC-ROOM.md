# T4-ACC-ROOM

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90

## Task instruction / prompt given
"Find all 3 fruits and both drink containers in the kitchen. Put the fruits in the fruit basket and the 2 drink containers on the kitchen counter."

## Affected object(s)
- apple: apple_0 is a room-localized uncertainty target; apple_1 is a remaining task object.
- bottle: bottle_0 is a room-localized uncertainty target; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Room Known

## Information supplied in instruction
- Specifies all three fruits, both drink containers, kitchen source room, basket destination, and kitchen-counter drink destination.
- Withholds source furniture for the selected apple and bottle, as well as instance ids.
- The exact goal counter remains known; knowing a destination does not reveal either target's starting support.

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

## Initial robot memory
- apple_0: in_room kitchen_0, pickupable; exact support and position withheld.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: in_room kitchen_0, pickupable; exact support and position withheld.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- Target existence and room facts are accurate; do not provide hidden exact source supports to the planner.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- in_room(apple_0, kitchen_0) and in_room(bottle_0, kitchen_0) hold throughout the grounded completion.

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 on counter_0 at unchanged BASE positions.
- Map the spreadsheet kitchen table to counter_0, the kitchen island; bottle destination propositions already hold in the true world.
- Start in kitchen_0. Reveal exact target supports only through observation, not initial memory.

## Constraints you MUST obey
- Localization uncertainty changes the target information, not true placements or non-target memory.
- Use next_to basket_0 on counter_0, never inside basket_0 or stacked fruit.
