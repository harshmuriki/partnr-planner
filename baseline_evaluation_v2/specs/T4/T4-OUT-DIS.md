# T4-OUT-DIS

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
- toy_fruits: toy_fruits_0 is an accurately remembered distractor near the true apple location.
- spray_bottle: spray_bottle_0 is an accurately remembered distractor near the true bottle location.
- No substitutes.

## Uncertainty being tested
- Internal robot memory: Outdated
- Distractors: Present

## Information supplied in instruction
- Specifies three fruits, two drink containers, basket, kitchen counter, fruit source, and destinations.
- Omits distractors, instance ids, drink-source furniture, and the staleness of target locations.
- Toy fruits are not edible fruit; the spray bottle is not a drink container.

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
- toy_fruits_0 and spray_bottle_0 remain on counter_0, unselected; toy_fruits_0 remains away from basket_0.

## Initial robot memory
- apple_0: on table_4 (living_room_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on table_4 (living_room_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- toy_fruits_0: on counter_0 (kitchen_0), pickupable; fruit-side placement, away from basket_0.
- spray_bottle_0: on counter_0 (kitchen_0), pickupable; drink-side placement.
- Only selected-target locations are stale; no current target-proximity links are supplied through distractor records.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- Preserve is_on_top(toy_fruits_0, counter_0) and is_on_top(spray_bottle_0, counter_0); neither can replace a required target.

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 at unchanged BASE positions on counter_0.
- Add toy_fruits ×1 beside apple_0 and spray_bottle ×1 beside bottle_0 without moving targets. Spawn no task objects on table_4.
- Map kitchen table to counter_0 and stale dining table to table_4 in living_room_0. Start in kitchen_0.

## Constraints you MUST obey
- Keep distractor and remaining-object records accurate; apply stale memory only to the spreadsheet targets.
- Do not mistake nearby distractors for fruit or drinks. Use next_to basket_0 on counter_0, not containment or stacking.
