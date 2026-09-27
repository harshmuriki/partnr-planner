# T4-ACC-CAND

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0; living_room_0 as the candidate dining area
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90
  - table_4 — living_room_0 — "Alameda" Dining Table

## Task instruction / prompt given
"Find all 3 fruits and both drink containers in the kitchen or the dining area of the living room. Put the fruits in the fruit basket and the 2 drink containers on the kitchen counter."

## Affected object(s)
- apple: apple_0 is a candidate-room uncertainty target; apple_1 is a remaining task object.
- bottle: bottle_0 is a candidate-room uncertainty target; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Candidate Rooms

## Information supplied in instruction
- Specifies three fruits, two drink containers, the basket, and the kitchen-counter destination.
- Supplies kitchen or dining area as source candidates, not an exact target room or support.
- The spreadsheet's dining room maps to the dining area in living_room_0; no separate dining_room id exists.

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
- No task object must be placed on table_4.

## Initial robot memory
- apple_0: in one of {kitchen_0, living_room_0}, pickupable; exact room, support, and position withheld.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: in one of {kitchen_0, living_room_0}, pickupable; exact room, support, and position withheld.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- Candidate sets are accurate disjunctions, not claims that targets occupy both rooms. Neither target has a remembered furniture id.

## Success criteria
- is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- Final in_room(apple_0, kitchen_0) and in_room(bottle_0, kitchen_0).

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 on counter_0 at BASE positions; spawn no task objects on table_4.
- Map kitchen table to counter_0 and the candidate dining area to the area containing table_4 in living_room_0.
- Start in kitchen_0. Keep target support facts out of planner-visible initial memory.

## Constraints you MUST obey
- Do not invent a dining_room id or move targets to a candidate room merely to create uncertainty.
- Basket is pickupable; use fruit next_to basket_0 on counter_0 without stacking.
