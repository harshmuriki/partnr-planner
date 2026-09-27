# T4-OUT-BASE

## Base task
Put all the 3 fruits in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: living_room_0, kitchen_0
- furniture used (id — room — catalog description):
  - table_4 — living_room_0 — "Alameda" Dining Table
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90
  - cabinet_1 — kitchen_0 — Kitchen cabinet with drawers

## Task instruction / prompt given
"Put all the 3 fruits in the fruit basket and the 2 drink containers on the kitchen counter."

## Affected object(s)
- apple_0 (apple, asset Apple_26): uncertainty target.
- apple_1 (apple, asset Apple_4): remaining task object.
- orange_0 (orange, asset 017_orange): remaining task object.
- basket_0 (basket, asset 87ed6d0c785b3245207bb217eb3ee3fc079f8633): remaining task object.
- bottle_0 (bottle, asset 2f92d709c9e94753a896c065cb62ed33765837c0): uncertainty target.
- bottle_1 (bottle, asset xxxxe3f9a974xfd88x433bx92c8x5f4d72d43be3): remaining task object.

## Entity registry
- apple_0: apple, uncertainty target
- apple_1: apple, remaining task object
- orange_0: orange, remaining task object
- basket_0: basket, remaining task object
- bottle_0: bottle, uncertainty target
- bottle_1: bottle, remaining task object

## Uncertainty being tested
- Internal robot memory: Outdated
- Memory in this version: robot memory places apple_0, apple_1, orange_0, bottle_0, bottle_1 at stale locations.

## Information supplied in instruction
- Specified: the object counts (3 fruits, 2 drink containers), the fruit basket, and the kitchen counter.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- apple_0: on counter_0 (kitchen_0), is_clean
- apple_1: on counter_0 (kitchen_0), is_clean
- orange_0: on counter_0 (kitchen_0)
- basket_0: on counter_0 (kitchen_0), is_clean, is_empty
- bottle_0: on cabinet_1 (kitchen_0), is_clean, is_empty
- bottle_1: on cabinet_1 (kitchen_0), is_clean, is_empty

## Final expected world state
- apple_0: within basket_0 (kitchen_0), is_clean
- apple_1: within basket_0 (kitchen_0), is_clean
- orange_0: within basket_0 (kitchen_0)
- basket_0: on counter_0 (kitchen_0), is_clean, is_empty
- bottle_0: on counter_0 (kitchen_0), is_clean, is_empty
- bottle_1: on counter_0 (kitchen_0), is_clean, is_empty

## Initial robot memory
- apple_0: on table_4 (living_room_0), is_clean, stale record
- apple_1: on table_4 (living_room_0), is_clean, stale record
- orange_0: on table_4 (living_room_0), stale record
- basket_0: on counter_0 (kitchen_0), is_clean, is_empty
- bottle_0: on table_4 (living_room_0), is_clean, is_empty, stale record
- bottle_1: on table_4 (living_room_0), is_clean, is_empty, stale record

## Success criteria
- is_inside(apple_0, basket_0)
- is_inside(apple_1, basket_0)
- is_inside(orange_0, basket_0)
- is_on_top(basket_0, counter_0)
- is_on_top(bottle_0, counter_0)
- is_on_top(bottle_1, counter_0)

## Spawn / planner notes
- Scene 107734176_176000019; the robot starts in kitchen_0.
- Spawn apple x1 as apple_0 on counter_0 (kitchen_0), pinned asset Apple_26; start states: is_clean.
- Spawn apple x1 as apple_1 on counter_0 (kitchen_0), pinned asset Apple_4; start states: is_clean.
- Spawn orange x1 as orange_0 on counter_0 (kitchen_0), pinned asset 017_orange; start states: no object states.
- Spawn basket x1 as basket_0 on counter_0 (kitchen_0), pinned asset 87ed6d0c785b3245207bb217eb3ee3fc079f8633; start states: is_clean, is_empty.
- Spawn bottle x1 as bottle_0 on cabinet_1 (kitchen_0), pinned asset 2f92d709c9e94753a896c065cb62ed33765837c0; start states: is_clean, is_empty.
- Spawn bottle x1 as bottle_1 on cabinet_1 (kitchen_0), pinned asset xxxxe3f9a974xfd88x433bx92c8x5f4d72d43be3; start states: is_clean, is_empty.
- The kitchen has no table: the top of cabinet_1 (kitchen cabinet with drawers) stands in for the kitchen table where the drinks start. The kitchen counter is the island counter_0.
- The dining table used for stale memory is table_4, the dining table in living_room_0.
- Fruits must end physically inside basket_0, scored with is_inside(fruit, basket_0). The basket stays on counter_0. Do not substitute next-to or on-counter goals for containment.
- Basket containment support, physical fit and planner placement into this pickupable asset require verification before generation is unblocked. The basket's is_empty state means empty of water, not empty of fruit.
- Per the sheet, Incomplete memory loses all three fruits and both drinks (only the basket is known) and Outdated memory places all five on the dining table; apple_0 and bottle_0 are the selected targets.
- CON puts all five fruits and drinks inside fridge_0, as the sheet says; check that they physically fit.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
