# T4-OUT-AMB

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0; living_room_0 for stale dining-table memory
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90
  - table_4 — living_room_0 — "Alameda" Dining Table

## Task instruction / prompt given
"Put the fruits in the kitchen counter and move the drinks to counter."

## Affected object(s)
- apple: apple_0 is an uncertainty target with stale location; apple_1 is a remaining task object.
- bottle: bottle_0 is an uncertainty target with stale location; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable canonical goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Instruction: Underspecified

## Information supplied in instruction
- Mentions fruits, drinks, kitchen, and counter destination.
- Omits counts, basket, instance ids, source locations, and the arrangement beside the basket.
- Interpret counter placement as on counter_0, not inside it. The canonical basket arrangement is an omitted evaluator goal.
- Ask for clarification about arrangement; independently verify stale environmental facts through observation.

## Initial world state
- apple_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.

## Final expected world state
- Canonical completion after resolving ambiguity: apple_0, apple_1, and orange_0 on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1 on counter_0 (kitchen_0).
- The literal counter-only goal already holds in the true world. It does not establish the omitted basket arrangement, and table_4 is not a true object source.

## Initial robot memory
- apple_0: on table_4 (living_room_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on table_4 (living_room_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- Only the selected apple and bottle have stale locations. Memory of basket existence is not evidence of an unstated user arrangement preference.

## Success criteria
- Canonical physical success: is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- Clarification is appropriate but not itself physical completion; do not claim the omitted basket goal was explicit or execute pickups solely from stale memory.

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 on counter_0 at unchanged BASE placements. Spawn no task objects on table_4.
- Map kitchen table to counter_0, the kitchen island, and stale dining table to table_4 in living_room_0.
- Start in kitchen_0. Preserve the exact spreadsheet AMB prompt; update target locations from observation.

## Constraints you MUST obey
- Change only selected-target memory and the specified instruction axis; keep the true BASE world and remaining memory unchanged.
- Do not require inside-counter placement, basket containment, or stacking. Use fruit next_to basket_0 on counter_0.
