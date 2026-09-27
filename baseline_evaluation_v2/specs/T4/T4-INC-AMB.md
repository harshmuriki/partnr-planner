# T4-INC-AMB

## Base task
Put all the 3 fruits from the kitchen counter in the fruit basket and the 2 drink containers on the kitchen counter.

## Scene
- scene_id: 107734176_176000019
- rooms used: kitchen_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — Kitchen island, 60x150x90

## Task instruction / prompt given
"Put the fruits in the kitchen counter and move the drinks to counter."

## Affected object(s)
- apple: apple_0 is an uncertainty target omitted from memory; apple_1 is a remaining task object.
- bottle: bottle_0 is an uncertainty target omitted from memory; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable canonical goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Instruction: Underspecified

## Information supplied in instruction
- Mentions fruits, drinks, kitchen, and counter destination.
- Omits counts, basket, instance ids, source locations, and the arrangement beside the basket.
- Interpret placement in the counter as on counter_0. Canonical basket arrangement is an omitted evaluator goal, not an explicit instruction.
- Ask for clarification about intended inventory and arrangement; observation is also needed for missing environmental records.

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
- The literal counter-only reading already holds initially; it does not establish the omitted basket arrangement or justify ignoring unremembered objects.

## Initial robot memory
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.
- No uncertainty-target inventory or location records are supplied; the remembered object list is not exhaustive.

## Success criteria
- Canonical physical success: is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- Clarification is appropriate but does not itself satisfy the physical propositions; do not claim the omitted basket goal was explicit.

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 on counter_0 at unchanged BASE placements.
- Map the spreadsheet kitchen table to counter_0, the kitchen island. Bottle goals already hold in the world.
- Start in kitchen_0. Preserve the exact AMB prompt and omit only the two selected targets from memory.

## Constraints you MUST obey
- Do not equate incomplete memory with absent objects or infer a user preference solely from basket existence.
- No inside-counter placement, basket containment, or stacking; use next_to basket_0 on counter_0.
