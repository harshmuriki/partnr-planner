# T4-ACC-AMB

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
- apple: apple_0 is an uncertainty target; apple_1 is a remaining task object.
- bottle: bottle_0 is an uncertainty target; bottle_1 is a remaining task object.
- orange: orange_0 is a remaining task object.
- basket: basket_0 is the pickupable canonical goal anchor.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Instruction: Underspecified

## Information supplied in instruction
- Mentions fruits, drinks, the kitchen, and a counter destination.
- Omits both counts, the basket, fruit species, instance ids, source locations, and the arrangement beside the basket.
- Interpret counter placement legally as on counter_0, not inside the counter.
- The canonical basket arrangement is an evaluator goal omitted from the prompt; ask for clarification rather than treating it as explicitly stated.

## Initial world state
- apple_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.

## Final expected world state
- Canonical completion after resolving the omitted arrangement: apple_0, apple_1, and orange_0 on counter_0 (kitchen_0), each next_to basket_0.
- basket_0, bottle_0, and bottle_1 on counter_0 (kitchen_0).
- The literal counter-only reading already holds initially and must not be misreported as proof of the omitted basket arrangement.

## Initial robot memory
- apple_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- apple_1: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- orange_0: on counter_0 (kitchen_0), pickupable; not next_to basket_0.
- basket_0: on counter_0 (kitchen_0), pickupable.
- bottle_0: on counter_0 (kitchen_0), pickupable.
- bottle_1: on counter_0 (kitchen_0), pickupable.

## Success criteria
- Canonical physical success: is_on_top(apple_0, counter_0), is_on_top(apple_1, counter_0), is_on_top(orange_0, counter_0).
- is_next_to(apple_0, basket_0), is_next_to(apple_1, basket_0), is_next_to(orange_0, basket_0), all on counter_0.
- is_on_top(basket_0, counter_0), is_on_top(bottle_0, counter_0), is_on_top(bottle_1, counter_0).
- A clarification request is appropriate ambiguity handling but is not itself physical task completion.

## Spawn / planner notes
- Spawn apple ×2, orange ×1, basket ×1, bottle ×2 on counter_0 at BASE positions.
- Map the spreadsheet kitchen table to the catalog kitchen island counter_0. Both bottle destination propositions already hold.
- Start in kitchen_0. Do not reveal missing counts or the basket arrangement through an altered initial prompt.

## Constraints you MUST obey
- Preserve the spreadsheet's underspecified instruction verbatim.
- World knowledge does not supply an unstated user preference. Do not use inside-counter placement or basket containment; no stacking.
