# T1-OUT-BASE

## Base task
Turn off all the lights in the bedroom and bring a full water jug from the kitchen to the living-room table. Resolve the bedroom to bedroom_0, its controllable lights to lamp_0 on chest_of_drawers_0, the kitchen counter to counter_0, and the destination to table_0 in living_room_0.

## Scene
- scene_id: 103997895_171031182
- rooms used: bedroom_0, kitchen_0, living_room_0
- furniture used (id — room — catalog description):
  - chest_of_drawers_0 — bedroom_0 — Presby Nightstand, White
  - counter_0 — kitchen_0 — Kitchen island, 60x100x90
  - table_0 — living_room_0 — Tulip Table (90cm)

## Task instruction / prompt given
"Turn off all the lights in bedroom_0 and bring the full water jug on the kitchen island to the Tulip table in the living room."

## Affected object(s)
- jug: jug_0, full-water uncertainty target, truly on counter_0 but remembered on table_0.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated

## Information supplied in instruction
- Specifies all bedroom_0 lights, one full water jug, exact current kitchen source, and exact living-room destination.
- Does not disclose that stored target memory is stale; its source hint conflicts with that memory. No arrangement is requested.

## Initial world state
- jug_0: on counter_0 (kitchen_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- Articulated furniture requiring Open/Close: none.

## Final expected world state
- jug_0: on table_0 (living_room_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off.

## Initial robot memory
- jug_0: on table_0 (living_room_0), filled with water. This is the injected stale record, not a true placement.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.

## Success criteria
- is_on_top(jug_0, table_0).
- is_filled(jug_0).
- is_powered_off(lamp_0).
- is_on_top(lamp_0, chest_of_drawers_0).
- Validate the physical delivery rather than treating the stale destination placement as already satisfied.

## Spawn / planner notes
- Spawn jug ×1 on counter_0, prefilled with water, and lamp ×1 on chest_of_drawers_0, powered on.
- There is no catalog dining room; the dining-style Tulip table, table_0 in living_room_0, is the stale dining-table proxy. Do not spawn a duplicate jug there.
- PowerOff uses lamp_0, not ceiling fixtures. Robot starts in living_room_0.
