# T1-INC-BASE

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
- jug: jug_0, full-water uncertainty target, exists but is absent from initial robot memory.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete

## Information supplied in instruction
- Specifies all bedroom_0 lights, one full water jug, exact kitchen source furniture, and exact living-room destination.
- The instruction provides a search hint despite the missing memory record. No arrangement is requested.

## Initial world state
- jug_0: on counter_0 (kitchen_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- Articulated furniture requiring Open/Close: none.

## Final expected world state
- jug_0: on table_0 (living_room_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off.

## Initial robot memory
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- No jug object record, state, or placement is supplied; omission is not an assertion that no jug exists.

## Success criteria
- is_on_top(jug_0, table_0).
- is_filled(jug_0).
- is_powered_off(lamp_0).
- is_on_top(lamp_0, chest_of_drawers_0).
- Discover and ground the jug before manipulating it rather than inventing a remembered instance.

## Spawn / planner notes
- Spawn jug ×1 on counter_0, prefilled with water, and lamp ×1 on chest_of_drawers_0, powered on.
- Apply the memory omission only to the jug; retain the lamp record and unchanged physical world.
- PowerOff targets lamp_0, not ceiling fixtures. Robot starts in living_room_0.
