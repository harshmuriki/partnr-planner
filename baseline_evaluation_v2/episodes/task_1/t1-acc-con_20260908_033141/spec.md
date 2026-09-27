# T1-ACC-CON

## Base task
Turn off all the lights in the bedroom and bring a full water jug from the kitchen to the living-room table. Resolve the bedroom to bedroom_0, its controllable lights to lamp_0 on chest_of_drawers_0, the kitchen counter to counter_0, and the destination to table_0 in living_room_0.

## Scene
- scene_id: 103997895_171031182
- rooms used: bedroom_0, kitchen_0, living_room_0
- furniture used (id — room — catalog description):
  - chest_of_drawers_0 — bedroom_0 — Presby Nightstand, White
  - counter_0 — kitchen_0 — Kitchen island, 60x100x90
  - table_0 — living_room_0 — Tulip Table (90cm)
  - fridge_0 — kitchen_0 — American fridge freezer

## Task instruction / prompt given
"Turn off all the lights in bedroom_0 and bring the full water jug inside the closed kitchen fridge to the Tulip table in the living room."

## Affected object(s)
- jug: jug_0, uncertainty target, full of water inside fridge_0.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies all bedroom_0 lights, one full water jug, exact fridge source and closed state, and exact living-room destination.
- No relative arrangement or final fridge state is requested.

## Initial world state
- jug_0: within fridge_0 (kitchen_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- fridge_0: closed.
- Articulated furniture requiring Open/Close: fridge_0 supports Open/Close and must start closed.

## Final expected world state
- jug_0: on table_0 (living_room_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off.
- fridge_0 may end open or closed after retrieval.

## Initial robot memory
- jug_0: within fridge_0 (kitchen_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- fridge_0: closed.

## Success criteria
- is_on_top(jug_0, table_0).
- is_filled(jug_0).
- is_powered_off(lamp_0).
- is_on_top(lamp_0, chest_of_drawers_0).
- Initial containment is is_inside(jug_0, fridge_0); final delivery must remove it from the fridge.

## Spawn / planner notes
- Spawn jug ×1 within fridge_0, prefilled with water; close fridge_0 before evaluation.
- Spawn lamp ×1 on chest_of_drawers_0, powered on; it represents all task bedroom lights.
- Enable Open/Close in addition to navigation, manipulation, and PowerOff. Do not target ceiling fixtures. Robot starts in living_room_0.
