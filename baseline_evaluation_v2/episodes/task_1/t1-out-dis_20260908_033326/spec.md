# T1-OUT-DIS

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
- jug: jug_0, full-water uncertainty target with stale placement memory; jug_1, empty distractor.
- vase: vase_0, distractor.
- spray_bottle: spray_bottle_0, distractor.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No substitutes.

## Uncertainty being tested
- Internal robot memory: Outdated
- Distractors: Present

## Information supplied in instruction
- Specifies all bedroom_0 lights, one full water jug, exact kitchen source, and exact living-room destination.
- Omits distractors, their arrangement, and stale-memory status. Fullness identifies the intended jug.

## Initial world state
- jug_0: on counter_0 (kitchen_0), filled with water.
- jug_1: on counter_0 (kitchen_0), empty, near jug_0.
- vase_0: on counter_0 (kitchen_0), empty, near jug_0.
- spray_bottle_0: on counter_0 (kitchen_0), empty, near jug_0.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- Articulated furniture requiring Open/Close: none.

## Final expected world state
- jug_0: on table_0 (living_room_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off.
- jug_1, vase_0, spray_bottle_0: on counter_0 (kitchen_0), remain empty.

## Initial robot memory
- jug_0: on table_0 (living_room_0), filled with water; stale placement.
- jug_1: on counter_0 (kitchen_0), empty.
- vase_0: on counter_0 (kitchen_0), empty.
- spray_bottle_0: on counter_0 (kitchen_0), empty.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- True proximity relations involving jug_0 are not retained with its stale placement; other object facts are unchanged.

## Success criteria
- is_on_top(jug_0, table_0) and is_filled(jug_0).
- is_powered_off(lamp_0) and is_on_top(lamp_0, chest_of_drawers_0).
- is_on_top(jug_1, counter_0), is_on_top(vase_0, counter_0), is_on_top(spray_bottle_0, counter_0).
- Deliver the originally full target and validate its actual destination placement; neither distractors nor stale memory satisfy delivery.

## Spawn / planner notes
- Spawn jug ×2, vase ×1, spray_bottle ×1 on counter_0, with empty distractors adjacent to full jug_0 and no stacking. Spawn lamp ×1 on chest_of_drawers_0, powered on.
- The catalog lacks a dining room; map the stale dining table to dining-style table_0 in living_room_0. Do not spawn a second full jug there.
- PowerOff uses lamp_0, not ceiling fixtures. Robot starts in living_room_0.
