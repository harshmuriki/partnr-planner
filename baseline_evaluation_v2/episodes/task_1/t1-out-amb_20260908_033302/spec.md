# T1-OUT-AMB

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
"Bring me water to the living room and turn of the lights in the bedroom ."

## Affected object(s)
- jug: jug_0, full-water uncertainty target and only water container, remembered at the wrong support.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Instruction: Underspecified

## Information supplied in instruction
- Mentions water, living room, bedroom, and lights off.
- Omits container class/count, fullness, source room/furniture, exact bedroom identifier, and destination table.
- Memory gives an exact but stale target location. Canonical evaluation bindings remain bedroom_0 and table_0; no handover or recipient object is required.

## Initial world state
- jug_0: on counter_0 (kitchen_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- Articulated furniture requiring Open/Close: none.

## Final expected world state
- jug_0: on table_0 (living_room_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off.

## Initial robot memory
- jug_0: on table_0 (living_room_0), filled with water; injected stale placement.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.

## Success criteria
- is_on_top(jug_0, table_0).
- is_filled(jug_0).
- in_room(jug_0, living_room_0).
- is_powered_off(lamp_0) and is_on_top(lamp_0, chest_of_drawers_0).
- Resolve omitted delivery details to the canonical task bindings and verify physical completion, not merely the stale memory claim. A clarification question alone does not satisfy the goals.

## Spawn / planner notes
- Spawn jug ×1 on counter_0, prefilled with water, and lamp ×1 on chest_of_drawers_0, powered on.
- Preserve the AMB prompt verbatim. Map the spreadsheet's stale dining table to dining-style table_0 in living_room_0 because no dining room is cataloged; do not reveal the true source in initial memory.
- PowerOff uses lamp_0, not ceiling fixtures. Robot starts in living_room_0; table_0 is the canonical delivery support.
