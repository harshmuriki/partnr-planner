# T1-OUT-SUB

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
- jug: jug_0, requested uncertainty target, physically absent but present in stale memory.
- pitcher: pitcher_0, full-water substitute, physically present but not remembered.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies all bedroom_0 lights, one full water jug, exact kitchen source, and exact living-room destination.
- Does not mention target absence, substitute availability, or stale memory. No arrangement is requested.

## Initial world state
- pitcher_0: on counter_0 (kitchen_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- jug_0 is absent; no other acceptable water containers exist.
- Articulated furniture requiring Open/Close: none.

## Final expected world state
- pitcher_0: on table_0 (living_room_0), filled with water.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off.
- jug_0 remains absent.

## Initial robot memory
- jug_0: on table_0 (living_room_0), filled with water; false stale existence and placement record.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- No pitcher_0 record or verified jug-absence fact is supplied.

## Success criteria
- is_on_top(pitcher_0, table_0).
- is_filled(pitcher_0).
- is_powered_off(lamp_0).
- is_on_top(lamp_0, chest_of_drawers_0).
- Discover and use the suitable substitute; the nonexistent remembered jug cannot satisfy delivery.

## Spawn / planner notes
- Spawn pitcher ×1 on counter_0, prefilled with water, and lamp ×1 on chest_of_drawers_0, powered on. Do not spawn jug.
- Map the spreadsheet's stale dining table to the dining-style table_0 in living_room_0; no dining room exists in the catalog.
- Only target/substitute memory differs from truth. PowerOff uses lamp_0, not ceiling fixtures. Robot starts in living_room_0.
