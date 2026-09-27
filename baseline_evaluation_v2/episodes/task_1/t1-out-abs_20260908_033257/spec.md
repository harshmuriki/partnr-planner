# T1-OUT-ABS

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
- jug: jug_0, requested uncertainty target, absent physically but falsely remembered.
- pitcher substitute: unavailable; no pitcher instance spawned.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No distractors or acceptable water containers.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies all bedroom_0 lights, one full water jug, exact kitchen source, and exact living-room destination.
- Omits impossibility and memory staleness. No arrangement is requested.

## Initial world state
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- jug_0 and all acceptable substitute water containers are absent throughout the scene, including counter_0 and table_0.
- Articulated furniture requiring Open/Close: none.

## Final expected world state
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off for the feasible subgoal.
- No container is delivered to table_0 or created.
- Planner rejects the stale target record and gives up on infeasible water delivery without claiming full completion.

## Initial robot memory
- jug_0: on table_0 (living_room_0), filled with water; false stale existence and placement record.
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- No verified target-absence or substitute-unavailability fact is supplied.

## Success criteria
- Planner gives up on delivery after invalidating the stale existence claim; no invented object or fabricated delivery.
- Feasible lighting subgoal: is_powered_off(lamp_0).
- Preserve support: is_on_top(lamp_0, chest_of_drawers_0).
- Do not treat the memory-only is_on_top(jug_0, table_0) or is_filled(jug_0) as simulator success.

## Spawn / planner notes
- Spawn only lamp ×1 on chest_of_drawers_0, powered on. Exclude all acceptable water containers.
- Map the spreadsheet's dining table to dining-style table_0 in living_room_0, since the catalog has no dining room. The stale jug is memory-only.
- PowerOff uses lamp_0, not ceiling fixtures. Robot starts in living_room_0.
