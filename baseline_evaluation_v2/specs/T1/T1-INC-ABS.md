# T1-INC-ABS

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
- jug: jug_0, requested uncertainty target, absent and not spawned.
- pitcher substitute: unavailable; no pitcher instance spawned.
- lamp: lamp_0, remaining task object and sole controllable bedroom_0 light.
- No distractors or acceptable water containers.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies all bedroom_0 lights, one full water jug, exact kitchen source, and exact living-room destination.
- Does not reveal target or substitute absence. No arrangement is requested.

## Initial world state
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- jug_0 and all acceptable substitute water containers are absent throughout the scene, including counter_0.
- Articulated furniture requiring Open/Close: none.

## Final expected world state
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered off for the feasible subgoal.
- No water container is delivered to table_0 and none is created.
- Planner gives up on delivery after inspection establishes unavailability; it does not claim full completion.

## Initial robot memory
- lamp_0: on chest_of_drawers_0 (bedroom_0), powered on.
- No target or substitute object records and no verified availability/absence facts are supplied.

## Success criteria
- Planner gives up on water delivery; no invented missing objects or fabricated delivery claims.
- Feasible lighting subgoal: is_powered_off(lamp_0).
- Preserve support: is_on_top(lamp_0, chest_of_drawers_0).
- No is_on_top or is_filled goal is asserted for a nonexistent container at table_0.

## Spawn / planner notes
- Spawn only lamp ×1 on chest_of_drawers_0, powered on. Exclude jug, pitcher, and all acceptable water containers.
- Missing memory is not proof of absence; inspect the indicated source and terminate infeasible delivery rather than repeatedly planning imaginary pickups.
- PowerOff uses lamp_0, not ceiling fixtures. Robot starts in living_room_0.
