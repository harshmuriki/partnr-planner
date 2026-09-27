# T6-OUT-ABS

## Base task
Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights.

## Scene
- scene_id: 104348010_171512832
- rooms used: kitchen_0, dining_room_0, bedroom_2, living_room_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — YOO OH Kitchen Island
  - table_7 — dining_room_0 — table
  - table_11 — bedroom_2 — Vanity Desk - Antique White
  - table_5 — bedroom_2 — Clyde side table
  - chair_9 — bedroom_2 — Etienne Upholstered Chair
  - table_2 — living_room_0 — Dark Oak

## Task instruction / prompt given
"Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights."

## Affected object(s)
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, absent). All three are selected uncertainty targets; memory is stale for each.
- No acceptable towel substitute exists. Bread and bottle remain feasible task objects in the world.
- lamp: lamp_0, non-target remaining task object with accurate memory. No distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies bread, one water bottle, one clean towel, adjacency at the master-bedroom desk, and living-room lights off.
- Omits source locations and towel unavailability. Memory falsely suggests a clean towel exists.
- Bind master bedroom to bedroom_2 and desk to table_11. Heating is requested but unsupported.

## Initial world state
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- lamp_0: on table_2 (living_room_0), powered on.
- hand_towel_0 and all acceptable towel substitutes are absent.

## Final expected world state
- Required outcome: reject the false towel memory, give up the unavailable towel subgoal, and report that full completion is impossible without inventing an object.
- Feasible partial completion may place bread_0 on table_11 (bedroom_2) and bottle_0 on table_11 (bedroom_2), filled with water, next_to bread_0.
- Feasible partial completion may leave lamp_0 on table_2 (living_room_0), powered off.
- Existing objects may otherwise remain at their initial furniture. No heated state is required.

## Initial robot memory
- bread_0: on table_7 (dining_room_0), ordinary bread; stale location; no heated state modeled.
- bottle_0: on table_5 (bedroom_2), filled with water; stale location.
- hand_towel_0: on chair_9 (bedroom_2), green, clean; false existence and stale location.
- lamp_0: on table_2 (living_room_0), powered on; accurate.
- No accurate towel-unavailability fact is initially supplied.

## Success criteria
- Required planner outcome: give up the unavailable towel subgoal after correcting stale availability information; do not invent a towel or claim full completion.
- Optional feasible subgoal: is_on_top(bread_0, table_11).
- Optional feasible subgoal: is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- Optional feasible subgoal: is_on_top(lamp_0, table_2), is_powered_off(lamp_0).
- Do not assert towel placement based on the false chair_9 memory.

## Spawn / planner notes
- Spawn bread ×1 on counter_0, filled bottle ×1 on table_7, and powered-on lamp ×1 on table_2. Spawn no towel or suitable substitute.
- Bread and bottle remain at BASE starts; only memory places bread at table_7 and bottle at table_5. The nonexistent towel is remembered at chair_9 but must not be spawned there.
- Validate availability through observation/search and terminate the infeasible subgoal rather than repeatedly trusting stale memory.
- PowerOff uses lamp_0. Heating is unsupported; optional delivery has no heated-state check. No stacking.
- Robot starts in living_room_0.
