# T6-ACC-ABS

## Base task
Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights.

## Scene
- scene_id: 104348010_171512832
- rooms used: kitchen_0, dining_room_0, bedroom_2, living_room_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — YOO OH Kitchen Island
  - table_7 — dining_room_0 — table
  - table_11 — bedroom_2 — Vanity Desk - Antique White
  - table_2 — living_room_0 — Dark Oak

## Task instruction / prompt given
"Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights."

## Affected object(s)
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, absent). All three are selected uncertainty targets; only towel availability changes.
- No suitable towel substitute exists.
- lamp: lamp_0, remaining task object. No distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies bread, one water bottle, one clean towel, destination at the master-bedroom desk, adjacency, and living-room lights off.
- Does not disclose towel absence or source locations; accurate memory supplies availability and exact sources.
- Bind master bedroom to bedroom_2 and desk to table_11. Heating is requested but unsupported.

## Initial world state
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- lamp_0: on table_2 (living_room_0), powered on.
- hand_towel_0 and all acceptable towel substitutes are absent.

## Final expected world state
- Required outcome: give up the unavailable towel subgoal and report that full completion is impossible; no towel is invented.
- Feasible partial completion may place bread_0 on table_11 (bedroom_2) and bottle_0 on table_11 (bedroom_2), filled with water, next_to bread_0.
- Feasible partial completion may leave lamp_0 on table_2 (living_room_0), powered off.
- Without partial execution, existing objects may remain at their initial furniture. No heated state is required.

## Initial robot memory
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- lamp_0: on table_2 (living_room_0), powered on.
- hand_towel_0 and all acceptable towel substitutes are known absent.

## Success criteria
- Required planner outcome: give up the unavailable towel subgoal without inventing an object or claiming full task success.
- Optional feasible subgoal: is_on_top(bread_0, table_11).
- Optional feasible subgoal: is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- Optional feasible subgoal: is_on_top(lamp_0, table_2), is_powered_off(lamp_0).
- No towel placement proposition is required or may be falsely asserted.

## Spawn / planner notes
- Spawn bread ×1 on counter_0, filled bottle ×1 on table_7, and powered-on lamp ×1 on table_2. Spawn no towel or acceptable substitute.
- Keep bread, bottle, and lamp at their BASE starts. PowerOff uses the lamp, not ceiling fixtures.
- Heating is unsupported; feasible bread delivery requires no heated state. Use adjacency, not stacking.
- Robot starts in living_room_0.
