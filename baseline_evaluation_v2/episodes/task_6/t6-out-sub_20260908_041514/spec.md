# T6-OUT-SUB

## Base task
Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights.

## Scene
- scene_id: 104348010_171512832
- rooms used: kitchen_0, dining_room_0, bathroom_2, bedroom_2, living_room_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — YOO OH Kitchen Island
  - table_7 — dining_room_0 — table
  - shelves_11 — bathroom_2 — Parson 5-Shelf Bookcase
  - table_11 — bedroom_2 — Vanity Desk - Antique White
  - table_5 — bedroom_2 — Clyde side table
  - chair_9 — bedroom_2 — Etienne Upholstered Chair
  - table_2 — living_room_0 — Dark Oak

## Task instruction / prompt given
"Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights."

## Affected object(s)
- bread: bread_0; bottle: bottle_0; hand_towel: requested hand_towel_0 (green, absent). All three are selected uncertainty targets represented at stale locations.
- hand_towel: hand_towel_1 (blue, clean), suitable substitute present in the world but not memory.
- lamp: lamp_0, remaining task object with accurate memory. No distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies bread, one water bottle, one clean towel, adjacency at the master-bedroom desk, and living-room lights off.
- Omits source locations, towel color, requested towel absence, and substitute availability. A clean blue towel satisfies the goal.
- Bind master bedroom to bedroom_2 and desk to table_11. Heating is requested but unsupported.

## Initial world state
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- hand_towel_1: on shelves_11 (bathroom_2), blue, clean.
- lamp_0: on table_2 (living_room_0), powered on.
- hand_towel_0 is absent.

## Final expected world state
- bread_0: on table_11 (bedroom_2); no heating requirement.
- bottle_0: on table_11 (bedroom_2), filled with water, next_to bread_0.
- hand_towel_1: on table_11 (bedroom_2), blue, clean, next_to bread_0.
- lamp_0: on table_2 (living_room_0), powered off.
- hand_towel_0 remains absent.

## Initial robot memory
- bread_0: on table_7 (dining_room_0), ordinary bread; stale location; no heated state modeled.
- bottle_0: on table_5 (bedroom_2), filled with water; stale location.
- hand_towel_0: on chair_9 (bedroom_2), green, clean; falsely remembered as existing at its stale location.
- lamp_0: on table_2 (living_room_0), powered on; accurate.
- No record of hand_towel_1 or its source location. Memory has the missing requested towel, not the substitute.

## Success criteria
- is_on_top(bread_0, table_11).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_1, table_11), is_clean(hand_towel_1), is_next_to(hand_towel_1, bread_0); substitute counts as success.
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).
- Do not invent or claim delivery of hand_towel_0.

## Spawn / planner notes
- Spawn bread ×1 on counter_0, filled bottle ×1 on table_7, clean blue hand_towel ×1 on shelves_11, and powered-on lamp ×1 on table_2. Do not spawn the green towel.
- Preserve BASE bread, bottle, and lamp starts. Stale bedroom locations are table_5 for bottle and chair_9 for requested towel; these are memory-only.
- Revise false requested-towel memory and discover the substitute. Color is an instance attribute of hand_towel.
- PowerOff uses lamp_0. Heating is unsupported; bread delivery needs no heated-state check. No stacking.
- Robot starts in living_room_0.
