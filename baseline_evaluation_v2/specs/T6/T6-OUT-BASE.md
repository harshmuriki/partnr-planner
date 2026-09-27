# T6-OUT-BASE

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
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, clean). All three are uncertainty targets with stale memory locations.
- lamp: lamp_0, remaining task object with accurate memory. No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated

## Information supplied in instruction
- Specifies bread, one water bottle, one clean towel, the master-bedroom desk destination, adjacency, and living-room lights off.
- Omits source locations and towel color. Memory supplies incorrect exact sources, not candidate-room hints.
- Bind master bedroom to bedroom_2 and desk to table_11. Heating is requested but unsupported.

## Initial world state
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- hand_towel_0: on shelves_11 (bathroom_2), green, clean.
- lamp_0: on table_2 (living_room_0), powered on.

## Final expected world state
- bread_0: on table_11 (bedroom_2); no heating requirement.
- bottle_0: on table_11 (bedroom_2), filled with water, next_to bread_0.
- hand_towel_0: on table_11 (bedroom_2), green, clean, next_to bread_0.
- lamp_0: on table_2 (living_room_0), powered off.

## Initial robot memory
- bread_0: on table_7 (dining_room_0), ordinary bread; stale dining-table location; no heated state modeled.
- bottle_0: on table_5 (bedroom_2), filled with water; stale bedroom location.
- hand_towel_0: on chair_9 (bedroom_2), green, clean; stale bedroom-chair location.
- lamp_0: on table_2 (living_room_0), powered on; accurate.

## Success criteria
- is_on_top(bread_0, table_11).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_0, table_11), is_clean(hand_towel_0), is_next_to(hand_towel_0, bread_0).
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).

## Spawn / planner notes
- Spawn bread ×1 on counter_0, filled bottle ×1 on table_7, clean green hand_towel ×1 on shelves_11, and powered-on lamp ×1 on table_2. Preserve BASE physical starts.
- Bind the spreadsheet's stale bottle-in-bedroom location to table_5 and stale towel-on-bedroom-chair to chair_9. These are memory-only placements; never spawn duplicate targets there.
- Validate and revise stale target locations through observation. Lamp memory remains accurate.
- PowerOff uses lamp_0. Heating is unsupported; deliver bread without a heated-state check. No stacking.
- Robot starts in living_room_0.
