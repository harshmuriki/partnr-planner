# T6-ACC-SUB

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
  - table_2 — living_room_0 — Dark Oak

## Task instruction / prompt given
"Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights."

## Affected object(s)
- bread: bread_0; bottle: bottle_0; hand_towel: requested hand_towel_0 (green, absent). These are the selected uncertainty targets.
- hand_towel: hand_towel_1 (blue, clean), suitable substitute for hand_towel_0 and a target of memory evaluation.
- lamp: lamp_0, remaining task object. No distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: Substitute Available

## Information supplied in instruction
- Specifies bread, one water bottle, one clean towel, the master-bedroom desk, adjacency to bread, and living-room lights off.
- Requests unsupported heating. Omits towel color, source locations, furniture ids, and substitute identity.
- Bind master bedroom to bedroom_2 and desk to table_11. A clean blue towel satisfies the towel goal.

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
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- hand_towel_1: on shelves_11 (bathroom_2), blue, clean; suitable substitute.
- lamp_0: on table_2 (living_room_0), powered on.
- hand_towel_0 is known absent.

## Success criteria
- is_on_top(bread_0, table_11).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_1, table_11), is_clean(hand_towel_1), is_next_to(hand_towel_1, bread_0); substitute delivery counts as success.
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).

## Spawn / planner notes
- Spawn bread ×1 on counter_0, bottle ×1 on table_7, blue hand_towel ×1 on shelves_11, and lamp ×1 on table_2. Do not spawn the green towel.
- Initialize water and cleanliness; color is an instance attribute, not a separate spawn class.
- PowerOff operates on lamp_0. Heating is unsupported and replaced by delivery, with no heated-state check. Do not stack.
- Robot starts in living_room_0. No containment or distractors.
