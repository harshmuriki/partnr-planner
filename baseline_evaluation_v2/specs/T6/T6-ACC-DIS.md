# T6-ACC-DIS

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
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, clean). These are uncertainty targets.
- Distractors: toy_food: toy_food_0; spray_bottle: spray_bottle_0; hand_towel: hand_towel_1 (dirty).
- lamp: lamp_0, remaining task object. No substitutes.

## Uncertainty being tested
- Internal robot memory: Accurate
- Distractors: Present

## Information supplied in instruction
- Specifies bread, one water bottle, one clean towel, destination, adjacency to bread, and living-room lights off.
- Omits source furniture, towel color, and distractor identities. Cleanliness and object type distinguish the intended objects.
- Bind master bedroom to bedroom_2 and desk to table_11. Heating is requested but unsupported.

## Initial world state
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- hand_towel_0: on shelves_11 (bathroom_2), green, clean.
- lamp_0: on table_2 (living_room_0), powered on.
- toy_food_0: on counter_0 (kitchen_0), next_to bread_0, imitation food.
- spray_bottle_0: on table_7 (dining_room_0), next_to bottle_0, not a drinking-water bottle.
- hand_towel_1: on shelves_11 (bathroom_2), next_to hand_towel_0, dirty.

## Final expected world state
- bread_0: on table_11 (bedroom_2); no heating requirement.
- bottle_0: on table_11 (bedroom_2), filled with water, next_to bread_0.
- hand_towel_0: on table_11 (bedroom_2), green, clean, next_to bread_0.
- lamp_0: on table_2 (living_room_0), powered off.
- toy_food_0: on counter_0 (kitchen_0), imitation food.
- spray_bottle_0: on table_7 (dining_room_0), not a drinking-water bottle.
- hand_towel_1: on shelves_11 (bathroom_2), dirty.

## Initial robot memory
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- hand_towel_0: on shelves_11 (bathroom_2), green, clean.
- lamp_0: on table_2 (living_room_0), powered on.
- toy_food_0: on counter_0 (kitchen_0), next_to bread_0, imitation food.
- spray_bottle_0: on table_7 (dining_room_0), next_to bottle_0, not a drinking-water bottle.
- hand_towel_1: on shelves_11 (bathroom_2), next_to hand_towel_0, dirty.

## Success criteria
- is_on_top(bread_0, table_11).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_0, table_11), is_clean(hand_towel_0), is_next_to(hand_towel_0, bread_0).
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).
- is_on_top(toy_food_0, counter_0), is_on_top(spray_bottle_0, table_7), is_on_top(hand_towel_1, shelves_11); distractors are not delivered.

## Spawn / planner notes
- Spawn bread and toy_food on counter_0; bottle and spray_bottle on table_7; two hand_towel instances on shelves_11; lamp on table_2. Place each distractor next to its corresponding target.
- Initialize bottle_0 filled, hand_towel_0 clean and green, hand_towel_1 dirty, and lamp_0 on. Keep task objects at BASE starts.
- Use lamp_0 for PowerOff. Heating is unsupported; delivery replaces it without a heated-state check. Do not stack or clean a distractor in place of selecting the clean target.
- Robot starts in living_room_0.
