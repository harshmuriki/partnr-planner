# Task success criteria

Taken from `## Success criteria` in `baseline_evaluation_v3/specs/`. Within a task, Accurate, Incomplete, and Outdated variants of the same axis use the same criteria. Only the axis changes the propositions: BASE and CON match; ABS, SUB, and DIS differ.

## T1

Turn off all the lights in the bedroom and bring a full jug of water to the living-room table.

BASE, CON (`T1-ACC-BASE`, `T1-ACC-CON`, `T1-INC-BASE`, `T1-INC-CON`, `T1-OUT-BASE`, `T1-OUT-CON`):

- `is_on_top(jug_0, table_0)`
- `is_filled(jug_0)`
- `is_powered_off(lamp_0)`

ABS (`T1-ACC-ABS`, `T1-INC-ABS`, `T1-OUT-ABS`):

- `is_powered_off(lamp_0)`
- The jug (`jug_0`) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

SUB (`T1-ACC-SUB`, `T1-INC-SUB`, `T1-OUT-SUB`):

- `is_on_top(pitcher_0, table_0)`
- `is_filled(pitcher_0)`
- `is_powered_off(lamp_0)`
- Using `pitcher_0` in place of the missing `jug_0` counts as success; do not require or invent a jug.

DIS (`T1-ACC-DIS`, `T1-INC-DIS`, `T1-OUT-DIS`):

- `is_on_top(jug_0, table_0)`
- `is_filled(jug_0)`
- `is_powered_off(lamp_0)`
- Distractors (`vase_0`, `spray_bottle_0`) must not be used in place of the task objects; they have no placement goals.

## T2

Move both packages to the living-room center table and place them next to each other. Bring the scissors and place beside them to open the packages.

BASE, CON (`T2-ACC-BASE`, `T2-ACC-CON`, `T2-INC-BASE`, `T2-INC-CON`, `T2-OUT-BASE`, `T2-OUT-CON`):

- `is_on_top(box_0, table_16)`
- `is_on_top(box_1, table_16)`
- `is_on_top(scissors_0, table_16)`
- `is_next_to(box_0, box_1)`
- `is_next_to(scissors_0, box_0)`
- `is_next_to(scissors_0, box_1)`

ABS (`T2-ACC-ABS`, `T2-INC-ABS`, `T2-OUT-ABS`):

- `is_on_top(box_0, table_16)`
- `is_on_top(box_1, table_16)`
- `is_next_to(box_0, box_1)`
- The scissors (`scissors_0`) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

SUB (`T2-ACC-SUB`, `T2-INC-SUB`, `T2-OUT-SUB`):

- `is_on_top(box_0, table_16)`
- `is_on_top(box_1, table_16)`
- `is_on_top(knife_0, table_16)`
- `is_next_to(box_0, box_1)`
- `is_next_to(knife_0, box_0)`
- `is_next_to(knife_0, box_1)`
- Using `knife_0` in place of the missing `scissors_0` counts as success; do not require or invent a scissors.

DIS (`T2-ACC-DIS`, `T2-INC-DIS`, `T2-OUT-DIS`):

- `is_on_top(box_0, table_16)`
- `is_on_top(box_1, table_16)`
- `is_on_top(scissors_0, table_16)`
- `is_next_to(box_0, box_1)`
- `is_next_to(scissors_0, box_0)`
- `is_next_to(scissors_0, box_1)`
- Distractors (`board_game_0`, `distractor_knife_0`, `screwdriver_0`) must not be used in place of the task objects; they have no placement goals.

## T3

Collect the 2 green books and place them on the living room table and turn off the laptop as well.

BASE, CON (`T3-ACC-BASE`, `T3-ACC-CON`, `T3-INC-BASE`, `T3-INC-CON`, `T3-OUT-BASE`, `T3-OUT-CON`):

- `is_on_top(book_0, table_0)`
- `is_on_top(book_1, table_0)`
- `is_powered_off(laptop_0)`

ABS (`T3-ACC-ABS`, `T3-INC-ABS`, `T3-OUT-ABS`):

- `is_on_top(book_0, table_0)`
- `is_powered_off(laptop_0)`
- The book (`book_1`) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

DIS (`T3-ACC-DIS`, `T3-INC-DIS`, `T3-OUT-DIS`):

- `is_on_top(book_0, table_0)`
- `is_on_top(book_1, table_0)`
- `is_powered_off(laptop_0)`
- Distractors (`folder_0`, `phone_0`) must not be used in place of the task objects; they have no placement goals.

T3 has no SUB variants.

## T4

Put all the 3 fruits in the fruit basket and the 2 drink containers on the kitchen counter.

BASE, CON (`T4-ACC-BASE`, `T4-ACC-CON`, `T4-INC-BASE`, `T4-INC-CON`, `T4-OUT-BASE`, `T4-OUT-CON`):

- `is_inside(apple_0, basket_0)`
- `is_inside(apple_1, basket_0)`
- `is_inside(orange_0, basket_0)`
- `is_on_top(basket_0, counter_0)`
- `is_on_top(bottle_0, counter_0)`
- `is_on_top(bottle_1, counter_0)`

ABS (`T4-ACC-ABS`, `T4-INC-ABS`, `T4-OUT-ABS`):

- `is_inside(apple_1, basket_0)`
- `is_inside(orange_0, basket_0)`
- `is_on_top(basket_0, counter_0)`
- `is_on_top(bottle_0, counter_0)`
- `is_on_top(bottle_1, counter_0)`
- The apple (`apple_0`) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

DIS (`T4-ACC-DIS`, `T4-INC-DIS`, `T4-OUT-DIS`):

- `is_inside(apple_0, basket_0)`
- `is_inside(apple_1, basket_0)`
- `is_inside(orange_0, basket_0)`
- `is_on_top(basket_0, counter_0)`
- `is_on_top(bottle_0, counter_0)`
- `is_on_top(bottle_1, counter_0)`
- Distractors (`toy_fruit_0`, `spray_bottle_0`) must not be used in place of the task objects; they have no placement goals.

T4 has no SUB variants.

## T5

Get soap to clean both glasses, fill them, and place them on the living-room table for our guests to drink. Get a white plate and place it next to them.

BASE, CON (`T5-ACC-BASE`, `T5-ACC-CON`, `T5-INC-BASE`, `T5-INC-CON`, `T5-OUT-BASE`, `T5-OUT-CON`):

- `is_on_top(glass_0, table_0)`
- `is_on_top(glass_1, table_0)`
- `is_on_top(plate_0, table_0)`
- `is_next_to(plate_0, glass_0)`
- `is_next_to(plate_0, glass_1)`
- `is_next_to(soap_dispenser_0, glass_0)`
- `is_clean(glass_0)`
- order: `is_next_to(soap_dispenser_0, glass_0)` before `is_clean(glass_0)`
- `is_filled(glass_0)`
- `is_next_to(soap_dispenser_0, glass_1)`
- `is_clean(glass_1)`
- order: `is_next_to(soap_dispenser_0, glass_1)` before `is_clean(glass_1)`
- `is_filled(glass_1)`

ABS (`T5-ACC-ABS`, `T5-INC-ABS`, `T5-OUT-ABS`):

- `is_on_top(glass_0, table_0)`
- `is_on_top(glass_1, table_0)`
- `is_next_to(soap_dispenser_0, glass_0)`
- `is_clean(glass_0)`
- order: `is_next_to(soap_dispenser_0, glass_0)` before `is_clean(glass_0)`
- `is_filled(glass_0)`
- `is_next_to(soap_dispenser_0, glass_1)`
- `is_clean(glass_1)`
- order: `is_next_to(soap_dispenser_0, glass_1)` before `is_clean(glass_1)`
- `is_filled(glass_1)`
- The plate (`plate_0`) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

SUB (`T5-ACC-SUB`, `T5-INC-SUB`, `T5-OUT-SUB`):

- `is_on_top(glass_0, table_0)`
- `is_on_top(glass_1, table_0)`
- `is_on_top(plate_black_0, table_0)`
- `is_next_to(plate_black_0, glass_0)`
- `is_next_to(plate_black_0, glass_1)`
- `is_next_to(soap_dispenser_0, glass_0)`
- `is_clean(glass_0)`
- order: `is_next_to(soap_dispenser_0, glass_0)` before `is_clean(glass_0)`
- `is_filled(glass_0)`
- `is_next_to(soap_dispenser_0, glass_1)`
- `is_clean(glass_1)`
- order: `is_next_to(soap_dispenser_0, glass_1)` before `is_clean(glass_1)`
- `is_filled(glass_1)`
- Using `plate_black_0` in place of the missing `plate_0` counts as success; do not require or invent a plate.

DIS (`T5-ACC-DIS`, `T5-INC-DIS`, `T5-OUT-DIS`):

- `is_on_top(glass_0, table_0)`
- `is_on_top(glass_1, table_0)`
- `is_on_top(plate_0, table_0)`
- `is_next_to(plate_0, glass_0)`
- `is_next_to(plate_0, glass_1)`
- `is_next_to(soap_dispenser_0, glass_0)`
- `is_clean(glass_0)`
- order: `is_next_to(soap_dispenser_0, glass_0)` before `is_clean(glass_0)`
- `is_filled(glass_0)`
- `is_next_to(soap_dispenser_0, glass_1)`
- `is_clean(glass_1)`
- order: `is_next_to(soap_dispenser_0, glass_1)` before `is_clean(glass_1)`
- `is_filled(glass_1)`
- Distractors (`soap_dish_0`, `vase_0`, `plant_saucer_0`) must not be used in place of the task objects; they have no placement goals.

## T6

Heat up the bread, bring it to the bedroom desk, bring a water bottle and a clean hand towel and place them next to the bread. Turn off the living-room lights.

BASE, CON (`T6-ACC-BASE`, `T6-ACC-CON`, `T6-INC-BASE`, `T6-INC-CON`, `T6-OUT-BASE`, `T6-OUT-CON`):

- `is_inside(bread_0, microwave_0)`
- `is_on_top(bread_0, table_11)`
- order: `is_inside(bread_0, microwave_0)` before `is_on_top(bread_0, table_11)`
- `is_on_top(bottle_0, table_11)`
- `is_on_top(hand_towel_0, table_11)`
- `is_next_to(bottle_0, bread_0)`
- `is_next_to(hand_towel_0, bread_0)`
- `is_filled(bottle_0)`
- `is_clean(hand_towel_0)`
- `is_powered_off(lamp_living_0)`

ABS (`T6-ACC-ABS`, `T6-INC-ABS`, `T6-OUT-ABS`):

- `is_inside(bread_0, microwave_0)`
- `is_on_top(bread_0, table_11)`
- order: `is_inside(bread_0, microwave_0)` before `is_on_top(bread_0, table_11)`
- `is_on_top(bottle_0, table_11)`
- `is_next_to(bottle_0, bread_0)`
- `is_filled(bottle_0)`
- `is_powered_off(lamp_living_0)`
- The hand towel (`hand_towel_0`) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

SUB (`T6-ACC-SUB`, `T6-INC-SUB`, `T6-OUT-SUB`):

- `is_inside(bread_0, microwave_0)`
- `is_on_top(bread_0, table_11)`
- order: `is_inside(bread_0, microwave_0)` before `is_on_top(bread_0, table_11)`
- `is_on_top(cup_0, table_11)`
- `is_on_top(blue_hand_towel_0, table_11)`
- `is_next_to(cup_0, bread_0)`
- `is_next_to(blue_hand_towel_0, bread_0)`
- `is_filled(cup_0)`
- `is_clean(blue_hand_towel_0)`
- `is_powered_off(lamp_living_0)`
- Using `blue_hand_towel_0` in place of the missing `hand_towel_0` counts as success; do not require or invent a hand towel.
- Using `cup_0` in place of the missing `bottle_0` counts as success; do not require or invent a bottle.

DIS (`T6-ACC-DIS`, `T6-INC-DIS`, `T6-OUT-DIS`):

- `is_inside(bread_0, microwave_0)`
- `is_on_top(bread_0, table_11)`
- order: `is_inside(bread_0, microwave_0)` before `is_on_top(bread_0, table_11)`
- `is_on_top(bottle_0, table_11)`
- `is_on_top(hand_towel_0, table_11)`
- `is_next_to(bottle_0, bread_0)`
- `is_next_to(hand_towel_0, bread_0)`
- `is_filled(bottle_0)`
- `is_clean(hand_towel_0)`
- `is_powered_off(lamp_living_0)`
- Distractors (`dirty_hand_towel_0`, `bread_25_0`, `supplement_bottle_0`) must not be used in place of the task objects; they have no placement goals.

## T7

Put away all toys from the living room into the wardrobe, wash and put away the dirty dishes from the sink, and turn off the living-room and kitchen lights.

BASE (`T7-ACC-BASE`, `T7-INC-BASE`, `T7-OUT-BASE`):

- `is_inside(toy_truck_0, wardrobe_0)`
- `is_inside(stuffed_toy_0, wardrobe_0)`
- `is_inside(plate_0, cabinet_7)`
- `is_clean(plate_0)`
- `is_powered_off(lamp_living_0)`
- `is_powered_off(lamp_kitchen_0)`

DIS (`T7-ACC-DIS`, `T7-INC-DIS`, `T7-OUT-DIS`):

- `is_inside(toy_truck_0, wardrobe_0)`
- `is_inside(stuffed_toy_0, wardrobe_0)`
- `is_inside(plate_0, cabinet_7)`
- `is_clean(plate_0)`
- `is_powered_off(lamp_living_0)`
- `is_powered_off(lamp_kitchen_0)`
- Distractors (`clean_plate_0`, `cushion_0`) must not be used in place of the task objects; they have no placement goals.

T7 has no ABS, SUB, or CON variants.
