# T7-INC-ABS

## Base task
Put away all toys from the living room into the wardrobe, wash and put away the dirty dishes from sink, and turn off the living-room and kitchen lights.

## Scene
- scene_id: 102816756
- rooms used: living_room_0, kitchen_0, bedroom_3
- furniture used (id — room — catalog description):
  - couch_0 — living_room_0 — Delano 3 Piece Sectional With Left Arm Facing Chaise, Pearl
  - table_3 — living_room_0 — Plinth Coffee Table, Carrara
  - counter_0 — kitchen_0 — Kitchen island, 90x150x90
  - cabinet_5 — kitchen_0 — Kitchen cabinet with sink, double
  - cabinet_8 — kitchen_0 — Kitchen cabinet with 2 doors
  - wardrobe_0 — bedroom_3 — Designer Double Wardrobe White & White Gloss
- floor_living_room_0 denotes the living-room floor referenced in the request.

## Task instruction / prompt given
"Put all five living-room toys into wardrobe_0 in bedroom_3: the fire truck on the floor, the stuffed toy and doll on couch_0, and the airplane and animal on table_3. Wash the dirty plate on counter_0 and the other dirty plate and bowl at the sink on cabinet_5, then store them in cabinet_8. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Absent uncertainty targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies five toys, two plates, a bowl, exact source references, dirty states, cleaning/storage, and two lights.
- Absence is not disclosed. Memory lacks the selected objects rather than explicitly asserting that they do not exist.

## Initial world state
- toy_fire_truck_0, stuffed_toy_0, plate_0: absent; no suitable substitutes exist.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Final expected world state
- Missing targets remain absent; their subgoals are abandoned and reported infeasible after bounded verification.
- Optional feasible remainder: toy_airplane_0, doll_0, toy_animal_0 within wardrobe_0 (bedroom_3); plate_1 and bowl_0 within cabinet_8 (kitchen_0), clean.
- lamp_0 remains on table_3 (living_room_0); lamp_1 remains on counter_0 (kitchen_0). Each may be powered off as a feasible subgoal.
- If remaining subgoals are not attempted, their objects retain the initial world state.

## Initial robot memory
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.
- No records or confirmed absence information for the three uncertainty targets or any substitutes.

## Success criteria
- After reasonable availability checks, give up on unavailable subgoals. Do not invent targets, substitute unrelated items, or report full completion.
- Optional feasible propositions: is_inside(t, wardrobe_0) for t in {toy_airplane_0, doll_0, toy_animal_0}.
- Optional feasible propositions: is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_1, bowl_0}.
- Optional feasible propositions: is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).

## Spawn / planner notes
- The task row marks availability-none N/A. This is the requested matrix's synthetic absence ablation, removing only the uncertainty targets, not a row-provided absence scenario.
- Spawn seven existing instances at BASE surfaces and no replacements. Do not equate an absent memory record with verified physical absence.
- Clean at cabinet_5's faucet. Open storage before placement; use spawned lamps for PowerOff. Start in living_room_0.
