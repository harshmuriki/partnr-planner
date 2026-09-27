# T7-ACC-DIS

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
- floor_living_room_0 denotes the actual living-room floor.

## Task instruction / prompt given
"Put all five living-room toys into wardrobe_0 in bedroom_3: the fire truck on the floor, the stuffed toy and doll on couch_0, and the airplane and animal on table_3. Wash the dirty plate on counter_0 and the other dirty plate and bowl at the sink on cabinet_5, then store them in cabinet_8. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Uncertainty targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- Distractors: statue → statue_0; cushion → cushion_0; plate → plate_2, initially clean. No substitutes.

## Uncertainty being tested
- Internal robot memory: Accurate
- Distractors: Present

## Information supplied in instruction
- Names and counts distinguish the five toys from decoration and cushions; dirty state distinguishes the two requested plates from the clean plate.
- Gives exact task-object sources, storage destinations, cleaning, and lights. Distractors and door states are omitted.

## Initial world state
- toy_fire_truck_0: floor floor_living_room_0 (living_room_0), beside table_3.
- stuffed_toy_0: on couch_0 (living_room_0).
- plate_0: on counter_0 (kitchen_0), dirty.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- statue_0: on table_3 (living_room_0), at the edge near toy_fire_truck_0.
- cushion_0: on couch_0 (living_room_0), adjacent to stuffed_toy_0.
- plate_2: on counter_0 (kitchen_0), clean, adjacent to plate_0.
- wardrobe_0: Close; cabinet_8: Close.

## Final expected world state
- toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0: within wardrobe_0 (bedroom_3).
- plate_0, plate_1, bowl_0: within cabinet_8 (kitchen_0), clean.
- lamp_0: on table_3 (living_room_0), powered off.
- lamp_1: on counter_0 (kitchen_0), powered off.
- statue_0: on table_3 (living_room_0); cushion_0: on couch_0 (living_room_0); plate_2: on counter_0 (kitchen_0), clean.

## Initial robot memory
- toy_fire_truck_0: floor floor_living_room_0 (living_room_0), beside table_3.
- stuffed_toy_0: on couch_0 (living_room_0).
- plate_0: on counter_0 (kitchen_0), dirty.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- statue_0: on table_3 (living_room_0), at the edge near toy_fire_truck_0.
- cushion_0: on couch_0 (living_room_0), adjacent to stuffed_toy_0.
- plate_2: on counter_0 (kitchen_0), clean, adjacent to plate_0.
- wardrobe_0: Close; cabinet_8: Close.

## Success criteria
- is_inside(t, wardrobe_0) for t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_0, plate_1, bowl_0}.
- is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).
- is_on_top(statue_0, table_3); is_on_top(cushion_0, couch_0); is_on_top(plate_2, counter_0) AND is_clean(plate_2).

## Spawn / planner notes
- Spawn thirteen instances at the stated relations. Keep BASE task-object poses; choose the BASE truck pose beside table_3 so the added tabletop statue is nearby without spawning it on an unauthorized floor.
- Keep distractors visually separate and pickupable; do not treat the statue as a toy, cushion as a stuffed toy, or clean plate as the dirty target.
- Clean at cabinet_5's faucet. Open storage before placement. PowerOff uses spawned lamps; start in living_room_0.
