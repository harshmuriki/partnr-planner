# T7-INC-DIS

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
- Omitted uncertainty targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- Distractors retained in memory: statue → statue_0; cushion → cushion_0; plate → plate_2, clean. No substitutes.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Distractors: Present

## Information supplied in instruction
- Specifies five named toys and three dirty dishes, exact source references, destinations, cleaning, and lights.
- Omits distractors. Toy identities and dirty state distinguish task objects from nearby non-task items.

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
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- statue_0: on table_3 (living_room_0), at the edge.
- cushion_0: on couch_0 (living_room_0).
- plate_2: on counter_0 (kitchen_0), clean.
- wardrobe_0: Close; cabinet_8: Close.
- Target records and distractor-to-target adjacency links are omitted; distractor records and poses remain accurate.

## Success criteria
- Recover missing targets through observation; do not use a remembered distractor to satisfy a missing target record.
- is_inside(t, wardrobe_0) for t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_0, plate_1, bowl_0}.
- is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).
- is_on_top(statue_0, table_3); is_on_top(cushion_0, couch_0); is_on_top(plate_2, counter_0) AND is_clean(plate_2).

## Spawn / planner notes
- Spawn thirteen instances. Keep BASE task poses, including the truck beside table_3, and add the statue on the nearby table edge, cushion beside the stuffed toy, and clean plate beside the dirty plate.
- Omit only the three target records and links that would reveal them; retain all remaining task objects and distractors in memory.
- Clean at cabinet_5's faucet; open storage before placement; PowerOff spawned lamps. Start in living_room_0.
