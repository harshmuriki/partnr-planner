# T7-ACC-ROOM

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
"Put all five toys from living_room_0 into wardrobe_0 in bedroom_3: the fire truck, stuffed toy, doll on couch_0, and airplane and animal on table_3. The fire truck and stuffed toy are somewhere in living_room_0. Wash both dirty plates and the dirty bowl in kitchen_0, then store them in cabinet_8. One dirty plate is somewhere in kitchen_0; the other plate and bowl are at the sink on cabinet_5. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Room-localized uncertainty targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Room Known

## Information supplied in instruction
- Specifies all object identities/counts, dirty states, storage destinations, cleaning, and both lights.
- Gives only living_room_0 for the two target toys and kitchen_0 for plate_0. Target surfaces, floor relation, and exact poses are omitted.
- Remaining task-object furniture is supplied; its mention does not establish target co-location.

## Initial world state
- toy_fire_truck_0: floor floor_living_room_0 (living_room_0).
- stuffed_toy_0: on couch_0 (living_room_0).
- plate_0: on counter_0 (kitchen_0), dirty.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Final expected world state
- toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0: within wardrobe_0 (bedroom_3).
- plate_0, plate_1, bowl_0: within cabinet_8 (kitchen_0), clean.
- lamp_0: on table_3 (living_room_0), powered off.
- lamp_1: on counter_0 (kitchen_0), powered off.

## Initial robot memory
- toy_fire_truck_0: in_room living_room_0; no support relation, furniture id, or pose recorded.
- stuffed_toy_0: in_room living_room_0; no support relation, furniture id, or pose recorded.
- plate_0: in_room kitchen_0, dirty; no support relation, furniture id, or pose recorded.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Success criteria
- Initial search constraints: in_room(toy_fire_truck_0, living_room_0), in_room(stuffed_toy_0, living_room_0), in_room(plate_0, kitchen_0); discover exact supports through observation.
- Final is_inside(t, wardrobe_0) for t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_0, plate_1, bowl_0}.
- is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).

## Spawn / planner notes
- Spawn all ten objects at exact BASE locations. The evaluator's exact world state must not leak into the target memory or prompt.
- Search the known target rooms, then update memory from observations. Start in living_room_0.
- Clean uses cabinet_5's faucet. Open wardrobe_0 and cabinet_8 before placement; PowerOff uses the two lamps.
