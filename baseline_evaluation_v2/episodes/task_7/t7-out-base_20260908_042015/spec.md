# T7-OUT-BASE

## Base task
Put away all toys from the living room into the wardrobe, wash and put away the dirty dishes from sink, and turn off the living-room and kitchen lights.

## Scene
- scene_id: 102816756
- rooms used: living_room_0, kitchen_0, bedroom_3
- furniture used (id — room — catalog description):
  - couch_0 — living_room_0 — Delano 3 Piece Sectional With Left Arm Facing Chaise, Pearl
  - table_3 — living_room_0 — Plinth Coffee Table, Carrara
  - table_0 — living_room_0 — Palerma Extendable Dining Table, White
  - table_8 — bedroom_3 — Dino Dresser, White
  - counter_0 — kitchen_0 — Kitchen island, 90x150x90
  - cabinet_5 — kitchen_0 — Kitchen cabinet with sink, double
  - cabinet_8 — kitchen_0 — Kitchen cabinet with 2 doors
  - wardrobe_0 — bedroom_3 — Designer Double Wardrobe White & White Gloss
- floor_living_room_0 denotes the actual living-room floor.

## Task instruction / prompt given
"Put all five living-room toys into wardrobe_0 in bedroom_3: the fire truck on the floor, the stuffed toy and doll on couch_0, and the airplane and animal on table_3. Wash the dirty plate on counter_0 and the other dirty plate and bowl at the sink on cabinet_5, then store them in cabinet_8. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Stale-location targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated

## Information supplied in instruction
- Specifies five toys, two dirty plates, a dirty bowl, true exact sources, cleaning/storage, and both lights.
- Door positions and action order are omitted. Current instruction locations conflict with stale target memory; they are not changed to match it.

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
- toy_fire_truck_0: on table_8 (bedroom_3).
- stuffed_toy_0: on table_8 (bedroom_3).
- plate_0: on table_0 (living_room_0), dirty.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Success criteria
- Reconcile stale target locations with current observations; do not invent duplicate objects at remembered locations.
- is_inside(t, wardrobe_0) for t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_0, plate_1, bowl_0}.
- is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).

## Spawn / planner notes
- Spawn all ten objects only at true BASE locations. table_8 and table_0 are stale-memory supports, not extra spawn locations.
- Ground the row's remembered bedroom toys to table_8 in bedroom_3 and dining-table plate to table_0 in living_room_0; no separate dining room is cataloged.
- Change only the three target memory locations. Clean at cabinet_5's faucet; open storage before placement; PowerOff spawned lamps. Start in living_room_0.
