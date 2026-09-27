# T7-ACC-CAND

## Base task
Put away all toys from the living room into the wardrobe, wash and put away the dirty dishes from sink, and turn off the living-room and kitchen lights.

## Scene
- scene_id: 102816756
- rooms used: living_room_0, kitchen_0, bedroom_3
- furniture used (id — room — catalog description):
  - couch_0 — living_room_0 — Delano 3 Piece Sectional With Left Arm Facing Chaise, Pearl
  - table_3 — living_room_0 — Plinth Coffee Table, Carrara
  - table_0 — living_room_0 — Palerma Extendable Dining Table, White
  - counter_0 — kitchen_0 — Kitchen island, 90x150x90
  - cabinet_5 — kitchen_0 — Kitchen cabinet with sink, double
  - cabinet_8 — kitchen_0 — Kitchen cabinet with 2 doors
  - wardrobe_0 — bedroom_3 — Designer Double Wardrobe White & White Gloss
- floor_living_room_0 denotes the actual living-room floor.

## Task instruction / prompt given
"Put the five toys belonging to the living room into wardrobe_0 in bedroom_3. Search living_room_0 or bedroom_3 for the fire truck and stuffed toy; the doll is on couch_0, and the airplane and animal are on table_3. Wash both dirty plates and the dirty bowl, then store them in cabinet_8 in kitchen_0. Search kitchen_0 or the dining area of living_room_0 for one dirty plate; the other dirty plate and bowl are at the sink on cabinet_5. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Candidate-room uncertainty targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Candidate Rooms

## Information supplied in instruction
- Gives full toy/dish scope, counts, dirty states, storage destinations, cleaning, and lights.
- Supplies only candidate rooms for targets: living_room_0 / bedroom_3 for each target toy; kitchen_0 / living_room_0 for plate_0.
- Exact target room, furniture, support relation, and pose are omitted. Remaining task objects retain exact locations.

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
- toy_fire_truck_0: candidate rooms {living_room_0, bedroom_3}; no exact room, furniture, support, or pose.
- stuffed_toy_0: candidate rooms {living_room_0, bedroom_3}; no exact room, furniture, support, or pose.
- plate_0: candidate rooms {kitchen_0, living_room_0}, dirty; no exact room, furniture, support, or pose.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Success criteria
- Observe targets within their candidate sets; do not assert one candidate as known before observation.
- is_inside(t, wardrobe_0) for t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_0, plate_1, bowl_0}.
- is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).

## Spawn / planner notes
- Spawn ten instances at unchanged BASE positions; keep exact target placement evaluator-only until observation.
- Map the row's bedroom candidate to bedroom_3. There is no dining_room in the catalog: use living_room_0, whose table_0 defines the dining area, without assigning plate_0 to that furniture in memory.
- Clean at cabinet_5's faucet; open storage before placement; PowerOff the spawned lamps. Start in living_room_0.
