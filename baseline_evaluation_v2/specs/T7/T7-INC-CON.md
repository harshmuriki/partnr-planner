# T7-INC-CON

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
  - cabinet_2 — kitchen_0 — Kitchen cabinet with door, narrow
  - wardrobe_0 — bedroom_3 — Designer Double Wardrobe White & White Gloss
  - wardrobe_1 — bedroom_3 — Designer Double Wardrobe White & White Gloss

## Task instruction / prompt given
"Put all five toys belonging to the living room into wardrobe_0 in bedroom_3. Include the fire truck and stuffed toy temporarily inside wardrobe_1 in bedroom_3, the doll on couch_0, and the airplane and animal on table_3. Wash the dirty plate inside cabinet_2 and the other dirty plate and bowl at the sink on cabinet_5 in kitchen_0, then store them in cabinet_8. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Omitted containment targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Gives all toys/dishes, target containment sources, storage destinations, dirty states, cleaning, and both lights.
- Includes the relocated living-room toys explicitly. Door positions are omitted; source descriptions in the prompt may guide discovery despite missing prior memory.

## Initial world state
- toy_fire_truck_0: within wardrobe_1 (bedroom_3).
- stuffed_toy_0: within wardrobe_1 (bedroom_3).
- plate_0: within cabinet_2 (kitchen_0), dirty.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_1: Close; cabinet_2: Close; wardrobe_0: Close; cabinet_8: Close.

## Final expected world state
- toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0: within wardrobe_0 (bedroom_3).
- plate_0, plate_1, bowl_0: within cabinet_8 (kitchen_0), clean.
- lamp_0: on table_3 (living_room_0), powered off.
- lamp_1: on counter_0 (kitchen_0), powered off.
- Source receptacles no longer contain the targets; doors may be closed afterward.

## Initial robot memory
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_1: Close; cabinet_2: Close; wardrobe_0: Close; cabinet_8: Close.
- Target records and all target-to-receptacle links are omitted. Known closed furniture is not known empty furniture.

## Success criteria
- Discover targets after opening their sources; do not declare them absent based only on missing memory or closed doors.
- is_inside(t, wardrobe_0) for t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_0, plate_1, bowl_0}.
- is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).

## Spawn / planner notes
- No explicitly containable living-room cabinet is cataloged. Normalize the row's toy-source cabinet to wardrobe_1 in bedroom_3; this disclosed room deviation is necessary to avoid inventing containment furniture. The prompt explicitly preserves toy scope.
- Spawn targets within closed wardrobe_1/cabinet_2; keep all seven non-target objects at BASE positions.
- Memory is the pre-instruction snapshot. Omit target containment, not the existence or door states of the furniture.
- Open sources before retrieval and destinations before placement. Clean at cabinet_5's faucet; PowerOff spawned lamps. Start in living_room_0.
