# T7-ACC-CON

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
- Containment targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies all toys and dishes, exact containment sources, destinations, cleaning, and two light locations.
- Explicitly includes the two temporarily relocated living-room toys. Closed-door states and action order are omitted from the instruction.

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
- Source receptacles no longer contain their target objects. Doors may be closed after retrieval/storage.

## Initial robot memory
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

## Success criteria
- is_inside(t, wardrobe_0) for each t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- is_inside(d, cabinet_8) AND is_clean(d) for each d in {plate_0, plate_1, bowl_0}.
- is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0).
- is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).

## Spawn / planner notes
- Catalog normalization: no explicitly containable living-room cabinet is listed. Use wardrobe_1 as the legal closed toy-source receptacle instead; this is a disclosed room deviation from the row, not a fabricated living-room cabinet. The instruction preserves target scope.
- Spawn the three targets within their closed sources; all seven remaining objects retain BASE locations.
- Open wardrobe_1 and cabinet_2 before picking. Open wardrobe_0 and cabinet_8 before storing. Clean at cabinet_5's faucet; PowerOff the spawned lamps. Start in living_room_0.
