# T7-ACC-AMB

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
"Tidy the living room and clean up the dishes."

## Affected object(s)
- Uncertainty targets: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Instruction: Underspecified

## Information supplied in instruction
- Mentions living-room tidying and dishes only.
- Omits toy identities/count, dish count and dirty states, kitchen location, all source furniture, wardrobe and dish-storage destinations, and both light-off subgoals.
- Accurate environmental memory does not itself specify the user's intended storage or lighting policy.

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
- Canonical reference after clarification: toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0 within wardrobe_0 (bedroom_3).
- plate_0, plate_1, bowl_0: within cabinet_8 (kitchen_0), clean.
- lamp_0: on table_3 (living_room_0), powered off after lighting confirmation.
- lamp_1: on counter_0 (kitchen_0), powered off after lighting confirmation.
- If clarification is unavailable, do not claim the omitted storage and lighting goals were specified; preserve unconfirmed objects rather than fabricate intent.

## Initial robot memory
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

## Success criteria
- Detect underspecification and request confirmation of toy scope, storage destinations, and whether both lights should be turned off.
- After canonical-goal confirmation: is_inside(t, wardrobe_0) for t in {toy_fire_truck_0, stuffed_toy_0, toy_airplane_0, doll_0, toy_animal_0}.
- After confirmation: is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_0, plate_1, bowl_0}.
- After lighting confirmation: is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).
- Without confirmation, score ambiguity handling separately; do not require inference of unspoken lighting goals.

## Spawn / planner notes
- Spawn the ten BASE instances with no environmental changes. Start in living_room_0.
- Keep this exact underspecified prompt; do not append the canonical task or destinations to the initial instruction.
- If clarification is supported, confirm all five toys to wardrobe_0, all three dirty dishes cleaned and stored in cabinet_8, and both lamps off.
- Clean at cabinet_5's faucet; open storage before placement; PowerOff applies only to spawned lamps.
