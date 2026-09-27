# T7-OUT-ABS

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
- floor_living_room_0 denotes the living-room floor referenced in the request.

## Task instruction / prompt given
"Put all five living-room toys into wardrobe_0 in bedroom_3: the fire truck on the floor, the stuffed toy and doll on couch_0, and the airplane and animal on table_3. Wash the dirty plate on counter_0 and the other dirty plate and bowl at the sink on cabinet_5, then store them in cabinet_8. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Absent uncertainty targets with false memory records: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: No Suitable Object

## Information supplied in instruction
- Specifies five toys, two dirty plates, a dirty bowl, source references, destinations, cleaning, and lights.
- Actual absence is omitted. Both requested source descriptions and stale memory must be checked against observed availability.

## Initial world state
- toy_fire_truck_0, stuffed_toy_0, plate_0: absent; substitutes unavailable.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Final expected world state
- The three targets remain absent. Their subgoals are abandoned after verification and reported infeasible; stale existence records are corrected.
- Optional feasible remainder: toy_airplane_0, doll_0, toy_animal_0 within wardrobe_0 (bedroom_3); plate_1 and bowl_0 within cabinet_8 (kitchen_0), clean.
- lamp_0 remains on table_3 (living_room_0); lamp_1 remains on counter_0 (kitchen_0). Either feasible light-off subgoal may be completed.
- Unattempted remaining subgoals retain their initial object states.

## Initial robot memory
- toy_fire_truck_0: on table_8 (bedroom_3), falsely remembered to exist.
- stuffed_toy_0: on table_8 (bedroom_3), falsely remembered to exist.
- plate_0: on table_0 (living_room_0), dirty, falsely remembered to exist.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Success criteria
- Give up on missing-target subgoals after bounded verification; do not invent objects or report full success based on memory.
- Optional feasible propositions: is_inside(t, wardrobe_0) for t in {toy_airplane_0, doll_0, toy_animal_0}.
- Optional feasible propositions: is_inside(d, cabinet_8) AND is_clean(d) for d in {plate_1, bowl_0}.
- Optional feasible propositions: is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).

## Spawn / planner notes
- The row's availability-none entry is N/A. This file implements the requested matrix's synthetic absence ablation by removing only uncertainty targets; it is not a row-provided absence scenario.
- Spawn only seven remaining objects. Stale table_8 and table_0 records do not create physical objects.
- table_8 is the grounded bedroom stale location; table_0 is the catalog dining table in living_room_0. Keep all non-target memory accurate.
- Clean at cabinet_5's faucet; open storage before placement; PowerOff the spawned lamps. Start in living_room_0.
