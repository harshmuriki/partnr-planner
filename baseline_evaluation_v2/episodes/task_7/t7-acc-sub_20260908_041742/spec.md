# T7-ACC-SUB

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
- floor_living_room_0 denotes the living-room floor referenced by the request, not furniture.

## Task instruction / prompt given
"Put all five living-room toys into wardrobe_0 in bedroom_3: the fire truck on the floor, the stuffed toy and doll on couch_0, and the airplane and animal on table_3. Wash the dirty plate on counter_0 and the other dirty plate and bowl at the sink on cabinet_5, then store them in cabinet_8. Turn off the living-room lamp on table_3 and the kitchen lamp on counter_0."

## Affected object(s)
- Requested uncertainty targets, not spawned: toy_fire_truck → toy_fire_truck_0; stuffed_toy → stuffed_toy_0; plate → plate_0.
- Remaining task objects: toy_airplane → toy_airplane_0; doll → doll_0; toy_animal → toy_animal_0; plate → plate_1; bowl → bowl_0; lamp → lamp_0, lamp_1.
- Suitable substitutes: none authorized; the task row says N/A. No distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: Substitute Available

## Information supplied in instruction
- Requests five named toys, two dirty plates, one dirty bowl, exact sources, cleaning/storage destinations, and two lights.
- Does not disclose missing targets or authorize replacements. The retained source descriptions are requests, not assertions overriding the world.

## Initial world state
- toy_fire_truck_0, stuffed_toy_0, plate_0: absent; no replacement instances exist.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Final expected world state
- Diagnostic feasible remainder: toy_airplane_0, doll_0, toy_animal_0 within wardrobe_0 (bedroom_3).
- plate_1, bowl_0: within cabinet_8 (kitchen_0), clean.
- lamp_0: on table_3 (living_room_0), powered off.
- lamp_1: on counter_0 (kitchen_0), powered off.
- Missing targets remain absent. A substitute-success world cannot be specified from the supplied row.

## Initial robot memory
- toy_fire_truck_0, stuffed_toy_0, plate_0: known absent; no suitable substitutes available.
- toy_airplane_0: on table_3 (living_room_0).
- doll_0: on couch_0 (living_room_0).
- toy_animal_0: on table_3 (living_room_0).
- plate_1: on cabinet_5 (kitchen_0), dirty.
- bowl_0: on cabinet_5 (kitchen_0), dirty.
- lamp_0: on table_3 (living_room_0), powered on.
- lamp_1: on counter_0 (kitchen_0), powered on.
- wardrobe_0: Close; cabinet_8: Close.

## Success criteria
- No valid SUB score: no authorized substitute can satisfy a substitute proposition. Do not report full or substitute success.
- Diagnostic remainder: is_inside(t, wardrobe_0) for each t in {toy_airplane_0, doll_0, toy_animal_0}.
- Diagnostic remainder: is_inside(d, cabinet_8) AND is_clean(d) for each d in {plate_1, bowl_0}.
- Diagnostic remainder: is_on_top(lamp_0, table_3) AND is_powered_off(lamp_0); is_on_top(lamp_1, counter_0) AND is_powered_off(lamp_1).
- Report the unavailable subgoals; do not invent objects or count existing remaining objects twice.

## Spawn / planner notes
- This is an explicitly unsupported matrix entry, not a valid SUB benchmark. The absence-only diagnostic fixture above is supplied to keep the file executable without inventing substitutes; exclude it from SUB evaluation.
- Spawn only the seven existing instances at the listed surfaces. Leave all non-target objects in their BASE positions.
- Clean at cabinet_5's faucet; use spawned lamps for PowerOff. Open wardrobe_0 and cabinet_8 before placement. Start in living_room_0.
