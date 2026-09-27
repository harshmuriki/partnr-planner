# T7-INC-BASE

## Base task
Put away all toys from the living room into the wardrobe, wash and put away the dirty dishes from sink, and turn off the living-room and kitchen lights.

## Scene
- scene_id: 102816756
- rooms used: bedroom_3, living_room_0, kitchen_0
- furniture used (id — room — catalog description):
  - table_8 — bedroom_3 — Dino Dresser, White
  - table_0 — living_room_0 — Palerma Extendable Dining Table, White
  - couch_0 — living_room_0 — Delano 3 Piece Sectional With Left Arm Facing Chaise, Pearl
  - cabinet_5 — kitchen_0 — Kitchen cabinet with sink, double
  - table_3 — living_room_0 — Plinth Coffee Table, Carrara
  - counter_0 — kitchen_0 — Kitchen island, 90x150x90
  - wardrobe_0 — bedroom_3 — Designer Double Wardrobe White &amp; White Gloss
  - cabinet_7 — kitchen_0 — Kitchen corner cabinet

## Task instruction / prompt given
"Put away all toys from the living room into the wardrobe, wash and put away the dirty dishes from sink, and turn off the living-room and kitchen lights."

## Affected object(s)
- toy_truck_0 (toy_fire_truck, asset FIRE_ENGINE): uncertainty target.
- stuffed_toy_0 (stuffed_toy, asset b3e8be210978ec373be6eb5fcffec36c6a9712e6): uncertainty target.
- plate_0 (plate, asset Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring): uncertainty target.
- lamp_living_0 (lamp, asset B07HK83QRB): remaining task object.
- lamp_kitchen_0 (lamp, asset B07HK3PNSK): remaining task object.

## Entity registry
- toy_truck_0: toy_fire_truck, uncertainty target
- stuffed_toy_0: stuffed_toy, uncertainty target
- plate_0: plate, uncertainty target
- lamp_living_0: lamp, remaining task object
- lamp_kitchen_0: lamp, remaining task object

## Uncertainty being tested
- Internal robot memory: Incomplete
- Memory in this version: robot memory has no record of toy_truck_0, stuffed_toy_0, plate_0.

## Information supplied in instruction
- Specified: the toys in the living room, the wardrobe, washing and putting away the dirty dishes from the sink, and the living-room and kitchen lights.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- toy_truck_0: floor floor_living_room_0 (living_room_0), is_clean, is_empty, is_powered_off
- stuffed_toy_0: on couch_0 (living_room_0), is_clean
- plate_0: on cabinet_5 (kitchen_0), is_dirty
- lamp_living_0: on table_3 (living_room_0), is_clean, is_empty, is_powered_on
- lamp_kitchen_0: on counter_0 (kitchen_0), is_clean, is_empty, is_powered_on

## Final expected world state
- toy_truck_0: within wardrobe_0 (bedroom_3), is_clean, is_empty, is_powered_off
- stuffed_toy_0: within wardrobe_0 (bedroom_3), is_clean
- plate_0: within cabinet_7 (kitchen_0), is_clean
- lamp_living_0: on table_3 (living_room_0), is_clean, is_empty, is_powered_off
- lamp_kitchen_0: on counter_0 (kitchen_0), is_clean, is_empty, is_powered_off

## Initial robot memory
- lamp_living_0: on table_3 (living_room_0), is_clean, is_empty, is_powered_on
- lamp_kitchen_0: on counter_0 (kitchen_0), is_clean, is_empty, is_powered_on

## Success criteria
- is_inside(toy_truck_0, wardrobe_0)
- is_inside(stuffed_toy_0, wardrobe_0)
- is_inside(plate_0, cabinet_7)
- is_clean(plate_0)
- is_powered_off(lamp_living_0)
- is_powered_off(lamp_kitchen_0)

## Spawn / planner notes
- Scene 102816756; the robot starts in living_room_0.
- Spawn toy_fire_truck x1 as toy_truck_0 on the living_room_0 floor, pinned asset FIRE_ENGINE; start states: is_clean, is_empty, is_powered_off.
- Spawn stuffed_toy x1 as stuffed_toy_0 on couch_0 (living_room_0), pinned asset b3e8be210978ec373be6eb5fcffec36c6a9712e6; start states: is_clean.
- Spawn plate x1 as plate_0 on cabinet_5 (kitchen_0), pinned asset Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring; start states: is_dirty.
- Spawn lamp x1 as lamp_living_0 on table_3 (living_room_0), pinned asset B07HK83QRB; start states: is_clean, is_empty, is_powered_on.
- Spawn lamp x1 as lamp_kitchen_0 on counter_0 (kitchen_0), pinned asset B07HK3PNSK; start states: is_clean, is_empty, is_powered_on.
- The sink is the kitchen sink cabinet cabinet_5; the living-room sofa is couch_0; the wardrobe is wardrobe_0 (bedroom_3). Kitchen cabinet_8 has no interior receptacle, so the dishes are put away inside the kitchen corner cabinet cabinet_7 (the narrow island cabinet is too small for plate_0).
- The dining table used for stale memory is table_0 (living_room_0). floor_living_room_0 denotes the living-room floor.
- The sheet also lists a toy airplane, doll, toy animal, a second dirty plate and a bowl; they are not in variant_object_assets.csv, so they are not spawned.
- Washing plate_0 needs a faucet-marked object within 1.5 m. This scene's faucet objects are bathroom vanities and a shower; the kitchen sink cabinet (cabinet_5) has no faucet markers. Reachability is not verified.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
