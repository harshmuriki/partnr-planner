# T6-ACC-CON

## Base task
Heat up the bread, bring it to the bedroom desk, bring a water bottle and a clean hand towel and place them next to the bread. Turn off the living-room lights.

## Scene
- scene_id: 104348010_171512832
- rooms used: dining_room_0, bedroom_2, kitchen_0, garage_0, laundryroom/mudroom_0, living_room_0
- furniture used (id — room — catalog description):
  - table_7 — dining_room_0 — table
  - table_5 — bedroom_2 — Clyde side table
  - chair_9 — bedroom_2 — Etienne Upholstered Chair
  - cabinet_1 — kitchen_0 — Kitchen cabinet with 2 doors
  - fridge_0 — garage_0 — KEW - Fridge
  - cabinet_23 — laundryroom/mudroom_0 — Kitchen cabinet with door
  - table_2 — living_room_0 — Dark Oak
  - table_11 — bedroom_2 — Vanity Desk - Antique White
  - microwave_0 — kitchen_0 — Samsung Convection Oven with Microwave and Grill.

## Task instruction / prompt given
"Heat up the bread, bring it to the bedroom desk, bring a water bottle and a clean hand towel and place them next to the bread. Turn off the living-room lights."

## Affected object(s)
- bread_0 (bread, asset Bread_8): uncertainty target.
- bottle_0 (bottle, asset 03758534dd2a3a8303e742cf4fc10fedd4c48843): uncertainty target.
- hand_towel_0 (hand_towel, asset Tag_Dishtowel_Green): uncertainty target.
- lamp_living_0 (lamp, asset B07HK3PNSK): remaining task object.

## Entity registry
- bread_0: bread, uncertainty target
- bottle_0: bottle, uncertainty target
- hand_towel_0: hand_towel, uncertainty target
- lamp_living_0: lamp, remaining task object

## Uncertainty being tested
- Internal robot memory: Accurate
- Containment: Inside Closed Receptacle
- Memory in this version: robot memory matches the initial world state.

## Information supplied in instruction
- Specified: heating the bread, the water bottle, the clean hand towel, the bedroom desk as the destination, and the living-room lights.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- bread_0: within cabinet_1 (kitchen_0), cabinet_1 starts closed
- bottle_0: within fridge_0 (garage_0), is_clean, is_filled, fridge_0 starts closed
- hand_towel_0: within cabinet_23 (laundryroom/mudroom_0), is_clean, cabinet_23 starts closed
- lamp_living_0: on table_2 (living_room_0), is_clean, is_empty, is_powered_on

## Final expected world state
- bread_0: on table_11 (bedroom_2), next to bottle_0, next to hand_towel_0
- bottle_0: on table_11 (bedroom_2), is_clean, is_filled, next to bread_0
- hand_towel_0: on table_11 (bedroom_2), is_clean, next to bread_0
- lamp_living_0: on table_2 (living_room_0), is_clean, is_empty, is_powered_off

## Initial robot memory
- bread_0: within cabinet_1 (kitchen_0), cabinet_1 starts closed
- bottle_0: within fridge_0 (garage_0), is_clean, is_filled, fridge_0 starts closed
- hand_towel_0: within cabinet_23 (laundryroom/mudroom_0), is_clean, cabinet_23 starts closed
- lamp_living_0: on table_2 (living_room_0), is_clean, is_empty, is_powered_on

## Success criteria
- is_inside(bread_0, microwave_0)
- is_on_top(bread_0, table_11)
- order: is_inside(bread_0, microwave_0) before is_on_top(bread_0, table_11)
- is_on_top(bottle_0, table_11)
- is_on_top(hand_towel_0, table_11)
- is_next_to(bottle_0, bread_0)
- is_next_to(hand_towel_0, bread_0)
- is_filled(bottle_0)
- is_clean(hand_towel_0)
- is_powered_off(lamp_living_0)

## Spawn / planner notes
- Scene 104348010_171512832; the robot starts in living_room_0.
- Spawn bread x1 as bread_0 within cabinet_1 (kitchen_0), pinned asset Bread_8; start states: no object states.
- Spawn bottle x1 as bottle_0 within fridge_0 (garage_0), pinned asset 03758534dd2a3a8303e742cf4fc10fedd4c48843; start states: is_clean, is_filled.
- Spawn hand_towel x1 as hand_towel_0 within cabinet_23 (laundryroom/mudroom_0), pinned asset Tag_Dishtowel_Green; start states: is_clean.
- Spawn lamp x1 as lamp_living_0 on table_2 (living_room_0), pinned asset B07HK3PNSK; start states: is_clean, is_empty, is_powered_on.
- Close cabinet_1 after spawning; the robot must open it to retrieve bread_0.
- Close cabinet_23 after spawning; the robot must open it to retrieve hand_towel_0.
- Close fridge_0 after spawning; the robot must open it to retrieve bottle_0.
- The master-bedroom desk is the vanity desk table_11 (bedroom_2); 'master' has no scene referent, so the instruction says 'bedroom'. The living-room lights are the spawned lamp_living_0.
- The scene's fridge is in the garage (fridge_0). Kitchen cabinet_0 has no interior receptacle, so the bread cabinet is cabinet_1. No bathroom furniture has an interior receptacle, so the towel cabinet is cabinet_23 in the laundry room.
- The sheet's 'baguette' distractor is bread_25_0 (Bread_25). Heating has no skill or object state, so it is scored as a microwave visit: bread_0 must be placed inside microwave_0 (kitchen_0) before it is placed on table_11, and it does not need to remain in the microwave.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
