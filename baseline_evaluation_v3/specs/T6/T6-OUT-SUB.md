# T6-OUT-SUB

## Base task
Heat up the bread, bring it to the bedroom desk, bring a water bottle and a clean hand towel and place them next to the bread. Turn off the living-room lights.

## Scene
- scene_id: 104348010_171512832
- rooms used: dining_room_0, bedroom_2, kitchen_0, bathroom_2, living_room_0
- furniture used (id — room — catalog description):
  - table_7 — dining_room_0 — table
  - table_5 — bedroom_2 — Clyde side table
  - chair_9 — bedroom_2 — Etienne Upholstered Chair
  - counter_0 — kitchen_0 — YOO OH Kitchen Island
  - shelves_11 — bathroom_2 — Parson 5-Shelf Bookcase
  - table_2 — living_room_0 — Dark Oak
  - table_11 — bedroom_2 — Vanity Desk - Antique White
  - microwave_0 — kitchen_0 — Samsung Convection Oven with Microwave and Grill.

## Task instruction / prompt given
"Heat up the bread, bring it to the bedroom desk, bring a water bottle and a clean hand towel and place them next to the bread. Turn off the living-room lights."

## Affected object(s)
- bread_0 (bread, asset Bread_8): uncertainty target.
- cup_0 (cup, asset ed95f32cc00075ceed5bd4b593d2a06a444c0eec): substitute.
- blue_hand_towel_0 (hand_towel, asset Tag_Dishtowel_Dobby_Stripe_Blue_18_x_26): substitute.
- lamp_living_0 (lamp, asset B07HK3PNSK): remaining task object.
- bottle_0 (bottle): uncertainty target, absent from the scene in this variant; cup_0 is available instead.
- hand_towel_0 (hand_towel): uncertainty target, absent from the scene in this variant; blue_hand_towel_0 is available instead.

## Entity registry
- bread_0: bread, uncertainty target
- cup_0: cup, substitute
- blue_hand_towel_0: hand_towel, substitute
- lamp_living_0: lamp, remaining task object
- bottle_0: bottle, uncertainty target
- hand_towel_0: hand_towel, uncertainty target

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: Substitute Available
- Memory in this version: robot memory places bread_0, bottle_0, hand_towel_0 at stale locations; it has no record of cup_0, blue_hand_towel_0.

## Information supplied in instruction
- Specified: heating the bread, the water bottle, the clean hand towel, the bedroom desk as the destination, and the living-room lights.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- bread_0: on counter_0 (kitchen_0)
- cup_0: on table_7 (dining_room_0), is_clean, is_filled
- blue_hand_towel_0: on shelves_11 (bathroom_2), is_clean
- lamp_living_0: on table_2 (living_room_0), is_clean, is_empty, is_powered_on

## Final expected world state
- bread_0: on table_11 (bedroom_2), next to cup_0, next to blue_hand_towel_0
- cup_0: on table_11 (bedroom_2), is_clean, is_filled, next to bread_0
- blue_hand_towel_0: on table_11 (bedroom_2), is_clean, next to bread_0
- lamp_living_0: on table_2 (living_room_0), is_clean, is_empty, is_powered_off

## Initial robot memory
- bread_0: on table_7 (dining_room_0), stale record
- lamp_living_0: on table_2 (living_room_0), is_clean, is_empty, is_powered_on
- bottle_0: on table_5 (bedroom_2), is_clean, is_filled, stale record; the object is not actually in the scene
- hand_towel_0: on chair_9 (bedroom_2), is_clean, stale record; the object is not actually in the scene

## Success criteria
- is_inside(bread_0, microwave_0)
- is_on_top(bread_0, table_11)
- order: is_inside(bread_0, microwave_0) before is_on_top(bread_0, table_11)
- is_on_top(cup_0, table_11)
- is_on_top(blue_hand_towel_0, table_11)
- is_next_to(cup_0, bread_0)
- is_next_to(blue_hand_towel_0, bread_0)
- is_filled(cup_0)
- is_clean(blue_hand_towel_0)
- is_powered_off(lamp_living_0)
- Using blue_hand_towel_0 in place of the missing hand_towel_0 counts as success; do not require or invent a hand_towel.
- Using cup_0 in place of the missing bottle_0 counts as success; do not require or invent a bottle.

## Spawn / planner notes
- Scene 104348010_171512832; the robot starts in living_room_0.
- Spawn bread x1 as bread_0 on counter_0 (kitchen_0), pinned asset Bread_8; start states: no object states.
- Spawn cup x1 as cup_0 on table_7 (dining_room_0), pinned asset ed95f32cc00075ceed5bd4b593d2a06a444c0eec; start states: is_clean, is_filled.
- Spawn hand_towel x1 as blue_hand_towel_0 on shelves_11 (bathroom_2), pinned asset Tag_Dishtowel_Dobby_Stripe_Blue_18_x_26; start states: is_clean.
- Spawn lamp x1 as lamp_living_0 on table_2 (living_room_0), pinned asset B07HK3PNSK; start states: is_clean, is_empty, is_powered_on.
- Do not spawn bottle_0 (bottle) in this variant.
- Do not spawn hand_towel_0 (hand_towel) in this variant.
- The master-bedroom desk is the vanity desk table_11 (bedroom_2); 'master' has no scene referent, so the instruction says 'bedroom'. The living-room lights are the spawned lamp_living_0.
- The scene's fridge is in the garage (fridge_0). Kitchen cabinet_0 has no interior receptacle, so the bread cabinet is cabinet_1. No bathroom furniture has an interior receptacle, so the towel cabinet is cabinet_23 in the laundry room.
- The sheet's 'baguette' distractor is bread_25_0 (Bread_25). Heating has no skill or object state, so it is scored as a microwave visit: bread_0 must be placed inside microwave_0 (kitchen_0) before it is placed on table_11, and it does not need to remain in the microwave.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
