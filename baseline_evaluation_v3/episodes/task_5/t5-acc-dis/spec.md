# T5-ACC-DIS

## Base task
Get soap to clean both glasses, fill them, and place them on the living-room table for our guests to drink. Get a white plate and place it next to them.

## Scene
- scene_id: 106878915_174887025
- rooms used: bathroom_1, kitchen_0, living_room_0
- furniture used (id — room — catalog description):
  - washer_dryer_0 — bathroom_1 — Washing machine
  - cabinet_0 — kitchen_0 — Kitchen
  - table_2 — kitchen_0 — SKOGSTA Dining table
  - table_3 — living_room_0 — Marrakesh Console Table
  - table_0 — living_room_0 — Dip-Dyed Side Table

## Task instruction / prompt given
"Get soap to clean both glasses, fill them, and place them on the living-room table for our guests to drink. Get a white plate and place it next to them."

## Affected object(s)
- soap_dispenser_0 (soap_dispenser, asset Soap_Bottle_26): uncertainty target.
- glass_0 (glass, asset 9428caf70c42e50bb3c3c604eba00c43b8efb58f): uncertainty target.
- glass_1 (glass, asset f9ce9e87b840524cfd4404f4a0cfc52d88f2c794): remaining task object.
- plate_0 (plate, asset Plate_26): uncertainty target.
- soap_dish_0 (soap_dish, asset Threshold_Bamboo_Ceramic_Soap_Dish): distractor, placed next to soap_dispenser_0.
- vase_0 (vase, asset 04f409f17d79445a3afd109c557c6a2702e15cd8): distractor, placed next to glass_0.
- plant_saucer_0 (plant_saucer, asset Cole_Hardware_Plant_Saucer_Brown_125): distractor, placed next to plate_0.

## Entity registry
- soap_dispenser_0: soap_dispenser, uncertainty target
- glass_0: glass, uncertainty target
- glass_1: glass, remaining task object
- plate_0: plate, uncertainty target
- soap_dish_0: soap_dish, distractor
- vase_0: vase, distractor
- plant_saucer_0: plant_saucer, distractor

## Uncertainty being tested
- Internal robot memory: Accurate
- Distractors: Present
- Memory in this version: robot memory matches the initial world state.

## Information supplied in instruction
- Specified: the soap, both glasses, cleaning and filling them, the white plate, and the living-room table as the destination.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), is_clean, is_empty
- glass_0: on table_2 (kitchen_0), is_dirty, is_empty
- glass_1: on table_3 (living_room_0), is_dirty, is_empty
- plate_0: on cabinet_0 (kitchen_0), is_clean
- soap_dish_0: on cabinet_0 (kitchen_0), is_clean, is_empty, next to soap_dispenser_0
- vase_0: on table_2 (kitchen_0), is_clean, is_empty, next to glass_0
- plant_saucer_0: on cabinet_0 (kitchen_0), is_clean, is_empty, next to plate_0

## Final expected world state
- soap_dispenser_0: on cabinet_0 (kitchen_0), is_clean, is_empty
- glass_0: on table_0 (living_room_0), is_clean, is_filled, next to plate_0
- glass_1: on table_0 (living_room_0), is_clean, is_filled, next to plate_0
- plate_0: on table_0 (living_room_0), is_clean, next to glass_0, next to glass_1
- soap_dish_0: on cabinet_0 (kitchen_0), is_clean, is_empty, next to soap_dispenser_0
- vase_0: on table_2 (kitchen_0), is_clean, is_empty
- plant_saucer_0: on cabinet_0 (kitchen_0), is_clean, is_empty

## Initial robot memory
- soap_dispenser_0: on cabinet_0 (kitchen_0), is_clean, is_empty
- glass_0: on table_2 (kitchen_0), is_dirty, is_empty
- glass_1: on table_3 (living_room_0), is_dirty, is_empty
- plate_0: on cabinet_0 (kitchen_0), is_clean
- soap_dish_0: on cabinet_0 (kitchen_0), is_clean, is_empty, next to soap_dispenser_0
- vase_0: on table_2 (kitchen_0), is_clean, is_empty, next to glass_0
- plant_saucer_0: on cabinet_0 (kitchen_0), is_clean, is_empty, next to plate_0

## Success criteria
- is_on_top(glass_0, table_0)
- is_on_top(glass_1, table_0)
- is_on_top(plate_0, table_0)
- is_next_to(plate_0, glass_0)
- is_next_to(plate_0, glass_1)
- is_next_to(soap_dispenser_0, glass_0)
- is_clean(glass_0)
- order: is_next_to(soap_dispenser_0, glass_0) before is_clean(glass_0)
- is_filled(glass_0)
- is_next_to(soap_dispenser_0, glass_1)
- is_clean(glass_1)
- order: is_next_to(soap_dispenser_0, glass_1) before is_clean(glass_1)
- is_filled(glass_1)
- Distractors (soap_dish_0, vase_0, plant_saucer_0) must not be used in place of the task objects; they have no placement goals.

## Spawn / planner notes
- Scene 106878915_174887025; the robot starts in living_room_0.
- Spawn soap_dispenser x1 as soap_dispenser_0 on cabinet_0 (kitchen_0), pinned asset Soap_Bottle_26; start states: is_clean, is_empty.
- Spawn glass x1 as glass_0 on table_2 (kitchen_0), pinned asset 9428caf70c42e50bb3c3c604eba00c43b8efb58f; start states: is_dirty, is_empty.
- Spawn glass x1 as glass_1 on table_3 (living_room_0), pinned asset f9ce9e87b840524cfd4404f4a0cfc52d88f2c794; start states: is_dirty, is_empty.
- Spawn plate x1 as plate_0 on cabinet_0 (kitchen_0), pinned asset Plate_26; start states: is_clean.
- Spawn soap_dish x1 as soap_dish_0 on cabinet_0 (kitchen_0), next to soap_dispenser_0, pinned asset Threshold_Bamboo_Ceramic_Soap_Dish; start states: is_clean, is_empty.
- Spawn vase x1 as vase_0 on table_2 (kitchen_0), next to glass_0, pinned asset 04f409f17d79445a3afd109c557c6a2702e15cd8; start states: is_clean, is_empty.
- Spawn plant_saucer x1 as plant_saucer_0 on cabinet_0 (kitchen_0), next to plate_0, pinned asset Cole_Hardware_Plant_Saucer_Brown_125; start states: is_clean, is_empty.
- The scene has no dining room and a single kitchen cabinet: the dining table is table_2 (kitchen_0), the kitchen counter / sink area is the top of cabinet_0, and both the lower and upper closed cabinets are cabinet_0's interior.
- glass_0 is the selected glass (on the dining table); glass_1 starts on the living-room console table_3 so the two glasses are on different tables. The destination living-room table is table_0.
- Sheet note: 'Uncertainty targets' says 'one bottle', but every memory cell varies the selected glass; glass_0 is used.
- Cleaning and filling the glasses need a faucet-marked object within 1.5 m. This scene's faucet objects are bathroom vanities, a bath mixer and a freestanding bath (bathroom_1, near washer_dryer_0); the kitchen cabinet has no faucet markers. Reachability is not verified.
- Soap use has no skill or object state, so it is scored as bringing the soap: soap_dispenser_0 must be next to each glass before that glass becomes clean. The soap does not need to stay there; its final location is not scored.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
