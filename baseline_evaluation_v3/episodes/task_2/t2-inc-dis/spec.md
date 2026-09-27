# T2-INC-DIS

## Base task
Move both packages to the living-room center table and place them next to each other. Bring the scissors and place beside them to open the packages.

## Scene
- scene_id: 106878960_174887073
- rooms used: living_room_0, dining_room_0, entryway/foyer/lobby_0
- furniture used (id — room — catalog description):
  - table_0 — dining_room_0 — Livingston Dining Table 62"
  - chest_of_drawers_2 — entryway/foyer/lobby_0 — Bombay Two Drawer Chest, Black
  - table_16 — living_room_0 — Madison Park Signature Bordeaux Coffee Table

## Task instruction / prompt given
"Move both packages to the living-room center table and place them next to each other. Bring the scissors and place beside them to open the packages."

## Affected object(s)
- box_0 (box, asset B073PB1H88): remaining task object.
- box_1 (box, asset B071225BBS): uncertainty target.
- scissors_0 (scissors, asset Diamond_Visions_Scissors_Red): uncertainty target.
- board_game_0 (board_game, asset Clue_Board_Game_Classic_Edition): distractor, placed next to box_1.
- distractor_knife_0 (knife, asset ButterKnife_1): distractor, placed next to scissors_0.
- screwdriver_0 (screwdriver, asset Craftsman_Grip_Screwdriver_Phillips_Cushion): distractor, placed next to scissors_0.

## Entity registry
- box_0: box, remaining task object
- box_1: box, uncertainty target
- scissors_0: scissors, uncertainty target
- board_game_0: board_game, distractor
- distractor_knife_0: knife, distractor
- screwdriver_0: screwdriver, distractor

## Uncertainty being tested
- Internal robot memory: Incomplete
- Distractors: Present
- Memory in this version: robot memory has no record of box_1, scissors_0.

## Information supplied in instruction
- Specified: both packages, the living-room center table, their side-by-side arrangement, and the scissors.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- box_0: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, is_empty
- box_1: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, is_empty
- scissors_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), is_clean
- board_game_0: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, next to box_1
- distractor_knife_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), is_clean, next to scissors_0
- screwdriver_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), is_clean, next to scissors_0

## Final expected world state
- box_0: on table_16 (living_room_0), is_clean, is_empty, next to box_1, next to scissors_0
- box_1: on table_16 (living_room_0), is_clean, is_empty, next to box_0, next to scissors_0
- scissors_0: on table_16 (living_room_0), is_clean, next to box_0, next to box_1
- board_game_0: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean
- distractor_knife_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), is_clean
- screwdriver_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), is_clean

## Initial robot memory
- box_0: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, is_empty
- board_game_0: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, next to box_1
- distractor_knife_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), is_clean, next to scissors_0
- screwdriver_0: on chest_of_drawers_2 (entryway/foyer/lobby_0), is_clean, next to scissors_0

## Success criteria
- is_on_top(box_0, table_16)
- is_on_top(box_1, table_16)
- is_on_top(scissors_0, table_16)
- is_next_to(box_0, box_1)
- is_next_to(scissors_0, box_0)
- is_next_to(scissors_0, box_1)
- Distractors (board_game_0, distractor_knife_0, screwdriver_0) must not be used in place of the task objects; they have no placement goals.

## Spawn / planner notes
- Scene 106878960_174887073; the robot starts in entryway/foyer/lobby_0.
- Spawn box x1 as box_0 on the entryway/foyer/lobby_0 floor, pinned asset B073PB1H88; start states: is_clean, is_empty.
- Spawn box x1 as box_1 on the entryway/foyer/lobby_0 floor, pinned asset B071225BBS; start states: is_clean, is_empty.
- Spawn scissors x1 as scissors_0 on chest_of_drawers_2 (entryway/foyer/lobby_0), pinned asset Diamond_Visions_Scissors_Red; start states: is_clean.
- Spawn board_game x1 as board_game_0 on the entryway/foyer/lobby_0 floor, next to box_1, pinned asset Clue_Board_Game_Classic_Edition; start states: is_clean.
- Spawn knife x1 as distractor_knife_0 on chest_of_drawers_2 (entryway/foyer/lobby_0), next to scissors_0, pinned asset ButterKnife_1; start states: is_clean.
- Spawn screwdriver x1 as screwdriver_0 on chest_of_drawers_2 (entryway/foyer/lobby_0), next to scissors_0, pinned asset Craftsman_Grip_Screwdriver_Phillips_Cushion; start states: is_clean.
- The entryway table is the top of chest_of_drawers_2, and the entryway drawer is its interior. floor_entryway_foyer_lobby_0 and floor_living_room_0 denote those rooms' floors.
- The living-room center table is the coffee table table_16. The dining table used for stale memory is table_0 (dining_room_0).
- box_1 is the selected package whose memory varies; box_0 is always remembered correctly.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
