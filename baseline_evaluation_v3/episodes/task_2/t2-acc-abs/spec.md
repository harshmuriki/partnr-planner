# T2-ACC-ABS

## Base task
Move both packages to the living-room center table and place them next to each other. Bring the scissors and place beside them to open the packages.

## Scene
- scene_id: 106878960_174887073
- rooms used: living_room_0, dining_room_0, entryway/foyer/lobby_0
- furniture used (id — room — catalog description):
  - table_0 — dining_room_0 — Livingston Dining Table 62"
  - table_16 — living_room_0 — Madison Park Signature Bordeaux Coffee Table

## Task instruction / prompt given
"Move both packages to the living-room center table and place them next to each other. Bring the scissors and place beside them to open the packages."

## Affected object(s)
- box_0 (box, asset B073PB1H88): remaining task object.
- box_1 (box, asset B071225BBS): uncertainty target.
- scissors_0 (scissors): uncertainty target, absent from the scene in this variant; no suitable substitute exists.

## Entity registry
- box_0: box, remaining task object
- box_1: box, uncertainty target
- scissors_0: scissors, uncertainty target

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: No Suitable Object
- Memory in this version: robot memory matches the initial world state and explicitly knows that scissors_0 does not exist.

## Information supplied in instruction
- Specified: both packages, the living-room center table, their side-by-side arrangement, and the scissors.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- box_0: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, is_empty
- box_1: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, is_empty

## Final expected world state
- box_0: on table_16 (living_room_0), is_clean, is_empty, next to box_1
- box_1: on table_16 (living_room_0), is_clean, is_empty, next to box_0

## Initial robot memory
- box_0: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, is_empty
- box_1: floor floor_entryway_foyer_lobby_0 (entryway/foyer/lobby_0), is_clean, is_empty
- scissors_0: known absent; no scissors or acceptable substitute (knife) exists anywhere in the scene

## Success criteria
- is_on_top(box_0, table_16)
- is_on_top(box_1, table_16)
- is_next_to(box_0, box_1)
- The scissors (scissors_0) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

## Spawn / planner notes
- Scene 106878960_174887073; the robot starts in entryway/foyer/lobby_0.
- Spawn box x1 as box_0 on the entryway/foyer/lobby_0 floor, pinned asset B073PB1H88; start states: is_clean, is_empty.
- Spawn box x1 as box_1 on the entryway/foyer/lobby_0 floor, pinned asset B071225BBS; start states: is_clean, is_empty.
- Do not spawn scissors_0 (scissors) in this variant or any substitute.
- The entryway table is the top of chest_of_drawers_2, and the entryway drawer is its interior. floor_entryway_foyer_lobby_0 and floor_living_room_0 denote those rooms' floors.
- The living-room center table is the coffee table table_16. The dining table used for stale memory is table_0 (dining_room_0).
- box_1 is the selected package whose memory varies; box_0 is always remembered correctly.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
