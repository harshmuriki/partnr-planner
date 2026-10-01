# T1-OUT-SUB

## Base task
Turn off all the lights in the bedroom and bring a full jug of water to the living-room table.

## Scene
- scene_id: 103997895_171031182
- rooms used: kitchen_0, bedroom_0, living_room_0
- furniture used (id — room — catalog description):
  - table_2 — kitchen_0 — Bar
  - counter_0 — kitchen_0 — Kitchen island, 60x100x90
  - chest_of_drawers_0 — bedroom_0 — Presby Nightstand, White
  - table_0 — living_room_0 — Tulip Table (90cm)

## Task instruction / prompt given
"Turn off all the lights in the bedroom and bring a full jug of water to the living-room table."

## Affected object(s)
- pitcher_0 (pitcher, asset Threshold_Porcelain_Pitcher_White): substitute.
- lamp_0 (lamp, asset B07HK83QRB): remaining task object.
- jug_0 (jug): uncertainty target, absent from the scene in this variant; pitcher_0 is available instead.

## Entity registry
- pitcher_0: pitcher, substitute
- lamp_0: lamp, remaining task object
- jug_0: jug, uncertainty target

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: Substitute Available
- Memory in this version: robot memory places jug_0 at stale locations; it has no record of pitcher_0.

## Information supplied in instruction
- Specified: the bedroom lights, a full jug of water, and the living-room table as the destination.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- pitcher_0: on counter_0 (kitchen_0), is_clean, is_empty
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_on

## Final expected world state
- pitcher_0: on table_0 (living_room_0), is_clean, is_filled
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_off

## Initial robot memory
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_on
- jug_0: on table_2 (kitchen_0), is_clean, is_empty, stale record; the object is not actually in the scene

## Success criteria
- is_on_top(pitcher_0, table_0)
- is_filled(pitcher_0)
- is_powered_off(lamp_0)
- Using pitcher_0 in place of the missing jug_0 counts as success; do not require or invent a jug.

## Spawn / planner notes
- Scene 103997895_171031182; the robot starts in living_room_0.
- Spawn pitcher x1 as pitcher_0 on counter_0 (kitchen_0), pinned asset Threshold_Porcelain_Pitcher_White; start states: is_clean, is_empty.
- Spawn lamp x1 as lamp_0 on chest_of_drawers_0 (bedroom_0), pinned asset B07HK83QRB; start states: is_clean, is_empty, is_powered_on.
- Do not spawn jug_0 (jug) in this variant.
- The scene has no dining room: the kitchen bar table (table_2) stands in for the dining table where Outdated memory places the jug.
- The bedroom whose lights are turned off is bedroom_0 (the bedroom with a nightstand for the lamp); its lights are the spawned lamp_0.
- The living-room table is the Tulip table table_0.
- Filling needs the robot within 1.5 m of a faucet-marked object. The usable faucet is the kitchen sink cabinet cabinet_6 (kitchen_0), which the runtime world graph names cabinet_34; the robot can fill the jug while holding it there (verified in the simulator). A second sink, cabinet_5, is in laundryroom/mudroom_0.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
