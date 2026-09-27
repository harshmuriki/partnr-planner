# T1-ACC-ABS

## Base task
Turn off all the lights in the bedroom and bring a full jug of water to the living-room table.

## Scene
- scene_id: 103997895_171031182
- rooms used: kitchen_0, bedroom_0
- furniture used (id — room — catalog description):
  - table_2 — kitchen_0 — Bar
  - chest_of_drawers_0 — bedroom_0 — Presby Nightstand, White

## Task instruction / prompt given
"Turn off all the lights in the bedroom and bring a full jug of water to the living-room table."

## Affected object(s)
- lamp_0 (lamp, asset B07HK83QRB): remaining task object.
- jug_0 (jug): uncertainty target, absent from the scene in this variant; no suitable substitute exists.

## Entity registry
- lamp_0: lamp, remaining task object
- jug_0: jug, uncertainty target

## Uncertainty being tested
- Internal robot memory: Accurate
- Object availability: No Suitable Object
- Memory in this version: robot memory matches the initial world state and explicitly knows that jug_0 does not exist.

## Information supplied in instruction
- Specified: the bedroom lights, a full jug of water, and the living-room table as the destination.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_on

## Final expected world state
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_off

## Initial robot memory
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_on
- jug_0: known absent; no jug or acceptable substitute (pitcher) exists anywhere in the scene

## Success criteria
- is_powered_off(lamp_0)
- The jug (jug_0) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

## Spawn / planner notes
- Scene 103997895_171031182; the robot starts in living_room_0.
- Spawn lamp x1 as lamp_0 on chest_of_drawers_0 (bedroom_0), pinned asset B07HK83QRB; start states: is_clean, is_empty, is_powered_on.
- Do not spawn jug_0 (jug) in this variant or any substitute.
- The scene has no dining room: the kitchen bar table (table_2) stands in for the dining table where Outdated memory places the jug.
- The bedroom whose lights are turned off is bedroom_0 (the bedroom with a nightstand for the lamp); its lights are the spawned lamp_0.
- The living-room table is the Tulip table table_0.
- Filling needs the robot within 1.5 m of a faucet-marked object. This scene's faucet objects are a double washbasin and a bathtub; no kitchen furniture carries faucet markers. Reachability is not verified.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
