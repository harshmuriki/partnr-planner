# T1-OUT-CON

## Base task
Turn off all the lights in the bedroom and bring a full jug of water to the living-room table.

## Scene
- scene_id: 103997895_171031182
- rooms used: kitchen_0, bedroom_0, living_room_0
- furniture used (id — room — catalog description):
  - table_2 — kitchen_0 — Bar
  - fridge_0 — kitchen_0 — American fridge freezer
  - chest_of_drawers_0 — bedroom_0 — Presby Nightstand, White
  - table_0 — living_room_0 — Tulip Table (90cm)

## Task instruction / prompt given
"Turn off all the lights in the bedroom and bring a full jug of water to the living-room table."

## Affected object(s)
- jug_0 (jug, asset 264b8bf954ce7d7d2f0c135151bcb6bf6b11acb4): uncertainty target.
- lamp_0 (lamp, asset B07HK83QRB): remaining task object.

## Entity registry
- jug_0: jug, uncertainty target
- lamp_0: lamp, remaining task object

## Uncertainty being tested
- Internal robot memory: Outdated
- Containment: Inside Closed Receptacle
- Memory in this version: robot memory places jug_0 at stale locations.

## Information supplied in instruction
- Specified: the bedroom lights, a full jug of water, and the living-room table as the destination.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- jug_0: within fridge_0 (kitchen_0), is_clean, is_empty, fridge_0 starts closed
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_on

## Final expected world state
- jug_0: on table_0 (living_room_0), is_clean, is_filled
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_off

## Initial robot memory
- jug_0: on table_2 (kitchen_0), is_clean, is_empty, stale record
- lamp_0: on chest_of_drawers_0 (bedroom_0), is_clean, is_empty, is_powered_on

## Success criteria
- is_on_top(jug_0, table_0)
- is_filled(jug_0)
- is_powered_off(lamp_0)

## Spawn / planner notes
- Scene 103997895_171031182; the robot starts in living_room_0.
- Spawn jug x1 as jug_0 within fridge_0 (kitchen_0), pinned asset 264b8bf954ce7d7d2f0c135151bcb6bf6b11acb4; start states: is_clean, is_empty.
- Spawn lamp x1 as lamp_0 on chest_of_drawers_0 (bedroom_0), pinned asset B07HK83QRB; start states: is_clean, is_empty, is_powered_on.
- Close fridge_0 after spawning; the robot must open it to retrieve jug_0.
- The scene has no dining room: the kitchen bar table (table_2) stands in for the dining table where Outdated memory places the jug.
- The bedroom whose lights are turned off is bedroom_0 (the bedroom with a nightstand for the lamp); its lights are the spawned lamp_0.
- The living-room table is the Tulip table table_0.
- Filling needs the robot within 1.5 m of a faucet-marked object. This scene's faucet objects are a double washbasin and a bathtub; no kitchen furniture carries faucet markers. Reachability is not verified.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
