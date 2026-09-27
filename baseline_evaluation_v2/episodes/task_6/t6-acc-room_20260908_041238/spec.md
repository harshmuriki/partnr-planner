# T6-ACC-ROOM

## Base task
Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights.

## Scene
- scene_id: 104348010_171512832
- rooms used: kitchen_0, dining_room_0, bathroom_2, bedroom_2, living_room_0
- furniture used (id — room — catalog description):
  - counter_0 — kitchen_0 — YOO OH Kitchen Island
  - table_7 — dining_room_0 — table
  - shelves_11 — bathroom_2 — Parson 5-Shelf Bookcase
  - table_11 — bedroom_2 — Vanity Desk - Antique White
  - table_2 — living_room_0 — Dark Oak

## Task instruction / prompt given
"Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights. The bread is in the kitchen, the water bottle is in the dining room, and the clean towel is in the bathroom."

## Affected object(s)
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, clean). All three are uncertainty targets with room-only source knowledge.
- lamp: lamp_0, remaining task object. No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Localization: Room Known

## Information supplied in instruction
- Specifies one bread, one water bottle, one clean towel, adjacency at the master-bedroom desk, and living-room lights off.
- Supplies source rooms only: kitchen_0, dining_room_0, and bathroom_2, respectively. The bathroom reference is bound to bathroom_2.
- Omits exact source furniture and towel color. Bind master bedroom to bedroom_2 and destination desk to table_11.
- Heating is requested but unsupported.

## Initial world state
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- hand_towel_0: on shelves_11 (bathroom_2), green, clean.
- lamp_0: on table_2 (living_room_0), powered on.

## Final expected world state
- bread_0: on table_11 (bedroom_2); no heating requirement.
- bottle_0: on table_11 (bedroom_2), filled with water, next_to bread_0.
- hand_towel_0: on table_11 (bedroom_2), green, clean, next_to bread_0.
- lamp_0: on table_2 (living_room_0), powered off.

## Initial robot memory
- bread_0: in_room kitchen_0, ordinary bread; exact source furniture withheld; no heated state modeled.
- bottle_0: in_room dining_room_0, filled with water; exact source furniture withheld.
- hand_towel_0: in_room bathroom_2, green, clean; exact source furniture withheld.
- lamp_0: on table_2 (living_room_0), powered on.
- Destination binding: master-bedroom desk is table_11 (bedroom_2). No target-to-source-furniture association is supplied.

## Success criteria
- is_on_top(bread_0, table_11), in_room(bread_0, bedroom_2).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_0, table_11), is_clean(hand_towel_0), is_next_to(hand_towel_0, bread_0).
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).

## Spawn / planner notes
- Spawn bread ×1 on counter_0, filled bottle ×1 on table_7, clean green hand_towel ×1 on shelves_11, and powered-on lamp ×1 on table_2. Physical placements remain exactly BASE.
- Provide only room-level target localization to the robot; the exact initial placements in this specification are evaluator-only until observed. Room search must resolve the furniture.
- PowerOff uses lamp_0. Heating is unsupported; delivery needs no heated-state test. No stacking, containment, or distractors.
- Robot starts in living_room_0.
