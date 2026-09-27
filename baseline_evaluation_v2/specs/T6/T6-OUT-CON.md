# T6-OUT-CON

## Base task
Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights.

## Scene
- scene_id: 104348010_171512832
- rooms used: kitchen_0, garage_0, dining_room_0, bedroom_2, living_room_0
- furniture used (id — room — catalog description):
  - cabinet_0 — kitchen_0 — Kitchen cabinet with 2 doors
  - fridge_0 — garage_0 — KEW - Fridge
  - cabinet_11 — bedroom_2 — Antique Gray Accent Cabinet
  - table_7 — dining_room_0 — table
  - table_5 — bedroom_2 — Clyde side table
  - chair_9 — bedroom_2 — Etienne Upholstered Chair
  - table_11 — bedroom_2 — Vanity Desk - Antique White
  - table_2 — living_room_0 — Dark Oak

## Task instruction / prompt given
"Heat up the bread, bring it to the master-bedroom desk, bring a water bottle and a clean towel and place them next to the bread. Turn off the living-room lights."

## Affected object(s)
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, clean). All three are uncertainty targets, inside closed furniture but remembered on old surfaces.
- lamp: lamp_0, remaining task object with accurate memory. No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Outdated
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Specifies bread, one water bottle, one clean towel, adjacency at the master-bedroom desk, and living-room lights off.
- Omits sources and containment. Memory supplies stale on-surface target locations.
- Bind master bedroom to bedroom_2 and desk to table_11. Heating is requested but unsupported.

## Initial world state
- bread_0: within cabinet_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: within fridge_0 (garage_0), filled with water.
- hand_towel_0: within cabinet_11 (bedroom_2), green, clean.
- lamp_0: on table_2 (living_room_0), powered on.
- cabinet_0 (kitchen_0): Open/Close = Closed.
- fridge_0 (garage_0): Open/Close = Closed.
- cabinet_11 (bedroom_2): Open/Close = Closed.

## Final expected world state
- bread_0: on table_11 (bedroom_2); no heating requirement.
- bottle_0: on table_11 (bedroom_2), filled with water, next_to bread_0.
- hand_towel_0: on table_11 (bedroom_2), green, clean, next_to bread_0.
- lamp_0: on table_2 (living_room_0), powered off.
- cabinet_0, fridge_0, and cabinet_11 may finish open or closed; closing is not a goal.

## Initial robot memory
- bread_0: on table_7 (dining_room_0), ordinary bread; stale surface location; no heated state modeled.
- bottle_0: on table_5 (bedroom_2), filled with water; stale surface location.
- hand_towel_0: on chair_9 (bedroom_2), green, clean; stale surface location.
- lamp_0: on table_2 (living_room_0), powered on; accurate.
- cabinet_0 (kitchen_0): Open/Close = Closed.
- fridge_0 (garage_0): Open/Close = Closed.
- cabinet_11 (bedroom_2): Open/Close = Closed.
- No current target containment associations are supplied; furniture states remain accurate.

## Success criteria
- is_on_top(bread_0, table_11).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_0, table_11), is_clean(hand_towel_0), is_next_to(hand_towel_0, bread_0).
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).

## Spawn / planner notes
- Spawn bread ×1 within cabinet_0, filled bottle ×1 within fridge_0, clean green hand_towel ×1 within cabinet_11, and powered-on lamp ×1 on table_2. Initialize all three receptacles closed.
- The catalog has no bathroom cabinet: use the containing cabinet_11 in bedroom_2 as the towel-cabinet fallback. The catalog fridge is in garage_0. Do not invent receptacles or place targets within shelving.
- Stale locations are memory-only: dining table_7, bedroom table_5, and bedroom chair_9. Discover actual containment and Open before retrieval. Lamp stays at BASE.
- PowerOff uses lamp_0. Heating is unsupported; deliver bread without a heated-state check. No stacking.
- Robot starts in living_room_0.
