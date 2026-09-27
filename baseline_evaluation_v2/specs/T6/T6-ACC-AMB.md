# T6-ACC-AMB

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
"Bring some hot bread, some water, and a towel to me in the master-bedroom desk and turn off lights in living room."

## Affected object(s)
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, clean). All three are uncertainty targets.
- lamp: lamp_0, remaining task object. No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Accurate
- Instruction: Underspecified

## Information supplied in instruction
- Mentions hot bread, water, a towel, master-bedroom desk destination, and living-room lights off.
- Omits precise bread/water counts, the bottle as water container, towel cleanliness/color, explicit adjacency, and all source locations.
- Use the unique available bread, filled bottle, and clean towel as intended referents. The canonical evaluation retains adjacency and cleanliness; these are not explicit in this variant's prompt.
- Bind master bedroom to bedroom_2 and desk to table_11. Hot bread cannot be represented as a heated state.

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
- bread_0: on counter_0 (kitchen_0), ordinary bread; no heated state modeled.
- bottle_0: on table_7 (dining_room_0), filled with water.
- hand_towel_0: on shelves_11 (bathroom_2), green, clean.
- lamp_0: on table_2 (living_room_0), powered on.

## Success criteria
- is_on_top(bread_0, table_11).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_0, table_11), is_clean(hand_towel_0), is_next_to(hand_towel_0, bread_0).
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).

## Spawn / planner notes
- Spawn bread ×1 on counter_0, filled bottle ×1 on table_7, clean green hand_towel ×1 on shelves_11, and powered-on lamp ×1 on table_2.
- Preserve the BASE world; only the instruction is underspecified. No containment or distractors.
- Use lamp_0 for PowerOff. Heating is unsupported; deliver bread without a heat action or heated-state test. Place objects next to bread, not stacked.
- Robot starts in living_room_0.
