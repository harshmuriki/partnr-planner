# T6-INC-AMB

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
- bread: bread_0; bottle: bottle_0; hand_towel: hand_towel_0 (green, clean). All three are uncertainty targets omitted from memory.
- lamp: lamp_0, remaining task object retained in memory. No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Instruction: Underspecified

## Information supplied in instruction
- Mentions hot bread, water, a towel, the master-bedroom desk destination, and living-room lights off.
- Omits precise bread/water counts, the bottle container, towel cleanliness/color, explicit adjacency, and source locations. Memory also lacks the three target records.
- The intended referents are the unique available bread, filled bottle, and clean towel. Canonical evaluation retains cleanliness and adjacency, although the prompt does not explicitly state them.
- Bind master bedroom to bedroom_2 and desk to table_11. Hot bread has no supported heated state.

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
- lamp_0: on table_2 (living_room_0), powered on.
- No records for bread_0, bottle_0, or hand_towel_0; existence, state, and location facts for these targets are omitted rather than marked absent.
- Destination binding remains table_11 (bedroom_2).

## Success criteria
- is_on_top(bread_0, table_11).
- is_on_top(bottle_0, table_11), is_filled(bottle_0), is_next_to(bottle_0, bread_0).
- is_on_top(hand_towel_0, table_11), is_clean(hand_towel_0), is_next_to(hand_towel_0, bread_0).
- is_on_top(lamp_0, table_2), is_powered_off(lamp_0).

## Spawn / planner notes
- Spawn bread ×1 on counter_0, filled bottle ×1 on table_7, clean green hand_towel ×1 on shelves_11, and powered-on lamp ×1 on table_2.
- Keep BASE physical placements. The prompt is underspecified and target memory is incomplete; observations resolve available referents and locations.
- PowerOff uses lamp_0. Heating is unsupported; deliver bread without a heat action or heated-state test. Arrange items next to bread, not stacked.
- Robot starts in living_room_0. No containment or distractors.
