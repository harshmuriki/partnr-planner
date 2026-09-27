# T2-OUT-ABS

## Base task
Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them.

## Scene
- scene_id: 106878960_174887073
- rooms used: entryway/foyer/lobby_0, living_room_0, dining_room_0
- furniture used (id — room — catalog description):
  - chest_of_drawers_2 — entryway/foyer/lobby_0 — Bombay Two Drawer Chest, Black
  - table_2 — living_room_0 — Madison Park Signature Bordeaux Coffee Table
  - table_0 — dining_room_0 — Livingston Dining Table 62"

## Task instruction / prompt given
"Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them."

## Affected object(s)
- box: box_0 is the selected uncertainty target with stale memory; box_1 is the remaining task object with accurate memory. Both physically exist.
- scissors: scissors_0 is the absent requested uncertainty target, represented only by a false memory record.
- No knife substitute, other acceptable cutting tool, or distractors exist.

## Uncertainty being tested
- Internal robot memory: Outdated
- Object availability: No Suitable Object

## Information supplied in instruction
- Mentions two packages, entryway source, living-room center table destination, mutual adjacency, and scissors beside the packages.
- Omits tool unavailability, source furniture, exact floor positions, instance ids, and states. Stale memory falsely suggests that the tool subgoal is feasible.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0 is absent; no suitable cutting tool exists, including on chest_of_drawers_2 or table_0.

## Final expected world state
- box_0: on table_2 (living_room_0).
- box_1: on table_2 (living_room_0), next to box_0.
- No cutting tool is fabricated or delivered. The tool-delivery subgoal is declared infeasible.

## Initial robot memory
- box_0: on table_2 (living_room_0), movable; stale location.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: on table_0 (dining_room_0), movable; false existence and stale location.

## Success criteria
- Verify and reject the false tool record, then give up on the unavailable tool subgoal; do not invent tools or claim complete fulfillment.
- Feasible package subgoals: is_on_top(box_0, table_2) and is_on_top(box_1, table_2), checked against actual placements.
- Feasible arrangement: is_next_to(box_0, box_1), with both on table_2.
- No scissors or substitute placement proposition is required.

## Spawn / planner notes
- Spawn only box ×2 on the entryway floor. Spawn no scissors, knife, other acceptable cutting tool, or stale-location duplicates.
- chest_of_drawers_2's top is the mapped entryway-table search location. table_2 is the center table and stale box support.
- No dining bench is cataloged; use dining-room table_0 as the closest same-room support for the stale scissors record.
- Start in entryway/foyer/lobby_0. Complete feasible package moves, invalidate stale availability, and terminate the tool search rather than repeatedly attempting phantom pickups.
