# T2-INC-CON

## Base task
Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them.

## Scene
- scene_id: 106878960_174887073
- rooms used: entryway/foyer/lobby_0, living_room_0
- furniture used (id — room — catalog description):
  - chest_of_drawers_2 — entryway/foyer/lobby_0 — Bombay Two Drawer Chest, Black
  - table_2 — living_room_0 — Madison Park Signature Bordeaux Coffee Table

## Task instruction / prompt given
"Move both packages from the entryway to the living-room center table and place them next to each other. Bring the scissors beside them."

## Affected object(s)
- box: box_0 is the selected uncertainty target, exposed but omitted from memory; box_1 is the remaining task object.
- scissors: scissors_0 is the contained uncertainty target, omitted from memory along with its containment association.
- No substitutes or distractors.

## Uncertainty being tested
- Internal robot memory: Incomplete
- Containment: Inside Closed Receptacle

## Information supplied in instruction
- Mentions two packages, entryway source, living-room center table destination, mutual adjacency, and scissors beside the packages.
- Omits scissors containment, drawer state, source furniture, exact floor positions, and instance ids.

## Initial world state
- box_0: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable and exposed.
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- scissors_0: within chest_of_drawers_2 (entryway/foyer/lobby_0), movable.
- chest_of_drawers_2: Open/Close = Closed; the containing drawer starts closed.

## Final expected world state
- box_0: on table_2 (living_room_0).
- box_1: on table_2 (living_room_0), next to box_0.
- scissors_0: on table_2 (living_room_0), next to both boxes; no longer inside chest_of_drawers_2.
- chest_of_drawers_2 may remain open after retrieval; reclosing is not required.

## Initial robot memory
- box_1: floor floor_entryway/foyer/lobby_0 (entryway/foyer/lobby_0), movable.
- chest_of_drawers_2: Open/Close = Closed.
- No selected-target records or containment/location associations are supplied; knowing the drawer is closed does not reveal its contents.

## Success criteria
- is_on_top(box_0, table_2) and is_on_top(box_1, table_2).
- is_next_to(box_0, box_1), with both on table_2.
- is_on_top(scissors_0, table_2).
- is_next_to(scissors_0, box_0) and is_next_to(scissors_0, box_1), all on table_2.
- Initial containment is is_inside(scissors_0, chest_of_drawers_2); successful delivery removes that containment.

## Spawn / planner notes
- Spawn box ×2 on the entryway floor and scissors ×1 within a valid drawer of chest_of_drawers_2. Close the containing drawer before the episode.
- The selected box remains exposed; do not place either box inside furniture.
- Start in entryway/foyer/lobby_0. Search and open the mapped entryway drawer to discover scissors; add Open to Navigate, Pick, Place. Close afterward is optional.
