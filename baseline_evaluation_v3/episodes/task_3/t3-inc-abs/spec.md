# T3-INC-ABS

## Base task
Collect the 2 green books & place them on the living room table and turn off the laptop as well.

## Scene
- scene_id: 106878915_174887025
- rooms used: kitchen_0, bedroom_1, living_room_0
- furniture used (id — room — catalog description):
  - table_2 — kitchen_0 — SKOGSTA Dining table
  - bed_1 — bedroom_1 — Dream bed
  - table_0 — living_room_0 — Dip-Dyed Side Table

## Task instruction / prompt given
"Collect the 2 green books & place them on the living room table and turn off the laptop as well."

## Affected object(s)
- book_0 (book, asset Book_17): uncertainty target.
- laptop_0 (laptop, asset Laptop_2): remaining task object.
- book_1 (book): uncertainty target, absent from the scene in this variant; no suitable substitute exists.

## Entity registry
- book_0: book, uncertainty target
- laptop_0: laptop, remaining task object
- book_1: book, uncertainty target

## Uncertainty being tested
- Internal robot memory: Incomplete
- Object availability: No Suitable Object
- Memory in this version: robot memory has no record of book_0, book_1 (no location and no existence or absence fact).

## Information supplied in instruction
- Specified: two green books, the living-room table as the destination, and the laptop to turn off.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- book_0: on bed_1 (bedroom_1), is_clean
- laptop_0: on table_2 (kitchen_0), is_clean, is_powered_on

## Final expected world state
- book_0: on table_0 (living_room_0), is_clean
- laptop_0: on table_2 (kitchen_0), is_clean, is_powered_off

## Initial robot memory
- laptop_0: on table_2 (kitchen_0), is_clean, is_powered_on
- book_0: no record; the robot does not know whether it exists or where it is
- book_1: no record; the robot does not know whether it exists or where it is

## Success criteria
- is_on_top(book_0, table_0)
- is_powered_off(laptop_0)
- The book (book_1) and any acceptable substitute are absent: the planner should report that no suitable object exists and give up on that subgoal, without inventing an object. The remaining propositions above are still expected.

## Spawn / planner notes
- Scene 106878915_174887025; the robot starts in living_room_0.
- Spawn book x1 as book_0 on bed_1 (bedroom_1), pinned asset Book_17; start states: is_clean.
- Spawn laptop x1 as laptop_0 on table_2 (kitchen_0), pinned asset Laptop_2; start states: is_clean, is_powered_on.
- Do not spawn book_1 (book) in this variant or any substitute.
- The master bedroom is bedroom_1 (the bedroom with both a bed and a table): one book on bed_1, the other on table_1. The bedroom cabinet is the TV storage unit stand_0, which has an interior receptacle.
- The dining table is table_2 in kitchen_0 (the scene has no dining room). The living-room table is table_0.
- The scene has no study, so the candidate rooms are the living room and both bedrooms.
- User decision: in every OUT variant, laptop_0 is remembered on bedroom table_1 while actually on dining table_2. ACC/INC laptop memory remains unchanged.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
