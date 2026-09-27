# T3-OUT-DIS

## Base task
Collect the 2 green books & place them on the living room table and turn off the laptop as well.

## Scene
- scene_id: 106878915_174887025
- rooms used: kitchen_0, bedroom_1, living_room_0
- furniture used (id — room — catalog description):
  - table_2 — kitchen_0 — SKOGSTA Dining table
  - bed_1 — bedroom_1 — Dream bed
  - table_1 — bedroom_1 — Small African Table
  - table_0 — living_room_0 — Dip-Dyed Side Table

## Task instruction / prompt given
"Collect the 2 green books & place them on the living room table and turn off the laptop as well."

## Affected object(s)
- book_0 (book, asset Book_17): uncertainty target.
- book_1 (book, asset Book_20): uncertainty target.
- laptop_0 (laptop, asset Laptop_2): remaining task object.
- folder_0 (folder, asset B01B254OVC): distractor, placed next to book_0.
- phone_0 (phone, asset Cellphone_5): distractor, placed next to laptop_0.

## Entity registry
- book_0: book, uncertainty target
- book_1: book, uncertainty target
- laptop_0: laptop, remaining task object
- folder_0: folder, distractor
- phone_0: phone, distractor

## Uncertainty being tested
- Internal robot memory: Outdated
- Distractors: Present
- Memory in this version: robot memory places book_0, book_1, laptop_0 at stale locations.

## Information supplied in instruction
- Specified: two green books, the living-room table as the destination, and the laptop to turn off.
- Not specified: where each object currently is; the robot relies on its memory or exploration.

## Initial world state
- book_0: on bed_1 (bedroom_1), is_clean
- book_1: on table_1 (bedroom_1), is_clean
- laptop_0: on table_2 (kitchen_0), is_clean, is_powered_on
- folder_0: on bed_1 (bedroom_1), is_clean, next to book_0
- phone_0: on table_2 (kitchen_0), is_clean, is_powered_on, next to laptop_0

## Final expected world state
- book_0: on table_0 (living_room_0), is_clean
- book_1: on table_0 (living_room_0), is_clean
- laptop_0: on table_2 (kitchen_0), is_clean, is_powered_off
- folder_0: on bed_1 (bedroom_1), is_clean
- phone_0: on table_2 (kitchen_0), is_clean, is_powered_on, next to laptop_0

## Initial robot memory
- book_0: on table_2 (kitchen_0), is_clean, stale record
- book_1: on table_2 (kitchen_0), is_clean, stale record
- laptop_0: on table_1 (bedroom_1), is_clean, is_powered_on, stale record
- folder_0: on bed_1 (bedroom_1), is_clean, next to book_0
- phone_0: on table_2 (kitchen_0), is_clean, is_powered_on

## Success criteria
- is_on_top(book_0, table_0)
- is_on_top(book_1, table_0)
- is_powered_off(laptop_0)
- Distractors (folder_0, phone_0) must not be used in place of the task objects; they have no placement goals.

## Spawn / planner notes
- Scene 106878915_174887025; the robot starts in living_room_0.
- Spawn book x1 as book_0 on bed_1 (bedroom_1), pinned asset Book_17; start states: is_clean.
- Spawn book x1 as book_1 on table_1 (bedroom_1), pinned asset Book_20; start states: is_clean.
- Spawn laptop x1 as laptop_0 on table_2 (kitchen_0), pinned asset Laptop_2; start states: is_clean, is_powered_on.
- Spawn folder x1 as folder_0 on bed_1 (bedroom_1), next to book_0, pinned asset B01B254OVC; start states: is_clean.
- Spawn phone x1 as phone_0 on table_2 (kitchen_0), next to laptop_0, pinned asset Cellphone_5; start states: is_clean, is_powered_on.
- The master bedroom is bedroom_1 (the bedroom with both a bed and a table): one book on bed_1, the other on table_1. The bedroom cabinet is the TV storage unit stand_0, which has an interior receptacle.
- The dining table is table_2 in kitchen_0 (the scene has no dining room). The living-room table is table_0.
- The scene has no study, so the candidate rooms are the living room and both bedrooms.
- User decision: in every OUT variant, laptop_0 is remembered on bedroom table_1 while actually on dining table_2. ACC/INC laptop memory remains unchanged.
- The Accurate, Incomplete and Outdated versions of this variant spawn exactly the same objects, assets, placements and states; only the robot's initial memory differs.
