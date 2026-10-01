# Assumptions

Written in the scenario viewer. A task note (e.g. T4) applies to every variant of that task; the All tasks note applies to T1-T7.

## All tasks (T1-T7)

- In INC or OUT variants, navigate to a room first (e.g. `Navigate bedroom_0` instead of `Navigate lamp_0`).
- Close all opened pieces of furniture after placing/removing objects inside them.
- Doesn't attempt to pick up an object to know that it doesn't exist. Simply exploring a room or opening a piece of furniture to check inside let's GT know that an object doesn't exist. It only tries to pick it up if the object is seen visually.
- Whether a task is a CON variant or not, GT checks within rooms that an object is likely to be in + furniture that an object is likely to be in (if able to be opened -> otherwise the Explore skill is enough).
- A singular Explore skill can surface multiple objects (if in the same room).
- The order of tasks being complete doesn't exactly matter as long as everything gets done.
- If a visit to a piece of furniture to get an object revealed information about a different uncertainty target, it was treated as now known.

## T2 (every variant)

- Exploring the entryway/foyer/lobby_0 reveals the location of the scissors

## T5 (every variant)

- QUESTION: Should the soap start out somewhere else (so the robot actually has to go fetch it)?

## T7 (every variant)

- Although kitchen_0 is the most likely room for the plate (when location unknown), since the instructions for this task stated "put away the dirty dishes from sink," GT is able to make an educated guess and navigate directly to the sink first instead of exploring the kitchen.
