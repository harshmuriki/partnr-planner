# Task 6 microwave waiting check

New runs of all 15 Task 6 variants include the intermediate evaluator proposition
`is_waited_inside(bread_0, microwave_0)`. It contributes to the standard task
completion percentage and task success. BASE/CON/DIS/SUB now have 10 criteria;
ABS has 7. Existing saved runs retain their original evaluation.

The shared Wait motor skill records a successful completed action only when the
actual simulator bread remains inside the specified microwave at every skill
step, including action start and completion. The predicate uses the same
simulator containment test as `is_inside`, not the robot's scene graph. API
latency, another action, a cancelled/failed Wait, or bread inserted midway do
not count. Evidence is cleared on episode reset and persists after the bread
is removed for delivery. The bread-delivery criterion is evaluated only after
this wait criterion has been satisfied.

The existing Wait duration is unchanged (threshold `sim_freq * 5` skill steps,
currently approximately five simulated seconds). There is no added requirement
for microwave power or door state. This check records waiting; the simulator
does not expose a physical bread-temperature state.

Evaluation is shared by ReAct, VLM-TAMP, and TRU-POMDP. It scores actions the
planner actually executes; it does not add Wait to a planner's search domain or
automatically insert one.

Source specs, audited specs, per-variant JSON/gzip datasets and the aggregate
review dataset are kept consistent by the idempotent migration:

```bash
python -m dataset_generation.benchmark_generation.add_task6_wait_check
```

The explicit-spec compiler also accepts `is_waited_inside` for future generation.
