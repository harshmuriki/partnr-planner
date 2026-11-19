# Startup memory in planner prompts

Episode folders retain the filename `initial_robot_memory.json`. Only objects
marked `in_initial_robot_memory: true` enter the initial memory prompt.

- ACC and INC: resolve selected objects by simulator handle/name in the initialized
  GT graph at episode start. Send actual furniture, room, on/inside relation and
  object states, plus available container states. Never use viewer placements as
  a fallback. A selected object missing from GT is reported absent.
- OUT: omit the separate `Initial robot memory` prompt block. The existing
  `Objects:` description retains stale locations through graph seeding, and
  TRU-POMDP's remembered placements remain in place. Objects without a stale-location
  override still use GT in the existing graph-seeding path.
- Objects marked false remain excluded from this startup memory; perception can
  subsequently discover them.

ACC's `initial_memory` list retains explicit absence statements. Runtime locations
are generated when an episode starts and are not written back to shared JSON
files. `initial_memory_source` documents the source policy for each variant.

To synchronize source metadata and ACC absence statements from the viewer specs:

```bash
python -m dataset_generation.benchmark_generation.sync_accurate_initial_memory
```

For ACC and INC, ReAct and VLM-TAMP include the snapshot in their initial scene description.
TRU-POMDP includes them in its first Level 1/2 hypothesis prompt, alongside its
separate remembered placements and sensor observations. The block describes
episode-start knowledge; subsequent action results and observations update the
robot's knowledge. It is not reinserted as a fresh observation after replanning.

Known-absent objects are statements only: they are not added as scene-graph nodes
or action targets. The object-selection flags and OUT stale placements are preserved.
Existing saved runs are not rewritten.
