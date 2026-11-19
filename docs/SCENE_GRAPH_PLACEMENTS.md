# Object placement in baseline scene descriptions

ReAct, VLM-TAMP, and TRU-POMDP distinguish furniture interiors from surfaces in
the scene descriptions sent to the model:

```text
plate_2: inside cabinet_3 in kitchen_1
jug_0: on counter_0 in kitchen_1
```

TRU-POMDP uses the equivalent sentences `plate_2 is inside cabinet_3` and
`jug_0 is on counter_0`. Held-object descriptions remain explicit.

The relation comes from the robot's world graph, not the furniture's name or
hidden simulator state. An object connected by an `on` edge to a receptacle of
type `within` is **inside** its parent furniture. An object on the cabinet's top
surface is **on** the cabinet. Direct `in`, `inside`, and `within` edges are
displayed as `inside`. Unknown placement is not guessed.

The shared world description and incremental object descriptions use the same
relation lookup. TRU-POMDP remembers observed placement through missing camera
frames, updates it after successful Pick/Place actions, and clears it on reset.
This adds prompt context without changing its symbolic search or visibility
rules. Existing saved prompts and historical run artifacts are unchanged; the
new descriptions apply to subsequent runs.
