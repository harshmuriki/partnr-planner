# Faucets in the baseline_evaluation_v3 scenes

Which objects can fill a jug or wash a plate, in each of the six apartments, and what
they are in the house. Regenerate with `python3 scripts/audit_faucets.py`.

## How filling works

`Fill[jug_0]` is a compound skill: navigate to the object, then fill in place
([oracle_fill_skills.py](../habitat_llm/tools/motor_skills/object_states/oracle_fill_skills.py)).
It succeeds only when all three hold:

1. The object affords `is_filled`.
2. The agent is within **1.5 m of the object**.
3. The agent is within **1.5 m of a faucet**, measured to the nearest surface of the
   whole object carrying the marker, not to the marker point.

A faucet is any object whose asset config declares a `"faucets"` marker set; the runtime
collects them with `get_faucet_points` ([utils/sim.py](../habitat_llm/utils/sim.py)) over
both rigid and articulated objects, and marks the owning furniture with the `faucet`
component in the world graph.

**You cannot fill an object you are carrying.** The navigate sub-skill aborts with
"Failed! This object is already held by this agent"
([oracle_nav_skill.py:231](../habitat_llm/tools/motor_skills/nav/oracle_nav_skill.py#L231)).
Put the object down at the sink first. In the ground-truth runner that is:

```
Pick   jug_0
Place  jug_0,on,cabinet_6,none,none      # any receptacle of the sink cabinet
Fill   jug_0
```

The baseline planner's own few-shot example does the same thing with
`Rearrange[jug_0, on, counter_10, None, None]` followed by `Fill[jug_0]`.

## Usable faucets, by scene

A faucet is usable only if its object also keeps at least one **active receptacle**: the
world graph creates furniture nodes from the receptacle dictionary
([perception_sim.py](../habitat_llm/perception/perception_sim.py)), so an object with every
receptacle filtered out never becomes an entity, cannot be navigated to, and never appears
in the runner's apartment tree. Every usable faucet in this benchmark is an articulated
sink cabinet.

| Scene | Tasks | Entity | Room | What it is in the house | Where to put the object |
| --- | --- | --- | --- | --- | --- |
| 103997895_171031182 | T1 | `cabinet_6` | kitchen_0 | Kitchen cabinet with sink, double | `sink_top`, `sink_basin` |
| 103997895_171031182 | T1 | `cabinet_5` | laundryroom/mudroom_0 | Kitchen Cabinet with sink, 2 doors | `counter`, `sink_right`, `cabinet` (within) |
| 106878960_174887073 | T2 | — | — | **no usable faucet in this apartment** | — |
| 106878915_174887025 | T3, T5 | `cabinet_0` | kitchen_0 | "Kitchen" run with integrated sink | `countertop`, `inside_door01` (within) |
| 107734176_176000019 | T4 | `cabinet_2` | bathroom_0 | Bathroom Vanity Unit | `top`, `drawer03` (within) |
| 104348010_171512832 | T6 | `cabinet_15` | kitchen_0 | Kitchen cabinet with sink, single | `top`, `cabinet` (within) |
| 102816756 | T7 | `cabinet_5` | kitchen_0 | Kitchen cabinet with sink, double | `top` |

Only T1 and T5 specs require `is_filled`, and both of those apartments have a usable
kitchen sink. T2 has none, and no T2 spec needs one.

## Faucet objects that do not work

Every apartment also holds bathroom fixtures with faucet markers — washbasins, vanity
units, bath fillers, shower heads, freestanding baths. **All of them have no active
receptacle**, so none is an entity in the world graph: they cannot be navigated to, do not
appear in the apartment tree, and cannot be named as a skill target.

They still satisfy the 1.5 m proximity check if the agent happens to stand near one, so a
fill can succeed beside a fixture the agent reached by navigating to adjacent furniture.
That is incidental, not a plan you can express.

Counts per scene: T1 2 (double washbasin, bathtub), T2 2 (basin mixer, deck bath-shower
mixer), T3/T5 4 (two vanities, a bath mixer, a freestanding bath), T4 2 (a vanity and a
kitchen "Sink" model), T6 9 (three vanities, two bath fillers, four shower fittings),
T7 4 (three vanities, a shower).

## The specs get this wrong

[T1-ACC-BASE.md:58](specs/T1/T1-ACC-BASE.md#L58) states "This scene's faucet objects are a
double washbasin and a bathtub; no kitchen furniture carries faucet markers", and
[T5-ACC-BASE.md:81](specs/T5/T5-ACC-BASE.md#L81) states "the kitchen cabinet has no faucet
markers". Both are wrong, and in the same way: the scan behind those notes read rigid
`.object_config.json` files only and missed articulated furniture, whose markers live in
`.ao_config.json` under `data/hssd-hab/urdf/<asset>/`.

The two objects those notes point at are the ones that cannot be used at all, and the
kitchen sink cabinets they deny are the only ones that can. The ground-truth path for T1
and T5 is the kitchen sink, not a trip to the bathroom. Every spec carrying this note
(all T1 and T5 variants) needs it regenerated.

## Unnamed entities in the T1 tree

The runner's apartment tree shows three `unknown_*` entities for
103997895_171031182 — furniture with active receptacles whose category is missing from the
metadata, so perception falls back to `unknown_<node index>`. None has a faucet:

| Entity | Room | Receptacles | What it is |
| --- | --- | --- | --- |
| `unknown_22` | living_room_0 | 2 | KIT-Beacon Hall Tree, White |
| `unknown_33` | bedroom_0 | 2 | Bright & Stylish Play Kitchen (a toy kitchen — no water) |
| `unknown_35` | bedroom_0 | 7 | Angeles Changing Table with Stairs |

Two other names in that scene are worth knowing: `counter_0` (kitchen_0) is the kitchen
island, not a sink, and `cabinet_0` sits in bathroom_1 but is modelled as a "Kitchen
cabinet with hobs and oven".
