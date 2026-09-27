# Tru-POMDP planner

An adaptation of **Tru-POMDP** ([Tang et al., NeurIPS 2025](https://arxiv.org/abs/2506.02860)) to
Habitat/PARTNR. The planner reasons about *uncertain* task goals and *hidden* object locations by
maintaining an explicit weighted particle belief and solving a POMDP online, instead of committing to
a single guess the way a ReAct-style LLM planner does.

The paper's three modules are preserved:

| Module | Paper | Here |
|---|---|---|
| Tree of Hypotheses | Section 4.1, prompts in Appendix A.1 | [`toh.py`](toh.py) |
| Hybrid belief update | Section 4.2 | [`belief.py`](belief.py) |
| Online POMDP planning (DESPOT + rollout) | Section 4.3, rollout in Appendix A.2 | [`search.py`](search.py) |

What changed is the *domain* and the *execution interface*, not the planning method. The paper plans
RoboCasa kitchen rearrangement with `OPEN`/`PICK`/`PLACE`; this version plans PARTNR tasks, which also
require `Explore`, object-state changes (`PowerOff`/`Fill`/`Clean`) and `next_to` spatial goals, and
which hide objects behind a per-agent `WorldGraph` rather than a fully-visible open-area model.

Reproducing the paper's RoboCasa benchmark is explicitly out of scope.

## Flow

```
instruction + WorldGraph
        |
        v
  Tree of Hypotheses (LLM)  ---- L1 target sets, L2 placement goals, L3 hidden locations
        |
        v
  weighted particle belief
        |
        v
  DESPOT + Appendix A.2 rollout        <-- no LLM inside the search tree
        |
        v
  one high-level Habitat skill (Open / Pick / Place / Explore / PowerOff / Fill / Clean)
        |
        v
  new observation --> predict, eliminate, replenish --> belief
```

The LLM is queried only by the Tree of Hypotheses: once at the start, and again when the belief
collapses. DESPOT never calls it, which is what keeps token usage low (the paper's Figure 3 result).

## Modules

### `scene.py` — symbolic domain

Sim-free model of the world that DESPOT searches over.

- `SceneState` — `object_parent` (furniture id or `HELD`), `furniture_room`, `furniture_open`,
  `object_states` (powered/filled/clean), `spatial_relations` (`next_to`), `robot_area`,
  `inspected_areas`, plus `hypothesized`/`grounding` bookkeeping and `previous_parent`.
- `GoalAtom` — one goal requirement: `obj`, `target_area` (may be `None` for a state-only goal),
  `relation`, `next_to`, `states`.
- `Particle` — `scene`, `goal_atoms` (the **complete** goal, including already-satisfied atoms), `weight`.
- `SymbolicDomain` — the POMDP itself:

```python
step(state, action)   -> (next_particle, reward, terminal)
observe(state, action) -> observation_key      # hashable, deterministic
legal_actions(belief) -> [Action, ...]         # dynamic, derived from the belief
goal_satisfied(state, goal) -> bool
```

Plain dicts and sets throughout. No PDDL and no general predicate engine.

**Actions.** `OPEN(area)`, `PICK(area, obj)`, `PLACE(area, relation, next_to)`, `NULL` from the paper,
plus the Habitat extensions `EXPLORE(room)`, `POWER_ON`/`POWER_OFF`, `FILL`, `CLEAN`. Navigation is
implicit, as in the paper, and is charged as a distance-dependent cost.

**Dynamic action space** (paper Section 4.3): the union over belief particles of `OPEN` for each known
closed container, `PICK` for each visible graspable hypothesized target, and `PLACE` into valid open
areas — *including temporary placements*, not only goal areas — plus the extension actions when the
goal has atoms that need them. `NULL` is offered only when nothing else is legal, so the search cannot
prefer idling over acting.

**Reward** (the paper's *planning* reward, Section 3.2): manipulation 5, navigation 0–27 by distance,
infeasible 100, subgoal 200, completion 200. New action costs: `Explore` 10, state skills 5. This is
the search objective and is deliberately separate from the PARTNR success predicates used for
evaluation — do not read planner reward as a benchmark score.

### `search.py` — DESPOT and the A.2 rollout

A Python implementation of DESPOT ([Ye et al., JAIR 2017](https://arxiv.org/abs/1609.03250)), not a
generic lookahead: `k` sampled scenarios, action branching, observation branching, trials guided by
upper bound then weighted excess uncertainty, and Bellman backup of both bounds along the path.

**Observation branching** is the part that makes this a belief-tree search. In `DespotSolver._expand`,
for each candidate action every scenario in the node is stepped, `domain.observe(...)` gives that
scenario's predicted observation key, and scenarios are grouped by key. Each distinct key becomes one
child belief node holding exactly the particles that produced it. Child weight is the sum of those
particle weights, and the branch probability used in backup is `child.weight / parent.weight`, which
is exact because the observation model is deterministic. So opening a cabinet that one hypothesis says
holds the mug and another does not yields two children with disjoint particle sets.

`a2_next_action` is a **transcription** of the rollout policy published in the paper's Appendix A.2 —
the C++ `NextAction` the authors generated once and adopted unmodified. Its structure, case ordering
and comments are preserved. No LLM generates rollout code here. Three blocks are additions, each
marked `Habitat extension`: `Explore` for a target in an open but uninspected area, the in-place state
skills for unmet state atoms, and `next_to` on `PLACE` once the anchor is in position.

### `toh.py` — Tree of Hypotheses

Three-level hierarchical LLM belief generation, using the Appendix A.1 prompt content with PARTNR
terminology substituted (furniture ids, rooms, skills) rather than a shortened rewrite.

- **L1 + L2** (one call): alternative target-object sets with confidences, and a placement goal per
  set. `states`/`next_to` are added only when the instruction calls for them.
- **L3** (one call per unobserved target): candidate current locations, drawn from closed containers
  and uninspected areas. A directly observed object locks its location.

Each root-to-leaf path becomes a particle whose weight is the **product** of the confidences along it,
then normalized. Hypothesized object names are kept distinct from real Habitat ids until an
observation grounds them. Regeneration context includes inspected locations and failed attempts.

### `belief.py` — hybrid belief update

Runs after each completed high-level action:

1. **Predict** with the action *and* its execution outcome — a successful navigate followed by a failed
   pick is not the same as doing nothing.
2. **Eliminate** particles inconsistent with the observation.
3. Compute surviving mass `w_BF` **before** normalization.
4. If `w_BF < 0.3`, regenerate the Tree of Hypotheses from history and mix
   `b_new = b_BF + (1 - w_BF) * b_LLM`.
5. Otherwise normalize the survivors.

### `planner.py` — PARTNR lifecycle

`TruPOMDPPlanner(Planner)`, the only module that touches the Habitat side. It converts the agent's
`WorldGraph` into an `Observation`, holds the current skill until
`Planner.process_high_level_actions` returns a non-empty response, then runs the hybrid update and
asks DESPOT for the next action. It inserts an implicit `Navigate` when the target is beyond
`navigation_threshold`.

## Observation contract

The same contract is used by the symbolic search and the Habitat adapter, so the model DESPOT plans
with matches what execution actually reveals:

- Closed containers hide their contents.
- Objects are revealed only in inspected areas.
- Knowing a furniture node exists does **not** mean its contents were inspected.
- A missing object refutes a hypothesis only after a sufficiently complete inspection of that area.
- Only observed object-state flags constrain the belief.
- Newly observed distractors are folded into particle bookkeeping; they must not wipe the belief.

This is the main departure from the paper, which assumes every open area is fully visible. In Habitat
an open surface is only known once the robot has actually looked at it, which is why `Explore` is a
planning action here rather than a free assumption.

## Configuration

[`habitat_llm/conf/planner/tru_pomdp_planner.yaml`](../../conf/planner/tru_pomdp_planner.yaml).
Defaults follow the paper's Appendix A.5 where one exists.

| Key | Default | Meaning |
|---|---|---|
| `c1`, `c2` | 3, 3 | TOH candidates at L1/L2 and L3 (paper `C1`, `C2`) |
| `max_goal_objects` | 4 | objects per goal combination |
| `max_particles` | 48 | cap on root-to-leaf paths retained |
| `toh_max_tokens` | 4096 | the A.1 reasoning chain is long |
| `replenish_threshold` | 0.3 | regenerate TOH below this surviving mass |
| `trust_observed_areas` | `False` | if `True`, an area holding an observed object may also refute |
| `num_scenarios` | 30 | DESPOT `k` |
| `max_search_depth` | 20 | DESPOT `d_s` |
| `rollout_depth` | 10 | DESPOT `d_r` |
| `discount`, `xi` | 0.95, 0.95 | discount and excess-uncertainty regularization |
| `num_trials`, `planning_time_s` | 500, 1.0 | anytime budget per decision |
| `max_decisions` | 50 | high-level actions before giving up |
| `max_place_targets` | 8 | open areas offered for a temporary placement |
| `navigation_threshold` | 1.5 m | beyond this an implicit `Navigate` is issued |

TOH temperature 0.1 is set in the baseline config.

## Running

```bash
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/single_agent_tru_pomdp.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
```

Requires `OPENAI_API_KEY` (loaded from `.env` by `habitat_llm/llm/openai_chat.py`) with access to the
model in [`conf/llm/openai_chat.yaml`](../../conf/llm/openai_chat.yaml). The baseline runs a single
agent with `gt_graph` + `partial_obs: True` and the
`oracle_rearrange_object_states_agent_motortoolsonly` tool set.

## Tests

`scene.py`, `search.py`, `toh.py` and `belief.py` must stay importable **without** habitat,
habitat_sim or a simulator, so they can be unit tested standalone. This is why
[`__init__.py`](__init__.py) deliberately does not import `planner`, and why Habitat types appear only
under `TYPE_CHECKING`. Do not break this.

```bash
python -m pytest habitat_llm/tests/test_tru_pomdp_*.py -q
```

The suite needs no simulator, GPU, API key or network, and covers hidden-object discovery,
observation-conditioned decisions, belief collapse and replenishment through a stub LLM, failed skills
that are not no-ops, and preservation of already-satisfied goal atoms.

## Documented deviations from the paper

- **DESPOT is reimplemented in Python.** The reference implementations
  ([AdaCompNUS/despot](https://github.com/AdaCompNUS/despot),
  [RoboticSJTU/tru_pomdp](https://github.com/RoboticSJTU/tru_pomdp)) require the POMDP model itself to
  be written in C++ and linked against the solver, which is not usable from this Python planner stack.
  The algorithm is unchanged; only the language is.
- **`GetObjectParent` on a held object.** The published A.2 code has a branch that returns a wrongly
  held object "to its parent", but a held object's parent is the robot node, making that branch
  unusable as written. The branch is kept verbatim and the intended area is resolved through
  `SceneState.previous_parent`.
- **Observed state flags are folded into particles rather than used to eliminate them.** Flags are a
  deterministic reading of shared state, not part of the hypothesis space (goals and hidden
  locations), so eliminating on them would discard the belief over benign prediction slips.
- **Inspection is deliberate by default.** Only opening an area, navigating to it, or completing an
  `Explore` that covers it makes it refutation-capable; `trust_observed_areas` relaxes this.
- **`wrong_goal_states` comes from belief collapse, not an oracle.** PARTNR gives the robot no signal
  that a completed goal was the wrong one, so a collapsed belief's previous MAP goal is used instead.
- **Caps on `max_particles` and `max_place_targets`** keep the Cartesian product of L3 choices and the
  placement branching tractable in houses with tens of furniture nodes. Lowest-weight particles are
  pruned and the belief renormalized.

## Smoke evaluation status

Verified live on `baseline_evaluation_v3/.archive/episodes/task_1/t1-acc-con_20260913_182645`
(scene 107734176_176000019), with `gpt-4o-mini` as the TOH model:

- The A.1 prompts parse. The model follows the reasoning structure and emits the fenced JSON block on
  every call; no parser fallbacks were needed.
- DESPOT meets its budget: mean 0.957 s, max 1.47 s per decision against `planning_time_s: 1.0`.
  The budget is checked between trials, so a single trial can overrun by roughly 50 percent. The first
  decision on a 50-furniture house managed 13 trials and 683 nodes.
- Every emitted action was accepted by the real oracle skills, including the five-field `Place`
  string. `_response_is_success` correctly classified both `Successful execution!` and
  `Unexpected failure! - ...`.
- The hybrid update, the 0.3 replenish threshold, and grounding all fire live: surviving mass fell
  0.80 → 0.38 → 0.00, TOH regenerated, and the goal moved from the hypothesized
  `jug_of_water on table_22` to the grounded `jug_0 on table_15 [is_filled]`.
- The implicit `Navigate` threshold behaved: inserted when the robot was elsewhere, skipped for
  manipulation at the current station. Nothing failed from a skipped navigation.

Note that the repo default model is `gpt-5.2`
([`conf/llm/openai_chat.yaml`](../../conf/llm/openai_chat.yaml)); the evidence above was gathered on
`gpt-4o-mini`. To run cheaply, override on the command line rather than editing that shared file,
which other baselines also use:

```
evaluation.agents.agent_0.planner.plan_config.llm.generation_params.model=gpt-4o-mini
```

## Known limitations

- **`Fill` has no faucet-proximity precondition, and a failing action can be retried until the cap.**
  `SymbolicDomain.feasible` checks only visibility and state affordance for `FILL`, so the symbolic
  model believes the action is available even when the robot is nowhere near a water source. The skill
  fails with "The object is not close enough to a water source", the failure changes nothing in the
  belief, and DESPOT re-selects it. Replenishment does not rescue this because surviving mass stays at
  1.0. Closing it requires modelling faucet locations and proximity. `max_decisions` contains the
  symptom — the episode still terminates cleanly with metrics — but any T1-style "bring a full jug"
  task will stall here.
- **`PowerOff`, the `next_to` `Place` path, and `Explore`-reveals-object are unverified in a live
  episode.** The unit tests cover them; no smoke episode produced a goal atom that exercised them.
- **`_resolve_area` silently fuzzy-matches an unrecognized area name** to the best token overlap. This
  once turned a room id into `floor_<room>`. The prompt now forbids room ids in `target_area`, and
  `verbose: True` makes it diagnosable, but a bad LLM answer is still absorbed rather than surfaced.
- Single agent only. There is no centralized/decentralized multi-agent variant.
- Spec-level "Initial robot memory" from `baseline_evaluation_v3` is not consumed anywhere at runtime,
  so ACC/INC/OUT memory variants are not yet reflected in the planner's initial belief.
- `stop` must be `""` and not `null` in the LLM generation params: `OpenAIChat.generate` calls
  `len(...)` on it, so `None` raises `TypeError` on the first query.
