# TRU-POMDP for PARTNR

A contained Python adaptation of [Tang et al., TRU-POMDP](https://arxiv.org/html/2506.02860v2),
not a reproduction of the paper's RoboCasa benchmark. The public entry point remains
`TruPOMDPPlanner`; the shared simulator, other planners, datasets, and ground-truth records
are not changed by this baseline.

## Paper components and implementation

| Component | Implementation | Contract |
|---|---|---|
| Tree of Hypotheses (§4.1, A.1) | `toh.py` | L1 object combinations, L2 complete goals, L3 hidden locations; product-of-confidence weights |
| Hybrid update (§4.2) | `belief.py` | Predict execution, eliminate contradictions, supplement below surviving mass 0.3 |
| Online belief-tree planning (§4.3) | `search.py` | Sample scenarios, branch by action/observation, back up bounds, execute one action and replan |
| Rearrangement model (§3.2) | `scene.py` | Deterministic symbolic transitions and planning rewards |
| PARTNR adapter | `planner.py` | Sensor evidence, memory, skill execution, and traces |

The [authors' implementation](https://github.com/RoboticSJTU/tru_pomdp) and
[DESPOT reference](https://github.com/AdaCompNUS/despot/blob/master/src/solver/despot.cpp)
are comparison sources, not runtime dependencies. No C++ extension or LLM calls inside
search are required.

### Hypotheses and belief

Each hypothesis retains its **complete** goal, including already satisfied requirements.
There is no default four-object cap. `max_goal_objects`, if explicitly set, rejects oversized
combinations; it never truncates them. Invalid area IDs, malformed atoms, unsupported state
literals, and invalid confidence scores do not silently turn into easier goals. Area resolution
accepts exact IDs and case-insensitive exact matches, not token-overlap guesses.

Small hypothesis products are enumerated exactly. Products exceeding `max_particles` are
sampled directly from the joint distribution, with replacement and equal sample weights;
this bounds memory without preferentially deleting every low-probability alternative.
It is an explicit approximation. A malformed L3 response no longer fabricates uniform locations.

For surviving pre-normalization mass `w`, supplementation uses
`b_new = b_filtered + (1 - w) * b_LLM`. Generated particles are checked against evidence
before mixing. If generation yields nothing usable, valid survivors are normalized; an empty
belief stops with `empty_belief`. Contradicted particles are never restored.

Regeneration receives completed actions, execution outcomes, and observation history.
Disproving a location does not mean the task goal was wrong. The optional `wrong_goal_states`
input is reserved for explicit goal-rejection feedback; the PARTNR adapter does not invent it.

### Search and bounds

Search retains the Appendix A.2 action priorities with documented PARTNR additions. At each
rollout belief, scenario proposals are combined into one weighted-vote action, and subsequent
actions may diverge only after different observations. Averaging clairvoyant actions from
indistinguishable scenarios would not provide an executable lower-bound policy.

Values are conditional expected returns. Weighted excess uncertainty is therefore
`gamma^depth * mass * local_gap - xi * mass * root_gap`, with mass relative to the root.
`xi` is an uncertainty-reduction target, **not** a policy-size penalty. This implementation
uses zero policy-size regularization; it does not implement positive-penalty blocker pruning.
The final root choice includes the default rollout policy. Both bounds respect the configured
search horizon; rollout truncation is followed by a feasible zero-return NULL policy.

The optimistic bound allows repeated subgoal rewards after a previously satisfied goal is
disturbed. It is deliberately loose. Runtime is checked between trials, so an individual
expansion can exceed the nominal per-decision time budget. These finite budgets, particle/action
caps, and domain adaptations preclude claiming the reference solver's asymptotic guarantees.

## Memory, observation, and execution

**LLM inputs are text-only.** `TreeOfHypotheses._query` sends a string containing
the task, symbolic observations, remembered locations, and interaction history.
Robot sensor detections feed the perception adapter, but RGB frames, depth images,
and image crops are not attached to these LLM requests. A model capable of image
input does not make these particular calls multimodal.

Initial memory is a revisable prior over locations. The adapter reuses the existing episode
memory loader and furniture alias resolver, reading accurate claims from the already seeded
agent graph and explicit remembered locations from memory records. Memory-only objects are
not currently visible, and stale graph ghosts are never sensor evidence. Benchmark truth labels
such as `memory_status` and `present_in_scene` are not sent to the LLM.

Fresh observations override memory claims. The adapter captures sensor subgraphs before the
shared runner merges them into its persistent graph, including transient detections produced
by opening containers. This observer is scoped to the baseline's perception instance and is
removed on reset. The planner does not read hidden object placements or evaluation goals.
Positive observations persist under the static-world assumption and are updated by executed
skills. Observing one object does not reveal every hypothetical object on the same furniture.

Negative evidence is conservative: successful navigation or a room tour alone does not prove
complete visibility. Only certified inspection (currently successful Open under the existing
container-reveal contract) can refute an unseen object by absence. Skipped furniture during
Explore is never marked inspected. Predicted full-room coverage is replaced by actual evidence
after execution. `trust_observed_areas: true` explicitly relaxes this contract and defaults off.

**Exploration is physical and full:** the baseline overrides
`max_furniture_samples_per_room: 0`, visiting all furniture candidates instead of sampling a
subset. It has no fast-explore shortcut or approximate exploration execution. Standard skill
and navigation timeouts still apply; failure/timeout does not certify coverage.

PARTNR extensions include Explore, state-only and placement-plus-state goals, and `next_to`.
Fill requires placement at known faucet furniture. Faucet-dependent Clean uses the existing
semantic affordance metadata. The rollout and dynamic action generator can Pick, Place at the
faucet, apply the state action, and continue toward the final destination. Faucet locations come
from known furniture components, not hidden-object queries.

Only confirmed successful navigation is recorded as movement. Unrecognized/error skill responses
are failures, not successful symbolic transitions. Failed action/context pairs are excluded from
root selection until context changes; a successful Explore that provides no new information is
also prevented from repeating indefinitely in the same context. Spatial evidence comes from
observed graph relations and successful constrained Place outcomes; missing relation annotations
are not treated as measured negative evidence.

Belief completion requires **all surviving hypotheses**, not only the MAP hypothesis, to satisfy
their goals. Completion of a hypothesized goal is still distinct from benchmark success.
Termination reasons are `belief_complete`, `empty_belief`, `no_useful_action`, or
`budget_exhaustion`. Existing runner logs include weighted hypothesis snapshots, evidence,
memory revisions, regeneration diagnostics, actions, and termination reasons.

## Running and validation

Use the installed Habitat environment (the package import tree requires Habitat dependencies):

```bash
/home/harshmuriki/miniconda3/envs/habitat/bin/python -m pytest habitat_llm/tests/test_tru_pomdp_*.py -q

LD_LIBRARY_PATH=/home/harshmuriki/miniconda3/envs/habitat/lib:${LD_LIBRARY_PATH:-} \
/home/harshmuriki/miniconda3/envs/habitat/bin/python -m habitat_llm.examples.planner_demo \
  --config-name baselines/single_agent_tru_pomdp.yaml \
  habitat.dataset.data_path=baseline_evaluation_v3/episodes/task_1/t1-inc-con \
  evaluation.generate_prediviz=false
```

Model selection remains in the existing LLM configuration. Baseline defaults remain C1/C2=3,
48 particles, 30 scenarios, search depth 20, rollout depth 10, discount/xi=0.95,
500 trials/1 second per decision, 50 decisions, and the existing 600-second combined budget.
The shared LLM wrapper controls which generation settings actually reach the provider.

Tests cover exact small-POMDP comparisons, non-clairvoyant bounds, information gathering,
weighted sampling, invalid complete goals, failed supplementation, memory aliases and reset,
sensor provenance, incomplete inspection, faucet relocation, and partial goal completion.
Live validation artifacts are under `results/tru_pomdp_validation/`; the `full-explore` runs use
the requested full exploration configuration. Check their measured success and termination
reason instead of inferring benchmark success from passing unit tests.

The HTML viewer automatically recognizes TRU-POMDP traces and uses the same layout,
styles, and skill cards as the other baselines. Each card identifies its symbolic decision. It displays recorded skill outcomes separately from benchmark
success, alongside hypothesis weights, observation changes, memory, and search diagnostics.
Missing feedback is shown as unknown. Regenerate a saved page without rerunning the episode:

```bash
python -m scripts.view_trace_logs path/to/trace-episode_ID_0-0.txt
```

The planner exports the shared LLM usage tracker's model, reasoning effort, API
token counts, cached tokens, and computed token cost in `cost_metrics`, resetting
the tracker between episodes. For older traces without usage records, the viewer
recovers model and effort from that run's archived Hydra configuration. If complete
prompt-length and response logs exist, it labels a visible-text cost estimate
explicitly; hidden reasoning tokens, caching, and exact billed cost cannot be
recovered from those logs. Recorded API usage takes precedence over estimates.

The final validation run passed 123 baseline tests. T1-INC-CON achieved 100% task
completion and stopped with `belief_complete`; T2-OUT-CON achieved 0% and exhausted
50 decisions. T2 revised stale box memory and executed constrained placement skills,
but the evaluator accepted none of its placement/adjacency requirements and the
scissors remained unfound. See the saved
[validation report](../../../results/tru_pomdp_validation/validation_report.md)
for traces, configuration, and the superseded diagnostic runs. These two episodes
do not establish benchmark-wide performance.

## Remaining adaptations and limits

- Single agent; deterministic symbolic model with execution feedback, not a noisy observation model.
- Open/closed visibility is represented at furniture granularity. Exact occlusion/compartment geometry
  and proof of absence on open surfaces are not modeled. This can leave unresolved hypotheses.
- Temporary placement candidates retain the existing cap of eight; mandatory faucet preparation
  placements are generated separately and are not lost to that cap.
- Goals cover final placements, supported state literals, and one `next_to` anchor per atom.
  General temporal instructions, absence-report scoring, object-in-object containment, and arbitrary
  predicates require separate domain extensions; no privileged scoring constraints are imported.
- Memory loading follows the existing per-episode folder/sidecar convention. A combined dataset
  without those sidecars does not automatically provide per-episode memory.
- The observation-consistent rollout and conservative finite-horizon bound are implementation
  adaptations; this Python solver should not be described as an unchanged port of every DESPOT feature.
