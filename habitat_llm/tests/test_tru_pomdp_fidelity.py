"""Paper invariants and Habitat adaptation regressions; no API or simulator required."""

import json
from types import SimpleNamespace

import pytest

from habitat_llm.planner.tru_pomdp.belief import (
    Belief,
    ExecutionOutcome,
    HybridBeliefUpdater,
    Observation,
)
from habitat_llm.planner.tru_pomdp.scene import (
    Action,
    ActionType,
    GoalAtom,
    Particle,
    SceneState,
    SymbolicDomain,
    HELD,
    NULL_ACTION,
)
from habitat_llm.planner.tru_pomdp.search import (
    DespotConfig,
    DespotSolver,
    Scenario,
    VNode,
    a2_next_action,
)
from habitat_llm.planner.tru_pomdp.toh import TreeOfHypotheses, TohConfig
from habitat_llm.tests.test_tru_pomdp_planner import make_planner, make_world_graph


class Replies:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps(next(self.answers))


def generator(objects, *, cap=48, locations=None):
    llm = Replies(
        [{"answer": [{"objects": objects, "probability": 1.0}]}]
        + [
            {"answer": [{"initial_area": a, "probability": w} for a, w in locations]}
            for _ in objects
        ]
        if locations
        else [{"answer": [{"objects": objects, "probability": 1.0}]}]
    )
    domain = SymbolicDomain(
        furniture_room={"table": "room", "cabinet": "room", "drawer": "room"}
    )
    return TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=2, max_particles=cap))


def test_complete_five_object_goal_is_not_truncated():
    objects = [{"object": f"cup_{i}", "target_area": "table"} for i in range(5)]
    toh = generator(objects)
    obs = Observation(object_parent={f"cup_{i}": "cabinet" for i in range(5)})
    belief = toh.generate("move all five cups", obs)
    assert len(belief.particles[0].goal_atoms) == 5


@pytest.mark.parametrize(
    "bad",
    [
        {"object": "cup", "target_area": "invented_table"},
        {"object": "cup", "states": ["is_magic"]},
        {"object": "cup", "target_area": "table", "relation": "under"},
    ],
)
def test_invalid_atom_rejects_entire_combination(bad):
    toh = generator([{"object": "plate", "target_area": "table"}, bad])
    assert not len(
        toh.generate("move both", Observation(object_parent={"plate": "cabinet"}))
    )
    assert toh.rejected_hypotheses == 1


def test_large_product_is_sampled_without_erasing_low_probability_branch():
    objects = [{"object": f"object{i}", "target_area": "table"} for i in range(12)]
    toh = generator(objects, cap=100, locations=[("cabinet", 0.9), ("drawer", 0.1)])
    belief = toh.generate(
        "move all", Observation(known_furniture={"table", "cabinet", "drawer"})
    )
    assert len(belief) == 100
    assert sum(p.weight for p in belief) == pytest.approx(1)
    counts = sum(list(p.scene.object_parent.values()).count("drawer") for p in belief)
    assert 70 < counts < 180
    assert all(len(p.goal_atoms) == 12 for p in belief)


def test_memory_seeds_location_without_querying_or_claiming_visibility():
    toh = generator([{"object": "cup_0", "target_area": "table"}])
    belief = toh.generate("move cup", Observation(), memory={"cup_0": "cabinet"})
    p = belief.particles[0]
    assert p.scene.object_parent["cup_0"] == "cabinet"
    assert not toh.domain.is_visible(p.scene, "cup_0")
    assert len(toh.llm.prompts) == 1
    assert "Remembered placements" in toh.llm.prompts[0]


def test_fresh_perception_overrides_remembered_location():
    toh = generator([{"object": "cup_0", "target_area": "table"}])
    belief = toh.generate(
        "move cup",
        Observation(object_parent={"cup_0": "drawer"}),
        memory={"cup_0": "cabinet"},
    )
    assert belief.particles[0].scene.object_parent["cup_0"] == "drawer"


def test_generated_particles_are_checked_before_mixing():
    domain = SymbolicDomain()
    goal = (GoalAtom("cup", "goal"),)
    contradicted = Particle(SceneState(object_parent={"cup": "empty"}), goal, 0.8)
    survivor = Particle(SceneState(object_parent={"cup": "elsewhere"}), goal, 0.2)
    toh = SimpleNamespace(generate=lambda **kw: Belief([contradicted.copy()]))
    updater = HybridBeliefUpdater(domain, toh)
    result, info = updater.update(
        Belief([contradicted, survivor]),
        ExecutionOutcome(NULL_ACTION, True),
        Observation(fully_inspected_areas={"empty"}),
    )
    assert info["surviving_mass"] == pytest.approx(0.2)
    assert info["rejected_generated"] == 1
    assert len(result) == 1 and result.particles[0].weight == 1


class TwoDoors(SymbolicDomain):
    """Hidden left/right goal, identical initial observation, informative OPEN."""

    def legal_actions(self, belief):
        return [
            Action(ActionType.OPEN),
            Action(ActionType.PICK, obj="left"),
            Action(ActionType.PICK, obj="right"),
        ]

    def goal_satisfied(self, scene, goal):
        return False

    def step(self, state, action):
        nxt = state.copy()
        if action.action_type is ActionType.OPEN:
            nxt.scene.inspected_areas.add("sensor")
            return nxt, -1.0, False
        return (
            nxt,
            (10.0 if action.obj == state.scene.object_parent["hidden"] else -10.0),
            True,
        )

    def observe(self, state, action):
        return (
            state.scene.object_parent["hidden"]
            if "sensor" in state.scene.inspected_areas
            else "unknown"
        )

    def optimistic_value(self, state, discount=0.95, horizon=20):
        return 10.0 if horizon else 0.0


def exact_value(domain, scenarios, depth, discount=0.95):
    if depth == 0:
        return 0.0
    values = [0.0]
    for action in domain.legal_actions(s.state for s in scenarios):
        total = 0.0
        groups = {}
        for scenario in scenarios:
            nxt, reward, terminal = domain.step(scenario.state, action)
            total += scenario.weight * reward
            if not terminal:
                groups.setdefault(domain.observe(nxt, action), []).append(
                    Scenario(nxt, scenario.weight)
                )
        for group in groups.values():
            mass = sum(s.weight for s in group)
            normalized = [Scenario(s.state, s.weight / mass) for s in group]
            total += (
                discount * mass * exact_value(domain, normalized, depth - 1, discount)
            )
        values.append(total)
    return max(values)


def test_search_bounds_and_information_action_match_exhaustive_pomdp():
    domain = TwoDoors()
    solver = DespotSolver(
        domain,
        DespotConfig(
            num_scenarios=2,
            max_search_depth=2,
            rollout_depth=2,
            num_trials=100,
            planning_time_s=10,
        ),
    )
    scenarios = [
        Scenario(Particle(SceneState(object_parent={"hidden": side})), 0.5)
        for side in ("left", "right")
    ]
    solver.sample_scenarios = lambda belief: scenarios
    # Proposals depend on hypothesized state, but execution must use one shared action.
    solver.rollout.next_action = lambda p: Action(
        ActionType.PICK, obj=p.scene.object_parent["hidden"]
    )
    root = VNode(scenarios, 1.0, 0)
    solver._init_bounds(root)
    optimum = exact_value(domain, scenarios, 2)
    assert root.lower <= optimum <= root.upper
    assert root.lower == 0.0  # Averaging clairvoyant actions would incorrectly give 10.
    action, stats = solver.plan([])
    assert action.action_type is ActionType.OPEN
    assert optimum == pytest.approx(8.5)
    assert stats.root_lower == pytest.approx(optimum)
    assert stats.root_lower <= optimum <= stats.root_upper


def test_weighted_excess_uncertainty_scales_both_terms():
    solver = DespotSolver(TwoDoors())
    root, child = VNode([], 1.0, 0), VNode([], 0.1, 1)
    root.lower, root.upper = 0.0, 10.0
    child.lower, child.upper = 0.0, 20.0
    assert solver._weighted_excess_uncertainty(child, root) == pytest.approx(0.95)


def test_zero_horizon_cannot_roll_out_or_expand():
    solver = DespotSolver(TwoDoors(), DespotConfig(max_search_depth=0))
    action, stats = solver.plan(
        [Particle(SceneState(object_parent={"hidden": "left"}))]
    )
    assert action == NULL_ACTION
    assert stats.root_upper == stats.root_lower == 0.0
    assert stats.num_actions == 0


def test_faucet_relocation_is_planned_for_state_only_goal():
    domain = SymbolicDomain(
        furniture_room={"table": "room", "sink": "room"}, faucet_areas={"sink"}
    )
    scene = SceneState(
        object_parent={"cup": "table"}, inspected_areas={"table", "sink"}
    )
    goal = (GoalAtom("cup", states=("is_filled",)),)
    p = Particle(scene, goal)
    actions = []
    for _ in range(3):
        action = a2_next_action(
            domain.unsatisfied_atoms(p.scene, goal), p.scene, domain
        )
        assert action in domain.legal_actions([p])
        actions.append(action.action_type)
        p, _, _ = domain.step(p, action)
    assert actions == [ActionType.PICK, ActionType.PLACE, ActionType.FILL]
    assert domain.goal_satisfied(p.scene, goal)


def test_unknown_state_and_unobserved_object_do_not_count_as_completed():
    domain = SymbolicDomain()
    goal = (GoalAtom("lamp", states=("is_powered_off",)),)
    assert not domain.goal_satisfied(SceneState(), goal)
    assert not domain.goal_satisfied(SceneState(object_parent={"lamp": "table"}), goal)


def test_rollout_picks_observed_object_without_exhaustive_area_inspection():
    domain = SymbolicDomain(furniture_room={"floor": "room", "table": "room"})
    scene = SceneState(
        object_parent={"box": "floor", "other": "floor"},
        observed_objects={"box"},
    )
    action = a2_next_action((GoalAtom("box", "table"),), scene, domain)
    assert action == Action(ActionType.PICK, obj="box", area="floor")
    assert domain.feasible(scene, action)[0]
    # Seeing the box does not license picking an unobserved object beside it.
    other = a2_next_action((GoalAtom("other", "table"),), scene, domain)
    assert other == Action(ActionType.EXPLORE, area="room")


def test_map_completion_does_not_stop_other_hypotheses():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    scene = SceneState(
        object_parent={"plate_2": "counter_24"}, observed_objects={"plate_2"}
    )
    planner.belief = Belief(
        [
            Particle(scene, (GoalAtom("plate_2", "counter_24"),), 0.8),
            Particle(scene.copy(), (GoalAtom("plate_2", "table_10"),), 0.2),
        ]
    )
    planner._plan_next_symbolic_action(graph)
    assert planner._termination_reason != "belief_complete"
    assert planner._num_decisions == 1


def test_failed_navigation_does_not_record_motion_or_inspection():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    planner.last_high_level_actions = {0: ("Navigate", "cabinet_3", None)}
    planner._symbolic_action = Action(ActionType.OPEN, area="cabinet_3")
    outcomes = []
    planner._hybrid_update = lambda success, response, graph: outcomes.append(
        planner._navigated
    )
    planner._on_skill_finished("Unexpected failure!", graph)
    assert outcomes == [False]
    assert not planner._deliberately_inspected


def test_explore_success_is_not_a_complete_room_inspection():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    planner.last_high_level_actions = {0: ("Explore", "kitchen_1", None)}
    planner._symbolic_action = Action(ActionType.EXPLORE, area="kitchen_1")
    planner._on_skill_finished("Successful execution!", graph)
    assert not planner._deliberately_inspected


def test_stale_ghost_is_never_observation_and_reset_clears_memory():
    planner = make_planner()
    graph = make_world_graph()
    graph.get_node_from_name("plate_2").sim_handle = "plate_2.stale_memory"
    planner._sync_domain(graph)
    assert "plate_2" not in planner._build_observation(graph).object_parent
    planner._memory["plate_2"] = "counter_24"
    planner._history.append("old episode")
    planner.reset()
    assert not planner._memory and not planner._history


def test_observing_one_object_does_not_reveal_all_hypothesized_contents():
    toh = generator(
        [{"object": "mug", "target_area": "table"}], locations=[("cabinet", 1.0)]
    )
    obs = Observation(object_parent={"plate": "cabinet"}, inspected_areas={"cabinet"})
    p = toh.generate("move mug", obs).particles[0]
    assert toh.domain.is_visible(p.scene, "plate")
    assert not toh.domain.is_visible(p.scene, "mug")


def test_memory_loader_preserves_aliases_without_truth_labels(tmp_path):
    graph = make_world_graph()
    graph.get_node_from_name("plate_2").sim_handle = "plate-handle"
    graph.get_node_from_name("table_10").sim_handle = "table-handle"
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "info": {
                            "variant_spec": {
                                "entity_handles": {"plate_original": "plate-handle"}
                            }
                        }
                    }
                ]
            }
        )
    )
    (tmp_path / "scene_info.json").write_text(
        json.dumps({"receptacle_to_handle": {"table_16": "table-handle"}})
    )
    (tmp_path / "initial_robot_memory.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "entity": "plate_original",
                        "in_initial_robot_memory": True,
                        "memory_status": "accurate",
                    },
                    {
                        "entity": "scissors_0",
                        "in_initial_robot_memory": True,
                        "memory_status": "outdated",
                        "outdated_location": "on table_16 (living_room_1)",
                        "present_in_scene": False,
                    },
                    {"entity": "secret", "in_initial_robot_memory": False},
                ]
            }
        )
    )
    planner = make_planner()
    planner.env_interface = SimpleNamespace(
        conf=SimpleNamespace(
            habitat=SimpleNamespace(dataset=SimpleNamespace(data_path=str(tmp_path)))
        )
    )
    planner._initialize_memory(graph)
    assert planner._memory == {"plate_2": "counter_24", "scissors_0": "table_10"}
    assert "outdated" not in json.dumps(planner._memory)
    assert "present_in_scene" not in json.dumps(planner._memory)


def test_sensor_subgraph_is_wrapped_and_excludes_unseen_memory():
    from habitat_llm.world_model import Graph

    graph = make_world_graph()
    graph.get_node_from_name("plate_2").sim_handle = "plate-handle"
    graph.get_node_from_name("lamp_0").sim_handle = "lamp-handle"
    from habitat_llm.world_model import House

    house = House("house", {"type": "root"})
    graph.add_node(house)
    for room in graph.get_all_rooms():
        graph.add_edge(room, house, "inside", "contains")
    only_plate = Graph(graph.get_subgraph(["plate_2", "agent_0"]).graph)
    calls = []

    def recent(uids, obs):
        calls.append(uids)
        return only_plate

    planner = make_planner()
    planner._sync_domain(graph)
    planner._memory_initialized = True
    planner.env_interface = SimpleNamespace(
        perception=SimpleNamespace(get_recent_subgraph=recent),
        env=SimpleNamespace(
            habitat_env=SimpleNamespace(
                sim=SimpleNamespace(get_sensor_observations=lambda: {})
            )
        ),
    )
    planner._refresh_perception()
    observation = planner._build_observation(graph)
    assert calls == [["0"]]
    assert observation.object_parent == {"plate_2": "counter_24"}


def test_collapse_does_not_label_a_goal_wrong_and_records_history():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    planner.belief = Belief(
        [
            Particle(
                SceneState(object_parent={"mug": "cabinet_3"}),
                (GoalAtom("mug", "table_10"),),
            )
        ]
    )
    captured = []

    def update(belief, outcome, observation, toh_context):
        captured.append(toh_context)
        return Belief(), {
            "surviving_mass": 0.0,
            "eliminated": 1,
            "replenished": False,
            "replenishment_failed": True,
        }

    planner.updater.update = update
    planner._symbolic_action = Action(ActionType.OPEN, area="cabinet_3")
    planner._hybrid_update(True, "Successful execution!", graph)
    assert planner._wrong_goal_states == []
    assert captured[0]["history"] and captured[0]["wrong_goal_states"] == []
    assert planner.is_done and planner._termination_reason == "empty_belief"


def test_positive_spatial_evidence_survives_missing_relation_sensor():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    planner._observed_relations["plate_2"] = {"anchor"}
    assert planner._build_observation(graph).spatial_relations["plate_2"] == {"anchor"}


def test_unknown_and_negated_success_responses_are_failures():
    assert not make_planner()._response_is_success(
        "Object is not close enough to a water source"
    )
    assert not make_planner()._response_is_success("Not successful")


def test_explore_prediction_cannot_invent_verified_runtime_coverage():
    domain = SymbolicDomain(furniture_room={"table": "room"})
    particle = Particle(
        SceneState(object_parent={"cup": "table"}), (GoalAtom("cup", "elsewhere"),)
    )
    belief, _ = HybridBeliefUpdater(domain).update(
        Belief([particle]),
        ExecutionOutcome(Action(ActionType.EXPLORE, area="room"), True),
        Observation(),
    )
    assert not belief.particles[0].scene.inspected_areas
    assert not domain.is_visible(belief.particles[0].scene, "cup")


def test_one_shot_perception_capture_survives_a_later_empty_frame_and_resets():
    from habitat_llm.world_model import WorldGraph

    planner = make_planner()
    graph = make_world_graph()
    original = lambda agent_uids, obs: obs
    perception = SimpleNamespace(get_recent_subgraph=original)
    planner._observe_perception_calls(perception)
    perception.get_recent_subgraph(["0"], graph)
    perception.get_recent_subgraph(["0"], WorldGraph())
    assert planner._observed_locations["plate_2"] == "counter_24"
    planner.reset()
    assert perception.get_recent_subgraph is original
    assert planner._observed_locations == {}


def test_root_can_choose_default_policy_over_expanded_actions():
    from habitat_llm.planner.tru_pomdp.search import QNode

    solver = DespotSolver(TwoDoors(), DespotConfig(num_trials=1))
    default = Action(ActionType.PICK, obj="left")

    def init(node):
        node.default_action = default
        node.default_value = node.lower = 5.0
        node.upper = 10.0

    def expand(node):
        qnode = QNode(node, Action(ActionType.OPEN))
        qnode.lower = qnode.upper = 1.0
        node.children[qnode.action] = qnode

    solver._init_bounds = init
    solver._expand = expand
    action, _ = solver.plan([Particle(SceneState(object_parent={"hidden": "left"}))])
    assert action == default


def test_repeated_subgoal_rewards_stay_below_upper_bound():
    domain = SymbolicDomain(furniture_room={"table": "room", "other": "room"})
    p = Particle(
        SceneState(
            object_parent={"cup": "table", "plate": "other"},
            inspected_areas={"table", "other"},
        ),
        (GoalAtom("cup", "table"), GoalAtom("plate", "table")),
    )
    bound = domain.optimistic_value(p, 0.95, 10)
    total = 0.0
    for i in range(10):
        action = (
            Action(ActionType.PICK, area="table", obj="cup")
            if i % 2 == 0
            else Action(ActionType.PLACE, area="table")
        )
        p, reward, terminal = domain.step(p, action)
        total += 0.95**i * reward
        assert not terminal
    assert total > 400.0  # The old one-time remaining-subgoals bound was unsound.
    assert total <= bound


def test_physical_exploration_is_charged_using_actual_steps():
    from habitat_llm.utils.episode_cost import combined_time_breakdown

    planner = make_planner()
    metrics = planner._planner_info({}, False)["cost_metrics"]
    assert metrics["physical_explore"] is True
    budget = combined_time_breakdown(
        physical_explore=metrics["physical_explore"],
        action_sim_steps={"Explore": 240, "Pick": 120},
        sim_freq=120.0,
    )
    assert budget["used_s"] == 3.0


def test_rollout_completes_mutual_next_to_goals_without_repeated_pick():
    domain = SymbolicDomain(furniture_room={"source": "room", "target": "room"})
    goals = (
        GoalAtom("box_0", "target", next_to="box_1"),
        GoalAtom("box_1", "target", next_to="box_0"),
        GoalAtom("scissors", "target", next_to="box_0"),
    )
    p = Particle(
        SceneState(
            object_parent={"box_0": "source", "box_1": "source", "scissors": "source"},
            inspected_areas={"source", "target"},
        ),
        goals,
    )
    actions = []
    for _ in range(6):
        action = a2_next_action(
            domain.unsatisfied_atoms(p.scene, goals), p.scene, domain
        )
        actions.append(action)
        assert domain.feasible(p.scene, action)[0]
        p, _, terminal = domain.step(p, action)
    assert terminal
    assert [a.obj for a in actions if a.action_type is ActionType.PICK] == [
        "box_0",
        "box_1",
        "scissors",
    ]


def test_api_usage_is_exported_and_resets_between_episodes():
    from habitat_llm.utils.llm_usage import TokenUsageTracker

    planner = make_planner()
    tracker = TokenUsageTracker("gpt-5.2")
    tracker.record(prompt_tokens=1000, completion_tokens=200, cached_tokens=100)
    planner.llm.token_usage = tracker
    planner.llm.generation_params = {"model": "gpt-5.2", "reasoning_effort": "high"}
    metrics = planner._planner_info({}, False)["cost_metrics"]
    assert metrics["llm_model"] == "gpt-5.2"
    assert metrics["llm_reasoning_effort"] == "high"
    assert metrics["prompt_tokens"] == 1000
    assert metrics["completion_tokens"] == 200
    assert metrics["cached_tokens"] == 100
    assert metrics["llm_usd"] > 0
    assert metrics["llm_usd_source"] == "api"
    planner.reset()
    metrics = planner._planner_info({}, False)["cost_metrics"]
    assert metrics["prompt_tokens"] == 0
    assert "llm_usd" not in metrics
