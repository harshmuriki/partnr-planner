#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
DESPOT and Appendix A.2 rollout tests. These run without habitat, a GPU or a
network.
"""

from habitat_llm.planner.tru_pomdp.scene import (
    HELD,
    Action,
    ActionType,
    GoalAtom,
    Particle,
    SceneState,
    SymbolicDomain,
)
from habitat_llm.planner.tru_pomdp.search import (
    DespotConfig,
    DespotSolver,
    VNode,
    a2_next_action,
)

FURNITURE_ROOM = {
    "counter_24": "kitchen_1",
    "fridge_0": "kitchen_1",
    "cabinet_3": "kitchen_1",
    "table_10": "living_room_1",
}

DISTANCES = {
    ("cabinet_3", "counter_24"): 2.0,
    ("counter_24", "fridge_0"): 3.0,
    ("cabinet_3", "fridge_0"): 4.0,
    ("counter_24", "table_10"): 6.0,
    ("cabinet_3", "table_10"): 7.0,
    ("fridge_0", "table_10"): 8.0,
}


def make_domain(**kwargs) -> SymbolicDomain:
    params = {
        "furniture_room": dict(FURNITURE_ROOM),
        "articulated": {"fridge_0", "cabinet_3"},
        "distances": dict(DISTANCES),
    }
    params.update(kwargs)
    return SymbolicDomain(**params)


def make_scene(
    object_parent=None,
    furniture_open=None,
    inspected=None,
    robot_area=None,
    previous_parent=None,
) -> SceneState:
    return SceneState(
        object_parent=dict(object_parent or {}),
        furniture_room=dict(FURNITURE_ROOM),
        furniture_open=dict(furniture_open or {"fridge_0": False, "cabinet_3": False}),
        robot_area=robot_area,
        inspected_areas=set(inspected or ()),
        previous_parent=dict(previous_parent or {}),
    )


def fast_config(**kwargs) -> DespotConfig:
    params = {
        "num_scenarios": 6,
        "max_search_depth": 12,
        "rollout_depth": 6,
        "num_trials": 40,
        "planning_time_s": 10.0,
        "seed": 7,
    }
    params.update(kwargs)
    return DespotConfig(**params)


# ---------------------------------------------------------------------------
# Appendix A.2 rollout
# ---------------------------------------------------------------------------


def test_a2_rollout_opens_the_hypothesised_container_before_picking():
    domain = make_domain()
    atom = GoalAtom("cup_0", "counter_24")
    scene = make_scene(object_parent={"cup_0": "cabinet_3"}, robot_area="counter_24")
    assert a2_next_action([atom], scene, domain) == Action(
        ActionType.OPEN, area="cabinet_3"
    )


def test_a2_rollout_picks_once_the_container_is_open_and_inspected():
    domain = make_domain()
    atom = GoalAtom("cup_0", "counter_24")
    scene = make_scene(
        object_parent={"cup_0": "cabinet_3"},
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    assert a2_next_action([atom], scene, domain) == Action(
        ActionType.PICK, area="cabinet_3", obj="cup_0"
    )


def test_a2_rollout_explores_an_open_but_uninspected_area():
    domain = make_domain()
    atom = GoalAtom("cup_0", "table_10")
    scene = make_scene(object_parent={"cup_0": "counter_24"}, robot_area="table_10")
    # counter_24 is open but has not been inspected, so the object is not yet
    # graspable: the Habitat extension searches the room instead.
    assert a2_next_action([atom], scene, domain) == Action(
        ActionType.EXPLORE, area="kitchen_1"
    )


def test_a2_rollout_places_the_held_goal_object():
    domain = make_domain()
    atom = GoalAtom("cup_0", "table_10")
    scene = make_scene(object_parent={"cup_0": HELD}, robot_area="table_10")
    assert a2_next_action([atom], scene, domain) == Action(
        ActionType.PLACE, area="table_10", relation="on"
    )


def test_a2_rollout_opens_a_closed_goal_area_before_placing():
    domain = make_domain()
    atom = GoalAtom("cup_0", "fridge_0", relation="within")
    scene = make_scene(object_parent={"cup_0": HELD}, robot_area="fridge_0")
    assert a2_next_action([atom], scene, domain) == Action(
        ActionType.OPEN, area="fridge_0"
    )


def test_a2_rollout_uses_next_to_only_once_the_anchor_is_there():
    domain = make_domain()
    atom = GoalAtom("cup_0", "table_10", next_to="plate_2")
    without = make_scene(
        object_parent={"cup_0": HELD, "plate_2": "counter_24"}, robot_area="table_10"
    )
    assert a2_next_action([atom], without, domain).next_to is None

    with_anchor = make_scene(
        object_parent={"cup_0": HELD, "plate_2": "table_10"}, robot_area="table_10"
    )
    assert a2_next_action([atom], with_anchor, domain).next_to == "plate_2"


def test_a2_rollout_puts_a_wrong_held_object_back():
    domain = make_domain()
    atom = GoalAtom("cup_0", "table_10")
    scene = make_scene(
        object_parent={"cup_0": "table_10", "book_1": HELD},
        inspected={"table_10"},
        robot_area="table_10",
        previous_parent={"book_1": "counter_24"},
    )
    # cup_0 is already on table_10 but the goal is unsatisfied only if it is not;
    # move it off first so the goal is live.
    scene.object_parent["cup_0"] = "counter_24"
    scene.inspected_areas.add("counter_24")
    assert a2_next_action([atom], scene, domain) == Action(
        ActionType.PLACE, area="counter_24"
    )


def test_a2_rollout_returns_null_when_nothing_is_left():
    domain = make_domain()
    scene = make_scene(object_parent={"cup_0": "table_10"})
    assert a2_next_action([], scene, domain).action_type is ActionType.NULL


def test_a2_rollout_handles_a_state_only_atom():
    domain = make_domain()
    atom = GoalAtom("lamp_0", None, states=("is_powered_off",))
    scene = make_scene(
        object_parent={"lamp_0": "table_10"},
        inspected={"table_10"},
        robot_area="table_10",
    )
    scene.object_states["lamp_0"] = {"is_powered_on": True}
    assert a2_next_action([atom], scene, domain) == Action(
        ActionType.POWER_OFF, obj="lamp_0"
    )


# ---------------------------------------------------------------------------
# Observation branching and particle subsets
# ---------------------------------------------------------------------------


def hidden_cup_belief():
    """Two equally likely hypotheses for a cup hidden in one of two containers."""
    goal = (GoalAtom("cup_0", "counter_24"),)
    return [
        Particle(
            make_scene(object_parent={"cup_0": "cabinet_3"}, robot_area="counter_24"),
            goal,
            0.5,
        ),
        Particle(
            make_scene(object_parent={"cup_0": "fridge_0"}, robot_area="counter_24"),
            goal,
            0.5,
        ),
    ]


def test_observation_branching_splits_the_particle_subsets():
    domain = make_domain()
    solver = DespotSolver(domain, fast_config(num_scenarios=20))
    scenarios = solver.sample_scenarios(hidden_cup_belief())
    total = sum(s.weight for s in scenarios)
    root = VNode(
        scenarios=[s for s in scenarios if not s.terminal], weight=total, depth=0
    )
    solver._init_bounds(root)
    solver._expand(root)

    open_cabinet = Action(ActionType.OPEN, area="cabinet_3")
    assert open_cabinet in root.children
    qnode = root.children[open_cabinet]

    # Opening cabinet_3 either reveals the cup or reveals that it is empty: two
    # distinct observation keys, each keeping exactly the matching particles.
    assert len(qnode.children) == 2
    assert abs(sum(c.weight for c in qnode.children.values()) - total) < 1e-9
    for child in qnode.children.values():
        parents = {s.state.scene.object_parent["cup_0"] for s in child.scenarios}
        assert len(parents) == 1, "an observation branch mixed incompatible particles"

    # The branch where the cup was in cabinet_3 now sees it.
    revealed = [
        child
        for child in qnode.children.values()
        if any(
            domain.is_visible(s.state.scene, "cup_0") for s in child.scenarios
        )
    ]
    assert len(revealed) == 1
    assert all(
        s.state.scene.object_parent["cup_0"] == "cabinet_3"
        for s in revealed[0].scenarios
    )


def test_observation_branching_does_not_split_indistinguishable_particles():
    domain = make_domain()
    solver = DespotSolver(domain, fast_config(num_scenarios=20))
    scenarios = solver.sample_scenarios(hidden_cup_belief())
    total = sum(s.weight for s in scenarios)
    root = VNode(
        scenarios=[s for s in scenarios if not s.terminal], weight=total, depth=0
    )
    solver._init_bounds(root)
    solver._expand(root)

    # Exploring the kitchen reveals nothing about closed containers, so both
    # hypotheses stay on the same observation branch.
    explore = Action(ActionType.EXPLORE, area="kitchen_1")
    assert explore in root.children
    assert len(root.children[explore].children) == 1


# ---------------------------------------------------------------------------
# End-to-end decisions
# ---------------------------------------------------------------------------


def test_hidden_object_discovery_opens_a_hypothesised_container():
    domain = make_domain()
    solver = DespotSolver(domain, fast_config())
    action, stats = solver.plan(hidden_cup_belief())
    assert action.action_type is ActionType.OPEN
    assert action.area in ("cabinet_3", "fridge_0")
    assert stats.trials > 0
    assert stats.num_actions >= 2


def test_decision_is_conditioned_on_the_observation_open_versus_pick():
    domain = make_domain()
    goal = (GoalAtom("cup_0", "table_10"),)

    hidden = [
        Particle(
            make_scene(object_parent={"cup_0": "cabinet_3"}, robot_area="cabinet_3"),
            goal,
        )
    ]
    hidden_action, _ = DespotSolver(domain, fast_config()).plan(hidden)
    assert hidden_action == Action(ActionType.OPEN, area="cabinet_3")

    observed = [
        Particle(
            make_scene(
                object_parent={"cup_0": "cabinet_3"},
                furniture_open={"cabinet_3": True, "fridge_0": False},
                inspected={"cabinet_3"},
                robot_area="cabinet_3",
            ),
            goal,
        )
    ]
    observed_action, _ = DespotSolver(domain, fast_config()).plan(observed)
    assert observed_action == Action(ActionType.PICK, area="cabinet_3", obj="cup_0")


def test_solver_places_the_held_object_on_its_goal_area():
    domain = make_domain()
    goal = (GoalAtom("cup_0", "table_10"),)
    belief = [
        Particle(
            make_scene(object_parent={"cup_0": HELD}, robot_area="table_10"), goal
        )
    ]
    action, _ = DespotSolver(domain, fast_config()).plan(belief)
    assert action.action_type is ActionType.PLACE
    assert action.area == "table_10"


def test_solver_returns_null_for_a_satisfied_goal():
    domain = make_domain()
    goal = (GoalAtom("cup_0", "table_10"),)
    belief = [
        Particle(
            make_scene(
                object_parent={"cup_0": "table_10"},
                inspected={"table_10"},
                robot_area="table_10",
            ),
            goal,
        )
    ]
    action, _ = DespotSolver(domain, fast_config()).plan(belief)
    assert action.action_type is ActionType.NULL


def test_root_bounds_are_ordered_and_scenarios_are_weighted():
    domain = make_domain()
    solver = DespotSolver(domain, fast_config(num_scenarios=10))
    scenarios = solver.sample_scenarios(
        [
            Particle(
                make_scene(object_parent={"cup_0": "cabinet_3"}), (GoalAtom("cup_0", "counter_24"),), 0.9
            ),
            Particle(
                make_scene(object_parent={"cup_0": "fridge_0"}), (GoalAtom("cup_0", "counter_24"),), 0.1
            ),
        ]
    )
    assert len(scenarios) == 10
    assert abs(sum(s.weight for s in scenarios) - 1.0) < 1e-9
    _, stats = solver.plan(hidden_cup_belief())
    assert stats.root_upper >= stats.root_lower
