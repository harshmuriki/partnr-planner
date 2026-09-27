#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""Symbolic domain tests. These run without habitat, a GPU or a network."""

from habitat_llm.planner.tru_pomdp.scene import (
    COMPLETION_REWARD,
    HELD,
    INFEASIBLE_COST,
    MANIPULATION_COST,
    NAV_COST_MAX,
    SUBGOAL_REWARD,
    Action,
    ActionType,
    GoalAtom,
    Particle,
    SceneState,
    SymbolicDomain,
    names_match,
)

FURNITURE_ROOM = {
    "counter_24": "kitchen_1",
    "fridge_0": "kitchen_1",
    "cabinet_3": "kitchen_1",
    "table_10": "living_room_1",
    "couch_5": "living_room_1",
}

ARTICULATED = {"fridge_0", "cabinet_3"}

DISTANCES = {
    ("cabinet_3", "counter_24"): 2.0,
    ("counter_24", "fridge_0"): 3.0,
    ("cabinet_3", "fridge_0"): 4.0,
}


def make_domain(**kwargs) -> SymbolicDomain:
    params = {
        "furniture_room": dict(FURNITURE_ROOM),
        "articulated": set(ARTICULATED),
        "distances": dict(DISTANCES),
    }
    params.update(kwargs)
    return SymbolicDomain(**params)


def make_scene(
    object_parent=None,
    furniture_open=None,
    inspected=None,
    robot_area=None,
    object_states=None,
) -> SceneState:
    if furniture_open is None:
        furniture_open = {"fridge_0": False, "cabinet_3": False}
    return SceneState(
        object_parent=dict(object_parent or {}),
        furniture_room=dict(FURNITURE_ROOM),
        furniture_open=dict(furniture_open),
        object_states={k: dict(v) for k, v in (object_states or {}).items()},
        robot_area=robot_area,
        inspected_areas=set(inspected or ()),
    )


# ---------------------------------------------------------------------------
# Observation contract
# ---------------------------------------------------------------------------


def test_closed_container_hides_contents():
    domain = make_domain()
    scene = make_scene(
        object_parent={"cup_0": "cabinet_3"}, inspected={"cabinet_3", "counter_24"}
    )
    # cabinet_3 is closed, so its contents are hidden even though the robot has
    # inspected the node.
    assert not domain.is_visible(scene, "cup_0")

    scene.furniture_open["cabinet_3"] = True
    assert domain.is_visible(scene, "cup_0")


def test_known_furniture_is_not_an_inspected_one():
    domain = make_domain()
    # counter_24 is an open surface the robot knows about but has not looked at.
    scene = make_scene(object_parent={"cup_0": "counter_24"}, inspected=set())
    assert not domain.is_visible(scene, "cup_0")

    scene.inspected_areas.add("counter_24")
    assert domain.is_visible(scene, "cup_0")


def test_held_object_is_always_visible():
    domain = make_domain()
    scene = make_scene(object_parent={"cup_0": HELD})
    assert domain.is_visible(scene, "cup_0")


def test_observation_key_only_exposes_visible_objects():
    domain = make_domain()
    goal = (GoalAtom("cup_0", "counter_24"),)
    hidden = Particle(
        make_scene(object_parent={"cup_0": "cabinet_3"}, inspected={"cabinet_3"}), goal
    )
    elsewhere = Particle(
        make_scene(object_parent={"cup_0": "fridge_0"}, inspected={"cabinet_3"}), goal
    )
    # Both hypotheses hide the cup, so they are indistinguishable.
    action = Action(ActionType.NULL)
    assert domain.observe(hidden, action) == domain.observe(elsewhere, action)

    # Opening cabinet_3 makes them distinguishable.
    opened_hidden, _, _ = domain.step(hidden, Action(ActionType.OPEN, area="cabinet_3"))
    opened_elsewhere, _, _ = domain.step(
        elsewhere, Action(ActionType.OPEN, area="cabinet_3")
    )
    assert domain.observe(opened_hidden, action) != domain.observe(
        opened_elsewhere, action
    )


def test_explore_only_reveals_open_furniture():
    domain = make_domain()
    scene = make_scene(
        object_parent={"cup_0": "cabinet_3", "book_1": "counter_24"},
        furniture_open={"cabinet_3": False, "fridge_0": False},
    )
    particle = Particle(scene, (GoalAtom("cup_0", "counter_24"),))
    nxt, _, _ = domain.step(particle, Action(ActionType.EXPLORE, area="kitchen_1"))
    assert "counter_24" in nxt.scene.inspected_areas
    assert "cabinet_3" not in nxt.scene.inspected_areas
    assert domain.is_visible(nxt.scene, "book_1")
    assert not domain.is_visible(nxt.scene, "cup_0")


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_visibility_alone_does_not_imply_a_successful_pick():
    domain = make_domain()
    scene = make_scene(
        object_parent={"cup_0": "counter_24", "book_1": HELD},
        inspected={"counter_24"},
        robot_area="counter_24",
    )
    assert domain.is_visible(scene, "cup_0")
    # The gripper is full.
    ok, reason = domain.feasible(scene, Action(ActionType.PICK, area="counter_24", obj="cup_0"))
    assert not ok
    assert "already holding" in reason


def test_place_within_a_closed_container_is_infeasible():
    domain = make_domain()
    scene = make_scene(object_parent={"cup_0": HELD}, robot_area="counter_24")
    ok, reason = domain.feasible(
        scene, Action(ActionType.PLACE, area="fridge_0", relation="within")
    )
    assert not ok
    assert "closed" in reason


def test_next_to_requires_the_anchor_on_the_target():
    domain = make_domain()
    scene = make_scene(
        object_parent={"cup_0": HELD, "plate_2": "table_10"},
        inspected={"counter_24", "table_10"},
        robot_area="counter_24",
    )
    bad = Action(ActionType.PLACE, area="counter_24", next_to="plate_2")
    assert not domain.feasible(scene, bad)[0]

    good = Action(ActionType.PLACE, area="table_10", next_to="plate_2")
    assert domain.feasible(scene, good)[0]


def test_open_requires_an_articulated_closed_container():
    domain = make_domain()
    scene = make_scene()
    assert domain.feasible(scene, Action(ActionType.OPEN, area="cabinet_3"))[0]
    assert not domain.feasible(scene, Action(ActionType.OPEN, area="counter_24"))[0]
    scene.furniture_open["cabinet_3"] = True
    assert not domain.feasible(scene, Action(ActionType.OPEN, area="cabinet_3"))[0]


def test_state_action_requires_an_observed_object():
    domain = make_domain()
    scene = make_scene(object_parent={"lamp_0": "table_10"}, inspected=set())
    assert not domain.feasible(scene, Action(ActionType.POWER_OFF, obj="lamp_0"))[0]
    scene.inspected_areas.add("table_10")
    assert domain.feasible(scene, Action(ActionType.POWER_OFF, obj="lamp_0"))[0]


def test_declared_affordances_restrict_state_actions():
    domain = make_domain(state_affordances={"lamp_0": {"is_powered_on"}})
    scene = make_scene(object_parent={"lamp_0": "table_10"}, inspected={"table_10"})
    assert domain.feasible(scene, Action(ActionType.POWER_OFF, obj="lamp_0"))[0]
    assert not domain.feasible(scene, Action(ActionType.FILL, obj="lamp_0"))[0]
    # An object with no declared affordances is unrestricted.
    scene.object_parent["cup_0"] = "table_10"
    assert domain.feasible(scene, Action(ActionType.FILL, obj="cup_0"))[0]


# ---------------------------------------------------------------------------
# Transition and reward
# ---------------------------------------------------------------------------


def test_manipulation_and_navigation_costs():
    domain = make_domain()
    scene = make_scene(
        object_parent={"cup_0": "cabinet_3"},
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        robot_area="counter_24",
    )
    particle = Particle(scene, (GoalAtom("cup_0", "table_10"),))
    nxt, reward, terminal = domain.step(
        particle, Action(ActionType.PICK, area="cabinet_3", obj="cup_0")
    )
    assert nxt.scene.object_parent["cup_0"] == HELD
    assert nxt.scene.robot_area == "cabinet_3"
    assert not terminal
    # 2 m from counter_24 to cabinet_3 at 2.25 cost/m, plus one manipulation.
    assert reward == -(2.0 * 2.25) - MANIPULATION_COST


def test_navigation_cost_is_capped():
    domain = make_domain(distances={("counter_24", "table_10"): 100.0})
    assert domain.nav_cost("counter_24", "table_10") == NAV_COST_MAX


def test_subgoal_and_completion_reward():
    domain = make_domain()
    goal = (GoalAtom("cup_0", "table_10"), GoalAtom("book_1", "table_10"))
    scene = make_scene(
        object_parent={"cup_0": HELD, "book_1": "table_10"},
        inspected={"table_10"},
        robot_area="table_10",
    )
    particle = Particle(scene, goal)
    _, reward, terminal = domain.step(particle, Action(ActionType.PLACE, area="table_10"))
    assert terminal
    assert reward == SUBGOAL_REWARD + COMPLETION_REWARD - MANIPULATION_COST


def test_infeasible_action_is_mapped_to_null_and_penalised():
    domain = make_domain()
    scene = make_scene(object_parent={"cup_0": "cabinet_3"}, robot_area="counter_24")
    particle = Particle(scene, (GoalAtom("cup_0", "table_10"),))
    nxt, reward, _ = domain.step(
        particle, Action(ActionType.PICK, area="cabinet_3", obj="cup_0")
    )
    assert reward == -INFEASIBLE_COST
    assert nxt.scene.object_parent["cup_0"] == "cabinet_3"
    assert nxt.scene.robot_area == "counter_24"


def test_state_only_goal_has_no_placement_requirement():
    domain = make_domain()
    atom = GoalAtom.from_dict(
        {
            "object": "lamp_0",
            "target_area": None,
            "next_to": None,
            "states": ["is_powered_off"],
        }
    )
    assert atom.target_area is None
    scene = make_scene(
        object_parent={"lamp_0": "table_10"},
        inspected={"table_10"},
        robot_area="table_10",
        object_states={"lamp_0": {"is_powered_on": True}},
    )
    particle = Particle(scene, (atom,))
    assert not domain.goal_satisfied(scene, particle.goal_atoms)
    nxt, reward, terminal = domain.step(
        particle, Action(ActionType.POWER_OFF, obj="lamp_0")
    )
    assert terminal
    assert nxt.scene.object_states["lamp_0"]["is_powered_on"] is False
    assert reward == SUBGOAL_REWARD + COMPLETION_REWARD - MANIPULATION_COST


def test_next_to_placement_sets_the_spatial_relation():
    domain = make_domain()
    goal = (GoalAtom("cup_0", "table_10", next_to="plate_2"),)
    scene = make_scene(
        object_parent={"cup_0": HELD, "plate_2": "table_10"},
        inspected={"table_10"},
        robot_area="table_10",
    )
    particle = Particle(scene, goal)
    without_anchor, _, terminal = domain.step(
        particle, Action(ActionType.PLACE, area="table_10")
    )
    assert not terminal
    assert without_anchor.scene.object_parent["cup_0"] == "table_10"

    with_anchor, _, terminal = domain.step(
        particle, Action(ActionType.PLACE, area="table_10", next_to="plate_2")
    )
    assert terminal
    assert "plate_2" in with_anchor.scene.spatial_relations["cup_0"]


# ---------------------------------------------------------------------------
# Goal bookkeeping
# ---------------------------------------------------------------------------


def test_already_satisfied_atoms_stay_in_the_goal_and_stay_satisfied():
    domain = make_domain()
    satisfied = GoalAtom("book_1", "table_10")
    pending = GoalAtom("cup_0", "table_10")
    scene = make_scene(
        object_parent={"book_1": "table_10", "cup_0": HELD},
        inspected={"table_10"},
        robot_area="table_10",
    )
    particle = Particle(scene, (satisfied, pending))
    assert domain.atom_satisfied(scene, satisfied)
    assert domain.unsatisfied_atoms(scene, particle.goal_atoms) == [pending]

    nxt, reward, terminal = domain.step(
        particle, Action(ActionType.PLACE, area="table_10")
    )
    # The complete goal is preserved, including the atom that was already true.
    assert nxt.goal_atoms == (satisfied, pending)
    assert domain.atom_satisfied(nxt.scene, satisfied)
    assert domain.unsatisfied_atoms(nxt.scene, nxt.goal_atoms) == []
    assert terminal
    # Exactly one new subgoal, so no double credit for the pre-satisfied atom.
    assert reward == SUBGOAL_REWARD + COMPLETION_REWARD - MANIPULATION_COST


def test_a_disturbed_goal_atom_becomes_unsatisfied_again():
    domain = make_domain()
    atom = GoalAtom("cup_0", "table_10")
    scene = make_scene(
        object_parent={"cup_0": "table_10"},
        inspected={"table_10"},
        robot_area="table_10",
    )
    particle = Particle(scene, (atom,))
    assert domain.goal_satisfied(scene, particle.goal_atoms)
    nxt, _, _ = domain.step(
        particle, Action(ActionType.PICK, area="table_10", obj="cup_0")
    )
    assert nxt.goal_atoms == (atom,)
    assert not domain.goal_satisfied(nxt.scene, nxt.goal_atoms)


# ---------------------------------------------------------------------------
# Dynamic action space
# ---------------------------------------------------------------------------


def test_dynamic_actions_open_every_known_closed_container():
    domain = make_domain()
    goal = (GoalAtom("cup_0", "counter_24"),)
    belief = [
        Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal, 0.5),
        Particle(make_scene(object_parent={"cup_0": "fridge_0"}), goal, 0.5),
    ]
    actions = domain.legal_actions(belief)
    assert Action(ActionType.OPEN, area="cabinet_3") in actions
    assert Action(ActionType.OPEN, area="fridge_0") in actions
    # The cup is hidden in both hypotheses, so no Pick is offered.
    assert not any(a.action_type is ActionType.PICK for a in actions)
    # Exploring the room the hypothesised container is in is offered.
    assert Action(ActionType.EXPLORE, area="kitchen_1") in actions


def test_dynamic_actions_offer_temporary_placements():
    domain = make_domain(max_place_targets=8)
    goal = (GoalAtom("cup_0", "fridge_0", relation="within"),)
    scene = make_scene(
        object_parent={"cup_0": HELD},
        furniture_open={"fridge_0": False, "cabinet_3": True},
        inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    actions = domain.legal_actions([Particle(scene, goal)])
    place_areas = {a.area for a in actions if a.action_type is ActionType.PLACE}
    # Not only the goal area: any open area is a candidate temporary placement.
    assert "counter_24" in place_areas
    assert "cabinet_3" in place_areas
    assert "table_10" in place_areas


def test_null_is_only_offered_when_nothing_else_is_available():
    domain = make_domain()
    scene = make_scene(
        object_parent={"cup_0": "table_10"},
        furniture_open={},
        inspected={"table_10"},
        robot_area="table_10",
    )
    actions = domain.legal_actions([Particle(scene, (GoalAtom("cup_0", "table_10"),))])
    assert actions == [Action(ActionType.NULL)]


# ---------------------------------------------------------------------------
# Name grounding helpers
# ---------------------------------------------------------------------------


def test_names_match_ignores_indices_and_word_order():
    assert names_match("cereal_box", "box_of_cereal_12")
    assert names_match("wine glass", "wine_glass_3")
    assert not names_match("cereal_box", "hammer_1")
