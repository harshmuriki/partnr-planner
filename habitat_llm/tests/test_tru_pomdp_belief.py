#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Hybrid belief update tests. These run without habitat, a GPU or a network; the
LLM is a stub that replays canned responses.
"""

import json

from habitat_llm.planner.tru_pomdp.belief import (
    Belief,
    ExecutionOutcome,
    HybridBeliefUpdater,
    Observation,
)
from habitat_llm.planner.tru_pomdp.scene import (
    HELD,
    Action,
    ActionType,
    GoalAtom,
    Particle,
    SceneState,
    SymbolicDomain,
)
from habitat_llm.planner.tru_pomdp.toh import TohConfig, TreeOfHypotheses

FURNITURE_ROOM = {
    "counter_24": "kitchen_1",
    "fridge_0": "kitchen_1",
    "cabinet_3": "kitchen_1",
    "table_10": "living_room_1",
}


class StubLLM:
    """Replays canned responses; the last one repeats."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, stop=None, max_length=None):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]


def make_domain(**kwargs) -> SymbolicDomain:
    params = {
        "furniture_room": dict(FURNITURE_ROOM),
        "articulated": {"fridge_0", "cabinet_3"},
    }
    params.update(kwargs)
    return SymbolicDomain(**params)


def make_scene(
    object_parent=None,
    furniture_open=None,
    inspected=None,
    robot_area=None,
    hypothesized=None,
) -> SceneState:
    return SceneState(
        object_parent=dict(object_parent or {}),
        furniture_room=dict(FURNITURE_ROOM),
        furniture_open=dict(furniture_open or {"fridge_0": False, "cabinet_3": False}),
        robot_area=robot_area,
        inspected_areas=set(inspected or ()),
        hypothesized=set(hypothesized or ()),
    )


def make_observation(
    object_parent=None,
    furniture_open=None,
    inspected=None,
    fully_inspected=None,
    robot_area=None,
    object_states=None,
) -> Observation:
    return Observation(
        object_parent=dict(object_parent or {}),
        furniture_open=dict(furniture_open or {"fridge_0": False, "cabinet_3": False}),
        object_states={k: dict(v) for k, v in (object_states or {}).items()},
        robot_area=robot_area,
        inspected_areas=set(inspected or ()),
        fully_inspected_areas=set(fully_inspected or ()),
        known_furniture=set(FURNITURE_ROOM),
    )


def level12_response(objects, probability=1.0):
    payload = {"answer": [{"objects": objects, "probability": probability}]}
    return "Reasoning here.\n\n```json\n" + json.dumps(payload) + "\n```"


def level3_response(pairs):
    payload = {
        "answer": [
            {"initial_area": area, "probability": probability}
            for area, probability in pairs
        ]
    }
    return "Reasoning here.\n\n```json\n" + json.dumps(payload) + "\n```"


# ---------------------------------------------------------------------------
# Prediction: a failed skill is not a no-op
# ---------------------------------------------------------------------------


def test_failed_skill_still_moves_the_robot_but_applies_no_manipulation():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    scene = make_scene(
        object_parent={"cup_0": "cabinet_3"},
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        robot_area="counter_24",
    )
    belief = Belief([Particle(scene, (GoalAtom("cup_0", "table_10"),))])
    action = Action(ActionType.PICK, area="cabinet_3", obj="cup_0")

    failed = updater.predict(
        belief, ExecutionOutcome(action, success=False, navigated=True)
    )
    particle = failed.particles[0]
    assert particle.scene.robot_area == "cabinet_3", "navigation did happen"
    assert particle.scene.object_parent["cup_0"] == "cabinet_3", "pick did not happen"

    succeeded = updater.predict(
        belief, ExecutionOutcome(action, success=True, navigated=True)
    )
    assert succeeded.particles[0].scene.object_parent["cup_0"] == HELD


def test_failed_skill_differs_from_null():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    scene = make_scene(object_parent={"cup_0": "cabinet_3"}, robot_area="counter_24")
    belief = Belief([Particle(scene, (GoalAtom("cup_0", "table_10"),))])
    action = Action(ActionType.PICK, area="cabinet_3", obj="cup_0")

    after_failure = updater.predict(
        belief, ExecutionOutcome(action, success=False, navigated=True)
    )
    after_null = updater.predict(
        belief,
        ExecutionOutcome(Action(ActionType.NULL), success=True, navigated=False),
    )
    assert after_failure.particles[0].scene.robot_area == "cabinet_3"
    assert after_null.particles[0].scene.robot_area == "counter_24"


def test_prediction_leaves_a_particle_that_considers_the_action_infeasible():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    # This particle believes the cup is in fridge_0, so picking it from cabinet_3
    # is infeasible: the state cannot change, and elimination decides its fate.
    scene = make_scene(object_parent={"cup_0": "fridge_0"}, robot_area="counter_24")
    belief = Belief([Particle(scene, (GoalAtom("cup_0", "table_10"),))])
    predicted = updater.predict(
        belief,
        ExecutionOutcome(
            Action(ActionType.PICK, area="cabinet_3", obj="cup_0"),
            success=True,
            navigated=True,
        ),
    )
    assert predicted.particles[0].scene.object_parent["cup_0"] == "fridge_0"


# ---------------------------------------------------------------------------
# Elimination
# ---------------------------------------------------------------------------


def test_hidden_object_discovery_eliminates_the_wrong_hypothesis():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    goal = (GoalAtom("cup_0", "table_10"),)
    belief = Belief(
        [
            Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal, 0.5),
            Particle(make_scene(object_parent={"cup_0": "fridge_0"}), goal, 0.5),
        ]
    )
    # cabinet_3 was opened and inspected, and the cup is there.
    observation = make_observation(
        object_parent={"cup_0": "cabinet_3"},
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        fully_inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    new_belief, info = updater.update(
        belief,
        ExecutionOutcome(Action(ActionType.OPEN, area="cabinet_3"), success=True),
        observation,
    )
    assert info["eliminated"] == 1
    assert len(new_belief) == 1
    assert abs(new_belief.total_weight() - 1.0) < 1e-9
    assert new_belief.particles[0].scene.object_parent["cup_0"] == "cabinet_3"


def test_missing_object_only_eliminates_after_a_complete_inspection():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    goal = (GoalAtom("cup_0", "table_10"),)
    belief = Belief([Particle(make_scene(object_parent={"cup_0": "counter_24"}), goal)])

    # Glimpsed only: absence is not evidence.
    partial = make_observation(
        inspected={"counter_24"}, fully_inspected=set(), robot_area="counter_24"
    )
    _, mass, eliminated = updater.eliminate(belief.copy(), partial)
    assert not eliminated
    assert abs(mass - 1.0) < 1e-9

    # Inspected completely: the hypothesis is refuted.
    complete = make_observation(
        inspected={"counter_24"},
        fully_inspected={"counter_24"},
        robot_area="counter_24",
    )
    _, mass, eliminated = updater.eliminate(belief.copy(), complete)
    assert len(eliminated) == 1
    assert mass == 0.0


def test_new_distractors_do_not_wipe_the_belief():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    goal = (GoalAtom("cup_0", "fridge_0"),)
    belief = Belief([Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal)])
    observation = make_observation(
        object_parent={"cup_0": "cabinet_3", "hammer_9": "cabinet_3"},
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        fully_inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    survivors, mass, eliminated = updater.eliminate(belief, observation)
    assert not eliminated
    assert mass == 1.0
    # The distractor is folded into the particle's bookkeeping.
    assert survivors.particles[0].scene.object_parent["hammer_9"] == "cabinet_3"


def test_observed_state_flags_are_folded_in_not_used_to_eliminate():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    atom = GoalAtom("lamp_0", None, states=("is_powered_off",))
    scene = make_scene(object_parent={"lamp_0": "table_10"}, inspected={"table_10"})
    scene.object_states["lamp_0"] = {"is_powered_on": False}
    belief = Belief([Particle(scene, (atom,))])
    observation = make_observation(
        object_parent={"lamp_0": "table_10"},
        inspected={"table_10"},
        fully_inspected={"table_10"},
        robot_area="table_10",
        object_states={"lamp_0": {"is_powered_on": True}},
    )
    survivors, mass, eliminated = updater.eliminate(belief, observation)
    assert not eliminated
    assert mass == 1.0
    assert survivors.particles[0].scene.object_states["lamp_0"]["is_powered_on"] is True


def test_hypothesised_name_is_grounded_onto_the_observed_object_id():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    goal = (GoalAtom("cereal_box", "table_10"),)
    scene = make_scene(
        object_parent={"cereal_box": "cabinet_3"}, hypothesized={"cereal_box"}
    )
    belief = Belief([Particle(scene, goal)])
    observation = make_observation(
        object_parent={"box_of_cereal_7": "cabinet_3"},
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        fully_inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    survivors, mass, eliminated = updater.eliminate(belief, observation)
    assert not eliminated, "grounding should keep the hypothesis alive"
    assert mass == 1.0
    particle = survivors.particles[0]
    assert "cereal_box" not in particle.scene.object_parent
    assert particle.scene.object_parent["box_of_cereal_7"] == "cabinet_3"
    assert particle.goal_atoms[0].obj == "box_of_cereal_7"
    assert particle.scene.grounding["cereal_box"] == "box_of_cereal_7"


def test_satisfied_goal_atoms_survive_the_update_and_stay_satisfied():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain)
    done = GoalAtom("book_1", "table_10")
    pending = GoalAtom("cup_0", "table_10")
    scene = make_scene(
        object_parent={"book_1": "table_10", "cup_0": HELD},
        inspected={"table_10"},
        robot_area="table_10",
    )
    belief = Belief([Particle(scene, (done, pending))])
    observation = make_observation(
        object_parent={"book_1": "table_10", "cup_0": "table_10"},
        inspected={"table_10"},
        fully_inspected={"table_10"},
        robot_area="table_10",
    )
    new_belief, info = updater.update(
        belief,
        ExecutionOutcome(Action(ActionType.PLACE, area="table_10"), success=True),
        observation,
    )
    assert info["eliminated"] == 0
    particle = new_belief.particles[0]
    assert particle.goal_atoms == (done, pending)
    assert domain.atom_satisfied(particle.scene, done)
    assert domain.goal_satisfied(particle.scene, particle.goal_atoms)


# ---------------------------------------------------------------------------
# Replenishment
# ---------------------------------------------------------------------------


def test_belief_collapse_triggers_toh_replenishment():
    domain = make_domain()
    llm = StubLLM(
        [
            level12_response([{"object": "mug", "target_area": "table_10"}]),
            level3_response([("fridge_0", 1.0)]),
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=1))
    updater = HybridBeliefUpdater(domain, toh, replenish_threshold=0.3)

    goal = (GoalAtom("cup_0", "table_10"),)
    belief = Belief(
        [
            Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal, 0.5),
            Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal, 0.5),
        ]
    )
    # cabinet_3 turns out to be empty, so the whole belief is refuted.
    observation = make_observation(
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        fully_inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    new_belief, info = updater.update(
        belief,
        ExecutionOutcome(Action(ActionType.OPEN, area="cabinet_3"), success=True),
        observation,
        toh_context={"instruction": "tidy up the mug"},
    )
    assert info["surviving_mass"] == 0.0
    assert info["replenished"] is True
    assert len(new_belief) == 1
    assert abs(new_belief.total_weight() - 1.0) < 1e-9
    particle = new_belief.particles[0]
    assert particle.scene.object_parent["mug"] == "fridge_0"
    assert "mug" in particle.scene.hypothesized
    assert llm.prompts, "the TOH prompts were not sent"


def test_partial_collapse_mixes_the_filtered_and_llm_beliefs():
    domain = make_domain()
    llm = StubLLM(
        [
            level12_response([{"object": "mug", "target_area": "table_10"}]),
            level3_response([("fridge_0", 1.0)]),
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=1))
    updater = HybridBeliefUpdater(domain, toh, replenish_threshold=0.3)

    goal = (GoalAtom("cup_0", "table_10"),)
    belief = Belief(
        [
            # Survives: the cup really is in fridge_0.
            Particle(make_scene(object_parent={"cup_0": "fridge_0"}), goal, 0.2),
            # Refuted: cabinet_3 was inspected and is empty.
            Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal, 0.8),
        ]
    )
    observation = make_observation(
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        fully_inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    new_belief, info = updater.update(
        belief,
        ExecutionOutcome(Action(ActionType.OPEN, area="cabinet_3"), success=True),
        observation,
        toh_context={"instruction": "tidy up the mug"},
    )
    assert abs(info["surviving_mass"] - 0.2) < 1e-9
    assert info["replenished"] is True
    assert len(new_belief) == 2
    assert abs(new_belief.total_weight() - 1.0) < 1e-9
    weights = {
        tuple(sorted(p.scene.object_parent)): p.weight for p in new_belief.particles
    }
    # b_new = b_BF + (1 - w_BF) * b_LLM, so the survivor keeps mass 0.2.
    assert any(abs(w - 0.2) < 1e-9 for w in weights.values())
    assert any(abs(w - 0.8) < 1e-9 for w in weights.values())


def test_healthy_belief_is_only_normalised():
    domain = make_domain()
    llm = StubLLM([level12_response([{"object": "mug", "target_area": "table_10"}])])
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=1))
    updater = HybridBeliefUpdater(domain, toh, replenish_threshold=0.3)

    goal = (GoalAtom("cup_0", "table_10"),)
    belief = Belief(
        [
            Particle(make_scene(object_parent={"cup_0": "fridge_0"}), goal, 0.6),
            Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal, 0.4),
        ]
    )
    observation = make_observation(
        furniture_open={"cabinet_3": False, "fridge_0": False},
        robot_area="counter_24",
    )
    new_belief, info = updater.update(
        belief,
        ExecutionOutcome(Action(ActionType.NULL), success=True),
        observation,
    )
    assert info["replenished"] is False
    assert len(new_belief) == 2
    assert not llm.prompts, "no LLM call should happen while the belief is healthy"


def test_collapse_without_a_toh_reports_empty_belief():
    domain = make_domain()
    updater = HybridBeliefUpdater(domain, toh=None, replenish_threshold=0.3)
    goal = (GoalAtom("cup_0", "table_10"),)
    belief = Belief([Particle(make_scene(object_parent={"cup_0": "cabinet_3"}), goal)])
    observation = make_observation(
        furniture_open={"cabinet_3": True, "fridge_0": False},
        inspected={"cabinet_3"},
        fully_inspected={"cabinet_3"},
        robot_area="cabinet_3",
    )
    new_belief, info = updater.update(
        belief,
        ExecutionOutcome(Action(ActionType.OPEN, area="cabinet_3"), success=True),
        observation,
    )
    assert info["surviving_mass"] == 0.0
    assert info["termination_reason"] == "empty_belief"
    assert len(new_belief) == 0
