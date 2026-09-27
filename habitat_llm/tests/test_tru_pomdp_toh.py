#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Tree of Hypotheses tests. These run without habitat, a GPU or a network; the LLM
is a stub that replays canned responses.
"""

import ast
import json
from pathlib import Path

from habitat_llm.planner.tru_pomdp.belief import Observation
from habitat_llm.planner.tru_pomdp.scene import GoalAtom, SymbolicDomain
from habitat_llm.planner.tru_pomdp.toh import (
    TOH_LEVEL3_SYSTEM_PROMPT,
    TOH_LEVEL12_SYSTEM_PROMPT,
    TohConfig,
    TreeOfHypotheses,
    build_observation_text,
    parse_json_answer,
)

FURNITURE_ROOM = {
    "counter_24": "kitchen_1",
    "fridge_0": "kitchen_1",
    "cabinet_3": "kitchen_1",
    "table_10": "living_room_1",
}

#: The four modules that must stay importable without a simulator.
SIM_FREE_MODULES = ("scene.py", "toh.py", "belief.py", "search.py")

#: Module prefixes that would drag a simulator into those four modules.
FORBIDDEN_MODULE_PREFIXES = (
    "habitat.",
    "habitat_sim",
    "habitat_llm.world_model",
    "habitat_llm.agent",
    "habitat_llm.tools",
)


def imported_modules(source: str):
    """Every module name imported at any level of a source file."""
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class StubLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, stop=None, max_length=None):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[index]


def make_domain() -> SymbolicDomain:
    return SymbolicDomain(
        furniture_room=dict(FURNITURE_ROOM), articulated={"fridge_0", "cabinet_3"}
    )


def make_observation(**kwargs) -> Observation:
    defaults = {
        "object_parent": {"plate_2": "counter_24"},
        "furniture_open": {"fridge_0": False, "cabinet_3": False},
        "inspected_areas": {"counter_24"},
        "fully_inspected_areas": {"counter_24"},
        "robot_area": "counter_24",
        "known_furniture": set(FURNITURE_ROOM),
    }
    defaults.update(kwargs)
    return Observation(**defaults)


def fenced(payload) -> str:
    return "## Section 1\nsome reasoning\n\n```json\n" + json.dumps(payload) + "\n```\n"


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


def test_prompts_keep_the_appendix_a1_reasoning_chain():
    level12 = TOH_LEVEL12_SYSTEM_PROMPT.format(k=3, max_objects=4)
    for section in (
        "Section 1: OBJECTS OF INTEREST IDENTIFICATION",
        "Section 2: TARGET AREA IDENTIFICATION",
        "Section 3: VALIDATION LOOP",
        "Section 4: COMBINATION GENERATION",
        "Section 5: FINAL ANSWER",
        "Critical Rules (Must Read First)",
        "Object Deficit Resolution Protocol",
        "Sequential Locking Protocol",
    ):
        assert section in level12
    # Terminology substitutions: PARTNR furniture ids and skills, not RoboCasa.
    assert "counter_24" in level12
    assert "PowerOff" in level12
    assert "Human_Hand" not in level12
    # The extra PARTNR goal fields.
    assert "is_powered_off" in level12
    assert "next_to" in level12

    level3 = TOH_LEVEL3_SYSTEM_PROMPT.format(k=3)
    for step in (
        "Step 1: Object Visibility Check",
        "Step 2: Object initial_area Guess",
        "Step 3: Object Initial_Area Double Check",
        "Step 4: Final Answer",
    ):
        assert step in level3
    # Habitat adaptation: uninspected areas can hide an object too.
    assert "have not been inspected yet" in level3


def test_observation_text_carries_the_regeneration_context():
    domain = make_domain()
    text = build_observation_text(
        make_observation(),
        domain,
        wrong_goal_states=[(GoalAtom("cup_0", "table_10"),)],
        satisfied_objects=["plate_2"],
        failed_attempts=["Pick[cup_0] -> Unexpected failure"],
    )
    assert "The closed areas are: cabinet_3, fridge_0," in text
    assert "counter_24" in text
    assert "plate_2 is in counter_24" in text
    assert "The wrong goal states are:" in text
    assert "cup_0 on table_10" in text
    assert "Pick[cup_0] -> Unexpected failure" in text
    assert "Objects already in target areas: plate_2," in text
    assert "have already been inspected are: counter_24," in text
    assert "kitchen_1: cabinet_3, counter_24, fridge_0" in text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_json_answer_survives_a_long_reasoning_preamble():
    payload = {"answer": [{"initial_area": "fridge_0", "probability": 1.0}]}
    parsed = parse_json_answer(fenced(payload))
    assert parsed == payload


def test_parse_json_answer_handles_an_unfenced_object():
    payload = {"answer": [{"initial_area": "fridge_0", "probability": 1.0}]}
    parsed = parse_json_answer("thinking...\n" + json.dumps(payload))
    assert parsed == payload


def test_parse_json_answer_returns_none_on_garbage():
    assert parse_json_answer("no json at all") is None
    assert parse_json_answer("") is None


# ---------------------------------------------------------------------------
# Particle construction
# ---------------------------------------------------------------------------


def test_particle_weights_are_the_product_along_the_root_to_leaf_path():
    domain = make_domain()
    llm = StubLLM(
        [
            fenced(
                {
                    "answer": [
                        {
                            "objects": [{"object": "mug", "target_area": "table_10"}],
                            "probability": 0.8,
                        },
                        {
                            "objects": [{"object": "bowl", "target_area": "table_10"}],
                            "probability": 0.2,
                        },
                    ]
                }
            ),
            fenced(
                {
                    "answer": [
                        {"initial_area": "cabinet_3", "probability": 0.75},
                        {"initial_area": "fridge_0", "probability": 0.25},
                    ]
                }
            ),
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=2, c2=2))
    belief = toh.generate(instruction="put the mug away", observation=make_observation())

    assert len(belief) == 4
    assert abs(belief.total_weight() - 1.0) < 1e-9
    lookup = {}
    for particle in belief.particles:
        name = particle.goal_atoms[0].obj
        lookup[(name, particle.scene.object_parent[name])] = particle.weight
    # 0.8 * 0.75 and 0.8 * 0.25 for the mug, 0.2 * 0.75 and 0.2 * 0.25 for the bowl.
    assert abs(lookup[("mug", "cabinet_3")] - 0.6) < 1e-9
    assert abs(lookup[("mug", "fridge_0")] - 0.2) < 1e-9
    assert abs(lookup[("bowl", "cabinet_3")] - 0.15) < 1e-9
    assert abs(lookup[("bowl", "fridge_0")] - 0.05) < 1e-9


def test_hypothesised_names_stay_distinct_from_habitat_ids():
    domain = make_domain()
    llm = StubLLM(
        [
            fenced(
                {
                    "answer": [
                        {
                            "objects": [{"object": "mug", "target_area": "table_10"}],
                            "probability": 1.0,
                        }
                    ]
                }
            ),
            fenced({"answer": [{"initial_area": "cabinet_3", "probability": 1.0}]}),
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=1))
    belief = toh.generate(instruction="put the mug away", observation=make_observation())
    particle = belief.particles[0]
    assert particle.scene.hypothesized == {"mug"}
    assert particle.scene.object_parent["mug"] == "cabinet_3"
    # The observed object is untouched and not hypothesised.
    assert particle.scene.object_parent["plate_2"] == "counter_24"


def test_observed_target_locks_its_current_location_without_a_level_3_query():
    domain = make_domain()
    llm = StubLLM(
        [
            fenced(
                {
                    "answer": [
                        {
                            "objects": [{"object": "plate_2", "target_area": "table_10"}],
                            "probability": 1.0,
                        }
                    ]
                }
            )
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=1))
    belief = toh.generate(instruction="move the plate", observation=make_observation())
    assert len(llm.prompts) == 1, "no Level 3 query for an observed object"
    particle = belief.particles[0]
    assert particle.scene.object_parent["plate_2"] == "counter_24"
    assert not particle.scene.hypothesized


def test_level_3_candidates_are_closed_or_uninspected_areas():
    domain = make_domain()
    toh = TreeOfHypotheses(StubLLM([""]), domain, TohConfig())
    candidates = toh.candidate_hidden_areas(make_observation())
    assert "cabinet_3" in candidates and "fridge_0" in candidates
    assert "table_10" in candidates, "an uninspected open area can hide an object"
    assert "counter_24" not in candidates, "an inspected open area cannot"


def test_state_only_goal_survives_a_null_target_area():
    domain = make_domain()
    llm = StubLLM(
        [
            fenced(
                {
                    "answer": [
                        {
                            "objects": [
                                {
                                    "object": "lamp_0",
                                    "target_area": None,
                                    "states": ["is_powered_off"],
                                }
                            ],
                            "probability": 1.0,
                        }
                    ]
                }
            ),
            fenced({"answer": [{"initial_area": "table_10", "probability": 1.0}]}),
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=1))
    belief = toh.generate(instruction="turn the lamp off", observation=make_observation())
    atom = belief.particles[0].goal_atoms[0]
    assert atom.target_area is None
    assert atom.states == ("is_powered_off",)


def test_invented_furniture_is_resolved_or_dropped():
    domain = make_domain()
    llm = StubLLM(
        [
            fenced(
                {
                    "answer": [
                        {
                            "objects": [
                                {"object": "mug", "target_area": "living room table"}
                            ],
                            "probability": 1.0,
                        }
                    ]
                }
            ),
            fenced({"answer": [{"initial_area": "cabinet_3", "probability": 1.0}]}),
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=1))
    belief = toh.generate(instruction="put the mug away", observation=make_observation())
    assert belief.particles[0].goal_atoms[0].target_area == "table_10"


def test_unparseable_llm_output_yields_an_empty_belief():
    domain = make_domain()
    toh = TreeOfHypotheses(StubLLM(["I refuse."]), domain, TohConfig())
    assert len(toh.generate(instruction="do something", observation=make_observation())) == 0


def test_max_particles_prunes_the_lowest_weight_leaves():
    domain = make_domain()
    llm = StubLLM(
        [
            fenced(
                {
                    "answer": [
                        {
                            "objects": [
                                {"object": "mug", "target_area": "table_10"},
                                {"object": "bowl", "target_area": "table_10"},
                            ],
                            "probability": 1.0,
                        }
                    ]
                }
            ),
            fenced(
                {
                    "answer": [
                        {"initial_area": "cabinet_3", "probability": 0.7},
                        {"initial_area": "fridge_0", "probability": 0.3},
                    ]
                }
            ),
        ]
    )
    toh = TreeOfHypotheses(llm, domain, TohConfig(c1=1, c2=2, max_particles=2))
    belief = toh.generate(instruction="tidy up", observation=make_observation())
    assert len(belief) == 2
    assert abs(belief.total_weight() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Simulator-free import guard
# ---------------------------------------------------------------------------


def test_core_modules_do_not_import_the_simulator():
    package = Path(__file__).resolve().parents[1] / "planner" / "tru_pomdp"
    for name in SIM_FREE_MODULES:
        modules = imported_modules((package / name).read_text())
        for module in modules:
            assert module == "habitat_llm" or not module.startswith(
                FORBIDDEN_MODULE_PREFIXES
            ), f"{name} must not import {module}"


def test_planner_module_is_not_reexported_by_the_package():
    """
    Importing the package must not pull the Habitat side in through
    habitat_llm/planner/tru_pomdp/__init__.py.
    """
    package = Path(__file__).resolve().parents[1] / "planner" / "tru_pomdp"
    modules = imported_modules((package / "__init__.py").read_text())
    assert "habitat_llm.planner.tru_pomdp.planner" not in modules
