#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Hydra composition checks for the Tru-POMDP planner and baseline configs.

These need habitat installed (the baseline pulls in habitat task action config
nodes) but never start a simulator.

NOTE on import order: habitat_llm.planner must come before anything that imports
habitat, otherwise magnum's static plugins are registered twice and the
interpreter segfaults. That ordering is pre-existing in this repository.
"""

from habitat_llm.planner import Planner, TruPOMDPPlanner

import habitat_llm.agent.env.actions  # noqa: E402,F401  registers habitat/task/actions/*
from hydra import compose, initialize_config_module  # noqa: E402
from hydra.utils import get_class  # noqa: E402

#: The skills the planner emits, and the tool configs that provide them.
EXPECTED_TOOLS = {
    "oracle_nav",
    "oracle_pick",
    "oracle_place",
    "oracle_open",
    "oracle_explore",
    "oracle_power_off_in_place",
    "oracle_power_on_in_place",
    "oracle_fill_in_place",
    "oracle_clean_in_place",
    "wait",
}


def test_planner_yaml_composes_and_target_resolves():
    with initialize_config_module(config_module="habitat_llm.conf", version_base=None):
        cfg = compose(config_name="planner/tru_pomdp_planner")
    planner = cfg.planner
    assert planner._target_ == "habitat_llm.planner.TruPOMDPPlanner"
    assert get_class(planner._target_) is TruPOMDPPlanner
    assert planner._partial_ is True
    assert planner._recursive_ is False

    plan_config = planner.plan_config
    # Paper Appendix A.5 hyperparameters.
    assert plan_config.c1 == 3
    assert plan_config.c2 == 3
    assert plan_config.num_scenarios == 30
    assert plan_config.max_search_depth == 20
    assert plan_config.rollout_depth == 10
    assert plan_config.replenish_threshold == 0.3
    assert plan_config.llm.llm._target_ == "habitat_llm.llm.OpenAIChat"


def test_baseline_yaml_composes_with_the_expected_agent_and_llm():
    with initialize_config_module(config_module="habitat_llm.conf", version_base=None):
        cfg = compose(config_name="baselines/single_agent_tru_pomdp")

    planner = cfg.evaluation.agents.agent_0.planner
    assert get_class(planner._target_) is TruPOMDPPlanner

    generation = planner.plan_config.llm.generation_params
    assert generation.temperature == 0.1
    assert generation.max_tokens >= 4096
    # OpenAIChat.generate calls len() on this, so None crashes the first query.
    assert generation.stop == ""

    tools = set(cfg.evaluation.agents.agent_0.config.tools.motor_skills.keys())
    assert EXPECTED_TOOLS <= tools

    assert cfg.world_model.partial_obs is True


def test_planner_subclasses_the_base_planner_and_overrides_the_lifecycle():
    assert issubclass(TruPOMDPPlanner, Planner)
    assert TruPOMDPPlanner.get_next_action is not Planner.get_next_action
    assert TruPOMDPPlanner.reset is not Planner.reset
