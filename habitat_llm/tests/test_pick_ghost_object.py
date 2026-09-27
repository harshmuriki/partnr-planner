#!/usr/bin/env python3

from types import SimpleNamespace

import pytest
import torch

import habitat_llm.tools.motor_skills.nav.oracle_nav_skill as nav_module
import habitat_llm.tools.motor_skills.pick.oracle_pick_skill as pick_module
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils import sim as sim_utils
from habitat_llm.world_model.entities.furniture import Furniture
from habitat_llm.world_model.entity import Object


class _RaisingGraph:
    def __init__(self):
        self.removed = []

    def get_node_from_sim_handle(self, _handle):
        raise ValueError("Node with sim_handle missing not present in the graph.")

    def remove_object_from_graph(self, name):
        self.removed.append(name)


class _StaleObjectGraph:
    def __init__(self, target, furniture):
        self.target = target
        self.furniture = furniture
        self.removed = []

    def get_node_from_name(self, name):
        if name == self.target.name:
            return self.target
        if name == self.furniture.name:
            return self.furniture
        raise ValueError(name)

    def find_furniture_for_object(self, target):
        assert target is self.target
        return self.furniture

    def remove_object_from_graph(self, name):
        self.removed.append(name)


def _dummy_action():
    return torch.zeros((1, 8))


def test_oracle_nav_redirects_stale_object_to_parent_furniture(monkeypatch):
    stale_object = Object(
        "cup_1",
        {"translation": [1.0, 0.0, 1.0]},
        sim_handle="cup_1.stale_memory",
    )
    furniture = Furniture(
        "table_0",
        {"translation": [4.0, 0.0, 5.0]},
        sim_handle="table_handle",
    )
    graph = _StaleObjectGraph(stale_object, furniture)
    env = SimpleNamespace(world_graph={0: graph}, sim=SimpleNamespace())
    skill = nav_module.OracleNavSkill.__new__(nav_module.OracleNavSkill)
    skill.env = env
    skill.agent_uid = 0
    skill.target_is_set = False
    skill.failed = False
    skill.get_agent_object_ids = lambda: ([], [])
    redirected_targets = []
    skill.set_target = lambda name, redirected_env: redirected_targets.append(
        (name, redirected_env)
    )
    monkeypatch.setattr(nav_module, "get_obj_from_handle", lambda _sim, _handle: None)

    nav_module.OracleNavSkill.set_target(skill, stale_object.name, env)

    assert redirected_targets == [(furniture.name, env)]
    assert graph.removed == []


def test_base_skill_target_validation_does_not_remove_object():
    target = Object("cup_1", {"translation": [1.0, 0.0, 1.0]})
    furniture = Furniture(
        "table_0",
        {"translation": [4.0, 0.0, 5.0]},
        sim_handle="table_handle",
    )
    graph = _StaleObjectGraph(target, furniture)
    skill = SimpleNamespace(
        env=SimpleNamespace(world_graph={0: graph}),
        agent_uid=0,
        target_is_set=False,
    )

    with pytest.raises(ValueError, match="does not have a simulator handle"):
        SkillPolicy.set_target(skill, target.name, skill.env)

    assert graph.removed == []


def test_check_if_object_is_moveable_ghost_object():
    env = SimpleNamespace(full_world_graph=_RaisingGraph())
    action = _dummy_action()
    out_action, message, failed = sim_utils.check_if_the_object_is_moveable(
        env, action, "ghost_handle.object_config.json"
    )
    assert failed is True
    assert message == "Failed to pick! Object does not exist."
    assert torch.equal(out_action, torch.zeros_like(action))


def test_check_if_object_is_inside_furniture_ghost_object():
    env = SimpleNamespace(full_world_graph=_RaisingGraph())
    action = _dummy_action()
    out_action, message, failed = sim_utils.check_if_the_object_is_inside_furniture(
        env, action, "ghost_handle.object_config.json", threshold_for_ao_state=0.4
    )
    assert failed is True
    assert message == "Failed to pick! Object does not exist."
    assert torch.equal(out_action, torch.zeros_like(action))


def test_check_if_gripper_is_full_ghost_object():
    grasp_mgr = SimpleNamespace(is_grasped=True, snap_idx=1)
    rom = SimpleNamespace(get_object_handle_by_id=lambda _idx: "grasped_handle")
    env = SimpleNamespace(
        full_world_graph=_RaisingGraph(),
        sim=SimpleNamespace(get_rigid_object_manager=lambda: rom),
    )
    action = _dummy_action()
    out_action, message, failed = sim_utils.check_if_gripper_is_full(
        env, action, grasp_mgr, "ghost_handle.object_config.json"
    )
    assert failed is True
    assert message == "Failed to pick! Object does not exist."
    assert torch.equal(out_action, torch.zeros_like(action))


def test_check_if_object_is_held_by_agent_ghost_object(monkeypatch):
    monkeypatch.setattr(sim_utils, "get_obj_from_handle", lambda _sim, _handle: None)
    env = SimpleNamespace(
        sim=SimpleNamespace(agents_mgr=SimpleNamespace(agent_names=[])),
        world_graph={0: _RaisingGraph()},
    )
    action = _dummy_action()
    out_action, message, failed = sim_utils.check_if_the_object_is_held_by_agent(
        env, action, "ghost_handle.object_config.json", this_agent_uid=0
    )
    assert failed is True
    assert message == "Failed to pick! Object does not exist."
    assert torch.equal(out_action, torch.zeros_like(action))


def test_oracle_pick_removes_ghost_target_after_precheck_failure(monkeypatch):
    spy_graph = _RaisingGraph()
    skill = pick_module.OraclePickSkill.__new__(pick_module.OraclePickSkill)
    skill.env = SimpleNamespace(world_graph={0: spy_graph})
    skill.agent_uid = 0
    skill.failed = False
    skill.steps = 0
    skill.target_handle = "ghost_handle.object_config.json"
    skill._target_name = "ghost_object"
    skill.grasp_mgr = SimpleNamespace()
    skill.thresh_for_art_state = 0.4

    action = _dummy_action()
    monkeypatch.setattr(
        pick_module,
        "check_if_gripper_is_full",
        lambda env, action, grasp_mgr, target_handle: (
            torch.zeros_like(action),
            sim_utils.GHOST_OBJECT_PICK_FAILURE,
            True,
        ),
    )

    pick_module.OraclePickSkill._internal_act(
        skill,
        observations={},
        rnn_hidden_states=None,
        prev_actions=action,
        masks=torch.ones((1, 1)),
        cur_batch_idx=0,
    )

    assert spy_graph.removed == ["ghost_object"]
    assert skill.termination_message == sim_utils.GHOST_OBJECT_PICK_FAILURE
    assert skill.failed is True


def test_oracle_pick_removes_ghost_target_after_missing_sim_object(monkeypatch):
    spy_graph = _RaisingGraph()
    skill = pick_module.OraclePickSkill.__new__(pick_module.OraclePickSkill)
    skill.env = SimpleNamespace(
        world_graph={0: spy_graph},
        sim=SimpleNamespace(),
    )
    skill.agent_uid = 0
    skill.failed = False
    skill.steps = 0
    skill.target_handle = "ghost_handle.object_config.json"
    skill._target_name = "ghost_object"
    skill.grasp_mgr = SimpleNamespace()
    skill.thresh_for_art_state = 0.4
    skill.articulated_agent = SimpleNamespace(
        ee_transform=lambda: SimpleNamespace(translation=[0.0, 0.0, 0.0]),
        base_pos=[0.0, 0.0, 0.0],
    )
    skill._config = SimpleNamespace(grasping_distance=1.0)
    skill.grip_index = 0
    skill.object_index = 1
    skill._is_action_issued = torch.zeros(1)

    action = _dummy_action()
    monkeypatch.setattr(
        pick_module,
        "check_if_gripper_is_full",
        lambda env, action, grasp_mgr, target_handle: (action, None, False),
    )
    monkeypatch.setattr(
        pick_module,
        "check_if_the_object_is_moveable",
        lambda env, action, target_handle: (action, None, False),
    )
    monkeypatch.setattr(
        pick_module,
        "check_if_the_object_is_held_by_agent",
        lambda env, action, target_handle, agent_uid: (action, None, False),
    )
    monkeypatch.setattr(
        pick_module,
        "check_if_the_object_is_inside_furniture",
        lambda env, action, target_handle, thresh: (action, None, False),
    )
    monkeypatch.setattr(pick_module.sutils, "get_obj_from_handle", lambda sim, handle: None)

    pick_module.OraclePickSkill._internal_act(
        skill,
        observations={},
        rnn_hidden_states=None,
        prev_actions=action,
        masks=torch.ones((1, 1)),
        cur_batch_idx=0,
    )

    assert spy_graph.removed == ["ghost_object"]
    assert skill.termination_message == sim_utils.GHOST_OBJECT_PICK_FAILURE
    assert skill.failed is True


def test_oracle_pick_does_not_remove_node_for_non_ghost_failure(monkeypatch):
    spy_graph = _RaisingGraph()
    skill = pick_module.OraclePickSkill.__new__(pick_module.OraclePickSkill)
    skill.env = SimpleNamespace(world_graph={0: spy_graph})
    skill.agent_uid = 0
    skill.failed = False
    skill.steps = 0
    skill.target_handle = "real_handle.object_config.json"
    skill._target_name = "real_object"
    skill.grasp_mgr = SimpleNamespace()

    action = _dummy_action()
    monkeypatch.setattr(
        pick_module,
        "check_if_gripper_is_full",
        lambda env, action, grasp_mgr, target_handle: (
            torch.zeros_like(action),
            "Failed to pick! The arm is currently grasping another object.",
            True,
        ),
    )

    pick_module.OraclePickSkill._internal_act(
        skill,
        observations={},
        rnn_hidden_states=None,
        prev_actions=action,
        masks=torch.ones((1, 1)),
        cur_batch_idx=0,
    )

    assert spy_graph.removed == []
    assert skill.failed is True
