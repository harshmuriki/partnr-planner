"""Habitat integration regressions for movable-container placement (requires GPU/assets)."""
import random
from pathlib import Path

import habitat_sim
import magnum as mn
import pytest
from habitat.sims.habitat_simulator import sim_utilities as sutils

from dataset_generation.benchmark_generation.generate_episodes import (
    default_gen_config, default_metadata_dict, initialize_generator,
)
from dataset_generation.benchmark_generation.generate_verified_specs import compile_success
from habitat_llm.utils.movable_containers import container_bounds, sample_in_container


@pytest.fixture(scope="module")
def sim():
    generator = initialize_generator(default_gen_config, default_metadata_dict)
    generator.initialize_fresh_scene("107734176_176000019")
    yield generator.sim
    generator.sim.close()


def spawn(sim, asset):
    templates = sim.get_object_template_manager().get_file_template_handles(asset)
    handle = next(h for h in templates if Path(h).name == asset + ".object_config.json")
    return sim.get_rigid_object_manager().add_object_by_template_handle(handle)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_three_fruits_settle_inside_moved_rotated_basket(sim, seed):
    random.seed(seed)
    objects = []
    try:
        basket = spawn(sim, "87ed6d0c785b3245207bb217eb3ee3fc079f8633")
        objects.append(basket)
        basket.translation = mn.Vector3(seed, 4, 0)
        basket.rotation = mn.Quaternion.rotation(mn.Rad(seed * 0.7), mn.Vector3.y_axis())
        basket.motion_type = habitat_sim.physics.MotionType.STATIC
        for asset in ["Apple_26", "Apple_4", "017_orange"]:
            fruit = spawn(sim, asset)
            objects.append(fruit)
            fruit.translation = mn.Vector3(seed, 5, 0)
            original = fruit.transformation
            poses = sample_in_container(sim, basket, fruit)
            assert fruit.transformation == original
            assert poses
            fruit.translation, fruit.rotation = poses[0]
            for _ in range(120):
                sim.step_physics(1 / 60)
        # Check all three after the last placement, not just each immediately.
        for fruit in objects[1:]:
            assert basket.object_id in sutils.within(sim, fruit)
        outside = objects[-1]
        outside.translation = basket.translation + mn.Vector3(1, 0, 0)
        assert basket.object_id not in sutils.within(sim, outside)
    finally:
        for obj in reversed(objects):
            sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)


def test_sampler_restores_pose_on_failure_and_rejects_invalid_container(sim, monkeypatch):
    basket = spawn(sim, "87ed6d0c785b3245207bb217eb3ee3fc079f8633")
    fruit = spawn(sim, "Apple_26")
    try:
        assert container_bounds(fruit) is None
        with pytest.raises(ValueError, match="no annotated"):
            sample_in_container(sim, fruit, basket)
        with pytest.raises(ValueError, match="itself"):
            sample_in_container(sim, basket, basket)
        basket.rotation = mn.Quaternion.rotation(mn.Rad(3.14159), mn.Vector3.x_axis())
        with pytest.raises(ValueError, match="upright"):
            sample_in_container(sim, basket, fruit)
        basket.rotation = mn.Quaternion()
        original = fruit.transformation
        def fail(*args, **kwargs):
            raise RuntimeError("test collision-query error")
        monkeypatch.setattr(sutils, "snap_down", fail)
        with pytest.raises(RuntimeError, match="collision-query"):
            sample_in_container(sim, basket, fruit)
        assert fruit.transformation == original
    finally:
        for obj in [fruit, basket]:
            sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)


def test_goal_binds_exact_basket_instance():
    spec = {"success": "- is_inside(apple_0, basket_0)\n- is_on_top(basket_0, counter_0)"}
    handles = {"apple_0": "apple-instance", "basket_0": "basket-instance"}
    props, unscored, order = compile_success(spec, handles, {"counter_0": "counter-instance"})
    assert props[0].args["receptacle_handles"] == ["basket-instance"]
    assert props[1].args["object_handles"] == ["basket-instance"]
    assert props[1].args["receptacle_handles"] == ["counter-instance"]
    assert not unscored and not order
