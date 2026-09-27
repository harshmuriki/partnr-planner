"""Occlusion-aware boxes from a semantic render aligned with the RGB camera."""
from contextlib import contextmanager

import numpy as np


@contextmanager
def temporary_instance_ids(objects):
    """Assign private render IDs, restoring perception's IDs even on failure."""
    saved = []
    try:
        for instance_id, obj in objects:
            for node in obj.visual_scene_nodes:
                saved.append((node, node.semantic_id))
                node.semantic_id = instance_id
        yield
    finally:
        for node, semantic_id in reversed(saved):
            node.semantic_id = semantic_id


def boxes_from_mask(mask, entities, min_pixels=16):
    """Box only visible pixels; omit fully occluded objects and tiny fragments."""
    boxes = []
    for instance_id, name, color in entities:
        ys, xs = np.nonzero(mask == instance_id)
        if len(xs) < min_pixels:
            continue
        boxes.append((name, color, float(xs.min()), float(ys.min()),
                      float(xs.max() + 1), float(ys.max() + 1)))
    return boxes


def visible_entity_boxes(sim, entities, color_for_name, width, height):
    import habitat_sim
    from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle

    rgb = sim.agents[0]._sensors['agent_0_third_rgb']
    uuid = 'agent_0_vlm_visibility_mask'
    if uuid not in sim.agents[0]._sensors:
        source = rgb.specification()
        spec = habitat_sim.CameraSensorSpec()
        for field in ("resolution", "hfov", "near", "far", "sensor_subtype"):
            setattr(spec, field, getattr(source, field))
        spec.uuid = uuid
        spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        spec.gpu2gpu_transfer = False
        sim.add_sensor(spec, agent_id=0)
    semantic = sim.agents[0]._sensors[uuid]
    # Articulated cameras move independently of the agent scene node. Match the
    # actual world pose every time, not the configured initial sensor pose.
    semantic.node.transformation = (
        semantic.node.parent.absolute_transformation().inverted()
        @ rgb.node.absolute_transformation()
    )
    objects, labels = [], []
    seen_handles = set()
    for entity in entities:
        handle = getattr(entity, 'sim_handle', None)
        if not handle or handle == 'floor' or handle in seen_handles:
            continue
        obj = get_obj_from_handle(sim, handle)
        if obj is None:
            continue
        seen_handles.add(handle)
        # Keep private IDs far from Habitat's normal object/scene semantic IDs.
        instance_id = 1_000_000_000 + len(objects)
        objects.append((instance_id, obj))
        labels.append((instance_id, entity.name, color_for_name(entity.name)))
    with temporary_instance_ids(objects):
        sensor = sim._sensors[uuid]
        sensor.draw_observation()
        mask = np.asarray(sensor.get_observation()).copy()
    if mask.shape != (height, width):
        raise ValueError(f'Visibility mask {mask.shape} does not match RGB {(height, width)}')
    return boxes_from_mask(mask, labels)
