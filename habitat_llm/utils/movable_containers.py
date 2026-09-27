"""Placement into explicitly annotated pickupable containers.

Keep these objects as Objects in the world graph: owning an interior does not
turn a movable basket into scene furniture. Regions follow the current object
transform, including after moving the container.
"""
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import magnum as mn
from habitat.sims.habitat_simulator import sim_utilities as sutils


@lru_cache(maxsize=1)
def _regions():
    path = Path(__file__).parents[1] / "sims/scene_override_data/movable_containers.json"
    return json.loads(path.read_text())


def container_bounds(obj):
    """Return the annotated local interior, or None for unsupported objects."""
    asset = Path(obj.creation_attributes.handle).name.removesuffix(".object_config.json")
    region = _regions().get(asset)
    if region is None:
        return None
    bb = obj.aabb
    return mn.Range3D(
        bb.min + bb.size() * mn.Vector3(region["lower"]),
        bb.min + bb.size() * mn.Vector3(region["upper"]),
    )


def sample_in_container(sim, container, held, spatial_constraint=None,
                        reference_handle=None, max_samples=10, max_tries=100):
    """Find collision-checked interior poses; always restore the held object.

Only upright, explicitly annotated containers are supported. The stock snap-down
and ray-based containment predicates must both accept each candidate.
"""
    bounds = container_bounds(container)
    if bounds is None:
        raise ValueError("Destination has no annotated movable-container interior")
    if held.object_id == container.object_id:
        raise ValueError("An object cannot contain itself")
    if spatial_constraint not in (None, "next_to"):
        raise ValueError("Only next_to is supported as an additional constraint")
    if container.transformation.transform_vector(mn.Vector3.y_axis()).y < 0.95:
        raise ValueError("Container must be upright before placing objects inside")
    reference = None
    if spatial_constraint:
        reference = sutils.get_obj_from_handle(sim, reference_handle)
        if reference is None:
            raise ValueError("Reference object is unavailable")
    original = held.transformation
    poses = []
    try:
        for _ in range(max_tries):
            # Start above the interior floor; snap_down checks actual collision
            # geometry, including previously placed contents.
            local = mn.Vector3(random.uniform(bounds.min.x, bounds.max.x),
                               bounds.max.y + held.aabb.size().y,
                               random.uniform(bounds.min.z, bounds.max.z))
            held.translation = container.transformation.transform_point(local)
            held.rotation = mn.Quaternion.rotation(mn.Rad(random.uniform(0, math.tau)), mn.Vector3.y_axis())
            if not sutils.snap_down(sim, held, support_obj_ids=[container.object_id]):
                continue
            if container.object_id not in sutils.within(sim, held):
                continue
            if reference is not None and not sutils.obj_next_to(sim, held.object_id, reference.object_id):
                continue
            if any((held.translation - p).length() < 0.025 for p, _ in poses):
                continue
            poses.append((held.translation, held.rotation))
            if len(poses) >= max_samples:
                break
    finally:
        held.transformation = original
    return poses
