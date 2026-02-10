#!/usr/bin/env python3
# isort: skip_file

"""
Extract furniture locations and bounding boxes from the Habitat simulator.

Boots the simulator for a given episode, queries the world graph for all furniture,
extracts AABB (bounding box) data from the sim, and exports to a JSON file.

The output JSON can be used by add_objects_to_scene.py for accurate object placement
(X, Z from world graph; Y from AABB top_surface_y).

Usage:
    python habitat_llm/examples/extract_furniture_data.py \
        +skill_runner_episode_id=101 \
        hydra.run.dir="."

    # Custom output path:
    python habitat_llm/examples/extract_furniture_data.py \
        +skill_runner_episode_id=101 \
        +extract_output_path="path/to/output.json" \
        hydra.run.dir="."
"""

import sys
import json
import os

sys.path.append("..")
sys.path.insert(0, os.getcwd())
import omegaconf
import hydra

from habitat_llm.utils import cprint, setup_config, fix_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle
from habitat_llm.world_model import Room


def extract_furniture_data(sim, world_graph, ep_info):
    """
    Extract furniture locations and bounding box data from the simulator.

    For each furniture in the world graph, queries the sim for the AABB
    and builds a dictionary with translation, sim_handle, type, and AABB info.

    :param sim: The Habitat simulator instance
    :param world_graph: The active WorldGraph
    :param ep_info: Episode info object (sim.ep_info)
    :return: Dictionary with scene_id, episode_id, and furniture data
    """
    all_furniture = world_graph.get_all_furnitures()
    cprint(f"Found {len(all_furniture)} furniture pieces in world graph", "green")

    furniture_dict = {}

    for furn_node in all_furniture:
        name = furn_node.name
        sim_handle = furn_node.sim_handle or ""
        translation = furn_node.properties.get("translation", None)
        furn_type = furn_node.properties.get("type", "")
        is_articulated = furn_node.properties.get("is_articulated", False)

        entry = {
            "furniture_name": name,  # Store the actual furniture name from world graph
            "translation": list(translation) if translation is not None else [0, 0, 0],
            "sim_handle": sim_handle,
            "type": furn_type,
            "is_articulated": bool(is_articulated),
        }

        # Get room name from graph edges
        room_nodes = [
            neighbor for neighbor in world_graph.graph[furn_node]
            if isinstance(neighbor, Room)
        ]
        room_name = room_nodes[0].name if room_nodes else "unknown_room"
        entry["room_name"] = room_name

        if room_nodes:
            cprint(f"  {name}: in room '{room_name}'", "white")
        else:
            cprint(f"  {name}: room not found, using 'unknown_room'", "yellow")

        # Try to get AABB from simulator
        aabb_data = _get_aabb_from_sim(sim, sim_handle, name)
        if aabb_data:
            entry["aabb"] = aabb_data

        furniture_dict[name] = entry

    scene_id = ep_info.scene_id if hasattr(ep_info, "scene_id") else ""
    # Strip path prefix if present (e.g., "data/hssd-hab/.../scene_id" -> "scene_id")
    if "/" in scene_id:
        scene_id = scene_id.split("/")[-1]

    episode_id = str(ep_info.episode_id) if hasattr(ep_info, "episode_id") else ""

    return {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "furniture": furniture_dict,
    }


def _get_aabb_from_sim(sim, sim_handle, furniture_name):
    """
    Query the simulator for a furniture's AABB (axis-aligned bounding box).

    Tries fur_obj.aabb first, then falls back to root_scene_node.cumulative_bb
    for articulated objects.

    :param sim: The Habitat simulator
    :param sim_handle: The furniture's sim handle string
    :param furniture_name: Name for logging
    :return: Dictionary with center, size, min, max, top_surface_y, or None
    """
    if not sim_handle:
        cprint(f"  {furniture_name}: No sim_handle, skipping AABB", "yellow")
        return None

    try:
        fur_obj = get_obj_from_handle(sim, sim_handle)
    except Exception as e:
        cprint(f"  {furniture_name}: Could not get sim object: {e}", "yellow")
        return None

    if fur_obj is None:
        cprint(f"  {furniture_name}: sim object is None", "yellow")
        return None

    aabb = None

    # Try .aabb first
    try:
        aabb = fur_obj.aabb
    except Exception:
        pass

    # Fallback to root_scene_node.cumulative_bb for articulated objects
    if aabb is None:
        try:
            aabb = fur_obj.root_scene_node.cumulative_bb
        except Exception:
            pass

    if aabb is None:
        cprint(f"  {furniture_name}: Could not get AABB", "yellow")
        return None

    # Transform AABB to global coordinates
    try:
        transform = fur_obj.transformation
        global_min = transform.transform_point(aabb.min)
        global_max = transform.transform_point(aabb.max)

        # Ensure min < max for each axis (transform can flip)
        g_min = [min(global_min[i], global_max[i]) for i in range(3)]
        g_max = [max(global_min[i], global_max[i]) for i in range(3)]

        center = [(g_min[i] + g_max[i]) / 2.0 for i in range(3)]
        size = [g_max[i] - g_min[i] for i in range(3)]
        top_surface_y = g_max[1]  # Max Y = top surface

        cprint(
            f"  {furniture_name}: size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}), "
            f"top_y={top_surface_y:.3f}",
            "white",
        )

        return {
            "center": center,
            "size": size,
            "min": g_min,
            "max": g_max,
            "top_surface_y": top_surface_y,
        }
    except Exception as e:
        cprint(f"  {furniture_name}: Error transforming AABB: {e}", "yellow")
        return None


@hydra.main(
    config_path="../conf", config_name="examples/skill_runner_default_config.yaml"
)
def extract_data(config: omegaconf.DictConfig) -> None:
    """
    Main function to extract furniture data from the simulator.

    Uses the same initialization pattern as skill_runner.py but skips
    planner/agent setup and the interactive loop.
    """
    fix_config(config)
    seed = 47668090

    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
    config = setup_config(config, seed)

    # No video needed for extraction
    remove_visual_sensors(config)

    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset and environment
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    config.habitat.dataset.data_path = "data/datasets/partnr_episodes/v0_0/val.json.gz"
    cprint(f"Loading dataset from: {config.habitat.dataset.data_path}", "blue")
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    # Select episode
    assert not (
        hasattr(config, "skill_runner_episode_index")
        and hasattr(config, "skill_runner_episode_id")
    ), "Episode selection options are mutually exclusive."

    if hasattr(config, "skill_runner_episode_index"):
        episode_index = config.skill_runner_episode_index
        cprint(f"Loading episode_index = {episode_index}", "blue")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_index(
            episode_index
        )
    elif hasattr(config, "skill_runner_episode_id"):
        episode_id = config.skill_runner_episode_id
        cprint(f"Loading episode_id = {episode_id}", "blue")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )

    env_interface.reset_environment()

    sim = env_interface.sim
    agent_uid = config.robot_agent_uid
    world_graph = env_interface.world_graph[agent_uid]

    cprint(
        f"Episode {sim.ep_info.episode_id} loaded in scene {sim.ep_info.scene_id}",
        "green",
    )

    # Extract furniture data
    data = extract_furniture_data(sim, world_graph, sim.ep_info)

    # Determine output path
    if hasattr(config, "extract_output_path"):
        output_file = config.extract_output_path
    else:
        output_dir = os.path.join("scripts", "episode_editor", "static", "furniture_data")
        os.makedirs(output_dir, exist_ok=True)
        scene_id = data["scene_id"]
        ep_id = data["episode_id"]
        output_file = os.path.join(output_dir, f"furniture_{scene_id}_{ep_id}.json")

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    cprint(f"\nExported furniture data to: {output_file}", "green")
    cprint(f"  Scene: {data['scene_id']}", "green")
    cprint(f"  Episode: {data['episode_id']}", "green")
    cprint(f"  Furniture count: {len(data['furniture'])}", "green")

    # Print summary
    with_aabb = sum(1 for v in data["furniture"].values() if "aabb" in v)
    with_room = sum(1 for v in data["furniture"].values() if v.get("room_name") and v["room_name"] != "unknown_room")
    unique_rooms = set(v.get("room_name", "unknown_room") for v in data["furniture"].values())

    cprint(f"  With AABB data: {with_aabb}/{len(data['furniture'])}", "green")
    cprint(f"  With room data: {with_room}/{len(data['furniture'])}", "green")
    cprint(f"  Unique rooms: {len(unique_rooms)} ({', '.join(sorted(unique_rooms))})", "green")


if __name__ == "__main__":
    cprint("\nExtracting furniture data from simulator...", "blue")
    extract_data()
    cprint("\nDone.", "blue")
