#!/usr/bin/env python3
# isort: skip_file

"""
Extract spatial data from episodes using EnvironmentInterface and WorldGraph.

This is a Hydra wrapper that properly initializes EnvironmentInterface with full config,
then extracts spatial data (rooms, furniture, objects) for use by viz.py and other tools.

Usage from viz.py:
    from habitat_llm.examples.extract_spatial_data import extract_spatial_data_for_episode
    spatial_data = extract_spatial_data_for_episode(dataset_path, episode_id, metadata_dir)

Direct usage:
    python habitat_llm/examples/extract_spatial_data.py \
        +skill_runner_episode_id=666 \
        habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val.json.gz \
        habitat.dataset.metadata.metadata_folder=data/versioned_data/partnr_episodes/v0_0/metadata/ \
        hydra.run.dir="."
"""

import sys
import json
import os
from typing import Optional, Dict, Any

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
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle


def _get_aabb_from_sim(sim, sim_handle: str, obj_name: str) -> Optional[Dict[str, Any]]:
    """
    Get AABB (axis-aligned bounding box) data for an object from the simulator.
    
    :param sim: The Habitat simulator instance
    :param sim_handle: The simulator handle for the object
    :param obj_name: Human-readable name for logging
    :return: Dictionary with AABB bounds, center, size, or None if not found
    """
    if not sim_handle:
        return None

    try:
        # Try to get the object from simulator
        obj = get_obj_from_handle(sim, sim_handle)
        if obj is None:
            return None

        # Get AABB
        aabb = obj.aabb
        if aabb is None:
            return None

        # Extract bounds
        center = aabb.center
        sizes = aabb.sizes
        min_bound = center - sizes / 2
        max_bound = center + sizes / 2

        return {
            "min": list(min_bound),
            "max": list(max_bound),
            "center": list(center),
            "size": list(sizes),
            "top_surface_y": max_bound[1],
        }
    except Exception as e:
        cprint(f"Warning: Could not get AABB for {obj_name} ({sim_handle}): {e}", "yellow")
        return None


def extract_spatial_data_with_env_interface(
    sim, world_graph, episode_data: dict
) -> dict:
    """
    Extract spatial data using EnvironmentInterface's world_graph.
    
    :param sim: The Habitat simulator instance
    :param world_graph: The WorldGraph instance
    :param episode_data: Episode metadata dictionary
    :return: Dictionary with spatial data (rooms, furniture, objects)
    """
    spatial_data = {
        "scene_id": sim.ep_info.scene_id,
        "episode_id": sim.ep_info.episode_id,
        "rooms": {},
        "furniture": {},
        "objects": {},
    }

    # Extract rooms from world_graph
    all_rooms = world_graph.get_all_rooms()
    cprint(f"Found {len(all_rooms)} rooms in world_graph", "green")

    for room_node in all_rooms:
        room_name = room_node.name
        room_bounds = room_node.properties.get("bounds", None)

        room_data = {
            "name": room_name,
            "bounds": room_bounds,
        }

        spatial_data["rooms"][room_name] = room_data

    # Extract furniture from world_graph
    all_furniture = world_graph.get_all_furnitures()
    cprint(f"Found {len(all_furniture)} furniture pieces in world_graph", "green")

    for furn_node in all_furniture:
        name = furn_node.name
        sim_handle = furn_node.sim_handle or ""
        translation = furn_node.properties.get("translation", None)
        furn_type = furn_node.properties.get("type", "")

        furn_data = {
            "handle": sim_handle,
            "position": list(translation) if translation is not None else [0, 0, 0],
            "type": furn_type,
            "rotation": [0, 0, 0, 1],
        }

        # Get AABB from simulator
        aabb_data = _get_aabb_from_sim(sim, sim_handle, name)
        if aabb_data:
            furn_data.update(aabb_data)

        spatial_data["furniture"][name] = furn_data

    # Extract objects from world_graph
    all_objects = world_graph.get_all_objects()
    cprint(f"Found {len(all_objects)} objects in world_graph", "green")

    for obj_node in all_objects:
        name = obj_node.name
        sim_handle = obj_node.sim_handle or ""
        translation = obj_node.properties.get("translation", None)

        obj_data = {
            "handle": sim_handle,
            "position": list(translation) if translation is not None else [0, 0, 0],
            "rotation": [0, 0, 0, 1],
        }

        # Get AABB from simulator
        aabb_data = _get_aabb_from_sim(sim, sim_handle, name)
        if aabb_data:
            obj_data.update(aabb_data)

        spatial_data["objects"][name] = obj_data

    return spatial_data


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="examples/skill_runner_default_config",
)
def main(config):
    """Main function with Hydra decorator for proper config initialization."""

    # Get episode ID from config
    episode_id = config.get("skill_runner_episode_id", None)
    if episode_id is None:
        raise ValueError("Must provide +skill_runner_episode_id=<ID>")

    # Setup config
    seed = 47668090
    config = setup_config(config, seed)

    # Create dataset and environment
    cprint(f"Initializing environment for episode {episode_id}...", "cyan")
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    # Load specific episode
    env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(str(episode_id))
    env_interface.reset_environment()

    sim = env_interface.sim
    agent_uid = config.robot_agent_uid
    world_graph = env_interface.world_graph[agent_uid]

    cprint(f"Episode {sim.ep_info.episode_id} loaded in scene {sim.ep_info.scene_id}", "green")

    # Extract spatial data
    episode_data = {"episode_id": episode_id}
    spatial_data = extract_spatial_data_with_env_interface(sim, world_graph, episode_data)

    # Save to file if output path specified
    output_path = config.get("spatial_data_output_path", None)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(spatial_data, f, indent=2)
        cprint(f"Saved spatial data to {output_path}", "green")
    else:
        # Print to stdout for piping
        print(json.dumps(spatial_data, indent=2))

    # Cleanup
    env_interface.close()
    cprint("Environment closed", "cyan")


def extract_spatial_data_for_episode(
    dataset_path: str,
    episode_id: int,
    metadata_dir: str,
    output_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Programmatic API to extract spatial data for an episode using subprocess.
    
    This function calls the Hydra-wrapped script as a subprocess to get proper config.
    
    :param dataset_path: Path to episode dataset JSON/JSON.gz
    :param episode_id: Episode ID to extract
    :param metadata_dir: Path to metadata directory
    :param output_path: Optional path to save JSON output
    :return: Dictionary with spatial data, or None if failed
    """
    import subprocess
    import tempfile

    # Use temp file if no output path specified
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_path = temp_file.name
        temp_file.close()
        cleanup_temp = True
    else:
        cleanup_temp = False

    try:
        # Build command
        cmd = [
            "python",
            "-m",
            "habitat_llm.examples.extract_spatial_data",
            f"+skill_runner_episode_id={episode_id}",
            f"habitat.dataset.data_path={dataset_path}",
            f"+habitat.dataset.metadata.metadata_folder={metadata_dir}",
            f"+spatial_data_output_path={output_path}",
            "hydra.run.dir=.",
        ]

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        )

        if result.returncode != 0:
            print(f"Error extracting spatial data: {result.stderr}")
            return None

        # Load the output
        with open(output_path, "r") as f:
            spatial_data = json.load(f)

        return spatial_data

    finally:
        # Cleanup temp file
        if cleanup_temp and os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == "__main__":
    main()
