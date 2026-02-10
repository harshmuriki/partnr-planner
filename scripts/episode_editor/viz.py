#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

#  python scripts/episode_editor/viz.py   --dataset /home/harshmuriki/Documents/partnr-planner/data/datasets/partnr_episodes/v0_0/val.json.gz   --metadata-dir data/versioned_data/partnr_episodes/v0_0/metadata/   --save-path visualizations/ --episode-id 101

import argparse
import gzip
import itertools
import json
import os
import random
import sys
import traceback
from collections import defaultdict

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# Add project root to path for habitat imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.prediviz.entities.constants import (
    CROPPED_RECEPTACLE_ICONS_PATH,
    FONTS_DIR_PATH,
    RECEPTACLE_ICONS_PATH,
)
from scripts.prediviz.entities.object import Object
from scripts.prediviz.entities.prediviz import PrediViz
from scripts.prediviz.entities.receptacle import Receptacle
from scripts.prediviz.entities.room import Room
from scripts.prediviz.entities.scene import Scene
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from habitat.utils.visualizations import maps

    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("Warning: Habitat not available. Top-down maps will not be generated.")

# Import habitat_llm components for EnvironmentInterface approach
try:
    from habitat_llm.agent.env import (
        EnvironmentInterface,
        register_actions,
        register_measures,
        register_sensors,
        remove_visual_sensors,
    )
    from habitat_llm.agent.env.dataset import CollaborationDatasetV0
    from habitat_llm.utils import setup_config, fix_config
    from habitat_llm.world_model import Room as WMRoom, Furniture, WorldGraph
    from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle
    from habitat_llm.examples.extract_spatial_data import extract_spatial_data_for_episode
    import omegaconf
    HABITAT_LLM_AVAILABLE = True
except ImportError:
    HABITAT_LLM_AVAILABLE = False
    extract_spatial_data_for_episode = None
    print("Warning: habitat_llm not available. Using legacy sim creation.")

# Try to import scene_utils for getting spatial data
try:
    from scripts.episode_editor import scene_utils
    SCENE_UTILS_AVAILABLE = True
except ImportError:
    try:
        import scene_utils
        SCENE_UTILS_AVAILABLE = True
    except ImportError:
        SCENE_UTILS_AVAILABLE = False
        print("Warning: scene_utils not available. Spatial data will not be included.")

matplotlib.use("Agg")


def load_configuration():
    """
    Load configuration from config.yaml file.
    """
    # Config is in prediviz directory
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "prediviz", "conf", "config.yaml"
    )
    return OmegaConf.load(config_path)


def initialize_environment_from_episode(dataset_path, episode_id, metadata_dir):
    """
    Initialize EnvironmentInterface for a specific episode using the same pattern as extract_furniture_data.py.
    
    Args:
        dataset_path: Path to the dataset JSON/JSON.gz file
        episode_id: Episode ID to load
        metadata_dir: Path to metadata directory (required)
    
    Returns:
        Tuple of (sim, world_graph, env_interface) or None if habitat_llm not available
    """
    if not HABITAT_LLM_AVAILABLE:
        print("Warning: habitat_llm not available, falling back to legacy sim creation")
        return None

    # Load base config manually without Hydra decorators
    import omegaconf
    from pathlib import Path

    config_file = Path(__file__).parent.parent.parent / "habitat_llm/conf/examples/skill_runner_default_config.yaml"

    if not config_file.exists():
        print(f"Warning: Config file not found: {config_file}")
        return None

    # Load config from file
    config = omegaconf.OmegaConf.load(config_file)

    # Apply overrides with open_dict to allow modifications
    with omegaconf.open_dict(config):
        config.habitat.dataset.data_path = dataset_path
        config.habitat.dataset.split = "val"  # Required by CollaborationDatasetV0
        config.habitat.dataset.content_scenes = ["*"]  # Load all scenes
        # Update metadata folder
        if not hasattr(config.habitat.dataset, 'metadata'):
            config.habitat.dataset.metadata = {}
        config.habitat.dataset.metadata.metadata_folder = metadata_dir

    seed = 47668090
    config = setup_config(config, seed)

    # No video needed for visualization extraction
    # Skip remove_visual_sensors since config may not have gym.obs_keys
    # Visual sensors aren't needed for spatial data extraction anyway

    # Create dataset and environment
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    # Load specific episode
    print(f"Loading episode_id = {episode_id}")
    env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(str(episode_id))
    env_interface.reset_environment()

    sim = env_interface.sim
    agent_uid = config.robot_agent_uid
    world_graph = env_interface.world_graph[agent_uid]

    print(f"Episode {sim.ep_info.episode_id} loaded in scene {sim.ep_info.scene_id}")

    return sim, world_graph, env_interface


def save_topdown_map(episode_data, run_data, save_path, map_resolution=512):
    """
    Generate and save a colored top-down map of the environment with rooms and semantic information.

    Args:
        episode_data: Episode metadata dictionary
        run_data: Run data dictionary containing episode info
        save_path: Directory to save the map image
        map_resolution: Resolution of the top-down map (reduced default to avoid memory issues)
    """
    if not HABITAT_AVAILABLE:
        return None

    try:
        import habitat_sim
        from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
        from habitat_sim.nav import NavMeshSettings

        # Get scene information from run_data or episode_data
        scene_id = run_data.get("scene_id")
        scene_dataset_config = run_data.get("scene_dataset_config")

        if not scene_id:
            # Try to get from episode_data if not in run_data
            scene_id = episode_data.get("scene_id")
            scene_dataset_config = episode_data.get("scene_dataset_config")

        if not scene_id:
            print("Warning: No scene_id found for episode, skipping top-down map")
            return None

        if not scene_dataset_config:
            scene_dataset_config = (
                "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"
            )

        # Create simulator configuration similar to metadata_extractor
        cfg = get_config_defaults()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = scene_dataset_config
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = False  # Don't need physics for top-down map
        backend_cfg.create_renderer = True  # Need renderer for RGB top-down view

        # Create agent config with RGB camera sensor for top-down rendering
        sensor_specs = []
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        # Use PINHOLE (Perspective) for standard camera behavior we can easily calibrate
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        rgb_sensor_spec.resolution = [1024, 1024]  # High resolution for detailed view
        rgb_sensor_spec.position = [0.0, 0.0, 0.0]
        sensor_specs.append(rgb_sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        # Initialize simulator
        sim = habitat_sim.Simulator(hab_cfg)

        # Generate navmesh (required for top-down map)
        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = cfg.agent_radius
        navmesh_settings.agent_height = cfg.agent_height
        navmesh_settings.include_static_objects = True
        navmesh_settings.agent_max_climb = cfg.agent_max_climb
        navmesh_settings.agent_max_slope = cfg.agent_max_slope

        if not sim.pathfinder.is_loaded:
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # Get scene bounding box to position camera
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        scene_center = scene_bb.center()
        scene_size = scene_bb.size()

        # Calculate camera height - position it high above the scene
        # Use the maximum dimension to ensure we capture everything
        max_dimension = max(scene_size[0], scene_size[2])
        camera_height = scene_center[1] + max_dimension * 1  # Original height

        # For orthographic camera, we need to set the scale (width/height in world units)
        # Habitat Sim might not expose this directly in CameraSensorSpec for Python binding easily?
        # Usually it's done via setting the projection matrix or specific parameters.
        # If SensorSubType.ORTHOGRAPHIC is used, we need to check if we can set the scale.
        # Assuming default orthographic behavior might need adjustment.
        # Let's try positioning and see. For ortho, usually FOV is replaced by size.
        # If we can't set size easily, we might need to stick to perspective or adjust scale later.
        # However, looking at codebase search results, 'ortho_rgba_sensor' uses SensorSubType.ORTHOGRAPHIC.

        # Position camera looking straight down
        agent = sim.get_agent(0)

        # Ensure North is Up (-Z)
        import magnum as mn

        mn.Vector3(scene_center[0], camera_height, scene_center[2])
        mn.Vector3(scene_center[0], scene_center[1], scene_center[2])
        mn.Vector3(0, 0, -1)  # World -Z is Up on the map (North)

        # Create look_at matrix
        # Note: Matrix4.look_at returns a transform that transforms local camera vectors to world.
        # Habitat AgentState.rotation expects the rotation part of this transform.
        # Camera coordinate system: -Z is forward (view direction), +Y is up, +X is right.
        # We want View (-Z) to point to World -Y (Down).
        # We want Up (+Y) to point to World -Z (North).
        # We want Right (+X) to point to World +X (East).

        # Let's construct the rotation matrix explicitly to be sure
        # Forward (Camera -Z) = (0, -1, 0)  (Down)
        # Up (Camera +Y)      = (0, 0, -1)  (North)
        # Right (Camera +X)   = (1, 0, 0)   (East)

        # Check Cross Product: Right x Up = (1,0,0) x (0,0,-1) = (0, 1, 0)? No.
        # (1,0,0) x (0,0,-1) => y = 1*(-1)*(-1)? No.
        # i x -k = j. (1,0,0)x(0,0,-1) = (0,1,0).
        # This gives +Y (Up in World). But Camera Forward is -Z.
        # If Camera Fwd (-Z) is (0,-1,0) => Camera Back (+Z) is (0,1,0).
        # Right x Up = Back?
        # i x j = k.
        # Here Right(i) x Up(-k) = j.
        # Back(k') should be j. Correct.

        # So we want:
        # X column: (1, 0, 0)
        # Y column: (0, 0, -1)
        # Z column: (0, 1, 0)  (Back points Up)

        rotation_matrix = mn.Matrix3(
            mn.Vector3(1.0, 0.0, 0.0),
            mn.Vector3(0.0, 0.0, -1.0),
            mn.Vector3(0.0, 1.0, 0.0),
        )

        agent_state = habitat_sim.AgentState()
        agent_state.position = [scene_center[0], camera_height, scene_center[2]]
        # Habitat requires quaternion coefficients [x, y, z, w]
        quat = mn.Quaternion.from_matrix(rotation_matrix)
        agent_state.rotation = [
            quat.vector.x,
            quat.vector.y,
            quat.vector.z,
            quat.scalar,
        ]
        agent.set_state(agent_state)

        # Render the top-down RGB view
        observations = sim.get_sensor_observations()
        rgb_obs = observations.get("rgb")

        use_rgb_rendering = rgb_obs is not None

        if not use_rgb_rendering:
            # Fallback to occupancy map if rendering fails
            print("Warning: RGB rendering failed, falling back to occupancy map")
            try:
                top_down_map = maps.get_topdown_map_from_sim(
                    sim, map_resolution=map_resolution, draw_border=True
                )
            except MemoryError:
                print(
                    f"Warning: Memory error with resolution {map_resolution}, trying 256..."
                )
                top_down_map = maps.get_topdown_map_from_sim(
                    sim, map_resolution=256, draw_border=True
                )
                map_resolution = 256
            top_down_map_colored = maps.colorize_topdown_map(top_down_map)
        else:
            # Use the rendered RGB image
            top_down_map_colored = rgb_obs
            # Create a dummy top_down_map for shape reference (not used for RGB rendering)
            # We'll use the RGB image dimensions instead
            top_down_map = None

        # Get room information from episode_data
        rooms_info = episode_data.get("rooms", [])
        episode_data.get("recep_to_room", {})
        episode_data.get("room_to_id", {})

        # Create a room color mapping
        room_colors = {}
        if len(rooms_info) > 0:
            # Generate distinct colors for each room using a colormap
            cmap = plt.cm.get_cmap("tab20", max(len(rooms_info), 20))
            for i, room_name in enumerate(rooms_info):
                rgba = cmap(i)
                room_colors[room_name] = tuple(int(c * 255) for c in rgba[:3])

        # Create figure with room overlays
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(top_down_map_colored)
        ax.axis("off")

        # Get semantic scene information for room coloring
        semantic_scene = sim.semantic_scene
        if len(semantic_scene.regions) > 0 and len(rooms_info) > 0:
            # Get map dimensions and coordinate system
            if use_rgb_rendering:
                # Use RGB image dimensions
                map_height = top_down_map_colored.shape[0]
                map_width = top_down_map_colored.shape[1]

                # Calibrate coordinate system using scene center and camera height
                # For perspective camera looking straight down:
                # The visible extent at floor level depends on FOV and height above floor.
                # Default vertical FOV is 90 degrees
                fov_y_deg = 90.0
                fov_y_rad = np.deg2rad(fov_y_deg)

                # Estimate floor height from scene bounding box (bottom)
                floor_height = scene_bb.min[1]
                height_above_floor = camera_height - floor_height

                # Visible height at floor level
                world_height_visible = 2 * height_above_floor * np.tan(fov_y_rad / 2)

                # Visible width (assuming square aspect ratio for now, or aspect ratio of image)
                aspect_ratio = map_width / map_height
                world_width_visible = world_height_visible * aspect_ratio

                # Calculate meters per pixel
                meters_per_pixel = world_height_visible / map_height

                # Map origin: Top-Left corner of the visible area at floor level
                # Camera is at (scene_center[0], ..., scene_center[2]) which corresponds to image center.
                # Top-Left World (x, z)
                # X (Left) = Center X - Width / 2
                # Z (Top) = Center Z - Height / 2  (Wait: Top of image corresponds to -Z relative to center)
                # Image V=0 (Top) -> World Z = Center Z - Height/2 (North)
                # Image V=H (Bottom) -> World Z = Center Z + Height/2 (South)

                map_origin_x = scene_center[0] - world_width_visible / 2
                map_origin_z = scene_center[2] - world_height_visible / 2
            else:
                # Use occupancy map dimensions
                meters_per_pixel = maps.calculate_meters_per_pixel(
                    map_resolution, pathfinder=sim.pathfinder
                )
                map_height = top_down_map.shape[0]
                map_width = top_down_map.shape[1]
                # For occupancy map, origin is at center
                pathfinder_bounds = sim.pathfinder.get_bounds()
                pathfinder_bounds[0]
                map_origin_x = 0
                map_origin_z = 0

            # Create a semi-transparent overlay for rooms
            room_overlay = np.zeros((map_height, map_width, 4), dtype=np.float32)

            # Map semantic regions to rooms and color them
            for _region_idx, region in enumerate(semantic_scene.regions):
                region_name = (
                    region.category.name().split("/")[0].replace(" ", "_").lower()
                )

                # Find matching room name from episode_data
                matching_room = None
                for room_name in rooms_info:
                    room_base = room_name.split("_")[0].lower()
                    if (
                        region_name == room_base
                        or region_name in room_base
                        or room_base in region_name
                    ):
                        matching_room = room_name
                        break

                if matching_room and matching_room in room_colors:
                    color = room_colors[matching_room]
                    # Get region bounding box
                    aabb = region.aabb
                    # Range3D has center() and size() methods
                    center = aabb.center()
                    size = aabb.size()

                    # Convert 3D coordinates to map pixel coordinates
                    if use_rgb_rendering:
                        # For RGB rendering:
                        # map_origin_x is World X at Image Left (u=0)
                        # map_origin_z is World Z at Image Top (v=0)
                        # u = (x - origin_x) / meters_per_pixel
                        # v = (z - origin_z) / meters_per_pixel

                        x_min = max(
                            0,
                            int(
                                (center[0] - size[0] / 2 - map_origin_x)
                                / meters_per_pixel
                            ),
                        )
                        x_max = min(
                            map_width,
                            int(
                                (center[0] + size[0] / 2 - map_origin_x)
                                / meters_per_pixel
                            ),
                        )

                        y_min = max(
                            0,
                            int(
                                (center[2] - size[2] / 2 - map_origin_z)
                                / meters_per_pixel
                            ),
                        )
                        y_max = min(
                            map_height,
                            int(
                                (center[2] + size[2] / 2 - map_origin_z)
                                / meters_per_pixel
                            ),
                        )
                    else:
                        # For occupancy map: map origin is at center
                        map_center_x = map_width // 2
                        map_center_y = map_height // 2
                        # Convert world coordinates (x, z) to map coordinates
                        # Note: y is up, so we use x and z for the 2D map
                        x_min = max(
                            0,
                            int(
                                map_center_x
                                + (center[0] - size[0] / 2) / meters_per_pixel
                            ),
                        )
                        x_max = min(
                            map_width,
                            int(
                                map_center_x
                                + (center[0] + size[0] / 2) / meters_per_pixel
                            ),
                        )
                        y_min = max(
                            0,
                            int(
                                map_center_y
                                - (center[2] + size[2] / 2) / meters_per_pixel
                            ),
                        )
                        y_max = min(
                            map_height,
                            int(
                                map_center_y
                                - (center[2] - size[2] / 2) / meters_per_pixel
                            ),
                        )

                    # Only color navigable areas (where top_down_map == 1, or all areas for RGB rendering)
                    if x_min < x_max and y_min < y_max:
                        if use_rgb_rendering:
                            # For RGB rendering, color all areas in the bounding box
                            mask = np.ones((y_max - y_min, x_max - x_min), dtype=bool)
                        else:
                            # For occupancy map, only color navigable areas
                            mask = top_down_map[y_min:y_max, x_min:x_max] == 1
                        if np.any(mask):
                            room_overlay[y_min:y_max, x_min:x_max, 0][mask] = (
                                color[0] / 255.0
                            )
                            room_overlay[y_min:y_max, x_min:x_max, 1][mask] = (
                                color[1] / 255.0
                            )
                            room_overlay[y_min:y_max, x_min:x_max, 2][mask] = (
                                color[2] / 255.0
                            )
                            room_overlay[y_min:y_max, x_min:x_max, 3][
                                mask
                            ] = 0.4  # Semi-transparent

                            # Add room label at center of region
                            label_x = (x_min + x_max) // 2
                            label_y = (y_min + y_max) // 2
                            if 0 <= label_x < map_width and 0 <= label_y < map_height:
                                # Show full room name with instance number (e.g., living_room_0)
                                room_display_name = matching_room
                                ax.text(
                                    label_x,
                                    label_y,
                                    room_display_name,
                                    fontsize=11,
                                    color="black",
                                    weight="bold",
                                    bbox=dict(
                                        boxstyle="round,pad=0.5",
                                        facecolor="white",
                                        edgecolor="black",
                                        alpha=0.8,
                                        linewidth=1.5,
                                    ),
                                    ha="center",
                                    va="center",
                                )

            # Overlay room colors
            ax.imshow(room_overlay, alpha=0.6)

        # Legend removed per user request

        plt.tight_layout()
        map_path = os.path.join(save_path, "topdown_map.png")
        plt.savefig(map_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        sim.close()
        del sim

        return map_path
    except MemoryError as e:
        print(
            f"Warning: Memory error generating top-down map (scene may be too large): {e}"
        )
        return None
    except Exception as e:
        print(f"Warning: Failed to generate top-down map: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_episode_data(metadata_dir, episode_id):
    """
    Load episode data from JSON file.
    """
    with open(os.path.join(metadata_dir, f"{episode_id}.json")) as f:
        return json.load(f)


def get_object_dimensions_from_sim(scene_id: str, episode_data: dict) -> tuple:
    """
    Load simulator and extract object/furniture dimensions from their bounding boxes.

    Args:
        scene_id: The scene ID
        episode_data: Episode data with object and receptacle handles

    Returns:
        Tuple of (object_dims, receptacle_dims) dictionaries with bounds/size info
    """
    if not HABITAT_AVAILABLE:
        return {}, {}

    try:
        import habitat_sim
        from habitat.datasets.rearrange.run_episode_generator import get_config_defaults

        # Create minimal simulator configuration
        cfg = get_config_defaults()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = True  # Need physics for object managers
        backend_cfg.create_renderer = False

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        # Initialize simulator
        sim = habitat_sim.Simulator(hab_cfg)

        object_dims = {}
        receptacle_dims = {}

        # Get dimensions for regular objects
        rom = sim.get_rigid_object_manager()
        if "object_to_handle" in episode_data:
            for obj_name, obj_handle in episode_data["object_to_handle"].items():
                try:
                    obj = rom.get_object_by_handle(obj_handle)
                    if obj is not None:
                        aabb = obj.root_scene_node.cumulative_bb
                        min_pt = aabb.min
                        max_pt = aabb.max
                        size = aabb.size()

                        object_dims[obj_name] = {
                            "bounds": {
                                "min": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
                                "max": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
                            },
                            "size": [float(size[0]), float(size[1]), float(size[2])]
                        }
                except Exception as e:
                    print(f"Warning: Could not get dimensions for object {obj_name}: {e}")

        # Get dimensions for receptacles (furniture) - check both ROM and AOM
        if "recep_to_handle" in episode_data:
            aom = sim.get_articulated_object_manager()
            for recep_name, recep_handle in episode_data["recep_to_handle"].items():
                try:
                    # Try articulated object manager first (furniture with joints)
                    obj = aom.get_object_by_handle(recep_handle)
                    if obj is None:
                        # Try rigid object manager (static furniture)
                        obj = rom.get_object_by_handle(recep_handle)

                    if obj is not None:
                        aabb = obj.root_scene_node.cumulative_bb
                        min_pt = aabb.min
                        max_pt = aabb.max
                        size = aabb.size()

                        receptacle_dims[recep_name] = {
                            "bounds": {
                                "min": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
                                "max": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
                            },
                            "size": [float(size[0]), float(size[1]), float(size[2])]
                        }
                except Exception as e:
                    print(f"Warning: Could not get dimensions for receptacle {recep_name}: {e}")

        sim.close()
        return object_dims, receptacle_dims

    except Exception as e:
        print(f"Warning: Could not load simulator to get object dimensions: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}


def get_room_bounds_from_sim(scene_id: str, episode_data: dict) -> dict:
    """
    Load simulator and extract room bounds from semantic scene.

    Args:
        scene_id: The scene ID
        episode_data: Episode data with room information

    Returns:
        Dictionary mapping room names to their bounds and centers
    """
    if not HABITAT_AVAILABLE:
        return {}

    try:
        import habitat_sim
        from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
        from habitat_sim.nav import NavMeshSettings

        # Create minimal simulator configuration
        cfg = get_config_defaults()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = False
        backend_cfg.create_renderer = False  # Don't need renderer for bounds

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        # Initialize simulator
        sim = habitat_sim.Simulator(hab_cfg)

        # Get semantic scene
        semantic_scene = sim.semantic_scene
        if not semantic_scene or len(semantic_scene.regions) == 0:
            sim.close()
            return {}

        room_bounds_data = {}
        rooms_info = episode_data.get("rooms", [])

        # Map semantic regions to rooms and extract bounds
        for region in semantic_scene.regions:
            region_name = region.category.name().split("/")[0].replace(" ", "_").lower()

            # Find matching room name from episode_data
            matching_room = None
            for room_name in rooms_info:
                room_base = room_name.split("_")[0].lower()
                if (region_name == room_base
                    or region_name in room_base
                    or room_base in region_name):
                    matching_room = room_name
                    break

            if matching_room:
                aabb = region.aabb
                center = aabb.center()
                size = aabb.size()
                min_pt = aabb.min
                max_pt = aabb.max

                room_bounds_data[matching_room] = {
                    "bounds": {
                        "min": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
                        "max": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
                    },
                    "center": [float(center[0]), float(center[1]), float(center[2])],
                    "size": [float(size[0]), float(size[1]), float(size[2])]
                }

        sim.close()
        return room_bounds_data

    except Exception as e:
        print(f"Warning: Could not load simulator to get room bounds: {e}")
        return {}


def get_all_scene_objects_from_file(scene_id: str) -> dict:
    """
    Read all objects from the scene_instance.json file.

    Args:
        scene_id: The scene ID

    Returns:
        Dictionary mapping object keys to dict with position, rotation, scale
    """
    from pathlib import Path

    # Get scene file path
    project_root = Path(__file__).parent.parent.parent
    scene_path = project_root / f"data/hssd-hab/scenes-partnr-filtered/{scene_id}.scene_instance.json"

    if not scene_path.exists():
        print(f"Warning: Scene file not found: {scene_path}")
        return {}

    try:
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)

        all_objects = {}

        # Process object_instances (regular objects/furniture)
        if 'object_instances' in scene_data:
            for i, obj in enumerate(scene_data['object_instances']):
                template_name = obj.get('template_name', '')
                if template_name:
                    # Create unique key for each object instance
                    obj_key = f"{template_name}_{i}"

                    all_objects[obj_key] = {
                        'template_name': template_name,
                        'position': obj.get('translation', [0, 0, 0]),
                        'rotation': obj.get('rotation', [0, 0, 0, 1]),
                        'scale': obj.get('non_uniform_scale', [1, 1, 1]),
                        'motion_type': obj.get('motion_type', 'static'),
                        'object_type': 'object',
                        'index': i
                    }

        # Process articulated_object_instances (furniture with movable parts)
        if 'articulated_object_instances' in scene_data:
            for i, obj in enumerate(scene_data['articulated_object_instances']):
                template_name = obj.get('template_name', '')
                if template_name:
                    # Create unique key for each articulated object instance
                    obj_key = f"{template_name}_articulated_{i}"

                    all_objects[obj_key] = {
                        'template_name': template_name,
                        'position': obj.get('translation', [0, 0, 0]),
                        'rotation': obj.get('rotation', [0, 0, 0, 1]),
                        'scale': obj.get('non_uniform_scale', [1, 1, 1]),
                        'motion_type': obj.get('motion_type', 'static'),
                        'object_type': 'articulated_object',
                        'index': i
                    }

        print(f"Loaded {len(all_objects)} objects from scene file")
        return all_objects

    except Exception as e:
        print(f"Error reading scene file: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_object_bounds_from_sim(scene_id: str, template_data: dict) -> dict:
    """
    Load simulator and get bounding box sizes for objects from template data.

    Args:
        scene_id: The scene ID
        template_data: Dictionary mapping object keys to their template info

    Returns:
        Dictionary mapping object keys to bounds/size info
    """
    if not HABITAT_AVAILABLE:
        return {}

    try:
        import habitat_sim
        from habitat.datasets.rearrange.run_episode_generator import get_config_defaults

        # Create minimal simulator configuration
        cfg = get_config_defaults()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = False
        backend_cfg.create_renderer = False

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        # Initialize simulator
        sim = habitat_sim.Simulator(hab_cfg)

        object_bounds = {}

        # Get template managers
        obj_template_mgr = sim.get_object_template_manager()

        # Process each template to get default bounds
        template_to_bounds = {}
        for obj_key, obj_data in template_data.items():
            template_name = obj_data['template_name']

            # Skip if we already processed this template
            if template_name in template_to_bounds:
                object_bounds[obj_key] = template_to_bounds[template_name].copy()
                continue

            try:
                # Get the template
                template = obj_template_mgr.get_template_by_handle(template_name)
                if template:
                    # Get bounding box from template
                    bb = template.bounding_box_diagonal
                    size = [float(bb[0]), float(bb[1]), float(bb[2])]

                    bounds_data = {
                        "bounds": {
                            "min": [0, 0, 0],  # Relative to object center
                            "max": size
                        },
                        "size": size
                    }

                    template_to_bounds[template_name] = bounds_data
                    object_bounds[obj_key] = bounds_data
            except Exception as e:
                # Skip objects that can't be loaded
                pass

        sim.close()
        print(f"Got bounds for {len(object_bounds)} objects from simulator")
        return object_bounds

    except Exception as e:
        print(f"Warning: Could not load simulator to get object bounds: {e}")
        import traceback
        traceback.print_exc()
        return {}


def extract_spatial_data_from_world_graph(sim, world_graph, episode_data: dict) -> dict:
    """
    Extract spatial information (position, rotation, bounds) from world_graph and sim.
    
    This replaces the old extract_spatial_data() function and uses the EnvironmentInterface
    pattern like extract_furniture_data.py for better performance and accuracy.

    Args:
        sim: Habitat simulator instance from EnvironmentInterface
        world_graph: WorldGraph instance from EnvironmentInterface
        episode_data: The episode data dictionary (for compatibility)

    Returns:
        Dictionary containing spatial information for objects, receptacles, and rooms
    """
    if sim is None or world_graph is None:
        print("Warning: sim or world_graph not available, using legacy approach")
        return extract_spatial_data(episode_data)

    scene_id = sim.ep_info.scene_id if hasattr(sim.ep_info, "scene_id") else ""
    if "/" in scene_id:
        scene_id = scene_id.split("/")[-1]

    print(f"Extracting spatial data from world_graph for scene {scene_id}")

    # Initialize spatial data containers
    spatial_data = {
        "scene_id": scene_id,
        "objects": {},
        "receptacles": {},
        "rooms": {},
        "scene_objects": {}
    }

    # Extract furniture/receptacles from world_graph
    all_furniture = world_graph.get_all_furnitures()
    print(f"Found {len(all_furniture)} furniture pieces in world_graph")

    for furn_node in all_furniture:
        name = furn_node.name
        sim_handle = furn_node.sim_handle or ""
        translation = furn_node.properties.get("translation", None)
        furn_type = furn_node.properties.get("type", "")

        recep_data = {
            "handle": sim_handle,
            "position": list(translation) if translation is not None else [0, 0, 0],
            "rotation": [0, 0, 0, 1],  # World graph doesn't store rotation
            "type": furn_type,
        }

        # Get AABB from simulator
        aabb_data = _get_aabb_from_sim_for_viz(sim, sim_handle, name)
        if aabb_data:
            recep_data.update(aabb_data)

        spatial_data["receptacles"][name] = recep_data
        spatial_data["scene_objects"][name] = recep_data.copy()

    # Extract objects from world_graph
    all_objects = world_graph.get_all_objects()
    print(f"Found {len(all_objects)} objects in world_graph")

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
        aabb_data = _get_aabb_from_sim_for_viz(sim, sim_handle, name)
        if aabb_data:
            obj_data.update(aabb_data)

        spatial_data["objects"][name] = obj_data

    # Extract rooms from world_graph
    all_rooms = world_graph.get_all_rooms()
    print(f"Found {len(all_rooms)} rooms in world_graph")

    # Get room bounds from semantic scene
    semantic_scene = sim.semantic_scene
    room_bounds_map = {}

    if semantic_scene and len(semantic_scene.regions) > 0:
        for region in semantic_scene.regions:
            region_name = region.category.name().split("/")[0].replace(" ", "_").lower()
            aabb = region.aabb

            if aabb:
                bounds = {
                    "min": [aabb.min[0], aabb.min[1], aabb.min[2]],
                    "max": [aabb.max[0], aabb.max[1], aabb.max[2]],
                }
                center = [
                    (aabb.min[0] + aabb.max[0]) / 2,
                    (aabb.min[1] + aabb.max[1]) / 2,
                    (aabb.min[2] + aabb.max[2]) / 2,
                ]
                size = [
                    aabb.max[0] - aabb.min[0],
                    aabb.max[1] - aabb.min[1],
                    aabb.max[2] - aabb.min[2],
                ]

                room_bounds_map[region_name] = {
                    "bounds": bounds,
                    "center": center,
                    "size": size,
                }

    for room_node in all_rooms:
        name = room_node.name
        room_id = room_node.properties.get("id", name)

        # Match room bounds by name
        room_info = {}
        for region_name, bounds_data in room_bounds_map.items():
            if region_name in name.lower() or name.lower() in region_name:
                room_info = bounds_data
                break

        spatial_data["rooms"][name] = {
            "name": name,
            "room_id": room_id,
            "bounds": room_info.get("bounds"),
            "center": room_info.get("center"),
            "size": room_info.get("size"),
        }

    return spatial_data


def _get_aabb_from_sim_for_viz(sim, sim_handle, entity_name):
    """
    Helper function to get AABB from simulator (reused from extract_furniture_data.py pattern).
    
    Args:
        sim: Habitat simulator
        sim_handle: Entity's sim handle
        entity_name: Name for logging
    
    Returns:
        Dictionary with bounds, center, size, top_surface_y, or None
    """
    if not sim_handle:
        return None

    try:
        entity_obj = get_obj_from_handle(sim, sim_handle)
    except Exception:
        return None

    if entity_obj is None:
        return None

    aabb = None

    # Try .aabb first
    try:
        aabb = entity_obj.aabb
    except Exception:
        pass

    # Fallback to root_scene_node.cumulative_bb for articulated objects
    if aabb is None:
        try:
            aabb = entity_obj.root_scene_node.cumulative_bb
        except Exception:
            pass

    if aabb is None:
        return None

    # Transform AABB to global coordinates
    try:
        transform = entity_obj.transformation
        global_min = transform.transform_point(aabb.min)
        global_max = transform.transform_point(aabb.max)

        # Ensure min < max for each axis
        g_min = [min(global_min[i], global_max[i]) for i in range(3)]
        g_max = [max(global_min[i], global_max[i]) for i in range(3)]

        center = [(g_min[i] + g_max[i]) / 2.0 for i in range(3)]
        size = [g_max[i] - g_min[i] for i in range(3)]
        top_surface_y = g_max[1]

        return {
            "bounds": {"min": g_min, "max": g_max},
            "center": center,
            "size": size,
            "top_surface_y": top_surface_y,
        }
    except Exception:
        return None


def extract_spatial_data(episode_data: dict) -> dict:
    """
    Legacy function for extracting spatial data when habitat_llm is not available.
    Uses the old approach with multiple simulator instances.
    
    DEPRECATED: Use extract_spatial_data_from_world_graph() with EnvironmentInterface instead.
    
    Args:
        episode_data: The episode data dictionary

    Returns:
        Dictionary containing spatial information for objects, receptacles, and rooms
    """
    if not SCENE_UTILS_AVAILABLE:
        print("Warning: scene_utils not available, skipping spatial information")
        return None

    # Get scene_id from run_data or episode_data
    scene_id = '106366386_174226770'  # Hardcoded for testing
    # if run_data:
    #     scene_id = run_data.get("scene_id")
    # if not scene_id and "scene_id" in episode_data:
    #     scene_id = episode_data["scene_id"]

    # if not scene_id:
    #     print("Warning: No scene_id found, skipping spatial information")
    #     return None

    # Initialize spatial data containers
    spatial_data = {
        "scene_id": scene_id,
        "objects": {},
        "receptacles": {},
        "rooms": {},
        "scene_objects": {}  # All objects from scene file
    }

    # Get spatial data for objects
    if "object_to_handle" in episode_data:
        for obj_name, obj_handle in episode_data["object_to_handle"].items():
            try:
                obj_info = scene_utils.get_object_position(scene_id, obj_handle)
                if obj_info:
                    spatial_data["objects"][obj_name] = {
                        "handle": obj_handle,
                        "position": obj_info["position"],
                        "rotation": obj_info["rotation"],
                        "template_name": obj_info["template_name"],
                        "type": obj_info["object_type"]
                    }
            except Exception as e:
                print(f"Warning: Could not get spatial data for object {obj_name}: {e}")

    # Get spatial data for receptacles (furniture)
    if "recep_to_handle" in episode_data:
        for recep_name, recep_handle in episode_data["recep_to_handle"].items():
            try:
                recep_info = scene_utils.get_object_position(scene_id, recep_handle)
                if recep_info:
                    spatial_data["receptacles"][recep_name] = {
                        "handle": recep_handle,
                        "position": recep_info["position"],
                        "rotation": recep_info["rotation"],
                        "template_name": recep_info["template_name"],
                        "type": recep_info["object_type"]
                    }
            except Exception as e:
                print(f"Warning: Could not get spatial data for receptacle {recep_name}: {e}")

    # Get spatial data for rooms (bounding boxes from semantic regions)
    print("Extracting room bounds from simulator...")
    room_bounds_data = get_room_bounds_from_sim(scene_id, episode_data)

    if "rooms" in episode_data:
        for room_name in episode_data["rooms"]:
            room_id = episode_data.get("room_to_id", {}).get(room_name, room_name)

            # Get bounds from simulator if available
            room_info = room_bounds_data.get(room_name, {})

            spatial_data["rooms"][room_name] = {
                "name": room_name,
                "room_id": room_id,
                "bounds": room_info.get("bounds"),
                "center": room_info.get("center"),
                "size": room_info.get("size")
            }

    # Get object and receptacle dimensions (bounds and sizes)
    print("Extracting object and receptacle dimensions from simulator...")
    object_dims, receptacle_dims = get_object_dimensions_from_sim(scene_id, episode_data)

    # Merge dimension data into spatial_data
    for obj_name in spatial_data["objects"]:
        if obj_name in object_dims:
            spatial_data["objects"][obj_name].update(object_dims[obj_name])

    for recep_name in spatial_data["receptacles"]:
        if recep_name in receptacle_dims:
            spatial_data["receptacles"][recep_name].update(receptacle_dims[recep_name])

    # Get all scene objects from the scene file
    print("Extracting all scene objects from scene file...")
    all_scene_objects = get_all_scene_objects_from_file(scene_id)

    # Get bounds for all scene objects
    if all_scene_objects:
        print("Extracting bounds for all scene objects from simulator...")
        scene_object_bounds = get_object_bounds_from_sim(scene_id, all_scene_objects)

        # Merge bounds into scene objects data
        for obj_key, obj_data in all_scene_objects.items():
            spatial_data["scene_objects"][obj_key] = obj_data.copy()
            if obj_key in scene_object_bounds:
                spatial_data["scene_objects"][obj_key].update(scene_object_bounds[obj_key])

    return spatial_data


def load_run_data(run_data, episode_id):
    """
    Load run data and retrieve episode data.
    """
    for episode in run_data["episodes"]:
        if episode["episode_id"] == str(episode_id):
            return episode
    return None


def plot_scene(
    config,
    episode_data,
    propositions,
    constraints,
    receptacle_icon_mapping,
    cropped_receptacle_icon_mapping,
    instruction=None,
    save_path=None,
    object_to_recep=None,
    object_to_room=None,
    object_to_states=None,
):
    objects = []
    # Initial Objects and States
    for obj_id in episode_data["object_to_room"]:
        new_obj = Object(config, obj_id)
        if object_to_states is not None and obj_id in object_to_states:
            new_obj.states = object_to_states[obj_id]
            new_obj.previous_states = object_to_states[obj_id].copy()
        objects.append(new_obj)

    rooms = []
    for room_id in episode_data["rooms"]:
        room_receptacles = []
        # Initial Receptacles and States
        for receptacle_id, r_room_id in episode_data["recep_to_room"].items():
            if r_room_id == room_id:
                icon_path = receptacle_icon_mapping.get(
                    receptacle_id, f"{RECEPTACLE_ICONS_PATH}/chair@2x.png"
                )
                new_recep = Receptacle(config, receptacle_id, icon_path)

                # NOTE: Receptacle also have states, but they MIGHT be present in the object_to_states
                if object_to_states is not None and receptacle_id in object_to_states:
                    new_recep.states = object_to_states[receptacle_id]
                    new_recep.previous_states = object_to_states[receptacle_id].copy()
                room_receptacles.append(new_recep)

        # Objects in the room
        room_objects = [
            obj
            for obj in objects
            if episode_data["object_to_room"][obj.object_id] == room_id
        ]
        room = Room(
            config,
            room_id,
            room_receptacles,
            room_objects,
            object_to_recep=object_to_recep,
        )
        rooms.append(room)

    scene = Scene(
        config,
        rooms,
        episode_data["instruction"] if instruction is None else instruction,
        object_to_recep,
        object_to_room,
    )
    prediviz = PrediViz(config, scene)
    result_fig_data = prediviz.plot(
        propositions,
        constraints,
        receptacle_icon_mapping,
        cropped_receptacle_icon_mapping,
        show_instruction=config.show_instruction,
    )
    step_id_to_path_mapping = {}
    for step_idx, (fig, ax, final_height, final_width) in enumerate(result_fig_data):
        width_inches = config.width_inches
        fig.set_size_inches(width_inches, (final_height / final_width) * width_inches)

        plt.sca(ax)
        if config.show_instruction:
            plt.subplots_adjust(right=0.98, left=0.02, bottom=0.02, top=0.95)
        else:
            # tight
            plt.subplots_adjust(right=0.99, left=0.01, bottom=0.01, top=0.99)
        if save_path:
            # Save each step as a separate image
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"step_{step_idx}.png"), dpi=300)
            step_id_to_path_mapping[step_idx] = os.path.join(
                save_path, f"step_{step_idx}.png"
            )
        else:
            fig.show()
        plt.close(fig)
    scene.cleanup()
    del scene
    return step_id_to_path_mapping


def get_episode_data_for_plot(metadata_dir, episode_id, loaded_run_data):
    episode_data = load_episode_data(metadata_dir, episode_id)
    handle_to_recep = {v: k for k, v in episode_data["recep_to_handle"].items()}
    handle_to_object = {v: k for k, v in episode_data["object_to_handle"].items()}
    id_to_room = {v: k for k, v in episode_data["room_to_id"].items()}
    for receptacle_id in episode_data["recep_to_description"]:
        if not os.path.exists(
            f'{RECEPTACLE_ICONS_PATH}/{"_".join(receptacle_id.split("_")[:-1])}@2x.png'
        ):
            raise NotImplementedError(
                f"Missing receptacle asset for receptacle ID: {receptacle_id}"
            )

    receptacle_icon_mapping = {
        receptacle_id: f'{RECEPTACLE_ICONS_PATH}/{"_".join(receptacle_id.split("_")[:-1])}@2x.png'
        for receptacle_id in episode_data["recep_to_description"]
    }
    cropped_receptacle_icon_mapping = {
        receptacle_id: f'{CROPPED_RECEPTACLE_ICONS_PATH}/{"_".join(receptacle_id.split("_")[:-1])}@2x.png'
        for receptacle_id in episode_data["recep_to_description"]
    }
    run_data = load_run_data(loaded_run_data, episode_id)

    if run_data is None:
        raise ValueError(f"Episode {episode_id} not found in run data. Make sure the episode has been evaluated and is present in the dataset.")

    propositions = run_data["evaluation_propositions"]
    for proposition in propositions:
        if proposition["function_name"] not in [
            "is_on_top",
            "is_inside",
            "is_on_floor",
            "is_in_room",
            "is_next_to",
            "is_filled",
            "is_powered_on",
            "is_powered_off",
            "is_clean",
        ]:
            raise NotImplementedError(
                f'Not implemented for function_name {proposition["function_name"]}'
            )
        if "object_handles" in proposition["args"]:
            if proposition["function_name"] in [
                "is_clean",
                "is_filled",
                "is_powered_on",
                "is_powered_off",
            ]:
                for handle in proposition["args"]["object_handles"]:
                    if handle in handle_to_recep:
                        if "receptacle_names" not in proposition["args"]:
                            proposition["args"]["receptacle_names"] = []
                        proposition["args"]["receptacle_names"].append(
                            handle_to_recep[handle]
                        )
                    else:
                        if "object_names" not in proposition["args"]:
                            proposition["args"]["object_names"] = []
                        proposition["args"]["object_names"].append(
                            handle_to_object[handle]
                        )
            else:
                proposition["args"]["object_names"] = []
                for object_handle in proposition["args"]["object_handles"]:
                    proposition["args"]["object_names"].append(
                        handle_to_object[object_handle]
                    )

        if "receptacle_handles" in proposition["args"]:
            proposition["args"]["receptacle_names"] = []
            for recep_handle in proposition["args"]["receptacle_handles"]:
                proposition["args"]["receptacle_names"].append(
                    handle_to_recep[recep_handle]
                )

        if "room_ids" in proposition["args"]:
            proposition["args"]["room_names"] = []
            for room_id in proposition["args"]["room_ids"]:
                proposition["args"]["room_names"].append(id_to_room[room_id])
        if "entity_handles_a" in proposition["args"]:
            for entity_index in ["a", "b"]:
                proposition["args"][
                    f"entity_handles_{entity_index}_names_and_types"
                ] = []
                for entity_handle in proposition["args"][
                    f"entity_handles_{entity_index}"
                ]:
                    if entity_handle in handle_to_object:
                        proposition["args"][
                            f"entity_handles_{entity_index}_names_and_types"
                        ].append((handle_to_object[entity_handle], "object"))
                    elif entity_handle in handle_to_recep:
                        proposition["args"][
                            f"entity_handles_{entity_index}_names_and_types"
                        ].append((handle_to_recep[entity_handle], "receptacle"))
                    else:
                        raise ValueError(
                            f"Unknown entity type for handle {entity_handle}. Should be either object or receptacle."
                        )

    # Handle Constraints
    constraints = run_data["evaluation_constraints"]
    for _idx, constraint in enumerate(constraints):
        if constraint["type"] == "TemporalConstraint":
            digraph = nx.DiGraph(constraint["args"]["dag_edges"])
            constraint["toposort"] = [
                sorted(generation) for generation in nx.topological_generations(digraph)
            ]
        elif constraint["type"] == "TerminalSatisfactionConstraint":
            continue
        elif constraint["type"] == "SameArgConstraint":
            same_args = []
            for proposition_index, arg_name in zip(
                constraint["args"]["proposition_indices"],
                constraint["args"]["arg_names"],
            ):
                if arg_name == "object_handles" or arg_name == "receptacle_handles":
                    if arg_name == "object_handles":
                        left_name = "object_names"
                        if (
                            "receptacle_names"
                            in propositions[proposition_index]["args"]
                        ):
                            right_name = "receptacle_names"
                        elif "room_names" in propositions[proposition_index]["args"]:
                            right_name = "room_names"
                        else:
                            raise NotImplementedError(
                                f"Not implemented for `arg_name`: {arg_name} and no receptacle or room names."
                            )
                    elif arg_name == "receptacle_handles":
                        left_name = "receptacle_names"
                        right_name = "object_names"

                    same_args.append(
                        {
                            "common_entities": [
                                (item, left_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    left_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "entity_handles_a" or arg_name == "entity_handles_b":
                    entity_index = arg_name.split("_")[-1]
                    opposite_entity_index = "b" if entity_index == "a" else "a"
                    same_args.append(
                        {
                            "common_entities": propositions[proposition_index]["args"][
                                f"entity_handles_{entity_index}_names_and_types"
                            ],
                            "corresponding_entities": propositions[proposition_index][
                                "args"
                            ][
                                f"entity_handles_{opposite_entity_index}_names_and_types"
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        f"entity_handles_{entity_index}_names_and_types"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "room_ids":
                    right_name = "object_names"
                    same_args.append(
                        {
                            "common_entities": [
                                (item, arg_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    arg_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                else:
                    raise NotImplementedError(
                        f"Not implemented SameArg for arg name: {arg_name}"
                    )
            constraint["same_args_data"] = {
                "proposition_indices": constraint["args"]["proposition_indices"],
                "data": same_args,
            }
        elif constraint["type"] == "DifferentArgConstraint":
            diff_args = []
            for proposition_index, arg_name in zip(
                constraint["args"]["proposition_indices"],
                constraint["args"]["arg_names"],
            ):
                if arg_name == "object_handles" or arg_name == "receptacle_handles":
                    if arg_name == "object_handles":
                        left_name = "object_names"
                        if (
                            "receptacle_names"
                            in propositions[proposition_index]["args"]
                        ):
                            right_name = "receptacle_names"
                        elif "room_names" in propositions[proposition_index]["args"]:
                            right_name = "room_names"
                        else:
                            raise NotImplementedError(
                                f"Not implemented for `arg_name`: {arg_name} and no receptacle or room names."
                            )
                    elif arg_name == "receptacle_handles":
                        left_name = "receptacle_names"
                        right_name = "object_names"

                    diff_args.append(
                        {
                            "different_entities": [
                                (item, left_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    left_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "entity_handles_a" or arg_name == "entity_handles_b":
                    entity_index = arg_name.split("_")[-1]
                    opposite_entity_index = "b" if entity_index == "a" else "b"
                    diff_args.append(
                        {
                            "different_entities": propositions[proposition_index][
                                "args"
                            ][f"entity_handles_{entity_index}_names_and_types"],
                            "corresponding_entities": propositions[proposition_index][
                                "args"
                            ][
                                f"entity_handles_{opposite_entity_index}_names_and_types"
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        f"entity_handles_{entity_index}_names_and_types"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "room_ids":
                    right_name = "object_names"
                    diff_args.append(
                        {
                            "different_entities": [
                                (item, arg_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    arg_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                else:
                    raise NotImplementedError(
                        f"Not implemented SameArg for arg name: {arg_name}"
                    )
            constraint["diff_args_data"] = {
                "proposition_indices": constraint["args"]["proposition_indices"],
                "data": diff_args,
            }
        else:
            raise NotImplementedError(
                f"Constraint type {constraint['type']} is not handled currently."
            )
    return (
        episode_data,
        run_data,
        receptacle_icon_mapping,
        cropped_receptacle_icon_mapping,
        propositions,
        constraints,
    )


def sample_episodes(loaded_run_data, sample_size, metadata_dir):
    """
    Repeatedly sample an episode from each scene without replacement until reaching the
    desired sample size. Calls get_episode_data_for_plot() to validate that the episode
    can be visualized.
    """

    # Group episodes by scene_id
    grouped_episodes = defaultdict(list)
    for ep in loaded_run_data["episodes"]:
        grouped_episodes[ep["scene_id"]].append(ep)

    # Shuffle scene IDs to ensure random order
    scene_ids = list(grouped_episodes.keys())
    random.shuffle(scene_ids)
    shuffled_grouped_episodes = [grouped_episodes[sid] for sid in scene_ids]

    # merge scene episode lists
    shuffled_episodes = []
    for elements in itertools.zip_longest(*shuffled_grouped_episodes):
        shuffled_episodes.extend(filter(lambda x: x is not None, elements))

    # Sample one episode from each scene until reaching the desired sample size
    sampled_eids, idx = [], 0
    while len(sampled_eids) < sample_size:
        selected_episode = shuffled_episodes[idx]
        eid = selected_episode["episode_id"]
        sid = selected_episode["scene_id"]
        try:
            get_episode_data_for_plot(
                metadata_dir,
                eid,
                loaded_run_data,
            )
            sampled_eids.append(eid)
        except Exception as e:
            print(f"[Skipped] sid:{sid} eid:{eid} error: {e}")
            continue

        idx += 1
        if len(sampled_eids) >= sample_size or idx >= len(shuffled_episodes):
            break

    return sampled_eids


def build_hierarchy(episode_data):
    """
    Build hierarchical structure: room -> furniture/receptacles -> objects

    :param episode_data: Episode data containing mappings
    :return: Hierarchical dictionary structure
    """
    hierarchy = {}

    # Initialize rooms
    rooms_list = episode_data.get("rooms", [])
    room_to_id = episode_data.get("room_to_id", {})

    for room_name in rooms_list:
        hierarchy[room_name] = {
            "room_id": room_to_id.get(room_name),
            "receptacles": {}
        }

    # Add receptacles to rooms
    recep_to_room = episode_data.get("recep_to_room", {})
    recep_to_description = episode_data.get("recep_to_description", {})
    recep_to_handle = episode_data.get("recep_to_handle", {})

    for recep_name, recep_room in recep_to_room.items():
        if recep_room in hierarchy:
            hierarchy[recep_room]["receptacles"][recep_name] = {
                "description": recep_to_description.get(recep_name, ""),
                "handle": recep_to_handle.get(recep_name),
                "objects": []
            }

    # Add objects to receptacles
    object_to_recep = episode_data.get("object_to_recep", {})
    object_to_room = episode_data.get("object_to_room", {})
    object_to_handle = episode_data.get("object_to_handle", {})
    object_to_states = episode_data.get("object_to_states", {})

    for obj_name, recep_name in object_to_recep.items():
        obj_room = object_to_room.get(obj_name)

        # If receptacle exists in hierarchy, add object
        if obj_room and obj_room in hierarchy:
            if recep_name in hierarchy[obj_room]["receptacles"]:
                obj_info = {
                    "name": obj_name,
                    "handle": object_to_handle.get(obj_name),
                    "states": object_to_states.get(obj_name, {})
                }
                hierarchy[obj_room]["receptacles"][recep_name]["objects"].append(obj_info)

    return hierarchy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot scene")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the dataset file (.json or .json.gz)",
    )
    parser.add_argument(
        "--metadata-dir",
        required=True,
        type=str,
        help="Directory containing the episode metadata JSON files",
    )
    parser.add_argument(
        "--save-path", required=True, type=str, help="Directory to save the figures"
    )
    parser.add_argument(
        "--episode-id",
        required=False,
        default=None,
        type=int,
        help="Index of episode",
    )
    parser.add_argument(
        "--sample-size",
        required=False,
        type=int,
        default=0,
        help="If only a random subset of all the episodes is to be visualized, the sample size.",
    )
    parser.add_argument(
        "--save-hierarchy",
        action="store_true",
        help="Save hierarchical JSON file with room -> furniture -> object structure",
    )
    return parser.parse_args()


def main():
    """
    Main function to plot scenes based on provided arguments.
    """
    args = parse_arguments()
    config = load_configuration()
    font_files = font_manager.findSystemFonts(fontpaths=[FONTS_DIR_PATH])
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    plt.rcParams["font.family"] = "Inter"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["text.color"] = "white"

    if args.dataset.endswith(".gz"):
        with gzip.open(args.dataset, "rt") as f:
            loaded_run_data = json.load(f)
    else:
        with open(args.dataset, "r") as f:
            loaded_run_data = json.load(f)

    if args.episode_id is not None:
        eids = [args.episode_id]
    else:
        if args.sample_size:
            eids = sample_episodes(loaded_run_data, args.sample_size, args.metadata_dir)
        else:
            eids = sorted([int(ep["episode_id"]) for ep in loaded_run_data["episodes"]])

    # Create a dictionary to store run data for episodes with correct visualizations
    run_data_dict = {"config": None, "episodes": []}

    os.makedirs(args.save_path, exist_ok=True)
    for episode_id in tqdm(eids, dynamic_ncols=True):
        env_interface = None
        sim = None
        world_graph = None

        try:
            # Try to use EnvironmentInterface approach for better performance
            if HABITAT_LLM_AVAILABLE:
                print(f"\nInitializing environment with EnvironmentInterface for episode {episode_id}...")
                try:
                    result = initialize_environment_from_episode(
                        args.dataset, episode_id, args.metadata_dir
                    )
                    if result:
                        sim, world_graph, env_interface = result
                        print("✓ EnvironmentInterface initialized successfully")
                except Exception as e:
                    print(f"Warning: Failed to initialize EnvironmentInterface: {e}")
                    print("Falling back to legacy sim creation...")

            (
                episode_data,
                run_data,
                receptacle_icon_mapping,
                cropped_receptacle_icon_mapping,
                propositions,
                constraints,
            ) = get_episode_data_for_plot(
                args.metadata_dir, episode_id, loaded_run_data
            )

            # Save episode_data as JSON inside the folder
            ep_data_f = os.path.join(args.save_path, f"episode_data_{episode_id}.json")
            with open(ep_data_f, "w") as f:
                json.dump(episode_data, f, indent=4)

            # Extract and save spatial information to separate file
            # Try Hydra wrapper first (best performance, single sim)
            spatial_data = None
            if HABITAT_LLM_AVAILABLE and extract_spatial_data_for_episode:
                try:
                    print("Extracting spatial data using Hydra wrapper...")
                    spatial_data = extract_spatial_data_for_episode(
                        args.dataset, episode_id, args.metadata_dir
                    )
                    if spatial_data:
                        print("✓ Spatial data extracted via Hydra wrapper")
                except Exception as e:
                    print(f"Warning: Hydra wrapper failed: {e}")
                    spatial_data = None

            # Fallback to legacy extraction if Hydra wrapper failed
            if not spatial_data:
                spatial_data = extract_spatial_data(episode_data)

            if spatial_data:
                spatial_f = os.path.join(args.save_path, f"spatial_data_{episode_id}.json")
                with open(spatial_f, "w") as f:
                    json.dump(spatial_data, f, indent=4)
                print(f"Saved spatial data to {spatial_f}")

            # Save hierarchical structure if flag is enabled
            if args.save_hierarchy:
                hierarchy = build_hierarchy(episode_data)
                hierarchy_f = os.path.join(args.save_path, f"viz_{episode_id}", f"hierarchy_{episode_id}.json")
                os.makedirs(os.path.dirname(hierarchy_f), exist_ok=True)
                with open(hierarchy_f, "w") as f:
                    json.dump(hierarchy, f, indent=4)
                print(f"Saved hierarchy to {hierarchy_f}")

            step_id_to_path_mapping = plot_scene(
                config,
                episode_data,
                propositions,
                constraints,
                receptacle_icon_mapping,
                cropped_receptacle_icon_mapping,
                instruction=run_data["instruction"],
                save_path=os.path.join(args.save_path, f"viz_{episode_id}"),
                object_to_recep=episode_data["object_to_recep"],
                object_to_room=episode_data["object_to_room"],
                object_to_states=episode_data.get("object_to_states", None),
            )

            # Generate and save top-down map
            # If we have sim from EnvironmentInterface, reuse it; otherwise create one in save_topdown_map
            topdown_map_path = save_topdown_map(
                episode_data,
                run_data,
                os.path.join(args.save_path, f"viz_{episode_id}"),
                map_resolution=1024,
            )

            # Add run data for the current episode to the dictionary
            run_data["viz_paths"] = step_id_to_path_mapping
            if topdown_map_path:
                run_data["topdown_map_path"] = topdown_map_path
            run_data_dict["episodes"].append(run_data)

            # Save the run data dictionary to a JSON file
            with open(f"{args.save_path}/run_data.json", "w") as f:
                json.dump(run_data_dict, f, indent=4)

        except Exception:
            print(f"Episode ID: {episode_id}")
            print(traceback.format_exc())

        finally:
            # Clean up: close environment interface for this episode
            if env_interface is not None:
                try:
                    print(f"Closing EnvironmentInterface for episode {episode_id}...")
                    env_interface.close()
                    print("✓ EnvironmentInterface closed")
                except Exception as e:
                    print(f"Warning: Error closing EnvironmentInterface: {e}")


if __name__ == "__main__":
    main()
