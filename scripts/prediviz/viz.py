#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

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
from entities.constants import (
    CROPPED_RECEPTACLE_ICONS_PATH,
    FONTS_DIR_PATH,
    RECEPTACLE_ICONS_PATH,
)
from entities.object import Object
from entities.prediviz import PrediViz
from entities.receptacle import Receptacle
from entities.room import Room
from entities.scene import Scene
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path for habitat imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from habitat.utils.visualizations import maps

    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("Warning: Habitat not available. Top-down maps will not be generated.")

matplotlib.use("Agg")


def load_configuration():
    """
    Load configuration from config.yaml file.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "conf/config.yaml"
    )
    return OmegaConf.load(config_path)


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
        try:
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

            # Save hierarchical structure if flag is enabled
            if args.save_hierarchy:
                hierarchy = build_hierarchy(episode_data)
                hierarchy_f = os.path.join(args.save_path, f"viz_{episode_id}", f"hierarchy_{episode_id}.json")
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


if __name__ == "__main__":
    main()
