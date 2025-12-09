#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox

# Add project root to path for habitat imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import habitat_sim
    import magnum as mn
    from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
    from habitat_sim.nav import NavMeshSettings

    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("Warning: Habitat not available. Cannot visualize objects on map.")

# Note: We reimplement map generation here to avoid circular imports


def load_episode(dataset_path: str, episode_id: int) -> Tuple[Dict, Dict]:
    """Load episode from dataset file."""
    if dataset_path.endswith(".gz"):
        with gzip.open(dataset_path, "rt") as f:
            dataset = json.load(f)
    else:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

    episodes = dataset.get("episodes", [])
    if episode_id >= len(episodes):
        raise ValueError(
            f"Episode ID {episode_id} not found. Valid IDs: 0-{len(episodes)-1}"
        )

    episode = episodes[episode_id]
    return episode, dataset


def get_object_positions_from_episode(episode: Dict) -> List[Tuple[str, np.ndarray]]:
    """Extract object positions from episode rigid_objs."""
    object_positions = []
    rigid_objs = episode.get("rigid_objs", [])

    for obj_entry in rigid_objs:
        # Handle both tuple and list formats
        if isinstance(obj_entry, (list, tuple)) and len(obj_entry) >= 2:
            obj_handle = obj_entry[0]
            transform = obj_entry[1]
        else:
            continue

        # Transform is a 4x4 matrix, extract translation (x, y, z)
        if isinstance(transform, list):
            transform = np.array(transform)

        # Handle different transform formats
        if transform.shape == (4, 4):
            # Standard 4x4 matrix - position is in last column, first 3 rows
            position = transform[:3, 3]
        elif transform.shape == (3,):
            # Already a position vector
            position = transform
        elif len(transform.shape) == 1 and len(transform) >= 3:
            # Flattened array, take first 3 elements
            position = transform[:3]
        else:
            print(
                f"Warning: Unknown transform format for {obj_handle}, shape: {transform.shape}"
            )
            continue

        object_positions.append((obj_handle, position))

    return object_positions


def world_to_map_coords(
    world_pos: np.ndarray,
    map_origin_x: float,
    map_origin_z: float,
    meters_per_pixel: float,
) -> Tuple[int, int]:
    """Convert world coordinates (x, z) to map pixel coordinates."""
    map_x = int((world_pos[0] - map_origin_x) / meters_per_pixel)
    map_y = int((world_pos[2] - map_origin_z) / meters_per_pixel)
    return map_x, map_y


def map_to_world_coords(
    map_x: int,
    map_y: int,
    map_origin_x: float,
    map_origin_z: float,
    meters_per_pixel: float,
) -> Tuple[float, float]:
    """Convert map pixel coordinates to world coordinates (x, z)."""
    world_x = map_x * meters_per_pixel + map_origin_x
    world_z = map_y * meters_per_pixel + map_origin_z
    return world_x, world_z


def get_map_calibration(
    sim: habitat_sim.Simulator,
    top_down_map_colored: np.ndarray,
    use_rgb_rendering: bool,
    map_resolution: int,
    scene_center: np.ndarray,
    camera_height: float,
    scene_bb: mn.Range3D,
) -> Tuple[float, float, float]:
    """Get map calibration parameters (origin and meters_per_pixel)."""
    if use_rgb_rendering:
        map_height = top_down_map_colored.shape[0]
        map_width = top_down_map_colored.shape[1]

        # Calibrate coordinate system using scene center and camera height
        fov_y_deg = 90.0
        fov_y_rad = np.deg2rad(fov_y_deg)
        floor_height = scene_bb.min[1]
        height_above_floor = camera_height - floor_height
        world_height_visible = 2 * height_above_floor * np.tan(fov_y_rad / 2)
        aspect_ratio = map_width / map_height
        world_width_visible = world_height_visible * aspect_ratio
        meters_per_pixel = world_height_visible / map_height

        map_origin_x = scene_center[0] - world_width_visible / 2
        map_origin_z = scene_center[2] - world_height_visible / 2
    else:
        from habitat.utils.visualizations import maps

        meters_per_pixel = maps.calculate_meters_per_pixel(
            map_resolution, pathfinder=sim.pathfinder
        )
        map_origin_x = 0
        map_origin_z = 0

    return map_origin_x, map_origin_z, meters_per_pixel


def normalize_object_handle(handle: str) -> str:
    """
    Normalize object handle to format expected by episode.
    Extracts shortname from full path.
    """
    # Extract shortname: remove path, extension, and instance suffix
    # e.g., "/path/to/013_apple.object_config.json" -> "013_apple"
    shortname = handle.split("/")[-1].split(".")[0].split("_:")[0]
    return shortname


def get_available_objects(sim: habitat_sim.Simulator, episode: Dict) -> List[str]:
    """Get list of available object config handles from the simulator."""
    otm = sim.get_object_template_manager()

    # Get additional object paths from episode
    additional_object_paths = episode.get("additional_obj_config_paths", [])
    if not additional_object_paths:
        # Use default paths
        additional_object_paths = [
            "data/objects/ycb/configs/",
            "data/objects_ovmm/train_val/ai2thorhab/configs/objects",
            "data/objects_ovmm/train_val/amazon_berkeley/configs",
            "data/objects_ovmm/train_val/google_scanned/configs",
            "data/objects_ovmm/train_val/hssd/configs/objects",
        ]

    # Load configs from additional paths
    for object_path in additional_object_paths:
        abs_path = os.path.abspath(object_path)
        if os.path.exists(abs_path):
            try:
                otm.load_configs(abs_path)
            except Exception as e:
                print(f"Warning: Could not load configs from {abs_path}: {e}")

    # Get all template handles
    all_handles = otm.get_file_template_handles()

    # Filter to only .object_config.json files and sort
    object_handles = [
        handle for handle in all_handles if handle.endswith(".object_config.json")
    ]
    object_handles.sort()

    return object_handles


def get_regions_at_position(
    sim: habitat_sim.Simulator,
    world_pos: np.ndarray,
) -> List[str]:
    """
    Get semantic regions at a world position.
    Returns list of region semantic names.
    """
    try:
        point = mn.Vector3(world_pos[0], world_pos[1], world_pos[2])
        # Get regions for this point using semantic scene
        # Check which regions contain this point
        regions = []
        for region in sim.semantic_scene.regions:
            try:
                # Check if point is within region bounds
                if region.aabb.contains(point) and region.category is not None:
                    # Get semantic name from category
                    region_name = region.category.name()
                    # Normalize to match format used in episodes (e.g., "living room" -> "living_room_0")
                    region_name = region_name.lower().replace(" ", "_")
                    if region_name not in regions:
                        regions.append(region_name)
            except Exception:
                continue

        # If no regions found, try using get_regions_for_points if available
        if not regions:
            try:
                # Create a temporary object at this position to get regions
                # This is a workaround - we'll check all regions manually
                for region in sim.semantic_scene.regions:
                    if region.aabb.contains(point) and region.category is not None:
                        region_name = region.category.name().lower().replace(" ", "_")
                        if region_name not in regions:
                            regions.append(region_name)
            except Exception:
                pass

        return regions
    except Exception as e:
        print(f"Warning: Could not get regions: {e}")
        return []


def format_receptacle_for_name_to_receptacle(
    sim: habitat_sim.Simulator,
    receptacle_unique_name: str,
) -> Optional[str]:
    """
    Format receptacle unique_name for name_to_receptacle mapping.
    The receptacle.unique_name is already in the format: "parent_handle|receptacle_name"
    We need to ensure the parent_handle has the _:0000 suffix if needed.
    """
    try:
        from habitat_llm.utils.sim import find_receptacles

        receptacles = find_receptacles(sim, filter_receptacles=False)

        for rec in receptacles:
            if rec.unique_name == receptacle_unique_name:
                # The unique_name format is: parent_object_handle|receptacle_name
                # We need to ensure parent_handle has _:0000 suffix
                parent_handle = rec.parent_object_handle
                receptacle_name = rec.name

                # Check if parent_handle already has _:0000 suffix
                if "_:0000" not in parent_handle:
                    # Add _:0000 suffix if not present
                    parent_handle = f"{parent_handle}_:0000"

                # Return formatted: "parent_handle|receptacle_name"
                return f"{parent_handle}|{receptacle_name}"

        return None
    except Exception as e:
        print(f"Warning: Could not format receptacle name: {e}")
        return None


def get_furniture_name_from_receptacle(
    sim: habitat_sim.Simulator,
    receptacle_name: str,
) -> Optional[str]:
    """
    Get furniture name from a receptacle unique name.
    Returns the parent object handle or semantic name.
    """
    try:
        from habitat_llm.utils.sim import find_receptacles

        receptacles = find_receptacles(sim, filter_receptacles=False)

        for rec in receptacles:
            if rec.unique_name == receptacle_name:
                # Get parent object handle
                parent_handle = rec.parent_object_handle

                # Try to get semantic name from the parent object
                try:
                    aom = sim.get_articulated_object_manager()
                    if parent_handle in aom.get_object_handles():
                        ao = aom.get_object_by_handle(parent_handle)
                        if ao.semantic_id > 0:
                            semantic_obj = sim.semantic_scene.get_object(ao.semantic_id)
                            if (
                                semantic_obj is not None
                                and semantic_obj.category is not None
                            ):
                                return semantic_obj.category.name()

                    # Fallback: use handle directly (extract short name)
                    # Handle format might be like "table_5" or a hash
                    return (
                        parent_handle.split("_")[0]
                        if "_" in parent_handle
                        else parent_handle
                    )
                except Exception:
                    # Fallback: return parent handle
                    return parent_handle

        return None
    except Exception as e:
        print(f"Warning: Could not get furniture name: {e}")
        return None


def get_receptacles_at_position(
    sim: habitat_sim.Simulator,
    world_pos: np.ndarray,
    max_dist: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Find receptacles near a world position.
    Returns list of (receptacle_unique_name, distance) tuples.
    """
    try:
        from habitat_llm.utils.sim import find_receptacles

        # Get all receptacles in the scene
        # Use filter_receptacles=False to avoid issues with missing filter files
        receptacles = find_receptacles(sim, filter_receptacles=False)

        if not receptacles:
            return []

        # Check which receptacles contain or are near this point
        matches = []
        point = mn.Vector3(world_pos[0], world_pos[1], world_pos[2])

        for rec in receptacles:
            try:
                # Get receptacle bounds
                rec_aabb = rec.get_global_bounds(sim)

                if rec_aabb is None:
                    continue

                # Check if point is within receptacle bounds (with some tolerance)
                if rec_aabb.contains(point):
                    # Calculate distance to receptacle center
                    rec_center = rec_aabb.center()
                    dist = float((point - rec_center).length())
                    matches.append((rec.unique_name, dist))
                else:
                    # Check if point is close to receptacle surface
                    rec_center = rec_aabb.center()
                    dist = float((point - rec_center).length())
                    if dist < max_dist:
                        matches.append((rec.unique_name, dist))
            except Exception:
                # Skip receptacles that can't be accessed
                continue

        # Sort by distance
        matches.sort(key=lambda x: x[1])
        return matches
    except Exception as e:
        # Fallback: try using semantic scene to find objects at position
        try:
            return get_receptacles_from_semantic_scene(sim, world_pos, max_dist)
        except Exception:
            print(f"Warning: Could not find receptacles: {e}")
            return []


def get_receptacles_from_semantic_scene(
    sim: habitat_sim.Simulator,
    world_pos: np.ndarray,
    max_dist: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Fallback method: Find receptacles using semantic scene objects.
    """
    matches = []
    point = mn.Vector3(world_pos[0], world_pos[1], world_pos[2])

    try:
        semantic_scene = sim.semantic_scene
        if semantic_scene is None:
            return []

        # Check rigid objects
        rom = sim.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            try:
                obj = rom.get_object_by_handle(obj_handle)
                if obj is None:
                    continue

                obj_pos = obj.translation
                # Check if point is near this object (could be a receptacle)
                dist = float((point - obj_pos).length())
                if dist < max_dist:
                    # Use object handle as receptacle identifier
                    matches.append((obj_handle, dist))
            except Exception:
                continue

        # Check articulated objects (furniture)
        aom = sim.get_articulated_object_manager()
        for obj_handle in aom.get_object_handles():
            try:
                obj = aom.get_object_by_handle(obj_handle)
                if obj is None:
                    continue

                obj_pos = obj.translation
                dist = float((point - obj_pos).length())
                if dist < max_dist:
                    matches.append((obj_handle, dist))
            except Exception:
                continue

        matches.sort(key=lambda x: x[1])
        return matches
    except Exception:
        return []


def get_receptacle_surface_position(
    sim: habitat_sim.Simulator,
    receptacle_name: str,
    world_x: float,
    world_z: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Get proper 3D position on top of a receptacle surface.
    Returns (x, y, z) position.
    """
    try:
        from habitat_llm.utils.sim import find_receptacles

        receptacles = find_receptacles(sim, filter_receptacles=False)

        # Find the receptacle
        target_rec = None
        for rec in receptacles:
            try:
                if rec.unique_name == receptacle_name:
                    target_rec = rec
                    break
            except Exception:
                continue

        if target_rec is not None:
            try:
                # Get receptacle bounds
                rec_aabb = target_rec.get_global_bounds(sim)
                if rec_aabb is not None:
                    # Get the top surface of the receptacle
                    rec_top = rec_aabb.max[1]  # Y coordinate of top
                    # Use the clicked x, z coordinates, but use receptacle top for y
                    return (world_x, rec_top, world_z)
            except Exception:
                pass

        # Fallback: try to get position from object handle
        try:
            # Check if receptacle_name is an object handle
            rom = sim.get_rigid_object_manager()
            if receptacle_name in rom.get_object_handles():
                obj = rom.get_object_by_handle(receptacle_name)
                if obj is not None:
                    obj_bb = obj.root_scene_node.cumulative_bb
                    rec_top = obj_bb.max[1]
                    return (world_x, rec_top, world_z)
        except Exception:
            pass

        try:
            # Check articulated objects
            aom = sim.get_articulated_object_manager()
            if receptacle_name in aom.get_object_handles():
                obj = aom.get_object_by_handle(receptacle_name)
                if obj is not None:
                    obj_bb = obj.root_scene_node.cumulative_bb
                    rec_top = obj_bb.max[1]
                    return (world_x, rec_top, world_z)
        except Exception:
            pass

        return None
    except Exception as e:
        print(f"Warning: Could not get receptacle surface position: {e}")
        return None


def create_simulator_for_map(
    episode: Dict,
) -> Tuple[habitat_sim.Simulator, np.ndarray, Dict]:
    """Create simulator and generate top-down map, returning calibration info."""
    if not HABITAT_AVAILABLE:
        raise RuntimeError("Habitat not available")

    scene_id = episode.get("scene_id")
    scene_dataset_config = episode.get("scene_dataset_config")

    if not scene_id:
        raise ValueError("No scene_id found in episode")

    if not scene_dataset_config:
        scene_dataset_config = "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"

    # Create simulator configuration
    cfg = get_config_defaults()
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = scene_dataset_config
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = False
    backend_cfg.create_renderer = True

    # Create agent config with RGB camera
    sensor_specs = []
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_sensor_spec.resolution = [1024, 1024]
    rgb_sensor_spec.position = [0.0, 0.0, 0.0]
    sensor_specs.append(rgb_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

    # Initialize simulator
    sim = habitat_sim.Simulator(hab_cfg)

    # Generate navmesh
    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = cfg.agent_radius
    navmesh_settings.agent_height = cfg.agent_height
    navmesh_settings.include_static_objects = True
    navmesh_settings.agent_max_climb = cfg.agent_max_climb
    navmesh_settings.agent_max_slope = cfg.agent_max_slope

    if not sim.pathfinder.is_loaded:
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    # Get scene bounding box
    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    scene_center = scene_bb.center()
    scene_size = scene_bb.size()

    # Calculate camera height
    max_dimension = max(scene_size[0], scene_size[2])
    camera_height = scene_center[1] + max_dimension * 1

    # Position camera looking straight down
    agent = sim.get_agent(0)
    rotation_matrix = mn.Matrix3(
        mn.Vector3(1.0, 0.0, 0.0),
        mn.Vector3(0.0, 0.0, -1.0),
        mn.Vector3(0.0, 1.0, 0.0),
    )

    agent_state = habitat_sim.AgentState()
    agent_state.position = [scene_center[0], camera_height, scene_center[2]]
    quat = mn.Quaternion.from_matrix(rotation_matrix)
    agent_state.rotation = [
        quat.vector.x,
        quat.vector.y,
        quat.vector.z,
        quat.scalar,
    ]
    agent.set_state(agent_state)

    # Render top-down view
    observations = sim.get_sensor_observations()
    rgb_obs = observations.get("rgb")

    use_rgb_rendering = rgb_obs is not None

    if not use_rgb_rendering:
        from habitat.utils.visualizations import maps

        try:
            top_down_map = maps.get_topdown_map_from_sim(
                sim, map_resolution=1024, draw_border=True
            )
        except MemoryError:
            top_down_map = maps.get_topdown_map_from_sim(
                sim, map_resolution=256, draw_border=True
            )
        top_down_map_colored = maps.colorize_topdown_map(top_down_map)
    else:
        top_down_map_colored = rgb_obs

    # Get calibration
    map_origin_x, map_origin_z, meters_per_pixel = get_map_calibration(
        sim,
        top_down_map_colored,
        use_rgb_rendering,
        1024,
        scene_center,
        camera_height,
        scene_bb,
    )

    calibration = {
        "map_origin_x": map_origin_x,
        "map_origin_z": map_origin_z,
        "meters_per_pixel": meters_per_pixel,
        "use_rgb_rendering": use_rgb_rendering,
    }

    return sim, top_down_map_colored, calibration


class ObjectEditor:
    """Interactive editor for object positions on top-down map."""

    def __init__(
        self,
        episode: Dict,
        dataset: Dict,
        top_down_map: np.ndarray,
        calibration: Dict,
        sim: habitat_sim.Simulator,
        output_path: Optional[str] = None,
    ):
        self.episode = episode
        self.dataset = dataset
        self.top_down_map = top_down_map
        self.calibration = calibration
        self.sim = sim
        self.output_path = output_path

        # Get available objects
        self.available_objects = get_available_objects(sim, episode)

        # Get object positions
        self.object_positions = get_object_positions_from_episode(episode)

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(16, 16))
        self.ax.imshow(top_down_map)
        self.ax.axis("off")

        # Plot objects
        self.object_circles: Dict[str, Any] = {}
        self.object_labels: Dict[str, Any] = {}
        self.selected_object = None
        self.adding_object = False
        self.moving_object = False  # Track if we're moving an object
        self.pending_object_handle = None
        self.pending_position = None  # Store position before confirmation
        self.pending_receptacle = None  # Store receptacle name if placing on receptacle
        self.preview_circle = None  # Preview circle for pending placement

        # GUI widgets for object selection
        self.object_selection_widgets = None
        self.search_textbox = None
        self.object_listbox = None
        self.confirm_button = None
        self.cancel_button = None

        # Instructions (create before update_object_plot so it can be updated)
        self.instructions_text = self.ax.text(
            0.02,
            0.98,
            "Click object to select | Click empty to move | 'd' to delete | 'a' to add | 's' to save | 'q' to quit",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self.update_object_plot()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        plt.tight_layout()

    def update_object_plot(self):
        """Update the plot with current object positions."""
        # Clear existing circles and labels
        for circle in self.object_circles.values():
            circle.remove()
        self.object_circles.clear()

        for label in self.object_labels.values():
            label.remove()
        self.object_labels.clear()

        # Clear preview circle if not in adding or moving mode
        if (
            not self.adding_object
            and not self.moving_object
            and hasattr(self, "preview_circle")
            and self.preview_circle is not None
        ):
            self.preview_circle.remove()
            self.preview_circle = None

        # Plot each object
        for obj_handle, position in self.object_positions:
            map_x, map_y = world_to_map_coords(
                position,
                self.calibration["map_origin_x"],
                self.calibration["map_origin_z"],
                self.calibration["meters_per_pixel"],
            )

            # Check if within map bounds
            if (
                0 <= map_x < self.top_down_map.shape[1]
                and 0 <= map_y < self.top_down_map.shape[0]
            ):
                facecolor = "red" if obj_handle == self.selected_object else "blue"
                circle = Circle(
                    (map_x, map_y),
                    radius=10,
                    facecolor=facecolor,
                    fill=True,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=2,
                )
                self.ax.add_patch(circle)
                self.object_circles[obj_handle] = circle

                # Add label
                obj_name = obj_handle.split("/")[-1].replace(".object_config.json", "")
                if len(obj_name) > 20:
                    obj_name = obj_name[:17] + "..."
                label = self.ax.text(
                    map_x,
                    map_y - 15,
                    obj_name,
                    fontsize=8,
                    ha="center",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
                self.object_labels[obj_handle] = label

        # Update instructions (only if it exists)
        if hasattr(self, "instructions_text") and self.instructions_text is not None:
            if self.adding_object and self.pending_position is None:
                obj_name = (
                    self.pending_object_handle.split("/")[-1].replace(
                        ".object_config.json", ""
                    )
                    if self.pending_object_handle
                    else "object"
                )
                self.instructions_text.set_text(
                    f"Adding '{obj_name}': Click on map to place | 'Esc' to cancel"
                )
            elif self.adding_object and self.pending_position is not None:
                world_x, world_y, world_z = self.pending_position
                self.instructions_text.set_text(
                    f"Confirm placement at ({world_x:.2f}, {world_y:.2f}, {world_z:.2f}) | 'Enter' to confirm | 'Esc' to cancel"
                )
            elif self.moving_object and self.pending_position is not None:
                world_x, world_y, world_z = self.pending_position
                self.instructions_text.set_text(
                    f"Confirm move to ({world_x:.2f}, {world_y:.2f}, {world_z:.2f}) | 'Enter' to confirm | 'Esc' to cancel"
                )
            else:
                self.instructions_text.set_text(
                    "Click object to select | Click empty to move | 'd' to delete | 'a' to add | 's' to save | 'q' to quit"
                )

        self.fig.canvas.draw()

    def on_click(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            click_x, click_y = int(event.xdata), int(event.ydata)

            # Don't allow clicking to place if already have a pending position (waiting for confirmation)
            if (self.adding_object and self.pending_position is not None) or (
                self.moving_object and self.pending_position is not None
            ):
                print(
                    "Please confirm or cancel the current placement/move first (Enter/Esc)"
                )
                return

            # Handle adding new object
            if self.adding_object:
                world_x, world_z = map_to_world_coords(
                    click_x,
                    click_y,
                    self.calibration["map_origin_x"],
                    self.calibration["map_origin_z"],
                    self.calibration["meters_per_pixel"],
                )

                # Try to find receptacle at this position
                world_pos_2d = np.array(
                    [world_x, 0.0, world_z]
                )  # Use floor level for initial check
                receptacle_matches = get_receptacles_at_position(
                    self.sim, world_pos_2d, max_dist=0.5
                )

                # Store receptacle info for confirmation
                self.pending_receptacle = None
                if receptacle_matches:
                    # Use the closest receptacle
                    self.pending_receptacle = receptacle_matches[0][0]
                    # Get proper surface position
                    surface_pos = get_receptacle_surface_position(
                        self.sim, self.pending_receptacle, world_x, world_z
                    )
                    if surface_pos:
                        world_x, world_y, world_z = surface_pos
                    else:
                        # Fallback to default height
                        world_y = 1.0
                else:
                    # No receptacle found, use floor (default height)
                    world_y = 0.1  # Slightly above floor
                    self.pending_receptacle = None

                new_position = np.array([world_x, world_y, world_z])

                # Store pending position for confirmation
                self.pending_position = (world_x, world_y, world_z)

                # Show preview circle
                map_x, map_y = world_to_map_coords(
                    new_position,
                    self.calibration["map_origin_x"],
                    self.calibration["map_origin_z"],
                    self.calibration["meters_per_pixel"],
                )

                # Add preview circle (yellow for pending)
                if hasattr(self, "preview_circle") and self.preview_circle is not None:
                    self.preview_circle.remove()

                self.preview_circle = Circle(
                    (map_x, map_y),
                    radius=12,
                    facecolor="yellow",
                    fill=True,
                    alpha=0.5,
                    edgecolor="orange",
                    linewidth=2,
                )
                self.ax.add_patch(self.preview_circle)

                if self.pending_receptacle:
                    print(
                        f"Preview: {self.pending_object_handle} on {self.pending_receptacle} at ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"
                    )
                else:
                    print(
                        f"Preview: {self.pending_object_handle} on floor at ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"
                    )
                print("Press 'Enter' to confirm placement or 'Esc' to cancel")

                self.update_object_plot()
                return

            # Check if clicking on an object
            clicked_object = None
            min_dist = float("inf")

            for obj_handle, position in self.object_positions:
                map_x, map_y = world_to_map_coords(
                    position,
                    self.calibration["map_origin_x"],
                    self.calibration["map_origin_z"],
                    self.calibration["meters_per_pixel"],
                )

                dist = np.sqrt((map_x - click_x) ** 2 + (map_y - click_y) ** 2)
                if dist < 15 and dist < min_dist:
                    min_dist = dist
                    clicked_object = obj_handle

            if clicked_object:
                # Select object
                self.selected_object = clicked_object
                print(f"Selected object: {clicked_object}")
                self.update_object_plot()
            elif self.selected_object and not self.moving_object:
                # Move selected object to click position (with preview/confirmation)
                # Only allow moving if not already in moving mode
                world_x, world_z = map_to_world_coords(
                    click_x,
                    click_y,
                    self.calibration["map_origin_x"],
                    self.calibration["map_origin_z"],
                    self.calibration["meters_per_pixel"],
                )

                # Try to find receptacle at this position
                world_pos_2d = np.array([world_x, 0.0, world_z])
                receptacle_matches = get_receptacles_at_position(
                    self.sim, world_pos_2d, max_dist=0.5
                )

                # Store receptacle info for confirmation
                self.pending_receptacle = None
                if receptacle_matches:
                    # Use the closest receptacle
                    self.pending_receptacle = receptacle_matches[0][0]
                    # Get proper surface position
                    surface_pos = get_receptacle_surface_position(
                        self.sim, self.pending_receptacle, world_x, world_z
                    )
                    if surface_pos:
                        world_x, world_y, world_z = surface_pos
                    else:
                        # Fallback: keep current height
                        for i, (obj_handle, _) in enumerate(self.object_positions):
                            if obj_handle == self.selected_object:
                                world_y = self.object_positions[i][1][1]
                                break
                else:
                    # No receptacle found, use floor
                    world_y = 0.1  # Slightly above floor
                    self.pending_receptacle = None

                new_position = np.array([world_x, world_y, world_z])

                # Store pending position for confirmation
                self.moving_object = True
                self.pending_position = (world_x, world_y, world_z)

                # Show preview circle
                map_x, map_y = world_to_map_coords(
                    new_position,
                    self.calibration["map_origin_x"],
                    self.calibration["map_origin_z"],
                    self.calibration["meters_per_pixel"],
                )

                # Add preview circle (cyan for moving)
                if hasattr(self, "preview_circle") and self.preview_circle is not None:
                    self.preview_circle.remove()

                self.preview_circle = Circle(
                    (map_x, map_y),
                    radius=12,
                    facecolor="cyan",
                    fill=True,
                    alpha=0.5,
                    edgecolor="blue",
                    linewidth=2,
                )
                self.ax.add_patch(self.preview_circle)

                if self.pending_receptacle:
                    print(
                        f"Preview: Move {self.selected_object} to {self.pending_receptacle} at ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"
                    )
                else:
                    print(
                        f"Preview: Move {self.selected_object} to floor at ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"
                    )
                print("Press 'Enter' to confirm move or 'Esc' to cancel")

                self.update_object_plot()

    def on_key(self, event):
        """Handle key presses."""
        if event.key == "d" and self.selected_object and not self.adding_object:
            # Delete selected object from both rigid_objs and initial_state
            self.object_positions = [
                (handle, pos)
                for handle, pos in self.object_positions
                if handle != self.selected_object
            ]

            # Remove from rigid_objs if present
            if "rigid_objs" in self.episode:
                self.episode["rigid_objs"] = [
                    (handle, transform)
                    for handle, transform in self.episode["rigid_objs"]
                    if handle != self.selected_object
                ]

            # Remove from initial_state if present
            if "info" in self.episode and "initial_state" in self.episode["info"]:
                # Find and remove matching entries
                initial_state = self.episode["info"]["initial_state"]
                # Try to match by object class extracted from handle
                object_class = self.get_object_class_from_handle(self.selected_object)
                self.episode["info"]["initial_state"] = [
                    entry
                    for entry in initial_state
                    if not (
                        isinstance(entry, dict)
                        and entry.get("object_classes")
                        and object_class in entry.get("object_classes", [])
                    )
                ]

            print(f"Deleted object: {self.selected_object}")
            self.selected_object = None
            self.update_object_plot()

        elif event.key == "a" and not self.adding_object and not self.moving_object:
            # Add new object - show GUI dropdown
            if not self.available_objects:
                print(
                    "No available objects found. Check additional_obj_config_paths in episode."
                )
                return

            self.show_object_selection_dialog()

        elif event.key == "escape":
            # Cancel adding or moving object
            if self.adding_object:
                if hasattr(self, "preview_circle") and self.preview_circle is not None:
                    self.preview_circle.remove()
                    self.preview_circle = None
                self.adding_object = False
                self.pending_object_handle = None
                self.pending_position = None
                self.pending_receptacle = None
                print("Cancelled adding object")
                self.update_object_plot()
            elif self.moving_object:
                if hasattr(self, "preview_circle") and self.preview_circle is not None:
                    self.preview_circle.remove()
                    self.preview_circle = None
                self.moving_object = False
                self.pending_position = None
                self.pending_receptacle = None
                print("Cancelled moving object")
                self.update_object_plot()

        elif event.key in ["enter", "return"]:
            # Confirm placement or move
            if self.adding_object and self.pending_position is not None:
                # Show dialog to select regions and furniture
                world_x, world_y, world_z = self.pending_position
                world_pos = np.array([world_x, world_y, world_z])

                # Get detected regions and furniture
                detected_regions = get_regions_at_position(self.sim, world_pos)
                detected_furniture = None
                if self.pending_receptacle:
                    detected_furniture = get_furniture_name_from_receptacle(
                        self.sim, self.pending_receptacle
                    )

                # Show selection dialog
                (
                    selected_regions,
                    selected_furniture,
                ) = self.show_region_furniture_dialog(
                    detected_regions, detected_furniture
                )

                if selected_regions is None:  # User cancelled
                    print("Cancelled adding object")
                    return

                # Normalize handle to shortname format
                normalized_handle = normalize_object_handle(self.pending_object_handle)

                # Get object class from handle (extract from config or use handle name)
                object_class = self.get_object_class_from_handle(
                    self.pending_object_handle
                )

                # Add to initial_state
                initial_state_entry = {
                    "number": 1,
                    "object_classes": [object_class],
                    "allowed_regions": selected_regions,
                    "furniture_names": selected_furniture
                    if selected_furniture
                    else ["floor"],
                }

                # Initialize initial_state if not present
                if "info" not in self.episode:
                    self.episode["info"] = {}
                if "initial_state" not in self.episode["info"]:
                    self.episode["info"]["initial_state"] = []

                self.episode["info"]["initial_state"].append(initial_state_entry)

                # Create transform matrix for rigid_objs
                transform = np.eye(4)
                transform[:3, 3] = world_pos
                # Initialize rigid_objs if not present
                if "rigid_objs" not in self.episode:
                    self.episode["rigid_objs"] = []
                self.episode["rigid_objs"].append(
                    (normalized_handle, transform.tolist())
                )

                # Add to name_to_receptacle if receptacle is present
                if self.pending_receptacle:
                    # Initialize name_to_receptacle if not present
                    if "name_to_receptacle" not in self.episode:
                        self.episode["name_to_receptacle"] = {}

                    # Format receptacle name for name_to_receptacle
                    receptacle_mapping = format_receptacle_for_name_to_receptacle(
                        self.sim, self.pending_receptacle
                    )
                    if receptacle_mapping:
                        # Format: "object_handle_:0000": "receptacle_mapping"
                        object_key = f"{normalized_handle}_:0000"
                        self.episode["name_to_receptacle"][
                            object_key
                        ] = receptacle_mapping

                # Also add to object_positions for visualization
                self.object_positions.append((normalized_handle, world_pos))

                print(f"✓ Added object {object_class} to initial_state:")
                print(f"  - Regions: {selected_regions}")
                print(
                    f"  - Furniture: {selected_furniture if selected_furniture else ['floor']}"
                )
                if self.pending_receptacle:
                    print(
                        f"  - Receptacle mapping: {normalized_handle}_:0000 -> {self.episode['name_to_receptacle'].get(f'{normalized_handle}_:0000', 'N/A')}"
                    )

                # Remove preview circle
                if hasattr(self, "preview_circle") and self.preview_circle is not None:
                    self.preview_circle.remove()
                    self.preview_circle = None

                self.adding_object = False
                self.pending_object_handle = None
                self.pending_position = None
                self.pending_receptacle = None
                self.update_object_plot()

            elif self.moving_object and self.pending_position is not None:
                # Confirm moving object
                world_x, world_y, world_z = self.pending_position
                new_position = np.array([world_x, world_y, world_z])

                # Update object position
                for i, (obj_handle, _) in enumerate(self.object_positions):
                    if obj_handle == self.selected_object:
                        self.object_positions[i] = (obj_handle, new_position)
                        break

                # Update episode rigid_objs
                for j, (handle, transform) in enumerate(self.episode["rigid_objs"]):
                    if handle == self.selected_object:
                        if isinstance(transform, list):
                            transform = np.array(transform)
                        transform[:3, 3] = new_position
                        self.episode["rigid_objs"][j] = (handle, transform.tolist())
                        break

                print(
                    f"✓ Moved {self.selected_object} to ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"
                )

                # Remove preview circle
                if hasattr(self, "preview_circle") and self.preview_circle is not None:
                    self.preview_circle.remove()
                    self.preview_circle = None

                self.moving_object = False
                self.pending_position = None
                self.pending_receptacle = None
                # Keep object selected so user can continue editing
                # Clear any preview circle
                if hasattr(self, "preview_circle") and self.preview_circle is not None:
                    self.preview_circle.remove()
                    self.preview_circle = None
                self.update_object_plot()

        elif event.key == "s":
            # Save episode
            self.save_episode()

        elif event.key == "q":
            # Quit
            plt.close(self.fig)

    def get_object_class_from_handle(self, handle: str) -> str:
        """Extract object class name from handle."""
        # Try to get from object config
        try:
            otm = self.sim.get_object_template_manager()
            if handle in otm.get_file_template_handles():
                obj_template = otm.get_template_by_handle(handle)
                # Try to get semantic class or use handle name
                if (
                    hasattr(obj_template, "semantic_id")
                    and obj_template.semantic_id > 0
                ):
                    semantic_obj = self.sim.semantic_scene.get_object(
                        obj_template.semantic_id
                    )
                    if semantic_obj and semantic_obj.category:
                        return semantic_obj.category.name().lower().replace(" ", "_")
        except Exception:
            pass

        # Fallback: extract from handle name
        # e.g., "013_apple.object_config.json" -> "apple"
        shortname = normalize_object_handle(handle)
        # Remove numeric prefix if present (e.g., "013_apple" -> "apple")
        parts = shortname.split("_", 1)
        if len(parts) > 1 and parts[0].isdigit():
            return parts[1]
        return shortname

    def show_region_furniture_dialog(
        self, detected_regions: List[str], detected_furniture: Optional[str]
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Show dialog to select regions and furniture names."""
        # Create a new figure for the selection dialog
        dialog_fig = plt.figure(figsize=(14, 10))
        dialog_fig.canvas.manager.set_window_title("Select Regions and Furniture")

        # Layout: regions on left, furniture on right
        ax_regions_title = dialog_fig.add_axes([0.05, 0.85, 0.4, 0.05])
        ax_regions_list = dialog_fig.add_axes([0.05, 0.15, 0.4, 0.65])
        ax_furniture_title = dialog_fig.add_axes([0.55, 0.85, 0.4, 0.05])
        ax_furniture_input = dialog_fig.add_axes([0.55, 0.75, 0.4, 0.05])
        ax_furniture_list = dialog_fig.add_axes([0.55, 0.15, 0.4, 0.55])
        ax_confirm = dialog_fig.add_axes([0.6, 0.05, 0.15, 0.05])
        ax_cancel = dialog_fig.add_axes([0.8, 0.05, 0.15, 0.05])

        # Get all available regions from semantic scene
        all_regions = []
        try:
            for region in self.sim.semantic_scene.regions:
                if region.category is not None:
                    region_name = region.category.name().lower().replace(" ", "_")
                    if region_name not in all_regions:
                        all_regions.append(region_name)
        except Exception:
            pass

        if not all_regions:
            all_regions = detected_regions if detected_regions else ["unknown_region"]

        # Selected items
        selected_regions = detected_regions[:] if detected_regions else []
        furniture_text = detected_furniture if detected_furniture else ""
        selected_furniture_result = [None]
        dialog_closed = [False]

        # Furniture textbox
        furniture_textbox = TextBox(
            ax_furniture_input, "Furniture name: ", initial=furniture_text
        )

        def update_regions_display():
            """Update the regions display."""
            ax_regions_list.clear()
            ax_regions_list.set_xlim(0, 1)
            ax_regions_list.set_ylim(0, 1)
            ax_regions_list.axis("off")

            if not all_regions:
                ax_regions_list.text(
                    0.5, 0.5, "No regions found", ha="center", va="center", fontsize=10
                )
                dialog_fig.canvas.draw()
                return

            y_positions = np.linspace(0.95, 0.05, len(all_regions))
            for i, region in enumerate(all_regions):
                is_selected = region in selected_regions
                color = "lightgreen" if is_selected else "white"
                ax_regions_list.text(
                    0.05,
                    y_positions[i],
                    f"{'[X]' if is_selected else '[ ]'} {region}",
                    fontsize=9,
                    va="center",
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                    picker=True,
                )
            dialog_fig.canvas.draw()

        def on_region_click(event):
            """Handle clicks on region names."""
            if event.inaxes == ax_regions_list:
                for i, region in enumerate(all_regions):
                    y_pos = np.linspace(0.95, 0.05, len(all_regions))[i]
                    if abs(event.ydata - y_pos) < 0.02:
                        if region in selected_regions:
                            selected_regions.remove(region)
                        else:
                            selected_regions.append(region)
                        update_regions_display()
                        break

        def confirm_selection():
            """Confirm the selection."""
            furniture_name = furniture_textbox.text.strip()
            if furniture_name:
                selected_furniture_result[0] = [furniture_name]
            else:
                selected_furniture_result[0] = ["floor"]

            if not selected_regions:
                print(
                    "Warning: No regions selected. Using detected regions or default."
                )
                if detected_regions:
                    selected_regions.extend(detected_regions)
                else:
                    selected_regions.append("unknown_region")

            dialog_closed[0] = True
            plt.close(dialog_fig)

        def cancel_selection():
            """Cancel selection."""
            selected_furniture_result[0] = None
            dialog_closed[0] = True
            plt.close(dialog_fig)

        # Titles
        ax_regions_title.text(
            0.5,
            0.5,
            "Select Regions (click to toggle)",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            transform=ax_regions_title.transAxes,
        )
        ax_regions_title.axis("off")

        ax_furniture_title.text(
            0.5,
            0.5,
            "Furniture Name",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            transform=ax_furniture_title.transAxes,
        )
        ax_furniture_title.axis("off")

        # Furniture list (show detected furniture)
        ax_furniture_list.clear()
        ax_furniture_list.set_xlim(0, 1)
        ax_furniture_list.set_ylim(0, 1)
        ax_furniture_list.axis("off")
        if detected_furniture:
            ax_furniture_list.text(
                0.1,
                0.9,
                f"Detected: {detected_furniture}",
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            )
        ax_furniture_list.text(
            0.1,
            0.7,
            'Enter furniture name above\n(leave empty for "floor")',
            fontsize=9,
            va="top",
            style="italic",
        )

        # Connect events
        dialog_fig.canvas.mpl_connect("button_press_event", on_region_click)
        furniture_textbox.on_submit(lambda x: None)  # Allow Enter to submit

        # Create buttons
        confirm_btn = Button(ax_confirm, "Confirm")
        cancel_btn = Button(ax_cancel, "Cancel")
        confirm_btn.on_clicked(lambda x: confirm_selection())
        cancel_btn.on_clicked(lambda x: cancel_selection())

        # Store to prevent garbage collection
        dialog_fig._confirm_btn = confirm_btn
        dialog_fig._cancel_btn = cancel_btn
        dialog_fig._furniture_textbox = furniture_textbox

        # Initial display
        update_regions_display()
        dialog_fig.canvas.draw()
        dialog_fig.canvas.flush_events()

        # Handle Enter key for furniture textbox
        def on_key(event):
            if event.key == "enter" and event.inaxes == ax_furniture_input:
                confirm_selection()

        dialog_fig.canvas.mpl_connect("key_press_event", on_key)

        # Show dialog
        plt.show(block=False)

        # Wait for dialog to close
        import time

        while plt.fignum_exists(dialog_fig.number):
            dialog_fig.canvas.flush_events()
            time.sleep(0.05)

        # Return results
        if selected_furniture_result[0] is None:
            return None, None

        return selected_regions, selected_furniture_result[0]

    def show_object_selection_dialog(self):
        """Show a GUI dialog with searchable object list."""
        # Create a new figure for the selection dialog
        dialog_fig = plt.figure(figsize=(12, 8))
        dialog_fig.canvas.manager.set_window_title("Select Object to Add")

        ax_search = dialog_fig.add_axes([0.1, 0.85, 0.8, 0.05])
        ax_list = dialog_fig.add_axes([0.1, 0.15, 0.8, 0.65])
        ax_confirm = dialog_fig.add_axes([0.6, 0.05, 0.15, 0.05])
        ax_cancel = dialog_fig.add_axes([0.8, 0.05, 0.15, 0.05])

        # Search textbox
        search_textbox = TextBox(ax_search, "Search: ", initial="")

        # Filtered list
        filtered_objects = self.available_objects[:]
        selected_index = [0]  # Use list to allow modification in nested functions
        dialog_closed = [False]  # Flag to track if dialog should close
        selected_handle_result = [None]  # Store selected handle

        def update_list(search_text=""):
            """Update the displayed list based on search text."""
            ax_list.clear()
            ax_list.set_xlim(0, 1)
            ax_list.set_ylim(0, 1)
            ax_list.axis("off")

            # Filter objects
            search_lower = search_text.lower()
            filtered_objects[:] = [
                obj
                for obj in self.available_objects
                if search_lower in obj.lower()
                or search_lower in obj.split("/")[-1].lower()
            ][
                :200
            ]  # Limit to 200 for performance

            if not filtered_objects:
                ax_list.text(
                    0.5, 0.5, "No objects found", ha="center", va="center", fontsize=12
                )
                dialog_fig.canvas.draw()
                return

            # Display objects
            num_objects = min(len(filtered_objects), 30)  # Show max 30 at a time
            y_positions = np.linspace(0.95, 0.05, num_objects)

            for i, obj_handle in enumerate(filtered_objects[:num_objects]):
                obj_name = obj_handle.split("/")[-1].replace(".object_config.json", "")
                if len(obj_name) > 60:
                    obj_name = obj_name[:57] + "..."

                color = "yellow" if i == selected_index[0] else "white"
                ax_list.text(
                    0.05,
                    y_positions[i],
                    f"{i+1}. {obj_name}",
                    fontsize=10,
                    va="center",
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                    picker=True,
                )

            if len(filtered_objects) > num_objects:
                ax_list.text(
                    0.5,
                    0.02,
                    f"... and {len(filtered_objects) - num_objects} more (use search to filter)",
                    ha="center",
                    fontsize=9,
                    style="italic",
                )

            dialog_fig.canvas.draw()

        def on_search_change(text):
            """Handle search text changes."""
            selected_index[0] = 0
            update_list(text)

        def on_click(event):
            """Handle clicks on object names."""
            if event.inaxes == ax_list:
                # Find which object was clicked
                for i, _obj_handle in enumerate(filtered_objects[:30]):
                    y_pos = np.linspace(0.95, 0.05, min(len(filtered_objects), 30))[i]
                    if abs(event.ydata - y_pos) < 0.02:
                        selected_index[0] = i
                        update_list(search_textbox.text)
                        break

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == "up" and selected_index[0] > 0:
                selected_index[0] -= 1
                update_list(search_textbox.text)
            elif event.key == "down" and selected_index[0] < min(
                len(filtered_objects) - 1, 29
            ):
                selected_index[0] += 1
                update_list(search_textbox.text)
            elif event.key == "enter":
                confirm_selection()

        def confirm_selection():
            """Confirm the selected object."""
            if filtered_objects and 0 <= selected_index[0] < len(filtered_objects):
                selected_handle = filtered_objects[selected_index[0]]
                selected_handle_result[0] = selected_handle
                dialog_closed[0] = True
                plt.close(dialog_fig)

        def cancel_selection():
            """Cancel object selection."""
            dialog_closed[0] = True
            selected_handle_result[0] = None
            plt.close(dialog_fig)

        # Connect events
        search_textbox.on_submit(on_search_change)
        dialog_fig.canvas.mpl_connect("button_press_event", on_click)
        dialog_fig.canvas.mpl_connect("key_press_event", on_key)

        # Create buttons and store them to prevent garbage collection
        confirm_btn = Button(ax_confirm, "Confirm")
        cancel_btn = Button(ax_cancel, "Cancel")

        # Set up button callbacks properly
        def on_confirm_click(event):
            confirm_selection()

        def on_cancel_click(event):
            cancel_selection()

        confirm_btn.on_clicked(on_confirm_click)
        cancel_btn.on_clicked(on_cancel_click)

        # Store buttons to prevent garbage collection
        dialog_fig._confirm_btn = confirm_btn
        dialog_fig._cancel_btn = cancel_btn
        dialog_fig._search_textbox = search_textbox

        # Initial display
        update_list()

        # Instructions
        ax_list.text(
            0.5,
            0.98,
            "Use search box to filter, click to select, or use arrow keys",
            ha="center",
            fontsize=9,
            style="italic",
            transform=ax_list.transAxes,
        )

        # Ensure canvas is drawn
        dialog_fig.canvas.draw()
        dialog_fig.canvas.flush_events()

        # Show dialog without blocking to avoid event loop conflicts
        plt.show(block=False)

        # Wait for dialog to close by polling
        import time

        while plt.fignum_exists(dialog_fig.number):
            dialog_fig.canvas.flush_events()
            time.sleep(0.05)  # Small sleep to avoid busy waiting

        # Process the result after dialog closes
        if selected_handle_result[0] is not None:
            selected_handle = selected_handle_result[0]
            # Store the full handle for display, but we'll normalize when saving
            self.adding_object = True
            self.pending_object_handle = selected_handle
            self.pending_position = None
            obj_name = selected_handle.split("/")[-1].replace(".object_config.json", "")
            print(f"\nSelected: {obj_name}")
            print(
                "Click on map to place object, then press 'Enter' to confirm or 'Esc' to cancel"
            )
            self.update_object_plot()
        else:
            print("Object selection cancelled.")

    def save_episode(self):
        """Save modified episode to file (only the edited episode, not the entire dataset)."""
        if self.output_path is None:
            self.output_path = input(
                "Enter output path (or press Enter to use default): "
            )
            if not self.output_path:
                self.output_path = "edited_episode.json.gz"

        # Normalize all object handles in rigid_objs to shortname format
        normalized_rigid_objs = []
        for handle, transform in self.episode["rigid_objs"]:
            normalized_handle = normalize_object_handle(handle)
            normalized_rigid_objs.append((normalized_handle, transform))
        self.episode["rigid_objs"] = normalized_rigid_objs

        # Create a dataset structure with only this episode
        single_episode_dataset = {
            "config": self.dataset.get("config"),
            "episodes": [self.episode],
        }

        # Save only the single episode
        if self.output_path.endswith(".gz"):
            with gzip.open(self.output_path, "wt") as f:
                json.dump(single_episode_dataset, f, indent=2)
        else:
            with open(self.output_path, "w") as f:
                json.dump(single_episode_dataset, f, indent=2)

        print(f"Saved episode to {self.output_path}")

    def show(self):
        """Show the interactive plot."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive editor for object positions in PARTNR episodes"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the dataset file (.json or .json.gz)",
    )
    parser.add_argument(
        "--episode-id",
        required=True,
        type=int,
        help="Episode ID to edit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for edited episode (default: edited_episode.json.gz)",
    )

    args = parser.parse_args()

    if not HABITAT_AVAILABLE:
        print("Error: Habitat not available. Cannot visualize objects.")
        return

    # Load episode
    print(f"Loading episode {args.episode_id} from {args.dataset}...")
    episode, dataset = load_episode(args.dataset, args.episode_id)

    # Create simulator and get map
    print("Generating top-down map...")
    sim, top_down_map, calibration = create_simulator_for_map(episode)

    # Create editor
    editor = ObjectEditor(episode, dataset, top_down_map, calibration, sim, args.output)

    print("\nInteractive editor ready!")
    print("Controls:")
    print("  - Click object to select")
    print("  - Click empty space to move selected object")
    print("  - 'd' to delete selected object")
    print("  - 'a' to add new object")
    print("  - 's' to save changes")
    print("  - 'q' to quit")
    print()

    # Show interactive plot
    editor.show()

    # Cleanup
    sim.close()
    del sim


if __name__ == "__main__":
    main()
