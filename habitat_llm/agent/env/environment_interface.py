# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict
from termcolor import cprint
import cv2
import gym
import habitat
import imageio
import numpy as np
import torch
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# HABITAT
from habitat_baselines.utils.common import batch_obs, get_num_actions
from habitat_sim.utils.viz_utils import depth_to_rgb

from habitat_llm.agent.env.sensors import SENSOR_MAPPINGS
from habitat_llm.perception import PerceptionObs, PerceptionSim
from habitat_llm.sims.metadata_interface import get_metadata_dict_from_config
from habitat_llm.utils.core import separate_agent_idx

# LOCAL
from habitat_llm.world_model import DynamicWorldGraph, WorldGraph, Furniture, Object

if hasattr(torch, "inference_mode"):
    inference_mode = torch.inference_mode
else:
    inference_mode = torch.no_grad


def camera_spec_to_intrinsics(camera_spec):
    def f(length, fov):
        return length / (2.0 * np.tan(hfov / 2.0))

    hfov = np.deg2rad(float(camera_spec.hfov))
    image_height, image_width = np.array(camera_spec.resolution).tolist()
    fx = f(image_height, hfov)
    fy = f(image_width, hfov)
    cx = image_height / 2.0
    cy = image_width / 2.0
    return np.array([[fx, fy, cx, cy]])


class EnvironmentInterface:
    def __init__(
        self, conf, dataset=None, init_wg=True, init_env=True, gym_habitat_env=None
    ):
        if init_env:
            self.env = habitat.registry.get_env("GymHabitatEnv")(
                config=conf, dataset=dataset
            )
        else:
            if gym_habitat_env is None:
                raise ValueError(
                    "Expected env to be a Habitat Env variable got None instead!"
                )
            self.env = gym_habitat_env
        self.sim = self.env.env.env._env.sim
        self.sim.dynamic_target = np.zeros(3)

        obs = self.env.reset()

        if conf.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda",
                conf.habitat_baselines.torch_gpu_id,
            )

        self.conf = conf
        self.mappings = SENSOR_MAPPINGS
        self._dry_run = self.conf.dry_run

        # Set human and robot agent uids
        self.robot_agent_uid = self.conf.robot_agent_uid
        self.human_agent_uid = self.conf.human_agent_uid

        # merge metadata config and defaults
        self.metadata_dict = get_metadata_dict_from_config(conf.habitat.dataset)

        # Create instance perceptionSim and WorldModel
        # FIXME: below is same as self.wm_update_mode, remove one in favor of other
        self.perception_mode = conf.world_model.update_mode
        if init_wg:
            self.initialize_perception_and_world_graph()

        if "main_agent" in self.conf.trajectory.agent_names:
            self._single_agent_mode = True
        else:
            self._single_agent_mode = False

        self.ppo_cfg = conf.habitat_baselines.rl.ppo

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.obs_transforms = get_active_obs_transforms(conf)

        self.orig_action_space = self.env.original_action_space
        self.observation_space = apply_obs_transforms_obs_space(
            self.observation_space, self.obs_transforms
        )

        self.__get_internal_obs_space()

        self.frames = []
        self.batch = self.__parse_observations(obs)
        # self.reset_environment()

        # Container to store state history of both agents
        self.agent_state_history = defaultdict(list)

        # Container to store actions history of both agents
        self.agent_action_history = defaultdict(list)

        # container to store results from composite skills
        self._composite_action_response = {}
        # a dictionary where key == agent_uid and the value is a tuple with
        # ( "last-action", "arg-string", "result")

        # empty variables to store the trajectory data initialized in
        # setup_logging_for_current_episode when save_trajectory is True
        self.save_trajectory: bool = self.conf.trajectory.save
        self.save_options: list = None
        self.trajectory_agent_names: list = None
        self.trajectory_save_paths: Dict[str, str] = None
        self.trajectory_save_prefix: str = None
        self._trajectory_idx: int = None
        self._setup_current_episode_logging: bool = False

    def initialize_perception_and_world_graph(self):
        """
        This method initializes perception and world graph
        """
        # Create instance of perception
        if self.perception_mode == "gt":
            self.perception = PerceptionSim(self.sim, self.metadata_dict)
        else:
            self.perception = PerceptionObs(self.sim, self.metadata_dict)
        # Set the partial observability flag
        self.partial_obs = self.conf.world_model.partial_obs

        # set update mode flag: str: gt or obs
        self.wm_update_mode: str = self.conf.world_model.update_mode

        # Create instance of the world model
        # static world-graph for full obs setting
        # dynamic world-graph for partial obs setting

        # each agent has its own world-graph
        self.world_graph: Dict[int, Any] = {}
        # Concept graphs always require DynamicWorldGraph, regardless of partial_obs
        if self.partial_obs or self.conf.world_model.type == "concept_graph":
            # Extract DynamicWorldGraph config parameters
            cprint("Using Dynamic WorldGraph for both agents", "green")
            max_detection_distance = getattr(self.conf.world_model, 'max_detection_distance', 0.5)
            use_gt_object_locations = getattr(self.conf.world_model, 'use_gt_object_locations', False)

            self.world_graph = {
                self.robot_agent_uid: DynamicWorldGraph(
                    max_detection_distance=max_detection_distance,
                    use_gt_object_locations=use_gt_object_locations
                ),
                self.human_agent_uid: DynamicWorldGraph(
                    max_detection_distance=max_detection_distance,
                    use_gt_object_locations=use_gt_object_locations
                ),
            }
        else:
            cprint("Using static WorldGraph for both agents", "green")
            self.world_graph = {
                self.robot_agent_uid: WorldGraph(),
                self.human_agent_uid: WorldGraph(),
            }

        # set agent-asymmetry flag if True
        if self.conf.agent_asymmetry:
            for agent_key in self.world_graph:
                self.world_graph[agent_key].agent_asymmetry = True

        # set articulated agents for each dynamic world-graph
        articulated_agents = {}
        for agent_uid in self.world_graph:
            articulated_agents[agent_uid] = self.sim.agents_mgr[
                agent_uid
            ].articulated_agent
        for agent_id in self.world_graph:
            if isinstance(self.world_graph[agent_id], DynamicWorldGraph):
                self.world_graph[agent_id].set_articulated_agents(articulated_agents)

        # maintain a copy of fully-observable world-graph
        self.full_world_graph = WorldGraph()
        most_recent_graph = self.perception.initialize(False)
        self.full_world_graph.update(most_recent_graph, False, "gt", add_only=True)

        # based on the type of world-model being used, setup the data-source
        if self.conf.world_model.type == "concept_graph":
            self.world_graph[self.robot_agent_uid].world_model_type = "non_privileged"
            # initialize the human agent's world-graph from sim with partial observability
            subgraph = self.perception.initialize(partial_obs=self.partial_obs)
            self.world_graph[self.conf.human_agent_uid].update(
                subgraph, self.partial_obs, "gt", add_only=True
            )

            # initialize robot agent's world-graph from CG
            cg_json = None
            cg_json_path = self.conf.world_model.world_model_data_path

            # CG for a given scene should be read from data based on scene-id
            current_episode_metadata = self.env.env.env._env.current_episode
            current_episode_id = current_episode_metadata.episode_id
            current_scene_id = current_episode_metadata.scene_id
            glob_expr = os.path.join(cg_json_path, f"*{current_scene_id}.json")
            cg_file = glob.glob(glob_expr)

            # handle the case if there is not CG for this scene
            if not cg_file:
                raise FileNotFoundError(
                    f"Skipping Episode# {current_episode_id}, Scene# {current_scene_id} as we do not have CG for this",
                )
            if len(cg_file) > 1:
                raise RuntimeError(
                    f"Found more than 1 CG for scene: {current_scene_id}; skipping",
                )
            print(f"Found 1 CG for scene: {current_scene_id}; file: {cg_file[0]}")
            with open(cg_file[0], "r") as f:
                cg_json = json.load(f)

            if not isinstance(
                self.world_graph[self.conf.robot_agent_uid], DynamicWorldGraph
            ):
                raise ValueError(
                    "Expected robot's world-graph to be of type DynamicWorldGraph, however found: ",
                    type(self.world_graph[self.conf.robot_agent_uid]),
                )
            self.world_graph[self.conf.robot_agent_uid].create_cg_edges(
                cg_json, include_objects=self.conf.world_model.include_objects
            )

            # Choose between spatial data reassignment or full GT structure replacement
            use_gt_static = self.conf.world_model.get("use_gt_static_structure", False)

            if use_gt_static:
                # Replace CG furniture/floors/rooms with GT while keeping objects discoverable
                # This gives non-privileged planners accurate spatial structure without GT object positions
                # ~ GT graph but can use observations for object additions
                cprint("Using GT static structure (furniture/floors/rooms) for non-privileged graph", "green")
                self.world_graph[self.conf.robot_agent_uid].replace_cg_with_gt_static_structure(
                    gt_world_graph=self.full_world_graph,
                    verbose=False
                )
            else:
                # Original behavior: reassign CG furniture using GT room boundaries extracted by the helper method
                # Extract spatial_data for rooms from the full world graph
                cprint("Reassigning furniture to rooms using spatial data from full world graph", "green")
                spatial_data = self._extract_room_spatial_data_from_world_graph(self.full_world_graph)
                self.world_graph[self.conf.robot_agent_uid].reassign_furniture_to_rooms_from_spatial_data(
                    spatial_data,
                    verbose=False
                )

            self.world_graph[self.conf.robot_agent_uid].initialize_agent_nodes(subgraph)
            # ! TODO: Check if we need to set sim handles for non-privileged graph
            # self.world_graph[
            #     self.robot_agent_uid
            # ]._set_sim_handles_for_non_privileged_graph(self.perception)
        elif self.conf.world_model.type == "gt_graph":
            subgraph = self.perception.initialize(self.partial_obs)
            # Get ground truth subgraph from the current observations.
            # since the graph is being initialized, we only add the new nodes and edges
            for agent_key in self.world_graph:
                self.world_graph[agent_key].update(
                    subgraph, self.partial_obs, self.wm_update_mode, add_only=True
                )
        else:
            raise ValueError(
                f"World model not implemented for type: {self.conf.world_model.type}"
            )

        return

    def get_observations(self):
        """
        Obtains the environment observations. In the form of a dictionary of tensors.
        """
        return self.batch

    def parse_observations(self, obs):
        # obs: Dict of all observations from everything habitat env
        return self.__parse_observations(obs)

    def reset_environment(self, move_to_next_episode=True, episode_id=None):
        """
        Resets the environment, moving to the next episode and obtaining a new set of observations
        :param move_to_next_episode: by default, reset moves to the next episode. If set to False, will reset the environment to the same episode.
        :param episode_id: If set, reset the environment to a given episode id. Otherwise, moves to the next episode.
        """
        if not move_to_next_episode:
            # We set this variable to reset the environment but stay in the same episode
            self.env.env.env._env.current_episode = (
                self.env.env.env._env.current_episode
            )

        if episode_id is not None:
            assert type(episode_id) == str
            episode_interest = [
                epi
                for epi in self.env.env.env._env._dataset.episodes
                if epi.episode_id == episode_id
            ][0]
            self.env.env.env._env.current_episode = episode_interest

        obs = self.env.reset()
        self.batch = self.__parse_observations(obs)
        self.initialize_perception_and_world_graph()
        self.sim = self.env.env.env._env.sim
        self.reset_internals()
        self.video_name = "debug.mp4"

        # if self.frames != []:
        #     self.__make_video()

        self.frames = []

        # Container to store state history of agents
        self.agent_state_history = defaultdict(list)

        # Container to store action history of agents
        self.agent_action_history = defaultdict(list)

        # reset episode logging to create dir structure for new episode
        self.reset_logging()

    def reset_internals(self):
        self.recurrent_hidden_states = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            self.conf.habitat_baselines.rl.ddppo.num_recurrent_layers
            * 2,  # TODO why 2?
            self.ppo_cfg.hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            *(get_num_actions(self.action_space),),
            device=self.device,
            dtype=torch.float,
        )
        self.not_done_masks = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )

    def reset_logging(self):
        # empty variables to store the trajectory data initialized in
        # setup_logging_for_current_episode when save_trajectory is True
        self.save_options = None
        self.trajectory_agent_names = None
        self.trajectory_save_paths = None
        self.trajectory_save_prefix = None
        self._trajectory_idx = None
        self._setup_current_episode_logging = False

    def reset_composite_action_response(self):
        """resets _composite_action_response to empty"""
        self._composite_action_response = {}

    def get_room_bounds_from_simulator(self, room_names: list) -> dict:
        """
        Extract un-overlapping room bounding boxes from simulator's semantic scene.
        Also creates a visualization showing room bounds overlaid on the scene.
        
        Args:
            room_names: List of GT room names from the world graph
            
        Returns:
            Dictionary mapping room names to their bounds, center, and size
        """
        try:
            # Get semantic scene from the simulator
            semantic_scene = self.sim.semantic_scene
            if not semantic_scene or len(semantic_scene.regions) == 0:
                cprint("WARNING: No semantic scene regions found in simulator!", "red")
                return {}

            room_bounds_data = {}

            # Debug: print all available regions
            cprint(f"Semantic scene has {len(semantic_scene.regions)} regions", "cyan")
            for region in semantic_scene.regions:
                region_full_name = region.category.name()
                cprint(f"  Region: {region_full_name}", "cyan")

            # Map rooms to semantic regions based on POSITION
            # Each room in the world graph has a position - find which semantic region contains it
            from habitat_llm.world_model import Room

            all_rooms = self.full_world_graph.get_all_nodes_of_type(Room)
            room_dict = {room.name: room for room in all_rooms}

            for room_name in room_names:
                # Get room position from the world graph (GT)
                if "other_room" in room_name:
                    continue

                if room_name not in room_dict:
                    cprint(f"  WARNING: Room '{room_name}' not found in world graph", "yellow")
                    continue

                room_node = room_dict[room_name]
                if 'translation' not in room_node.properties:
                    cprint(f"  WARNING: Room '{room_name}' has no position in world graph", "yellow")
                    cprint(f"    Properties: {room_node.properties}", "yellow")
                    continue
                room_position = room_node.properties['translation']
                room_base = room_name.split("_")[0].lower()  # e.g., "bathroom" from "bathroom_1"

                # Find which semantic region contains this position AND has matching name
                matching_region = None
                for region in semantic_scene.regions:
                    aabb = region.aabb
                    min_pt = aabb.min
                    max_pt = aabb.max

                    # First check: position must be inside this region's AABB
                    position_inside = (min_pt[0] <= room_position[0] <= max_pt[0]
                                      and min_pt[1] <= room_position[1] <= max_pt[1]
                                      and min_pt[2] <= room_position[2] <= max_pt[2])

                    if not position_inside:
                        continue  # Skip if position doesn't match

                    # Second check: name must match
                    region_name = region.category.name().split("/")[0].replace(" ", "_").lower()
                    name_match = (region_name == room_base
                                 or region_name in room_base
                                 or room_base in region_name)

                    if position_inside and name_match:
                        matching_region = region
                        break  # Found a match with both conditions

                if matching_region:
                    aabb = matching_region.aabb
                    center = aabb.center()
                    size = aabb.size()
                    min_pt = aabb.min
                    max_pt = aabb.max

                    # Create polygon from AABB corners (XZ plane)
                    polygon_points = [
                        [float(min_pt[0]), float(min_pt[2])],  # bottom-left
                        [float(max_pt[0]), float(min_pt[2])],  # bottom-right
                        [float(max_pt[0]), float(max_pt[2])],  # top-right
                        [float(min_pt[0]), float(max_pt[2])]   # top-left
                    ]

                    room_bounds_data[room_name] = {
                        "bounds": {
                            "min": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
                            "max": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
                        },
                        "center": [float(center[0]), float(center[1]), float(center[2])],
                        "size": [float(size[0]), float(size[1]), float(size[2])],
                        "polygon": polygon_points  # XZ coordinates of AABB as polygon
                    }
                    region_name_matched = matching_region.category.name().split("/")[0]
                    # cprint(f"  Matched '{room_name}' -> semantic region '{region_name_matched}'", "green")

            # Debug: Print which rooms were matched
            matched_rooms = set(room_bounds_data.keys())
            unmatched_rooms = set(room_names) - matched_rooms

            # cprint(f"Extracted bounds for {len(room_bounds_data)}/{len(room_names)} rooms from simulator", "green")
            if unmatched_rooms:
                cprint(f"WARNING: Could not find bounds for rooms: {unmatched_rooms}", "yellow")

                # Fallback: match unmatched rooms by name only (no position check)
                cprint(f"Attempting fallback matching for {len(unmatched_rooms)} unmatched rooms...", "cyan")
                for room_name in unmatched_rooms:
                    # if "other_room" in room_name:
                    #     continue

                    room_base = room_name.split("_")[0].lower()

                    # Find semantic region that matches by name
                    matching_region = None
                    for region in semantic_scene.regions:
                        region_name = region.category.name().split("/")[0].replace(" ", "_").lower()
                        name_match = (region_name == room_base
                                     or region_name in room_base
                                     or room_base in region_name)

                        if name_match:
                            # Check if this region is already assigned
                            already_used = False
                            for assigned_room_data in room_bounds_data.values():
                                if (assigned_room_data["bounds"]["min"] == [float(region.aabb.min[0]), float(region.aabb.min[1]), float(region.aabb.min[2])]
                                    and assigned_room_data["bounds"]["max"] == [float(region.aabb.max[0]), float(region.aabb.max[1]), float(region.aabb.max[2])]):
                                    already_used = True
                                    break

                            if not already_used:
                                matching_region = region
                                break

                    if matching_region:
                        aabb = matching_region.aabb
                        center = aabb.center()
                        size = aabb.size()
                        min_pt = aabb.min
                        max_pt = aabb.max

                        # Create polygon from AABB corners (XZ plane)
                        polygon_points = [
                            [float(min_pt[0]), float(min_pt[2])],  # bottom-left
                            [float(max_pt[0]), float(min_pt[2])],  # bottom-right
                            [float(max_pt[0]), float(max_pt[2])],  # top-right
                            [float(min_pt[0]), float(max_pt[2])]   # top-left
                        ]

                        room_bounds_data[room_name] = {
                            "bounds": {
                                "min": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
                                "max": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
                            },
                            "center": [float(center[0]), float(center[1]), float(center[2])],
                            "size": [float(size[0]), float(size[1]), float(size[2])],
                            "polygon": polygon_points  # XZ coordinates of AABB as polygon
                        }
                        region_name_matched = matching_region.category.name().split("/")[0]
                        cprint(f"  Fallback matched '{room_name}' -> semantic region '{region_name_matched}' (name only)", "yellow")

                        # Save properties to the GT world graph
                        if room_name in room_dict:
                            room_node = room_dict[room_name]
                            room_node.properties["translation"] = [float(center[0]), float(center[1]), float(center[2])]
                            room_node.properties["size"] = [float(size[0]), float(size[1]), float(size[2])]
                            room_node.properties["bounds_min"] = [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])]
                            room_node.properties["bounds_max"] = [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
                            cprint(f"    Updated properties for '{room_name}' in world graph", "green")

                cprint(f"After fallback: {len(room_bounds_data)}/{len(room_names)} rooms have bounds", "green")

            # Resolve overlaps between rooms
            room_bounds_data = self._resolve_room_overlaps(room_bounds_data)

            # Create visualization of room bounds
            self._visualize_room_bounds(room_bounds_data)

            return room_bounds_data

        except Exception as e:
            cprint(f"WARNING: Could not extract room bounds from simulator: {e}", "red")
            import traceback
            traceback.print_exc()
            return {}

    def _check_overlap(self, room1_data, room2_data):
        """
        Check if two rooms overlap using their polygon representations.
        
        Args:
            room1_data: dict with 'polygon' key (list of [x, z] points)
            room2_data: dict with 'polygon' key (list of [x, z] points)
        
        Returns:
            bool: True if polygons intersect
        """
        try:
            from shapely.geometry import Polygon

            poly1 = Polygon(room1_data['polygon'])
            poly2 = Polygon(room2_data['polygon'])

            return poly1.intersects(poly2)
        except Exception as e:
            # Fallback to AABB check if polygon fails
            bounds1 = room1_data['bounds']
            bounds2 = room2_data['bounds']
            x_overlap = bounds1['min'][0] < bounds2['max'][0] and bounds1['max'][0] > bounds2['min'][0]
            z_overlap = bounds1['min'][2] < bounds2['max'][2] and bounds1['max'][2] > bounds2['min'][2]
            return x_overlap and z_overlap

    def _shrink_room_to_exclude(self, large_room_data, small_room_data):
        """
        Shrink the larger room's polygon to exclude the smaller room using geometric difference.
        
        Args:
            large_room_data: dict with 'polygon', 'bounds', 'center', 'size' keys
            small_room_data: dict with 'polygon', 'bounds', 'center', 'size' keys
        
        Returns:
            dict: Modified large_room_data with updated polygon and bounds
        """
        try:
            from shapely.geometry import Polygon
            import numpy as np

            large_poly = Polygon(large_room_data['polygon'])
            small_poly = Polygon(small_room_data['polygon'])

            # Compute geometric difference (large polygon minus small polygon)
            diff_poly = large_poly.difference(small_poly)

            # Handle multipolygon result (take largest component)
            if diff_poly.is_empty:
                return large_room_data

            if diff_poly.geom_type == 'MultiPolygon':
                # Take the largest polygon
                diff_poly = max(diff_poly.geoms, key=lambda p: p.area)

            # Extract new polygon vertices
            if diff_poly.geom_type == 'Polygon':
                new_polygon = [[float(x), float(z)] for x, z in diff_poly.exterior.coords[:-1]]
            else:
                # Fallback if not a polygon
                return large_room_data

            # Recalculate bounds from new polygon
            polygon_array = np.array(new_polygon)
            new_min_x, new_min_z = polygon_array.min(axis=0)
            new_max_x, new_max_z = polygon_array.max(axis=0)

            # Update room data
            new_bounds = {
                'min': [new_min_x, large_room_data['bounds']['min'][1], new_min_z],
                'max': [new_max_x, large_room_data['bounds']['max'][1], new_max_z]
            }

            new_center = [
                (new_min_x + new_max_x) / 2,
                large_room_data['center'][1],
                (new_min_z + new_max_z) / 2
            ]

            new_size = [
                new_max_x - new_min_x,
                large_room_data['size'][1],
                new_max_z - new_min_z
            ]

            return {
                'bounds': new_bounds,
                'center': new_center,
                'size': new_size,
                'polygon': new_polygon
            }

        except Exception as e:
            cprint(f"WARNING: Polygon shrink failed, keeping original: {e}", "yellow")
            return large_room_data

    def _resolve_room_overlaps(self, room_bounds_data):
        """
        Resolve overlaps between rooms by shrinking larger rooms to exclude smaller ones.
        Iteratively processes rooms from smallest to largest.
        
        Args:
            room_bounds_data: dict mapping room_name -> {bounds, center, size}
        
        Returns:
            dict: Modified room_bounds_data with non-overlapping rooms
        """
        print("\nResolving room overlaps...")

        # Calculate XZ plane area for each room
        room_areas = {}
        for room_name, data in room_bounds_data.items():
            bounds = data['bounds']
            area = (bounds['max'][0] - bounds['min'][0]) * (bounds['max'][2] - bounds['min'][2])
            room_areas[room_name] = area

        # Sort rooms by area (smallest first)
        sorted_rooms = sorted(room_areas.items(), key=lambda x: x[1])
        room_order = [name for name, _ in sorted_rooms]

        print(f"Room sizes (XZ area): {len(room_order)} rooms")

        # Iteratively resolve overlaps
        max_iterations = 10  # Arbitrary limit to prevent infinite loops
        for iteration in range(max_iterations):
            overlaps_found = 0

            # Check each pair of rooms
            for i, small_room in enumerate(room_order):
                for large_room in room_order[i+1:]:
                    small_data = room_bounds_data[small_room]
                    large_data = room_bounds_data[large_room]

                    # Check if they overlap (using polygons)
                    if self._check_overlap(small_data, large_data):
                        overlaps_found += 1

                        # Shrink larger room to exclude smaller room (polygon difference)
                        new_large_data = self._shrink_room_to_exclude(large_data, small_data)

                        # Update room_bounds_data with new polygon and bounds
                        room_bounds_data[large_room] = new_large_data

            print(f"Iteration {iteration + 1}: Found and resolved {overlaps_found} overlaps")

            # If no overlaps found, we're done
            if overlaps_found == 0:
                print("All overlaps resolved!")
                break

        if overlaps_found > 0:
            print(f"Warning: Some overlaps may remain after {max_iterations} iterations")

        return room_bounds_data

    def _visualize_room_bounds(self, room_bounds_data: dict):
        """
        Create a simple visualization showing colored room bounding boxes with labels.
        
        Args:
            room_bounds_data: Dictionary mapping room names to their bounds
        """
        try:
            if not room_bounds_data:
                return

            # Create figure with white background
            _, ax = plt.subplots(figsize=(16, 12))
            ax.set_facecolor('white')

            # Generate distinct colors for each room
            colors = plt.cm.tab20(np.linspace(0, 1, len(room_bounds_data)))

            # Find overall bounds to set axes limits
            all_mins = []
            all_maxs = []
            for room_name, data in room_bounds_data.items():
                bounds = data["bounds"]
                all_mins.append(bounds["min"])
                all_maxs.append(bounds["max"])

            all_mins = np.array(all_mins)
            all_maxs = np.array(all_maxs)

            # Get X-Z bounds (top-down view, Y is vertical)
            x_min, x_max = all_mins[:, 0].min(), all_maxs[:, 0].max()
            z_min, z_max = all_mins[:, 2].min(), all_maxs[:, 2].max()

            # Add padding
            padding = 2.0
            x_min -= padding
            x_max += padding
            z_min -= padding
            z_max += padding

            # Draw each room's polygon or bounding box
            for idx, (room_name, data) in enumerate(sorted(room_bounds_data.items())):
                # Get polygon if available, otherwise use AABB
                polygon = data.get("polygon", [])

                if len(polygon) >= 3:
                    # Draw polygon
                    from matplotlib.patches import Polygon as MPLPolygon
                    poly_patch = MPLPolygon(
                        polygon,
                        linewidth=2,
                        edgecolor=colors[idx],
                        facecolor=colors[idx],
                        alpha=0.4,
                        label=room_name
                    )
                    ax.add_patch(poly_patch)

                    # Draw border
                    border_patch = MPLPolygon(
                        polygon,
                        linewidth=2,
                        edgecolor=colors[idx],
                        facecolor='none',
                        alpha=1.0
                    )
                    ax.add_patch(border_patch)

                    # Calculate polygon centroid for label
                    poly_array = np.array(polygon)
                    center_x = poly_array[:, 0].mean()
                    center_z = poly_array[:, 1].mean()
                else:
                    # Fallback to AABB rectangle
                    bounds = data["bounds"]
                    bbox_min = bounds["min"]
                    bbox_max = bounds["max"]

                    x_min_room = bbox_min[0]
                    z_min_room = bbox_min[2]
                    width = bbox_max[0] - bbox_min[0]
                    depth = bbox_max[2] - bbox_min[2]

                    # Draw filled rectangle
                    rect = patches.Rectangle(
                        (x_min_room, z_min_room),
                        width,
                        depth,
                        linewidth=2,
                        edgecolor=colors[idx],
                        facecolor=colors[idx],
                        alpha=0.4,
                        label=room_name
                    )
                    ax.add_patch(rect)

                    # Draw border
                    border = patches.Rectangle(
                        (x_min_room, z_min_room),
                        width,
                        depth,
                        linewidth=2,
                        edgecolor=colors[idx],
                        facecolor='none',
                        alpha=1.0
                    )
                    ax.add_patch(border)

                    center_x = (bbox_min[0] + bbox_max[0]) / 2
                    center_z = (bbox_min[2] + bbox_max[2]) / 2

                # Add room label at center
                ax.text(
                    center_x,
                    center_z,
                    room_name,
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=colors[idx], linewidth=2)
                )

                print(f"  Added room '{room_name}' at center ({center_x:.2f}, {center_z:.2f})")

            # Add furniture markers from concept graph
            try:

                # Get concept graph for the robot agent
                concept_graph = self.world_graph.get(self.conf.robot_agent_uid)
                if concept_graph is None:
                    cprint("WARNING: Could not access concept graph for furniture markers", "yellow")
                else:
                    all_furniture = concept_graph.get_all_furnitures()

                    cprint(f"\nAdding {len(all_furniture)} furniture markers from concept graph...", "cyan")

                    furniture_count = 0
                    for fur_node in all_furniture:
                        fur_pos = fur_node.properties.get('translation')
                        if fur_pos is not None:
                            fur_x, fur_z = fur_pos[0], fur_pos[2]

                            # Draw small dot for furniture (red for CG)
                            ax.plot(fur_x, fur_z, 'o', color='darkred', markersize=5, zorder=10, label='CG Furniture' if furniture_count == 0 else '')

                            # Add furniture name above the dot
                            ax.text(
                                fur_x,
                                fur_z + 0.2,  # Offset above the dot
                                fur_node.name,
                                ha='center',
                                va='bottom',
                                fontsize=7,
                                color='darkred',
                                weight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='darkred', alpha=0.85, linewidth=0.8)
                            )
                            furniture_count += 1

                    cprint(f"✓ Added {furniture_count} furniture markers from concept graph", "green")

                # Add GT furniture markers
                all_gt_furniture = self.full_world_graph.get_all_nodes_of_type(Furniture)

                cprint(f"\nAdding {len(all_gt_furniture)} furniture markers from GT graph...", "cyan")

                gt_furniture_count = 0
                for fur_node in all_gt_furniture:
                    fur_pos = fur_node.properties.get('translation')
                    if fur_pos is not None:
                        fur_x, fur_z = fur_pos[0], fur_pos[2]

                        # Draw small dot for GT furniture (blue for GT)
                        ax.plot(fur_x, fur_z, 's', color='darkblue', markersize=5, zorder=10, label='GT Furniture' if gt_furniture_count == 0 else '')

                        # Add furniture name below the dot
                        ax.text(
                            fur_x,
                            fur_z - 0.2,  # Offset below the dot
                            fur_node.name,
                            ha='center',
                            va='top',
                            fontsize=7,
                            color='darkblue',
                            weight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='darkblue', alpha=0.85, linewidth=0.8)
                        )
                        gt_furniture_count += 1

                cprint(f"✓ Added {gt_furniture_count} furniture markers from GT graph", "green")

            except Exception as e:
                cprint(f"WARNING: Could not add furniture markers: {e}", "yellow")
                import traceback
                traceback.print_exc()

            # Set axis limits and labels
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            ax.set_xlabel('X (meters)', fontsize=12)
            ax.set_ylabel('Z (meters)', fontsize=12)
            ax.set_title('Room Bounding Boxes - Top-Down View', fontsize=14, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)

            # Add legend (limit to 20 rooms to avoid clutter)
            if len(room_bounds_data) <= 20:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            output_dir = self.conf.paths.results_dir + "/visualizations"
            os.makedirs(output_dir, exist_ok=True)

            # Get episode ID if available, otherwise use generic name
            try:
                episode_id = self.env.env.env._env.current_episode.episode_id
                output_path_combined = os.path.join(output_dir, f"room_bounds_cg_gt_{episode_id}.png")
            except:
                output_path_combined = os.path.join(output_dir, "room_bounds_cg_gt.png")

            plt.tight_layout()
            plt.savefig(output_path_combined, dpi=150, bbox_inches='tight')

            cprint(f"✓ Saved combined CG+GT visualization to: {output_path_combined}", "green")

            # Create GT-only visualization
            try:
                fig_gt, ax_gt = plt.subplots(figsize=(16, 12))
                ax_gt.set_facecolor('white')

                # Redraw rooms
                for idx, (room_name, data) in enumerate(sorted(room_bounds_data.items())):
                    polygon = data.get("polygon", [])

                    if len(polygon) >= 3:
                        from matplotlib.patches import Polygon as MPLPolygon
                        poly_patch = MPLPolygon(polygon, linewidth=2, edgecolor=colors[idx], facecolor=colors[idx], alpha=0.4, label=room_name)
                        ax_gt.add_patch(poly_patch)
                        border_patch = MPLPolygon(polygon, linewidth=2, edgecolor=colors[idx], facecolor='none', alpha=1.0)
                        ax_gt.add_patch(border_patch)
                        poly_array = np.array(polygon)
                        center_x = poly_array[:, 0].mean()
                        center_z = poly_array[:, 1].mean()
                    else:
                        bounds = data["bounds"]
                        bbox_min = bounds["min"]
                        bbox_max = bounds["max"]
                        x_min_room = bbox_min[0]
                        z_min_room = bbox_min[2]
                        width = bbox_max[0] - bbox_min[0]
                        depth = bbox_max[2] - bbox_min[2]
                        rect = patches.Rectangle((x_min_room, z_min_room), width, depth, linewidth=2, edgecolor=colors[idx], facecolor=colors[idx], alpha=0.4, label=room_name)
                        ax_gt.add_patch(rect)
                        border = patches.Rectangle((x_min_room, z_min_room), width, depth, linewidth=2, edgecolor=colors[idx], facecolor='none', alpha=1.0)
                        ax_gt.add_patch(border)
                        center_x = (bbox_min[0] + bbox_max[0]) / 2
                        center_z = (bbox_min[2] + bbox_max[2]) / 2

                    ax_gt.text(center_x, center_z, room_name, ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=colors[idx], linewidth=2))

                # Add GT furniture only
                all_gt_furniture = self.full_world_graph.get_all_nodes_of_type(Furniture)
                for fur_node in all_gt_furniture:
                    fur_pos = fur_node.properties.get('translation')
                    if fur_pos is not None:
                        fur_x, fur_z = fur_pos[0], fur_pos[2]
                        ax_gt.plot(fur_x, fur_z, 'o', color='darkblue', markersize=5, zorder=10)
                        ax_gt.text(fur_x, fur_z + 0.2, fur_node.name, ha='center', va='bottom', fontsize=7, color='darkblue', weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='darkblue', alpha=0.85, linewidth=0.8))

                # Add GT objects
                all_gt_objects = self.full_world_graph.get_all_nodes_of_type(Object)
                for obj_node in all_gt_objects:
                    obj_pos = obj_node.properties.get('translation')
                    if obj_pos is not None:
                        obj_x, obj_z = obj_pos[0], obj_pos[2]
                        ax_gt.plot(obj_x, obj_z, '^', color='darkgreen', markersize=5, zorder=10)
                        ax_gt.text(obj_x, obj_z - 0.2, obj_node.name, ha='center', va='top', fontsize=7, color='darkgreen', weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='darkgreen', alpha=0.85, linewidth=0.8))

                ax_gt.set_xlim(x_min, x_max)
                ax_gt.set_ylim(z_min, z_max)
                ax_gt.set_xlabel('X (meters)', fontsize=12)
                ax_gt.set_ylabel('Z (meters)', fontsize=12)
                ax_gt.set_title('Room Bounding Boxes with GT Furniture and Objects - Top-Down View', fontsize=14, fontweight='bold')
                ax_gt.set_aspect('equal', adjustable='box')
                ax_gt.grid(True, alpha=0.3)

                if len(room_bounds_data) <= 20:
                    ax_gt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

                try:
                    episode_id = self.env.env.env._env.current_episode.episode_id
                    output_path_gt = os.path.join(output_dir, f"room_bounds_gt_only_{episode_id}.png")
                except:
                    output_path_gt = os.path.join(output_dir, "room_bounds_gt_only.png")

                plt.tight_layout()
                plt.savefig(output_path_gt, dpi=150, bbox_inches='tight')
                plt.close(fig_gt)

                cprint(f"✓ Saved GT-only visualization to: {output_path_gt}", "green")
            except Exception as e:
                cprint(f"WARNING: Could not create GT-only visualization: {e}", "yellow")

            plt.close()

        except Exception as e:
            cprint(f"WARNING: Could not create room bounds visualization: {e}", "yellow")
            import traceback
            traceback.print_exc()

    def _extract_room_spatial_data_from_world_graph(self, world_graph: WorldGraph) -> Dict[str, Any]:
        """
        Extract room spatial data (position, size, bbox) from the full world graph.
        Gets room bounding boxes from the simulator's semantic scene.
        
        Args:
            world_graph: The full world graph with ground truth room information
            
        Returns:
            Dictionary with 'rooms' key containing room spatial data
        """
        from habitat_llm.world_model import Room

        spatial_data = {"rooms": {}}

        # Get all rooms from the world graph
        all_rooms = world_graph.get_all_nodes_of_type(Room)
        room_names = [room.name for room in all_rooms]

        # Get room bounds from simulator's semantic scene
        room_bounds_data = self.get_room_bounds_from_simulator(room_names)

        for room_node in all_rooms:
            room_name = room_node.name
            room_props = room_node.properties

            # Get position from room properties (fallback)
            position = room_props.get('translation', [0, 0, 0])

            room_bounds = room_bounds_data.get(room_name, {})

            if room_bounds:
                # Use simulator-provided bounds
                bounds = room_bounds.get("bounds", {})
                bbox_min = bounds.get("min")
                bbox_max = bounds.get("max")
                polygon = room_bounds.get("polygon", [])

                if bbox_min and bbox_max:
                    bbox_min = np.array(bbox_min)
                    bbox_max = np.array(bbox_max)
                    size = bbox_max - bbox_min
                    # Use center from bounds
                    position = room_bounds.get("center", position)
                else:
                    size = [0, 0, 0]
            else:
                # No bounds from simulator - use zero size
                size = [0, 0, 0]
                polygon = []

            spatial_data["rooms"][room_name] = {
                "name": room_name,
                "position": position if isinstance(position, list) else position.tolist(),
                "size": size if isinstance(size, list) else size.tolist(),
                "polygon": polygon,  # XZ plane polygon vertices
                "room_type": room_props.get('type', 'unknown'),
                "rotation": room_props.get('rotation', [0, 0, 0, 1])
            }

        if not spatial_data["rooms"]:
            cprint("WARNING: No rooms found in world graph!", "red")

        return spatial_data

    def setup_logging_for_current_episode(self):
        """
        book-keeping to dump out trajectories of given agents
        see config: conf/trajectory/trajectory_logger.yaml for details
        """
        current_episode = self.env.env.env._env.current_episode
        self.save_trajectory = self.conf.trajectory.save
        self.save_options = []
        self.trajectory_agent_names = []
        if self.save_trajectory:
            self.trajectory_agent_names = self.conf.trajectory.agent_names
            assert len(self.trajectory_agent_names) > 0
            self.trajectory_save_prefix = (
                self.conf.trajectory.save_path
                + f"epidx_{current_episode.episode_id}_scene_{current_episode.scene_id}"
            )
            self.save_options = self.conf.trajectory.save_options
            self._trajectory_idx = 0
            self.trajectory_save_paths = {}

            # create a parent directory for given episode/scene combo
            # then create agent-specific directories within it for each agent
            for curr_agent, camera_source in zip(
                self.trajectory_agent_names, self.conf.trajectory.camera_prefixes
            ):
                # sensor_uuid is different depending upon if this is a single-agent
                # or multi-agent planning problem; we expect config to send in
                # consistent naming here
                if self._single_agent_mode:
                    sensor_uuid = f"{camera_source}_rgb"
                else:
                    sensor_uuid = f"{curr_agent}_{camera_source}_rgb"
                self.trajectory_save_paths[curr_agent] = os.path.join(
                    self.trajectory_save_prefix, curr_agent
                )
                if not os.path.exists(self.trajectory_save_paths[curr_agent]):
                    os.makedirs(self.trajectory_save_paths[curr_agent])

                    # save intrinsics for current agent
                    intrinsics_array = camera_spec_to_intrinsics(
                        self.sim.agents[0]._sensors[sensor_uuid].specification()
                    )
                    np.save(
                        os.path.join(
                            self.trajectory_save_paths[curr_agent], "intrinsics.npy"
                        ),
                        intrinsics_array,
                    )

                    # create other sub-directories
                    if "rgb" in self.save_options:
                        os.makedirs(
                            os.path.join(self.trajectory_save_paths[curr_agent], "rgb")
                        )
                    if "depth" in self.save_options:
                        os.makedirs(
                            os.path.join(
                                self.trajectory_save_paths[curr_agent], "depth"
                            )
                        )
                    if "panoptic" in self.save_options:
                        os.makedirs(
                            os.path.join(
                                self.trajectory_save_paths[curr_agent], "panoptic"
                            )
                        )
                    os.makedirs(
                        os.path.join(self.trajectory_save_paths[curr_agent], "pose")
                    )

    def get_final_action_vector(
        self, low_level_actions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Takes in low_level_actions and returns a joint-space final_action_vector over all agents
        """

        # Get list of action tensors
        low_level_action_list = list(low_level_actions.values())

        # Make sure that the actions are never None
        if any(action is None for action in low_level_action_list):
            raise ValueError("low level actions cannot be None!")

        # Construct final actions vector
        if len(low_level_action_list) == 0:
            raise Exception("Cannot step through environment without low level actions")
        if len(low_level_action_list) == 1:
            final_action_vector = low_level_action_list[0]
        elif len(low_level_action_list) == 2:
            final_action_vector = low_level_action_list[0] + low_level_action_list[1]

        return final_action_vector

    def update_world_graphs(self, obs: Dict[str, Any]):
        """
        Simulates perception step using sim for GT condition and observations for non-GT condition
        Additionally, saves this trajectory step if trajectory_logger is enabled
        """

        # Update fully observed world graph (ground truth)
        # This graph is used when planner is working under full observability
        # THis graph is also used by skills to check if a certain furniture is articulated or not etc.
        most_recent_graph = self.perception.get_recent_graph()
        self.full_world_graph.update(
            most_recent_graph, partial_obs=False, update_mode="gt"
        )

        # Update agents world graphs using concept graph
        if self.conf.world_model.type == "concept_graph" and isinstance(
            self.perception, PerceptionObs
        ):
            self.update_world_graphs_using_concept_graph(obs)

        # Update agents world graphs using simulator
        elif self.conf.world_model.type == "gt_graph" and not isinstance(
            self.perception, PerceptionObs
        ):
            self.update_world_graphs_using_sim(obs)

        # if applicable save the data from trajectory step
        self.save_trajectory_step(obs)

        return

    def update_world_graphs_using_sim(self, obs):
        """
        This method updates world graphs for both agents using
        simulated perception and simulated graph
        """

        # Case 1: FULL OBSERVABILITY
        # Set both agents graphs equal to full world graph
        if not self.partial_obs:
            for agent_uid in self.world_graph:
                self.world_graph[agent_uid] = self.full_world_graph

        # Case 2: PARTIAL OBSERVABILITY
        # Update both graphs using both human and robot observations
        else:
            # Get robots subgraph using both human and robot observations
            most_recent_robot_subgraph = self.perception.get_recent_subgraph(
                [str(self.robot_agent_uid), str(self.human_agent_uid)], obs
            )

            # Get human subgraph using only human observations
            observation_sources = []
            if self.conf.agent_asymmetry:
                # under asymmetry condition human's world-graph only uses human's own observations
                observation_sources = [str(self.human_agent_uid)]
            else:
                # under symmetry condition the human's world-graph uses both human's and Spot's observations
                observation_sources = [
                    str(self.robot_agent_uid),
                    str(self.human_agent_uid),
                ]
            most_recent_human_subgraph = self.perception.get_recent_subgraph(
                observation_sources,
                obs,
            )

            # Update robot graph
            self.world_graph[self.robot_agent_uid].update(
                most_recent_robot_subgraph, self.partial_obs, self.wm_update_mode
            )

            # Update human graph
            self.world_graph[self.human_agent_uid].update(
                most_recent_human_subgraph, self.partial_obs, self.wm_update_mode
            )

        return

    def update_world_graphs_using_concept_graph(self, obs):
        """
        This method updates world graphs for both agents using
        concept graph and simulator. Under concept graph regime,
        we always operate under partial observability.
        """
        # process obs to get objects detected in the frame
        if not isinstance(self.perception, PerceptionObs):
            raise ValueError(
                "Concept graph update mode requires PerceptionObs object for perception"
            )
        processed_obs = self.perception.preprocess_obs_for_non_privileged_graph_update(
            self.sim, obs, single_agent_mode=self._single_agent_mode
        )
        # get frame-description from perception
        object_detections_in_frame = (
            self.perception.get_object_detections_for_non_privileged_graph_update(
                processed_obs
            )
        )
        # update robot's WG using object detections
        full_state_object_dict = self.sim.object_state_machine.get_snapshot_dict(
            self.sim
        )
        if object_detections_in_frame is not None:
            self.world_graph[
                self.robot_agent_uid
            ].update_non_privileged_graph_with_detected_objects(
                object_detections_in_frame,
                object_state_dict=full_state_object_dict,
            )
        else:
            raise ValueError("Frame description is None")
        # update the world-graph for the human agent
        if self.conf.agent_asymmetry:
            most_recent_human_subgraph = self.perception.get_recent_subgraph(
                [str(self.human_agent_uid)], obs
            )
        else:
            most_recent_human_subgraph = self.perception.get_recent_subgraph(
                [str(self.robot_agent_uid), str(self.human_agent_uid)],
                obs,
            )
        self.world_graph[self.conf.human_agent_uid].update(
            most_recent_human_subgraph,
            self.partial_obs,
            "gt",  # human's WG is always updated in privileged mode
        )

    def get_frame_description(self, obs):
        """
        This method returns frame_description which is used to update world graph
        when using conceptgraph
        """
        raise NotImplementedError(
            "Processing frames for object-descriptions for CG updates is not implemented yet"
        )

    def save_trajectory_step(self, obs):
        # save data from this time-step; for current episode_id and scene
        # also save the episode description in folder
        if self.save_trajectory and self.trajectory_agent_names is not None:
            for curr_agent, camera_source in zip(
                self.trajectory_agent_names, self.conf.trajectory.camera_prefixes
            ):
                if "rgb" in self.save_options:
                    if self._single_agent_mode:
                        rgb = obs[f"{camera_source}_rgb"]
                    else:
                        rgb = obs[f"{curr_agent}_{camera_source}_rgb"]
                    np.save(
                        f"{self.trajectory_save_paths[curr_agent]}/rgb/{self._trajectory_idx}.npy",
                        rgb,
                    )
                    imageio.imwrite(
                        f"{self.trajectory_save_paths[curr_agent]}/rgb/{self._trajectory_idx}.jpg",
                        rgb,
                    )
                if "depth" in self.save_options:
                    if self._single_agent_mode:
                        depth = obs[f"{camera_source}_depth"]
                    else:
                        depth = obs[f"{curr_agent}_{camera_source}_depth"]
                    depth_image = depth_to_rgb(depth)
                    cv2.imwrite(
                        f"{self.trajectory_save_paths[curr_agent]}/depth/{self._trajectory_idx}.png",
                        depth_image,
                    )
                    np.save(
                        f"{self.trajectory_save_paths[curr_agent]}/depth/{self._trajectory_idx}.npy",
                        depth,
                    )
                if "panoptic" in self.save_options:
                    if self._single_agent_mode:
                        panoptic = obs[f"{camera_source}_panoptic"]
                    else:
                        panoptic = obs[f"{curr_agent}_{camera_source}_panoptic"]
                    cv2.imwrite(
                        f"{self.trajectory_save_paths[curr_agent]}/panoptic/{self._trajectory_idx}.png",
                        panoptic,
                    )
                    np.save(
                        f"{self.trajectory_save_paths[curr_agent]}/panoptic/{self._trajectory_idx}.npy",
                        panoptic,
                    )
                # NOTE: this assumes poses for head_rgb and head_depth are the exact
                # same
                if self._single_agent_mode:
                    pose = np.linalg.inv(
                        self.sim.agents[0]
                        ._sensors[f"{camera_source}_rgb"]
                        .render_camera.camera_matrix
                    )
                else:
                    pose = np.linalg.inv(
                        self.sim.agents[0]
                        ._sensors[f"{curr_agent}_{camera_source}_rgb"]
                        .render_camera.camera_matrix
                    )
                np.save(
                    f"{self.trajectory_save_paths[curr_agent]}/pose/{self._trajectory_idx}.npy",
                    pose,
                )
                # NOTE: another way of accessing camera pose
                # fixed_pose = get_camera_transform(
                #     self.sim.agents_mgr._all_agent_data[
                #         0
                #     ].articulated_agent,
                #     camera_name=f"{curr_agent}_{camera_source}_rgb",
                # )
                # inv_T = self.sim._default_agent.scene_node.transformation
                # fixed_pose = inv_T @ fixed_pose
            self._trajectory_idx += 1

    @property
    def agents(self):
        """
        Return the agents defined in this environment
        """
        return self.sim.agents_mgr._all_agent_data

    def step(self, low_level_actions):
        """
        This method performs a single step through the environment given list of
        low level action vectors for one or both of the agents.
        """

        # Setup trajectory logging mechanism if not already done
        if self.save_trajectory and not self._setup_current_episode_logging:
            self.setup_logging_for_current_episode()
            self._setup_current_episode_logging = True

        # get the joint final_action_vector
        final_action_vector = self.get_final_action_vector(low_level_actions)

        # PHYSICS!!!
        obs, reward, done, info = self.env.step(final_action_vector)

        # Update world graphs
        self.update_world_graphs(obs)

        return obs, reward, done, info

    def filter_obs_space(self, batch, agent_uid):
        """
        This method returns observations belonging to the specified agent
        """
        agent_name = f"agent_{agent_uid}"
        agent_name_bar = f"{agent_name}_"
        output_batch = {
            obs_name.replace(agent_name_bar, ""): obs_value
            for obs_name, obs_value in batch.items()
            if agent_name in obs_name
        }
        return output_batch

    def __get_internal_obs_space(self):
        inner_observation_space = {}
        for key, value in self.observation_space.items():
            agent_id, no_agent_id_key = separate_agent_idx(key)
            if no_agent_id_key in self.mappings:
                inner_observation_space[
                    agent_id + "_" + self.mappings[no_agent_id_key]
                ] = value
            else:
                inner_observation_space[key] = value
        self.internal_observation_space = gym.spaces.Dict(inner_observation_space)

    def __parse_observations(self, obs):
        new_obs = []
        if self._single_agent_mode:
            for key in sorted(obs.keys()):
                new_obs.append((key, obs[key]))
        else:
            for key in sorted(obs.keys()):
                agent_id, no_agent_id_key = separate_agent_idx(key)
                if no_agent_id_key in self.mappings:
                    new_obs.append(
                        (agent_id + "_" + self.mappings[no_agent_id_key], obs[key])
                    )
                else:
                    new_obs.append((key, obs[key]))
        new_obs = [OrderedDict(new_obs)]
        batch = batch_obs(new_obs, device=self.device)
        return apply_obs_transforms_batch(batch, self.obs_transforms)
