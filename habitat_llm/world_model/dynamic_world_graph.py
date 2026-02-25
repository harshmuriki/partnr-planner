#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
#
# ---
# This module implements PARTNR logic for non-privileged world-graph (i.e. maintains state without
# accessing any sim information) and partial-observability logic for privileged world-graph (mainly
# how to change state based on last-action and last-action's result from either agent). All  modules
# specific to non-privileged graph have non-privileged in the name or in the docstring.
# ---

import logging
import random
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from habitat_llm.utils.core import cprint

if TYPE_CHECKING:
    from habitat_llm.perception.perception_obs import PerceptionObs

import numpy as np
import torch

from habitat_llm.utils.geometric import (
    opengl_to_opencv,
    unproject_masked_depth_to_xyz_coordinates,
)
from habitat_llm.utils.semantic_constants import EPISODE_OBJECTS
from habitat_llm.world_model import (
    Entity,
    Floor,
    Furniture,
    House,
    Human,
    Object,
    Receptacle,
    Room,
    SpotRobot,
    UncategorizedEntity,
    WorldGraph,
)
from habitat_llm.world_model.world_graph import flip_edge


class DynamicWorldGraph(WorldGraph):
    """
    This derived class collects all methods specific to world-graph created and
    maintained based on observations instead of privileged sim data.
    """

    def __init__(
        self,
        max_neighbors_for_room_assignment: int = 5,
        num_closest_entities_for_entity_matching: int = 5,
        max_detection_distance: float = 0.5,
        use_gt_object_locations: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.MAX_NEIGHBORS_FOR_ROOM_ASSIGNMENT = max_neighbors_for_room_assignment
        self.NUM_CLOSEST_ENTITIES_FOR_ENTITY_MATCHING = (
            num_closest_entities_for_entity_matching
        )
        self.max_detection_distance = max_detection_distance
        self.use_gt_object_locations = use_gt_object_locations
        self.include_objects = False
        self._sim_objects = EPISODE_OBJECTS
        self._entity_names: List[str] = []
        self._logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        # Set habitat_llm logger hierarchy to DEBUG if not already set
        habitat_llm_logger = logging.getLogger("habitat_llm")
        if habitat_llm_logger.level == logging.NOTSET:
            habitat_llm_logger.setLevel(logging.DEBUG)
        self._logger.setLevel(logging.DEBUG)
        self._sim_object_to_detected_object_map: dict = {}
        self._articulated_agents: dict = {}

    def _cg_object_to_object_uid(self, cg_object: dict) -> str:
        cg_object["object_tag"] = cg_object["object_tag"].lower()
        return f"{cg_object['id']}_{cg_object['object_tag'].replace(' ', '_').replace('/', '_or_').replace('-', '_')}"

    def _is_object(self, object_category: str, sim: bool = True) -> bool:
        if sim:
            return object_category in self._sim_objects
        return True

    def set_articulated_agents(self, articulated_agent: dict):
        self._articulated_agents = articulated_agent

    def create_cg_edges(
        self,
        cg_dict_list: Optional[dict] = None, # the json file of the concept graph
        include_objects: bool = False,
        verbose: bool = False,
    ):
        """
        This method populates the graph from the dict output of CG. Creates a graph to store
        different entities in the world and their relations to one another
        """
        self.include_objects = include_objects
        self._raw_cg = cg_dict_list

        def to_entity_input(obj: dict):
            translation = obj["bbox_center"]
            if obj.get("fix_bbox", False):
                translation = [translation[0], translation[2], translation[1]]
            bbox_min = np.array(
                np.array(translation) - np.array(obj["bbox_extent"])
            ).tolist()
            bbox_max = np.array(
                np.array(translation) + np.array(obj["bbox_extent"])
            ).tolist()
            return {
                "name": self._cg_object_to_object_uid(obj),
                "properties": {
                    "type": obj["category_tag"],
                    "translation": translation,
                    "bbox_extent": obj["bbox_extent"],
                    "bbox_max": bbox_max,
                    "bbox_min": bbox_min,
                },
            }

        def is_valid_obj_or_furniture(obj: dict, include_objects: bool):
            # check that object is valid and not a wall or floor
            tag_is_OK: bool = (
                obj["object_tag"] != "invalid"
                and "floor" not in obj["object_tag"]
                and "wall" not in obj["object_tag"]
            )
            # check that object is an object or furniture
            is_furniture: bool = tag_is_OK and obj["category_tag"] == "furniture"
            is_object: bool = tag_is_OK and obj["category_tag"] == "object"
            return is_furniture or (is_object and include_objects)

        # Create root node
        house = House("house", {"type": "root"}, "house_0")
        self.add_node(house)
        self._entity_names.append("house")

        if cg_dict_list is None or not cg_dict_list:
            raise ValueError("Need a list of CG edges to create the graph")

        for edge_candidate in cg_dict_list:
            object1 = edge_candidate["object1"]
            object2 = edge_candidate["object2"]
            edge_relation = edge_candidate["object_relation"].lower()
            if verbose:
                self._logger.info(f"RAW CG OUTPUT\n:{edge_candidate}")
            object_nodes: List[Entity] = []
            for obj in [object1, object2]:
                obj_uid = self._cg_object_to_object_uid(obj)
                if (
                    is_valid_obj_or_furniture(obj, include_objects)
                    and obj_uid not in self._entity_names
                ):
                    obj["object_tag"] = obj["object_tag"].lower()
                    obj["category_tag"] = obj["category_tag"].lower()
                    obj["room_region"] = obj["room_region"].lower()
                    obj_entity_input_dict = to_entity_input(obj)
                    if obj["category_tag"] == "object":
                        object_nodes.append(Object(**obj_entity_input_dict))
                        self.add_node(object_nodes[-1])
                        self._entity_names.append(obj_entity_input_dict["name"])
                    elif obj["category_tag"] == "furniture":
                        object_nodes.append(Furniture(**obj_entity_input_dict))
                        self.add_node(object_nodes[-1])
                        self._entity_names.append(obj_entity_input_dict["name"])
                    elif obj["category_tag"] == "invalid":
                        object_nodes.append(
                            UncategorizedEntity(**obj_entity_input_dict)
                        )
                        self.add_node(object_nodes[-1])
                        self._entity_names.append(obj_entity_input_dict["name"])
                    if verbose:
                        self._logger.info(f"Added new entity: {object_nodes[-1].name}")
                    # make a child of room_region allocated
                    room_region = obj["room_region"].replace(" ", "_")
                    room_node = None
                    try:
                        room_node = self.get_node_from_name(room_region)
                    except ValueError as e:
                        self._logger.info(e)
                    if room_node is None and room_region != "fail":
                        room_node = Room(
                            **{"properties": {"type": room_region}, "name": room_region}
                        )
                        self.add_node(room_node)
                        self._entity_names.append(room_region)
                        self.add_edge(
                            room_node,
                            house,
                            "inside",
                            opposite_label=flip_edge("inside"),
                        )
                        room_floor = Floor(f"floor_{room_node.name}", {})
                        self.add_node(room_floor)
                        self._entity_names.append(room_floor.name)
                        self.add_edge(
                            room_floor, room_node, "inside", flip_edge("inside")
                        )
                        if verbose:
                            self._logger.info(f"Added new room: {room_node.name}")
                    assert room_node is not None
                    self.add_edge(
                        obj_entity_input_dict["name"],
                        room_node.name,
                        "inside",
                        opposite_label=flip_edge("inside"),
                    )
                    if verbose:
                        self._logger.info(
                            f"Added above object to room: {room_node.name}"
                        )
                elif obj_uid in self._entity_names:
                    object_nodes.append(self.get_node_from_name(obj_uid))
                    if verbose:
                        self._logger.info(
                            f"Found existing entity: {object_nodes[-1].name}"
                        )
            # add edge between object1 and object2
            if len(object_nodes) == 2:
                if edge_relation in ["none of these", "fail"]:
                    continue

                if "next to" in edge_relation:
                    self.add_edge(
                        object_nodes[0],
                        object_nodes[1],
                        "next to",
                        opposite_label=flip_edge("next to"),
                    )
                elif edge_relation == "a on b":
                    self.add_edge(
                        object_nodes[0],
                        object_nodes[1],
                        "on",
                        opposite_label=flip_edge("on"),
                    )
                elif edge_relation == "b on a":
                    self.add_edge(
                        object_nodes[1],
                        object_nodes[0],
                        "on",
                        opposite_label=flip_edge("on"),
                    )
                elif edge_relation == "a in b":
                    self.add_edge(
                        object_nodes[0],
                        object_nodes[1],
                        "inside",
                        opposite_label=flip_edge("inside"),
                    )
                elif edge_relation == "b in a":
                    self.add_edge(
                        object_nodes[1],
                        object_nodes[0],
                        "inside",
                        opposite_label=flip_edge("inside"),
                    )
                else:
                    raise ValueError(
                        f"Unknown edge candidate: {edge_relation}, between objects: {object1} and {object2}"
                    )
                if verbose:
                    self._logger.info(
                        f"Added edge {edge_relation} b/w {object_nodes[0].name} and {object_nodes[1].name}"
                    )
        if verbose:
            self._logger.info("Before pruning")
            self.display_hierarchy()
        self._fix_furniture_without_assigned_room()
        self._clean_up_room_and_floor_locations()
        if verbose:
            self._logger.info("After pruning")
            self.display_hierarchy()

    def _fix_furniture_without_assigned_room(self):
        """
        Makes sure each furniture is assigned to some room; default=unknown
        """
        furnitures = self.get_all_furnitures()
        all_rooms = self.get_all_nodes_of_type(Room)
        default_room = None
        for room in all_rooms:
            if "unknown" in room.name:
                default_room = room
                break
        for fur in furnitures:
            room = self.get_neighbors_of_type(fur, Room)
            if len(room) == 0:
                self.add_edge(default_room, fur, "in", flip_edge("in"))
        fur_room_count = [
            1 if len(self.get_neighbors_of_type(fur, Room)) > 0 else 0
            for fur in furnitures
        ]
        assert sum(fur_room_count) == len(fur_room_count)

    def _clean_up_room_and_floor_locations(self):
        """
        Iterates over the now-filled graph and attaches positions of known furniture
        belonging to a room to the floor of that room as translation property.
        Also prunes out rooms without any object/furniture in it.

        We use furniture in a room to set the room's location, if a room does not have
        furniture we can't get this geometric information and hence we remove such rooms.
        """
        # find rooms without any furniture in it
        prune_list = []
        rooms = self.get_all_rooms()
        for current_room in rooms:
            furniture = self.get_neighbors_of_type(current_room, Furniture)
            random.shuffle(furniture)
            # remove rooms with just a floor edge or no edges to furniture
            if furniture is None:
                prune_list.append(current_room)
            elif len(furniture) == 1 and isinstance(furniture[0], Floor):
                if isinstance(furniture[0], Floor):
                    room_floor = furniture[0]
                    prune_list.append(room_floor)
                prune_list.append(current_room)
            else:
                # if a room has furniture then choose an arbitrary one which has
                # translation and set the location of the floor and the room
                # to be same as this furniture
                room_floor = [fnode for fnode in furniture if isinstance(fnode, Floor)][
                    0
                ]
                valid_translation = None
                for fur in furniture:
                    if "translation" in fur.properties:
                        valid_translation = fur.properties["translation"]
                        break
                room_floor.properties["translation"] = valid_translation
                current_room.properties["translation"] = valid_translation

        for prune_room in prune_list:
            cprint(f"Pruning room '{prune_room.name}'", "red")
            self.remove_node(prune_room)

    def reassign_furniture_to_rooms_from_spatial_data(
        self,
        spatial_data: Dict[str, Any],
        verbose: bool = True
    ):
        """
        Rebuild concept graph with correct room assignments from spatial_data of rooms.
        Creates all rooms from spatial_data with exact names, then assigns each
        furniture to the closest room. Replaces the existing graph structure.
        
        Args:
            spatial_data: Dict with 'rooms' key containing ground truth room data
            verbose: Whether to print detailed debug logs
        """
        self._logger.info("=" * 80)
        self._logger.info("STARTING ROOM REASSIGNMENT FROM SPATIAL_DATA")
        self._logger.info("=" * 80)

        if not spatial_data:
            self._logger.error("Empty spatial_data provided!")
            return

        # Step 1: Create all rooms from spatial_data
        self._logger.info("\n[STEP 1] Creating rooms from spatial_data...")
        rooms_data = spatial_data.get('rooms', {})
        self._logger.info(f"Found {len(rooms_data)} rooms in spatial_data: {list(rooms_data.keys())}")

        # Store room information for distance calculations
        room_info = {}  # room_name -> (center_position, room_type, bbox_min, bbox_max, size)
        new_room_nodes = {}  # room_name -> Room node

        # Get house node
        house_nodes = self.get_all_nodes_of_type(House)
        house_node = house_nodes[0] if house_nodes else None
        self._logger.debug(f"House node: {house_node.name if house_node else 'None'}")

        for room_name, room_props in rooms_data.items():
            position = np.array(room_props.get('position', [0, 0, 0]))
            polygon = np.array(room_props.get('polygon', []))
            size = np.array(room_props.get('size', [0, 0, 0]))

            # Compute bbox
            half_size = size / 2.0
            bbox_min = position - half_size
            bbox_max = position + half_size

            # Extract room type (keep full name for node)
            room_type = room_name.rsplit('_', 1)[0] if '_' in room_name else room_name

            room_info[room_name] = (position, room_type, bbox_min, bbox_max, size, polygon)

            # Create new room node
            room_node = Room(
                room_name,
                {
                    "type": room_type,
                    "translation": position.tolist(),
                    "bbox_min": bbox_min.tolist(),
                    "bbox_max": bbox_max.tolist(),
                    "size": size.tolist(),
                    "polygon": polygon.tolist(),
                }
            )
            new_room_nodes[room_name] = room_node

            self._logger.debug(
                f"  Created room '{room_name}' (type={room_type}) at pos={position} size={size}"
            )

        self._logger.info(f"Created {len(new_room_nodes)} new room nodes\n")

        # Step 2: Get all existing furniture and objects from CG
        self._logger.info("[STEP 2] Collecting existing furniture and objects from concept graph...")
        all_furniture_with_floors = self.get_all_furnitures()
        # all_objects = self.get_all_objects()

        # Separate floors from furniture - floors will be recreated in ensure_floors_exist()
        all_floors = [f for f in all_furniture_with_floors if isinstance(f, Floor)]
        all_furniture = [f for f in all_furniture_with_floors if not isinstance(f, Floor)]

        # Track assignments
        furniture_assignments = {}  # furniture_node -> (room_name, method)
        object_assignments = {}  # object_node -> (room_name, method)
        items_without_position = []
        items_too_far = []

        # Maximum distance threshold for "closest room" fallback (in meters)
        MAX_DISTANCE_THRESHOLD = 5.0

        # Step 3: Assign each furniture to room (polygon first, then closest)
        self._logger.info("\n[STEP 3] Assigning furniture to rooms...")
        for fur_node in all_furniture:
            fur_pos = fur_node.properties.get('translation')
            if fur_pos is None:
                self._logger.warning(f"  Furniture {fur_node.name} has no position, skipping")
                items_without_position.append(('furniture', fur_node.name))
                continue

            fur_pos = np.array(fur_pos)

            # Single loop: check polygon containment first, track closest room as fallback
            assigned_room = None
            assignment_method = None
            closest_room = None
            min_distance = float('inf')

            for room_name, (room_center, _, bbox_min, bbox_max, _, polygon) in room_info.items():
                # Method 1: Check polygon containment first (most accurate)
                if self._is_point_within_polygon(fur_pos, polygon):
                    assigned_room = room_name
                    assignment_method = "polygon_containment"
                    break  # Found exact match, stop searching

                # Method 2: While looping, track closest room as fallback
                distance = np.linalg.norm(fur_pos - room_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_room = room_name

            # Use closest room if no polygon match found
            if assigned_room is None:
                # Only assign if within reasonable distance threshold
                if closest_room and min_distance < MAX_DISTANCE_THRESHOLD:
                    assigned_room = closest_room
                    assignment_method = f"closest_distance({min_distance:.2f}m)"
                elif closest_room:
                    items_too_far.append(('furniture', fur_node.name, min_distance, closest_room))
                    self._logger.warning(
                        f"  {fur_node.name} at {fur_pos} is {min_distance:.2f}m from closest room "
                        f"{closest_room} - exceeds threshold {MAX_DISTANCE_THRESHOLD}m, skipping"
                    )

            if assigned_room:
                furniture_assignments[fur_node] = (assigned_room, assignment_method)
                self._logger.debug(
                    f"  {fur_node.name} ({fur_node.properties.get('type')}) at {fur_pos} "
                    f"-> {assigned_room} (method: {assignment_method})"
                )

        self._logger.info(f"Assigned {len(furniture_assignments)} furniture items to rooms\n")

        # Step 4: Remove old floors and room structure
        self._logger.info("[STEP 4] Removing old floors and room structure...")

        # Remove all floors first
        for floor in all_floors:
            self.remove_node(floor)
            if floor.name in self._entity_names:
                self._entity_names.remove(floor.name)
            self._logger.debug(f"  Removed old floor: {floor.name}")

        self._logger.info(f"Removed {len(all_floors)} old floors")

        # Remove old rooms
        old_rooms = self.get_all_nodes_of_type(Room)
        self._logger.info(f"Removing {len(old_rooms)} old room nodes")

        for old_room in old_rooms:
            # Remove all edges connected to this room
            self.remove_node(old_room)
            if old_room.name in self._entity_names:
                self._entity_names.remove(old_room.name)
            self._logger.debug(f"  Removed old room: {old_room.name}")

        # Step 5: Add new rooms to graph
        self._logger.info("\n[STEP 5] Adding new room nodes to graph...")
        for room_name, room_node in new_room_nodes.items():
            self.add_node(room_node)
            self._entity_names.append(room_name)

            if house_node:
                self.add_edge(room_node, house_node, "in", opposite_label="contains")

            self._logger.debug(f"  Added room node: {room_name}")

        # Step 6: Connect furniture to new rooms
        self._logger.info("\n[STEP 6] Connecting furniture to new rooms...")
        for fur_node, (room_name, method) in furniture_assignments.items():
            room_node = new_room_nodes[room_name]
            self.add_edge(fur_node, room_node, "in", opposite_label="contains")
            # self._logger.debug(f"  Connected {fur_node.name} -> {room_name} (via {method})")

        # Final summary
        self._logger.info("\n" + "=" * 80)
        self._logger.info("ROOM REASSIGNMENT COMPLETE")
        self._logger.info("=" * 80)
        cprint(f"✓ Created {len(new_room_nodes)} rooms from spatial_data", "green")
        cprint(f"✓ Assigned {len(furniture_assignments)} furniture items", "green")
        cprint(f"✓ Assigned {len(object_assignments)} objects", "green")

        # Count assignment methods
        poly_furniture = sum(1 for _, (_, m) in furniture_assignments.items() if m == "polygon_containment")
        dist_furniture = len(furniture_assignments) - poly_furniture
        poly_objects = sum(1 for _, (_, m) in object_assignments.items() if m == "polygon_containment")
        dist_objects = len(object_assignments) - poly_objects

        cprint(f"  Furniture: {poly_furniture} by polygon, {dist_furniture} by distance", "cyan")
        cprint(f"  Objects: {poly_objects} by polygon, {dist_objects} by distance", "cyan")

        if items_without_position:
            cprint(f"⚠ {len(items_without_position)} items without position:", "yellow")
            for item_type, item_name in items_without_position[:10]:
                self._logger.warning(f"  - {item_type}: {item_name}")
            if len(items_without_position) > 10:
                self._logger.warning(f"  ... and {len(items_without_position) - 10} more")

        if items_too_far:
            cprint(f"⚠ {len(items_too_far)} items too far from any room (>{MAX_DISTANCE_THRESHOLD}m):", "yellow")
            for item_type, item_name, dist, closest in items_too_far[:10]:
                self._logger.warning(f"  - {item_type}: {item_name} ({dist:.2f}m from {closest})")
            if len(items_too_far) > 10:
                self._logger.warning(f"  ... and {len(items_too_far) - 10} more")

        # Print room distribution
        room_counts = {}
        for room_name in new_room_nodes.keys():
            furniture_count = sum(1 for _, (r, _) in furniture_assignments.items() if r == room_name)
            object_count = sum(1 for _, (r, _) in object_assignments.items() if r == room_name)
            room_counts[room_name] = (furniture_count, object_count)

        self._logger.info("\nRoom distribution:")
        for room_name, (fur_count, obj_count) in sorted(room_counts.items()):
            self._logger.info(f"  {room_name}: {fur_count} furniture, {obj_count} objects")

        self._logger.info("=" * 80)

        # Ensure all rooms have floors
        self.ensure_floors_exist(verbose=verbose)

        cprint(f"\nFinal CG graph: {self.to_string()}", "green")

    def ensure_floors_exist(self, verbose: bool = False):
        """
        Check if each room has a floor and create missing floors.
        Floors are created with properties based on room translation or furniture positions.
        
        Args:
            verbose: Whether to print detailed debug logs
        """
        self._logger.info("\n" + "=" * 80)
        self._logger.info("ENSURING ALL ROOMS HAVE FLOORS")
        self._logger.info("=" * 80)

        all_rooms = self.get_all_nodes_of_type(Room)
        created_floors = []

        for room in all_rooms:
            # Check if room already has a floor
            existing_floors = self.get_neighbors_of_type(room, Floor)

            if existing_floors:
                if verbose:
                    self._logger.debug(f"  Room '{room.name}' already has floor: {existing_floors[0].name}")
                continue

            # Room has no floor - create one
            floor_name = f"floor_{room.name}"

            # Determine floor properties
            floor_props = {"type": "floor"}

            # Try to get translation from room properties
            if "translation" in room.properties and room.properties["translation"] is not None:
                room_translation = np.array(room.properties["translation"])
                # Set y=0.0 for floor level
                floor_translation = room_translation.copy()
                floor_translation[1] = 0.0
                floor_props["translation"] = floor_translation.tolist()
                if verbose:
                    self._logger.debug(
                        f"  Creating floor '{floor_name}' with room translation (y=0): {floor_translation}"
                    )
            else:
                # Fallback: average positions of non-Floor furniture in the room
                room_furniture = [
                    f for f in self.get_neighbors_of_type(room, Furniture)
                    if not isinstance(f, Floor) and "translation" in f.properties and f.properties["translation"] is not None
                ]

                if room_furniture:
                    furniture_positions = np.array([f.properties["translation"] for f in room_furniture])
                    avg_position = np.mean(furniture_positions, axis=0)
                    avg_position[1] = 0.0  # Set y=0.0 for floor level
                    floor_props["translation"] = avg_position.tolist()
                    if verbose:
                        self._logger.debug(
                            f"  Creating floor '{floor_name}' with averaged furniture position (y=0): {avg_position}"
                        )
                else:
                    if verbose:
                        self._logger.debug(
                            f"  Creating floor '{floor_name}' without translation (no room/furniture position available)"
                        )

            # Create and add floor node
            floor = Floor(floor_name, floor_props)
            self.add_node(floor)
            self._entity_names.append(floor_name)

            # Connect floor to room with "inside" edge
            self.add_edge(floor, room, "inside", flip_edge("inside"))

            created_floors.append(floor_name)
            if verbose:
                self._logger.info(f"  ✓ Created floor '{floor_name}' for room '{room.name}'")

        # Summary
        self._logger.info("\n" + "=" * 80)
        self._logger.info("FLOOR CREATION COMPLETE")
        self._logger.info("=" * 80)
        cprint(f"✓ Created {len(created_floors)} new floors", "green")

        if created_floors and verbose:
            self._logger.info("Created floors:")
            for floor_name in created_floors:
                self._logger.info(f"  - {floor_name}")

        if len(created_floors) == 0:
            cprint("✓ All rooms already have floors", "green")

        self._logger.info("=" * 80)

    def replace_cg_with_gt_static_structure(
        self,
        gt_world_graph: "WorldGraph",
        verbose: bool = True
    ):
        """
        Replace concept graph furniture, floors, and rooms with ground truth data
        while preserving dynamic object discovery for non-privileged planners.
        
        This method provides accurate spatial structure (GT furniture/floors/rooms)
        without privileged object knowledge - objects remain discoverable through
        navigation and observation.
        
        Args:
            gt_world_graph: Fully initialized ground truth WorldGraph with GT entities
            verbose: Whether to print detailed debug logs
            
        Usage:
            Called after CG initialization but before agent initialization in
            non-privileged graph setup. Gives planners accurate spatial scaffolding
            without GT object positions.
        """
        self._logger.info("=" * 80)
        self._logger.info("REPLACING CG WITH GT STATIC STRUCTURE")
        self._logger.info("=" * 80)

        # ===================================================================
        # STEP 1: Extract GT entities (furniture, floors, rooms - NO objects)
        # ===================================================================
        self._logger.info("\n[STEP 1] Extracting GT furniture, floors, and rooms...")

        gt_furniture = gt_world_graph.get_all_furnitures()
        gt_rooms = gt_world_graph.get_all_nodes_of_type(Room)
        gt_house = gt_world_graph.get_all_nodes_of_type(House)

        # Separate floors from furniture
        gt_floors = [f for f in gt_furniture if isinstance(f, Floor)]
        gt_furniture_only = [f for f in gt_furniture if not isinstance(f, Floor)]

        self._logger.info(f"Found {len(gt_furniture_only)} GT furniture")
        self._logger.info(f"Found {len(gt_floors)} GT floors")
        self._logger.info(f"Found {len(gt_rooms)} GT rooms")

        if verbose:
            self._logger.debug(f"GT furniture: {[f.name for f in gt_furniture_only]}")
            self._logger.debug(f"GT floors: {[f.name for f in gt_floors]}")
            self._logger.debug(f"GT rooms: {[r.name for r in gt_rooms]}")

        # ===================================================================
        # STEP 2: Remove all CG furniture (including floors)
        # ===================================================================
        self._logger.info("\n[STEP 2] Removing CG furniture and floors...")

        cg_furniture = self.get_all_furnitures()
        cg_floors = [f for f in cg_furniture if isinstance(f, Floor)]
        cg_furniture_only = [f for f in cg_furniture if not isinstance(f, Floor)]

        removed_furniture_count = 0
        removed_floor_count = 0

        # Remove all CG furniture (including floors)
        for fur in cg_furniture:
            self.remove_node(fur)
            if fur.name in self._entity_names:
                self._entity_names.remove(fur.name)
            if isinstance(fur, Floor):
                removed_floor_count += 1
            else:
                removed_furniture_count += 1
            if verbose:
                self._logger.debug(f"  Removed CG: {fur.name}")

        self._logger.info(f"Removed {removed_furniture_count} CG furniture")
        self._logger.info(f"Removed {removed_floor_count} CG floors")

        # ===================================================================
        # STEP 3: Remove all CG rooms
        # ===================================================================
        self._logger.info("\n[STEP 3] Removing CG rooms...")

        cg_rooms = self.get_all_nodes_of_type(Room)
        removed_room_count = 0

        for room in cg_rooms:
            self.remove_node(room)
            if room.name in self._entity_names:
                self._entity_names.remove(room.name)
            removed_room_count += 1
            if verbose:
                self._logger.debug(f"  Removed CG room: {room.name}")

        self._logger.info(f"Removed {removed_room_count} CG rooms")

        # ===================================================================
        # STEP 4: Add GT rooms to graph
        # ===================================================================
        self._logger.info("\n[STEP 4] Adding GT rooms...")

        # Get house node (should exist from CG initialization)
        house_nodes = self.get_all_nodes_of_type(House)
        house_node = house_nodes[0] if house_nodes else None

        if house_node is None:
            self._logger.warning("No house node found, creating one")
            house_node = gt_house[0] if gt_house else House("house", {"type": "root"}, "house_0")
            self.add_node(house_node)
            self._entity_names.append(house_node.name)

        added_room_count = 0
        for gt_room in gt_rooms:
            # Create new room node with GT properties
            new_room = Room(
                gt_room.name,
                gt_room.properties.copy()
            )
            self.add_node(new_room)
            self._entity_names.append(new_room.name)

            # Connect room to house
            self.add_edge(new_room, house_node, "in", opposite_label="contains")

            added_room_count += 1
            if verbose:
                self._logger.debug(f"  Added GT room: {new_room.name}")

        self._logger.info(f"Added {added_room_count} GT rooms")

        # ===================================================================
        # STEP 5: Add GT floors to graph
        # ===================================================================
        self._logger.info("\n[STEP 5] Adding GT floors...")

        added_floor_count = 0
        for gt_floor in gt_floors:
            # Create new floor node with GT properties
            new_floor = Floor(
                gt_floor.name,
                gt_floor.properties.copy()
            )
            # Copy sim_handle if present
            if hasattr(gt_floor, 'sim_handle') and gt_floor.sim_handle is not None:
                new_floor.sim_handle = gt_floor.sim_handle

            self.add_node(new_floor)
            self._entity_names.append(new_floor.name)

            # Find parent room from GT graph and replicate edge
            gt_room_neighbors = gt_world_graph.get_neighbors_of_type(gt_floor, Room)
            if gt_room_neighbors:
                gt_parent_room = gt_room_neighbors[0]
                # Find corresponding room in current graph
                try:
                    current_room = self.get_node_from_name(gt_parent_room.name)
                    self.add_edge(new_floor, current_room, "inside", flip_edge("inside"))
                    if verbose:
                        self._logger.debug(f"  Added GT floor: {new_floor.name} -> {current_room.name}")
                except ValueError:
                    self._logger.warning(f"Could not find room {gt_parent_room.name} for floor {new_floor.name}")

            added_floor_count += 1

        self._logger.info(f"Added {added_floor_count} GT floors")

        # ===================================================================
        # STEP 6: Add GT furniture to graph with computed bbox
        # ===================================================================
        self._logger.info("\n[STEP 6] Adding GT furniture with bbox properties...")

        added_furniture_count = 0
        furniture_with_bbox = 0
        furniture_without_bbox = 0

        for gt_fur in gt_furniture_only:
            # Create new furniture node with GT properties
            new_furniture = Furniture(
                gt_fur.name,
                gt_fur.properties.copy()
            )
            # Copy sim_handle (critical for skills to work)
            if hasattr(gt_fur, 'sim_handle') and gt_fur.sim_handle is not None:
                new_furniture.sim_handle = gt_fur.sim_handle

            # Compute and add bbox properties if missing
            if "translation" in new_furniture.properties:
                # Check if bbox already exists
                has_bbox = ("bbox_min" in new_furniture.properties
                           and "bbox_max" in new_furniture.properties)

                if not has_bbox:
                    translation = np.array(new_furniture.properties["translation"])

                    # Try to get bbox_extent, or use default size
                    if "bbox_extent" in new_furniture.properties:
                        extent = np.array(new_furniture.properties["bbox_extent"])
                    else:
                        # Default extent for furniture without explicit size (0.5m cube)
                        extent = np.array([0.5, 0.5, 0.5])

                    # Compute bbox
                    new_furniture.properties["bbox_min"] = (translation - extent).tolist()
                    new_furniture.properties["bbox_max"] = (translation + extent).tolist()
                    new_furniture.properties["bbox_extent"] = extent.tolist()
                    furniture_without_bbox += 1
                else:
                    furniture_with_bbox += 1

            self.add_node(new_furniture)
            self._entity_names.append(new_furniture.name)

            added_furniture_count += 1
            if verbose:
                self._logger.debug(f"  Added GT furniture: {new_furniture.name}")

        self._logger.info(f"Added {added_furniture_count} GT furniture")
        self._logger.info(f"  {furniture_with_bbox} had existing bbox, {furniture_without_bbox} computed from translation")

        # ===================================================================
        # STEP 7: Rebuild spatial relationships (furniture -> rooms)
        # ===================================================================
        self._logger.info("\n[STEP 7] Rebuilding spatial relationships...")

        all_rooms_dict = {room.name: room for room in self.get_all_nodes_of_type(Room)}
        all_furniture = self.get_all_nodes_of_type(Furniture)
        all_furniture = [f for f in all_furniture if not isinstance(f, Floor)]

        assignment_count = 0
        no_room_found = []

        for fur in all_furniture:
            # Find parent room from GT graph
            gt_fur_node = None
            for gt_f in gt_furniture_only:
                if gt_f.name == fur.name:
                    gt_fur_node = gt_f
                    break

            if gt_fur_node is None:
                self._logger.warning(f"Could not find GT node for furniture {fur.name}")
                no_room_found.append(fur.name)
                continue

            # Get room from GT graph
            gt_room_neighbors = gt_world_graph.get_neighbors_of_type(gt_fur_node, Room)

            if gt_room_neighbors:
                gt_parent_room = gt_room_neighbors[0]
                # Find corresponding room in current graph
                if gt_parent_room.name in all_rooms_dict:
                    current_room = all_rooms_dict[gt_parent_room.name]
                    self.add_edge(fur, current_room, "in", opposite_label="contains")
                    assignment_count += 1
                    if verbose:
                        self._logger.debug(f"  {fur.name} -> {current_room.name}")
                else:
                    self._logger.warning(f"Room {gt_parent_room.name} not found for furniture {fur.name}")
                    no_room_found.append(fur.name)
            else:
                self._logger.warning(f"No room found in GT graph for furniture {fur.name}")
                no_room_found.append(fur.name)

        self._logger.info(f"Connected {assignment_count} furniture to rooms")

        if no_room_found:
            cprint(f"⚠ {len(no_room_found)} furniture without room assignment:", "yellow")
            for fur_name in no_room_found[:10]:
                self._logger.warning(f"  - {fur_name}")
            if len(no_room_found) > 10:
                self._logger.warning(f"  ... and {len(no_room_found) - 10} more")

        # ===================================================================
        # STEP 8: Verify graph integrity
        # ===================================================================
        self._logger.info("\n[STEP 8] Verifying graph integrity...")

        # Verify all furniture has sim_handles
        furniture_with_handles = 0
        furniture_without_handles = []

        for fur in self.get_all_nodes_of_type(Furniture):
            if isinstance(fur, Floor):
                continue  # Floors may not have sim_handles
            if hasattr(fur, 'sim_handle') and fur.sim_handle is not None:
                furniture_with_handles += 1
            else:
                furniture_without_handles.append(fur.name)

        if furniture_without_handles:
            cprint(f"⚠ {len(furniture_without_handles)} furniture missing sim_handles:", "yellow")
            for fur_name in furniture_without_handles[:10]:
                self._logger.warning(f"  - {fur_name}")
        else:
            cprint(f"✓ All {furniture_with_handles} furniture have sim_handles", "green")

        # Verify no objects were copied
        current_objects = self.get_all_objects()
        if current_objects:
            cprint(f"✓ Graph has {len(current_objects)} objects (preserved from before or added via observation)", "green")
        else:
            cprint(f"✓ Graph has 0 objects (objects will be discovered via navigation)", "green")

        # Final summary
        self._logger.info("\n" + "=" * 80)
        self._logger.info("GT STATIC STRUCTURE REPLACEMENT COMPLETE")
        self._logger.info("=" * 80)
        cprint(f"✓ Removed: {removed_room_count} CG rooms, {removed_furniture_count} CG furniture, {removed_floor_count} CG floors", "green")
        cprint(f"✓ Added: {added_room_count} GT rooms, {added_furniture_count} GT furniture, {added_floor_count} GT floors", "green")
        cprint(f"✓ Connected {assignment_count} furniture to rooms", "green")
        cprint(f"✓ Objects: {len(current_objects)} in graph (discoverable via observation)", "cyan")
        self._logger.info("=" * 80)

        if verbose:
            self._logger.info("\nFinal graph structure:")
            self.display_hierarchy()

    def add_agent_node_and_update_room(self, agent_node: Union[Human, SpotRobot]):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Add agent-node to the graph and assign room-label based on proximity logic
        """
        self.add_node(agent_node)
        self._entity_names.append(agent_node.name)
        room_node = self.find_room_of_entity(agent_node)
        if room_node is None:
            raise ValueError(
                f"[DynamicWorldGraph.initialize_agent_nodes] No room found for {agent_node.name}"
            )
        self.add_edge(agent_node, room_node, "in", opposite_label="contains")

    def initialize_agent_nodes(self, subgraph: WorldGraph):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Initializes the agent nodes in the graph.
        """
        human_node = subgraph.get_all_nodes_of_type(Human)
        if len(human_node) == 0:
            self._logger.debug("No human node found")
        else:
            human_node = human_node[0]
            dynamic_human_node = Human(human_node.name, {"type": "agent"})
            dynamic_human_node.properties["translation"] = human_node.properties[
                "translation"
            ].copy()
            self.add_agent_node_and_update_room(dynamic_human_node)

        agent_node = subgraph.get_all_nodes_of_type(SpotRobot)
        if len(agent_node) == 0:
            self._logger.debug("No SpotRobot node found")
        else:
            agent_node = agent_node[0]
            dynamic_agent_node = SpotRobot(agent_node.name, {"type": "agent"})
            dynamic_agent_node.properties["translation"] = agent_node.properties[
                "translation"
            ].copy()
            self.add_agent_node_and_update_room(dynamic_agent_node)

    def _set_sim_handles_for_non_privileged_graph(self, perception: "PerceptionObs"):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Sets/Matches sim-handles for each non-privileged entity based on proximity matching to sim entities
        s.t. simulator skills can use these as arguments
        """
        # find closest entity to each entity in non-privileged graph and assign as proxy sim-handle
        all_gt_entities = perception.gt_graph.get_all_nodes_of_type(Furniture)
        # only keep furniture with placeable receptacle
        all_gt_entities = [
            ent
            for ent in all_gt_entities
            if ent.sim_handle in perception.fur_obj_handle_to_recs
        ]
        # only keep entities that have a translation property
        all_gt_entities = [
            ent for ent in all_gt_entities if "translation" in ent.properties
        ]
        all_gt_entity_positions = np.array(
            [np.array(entity.properties["translation"]) for entity in all_gt_entities]
        )

        # Collect mappings to write to file
        mappings = []

        non_privileged_graph_furniture = self.get_all_nodes_of_type(Furniture)
        if non_privileged_graph_furniture is not None:
            for current_fur in non_privileged_graph_furniture:
                # find the closest entity to given target
                entity_distance = np.linalg.norm(
                    all_gt_entity_positions
                    - np.array(current_fur.properties["translation"]),
                    axis=1,
                )
                closest_entity_idx = np.argmin(entity_distance)
                current_fur.sim_handle = all_gt_entities[closest_entity_idx].sim_handle

                # Copy properties from GT furniture to CG furniture
                gt_furniture = all_gt_entities[closest_entity_idx]

                # Copy is_articulated property
                if "is_articulated" in gt_furniture.properties:
                    current_fur.properties["is_articulated"] = gt_furniture.properties["is_articulated"]

                # Copy components property (e.g., faucets, outlets)
                if "components" in gt_furniture.properties:
                    current_fur.properties["components"] = gt_furniture.properties["components"]

                self._sim_object_to_detected_object_map[
                    all_gt_entities[closest_entity_idx].name
                ] = current_fur

                cprint(
                    f"Assigned cg furniture {current_fur.name} to GT furniture {all_gt_entities[closest_entity_idx].name}",
                    "cyan",
                )

                # Collect mapping
                mappings.append(
                    f"{all_gt_entities[closest_entity_idx].name} -> {current_fur.name}"
                )

        # Write mappings to file
        if mappings and hasattr(perception, 'env_interface'):
            try:
                import os
                results_dir = perception.env_interface.conf.paths.results_dir
                episode_id = perception.env_interface.env.env.env._env.current_episode.episode_id
                mapping_dir = os.path.join(results_dir, str(episode_id))
                os.makedirs(mapping_dir, exist_ok=True)
                mapping_file = os.path.join(mapping_dir, "mapping.txt")

                with open(mapping_file, 'w') as f:
                    f.write("GT Furniture -> CG Furniture Mapping\n")
                    f.write("=" * 50 + "\n\n")
                    for mapping in mappings:
                        f.write(mapping + "\n")

                self._logger.info(f"Wrote {len(mappings)} mappings to {mapping_file}")
            except Exception as e:
                self._logger.warning(f"Failed to write mapping file: {e}")

        # make sure each entity has a sim-handle except House and Room
        for entity in self.graph:
            if isinstance(entity, Furniture):
                assert entity.sim_handle is not None

    def find_room_of_entity(
        self, entity_node: Union[Human, SpotRobot], verbose: bool = False
    ) -> Optional[Room]:
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        This method finds the room node that the agent is in

        Logic: Find the objects closest to the agent and assign the agent to the room
        that contains the most number of these objects
        """
        room_node = None
        closest_objects = self.get_closest_entities(
            self.MAX_NEIGHBORS_FOR_ROOM_ASSIGNMENT,
            object_node=entity_node,
            dist_threshold=-1.0,
        )
        room_counts: Dict[Room, int] = {}
        for obj in closest_objects:
            for room in self.get_neighbors_of_type(obj, Room):
                if verbose:
                    self._logger.info(
                        f"{entity_node.name} --> Closest object: {obj.name} is in room: {room.name}"
                    )
                if room in room_counts:
                    room_counts[room] += 1
                else:
                    room_counts[room] = 1
        if room_counts:
            if verbose:
                self._logger.info(f"{room_counts=}")
            room_node = max(room_counts, key=room_counts.get)
        return room_node

    def move_object_from_agent_to_placement_node(
        self,
        object_node: Union[Entity, Object],
        agent_node: Union[Entity, Human, SpotRobot],
        placement_node: Union[Entity, Furniture],
        verbose: bool = True,
    ):
        """
        Utility method to move object to a placement node from a given agent. Does in-place manipulation of the world-graph
        """
        # Detach the object from the agent
        self.remove_edge(object_node, agent_node)

        # Add new edge from object to the receptacle
        # TODO: We should add edge to default receptacle instead of fur
        self.add_edge(object_node, placement_node, "on", flip_edge("on"))
        # snap the object to furniture's center in absence of actual location
        object_node.properties["translation"] = placement_node.properties["translation"]
        if verbose:
            self._logger.info(
                f"Moved {object_node.name} from {agent_node.name} to {placement_node.name}"
            )

    def _non_privileged_graph_check_if_object_is_redundant(
        self,
        new_object_node: Object,
        closest_objects: List[Union[Object, Furniture, Room]],
        merge_threshold: float = 0.25,
        verbose: bool = False,
    ):
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING; ONLY FOR SIM
        Check if this object already exists in the world-graph we do this by
        calculating the euclidean distance between the new object and all
        existing objects in the graph

        :param new_object_node: the Object node created based on observations
        :param closest_objects: list of Object nodes closest to newly found object
        :param merge_threshold: if dist(existing_object, new_object) < dist-threshold and
                type of existing_object is same as type of new_object, new-object-node is discarded as a duplicate
        :param verbose: boolean to toggle verbosity
        """
        redundant_object = False
        matching_object = None
        for wg_object in closest_objects:
            euclid_dist = np.linalg.norm(
                np.array(new_object_node.properties["translation"])
                - np.array(wg_object.properties["translation"])
            )
            if verbose:
                print(f"{euclid_dist=} b/w {new_object_node.name} and {wg_object.name}")
                print(
                    f"{new_object_node.name} at {new_object_node.properties['translation']}"
                )
                print(f"{wg_object.name} at {wg_object.properties['translation']}")
                print(
                    f"{wg_object.properties['type']=}; {new_object_node.properties['type']=}"
                )
            if (
                euclid_dist < merge_threshold
                and wg_object.properties["type"] == new_object_node.properties["type"]
            ):
                if verbose:
                    print("SAME OBJECT")
                redundant_object = True
                matching_object = wg_object
                break
        return redundant_object, matching_object

    def _non_privileged_graph_check_if_object_is_held(
        self, new_object_node: Object, verbose: bool = False
    ):
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING
        Check if the object is being held by an agent
        :param new_object_node: the Object node created based on observations
        :param verbose: boolean to toggle verbosity
        """
        agent_nodes = self.get_agents()
        dist_threshold = 0.25
        articulated_agent = None
        for a_node in agent_nodes:
            articulated_agent = self._articulated_agents[
                int(a_node.name.split("_")[1])
            ]  # int(['agent', '1'][1]); assumes agent names are agent_0 and agent_1
            ee_pos = np.array(articulated_agent.ee_transform().translation)
            is_close_to_agent_ee = (
                np.linalg.norm(
                    ee_pos - np.array(new_object_node.properties["translation"])
                )
                < dist_threshold
            )
            if is_close_to_agent_ee:
                # if verbose:
                #     self._logger.debug(
                #         f"NEWLY DETECTED OBJECT, {new_object_node.name}, IS BEING HELD by {a_node.name}"
                #     )
                matching_node = a_node.properties.get("last_held_object", None)
                return True, matching_node
        return False, None

    def update_held_objects(self, agent_node: Union[Human, SpotRobot]):
        """
        If any agent is holding an object, update the object's location to be the agent's

        :param agent_node: Agent node of type Human or SpotRobot holding data associated with pick/place
        """
        held_entity_node = agent_node.properties.get("last_held_object", None)
        if held_entity_node is not None:
            held_entity_node = self.get_node_from_name(held_entity_node.name)
            if held_entity_node.properties.get("time_of_update", None) is not None and (
                time.time() - held_entity_node.properties["time_of_update"] > 1.0
            ):
                held_entity_node.properties["translation"][0] = agent_node.properties[
                    "translation"
                ][0]
                held_entity_node.properties["translation"][2] = agent_node.properties[
                    "translation"
                ][2]
                held_entity_node.properties["time_of_update"] = time.time()
        return

    def update_agent_locations(self, detector_frames: Dict[int, Dict[str, Any]]):
        """
        Use camera pose to update agent locations in the world-graph
        """
        agent_nodes = self.get_agents()
        for uid, detector_frame in detector_frames.items():
            agent_node = [
                a_node for a_node in agent_nodes if a_node.name == f"agent_{uid}"
            ][0]
            agent_node.properties["translation"] = detector_frame["camera_pose"][
                :3, 3
            ].tolist()
            prev_room: Room = self.get_neighbors_of_type(agent_node, Room)[0]
            self.remove_edge(agent_node, prev_room)
            room_node: Optional[Room] = self.find_room_of_entity(agent_node)
            if room_node is None:
                all_rooms = self.get_all_rooms()
                random.shuffle(all_rooms)
                room_node = all_rooms[-1]
            self.add_edge(agent_node, room_node, "in", opposite_label="contains")
            self.update_held_objects(agent_node)

    def add_object_to_graph(
        self,
        object_class: str,
        position: Optional[List[float]] = None,
        furniture_name: Optional[str] = None,
        sim_handle: Optional[str] = None,
        object_states: Optional[Dict[str, Any]] = None,
        connect_to_entities: bool = True,
        height_offset: float = 0.1,
        verbose: bool = False,
    ) -> Object:
        """
        Programmatically add an object of a given class to the concept graph.
        
        This method creates an Object node with the specified properties and adds it
        to the graph. Can place object at a specific position or on top of furniture.
        
        Args:
            object_class: Type/class of object (e.g., "bottle", "cup", "apple")
            position: [x, y, z] position in world coordinates (optional if furniture_name provided)
            furniture_name: Name of furniture to place object on (e.g., "couch_45")
            sim_handle: Optional simulator handle for the object
            object_states: Optional dictionary of object states (e.g., {"is_clean": True})
            connect_to_entities: Whether to automatically connect to nearest furniture/room
            height_offset: Height offset above furniture surface (default 0.1)
            verbose: Whether to print debug information
        
        Returns:
            The created Object node
        
        Example:
            >>> cg = DynamicWorldGraph()
            >>> # Place at specific position
            >>> bottle = cg.add_object_to_graph("bottle", position=[1.0, 0.5, 2.0])
            >>> # Place on furniture
            >>> cup = cg.add_object_to_graph("cup", furniture_name="couch_45")
        """
        # Generate unique object name
        object_name = f"{len(self._entity_names)+1}_{object_class.replace(' ', '_')}"

        # Determine position and target furniture
        target_furniture = None
        if furniture_name is not None:
            # Find furniture by name
            try:
                target_furniture = self.get_node_from_name(furniture_name)
                if not isinstance(target_furniture, Furniture):
                    if verbose:
                        self._logger.warning(
                            f"Node '{furniture_name}' is not a Furniture, treating as position-based"
                        )
                    target_furniture = None
                else:
                    # Get furniture position and place object on top
                    fur_pos = target_furniture.properties.get('translation')
                    if fur_pos is not None:
                        position = [fur_pos[0], fur_pos[1] + height_offset, fur_pos[2]]
                        if verbose:
                            self._logger.info(
                                f"Placing {object_name} on furniture {furniture_name} at {position}"
                            )
                    else:
                        if verbose:
                            self._logger.warning(
                                f"Furniture {furniture_name} has no position, using default"
                            )
                        if position is None:
                            position = [0.0, 0.0, 0.0]
            except Exception as e:
                if verbose:
                    self._logger.warning(
                        f"Could not find furniture '{furniture_name}': {e}"
                    )
                if position is None:
                    position = [0.0, 0.0, 0.0]

        # Ensure position is set
        if position is None:
            if verbose:
                self._logger.warning("No position or furniture specified, using origin [0, 0, 0]")
            position = [0.0, 0.0, 0.0]

        # Create properties dict
        properties = {
            "type": object_class,
            "translation": position,
            "camera_pose_of_view": None,
        }

        # Add states if provided
        if object_states is not None:
            properties["states"] = object_states
        else:
            properties["states"] = {}

        # Create object node
        new_object = Object(object_name, properties, sim_handle=sim_handle)

        # Add to graph
        self.add_node(new_object)
        self._entity_names.append(new_object.name)

        if verbose:
            self._logger.info(f"Added object {object_name} at position {position}")

        # Connect to entities
        if connect_to_entities:
            if target_furniture is not None:
                # Directly connect to specified furniture with "on" relation
                self.add_edge(
                    target_furniture,
                    new_object,
                    "on",
                    flip_edge("on"),
                )
                if verbose:
                    self._logger.info(
                        f"Connected {object_name} to furniture {target_furniture.name} with 'on' relation"
                    )
            else:
                # Use spatial proximity to find connection
                self._connect_object_to_nearest_entity(new_object, verbose=verbose)

        return new_object

    def remove_object_from_graph(self, entity_name: str) -> None:
        super().remove_object_from_graph(entity_name)
        if entity_name in self._entity_names:
            self._entity_names.remove(entity_name)

    def _connect_object_to_nearest_entity(
        self, object_node: Object, verbose: bool = False
    ):
        """
        Connect an object to the nearest furniture or room based on spatial proximity.
        Uses geometric heuristics to determine the most appropriate connection.
        
        Args:
            object_node: The object node to connect
            verbose: Whether to print debug information
        """
        # Get closest entities (furniture and objects for room inference)
        closest_entities = self.get_closest_entities(
            5,
            object_node=object_node,
            include_furniture=True,
            include_rooms=False,
            include_objects=True,
        )

        if not closest_entities:
            if verbose:
                self._logger.warning(
                    f"No nearby entities found for {object_node.name}, skipping connection"
                )
            return

        # Try to connect to furniture using geometric relation check
        reference_furniture, relation = self._cg_check_for_relation(object_node)

        if reference_furniture is not None and relation is not None:
            # Found a furniture with valid spatial relation
            self.add_edge(
                reference_furniture,
                object_node,
                relation,
                flip_edge(relation),
            )
            if verbose:
                self._logger.info(
                    f"Connected {object_node.name} to furniture {reference_furniture.name} "
                    f"via relation '{relation}'"
                )
            return

        # No furniture relation found, try to connect to a room
        # Strategy: Find rooms from nearest objects
        room_counts: Dict[Room, int] = {}

        for entity in closest_entities:
            if isinstance(entity, Object):
                # Get rooms this object is connected to
                obj_rooms = self.get_neighbors_of_type(entity, Room)
                for room in obj_rooms:
                    if room in room_counts:
                        room_counts[room] += 1
                    else:
                        room_counts[room] = 1
                    break  # Use first room
            elif isinstance(entity, Furniture):
                # Get rooms this furniture is connected to
                fur_rooms = self.get_neighbors_of_type(entity, Room)
                for room in fur_rooms:
                    if room in room_counts:
                        room_counts[room] += 1
                    else:
                        room_counts[room] = 1
                    break  # Use first room

        # Connect to most common room
        if room_counts:
            closest_room = max(room_counts, key=room_counts.get)
            self.add_edge(
                object_node,
                closest_room,
                "in",
                "contains",
            )
            if verbose:
                self._logger.info(
                    f"Connected {object_node.name} to room {closest_room.name}"
                )
        else:
            # Fallback: connect to first available room
            all_rooms = self.get_all_rooms()
            if all_rooms:
                self.add_edge(
                    object_node,
                    all_rooms[0],
                    "in",
                    "contains",
                )
                if verbose:
                    self._logger.warning(
                        f"No nearby room found, connected {object_node.name} to "
                        f"default room {all_rooms[0].name}"
                    )

    def move_robot_to_room(
        self, room_name: str, agent_id: int = 0, verbose: bool = False, sim=None
    ) -> bool:
        """
        Move the robot agent to a specific room's floor location.
        Updates the robot's translation property and reconnects it to the target room.
        Also updates the physical agent position in the simulator if sim is provided.
        
        Args:
            room_name: Name of the room to move the robot to (e.g., "kitchen_1")
            agent_id: Agent ID (default 0 for SpotRobot)
            verbose: Whether to print debug information
            sim: Habitat simulator instance (optional, needed to move physical agent)
        
        Returns:
            True if successful, False otherwise
        
        Example:
            >>> cg = DynamicWorldGraph()
            >>> cg.move_robot_to_room("kitchen_1", sim=env_interface.sim)
            >>> # Robot is now positioned at kitchen_1's floor location in both SG and sim
        """
        # Get the robot node
        robot_nodes = self.get_all_nodes_of_type(SpotRobot)
        if not robot_nodes:
            if verbose:
                self._logger.error("No robot node found in graph")
            return False

        robot_node = robot_nodes[0]  # Assume first robot

        # Get the target room
        try:
            room_node = self.get_node_from_name(room_name)
            if not isinstance(room_node, Room):
                if verbose:
                    self._logger.error(f"'{room_name}' is not a Room node")
                return False
        except Exception as e:
            if verbose:
                self._logger.error(f"Could not find room '{room_name}': {e}")
            return False

        # Get the floor node for this room
        floor_name = f"floor_{room_name}"
        try:
            floor_node = self.get_node_from_name(floor_name)
            if not isinstance(floor_node, Floor):
                if verbose:
                    self._logger.error(f"'{floor_name}' is not a Floor node")
                return False
        except Exception as e:
            if verbose:
                self._logger.error(f"Could not find floor '{floor_name}': {e}")
            return False

        # Get floor position
        floor_position = floor_node.properties.get("translation")
        if floor_position is None:
            # Try room position as fallback
            floor_position = room_node.properties.get("translation")
            if floor_position is None:
                if verbose:
                    self._logger.error(f"No position found for {room_name} or its floor")
                return False

        # Update robot position
        robot_node.properties["translation"] = list(floor_position)

        # Reconnect robot to new room
        # Remove old room connections
        old_rooms = self.get_neighbors_of_type(robot_node, Room)
        for old_room in old_rooms:
            self.remove_edge(robot_node, old_room)

        # Add new room connection
        self.add_edge(robot_node, room_node, "inside", flip_edge("inside"))

        # Update physical agent position in simulator if sim is provided
        if sim is not None:
            try:
                import magnum as mn
                # Get the articulated agent from the simulator
                if hasattr(sim, 'agents_mgr') and agent_id < len(sim.agents_mgr._all_agent_data):
                    articulated_agent = sim.agents_mgr._all_agent_data[agent_id].articulated_agent
                    # Set the base position (y is kept at agent's current height)
                    current_height = articulated_agent.base_pos.y
                    new_position = mn.Vector3(floor_position[0], current_height, floor_position[2])
                    articulated_agent.base_pos = new_position
                    if verbose:
                        self._logger.info(f"Updated physical agent position to {new_position}")
                else:
                    if verbose:
                        self._logger.warning(f"Could not access articulated agent {agent_id} in simulator")
            except Exception as e:
                if verbose:
                    self._logger.warning(f"Could not update physical agent position: {e}")

        if verbose:
            self._logger.info(
                f"Moved {robot_node.name} to {room_name} at position {floor_position}"
            )

        return True

    def get_object_from_obs(
        self,
        detector_frame: dict,
        object_id: int,
        uid: int,
        verbose: bool = False,
        object_state_dict: Optional[dict] = None,
    ) -> Optional[Object]:
        """
        Given the processed observation, extract the object's centroid and convert to a
        node
        NOTE: We use Sim information to populate locations for all objects detected by
        Human. Needs to be refactored post bug-fix in KinematicHumanoid class
        @zephirefaith @xavipuig
        """
        obj_id_to_category_mapping = detector_frame["object_category_mapping"]
        obj_id_to_handle_mapping = detector_frame["object_handle_mapping"]
        object_mask = detector_frame["object_masks"][object_id]
        object_handle = obj_id_to_handle_mapping[object_id]
        # NOTE: can add another area based check here to ignore very small objects from far away
        if not np.any(object_mask):
            return None
        if verbose:
            print(
                f"Found object: {obj_id_to_category_mapping[object_id]} with id: {object_id}, from agent: {uid}"
            )
        # TODO: remove after testing RGB-depth alignment from KinematicHumanoid class
        if uid == 1:
            # RGB+depth from human class was misaligned. Using location information sent from human as is
            # This will be updated to use RGB+depth like for agent_0 once that misalignment fix has been tested
            # print("Using human-detected object locations from sim!!!!!")
            if "object_locations" in detector_frame:
                object_centroid = detector_frame["object_locations"][object_id]
            else:
                raise KeyError(
                    "[DynamicWorldGraph.get_object_from_obs] No object_locations found in detector_frame for human-detected objects"
                )
        else:
            depth_numpy = detector_frame["depth"]
            H, W, C = depth_numpy.shape
            pose = opengl_to_opencv(detector_frame["camera_pose"])
            depth_tensor = torch.from_numpy(depth_numpy.reshape(1, C, H, W))
            pose_tensor = torch.from_numpy(pose.reshape(1, 4, 4))
            inv_intrinsics_tensor = torch.from_numpy(
                np.linalg.inv(detector_frame["camera_intrinsics"]).reshape(1, 3, 3)
            )
            mask_tensor = torch.from_numpy(object_mask.reshape(1, C, H, W))
            mask_tensor = ~mask_tensor.bool()
            object_xyz = unproject_masked_depth_to_xyz_coordinates(
                depth_tensor,
                pose_tensor,
                inv_intrinsics_tensor,
                mask_tensor,
            )
            object_centroid = object_xyz.mean(dim=0).numpy().tolist()
        if verbose:
            print(f"{object_centroid=}")

        # Calculate distance from camera to object
        camera_position = detector_frame["camera_pose"][:3, 3]  # Extract translation from 4x4 pose matrix
        object_position = np.array(object_centroid)
        detection_distance = np.linalg.norm(object_position - camera_position)

        # Filter objects that are too far (likely depth/localization errors)
        if not self.use_gt_object_locations and (detection_distance > self.max_detection_distance):
            cprint(
                f"[FILTERED] Object {obj_id_to_category_mapping[object_id]} at distance {detection_distance:.2f}m < threshold {self.max_detection_distance}m",
                "yellow"
            )
            return None

        if verbose:
            cprint(f"Object {obj_id_to_category_mapping[object_id]} detected at distance {detection_distance:.2f}m from camera", "green")

            # Save detection image for debugging/visualization
            self._save_detection_image(
                detector_frame,
                object_id,
                obj_id_to_category_mapping[object_id],
                object_mask,
                uid
            )

        # Override with GT location if requested
        if self.use_gt_object_locations:
            # Get ground truth position from simulator
            if "rom" in detector_frame and detector_frame["rom"] is not None:
                rom = detector_frame["rom"]
                sim_object = rom.get_object_by_handle(object_handle)
                if sim_object is not None:
                    gt_translation = sim_object.translation
                    object_centroid = [gt_translation.x, gt_translation.y, gt_translation.z]
                    if verbose:
                        cprint(f"Using GT location for {obj_id_to_category_mapping[object_id]}: {object_centroid}", "cyan")
                else:
                    self._logger.warning(f"Could not find GT object for handle {object_handle}, using detected position")
            else:
                self._logger.warning("ROM not available in detector_frame, using detected position")

        # add this object to the graph
        new_object_node = Object(
            f"{len(self._entity_names)+1}_{obj_id_to_category_mapping[object_id]}",
            {
                "type": obj_id_to_category_mapping[object_id],
                "translation": object_centroid,
                "camera_pose_of_view": detector_frame["camera_pose"],
            },
        )
        # store sim handle for this object; this information is only used to pass
        # to our skills when needed for kinematics simulation. Not used for any privileged perception tasks
        new_object_node.sim_handle = object_handle
        if object_state_dict is not None:
            for state_name, object_state_values in object_state_dict.items():
                if object_handle in object_state_values:
                    new_object_node.set_state(
                        {state_name: object_state_values[object_handle]}
                    )

        return new_object_node

    def _save_detection_image(
        self,
        detector_frame: dict,
        object_id: int,
        object_category: str,
        object_mask: np.ndarray,
        agent_uid: int,
    ):
        """
        Save RGB image with detected object visualization for debugging.
        
        Args:
            detector_frame: Dict containing RGB image and detection data
            object_id: ID of the detected object
            object_category: Category/type of the detected object
            object_mask: Binary mask for the detected object
            agent_uid: Agent ID that detected the object
        """
        try:
            import cv2
            import time
            import os

            # Try all possible image sources in detector_frame
            rgb_image = None
            image_source = None

            # Try 1: out_img (might be segmentation or RGB)
            try:
                if "out_img" in detector_frame:
                    img_data = detector_frame["out_img"]
                    if img_data is not None and len(img_data.shape) >= 2:
                        # Try to convert to uint8 if not already
                        if img_data.dtype in [np.float32, np.float64]:
                            img_data = (img_data * 255).astype(np.uint8)
                        elif img_data.dtype == np.int32:
                            # Normalize int32 to uint8 range
                            img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255).astype(np.uint8)
                        else:
                            img_data = img_data.astype(np.uint8)

                        # Convert to BGR
                        if len(img_data.shape) == 2:
                            rgb_image = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                        elif len(img_data.shape) == 3 and img_data.shape[2] == 1:
                            rgb_image = cv2.cvtColor(img_data.squeeze(), cv2.COLOR_GRAY2BGR)
                        elif len(img_data.shape) == 3 and img_data.shape[2] == 3:
                            rgb_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        else:
                            rgb_image = img_data
                        image_source = "out_img"
                        # self._logger.debug(f"Successfully loaded image from 'out_img': shape={img_data.shape}, dtype={detector_frame['out_img'].dtype}")
            except Exception as e:
                self._logger.debug(f"Failed to load 'out_img': {e}")

            # Try 2: depth (visualized with colormap)
            if rgb_image is None:
                try:
                    if "depth" in detector_frame:
                        depth_data = detector_frame["depth"]
                        if depth_data is not None and len(depth_data.shape) >= 2:
                            depth_vis = depth_data.squeeze()
                            depth_normalized = ((depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8) * 255).astype(np.uint8)
                            rgb_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                            image_source = "depth"
                            self._logger.debug(f"Successfully loaded image from 'depth': shape={depth_data.shape}")
                except Exception as e:
                    self._logger.debug(f"Failed to load 'depth': {e}")

            # Try 3: Check all other keys for array-like data
            if rgb_image is None:
                for key in detector_frame.keys():
                    if key in ["object_masks", "object_category_mapping", "object_handle_mapping",
                               "camera_intrinsics", "camera_pose", "object_locations"]:
                        continue  # Skip non-image keys
                    try:
                        data = detector_frame[key]
                        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                            self._logger.debug(f"Trying key '{key}': shape={data.shape}, dtype={data.dtype}")
                            # Attempt conversion
                            if data.dtype in [np.float32, np.float64]:
                                data = (data * 255).astype(np.uint8)
                            else:
                                data = data.astype(np.uint8)

                            if len(data.shape) == 2:
                                rgb_image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
                            elif len(data.shape) == 3 and data.shape[2] == 3:
                                rgb_image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                            else:
                                rgb_image = data
                            image_source = key
                            # self._logger.info(f"Successfully loaded image from '{key}'")
                            break
                    except Exception as e:
                        self._logger.debug(f"Failed to load '{key}': {e}")

            if rgb_image is None:
                self._logger.debug("No suitable image found in detector_frame for visualization")
                return

            # Create overlay
            overlay = rgb_image.copy()

            # Create boolean mask from object_mask
            mask_bool = object_mask.squeeze() > 0 if len(object_mask.shape) > 2 else object_mask > 0

            # Create green overlay for masked region
            overlay[mask_bool] = cv2.addWeighted(
                overlay[mask_bool],
                0.6,
                np.full_like(overlay[mask_bool], [0, 255, 0]),  # BGR green
                0.4,
                0
            )

            # Find contours for bounding box
            mask_uint8 = (mask_bool * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contours[0])
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add label with background
                label = f"{object_category} (ID:{object_id})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw label background
                cv2.rectangle(overlay, (x, y - label_h - 10), (x + label_w + 10, y), (0, 255, 0), -1)
                cv2.putText(overlay, label, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)

            # Create output directory
            output_dir = "outputs/detected_objects"
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = int(time.time() * 1000)  # milliseconds
            filename = f"agent{agent_uid}_obj{object_id}_{object_category.replace(' ', '_')}_{timestamp}_{image_source}.jpg"
            output_path = os.path.join(output_dir, filename)

            # Save image
            cv2.imwrite(output_path, overlay)
            # self._logger.debug(f"Saved detection image: {output_path}")

        except Exception as e:
            self._logger.warning(f"Failed to save detection image: {e}")

    def _is_point_within_bbox(self, point, bbox_min, bbox_max):
        # Check if point is within all dimensions of the bounding box
        return np.all((point >= bbox_min) & (point <= bbox_max))

    def _is_point_within_polygon(self, point_3d, polygon):
        """
        Check if a 3D point is within a 2D polygon on the XZ plane.
        
        Args:
            point_3d: 3D point [X, Y, Z]
            polygon: List of [X, Z] vertices forming the polygon
            
        Returns:
            bool: True if point's XZ projection is inside polygon
        """
        # Handle empty or invalid polygon
        if polygon is None:
            return False

        # Convert to numpy array if needed
        if isinstance(polygon, list):
            if len(polygon) < 3:
                return False
            polygon = np.array(polygon)
        elif isinstance(polygon, np.ndarray):
            if polygon.size == 0 or len(polygon) < 3:
                return False
        else:
            return False

        try:
            from shapely.geometry import Point, Polygon
            # Extract X, Z coordinates from 3D point
            point_2d = Point(point_3d[0], point_3d[2])
            # Create polygon from XZ vertices
            poly = Polygon(polygon)
            return poly.contains(point_2d)
        except Exception as e:
            # Fallback to simple bbox check if Shapely fails
            self._logger.warning(f"Polygon containment failed, using bbox fallback: {e}")
            # Calculate polygon bbox
            polygon_array = np.array(polygon)
            min_x, min_z = polygon_array.min(axis=0)
            max_x, max_z = polygon_array.max(axis=0)
            # Check if point is within XZ bounds
            return (min_x <= point_3d[0] <= max_x) and (min_z <= point_3d[2] <= max_z)

    def _is_point_on_bbox(self, point, bbox_min, bbox_max, on_tolerance=0.15, penetration_tolerance=0.08):
        """
        Check if a point is on top of a bounding box with adaptive tolerance.
        Uses Y-up coordinate system (Habitat convention).
        
        Args:
            point: 3D point [X, Y, Z] in world coordinates (Y-up)
            bbox_min, bbox_max: bbox corners [X, Y, Z] (Y-up)
            on_tolerance: how far above surface to still consider "on" (meters)
            penetration_tolerance: how far below surface to allow (meters)
        
        Returns:
            bool: True if point is on bbox surface within tolerance
        """
        point = np.array(point)
        bbox_min = np.array(bbox_min)
        bbox_max = np.array(bbox_max)

        # Check XZ footprint (horizontal plane, Y=vertical) with small margin for detection noise
        margin = 0.02  # 2cm margin for bbox boundary
        within_xz = np.all((point[[0, 2]] >= (bbox_min[[0, 2]] - margin))
                           & (point[[0, 2]] <= (bbox_max[[0, 2]] + margin)))

        # Check vertical position relative to top surface (Y-axis is up)
        # Allow objects from penetration_tolerance below to on_tolerance above
        vertical_distance = point[1] - bbox_max[1]
        on_surface = (-penetration_tolerance <= vertical_distance <= on_tolerance)

        # Log for debugging when close but not matching
        if within_xz and not on_surface:
            self._logger.debug(
                f"Point within XZ footprint but vertical distance {vertical_distance:.3f}m "
                f"outside tolerance [{-penetration_tolerance}, {on_tolerance}]"
            )

        return within_xz and on_surface

    def _cg_check_for_relation(self, object_node):
        """
        Uses geometric heuristics to check for containment or support relation b/w provided object and closest furniture
        """
        obj_pos = np.array(object_node.properties.get('translation', [0, 0, 0]))

        closest_furniture = self.get_closest_entities(
            n=5,
            object_node=object_node,
            include_furniture=True,
            include_objects=False,
            include_rooms=False,
            dist_threshold=5.0  # Only check furniture within 5m
        )
        for fur in closest_furniture:
            if not isinstance(fur, Floor):
                if "bbox_min" not in fur.properties or "bbox_max" not in fur.properties:
                    cprint(f"  Furniture {fur.name} missing bbox properties, skipping", "yellow")
                    continue
                is_within = self._is_point_within_bbox(
                    object_node.properties["translation"],
                    fur.properties["bbox_min"],
                    fur.properties["bbox_max"],
                )
                if is_within:
                    return fur, "in"
                is_on = self._is_point_on_bbox(
                    object_node.properties["translation"],
                    fur.properties["bbox_min"],
                    fur.properties["bbox_max"],
                )
                if is_on:
                    return fur, "on"
        return None, None

    def update_non_privileged_graph_with_detected_objects(
        self,
        frame_desc: Dict[int, Dict[str, Any]],
        object_state_dict: dict = None,
        verbose: bool = False,
    ):
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING
        This method updates the graph based on the processed observations
        """
        # finally update the agent locations based on camera pose
        self.update_agent_locations(frame_desc)

        # create masked point-clouds per object and then extract centroid
        # as a proxy for object's location
        # NOTE: using bboxes may also include non-object points to contribute
        # to the object's position...we can fix this with nano-SAM or using
        # analytical approaches to prune object PCD
        for uid, detector_frame in frame_desc.items():
            if detector_frame["object_category_mapping"]:
                obj_id_to_category_mapping = detector_frame["object_category_mapping"]
                detector_frame["object_handle_mapping"]  # for sensing states
                for object_id in detector_frame["object_masks"]:
                    if not self._is_object(obj_id_to_category_mapping[object_id]):
                        continue
                    new_object_node = self.get_object_from_obs(
                        detector_frame,
                        object_id,
                        uid,
                        verbose,
                        object_state_dict=object_state_dict,
                    )
                    if new_object_node is None:
                        continue
                    new_object_node.properties["time_of_update"] = time.time()

                    # add an edge to the closest room to this object
                    # get top N closest objects (N defined by self.max_neighbors_for_room_matching)
                    closest_objects = self.get_closest_entities(
                        self.MAX_NEIGHBORS_FOR_ROOM_ASSIGNMENT,
                        object_node=new_object_node,
                        include_furniture=False,
                        include_rooms=False,
                    )
                    merge_threshold = 0.25  # default threshold
                    if uid == 1:
                        merge_threshold = 0.5  # increase threshold for human as object is held higher up
                    (
                        redundant_object,
                        matching_object,
                    ) = self._non_privileged_graph_check_if_object_is_redundant(
                        new_object_node,
                        closest_objects,
                        merge_threshold=merge_threshold,
                        verbose=verbose,
                    )
                    (
                        held_object,
                        held_object_node,
                    ) = self._non_privileged_graph_check_if_object_is_held(
                        new_object_node, verbose=True
                    )
                    # only add this object is it is not being held by an agent
                    # or if it is not already in the world-graph
                    skip_adding_object = redundant_object | held_object

                    if skip_adding_object:
                        # update the matching object's translation and states
                        if matching_object is None and held_object_node is not None:
                            matching_object = held_object_node
                        if matching_object is not None:
                            matching_object.properties[
                                "translation"
                            ] = new_object_node.properties["translation"]
                            if "states" in new_object_node.properties:
                                matching_object.properties[
                                    "states"
                                ] = new_object_node.properties["states"]
                            # add current time to the object's properties
                            matching_object.properties[
                                "time_of_update"
                            ] = new_object_node.properties["time_of_update"]
                            # verify translation update worked
                            matching_node_from_wg = self.get_node_from_name(
                                matching_object.name
                            )
                            # TODO: logic for checking surface-placement over another furniture
                            assert (
                                matching_node_from_wg.properties["translation"]
                                == new_object_node.properties["translation"]
                            )
                        continue

                    # Add new object to the graph detected by the agent/panoptic sensor
                    self.add_node(new_object_node)
                    self._entity_names.append(new_object_node.name)
                    # self._logger.info(f"Added new object to CG: {new_object_node}")
                    reference_furniture, relation = self._cg_check_for_relation(
                        new_object_node
                    )
                    if reference_furniture is not None and relation is not None:
                        self._logger.debug(
                            f"[DEBUG] Object {new_object_node.name} connected to furniture "
                            f"{reference_furniture.name} via relation {relation}"
                        )
                        self.add_edge(
                            reference_furniture,
                            new_object_node,
                            relation,
                            flip_edge(relation),
                        )
                    else:
                        self._logger.debug(
                            f"[DEBUG] Object {new_object_node.name}: No furniture relation found. "
                            f"reference_furniture={reference_furniture}, relation={relation}"
                        )
                        # if not redundant and not belonging to a furniture
                        # then find the room this object should belong to
                        # Strategy 1: Check closest objects for room connections
                        # Strategy 2: Check closest furniture for room connections (even if spatial check failed)
                        # Strategy 3: Check floor furniture in rooms
                        room_counts: Dict[Union[Object, Furniture], int] = {}

                        # Strategy 1: Check closest objects
                        for obj in closest_objects:
                            obj_rooms = self.get_neighbors_of_type(obj, Room)
                            if verbose or len(obj_rooms) == 0:
                                self._logger.debug(
                                    f"[DEBUG] Closest object {obj.name} has {len(obj_rooms)} room neighbors"
                                )
                            for room in obj_rooms:
                                if verbose:
                                    self._logger.info(
                                        f"Adding {new_object_node.name} --> Closest object: {obj.name} is in room: {room.name}"
                                    )
                                if room in room_counts:
                                    room_counts[room] += 1
                                else:
                                    room_counts[room] = 1
                                # only use the first Room neighbor, i.e. closest room node
                                break

                        # Strategy 2: Check closest furniture (even if spatial relation check failed)
                        # if not room_counts:
                        #     closest_furniture = self.get_closest_entities(
                        #         5,
                        #         object_node=new_object_node,
                        #         include_furniture=True,
                        #         include_objects=False,
                        #         include_rooms=False,
                        #     )
                        #     self._logger.debug(
                        #         f"[DEBUG] Trying Strategy 2: Checking {len(closest_furniture)} closest furniture for room connections"
                        #     )
                        #     for fur in closest_furniture:
                        #         fur_rooms = self.get_neighbors_of_type(fur, Room)
                        #         if len(fur_rooms) > 0:
                        #             self._logger.debug(
                        #                 f"[DEBUG] Furniture {fur.name} is in {len(fur_rooms)} room(s): {[r.name for r in fur_rooms]}"
                        #             )
                        #             for room in fur_rooms:
                        #                 if room in room_counts:
                        #                     room_counts[room] += 1
                        #                 else:
                        #                     room_counts[room] = 1
                        #                 break  # Use first room

                        # # Strategy 3: Check floor furniture in rooms (fallback)
                        # if not room_counts:
                            self._logger.debug(
                                f"[DEBUG] Trying Strategy 3: Checking floor furniture in rooms"
                            )
                            all_rooms = self.get_all_rooms()
                            for room in all_rooms:
                                # Find floor furniture in this room
                                room_furniture = self.get_neighbors_of_type(room, Furniture)
                                for fur in room_furniture:
                                    if isinstance(fur, Floor) or "floor" in fur.name.lower():
                                        # Check if object is close to this floor
                                        if "translation" in new_object_node.properties and "translation" in fur.properties:
                                            obj_pos = np.array(new_object_node.properties["translation"])
                                            fur_pos = np.array(fur.properties["translation"])
                                            dist = np.linalg.norm(obj_pos - fur_pos)
                                            if dist < 5.0:  # Within 5 meters
                                                self._logger.debug(
                                                    f"[DEBUG] Object {new_object_node.name} is close to floor {fur.name} in room {room.name} (dist={dist:.2f})"
                                                )
                                                room_counts[room] = 1
                                                break
                                if room_counts:
                                    break

                        if room_counts:
                            closest_room = max(room_counts, key=room_counts.get)
                            self._logger.debug(
                                f"[DEBUG] Object {new_object_node.name} connected to room {closest_room.name}"
                            )
                            self.add_edge(
                                new_object_node,
                                closest_room,
                                "in", # Default to "in" relation for room containment
                                opposite_label="contains",
                            )
                        else:
                            self._logger.warning(
                                f"[DEBUG] Object {new_object_node.name} could not be connected to any room! "
                                f"Checked {len(closest_objects)} closest objects and tried furniture-based fallback."
                            )

    def update_by_action(
        self,
        agent_uid: int,
        high_level_action: Tuple[str, str, Optional[str]],
        action_response: str,
        verbose: bool = False,
    ):
        """
        Deterministically updates the world-graph based on last-action taken by agent_{agent_uid} based on the result of that action.
        Only updates the graph if the action was successful. Applicable only when one wants to change agent_{agent_uid}'s graph
        based on agent_{agent_uid}'s actions.

        Please look at update_by_other_agent_action or update_non_privileged_graph_by_other_agent_action for updating self graph based on another agent's actions.
        """
        if "success" in action_response.lower():
            self._logger.debug(
                f"{agent_uid=}: {high_level_action=}, {action_response=}"
            )
            agent_node = self.get_node_from_name(f"agent_{agent_uid}")
            if (
                "place" in high_level_action[0].lower()
                or "rearrange" in high_level_action[0].lower()
            ):
                # update object's new place to be the furniture
                if "place" in high_level_action[0].lower():
                    high_level_actions = high_level_action[1].split(",")
                    # remove the proposition
                    # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                    object_node = self.get_node_from_name(high_level_actions[0].strip())
                    # TODO: Add floor support
                    placement_node = self.get_node_from_name(
                        high_level_actions[2].strip()
                    )
                elif "rearrange" in high_level_action[0].lower():
                    # Split the comma separated pair into object name and receptacle name
                    try:
                        # Handle the case for rearrange proposition usage for place skills
                        high_level_actions = high_level_action[1].split(",")
                        # remove the proposition
                        # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                        high_level_actions = [
                            high_level_actions[0],
                            high_level_actions[2],
                        ]
                        object_node, placement_node = [
                            self.get_node_from_name(value.strip())
                            for value in high_level_actions
                        ]
                    except Exception as e:
                        self._logger.info(f"Issue when split comma: {e}")
                else:
                    raise ValueError(
                        f"Cannot update world graph with action {high_level_action}"
                    )

                # TODO: replace following with the right inside/on relation
                # based on 2nd string argument to Pick when implemented
                # TODO: Temp hack do not add something in placement_node if it is None
                if placement_node is not None:
                    self.move_object_from_agent_to_placement_node(
                        object_node, agent_node, placement_node
                    )
                    if verbose:
                        self._logger.info(
                            f"{self.update_by_action.__name__} Moved object: {object_node.name} from {agent_node.name} to {placement_node.name}"
                        )
                else:
                    if verbose:
                        self._logger.info(
                            f"{self.update_by_action.__name__} Could not move object from agent to placement-node: {high_level_action}"
                        )
            elif (
                "pour" in high_level_action[0].lower()
                or "fill" in high_level_action[0].lower()
            ):
                entity_name = high_level_action[1]
                entity_node = self.get_node_from_name(entity_name)
                entity_node.set_state({"is_filled": True})
                if verbose:
                    self._logger.info(
                        f"{entity_node.name} is now filled, {entity_node.properties}"
                    )
            elif "power" in high_level_action[0].lower():
                entity_name = high_level_action[1]
                entity_node = self.get_node_from_name(entity_name)
                if "on" in high_level_action[0].lower():
                    entity_node.set_state({"is_powered_on": True})
                    if verbose:
                        self._logger.info(
                            f"{entity_node.name} is now powered on, {entity_node.properties}"
                        )
                elif "off" in high_level_action[0].lower():
                    entity_node.set_state({"is_powered_on": False})
                    if verbose:
                        self._logger.info(
                            f"{entity_node.name} is now powered off, {entity_node.properties}"
                        )
                else:
                    raise ValueError(
                        "Expected 'on' or 'off' in power action, got: ",
                        high_level_action[0],
                    )
            elif "clean" in high_level_action[0].lower():
                entity_name = high_level_action[1]
                entity_node = self.get_node_from_name(entity_name)
                entity_node.set_state({"is_clean": True})
                if verbose:
                    self._logger.info(
                        f"{entity_node.name} is now clean, {entity_node.properties}"
                    )
            else:
                if verbose:
                    self._logger.info(
                        "Not updating world graph for successful action: ",
                        high_level_action,
                    )
        return

    def update_non_privileged_graph_by_action(
        self,
        agent_uid: int,
        high_level_action: Tuple[str, str, Optional[str]],
        action_response: str,
        verbose: bool = False,
        drop_placed_object_flag: bool = True,
    ):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Deterministically updates the world-graph based on last-action taken by agent_{agent_uid} based on the result of that action.
        Only updates the graph if the action was successful. Applicable only when one wants to change agent_{agent_uid}'s graph
        based on agent_{agent_uid}'s actions. If drop_placed_object_flag is True then whenever an object is placed it is simply deleter from the graph instead of being read to the receptacle.
        This method is different from update_by_action as it expects non-privileged entities as input and not GT sim entities.

        Please look at update_by_other_agent_action or update_non_privileged_graph_by_other_agent_action for updating self graph based on another agent's actions.
        """
        if (
            isinstance(action_response, str)
            and isinstance(high_level_action[0], str)
            and "success" in action_response.lower()
        ):
            self._logger.debug(
                f"{agent_uid=}: {high_level_action=}, {action_response=}"
            )
            agent_node = self.get_node_from_name(f"agent_{agent_uid}")
            if (
                "place" in high_level_action[0].lower()
                or "rearrange" in high_level_action[0].lower()
            ):
                placement_node = None
                object_node = None
                # Split the comma separated pair into object name and receptacle name
                try:
                    # Handle the case for rearrange proposition usage for place skills
                    high_level_args = high_level_action[1].split(",")
                    # remove the proposition
                    # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                    high_level_args = [
                        high_level_args[0],
                        high_level_args[2],
                    ]
                    object_node, placement_node = [
                        self.get_node_from_name(value.strip())
                        for value in high_level_args
                    ]
                except Exception as e:
                    self._logger.info(f"Issue when split comma: {e}")

                if object_node is not None:
                    if drop_placed_object_flag:
                        self.remove_node(object_node)
                        self._entity_names.remove(object_node.name)
                        if "last_held_object" in agent_node.properties:
                            del agent_node.properties["last_held_object"]
                        self._logger.debug("Object deleted once robot placed it")
                    elif placement_node is not None:
                        self.move_object_from_agent_to_placement_node(
                            object_node, agent_node, placement_node
                        )
                        if verbose:
                            self._logger.info(
                                f"Moved object: {object_node.name} from {agent_node.name} to {placement_node.name}"
                            )
                else:
                    if verbose:
                        self._logger.info(
                            f"Could not move object from agent to placement-node: {high_level_action}"
                        )
            elif "pick" in high_level_action[0].lower():
                object_name = high_level_action[1]
                try:
                    obj_node = self.get_node_from_name(object_name)
                    # remove all current edges from this node
                    obj_neighbors = self.get_neighbors(obj_node).copy()
                    edges_to_remove = []
                    for neighbor in obj_neighbors:
                        edges_to_remove.append((obj_node, neighbor))
                    for edge in edges_to_remove:
                        self.remove_edge(*edge)
                    # add edge b/w obj and the agent
                    self.add_edge(
                        obj_node, agent_node, "on", opposite_label=flip_edge("on")
                    )
                    agent_node.properties["last_held_object"] = obj_node
                    self._logger.debug(
                        f"[{self.update_non_privileged_graph_by_action.__name__}] {agent_node.name} PICKED OBJECT {obj_node.name}"
                    )
                except KeyError as e:
                    self._logger.info(
                        f"Could not find matching receptacle in agent\nException: {e}"
                    )
            else:
                if verbose:
                    self._logger.info(
                        "Not updating world graph for successful action: ",
                        high_level_action,
                    )
        return

    def _cg_find_self_entity_match_to_human_entity(
        self,
        human_entity_name: str,
        human_agent_node: Human,
        is_furniture: bool = False,
    ) -> Optional[Union[Object, Furniture, Room]]:
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Reusable function for finding matching objects to human-held object or furniture to human-held furniture
        """
        human_entity_type = None
        match = re.match(
            r"^(.*)_\d+$",
            human_entity_name,
        )
        if match:
            human_entity_type = match.group(1)
        # now find the object of above type closest to last known human location
        dist_threshold = 0.0
        include_objects = True
        include_furniture = True
        if is_furniture:
            include_objects = False
        else:
            dist_threshold = 2.25  # actuation distance is 2.0 for oracle-skills; adding 0.25 for noise handling
            include_furniture = False
        closest_objects = self.get_closest_entities(
            self.NUM_CLOSEST_ENTITIES_FOR_ENTITY_MATCHING,
            object_node=human_agent_node,
            include_objects=include_objects,
            include_furniture=include_furniture,
            include_rooms=False,
            dist_threshold=dist_threshold,
        )
        if not closest_objects and dist_threshold > 0.0 and is_furniture:
            closest_objects = self.get_closest_entities(
                self.NUM_CLOSEST_ENTITIES_FOR_ENTITY_MATCHING,
                object_node=human_agent_node,
                include_objects=include_objects,
                include_furniture=include_furniture,
                include_rooms=False,
                dist_threshold=-1.0,
            )
        most_likely_matching_object = [
            ent
            for ent in closest_objects
            if ent.properties["type"] == human_entity_type
        ]
        if len(most_likely_matching_object) > 0:
            self._logger.debug(
                f"Mapped node {most_likely_matching_object[0].name} in robot's WG to Human held object: {human_entity_name}; based on both proximity and type"
            )
            return most_likely_matching_object[0]
        if len(closest_objects) > 0:
            most_likely_matching_object_node = closest_objects[0]
            self._logger.debug(
                f"Mapped human's node {human_entity_name} to {most_likely_matching_object_node} based on proximity"
            )
        else:
            most_likely_matching_object_node = None
            self._logger.debug(
                f"Mapped human's node {human_entity_name} to {most_likely_matching_object_node} based on default"
            )
        return most_likely_matching_object_node

    def update_non_privileged_graph_by_other_agent_action(
        self,
        other_agent_uid: int,
        high_level_action_and_args: Tuple[str, str, Optional[str]],
        action_results: str,
        verbose: bool = False,
        drop_placed_object_flag: bool = True,
    ):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Deterministically change self graph based on successful execution of a given action by another agent. The arguments to action
        are based on other agent's identifiers so this method implements essential logic for mapping them back to most likely match in
        self graph, e.g. what the Human agent calls 161_chest_of_drawers may be 11_chest_of_drawers for Spot.
        """
        if "success" in action_results.lower():
            if verbose:
                self._logger.debug(
                    f"{self.update_non_privileged_graph_by_other_agent_action.__name__}{other_agent_uid=}: {high_level_action_and_args=}, {action_results=}"
                )
            agent_node = self.get_human()
            if "pick" in high_level_action_and_args[0].lower():
                # find the matching node and add edge to the other agent's node
                # breakdown what human is holding to its type
                human_picked_object_name = high_level_action_and_args[1].strip()
                most_likely_held_object = (
                    self._cg_find_self_entity_match_to_human_entity(
                        human_picked_object_name, agent_node
                    )
                )
                if most_likely_held_object is not None:
                    object_prev_neighbors = self.get_neighbors(most_likely_held_object)
                    edges_to_remove = []
                    for neighbor in object_prev_neighbors:
                        edges_to_remove.append((most_likely_held_object, neighbor))
                    for edge in edges_to_remove:
                        self.remove_edge(*edge)
                    self.add_edge(
                        most_likely_held_object, agent_node, "on", flip_edge("on")
                    )
                    most_likely_held_object.properties[
                        "translation"
                    ] = agent_node.properties["translation"]
                    # also update last_held_object property
                    agent_node.properties["last_held_object"] = most_likely_held_object
                    self._logger.debug(
                        f"CG updated per human picking {most_likely_held_object.name}"
                    )
                else:
                    self._logger.info(
                        f"Could not find any matching object in robot's WG to {human_picked_object_name=}. Expect funky behavior."
                    )
            if (
                "place" in high_level_action_and_args[0].lower()
                or "rearrange" in high_level_action_and_args[0].lower()
            ):
                all_args = high_level_action_and_args[1].split(",")
                human_placement_furniture_name = all_args[2].strip()
                most_likely_held_object = agent_node.properties.get(
                    "last_held_object", None
                )
                if most_likely_held_object is not None:
                    most_likely_placement_node = (
                        self._sim_object_to_detected_object_map.get(
                            human_placement_furniture_name,
                            self._cg_find_self_entity_match_to_human_entity(
                                human_placement_furniture_name,
                                agent_node,
                                is_furniture=True,
                            ),
                        )
                    )
                    if (
                        most_likely_held_object is not None
                        and most_likely_placement_node is not None
                        and not drop_placed_object_flag
                    ):
                        self.move_object_from_agent_to_placement_node(
                            most_likely_held_object,
                            agent_node,
                            most_likely_placement_node,
                            verbose=verbose,
                        )
                        del agent_node.properties["last_held_object"]
                        self._logger.debug("CG updated per Human place")
                    elif (
                        most_likely_held_object is not None and drop_placed_object_flag
                    ):
                        self.remove_node(most_likely_held_object)
                        self._entity_names.remove(most_likely_held_object.name)
                        del agent_node.properties["last_held_object"]
                        self._logger.debug(
                            "CG updated per Human place; we just removed the object"
                        )
                else:
                    self._logger.debug(
                        "Can't update CG based on human placement; CG did not register Pick"
                    )
            elif (
                "pour" in high_level_action_and_args[0].lower()
                or "fill" in high_level_action_and_args[0].lower()
            ):
                object_name = high_level_action_and_args[1]
                object_node = self._cg_find_self_entity_match_to_human_entity(
                    object_name, agent_node
                )
                if object_node is not None:
                    object_node.set_state({"is_filled": True})
                    if verbose:
                        self._logger.debug(
                            f"{object_node.name} is now filled, {object_node.properties}"
                        )
            elif "power" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self._cg_find_self_entity_match_to_human_entity(
                    object_name, agent_node
                )
                if object_node is not None:
                    if "on" in high_level_action_and_args[0].lower():
                        object_node.set_state({"is_powered_on": True})
                        if verbose:
                            self._logger.debug(
                                f"{object_node.name} is now powered on, {object_node.properties}"
                            )
                    elif "off" in high_level_action_and_args[0].lower():
                        object_node.set_state({"is_powered_on": False})
                        if verbose:
                            self._logger.debug(
                                f"{object_node.name} is now powered off, {object_node.properties}"
                            )
                    else:
                        raise ValueError(
                            "Expected 'on' or 'off' in power action, got: ",
                            high_level_action_and_args[0],
                        )
            elif "clean" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self._cg_find_self_entity_match_to_human_entity(
                    object_name, agent_node
                )
                if object_node is not None:
                    object_node.set_state({"is_clean": True})
                    if verbose:
                        self._logger.debug(
                            f"{object_node.name} is now clean, {object_node.properties}"
                        )

    def _update_gt_graph_by_other_agent_action(
        self,
        other_agent_uid: int,
        high_level_action_and_args: Tuple[str, str, Optional[str]],
        action_results: str,
        verbose: bool = False,
    ):
        """
        Uses the exact object and receptacle names given by the other agent to update the
        graph. We assume that the other agent's identifiers exactly match self identifiers.
        """
        if "success" in action_results.lower():
            self._logger.debug(f"{high_level_action_and_args=} {other_agent_uid=}")
            agent_node = self.get_node_from_name(f"agent_{other_agent_uid}")
            # parse out the object-name and the closest furniture-name
            # if the object is not already in the graph, add it
            # if the placement furniture is not already in the graph, add it as a new
            # node
            if (
                "place" in high_level_action_and_args[0].lower()
                or "rearrange" in high_level_action_and_args[0].lower()
            ):
                # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                # get the object from agent properties
                high_level_action_args = high_level_action_and_args[1].split(",")
                object_node = self.get_node_from_name(high_level_action_args[0].strip())
                try:
                    placement_node = self.get_node_from_name(
                        high_level_action_args[2].strip()
                    )
                    self.move_object_from_agent_to_placement_node(
                        object_node, agent_node, placement_node
                    )
                    self._logger.debug(
                        f"From the perspective of agent_{1-int(other_agent_uid)}:\n{agent_node.name} PLACED OBJECT {object_node.name} on {placement_node.name}"
                    )
                except KeyError as e:
                    self._logger.info(
                        f"Could not find matching receptacle in agent {1-int(other_agent_uid)} graph for {high_level_action_args[2].strip()} that agent {other_agent_uid} is trying to place on.\nException: {e}"
                    )
            elif (
                "pour" in high_level_action_and_args[0].lower()
                or "fill" in high_level_action_and_args[0].lower()
            ):
                object_name = high_level_action_and_args[1]
                object_node = self.get_node_from_name(object_name)
                object_node.set_state({"is_filled": True})
                if verbose:
                    self._logger.info(
                        f"{object_node.name} is now filled, {object_node.properties}"
                    )
            elif "power" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self.get_node_from_name(object_name)
                if "on" in high_level_action_and_args[0].lower():
                    object_node.set_state({"is_powered_on": True})
                    if verbose:
                        self._logger.info(
                            f"{object_node.name} is now powered on, {object_node.properties}"
                        )
                elif "off" in high_level_action_and_args[0].lower():
                    object_node.set_state({"is_powered_on": False})
                    if verbose:
                        self._logger.info(
                            f"{object_node.name} is now powered off, {object_node.properties}"
                        )
                else:
                    raise ValueError(
                        "Expected 'on' or 'off' in power action, got: ",
                        high_level_action_and_args[0],
                    )
            elif "clean" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self.get_node_from_name(object_name)
                object_node.set_state({"is_clean": True})
                if verbose:
                    self._logger.info(
                        f"{object_node.name} is now clean, {object_node.properties}"
                    )
            else:
                if verbose:
                    self._logger.info(
                        "Not updating world graph for successful action: ",
                        high_level_action_and_args,
                    )
        return

    def update_by_other_agent_action(
        self,
        other_agent_uid: int,
        high_level_action_and_args: Tuple[str, str, Optional[str]],
        action_results: str,
        use_semantic_similarity: bool = False,
        verbose: bool = False,
    ):
        if use_semantic_similarity:
            raise NotImplementedError(
                "Semantic similarity based WG update is not supported. Code currently supports closed-vocab naming of object and furniture"
            )
        self._update_gt_graph_by_other_agent_action(
            other_agent_uid,
            high_level_action_and_args,
            action_results,
            verbose=verbose,
        )
