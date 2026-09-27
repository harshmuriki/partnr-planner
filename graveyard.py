#    Code in dynamic_world_graph.py which was used to add all furniture, receptacles, and agents from GT graph to CG graph.
   
    # def add_all_gt_furniture_with_receptacles(self, full_world_graph, sim):
    #     """
    #     Add all furniture, receptacles, and agents from GT graph to CG graph.
    #     Does NOT add objects - only furniture, receptacles, and agents for navigation/placement.

    #     :param full_world_graph: The GT world graph to copy furniture from
    #     :param sim: The simulator instance to extract bbox information
    #     """
    #     # Clean out any existing furniture and receptacles first
    #     # self.clean_entire_cg_furniture_and_receptacles()

    #     # Get all furniture from GT graph (excluding floors)
    #     gt_furniture_list = [
    #         f for f in full_world_graph.get_all_nodes_of_type(Furniture)
    #         if not isinstance(f, Floor)
    #     ]

    #     cg_furniture_names = {
    #         f.name for f in self.get_all_nodes_of_type(Furniture)
    #         if not isinstance(f, Floor)
    #     }

    #     try:
    #         cg_house = self.get_node_from_name("house")
    #     except ValueError:
    #         cg_house = None
    #         self._logger.warning("No 'house' node found in CG")

    #     furniture_added_count = 0
    #     receptacle_added_count = 0
    #     new_room_added_count = 0

    #     # Bbox extraction statistics
    #     furniture_bbox_stats = {
    #         "total": 0,
    #         "bbox_extracted": 0,
    #         "bbox_from_extent": 0,
    #         "bbox_missing": 0,
    #         "sim_handle_missing": 0
    #     }

    #     # Add the "house" node to the CG graph if it doesn't exist
    #     if cg_house is None:
    #         cg_house = House("house", {"type": "house"})
    #         if cg_house.name not in self._entity_names:
    #             self.add_node(cg_house)
    #             self._entity_names.append(cg_house.name)
    #         self._logger.info("Created and added 'house' node to CG")

    #     for gt_furniture in gt_furniture_list:
    #         try:
    #             furniture_bbox_stats["total"] += 1

    #             # Skip if furniture already exists in CG
    #             if gt_furniture.name in cg_furniture_names:
    #                 self._logger.debug(f"Furniture '{gt_furniture.name}' already exists in CG")
    #                 continue

    #             # Copy furniture properties
    #             furniture_props = dict(gt_furniture.properties)
    #             if "type" not in furniture_props:
    #                 furniture_props["type"] = "furniture"

    #             # Extract bbox from simulator if sim_handle is available
    #             self.extract_bbox_from_simulator(gt_furniture, sim, furniture_props)

    #             # Create furniture node
    #             cg_furniture = Furniture(gt_furniture.name, furniture_props, gt_furniture.sim_handle)
    #             self._logger.debug(f"Created furniture node: {cg_furniture.name}")
    #             if cg_furniture.name not in self._entity_names:
    #                 self.add_node(cg_furniture)
    #                 self._logger.debug(f"Added furniture node to CG: {cg_furniture.name}")
    #                 self._entity_names.append(cg_furniture.name)
    #                 furniture_added_count += 1

    #                 # Initialize cg_room to None - will be set if room is found/created
    #                 cg_room = None

    #                 # Try to find the room this furniture belongs to in GT
    #                 gt_rooms = full_world_graph.get_neighbors_of_type(gt_furniture, Room)
    #                 self._logger.debug(f"Found {len(gt_rooms)} rooms for furniture '{gt_furniture.name}'")
    #                 if gt_rooms:
    #                     gt_room = gt_rooms[0]  # Take first room
    #                     gt_room_name = gt_room.name  # Use exact GT room name

    #                     self._logger.debug(
    #                         f"GT room name: '{gt_room_name}' for furniture '{gt_furniture.name}'"
    #                     )

    #                     # Find CG room by exact name match
    #                     all_cg_rooms = self.get_all_rooms() or []
    #                     self._logger.debug(
    #                         f"Looking for room '{gt_room_name}' in {len(all_cg_rooms)} existing CG rooms: "
    #                         f"{[r.name for r in all_cg_rooms]}"
    #                     )
    #                     for candidate_room in all_cg_rooms:
    #                         if candidate_room.name == gt_room_name:
    #                             cg_room = candidate_room
    #                             self._logger.debug(
    #                                 f"Found CG room '{cg_room.name}' matching GT room '{gt_room_name}'"
    #                             )
    #                             break

    #                     # If no matching room found, create one with exact GT name
    #                     if cg_room is None:
    #                         self._logger.debug(
    #                             f"No CG room found with name '{gt_room_name}', creating new room"
    #                         )
    #                         room_props = dict(gt_room.properties)
    #                         # Use exact GT room name
    #                         cg_room = Room(gt_room_name, room_props)

    #                         if cg_room.name not in self._entity_names:
    #                             self.add_node(cg_room)
    #                             self._entity_names.append(cg_room.name)
    #                             self._logger.debug(f"Created new room in CG: {cg_room.name}")
    #                             new_room_added_count += 1
    #                             # Create floor for the room - try to get floor properties from GT
    #                             gt_floor_name = f"floor_{gt_room.name}"
    #                             try:
    #                                 gt_floor = full_world_graph.get_node_from_name(gt_floor_name)
    #                                 floor_props = dict(gt_floor.properties)
    #                             except ValueError:
    #                                 # GT floor not found, use empty properties
    #                                 floor_props = {}

    #                             room_floor = Floor(f"floor_{cg_room.name}", floor_props)
    #                             if room_floor.name not in self._entity_names:
    #                                 self.add_node(room_floor)
    #                                 self._entity_names.append(room_floor.name)
    #                                 self.add_edge(
    #                                     room_floor, cg_room, "inside", flip_edge("inside")
    #                                 )

    #                             # Link room to house
    #                             if cg_house is not None:
    #                                 self.add_edge(cg_room, cg_house, "inside", flip_edge("inside"))

    #                     # Link furniture to room (whether found or created)
    #                     self.add_edge(
    #                         cg_furniture, cg_room, "inside", opposite_label=flip_edge("inside")
    #                     )
    #                     self._logger.debug(
    #                         f"Linked furniture '{gt_furniture.name}' to CG room '{cg_room.name}'"
    #                     )
    #                 else:
    #                     # No room found in GT, link to house
    #                     self._logger.debug(f"No room found in GT for furniture '{gt_furniture.name}'")
    #                     if cg_house is not None:
    #                         self.add_edge(
    #                             cg_furniture, cg_house, "inside", opposite_label=flip_edge("inside")
    #                         )

    #                 # Add all receptacles for this furniture
    #                 gt_receptacles = full_world_graph.get_neighbors_of_type(gt_furniture, Receptacle)
    #                 self._logger.debug(f"Found {len(gt_receptacles)} receptacles for furniture '{gt_furniture.name}'")
    #                 for gt_rec in gt_receptacles:
    #                     rec_props = dict(gt_rec.properties)
    #                     cg_receptacle = Receptacle(gt_rec.name, rec_props)

    #                     if cg_receptacle.name not in self._entity_names:
    #                         self.add_node(cg_receptacle)
    #                         self._entity_names.append(cg_receptacle.name)
    #                         receptacle_added_count += 1

    #                         # Link receptacle to furniture
    #                         self.add_edge(
    #                             cg_receptacle, cg_furniture, "on", opposite_label=flip_edge("on")
    #                         )

    #                 # Log furniture addition (handle case where cg_room might be None)
    #                 room_name = cg_room.name if cg_room is not None else "house"
    #                 cprint(f"Added gt furniture {gt_furniture.name} to CG room {room_name} new rooms added: {new_room_added_count}", "green")
    #                 print("#"*30)
    #         except Exception as e:
    #             self._logger.error(
    #                 f"Failed to add furniture '{gt_furniture.name}' from GT to CG: {e}",
    #                 exc_info=True
    #             )
    #             continue

    #     # Add agents from GT graph to CG
    #     agent_added_count = 0
    #     gt_humans = full_world_graph.get_all_nodes_of_type(Human)
    #     gt_robots = full_world_graph.get_all_nodes_of_type(SpotRobot)

    #     cg_agent_names = {
    #         agent.name for agent in (self.get_all_nodes_of_type(Human) or [])
    #     } | {
    #         agent.name for agent in (self.get_all_nodes_of_type(SpotRobot) or [])
    #     }

    #     for gt_agent in (gt_humans or []) + (gt_robots or []):
    #         if gt_agent.name not in cg_agent_names:
    #             agent_props = dict(gt_agent.properties)

    #             # Create agent node
    #             if isinstance(gt_agent, Human):
    #                 cg_agent = Human(gt_agent.name, agent_props)
    #             else:
    #                 cg_agent = SpotRobot(gt_agent.name, agent_props)

    #             if cg_agent.name not in self._entity_names:
    #                 self.add_node(cg_agent)
    #                 self._entity_names.append(cg_agent.name)
    #                 agent_added_count += 1
    #                 self._logger.debug(f"Added agent '{cg_agent.name}' from GT to CG")

    #                 # Try to link agent to their room
    #                 gt_rooms = full_world_graph.get_neighbors_of_type(gt_agent, Room)
    #                 if gt_rooms:
    #                     gt_room = gt_rooms[0]
    #                     gt_room_type = gt_room.properties.get("type", gt_room.name)

    #                     # Find CG room by matching room type
    #                     cg_room = None
    #                     all_cg_rooms = self.get_all_rooms() or []
    #                     for candidate_room in all_cg_rooms:
    #                         candidate_type = candidate_room.properties.get("type", candidate_room.name)

    #                         if candidate_type == gt_room_type:
    #                             cg_room = candidate_room
    #                             break

    #                     if cg_room is not None:
    #                         self.add_edge(
    #                             cg_agent, cg_room, "in", opposite_label=flip_edge("in")
    #                         )
    #                         self._logger.debug(
    #                             f"Linked agent '{cg_agent.name}' to room '{cg_room.name}' "
    #                             f"(GT room '{gt_room.name}' -> type '{gt_room_type}')"
    #                         )
    #                     else:
    #                         self._logger.warning(
    #                             f"No CG room found matching type '{gt_room_type}' "
    #                             f"for agent '{cg_agent.name}' (GT room: '{gt_room.name}')"
    #                         )

    #     self._logger.info(
    #         f"Added {furniture_added_count} furniture, {receptacle_added_count} receptacles, "
    #         f"and {agent_added_count} agents from GT to CG"
    #     )

    #     # Set floor translations based on furniture positions for any floors that lack them
    #     self.set_floor_translations_from_furniture()

    #     # Validate furniture-room assignments
    #     self.validate_furniture_room_assignments()

    #     cprint(
    #         f"Added {furniture_added_count} furniture, {receptacle_added_count} receptacles, "
    #         f"and {agent_added_count} agents from GT to CG",
    #         "green"
    #     )
    #     # cprint(f"Final CG graph: {self.to_string()}", "green")
    #     # breakpoint()
    
    
    
    # def extract_bbox_from_simulator(self, furniture_node: Furniture, sim, furniture_props: Dict[str, Any]) -> bool:
    #     """
    #     Extract bounding box (bbox_min, bbox_max) from simulator for a furniture node.
        
    #     :param furniture_node: The furniture node to extract bbox for
    #     :param sim: The simulator instance
    #     :param furniture_props: Dictionary of furniture properties to update with bbox
    #     :return: True if bbox was successfully extracted, False otherwise
    #     """
    #     if furniture_node.sim_handle is None or sim is None:
    #         self._logger.debug(
    #             f"No sim_handle available for furniture '{furniture_node.name}', "
    #             "bbox will not be extracted"
    #         )
    #         return False

    #     try:
    #         # Try to get the furniture object from simulator
    #         import habitat.sims.habitat_simulator.sim_utilities as sutils

    #         fur_obj = sutils.get_obj_from_handle(sim, furniture_node.sim_handle)

    #         if fur_obj is not None:
    #             # Get the AABB and transform
    #             aabb = fur_obj.aabb
    #             transform = fur_obj.transformation

    #             # Transform AABB from local to global coordinates
    #             global_min = transform.transform_point(aabb.min)
    #             global_max = transform.transform_point(aabb.max)

    #             # Extract bbox_min and bbox_max
    #             furniture_props["bbox_min"] = [global_min[0], global_min[1], global_min[2]]
    #             furniture_props["bbox_max"] = [global_max[0], global_max[1], global_max[2]]

    #             self._logger.debug(
    #                 f"Extracted bbox for furniture '{furniture_node.name}': "
    #                 f"min={furniture_props['bbox_min']}, max={furniture_props['bbox_max']}"
    #             )
    #             return True
    #         else:
    #             self._logger.warning(
    #                 f"Could not find sim object for furniture '{furniture_node.name}' "
    #                 f"with handle '{furniture_node.sim_handle}'"
    #             )
    #             return False
    #     except Exception as e:
    #         self._logger.warning(
    #             f"Failed to extract bbox for furniture '{furniture_node.name}': {e}"
    #         )
    #         return False



    # def set_floor_translations_from_furniture(self):
    #     """
    #     Set floor translations based on furniture positions in each room.
    #     For each room, if its floor lacks a translation but the room has furniture
    #     with translations, sets the floor's translation to the average position of
    #     the furniture (at floor level, y=0).
        
    #     This is useful when floors are created dynamically but don't have
    #     translation properties from the GT graph.
    #     """
    #     all_rooms = self.get_all_rooms() or []
    #     floors_updated = 0

    #     for room in all_rooms:
    #         # Get floor for this room
    #         floors = self.get_neighbors_of_type(room, Floor)
    #         if not floors:
    #             self._logger.debug(f"Room '{room.name}' has no floor node")
    #             continue

    #         floor = floors[0]

    #         # Check if floor already has a translation
    #         if "translation" in floor.properties and floor.properties["translation"] is not None:
    #             continue

    #         # Get all furniture in this room (excluding floors)
    #         room_furniture = [
    #             f for f in self.get_neighbors_of_type(room, Furniture)
    #             if not isinstance(f, Floor) and "translation" in f.properties
    #         ]

    #         if not room_furniture:
    #             self._logger.debug(
    #                 f"Room '{room.name}' has no furniture with translations to derive floor position"
    #             )
    #             continue

    #         # Calculate average position from furniture (use floor level y=0)
    #         furniture_positions = [
    #             np.array(f.properties["translation"]) for f in room_furniture
    #         ]
    #         avg_position = np.mean(furniture_positions, axis=0)
    #         # Set y to 0 for floor level
    #         avg_position[1] = 0.0

    #         # Snap to nearest navigable point to ensure floor translation is valid
    #         if hasattr(self, 'sim') and self.sim is not None and hasattr(self.sim, 'pathfinder'):
    #             try:
    #                 import magnum as mn
    #                 snapped = self.sim.pathfinder.snap_point(mn.Vector3(avg_position))
    #                 if not np.any(np.isnan(snapped)):
    #                     floor.properties["translation"] = snapped.tolist()
    #                     self._logger.debug(
    #                         f"Snapped floor '{floor.name}' to navigable point: {snapped.tolist()}"
    #                     )
    #                 else:
    #                     floor.properties["translation"] = avg_position.tolist()
    #                     self._logger.warning(
    #                         f"Could not snap floor '{floor.name}' to navmesh, using average position"
    #                     )
    #             except Exception as e:
    #                 self._logger.warning(f"Error snapping floor position: {e}")
    #                 floor.properties["translation"] = avg_position.tolist()
    #         else:
    #             floor.properties["translation"] = avg_position.tolist()

    #         floors_updated += 1
    #         self._logger.debug(
    #             f"Set floor '{floor.name}' translation to {floor.properties['translation']} "
    #             f"(derived from {len(room_furniture)} furniture items)"
    #         )

    #     self._logger.info(f"Updated {floors_updated} floor translations from furniture positions")
