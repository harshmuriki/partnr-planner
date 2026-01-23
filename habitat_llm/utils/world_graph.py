#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from habitat_llm.utils.core import cprint
from habitat_llm.world_model.world_graph import WorldGraph
from habitat_llm.world_model import Furniture, Object, Receptacle


def print_all_entities(world_graph: WorldGraph) -> None:
    """
    Prints all relevant WorldGraph.Entity to the console by type. Makes it easier to sandbox skill commands by providing active entities to target.

    :param world_graph: The active WorldGraph with all instantiated entities.
    """
    print("\n")
    cprint("Currently available Entities:", "green")
    cprint(" Rooms: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_rooms()]}", "yellow")
    cprint(" Furniture: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_furnitures()]}", "yellow")
    cprint(" Objects: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_objects()]}", "yellow")
    cprint(" Receptacles: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_receptacles()]}", "yellow")
    print("\n")


def print_furniture_entity_handles(world_graph: WorldGraph) -> None:
    """
    Prints a map of active Entity.Furniture names to their sim_handles to console. Makes it easier to debug by mapping planner commands to simulation objects.

    :param world_graph: The active WorldGraph with all instantiated entities.
    """
    print("\n")
    cprint("Furniture Names to Handles:", "green")
    for entity in world_graph.get_all_furnitures():
        sim_handle = world_graph.get_node_from_name(entity.name).sim_handle
        cprint(f"  {entity.name} : {sim_handle}", "yellow")
    print("\n")


def print_object_entity_handles(world_graph: WorldGraph) -> None:
    """
    Prints a map of active Entity.Object names to their sim_handles to console. Makes it easier to debug by mapping planner commands to simulation objects.

    :param world_graph: The active WorldGraph with all instantiated entities.
    """
    print("\n")
    cprint("Object Names to Handles:", "green")
    for entity in world_graph.get_all_objects():
        sim_handle = world_graph.get_node_from_name(entity.name).sim_handle
        cprint(f"  {entity.name} : {sim_handle}", "yellow")
    print("\n")


def get_all_entity_names(world_graph: WorldGraph) -> List[str]:
    """
    Get a list of semantic names for all navigable entities.

    :param world_graph: The active WorldGraph with all instantiated entities.
    :return: The list of all names for navigable entities. For example, to quickly do nav to all testing.
    """
    rooms = [node.name for node in world_graph.get_all_rooms()]
    furniture = [node.name for node in world_graph.get_all_furnitures()]
    objs = [node.name for node in world_graph.get_all_objects()]
    recs = [node.name for node in world_graph.get_all_receptacles()]
    return rooms + furniture + objs + recs


def print_hierarchical_graph(world_graph: WorldGraph) -> None:
    """
    Print a hierarchical representation of the scene graph showing all objects
    and their relationships within rooms/house.

    Structure:
    House
      Room: room_name
        Furniture: furniture_name
          Receptacle: receptacle_name
            Object: object_name
          Object: object_name (if directly on furniture)
        Object: object_name (if directly in room)
      Human: human_name
        Object: object_name (if held)
      SpotRobot: robot_name
        Object: object_name (if held)

    :param world_graph: The active WorldGraph with all instantiated entities.
    """
    print("\n")
    cprint("=" * 80, "green")
    cprint("Hierarchical Scene Graph:", "green")
    cprint("=" * 80, "green")
    print("\n")

    try:
        # Use the built-in display_hierarchy method which starts from "house" node
        # and does a DFS traversal showing the hierarchy
        world_graph.display_hierarchy()
    except ValueError:
        # If "house" node doesn't exist, print a simplified hierarchy
        cprint("House node not found. Printing simplified hierarchy:", "yellow")
        print("\n")

        # Print rooms and their contents
        rooms = world_graph.get_all_rooms()
        for room in rooms:
            cprint(f"Room: {room.name}", "cyan")

            # Get furniture in this room
            furniture_list = world_graph.get_neighbors_of_type(room, Furniture)

            for furniture in furniture_list:
                cprint(f"  Furniture: {furniture.name}", "yellow")

                # Get receptacles on this furniture
                receptacles = world_graph.get_neighbors_of_type(furniture, Receptacle)

                for receptacle in receptacles:
                    cprint(f"    Receptacle: {receptacle.name}", "magenta")

                    # Get objects on this receptacle
                    objects = world_graph.get_neighbors_of_type(receptacle, Object)

                    for obj in objects:
                        # Show object properties if available
                        props_str = ""
                        if obj.properties:
                            state_props = {k: v for k, v in obj.properties.items()
                                         if k in ["is_powered_on", "is_filled", "is_clean"]}
                            if state_props:
                                props_str = f" [{', '.join(f'{k}={v}' for k, v in state_props.items())}]"
                        cprint(f"      Object: {obj.name}{props_str}", "white")

                # Get objects directly on furniture (not on receptacles)
                objects_on_furniture = world_graph.get_neighbors_of_type(furniture, Object)
                for obj in objects_on_furniture:
                    # Check if object is not already on a receptacle of this furniture
                    rec_for_obj = world_graph.find_receptacle_for_object(obj)
                    if rec_for_obj is None:
                        props_str = ""
                        if obj.properties:
                            state_props = {k: v for k, v in obj.properties.items()
                                         if k in ["is_powered_on", "is_filled", "is_clean"]}
                            if state_props:
                                props_str = f" [{', '.join(f'{k}={v}' for k, v in state_props.items())}]"
                        cprint(f"    Object: {obj.name}{props_str}", "white")

            # Get objects directly in room (not on furniture)
            objects_in_room = world_graph.get_neighbors_of_type(room, Object)
            for obj in objects_in_room:
                furniture_for_obj = world_graph.find_furniture_for_object(obj)
                if furniture_for_obj is None:
                    props_str = ""
                    if obj.properties:
                        state_props = {k: v for k, v in obj.properties.items()
                                     if k in ["is_powered_on", "is_filled", "is_clean"]}
                        if state_props:
                            props_str = f" [{', '.join(f'{k}={v}' for k, v in state_props.items())}]"
                    cprint(f"  Object: {obj.name}{props_str}", "white")

        # Print agents and held objects
        try:
            agents = world_graph.get_agents()
            for agent in agents:
                agent_type = "Human" if agent.__class__.__name__ == "Human" else "SpotRobot"
                cprint(f"{agent_type}: {agent.name}", "green")

                # Get objects held by agent
                held_objects = world_graph.get_neighbors_of_type(agent, Object)

                for obj in held_objects:
                    props_str = ""
                    if obj.properties:
                        state_props = {k: v for k, v in obj.properties.items()
                                     if k in ["is_powered_on", "is_filled", "is_clean"]}
                        if state_props:
                            props_str = f" [{', '.join(f'{k}={v}' for k, v in state_props.items())}]"
                    cprint(f"  Object: {obj.name}{props_str}", "white")
        except ValueError:
            pass  # No agents found

    print("\n")
    cprint("=" * 80, "green")
    print("\n")
