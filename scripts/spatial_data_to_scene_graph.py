#!/usr/bin/env python3

"""
Utility to convert spatial_data JSON files (GT furniture data) into a PARTNR scene graph.
This allows loading furniture with room assignments without requiring the full simulator.

Usage:
    python scripts/spatial_data_to_scene_graph.py --input path/to/spatial_data_101.json --output path/to/output_scene_graph.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

from habitat_llm.world_model import (
    Furniture,
    House,
    Receptacle,
    Room,
    WorldGraph,
)
from habitat_llm.world_model.world_graph import flip_edge

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_bbox_from_bounds_and_size(position: list, bounds: dict, size: list) -> tuple:
    """
    Compute bbox_min and bbox_max from spatial_data format.
    
    Args:
        position: [x, y, z] position of furniture
        bounds: dict with 'min' and 'max' keys containing local bounds
        size: [width, height, depth] dimensions
    
    Returns:
        (bbox_min, bbox_max) as lists
    """
    # Bounds in spatial_data are local coordinates relative to position
    # We need to transform to global coordinates
    local_min = np.array(bounds["min"])
    local_max = np.array(bounds["max"])
    pos = np.array(position)

    # Global bbox
    bbox_min = (pos + local_min).tolist()
    bbox_max = (pos + local_max).tolist()

    return bbox_min, bbox_max


def normalize_room_type(room_name: str) -> str:
    """Normalize room type to match PARTNR conventions."""
    # Remove trailing _0, _1, etc.
    room_type = room_name.rsplit('_', 1)[0] if '_' in room_name else room_name
    # Handle special cases
    room_type = room_type.replace('/', '_or_')
    return room_type


def create_scene_graph_from_spatial_data(spatial_data: Dict[str, Any], include_receptacles: bool = True) -> WorldGraph:
    """
    Create a WorldGraph (scene graph) from spatial_data JSON.
    
    Args:
        spatial_data: Loaded JSON data from spatial_data file
        include_receptacles: Whether to add receptacles (default: True)
    
    Returns:
        WorldGraph populated with furniture and room relationships
    """
    wg = WorldGraph()

    # Create root house node
    house = House("house", {"type": "root"}, "house_0")
    wg.add_node(house)

    # Create rooms from spatial_data
    room_nodes = {}
    if "rooms" in spatial_data:
        logger.info(f"Creating {len(spatial_data['rooms'])} rooms...")
        for room_name, room_data in spatial_data["rooms"].items():
            room_type = normalize_room_type(room_name)
            room_props = {
                "type": room_type,
                "position": room_data.get("position", [0, 0, 0]),
                "size": room_data.get("size", [0, 0, 0])
            }

            # Add bounds if available
            if "bounds" in room_data:
                bbox_min, bbox_max = compute_bbox_from_bounds_and_size(
                    room_props["position"],
                    room_data["bounds"],
                    room_props["size"]
                )
                room_props["bbox_min"] = bbox_min
                room_props["bbox_max"] = bbox_max

            room_node = Room(room_name, room_props)
            wg.add_node(room_node)
            room_nodes[room_name] = room_node

            # Link room to house
            wg.add_edge(room_node, house, "inside", flip_edge("inside"))
            logger.debug(f"  Created room: {room_name} (type: {room_type})")

    # Create furniture from receptacles in spatial_data
    furniture_nodes = {}
    furniture_to_room = {}  # Track which room each furniture belongs to

    if "receptacles" in spatial_data:
        logger.info(f"Creating {len(spatial_data['receptacles'])} furniture items...")
        for furniture_name, furniture_data in spatial_data["receptacles"].items():
            # Extract furniture type from name (e.g., "table_0" -> "table")
            furniture_type = furniture_name.rsplit('_', 1)[0] if '_' in furniture_name else furniture_name

            furniture_props = {
                "type": furniture_type,
                "translation": furniture_data.get("position", [0, 0, 0]),
                "handle": furniture_data.get("handle", ""),
                "template_name": furniture_data.get("template_name", ""),
            }

            # Compute bbox from bounds and size
            if "bounds" in furniture_data and "size" in furniture_data:
                bbox_min, bbox_max = compute_bbox_from_bounds_and_size(
                    furniture_props["translation"],
                    furniture_data["bounds"],
                    furniture_data["size"]
                )
                furniture_props["bbox_min"] = bbox_min
                furniture_props["bbox_max"] = bbox_max
                furniture_props["size"] = furniture_data["size"]

                # Also compute bbox_extent for compatibility
                extent = np.array(furniture_data["size"]) / 2.0
                furniture_props["bbox_extent"] = extent.tolist()
            else:
                logger.warning(f"Furniture '{furniture_name}' missing bounds/size data!")
                furniture_props["_bbox_unavailable"] = True

            # Create furniture node
            furniture_node = Furniture(
                furniture_name,
                furniture_props,
                sim_handle=furniture_data.get("handle")
            )
            wg.add_node(furniture_node)
            furniture_nodes[furniture_name] = furniture_node

            # Try to determine which room this furniture belongs to based on position
            # Simple heuristic: find room whose bbox contains furniture position
            assigned_room = None
            furniture_pos = np.array(furniture_props["translation"])

            for room_name, room_node in room_nodes.items():
                if "bbox_min" in room_node.properties and "bbox_max" in room_node.properties:
                    room_min = np.array(room_node.properties["bbox_min"])
                    room_max = np.array(room_node.properties["bbox_max"])

                    # Check if furniture position is within room bbox
                    if np.all((furniture_pos >= room_min) & (furniture_pos <= room_max)):
                        assigned_room = room_node
                        break

            # Link furniture to room
            if assigned_room:
                wg.add_edge(furniture_node, assigned_room, "inside", flip_edge("inside"))
                furniture_to_room[furniture_name] = assigned_room.name
                # Store room assignment in furniture properties for later export
                furniture_props["assigned_room"] = assigned_room.name
                logger.debug(f"  {furniture_name} -> {assigned_room.name}")
            else:
                # Fallback: link to house
                wg.add_edge(furniture_node, house, "inside", flip_edge("inside"))
                logger.warning(f"  {furniture_name} -> house (no room match)")

            # Create receptacles if requested
            if include_receptacles:
                # For each furniture, create a default receptacle
                # In real spatial_data, receptacles are implicit in furniture
                rec_name = f"receptacle_{furniture_name}"
                rec_props = {
                    "type": "receptacle",
                    "parent_object_handle": furniture_data.get("handle", "")
                }
                rec_node = Receptacle(rec_name, rec_props)
                wg.add_node(rec_node)
                wg.add_edge(rec_node, furniture_node, "on", flip_edge("on"))

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Scene Graph Creation Summary:")
    logger.info(f"  Rooms: {len(room_nodes)}")
    logger.info(f"  Furniture: {len(furniture_nodes)}")

    # Room distribution
    room_distribution = {}
    for fur_name, room_name in furniture_to_room.items():
        room_distribution[room_name] = room_distribution.get(room_name, 0) + 1

    logger.info("\n  Furniture per room:")
    for room_name in sorted(room_distribution.keys()):
        logger.info(f"    {room_name}: {room_distribution[room_name]} items")

    unassigned_count = len(furniture_nodes) - len(furniture_to_room)
    if unassigned_count > 0:
        logger.warning(f"\n  ⚠ {unassigned_count} furniture items not assigned to rooms")

    logger.info("="*60)

    return wg, furniture_nodes, room_nodes


def save_scene_graph_to_conceptgraph_format(furniture_nodes: dict, room_nodes: dict, output_path: Path):
    """
    Save furniture scene graph in ConceptGraph JSON format (list of edge dictionaries).
    
    Args:
        furniture_nodes: Dict of furniture name -> Furniture node
        room_nodes: Dict of room name -> Room node
        output_path: Path to save JSON file
    """
    # ConceptGraph format is a list of edge dictionaries with object1/object2
    edges_list = []

    # Create a mapping from furniture to ID
    furniture_to_id = {name: idx for idx, name in enumerate(furniture_nodes.keys())}
    room_to_id = {name: idx + len(furniture_nodes) for idx, name in enumerate(room_nodes.keys())}

    # Add furniture-to-room relationships
    for fur_name, fur_node in furniture_nodes.items():
        fur_id = furniture_to_id[fur_name]
        fur_props = fur_node.properties

        # Create object1 dict (furniture)
        object1_dict = {
            "id": fur_id,
            "object_tag": fur_props.get("type", fur_name.rsplit('_', 1)[0]),
            "category_tag": "furniture",
            "original_class_name": [fur_props.get("template_name", fur_name)],  # CG compatibility
            "fix_bbox": True
        }

        # Add bbox_extent if available
        if "bbox_extent" in fur_props:
            object1_dict["bbox_extent"] = fur_props["bbox_extent"]
        elif "bbox_min" in fur_props and "bbox_max" in fur_props:
            # Compute extent from bbox
            bbox_min = np.array(fur_props["bbox_min"])
            bbox_max = np.array(fur_props["bbox_max"])
            extent = (bbox_max - bbox_min) / 2.0
            object1_dict["bbox_extent"] = extent.tolist()

        # Add bbox_center
        if "translation" in fur_props:
            object1_dict["bbox_center"] = fur_props["translation"]
        elif "bbox_min" in fur_props and "bbox_max" in fur_props:
            bbox_min = np.array(fur_props["bbox_min"])
            bbox_max = np.array(fur_props["bbox_max"])
            center = (bbox_min + bbox_max) / 2.0
            object1_dict["bbox_center"] = center.tolist()

        # Find which room this furniture belongs to (stored during creation)
        room_name = fur_props.get("assigned_room")
        if room_name and room_name in room_nodes:
            room_id = room_to_id[room_name]
            room_node = room_nodes[room_name]
            room_props = room_node.properties

            # Normalize room type
            room_type = room_props.get("type", room_name.rsplit('_', 1)[0])
            object1_dict["room_region"] = room_type

            # Create object2 dict (room)
            object2_dict = {
                "id": room_id,
                "object_tag": room_type,
                "category_tag": "room",
                "original_class_name": [room_name],  # CG compatibility
                "fix_bbox": True
            }

            # Add room bbox if available
            if "bbox_extent" in room_props:
                object2_dict["bbox_extent"] = room_props["bbox_extent"]
            elif "bbox_min" in room_props and "bbox_max" in room_props:
                bbox_min = np.array(room_props["bbox_min"])
                bbox_max = np.array(room_props["bbox_max"])
                extent = (bbox_max - bbox_min) / 2.0
                object2_dict["bbox_extent"] = extent.tolist()

            if "position" in room_props:
                object2_dict["bbox_center"] = room_props["position"]
            elif "bbox_min" in room_props and "bbox_max" in room_props:
                bbox_min = np.array(room_props["bbox_min"])
                bbox_max = np.array(room_props["bbox_max"])
                center = (bbox_min + bbox_max) / 2.0
                object2_dict["bbox_center"] = center.tolist()

            # Create edge entry in ConceptGraph format
            edge_dict = {
                "object1": object1_dict,
                "object2": object2_dict,
                "object_relation": "inside",
                "reason": f"{fur_name} is inside {room_name}"
            }
            edges_list.append(edge_dict)

    # Save to JSON (list format, not dict)
    with open(output_path, 'w') as f:
        json.dump(edges_list, f, indent=2)

    logger.info(f"Saved ConceptGraph format scene graph to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert spatial_data JSON to PARTNR scene graph"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to spatial_data JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output scene graph JSON (default: <input>_scene_graph.json)"
    )
    parser.add_argument(
        "--no-receptacles",
        action="store_true",
        help="Don't create receptacle nodes (only furniture)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load input spatial_data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Loading spatial_data from: {input_path}")
    with open(input_path, 'r') as f:
        spatial_data = json.load(f)

    # Create scene graph
    logger.info("Creating scene graph...")
    wg, furniture_nodes, room_nodes = create_scene_graph_from_spatial_data(
        spatial_data,
        include_receptacles=not args.no_receptacles
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_conceptgraph.json"

    # Save scene graph in ConceptGraph format
    save_scene_graph_to_conceptgraph_format(furniture_nodes, room_nodes, output_path)

    logger.info("✓ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
