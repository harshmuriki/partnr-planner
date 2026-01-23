#!/usr/bin/env python3

"""
Scene utilities for querying object and furniture positions.

Provides functions to get global positions of objects and furniture from scene files.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


def get_object_position(
    scene_id: str,
    object_identifier: str,
    data_root: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the global position and metadata of an object/furniture from a scene.
    
    Args:
        scene_id: The scene ID (e.g., "107733960_175999701")
        object_identifier: Can be:
            - Full handle (e.g., "267879e5c0249074c75b2dba46348b87c4342223_:0000")
            - Short handle (e.g., "267879e5c0249074c75b2dba46348b87c4342223")
            - Partial name (searches by substring)
        data_root: Optional root directory for data (defaults to project root)
    
    Returns:
        Dictionary containing:
            - 'position': (x, y, z) tuple
            - 'rotation': (qx, qy, qz, qw) quaternion tuple
            - 'template_name': full template name
            - 'object_type': 'object' or 'articulated_object'
            - 'index': index in the scene file
        Returns None if not found.
    
    Example:
        >>> pos_info = get_object_position("107733960_175999701", "couch_0")
        >>> if pos_info:
        ...     x, y, z = pos_info['position']
        ...     print(f"Position: ({x}, {y}, {z})")
    """
    if data_root is None:
        # Get project root (3 levels up from this file)
        data_root = Path(__file__).parent.parent.parent
    
    scene_path = data_root / f"data/hssd-hab/scenes-partnr-filtered/{scene_id}.scene_instance.json"
    
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}")
        return None
    
    with open(scene_path, 'r') as f:
        scene_data = json.load(f)
    
    # Clean up identifier - remove instance suffix if present
    identifier_clean = object_identifier.replace('_:0000', '').replace('_:0001', '')
    
    # Search in object_instances (regular objects/furniture)
    if 'object_instances' in scene_data:
        for i, obj in enumerate(scene_data['object_instances']):
            template = obj.get('template_name', '')
            
            # Match by handle or substring
            if (identifier_clean in template or 
                object_identifier in template or
                template.endswith(identifier_clean)):
                
                return {
                    'position': tuple(obj.get('translation', [0, 0, 0])),
                    'rotation': tuple(obj.get('rotation', [0, 0, 0, 1])),
                    'template_name': template,
                    'object_type': 'object',
                    'index': i,
                }
    
    # Search in articulated_object_instances (furniture with movable parts)
    if 'articulated_object_instances' in scene_data:
        for i, obj in enumerate(scene_data['articulated_object_instances']):
            template = obj.get('template_name', '')
            
            if (identifier_clean in template or 
                object_identifier in template or
                template.endswith(identifier_clean)):
                
                return {
                    'position': tuple(obj.get('translation', [0, 0, 0])),
                    'rotation': tuple(obj.get('rotation', [0, 0, 0, 1])),
                    'template_name': template,
                    'object_type': 'articulated_object',
                    'index': i,
                }
    
    return None


def get_receptacle_surface_position(
    scene_id: str,
    receptacle_handle: str,
    estimated_height_offset: float = 0.45,
    data_root: Optional[Path] = None
) -> Optional[Tuple[float, float, float]]:
    """
    Get the top surface position of a receptacle (where objects would be placed).
    
    Args:
        scene_id: The scene ID
        receptacle_handle: The receptacle handle or identifier
        estimated_height_offset: Estimated height of top surface (default: 0.45m for furniture)
        data_root: Optional root directory for data
    
    Returns:
        (x, y, z) tuple for the top surface position, or None if not found.
    
    Note:
        This uses an estimated height offset. For precise positions, use
        habitat-sim to get the actual bounding box.
    """
    obj_info = get_object_position(scene_id, receptacle_handle, data_root)
    
    if obj_info is None:
        return None
    
    x, base_y, z = obj_info['position']
    
    # For furniture, add estimated height to get top surface
    # For floor-level objects (y near 0), use the offset
    # For elevated objects, add a smaller offset
    if abs(base_y) < 0.1:  # Ground level furniture
        surface_y = base_y + estimated_height_offset
    else:  # Already elevated
        surface_y = base_y + 0.05  # Small offset for top surface
    
    return (x, surface_y, z)


def find_objects_by_type(
    scene_id: str,
    object_type: str,
    data_root: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Find all objects of a certain type in a scene.
    
    Args:
        scene_id: The scene ID
        object_type: Type to search for (substring match, e.g., "couch", "table", "chair")
        data_root: Optional root directory for data
    
    Returns:
        List of dictionaries with object information (same format as get_object_position)
    
    Example:
        >>> couches = find_objects_by_type("107733960_175999701", "couch")
        >>> print(f"Found {len(couches)} couches")
        >>> for couch in couches:
        ...     print(f"  Position: {couch['position']}")
    """
    if data_root is None:
        data_root = Path(__file__).parent.parent.parent
    
    scene_path = data_root / f"data/hssd-hab/scenes-partnr-filtered/{scene_id}.scene_instance.json"
    
    if not scene_path.exists():
        return []
    
    with open(scene_path, 'r') as f:
        scene_data = json.load(f)
    
    results = []
    object_type_lower = object_type.lower()
    
    # Search in object_instances
    if 'object_instances' in scene_data:
        for i, obj in enumerate(scene_data['object_instances']):
            template = obj.get('template_name', '').lower()
            
            if object_type_lower in template:
                results.append({
                    'position': tuple(obj.get('translation', [0, 0, 0])),
                    'rotation': tuple(obj.get('rotation', [0, 0, 0, 1])),
                    'template_name': obj.get('template_name', ''),
                    'object_type': 'object',
                    'index': i,
                })
    
    # Search in articulated_object_instances
    if 'articulated_object_instances' in scene_data:
        for i, obj in enumerate(scene_data['articulated_object_instances']):
            template = obj.get('template_name', '').lower()
            
            if object_type_lower in template:
                results.append({
                    'position': tuple(obj.get('translation', [0, 0, 0])),
                    'rotation': tuple(obj.get('rotation', [0, 0, 0, 1])),
                    'template_name': obj.get('template_name', ''),
                    'object_type': 'articulated_object',
                    'index': i,
                })
    
    return results


def get_transformation_matrix(position: Tuple[float, float, float]) -> List[List[float]]:
    """
    Create a transformation matrix for the given position (identity rotation).
    
    Args:
        position: (x, y, z) tuple
    
    Returns:
        4x4 transformation matrix as nested list (for episode rigid_objs format)
    
    Example:
        >>> matrix = get_transformation_matrix((-23.23, 0.45, 4.78))
        >>> print(matrix)
        [[1.0, 0.0, 0.0, -23.23],
         [0.0, 1.0, 0.0, 0.45],
         [0.0, 0.0, 1.0, 4.78],
         [0.0, 0.0, 0.0, 1.0]]
    """
    x, y, z = position
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ]


if __name__ == "__main__":
    # Example usage / testing
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python scene_utils.py <scene_id> <object_identifier>")
        print("\nExample:")
        print("  python scene_utils.py 107733960_175999701 couch_0")
        print("  python scene_utils.py 107733960_175999701 267879e5c0249074c75b2dba46348b87c4342223")
        sys.exit(1)
    
    scene_id = sys.argv[1]
    obj_id = sys.argv[2]
    
    print(f"Scene ID: {scene_id}")
    print(f"Looking for: {obj_id}")
    print("=" * 70)
    
    # Get object position
    result = get_object_position(scene_id, obj_id)
    
    if result:
        print(f"\n✓ Found object!")
        print(f"\nTemplate: {result['template_name']}")
        print(f"Type: {result['object_type']}")
        print(f"Index: {result['index']}")
        
        x, y, z = result['position']
        print(f"\nBase Position:")
        print(f"  X: {x:.5f}")
        print(f"  Y: {y:.5f}")
        print(f"  Z: {z:.5f}")
        
        qx, qy, qz, qw = result['rotation']
        print(f"\nRotation (quaternion):")
        print(f"  [{qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}]")
        
        # Get surface position
        surface_pos = get_receptacle_surface_position(scene_id, obj_id)
        if surface_pos:
            sx, sy, sz = surface_pos
            print(f"\nTop Surface Position (for placing objects):")
            print(f"  X: {sx:.5f}")
            print(f"  Y: {sy:.5f}")
            print(f"  Z: {sz:.5f}")
            
            print(f"\nTransformation Matrix:")
            matrix = get_transformation_matrix(surface_pos)
            print("[")
            for row in matrix:
                print(f"  {row},")
            print("]")
    else:
        print("\n✗ Object not found")
        print("\nTrying to search by type...")
        
        # Try to find similar objects
        results = find_objects_by_type(scene_id, obj_id)
        if results:
            print(f"\nFound {len(results)} matching objects:")
            for i, obj in enumerate(results[:5]):  # Show first 5
                print(f"\n{i+1}. {obj['template_name']}")
                x, y, z = obj['position']
                print(f"   Position: ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"\nNo objects found matching '{obj_id}'")

