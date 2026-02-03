#!/usr/bin/env python3
"""
3D Apartment Visualizer using Open3D

This script loads spatial data JSON and creates an interactive 3D visualization
of the entire apartment with rooms, furniture, and objects.

Usage:
    python visualize_3d_apartment.py [--file spatial_data_101.json]
"""

import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import threading
import time


# Global variable to store picked points
picked_points_list = []


def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    
    Args:
        quat: Quaternion as [x, y, z, w]
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    x, y, z, w = quat

    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return R


def create_box_mesh(center, size, color, wireframe=False, rotation=None):
    """Create a box mesh at the given center with given size and color.
    
    Args:
        center: Center position [x, y, z]
        size: Size [width, height, depth]
        color: RGB color [r, g, b]
        wireframe: If True, create wireframe box
        rotation: Optional quaternion [x, y, z, w] for rotation
    """
    # Ensure minimum size to avoid Open3D errors
    min_size = 0.01
    safe_size = [max(s, min_size) for s in size]

    mesh = o3d.geometry.TriangleMesh.create_box(
        width=safe_size[0], height=safe_size[1], depth=safe_size[2]
    )

    # Move box so center is at origin
    mesh.translate([-safe_size[0]/2, -safe_size[1]/2, -safe_size[2]/2])

    # Apply rotation if provided (rotate around origin first)
    if rotation is not None:
        R = quaternion_to_rotation_matrix(rotation)
        mesh.rotate(R, center=[0, 0, 0])

    # Then translate to desired position
    mesh.translate(center)

    if wireframe:
        # Create wireframe
        mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh.paint_uniform_color(color)
    else:
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()

    return mesh


def create_coordinate_frame(size=1.0):
    """Create a coordinate frame for reference."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def load_spatial_data(json_path):
    """Load spatial data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_color_name_and_rgb(index, total):
    """Get a color name and RGB value from a palette."""
    colors = [
        ("Red", [0.9, 0.1, 0.1]),
        ("Orange", [0.9, 0.5, 0.1]),
        ("Yellow", [0.9, 0.9, 0.1]),
        ("Lime", [0.5, 0.9, 0.1]),
        ("Green", [0.1, 0.9, 0.1]),
        ("Cyan", [0.1, 0.9, 0.9]),
        ("Blue", [0.1, 0.5, 0.9]),
        ("Purple", [0.5, 0.1, 0.9]),
        ("Magenta", [0.9, 0.1, 0.9]),
        ("Pink", [0.9, 0.1, 0.5]),
        ("Brown", [0.6, 0.3, 0.1]),
        ("Teal", [0.1, 0.6, 0.6]),
        ("Navy", [0.1, 0.2, 0.6]),
        ("Maroon", [0.6, 0.1, 0.2]),
        ("Olive", [0.5, 0.5, 0.1]),
        ("Coral", [0.9, 0.5, 0.3]),
        ("Salmon", [0.9, 0.6, 0.5]),
        ("Gold", [0.9, 0.8, 0.1]),
        ("Violet", [0.7, 0.1, 0.9]),
        ("Indigo", [0.3, 0.1, 0.6]),
    ]
    return colors[index % len(colors)]


def visualize_apartment(spatial_data, show_rooms=True, show_furniture=True, show_objects=True):
    """
    Create Open3D visualization from spatial data.
    
    Args:
        spatial_data: Dictionary containing spatial information
        show_rooms: Whether to show room boundaries
        show_furniture: Whether to show furniture/receptacles
        show_objects: Whether to show objects
    
    Returns:
        List of Open3D geometries to visualize
    """
    geometries = []

    # Calculate scene center from room bounds
    scene_min = np.array([float('inf'), float('inf'), float('inf')])
    scene_max = np.array([float('-inf'), float('-inf'), float('-inf')])

    for room_data in spatial_data.get('rooms', {}).values():
        if room_data.get('bounds'):
            bounds = room_data['bounds']
            scene_min = np.minimum(scene_min, bounds['min'])
            scene_max = np.maximum(scene_max, bounds['max'])

    scene_center = (scene_min + scene_max) / 2

    # Add coordinate frame at origin (smaller)
    coord_frame = create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)

    # Add grid/ground plane at actual floor level
    # Set default floor level, will be updated after processing objects
    grid_y = 0.0
    grid_size = 50
    grid_step = 1.0
    points = []
    lines = []

    # Create grid points and lines
    for i in range(-grid_size, grid_size + 1):
        # Lines parallel to X axis
        points.append([i * grid_step, grid_y, -grid_size * grid_step])
        points.append([i * grid_step, grid_y, grid_size * grid_step])
        lines.append([len(points) - 2, len(points) - 1])

        # Lines parallel to Z axis
        points.append([-grid_size * grid_step, grid_y, i * grid_step])
        points.append([grid_size * grid_step, grid_y, i * grid_step])
        lines.append([len(points) - 2, len(points) - 1])

    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    grid.paint_uniform_color([0.5, 0.5, 0.5])  # Lighter gray for better visibility
    geometries.append(grid)

    # Visualize rooms
    room_colors = {}
    if show_rooms and 'rooms' in spatial_data:
        print(f"\nAdding {len(spatial_data['rooms'])} rooms...")
        print("\n" + "=" * 80)
        print("ROOM COLOR LEGEND:")
        print("=" * 80)

        # Assign colors to rooms
        for i, room_name in enumerate(spatial_data['rooms'].keys()):
            color_name, rgb = get_color_name_and_rgb(i, len(spatial_data['rooms']))
            room_colors[room_name] = (color_name, rgb)
            print(f"  {room_name:30s} -> {color_name}")
        print("=" * 80 + "\n")

        for room_name, room_data in spatial_data['rooms'].items():
            if room_data.get('bounds') and room_data['bounds'].get('min') and room_data['bounds'].get('max'):
                min_pt = np.array(room_data['bounds']['min'])
                max_pt = np.array(room_data['bounds']['max'])
                size = max_pt - min_pt
                center = (min_pt + max_pt) / 2

                # Create line box (edges only) for room bounds
                # Define 8 corners of the bounding box
                corners = [
                    min_pt,
                    [max_pt[0], min_pt[1], min_pt[2]],
                    [max_pt[0], min_pt[1], max_pt[2]],
                    [min_pt[0], min_pt[1], max_pt[2]],
                    [min_pt[0], max_pt[1], min_pt[2]],
                    [max_pt[0], max_pt[1], min_pt[2]],
                    max_pt,
                    [min_pt[0], max_pt[1], max_pt[2]],
                ]

                # Define the 12 edges of the box
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
                ]

                # Create thick visible lines using cylinders
                color_name, rgb = room_colors[room_name]

                # Convert corners to numpy arrays
                corners_np = [np.array(c) for c in corners]

                # Create cylinders for each edge
                edge_pairs = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
                ]

                for start_idx, end_idx in edge_pairs:
                    start = corners_np[start_idx]
                    end = corners_np[end_idx]

                    # Create cylinder between two points
                    direction = end - start
                    length = np.linalg.norm(direction)

                    if length > 0:
                        # Create cylinder
                        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=length)

                        # Align cylinder with the edge
                        # Default cylinder is along Z-axis, need to rotate to align with direction
                        direction_normalized = direction / length
                        z_axis = np.array([0, 0, 1])

                        # Rotation axis and angle
                        rotation_axis = np.cross(z_axis, direction_normalized)
                        rotation_axis_norm = np.linalg.norm(rotation_axis)

                        if rotation_axis_norm > 1e-6:
                            rotation_axis = rotation_axis / rotation_axis_norm
                            angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1.0, 1.0))

                            # Create rotation matrix using Rodrigues' formula
                            K = np.array([
                                [0, -rotation_axis[2], rotation_axis[1]],
                                [rotation_axis[2], 0, -rotation_axis[0]],
                                [-rotation_axis[1], rotation_axis[0], 0]
                            ])
                            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

                            cylinder.rotate(R, center=[0, 0, 0])

                        # Translate to position
                        midpoint = (start + end) / 2
                        cylinder.translate(midpoint)
                        cylinder.paint_uniform_color(rgb)
                        cylinder.compute_vertex_normals()
                        geometries.append(cylinder)

    # Track minimum y position for grid placement
    min_y = float('inf')

    # Visualize furniture/receptacles
    furniture_colors = {}
    if show_furniture and 'receptacles' in spatial_data:
        print(f"\nAdding {len(spatial_data['receptacles'])} furniture items...")
        print("\n" + "=" * 80)
        print("FURNITURE COLOR LEGEND:")
        print("=" * 80)

        # Assign colors to furniture
        for i, furniture_name in enumerate(spatial_data['receptacles'].keys()):
            color_name, rgb = get_color_name_and_rgb(i, len(spatial_data['receptacles']))
            furniture_colors[furniture_name] = (color_name, rgb)
            print(f"  {furniture_name:30s} -> {color_name}")
        print("=" * 80 + "\n")

        for furniture_name, furniture_data in spatial_data['receptacles'].items():
            position = np.array(furniture_data['position'])
            rotation = furniture_data.get('rotation')  # Quaternion [x, y, z, w]

            # Use bounds if available (bounds are in local coordinates, need to add position)
            if furniture_data.get('bounds') and furniture_data.get('size'):
                # Bounds are in local space, just use the size directly
                size = np.array(furniture_data['size'])
                center = position  # Position is already the center in world space

                # Adjust center so object sits on its bottom face
                center[1] += size[1] / 2.0

                # Track minimum y for grid placement
                min_y = min(min_y, center[1] - size[1] / 2.0)
            else:
                size = np.array([0.5, 0.5, 0.5])
                center = position
                center[1] += size[1] / 2.0
                min_y = min(min_y, center[1] - size[1] / 2.0)

            # Get unique color for this furniture
            color_name, color = furniture_colors[furniture_name]

            # Create mesh box for furniture with rotation and unique color
            furniture_mesh = create_box_mesh(center, size, color, rotation=rotation)
            geometries.append(furniture_mesh)

    # Visualize objects
    object_colors = {}
    if show_objects and 'objects' in spatial_data and len(spatial_data['objects']) > 0:
        print(f"\nAdding {len(spatial_data['objects'])} objects...")
        print("\n" + "=" * 80)
        print("OBJECT COLOR LEGEND:")
        print("=" * 80)

        # Assign colors to objects
        for i, object_name in enumerate(spatial_data['objects'].keys()):
            color_name, rgb = get_color_name_and_rgb(i, len(spatial_data['objects']))
            object_colors[object_name] = (color_name, rgb)
            print(f"  {object_name:30s} -> {color_name}")
        print("=" * 80 + "\n")

        for object_name, object_data in spatial_data['objects'].items():
            position = np.array(object_data['position'])
            rotation = object_data.get('rotation')  # Quaternion [x, y, z, w]

            # Get size if available
            if object_data.get('size'):
                size = np.array(object_data['size'])
                # Adjust position so object sits on its bottom face
                position[1] += size[1] / 2.0
                min_y = min(min_y, position[1] - size[1] / 2.0)

            # Use size if available (bounds are in local coordinates)
            if object_data.get('size'):
                size = np.array(object_data['size'])
                center = position  # Position is already the center in world space
            else:
                size = np.array([0.2, 0.2, 0.2])
                center = position

            # Get unique color for this object
            color_name, color = object_colors[object_name]

            # Create mesh box for object with rotation and unique color
            object_mesh = create_box_mesh(center, size, color, rotation=rotation)
            geometries.append(object_mesh)

    # Print grid placement info
    if min_y != float('inf'):
        print(f"\n{'='*80}")
        print(f"GRID PLACEMENT INFO:")
        print(f"Floor level (min_y): {min_y:.3f}")
        print(f"Grid will be visible at y = {min_y:.3f}")
        print(f"{'='*80}\n")

    return geometries


class PointPickerCallback:
    """Callback for picking points in the visualization."""

    def __init__(self, geometries, spatial_data):
        self.geometries = geometries
        self.spatial_data = spatial_data
        self.picked_points = []

    def __call__(self, vis):
        # Get picked points
        picked = vis.get_picked_points()

        if picked:
            for point_idx in picked:
                # Get the 3D coordinate
                point = self.geometries[point_idx.index].get_center()
                self.picked_points.append(point)
                print(f"\nPicked point {len(self.picked_points)}:")
                print(f"  Position: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

                # Add a small sphere at the picked location
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                marker.translate(point)
                marker.paint_uniform_color([1.0, 0.0, 0.0])
                vis.add_geometry(marker)

        return False


def main():
    parser = argparse.ArgumentParser(description='Visualize apartment in 3D using spatial data')
    parser.add_argument('--file', type=str, default='static/spatial_data_101.json',
                       help='Path to spatial data JSON file')
    parser.add_argument('--no-rooms', action='store_true',
                       help='Hide rooms')
    parser.add_argument('--no-furniture', action='store_true',
                       help='Hide furniture')
    parser.add_argument('--no-objects', action='store_true',
                       help='Hide objects')

    args = parser.parse_args()

    # Load spatial data
    json_path = Path(args.file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return

    print(f"Loading spatial data from: {json_path}")
    spatial_data = load_spatial_data(json_path)
    print(f"Scene ID: {spatial_data.get('scene_id', 'Unknown')}")

    # Create geometries
    geometries = visualize_apartment(
        spatial_data,
        show_rooms=not args.no_rooms,
        show_furniture=not args.no_furniture,
        show_objects=not args.no_objects
    )

    print(f"\nTotal geometries: {len(geometries)}")
    print("\nControls:")
    print("  - Mouse left: Rotate")
    print("  - Mouse right: Translate")
    print("  - Mouse wheel: Zoom")
    print("  - Shift + Left click: Pick point on surface (shows 3D coordinates)")
    print("  - Ctrl + Left click: Also picks points")
    print("  - Press 'H' for help menu")
    print("  - Press 'Q' or ESC to quit")

    # Calculate scene bounds for better camera positioning
    scene_min = np.array([float('inf'), float('inf'), float('inf')])
    scene_max = np.array([float('-inf'), float('-inf'), float('-inf')])

    for room_data in spatial_data.get('rooms', {}).values():
        if room_data.get('bounds'):
            bounds = room_data['bounds']
            scene_min = np.minimum(scene_min, bounds['min'])
            scene_max = np.maximum(scene_max, bounds['max'])

    scene_center = (scene_min + scene_max) / 2
    scene_size = np.linalg.norm(scene_max - scene_min)

    print(f"Scene bounds: min={scene_min}, max={scene_max}")
    print(f"Scene center: {scene_center}")
    print(f"Scene size: {scene_size:.2f}")

    # Save combined geometry to file for debugging
    all_meshes = [g for g in geometries if isinstance(g, o3d.geometry.TriangleMesh)]
    if all_meshes:
        combined = all_meshes[0]
        for mesh in all_meshes[1:]:
            combined += mesh
        o3d.io.write_triangle_mesh("debug_scene.ply", combined)
        print(f"Saved debug scene to debug_scene.ply with {len(all_meshes)} meshes")

    # Create visualizer with editing mode
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="3D Apartment Viewer - Shift+Click to pick points", width=1920, height=1080)

    # Add all geometries
    for geom in geometries:
        vis.add_geometry(geom)

    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    opt.point_size = 10.0
    opt.line_width = 20.0  # Very thick lines for room edges
    opt.show_coordinate_frame = False  # Disable to avoid confusion
    opt.mesh_show_back_face = True

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])  # Look at origin first
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.5)

    # Print furniture positions for reference
    print("\n" + "=" * 80)
    print("FURNITURE POSITIONS (X, Y, Z):")
    print("=" * 80)
    for name, data in spatial_data.get('receptacles', {}).items():
        pos = data.get('position', [0, 0, 0])
        size = data.get('size', [0.5, 0.5, 0.5])
        # Adjusted position (bottom face)
        adj_y = pos[1] + size[1] / 2.0
        print(f"{name:30s} -> ({pos[0]:7.2f}, {adj_y:7.2f}, {pos[2]:7.2f})")
    print("=" * 80)

    # Run visualization
    print("\nStarting visualization...")
    print("\n" + "=" * 80)
    print("HOW TO PICK POINTS:")
    print("=" * 80)
    print("  1. Hold SHIFT key")
    print("  2. Click LEFT mouse button on a surface")
    print("  3. Keep adding points")
    print("  4. Close window (Q or ESC) when done")
    print("")
    print("OTHER CONTROLS:")
    print("  Mouse left drag: Rotate view")
    print("  Mouse right drag: Pan/translate")
    print("  Mouse wheel: Zoom in/out")
    print("  H: Show help menu")
    print("=" * 80 + "\n")

    vis.run()

    # Get picked points
    picked_indices = vis.get_picked_points()

    if picked_indices:
        print("\n" + "=" * 80)
        print(f"PICKED {len(picked_indices)} POINT(S):")
        print("=" * 80)

        # Collect all vertices from all meshes
        all_vertices = []
        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                vertices = np.asarray(geom.vertices)
                all_vertices.extend(vertices)

        # Print coordinates for each picked point
        picked_coords = []
        for i, picked_point in enumerate(picked_indices):
            idx = picked_point.index
            if idx < len(all_vertices):
                point = all_vertices[idx]
                picked_coords.append(point)
                print(f"\nPoint {i+1}:")
                print(f"  X: {point[0]:8.3f} m")
                print(f"  Y: {point[1]:8.3f} m (height)")
                print(f"  Z: {point[2]:8.3f} m")

        print("\n" + "=" * 80)

        # Now create new visualization with markers
        if picked_coords:
            print("\nRe-opening visualization with markers at picked points...")

            # Create marker spheres
            markers = []
            for i, coord in enumerate(picked_coords):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
                sphere.translate(coord)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
                sphere.compute_vertex_normals()
                markers.append(sphere)

            # Create new visualizer
            vis2 = o3d.visualization.Visualizer()
            vis2.create_window(window_name=f"3D Apartment with {len(picked_coords)} Picked Points",
                              width=1920, height=1080)

            # Add all original geometries
            for geom in geometries:
                vis2.add_geometry(geom)

            # Add marker spheres
            for marker in markers:
                vis2.add_geometry(marker)

            # Set same rendering options
            opt2 = vis2.get_render_option()
            opt2.background_color = np.array([1.0, 1.0, 1.0])
            opt2.point_size = 10.0
            opt2.line_width = 20.0
            opt2.mesh_show_back_face = True

            # Set camera
            ctr2 = vis2.get_view_control()
            ctr2.set_lookat([0, 0, 0])
            ctr2.set_up([0, 1, 0])
            ctr2.set_zoom(0.5)

            print("\n" + "=" * 80)
            print(f"RED SPHERES mark your {len(picked_coords)} picked point(s)")
            print("Close window when done viewing")
            print("=" * 80 + "\n")

            vis2.run()
            vis2.destroy_window()
    else:
        print("\n" + "=" * 80)
        print("NO POINTS WERE PICKED")
        print("=" * 80)
        print("\nRemember to:")
        print("  1. Hold SHIFT while clicking")
        print("  2. Click on furniture/surfaces (colored boxes)")
        print("  3. Make sure window has focus")
        print("=" * 80)

    vis.destroy_window()
    print("\nVisualization closed.")


if __name__ == "__main__":
    main()
