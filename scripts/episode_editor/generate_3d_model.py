#!/usr/bin/env python3
"""
3D Apartment Model Generator

Generates an interactive 3D visualization of an apartment from a partnr-planner episode file
or scene ID. Loads actual GLB meshes for objects/furniture and displays rooms as colored
wireframe boxes.

Visual indicators:
  - RED bboxes: Scene furniture (from scene_instance.json)
  - GREEN bboxes + arrows: Episode objects (from episode rigid_objs)
  - ORANGE spheres + arrows: Missing objects (GLB file not found)

Usage:
    conda activate habitat

    # From episode file (recommended - includes episode objects)
    python generate_3d_model.py --episode data/versioned_data/partnr_episodes/v0_0/val_mini.json.gz

    # With specific episode ID in multi-episode file
    python generate_3d_model.py --episode data/.../train.json.gz --episode-id 42

    # Legacy: Scene ID only (no episode objects)
    python generate_3d_model.py --scene-id 102344193

Examples:
    python generate_3d_model.py --episode path/to/episode.json.gz
    python generate_3d_model.py --episode path/to/episode.json.gz --episode-id 5
    python generate_3d_model.py --scene-id 102344193 --export apartment.ply
    python generate_3d_model.py --episode ... --no-scene-objects  # Only episode objects
    python generate_3d_model.py --episode ... --no-arrows  # Hide green arrows
"""

import argparse
import gzip
import json
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration paths for the partnr-planner data."""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Get project root (2 levels up from this file: scripts/episode_editor -> project root)
            project_root = Path(__file__).parent.parent.parent

        self.project_root = project_root
        self.data_root = project_root / "data"

        # Scene files (try new-scenes-partnr-filtered first, fallback to scenes-partnr-filtered)
        self.scene_dirs = [
            self.data_root / "hssd-hab/new-scenes-partnr-filtered",
            self.data_root / "hssd-hab/scenes-partnr-filtered"
        ]

        # Versioned data paths
        self.versioned_data = self.data_root / "versioned_data/hssd-hab"
        self.objects_dir = self.versioned_data / "objects"
        self.urdf_dir = self.versioned_data / "urdf"
        self.stages_dir = self.versioned_data / "stages"
        self.semantics_dir = self.versioned_data / "semantics/scenes"


# ============================================================================
# Color Palette
# ============================================================================

def get_room_color_palette() -> List[Tuple[str, List[float]]]:
    """Get a predefined color palette for rooms."""
    return [
        ("Blue", [0.2, 0.4, 0.9]),
        ("Green", [0.2, 0.7, 0.3]),
        ("Orange", [0.9, 0.5, 0.1]),
        ("Cyan", [0.1, 0.7, 0.8]),
        ("Purple", [0.6, 0.2, 0.8]),
        ("Yellow", [0.9, 0.8, 0.2]),
        ("Magenta", [0.9, 0.2, 0.6]),
        ("Teal", [0.2, 0.6, 0.6]),
        ("Red", [0.9, 0.2, 0.2]),
        ("Lime", [0.5, 0.9, 0.2]),
        ("Pink", [0.9, 0.4, 0.6]),
        ("Brown", [0.6, 0.3, 0.1]),
        ("Navy", [0.1, 0.2, 0.6]),
        ("Coral", [0.9, 0.5, 0.3]),
        ("Gold", [0.9, 0.75, 0.1]),
    ]


def get_room_color(room_label: str, room_index: int) -> Tuple[str, List[float]]:
    """Get color for a room based on label or index."""
    # Label-based color mapping for common rooms
    label_colors = {
        "living": ("Blue", [0.2, 0.4, 0.9]),
        "bedroom": ("Green", [0.2, 0.7, 0.3]),
        "kitchen": ("Orange", [0.9, 0.5, 0.1]),
        "bathroom": ("Cyan", [0.1, 0.7, 0.8]),
        "office": ("Purple", [0.6, 0.2, 0.8]),
        "hallway": ("Yellow", [0.9, 0.8, 0.2]),
        "closet": ("Magenta", [0.9, 0.2, 0.6]),
        "laundry": ("Teal", [0.2, 0.6, 0.6]),
        "dining": ("Coral", [0.9, 0.5, 0.3]),
        "garage": ("Brown", [0.6, 0.3, 0.1]),
    }

    label_lower = room_label.lower()
    for key, color in label_colors.items():
        if key in label_lower:
            return color

    # Fallback to palette cycling
    palette = get_room_color_palette()
    return palette[room_index % len(palette)]


# ============================================================================
# Path Resolution Functions
# ============================================================================

def find_scene_file(config: Config, scene_id: str) -> Optional[Path]:
    """Find the scene instance JSON file for a given scene ID."""
    for scene_dir in config.scene_dirs:
        scene_path = scene_dir / f"{scene_id}.scene_instance.json"
        if scene_path.exists():
            return scene_path
    return None


def get_object_glb_path(config: Config, template_name: str) -> Optional[Path]:
    """
    Get the GLB file path for a regular object.
    Objects are stored as: objects/{first_char}/{hash}.glb
    """
    if not template_name:
        return None

    first_char = template_name[0].lower()
    glb_path = config.objects_dir / first_char / f"{template_name}.glb"

    if glb_path.exists():
        return glb_path
    return None


def get_articulated_object_glb_paths(config: Config, template_name: str) -> List[Path]:
    """
    Get all GLB file paths for an articulated object.
    Articulated objects are in: urdf/{hash}/ with multiple GLB files.
    Returns list of GLB paths, excluding receptacle meshes.
    """
    if not template_name:
        return []

    urdf_dir = config.urdf_dir / template_name
    if not urdf_dir.exists():
        return []

    # Get all GLB files, excluding receptacle meshes
    glb_files = [f for f in urdf_dir.glob("*.glb")
                 if "receptacle" not in f.name.lower()]

    return glb_files


def get_semantic_config_path(config: Config, scene_id: str) -> Optional[Path]:
    """Get the semantic config JSON file path for room definitions."""
    semantic_path = config.semantics_dir / f"{scene_id}.semantic_config.json"
    if semantic_path.exists():
        return semantic_path
    return None


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_scene_data(scene_path: Path) -> Dict[str, Any]:
    """Load scene instance JSON data."""
    with open(scene_path, 'r') as f:
        return json.load(f)


def load_semantic_config(semantic_path: Path) -> Dict[str, Any]:
    """Load semantic config for room definitions."""
    with open(semantic_path, 'r') as f:
        return json.load(f)


def load_episode_file(episode_path: Path) -> Dict[str, Any]:
    """Load episode data from JSON or gzipped JSON file."""
    if str(episode_path).endswith('.gz'):
        with gzip.open(episode_path, 'rt') as f:
            return json.load(f)
    else:
        with open(episode_path, 'r') as f:
            return json.load(f)


def extract_episode(data: Dict[str, Any], episode_id: Optional[str] = None,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Extract a specific episode from dataset.

    Args:
        data: Full dataset containing {"episodes": [...], "config": ...}
        episode_id: Specific episode ID to extract. If None, returns first episode.
        verbose: Print info about available episodes

    Returns:
        Single episode dict
    """
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError("No episodes found in file")

    # Get list of episode IDs
    episode_ids = [ep.get("episode_id", str(i)) for i, ep in enumerate(episodes)]

    if verbose:
        print(f"Found {len(episodes)} episode(s) in file")
        if len(episodes) <= 10:
            print(f"  Episode IDs: {episode_ids}")
        else:
            print(f"  Episode IDs: {episode_ids[:5]} ... {episode_ids[-5:]}")

    if episode_id:
        for ep in episodes:
            if str(ep.get("episode_id")) == str(episode_id):
                if verbose:
                    print(f"  Using episode: {episode_id}")
                return ep
        raise ValueError(f"Episode '{episode_id}' not found. Available: {episode_ids}")

    # Return first episode
    if verbose:
        print(f"  Using first episode: {episode_ids[0]}")
    return episodes[0]


# ============================================================================
# Transformation Functions
# ============================================================================

def quaternion_to_rotation_matrix(quat: List[float]) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
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


def create_4x4_transform(translation: List[float], rotation: List[float],
                         scale: Optional[List[float]] = None) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)

    # Rotation
    R = quaternion_to_rotation_matrix(rotation)
    T[:3, :3] = R

    # Translation
    T[:3, 3] = translation

    return T


# ============================================================================
# Mesh Loading Functions
# ============================================================================

def load_glb_mesh(glb_path: Path) -> Optional[o3d.geometry.TriangleMesh]:
    """Load a GLB file using trimesh and convert to Open3D mesh."""
    if not glb_path.exists():
        return None

    try:
        scene = trimesh.load(str(glb_path), force='scene')

        # Handle both single mesh and scene with multiple meshes
        if isinstance(scene, trimesh.Scene):
            # Combine all meshes in the scene
            meshes = []
            for name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)

            if not meshes:
                return None

            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        # Convert to Open3D
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)

        # Flip triangle winding order to fix inverted normals
        faces = np.asarray(mesh.faces)
        faces_flipped = faces[:, [0, 2, 1]]  # Reverse winding order
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_flipped)

        # Try to get vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            # Default gray color
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])

        o3d_mesh.compute_vertex_normals()
        return o3d_mesh

    except Exception as e:
        print(f"  Warning: Failed to load {glb_path.name}: {e}")
        return None


def load_and_transform_object(config: Config, template_name: str,
                              translation: List[float], rotation: List[float],
                              scale: Optional[List[float]] = None,
                              is_articulated: bool = False) -> List[o3d.geometry.TriangleMesh]:
    """
    Load an object mesh and apply transformation.
    Returns list of meshes (articulated objects may have multiple parts).
    """
    meshes = []

    if is_articulated:
        glb_paths = get_articulated_object_glb_paths(config, template_name)
    else:
        glb_path = get_object_glb_path(config, template_name)
        glb_paths = [glb_path] if glb_path else []

    for glb_path in glb_paths:
        mesh = load_glb_mesh(glb_path)
        if mesh is None:
            continue

        # Fix upside-down meshes: Rotate 180 degrees around X-axis
        # This corrects for coordinate system differences (Y-up vs Z-up)
        flip_rotation = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        mesh.rotate(flip_rotation, center=[0, 0, 0])

        # Apply scale if provided
        if scale is not None:
            mesh.scale(scale[0], center=mesh.get_center())  # Uniform scale for simplicity

        # Apply rotation
        R = quaternion_to_rotation_matrix(rotation)
        mesh.rotate(R, center=[0, 0, 0])
        # Apply translation
        mesh.translate(translation)

        meshes.append(mesh)

    return meshes


# ============================================================================
# Room Wireframe Functions
# ============================================================================

def create_cylinder_between_points(start: np.ndarray, end: np.ndarray,
                                   radius: float, color: List[float]) -> Optional[o3d.geometry.TriangleMesh]:
    """Create a cylinder mesh between two 3D points."""
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return None

    # Create cylinder along Z-axis
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

    # Compute rotation to align Z-axis with direction
    direction_normalized = direction / length
    z_axis = np.array([0, 0, 1])

    # Rotation axis and angle
    rotation_axis = np.cross(z_axis, direction_normalized)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm > 1e-6:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1.0, 1.0))

        # Rodrigues' formula for rotation matrix
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        cylinder.rotate(R, center=[0, 0, 0])

    # Translate to midpoint
    midpoint = (start + end) / 2
    cylinder.translate(midpoint)

    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()

    return cylinder


def create_room_wireframe(min_bounds: List[float], max_bounds: List[float],
                          color: List[float], line_radius: float = 0.03) -> List[o3d.geometry.TriangleMesh]:
    """
    Create a wireframe box for a room using cylinder edges.
    Returns list of cylinder meshes forming the 12 edges.
    """
    min_pt = np.array(min_bounds)
    max_pt = np.array(max_bounds)

    # Define 8 corners of the bounding box
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],  # 0
        [max_pt[0], min_pt[1], min_pt[2]],  # 1
        [max_pt[0], min_pt[1], max_pt[2]],  # 2
        [min_pt[0], min_pt[1], max_pt[2]],  # 3
        [min_pt[0], max_pt[1], min_pt[2]],  # 4
        [max_pt[0], max_pt[1], min_pt[2]],  # 5
        [max_pt[0], max_pt[1], max_pt[2]],  # 6
        [min_pt[0], max_pt[1], max_pt[2]],  # 7
    ])

    # Define 12 edges as pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]

    cylinders = []
    for start_idx, end_idx in edges:
        cyl = create_cylinder_between_points(corners[start_idx], corners[end_idx],
                                             line_radius, color)
        if cyl is not None:
            cylinders.append(cyl)

    return cylinders


def create_room_geometries(config: Config, scene_id: str,
                           verbose: bool = True) -> Tuple[List[o3d.geometry.TriangleMesh], Dict[str, str]]:
    """
    Create room wireframes from semantic config data.
    Returns: (list of geometries, dict mapping room_name -> color_name)
    """
    geometries = []
    room_colors = {}

    semantic_path = get_semantic_config_path(config, scene_id)
    if semantic_path is None:
        print(f"Warning: Semantic config not found for scene {scene_id}")
        return geometries, room_colors

    semantic_data = load_semantic_config(semantic_path)
    regions = semantic_data.get("region_annotations", [])

    if verbose:
        print(f"\nCreating {len(regions)} room wireframes...")
        print("=" * 60)
        print("ROOM COLORS:")
        print("=" * 60)

    for i, region in enumerate(regions):
        room_name = region.get("name", f"room_{i}")
        room_label = region.get("label", room_name)
        min_bounds = region.get("min_bounds")
        max_bounds = region.get("max_bounds")

        if min_bounds is None or max_bounds is None:
            continue

        # Get color for this room
        color_name, rgb = get_room_color(room_label, i)
        room_colors[room_name] = color_name

        if verbose:
            print(f"  {room_name:30s} -> {color_name}")

        # Create wireframe
        cylinders = create_room_wireframe(min_bounds, max_bounds, rgb)
        geometries.extend(cylinders)

    if verbose:
        print("=" * 60)

    return geometries, room_colors


def create_bounding_box_wireframe(mesh: o3d.geometry.TriangleMesh,
                                   color: List[float] = [1.0, 0.0, 0.0],
                                   line_radius: float = 0.02) -> List[o3d.geometry.TriangleMesh]:
    """
    Create a wireframe bounding box for a given mesh.
    Returns list of cylinder meshes forming the 12 edges of the bounding box.
    """
    # Get axis-aligned bounding box
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bounds = aabb.get_min_bound()
    max_bounds = aabb.get_max_bound()

    return create_room_wireframe(min_bounds.tolist(), max_bounds.tolist(), color, line_radius)


def create_arrow_pointing_down(position: List[float], color: List[float] = [0.0, 0.8, 0.2],
                                arrow_length: float = 5, cone_radius: float = 0.08,
                                shaft_radius: float = 0.02) -> List[o3d.geometry.TriangleMesh]:
    """
    Create a downward-pointing arrow above the given position.

    Args:
        position: [x, y, z] position of the object (arrow points to this)
        color: RGB color for the arrow
        arrow_length: Total length of the arrow
        cone_radius: Radius of the arrow cone (head)
        shaft_radius: Radius of the arrow shaft

    Returns:
        List of meshes forming the arrow (cone + cylinder)
    """
    geometries = []

    # Arrow tip is at object position + small offset above
    tip_y = position[1] + 0.1
    # Arrow starts above that
    start_y = tip_y + arrow_length

    # Create cone (arrow head) - points downward
    cone_height = arrow_length * 0.3
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    # Rotate to point downward (default points up along Z)
    cone.rotate(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), center=[0, 0, 0])
    # Position cone so tip is at the target
    cone.translate([position[0], tip_y + cone_height, position[2]])
    cone.paint_uniform_color(color)
    cone.compute_vertex_normals()
    geometries.append(cone)

    # Create shaft (cylinder)
    shaft_length = arrow_length - cone_height
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=shaft_radius, height=shaft_length)
    # Rotate to align with Y-axis
    shaft.rotate(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), center=[0, 0, 0])
    # Position shaft above the cone
    shaft.translate([position[0], tip_y + cone_height + shaft_length / 2, position[2]])
    shaft.paint_uniform_color(color)
    shaft.compute_vertex_normals()
    geometries.append(shaft)

    return geometries


def create_placeholder_sphere(position: List[float], color: List[float] = [1.0, 0.5, 0.0],
                               radius: float = 0.1) -> o3d.geometry.TriangleMesh:
    """
    Create a small sphere as a placeholder for missing objects.

    Args:
        position: [x, y, z] position
        color: RGB color (default orange for visibility)
        radius: Sphere radius

    Returns:
        Sphere mesh
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


# ============================================================================
# Object Loading Functions
# ============================================================================

def load_all_objects(config: Config, scene_data: Dict[str, Any],
                     include_bboxes: bool = True,
                     verbose: bool = True) -> Tuple[List[o3d.geometry.TriangleMesh], List[o3d.geometry.TriangleMesh], Dict[str, int]]:
    """
    Load all objects from a scene.
    Returns: (list of object meshes, list of bounding box wireframes, stats dict with counts)
    """
    meshes = []
    bboxes = []
    stats = {"loaded": 0, "failed": 0, "total": 0, "bboxes": 0}

    # Bounding box color (red wireframes)
    bbox_color = [1.0, 0.0, 0.0]

    # Load regular objects
    object_instances = scene_data.get("object_instances", [])
    if verbose:
        print(f"\nLoading {len(object_instances)} regular objects...")

    for obj in object_instances:
        stats["total"] += 1
        template_name = obj.get("template_name", "")
        translation = obj.get("translation", [0, 0, 0])
        rotation = obj.get("rotation", [0, 0, 0, 1])
        scale = obj.get("non_uniform_scale")

        obj_meshes = load_and_transform_object(
            config, template_name, translation, rotation, scale, is_articulated=False
        )

        if obj_meshes:
            meshes.extend(obj_meshes)
            stats["loaded"] += 1

            # Create bounding boxes for each mesh
            if include_bboxes:
                for mesh in obj_meshes:
                    bbox_wireframes = create_bounding_box_wireframe(mesh, bbox_color)
                    bboxes.extend(bbox_wireframes)
                    stats["bboxes"] += 1
        else:
            stats["failed"] += 1

    # Load articulated objects
    articulated_instances = scene_data.get("articulated_object_instances", [])
    if verbose:
        print(f"Loading {len(articulated_instances)} articulated objects...")

    for obj in articulated_instances:
        stats["total"] += 1
        template_name = obj.get("template_name", "")
        translation = obj.get("translation", [0, 0, 0])
        rotation = obj.get("rotation", [0, 0, 0, 1])

        obj_meshes = load_and_transform_object(
            config, template_name, translation, rotation, scale=None, is_articulated=True
        )

        if obj_meshes:
            meshes.extend(obj_meshes)
            stats["loaded"] += 1

            # Create bounding boxes for each mesh
            if include_bboxes:
                for mesh in obj_meshes:
                    bbox_wireframes = create_bounding_box_wireframe(mesh, bbox_color)
                    bboxes.extend(bbox_wireframes)
                    stats["bboxes"] += 1
        else:
            stats["failed"] += 1

    if verbose:
        print(f"  Loaded: {stats['loaded']}/{stats['total']} objects")
        if stats["failed"] > 0:
            print(f"  Failed: {stats['failed']} objects (missing GLB files)")
        if include_bboxes:
            print(f"  Created {stats['bboxes']} bounding boxes")

    return meshes, bboxes, stats


def load_episode_rigid_objects(config: Config, rigid_objs: List,
                                include_bboxes: bool = True,
                                include_arrows: bool = True,
                                verbose: bool = True) -> Tuple[List[o3d.geometry.TriangleMesh],
                                                                List[o3d.geometry.TriangleMesh],
                                                                Dict[str, Any]]:
    """
    Load rigid objects from episode's rigid_objs list.

    Each item in rigid_objs is: [template_name, 4x4_transform_matrix]
    where transform_matrix is:
        [[r00, r01, r02, x],
         [r10, r11, r12, y],
         [r20, r21, r22, z],
         [0.0, 0.0, 0.0, 1.0]]

    Args:
        config: Config object with paths
        rigid_objs: List of [template_name, transform_matrix] pairs
        include_bboxes: Whether to create bounding box wireframes
        include_arrows: Whether to create green arrows pointing to objects
        verbose: Print progress messages

    Returns:
        (list of object meshes, list of indicator geometries, stats dict)
    """
    meshes = []
    indicators = []  # Arrows, bboxes, and placeholder spheres
    stats = {"loaded": 0, "failed": 0, "total": 0, "bboxes": 0, "arrows": 0, "placeholders": 0}
    failed_objects = []  # Track failed objects for reporting

    # Colors
    bbox_color = [0.0, 0.8, 0.2]  # Green for bboxes
    arrow_color = [0.0, 0.8, 0.2]  # Green for arrows
    placeholder_color = [1.0, 0.5, 0.0]  # Orange for missing object placeholders

    if verbose:
        print(f"\nLoading {len(rigid_objs)} episode objects...")

    for obj_data in rigid_objs:
        stats["total"] += 1

        if not isinstance(obj_data, list) or len(obj_data) < 2:
            stats["failed"] += 1
            failed_objects.append({"name": "invalid_data", "position": [0, 0, 0]})
            continue

        template_name = obj_data[0]
        transform_matrix = np.array(obj_data[1])

        # Clean template name (remove .object_config.json suffix if present)
        original_name = template_name
        if template_name.endswith('.object_config.json'):
            template_name = template_name.replace('.object_config.json', '')

        # Extract translation from column 3 (rows 0-2)
        translation = transform_matrix[:3, 3].tolist()

        # Extract rotation matrix (3x3 submatrix)
        rotation_matrix = transform_matrix[:3, :3]

        # Load mesh
        glb_path = get_object_glb_path(config, template_name)
        if glb_path is None:
            stats["failed"] += 1
            failed_objects.append({"name": original_name, "position": translation})
            # Create placeholder sphere for missing object
            sphere = create_placeholder_sphere(translation, placeholder_color)
            indicators.append(sphere)
            stats["placeholders"] += 1
            # Also add arrow to placeholder
            if include_arrows:
                arrow_geoms = create_arrow_pointing_down(translation, placeholder_color)
                indicators.extend(arrow_geoms)
            continue

        mesh = load_glb_mesh(glb_path)
        if mesh is None:
            stats["failed"] += 1
            failed_objects.append({"name": original_name, "position": translation})
            # Create placeholder sphere for failed load
            sphere = create_placeholder_sphere(translation, placeholder_color)
            indicators.append(sphere)
            stats["placeholders"] += 1
            if include_arrows:
                arrow_geoms = create_arrow_pointing_down(translation, placeholder_color)
                indicators.extend(arrow_geoms)
            continue

        # Fix upside-down meshes: Rotate 180 degrees around X-axis
        flip_rotation = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        mesh.rotate(flip_rotation, center=[0, 0, 0])

        # Apply rotation from transform matrix
        mesh.rotate(rotation_matrix, center=[0, 0, 0])

        # Apply translation
        mesh.translate(translation)

        meshes.append(mesh)
        stats["loaded"] += 1

        # Create bounding box
        if include_bboxes:
            bbox_wireframes = create_bounding_box_wireframe(mesh, bbox_color)
            indicators.extend(bbox_wireframes)
            stats["bboxes"] += 1

        # Create arrow pointing to object
        if include_arrows:
            arrow_geoms = create_arrow_pointing_down(translation, arrow_color)
            indicators.extend(arrow_geoms)
            stats["arrows"] += 1

    if verbose:
        print(f"  Loaded: {stats['loaded']}/{stats['total']} episode objects")
        if stats["failed"] > 0:
            print(f"  Failed: {stats['failed']} objects (missing GLB files)")
            print(f"  Placeholders created: {stats['placeholders']} (ORANGE spheres)")
            print("\n  Missing GLB files:")
            for obj in failed_objects:
                pos = obj['position']
                print(f"    - {obj['name'][:50]}... at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        if include_bboxes:
            print(f"  Created {stats['bboxes']} bounding boxes (GREEN)")
        if include_arrows:
            print(f"  Created {stats['arrows']} arrows (GREEN)")

    # Store failed objects in stats for metadata
    stats["failed_objects"] = failed_objects

    return meshes, indicators, stats


# ============================================================================
# Main Assembly Function
# ============================================================================

def generate_apartment_model(scene_id: Optional[str] = None,
                             episode_path: Optional[Path] = None,
                             episode_id: Optional[str] = None,
                             include_rooms: bool = True,
                             include_objects: bool = True,
                             include_episode_objects: bool = True,
                             include_bboxes: bool = True,
                             include_arrows: bool = True,
                             verbose: bool = True) -> Tuple[List[o3d.geometry.Geometry], Dict[str, Any]]:
    """
    Generate complete 3D model of an apartment.

    Args:
        scene_id: Scene ID (e.g., "102344193") - used if no episode_path
        episode_path: Path to episode file (.json or .json.gz)
        episode_id: Specific episode ID within file (default: first episode)
        include_rooms: Include room wireframes
        include_objects: Include furniture/object meshes from scene
        include_episode_objects: Include rigid objects from episode
        include_bboxes: Include 3D bounding boxes around furniture
        include_arrows: Include green arrows pointing to episode objects
        verbose: Print progress messages

    Returns:
        Tuple of (list of all geometries, metadata dict)
    """
    config = Config()
    geometries = []
    episode_data = None
    rigid_objs = []

    # Load episode if provided
    if episode_path is not None:
        if verbose:
            print(f"Loading episode file: {episode_path}")
        episode_file_data = load_episode_file(episode_path)
        episode_data = extract_episode(episode_file_data, episode_id, verbose)
        scene_id = episode_data.get("scene_id")
        rigid_objs = episode_data.get("rigid_objs", [])
        if verbose:
            print(f"  Scene ID from episode: {scene_id}")
            print(f"  Rigid objects in episode: {len(rigid_objs)}")

    if scene_id is None:
        print("Error: No scene_id provided and no episode file specified")
        return geometries, {}

    metadata = {
        "scene_id": scene_id,
        "episode_id": episode_id or (episode_data.get("episode_id") if episode_data else None),
        "room_colors": {},
        "object_stats": {},
        "episode_object_stats": {},
    }

    # Find and load scene file
    scene_path = find_scene_file(config, scene_id)
    if scene_path is None:
        print(f"Error: Scene file not found for scene_id: {scene_id}")
        print(f"Searched in: {[str(d) for d in config.scene_dirs]}")
        return geometries, metadata

    if verbose:
        print(f"Loading scene: {scene_path.name}")

    scene_data = load_scene_data(scene_path)

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)

    # Create room wireframes
    if include_rooms:
        room_geometries, room_colors = create_room_geometries(config, scene_id, verbose)
        geometries.extend(room_geometries)
        metadata["room_colors"] = room_colors

    # Load scene objects (furniture)
    if include_objects:
        object_meshes, object_bboxes, object_stats = load_all_objects(
            config, scene_data, include_bboxes=include_bboxes, verbose=verbose
        )
        geometries.extend(object_meshes)
        if include_bboxes:
            geometries.extend(object_bboxes)
        metadata["object_stats"] = object_stats

    # Load episode rigid objects (objects placed in the episode)
    if include_episode_objects and rigid_objs:
        ep_meshes, ep_indicators, ep_stats = load_episode_rigid_objects(
            config, rigid_objs,
            include_bboxes=include_bboxes,
            include_arrows=include_arrows,
            verbose=verbose
        )
        geometries.extend(ep_meshes)
        # ep_indicators includes bboxes, arrows, and placeholder spheres
        geometries.extend(ep_indicators)
        metadata["episode_object_stats"] = ep_stats

    if verbose:
        print(f"\nTotal geometries: {len(geometries)}")

    return geometries, metadata


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_apartment(geometries: List[o3d.geometry.Geometry],
                        metadata: Dict[str, Any],
                        window_title: str = "3D Apartment Viewer"):
    """Open interactive Open3D visualization."""
    if not geometries:
        print("No geometries to visualize.")
        return

    print("\n" + "=" * 60)
    print("VISUALIZATION CONTROLS:")
    print("=" * 60)
    print("  Mouse left drag:  Rotate view")
    print("  Mouse right drag: Pan/translate")
    print("  Mouse wheel:      Zoom in/out")
    print("  H:                Show help menu")
    print("  Q or ESC:         Quit")
    print("=" * 60 + "\n")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1920, height=1080)

    # Add all geometries
    for geom in geometries:
        vis.add_geometry(geom)

    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    opt.mesh_show_back_face = True
    opt.point_size = 5.0

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.5)

    # Run visualization
    vis.run()
    vis.destroy_window()


def export_combined_mesh(geometries: List[o3d.geometry.Geometry],
                         output_path: Path):
    """Export all triangle meshes to a combined file."""
    # Filter to only TriangleMesh objects
    meshes = [g for g in geometries if isinstance(g, o3d.geometry.TriangleMesh)]

    if not meshes:
        print("No meshes to export.")
        return

    # Combine all meshes
    combined = meshes[0]
    for mesh in meshes[1:]:
        combined += mesh

    # Export
    o3d.io.write_triangle_mesh(str(output_path), combined)
    print(f"Exported combined mesh to: {output_path}")
    print(f"  - {len(meshes)} meshes combined")
    print(f"  - {len(combined.vertices)} vertices")
    print(f"  - {len(combined.triangles)} triangles")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D model of apartment from episode file or scene ID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From episode file (recommended - includes episode objects)
  python generate_3d_model.py --episode data/versioned_data/partnr_episodes/v0_0/val_mini.json.gz

  # With specific episode ID in multi-episode file
  python generate_3d_model.py --episode data/.../train.json.gz --episode-id 42

  # Legacy: Scene ID only (no episode objects)
  python generate_3d_model.py --scene-id 102344193

  # Other options
  python generate_3d_model.py --episode ... --export apartment.ply
  python generate_3d_model.py --episode ... --no-scene-objects  # Only episode objects
  python generate_3d_model.py --episode ... --no-episode-objects  # Only scene furniture
        """
    )
    # Input options (mutually exclusive in practice)
    parser.add_argument('--episode', type=str, default=None,
                        help='Path to episode file (.json or .json.gz)')
    parser.add_argument('--episode-id', type=str, default=None,
                        help='Episode ID within file (default: first episode)')
    parser.add_argument('--scene-id', type=str, default=None,
                        help='Scene ID (legacy mode, no episode objects)')

    # Display options
    parser.add_argument('--export', type=str, default=None,
                        help='Export combined mesh to file (e.g., output.ply)')
    parser.add_argument('--no-rooms', action='store_true',
                        help='Do not include room wireframes')
    parser.add_argument('--no-scene-objects', action='store_true',
                        help='Do not include scene furniture (RED bboxes)')
    parser.add_argument('--no-episode-objects', action='store_true',
                        help='Do not include episode objects (GREEN bboxes/arrows)')
    parser.add_argument('--no-bboxes', action='store_true',
                        help='Do not include 3D bounding boxes')
    parser.add_argument('--no-arrows', action='store_true',
                        help='Do not include green arrows pointing to episode objects')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Do not open visualization window (export only)')

    args = parser.parse_args()

    # Validate input
    if args.episode is None and args.scene_id is None:
        parser.error("Must specify either --episode or --scene-id")

    print("=" * 60)
    print("3D APARTMENT MODEL GENERATOR")
    print("=" * 60)
    if args.episode:
        print(f"Episode file: {args.episode}")
        if args.episode_id:
            print(f"Episode ID: {args.episode_id}")
    else:
        print(f"Scene ID: {args.scene_id}")
    print(f"Include rooms: {not args.no_rooms}")
    print(f"Include scene objects: {not args.no_scene_objects}")
    print(f"Include episode objects: {not args.no_episode_objects}")
    print(f"Include bounding boxes: {not args.no_bboxes}")
    print(f"Include arrows: {not args.no_arrows}")
    print("=" * 60)

    # Generate model
    geometries, metadata = generate_apartment_model(
        scene_id=args.scene_id,
        episode_path=Path(args.episode) if args.episode else None,
        episode_id=args.episode_id,
        include_rooms=not args.no_rooms,
        include_objects=not args.no_scene_objects,
        include_episode_objects=not args.no_episode_objects,
        include_bboxes=not args.no_bboxes,
        include_arrows=not args.no_arrows
    )

    if not geometries:
        print("\nNo geometries generated. Check episode file or scene ID.")
        return

    # Export if requested
    if args.export:
        export_combined_mesh(geometries, Path(args.export))

    # Visualize unless disabled
    if not args.no_visualize:
        scene_id = metadata.get("scene_id", "unknown")
        episode_id = metadata.get("episode_id")
        title = f"3D Apartment - Scene {scene_id}"
        if episode_id:
            title += f" (Episode {episode_id})"
        visualize_apartment(geometries, metadata, window_title=title)

    print("\nDone.")


if __name__ == "__main__":
    main()
