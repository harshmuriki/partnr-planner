#!/usr/bin/env python3

"""
Add Objects to Scene - A web-based tool for adding objects to PARTNR episodes with hierarchical visualization.

Usage:
    python scripts/episode_editor/add_objects_to_scene.py --dataset path/to/dataset.json.gz --episode-id 966

Then open http://localhost:5000 in your browser.
"""

import argparse
import gzip
import io
import json
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, jsonify, render_template, request, send_file, Response

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import scene utilities for getting furniture positions/rotations
try:
    from scripts.episode_editor import scene_utils
    SCENE_UTILS_AVAILABLE = True
except ImportError:
    SCENE_UTILS_AVAILABLE = False
    print("Warning: scene_utils not available. Object rotations will be identity.")

# Try to import habitat_sim for getting actual receptacle positions
try:
    import habitat_sim
    import magnum as mn
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("Warning: habitat_sim not available. Object positions will be approximate.")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global state
episode_data: Dict[str, Any] = {}
dataset_path: str = ""
episode_id: str = ""
metadata_dir: str = ""
save_dir: str = ""
hierarchy_data: Dict[str, Any] = {}
viz_image_paths: Dict[str, str] = {}
object_database: List[Dict[str, str]] = []

def load_dataset_file(path: str) -> Dict[str, Any]:
    """Load dataset from JSON or JSON.gz file."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def extract_episode_from_dataset(dataset_path: str, episode_id: str) -> Optional[Dict[str, Any]]:
    """Extract a specific episode from a dataset file."""
    dataset = load_dataset_file(dataset_path)
    episodes = dataset.get('episodes', [])

    for episode in episodes:
        ep_id = episode.get('episode_id')
        if str(ep_id) == str(episode_id) or ep_id == episode_id:
            return {
                "episodes": [episode],
                "config": dataset.get("config", {})
            }
    return None


def save_episode_gz(path: str, episode_data: Dict[str, Any]) -> None:
    """Save episode to JSON.gz file."""
    if "episodes" in episode_data and len(episode_data["episodes"]) > 0:
        single_episode_data = {
            "episodes": [episode_data["episodes"][0]],
            "config": episode_data.get("config", {})
        }
    else:
        single_episode_data = {"episodes": [episode_data], "config": {}}

    with gzip.open(path, 'wt') as f:
        json.dump(single_episode_data, f, indent=2)


def save_episode(path: str, data: Dict[str, Any]) -> None:
    """Save episode to JSON or JSON.gz file."""
    if path.endswith('.gz'):
        save_episode_gz(path, data)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def run_viz_generation(dataset_path: str, metadata_dir: str, save_path: str, episode_id: str) -> bool:
    """Run viz.py to generate hierarchy and PrediViz images."""
    print(f"Generating visualization for episode {episode_id}...")

    viz_cmd = [
        sys.executable,
        str(project_root / "scripts/episode_editor/viz.py"),
        "--dataset", dataset_path,
        "--metadata-dir", metadata_dir,
        "--save-path", save_path,
        "--episode-id", episode_id,
        "--save-hierarchy"
    ]

    try:
        result = subprocess.run(
            viz_cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"Error running viz.py: {result.stderr}")
            return False

        print("Visualization generated successfully")
        return True
    except subprocess.TimeoutExpired:
        print("Error: Visualization generation timed out")
        return False
    except Exception as e:
        print(f"Error running viz.py: {e}")
        return False


def load_hierarchy(viz_output_dir: str, episode_id: str) -> Dict[str, Any]:
    """Load hierarchy JSON from viz output."""
    hierarchy_path = Path(viz_output_dir) / f"viz_{episode_id}" / f"hierarchy_{episode_id}.json"

    if not hierarchy_path.exists():
        print(f"Warning: Hierarchy file not found at {hierarchy_path}")
        return {}

    with open(hierarchy_path, 'r') as f:
        return json.load(f)


def scan_viz_images(viz_output_dir: str, episode_id: str) -> Dict[str, str]:
    """Scan for generated PrediViz images."""
    viz_dir = Path(viz_output_dir) / f"viz_{episode_id}"
    image_paths = {}

    if not viz_dir.exists():
        return {}

    # Find all step images
    for i in range(100):  # Assume max 100 steps
        step_path = viz_dir / f"step_{i}.png"
        if step_path.exists():
            image_paths[f"step_{i}"] = str(step_path)
        else:
            break

    # Check for topdown map
    topdown_path = viz_dir / "topdown_map.png"
    if topdown_path.exists():
        image_paths["topdown"] = str(topdown_path)

    return image_paths


def scan_object_database() -> List[Dict[str, str]]:
    """Load objects from metadata CSV files."""
    import csv

    objects = []
    seen_ids = set()

    # Load from object_categories_one_per_class.csv
    fpmodels_csv = "scripts/episode_editor/objects_mapping/object_categories_one_per_class.csv"
    if Path(fpmodels_csv).exists():
        with open(fpmodels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = row.get("id", "").strip()
                category = row.get("clean_category", "").strip()

                if obj_id and obj_id not in seen_ids:
                    seen_ids.add(obj_id)
                    objects.append({
                        "id": obj_id,
                        "name": category if category else obj_id,
                        "category": category if category else "other",
                        "source": "fpmodels",
                        "super_category": category if category else "other",
                    })

    print(f"Loaded {len(objects)} objects from metadata CSVs")
    return sorted(objects, key=lambda x: (x.get("category", ""), x["id"]))


# Flask Routes

@app.route("/")
def index():
    """Main page."""
    return render_template("add_objects.html")


@app.route("/api/hierarchy")
def get_hierarchy():
    """Get hierarchical scene data."""
    return jsonify(hierarchy_data)


@app.route("/api/viz_image/<image_name>")
def get_viz_image(image_name):
    """Serve PrediViz images."""
    if image_name in viz_image_paths:
        return send_file(viz_image_paths[image_name], mimetype="image/png")
    return jsonify({"error": "Image not found"}), 404


@app.route("/api/categories")
def get_categories():
    """Get unique object categories."""
    categories = sorted(set(o["category"] for o in object_database))
    return jsonify(categories)


@app.route("/api/objects/search")
def search_objects():
    """Search objects by query and category."""
    query = request.args.get("q", "").lower()
    category = request.args.get("category", "")

    filtered = object_database
    if query:
        filtered = [
            o for o in filtered
            if query in o["id"].lower()
            or query in o.get("name", "").lower()
            or query in o.get("category", "").lower()
        ]
    if category:
        filtered = [o for o in filtered if o["category"] == category]

    return jsonify(filtered[:100])


@app.route("/api/rooms")
def get_rooms():
    """Get available rooms from hierarchy."""
    rooms = list(hierarchy_data.keys())
    return jsonify(sorted(rooms))


@app.route("/api/receptacles/<room>")
def get_receptacles(room):
    """Get receptacles in a specific room, filtered to only articulatable furniture."""
    if room not in hierarchy_data:
        return jsonify([])

    room_data = hierarchy_data[room]
    receptacles = []
    print("*"*40)

    for recep_name, recep_data in room_data.get("receptacles", {}).items():
        handle = recep_data.get("handle", "")

        # Extract base hash from handle to check if it's articulatable
        receptacles.append({
            "name": recep_name,
            "description": recep_data.get("description", ""),
            "handle": handle,
            "objects_count": len(recep_data.get("objects", []))
        })

    print("*"*40)

    return jsonify(receptacles)


@app.route("/api/articulated_furniture")
def get_articulated_furniture():
    """Get list of articulated furniture from metadata that can be used as doors."""
    import csv

    # Best recommended door options (curated list)
    recommended_doors = [
        {
            "hash": "d47861b542c4f2e8fe9adcf86d55e26e12d1a213",
            "num_doors": 3,
            "num_drawers": 0,
            "description": "Wardrobe0097 - 3 doors (RECOMMENDED)",
            "name": "Wardrobe0097"
        },
        {
            "hash": "00b0d5e167ae6b42666de010025efad4506563f1",
            "num_doors": 1,
            "num_drawers": 0,
            "description": "Oven - 1 door (compact)",
            "name": "Oven0001"
        },
        {
            "hash": "440aa75617a2db12da11307ccfa049f48f76e554",
            "num_doors": 4,
            "num_drawers": 0,
            "description": "Wardrobe0074 - 4 doors",
            "name": "Wardrobe0074"
        }
    ]

    furniture_list = []

    # Try to load from articulatable_furniture_debug.csv
    metadata_csv = project_root / "data/hssd-hab/metadata/articulatable_furniture_debug.csv"

    if metadata_csv.exists():
        try:
            with open(metadata_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    furniture_hash = row.get("furniture_hash", "").strip()
                    num_doors = row.get("num_doors", "0")

                    if furniture_hash and int(num_doors) > 0:
                        furniture_list.append({
                            "hash": furniture_hash,
                            "num_doors": int(num_doors),
                            "num_drawers": int(row.get("num_drawers", 0)),
                            "description": f"Doors: {num_doors}, Drawers: {row.get('num_drawers', 0)}"
                        })
        except Exception as e:
            print(f"Error loading articulated furniture CSV: {e}")

    # If we got data from CSV, use it; otherwise use recommended list
    if furniture_list:
        # Sort by number of doors (prefer single-door furniture for simple doors)
        furniture_list.sort(key=lambda x: x["num_doors"])
        return jsonify(furniture_list[:20])  # Return top 20
    else:
        print("No CSV data found, using recommended door options")
        return jsonify(recommended_doors)



def quaternion_to_rotation_matrix(quat: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix.
    
    Args:
        quat: Quaternion as (qx, qy, qz, qw) tuple
    
    Returns:
        3x3 rotation matrix as numpy array
    """
    qx, qy, qz, qw = quat

    # Normalize quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 0:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Convert to rotation matrix
    # Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    r00 = 1 - 2 * (qy*qy + qz*qz)
    r01 = 2 * (qx*qy - qz*qw)
    r02 = 2 * (qx*qz + qy*qw)

    r10 = 2 * (qx*qy + qz*qw)
    r11 = 1 - 2 * (qx*qx + qz*qz)
    r12 = 2 * (qy*qz - qx*qw)

    r20 = 2 * (qx*qz - qy*qw)
    r21 = 2 * (qy*qz + qx*qw)
    r22 = 1 - 2 * (qx*qx + qy*qy)

    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ], dtype=np.float32)

def find_receptacle_mesh_name(furniture_handle: str) -> Optional[str]:
    """
    Find the actual receptacle mesh filename from the furniture's URDF directory.
    
    Args:
        furniture_handle: Furniture handle like "317294cbd71b7a56a3a38f6d5b912a19bf04ed81_:0000"
    
    Returns:
        Receptacle mesh name like "chest_of_drawers0002_receptacle_mesh" or None
    """
    # Extract base hash from furniture handle
    base_hash = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle

    # Look in URDF directory for this furniture
    urdf_dir = project_root / "data" / "hssd-hab" / "urdf" / base_hash

    if not urdf_dir.exists():
        print(f"⚠ URDF directory not found: {urdf_dir}")
        return None

    # Find receptacle mesh files (they end with _receptacle_mesh.glb)
    receptacle_files = list(urdf_dir.glob("*_receptacle_mesh.glb"))

    if not receptacle_files:
        print(f"⚠ No receptacle mesh files found in {urdf_dir}")
        return None

    # Use the first receptacle mesh file (most furniture has one primary receptacle)
    receptacle_file = receptacle_files[0]
    # Remove .glb extension to get the mesh name
    receptacle_mesh_name = receptacle_file.stem

    print(f"✓ Found receptacle mesh: {receptacle_mesh_name}")
    if len(receptacle_files) > 1:
        print(f"  Note: {len(receptacle_files)} receptacle meshes found, using first one")
        for rf in receptacle_files:
            print(f"    - {rf.stem}")

    return receptacle_mesh_name


def create_sim_for_episode(episode: Dict) -> Optional[habitat_sim.Simulator]:
    """Create a simulator instance for the episode to query receptacle positions."""
    if not HABITAT_AVAILABLE:
        return None

    try:
        # Create simulator config
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = episode["scene_id"]
        sim_cfg.scene_dataset_config_file = episode["scene_dataset_config"]

        # Create agent config (minimal, just for scene loading)
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # Create configuration
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        # Create simulator
        sim = habitat_sim.Simulator(cfg)

        # Load additional object configs if specified
        if "additional_obj_config_paths" in episode:
            for obj_path in episode["additional_obj_config_paths"]:
                abs_path = project_root / obj_path
                if abs_path.exists():
                    sim.get_object_template_manager().load_configs(str(abs_path))

        return sim
    except Exception as e:
        print(f"Warning: Could not create simulator: {e}")
        return None


def get_furniture_bounds_from_sim(sim: habitat_sim.Simulator, furniture_handle: str) -> Optional[Dict[str, Any]]:
    """
    Get the 3D bounding box and dimensions of furniture from the simulator.
    
    Args:
        sim: Habitat simulator instance
        furniture_handle: The furniture handle (with or without instance suffix)
    
    Returns:
        Dictionary with:
            - 'min': (x, y, z) minimum bounds
            - 'max': (x, y, z) maximum bounds
            - 'center': (x, y, z) center point
            - 'size': (width, height, depth) dimensions
            - 'top_surface_y': Y coordinate of top surface
    """
    if sim is None:
        return None

    try:
        # Get object managers
        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()

        # Remove instance suffix for matching
        handle_base = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle

        obj = None
        matched_handle = None

        # Get all handles for debugging
        rigid_handles = rom.get_object_handles()
        articulated_handles = aom.get_object_handles()

        print(f"  Looking for furniture with handle: {handle_base}")
        print(f"  Rigid objects in scene: {len(rigid_handles)}")
        # Write all handles to a file for debugging
        debug_file = project_root / "scripts/episode_editor/static/debug_handles.txt"
        with open(debug_file, 'w') as f:
            f.write(f"Looking for furniture: {handle_base}\n")
            f.write(f"Total rigid objects: {len(rigid_handles)}\n")
            f.write(f"Total articulated objects: {len(articulated_handles)}\n\n")
            f.write("All rigid handles:\n")
            for h in rigid_handles:
                f.write(f"  {h}\n")
                f.write("\nAll articulated handles:\n")
            for h in articulated_handles:
                f.write(f"  {h}\n")
        print(f"  Debug info written to {debug_file}")
        print(f"  Articulated objects in scene: {len(articulated_handles)}")

        # Try to find the object by iterating all objects (most robust)
        print(f"  Starting robust search for {handle_base}...")

        # Method 1: Iterate all objects via substring API to get actual references
        all_objects_map = rom.get_objects_by_handle_substring("")
        print(f"  Total accessible objects in ROM: {len(all_objects_map)}")

        for h, o in all_objects_map.items():
            if handle_base in h:
                print(f"  ✓ found match in ROM iteration: {h}")
                obj = o
                matched_handle = h
                break

        # Method 2: Check articulated objects if not found in rigid
        if obj is None:
            all_ao_map = aom.get_objects_by_handle_substring("")
            print(f"  Total accessible objects in AOM: {len(all_ao_map)}")
            for h, o in all_ao_map.items():
                if handle_base in h:
                    print(f"  ✓ found match in AOM iteration: {h}")
                    obj = o
                    matched_handle = h
                    break

        # Method 3: Fallback to existing logic if maps were empty (unlikely but possible)
        if obj is None:
            # ... existing logic or failure ...
            pass

        if obj is None:
            print(f"  ✗ Could not find furniture '{handle_base}' in simulator after robust search")
            return None

        print(f"  ✓ Successfully matched handle: {matched_handle}")

        # Get axis-aligned bounding box in world coordinates
        bb = obj.root_scene_node.cumulative_bb

        # Convert Magnum Vector3 to tuples using index access
        min_pt = (bb.min[0], bb.min[1], bb.min[2])
        max_pt = (bb.max[0], bb.max[1], bb.max[2])
        center_pt = bb.center()
        center = (center_pt[0], center_pt[1], center_pt[2])
        size_pt = bb.size()
        size = (size_pt[0], size_pt[1], size_pt[2])

        return {
            'min': min_pt,
            'max': max_pt,
            'center': center,
            'size': size,
            'top_surface_y': max_pt[1],
        }

    except Exception as e:
        print(f"Error getting furniture bounds from simulator: {e}")
        return None


def calculate_object_placement_on_furniture(
    furniture_bounds: Dict[str, Any],
    randomize: bool = True
) -> Tuple[float, float, float]:
    """
    Calculate where to place an object on top of furniture.
    
    Args:
        furniture_bounds: Bounding box info from get_furniture_bounds_from_sim
        randomize: If True, randomize position within furniture surface bounds
    
    Returns:
        (x, y, z) position for the object
    """
    import random

    # Get top surface Y coordinate
    top_y = furniture_bounds['top_surface_y']

    # Get furniture center for X/Z
    center_x, _, center_z = furniture_bounds['center']

    if randomize:
        # Get furniture dimensions
        width, height, depth = furniture_bounds['size']

        # Place randomly within 70% of the furniture's top surface
        # (to avoid edges where objects might fall off)
        margin_factor = 0.7
        x_range = width * margin_factor / 2
        z_range = depth * margin_factor / 2

        x = center_x + random.uniform(-x_range, x_range)
        z = center_z + random.uniform(-z_range, z_range)
    else:
        # Place at center of furniture
        x = center_x
        z = center_z

    # Add small offset above surface to prevent clipping
    y = top_y + 0.05

    return (x, y, z)


@app.route("/api/add_door", methods=["POST"])
def add_door():
    """Add an articulated object (door) to the episode as a piece of furniture."""
    data = request.json

    door_hash = data.get("door_hash")  # Hash of articulated furniture (e.g., wardrobe)
    position = data.get("position", {"x": 0, "y": 0, "z": 0})
    rotation = data.get("rotation", {"qx": 0, "qy": 0, "qz": 0, "qw": 1})
    motion_type = data.get("motion_type", "static")
    base_type = data.get("base_type", "fixed")

    if not door_hash:
        return jsonify({"error": "Missing door_hash"}), 400

    ep = episode_data["episodes"][0]
    scene_id = ep.get("scene_id", "")

    # Load scene file to add articulated object
    if not scene_id:
        return jsonify({"error": "Episode has no scene_id"}), 400

    scene_path = project_root / f"data/hssd-hab/scenes-partnr-filtered/{scene_id}.scene_instance.json"

    if not scene_path.exists():
        return jsonify({"error": f"Scene file not found: {scene_path}"}), 404

    print(f"\n{'='*60}")
    print(f"Adding articulated door to scene")
    print(f"  Door hash: {door_hash}")
    print(f"  Position: ({position['x']}, {position['y']}, {position['z']})")
    print(f"  Rotation: (qx={rotation['qx']}, qy={rotation['qy']}, qz={rotation['qz']}, qw={rotation['qw']})")
    print(f"  Scene: {scene_id}")
    print(f"{'='*60}\n")

    # Load scene data
    with open(scene_path, 'r') as f:
        scene_data = json.load(f)

    # Add to articulated_object_instances in scene file
    if 'articulated_object_instances' not in scene_data:
        scene_data['articulated_object_instances'] = []

    new_articulated_obj = {
        "template_name": door_hash,
        "translation": [position['x'], position['y'], position['z']],
        "rotation": [rotation['qx'], rotation['qy'], rotation['qz'], rotation['qw']],
        "translation_origin": "COM",
        "motion_type": motion_type,
        "base_type": base_type,
        "auto_clamp_joint_limits": False
    }

    scene_data['articulated_object_instances'].append(new_articulated_obj)

    # Save modified scene file
    with open(scene_path, 'w') as f:
        json.dump(scene_data, f, indent=2)

    print(f"✓ Added articulated object to scene file: {scene_path}")
    print(f"  Total articulated objects in scene: {len(scene_data['articulated_object_instances'])}")

    return jsonify({
        "success": True,
        "message": f"Door added to scene at ({position['x']}, {position['y']}, {position['z']})",
        "door_count": len(scene_data['articulated_object_instances'])
    })


@app.route("/api/add_object", methods=["POST"])
def add_object():
    """Add a new object to the episode."""
    data = request.json

    object_id = data.get("object_id")
    object_class = data.get("object_class")
    room = data.get("room")
    furniture = data.get("furniture")
    receptacle_handle = data.get("receptacle_handle")
    preposition = data.get("preposition", "on")
    position = data.get("position", {"x": 0, "y": 0.5, "z": 0})

    if not all([object_id, object_class, room, furniture]):
        return jsonify({"error": "Missing required fields"}), 400

    # Validate that object exists in metadata CSV
    # Extract base object ID (remove path, extension, instance suffix)
    base_object_id = object_id.split('/')[-1].replace('.object_config.json', '')
    if '_:' in base_object_id:
        base_object_id = base_object_id.split('_:')[0]

    # Check if object is in database
    object_found = any(obj['id'] == base_object_id for obj in object_database)
    if not object_found:
        print(f"WARNING: Object '{base_object_id}' not found in metadata CSV files!")
        print(f"This object may appear as 'unknown_<index>' in the scene graph.")
        print(f"To fix: Add '{base_object_id},{object_class}' to data/hssd-hab/metadata/object_categories_filtered.csv")

    ep = episode_data["episodes"][0]

    # Count explicit objects from initial_state (excludes clutter/template entries)
    # This tells us where to insert in rigid_objs and name_to_receptacle
    explicit_object_count = 0
    for state in ep["info"]["initial_state"]:
        if (
            "name" not in state
            and "template_task_number" not in state
            and state.get("object_classes")
        ):
            explicit_object_count += 1

    # 1. Add to initial_state (before any "common sense"/clutter entries)
    new_state = {
        "number": 1,
        "object_classes": [object_class],
        "allowed_regions": [room],
        "furniture_names": [furniture],
    }

    # Find position to insert (before first entry with "name" or "template_task_number")
    state_insert_idx = len(ep["info"]["initial_state"])
    for i, state in enumerate(ep["info"]["initial_state"]):
        if "name" in state or "template_task_number" in state:
            state_insert_idx = i
            break

    ep["info"]["initial_state"].insert(state_insert_idx, new_state)

    print(f"\n{'='*60}")
    print(f"Adding object: {object_class}")
    print(f"  Object ID: {object_id}")
    print(f"  Room: {room}")
    print(f"  Furniture: {furniture}")
    print(f"  Receptacle handle: {receptacle_handle}")
    print(f"  Position in arrays: {explicit_object_count} (before clutter)")
    print(f"{'='*60}\n")

    # 2. Calculate proper object position and rotation based on furniture
    # See docs/ADDING_OBJECTS_TO_EPISODE.md for transformation matrix format
    scene_id = ep.get("scene_id", "")

    # Get furniture position and rotation from scene file
    # Use receptacle_handle instead of furniture name, as furniture name may be incorrect
    furniture_info = None
    rotation_matrix = np.eye(3)  # Default to identity rotation

    if receptacle_handle and furniture != "floor" and SCENE_UTILS_AVAILABLE and scene_id:
        try:
            # Extract the parent furniture handle from receptacle_handle
            # Format: "PARENT_HANDLE_:0000" or "PARENT_HANDLE"
            furniture_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
            # Remove instance suffix if present
            furniture_handle_base = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle

            print(f"Getting furniture position/rotation from scene file using handle: {furniture_handle_base}")
            furniture_info = scene_utils.get_object_position(scene_id, furniture_handle_base)
            if furniture_info:
                print(f"✓ Found furniture in scene file:")
                print(f"  Template: {furniture_info['template_name']}")
                print(f"  Position: {furniture_info['position']}")
                print(f"  Rotation (quaternion): {furniture_info['rotation']}")

                # Convert quaternion to rotation matrix
                rotation_matrix = quaternion_to_rotation_matrix(furniture_info['rotation'])
                print(f"  Rotation matrix extracted")
            else:
                print(f"⚠ Could not find furniture with handle '{furniture_handle_base}' in scene file")
        except Exception as e:
            print(f"⚠ Warning: Could not get furniture info from scene file: {e}")

    # Calculate object position using actual furniture bounding box from simulator
    obj_position = None

    if furniture != "floor" and HABITAT_AVAILABLE:
        # Create simulator to get actual furniture dimensions
        print(f"\nCreating simulator to get furniture bounding box...")
        sim = create_sim_for_episode(ep)

        if sim:
            try:
                # Get furniture bounding box from simulator
                furniture_handle_base = receptacle_handle.split('|')[0] if receptacle_handle and '|' in receptacle_handle else (receptacle_handle if receptacle_handle else furniture)
                if '_:' in furniture_handle_base:
                    furniture_handle_base = furniture_handle_base.split('_:')[0]

                furniture_bounds = get_furniture_bounds_from_sim(sim, furniture_handle_base)

                if furniture_bounds:
                    width, height, depth = furniture_bounds['size']
                    print(f"✓ Got furniture bounding box from simulator:")
                    print(f"  Dimensions: {width:.3f}m × {height:.3f}m × {depth:.3f}m (W×H×D)")
                    print(f"  Center: ({furniture_bounds['center'][0]:.3f}, {furniture_bounds['center'][1]:.3f}, {furniture_bounds['center'][2]:.3f})")
                    print(f"  Top surface Y: {furniture_bounds['top_surface_y']:.3f}m")

                    # Calculate object placement on furniture
                    obj_position = calculate_object_placement_on_furniture(furniture_bounds, randomize=True)
                    print(f"✓ Calculated object placement: ({obj_position[0]:.3f}, {obj_position[1]:.3f}, {obj_position[2]:.3f})")
                else:
                    print(f"⚠ Could not get furniture bounds from simulator, falling back to scene file position")
            finally:
                # Always close simulator to prevent memory leaks
                sim.close()
                print(f"✓ Simulator closed\n")

    # Fallback: Use furniture position + height offset if simulator method didn't work
    if obj_position is None:
        if furniture_info and furniture != "floor":
            # Use furniture position with height offset for top surface
            base_pos = furniture_info['position']
            # Add estimated height for top surface (same logic as scene_utils)
            if abs(base_pos[1]) < 0.1:  # Ground level furniture
                surface_y = base_pos[1] + 0.45  # Estimated height offset
            else:  # Already elevated
                surface_y = base_pos[1] + 0.05  # Small offset for top surface
            obj_position = (base_pos[0], surface_y, base_pos[2])
            print(f"Using furniture position with estimated height offset: {obj_position}")
        else:
            obj_position = (position.get("x", 0.0), position.get("y", 0.5), position.get("z", 0.0))
            print(f"Using default/provided position: {obj_position}")

    # 3. Add to rigid_objs BEFORE clutter objects
    # Transformation matrix format (see docs/ADDING_OBJECTS_TO_EPISODE.md):
    #   [[r00, r01, r02, x],
    #    [r10, r11, r12, y],
    #    [r20, r21, r22, z],
    #    [0.0, 0.0, 0.0, 1.0]]
    # Use furniture's rotation matrix if available
    transform_matrix = [
        [float(rotation_matrix[0, 0]), float(rotation_matrix[0, 1]), float(rotation_matrix[0, 2]), float(obj_position[0])],
        [float(rotation_matrix[1, 0]), float(rotation_matrix[1, 1]), float(rotation_matrix[1, 2]), float(obj_position[1])],
        [float(rotation_matrix[2, 0]), float(rotation_matrix[2, 1]), float(rotation_matrix[2, 2]), float(obj_position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]

    new_rigid_obj = [
        f"{object_id}.object_config.json",
        transform_matrix,
    ]
    # Insert at position equal to number of explicit objects (before clutter)
    ep["rigid_objs"].insert(explicit_object_count, new_rigid_obj)
    print(f"✓ Added to rigid_objs at position {explicit_object_count}")

    # 4. Add to name_to_receptacle BEFORE clutter objects
    #
    # CRITICAL RECEPTACLE FORMAT DOCUMENTATION:
    # ==========================================
    # The receptacle value format is: "PARENT_HANDLE|RECEPTACLE_MESH_NAME"
    #
    # Key rules:
    # 1. PARENT_HANDLE must include the instance suffix (e.g., "_:0004")
    # 2. RECEPTACLE_MESH_NAME always uses the BASE HASH (without instance suffix)
    #
    # Example:
    #   Furniture: chest_of_drawers_51 -> Handle: 084ff2a0e018cec0a68d318cc0f37f0b7624c8b8_:0004
    #   Correct format: "084ff2a0e018cec0a68d318cc0f37f0b7624c8b8_:0004|receptacle_mesh_084ff2a0e018cec0a68d318cc0f37f0b7624c8b8.0000"
    #   WRONG format:   "084ff2a0e018cec0a68d318cc0f37f0b7624c8b8_:0004|receptacle_mesh_084ff2a0e018cec0a68d318cc0f37f0b7624c8b8_:0004.0000"
    #
    # The base hash (084ff2a0e018cec0a68d318cc0f37f0b7624c8b8) is shared across all instances
    # of the same furniture type, but each instance has a unique suffix (_:0000, _:0001, etc.)
    #
    # See docs/ADDING_OBJECTS_TO_EPISODE.md for more details
    #
    # We need to get the actual receptacle name from the simulator, not construct it
    if receptacle_handle and furniture != "floor":
        parent_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
        # Ensure parent handle has instance suffix
        parent_handle_with_suffix = parent_handle if "_:" in parent_handle else f"{parent_handle}_:0000"

        # Find the actual receptacle mesh name from URDF directory
        receptacle_mesh_name = find_receptacle_mesh_name(parent_handle)

        if receptacle_mesh_name:
            # Format: "PARENT_HANDLE_:XXXX|receptacle_mesh_name.0000"
            # Example: "317294cbd71b7a56a3a38f6d5b912a19bf04ed81_:0000|chest_of_drawers0002_receptacle_mesh.0000"
            recep_value = f"{parent_handle_with_suffix}|{receptacle_mesh_name}.0000"
            print(f"✓ Using receptacle from URDF: {recep_value}")
        else:
            # Last resort fallback - construct a generic name
            parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
            recep_value = f"{parent_handle_with_suffix}|receptacle_mesh_{parent_handle_base}.0000"
            print(f"⚠ WARNING: Could not find receptacle mesh in URDF, using fallback: {recep_value}")
            print(f"  This will likely cause errors when loading the episode!")
    else:
        recep_value = "floor"

    # Insert at position equal to number of explicit objects (before clutter)
    recep_items = list(ep["name_to_receptacle"].items())
    new_handle = f"{object_id}_:0000"
    recep_items.insert(explicit_object_count, (new_handle, recep_value))
    ep["name_to_receptacle"] = dict(recep_items)
    print(f"✓ Added to name_to_receptacle at position {explicit_object_count}")
    print(f"  {new_handle} -> {recep_value[:60]}...")

    print(f"\n✓ Object '{object_class}' added successfully!")
    print(f"  Will be named: {object_class}_{explicit_object_count}")
    print(f"  Refresh visualization to see the new object.\n")

    return jsonify({
        "success": True,
        "object_name": f"{object_class}_{explicit_object_count}",
        "index": explicit_object_count,
    })


@app.route("/api/export", methods=["POST"])
def export_episode():
    """Export the current episode as .json and .json.gz files (legacy endpoint for server-side save)."""
    data = request.get_json(silent=True) or {}
    custom_path = data.get("path", "")

    # Determine base save path
    if custom_path:
        base_path = custom_path
    else:
        # Default: same directory as dataset
        dataset_dir = os.path.dirname(dataset_path)
        base_name = f"episode_{episode_id}_edited"
        base_path = os.path.join(dataset_dir, base_name)

    # Ensure directory exists
    os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else '.', exist_ok=True)

    # Save both formats
    json_path = base_path + ".json" if not base_path.endswith('.json') else base_path
    gz_path = json_path + ".gz"

    try:
        # Save uncompressed JSON
        with open(json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        # Save compressed JSON.gz
        save_episode_gz(gz_path, episode_data)

        return jsonify({
            "success": True,
            "paths": {
                "json": json_path,
                "gz": gz_path
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export_data")
def export_data():
    """Get episode data for client-side export."""
    return jsonify({"episode_data": episode_data})


@app.route("/api/export_gz")
def export_gz():
    """Get compressed episode data for client-side export."""
    # Create in-memory gzip file
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
        gz_file.write(json.dumps(episode_data, indent=2).encode('utf-8'))

    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/gzip',
        as_attachment=True,
        download_name=f'episode_{episode_id}_edited.json.gz'
    )


@app.route("/api/refresh_viz", methods=["POST"])
def refresh_viz():
    """Regenerate visualization after adding objects."""
    global hierarchy_data, viz_image_paths

    # Save current episode to temp file with updated data
    temp_dir = project_root / "scripts/episode_editor/static/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = int(time.time())
    temp_dataset_path = temp_dir / f"episode_{episode_id}_{timestamp}.json.gz"

    # Save the current in-memory episode data
    save_episode_gz(str(temp_dataset_path), episode_data)
    print(f"Saved updated episode to: {temp_dataset_path}")

    # Also need to generate/update metadata for the new objects
    metadata_temp_dir = temp_dir / "metadata"
    os.makedirs(metadata_temp_dir, exist_ok=True)

    # Run metadata extraction first to ensure new objects are included
    print("Running metadata extraction on updated episode...")
    metadata_cmd = [
        sys.executable,
        str(project_root / "dataset_generation/benchmark_generation/metadata_extractor.py"),
        "--dataset-path", str(temp_dataset_path),
        "--save-dir", str(metadata_temp_dir),
    ]

    try:
        metadata_result = subprocess.run(
            metadata_cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=60
        )
        if metadata_result.returncode != 0:
            print(f"Metadata extraction warning: {metadata_result.stderr}")
    except Exception as e:
        print(f"Warning: Metadata extraction failed: {e}")

    # Run viz generation with the updated metadata
    print("Generating visualization with new objects...")
    success = run_viz_generation(
        str(temp_dataset_path),
        str(metadata_temp_dir),
        save_dir,
        episode_id
    )

    if not success:
        return jsonify({"error": "Failed to regenerate visualization"}), 500

    # Reload hierarchy and image paths
    hierarchy_data = load_hierarchy(save_dir, episode_id)
    viz_image_paths = scan_viz_images(save_dir, episode_id)

    print(f"Visualization refreshed successfully with {len(viz_image_paths)} images")

    return jsonify({
        "success": True,
        "image_count": len([k for k in viz_image_paths.keys() if k.startswith("step_")]),
        "timestamp": timestamp  # Send timestamp for cache busting
    })


@app.route("/api/thumbnail/<path:object_id>")
def get_thumbnail(object_id):
    """Serve object thumbnail if available."""
    thumbnail_paths = [
        project_root / f"data/objects_ovmm/train_val/google_scanned/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root / f"data/objects_ovmm/train_val/ai2thorhab/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root / f"data/objects_ovmm/train_val/amazon_berkeley/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root / f"data/objects_ovmm/train_val/hssd/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root / f"data/objects/ycb/assets/{object_id}/thumbnails/0.jpg",
    ]

    for path in thumbnail_paths:
        if path.exists():
            return send_file(str(path), mimetype="image/jpeg")

    return jsonify({"error": "Thumbnail not available"}), 404


def main():
    global episode_data, dataset_path, episode_id, metadata_dir, save_dir
    global hierarchy_data, viz_image_paths, object_database

    parser = argparse.ArgumentParser(description="Add Objects to Scene GUI")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON.gz file")
    parser.add_argument("--episode-id", type=str, required=True, help="Episode ID to load from dataset")
    parser.add_argument("--metadata-dir", type=str, default="", help="Path to metadata directory")
    parser.add_argument("--save-dir", type=str, default="", help="Directory to save visualizations")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--skip-viz", action="store_true", help="Skip initial visualization generation")

    args = parser.parse_args()

    # Set global paths
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = str(project_root / dataset_path)

    episode_id = args.episode_id

    # Set metadata directory
    if args.metadata_dir:
        metadata_dir = args.metadata_dir
    else:
        metadata_dir = str(project_root / "data/versioned_data/partnr_episodes/v0_0/metadata")

    # Set save directory for visualizations
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = str(project_root / "scripts/episode_editor/static")

    # Load episode
    print(f"Loading episode {episode_id} from {dataset_path}...")
    episode_data = extract_episode_from_dataset(dataset_path, episode_id)

    if episode_data is None:
        print(f"Error: Episode {episode_id} not found in dataset")
        sys.exit(1)

    print(f"Episode loaded successfully")

    # Generate visualization
    if not args.skip_viz:
        success = run_viz_generation(dataset_path, metadata_dir, save_dir, episode_id)
        if not success:
            print("Warning: Visualization generation failed, continuing without viz")

    # Load hierarchy and images
    hierarchy_data = load_hierarchy(save_dir, episode_id)
    viz_image_paths = scan_viz_images(save_dir, episode_id)

    print(f"Loaded hierarchy with {len(hierarchy_data)} rooms")
    print(f"Found {len(viz_image_paths)} visualization images")

    # Scan object database
    print("Scanning object database...")
    object_database = scan_object_database()
    print(f"Found {len(object_database)} objects")

    print(f"\n🚀 Starting Add Objects to Scene at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
