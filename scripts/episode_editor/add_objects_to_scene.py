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

# Import WorldGraph for furniture information
try:
    from habitat_llm.world_model import WorldGraph, Furniture
    from habitat_llm.perception import PerceptionSim
    WORLDGRAPH_AVAILABLE = True
except ImportError:
    WORLDGRAPH_AVAILABLE = False
    print("Warning: WorldGraph not available. Will use fallback methods.")

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
furniture_data_cache: Dict[str, Any] = {}


def load_furniture_data(scene_id: str, episode_id_str: str) -> Optional[Dict[str, Any]]:
    """
    Load pre-extracted furniture data from JSON file.

    Searches for furniture data exported by extract_furniture_data.py in the
    standard location: scripts/episode_editor/static/furniture_data/

    Args:
        scene_id: Scene ID (e.g., "106366410_174226806")
        episode_id_str: Episode ID (e.g., "101")

    Returns:
        Dictionary with furniture data or None if not found.
    """
    global furniture_data_cache

    cache_key = f"{scene_id}_{episode_id_str}"
    if cache_key in furniture_data_cache:
        return furniture_data_cache[cache_key]

    # Also check scene-only key (different episodes in same scene share furniture)
    if scene_id in furniture_data_cache:
        return furniture_data_cache[scene_id]

    search_paths = [
        project_root / f"scripts/episode_editor/static/furniture_data/furniture_{scene_id}_{episode_id_str}.json",
        project_root / f"scripts/episode_editor/static/furniture_data/furniture_{scene_id}.json",
    ]

    for path in search_paths:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            furniture_data_cache[cache_key] = data
            print(f"Loaded pre-extracted furniture data from: {path}")
            print(f"  Contains {len(data.get('furniture', {}))} furniture items")
            return data

    return None


def find_furniture_in_data(
    furniture_data: Dict[str, Any],
    furniture_handle: str,
    furniture_name: str = ""
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Find a specific furniture item in the pre-extracted data.

    Matches by sim_handle with instance suffix (e.g., "abc_:0002") first,
    then falls back to base handle (e.g., "abc"), then name.

    Args:
        furniture_data: The full pre-extracted data dict
        furniture_handle: The sim handle to match (e.g., "abc123_:0000")
        furniture_name: Optional furniture name to match (e.g., "table_1")

    Returns:
        Tuple of (furniture_name, furniture_entry_dict) or None
        The furniture_name is the key from the JSON (e.g., "cabinet_65")
    """
    # First pass: Try to match the FULL handle with instance suffix
    # This correctly distinguishes between multiple instances of the same model
    for furn_name, furn_data in furniture_data.get("furniture", {}).items():
        furn_handle = furn_data.get("sim_handle", "")
        if furniture_handle and furn_handle and furniture_handle == furn_handle:
            return (furn_name, furn_data)

    # Second pass: Fall back to matching base handle (without instance suffix)
    # This is needed for cases where only the base handle is provided
    handle_base = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle

    for furn_name, furn_data in furniture_data.get("furniture", {}).items():
        furn_handle = furn_data.get("sim_handle", "")
        furn_handle_base = furn_handle.split('_:')[0] if '_:' in furn_handle else furn_handle

        if handle_base and furn_handle_base and handle_base == furn_handle_base:
            return (furn_name, furn_data)

    # Third pass: Fallback to match by name
    if furniture_name and furniture_name in furniture_data.get("furniture", {}):
        return (furniture_name, furniture_data["furniture"][furniture_name])

    return None


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


def get_furniture_articulation_info(furniture_handle: str) -> Tuple[bool, int]:
    """
    Check if furniture is articulated and count drawers/compartments.

    Args:
        furniture_handle: Furniture handle (e.g., "abc123_:0000")

    Returns:
        (is_articulated, num_drawers) tuple
        Note: num_drawers represents the number of compartments the user can target.
              For double-sided cabinets, this is divided by 2 since each instance
              represents one side.
    """
    if not furniture_handle:
        return False, 0

    base_hash = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle
    urdf_dir = project_root / "data" / "versioned_data" / "hssd-hab" / "urdf" / base_hash

    if not urdf_dir.exists():
        return False, 0

    # Count drawer receptacle meshes
    drawer_meshes = list(urdf_dir.glob("*drawer*receptacle*.glb"))
    num_drawers = len(drawer_meshes)

    # Check if this is a double-sided cabinet
    # Double-sided cabinets have drawers on both sides, but each instance represents one side
    is_double_sided = any("double_sided" in m.name for m in drawer_meshes)
    if is_double_sided and num_drawers > 0:
        num_drawers = num_drawers // 2  # Each side gets half the drawers
        print(f"  Double-sided cabinet detected: {num_drawers} drawers per side")

    # If no drawer meshes, check for cabinet/fridge (treat as 1 compartment)
    if num_drawers == 0:
        ao_config = urdf_dir / f"{base_hash}.ao_config.json"
        if ao_config.exists():
            num_drawers = 1  # Treat as single compartment

    return True, num_drawers


def get_shelf_info_from_receptacle(receptacle_handle: str) -> Tuple[bool, int, int]:
    """
    Extract shelf/drawer info from receptacle handle.

    Parses patterns like:
    - "fridge0014_shelf_01_receptacle_mesh.0000" → shelf index 0
    - "cabinet_drawer_02_receptacle_mesh.0000" → drawer index 1

    Args:
        receptacle_handle: Full receptacle handle string

    Returns:
        (is_internal, shelf_index, estimated_total_shelves)
        - is_internal: True if object should be placed inside (shelf/drawer detected)
        - shelf_index: 0-based index of the shelf/drawer
        - estimated_total_shelves: Estimated number of shelves (default 4 for fridges)
    """
    import re

    if not receptacle_handle:
        return False, -1, 0

    # Get the receptacle mesh part (after the pipe)
    receptacle_part = receptacle_handle.split('|')[-1] if '|' in receptacle_handle else receptacle_handle

    # Look for shelf or drawer patterns
    # Patterns: shelf_01, shelf_02, drawer_01, drawer_02, etc.
    shelf_match = re.search(r'shelf[_]?(\d+)', receptacle_part, re.IGNORECASE)
    drawer_match = re.search(r'drawer[_]?(\d+)', receptacle_part, re.IGNORECASE)

    if shelf_match:
        # shelf_01 → index 0, shelf_02 → index 1, etc.
        shelf_num = int(shelf_match.group(1))
        shelf_index = shelf_num - 1 if shelf_num > 0 else 0
        # Fridges typically have 4-5 shelves
        estimated_total = 4
        return True, shelf_index, estimated_total

    if drawer_match:
        drawer_num = int(drawer_match.group(1))
        drawer_index = drawer_num - 1 if drawer_num > 0 else 0
        # Cabinets typically have 2-4 drawers
        estimated_total = 3
        return True, drawer_index, estimated_total

    return False, -1, 0


@app.route("/api/receptacles/<room>")
def get_receptacles(room):
    """Get receptacles in a specific room with articulation info."""
    if room not in hierarchy_data:
        return jsonify([])

    room_data = hierarchy_data[room]
    receptacles = []
    print("*"*40)

    for recep_name, recep_data in room_data.get("receptacles", {}).items():
        handle = recep_data.get("handle", "")

        # Check if furniture is articulated and count drawers
        is_articulated, num_drawers = get_furniture_articulation_info(handle)

        receptacles.append({
            "name": recep_name,
            "description": recep_data.get("description", ""),
            "handle": handle,
            "objects_count": len(recep_data.get("objects", [])),
            "is_articulated": is_articulated,
            "num_drawers": num_drawers
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

def find_receptacle_mesh_name(furniture_handle: str) -> Tuple[Optional[str], bool]:
    """
    Find the actual receptacle mesh name for furniture.

    Args:
        furniture_handle: Furniture handle like "317294cbd71b7a56a3a38f6d5b912a19bf04ed81_:0000"

    Returns:
        Tuple of (receptacle_name, is_articulated):
        - receptacle_name: The exact name to use
        - is_articulated: Whether this is articulated furniture

    Note:
        - Articulated furniture (URDF): receptacle value needs .0000 suffix
        - Non-articulated furniture: receptacle value does NOT need .0000 suffix
    """
    # Extract base hash from furniture handle
    base_hash = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle

    # Check for sanitized/placeholder handles
    if base_hash.startswith('xxxx'):
        print(f"⚠ Detected sanitized handle: {base_hash}")
        print(f"  This furniture may not have valid receptacle data")
        return None, False

    # Strategy 1: Look in URDF directory (articulated furniture)
    urdf_dir = project_root / "data" / "versioned_data" / "hssd-hab" / "urdf" / base_hash

    if urdf_dir.exists():
        # Find receptacle mesh files (they end with _receptacle_mesh.glb)
        receptacle_files = list(urdf_dir.glob("*_receptacle_mesh.glb"))

        if receptacle_files:
            # Use the first receptacle mesh file (most furniture has one primary receptacle)
            receptacle_file = receptacle_files[0]
            receptacle_mesh_name = receptacle_file.stem

            print(f"✓ Found articulated receptacle mesh: {receptacle_mesh_name}")
            if len(receptacle_files) > 1:
                print(f"  Note: {len(receptacle_files)} receptacle meshes found, using first one")

            # Articulated furniture DOES need .0000 suffix
            return receptacle_mesh_name, True

    # Strategy 2: Look in object config (non-articulated furniture)
    # These have user_defined receptacles in their .object_config.json
    for subdir in range(10):  # Objects are in numbered subdirs
        obj_config = project_root / f"data/versioned_data/hssd-hab/objects/{subdir}/{base_hash}.object_config.json"
        if obj_config.exists():
            try:
                with open(obj_config) as f:
                    config = json.load(f)
                user_defined = config.get('user_defined', {})
                for key in user_defined:
                    if 'receptacle_mesh' in key:
                        name = user_defined[key].get('name', key) if isinstance(user_defined[key], dict) else key
                        print(f"✓ Found non-articulated receptacle: {name}")
                        # Non-articulated furniture does NOT need .0000 suffix
                        return name, False
            except Exception as e:
                print(f"⚠ Error reading {obj_config}: {e}")

    print(f"⚠ No receptacle found for: {base_hash}")
    return None, False


def get_furniture_bounds_from_glb(
    sim: habitat_sim.Simulator,
    furniture_handle_base: str,
    episode: Dict,
    furniture_translation: Optional[Tuple[float, float, float]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get furniture bounding box by loading its .glb template directly.
    
    This avoids simulator instantiation overhead and gets exact mesh dimensions
    from the source .glb files.
    
    Args:
        sim: Habitat simulator instance (for template manager access)
        furniture_handle_base: Base furniture handle without instance suffix
        episode: Episode dictionary containing scene info
        furniture_translation: Optional (x, y, z) position from scene file
    
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

    print(f"\n{'='*60}")
    print(f"Loading furniture .glb template for bbox extraction")
    print(f"  Furniture handle base: {furniture_handle_base}")
    print(f"{'='*60}")

    try:
        # Try articulated object template manager first (for furniture with joints)
        ao_template_mgr = sim.get_articulated_object_template_manager()
        obj_template_mgr = sim.get_object_template_manager()

        template_name = furniture_handle_base
        template = None
        template_type = None

        print(f"  Attempting to load template: {template_name}")

        # Method 1: Try articulated object template manager (for furniture)
        print(f"  Checking articulated object template manager...")
        try:
            template = ao_template_mgr.get_template_by_handle(template_name)
            if template is not None:
                template_type = "articulated"
                print(f"  ✓ Found in articulated object template manager")
        except Exception as e:
            print(f"    Not found in articulated template manager: {e}")

        # Method 2: Try regular object template manager (for rigid objects)
        if template is None:
            print(f"  Checking regular object template manager...")
            try:
                template = obj_template_mgr.get_template_by_handle(template_name)
                if template is not None:
                    template_type = "rigid"
                    print(f"  ✓ Found in object template manager")
            except Exception as e:
                print(f"    Not found in object template manager: {e}")

        # Method 3: Check scene file and try loading with full path
        if template is None:
            print(f"  ⚠ Template not found with base handle, checking scene file...")

            scene_id = episode.get("scene_id", "")
            if scene_id:
                scene_path = project_root / f"data/hssd-hab/scenes-partnr-filtered/{scene_id}.scene_instance.json"
                if scene_path.exists():
                    with open(scene_path, 'r') as f:
                        scene_data = json.load(f)

                    # Check articulated_object_instances first
                    for obj in scene_data.get('articulated_object_instances', []):
                        obj_template = obj.get('template_name', '')
                        if furniture_handle_base in obj_template or obj_template in furniture_handle_base:
                            template_name = obj_template
                            print(f"    Found in scene articulated_object_instances: {template_name}")
                            # Try to load it
                            template = ao_template_mgr.get_template_by_handle(template_name)
                            if template is not None:
                                template_type = "articulated"
                                break

                    # Check regular object_instances if not found
                    if template is None:
                        for obj in scene_data.get('object_instances', []):
                            obj_template = obj.get('template_name', '')
                            if furniture_handle_base in obj_template or obj_template in furniture_handle_base:
                                template_name = obj_template
                                print(f"    Found in scene object_instances: {template_name}")
                                template = obj_template_mgr.get_template_by_handle(template_name)
                                if template is not None:
                                    template_type = "rigid"
                                    break

        if template is None:
            print(f"  ✗ Could not load template for furniture '{furniture_handle_base}'")
            print(f"    Available articulated templates: {len(ao_template_mgr.get_template_handles())} total")
            print(f"    Available rigid templates: {obj_template_mgr.get_template_handles()[:5]}...")
            return None

        print(f"  ✓ Successfully loaded template: {template_name}")
        print(f"  Template type: {template_type}")

        # Get bounding box diagonal (width, height, depth)
        bb_diagonal = template.bounding_box_diagonal
        width, height, depth = float(bb_diagonal[0]), float(bb_diagonal[1]), float(bb_diagonal[2])

        print(f"  📦 Template bounding box dimensions:")
        print(f"    Width (X):  {width:.4f} m")
        print(f"    Height (Y): {height:.4f} m")
        print(f"    Depth (Z):  {depth:.4f} m")
        print(f"    Volume:     {width * height * depth:.4f} m³")

        # Check if this is articulated furniture (has URDF)
        urdf_dir = project_root / "data" / "versioned_data" / "hssd-hab" / "urdf" / furniture_handle_base
        if urdf_dir.exists():
            print(f"  🔧 Articulated furniture detected (URDF found)")
            print(f"    URDF dir: {urdf_dir}")
            # List available .glb files
            glb_files = list(urdf_dir.glob("*.glb"))
            print(f"    Available .glb meshes: {len(glb_files)}")
            for glb in glb_files[:5]:  # Show first 5
                print(f"      - {glb.name}")

        # Use furniture translation if provided, otherwise estimate from bbox
        if furniture_translation:
            center_x, center_y, center_z = furniture_translation
            print(f"  📍 Using provided furniture position: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
        else:
            # Default to origin if no position provided
            center_x, center_y, center_z = 0.0, height / 2, 0.0
            print(f"  📍 No position provided, using estimated center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")

        # Calculate bounding box in world coordinates
        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2

        min_pt = (
            center_x - half_width,
            center_y - half_height,
            center_z - half_depth
        )
        max_pt = (
            center_x + half_width,
            center_y + half_height,
            center_z + half_depth
        )
        center = (center_x, center_y, center_z)
        size = (width, height, depth)
        # Furniture models typically have origin at base (min Y ≈ 0)
        # So top surface = translation_y + full_height (not half_height)
        top_surface_y = center_y + height

        print(f"  📊 Calculated world-space bounding box:")
        print(f"    Min: ({min_pt[0]:.3f}, {min_pt[1]:.3f}, {min_pt[2]:.3f})")
        print(f"    Max: ({max_pt[0]:.3f}, {max_pt[1]:.3f}, {max_pt[2]:.3f})")
        print(f"    Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"    Top surface Y: {top_surface_y:.3f} m")
        print(f"{'='*60}\n")

        return {
            'min': min_pt,
            'max': max_pt,
            'center': center,
            'size': size,
            'top_surface_y': top_surface_y,
            'template_name': template_name,
        }

    except Exception as e:
        print(f"  ✗ Error loading furniture template: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_furniture_size_from_glb(furniture_handle_base: str) -> Optional[Tuple[float, float, float]]:
    """
    Read furniture dimensions from its GLB file (for rigid objects without URDF).

    Args:
        furniture_handle_base: Base furniture handle without instance suffix

    Returns:
        (width, height, depth) tuple or None if not found
    """
    try:
        import trimesh

        # GLB path: objects/{first_char}/{hash}.glb
        first_char = furniture_handle_base[0].lower()
        glb_path = project_root / "data" / "versioned_data" / "hssd-hab" / "objects" / first_char / f"{furniture_handle_base}.glb"

        if not glb_path.exists():
            print(f"  GLB file not found: {glb_path}")
            return None

        print(f"  Loading GLB file: {glb_path.name}")
        scene = trimesh.load(str(glb_path), force='scene')

        if isinstance(scene, trimesh.Scene):
            # Combine all meshes in the scene
            meshes = [geom for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if not meshes:
                return None
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        # Get bounding box dimensions
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        size = bounds[1] - bounds[0]  # [width, height, depth]
        print(f"  ✓ Read dimensions from GLB: {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")
        return tuple(size)
    except Exception as e:
        print(f"  Warning: Could not read GLB: {e}")
        return None


def get_furniture_size_from_urdf(furniture_handle_base: str) -> Optional[Tuple[float, float, float]]:
    """
    Read furniture dimensions from its URDF file on disk (lightweight, no simulator needed).
    Falls back to GLB file if URDF is not available.

    Args:
        furniture_handle_base: Base furniture handle without instance suffix

    Returns:
        (width, height, depth) tuple or None if not found
    """
    try:
        import xml.etree.ElementTree as ET

        urdf_dir = project_root / "data" / "versioned_data" / "hssd-hab" / "urdf" / furniture_handle_base
        urdf_file = urdf_dir / f"{furniture_handle_base}.urdf"

        if not urdf_file.exists():
            # Fallback: try GLB file (for rigid objects without URDF)
            print(f"  URDF not found, trying GLB fallback...")
            return get_furniture_size_from_glb(furniture_handle_base)

        tree = ET.parse(urdf_file)
        root = tree.getroot()

        # Find the base link's collision box
        for link in root.findall('link'):
            collision = link.find('collision')
            if collision is not None:
                geometry = collision.find('geometry')
                if geometry is not None:
                    box = geometry.find('box')
                    if box is not None:
                        size_str = box.get('size')
                        if size_str:
                            # Parse "width height depth" string
                            dims = [float(x) for x in size_str.split()]
                            if len(dims) == 3:
                                return tuple(dims)

        # URDF exists but no collision box found, try GLB
        print(f"  URDF found but no collision box, trying GLB fallback...")
        return get_furniture_size_from_glb(furniture_handle_base)
    except Exception as e:
        print(f"  Warning: Could not parse URDF: {e}")
        # Fallback to GLB
        return get_furniture_size_from_glb(furniture_handle_base)


def get_furniture_bounds_from_scene_and_urdf(
    furniture_handle_base: str,
    furniture_position: Tuple[float, float, float],
    episode: Dict
) -> Optional[Dict[str, Any]]:
    """
    Get furniture bounding box by combining scene file position with URDF dimensions.
    This is lightweight and works without Bullet physics or WorldGraph initialization.
    
    Args:
        furniture_handle_base: Base furniture handle without instance suffix
        furniture_position: (x, y, z) position from scene file
        episode: Episode dictionary containing scene info
    
    Returns:
        Dictionary with:
            - 'min': (x, y, z) minimum bounds
            - 'max': (x, y, z) maximum bounds
            - 'center': (x, y, z) center point
            - 'size': (width, height, depth) dimensions
            - 'top_surface_y': Y coordinate of top surface
    """
    try:
        # Try to get dimensions from URDF file
        size = get_furniture_size_from_urdf(furniture_handle_base)

        if size is None:
            return None

        width, height, depth = size
        center_x, center_y, center_z = furniture_position

        print(f"  ✓ Got furniture dimensions from URDF: {width:.2f}m × {height:.2f}m × {depth:.2f}m")

        # Calculate world-space bounding box
        # Furniture models have origin at BASE (min Y ≈ 0 in local space)
        # So the furniture extends from translation_y to translation_y + height
        half_width = width / 2
        half_depth = depth / 2

        min_pt = (
            center_x - half_width,
            center_y,  # Base of furniture (origin at base)
            center_z - half_depth
        )
        max_pt = (
            center_x + half_width,
            center_y + height,  # Top of furniture
            center_z + half_depth
        )
        center = (center_x, center_y + height / 2, center_z)  # Actual center
        top_surface_y = center_y + height  # Top surface

        return {
            'min': min_pt,
            'max': max_pt,
            'center': center,
            'size': size,
            'top_surface_y': top_surface_y,
        }

    except Exception as e:
        print(f"  Warning: Could not get furniture bounds from URDF: {e}")
        return None


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

        # Search rigid objects first
        for h in rom.get_object_handles():
            if handle_base not in h:
                continue
            candidate = rom.get_object_by_handle(h)
            if candidate is not None:
                obj = candidate
                matched_handle = h
                break

        # Search articulated objects if not found
        if obj is None:
            for h in aom.get_object_handles():
                if handle_base not in h:
                    continue
                candidate = aom.get_object_by_handle(h)
                if candidate is not None:
                    obj = candidate
                    matched_handle = h
                    break

        if obj is None:
            # Furniture not found in simulator (common for articulated objects)
            return None

        print(f"  ✓ Found furniture in simulator: {matched_handle}")

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

    top_y = furniture_bounds['top_surface_y']
    center_x, _, center_z = furniture_bounds['center']

    if randomize:
        width, height, depth = furniture_bounds['size']
        margin_factor = 0.8
        x_range = width * margin_factor / 2
        z_range = depth * margin_factor / 2
        x = center_x + random.uniform(-x_range, x_range)
        z = center_z + random.uniform(-z_range, z_range)
    else:
        x = center_x
        z = center_z

    # Add small offset above surface to prevent clipping
    y = top_y + 0.05

    print(f"  ✓ Placement on furniture: ({x:.3f}, {y:.3f}, {z:.3f})")
    return (x, y, z)


def calculate_object_placement_inside_furniture(
    furniture_bounds: Dict[str, Any],
    drawer_index: int = -1,
    num_drawers: int = 1,
    randomize: bool = True
) -> Tuple[float, float, float]:
    """
    Calculate position INSIDE furniture (drawer/cabinet).

    Uses simple estimation: divide furniture height by number of drawers,
    place object in the center of the selected drawer.

    Args:
        furniture_bounds: Bounding box info with 'center', 'size', 'min' keys
        drawer_index: Which drawer (0 = top drawer, -1 = bottommost drawer)
        num_drawers: Total number of drawers
        randomize: If True, randomize X/Z within drawer bounds

    Returns:
        (x, y, z) position for the object
    """
    import random

    center_x = furniture_bounds['center'][0]
    center_z = furniture_bounds['center'][2]
    width, height, depth = furniture_bounds['size']
    base_y = furniture_bounds['min'][1]

    # Get num_drawers from bounds if not specified or invalid
    if num_drawers <= 0:
        num_drawers = furniture_bounds.get('num_drawers', 1) or 1

    # Default to bottommost drawer if drawer_index is -1 or not specified
    if drawer_index < 0:
        drawer_index = num_drawers - 1  # Bottommost drawer
        print(f"  Using bottommost drawer (index {drawer_index}) of {num_drawers} drawers")

    # Calculate Y position for drawer/shelf
    # Drawers are indexed top-to-bottom (0 = top drawer/shelf)
    drawer_height = height / num_drawers
    drawer_center_y = base_y + (num_drawers - drawer_index - 0.5) * drawer_height
    y = drawer_center_y

    if randomize:
        # Random X/Z within 50% of drawer dimensions
        margin = 0.5
        x_range = width * margin / 2
        z_range = depth * margin / 2
        x = center_x + random.uniform(-x_range, x_range)
        z = center_z + random.uniform(-z_range, z_range)
    else:
        x = center_x
        z = center_z

    print(f"  ✓ Placement inside furniture (shelf {drawer_index + 1}/{num_drawers}): ({x:.3f}, {y:.3f}, {z:.3f})")
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
    placement_mode = data.get("placement_mode", "on")  # 'on' or 'within'
    drawer_index = data.get("drawer_index", -1)  # Which drawer (0 = top)

    # Auto-detect placement mode from receptacle handle (shelf/drawer detection)
    if receptacle_handle:
        is_internal, detected_index, estimated_total = get_shelf_info_from_receptacle(receptacle_handle)
        if is_internal and placement_mode == "on":
            # Override to internal placement if shelf/drawer detected
            placement_mode = "within"
            drawer_index = detected_index
            print(f"  Auto-detected internal placement: shelf/drawer index {detected_index} of ~{estimated_total}")

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

    # LOOKUP CORRECT NAMES FROM FURNITURE DATA
    # Instead of using frontend names directly, look them up in pre-extracted furniture data
    scene_id = ep.get("scene_id", "")
    episode_id_str = str(ep.get("episode_id", ""))

    correct_furniture_name = furniture
    correct_room_name = room

    # Try to get correct names from pre-extracted furniture data
    if receptacle_handle and furniture != "floor":
        pre_extracted = load_furniture_data(scene_id, episode_id_str)
        if pre_extracted:
            # Extract parent handle from receptacle_handle
            parent_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
            parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle

            # Find furniture in extracted data
            furn_result = find_furniture_in_data(pre_extracted, parent_handle_base, furniture)

            if furn_result:
                # Unpack the tuple: (furniture_name, furniture_data)
                extracted_furniture_name, furn_data = furn_result

                # Use the furniture name from the JSON key
                correct_furniture_name = extracted_furniture_name

                # Get room name directly from furniture data
                extracted_room_name = furn_data.get("room_name")

                if extracted_room_name and extracted_room_name != "unknown_room":
                    correct_room_name = extracted_room_name
                else:
                    # Fallback: Try to infer room name from existing initial_state
                    for state in ep["info"]["initial_state"]:
                        if (state.get("furniture_names")
                            and extracted_furniture_name in state.get("furniture_names", [])):
                            # Found this furniture in initial_state, use its room
                            allowed_regions = state.get("allowed_regions", [])
                            if allowed_regions:
                                extracted_room_name = allowed_regions[0]
                                correct_room_name = extracted_room_name
                                break

                # If still no room found, keep using frontend room name (already set above)

                print(f"\n{'='*60}")
                print(f"✓ Using correct names from furniture data:")
                print(f"  Furniture: '{furniture}' → '{correct_furniture_name}'")
                print(f"  Room: '{room}' → '{correct_room_name}'")
                print(f"{'='*60}\n")
            else:
                print(f"⚠ Furniture with handle '{parent_handle_base}' not found in extracted data")
                print(f"  Using frontend names: furniture='{furniture}', room='{room}'")
        else:
            print(f"⚠ No pre-extracted furniture data available for scene {scene_id}")
            print(f"  Tip: Run extract_furniture_data.py first for accurate names")
            print(f"  Using frontend names: furniture='{furniture}', room='{room}'")

    # 1. Add to initial_state (before any "common sense"/clutter entries)
    new_state = {
        "number": 1,
        "object_classes": [object_class],
        "allowed_regions": [correct_room_name],
        "furniture_names": [correct_furniture_name],
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
    print(f"  Room (frontend): {room}")
    print(f"  Furniture (frontend): {furniture}")
    print(f"  Receptacle handle: {receptacle_handle}")
    print(f"  Placement mode: {placement_mode}" + (f" (drawer {drawer_index + 1})" if drawer_index >= 0 else ""))
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

    # Calculate object position
    # Priority 1: Pre-extracted furniture data from JSON (most accurate)
    # Priority 2: Simulator-based bounding box
    # Priority 3: URDF-based dimensions
    # Priority 4: Estimated height offset
    # Priority 5: Manual/default position
    obj_position = None

    if furniture != "floor":
        # Priority 1: Try pre-extracted furniture data from JSON
        pre_extracted = load_furniture_data(scene_id, str(ep.get("episode_id", "")))
        if pre_extracted:
            furniture_handle_for_lookup = receptacle_handle.split('|')[0] if receptacle_handle and '|' in receptacle_handle else (receptacle_handle if receptacle_handle else furniture)
            furn_result = find_furniture_in_data(pre_extracted, furniture_handle_for_lookup, furniture)

            if furn_result:
                _, furn_entry = furn_result
            else:
                furn_entry = None

            if furn_entry and "aabb" in furn_entry:
                aabb_data = furn_entry["aabb"]
                furniture_bounds = {
                    'min': tuple(aabb_data["min"]),
                    'max': tuple(aabb_data["max"]),
                    'center': tuple(aabb_data["center"]),
                    'size': tuple(aabb_data["size"]),
                    'top_surface_y': aabb_data["top_surface_y"],
                }

                print(f"Using pre-extracted furniture data from JSON")
                print(f"  Top surface Y: {furniture_bounds['top_surface_y']:.3f}m")
                print(f"  Size: ({furniture_bounds['size'][0]:.3f}, {furniture_bounds['size'][1]:.3f}, {furniture_bounds['size'][2]:.3f})")

                if placement_mode == "within" and drawer_index >= 0:
                    _, num_drawers = get_furniture_articulation_info(receptacle_handle)
                    furniture_bounds['num_drawers'] = num_drawers
                    obj_position = calculate_object_placement_inside_furniture(
                        furniture_bounds, drawer_index, num_drawers, randomize=True
                    )
                else:
                    obj_position = calculate_object_placement_on_furniture(furniture_bounds, randomize=False)
            elif furn_entry:
                print(f"Furniture found in JSON but no AABB data, falling back...")
            else:
                print(f"Furniture not found in pre-extracted JSON data, falling back...")

    # Priority 2: Simulator-based bounding box
    if obj_position is None and furniture != "floor" and HABITAT_AVAILABLE:
        # Create simulator to get furniture dimensions
        print(f"\n{'='*60}")
        print(f"Creating simulator to get furniture bounding box...")
        print(f"{'='*60}")
        sim = create_sim_for_episode(ep)

        if sim:
            try:
                # Get furniture bounding box from simulator (instantiated object)
                furniture_handle_base = receptacle_handle.split('|')[0] if receptacle_handle and '|' in receptacle_handle else (receptacle_handle if receptacle_handle else furniture)
                if '_:' in furniture_handle_base:
                    furniture_handle_base = furniture_handle_base.split('_:')[0]

                print(f"Looking for furniture with handle: {furniture_handle_base}")
                furniture_bounds = get_furniture_bounds_from_sim(sim, furniture_handle_base)

                if furniture_bounds:
                    width, height, depth = furniture_bounds['size']
                    print(f"✓ Got furniture bounding box from simulator:")
                    print(f"  Dimensions: {width:.3f}m × {height:.3f}m × {depth:.3f}m (W×H×D)")
                    print(f"  Center: ({furniture_bounds['center'][0]:.3f}, {furniture_bounds['center'][1]:.3f}, {furniture_bounds['center'][2]:.3f})")
                    print(f"  Top surface Y: {furniture_bounds['top_surface_y']:.3f}m")

                    # Calculate object placement based on mode
                    if placement_mode == "within" and drawer_index >= 0:
                        # Get num_drawers from articulation info
                        _, num_drawers = get_furniture_articulation_info(receptacle_handle)
                        furniture_bounds['num_drawers'] = num_drawers
                        obj_position = calculate_object_placement_inside_furniture(
                            furniture_bounds, drawer_index, num_drawers, randomize=True
                        )
                    else:
                        obj_position = calculate_object_placement_on_furniture(furniture_bounds, randomize=True)
                else:
                    print(f"⚠ Could not get furniture bounds from simulator")

                    # Fallback: Read dimensions from URDF file (lightweight, no crashes)
                    if furniture_info:
                        print(f"  Attempting to get furniture dimensions from URDF file...")
                        furniture_bounds = get_furniture_bounds_from_scene_and_urdf(
                            furniture_handle_base,
                            furniture_info['position'],
                            ep
                        )

                        if furniture_bounds:
                            # Calculate object placement based on mode
                            if placement_mode == "within" and drawer_index >= 0:
                                _, num_drawers = get_furniture_articulation_info(receptacle_handle)
                                furniture_bounds['num_drawers'] = num_drawers
                                obj_position = calculate_object_placement_inside_furniture(
                                    furniture_bounds, drawer_index, num_drawers, randomize=True
                                )
                            else:
                                obj_position = calculate_object_placement_on_furniture(furniture_bounds, randomize=True)
                        else:
                            print(f"  Could not read URDF either, will use scene file position")
            finally:
                # Always close simulator to prevent memory leaks
                sim.close()
                print(f"✓ Simulator closed\n")

    # Fallback: Use furniture position + height offset if simulator method didn't work
    if obj_position is None:
        if furniture_info and furniture != "floor":
            base_pos = furniture_info['position']

            # Check if this is internal placement (shelf/drawer)
            if placement_mode == "within":
                # Get estimated total shelves from receptacle detection
                _, _, estimated_total = get_shelf_info_from_receptacle(receptacle_handle) if receptacle_handle else (False, -1, 3)
                if estimated_total <= 0:
                    estimated_total = 3  # Default: most drawers have 3 sections

                # Default to bottommost drawer if not specified
                actual_drawer_index = drawer_index if drawer_index >= 0 else (estimated_total - 1)
                print(f"  Using bottommost drawer (index {actual_drawer_index}) of {estimated_total} drawers")

                # Estimate furniture height (typical fridge ~1.7m, cabinet ~0.9m)
                is_fridge = receptacle_handle and 'fridge' in receptacle_handle.lower()
                estimated_height = 1.7 if is_fridge else 0.9

                # Calculate shelf height: shelves are evenly distributed
                # shelf_0 = top shelf, shelf_n = bottom shelf
                shelf_height = estimated_height / estimated_total
                # Invert index: shelf_01 is typically near top, so higher Y
                inverted_index = estimated_total - actual_drawer_index - 1
                surface_y = base_pos[1] + (inverted_index + 0.5) * shelf_height

                obj_position = (base_pos[0], surface_y, base_pos[2])
                print(f"Using estimated shelf position: shelf {actual_drawer_index + 1} of {estimated_total}, Y = {surface_y:.3f}m")
            else:
                # Standard on-top placement
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

    # Adjust Y position for beds (lower them to account for frame height)
    if "bed" in furniture.lower():
        adjusted_y = obj_position[1] * 0.75
        obj_position = (obj_position[0], adjusted_y, obj_position[2])
        print(f"  Adjusted Y position for bed: {adjusted_y:.5f} (was {obj_position[1] / 0.75:.5f})")

    print(f"\n{'='*60}")
    print(f"Building 4×4 transformation matrix for object")
    print(f"{'='*60}")
    print(f"  Rotation matrix (3×3):")
    print(f"    [{rotation_matrix[0, 0]:8.5f}, {rotation_matrix[0, 1]:8.5f}, {rotation_matrix[0, 2]:8.5f}]")
    print(f"    [{rotation_matrix[1, 0]:8.5f}, {rotation_matrix[1, 1]:8.5f}, {rotation_matrix[1, 2]:8.5f}]")
    print(f"    [{rotation_matrix[2, 0]:8.5f}, {rotation_matrix[2, 1]:8.5f}, {rotation_matrix[2, 2]:8.5f}]")
    print(f"  Translation (position):")
    print(f"    X: {obj_position[0]:.5f}")
    print(f"    Y: {obj_position[1]:.5f}")
    print(f"    Z: {obj_position[2]:.5f}")

    transform_matrix = [
        [float(rotation_matrix[0, 0]), float(rotation_matrix[0, 1]), float(rotation_matrix[0, 2]), float(obj_position[0])],
        [float(rotation_matrix[1, 0]), float(rotation_matrix[1, 1]), float(rotation_matrix[1, 2]), float(obj_position[1])],
        [float(rotation_matrix[2, 0]), float(rotation_matrix[2, 1]), float(rotation_matrix[2, 2]), float(obj_position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]

    print(f"  Final transformation matrix (4×4):")
    for i, row in enumerate(transform_matrix):
        print(f"    [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}, {row[3]:8.5f}]")
    print(f"{'='*60}\n")

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
    # 2. RECEPTACLE_MESH_NAME format depends on furniture type:
    #
    #    ARTICULATED furniture (URDF, has drawers/doors):
    #      - Needs .0000 suffix on receptacle name
    #      - Example: "hash_:0004|chest_of_drawers0002_receptacle_mesh.0000"
    #
    #    NON-ARTICULATED furniture (tables, benches):
    #      - NO .0000 suffix on receptacle name!
    #      - Example: "hash_:0000|receptacle_mesh_hash"
    #
    # Why this matters:
    #   Habitat builds sim_handle_to_name from receptacle.unique_name
    #   which is: parent_object_handle + "|" + receptacle.name
    #   The receptacle name comes from the config file WITHOUT .0000 for non-articulated.
    #   If we add .0000 incorrectly, the lookup fails and objects fall to the floor!
    #
    # See docs/ADDING_OBJECTS_TO_EPISODE.md for more details
    if receptacle_handle and furniture != "floor":
        parent_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
        # Ensure parent handle has instance suffix
        parent_handle_with_suffix = parent_handle if "_:" in parent_handle else f"{parent_handle}_:0000"

        # Find the actual receptacle mesh name
        # Returns (receptacle_name, is_articulated) tuple
        receptacle_mesh_name, is_articulated = find_receptacle_mesh_name(parent_handle)

        if receptacle_mesh_name:
            if is_articulated:
                # Articulated furniture (URDF): already has .0000 suffix from find_receptacle_mesh_name
                # Example: "317294cbd71b7a56a3a38f6d5b912a19bf04ed81_:0000|chest_of_drawers0002_receptacle_mesh.0000"
                recep_value = f"{parent_handle_with_suffix}|{receptacle_mesh_name}.0000"
                print(f"✓ Using articulated receptacle: {recep_value}")
            else:
                # Non-articulated furniture: uses format receptacle_mesh_<hash>.0000
                # Example: "033774ae209de7a75a61ef44b69e42189f9af358_:0000|receptacle_mesh_033774ae209de7a75a61ef44b69e42189f9af358.0000"
                parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
                recep_value = f"{parent_handle_with_suffix}|receptacle_mesh_{parent_handle_base}.0000"
                print(f"✓ Using non-articulated receptacle: {recep_value}")
        else:
            # Last resort fallback - construct a generic name (will likely fail for sanitized handles)
            parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
            recep_value = f"{parent_handle_with_suffix}|receptacle_mesh_{parent_handle_base}"
            print(f"⚠ WARNING: Could not find receptacle mesh, using fallback: {recep_value}")
            print(f"  This may cause errors when loading the episode!")
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
    print(f"  Saved with furniture_name: '{correct_furniture_name}'")
    print(f"  Saved with room (allowed_region): '{correct_room_name}'")
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
    parser.add_argument("--furniture-data", type=str, default="", help="Path to pre-extracted furniture JSON file from extract_furniture_data.py")

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

    # Extract furniture data using extract_furniture_data.py
    print(f"Extracting furniture data for episode {episode_id}...")
    extract_furniture_cmd = [
        sys.executable,
        "-m", "habitat_llm.examples.extract_furniture_data",
        f"+skill_runner_episode_id={episode_id}",
        'hydra.run.dir=.',
        f'habitat.dataset.data_path={dataset_path}',
    ]

    try:
        result = subprocess.run(
            extract_furniture_cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=45  # 1.5 minute timeout
        )

        if result.returncode == 0:
            print("✓ Furniture data extracted successfully in folder {scripts/episode_editor/furniture_data/.json}")
        else:
            print(f"Warning: Furniture extraction failed: {result.stderr}")
            print("Continuing without pre-extracted furniture data...")
    except subprocess.TimeoutExpired:
        print("Warning: Furniture extraction timed out")
        print("Continuing without pre-extracted furniture data...")
    except Exception as e:
        print(f"Warning: Could not extract furniture data: {e}")
        print("Continuing without pre-extracted furniture data...")

    # Pre-load furniture data if provided
    if args.furniture_data and os.path.exists(args.furniture_data):
        with open(args.furniture_data, 'r') as f:
            furn_data = json.load(f)
        furn_scene_id = furn_data.get("scene_id", "")
        furn_ep_id = furn_data.get("episode_id", "")
        cache_key = f"{furn_scene_id}_{furn_ep_id}" if furn_scene_id else "manual"
        furniture_data_cache[cache_key] = furn_data
        # Also cache by scene_id alone for cross-episode lookup
        if furn_scene_id:
            furniture_data_cache[furn_scene_id] = furn_data
        print(f"Pre-loaded furniture data: {len(furn_data.get('furniture', {}))} items from {args.furniture_data}")

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
