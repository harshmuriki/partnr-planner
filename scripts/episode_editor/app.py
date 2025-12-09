"""
Episode Editor GUI - A web-based tool for adding objects to PARTNR episodes.

Usage:
    python scripts/episode_editor/app.py --episode path/to/episode.json

Then open http://localhost:5000 in your browser.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import habitat for top-down map generation
try:
    import habitat_sim
    import magnum as mn
    from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
    from habitat_sim.nav import NavMeshSettings

    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("Warning: Habitat not available. Using simplified map.")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global state
episode_data: Dict[str, Any] = {}
episode_path: str = ""
metadata_cache: Dict[str, Any] = {}
object_database: List[Dict[str, str]] = []
map_image_path: str = ""
map_calibration: Dict[str, float] = {}  # Stores coordinate transformation data


def load_episode(path: str) -> Dict[str, Any]:
    """Load episode from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_episode(path: str, data: Dict[str, Any]) -> None:
    """Save episode to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def scan_object_database() -> List[Dict[str, str]]:
    """Load objects from metadata CSV files."""
    import csv

    objects = []
    seen_ids = set()

    # Load from object_categories_filtered.csv (pickupable objects)
    filtered_csv = (
        project_root / "data/hssd-hab/metadata/object_categories_filtered.csv"
    )
    if filtered_csv.exists():
        with open(filtered_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = row.get("id", "").strip()
                category = row.get("clean_category", "other").strip()
                if obj_id and obj_id not in seen_ids:
                    seen_ids.add(obj_id)
                    objects.append(
                        {
                            "id": obj_id,
                            "name": obj_id,  # Use ID as name
                            "category": category,
                            "source": "pickupable",
                            "super_category": category,
                        }
                    )

    # Load from fpmodels-with-decomposed.csv (all models including receptacles)
    fpmodels_csv = project_root / "data/hssd-hab/metadata/fpmodels-with-decomposed.csv"
    if fpmodels_csv.exists():
        with open(fpmodels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = row.get("id", "").strip()
                name = row.get("name", obj_id).strip()
                category = row.get("main_category", "").strip()
                super_category = row.get("super_category", "").strip()
                notes = row.get("notes", "").strip()

                # Skip if already seen or if it's a part/decomposed object
                if obj_id and obj_id not in seen_ids and notes == "pickupable":
                    # Only include pickupable objects for adding to scenes
                    seen_ids.add(obj_id)
                    objects.append(
                        {
                            "id": obj_id,
                            "name": name if name else obj_id,
                            "category": category if category else "other",
                            "source": "fpmodels",
                            "super_category": super_category
                            if super_category
                            else category,
                        }
                    )

    print(f"Loaded {len(objects)} objects from metadata CSVs")
    return sorted(objects, key=lambda x: (x.get("category", ""), x["id"]))


def generate_topdown_map(
    episode: Dict, save_dir: str
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Generate a top-down map of the scene using habitat-sim.
    Returns the path to the saved image and calibration data for coordinate mapping.
    """
    global map_calibration

    if not HABITAT_AVAILABLE:
        return None, {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = episode["episodes"][0]
        scene_id = ep.get("scene_id")
        scene_dataset_config = ep.get(
            "scene_dataset_config",
            "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json",
        )

        if not scene_id:
            print("Warning: No scene_id found")
            return None, {}

        # Create simulator configuration
        cfg = get_config_defaults()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = scene_dataset_config
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = False
        backend_cfg.create_renderer = True

        # Create sensor for top-down view
        sensor_specs = []
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        rgb_sensor_spec.resolution = [1024, 1024]
        rgb_sensor_spec.position = [0.0, 0.0, 0.0]
        sensor_specs.append(rgb_sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        # Initialize simulator
        sim = habitat_sim.Simulator(hab_cfg)

        # Generate navmesh
        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = cfg.agent_radius
        navmesh_settings.agent_height = cfg.agent_height
        navmesh_settings.include_static_objects = True

        if not sim.pathfinder.is_loaded:
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # Get scene bounding box - try multiple methods
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        scene_center = scene_bb.center()
        scene_size = scene_bb.size()

        # Check for NaN values and use pathfinder bounds as fallback
        if (
            np.isnan(scene_center[0])
            or np.isnan(scene_center[1])
            or np.isnan(scene_center[2])
        ):
            print("Warning: Scene bounding box returned NaN, using pathfinder bounds")
            if sim.pathfinder.is_loaded:
                bounds = sim.pathfinder.get_bounds()
                scene_center = [
                    (bounds[0][0] + bounds[1][0]) / 2,
                    (bounds[0][1] + bounds[1][1]) / 2,
                    (bounds[0][2] + bounds[1][2]) / 2,
                ]
                scene_size = [
                    bounds[1][0] - bounds[0][0],
                    bounds[1][1] - bounds[0][1],
                    bounds[1][2] - bounds[0][2],
                ]
            else:
                # Ultimate fallback - use origin with reasonable size
                print("Warning: Pathfinder not loaded, using default bounds")
                scene_center = [0.0, 0.0, 0.0]
                scene_size = [30.0, 10.0, 30.0]

        # Position camera high above scene looking down
        max_dimension = max(scene_size[0], scene_size[2])
        camera_height = scene_center[1] + max_dimension * 1.0

        # Ensure camera_height is valid
        if np.isnan(camera_height) or camera_height < 1:
            camera_height = 20.0

        # Set up rotation matrix for top-down view (looking straight down, north up)
        rotation_matrix = mn.Matrix3(
            mn.Vector3(1.0, 0.0, 0.0),
            mn.Vector3(0.0, 0.0, -1.0),
            mn.Vector3(0.0, 1.0, 0.0),
        )

        agent = sim.get_agent(0)
        agent_state = habitat_sim.AgentState()

        # Ensure position values are valid floats
        pos_x = float(scene_center[0]) if not np.isnan(scene_center[0]) else 0.0
        pos_y = float(camera_height) if not np.isnan(camera_height) else 20.0
        pos_z = float(scene_center[2]) if not np.isnan(scene_center[2]) else 0.0

        agent_state.position = [pos_x, pos_y, pos_z]
        quat = mn.Quaternion.from_matrix(rotation_matrix)
        agent_state.rotation = [
            quat.vector.x,
            quat.vector.y,
            quat.vector.z,
            quat.scalar,
        ]

        print(f"Setting agent position: [{pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}]")
        agent.set_state(agent_state)

        # Render top-down view
        observations = sim.get_sensor_observations()
        rgb_obs = observations.get("rgb")

        if rgb_obs is None:
            sim.close()
            return None, {}

        # Calculate calibration data for coordinate mapping
        fov_y_deg = 90.0
        fov_y_rad = np.deg2rad(fov_y_deg)

        # Use the validated position values for calibration
        center_x = pos_x
        center_z = pos_z

        # Calculate floor height from bounds or use default
        try:
            if sim.pathfinder.is_loaded:
                bounds = sim.pathfinder.get_bounds()
                floor_height = bounds[0][1]
            else:
                floor_height = 0.0
        except Exception:
            floor_height = 0.0

        height_above_floor = pos_y - floor_height
        if height_above_floor <= 0:
            height_above_floor = pos_y  # Use absolute height if floor is above camera

        world_height_visible = 2 * height_above_floor * np.tan(fov_y_rad / 2)
        aspect_ratio = rgb_obs.shape[1] / rgb_obs.shape[0]
        world_width_visible = world_height_visible * aspect_ratio

        meters_per_pixel = world_height_visible / rgb_obs.shape[0]

        # Map origin (top-left corner in world coordinates)
        map_origin_x = center_x - world_width_visible / 2
        map_origin_z = center_z - world_height_visible / 2

        # Validate all values are finite
        if (
            np.isnan(map_origin_x)
            or np.isnan(map_origin_z)
            or np.isnan(meters_per_pixel)
        ):
            print("Warning: Calibration values are NaN, using defaults")
            world_width_visible = 40.0
            world_height_visible = 40.0
            meters_per_pixel = world_height_visible / rgb_obs.shape[0]
            map_origin_x = center_x - world_width_visible / 2
            map_origin_z = center_z - world_height_visible / 2

        # Extract room bounds from semantic scene
        room_bounds: Dict[str, Dict[str, float]] = {}
        rooms_info = metadata_cache.get("rooms", [])
        used_rooms: set[str] = set()  # Track which room names have been used

        try:
            semantic_scene = sim.semantic_scene
            if len(semantic_scene.regions) > 0:
                for region in semantic_scene.regions:
                    region_name = (
                        region.category.name().split("/")[0].replace(" ", "_").lower()
                    )

                    # Find matching room name from metadata (that hasn't been used yet)
                    matching_room: Optional[str] = None
                    for room_name in rooms_info:
                        if room_name in used_rooms:
                            continue  # Skip already used rooms
                        room_base = room_name.split("_")[0].lower()
                        if (
                            region_name == room_base
                            or region_name in room_base
                            or room_base in region_name
                        ):
                            matching_room = room_name
                            used_rooms.add(room_name)
                            break

                    # If no unused match found, create a unique name based on region
                    if not matching_room:
                        # Use region's own ID for uniqueness
                        region_id = (
                            region.id if hasattr(region, "id") else len(room_bounds)
                        )
                        matching_room = f"{region_name}_{region_id}"

                    aabb = region.aabb
                    center = aabb.center()
                    size = aabb.size()

                    # Store room bounds in world coordinates
                    room_bounds[matching_room] = {
                        "center_x": float(center[0]),
                        "center_z": float(center[2]),
                        "min_x": float(center[0] - size[0] / 2),
                        "max_x": float(center[0] + size[0] / 2),
                        "min_z": float(center[2] - size[2] / 2),
                        "max_z": float(center[2] + size[2] / 2),
                        "width": float(size[0]),
                        "height": float(size[2]),
                    }
                    print(
                        f"Room {matching_room}: center=({center[0]:.2f}, {center[2]:.2f}), size=({size[0]:.2f}x{size[2]:.2f})"
                    )
        except Exception as e:
            print(f"Warning: Could not extract room bounds from semantic scene: {e}")

        calibration = {
            "origin_x": float(map_origin_x),
            "origin_z": float(map_origin_z),
            "meters_per_pixel": float(meters_per_pixel),
            "width_pixels": rgb_obs.shape[1],
            "height_pixels": rgb_obs.shape[0],
            "world_width": float(world_width_visible),
            "world_height": float(world_height_visible),
            "center_x": float(center_x),
            "center_z": float(center_z),
            "room_bounds": room_bounds,  # Add actual room bounds
        }

        print(
            f"Calibration: origin=({map_origin_x:.2f}, {map_origin_z:.2f}), size=({world_width_visible:.2f}x{world_height_visible:.2f})"
        )
        print(f"Found {len(room_bounds)} room bounds")

        # Save the image
        os.makedirs(save_dir, exist_ok=True)
        map_path = os.path.join(save_dir, "topdown_map.png")

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(rgb_obs)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(map_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

        sim.close()
        del sim

        print(f"Generated top-down map: {map_path}")
        print(f"Calibration: {calibration}")

        return map_path, calibration

    except Exception as e:
        print(f"Error generating top-down map: {e}")
        import traceback

        traceback.print_exc()
        return None, {}


def load_metadata(scene_id: str) -> Dict[str, Any]:
    """Load or generate metadata for a scene."""
    # Try to load from cache
    cache_path = project_root / f"metadata_output/{scene_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Return empty metadata if not found
    return {
        "recep_to_room": {},
        "recep_to_handle": {},
        "recep_to_description": {},
    }


def extract_scene_layout(episode: Dict) -> Dict[str, Any]:
    """Extract scene layout information from episode."""
    ep = episode["episodes"][0]

    # Parse existing objects
    objects = []
    rigid_objs = ep.get("rigid_objs", [])
    name_to_recep = ep.get("name_to_receptacle", {})
    initial_state = ep.get("info", {}).get("initial_state", [])

    # Get object-to-room mapping from metadata if available
    object_to_room = metadata_cache.get("object_to_room", {})
    recep_to_room = metadata_cache.get("recep_to_room", {})

    # Build object list from initial_state
    obj_idx = 0
    for state in initial_state:
        if "name" in state or "template_task_number" in state:
            continue
        if not state.get("object_classes"):
            continue

        obj_class = state["object_classes"][0]
        obj_name = f"{obj_class}_{obj_idx}"

        # Get room from metadata (object_to_room) or fall back to receptacle mapping
        room = object_to_room.get(obj_name, "")
        if not room:
            # Try to get room from receptacle mapping
            recep = name_to_recep.get(obj_name, "")
            if recep:
                # Extract receptacle name (remove instance suffix if present)
                recep_base = recep.split("_:")[0] if "_:" in recep else recep
                room = recep_to_room.get(
                    recep_base, state.get("allowed_regions", ["unknown"])[0]
                )
            else:
                room = state.get("allowed_regions", ["unknown"])[0]

        furniture = state.get("furniture_names", ["unknown"])[0]

        # Get position from rigid_objs if available
        position = {"x": 0, "y": 0, "z": 0}
        if obj_idx < len(rigid_objs):
            matrix = rigid_objs[obj_idx][1]
            position = {"x": matrix[0][3], "y": matrix[1][3], "z": matrix[2][3]}

        objects.append(
            {
                "name": obj_name,
                "class": obj_class,
                "room": room,
                "furniture": furniture,
                "position": position,
                "index": obj_idx,
            }
        )
        obj_idx += 1

    return {
        "scene_id": ep.get("scene_id", "unknown"),
        "objects": objects,
        "instruction": ep.get("instruction", ""),
    }


def find_nearest_receptacle(x: float, z: float, metadata: Dict) -> Tuple[str, str, str]:
    """
    Find the nearest receptacle to the given coordinates.
    Returns (receptacle_name, room_name, receptacle_handle).

    This is a simplified version - in practice you'd use actual geometry.
    """
    # For now, return a default - the actual implementation would need
    # scene geometry data to determine which receptacle contains the point
    recep_to_room = metadata.get("recep_to_room", {})
    recep_to_handle = metadata.get("recep_to_handle", {})

    # Default to first available receptacle in the room
    # In a full implementation, you'd do spatial queries
    if recep_to_room:
        first_recep = list(recep_to_room.keys())[0]
        return (
            first_recep,
            recep_to_room.get(first_recep, "unknown"),
            recep_to_handle.get(first_recep, ""),
        )

    return ("unknown", "unknown", "")


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/episode")
def get_episode():
    """Get current episode data."""
    layout = extract_scene_layout(episode_data)
    return jsonify(
        {
            "episode": layout,
            "metadata": metadata_cache,
            "path": episode_path,
            "map_calibration": map_calibration,
            "has_map": bool(map_image_path and os.path.exists(map_image_path)),
        }
    )


@app.route("/api/map")
def get_map():
    """Serve the top-down map image."""
    if map_image_path and os.path.exists(map_image_path):
        return send_file(map_image_path, mimetype="image/png")
    return jsonify({"error": "Map not available"}), 404


@app.route("/api/map/calibration")
def get_map_calibration():
    """Get map calibration data for coordinate mapping."""
    return jsonify(map_calibration)


@app.route("/api/thumbnail/<path:object_id>")
def get_thumbnail(object_id):
    """Serve object thumbnail if available."""
    # Try different thumbnail locations
    thumbnail_paths = [
        project_root
        / f"data/objects_ovmm/train_val/google_scanned/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root
        / f"data/objects_ovmm/train_val/ai2thorhab/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root
        / f"data/objects_ovmm/train_val/amazon_berkeley/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root
        / f"data/objects_ovmm/train_val/hssd/assets/objects/{object_id}/thumbnails/0.jpg",
        project_root / f"data/objects/ycb/assets/{object_id}/thumbnails/0.jpg",
    ]

    for path in thumbnail_paths:
        if path.exists():
            return send_file(str(path), mimetype="image/jpeg")

    # Return 404 if no thumbnail found
    return jsonify({"error": "Thumbnail not available"}), 404


@app.route("/api/objects")
def get_objects():
    """Get available objects database."""
    query = request.args.get("q", "").lower()
    category = request.args.get("category", "")

    filtered = object_database
    if query:
        # Search in both id and name fields
        filtered = [
            o
            for o in filtered
            if query in o["id"].lower()
            or query in o.get("name", "").lower()
            or query in o.get("category", "").lower()
        ]
    if category:
        filtered = [o for o in filtered if o["category"] == category]

    return jsonify(filtered[:100])  # Limit results


@app.route("/api/categories")
def get_categories():
    """Get unique object categories."""
    categories = sorted(set(o["category"] for o in object_database))
    return jsonify(categories)


@app.route("/api/receptacles")
def get_receptacles():
    """Get available receptacles from metadata."""
    recep_to_room = metadata_cache.get("recep_to_room", {})
    recep_to_desc = metadata_cache.get("recep_to_description", {})
    recep_to_handle = metadata_cache.get("recep_to_handle", {})

    receptacles = []
    for name, room in recep_to_room.items():
        receptacles.append(
            {
                "name": name,
                "room": room,
                "description": recep_to_desc.get(name, ""),
                "handle": recep_to_handle.get(name, ""),
            }
        )

    return jsonify(sorted(receptacles, key=lambda x: (x["room"], x["name"])))


@app.route("/api/add_object", methods=["POST"])
def add_object():
    """Add a new object to the episode."""
    data = request.json

    object_id = data.get("object_id")
    object_class = data.get("object_class")
    room = data.get("room")
    furniture = data.get("furniture")
    receptacle_handle = data.get("receptacle_handle")
    position = data.get("position", {"x": 0, "y": 0.5, "z": 0})

    if not all([object_id, object_class, room, furniture]):
        return jsonify({"error": "Missing required fields"}), 400

    ep = episode_data["episodes"][0]

    # Count existing objects in initial_state to determine insert position
    insert_idx = 0
    for state in ep["info"]["initial_state"]:
        if (
            "name" not in state
            and "template_task_number" not in state
            and state.get("object_classes")
        ):
            insert_idx += 1

    # 1. Add to initial_state (before "common sense" entry)
    new_state = {
        "number": 1,
        "object_classes": [object_class],
        "allowed_regions": [room],
        "furniture_names": [furniture],
    }

    # Find position to insert (before common sense)
    state_insert_idx = 0
    for i, state in enumerate(ep["info"]["initial_state"]):
        if "name" in state:
            state_insert_idx = i
            break
        state_insert_idx = i + 1

    ep["info"]["initial_state"].insert(state_insert_idx, new_state)

    # 2. Add to rigid_objs at the correct position (must match name_to_receptacle order)
    new_rigid_obj = [
        f"{object_id}.object_config.json",
        [
            [1.0, 0.0, 0.0, position["x"]],
            [0.0, 1.0, 0.0, position["y"]],
            [0.0, 0.0, 1.0, position["z"]],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ]
    ep["rigid_objs"].insert(insert_idx, new_rigid_obj)

    # 3. Add to name_to_receptacle at correct position
    # Build the receptacle value
    if receptacle_handle:
        recep_value = f"{receptacle_handle}|receptacle_mesh_{receptacle_handle.split('_:')[0]}.0000"
    else:
        recep_value = "floor"

    # Convert name_to_receptacle to list to maintain order
    recep_items = list(ep["name_to_receptacle"].items())

    # Insert at the correct position (insert_idx)
    new_handle = f"{object_id}_:0000"
    recep_items.insert(insert_idx, (new_handle, recep_value))

    # Rebuild dict
    ep["name_to_receptacle"] = dict(recep_items)

    return jsonify(
        {
            "success": True,
            "object_name": f"{object_class}_{insert_idx}",
            "index": insert_idx,
        }
    )


@app.route("/api/remove_object", methods=["POST"])
def remove_object():
    """Remove an object from the episode."""
    data = request.json
    index = data.get("index")

    if index is None:
        return jsonify({"error": "Missing index"}), 400

    ep = episode_data["episodes"][0]

    # Find and remove from initial_state
    state_idx = 0
    for i, state in enumerate(ep["info"]["initial_state"]):
        if "name" in state or "template_task_number" in state:
            continue
        if not state.get("object_classes"):
            continue
        if state_idx == index:
            ep["info"]["initial_state"].pop(i)
            break
        state_idx += 1

    # Remove from rigid_objs
    if index < len(ep["rigid_objs"]):
        ep["rigid_objs"].pop(index)

    # Remove from name_to_receptacle
    recep_items = list(ep["name_to_receptacle"].items())
    if index < len(recep_items):
        recep_items.pop(index)
        ep["name_to_receptacle"] = dict(recep_items)

    return jsonify({"success": True})


@app.route("/api/save", methods=["POST"])
def save():
    """Save the current episode to file."""
    global episode_path

    try:
        # Get custom path from request body if provided
        data = request.get_json(silent=True) or {}
        save_path = data.get("path", episode_path)

        # Ensure path is absolute
        if not os.path.isabs(save_path):
            save_path = str(project_root / save_path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_episode(save_path, episode_data)

        # Update the current episode path if saved to new location
        episode_path = save_path

        return jsonify({"success": True, "path": save_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episode/export")
def export_episode():
    """Export the current episode data as JSON for download."""

    from flask import Response

    # Create JSON string with proper formatting
    json_str = json.dumps(episode_data, indent=2)

    return Response(
        json_str,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=episode.json"},
    )


@app.route("/api/export", methods=["POST"])
def export():
    """Export to a new file (legacy, use /api/save with path instead)."""
    data = request.json
    new_path = data.get("path")

    if not new_path:
        return jsonify({"error": "Missing path"}), 400

    try:
        save_episode(new_path, episode_data)
        return jsonify({"success": True, "path": new_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    global episode_data, episode_path, metadata_cache, object_database, map_image_path, map_calibration

    parser = argparse.ArgumentParser(description="Episode Editor GUI")
    parser.add_argument(
        "--episode", type=str, required=True, help="Path to episode JSON file"
    )
    parser.add_argument(
        "--metadata", type=str, default="", help="Path to metadata JSON file (optional)"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--no-map", action="store_true", help="Skip top-down map generation"
    )
    parser.add_argument(
        "--map-cache",
        type=str,
        default="",
        help="Path to pre-generated map image (optional)",
    )

    args = parser.parse_args()

    # Load episode
    episode_path = args.episode
    print(f"Loading episode from: {episode_path}")
    episode_data = load_episode(episode_path)

    # Load metadata
    scene_id = episode_data["episodes"][0].get("scene_id", "0")
    if args.metadata:
        with open(args.metadata) as f:
            metadata_cache = json.load(f)
    else:
        # Try to load from metadata_output
        metadata_path = project_root / f"metadata_output/{scene_id}.json"
        if not metadata_path.exists():
            metadata_path = project_root / "metadata_output/0.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_cache = json.load(f)
            print(f"Loaded metadata from: {metadata_path}")

    # Scan object database
    print("Scanning object database...")
    object_database = scan_object_database()
    print(f"Found {len(object_database)} objects")

    # Generate or load top-down map
    if args.map_cache and os.path.exists(args.map_cache):
        map_image_path = args.map_cache
        # Try to load calibration from accompanying JSON
        calib_path = args.map_cache.replace(".png", "_calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                map_calibration = json.load(f)
        print(f"Using cached map: {map_image_path}")
    elif not args.no_map and HABITAT_AVAILABLE:
        print("Generating top-down map (this may take a moment)...")
        cache_dir = str(project_root / "scripts/episode_editor/static/maps")
        map_image_path, map_calibration = generate_topdown_map(episode_data, cache_dir)
        if map_image_path:
            # Save calibration for future use
            calib_path = map_image_path.replace(".png", "_calibration.json")
            with open(calib_path, "w") as f:
                json.dump(map_calibration, f, indent=2)
    else:
        print("Skipping top-down map generation")
        map_image_path = ""
        map_calibration = {}

    print(f"\n🚀 Starting Episode Editor at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
