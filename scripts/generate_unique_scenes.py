"""
Render all unique scenes from a PARTNR episode dataset and save:
1) An RGB PNG per scene (agent at episode start pose).
2) A text report with scene id, number of objects, and task instruction.

Usage:
    conda activate habitat
    python scripts/save_all_unique_scenes.py \
        --dataset data/datasets/partnr_episodes/v0_0/val_mini/edited_episode.json.gz \
        --output-dir outputs/scene_renders
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import habitat_sim
    import magnum as mn
except ImportError as exc:
    raise ImportError(
        "habitat_sim is required. Activate the habitat conda env first."
    ) from exc

# Global caches reused by topdown generator
metadata_cache: Dict[str, Any] = {}
map_calibration: Dict[str, float] = {}


def load_episodes(dataset_path: Path) -> Dict:
    if dataset_path.suffix == ".gz":
        with gzip.open(dataset_path, "rt") as f:
            return json.load(f)
    with open(dataset_path, "r") as f:
        return json.load(f)


def build_simulator(scene_id: str, scene_dataset_config: str) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.scene_dataset_config_file = scene_dataset_config

    physics_path = Path("data/default.physics_config.json")
    if physics_path.exists():
        backend_cfg.physics_config_file = str(physics_path)

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.resolution = [640, 480]
    rgb_sensor.position = [0.0, 1.5, 0.0]
    rgb_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]
    agent_cfg.height = 1.5

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def load_object_templates(sim: habitat_sim.Simulator, episode: Dict) -> None:
    otm = sim.get_object_template_manager()
    for obj_path in episode.get("additional_obj_config_paths", []):
        abs_path = Path(obj_path).expanduser().resolve()
        if abs_path.exists():
            otm.load_configs(str(abs_path))


def add_objects(sim: habitat_sim.Simulator, episode: Dict) -> None:
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()

    for entry in episode.get("rigid_objs", []):
        if len(entry) != 2:
            continue
        handle, tf = entry
        if not otm.get_template_by_handle(handle):
            # If template not loaded, skip gracefully
            continue

        obj = rom.add_object_by_template_handle(handle)
        if obj is None:
            continue

        mat = np.array(tf, dtype=np.float32)
        if mat.shape == (4, 4):
            obj.transformation = mn.Matrix4(mat)


def set_agent_start(sim: habitat_sim.Simulator, episode: Dict) -> None:
    ep = episode
    pos = ep.get("start_position", [0.0, 1.5, 0.0])
    rot = ep.get("start_rotation", [0.0, 0.0, 0.0, 1.0])  # xyzw

    agent = sim.get_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(pos, dtype=np.float32)
    if isinstance(rot, (list, tuple, np.ndarray)) and len(rot) == 4:
        agent_state.rotation = np.array(rot, dtype=np.float32)
    else:
        # Fallback to identity if rotation is invalid
        agent_state.rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    agent.set_state(agent_state, reset_sensors=True)


def render_scene(sim: habitat_sim.Simulator) -> Image.Image:
    obs = sim.get_sensor_observations()
    rgb = obs.get("rgb")
    if rgb is None:
        raise RuntimeError("RGB observation is None.")
    return Image.fromarray(rgb, mode="RGBA" if rgb.shape[2] == 4 else "RGB")


def generate_topdown_map(episode: Dict, save_dir: str) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Generate a top-down map of the scene using habitat-sim (adapted from episode_editor/app.py).
    Returns the path to the saved image and calibration data.
    """
    global map_calibration, metadata_cache

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
        from habitat_sim.nav import NavMeshSettings
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"Error importing rendering deps: {exc}")
        return None, {}

    if "episodes" not in episode or len(episode["episodes"]) == 0:
        return None, {}

    ep = episode["episodes"][0]
    scene_id = ep.get("scene_id")
    scene_dataset_config = ep.get(
        "scene_dataset_config",
        "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json",
    )

    if not scene_id:
        print("Warning: No scene_id found")
        return None, {}

    try:
        cfg = get_config_defaults()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = scene_dataset_config
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = False
        backend_cfg.create_renderer = True

        # Create sensor for top-down view
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        rgb_sensor_spec.resolution = [1024, 1024]
        rgb_sensor_spec.position = [0.0, 0.0, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor_spec]
        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        sim = habitat_sim.Simulator(hab_cfg)

        # Navmesh
        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = cfg.agent_radius
        navmesh_settings.agent_height = cfg.agent_height
        navmesh_settings.include_static_objects = True
        if not sim.pathfinder.is_loaded:
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # Scene bounds
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        scene_center = scene_bb.center()
        scene_size = scene_bb.size()

        if any(np.isnan(scene_center)):
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
                scene_center = [0.0, 0.0, 0.0]
                scene_size = [30.0, 10.0, 30.0]

        max_dimension = max(scene_size[0], scene_size[2])
        camera_height = scene_center[1] + max_dimension * 1.0
        if np.isnan(camera_height) or camera_height < 1:
            camera_height = 20.0

        rotation_matrix = mn.Matrix3(
            mn.Vector3(1.0, 0.0, 0.0),
            mn.Vector3(0.0, 0.0, -1.0),
            mn.Vector3(0.0, 1.0, 0.0),
        )

        agent = sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
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
        agent.set_state(agent_state)

        observations = sim.get_sensor_observations()
        rgb_obs = observations.get("rgb")
        if rgb_obs is None:
            sim.close()
            return None, {}

        # Calibration
        fov_y_deg = 90.0
        fov_y_rad = np.deg2rad(fov_y_deg)
        center_x = pos_x
        center_z = pos_z

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
            height_above_floor = pos_y

        world_height_visible = 2 * height_above_floor * np.tan(fov_y_rad / 2)
        aspect_ratio = rgb_obs.shape[1] / rgb_obs.shape[0]
        world_width_visible = world_height_visible * aspect_ratio
        meters_per_pixel = world_height_visible / rgb_obs.shape[0]
        map_origin_x = center_x - world_width_visible / 2
        map_origin_z = center_z - world_height_visible / 2

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

        # Rooms
        room_bounds: Dict[str, Dict[str, float]] = {}
        rooms_info = metadata_cache.get("rooms", [])
        used_rooms: set[str] = set()
        try:
            semantic_scene = sim.semantic_scene
            if len(semantic_scene.regions) > 0:
                for region in semantic_scene.regions:
                    region_name = (
                        region.category.name().split("/")[0].replace(" ", "_").lower()
                    )
                    matching_room: Optional[str] = None
                    for room_name in rooms_info:
                        if room_name in used_rooms:
                            continue
                        room_base = room_name.split("_")[0].lower()
                        if (
                            region_name == room_base
                            or region_name in room_base
                            or room_base in region_name
                        ):
                            matching_room = room_name
                            used_rooms.add(room_name)
                            break
                    if not matching_room:
                        region_id = region.id if hasattr(region, "id") else len(room_bounds)
                        matching_room = f"{region_name}_{region_id}"
                    aabb = region.aabb
                    center = aabb.center()
                    size = aabb.size()
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
            "room_bounds": room_bounds,
        }
        map_calibration = calibration

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

        return map_path, calibration

    except Exception as exc:
        print(f"Error generating top-down map: {exc}")
        import traceback

        traceback.print_exc()
        return None, {}


def process_episode(
    episode: Dict, output_dir: Path, seen_scenes: set, report_lines: List[str]
) -> None:
    scene_id = episode.get("scene_id")
    if scene_id in seen_scenes:
        return
    seen_scenes.add(scene_id)

    scene_dataset_config = episode.get(
        "scene_dataset_config",
        "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json",
    )

    sim = build_simulator(scene_id, scene_dataset_config)
    try:
        load_object_templates(sim, episode)
        add_objects(sim, episode)
        set_agent_start(sim, episode)
        # top-down via shared helper; fallback to forward view if it fails
        map_path, _ = generate_topdown_map({"episodes": [episode]}, str(output_dir))
        if map_path:
            target = output_dir / f"{scene_id}.png"
            if map_path != str(target):
                Path(map_path).rename(target)
    finally:
        sim.close()

    num_objects = len(episode.get("rigid_objs", []))
    instruction = episode.get("instruction", "").replace("\n", " ").strip()
    report_lines.append(f"{scene_id}: {num_objects} objects | {instruction}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render unique scenes and save summary."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to PARTNR episode dataset (.json or .json.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/scene_renders",
        help="Directory to save PNGs and summary.txt",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_episodes(dataset_path)
    episodes: List[Dict] = data.get("episodes", [])

    seen_scenes: set = set()
    report_lines: List[str] = []

    for ep in episodes:
        process_episode(ep, output_dir, seen_scenes, report_lines)

    report_path = output_dir / "summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Saved {len(seen_scenes)} scene images to {output_dir}")
    print(f"Summary written to {report_path}")


if __name__ == "__main__":
    main()
