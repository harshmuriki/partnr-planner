#!/usr/bin/env python3
"""
CLI Object Adder - A simple command-line tool for adding objects to PARTNR episodes.

Usage:
    conda activate habitat
    python scripts/episode_editor/cli_object_adder.py \
        --dataset data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
        --episode-id 334 \
        --output outputs/modified_episode.json.gz

This tool provides:
- Interactive receptacle browsing
- Physics-based object placement with validation
- Proper episode JSON modification following PARTNR conventions
"""

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

try:
    import habitat_sim
    import habitat.sims.habitat_simulator.sim_utilities as sutils
    from habitat.datasets.rearrange.samplers.receptacle import (
        Receptacle,
        find_receptacles,
    )
except ImportError as exc:
    raise ImportError(
        "habitat_sim is required. Activate the habitat conda env first."
    ) from exc


# ============================================================================
# Episode I/O Functions
# ============================================================================


def load_episode_data(dataset_path: Path) -> Dict:
    """Load episode dataset from JSON or gzipped JSON file."""
    if dataset_path.suffix == ".gz":
        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_episode_data(output_path: Path, data: Dict) -> None:
    """Save episode dataset to JSON or gzipped JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    print(f"✓ Saved to: {output_path}")


def find_episode_by_id(data: Dict, episode_id: str) -> Optional[Dict]:
    """Find episode by ID in the dataset."""
    for ep in data.get("episodes", []):
        if str(ep.get("episode_id")) == str(episode_id):
            return ep
    return None


# ============================================================================
# Simulator Setup
# ============================================================================


def create_simulator(episode: Dict) -> habitat_sim.Simulator:
    """Create a Habitat simulator instance for the episode."""
    # Backend configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = episode.get("scene_id")
    sim_cfg.scene_dataset_config_file = episode.get(
        "scene_dataset_config",
        "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json",
    )
    sim_cfg.enable_physics = True

    # Check for physics config
    physics_path = Path("data/default.physics_config.json")
    if physics_path.exists():
        sim_cfg.physics_config_file = str(physics_path)

    # Agent configuration (minimal - no sensors needed for CLI)
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Create simulator
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Load additional object templates
    otm = sim.get_object_template_manager()
    for obj_path in episode.get("additional_obj_config_paths", []):
        abs_path = Path(obj_path).expanduser().resolve()
        if abs_path.exists():
            otm.load_configs(str(abs_path))
        else:
            print(f"Warning: Object config not found: {obj_path}")

    return sim


def load_episode_objects(sim: habitat_sim.Simulator, episode: Dict) -> None:
    """Load existing objects from episode into simulator for context."""
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()

    for entry in episode.get("rigid_objs", []):
        if len(entry) != 2:
            continue
        handle, transform = entry

        # Check if template exists
        if not otm.get_library_has_handle(handle):
            continue

        # Add object
        obj = rom.add_object_by_template_handle(handle)
        if obj is None:
            continue

        # Apply transformation
        mat = np.array(transform, dtype=np.float32)
        if mat.shape == (4, 4):
            obj.transformation = mn.Matrix4(mat)


# ============================================================================
# Receptacle Discovery and Utilities
# ============================================================================


def list_all_receptacles(sim: habitat_sim.Simulator) -> List[Receptacle]:
    """Find all receptacles in the scene."""
    # Call find_receptacles without filtering (exclude_filter_strings=None means no filtering)
    return find_receptacles(sim, exclude_filter_strings=None)


def get_region_for_position(sim: habitat_sim.Simulator, position: mn.Vector3) -> str:
    """Get the region/room name for a given position."""
    try:
        semantic_scene = sim.semantic_scene
        if semantic_scene is None or len(semantic_scene.regions) == 0:
            return "unknown"

        # Check which region contains this point
        for region in semantic_scene.regions:
            try:
                if region.aabb and region.aabb.contains(position):
                    if region.category is not None:
                        region_name = region.category.name()
                        # Clean up the name
                        region_name = region_name.split("/")[0].replace(" ", "_").lower()
                        return region_name
            except Exception:
                continue

        return "unknown"
    except Exception:
        return "unknown"


def format_receptacle_info(sim: habitat_sim.Simulator, receptacle: Receptacle) -> str:
    """Format receptacle information for display including room/region."""
    parent = receptacle.parent_object_handle or "scene/floor"
    name = receptacle.unique_name

    # Get position and determine region
    try:
        transform = receptacle.get_global_transform(sim)
        position = transform.translation
        region = get_region_for_position(sim, position)
    except Exception:
        region = "unknown"

    return f"{name} (parent: {parent}, room: {region})"


def extract_furniture_name(receptacle_unique_name: str) -> str:
    """Extract furniture name from receptacle unique name."""
    # Receptacle format: "parent_handle|receptacle_name.0000" or similar
    if "|" in receptacle_unique_name:
        parent_part = receptacle_unique_name.split("|")[0]
        # Remove _:0000 suffix if present
        if "_:" in parent_part:
            parent_part = parent_part.split("_:")[0]
        return parent_part
    return "floor"


def is_articulated_furniture(sim: habitat_sim.Simulator, receptacle: Receptacle) -> bool:
    """Determine if receptacle parent is articulated furniture."""
    if receptacle.parent_object_handle is None:
        return False

    # Check if it's in the articulated object manager
    aom = sim.get_articulated_object_manager()
    for ao in aom.get_objects_by_handle_substring(receptacle.parent_object_handle):
        return True

    return False


# ============================================================================
# Object Placement with Validation
# ============================================================================


def sample_object_placement(
    sim: habitat_sim.Simulator,
    object_handle: str,
    receptacle: Receptacle,
    max_attempts: int = 20,
    use_snap_down: bool = True,
    orientation_sample: Optional[str] = "up",
) -> Optional[Tuple[np.ndarray, mn.Quaternion]]:
    """
    Attempt to place an object on a receptacle with physics validation.
    
    Args:
        sim: Habitat simulator instance
        object_handle: Object template handle to place
        receptacle: Receptacle to place object on
        max_attempts: Maximum placement attempts
        use_snap_down: Whether to use snap_down for placement
        orientation_sample: "up" for Y-axis rotation, "all" for random quat, None for no rotation
    
    Returns:
        (position, rotation) tuple if successful, None otherwise
    """
    rom = sim.get_rigid_object_manager()
    otm = sim.get_object_template_manager()

    # Verify template exists
    if not otm.get_library_has_handle(object_handle):
        print(f"✗ Object template not found: {object_handle}")
        return None

    # Get receptacle up vector
    rec_up_global = (
        receptacle.get_global_transform(sim)
        .transform_vector(receptacle.up)
        .normalized()
    )

    new_object = None

    for attempt in range(max_attempts):
        # Sample position on receptacle
        target_position = (
            receptacle.sample_uniform_global(sim, sample_region_scale=0.8)
            + 0.08 * rec_up_global  # Small upward offset
        )

        # Create object if not already created
        if new_object is None:
            new_object = rom.add_object_by_template_handle(object_handle)
            if new_object is None:
                print(f"✗ Failed to instantiate object: {object_handle}")
                return None

        # Set position
        new_object.translation = target_position

        # Set orientation if requested
        if orientation_sample == "up":
            # Random rotation around Y-axis
            rot_angle = random.uniform(0, 2 * np.pi)
            new_object.rotation = mn.Quaternion.rotation(
                mn.Rad(rot_angle), mn.Vector3.y_axis()
            )
        elif orientation_sample == "all":
            # Random quaternion
            new_object.rotation = habitat_sim.utils.common.random_quaternion()

        # Validate placement
        placement_valid = False

        if use_snap_down:
            # Use physics-based snap down
            support_object_ids = receptacle.get_support_object_ids(sim)
            snap_success = sutils.snap_down(
                sim, new_object, support_object_ids
            )
            if snap_success and not new_object.contact_test():
                placement_valid = True
        else:
            # Simple collision test
            if not new_object.contact_test():
                placement_valid = True

        if placement_valid:
            # Success! Record final pose and cleanup
            final_position = np.array(new_object.translation)
            final_rotation = new_object.rotation
            rom.remove_object_by_handle(new_object.handle)

            print(f"✓ Placement successful (attempt {attempt + 1}/{max_attempts})")
            return (final_position, final_rotation)

    # Failed - cleanup
    if new_object is not None:
        rom.remove_object_by_handle(new_object.handle)

    print(f"✗ Failed to place object after {max_attempts} attempts")
    return None


# ============================================================================
# Episode JSON Modification
# ============================================================================


def build_transformation_matrix(
    position: np.ndarray, rotation: mn.Quaternion
) -> List[List[float]]:
    """Build 4x4 transformation matrix from position and rotation."""
    rot_matrix = rotation.to_matrix()

    return [
        [float(rot_matrix[0, 0]), float(rot_matrix[0, 1]), float(rot_matrix[0, 2]), float(position[0])],
        [float(rot_matrix[1, 0]), float(rot_matrix[1, 1]), float(rot_matrix[1, 2]), float(position[1])],
        [float(rot_matrix[2, 0]), float(rot_matrix[2, 1]), float(rot_matrix[2, 2]), float(position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def find_clutter_insertion_index(episode: Dict) -> Tuple[int, int]:
    """
    Find insertion indices for non-clutter objects.
    Returns (initial_state_index, rigid_objs_index).
    
    Clutter objects have 'common_sense_object_classes' or 'template_task_number' fields.
    """
    initial_state = episode.get("info", {}).get("initial_state", [])

    # Find first clutter entry
    for i, entry in enumerate(initial_state):
        if (
            "common_sense_object_classes" in entry
            or "template_task_number" in entry
        ):
            # Insert before clutter
            return i, i

    # No clutter found - append at end
    return len(initial_state), len(episode.get("rigid_objs", []))


def format_receptacle_for_name_to_receptacle(
    sim: habitat_sim.Simulator,
    receptacle: Receptacle,
) -> str:
    """
    Format receptacle unique name for name_to_receptacle field.
    
    CRITICAL: Articulated furniture needs .0000 suffix on receptacle mesh name,
    non-articulated furniture should NOT have it.
    """
    is_articulated = is_articulated_furniture(sim, receptacle)
    unique_name = receptacle.unique_name

    # The unique_name format is typically: "parent_handle|receptacle_mesh_name"
    # For articulated: ensure receptacle part has .0000
    # For non-articulated: ensure receptacle part does NOT have .0000

    if "|" in unique_name:
        parent, recep_mesh = unique_name.split("|", 1)

        if is_articulated:
            # Add .0000 if not present
            if not recep_mesh.endswith(".0000"):
                recep_mesh += ".0000"
        else:
            # Remove .0000 if present
            if recep_mesh.endswith(".0000"):
                recep_mesh = recep_mesh[:-5]

        return f"{parent}|{recep_mesh}"

    return unique_name


def add_object_to_episode(
    sim: habitat_sim.Simulator,
    episode: Dict,
    object_handle: str,
    object_class: str,
    position: np.ndarray,
    rotation: mn.Quaternion,
    receptacle: Receptacle,
    room: str = "unknown",
) -> None:
    """
    Add a new object to the episode, modifying all required fields.
    
    Modifies:
        - episode["info"]["initial_state"]
        - episode["rigid_objs"]
        - episode["name_to_receptacle"]
    """
    # Ensure required fields exist
    if "info" not in episode:
        episode["info"] = {}
    if "initial_state" not in episode["info"]:
        episode["info"]["initial_state"] = []
    if "rigid_objs" not in episode:
        episode["rigid_objs"] = []
    if "name_to_receptacle" not in episode:
        episode["name_to_receptacle"] = {}

    # Find insertion indices (before clutter objects)
    state_idx, rigid_idx = find_clutter_insertion_index(episode)

    # Extract furniture name
    furniture_name = extract_furniture_name(receptacle.unique_name)

    # 1. Add to initial_state
    initial_state_entry = {
        "number": 1,
        "object_classes": [object_class],
        "allowed_regions": [room],
        "furniture_names": [furniture_name],
    }
    episode["info"]["initial_state"].insert(state_idx, initial_state_entry)

    # 2. Build and add transformation matrix to rigid_objs
    transform_matrix = build_transformation_matrix(position, rotation)

    # Extract just the filename from the full path
    # Original format: /path/to/Apple_26.object_config.json
    # Episode needs: Apple_26.object_config.json
    object_filename = object_handle.split("/")[-1]
    if not object_filename.endswith(".object_config.json"):
        object_filename = f"{object_filename}.object_config.json"

    episode["rigid_objs"].insert(
        rigid_idx, [object_filename, transform_matrix]
    )

    # 3. Ensure object directory is in additional_obj_config_paths
    # Extract directory path from full handle
    if "/" in object_handle:
        object_dir = "/".join(object_handle.split("/")[:-1])
        if "additional_obj_config_paths" not in episode:
            episode["additional_obj_config_paths"] = []
        if object_dir not in episode["additional_obj_config_paths"]:
            episode["additional_obj_config_paths"].append(object_dir)
            print(f"  Added object directory to episode: {object_dir}")

    # 4. Add to name_to_receptacle
    # Extract base handle (remove .object_config.json and path)
    base_handle = object_filename.replace(".object_config.json", "")
    object_key = f"{base_handle}_:0000"

    receptacle_value = format_receptacle_for_name_to_receptacle(sim, receptacle)
    episode["name_to_receptacle"][object_key] = receptacle_value

    print(f"✓ Added '{object_class}' to episode")
    print(f"  - Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    print(f"  - Receptacle: {receptacle.unique_name}")
    print(f"  - Furniture: {furniture_name}")
    print(f"  - Room: {room}")


# ============================================================================
# CLI Interface
# ============================================================================


def print_header():
    """Print CLI header."""
    print("\n" + "=" * 70)
    print("  PARTNR CLI Object Adder")
    print("=" * 70)


def print_menu():
    """Print main menu."""
    print("\n--- Main Menu ---")
    print("1. List all receptacles")
    print("2. Search receptacles by keyword")
    print("3. Add object to receptacle")
    print("4. Show current episode info")
    print("5. Show object handle examples")
    print("6. Search for objects by keyword")
    print("7. List all available objects")
    print("8. Save and exit")
    print("9. Exit without saving")


def list_receptacles_interactive(sim: habitat_sim.Simulator, receptacles: List[Receptacle]) -> None:
    """List all receptacles with pagination."""
    print(f"\nFound {len(receptacles)} receptacles in scene:")
    print("-" * 90)

    page_size = 20
    for i in range(0, len(receptacles), page_size):
        batch = receptacles[i:i + page_size]
        for j, rec in enumerate(batch):
            idx = i + j
            print(f"  [{idx:3d}] {format_receptacle_info(sim, rec)}")

        if i + page_size < len(receptacles):
            response = input(f"\nShowing {i+1}-{min(i+page_size, len(receptacles))} of {len(receptacles)}. Continue? (y/n): ")
            if response.lower() != 'y':
                break


def search_receptacles_interactive(sim: habitat_sim.Simulator, receptacles: List[Receptacle]) -> None:
    """Search receptacles by keyword."""
    keyword = input("\nEnter search keyword: ").strip().lower()

    matches = [
        (i, rec) for i, rec in enumerate(receptacles)
        if keyword in rec.unique_name.lower()
        or (rec.parent_object_handle and keyword in rec.parent_object_handle.lower())
    ]

    if not matches:
        print(f"No receptacles found matching '{keyword}'")
        return

    print(f"\nFound {len(matches)} matching receptacles:")
    print("-" * 90)
    for idx, rec in matches:
        print(f"  [{idx:3d}] {format_receptacle_info(sim, rec)}")


def list_available_objects(sim: habitat_sim.Simulator) -> List[str]:
    """List all available object templates. Returns list of handles."""
    otm = sim.get_object_template_manager()
    handles = otm.get_template_handles()

    print(f"\nFound {len(handles)} object templates in scene dataset:")
    print("-" * 80)
    print("NOTE: When adding objects, you must use the FULL HANDLE PATH shown below.")
    print("-" * 80)

    page_size = 20
    for i in range(0, len(handles), page_size):
        batch = handles[i:i + page_size]
        for j, handle in enumerate(batch):
            idx = i + j
            # Extract basename for display
            basename = handle.split("/")[-1].replace(".object_config.json", "")
            # Try to extract class from basename
            parts = basename.split("_")
            if len(parts) >= 2:
                suggested_class = parts[0] if parts[0].replace("-", "").isalpha() else "_".join(parts[:-1])
            else:
                suggested_class = basename

            print(f"  [{idx:4d}] {basename:35s}")
            print(f"        Handle: {handle}")
            print(f"        Class:  {suggested_class}")
            print()

        if i + page_size < len(handles):
            response = input(f"\nShowing {i+1}-{min(i+page_size, len(handles))} of {len(handles)}. Continue? (y/n): ")
            if response.lower() != 'y':
                break

    return handles


def search_objects(sim: habitat_sim.Simulator) -> Optional[List[str]]:
    """Search for object templates by keyword. Returns list of matching handles."""
    keyword = input("\nEnter search keyword (e.g., 'apple', 'chair', 'plate'): ").strip().lower()

    otm = sim.get_object_template_manager()
    handles = otm.get_template_handles()

    matches = [
        handle for handle in handles
        if keyword in handle.lower()
    ]

    if not matches:
        print(f"No objects found matching '{keyword}'")
        return None

    print(f"\nFound {len(matches)} matching objects:")
    print("-" * 80)
    print("TIP: Copy the full 'Handle' path when adding objects!")
    print("-" * 80)

    for i, handle in enumerate(matches):
        basename = handle.split("/")[-1].replace(".object_config.json", "")
        parts = basename.split("_")
        if len(parts) >= 2:
            suggested_class = parts[0] if parts[0].replace("-", "").isalpha() else "_".join(parts[:-1])
        else:
            suggested_class = basename

        print(f"  [{i:3d}] {basename}")
        print(f"        Full Handle: {handle}")
        print(f"        Suggested Class: {suggested_class}")
        print()

        if (i + 1) % 15 == 0 and i + 1 < len(matches):
            response = input(f"\nShowing {i+1} of {len(matches)}. Continue? (y/n): ")
            if response.lower() != 'y':
                break

    return matches


def show_object_examples() -> None:
    """Show examples of common objects and their formats."""
    print("\n--- Object Handle and Class Examples ---")
    print("-" * 70)
    print("Common object pattern: <id>_<name>_<variant>")
    print("\nExamples:")
    print("  Handle: '102_dolly_0'           -> Class: 'dolly'")
    print("  Handle: '003_cracker_box'       -> Class: 'cracker_box'")
    print("  Handle: '024_bowl_0'            -> Class: 'bowl'")
    print("  Handle: '037_mug_1'             -> Class: 'mug'")
    print("  Handle: '056_tennis_ball_2'     -> Class: 'tennis_ball'")
    print("\nTips:")
    print("  - Use option 7 to search for objects by keyword")
    print("  - Use option 8 to browse all available objects")
    print("  - The class is usually the middle part of the handle")
    print("  - You can omit '.object_config.json' suffix when entering handle")
    print("-" * 70)


def show_episode_info(episode: Dict) -> None:
    """Display current episode information."""
    print(f"\n--- Episode Information ---")
    print(f"Episode ID: {episode.get('episode_id')}")
    print(f"Scene ID: {episode.get('scene_id')}")
    print(f"Instruction: {episode.get('instruction', 'N/A')[:80]}...")
    print(f"Number of objects: {len(episode.get('rigid_objs', []))}")
    print(f"Initial state entries: {len(episode.get('info', {}).get('initial_state', []))}")


def add_object_interactive(
    sim: habitat_sim.Simulator,
    episode: Dict,
    receptacles: List[Receptacle],
    last_search_results: Optional[List[str]] = None,
) -> bool:
    """Interactive object addition workflow. Returns True if object was added."""
    print("\n--- Add Object ---")
    print("TIP: Use menu option 6 to search for objects first!")
    print()

    # Get receptacle index
    try:
        recep_idx = int(input("Enter receptacle index (or -1 to cancel): "))
        if recep_idx == -1:
            return False
        if recep_idx < 0 or recep_idx >= len(receptacles):
            print(f"✗ Invalid index. Must be 0-{len(receptacles)-1}")
            return False
    except ValueError:
        print("✗ Invalid input. Please enter a number.")
        return False

    receptacle = receptacles[recep_idx]
    print(f"Selected: {format_receptacle_info(sim, receptacle)}")
    print()

    # Detect region from receptacle position
    try:
        transform = receptacle.get_global_transform(sim)
        position = transform.translation
        detected_region = get_region_for_position(sim, position)
    except Exception:
        detected_region = "living_room"

    # Get object handle
    print("Enter object handle:")
    if last_search_results:
        print(f"  - Enter index [0-{len(last_search_results)-1}] from last search, OR")
    print("  - Enter full handle path (e.g., 'data/objects/apple.object_config.json'), OR")
    print("  - Type 'search' to search for objects now")

    object_input = input("> ").strip()
    if not object_input:
        print("✗ Object handle cannot be empty")
        return False

    # Check if user wants to search
    if object_input.lower() == 'search':
        results = search_objects(sim)
        if results:
            try:
                idx = int(input("\nSelect object index from search results: "))
                if 0 <= idx < len(results):
                    object_handle = results[idx]
                else:
                    print("✗ Invalid index")
                    return False
            except ValueError:
                print("✗ Invalid input")
                return False
        else:
            return False
    # Check if user entered an index from last search
    elif last_search_results and object_input.isdigit():
        idx = int(object_input)
        if 0 <= idx < len(last_search_results):
            object_handle = last_search_results[idx]
            print(f"Selected: {object_handle}")
        else:
            print(f"✗ Invalid index. Must be 0-{len(last_search_results)-1}")
            return False
    else:
        object_handle = object_input

    # Get object class
    # Auto-suggest class from handle
    basename = object_handle.split("/")[-1].replace(".object_config.json", "")
    parts = basename.split("_")
    if len(parts) >= 2:
        suggested_class = parts[0] if parts[0].replace("-", "").isalpha() else "_".join(parts[:-1])
    else:
        suggested_class = basename

    object_class = input(f"Enter object class (suggested: '{suggested_class}'): ").strip()
    if not object_class:
        object_class = suggested_class
        print(f"Using suggested class: '{object_class}'")

    # Get room (use detected region as default)
    room = input(f"Enter room name (detected: '{detected_region}'): ").strip()
    if not room:
        room = detected_region
        print(f"Using detected region: '{room}'")

    # Attempt placement
    print("\nAttempting placement...")
    result = sample_object_placement(
        sim, object_handle, receptacle, max_attempts=20
    )

    if result is None:
        print("✗ Failed to place object. Try a different receptacle or object.")
        return False

    position, rotation = result

    # Add to episode
    add_object_to_episode(
        sim, episode, object_handle, object_class,
        position, rotation, receptacle, room
    )

    return True


def run_interactive_session(
    sim: habitat_sim.Simulator,
    episode: Dict,
    output_path: Optional[Path],
) -> None:
    """Run the interactive CLI session."""
    print_header()
    print(f"Loaded episode {episode.get('episode_id')} from {episode.get('scene_id')}")

    # Discover receptacles
    print("\nDiscovering receptacles in scene...")
    receptacles = list_all_receptacles(sim)
    print(f"✓ Found {len(receptacles)} receptacles")

    # Load existing objects for context
    print("Loading existing objects...")
    load_episode_objects(sim, episode)
    print(f"✓ Loaded {len(episode.get('rigid_objs', []))} objects")

    # Main loop
    last_search_results = None  # Track last object search results

    while True:
        print_menu()
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            list_receptacles_interactive(sim, receptacles)

        elif choice == "2":
            search_receptacles_interactive(sim, receptacles)
        elif choice == "3":
            add_object_interactive(sim, episode, receptacles, last_search_results)
        elif choice == "4":
            show_episode_info(episode)

        elif choice == "5":
            show_object_examples()

        elif choice == "6":
            results = search_objects(sim)
            if results:
                last_search_results = results
                print(f"\n✓ {len(results)} results saved. You can now use option 3 and enter an index.")

        elif choice == "7":
            list_available_objects(sim)

        elif choice == "8":
            if output_path is None:
                output_path = Path(input("Enter output path: ").strip())

            # Save entire dataset with modified episode
            data = {"episodes": [episode]}
            save_episode_data(output_path, data)
            print("\n✓ Saved successfully. Exiting...")
            break

        elif choice == "9":
            print("\nExiting without saving...")
            break

        else:
            print("✗ Invalid choice. Please enter 1-9.")
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for adding objects to PARTNR episodes"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to PARTNR dataset (.json or .json.gz)",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        required=True,
        help="Episode ID to edit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for modified episode (default: ask during save)",
    )

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"✗ Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}...")
    data = load_episode_data(dataset_path)

    # Find episode
    episode = find_episode_by_id(data, args.episode_id)
    if episode is None:
        print(f"✗ Episode {args.episode_id} not found in dataset")
        sys.exit(1)

    # Create simulator
    print("Creating simulator...")
    sim = create_simulator(episode)

    try:
        # Run interactive session
        output_path = Path(args.output) if args.output else None
        run_interactive_session(sim, episode, output_path)
    finally:
        # Cleanup
        sim.close()
        print("✓ Simulator closed")


if __name__ == "__main__":
    main()
