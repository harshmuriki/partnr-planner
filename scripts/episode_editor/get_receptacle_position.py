#!/usr/bin/env python3

"""
Get the global position of a receptacle from a scene.

Usage:
    python get_receptacle_position.py <episode_file.json> <furniture_name>

Example:
    python get_receptacle_position.py static/tmp/new_test_file.json couch_0
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import habitat_sim
    import magnum as mn
    HABITAT_AVAILABLE = True
except ImportError:
    print("Error: habitat_sim not available. Please install habitat-sim.")
    sys.exit(1)


def load_episode(episode_path: str):
    """Load episode from JSON file."""
    with open(episode_path, 'r') as f:
        data = json.load(f)
    return data['episodes'][0]


def create_simulator(episode: dict):
    """Create a simulator instance for the episode."""
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


def get_receptacle_info(sim: habitat_sim.Simulator, furniture_name: str, episode: dict):
    """Get position and bounding box info for a receptacle."""
    try:
        from habitat_llm.utils.sim import find_receptacles
        
        # Load metadata to map furniture_name to handle
        # First, we need to find the receptacle handle for this furniture
        print(f"\nSearching for furniture: {furniture_name}")
        print("=" * 70)
        
        # Get all receptacles
        receptacles = find_receptacles(sim, filter_receptacles=False)
        print(f"Found {len(receptacles)} total receptacles in scene")
        
        # Try to find matching receptacles
        matches = []
        for rec in receptacles:
            try:
                rec_name = rec.unique_name
                # Check if this might be our furniture
                # Furniture names in metadata are simplified (e.g., couch_0)
                # But handles are full hashes
                if furniture_name in str(rec_name).lower():
                    matches.append(rec)
                    
            except Exception as e:
                continue
        
        if not matches:
            # Try another approach: look through name_to_receptacle in episode
            # to find receptacle handles associated with this furniture
            print(f"\nSearching in episode metadata for {furniture_name}...")
            
            # Load metadata if available
            metadata_dir = project_root / "scripts/episode_editor/static/tmp/metadata"
            metadata_file = metadata_dir / f"{episode['episode_id']}.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Find in recep_to_handle
                if 'recep_to_handle' in metadata:
                    if furniture_name in metadata['recep_to_handle']:
                        handle = metadata['recep_to_handle'][furniture_name]
                        print(f"Found handle in metadata: {handle}")
                        
                        # Now find this receptacle
                        for rec in receptacles:
                            try:
                                # Match by handle or parent object
                                if handle in rec.unique_name or rec.unique_name in handle:
                                    matches.append(rec)
                                    break
                            except:
                                continue
        
        if not matches:
            print(f"\nError: Could not find receptacle '{furniture_name}'")
            print(f"\nTrying to list all furniture in the scene...")
            
            # Try to extract from articulated objects
            aom = sim.get_articulated_object_manager()
            furniture_list = []
            for handle in aom.get_object_handles():
                obj = aom.get_object_by_handle(handle)
                if obj:
                    furniture_list.append(handle)
            
            print(f"\nFound {len(furniture_list)} articulated objects:")
            for i, furn in enumerate(furniture_list[:20]):  # Show first 20
                print(f"  {i+1}. {furn}")
            
            return None
        
        # Display info for all matches
        print(f"\nFound {len(matches)} matching receptacle(s):")
        print("=" * 70)
        
        for i, rec in enumerate(matches):
            print(f"\nReceptacle {i+1}:")
            print(f"  Unique Name: {rec.unique_name}")
            
            try:
                # Get bounding box
                rec_aabb = rec.get_global_bounds(sim)
                
                if rec_aabb:
                    center = rec_aabb.center()
                    min_pt = rec_aabb.min
                    max_pt = rec_aabb.max
                    
                    print(f"\n  Bounding Box:")
                    print(f"    Min: ({min_pt[0]:.3f}, {min_pt[1]:.3f}, {min_pt[2]:.3f})")
                    print(f"    Max: ({max_pt[0]:.3f}, {max_pt[1]:.3f}, {max_pt[2]:.3f})")
                    print(f"    Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                    print(f"    Size: ({max_pt[0]-min_pt[0]:.3f}, {max_pt[1]-min_pt[1]:.3f}, {max_pt[2]-min_pt[2]:.3f})")
                    
                    # Top surface position (where objects would be placed)
                    top_y = max_pt[1]
                    print(f"\n  Top Surface Position:")
                    print(f"    X: {center[0]:.3f}")
                    print(f"    Y: {top_y:.3f} (top surface)")
                    print(f"    Z: {center[2]:.3f}")
                    
                    print(f"\n  For placing objects:")
                    print(f"    Position: [{center[0]:.5f}, {top_y + 0.05:.5f}, {center[2]:.5f}]")
                    
            except Exception as e:
                print(f"  Error getting bounds: {e}")
        
        return matches[0] if matches else None
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python get_receptacle_position.py <episode_file.json> <furniture_name>")
        print("\nExample:")
        print("  python get_receptacle_position.py static/tmp/new_test_file.json couch_0")
        sys.exit(1)
    
    episode_path = sys.argv[1]
    furniture_name = sys.argv[2]
    
    if not Path(episode_path).exists():
        print(f"Error: File not found: {episode_path}")
        sys.exit(1)
    
    print(f"Loading episode from: {episode_path}")
    episode = load_episode(episode_path)
    print(f"Scene ID: {episode['scene_id']}")
    
    print("\nCreating simulator...")
    sim = create_simulator(episode)
    print("Simulator created successfully")
    
    receptacle = get_receptacle_info(sim, furniture_name, episode)
    
    sim.close()
    print("\n" + "=" * 70)
    if receptacle:
        print("Success! Receptacle position retrieved.")
    else:
        print("Could not find receptacle. Try running metadata extraction first:")
        print(f"  python dataset_generation/benchmark_generation/metadata_extractor.py \\")
        print(f"    --dataset-path {episode_path} \\")
        print(f"    --save-dir scripts/episode_editor/static/tmp/metadata")


if __name__ == "__main__":
    main()

