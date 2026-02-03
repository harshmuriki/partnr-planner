#!/usr/bin/env python3
"""
Simple script to inspect and display episode data from PARTNR dataset
"""
import json
import gzip
import sys
from typing import Dict, Any

def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load dataset from JSON or JSON.gz file"""
    if dataset_path.endswith('.gz'):
        with gzip.open(dataset_path, 'rt') as f:
            return json.load(f)
    else:
        with open(dataset_path, 'r') as f:
            return json.load(f)

def print_episode_info(episode: Dict[str, Any], episode_idx: int = None):
    """Print formatted episode information"""
    print("=" * 80)
    if episode_idx is not None:
        print(f"Episode ID: {episode.get('episode_id', episode_idx)}")
    else:
        print(f"Episode ID: {episode.get('episode_id', 'unknown')}")
    print("=" * 80)

    print(f"\n📋 Task Instruction:")
    print(f"   {episode.get('instruction', 'N/A')}")

    print(f"\n🏠 Scene Information:")
    print(f"   Scene ID: {episode.get('scene_id', 'N/A')}")
    print(f"   Scene Config: {episode.get('scene_dataset_config', 'N/A')}")

    print(f"\n🎯 Evaluation Propositions ({len(episode.get('evaluation_propositions', []))}):")
    for i, prop in enumerate(episode.get('evaluation_propositions', [])):
        func_name = prop.get('function_name', 'unknown')
        args = prop.get('args', {})
        obj_handles = args.get('object_handles', [])
        rec_handles = args.get('receptacle_handles', [])
        print(f"   {i+1}. {func_name}")
        print(f"      Objects: {obj_handles}")
        print(f"      Receptacles: {rec_handles}")

    print(f"\n📦 Objects ({len(episode.get('rigid_objs', []))}):")
    for i, obj in enumerate(episode.get('rigid_objs', [])):
        obj_name = obj[0] if isinstance(obj, list) and len(obj) > 0 else 'unknown'
        # Extract just the base name
        obj_display = obj_name.replace('.object_config.json', '')
        if len(obj_display) > 50:
            obj_display = obj_display[:47] + '...'
        print(f"   {i+1}. {obj_display}")

    print(f"\n🎯 Target Receptacles:")
    name_to_rec = episode.get('name_to_receptacle', {})
    if name_to_rec:
        for obj_name, rec_name in list(name_to_rec.items())[:5]:  # Show first 5
            obj_short = obj_name.split('_')[0][:20] if '_' in obj_name else obj_name[:20]
            rec_short = rec_name.split('|')[0][:30] if '|' in rec_name else rec_name[:30]
            print(f"   {obj_short} -> {rec_short}")
        if len(name_to_rec) > 5:
            print(f"   ... and {len(name_to_rec) - 5} more")
    else:
        print("   None specified")

    print(f"\n📊 Constraints:")
    for constraint in episode.get('evaluation_constraints', []):
        print(f"   - {constraint.get('type', 'unknown')}")

    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_episode.py <dataset_path> [episode_id]")
        print("Example: python inspect_episode.py data/datasets/partnr_episodes/v0_0/val_mini.json 0")
        sys.exit(1)

    dataset_path = sys.argv[1]
    episode_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)

    episodes = dataset.get('episodes', [])
    print(f"\nTotal episodes in dataset: {len(episodes)}\n")

    if episode_id is not None:
        # Display specific episode
        if episode_id < len(episodes):
            print_episode_info(episodes[episode_id], episode_id)
        else:
            print(f"Error: Episode ID {episode_id} not found. Valid IDs: 0-{len(episodes)-1}")
    else:
        # Display summary of all episodes
        print("Dataset Summary:")
        print(f"  Total episodes: {len(episodes)}")

        # Count unique scenes
        scenes = set(ep.get('scene_id', 'unknown') for ep in episodes)
        print(f"  Unique scenes: {len(scenes)}")

        # Show first 3 episodes
        print("\n" + "=" * 80)
        print("First 3 Episodes:")
        print("=" * 80)
        for i in range(min(3, len(episodes))):
            print_episode_info(episodes[i], i)

if __name__ == "__main__":
    main()
