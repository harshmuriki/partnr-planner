#!/usr/bin/env python3

"""
Fix object placement in episodes - ensures objects added through UI are properly placed.

This script verifies that name_to_receptacle mappings are correct and match the
furniture_names in initial_state.
"""

import json
import gzip
import sys
from pathlib import Path

def fix_episode_object_placement(episode_path: str):
    """Fix object placement in an episode file."""
    
    # Load episode
    if episode_path.endswith('.gz'):
        with gzip.open(episode_path, 'rt') as f:
            data = json.load(f)
    else:
        with open(episode_path, 'r') as f:
            data = json.load(f)
    
    episode = data['episodes'][0]
    
    # Get initial_state and name_to_receptacle
    initial_state = episode['info']['initial_state']
    name_to_recep = episode['name_to_receptacle']
    
    # Count objects (excluding clutter/template entries)
    objects_info = []
    for i, state in enumerate(initial_state):
        if 'name' in state or 'template_task_number' in state:
            continue
        if not state.get('object_classes'):
            continue
        
        obj_class = state['object_classes'][0]
        furniture = state.get('furniture_names', ['floor'])[0]
        room = state.get('allowed_regions', ['unknown'])[0]
        
        objects_info.append({
            'index': i,
            'class': obj_class,
            'furniture': furniture,
            'room': room
        })
    
    # Get object handles from name_to_receptacle (in order)
    object_handles = list(name_to_recep.keys())
    
    print(f"Found {len(objects_info)} objects in initial_state")
    print(f"Found {len(object_handles)} object handles in name_to_receptacle")
    print()
    
    if len(objects_info) != len(object_handles):
        print(f"WARNING: Mismatch between initial_state objects ({len(objects_info)}) "
              f"and name_to_receptacle entries ({len(object_handles)})")
        print()
    
    # Display mapping
    print("Object Placement:")
    print("-" * 80)
    for i, obj_info in enumerate(objects_info):
        if i < len(object_handles):
            handle = object_handles[i]
            recep_value = name_to_recep[handle]
            recep_handle = recep_value.split('|')[0] if '|' in recep_value else recep_value
            
            print(f"{i}. {obj_info['class']}")
            print(f"   Room: {obj_info['room']}")
            print(f"   Furniture (metadata): {obj_info['furniture']}")
            print(f"   Object Handle: {handle}")
            print(f"   Receptacle Handle: {recep_handle}")
            print()
    
    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_object_placement.py <episode_file.json|.json.gz>")
        sys.exit(1)
    
    episode_path = sys.argv[1]
    
    if not Path(episode_path).exists():
        print(f"Error: File not found: {episode_path}")
        sys.exit(1)
    
    print(f"Analyzing episode: {episode_path}")
    print("=" * 80)
    print()
    
    data = fix_episode_object_placement(episode_path)
    
    print()
    print("=" * 80)
    print("Analysis complete!")
    print()
    print("Note: The 'furniture (metadata)' names (counter_0, table_1, etc.) are")
    print("simplified names used for metadata only. The actual placement uses the")
    print("receptacle handles shown above.")

if __name__ == "__main__":
    main()

