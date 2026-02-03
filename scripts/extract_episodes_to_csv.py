#!/usr/bin/env python3
"""
Script to extract episode information from all dataset files and write to CSV.

Extracts:
- Dataset type (train, val, test, etc.)
- Episode ID
- Instruction
- Scene ID

Usage:
    python scripts/extract_episodes_to_csv.py --output episodes.csv
"""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from typing import List, Dict


def load_dataset_file(filepath: str) -> Dict:
    """Load dataset from JSON or JSON.gz file"""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def extract_dataset_type(filepath: str) -> str:
    """Extract dataset type from filepath (e.g., 'val', 'train', 'test')"""
    filename = os.path.basename(filepath)
    # Remove .json.gz or .json extension
    if filename.endswith('.json.gz'):
        filename = filename[:-8]
    elif filename.endswith('.json'):
        filename = filename[:-5]

    # Try to extract type from filename
    # Common patterns: val.json.gz, train.json.gz, test.json.gz, val_mini.json.gz
    if '_' in filename:
        # Handle cases like val_mini -> val
        parts = filename.split('_')
        # Check if first part is a known dataset type
        if parts[0] in ['train', 'val', 'test', 'dev']:
            return parts[0]

    # Default to filename if no clear pattern
    return filename


def find_all_dataset_files(base_dir: str) -> List[str]:
    """Find all JSON/JSON.gz dataset files recursively"""
    dataset_files = []
    base_path = Path(base_dir)

    for filepath in base_path.rglob('*.json.gz'):
        if filepath.is_file():  # Only include actual files, not directories
            if '132k' not in filepath.name:
                if filepath.name == 'val.json.gz':
                    print("adding", filepath.name)
                    dataset_files.append(str(filepath))
    # for filepath in base_path.rglob('*.json'):
    #     # Skip concept graphs and other non-dataset files
    #     if (filepath.is_file()
    #         and 'conceptgraph' not in str(filepath).lower()
    #         and 'metadata' not in str(filepath).lower()):
    #         dataset_files.append(str(filepath))
    print("dataset_files", dataset_files)
    return sorted(dataset_files)


def extract_episodes_from_file(filepath: str, include_path: bool = False) -> List[Dict]:
    """Extract episode information from a dataset file"""
    try:
        data = load_dataset_file(filepath)
        episodes = data.get('episodes', [])

        dataset_type = extract_dataset_type(filepath)
        relative_path = os.path.relpath(filepath, os.path.dirname(filepath))

        episode_info = []
        for episode in episodes:
            ep_dict = {
                'dataset_type': dataset_type,
                'dataset_file': os.path.basename(filepath),
                'episode_id': episode.get('episode_id', 'N/A'),
                'scene_id': episode.get('scene_id', 'N/A'),
                'instruction': episode.get('instruction', 'N/A'),
            }
            if include_path:
                ep_dict['dataset_path'] = relative_path
            episode_info.append(ep_dict)

        return episode_info
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract episode information from all dataset files to CSV"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='data/datasets/partnr_episodes/v0_0',
        help='Base directory to search for dataset files (default: data/datasets/partnr_episodes)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='episodes.csv',
        help='Output CSV file path (default: episodes.csv)'
    )
    parser.add_argument(
        '--include-path',
        action='store_true',
        help='Include full dataset file path in CSV'
    )
    args = parser.parse_args()

    # Find all dataset files
    print(f"Searching for dataset files in: {args.base_dir}")
    dataset_files = find_all_dataset_files(args.base_dir)
    print(f"Found {len(dataset_files)} dataset files")

    # Extract episodes from all files
    all_episodes = []
    for filepath in dataset_files:
        # Skip directories that were incorrectly matched
        if os.path.isdir(filepath):
            continue
        print(f"Processing: {filepath}")
        episodes = extract_episodes_from_file(filepath, include_path=args.include_path)
        all_episodes.extend(episodes)
        print(f"  Found {len(episodes)} episodes")

    print(f"\nTotal episodes found: {len(all_episodes)}")

    # Write to CSV
    if not all_episodes:
        print("No episodes found. Exiting.")
        return

    fieldnames = ['dataset_type', 'dataset_file', 'episode_id', 'scene_id', 'instruction']
    if args.include_path:
        fieldnames.insert(2, 'dataset_path')

    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)

    print(f"\nCSV file written to: {args.output}")
    print(f"Columns: {', '.join(fieldnames)}")


if __name__ == '__main__':
    main()
