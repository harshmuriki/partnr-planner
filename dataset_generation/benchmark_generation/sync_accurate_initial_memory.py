"""Configure GT startup memory for ACC/INC and retain OUT stale locations.

The viewer fetches specs/<task>/<variant>.md and displays the bullet lines in
the Initial robot memory section. Keep the same source and extraction here.
Only explicit ACC absence statements are copied. ACC/INC locations and states
are resolved at runtime from GT. Existing object records, flags and OUT stale
placements are preserved.
"""

import argparse
import json
import re
from pathlib import Path


def memory_items(text):
    match = re.search(
        r"^## Initial robot memory\n(.*?)(?=^## |\Z)", text, re.M | re.S
    )
    if match is None:
        raise ValueError("Missing Initial robot memory section")
    return [line[2:] for line in match[1].strip().splitlines() if line.startswith("- ")]


def sync(root):
    updates = []
    for path in sorted((root / "episodes").glob("task_*/*/initial_robot_memory.json")):
        data = json.loads(path.read_text())
        variant = data["variant"]
        data["initial_memory_source"] = (
            "remembered_stale_locations" if "-OUT-" in variant else "ground_truth_at_start"
        )
        if "-ACC-" in variant:
            source = root / "specs" / variant.split("-")[0] / (variant + ".md")
            data["initial_memory"] = [
                item for item in memory_items(source.read_text()) if ": known absent" in item
            ]
        updates.append((path, data))
    # Validate all sources before writing any files.
    for path, data in updates:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return len(updates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("baseline_evaluation_v3"))
    print(f"Updated {sync(parser.parse_args().root)} memory files")
