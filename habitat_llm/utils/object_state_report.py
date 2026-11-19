"""Episode boundary reports from simulator truth, never planner observations."""

import copy
import json
from pathlib import Path


def snapshot_objects(env_interface):
    """Capture all movable objects and furniture, including unobserved entities."""
    graph = env_interface.perception.get_recent_graph()
    objects = graph.get_all_objects()
    furniture = graph.get_all_furnitures()
    object_names = {node.name for node in objects}
    agents = {node.name for node in graph.get_agents()}

    def room_for(node):
        # Objects usually link to a receptacle, container or holder, not a room.
        pending, seen = [node], set()
        while pending:
            current = pending.pop(0)
            if current.name in seen:
                continue
            seen.add(current.name)
            try:
                room = graph.get_room_for_entity(current)
                return room.name
            except ValueError:
                pending.extend(
                    neighbor for neighbor, relation in graph.graph[current].items()
                    if relation in {"on", "inside", "in", "joint"}
                )
        return None

    result = {}
    for node in sorted(objects + furniture, key=lambda node: node.name):
        is_object = node.name in object_names
        support = graph.find_furniture_for_object(node) if is_object else None
        relations = graph.graph[node]
        result[node.name] = {
            "kind": "object" if is_object else "furniture",
            "sim_handle": node.sim_handle,
            "location": {
                "room": room_for(node),
                "furniture": support.name if support else None,
                "held_by": next((n.name for n in relations if n.name in agents), None),
                "position_xyz": [float(v) for v in node.properties["translation"]]
                if node.properties.get("translation") is not None else None,
                "relations": {n.name: relation for n, relation in relations.items()
                              if relation in {"on", "inside", "in"}},
            },
            "states": copy.deepcopy(node.properties.get("states", {})),
        }
    return result


def write_object_state_report(output_dir, episode_name, instruction, initial, final, status="completed"):
    """Pair boundary snapshots; null means unavailable, never an assumed state."""
    path = Path(output_dir) / "object_states" / (Path(episode_name).name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "episode_name": episode_name,
        "instruction": instruction,
        "source": "simulator_ground_truth",
        "status": status,
        "complete": status == "completed",
        "final_snapshot_available": bool(final),
        "state_note": "Only applicable recorded states are included; missing states and null locations are unknown. Positions are simulator world coordinates in meters.",
        "objects": {
            name: {"initial": initial.get(name), "final": final.get(name)}
            for name in sorted(initial.keys() | final.keys())
        },
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    return path
