#!/usr/bin/env python3
"""JSON serializers for skill_runner world-graph inspectors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from habitat_llm.world_model import Furniture, Object, Receptacle


def _safe_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert property values to JSON-serializable forms."""
    out: Dict[str, Any] = {}
    for key, value in (props or {}).items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            out[key] = [
                float(x) if isinstance(x, (int, float)) else x for x in value
            ]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def get_entities(world_graph) -> Dict[str, Any]:
    rooms = [n.name for n in world_graph.get_all_rooms()]
    furniture = []
    for entity in world_graph.get_all_furnitures():
        furniture.append(
            {
                "name": entity.name,
                "sim_handle": getattr(entity, "sim_handle", None),
            }
        )
    objects = []
    for entity in world_graph.get_all_objects():
        objects.append(
            {
                "name": entity.name,
                "sim_handle": getattr(entity, "sim_handle", None),
            }
        )
    receptacles = [n.name for n in world_graph.get_all_receptacles()]
    return {
        "rooms": rooms,
        "furniture": furniture,
        "objects": objects,
        "receptacles": receptacles,
        "entity_names": rooms
        + [f["name"] for f in furniture]
        + [o["name"] for o in objects]
        + receptacles,
    }


def _object_state_props(obj) -> Dict[str, Any]:
    if not hasattr(obj, "properties"):
        return {}
    states = obj.properties.get("states", {})
    if states:
        return _safe_props(states)
    state_props = {
        k: v
        for k, v in obj.properties.items()
        if k in ["is_powered_on", "is_filled", "is_clean", "is_open"]
    }
    return _safe_props(state_props)


def _node_entry(node, children: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"name": node.name, "type": node.__class__.__name__}
    if children:
        entry["children"] = children
    states = _object_state_props(node)
    if states:
        entry["states"] = states
    return entry


def refresh_graph_states(env_interface, world_graph=None) -> None:
    """Pull simulator states onto the GT graph, then copy them onto world_graph."""
    if not (
        env_interface
        and hasattr(env_interface, "perception")
        and env_interface.perception
    ):
        return
    env_interface.perception.update_object_and_furniture_states()
    gt_graph = env_interface.perception.gt_graph
    if world_graph is None or world_graph is gt_graph:
        return
    for getter in (gt_graph.get_all_objects, gt_graph.get_all_furnitures):
        for src in getter() or []:
            try:
                dst = world_graph.get_node_from_name(src.name)
            except (ValueError, KeyError):
                continue
            states = src.properties.get("states") if hasattr(src, "properties") else None
            if states and hasattr(dst, "set_state"):
                dst.set_state(dict(states))


def get_object_states(env_interface, object_name: Optional[str] = None) -> Dict[str, Any]:
    if not (
        env_interface
        and hasattr(env_interface, "perception")
        and env_interface.perception
    ):
        return {"error": "No GT graph available (perception not initialized)"}

    refresh_graph_states(env_interface)
    gt_graph = env_interface.perception.gt_graph
    objects = gt_graph.get_all_objects()
    result: Dict[str, Any] = {"source": "GT", "objects": {}}

    found = False
    for obj in objects:
        if object_name and obj.name != object_name:
            continue
        found = True
        states = _object_state_props(obj)
        result["objects"][obj.name] = states if states else {"note": "No states available"}

    if object_name and not found:
        result["error"] = f"Object '{object_name}' not found"
        result["available"] = sorted([o.name for o in objects])[:50]
    return result


def get_locations(
    env_interface, entity_name: Optional[str] = None
) -> Dict[str, Any]:
    if not (
        env_interface
        and hasattr(env_interface, "perception")
        and env_interface.perception
    ):
        return {"error": "No GT graph available (perception not initialized)"}

    gt_graph = env_interface.perception.gt_graph
    locations: Dict[str, Any] = {"source": "GT", "locations": {}}

    def _loc_entry(entity) -> Optional[Dict[str, Any]]:
        translation = entity.properties.get("translation", None)
        if translation is None:
            return None
        loc = list(translation) if hasattr(translation, "__iter__") else translation
        entry: Dict[str, Any] = {"xyz": [float(x) for x in loc]}
        rotation = entity.properties.get("rotation", None)
        if rotation is not None:
            entry["rotation"] = [float(x) for x in rotation]
        if getattr(entity, "sim_handle", None):
            entry["sim_handle"] = entity.sim_handle
        return entry

    if entity_name:
        try:
            entity = gt_graph.get_node_from_name(entity_name)
            entry = _loc_entry(entity)
            if entry is None:
                return {"error": f"No translation found for '{entity_name}'"}
            locations["locations"][entity_name] = entry
            return locations
        except (ValueError, KeyError) as exc:
            available = [obj.name for obj in gt_graph.get_all_objects()]
            available.extend([f.name for f in gt_graph.get_all_furnitures()])
            return {
                "error": f"Entity '{entity_name}' not found: {exc}",
                "available": sorted(available)[:50],
            }

    for obj in sorted(gt_graph.get_all_objects(), key=lambda o: o.name):
        entry = _loc_entry(obj)
        if entry:
            locations["locations"][obj.name] = entry
    for furn in sorted(gt_graph.get_all_furnitures(), key=lambda f: f.name):
        entry = _loc_entry(furn)
        if entry:
            locations["locations"][furn.name] = entry
    locations["count"] = len(locations["locations"])
    return locations


def get_articulated(world_graph, env_interface=None) -> Dict[str, Any]:
    gt_graph = None
    state_source = "concept graph"
    if (
        env_interface
        and hasattr(env_interface, "perception")
        and env_interface.perception
    ):
        refresh_graph_states(env_interface, world_graph)
        gt_graph = env_interface.perception.gt_graph
        state_source = "GT graph"

    all_furniture = world_graph.get_all_furnitures()
    articulated = [f for f in all_furniture if f.is_articulated()]
    by_type: Dict[str, List[Dict[str, Any]]] = {}

    for furn in articulated:
        parts = furn.name.rsplit("_", 1)
        furn_type = (
            parts[0] if len(parts) > 1 and parts[1].isdigit() else furn.name
        )
        if gt_graph:
            try:
                gt_furn = gt_graph.get_node_from_name(furn.name)
                states = gt_furn.properties.get("states", {})
            except (ValueError, KeyError):
                states = furn.properties.get("states", {})
        else:
            states = furn.properties.get("states", {})

        is_open = states.get("is_open")
        entry = {
            "name": furn.name,
            "is_open": is_open,
            "state": (
                "OPEN" if is_open is True else "CLOSED" if is_open is False else "unknown"
            ),
        }
        by_type.setdefault(furn_type, []).append(entry)

    for furn_type in by_type:
        by_type[furn_type] = sorted(by_type[furn_type], key=lambda x: x["name"])

    return {
        "state_source": state_source,
        "total_furniture": len(all_furniture),
        "total_articulated": len(articulated),
        "by_type": by_type,
    }


def _build_hierarchy_node(world_graph, node, visited) -> Optional[Dict[str, Any]]:
    if node in visited:
        return None
    visited.add(node)

    children: List[Dict[str, Any]] = []
    for neighbor in sorted(world_graph.graph[node], key=lambda n: n.name):
        child = _build_hierarchy_node(world_graph, neighbor, visited)
        if child is not None:
            children.append(child)
    return _node_entry(node, children)


def get_hierarchical_graph(world_graph) -> Dict[str, Any]:
    try:
        house = world_graph.get_node_from_name("house")
        tree = _build_hierarchy_node(world_graph, house, set())
        return {"root": tree, "text": world_graph.to_string()}
    except ValueError:
        rooms_out = []
        for room in world_graph.get_all_rooms():
            room_entry = _node_entry(room)
            room_entry["furniture"] = []
            room_entry["objects"] = []
            for furniture in world_graph.get_neighbors_of_type(room, Furniture):
                furn_entry = _node_entry(furniture)
                furn_entry["receptacles"] = []
                furn_entry["objects"] = []
                for receptacle in world_graph.get_neighbors_of_type(
                    furniture, Receptacle
                ):
                    rec_entry = _node_entry(receptacle)
                    rec_entry["objects"] = [
                        _node_entry(o)
                        for o in world_graph.get_neighbors_of_type(receptacle, Object)
                    ]
                    furn_entry["receptacles"].append(rec_entry)
                for obj in world_graph.get_neighbors_of_type(furniture, Object):
                    if world_graph.find_receptacle_for_object(obj) is None:
                        furn_entry["objects"].append(_node_entry(obj))
                room_entry["furniture"].append(furn_entry)
            for obj in world_graph.get_neighbors_of_type(room, Object):
                if world_graph.find_furniture_for_object(obj) is None:
                    room_entry["objects"].append(_node_entry(obj))
            rooms_out.append(room_entry)
        return {"rooms": rooms_out, "note": "House node not found; simplified hierarchy"}


def get_gt_graph(env_interface) -> Dict[str, Any]:
    if not hasattr(env_interface, "perception") or env_interface.perception is None:
        return {"error": "No GT graph available (perception not initialized)"}
    refresh_graph_states(env_interface)
    result = get_hierarchical_graph(env_interface.perception.gt_graph)
    result["source"] = "gt"
    return result


def get_robot_graph(world_graph, env_interface=None) -> Dict[str, Any]:
    if world_graph is None:
        return {"error": "No robot world graph available"}
    refresh_graph_states(env_interface, world_graph)
    result = get_hierarchical_graph(world_graph)
    result["source"] = "robot"
    return result


def inspect(
    kind: str,
    world_graph,
    env_interface,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch inspector by kind string."""
    if kind == "entities":
        return get_entities(world_graph)
    if kind == "graph":
        return get_robot_graph(world_graph, env_interface)
    if kind == "state":
        return get_object_states(env_interface, name)
    if kind == "location":
        return get_locations(env_interface, name)
    if kind == "articulated":
        return get_articulated(world_graph, env_interface)
    if kind == "gt":
        return get_gt_graph(env_interface)
    return {"error": f"Unknown inspector kind: {kind}"}
