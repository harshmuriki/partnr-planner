"""Load remembered object names from an episode folder's initial_robot_memory.json."""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from habitat_llm.world_model import Floor, Furniture, Object, Receptacle
from habitat_llm.world_model.world_graph import flip_edge


MEMORY_FILENAME = "initial_robot_memory.json"
SCENE_INFO_FILENAME = "scene_info.json"

_ON_LOCATION_RE = re.compile(
    r"^(on|within|inside|in)\s+(\S+)(?:\s+\(([^)]+)\))?$",
    re.IGNORECASE,
)
_FLOOR_LOCATION_RE = re.compile(
    r"^floor\s+(\S+)(?:\s+\(([^)]+)\))?$",
    re.IGNORECASE,
)
_ROOM_SUFFIX_RE = re.compile(r"_\d+$")


@dataclass(frozen=True)
class ParsedPlacement:
    relation: str
    name: str
    room: Optional[str]


@dataclass(frozen=True)
class RememberedObject:
    entity: str
    sim_handle: Optional[str]
    outdated_location: Optional[str]


def resolve_initial_robot_memory_path(data_path: Optional[str]) -> Optional[str]:
    folder = _episode_folder(data_path)
    if folder is None:
        return None
    candidate = os.path.join(folder, MEMORY_FILENAME)
    if os.path.isfile(candidate):
        return candidate
    return None


def load_remembered_object_names(data_path: Optional[str]) -> List[str]:
    return [record.entity for record in load_remembered_object_records(data_path)]


def load_remembered_object_refs(
    data_path: Optional[str],
) -> List[Tuple[str, Optional[str]]]:
    """Load remembered logical names and their simulator handles."""
    return [
        (record.entity, record.sim_handle)
        for record in load_remembered_object_records(data_path)
    ]


def load_remembered_object_records(data_path: Optional[str]) -> List[RememberedObject]:
    json_path = resolve_initial_robot_memory_path(data_path)
    if json_path is None:
        return []

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    handles_by_entity: Dict[str, str] = {}
    dataset_path = _resolve_dataset_path(data_path)
    if dataset_path is not None:
        opener = gzip.open if dataset_path.endswith(".gz") else open
        with opener(dataset_path, "rt", encoding="utf-8") as handle:
            dataset = json.load(handle)
        episodes = dataset.get("episodes", [])
        if episodes:
            handles_by_entity = (
                episodes[0]
                .get("info", {})
                .get("variant_spec", {})
                .get("entity_handles", {})
            )

    records = []
    for item in payload.get("objects", []):
        if not isinstance(item, dict) or item.get("in_initial_robot_memory") is not True:
            continue
        entity = item.get("entity")
        if not isinstance(entity, str) or not entity:
            continue
        location = item.get("outdated_location")
        if not isinstance(location, str) or not location.strip():
            location = None
        records.append(
            RememberedObject(
                entity=entity,
                sim_handle=handles_by_entity.get(entity),
                outdated_location=location,
            )
        )
    return records


def load_scene_info(data_path: Optional[str]) -> Dict[str, Any]:
    folder = _episode_folder(data_path)
    if folder is None:
        return {}
    candidate = os.path.join(folder, SCENE_INFO_FILENAME)
    if not os.path.isfile(candidate):
        return {}
    with open(candidate, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def parse_outdated_location(location: str) -> Optional[ParsedPlacement]:
    text = (location or "").strip()
    if not text:
        return None
    floor_match = _FLOOR_LOCATION_RE.match(text)
    if floor_match:
        return ParsedPlacement(
            relation="floor",
            name=floor_match.group(1),
            room=floor_match.group(2),
        )
    on_match = _ON_LOCATION_RE.match(text)
    if on_match:
        relation = on_match.group(1).lower()
        if relation in ("within", "inside", "in"):
            relation = "within"
        else:
            relation = "on"
        return ParsedPlacement(
            relation=relation,
            name=on_match.group(2),
            room=on_match.group(3),
        )
    return None


def resolve_placement_node(graph, location: str, scene_info: Optional[Dict[str, Any]] = None):
    parsed = parse_outdated_location(location)
    if parsed is None or graph is None:
        return None
    if parsed.relation == "floor":
        return _resolve_floor_node(graph, parsed)
    node = _resolve_named_furniture(graph, parsed.name, scene_info)
    if node is not None:
        return node
    return _resolve_named_furniture_by_room(graph, parsed)


def apply_outdated_placements(
    agent_graph,
    gt_graph,
    records: List[RememberedObject],
    scene_info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Insert stale-memory ghosts at outdated furniture/floors in the agent graph."""
    warnings: List[str] = []
    for record in records:
        if not record.outdated_location:
            continue
        placement = resolve_placement_node(
            agent_graph, record.outdated_location, scene_info
        )
        if placement is None and gt_graph is not None:
            placement = resolve_placement_node(
                gt_graph, record.outdated_location, scene_info
            )
            placement = _corresponding_node(agent_graph, placement)
        if placement is None:
            warnings.append(
                f"{record.entity} stale location '{record.outdated_location}' not in graph"
            )
            continue
        source = None
        if gt_graph is not None:
            source = _find_object_node(gt_graph, record)
        ghost = _make_memory_object(record.entity, placement, agent_graph, source)
        agent_graph.add_node(ghost)
        _reparent_object(agent_graph, ghost, placement)
    return warnings


def _episode_folder(data_path: Optional[str]) -> Optional[str]:
    if not data_path:
        return None
    path = os.path.abspath(str(data_path))
    return path if os.path.isdir(path) else os.path.dirname(path)


def _resolve_dataset_path(data_path: Optional[str]) -> Optional[str]:
    if not data_path:
        return None
    path = os.path.abspath(str(data_path))
    if os.path.isdir(path):
        for filename in ("dataset.json", "dataset.json.gz"):
            candidate = os.path.join(path, filename)
            if os.path.isfile(candidate):
                return candidate
        return None
    return path if os.path.isfile(path) else None


def _normalize_room_key(name: str) -> str:
    text = name.strip().lower().replace("/", "_").replace(" ", "_")
    text = text.replace("floor_", "")
    return _ROOM_SUFFIX_RE.sub("", text)


def _resolve_named_furniture(graph, name: str, scene_info: Optional[Dict[str, Any]]):
    handles = (scene_info or {}).get("receptacle_to_handle") or {}
    handle = handles.get(name)
    if handle:
        try:
            return graph.get_node_from_sim_handle(handle)
        except ValueError:
            pass
    try:
        node = graph.get_node_from_name(name)
    except ValueError:
        return None
    if isinstance(node, Furniture) and not isinstance(node, Floor):
        return node
    return None


def _resolve_named_furniture_by_room(graph, parsed: ParsedPlacement):
    if not parsed.room:
        return None
    room_key = _normalize_room_key(parsed.room)
    prefix = parsed.name.rsplit("_", 1)[0] if "_" in parsed.name else parsed.name
    for node in graph.get_all_furnitures():
        if isinstance(node, Floor):
            continue
        if not node.name.startswith(prefix):
            continue
        try:
            room = graph.get_room_for_entity(node)
        except Exception:
            room = None
        if room is not None and _normalize_room_key(room.name) == room_key:
            return node
    return None


def _resolve_floor_node(graph, parsed: ParsedPlacement):
    try:
        node = graph.get_node_from_name(parsed.name)
        if isinstance(node, Floor):
            return node
    except ValueError:
        pass
    room_key = _normalize_room_key(parsed.room or parsed.name)
    for node in graph.get_all_furnitures():
        if not isinstance(node, Floor):
            continue
        if _normalize_room_key(node.name) == room_key:
            return node
        try:
            room = graph.get_room_for_entity(node)
        except Exception:
            room = None
        if room is not None and _normalize_room_key(room.name) == room_key:
            return node
    return None


def _find_object_node(graph, record: RememberedObject):
    if record.sim_handle:
        try:
            node = graph.get_node_from_sim_handle(record.sim_handle)
            if isinstance(node, Object):
                return node
        except ValueError:
            pass
    try:
        node = graph.get_node_from_name(record.entity)
        if isinstance(node, Object):
            return node
    except ValueError:
        return None
    return None


def _corresponding_node(graph, node):
    if node is None:
        return None
    if graph.has_node(node):
        return node
    if getattr(node, "sim_handle", None) and node.sim_handle != "floor":
        try:
            return graph.get_node_from_sim_handle(node.sim_handle)
        except ValueError:
            pass
    try:
        return graph.get_node_from_name(node.name)
    except ValueError:
        return None


def next_indexed_object_name(graph, entity: str) -> str:
    prefix, sep, suffix = entity.rpartition("_")
    if not sep or not suffix.isdigit():
        prefix = entity
        start = 0
    else:
        start = int(suffix)
    existing = [start]
    for node in graph.get_all_objects():
        node_prefix, node_sep, node_suffix = node.name.rpartition("_")
        if node_sep and node_prefix == prefix and node_suffix.isdigit():
            existing.append(int(node_suffix))
    return f"{prefix}_{max(existing) + 1}"


def _make_memory_object(entity: str, placement, graph, source=None) -> Object:
    ghost_name = next_indexed_object_name(graph, entity)
    object_type = entity.rsplit("_", 1)[0] if "_" in entity else entity
    properties: Dict[str, Any] = {"type": object_type}
    translation = getattr(placement, "properties", {}).get("translation")
    if translation is None and source is not None:
        translation = source.properties.get("translation")
    if translation is not None:
        properties["translation"] = translation
    if source is not None and "states" in source.properties:
        properties["states"] = dict(source.properties["states"])
    return Object(
        ghost_name,
        properties,
        sim_handle=f"{ghost_name}.stale_memory",
    )


def _reparent_object(graph, object_node, placement) -> None:
    if not graph.has_node(object_node):
        graph.add_node(object_node)
    if not graph.has_node(placement):
        graph.add_node(placement)
    neighbors = list(graph.get_neighbors(object_node).keys())
    for neighbor in neighbors:
        if isinstance(neighbor, (Furniture, Receptacle)):
            graph.remove_edge(object_node, neighbor)
    edge = "inside" if isinstance(placement, Object) else "on"
    graph.add_edge(object_node, placement, edge, flip_edge(edge))
