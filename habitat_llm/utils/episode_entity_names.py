"""Map simulator handles to logical names from episode and scene metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional


def furniture_names_by_sim_handle(
    scene_info: Dict[str, Any],
    generated_names: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Prefer saved scene names; fill missing handles from generation metadata.

    Explicit spec aliases refer to the same node as their canonical runtime_name.
    Reserve all saved names so newly discovered furniture cannot steal them.
    """
    saved = scene_info.get("receptacle_to_handle") or {}
    aliases = scene_info.get("spec_furniture_aliases") or {}
    names: Dict[str, str] = {}
    for name, handle in saved.items():
        canonical = aliases.get(name, {}).get("runtime_name")
        if canonical and saved.get(canonical) == handle:
            continue
        names.setdefault(handle, name)
    reserved = set(saved)
    for name, handle in (generated_names or {}).items():
        if handle not in names and name not in reserved:
            names[handle] = name
            reserved.add(name)
    return names


def episode_info_dict(ep_info: Any) -> Dict[str, Any]:
    if ep_info is None:
        return {}
    if isinstance(ep_info, dict):
        info = ep_info.get("info")
        return info if isinstance(info, dict) else {}
    info = getattr(ep_info, "info", None)
    return info if isinstance(info, dict) else {}


def object_names_by_sim_handle(ep_info: Any) -> Dict[str, str]:
    """Return sim handle -> logical name (e.g. bottle_0) from variant_spec / sample_configs."""
    info = episode_info_dict(ep_info)
    names: Dict[str, str] = {}

    variant = info.get("variant_spec") or {}
    entity_handles = (
        variant.get("entity_handles") if isinstance(variant, dict) else None
    )
    if isinstance(entity_handles, dict):
        for name, handle in entity_handles.items():
            if isinstance(name, str) and name and isinstance(handle, str) and handle:
                names[handle] = name

    sample_configs = info.get("sample_configs") or {}
    if isinstance(sample_configs, dict):
        for item in sample_configs.values():
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            instances = item.get("object_instances") or []
            if not isinstance(name, str) or not name or not instances:
                continue
            instance = instances[0]
            if not isinstance(instance, str) or not instance:
                continue
            handle = f"{instance}_:0000"
            names.setdefault(handle, name)

    return names


def graph_object_name(
    obj_handle: str,
    obj_type: str,
    object_index: int,
    names_by_handle: Optional[Dict[str, str]] = None,
    existing_names: Optional[set] = None,
) -> str:
    """Prefer the dataset logical name; otherwise type_index (Habitat-LLM default)."""
    existing = existing_names or set()
    preferred = None
    if names_by_handle:
        preferred = names_by_handle.get(obj_handle)
    if preferred and preferred not in existing:
        return preferred
    fallback = f"{obj_type}_{object_index}"
    if fallback not in existing:
        return fallback
    suffix = 0
    while f"{obj_type}_{suffix}" in existing:
        suffix += 1
    return f"{obj_type}_{suffix}"
