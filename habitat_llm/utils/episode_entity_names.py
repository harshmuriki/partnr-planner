"""Map simulator object handles to episode logical names from dataset.json."""

from __future__ import annotations

from typing import Any, Dict, Optional


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
