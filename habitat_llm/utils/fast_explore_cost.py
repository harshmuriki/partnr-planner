"""Approximate OracleExploreSkill cost from navmesh geodesics (no walking)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FORWARD_VELOCITY = 10.0
SIM_FREQ = 120.0
MAX_NAV_STEPS_PER_FURNITURE = 300
MAX_SKILL_STEPS = 2400
MAX_FURNITURE_SAMPLES_PER_ROOM = 10


def meters_per_step(
    forward_velocity: float = FORWARD_VELOCITY, sim_freq: float = SIM_FREQ
) -> float:
    if sim_freq <= 0:
        raise ValueError("sim_freq must be positive")
    return forward_velocity / sim_freq


def sim_time_from_steps(steps: int, sim_freq: float = SIM_FREQ) -> float:
    if sim_freq <= 0:
        raise ValueError("sim_freq must be positive")
    return float(steps) / sim_freq


def walk_to_approx_ratio(walked_steps: int, approx_steps: int) -> Optional[float]:
    """walked_sim_steps / geodesic_approx_steps for scaling later fast Explores."""
    if walked_steps <= 0 or approx_steps <= 0:
        return None
    ratio = float(walked_steps) / float(approx_steps)
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    return ratio


def scale_explore_steps(approx_steps: int, ratio: Optional[float]) -> int:
    raw = max(int(approx_steps), 0)
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return raw
    return max(0, int(round(raw * float(ratio))))


def approx_steps_from_geodesic(
    distance_m: float,
    forward_velocity: float = FORWARD_VELOCITY,
    sim_freq: float = SIM_FREQ,
    max_steps: int = MAX_NAV_STEPS_PER_FURNITURE,
) -> int:
    """Convert geodesic meters to integer sim steps, capped per hop."""
    if distance_m <= 0 or not np.isfinite(distance_m):
        return 0
    raw = distance_m / meters_per_step(forward_velocity, sim_freq)
    steps = int(np.ceil(raw))
    if max_steps > 0:
        steps = min(steps, max_steps)
    return max(steps, 0)


def sample_explore_furniture_queue(
    furniture: Sequence[Any],
    max_samples: int = MAX_FURNITURE_SAMPLES_PER_ROOM,
    rng: Optional[np.random.Generator] = None,
) -> List[Any]:
    """Match OracleExploreSkill.set_target furniture sampling."""
    furniture_list = list(furniture)
    if max_samples > 0 and len(furniture_list) > max_samples:
        choice_rng = rng if rng is not None else np.random
        sampled_idxs = choice_rng.choice(
            len(furniture_list), size=max_samples, replace=False
        )
        return [furniture_list[int(i)] for i in sampled_idxs]
    return furniture_list


def _as_xyz(pos: Any) -> Optional[np.ndarray]:
    if pos is None:
        return None
    try:
        arr = np.array(pos, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[:3]


def furniture_translation(node: Any) -> Optional[np.ndarray]:
    if node is None:
        return None
    if hasattr(node, "get_property"):
        try:
            value = node.get_property("translation")
        except Exception:
            value = None
        xyz = _as_xyz(value)
        if xyz is not None:
            return xyz
    props = getattr(node, "properties", None)
    if isinstance(props, dict):
        return _as_xyz(props.get("translation"))
    return None


def furniture_name(node: Any, fallback: str = "furniture") -> str:
    name = getattr(node, "name", None)
    if isinstance(name, str) and name:
        return name
    return fallback


def snap_nav_point(pathfinder: Any, pos: np.ndarray) -> Optional[np.ndarray]:
    if pathfinder is None:
        return _as_xyz(pos)
    snap = getattr(pathfinder, "snap_point", None)
    if snap is None:
        return _as_xyz(pos)
    snapped = snap(pos)
    return _as_xyz(snapped)


def geodesic_distance(pathfinder: Any, start: np.ndarray, end: np.ndarray) -> Optional[float]:
    if pathfinder is None:
        return float(np.linalg.norm((end - start)[[0, 2]]))
    find_path = getattr(pathfinder, "find_path", None)
    if find_path is None:
        return None
    try:
        import habitat_sim
    except ImportError:
        return None
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    found = find_path(path)
    if not found:
        return None
    dist = float(getattr(path, "geodesic_distance", float("nan")))
    if not np.isfinite(dist):
        return None
    return dist


def estimate_explore_tour(
    pathfinder: Any,
    start_pos: Any,
    furniture: Sequence[Any],
    *,
    room_name: Optional[str] = None,
    max_samples: int = MAX_FURNITURE_SAMPLES_PER_ROOM,
    max_nav_steps_per_furniture: int = MAX_NAV_STEPS_PER_FURNITURE,
    max_skill_steps: int = MAX_SKILL_STEPS,
    forward_velocity: float = FORWARD_VELOCITY,
    sim_freq: float = SIM_FREQ,
    rng: Optional[np.random.Generator] = None,
    geodesic_fn=None,
) -> Dict[str, Any]:
    """Tour current pose through the OracleExploreSkill furniture queue."""
    start_xyz = _as_xyz(start_pos)
    queue = sample_explore_furniture_queue(furniture, max_samples=max_samples, rng=rng)
    legs: List[Dict[str, Any]] = []
    if start_xyz is None:
        return {
            "total_meters": 0.0,
            "total_steps": 0,
            "total_time_s": 0.0,
            "legs": legs,
            "furniture_count": len(queue),
            "summary_lines": ["[fast_explore] approx tour skipped: robot pose unavailable"],
        }

    current = snap_nav_point(pathfinder, start_xyz)
    if current is None:
        current = start_xyz
    prev_name = "robot"
    total_meters = 0.0
    total_steps = 0
    geo = geodesic_fn if geodesic_fn is not None else geodesic_distance

    for furn in queue:
        if max_skill_steps > 0 and total_steps >= max_skill_steps:
            break
        name = furniture_name(furn)
        target_xyz = furniture_translation(furn)
        if target_xyz is None:
            legs.append(
                {
                    "from": prev_name,
                    "to": name,
                    "meters": None,
                    "steps": 0,
                    "skipped": True,
                    "reason": "no translation",
                }
            )
            continue
        snapped = snap_nav_point(pathfinder, target_xyz)
        if snapped is None:
            legs.append(
                {
                    "from": prev_name,
                    "to": name,
                    "meters": None,
                    "steps": 0,
                    "skipped": True,
                    "reason": "not on navmesh",
                }
            )
            continue
        dist = geo(pathfinder, current, snapped)
        if dist is None:
            legs.append(
                {
                    "from": prev_name,
                    "to": name,
                    "meters": None,
                    "steps": 0,
                    "skipped": True,
                    "reason": "no path",
                }
            )
            continue
        remaining = (
            max_skill_steps - total_steps if max_skill_steps > 0 else max_nav_steps_per_furniture
        )
        hop_cap = max_nav_steps_per_furniture
        if remaining > 0:
            hop_cap = min(hop_cap, remaining) if hop_cap > 0 else remaining
        steps = approx_steps_from_geodesic(
            dist,
            forward_velocity=forward_velocity,
            sim_freq=sim_freq,
            max_steps=hop_cap,
        )
        total_meters += dist
        total_steps += steps
        legs.append(
            {
                "from": prev_name,
                "to": name,
                "meters": dist,
                "steps": steps,
                "skipped": False,
                "reason": None,
            }
        )
        current = snapped
        prev_name = name

    total_time_s = sim_time_from_steps(total_steps, sim_freq)
    visited = sum(1 for leg in legs if not leg.get("skipped"))
    room_label = f" {room_name}" if room_name else ""
    summary_lines = [
        (
            f"[fast_explore] approx tour{room_label}: {visited} furniture, "
            f"{total_meters:.1f} m, ~{total_steps} sim steps, "
            f"~{total_time_s:.2f} s sim time"
        )
    ]
    for leg in legs:
        if leg.get("skipped"):
            summary_lines.append(
                f"  {leg['from']} -> {leg['to']}  skipped ({leg.get('reason')})"
            )
            continue
        meters = leg.get("meters") or 0.0
        summary_lines.append(
            f"  {leg['from']} -> {leg['to']}  {meters:.1f} m  ~{leg['steps']} steps"
        )
    return {
        "total_meters": total_meters,
        "total_steps": total_steps,
        "total_time_s": total_time_s,
        "legs": legs,
        "furniture_count": len(queue),
        "summary_lines": summary_lines,
    }
