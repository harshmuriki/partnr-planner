"""Privileged placement oracle: where each object should end up.

An upper bound on placement grounding, not a deployable component: it reads the
episode's evaluation propositions (is_on_top / is_inside / is_next_to, and
is_waited_inside). Only propositions that must hold in the final state give an
object's destination; the others (e.g. bread inside the microwave before it is
served) are intermediate stops, hinted as such.

  hint   tells the VLM, in the planning prompt, where known objects belong
  place  adds ``next_to, <partner>`` to Place calls so adjacency goals are met

Hints only cover objects already in the robot's world graph, so the oracle
reveals where objects go, never which objects exist.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

MODES = ("none", "hint", "place", "full")


@dataclass
class PlacementGoal:
    obj: str
    relation: Optional[str] = None  # "on" | "within"
    receptacles: List[str] = field(default_factory=list)
    next_to: Set[str] = field(default_factory=set)
    # Intermediate placements (relation, receptacle) before the final one, and
    # receptacles the object must be waited inside (is_waited_inside).
    via: List[Tuple[str, str]] = field(default_factory=list)
    wait_in: List[str] = field(default_factory=list)


def episode_propositions(env_interface) -> list:
    try:
        episode = env_interface.env.env.env._env.current_episode
        return list(episode.evaluation_propositions)
    except AttributeError:
        return []


def terminal_indices(env_interface) -> Optional[Set[int]]:
    """Indices of propositions that must hold at the end; None if every one must."""
    try:
        constraints = env_interface.env.env.env._env.current_episode.evaluation_constraints
    except AttributeError:
        return None
    for c in constraints or []:
        if isinstance(c, dict):
            if c.get("type") == "TerminalSatisfactionConstraint":
                return set(c.get("args", {}).get("proposition_indices", []))
        elif type(c).__name__ == "TerminalSatisfactionConstraint":
            return set(getattr(c, "proposition_indices", []))
    return None


def proposition_args(prop) -> dict:
    return prop.get("args", {}) if isinstance(prop, dict) else getattr(prop, "args", {})


def proposition_name(prop) -> str:
    if isinstance(prop, dict):
        return prop.get("function_name", "")
    return getattr(prop, "function_name", "")


def handle_names(env_interface, handles: Optional[Iterable[str]]) -> List[str]:
    """World-graph names for sim handles, in order; unmappable handles are skipped."""
    graph = getattr(env_interface, "full_world_graph", None)
    names: List[str] = []
    for handle in handles or []:
        try:
            name = graph.get_node_from_sim_handle(handle).name
        except Exception:
            continue
        if name not in names:
            names.append(name)
    return names


_RELATIONS = {"is_on_top": "on", "is_inside": "within"}


def placement_goals(env_interface) -> Dict[str, PlacementGoal]:
    goals: Dict[str, PlacementGoal] = {}

    def goal(name: str) -> PlacementGoal:
        return goals.setdefault(name, PlacementGoal(obj=name))

    terminal = terminal_indices(env_interface)
    for i, prop in enumerate(episode_propositions(env_interface)):
        fn, args = proposition_name(prop), proposition_args(prop)
        if fn in _RELATIONS or fn == "is_waited_inside":
            recs = handle_names(env_interface, args.get("receptacle_handles"))
            for obj in handle_names(env_interface, args.get("object_handles")):
                g = goal(obj)
                if fn == "is_waited_inside":
                    g.wait_in += [r for r in recs if r not in g.wait_in]
                elif terminal is None or i in terminal:
                    g.relation = _RELATIONS[fn]
                    g.receptacles += [r for r in recs if r not in g.receptacles]
                else:
                    g.via += [(_RELATIONS[fn], r) for r in recs if (_RELATIONS[fn], r) not in g.via]
        elif fn == "is_next_to":
            side_a = handle_names(env_interface, args.get("entity_handles_a"))
            side_b = handle_names(env_interface, args.get("entity_handles_b"))
            for a in side_a:
                for b in side_b:
                    if a != b:
                        goal(a).next_to.add(b)
                        goal(b).next_to.add(a)
    return goals


def format_hint(goals: Dict[str, PlacementGoal], known: Set[str]) -> str:
    lines = []
    for name in sorted(goals):
        g = goals[name]
        if name not in known:
            continue
        parts = []
        if g.relation and g.receptacles:
            where = " or ".join(g.receptacles)
            parts.append(f"{'on' if g.relation == 'on' else 'inside'} {where}")
        partners = sorted(g.next_to)
        if partners:
            parts.append("next to " + " and ".join(partners))
        line = f"- {name} must end up {', '.join(parts)}" if parts else ""
        stops = [r for _, r in g.via] + [r for r in g.wait_in if r not in {r for _, r in g.via}]
        if stops:
            stop = f"inside {' or '.join(stops)}" + (" and waited there (waited_in)" if g.wait_in else "")
            line = f"{line}; before that it must be {stop}" if line else f"- {name} must first be {stop}"
        if line:
            lines.append(line)
    if not lines:
        return ""
    return (
        "\nPlacement oracle (privileged, always correct; use these destinations):\n"
        + "\n".join(lines)
        + "\n"
    )


def place_constraint(
    goal: Optional[PlacementGoal],
    on_furniture: Iterable[str],
    placed_order: Sequence[str] = (),
) -> Optional[str]:
    """A next_to partner already on the destination; the latest placed wins."""
    if goal is None or not goal.next_to:
        return None
    candidates = set(goal.next_to) & set(on_furniture)
    if not candidates:
        return None
    for name in reversed(placed_order):
        if name in candidates:
            return name
    return sorted(candidates)[0]
