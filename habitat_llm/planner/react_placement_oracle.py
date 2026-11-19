"""Privileged placement oracle for the ReAct planner (ablation, not deployable).

Reads the episode's evaluation propositions and tells the planner where each
object it already knows about must be placed, using world-graph names the
planner can act on (e.g. ``book_0 must end up on table_0``).

Compared with habitat_llm.rail_planner.placement_oracle.format_hint (whose
helpers it reuses), it keeps adjacency goals that must hold at the end apart
from ones that only have to hold before another goal (a TemporalConstraint
edge, e.g. a glass next to the soap before it is cleaned).

Only objects already in the robot's world graph are covered, so the oracle
reveals where objects go, never which objects exist or where they are now.
Enabled with plan_config.placement_oracle=hint (default none).
"""

from typing import Dict, List, Optional, Set, Tuple

from habitat_llm.rail_planner import placement_oracle as po

MODES = ("none", "hint")

_STATE_VERBS = {
    "is_clean": "cleaned",
    "is_filled": "filled",
    "is_powered_off": "turned off",
    "is_powered_on": "turned on",
}


def _dag_edges(env_interface) -> List[Tuple[int, int]]:
    try:
        constraints = env_interface.env.env.env._env.current_episode.evaluation_constraints
    except AttributeError:
        return []
    edges: List[Tuple[int, int]] = []
    for c in constraints or []:
        if isinstance(c, dict):
            if c.get("type") == "TemporalConstraint":
                edges += [tuple(e) for e in c.get("args", {}).get("dag_edges", [])]
        elif type(c).__name__ == "TemporalConstraint":
            dag = getattr(c, "dag", None)  # networkx DiGraph built from dag_edges
            edges += [tuple(e) for e in (dag.edges if dag is not None else [])]
    return edges


class ReactPlacementOracle:
    """Builds hint lines once per episode; renders those for known objects."""

    def __init__(self, env_interface) -> None:
        self.env_interface = env_interface
        self._lines: Optional[Dict[str, List[str]]] = None  # object -> hint lines
        self._sent: Set[str] = set()

    def reset(self) -> None:
        self._lines = None
        self._sent = set()

    def _build(self) -> Dict[str, List[str]]:
        env = self.env_interface
        props = po.episode_propositions(env)
        terminal = po.terminal_indices(env)
        is_terminal = lambda i: terminal is None or i in terminal  # noqa: E731
        names = lambda key, args: po.handle_names(env, args.get(key))  # noqa: E731

        dest: Dict[str, Tuple[str, List[str]]] = {}
        next_to: Dict[str, Set[str]] = {}
        before: Dict[str, List[str]] = {}  # object -> "next to X before Y is cleaned"
        edges = _dag_edges(env)
        for i, prop in enumerate(props):
            fn, args = po.proposition_name(prop), po.proposition_args(prop)
            if fn in ("is_on_top", "is_inside") and is_terminal(i):
                rel = "on" if fn == "is_on_top" else "inside"
                recs = names("receptacle_handles", args)
                for obj in names("object_handles", args):
                    dest[obj] = (rel, recs)
            elif fn == "is_next_to":
                side_a, side_b = names("entity_handles_a", args), names("entity_handles_b", args)
                pairs = [(a, b) for a in side_a for b in side_b if a != b]
                if is_terminal(i):
                    for a, b in pairs:
                        next_to.setdefault(a, set()).add(b)
                        next_to.setdefault(b, set()).add(a)
                    continue
                # Intermediate adjacency: phrase it against the goals it must precede.
                for src, dst in edges:
                    if src != i or dst >= len(props):
                        continue
                    later = props[dst]
                    verb = _STATE_VERBS.get(po.proposition_name(later))
                    targets = names("object_handles", po.proposition_args(later))
                    for a, b in pairs:
                        for obj, other in ((a, b), (b, a)):
                            if verb and obj in targets:
                                before.setdefault(obj, []).append(
                                    f"{obj} must be next to {other} before {obj} is {verb}"
                                )

        lines: Dict[str, List[str]] = {}
        for obj in sorted(set(dest) | set(next_to) | set(before)):
            parts = []
            if obj in dest:
                rel, recs = dest[obj]
                parts.append(f"{rel} {' or '.join(recs)}")
            if next_to.get(obj):
                parts.append("next to " + " and ".join(sorted(next_to[obj])))
            out = [f"{obj} must end up {', '.join(parts)}"] if parts else []
            out += before.get(obj, [])
            if out:
                lines[obj] = out
        return lines

    def new_hint(self, world_graph) -> str:
        """Hint text for goal objects that became known since the last call ('' if none)."""
        if self._lines is None:
            self._lines = self._build()
        known = {o.name for o in world_graph.get_all_objects()}
        fresh = [o for o in self._lines if o in known and o not in self._sent]
        if not fresh:
            return ""
        self._sent.update(fresh)
        body = "\n".join(f"- {line}" for o in fresh for line in self._lines[o])
        return f"\nPlacement oracle (privileged, always correct; use these destinations):\n{body}"
