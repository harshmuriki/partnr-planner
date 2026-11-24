"""When to re-query the VLM after a skill finishes, besides on failure, and
whether to check the goal before accepting a finished plan.

Triggers, from least to most reactive:
  failure     never fires here; the base planner replans only on failure/Explore
  oracle      fires on task-relevant changes, judged with privileged ground truth
  any_change  fires whenever the robot's observed scene changed
  every_step  fires after every completed skill

End-of-plan checks (when the active branch runs out of subgoals):
  none             accept the finished plan (base behaviour)
  oracle           replan instead if the simulator's goal check is not satisfied
  oracle_feedback  same, and tell the VLM which required objects it has not
                   found and which known objects are not placed correctly or
                   not in the required state (never where they should go)
  verify_symbolic  deployable (no privileged input): the VLM writes the final
                   conditions the instruction requires, they are checked
                   against the robot's own scene state, and unmet ones are
                   fed back (habitat_llm.rail_planner.goal_verifier)

Both oracles are upper bounds, not deployable: they read the episode's
evaluation propositions or the simulator's goal check. Only a replan/no-replan
bit reaches the planner, the trigger only looks at changes the robot itself
observed, and the replanning prompt is built from the robot's own observations,
so no ground truth enters it.
"""

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from habitat_llm.rail_planner.placement_oracle import (
    episode_propositions,
    handle_names,
    proposition_args,
    proposition_name,
)
from habitat_llm.rail_planner.subgoal_expanders import build_scene_state

TRIGGERS = ("failure", "every_step", "any_change", "oracle")
END_OF_PLAN_CHECKS = ("none", "oracle", "oracle_feedback", "verify_symbolic")

# kind: new_object | moved | vanished | state
Change = namedtuple("Change", ["kind", "entity", "before", "after"])

_STATE_SETS = ("open", "known_closed", "powered_on", "filled", "cleaned")


@dataclass
class Snapshot:
    """What the robot believes about the scene, minus its own pose and gripper."""

    objects: Set[str] = field(default_factory=set)
    placement: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    holding: Optional[str] = None
    states: Dict[str, FrozenSet[str]] = field(default_factory=dict)


def take_snapshot(world_graph, agent_uid: int, sync_state=None) -> Snapshot:
    """``sync_state(scene_state)`` corrects stale world-graph facts (e.g. the
    held object) before the snapshot is taken."""
    s = build_scene_state(world_graph, agent_uid)
    if sync_state is not None:
        held = s.holding
        sync_state(s)
        if s.holding != held:
            # Rebuild so a stale held object gets its placement back.
            held, s = s.holding, build_scene_state(world_graph, agent_uid)
            s.holding = held
            s.placement.pop(held, None)
    states: Dict[str, Set[str]] = {}
    for attr in _STATE_SETS:
        for name in getattr(s, attr):
            states.setdefault(name, set()).add(attr)
    return Snapshot(
        objects=set(s.objects),
        placement=dict(s.placement),
        holding=s.holding,
        states={k: frozenset(v) for k, v in states.items()},
    )


def action_entities(hl_action) -> Set[str]:
    """Entity names in a ("Tool", "arg1, arg2, ...", ...) high-level action."""
    if not hl_action or len(hl_action) < 2 or not hl_action[1]:
        return set()
    return {a.strip() for a in str(hl_action[1]).split(",") if a.strip()}


def compute_delta(
    prev: Optional[Snapshot], cur: Snapshot, own_action_args: Iterable[str] = ()
) -> List[Change]:
    """Changes between two snapshots that the robot's own action does not explain.

    Entities named in the just-finished action are skipped: their change is the
    expected effect of Pick/Place/Open/... Contents revealed by Open are kept.
    """
    if prev is None:
        return []
    own = set(own_action_args)
    out: List[Change] = []
    for obj in sorted(cur.objects - prev.objects):
        if obj not in own:
            out.append(Change("new_object", obj, None, cur.placement.get(obj)))
    for obj in sorted(prev.objects & cur.objects):
        if obj in own or obj in (prev.holding, cur.holding):
            continue  # picked up or put down by the robot itself
        before, after = prev.placement.get(obj), cur.placement.get(obj)
        if before == after:
            continue
        if after is None:
            if before is not None:
                out.append(Change("vanished", obj, before, None))
        else:
            out.append(Change("moved", obj, before, after))
    for name in sorted(set(prev.states) | set(cur.states)):
        if name in own:
            continue
        before = prev.states.get(name, frozenset())
        after = cur.states.get(name, frozenset())
        if before != after:
            out.append(Change("state", name, tuple(sorted(before)), tuple(sorted(after))))
    return out


def goal_entities(env_interface) -> Set[str]:
    """Names of the objects/receptacles in the episode's evaluation propositions.

    Privileged: this is the ground-truth goal, including the true referent for
    abstract/substitute variants. Unmappable handles are skipped.
    """
    names: Set[str] = set()
    for prop in episode_propositions(env_interface):
        args = proposition_args(prop)
        for key in ("object_handles", "receptacle_handles"):
            names.update(handle_names(env_interface, args.get(key)))
    return names


def gt_goal_status(env_interface) -> Tuple[Optional[bool], Optional[float]]:
    """(task satisfied, percent complete) from the simulator's evaluation measures.

    Privileged. Read-only: the measures are refreshed on every env step, and
    updating them here would append an extra step to the scored state sequence.
    Returns (None, None) if the measures are unavailable.
    """
    try:
        measures = env_interface.env.env.env._env.task.measurements.measures
        success = measures["task_state_success"].get_metric()
        percent = measures["task_percent_complete"].get_metric()
    except Exception:
        return None, None
    return bool(success), float(percent)


_PLACEMENT_FNS = ("is_on_top", "is_inside", "is_in_room", "is_on_floor", "is_next_to", "is_clustered")
_STATE_WORDS = {
    "is_clean": "clean",
    "is_dirty": "dirty",
    "is_powered_on": "powered on",
    "is_powered_off": "powered off",
    "is_filled": "filled",
    "is_empty": "empty",
}


def _entity_type(env_interface, name: str) -> str:
    try:
        node = env_interface.full_world_graph.get_node_from_name(name)
        return str(node.properties.get("type") or name.rsplit("_", 1)[0])
    except Exception:
        return name.rsplit("_", 1)[0]


def unmet_goal_feedback(env_interface, known: Set[str]) -> List[str]:
    """What the planner has not found or not placed correctly, one line each.

    Privileged and read-only: reads which evaluation propositions are unmet in
    the latest evaluated state. Objects the robot has not observed are named
    only by type, and receptacles are never named, so the feedback says what is
    wrong but not where things belong.
    """
    try:
        measures = env_interface.env.env.env._env.task.measurements.measures
        data = measures["auto_eval_proposition_tracker"].get_metric()
        props, latest = data["propositions"], data["state_sequence"][-1]
    except Exception:
        return []
    lines: List[str] = []
    for prop, result in zip(props, latest):
        if getattr(result, "is_satisfied", False):
            continue
        fn, args = proposition_name(prop), proposition_args(prop)
        objs = handle_names(env_interface, args.get("object_handles") or args.get("entity_handles_a"))
        seen = [o for o in objs if o in known]
        if not objs:
            continue
        if not seen:
            line = f"Not found yet: a required {_entity_type(env_interface, objs[0])}."
        elif fn in _STATE_WORDS:
            line = f"{' or '.join(seen)} is not {_STATE_WORDS[fn]} as required."
        elif fn == "is_next_to":
            partners = [o for o in handle_names(env_interface, args.get("entity_handles_b")) if o in known]
            where = f" next to {' or '.join(partners)}" if partners else ""
            line = f"{' or '.join(seen)} is not placed correctly{where}."
        elif fn in _PLACEMENT_FNS:
            line = f"{' or '.join(seen)} is not placed correctly."
        elif fn == "is_waited_inside":
            line = f"{' or '.join(seen)} has not been left to wait inside an appliance as required (waited_in)."
        else:
            line = f"A required condition on {' or '.join(seen)} is not satisfied."
        if line not in lines:
            lines.append(line)
    return lines


def oracle_rule(change: Change, goal: Set[str], plan: Set[str]) -> Optional[str]:
    """R1: a goal entity the plan does not use. R2: a plan entity changed under it."""
    if change.entity in goal and change.entity not in plan:
        return "R1_unused_goal_entity"
    if change.entity in plan and change.kind in ("moved", "vanished", "state"):
        return "R2_plan_assumption_changed"
    return None


def should_replan(
    trigger: str,
    delta: List[Change],
    goal: Set[str],
    plan: Set[str],
    seen: Set[Tuple],
) -> Tuple[bool, Optional[str]]:
    """Return (fire, rule). ``seen`` dedupes oracle firings and is updated in place."""
    if trigger == "failure":
        return False, None
    if trigger == "every_step":
        return True, "every_step"
    if trigger == "any_change":
        return (True, "any_change") if delta else (False, None)
    fired = None
    for change in delta:
        rule = oracle_rule(change, goal, plan)
        key = (change.kind, change.entity, change.after)
        if rule and key not in seen:
            seen.add(key)
            fired = fired or rule
    return fired is not None, fired
