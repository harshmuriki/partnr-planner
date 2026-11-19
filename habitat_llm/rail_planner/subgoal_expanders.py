"""Expand VLM-TAMP symbolic subgoals into oracle-skill action sequences.

Replaces the hand-written STRIPS domain + PDDLStream/Fast Downward solver. With
oracle skills and one subgoal at a time, every subgoal type has a fixed skill
sequence modulo a few state checks (already holding, container closed, ...), so
each type gets one explicit expander function instead of a symbolic search.

Plan steps keep the old PDDL action names and argument positions
(``Action(name, args)`` with the agent at ``args[0]``) so the skill mapping,
logs, and HTML renderers are unchanged. Navigate steps carry the real target.
"""

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from habitat_llm.world_model import Furniture, Human, Object, Receptacle, Room, SpotRobot

# Same name and fields as pddlstream's Action, so reprs in logs stay identical.
Action = namedtuple("Action", ["name", "args"])

UNKNOWN_ROOM = "unknown_room"


class SubgoalPlanError(Exception):
    """The subgoal cannot be expanded in the current state; message is shown to the VLM."""


@dataclass
class SceneState:
    agent: str
    holding: Optional[str] = None
    objects: Set[str] = field(default_factory=set)
    furniture: Set[str] = field(default_factory=set)
    rooms: Set[str] = field(default_factory=set)
    room_of: Dict[str, str] = field(default_factory=dict)
    position: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    # object -> (furniture, "on" | "inside")
    placement: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    articulated: Set[str] = field(default_factory=set)
    # Articulated furniture with a known open state; unknown counts as closed.
    open: Set[str] = field(default_factory=set)
    known_closed: Set[str] = field(default_factory=set)
    containers: Set[str] = field(default_factory=set)
    # Movable objects with an annotated interior (e.g. a fruit basket), and
    # object -> the movable container it is inside.
    object_containers: Set[str] = field(default_factory=set)
    contained_in: Dict[str, str] = field(default_factory=dict)
    faucets: Set[str] = field(default_factory=set)
    powered_on: Set[str] = field(default_factory=set)
    # Entities with a power state, and (object, appliance) pairs already waited.
    powerable: Set[str] = field(default_factory=set)
    waited: Set[Tuple[str, str]] = field(default_factory=set)
    filled: Set[str] = field(default_factory=set)
    cleaned: Set[str] = field(default_factory=set)

    def room(self, name: str) -> str:
        return self.room_of.get(name, UNKNOWN_ROOM)

    def where(self, name: str) -> Optional[Tuple[float, ...]]:
        """Position of an entity; a held object is wherever the agent is."""
        return self.position.get(self.agent if name == self.holding else name)

    def resolve(self, name: str) -> str:
        """Map a VLM-written name (spaces -> '_', any case) to the entity name."""
        known = self.objects | self.furniture | self.rooms
        if name in known:
            return name
        key = _name_key(name)
        for entity in known:
            if _name_key(entity) == key:
                return entity
        return name

    def facts(self) -> List[Tuple]:
        """Flat fact list, used only for the per-subgoal log dump."""
        out: List[Tuple] = [("agent", self.agent), ("at", self.agent, self.room(self.agent))]
        out.append(("holding", self.agent, self.holding) if self.holding else ("handempty", self.agent))
        for obj, (furn, rel) in sorted(self.placement.items()):
            out.append(("in" if rel == "inside" else "on", obj, furn))
        out += [("in", obj, box) for obj, box in sorted(self.contained_in.items())]
        out += [("container", c) for c in sorted(self.object_containers)]
        out += [("open", f) for f in sorted(self.open)]
        out += [("closed", f) for f in sorted(self.known_closed)]
        out += [("container", f) for f in sorted(self.containers)]
        out += [("has_faucet", f) for f in sorted(self.faucets)]
        out += [("powered_on", x) for x in sorted(self.powered_on)]
        out += [("filled", x) for x in sorted(self.filled)]
        out += [("cleaned", x) for x in sorted(self.cleaned)]
        out += [("waited_in", obj, app) for obj, app in sorted(self.waited)]
        return out


def _name_key(name: str) -> str:
    return name.strip().replace(" ", "_").lower()


# ---------------------------------------------------------------------------
# WorldGraph -> SceneState
# ---------------------------------------------------------------------------


def _resolve_agent_node(world_graph, agent_uid: int):
    agents = world_graph.get_agents()
    for node in agents:
        if node.name == f"agent_{agent_uid}":
            return node
    return agents[0] if agents else None


def _room_for_entity(world_graph, entity) -> Optional[str]:
    rooms = world_graph.get_neighbors_of_type(entity, Room)
    if rooms:
        return rooms[0].name
    furn = None
    if isinstance(entity, Object):
        furn = world_graph.find_furniture_for_object(entity)
    elif isinstance(entity, Receptacle):
        try:
            furn = world_graph.find_furniture_for_receptacle(entity)
        except ValueError:
            furn = None
    if furn is not None:
        rooms = world_graph.get_neighbors_of_type(furn, Room)
        if rooms:
            return rooms[0].name
    return None


def _record_position(state: SceneState, entity) -> None:
    translation = entity.properties.get("translation")
    if translation is not None:
        state.position[entity.name] = tuple(float(v) for v in translation)


def _held_object_name(world_graph, agent_node) -> Optional[str]:
    held = agent_node.properties.get("last_held_object") if agent_node else None
    if held is not None:
        return held.name if hasattr(held, "name") else str(held)
    for obj in world_graph.get_all_objects():
        if world_graph.is_object_with_agent(obj, agent_type="robot"):
            return obj.name
    return None


def build_scene_state(world_graph, agent_uid: int) -> SceneState:
    agent_node = _resolve_agent_node(world_graph, agent_uid)
    state = SceneState(agent=agent_node.name if agent_node else f"agent_{agent_uid}")
    if agent_node is not None:
        _record_position(state, agent_node)
        room = _room_for_entity(world_graph, agent_node)
        if room:
            state.room_of[state.agent] = room
    state.holding = _held_object_name(world_graph, agent_node)
    state.rooms = {r.name for r in world_graph.get_all_rooms()}

    # Same container definition the VLM prompt uses: furniture with a receptacle.
    for rec in world_graph.get_all_receptacles():
        try:
            state.containers.add(world_graph.find_furniture_for_receptacle(rec).name)
        except ValueError:
            pass

    for furn in world_graph.get_all_furnitures():
        state.furniture.add(furn.name)
        _record_position(state, furn)
        room = _room_for_entity(world_graph, furn)
        if room:
            state.room_of[furn.name] = room
        states = furn.properties.get("states", {})
        if furn.properties.get("is_articulated", False):
            state.articulated.add(furn.name)
            is_open = states.get("is_open")
            if is_open is True:
                state.open.add(furn.name)
            elif is_open is False:
                state.known_closed.add(furn.name)
        components = furn.properties.get("components", [])
        if isinstance(components, list) and "faucet" in components:
            state.faucets.add(furn.name)
        _add_bool_states(state, furn.name, states)

    for obj in world_graph.get_all_objects():
        state.objects.add(obj.name)
        _record_position(state, obj)
        room = _room_for_entity(world_graph, obj)
        if room:
            state.room_of[obj.name] = room
        _add_bool_states(state, obj.name, obj.properties.get("states", {}))
        if obj.name == state.holding:
            continue
        furn = world_graph.find_furniture_for_object(obj)
        relation = world_graph.get_object_furniture_relation(obj)
        if furn is not None and relation is not None:
            state.placement[obj.name] = (furn.name, relation)
    return state


def _add_bool_states(state: SceneState, name: str, states: Dict) -> None:
    if "is_powered_on" in states:
        state.powerable.add(name)
    if states.get("is_powered_on", False):
        state.powered_on.add(name)
    if states.get("is_filled", False):
        state.filled.add(name)
    if states.get("is_clean", False):
        state.cleaned.add(name)


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------


def _navigate(s: SceneState, target: str) -> Action:
    return Action("navigate", (s.agent, target, s.room(target)))


def _open_if_closed(s: SceneState, furn: str) -> List[Action]:
    if furn not in s.articulated or furn in s.open:
        return []
    return [_navigate(s, furn), Action("open", (s.agent, furn, s.room(furn)))]


def _require_object(s: SceneState, obj: str) -> None:
    if obj not in s.objects:
        raise SubgoalPlanError(f"'{obj}' is not a known movable object")


def _require_furniture(s: SceneState, furn: str) -> None:
    if furn not in s.furniture:
        raise SubgoalPlanError(f"'{furn}' is not a known furniture")


def _nearest_faucet(s: SceneState, near: str) -> str:
    """Closest faucet furniture to ``near`` by position, else one in the same room."""
    if not s.faucets:
        raise SubgoalPlanError("no furniture with a faucet is known in the scene")
    origin = s.where(near)
    placed = sorted(f for f in s.faucets if f in s.position)
    if origin is not None and placed:
        return min(
            placed,
            key=lambda f: sum((a - b) ** 2 for a, b in zip(origin, s.position[f])),
        )
    room = s.room(s.agent if near == s.holding else near)
    same_room = sorted(f for f in s.faucets if s.room(f) == room != UNKNOWN_ROOM)
    return same_room[0] if same_room else sorted(s.faucets)[0]


def _free_hand(s: SceneState, keep: Optional[str], drop_at: Optional[str]) -> List[Action]:
    """Put down the held object unless it is ``keep``; placed on ``drop_at``."""
    if s.holding is None or s.holding == keep:
        return []
    if drop_at is None:
        raise SubgoalPlanError(
            f"agent is holding '{s.holding}' and there is no known furniture to put it on"
        )
    held = s.holding
    return [_navigate(s, drop_at), Action("place_on", (s.agent, held, drop_at, s.room(drop_at)))]


def _acquire(s: SceneState, obj: str, drop_at: Optional[str] = None) -> List[Action]:
    """Steps that end with ``obj`` in hand."""
    _require_object(s, obj)
    if s.holding == obj:
        return []
    src = s.placement.get(obj)
    steps = _free_hand(s, keep=obj, drop_at=drop_at or (src[0] if src else None))
    if src and src[1] == "inside":
        steps += _open_if_closed(s, src[0])
    return steps + [_navigate(s, obj), Action("pick", (s.agent, obj, s.room(obj)))]


# ---------------------------------------------------------------------------
# One expander per subgoal type. Each returns [] when already satisfied.
# ---------------------------------------------------------------------------


def expand_picked(s: SceneState, obj: str) -> List[Action]:
    return _acquire(s, obj)


def _expand_place(s: SceneState, obj: str, furn: str, relation: str) -> List[Action]:
    _require_object(s, obj)
    _require_furniture(s, furn)
    if s.holding != obj and s.placement.get(obj) == (furn, relation):
        return []
    if relation == "inside" and furn not in s.containers:
        raise SubgoalPlanError(f"'{furn}' has no receptacle to place objects inside")
    steps = []
    if relation == "inside":
        steps += _open_if_closed(s, furn)
    steps += _acquire(s, obj, drop_at=furn)
    action = "place_in" if relation == "inside" else "place_on"
    return steps + [_navigate(s, furn), Action(action, (s.agent, obj, furn, s.room(furn)))]


def expand_on(s: SceneState, obj: str, furn: str) -> List[Action]:
    return _expand_place(s, obj, furn, "on")


def _expand_into_object(s: SceneState, obj: str, box: str) -> List[Action]:
    """Put ``obj`` inside a movable container object; the place skill samples its interior."""
    _require_object(s, obj)
    if obj == box:
        raise SubgoalPlanError(f"'{obj}' cannot be placed inside itself")
    if s.holding != obj and s.contained_in.get(obj) == box:
        return []
    if s.holding == box:
        raise SubgoalPlanError(
            f"agent is holding '{box}'; place it on a furniture first, then put '{obj}' inside it"
        )
    support = s.placement.get(box)
    steps = _acquire(s, obj, drop_at=support[0] if support else None)
    return steps + [_navigate(s, box), Action("place_in", (s.agent, obj, box, s.room(box)))]


def expand_in(s: SceneState, obj: str, furn: str) -> List[Action]:
    if furn in s.object_containers:
        return _expand_into_object(s, obj, furn)
    return _expand_place(s, obj, furn, "inside")


def expand_open(s: SceneState, furn: str) -> List[Action]:
    _require_furniture(s, furn)
    if furn not in s.articulated:
        raise SubgoalPlanError(f"'{furn}' is not articulated and cannot be opened")
    if furn in s.open:
        return []
    return [_navigate(s, furn), Action("open", (s.agent, furn, s.room(furn)))]


def expand_closed(s: SceneState, furn: str) -> List[Action]:
    _require_furniture(s, furn)
    if furn not in s.articulated:
        raise SubgoalPlanError(f"'{furn}' is not articulated and cannot be closed")
    if furn in s.known_closed:
        return []
    return [_navigate(s, furn), Action("close", (s.agent, furn, s.room(furn)))]


def _expand_power(s: SceneState, target: str, on: bool) -> List[Action]:
    if target not in s.objects and target not in s.furniture:
        raise SubgoalPlanError(f"'{target}' is not a known object or furniture")
    if (target in s.powered_on) == on:
        return []
    name = "power_on" if on else "power_off"
    if target in s.objects:
        name += "_object"
    return [_navigate(s, target), Action(name, (s.agent, target, s.room(target)))]


def expand_powered_on(s: SceneState, target: str) -> List[Action]:
    return _expand_power(s, target, on=True)


def expand_powered_off(s: SceneState, target: str) -> List[Action]:
    return _expand_power(s, target, on=False)


def expand_waited_in(s: SceneState, obj: str, appliance: str) -> List[Action]:
    """Leave ``obj`` inside ``appliance`` (closed and powered on if it can be) and Wait.

    Satisfied once a Wait has run for this pair; take the object out afterwards
    with a separate picked()/on() subgoal.
    """
    _require_object(s, obj)
    _require_furniture(s, appliance)
    if (obj, appliance) in s.waited:
        return []
    inside = s.holding != obj and s.placement.get(obj) == (appliance, "inside")
    steps = [] if inside else _expand_place(s, obj, appliance, "inside")
    # Placing inside leaves an articulated appliance open.
    if appliance in s.articulated and (not inside or appliance not in s.known_closed):
        steps += [_navigate(s, appliance), Action("close", (s.agent, appliance, s.room(appliance)))]
    if appliance in s.powerable and appliance not in s.powered_on:
        steps += [_navigate(s, appliance), Action("power_on", (s.agent, appliance, s.room(appliance)))]
    return steps + [Action("wait", (s.agent, obj, appliance, s.room(appliance)))]


def _fill_held(s: SceneState, obj: str) -> List[Action]:
    faucet = _nearest_faucet(s, obj)
    return [_navigate(s, faucet), Action("fill_held", (s.agent, obj, faucet, s.room(faucet)))]


def expand_filled(s: SceneState, obj: str) -> List[Action]:
    _require_object(s, obj)
    if obj in s.filled:
        return []
    return _acquire(s, obj) + _fill_held(s, obj)


def expand_poured_into(s: SceneState, target: str) -> List[Action]:
    _require_object(s, target)
    if target in s.filled:
        return []
    src = s.holding
    if src is None or src == target:
        raise SubgoalPlanError(
            f"pouring into '{target}' needs a held source container; "
            f"pick up (and fill) a source object first"
        )
    steps = [] if src in s.filled else _fill_held(s, src)
    return steps + [_navigate(s, target), Action("pour", (s.agent, src, target, s.room(target)))]


def expand_cleaned(s: SceneState, target: str) -> List[Action]:
    if target in s.cleaned:
        return []
    if target in s.furniture:
        return [_navigate(s, target), Action("clean_furniture", (s.agent, target, s.room(target)))]
    _require_object(s, target)
    faucet = _nearest_faucet(s, target)
    return _acquire(s, target) + [
        _navigate(s, faucet),
        Action("clean_object", (s.agent, target, faucet, s.room(faucet))),
    ]


def expand_handempty(s: SceneState) -> List[Action]:
    if s.holding is None:
        return []
    furn_here = sorted(
        f for f in s.furniture if s.room(f) == s.room(s.agent) != UNKNOWN_ROOM
    )
    return _free_hand(s, keep=None, drop_at=furn_here[0] if furn_here else None)


def expand_at(s: SceneState, room: str) -> List[Action]:
    if s.room(s.agent) == room:
        return []
    furn_in_room = sorted(f for f in s.furniture if s.room(f) == room)
    if not furn_in_room:
        raise SubgoalPlanError(f"no known furniture in room '{room}' to navigate to")
    return [_navigate(s, furn_in_room[0])]


WAITED_IN_PREDS = ("waited_in", "waited-in", "waited_inside", "heated_in", "heated-in")

# Subgoal predicate (and accepted aliases) -> (expander, number of entity args).
_EXPANDERS: Dict[str, Tuple[Callable[..., List[Action]], int]] = {}


def _register(names, fn, arity):
    for n in names:
        _EXPANDERS[n] = (fn, arity)


_register(("picked", "pick", "holding"), expand_picked, 1)
_register(("on", "placed-on", "place-on"), expand_on, 2)
_register(("in", "inside", "within"), expand_in, 2)
_register(("opened-door", "opened-drawer", "open", "opened"), expand_open, 1)
_register(("closed-door", "closed-drawer", "close", "closed"), expand_closed, 1)
_register(("powered_on", "powered-on", "power_on", "turned-on", "on_state"), expand_powered_on, 1)
_register(("powered_off", "powered-off", "power_off", "turned-off"), expand_powered_off, 1)
_register(("filled", "fill", "is_filled"), expand_filled, 1)
_register(("poured_into", "poured-into", "pour"), expand_poured_into, 1)
_register(("cleaned", "clean", "is_clean"), expand_cleaned, 1)
_register(WAITED_IN_PREDS, expand_waited_in, 2)
_register(("handempty",), expand_handempty, 0)
_register(("at",), expand_at, 1)


def expand_subgoal(state: SceneState, pred: str, args: List[str]) -> List[Action]:
    """Return the action sequence for one parsed subgoal; raises SubgoalPlanError."""
    entry = _EXPANDERS.get(pred)
    if entry is None:
        raise SubgoalPlanError(f"unsupported subgoal predicate '{pred}'")
    fn, arity = entry
    if pred in ("picked", "pick", "holding", "at") and len(args) >= 2:
        args = args[1:]  # holding(agent, obj), at(agent, room)
    if len(args) < arity:
        raise SubgoalPlanError(f"'{pred}' expects {arity} argument(s), got {len(args)}")
    return fn(state, *(state.resolve(a) for a in args[:arity]))
