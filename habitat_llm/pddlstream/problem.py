import os
from typing import Dict, Optional, Set, Tuple

from habitat_llm.world_model import Object, Furniture, Receptacle, Room, SpotRobot, Human
from habitat_llm.pddlstream import __init__  # noqa: F401
from habitat_llm.pddlstream.streams import make_stream_map

from pddlstream.language.constants import And, PDDLProblem
from pddlstream.utils import read

_DOMAIN_PATH = os.path.join(os.path.dirname(__file__), "pddl_domains", "partnr_rearrange_domain.pddl")
_STREAM_PATH = os.path.join(os.path.dirname(__file__), "pddl_domains", "partnr_rearrange_stream.pddl")


def _sanitize(name: str) -> str:
    return name.replace(" ", "_").replace(":", "_").replace("/", "_")


def _resolve_agent_node(world_graph, agent_uid: int):
    agent_name = f"agent_{agent_uid}"
    for node in world_graph.get_agents():
        if node.name == agent_name:
            return node
    agents = world_graph.get_agents()
    return agents[0] if agents else None


def _room_for_entity(world_graph, entity) -> Optional[Room]:
    rooms = world_graph.get_neighbors_of_type(entity, Room)
    if rooms:
        return rooms[0]
    if isinstance(entity, Object):
        furn = world_graph.find_furniture_for_object(entity)
        if furn is not None:
            rooms = world_graph.get_neighbors_of_type(furn, Room)
            if rooms:
                return rooms[0]
    if isinstance(entity, Receptacle):
        try:
            furn = world_graph.find_furniture_for_receptacle(entity)
            rooms = world_graph.get_neighbors_of_type(furn, Room)
            if rooms:
                return rooms[0]
        except Exception:
            return None
    if isinstance(entity, Furniture):
        rooms = world_graph.get_neighbors_of_type(entity, Room)
        if rooms:
            return rooms[0]
    return None


def _entity_type(entity):
    if isinstance(entity, Object):
        return "object"
    if isinstance(entity, Furniture):
        return "furniture"
    if isinstance(entity, Receptacle):
        return "receptacle"
    if isinstance(entity, Room):
        return "room"
    if isinstance(entity, (SpotRobot, Human)):
        return "agent"
    return None


_CONTAINER_KEYWORDS = {
    "cabinet",
    "fridge",
    "chest",
    "drawer",
    "counter",
    "washer",
    "dryer",
    "stand",
    "microwave",
    "oven",
    "dishwasher",
    "wardrobe",
    "shelves",
    "shelf",
    "box",
    "bag",
    "basket",
    "bin",
    "bucket",
    "hamper",
    "pot",
    "pan",
    "bowl",
}


def _is_container(furn_name: str) -> bool:
    """Return True for furniture that supports within-type placement (not surface tables/chairs)."""
    lower = furn_name.lower()
    return any(kw in lower for kw in _CONTAINER_KEYWORDS)


def _has_faucet(furn) -> bool:
    components = furn.properties.get("components", [])
    return isinstance(components, list) and "faucet" in components


def _obj_current_furniture(world_graph, obj):
    """Return the furniture entity that obj currently rests on/in, or None."""
    neighbors = world_graph.get_neighbors(obj)
    for neighbor, _edge in neighbors.items():
        if isinstance(neighbor, Furniture):
            return neighbor
        if isinstance(neighbor, Receptacle):
            try:
                return world_graph.find_furniture_for_receptacle(neighbor)
            except Exception:
                return None
    return None


def extract_scope_names(world_graph, agent_uid: int, goal_literal: tuple) -> Set[str]:
    """Return the minimal set of *original* entity names needed to plan for goal_literal.

    For each subgoal we only register:
      - The agent
      - Entities explicitly named in the goal (objects, furniture, rooms)
      - Their containing rooms (for inroom / at facts)
      - The current furniture of any object in the goal (for on/in init facts and pick preconditions)
      - For 'at' goals: one furniture in the target room (so navigate has a valid ?x)
    """
    scope: Set[str] = set()

    agent_node = _resolve_agent_node(world_graph, agent_uid)
    if agent_node:
        scope.add(agent_node.name)
        r = _room_for_entity(world_graph, agent_node)
        if r:
            scope.add(r.name)

    agent_pddl = _sanitize(agent_node.name) if agent_node else f"agent_{agent_uid}"

    # PDDL names explicitly in the goal (skip the predicate and agent)
    goal_pddl_names = {
        a for a in goal_literal[1:]
        if a != agent_pddl and not a.startswith("agent_")
    }

    # Build reverse lookup: pddl_name -> entity object
    by_pddl: Dict[str, object] = {}
    for e in (list(world_graph.get_all_objects()) +
              list(world_graph.get_all_furnitures()) +
              list(world_graph.get_all_receptacles()) +
              list(world_graph.get_all_rooms())):
        by_pddl[_sanitize(e.name)] = e

    involved_rooms: Set[str] = set()

    for pddl_name in goal_pddl_names:
        entity = by_pddl.get(pddl_name)
        if entity is None:
            continue
        scope.add(entity.name)

        room = _room_for_entity(world_graph, entity)
        if room:
            scope.add(room.name)
            involved_rooms.add(room.name)

        # For objects: add the furniture they currently rest on (pick precondition + on/in fact)
        if isinstance(entity, Object):
            cur_furn = _obj_current_furniture(world_graph, entity)
            if cur_furn:
                scope.add(cur_furn.name)
                r2 = _room_for_entity(world_graph, cur_furn)
                if r2:
                    scope.add(r2.name)
                    involved_rooms.add(r2.name)

    # For 'at(agent, room)' goals, include one furniture in the target room so the
    # navigate action has a valid ?x with (inroom ?x ?r).
    pred = goal_literal[0] if goal_literal else ""
    if pred == "at" and len(goal_literal) >= 3:
        target_room_pddl = goal_literal[2]
        target_room_entity = by_pddl.get(target_room_pddl)
        if target_room_entity and isinstance(target_room_entity, Room):
            for furn in world_graph.get_all_furnitures():
                r = _room_for_entity(world_graph, furn)
                if r and r.name == target_room_entity.name:
                    scope.add(furn.name)
                    scope.add(r.name)
                    break

    # For fill and clean_object goals, include at least one faucet furniture so the
    # navigate→fill/clean chain can be grounded. Try same room first; fall back globally.
    if pred in ("filled", "cleaned") and len(goal_literal) >= 2:
        target_obj = by_pddl.get(goal_literal[1])
        target_room = _room_for_entity(world_graph, target_obj) if target_obj else None
        if target_room is None and agent_node is not None:
            target_room = _room_for_entity(world_graph, agent_node)
        faucet_added = False
        # Pass 1: same room
        if target_room is not None:
            print("--------------------------------")
            print("Searching for faucet in the same room")
            print("--------------------------------")
            for furn in world_graph.get_all_furnitures():
                room = _room_for_entity(world_graph, furn)
                if room and room.name == target_room.name and _has_faucet(furn):
                    scope.add(furn.name)
                    scope.add(room.name)
                    faucet_added = True
                    print(f"Faucet found in the same room: {furn.name}")
                    break
        # Pass 2: any faucet globally
        if not faucet_added:
            print("--------------------------------")
            print("No faucet found in the same room, searching globally")
            print("--------------------------------")
            for furn in world_graph.get_all_furnitures():
                if _has_faucet(furn):
                    scope.add(furn.name)
                    room = _room_for_entity(world_graph, furn)
                    if room:
                        scope.add(room.name)
                    break

    return scope


def build_pddlstream_problem(
    world_graph,
    agent_uid: int,
    goal_literal: Tuple,
    scope_names: Optional[Set[str]] = None,
):
    """Build a PDDLStream problem from the WorldGraph.

    scope_names: if provided, only entities whose *original* names are in this
    set (plus the agent) are registered. This scopes the grounding to only the
    relevant objects/furniture/rooms for the current subgoal, preventing the
    grounding explosion that occurs with 200+ entities.
    """
    domain_pddl = read(_DOMAIN_PATH)
    stream_pddl = read(_STREAM_PATH)

    init = []
    name_to_entity: Dict[str, Dict] = {}
    pddl_name_map: Dict[str, str] = {}
    reverse_pddl_name_map: Dict[str, str] = {}

    def in_scope(original_name: str) -> bool:
        return scope_names is None or original_name in scope_names

    def register(name: str, typ: str):
        pddl_name = _sanitize(name)
        pddl_name_map[name] = pddl_name
        reverse_pddl_name_map[pddl_name] = name
        if pddl_name not in name_to_entity:
            name_to_entity[pddl_name] = {"type": typ, "types": {typ}, "name": name}
        else:
            name_to_entity[pddl_name]["types"].add(typ)
        if (typ, pddl_name) not in init:
            init.append((typ, pddl_name))
        return pddl_name

    # Rooms
    for room in world_graph.get_all_rooms():
        if in_scope(room.name):
            register(room.name, "room")

    # Furniture
    for furn in world_graph.get_all_furnitures():
        if in_scope(furn.name):
            pddl_furn = register(furn.name, "furniture")
            if furn.properties.get("is_articulated", False):
                register(furn.name, "joint")
            if _is_container(furn.name):
                if ("container", pddl_furn) not in init:
                    init.append(("container", pddl_furn))
            if _has_faucet(furn):
                if ("has_faucet", pddl_furn) not in init:
                    init.append(("has_faucet", pddl_furn))

    # Receptacles (skip — the domain uses furniture directly, not receptacles)

    # Objects
    for obj in world_graph.get_all_objects():
        if in_scope(obj.name):
            register(obj.name, "object")

    # Agent (always included)
    agent_node = _resolve_agent_node(world_graph, agent_uid)
    agent_name = register(agent_node.name if agent_node else f"agent_{agent_uid}", "agent")

    # Agent room
    if agent_node is not None:
        room = _room_for_entity(world_graph, agent_node)
        if room is not None and in_scope(room.name):
            init.append(("at", agent_name, register(room.name, "room")))

    # inroom facts — only for entities already registered
    registered_pddl = set(name_to_entity.keys())
    for entity in list(world_graph.graph.keys()):
        if isinstance(entity, Room):
            continue
        typ = _entity_type(entity)
        if typ is None:
            continue
        pddl_name = _sanitize(entity.name)
        if pddl_name not in registered_pddl:
            continue
        room = _room_for_entity(world_graph, entity)
        if room is None:
            continue
        room_pddl = _sanitize(room.name)
        if room_pddl not in registered_pddl:
            # register the room even if it wasn't in scope (navigation needs it)
            register(room.name, "room")
        init.append(("inroom", pddl_name, _sanitize(room.name)))

    # Holding / handempty
    held_name = None
    if agent_node is not None:
        held = agent_node.properties.get("last_held_object", None)
        if held is not None:
            held_name = held.name if hasattr(held, "name") else str(held)
    if held_name:
        init.append(("holding", agent_name, register(held_name, "object")))
    else:
        init.append(("handempty", agent_name))

    # On/In relations (only for registered objects)
    for obj in world_graph.get_all_objects():
        obj_pddl = _sanitize(obj.name)
        if obj_pddl not in registered_pddl:
            continue
        obj_name = register(obj.name, "object")
        if held_name and obj.name == held_name:
            continue
        neighbors = world_graph.get_neighbors(obj)
        for neighbor, edge in neighbors.items():
            if isinstance(neighbor, Furniture):
                furn_pddl = _sanitize(neighbor.name)
                if furn_pddl not in registered_pddl:
                    register(neighbor.name, "furniture")
                furn_name = _sanitize(neighbor.name)
                if edge in ("inside", "in", "within"):
                    init.append(("in", obj_name, furn_name))
                else:
                    init.append(("on", obj_name, furn_name))
                break
            if isinstance(neighbor, Receptacle):
                try:
                    furn = world_graph.find_furniture_for_receptacle(neighbor)
                    if furn is not None:
                        furn_pddl = _sanitize(furn.name)
                        if furn_pddl not in registered_pddl:
                            register(furn.name, "furniture")
                        furn_name = _sanitize(furn.name)
                        if edge in ("inside", "in", "within"):
                            init.append(("in", obj_name, furn_name))
                        else:
                            init.append(("on", obj_name, furn_name))
                        break
                except Exception:
                    continue

    # Articulated joint states (only registered joints)
    for furn in world_graph.get_all_furnitures():
        if not furn.properties.get("is_articulated", False):
            continue
        furn_pddl = _sanitize(furn.name)
        if furn_pddl in registered_pddl:
            init.append(("closed", register(furn.name, "joint")))

    # Pre-certify reachable(agent, entity) for all registered non-agent entities.
    # Oracle skills are always reachable — no motion planning needed.
    for pddl_name, meta in list(name_to_entity.items()):
        if "agent" not in meta.get("types", {meta.get("type", "")}):
            init.append(("reachable", agent_name, pddl_name))

    # Object states: powered_on / powered_off, filled
    for obj in world_graph.get_all_objects():
        obj_pddl = _sanitize(obj.name)
        if obj_pddl not in registered_pddl:
            continue
        obj_name = register(obj.name, "object")
        states = obj.properties.get("states", {})
        is_powered_on = states.get("is_powered_on", False)
        if is_powered_on:
            init.append(("powered_on", obj_name))
        else:
            init.append(("powered_off", obj_name))
        if states.get("is_filled", False):
            init.append(("filled", obj_name))
        if states.get("is_clean", False):
            init.append(("cleaned", obj_name))

    # Furniture power states: powered_on / powered_off, cleaned
    for furn in world_graph.get_all_furnitures():
        furn_pddl = _sanitize(furn.name)
        if furn_pddl not in registered_pddl:
            continue
        furn_name = register(furn.name, "furniture")
        states = furn.properties.get("states", {})
        is_powered_on = states.get("is_powered_on", False)
        if is_powered_on:
            init.append(("powered_on", furn_name))
        else:
            init.append(("powered_off", furn_name))
        if states.get("is_clean", False):
            init.append(("cleaned", furn_name))

    stream_map = make_stream_map(name_to_entity)
    goal = And(goal_literal)

    problem = PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)
    return problem, init, pddl_name_map, reverse_pddl_name_map
