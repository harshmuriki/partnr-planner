#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Symbolic rearrangement domain for the Tru-POMDP planner.

This module must stay importable without habitat, habitat_sim or
habitat_llm.world_model so that the belief, search and TOH machinery can be unit
tested without a simulator, a GPU or network access. Only
habitat_llm/planner/tru_pomdp/planner.py may depend on the Habitat side.

The POMDP model follows Tru-POMDP (arXiv 2506.02860) section 3.2: a state is a
pair (scene graph, hypothesised placement goal), observations are the visible
part of the scene graph, transitions are deterministic, and infeasible actions
are mapped to NULL. Habitat/PARTNR adds Explore, in-place object-state skills,
next_to placement, and the stronger observation model documented on
``SymbolicDomain.observe``.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# Sentinel parent used for the object currently in the robot's gripper. The
# paper reassigns a picked object's parent to the ROBOT node; this is that node.
HELD = "__held__"

# ---------------------------------------------------------------------------
# Planning reward (Tru-POMDP section 3.2). These are the *planner's* internal
# reward scale and are deliberately unrelated to the PARTNR success predicates
# used by the Habitat evaluation metrics.
# ---------------------------------------------------------------------------
MANIPULATION_COST = 5.0
NAV_COST_MAX = 27.0
INFEASIBLE_COST = 100.0
SUBGOAL_REWARD = 200.0
COMPLETION_REWARD = 200.0

# Costs for the Habitat extensions, which the paper does not price.
# Explore is a multi-furniture tour of a whole room, so it costs more than a
# single manipulation but must stay far below the infeasible penalty, otherwise
# the planner would rather guess than look. PowerOn/PowerOff/Fill/Clean are
# in-place manipulations and are priced like Pick/Place/Open.
EXPLORE_COST = 10.0
OBJECT_STATE_COST = 5.0


class ActionType(str, Enum):
    """Parameterised operations on the scene graph."""

    NULL = "NULL"
    OPEN = "OPEN"
    PICK = "PICK"
    PLACE = "PLACE"
    # Habitat/PARTNR extensions.
    EXPLORE = "EXPLORE"
    POWER_ON = "POWER_ON"
    POWER_OFF = "POWER_OFF"
    FILL = "FILL"
    CLEAN = "CLEAN"


#: Object-state actions and the (flag, value) they establish.
STATE_ACTION_EFFECTS: Dict[ActionType, Tuple[str, bool]] = {
    ActionType.POWER_ON: ("is_powered_on", True),
    ActionType.POWER_OFF: ("is_powered_on", False),
    ActionType.FILL: ("is_filled", True),
    ActionType.CLEAN: ("is_clean", True),
}

#: Goal-atom state literals mapped onto the Habitat object-state flags.
STATE_LITERALS: Dict[str, Tuple[str, bool]] = {
    "is_powered_on": ("is_powered_on", True),
    "is_powered_off": ("is_powered_on", False),
    "is_filled": ("is_filled", True),
    "is_empty": ("is_filled", False),
    "is_clean": ("is_clean", True),
    "is_dirty": ("is_clean", False),
}

#: Which action establishes a given state literal.
STATE_LITERAL_ACTIONS: Dict[str, ActionType] = {
    "is_powered_on": ActionType.POWER_ON,
    "is_powered_off": ActionType.POWER_OFF,
    "is_filled": ActionType.FILL,
    "is_clean": ActionType.CLEAN,
}


@dataclass(frozen=True)
class Action:
    """
    A grounded high-level action.

    :param action_type: The kind of operation.
    :param area: Furniture id the action operates on. For EXPLORE this is a room
        id instead, because the PARTNR Explore skill takes a room name.
    :param obj: Object id for PICK and the object-state actions.
    :param relation: "on" or "within", used by PLACE.
    :param next_to: Anchor object for a next_to placement, used by PLACE.
    """

    action_type: ActionType
    area: Optional[str] = None
    obj: Optional[str] = None
    relation: str = "on"
    next_to: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.action_type.value]
        if self.obj is not None:
            parts.append(self.obj)
        if self.area is not None:
            parts.append(self.area)
        if self.action_type is ActionType.PLACE:
            parts.append(self.relation)
            if self.next_to is not None:
                parts.append(f"next_to={self.next_to}")
        return "(" + " ".join(parts) + ")"


NULL_ACTION = Action(ActionType.NULL)


@dataclass(frozen=True)
class GoalAtom:
    """
    One requirement of a hypothesised placement goal.

    ``target_area`` may be None: state-only goals carry no placement
    requirement, e.g. GoalAtom("lamp_0", states=("is_powered_off",)).
    """

    obj: str
    target_area: Optional[str] = None
    relation: str = "on"
    next_to: Optional[str] = None
    states: Tuple[str, ...] = ()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GoalAtom":
        states = data.get("states") or ()
        if isinstance(states, str):
            states = (states,)
        return GoalAtom(
            obj=data["object"],
            target_area=data.get("target_area") or None,
            relation=data.get("relation") or "on",
            next_to=data.get("next_to") or None,
            states=tuple(states),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": self.obj,
            "target_area": self.target_area,
            "relation": self.relation,
            "next_to": self.next_to,
            "states": list(self.states),
        }

    def rename(self, old: str, new: str) -> "GoalAtom":
        """Rename an object id, used when a hypothesised name gets grounded."""
        if self.obj != old and self.next_to != old:
            return self
        return GoalAtom(
            obj=new if self.obj == old else self.obj,
            target_area=self.target_area,
            relation=self.relation,
            next_to=new if self.next_to == old else self.next_to,
            states=self.states,
        )

    def __str__(self) -> str:
        out = self.obj
        if self.target_area is not None:
            out += f" {self.relation} {self.target_area}"
        if self.next_to is not None:
            out += f" next_to {self.next_to}"
        if self.states:
            out += " [" + ", ".join(self.states) + "]"
        return out


@dataclass
class SceneState:
    """
    The scene-graph half of a POMDP state.

    :param object_parent: object id -> furniture id, or HELD while carried.
        Hypothesised objects appear here with their hypothesised parent.
    :param furniture_room: furniture id -> room id.
    :param furniture_open: furniture id -> is_open. Furniture absent from this
        map is treated as permanently open (a surface, not a container).
    :param object_states: object id -> {"is_powered_on"/"is_filled"/"is_clean": bool}.
    :param spatial_relations: object id -> set of anchor objects it is next_to.
    :param robot_area: furniture id the robot is currently at, or None.
    :param inspected_areas: furniture ids whose contents the robot has looked at.
    :param hypothesized: object ids that came from the TOH and are not yet
        matched to a real Habitat object id.
    :param grounding: hypothesised name -> Habitat object id, recorded when a
        hypothesised object is grounded by an observation.
    :param previous_parent: object id -> the area it was picked from. Not in the
        paper's state; required by the Appendix A.2 rollout, whose
        "place it back in its parent" branch needs a real area for a held object.
    """

    object_parent: Dict[str, Optional[str]] = field(default_factory=dict)
    furniture_room: Dict[str, str] = field(default_factory=dict)
    furniture_open: Dict[str, bool] = field(default_factory=dict)
    object_states: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    spatial_relations: Dict[str, Set[str]] = field(default_factory=dict)
    robot_area: Optional[str] = None
    inspected_areas: Set[str] = field(default_factory=set)
    hypothesized: Set[str] = field(default_factory=set)
    grounding: Dict[str, str] = field(default_factory=dict)
    previous_parent: Dict[str, str] = field(default_factory=dict)
    observed_objects: Set[str] = field(default_factory=set)

    def copy(self) -> "SceneState":
        return SceneState(
            object_parent=dict(self.object_parent),
            furniture_room=self.furniture_room,
            furniture_open=dict(self.furniture_open),
            object_states={k: dict(v) for k, v in self.object_states.items()},
            spatial_relations={k: set(v) for k, v in self.spatial_relations.items()},
            robot_area=self.robot_area,
            inspected_areas=set(self.inspected_areas),
            hypothesized=set(self.hypothesized),
            grounding=dict(self.grounding),
            previous_parent=dict(self.previous_parent),
            observed_objects=set(self.observed_objects),
        )

    def held_object(self) -> Optional[str]:
        for obj, parent in self.object_parent.items():
            if parent == HELD:
                return obj
        return None

    def state_flag(self, obj: str, flag: str) -> Optional[bool]:
        return self.object_states.get(obj, {}).get(flag)

    def rename_object(self, old: str, new: str) -> None:
        """Replace an object id everywhere in the scene."""
        if old == new:
            return
        if old in self.object_parent:
            self.object_parent[new] = self.object_parent.pop(old)
        if old in self.object_states:
            self.object_states[new] = self.object_states.pop(old)
        if old in self.spatial_relations:
            self.spatial_relations[new] = self.spatial_relations.pop(old)
        for anchors in self.spatial_relations.values():
            if old in anchors:
                anchors.discard(old)
                anchors.add(new)
        if old in self.previous_parent:
            self.previous_parent[new] = self.previous_parent.pop(old)
        if old in self.hypothesized:
            self.hypothesized.discard(old)
        self.observed_objects.discard(old)
        self.observed_objects.add(new)
        self.grounding[old] = new


@dataclass
class Particle:
    """
    One POMDP state: a hypothesised scene graph plus a hypothesised complete goal.

    ``goal_atoms`` is the COMPLETE goal, including atoms that are already
    satisfied. Unsatisfied atoms are always derived from it, never removed from
    it, so a goal that is achieved and later disturbed is still tracked.
    """

    scene: SceneState
    goal_atoms: Tuple[GoalAtom, ...] = ()
    weight: float = 1.0

    def copy(self) -> "Particle":
        return Particle(
            scene=self.scene.copy(),
            goal_atoms=self.goal_atoms,
            weight=self.weight,
        )


_TOKEN_STOPWORDS = frozenset({"the", "a", "an", "of", "and"})


def normalize_object_name(name: str) -> Tuple[str, ...]:
    """
    Reduce an object id or a hypothesised name to comparable tokens.

    ``cereal_box_12`` and ``box of cereal`` both reduce to ("box", "cereal").
    Used to ground hypothesised TOH names onto real Habitat object ids.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.lower())
    tokens = [t for t in cleaned.split("_") if t and not t.isdigit()]
    return tuple(sorted(t for t in tokens if t not in _TOKEN_STOPWORDS))


def names_match(hypothesised: str, candidate: str) -> bool:
    """True if a hypothesised object name plausibly refers to a real object id."""
    a = set(normalize_object_name(hypothesised))
    b = set(normalize_object_name(candidate))
    if not a or not b:
        return False
    # Do not merge distinct numbered instances, or objects sharing only a
    # generic modifier (e.g. a water jug and a water bottle).
    h_id = re.match(r"^(.*)_(\d+)$", hypothesised)
    c_id = re.match(r"^(.*)_(\d+)$", candidate)
    if h_id and c_id and h_id.group(1) == c_id.group(1):
        return h_id.group(2) == c_id.group(2)
    return a <= b or b <= a


class SymbolicDomain:
    """
    Deterministic transition, observation, reward and feasibility model.

    The domain owns the static scene metadata (which furniture exists, which
    room it is in, which furniture is articulated, pairwise distances). Particles
    own everything that can differ between hypotheses.
    """

    def __init__(
        self,
        furniture_room: Optional[Dict[str, str]] = None,
        articulated: Optional[Iterable[str]] = None,
        within_capable: Optional[Iterable[str]] = None,
        distances: Optional[Dict[Tuple[str, str], float]] = None,
        graspable: Optional[Iterable[str]] = None,
        state_affordances: Optional[Dict[str, Iterable[str]]] = None,
        max_place_targets: int = 8,
        same_room_distance: float = 4.0,
        cross_room_distance: float = 12.0,
        distance_to_cost: float = 2.25,
        faucet_areas: Optional[Iterable[str]] = None,
        faucet_clean_objects: Optional[Iterable[str]] = None,
    ) -> None:
        """
        :param furniture_room: furniture id -> room id for every known furniture.
        :param articulated: furniture that can be opened and closed. Everything
            else is a surface whose contents are visible once inspected.
        :param within_capable: furniture that accepts a "within" placement.
            None means unrestricted.
        :param distances: optional pairwise navigation distances in metres.
        :param graspable: objects that can be picked. None means all.
        :param state_affordances: object id -> the object-state flags it
            supports. None means unrestricted.
        :param max_place_targets: cap on how many open areas the dynamic action
            space offers for a temporary placement, for tractability.
        :param distance_to_cost: metres -> planning cost, clamped at
            NAV_COST_MAX so navigation stays inside the paper's 0..27 band.
        """
        self.furniture_room: Dict[str, str] = dict(furniture_room or {})
        self.articulated: Set[str] = set(articulated or ())
        self.within_capable: Optional[Set[str]] = (
            None if within_capable is None else set(within_capable)
        )
        self.distances: Dict[Tuple[str, str], float] = dict(distances or {})
        self.graspable: Optional[Set[str]] = (
            None if graspable is None else set(graspable)
        )
        self.state_affordances: Optional[Dict[str, Set[str]]] = (
            None
            if state_affordances is None
            else {k: set(v) for k, v in state_affordances.items()}
        )
        self.faucet_areas = set(faucet_areas or ())
        self.faucet_clean_objects = set(faucet_clean_objects or ())
        self.max_place_targets = max_place_targets
        self.same_room_distance = same_room_distance
        self.cross_room_distance = cross_room_distance
        self.distance_to_cost = distance_to_cost

        self.room_furniture: Dict[str, List[str]] = {}
        for furniture, room in self.furniture_room.items():
            self.room_furniture.setdefault(room, []).append(furniture)
        for furniture_list in self.room_furniture.values():
            furniture_list.sort()

    # -- SceneGraphSimple accessors used by the Appendix A.2 rollout ---------

    def check_area_in_scene(self, scene: SceneState, area: Optional[str]) -> bool:
        if area is None:
            return False
        return area in self.furniture_room or area in scene.furniture_open

    def check_object_in_scene(self, scene: SceneState, obj: Optional[str]) -> bool:
        return obj is not None and obj in scene.object_parent

    def get_object_in_hand(self, scene: SceneState) -> Optional[str]:
        return scene.held_object()

    def get_object_parent(self, scene: SceneState, obj: str) -> Optional[str]:
        """The area containing the object; HELD while carried, None if unknown."""
        return scene.object_parent.get(obj)

    def get_area_open_from_id(self, scene: SceneState, area: Optional[str]) -> bool:
        """Open/closed status. Non-articulated furniture is always open."""
        if area is None:
            return False
        if area == HELD:
            return True
        return bool(scene.furniture_open.get(area, True))

    def room_of(self, area: Optional[str]) -> Optional[str]:
        if area is None:
            return None
        return self.furniture_room.get(area)

    # -- geometry ------------------------------------------------------------

    def distance(self, source: Optional[str], target: Optional[str]) -> float:
        if source is None or target is None or source == target:
            return 0.0
        for key in ((source, target), (target, source)):
            if key in self.distances:
                return self.distances[key]
        source_room = self.furniture_room.get(source)
        target_room = self.furniture_room.get(target)
        if source_room is not None and source_room == target_room:
            return self.same_room_distance
        return self.cross_room_distance

    def nav_cost(self, source: Optional[str], target: Optional[str]) -> float:
        """Navigation cost in the paper's 0..27 band."""
        return min(NAV_COST_MAX, self.distance(source, target) * self.distance_to_cost)

    def nav_target(self, scene: SceneState, action: Action) -> Optional[str]:
        """The area the robot has to stand at for the action, or None."""
        kind = action.action_type
        if kind in (ActionType.OPEN, ActionType.PLACE):
            return action.area
        if kind is ActionType.PICK:
            return action.area
        if kind is ActionType.EXPLORE:
            furniture = self.room_furniture.get(action.area or "", [])
            return furniture[-1] if furniture else None
        if kind in STATE_ACTION_EFFECTS:
            parent = scene.object_parent.get(action.obj or "")
            if parent in (None, HELD):
                return scene.robot_area
            return parent
        return None

    # -- visibility and the observation model --------------------------------

    def is_visible(self, scene: SceneState, obj: str) -> bool:
        """
        Whether the robot can currently see the object.

        A held object is always visible. Otherwise its containing area must be
        open and either the object itself observed or the area fully inspected.
        Knowing that a furniture node exists is not evidence of its contents.
        """
        parent = scene.object_parent.get(obj)
        if parent is None:
            return False
        if parent == HELD:
            return True
        if parent not in scene.inspected_areas and obj not in scene.observed_objects:
            return False
        return self.get_area_open_from_id(scene, parent)

    def visible_objects(self, scene: SceneState) -> List[str]:
        return sorted(obj for obj in scene.object_parent if self.is_visible(scene, obj))

    def observe(self, state: Particle, action: Action) -> Tuple:
        """
        Predicted observation key for a state, used for DESPOT's observation
        branching. Two particles produce the same key exactly when they are
        indistinguishable to the robot.

        The contract, identical here and in the Habitat adapter
        (see planner.TruPOMDPPlanner._build_observation):

        * closed containers hide their contents;
        * objects are revealed only in inspected areas;
        * a known furniture node does not mean its contents were inspected;
        * only observed object-state flags appear in the key.

        Robot pose, open/closed flags and the inspected set are included for
        completeness. They are functions of the action history and therefore
        identical across particles, so they add no spurious branching.
        """
        del action  # observations here depend only on the resulting state
        scene = state.scene
        visible = self.visible_objects(scene)
        placements = tuple((obj, scene.object_parent[obj]) for obj in visible)
        flags = tuple(
            (obj, flag, value)
            for obj in visible
            for flag, value in sorted(scene.object_states.get(obj, {}).items())
        )
        relations = tuple(
            (obj, anchor)
            for obj in visible
            for anchor in sorted(scene.spatial_relations.get(obj, ()))
        )
        return (
            scene.robot_area,
            placements,
            flags,
            relations,
            tuple(sorted(scene.inspected_areas)),
        )

    # -- goals ---------------------------------------------------------------

    def state_satisfied(self, scene: SceneState, obj: str, literal: str) -> bool:
        if literal not in STATE_LITERALS:
            return False
        flag, wanted = STATE_LITERALS[literal]
        current = scene.state_flag(obj, flag)
        if current is None:
            # Unknown is not evidence of a satisfied state goal.
            return False
        return current == wanted

    def placement_satisfied(self, scene: SceneState, atom: GoalAtom) -> bool:
        if atom.target_area is None:
            return True
        if scene.object_parent.get(atom.obj) != atom.target_area:
            return False
        if atom.next_to is not None:
            if atom.next_to not in scene.spatial_relations.get(atom.obj, set()):
                return False
        return True

    def atom_satisfied(self, scene: SceneState, atom: GoalAtom) -> bool:
        if atom.obj not in scene.object_parent:
            return False
        if not self.placement_satisfied(scene, atom):
            return False
        return all(self.state_satisfied(scene, atom.obj, s) for s in atom.states)

    def goal_satisfied(
        self,
        state: Union[SceneState, Particle],
        goal: Sequence[GoalAtom],
    ) -> bool:
        scene = state.scene if isinstance(state, Particle) else state
        return all(self.atom_satisfied(scene, atom) for atom in goal)

    def unsatisfied_atoms(
        self,
        state: Union[SceneState, Particle],
        goal: Sequence[GoalAtom],
    ) -> List[GoalAtom]:
        scene = state.scene if isinstance(state, Particle) else state
        return [atom for atom in goal if not self.atom_satisfied(scene, atom)]

    def _satisfied_count(self, scene: SceneState, goal: Sequence[GoalAtom]) -> int:
        return sum(1 for atom in goal if self.atom_satisfied(scene, atom))

    # -- feasibility ---------------------------------------------------------

    def is_graspable(self, obj: str) -> bool:
        return self.graspable is None or obj in self.graspable

    def supports_state(self, obj: str, flag: str) -> bool:
        """
        Whether the object can carry the flag. Objects with no declared
        affordances are unrestricted: PARTNR only annotates the objects whose
        states matter, and refusing everything else would make the planner blind
        to state goals it could actually achieve.
        """
        if self.state_affordances is None:
            return True
        affordances = self.state_affordances.get(obj)
        if affordances is None:
            return True
        return flag in affordances

    def feasible(self, scene: SceneState, action: Action) -> Tuple[bool, str]:
        """
        Precondition check mirroring the PARTNR oracle skills. Visibility alone
        never implies success: Pick also needs an empty gripper and a graspable
        object, Place needs an open target that accepts the relation, and a
        next_to placement needs the anchor to already be on the target.
        """
        kind = action.action_type
        if kind is ActionType.NULL:
            return True, ""
        held = scene.held_object()

        if kind is ActionType.OPEN:
            if not self.check_area_in_scene(scene, action.area):
                return False, f"unknown furniture {action.area}"
            if action.area not in self.articulated:
                return False, f"{action.area} is not articulated"
            if self.get_area_open_from_id(scene, action.area):
                return False, f"{action.area} is already open"
            return True, ""

        if kind is ActionType.PICK:
            if action.obj is None:
                return False, "Pick without an object"
            if held is not None:
                return False, f"already holding {held}"
            if scene.object_parent.get(action.obj) != action.area:
                return False, f"{action.obj} is not in {action.area}"
            if not self.is_visible(scene, action.obj):
                return False, f"{action.obj} has not been observed"
            if not self.is_graspable(action.obj):
                return False, f"{action.obj} is not graspable"
            return True, ""

        if kind is ActionType.PLACE:
            if held is None:
                return False, "Place with an empty gripper"
            if not self.check_area_in_scene(scene, action.area):
                return False, f"unknown furniture {action.area}"
            if not self.get_area_open_from_id(scene, action.area):
                return False, f"{action.area} is closed"
            if action.relation == "within" and self.within_capable is not None:
                if action.area not in self.within_capable:
                    return False, f"{action.area} has no within receptacle"
            if action.next_to is not None:
                if action.next_to == held:
                    return False, "cannot place an object next to itself"
                if scene.object_parent.get(action.next_to) != action.area:
                    return False, f"{action.next_to} is not on {action.area}"
            return True, ""

        if kind is ActionType.EXPLORE:
            if action.area not in self.room_furniture:
                return False, f"unknown room {action.area}"
            return True, ""

        if kind in STATE_ACTION_EFFECTS:
            flag, _ = STATE_ACTION_EFFECTS[kind]
            if action.obj is None:
                return False, f"{kind.value} without an object"
            if action.obj not in scene.object_parent:
                return False, f"unknown object {action.obj}"
            if not self.is_visible(scene, action.obj):
                return False, f"{action.obj} has not been observed"
            if not self.supports_state(action.obj, flag):
                return False, f"{action.obj} does not support {flag}"
            if (
                self.requires_faucet(action)
                and scene.object_parent.get(action.obj) not in self.faucet_areas
            ):
                return False, "object must be placed at a known faucet"
            return True, ""

        return False, f"unsupported action {kind}"

    def requires_faucet(self, action: Action) -> bool:
        return action.action_type is ActionType.FILL or (
            action.action_type is ActionType.CLEAN
            and action.obj in self.faucet_clean_objects
        )

    def faucet_preparation(self, scene: SceneState, obj: str) -> Optional[Action]:
        if not self.faucet_areas:
            return None
        parent = scene.object_parent.get(obj)
        if parent in self.faucet_areas:
            return None
        if parent == HELD:
            area = min(
                self.faucet_areas, key=lambda a: (self.distance(scene.robot_area, a), a)
            )
            if not self.get_area_open_from_id(scene, area):
                return Action(ActionType.OPEN, area=area)
            return Action(ActionType.PLACE, area=area)
        candidate = Action(ActionType.PICK, area=parent, obj=obj)
        return candidate if self.feasible(scene, candidate)[0] else None

    # -- transition ----------------------------------------------------------

    def apply_effects(self, scene: SceneState, action: Action) -> float:
        """
        Apply the deterministic effects of a feasible action in place and return
        the manipulation cost (excluding navigation). Exposed separately so the
        belief updater can reuse the exact same effects.
        """
        kind = action.action_type
        if kind is ActionType.NULL:
            return 0.0

        if kind is ActionType.OPEN:
            scene.furniture_open[action.area] = True
            # Standing in front of an opened container reveals its contents.
            scene.inspected_areas.add(action.area)
            return MANIPULATION_COST

        if kind is ActionType.PICK:
            obj = action.obj
            parent = scene.object_parent.get(obj)
            if parent is not None and parent != HELD:
                scene.previous_parent[obj] = parent
            scene.object_parent[obj] = HELD
            scene.spatial_relations.pop(obj, None)
            for anchors in scene.spatial_relations.values():
                anchors.discard(obj)
            return MANIPULATION_COST

        if kind is ActionType.PLACE:
            held = scene.held_object()
            scene.object_parent[held] = action.area
            if action.next_to is not None:
                scene.spatial_relations.setdefault(held, set()).add(action.next_to)
                scene.spatial_relations.setdefault(action.next_to, set()).add(held)
            else:
                scene.spatial_relations.pop(held, None)
            return MANIPULATION_COST

        if kind is ActionType.EXPLORE:
            # The skill tours the furniture of one room. It only reveals what it
            # actually covers: closed containers stay uninspected.
            for furniture in self.room_furniture.get(action.area or "", []):
                if self.get_area_open_from_id(scene, furniture):
                    scene.inspected_areas.add(furniture)
            return EXPLORE_COST

        if kind in STATE_ACTION_EFFECTS:
            flag, value = STATE_ACTION_EFFECTS[kind]
            scene.object_states.setdefault(action.obj, {})[flag] = value
            return OBJECT_STATE_COST

        return 0.0

    def step(self, state: Particle, action: Action) -> Tuple[Particle, float, bool]:
        """
        Deterministic transition with the paper's planning reward.

        :return: (next state, reward, terminal). Infeasible actions are mapped to
            NULL and charged INFEASIBLE_COST, as in the paper.
        """
        scene = state.scene
        goal = state.goal_atoms
        already_satisfied = self._satisfied_count(scene, goal)
        was_terminal = already_satisfied == len(goal)

        if action.action_type is ActionType.NULL:
            return state.copy(), 0.0, was_terminal

        ok, _reason = self.feasible(scene, action)
        if not ok:
            return state.copy(), -INFEASIBLE_COST, was_terminal

        nxt = state.copy()
        next_scene = nxt.scene
        reward = 0.0

        # Navigation is implicit: the skills move the robot when it is out of
        # range, and the cost depends on the distance travelled.
        target = self.nav_target(next_scene, action)
        if target is not None and target != next_scene.robot_area:
            reward -= self.nav_cost(next_scene.robot_area, target)
            next_scene.robot_area = target

        reward -= self.apply_effects(next_scene, action)

        now_satisfied = self._satisfied_count(next_scene, goal)
        if now_satisfied > already_satisfied:
            reward += SUBGOAL_REWARD * (now_satisfied - already_satisfied)
        terminal = now_satisfied == len(goal)
        if terminal and not was_terminal:
            reward += COMPLETION_REWARD
        return nxt, reward, terminal

    def optimistic_value(
        self, state: Particle, discount: float = 0.95, horizon: int = 20
    ) -> float:
        """Safe finite-horizon bound, including repeatedly disturbed subgoals."""
        if horizon <= 0 or self.goal_satisfied(state.scene, state.goal_atoms):
            return 0.0
        series = horizon if discount == 1 else (1 - discount**horizon) / (1 - discount)
        return SUBGOAL_REWARD * len(state.goal_atoms) * series + COMPLETION_REWARD

    # -- dynamic action space ------------------------------------------------

    def legal_actions(self, belief: Iterable[Particle]) -> List[Action]:
        """
        Build the dynamic action space from the belief (paper section 4.3).

        OPEN for every known closed container; PICK for every visible graspable
        hypothesised target; PLACE of the held object into valid open areas,
        including temporary placements rather than only goal areas; plus the
        Habitat extensions (Explore, object-state skills) when the corresponding
        atoms exist. The action list includes NULL only when nothing else is
        available; search also considers its zero-return default policy.
        """
        particles = list(belief)
        actions: List[Action] = []
        seen: Set[Action] = set()

        def add(action: Action) -> None:
            if action not in seen:
                seen.add(action)
                actions.append(action)

        held_objects: Set[str] = set()
        open_areas: Set[str] = set()
        goal_areas: List[str] = []

        for particle in particles:
            scene = particle.scene
            held = scene.held_object()
            if held is not None:
                held_objects.add(held)

            for furniture, is_open in scene.furniture_open.items():
                if is_open:
                    open_areas.add(furniture)
                elif furniture in self.articulated:
                    add(Action(ActionType.OPEN, area=furniture))
            for furniture in self.furniture_room:
                if furniture not in scene.furniture_open:
                    open_areas.add(furniture)

            for atom in particle.goal_atoms:
                if atom.target_area is not None:
                    goal_areas.append(atom.target_area)

                obj = atom.obj
                parent = scene.object_parent.get(obj)

                if held is None and not self.placement_satisfied(scene, atom):
                    candidate = Action(ActionType.PICK, area=parent, obj=obj)
                    if self.feasible(scene, candidate)[0]:
                        add(candidate)

                if held == obj and atom.target_area is not None:
                    add(
                        Action(
                            ActionType.PLACE,
                            area=atom.target_area,
                            relation=atom.relation,
                            next_to=atom.next_to,
                        )
                    )
                    if atom.next_to is not None:
                        # The anchor may not be there yet; allow the plain
                        # placement so the search is not stuck.
                        add(
                            Action(
                                ActionType.PLACE,
                                area=atom.target_area,
                                relation=atom.relation,
                            )
                        )

                # Habitat extension: the target is hypothesised in an area that
                # has not been inspected, so search the room it is in.
                if (
                    parent is not None
                    and parent != HELD
                    and parent not in scene.inspected_areas
                ):
                    room = self.room_of(parent)
                    if room in self.room_furniture:
                        add(Action(ActionType.EXPLORE, area=room))

                # Habitat extension: unmet state atoms.
                for literal in atom.states:
                    if self.state_satisfied(scene, obj, literal):
                        continue
                    action_type = STATE_LITERAL_ACTIONS.get(literal)
                    if action_type is None:
                        continue
                    candidate = Action(action_type, obj=obj)
                    if self.feasible(scene, candidate)[0]:
                        add(candidate)
                    elif self.requires_faucet(candidate):
                        preparation = self.faucet_preparation(scene, obj)
                        if preparation is not None:
                            add(preparation)

        if held_objects:
            # Temporary placements: any open area, goal areas first, capped for
            # tractability.
            goal_area_set = set(goal_areas)
            ranked = [a for a in sorted(goal_area_set) if a in open_areas]
            ranked += [a for a in sorted(open_areas) if a not in goal_area_set]
            for area in ranked[: self.max_place_targets]:
                add(Action(ActionType.PLACE, area=area))

        if not actions:
            add(NULL_ACTION)
        return actions
