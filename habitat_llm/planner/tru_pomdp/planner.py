#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
PARTNR lifecycle for the Tru-POMDP planner.

This is the only module of habitat_llm/planner/tru_pomdp that depends on the
Habitat side. It translates the agent's WorldGraph into the symbolic
``Observation``, runs TOH -> DESPOT -> Habitat skill, holds the skill until
``process_high_level_actions`` returns a non-empty response, and then performs
the hybrid belief update.
"""

import time
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
from hydra.utils import instantiate

from habitat_llm.planner.planner import Planner
from habitat_llm.planner.tru_pomdp.belief import (
    Belief,
    ExecutionOutcome,
    HybridBeliefUpdater,
    Observation,
)
from habitat_llm.planner.tru_pomdp.scene import (
    HELD,
    STATE_ACTION_EFFECTS,
    Action,
    ActionType,
    Particle,
    SceneState,
    SymbolicDomain,
)
from habitat_llm.planner.tru_pomdp.search import DespotConfig, DespotSolver
from habitat_llm.planner.tru_pomdp.toh import TohConfig, TreeOfHypotheses

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph

#: Habitat skill names for the symbolic actions.
SKILL_NAMES: Dict[ActionType, str] = {
    ActionType.OPEN: "Open",
    ActionType.PICK: "Pick",
    ActionType.PLACE: "Place",
    ActionType.EXPLORE: "Explore",
    ActionType.POWER_ON: "PowerOn",
    ActionType.POWER_OFF: "PowerOff",
    ActionType.FILL: "Fill",
    ActionType.CLEAN: "Clean",
    ActionType.NULL: "Wait",
}

#: Object-state flags the planner tracks. PARTNR stores these on the node's
#: "states" property.
TRACKED_STATE_FLAGS = ("is_powered_on", "is_filled", "is_clean")

#: Words that mark a skill response as a failure.
_FAILURE_MARKERS = (
    "unexpected failure",
    "failed",
    "failure",
    "could not",
    "cannot",
    "unable",
    "not possible",
    "invalid",
    "no valid",
    "does not exist",
    "timeout",
)


class TruPOMDPPlanner(Planner):
    """Tru-POMDP: Tree of Hypotheses + hybrid belief update + DESPOT."""

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ) -> None:
        super().__init__(plan_config, env_interface)
        self.llm: Any = None
        self._initialize_llm()

        self.despot_config = DespotConfig(
            num_scenarios=int(plan_config.get("num_scenarios", 30)),
            max_search_depth=int(plan_config.get("max_search_depth", 20)),
            rollout_depth=int(plan_config.get("rollout_depth", 10)),
            discount=float(plan_config.get("discount", 0.95)),
            xi=float(plan_config.get("xi", 0.95)),
            num_trials=int(plan_config.get("num_trials", 500)),
            planning_time_s=float(plan_config.get("planning_time_s", 1.0)),
            seed=int(plan_config.get("seed", 0)),
        )
        self.toh_config = TohConfig(
            c1=int(plan_config.get("c1", 3)),
            c2=int(plan_config.get("c2", 3)),
            max_objects_per_combination=plan_config.get("max_goal_objects"),
            max_particles=int(plan_config.get("max_particles", 48)),
            max_tokens=int(plan_config.get("toh_max_tokens", 4096)),
            seed=int(plan_config.get("seed", 0)),
        )
        self.replenish_threshold = float(plan_config.get("replenish_threshold", 0.3))
        self.max_decisions = int(plan_config.get("max_decisions", 50))
        self.max_place_targets = int(plan_config.get("max_place_targets", 8))
        # A deliberately inspected area can refute a hypothesis. Areas that merely
        # happen to contain an observed object only confirm; set this to True if
        # the world model is trusted to be complete for open furniture.
        self.trust_observed_areas = bool(plan_config.get("trust_observed_areas", False))
        self.navigation_threshold = float(plan_config.get("navigation_threshold", 1.5))
        self.verbose = bool(plan_config.get("verbose", False))

        self.domain: Optional[SymbolicDomain] = None
        self.toh: Optional[TreeOfHypotheses] = None
        self.updater: Optional[HybridBeliefUpdater] = None
        self.solver: Optional[DespotSolver] = None
        self.belief: Optional[Belief] = None

        self.reset()

    # -- setup ---------------------------------------------------------------

    def _initialize_llm(self) -> None:
        """Instantiate the LLM exactly the way LLMPlanner does."""
        llm_conf = self.planner_config.llm
        self.llm = instantiate(llm_conf.llm)
        self.llm = self.llm(llm_conf)

    def reset(self) -> None:
        """Reset the planner state between episodes."""
        tracker = getattr(getattr(self, "llm", None), "token_usage", None)
        if tracker is not None:
            tracker.reset()
        hook = getattr(self, "_perception_hook", None)
        if hook is not None:
            perception, original, callback = hook
            if perception.get_recent_subgraph is callback:
                perception.get_recent_subgraph = original
        self._perception_hook = None
        for agent in self._agents:
            agent.reset()
        self.is_done = False
        self.last_high_level_actions = {}
        self._issued_high_level_actions: Dict[int, Tuple[str, str, Any]] = {}

        self.domain = None
        self.toh = None
        self.updater = None
        self.solver = None
        self.belief = None

        self._observed_locations = {}
        self._observed_flags = {}
        self._observed_relations = {}
        self._memory = {}
        self._memory_initialized = False
        self._history = []
        self._fresh_handles = None
        self._fresh_graph = None
        self._failed_contexts = set()
        self._action_context = None
        self._termination_reason = None
        self._memory_revisions = 0
        self._last_belief_update = {}
        self._instruction = ""
        self._queue: List[Tuple[str, str]] = []
        self._symbolic_action: Optional[Action] = None
        self._navigated = False
        self._deliberately_inspected: Set[str] = set()
        self._furniture_open_from_actions: Dict[str, bool] = {}
        self._failed_attempts: List[str] = []
        self._wrong_goal_states: List[Tuple[Any, ...]] = []
        self._num_decisions = 0
        self._trace = ""
        self._search_time_s = 0.0
        self._llm_time_s = 0.0
        self._last_search_stats: Dict[str, Any] = {}

    @property
    def _agent_uid(self) -> int:
        indices = self.agent_indices
        if not indices:
            raise ValueError("TruPOMDPPlanner has no agents assigned")
        return indices[0]

    # -- WorldGraph -> symbolic domain / observation --------------------------

    def _sync_domain(self, world_graph: "WorldGraph") -> None:
        """(Re)build the static domain metadata from the world graph."""
        furniture_nodes = world_graph.get_all_furnitures()
        furniture_room: Dict[str, str] = {}
        for furniture in furniture_nodes:
            try:
                room = world_graph.get_room_for_entity(furniture)
            except ValueError:
                continue
            furniture_room[furniture.name] = room.name

        articulated: Set[str] = set()
        for furniture in furniture_nodes:
            states = furniture.properties.get("states", {}) or {}
            if furniture.is_articulated() or "is_open" in states:
                articulated.add(furniture.name)

        if (
            self.domain is not None
            and set(self.domain.furniture_room) == set(furniture_room)
            and self.domain.articulated == articulated
        ):
            self.domain.faucet_clean_objects = self._faucet_clean_objects(world_graph)
            self.domain.state_affordances = self._state_affordances(world_graph)
            return

        distances = self._furniture_distances(world_graph, furniture_room)
        self.domain = SymbolicDomain(
            furniture_room=furniture_room,
            articulated=articulated,
            # PARTNR furniture may expose a "within" receptacle without being
            # articulated, so "within" placements are left unrestricted here and
            # the real skill decides.
            within_capable=None,
            distances=distances,
            state_affordances=self._state_affordances(world_graph),
            max_place_targets=self.max_place_targets,
            faucet_areas=[
                f.name
                for f in furniture_nodes
                if "faucet" in f.properties.get("components", [])
            ],
            faucet_clean_objects=self._faucet_clean_objects(world_graph),
        )
        self.toh = TreeOfHypotheses(
            self.llm, self.domain, self.toh_config, verbose=self.verbose
        )
        self.updater = HybridBeliefUpdater(
            self.domain, self.toh, self.replenish_threshold
        )
        self.solver = DespotSolver(self.domain, self.despot_config)

    def _furniture_distances(
        self, world_graph: "WorldGraph", furniture_room: Dict[str, str]
    ) -> Dict[Tuple[str, str], float]:
        """Pairwise navigation distances, approximated by Euclidean distance."""
        positions: Dict[str, Any] = {}
        for name in furniture_room:
            try:
                node = world_graph.get_node_from_name(name)
                translation = node.get_property("translation")
            except (ValueError, KeyError):
                continue
            if translation is None:
                continue
            positions[name] = np.asarray(list(translation), dtype=float)
        distances: Dict[Tuple[str, str], float] = {}
        names = sorted(positions)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                distances[(first, second)] = float(
                    np.linalg.norm(positions[first] - positions[second])
                )
        return distances

    def _state_affordances(self, world_graph: "WorldGraph") -> Dict[str, Set[str]]:
        affordances: Dict[str, Set[str]] = {}
        for obj in world_graph.get_all_objects():
            states = obj.properties.get("states", {}) or {}
            declared = {flag for flag in states if flag in TRACKED_STATE_FLAGS}
            if declared:
                affordances[obj.name] = declared
        return affordances

    def _faucet_clean_objects(self, graph):
        metadata = getattr(
            getattr(self.env_interface, "perception", None), "metadata_interface", None
        )
        classes = getattr(metadata, "affordance_info", {}).get(
            "cleaned under a faucet if dirty", []
        )
        return {
            obj.name
            for obj in graph.get_all_objects()
            if obj.properties.get("type") in classes
        }

    def _remember_sensor_graph(self, subgraph):
        """Capture transient sensor results, including one-shot Open detections."""
        from habitat_llm.world_model.world_graph import WorldGraph

        graph = WorldGraph(subgraph.graph)
        for obj in graph.get_all_objects():
            if str(obj.sim_handle).endswith(".stale_memory"):
                continue
            if graph.is_object_with_agent(obj, agent_type="robot"):
                parent = HELD
            else:
                furniture = graph.find_furniture_for_object(obj)
                if furniture is None:
                    continue
                parent = furniture.name
            old_parent = self._observed_locations.get(obj.name)
            if old_parent is not None and old_parent != parent:
                self._observed_relations.pop(obj.name, None)
                for anchors in self._observed_relations.values():
                    anchors.discard(obj.name)
            self._observed_locations[obj.name] = parent
            self._observed_flags.setdefault(obj.name, {}).update(
                {
                    k: bool(v)
                    for k, v in obj.properties.get("states", {}).items()
                    if k in TRACKED_STATE_FLAGS
                }
            )

    def _observe_perception_calls(self, perception):
        if self._perception_hook is not None and self._perception_hook[0] is perception:
            return
        original = perception.get_recent_subgraph

        def capture(agent_uids, obs):
            graph = original(agent_uids, obs)
            if str(self._agent_uid) in {str(uid) for uid in agent_uids}:
                self._remember_sensor_graph(graph)
            return graph

        # Scoped to this baseline's environment instance, restored on reset.
        # Capture before the shared runner merges away observation provenance.
        perception.get_recent_subgraph = capture
        self._perception_hook = (perception, original, capture)

    def _refresh_perception(self):
        """Read the robot's sensor-derived subgraph, never the privileged world graph."""
        if self.env_interface is None:
            return  # Explicit graph-only adapter used by unit tests.
        self._fresh_handles = set()
        self._fresh_graph = None
        try:
            perception = self.env_interface.perception
            self._observe_perception_calls(perception)
            obs = self.env_interface.env.habitat_env.sim.get_sensor_observations()
            from habitat_llm.world_model.world_graph import WorldGraph

            subgraph = perception.get_recent_subgraph([str(self._agent_uid)], obs)
            graph = WorldGraph(subgraph.graph)
            self._fresh_graph = graph
            self._fresh_handles = {obj.sim_handle for obj in graph.get_all_objects()}
        except (AttributeError, ValueError, KeyError) as exc:
            self._trace += f"Perception unavailable: {type(exc).__name__}: {exc}\n"

    def _initialize_memory(self, graph):
        if self._memory_initialized:
            return
        self._memory_initialized = True
        if self.env_interface is None:
            return
        # The shared environment has already loaded the episode's memory and aliases.
        # Copy only its remembered claims; do not read truth labels or object placements.
        from habitat_llm.utils.initial_robot_memory import (
            load_remembered_object_records,
        )

        data_path = self.env_interface.conf.habitat.dataset.data_path
        records = load_remembered_object_records(data_path)
        from habitat_llm.utils.initial_robot_memory import (
            load_scene_info,
            resolve_placement_node,
        )

        scene_info = load_scene_info(data_path)
        objects = graph.get_all_objects()
        for record in records:
            if record.outdated_location:
                placement = resolve_placement_node(
                    graph, record.outdated_location, scene_info
                )
                if placement is not None:
                    self._memory[record.entity] = placement.name
            else:
                obj = next(
                    (
                        o
                        for o in objects
                        if o.sim_handle == record.sim_handle and record.sim_handle
                    ),
                    None,
                )
                if obj is None:
                    obj = next((o for o in objects if o.name == record.entity), None)
                parent = (
                    graph.find_furniture_for_object(obj) if obj is not None else None
                )
                if parent is not None:
                    self._memory[obj.name] = parent.name

    @staticmethod
    def _observation_signature(observation):
        return json.dumps(
            {
                "parents": observation.object_parent,
                "robot_area": observation.robot_area,
                "states": observation.object_states,
                "open": observation.furniture_open,
                "relations": {
                    k: sorted(v) for k, v in observation.spatial_relations.items()
                },
                "inspected": sorted(observation.fully_inspected_areas),
            },
            sort_keys=True,
        )

    def _build_observation(self, world_graph: "WorldGraph") -> Observation:
        """
        Translate the world graph into the symbolic observation.

        Keep individual positive detections separate from exhaustive inspection.
        Known furniture and remembered objects do not establish visibility.
        """
        assert self.domain is not None
        self._initialize_memory(world_graph)
        furniture_nodes = world_graph.get_all_furnitures()
        known_furniture = {furniture.name for furniture in furniture_nodes}

        # PerceptionSim recomputes is_open from joint positions, but only on the
        # privileged gt_graph: WorldGraph.merge copies nothing but "translation"
        # onto nodes the agent graph already has, so the agent graph's is_open is
        # frozen at its episode-initial value. What the robot legitimately knows
        # beyond that is the outcome of its own Open/Close skills, so those are
        # layered on top.
        furniture_open: Dict[str, bool] = {}
        for furniture in furniture_nodes:
            states = furniture.properties.get("states", {}) or {}
            if "is_open" in states:
                furniture_open[furniture.name] = bool(states["is_open"])
            elif furniture.name in self.domain.articulated:
                furniture_open[furniture.name] = bool(
                    furniture.properties.get("is_open", False)
                )
        for name, is_open in self._furniture_open_from_actions.items():
            if name in known_furniture:
                furniture_open[name] = is_open

        object_parent: Dict[str, str] = {}
        object_states: Dict[str, Dict[str, bool]] = {}
        observed_graph = (
            self._fresh_graph if self._fresh_graph is not None else world_graph
        )
        for obj in observed_graph.get_all_objects():
            if str(obj.sim_handle).endswith(".stale_memory"):
                continue
            if (
                self._fresh_handles is not None
                and obj.sim_handle not in self._fresh_handles
            ):
                continue
            try:
                held = observed_graph.is_object_with_agent(obj, agent_type="robot")
            except (ValueError, KeyError):
                held = False
            if held:
                object_parent[obj.name] = HELD
            else:
                furniture = observed_graph.find_furniture_for_object(obj)
                if furniture is None:
                    continue
                object_parent[obj.name] = furniture.name
            states = obj.properties.get("states", {}) or {}
            flags = {
                flag: bool(value)
                for flag, value in states.items()
                if flag in TRACKED_STATE_FLAGS
            }
            if flags:
                object_states[obj.name] = flags

        occupied = {parent for parent in object_parent.values() if parent != HELD}
        inspected = set(self._deliberately_inspected) | occupied
        fully_inspected = set(self._deliberately_inspected)
        if self.trust_observed_areas:
            fully_inspected |= occupied

        relations = {}
        for obj in observed_graph.get_all_objects():
            if obj.name not in object_parent:
                continue
            anchors = {
                node.name
                for node, relation in observed_graph.get_neighbors(obj).items()
                if relation in ("next_to", "next to") and node.name in object_parent
            }
            if anchors:
                relations[obj.name] = anchors
        # Positive sensor evidence persists in this static-world model; current
        # FOV absence alone never deletes it. Skill transitions update it below.
        self._observed_locations.update(object_parent)
        for obj, flags in object_states.items():
            self._observed_flags.setdefault(obj, {}).update(flags)
        self._observed_relations.update(relations)
        for name, parent in list(self._memory.items()):
            if name in self._observed_locations or (
                parent in fully_inspected and furniture_open.get(parent, True)
            ):
                self._memory.pop(name)
                self._memory_revisions += 1
        object_parent = dict(self._observed_locations)
        object_states = {k: dict(v) for k, v in self._observed_flags.items()}
        relations = {k: set(v) for k, v in self._observed_relations.items()}
        inspected |= {p for p in object_parent.values() if p != HELD}

        return Observation(
            object_parent=object_parent,
            furniture_open=furniture_open,
            object_states=object_states,
            spatial_relations=relations,
            robot_area=self._robot_area(world_graph),
            inspected_areas=inspected,
            fully_inspected_areas=fully_inspected,
            known_furniture=known_furniture,
        )

    def _robot_area(self, world_graph: "WorldGraph") -> Optional[str]:
        """The furniture the robot is standing closest to, if it is in range."""
        assert self.domain is not None
        try:
            robot = world_graph.get_spot_robot()
            position = np.asarray(list(robot.get_property("translation")), dtype=float)
        except (ValueError, KeyError):
            return None
        best: Optional[str] = None
        best_distance = float("inf")
        for name in self.domain.furniture_room:
            try:
                node = world_graph.get_node_from_name(name)
                translation = node.get_property("translation")
            except (ValueError, KeyError):
                continue
            distance = float(
                np.linalg.norm(np.asarray(list(translation), dtype=float) - position)
            )
            if distance < best_distance:
                best_distance = distance
                best = name
        if best is None or best_distance > self.navigation_threshold:
            return None
        return best

    # -- symbolic action -> Habitat skills ------------------------------------

    def _translate(
        self, action: Action, world_graph: "WorldGraph", scene: SceneState
    ) -> List[Tuple[str, str]]:
        """
        Expand a symbolic action into the Habitat skill calls it needs.

        An implicit Navigate is prepended only when the oracle skill would be out
        of range, so a manipulation at the current station costs one skill call.
        """
        kind = action.action_type
        if kind is ActionType.NULL:
            return [("Wait", "")]
        if kind is ActionType.EXPLORE:
            return [("Explore", str(action.area))]

        queue: List[Tuple[str, str]] = []
        nav_target: Optional[str] = None
        if kind in (ActionType.OPEN, ActionType.PICK, ActionType.PLACE):
            nav_target = action.area
        elif kind in STATE_ACTION_EFFECTS:
            parent = scene.object_parent.get(action.obj or "")
            nav_target = None if parent in (None, HELD) else parent
        if nav_target is not None and self._needs_navigation(world_graph, nav_target):
            queue.append(("Navigate", nav_target))

        if kind is ActionType.OPEN:
            queue.append(("Open", str(action.area)))
        elif kind is ActionType.PICK:
            queue.append(("Pick", str(action.obj)))
        elif kind is ActionType.PLACE:
            held = scene.held_object()
            constraint = "next_to" if action.next_to else "None"
            reference = action.next_to or "None"
            queue.append(
                (
                    "Place",
                    f"{held}, {action.relation}, {action.area}, "
                    f"{constraint}, {reference}",
                )
            )
        elif kind in STATE_ACTION_EFFECTS:
            queue.append((SKILL_NAMES[kind], str(action.obj)))
        return queue

    def _needs_navigation(self, world_graph: "WorldGraph", target: str) -> bool:
        try:
            robot = world_graph.get_spot_robot()
            node = world_graph.get_node_from_name(target)
            position = np.asarray(list(robot.get_property("translation")), dtype=float)
            other = np.asarray(list(node.get_property("translation")), dtype=float)
            return float(np.linalg.norm(position - other)) > self.navigation_threshold
        except (ValueError, KeyError, TypeError):
            return True

    # -- PARTNR lifecycle ----------------------------------------------------

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Give the next low-level action.

        One high-level skill runs at a time. While it is running the same skill is
        re-issued unchanged; when ``process_high_level_actions`` returns a
        non-empty response the skill has finished and the hybrid belief update
        runs on the next call.
        """
        agent_uid = self._agent_uid
        self._instruction = instruction or self._instruction
        graph = world_graph[agent_uid]

        if self.is_done:
            self._issued_high_level_actions = {}
            return {}, self._planner_info({}, replanned=False), True

        self._sync_domain(graph)
        self._refresh_perception()
        self._build_observation(graph)

        # A skill is in flight: hold it until it reports back.
        if self.last_high_level_actions:
            self._issued_high_level_actions = dict(self.last_high_level_actions)
            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )
            response = responses.get(agent_uid, "")
            if response:
                self._on_skill_finished(response, graph)
            return (
                low_level_actions,
                self._planner_info(responses, replanned=False),
                self.is_done,
            )

        # Nothing in flight: continue the current symbolic action or plan a new one.
        replanned = False
        if not self._queue:
            replanned = True
            if not self._plan_next_symbolic_action(graph):
                self.is_done = True
                self._issued_high_level_actions = {}
                # No skill was assigned, so this is not a replanning step as far
                # as the evaluation runner's action history is concerned: it
                # asserts that a replanned step carries a high-level action.
                return {}, self._planner_info({}, replanned=False), True

        skill_name, skill_args = self._queue.pop(0)
        self.last_high_level_actions = {agent_uid: (skill_name, skill_args, None)}
        self._issued_high_level_actions = dict(self.last_high_level_actions)
        self._trace += f"Action: {skill_name}[{skill_args}]\n"
        low_level_actions, responses = self.process_high_level_actions(
            self.last_high_level_actions, observations
        )
        response = responses.get(agent_uid, "")
        if response:
            self._on_skill_finished(response, graph)
        return (
            low_level_actions,
            self._planner_info(responses, replanned=replanned),
            self.is_done,
        )

    def _plan_next_symbolic_action(self, world_graph: "WorldGraph") -> bool:
        """
        Run TOH if needed and then DESPOT, and queue the Habitat skills.

        :return: False when the planner should stop.
        """
        assert self.domain is not None and self.solver is not None
        if self._num_decisions >= self.max_decisions:
            self._termination_reason = "budget_exhaustion"
            return False

        observation = self._build_observation(world_graph)
        if not self._ensure_belief(observation):
            return False

        assert self.belief is not None
        particle = self.belief.map_particle()
        if all(self.domain.goal_satisfied(p.scene, p.goal_atoms) for p in self.belief):
            self._termination_reason = "belief_complete"
            return False

        start = time.time()
        context = self._observation_signature(observation)
        excluded = {
            action
            for action, signature in self._failed_contexts
            if signature == context
        }
        action, stats = self.solver.plan(self.belief, excluded_actions=excluded)
        self._action_context = context
        self._search_time_s += time.time() - start
        self._last_search_stats = {
            "trials": stats.trials,
            "nodes": stats.nodes,
            "num_actions": stats.num_actions,
            "root_lower": stats.root_lower,
            "root_upper": stats.root_upper,
            "elapsed_s": stats.elapsed_s,
        }
        self._num_decisions += 1

        if action.action_type is ActionType.NULL:
            self._termination_reason = "no_useful_action"
            return False

        reference = particle.scene if particle is not None else SceneState()
        self._symbolic_action = action
        self._navigated = False
        self._queue = self._translate(action, world_graph, reference)
        self._trace += f"Decision {self._num_decisions}: {action}\n"
        return bool(self._queue)

    def _ensure_belief(self, observation: Observation) -> bool:
        """Build the initial TOH belief if there is none yet."""
        if self.belief is not None and len(self.belief) > 0:
            return True
        assert self.toh is not None
        start = time.time()
        belief = self.toh.generate(
            instruction=self._instruction,
            observation=observation,
            wrong_goal_states=self._wrong_goal_states,
            failed_attempts=self._failed_attempts,
            memory=self._memory,
            history=self._history,
        )
        self._llm_time_s += time.time() - start
        if len(belief) == 0:
            # The Tree of Hypotheses produced nothing usable. There is no goal to
            # plan for, so stop rather than act arbitrarily.
            self._termination_reason = "empty_belief"
            self._trace += "Tree of Hypotheses returned no hypotheses; stopping.\n"
            return False
        self.belief = belief
        self._record_belief("initial")
        return True

    def _on_skill_finished(self, response: str, world_graph: "WorldGraph") -> None:
        """
        A skill reported back. Decide whether the symbolic action is complete and
        run the hybrid belief update when it is.
        """
        finished = self.last_high_level_actions.get(self._agent_uid)
        self.last_high_level_actions = {}
        success = self._response_is_success(response)
        self._trace += f"Result: {response}\n"

        if finished is not None and finished[0] == "Navigate" and success:
            self._navigated = True
            if self._queue:
                # The manipulation itself still has to run; the symbolic action is
                # not complete yet, so no belief update.
                return

        if not success:
            if self._symbolic_action is not None:
                self._failed_contexts.add((self._symbolic_action, self._action_context))
            self._queue = []
            if finished is not None:
                self._failed_attempts.append(
                    f"{finished[0]}[{finished[1]}] -> {response}"
                )

        if success and self._symbolic_action is not None:
            action = self._symbolic_action
            if action.action_type is ActionType.PICK and action.obj:
                self._observed_locations[action.obj] = HELD
                self._observed_relations.pop(action.obj, None)
                for anchors in self._observed_relations.values():
                    anchors.discard(action.obj)
            elif action.action_type is ActionType.PLACE:
                held = next(
                    (
                        obj
                        for obj, parent in self._observed_locations.items()
                        if parent == HELD
                    ),
                    None,
                )
                if held:
                    self._observed_locations[held] = action.area
                    self._observed_relations[held] = (
                        {action.next_to} if action.next_to else set()
                    )
            elif action.action_type in STATE_ACTION_EFFECTS and action.obj:
                flag, value = STATE_ACTION_EFFECTS[action.action_type]
                self._observed_flags.setdefault(action.obj, {})[flag] = value
            if action.action_type is ActionType.OPEN and action.area:
                self._deliberately_inspected.add(action.area)
                self._furniture_open_from_actions[action.area] = True
            elif action.action_type is ActionType.EXPLORE and action.area:
                assert self.domain is not None
                # A room tour is not proof of exhaustive visibility. Positive
                # detections still update belief; negative evidence needs an
                # explicit coverage certificate (Open supplies one below).
                pass

        if self._queue:
            return

        self._hybrid_update(success, response, world_graph)

    def _hybrid_update(
        self, success: bool, response: str, world_graph: "WorldGraph"
    ) -> None:
        assert self.updater is not None
        if self.belief is None or self._symbolic_action is None:
            return
        outcome = ExecutionOutcome(
            action=self._symbolic_action,
            success=success,
            navigated=self._navigated,
            message=response,
        )
        self._refresh_perception()
        observation = self._build_observation(world_graph)
        if success and self._symbolic_action.action_type is ActionType.EXPLORE:
            if self._action_context == self._observation_signature(observation):
                self._failed_contexts.add((self._symbolic_action, self._action_context))
        self._history.append(
            f"{self._symbolic_action}: success={success}, navigated={self._navigated}, "
            f"response={response}; observation={self._observation_signature(observation)}"
        )
        start = time.time()
        self.belief, info = self.updater.update(
            self.belief,
            outcome,
            observation,
            toh_context={
                "instruction": self._instruction,
                "wrong_goal_states": list(self._wrong_goal_states),
                "failed_attempts": list(self._failed_attempts),
                "memory": dict(self._memory),
                "history": list(self._history),
            },
        )
        self._trace += "Evidence: " + self._history[-1] + "\n"
        self._last_belief_update = info
        self._record_belief("updated")
        if info.get("replenished") or info.get("replenishment_failed"):
            self._llm_time_s += time.time() - start
        if len(self.belief) == 0:
            self._termination_reason = "empty_belief"
            self.is_done = True
        self._symbolic_action = None
        self._navigated = False
        self._trace += (
            f"Belief: mass={info['surviving_mass']:.2f} "
            f"eliminated={info['eliminated']} "
            f"replenished={info['replenished']} "
            f"particles={info.get('num_particles_after', 0)}\n"
        )

    def _record_belief(self, event):
        snapshot = [
            {
                "weight": p.weight,
                "goals": [g.to_dict() for g in p.goal_atoms],
                "placements": p.scene.object_parent,
                "hypothesized": sorted(p.scene.hypothesized),
            }
            for p in self.belief or ()
        ]
        self._trace += (
            f"Hypotheses ({event}): " + json.dumps(snapshot, sort_keys=True) + "\n"
        )
        self._trace += "Memory: " + json.dumps(self._memory, sort_keys=True) + "\n"

    def _map_goal_signature(self) -> Optional[Tuple[Any, ...]]:
        if self.belief is None:
            return None
        particle: Optional[Particle] = self.belief.map_particle()
        if particle is None:
            return None
        return tuple(particle.goal_atoms)

    @staticmethod
    def _response_is_success(response: str) -> bool:
        lowered = response.lower()
        if any(
            marker in lowered
            for marker in (*_FAILURE_MARKERS, "not successful", "unsuccessful")
        ):
            return False
        return "successful execution" in lowered or lowered.strip() == "success"

    # -- logging -------------------------------------------------------------

    def _planner_info(
        self, responses: Dict[int, str], replanned: bool
    ) -> Dict[str, Any]:
        from habitat_llm.utils.llm_usage import snapshot_from_llm

        agents = self.agents
        belief_summary = (
            self.belief.goal_summary() if self.belief is not None else "<no belief>"
        )
        info: Dict[str, Any] = {
            "replanned": {agent.uid: replanned for agent in agents},
            "replan_required": {agent.uid: replanned for agent in agents},
            "responses": responses,
            "is_done": {agent.uid: self.is_done for agent in agents},
            # The action issued on THIS step. ``last_high_level_actions`` is
            # cleared as soon as the skill reports back, which would leave the
            # runner with an empty dict on the step that produced the response.
            "high_level_actions": self._issued_high_level_actions,
            "prompts": {
                agent.uid: (
                    self.toh.last_prompts[-1]
                    if self.toh and self.toh.last_prompts
                    else ""
                )
                for agent in agents
            },
            "traces": {agent.uid: self._trace for agent in agents},
            "print": "",
            # Nested so DecentralizedEvaluationRunner can merge planner_info; it
            # only accepts dict or str values at the top level.
            "cost_metrics": {
                **snapshot_from_llm(self.llm),
                "physical_explore": True,
                "llm_planning_time_s": self._llm_time_s,
                "llm_call_count": len(self.toh.last_responses) if self.toh else 0,
                "despot_search_time_s": self._search_time_s,
            },
            "tru_pomdp": {
                "belief": belief_summary,
                "termination_reason": self._termination_reason,
                "memory_revisions": self._memory_revisions,
                "rejected_hypotheses": self.toh.rejected_hypotheses if self.toh else 0,
                "belief_update": dict(self._last_belief_update),
                "num_particles": len(self.belief) if self.belief is not None else 0,
                "num_decisions": self._num_decisions,
                "search_time_s": self._search_time_s,
                "llm_time_s": self._llm_time_s,
                "failed_attempts": len(self._failed_attempts),
                "inspected_areas": len(self._deliberately_inspected),
                **self._last_search_stats,
            },
        }
        if replanned or self.is_done:
            info["print"] = self._trace
        return info
