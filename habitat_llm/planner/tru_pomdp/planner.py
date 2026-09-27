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
            max_objects_per_combination=int(plan_config.get("max_goal_objects", 4)),
            max_particles=int(plan_config.get("max_particles", 48)),
            max_tokens=int(plan_config.get("toh_max_tokens", 4096)),
        )
        self.replenish_threshold = float(plan_config.get("replenish_threshold", 0.3))
        self.max_decisions = int(plan_config.get("max_decisions", 50))
        self.max_place_targets = int(plan_config.get("max_place_targets", 8))
        # A deliberately inspected area can refute a hypothesis. Areas that merely
        # happen to contain an observed object only confirm; set this to True if
        # the world model is trusted to be complete for open furniture.
        self.trust_observed_areas = bool(
            plan_config.get("trust_observed_areas", False)
        )
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

    def _state_affordances(
        self, world_graph: "WorldGraph"
    ) -> Dict[str, Set[str]]:
        affordances: Dict[str, Set[str]] = {}
        for obj in world_graph.get_all_objects():
            states = obj.properties.get("states", {}) or {}
            declared = {flag for flag in states if flag in TRACKED_STATE_FLAGS}
            if declared:
                affordances[obj.name] = declared
        return affordances

    def _build_observation(self, world_graph: "WorldGraph") -> Observation:
        """
        Translate the world graph into the symbolic observation.

        The contract matches ``SymbolicDomain.observe`` exactly: closed
        containers hide their contents, objects count as revealed only in
        inspected areas, and a known furniture node is not an inspected one.
        """
        assert self.domain is not None
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
        for obj in world_graph.get_all_objects():
            try:
                held = world_graph.is_object_with_agent(obj, agent_type="robot")
            except (ValueError, KeyError):
                held = False
            if held:
                object_parent[obj.name] = HELD
            else:
                furniture = world_graph.find_furniture_for_object(obj)
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

        occupied = {
            parent for parent in object_parent.values() if parent != HELD
        }
        inspected = set(self._deliberately_inspected) | occupied
        fully_inspected = set(self._deliberately_inspected)
        if self.trust_observed_areas:
            fully_inspected |= occupied

        return Observation(
            object_parent=object_parent,
            furniture_open=furniture_open,
            object_states=object_states,
            spatial_relations={},
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
            position = np.asarray(
                list(robot.get_property("translation")), dtype=float
            )
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
        if nav_target is not None and self._needs_navigation(
            world_graph, nav_target
        ):
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
            position = np.asarray(
                list(robot.get_property("translation")), dtype=float
            )
            other = np.asarray(
                list(node.get_property("translation")), dtype=float
            )
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
        if skill_name == "Navigate":
            self._navigated = True
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
            return False

        observation = self._build_observation(world_graph)
        if not self._ensure_belief(observation):
            return False

        assert self.belief is not None
        particle = self.belief.map_particle()
        if particle is not None and self.domain.goal_satisfied(
            particle.scene, particle.goal_atoms
        ):
            return False

        start = time.time()
        action, stats = self.solver.plan(self.belief)
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
        )
        self._llm_time_s += time.time() - start
        if len(belief) == 0:
            # The Tree of Hypotheses produced nothing usable. There is no goal to
            # plan for, so stop rather than act arbitrarily.
            self._trace += "Tree of Hypotheses returned no hypotheses; stopping.\n"
            return False
        self.belief = belief
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
            if finished[1]:
                self._deliberately_inspected.add(finished[1])
            if self._queue:
                # The manipulation itself still has to run; the symbolic action is
                # not complete yet, so no belief update.
                return

        if not success:
            self._queue = []
            if finished is not None:
                self._failed_attempts.append(
                    f"{finished[0]}[{finished[1]}] -> {response}"
                )

        if success and self._symbolic_action is not None:
            action = self._symbolic_action
            if action.action_type is ActionType.OPEN and action.area:
                self._deliberately_inspected.add(action.area)
                self._furniture_open_from_actions[action.area] = True
            elif action.action_type is ActionType.EXPLORE and action.area:
                assert self.domain is not None
                for furniture in self.domain.room_furniture.get(action.area, []):
                    self._deliberately_inspected.add(furniture)

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
        observation = self._build_observation(world_graph)
        previous_goal = self._map_goal_signature()
        start = time.time()
        self.belief, info = self.updater.update(
            self.belief,
            outcome,
            observation,
            toh_context={
                "instruction": self._instruction,
                "wrong_goal_states": list(self._wrong_goal_states),
                "failed_attempts": list(self._failed_attempts),
            },
        )
        if info.get("replenished") or info.get("fallback_to_predicted"):
            self._llm_time_s += time.time() - start
            if previous_goal is not None and previous_goal not in self._wrong_goal_states:
                # The belief collapsed: the goal it was pursuing is the best
                # available evidence of a wrong hypothesis, so future TOH queries
                # are told about it.
                self._wrong_goal_states.append(previous_goal)
        self._symbolic_action = None
        self._navigated = False
        self._trace += (
            f"Belief: mass={info['surviving_mass']:.2f} "
            f"eliminated={info['eliminated']} "
            f"replenished={info['replenished']} "
            f"particles={info.get('num_particles_after', 0)}\n"
        )

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
        if "successful" in lowered or "success" in lowered:
            return not any(marker in lowered for marker in ("unexpected failure",))
        return not any(marker in lowered for marker in _FAILURE_MARKERS)

    # -- logging -------------------------------------------------------------

    def _planner_info(
        self, responses: Dict[int, str], replanned: bool
    ) -> Dict[str, Any]:
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
                agent.uid: (self.toh.last_prompts[-1] if self.toh and self.toh.last_prompts else "")
                for agent in agents
            },
            "traces": {agent.uid: self._trace for agent in agents},
            "print": "",
            # Nested so DecentralizedEvaluationRunner can merge planner_info; it
            # only accepts dict or str values at the top level.
            "cost_metrics": {
                "llm_planning_time_s": self._llm_time_s,
                "llm_call_count": len(self.toh.last_responses) if self.toh else 0,
                "despot_search_time_s": self._search_time_s,
            },
            "tru_pomdp": {
                "belief": belief_summary,
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
