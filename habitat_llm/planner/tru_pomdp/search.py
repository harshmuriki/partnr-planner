#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
DESPOT belief tree search for the Tru-POMDP planner.

Why a Python DESPOT rather than a wrapper: AdaCompNUS/despot and
RoboticSJTU/tru_pomdp are C++ projects that require the POMDP model itself to be
written in C++ and linked against their solver. Neither is importable from this
repository without building a C++ extension and re-expressing the whole domain,
so they are not "trivially usable" here. The algorithm below is DESPOT
(Ye et al. 2017) as used by the paper, not a generic lookahead: it samples K
scenarios from the belief, branches on actions, branches on the *predicted
observation keys* while keeping the matching particle subset on each branch,
runs trials guided by upper bounds and weighted excess uncertainty, backs values
up with Bellman's principle, and uses the Appendix A.2 rollout as the leaf lower
bound.

Node values are stored per unit weight, i.e. as expected returns conditioned on
reaching the node. Observation-branch probabilities are the particle-subset
weight fractions, which is exact because the observation model is deterministic.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from habitat_llm.planner.tru_pomdp.scene import (
    HELD,
    NULL_ACTION,
    STATE_LITERAL_ACTIONS,
    Action,
    ActionType,
    GoalAtom,
    Particle,
    SceneState,
    SymbolicDomain,
)


@dataclass
class DespotConfig:
    """Hyperparameters. Defaults are the paper's Appendix A.5 values."""

    num_scenarios: int = 30  # k
    max_search_depth: int = 20  # d_s
    rollout_depth: int = 10  # d_r
    discount: float = 0.95
    xi: float = 0.95  # DESPOT regularisation on excess uncertainty
    num_trials: int = 500
    planning_time_s: float = 1.0
    seed: int = 0


@dataclass
class Scenario:
    """A sampled state plus its weight inside a belief node."""

    state: Particle
    weight: float
    terminal: bool = False


class QNode:
    """An action branch of a belief node."""

    __slots__ = ("parent", "action", "immediate_reward", "children", "lower", "upper")

    def __init__(self, parent: "VNode", action: Action) -> None:
        self.parent = parent
        self.action = action
        self.immediate_reward = 0.0
        self.children: Dict[Any, "VNode"] = {}
        self.lower = 0.0
        self.upper = 0.0


class VNode:
    """
    A belief node: the particle subset that reaches it, plus value bounds.

    ``scenarios`` holds only the non-terminal scenarios. ``weight`` is the total
    weight that reaches the node including terminal scenarios, whose future
    return is zero. Keeping the two separate makes the value estimates correct
    without splitting terminal and non-terminal particles into different
    observation branches.
    """

    __slots__ = (
        "scenarios",
        "weight",
        "depth",
        "parent",
        "observation",
        "children",
        "lower",
        "upper",
        "default_value",
        "default_action",
    )

    def __init__(
        self,
        scenarios: List[Scenario],
        weight: float,
        depth: int,
        parent: Optional[QNode] = None,
        observation: Any = None,
    ) -> None:
        self.scenarios = scenarios
        self.weight = weight
        self.depth = depth
        self.parent = parent
        self.observation = observation
        self.children: Dict[Action, QNode] = {}
        self.lower = 0.0
        self.upper = 0.0
        self.default_value = 0.0
        self.default_action: Action = NULL_ACTION

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_terminal(self) -> bool:
        return not self.scenarios


# ---------------------------------------------------------------------------
# Appendix A.2 rollout policy
# ---------------------------------------------------------------------------


def _anchor_ready(scene: SceneState, atom: GoalAtom, domain: SymbolicDomain) -> bool:
    """Whether a next_to anchor is already on the target area."""
    del domain
    if atom.next_to is None or atom.target_area is None:
        return False
    return scene.object_parent.get(atom.next_to) == atom.target_area


def _state_action_for(
    atom: GoalAtom, scene: SceneState, domain: SymbolicDomain
) -> Optional[Action]:
    """First feasible object-state action that advances an unmet state atom."""
    for literal in atom.states:
        if domain.state_satisfied(scene, atom.obj, literal):
            continue
        action_type = STATE_LITERAL_ACTIONS.get(literal)
        if action_type is None:
            continue
        candidate = Action(action_type, obj=atom.obj)
        if domain.feasible(scene, candidate)[0]:
            return candidate
    return None


def a2_next_action(
    unreached_goals: Sequence[GoalAtom],
    scene: SceneState,
    domain: SymbolicDomain,
) -> Action:
    """
    Transcription of the published Appendix A.2 rollout policy.

    The paper generates ``ActionSimple NextAction(...)`` once with OpenAI o1 and
    adopts the C++ code without modification. That published function is
    transcribed here statement by statement, including its comments and its
    ordering of the holding/not-holding cases; nothing about the policy was
    re-derived or regenerated by an LLM.

    Only the blocks marked "Habitat extension" are additions, and they are the
    three the plan allows:

    * a target hypothesised in an area that is open but not yet inspected leads
      to Explore of that room (the paper's model assumes open areas are fully
      visible, PARTNR's does not);
    * an unmet state atom on a visible object leads to PowerOn/PowerOff/Fill/Clean;
    * PLACE carries the atom's relation, and its next_to anchor when the anchor
      is already on the target area.

    One interpretation is documented rather than guessed: ``GetObjectParent`` for
    a held object returns the ROBOT node in the paper's scene graph, which makes
    the published "place it back in its parent" branch unusable. The branch is
    kept verbatim and the held object's parent is resolved through
    ``scene.previous_parent`` (the area it was picked from), which is what the
    published comment asks for.
    """
    # If no goals remain, return no-op
    if not unreached_goals:
        return NULL_ACTION

    # Check each goal
    for atom in unreached_goals:
        goal_area = atom.target_area
        goal_object = atom.obj

        # Skip if the area or object is not in the scene
        if not domain.check_object_in_scene(scene, goal_object):
            continue
        if goal_area is not None and not domain.check_area_in_scene(scene, goal_area):
            continue

        # Habitat extension: state-only atoms carry no placement requirement, so
        # the placement reasoning below does not apply to them.
        if goal_area is None:
            state_action = _state_action_for(atom, scene, domain)
            if state_action is not None:
                return state_action
            parent_of_target = domain.get_object_parent(scene, goal_object)
            explore = _explore_for_uninspected(scene, domain, parent_of_target)
            if explore is not None:
                return explore
            continue

        # If goal is already satisfied, skip it
        parent_of_goal = domain.get_object_parent(scene, goal_object)
        if parent_of_goal == goal_area and domain.atom_satisfied(scene, atom):
            continue

        # Check what the robot is currently holding
        object_in_hand = domain.get_object_in_hand(scene)

        if object_in_hand == goal_object:
            # Need to place it in goal_area
            if not domain.get_area_open_from_id(scene, goal_area):
                # Open the goal area first
                return Action(ActionType.OPEN, area=goal_area)
            else:
                # Place the object
                # Habitat extension: keep the atom's relation, and its anchor
                # once the anchor is actually on the target area.
                next_to = atom.next_to if _anchor_ready(scene, atom, domain) else None
                return Action(
                    ActionType.PLACE,
                    area=goal_area,
                    relation=atom.relation,
                    next_to=next_to,
                )

        elif object_in_hand is not None:
            # Holding a different object; place it back in its parent
            parent_of_held = domain.get_object_parent(scene, object_in_hand)
            if parent_of_held == HELD:
                parent_of_held = scene.previous_parent.get(object_in_hand)
            if parent_of_held is None:
                # No known place to put it; do nothing
                return NULL_ACTION
            if not domain.get_area_open_from_id(scene, parent_of_held):
                return Action(ActionType.OPEN, area=parent_of_held)
            else:
                return Action(ActionType.PLACE, area=parent_of_held)

        else:
            # Holding nothing; pick up the goal object
            if parent_of_goal is None:
                # No parent area known; do nothing
                return NULL_ACTION
            if not domain.get_area_open_from_id(scene, parent_of_goal):
                # Open the parent's area first
                return Action(ActionType.OPEN, area=parent_of_goal)
            else:
                # Habitat extension: an open area still hides its contents until
                # the robot has inspected it, so search that room first.
                explore = _explore_for_uninspected(scene, domain, parent_of_goal)
                if explore is not None:
                    return explore
                # Then pick the object
                return Action(
                    ActionType.PICK, area=parent_of_goal, obj=goal_object
                )

    # If all goals are satisfied or no action can be deduced, return no-op
    return NULL_ACTION


def _explore_for_uninspected(
    scene: SceneState, domain: SymbolicDomain, area: Optional[str]
) -> Optional[Action]:
    """Explore the room of an open-but-uninspected area, if there is one."""
    if area is None or area == HELD:
        return None
    if area in scene.inspected_areas:
        return None
    if not domain.get_area_open_from_id(scene, area):
        return None
    room = domain.room_of(area)
    if room is None or room not in domain.room_furniture:
        return None
    return Action(ActionType.EXPLORE, area=room)


class A2RolloutPolicy:
    """The Appendix A.2 policy used as DESPOT's scenario lower bound."""

    def __init__(self, domain: SymbolicDomain, config: DespotConfig) -> None:
        self.domain = domain
        self.config = config

    def next_action(self, state: Particle) -> Action:
        unreached = self.domain.unsatisfied_atoms(state.scene, state.goal_atoms)
        return a2_next_action(unreached, state.scene, self.domain)

    def value(self, state: Particle) -> Tuple[float, Action]:
        """
        Simulate the policy for ``rollout_depth`` steps and return the discounted
        return together with the first action it chose.
        """
        discount = self.config.discount
        total = 0.0
        gamma = 1.0
        current = state
        first_action = NULL_ACTION
        for step_index in range(self.config.rollout_depth):
            action = self.next_action(current)
            if step_index == 0:
                first_action = action
            if action.action_type is ActionType.NULL:
                break
            current, reward, terminal = self.domain.step(current, action)
            total += gamma * reward
            gamma *= discount
            if terminal:
                break
        return total, first_action


# ---------------------------------------------------------------------------
# DESPOT
# ---------------------------------------------------------------------------


@dataclass
class SearchStats:
    """Diagnostics for logging."""

    trials: int = 0
    nodes: int = 0
    root_lower: float = 0.0
    root_upper: float = 0.0
    num_actions: int = 0
    action_values: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0


class DespotSolver:
    """Anytime DESPOT over the particle belief."""

    def __init__(
        self,
        domain: SymbolicDomain,
        config: Optional[DespotConfig] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.domain = domain
        self.config = config or DespotConfig()
        self.rng = rng or random.Random(self.config.seed)
        self.rollout = A2RolloutPolicy(domain, self.config)
        self._node_count = 0

    # -- scenario sampling ---------------------------------------------------

    def sample_scenarios(self, belief: Iterable[Particle]) -> List[Scenario]:
        """
        Sample k scenarios from the belief with replacement, proportional to
        particle weight. Each scenario carries weight 1/k.
        """
        particles = [p for p in belief if p.weight > 0.0]
        if not particles:
            return []
        k = max(1, self.config.num_scenarios)
        total = sum(p.weight for p in particles)
        cumulative: List[float] = []
        running = 0.0
        for particle in particles:
            running += particle.weight / total
            cumulative.append(running)
        scenarios: List[Scenario] = []
        share = 1.0 / k
        for _ in range(k):
            draw = self.rng.random()
            index = 0
            while index < len(cumulative) - 1 and draw > cumulative[index]:
                index += 1
            chosen = particles[index]
            terminal = self.domain.goal_satisfied(chosen.scene, chosen.goal_atoms)
            scenarios.append(Scenario(chosen.copy(), share, terminal))
        return scenarios

    # -- bounds --------------------------------------------------------------

    def _init_bounds(self, node: VNode) -> None:
        if node.weight <= 0.0 or not node.scenarios:
            node.lower = 0.0
            node.upper = 0.0
            node.default_value = 0.0
            return
        lower = 0.0
        upper = 0.0
        best_action = NULL_ACTION
        best_weight = -1.0
        for scenario in node.scenarios:
            value, action = self.rollout.value(scenario.state)
            lower += scenario.weight * value
            upper += scenario.weight * self.domain.optimistic_value(scenario.state)
            if scenario.weight > best_weight and action.action_type is not ActionType.NULL:
                best_weight = scenario.weight
                best_action = action
        node.lower = lower / node.weight
        node.default_value = node.lower
        node.default_action = best_action
        node.upper = max(upper / node.weight, node.lower)

    # -- expansion -----------------------------------------------------------

    def _expand(self, node: VNode) -> None:
        """
        Create one QNode per legal action. For each action, step every scenario,
        group the resulting scenarios by their *predicted observation key* and
        create one child belief node per distinct key holding exactly the
        matching particle subset. This is DESPOT's observation branching.
        """
        belief = [scenario.state for scenario in node.scenarios]
        actions = self.domain.legal_actions(belief)
        for action in actions:
            qnode = QNode(node, action)
            immediate = 0.0
            groups: Dict[Any, List[Scenario]] = {}
            group_weight: Dict[Any, float] = {}
            for scenario in node.scenarios:
                next_state, reward, terminal = self.domain.step(scenario.state, action)
                immediate += scenario.weight * reward
                observation = self.domain.observe(next_state, action)
                groups.setdefault(observation, []).append(
                    Scenario(next_state, scenario.weight, terminal)
                )
                group_weight[observation] = (
                    group_weight.get(observation, 0.0) + scenario.weight
                )
            qnode.immediate_reward = immediate / node.weight
            for observation, scenarios in groups.items():
                surviving = [s for s in scenarios if not s.terminal]
                child = VNode(
                    scenarios=surviving,
                    weight=group_weight[observation],
                    depth=node.depth + 1,
                    parent=qnode,
                    observation=observation,
                )
                self._node_count += 1
                self._init_bounds(child)
                qnode.children[observation] = child
            self._backup_qnode(qnode)
            node.children[action] = qnode

    # -- Bellman backup ------------------------------------------------------

    def _backup_qnode(self, qnode: QNode) -> None:
        discount = self.config.discount
        parent_weight = qnode.parent.weight
        lower = qnode.immediate_reward
        upper = qnode.immediate_reward
        for child in qnode.children.values():
            probability = child.weight / parent_weight if parent_weight else 0.0
            lower += discount * probability * child.lower
            upper += discount * probability * child.upper
        qnode.lower = lower
        qnode.upper = max(upper, lower)

    def _backup_vnode(self, node: VNode) -> None:
        if node.is_leaf:
            return
        lower = node.default_value
        upper = node.default_value
        for qnode in node.children.values():
            lower = max(lower, qnode.lower)
            upper = max(upper, qnode.upper)
        node.lower = lower
        node.upper = max(upper, lower)

    def _backup(self, node: Optional[VNode]) -> None:
        current = node
        while current is not None:
            self._backup_vnode(current)
            qnode = current.parent
            if qnode is None:
                return
            self._backup_qnode(qnode)
            current = qnode.parent

    # -- trials --------------------------------------------------------------

    def _weighted_excess_uncertainty(self, node: VNode, root: VNode) -> float:
        if root.weight <= 0.0:
            return 0.0
        scale = (self.config.discount ** node.depth) * (node.weight / root.weight)
        return scale * (node.upper - node.lower) - self.config.xi * (
            root.upper - root.lower
        )

    def _select_best_upper_bound_action(self, node: VNode) -> Optional[QNode]:
        best: Optional[QNode] = None
        best_value = -math.inf
        for qnode in node.children.values():
            if qnode.upper > best_value:
                best_value = qnode.upper
                best = qnode
        return best

    def _select_best_weu_child(self, qnode: QNode, root: VNode) -> Optional[VNode]:
        best: Optional[VNode] = None
        best_value = 0.0
        for child in qnode.children.values():
            if child.is_terminal:
                continue
            value = self._weighted_excess_uncertainty(child, root)
            if value > best_value:
                best_value = value
                best = child
        return best

    def _trial(self, root: VNode) -> VNode:
        current = root
        while True:
            if current.depth >= self.config.max_search_depth or current.is_terminal:
                break
            if current.is_leaf:
                self._expand(current)
                if current.is_leaf:
                    break
            qnode = self._select_best_upper_bound_action(current)
            if qnode is None:
                break
            child = self._select_best_weu_child(qnode, root)
            if child is None:
                break
            current = child
            if self._weighted_excess_uncertainty(current, root) <= 0.0:
                break
        return current

    # -- entry point ---------------------------------------------------------

    def plan(self, belief: Iterable[Particle]) -> Tuple[Action, SearchStats]:
        """
        Run DESPOT and return the action with the best lower bound at the root,
        together with search diagnostics.
        """
        start = time.time()
        self._node_count = 0
        scenarios = self.sample_scenarios(belief)
        stats = SearchStats()
        if not scenarios:
            stats.elapsed_s = time.time() - start
            return NULL_ACTION, stats

        total_weight = sum(s.weight for s in scenarios)
        root = VNode(
            scenarios=[s for s in scenarios if not s.terminal],
            weight=total_weight,
            depth=0,
        )
        self._node_count += 1
        self._init_bounds(root)
        if root.is_terminal:
            stats.elapsed_s = time.time() - start
            stats.nodes = self._node_count
            return NULL_ACTION, stats

        trials = 0
        while trials < self.config.num_trials:
            if time.time() - start > self.config.planning_time_s and trials > 0:
                break
            leaf = self._trial(root)
            self._backup(leaf)
            trials += 1
            if root.upper - root.lower <= 1e-6:
                break

        best_action = root.default_action
        best_value = -math.inf
        for action, qnode in root.children.items():
            if qnode.lower > best_value:
                best_value = qnode.lower
                best_action = action
            stats.action_values[str(action)] = qnode.lower

        stats.trials = trials
        stats.nodes = self._node_count
        stats.root_lower = root.lower
        stats.root_upper = root.upper
        stats.num_actions = len(root.children)
        stats.elapsed_s = time.time() - start
        return best_action, stats
