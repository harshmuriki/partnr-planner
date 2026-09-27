#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Hybrid belief update for the Tru-POMDP planner (paper section 4.2).

After every completed high-level action:

1. predict with the action AND the execution outcome, because a navigate-then-
   failed-pick is not the same as NULL;
2. eliminate particles inconsistent with the observation contract;
3. compute the surviving mass w_BF BEFORE normalisation;
4. if w_BF < 0.3, regenerate the Tree of Hypotheses from the interaction history
   and mix b_new = b_BF + (1 - w_BF) * b_LLM;
5. otherwise normalise the survivors.
"""

import random
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from habitat_llm.planner.tru_pomdp.scene import (
    HELD,
    Action,
    ActionType,
    GoalAtom,
    Particle,
    SceneState,
    SymbolicDomain,
    names_match,
)

if TYPE_CHECKING:
    from habitat_llm.planner.tru_pomdp.toh import TreeOfHypotheses

#: Replenish the belief when the surviving mass drops below this (paper: 1-eps).
DEFAULT_REPLENISH_THRESHOLD = 0.3


@dataclass
class Observation:
    """
    What the robot actually saw, in the same vocabulary the symbolic domain uses.

    :param object_parent: observed object id -> furniture id, or HELD.
    :param furniture_open: observed open/closed flags.
    :param object_states: observed object id -> observed state flags. Only flags
        present here constrain the belief.
    :param spatial_relations: observed next_to relations.
    :param robot_area: where the robot is.
    :param inspected_areas: every area whose contents have been looked at.
    :param fully_inspected_areas: the subset of ``inspected_areas`` whose
        inspection was complete enough that a missing object is evidence of
        absence. Only these may eliminate a hypothesis.
    :param known_furniture: furniture nodes the robot knows exist.
    """

    object_parent: Dict[str, str] = field(default_factory=dict)
    furniture_open: Dict[str, bool] = field(default_factory=dict)
    object_states: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    spatial_relations: Dict[str, Set[str]] = field(default_factory=dict)
    robot_area: Optional[str] = None
    inspected_areas: Set[str] = field(default_factory=set)
    fully_inspected_areas: Set[str] = field(default_factory=set)
    known_furniture: Set[str] = field(default_factory=set)


@dataclass
class ExecutionOutcome:
    """
    The result of running one high-level action in Habitat.

    A failed skill is not a no-op: ``navigated`` records that the robot moved
    before the manipulation failed, and the failure itself is remembered so the
    Tree of Hypotheses can condition on it.
    """

    action: Action
    success: bool
    navigated: bool = False
    message: str = ""


class Belief:
    """A weighted particle set."""

    def __init__(self, particles: Optional[Iterable[Particle]] = None) -> None:
        self.particles: List[Particle] = list(particles or ())

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self) -> Iterator[Particle]:
        return iter(self.particles)

    def total_weight(self) -> float:
        return sum(p.weight for p in self.particles)

    def normalize(self) -> "Belief":
        total = self.total_weight()
        if total > 0.0:
            for particle in self.particles:
                particle.weight /= total
        return self

    def scale(self, factor: float) -> "Belief":
        for particle in self.particles:
            particle.weight *= factor
        return self

    def map_particle(self) -> Optional[Particle]:
        if not self.particles:
            return None
        return max(self.particles, key=lambda p: p.weight)

    def copy(self) -> "Belief":
        return Belief(p.copy() for p in self.particles)

    def goal_summary(self) -> str:
        particle = self.map_particle()
        if particle is None:
            return "<empty belief>"
        atoms = ", ".join(str(atom) for atom in particle.goal_atoms)
        return f"w={particle.weight:.2f} goal=[{atoms}]"


class HybridBeliefUpdater:
    """Prediction, elimination and TOH replenishment."""

    def __init__(
        self,
        domain: SymbolicDomain,
        toh: Optional["TreeOfHypotheses"] = None,
        replenish_threshold: float = DEFAULT_REPLENISH_THRESHOLD,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.domain = domain
        self.toh = toh
        self.replenish_threshold = replenish_threshold
        self.rng = rng or random.Random(0)

    # -- 1. prediction -------------------------------------------------------

    def predict(self, belief: Belief, outcome: ExecutionOutcome) -> Belief:
        """
        Apply the deterministic transition model conditioned on the *outcome*.

        On success the full effects are applied. On failure only the effects that
        actually happened are applied, which for the PARTNR oracle skills means
        the implicit navigation. A particle that considers the action infeasible
        is left unchanged; the elimination step then decides whether that
        particle can survive the observation.
        """
        action = outcome.action
        predicted: List[Particle] = []
        for particle in belief:
            nxt = particle.copy()
            scene = nxt.scene
            if outcome.success:
                feasible, _ = self.domain.feasible(scene, action)
                if feasible:
                    target = self.domain.nav_target(scene, action)
                    if target is not None:
                        scene.robot_area = target
                    self.domain.apply_effects(scene, action)
                elif outcome.navigated:
                    self._apply_navigation_only(scene, action)
            elif outcome.navigated:
                self._apply_navigation_only(scene, action)
            predicted.append(nxt)
        return Belief(predicted)

    def _apply_navigation_only(self, scene: SceneState, action: Action) -> None:
        """Partial effect of a failed skill: the robot still moved."""
        target = self.domain.nav_target(scene, action)
        if target is not None:
            scene.robot_area = target

    # -- 2. elimination ------------------------------------------------------

    def eliminate(
        self, belief: Belief, observation: Observation
    ) -> Tuple[Belief, float, List[Particle]]:
        """
        Drop particles inconsistent with the observation and fold the observation
        into the survivors.

        :return: (surviving belief, surviving mass before normalisation,
            eliminated particles).
        """
        survivors: List[Particle] = []
        eliminated: List[Particle] = []
        for particle in belief:
            candidate = particle.copy()
            self._ground_hypotheses(candidate, observation)
            if self._consistent(candidate, observation):
                self._incorporate(candidate, observation)
                survivors.append(candidate)
            else:
                eliminated.append(particle)
        surviving_mass = sum(p.weight for p in survivors)
        return Belief(survivors), surviving_mass, eliminated

    def _consistent(self, particle: Particle, observation: Observation) -> bool:
        scene = particle.scene
        for obj, observed_parent in observation.object_parent.items():
            if obj not in scene.object_parent:
                # A newly observed object (typically a distractor). It is folded
                # into the particle's bookkeeping rather than treated as
                # contradictory evidence, so distractors cannot wipe the belief.
                continue
            if scene.object_parent[obj] != observed_parent:
                return False
        for area in observation.fully_inspected_areas:
            if not observation.furniture_open.get(area, True):
                continue
            for obj, parent in scene.object_parent.items():
                if parent != area:
                    continue
                if obj not in observation.object_parent:
                    # The area was inspected completely enough that a missing
                    # object refutes the hypothesis that placed it there.
                    return False
        return True

    def _incorporate(self, particle: Particle, observation: Observation) -> None:
        """
        Fold observed facts into a surviving particle.

        Observed object-state flags are copied in rather than used to eliminate:
        the flags are a deterministic reading of shared state, not part of the
        hypothesis space (which covers goals and hidden locations).
        """
        scene = particle.scene
        scene.robot_area = observation.robot_area
        scene.furniture_open.update(observation.furniture_open)
        scene.inspected_areas = set(observation.fully_inspected_areas)
        scene.observed_objects.update(observation.object_parent)
        for obj, parent in observation.object_parent.items():
            scene.object_parent[obj] = parent
            scene.hypothesized.discard(obj)
        for obj, flags in observation.object_states.items():
            scene.object_states.setdefault(obj, {}).update(flags)
        for obj, anchors in observation.spatial_relations.items():
            scene.spatial_relations[obj] = set(anchors)

    def _ground_hypotheses(self, particle: Particle, observation: Observation) -> None:
        """
        Match hypothesised TOH names onto real Habitat object ids.

        A hypothesised object is grounded when the area it was hypothesised in has
        been inspected and holds an observed object with a compatible name. Until
        then the hypothesised name stays distinct from every Habitat id.
        """
        scene = particle.scene
        if not scene.hypothesized:
            return
        claimed = set(scene.object_parent) - scene.hypothesized
        for hypothesised in sorted(scene.hypothesized):
            if hypothesised in observation.object_parent:
                continue
            parent = scene.object_parent.get(hypothesised)
            if parent is None or parent == HELD:
                continue
            if parent not in observation.inspected_areas:
                continue
            match = None
            for observed, observed_parent in sorted(observation.object_parent.items()):
                if observed_parent != parent or observed in claimed:
                    continue
                if names_match(hypothesised, observed):
                    match = observed
                    break
            if match is None:
                continue
            claimed.add(match)
            scene.rename_object(hypothesised, match)
            particle.goal_atoms = tuple(
                atom.rename(hypothesised, match) for atom in particle.goal_atoms
            )

    # -- 3-5. the hybrid update ---------------------------------------------

    def update(
        self,
        belief: Belief,
        outcome: ExecutionOutcome,
        observation: Observation,
        toh_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Belief, Dict[str, Any]]:
        """
        One full hybrid belief update.

        :param toh_context: keyword arguments forwarded to
            ``TreeOfHypotheses.generate`` when replenishment is triggered.
        :return: (new belief, info dict for logging).
        """
        predicted = self.predict(belief, outcome)
        filtered, surviving_mass, eliminated = self.eliminate(predicted, observation)
        info: Dict[str, Any] = {
            "surviving_mass": surviving_mass,
            "eliminated": len(eliminated),
            "replenished": False,
            "num_particles_before": len(belief),
        }

        if surviving_mass < self.replenish_threshold:
            info["replenishment_cause"] = (
                "belief_collapsed" if len(filtered) == 0 else "low_surviving_mass"
            )
            llm_belief = self._generate_llm_belief(observation, toh_context)
            if llm_belief is not None:
                llm_belief, _, rejected = self.eliminate(llm_belief, observation)
                info["rejected_generated"] = len(rejected)
            if llm_belief is not None and len(llm_belief) > 0:
                llm_belief.normalize().scale(1.0 - surviving_mass)
                merged = Belief(list(filtered) + list(llm_belief))
                info["replenished"] = True
                info["num_llm_particles"] = len(llm_belief)
                info["num_particles_after"] = len(merged)
                return merged.normalize(), info
            info["replenishment_failed"] = True
            if len(filtered) == 0:
                info["termination_reason"] = "empty_belief"

        filtered.normalize()
        info["num_particles_after"] = len(filtered)
        return filtered, info

    def _generate_llm_belief(
        self,
        observation: Observation,
        toh_context: Optional[Dict[str, Any]],
    ) -> Optional[Belief]:
        if self.toh is None:
            return None
        context = dict(toh_context or {})
        context.setdefault("observation", observation)
        return self.toh.generate(**context)


def particle_from_observation(
    observation: Observation,
    goal_atoms: Sequence[GoalAtom] = (),
    furniture_room: Optional[Dict[str, str]] = None,
    weight: float = 1.0,
) -> Particle:
    """
    Build a fully observed particle. Used for tests and as the seed a TOH
    hypothesis is layered on top of.
    """
    scene = SceneState(
        object_parent=dict(observation.object_parent),
        furniture_room=dict(furniture_room or {}),
        furniture_open=dict(observation.furniture_open),
        object_states={k: dict(v) for k, v in observation.object_states.items()},
        spatial_relations={k: set(v) for k, v in observation.spatial_relations.items()},
        robot_area=observation.robot_area,
        inspected_areas=set(observation.fully_inspected_areas),
        observed_objects=set(observation.object_parent),
    )
    return Particle(scene=scene, goal_atoms=tuple(goal_atoms), weight=weight)


def null_outcome(navigated: bool = False) -> ExecutionOutcome:
    """A NULL execution, used to fold in an observation without acting."""
    return ExecutionOutcome(Action(ActionType.NULL), success=True, navigated=navigated)
