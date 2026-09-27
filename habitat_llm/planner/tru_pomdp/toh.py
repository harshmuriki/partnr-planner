#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Tree of Hypotheses (paper section 4.1, prompts from Appendix A.1).

Three levels, structurally unchanged from the paper:

* L1: alternative sets of target objects, with confidences.
* L2: a placement goal per set.
* L3: candidate current locations for every target object that is not observed.

The prompts below are the Appendix A.1 prompts with terminology substitutions
only: the paper's AREA nodes become PARTNR furniture ids, the kitchen becomes a
multi-room house, the "closed areas" rule in L3 becomes "closed containers or
furniture that has not been inspected yet" (PARTNR does not make open areas
fully visible), and L2 gains the optional ``relation``/``next_to``/``states``
fields that PARTNR goals can carry. The reasoning chains, the critical rules, the
section structure and the JSON schemas are the paper's.

A particle's weight is the product of the confidences along its root-to-leaf
path, normalised over the belief.
"""

import itertools
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from habitat_llm.planner.tru_pomdp.belief import Belief, Observation
from habitat_llm.planner.tru_pomdp.scene import (
    HELD,
    GoalAtom,
    Particle,
    SceneState,
    SymbolicDomain,
    names_match,
    normalize_object_name,
)

# ---------------------------------------------------------------------------
# Appendix A.1: system prompt for Level 1 & 2
# ---------------------------------------------------------------------------

TOH_LEVEL12_SYSTEM_PROMPT = """# Role

You are an assistant to solve an object rearrangement task in a household environment.

An AREA is a furniture node of the house, identified by an id such as table_0, counter_24,
fridge_0 or chest_of_drawers_2. Every AREA belongs to a room such as kitchen_1 or
living_room_1. A robot carries out the task using these skills: Navigate, Explore, Open,
Close, Pick, Place, Rearrange, PowerOn, PowerOff, Fill, Clean, Pour.

You'll receive:

1. the language instruction of the task, including the objects_of_interest and their target_areas.

2. the description of current observation of the environment, listing all open areas, closed areas and observed objects with their placements. If the object is inside a closed area, it is not visible.

3. A list of previously attempted(wrong) goal states, each containing combinations of objects and target areas that failed to achieve the goal.

4. a list of objects that have been already placed in the target areas.

# Task

You need to:

1. Identify the correct objects of interest based strictly on the instruction and observation. Only select objects can help complete the task.

2. Select valid target areas for those objects based on the task instruction, using areas explicitly mentioned in the observation.

3. Provide up to {k} possible valid combinations of objects and their target areas.

# Guidelines

## For objects of interest:

1. The instruction is ambiguous. You need to infer the intent of the language instruction and use common sense.

2. Consider both visual objects and, more importantly, unseen objects of interest in the closed areas and in the areas that have not been inspected yet.

3. Use '_' to connect multi-word object names (e.g., 'bell_pepper', not 'bell pepper').

4. For the observed objects, use its full name appeared the observation.

## For target areas:

1. The instruction is ambiguous. You need to infer the intent of the language instruction and use common sense.

2. The target areas can only be selected from the areas explicitly mentioned in the observation.

3. A target_area must be the id of a single AREA (a furniture node such as table_15 or
counter_24). It must never be a room id such as living_room_1 or kitchen_1, and never a list
of areas. When the instruction names a room, choose the one furniture in that room that best
fits the instruction rather than naming the room.

## For object states and spatial constraints:

1. Add a "relation" of "within" instead of "on" only when the instruction requires the object to go inside a container.

2. Add a "next_to" reference object only when the instruction requires the object to be placed next to another specific object.

3. Add "states" only when the instruction requires a state change of the object. Allowed state literals are: is_powered_on, is_powered_off, is_filled, is_empty, is_clean, is_dirty.

4. A goal may require only a state change and no relocation. In that case set "target_area" to null and give the required "states".

## For objects already in target areas:

1. These objects are checked by human that have been already placed in the target areas.

2. You should totally ignore these objects!!!!!

## For the final answer:

1. You must give out your reasoning process first. Then, you must give your final answer in json format same as the example json answer.

# Critical Rules (Must Read First)

1. Ignore objects already in target areas

- (Elaborated in Section 3 -> Step1. If an object's current area in the observation equals the designated target area, discard it from consideration.)

2. Cycle Control

- When returning to Section 1 for re-processing, do not reconsider objects that have already been blacklisted or discarded.

# Requirements: Your reasoning process should include the following sections (Section 1 to 5) and steps in each section. For each step, you should explicitly give out your reasoning and the phased results, and give out your final JSON answer at last.

## Section 1: OBJECTS OF INTEREST IDENTIFICATION

### Step1: OBJECTS OF INTEREST IDENTIFICATION

- Generate up to 10 objects of interest.

- Focus on "fresh" objects not in the blacklist.

- Pay less attention on objects in wrong goal states.

- The more time the object appeared in wrong goal states, the less attention you should pay to it, and the more probability you should add it to the blacklist.

- Example: wrong goal states are: 1. chamomile_tea in cabinet_5. Then, pay less attention to chamomile_tea, and consider adding it to the blacklist.

- Totally ignore the objects that have been already placed in the target areas.

- Example: Objects already in target areas: chamomile_tea, then you should totally ignore chamomile_tea, and move it to the blacklist.

- Object Deficit Resolution Protocol:

- If the observed objects are insufficient (<= instruction requirements),

-> Generate hypothetical (unseen) objects that:

1. Fit the instruction's context and patterns.

2. Pass semantic coherence checks.

3. Comply with resource constraints.

4. Do not duplicate blacklisted properties.

5. Fulfill missing capabilities in the current object pool.

## Section 2: TARGET AREA IDENTIFICATION

### Step1: CONTEXTUAL TARGETING

- Select the most probable target_area for each object.

- Target_area Identification Protocol:

- Fit the instruction's context and patterns.

- Reject common-sense conflicts (e.g., placing trash in the refrigerator).

- Prefer an area in the room that the instruction refers to, if it refers to one.

## Section 3: VALIDATION LOOP

### Step1: TARGET_AREA CHECK

- For each candidate object visible in the observation:

- If object in list that have been already placed in the target areas -> Discard this object entirely.

- If current area in observation == target_area -> Discard this object entirely.

- Otherwise, keep it in working memory.

### Step2: COMPLETENESS TEST

- If the remaining objects after discarding cannot fulfill the instruction:

- Add current candidates to the blacklist

- Return to Section 1 but exclude blacklisted objects in the next iteration.

## Section 4: COMBINATION GENERATION

### Step1: OBJECT-CENTRIC COMBINATION ENGINE

- Generate up to {k} object-only combinations from the pool of valid objects.

- No target_area assignments yet.

- Each combination must contain no more than {max_objects} objects.

- Prioritize logical groupings and auto-prune duplicates or redundant patterns.

### Step2: POST-HOC TARGET_AREA ASSIGNMENT

- For each combination from Step1:

1. Per-object resolution

- Select the highest-validity target_area option (per Section 2).

2. Cross-combination locking

- The first assignment chosen for an object -> target_area locks that mapping.

- Subsequent combinations must reuse the same mapping.

### Step3: CROSS-MATRIX VALIDATION

- Consistency Audit

- Check that every object consistently uses the same target_area in all generated combinations.

- Failure Modes

- If any target_area mismatch is detected, remove all conflicting combinations.

- If an object conflict arises, revisit Section 4 step 1 with penalty weighting.

#### Final Safeguards (Section 4)

1. Sequential Locking Protocol

- The first valid combination's object->target_area assignments bind subsequent combinations.

2. Retroactive Consistency

- Any new combinations must respect existing locked mappings.

3. Combination Quarantine

- Combinations involving any unvalidated object-target pair are kept aside until validated.

#### Example (Section 4)

Expected combinations:

- combination 1: object: blender, target_area: counter_24; object: cheese_grater, target_area: table_10

- combination 2: object: blender, target_area: counter_24; object: potato_peeler, target_area: shelves_3

Explanation: the same object in 2 combinations (blender) has the same target_area (counter_24). The combination of objects in 2 combinations are different (blender and cheese_grater, blender and potato_peeler)

Unexpected combinations:

- combination 1: object: blender, target_area: counter_24; object: cheese_grater, target_area: table_10

- combination 2: object: blender, target_area: shelves_3; object: cheese_grater, target_area: table_10

Explanation: the same object in 2 combinations (blender) has the different target_area (counter_24 and shelves_3). The combination of objects in 2 combinations are the same (blender and cheese_grater)

## Section 5: FINAL ANSWER

### Step1: RESULT AGGREGATION & VALIDATION

- Combine all validated combinations.

- Explicitly present the final set of object -> target_area mappings.

- Ensure 100% target-area consistency with the locked pairs.

### Step2: FINAL OUTPUT CERTIFICATION

- Only execute after successful validation of sections 1-4.

- Output the final JSON answer if:

1. All rules in sections 1-4 are satisfied.

2. Resource allocations remain within bounds.

- The probability values across all combinations must sum to 1.0.

- The final JSON answer should be the same format with Example output JSON data:

- Put your json data between ```json and ```

- Example output JSON data:

```json
{{
  "answer": [
    {{
      "objects": [
        {{"object": "apple", "target_area": "counter_24", "relation": "on", "next_to": null, "states": []}},
        {{"object": "banana", "target_area": "counter_24", "relation": "on", "next_to": null, "states": []}}
      ],
      "probability": 0.7
    }},
    {{
      "objects": [
        {{"object": "orange", "target_area": "fridge_0", "relation": "within", "next_to": null, "states": []}},
        {{"object": "banana", "target_area": "counter_24", "relation": "on", "next_to": null, "states": []}}
      ],
      "probability": 0.3
    }}
  ]
}}
```
"""


# ---------------------------------------------------------------------------
# Appendix A.1: system prompt for Level 3
# ---------------------------------------------------------------------------

TOH_LEVEL3_SYSTEM_PROMPT = """# Role

You are an expert assistant specialized in object relocation within household environments.

An AREA is a furniture node of the house, identified by an id such as table_0, counter_24,
fridge_0 or chest_of_drawers_2.

You'll receive:

1. the language instruction of the task.

2. the description of current observation of the environment, listing all open areas, closed areas and observed objects with their placements. If the object is inside a closed area, it is not visible.

3. the name of the object of interest you should now focus on.

# Task

You need to identify up to {k} most probable initial_areas for every missing object and their probability.

# Guidelines

## For the Initial_areas

1. The object's placement is consistent with common sense.

2. The initial_areas can only be selected from the areas explicitly mentioned in the observation.

## For the final answer:

1. You must give out your reasoning process first. Then, you must give your final answer in json format same as the example json answer.

Now, carefully read the following requirements, then step by step give your reasoning, and finally, generate your answer in JSON format.

# Requirements: Your reasoning process should include the following steps. For each step, you should explicitly give out your reasoning and the phased results.

## Step 1: Object Visibility Check

- Check whether the current object of interest is visible in the observation.

- If the object is visible, set the probability of the object placed in the area to 1.0, and jump to step 4 and give out the final answer.

- If the object's current area is the robot's hand, the selected area should be 'robot'.

## Step 2: Object initial_area Guess

- List all the closed areas in the observation, and all the areas that have not been inspected yet.

- Reason/Guess the up to {k} possible initial_areas for the current object of interest from those areas and corresponding probability using common sense.

## Step 3: Object Initial_Area Double Check

- Double check the initial_areas you proposed:

1. Every proposed area is a closed area, or an area that has not been inspected yet. An area that has already been inspected and did not contain the object cannot hold it.

2. The probability sum for all candidate areas must sum to 1.0 for each object.

- Return to step 2 if the double check fails.

## Step 4: Final Answer

- Give out your final JSON answer in the same format of Example Json Answer.

- Put your json data between ```json and ```

- Example Json Answer:

```json
{{
  "answer": [
    {{"initial_area": "chest_of_drawers_2", "probability": 0.7}},
    {{"initial_area": "fridge_0", "probability": 0.3}}
  ]
}}
```
"""


@dataclass
class TohConfig:
    """
    TOH hyperparameters. ``c1``/``c2`` are the paper's Appendix A.5 values.

    :param c1: number of Level 1/2 candidate combinations.
    :param c2: number of Level 3 candidate locations per invisible object.
    :param max_objects_per_combination: the paper's cap of 4 objects per goal.
    :param max_particles: cap on the number of root-to-leaf paths kept. The
        Cartesian product over Level 3 choices can reach c1 * c2 ** 4 leaves; the
        lowest-weight leaves are pruned and the belief renormalised.
    :param max_tokens: generation budget. The A.1 prompts ask for a long
        reasoning chain before the JSON, so this has to be generous.
    """

    c1: int = 3
    c2: int = 3
    max_objects_per_combination: int = 4
    max_particles: int = 48
    max_tokens: int = 4096


def _format_list(values: Iterable[str]) -> str:
    values = list(values)
    if not values:
        return "none"
    return ", ".join(values) + ","


def build_observation_text(
    observation: Observation,
    domain: SymbolicDomain,
    wrong_goal_states: Sequence[Sequence[GoalAtom]] = (),
    satisfied_objects: Sequence[str] = (),
    failed_attempts: Sequence[str] = (),
) -> str:
    """
    Build the textual observation T_z of Appendix A.1, extended with the two
    pieces of regeneration context the plan requires: which areas have been
    inspected, and which skill executions failed.
    """
    known = sorted(observation.known_furniture or domain.furniture_room.keys())
    closed = [f for f in known if not observation.furniture_open.get(f, True)]
    open_areas = [f for f in known if observation.furniture_open.get(f, True)]

    lines: List[str] = ["Current Observation:", ""]

    rooms: Dict[str, List[str]] = {}
    for furniture in known:
        room = domain.room_of(furniture)
        if room is not None:
            rooms.setdefault(room, []).append(furniture)
    if rooms:
        lines.append("The house layout is:")
        for room in sorted(rooms):
            lines.append(f"{room}: {', '.join(sorted(rooms[room]))}")
        lines.append("")

    lines.append(f"The closed areas are: {_format_list(closed)}")
    lines.append("")
    lines.append(f"The open areas are: {_format_list(open_areas)}")
    lines.append("")

    placements = []
    for obj in sorted(observation.object_parent):
        parent = observation.object_parent[obj]
        where = "the robot's hand" if parent == HELD else parent
        placements.append(f"{obj} is in {where}")
    lines.append(f"The observed objects and their initial areas are: {_format_list(placements)}")
    lines.append("")

    states = []
    for obj in sorted(observation.object_states):
        flags = observation.object_states[obj]
        if not flags:
            continue
        rendered = ", ".join(f"{k}={bool(v)}" for k, v in sorted(flags.items()))
        states.append(f"{obj}: {rendered}")
    if states:
        lines.append("The observed object states are: " + "; ".join(states))
        lines.append("")

    lines.append(
        f"The areas that have already been inspected are: "
        f"{_format_list(sorted(observation.fully_inspected_areas))}"
    )
    lines.append("")
    lines.append(
        f"The areas that have NOT been inspected yet are: "
        f"{_format_list([f for f in known if f not in observation.inspected_areas])}"
    )
    lines.append("")

    lines.append("The wrong goal states are:")
    lines.append("")
    if wrong_goal_states:
        for index, goal in enumerate(wrong_goal_states, start=1):
            rendered = ", ".join(str(atom) for atom in goal)
            lines.append(f"{index}. {rendered}.")
            lines.append("")
    else:
        lines.append("none")
        lines.append("")

    lines.append("The failed skill executions are:")
    lines.append("")
    if failed_attempts:
        for index, attempt in enumerate(failed_attempts, start=1):
            lines.append(f"{index}. {attempt}")
            lines.append("")
    else:
        lines.append("none")
        lines.append("")

    lines.append(f"Objects already in target areas: {_format_list(sorted(satisfied_objects))}")
    return "\n".join(lines)


def parse_json_answer(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the JSON object the A.1 prompts ask for.

    Tolerant on purpose: the prompts ask for a long reasoning chain first, and
    the fenced block is sometimes unlabelled or unterminated.
    """
    if not text:
        return None
    candidates: List[str] = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        candidates.append(match.group(1))
    tail = re.search(r"```(?:json)?\s*(\{.*)\Z", text, re.DOTALL)
    if tail is not None:
        candidates.append(tail.group(1))
    for start in (m.start() for m in re.finditer(r"\{", text)):
        candidates.append(_balanced_object(text, start))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    return None


def _balanced_object(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


class TreeOfHypotheses:
    """
    Queries an LLM for the three TOH levels and turns the result into particles.

    The LLM only needs a ``generate(prompt, stop=None, max_length=None)`` method,
    which keeps this class testable with a stub.
    """

    def __init__(
        self,
        llm: Any,
        domain: SymbolicDomain,
        config: Optional[TohConfig] = None,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.domain = domain
        self.config = config or TohConfig()
        self.verbose = verbose
        self.last_prompts: List[str] = []
        self.last_responses: List[str] = []

    # -- LLM plumbing --------------------------------------------------------

    def _query(self, prompt: str) -> str:
        self.last_prompts.append(prompt)
        response = self.llm.generate(
            prompt, stop=None, max_length=self.config.max_tokens
        )
        if not isinstance(response, str):
            response = str(response)
        self.last_responses.append(response)
        if self.verbose:
            print(f"[toh] raw response {len(self.last_responses)}:\n{response}\n")
        return response

    # -- level 1 & 2 ---------------------------------------------------------

    def query_levels_1_2(
        self, instruction: str, observation_text: str
    ) -> List[Tuple[List[GoalAtom], float]]:
        prompt = (
            TOH_LEVEL12_SYSTEM_PROMPT.format(
                k=self.config.c1,
                max_objects=self.config.max_objects_per_combination,
            )
            + "\n\n# Task instruction\n\n"
            + instruction
            + "\n\n"
            + observation_text
            + "\n\nNow give your reasoning and then the final JSON answer.\n"
        )
        parsed = parse_json_answer(self._query(prompt))
        if parsed is None:
            return []
        combinations: List[Tuple[List[GoalAtom], float]] = []
        for entry in parsed.get("answer", [])[: self.config.c1]:
            if not isinstance(entry, dict):
                continue
            atoms: List[GoalAtom] = []
            for item in entry.get("objects", [])[
                : self.config.max_objects_per_combination
            ]:
                atom = self._parse_goal_item(item)
                if atom is not None:
                    atoms.append(atom)
            if not atoms:
                continue
            try:
                probability = float(entry.get("probability", 0.0))
            except (TypeError, ValueError):
                probability = 0.0
            if probability <= 0.0:
                probability = 1e-3
            combinations.append((atoms, probability))
        return combinations

    def _parse_goal_item(self, item: Any) -> Optional[GoalAtom]:
        if not isinstance(item, dict) or not item.get("object"):
            return None
        name = str(item["object"]).strip().replace(" ", "_")
        target = item.get("target_area")
        area = self._resolve_area(target) if target else None
        if target and area is None:
            # The LLM invented a furniture id that does not exist in this house.
            return None
        relation = str(item.get("relation") or "on").strip().lower()
        if relation not in ("on", "within"):
            relation = "on"
        next_to = item.get("next_to")
        states = item.get("states") or []
        if isinstance(states, str):
            states = [states]
        states = tuple(str(s).strip() for s in states if str(s).strip())
        if area is None and not states:
            # Neither a placement nor a state change: nothing to plan for.
            return None
        return GoalAtom(
            obj=name,
            target_area=area,
            relation=relation,
            next_to=str(next_to) if next_to else None,
            states=states,
        )

    def _resolve_area(self, name: Any) -> Optional[str]:
        """Map an LLM-produced area name onto a real furniture id."""
        if not name:
            return None
        text = str(name).strip()
        if text in self.domain.furniture_room:
            return text
        lowered = text.lower()
        for furniture in self.domain.furniture_room:
            if furniture.lower() == lowered:
                return furniture
        wanted = set(normalize_object_name(text))
        best: Optional[str] = None
        best_overlap = 0
        for furniture in sorted(self.domain.furniture_room):
            overlap = len(wanted & set(normalize_object_name(furniture)))
            if overlap > best_overlap:
                best_overlap = overlap
                best = furniture
        return best

    # -- level 3 -------------------------------------------------------------

    def query_level_3(
        self,
        instruction: str,
        observation_text: str,
        obj: str,
        candidate_areas: Sequence[str],
    ) -> List[Tuple[str, float]]:
        prompt = (
            TOH_LEVEL3_SYSTEM_PROMPT.format(k=self.config.c2)
            + "\n\n# Task instruction\n\n"
            + instruction
            + "\n\n"
            + observation_text
            + f"\n\nThe object of interest you should now focus on is: {obj}\n"
            + "\nNow give your reasoning and then the final JSON answer.\n"
        )
        parsed = parse_json_answer(self._query(prompt))
        results: List[Tuple[str, float]] = []
        if parsed is not None:
            allowed = set(candidate_areas)
            for entry in parsed.get("answer", []):
                if not isinstance(entry, dict):
                    continue
                area = self._resolve_area(entry.get("initial_area"))
                if area is None or (allowed and area not in allowed):
                    continue
                try:
                    probability = float(entry.get("probability", 0.0))
                except (TypeError, ValueError):
                    probability = 0.0
                if probability <= 0.0:
                    probability = 1e-3
                results.append((area, probability))
                if len(results) >= self.config.c2:
                    break
        if not results and candidate_areas:
            # Never leave a hypothesised object without a location: spread the
            # mass uniformly over the plausible hidden areas.
            share = 1.0 / min(self.config.c2, len(candidate_areas))
            for area in list(candidate_areas)[: self.config.c2]:
                results.append((area, share))
        return results

    def candidate_hidden_areas(self, observation: Observation) -> List[str]:
        """
        Areas that could hide an unobserved object: closed containers, and areas
        that have not been inspected yet. This is the Habitat adaptation of the
        paper's "closed areas only" rule.
        """
        known = sorted(observation.known_furniture or self.domain.furniture_room.keys())
        closed = [f for f in known if not observation.furniture_open.get(f, True)]
        uninspected = [
            f
            for f in known
            if f not in observation.inspected_areas and f not in closed
        ]
        return closed + uninspected

    # -- particle construction ----------------------------------------------

    def generate(
        self,
        instruction: str,
        observation: Observation,
        wrong_goal_states: Sequence[Sequence[GoalAtom]] = (),
        satisfied_objects: Sequence[str] = (),
        failed_attempts: Sequence[str] = (),
    ) -> Belief:
        """
        Run the three levels and build the particle belief b_LLM.

        Each root-to-leaf path becomes one particle whose weight is the product of
        the confidences along the path; the belief is then normalised.
        """
        observation_text = build_observation_text(
            observation,
            self.domain,
            wrong_goal_states=wrong_goal_states,
            satisfied_objects=satisfied_objects,
            failed_attempts=failed_attempts,
        )
        combinations = self.query_levels_1_2(instruction, observation_text)
        if not combinations:
            return Belief()

        hidden_areas = self.candidate_hidden_areas(observation)
        # Level 3 is queried once per (object, combination-independent) basis: the
        # paper queries each invisible target object independently.
        location_cache: Dict[str, List[Tuple[str, float]]] = {}

        leaves: List[Tuple[Particle, float]] = []
        for atoms, combination_weight in combinations:
            invisible = [
                atom.obj
                for atom in atoms
                if atom.obj not in observation.object_parent
            ]
            unique_invisible = sorted(set(invisible))
            for obj in unique_invisible:
                if obj in location_cache:
                    continue
                grounded = self._observed_alias(obj, observation)
                if grounded is not None:
                    location_cache[obj] = [(observation.object_parent[grounded], 1.0)]
                    continue
                location_cache[obj] = self.query_level_3(
                    instruction, observation_text, obj, hidden_areas
                )

            option_lists = [location_cache.get(obj, []) for obj in unique_invisible]
            if any(len(options) == 0 for options in option_lists):
                # An object with no plausible location at all: keep the goal but
                # leave the object's location unknown.
                option_lists = [options or [(None, 1.0)] for options in option_lists]

            for assignment in itertools.product(*option_lists) if unique_invisible else [()]:
                weight = combination_weight
                placements: Dict[str, Optional[str]] = {}
                for obj, (area, probability) in zip(unique_invisible, assignment):
                    weight *= probability
                    placements[obj] = area
                particle = self._build_particle(
                    observation, atoms, placements, weight
                )
                leaves.append((particle, weight))

        leaves.sort(key=lambda pair: pair[1], reverse=True)
        leaves = leaves[: self.config.max_particles]
        belief = Belief(particle for particle, _ in leaves)
        return belief.normalize()

    def _observed_alias(self, obj: str, observation: Observation) -> Optional[str]:
        """If an observed object already matches this hypothesised name, use it."""
        for observed in sorted(observation.object_parent):
            if names_match(obj, observed):
                return observed
        return None

    def _build_particle(
        self,
        observation: Observation,
        atoms: Sequence[GoalAtom],
        hypothesised_placements: Dict[str, Optional[str]],
        weight: float,
    ) -> Particle:
        scene = SceneState(
            object_parent=dict(observation.object_parent),
            furniture_room=self.domain.furniture_room,
            furniture_open=dict(observation.furniture_open),
            object_states={k: dict(v) for k, v in observation.object_states.items()},
            spatial_relations={
                k: set(v) for k, v in observation.spatial_relations.items()
            },
            robot_area=observation.robot_area,
            inspected_areas=set(observation.inspected_areas),
        )
        resolved_atoms: List[GoalAtom] = []
        hypothesised: Set[str] = set()
        for atom in atoms:
            name = atom.obj
            if name in observation.object_parent:
                resolved_atoms.append(atom)
                continue
            alias = self._observed_alias(name, observation)
            if alias is not None:
                resolved_atoms.append(atom.rename(name, alias))
                continue
            # Hypothesised object: keep the TOH name distinct from every Habitat
            # id until an observation grounds it.
            area = hypothesised_placements.get(name)
            if area is not None:
                scene.object_parent[name] = area
            else:
                scene.object_parent[name] = None
            hypothesised.add(name)
            resolved_atoms.append(atom)
        scene.hypothesized = hypothesised
        return Particle(
            scene=scene, goal_atoms=tuple(resolved_atoms), weight=max(weight, 1e-9)
        )
