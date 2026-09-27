#!/usr/bin/env python3

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    EvaluationConstraint,
    EvaluationProposition,
    TemporalConstraint,
    TerminalSatisfactionConstraint,
)

OBJECT_STATE_NEGATIONS = {
    "is_clean": "is_dirty",
    "is_filled": "is_empty",
    "is_powered_on": "is_powered_off",
    "is_dirty": "is_clean",
    "is_empty": "is_filled",
    "is_powered_off": "is_powered_on",
}

_ON = {"on", "ontop", "on_top"}
_WITHIN = {"within", "in", "inside"}
_FLOOR = {"floor"}
_ROOM = {"room", "in_room"}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _entry_number(entry: Dict[str, Any]) -> int:
    return int(entry.get("number", 1))


def _first_str(entry: Dict[str, Any], key: str) -> str:
    values = _as_list(entry.get(key, []))
    if not values:
        return ""
    return str(values[0]).strip()


def _normalize_location(entry: Dict[str, Any]) -> str:
    furniture = _first_str(entry, "furniture_names").lower()
    raw = str(entry.get("location", "on")).lower().strip().replace("-", "_")
    if raw in _WITHIN:
        return "within"
    if raw in _FLOOR or furniture == "floor":
        return "floor"
    if raw in _ROOM:
        return "in_room"
    if raw in _ON:
        return "on"
    return "on"


def _take_handles(
    queues: Dict[str, List[str]], object_class: str, count: int
) -> Tuple[List[str], Optional[str]]:
    available = queues.get(object_class, [])
    if len(available) < count:
        return [], (
            f"goal_state needs {count} '{object_class}' but only "
            f"{len(available)} were spawned in initial_state"
        )
    taken = available[:count]
    queues[object_class] = available[count:]
    return taken, None


def _object_state_predicate(name: str, value: Any) -> Optional[str]:
    if name not in OBJECT_STATE_NEGATIONS:
        return None
    if value is True:
        return name
    if value is False:
        return OBJECT_STATE_NEGATIONS[name]
    return None


def build_evaluation_from_goal_state(
    goal_state: List[Dict[str, Any]],
    spawned_by_class: Dict[str, List[str]],
    furniture_name_to_handle: Dict[str, str],
    room_name_to_id: Dict[str, str],
) -> Tuple[
    List[EvaluationProposition], List[EvaluationConstraint], Optional[str]
]:
    """Compile goal_state rows into PARTNR evaluation propositions.

    Each row consumes the next spawned instances of that object class (same
    order as initial_state sampling). Returns (propositions, constraints, error).
    """
    if not goal_state:
        return [], [], None

    queues = {cls: list(handles) for cls, handles in spawned_by_class.items()}
    propositions: List[EvaluationProposition] = []
    phase_to_indices: Dict[int, List[int]] = defaultdict(list)

    for entry in goal_state:
        object_class = _first_str(entry, "object_classes")
        if not object_class:
            return [], [], "goal_state entry is missing object_classes"
        count = _entry_number(entry)
        handles, err = _take_handles(queues, object_class, count)
        if err:
            return [], [], err

        furniture_name = _first_str(entry, "furniture_names")
        region_name = _first_str(entry, "allowed_regions")
        location = _normalize_location(entry)
        phase = int(entry.get("phase", 0))

        def _add_prop(prop: EvaluationProposition, _phase: int = phase) -> None:
            propositions.append(prop)
            phase_to_indices[_phase].append(len(propositions) - 1)

        if location == "in_room":
            if region_name not in room_name_to_id:
                return [], [], f"goal_state unknown room '{region_name}'"
            _add_prop(
                EvaluationProposition(
                    function_name="is_in_room",
                    args={
                        "object_handles": handles,
                        "room_ids": [room_name_to_id[region_name]],
                        "number": count,
                        "is_same_room": True,
                    },
                )
            )
        elif location == "floor":
            _add_prop(
                EvaluationProposition(
                    function_name="is_on_floor",
                    args={"object_handles": handles, "number": count},
                )
            )
            if region_name:
                if region_name not in room_name_to_id:
                    return [], [], f"goal_state unknown room '{region_name}'"
                _add_prop(
                    EvaluationProposition(
                        function_name="is_in_room",
                        args={
                            "object_handles": list(handles),
                            "room_ids": [room_name_to_id[region_name]],
                            "number": count,
                            "is_same_room": True,
                        },
                    )
                )
        elif location == "within":
            if furniture_name not in furniture_name_to_handle:
                return [], [], f"goal_state unknown furniture '{furniture_name}'"
            _add_prop(
                EvaluationProposition(
                    function_name="is_inside",
                    args={
                        "object_handles": handles,
                        "receptacle_handles": [
                            furniture_name_to_handle[furniture_name]
                        ],
                        "number": count,
                        "is_same_receptacle": True,
                    },
                )
            )
        else:
            if furniture_name not in furniture_name_to_handle:
                return [], [], f"goal_state unknown furniture '{furniture_name}'"
            _add_prop(
                EvaluationProposition(
                    function_name="is_on_top",
                    args={
                        "object_handles": handles,
                        "receptacle_handles": [
                            furniture_name_to_handle[furniture_name]
                        ],
                        "number": count,
                        "is_same_receptacle": True,
                    },
                )
            )

        for next_cls in _as_list(entry.get("next_to", [])):
            next_cls = str(next_cls).strip()
            ref_handles = spawned_by_class.get(next_cls, [])
            if not ref_handles:
                return [], [], (
                    f"goal_state next_to '{next_cls}' was not spawned in initial_state"
                )
            _add_prop(
                EvaluationProposition(
                    function_name="is_next_to",
                    args={
                        "entity_handles_a": list(handles),
                        "entity_handles_b": list(ref_handles),
                        "number": count,
                        "is_same_b": False,
                        "l2_threshold": 0.5,
                    },
                )
            )

        states = entry.get("object_states") or {}
        if isinstance(states, dict):
            for state_name, state_val in states.items():
                pred = _object_state_predicate(str(state_name), state_val)
                if pred is None:
                    continue
                _add_prop(
                    EvaluationProposition(
                        function_name=pred,
                        args={"object_handles": list(handles), "number": count},
                    )
                )

    n_props = len(propositions)
    if n_props == 0:
        return [], [], "goal_state produced no propositions"

    groups = [phase_to_indices[p] for p in sorted(phase_to_indices)]
    dag_edges: List[Tuple[int, int]] = []
    for gen_idx in range(1, len(groups)):
        for i in groups[gen_idx - 1]:
            for j in groups[gen_idx]:
                dag_edges.append((i, j))

    constraints: List[EvaluationConstraint] = [
        TemporalConstraint(dag_edges=dag_edges, n_propositions=n_props),
        TerminalSatisfactionConstraint(
            proposition_indices=list(range(n_props)), n_propositions=n_props
        ),
    ]
    return propositions, constraints, None


def evaluation_payload(
    propositions: List[EvaluationProposition],
    constraints: List[EvaluationConstraint],
) -> List[Dict[str, Any]]:
    return [
        {"function_name": p.function_name, "args": deepcopy(p.args)}
        for p in propositions
    ]
