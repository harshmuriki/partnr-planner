#!/usr/bin/env python3

from dataset_generation.benchmark_generation.evaluation_generation.goal_state_propositions import (
    build_evaluation_from_goal_state,
)
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    TemporalConstraint,
    TerminalSatisfactionConstraint,
)


def test_on_top_within_and_clean():
    spawned = {"apple": ["Apple_12_:0000"], "orange": ["017_orange_:0000"]}
    furniture = {
        "table_4": "table_handle",
        "fridge_0": "fridge_handle",
    }
    rooms = {"living_room_0": "living_room", "kitchen_0": "kitchen"}
    goal = [
        {
            "number": 1,
            "object_classes": ["apple"],
            "furniture_names": ["fridge_0"],
            "allowed_regions": ["kitchen_0"],
            "location": "within",
        },
        {
            "number": 1,
            "object_classes": ["orange"],
            "furniture_names": ["table_4"],
            "allowed_regions": ["living_room_0"],
            "location": "on",
            "object_states": {"is_clean": True},
        },
    ]
    props, constraints, err = build_evaluation_from_goal_state(
        goal, spawned, furniture, rooms
    )
    assert err is None
    assert [p.function_name for p in props] == ["is_inside", "is_on_top", "is_clean"]
    assert props[0].args["object_handles"] == ["Apple_12_:0000"]
    assert props[0].args["receptacle_handles"] == ["fridge_handle"]
    assert props[1].args["object_handles"] == ["017_orange_:0000"]
    assert props[1].args["receptacle_handles"] == ["table_handle"]
    assert props[2].args["object_handles"] == ["017_orange_:0000"]
    assert isinstance(constraints[0], TemporalConstraint)
    assert isinstance(constraints[1], TerminalSatisfactionConstraint)
    assert constraints[1].proposition_indices == [0, 1, 2]


def test_missing_spawn_is_error():
    _, _, err = build_evaluation_from_goal_state(
        [
            {
                "number": 2,
                "object_classes": ["apple"],
                "furniture_names": ["table_4"],
                "allowed_regions": ["living_room_0"],
                "location": "on",
            }
        ],
        {"apple": ["Apple_12_:0000"]},
        {"table_4": "table_handle"},
        {"living_room_0": "living_room"},
    )
    assert err is not None
    assert "apple" in err


def test_phase_builds_temporal_edges():
    spawned = {"apple": ["a0"], "orange": ["o0"]}
    furniture = {"table_4": "t", "fridge_0": "f"}
    rooms = {"kitchen_0": "kitchen"}
    goal = [
        {
            "number": 1,
            "object_classes": ["apple"],
            "furniture_names": ["table_4"],
            "allowed_regions": ["kitchen_0"],
            "location": "on",
            "phase": 0,
        },
        {
            "number": 1,
            "object_classes": ["orange"],
            "furniture_names": ["fridge_0"],
            "allowed_regions": ["kitchen_0"],
            "location": "within",
            "phase": 1,
        },
    ]
    props, constraints, err = build_evaluation_from_goal_state(
        goal, spawned, furniture, rooms
    )
    assert err is None
    assert len(props) == 2
    dag_edges = constraints[0].args["dag_edges"]
    assert dag_edges == [(0, 1)]


def test_floor_and_in_room():
    props, _, err = build_evaluation_from_goal_state(
        [
            {
                "number": 1,
                "object_classes": ["apple"],
                "furniture_names": ["floor"],
                "allowed_regions": ["kitchen_0"],
                "location": "floor",
            }
        ],
        {"apple": ["Apple_12_:0000"]},
        {},
        {"kitchen_0": "kitchen"},
    )
    assert err is None
    assert [p.function_name for p in props] == ["is_on_floor", "is_in_room"]
    assert props[1].args["room_ids"] == ["kitchen"]


def test_in_alias_is_within():
    from dataset_generation.benchmark_generation.evaluation_generation.goal_state_propositions import (
        _normalize_location,
    )

    assert _normalize_location({"location": "in", "furniture_names": ["fridge_0"]}) == "within"
    assert _normalize_location({"location": "inside"}) == "within"
    assert _normalize_location({"location": "on"}) == "on"
