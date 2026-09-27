#!/usr/bin/env python3

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".claude" / "skills" / "spec-generation" / "scripts"))

from validate_spec import validate  # noqa: E402

SCENE_INFO = {
    "all_furniture": ["bed_0", "table_0"],
    "all_rooms": ["bedroom_0", "living_room_0"],
    "objects": ["jug", "lamp"],
    "furniture": {"bedroom_0": ["bed_0"], "living_room_0": ["table_0"]},
}
ALLOWED_CLASSES = ["jug", "lamp"]


def _clean_spec(world_furniture: str = "table_0") -> str:
    return f"""# T1-ACC-BASE

## Base task
Bring the jug to the table and turn off the lamp.

## Scene
- scene_id: scene_x
- rooms used: bedroom_0, living_room_0
- furniture used (id — room — catalog description):
  - bed_0 — bedroom_0 — Bed
  - table_0 — living_room_0 — Table

## Task instruction / prompt given
"Bring the jug to the table and turn off the lamp."

## Affected object(s)
- jug is the uncertainty target; lamp is a remaining task object.

## Entity registry
- jug_0: jug, uncertainty target
- lamp_0: lamp, remaining task object

## Uncertainty being tested
- Internal robot memory: Accurate

## Information supplied in instruction
- Everything is specified.

## Initial world state
- jug_0: on {world_furniture} (living_room_0), filled with water.
- lamp_0: on bed_0 (bedroom_0), powered on.

## Final expected world state
- jug_0: on table_0 (living_room_0), filled with water.
- lamp_0: on bed_0 (bedroom_0), powered off.

## Initial robot memory
- jug_0: on {world_furniture} (living_room_0), filled with water.
- lamp_0: on bed_0 (bedroom_0), powered on.

## Success criteria
- is_on_top(jug_0, table_0).
- is_filled(jug_0).
- is_powered_off(lamp_0).

## Spawn / planner notes
- Spawn jug on table_0, lamp on bed_0.
"""


def test_clean_spec_passes():
    result = validate(_clean_spec(), "T1-ACC-BASE", SCENE_INFO, ALLOWED_CLASSES)
    assert result.ok, [str(i) for i in result.errors]


def test_fabricated_furniture_id_is_flagged():
    spec = _clean_spec(world_furniture="table_99")
    result = validate(spec, "T1-ACC-BASE", SCENE_INFO, ALLOWED_CLASSES)
    assert not result.ok
    messages = " ".join(str(i) for i in result.errors)
    assert "table_99" in messages
    assert "not in the real scene catalog" in messages


def test_incomplete_variant_leaking_target_into_memory_is_flagged():
    spec = _clean_spec()
    spec = spec.replace("# T1-ACC-BASE", "# T1-INC-BASE")
    spec = spec.replace(
        "- Internal robot memory: Accurate", "- Internal robot memory: Incomplete"
    )
    # An INC spec must OMIT the uncertainty target (jug_0) from memory; this one still
    # includes it, which is exactly the leak the validator should catch.
    result = validate(spec, "T1-INC-BASE", SCENE_INFO, ALLOWED_CLASSES)
    assert not result.ok
    messages = " ".join(str(i) for i in result.errors)
    assert "INC variant must OMIT uncertainty target 'jug_0'" in messages


def test_outdated_variant_with_unchanged_memory_location_is_flagged():
    spec = _clean_spec()
    spec = spec.replace("# T1-ACC-BASE", "# T1-OUT-BASE")
    spec = spec.replace(
        "- Internal robot memory: Accurate", "- Internal robot memory: Outdated"
    )
    # An OUT spec must place the uncertainty target (jug_0) at a DIFFERENT furniture id in
    # memory than in the initial world state; this one leaves it unchanged (table_0 in both).
    result = validate(spec, "T1-OUT-BASE", SCENE_INFO, ALLOWED_CLASSES)
    assert not result.ok
    messages = " ".join(str(i) for i in result.errors)
    assert "OUT variant memory location for 'jug_0' must differ" in messages


def _localization_spec(axis: str, target_memory: str) -> str:
    spec = _clean_spec()
    spec = spec.replace("T1-ACC-BASE", f"T1-ACC-{axis}")
    axis_text = {
        "ROOM": "Localization: Room Known",
        "CAND": "Localization: Candidate Rooms",
    }[axis]
    spec = spec.replace("- Internal robot memory: Accurate", f"- Internal robot memory: Accurate\n- {axis_text}")
    spec = spec.replace(
        "- jug_0: on table_0 (living_room_0), filled with water.\n- lamp_0: on bed_0 (bedroom_0), powered on.\n\n## Success criteria",
        f"- jug_0: {target_memory}\n- lamp_0: on bed_0 (bedroom_0), powered on.\n\n## Success criteria",
    )
    return spec


def test_room_memory_accepts_actual_room_without_furniture():
    spec = _localization_spec("ROOM", "in_room living_room_0, is_filled, exact furniture not remembered")
    result = validate(spec, "T1-ACC-ROOM", SCENE_INFO, ALLOWED_CLASSES)
    assert result.ok, [str(i) for i in result.errors]
    assert not result.warnings


def test_room_memory_rejects_wrong_room():
    spec = _localization_spec("ROOM", "in_room bedroom_0, is_filled, exact furniture not remembered")
    result = validate(spec, "T1-ACC-ROOM", SCENE_INFO, ALLOWED_CLASSES)
    assert "must name its actual room" in " ".join(str(i) for i in result.errors)


def test_candidate_memory_accepts_actual_room_among_candidates():
    spec = _localization_spec(
        "CAND", "in one of bedroom_0, living_room_0, is_filled, actual room and furniture not remembered"
    )
    result = validate(spec, "T1-ACC-CAND", SCENE_INFO, ALLOWED_CLASSES)
    assert result.ok, [str(i) for i in result.errors]
    assert not result.warnings


def test_candidate_memory_rejects_missing_actual_room():
    spec = _localization_spec("CAND", "in one of bedroom_0, actual room and furniture not remembered")
    result = validate(spec, "T1-ACC-CAND", SCENE_INFO, ALLOWED_CLASSES)
    assert "must include its actual room" in " ".join(str(i) for i in result.errors)


def test_candidate_memory_rejects_fabricated_room():
    spec = _localization_spec("CAND", "in one of bedroom_0, basement_9, living_room_0")
    result = validate(spec, "T1-ACC-CAND", SCENE_INFO, ALLOWED_CLASSES)
    assert "room 'basement_9' is not a real room" in " ".join(str(i) for i in result.errors)


def test_room_memory_rejects_non_target_entity():
    spec = _localization_spec("ROOM", "in_room living_room_0")
    spec = spec.replace(
        "- lamp_0: on bed_0 (bedroom_0), powered on.\n\n## Success criteria",
        "- lamp_0: in_room bedroom_0\n\n## Success criteria",
    )
    result = validate(spec, "T1-ACC-ROOM", SCENE_INFO, ALLOWED_CLASSES)
    assert "only allowed for uncertainty targets" in " ".join(str(i) for i in result.errors)


def test_final_containment_can_target_registered_pickupable():
    spec = _clean_spec()
    spec = spec.replace(
        "- lamp_0: lamp, remaining task object",
        "- lamp_0: lamp, remaining task object\n- basket_0: basket, remaining task object",
    )
    spec = spec.replace(
        "## Final expected world state\n- jug_0: on table_0",
        "## Final expected world state\n- jug_0: within basket_0",
    )
    scene_info = {**SCENE_INFO, "objects": SCENE_INFO["objects"] + ["basket"]}
    result = validate(spec, "T1-ACC-BASE", scene_info, ALLOWED_CLASSES + ["basket"])
    assert result.ok, [str(i) for i in result.errors]


def test_final_containment_rejects_unregistered_destination():
    spec = _clean_spec().replace(
        "## Final expected world state\n- jug_0: on table_0",
        "## Final expected world state\n- jug_0: within basket_9",
    )
    result = validate(spec, "T1-ACC-BASE", SCENE_INFO, ALLOWED_CLASSES)
    assert "basket_9" in " ".join(str(i) for i in result.errors)
