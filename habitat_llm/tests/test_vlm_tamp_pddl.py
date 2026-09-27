#!/usr/bin/env python3

import pytest

from habitat_llm.pddlstream.problem import build_pddlstream_problem
from habitat_llm.planner.vlm_tamp_pddl_planner import (
    VlmTampPddlPlanner,
    _box_iou,
    _clip_and_accept_gt_box,
    _rgb_for_label,
    _suppress_similar_overlapping_boxes,
)
from habitat_llm.world_model import Furniture, Object, Room, SpotRobot
from habitat_llm.world_model.world_graph import WorldGraph


def _make_simple_graph():
    wg = WorldGraph()
    room = Room("kitchen", {"type": "room"})
    table = Furniture("kitchen table", {"type": "furniture"})
    mug = Object("mug", {"type": "object"})
    agent = SpotRobot("agent_0", {"type": "agent"})

    for node in [room, table, mug, agent]:
        wg.add_node(node)

    wg.add_edge(table, room, "inside", "contains")
    wg.add_edge(mug, table, "on", "under")
    wg.add_edge(agent, room, "inside", "contains")
    return wg


def test_world_graph_to_init_facts():
    wg = _make_simple_graph()
    goal = ("on", "mug", "kitchen_table")
    _problem, init, name_map, _reverse = build_pddlstream_problem(wg, 0, goal)

    assert ("room", "kitchen") in init
    assert ("furniture", "kitchen_table") in init
    assert ("object", "mug") in init
    assert ("agent", "agent_0") in init
    assert ("at", "agent_0", "kitchen") in init
    assert ("inroom", "kitchen_table", "kitchen") in init
    assert ("inroom", "mug", "kitchen") in init
    assert ("on", "mug", "kitchen_table") in init
    assert name_map["kitchen table"] == "kitchen_table"


def test_subgoal_to_pddl_goal_literal_sanitizes_names():
    wg = _make_simple_graph()
    _problem, _init, name_map, _reverse = build_pddlstream_problem(
        wg, 0, ("handempty", "agent_0")
    )

    planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    literal = planner._subgoal_to_goal_literal(
        "on(mug, kitchen table)", "agent_0", name_map
    )
    assert literal == ("on", "mug", "kitchen_table")


def test_build_objects_by_type_adds_room_grouped_visible_objects():
    wg = _make_simple_graph()
    planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)

    by_type = planner._build_objects_by_type(wg, visible_object_names={"mug"})
    assert by_type["furniture_by_room"]["kitchen"] == ["kitchen table"]
    assert by_type["objects_by_room"]["kitchen"] == ["mug"]

    by_type_none_visible = planner._build_objects_by_type(wg, visible_object_names=set())
    assert by_type_none_visible["objects_by_room"] == {}


def test_build_scene_description_filters_to_visible_entities():
    wg = _make_simple_graph()
    room = wg.get_node_from_name("kitchen")
    cabinet = Furniture(
        "cabinet_1",
        {"type": "furniture", "is_articulated": True, "is_open": False},
    )
    hidden_cabinet = Furniture(
        "cabinet_2",
        {"type": "furniture", "is_articulated": True, "is_open": True},
    )
    for furn in [cabinet, hidden_cabinet]:
        wg.add_node(furn)
        wg.add_edge(furn, room, "inside", "contains")

    planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    desc = planner._build_scene_description(
        wg,
        agent_uid=0,
        visible_entity_names={"mug", "cabinet_1"},
    )

    assert "Robot is in kitchen." in desc
    assert "Robot hand is empty." in desc
    assert "mug is on kitchen table." in desc
    assert "cabinet_1 is closed." in desc
    assert "cabinet_2 is open." not in desc


def test_clip_and_accept_gt_box_drops_mostly_offscreen_furniture():
    img_w, img_h = 640, 480
    accepted = _clip_and_accept_gt_box(40, 40, 200, 180, img_w, img_h)
    assert accepted is not None

    # Tiny corner of a huge off-screen AABB clipping the image.
    rejected = _clip_and_accept_gt_box(-4000, -3000, 20, 20, img_w, img_h)
    assert rejected is None


def test_suppress_similar_overlapping_boxes_keeps_nested_furniture():
    red = (255, 0, 0)
    bed = ("bed_21", red, 50.0, 40.0, 500.0, 400.0)
    nightstand = ("table_25", red, 60.0, 200.0, 140.0, 280.0)
    chair_a = ("chair_17", red, 400.0, 200.0, 520.0, 360.0)
    chair_b = ("chair_23", red, 410.0, 210.0, 515.0, 350.0)
    mug = ("mug_1", red, 80.0, 90.0, 100.0, 110.0)

    kept = _suppress_similar_overlapping_boxes(
        [bed, nightstand, chair_a, chair_b, mug],
        movable_names={"mug_1"},
    )
    names = {box[0] for box in kept}
    assert "bed_21" in names
    assert "table_25" in names
    assert "mug_1" in names
    assert ("chair_17" in names) ^ ("chair_23" in names)


def test_box_iou_identical_and_disjoint():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 30.0, 30.0)
    assert _box_iou(a, a) == pytest.approx(1.0)
    assert _box_iou(a, b) == 0.0


# Unclipped AABBs that reproduce the two failure modes: off-screen house
# furniture clipped onto the image edge, and near-duplicate chairs stacked
# on one object. Coordinates are for a 512x512 third-person frame.
_TEST_RAW_BOXES = [
    ("washer_dryer_12", -2800.0, -2200.0, 48.0, 90.0),
    ("cabinet_31", -900.0, 250.0, 35.0, 400.0),
    ("bed_21", 110.0, 140.0, 410.0, 340.0),
    ("table_25", 95.0, 205.0, 165.0, 275.0),
    ("table_24", 340.0, 205.0, 410.0, 275.0),
    ("bench_44", 175.0, 285.0, 325.0, 335.0),
    ("chair_17", 385.0, 245.0, 500.0, 405.0),
    ("chair_23", 395.0, 255.0, 495.0, 395.0),
    ("stool_31", 400.0, 280.0, 470.0, 360.0),
    ("table_59", 410.0, 240.0, 475.0, 300.0),
    ("mug_1", 200.0, 250.0, 228.0, 278.0),
]
_TEST_MOVABLE_NAMES = {"mug_1"}
_TEST_IMAGE_SIZE = 512


def _old_style_clip_boxes(raw_boxes, img_w, img_h):
    """Old collector: if any part of the AABB hits the image, clip it on-screen."""
    margin = 5
    out = []
    for name, u_min, v_min, u_max, v_max in raw_boxes:
        hits = not (
            u_max < -margin
            or v_max < -margin
            or u_min > img_w + margin
            or v_min > img_h + margin
        )
        if not hits:
            continue
        out.append(
            (
                name,
                _rgb_for_label(name),
                float(max(0.0, min(img_w - 1, u_min))),
                float(max(0.0, min(img_h - 1, v_min))),
                float(max(0.0, min(img_w - 1, u_max))),
                float(max(0.0, min(img_h - 1, v_max))),
            )
        )
    return out


def _new_style_filter_boxes(raw_boxes, img_w, img_h, movable_names):
    accepted = []
    for name, u_min, v_min, u_max, v_max in raw_boxes:
        is_movable = name in movable_names
        clipped = _clip_and_accept_gt_box(
            u_min,
            v_min,
            u_max,
            v_max,
            img_w,
            img_h,
            min_side_px=8.0 if is_movable else 12.0,
            min_on_screen_fraction=0.15 if is_movable else 0.3,
        )
        if clipped is None:
            continue
        accepted.append((name, _rgb_for_label(name), *clipped))
    return _suppress_similar_overlapping_boxes(
        accepted, movable_names=movable_names
    )


def _draw_room_background(img_w, img_h):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (img_w, img_h), (46, 78, 84))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 300, img_w, img_h], fill=(110, 78, 48))
    draw.rectangle([118, 155, 392, 322], fill=(28, 28, 32))
    draw.rectangle([125, 148, 385, 200], fill=(236, 232, 224))
    draw.rectangle([98, 208, 160, 276], fill=(42, 36, 32))
    draw.rectangle([348, 208, 412, 276], fill=(42, 36, 32))
    draw.rectangle([176, 288, 328, 336], fill=(50, 42, 38))
    draw.rectangle([388, 248, 498, 408], fill=(228, 226, 220))
    return img


def render_gt_bbox_filter_test_image(output_path: str) -> str:
    """Write a before/after VLM overlay image using the real box drawer."""
    from PIL import Image, ImageDraw, ImageFont

    img_w = img_h = _TEST_IMAGE_SIZE
    before_boxes = _old_style_clip_boxes(_TEST_RAW_BOXES, img_w, img_h)
    after_boxes = _new_style_filter_boxes(
        _TEST_RAW_BOXES, img_w, img_h, _TEST_MOVABLE_NAMES
    )

    planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    left = _draw_room_background(img_w, img_h)
    right = _draw_room_background(img_w, img_h)
    planner._draw_gt_box_list_on_image(left, before_boxes)
    planner._draw_gt_box_list_on_image(right, after_boxes)

    header = 48
    gap = 16
    canvas = Image.new(
        "RGB", (img_w * 2 + gap, img_h + header), (18, 18, 18)
    )
    canvas.paste(left, (0, header))
    canvas.paste(right, (img_w + gap, header))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except Exception:
        font = ImageFont.load_default()
    before_names = ", ".join(b[0] for b in before_boxes)
    after_names = ", ".join(b[0] for b in after_boxes)
    draw.text(
        (8, 6),
        f"BEFORE ({len(before_boxes)}): clip-all  |  {before_names}",
        fill=(235, 235, 235),
        font=font,
    )
    draw.text(
        (img_w + gap + 8, 6),
        f"AFTER ({len(after_boxes)}): filtered  |  {after_names}",
        fill=(235, 235, 235),
        font=font,
    )
    canvas.save(output_path)
    return output_path


def test_render_gt_bbox_filter_test_image(tmp_path):
    out = tmp_path / "vlm_gt_bbox_filter_test.png"
    render_gt_bbox_filter_test_image(str(out))
    assert out.is_file()
    assert out.stat().st_size > 1000
    before = _old_style_clip_boxes(_TEST_RAW_BOXES, 512, 512)
    after = _new_style_filter_boxes(
        _TEST_RAW_BOXES, 512, 512, _TEST_MOVABLE_NAMES
    )
    before_names = {b[0] for b in before}
    after_names = {b[0] for b in after}
    assert "washer_dryer_12" in before_names
    assert "cabinet_31" in before_names
    assert "washer_dryer_12" not in after_names
    assert "cabinet_31" not in after_names
    assert "bed_21" in after_names
    assert "table_25" in after_names
    assert "mug_1" in after_names
    chair_kept = [n for n in ("chair_17", "chair_23") if n in after_names]
    assert len(chair_kept) <= 1
