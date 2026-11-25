#!/usr/bin/env python3
# Tests for RailPlanner's overhead map (plan_config.overhead_map): objects come only from the robot's world graph and
# are drawn at their parent furniture/room (never at a node's translation), the explored mask, the text block, VLM
# image captions, and that overhead_map=none changes nothing. No sim, GPU or API.

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from habitat_llm.llm.openai_chat import generate_message
from habitat_llm.rail_planner import overhead_map as om
from habitat_llm.vlm_tamp.vlm_api import DEFAULT_EXTRA_IMAGE_LABEL, image_label

# ---------------------------------------------------------------------------
# Fakes: two rooms side by side (x in [0,4] and [4,8], z in [0,4]), three tables
# ---------------------------------------------------------------------------


def _static(label_all=False):
    mpp = 0.05
    nav = np.ones((int(4 / mpp), int(8 / mpp)), np.uint8)
    rooms = {
        "kitchen_1": om.RoomPoly("kitchen_1", [[0, 0], [4, 0], [4, 4], [0, 4]]),
        "living_room_1": om.RoomPoly("living_room_1", [[4, 0], [8, 0], [8, 4], [4, 4]]),
    }

    def box(cx, cz, w, d):
        return [[cx - w / 2, cz - d / 2], [cx + w / 2, cz - d / 2], [cx + w / 2, cz + d / 2], [cx - w / 2, cz + d / 2]]

    furniture = {
        "table_0": om.Footprint("table_0", box(1.0, 1.0, 1.0, 0.6), 0.75, room="kitchen_1"),
        "table_1": om.Footprint("table_1", box(6.0, 2.0, 1.6, 1.0), 0.45, room="living_room_1"),
        "shelves_0": om.Footprint("shelves_0", box(7.6, 0.4, 0.6, 0.4), 1.8, room="living_room_1",
                                  labelled=label_all),
    }
    return om.StaticMap(nav=nav, origin=(0.0, 0.0), mpp=mpp, rooms=rooms, furniture=furniture)


class _Node(SimpleNamespace):
    def __hash__(self):
        return hash(self.name)


class FakeGraph:
    """world graph with objects related to furniture/rooms; translations deliberately point elsewhere."""

    def __init__(self, objects, held=()):
        # objects: name -> (furniture name | None, room name | None, translation, sim_handle)
        self.nodes = {}
        self.parent = {}
        self.room = {}
        for name, (fur, room, tr, handle) in objects.items():
            self.nodes[name] = _Node(name=name, properties={"translation": tr}, sim_handle=handle)
            self.parent[name] = fur
            self.room[name] = room
        self.held = set(held)

    def get_all_objects(self):
        return list(self.nodes.values())

    def is_object_with_robot(self, obj):
        return obj.name in self.held

    def find_furniture_for_object(self, obj):
        fur = self.parent.get(obj.name)
        return _Node(name=fur) if fur else None

    def get_room_for_entity(self, entity):
        room = self.room.get(entity.name)
        if room is None:
            raise ValueError("no room")
        return _Node(name=room)


def _graph():
    return FakeGraph({
        # Translation far from table_0 (bottom-right corner): must not be used.
        "cup_0": ("table_0", "kitchen_1", [7.8, 1.0, 3.8], "cup_0_handle"),
        # Outdated memory ghost: translation falls back to the TRUE position; must be drawn at its parent.
        "plate_1": ("table_1", "living_room_1", [0.2, 1.0, 3.8], "plate_1.stale_memory"),
        # On the floor of the living room: drawn at the room centroid.
        "box_0": ("floor_living_room_1", "living_room_1", [0.2, 0.0, 0.2], "box_0_handle"),
        "book_0": (None, None, [1.0, 0.0, 1.0], "book_0_handle"),  # no anchor: not drawn
        "keys_0": (None, None, [3.0, 0.0, 3.0], "keys_0_handle"),  # held
    }, held={"keys_0"})


def _red_pixels(img, x, z, omap, radius_px=10):
    """Count dot-red pixels around world (x, z) in a render of omap."""
    x0, z0, x1, z1 = omap._crop_world()
    scale = int(omap.cfg["max_side_px"]) / max(x1 - x0, z1 - z0)
    X, Y = int((x - x0) * scale), int((z - z0) * scale)
    a = np.asarray(img).astype(int)
    patch = a[max(0, Y - radius_px):Y + radius_px, max(0, X - radius_px):X + radius_px]
    red = (patch[..., 0] > 200) & (patch[..., 1] < 60) & (patch[..., 2] < 90)
    return int(red.sum())


def _map(level, **cfg):
    m = om.OverheadMap(_static(), level, {"max_side_px": 800, **cfg})
    m.set_start(2.0, 3.0, 0.0, room="kitchen_1")
    m.observe_pose(5.0, 3.0, math.pi / 2)
    return m


# ---------------------------------------------------------------------------
# Objects: world graph only, anchored at the parent
# ---------------------------------------------------------------------------


def test_object_anchors_use_relations_not_translation():
    anchors = {name: (kind, anchor) for name, kind, anchor in om.object_anchors(_graph(), _static())}
    assert anchors == {
        "book_0": ("unknown", None),
        "box_0": ("room", "living_room_1"),
        "cup_0": ("furniture", "table_0"),
        "keys_0": ("robot", None),
        "plate_1": ("furniture", "table_1"),  # ghost: same treatment as any remembered object
    }


def test_observed_render_draws_objects_at_parent_only():
    m = _map("observed")
    img = m.render(_graph())
    assert _red_pixels(img, *m.static.furniture["table_0"].center, m) > 20
    assert _red_pixels(img, *m.static.furniture["table_1"].center, m) > 20
    # The nodes' own translations (one of them the ghost's true position) stay empty.
    assert _red_pixels(img, 7.8, 3.8, m) == 0
    assert _red_pixels(img, 0.2, 3.8, m) == 0


def test_objects_absent_from_graph_are_never_drawn():
    m = _map("observed")
    img_full = np.asarray(m.render(_graph())).astype(int)
    img_empty = np.asarray(m.render(FakeGraph({}))).astype(int)
    red = lambda a: int(((a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] < 90)).sum())  # noqa: E731
    assert red(img_full) > 0 and red(img_empty) == 0


def test_furniture_level_draws_no_objects():
    m = _map("furniture")
    img = np.asarray(m.render(_graph())).astype(int)
    assert int(((img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 90)).sum()) == 0
    assert "Red dots" not in m.image_caption()


def test_render_size_capped():
    img = _map("observed", max_side_px=640).render(_graph())
    assert max(img.size) <= 641


# ---------------------------------------------------------------------------
# Text block
# ---------------------------------------------------------------------------


def test_text_block_lists_geometry_and_start_room_but_no_objects():
    text = _map("observed").to_text(_graph())
    assert "You started this task in kitchen_1" in text
    assert "in living_room_1, facing down" in text  # yaw pi/2 = +z = down in the image
    assert "table_1 (living_room_1): (6.0, 2.0), 1.6x1.0 m, top 0.45 m, wall 2.0 m, room center 0.0 m" in text
    assert "shelves_0" not in text  # not a receptacle: unlabelled box, not listed
    for name in ("cup_0", "plate_1", "box_0", "keys_0", "book_0"):
        assert name not in text
    assert "seen" not in text


def test_text_block_truncates():
    text = _map("furniture", max_text_chars=200).to_text()
    assert len(text) < 260 and text.endswith("[map text truncated]")


# ---------------------------------------------------------------------------
# Explored mask (robot's own depth)
# ---------------------------------------------------------------------------


def test_depth_projection_marks_seen_floor():
    m = _map("explored")
    assert m.room_explored_fraction("living_room_1") == 0.0
    # Camera at (6, 1.0, 3.5) looking along -z (OpenGL camera looks down its -z axis), level with the floor.
    pose = np.eye(4)
    pose[:3, 3] = [6.0, 1.0, 3.5]
    depth = np.full((64, 64), 3.0, np.float32)
    assert m.wants_depth()
    new = m.observe_depth(depth, pose, math.radians(90))
    assert new > 0
    assert m.room_explored_fraction("living_room_1") > 0.1
    assert m.room_explored_fraction("kitchen_1") < m.room_explored_fraction("living_room_1")
    assert "seen " in m.to_text()
    assert "Darker floor" in m.image_caption()
    m.render(_graph())


def test_wants_depth_only_for_explored_level():
    m = _map("observed")
    assert not any(m.wants_depth() for _ in range(30))
    m = _map("explored", explored_every=5)
    hits = [m.wants_depth() for _ in range(10)]
    assert hits[0] is True  # first frame always projects
    m._depth_frames = 1
    assert [m.wants_depth() for _ in range(5)].count(True) == 1


def test_bad_level_rejected():
    with pytest.raises(ValueError):
        om.OverheadMap(_static(), "none")
    with pytest.raises(ValueError):
        om.OverheadMap(_static(), "gt_objects")


def test_geometry_helpers():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    area, c = om.polygon_area_centroid(sq)
    assert area == pytest.approx(4.0) and c == pytest.approx((1.0, 1.0))
    assert om.point_in_polygon(1, 1, sq) and not om.point_in_polygon(3, 1, sq)
    assert om.distance_to_boundary(0.5, 1.0, sq) == pytest.approx(0.5)
    assert _static().room_at(5, 1) == "living_room_1"


# ---------------------------------------------------------------------------
# VLM plumbing
# ---------------------------------------------------------------------------


def test_image_labels_default_unchanged():
    assert image_label(None, 0) is None
    assert image_label(None, 1) == DEFAULT_EXTRA_IMAGE_LABEL
    labels = [None, DEFAULT_EXTRA_IMAGE_LABEL, "[Top-down map]"]
    assert [image_label(labels, i) for i in range(3)] == labels


def test_gpt4v_sends_map_caption_before_map(monkeypatch, tmp_path):
    from habitat_llm.vlm_tamp import vlm_api

    sent = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        sent["content"] = json["messages"][-1]["content"]
        return Resp()

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("VLM_API_PROMPT_LOG_PATH", str(tmp_path / "vlm_api_prompts.txt"))
    monkeypatch.setattr(vlm_api.requests, "post", fake_post)
    api = vlm_api.GPT4vApi(model_name="gpt-4o")
    try:
        api._ask("p", image_data_urls=["a", "b", "m"], image_labels=[None, DEFAULT_EXTRA_IMAGE_LABEL, "[MAP]"])
    except Exception:
        pass  # response parsing details are not under test
    texts = [c.get("text") for c in sent["content"] if c["type"] == "text"]
    assert texts == ["p", DEFAULT_EXTRA_IMAGE_LABEL, "[MAP]"]
    sent.clear()
    try:
        api._ask("p", image_data_urls=["a", "b"])
    except Exception:
        pass
    assert [c.get("text") for c in sent["content"] if c["type"] == "text"] == ["p", DEFAULT_EXTRA_IMAGE_LABEL]


def test_generate_message_high_detail_only_for_map():
    msg = generate_message([("text", "t"), ("image", "cam"), ("image_high_detail", "map")], image_detail="low")
    details = [c["image_url"]["detail"] for c in msg["content"] if c["type"] == "image_url"]
    assert details == ["low", "high"]


# ---------------------------------------------------------------------------
# Planner integration (planners built without __init__)
# ---------------------------------------------------------------------------


def test_base_planner_has_no_extra_images():
    from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner

    p = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    assert p._extra_vlm_images(None, "turn1") == ([], [])


def test_rail_overhead_map_none_is_inert():
    from habitat_llm.rail_planner import RailPlanner

    p = RailPlanner.__new__(RailPlanner)
    p.overhead_map_level, p._omap, p._omap_failed = "none", None, False
    p._omap_observe({}, None, 0)  # does not touch the sim
    assert p._omap is None
    assert p._extra_vlm_images(None, "turn1") == ([], [])
    assert not p._omap_attached("turn1")


def test_rail_attaches_map_per_config(tmp_path):
    from habitat_llm.rail_planner import RailPlanner

    p = RailPlanner.__new__(RailPlanner)
    p.overhead_map_level, p.overhead_map_format, p._omap_failed, p._omap_index = "observed", "both", False, 0
    p.overhead_map_cfg = {**om.DEFAULT_CFG, "attach_to": ["turn1", "explorer"]}
    p._omap = _map("observed")
    p._world_graph = _graph()
    p._log_dir = str(tmp_path)
    p._get_log_dir = lambda: str(tmp_path)
    p.events = []
    p._log_event = p.events.append
    p._last_vlm_image_paths = ["vlm_images/vlm_input_0000.png"]
    urls, labels = p._extra_vlm_images(_graph(), "turn1")
    assert len(urls) == 1 and urls[0].startswith("data:image") and "Top-down map" in labels[0]
    assert p._last_vlm_image_paths[-1].startswith("overhead_map/map_0000_turn1")
    assert (tmp_path / "overhead_map" / "map_0000_turn1.png").exists()
    assert (tmp_path / "overhead_map" / "summary.json").exists()
    assert p._extra_vlm_images(_graph(), "selector") == ([], [])  # not in attach_to
    parts = p._omap_explorer_parts()
    assert [t for t, _ in parts] == ["text", "text", "image_high_detail"]
    p.overhead_map_format = "text"
    assert p._extra_vlm_images(_graph(), "turn1") == ([], [])
    assert [t for t, _ in p._omap_explorer_parts()] == ["text"]


def test_explorer_appends_map_parts():
    from habitat_llm.rail_planner.react_explorer import ExploreReactPlanner

    e = ExploreReactPlanner.__new__(ExploreReactPlanner)
    e.curr_prompt = "PROMPT"
    e._send_action_image_enabled = lambda: False
    assert e._llm_prompt_with_action_image({}) == "PROMPT"
    e.overhead_map_provider = lambda: [("text", "MAP"), ("image_high_detail", "url")]
    assert e._llm_prompt_with_action_image({}) == [("text", "PROMPT"), ("text", "MAP"), ("image_high_detail", "url")]


def _rail_with_map(tmp_path, fmt, attach=("turn1", "selector", "explorer")):
    from habitat_llm.rail_planner import RailPlanner

    p = RailPlanner.__new__(RailPlanner)
    p.overhead_map_level, p.overhead_map_format, p._omap_failed, p._omap_index = "observed", fmt, False, 0
    p.overhead_map_cfg = {**om.DEFAULT_CFG, "attach_to": list(attach)}
    p._omap = _map("observed")
    p._world_graph = _graph()
    p.two_phase = False
    p._get_log_dir = lambda: str(tmp_path)
    p._log_event = lambda e: None
    p._last_vlm_image_paths = []
    return p


def test_image_format_is_the_default():
    import yaml

    from habitat_llm.rail_planner import planner as rail_module

    conf = yaml.safe_load(open(os.path.join(os.path.dirname(rail_module.__file__),
                                            "../conf/planner/rail_planner.yaml")))
    assert conf["plan_config"]["overhead_map_format"] == "image"
    assert 'get("overhead_map_format", "image")' in open(rail_module.__file__).read()


def test_image_format_sends_no_map_text(tmp_path, monkeypatch):
    from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner

    monkeypatch.setattr(VlmTampPddlPlanner, "_build_english_prompt", lambda self, **kw: "BASE PROMPT")
    p = _rail_with_map(tmp_path, "image")
    assert p._build_english_prompt(goal="g") == "BASE PROMPT"  # Turn 1 text untouched
    urls, labels = p._extra_vlm_images(_graph(), "turn1")
    assert len(urls) == 1 and urls[0].startswith("data:image")
    assert "Top-down map" in labels[0] and "Furniture (room" not in labels[0]
    parts = p._omap_explorer_parts()
    assert [t for t, _ in parts] == ["text", "image_high_detail"]  # caption + image, no metric block
    assert "Furniture (room" not in parts[0][1]
    # Text and both still append the metric block (opt-in).
    for fmt in ("text", "both"):
        q = _rail_with_map(tmp_path, fmt)
        assert "Top-down map (metres" in q._build_english_prompt(goal="g")
