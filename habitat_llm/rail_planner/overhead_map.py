"""Overhead (top-down) map of the house for RailPlanner (plan_config.overhead_map).

Levels (docs/overhead_map_design.md):
  none      - off (default); nothing in this module runs.
  furniture - floor plan: navigable floor, room outlines + names, static furniture footprints labelled with the
              world-graph IDs, robot pose/heading, START pose and the path driven so far.
  observed  - furniture + the objects in the robot's own world graph.
  explored  - observed + shading of the floor the robot has not seen yet (built from its own depth camera).

What comes from simulator ground truth, and what does not:
  * Scene structure only: the navmesh, HSSD room polygons, and the oriented footprints / top heights of furniture
    that the text world graph already lists. This stands in for a prior mapping run.
  * Objects never: object positions are NOT read from the simulator or from the nodes' `translation`. Each object in
    the planner's world graph is drawn at the furniture (or room) it is related to in that graph, so the
    acc / inc / out memory conditions keep their meaning. Outdated memory ghosts look exactly like other objects.
  * Robot pose and depth frames are the robot's own sensing.

The geometry/rendering half (StaticMap, OverheadMap) is pure numpy/PIL so it can be unit-tested without Habitat;
the sim reads are in build_static_map / read_robot_pose / read_jaw_depth.
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LEVELS = ("none", "furniture", "observed", "explored")
FORMATS = ("image", "text", "both")
ATTACH_POINTS = ("turn1", "selector", "explorer")

DEFAULT_CFG = {
    "meters_per_pixel": 0.05,
    "max_side_px": 1024,
    "label": "receptacles",  # receptacles | all
    "explored_every": 10,  # sim steps between depth projections (explored level only)
    "max_depth_m": 6.0,
    "max_text_chars": 12000,
    "attach_to": list(ATTACH_POINTS),
}

_EXCLUDED_ROOM_PREFIXES = ("outdoor", "porch", "unknown")
_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def polygon_area_centroid(poly: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Shoelace area and area-weighted centroid of an (N, 2) polygon in (x, z)."""
    x, z = poly[:, 0], poly[:, 1]
    x1, z1 = np.roll(x, -1), np.roll(z, -1)
    cross = x * z1 - x1 * z
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return 0.0, (float(x.mean()), float(z.mean()))
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cz = ((z + z1) * cross).sum() / (6.0 * a)
    return abs(float(a)), (float(cx), float(cz))


def point_in_polygon(x: float, z: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        xa, za = poly[i]
        xb, zb = poly[(i + 1) % n]
        if (za > z) != (zb > z):
            xi = xa + (z - za) * (xb - xa) / (zb - za)
            if x < xi:
                inside = not inside
    return inside


def distance_to_boundary(x: float, z: float, poly: np.ndarray) -> float:
    p = np.array([x, z], dtype=float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    denom = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.sqrt(((proj - p) ** 2).sum(axis=1)).min())


@dataclass
class RoomPoly:
    name: str
    poly: np.ndarray  # (N, 2) world (x, z)
    area: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        self.poly = np.asarray(self.poly, dtype=float)
        self.area, self.centroid = polygon_area_centroid(self.poly)

    def contains(self, x: float, z: float) -> bool:
        return point_in_polygon(x, z, self.poly)


@dataclass
class Footprint:
    name: str
    corners: np.ndarray  # (4, 2) world (x, z), oriented box
    top_height: float  # top surface above the floor, metres (nan if unknown)
    room: Optional[str] = None
    labelled: bool = True

    def __post_init__(self):
        self.corners = np.asarray(self.corners, dtype=float)

    @property
    def center(self) -> Tuple[float, float]:
        c = self.corners.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def size(self) -> Tuple[float, float]:
        e0 = float(np.linalg.norm(self.corners[1] - self.corners[0]))
        e1 = float(np.linalg.norm(self.corners[2] - self.corners[1]))
        return max(e0, e1), min(e0, e1)


@dataclass
class StaticMap:
    """Scene-static layers, built once per episode."""

    nav: np.ndarray  # (H, W) uint8, 1 = navigable; row = z, col = x
    origin: Tuple[float, float]  # world (x, z) of grid cell (0, 0)'s corner
    mpp: float
    rooms: Dict[str, RoomPoly]
    furniture: Dict[str, Footprint]
    room_masks: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (H, W) bool

    def __post_init__(self):
        if not self.room_masks:
            self.room_masks = {name: self._rasterize(r.poly) for name, r in self.rooms.items()}

    def world_to_grid(self, x, z):
        return (np.asarray(z) - self.origin[1]) / self.mpp, (np.asarray(x) - self.origin[0]) / self.mpp

    def _rasterize(self, poly: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageDraw

        H, W = self.nav.shape
        img = Image.new("L", (W, H), 0)
        rows, cols = self.world_to_grid(poly[:, 0], poly[:, 1])
        ImageDraw.Draw(img).polygon([(float(c), float(r)) for r, c in zip(rows, cols)], fill=1)
        return np.asarray(img, dtype=bool)

    def room_at(self, x: float, z: float) -> Optional[str]:
        best = None
        for name, room in self.rooms.items():
            if room.contains(x, z) and (best is None or room.area < self.rooms[best].area):
                best = name
        return best


# ---------------------------------------------------------------------------
# Sim reads (scene structure + robot's own sensing only)
# ---------------------------------------------------------------------------


def _region_polygon(region) -> Optional[np.ndarray]:
    pts = getattr(region, "poly_loop_points", None)
    if pts is None:
        return None
    pts = [[float(p[0]), float(p[1])] for p in pts]
    return np.asarray(pts) if len(pts) >= 3 else None


def build_static_map(env_interface, world_graph, agent_uid: int, cfg: Dict[str, Any]) -> StaticMap:
    """Navmesh grid, room polygons and furniture footprints. Reads no object state."""
    import magnum as mn
    from habitat.sims.habitat_simulator import sim_utilities as sutils

    sim = env_interface.sim
    mpp = float(cfg.get("meters_per_pixel", 0.05))
    robot = sim.agents_mgr[agent_uid].articulated_agent
    floor_y = float(robot.base_pos[1])
    nav = np.asarray(sim.pathfinder.get_topdown_view(meters_per_pixel=mpp, height=floor_y), dtype=np.uint8)
    lo, _hi = sim.pathfinder.get_bounds()
    origin = (float(lo[0]), float(lo[2]))

    id2name = dict(getattr(getattr(env_interface, "perception", None), "region_id_to_name", {}) or {})
    rooms: Dict[str, RoomPoly] = {}
    for region in sim.semantic_scene.regions:
        poly = _region_polygon(region)
        if poly is None:
            continue
        fh = float(getattr(region, "floor_height", floor_y))
        if math.isfinite(fh) and abs(fh - floor_y) > 1.5:
            continue  # another floor
        name = id2name.get(region.id) or region.category.name()
        if name in rooms:  # several polygons for one room: keep the largest
            if polygon_area_centroid(poly)[0] <= rooms[name].area:
                continue
        rooms[name] = RoomPoly(name, poly)

    receptacle_furniture = set()
    try:
        for rec in world_graph.get_all_receptacles():
            try:
                receptacle_furniture.add(world_graph.find_furniture_for_receptacle(rec).name)
            except Exception:
                pass
    except Exception:
        pass
    label_all = str(cfg.get("label", "receptacles")) == "all" or not receptacle_furniture
    furn_room = {f.name: r.name for f, r in world_graph.get_furniture_to_room_map().items()}

    furniture: Dict[str, Footprint] = {}
    for fur in world_graph.get_all_furnitures():
        if fur.name.startswith("floor_") or not getattr(fur, "sim_handle", None):
            continue
        try:
            obj = sutils.get_obj_from_handle(sim, fur.sim_handle)
            bb = obj.aabb if hasattr(obj, "aabb") else obj.root_scene_node.cumulative_bb
            T = obj.transformation
            pts = [
                T.transform_point(mn.Vector3(x, bb.min[1], z))
                for x, z in ((bb.min[0], bb.min[2]), (bb.max[0], bb.min[2]),
                             (bb.max[0], bb.max[2]), (bb.min[0], bb.max[2]))
            ]
            corners = [[float(p[0]), float(p[2])] for p in pts]
            top = max(
                float(T.transform_point(mn.Vector3(x, bb.max[1], z))[1])
                for x in (bb.min[0], bb.max[0]) for z in (bb.min[2], bb.max[2])
            ) - floor_y
        except Exception:
            continue  # no footprint: the furniture stays text-only (world description)
        fp = Footprint(fur.name, corners, top, room=furn_room.get(fur.name),
                       labelled=label_all or fur.name in receptacle_furniture)
        if fp.room is None:
            fp.room = next((n for n, r in rooms.items() if r.contains(*fp.center)), None)
        furniture[fur.name] = fp
    return StaticMap(nav=nav, origin=origin, mpp=mpp, rooms=rooms, furniture=furniture)


def read_robot_pose(env_interface, agent_uid: int) -> Tuple[float, float, float]:
    """(x, z, yaw) with yaw = atan2(forward_z, forward_x); Spot's forward is local +x."""
    import magnum as mn

    agent = env_interface.sim.agents_mgr[agent_uid].articulated_agent
    p = agent.base_pos
    fwd = agent.base_transformation.transform_vector(mn.Vector3(1, 0, 0))
    return float(p[0]), float(p[2]), math.atan2(float(fwd[2]), float(fwd[0]))


def read_jaw_depth(env_interface, observations, agent_uid: int):
    """(depth HxW metres, camera-to-world 4x4, hfov radians) from this step's observations, or None."""
    if not observations:
        return None
    keys = [k for k in observations if str(k).endswith("articulated_agent_jaw_depth")]
    prefixed = [k for k in keys if str(k).startswith(f"agent_{agent_uid}_")]
    key = (prefixed or keys or [None])[0]
    if key is None:
        return None
    depth = observations[key]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        return None
    sensors = env_interface.sim.agents[0]._sensors
    sensor = sensors.get(key) if hasattr(sensors, "get") else sensors[key]
    if sensor is None:
        return None
    pose = np.linalg.inv(np.asarray(sensor.render_camera.camera_matrix))
    hfov = math.radians(float(sensor.specification().hfov))
    return depth, pose, hfov


# ---------------------------------------------------------------------------
# Object anchors: world-graph relations only, never positions
# ---------------------------------------------------------------------------


def object_anchors(world_graph, static: StaticMap):
    """[(object, anchor_kind, anchor_name)] for every object in the planner's world graph.

    anchor_kind is 'robot' (held), 'furniture' (parent furniture has a footprint), 'room' (parent is a floor or
    furniture without footprint; drawn at the room centroid) or 'unknown' (not drawn). The node's `translation`
    is never used: for outdated memory ghosts it can fall back to the true object's position.
    """
    out = []
    for obj in world_graph.get_all_objects():
        try:
            if world_graph.is_object_with_robot(obj):
                out.append((obj.name, "robot", None))
                continue
        except Exception:
            pass
        fur = None
        try:
            fur = world_graph.find_furniture_for_object(obj)
        except Exception:
            fur = None
        if fur is not None and fur.name in static.furniture:
            out.append((obj.name, "furniture", fur.name))
            continue
        room = None
        if fur is not None:
            room = static.furniture[fur.name].room if fur.name in static.furniture else None
            if room is None:
                try:
                    room = world_graph.get_room_for_entity(fur).name
                except Exception:
                    room = None
        if room is None:
            try:
                room = world_graph.get_room_for_entity(obj).name
            except Exception:
                room = None
        if room is not None and room in static.rooms:
            out.append((obj.name, "room", room))
        else:
            out.append((obj.name, "unknown", None))
    return sorted(out)


# ---------------------------------------------------------------------------
# Map state + rendering
# ---------------------------------------------------------------------------


def _direction_words(yaw: float) -> str:
    # Image right = +x, image down = +z; yaw = atan2(dz, dx) is the angle in image coordinates.
    names = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
    return names[int(round(yaw / (math.pi / 4))) % 8]


def _load_fonts():
    from PIL import ImageFont

    try:
        from matplotlib import font_manager

        regular = font_manager.findfont("DejaVu Sans")
        bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
        return ImageFont.truetype(regular, 12), ImageFont.truetype(bold, 14)
    except Exception:
        f = ImageFont.load_default()
        return f, f


class OverheadMap:
    def __init__(self, static: StaticMap, level: str, cfg: Optional[Dict[str, Any]] = None):
        if level not in LEVELS or level == "none":
            raise ValueError(f"overhead_map level must be one of {LEVELS[1:]}, got {level!r}")
        self.static = static
        self.level = level
        self.cfg = {**DEFAULT_CFG, **dict(cfg or {})}
        self.start: Optional[Tuple[float, float, float]] = None
        self.start_room: Optional[str] = None
        self.pose: Optional[Tuple[float, float, float]] = None
        self.trail: List[Tuple[float, float]] = []
        self.explored = np.zeros_like(static.nav, dtype=bool)
        self._steps = 0
        self._depth_frames = 0
        self.depth_time_s = 0.0
        self.render_time_s = 0.0
        self.renders = 0
        self._fonts = None

    # ---- state ---------------------------------------------------------

    def set_start(self, x: float, z: float, yaw: float, room: Optional[str] = None) -> None:
        self.start = (x, z, yaw)
        self.start_room = room or self.static.room_at(x, z)
        self.observe_pose(x, z, yaw)

    def observe_pose(self, x: float, z: float, yaw: float) -> None:
        self.pose = (x, z, yaw)
        if not self.trail or math.hypot(x - self.trail[-1][0], z - self.trail[-1][1]) > 0.1:
            self.trail.append((x, z))

    def wants_depth(self) -> bool:
        """True on steps where the explored level should project a depth frame."""
        self._steps += 1
        if self.level != "explored":
            return False
        every = max(1, int(self.cfg.get("explored_every", 10)))
        return self._depth_frames == 0 or self._steps % every == 0

    def observe_depth(self, depth: np.ndarray, cam_to_world: np.ndarray, hfov: float, stride: int = 4) -> int:
        """Mark grid cells along the depth rays (free space + hit) as seen. Returns the number of new cells."""
        t0 = time.perf_counter()
        H, W = depth.shape
        f = (W / 2.0) / math.tan(hfov / 2.0)
        vs, us = np.mgrid[0:H:stride, 0:W:stride]
        d = depth[vs, us]
        ok = d > 0.05
        if not ok.any():
            return 0
        d = np.clip(d[ok], 0.0, float(self.cfg.get("max_depth_m", 6.0)))
        u, v = us[ok], vs[ok]
        cam = np.stack([(u - W / 2.0) / f * d, -(v - H / 2.0) / f * d, -d])  # OpenGL camera frame
        R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
        hits = (R @ cam).T + t
        ts = np.linspace(0.0, 1.0, 24)[:, None, None]
        pts = (t[None, None, :] + ts * (hits[None] - t[None, None, :])).reshape(-1, 3)
        rows, cols = self.static.world_to_grid(pts[:, 0], pts[:, 2])
        rows, cols = rows.astype(int), cols.astype(int)
        Hg, Wg = self.explored.shape
        m = (rows >= 0) & (rows < Hg) & (cols >= 0) & (cols < Wg)
        before = int(self.explored.sum())
        new = np.zeros_like(self.explored)
        new[rows[m], cols[m]] = True
        # Dilate by one cell: the sampled rays are sparse.
        grown = new.copy()
        grown[1:, :] |= new[:-1, :]
        grown[:-1, :] |= new[1:, :]
        grown[:, 1:] |= new[:, :-1]
        grown[:, :-1] |= new[:, 1:]
        self.explored |= grown
        self._depth_frames += 1
        self.depth_time_s += time.perf_counter() - t0
        return int(self.explored.sum()) - before

    def room_explored_fraction(self, name: str) -> Optional[float]:
        mask = self.static.room_masks.get(name)
        if mask is None:
            return None
        floor = mask & (self.static.nav > 0)
        if not floor.any():
            floor = mask
        if not floor.any():
            return None
        return float((self.explored & floor).sum()) / float(floor.sum())

    # ---- crop / transform ---------------------------------------------

    def _crop_world(self) -> Tuple[float, float, float, float]:
        pts = []
        for name, room in self.static.rooms.items():
            if not name.lower().startswith(_EXCLUDED_ROOM_PREFIXES):
                pts.append(room.poly)
        for fp in self.static.furniture.values():
            if fp.labelled:
                pts.append(fp.corners)
        for p in (self.start, self.pose):
            if p is not None:
                pts.append(np.array([[p[0], p[1]]]))
        if pts:
            allp = np.concatenate(pts, axis=0)
            x0, z0 = allp.min(axis=0) - 0.5
            x1, z1 = allp.max(axis=0) + 0.5
        else:
            H, W = self.static.nav.shape
            x0, z0 = self.static.origin
            x1, z1 = x0 + W * self.static.mpp, z0 + H * self.static.mpp
        return float(x0), float(z0), float(x1), float(z1)

    # ---- rendering -----------------------------------------------------

    def render(self, world_graph):
        from PIL import Image, ImageDraw

        t0 = time.perf_counter()
        st = self.static
        x0, z0, x1, z1 = self._crop_world()
        max_side = int(self.cfg.get("max_side_px", 1024))
        scale = max_side / max(x1 - x0, z1 - z0)  # px per metre
        W, H = max(1, int(round((x1 - x0) * scale))), max(1, int(round((z1 - z0) * scale)))

        def px(x, z):
            return (x - x0) * scale, (z - z0) * scale

        # Raster base: navigable white, the rest light grey; unexplored darkened at the explored level.
        Hg, Wg = st.nav.shape
        r0, c0 = st.world_to_grid(x0, z0)
        r1, c1 = st.world_to_grid(x1, z1)
        r0, c0, r1, c1 = int(math.floor(r0)), int(math.floor(c0)), int(math.ceil(r1)), int(math.ceil(c1))
        base = np.full((r1 - r0, c1 - c0, 3), 215, np.uint8)
        rr0, cc0, rr1, cc1 = max(r0, 0), max(c0, 0), min(r1, Hg), min(c1, Wg)
        if rr1 > rr0 and cc1 > cc0:
            sub = st.nav[rr0:rr1, cc0:cc1] > 0
            tile = base[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
            tile[sub] = 255
            if self.level == "explored":
                unseen = ~self.explored[rr0:rr1, cc0:cc1]
                tile[unseen] = (tile[unseen] * 0.62).astype(np.uint8)
        img = Image.fromarray(base).resize(
            (max(1, int(round((c1 - c0) * st.mpp * scale))), max(1, int(round((r1 - r0) * st.mpp * scale)))),
            Image.NEAREST,
        )
        canvas = Image.new("RGB", (W, H), (215, 215, 215))
        off = px(st.origin[0] + c0 * st.mpp, st.origin[1] + r0 * st.mpp)
        canvas.paste(img, (int(round(off[0])), int(round(off[1]))))
        d = ImageDraw.Draw(canvas, "RGBA")
        if self._fonts is None:
            self._fonts = _load_fonts()
        font, bold = self._fonts

        room_colors = {}
        for i, name in enumerate(sorted(st.rooms)):
            col = _PALETTE[i % len(_PALETTE)]
            room_colors[name] = col
            d.polygon([px(x, z) for x, z in st.rooms[name].poly], fill=col + (26,), outline=col + (220,))
        for fp in st.furniture.values():
            fill = (95, 95, 95, 120) if fp.labelled else (150, 150, 150, 90)
            d.polygon([px(x, z) for x, z in fp.corners], fill=fill, outline=(40, 40, 40, 255))

        # Trail, START, robot.
        if len(self.trail) > 1:
            d.line([px(x, z) for x, z in self.trail], fill=(30, 144, 255, 210), width=3)
        placed: List[Tuple[float, float, float, float]] = []
        marks: List[Tuple[float, float, float, float]] = []
        if self.start is not None:
            X, Y = px(self.start[0], self.start[1])
            d.rectangle([X - 8, Y - 8, X + 8, Y + 8], outline=(0, 128, 0), width=3)
            marks.append((X - 9, Y - 9, X + 9, Y + 9))
        if self.pose is not None:
            X, Y = px(self.pose[0], self.pose[1])
            yaw = self.pose[2]
            tip = (X + 24 * math.cos(yaw), Y + 24 * math.sin(yaw))
            left = (X + 11 * math.cos(yaw + 2.4), Y + 11 * math.sin(yaw + 2.4))
            right = (X + 11 * math.cos(yaw - 2.4), Y + 11 * math.sin(yaw - 2.4))
            d.polygon([tip, left, right], fill=(255, 140, 0), outline=(0, 0, 0))
            marks.append((X - 24, Y - 24, X + 24, Y + 24))

        # Objects (observed / explored): at their parent furniture or room, never at their own position.
        obj_points = []
        if self.level in ("observed", "explored"):
            groups: Dict[Tuple[str, Optional[str]], List[str]] = {}
            for name, kind, anchor in object_anchors(world_graph, st):
                if kind == "unknown":
                    continue
                groups.setdefault((kind, anchor), []).append(name)
            for (kind, anchor), names in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
                if kind == "robot":
                    if self.pose is None:
                        continue
                    cx, cy = px(self.pose[0], self.pose[1])
                elif kind == "furniture":
                    cx, cy = px(*st.furniture[anchor].center)
                else:
                    cx, cy = px(*st.rooms[anchor].centroid)
                    cy += 18  # leave the room name readable
                for k, name in enumerate(names):
                    ang = 2 * math.pi * k / max(1, len(names)) - math.pi / 2
                    rad = 0 if len(names) == 1 else 9
                    X, Y = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
                    d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=(220, 20, 60), outline=(0, 0, 0))
                    marks.append((X - 6, Y - 6, X + 6, Y + 6))
                    obj_points.append((name + (" (held)" if kind == "robot" else ""), X, Y))

        # Labels: START/ROBOT, rooms, objects, then furniture; greedy placement that avoids overlaps.
        def place(text, X, Y, fnt, fg, bg=(255, 255, 255, 215), leader=True, avoid_marks=True):
            w, h = _text_size(d, text, fnt)
            cands = [(0, 0)]
            for r in (14, 26, 40, 56):
                cands += [(0, -r), (0, r), (r + w / 2, 0), (-(r + w / 2), 0),
                          (r + w / 2, -r), (-(r + w / 2), -r), (r + w / 2, r), (-(r + w / 2), r)]
            chosen = None
            for dx, dy in cands:
                box = (X + dx - w / 2 - 2, Y + dy - h / 2 - 1, X + dx + w / 2 + 2, Y + dy + h / 2 + 1)
                if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
                    continue
                if any(_overlap(box, b) for b in placed):
                    continue
                if avoid_marks and any(_overlap(box, b) for b in marks):
                    continue
                chosen = (dx, dy, box)
                break
            if chosen is None:
                dx, dy = 0, 0
                box = (X - w / 2 - 2, Y - h / 2 - 1, X + w / 2 + 2, Y + h / 2 + 1)
            else:
                dx, dy, box = chosen
            if leader and (dx or dy):
                d.line([(X, Y), (X + dx, Y + dy)], fill=(60, 60, 60, 200), width=1)
            if bg is not None:
                d.rectangle(box, fill=bg)
            d.text((X + dx - w / 2, Y + dy - h / 2), text, fill=fg, font=fnt)
            placed.append(box)

        if self.start is not None:
            X, Y = px(self.start[0], self.start[1])
            place("START", X, Y + 16, bold, (0, 110, 0), leader=False)
        if self.pose is not None:
            X, Y = px(self.pose[0], self.pose[1])
            place("ROBOT", X, Y + 20, bold, (190, 85, 0), leader=False)
        for name in sorted(st.rooms):
            if name.lower().startswith(_EXCLUDED_ROOM_PREFIXES) and not any(
                fp.room == name and fp.labelled for fp in st.furniture.values()
            ):
                continue
            cx, cz = st.rooms[name].centroid
            col = room_colors[name]
            place(name.upper(), *px(cx, cz), bold, tuple(int(c * 0.7) for c in col) + (255,),
                  bg=(255, 255, 255, 150), leader=False, avoid_marks=False)
        for name, X, Y in obj_points:
            place(name, X + 8, Y - 8, font, (175, 0, 35))
        for fp in sorted(st.furniture.values(), key=lambda f: f.name):
            if fp.labelled:
                place(fp.name, *px(*fp.center), font, (0, 0, 0))

        # Object dots last, so no label (room names may overlap markers) hides one.
        for _name, X, Y in obj_points:
            d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=(220, 20, 60), outline=(0, 0, 0))

        # 2 m scale bar.
        bar = 2.0 * scale
        d.rectangle([10, H - 22, 10 + bar, H - 16], fill=(0, 0, 0))
        d.text((10 + bar + 6, H - 26), "2 m", fill=(0, 0, 0), font=font)
        self.renders += 1
        self.render_time_s += time.perf_counter() - t0
        return canvas

    # ---- text ----------------------------------------------------------

    def to_text(self, world_graph=None) -> str:
        st = self.static
        lines = [
            "Top-down map (metres, from a prior mapping run of the house; x grows to the right and z grows "
            "downwards in the map image):"
        ]
        if self.pose is not None:
            x, z, yaw = self.pose
            room = st.room_at(x, z) or "unknown room"
            lines.append(f"- You (robot): x={x:.1f}, z={z:.1f}, in {room}, facing {_direction_words(yaw)} in the map.")
        if self.start is not None:
            lines.append(
                f"- You started this task in {self.start_room or 'an unknown room'} "
                f"(x={self.start[0]:.1f}, z={self.start[1]:.1f}; START on the map)."
            )
        lines.append("Rooms (x range, z range" + (", fraction of floor seen so far" if self.level == "explored" else "")
                     + "):")
        for name in sorted(st.rooms):
            p = st.rooms[name].poly
            line = f"- {name}: x[{p[:, 0].min():.1f}, {p[:, 0].max():.1f}] z[{p[:, 1].min():.1f}, {p[:, 1].max():.1f}]"
            if self.level == "explored":
                frac = self.room_explored_fraction(name)
                if frac is not None:
                    line += f", seen {int(round(100 * frac))}%"
            lines.append(line)
        lines.append("Furniture (room: center x,z; footprint length x width; top height; "
                     "distance from center to nearest wall; distance to room center):")
        for fp in sorted(st.furniture.values(), key=lambda f: f.name):
            if not fp.labelled:
                continue
            cx, cz = fp.center
            L, Wd = fp.size
            parts = [f"- {fp.name} ({fp.room or 'unknown room'}): ({cx:.1f}, {cz:.1f}), {L:.1f}x{Wd:.1f} m"]
            if math.isfinite(fp.top_height):
                parts.append(f"top {fp.top_height:.2f} m")
            room = st.rooms.get(fp.room) if fp.room else None
            if room is not None:
                parts.append(f"wall {distance_to_boundary(cx, cz, room.poly):.1f} m")
                parts.append(f"room center {math.hypot(cx - room.centroid[0], cz - room.centroid[1]):.1f} m")
            lines.append(", ".join(parts))
        text = "\n".join(lines)
        cap = int(self.cfg.get("max_text_chars", 6000))
        if cap > 0 and len(text) > cap:
            text = text[:cap].rsplit("\n", 1)[0] + "\n...[map text truncated]"
        return text

    # ---- captions / persistence ---------------------------------------

    def image_caption(self) -> str:
        parts = [
            "[Top-down map of the house. Image up is -z; 2 m scale bar bottom-left. Coloured outlines are rooms; "
            "grey boxes are furniture, labelled with the furniture IDs you can use. Green START is where you began "
            "the task; the orange arrow is you and points where you face; the blue line is the path you drove."
        ]
        if self.level in ("observed", "explored"):
            parts.append(
                " Red dots are objects in your memory, drawn at the furniture (or room) your memory places them on "
                "or in, not at an exact position; they can be out of date."
            )
        if self.level == "explored":
            parts.append(" Darker floor is floor you have not seen yet.")
        return "".join(parts) + "]"

    def summary(self) -> Dict[str, Any]:
        return {
            "overhead_map_level": self.level,
            "overhead_map_renders": self.renders,
            "overhead_map_render_s": round(self.render_time_s, 4),
            "overhead_map_depth_frames": self._depth_frames,
            "overhead_map_depth_s": round(self.depth_time_s, 4),
            "overhead_map_furniture": len(self.static.furniture),
            "overhead_map_rooms": len(self.static.rooms),
        }


def save_map(img, text: str, out_dir: str, index: int, tag: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"map_{index:04d}_{tag}")
    img.save(stem + ".png")
    with open(stem + ".txt", "w", encoding="utf-8") as f:
        f.write(text)
    return stem + ".png", stem + ".txt"


def _text_size(draw, text: str, font) -> Tuple[float, float]:
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return float(b[2] - b[0]), float(b[3] - b[1])
    except Exception:
        return 6.0 * len(text), 11.0


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
