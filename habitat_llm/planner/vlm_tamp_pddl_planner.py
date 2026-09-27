import contextlib
import colorsys
import hashlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import magnum as mn
from omegaconf import DictConfig

from habitat_llm.planner.planner import Planner
from habitat_llm.utils.llm_usage import snapshot_from_llm
from habitat_llm.world_model.entities.floor import Floor
from habitat_llm.world_model.entity import Object, Receptacle
from habitat_llm.llm.instruct.utils import get_world_descr, pil_image_to_data_url
from habitat_llm.pddlstream.problem import build_pddlstream_problem, extract_scope_names
from habitat_llm.pddlstream.solve import solve_pddlstream_problem
from habitat_llm.vlm_tamp import (
    GPT4vApi,
    Claude3Api,
    build_english_subgoal_prompt,
    build_predicate_translation_prompt,
    build_failure_history,
    parse_branch_response,
)

# ANSI color codes (avoids importing habitat_llm.utils to keep this file self-contained)
_COLORS = {
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "cyan":    "\033[36m",
    "white":   "\033[97m",
    "gray":    "\033[37m",
    "reset":   "\033[0m",
    "bold":    "\033[1m",
}


def _color(text: str, c: str) -> str:
    return _COLORS.get(c, "") + text + _COLORS["reset"]


# Visual delimiter for subgoal execution blocks in PDDL baseline HTML / jsonl logs
_SUBGOAL_LOG_BORDER = "*" * 60


# Subgoal predicates whose single argument must be an articulated joint name
# (matches _subgoal_to_goal_literal door/drawer handling).
_SUBGOAL_JOINT_ARG_PREDS = frozenset(
    {
        "opened-door",
        "opened-drawer",
        "open",
        "opened",
        "closed-door",
        "closed-drawer",
        "close",
        "closed",
    }
)


class _TeeStream:
    """Write to a real stream while copying text into an in-memory buffer."""

    def __init__(self, real_stream, mirror_buffer):
        self._real = real_stream
        self._mirror = mirror_buffer

    def write(self, data):
        self._real.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self):
        self._real.flush()


# (name, rgb, u_min, v_min, u_max, v_max) from GT projection
GtProjectedBox = Tuple[str, Tuple[int, int, int], float, float, float, float]


def _rgb_for_label(label: str) -> Tuple[int, int, int]:
    """Stable saturated RGB per object name for bbox + label background."""
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


def _overlap_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx <= 0 or dy <= 0:
        return 0.0
    return float(dx * dy)


def _box_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    inter = _overlap_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _gt_box_rect(box: GtProjectedBox) -> Tuple[float, float, float, float]:
    return box[2], box[3], box[4], box[5]


def _clip_and_accept_gt_box(
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
    img_w: int,
    img_h: int,
    min_side_px: float = 12.0,
    min_on_screen_fraction: float = 0.3,
    max_image_coverage: float = 0.9,
) -> Optional[Tuple[float, float, float, float]]:
    """Keep a projected AABB only if a useful fraction of it is actually on-screen."""
    if u_max <= u_min or v_max <= v_min:
        return None
    if u_max < 0 or v_max < 0 or u_min >= img_w or v_min >= img_h:
        return None

    cu_min = max(u_min, 0.0)
    cv_min = max(v_min, 0.0)
    cu_max = min(u_max, float(img_w - 1))
    cv_max = min(v_max, float(img_h - 1))
    if cu_max <= cu_min or cv_max <= cv_min:
        return None

    raw_area = _box_area((u_min, v_min, u_max, v_max))
    on_area = _box_area((cu_min, cv_min, cu_max, cv_max))
    if raw_area <= 0.0 or on_area <= 0.0:
        return None
    if (cu_max - cu_min) < min_side_px or (cv_max - cv_min) < min_side_px:
        return None
    if on_area / raw_area < min_on_screen_fraction:
        return None
    image_area = float(max(img_w, 1) * max(img_h, 1))
    if on_area / image_area > max_image_coverage:
        return None
    return cu_min, cv_min, cu_max, cv_max


def _suppress_similar_overlapping_boxes(
    boxes: List[GtProjectedBox],
    movable_names: Optional[Set[str]] = None,
    iou_thresh: float = 0.55,
    max_area_ratio: float = 2.5,
) -> List[GtProjectedBox]:
    """Drop near-duplicate furniture boxes; keep nested different-size ones (bed vs nightstand)."""
    movable = set(movable_names or [])
    movables = [b for b in boxes if b[0] in movable]
    furniture = [b for b in boxes if b[0] not in movable]
    furniture.sort(key=lambda b: _box_area(_gt_box_rect(b)))
    kept: List[GtProjectedBox] = []
    for box in furniture:
        rect = _gt_box_rect(box)
        area = _box_area(rect)
        drop = False
        for prev in kept:
            prev_rect = _gt_box_rect(prev)
            prev_area = _box_area(prev_rect)
            if area <= 0.0 or prev_area <= 0.0:
                continue
            ratio = max(area, prev_area) / min(area, prev_area)
            if ratio <= max_area_ratio and _box_iou(rect, prev_rect) >= iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(box)
    return movables + kept


def _clamp_text_to_image(
    tx: float,
    ty: float,
    text: str,
    font,
    draw,
    pad: int,
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    """Shift (tx, ty) so the padded label background stays inside the image."""
    for _ in range(6):
        left, top, right, bottom = draw.textbbox((tx, ty), text, font=font)
        l, t, r, b = left - pad, top - pad, right + pad, bottom + pad
        dx = dy = 0.0
        if l < 0:
            dx -= l
        if t < 0:
            dy -= t
        if r > img_w:
            dx -= r - img_w
        if b > img_h:
            dy -= b - img_h
        if dx == 0 and dy == 0:
            break
        tx += dx
        ty += dy
    return tx, ty


def _candidate_label_positions(
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
    text: str,
    font,
    draw,
    pad: int,
    gap: int,
) -> List[Tuple[float, float]]:
    """Ordered (tx, ty) anchors to try so labels can avoid each other."""
    l0, t0, r0, b0 = draw.textbbox((0, 0), text, font=font)
    tw = r0 - l0
    th = b0 - t0
    lw = tw + 2 * pad
    lh = th + 2 * pad
    cx = (u_min + u_max) / 2.0
    cy = (v_min + v_max) / 2.0

    candidates: List[Tuple[float, float]] = []

    def add(tx: float, ty: float) -> None:
        candidates.append((tx, ty))

    # Prefer inside the 2D bbox: corners, then edges, then interior grid.
    add(u_min + 2, v_min + 2)
    add(u_max - tw - 2 * pad - 2, v_min + 2)
    add(u_min + 2, v_max - th - 2 * pad - 2)
    add(u_max - tw - 2 * pad - 2, v_max - th - 2 * pad - 2)
    add(cx - tw / 2.0, v_min + 2)
    add(cx - tw / 2.0, v_max - th - 2 * pad - 2)
    add(u_min + 2, cy - th / 2.0 - pad)
    add(u_max - tw - 2 * pad - 2, cy - th / 2.0 - pad)

    # Outside the bbox (common when objects are tight in screen space).
    add(cx - tw / 2.0, v_min - lh - gap)
    add(cx - tw / 2.0, v_max + gap)
    add(u_min - lw - gap, cy - th / 2.0 - pad)
    add(u_max + gap, cy - th / 2.0 - pad)
    add(u_max + gap, v_min + 2)

    bw = max(1.0, u_max - u_min - lw - 4)
    bh = max(1.0, v_max - v_min - lh - 4)
    for dy in range(0, int(min(140.0, bh)), 16):
        for dx in range(0, int(min(140.0, bw)), 16):
            add(u_min + 2 + dx, v_min + 2 + dy)

    seen = set()
    out: List[Tuple[float, float]] = []
    for tx, ty in candidates:
        key = (round(tx, 2), round(ty, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append((tx, ty))
    return out


def _pick_label_position(
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
    text: str,
    font,
    draw,
    pad: int,
    gap: int,
    placed: List[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, Tuple[float, float, float, float]]:
    """Choose (tx, ty) minimizing overlap with *placed* label rectangles."""
    candidates = _candidate_label_positions(
        u_min, v_min, u_max, v_max, text, font, draw, pad, gap
    )
    best: Optional[Tuple[float, float, Tuple[float, float, float, float]]] = None
    best_score = float("inf")

    for tx, ty in candidates:
        tx, ty = _clamp_text_to_image(tx, ty, text, font, draw, pad, img_w, img_h)
        left, top, right, bottom = draw.textbbox((tx, ty), text, font=font)
        rl, rt, rr, rb = left - pad, top - pad, right + pad, bottom + pad
        if rl < -0.5 or rt < -0.5 or rr > img_w + 0.5 or rb > img_h + 0.5:
            continue
        margin = 2.0
        score = sum(
            _overlap_area(
                (rl, rt, rr, rb),
                (p[0] - margin, p[1] - margin, p[2] + margin, p[3] + margin),
            )
            for p in placed
        )
        if score < best_score:
            best_score = score
            best = (tx, ty, (rl, rt, rr, rb))
        if score == 0.0:
            break

    if best is not None:
        return best[0], best[1], best[2]

    # Fallback: first clamped candidate that fits, even with overlap.
    for tx, ty in candidates:
        tx, ty = _clamp_text_to_image(tx, ty, text, font, draw, pad, img_w, img_h)
        left, top, right, bottom = draw.textbbox((tx, ty), text, font=font)
        rl, rt, rr, rb = left - pad, top - pad, right + pad, bottom + pad
        if rl >= -0.5 and rt >= -0.5 and rr <= img_w + 0.5 and rb <= img_h + 0.5:
            return tx, ty, (rl, rt, rr, rb)

    tx, ty = _clamp_text_to_image(u_min + 2, v_min + 2, text, font, draw, pad, img_w, img_h)
    left, top, right, bottom = draw.textbbox((tx, ty), text, font=font)
    rl, rt, rr, rb = left - pad, top - pad, right + pad, bottom + pad
    return tx, ty, (rl, rt, rr, rb)


def _draw_bbox_label_at(
    draw,
    font,
    tx: float,
    ty: float,
    text: str,
    color: Tuple[int, int, int],
    pad: int,
) -> None:
    """White text on solid *color* background at anchor (tx, ty)."""
    left, top, right, bottom = draw.textbbox((tx, ty), text, font=font)
    draw.rectangle(
        [left - pad, top - pad, right + pad, bottom + pad],
        fill=color,
    )
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)


def _split_gt_boxes_for_two_panels(boxes: List[GtProjectedBox]) -> List[List[GtProjectedBox]]:
    """Split boxes by horizontal center (u) for two VLM panels; fallback to single list."""
    if len(boxes) <= 1:
        return [boxes]
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] + b[4]) * 0.5)
    n = len(sorted_boxes)
    mid = (n + 1) // 2
    a = sorted_boxes[:mid]
    b = sorted_boxes[mid:]
    if not a or not b:
        return [boxes]
    return [a, b]


class VlmTampPddlPlanner(Planner):
    """VLM-TAMP planner for PARTNR / Habitat.

    Pipeline (mirrors kitchen-worlds LLAMPAgent):
      1. Observe scene → build typed object list + scene description
      2. VLM Turn 1  → English intermediate goals (with optional image)
      3. VLM Turn 2  → Translate to PDDL predicates; request N branches
      4. For each branch, for each subgoal:
           a. Build PDDLStream problem from WorldGraph
           b. Solve symbolically (oracle skills → no IK needed)
           c. Execute resulting action sequence via Habitat tools
      5. On failure: try next branch; on branch exhaustion: re-query VLM
    """

    def __init__(self, plan_config: DictConfig, env_interface):
        super().__init__(plan_config, env_interface)
        self.plan_config = plan_config
        self._init_vlm()
        self.reset()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_vlm(self):
        cfg = self.plan_config.get("vlm", {}) if isinstance(self.plan_config, DictConfig) else {}
        vlm_type = cfg.get("type", "gpt4v")
        model_name = cfg.get("model_name", "gpt-4o-mini")
        cache_path = cfg.get("cache_path", None)
        if vlm_type == "claude3":
            self.vlm = Claude3Api(model_name=model_name, cache_path=cache_path)
        else:
            self.vlm = GPT4vApi(model_name=model_name, cache_path=cache_path)
        self.vlm_max_tokens = cfg.get("max_completion_tokens", 1200)
        self.vlm_temperature = cfg.get("temperature", 0.2)
        self.vlm_reasoning_effort = cfg.get("reasoning_effort", None)
        if self.vlm_reasoning_effort:
            self.vlm.reasoning_effort = str(self.vlm_reasoning_effort).strip().lower()
            self.vlm.token_usage.set_reasoning_effort(self.vlm_reasoning_effort)
        self.use_images = bool(cfg.get("use_images", True))
        self.use_gt_bboxes = bool(cfg.get("use_gt_bboxes", False))
        _panels = int(cfg.get("gt_bbox_image_panels", 1))
        self.gt_bbox_image_panels = 2 if _panels >= 2 else 1
        self.write_log_html = bool(cfg.get("write_log_html", True))
        self.write_planning_tree_png = bool(cfg.get("write_planning_tree_png", True))
        self.planning_tree_show_failure_labels = bool(
            cfg.get("planning_tree_show_failure_labels", True)
        )
        self.planning_tree_failure_label_mode = str(
            cfg.get("planning_tree_failure_label_mode", "flag")
        )
        self.num_branches = int(self.plan_config.get("num_branches", 2))
        self.max_reprompts = int(self.plan_config.get("max_reprompts", 2))
        self.ntamp_max_pddl_replan_retries = max(
            0, int(self.plan_config.get("ntamp_max_pddl_replan_retries", 1))
        )
        self.log_dir_name = self.plan_config.get("log_dir", "vlm_tamp_pddl")
        self.verbose = bool(self.plan_config.get("verbose", True))
        self.enable_partial_obs_explore = bool(
            self.plan_config.get("enable_partial_obs_explore", False)
        )
        self.explore_on_missing_required_objects = bool(
            self.plan_config.get("explore_on_missing_required_objects", False)
        )
        self.enable_vlm_explore_subgoal = bool(
            self.plan_config.get("enable_vlm_explore_subgoal", True)
        )
        self.max_room_explore_passes_per_object = max(
            1, int(self.plan_config.get("max_room_explore_passes_per_object", 1))
        )
        self.explore_fast = bool(self.plan_config.get("explore_fast", False))
        self.explore_image_interval = max(1, int(self.plan_config.get("explore_image_interval", 30)))
        self.max_vlm_cycles = max(1, int(self.plan_config.get("max_vlm_cycles", 20)))
        self.max_searches_per_target = max(1, int(self.plan_config.get("max_searches_per_target", 2)))
        # evaluation.pddl_baseline (baseline yaml) or plan_config.pddl_baseline
        self.pddl_baseline_html = self._read_pddl_baseline_flag()

    def _read_pddl_baseline_flag(self) -> bool:
        if bool(self.plan_config.get("pddl_baseline", False)):
            return True
        conf = getattr(self.env_interface, "conf", None)
        if conf is None:
            return False
        try:
            ev = conf.get("evaluation") if hasattr(conf, "get") else None
            if ev is None and hasattr(conf, "evaluation"):
                ev = conf.evaluation
            if ev is not None:
                v = ev.get("pddl_baseline", False) if hasattr(ev, "get") else getattr(
                    ev, "pddl_baseline", False
                )
                return bool(v)
        except Exception:
            pass
        return False

    def _subgoal_exec_log_clear(self) -> None:
        if self.pddl_baseline_html:
            self._subgoal_exec_log_lines = []

    def _subgoal_exec_log_append(self, line: str) -> None:
        if self.pddl_baseline_html:
            if not self._subgoal_exec_log_lines:
                self._subgoal_exec_log_lines.append(_SUBGOAL_LOG_BORDER)
            self._subgoal_exec_log_lines.append(line)

    def _subgoal_exec_log_begin_subgoal(self) -> None:
        """Leading border on the terminal and seed for PDDL baseline log (before subgoal header)."""
        self._vprint(_SUBGOAL_LOG_BORDER)
        if self.pddl_baseline_html:
            self._subgoal_exec_log_lines = [_SUBGOAL_LOG_BORDER]

    def _vprint_subgoal_status_block(self, body_line: str, color: str = "green") -> None:
        """Border / status line / border on terminal and in subgoal_execution buffer."""
        self._vprint(_SUBGOAL_LOG_BORDER)
        self._vprint(body_line, color)
        self._vprint(_SUBGOAL_LOG_BORDER)
        if self.pddl_baseline_html:
            self._subgoal_exec_log_lines.append(_SUBGOAL_LOG_BORDER)
            self._subgoal_exec_log_lines.append(body_line.lstrip("\n"))
            self._subgoal_exec_log_lines.append(_SUBGOAL_LOG_BORDER)

    def _vprint_subgoal_complete_block(self, subgoal_text: str) -> None:
        """Visual block when a subgoal finishes successfully (normal plan execution)."""
        line = f"\n  ✓ SUBGOAL {self.subgoal_idx} COMPLETE: {subgoal_text}"
        self._vprint_subgoal_status_block(line, "green")

    def _emit_subgoal_execution(
        self, status: str, subgoal: str, *, extra_trailing_delimiter: bool = True
    ) -> None:
        if not self.pddl_baseline_html:
            return
        if extra_trailing_delimiter:
            if not self._subgoal_exec_log_lines:
                self._subgoal_exec_log_lines = [_SUBGOAL_LOG_BORDER, _SUBGOAL_LOG_BORDER]
            else:
                self._subgoal_exec_log_lines.append(_SUBGOAL_LOG_BORDER)
            self._vprint(_SUBGOAL_LOG_BORDER)
        text = "\n".join(self._subgoal_exec_log_lines)
        self._log_event(
            {
                "event": "subgoal_execution",
                "reprompt_round": self._reprompt_round,
                "branch": self.branch_idx,
                "subgoal_idx": self.subgoal_idx,
                "subgoal": subgoal,
                "status": status,
                "log_text": text,
            }
        )
        self._subgoal_exec_log_lines = []

    def reset(self):
        # Plan tree: list of branches; each branch is a list of subgoal strings
        self.branches: List[List[str]] = []
        self._branch_origin_round: List[int] = []
        self.branch_idx: int = 0
        # Within the active branch
        self.subgoal_idx: int = 0
        self.current_plan: List[Tuple[str, List[str]]] = []
        self.current_action_idx: int = 0
        self.last_high_level_actions: Dict[int, Tuple[str, str, str]] = {}
        self.reprompt_count: int = 0
        self.is_done: bool = False
        self.history: List[str] = []
        self.trace: str = ""
        self._log_dir: Optional[str] = None
        self._reverse_name_map: Dict[str, str] = {}
        self._last_failure: str = ""
        self._objects_by_type: Dict[str, List[str]] = {}
        self._episode_banner_printed: bool = False
        self._subgoals_completed: int = 0
        self._vlm_image_count: int = 0
        self._subgoal_image_count: int = 0
        self._trace_observations: Dict[str, Any] = {}
        self._explore_image_records: List[Dict[str, Any]] = []
        self._explore_image_tick = 0
        self._explore_image_session = 0
        self._subgoal_retry_count: int = 0
        self._vlm_cycles = 0
        self._vlm_wall_s = 0.0
        self._vlm_calls = 0
        self._search_counts = {}
        self._last_memory_snapshot = {}
        self._stop_reason = ""
        # Reprompt bookkeeping (kitchen-worlds-style)
        self._reprompt_round: int = 0
        self._already_succeeded_subgoals: List[str] = []
        self._planner_event_idx: int = 0
        # Partial-obs explore bookkeeping (CG memory based).
        self._explored_rooms: Set[str] = set()
        self._object_room_explore_counts: Dict[str, Dict[str, int]] = {}
        self._last_replanning_history_str: str = ""
        self._pending_replan_after_explore: bool = False
        self._subgoal_exec_log_lines: List[str] = []
        tracker = getattr(self.vlm, "token_usage", None)
        if tracker is not None and hasattr(tracker, "reset"):
            tracker.reset()

    # ------------------------------------------------------------------
    # Verbose output helpers
    # ------------------------------------------------------------------

    def _vprint(self, text: str, color: str = None, prefix: str = ""):
        """Print only when verbose mode is enabled."""
        if not self.verbose:
            return
        if color:
            print(_color(prefix + text, color))
        else:
            print(prefix + text)

    def _vprint_header(self, title: str, color: str = "cyan", width: int = 60):
        if not self.verbose:
            return
        bar = "═" * max(0, width - len(title) - 4)
        print(_color(f"\n══ {title} {bar}", color))

    def _vprint_banner(self, task: str):
        if not self.verbose:
            return
        width = min(70, max(50, len(task) + 14))
        border = "═" * (width - 2)
        task_line = f"  Task: {task}"
        print(_color(f"\n╔{border}╗", "cyan"))
        print(_color(f"║  VLM-TAMP PLANNER{' ' * (width - 20)}║", "cyan"))
        print(_color(f"╚{border}╝", "cyan"))
        print(_color(task_line, "white"))

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _trace_append(self, text: str):
        if not text:
            return
        if self.trace and not self.trace.endswith("\n"):
            self.trace += "\n"
        self.trace += text

    def _append_subgoal_exec_log_to_trace(self) -> None:
        """Mirror current subgoal execution log block into plain trace text."""
        if not self._subgoal_exec_log_lines:
            return
        self._trace_append("\n".join(self._subgoal_exec_log_lines))

    def _get_log_dir(self):
        if self._log_dir is None:
            results_dir = getattr(self.env_interface.conf.paths, "results_dir", ".")
            self._log_dir = os.path.join(results_dir, self.log_dir_name)
            os.makedirs(self._log_dir, exist_ok=True)
        return self._log_dir

    def _log_event(self, payload: Dict[str, Any]):
        if "wall_time" not in payload:
            payload["wall_time"] = time.time()
        if "seq_idx" not in payload:
            payload["seq_idx"] = self._planner_event_idx
            self._planner_event_idx += 1
        log_dir = self._get_log_dir()
        log_path = os.path.join(log_dir, "vlm_tamp_pddl_log.jsonl")
        with open(log_path, "a") as f:
            f.write(str(payload) + "\n")
        if payload.get("event") in ("vlm_english_subgoals", "planner_decision"):
            from habitat_llm.vlm_tamp.render_observation_history import render_observation_history
            render_observation_history(log_dir)

    def _append_vlm_prompt(
        self,
        prompt_kind: str,
        prompt_text: str,
        image_count: int = 0,
    ) -> None:
        """Persist exact prompts sent to the VLM for easier inspection."""
        log_dir = self._get_log_dir()
        prompts_path = os.path.join(log_dir, "vlm_prompts.txt")
        block = (
            "================================================================\n"
            f"reprompt_round: {self._reprompt_round}\n"
            f"prompt_kind: {prompt_kind}\n"
            f"image_count: {int(image_count)}\n"
            "----------------------------------------------------------------\n"
            f"{prompt_text}\n\n"
        )
        with open(prompts_path, "a", encoding="utf-8") as f:
            f.write(block)

    def _branch_origin(self, branch_idx: Optional[int] = None) -> Optional[int]:
        idx = self.branch_idx if branch_idx is None else branch_idx
        if idx is None:
            return None
        if 0 <= idx < len(self._branch_origin_round):
            return self._branch_origin_round[idx]
        return None

    def _log_subgoal_status(
        self,
        status: str,
        subgoal: str,
        branch_idx: Optional[int] = None,
        subgoal_idx: Optional[int] = None,
        failure_type: Optional[str] = None,
        failure_msg: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        bidx = self.branch_idx if branch_idx is None else branch_idx
        sidx = self.subgoal_idx if subgoal_idx is None else subgoal_idx
        payload: Dict[str, Any] = {
            "event": "subgoal_status",
            "status": status,
            "reprompt_round": self._reprompt_round,
            "branch": bidx,
            "branch_origin": self._branch_origin(bidx),
            "subgoal_idx": sidx,
            "subgoal": subgoal,
            "failure_type": failure_type,
            "failure_msg": failure_msg,
        }
        if extra:
            payload.update(extra)
        if status in ("started", "solved", "success", "failed", "already"):
            payload.update(self._save_subgoal_image(bidx, sidx, status))
        self._log_event(payload)

    def _save_subgoal_image(self, branch: int, subgoal_idx: int, status: str) -> Dict[str, Any]:
        """Record a camera frame at a subgoal boundary, without advancing physics."""
        if not getattr(self, "pddl_baseline_html", False):
            return {}
        try:
            from PIL import Image
            import numpy as np

            source = "fresh_sensor_render"
            try:
                observations = self.env_interface.sim.get_sensor_observations()
                rgb = observations.get("agent_0_third_rgb")
            except Exception:
                rgb = None
            if rgb is None:
                source = "latest_planner_observation"
                rgb = getattr(self, "_trace_observations", {}).get("agent_0_third_rgb")
            if rgb is None:
                return {"image_unavailable": "agent_0_third_rgb unavailable"}
            if hasattr(rgb, "detach"):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            while rgb.ndim > 3 and rgb.shape[0] == 1:
                rgb = rgb[0]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            idx = getattr(self, "_subgoal_image_count", 0)
            rel = f"subgoal_images/b{branch:03d}_s{subgoal_idx:03d}_{idx:05d}_{status}.png"
            path = os.path.join(self._get_log_dir(), rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(rgb).save(path)
            self._subgoal_image_count = idx + 1
            return {"image_path": rel, "image_source": source,
                    "image_capture_phase": status, "image_saved_wall_time": time.time()}
        except Exception as exc:
            return {"image_unavailable": str(exc)}

    def _log_pddl_problem(self, problem, subgoal: str, init_facts: List[Tuple]):
        log_dir = self._get_log_dir()
        dump_dir = os.path.join(log_dir, "pddl_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        domain_path = os.path.join(dump_dir, "domain.pddl")
        stream_path = os.path.join(dump_dir, "stream.pddl")
        if not os.path.exists(domain_path):
            with open(domain_path, "w") as f:
                f.write(problem.domain_pddl)
        if not os.path.exists(stream_path):
            with open(stream_path, "w") as f:
                f.write(problem.stream_pddl)
        problem_path = os.path.join(
            dump_dir, f"problem_b{self.branch_idx:02d}_s{self.subgoal_idx:02d}.txt"
        )
        with open(problem_path, "w") as f:
            f.write(f"branch: {self.branch_idx}, subgoal_idx: {self.subgoal_idx}\n")
            f.write(f"subgoal: {subgoal}\n")
            f.write("init:\n")
            for fact in init_facts:
                f.write(f"  {fact}\n")
            f.write(f"goal: {problem.goal}\n")

    def _log_plan_tree(self):
        log_dir = self._get_log_dir()
        tree_path = os.path.join(log_dir, "plan_tree.txt")
        with open(tree_path, "w") as f:
            for i, branch in enumerate(self.branches):
                marker = "* " if i == self.branch_idx else "  "
                origin = self._branch_origin_round[i] if i < len(self._branch_origin_round) else "?"
                f.write(f"{marker}Branch {i} (reprompt_round={origin}):\n")
                for j, sg in enumerate(branch):
                    status = ""
                    if i < self.branch_idx:
                        status = " [EXHAUSTED]"
                    elif i == self.branch_idx and j < self.subgoal_idx:
                        status = " [DONE]"
                    elif i == self.branch_idx and j == self.subgoal_idx:
                        status = " [CURRENT]"
                    f.write(f"    {j}: {sg}{status}\n")
        if getattr(self, "write_planning_tree_png", False):
            try:
                from habitat_llm.vlm_tamp.render_planning_tree import (
                    render_planning_tree_from_log_dir,
                )

                png_path = render_planning_tree_from_log_dir(
                    log_dir,
                    show_failure_labels=bool(
                        getattr(self, "planning_tree_show_failure_labels", True)
                    ),
                    failure_label_mode=str(
                        getattr(self, "planning_tree_failure_label_mode", "flag")
                    ),
                    write_layout_json=bool(getattr(self, "pddl_baseline_html", False)),
                )
                if png_path and self.verbose:
                    self._vprint(f"  [tree png]  {png_path}", "gray")
            except Exception as e:
                self._vprint(f"  [WARN] planning_tree.png failed: {e}", "red")

        if getattr(self, "write_log_html", False):
            try:
                if getattr(self, "pddl_baseline_html", False):
                    from habitat_llm.vlm_tamp.render_pddl_baseline_html import (
                        render_pddl_baseline_log_dir_to_html,
                    )

                    p = render_pddl_baseline_log_dir_to_html(log_dir)
                else:
                    from habitat_llm.vlm_tamp.render_log_html import render_log_dir_to_html

                    p = render_log_dir_to_html(log_dir)
                if p and self.verbose:
                    self._vprint(f"  [log html]  {p}", "gray")
            except Exception as e:
                self._vprint(f"  [WARN] log html failed: {e}", "red")

    # ------------------------------------------------------------------
    # Scene observation helpers
    # ------------------------------------------------------------------

    def _build_objects_by_type(
        self,
        world_graph,
        visible_object_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Group world-graph entities by role for typed VLM prompts."""
        movables = sorted({o.name for o in world_graph.get_all_objects()})
        furniture = sorted({f.name for f in world_graph.get_all_furnitures()})
        receptacles = sorted({r.name for r in world_graph.get_all_receptacles()})
        rooms = sorted({r.name for r in world_graph.get_all_rooms()})
        joints = sorted(
            {f.name for f in world_graph.get_all_furnitures()
             if f.properties.get("is_articulated", False)}
        )
        container_furniture_set: set = set()
        for rec in world_graph.get_all_receptacles():
            try:
                furn = world_graph.find_furniture_for_receptacle(rec)
            except Exception:
                furn = None
            if furn is not None and getattr(furn, "name", None):
                container_furniture_set.add(furn.name)
        container_furniture = sorted(container_furniture_set)
        faucet_furniture = sorted(
            f.name
            for f in world_graph.get_all_furnitures()
            if isinstance(f.properties.get("components"), list)
            and "faucet" in f.properties["components"]
        )
        furniture_to_room_map = world_graph.get_furniture_to_room_map()
        furniture_by_room: Dict[str, List[str]] = defaultdict(list)
        mapped: set = set()
        for furn, room in furniture_to_room_map.items():
            furniture_by_room[room.name].append(furn.name)
            mapped.add(furn.name)
        for fname in furniture:
            if fname not in mapped:
                furniture_by_room["unassigned"].append(fname)
        furniture_by_room = {
            k: sorted(v) for k, v in sorted(furniture_by_room.items())
        }
        objects_by_room: Dict[str, List[str]] = defaultdict(list)
        for obj in world_graph.get_all_objects():
            room_name: Optional[str] = None
            try:
                furn = world_graph.find_furniture_for_object(obj)
            except Exception:
                furn = None
            if furn is not None:
                room_node = furniture_to_room_map.get(furn)
                if room_node is not None and getattr(room_node, "name", None):
                    room_name = room_node.name
            if room_name is None:
                for neighbor in world_graph.get_neighbors(obj):
                    nname = getattr(neighbor, "name", None)
                    if nname in rooms:
                        room_name = nname
                        break
            if room_name is None:
                continue
            objects_by_room[room_name].append(obj.name)
        objects_by_room = {
            k: sorted(v) for k, v in sorted(objects_by_room.items())
        }
        obt = {
            "movable": movables,
            "furniture": furniture,
            "surface_furniture": furniture,
            "furniture_by_room": furniture_by_room,
            "objects_by_room": objects_by_room,
            "container_furniture": container_furniture,
            "faucet_furniture": faucet_furniture,
            "receptacle": receptacles,
            "room": rooms,
            "joint": joints,
        }

        self._vprint_header("SCENE OBSERVATION", "cyan")
        for label, names in [
            ("Movable", movables),
            ("Furniture", furniture),
            ("Receptacles", receptacles),
            ("Joints", joints),
            ("Rooms", rooms),
        ]:
            if names and False:
                self._vprint(f"  {label:<12}: {', '.join(names)}")

        return obt

    def _extract_visible_entity_names(
        self,
        observations: Dict[str, Any],
        world_graph,
    ) -> Optional[Set[str]]:
        """Infer visible names from instance pixels in the matching camera view."""
        rgb_data = observations.get("agent_0_third_rgb")
        if rgb_data is None:
            return None
        try:
            if "torch" in str(type(rgb_data)):
                rgb_data = rgb_data.detach().cpu().numpy()
            while len(rgb_data.shape) > 3 and rgb_data.shape[0] == 1:
                rgb_data = rgb_data[0]
            H, W = int(rgb_data.shape[0]), int(rgb_data.shape[1])
            boxes = self._collect_gt_projected_boxes(world_graph, W, H)
            return {name for name, *_rest in boxes}
        except Exception:
            return None

    def _build_scene_description(
        self,
        world_graph,
        agent_uid: int,
        visible_entity_names: Optional[Set[str]] = None,
    ) -> str:
        """Use the same remembered scene facts and serializer as PARTNR/ReAct.

        Visibility is used only for the requested image annotations, not to
        supply extra simulator-derived state claims in the text prompt.
        """
        return get_world_descr(
            world_graph, agent_uid=agent_uid, include_room_name=True,
            add_state_info=True, centralized=False,
        )

    @staticmethod
    def _memory_snapshot(world_graph):
        return {
            obj.name: {
                "relations": sorted((neighbor.name, edge) for neighbor, edge
                                    in world_graph.get_neighbors(obj).items()),
                "states": dict(obj.properties.get("states", {})),
            }
            for obj in world_graph.get_all_objects()
        }

    def _ask_vlm(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return self.vlm.ask(*args, **kwargs)
        finally:
            self._vlm_wall_s = getattr(self, "_vlm_wall_s", 0.0) + time.monotonic() - started
            self._vlm_calls = getattr(self, "_vlm_calls", 0) + 1

    def _finish_planning(self, reason, evidence=""):
        self.is_done = True
        self._stop_reason = reason
        self._log_event({"event": "planner_decision", "decision": reason, "evidence": evidence,
                         "evaluator_success": "not consulted"})
        self._trace_append(f"Planner decision: {reason}. {evidence}")

    def _observe_and_continue(self, instruction, observations, world_graph, reason, target=None):
        """Continue from the graph already updated by the shared env.step path."""
        before = getattr(self, "_last_memory_snapshot", {})
        fresh = self._observe_failure_state(observations, reason=reason)
        # Do not invoke a planner-specific perception update: PARTNR receives
        # the same graph after every environment step, including Open.
        after = self._memory_snapshot(world_graph)
        self._log_event({"event": "memory_update", "reason": reason, "target": target,
                         "new_objects": sorted(set(after) - set(before)),
                         "changed_objects": sorted(k for k in after if k in before and after[k] != before[k]),
                         "absence_policy": "not seen does not establish absence"})
        self._reprompt_round += 1
        self._log_event({"event": "reprompt_started", "reason": reason,
                         "branch": self.branch_idx, "subgoal_idx": self.subgoal_idx,
                         "target": target, "reprompt_round": self._reprompt_round})
        self.current_plan = []
        self.current_action_idx = 0
        self.last_high_level_actions = {}
        self._subgoal_retry_count = 0
        history = (f"Replanning reason: {reason}; target: {target}.\n"
                   f"Previously achieved subgoals: {self._already_succeeded_subgoals}.\n"
                   "Replan for the ORIGINAL task using this new observation.")
        self._generate_subgoals(instruction, world_graph, fresh, history_str=history,
                               append=True, observation_reason=reason)

    def _collect_gt_projected_boxes(self, world_graph, W: int, H: int) -> List[GtProjectedBox]:
        """Return visible-pixel boxes from the current RGB camera's instance mask."""
        from habitat_llm.vlm_tamp.visible_boxes import visible_entity_boxes

        entities = list(world_graph.get_all_objects()) + [
            fur for fur in world_graph.get_all_furnitures() if not isinstance(fur, Floor)
        ]
        return visible_entity_boxes(self.env_interface.sim, entities, _rgb_for_label, W, H)

    def _draw_gt_box_list_on_image(self, scene_img, boxes: List[GtProjectedBox]) -> None:
        """Draw projected boxes and labels on *scene_img* in-place (mutates image)."""
        from PIL import ImageDraw, ImageFont

        W, H = scene_img.size
        draw = ImageDraw.Draw(scene_img)
        font_size = max(11, H // 30)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size
            )
        except Exception:
            font = ImageFont.load_default()

        label_pad = 3
        label_gap = 4
        ordered = sorted(
            list(boxes),
            key=lambda b: (-(b[4] - b[2]) * (b[5] - b[3]), b[3], b[2]),
        )
        placed_label_rects: List[Tuple[float, float, float, float]] = []

        for name, color, u_min, v_min, u_max, v_max in ordered:
            draw.rectangle(
                [u_min, v_min, u_max, v_max], outline=color, width=1
            )
            tx, ty, lbl_rect = _pick_label_position(
                u_min,
                v_min,
                u_max,
                v_max,
                name,
                font,
                draw,
                label_pad,
                label_gap,
                placed_label_rects,
                W,
                H,
            )
            placed_label_rects.append(lbl_rect)
            _draw_bbox_label_at(draw, font, tx, ty, name, color, label_pad)

    def _draw_gt_bboxes_on_image(self, scene_img, world_graph):
        """Project GT 3D AABBs onto *scene_img* in-place (all visible entities)."""
        try:
            W, H = scene_img.size
            boxes = self._collect_gt_projected_boxes(world_graph, W, H)
            self._draw_gt_box_list_on_image(scene_img, boxes)
        except Exception as e:
            self._vprint(f"  [WARN] _draw_gt_bboxes_on_image failed: {e}", "red")

    def _get_image_data_url(self, observations: Dict[str, Any]):
        if not self.use_images:
            return None
        if "agent_0_third_rgb" not in observations:
            return None
        rgb_data = observations["agent_0_third_rgb"]
        if "torch" in str(type(rgb_data)):
            rgb_data = rgb_data.detach().cpu().numpy()
        while len(rgb_data.shape) > 3 and rgb_data.shape[0] == 1:
            rgb_data = rgb_data[0]
        if rgb_data.dtype != "uint8":
            rgb_data = (rgb_data * 255).astype("uint8")
        from PIL import Image
        pil_image = Image.fromarray(rgb_data)
        return pil_image_to_data_url(pil_image)

    def _get_annotated_vlm_image_urls(
        self, observations: Dict[str, Any], world_graph, *, annotate: bool = True
    ) -> List[str]:
        """Build data URL(s) for the VLM: raw frame, or GT bboxes on one or two panel images."""
        if not self.use_images:
            return []
        if "agent_0_third_rgb" not in observations:
            return []

        try:
            from PIL import Image

            rgb_data = observations["agent_0_third_rgb"]
            if "torch" in str(type(rgb_data)):
                rgb_data = rgb_data.detach().cpu().numpy()
            while len(rgb_data.shape) > 3 and rgb_data.shape[0] == 1:
                rgb_data = rgb_data[0]
            if rgb_data.dtype != "uint8":
                rgb_data = (rgb_data * 255).astype("uint8")

            base = Image.fromarray(rgb_data)
            W, H = base.size

            if not annotate or not self.use_gt_bboxes:
                self._save_vlm_images([base])
                return [pil_image_to_data_url(base)]

            boxes = self._collect_gt_projected_boxes(world_graph, W, H)
            use_split = self.gt_bbox_image_panels >= 2 and len(boxes) > 1
            panels = _split_gt_boxes_for_two_panels(boxes) if use_split else [boxes]

            if len(panels) == 1:
                img = base.copy()
                self._draw_gt_box_list_on_image(img, panels[0])
                self._save_vlm_images([img])
                return [pil_image_to_data_url(img)]

            img0 = base.copy()
            img1 = base.copy()
            self._draw_gt_box_list_on_image(img0, panels[0])
            self._draw_gt_box_list_on_image(img1, panels[1])
            self._save_vlm_images([img0, img1])
            return [pil_image_to_data_url(img0), pil_image_to_data_url(img1)]
        except Exception as e:
            import traceback
            self._vprint(f"  [WARN] annotated image failed: {e}", "red")
            self._vprint(f"  [DBG] traceback:\n{traceback.format_exc()}", "red")
            try:
                from PIL import Image as _PILImage

                rgb_data = observations["agent_0_third_rgb"]
                if "torch" in str(type(rgb_data)):
                    rgb_data = rgb_data.detach().cpu().numpy()
                while len(rgb_data.shape) > 3 and rgb_data.shape[0] == 1:
                    rgb_data = rgb_data[0]
                if rgb_data.dtype != "uint8":
                    rgb_data = (rgb_data * 255).astype("uint8")
                self._save_vlm_images([_PILImage.fromarray(rgb_data)])
            except Exception:
                pass
            one = self._get_image_data_url(observations)
            return [one] if one else []

    def _save_vlm_images(self, images: List[Any]) -> None:
        """Save PIL image(s) about to be sent to the VLM; one logical step → one counter tick."""
        self._last_vlm_image_paths = []
        if not images:
            return
        try:
            img_dir = os.path.join(self._get_log_dir(), "vlm_images")
            os.makedirs(img_dir, exist_ok=True)
            idx = self._vlm_image_count
            if len(images) == 1:
                fname = os.path.join(img_dir, f"vlm_input_{idx:04d}.png")
                images[0].save(fname)
                self._last_vlm_image_paths.append(os.path.relpath(fname, self._get_log_dir()))
                self._vprint(f"  [image saved]  {fname}", "gray")
            else:
                for si, im in enumerate(images):
                    fname = os.path.join(img_dir, f"vlm_input_{idx:04d}_p{si}.png")
                    im.save(fname)
                    self._last_vlm_image_paths.append(os.path.relpath(fname, self._get_log_dir()))
                    self._vprint(f"  [image saved]  {fname}", "gray")
            self._vlm_image_count += 1
        except Exception as e:
            self._vprint(f"  [WARN] Could not save VLM image: {e}", "red")

    # ------------------------------------------------------------------
    # VLM subgoal generation (two-step)
    # ------------------------------------------------------------------

    def _begin_explore_images(self, observations, world_graph, room_name):
        self._search_counts = getattr(self, "_search_counts", {})
        key = "explore:" + room_name
        self._search_counts[key] = self._search_counts.get(key, 0) + 1
        self._explore_image_records = []
        self._explore_image_tick = 0
        self._explore_image_session = getattr(self, "_explore_image_session", 0) + 1
        self._explore_image_room = room_name
        self._capture_explore_image(observations, world_graph, force=True)

    def _capture_explore_image(self, observations, world_graph, *, force=False):
        """Sample and annotate at capture time, while the camera pose is valid."""
        if not self.use_images:
            return
        tick = self._explore_image_tick
        self._explore_image_tick += 1
        if not force and tick % getattr(self, "explore_image_interval", 30):
            return
        try:
            from PIL import Image
            import numpy as np

            rgb = self.env_interface.sim.get_sensor_observations().get("agent_0_third_rgb")
            if rgb is None:
                raise ValueError("agent_0_third_rgb unavailable")
            if hasattr(rgb, "detach"):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            while rgb.ndim > 3 and rgb.shape[0] == 1:
                rgb = rgb[0]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            # Do not manufacture multiple views of a stationary robot.
            if self._explore_image_records and self._explore_image_records[-1]["sha256"] == digest:
                return
            im = Image.fromarray(rgb)
            boxes = self._collect_gt_projected_boxes(world_graph, *im.size)
            self._draw_gt_box_list_on_image(im, boxes)
            rel = f"explore_images/e{self._explore_image_session:04d}_t{tick:05d}.png"
            path = os.path.join(self._get_log_dir(), rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            im.save(path)
            record = {"image_path": rel, "capture_tick": tick, "sha256": digest,
                      "room": self._explore_image_room, "branch": self.branch_idx,
                      "subgoal_idx": self.subgoal_idx, "capture_wall_time": time.time(),
                      "visible_entity_names": [box[0] for box in boxes]}
            self._explore_image_records.append(record)
            self._log_event(dict(record, event="explore_image_capture"))
        except Exception as exc:
            self._log_event({"event": "explore_image_unavailable", "capture_tick": tick,
                             "error": str(exc)})

    def _explore_vlm_images(self):
        """Select first, temporal midpoint and last sampled exploration views."""
        from PIL import Image

        records = getattr(self, "_explore_image_records", [])
        if len(records) <= 3:
            selected = records
        else:
            mid = (records[0]["capture_tick"] + records[-1]["capture_tick"]) / 2
            middle = min(records[1:-1], key=lambda r: abs(r["capture_tick"] - mid))
            selected = [records[0], middle, records[-1]]
        images = []
        for record in selected:
            with Image.open(os.path.join(self._get_log_dir(), record["image_path"])) as im:
                images.append(im.copy())
        self._save_vlm_images(images)
        return [pil_image_to_data_url(im) for im in images], selected

    def _observe_failure_state(self, observations: Dict[str, Any], reason="failure") -> Dict[str, Any]:
        """Render the current failure state without taking a simulation step.

        Keep non-camera observations, but never reuse a stale planning image if
        the current camera cannot be read.
        """
        refreshed = dict(observations)
        refreshed.pop("agent_0_third_rgb", None)
        if not self.use_images:
            return refreshed
        event: Dict[str, Any] = {
            "event": "failure_observation" if reason == "failure" else "current_observation",
            "reason": reason,
            "reprompt_round": self._reprompt_round,
            "branch": self.branch_idx,
            "subgoal_idx": self.subgoal_idx,
            "image_source": "fresh_sensor_render",
        }
        try:
            sensors = self.env_interface.sim.get_sensor_observations()
            rgb = sensors.get("agent_0_third_rgb")
            if rgb is None:
                raise ValueError("agent_0_third_rgb unavailable")
            # Detach from simulator-owned buffers before rendering annotations.
            refreshed["agent_0_third_rgb"] = (
                rgb.clone() if hasattr(rgb, "clone") else rgb.copy()
            )
            event["image_available"] = True
        except Exception as exc:
            event.update(image_available=False, error=str(exc))
            self._vprint(f"  [WARN] Failure camera unavailable; replanning without an image: {exc}", "yellow")
        self._log_event(event)
        self._trace_observations = refreshed
        return refreshed

    def _generate_subgoals(
        self,
        instruction: str,
        world_graph,
        observations: Dict[str, Any],
        history_str: str = "",
        append: bool = False,
        after_explore: bool = False,
        explored_room: Optional[str] = None,
        refresh_failure_observation: bool = False,
        observation_reason: Optional[str] = None,
    ):
        """Two-step VLM prompting:
          Turn 1 — English intermediate goals (with scene image if available)
          Turn 2 — Translate to PDDL predicates; request self.num_branches alternatives
          If after_explore, append post-explore instructions (avoid redundant explore, prefer open joints).
        """
        self._vlm_cycles = getattr(self, "_vlm_cycles", 0) + 1
        if self._vlm_cycles > getattr(self, "max_vlm_cycles", 20):
            self._finish_planning("budget_exhausted", "VLM cycle limit reached")
            return
        if refresh_failure_observation:
            observations = self._observe_failure_state(observations)
        exploration_images = []
        explore_urls = []
        if after_explore and self.use_images:
            explore_urls, exploration_images = self._explore_vlm_images()
        visible_entity_names = self._extract_visible_entity_names(observations, world_graph)
        if exploration_images:
            visible_entity_names = {
                name for record in exploration_images for name in record["visible_entity_names"]
            }
        visible_object_names = None
        if visible_entity_names is not None:
            object_names = {obj.name for obj in world_graph.get_all_objects()}
            visible_object_names = visible_entity_names & object_names
        self._objects_by_type = self._build_objects_by_type(
            world_graph,
            visible_object_names=visible_object_names,
        )
        agent_uid = self._agents[0].uid if self._agents else 0
        self._last_memory_snapshot = self._memory_snapshot(world_graph)
        scene_desc = self._build_scene_description(
            world_graph,
            agent_uid,
            visible_entity_names=visible_entity_names,
        )
        self._log_event({"event": "memory_snapshot", "reprompt_round": self._reprompt_round,
                         "visible_entities": sorted(visible_entity_names or []),
                         "known_objects": sorted(o.name for o in world_graph.get_all_objects()),
                         "description": scene_desc,
                         "search_counts": dict(getattr(self, "_search_counts", {}))})

        self.vlm.new_session()

        # ── Turn 1: English subgoals ────────────────────────────────────
        english_prompt = build_english_subgoal_prompt(
            goal=instruction,
            objects_by_type=self._objects_by_type,
            scene_description=scene_desc,
            history=history_str + f"\nSearch/inspection counts: {getattr(self, '_search_counts', {})}. Per-target limit: {getattr(self, 'max_searches_per_target', 2)}.\n",
            after_explore=after_explore,
            explored_room=explored_room,
        )

        self._vprint_header(
            "VLM TURN 1: English Goals (post-explore)"
            if after_explore
            else "VLM TURN 1: English Goals",
            "magenta",
        )
        self._vprint(f"  [prompt excerpt]  {english_prompt}...", "gray")

        if after_explore and self.use_images:
            image_urls = explore_urls
        else:
            self._last_vlm_image_paths = []
            image_urls = self._get_annotated_vlm_image_urls(
                observations, world_graph, annotate=not (refresh_failure_observation or observation_reason)
            )
        english_prompt_send = english_prompt
        if after_explore:
            image_context = (
                f"These {len(image_urls)} annotated images were captured in chronological order "
                f"during Explore[{explored_room}]. They are different times/viewpoints, "
                "not label panels of one frame. Labels describe each capture time.\n"
                if image_urls else "No camera views were available for this Explore.\n"
            )
            for i, record in enumerate(exploration_images):
                image_context += f"Image {i + 1}: exploration tick {record['capture_tick']}.\n"
            english_prompt_send = image_context + "\n" + english_prompt
        elif observation_reason:
            english_prompt_send = f"Observation trigger: {observation_reason}. The attached image, if present, is the current third-person view.\n\n" + english_prompt
        elif refresh_failure_observation:
            english_prompt_send = (
                "The image is a fresh third-person camera observation at failure replanning.\n\n"
                if image_urls else "The current failure camera is unavailable; no image is attached.\n\n"
            ) + english_prompt
        elif len(image_urls) > 1:
            english_prompt_send = (
                "The next images are the same camera view of the scene. "
                "Object and furniture labels with bounding boxes are split across them so each image is less crowded; "
                "each annotated name appears on exactly one of the images.\n\n"
            ) + english_prompt
            self._vprint(
                f"  [image]  {len(image_urls)} annotated scene panels attached", "gray"
            )
        elif image_urls:
            self._vprint("  [image]  annotated scene image attached", "gray")

        self._append_vlm_prompt(
            prompt_kind="english_subgoals",
            prompt_text=english_prompt_send,
            image_count=len(image_urls),
        )
        english_response = self._ask_vlm(
            english_prompt_send,
            image_data_urls=image_urls if image_urls else None,
            max_completion_tokens=self.vlm_max_tokens,
            temperature=self.vlm_temperature,
            reasoning_effort=self.vlm_reasoning_effort,
        )
        self._log_event({
            "event": "vlm_english_subgoals",
            "reprompt_round": self._reprompt_round,
            "already_succeeded": list(self._already_succeeded_subgoals),
            "after_explore": bool(after_explore),
            "explored_room": explored_room,
            "image_count": len(image_urls),
            "image_context": "exploration_sequence" if after_explore else observation_reason or ("current_failure" if refresh_failure_observation else "initial_scene"),
            "vlm_cycle": self._vlm_cycles,
            "image_paths": list(getattr(self, "_last_vlm_image_paths", [])),
            "exploration_images": exploration_images,
            "prompt": english_prompt_send,
            "response": english_response,
            "api_response": self.vlm.get_last_response_metadata(),
        })

        try:
            decision = json.loads(str(english_response).strip().removeprefix("```json").removesuffix("```").strip())
        except (ValueError, TypeError):
            decision = None
        if isinstance(decision, dict) and decision.get("decision") in ("complete", "unsolvable"):
            self._finish_planning(decision["decision"], str(decision.get("reason", "")))
            return
        if not str(english_response).strip():
            self._finish_planning("invalid_response", "Empty VLM plan; inspect completion token limit")
            return
        self._vprint("  [response]", "magenta")
        for line in str(english_response).strip().splitlines():
            self._vprint(f"    {line}")
        self._trace_append(f"VLM English plan:\n{english_response}")

        # ── Turn 2: translate to predicates + request N branches ────────
        predicate_prompt = build_predicate_translation_prompt(
            self._objects_by_type,
            num_branches=self.num_branches,
            after_explore=after_explore,
        )

        self._vprint_header(f"VLM TURN 2: PDDL Predicates ({self.num_branches} branches)", "magenta")
        self._vprint(f"  [prompt excerpt]  {predicate_prompt[:200].strip()}...", "gray")

        self._append_vlm_prompt(
            prompt_kind="predicate_subgoals",
            prompt_text=predicate_prompt,
            image_count=0,
        )
        predicate_response = self._ask_vlm(
            predicate_prompt,
            max_completion_tokens=self.vlm_max_tokens,
            temperature=self.vlm_temperature,
            reasoning_effort=self.vlm_reasoning_effort,
        )
        branches = parse_branch_response(predicate_response)
        self._log_event({
            "event": "vlm_predicate_subgoals",
            "reprompt_round": self._reprompt_round,
            "already_succeeded": list(self._already_succeeded_subgoals),
            "append": bool(append),
            "after_explore": bool(after_explore),
            "explored_room": explored_room,
            "image_count": 0,
            "prompt": predicate_prompt,
            "response": predicate_response,
            "branches": branches,
            "api_response": self.vlm.get_last_response_metadata(),
        })

        raw_predicate_response = str(predicate_response).strip()
        self._vprint("  [response]", "magenta")
        if raw_predicate_response:
            preview_lines = raw_predicate_response.splitlines()
            for line in preview_lines[:20]:
                self._vprint(f"    {line}")
            if len(preview_lines) > 20:
                self._vprint("    ... [truncated]", "gray")
        else:
            self._vprint("    (empty raw response)", "red")

        if branches:
            self._vprint("  [parsed branches]", "magenta")
        for i, branch in enumerate(branches):
            self._vprint(f"    Branch {i}: {branch}", "yellow")

        if not branches or not any(branches):
            self._vprint("  [WARNING] VLM returned no valid subgoals.", "red")
            self._trace_append("VLM returned no valid subgoals.")
            self._finish_planning("invalid_response", "No executable subgoals or explicit terminal decision")
            return

        # Filter invalid subgoals (e.g. on(obj, movable)) and deduplicate branches
        filtered = [
            [sg for sg in b if self._validate_subgoal(sg)]
            for b in branches if b
        ]
        seen: List[List[str]] = []
        for b in filtered:
            if b and b not in seen:
                seen.append(b)

        if not seen:
            self._finish_planning("invalid_response", "All proposed subgoals failed validation")
            return

        if append and self.branches:
            old_len = len(self.branches)
            added_branches: List[List[str]] = []
            added_indices: List[int] = []
            for b in seen:
                if b:  # A new observation may justify retrying a previously proposed plan.
                    self.branches.append(b)
                    self._branch_origin_round.append(self._reprompt_round)
                    added_branches.append(b)
                    added_indices.append(len(self.branches) - 1)
            # When reprompting, start from the first newly-added branch.
            if len(self.branches) > old_len:
                self.branch_idx = old_len
            else:
                # No new branches; keep current pointer.
                self.branch_idx = min(self.branch_idx, max(0, len(self.branches) - 1))
            self._log_event(
                {
                    "event": "reprompt_branches_added",
                    "reprompt_round": self._reprompt_round,
                    "append": True,
                    "branch_start_idx": old_len,
                    "branch_count_before": old_len,
                    "branch_count_after": len(self.branches),
                    "added_indices": added_indices,
                    "added_branches": added_branches,
                    "active_branch": self.branch_idx,
                }
            )
        else:
            self.branches = seen
            self._branch_origin_round = [self._reprompt_round for _ in seen]
            self.branch_idx = 0
            self._log_event(
                {
                    "event": "reprompt_branches_added",
                    "reprompt_round": self._reprompt_round,
                    "append": False,
                    "branch_start_idx": 0,
                    "branch_count_before": 0,
                    "branch_count_after": len(self.branches),
                    "added_indices": list(range(len(self.branches))),
                    "added_branches": seen,
                    "active_branch": self.branch_idx,
                }
            )

        self.subgoal_idx = 0
        self.current_plan = []
        self.current_action_idx = 0
        self._log_event(
            {
                "event": "branch_start",
                "reprompt_round": self._reprompt_round,
                "branch": self.branch_idx,
                "branch_origin": self._branch_origin(),
                "subgoal_idx": self.subgoal_idx,
                "reason": "new_branches",
            }
        )
        self._trace_append(
            f"Plan tree: {len(self.branches)} branch(es)\n"
            + "\n".join(f"  Branch {i}: {b}" for i, b in enumerate(self.branches))
        )

        # Print plan tree
        self._vprint_header("PLAN TREE", "cyan")
        for i, branch in enumerate(self.branches):
            marker = "* " if i == self.branch_idx else "  "
            status = "[ACTIVE]" if i == self.branch_idx else "[EXHAUSTED]" if i < self.branch_idx else "[STANDBY]"
            origin = self._branch_origin_round[i] if i < len(self._branch_origin_round) else "?"
            self._vprint(f"  {marker}Branch {i} (r{origin})  {status}", "yellow" if i == self.branch_idx else "gray")
            for j, sg in enumerate(branch):
                cur = "  [CURRENT]" if (i == self.branch_idx and j == self.subgoal_idx) else ""
                self._vprint(f"      {j}: {sg}{cur}", "white" if i == self.branch_idx else "gray")

        self._log_plan_tree()
        return seen

    @property
    def _active_subgoals(self) -> List[str]:
        if self.branch_idx < len(self.branches):
            return self.branches[self.branch_idx]
        return []

    def _fast_explore_room(self, room_name: str, world_graph, agent_uid: int):
        """Directly populate the agent's world graph with all GT objects/furniture in room_name.

        Returns (summary_str, relation_lines, estimated_steps_saved).
        Mirrors OracleExploreSkill caps from oracle_explore.yaml defaults.
        """
        gt = self.env_interface.perception.gt_graph
        room_node = gt.get_node_from_name(room_name)
        if room_node is None:
            return f"fast_explore: room {room_name} not found in GT graph", [], 0

        # Mirror OracleExploreSkill caps from oracle_explore.yaml
        MAX_FURNITURE = 15
        MAX_STEPS_PER_FURNITURE = 300

        all_furniture = gt.get_furniture_in_room(room_node)
        num_furniture_to_visit = min(len(all_furniture), MAX_FURNITURE)
        estimated_steps = num_furniture_to_visit * MAX_STEPS_PER_FURNITURE

        nodes = [room_node]
        relation_lines = []
        for furn in all_furniture:
            nodes.append(furn)
            # Floor nodes connect objects directly; other furniture connects via Receptacle
            for obj in gt.get_neighbors_of_type(furn, Object):
                nodes.append(obj)
                relation_lines.append(f"    {obj.name} --[on]--> {furn.name}")
            for rec in gt.get_neighbors_of_type(furn, Receptacle):
                if rec.properties.get("type") == "within":
                    if furn.is_articulated():
                        is_open = furn.properties.get("states", {}).get("is_open")
                        if is_open is not None and not is_open:
                            continue  # furniture is closed — skip hidden objects
                for obj in gt.get_neighbors_of_type(rec, Object):
                    nodes.append(obj)
                    relation_lines.append(f"    {obj.name} --[on]--> {furn.name}")

        subgraph = gt.get_subgraph(nodes)
        world_graph.update(subgraph, partial_obs=True, update_mode="gt")

        summary = (
            f"Fast-explored {room_name}: {len(all_furniture)} furniture, "
            f"{len(relation_lines)} objects found. "
            f"Saved ~{estimated_steps} steps "
            f"({num_furniture_to_visit} furniture \u00d7 {MAX_STEPS_PER_FURNITURE} steps/furniture max)"
        )
        return summary, relation_lines, estimated_steps

    def _is_cg_partial_obs_mode(self) -> bool:
        world_model = getattr(getattr(self.env_interface, "conf", None), "world_model", None)
        wm_type = str(getattr(world_model, "type", "")).lower()
        return bool(getattr(self.env_interface, "partial_obs", False)) and wm_type == "concept_graph"

    def _required_objects_for_active_branch(self) -> List[str]:
        """Extract movable object names required by current branch subgoals."""
        required: List[str] = []
        for subgoal in self._active_subgoals:
            pred, args = self._parse_subgoal(subgoal)
            if not pred or not args:
                continue
            obj_name: Optional[str] = None
            if pred in ("picked", "pick", "holding"):
                obj_name = args[1] if len(args) >= 2 else args[0]
            elif pred in ("on", "placed-on", "place-on", "in", "inside", "within"):
                obj_name = args[0]
            if obj_name:
                if obj_name not in required:
                    required.append(obj_name)
        return required

    def _room_for_object_from_graph(self, world_graph, object_name: str) -> Optional[str]:
        """Best-effort object->room lookup using current graph relationships."""
        try:
            obj_node = world_graph.get_node_from_name(object_name)
        except Exception:
            return None

        try:
            furn = world_graph.find_furniture_for_object(obj_node)
            if furn is not None:
                return world_graph.get_room_for_entity(furn).name
        except Exception:
            pass

        try:
            return world_graph.get_room_for_entity(obj_node).name
        except Exception:
            pass

        try:
            if world_graph.is_object_with_agent(obj_node, agent_type="robot"):
                return world_graph.get_room_for_entity(world_graph.get_spot_robot()).name
            if world_graph.is_object_with_agent(obj_node, agent_type="human"):
                return world_graph.get_room_for_entity(world_graph.get_human()).name
        except Exception:
            pass
        return None

    def _missing_or_unlocated_required_objects(
        self, world_graph, required_objects: List[str]
    ) -> Tuple[List[str], List[str]]:
        known = {obj.name for obj in world_graph.get_all_objects()}
        missing: List[str] = []
        unlocated: List[str] = []
        for obj_name in required_objects:
            if obj_name not in known:
                missing.append(obj_name)
                continue
            if self._room_for_object_from_graph(world_graph, obj_name) is None:
                unlocated.append(obj_name)
        return missing, unlocated

    def _select_explore_room_for_objects(
        self, world_graph, target_objects: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        rooms = sorted(room.name for room in world_graph.get_all_rooms())
        if not rooms:
            return None, None
        # Prefer rooms not explored yet; then deterministic alphabetical order.
        room_order = sorted(rooms, key=lambda room: (room in self._explored_rooms, room))
        for obj_name in target_objects:
            room_counts = self._object_room_explore_counts.setdefault(obj_name, {})
            for room_name in room_order:
                if room_counts.get(room_name, 0) < self.max_room_explore_passes_per_object:
                    room_counts[room_name] = room_counts.get(room_name, 0) + 1
                    self._explored_rooms.add(room_name)
                    return room_name, obj_name
        return None, None

    def _advance_branch(self) -> bool:
        """Move to the next branch. Returns True if there is one, False if exhausted."""
        prev_idx = self.branch_idx
        next_idx = self.branch_idx + 1
        if next_idx < len(self.branches):
            self._vprint_header(
                f"BACKTRACK: branch {self.branch_idx} failed → trying branch {next_idx}", "yellow"
            )
            self._trace_append(
                f"Backtracking: branch {self.branch_idx} failed → trying branch {next_idx}"
            )
            self._log_event(
                {
                    "event": "branch_exhausted",
                    "reprompt_round": self._reprompt_round,
                    "branch": prev_idx,
                    "branch_origin": self._branch_origin(prev_idx),
                    "reason": "advance_branch",
                }
            )
            self.branch_idx = next_idx
            self.subgoal_idx = 0
            self.current_plan = []
            self.current_action_idx = 0
            self._subgoal_retry_count = 0
            self._log_event(
                {
                    "event": "branch_start",
                    "reprompt_round": self._reprompt_round,
                    "branch": self.branch_idx,
                    "branch_origin": self._branch_origin(),
                    "subgoal_idx": self.subgoal_idx,
                    "reason": "backtrack",
                }
            )

            # Print updated plan tree
            self._vprint_header("PLAN TREE (updated)", "cyan")
            for i, branch in enumerate(self.branches):
                marker = "* " if i == self.branch_idx else "  "
                status = "[ACTIVE]" if i == self.branch_idx else "[EXHAUSTED]" if i < self.branch_idx else "[STANDBY]"
                origin = self._branch_origin_round[i] if i < len(self._branch_origin_round) else "?"
                self._vprint(f"  {marker}Branch {i} (r{origin})  {status}", "yellow" if i == self.branch_idx else "gray")
                for j, sg in enumerate(branch):
                    cur = "  [CURRENT]" if (i == self.branch_idx and j == self.subgoal_idx) else ""
                    self._vprint(f"      {j}: {sg}{cur}", "white" if i == self.branch_idx else "gray")

            self._log_plan_tree()
            return True
        self._log_event(
            {
                "event": "branch_exhausted",
                "reprompt_round": self._reprompt_round,
                "branch": prev_idx,
                "branch_origin": self._branch_origin(prev_idx),
                "reason": "no_more_branches",
            }
        )
        return False

    # ------------------------------------------------------------------
    # PDDL planning helpers
    # ------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        return name.strip().replace(" ", "_")

    def _parse_subgoal(self, subgoal: str):
        if not subgoal:
            return None, []
        s = subgoal.strip()
        if "(" not in s or not s.endswith(")"):
            return None, []
        pred, arg_str = s.split("(", 1)
        pred = pred.strip().lower()
        args = [a.strip().strip("\"'") for a in arg_str[:-1].split(",")]
        args = [self._normalize_name(a) for a in args if a]
        return pred, args

    def _validate_subgoal(self, subgoal: str) -> bool:
        """Return False for subgoals with type violations (e.g. on(obj, movable))."""
        pred, args = self._parse_subgoal(subgoal)
        if not pred:
            return False
        if args and pred in ("explore", "opened-door", "opened-drawer"):
            key = ("explore:" if pred == "explore" else "inspect:") + args[0]
            if getattr(self, "_search_counts", {}).get(key, 0) >= getattr(self, "max_searches_per_target", 2):
                self._log_event({"event": "search_limit", "subgoal": subgoal, "target": key})
                return False
        movables = set(self._objects_by_type.get("movable", []))
        if pred in ("picked", "pick", "holding") and len(args) >= 1:
            obj_arg = args[1] if len(args) >= 2 else args[0]
            if obj_arg not in movables:
                self._vprint(
                    f"  [SKIP] {subgoal!r}: '{obj_arg}' is not in movable list "
                    f"(likely hallucinated/unknown; explore first).",
                    "red",
                )
                return False
        if pred in ("on", "in", "placed-on", "place-on", "inside", "within") and len(args) >= 1:
            if args[0] not in movables:
                self._vprint(
                    f"  [SKIP] {subgoal!r}: '{args[0]}' is not in movable list "
                    f"(likely hallucinated/unknown; explore first).",
                    "red",
                )
                return False
        if pred in ("on", "in") and len(args) >= 2 and args[1] in movables:
            self._vprint(
                f"  [SKIP] {subgoal!r}: '{args[1]}' is a movable, not a surface/space", "red"
            )
            return False
        if pred in ("on", "placed-on", "place-on") and len(args) >= 2:
            surface_furniture = set(self._objects_by_type.get("surface_furniture", []))
            if args[1] not in surface_furniture:
                self._vprint(
                    f"  [SKIP] {subgoal!r}: '{args[1]}' is not in the surface_furniture list "
                    f"(use a furniture name, not a receptacle id).",
                    "red",
                )
                return False
        if pred in ("in", "inside", "within") and len(args) >= 2:
            container_furniture = set(self._objects_by_type.get("container_furniture", []))
            if args[1] not in container_furniture:
                self._vprint(
                    f"  [SKIP] {subgoal!r}: '{args[1]}' is not in the container_furniture list "
                    f"(use a furniture name, not a receptacle id).",
                    "red",
                )
                return False
        joints = set(self._objects_by_type.get("joint", []))
        if pred in _SUBGOAL_JOINT_ARG_PREDS and len(args) >= 1:
            if args[0] not in joints:
                self._vprint(
                    f"  [SKIP] {subgoal!r}: '{args[0]}' is not in the joint list "
                    f"(use an openable furniture name from the joint list, not a room).",
                    "red",
                )
                return False
        if pred in ("explore", "searched-room", "search-room") and len(args) >= 1:
            rooms = set(self._objects_by_type.get("room", []))
            if args[0] not in rooms:
                self._vprint(
                    f"  [SKIP] {subgoal!r}: '{args[0]}' is not in the room list.",
                    "red",
                )
                return False
        return True

    def _subgoal_to_goal_literal(self, subgoal: str, agent_name: str, name_map: Dict[str, str]):
        pred, args = self._parse_subgoal(subgoal)
        if not pred:
            return None
        sanitized_map = {self._normalize_name(k): v for k, v in name_map.items()}

        def resolve(n):
            if n in name_map:
                return name_map[n]
            if n in name_map.values():
                return n
            if n in sanitized_map:
                return sanitized_map[n]
            key = n.lower()
            for k, v in name_map.items():
                if k.lower() == key:
                    return v
            return None

        if pred in ("picked", "pick", "holding") and len(args) >= 1:
            obj_arg = args[1] if len(args) >= 2 else args[0]
            obj = resolve(obj_arg)
            return ("holding", agent_name, obj) if obj else None
        if pred in ("on", "placed-on", "place-on") and len(args) >= 2:
            obj = resolve(args[0])
            furn = resolve(args[1])
            return ("on", obj, furn) if obj and furn else None
        if pred in ("in", "inside", "within") and len(args) >= 2:
            obj = resolve(args[0])
            furn = resolve(args[1])
            return ("in", obj, furn) if obj and furn else None
        if pred in ("opened-door", "opened-drawer", "open", "opened") and len(args) >= 1:
            j = resolve(args[0])
            return ("open", j) if j else None
        if pred in ("closed-door", "closed-drawer", "close", "closed") and len(args) >= 1:
            j = resolve(args[0])
            return ("closed", j) if j else None
        if pred in ("at",) and len(args) >= 2:
            room = resolve(args[1])
            return ("at", agent_name, room) if room else None
        if pred in ("handempty",):
            return ("handempty", agent_name)
        if pred in ("powered_on", "powered-on", "power_on", "turned-on", "on_state") and len(args) >= 1:
            obj = resolve(args[0])
            return ("powered_on", obj) if obj else None
        if pred in ("powered_off",) and len(args) >= 1:
            obj = resolve(args[0])
            return ("powered_off", obj) if obj else None
        if pred in ("filled", "fill", "is_filled") and len(args) >= 1:
            obj = resolve(args[0])
            return ("filled", obj) if obj else None
        if pred in ("poured_into", "poured-into", "pour") and len(args) >= 1:
            obj = resolve(args[0])
            return ("filled", obj) if obj else None
        if pred in ("cleaned", "clean", "is_clean") and len(args) >= 1:
            obj = resolve(args[0])
            return ("cleaned", obj) if obj else None
        return None

    def _plan_for_subgoal(self, subgoal: str, world_graph, agent_uid: int):
        # First pass: unscoped problem just to get the name maps for goal parsing
        dummy_goal = ("handempty", f"agent_{agent_uid}")
        _problem, _init, name_map, reverse_map = build_pddlstream_problem(
            world_graph, agent_uid, dummy_goal
        )
        agent_name = name_map.get(f"agent_{agent_uid}", f"agent_{agent_uid}")
        goal_literal = self._subgoal_to_goal_literal(subgoal, agent_name, name_map)

        if goal_literal is None:
            self._vprint(f"  [WARN] Could not parse subgoal to PDDL literal: {subgoal!r}", "red")
            self._subgoal_exec_log_append(
                f"  [WARN] Could not parse subgoal to PDDL literal: {subgoal!r}"
            )
            return None, name_map, reverse_map, _init

        # Second pass: scoped problem — only include entities relevant to this subgoal.
        # This prevents grounding explosion (e.g. place_on with 60 furniture × 8 objects).
        scope = extract_scope_names(world_graph, agent_uid, goal_literal)
        problem, init, name_map, reverse_map = build_pddlstream_problem(
            world_graph, agent_uid, goal_literal, scope_names=scope
        )
        self._log_pddl_problem(problem, subgoal, init)

        n_scope = len(scope) if scope else "full"
        hdr = f"PDDL SOLVER  Branch {self.branch_idx} | Subgoal {self.subgoal_idx}: {subgoal}"
        self._subgoal_exec_log_begin_subgoal()
        self._vprint_header(hdr, "blue")
        self._subgoal_exec_log_append(f"══ {hdr} ══")
        self._vprint(f"  Goal:  {goal_literal}", "white")
        self._subgoal_exec_log_append(f"  Goal:  {goal_literal}")
        self._vprint(f"  Scope: {n_scope} entities  |  Init: {len(init)} facts", "gray")
        self._subgoal_exec_log_append(
            f"  Scope: {n_scope} entities  |  Init: {len(init)} facts"
        )

        if goal_literal in init:
            self._vprint("  [already satisfied — skipping]", "green")
            self._subgoal_exec_log_append("  [already satisfied — skipping]")
            self._trace_append(f"Subgoal already satisfied: {subgoal}")
            return [], name_map, reverse_map, init

        solution = solve_pddlstream_problem(problem)
        plan, _cost, _evaluations = solution if solution is not None else (None, 0, [])
        self._trace_append(f"Subgoal: {subgoal}")
        self._trace_append(f"PDDL plan: {plan}")

        if plan is None or plan == []:
            if plan is None:
                self._vprint("  [FAIL] PDDLStream found no plan.", "red")
                self._subgoal_exec_log_append("  [FAIL] PDDLStream found no plan.")
            return plan, name_map, reverse_map, init

        self._vprint(f"  Plan ({len(plan)} steps):", "white")
        self._subgoal_exec_log_append(f"  Plan ({len(plan)} steps):")
        for i, step in enumerate(plan):
            line = f"    {i + 1}. {step[0]}({', '.join(str(a) for a in step[1])})"
            self._vprint(line, "gray")
            self._subgoal_exec_log_append(line)

        return plan, name_map, reverse_map, init

    # ------------------------------------------------------------------
    # Action execution helpers
    # ------------------------------------------------------------------

    def _restore_name(self, name: str) -> str:
        return self._reverse_name_map.get(name, name)

    def _action_to_tool(self, action, next_action=None):
        name, args = action
        name = name.lower()
        if name == "navigate":
            # Derive the navigation target from the next action rather than
            # the room-witness ?x, which PDDL picks arbitrarily.
            if next_action is not None:
                nname = next_action[0].lower()
                nargs = next_action[1]
                if nname == "pick":
                    # Navigate to the object itself; oracle_nav handles objects.
                    return "Navigate", self._restore_name(nargs[1])
                if nname in ("place_on", "place_in"):
                    # Navigate to the destination furniture.
                    return "Navigate", self._restore_name(nargs[2])
                if nname in ("open", "close", "power_on", "power_off", "power_on_object", "power_off_object", "clean_furniture"):
                    return "Navigate", self._restore_name(nargs[1])
                if nname == "clean_object":
                    # clean_object(?a, ?o, ?f, ?r) — navigate to faucet, not the object
                    return "Navigate", self._restore_name(nargs[2])
                if nname in ("fill", "fill_held"):
                    # fill(?a, ?o, ?f, ?r) — navigate to the faucet furniture, not the object
                    return "Navigate", self._restore_name(nargs[2])
            # Fallback: use the room-witness entity PDDL chose.
            return "Navigate", self._restore_name(args[1])
        if name == "pick":
            return "Pick", self._restore_name(args[1])
        if name == "place_on":
            obj = self._restore_name(args[1])
            furn = self._restore_name(args[2])
            return "Place", f"{obj}, on, {furn}, None, None"
        if name == "place_in":
            obj = self._restore_name(args[1])
            furn = self._restore_name(args[2])
            return "Place", f"{obj}, within, {furn}, None, None"
        if name == "open":
            return "Open", self._restore_name(args[1])
        if name == "close":
            return "Close", self._restore_name(args[1])
        if name in ("power_on", "power_on_object"):
            return "PowerOn", self._restore_name(args[1])
        if name in ("power_off", "power_off_object"):
            return "PowerOff", self._restore_name(args[1])
        if name in ("fill", "fill_held"):
            return "Fill", self._restore_name(args[1])
        if name in ("clean_object", "clean_furniture"):
            return "Clean", self._restore_name(args[1])
        if name == "pour":
            return "Pour", self._restore_name(args[2])
        return None, None

    def _response_failed(self, response: str) -> bool:
        if response is None:
            return False
        r = str(response).lower()
        fail_tokens = (
            "fail",
            "error",
            "could not",
            "unable",
            "cannot",
            "not possible",
            "unsuccessful",
            # MotorSkillTool: get_node_from_name / invalid entity (no "fail" substring)
            "not present in the graph",
            "not present in graph",
        )
        return any(t in r for t in fail_tokens)

    def _planner_info(
        self,
        agent_uid: int,
        *,
        hl_override: Optional[Dict[int, Tuple[str, str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build planner info for the decentralized evaluation runner.

        The runner and video code expect ``high_level_actions`` on every step
        (see ``evaluation_runner.run_instruction``).
        """
        hl: Dict[int, Tuple[str, str, str]]
        if hl_override is not None:
            hl = dict(hl_override)
        else:
            hl = dict(self.last_high_level_actions)
        out: Dict[str, Any] = {"high_level_actions": hl, "traces": {agent_uid: self.trace}}
        out.update(kwargs)
        out["cost_metrics"] = {"llm_planning_time_s": getattr(self, "_vlm_wall_s", 0.0),
                               "llm_requests": getattr(self, "_vlm_calls", 0),
                               "physical_explore": not self.explore_fast}
        usage = snapshot_from_llm(self.vlm)
        if usage:
            cost = dict(out.get("cost_metrics") or {})
            cost.update(usage)
            out["cost_metrics"] = cost
        return out

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def get_next_action(self, instruction: str, observations: Dict[str, Any], world_graphs: Dict[int, Any]):
        self._trace_observations = observations
        if self.is_done:
            uid = self._agents[0].uid if self._agents else 0
            return {}, self._planner_info(uid, hl_override={}), True

        agent_uid = self._agents[0].uid if self._agents else 0
        world_graph = world_graphs[agent_uid]

        if not self.trace:
            self._trace_append(f"Task: {instruction}")

        # ── Episode banner (printed once) ────────────────────────────────
        if not self._episode_banner_printed:
            self._vprint_banner(instruction)
            self._episode_banner_printed = True

        # ── Phase 1: generate subgoal branches if none yet ──────────────
        if not self.branches:
            self._generate_subgoals(instruction, world_graph, observations, append=False)
            if not self.branches:
                self._vprint("  [ABORT] No subgoals generated — ending episode.", "red")
                self.is_done = True
                return {}, self._planner_info(agent_uid, hl_override={}, is_done={agent_uid: True}), True

        # ── Partial-obs CG memory: explore for missing required objects ─
        if (
            self.enable_partial_obs_explore
            and self.explore_on_missing_required_objects
            and self._is_cg_partial_obs_mode()
            and not self.last_high_level_actions
            and self.current_action_idx >= len(self.current_plan)
        ):
            required_objects = self._required_objects_for_active_branch()
            if required_objects:
                missing, unlocated = self._missing_or_unlocated_required_objects(
                    world_graph, required_objects
                )
                targets = missing if missing else unlocated
                if targets:
                    room_name, trigger_obj = self._select_explore_room_for_objects(
                        world_graph, targets
                    )
                    if room_name:
                        reason = (
                            "missing"
                            if trigger_obj in missing
                            else "unlocated"
                        )
                        self._subgoal_exec_log_clear()
                        self._subgoal_exec_log_begin_subgoal()
                        self._vprint(
                            f"  [EXPLORE] Required object '{trigger_obj}' is {reason} in CG memory; "
                            f"exploring room '{room_name}'.",
                            "yellow",
                        )
                        self._trace_append(
                            f"Explore triggered: {trigger_obj} is {reason}; Explore[{room_name}]"
                        )
                        self._subgoal_exec_log_append(
                            f"  [EXPLORE] Required object '{trigger_obj}' is {reason} in CG memory; "
                            f"exploring room '{room_name}'."
                        )
                        self._subgoal_exec_log_append(f"  Action: Explore[{room_name}]")
                        self._log_event(
                            {
                                "event": "explore_triggered",
                                "reprompt_round": self._reprompt_round,
                                "branch": self.branch_idx,
                                "subgoal_idx": self.subgoal_idx,
                                "reason": reason,
                                "trigger_object": trigger_obj,
                                "target_room": room_name,
                                "required_objects": list(required_objects),
                                "missing_objects": list(missing),
                                "unlocated_objects": list(unlocated),
                            }
                        )
                        # Keep an in-flight marker so we don't progress Phase 2
                        # while Explore is still executing.
                        self.current_plan = [("__special_explore__", [room_name])]
                        self.current_action_idx = 0
                        self._pending_replan_after_explore = True
                        self._begin_explore_images(observations, world_graph, room_name)
                        hl_action = {agent_uid: ("Explore", room_name, None)}
                        self.last_high_level_actions = hl_action
                        low_level_actions, responses = self.process_high_level_actions(
                            hl_action, observations
                        )
                        return (
                            low_level_actions,
                            self._planner_info(agent_uid, responses=responses),
                            False,
                        )

        # ── Phase 2: need a PDDL plan for the current subgoal ───────────
        subgoals = self._active_subgoals
        if self.current_action_idx >= len(self.current_plan):
            if self.subgoal_idx >= len(subgoals):
                self._vprint_header(
                    f"ALL SUBGOALS DONE  (branch {self.branch_idx}, "
                    f"{self._subgoals_completed}/{sum(len(b) for b in self.branches[:self.branch_idx + 1])} completed)",
                    "green",
                )
                self._finish_planning(
                    "plan_completed", "All subgoals in the active branch are finished."
                )
                return {}, self._planner_info(
                    agent_uid, hl_override={}, is_done={agent_uid: True}
                ), True

            subgoal = subgoals[self.subgoal_idx]
            self._log_subgoal_status(
                status="started",
                subgoal=subgoal,
                branch_idx=self.branch_idx,
                subgoal_idx=self.subgoal_idx,
            )
            pred, args = self._parse_subgoal(subgoal)
            if (
                self.enable_vlm_explore_subgoal
                and pred in ("explore", "searched-room", "search-room")
                and len(args) >= 1
            ):
                room_name = args[0]
                self._subgoal_exec_log_begin_subgoal()
                self._vprint_header(
                    f"HIGH-LEVEL EXPLORE  Branch {self.branch_idx} | Subgoal {self.subgoal_idx}: {subgoal}",
                    "blue",
                )
                self._vprint(f"  Action: Explore[{room_name}]", "white")
                self._subgoal_exec_log_append(
                    f"HIGH-LEVEL EXPLORE  Branch {self.branch_idx} | Subgoal {self.subgoal_idx}: {subgoal}"
                )
                self._subgoal_exec_log_append(f"  Action: Explore[{room_name}]")
                self._trace_append(f"Subgoal: {subgoal}")
                self._trace_append(f"High-level special-case: Explore[{room_name}]")
                self._log_event(
                    {
                        "event": "special_subgoal_action",
                        "reprompt_round": self._reprompt_round,
                        "branch": self.branch_idx,
                        "subgoal_idx": self.subgoal_idx,
                        "subgoal": subgoal,
                        "action": "Explore",
                        "args": room_name,
                    }
                )
                if self.explore_fast:
                    self._begin_explore_images(observations, world_graph, room_name)
                    # Fast path: directly populate world graph from GT, no navigation.
                    summary, relation_lines, _est_steps = self._fast_explore_room(
                        room_name, world_graph, agent_uid
                    )
                    self._vprint(f"  [fast-explore] {summary}", "green")
                    for line in relation_lines:
                        self._vprint(line, "cyan")
                    log_text = summary + ("\n" + "\n".join(relation_lines) if relation_lines else "")
                    self._subgoal_exec_log_append(log_text)
                    self._explored_rooms.add(room_name)
                    self.history.append(f"explore({room_name})")
                    self._trace_append(f"Recorded formal action: explore({room_name}) (fast)")
                    self._log_subgoal_status(
                        status="success",
                        subgoal=subgoal,
                        branch_idx=self.branch_idx,
                        subgoal_idx=self.subgoal_idx,
                    )
                    self._emit_subgoal_execution("solved", subgoal)
                    self.subgoal_idx += 1
                    history_str = build_failure_history(
                        actions=self.history,
                        explore_room=room_name,
                        already_succeeded=self._already_succeeded_subgoals,
                    )
                    self._last_replanning_history_str = history_str
                    self._generate_subgoals(
                        instruction,
                        world_graph,
                        observations,
                        history_str=history_str,
                        append=True,
                        after_explore=True,
                        explored_room=room_name,
                    )
                    return {}, self._planner_info(
                        agent_uid, hl_override={}, replanned={agent_uid: True}
                    ), False

                # Mark an in-flight special step so Phase 2 doesn't keep re-issuing
                # Explore on every tick while the skill is still executing.
                self.current_plan = [("__special_explore__", [room_name])]
                self.current_action_idx = 0
                self._pending_replan_after_explore = True
                self._begin_explore_images(observations, world_graph, room_name)
                hl_action = {agent_uid: ("Explore", room_name, None)}
                self.last_high_level_actions = hl_action
                low_level_actions, responses = self.process_high_level_actions(
                    hl_action, observations
                )
                return (
                    low_level_actions,
                    self._planner_info(agent_uid, responses=responses),
                    False,
                )
            plan, _name_map, reverse_map, _init = self._plan_for_subgoal(subgoal, world_graph, agent_uid)
            self._reverse_name_map = reverse_map or {}
            self._log_event({
                "event": "pddl_plan",
                "reprompt_round": self._reprompt_round,
                "branch": self.branch_idx,
                "branch_origin": self._branch_origin_round[self.branch_idx] if self.branch_idx < len(self._branch_origin_round) else None,
                "subgoal_idx": self.subgoal_idx,
                "subgoal": subgoal,
                "status": "failed" if plan is None else "already" if plan == [] else "planned",
                "failure_type": "pddl_no_plan" if plan is None else None,
                "plan": plan,
            })

            if plan is None:
                failure_msg = f"Failed to plan for subgoal: {subgoal}"
                self._last_failure = failure_msg
                self.history.append(failure_msg)
                self._trace_append(failure_msg)
                self._log_subgoal_status(
                    status="failed",
                    subgoal=subgoal,
                    branch_idx=self.branch_idx,
                    subgoal_idx=self.subgoal_idx,
                    failure_type="pddl_no_plan",
                    failure_msg=failure_msg,
                )
                self._emit_subgoal_execution("failed", subgoal)

                if self._advance_branch():
                    return self.get_next_action(instruction, observations, world_graphs)

                self.reprompt_count += 1
                if self.reprompt_count > self.max_reprompts:
                    self._vprint_header(
                        f"MAX REPROMPTS REACHED ({self.max_reprompts}) — giving up", "red"
                    )
                    self.is_done = True
                    return {}, self._planner_info(agent_uid, hl_override={}, is_done={agent_uid: True}), True

                self._vprint_header(
                    f"REPLANNING (attempt {self.reprompt_count}/{self.max_reprompts})", "yellow"
                )
                self._reprompt_round += 1
                self._log_event(
                    {
                        "event": "reprompt_started",
                        "reprompt_round": self._reprompt_round,
                        "reason": "pddl_no_plan",
                        "failed_branch": self.branch_idx,
                        "failed_subgoal_idx": self.subgoal_idx,
                        "failed_subgoal": subgoal,
                        "failure_msg": self._last_failure,
                    }
                )
                history_str = build_failure_history(
                    actions=self.history,
                    failure=self._last_failure,
                    already_succeeded=self._already_succeeded_subgoals,
                )
                self._last_replanning_history_str = history_str
                self._generate_subgoals(
                    instruction, world_graph, observations, history_str=history_str,
                    append=True, refresh_failure_observation=True,
                )
                return {}, self._planner_info(agent_uid, hl_override={}, replanned={agent_uid: True}), False

            if plan == []:
                self._vprint_subgoal_status_block(
                    f"\n  ✓ Subgoal {self.subgoal_idx} already satisfied: {subgoal}",
                    "green",
                )
                self._log_subgoal_status(
                    status="already",
                    subgoal=subgoal,
                    branch_idx=self.branch_idx,
                    subgoal_idx=self.subgoal_idx,
                )
                self._emit_subgoal_execution(
                    "already", subgoal, extra_trailing_delimiter=False
                )
                self._log_plan_tree()
                self._subgoals_completed += 1
                if subgoal not in self._already_succeeded_subgoals:
                    self._already_succeeded_subgoals.append(subgoal)
                self.subgoal_idx += 1
                return self.get_next_action(instruction, observations, world_graphs)

            self.current_plan = plan
            self.current_action_idx = 0

        # ── Phase 3: execute / continue current action ──────────────────
        if self.last_high_level_actions:
            hl_snapshot = dict(self.last_high_level_actions)
            _active_action = hl_snapshot.get(agent_uid, (None, None, None))[0]
            if _active_action == "Explore":
                self._capture_explore_image(observations, world_graph)
                # Capture detailed Explore execution prints/logs so they can be
                # surfaced in subgoal_execution logs and rendered in HTML.
                _capture = io.StringIO()
                _tee_out = _TeeStream(sys.stdout, _capture)
                _tee_err = _TeeStream(sys.stderr, _capture)
                with contextlib.redirect_stdout(_tee_out), contextlib.redirect_stderr(
                    _tee_err
                ):
                    low_level_actions, responses = self.process_high_level_actions(
                        self.last_high_level_actions, observations
                    )
                _captured_text = _capture.getvalue().strip()
                if _captured_text:
                    self._subgoal_exec_log_append(_captured_text)
            else:
                low_level_actions, responses = self.process_high_level_actions(
                    self.last_high_level_actions, observations
                )
            if any(responses.values()):
                if _active_action == "Explore":
                    self._capture_explore_image(observations, world_graph, force=True)
                response = list(responses.values())[0]
                self._vprint(f"    Obs: {response}", "gray")
                self._trace_append(f"Observation: {response}")
                self._subgoal_exec_log_append(f"    Obs: {response}")
                if self._response_failed(response):
                    self._subgoal_exec_log_append(
                        "================================================================================\n"
                        "✗ ACTION RESULT: FAILED\n"
                        "================================================================================\n"
                        f"{response}"
                    )
                else:
                    self._subgoal_exec_log_append(
                        "================================================================================\n"
                        "✓ ACTION RESULT: SUCCESS\n"
                        "================================================================================\n"
                        f"{response}"
                    )

                # For high-level Explore, include the full execution block in the
                # plain trace text so trace HTML can show rich per-action logs.
                if hl_snapshot.get(agent_uid, (None, None, None))[0] == "Explore":
                    self._append_subgoal_exec_log_to_trace()

                if self._response_failed(response):
                    failure_msg = f"Action failed: {self.last_high_level_actions} -> {response}"
                    self._last_failure = failure_msg
                    self.history.append(failure_msg)
                    self._trace_append(failure_msg)
                    self._vprint(f"  [FAIL] {failure_msg}", "red")
                    active_subgoal = (
                        subgoals[self.subgoal_idx]
                        if self.subgoal_idx < len(subgoals)
                        else "unknown_subgoal"
                    )
                    self._log_subgoal_status(
                        status="failed",
                        subgoal=active_subgoal,
                        branch_idx=self.branch_idx,
                        subgoal_idx=self.subgoal_idx,
                        failure_type="execution_failure",
                        failure_msg=failure_msg,
                        extra={
                            "last_high_level_actions": self.last_high_level_actions,
                            "response": response,
                            "retry_count": self._subgoal_retry_count,
                            "ntamp_max_pddl_replan_retries": self.ntamp_max_pddl_replan_retries,
                        },
                    )
                    self._subgoal_exec_log_append(f"  [FAIL] {failure_msg}")

                    # Clear only the current plan step — preserve subgoal_idx so we
                    # re-plan for the same subgoal from the current world state.
                    self.current_plan = []
                    self.current_action_idx = 0
                    self._subgoal_retry_count += 1

                    if self._subgoal_retry_count <= self.ntamp_max_pddl_replan_retries:
                        # Re-plan for the same subgoal from the current world state (no VLM)
                        # up to ntamp_max_pddl_replan_retries consecutive execution failures.
                        self._vprint(
                            f"  [RETRY] Re-planning for subgoal {self.subgoal_idx} "
                            f"from current (newer) state "
                            f"(retry {self._subgoal_retry_count}/{self.ntamp_max_pddl_replan_retries})",
                            "yellow",
                        )
                        self._subgoal_exec_log_append(
                            f"  [RETRY] Re-planning for subgoal {self.subgoal_idx} "
                            f"from current (newer) state "
                            f"(retry {self._subgoal_retry_count}/{self.ntamp_max_pddl_replan_retries})"
                        )
                        self.last_high_level_actions = {}
                        return self.get_next_action(instruction, observations, world_graphs)

                    # Retry exhausted — escalate: try next branch, then reprompt.
                    self._subgoal_retry_count = 0
                    # Flush before either operation changes branch/subgoal identity.
                    self._emit_subgoal_execution("failed", active_subgoal)
                    if self._advance_branch():
                        self.last_high_level_actions = {}
                        return self.get_next_action(instruction, observations, world_graphs)

                    self.reprompt_count += 1
                    if self.reprompt_count > self.max_reprompts:
                        self._vprint_header(
                            f"MAX REPROMPTS REACHED ({self.max_reprompts}) — giving up", "red"
                        )
                        self.is_done = True
                        return low_level_actions, self._planner_info(
                            agent_uid, hl_override=dict(self.last_high_level_actions), is_done={agent_uid: True}
                        ), True

                    self._vprint_header(
                        f"REPLANNING (attempt {self.reprompt_count}/{self.max_reprompts})", "yellow"
                    )
                    self._reprompt_round += 1
                    self._log_event(
                        {
                            "event": "reprompt_started",
                            "reprompt_round": self._reprompt_round,
                            "reason": "execution_failure",
                            "failed_branch": self.branch_idx,
                            "failed_subgoal_idx": self.subgoal_idx,
                            "failed_subgoal": (
                                subgoals[self.subgoal_idx]
                                if self.subgoal_idx < len(subgoals)
                                else "unknown_subgoal"
                            ),
                            "failure_msg": self._last_failure,
                        }
                    )
                    history_str = build_failure_history(
                        actions=self.history,
                        failure=self._last_failure,
                        already_succeeded=self._already_succeeded_subgoals,
                    )
                    self._last_replanning_history_str = history_str
                    self._generate_subgoals(
                        instruction, world_graph, observations, history_str=history_str,
                        append=True, refresh_failure_observation=True,
                    )
                else:
                    if _active_action == "Open":
                        target = hl_snapshot[agent_uid][1]
                        active = subgoals[self.subgoal_idx]
                        if self.current_action_idx + 1 >= len(self.current_plan):
                            self._log_subgoal_status("solved", active)
                            self._emit_subgoal_execution("solved", active)
                            self._subgoals_completed += 1
                            if active not in self._already_succeeded_subgoals:
                                self._already_succeeded_subgoals.append(active)
                        else:
                            self._emit_subgoal_execution("observation_boundary", active)
                        self._search_counts = getattr(self, "_search_counts", {})
                        key = "inspect:" + str(target)
                        self._search_counts[key] = self._search_counts.get(key, 0) + 1
                        self.history.append(f"opened({target})")
                        self._observe_and_continue(instruction, observations, world_graph, "opened_furniture", target)
                        return low_level_actions, self._planner_info(agent_uid, hl_override=hl_snapshot, responses=responses), self.is_done
                    self.current_action_idx += 1
                    # Check if subgoal's plan is now fully executed
                    if self.current_action_idx >= len(self.current_plan):
                        if (
                            self._pending_replan_after_explore
                            and hl_snapshot.get(agent_uid, (None, None, None))[0] == "Explore"
                        ):
                            self._pending_replan_after_explore = False
                            self.current_plan = []
                            self.current_action_idx = 0
                            self._subgoal_retry_count = 0
                            self.last_high_level_actions = {}

                            if self.subgoal_idx < len(subgoals):
                                sg_ex = subgoals[self.subgoal_idx]
                                explore_pred, _ = self._parse_subgoal(sg_ex)
                                if explore_pred in ("explore", "searched-room", "search-room"):
                                    self._vprint_subgoal_complete_block(sg_ex)
                                    self._log_subgoal_status(
                                        status="solved", subgoal=sg_ex,
                                        branch_idx=self.branch_idx,
                                        subgoal_idx=self.subgoal_idx,
                                    )
                                    self._emit_subgoal_execution(
                                        "solved", sg_ex, extra_trailing_delimiter=False
                                    )
                                    self._subgoals_completed += 1
                                    if sg_ex not in self._already_succeeded_subgoals:
                                        self._already_succeeded_subgoals.append(sg_ex)
                                else:
                                    # Searching for a missing object does not
                                    # achieve the manipulation goal that needed it.
                                    self._emit_subgoal_execution("observation_boundary", sg_ex)
                                self._log_plan_tree()

                            self._vprint_header("REPLANNING AFTER EXPLORE", "yellow")
                            self._reprompt_round += 1
                            self._log_event(
                                {
                                    "event": "reprompt_started",
                                    "reprompt_round": self._reprompt_round,
                                    "reason": "explore_refresh",
                                    "branch": self.branch_idx,
                                    "subgoal_idx": self.subgoal_idx,
                                    "subgoal": (
                                        subgoals[self.subgoal_idx]
                                        if self.subgoal_idx < len(subgoals)
                                        else "unknown_subgoal"
                                    ),
                                    "last_action": hl_snapshot.get(agent_uid),
                                }
                            )
                            _tup = hl_snapshot.get(agent_uid, (None, None, None))
                            _explored_room = (
                                _tup[1]
                                if len(_tup) > 1 and _tup[0] == "Explore"
                                else None
                            )
                            # self.history is otherwise only failures; record successful
                            # explore(...) so the VLM "already taken actions" block is accurate.
                            if _explored_room:
                                self.history.append(f"explore({_explored_room})")
                                self._trace_append(
                                    f"Recorded formal action: explore({_explored_room}) (success)"
                                )
                            refresh_reason = (
                                f"state updated after explore({_explored_room}); re-plan with new observations"
                                if _explored_room
                                else "state updated after Explore; re-plan with new observations"
                            )
                            history_str = build_failure_history(
                                actions=self.history,
                                failure=refresh_reason,
                                already_succeeded=self._already_succeeded_subgoals,
                            )
                            self._last_replanning_history_str = history_str
                            self._generate_subgoals(
                                instruction,
                                world_graph,
                                observations,
                                history_str=history_str,
                                append=True,
                                after_explore=True,
                                explored_room=_explored_room,
                            )
                            return {}, self._planner_info(
                                agent_uid, hl_override={}, replanned={agent_uid: True}
                            ), False

                        done_sg = subgoals[self.subgoal_idx]
                        self._vprint_subgoal_complete_block(done_sg)
                        self._subgoals_completed += 1
                        self._log_subgoal_status(
                            status="solved",
                            subgoal=done_sg,
                            branch_idx=self.branch_idx,
                            subgoal_idx=self.subgoal_idx,
                        )
                        self._emit_subgoal_execution(
                            "solved", done_sg, extra_trailing_delimiter=False
                        )
                        self._log_plan_tree()
                        if done_sg not in self._already_succeeded_subgoals:
                            self._already_succeeded_subgoals.append(done_sg)
                        self.subgoal_idx += 1
                        self.current_plan = []
                        self.current_action_idx = 0
                        self._subgoal_retry_count = 0

                self.last_high_level_actions = {}
            return (
                low_level_actions,
                self._planner_info(agent_uid, hl_override=hl_snapshot, responses=responses),
                False,
            )

        action = self.current_plan[self.current_action_idx]
        next_action = (
            self.current_plan[self.current_action_idx + 1]
            if self.current_action_idx + 1 < len(self.current_plan)
            else None
        )
        tool_name, tool_args = self._action_to_tool(action, next_action=next_action)
        if tool_name is None:
            unsupported_msg = f"Unsupported action: {action}"
            self.history.append(unsupported_msg)
            self._trace_append(unsupported_msg)
            self._vprint(f"  [WARN] {unsupported_msg}", "yellow")
            active_subgoal = (
                subgoals[self.subgoal_idx] if self.subgoal_idx < len(subgoals) else "unknown_subgoal"
            )
            self._log_subgoal_status(
                status="failed",
                subgoal=active_subgoal,
                branch_idx=self.branch_idx,
                subgoal_idx=self.subgoal_idx,
                failure_type="unsupported_action",
                failure_msg=unsupported_msg,
            )
            self.current_plan = []
            self.current_action_idx = 0
            return {}, self._planner_info(agent_uid, hl_override={}, replanned={agent_uid: True}), False

        step_num = self.current_action_idx + 1
        total = len(self.current_plan)
        self._vprint(f"\n  ▶ [{step_num}/{total}]  {tool_name}[{tool_args}]", "cyan")
        self._subgoal_exec_log_append(
            f"\n  ▶ [{step_num}/{total}]  {tool_name}[{tool_args}]"
        )
        self._trace_append(f"Action: {tool_name}[{tool_args}]")

        hl_action = {agent_uid: (tool_name, tool_args, None)}
        self.last_high_level_actions = hl_action
        low_level_actions, responses = self.process_high_level_actions(hl_action, observations)
        return (
            low_level_actions,
            self._planner_info(agent_uid, responses=responses),
            False,
        )
