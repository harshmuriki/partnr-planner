"""Episode timing and sim-step cost summaries for console and HTML traces."""

from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SIM_FREQ = 120.0


def _as_nonneg_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number < 0:
            return None
        return number
    return None


def _as_nonneg_int(value: Any) -> Optional[int]:
    number = _as_nonneg_float(value)
    if number is None:
        return None
    return int(number)


def llm_request_total(llm_requests: Any) -> int:
    if isinstance(llm_requests, dict):
        return int(sum(int(v) for v in llm_requests.values() if isinstance(v, (int, float))))
    number = _as_nonneg_int(llm_requests)
    return number if number is not None else 0


def exact_action_sim_steps(action_sim_steps: Any) -> int:
    """Exact env.step counts; skip Explore (fast_explore does not walk)."""
    if not isinstance(action_sim_steps, dict):
        return 0
    total = 0
    for name, steps in action_sim_steps.items():
        if name == "Explore":
            continue
        count = _as_nonneg_int(steps)
        if count is not None:
            total += count
    return total


def sim_time_from_steps(steps: Any, sim_freq: float = DEFAULT_SIM_FREQ) -> Optional[float]:
    count = _as_nonneg_int(steps)
    freq = _as_nonneg_float(sim_freq)
    if count is None or not freq:
        return None
    return count / freq


def fmt_seconds(value: Any, digits: int = 2) -> str:
    number = _as_nonneg_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}s"


def fmt_steps(value: Any) -> str:
    number = _as_nonneg_int(value)
    if number is None:
        return "N/A"
    return str(number)


def fmt_meters(value: Any) -> str:
    number = _as_nonneg_float(value)
    if number is None:
        return "N/A"
    return f"{number:.1f} m"


def action_sim_rows(
    action_sim_steps: Any,
    sim_freq: float = DEFAULT_SIM_FREQ,
) -> List[Tuple[str, int, float]]:
    if not isinstance(action_sim_steps, dict):
        return []
    rows: List[Tuple[str, int, float]] = []
    for name in sorted(action_sim_steps.keys()):
        if name == "Explore":
            continue
        steps = _as_nonneg_int(action_sim_steps.get(name, 0)) or 0
        time_s = sim_time_from_steps(steps, sim_freq) or 0.0
        rows.append((str(name), steps, time_s))
    return rows


def build_episode_cost(
    *,
    action_sim_steps: Any = None,
    action_counts: Any = None,
    llm_planning_time_s: Any = None,
    llm_requests: Any = None,
    explore_approx_sim_steps: Any = None,
    explore_approx_sim_time_s: Any = None,
    explore_approx_meters: Any = None,
    runtime: Any = None,
    sim_step_count: Any = None,
    sim_freq: float = DEFAULT_SIM_FREQ,
    task_percent_complete: Any = None,
    task_state_success: Any = None,
) -> Dict[str, Any]:
    exact_steps = exact_action_sim_steps(action_sim_steps)
    exact_time_s = sim_time_from_steps(exact_steps, sim_freq)
    explore_steps = _as_nonneg_int(explore_approx_sim_steps)
    explore_time_s = _as_nonneg_float(explore_approx_sim_time_s)
    if explore_time_s is None and explore_steps is not None:
        explore_time_s = sim_time_from_steps(explore_steps, sim_freq)
    combined_steps = exact_steps + (explore_steps or 0)
    combined_time_s = (exact_time_s or 0.0) + (explore_time_s or 0.0)
    llm_time_s = _as_nonneg_float(llm_planning_time_s)
    llm_calls = llm_request_total(llm_requests)
    llm_avg_s = None
    if llm_time_s is not None and llm_calls > 0:
        llm_avg_s = llm_time_s / llm_calls
    runtime_s = _as_nonneg_float(runtime)
    non_llm_wall_s = None
    if runtime_s is not None and llm_time_s is not None:
        non_llm_wall_s = max(0.0, runtime_s - llm_time_s)

    env_steps = _as_nonneg_int(sim_step_count)
    if env_steps is None:
        env_steps = exact_steps

    return {
        "sim_freq": sim_freq,
        "exact_sim_steps": exact_steps,
        "exact_sim_time_s": exact_time_s,
        "explore_approx_sim_steps": explore_steps,
        "explore_approx_sim_time_s": explore_time_s,
        "explore_approx_meters": _as_nonneg_float(explore_approx_meters),
        "combined_sim_steps": combined_steps,
        "combined_sim_time_s": combined_time_s,
        "llm_planning_time_s": llm_time_s,
        "llm_requests": llm_calls,
        "llm_avg_s": llm_avg_s,
        "runtime_s": runtime_s,
        "non_llm_wall_s": non_llm_wall_s,
        "sim_step_count": env_steps,
        "action_rows": action_sim_rows(action_sim_steps, sim_freq),
        "action_counts": action_counts if isinstance(action_counts, dict) else {},
        "task_percent_complete": _as_nonneg_float(task_percent_complete),
        "task_state_success": _as_nonneg_float(task_state_success),
    }


def console_cost_lines(cost: Dict[str, Any]) -> List[str]:
    """Human-readable timing block for terminal logs."""
    llm_calls = int(cost.get("llm_requests") or 0)
    llm_avg = cost.get("llm_avg_s")
    avg_part = f", {fmt_seconds(llm_avg)} avg" if llm_avg is not None else ""
    lines = [
        "Timing & Cost",
        f"  LLM/VLM planning     {fmt_seconds(cost.get('llm_planning_time_s')):>10}  "
        f"({llm_calls} calls{avg_part})",
        f"  Exact actions        {fmt_steps(cost.get('exact_sim_steps')):>10} steps  "
        f"{fmt_seconds(cost.get('exact_sim_time_s'))} sim  @ {cost.get('sim_freq')} Hz",
    ]
    for name, steps, time_s in cost.get("action_rows") or []:
        lines.append(
            f"    {name:<18} {steps:>10} steps  {fmt_seconds(time_s)} sim"
        )
    meters = cost.get("explore_approx_meters")
    meters_part = f"  ({fmt_meters(meters)})" if meters is not None else ""
    lines.extend(
        [
            f"  Explore (calibrated) {fmt_steps(cost.get('explore_approx_sim_steps')):>10} steps  "
            f"{fmt_seconds(cost.get('explore_approx_sim_time_s'))} sim{meters_part}",
            f"  Combined (est.)      {fmt_steps(cost.get('combined_sim_steps')):>10} steps  "
            f"{fmt_seconds(cost.get('combined_sim_time_s'))} sim",
            f"  Env sim_step_count   {fmt_steps(cost.get('sim_step_count')):>10} steps",
            f"  Wall-clock runtime   {fmt_seconds(cost.get('runtime_s')):>10}  "
            f"(non-LLM {fmt_seconds(cost.get('non_llm_wall_s'))})",
        ]
    )
    task_pct = cost.get("task_percent_complete")
    task_ok = cost.get("task_state_success")
    if task_pct is not None or task_ok is not None:
        pct_str = "N/A" if task_pct is None else f"{task_pct:.3f}"
        ok_str = "N/A" if task_ok is None else f"{task_ok:.3f}"
        lines.append(
            f"  Task                 percent_complete={pct_str}  state_success={ok_str}"
        )
    return lines


DEFAULT_MAX_COMBINED_TIME_S = 600.0


def planner_cost_metrics(planner_info: Any) -> Dict[str, Any]:
    if not isinstance(planner_info, dict):
        return {}
    nested = planner_info.get("cost_metrics")
    if isinstance(nested, dict):
        return nested
    keys = (
        "llm_planning_time_s",
        "explore_approx_sim_time_s",
        "explore_approx_sim_steps",
        "explore_approx_meters",
        "explore_walk_ratio",
    )
    return {key: planner_info[key] for key in keys if key in planner_info}


def combined_time_breakdown(
    *,
    llm_planning_time_s: Any = None,
    action_sim_steps: Any = None,
    explore_approx_sim_time_s: Any = None,
    sim_freq: float = DEFAULT_SIM_FREQ,
    limit_s: Any = DEFAULT_MAX_COMBINED_TIME_S,
) -> Dict[str, Any]:
    """LLM wall time + exact action sim time + Explore approx sim time."""
    llm_s = _as_nonneg_float(llm_planning_time_s) or 0.0
    action_s = sim_time_from_steps(
        exact_action_sim_steps(action_sim_steps), sim_freq
    ) or 0.0
    explore_s = _as_nonneg_float(explore_approx_sim_time_s) or 0.0
    used_s = llm_s + action_s + explore_s
    limit = _as_nonneg_float(limit_s)
    if limit is None:
        limit = 0.0
    return {
        "llm_s": llm_s,
        "action_sim_s": action_s,
        "explore_approx_s": explore_s,
        "used_s": used_s,
        "limit_s": limit,
        "exceeded": limit > 0 and used_s >= limit,
    }
