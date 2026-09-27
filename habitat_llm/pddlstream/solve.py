import contextlib
import os
import traceback

from pddlstream.algorithms.focused import solve_focused
from pddlstream.utils import INF


def solve_pddlstream_problem(pddlstream_problem, max_time=60, planner="ff-astar1", verbose=False):
    """Solve a PDDLStream problem as pure STRIPS (no streams, no samplers).

    All geometric facts are pre-certified into init by build_pddlstream_problem,
    so the stream map is empty and solve_focused degenerates to a plain
    task-level search via Fast Downward — exactly the same as kitchen-worlds
    does for its oracle/symbolic planning layer.
    """
    try:
        # solve_abstract (solve_focused) prints iteration / plan dumps unconditionally;
        # keep the terminal quiet unless verbose=True.
        _run = lambda: solve_focused(
            pddlstream_problem,
            stream_info={},
            max_time=max_time,
            planner=planner,
            unit_costs=True,
            success_cost=INF,
            verbose=verbose,
        )
        if verbose:
            solution = _run()
        else:
            with open(os.devnull, "w") as _null:
                with contextlib.redirect_stdout(_null):
                    solution = _run()
    except Exception as e:
        print(f"\033[31m[PDDL SOLVER] solve_focused failed: {e}\033[0m")
        if verbose:
            traceback.print_exc()
        return None, 0, []

    if solution is None:
        return None, 0, []
    return solution
