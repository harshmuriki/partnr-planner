"""Goal ledger + symbolic check for RailPlanner's end_of_plan_check=verify_symbolic.

When the active branch runs out of subgoals, the VLM writes a goal ledger: the
final-state conditions the task instruction requires, bound to entities in the
robot's memory. It sees the instruction and the memory text only, not the
planner's subgoals, so it is an independent reading of the task. The ledger is
then evaluated against the robot's own scene state (the one the subgoal
expanders plan from), and unmet conditions become feedback lines for a replan.

Deployable: nothing here reads the episode's evaluation propositions or the
simulator's goal check. See docs/goal_verification_findings.md (R1).
"""

import json
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from habitat_llm.rail_planner.subgoal_expanders import (
    SceneState,
    SubgoalPlanError,
    expand_subgoal,
)

# Ledger predicate -> number of entity arguments. Final-state conditions only.
PREDS: Dict[str, int] = {
    "on": 2,
    "in": 2,
    "next_to": 2,
    "cleaned": 1,
    "filled": 1,
    "powered_on": 1,
    "powered_off": 1,
    "closed": 1,
    "waited_in": 2,
}
_ALIASES = {
    "inside": "in",
    "within": "in",
    "on_top": "on",
    "nextto": "next_to",
    "clean": "cleaned",
    "is_clean": "cleaned",
    "is_filled": "filled",
    "is_powered_on": "powered_on",
    "is_powered_off": "powered_off",
    "close": "closed",
    "is_closed": "closed",
    "heated_in": "waited_in",
}
_STATE_WORDS = {
    "cleaned": "clean",
    "filled": "filled",
    "powered_on": "powered on",
    "powered_off": "powered off",
    "closed": "closed",
}

# Horizontal centre distance for next_to; the evaluator's default l2_threshold
# (habitat_llm/agent/env/evaluation/predicate_wrappers.py, is_next_to).
NEXT_TO_THRESHOLD_M = 0.5

LEDGER_PROMPT = """You check the work of a mobile household robot with one arm. The robot was given this task:
``{goal}''

Write down what must be true in the house when the task is complete. Do not plan steps, and do not assume the task is
already done: state the required end result even if the memory below already shows it.

Robot memory (world graph; facts may be outdated or incomplete):
``{observed}''

Use these conditions only, with exact names from the robot memory:
- on(<object>, <furniture>): the object rests on top of the furniture
- in(<object>, <furniture or container object>): the object is inside it
- next_to(<object>, <object or furniture>): the two are placed next to each other
- cleaned(<object or furniture>), filled(<object>), powered_on(<x>), powered_off(<x>), closed(<furniture>)
- waited_in(<object>, <appliance>): the object was left inside the appliance for a while (e.g. heated)

Rules:
- One condition per required fact. If the task applies to several objects, write one condition for each of them.
- Only final results: skip intermediate steps such as picking up, opening or searching.
- If the task's wording fits more than one piece of furniture (or object) equally well for the last argument of a
  condition, put your best choice in "args" and list the other candidates in "alternatives". Otherwise leave it empty.
- If the task needs an object that is not in the robot memory, do not invent a name: add it to "missing" with its type.

Return exactly one JSON object and nothing else:
{{"conditions": [{{"pred": "on", "args": ["<object>", "<furniture>"], "alternatives": []}}],
  "missing": [{{"type": "<object type>", "why": "<short reason>"}}]}}
"""


def build_ledger_prompt(instruction: str, scene_description: str) -> str:
    return LEDGER_PROMPT.format(goal=instruction, observed=scene_description)


def _norm(name) -> str:
    return str(name).strip().strip("\"'").replace(" ", "_")


def parse_ledger(text) -> Tuple[List[dict], List[dict], List[dict]]:
    """(conditions, missing, dropped). Conditions with an unknown predicate or
    the wrong number of arguments go to ``dropped``; unparsable text gives three
    empty lists."""
    match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    data = None
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            data = None
    if not isinstance(data, dict):
        return [], [], []
    conditions: List[dict] = []
    dropped: List[dict] = []
    for raw in data.get("conditions") or []:
        if not isinstance(raw, dict):
            dropped.append({"raw": raw, "why": "not an object"})
            continue
        pred = str(raw.get("pred", "")).strip().lower().replace("-", "_")
        pred = _ALIASES.get(pred, pred)
        args = raw.get("args")
        args = [_norm(a) for a in args] if isinstance(args, list) else []
        if pred not in PREDS:
            dropped.append({"raw": raw, "why": f"unknown predicate {pred!r}"})
            continue
        if len(args) != PREDS[pred] or not all(args):
            dropped.append({"raw": raw, "why": f"{pred} expects {PREDS[pred]} argument(s)"})
            continue
        alts = raw.get("alternatives")
        alts = [_norm(a) for a in alts if a] if isinstance(alts, list) else []
        cond = {"pred": pred, "args": args, "alternatives": [a for a in alts if a != args[-1]]}
        if cond not in conditions:
            conditions.append(cond)
    missing: List[dict] = []
    for raw in data.get("missing") or []:
        kind = raw.get("type") if isinstance(raw, dict) else raw
        if kind and str(kind).strip():
            why = str(raw.get("why", "")) if isinstance(raw, dict) else ""
            missing.append({"type": str(kind).strip(), "why": why})
    return conditions, missing, dropped


def _type_of(name: str) -> str:
    head, _, tail = name.rpartition("_")
    return head if head and tail.isdigit() else name


def _holds(state: SceneState, pred: str, args: List[str]) -> Optional[bool]:
    """True/False, or None when the robot's memory cannot tell."""
    if pred == "next_to":
        if state.holding in args:
            return False
        a, b = state.where(args[0]), state.where(args[1])
        if a is None or b is None:
            return None
        if len(a) >= 3 and len(b) >= 3:
            dist = ((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5  # y is up
        else:
            dist = sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5
        return dist <= NEXT_TO_THRESHOLD_M
    try:
        return expand_subgoal(state, pred, list(args)) == []
    except SubgoalPlanError:
        return False


def _unmet_line(pred: str, args: List[str], targets: List[str]) -> str:
    where = " or ".join(targets)
    if pred == "on":
        return f"{args[0]} is not placed correctly (expected on {where})."
    if pred == "in":
        return f"{args[0]} is not placed correctly (expected inside {where})."
    if pred == "next_to":
        return f"{args[0]} is not placed correctly next to {where}."
    if pred == "waited_in":
        return f"{args[0]} has not been left to wait inside {where} as required (waited_in)."
    return f"{where} is not {_STATE_WORDS[pred]} as required."


def _claim(pred: str, args: List[str], targets: List[str]) -> str:
    where = " or ".join(targets)
    if pred == "on":
        return f"{args[0]} is on the intended furniture ({where})"
    if pred == "in":
        return f"{args[0]} is inside the intended container ({where})"
    if pred == "next_to":
        return f"{args[0]} is next to {where}"
    if pred == "waited_in":
        return f"{args[0]} was left to wait inside the intended appliance ({where})"
    return f"the intended one of {where} is {_STATE_WORDS[pred]}"


def check_ledger(
    state: SceneState,
    conditions: Iterable[dict],
    missing: Iterable[dict],
    unexplored_rooms: Iterable[str],
    soft_seen: Set[Tuple],
) -> Tuple[List[str], List[str], List[dict]]:
    """Evaluate a ledger against the robot's scene state.

    Returns (unmet lines, soft lines, per-condition detail). Unmet lines are
    conditions the state contradicts or objects still to be found. Soft lines
    are conditions the robot cannot confirm: an ambiguous referent, or a
    next_to without positions. Soft lines, and "not found" lines once every
    room has been explored, are issued once per condition; ``soft_seen``
    remembers them across checks and is updated in place.
    """
    known = state.objects | state.furniture | state.rooms
    rooms = sorted(unexplored_rooms)
    search = f"; unexplored rooms: {', '.join(rooms)}" if rooms else ""
    unmet: List[str] = []
    soft: List[str] = []
    detail: List[dict] = []

    def add(lines: List[str], line: str) -> None:
        if line not in unmet and line not in soft:
            lines.append(line)

    def not_found(name: str, key: Tuple) -> bool:
        """Report a required entity that is not in memory; False if suppressed."""
        if not rooms:
            if key in soft_seen:
                return False
            soft_seen.add(key)
        add(unmet, f"Not found yet: a required {_type_of(name)}{search}.")
        return True

    for cond in conditions:
        pred = cond["pred"]
        args = [state.resolve(a) for a in cond["args"]]
        alts = [state.resolve(a) for a in cond.get("alternatives") or []]
        key = (pred, tuple(args))
        entry = {"pred": pred, "args": args, "alternatives": alts}
        detail.append(entry)

        unknown = [a for a in args if a not in known]
        if unknown:
            issued = not_found(unknown[0], ("not_found",) + key)
            entry["status"] = "not_found" if issued else "not_found_suppressed"
            continue

        targets = [args[-1]] + [a for a in alts if a in known and a != args[-1]]
        results = [_holds(state, pred, args[:-1] + [t]) for t in targets]
        if len(targets) > 1:
            # Ambiguous referent: the robot cannot tell which candidate was meant.
            if not any(results):
                entry["status"] = "unmet"
                add(unmet, _unmet_line(pred, args, targets))
            elif key in soft_seen:
                entry["status"] = "ambiguous_suppressed"
            else:
                soft_seen.add(key)
                entry["status"] = "ambiguous"
                add(soft, f"Could not confirm {_claim(pred, args, targets)}; check it.")
            continue

        if results[0] is True:
            entry["status"] = "met"
        elif results[0] is False:
            entry["status"] = "unmet"
            add(unmet, _unmet_line(pred, args, targets))
        elif key in soft_seen:
            entry["status"] = "unverifiable_suppressed"
        else:
            soft_seen.add(key)
            entry["status"] = "unverifiable"
            add(soft, f"Could not confirm {_claim(pred, args, targets)}; check it.")

    for item in missing:
        kind = str(item.get("type", "")).strip()
        if not kind:
            continue
        detail.append({"missing": kind, "status": "not_found" if rooms else "not_found_suppressed"})
        if rooms:
            # With every room explored there is nowhere left to look: drop it.
            add(unmet, f"Not found yet: a required {kind}{search}.")
    return unmet, soft, detail
