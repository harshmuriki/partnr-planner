"""Object-centric subgoals for RailPlanner (plan_config.subgoal_mode=object_centric).

Stage 1 (decompose): from the instruction and the robot's memory, the VLM lists abstract object tasks, one per
object or object group, each with the conditions it must end in, e.g. for "turn off the bedroom lights and fill the
jug and put it on the living-room table":
    all lights in the bedroom -> turned off
    jug                       -> filled; on the living-room table
No IDs are chosen here.
Stage 2 (ground + execute): the tasks are handled one at a time. The current task is turned into a goal string for
the usual RailPlanner Turn 1/Turn 2 prompts, which ground it into predicate subgoals with the memory at that moment;
those are executed, and the next task is grounded with the updated memory. When every task is done (or skipped),
one review call confirms the overall task is complete or adds tasks.
Uses only the robot's own memory, history and camera: no privileged information.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

SUBGOAL_MODES = ("flat", "object_centric")
TERMINAL = ("complete", "unsolvable")

_TASK_FORMAT = (
    '{"tasks": [{"objects": "<object or object group, in words>", '
    '"requirements": ["<condition it must end in>", ...]}, ...]}'
)

DECOMPOSE_PROMPT = """You control a mobile household robot with one arm. Break this task into object tasks:
``{goal}''

An object task is everything the task requires of ONE object or ONE group of same-kind objects (e.g. "all lights in
the bedroom", "the jug", "both packages"). List each object or group once, with every condition it must end in
(e.g. "turned off", "filled", "clean", "on the living-room table", "inside the fridge", "next to the scissors").
- Describe objects and places in words, as the task does. Do NOT pick object or furniture IDs yet; that happens later
  with the robot's memory, one object task at a time.
- Include objects the task needs that are not in memory yet; they will be searched for later.
- Keep conditions the task states and the ones it clearly implies (e.g. "clean" before "fill" for dishes to drink
  from). Do not add conditions the task does not ask for.
- Order the object tasks in a sensible execution order (e.g. tools or soap before the objects they are used on,
  objects that must be next to each other after their reference object).

Robot memory (world graph; facts may be outdated or incomplete):
``{observed}''
{history}
Return exactly one JSON object and nothing else:
{task_format}
If the task is already complete or clearly impossible, return {{"decision":"complete","reason":"..."}} or
{{"decision":"unsolvable","reason":"..."}} instead.
"""

REVIEW_PROMPT = """You control a mobile household robot with one arm. The overall task is:
``{goal}''

It was broken into object tasks, which the robot has now worked through:
{tasks}
Robot memory now (world graph; facts may be outdated or incomplete):
``{observed}''
{history}
Check every requirement of the overall task against the robot's memory and the attached image, if any. Decide:
- {{"decision":"complete","reason":"evidence that every requested condition holds"}}, or
- {{"decision":"unsolvable","reason":"what remains unknown or impossible and what was tried"}}, or
- more object tasks for whatever is still missing or wrong, in the same format as before:
  {task_format}
  (only the objects that still need work; say what to fix, e.g. "on the living-room table, next to the box").
Return exactly one JSON object and nothing else.
"""

GOAL_TEMPLATE = """{instruction}

[OBJECT TASK {k} of {n}] Plan ONLY this object task now: {task}
Done object tasks: {done}
Skipped object tasks: {failed}
Object tasks still to do after this one (do not plan them yet): {todo}
Choose the concrete object and furniture IDs for this object task from the robot's memory. If this object task
already holds, return {{"decision":"complete","reason":"evidence"}}; if it cannot be done, return
{{"decision":"unsolvable","reason":"why"}}. These decisions apply to this object task only."""


@dataclass
class ObjectTask:
    idx: int
    objects: str
    requirements: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | active | done | failed
    regrounds: int = 0
    note: str = ""

    def describe(self) -> str:
        req = "; ".join(self.requirements) if self.requirements else "as the task requires"
        return f"{self.objects} -> {req}"

    def as_dict(self) -> dict:
        return {"idx": self.idx, "objects": self.objects, "requirements": list(self.requirements),
                "status": self.status, "regrounds": self.regrounds, "note": self.note}


def _json_object(text) -> Optional[dict]:
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def parse_task_list(text, start_idx: int = 0) -> Tuple[List[ObjectTask], Optional[str], str, bool]:
    """(tasks, decision, reason, parsed). decision is "complete"/"unsolvable" or None.

    Unparsable output gives ([], None, "", False); the caller falls back to one task covering the whole instruction.
    """
    data = _json_object(text)
    if data is None:
        return [], None, "", False
    decision = str(data.get("decision", "") or "").strip().lower()
    if decision in TERMINAL:
        return [], decision, str(data.get("reason", "")), True
    items = data.get("tasks")
    if not isinstance(items, list):
        return [], None, "", False
    tasks = []
    for item in items:
        if isinstance(item, str):
            objects, reqs = item.strip(), []
        elif isinstance(item, dict):
            objects = str(item.get("objects") or item.get("object") or "").strip()
            reqs = item.get("requirements") or item.get("conditions") or []
            if isinstance(reqs, str):
                reqs = [reqs]
            reqs = [str(r).strip() for r in reqs if str(r).strip()]
        else:
            continue
        if objects:
            tasks.append(ObjectTask(idx=start_idx + len(tasks), objects=objects, requirements=reqs))
    return tasks, None, "", bool(tasks)


def fallback_task(instruction: str, idx: int = 0) -> ObjectTask:
    return ObjectTask(idx=idx, objects="all objects in the task", requirements=[instruction.strip()],
                      note="fallback: decomposition was not parsable")


def _history_block(history: List[str], limit: int = 15) -> str:
    recent = [str(h) for h in (history or [])[-limit:]]
    return ("\nRecent actions and failures (oldest first):\n" + "".join(f"- {h}\n" for h in recent)) if recent else ""


def build_decompose_prompt(goal: str, scene_description: str, history: List[str]) -> str:
    return DECOMPOSE_PROMPT.format(goal=goal, observed=scene_description, history=_history_block(history),
                                   task_format=_TASK_FORMAT)


def format_task_list(tasks: List[ObjectTask]) -> str:
    return "".join(f"{t.idx + 1}. {t.describe()} [{t.status}{': ' + t.note if t.note else ''}]\n" for t in tasks)


def build_review_prompt(goal: str, tasks: List[ObjectTask], scene_description: str, history: List[str],
                        extra: str = "") -> str:
    hist = _history_block(history) + (f"\n{extra.strip()}\n" if extra and extra.strip() else "")
    return REVIEW_PROMPT.format(goal=goal, tasks=format_task_list(tasks), observed=scene_description,
                                history=hist, task_format=_TASK_FORMAT)


def task_goal_text(instruction: str, tasks: List[ObjectTask], cur: ObjectTask) -> str:
    """The goal string the Turn-1 prompt gets while grounding ``cur``."""
    def names(status):
        return "; ".join(t.describe() for t in tasks if t.status == status and t is not cur) or "none"

    todo = "; ".join(t.describe() for t in tasks if t.status == "pending" and t.idx > cur.idx) or "none"
    return GOAL_TEMPLATE.format(instruction=instruction.strip(), k=cur.idx + 1, n=len(tasks), task=cur.describe(),
                                done=names("done"), failed=names("failed"), todo=todo)
