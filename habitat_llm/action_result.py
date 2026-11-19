"""Simulator-free ReAct outcome classification using pre-action robot knowledge."""

import json
import re


_MISSING_OBJECT_SKIP = re.compile(
    r"(?:Unexpected failure!\s*-\s*)?Failed to pick!\s*Object does not exist\."
    r"(?:\s*(?:--|—|-)?\s*Should be skipped\.?)?",
    re.IGNORECASE,
)
REACT_ACTION_RE = re.compile(r"^[ \t]*([A-Z][A-Za-z]*)\[([^\]\n]*)\]", re.M)
SG_MARKER = "Robot SG before action: "
SG_POLICY = "robot_sg_before_action"
SG_POLICY_MARKER = "ReAct skip policy: " + SG_POLICY


def is_missing_object_skip(result):
    """Recognize the feedback only; this alone does NOT establish a skip."""
    return bool(_MISSING_OBJECT_SKIP.fullmatch(str(result or "").strip()))


def robot_scene_graph_marker(world_graph):
    """Snapshot remembered object names before dispatch, never simulator truth."""
    return SG_MARKER + json.dumps(sorted(obj.name for obj in world_graph.get_all_objects())) + "\n"


def mark_react_skip(step):
    """Exempt a failed pick only with verified prior robot-graph membership."""
    if is_missing_object_skip(step.get("result")):
        action = step.get("action", step.get("action_name"))
        skipped = (step.get("react_skip_policy") == "legacy" or
                   (action in {"Pick", "Rearrange"} and
                    step.get("target_in_robot_sg_before_action") is True))
        step.update(skipped=skipped, status="skipped" if skipped else "failure",
                    success=None if skipped else False)
        if skipped:
            step.pop("error", None)
    return step


def _trace_skip_policy(content):
    marked = re.search(r"^(?:" + re.escape(SG_POLICY_MARKER) + r"$|" + re.escape(SG_MARKER) + r")", content, re.M)
    return SG_POLICY if marked else "legacy"


def react_action_evidence(content, trace_path=None):
    """Read new-run snapshots; never reconstruct knowledge from old artifacts."""
    policy = _trace_skip_policy(content)
    records = []
    previous_end = 0
    for match in REACT_ACTION_RE.finditer(content):
        before = content[previous_end:match.start()]
        markers = list(re.finditer(r"^" + re.escape(SG_MARKER) + r"([^\n]*)", before, re.M))
        names, source = None, "unavailable"
        if markers:
            try:
                names = json.loads(markers[-1][1])
                if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                    names = None
            except (ValueError, TypeError):
                pass
            source = "pre_action_robot_graph" if names is not None else "unavailable"
        # Older reports can contain a composite Rearrange whose pick failed.
        target = match[2].split(",", 1)[0].strip() if match[1] == "Rearrange" else match[2].strip()
        records.append({"action": match[1], "args": match[2], "react_skip_policy": policy,
                        "target_in_robot_sg_before_action": target in names if names is not None else None,
                        "pre_action_robot_sg_source": source})
        previous_end = match.end()
    return records


def annotate_react_steps(steps, content, trace_path=None):
    evidence = react_action_evidence(content, trace_path)
    actions = [step for step in steps if step.get("action", step.get("action_name"))]
    aligned = len(actions) == len(evidence) and all(
        step.get("action", step.get("action_name")) == record["action"] and
        step.get("args", step.get("action_args", "")) == record["args"]
        for step, record in zip(actions, evidence)
    )
    for index, step in enumerate(actions):
        record = evidence[index] if aligned else {"target_in_robot_sg_before_action": None,
                                                 "pre_action_robot_sg_source": "unaligned_trace",
                                                 "react_skip_policy": _trace_skip_policy(content)}
        step.update({key: value for key, value in record.items() if key not in {"action", "args"}})
        mark_react_skip(step)
    return steps
