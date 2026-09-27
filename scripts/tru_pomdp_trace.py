"""TRU-POMDP parsing and data adapter for the shared baseline trace viewer."""

import csv
import json
import re
from pathlib import Path


def estimate_visible_text_usage(text, model, expected_calls, system_message=""):
    """Estimate only logged text; never invent hidden reasoning or cache usage."""
    from habitat_llm.utils.llm_usage import (
        TokenUsageTracker,
        estimate_tokens_from_chars,
    )

    lengths = re.findall(r"^length:\s*(\d+)\s*$", text, re.MULTILINE)
    responses = re.findall(
        r"^\[toh\] raw response (\d+):\n(.*?)(?=^\x1b|^length:|^\[?\d{4}-\d{2}-\d{2} |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if (
        not expected_calls
        or len(lengths) != expected_calls
        or len(responses) != expected_calls
    ):
        return {}
    if [int(number) for number, _ in responses] != list(range(1, expected_calls + 1)):
        return {}
    tracker = TokenUsageTracker(model)
    for length, (_, response) in zip(lengths, responses):
        tracker.record(
            prompt_tokens=estimate_tokens_from_chars(int(length) + len(system_message)),
            completion_tokens=estimate_tokens_from_chars(len(response.strip())),
            source="estimated",
        )
    estimate = tracker.snapshot()
    estimate["llm_usage_note"] = (
        "Visible-text estimate from logged prompt lengths and responses (characters / 4). "
        "Excludes hidden reasoning tokens and message framing; assumes uncached input. "
        "This is not the measured billed cost."
    )
    return estimate


def fill_saved_llm_metadata(source, dataset, episode, metrics):
    """Prefer recorded API usage; use only the matching run's archived config/log."""
    import yaml

    config_path = dataset.parent / "hydra/.hydra/config.yaml"
    llm_config = {}
    if config_path.is_file():
        try:
            config = yaml.safe_load(config_path.read_text())
            agent = config["evaluation"]["agents"][f"agent_{source.parent.name}"]
            llm_config = agent["planner"]["plan_config"]["llm"]
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            pass
    params = llm_config.get("generation_params", {})
    for field, key in (
        ("llm_model", "model"),
        ("llm_reasoning_effort", "reasoning_effort"),
    ):
        if not metrics.get(field) and params.get(key):
            metrics[field] = params[key]
    if metrics.get("llm_usd") is not None or metrics.get("llm_usd_source") == "api":
        return
    # An incomplete or multi-episode console log cannot establish this run's cost.
    if not metrics.get("llm_model") or llm_config.get("keep_message_history", True):
        return
    identifier = episode.removeprefix("episode_").rpartition("_")[0]
    for filename in ("run.log", "log.txt"):
        path = dataset.parent / filename
        if not path.is_file():
            continue
        text = path.read_text()
        if re.findall(r"^Episode (\S+)\s*$", text, re.MULTILINE) != [identifier]:
            continue
        estimate = estimate_visible_text_usage(
            text,
            metrics["llm_model"],
            metrics.get("llm_call_count"),
            llm_config.get("system_message", ""),
        )
        if estimate:
            metrics.update(estimate)
            return


def result_status(result):
    """Classify recorded skill feedback; missing feedback is not failure."""
    text = (result or "").strip().lower()
    if any(
        x in text
        for x in ("failed", "failure", "error", "unsuccessful", "not successful")
    ):
        return "failure"
    if "successful execution" in text or text == "success":
        return "success"
    return "unknown"


def parse_trace(content, source_path=None):
    data = {
        "planner_type": "tru_pomdp",
        "task": "",
        "steps": [],
        "decisions": [],
        "warnings": [],
        "source_path": str(source_path) if source_path else None,
    }
    current = None
    snapshot = None
    memory = None
    in_result = False
    for raw in content.splitlines():
        line = raw.strip()
        decision = re.match(r"^Decision (\d+):\s*(.*)$", line)
        action = re.match(r"^Action:\s*([A-Za-z_]+)\[(.*)\]$", line)
        hypotheses = re.match(r"^Hypotheses \((initial|updated)\):\s*(.*)$", line)
        recognized = line.startswith(
            ("Task:", "Result:", "Evidence:", "Memory:", "Belief:")
        )
        if decision or action or hypotheses or recognized:
            in_result = False
        if line.startswith("Task:"):
            data["task"] = line[5:].strip()
        elif decision:
            current = {
                "number": int(decision[1]),
                "action": decision[2],
                "steps": [],
                "before": snapshot,
                "after": None,
                "memory_before": memory,
                "memory_after": None,
                "evidence": None,
                "belief": "",
            }
            data["decisions"].append(current)
        elif action:
            step = {"action": action[1], "args": action[2], "result": ""}
            data["steps"].append(step)
            if current is not None:
                current["steps"].append(step)
        elif line.startswith("Result:") and data["steps"]:
            data["steps"][-1]["result"] = line[7:].strip()
            in_result = True
        elif hypotheses or line.startswith("Memory:"):
            try:
                value = json.loads(hypotheses[2] if hypotheses else line[7:].strip())
                if hypotheses:
                    if not isinstance(value, list):
                        raise ValueError("Expected a hypothesis list")
                    snapshot = value
                    if hypotheses[1] == "initial":
                        data["initial_hypotheses"] = value
                    elif current is not None:
                        current["after"] = value
                else:
                    if not isinstance(value, dict):
                        raise ValueError("Expected a memory map")
                    memory = value
                    if current is not None:
                        current["memory_after"] = value
            except (ValueError, TypeError):
                data["warnings"].append(
                    "Could not decode a recorded belief or memory snapshot."
                )
        elif line.startswith("Evidence:") and current is not None:
            current["evidence_text"] = line[9:].strip()
            if "; observation=" in line:
                try:
                    current["evidence"] = json.loads(line.split("; observation=", 1)[1])
                except ValueError:
                    data["warnings"].append("Could not decode a recorded observation.")
        elif line.startswith("Belief:") and current is not None:
            current["belief"] = line[7:].strip()
        elif in_result and line and data["steps"]:
            data["steps"][-1]["result"] += "\n" + line
    for step in data["steps"]:
        step["status"] = result_status(step["result"])
        step["success"] = {"success": True, "failure": False, "unknown": None}[
            step["status"]
        ]
    data["total_steps"] = len(data["steps"])
    return data


def load_context(source):
    """Read only this episode's saved diagnostics; never infer success from actions."""
    context = {"metrics": {}, "diagnostics": {}, "decisions": {}, "warnings": []}
    source = Path(source)
    if not source.stem.startswith("trace-"):
        return context
    episode = source.stem[6:].rsplit("-", 1)[0]
    dataset = source.parent.parent.parent
    planner_log = dataset / "planner-log" / ("planner-log-" + episode + ".json")
    if planner_log.is_file():
        try:
            log = json.loads(planner_log.read_text())
            context["task"] = log.get("task", "")
            action_steps = {}
            previous_sim = None
            previous_names = []
            for step in log.get("steps", []):
                sim = step.get("sim_step_count")
                names = [
                    str(action[0])
                    for action in (step.get("high_level_actions") or {}).values()
                    if action
                ]
                if sim is not None and previous_sim is not None:
                    delta = max(0, int(sim) - int(previous_sim))
                    for name in names or previous_names or ["Unknown"]:
                        action_steps[name] = action_steps.get(name, 0) + delta
                previous_sim = sim
                previous_names = names
                diag = step.get("tru_pomdp", {})
                number = diag.get("num_decisions")
                if number:
                    entry = context["decisions"].setdefault(
                        number, {"search": dict(diag)}
                    )
                    entry["after"] = dict(diag)
                context["diagnostics"] = diag
                context["metrics"].update(step.get("cost_metrics", {}))
                context["metrics"].update(step.get("stats", {}))
                context["metrics"]["sim_step_count"] = step.get("sim_step_count")
            context["metrics"]["action_sim_steps"] = action_steps
            context["metrics"]["llm_requests"] = context["metrics"].get(
                "llm_call_count"
            )
        except (OSError, ValueError, TypeError):
            context["warnings"].append("Planner diagnostics could not be loaded.")
    csv_path = dataset.parent / "episode_result_log.csv"
    if csv_path.is_file():
        # Match both episode and run, including episode IDs that contain underscores.
        identifier, _, run_id = episode.removeprefix("episode_").rpartition("_")
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("episode_id") == identifier and row.get("run_id") == run_id:
                    context["task"] = row.get("instruction") or context.get("task", "")
                    for key, value in row.items():
                        try:
                            context["metrics"][key] = float(value)
                        except (TypeError, ValueError):
                            pass
    fill_saved_llm_metadata(source, dataset, episode, context["metrics"])
    return context


def prepare_trace(trace, source):
    """Attach TRU-POMDP content to ordinary shared-viewer action cards."""
    context = load_context(source)
    prepared = dict(trace)
    prepared["task"] = trace.get("task") or context.get("task") or "Task not recorded"
    prepared["steps"] = [dict(step) for step in trace["steps"]]
    copies = {
        id(original): copy for original, copy in zip(trace["steps"], prepared["steps"])
    }
    for decision in trace["decisions"]:
        number = decision["number"]
        for step in decision["steps"]:
            copies[id(step)]["thought"] = f"Decision {number}: {decision['action']}"
            if not step.get("result"):
                copies[id(step)]["result"] = "No result recorded"
        if not decision["steps"]:
            continue
        diagnostics = context["decisions"].get(number, {})
        sections = [
            ("Belief update", decision["belief"] or "No update recorded"),
            ("Hypotheses before this decision", decision["before"]),
            ("Surviving hypotheses", decision["after"]),
            ("Observed evidence", decision["evidence"]),
            (
                "Revisable memory",
                {
                    "before": decision["memory_before"],
                    "after": decision["memory_after"],
                },
            ),
            ("Search and belief diagnostics", diagnostics or None),
        ]
        copies[id(decision["steps"][-1])]["trace_sections"] = [
            (label, value if isinstance(value, str) else json.dumps(value, indent=2))
            for label, value in sections
            if value is not None
        ]
    prepared["tru_context"] = context
    return prepared
