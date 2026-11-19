import argparse
import ast
import collections
import csv
import gzip
import json
import re
import sqlite3
from abc import ABC, abstractmethod
from functools import cache, cached_property
from pathlib import Path

GROUND_TRUTH = Path("/coc/flash5/vmuriki3/tamp/partnr-planner/baseline_evaluation_v3")

@cache
def ground_truth_search():
    """(object -> likely spec rooms, (object, scene) -> likely spec furniture, scene -> spec furniture -> sim handle),
    from the ground-truth recorder's database. Its snapshots alias spec furniture names to their own names."""
    db = sqlite3.connect(f"file:{GROUND_TRUTH / 'ground_truth.sqlite3'}?immutable=1", uri=True)
    rooms = {name: json.loads(found) for name, found in db.execute("select object, rooms from object_rooms")}
    furniture = {(name, scene): json.loads(found) for name, scene, found in db.execute("select object, scene, furniture from object_furniture")}
    handles = {}
    for (snapshot,) in db.execute("select snapshot from runs"):
        snapshot = json.loads(snapshot)
        own = {piece["name"]: piece["sim_handle"] for piece in snapshot.get("entities", {}).get("furniture", [])}
        scene = handles.setdefault(snapshot["episode"]["scene_id"], {})
        scene.update((spec, own[name]) for spec, name in (snapshot.get("aliases") or {}).items() if name in own)
    return rooms, furniture, handles

@cache
def furniture_catalog(task):
    """Spec room -> spec furniture in it, from specs/<task>/_furniture_catalog.txt."""
    rooms, current = {}, None
    for line in (GROUND_TRUTH / "specs" / task / "_furniture_catalog.txt").open():
        if match := re.match(r"^(\S+):\s*$", line):
            current = match.group(1)
        elif (match := re.match(r"^\s+- (\S+):", line)) and current:
            rooms.setdefault(current, []).append(match.group(1))
    return rooms

class Action:
    def __init__(self, entry):
        self.id = entry["action_id"]
        self.name = entry["action"]
        self.input = entry["action_input"] or ""
        self.response = entry["response"] or ""
        self.start = entry["started_sim_step"]
        self.end = entry.get("finished_sim_step")

    @property
    def target(self):
        return self.input.split(",")[0].strip()

    def reruns(self, previous):
        """VLM-TAMP executes an instantly failing action twice; the second copy starts on the next sim step."""
        return (
            previous is not None
            and (self.name, self.input) == (previous.name, previous.input)
            and previous.start == previous.end
            and self.start == previous.end + 1
        )

    def __str__(self):
        return f"a{self.id} {self.name}[{self.input}]: {self.response}"

class Run(dict):
    """A row of baseline_results.json, plus what the rules read from its run folder."""

    @property
    def folder(self):
        return Path(self["run_folder"])

    @property
    def planner(self):
        return self["id"].split("/")[0]

    @property
    def vlm_tamp_family(self):
        """VLM-TAMP and RailPlanner (a VlmTampPddlPlanner subclass) share the VLM loop, logs and trace format."""
        return self.planner == "vlm-tamp" or self.planner.startswith("rail")

    @property
    def vlm_log(self):
        log_dir = "vlm_tamp_pddl" if self.planner == "vlm-tamp" else "rail_planner"
        return self.folder / log_dir / "vlm_tamp_pddl_log.jsonl"

    @cached_property
    def actions(self):
        path = next((self.folder / "dataset" / "actions").glob("*.json"))
        actions = [Action(entry) for entry in json.loads(path.read_text())["actions"]]
        if not self.vlm_tamp_family:
            return actions
        return [action for previous, action in zip([None, *actions], actions) if not action.reruns(previous)]

    @cached_property
    def memory(self):
        """Object names in the robot's starting memory, from log.txt.

        The listed names and the Robot Scene Graph can name the same stale object differently
        (T5-OUT-ABS: plate_0 listed, plate_1 in the graph), so both are read.
        """
        text = (self.folder / "log.txt").read_text(errors="ignore")
        names = set()
        if listed := re.search(r"^Initial robot memory objects: (.*)$", text, re.MULTILINE):
            names |= {name.strip() for name in listed.group(1).split(",")}
        if "Robot Scene Graph:" in text:
            graph = text.split("Robot Scene Graph:", 1)[1].split("\n\n", 1)[0]
            names |= set(re.findall(r"^\t+Object: (\S+)", graph, re.MULTILINE))
        return names

    @cached_property
    def seen(self):
        """(object, furniture) pairs the robot's sensors registered: its starting Robot Scene Graph plus
        every world graph the planner logged. The world graph is updated by the environment from the
        camera view, not by the LLM, so an object counts as seen even if the planner ignored it."""
        pairs = set()
        text = (self.folder / "log.txt").read_text(errors="ignore")
        if "Robot Scene Graph:" in text:
            parent = ""
            for line in text.split("Robot Scene Graph:", 1)[1].split("\n\n", 1)[0].splitlines():
                if match := re.match(r"\t+Receptacle: rec_(\S+)_\d+$", line):
                    parent = match.group(1)
                elif match := re.match(r"\t+(?:Furniture|Floor): (\S+)", line):
                    parent = match.group(1)
                elif match := re.match(r"\t+Object: (\S+)", line):
                    pairs.add((match.group(1), parent))
        graphs = []
        if self.vlm_tamp_family:
            log = self.vlm_log.read_text(errors="ignore")
            graphs = [line for line in log.splitlines() if line.startswith("{'event': 'memory_snapshot'")]
        else:
            for step in self.planner_log:
                graphs += list((step.get("curr_graph") or {}).values())
        for graph in graphs:
            pairs |= set(re.findall(r"(\w+): (?:on|inside|in|within) (\w+)", graph))
        return pairs

    @cached_property
    def planner_log(self):
        """The planner's logged steps (world graph, TRU-POMDP termination reason, ...)."""
        path = next((self.folder / "dataset" / "planner-log").glob("planner-log-*.json"))
        return json.loads(path.read_text())["steps"]

    @cached_property
    def trace(self):
        path = next((self.folder / "dataset" / "traces" / "0").glob("trace-*.txt"))
        return path.read_text(errors="ignore")

    @cached_property
    def targets(self):
        """Objects that must be moved (is_on_top / is_inside) and have such a requirement never met."""
        return {
            self.names.get(handle, handle)
            for index, requirement in enumerate(self.requirements)
            if requirement["function_name"] in ("is_on_top", "is_inside") and self.satisfied_at[index] == -1
            for handle in requirement["args"]["object_handles"]
        }

    @cached_property
    def picked(self):
        """Names of objects the robot successfully picked up at least once."""
        return {action.target for action in self.actions if action.name == "Pick" and action.response.startswith("Successful")}

    @cached_property
    def absent(self):
        """Objects the variant spec lists as absent from the scene (ABS variants only)."""
        spec = (self.folder / "input" / "spec.md").read_text()
        return re.findall(r"^- (\S+) \(.*absent from the scene", spec, re.MULTILINE)

    @cached_property
    def search_plan(self):
        """The ground truth's likely rooms and furniture for the absent objects, in this run's names.
        A spec room becomes the run's room that most of its catalogued furniture is in."""
        likely_rooms, likely_furniture, handles = ground_truth_search()
        scene = json.loads(gzip.open(self.folder / "input" / "dataset.json.gz").read())["episodes"][0]["scene_id"]
        spec_handles = handles.get(scene, {})
        catalog = furniture_catalog(self["id"].split("/")[1].split("-")[0])
        named = {
            state["final"]["sim_handle"]: (name, state["final"]["location"]["room"])
            for name, state in self.object_states.items()
            if state["final"].get("kind") == "furniture"
        }
        rooms, furniture = set(), set()
        for name in self.absent:
            for spec_room in likely_rooms.get(name, []):
                votes = collections.Counter(
                    named[spec_handles[piece]][1] for piece in catalog.get(spec_room, []) if spec_handles.get(piece) in named
                )
                if votes:
                    rooms.add(votes.most_common(1)[0][0])
            for spec_piece in likely_furniture.get((name, scene), []):
                if spec_handles.get(spec_piece) in named:
                    furniture.add(named[spec_handles[spec_piece]][0])
        return rooms, furniture

    @cached_property
    def distractors(self):
        """Objects the variant spec's entity registry labels as distractors (DIS variants only)."""
        spec = (self.folder / "input" / "spec.md").read_text()
        return re.findall(r"^- (\S+): [^,\n]+, distractor$", spec, re.MULTILINE)

    @cached_property
    def object_states(self):
        """Simulator initial and final state of each object and furniture piece, by name."""
        path = next((self.folder / "dataset" / "object_states").glob("*.json"))
        return json.loads(path.read_text())["objects"]

    @cached_property
    def evaluator(self):
        path = self.folder / "dataset" / "evaluation-checkpoint.json"
        return json.loads(path.read_text())["evaluator_only_state"]

    @property
    def requirements(self):
        """The evaluator's propositions, indexed as in its own lists (baseline_results' criteria can merge them)."""
        return self.evaluator["auto_eval_proposition_tracker"]["propositions"]

    @property
    def satisfied_at(self):
        """Step at which each requirement was first met, -1 if never."""
        return self.evaluator["auto_eval_proposition_tracker"]["proposition_satisfied_at"]

    @property
    def constraints(self):
        """Each constraint as (its text, whether it left each requirement valid)."""
        texts = self.evaluator["auto_eval_proposition_tracker"]["constraints"]
        return list(zip(texts, self.evaluator["task_constraint_validation"]))

    def cancelled_by(self, kind, index):
        return any(text.startswith(kind) and not valid[index] for text, valid in self.constraints)

    @cached_property
    def prerequisites(self):
        """(before, after) requirement pairs: temporal constraint edges, plus "after_satisfied" dependencies (T6)."""
        tracker = self.evaluator["auto_eval_proposition_tracker"]
        pairs = [
            edge
            for text, _ in self.constraints
            if text.startswith("TemporalConstraint")
            for edge in ast.literal_eval(re.search(r"dag_edges=(\[.*?\])", text).group(1))
        ]
        pairs += [
            (before, after)
            for dependency in tracker["dependencies"]
            if dependency["relation_type"] == "after_satisfied"
            for after in dependency["proposition_indices"]
            for before in dependency["depends_on"]
        ]
        return pairs

    @cached_property
    def names(self):
        """Simulator handle -> name, for everything in the scene."""
        return {node["sim_handle"]: node["name"] for node in self.evaluator["reference_world_graph"]}

    def describe(self, index):
        """Requirement `index` in readable form, e.g. "is_on_top(box_0, table_16)"."""
        requirement = self.requirements[index]
        args = [
            "|".join(self.names.get(handle, handle) for handle in value)
            for value in requirement["args"].values()
            if isinstance(value, list)
        ]
        return f"{requirement['function_name']}({', '.join(args)})"

    @cached_property
    def repeated_blocks(self):
        """(actions, sequence size, repeats) for each back-to-back repeat; the shortest repeating sequence wins."""
        keys = [(action.name, action.input) for action in self.actions]
        blocks = []
        i = 0
        while i < len(keys):
            for size in range(1, 6):
                sequence = keys[i : i + size]
                count = 1
                while keys[i + count * size : i + (count + 1) * size] == sequence:
                    count += 1
                if len(sequence) < size or count < 2:
                    continue
                blocks.append((self.actions[i : i + size * count], size, count))
                i += size * count
                break
            else:
                i += 1
        return blocks

    @cached_property
    def repeated(self):
        """Ids of actions inside a repeated block."""
        return {action.id for block, _, _ in self.repeated_blocks for action in block}

    @cached_property
    def boundaries(self):
        """(sim step, whether each requirement held) at the start of every planner step, from the evaluator's evidence log."""
        path = next((self.folder / "dataset").glob("evidence-*.jsonl"))
        states = []
        for line in path.open():
            event = json.loads(line)
            if event["event"] not in ("boundary", "terminal") or event.get("sim_step_count") is None:
                continue
            tracker = event["evaluator_only_state"]["auto_eval_proposition_tracker"]
            states.append((event["sim_step_count"], [state["is_satisfied"] for state in tracker["last_state"]]))
        return states

    def state_before(self, action):
        """Whether each requirement held at the last boundary before the action started, or None."""
        earlier = [held for step, held in self.boundaries if step <= action.start]
        return earlier[-1] if earlier else None

    def objects(self, index):
        """Names of the objects requirement `index` is about (not its receptacles)."""
        args = self.requirements[index]["args"]
        handles = args.get("object_handles", []) + args.get("entity_handles_a", []) + args.get("entity_handles_b", [])
        return {self.names.get(handle, handle) for handle in handles}

class ObservedError(ABC):
    """Perceptible errors from episodes."""

    label: str

    @property
    def category(self):
        return self.label

    @abstractmethod
    def find(self, run) -> list[str]:
        """Evidence for each occurrence in the run."""

class Failure(ObservedError):
    """Why a run was unsuccessful in completing the given task."""

    kind = "failure"

class Inefficiency(ObservedError):
    """Avoidable or useless actions. Using a stale object from an outdated memory variant does not count."""

    kind = "inefficiency"
    messages: tuple[str, ...]

    def find(self, run):
        return [str(action) for action in run.actions if self.matches(run, action)]

    def matches(self, run, action):
        return any(re.search(message, action.response) for message in self.messages)

class TimeLimitReached(Failure):
    """The planner reached the allotted time to complete a task."""

    label = "Time Limit Reached"
    excludes = {"Object Not Found", "Object Untouched", "Missing Action"}

    def find(self, run):
        if run["execution_status"] != "timeout":
            return []
        return [f"Stopped after {run['runtime_s']:.0f} s"]

class ActionLimitReached(Failure):
    """The planner reached a pre-defined budget; dependent on framework type (for e.g., LLM calls for ReAct and higher-level planning decisions for Tru-POMDP)."""

    label = "Action Limit Reached"

    def find(self, run):
        termination = run["termination"] or {}
        if termination.get("reason") != "max_actions":
            return []
        return [f"{termination['message']} ({termination['limit_source']})"]

class ActionUndone(Failure):
    """A necessary step for successful task completion was reversed."""

    label = "Action Undone"

    def find(self, run):
        evidence = []
        for index in range(len(run.requirements)):
            if not run.cancelled_by("TerminalSatisfactionConstraint", index):
                continue
            evidence.append(f"{run.describe(index)}: met at step {run.satisfied_at[index]}, not true at the end")
        return evidence

class WrongOrder(Failure):
    """Required steps were completed out of order."""

    label = "Wrong Order"

    def find(self, run):
        evidence = []
        for before, after in run.prerequisites:
            if run.satisfied_at[after] == -1 or not run.cancelled_by("TemporalConstraint", after):
                continue
            if run.satisfied_at[before] <= run.satisfied_at[after]:
                continue
            evidence.append(
                f"{run.describe(after)} met at step {run.satisfied_at[after]}, "
                f"before {run.describe(before)} at step {run.satisfied_at[before]}"
            )
        return evidence

class MissingAction(Failure):
    """A required step to perform on an object was not completed. An attempted action that failed to execute does not count."""

    label = "Missing Action"
    state_steps = {"is_clean", "is_filled", "is_powered_off", "is_waited_inside"}

    def find(self, run):
        evidence = []
        for index, requirement in enumerate(run.requirements):
            if requirement["function_name"] not in self.state_steps or run.satisfied_at[index] != -1:
                continue
            evidence.append(f"{run.describe(index)}: never done")
        return evidence

class MissingPrerequisite(Failure):
    """A required step was completed, but one that was supposed to come before it was not."""

    label = "Missing Prerequisite"

    def find(self, run):
        evidence = []
        for before, after in run.prerequisites:
            if run.satisfied_at[after] == -1 or run.satisfied_at[before] != -1:
                continue
            evidence.append(f"{run.describe(before)}: never done, needed before {run.describe(after)}")
        return evidence

class IncorrectPlacement(Failure):
    """A target object was placed in the wrong location."""

    label = "Incorrect Placement"
    placements = {"is_on_top", "is_inside"}

    def find(self, run):
        targets = {}
        for index, requirement in enumerate(run.requirements):
            if requirement["function_name"] not in self.placements:
                continue
            for handle in requirement["args"]["object_handles"]:
                targets.setdefault(run.names.get(handle, handle), []).append(index)
        evidence = []
        for name, indices in targets.items():
            unmet = [index for index in indices if run.satisfied_at[index] == -1 and not self.waiting(run, index)]
            if not unmet or name not in run.picked:
                continue
            final = run.object_states[name]["final"]["location"]
            if final["held_by"]:
                continue
            goals = ", ".join(run.describe(index) for index in unmet)
            evidence.append(f"{name} ended on {final['furniture']} in {final['room']}; unmet: {goals}")
        return evidence

    def waiting(self, run, index):
        """Whether the requirement is held back by a prerequisite that was never met (T6 delivery before heating)."""
        return any(after == index and run.satisfied_at[before] == -1 for before, after in run.prerequisites)

class ObjectNotFound(Failure):
    """A target object was not found."""

    label = "Object Not Found"

    def find(self, run):
        evidence = []
        for name in sorted(run.targets):
            real = run.object_states[name]["initial"]["location"]
            if name in run.picked or (name, real["furniture"]) in run.seen:
                continue
            explored = [f"a{action.id}" for action in run.actions if action.name == "Explore" and action.target == real["room"]]
            note = f" (explored {real['room']} at {', '.join(explored)})" if explored else ""
            evidence.append(f"{name}: never seen on/in {real['furniture']} in {real['room']}{note}")
        return evidence

class ObjectUntouched(Failure):
    """The robot had accurate knowledge of or encountered a target object during the episode, but did not move it at all."""

    label = "Object Untouched"

    def find(self, run):
        evidence = []
        for name in sorted(run.targets):
            real = run.object_states[name]["initial"]["location"]
            if name in run.picked or (name, real["furniture"]) not in run.seen:
                continue
            evidence.append(f"{name}: seen on/in {real['furniture']} in {real['room']}, never moved")
        return evidence

class TaskAbandoned(Failure):
    """The planner prematurely gave up on an overall task without attempting one or more subtasks."""

    label = "Task Abandoned"
    give_up_words = re.compile(
        r"cannot|can't|can not|unable|impossible|not possible|unavailable|absent|no suitable|no acceptable|could not",
        re.IGNORECASE,
    )

    def find(self, run):
        if run["execution_status"] == "timeout" or (run["termination"] or {}).get("reason") == "max_actions":
            return []
        reason = self.gave_up(run)
        if reason is None:
            return []
        acted_on = {action.target for action in run.actions if action.name not in ("Navigate", "Explore")}
        unattempted = [
            run.describe(index)
            for index in range(len(run.requirements))
            if run.satisfied_at[index] == -1 and not run.objects(index) & acted_on
        ]
        if not unattempted:
            return []
        return [f"{reason}; never attempted: {', '.join(unattempted)}"]

    def gave_up(self, run):
        """The planner's own reason for giving up, or None if it stopped for another reason."""
        if run.vlm_tamp_family:
            decisions = re.findall(r"Planner decision: (\w+)\. ?(.*)", run.trace)
            if not decisions or decisions[-1][0] != "unsolvable":
                return None
            return f"unsolvable: {decisions[-1][1]}"
        if run.planner == "tru-pomdp":
            reasons = [step["tru_pomdp"]["termination_reason"] for step in run.planner_log if step.get("tru_pomdp", {}).get("termination_reason")]
            if not reasons or reasons[-1] not in ("empty_belief", "no_useful_action"):
                return None
            return reasons[-1]
        if not re.search(r"Done\[\]\s*(Assigned!)?\s*$", run.trace.strip()):
            return None
        thoughts = re.findall(r"Thought:(.*?)(?=\n[A-Z][A-Za-z]*\[|\Z)", run.trace, re.DOTALL)
        final = thoughts[-1].strip() if thoughts else ""
        if not self.give_up_words.search(final):
            return None
        return f"Done: {final}"

class WrongObject(Failure):
    """A distractor object was used in place of a real target object. Approved substitutes are valid and do not count for this case."""

    label = "Wrong Object"

    def find(self, run):
        evidence = []
        for name in run.distractors:
            used = [
                action.name
                for action in run.actions
                if action.target == name
                and action.name not in ("Navigate", "Explore")
                and action.response.startswith("Successful")
            ]
            if not used:
                continue
            final = run.object_states[name]["final"]["location"]
            evidence.append(f"{name} (distractor): {', '.join(used)}; ended on {final['furniture']} in {final['room']}")
        return evidence

class NonexistentObject(Inefficiency):
    """A skill was attempted on an object that does not exist in the scene."""

    label = "Nonexistent Object"
    messages = (r"Object does not exist", r"not present in the graph", r"Target is not a valid object")

    def matches(self, run, action):
        if action.target in run.memory:
            return False
        return super().matches(run, action)

class PreconditionError(Inefficiency):
    """The planner attempted an action without valid prerequisites."""

    label = "Precondition Error"
    messages = (
        r"not holding any object",
        r"Not close enough",
        r"too far from",
        r"not near faucet",
        r"not close enough to a water source",
        r"you need to open the furniture first",
    )

class AffordanceError(Inefficiency):
    """An action was incorrectly executed on an object without the capacity to support the action."""

    label = "Affordance Error"
    messages = (
        r"has no receptacle for proposition",
        r"No valid placements found for entity \w+\.",
        r"is not articulated",
        r"Destination must be furniture",
    )

class ExcessSearch(Inefficiency):
    """Planner continued searching despite evidence indicating a subgoal should be forgone."""

    label = "Excess Search"
    searches = ("Explore", "Open", "Navigate")

    def find(self, run):
        if run["uncertainty"] != "ABS" or run["memory"] == "ACC":
            return []
        rooms, furniture = run.search_plan
        if not rooms and not furniture:
            return []
        give_up = self.give_up_at(run, rooms, furniture)
        if give_up is None:
            return []
        index, reason = give_up
        extra = [action for action in run.actions[index + 1 :] if action.name in self.searches]
        if not extra:
            return []
        return [f"kept searching for {'/'.join(run.absent)} for {len(extra)} actions after {reason} (a{run.actions[index].id})"]

    def give_up_at(self, run, rooms, furniture):
        """First action after which the robot had searched enough to give up, and why: it checked every place
        in the ground-truth plan, or searched as many places as the plan has. Openable furniture is checked
        once opened; other furniture once visited or once its room was explored."""
        openable = {name for name, state in run.object_states.items() if "is_open" in state["initial"].get("states", {})}
        room_of = {name: state["final"]["location"]["room"] for name, state in run.object_states.items()}
        explored, opened, visited = set(), set(), set()
        for index, action in enumerate(run.actions):
            if action.response.startswith("Successful"):
                {"Explore": explored, "Open": opened, "Navigate": visited}.get(action.name, set()).add(action.target)
            checked = opened | {piece for piece in furniture - openable if piece in visited or room_of.get(piece) in explored}
            if rooms <= explored and furniture <= checked:
                return index, "checking every place in the ground-truth plan"
            if len(explored & set(room_of.values())) + len(opened & openable) >= len(rooms) + len(furniture):
                return index, f"searching as many places as the ground-truth plan ({len(rooms) + len(furniture)})"
        return None

class UnnecessaryAction(Inefficiency):
    """A broad category for an action that makes no progress towards the goal or exploration."""

    label = "Unnecessary Action"
    category = "Unnecessary Action"

    def matches(self, run, action):
        if action.name == "Explore":
            return run["memory"] == "ACC"
        if not action.response.startswith("Successful"):
            return False
        if action.name == "Place":
            return self.placed_before_filling(run, action)
        if action.name == "Pick":
            return self.repicked_finished(run, action)
        return False

    def placed_before_filling(self, run, action):
        to_fill = {
            run.names.get(handle, handle)
            for requirement in run.requirements
            if requirement["function_name"] == "is_filled"
            for handle in requirement["args"]["object_handles"]
        }
        if action.target not in to_fill:
            return False
        fills = [
            index
            for index, other in enumerate(run.actions)
            if other.name == "Fill" and other.target == action.target and other.response.startswith("Successful")
        ]
        return bool(fills) and min(fills) > run.actions.index(action)

    def repicked_finished(self, run, action):
        """Picked up an object while every requirement naming it already held."""
        if action.id in run.repeated:
            return False
        indices = [index for index in range(len(run.requirements)) if action.target in run.objects(index)]
        if not indices:
            return False
        state = run.state_before(action)
        if state is None:
            return False
        return all(state[index] for index in indices)

class RepeatedAction(UnnecessaryAction):
    """The planner outputs the same action or sequence of actions several times in a row."""

    label = "Repeated Action"

    def find(self, run):
        evidence = []
        for block, size, count in run.repeated_blocks:
            if any(self.stale(run, action) for action in block):
                continue
            steps = " -> ".join(f"{action.name}[{action.input}]" for action in block[:size])
            evidence.append(f"{steps} x{count} (a{block[0].id}-a{block[-1].id})")
        return evidence

    def stale(self, run, action):
        if action.target not in run.memory:
            return False
        return any(re.search(message, action.response) for message in NonexistentObject.messages)

FAILURES = [TimeLimitReached(), ActionLimitReached(), ActionUndone(), WrongOrder(), MissingAction(), MissingPrerequisite(), IncorrectPlacement(), WrongObject(), ObjectNotFound(), ObjectUntouched(), TaskAbandoned()]

INEFFICIENCIES = [NonexistentObject(), PreconditionError(), AffordanceError(), UnnecessaryAction(), RepeatedAction(), ExcessSearch()]

def classify(run):
    """(kind, category, label, evidence) for every error in the run; failures only on failed runs."""
    rules = INEFFICIENCIES if run["task_state_success"] == 1 else FAILURES + INEFFICIENCIES
    findings = [(rule, evidence) for rule in rules for evidence in rule.find(run)]
    timed_out = any(isinstance(rule, TimeLimitReached) for rule, _ in findings)
    for rule, evidence in findings:
        if timed_out and rule.label in TimeLimitReached.excludes:
            continue
        yield rule.kind, rule.category, rule.label, evidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "findings.csv")
    args = parser.parse_args()
    runs = [Run(row) for row in json.loads((args.results_dir / "baseline_results.json").read_text())]
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "kind", "category", "label", "evidence"])
        for run in runs:
            for row in classify(run):
                writer.writerow([run["id"], *row])

if __name__ == "__main__":
    main()