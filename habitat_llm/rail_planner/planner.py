"""RailPlanner: VLM-TAMP prompting with per-subgoal expanders instead of PDDL.

Everything except subgoal planning (VLM turns, branches, replanning, explore,
logging) is inherited from VlmTampPddlPlanner, so the two are directly
comparable. Each symbolic subgoal is expanded into oracle skills by
habitat_llm.rail_planner.subgoal_expanders rather than solved with
PDDLStream / Fast Downward.

plan_config.replan_trigger additionally re-queries the VLM after a skill
finishes (see habitat_llm.rail_planner.replan_trigger).

waited_in(<movable>, <container>) puts an object inside an appliance and runs
the Wait skill (e.g. heating bread in a microwave), which PDDL has no action for.

plan_config.placement_oracle feeds privileged placement goals to the VLM prompt
and/or the Place calls (see habitat_llm.rail_planner.placement_oracle).

plan_config.two_phase (default false) adds a VLM mode selector at every replan
point: explore runs a ReAct exploration phase (react_explorer.py, search-only
skills) and interact runs the loop above without explore(<room>) subgoals
(see habitat_llm.rail_planner.mode_selector). With the flag off none of this runs.

plan_config.overhead_map (default none) adds a top-down map of the house to the Turn-1, mode-selector and
explorer prompts (image and/or metric text; see habitat_llm.rail_planner.overhead_map). Ground truth is used
for scene structure only (floor, rooms, furniture footprints); objects come from the robot's world graph.

plan_config.subgoal_mode (default flat) = object_centric first asks the VLM for abstract object tasks (one per
object or object group, with the conditions it must end in, no IDs), then grounds and executes them one at a time
with the usual Turn 1/Turn 2 + expanders, and finishes with one review call (see
habitat_llm.rail_planner.object_centric). With flat none of this runs.
"""

import json
import os
import time

from habitat.sims.habitat_simulator import sim_utilities as sutils

from habitat_llm.llm.instruct.utils import get_world_descr
from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.llm.instruct.utils import pil_image_to_data_url
from habitat_llm.rail_planner import object_centric as oc
from habitat_llm.rail_planner import overhead_map as omap_lib
from habitat_llm.rail_planner import placement_oracle
from habitat_llm.rail_planner.goal_verifier import build_ledger_prompt, check_ledger, parse_ledger
from habitat_llm.rail_planner.mode_selector import build_selector_prompt, parse_selector_response
from habitat_llm.utils.llm_usage import snapshot_from_llm
from habitat_llm.rail_planner.replan_trigger import (
    END_OF_PLAN_CHECKS,
    TRIGGERS,
    action_entities,
    compute_delta,
    goal_entities,
    gt_goal_status,
    should_replan,
    take_snapshot,
    unmet_goal_feedback,
)
from habitat_llm.rail_planner.subgoal_expanders import (
    WAITED_IN_PREDS,
    SubgoalPlanError,
    build_scene_state,
    expand_subgoal,
)
from habitat_llm.utils.movable_containers import container_bounds
from habitat_llm.utils.sim import is_open as sim_is_open


class RailPlanner(VlmTampPddlPlanner):
    # Appended to the VLM's predicate list (Turn 2).
    EXTRA_SUBGOALS = (
        "\n'waited_in(<movable>, <container_furniture>)': the result of putting <movable> inside "
        "<container_furniture> (closed and switched on if it is an appliance) and waiting there, e.g. to heat, "
        "warm or cook <movable>; it contains two arguments. Take <movable> out afterwards with a separate subgoal."
    )

    def reset(self):
        super().reset()
        self.replan_trigger = str(self.plan_config.get("replan_trigger", "failure"))
        if self.replan_trigger not in TRIGGERS:
            raise ValueError(f"replan_trigger must be one of {TRIGGERS}, got {self.replan_trigger!r}")
        self.max_proactive_replans = int(self.plan_config.get("max_proactive_replans", 30))
        self._trigger_prev_snapshot = None
        self._trigger_seen = set()
        self._goal_entities = None
        self._proactive_replans = 0
        self._replan_pending = False
        self.end_of_plan_check = str(self.plan_config.get("end_of_plan_check", "none"))
        if self.end_of_plan_check not in END_OF_PLAN_CHECKS:
            raise ValueError(
                f"end_of_plan_check must be one of {END_OF_PLAN_CHECKS}, got {self.end_of_plan_check!r}"
            )
        self.max_end_of_plan_replans = int(self.plan_config.get("max_end_of_plan_replans", 3))
        self._end_of_plan_replans = 0
        self._end_replan_pending = False
        self._end_feedback = []
        self._world_graph = None
        # verify_symbolic: goal ledger written by the VLM at plan end (goal_verifier.py).
        self.log_gt_shadow = bool(self.plan_config.get("log_gt_shadow", True))
        self._instruction = ""
        self._ledger_cache = {}
        self._verifier_soft_seen = set()
        self._verifier_calls = 0
        self._verifier_wall_s = 0.0
        self.placement_oracle_mode = str(self.plan_config.get("placement_oracle", "none"))
        if self.placement_oracle_mode not in placement_oracle.MODES:
            raise ValueError(
                f"placement_oracle must be one of {placement_oracle.MODES}, got {self.placement_oracle_mode!r}"
            )
        self._placement_goals = None
        self._placed_order = []
        self._container_cache = {}
        self._waited = set()
        # two_phase: VLM mode selector + ReAct exploration phases. Nothing below is used when it is off.
        self.two_phase = bool(self.plan_config.get("two_phase", False))
        self.max_explore_phases = int(self.plan_config.get("max_explore_phases", 6))
        self.explore_phase_max_skills = int(self.plan_config.get("explore_phase_max_skills", 10))
        self._mode = "interact"
        self._explore_phases = 0
        self._explore_summaries = []
        self._phase = None
        self._two_phase_started = False
        self._explorer = getattr(self, "_explorer", None)
        if self._explorer is not None:
            self._explorer.reset()
        # overhead_map: top-down map in the prompts. Nothing below is used when it is none.
        self.overhead_map_level = str(self.plan_config.get("overhead_map", "none"))
        if self.overhead_map_level not in omap_lib.LEVELS:
            raise ValueError(f"overhead_map must be one of {omap_lib.LEVELS}, got {self.overhead_map_level!r}")
        self.overhead_map_format = str(self.plan_config.get("overhead_map_format", "image"))
        if self.overhead_map_format not in omap_lib.FORMATS:
            raise ValueError(
                f"overhead_map_format must be one of {omap_lib.FORMATS}, got {self.overhead_map_format!r}"
            )
        cfg = self.plan_config.get("overhead_map_cfg") or {}
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass
        self.overhead_map_cfg = {**omap_lib.DEFAULT_CFG, **dict(cfg)}
        bad = set(self.overhead_map_cfg["attach_to"]) - set(omap_lib.ATTACH_POINTS)
        if bad:
            raise ValueError(f"overhead_map_cfg.attach_to: unknown {sorted(bad)}; use {omap_lib.ATTACH_POINTS}")
        self._omap = None
        self._omap_failed = False
        self._omap_index = 0
        # subgoal_mode=object_centric: abstract object tasks, grounded one at a time. Unused when flat.
        self.subgoal_mode = str(self.plan_config.get("subgoal_mode", "flat"))
        if self.subgoal_mode not in oc.SUBGOAL_MODES:
            raise ValueError(f"subgoal_mode must be one of {oc.SUBGOAL_MODES}, got {self.subgoal_mode!r}")
        self.object_task_max_regrounds = int(self.plan_config.get("object_task_max_regrounds", 4))
        self.object_task_max_reviews = int(self.plan_config.get("object_task_max_reviews", 2))
        self._tasks = None
        self._task_advance_pending = False
        self._task_reviews = 0
        self._oc_final = False

    def get_next_action(self, instruction, observations, world_graphs):
        agent_uid = self._agents[0].uid if self._agents else 0
        world_graph = world_graphs[agent_uid]
        self._world_graph = world_graph
        self._instruction = instruction
        self._omap_observe(observations, world_graph, agent_uid)

        if getattr(self, "two_phase", False) and not self.is_done:
            if self._mode == "explore":
                return self._explore_step(instruction, observations, world_graphs, agent_uid)
            if not self.branches and not self._two_phase_started:
                # First decision of the episode. The base Phase 1 would abort on an explore choice
                # (no branches), so make it here.
                self._two_phase_started = True
                if not self.trace:
                    self._trace_append(f"Task: {instruction}")
                self._generate_subgoals(instruction, world_graph, observations, append=False)
                if self._mode == "explore":
                    return self._explore_step(instruction, observations, world_graphs, agent_uid)

        reason = (
            "new_observation" if self._replan_pending
            else "plan_finished" if self._end_replan_pending
            else "object_task_done" if getattr(self, "_task_advance_pending", False)
            else None
        )
        if reason and not self.last_high_level_actions and not self.is_done:
            # Same prompt for every trigger: the robot's own observations only.
            self._replan_pending = self._end_replan_pending = self._task_advance_pending = False
            self._vprint_header(f"REPLANNING ({reason})", "yellow")
            if reason == "object_task_done":
                self._advance_object_task(instruction, observations, world_graph)
            elif reason == "plan_finished" and self._end_feedback:
                self._replan_with_feedback(instruction, observations, world_graph)
            else:
                self._observe_and_continue(instruction, observations, world_graph, reason)
            self._trigger_prev_snapshot = self._trigger_snapshot(world_graph, agent_uid)
            info = self._planner_info(agent_uid, hl_override={}, replanned={agent_uid: True})
            if self.is_done:
                info["is_done"] = {agent_uid: True}
            return {}, info, self.is_done

        hl_before = dict(self.last_high_level_actions)
        out = super().get_next_action(instruction, observations, world_graphs)
        self._note_explored(hl_before.get(agent_uid), (out[1].get("responses") or {}).get(agent_uid))
        if self._end_replan_pending or getattr(self, "_task_advance_pending", False):
            # _finish_planning refused plan_completed (end-of-plan check, or the next object task): keep going.
            return {}, self._planner_info(agent_uid, hl_override={}), False
        if self.replan_trigger != "failure" and not self.is_done and getattr(self, "_mode", "") != "explore":
            self._check_replan_trigger(out[1], hl_before.get(agent_uid), world_graph, agent_uid)
        return out

    # ------------------------------------------------------------------
    # two_phase: mode selector + ReAct exploration phases
    # ------------------------------------------------------------------

    def _get_explorer(self):
        if self._explorer is None:
            from habitat_llm.rail_planner.react_explorer import ExploreReactPlanner

            cfg = self.plan_config.get("explore_react")
            if cfg is None:
                raise ValueError("two_phase=true needs plan_config.explore_react (see conf/planner/rail_planner.yaml)")
            explorer = ExploreReactPlanner(cfg, self.env_interface)
            # Share the agent (tools, world graph) without LLMPlanner.agents' setter, which would hand the
            # ReAct LLM to the agent's tools.
            explorer._agents = self._agents
            explorer.set_image_save_dir(os.path.join(self._get_log_dir(), "explore_react_images"))
            if self._omap_attached("explorer"):
                explorer.overhead_map_provider = self._omap_explorer_parts
            self._explorer = explorer
        return self._explorer

    @staticmethod
    def _selector_reason(kwargs) -> str:
        reason = kwargs.get("observation_reason")
        if reason == "explore_phase_done":
            return "an exploration phase just finished"
        if reason == "plan_finished":
            return "the plan finished but the task is not complete yet"
        if reason == "new_observation":
            return "the robot observed a change relevant to the plan"
        if kwargs.get("refresh_failure_observation"):
            return "a planned subgoal or action failed"
        if reason:
            return str(reason)
        return "start of the task" if not kwargs.get("append") else "replanning"

    def _choose_mode(self, instruction, world_graph, observations, kwargs) -> str:
        reason = self._selector_reason(kwargs)
        left = self.max_explore_phases - self._explore_phases
        if left <= 0:
            self._log_event({"event": "mode_selection", "reason": reason, "mode": "interact", "forced": True,
                             "explore_phases": self._explore_phases})
            return "interact"
        agent_uid = self._agents[0].uid if self._agents else 0
        # Same memory text the VLM sees; do not consume the one-time initial-location line.
        scene = get_world_descr(
            world_graph, agent_uid=agent_uid, include_room_name=True, add_state_info=True, centralized=False,
            include_initial_agent_location=not getattr(self, "_initial_agent_location_sent", False),
        )
        if self._omap_attached("selector") and self.overhead_map_format in ("text", "both"):
            scene = scene + "\n\n" + self._omap.to_text(world_graph)
        prompt = build_selector_prompt(instruction, scene, reason, self.history, self._explore_summaries, left)
        urls = []
        if self.use_images:
            fresh = self._observe_failure_state(observations, reason="mode_selection")
            urls = self._get_annotated_vlm_image_urls(fresh, world_graph, annotate=False)
            if urls:
                prompt = "The attached image is the robot's current third-person view.\n\n" + prompt
        extra_kwargs = {}
        map_urls, map_labels = self._extra_vlm_images(world_graph, "selector")
        if map_urls:
            extra_kwargs["image_labels"] = [None] * len(urls) + list(map_labels)
            urls = list(urls) + list(map_urls)
        self._vprint_header("MODE SELECTOR", "magenta")
        self.vlm.new_session()
        self._append_vlm_prompt(prompt_kind="mode_selection", prompt_text=prompt, image_count=len(urls))
        response = self._ask_vlm(
            prompt,
            image_data_urls=urls or None,
            max_completion_tokens=self.vlm_max_tokens,
            temperature=self.vlm_temperature,
            reasoning_effort=self.vlm_reasoning_effort,
            **extra_kwargs,
        )
        mode, why, goal, parsed = parse_selector_response(response)
        self._log_event({
            "event": "mode_selection", "reason": reason, "mode": mode, "selector_reason": why,
            "explore_goal": goal, "parsed": parsed, "forced": False, "explore_phases": self._explore_phases,
            "image_count": len(urls), "image_paths": self._omap_last_paths() if map_urls else [],
            "prompt": prompt, "response": response,
            "api_response": self.vlm.get_last_response_metadata(),
        })
        self._vprint(f"  [mode] {mode}: {why}" + (f" | goal: {goal}" if goal else ""), "magenta")
        self._trace_append(f"Mode selector ({reason}): {mode}. {why}" + (f" Explore goal: {goal}" if goal else ""))
        if mode == "explore":
            self._begin_explore_phase(goal, why, source="selector")
        return mode

    def _begin_explore_phase(self, goal: str, reason: str, source: str) -> None:
        self._explore_phases += 1
        self._mode = "explore"
        explorer = self._get_explorer()
        explorer.start_phase(goal, self._explore_summaries)
        world_graph = self._world_graph
        self._phase = {
            "index": self._explore_phases, "goal": explorer.phase_goal, "reason": reason, "source": source,
            "skills": 0, "llm_calls": 0, "actions": [],
            "start_memory": self._memory_snapshot(world_graph) if world_graph is not None else {},
        }
        self.current_plan = []
        self.current_action_idx = 0
        self.last_high_level_actions = {}
        self._subgoal_retry_count = 0
        self._replan_pending = self._end_replan_pending = False
        self._log_event({"event": "explore_phase_start", "phase": self._explore_phases, "goal": explorer.phase_goal,
                         "reason": reason, "source": source})
        self._vprint_header(f"EXPLORATION PHASE {self._explore_phases}: {explorer.phase_goal}", "cyan")
        self._trace_append(f"Exploration phase {self._explore_phases} ({source}): {explorer.phase_goal}")

    def _explore_step(self, instruction, observations, world_graphs, agent_uid):
        """One tick of the ReAct exploration phase; ends the phase and replans when it is over."""
        explorer, phase = self._explorer, self._phase
        world_graph = world_graphs[agent_uid]
        if explorer.replan_required and not explorer.is_done:
            # The ReAct LLM is about to choose a new action.
            if phase["skills"] >= self.explore_phase_max_skills:
                return self._end_explore_phase("skill_cap", instruction, observations, world_graph, agent_uid)
            if phase["llm_calls"] >= 2 * self.explore_phase_max_skills:
                return self._end_explore_phase("llm_call_cap", instruction, observations, world_graph, agent_uid)
            if self._action_cap_reached():
                return {}, self._planner_info(agent_uid, hl_override={}, is_done={agent_uid: True}), True
        asked = explorer.replan_required
        explorer.rejected = False
        low, info, done = explorer.get_next_action(
            explorer.phase_instruction(instruction), observations, world_graphs
        )
        if asked:
            phase["llm_calls"] += 1
        replanned = bool((info.get("replanned") or {}).get(agent_uid))
        hl = (info.get("high_level_actions") or {}).get(agent_uid)
        thought = (info.get("thought") or {}).get(agent_uid)
        responses = dict(info.get("responses") or {})
        if done:
            return self._end_explore_phase(
                "done" if not explorer._termination else str(explorer._termination.get("reason", "terminated")),
                instruction, observations, world_graph, agent_uid,
            )
        if replanned and explorer.rejected:
            self._log_event({"event": "explore_phase_step", "phase": phase["index"], "thought": thought,
                             "action": list(hl[:2]) if hl else None, "rejected": True,
                             "result": responses.get(agent_uid)})
            self._trace_append(f"[explore {phase['index']}] rejected {hl[0] if hl else hl}: not an exploration action")
            # Nothing was dispatched: do not log or count an action.
            return {}, self._planner_info(agent_uid, hl_override={}, replanned={agent_uid: False}), False
        if replanned and hl:
            self._num_actions += 1
            phase["skills"] += 1
            phase["actions"].append(f"{hl[0]}[{hl[1]}]")
            self._log_event({"event": "explore_phase_step", "phase": phase["index"], "thought": thought,
                             "action": list(hl[:2]), "rejected": False, "skill": phase["skills"]})
            self._trace_append(f"[explore {phase['index']}] Thought: {thought}\nAction: {hl[0]}[{hl[1]}]")
        result = responses.get(agent_uid)
        self._note_explored(hl, result)
        if result:
            phase["actions"][-1:] = [f"{a} -> {str(result)[:120]}" for a in phase["actions"][-1:]]
            self._log_event({"event": "explore_phase_result", "phase": phase["index"],
                             "action": list(hl[:2]) if hl else None, "result": result})
            self._trace_append(f"[explore {phase['index']}] Result: {result}")
        out = self._planner_info(
            agent_uid, hl_override={agent_uid: hl} if hl else {}, responses=responses,
            replanned={agent_uid: replanned}, thought={agent_uid: thought},
        )
        if info.get("action_image"):
            out["action_image"] = info["action_image"]
        return low, out, False

    def _end_explore_phase(self, why, instruction, observations, world_graph, agent_uid):
        phase, explorer = self._phase, self._explorer
        after = self._memory_snapshot(world_graph)
        before = phase["start_memory"]
        new = sorted(set(after) - set(before))
        moved = sorted(k for k in after if k in before and after[k] != before[k])
        summary = (
            f"phase {phase['index']} (goal: {phase['goal']}): {phase['skills']} actions"
            f" [{'; '.join(phase['actions']) or 'none'}]; newly found: {', '.join(new) or 'nothing'}"
            + (f"; changed: {', '.join(moved)}" if moved else "") + f"; ended: {why}"
        )
        self._explore_summaries.append(summary)
        self.history.append(f"Exploration {summary}")
        self._log_event({"event": "explore_phase_end", "phase": phase["index"], "why": why, "skills": phase["skills"],
                         "llm_calls": phase["llm_calls"], "actions": phase["actions"], "new_objects": new,
                         "changed_objects": moved})
        try:
            path = os.path.join(self._get_log_dir(), f"explore_phase_{phase['index']:02d}.txt")
            with open(path, "w") as f:
                f.write(explorer.curr_prompt)
        except Exception:
            pass
        self._trace_append(f"Exploration phase {phase['index']} ended ({why}). Newly found: {', '.join(new) or 'nothing'}")
        self._vprint_header(f"EXPLORATION PHASE {phase['index']} ENDED ({why})", "cyan")
        self._mode, self._phase = "interact", None
        explorer.last_high_level_actions = {}

        # Replan with the updated memory; the selector may start another phase.
        fresh = self._observe_failure_state(observations, reason="explore_phase_done")
        self._reprompt_round += 1
        self._log_event({"event": "reprompt_started", "reason": "explore_phase_done", "branch": self.branch_idx,
                         "subgoal_idx": self.subgoal_idx, "reprompt_round": self._reprompt_round})
        self.current_plan = []
        self.current_action_idx = 0
        self.last_high_level_actions = {}
        self._subgoal_retry_count = 0
        history = (
            "Replanning reason: an exploration phase just finished.\n"
            f"Exploration goal: {phase['goal']}\n"
            f"Exploration actions: {'; '.join(phase['actions']) or 'none'}\n"
            f"Newly found objects: {', '.join(new) or 'none'}\n"
            f"Previously achieved subgoals: {self._already_succeeded_subgoals}.\n"
            "Plan for the ORIGINAL task using the updated memory."
        )
        self._generate_subgoals(instruction, world_graph, fresh, history_str=history, append=True,
                                observation_reason="explore_phase_done")
        info = self._planner_info(agent_uid, hl_override={}, replanned={agent_uid: True})
        if self.is_done:
            info["is_done"] = {agent_uid: True}
        return {}, info, self.is_done

    def _handle_turn1_decision(self, decision: dict) -> bool:
        cur = self._current_task() if self._object_centric_on() else None
        if cur is not None and decision.get("decision") in oc.TERMINAL:
            # Grounding a single object task: complete/unsolvable end that task, not the episode.
            status = "done" if decision["decision"] == "complete" else "failed"
            self._end_task(cur, status, f"VLM: {decision['decision']}: {decision.get('reason', '')}")
            return True
        if getattr(self, "two_phase", False) and decision.get("decision") == "explore":
            if self._explore_phases >= self.max_explore_phases:
                self._finish_planning("budget_exhausted", "exploration requested with no exploration phases left")
                return True
            self._log_event({"event": "mode_selection", "reason": "interaction VLM requested exploration",
                             "mode": "explore", "explore_goal": str(decision.get("reason", "")), "forced": False,
                             "source": "interaction_vlm", "explore_phases": self._explore_phases})
            self._begin_explore_phase(str(decision.get("reason", "")), "interaction VLM requested exploration",
                                      source="interaction_vlm")
            return True
        return super()._handle_turn1_decision(decision)

    def _build_english_prompt(self, **kwargs) -> str:
        if self.two_phase:
            kwargs["allow_explore"] = False
        prompt = super()._build_english_prompt(**kwargs)
        if self._omap_attached("turn1") and self.overhead_map_format in ("text", "both"):
            prompt = prompt + "\n\n" + self._omap.to_text(self._world_graph)
        return prompt

    def _build_predicate_prompt(self, *args, **kwargs) -> str:
        if self.two_phase:
            kwargs["allow_explore"] = False
        return super()._build_predicate_prompt(*args, **kwargs)

    def _planner_info(self, agent_uid, **kwargs):
        out = super()._planner_info(agent_uid, **kwargs)
        tasks = getattr(self, "_tasks", None)
        if self._object_centric_on() and tasks is not None:
            cost = dict(out.get("cost_metrics") or {})
            cost["object_tasks"] = len(tasks)
            cost["object_tasks_done"] = sum(t.status == "done" for t in tasks)
            cost["object_tasks_failed"] = sum(t.status == "failed" for t in tasks)
            cost["object_task_reviews"] = self._task_reviews
            out["cost_metrics"] = cost
        if getattr(self, "end_of_plan_check", "none") == "verify_symbolic":
            # Ledger calls are already in llm_requests / tokens / USD; report them separately too.
            cost = dict(out.get("cost_metrics") or {})
            cost["verifier_llm_requests"] = getattr(self, "_verifier_calls", 0)
            cost["verifier_llm_time_s"] = getattr(self, "_verifier_wall_s", 0.0)
            out["cost_metrics"] = cost
        explorer = getattr(self, "_explorer", None)
        if not getattr(self, "two_phase", False) or explorer is None:
            return out
        # One episode budget: add the ReAct phase's LLM time, calls and tokens to the planner's own.
        cost = dict(out.get("cost_metrics") or {})
        cost["llm_planning_time_s"] = cost.get("llm_planning_time_s", 0.0) + explorer.llm_planning_time_s
        cost["llm_requests"] = cost.get("llm_requests", 0) + explorer.replanning_count
        cost["explore_phase_llm_time_s"] = explorer.llm_planning_time_s
        cost["explore_phase_llm_requests"] = explorer.replanning_count
        cost["explore_phases"] = self._explore_phases
        usage = snapshot_from_llm(explorer.llm)
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens", "llm_calls_with_usage"):
            if usage.get(key) is not None:
                cost[key] = (cost.get(key) or 0) + usage[key]
        if usage.get("llm_usd") is not None:
            cost["llm_usd"] = (cost.get("llm_usd") or 0.0) + usage["llm_usd"]
        out["cost_metrics"] = cost
        return out

    def _finish_planning(self, reason, evidence=""):
        """With an end_of_plan_check, refuse plan_completed while the goal is unmet.

        The base planner stops as soon as the active branch runs out of
        subgoals, without checking the goal. Only plan_completed is checked;
        the VLM's own terminal decisions and budget limits still end the episode.
        oracle / oracle_feedback read the simulator's goal check (privileged);
        oracle_feedback also records what is unmet for the replanning prompt.
        verify_symbolic uses the robot's own goal ledger instead (_verify_goal).
        With subgoal_mode=object_centric, plan_completed first only closes the
        current object task; the end-of-plan check runs when the review says the
        whole task is complete.
        """
        if self._object_centric_on() and reason == "plan_completed" and not self._oc_final:
            cur = self._current_task()
            if cur is not None:
                self._end_task(cur, "done", "its subgoals finished")
                self._task_advance_pending = True
                return
        self._oc_final = False
        if (
            reason == "plan_completed"
            and self.end_of_plan_check != "none"
            and self._end_of_plan_replans < self.max_end_of_plan_replans
        ):
            extra = {}
            feedback = []
            if self.end_of_plan_check == "verify_symbolic":
                satisfied, feedback, extra = self._verify_goal()
                percent = None
                if getattr(self, "log_gt_shadow", True):
                    # Evaluation only (verifier precision/recall): never reaches the planner.
                    gt_ok, gt_percent = gt_goal_status(self.env_interface)
                    known = (
                        {o.name for o in self._world_graph.get_all_objects()}
                        if self._world_graph is not None else set()
                    )
                    extra.update(
                        gt_goal_satisfied_shadow=gt_ok,
                        gt_percent_shadow=gt_percent,
                        gt_feedback_shadow=unmet_goal_feedback(self.env_interface, known),
                    )
            else:
                satisfied, percent = gt_goal_status(self.env_interface)
            replan = satisfied is False
            self._end_feedback = []
            if replan:
                self._end_of_plan_replans += 1
                self._end_replan_pending = True
                if self.end_of_plan_check == "oracle_feedback" and self._world_graph is not None:
                    known = {o.name for o in self._world_graph.get_all_objects()}
                    self._end_feedback = unmet_goal_feedback(self.env_interface, known)
                elif self.end_of_plan_check == "verify_symbolic":
                    self._end_feedback = list(feedback)
                self._trace_append(f"End-of-plan check ({self.end_of_plan_check}): goal not met; replanning")
            self._log_event({
                "event": "end_of_plan_check",
                "check": self.end_of_plan_check,
                # verify_symbolic: the verifier's verdict; ground truth is only in the *_shadow keys.
                "goal_satisfied": satisfied,
                "percent_complete": percent,
                "replan": replan,
                "feedback": list(self._end_feedback),
                "end_of_plan_replans": self._end_of_plan_replans,
                **extra,
            })
            if replan:
                return
        super()._finish_planning(reason, evidence)

    def _note_explored(self, hl_action, response) -> None:
        """Remember rooms a physical Explore skill finished in (the base only tracks fast explore)."""
        if hl_action and hl_action[0] == "Explore" and response and not self._response_failed(response):
            self.__dict__.setdefault("_explored_rooms", set()).add(str(hl_action[1]).strip())

    def _verify_goal(self):
        """Deployable end-of-plan check: (satisfied, feedback lines, log detail).

        The VLM writes the goal ledger from the instruction and the robot's
        memory text only (not the plan), in a fresh session; it is cached while
        the set of known objects is unchanged. The ledger is evaluated against
        the same synced scene state the subgoal expanders plan from. satisfied
        is None (plan accepted) when no usable ledger could be obtained.
        """
        world_graph = self._world_graph
        if world_graph is None:
            return None, [], {"verifier_error": "no world graph"}
        agent_uid = self._agents[0].uid if self._agents else 0
        extra = {}
        try:
            state = self._synced_scene_state(world_graph, agent_uid)
            key = frozenset(state.objects)
            ledger = self._ledger_cache.get(key)
            cached = ledger is not None
            if not cached:
                scene = get_world_descr(
                    world_graph, agent_uid=agent_uid, include_room_name=True, add_state_info=True, centralized=False
                )
                prompt = build_ledger_prompt(self._instruction, scene)
                self._vprint_header("GOAL LEDGER (end-of-plan verifier)", "magenta")
                self.vlm.new_session()
                self._append_vlm_prompt(prompt_kind="goal_ledger", prompt_text=prompt, image_count=0)
                started = time.monotonic()
                try:
                    response = self._ask_vlm(
                        prompt,
                        max_completion_tokens=self.vlm_max_tokens,
                        temperature=self.vlm_temperature,
                        reasoning_effort=self.vlm_reasoning_effort,
                    )
                finally:
                    self._verifier_calls += 1
                    self._verifier_wall_s += time.monotonic() - started
                conditions, missing, dropped = parse_ledger(response)
                ledger = {"conditions": conditions, "missing": missing, "dropped": dropped}
                self._log_event({
                    "event": "goal_ledger", "prompt": prompt, "response": response, **ledger,
                    "api_response": self.vlm.get_last_response_metadata(),
                })
                if conditions or missing:
                    self._ledger_cache[key] = ledger
            extra = {"ledger": ledger["conditions"], "ledger_missing": ledger["missing"],
                     "dropped": ledger["dropped"], "ledger_cached": cached}
            if not ledger["conditions"] and not ledger["missing"]:
                return None, [], {**extra, "verifier_error": "empty or unparsable ledger"}
            unexplored = sorted(state.rooms - set(getattr(self, "_explored_rooms", ())))
            unmet, soft, detail = check_ledger(
                state, ledger["conditions"], ledger["missing"], unexplored, self._verifier_soft_seen
            )
        except Exception as exc:
            self._vprint(f"  [WARN] goal verifier failed; accepting the plan: {exc}", "red")
            return None, [], {**extra, "verifier_error": repr(exc)}
        extra.update(unmet=unmet, soft=soft, conditions=detail, unexplored_rooms=unexplored)
        for line in unmet + soft:
            self._vprint(f"  [verify] {line}", "yellow")
        return not (unmet or soft), unmet + soft, extra

    def _replan_with_feedback(self, instruction, observations, world_graph) -> None:
        """Like _observe_and_continue, but tells the VLM what is still wrong."""
        fresh = self._observe_failure_state(observations, reason="plan_finished")
        self._reprompt_round += 1
        self._log_event({"event": "reprompt_started", "reason": "plan_finished",
                         "feedback": list(self._end_feedback), "branch": self.branch_idx,
                         "subgoal_idx": self.subgoal_idx, "reprompt_round": self._reprompt_round})
        self.current_plan = []
        self.current_action_idx = 0
        self.last_high_level_actions = {}
        self._subgoal_retry_count = 0
        if self.end_of_plan_check == "verify_symbolic":
            # self.history is what the two_phase mode selector reads: let it see e.g. "Not found yet".
            self.history.append("End-of-plan check: " + " ".join(self._end_feedback))
        history = (
            "Replanning reason: the plan finished, but the task is NOT complete yet.\n"
            + "".join(f"- {line}\n" for line in self._end_feedback)
            + f"Previously achieved subgoals: {self._already_succeeded_subgoals}.\n"
            "Replan for the ORIGINAL task and fix the problems above before declaring it complete."
        )
        self._generate_subgoals(instruction, world_graph, fresh, history_str=history,
                                append=True, observation_reason="plan_finished")

    def _trigger_snapshot(self, world_graph, agent_uid: int):
        # The world graph's held object goes stale after Place; trust the gripper.
        return take_snapshot(
            world_graph,
            agent_uid,
            lambda state: self._sync_holding_with_gripper(state, world_graph, agent_uid),
        )

    def _check_replan_trigger(self, info, hl_action, world_graph, agent_uid: int) -> None:
        """After a completed non-Explore skill, decide whether to replan next call."""
        if self._trigger_prev_snapshot is None or "replanned" in info:
            # First plan or the base just replanned: measure deltas from here.
            self._trigger_prev_snapshot = self._trigger_snapshot(world_graph, agent_uid)
            return
        responses = info.get("responses") or {}
        response = next((r for r in responses.values() if r), None)
        if not hl_action or response is None or self.last_high_level_actions:
            return  # skill still running, or a failure retry already issued the next one
        if hl_action[0] == "Explore" or self._response_failed(response):
            # The base replans after Explore and handles failures itself.
            self._trigger_prev_snapshot = self._trigger_snapshot(world_graph, agent_uid)
            return

        cur = self._trigger_snapshot(world_graph, agent_uid)
        delta = compute_delta(self._trigger_prev_snapshot, cur, action_entities(hl_action))
        self._trigger_prev_snapshot = cur
        goal, plan = set(), set()
        if self.replan_trigger == "oracle":
            if self._goal_entities is None:
                self._goal_entities = goal_entities(self.env_interface)
            goal = self._goal_entities
            for sg in self._active_subgoals[self.subgoal_idx:]:
                plan.update(self._parse_subgoal(sg)[1])
        fire, rule = should_replan(self.replan_trigger, delta, goal, plan, self._trigger_seen)
        capped = fire and self._proactive_replans >= self.max_proactive_replans
        if fire and not capped:
            self._replan_pending = True
            self._proactive_replans += 1
            self._trace_append(f"Replan trigger ({self.replan_trigger}) fired: {rule}")
        self._log_event({
            "event": "replan_trigger",
            "trigger": self.replan_trigger,
            "action": list(hl_action[:2]),
            "delta": [list(c) for c in delta],
            "rule": rule,
            "fired": bool(fire and not capped),
            "capped": capped,
            "proactive_replans": self._proactive_replans,
        })

    def _plan_for_subgoal(self, subgoal: str, world_graph, agent_uid: int):
        """Same contract as the PDDL version: (plan, name_map, reverse_map, state_facts).

        plan is None on failure and [] if already satisfied. Entity names are
        used verbatim, so both name maps are empty.
        """
        state = self._synced_scene_state(world_graph, agent_uid)
        facts = state.facts()

        hdr = f"SUBGOAL PLANNER  Branch {self.branch_idx} | Subgoal {self.subgoal_idx}: {subgoal}"
        self._subgoal_exec_log_begin_subgoal()
        self._vprint_header(hdr, "blue")
        self._subgoal_exec_log_append(f"══ {hdr} ══")
        held = f"holding {state.holding}" if state.holding else "hand empty"
        self._vprint(f"  State: agent in {state.room(state.agent)}, {held}", "gray")
        self._subgoal_exec_log_append(f"  State: agent in {state.room(state.agent)}, {held}")

        pred, args = self._parse_subgoal(subgoal)
        try:
            if not pred:
                raise SubgoalPlanError("could not parse subgoal")
            plan = expand_subgoal(state, pred, args)
        except SubgoalPlanError as e:
            self._log_subgoal_state(subgoal, facts, f"failed: {e}")
            self._trace_append(f"Subgoal: {subgoal}")
            self._trace_append(f"PDDL plan: None ({e})")
            self._vprint(f"  [FAIL] {e}", "red")
            self._subgoal_exec_log_append(f"  [FAIL] {e}")
            # The base class follows this with "Failed to plan for subgoal: ...";
            # the reason lands just before it in the history sent on replanning.
            self.history.append(f"Cannot plan {subgoal}: {e}")
            return None, {}, {}, facts

        self._log_subgoal_state(subgoal, facts, f"plan: {plan}")
        if not plan:
            self._vprint("  [already satisfied — skipping]", "green")
            self._subgoal_exec_log_append("  [already satisfied — skipping]")
            self._trace_append(f"Subgoal already satisfied: {subgoal}")
            return [], {}, {}, facts

        self._trace_append(f"Subgoal: {subgoal}")
        self._trace_append(f"PDDL plan: {plan}")
        self._vprint(f"  Plan ({len(plan)} steps):", "white")
        self._subgoal_exec_log_append(f"  Plan ({len(plan)} steps):")
        for i, step in enumerate(plan):
            line = f"    {i + 1}. {step[0]}({', '.join(str(a) for a in step[1])})"
            self._vprint(line, "gray")
            self._subgoal_exec_log_append(line)
        return plan, {}, {}, facts

    def _synced_scene_state(self, world_graph, agent_uid: int):
        """The robot's scene state with stale world-graph facts corrected: held object from the
        gripper, open/closed and movable-container contents from the sim, and finished waits."""
        state = build_scene_state(world_graph, agent_uid)
        self._sync_holding_with_gripper(state, world_graph, agent_uid)
        self._sync_open_with_sim(state, world_graph)
        self._sync_containers_with_sim(state, world_graph)
        state.waited |= self._waited
        return state

    def _sync_holding_with_gripper(self, state, world_graph, agent_uid: int) -> None:
        """Trust the robot's own gripper over the world graph's last_held_object.

        The world graph sets last_held_object when a Pick is issued, so a pick
        that silently fails leaves it stale; every replan would then emit the
        same doomed Place until the episode times out. It can also name a
        different object than the one actually grasped, which would emit a
        Place for the wrong object.
        """
        try:
            sim = self.env_interface.sim
            grasp_mgr = sim.agents_mgr[agent_uid].grasp_mgr
        except Exception:
            return
        if not grasp_mgr.is_grasped:
            if state.holding is not None:
                self._vprint(
                    f"  [SYNC] gripper is empty; world graph said holding {state.holding}",
                    "yellow",
                )
                state.holding = None
                agent_node = world_graph.get_node_from_name(state.agent)
                agent_node.properties.pop("last_held_object", None)
            return
        try:
            handle = sim.get_rigid_object_manager().get_object_handle_by_id(
                grasp_mgr.snap_idx
            )
            name = self.env_interface.full_world_graph.get_node_from_sim_handle(
                handle
            ).name
        except Exception:
            return
        if name not in state.objects or name == state.holding:
            return
        if state.holding is not None:
            self._vprint(
                f"  [SYNC] gripper holds {name}; world graph said holding {state.holding}",
                "yellow",
            )
            world_graph.get_node_from_name(state.agent).properties[
                "last_held_object"
            ] = world_graph.get_node_from_name(name)
        state.holding = name
        state.placement.pop(name, None)

    def _sync_open_with_sim(self, state, world_graph) -> None:
        """Read articulated open/closed from the sim; the robot's world graph never
        updates is_open after Open/Close, so it would re-open forever and treat
        closed(x) as satisfied while x is still open.
        """
        for name in state.articulated:
            try:
                is_open_now = sim_is_open(
                    world_graph.get_node_from_name(name), self.env_interface
                )
            except Exception:
                continue
            if is_open_now:
                state.open.add(name)
                state.known_closed.discard(name)
            else:
                state.open.discard(name)
                state.known_closed.add(name)

    def _sim_object(self, world_graph, name):
        try:
            handle = world_graph.get_node_from_name(name).sim_handle
            return self.env_interface.sim.get_rigid_object_manager().get_object_by_handle(handle)
        except Exception:
            return None

    def _movable_containers(self, world_graph, names) -> set:
        """Known objects with an annotated interior (habitat_llm/utils/movable_containers),
        i.e. the ones OraclePlaceSkill accepts with ``within``. Cached per episode."""
        cache = self.__dict__.setdefault("_container_cache", {})
        for name in names:
            if name not in cache:
                obj = self._sim_object(world_graph, name)
                cache[name] = obj is not None and container_bounds(obj) is not None
        return {n for n in names if cache[n]}

    def _sync_containers_with_sim(self, state, world_graph) -> None:
        """Record movable containers and their contents. The world graph keeps
        a basket as an Object, so the furniture-only state would never see an
        object inside it."""
        state.object_containers = self._movable_containers(world_graph, state.objects)
        if not state.object_containers:
            return
        ids = {}
        for name in state.object_containers:
            obj = self._sim_object(world_graph, name)
            if obj is not None:
                ids[obj.object_id] = name
        for name in state.objects - state.object_containers:
            obj = self._sim_object(world_graph, name)
            if obj is None or name == state.holding:
                continue
            try:
                parents = sutils.within(self.env_interface.sim, obj)
            except Exception:
                continue
            for parent_id in parents:
                if parent_id in ids:
                    state.contained_in[name] = ids[parent_id]
                    break

    def _build_objects_by_type(self, world_graph, visible_object_names=None):
        # The predicate prompt defines in(<movable>, <container_furniture>), so list
        # movable containers there too (they stay movable) or the VLM drops in(x, basket).
        out = super()._build_objects_by_type(world_graph, visible_object_names=visible_object_names)
        boxes = self._movable_containers(world_graph, out.get("movable", []))
        if boxes:
            out["container_furniture"] = sorted(set(out.get("container_furniture", [])) | boxes)
        return out

    def _validate_subgoal(self, subgoal: str) -> bool:
        # The base check rejects in(x, y) whenever y is movable; annotated movable
        # containers are valid destinations here (expand_in handles them).
        pred, args = self._parse_subgoal(subgoal)
        if getattr(self, "two_phase", False) and pred in ("explore", "searched-room", "search-room"):
            self._vprint(f"  [SKIP] {subgoal!r}: two_phase explores in a separate phase.", "red")
            return False
        if pred in WAITED_IN_PREDS:
            ok = (
                len(args) >= 2
                and args[0] in self._objects_by_type.get("movable", [])
                and args[1] in self._objects_by_type.get("container_furniture", [])
            )
            if not ok:
                self._vprint(
                    f"  [SKIP] {subgoal!r}: needs a movable and a name from the container_furniture list.",
                    "red",
                )
            return ok
        if pred in ("in", "inside", "within") and len(args) >= 2:
            movables = set(self._objects_by_type.get("movable", []))
            world_graph = getattr(self, "_world_graph", None)
            if (
                args[0] in movables
                and args[1] in movables
                and args[0] != args[1]
                and world_graph is not None
                and self._movable_containers(world_graph, [args[1]])
            ):
                return True
        return super()._validate_subgoal(subgoal)

    def _action_to_tool(self, action, next_action=None):
        # Expanders put the real navigation target in args[1]; the base class
        # instead infers it from the next action because PDDL picks it arbitrarily.
        name = action[0].lower()
        if name == "navigate":
            return "Navigate", self._restore_name(action[1][1])
        if name in ("place_on", "place_in"):
            return self._place_tool(name, action[1])
        if name == "place_next_to":
            # (agent, obj, furn, room, ref): the VLM's own next_to subgoal (plan_config.next_to_subgoal).
            obj, furn, ref = (self._restore_name(action[1][i]) for i in (1, 2, 4))
            return "Place", f"{obj}, on, {furn}, next_to, {ref}"
        if name == "wait":
            return "Wait", ""
        return super()._action_to_tool(action, next_action=next_action)

    def _on_skill_succeeded(self, action) -> None:
        # waited_in(obj, appliance) holds once its Wait step has finished.
        if action[0].lower() == "wait":
            self._waited.add((action[1][1], action[1][2]))

    # ------------------------------------------------------------------
    # overhead_map: top-down map (scene structure from the sim, objects from the world graph only)
    # ------------------------------------------------------------------

    def _omap_attached(self, point: str) -> bool:
        return (
            getattr(self, "overhead_map_level", "none") != "none"
            and getattr(self, "_omap", None) is not None
            and point in self.overhead_map_cfg.get("attach_to", ())
        )

    def _omap_observe(self, observations, world_graph, agent_uid: int) -> None:
        """Build the map on the first call, then track the robot pose (and depth for the explored level)."""
        if getattr(self, "overhead_map_level", "none") == "none" or self._omap_failed:
            return
        try:
            pose = omap_lib.read_robot_pose(self.env_interface, agent_uid)
            if self._omap is None:
                t0 = time.perf_counter()
                static = omap_lib.build_static_map(self.env_interface, world_graph, agent_uid, self.overhead_map_cfg)
                self._omap = omap_lib.OverheadMap(static, self.overhead_map_level, self.overhead_map_cfg)
                try:
                    start_room = world_graph.get_spot_robot().properties.get("initial_room")
                except Exception:
                    start_room = None
                if start_room not in static.rooms:
                    start_room = None
                self._omap.set_start(*pose, room=start_room)
                self._log_event({
                    "event": "overhead_map_built", "level": self.overhead_map_level,
                    "format": self.overhead_map_format, "rooms": sorted(static.rooms),
                    "furniture": len(static.furniture),
                    "labelled_furniture": sum(f.labelled for f in static.furniture.values()),
                    "grid_shape": list(static.nav.shape), "start_room": self._omap.start_room,
                    "build_s": round(time.perf_counter() - t0, 4),
                })
            else:
                self._omap.observe_pose(*pose)
            if self._omap.wants_depth():
                frame = omap_lib.read_jaw_depth(self.env_interface, observations, agent_uid)
                if frame is not None:
                    self._omap.observe_depth(*frame)
        except Exception as exc:
            import traceback

            self._omap_failed = True
            self._omap = None
            self._log_event({"event": "overhead_map_error", "error": repr(exc),
                             "traceback": traceback.format_exc()})
            self._vprint(f"  [WARN] overhead map disabled for this episode: {exc}", "red")

    def _omap_render(self, world_graph, tag: str):
        """Render + save the map; returns (data URL, caption, relative png path)."""
        img = self._omap.render(world_graph)
        text = self._omap.to_text(world_graph)
        out_dir = os.path.join(self._get_log_dir(), "overhead_map")
        png, _txt = omap_lib.save_map(img, text, out_dir, self._omap_index, tag)
        self._omap_index += 1
        rel = os.path.relpath(png, self._get_log_dir())
        self._omap_last_rel = rel
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self._omap.summary(), f, indent=2)
        self._log_event({"event": "overhead_map", "kind": tag, "path": rel, **self._omap.summary()})
        return pil_image_to_data_url(img), self._omap.image_caption(), rel

    def _omap_last_paths(self):
        rel = getattr(self, "_omap_last_rel", None)
        return [rel] if rel else []

    def _extra_vlm_images(self, world_graph, kind: str):
        if not self._omap_attached(kind) or self.overhead_map_format not in ("image", "both"):
            return [], []
        url, caption, rel = self._omap_render(world_graph, kind)
        if kind == "turn1":
            self._last_vlm_image_paths = list(getattr(self, "_last_vlm_image_paths", []) or []) + [rel]
        return [url], [caption]

    def _omap_explorer_parts(self):
        """Prompt parts appended to every ReAct explorer call (list of (type, value)), or None."""
        if not self._omap_attached("explorer") or self._world_graph is None:
            return None
        parts = []
        if self.overhead_map_format in ("text", "both"):
            parts.append(("text", "\n" + self._omap.to_text(self._world_graph) + "\n"))
        if self.overhead_map_format in ("image", "both"):
            url, caption, _rel = self._omap_render(self._world_graph, "explorer")
            parts += [("text", caption), ("image_high_detail", url)]
        return parts

    # ------------------------------------------------------------------
    # Placement oracle (privileged)
    # ------------------------------------------------------------------

    def _get_placement_goals(self):
        if self._placement_goals is None:
            self._placement_goals = placement_oracle.placement_goals(self.env_interface)
        return self._placement_goals

    def _generate_subgoals(self, instruction, world_graph, observations, history_str="", **kwargs):
        if self._object_centric_on():
            return self._object_centric_generate(instruction, world_graph, observations, history_str, kwargs)
        return self._rail_generate_subgoals(instruction, world_graph, observations, history_str=history_str, **kwargs)

    # ------------------------------------------------------------------
    # subgoal_mode=object_centric: abstract object tasks, grounded and executed one at a time
    # ------------------------------------------------------------------

    def _object_centric_on(self) -> bool:
        return getattr(self, "subgoal_mode", "flat") == "object_centric"

    def _current_task(self):
        return next((t for t in (getattr(self, "_tasks", None) or []) if t.status == "active"), None)

    def _start_next_task(self):
        task = next((t for t in self._tasks if t.status == "pending"), None)
        if task is not None:
            task.status = "active"
            self._log_event({"event": "object_task_start", "task": task.as_dict()})
            self._trace_append(f"Object task {task.idx + 1}/{len(self._tasks)} started: {task.describe()}")
            self._vprint_header(f"OBJECT TASK {task.idx + 1}/{len(self._tasks)}: {task.describe()}", "cyan")
        return task

    def _end_task(self, task, status: str, note: str = "") -> None:
        task.status, task.note = status, note
        self._log_event({"event": f"object_task_{'done' if status == 'done' else 'failed'}", "task": task.as_dict()})
        line = f"Object task {task.idx + 1} {'done' if status == 'done' else 'skipped'}: {task.describe()} ({note})"
        self.history.append(line)
        self._trace_append(line)
        self._vprint(f"  [object task] {line}", "cyan")

    def _oc_vlm_call(self, kind: str, prompt: str, observations, world_graph):
        """One VLM call outside Turn 1/2 (decompose / review); counts toward max_vlm_cycles. None = over budget."""
        self._vlm_cycles = getattr(self, "_vlm_cycles", 0) + 1
        if self._vlm_cycles > getattr(self, "max_vlm_cycles", 20):
            self._finish_planning("budget_exhausted", "VLM cycle limit reached")
            return None, 0
        urls = []
        if self.use_images:
            fresh = self._observe_failure_state(observations, reason=kind)
            urls = list(self._get_annotated_vlm_image_urls(fresh, world_graph, annotate=False) or [])
            if urls:
                prompt = "The attached image is the robot's current third-person view.\n\n" + prompt
        self._vprint_header(kind.replace("_", " ").upper(), "magenta")
        self.vlm.new_session()
        self._append_vlm_prompt(prompt_kind=kind, prompt_text=prompt, image_count=len(urls))
        response = self._ask_vlm(
            prompt,
            image_data_urls=urls or None,
            max_completion_tokens=self.vlm_max_tokens,
            temperature=self.vlm_temperature,
            reasoning_effort=self.vlm_reasoning_effort,
        )
        return response, len(urls)

    def _oc_scene(self, world_graph) -> str:
        # Always include the start room, without consuming the one-time line the Turn-1 prompt uses.
        agent_uid = self._agents[0].uid if self._agents else 0
        return get_world_descr(world_graph, agent_uid=agent_uid, include_room_name=True, add_state_info=True,
                               centralized=False, include_initial_agent_location=True)

    def _decompose_tasks(self, instruction, world_graph, observations) -> bool:
        """Stage 1. False if the episode ended (terminal decision or budget)."""
        prompt = oc.build_decompose_prompt(instruction, self._oc_scene(world_graph), self.history)
        response, n_images = self._oc_vlm_call("object_task_decompose", prompt, observations, world_graph)
        if self.is_done:
            return False
        tasks, decision, reason, parsed = oc.parse_task_list(response)
        if not decision and not tasks:
            tasks = [oc.fallback_task(instruction)]
        self._tasks = tasks
        self._log_event({"event": "object_tasks_defined", "tasks": [t.as_dict() for t in tasks],
                         "decision": decision, "reason": reason, "parsed": parsed, "image_count": n_images,
                         "prompt": prompt, "response": response,
                         "api_response": self.vlm.get_last_response_metadata()})
        self._trace_append("Object tasks:\n" + (oc.format_task_list(tasks) or f"(none; decision: {decision})"))
        if decision:
            self._finish_planning(decision, reason)
            return False
        return True

    def _review_tasks(self, instruction, world_graph, observations, history_str: str, eop_driven: bool) -> bool:
        """All object tasks are closed: True if the review added tasks, False if the episode is finishing."""
        if not eop_driven:
            self._task_reviews += 1
        if self._task_reviews > self.object_task_max_reviews:
            self._log_event({"event": "object_task_review", "skipped": "review limit reached",
                             "tasks": [t.as_dict() for t in self._tasks]})
            self._oc_final = True
            self._finish_planning("plan_completed", "All object tasks are closed (object-task review limit reached).")
            return False
        prompt = oc.build_review_prompt(instruction, self._tasks, self._oc_scene(world_graph), self.history,
                                        extra=history_str)
        response, n_images = self._oc_vlm_call("object_task_review", prompt, observations, world_graph)
        if self.is_done:
            return False
        tasks, decision, reason, parsed = oc.parse_task_list(response, start_idx=len(self._tasks))
        self._log_event({"event": "object_task_review", "review": self._task_reviews, "decision": decision,
                         "reason": reason, "new_tasks": [t.as_dict() for t in tasks], "parsed": parsed,
                         "image_count": n_images, "prompt": prompt, "response": response,
                         "api_response": self.vlm.get_last_response_metadata()})
        if tasks:
            self._tasks.extend(tasks)
            self._trace_append("Object-task review added:\n" + oc.format_task_list(tasks))
            return True
        self._trace_append(f"Object-task review: {decision or 'unparsable, treated as complete'}. {reason}")
        if decision == "unsolvable":
            self._finish_planning("unsolvable", reason)
            return False
        self._oc_final = True
        self._finish_planning("plan_completed", reason or "Object-task review: every object task is done.")
        return False

    def _object_centric_generate(self, instruction, world_graph, observations, history_str, kwargs):
        """Every replan point lands here: ground the current object task (or the next one, or review)."""
        self._world_graph = world_graph
        if self._tasks is None and not self._decompose_tasks(instruction, world_graph, observations):
            return None
        cur = self._current_task()
        if cur is not None and kwargs.get("refresh_failure_observation"):
            cur.regrounds += 1
            if cur.regrounds > self.object_task_max_regrounds:
                self._end_task(cur, "failed", f"still failing after {cur.regrounds - 1} re-groundings")
        eop_driven = kwargs.get("observation_reason") == "plan_finished"
        result = None
        for _ in range(4 * (len(self._tasks) + self.object_task_max_reviews) + 8):
            if self.is_done or getattr(self, "_mode", "") == "explore" or getattr(self, "_end_replan_pending", False):
                return result
            cur = self._current_task() or self._start_next_task()
            if cur is None:
                if not self._review_tasks(instruction, world_graph, observations, history_str, eop_driven):
                    return None
                eop_driven = False
                continue
            goal = oc.task_goal_text(instruction, self._tasks, cur)
            result = self._rail_generate_subgoals(goal, world_graph, observations, history_str=history_str, **kwargs)
            if cur.status == "active":
                return result
            # The Turn-1 decision closed this task without subgoals: ground the next one in this same call.
            kwargs = {**kwargs, "append": True, "refresh_failure_observation": False, "after_explore": False,
                      "explored_room": None, "observation_reason": "object_task_done"}
            history_str = f"Previously achieved subgoals: {self._already_succeeded_subgoals}.\n"
        return result

    def _advance_object_task(self, instruction, observations, world_graph) -> None:
        """The current object task's subgoals finished: ground the next task (or review)."""
        fresh = self._observe_failure_state(observations, reason="object_task_done")
        self._reprompt_round += 1
        self._log_event({"event": "reprompt_started", "reason": "object_task_done", "branch": self.branch_idx,
                         "subgoal_idx": self.subgoal_idx, "reprompt_round": self._reprompt_round})
        self.current_plan = []
        self.current_action_idx = 0
        self.last_high_level_actions = {}
        self._subgoal_retry_count = 0
        history = (
            "Replanning reason: the previous object task's subgoals finished.\n"
            f"Previously achieved subgoals: {self._already_succeeded_subgoals}.\n"
        )
        self._generate_subgoals(instruction, world_graph, fresh, history_str=history, append=True,
                                observation_reason="object_task_done")

    def _rail_generate_subgoals(self, instruction, world_graph, observations, history_str="", **kwargs):
        if getattr(self, "two_phase", False) and self._mode != "explore":
            # Every replan point goes through here: let the selector choose explore or interact first.
            self._world_graph = world_graph
            if self._choose_mode(instruction, world_graph, observations, kwargs) == "explore":
                return None
            if self._explore_phases >= self.max_explore_phases:
                history_str = (history_str or "") + (
                    "\nNo exploration phases are left: plan with the known objects, or return the unsolvable decision.\n"
                )
        if getattr(self, "placement_oracle_mode", "none") in ("hint", "full"):
            known = {o.name for o in world_graph.get_all_objects()}
            hint = placement_oracle.format_hint(self._get_placement_goals(), known)
            if hint:
                history_str = (history_str or "") + hint
                self._trace_append(hint.strip())
            self._log_event({"event": "placement_oracle_hint", "hint": hint})
        return super()._generate_subgoals(
            instruction, world_graph, observations, history_str=history_str, **kwargs
        )

    def _place_tool(self, name: str, args):
        obj = self._restore_name(args[1])
        furn = self._restore_name(args[2])
        relation = "within" if name == "place_in" else "on"
        if getattr(self, "placement_oracle_mode", "none") not in ("place", "full"):
            return "Place", f"{obj}, {relation}, {furn}, None, None"
        agent_uid = self._agents[0].uid if self._agents else 0
        state = build_scene_state(self.env_interface.world_graph[agent_uid], agent_uid)
        on_furn = [o for o, (f, _) in state.placement.items() if f == furn and o != obj]
        goal = self._get_placement_goals().get(obj)
        partner = placement_oracle.place_constraint(goal, on_furn, self._placed_order)
        self._log_event({
            "event": "placement_oracle_place",
            "object": obj,
            "furniture": furn,
            "next_to": partner,
            "goal_next_to": sorted(goal.next_to) if goal else [],
        })
        self._placed_order = [o for o in self._placed_order if o != obj] + [obj]
        if partner:
            return "Place", f"{obj}, {relation}, {furn}, next_to, {partner}"
        return "Place", f"{obj}, {relation}, {furn}, None, None"

    def _log_subgoal_state(self, subgoal: str, state_facts, outcome: str) -> None:
        dump_dir = os.path.join(self._get_log_dir(), "subgoal_states")
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"b{self.branch_idx:02d}_s{self.subgoal_idx:02d}.txt")
        with open(path, "w") as f:
            f.write(f"branch: {self.branch_idx}, subgoal_idx: {self.subgoal_idx}\n")
            f.write(f"subgoal: {subgoal}\n")
            f.write("state:\n")
            for fact in state_facts:
                f.write(f"  {fact}\n")
            f.write(f"outcome: {outcome}\n")
