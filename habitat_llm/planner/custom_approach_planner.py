import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig  # pyright: ignore[reportMissingImports]

from habitat_llm.custom_approach import render_custom_subgoal_log_html
from habitat_llm.custom_approach.prompt_builders import (
    build_english_subgoal_prompt as custom_build_english_subgoal_prompt,
    build_failure_history as custom_build_failure_history,
    build_predicate_translation_prompt as custom_build_predicate_translation_prompt,
)
from habitat_llm.custom_approach.prompt_templates_vlm_tamp_react import (
    SUBGOAL_EXPLORATION_PROMPT,
    SUBGOAL_TO_VLM_ACTIONS_PROMPT,
)
import habitat_llm.planner.vlm_tamp_pddl_planner as vlm_tamp_pddl_module
from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.vlm_tamp.render_vlm_prompts_html import render_vlm_prompts_chat_html
from ..custom_approach.custom_constants import (
    CUSTOM_ACTION_NAMES,
    CUSTOM_ACTIONS,
    CUSTOM_EXPLORATION_ACTIONS,
    OBJECT_ALIAS_HINTS,
)

HighLevelAction = Tuple[str, Optional[str], Optional[str]]


class CustomApproachPlanner(VlmTampPddlPlanner):
    """Custom-approach planner with per-subgoal retry/skip controls."""

    def __init__(self, plan_config: DictConfig, env_interface):
        super().__init__(plan_config, env_interface)
        backend = str(self.plan_config.get("subgoal_action_backend", "pddlstream")).lower()
        if backend not in ("pddlstream", "vlm"):
            self._vprint(
                f"[WARN] subgoal_action_backend={backend!r} is unknown; "
                "falling back to pddlstream.",
                "yellow",
            )
            backend = "pddlstream"
        self.subgoal_action_backend = backend
        self.max_pddl_replans_per_subgoal = max(
            0,
            int(
                self.plan_config.get(
                    "max_pddl_replans_per_subgoal",
                    self.ntamp_max_pddl_replan_retries,
                )
            ),
        )
        self.ntamp_max_pddl_replan_retries = self.max_pddl_replans_per_subgoal
        self.skip_failed_subgoals = bool(
            self.plan_config.get("skip_failed_subgoals", True)
        )
        self.print_custom_pipeline_details = bool(
            self.plan_config.get("print_custom_pipeline_details", False)
        )
        action_executor = str(
            self.plan_config.get("custom_action_executor", "pddlstream")
        ).lower()
        if action_executor not in ("pddlstream", "vlm"):
            self._vprint(
                f"[WARN] custom_action_executor={action_executor!r} is unknown; "
                "falling back to pddlstream.",
                "yellow",
            )
            action_executor = "pddlstream"
        self.custom_action_executor = action_executor
        self.max_vlm_action_reprompts_per_subgoal = max(
            0,
            int(
                self.plan_config.get(
                    "max_vlm_action_reprompts_per_subgoal",
                    self.max_pddl_replans_per_subgoal,
                )
            ),
        )

    def reset(self):
        super().reset()
        self._reset_direct_vlm_state()

    def _reset_direct_vlm_state(self) -> None:
        self._direct_needed_objects: List[str] = []
        self._direct_action_history: List[str] = []
        self._direct_last_action_error: str = ""
        self._direct_plan_phase: Optional[str] = None
        self._direct_pddl_subgoals: List[str] = []
        self._direct_pddl_subgoal_idx: int = 0
        self._direct_action_reasons: List[str] = []
        self._direct_completed_object_names: set = set()
        self._direct_action_reprompt_count: int = 0
        self._direct_logged_started_subgoals: set = set()

    def _generate_subgoals(self, *args, **kwargs):
        """Use custom prompt builders without modifying the base VLM-TAMP planner."""
        return self._with_custom_prompt_builders(
            super()._generate_subgoals,
            *args,
            **kwargs,
        )

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graphs: Dict[int, Any],
    ):
        if self.subgoal_action_backend == "vlm":
            return self._get_next_direct_vlm_action(
                instruction, observations, world_graphs
            )
        return self._with_custom_prompt_builders(
            super().get_next_action,
            instruction,
            observations,
            world_graphs,
        )

    def _with_custom_prompt_builders(self, callback, *args, **kwargs):
        old_english_builder = vlm_tamp_pddl_module.build_english_subgoal_prompt
        old_predicate_builder = vlm_tamp_pddl_module.build_predicate_translation_prompt
        old_failure_builder = vlm_tamp_pddl_module.build_failure_history
        vlm_tamp_pddl_module.build_english_subgoal_prompt = (
            custom_build_english_subgoal_prompt
        )
        vlm_tamp_pddl_module.build_predicate_translation_prompt = (
            custom_build_predicate_translation_prompt
        )
        vlm_tamp_pddl_module.build_failure_history = custom_build_failure_history
        try:
            return callback(*args, **kwargs)
        finally:
            vlm_tamp_pddl_module.build_english_subgoal_prompt = old_english_builder
            vlm_tamp_pddl_module.build_predicate_translation_prompt = (
                old_predicate_builder
            )
            vlm_tamp_pddl_module.build_failure_history = old_failure_builder

    def _parse_python_list(self, raw_output: Any) -> List[str]:
        if isinstance(raw_output, list):
            return [str(item).strip() for item in raw_output if str(item).strip()]

        text = str(raw_output or "").strip()
        if not text:
            return []
        if "```" in text:
            text = re.sub(r"```[a-zA-Z0-9]*", "", text).replace("```", "").strip()

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

        if "[" in text and "]" in text:
            content = text[text.find("[") + 1 : text.rfind("]")]
            try:
                parsed = ast.literal_eval("[" + content + "]")
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return []

    def _is_action_like(self, item: str) -> bool:
        text = item.strip()
        return any(
            text == action_name
            or text.startswith(f"{action_name} ")
            or text.startswith(f"{action_name}[")
            or text.startswith(f"{action_name}(")
            for action_name in CUSTOM_ACTION_NAMES
        )

    def _build_object_alias_hint(self, subgoal: str) -> str:
        subgoal_lower = subgoal.lower()
        matches = [
            f"- {term}: use {', '.join(aliases)} if present in the scene graph"
            for term, aliases in OBJECT_ALIAS_HINTS.items()
            if term in subgoal_lower
        ]
        if not matches:
            return ""
        return (
            "\n\n# Object Alias Grounding\n"
            "Use these scene-graph names when the subgoal uses a generic term:\n"
            + "\n".join(matches)
            + "\n"
        )

    def _parse_direct_action(self, action_text: str) -> Optional[HighLevelAction]:
        text = action_text.strip().strip("\"'")
        if not text:
            return None

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[\[(](.*)[\])]$", text)
        if match:
            action_name = match.group(1).strip()
            arg_text = match.group(2).strip()
        else:
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                return None
            action_name, arg_text = parts[0].strip(), parts[1].strip()

        canonical = next(
            (
                name
                for name in CUSTOM_ACTION_NAMES
                if name.lower() == action_name.lower()
            ),
            None,
        )
        if canonical is None:
            return None

        args = [arg.strip() for arg in arg_text.split(",") if arg.strip()]
        if args and args[0].isdigit():
            args = args[1:]
        if args:
            head, sep, tail = args[0].partition(" ")
            if sep and head.isdigit() and tail.strip():
                args[0] = tail.strip()
        if canonical == "Explore" and args and args[0].lower() == "none":
            return ("Explore", None, "exploration_required")
        if not args:
            return (canonical, None, "Missing action argument.")
        if canonical == "Place":
            return (canonical, ", ".join(args), None)
        return (canonical, args[0], None)

    def _parse_direct_actions(self, raw_actions: List[str]) -> List[HighLevelAction]:
        actions: List[HighLevelAction] = []
        for raw_action in raw_actions:
            parsed = self._parse_direct_action(raw_action)
            if parsed is not None:
                actions.append(parsed)
        return actions

    def _split_exploration_output(
        self,
        exploration_output: List[str],
    ) -> Tuple[List[str], List[str]]:
        cleaned_items: List[str] = []
        reasons: List[str] = []
        inline_reason_pattern = re.compile(
            r"^(.+?[\]\)])\s+(?:because|since|so that)\s+(.+)$",
            re.IGNORECASE,
        )

        for item in exploration_output:
            text = str(item).strip()
            if not text:
                continue
            if text.lower().startswith("reason:"):
                reason = text.split(":", 1)[1].strip()
                if reason:
                    reasons.append(reason)
                continue

            match = inline_reason_pattern.match(text)
            if match:
                cleaned_items.append(match.group(1).strip())
                reason = match.group(2).strip()
                if reason:
                    reasons.append(reason)
                continue

            cleaned_items.append(text)

        return cleaned_items, reasons

    def _format_direct_history(self, history: List[str]) -> str:
        if not history:
            return "(none)"
        lines = []
        for idx, item in enumerate(history, start=1):
            lines.append(f"{idx}. {item}")
        return "\n".join(lines)

    def _latest_direct_action_result(self) -> str:
        if not self._direct_action_history:
            return "(none)"
        return self._direct_action_history[-1]

    def _format_direct_completed_objects(self) -> str:
        if not self._direct_completed_object_names:
            return "(none)"
        return ", ".join(sorted(self._direct_completed_object_names))

    def _current_direct_movable_names(self) -> set:
        return set(getattr(self, "_objects_by_type", {}).get("movable", []))

    def _is_repeated_object_subgoal(self, subgoal: str) -> bool:
        text = subgoal.lower()
        return any(token in text for token in ("another", "other", "next"))

    def _filter_completed_direct_objects(
        self,
        candidates: List[str],
        subgoal: str,
    ) -> Tuple[List[str], List[str]]:
        if not self._direct_completed_object_names:
            return candidates, []
        if not self._is_repeated_object_subgoal(subgoal):
            return candidates, []
        movable_names = self._current_direct_movable_names()
        filtered: List[str] = []
        rejected: List[str] = []
        for candidate in candidates:
            if (
                candidate in self._direct_completed_object_names
                and candidate in movable_names
            ):
                rejected.append(candidate)
            else:
                filtered.append(candidate)
        return filtered, rejected

    def _predicates_with_completed_direct_objects(
        self,
        predicates: List[str],
        subgoal: str,
    ) -> List[str]:
        if not self._direct_completed_object_names:
            return []
        if not self._is_repeated_object_subgoal(subgoal):
            return []
        rejected: List[str] = []
        for predicate in predicates:
            for obj_name in self._direct_completed_object_names:
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(obj_name)}(?![A-Za-z0-9_])", predicate):
                    rejected.append(predicate)
                    break
        return rejected

    def _direct_predicate_fallback_navigation(
        self,
        raw_predicates_text: str,
        subgoal: str,
        world_graph,
        agent_uid: int,
        parsed_predicates: List[str],
    ) -> List[str]:
        """When predicate translation yields [] repeatedly, synthesize viable navigation predicate(s).

        The VLM is instructed not to emit [] for approach/navigation wording when names are known,
        but this fallback prevents an exploration/skip/ping-pong loop against empty lists.
        """
        if parsed_predicates:
            return []

        movables = set(self._objects_by_type.get("movable", []))
        rooms = sorted(room.name for room in world_graph.get_all_rooms())
        room_by_lower = {r.lower(): r for r in rooms}

        def _room_named_in_subgoal(line: str, room_name: str) -> bool:
            base = room_name.lower()
            spaced = base.replace("_", " ")
            escaped_u = re.escape(base)
            escaped_s = re.escape(spaced)
            if re.search(rf"(?<!\w){escaped_u}(?!\w)", line):
                return True
            if "_" in base and re.search(rf"(?<!\w){escaped_s}(?!\w)", line):
                return True
            return False

        needed = [name for name in self._direct_needed_objects if name in movables]

        text = subgoal.lower()
        is_nav_like = (
            "navigate" in text or "navigation" in text or "approach" in text
        )
        if not is_nav_like:
            for r in sorted(rooms, key=len, reverse=True):
                if _room_named_in_subgoal(text, r):
                    is_nav_like = True
                    break

        if needed and is_nav_like:
            staged = sorted(needed, key=len)
            picked = [f"picked({staged[0]})"]
            self._vprint(
                "  [FALLBACK] Predicate translation returned no predicates; "
                f"emitting staging navigation predicates: {picked}",
                "yellow",
            )
            self._trace_append(f"Fallback PDDL predicates: {picked}")
            self._direct_last_action_error = (
                "Predicate translator returned []; "
                f"applied navigation fallback ({', '.join(picked)}) from raw "
                f"{raw_predicates_text!r}"
            )
            self._log_event(
                {
                    "event": "custom_subgoal_to_pddl_goals_fallback",
                    "branch": self.branch_idx,
                    "subgoal_idx": self.subgoal_idx,
                    "subgoal": subgoal,
                    "needed_objects": list(self._direct_needed_objects),
                    "fallback_predicates": picked,
                    "raw_response": raw_predicates_text,
                }
            )
            return picked

        if is_nav_like and rooms:
            room_token = ""
            if needed:
                for obj_name in needed:
                    node = None
                    try:
                        node = world_graph.get_node_from_name(obj_name)
                    except Exception:
                        node = None
                    if node is None:
                        continue
                    try:
                        r_obj = world_graph.get_room_for_entity(node)
                    except Exception:
                        r_obj = None
                    if r_obj and r_obj.name in rooms:
                        room_token = r_obj.name.lower()
                        break
            if not room_token:
                for r in sorted(rooms, key=len, reverse=True):
                    if _room_named_in_subgoal(text, r):
                        room_token = r.lower()
                        break
            resolved = room_by_lower.get(room_token, "")
            if resolved in rooms:
                stub = f"stub_nav_{agent_uid}_{resolved}"
                at_pred = f"at({stub},{resolved})"
                self._vprint(
                    "  [FALLBACK] Predicate translation returned no predicates; "
                    f"emitting room at-milestone: {at_pred}",
                    "yellow",
                )
                self._trace_append(f"Fallback PDDL predicates: [{at_pred}]")
                self._direct_last_action_error = (
                    "Predicate translator returned []; "
                    f"applied at() room fallback from raw {raw_predicates_text!r}"
                )
                self._log_event(
                    {
                        "event": "custom_subgoal_to_pddl_goals_fallback",
                        "branch": self.branch_idx,
                        "subgoal_idx": self.subgoal_idx,
                        "subgoal": subgoal,
                        "needed_objects": list(self._direct_needed_objects),
                        "fallback_predicates": [at_pred],
                        "raw_response": raw_predicates_text,
                    }
                )
                return [at_pred]

        return []

    def _resolve_direct_explore_room(self, room_name: str, world_graph) -> str:
        room_name = str(room_name or "").strip()
        if not room_name:
            return room_name

        rooms = sorted(room.name for room in world_graph.get_all_rooms())
        if room_name in rooms:
            return room_name

        normalized = room_name.lower().replace(" ", "_")
        for room in rooms:
            if room.lower() == normalized:
                return room

        preferred = f"{normalized}_1"
        for room in rooms:
            if room.lower() == preferred:
                self._vprint(
                    f"  [room resolved] Explore[{room_name}] -> Explore[{room}]",
                    "yellow",
                )
                return room

        prefix_matches = [
            room for room in rooms if room.lower().startswith(f"{normalized}_")
        ]
        if len(prefix_matches) == 1:
            self._vprint(
                f"  [room resolved] Explore[{room_name}] -> Explore[{prefix_matches[0]}]",
                "yellow",
            )
            return prefix_matches[0]

        return room_name

    def _resolve_direct_navigation_target(self, target_name: str, world_graph) -> str:
        target_name = str(target_name or "").strip()
        if not target_name.startswith("rec_"):
            return target_name

        match = re.match(r"^rec_(.+)_\d+$", target_name)
        if not match:
            return target_name

        parent_name = match.group(1)
        if world_graph.has_node(parent_name):
            self._vprint(
                f"  [nav target resolved] Navigate[{target_name}] -> Navigate[{parent_name}]",
                "yellow",
            )
            return parent_name

        return target_name

    def _format_direct_visible_furniture(
        self,
        world_graph,
        visible_entity_names,
    ) -> str:
        if visible_entity_names is None:
            return "(camera-visible furniture unavailable)"

        lines: List[str] = []
        for furn in sorted(world_graph.get_all_furnitures(), key=lambda item: item.name):
            if furn.name not in visible_entity_names:
                continue
            suffix = ""
            if furn.properties.get("is_articulated", False):
                is_open = furn.properties.get("is_open", False)
                state = "open" if is_open else "closed"
                suffix = f" ({state}, articulated)"
            lines.append(f"- {furn.name}{suffix}")

        if not lines:
            return "(no furniture currently visible)"
        return "\n".join(lines)

    def _direct_scene_context(self, observations: Dict[str, Any], world_graph, agent_uid: int):
        visible_entity_names = self._extract_visible_entity_names(observations, world_graph)
        self._objects_by_type = self._build_objects_by_type(
            world_graph,
            visible_object_names=None,
        )
        try:
            graph_memory = world_graph.to_string()
        except Exception:
            graph_memory = "(hierarchical graph unavailable)"
        state_memory = self._build_scene_description(
            world_graph,
            agent_uid,
            visible_entity_names=None,
        )
        visible_furniture = self._format_direct_visible_furniture(
            world_graph,
            visible_entity_names,
        )
        scene_desc = (
            "Robot scene graph memory:\n"
            f"{graph_memory.strip() or '(empty)'}\n\n"
            "Robot current position scene state memory (current camera observations):\n"
            f"{state_memory}\n\n"
            "Observed furniture in current camera view:\n"
            f"{visible_furniture}"
        )
        return scene_desc

    def _direct_ask_vlm(
        self,
        prompt_kind: str,
        prompt_text: str,
        *,
        image_count: int = 0,
        image_urls: Optional[List[str]] = None,
    ) -> str:
        self.vlm.new_session()
        self._append_vlm_prompt(
            prompt_kind=prompt_kind,
            prompt_text=prompt_text,
            image_count=image_count,
        )
        if self.print_custom_pipeline_details:
            self._vprint("  [full prompt]", "gray")
            for line in prompt_text.splitlines():
                self._vprint(f"    {line}", "gray")
        response = self.vlm.ask(
            prompt_text,
            image_data_urls=image_urls if image_urls else None,
            max_completion_tokens=self.vlm_max_tokens,
            temperature=self.vlm_temperature,
        )
        self._append_custom_vlm_response(prompt_kind, response)
        if self.print_custom_pipeline_details:
            self._vprint("  [raw response]", "magenta")
            raw_response = str(response or "").strip()
            if raw_response:
                for line in raw_response.splitlines():
                    self._vprint(f"    {line}", "magenta")
            else:
                self._vprint("    (empty)", "red")
        self._log_event(
            {
                "event": prompt_kind,
                "reprompt_round": self._reprompt_round,
                "branch": self.branch_idx,
                "subgoal_idx": self.subgoal_idx,
                "prompt": prompt_text,
                "response": response,
                "image_count": image_count,
            }
        )
        self._refresh_custom_vlm_prompts_html()
        return str(response or "")

    def _append_custom_vlm_response(self, prompt_kind: str, response: Any) -> None:
        log_dir = self._get_log_dir()
        prompts_path = f"{log_dir}/vlm_prompts.txt"
        block = (
            "VLM_RESPONSE\n"
            f"prompt_kind: {prompt_kind}\n"
            "----------------------------------------------------------------\n"
            f"{response or ''}\n\n"
        )
        with open(prompts_path, "a", encoding="utf-8") as f:
            f.write(block)

    def _refresh_custom_vlm_prompts_html(self) -> None:
        if not getattr(self, "write_log_html", False):
            return
        log_dir = self._get_log_dir()
        try:
            render_vlm_prompts_chat_html(log_dir)
        except Exception as exc:
            self._vprint(f"  [WARN] vlm_prompts.html refresh failed: {exc}", "red")
        self._refresh_custom_subgoal_log_html()

    def _refresh_custom_subgoal_log_html(self) -> None:
        if not getattr(self, "write_log_html", False):
            return
        try:
            render_custom_subgoal_log_html(self._get_log_dir())
        except Exception as exc:
            self._vprint(f"  [WARN] custom index.html refresh failed: {exc}", "red")

    def _log_plan_tree(self):
        super()._log_plan_tree()
        self._refresh_custom_subgoal_log_html()

    def _vprint_list(
        self,
        title: str,
        items: List[Any],
        *,
        color: str = "white",
    ) -> None:
        if not self.print_custom_pipeline_details:
            return
        self._vprint(f"  [{title}] ({len(items)})", color)
        if not items:
            self._vprint("    (none)", "gray")
            return
        for idx, item in enumerate(items, start=1):
            self._vprint(f"    {idx:02d}. {item}", color)

    def _vprint_custom_action_set(self, action_set: Optional[str] = None) -> None:
        if not self.print_custom_pipeline_details:
            return
        self._vprint("  [available action set]", "cyan")
        for line in (action_set or CUSTOM_ACTIONS).splitlines():
            self._vprint(f"    {line}", "cyan")

    def _direct_generate_subgoals(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph,
        agent_uid: int,
    ) -> None:
        scene_desc = self._direct_scene_context(observations, world_graph, agent_uid)
        prompt = custom_build_english_subgoal_prompt(
            goal=instruction,
            objects_by_type=self._objects_by_type,
            scene_description=scene_desc,
            history=self._build_failure_history(
                actions=self._direct_action_history,
                failure=self._direct_last_action_error or None,
                already_succeeded=self._already_succeeded_subgoals,
            ),
        )
        self._vprint_header("CUSTOM VLM: Task To Object Subgoals", "magenta")
        if self.print_custom_pipeline_details:
            self._vprint(f"  [task] {instruction}", "white")
            self._vprint_custom_action_set()
        raw_subgoals = self._direct_ask_vlm("custom_task_to_subgoals", prompt)
        subgoals = self._parse_python_list(raw_subgoals)
        self._vprint_list("parsed object-scoped subgoals", subgoals, color="yellow")

        self.branches = [subgoals] if subgoals else []
        self._branch_origin_round = [self._reprompt_round] if subgoals else []
        self.branch_idx = 0
        self.subgoal_idx = 0
        self.current_plan = []
        self.current_action_idx = 0
        self._log_event(
            {
                "event": "custom_task_to_subgoals_parsed",
                "reprompt_round": self._reprompt_round,
                "subgoals": subgoals,
            }
        )
        if subgoals:
            self._trace_append(
                "Custom VLM subgoals:\n"
                + "\n".join(f"  {idx}: {sg}" for idx, sg in enumerate(subgoals))
            )
            self._log_plan_tree()

    def _direct_current_subgoal(self) -> Optional[str]:
        subgoals = self._active_subgoals
        if self.subgoal_idx < len(subgoals):
            return subgoals[self.subgoal_idx]
        return None

    def _direct_reset_subgoal_state(self) -> None:
        self._direct_needed_objects = []
        self._direct_last_action_error = ""
        self._direct_plan_phase = None
        self._direct_pddl_subgoals = []
        self._direct_pddl_subgoal_idx = 0
        self._direct_action_reasons = []
        self._direct_action_reprompt_count = 0
        self.current_plan = []
        self.current_action_idx = 0
        self.last_high_level_actions = {}

    def _direct_skip_current_subgoal(self) -> None:
        subgoal = self._direct_current_subgoal()
        if subgoal is None:
            return
        skip_reason = (
            self._last_failure
            or self._direct_last_action_error
            or "the custom planner could not produce or execute a valid plan"
        )
        self._vprint(
            f"  [SKIP] Direct VLM subgoal skipped because {skip_reason}: {subgoal}",
            "yellow",
        )
        self._trace_append(
            f"Direct VLM subgoal skipped because {skip_reason}: {subgoal}"
        )
        self._log_subgoal_status(
            status="failed",
            subgoal=subgoal,
            branch_idx=self.branch_idx,
            subgoal_idx=self.subgoal_idx,
            failure_type="custom_vlm_subgoal_failed",
            failure_msg=skip_reason,
        )
        self._emit_subgoal_execution("failed", subgoal)
        self.subgoal_idx += 1
        self._direct_reset_subgoal_state()
        self._log_plan_tree()

    def _direct_mark_current_subgoal_solved(self) -> None:
        subgoal = self._direct_current_subgoal()
        if subgoal is None:
            return
        movable_names = self._current_direct_movable_names()
        completed_now = [
            obj_name
            for obj_name in self._direct_needed_objects
            if obj_name in movable_names
        ]
        for obj_name in completed_now:
            self._direct_completed_object_names.add(obj_name)
        if completed_now:
            self._vprint(
                "  [completed object instances] " + ", ".join(completed_now),
                "green",
            )
            self._trace_append(
                "Completed object instances: " + ", ".join(completed_now)
            )
        self._vprint_subgoal_complete_block(subgoal)
        self._trace_append(f"Direct VLM subgoal solved: {subgoal}")
        self._subgoals_completed += 1
        self._log_subgoal_status(
            status="solved",
            subgoal=subgoal,
            branch_idx=self.branch_idx,
            subgoal_idx=self.subgoal_idx,
        )
        self._emit_subgoal_execution("solved", subgoal, extra_trailing_delimiter=False)
        if subgoal not in self._already_succeeded_subgoals:
            self._already_succeeded_subgoals.append(subgoal)
        self.subgoal_idx += 1
        self._direct_reset_subgoal_state()
        self._log_plan_tree()

    def _direct_prompt_exploration(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph,
        agent_uid: int,
    ) -> bool:
        subgoal = self._direct_current_subgoal()
        if subgoal is None:
            return False

        scene_desc = self._direct_scene_context(observations, world_graph, agent_uid)
        prompt = (
            SUBGOAL_EXPLORATION_PROMPT.format(
                high_level_task=instruction,
                sub_goal=subgoal,
                actions=CUSTOM_EXPLORATION_ACTIONS,
                scene_graph=scene_desc,
                last_action_result=self._latest_direct_action_result(),
                completed_objects=self._format_direct_completed_objects(),
                exploration_history=self._format_direct_history(
                    self._direct_action_history
                ),
            )
            + self._build_object_alias_hint(subgoal)
        )
        self._vprint_header(
            f"CUSTOM VLM: Exploration  Subgoal {self.subgoal_idx}",
            "magenta",
        )
        self._vprint(f"  [current subgoal] {subgoal}", "yellow")
        self._vprint_custom_action_set(CUSTOM_EXPLORATION_ACTIONS)
        self._vprint_list(
            "action history",
            self._direct_action_history,
            color="gray",
        )
        raw_exploration = self._direct_ask_vlm(
            "custom_subgoal_exploration", prompt
        )
        exploration_output = self._parse_python_list(raw_exploration)
        exploration_items, exploration_reasons = self._split_exploration_output(
            exploration_output
        )
        self._vprint_list(
            "parsed exploration output",
            exploration_output,
            color="yellow",
        )
        self._vprint_list(
            "parsed exploration reasons",
            exploration_reasons,
            color="gray",
        )
        self._log_event(
            {
                "event": "custom_subgoal_exploration_parsed",
                "branch": self.branch_idx,
                "subgoal_idx": self.subgoal_idx,
                "subgoal": subgoal,
                "parsed": exploration_output,
                "actions_or_objects": exploration_items,
                "reasons": exploration_reasons,
            }
        )

        if exploration_items and not any(
            self._is_action_like(item) for item in exploration_items
        ):
            filtered_items, rejected_items = self._filter_completed_direct_objects(
                exploration_items,
                subgoal,
            )
            if rejected_items:
                rejection_msg = (
                    "Rejected already completed object(s) for this repeated-object "
                    f"subgoal: {', '.join(rejected_items)}"
                )
                self._direct_last_action_error = rejection_msg
                self._direct_action_history.append(
                    f"FAILED CandidateSelection[{', '.join(rejected_items)}]: {rejection_msg}"
                )
                self._trace_append(rejection_msg)
                self._vprint(f"  [REJECT] {rejection_msg}", "yellow")
                self._log_event(
                    {
                        "event": "custom_exploration_rejected_completed_objects",
                        "branch": self.branch_idx,
                        "subgoal_idx": self.subgoal_idx,
                        "subgoal": subgoal,
                        "rejected_objects": rejected_items,
                        "completed_objects": sorted(
                            self._direct_completed_object_names
                        ),
                    }
                )
            if not filtered_items:
                return False

            self._direct_needed_objects = filtered_items
            skip_msg = (
                "Explore skipped because all needed objects were found in the "
                f"current scene graph: {', '.join(self._direct_needed_objects)}"
            )
            self._vprint(f"  [SKIP] {skip_msg}", "green")
            self._trace_append(skip_msg)
            self._vprint_list(
                "needed objects found",
                self._direct_needed_objects,
                color="green",
            )
            self._trace_append(
                f"Needed objects for subgoal {self.subgoal_idx}: "
                f"{self._direct_needed_objects}"
            )
            self._log_event(
                {
                    "event": "custom_exploration_skipped",
                    "branch": self.branch_idx,
                    "subgoal_idx": self.subgoal_idx,
                    "subgoal": subgoal,
                    "reason": "needed_objects_found",
                    "needed_objects": list(self._direct_needed_objects),
                }
            )
            return True

        direct_actions = self._parse_direct_actions(exploration_items)
        if not direct_actions:
            self._last_failure = (
                f"Exploration prompt returned no executable actions for subgoal: {subgoal}"
            )
            self._direct_last_action_error = self._last_failure
            self._direct_skip_current_subgoal()
            return False

        self.current_plan = direct_actions
        self.current_action_idx = 0
        self._direct_action_reasons = exploration_reasons
        self._direct_plan_phase = "explore"
        self._vprint_list(
            "parsed exploration actions",
            direct_actions,
            color="cyan",
        )
        return True

    def _direct_prompt_pddl_predicates(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph,
        agent_uid: int,
    ) -> bool:
        subgoal = self._direct_current_subgoal()
        if subgoal is None:
            return False

        scene_desc = self._direct_scene_context(observations, world_graph, agent_uid)
        action_history = list(self._direct_action_history)
        prompt = (
            custom_build_predicate_translation_prompt(
                objects_by_type=self._objects_by_type,
                high_level_task=instruction,
                sub_goal=subgoal,
                scene_description=scene_desc,
                action_history=self._format_direct_history(action_history),
                needed_objects=self._direct_needed_objects,
                replan_needed=bool(self._direct_action_reprompt_count),
                latest_failure=self._direct_last_action_error,
            )
            + self._build_object_alias_hint(subgoal)
        )
        self._vprint_header(
            f"CUSTOM VLM: Subgoal To PDDL Goals  Subgoal {self.subgoal_idx}",
            "magenta",
        )
        self._vprint(f"  [current subgoal] {subgoal}", "yellow")
        self._vprint_list(
            "needed objects",
            self._direct_needed_objects,
            color="green",
        )
        self._vprint_list("action history", action_history, color="gray")
        raw_predicates = self._direct_ask_vlm("custom_subgoal_to_pddl_goals", prompt)
        self._vprint(f"  [VLM raw PDDL output] {raw_predicates}", "yellow")
        predicates = self._parse_python_list(raw_predicates)
        if not predicates:
            predicates = self._direct_predicate_fallback_navigation(
                raw_predicates,
                subgoal,
                world_graph,
                agent_uid,
                predicates,
            )
        self._vprint(f"  [VLM parsed PDDL predicates] {predicates}", "yellow")
        self._vprint_list("parsed PDDL predicates", predicates, color="yellow")
        self._log_event(
            {
                "event": "custom_subgoal_to_pddl_goals_parsed",
                "branch": self.branch_idx,
                "subgoal_idx": self.subgoal_idx,
                "subgoal": subgoal,
                "needed_objects": list(self._direct_needed_objects),
                "parsed": predicates,
                "reprompt_count": self._direct_action_reprompt_count,
            }
        )
        if not predicates:
            self._direct_needed_objects = []
            self._direct_last_action_error = ""
            self.current_plan = []
            self.current_action_idx = 0
            self._direct_plan_phase = None
            return False

        rejected_predicates = self._predicates_with_completed_direct_objects(
            predicates,
            subgoal,
        )
        if rejected_predicates:
            rejection_msg = (
                "Rejected PDDL predicates that reuse completed object instances "
                f"for this repeated-object subgoal: {', '.join(rejected_predicates)}"
            )
            self._direct_last_action_error = rejection_msg
            self._direct_action_history.append(
                f"FAILED PredicateSelection[{', '.join(rejected_predicates)}]: {rejection_msg}"
            )
            self._trace_append(rejection_msg)
            self._vprint(f"  [REJECT] {rejection_msg}", "yellow")
            self._log_event(
                {
                    "event": "custom_pddl_predicates_rejected_completed_objects",
                    "branch": self.branch_idx,
                    "subgoal_idx": self.subgoal_idx,
                    "subgoal": subgoal,
                    "rejected_predicates": rejected_predicates,
                    "completed_objects": sorted(self._direct_completed_object_names),
                }
            )
            self._direct_needed_objects = []
            self.current_plan = []
            self.current_action_idx = 0
            self._direct_plan_phase = None
            return False

        self._direct_pddl_subgoals = predicates
        self._direct_pddl_subgoal_idx = 0
        return True

    def _direct_handle_pddl_plan_failure(self, predicate: str, failure_msg: str) -> bool:
        subgoal = self._direct_current_subgoal() or "unknown_subgoal"
        self._last_failure = failure_msg
        self._direct_last_action_error = failure_msg
        self.history.append(failure_msg)
        self._direct_action_history.append(f"FAILED PDDL_PLAN[{predicate}]: {failure_msg}")
        self._trace_append(failure_msg)
        self._vprint(f"  [FAIL] {failure_msg}", "red")
        self._log_subgoal_status(
            status="failed",
            subgoal=subgoal,
            branch_idx=self.branch_idx,
            subgoal_idx=self.subgoal_idx,
            failure_type="pddl_no_plan",
            failure_msg=failure_msg,
            extra={
                "predicate": predicate,
                "retry_count": self._direct_action_reprompt_count,
            },
        )
        self.current_plan = []
        self.current_action_idx = 0
        self._direct_pddl_subgoals = []
        self._direct_pddl_subgoal_idx = 0
        if self._direct_action_reprompt_count < self.max_vlm_action_reprompts_per_subgoal:
            self._direct_action_reprompt_count += 1
            return False
        self._direct_skip_current_subgoal()
        return False

    def _direct_plan_pddl_predicate(
        self,
        predicate: str,
        world_graph,
        agent_uid: int,
    ) -> bool:
        plan, _name_map, reverse_map, _init = self._plan_for_subgoal(
            predicate,
            world_graph,
            agent_uid,
        )
        self._reverse_name_map = reverse_map or {}
        self._log_event(
            {
                "event": "pddl_plan",
                "reprompt_round": self._reprompt_round,
                "branch": self.branch_idx,
                "branch_origin": (
                    self._branch_origin_round[self.branch_idx]
                    if self.branch_idx < len(self._branch_origin_round)
                    else None
                ),
                "subgoal_idx": self.subgoal_idx,
                "subgoal": predicate,
                "custom_object_subgoal": self._direct_current_subgoal(),
                "status": "failed" if plan is None else "already" if plan == [] else "planned",
                "failure_type": "pddl_no_plan" if plan is None else None,
                "plan": plan,
            }
        )

        if plan is None:
            return self._direct_handle_pddl_plan_failure(
                predicate,
                f"Failed to plan for PDDL predicate: {predicate}",
            )

        if plan == []:
            self._vprint(f"  [already satisfied] {predicate}", "green")
            return True

        high_level_plan: List[HighLevelAction] = []
        for idx, action in enumerate(plan):
            next_action = plan[idx + 1] if idx + 1 < len(plan) else None
            tool_name, tool_args = self._action_to_tool(action, next_action=next_action)
            if tool_name is None:
                return self._direct_handle_pddl_plan_failure(
                    predicate,
                    f"Unsupported PDDL action for predicate {predicate}: {action}",
                )
            high_level_plan.append((tool_name, tool_args, None))

        self.current_plan = high_level_plan
        self.current_action_idx = 0
        self._direct_plan_phase = "actions"
        self._vprint_list(
            f"PDDL high-level plan for {predicate}",
            high_level_plan,
            color="cyan",
        )
        return True

    def _direct_prompt_pddl_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph,
        agent_uid: int,
    ) -> bool:
        if not self._direct_pddl_subgoals and not self._direct_prompt_pddl_predicates(
            instruction,
            observations,
            world_graph,
            agent_uid,
        ):
            return False

        while self._direct_pddl_subgoal_idx < len(self._direct_pddl_subgoals):
            predicate = self._direct_pddl_subgoals[self._direct_pddl_subgoal_idx]
            if not self._direct_plan_pddl_predicate(predicate, world_graph, agent_uid):
                return False
            if self.current_plan:
                return True
            self._direct_pddl_subgoal_idx += 1

        self._direct_mark_current_subgoal_solved()
        return False

    def _direct_prompt_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph,
        agent_uid: int,
    ) -> bool:
        subgoal = self._direct_current_subgoal()
        if subgoal is None:
            return False
        if self.custom_action_executor == "pddlstream":
            return self._direct_prompt_pddl_actions(
                instruction,
                observations,
                world_graph,
                agent_uid,
            )

        scene_desc = self._direct_scene_context(observations, world_graph, agent_uid)
        action_history = list(self._direct_action_history)
        prompt = (
            SUBGOAL_TO_VLM_ACTIONS_PROMPT.format(
                high_level_task=instruction,
                sub_goal=f"{subgoal}\nNeeded objects: {self._direct_needed_objects}",
                actions=CUSTOM_ACTIONS,
                scene_graph=scene_desc,
                action_history=self._format_direct_history(action_history),
                replan_needed="True" if self._direct_action_reprompt_count else "False",
            )
            + self._build_object_alias_hint(subgoal)
        )
        self._vprint_header(
            f"CUSTOM VLM: Subgoal To Actions  Subgoal {self.subgoal_idx}",
            "magenta",
        )
        self._vprint(f"  [current subgoal] {subgoal}", "yellow")
        self._vprint_list(
            "needed objects",
            self._direct_needed_objects,
            color="green",
        )
        self._vprint_custom_action_set()
        self._vprint_list("action history", action_history, color="gray")
        raw_actions = self._direct_ask_vlm("custom_subgoal_to_actions", prompt)
        action_output = self._parse_python_list(raw_actions)
        direct_actions = self._parse_direct_actions(action_output)
        self._vprint_list("parsed action output", action_output, color="yellow")
        self._vprint_list(
            "parsed executable actions",
            direct_actions,
            color="cyan",
        )
        self._log_event(
            {
                "event": "custom_subgoal_to_actions_parsed",
                "branch": self.branch_idx,
                "subgoal_idx": self.subgoal_idx,
                "subgoal": subgoal,
                "needed_objects": list(self._direct_needed_objects),
                "parsed": action_output,
                "direct_actions": direct_actions,
                "reprompt_count": self._direct_action_reprompt_count,
            }
        )

        if any(
            action_name == "Explore" and error_msg == "exploration_required"
            for action_name, _action_args, error_msg in direct_actions
        ):
            self._direct_needed_objects = []
            self._direct_last_action_error = ""
            self.current_plan = []
            self.current_action_idx = 0
            self._direct_plan_phase = None
            return False

        valid_actions = [
            action for action in direct_actions if action[0] and action[2] is None
        ]
        if not valid_actions:
            self._last_failure = (
                f"Action prompt returned no executable actions for subgoal: {subgoal}"
            )
            self._direct_last_action_error = self._last_failure
            self._direct_skip_current_subgoal()
            return False

        self.current_plan = valid_actions
        self.current_action_idx = 0
        self._direct_plan_phase = "actions"
        return True

    def _direct_handle_action_failure(
        self,
        failed_action: HighLevelAction,
        response: str,
    ) -> None:
        subgoal = self._direct_current_subgoal() or "unknown_subgoal"
        failure_msg = f"Action failed: {failed_action} -> {response}"
        self._last_failure = failure_msg
        self._direct_last_action_error = failure_msg
        self._record_direct_action_effect(
            failed_action,
            response,
            success=False,
        )
        self.history.append(failure_msg)
        self._trace_append(failure_msg)
        self._vprint(f"  [FAIL] {failure_msg}", "red")
        self._log_subgoal_status(
            status="failed",
            subgoal=subgoal,
            branch_idx=self.branch_idx,
            subgoal_idx=self.subgoal_idx,
            failure_type="execution_failure",
            failure_msg=failure_msg,
            extra={
                "last_high_level_actions": self.last_high_level_actions,
                "response": response,
                "phase": self._direct_plan_phase,
                "retry_count": self._direct_action_reprompt_count,
            },
        )
        self.current_plan = []
        self.current_action_idx = 0
        self.last_high_level_actions = {}
        if (
            self.custom_action_executor == "pddlstream"
            and self._direct_plan_phase == "actions"
        ):
            self._direct_pddl_subgoals = []
            self._direct_pddl_subgoal_idx = 0

    def _direct_finish_action_plan(self) -> None:
        if self._direct_plan_phase != "actions":
            self.current_plan = []
            self.current_action_idx = 0
            self._direct_plan_phase = None
            return

        if self.custom_action_executor != "pddlstream":
            self._direct_mark_current_subgoal_solved()
            return

        predicate = (
            self._direct_pddl_subgoals[self._direct_pddl_subgoal_idx]
            if self._direct_pddl_subgoal_idx < len(self._direct_pddl_subgoals)
            else "unknown_predicate"
        )
        self._vprint(f"  [PDDL predicate complete] {predicate}", "green")
        self._trace_append(f"PDDL predicate solved: {predicate}")
        self.current_plan = []
        self.current_action_idx = 0
        self._direct_pddl_subgoal_idx += 1
        self._direct_plan_phase = None

        if self._direct_pddl_subgoal_idx >= len(self._direct_pddl_subgoals):
            self._direct_mark_current_subgoal_solved()

    def _record_direct_action_effect(
        self,
        action: HighLevelAction,
        response: str,
        *,
        success: bool,
    ) -> None:
        action_name, action_args, _action_error = action
        status = "SUCCESS" if success else "FAILED"
        entry = f"{status} {action_name}[{action_args}]: {response}"
        self._direct_action_history.append(entry)

    def _direct_apply_action_response(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graphs: Dict[int, Any],
        agent_uid: int,
        hl_snapshot: Dict[int, HighLevelAction],
        low_level_actions: Dict[int, Any],
        responses: Dict[int, str],
    ):
        if not any(responses.values()):
            return (
                low_level_actions,
                self._planner_info(agent_uid, responses=responses),
                False,
            )

        response = list(responses.values())[0]
        self._vprint(f"    Obs: {response}", "gray")
        self._trace_append(f"Observation: {response}")

        if self._response_failed(response):
            failed_action = hl_snapshot.get(agent_uid, (None, None, None))
            self._direct_handle_action_failure(failed_action, response)
            if (
                self._direct_plan_phase == "actions"
                and self._direct_action_reprompt_count
                < self.max_vlm_action_reprompts_per_subgoal
            ):
                self._direct_action_reprompt_count += 1
                return self._get_next_direct_vlm_action(
                    instruction, observations, world_graphs
                )
            if self._direct_plan_phase == "explore":
                return self._get_next_direct_vlm_action(
                    instruction, observations, world_graphs
                )
            self._direct_skip_current_subgoal()
            return self._get_next_direct_vlm_action(
                instruction, observations, world_graphs
            )

        completed_action = hl_snapshot.get(agent_uid, (None, None, None))
        self._record_direct_action_effect(
            completed_action,
            response,
            success=True,
        )
        self.current_action_idx += 1
        self.last_high_level_actions = {}
        if self.current_action_idx >= len(self.current_plan):
            self._direct_finish_action_plan()
        return (
            low_level_actions,
            self._planner_info(agent_uid, hl_override=hl_snapshot, responses=responses),
            False,
        )

    def _direct_continue_last_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graphs: Dict[int, Any],
        agent_uid: int,
    ):
        hl_snapshot = dict(self.last_high_level_actions)
        tool_name, tool_args, _tool_error = hl_snapshot.get(
            agent_uid,
            (None, None, None),
        )
        try:
            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )
        except Exception as exc:
            low_level_actions = {}
            responses = {
                agent_uid: (
                    f"Action error while executing {tool_name}[{tool_args}]: {exc}"
                )
            }
        return self._direct_apply_action_response(
            instruction,
            observations,
            world_graphs,
            agent_uid,
            hl_snapshot,
            low_level_actions,
            responses,
        )

    def _direct_execute_next_planned_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graphs: Dict[int, Any],
        agent_uid: int,
    ):
        action = self.current_plan[self.current_action_idx]
        tool_name, tool_args, tool_error = action
        if tool_error:
            self._direct_handle_action_failure(action, tool_error)
            return {}, self._planner_info(agent_uid, hl_override={}), False

        world_graph = world_graphs[agent_uid]
        if tool_name == "Explore" and tool_args:
            tool_args = self._resolve_direct_explore_room(tool_args, world_graph)
        elif tool_name == "Navigate" and tool_args:
            tool_args = self._resolve_direct_navigation_target(tool_args, world_graph)

        step_num = self.current_action_idx + 1
        total = len(self.current_plan)
        self._vprint(f"\n  ▶ [{step_num}/{total}]  {tool_name}[{tool_args}]", "cyan")
        self._trace_append(f"Action: {tool_name}[{tool_args}]")
        if (
            self._direct_plan_phase == "explore"
            and self.current_action_idx < len(self._direct_action_reasons)
        ):
            reason = self._direct_action_reasons[self.current_action_idx].strip()
            if reason:
                reason_line = f"Why: {reason}"
                self._vprint(f"    {reason_line}", "gray")
                self._trace_append(reason_line)
                self._subgoal_exec_log_append(f"    {reason_line}")
        hl_action = {agent_uid: (tool_name, tool_args, None)}

        if tool_name == "Explore" and self.explore_fast and tool_args:
            try:
                summary, relation_lines, _est_steps = self._fast_explore_room(
                    tool_args,
                    world_graph,
                    agent_uid,
                )
            except Exception as exc:
                rooms = sorted(room.name for room in world_graph.get_all_rooms())
                response = (
                    f"Fast Explore failed: room '{tool_args}' is not present in the graph. "
                    f"Available rooms: {', '.join(rooms)}. Error: {exc}"
                )
                return self._direct_apply_action_response(
                    instruction,
                    observations,
                    world_graphs,
                    agent_uid,
                    dict(hl_action),
                    {},
                    {agent_uid: response},
                )
            self._vprint(f"  [fast-explore] {summary}", "green")
            for line in relation_lines:
                self._vprint(line, "cyan")
            self._trace_append(f"Fast-explore: {summary}")
            self._subgoal_exec_log_append(f"  [fast-explore] {summary}")
            if relation_lines:
                self._subgoal_exec_log_append("\n".join(relation_lines))
            self._explored_rooms.add(tool_args)
            self.history.append(f"explore({tool_args})")
            self._log_event(
                {
                    "event": "custom_fast_explore",
                    "reprompt_round": self._reprompt_round,
                    "branch": self.branch_idx,
                    "subgoal_idx": self.subgoal_idx,
                    "room": tool_args,
                    "summary": summary,
                    "relation_lines": relation_lines,
                }
            )
            responses = {agent_uid: summary}
            return self._direct_apply_action_response(
                instruction,
                observations,
                world_graphs,
                agent_uid,
                dict(hl_action),
                {},
                responses,
            )

        self.last_high_level_actions = hl_action
        try:
            low_level_actions, responses = self.process_high_level_actions(
                hl_action, observations
            )
        except Exception as exc:
            low_level_actions = {}
            responses = {
                agent_uid: (
                    f"Action error while executing {tool_name}[{tool_args}]: {exc}"
                )
            }
        return self._direct_apply_action_response(
            instruction,
            observations,
            world_graphs,
            agent_uid,
            dict(hl_action),
            low_level_actions,
            responses,
        )

    def _get_next_direct_vlm_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graphs: Dict[int, Any],
    ):
        if self.is_done:
            uid = self._agents[0].uid if self._agents else 0
            return {}, self._planner_info(uid, hl_override={}), True

        agent_uid = self._agents[0].uid if self._agents else 0
        world_graph = world_graphs[agent_uid]

        if not self.trace:
            self._trace_append(f"Task: {instruction}")
        if not self._episode_banner_printed:
            self._vprint_banner(instruction)
            self._episode_banner_printed = True

        if not self.branches:
            self._direct_generate_subgoals(
                instruction,
                observations,
                world_graph,
                agent_uid,
            )
            if not self.branches:
                self._vprint("  [ABORT] No custom VLM subgoals generated.", "red")
                self.is_done = True
                return (
                    {},
                    self._planner_info(
                        agent_uid,
                        hl_override={},
                        is_done={agent_uid: True},
                    ),
                    True,
                )

        if self.last_high_level_actions:
            return self._direct_continue_last_action(
                instruction, observations, world_graphs, agent_uid
            )

        subgoal = self._direct_current_subgoal()
        if subgoal is None:
            self._vprint_header("ALL CUSTOM VLM SUBGOALS DONE", "green")
            self.is_done = True
            return (
                {},
                self._planner_info(
                    agent_uid,
                    hl_override={},
                    is_done={agent_uid: True},
                ),
                True,
            )

        subgoal_key = (self.branch_idx, self.subgoal_idx)
        if subgoal_key not in self._direct_logged_started_subgoals:
            self._direct_logged_started_subgoals.add(subgoal_key)
            self._log_subgoal_status(
                status="started",
                subgoal=subgoal,
                branch_idx=self.branch_idx,
                subgoal_idx=self.subgoal_idx,
            )
            self._subgoal_exec_log_begin_subgoal()

        if self.current_action_idx < len(self.current_plan):
            return self._direct_execute_next_planned_action(
                instruction, observations, world_graphs, agent_uid
            )

        if not self._direct_needed_objects:
            self._direct_prompt_exploration(
                instruction,
                observations,
                world_graph,
                agent_uid,
            )
            if self.current_action_idx < len(self.current_plan):
                return self._direct_execute_next_planned_action(
                    instruction, observations, world_graphs, agent_uid
                )
            return self._get_next_direct_vlm_action(
                instruction, observations, world_graphs
            )

        self._direct_prompt_actions(
            instruction,
            observations,
            world_graph,
            agent_uid,
        )
        if self.current_action_idx < len(self.current_plan):
            return self._direct_execute_next_planned_action(
                instruction, observations, world_graphs, agent_uid
            )
        return self._get_next_direct_vlm_action(
            instruction, observations, world_graphs
        )

    def _advance_branch(self) -> bool:
        if self.skip_failed_subgoals and self.subgoal_idx < len(self._active_subgoals):
            failed_subgoal = self._active_subgoals[self.subgoal_idx]
            self._vprint(
                f"  [SKIP] Subgoal failed after retries; moving to next subgoal: {failed_subgoal}",
                "yellow",
            )
            self._trace_append(
                f"Subgoal skipped after retries: {failed_subgoal}"
            )
            self._log_subgoal_status(
                status="failed",
                subgoal=failed_subgoal,
                branch_idx=self.branch_idx,
                subgoal_idx=self.subgoal_idx,
                failure_type="custom_approach_skip_failed_subgoal",
                failure_msg=self._last_failure,
            )
            self._emit_subgoal_execution("failed", failed_subgoal)
            self.subgoal_idx += 1
            self.current_plan = []
            self.current_action_idx = 0
            self._subgoal_retry_count = 0
            self.last_high_level_actions = {}
            self._log_plan_tree()
            return True
        return super()._advance_branch()

    def _build_english_subgoal_prompt(
        self,
        goal: str,
        objects_by_type: Dict[str, Any],
        scene_description: str,
        history: str = "",
        after_explore: bool = False,
        explored_room: Optional[str] = None,
    ) -> str:
        return custom_build_english_subgoal_prompt(
            goal=goal,
            objects_by_type=objects_by_type,
            scene_description=scene_description,
            history=history,
            after_explore=after_explore,
            explored_room=explored_room,
        )

    def _build_predicate_translation_prompt(
        self,
        objects_by_type: Dict[str, Any],
        num_branches: int = 1,
        after_explore: bool = False,
    ) -> str:
        return custom_build_predicate_translation_prompt(
            objects_by_type=objects_by_type,
            num_branches=num_branches,
            after_explore=after_explore,
        )

    def _build_failure_history(
        self,
        actions: List[str],
        failure: Optional[str] = None,
        already_succeeded: Optional[List[str]] = None,
        explore_room: Optional[str] = None,
    ) -> str:
        return custom_build_failure_history(
            actions=actions,
            failure=failure,
            already_succeeded=already_succeeded,
            explore_room=explore_room,
        )
