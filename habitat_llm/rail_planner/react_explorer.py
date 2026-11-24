"""ReAct exploration phase for RailPlanner's two_phase mode.

ExploreReactPlanner is the ReAct baseline (ZeroShotReactPlanner: same prompt template, Thought/Action parsing, camera
image and skill stepping) restricted to search actions. RailPlanner runs one instance per episode and starts a new
phase with start_phase(); the phase ends when the LLM says Done[], or RailPlanner stops it at the skill cap.
"""

from typing import Any, Dict, List

from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner

EXPLORE_TOOLS = ("Explore", "Navigate", "Open", "Close")

PHASE_INSTRUCTION = """{task}

[EXPLORATION PHASE] You are only gathering information for the task above; a separate module will do the manipulation
afterwards. Do not pick up, place or change any object. Allowed actions: {tools} and Done[].
Exploration goal: {goal}
Close anything you open once you have looked inside. Call Done[] as soon as you have found what the exploration goal
asks for, or when further searching is unlikely to help.{previous}"""


class ExploreReactPlanner(ZeroShotReactPlanner):
    def reset(self):
        # LLMPlanner.reset() also resets every agent; the agent is shared with RailPlanner, whose skill state must
        # survive, so drop the agents for the duration of the reset.
        agents, self._agents = getattr(self, "_agents", []), []
        try:
            super().reset()
        finally:
            self._agents = agents
        self.rejected = False
        self.phase_goal = ""

    def start_phase(self, goal: str, previous: List[str]) -> None:
        """Fresh ReAct transcript for a new phase; token usage and LLM time keep accumulating."""
        self.last_high_level_actions = {}
        self.replan_required = True
        self.is_done = False
        self._termination = {}
        self.curr_prompt = ""
        self.trace = ""
        self.rejected = False
        self.phase_goal = goal or "find the objects and furniture the task needs that are not in memory yet"
        self.previous = list(previous)

    def phase_instruction(self, task: str) -> str:
        previous = (
            "\nEarlier exploration phases:\n" + "\n".join(f"- {p}" for p in self.previous) if self.previous else ""
        )
        return PHASE_INSTRUCTION.format(task=task, tools=", ".join(EXPLORE_TOOLS), goal=self.phase_goal,
                                        previous=previous)

    def prepare_prompt(self, input_instruction: str, world_graph, **kwargs):
        prompt, params = super().prepare_prompt(input_instruction, world_graph, **kwargs)
        if "tool_descriptions" in params:
            tools = self.agents[0].tools
            params["tool_descriptions"] = "\n".join(
                f"- {name}: {tools[name].description}" for name in sorted(tools) if name in EXPLORE_TOOLS
            )
            prompt = self.prompt.format(**params)
        return prompt, params

    def _llm_prompt_with_action_image(self, observations: Dict[str, Any]):
        """ReAct prompt (+ camera frame), plus RailPlanner's overhead map when plan_config.overhead_map is on."""
        prompt = super()._llm_prompt_with_action_image(observations)
        provider = getattr(self, "overhead_map_provider", None)
        extra = provider() if provider is not None else None
        if not extra:
            return prompt
        parts = [("text", prompt)] if isinstance(prompt, str) else list(prompt)
        return parts + list(extra)

    def process_high_level_actions(self, high_level_actions: Dict[int, Any], observations):
        # Reject manipulation without dispatching it; the message goes back to the LLM as the action result.
        bad = {
            uid: action for uid, action in high_level_actions.items()
            if action and action[0] not in EXPLORE_TOOLS and not (len(action) > 2 and action[2])
        }
        if bad:
            self.rejected = True
            return {}, {
                uid: f"{action[0]} is not allowed in the exploration phase. Use only "
                     f"{', '.join(EXPLORE_TOOLS)}, or Done[] when you have found what you need."
                for uid, action in bad.items()
            }
        self.rejected = False
        return super().process_high_level_actions(high_level_actions, observations)
