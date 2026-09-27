#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from habitat.tasks.rearrange.utils import coll_name_matches
from hydra.utils import instantiate

from habitat_llm.llm.instruct.utils import (
    draw_agent_rgb_to_pil,
    get_objects_descr,
    get_rearranged_objects_descr,
    get_world_descr,
    pil_image_to_data_url,
)
from habitat_llm.planner.planner import Planner
from habitat_llm.utils.core import cprint
from habitat_llm.utils.fast_explore_cost import (
    estimate_explore_tour,
    scale_explore_steps,
    sim_time_from_steps,
    walk_to_approx_ratio,
)
from habitat_llm.utils.llm_usage import fmt_usd, snapshot_from_llm
from habitat_llm.utils.grammar import (
    FREE_TEXT,
    FURNITURE,
    NAV_TARGET,
    OBJECT,
    OBJECT_OR_FURNITURE,
    ROOM,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)
from habitat_llm.world_model import Object, Receptacle

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.agent import Agent
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.planner.rag import RAG
    from habitat_llm.world_model.world_graph import WorldGraph


class LLMPlanner(Planner):
    """
    High level planner policy used by agents to decide high level actions
    given task description and state of the world
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ):
        """
        Initialize the LLMPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        # Set the planner config
        super().__init__(plan_config, env_interface)
        # Initialize LLM
        self.__initialize_llm()

        # Initialize a variable to indicate if replanning is required
        self.replan_required: bool = True

        # Initialize a variable to count number of llm calls
        self.replanning_count: int = 0

        # Initialize container to store entire prompt and current object states
        self.curr_prompt: str = ""
        self.curr_obj_states: str = ""

        # Initialize container to store rollout without
        # any other material in prompt
        self.trace: str = ""
        self.rag: Optional["RAG"] = None
        self.fast_explore: bool = bool(self.planner_config.get("fast_explore", False))
        self.llm_planning_time_s: float = 0.0
        self.last_llm_call_s: float = 0.0
        self.explore_approx_sim_steps: int = 0
        self.explore_approx_sim_time_s: float = 0.0
        self.explore_approx_meters: float = 0.0
        self.explore_walk_ratio: Optional[float] = None
        self._explore_calibrating: bool = False
        self._explore_calib_approx_steps: int = 0
        self._explore_calib_walked_steps: int = 0
        self.image_save_dir: Optional[str] = None
        self.last_action_image_relpath: Optional[str] = None

        self.reset()

        # Build RAG dataset if we want to use RAG
        if self.enable_rag:
            from habitat_llm.planner.rag import RAG

            self.rag = RAG(
                plan_config.example_type,
                plan_config.rag_dataset_dir,
                plan_config.rag_data_source_name,
                plan_config.llm,
            )

    def reset(self):
        """
        Reset the planner state.
        """
        self.last_high_level_actions: Dict[int, Tuple[str, str, str]] = {}
        self.replan_required: bool = True
        self.replanning_count: int = 0
        self.is_done: bool = False

        # save agent observations to get feedback on skill execution
        self.latest_agent_response: Dict[int, str] = {}
        self.curr_prompt: str = ""
        self.trace: str = ""
        self.curr_obj_states: str = ""
        self.params: Dict[str, Any] = {}
        self.llm_planning_time_s = 0.0
        self.last_llm_call_s = 0.0
        self.explore_approx_sim_steps = 0
        self.explore_approx_sim_time_s = 0.0
        self.explore_approx_meters = 0.0
        self.explore_walk_ratio = None
        self._explore_calibrating = False
        self._explore_calib_approx_steps = 0
        self._explore_calib_walked_steps = 0
        self.last_action_image_relpath = None
        tracker = getattr(self.llm, "token_usage", None)
        if tracker is not None and hasattr(tracker, "reset"):
            tracker.reset()

        # Reset agents
        for agent in self._agents:
            agent.reset()

    def set_image_save_dir(self, save_dir: str) -> None:
        """Directory for per-replan robot camera PNGs shown in the HTML trace."""
        self.image_save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def _send_action_image_enabled(self) -> bool:
        cfg = self.planner_config
        if cfg is None:
            return False
        return bool(getattr(cfg, "send_action_image", False))

    def _action_camera_pil(self, observations: Dict[str, Any]):
        agent_uid = self._agents[0].uid if self._agents else 0
        env = self.env_interface
        sim = getattr(env, "sim", None) if env is not None else None
        if sim is not None:
            update = getattr(sim, "maybe_update_articulated_agent", None)
            if callable(update):
                update()
            pil_image = draw_agent_rgb_to_pil(sim, agent_uid)
            if pil_image is not None:
                return pil_image
        return None

    def _save_action_image(self, pil_image) -> Optional[str]:
        if pil_image is None:
            return None
        if not self.image_save_dir:
            conf = getattr(self.env_interface, "conf", None)
            results_dir = getattr(getattr(conf, "paths", None), "results_dir", None)
            if not results_dir:
                return None
            self.set_image_save_dir(
                os.path.join(str(results_dir), "dataset", "traces", "0", "images")
            )
        os.makedirs(self.image_save_dir, exist_ok=True)
        filename = f"step_{self.replanning_count:04d}.png"
        path = os.path.join(self.image_save_dir, filename)
        pil_image.save(path)
        self.last_action_image_relpath = os.path.join("images", filename)
        return self.last_action_image_relpath

    def prepare_next_subgoal(
        self,
        instruction: str,
        world_graph: Dict[int, "WorldGraph"],
    ) -> None:
        """Continue planning on the same world state while preserving LLM history."""
        self.last_high_level_actions = {}
        self.replan_required = True
        self.is_done = False
        self.latest_agent_response = {}
        self.params = {}

        current_world_graph = world_graph[self._agents[0].uid]
        world_description = get_world_descr(
            current_world_graph,
            agent_uid=self.agents[0].uid,
            add_state_info=self.planner_config.objects_response_include_states,
            include_room_name=True,
            centralized=self.planner_config.centralized,
        )
        self.curr_obj_states = get_objects_descr(
            current_world_graph,
            self._agents[0].uid,
            include_room_name=True,
            add_state_info=self.planner_config.objects_response_include_states,
            centralized=self.planner_config.centralized,
        )

        continuation_prompt = (
            f"{self.planner_config.llm.user_tag}"
            "Continue from the previous action history and current world state.\n"
            f"Task: {instruction}\n\n"
            f"{world_description}\n\n"
            "What is the next action to make progress towards completing the task?\n"
            "Return your response in the following format\n\n"
            "Thought: <reasoning for why you are taking the next action>\n"
            "<next action call>\n"
            f"Assigned!\n{self.planner_config.llm.eot_tag}"
            f"{self.planner_config.llm.assistant_tag}"
        )
        if self.curr_prompt and not self.curr_prompt.endswith("\n"):
            self.curr_prompt += "\n"
        self.curr_prompt += continuation_prompt

        if self.trace and not self.trace.endswith("\n"):
            self.trace += "\n"
        self.trace += (
            "\n"
            f"Task: {instruction}\n"
            "Thought: "
        )

    def build_tool_grammar(self, world_graph: "WorldGraph") -> str:
        """
        This method builds a grammar that accepts all valid tool calls based a world graph
        The grammar is specified in the EBNF grammar description format
        see https://github.com/epfl-dlab/transformers-CFG for details and examples

        :param world_graph: The world graph.
        """
        tool_grammar = {}
        objects = world_graph.get_all_objects()
        rules = []
        # we cannot include rules which have objects when there are no objects
        if len(objects) != 0:
            object_expansion = " | ".join(
                (f'"{x.name}"' for x in world_graph.get_all_objects())
            )
            object_rule = f"{OBJECT} ::= " + object_expansion
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM} | {OBJECT})"
            object_of_furniture_rule = (
                f"{OBJECT_OR_FURNITURE} ::= ({FURNITURE} | {OBJECT})"
            )
            rules.append(nav_target_rule)
            rules.append(object_rule)
            rules.append(object_of_furniture_rule)
        else:
            object_of_furniture_rule = f"{OBJECT_OR_FURNITURE} ::= {FURNITURE}"
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM})"
            rules.append(nav_target_rule)
            rules.append(object_of_furniture_rule)
        for agent in self.agents:
            for tool_name, tool in agent.tools.items():
                if tool_name not in tool_grammar:
                    # skip tools that require objects when there are no objects
                    if OBJECT in tool.argument_types and len(objects) == 0:
                        continue
                    tool_grammar[tool_name] = tool.grammar()
        tool_grammar["Done"] = '"Done[]"'
        grammar_str = "tool_call ::= " + " | ".join(tool_grammar.keys()) + "\n"
        for tool_name, tool_grammar_str in tool_grammar.items():
            grammar_str += f"{tool_name} ::= {tool_grammar_str}\n"

        # build rules for each of the argument types
        furniture_rule = f"{FURNITURE} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_furnitures())
        )
        room_rule = f"{ROOM} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_rooms())
        )
        spatial_constraint_rule = f'{SPATIAL_CONSTRAINT} ::= "next_to"'
        spatial_relation_rule = f'{SPATIAL_RELATION} ::= "on" | "within"'
        free_text_rule = f"{FREE_TEXT} ::= [ \"'.:,!a-zA-Z_0-9]*"
        white_space_rule = "WS ::= [ ]*"
        rules.append(furniture_rule)
        rules.append(room_rule)
        rules.append(spatial_constraint_rule)
        rules.append(spatial_relation_rule)
        rules.append(free_text_rule)
        rules.append(white_space_rule)
        grammar_str += "\n".join(rules)
        return grammar_str

    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        action_rules = []
        for i, agent in enumerate(self.agents):
            agent_id = agent.uid
            action_rule = f'action_{i} ::= "Agent_{agent_id}_Action: " tool_call'
            action_rules.append(action_rule)

        combined_action_rule = (
            "action ::= "
            + f' "{delimiter}" '.join(f"action_{i}" for i in range(len(self.agents)))
            + f' "{delimiter}Assigned!"'
        )
        termination_rule = 'termination ::= "Final Thought: Exit!"'
        root_role = f'root ::= {FREE_TEXT} "{delimiter}" (action | termination)'

        return "\n".join(
            [root_role, combined_action_rule]
            + action_rules
            + [termination_rule, tool_rules]
        )

    def __initialize_llm(self):
        """
        This method instantiates LLM as defined in the config
        """
        # Instantiate LLM from the Hydra config
        llm_conf = self.planner_config.llm
        self.llm = instantiate(llm_conf.llm)
        self.llm = self.llm(llm_conf)

        # Setup the LLM parameters
        # self.instruct = self.planner_config.llm.instruct
        self.instruct = self.planner_config.instruct
        self.prompt = self.instruct.prompt
        self.stopword = self.instruct.stopword
        self.end_expression = self.instruct.end_expression
        self.actions_parser = instantiate(self.instruct.actions_parser)

        # save agent observations to get feedback on skill execution
        self.latest_agent_response = {}

    def prepare_prompt(
        self, input_instruction: str, world_graph: "WorldGraph", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM.

        :param input_instruction: The input instruction.
        :param world_graph: The world graph.
        :return: The prepared prompt and parameters.
        """
        params = {
            "input": input_instruction,
            "tool_list": self.tool_list,
            "world_graph": world_graph,
            "id": self.agents[0].uid,
        }

        # We modify the prompt if we want to use RAG and the prompt has not
        # been modified
        if "{rag_examples}" in self.prompt:
            if self.rag is not None:
                _, index = self.rag.retrieve_top_k_given_query(
                    input_instruction, top_k=1, agent_id=self._agents[0].uid
                )
                index = index[0]

                example_str = (
                    f"{self.planner_config.llm.user_tag}Below are some example solutions from different settings:\nExample 1:\n"
                    + self.rag.data_dict[index]["trace"]
                    + "\n"
                )
                params["rag_examples"] = example_str
            else:
                params["rag_examples"] = ""
        if "{tool_descriptions}" in self.prompt:
            params["tool_descriptions"] = self.agents[0].tool_descriptions
        if "{agent_descriptions}" in self.prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{tool_list}" in self.prompt:
            params["tool_list"] = self.tool_list
        if "{system_tag}" in self.prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag
        if "{agent_role_description}" in self.prompt:
            # only support agent role description when planning for a single agent
            assert len(self.agents) == 1
            if str(self.agents[0].uid) == "1":
                agent_role_description = '\nYou are playing the role of the task giver. This means if the instruction says something like "You should move the object and I will wash it", then the other agent should be moving the object, and you should washing the it.\n'
            else:
                agent_role_description = '\nYou are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.\n'
            params["agent_role_description"] = agent_role_description
        if "{world_description}" in self.prompt:
            # only designed for the decentralized setting
            assert len(self.agents) == 1
            world_description = get_world_descr(
                world_graph,
                agent_uid=self.agents[0].uid,
                add_state_info=self.planner_config.objects_response_include_states,
                include_room_name=True,
                centralized=self.planner_config.centralized,
            )
            params["world_description"] = world_description

        if "should_format" in kwargs and not kwargs["should_format"]:
            # In some cases a subclass may want to fill the extra arguments here, so we don't format
            # because those arguments would be missing.
            output_prompt = ""
        else:
            output_prompt = self.prompt.format(**params)
        return output_prompt, params

    @property
    def tool_list(self) -> List[str]:
        """
        Returns a string listing the agents tools
        :return: A sorted list of tool names.
        """
        tool_set = set()
        for agent in self.agents:
            for tool_name in agent.tools:
                tool_set.add(tool_name)

        return sorted(tool_set)

    @property
    def agents(self) -> List["Agent"]:
        """
        Get the list of agents associated with this planner.

        :return: A list of Agent objects.
        """
        return self._agents

    @agents.setter
    def agents(self, agents: List["Agent"]) -> None:
        """
        Set the list of agents for this planner.

        :param agents: A list of Agent objects to be associated with this planner.
        """
        self._agents = agents
        # Pass on respective LLM instance into agent tools
        for agent in self._agents:
            agent.pass_llm_to_tools(self.llm)

    def get_last_agent_states(self) -> Dict[int, str]:
        """
        Get the last state descriptions for all agents.

        :return: A dictionary mapping agent UIDs to their last state descriptions.
        """
        # Container to store state descriptions
        agent_states = {}

        # Loop through the agents and populate state descriptions
        for agent in self._agents:
            agent_states[agent.uid] = agent.get_last_state_description()

        return agent_states

    # TODO: @zephirefaith implement agent's room affiliations in the world graph
    # and edit this function to read from it
    def get_last_agent_positions(self) -> Dict[str, Any]:
        """
        Get the last positions for all agents.

        :return: A dictionary mapping agent names to their positions.
        """
        # Container to store agent positions
        agent_positions = {}

        # get agent nodes
        agents = self.env_interface.full_world_graph.get_agents()

        # Loop through the agents and populate nodes
        for agent in agents:
            agent_positions[agent.name] = agent.get_property("translation")

        return agent_positions

    def get_agent_collisions(self) -> Dict[int, bool]:
        """
        Check if the agents are colliding.

        :return: A dictionary mapping agent UIDs to collision status.
        """
        # set collision to false
        collision = False

        # Get list of agent ids
        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in self.env_interface.sim.agents_mgr.articulated_agents_iter
        ]

        # Return false if only one agent is in the scene
        if len(agent_ids) == 2:
            # Perform collision check
            self.env_interface.sim.perform_discrete_collision_detection()
            contact_points = self.env_interface.sim.get_physics_contact_points()

            for cp in contact_points:
                if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                    cp, agent_ids[1]
                ):
                    collision = True

        # Declare output container
        out = {}

        # update the output
        for agent in self._agents:
            out[agent.uid] = collision

        return out

    def format_response(
        self, response: str, end_expression: Union[str, List[str]]
    ) -> str:
        """
        Format the LLM response by trimming it up to the first appearance of end_expression.

        :param response: The LLM response to format.
        :param end_expression: The end expression(s) to look for.
        :return: The formatted response.
        """
        response = response.rstrip("\n")
        if type(end_expression) == str:
            index = response.find(end_expression)
            target_end_expression = end_expression
        else:
            # end_expression is a list of string
            index = -1
            target_end_expression = ""
            for _end_expression in end_expression:
                _index = response.find(_end_expression)
                if _index < index or index == -1:
                    index = _index
                    target_end_expression = _end_expression
        return (
            response[: index + len(target_end_expression)] if index != -1 else response
        )

    def parse_thought(self, input_string: str) -> str:
        """
        Extract thought from the LLM response.

        :param input_string: The input string to parse.
        :return: The extracted thought.
        """
        # Define the patterns for Agent actions
        pattern = r"\n|Final Thought"

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the text before the pattern
            return input_string[: match.start()].strip()
        else:
            # If no pattern is found, return the whole string
            return ""

    def _add_responses_to_prompt(self, responses: Dict[int, str]) -> str:
        """
        Add agent responses to the prompt.

        :param responses: A dictionary of agent responses.
        :return: The updated print string.
        """
        print_str = ""
        prompt_addition = ""
        add_object_update = False
        for agent_uid in sorted(responses.keys()):
            # If the response for a given agent is valid, add to the prompt and printout
            if responses[agent_uid]:
                # Print color-coded result with enhanced formatting
                cprint("\n" + "="*80, "magenta")
                if "success" in responses[agent_uid].lower() or responses[agent_uid] == "":
                    cprint("✓ ACTION RESULT: SUCCESS", "green")
                    cprint("="*80, "magenta")
                    cprint(f"Agent {agent_uid}: {responses[agent_uid]}", "white")
                else:
                    cprint("✗ ACTION RESULT: FAILURE", "red")
                    cprint("="*80, "magenta")
                    cprint(f"Agent {agent_uid}: {responses[agent_uid]}", "white")
                cprint("="*80 + "\n", "magenta")

                # Update print string
                print_str += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # Update the prompt
                prompt_addition += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # self.curr_prompt += prompt_addition
                self.trace += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )

            # If the response is empty then indicate the action is still in progress
            # only when replanning was required
            elif self.replan_required:
                responses[
                    agent_uid
                ] = f"Action {self.last_high_level_actions[agent_uid][0]}[{self.last_high_level_actions[agent_uid][1]}] is still in progress."

                # Print in-progress status
                cprint("─" * 80, "blue")
                cprint("⏳ ACTION STATUS: IN PROGRESS", "yellow")
                cprint(f"Agent {agent_uid}: {responses[agent_uid]}", "white")
                cprint("─" * 80, "blue")

                # Update print string
                print_str += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )

                # Update the prompt
                prompt_addition += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # self.curr_prompt += prompt_addition
                self.trace += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                add_object_update = True

            # save agent observations to get feedback on skill execution
            self.latest_agent_response[agent_uid] = responses[agent_uid]

        if prompt_addition != "":
            self.curr_prompt += self.planner_config.llm.user_tag + prompt_addition
            if (
                self.planner_config.objects_response
                and add_object_update
                and self.planner_config.centralized
            ):
                world_graph = self.env_interface.world_graph[agent_uid]
                objects = get_objects_descr(
                    world_graph,
                    agent_uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                    centralized=self.planner_config.centralized,
                )
                if self.planner_config.prompt_w_updatedobjects_only:
                    # add details on what changed in the world.
                    # TODO: this currently assumes symmetric world graph,
                    # extend for decentralized/asymmetric WG
                    updated_objects = get_rearranged_objects_descr(
                        obj_descr_t=objects, obj_descr_t_1=self.curr_obj_states
                    )
                    self.curr_obj_states = objects
                    if updated_objects != "":
                        result = f"Newly found objects/updates on known objects: {updated_objects}\n"
                    else:
                        result = (
                            "No new objects or updates on known objects were found.\n"
                        )
                else:
                    result = f"Objects: {objects}\n"
                self.curr_prompt += result
                self.trace += result
                print_str += result
            self.curr_prompt += self.planner_config.llm.eot_tag
            # Suppress detailed prompt printing (too verbose)
            # print(self.curr_prompt)

        # Force add thought after every observation
        if self.planner_config.planning_mode.lower() == "cot":
            for agent_uid in sorted(responses.keys()):
                if responses[agent_uid]:
                    print_str += "Thought:"
                    prompt_addition = f"{self.planner_config.llm.assistant_tag}Thought:"
                    self.curr_prompt += prompt_addition
                    # Ensure newline before Thought: if trace doesn't end with one
                    if not self.trace.endswith('\n'):
                        self.trace += '\n'
                    self.trace += "Thought:"
                    break
        return print_str

    def _llm_prompt_with_action_image(self, observations: Dict[str, Any]):
        """Text prompt, plus robot camera frame when send_action_image is on."""
        if not self._send_action_image_enabled():
            return self.curr_prompt
        pil_image = self._action_camera_pil(observations)
        if pil_image is None:
            try:
                keys = [str(key) for key in list(observations.keys())[:24]]
            except Exception:
                keys = [type(observations).__name__]
            cprint(
                "[action_image] no robot RGB found; sending text-only prompt"
                f" keys={keys}",
                "yellow",
            )
            return self.curr_prompt
        self._save_action_image(pil_image)
        text = (
            self.curr_prompt
            + "\n[Robot overhead camera after the last action. Use it with the text state.]\n"
        )
        return [("text", text), ("image", pil_image_to_data_url(pil_image))]

    def replan(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ):
        """
        Replan a high level action using the LLM/VLM
        """
        prompt = self._llm_prompt_with_action_image(observations)
        # Generate response
        if self.planner_config.get("constrained_generation", False):
            llm_response = self.llm.generate(
                prompt,
                self.stopword,
                generation_args={
                    "grammar_definition": self.build_response_grammar(
                        world_graph[self._agents[0].uid]
                    )
                },
            )
        else:
            llm_response = self.llm.generate(prompt, self.stopword)

        # Format the response
        # This removes extra text followed by end expression when needed.
        llm_response = self.format_response(llm_response, self.end_expression)

        info = {"llm_response": llm_response}
        return info

    def _fast_explore_room(
        self, room_name: str, world_graph: "WorldGraph"
    ) -> Tuple[bool, str, List[str]]:
        """Populate the current world graph with GT objects on furniture/floor in a room."""
        gt = getattr(getattr(self.env_interface, "perception", None), "gt_graph", None)
        if gt is None:
            return False, "Fast explore failed: GT graph is unavailable.", []

        try:
            room_node = gt.get_node_from_name(room_name)
        except ValueError:
            return False, f"Fast explore failed: room '{room_name}' was not found.", []

        object_nodes: List[Object] = []
        relation_lines: List[str] = []
        seen_objects = set()
        all_furniture = gt.get_furniture_in_room(room_node)

        for furn in all_furniture:
            for obj in gt.get_neighbors_of_type(furn, Object):
                if obj.name in seen_objects:
                    continue
                seen_objects.add(obj.name)
                object_nodes.append(obj)
                relation_lines.append(f"{obj.name} --[on]--> {furn.name}")
            for rec in gt.get_neighbors_of_type(furn, Receptacle):
                if rec.properties.get("type") == "within":
                    continue
                for obj in gt.get_neighbors_of_type(rec, Object):
                    if obj.name in seen_objects:
                        continue
                    seen_objects.add(obj.name)
                    object_nodes.append(obj)
                    relation_lines.append(f"{obj.name} --[on]--> {furn.name}")

        if object_nodes:
            subgraph = gt.get_subgraph(object_nodes)
            world_graph.update(subgraph, partial_obs=True, update_mode="gt")

        summary = (
            f"Fast explore success in {room_name}: found {len(object_nodes)} object(s) "
            "on top of furniture/floor."
        )
        return True, summary, relation_lines

    def _robot_base_pos(self, agent_uid: int) -> Optional[Any]:
        try:
            sim = self.env_interface.sim
            return sim.agents_mgr[agent_uid].articulated_agent.base_pos
        except Exception:
            return None

    def _pathfinder(self) -> Optional[Any]:
        try:
            return self.env_interface.sim.pathfinder
        except Exception:
            return None

    def _furniture_for_explore(
        self, room_name: str, world_graph: "WorldGraph"
    ) -> List[Any]:
        try:
            return list(world_graph.get_furniture_in_room(room_name))
        except (ValueError, KeyError, TypeError):
            pass
        gt = getattr(getattr(self.env_interface, "perception", None), "gt_graph", None)
        if gt is None:
            return []
        try:
            return list(gt.get_furniture_in_room(room_name))
        except (ValueError, KeyError, TypeError):
            return []

    def _estimate_explore_tour(
        self, room_name: str, agent_uid: int, world_graph: "WorldGraph"
    ) -> Optional[Dict[str, Any]]:
        try:
            furniture = self._furniture_for_explore(room_name, world_graph)
            return estimate_explore_tour(
                self._pathfinder(),
                self._robot_base_pos(agent_uid),
                furniture,
                room_name=room_name,
            )
        except Exception:
            return None

    def _approx_explore_cost(
        self, room_name: str, agent_uid: int, world_graph: "WorldGraph"
    ) -> List[str]:
        result = self._estimate_explore_tour(room_name, agent_uid, world_graph)
        if result is None:
            return [f"[fast_explore] approx tour {room_name}: failed"]
        raw_steps = int(result["total_steps"])
        raw_time_s = float(result["total_time_s"])
        steps = scale_explore_steps(raw_steps, self.explore_walk_ratio)
        time_s = sim_time_from_steps(steps)
        self.explore_approx_sim_steps += steps
        self.explore_approx_sim_time_s += time_s
        self.explore_approx_meters += float(result["total_meters"])
        lines = list(result["summary_lines"])
        if lines:
            visited = sum(1 for leg in result.get("legs") or [] if not leg.get("skipped"))
            lines[0] = (
                f"[fast_explore] approx tour {room_name}: {visited} furniture, "
                f"{float(result['total_meters']):.1f} m, "
                f"~{steps} sim steps, ~{time_s:.2f} s sim time"
            )
        if self.explore_walk_ratio is not None:
            lines.append(
                f"[fast_explore] scaled x{self.explore_walk_ratio:.3f}: "
                f"{raw_steps} geodesic steps ({raw_time_s:.2f}s) -> "
                f"{steps} steps ({time_s:.2f}s sim)"
            )
        return lines

    def _hla_is_all_explore(
        self, high_level_actions: Dict[int, Tuple[str, str, str]]
    ) -> bool:
        if not high_level_actions:
            return False
        for action_tuple in high_level_actions.values():
            if action_tuple[0] != "Explore" or not action_tuple[1]:
                return False
        return True

    def _start_explore_calibration(
        self,
        high_level_actions: Dict[int, Tuple[str, str, str]],
        world_graph: Dict[int, "WorldGraph"],
    ) -> None:
        agent_uid, action_tuple = next(iter(high_level_actions.items()))
        room_name = action_tuple[1]
        result = self._estimate_explore_tour(
            room_name, agent_uid, world_graph[agent_uid]
        )
        approx_steps = int(result["total_steps"]) if result is not None else 0
        approx_time_s = float(result["total_time_s"]) if result is not None else 0.0
        approx_meters = float(result["total_meters"]) if result is not None else 0.0
        self._explore_calibrating = True
        self._explore_calib_approx_steps = approx_steps
        self._explore_calib_walked_steps = 0
        self.explore_approx_meters += approx_meters
        cprint(
            f"[explore_calib] walking Explore[{room_name}] once "
            f"(geodesic ~{approx_steps} steps / {approx_time_s:.2f}s) "
            "to measure walked/geodesic ratio",
            "yellow",
        )
        if result is not None:
            for line in result["summary_lines"]:
                cprint(line, "cyan")

    def _tick_explore_calibration(self) -> None:
        if not self._explore_calibrating:
            return
        last = self.last_high_level_actions or {}
        if not any(action_tuple[0] == "Explore" for action_tuple in last.values()):
            return
        self._explore_calib_walked_steps += 1
        self.explore_approx_sim_steps += 1
        self.explore_approx_sim_time_s = sim_time_from_steps(
            self.explore_approx_sim_steps
        )

    def _maybe_finish_explore_calibration(self, responses: Dict[int, str]) -> None:
        if not self._explore_calibrating:
            return
        if not any(responses.values()):
            return
        walked = self._explore_calib_walked_steps
        approx = self._explore_calib_approx_steps
        ratio = walk_to_approx_ratio(walked, approx)
        self._explore_calibrating = False
        walked_time_s = sim_time_from_steps(walked)
        approx_time_s = sim_time_from_steps(approx)
        if ratio is None:
            cprint(
                f"[explore_calib] skipped ratio "
                f"(walked {walked} steps / geodesic {approx} steps)",
                "yellow",
            )
            return
        self.explore_walk_ratio = ratio
        cprint(
            f"[explore_calib] walked {walked} steps ({walked_time_s:.2f}s sim) vs "
            f"geodesic {approx} steps ({approx_time_s:.2f}s) => ratio {ratio:.3f}x "
            "(applied to later fast Explore estimates)",
            "yellow",
        )

    def _cost_metrics(self) -> Dict[str, Any]:
        # Nested dict so DecentralizedEvaluationRunner can merge planner_info
        # (it only accepts dict or str values at the top level).
        metrics: Dict[str, Any] = {
            "llm_planning_time_s": self.llm_planning_time_s,
            "last_llm_call_s": self.last_llm_call_s,
            "llm_call_count": self.replanning_count,
            "explore_approx_sim_steps": self.explore_approx_sim_steps,
            "explore_approx_sim_time_s": self.explore_approx_sim_time_s,
            "explore_approx_meters": self.explore_approx_meters,
        }
        metrics.update(snapshot_from_llm(self.llm))
        if self.explore_walk_ratio is not None:
            metrics["explore_walk_ratio"] = self.explore_walk_ratio
        return {"cost_metrics": metrics}

    def _try_fast_explore_actions(
        self,
        high_level_actions: Dict[int, Tuple[str, str, str]],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Optional[Dict[int, str]]:
        """Return synthetic responses for fast Explore actions when enabled."""
        if not self.fast_explore:
            return None
        if not self._hla_is_all_explore(high_level_actions):
            return None
        if self.explore_walk_ratio is None:
            if not self._explore_calibrating:
                self._start_explore_calibration(high_level_actions, world_graph)
            return None

        responses: Dict[int, str] = {}
        for agent_uid, action_tuple in high_level_actions.items():
            room_name = action_tuple[1]
            agent_world_graph = world_graph[agent_uid]
            success, summary, relation_lines = self._fast_explore_room(
                room_name, agent_world_graph
            )
            color = "green" if success else "yellow"
            cprint(f"[fast_explore] {summary}", color)
            for line in relation_lines:
                cprint(f"  {line}", "cyan")

            response_text = summary
            if relation_lines:
                response_text += "\n" + "\n".join(relation_lines)
            if success:
                tour_lines = self._approx_explore_cost(
                    room_name, agent_uid, agent_world_graph
                )
                for line in tour_lines:
                    cprint(line, "cyan")
                if tour_lines:
                    response_text += "\n" + "\n".join(tour_lines)
            responses[agent_uid] = response_text

        return responses


    # ! IMPT: Method used to get the next actions.
    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Get the next low-level action to execute.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :param verbose: Whether to print verbose output. Defaults to False.
        :return: A tuple containing:
                 - The low-level actions for each agent
                 - Planner information
                 - Whether the planner is done
        """
        planner_info: Dict[str, Union[Any, str]] = {}
        # Early return if planner is already done
        if self.is_done:
            planner_info = {
                "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                "traces": {agent.uid: self.trace for agent in self.agents},
                "replanning_count": {
                    agent.uid: self.replanning_count for agent in self.agents
                },
                "replanned": {agent.uid: False for agent in self.agents},
                "replan_required": {
                    agent.uid: self.replan_required for agent in self.agents
                },
                "is_done": {agent.uid: self.is_done for agent in self.agents},
            }
            planner_info.update(self._cost_metrics())
            return {}, planner_info, self.is_done

        if self.curr_prompt == "":
            # Prepare prompts
            # Initial prompt is prepared here
            self.curr_prompt, self.params = self.prepare_prompt(
                instruction, world_graph[self._agents[0].uid], observations=observations
            )
            self.curr_obj_states = get_objects_descr(
                world_graph[self._agents[0].uid],
                self._agents[0].uid,
                include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states,
                centralized=self.planner_config.centralized,
            )
            # Eg: [(ball on table_0 in living room), (chair on table_0 in living room), ...]

        if self.trace == "":
            self.trace += f"Task: {instruction}\nThought: "

        print_str = ""
        self.is_done = False

        if self.replan_required:

            # This step occurs when the previous high level action has failed and we need
            # to get new low level actions for the same high level action

            planner_info["replanned"] = {agent.uid: True for agent in self.agents}
            start_time = time.time()
            response_info = self.replan(instruction, observations, world_graph)
            elapsed = time.time() - start_time
            self.last_llm_call_s = elapsed
            self.llm_planning_time_s += elapsed
            llm_response = response_info["llm_response"]
            # llm_response: 'Thought: Since there are no objects found yet, I should explore the living room first, as it is the shortest path to locate the white table for placing the candle, candle holder, and plant.\nExplore[living_room_1]'

            # Print LLM thought/response with color coding
            cprint("\n" + "="*80, "magenta")
            cprint("🤖 LLM RESPONSE", "cyan")
            cprint("="*80, "magenta")
            cprint(llm_response, "white")
            cprint("="*80 + "\n", "magenta")

            # parse thought from the response
            thought = self.parse_thought(llm_response)

            cprint(
                f"[timing] LLM call #{self.replanning_count + 1}: {elapsed:.2f}s  "
                f"(cumulative {self.llm_planning_time_s:.2f}s)",
                "yellow",
            )
            usage = snapshot_from_llm(self.llm)
            if usage.get("total_tokens"):
                cost_str = fmt_usd(usage.get("llm_usd"))
                cprint(
                    f"[usage] {usage.get('llm_model') or 'model'}  "
                    f"{usage.get('prompt_tokens', 0)} in / "
                    f"{usage.get('completion_tokens', 0)} out  "
                    f"episode {cost_str}",
                    "yellow",
                )

            # Update prompt with the first response
            print_str += f"""{llm_response}\n{self.stopword}\n"""
            prompt_addition = (
                f"""{llm_response}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
            )
            self.curr_prompt += prompt_addition
            # Ensure newline before adding LLM response if trace doesn't end with one
            # This prevents "Thought:" from being concatenated to the end of Objects section
            if not self.trace.endswith('\n') and llm_response.strip().startswith('Thought:'):
                self.trace += '\n'
            self.trace += prompt_addition
            # Trace: 'Task: Place the candle and candle holder on the white table in the living room. Move the plant to the same table.
            # \nThought: Thought: Since there are no objects found yet, I should explore the living room first,
            # as it is the shortest path to locate the white table for placing the candle, candle holder, and plant.\nExplore[living_room_1]\nAssigned!'

            # Check if the planner should stop
            # Stop if the replanning count exceed a certain threshold
            # or end expression is found in llm response
            # This is helpful to break infinite planning loop.
            self.is_done = (self.check_if_agent_done(llm_response)) or (
                self.replanning_count == self.planner_config.replanning_threshold
            )
            # Increment the llm call counter on every replan
            # doesn't get incremented before comparison as first "replan" is technically
            # the first required plan
            self.replanning_count += 1

            # Early return if stop is required
            if self.is_done:
                planner_info = {
                    "print": print_str,
                    # "print_no_tags": print_str_no_tags,
                    "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                    "traces": {agent.uid: self.trace for agent in self.agents},
                    "replanning_count": {
                        agent.uid: self.replanning_count for agent in self.agents
                    },
                    "replan_required": {
                        agent.uid: self.replan_required for agent in self.agents
                    },
                    "replanned": {agent.uid: True for agent in self.agents},
                    "is_done": {agent.uid: self.is_done for agent in self.agents},
                    "thought": {agent.uid: thought for agent in self.agents},
                    "high_level_actions": {
                        agent.uid: ("Done", None, None) for agent in self.agents
                    },
                }
                planner_info.update(self._cost_metrics())
                if self.last_action_image_relpath:
                    planner_info["action_image"] = self.last_action_image_relpath
                return {}, planner_info, self.is_done

            # Parse high level action directives from llm response
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params
            )

            # llm_response 'Thought: Since there are no objects found yet, I should explore the living room first,
            # as it is the shortest path to locate the white table for placing the candle, candle holder, and plant.\nExplore[living_room_1]'

            # self.params.keys: dict_keys(['input', 'tool_list', 'world_graph', 'id', 'rag_examples', 'tool_descriptions', 'system_tag',
            # 'user_tag', 'assistant_tag', 'eot_tag', 'agent_role_description', 'world_description'])

            # high_level_actions: {0: ('Explore', 'living_room_1', None)} or {0: ('Rearrange', 'candle_0, on, table_10, None, None', None)}

            # Print action being executed with color coding
            cprint("\n" + "="*80, "cyan")
            cprint("🎯 EXECUTING ACTION", "yellow")
            cprint("="*80, "cyan")
            for agent_id, action_tuple in high_level_actions.items():
                action_name = action_tuple[0]
                action_args = action_tuple[1]
                cprint(f"Agent {agent_id}: {action_name}[{action_args}]", "yellow")
            cprint("="*80 + "\n", "cyan")

            # Store last executed high level action
            self.last_high_level_actions = high_level_actions

            fast_explore_responses = self._try_fast_explore_actions(
                high_level_actions, world_graph
            )
            if fast_explore_responses is not None:
                low_level_actions = {}
                responses = fast_explore_responses
            else:
                # Get low level actions and/or responses
                low_level_actions, responses = self.process_high_level_actions(
                    high_level_actions, observations
                )
                self._maybe_finish_explore_calibration(responses)

            # low_level_actions: literally a np array of (290, ) 0s with the action so index 10 = -10.
            # So the actual low-level action is -10 for the movement

            # observations keys: dict_keys(['agent_0_articulated_agent_arm_depth', 'agent_0_articulated_agent_arm_panoptic', 'agent_0_articulated_agent_arm_rgb',
            # 'agent_0_articulated_agent_jaw_depth', 'agent_0_articulated_agent_jaw_panoptic', 'agent_0_articulated_agent_jaw_rgb', 'agent_0_ee_pos',
            # 'agent_0_goal_to_agent_gps_compass', 'agent_0_head_depth', 'agent_0_head_rgb', 'agent_0_humanoid_detector_sensor', 'agent_0_is_holding',
            # 'agent_0_joint', 'agent_0_obj_goal_sensor', 'agent_0_obj_start_sensor', 'agent_0_relative_resting_position', 'agent_0_third_rgb',
            # 'agent_1_articulated_agent_arm_depth', 'agent_1_articulated_agent_arm_rgb', 'agent_1_ee_pos', 'agent_1_goal_to_agent_gps_compass',
            # 'agent_1_head_depth', 'agent_1_head_panoptic', 'agent_1_head_rgb', 'agent_1_is_holding', 'agent_1_joint', 'agent_1_obj_goal_sensor',
            # 'agent_1_obj_start_sensor', 'agent_1_relative_resting_position', 'agent_1_third_rgb'])

            # breakpoint()

        else:
            planner_info["replanned"] = {agent.uid: False for agent in self.agents}
            # Set thought to None
            thought = None
            # This runs when the state goes to a new step because the previous high level action has been executed and we need to
            # get the next low level actions for a new high level action
            # Get low level actions and/or responses using last high level actions
            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )
            self._tick_explore_calibration()
            self._maybe_finish_explore_calibration(responses)

        # Log if replanning was done or not before overwriting the value
        planner_info["replan_required"] = {
            agent.uid: self.replan_required for agent in self.agents
        }

        # Check if replanning is required
        # Replanning is required when any of the actions being executed
        # have a response indicating success or failure (and the reason)??
        # When it fails, it gives an error {0: 'Unexpected failure! - No valid placements found for entity table_46.'} so should replan
        # When it succeeds, it gives a success message when there is no error {0: ''} ({0: ('Navigate', 'table_32', None)}) then it will not replan

        self.replan_required = any(responses.values()) # When the values() part have any data so any string
        print_str += self._add_responses_to_prompt(responses)

        # Update planner info
        planner_info["responses"] = responses
        planner_info["thought"] = {agent.uid: thought for agent in self.agents}
        planner_info["is_done"] = {agent.uid: self.is_done for agent in self.agents}
        planner_info["print"] = print_str
        # planner_info["print_no_tags"] = print_str_no_tags
        planner_info["high_level_actions"] = self.last_high_level_actions
        planner_info["prompts"] = {agent.uid: self.curr_prompt for agent in self.agents}
        planner_info["traces"] = {agent.uid: self.trace for agent in self.agents}
        planner_info["replanning_count"] = {
            agent.uid: self.replanning_count for agent in self.agents
        }
        planner_info["agent_states"] = self.get_last_agent_states()
        planner_info["agent_positions"] = self.get_last_agent_positions()
        planner_info["agent_collisions"] = self.get_agent_collisions()
        planner_info.update(self._cost_metrics())
        if planner_info.get("replanned") and self.last_action_image_relpath:
            planner_info["action_image"] = self.last_action_image_relpath
        return low_level_actions, planner_info, self.is_done

    def check_if_agent_done(self, llm_response: str) -> bool:
        """
        Check if the agent is done based on the LLM response.

        :param llm_response: The LLM response to check.
        :return: True if the agent is done, False otherwise.
        """
        return self.end_expression in llm_response
