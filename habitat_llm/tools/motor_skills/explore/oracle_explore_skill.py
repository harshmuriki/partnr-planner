# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import ROOM


class OracleExploreSkill(SkillPolicy):
    """
    This skill uses nav skill to navigate objects to random furniture
    in the world in order to search for an instance of the queried type.
    Whenever an instance of the queried type appears in the world, the
    skill exists with success.
    """

    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=True,
            agent_uid=agent_uid,
        )

        self.env = env

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Construct nav skill
        self.nav_skill = OracleNavSkill(
            config.nav_skill_config,
            observation_space,
            action_space,
            batch_size,
            env,
            agent_uid,
        )

        self.visited_node_count = 0
        self.target_node_reached = False

        # Variable to indicate end of exploration
        self._is_exploration_done = torch.zeros(self._batch_size)

        # Get agent pose
        agent_pos = np.array(self.articulated_agent.base_pos)

        # Set target node pose
        self.target_node_pose = [
            math.floor(agent_pos[0]),
            math.floor(agent_pos[1]),
            agent_pos[2],
        ]

        self.object_found = False
        self.target_object = None
        self.target_handle = None
        self.target_room_name: str = None
        self.fur_queue = []
        self.target_fur_name = None
        self.max_nav_steps_per_furniture = int(
            getattr(config, "max_nav_steps_per_furniture", -1)
        )
        self.max_furniture_samples_per_room = int(
            getattr(config, "max_furniture_samples_per_room", 10)
        )
        self.skip_failed_furniture = bool(
            getattr(config, "skip_failed_furniture", True)
        )
        self._current_furniture_nav_steps = 0
        self._skipped_furniture = []

    def set_target(self, target_room_name, env):
        """
        Sets target room
        """

        # Early return if the target is already set
        if self.target_is_set:
            return

        # Set target room name
        self.target_room_name = target_room_name

        # Populate the queue with furniture from this room
        # Only do this when the queue is empty
        if len(self.fur_queue) == 0 and self.target_fur_name is None:
            room_furniture = self.env.world_graph[self.agent_uid].get_furniture_in_room(
                target_room_name
            )
            if (
                self.max_furniture_samples_per_room > 0
                and len(room_furniture) > self.max_furniture_samples_per_room
            ):
                sampled_idxs = np.random.choice(
                    len(room_furniture),
                    size=self.max_furniture_samples_per_room,
                    replace=False,
                )
                self.fur_queue = [room_furniture[i] for i in sampled_idxs]
            else:
                self.fur_queue = room_furniture

        # Set flag to true
        self.target_is_set = True

        return

    def reset(self, batch_idxs):
        """
        This method is executed each time this skill
        is called for the first time by the high level planner.
        It resets the critical members of tha class.
        """
        super().reset(batch_idxs)
        self.nav_skill.reset(batch_idxs)
        self._is_exploration_done[batch_idxs] = False
        # self._is_exploration_done = torch.zeros(self._batch_size)

        self.fur_queue = []
        self.target_fur_name = None
        self.target_node_reached = False
        self.target_room_name = None
        self._current_furniture_nav_steps = 0
        self._skipped_furniture = []

        # Get agent pose
        agent_pos = np.array(self.articulated_agent.base_pos)

        # Set target node pose
        self.target_node_pose = [
            math.floor(agent_pos[0]),
            math.floor(agent_pos[1]),
            agent_pos[2],
        ]

        return

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Throw error if nav skill is none
        if self.nav_skill == None:
            raise ValueError(
                "Navigation skill cannot be None in the OracleExploreSkill"
            )

        # Initialize the actions and hidden states to zeros and None
        action = torch.zeros(prev_actions.shape, device=masks.device)
        hxs = None

        # Check if the target node has been reached
        self.target_node_reached = (
            self.nav_skill._is_skill_done(
                observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx
            ).sum()
            > 0
        )

        # Check if exploration is done
        if self.target_node_reached and len(self.fur_queue) == 0:
            self._is_exploration_done[cur_batch_idx] = True

            # Fetch termination message from the skill
            self.termination_message = self.nav_skill.termination_message
            self.failed = self.nav_skill.failed

            return action, hxs

        # Check if there are no furniture in the room to start with
        if self.target_fur_name is None and len(self.fur_queue) == 0:
            self._is_exploration_done[cur_batch_idx] = True

            self.failed = False

            # Fetch termination message from the skill
            self.termination_message = f"There is no furniture in {self.target_room_name} room. Explore another room."

            return action, hxs

        # Check if the target node has been reached
        # or if target was never set
        if self.target_fur_name is None or self.target_node_reached:
            # Select next furniture to visit
            fur_node = self.fur_queue.pop(0)
            self.target_fur_name = fur_node.name
            self._current_furniture_nav_steps = 0

            # Reset the nav skill
            self.nav_skill.reset(cur_batch_idx)

            # Increase node counter
            self.visited_node_count += 1
            # print(f"\nself.visited_node_count {self.visited_node_count}")

            # Reset the target_is_set so that the agent can set a new target continuously
            self.nav_skill.target_is_set = False

        # Set nav skill target
        self.nav_skill.set_target(self.target_fur_name, self.env)

        # Get action for the navigation skill
        action, hxs = self.nav_skill._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )
        self._current_furniture_nav_steps += 1

        # Fetch termination message from the skill
        self.termination_message = self.nav_skill.termination_message
        self.failed = self.nav_skill.failed

        nav_timed_out = (
            self.max_nav_steps_per_furniture > 0
            and self._current_furniture_nav_steps >= self.max_nav_steps_per_furniture
        )
        nav_failed = bool(self.nav_skill.failed)

        if (
            self.skip_failed_furniture
            and self.target_fur_name is not None
            and (nav_failed or nav_timed_out)
        ):
            skipped_name = self.target_fur_name
            reason = "nav timeout" if nav_timed_out else "nav failure"
            self._skipped_furniture.append(skipped_name)

            # Reset nav and advance to the next furniture candidate.
            self.nav_skill.reset(cur_batch_idx)
            self.nav_skill.target_is_set = False
            self.target_fur_name = None
            self.target_node_reached = False
            self._current_furniture_nav_steps = 0
            self.failed = False

            if len(self.fur_queue) > 0:
                self.termination_message = (
                    f"Skipping furniture '{skipped_name}' ({reason}); continuing Explore."
                )
                return action, hxs

            # No furniture left after skipping this one: finish Explore gracefully.
            self._is_exploration_done[cur_batch_idx] = True
            self.termination_message = (
                f"Finished exploring room '{self.target_room_name}' after skipping "
                f"'{skipped_name}' ({reason})."
            )
            return action, hxs

        # print(f"agent {self.agent_uid} is exploring {self.target_fur_name} in {self.target_room_name}, {len(self.fur_queue)}")

        return action, hxs

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        return (self._is_exploration_done[batch_idx] > 0.0).to(masks.device)

    def get_state_description(self):
        return self.nav_skill.get_state_description()

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleExploreSkill.

        :return: List of argument types.
        """
        return [ROOM]
