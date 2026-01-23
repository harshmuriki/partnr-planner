#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np

from habitat_llm.agent.env import EnvironmentInterface


class DebugVideoUtil:
    """
    This class provides an interface wrapper for creating, saving, and viewing third person videos of individual skill runs using the EnvironmentInterface API.

    For example, see `execute_skill` function below.
    NOTE: This code was largely adapted from the evaluation_runner.py
    """

    def __init__(
        self, env_interface_arg: EnvironmentInterface, output_dir: str
    ) -> None:
        """
        Construct the DebugVideoUtil instance from an EnvironmentInterface.

        :param env_interface_arg: The EnvironmentInterface instance.
        :param output_dir: The desired directory for saving output frames and videos.
        """

        self.env_interface = env_interface_arg

        # Declare container to store frames used for generating video
        # NOTE: For memory efficiency, we now write frames incrementally
        # This list is kept for backward compatibility but should remain empty
        self.frames: List[Any] = []

        self.output_dir = output_dir

        self.num_agents = 0
        for _agent_conf in self.env_interface.conf.evaluation.agents.values():
            self.num_agents += 1

        # Video writer for incremental writing (opened on first frame)
        self._video_writer = None
        self._video_file_path = None
        self._frame_count = 0

    def __get_combined_frames(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        For each agent, extract the observation from the "third_rgb" sensor and merge them into a single split-screen image.

        :param batch: A dict mapping observation names to values.
        :return: The composite image as a numpy array.
        """
        # Extract first agent frame
        images = []
        for obs_name, obs_value in batch.items():
            if "third_rgb" in obs_name:
                if self.num_agents == 1:
                    if "0" in obs_name or "main_agent" in obs_name:
                        images.append(obs_value)
                else:
                    images.append(obs_value)

        # Extract dimensions of the first image
        height, width = images[0].shape[1:3]

        # Create an empty canvas to hold the concatenated images
        concat_image = np.zeros((height, width * len(images), 3), dtype=np.uint8)

        # Iterate through the images and concatenate them horizontally
        for i, image in enumerate(images):
            concat_image[:, i * width : (i + 1) * width] = image.cpu()

        return concat_image

    def _store_for_video(
        self,
        observations: Dict[str, Any],
        hl_actions: Dict[int, Any],
        postfix: str = "",
        responses: Optional[Dict[int, str]] = None,
        thoughts: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Store a video with observations and text from an observation dict and an agent to action metadata dict.
        NOTE: Could probably go into utils?
        
        This method now writes frames incrementally to disk to avoid memory explosion.

        :param observations: A dict mapping observation names to values.
        :param hl_actions: A dict mapping agent action indices to actions.
        :param postfix: Optional postfix for the video file name (used to initialize writer).
        :param responses: Optional dict mapping agent indices to skill execution results.
        :param thoughts: Optional dict mapping agent indices to reasoning/thoughts.
        """
        frames_concat = self.__get_combined_frames(observations)
        frames_concat = np.ascontiguousarray(frames_concat)

        # Get frame dimensions for text placement
        height, width = frames_concat.shape[:2]
        y_offset = 30
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        max_y = height - 20  # Leave margin at bottom

        for idx, action in hl_actions.items():
            agent_name = "Human" if str(idx) == "1" else "Robot"
            y_pos = (int(idx) + 1) * y_offset

            # Display action
            action_text = f"{agent_name}: {action[0]}[{action[1]}]"
            frames_concat = cv2.putText(
                frames_concat,
                action_text,
                (20, y_pos),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

            # Display thought if available (truncate if too long)
            if thoughts and idx in thoughts and thoughts[idx]:
                thought = thoughts[idx]
                # Remove "Thought:" prefix if present
                thought = thought.replace("Thought:", "").strip()
                # Truncate if too long for display
                max_chars = min(80, width // 8)  # Approximate chars per line
                if len(thought) > max_chars:
                    thought = thought[:max_chars-3] + "..."
                y_pos += line_height
                # Check bounds
                if y_pos < max_y:
                    frames_concat = cv2.putText(
                        frames_concat,
                        f"  {thought}",
                        (20, y_pos),
                        font,
                        font_scale * 0.8,
                        (200, 200, 255),  # Light blue for thoughts
                        font_thickness - 1,
                    )

            # Display result/response if available
            if responses and idx in responses and responses[idx]:
                response = responses[idx].strip()
                # Truncate if too long
                max_chars = min(60, width // 8)
                if len(response) > max_chars:
                    response = response[:max_chars-3] + "..."
                y_pos += line_height
                # Check bounds
                if y_pos < max_y:
                    # Color based on success/failure
                    color = (0, 255, 0) if "Successful" in response or "success" in response.lower() else (0, 0, 255)
                    frames_concat = cv2.putText(
                        frames_concat,
                        f"  Result: {response}",
                        (20, y_pos),
                        font,
                        font_scale * 0.7,
                        color,
                        font_thickness - 1,
                    )

        # Initialize video writer on first frame if not already open
        if self._video_writer is None:
            # Use postfix if provided, otherwise generate a temporary name
            if not postfix:
                import time
                postfix = f"temp_{int(time.time())}"
            self._video_file_path = f"{self.output_dir}/videos/video-{postfix}.mp4"
            os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
            self._video_writer = imageio.get_writer(
                self._video_file_path,
                fps=30,
                quality=4,
            )

        # Write frame immediately to disk (incremental writing)
        self._video_writer.append_data(frames_concat)
        self._frame_count += 1

        # Keep frames list empty for memory efficiency (backward compatibility)
        # self.frames.append(frames_concat)  # Commented out to save memory
        return

    def _make_video(self, play: bool = True, postfix: str = "") -> None:
        """
        Makes a video from a pre-processed set of frames using imageio and saves it to the output directory.
        
        If frames were written incrementally (via _store_for_video), this just closes the writer.
        Otherwise, it writes all frames from self.frames (backward compatibility).

        :param play: Whether or not to play the video immediately.
        :param postfix: An optional postfix for the video file name.
        """
        # If writer is already open (incremental writing mode), just close it
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
            out_file = self._video_file_path
            print(f"Saved video to {out_file} ({self._frame_count} frames)")
            self._frame_count = 0
            self._video_file_path = None
        else:
            # Backward compatibility: write from self.frames if writer wasn't used
            out_file = f"{self.output_dir}/videos/video-{postfix}.mp4"
            print(f"Saving video to {out_file}")
            os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
            writer = imageio.get_writer(
                out_file,
                fps=30,
                quality=4,
            )
            for frame in self.frames:
                writer.append_data(frame)
            writer.close()
            print(f"Saved video to {out_file} ({len(self.frames)} frames)")

        if play:
            print("     ...playing video, press 'q' to continue...")
            self.play_video(out_file)

    def play_video(self, filename: str) -> None:
        """
        Play and loop video from a filepath with cv2.

        :param filename: The filepath of the video.
        """
        cap = cv2.VideoCapture(filename)
        last_time = time.time()
        while cap.isOpened():
            if time.time() - last_time > 1.0 / 30:
                last_time = time.time()
                ret, frame = cap.read()
                # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

                if ret:
                    cv2.imshow("Image", frame)
                else:
                    # looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def execute_skill(
    high_level_skill_actions: Dict[Any, Any],
    llm_env,
    make_video: bool = True,
    vid_postfix: str = "",
    play_video: bool = True,
) -> Tuple[Dict[Any, Any], Dict[Any, Any], List[Any]]:
    """
    Execute a high-level skill from a string (e.g. as produced by the planner).
    Can create and display a video of the running skill.

    :param high_level_skill_actions: The map of agent indices to actions. TODO: typing
    :param llm_env: The planner instance. TODO: typing
    :param make_video: whether or not to create, save, and display a video of the skill.
    :param vid_postfix: An optional postfix for the video file. For example, the action name.
    :param play_video: Whether or not to immediately play the generated video.
    :return: A tuple with two dict(the first contains responses per-agent skill, the second contains the number of skill steps taken) and a list of frames.
    """
    dvu = DebugVideoUtil(
        llm_env.env_interface, llm_env.env_interface.conf.paths.results_dir
    )

    # Get the env observations
    observations = llm_env.env_interface.get_observations()
    agent_idx = list(high_level_skill_actions.keys())[0]
    skill_name = high_level_skill_actions[agent_idx][0]

    # Set up the variables
    skill_steps = 0
    max_skill_steps = 1500
    skill_done = None

    # While loop for executing skills
    while not skill_done:
        # Check if the maximum number of steps is reached
        assert (
            skill_steps < max_skill_steps
        ), f"Maximum number of steps reached: {skill_name} skill fails."

        # Get low level actions and responses
        low_level_actions, responses = llm_env.process_high_level_actions(
            high_level_skill_actions, observations
        )

        assert (
            len(low_level_actions) > 0
        ), f"No low level actions returned. Response: {responses.values()}"

        # Check if the agent finishes
        if any(responses.values()):
            skill_done = True

        # Get the observations
        obs, reward, done, info = llm_env.env_interface.step(low_level_actions)
        observations = llm_env.env_interface.parse_observations(obs)

        if make_video:
            # Initialize writer with postfix on first frame
            if dvu._video_writer is None:
                dvu._store_for_video(observations, high_level_skill_actions, postfix=vid_postfix)
            else:
                dvu._store_for_video(observations, high_level_skill_actions)

        # Increase steps
        skill_steps += 1

    if make_video and skill_steps > 1:
        dvu._make_video(postfix=vid_postfix, play=play_video)

    return responses, {"skill_steps": skill_steps}, dvu.frames
