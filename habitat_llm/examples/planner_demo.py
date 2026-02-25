#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import sys
import time
import os
import traceback
import json
import shutil
from pathlib import Path
from omegaconf import OmegaConf, open_dict

from scripts import view_trace_logs
from habitat_llm.world_model import SpotRobot


# append the path of the
# parent directory
sys.path.append("..")

import hydra
from typing import Dict

from torch import multiprocessing as mp

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
)

from habitat_llm.utils import cprint, setup_config, fix_config


from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
    DecentralizedEvaluationRunner,
    EvaluationRunner,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_baselines.utils.info_dict import extract_scalars_from_info


def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.paths.results_dir,
        dataset_file,
        "stats",
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    # Sort the dictionary by keys
    # Needed to ensure sanity in multi-process operation
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


def save_exception_message(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def save_success_message(config, env_interface, info):
    output_file = get_output_file(config, env_interface)
    failure_dict = {"success": True, "stats": json.dumps(info)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


# Write the config file into the results folders
def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))

    # Copy over the RLM config
    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
        # Centralized
        if "plan_config" in config.evaluation.planner is not None:
            planner_configs = [config.evaluation.planner.plan_config]
            suffixes = [""]
    else:
        for agent_name in config.evaluation.agents:
            suffixes.append(f"_{agent_name}")
            planner_configs.append(
                config.evaluation.agents[agent_name].planner.plan_config
            )

    for plan_config, suffix_rlm in zip(planner_configs, suffixes):
        if "llm" in plan_config and "serverdir" in plan_config.llm:
            yaml_rlm_path = plan_config.llm.serverdir
            if len(yaml_rlm_path) > 0:
                yaml_rlm_file = f"{yaml_rlm_path}/config.yaml"
                if os.path.isfile(yaml_rlm_file):
                    shutil.copy(
                        yaml_rlm_file, f"{output_file}/config_rlm{suffix_rlm}.yaml"
                    )


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def run_eval(config):
    fix_config(config)
    # Setup a seed
    # seed = 48212516
    seed = 47668090
    t0 = time.time()
    # Setup config
    config = setup_config(config, seed)

    # Normalize dataset path: if folder, pick .json.gz / .json and optional .yaml
    if hasattr(config, "habitat") and hasattr(config.habitat, "dataset"):
        data_path = getattr(config.habitat.dataset, "data_path", None)
        if data_path is not None:
            data_path_str = str(data_path)
            if os.path.isdir(data_path_str):
                folder = Path(data_path_str)
                json_gz_files = sorted(folder.glob("*.json.gz"))
                json_files = sorted(folder.glob("*.json"))
                chosen_dataset = None
                if json_gz_files:
                    chosen_dataset = json_gz_files[0]
                elif json_files:
                    chosen_dataset = json_files[0]
                if chosen_dataset is not None:
                    with open_dict(config):
                        config.habitat.dataset.data_path = str(chosen_dataset)
                    data_path_str = str(chosen_dataset)
                if (not hasattr(config, "runtime_config_path") or not config.runtime_config_path):
                    yaml_files = sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")))
                    if yaml_files:
                        with open_dict(config):
                            config.runtime_config_path = str(yaml_files[0])
            if data_path_str.endswith(".json") and not data_path_str.endswith(".json.gz"):
                with open_dict(config):
                    config.habitat.dataset.data_path = data_path_str + ".gz"

    dataset = CollaborationDatasetV0(config.habitat.dataset)

    write_config(config)
    if config.get("resume", False):
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # stats_dir = os.path.join(config.paths.results_dir, dataset_file, "stats")
        plan_log_dir = os.path.join(
            config.paths.results_dir, dataset_file, "planner-log"
        )

        # Find incomplete episodes
        incomplete_episodes = []
        for episode in dataset.episodes:
            episode_id = episode.episode_id
            # stats_file = os.path.join(stats_dir, f"{episode_id}.json")
            planlog_file = os.path.join(
                plan_log_dir, f"planner-log-episode_{episode_id}_0.json"
            )
            if not os.path.exists(planlog_file):
                incomplete_episodes.append(episode)
        print(
            f"Resuming with {len(incomplete_episodes)} incomplete episodes: {[e.episode_id for e in incomplete_episodes]}"
        )
        # Update dataset with only incomplete episodes
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=incomplete_episodes
        )

    # filter episodes by mod for running on multiple nodes
    if config.get("episode_mod_filter", None) is not None:
        rem, mod = config.episode_mod_filter
        episode_subset = [x for x in dataset.episodes if int(x.episode_id) % mod == rem]
        print(f"Mod filter: {rem}, {mod}")
        print(f"Episodes: {[e.episode_id for e in episode_subset]}")
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )

    num_episodes = len(dataset.episodes)
    if config.num_proc == 1:
        if config.get("episode_indices", None) is not None:
            if config.get("resume", False):
                raise ValueError("episode_indices and resume cannot be used together")
            episode_subset = [dataset.episodes[x] for x in config.episode_indices]
            dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )
        run_planner(config, dataset)
    else:
        # Process episodes in parallel
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []
        config.num_proc = min(config.num_proc, num_episodes)
        ochunk_size = num_episodes // config.num_proc
        # Prepare chunked datasets
        chunked_datasets = []
        # TODO: we may want to chunk by scene
        start = 0
        for i in range(config.num_proc):
            chunk_size = ochunk_size
            if i < (num_episodes % config.num_proc):
                chunk_size += 1
            end = min(start + chunk_size, num_episodes)
            indices = slice(start, end)
            chunked_datasets.append(indices)
            start += chunk_size

        for episode_index_chunk in chunked_datasets:
            episode_subset = dataset.episodes[episode_index_chunk]
            new_dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )

            parent_conn, child_conn = mp_ctx.Pipe()
            proc_args = (config, new_dataset, child_conn)
            p = mp_ctx.Process(target=run_planner, args=proc_args)
            p.start()
            proc_infos.append((parent_conn, p))
            print("START PROCESS")

        # Get back info
        all_stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }
        for conn, proc in proc_infos:
            stats_episodes = conn.recv()
            for run_id, stats_run in stats_episodes.items():
                all_stats_episodes[str(run_id)].update(stats_run)
            proc.join()

        all_metrics = aggregate_measures(
            {run_id: aggregate_measures(v) for run_id, v in all_stats_episodes.items()}
        )
        cprint("\n---------------------------------", "blue")
        cprint("Metrics Across All Runs:", "blue")
        for k, v in all_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("\n---------------------------------", "blue")
        # Write aggregated results across experiment
        write_to_csv(config.paths.end_result_file_path, all_metrics)

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")


def run_planner(config, dataset: CollaborationDatasetV0 = None, conn=None):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # Setup interface with the simulator if the planner depends on it
    if config.env == "habitat":
        # Remove sensors if we are not saving video

        # TODO: have a flag for this, or some check
        keep_rgb = False
        if "use_rgb" in config.evaluation:
            keep_rgb = config.evaluation.use_rgb
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)

        # TODO: Can we move this inside the EnvironmentInterface?
        # We register the dynamic habitat sensors
        register_sensors(config)
        # We register custom actions
        register_actions(config)
        # We register custom measures
        register_measures(config)

        # Initialize the environment interface for the agent
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
            if config.evaluation.log_data:
                save_exception_message(config, env_interface)
    else:
        env_interface = None

    # Instantiate the agent planner
    eval_runner: EvaluationRunner = None
    if config.evaluation.type == "centralized":
        eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)
    elif config.evaluation.type == "decentralized":
        eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    else:
        cprint(
            "Invalid planner type. Please select between 'centralized' or 'decentralized'. Exiting",
            "red",
        )
        return

    # Print the planner
    cprint(f"Successfully constructed the '{config.evaluation.type}' planner!", "green")
    print(eval_runner)

    # Declare observability mode
    cprint(
        f"Partial observability is set to: '{config.world_model.partial_obs}'", "green"
    )

    # Print the agent list
    print("\nAgent List:")
    print(eval_runner.agent_list)

    # Print the agent description
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # Try to get model info from different config structures
    try:
        if hasattr(config, "planner") and hasattr(config.planner, "llm"):
            model_target = config.planner.llm.llm._target_
        elif hasattr(config, "planner") and hasattr(config.planner, "vlm"):
            model_target = config.planner.vlm.vlm._target_
        else:
            # Decentralized config structure
            agent_config = config.evaluation.agents.agent_0.planner.plan_config
            if hasattr(agent_config, "vlm"):
                model_target = agent_config.vlm.vlm._target_
            elif hasattr(agent_config, "llm"):
                model_target = agent_config.llm.llm._target_
            else:
                model_target = "Unknown"
        cprint(f"Model: {model_target}", "blue")
    except Exception:
        cprint("Model: Unable to determine planner model", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    # Print whether GT object locations or RGB-D observations are being used
    use_gt_locs = getattr(config.world_model, 'use_gt_object_locations', False)
    if use_gt_locs:
        cprint("Perception: Using GT (ground truth) object locations from simulator", "green")
    else:
        cprint("Perception: Using RGB-D observations for object detection and localization", "green")
    cprint("---------------------------------------\n", "blue")

    os.makedirs(config.paths.results_dir, exist_ok=True)

    # Set up image save directory for VLM planners
    vlm_images_dir = os.path.join(config.paths.results_dir, "vlm_images")
    # Handle both centralized (single planner) and decentralized (dict of planners)
    planners_to_check = []
    if hasattr(eval_runner, "planner"):
        if isinstance(eval_runner.planner, dict):
            planners_to_check = list(eval_runner.planner.values())
        else:
            planners_to_check = [eval_runner.planner]
    for planner in planners_to_check:
        if hasattr(planner, "set_image_save_dir"):
            planner.set_image_save_dir(vlm_images_dir)
            # cprint(f"VLM images will be saved to: {vlm_images_dir}", "green")

    # Run the planner
    # Initialize stats_episodes for both CLI and non-CLI modes
    stats_episodes: Dict[str, Dict] = {
        str(i): {} for i in range(config.num_runs_per_episode)
    }

    if config.mode == "cli":
        # Get instruction from config if provided, otherwise use episode instruction
        if config.instruction:
            instruction = config.instruction
        else:
            # Get instruction from current episode
            instruction = env_interface.env.env.env._env.current_episode.instruction

        # Ensure videos are saved in CLI mode, even if task fails
        # if not config.evaluation.save_video:
        #     config.evaluation.save_video = True
        #     cprint("Enabling video saving for CLI mode", "yellow")

        # Load runtime configuration if provided (for robot location and runtime objects)
        runtime_config = None
        if hasattr(config, 'runtime_config_path') and config.runtime_config_path:
            import yaml
            runtime_config_path = Path(config.runtime_config_path)
            if runtime_config_path.exists():
                cprint(f"\n📋 Loading runtime configuration from: {runtime_config_path}", "cyan")
                with open(runtime_config_path, 'r') as f:
                    runtime_config = yaml.safe_load(f)
                cprint(f"✓ Runtime config loaded successfully", "green")

                # Override instruction with task from runtime config if present
                if runtime_config and 'task' in runtime_config and runtime_config['task']:
                    instruction = runtime_config['task']
                    cprint(f"✓ Instruction overridden from runtime config 'task'", "green")
            else:
                cprint(f"⚠ Runtime config file not found: {runtime_config_path}", "yellow")

        cprint(f'\nExecuting instruction: "{instruction}"', "blue")

        # Apply runtime modifications in partial observability mode
        if config.world_model.partial_obs and runtime_config:
            agent_graph = env_interface.world_graph.get(config.robot_agent_uid)
            if agent_graph is not None:
                # 1. Move robot to specified room if provided
                if 'runtime_robot_location' in runtime_config and 'room' in runtime_config['runtime_robot_location']:
                    target_room = runtime_config['runtime_robot_location']['room']
                    cprint(f"\n🤖 Moving robot to {target_room}...", "cyan")
                    success = agent_graph.move_robot_to_room(target_room, verbose=True, sim=env_interface.sim)
                    if success:
                        robot_nodes = agent_graph.get_all_nodes_of_type(SpotRobot)
                        if robot_nodes:
                            robot_pos = robot_nodes[0].properties.get('translation', [0, 0, 0])
                            cprint(f"✓ Robot moved to {target_room} at position {robot_pos}", "green")
                    else:
                        cprint(f"⚠ Failed to move robot to {target_room}", "yellow")

                # 2. Add runtime objects if provided
                if 'runtime_objects' in runtime_config:
                    runtime_objs = runtime_config['runtime_objects']
                    if runtime_objs.get('enabled', False) and 'objects' in runtime_objs:
                        cprint(f"\n🍾 Adding {len(runtime_objs['objects'])} runtime object(s) to scene graph...", "cyan")
                        for obj_config in runtime_objs['objects']:
                            obj_handle = obj_config.get('handle')
                            obj_class = obj_config.get('class')
                            furniture_name = obj_config.get('furniture_name')
                            position = obj_config.get('position')
                            rotation = obj_config.get('rotation')

                            if not obj_class:
                                cprint(f"⚠ Skipping object: 'class' not specified", "yellow")
                                continue

                            try:
                                if furniture_name:
                                    # Place on furniture
                                    added_obj = agent_graph.add_object_to_graph(
                                        object_class=obj_class,
                                        furniture_name=furniture_name,
                                        connect_to_entities=True,
                                        verbose=True
                                    )
                                    cprint(f"✓ Added {added_obj.name} (class: {obj_class}) on {furniture_name} at {added_obj.properties.get('translation')}", "green")
                                elif position:
                                    # Place at absolute position
                                    added_obj = agent_graph.add_object_to_graph(
                                        object_class=obj_class,
                                        position=position,
                                        rotation=rotation if rotation else [0, 0, 0, 1],
                                        connect_to_entities=True,
                                        verbose=True
                                    )
                                    cprint(f"✓ Added {added_obj.name} (class: {obj_class}) at position {position}", "green")
                                else:
                                    cprint(f"⚠ Skipping {obj_class}: neither 'furniture_name' nor 'position' specified", "yellow")
                            except Exception as e:
                                cprint(f"✗ Failed to add {obj_class}: {e}", "red")
            else:
                cprint("⚠ Agent world graph not available, skipping runtime modifications", "yellow")

        # Print the robot's scene graph before episode starts
        cprint("\nRobot Scene Graph:", "magenta")
        cprint("=" * 80, "magenta")
        agent_graph = env_interface.world_graph.get(config.robot_agent_uid)
        if agent_graph is not None:
            agent_graph.display_hierarchy()
        else:
            cprint("Could not access robot world graph", "yellow")
        cprint("=" * 80 + "\n", "magenta")

        try:
            # ! Important call to eval method that runs the planner
            info = eval_runner.run_instruction(instruction)

            # Print action counts and simulation steps
            if "action_counts" in info and info["action_counts"]:
                cprint("\n---------------------------------", "cyan")
                cprint("Action Counts (times selected):", "cyan")
                for action_name, count in sorted(info["action_counts"].items()):
                    cprint(f"  {action_name}: {count}", "cyan")
                cprint("---------------------------------", "cyan")
            if "action_sim_steps" in info and info["action_sim_steps"]:
                cprint("\n---------------------------------", "cyan")
                cprint("Action Simulation Steps (total steps per action):", "cyan")
                for action_name, steps in sorted(info["action_sim_steps"].items()):
                    cprint(f"  {action_name}: {steps}", "cyan")
                cprint("---------------------------------", "cyan")

            # Store action counts for HTML generation
            last_info = info.copy()
        except Exception as e:
            print("An error occurred inside of this method:", e)
            # Ensure video is saved even if exception occurs
            if config.evaluation.save_video and hasattr(eval_runner, 'dvu') and len(eval_runner.dvu.frames) > 0:
                try:
                    eval_runner.dvu._make_video(play=False, postfix=eval_runner.episode_filename)
                except Exception as video_error:
                    print(f"Warning: Failed to save video after exception: {video_error}")

    else:
        num_episodes = len(env_interface.env.episodes)
        last_info = None  # Track last episode's info for HTML generation
        for run_id in range(config.num_runs_per_episode):
            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id

                # Get instruction
                instruction = env_interface.env.env.env._env.current_episode.instruction
                print("\n\nEpisode", episode_id)

                try:
                    info = eval_runner.run_instruction(
                        output_name=f"episode_{episode_id}_{run_id}"
                    )

                    # Store last info for HTML generation
                    last_info = info

                    # Print action counts and simulation steps
                    if "action_counts" in info and info["action_counts"]:
                        cprint("\n---------------------------------", "cyan")
                        cprint(f"Action Counts (times selected) For Run {run_id} Episode {episode_id}:", "cyan")
                        for action_name, count in sorted(info["action_counts"].items()):
                            cprint(f"  {action_name}: {count}", "cyan")
                        cprint("---------------------------------", "cyan")
                    if "action_sim_steps" in info and info["action_sim_steps"]:
                        cprint("\n---------------------------------", "cyan")
                        cprint(f"Action Simulation Steps (total steps per action) For Run {run_id} Episode {episode_id}:", "cyan")
                        for action_name, steps in sorted(info["action_sim_steps"].items()):
                            cprint(f"  {action_name}: {steps}", "cyan")
                        cprint("---------------------------------", "cyan")

                    info_episode = {
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                    }
                    stats_keys = {
                        "task_percent_complete",
                        "task_state_success",
                        "sim_step_count",
                        "replanning_count",
                        "runtime",
                    }

                    # add replanning counts to stats_keys as scalars if replanning_count is a dict
                    if "replanning_count" in info and isinstance(
                        info["replanning_count"], dict
                    ):
                        for agent_id, replan_count in info["replanning_count"].items():
                            stats_keys.add(f"replanning_count_{agent_id}")
                            info[f"replanning_count_{agent_id}"] = replan_count

                    stats_episode = extract_scalars_from_info(
                        info, ignore_keys=info.keys() - stats_keys
                    )
                    stats_episodes[str(run_id)][episode_id] = stats_episode

                    cprint("\n---------------------------------", "blue")
                    cprint(f"Metrics For Run {run_id} Episode {episode_id}:", "blue")
                    for k, v in stats_episodes[str(run_id)][episode_id].items():
                        cprint(f"{k}: {v:.3f}", "blue")
                    cprint("\n---------------------------------", "blue")
                    # Log results onto a CSV
                    epi_metrics = stats_episodes[str(run_id)][episode_id] | info_episode
                    if config.evaluation.log_data:
                        save_success_message(config, env_interface, stats_episode)
                    write_to_csv(config.paths.epi_result_file_path, epi_metrics)
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while running the episode:", e)
                    print(f"Skipping evaluating episode: {episode_id}")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                try:
                    # Reset env_interface (moves onto the next episode in the dataset)
                    # env_interface.reset_environment()
                    # ! We never call this because we are only running one episode at a time
                    pass
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while resetting the env_interface:", e)
                    print("Skipping evaluating episode.")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                # Reset evaluation runner
                eval_runner.reset()

            # aggregate metrics across the current run.
            run_metrics = aggregate_measures(stats_episodes[str(run_id)])
            cprint("\n---------------------------------", "blue")
            cprint(f"Metrics For Run {run_id}:", "blue")
            for k, v in run_metrics.items():
                cprint(f"{k}: {v:.3f}", "blue")
            cprint("\n---------------------------------", "blue")

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)

    # Generate HTML visualization of trace logs
    try:
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # Remove .json.gz extension to match output_dir structure
        if dataset_file.endswith('.json.gz'):
            dataset_file = dataset_file[:-8]
        elif dataset_file.endswith('.json'):
            dataset_file = dataset_file[:-5]

        # Construct the traces directory path
        traces_dir = os.path.join(
            config.paths.results_dir,
            dataset_file,
            "traces",
            "0"
        )

        # Find the first .txt file in the traces directory
        trace_file_path = None
        if os.path.exists(traces_dir):
            txt_files = [f for f in os.listdir(traces_dir) if f.endswith('.txt')]
            if txt_files:
                trace_file_path = os.path.join(traces_dir, txt_files[0])

        if trace_file_path and os.path.exists(trace_file_path):
            cprint(f"\n📊 Generating HTML trace visualization...", "cyan")
            trace_data = view_trace_logs.parse_trace_file(trace_file_path)
            html_output = str(Path(trace_file_path).with_suffix('.html'))
            # Pass action counts, runtime, and LLM requests from info if available
            action_counts = None
            action_sim_steps = None
            runtime = None
            llm_requests = None
            if 'last_info' in locals() and last_info is not None:
                action_counts = last_info.get("action_counts")
                action_sim_steps = last_info.get("action_sim_steps")
                runtime = last_info.get("runtime")
                # replanning_count maps agent IDs to LLM request counts
                llm_requests = last_info.get("replanning_count")
                print(f"Action counts: {action_counts}")
                print(f"Action simulation steps: {action_sim_steps}")
            view_trace_logs.generate_html(trace_data, html_output, action_counts=action_counts, action_sim_steps=action_sim_steps, runtime=runtime, llm_requests=llm_requests)
            cprint(f"✓ HTML trace file generated at:", "green")
            cprint(f"  {html_output}", "green")
        else:
            cprint(f"⚠ Trace file not found in directory: {traces_dir}", "yellow")
    except Exception as trace_error:
        cprint(f"⚠ Failed to generate trace HTML: {trace_error}", "yellow")

    # aggregate metrics across all runs.
    if conn is None:
        all_metrics = aggregate_measures(
            {run_id: aggregate_measures(v) for run_id, v in stats_episodes.items()}
        )
        cprint("\n---------------------------------", "blue")
        cprint("Metrics Across All Runs:", "blue")
        for k, v in all_metrics.items():
            cprint(f"{k}: {v:.3f}", "blue")
        cprint("\n---------------------------------", "blue")
        # Write aggregated results across experiment
        write_to_csv(config.paths.end_result_file_path, all_metrics)
    else:
        conn.send(stats_episodes)

    env_interface.env.close()
    del env_interface

    if conn is not None:
        # Potentially we may want to send something

        conn.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run planner
    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )
