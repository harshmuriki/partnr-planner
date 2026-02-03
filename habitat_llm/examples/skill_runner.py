#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import List, Tuple, Any


# append the path of the
# parent directory
sys.path.append("..")

import omegaconf
import hydra

from hydra.utils import instantiate

from habitat_llm.utils import cprint, setup_config, fix_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_llm.utils.sim import init_agents
from habitat_llm.examples.example_utils import execute_skill, DebugVideoUtil
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
    print_hierarchical_graph,
)


def print_object_states(world_graph, object_name: str = None):
    """
    Print the states of an object or all objects.
    
    :param world_graph: The active WorldGraph
    :param object_name: Optional specific object name. If None, shows all objects.
    """
    from habitat_llm.world_model import Object

    if object_name:
        cprint(f"Object State for: {object_name}", "green")
    else:
        cprint("All Object States:", "green")

    objects = world_graph.get_all_objects()

    for obj in objects:
        if object_name and obj.name != object_name:
            continue

        # Get states from properties
        states = obj.properties.get("states", {})

        if states:
            state_str = ", ".join([f"{k}={v}" for k, v in states.items()])
            cprint(f"{obj.name}: {state_str}", "yellow")
        else:
            # Check for direct state properties (backward compatibility)
            state_props = {k: v for k, v in obj.properties.items()
                          if k in ["is_powered_on", "is_filled", "is_clean"]}
            if state_props:
                state_str = ", ".join([f"{k}={v}" for k, v in state_props.items()])
                cprint(f"{obj.name}: {state_str}", "yellow")
            else:
                cprint(f"{obj.name}: No states available", "white")


def print_articulated_furniture(world_graph):
    """
    Print all articulated furniture (furniture that can be opened/closed).
    
    :param world_graph: The active WorldGraph
    """
    from habitat_llm.world_model import Furniture

    cprint("Articulated Furniture (can be opened/closed):", "green")

    all_furniture = world_graph.get_all_furnitures()
    print(f"Total furniture pieces in scene: {len(all_furniture)}")
    articulated_furniture = []

    for furniture in all_furniture:
        if furniture.is_articulated():
            articulated_furniture.append(furniture)

    if not articulated_furniture:
        cprint("  No articulated furniture found in the scene.", "yellow")
        return

    # Group by furniture type
    furniture_by_type = {}
    for furn in articulated_furniture:
        # Extract furniture type from name (e.g., "chest_of_drawers_1" -> "chest_of_drawers")
        parts = furn.name.rsplit('_', 1)
        furn_type = parts[0] if len(parts) > 1 and parts[1].isdigit() else furn.name

        if furn_type not in furniture_by_type:
            furniture_by_type[furn_type] = []
        furniture_by_type[furn_type].append(furn.name)

    # Display grouped by type
    total_count = 0
    for furn_type, furn_list in sorted(furniture_by_type.items()):
        cprint(f"\n  {furn_type.upper()} ({len(furn_list)} items):", "cyan")
        for furn_name in sorted(furn_list):
            cprint(f"    - {furn_name}", "white")
            total_count += 1

    cprint(f"\nTotal: {total_count} articulated furniture pieces", "green")


def print_gt_graph(env_interface):
    """
    Print the Ground Truth (GT) graph structure.
    
    :param env_interface: The EnvironmentInterface containing both CG and GT graphs
    """
    from habitat_llm.world_model import Room, Furniture, Object

    cprint("\n" + "=" * 80, "cyan")
    cprint("GROUND TRUTH (GT) GRAPH", "cyan")
    cprint("=" * 80, "cyan")

    # Get GT graph from perception
    if not hasattr(env_interface, 'perception') or env_interface.perception is None:
        cprint("No GT graph available (perception not initialized)", "red")
        return

    gt_graph = env_interface.perception.gt_graph

    # Print summary
    rooms = gt_graph.get_all_nodes_of_type(Room)
    furniture = gt_graph.get_all_nodes_of_type(Furniture)
    objects = gt_graph.get_all_nodes_of_type(Object)

    cprint(f"\nSummary:", "green")
    cprint(f"  Rooms: {len(rooms)}", "white")
    cprint(f"  Furniture: {len(furniture)}", "white")
    cprint(f"  Objects: {len(objects)}", "white")

    # Print hierarchical structure
    cprint(f"\nHierarchical Structure:", "green")
    for room in sorted(rooms, key=lambda r: r.name):
        cprint(f"\n  {room.name} ({room.properties.get('type', 'unknown')})", "cyan")

        # Get furniture in this room
        room_furniture = gt_graph.get_neighbors_of_type(room, Furniture)
        if room_furniture:
            cprint(f"    Furniture ({len(room_furniture)}):", "yellow")
            for furn in sorted(room_furniture, key=lambda f: f.name)[:10]:  # Limit to 10 per room
                furn_type = furn.properties.get('type', 'unknown')
                is_art = '(articulated)' if furn.is_articulated() else ''
                cprint(f"      - {furn.name} ({furn_type}) {is_art}", "white")
            if len(room_furniture) > 10:
                cprint(f"      ... and {len(room_furniture) - 10} more", "white")

        # Get objects in this room
        room_objects = gt_graph.get_neighbors_of_type(room, Object)
        if room_objects:
            cprint(f"    Objects ({len(room_objects)}):", "yellow")
            for obj in sorted(room_objects, key=lambda o: o.name)[:10]:  # Limit to 10 per room
                obj_type = obj.properties.get('type', 'unknown')
                cprint(f"      - {obj.name} ({obj_type})", "white")
            if len(room_objects) > 10:
                cprint(f"      ... and {len(room_objects) - 10} more", "white")

    cprint("\n" + "=" * 80, "cyan")

    # Print full GT graph string representation
    cprint("\nFull GT Graph String:", "green")
    cprint(gt_graph.to_string(), "white")
    cprint("=" * 80 + "\n", "cyan")


# Method to load agent planner from the config
@hydra.main(
    config_path="../conf", config_name="examples/skill_runner_default_config.yaml"
)
def run_skills(config: omegaconf.DictConfig) -> None:
    """
    The main function for executing the skill_runner tool. A default config is provided.
    See the `main` function for example CLI command to run the tool.

    :param config: input is a habitat-llm config from Hydra. Can contain CLI overrides.
    """
    fix_config(config)
    # Setup a seed
    seed = 47668090
    # Setup some hardcoded config overrides (e.g. the metadata path)
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
    config = setup_config(config, seed)

    assert config.env == "habitat", "Only valid for Habitat skill testing."

    # whether or not to show blocking videos after each command call
    show_command_videos = (
        config.skill_runner_show_videos
        if hasattr(config, "skill_runner_show_videos")
        else True
    )
    # make videos only if showing or saving them
    make_video = config.evaluation.save_video or show_command_videos

    if not make_video:
        remove_visual_sensors(config)

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # We register custom measures
    register_measures(config)

    # create the dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")
    # Initialize the environment interface for the agent
    # Note: init_wg=False because reset_environment() will be called below
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    ##########################################
    # select and initialize the desired episode by index or id
    # NOTE: use "+skill_runner_episode_index=2" in CLI to set the episode index ( e.g. episode 2)
    # NOTE: use "+skill_runner_episode_id=<id>" in CLI to set the episode id ( e.g. episode "")
    assert not (
        hasattr(config, "skill_runner_episode_index")
        and hasattr(config, "skill_runner_episode_id")
    ), "Episode selection options are mutually exclusive."
    if hasattr(config, "skill_runner_episode_index"):
        episode_index = config.skill_runner_episode_index
        print(f"Loading episode_index = {episode_index}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_index(
            episode_index
        )
    elif hasattr(config, "skill_runner_episode_id"):
        episode_id = config.skill_runner_episode_id
        print(f"Loading episode_id = {episode_id}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )
    env_interface.reset_environment()
    ###########################################

    # Initialize the planner
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)
    planner.reset()

    sim = env_interface.sim

    # Use the robot agent's world graph for displaying entities
    agent_uid = config.robot_agent_uid
    active_world_graph = env_interface.world_graph[agent_uid]

    # show the topdown map if requested
    if hasattr(config, "skill_runner_show_topdown"):
        dbv = DebugVisualizer(sim, config.paths.results_dir)
        dbv.create_dbv_agent(resolution=(1000, 1000))
        try:
            top_down_map = dbv.peek("stage")
            if show_command_videos:
                top_down_map.show()
            if config.evaluation.save_video:
                top_down_map.save(output_path=config.paths.results_dir, prefix="topdown")
        except (ValueError, RuntimeError) as e:
            cprint(
                f"Warning: Could not generate topdown map. Scene bounding box may be invalid. Error: {str(e)}",
                "yellow",
            )
        finally:
            dbv.remove_dbv_agent()
            dbv.create_dbv_agent()
            dbv.remove_dbv_agent()

    ############################
    # done with setup, prompt the user and start running skills

    # available skills
    skills = {
        "Navigate": "Navigate <agent_index> <entity_name> \n",
        "Explore": "Explore <agent_index> <room_name> \n",
        "Open": "Open <agent_index> <entity_name> \n",
        "Close": "Close <agent_index> <entity_name> \n",
        "Pick": "Pick <agent_index> <entity_name> \n",
        # Place skill requires 5 arguments, comma separated, no spaces:
        "Place": "Place <agent_index> <entity_name_0,relation_0,entity_name_1,relation_1,entity_name_2>. Eg (Place 0 cup_0,on,counter_22,None,None) \n",
        "Fill": "Fill <agent_index> <entity_name> \n",
        "Pour": "Pour <agent_index> <entity_name>. Eg (Pour 0 cup_1) Should pour from the container already held by the agent into cup_1. \n",
        "Clean": "Clean <agent_index> <entity_name>. Some objects require being near a faucet to clean. \n",
        "PowerOn": "PowerOn <agent_index> <entity_name> \n",
        "PowerOff": "PowerOff <agent_index> <entity_name> \n",
    }
    exit_skill = "exit"
    help_skill = "help"
    entity_skill = "entities"
    graph_skill = "graph"
    state_skill = "state"
    articulated_skill = "articulated"
    gt_graph_skill = "gt"
    pdb_skill = "debug"
    cumulative_video_skill = "make_video"

    cprint("Welcome to skill_runner!", "green")
    cprint(
        f"Current Episode (id=={sim.ep_info.episode_id}) is running in scene {sim.ep_info.scene_id} with info: {sim.ep_info.info}.",
        "green",
    )

    print_all_entities(active_world_graph)
    print_furniture_entity_handles(active_world_graph)
    print_object_entity_handles(active_world_graph)

    skills_str = "Available skills:\n" + "\n".join(
        [f"  {k}: {v.strip()}" for k, v in skills.items()]
    )
    help_text = (
        f"{skills_str}\n"
        "Type a skill to begin.\n"
        "Alternatively, type one of:\n"
        f"  '{exit_skill}' - exit the program\n"
        f"  '{help_skill}' - display help text\n"
        f"  '{entity_skill}' - display all available entities\n"
        f"  '{graph_skill}' - display hierarchical scene graph (rooms -> furniture -> receptacles -> objects)\n"
        f"  '{state_skill}' or 'state <object_name>' - display object states (is_filled, is_powered_on, is_clean)\n"
        f"  '{articulated_skill}' - display all articulated furniture (can be opened/closed)\n"
        f"  '{gt_graph_skill}' - display Ground Truth (GT) graph structure\n"
        f"  '{pdb_skill}' - enter pdb breakpoint for interactive debugging\n"
        f"  '{cumulative_video_skill}' - make a single cumulative video out of all individual command clips"
    )
    cprint(help_text, "green")

    # setup a sequence of commands to run immediately without manual input
    scripted_commands: List[str] = []
    if hasattr(config, "skill_runner_scripted_commands"):
        scripted_commands = config.skill_runner_scripted_commands
        # we need special handling for "Place" skill because arguements are comma separated and need to be joined
        place_indices = [i for i, x in enumerate(scripted_commands) if "Place" in x]
        for i, place_ix in enumerate(place_indices):
            corrected_ix = place_ix - i * 4  # account for removed elements
            for j in range(1, 5):
                # concat the elements
                scripted_commands[corrected_ix] += (
                    "," + scripted_commands[corrected_ix + j]
                )
            scripted_commands = (
                scripted_commands[: corrected_ix + 1]
                + scripted_commands[corrected_ix + 5 :]
            )
    print(scripted_commands)

    # collect debug frames to create a final video
    cumulative_frames: List[Any] = []

    command_index = 0
    # history of skill commands and their responses
    command_history: List[Tuple[str, str]] = []
    while True:
        cprint("Enter Command", "blue")
        if len(scripted_commands) > command_index:
            user_input = scripted_commands[command_index]
            print(user_input)
        else:
            user_input = input("> ")

        selected_skill = None

        if user_input == exit_skill:
            print("==========================")
            print("Exiting. Command History:")
            for ix, t in enumerate(command_history):
                print(f" [{ix}]: '{t[0]}' -> '{t[1]}'")
            print("==========================")
            exit()
        elif user_input == help_skill:
            cprint(help_text, "green")
        elif user_input == entity_skill:
            print_all_entities(active_world_graph)
        elif user_input == graph_skill:
            print_hierarchical_graph(active_world_graph)
            # from habitat_llm.llm.instruct.utils import (
            #     get_world_descr,
            # )
            # # Use planner's config if available, otherwise use defaults
            # add_state_info = False
            # centralized = True
            # if hasattr(planner, 'planner_config'):
            #     add_state_info = planner.planner_config.get('objects_response_include_states', False)
            #     centralized = planner.planner_config.get('centralized', False)

            # world_description = get_world_descr(
            #     active_world_graph,
            #     agent_uid=agent_uid,
            #     add_state_info=add_state_info,
            #     include_room_name=True,
            #     centralized=centralized,
            # )
            # print("\n" + "="*60)
            # print("WORLD DESCRIPTION (as sent to LLM):")
            # print("="*60)
            # print(world_description)
            # print("="*60 + "\n")
        elif user_input == state_skill:
            print_object_states(active_world_graph)
        elif user_input.startswith(state_skill + " "):
            object_name = user_input.split(" ", 1)[1].strip()
            print_object_states(active_world_graph, object_name)
        elif user_input == articulated_skill:
            print_articulated_furniture(active_world_graph)
        elif user_input == gt_graph_skill:
            print_gt_graph(env_interface)
        elif user_input == pdb_skill:
            # peek an entity
            dbv = DebugVisualizer(sim, config.paths.results_dir)
            dbv.create_dbv_agent()
            # NOTE: do debugging calls here
            # example to peek an entity: dbv.peek(env_interface.world_graph.get_node_from_name('table_50').sim_handle).show()
            breakpoint()
            dbv.remove_dbv_agent()
        elif user_input == cumulative_video_skill:
            # create a video of all accumulated frames thus far and play it
            if len(cumulative_frames) > 0:
                dvu = DebugVideoUtil(
                    env_interface, env_interface.conf.paths.results_dir
                )
                dvu.frames = cumulative_frames
                dvu._make_video(postfix="cumulative", play=show_command_videos)
        elif user_input in skills:
            # fill information piece by piece
            selected_skill = user_input
            # get the agent index
            agent_ix = input("Agent Index (0=robot, 1=human) = ")
            if agent_ix not in ["0", "1"]:
                cprint("... invalid Agent Index, aborting.", "red")
                continue
            target_entity_name = input("Target Entity = ")
        elif user_input.split(" ")[0].rstrip(",") in skills:
            # attempt to parse full skill definition from string
            skill_components = user_input.split(" ")
            if len(skill_components) < 3:
                cprint("... invalid command. Expected format: <skill> <agent_index> <target_entity>", "red")
                continue
            selected_skill = skill_components[0].rstrip(",")
            agent_ix = skill_components[1].rstrip(",")
            if agent_ix not in ["0", "1"]:
                cprint("... invalid Agent Index, aborting.", "red")
                continue
            # Join all remaining components as the target entity name
            # This handles cases where the target entity contains spaces or is comma-separated
            # For Place/Rearrange skills, the target_entity should be comma-separated: "object,relation,furniture,constraint,reference"
            target_entity_name = " ".join(skill_components[2:])
        else:
            cprint("... invalid command.", "red")

        # configure and run the skill
        if selected_skill is not None:
            high_level_skill_actions = {
                int(agent_ix): (selected_skill, target_entity_name, None)
            }

            ############################
            # run the skill
            try:
                responses, _, frames = execute_skill(
                    high_level_skill_actions,
                    planner,
                    vid_postfix=f"{command_index}_",
                    make_video=make_video,
                    play_video=show_command_videos,
                )
                command_history.append((user_input, responses[int(agent_ix)]))
                skill_name = high_level_skill_actions[int(agent_ix)][0]
                print(
                    f"{skill_name} completed. Response = '{responses[int(agent_ix)]}'"
                )
                cumulative_frames.extend(frames)
            except Exception as e:
                failure_string = f"Failed to execute skill with exception: {str(e)}"
                print(failure_string)
                command_history.append((user_input, failure_string))
        command_index += 1


##########################################
# CLI Example:
# HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir="."
# or
# python habitat_llm/examples/skill_runner.py
#
# NOTE: conf/examples/skill_runner_default_config.yaml is consumed to initialize parameters
##########################################
# Script Specific CLI overrides:
#
# (mutually exclusive)
# - '+skill_runner_episode_index=0' - initialize the episode with the specified index within the dataset
# - '+skill_runner_episode_id=' - initialize the episode with the specified "id" within the dataset
#
# - '+skill_runner_show_topdown=True' - (default False) show a topdown view of the scene upon initialization for context
#
# (output control options)
# - '+skill_runner_show_videos=False' - (default True) turn off showing videos immediately after running a command
# - 'evaluation.save_video=False' - (default True) option to save videos to files. Also affects cumulative videos produced with "make_video" command.
# NOTE: videos are made only if either of the above options are True
# - 'paths.results_dir=<relative_path>' (default './results/') relative path to desired output directory for evaluation
#
##########################################
# Other useful CLI overrides:
#
# - 'habitat.dataset.data_path="<path to dataset .json.gz>"' - set the desired episode dataset
#
if __name__ == "__main__":
    cprint(
        "\nStart of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )

    # Run the skills
    run_skills()

    cprint(
        "\nEnd of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )
