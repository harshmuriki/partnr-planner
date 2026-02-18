# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import attr
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode,
)
from habitat.datasets.utils import check_and_gen_physics_config

import habitat_llm.agent.env.evaluation.evaluation_functions as evaluation_functions
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    EvaluationConstraint,
    EvaluationProposition,
    EvaluationPropositionDependency,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _merge_initial_state_object_states_into_episode(episode: Dict[str, Any]) -> None:
    """
    Merge object_states from info.initial_state[] into the top-level episode["object_states"]
    so that states set in the episode editor (e.g. is_powered_on in a placement spec) are
    applied by the sim and reflected in the GT/CG graphs.
    The sim loads rigid_objs sorted by config path; we build the same handle->index mapping.
    """
    rigid_objs = episode.get("rigid_objs") or []
    info = episode.get("info") or {}
    initial_state = info.get("initial_state") or []
    if not rigid_objs or not initial_state:
        return

    # Placement entries: have object_classes and a placement target (furniture/region)
    def is_placement_entry(e: dict) -> bool:
        if "object_classes" not in e:
            return False
        return "furniture_names" in e or "allowed_regions" in e

    placement_entries = [e for e in initial_state if is_placement_entry(e)]
    if not placement_entries:
        return

    # Build handles in the same order as the sim (sorted by config path)
    sorted_objs = sorted(rigid_objs, key=lambda x: x[0])
    obj_counts = {}
    handles_sorted = []
    for obj_handle, _ in sorted_objs:
        config_path = obj_handle if isinstance(obj_handle, str) else obj_handle
        base = config_path.split(".")[0]
        c = obj_counts.get(config_path, 0)
        obj_counts[config_path] = c + 1
        handles_sorted.append(base + f"_:{c:04d}")

    # Map file index k -> handle (sim uses sorted order; same config path gets handles by occurrence)
    file_index_to_handle = []
    for k in range(len(rigid_objs)):
        config_path = rigid_objs[k][0] if rigid_objs[k] else None
        if config_path is None:
            file_index_to_handle.append(None)
            continue
        count_before = sum(1 for i in range(k) if (rigid_objs[i][0] if rigid_objs[i] else None) == config_path)
        sorted_positions = [j for j in range(len(sorted_objs)) if sorted_objs[j][0] == config_path]
        if count_before < len(sorted_positions):
            j = sorted_positions[count_before]
            file_index_to_handle.append(handles_sorted[j])
        else:
            file_index_to_handle.append(None)

    if episode.get("object_states") is None:
        episode["object_states"] = {}

    for i, entry in enumerate(placement_entries):
        if i >= len(file_index_to_handle):
            break
        states = entry.get("object_states")
        if not states or not isinstance(states, dict):
            continue
        handle = file_index_to_handle[i]
        if handle is None:
            continue
        for state_name, value in states.items():
            if state_name not in episode["object_states"]:
                episode["object_states"][state_name] = {}
            episode["object_states"][state_name][handle] = value


@attr.s(auto_attribs=True, kw_only=True)
class CollaborationEpisode(RearrangeEpisode):
    """Specifies additional instruction and evaluation data for a particular instance of a collaboration task.

    For a definition of inherited keys, see RearrangeEpisode.

    :property instruction: the textual instruction provided to the agents performing the task.
    :property evaluation_propositions: Contains the propositions in dictionary format.
    :property evaluation_proposition_dependencies: A list of EvaluationPropositionDependency
        where a dependency establishes that a proposition will not be considered for
        satisfaction unless a "depends_on" proposition has some particular satisfaction state.
    :property evaluation_constraints: A list of EvaluationConstraint where a constraint
        is applied over propositions. Examples include temporal constraints and tied
        quantification. Defaults to empty.
    :property object_states: A map of object state unique identifier strings to object instance handles and their desired initial states.
    """

    instruction: str = ""
    evaluation_propositions: List[EvaluationProposition] = attr.ib(factory=list)
    evaluation_proposition_dependencies: List[
        EvaluationPropositionDependency
    ] = attr.ib(factory=list)
    evaluation_constraints: List[EvaluationConstraint] = attr.ib(factory=list)
    object_states: Dict[str, Dict[str, Any]] = attr.ib(factory=dict)


@registry.register_dataset(name="CollaborationDataset-v0")
class CollaborationDatasetV0(RearrangeDatasetV0):
    episodes: List[CollaborationEpisode]

    def __init__(
        self,
        config: Optional["DictConfig"] = None,
        episodes: List[CollaborationEpisode] = None,
    ) -> None:
        self.config = config
        if episodes is not None:
            # If the episodes are given, init the dataset with these episodes.
            # We do this so that we can load a partial dataset.
            self.episodes = episodes
        else:
            # Otherwise, init the dataset with the episode specified in config
            if config and not self.check_config_paths_exist(config):
                data_p = config.data_path.format(split=config.split)
                scenes_p = config.scenes_dir
                raise ValueError(
                    f"Collaboration task assets are not downloaded locally. Either {data_p} or {scenes_p} do not exist."
                )

            check_and_gen_physics_config()
            super(RearrangeDatasetV0, self).__init__(config)

    def apply_scene_dir_prefix(
        self, episode: CollaborationEpisode, scenes_dir: Optional[str] = None
    ) -> CollaborationEpisode:
        """Overrides the scene directory to `scene_dataset_config` if provided."""
        if not scenes_dir:
            return episode

        episode.scene_dataset_config = os.path.join(
            scenes_dir, os.path.basename(episode.scene_dataset_config)
        )
        return episode

    def to_json(self) -> str:
        """Serializes the current dataset into a string JSON representation."""
        tmp_cfg = self.config
        self.config = None
        result = DatasetFloatJSONEncoder().encode(self)
        self.config = tmp_cfg
        return result

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        """Deserializes a dataset from a string JSON representation."""
        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            _merge_initial_state_object_states_into_episode(episode)
            collaboration_ep = CollaborationEpisode(**episode)

            for i, prop in enumerate(collaboration_ep.evaluation_propositions):
                collaboration_ep.evaluation_propositions[i] = EvaluationProposition(
                    **prop
                )  # type: ignore

            for i, dep in enumerate(
                collaboration_ep.evaluation_proposition_dependencies
            ):
                collaboration_ep.evaluation_proposition_dependencies[
                    i
                ] = EvaluationPropositionDependency(
                    **dep
                )  # type: ignore

            for i, constraint in enumerate(collaboration_ep.evaluation_constraints):
                constraint_cls = getattr(evaluation_functions, constraint["type"])  # type: ignore
                collaboration_ep.evaluation_constraints[i] = constraint_cls(
                    **constraint["args"]  # type: ignore
                )

            collaboration_ep = self.apply_scene_dir_prefix(collaboration_ep, scenes_dir)
            self.episodes.append(collaboration_ep)

    def from_binary(
        self, data_dict: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> None:
        """Load the dataset from a pickle compatible Dict."""
        all_T = data_dict["all_transforms"]
        idx_to_name = data_dict["idx_to_name"]
        for ep in data_dict["all_eps"]:
            ep["rigid_objs"] = [
                [idx_to_name[ni], all_T[ti]] for ni, ti in ep["rigid_objs"]
            ]
            ep["ao_states"] = {idx_to_name[ni]: v for ni, v in ep["ao_states"].items()}
            ep["name_to_receptacle"] = {
                idx_to_name[k]: idx_to_name[v] for k, v in ep["name_to_receptacle"]
            }

            new_markers = []
            for name, mtype, offset, link, obj in ep["markers"]:
                new_markers.append(
                    {
                        "name": idx_to_name[name],
                        "type": idx_to_name[mtype],
                        "params": {
                            "offset": offset,
                            "link": idx_to_name[link],
                            "object": idx_to_name[obj],
                        },
                    }
                )
            ep["markers"] = new_markers

            collaboration_ep = CollaborationEpisode(**ep)
            collaboration_ep = self.apply_scene_dir_prefix(collaboration_ep, scenes_dir)
            self.episodes.append(collaboration_ep)
