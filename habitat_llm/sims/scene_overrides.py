"""Project-local fixes for HSSD scenes, applied when a scene is loaded.

The downloaded dataset under data/hssd-hab stays unmodified: each fix lives here and
applies only to the scenes listed in SCENE_OVERRIDES.
"""
import os
from pathlib import Path

import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle

OVERRIDE_DATA = Path(__file__).resolve().parent / "scene_override_data"

SCENE_103997895_FRIDGE = "ec6d929832a895b456fd6cca745a18c192c41f4b_:0000"

SCENE_OVERRIDES = {
    "103997895_171031182": {
        # fridge_0 (Wardrobe0001 model) has no annotated default link, so Open, Close
        # and is_open used the computed default: the bottom drawer (link 6), which
        # cannot hold the task jug. Link 3 is the right door of the main section.
        "default_links": {SCENE_103997895_FRIDGE: 3},
        # The dataset filter file, except fridge_0's lower_cabinet is an active
        # "within" receptacle and its bottom-drawer receptacle is manually filtered.
        "rec_filter_file": OVERRIDE_DATA / "103997895_171031182.rec_filter.json",
    },
}


def scene_id_from_handle(scene_handle: str) -> str:
    """Scene id from a scene handle or scene instance path."""
    return os.path.basename(str(scene_handle)).split(".")[0]


def get_rec_filter_filepath(mm, scene_handle: str):
    """Receptacle filter file for a scene, preferring the project override."""
    override = SCENE_OVERRIDES.get(scene_id_from_handle(scene_handle), {}).get(
        "rec_filter_file"
    )
    if override is not None:
        return str(override)
    return hab_receptacle.get_scene_rec_filter_filepath(mm, scene_handle)


def apply_default_link_overrides(sim) -> None:
    """Annotate default links on the loaded scene's articulated objects."""
    scene_id = scene_id_from_handle(sim.curr_scene_name)
    links = SCENE_OVERRIDES.get(scene_id, {}).get("default_links", {})
    aom = sim.get_articulated_object_manager()
    for handle, link in links.items():
        ao = aom.get_object_by_handle(handle)
        if ao is None:
            raise ValueError(
                f"Scene override for {scene_id} references missing articulated object {handle}"
            )
        ao.user_attributes.set("default_link", link)
