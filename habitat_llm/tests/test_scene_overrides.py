import json
from pathlib import Path

import pytest

from habitat_llm.sims.scene_overrides import (
    SCENE_103997895_FRIDGE,
    SCENE_OVERRIDES,
    get_rec_filter_filepath,
    scene_id_from_handle,
)

DATASET_FILTER = Path(
    "data/hssd-hab/scene_filter_files/103997895_171031182.rec_filter.json"
)


class NoFilterMetadataMediator:
    def get_scene_user_defined(self, scene_handle):
        return None


def test_scene_id_from_handle():
    assert scene_id_from_handle("103997895_171031182") == "103997895_171031182"
    assert (
        scene_id_from_handle(
            "data/hssd-hab/scenes-partnr-filtered/103997895_171031182.scene_instance.json"
        )
        == "103997895_171031182"
    )


def test_filter_path_prefers_override_and_falls_back():
    mm = NoFilterMetadataMediator()
    override = SCENE_OVERRIDES["103997895_171031182"]["rec_filter_file"]
    assert get_rec_filter_filepath(mm, "103997895_171031182") == str(override)
    assert get_rec_filter_filepath(mm, "106878915_174887025") is None


@pytest.mark.skipif(not DATASET_FILTER.exists(), reason="HSSD dataset not available")
def test_filter_override_only_changes_fridge_receptacles():
    """The override file must equal the dataset filter plus the documented fridge fix."""
    expected = json.loads(DATASET_FILTER.read_text())
    lower = f"{SCENE_103997895_FRIDGE}|Wardrobe0001_lower_cabinet_receptacle_mesh.0000"
    drawer = f"{SCENE_103997895_FRIDGE}|Wardrobe0001_door06_receptacle_mesh.0000"
    expected["access_filtered"].remove(lower)
    expected["active"].append(lower)
    expected["within_set"].append(lower)
    expected["active"].remove(drawer)
    expected["within_set"].remove(drawer)
    expected["manually_filtered"].append(drawer)
    override = SCENE_OVERRIDES["103997895_171031182"]["rec_filter_file"]
    assert json.loads(override.read_text()) == expected
