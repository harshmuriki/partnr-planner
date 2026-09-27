import numpy as np

from habitat_llm.llm.instruct.utils import (
    image_data_to_pil,
    matching_sensor_names,
    observation_rgb_to_data_url,
    observation_rgb_to_pil,
    ranked_robot_rgb_names,
)


def test_observation_rgb_to_data_url_encodes_head_camera():
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    url = observation_rgb_to_data_url({"agent_0_head_rgb": rgb}, agent_uid=0)
    assert url is not None
    assert url.startswith("data:image/png;base64,")
    image = image_data_to_pil(url)
    assert image.size == (12, 8)


def test_observation_rgb_to_data_url_missing_camera_returns_none():
    assert observation_rgb_to_data_url({"agent_0_head_depth": 1}) is None


def test_observation_rgb_to_pil_head_camera():
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    image = observation_rgb_to_pil({"agent_0_head_rgb": rgb}, agent_uid=0)
    assert image is not None
    assert image.size == (12, 8)


def test_observation_rgb_to_pil_unprefixed_head_camera():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[0, 0] = [0, 255, 0]
    image = observation_rgb_to_pil({"head_rgb": rgb}, agent_uid=0)
    assert image is not None
    assert image.size == (6, 4)


def test_observation_rgb_to_pil_ignores_other_agent_camera():
    other = np.zeros((8, 10, 3), dtype=np.uint8)
    other[:] = 9
    head = np.zeros((4, 6, 3), dtype=np.uint8)
    head[:] = 1
    image = observation_rgb_to_pil(
        {"agent_1_head_rgb": other, "agent_0_head_rgb": head},
        agent_uid=0,
    )
    assert image is not None
    assert image.size == (6, 4)


def test_matching_sensor_names_prefers_prefixed_robot_head():
    names = [
        "head_rgb",
        "agent_0_head_rgb",
        "agent_1_head_rgb",
        "agent_0_third_rgb",
    ]
    assert matching_sensor_names(names, 0, "head") == [
        "head_rgb",
        "agent_0_head_rgb",
    ]
    assert matching_sensor_names(names, 0, "third") == ["agent_0_third_rgb"]
    assert ranked_robot_rgb_names(names, 0)[0] == "agent_0_third_rgb"


def test_ranked_robot_rgb_names_prefers_third_over_head():
    names = [
        "agent_0_head_rgb",
        "agent_0_articulated_agent_jaw_rgb",
        "agent_0_third_rgb",
    ]
    ranked = ranked_robot_rgb_names(names, 0)
    assert ranked[0] == "agent_0_third_rgb"
    assert all("jaw" not in name for name in ranked)


def test_observation_rgb_to_pil_prefers_third_over_head():
    head = np.zeros((4, 6, 3), dtype=np.uint8)
    head[:] = 1
    third = np.zeros((8, 10, 3), dtype=np.uint8)
    third[:] = 2
    image = observation_rgb_to_pil(
        {"agent_0_third_rgb": third, "agent_0_head_rgb": head},
        agent_uid=0,
    )
    assert image is not None
    assert image.size == (10, 8)
