from types import SimpleNamespace
import numpy as np
import pytest

from habitat_llm.vlm_tamp.visible_boxes import boxes_from_mask, temporary_instance_ids


def test_boxes_only_cover_visible_pixels_and_exclude_hidden_or_tiny_objects():
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[3:7, 5:10] = 100
    mask[0, 0] = 101
    boxes = boxes_from_mask(mask, [(100, 'visible', 'red'),
                                  (101, 'tiny', 'blue'), (102, 'behind_wall', 'green')])
    assert boxes == [('visible', 'red', 5., 3., 10., 7.)]


def test_instance_ids_restore_after_render_failure():
    nodes = [SimpleNamespace(semantic_id=15), SimpleNamespace(semantic_id=16)]
    obj = SimpleNamespace(visual_scene_nodes=nodes)
    with pytest.raises(RuntimeError):
        with temporary_instance_ids([(1000000000, obj)]):
            assert all(n.semantic_id == 1000000000 for n in nodes)
            raise RuntimeError('render failed')
    assert [n.semantic_id for n in nodes] == [15, 16]
