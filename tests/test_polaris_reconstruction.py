import numpy as np
import pytest

from owac.runtime.polaris_reconstruction import carry_anchors, settled_approach_layout


def test_scene_anchors_exclude_dry_closes_but_preserve_regrasp():
    actions = np.zeros((12, 8))
    actions[[1, 4, 5, 8, 9], 7] = 1
    tcp = np.zeros((12, 3))
    tcp[2, 0] = 0.02
    tcp[6, 0] = 0.3
    tcp[10, 0] = 0.4
    anchors, cycles = carry_anchors(actions, tcp)
    assert anchors == (4, 10)
    assert [c["close"] for c in cycles] == [1, 4, 8]
    assert np.array_equal(actions[[1, 4, 5, 8, 9], 7], np.ones(5))


def test_dry_gripper_test_cannot_be_presented_as_a_carry():
    actions = np.zeros((5, 8))
    actions[1:3, 7] = 1
    with pytest.raises(ValueError, match="dry close"):
        carry_anchors(actions, np.zeros((5, 3)))


def test_support_plane_and_readiness_follow_approach_without_editing_input():
    config = {
        "cube_size": 0.04,
        "table_top": 0.13,
        "cube_initial_position": [0.4, 0.1, 0.15],
        "target_center": [0.4, -0.1, 0.13],
        "grasp_assistance": {"cycles": [{"close": 1, "open": 7, "object": "cube"}]},
    }
    tcp = np.zeros((8, 3))
    tcp[:, 2] = [0.2, 0.15, 0.14, 0.135, 0.14, 0.17, 0.2, 0.3]
    updated = settled_approach_layout(config, tcp)
    assert updated["table_top"] == pytest.approx(0.115)
    assert updated["cube_initial_position"] == pytest.approx([0.4, 0.1, 0.136])
    assert updated["grasp_assistance"]["cycles"][0]["not_before"] == 3
    assert config["table_top"] == 0.13
    assert "not_before" not in config["grasp_assistance"]["cycles"][0]
