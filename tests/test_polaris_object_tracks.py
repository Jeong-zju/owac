import numpy as np

from owac.runtime.polaris_object_tracks import infer_tracks


def test_object_tracking_keeps_regrasps_separate_from_new_object():
    actions = np.zeros((25, 8))
    states = np.zeros((25, 8))
    tcp = np.zeros((25, 3))
    for start, end, pickup, release in [(1, 6, 0.1, 0.3), (8, 13, 0.31, 0.5), (16, 21, 0.1, 0.5)]:
        actions[start:end, 7] = 1
        states[start + 1 : end, 7] = 0.5
        tcp[start, 0] = pickup
        tcp[end, 0] = release
    actions[23, 7] = 1  # A one-frame empty close must not create another object.
    result = infer_tracks(actions, states, tcp)
    assert len(result["candidate_objects"]) == 2
    assert result["candidate_objects"][0]["grasp_cycles"] == [[1, 6], [8, 13]]
    assert len(result["unresolved_grasps"]) == 1
    assert result["object_states_recovered"] is False
    assert result["object_count_visually_verified"] is False


def test_full_gripper_closure_does_not_prove_a_thin_source_object_is_absent():
    actions = np.zeros((8, 8))
    actions[1:6, 7] = 1
    states = actions.copy()
    tcp = np.zeros((8, 3))
    tcp[2:5, 2] = 0.1
    tcp[6, 0] = 0.2
    result = infer_tracks(actions, states, tcp)
    assert len(result["candidate_objects"]) == 1
    assert result["unresolved_grasps"][0]["reason"].startswith("fully closed")
