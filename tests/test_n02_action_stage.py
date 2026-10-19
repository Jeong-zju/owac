import numpy as np

from owac.evaluation.n02_full.action_stage import action_stages, correction_direction


def test_stage_requires_actual_closure_and_height_change_not_relative_time():
    a = np.zeros((100, 14))
    a[:, [6, 13]] = 1
    ee = np.zeros((100, 7))
    assert set(action_stages(a, ee, ee)) == {"approach"}
    a[30:40, [6, 13]] = 0.5
    a[40:, [6, 13]] = 0
    ee[60:, 2] = 0.03
    stages = action_stages(a, ee, ee)
    assert np.all(stages[:30] == "approach") and np.all(stages[30:60] == "grasp")
    assert np.all(stages[60:] == "lift")
    assert "lift" not in action_stages(a, ee, np.zeros_like(ee))


def test_direction_distinguishes_toward_but_overshooting_from_error_reduction():
    b = np.zeros((1, 2, 14))
    y = np.ones_like(b)
    p = np.ones_like(b) * 3
    mask = np.array([[True, False]])
    p[:, 1] = np.nan
    r = correction_direction(b, p, y, mask, np.ones(14))
    assert r["valid_steps"] == 1 and r["direction_toward_target_fraction"] == 1
    assert r["joint_error_reduced_fraction"] == 0
