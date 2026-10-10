import numpy as np
import pytest

from owac.evaluation.pi05_diagnostics import action_metrics, rotation_error_degrees


def test_rotation_metric_respects_equivalent_pi_representations():
    assert rotation_error_degrees(np.array([[np.pi, 0, 0]]), np.array([[-np.pi, 0, 0]]))[
        0
    ] == pytest.approx(0)
    assert rotation_error_degrees(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, np.pi / 2]]))[
        0
    ] == pytest.approx(90)


def test_action_metrics_separate_physical_distance_from_gripper_sign():
    target = np.zeros((2, 7))
    target[:, 6] = [1, -1]
    predicted = target.copy()
    predicted[:, :3] = [0.03, 0.04, 0]
    predicted[1, 6] = 0.1
    metrics = action_metrics(predicted, target)
    assert metrics["xyz_error_mean_m"] == pytest.approx(0.05)
    assert metrics["gripper_sign_accuracy"] == pytest.approx(0.5)
