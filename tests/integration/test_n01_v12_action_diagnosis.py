"""Control metrics cannot be diluted by padded dimensions or invalid future steps."""

import numpy as np
import pytest

from owac.evaluation.n01_v12_action_diagnosis import action_metrics


def test_real_control_dimensions_and_mask():
    target = np.zeros((10, 7))
    predicted = np.zeros((10, 32))
    predicted[:5, :7] = 2
    predicted[:, 7:] = 999
    predicted[5:, :7] = 999
    result = action_metrics(predicted, target, np.arange(10) < 5)
    assert result["mse7"] == 4
    assert result["executed_first5_mae7"] == 2
    assert result["valid_steps"] == 5


def test_invalid_targets_are_not_a_score():
    with pytest.raises(ValueError):
        action_metrics(np.zeros((10, 32)), np.zeros((10, 7)), np.zeros(10, dtype=bool))
