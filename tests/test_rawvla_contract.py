"""Numerical and episode-boundary checks for the external RawVLA runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

from owac.backends.rawvla import load_audited
from owac.runtime.rawvla_robotwin import (
    ModelClient,
    normalize_state,
    reset_model,
    unnormalize_actions,
)


def test_robotwin_joint_order_and_zscore_preserve_gripper_values():
    mean = np.arange(14, dtype=np.float32)
    std = np.full(14, 0.25, dtype=np.float32)
    state = mean + np.arange(14, dtype=np.float32) / 10
    actual = normalize_state(state, {"mean": mean, "std": std})
    np.testing.assert_allclose(actual, (state - mean) / (std + 1e-8))
    assert actual[6] > 1  # No Qwen-style joint reorder, clipping or binarization.


def test_robotwin_action_unnormalization_keeps_left_and_right_grippers():
    mean = np.arange(14, dtype=np.float32)
    std = np.arange(1, 15, dtype=np.float32)
    normalized = np.ones((50, 32), dtype=np.float32)
    actual = unnormalize_actions(normalized, {"mean": mean, "std": std})
    np.testing.assert_allclose(actual, np.broadcast_to(mean + std, (50, 14)))
    assert actual[0, 6] == 13 and actual[0, 13] == 27


def test_robotwin_rejects_endpose_as_joint_state():
    with pytest.raises(ValueError, match="14 finite"):
        normalize_state(np.zeros(16), {"mean": np.zeros(14), "std": np.ones(14)})


def test_action_chunk_starts_at_actual_first_step_and_resets_between_episodes():
    class Client:
        calls = 0

        def predict_action(self, query):
            self.calls += 1
            actions = np.broadcast_to(np.arange(50)[:, None], (50, 14)).copy()
            return {"ok": True, "data": {"actions": actions}}

    model = ModelClient.__new__(ModelClient)
    model.client = Client()
    reset_model(model)
    assert model.step({}, 3)[0] == 0
    assert model.step({}, 52)[0] == 49
    assert model.step({}, 53)[0] == 0
    assert model.client.calls == 2
    reset_model(model)
    assert model.step({}, 0)[0] == 0
    assert model.client.calls == 3


class _Model:
    def state_dict(self):
        return {"action_head.action_out_proj.weight": np.zeros((14, 4))}

    def _normalize_state_dict(self, state):
        return state

    def _is_expected_openpi_missing_key(self, key):
        return False

    def load_state_dict(self, state, strict):
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"action_head.action_out_proj.weight": np.zeros((4, 14))},
        {"action_head.action_out_proj.weight": np.zeros((14, 4)), "unknown": np.zeros(1)},
    ],
)
def test_checkpoint_audit_rejects_missing_transposed_and_unknown_weights(weights):
    with pytest.raises(ValueError, match="does not match"):
        load_audited(_Model(), weights)


def test_checkpoint_audit_accepts_matching_readout():
    audit = load_audited(_Model(), {"action_head.action_out_proj.weight": np.zeros((14, 4))})
    assert audit["active_missing"] == audit["unexpected"] == audit["shape_mismatch"] == []
