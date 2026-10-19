"""Regression for the native state token layout, independent of encoder architecture."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.backends.rawvla_language import prepare_language_state
from owac.evaluation.n02_full.v12 import EvalError, _language_batch, load_checkpoint_into
from owac.training.n02_v12 import CHECKPOINT_FORMAT, N02V12Trainer


class Recorder:
    def __init__(self):
        self.states = []

    def tokenize(self, prompt, state=None):
        self.states.append(None if state is None else np.asarray(state).copy())
        width = 1 if state is None else len(state)
        return np.arange(width), np.ones(width, dtype=bool)


def policy(name="robotwin", pad=0):
    return SimpleNamespace(
        protocol=SimpleNamespace(name=name, max_state_dim=32),
        base=SimpleNamespace(
            tokenizer=Recorder(), state_pad_value=pad, action_head=SimpleNamespace(max_state_dim=32)
        ),
    )


def test_native_state_padding_precedes_tokenization_in_all_entrypoints():
    bridge = policy(pad=-2)
    measured = np.arange(14, dtype=np.float32)[None]
    trainer = object.__new__(N02V12Trainer)
    trainer.bridge, trainer.device = bridge, torch.device("cpu")
    training = trainer.language(["grab roller"], measured)
    evaluation = _language_batch(bridge, ["grab roller"], measured, None)
    serving = prepare_language_state(bridge, ["grab roller"], measured, device="cpu")
    assert torch.equal(training[0], evaluation[0]) and torch.equal(training[0], serving[0])
    assert training[0].shape == (1, 32)
    for state in bridge.base.tokenizer.states:
        np.testing.assert_array_equal(state[:14], measured[0])
        np.testing.assert_array_equal(state[14:], np.full(18, -2))
    assert torch.equal(training[2][0, 14:], torch.full((18,), -2))


def test_libero_retains_no_discrete_state_text():
    bridge = policy("libero")
    _, _, state = prepare_language_state(bridge, ["pick cube"], np.ones((1, 8)), device="cpu")
    assert bridge.base.tokenizer.states == [None]
    assert state.shape == (1, 32) and not state[:, 8:].any()


def test_language_rejects_missing_instruction_or_nonfinite_proprio():
    bridge = policy()
    with pytest.raises(ValueError, match="instruction"):
        prepare_language_state(bridge, [""], np.zeros((1, 14)), device="cpu")
    with pytest.raises(ValueError, match="finite proprio"):
        prepare_language_state(bridge, ["pick"], np.full((1, 14), np.nan), device="cpu")


def test_old_policy_inputs_cannot_silently_resume_or_evaluate_as_repaired(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "r5.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "execution_revision": "n02-v12-p2-train-r5",
            "model": model.state_dict(),
            "step": 900,
        },
        path,
    )
    trainer = object.__new__(N02V12Trainer)
    with pytest.raises(ValueError, match="legacy execution"):
        trainer.load_for_resume(path)
    with pytest.raises(EvalError, match="legacy checkpoint"):
        load_checkpoint_into(model, path)
    receipt = load_checkpoint_into(model, path, allow_legacy_weights_probe=True)
    assert receipt["weights_only_probe"] is True
    assert receipt["runtime_execution_revision"] == "n02-v12-p2-spatial-r7"
