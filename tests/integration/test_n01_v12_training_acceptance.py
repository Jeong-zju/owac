"""Training correctness: interruption, validation isolation and resume identity."""

from __future__ import annotations

import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from owac.frameworks.n01_full.model import N01V12Config
from owac.training import n01_v12 as training


@pytest.mark.parametrize("initial_mode", [True, False])
def test_validation_restores_mode_and_all_rng_streams_on_error(initial_mode):
    model = nn.Linear(2, 2).train(initial_mode)
    before = training._rng_state()
    with pytest.raises(RuntimeError, match="validation failed"):
        with training._validation_state(model):
            assert not model.training
            random.random()
            np.random.rand()
            torch.rand(3)
            if torch.cuda.is_available():
                torch.rand(3, device="cuda")
            raise RuntimeError("validation failed")
    actual = (random.random(), np.random.rand(), torch.rand(3))
    actual_cuda = torch.rand(3, device="cuda") if torch.cuda.is_available() else None
    training._restore_rng(before)
    assert actual[:2] == (random.random(), np.random.rand())
    assert torch.equal(actual[2], torch.rand(3))
    if actual_cuda is not None:
        assert torch.equal(actual_cuda, torch.rand(3, device="cuda"))
    assert model.training is initial_mode


def test_legacy_and_changed_contract_are_rejected():
    with pytest.raises(ValueError, match="legacy checkpoint"):
        training._check_resume({"step": 500, "torch_rng": torch.get_rng_state()}, {})
    with pytest.raises(ValueError, match="contract differs"):
        training._check_resume(
            {"resume_schema": "n01-v12-training-v2", "training_contract": {"accum": 4}},
            {"accum": 2},
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_interrupted_training_matches_uninterrupted_across_epoch(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA resume checked on the remote GPU")
    # Real AdamW updates and checkpoint serialization; only expensive data/backend IO
    # is replaced. Four optimizer updates cross two five-sample epochs at accum=3.
    from owac.raw import rawvla_sequence
    from owac.training import n01_action

    seen = []

    class Dataset:
        def __init__(self, region):
            self.region = region

        paths = ["episode-a", "episode-b"]
        protocol = Namespace(action_dim=1)

        def __len__(self):
            return 5

        def __getitem__(self, index):
            if self.region == "train":
                seen.append(index)
            return {
                "index": index,
                "action": torch.tensor([[[float(index)]]]),
                "action_mask": torch.ones(1, 1, dtype=torch.bool),
            }

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(1, 2)
            self.bridge = nn.Linear(2, 1)

        def parameter_report(self):
            return {}

    class Bridge:
        def __init__(self, model):
            self.model = model

        def forward_from_observation_with_features(self, obs, tokens, actions, **kwargs):
            value = self.model.bridge(self.model.encoder(tokens))
            return {"v_t": value, "u_t": actions + torch.rand_like(actions)}

    monkeypatch.setattr(training, "_protocol_path", lambda *args: Path("unused"))
    monkeypatch.setattr(training, "_build_datasets", lambda *args: Dataset(args[3]))
    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: Model().to(device))
    monkeypatch.setattr(training, "_weights_and_tokenizer", lambda *args: (None, None))
    monkeypatch.setattr(training, "_v12_bridge", lambda model, *args: Bridge(model))
    monkeypatch.setattr(
        training,
        "encode_windows_v12",
        lambda model, batch, *args, **kwargs: (
            torch.tensor([[[float(batch["index"])]]], device=device),
            {},
        ),
    )
    monkeypatch.setattr(rawvla_sequence, "collate_sequence_samples", lambda samples: samples[0])
    monkeypatch.setattr(n01_action, "load_backend", lambda *args: (Namespace(action_dim=1), {}))
    monkeypatch.setattr(n01_action, "observation_from_batch", lambda *args: None)
    monkeypatch.setattr(
        n01_action,
        "action_masked_loss",
        lambda prediction, target, *args: (prediction - target).square().mean(),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(__import__("json").dumps(N01V12Config().to_dict()))

    def run(name, steps, validation_every=0):
        root = tmp_path / name
        args = Namespace(
            repo_root=tmp_path,
            deploy_root=tmp_path,
            run_root=root,
            output=root / "output",
            config=config_path,
            device=device,
            protocol=None,
            gpu_index=None,
            diagnostic=True,
            max_steps=steps,
            lr=1e-3,
            weight_decay=0.0,
            resume=None,
            accum=3,
            crop=0,
            tbptt=8,
            log_every=1,
            val_every=validation_every,
            val_samples=1,
            ckpt_every=1,
        )
        training.train(args)
        return training._load_checkpoint(root / "output/checkpoints/latest.pt")

    full = run("full", 4)
    full_seen = list(seen)
    seen.clear()
    run("resumed", 2, validation_every=1)
    resumed = run("resumed", 4, validation_every=1)
    assert seen == full_seen
    assert resumed["samples_consumed"] == full["samples_consumed"] == 12
    for key, value in full["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    assert torch.equal(full["torch_rng"], resumed["torch_rng"])
    for key, state in full["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed["optimizer"]["state"][key][name])


def test_incomplete_model_and_top_left_crop_require_diagnostic_label():
    model = Namespace(encoder=Namespace(motion_provider=None))
    with pytest.raises(ValueError, match="top-left diagnostic crop"):
        training._require_training_scope(Namespace(diagnostic=False, crop=128), model)
    with pytest.raises(ValueError, match="motion correspondence is not connected"):
        training._require_training_scope(Namespace(diagnostic=False, crop=0), model)
    training._require_training_scope(Namespace(diagnostic=True, crop=128), model)
