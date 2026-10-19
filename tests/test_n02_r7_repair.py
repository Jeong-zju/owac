"""Spatial path must retain input geometry; auxiliary-only gradients cannot clip the readout."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.encoder_v12 import RawEncoderV12
from owac.frameworks.n02_full.readout_v12 import N02ReadoutV12
from owac.frameworks.n02_full.state import ObservationBundle
from owac.training.n02_v12 import N02V12Trainer


def spatial_fixture():
    torch.manual_seed(12)
    frame = MeasurementFrame(
        values=np.random.default_rng(12).random((1, 3, 20, 28)).astype("float32"),
        valid_mask=np.ones((1, 20, 28), bool),
        camera_id="head",
    )
    encoder = RawEncoderV12(
        camera_ids=("head",), dims=(8, 16, 32, 64), depths=(1, 1, 1, 1), d_model=8
    )
    spatial = encoder(ObservationBundle(frames={"head": frame}, context=EmbodimentContext()))
    readout = N02ReadoutV12(
        spatial_dims=encoder.spatial_dims,
        latent_dim=8,
        n_latents=4,
        n_queries=4,
        out_dim=16,
        num_heads=2,
        state_dim=8,
        num_bands=2,
        preserve_spatial_tokens=True,
    )
    return encoder, spatial, readout


def test_spatial_readout_keeps_positions_and_gradients():
    encoder, spatial, readout = spatial_fixture()
    tokens, _ = readout(
        features=spatial, state_tokens=torch.randn(1, 2, 3, 8), support=torch.ones(1, 2) / 2
    )
    assert tokens.std(dim=1).mean() > 0.1
    (tokens * torch.randn_like(tokens)).sum().backward()
    assert readout.spatial_skip.weight.grad.abs().sum() > 0
    assert next(encoder.parameters()).grad.abs().sum() > 0


@pytest.mark.parametrize("invalid_value", [1e6, float("nan")])
def test_spatial_sampling_ignores_padded_values(invalid_value):
    _, spatial, readout = spatial_fixture()
    before = readout.spatial_output_tokens(spatial)
    for key, value in spatial.levels.items():
        invalid = ~spatial.valid[key].reshape(1, 1, *value.shape[-2:])
        spatial.levels[key] = value.masked_fill(invalid, invalid_value)
    after = readout.spatial_output_tokens(spatial)
    torch.testing.assert_close(before, after)


def test_auxiliary_group_norm_cannot_shrink_readout_gradient():
    action = torch.nn.Parameter(torch.zeros(2))
    auxiliary = torch.nn.Parameter(torch.zeros(2))
    action.grad = torch.tensor([3.0, 4.0])
    auxiliary.grad = torch.tensor([10000.0, 0.0])
    trainer = object.__new__(N02V12Trainer)
    trainer.plan = SimpleNamespace(config={"optim": {"clip_scope": "parameter_group"}})
    trainer.grad_clip = 1.0
    trainer.optimizer = torch.optim.AdamW(
        [{"params": [action], "name": "readout"}, {"params": [auxiliary], "name": "decoder"}]
    )
    trainer.clip_gradients()
    torch.testing.assert_close(action.grad, torch.tensor([0.6, 0.8]))
    assert trainer.last_clip_norms["decoder"] == 10000.0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inactive_auxiliary_group_norm_is_device_independent(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    action = torch.nn.Parameter(torch.zeros(2, device=device))
    auxiliary = torch.nn.Parameter(torch.zeros(2, device=device))
    action.grad = torch.tensor([3.0, 4.0], device=device)
    trainer = object.__new__(N02V12Trainer)
    trainer.plan = SimpleNamespace(config={"optim": {"clip_scope": "parameter_group"}})
    trainer.grad_clip = 1.0
    trainer.optimizer = torch.optim.AdamW(
        [{"params": [action], "name": "readout"}, {"params": [auxiliary], "name": "decoder"}]
    )
    assert trainer.clip_gradients() == 5.0
    assert trainer.last_clip_norms["decoder"] == 0.0


def test_fixed_dev_selection_skips_all_padding_and_records_it(monkeypatch):
    from owac.training import n02_v12

    trainer = object.__new__(N02V12Trainer)
    source = SimpleNamespace(
        anchor_count=6,
        targets=lambda start, stop: {"action_mask": np.array([[start not in (2, 4)]])},
        anchor=lambda index: {"frame": index},
    )
    trainer.sources = {"dev": source}
    trainer.plan = SimpleNamespace(dev_windows=2, config={"data": {"data_root": "unused"}})
    trainer.window = lambda src, start, stop: SimpleNamespace(anchors=[{"relative_path": "x"}])
    receipts = {}
    trainer._write_json = lambda name, data: receipts.update({name: data})
    monkeypatch.setattr(n02_v12, "read_task_prompts", lambda paths, root: ["task"])
    windows = trainer.dev_batch()
    assert len(windows) == 2
    assert receipts["dev_windows.json"]["indices"] == [3, 5]
    assert receipts["dev_windows.json"]["skipped_all_padding"] == [2, 4]
    assert all(targets["action_mask"].any() for _, targets, _ in windows)
    source.targets = lambda start, stop: {"action_mask": np.array([[False]])}
    with pytest.raises(ValueError, match="too few"):
        trainer.dev_batch()
