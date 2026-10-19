"""Repair invariants: dead features stay dead and scale cannot erase geometry."""

import pytest
import torch

from owac.frameworks.n02_full.representation_v12 import (
    RawPatchHead,
    local_vicreg,
    scale_rms,
    spatial_contrast,
)


def test_spatial_contrast_rejects_dc_shortcut_and_invalid_nan():
    x = torch.randn(2, 20, 8, requires_grad=True)
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[:, -3:] = False
    corrupted = x.masked_fill(~valid.unsqueeze(-1), float("nan"))
    out = spatial_contrast(corrupted, valid)
    expected = spatial_contrast(x * 1000 + 3000, valid)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    assert (out[:, -3:] == 0).all() and torch.isfinite(out).all()
    assert spatial_contrast(torch.ones_like(x)).abs().max() == 0
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_local_vicreg_penalizes_constant_and_redundant_features():
    generator = torch.Generator().manual_seed(42)
    diverse = torch.randn(1024, 8, generator=generator)
    collapsed = torch.ones_like(diverse, requires_grad=True)
    good, _ = local_vicreg(diverse, diverse)
    bad, terms = local_vicreg(collapsed, collapsed)
    assert bad > good + 20
    assert terms["variance"] > 0.9
    bad.backward()
    assert torch.isfinite(collapsed.grad).all()
    with pytest.raises(ValueError):
        local_vicreg(diverse[:1], diverse[:1])
    correlated = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    _, terms = local_vicreg(correlated, correlated)
    assert terms["covariance"] == 1  # Mean across views; official sum convention is 2.


def test_scale_rms_preserves_global_content_and_finite_empty_support():
    x = torch.ones(2, 5, 8, requires_grad=True)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1] = False
    out = scale_rms(x.masked_fill(~valid.unsqueeze(-1), float("nan")), valid)
    torch.testing.assert_close(out[0], torch.ones_like(out[0]))
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_raw_patch_head_trains_on_unpadded_raw_support():
    raw = torch.randn(2, 3, 32, 48)
    features = torch.randn(2, 8, 3, 3, requires_grad=True)
    head = RawPatchHead(8)
    loss = head(features, raw)
    loss.backward()
    assert torch.isfinite(loss) and features.grad.abs().sum() > 0
    assert features.grad[:, :, -1].abs().sum() == 0


@pytest.mark.parametrize("mode", ["contrast", "rms"])
def test_balanced_fusion_resists_huge_deep_dc_and_scale(mode):
    from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
    from owac.frameworks.n02_full.encoder_v12 import RawEncoderV12
    from owac.frameworks.n02_full.readout_v12 import N02ReadoutV12
    from owac.frameworks.n02_full.state import ObservationBundle

    torch.manual_seed(12)
    encoder = RawEncoderV12(
        camera_ids=("head",), dims=(8, 16, 32, 64), depths=(1, 1, 1, 1), d_model=8
    )
    spatial = encoder(
        ObservationBundle(
            frames={"head": MeasurementFrame(values=torch.rand(1, 3, 20, 28), camera_id="head")},
            context=EmbodimentContext(),
        )
    )
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
        scale_balance=True,
        scale_balance_mode=mode,
    )
    before = readout.spatial_output_tokens(spatial)
    spatial.levels["stride16"] = spatial.levels["stride16"] * 10000
    if mode == "contrast":
        spatial.levels["stride16"] = spatial.levels["stride16"] + 3000
    after = readout.spatial_output_tokens(spatial)
    torch.testing.assert_close(before, after, atol=0.01, rtol=0.01)
    if mode == "rms":
        # A scene-wide channel vector is useful information, not automatically a defect.
        spatial.levels["stride16"] += torch.randn(1, 8, 1, 1) * 10000
        changed_global = readout.spatial_output_tokens(spatial)
        assert (changed_global - after).norm() / after.norm() > 0.1
        # Even a very large constant deep level cannot erase the shallow image response.
        spatial.levels["stride16"] = torch.ones_like(spatial.levels["stride16"]) * 1e8
        after = readout.spatial_output_tokens(spatial)
    spatial.levels["stride4"] = spatial.levels["stride4"].flip(-1)
    perturbed = readout.spatial_output_tokens(spatial)
    assert (perturbed - after).norm() / after.norm() > 0.1


def test_raw_npz_pool_preserves_camera_and_channel_axes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from owac.training import n02_v12_representation as training

    raw = np.broadcast_to(np.array([0.1, 0.2, 0.3], dtype=np.float32), (1, 240, 320, 3)).copy()
    np.savez(
        tmp_path / "episode.npz",
        **{
            c + "_camera_raw_float32": raw * (i + 1)
            for i, c in enumerate(("head", "left", "right"))
        },
    )
    source = SimpleNamespace(
        anchor_count=1,
        anchor=lambda _: SimpleNamespace(relative_path="episode.npz", current_frame_index=0),
        identity=lambda: {},
    )
    monkeypatch.setattr(training, "build_source", lambda *a, **k: source)
    pool, _, _ = training.frame_pool({"data": {"data_root": str(tmp_path)}}, "train")
    assert pool.shape == (1, 3, 3, 240, 320)
    torch.testing.assert_close(pool[0, 1, :, 17, 22], torch.tensor([0.2, 0.4, 0.6]))
