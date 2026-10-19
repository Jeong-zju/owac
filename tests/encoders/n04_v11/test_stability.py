"""Counterexamples for the observed all-invalid flow and unstable latent dynamics."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import torch

from owac.baselines._vendor.rkn.cell import RKNCell
from owac.frameworks.n04.correspondence import RaftCorrespondenceLoss, RaftLossConfig, RawScale
from owac.frameworks.n04.encoder_v11 import N04V11Encoder
from owac.frameworks.n04.rkn_banded import band_layout, predict_banded, predict_dense_bounded


def test_resume_latest_selects_completed_step_and_never_silently_starts_fresh(tmp_path):
    import pytest

    script = Path(__file__).resolve().parents[3] / "scripts/n04_full/run_n04_codex.py"
    resolve = runpy.run_path(str(script))["resolve_resume"]
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path, "latest")
    for name in (
        "checkpoint-step0000008.pt",
        "checkpoint-step0000009.pt",
        "checkpoint-best.pt",
        "checkpoint-step0000010.pt.tmp",
    ):
        (tmp_path / name).touch()
    assert resolve(tmp_path, "latest").endswith("checkpoint-step0000009.pt")


def test_all_outside_flow_keeps_gradient_without_valid_matches():
    loss = RaftCorrespondenceLoss(scale=RawScale("unit"), config=RaftLossConfig(boundary=1))
    raw = torch.rand(1, 3, 32, 64)
    flow = torch.full((1, 2, 32, 64), 200.0, requires_grad=True)
    parts = loss(raw, raw, flow)
    assert parts.support == 0 and parts.source_support == 1
    assert parts.boundary > 1 and parts.total > 1
    parts.total.backward()
    assert torch.isfinite(flow.grad).all() and bool((flow.grad > 0).all())
    masked = loss(raw, raw, flow, valid=torch.zeros(1, 32, 64, dtype=torch.bool))
    assert masked.source_support == 0 and masked.total == 0


def test_geometric_distance_uses_xy_and_rectangular_sensor_units():
    raw = torch.rand(1, 3, 32, 64)
    flow = torch.zeros(1, 2, 32, 64)
    flow[:, 0] = 64
    loss = RaftCorrespondenceLoss(scale=RawScale("unit"), config=RaftLossConfig(boundary=1))
    expected = ((torch.arange(64).float() + 1) / 64).square().mean()
    torch.testing.assert_close(loss(raw, raw, flow).boundary, expected)


def test_latent_rms_is_scale_invariant_and_keeps_direction_gradients():
    helper = SimpleNamespace(config=SimpleNamespace(observation_normalization="rms"))
    latent = torch.randn(2, 256, 4, 8, requires_grad=True)
    normalized = N04V11Encoder.normalize_observation(helper, latent)
    torch.testing.assert_close(normalized.square().mean(1), torch.ones(2, 4, 8))
    torch.testing.assert_close(
        normalized, N04V11Encoder.normalize_observation(helper, latent * 1000)
    )
    normalized[:, 0].sum().backward()
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0


def make_cell():
    cfg = RKNCell.get_default_config()
    cfg.num_basis, cfg.bandwidth, cfg.trans_net_hidden_units = 3, 1, [12]
    cfg.finalize_modifying()
    return RKNCell(8, cfg)


def test_bounded_bands_match_full_dense_covariance_and_gradients():
    torch.manual_seed(114)
    cell = make_cell()
    with torch.no_grad():
        for basis in (cell._tm_11_basis, cell._tm_12_basis, cell._tm_21_basis, cell._tm_22_basis):
            basis.normal_(0, 0.5)
    mean = torch.randn(4, 16, requires_grad=True)
    upper = torch.rand(4, 8, requires_grad=True)
    lower = torch.rand(4, 8, requires_grad=True)
    side = torch.zeros(4, 8, requires_grad=True)
    blocks, noise = cell.get_transition_model(mean)
    matrix = torch.cat((torch.cat(blocks[:2], -1), torch.cat(blocks[2:], -1)), -2)
    bound = (matrix.abs().sum(-2).amax(-1) * matrix.abs().sum(-1).amax(-1)).sqrt()
    matrix = matrix * (0.95 / bound).clamp_max(1)[:, None, None]
    assert torch.linalg.matrix_norm(matrix, ord=2).amax() <= 0.95001
    covariance = torch.diag_embed(torch.cat((upper, lower), -1))
    # side is zero here, but keep its derivatives in the independent full-matrix reference.
    covariance = covariance + torch.nn.functional.pad(torch.diag_embed(side), (8, 0, 0, 8))
    covariance = covariance + torch.nn.functional.pad(torch.diag_embed(side), (0, 8, 8, 0))
    predicted_cov = matrix @ covariance @ matrix.transpose(-1, -2) + torch.diag_embed(
        noise.expand(4, -1)
    )
    reference = (
        (matrix @ mean[..., None])[..., 0],
        predicted_cov.diagonal(dim1=-2, dim2=-1)[:, :8],
        predicted_cov.diagonal(dim1=-2, dim2=-1)[:, 8:],
        predicted_cov[:, :8, 8:].diagonal(dim1=-2, dim2=-1),
    )
    banded = predict_banded(
        cell, mean, upper, lower, side, *band_layout(cell), max_operator_norm=0.95
    )
    dense = predict_dense_bounded(cell, mean, upper, lower, side, 0.95)
    for expected, actual, full in zip(reference, banded, dense, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual, full, rtol=2e-5, atol=2e-6)
    parameters = [mean, upper, lower, side, *cell.parameters()]
    grads = torch.autograd.grad(sum(v.square().mean() for v in reference), parameters)
    actual = torch.autograd.grad(sum(v.square().mean() for v in banded), parameters)
    for a, b in zip(grads, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


def test_expansive_diagonal_transition_cannot_exponentially_grow_covariance():
    cell = make_cell()
    with torch.no_grad():
        for basis in (cell._tm_11_basis, cell._tm_12_basis, cell._tm_21_basis, cell._tm_22_basis):
            basis.zero_()
        cell._tm_11_basis[:, cell._diag_idx] = 0.2
        cell._tm_22_basis[:, cell._diag_idx] = 0.2
    mean, upper, lower, side = (
        torch.ones(1, 16),
        torch.ones(1, 8),
        torch.ones(1, 8),
        torch.zeros(1, 8),
    )
    original = cell._predict(mean, [upper, lower, side])
    assert original[1][0].min() > 1.5
    with torch.no_grad():
        for _ in range(256):
            mean, upper, lower, side = predict_banded(
                cell, mean, upper, lower, side, *band_layout(cell), max_operator_norm=1.0
            )
    assert torch.isfinite(upper).all() and upper.max() < 27
    torch.testing.assert_close(mean, torch.ones_like(mean), rtol=1e-4, atol=1e-4)
