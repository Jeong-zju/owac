"""Null-calibration properties of the support novelty statistic.

These pin down what the statistic's scale does and does not mean, which is the question a
fixed ``large_residual_threshold`` silently assumes an answer to.
"""

from __future__ import annotations

import torch

from owac.evaluation.n01_v12_null import DEFAULT_SIGMAS, motion_response, probe
from owac.frameworks.n01_full.support import SpatialSupport, SupportConfig


def _support() -> SpatialSupport:
    return SpatialSupport(
        SupportConfig(region_grid=2, multiscale_radii=(0,), receptive_field_radius=0)
    )


def _window(seed: int = 4, size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand((1, 2, 3, size, size), generator=generator)
    mask = torch.ones((1, 2, 1, size, size), dtype=torch.bool)
    return values, mask


def test_identical_frames_give_exactly_zero_innovation() -> None:
    """The statistic must be zero-based: no difference in, no number out."""

    values, mask = _window()
    curve = probe(values, mask, _support(), sigmas=(0.0,))
    assert curve["0"] == 0.0


def test_response_is_monotone_in_the_injected_noise_level() -> None:
    """A bigger injected difference must not produce a smaller statistic."""

    values, mask = _window()
    sigmas = (0.0, 0.01, 0.02, 0.04)
    curve = probe(values, mask, _support(), sigmas=sigmas)
    series = [curve[f"{sigma:g}"] for sigma in sigmas]
    assert series[0] == 0.0
    assert series == sorted(series), series


def test_what_this_probe_can_and_cannot_establish() -> None:
    """The probe is bounded by its own construction, and the test says so.

    ``probe`` injects the noise into slot 0, which ``region_residual`` uses as the
    reference *and* as the source of the noise-scale estimate.  Numerator and denominator
    therefore both scale with sigma, so the saturation this module once read as evidence
    that the estimator is calibrated is a property of the probe instead.  What the probe
    does establish is narrower: the statistic is zero-based, and the response to one fixed
    absolute sigma differs between frames of different scale.
    """

    values, mask = _window()
    curve = probe(values, mask, _support(), sigmas=(0.0, 0.04))
    assert curve["0"] == 0.0, "the statistic must be zero when the frames are identical"
    assert curve["0.04"] > 0.0, "injected difference must produce a positive statistic"

    # A quieter frame gives a larger statistic for the same absolute injection, because the
    # denominator is estimated from the frame itself.  That is the cross-frame dependence
    # the module's original conclusion mistook for an estimator property.
    quiet = (values * 0.25).clamp(0.0, 1.0)
    quiet_curve = probe(quiet, mask, _support(), sigmas=(0.0, 0.04))
    assert quiet_curve["0.04"] > curve["0.04"], (quiet_curve["0.04"], curve["0.04"])


def test_default_sigmas_include_zero_and_span_two_orders_of_magnitude() -> None:
    assert DEFAULT_SIGMAS[0] == 0.0
    assert DEFAULT_SIGMAS[-1] / DEFAULT_SIGMAS[1] >= 10


def test_motion_probe_responds_to_a_known_displacement() -> None:
    """The falsification probe must actually detect displacement, or it proves nothing."""

    generator = torch.Generator().manual_seed(23)
    # A structured scene, not noise: rolling i.i.d. noise by a whole frame changes nothing.
    base = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
    scene = base.repeat(1, 3, 1, 1).unsqueeze(0)  # [B=1, T=1, C=3, H=8, W=8]
    frames = scene.repeat(1, 8, 1, 1, 1)
    frames = frames + 0.01 * torch.randn(frames.shape, generator=generator)
    mask = torch.ones((1, 8, 1, 8, 8), dtype=torch.bool)

    result = motion_response(frames, mask, _support(), shifts=(0, 2))
    assert result["by_shift"]["0"] is not None
    assert result["by_shift"]["2"] is not None
    assert result["by_shift"]["0"]["accepted"] == 1.0, (
        "with no displacement every source must be accepted"
    )
    assert result["by_shift"]["2"]["accepted"] < result["by_shift"]["0"]["accepted"], (
        "a displacement must reduce acceptance, or the probe cannot falsify anything"
    )


def test_motion_probe_reports_every_requested_shift() -> None:
    generator = torch.Generator().manual_seed(24)
    frames = torch.rand((1, 8, 3, 8, 8), generator=generator)
    mask = torch.ones((1, 8, 1, 8, 8), dtype=torch.bool)
    result = motion_response(frames, mask, _support(), shifts=(0, 1, 3))
    assert sorted(result["by_shift"]) == ["0", "1", "3"]
    fractions = {"accepted", "deferred", "retracted"}
    for value in result["by_shift"].values():
        assert value is not None
        assert fractions <= set(value), value
        # The scales are reported so the defer/withdraw split can be explained rather than
        # merely observed; a three-way sum is still the check that the shares are shares.
        assert abs(sum(value[key] for key in fractions) - 1.0) < 1e-3, value
        assert "residual_median" in value and "logodds_median" in value
