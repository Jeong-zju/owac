"""N01 v1.2 P3 spatial support and temporal revision.

Two things are checked here and they are different:

1. ``TruncatedBOCPD`` reproduces the author's ``bocpd()`` numerical update on a
   short prefix.  The reference below is a transcription of
   ``DSM-bocd/bocpd.py`` (commit ``9d05117b``, MIT) with the author's
   ``GaussianUnknownMean`` predictive, kept in the test so the comparison is a
   real oracle rather than a shape assertion.
2. The support mechanism changes the content path and distinguishes an
   *edge exit* from a *source deletion*.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from owac.frameworks.n01_full.support import (
    SUPPORT_ACCEPTED,
    SUPPORT_INVALID,
    SUPPORT_RETRACTED,
    SpatialSupport,
    SupportConfig,
    TruncatedBOCPD,
)


def _logsumexp(values: np.ndarray) -> np.ndarray:
    """``scipy.special.logsumexp`` for the author reference, without the dependency."""

    peak = np.max(values)
    if not np.isfinite(peak):
        return peak
    return peak + np.log(np.sum(np.exp(values - peak)))


# ---------------------------------------------------------------------------
# Author reference: DSM-bocd/bocpd.py with GaussianUnknownMean.
# ---------------------------------------------------------------------------


class _AuthorGaussianUnknownMean:
    """``models.py::GaussianUnknownMean`` (Normal known variance, unknown mean)."""

    def __init__(self, mean0: float, var0: float, varx: float) -> None:
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0], dtype=np.float64)
        self.prec_params = np.array([1.0 / var0], dtype=np.float64)

    @property
    def var_params(self) -> np.ndarray:
        return 1.0 / self.prec_params + self.varx

    def log_pred_prob(self, t: int, data: np.ndarray, indices: np.ndarray) -> np.ndarray:
        x = data[t - 1]
        post_means = self.mean_params[indices]
        post_stds = np.sqrt(self.var_params[indices])
        z = (x - post_means) / post_stds
        return -0.5 * z * z - np.log(post_stds) - 0.5 * np.log(2.0 * np.pi)

    def update_params(self, t: int, data: np.ndarray) -> None:
        x = data[t - 1]
        new_prec = self.prec_params + (1.0 / self.varx)
        new_mean = (self.mean_params * self.prec_params + (x / self.varx)) / new_prec
        self.mean_params = np.append([self.mean0], new_mean)
        self.prec_params = np.append([1.0 / self.var0], new_prec)


def _author_bocpd(data: np.ndarray, hazard: float, model, keep: int) -> np.ndarray:
    """Transcription of ``bocpd.py::bocpd`` with a constant hazard."""

    length = len(data)
    log_r = -np.inf * np.ones((length + 1, length + 1))
    log_r[0, 0] = 0
    log_message = np.float64(0.0)
    max_indices = np.array([0])
    for t in range(1, length + 1):
        count = min(t, keep)
        log_h = np.log(np.full(count, hazard))
        log_1mh = np.log(1.0 - np.full(count, hazard))
        log_pred_prob = model.log_pred_prob(t, data, max_indices)
        log_growth = log_pred_prob + log_message + log_1mh
        log_cp = _logsumexp(log_pred_prob + log_message + log_h)
        new_log_joint = np.full(t + 1, -np.inf)
        new_log_joint[0] = log_cp
        new_log_joint[max_indices + 1] = log_growth
        max_indices = (-new_log_joint).argsort()[:keep]
        log_r[t, : t + 1] = new_log_joint
        log_r[t, : t + 1] -= _logsumexp(new_log_joint)
        model.update_params(t, data)
        log_message = new_log_joint[max_indices]
    return log_r


def _regime_series() -> np.ndarray:
    """A short real-shaped prefix: a stable segment, then a persistent shift."""

    generator = np.random.default_rng(101)
    stable = generator.normal(0.0, 1.0, size=8)
    shifted = generator.normal(4.0, 1.0, size=6)
    return np.concatenate([stable, shifted]).astype(np.float64)


def test_truncated_bocpd_reproduces_the_author_update() -> None:
    data = _regime_series()
    hazard = 1.0 / 8.0
    mean0, var0, varx = 0.0, 1.0, 1.0
    keep = 50  # no truncation for this length, so every index is kept

    reference = _author_bocpd(data, hazard, _AuthorGaussianUnknownMean(mean0, var0, varx), keep)

    config = SupportConfig(
        hazard=hazard,
        keep_run_lengths=keep,
        prior_mean=mean0,
        prior_var=var0,
        obs_var=varx,
        robust_omega=1.0 / (2.0 * varx),
        # A very wide bounded-influence scale turns the generalized-Bayes update
        # back into the author's plain Gaussian one, so this compares arithmetic
        # and not the robustness choice.
        robust_scale=1.0e6,
    )
    detector = TruncatedBOCPD(config, 1, dtype=torch.float64)
    posteriors = []
    for value in data:
        detector.update(torch.tensor([value], dtype=torch.float64))
        posteriors.append(detector.run_length_log_probability()[0].numpy())

    for t, got in enumerate(posteriors, start=1):
        expected = reference[t, : t + 1]
        assert np.isfinite(expected).sum() == len(expected)
        assert np.allclose(got, expected, atol=1e-8), (t, got, expected)


def test_changepoint_posterior_reacts_to_a_persistent_shift() -> None:
    """A real shift raises the changepoint log-odds; a stationary prefix does not."""

    # robust_omega is the generalized-Bayes precision increment; pairing it with
    # obs_var makes the detector a plain Normal-Normal model at unit information.
    config = SupportConfig(hazard=1.0 / 16.0, keep_run_lengths=16, obs_var=1.0, robust_omega=0.5)
    generator = torch.Generator().manual_seed(7)
    stationary = TruncatedBOCPD(config, 1)
    levels = [float(stationary.update(torch.randn(1, generator=generator))) for _ in range(12)]
    assert levels[-1] < 0.0

    moving = TruncatedBOCPD(config, 1)
    for _ in range(8):
        moving.update(torch.randn(1, generator=generator))
    shifted = [float(moving.update(torch.full((1,), 6.0))) for _ in range(4)]
    assert shifted[-1] > levels[-1]
    assert shifted[-1] > 0.0


def test_bocpd_state_round_trips() -> None:
    config = SupportConfig()
    detector = TruncatedBOCPD(config, 2, 3)
    generator = torch.Generator().manual_seed(3)
    for _ in range(5):
        detector.update(torch.randn(2, 3, generator=generator))
    restored = TruncatedBOCPD.from_state_dict(config, detector.state_dict())
    assert torch.equal(restored.log_message, detector.log_message)
    assert torch.equal(restored.run_index, detector.run_index)
    assert restored.step == detector.step


# ---------------------------------------------------------------------------
# Spatial support: candidates, decisions, and the conservative cell expansion.
# ---------------------------------------------------------------------------


def _window(batch: int = 1, frames: int = 4, channels: int = 3, size: int = 32):
    generator = torch.Generator().manual_seed(11)
    values = torch.rand((batch, frames, channels, size, size), generator=generator)
    mask = torch.ones((batch, frames, 1, size, size), dtype=torch.bool)
    return values, mask


def test_multiscale_candidates_are_finite_and_do_not_take_a_closure() -> None:
    support = SpatialSupport(SupportConfig(region_grid=4, multiscale_radii=(0, 1)))
    mask = support.neighbourhood_mask(4, 4, torch.device("cpu"))
    assert mask.shape == (16, 16)
    assert bool(mask.diagonal().all())
    # Radius 1 on a 4x4 grid: every region reaches at most 9 candidates, never all 16.
    per_row = mask.sum(-1)
    assert int(per_row.max()) == 9
    assert int(per_row.min()) == 4
    # Transitively, the closure of corner (0,0) is the whole grid; the declared
    # candidate set must stay strictly smaller than that.
    assert int(per_row.min()) < 16


def test_one_stable_observation_is_accepted_and_a_short_shift_is_not() -> None:
    """A single-observation smoke check, NOT the retraction property.

    The old name claimed "a shift is retracted" while the body only asserted that some
    state is not ACCEPTED, which the inverted rule satisfied just as well -- and it stopped
    one arrival short of the regime where production retracted everything.  The properties
    that matter are covered by ``test_a_long_stationary_history_stays_usable`` and
    ``test_a_real_change_withdraws_the_history_then_recovers_as_it_settles``, which drive
    the production path; this one is kept only as a cheap shape-level smoke test.
    """

    support = SpatialSupport(SupportConfig(region_grid=4, multiscale_radii=(0,), hazard=1.0 / 8.0))
    values, mask = _window(frames=4, size=32)
    state = support.initial_state(
        values.shape[0], values.shape[1] - 1, values.shape[-2], values.shape[-1], values.device
    )
    source_valid = torch.ones(values.shape[0], values.shape[1], dtype=torch.bool)

    # A perfectly stable history has no innovation at all.
    stable = values[:, :1].repeat(1, values.shape[1], 1, 1, 1)
    state, decision = support.observe(state, stable, mask, source_valid)
    assert int((decision.state == SUPPORT_RETRACTED).sum()) == 0
    assert bool(decision.source_usable.any())

    # Frames 1..T-1 shifted far away from the reference are not the same content.
    shifted = values.clone()
    shifted[:, 1:] += 5.0
    for _ in range(6):
        state, decision = support.observe(state, shifted, mask, source_valid)
    assert int((decision.state != SUPPORT_ACCEPTED).sum()) > 0
    assert bool(decision.deferred.any())


def _drive_model(frames: torch.Tensor):
    """Run frames through the production path and return the final support decision.

    This goes through ``N01V12Model.step`` rather than ``SpatialSupport.observe`` on
    purpose: the inversion this guards against only appeared with the production window,
    mask and default configuration, and a support-level harness with a hand-built window
    did not reproduce it.
    """

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
    from owac.frameworks.n01_full.model import N01V12Config, N01V12Model

    length = frames.shape[1]
    model = N01V12Model(
        N01V12Config(
            arm="raw_n01",
            platform="libero",
            window_frames=length,
            # Narrow and shallow: the property under test is a decision rule, and the
            # full-width defaults made this test a minute long.
            feature_width=32,
            stem_width=8,
            residual_blocks=1,
        )
    )
    state = model.init_state(
        1, {"camera_ids": ["agentview"], "shape": frames.shape[-2:]}, torch.device("cpu")
    )
    budget = ComputeBudget(max_updates=length, max_history=length)
    decision = None
    for index in range(length):
        frame = MeasurementFrame(
            values=frames[:, index],
            valid_mask=torch.ones(1, dtype=torch.bool),
            bad_mask=torch.zeros(1, dtype=torch.bool),
            saturated_mask=torch.zeros(1, dtype=torch.bool),
            frame_ids=torch.tensor([index]),
            camera_id="agentview",
            timestamp=torch.tensor([float(index)]),
            time_unit="frame",
            new_measurement=torch.ones(1, dtype=torch.bool),
        )
        output = model.step(frame, None, state, budget)
        state = output.state
        decision = model.encoder.last_support
    return decision


def test_a_long_stationary_history_stays_usable() -> None:
    """The regression the old suite missed by stopping at six observations.

    ``change_logodds`` is exactly 0 while the two run-length hypotheses are still
    indistinguishable, and it settles on the *negative* side for a stationary series.  A
    retract test written as ``odds <= threshold`` therefore withdrew precisely the stable
    histories the mechanism exists to accumulate, from the ninth arrival onward -- and the
    old test, one observation for the stationary half and six for the shift half, never
    reached that regime.
    """

    base = torch.rand((1, 3, 32, 32)).unsqueeze(1)
    decision = _drive_model(base.repeat(1, 32, 1, 1, 1))
    assert int((decision.state == SUPPORT_RETRACTED).sum()) == 0, (
        "a stationary history was withdrawn"
    )
    assert bool(decision.source_usable.all()), "every stable source must stay usable"


def test_a_real_change_withdraws_the_history_then_recovers_as_it_settles() -> None:
    """Retraction fires on changed content, and is revisable afterwards.

    A one-way RETRACTED state would withdraw a source for the life of the buffer, which
    is the opposite of maintained evidence.
    """

    base = torch.rand((1, 3, 32, 32)).unsqueeze(1)  # [B, 1, C, H, W]
    shifted = base + 5.0
    frames = torch.cat([base.repeat(1, 16, 1, 1, 1), shifted.repeat(1, 16, 1, 1, 1)], dim=1)

    # The detector reports exactly zero odds until both run-length hypotheses are present,
    # so a two-arrival scene carries no evidence either way; the regimes are sampled where
    # the decision is actually made.
    stable = _drive_model(frames[:, :16])
    assert bool(stable.source_usable.all()), "the stationary phase must be usable"

    after_shift = _drive_model(frames[:, :20])
    assert int(after_shift.source_usable.sum()) == 0, (
        "the pre-change history must not stay usable once the shift is detected"
    )

    # Revisability is asserted on the state codes rather than on ``source_usable``: the
    # conservative receptive-field expansion masks a region whose neighbours are still
    # withdrawn, so a handful of recovered regions do not become usable on their own.  The
    # property that matters here is that RETRACTED is not absorbing.
    settled = _drive_model(frames[:, :32])
    recovered = int((settled.state != SUPPORT_RETRACTED).sum())
    assert recovered > 0, (
        "RETRACTED must not be absorbing: once the post-change frames form the history "
        "they must be able to return to an accepted or deferred state"
    )


def test_invalid_sources_never_become_usable() -> None:
    support = SpatialSupport(SupportConfig(region_grid=4, multiscale_radii=(0,)))
    values, mask = _window(frames=3, size=32)
    # Slot 0 is the reference; slots 1 and 2 are the history sources.
    source_valid = torch.tensor([[True, False, False]])
    state = support.initial_state(1, 2, values.shape[-2], values.shape[-1], values.device)
    _, decision = support.observe(state, values, mask, source_valid)
    assert int(decision.state[0, 0].max()) == SUPPORT_INVALID
    assert not bool(decision.source_usable[0, 0].any())
    assert not bool(decision.source_usable[0, 1].any())


def test_cell_expansion_is_conservative_across_regions() -> None:
    """A cell needs every overlapping candidate region to accept the source."""

    support = SpatialSupport(SupportConfig(region_grid=2, multiscale_radii=(0, 1)))
    height = width = 8
    state = torch.full((1, 1, 4), SUPPORT_ACCEPTED, dtype=torch.int64)
    state[0, 0, 0] = SUPPORT_RETRACTED
    source_valid = torch.ones(1, 1, dtype=torch.bool)
    usable = support.expand_to_cells(state, source_valid, height, width)
    assert usable.shape == (1, 1, height, width)
    # Region 0 is retracted; because radius 1 makes every region a candidate of
    # its neighbours, no cell may stay usable.
    assert not bool(usable.any())


def test_support_masks_the_content_path_input() -> None:
    """The decision must be able to remove a source from the fusion input."""

    support = SpatialSupport(SupportConfig(region_grid=4, multiscale_radii=(0,)))
    values, mask = _window(batch=1, frames=3, size=32)
    source_valid = torch.ones(1, 3, dtype=torch.bool)
    state = support.initial_state(1, 2, 32, 32, values.device)
    _, decision = support.observe(state, values, mask, source_valid)
    history = values[:, 1:]
    # The decision covers the history slots, so the mask must be applied there.
    mask_5d = decision.source_usable.unsqueeze(2)
    # The decision is per cell, so it broadcasts over channels rather than matching them.
    assert mask_5d.shape == (history.shape[0], history.shape[1], 1, *history.shape[-2:])
    masked = torch.where(mask_5d, history, torch.zeros_like(history))
    assert torch.equal(masked, history)
    # With a source withdrawn the content really changes: the mask is not decorative.
    withdrawn = torch.zeros_like(decision.source_usable)
    blocked = torch.where(withdrawn.unsqueeze(2), history, torch.zeros_like(history))
    assert torch.equal(blocked, torch.zeros_like(history))
    assert not torch.equal(blocked, history)


def test_config_rejects_unsound_geometry() -> None:
    with pytest.raises(ValueError):
        SupportConfig(multiscale_radii=(1, 2))
    with pytest.raises(ValueError):
        SupportConfig(hazard=1.5)
    with pytest.raises(ValueError):
        SupportConfig(keep_run_lengths=1)
    # The withdraw bar must sit ABOVE the stable bar.  ``change_logodds`` is positive
    # when a changepoint happened recently, so this ordering is what leaves a band in
    # which a withdrawn source can be judged stable again; the reverse ordering (which
    # this test previously demanded) encoded the inverted comparison.
    with pytest.raises(ValueError):
        SupportConfig(accept_logodds=1.0, retract_logodds=-1.0)
