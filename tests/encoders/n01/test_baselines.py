"""Numerical and behavioural tests for the N01 mechanism baselines.

Checks real behaviour, not output shape alone: hand-computed EMA/Kalman/FixedAverage
recursions, the single-frame control's indifference to history, causal prefix
independence, per-frame vs. whole-window equivalence of every recurrent baseline,
idempotent ``reset_state``, one-observation-per-frame accounting and the shared
correspondence contract of the motion-compensated integrator.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from owac.baselines.n01 import (
    ConvGruIntegrator,
    EmaIntegrator,
    FixedAverageIntegrator,
    KalmanIntegrator,
    MechanismProbe,
    MotionCompensatedIntegrator,
    N01FixedHorizonProbe,
    N01NoMergeProbe,
    N01Probe,
    SingleFrameRawCnn,
    baseline_registry,
    mechanism_baselines,
    strongest_pair,
)


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    """Pin torch to one thread for this module.

    The tensors here are tiny, so multi-thread spinning dominates the runtime (measured
    2.9 s vs 0.18 s for one candidate window at 24 threads). The previous setting is
    restored afterwards for the rest of the session.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


WIDTH = 16
CHANNELS = 3
MAX_HISTORY = 8
SIZE = 8
EPS = 1e-8  # mirrors the probe's default Kalman epsilon


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _sequence(
    batch: int = 2,
    steps: int = 5,
    channels: int = CHANNELS,
    size: int = SIZE,
    seed: int = 0,
    invalid_band: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    frames = torch.randn(batch, steps, channels, size, size, generator=generator)
    valid = torch.ones_like(frames)
    if invalid_band:
        valid[:, :, :, :2, :] = 0.0
    variance = 0.05 + 0.5 * torch.rand(batch, steps, channels, size, size, generator=generator)
    timestamps = torch.arange(steps, dtype=torch.float32).expand(batch, steps).clone()
    return frames, valid, variance, timestamps


def _bump(size: int = 16, channels: int = 1) -> torch.Tensor:
    """Two narrow Gaussian bumps: a unique correlation peak that smearing destroys."""
    rows = torch.arange(size, dtype=torch.float32)[:, None]
    cols = torch.arange(size, dtype=torch.float32)[None, :]
    main = torch.exp(-((rows - 8.0) ** 2 + (cols - 8.0) ** 2) / (2.0 * 2.0**2))
    side = 0.6 * torch.exp(-((rows - 5.0) ** 2 + (cols - 11.0) ** 2) / (2.0 * 1.0**2))
    pattern = (main + side)[None, None]
    return pattern.expand(1, channels, size, size).contiguous()


class _StubCorrespondence(nn.Module):
    """Test double with the ``CausalCorrespondence`` contract (not a reimplementation).

    ``forward`` reports a fixed integer displacement for ``other`` relative to
    ``reference`` (``None`` means "no motion"), and ``align`` samples the source into
    the reference grid exactly like the real module documents.
    """

    def __init__(self, shift: tuple[float, float] | None = None) -> None:
        super().__init__()
        self.shift = shift
        self.calls = 0
        self.fields: list[object] = []
        self.aligned_fields: list[object] = []

    def forward(
        self,
        reference: torch.Tensor,
        other: torch.Tensor,
        reference_valid: torch.Tensor | None = None,
        other_valid: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        batch = reference.shape[0]
        if self.shift is None:
            shift = torch.zeros((batch, 2), dtype=reference.dtype, device=reference.device)
        else:
            shift = torch.tensor(self.shift, dtype=reference.dtype, device=reference.device)
            shift = shift[None, :].expand(batch, 2).clone()
        self.calls += 1
        field = SimpleNamespace(
            shift=shift,
            residual=torch.zeros(batch, dtype=reference.dtype, device=reference.device),
            overlap=torch.ones(batch, dtype=reference.dtype, device=reference.device),
            valid=torch.ones(batch, dtype=torch.bool, device=reference.device),
            fractional=None,
        )
        self.fields.append(field)
        return field

    def align(
        self,
        source_values: torch.Tensor,
        source_variance: torch.Tensor,
        source_valid: torch.Tensor,
        field: SimpleNamespace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.aligned_fields.append(field)
        shift = torch.round(field.shift).to(torch.int64)
        batch, channels, height, width = source_values.shape
        rows = torch.arange(height)[None, :, None] + shift[:, 0][:, None, None]
        cols = torch.arange(width)[None, None, :] + shift[:, 1][:, None, None]
        inside = ((rows >= 0) & (rows < height) & (cols >= 0) & (cols < width))[:, None]
        flat = (rows.clamp(0, height - 1) * width + cols.clamp(0, width - 1)).reshape(
            batch, height * width
        )

        def gather(values: torch.Tensor) -> torch.Tensor:
            index = flat[:, None].expand(batch, channels, height * width)
            flat_values = values.reshape(batch, channels, height * width)
            return torch.gather(flat_values, 2, index).reshape(batch, channels, height, width)

        out_valid = (gather(source_valid.float()) > 0.5) & inside
        zeros = torch.zeros_like(source_values)
        out_values = torch.where(out_valid, gather(source_values), zeros)
        return out_values, gather(source_variance).clamp_min(1e-12), out_valid


def _probes() -> dict[str, MechanismProbe]:
    return {
        SingleFrameRawCnn.name: SingleFrameRawCnn(
            width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY
        ),
        FixedAverageIntegrator.name: FixedAverageIntegrator(
            width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY
        ),
        EmaIntegrator.name: EmaIntegrator(width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY),
        KalmanIntegrator.name: KalmanIntegrator(
            width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY
        ),
        MotionCompensatedIntegrator.name: MotionCompensatedIntegrator(
            width=WIDTH,
            channels=CHANNELS,
            max_history=MAX_HISTORY,
            correspondence=_StubCorrespondence(),
        ),
        ConvGruIntegrator.name: ConvGruIntegrator(
            width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY
        ),
    }


def _blob(batch: int, channels: int, size: int, seed: int = 3) -> torch.Tensor:
    """A smooth, unambiguous pattern with a single correlation peak."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(batch, channels, size, size, generator=generator)
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel = (kernel / kernel.sum())[None, None].expand(channels, 1, 3, 3)
    base = torch.nn.functional.conv2d(base, kernel, padding=1, groups=channels)
    return torch.tanh(3.0 * base)


# --------------------------------------------------------------------------------------
# shared interface
# --------------------------------------------------------------------------------------


def test_encode_signature_exposes_no_ground_truth():
    parameters = list(inspect.signature(MechanismProbe.encode).parameters)
    assert parameters == ["self", "frames", "valid", "variance", "timestamps"]


def test_every_probe_returns_a_finite_feature_field():
    frames, valid, variance, timestamps = _sequence()
    for name, probe in _probes().items():
        probe.reset_state()
        with torch.no_grad():
            feature = probe.encode(frames, valid, variance, timestamps)
        assert feature.shape == (2, WIDTH, SIZE, SIZE), name
        assert torch.isfinite(feature).all(), name
        expected_steps = frames.shape[1] if probe.needs_history else 1
        assert probe.diagnostics()["steps"] == expected_steps, name
        assert probe.diagnostics()["name"] == name


def test_input_spec_declares_the_frozen_keys():
    for name, probe in _probes().items():
        spec = probe.input_spec()
        assert set(spec) == {
            "frames",
            "channels",
            "causal",
            "uses_correspondence",
            "notes",
        }, name
        assert spec["causal"] is True, name
        assert spec["channels"] == CHANNELS, name
        assert isinstance(spec["frames"], int) and spec["frames"] >= 1, name
        assert isinstance(spec["notes"], str) and spec["notes"], name
    probes = _probes()
    assert probes["motion_compensated"].input_spec()["uses_correspondence"] is True
    assert probes["conv_gru"].input_spec()["uses_correspondence"] is False


def test_no_probe_uses_batchnorm():
    for name, probe in _probes().items():
        batch_norms = [
            module
            for module in probe.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        assert batch_norms == [], name
        assert any(isinstance(module, nn.GroupNorm) for module in probe.modules()), name


def test_construction_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError):
        KalmanIntegrator(width=0, channels=CHANNELS, max_history=4)
    with pytest.raises(ValueError):
        EmaIntegrator(width=WIDTH, channels=CHANNELS, max_history=4, decay=1.0)
    with pytest.raises(ValueError):
        FixedAverageIntegrator(width=WIDTH, channels=CHANNELS, max_history=4, window=0)


def test_encode_rejects_mismatched_shapes():
    probe = KalmanIntegrator(width=WIDTH, channels=CHANNELS, max_history=4)
    frames, valid, variance, timestamps = _sequence(steps=2)
    with pytest.raises(ValueError):
        probe.encode(frames[0], valid[0], variance[0])
    with pytest.raises(ValueError):
        probe.encode(frames[:, :, :2], valid[:, :, :2], variance[:, :, :2])
    with pytest.raises(ValueError):
        probe.encode(frames, valid[:, :, :1], variance, timestamps)
    with pytest.raises(ValueError):
        probe.encode(frames, valid, variance, timestamps[:, :1])
    with pytest.raises(ValueError):
        probe.encode(frames[:, :0], valid[:, :0], variance[:, :0], timestamps[:, :0])
    with pytest.raises(TypeError):
        probe.encode(frames.numpy(), valid, variance, timestamps)


# --------------------------------------------------------------------------------------
# single-frame control
# --------------------------------------------------------------------------------------


def test_single_frame_control_ignores_history_but_follows_the_current_frame():
    probe = SingleFrameRawCnn(width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY)
    frames, valid, variance, timestamps = _sequence(steps=4)
    with torch.no_grad():
        reference = probe.encode(frames, valid, variance, timestamps)

        history_changed = frames.clone()
        history_changed[:, :-1] = 5.0 + torch.flip(history_changed[:, :-1], dims=[2])
        probe.reset_state()
        after_history_change = probe.encode(history_changed, valid, variance, timestamps)
        assert torch.equal(reference, after_history_change)

        current_changed = frames.clone()
        current_changed[:, -1] = frames[:, -1] + 1.0
        probe.reset_state()
        after_current_change = probe.encode(current_changed, valid, variance, timestamps)
        assert not torch.allclose(reference, after_current_change)


# --------------------------------------------------------------------------------------
# fixed average
# --------------------------------------------------------------------------------------


def test_fixed_average_of_two_frames_matches_hand_computation():
    probe = FixedAverageIntegrator(
        width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY, window=2
    )
    frames, valid, variance, _ = _sequence(steps=3)
    with torch.no_grad():
        probe.encode(frames, valid, variance)
    mean = probe.statistics()["mean"]
    count = probe.statistics()["count"]
    # window=2 -> only frames 1 and 2 contribute, with uniform weights.
    expected_count = valid[:, 1] + valid[:, 2]
    expected_mean = (frames[:, 1] * valid[:, 1] + frames[:, 2] * valid[:, 2]) / (
        expected_count + EPS
    )
    assert torch.allclose(count, expected_count, atol=1e-6)
    assert torch.allclose(mean, expected_mean, atol=1e-6)
    assert probe.input_spec()["frames"] == 2


def test_fixed_average_default_is_the_degenerate_max_window_mean():
    probe = FixedAverageIntegrator(width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY)
    frames, valid, variance, _ = _sequence(steps=4)
    with torch.no_grad():
        probe.encode(frames, valid, variance)
    mean = probe.statistics()["mean"]
    count = valid.sum(dim=1)
    expected = (frames * valid).sum(dim=1) / (count + EPS)
    assert torch.allclose(mean, expected, atol=1e-6)
    assert torch.allclose(probe.statistics()["count"], count, atol=1e-6)
    assert probe.diagnostics()["degenerate_max_window"] is True
    assert probe.input_spec()["frames"] == MAX_HISTORY


def test_fixed_average_ignores_variance_and_unsupported_pixels_stay_zero():
    frames, valid, variance, _ = _sequence(steps=3)
    other_variance = variance * 100.0 + 1.0
    first = FixedAverageIntegrator(width=WIDTH, channels=CHANNELS, max_history=4)
    second = FixedAverageIntegrator(width=WIDTH, channels=CHANNELS, max_history=4)
    with torch.no_grad():
        first.encode(frames, valid, variance)
        second.encode(frames, valid, other_variance)
    assert torch.equal(first.statistics()["mean"], second.statistics()["mean"])
    # The fixture's invalid band is unmeasured in every frame, so its mean stays zero.
    unsupported = valid.sum(dim=1) == 0
    assert torch.all(unsupported[:, :, :2, :])
    assert not unsupported[:, :, 2:, :].any()
    band = first.statistics()["mean"][:, :, :2, :]
    assert torch.all(band == 0.0)


# --------------------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------------------


def test_ema_of_two_frames_matches_hand_computation():
    decay = 0.75
    probe = EmaIntegrator(width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY, decay=decay)
    frames, valid, variance, _ = _sequence(steps=2)
    with torch.no_grad():
        probe.encode(frames, valid, variance)
    mean = probe.statistics()["mean"]
    # Frame 0: weight = (1-d), numerator = (1-d)*x0 -> mean = x0.
    # Frame 1: weight = d*(1-d) + (1-d) = (1-d)(1+d)
    #          numerator = d(1-d)x0 + (1-d)x1 -> mean = (d*x0 + x1)/(1+d).
    expected = (decay * valid[:, 0] * frames[:, 0] + valid[:, 1] * frames[:, 1]) / (
        decay * valid[:, 0] + valid[:, 1] + EPS
    )
    expected = torch.where((valid[:, 0] + valid[:, 1]) > 0, expected, torch.zeros_like(expected))
    assert torch.allclose(mean, expected, atol=1e-6)
    expected_weight = (1.0 - decay) * (decay * valid[:, 0] + valid[:, 1])
    assert torch.allclose(probe.statistics()["weight"], expected_weight, atol=1e-6)
    # The recursion is one fixed time constant: the measured variance never enters.
    first = EmaIntegrator(width=WIDTH, channels=CHANNELS, max_history=4, decay=decay)
    second = copy.deepcopy(first)
    with torch.no_grad():
        a = first.encode(frames, valid, variance)
        b = second.encode(frames, valid, variance * 50.0 + 2.0)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------------------
# Kalman
# --------------------------------------------------------------------------------------


def test_kalman_of_two_frames_matches_hand_computation():
    probe = KalmanIntegrator(width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY)
    frames, valid, variance, _ = _sequence(steps=2)
    with torch.no_grad():
        probe.encode(frames, valid, variance)
    state = probe.statistics()
    # Frame 0 initialises: mean = x0, var = v0.
    # Frame 1: K = v0/(v0+v1+eps); mean = x0 + K*(x1-x0); var = (1-K)*v0.
    gain = variance[:, 0] / (variance[:, 0] + variance[:, 1] + EPS)
    expected_mean = frames[:, 0] + gain * (frames[:, 1] - frames[:, 0])
    expected_var = (1.0 - gain) * variance[:, 0]
    observed = valid[:, 0] + valid[:, 1] > 0
    assert torch.allclose(state["mean"][observed], expected_mean[observed], atol=1e-6)
    assert torch.allclose(state["var"][observed], expected_var[observed], atol=1e-6)
    # Unsupported pixels were never observed.
    assert torch.all(state["observed"] == (valid.sum(dim=1) > 0))


def test_kalman_gain_follows_the_supplied_noise_variance():
    frames, valid, variance, _ = _sequence(steps=2)
    low_noise = variance.clone()
    low_noise[:, 1] = 0.01
    high_noise = variance.clone()
    high_noise[:, 1] = 100.0
    with torch.no_grad():
        cheap = KalmanIntegrator(width=WIDTH, channels=CHANNELS, max_history=4)
        cheap.encode(frames, valid, low_noise)
        reference = cheap.statistics()["mean"]
        expensive = KalmanIntegrator(width=WIDTH, channels=CHANNELS, max_history=4)
        expensive.encode(frames, valid, high_noise)
        distrusting = expensive.statistics()["mean"]
    step_towards_new = (reference - frames[:, 0]).abs().mean()
    step_with_high_variance = (distrusting - frames[:, 0]).abs().mean()
    assert step_with_high_variance < step_towards_new


def test_kalman_keeps_its_state_where_the_frame_is_invalid():
    probe = KalmanIntegrator(width=WIDTH, channels=CHANNELS, max_history=4)
    frames, valid, variance, _ = _sequence(steps=2)
    masked = valid.clone()
    masked[:, 1] = 0.0
    with torch.no_grad():
        probe.encode(frames, masked, variance)
    state = probe.statistics()
    observed = valid[:, 0] > 0
    assert torch.allclose(state["mean"][observed], frames[:, 0][observed], atol=1e-6)
    assert torch.allclose(state["var"][observed], variance[:, 0][observed], atol=1e-6)
    assert torch.all(state["mean"][~observed] == 0.0)


# --------------------------------------------------------------------------------------
# recurrent consistency, causality and reset
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ema", "kalman", "conv_gru", "motion_compensated"])
def test_recurrent_probes_match_per_frame_encoding(name: str):
    window = _probes()[name]
    per_frame = copy.deepcopy(window)
    window.reset_state()
    per_frame.reset_state()
    frames, valid, variance, timestamps = _sequence(steps=4)
    with torch.no_grad():
        whole = window.encode(frames, valid, variance, timestamps)
        pieces = []
        for index in range(frames.shape[1]):
            pieces.append(
                per_frame.encode(
                    frames[:, index : index + 1],
                    valid[:, index : index + 1],
                    variance[:, index : index + 1],
                    timestamps[:, index : index + 1],
                )
            )
    assert torch.equal(whole, pieces[-1])
    assert window.diagnostics()["steps"] == 4
    assert per_frame.diagnostics()["steps"] == 4


@pytest.mark.parametrize("name", ["ema", "kalman", "conv_gru", "motion_compensated"])
def test_reset_state_is_idempotent(name: str):
    probe = _probes()[name]
    frames, valid, variance, timestamps = _sequence(steps=3)
    with torch.no_grad():
        probe.reset_state()
        probe.reset_state()
        first = probe.encode(frames, valid, variance, timestamps)
        probe.reset_state()
        probe.reset_state()
        second = probe.encode(frames, valid, variance, timestamps)
    assert torch.equal(first, second)
    assert probe.diagnostics()["steps"] == 3


@pytest.mark.parametrize("name", ["ema", "kalman", "conv_gru", "motion_compensated"])
def test_carried_state_changes_the_next_window(name: str):
    probe = _probes()[name]
    frames, valid, variance, timestamps = _sequence(steps=3, seed=1)
    other, other_valid, other_variance, other_times = _sequence(steps=3, seed=2)
    with torch.no_grad():
        probe.reset_state()
        fresh = probe.encode(frames, valid, variance, timestamps)
        probe.reset_state()
        probe.encode(other, other_valid, other_variance, other_times)
        carried = probe.encode(frames, valid, variance, timestamps)
    assert not torch.allclose(fresh, carried)


@pytest.mark.parametrize("name", ["ema", "kalman", "conv_gru", "motion_compensated"])
def test_prefix_output_is_independent_of_later_frames(name: str):
    prefix_probe = _probes()[name]
    full_probe = copy.deepcopy(prefix_probe)
    prefix_probe.reset_state()
    full_probe.reset_state()
    frames, valid, variance, timestamps = _sequence(steps=4)
    future_changed = frames.clone()
    future_changed[:, 2:] = -7.0
    with torch.no_grad():
        prefix = prefix_probe.encode(
            frames[:, :2], valid[:, :2], variance[:, :2], timestamps[:, :2]
        )
        prefix_with_changed_future = full_probe.encode(
            future_changed[:, :2], valid[:, :2], variance[:, :2], timestamps[:, :2]
        )
        assert torch.equal(prefix, prefix_with_changed_future)
        # ... and adding a later frame does move the output, so the check is meaningful.
        extended = full_probe.encode(
            future_changed[:, 2:3],
            valid[:, 2:3],
            variance[:, 2:3],
            timestamps[:, 2:3],
        )
    assert not torch.allclose(prefix, extended)


@pytest.mark.parametrize("name", ["fixed_average", "ema", "kalman", "motion_compensated"])
def test_one_frame_is_counted_as_one_observation(name: str):
    probe = _probes()[name]
    frames, valid, variance, timestamps = _sequence(steps=1)
    with torch.no_grad():
        probe.encode(frames, valid, variance, timestamps)
    state = probe.statistics()
    if name == "fixed_average":
        assert torch.allclose(state["count"], valid[:, 0], atol=1e-6)
    elif name == "ema":
        assert torch.allclose(state["weight"], (1.0 - 0.9) * valid[:, 0], atol=1e-6)
    elif name == "kalman":
        assert torch.allclose(state["mean"], frames[:, 0] * (valid[:, 0] > 0), atol=1e-6)
        assert torch.allclose(state["var"], variance[:, 0] * (valid[:, 0] > 0), atol=1e-6)
    else:
        assert torch.allclose(state["count"], valid[:, 0], atol=1e-6)
    assert probe.diagnostics()["steps"] == 1


# --------------------------------------------------------------------------------------
# motion compensation
# --------------------------------------------------------------------------------------


def test_motion_compensation_without_motion_equals_fixed_average():
    frames, valid, variance, timestamps = _sequence(steps=5)
    stub = _StubCorrespondence()
    compensated = MotionCompensatedIntegrator(
        width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY, correspondence=stub
    )
    average = FixedAverageIntegrator(width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY)
    with torch.no_grad():
        compensated.encode(frames, valid, variance, timestamps)
        average.encode(frames, valid, variance, timestamps)
    assert torch.allclose(compensated.statistics()["mean"], average.statistics()["mean"], atol=1e-6)
    assert torch.allclose(
        compensated.statistics()["count"], average.statistics()["count"], atol=1e-6
    )
    assert stub.calls == frames.shape[1] - 1


def test_motion_compensation_passes_the_estimated_field_to_align_unchanged():
    frames, valid, variance, timestamps = _sequence(steps=3)
    stub = _StubCorrespondence(shift=(0.0, 1.0))
    probe = MotionCompensatedIntegrator(
        width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY, correspondence=stub
    )
    with torch.no_grad():
        probe.encode(frames, valid, variance, timestamps)
    # Two align calls per new frame (statistics and count), each handed the exact field
    # object that ``forward`` returned, without rescaling or reinterpretation.
    assert len(stub.aligned_fields) == 2 * (frames.shape[1] - 1)
    for index, field in enumerate(stub.fields):
        assert stub.aligned_fields[2 * index] is field
        assert stub.aligned_fields[2 * index + 1] is field
    assert probe.diagnostics()["correspondence_calls"] == frames.shape[1] - 1


def test_motion_compensation_recovers_a_moving_pattern():
    base = _bump(size=16, channels=1)
    steps = 5
    # Content moves right by one pixel per frame; a shift of -1 in the field means
    # "sample the source at x-1", which undoes that motion.
    frames = torch.stack([torch.roll(base, shifts=index, dims=3) for index in range(steps)], dim=1)
    valid = torch.ones_like(frames)
    variance = torch.full_like(frames, 0.05)
    stub = _StubCorrespondence(shift=(0.0, -1.0))
    compensated = MotionCompensatedIntegrator(
        width=WIDTH, channels=1, max_history=MAX_HISTORY, correspondence=stub
    )
    average = FixedAverageIntegrator(width=WIDTH, channels=1, max_history=MAX_HISTORY)
    with torch.no_grad():
        compensated.encode(frames, valid, variance)
        average.encode(frames, valid, variance)
    interior = slice(4, 12)
    # The accumulator lives in the current frame's coordinates, so the reference is the
    # last frame, not the untranslated pattern. Every frame is a rolled copy of the same
    # pattern, so five correctly aligned observations must reconstruct it exactly.
    target = frames[:, -1][:, :, interior, interior]
    compensated_mean = compensated.statistics()["mean"][:, :, interior, interior]
    average_mean = average.statistics()["mean"][:, :, interior, interior]
    assert torch.allclose(compensated_mean, target, atol=1e-5)
    counts = compensated.statistics()["count"][:, :, interior, interior]
    assert torch.allclose(counts, torch.full_like(counts, float(steps)))
    average_error = (average_mean - target).abs()
    assert average_error.mean() > 0.05  # the unaligned average really is smeared
    assert (compensated_mean - target).abs().mean() < 0.1 * average_error.mean()


@pytest.mark.parametrize("mode", ["fixed", "candidate"])
def test_real_correspondence_runs_end_to_end_without_motion(mode: str):
    motion = pytest.importorskip("owac.frameworks.n01.motion")
    config = pytest.importorskip("owac.frameworks.n01.config")
    settings = dataclasses.replace(config.N01Config.tiny(), alignment=mode)
    correspondence = motion.CausalCorrespondence(settings, neural=False)
    probe = MotionCompensatedIntegrator(
        width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY, correspondence=correspondence
    )
    base = _blob(batch=1, channels=CHANNELS, size=16)
    frames = base[:, None].expand(1, 4, CHANNELS, 16, 16).contiguous()
    valid = torch.ones_like(frames)
    variance = torch.full_like(frames, 0.05)
    average = FixedAverageIntegrator(width=WIDTH, channels=CHANNELS, max_history=MAX_HISTORY)
    with torch.no_grad():
        probe.encode(frames, valid, variance)
        average.encode(frames, valid, variance)
    assert torch.isfinite(probe.statistics()["mean"]).all()
    assert probe.diagnostics()["correspondence_calls"] == 3
    # Identical frames must correspond at zero displacement, so alignment is a no-op.
    assert torch.allclose(probe.statistics()["mean"], average.statistics()["mean"], atol=1e-5)


def test_every_probe_backpropagates_within_a_window():
    """A probe must be trainable through ``encode`` under the shared probe head."""
    frames, valid, variance, timestamps = _sequence(batch=1, steps=3)
    for name, probe in _probes().items():
        probe.reset_state()
        feature = probe.encode(frames, valid, variance, timestamps)
        feature.pow(2).mean().backward()
        grads = [parameter.grad for parameter in probe.parameters() if parameter.grad is not None]
        assert grads, name
        assert all(torch.isfinite(grad).all() for grad in grads), name
        assert any(float(grad.abs().sum()) > 0.0 for grad in grads), name


def test_n01_probe_backpropagates_within_a_window():
    probe = _n01_probes()["n01"]
    frames, valid, variance, timestamps = _sequence(batch=1, steps=2, size=8)
    feature = probe.encode(frames, valid, variance, timestamps)
    feature.pow(2).mean().backward()
    grads = [parameter.grad for parameter in probe.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(float(grad.abs().sum()) > 0.0 for grad in grads)


# --------------------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------------------


def test_registry_exposes_every_mechanism_and_materialises_probes():
    registry = baseline_registry()
    assert set(registry) == {
        "single_frame_raw_cnn",
        "fixed_average",
        "ema",
        "kalman",
        "motion_compensated",
        "conv_gru",
        "n01",
        "n01_no_merge",
        "n01_fixed_horizon",
    }
    for name, factory in registry.items():
        assert callable(factory), name


def test_strongest_pair_picks_the_two_smallest_position_errors():
    metrics = {
        "fixed_average": {"position_error": 0.42},
        "ema": {"position_error": 0.11},
        "kalman": {"position_error": 0.07},
        "conv_gru": {"position_error": 0.31},
        "n01": {"position_error": 0.01},
    }
    assert strongest_pair(metrics) == ["kalman", "ema"]
    metrics["ema"]["position_error"] = 0.07
    assert strongest_pair(metrics) == ["ema", "kalman"]  # ties break by name
    assert strongest_pair({**metrics, "kalman": {"position_error": float("nan")}}) == [
        "ema",
        "conv_gru",
    ]


def test_strongest_pair_rejects_a_table_without_two_baselines():
    with pytest.raises(ValueError):
        strongest_pair({"kalman": {"position_error": 0.1}})
    with pytest.raises(ValueError):
        strongest_pair({"n01": {"position_error": 0.1}, "ema": {"position_error": None}})


def test_registry_factories_accept_an_n01_config_when_it_exists():
    module = pytest.importorskip("owac.frameworks.n01.config")
    config = module.N01Config.tiny()
    probes = mechanism_baselines(config)
    assert "n01" not in probes
    assert len(probes) == 8
    for name, probe in probes.items():
        assert probe.width == config.width, name
        assert probe.channels == config.channels, name
        assert probe.max_history == config.max_history, name


# --------------------------------------------------------------------------------------
# candidate and candidate ablations
# --------------------------------------------------------------------------------------


def _n01_probes(**overrides: object) -> dict[str, MechanismProbe]:
    config_module = pytest.importorskip("owac.frameworks.n01.config")
    pytest.importorskip("owac.frameworks.n01.model")
    settings: dict[str, object] = {"alignment": "none", "learnable_gate": False}
    settings.update(overrides)
    config = config_module.N01Config.tiny(**settings)
    return {
        "n01": N01Probe(config, max_updates=1),
        "n01_no_merge": N01NoMergeProbe(config, max_updates=1),
        "n01_fixed_horizon": N01FixedHorizonProbe(config, max_updates=1),
    }


def _edge_sequence(steps: int = 10, size: int = 8, block: slice = slice(2, 6)):
    """Smooth background plus a hard block, then smooth again.

    The block's boundary pixels are a strong measured contradiction, so the candidate
    closes those edges (split) and re-opens them once the block disappears (merge):
    six refuted frames are needed because ``edge_forget=0.1`` only multiplies by 0.9.
    """
    generator = torch.Generator().manual_seed(7)
    frames = 0.01 * torch.randn(1, steps, CHANNELS, size, size, generator=generator)
    for index in range(4, steps - 3):
        frames[:, index, :, block, block] = 1.0
    valid = torch.ones_like(frames)
    variance = torch.full_like(frames, 0.05)
    return frames, valid, variance


def test_n01_probe_returns_the_encoder_representation():
    probes = _n01_probes()
    probe = probes["n01"]
    frames, valid, variance, timestamps = _sequence(batch=1, steps=3, size=8)
    with torch.no_grad():
        feature = probe.encode(frames, valid, variance, timestamps)
    assert feature.shape == (1, probe.width, 8, 8)
    assert torch.isfinite(feature).all()
    info = probe.diagnostics()
    assert info["steps"] == 3
    assert info["depth"] == max(probe.config.time_depths)
    assert info["merge_count"] >= 0.0 and info["split_count"] >= 0.0
    spec = probe.input_spec()
    assert spec["causal"] is True
    assert spec["frames"] == probe.config.max_history
    assert spec["uses_correspondence"] == (probe.config.alignment != "none")
    assert probes["n01_no_merge"].input_spec()["uses_correspondence"] is False


def test_n01_probe_forwards_the_measured_variance_to_the_encoder():
    probe = _n01_probes()["n01"]
    frames, valid, variance, timestamps = _sequence(batch=1, steps=1, size=8)
    with torch.no_grad():
        probe.encode(frames, valid, variance, timestamps)
    stored = probe.encoder_state.variance[0]
    assert torch.allclose(stored, variance[:, 0], atol=1e-6)


@pytest.mark.parametrize("name", ["n01", "n01_no_merge", "n01_fixed_horizon"])
def test_n01_probes_match_per_frame_encoding_and_reset(name: str):
    probe = _n01_probes()[name]
    per_frame = copy.deepcopy(probe)
    frames, valid, variance, timestamps = _sequence(batch=1, steps=3, size=8)
    with torch.no_grad():
        probe.reset_state()
        probe.reset_state()
        whole = probe.encode(frames, valid, variance, timestamps)
        for index in range(frames.shape[1]):
            piece = per_frame.encode(
                frames[:, index : index + 1],
                valid[:, index : index + 1],
                variance[:, index : index + 1],
                timestamps[:, index : index + 1],
            )
        probe.reset_state()
        probe.reset_state()
        repeated = probe.encode(frames, valid, variance, timestamps)
    assert torch.equal(whole, piece)
    assert torch.equal(whole, repeated)


def test_n01_no_merge_ablation_changes_merge_and_split_diagnostics():
    probes = _n01_probes()
    candidate = probes["n01"]
    ablation = probes["n01_no_merge"]
    ablation.load_state_dict(candidate.state_dict())
    frames, valid, variance = _edge_sequence()
    with torch.no_grad():
        candidate_feature = candidate.encode(frames, valid, variance)
        ablation_feature = ablation.encode(frames, valid, variance)
    candidate_info = candidate.diagnostics()
    ablation_info = ablation.diagnostics()
    # The candidate closes refuted edges and re-opens them; the ablation has no
    # transition left to count because every edge is forced open from the start.
    # Measured with the tiny defaults (merge 0.55 / split 0.35 / alpha 0.35 / forget 0.1):
    # candidate 48 merges, 160 splits; ablation 0 and 0.
    assert candidate_info["split_count"] > 0.0
    assert candidate_info["merge_count"] > 0.0
    assert ablation_info["split_count"] == 0.0
    assert ablation_info["merge_count"] == 0.0
    # The raw analytic edge measurement is threshold independent, so the difference comes
    # from the update rule and not from a different gate; the resulting weighted support
    # width and the representation do change.
    assert ablation_info["edge_mean"] == candidate_info["edge_mean"]
    assert ablation_info["mean_support_width"] != candidate_info["mean_support_width"]
    # The maintained edge state is what the rule changes: the candidate forgets a refuted
    # edge, the forced-open ablation only relaxes it towards the measured gate.
    candidate_edges = candidate.encoder_state.support[candidate.camera_id].edges
    ablation_edges = ablation.encoder_state.support[ablation.camera_id].edges
    assert not torch.allclose(candidate_edges, ablation_edges)
    assert bool(ablation.encoder_state.support[ablation.camera_id].open_edges.all())
    # NOTE: with this tiny configuration the selector reads a radius-0 domain for both
    # variants, so the readout feature can coincide even though the support state differs;
    # the ablation is therefore checked on diagnostics and state, not on the feature.
    assert candidate_feature.shape == ablation_feature.shape
    assert ablation.config.split_threshold < 0.0
    assert ablation.config.merge_threshold == 0.0


def test_n01_fixed_horizon_ablation_pins_the_depth_under_a_low_budget():
    probes = _n01_probes(budget_conditioning=True)
    candidate = probes["n01"]
    ablation = probes["n01_fixed_horizon"]
    ablation.load_state_dict(candidate.state_dict())
    frames, valid, variance = _edge_sequence(steps=6)
    with torch.no_grad():
        candidate.encode(frames, valid, variance)
        ablation.encode(frames, valid, variance)
    candidate_info = candidate.diagnostics()
    ablation_info = ablation.diagnostics()
    assert candidate_info["budget_conditioning"] is True
    assert candidate_info["depth"] == 4  # one update keeps two of (1, 4, 8) levels
    assert ablation_info["budget_conditioning"] is False
    assert ablation_info["time_depths"] == (8,)
    assert ablation_info["depth"] == 8
    assert ablation_info["depth"] > candidate_info["depth"]
