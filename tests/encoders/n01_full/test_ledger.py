"""Identity, validity and lifecycle semantics of the N01 v1.0 measurement ledger.

These tests are behavioural, not smoke tests: they assert the counts, the raw
records and the excluded reasons that the contract fixes, and the 1-D example is
checked against the independent NumPy FP64 oracle rather than against production
output.
"""

import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full import reference as reference_module
from owac.frameworks.n01_full.contracts import SourceKey
from owac.frameworks.n01_full.ledger import (
    BAD,
    MISSING,
    SATURATED,
    VALID,
    ContentReplacement,
    MeasurementLedger,
)
from owac.frameworks.n01_full.sequence1d import (
    make_sequence,
    observe_sequence,
    sequence1d_example,
    sequence_noise_model,
    sequence_sources,
)


def frame(
    values,
    *,
    frame_id=0,
    episode_id="e0",
    camera_id="c0",
    valid=None,
    bad=None,
    saturated=None,
    new_measurement=True,
    timestamp=0.0,
):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, 1, -1)
    return MeasurementFrame(
        values=tensor,
        valid_mask=None if valid is None else torch.as_tensor(valid).reshape(tensor.shape),
        bad_mask=None if bad is None else torch.as_tensor(bad).reshape(tensor.shape),
        saturated_mask=(
            None if saturated is None else torch.as_tensor(saturated).reshape(tensor.shape)
        ),
        episode_ids=[episode_id],
        frame_ids=[frame_id],
        camera_id=camera_id,
        timestamp=torch.tensor([float(timestamp)]),
        new_measurement=torch.tensor([new_measurement]),
    )


def key(frame_id=0, position=0, *, episode_id="e0", camera_id="c0", channel=0):
    return SourceKey("ds", episode_id, camera_id, frame_id, channel, 0, position)


def test_duplicate_packet_does_not_advance_counts_or_precision():
    ledger = MeasurementLedger("ds")
    first = ledger.observe(frame([1.0, 3.0]))
    second = ledger.observe(frame([1.0, 3.0]))
    assert first.new_samples == 2
    assert second.new_samples == 0
    assert second.duplicate_frames == first.registered_frames
    assert ledger.unique_frame_count(camera_id="c0") == 1
    assert ledger.frame_count() == 1
    record = ledger.get_source(key(0, 1))
    assert record.raw_value == 3.0
    assert record.validity_kind == "valid"


def test_new_exposure_with_identical_values_is_a_new_measurement():
    ledger = MeasurementLedger("ds")
    ledger.observe(frame([1.0, 3.0], frame_id=0))
    ledger.observe(frame([1.0, 3.0], frame_id=1))
    assert ledger.unique_frame_count(camera_id="c0") == 2
    assert ledger.get_source(key(0, 0)).raw_value == ledger.get_source(key(1, 0)).raw_value
    assert key(0, 0).identity() != key(1, 0).identity()


def test_unflagged_repeat_is_skipped_not_invented():
    ledger = MeasurementLedger("ds")
    outcome = ledger.observe(frame([1.0, 3.0], frame_id=5, new_measurement=False))
    assert outcome.registered_frames == ()
    assert outcome.skipped_samples == 2
    assert ledger.frame_count() == 0
    assert not ledger.has_source(key(5, 0))


def test_validity_precedence_and_values():
    ledger = MeasurementLedger("ds")
    ledger.observe(
        frame(
            [10.0, 1.0, 999.0, 3.0, 55.0],
            valid=[False, True, False, True, True],
            bad=[False, False, True, False, False],
            saturated=[False, False, False, False, True],
        )
    )
    reasons = [ledger.get_source(key(0, position)).validity_kind for position in range(5)]
    assert reasons == ["missing", "valid", "bad", "valid", "saturated"]
    assert ledger.get_source(key(0, 0)).raw_value is None
    # bad and saturated keep their measured value: the sensor state is not rewritten.
    assert ledger.get_source(key(0, 2)).raw_value == 999.0
    assert ledger.get_source(key(0, 4)).raw_value == 55.0
    codes = ledger.frames()[0].validity.reshape(-1).tolist()
    assert codes == [MISSING, VALID, BAD, VALID, SATURATED]


def test_partial_reset_keeps_other_batch_members_and_cameras():
    ledger = MeasurementLedger("ds")
    batch = MeasurementFrame(
        values=torch.tensor([[[[1.0, 3.0]]], [[[7.0, 9.0]]]], dtype=torch.float64),
        episode_ids=["b0", "b1"],
        frame_ids=[0, 0],
        camera_id="c0",
        timestamp=torch.tensor([0.0, 0.0]),
        new_measurement=torch.tensor([True, True]),
    )
    ledger.observe(batch)
    ledger.observe(frame([100.0, 300.0], frame_id=0, camera_id="c1"))
    result = ledger.reset(["b0"])
    assert len(result["removed_frames"]) == 1
    assert not ledger.has_source(key(0, 0, episode_id="b0"))
    assert ledger.get_source(key(0, 0, episode_id="b1")).raw_value == 7.0
    assert ledger.get_source(key(0, 1, camera_id="c1")).raw_value == 300.0
    assert ledger.unique_frame_count(camera_id="c0") == 1


def test_equal_frame_ids_across_cameras_and_episodes_do_not_collide():
    ledger = MeasurementLedger("ds")
    ledger.observe(frame([1.0], frame_id=0, camera_id="c0", episode_id="e0"))
    ledger.observe(frame([2.0], frame_id=0, camera_id="c1", episode_id="e0"))
    ledger.observe(frame([3.0], frame_id=0, camera_id="c0", episode_id="e1"))
    assert ledger.get_source(key(0, 0, camera_id="c0", episode_id="e0")).raw_value == 1.0
    assert ledger.get_source(key(0, 0, camera_id="c1", episode_id="e0")).raw_value == 2.0
    assert ledger.get_source(key(0, 0, camera_id="c0", episode_id="e1")).raw_value == 3.0
    assert ledger.frame_count() == 3


def test_content_replacement_is_a_view_and_never_rewrites_the_raw_record():
    ledger = MeasurementLedger("ds")
    ledger.observe(frame([1.0, 3.0]))
    target = key(0, 1)
    replaced = ledger.with_content(ContentReplacement("garbage", {target: 1234.0}))
    assert replaced.get_source(target).raw_value == 1234.0
    assert "content-replacement" in replaced.get_source(target).value_units
    # The raw ledger and its record are untouched.
    assert ledger.get_source(target).raw_value == 3.0
    # Every public read on the derived view must see replaced content; the base
    # ledger is the immutable raw record and remains unchanged.
    assert replaced.get_frame("e0", "c0", 0).values[0, 0, 1] == 1234.0
    assert ledger.get_frame("e0", "c0", 0).values[0, 0, 1] == 3.0


def test_counterfactual_view_removes_only_the_listed_sources():
    ledger = MeasurementLedger("ds")
    ledger.observe(frame([1.0, 3.0]))
    target = key(0, 1)
    view = ledger.without([target])
    assert not view.has_source(target)
    assert view.has_source(key(0, 0))
    assert view.unique_frame_count(camera_id="c0") == 1  # the exposure still happened


def test_sequence1d_example_matches_independent_reference_and_dedup_rules():
    spec = make_sequence(frames=3, length=6, seed=5)
    ledger = MeasurementLedger("sequence-1d")
    outcomes = observe_sequence(spec, ledger, repeat_first=True)
    assert outcomes[-1].new_samples == 0
    assert ledger.unique_frame_count(camera_id=spec.camera_id) == spec.frames
    keys = sequence_sources(spec)
    noise = sequence_noise_model(spec)
    observations = [
        reference_module.ReferenceObservation(
            source_id=key,
            phase=0,
            xy=ledger.sample_coordinates(key),
            value=float(ledger.get_source(key).raw_value),
            variance=float(noise.variance_of(key)),
        )
        for key in keys
    ]
    oracle = reference_module.reference_fit(observations, basis="affine", dimension=1)
    example = sequence1d_example(frames=3, length=6, seed=5)
    oracle_mean, _ = oracle.predict(0, tuple(example["probe_coordinate"]))
    assert abs(example["probe_mean"] - oracle_mean) <= 1e-10 + 1e-8 * abs(oracle_mean)
    assert example["retracted_observation_count"] == example["independent_observation_count"] - 1
    assert example["raw_record_untouched"]


def test_loader_style_per_channel_offsets_expand_to_sample_coordinates():
    """The real loader declares one (row, col) offset per channel, not per sample."""
    from owac.runtime.n01_full_ledger_check import _sample_grid

    values = torch.zeros((1, 3, 2, 2), dtype=torch.float64)
    offsets = torch.tensor([[0, 0], [0, 1], [1, 0]])
    grid = _sample_grid(values, offsets)
    assert grid.shape == (1, 3, 2, 2, 2)
    assert grid[0, 0, 0, 1].tolist() == [1.0, 0.0]
    assert grid[0, 1, 0, 0].tolist() == [1.0, 0.0]
    assert grid[0, 2, 0, 0].tolist() == [0.0, 1.0]
    # No declaration, or a shape that is not one offset per channel, is not guessed.
    assert _sample_grid(values, None) is None
    with pytest.raises(ValueError, match="channel count"):
        _sample_grid(values, torch.zeros((2, 2)))

    ledger = MeasurementLedger("ds")
    ledger.observe(
        MeasurementFrame(
            values=values,
            sample_xy=grid,
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
            new_measurement=torch.tensor([True]),
        )
    )
    key = SourceKey("ds", "e0", "c0", 0, 1, 0, 0)
    assert ledger.sample_coordinates(key) == (1.0, 0.0)


def test_window_frames_iterate_the_time_axis_and_keep_declared_identity():
    """A loader window is [B, T, C, H, W]: arrivals must be indexed on axis 1."""
    from types import SimpleNamespace

    from owac.runtime.n01_full_ledger_check import iter_window_frames

    window = SimpleNamespace(
        values=torch.zeros((1, 3, 2, 2, 2), dtype=torch.float64),
        valid_mask=torch.ones((1, 3, 2, 2, 2), dtype=torch.bool),
        bad_mask=torch.zeros((1, 3, 2, 2, 2), dtype=torch.bool),
        saturated_mask=torch.zeros((1, 3, 2, 2, 2), dtype=torch.bool),
        frame_ids=torch.tensor([[1, 2, 3]]),
        timestamp=torch.tensor([[0.0, 1.0, 2.0]]),
        time_unit="frame",
        new_measurement=torch.tensor([[True, True, False]]),
        sample_xy=torch.tensor([[0, 0], [0, 0]]),
    )
    frames = list(iter_window_frames(window, ["ep0"], "c0"))
    assert [step for step, _ in frames] == [0, 1, 2]
    assert [int(frame.frame_ids[0]) for _, frame in frames] == [1, 2, 3]
    assert [bool(frame.new_measurement[0]) for _, frame in frames] == [True, True, False]
    assert [frame.episode_ids[0] for _, frame in frames] == ["ep0", "ep0", "ep0"]
    ledger = MeasurementLedger("ds")
    outcomes = [ledger.observe(frame) for _, frame in frames]
    assert [len(outcome.registered_frames) for outcome in outcomes] == [1, 1, 0]
    assert [len(outcome.skipped_frames) for outcome in outcomes] == [0, 0, 1]
    assert ledger.unique_frame_count(camera_id="c0") == 2


def test_tensor_identity_slices_deduplicate_like_python_ids():
    """The real loader passes 0-dim tensor ids; they must hash as values, not objects."""
    window = torch.zeros((1, 2, 1, 1, 2), dtype=torch.float64)
    frame_ids = torch.tensor([[0, 0]])
    new_measurement = torch.tensor([[True, True]])
    ledger = MeasurementLedger("ds")
    outcomes = []
    for step in range(2):
        outcomes.append(
            ledger.observe(
                MeasurementFrame(
                    values=window[:, step],
                    frame_ids=frame_ids[:, step],
                    episode_ids=["e0"],
                    camera_id="c0",
                    new_measurement=new_measurement[:, step],
                )
            )
        )
    assert len(outcomes[0].registered_frames) == 1
    assert len(outcomes[1].duplicate_frames) == 1
    assert outcomes[1].new_samples == 0
    assert ledger.unique_frame_count(camera_id="c0") == 1
    # The tensor id and the equivalent Python id are the same identity.
    key = SourceKey("ds", "e0", "c0", 0, 0, 0, 1)
    assert ledger.get_source(key).raw_value == 0.0
