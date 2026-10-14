"""Regression for the collated window API on the ``raw_adapter`` path.

``owac.raw.rawvla_sequence.MeasurementWindowBatch`` (the object the real loader
collates) has no ``unique_frame_count()`` helper -- only the single-window
``MeasurementWindow`` does.  The preflight therefore rebuilds the per-member
unique count from the identity arrays the batch really exposes
(``episode_ids``/``frame_ids``/``new_measurement``, each ``[B, T]``), and this
file pins both the semantics and the end-to-end call that failed on real data:

* a re-sent exposure counts once (identity is ``(episode_id, frame_id)``, never
  the timestamp);
* the same ``frame_id`` under two episode ids counts twice;
* the adapter's own ``unique_arrivals`` must match the window's new-arrival
  count member by member, which is exactly what ``_adapter_encode_batch``
  reports before the training forward.
"""

from __future__ import annotations

import pytest
import torch

from owac.frameworks.n01_full.adapters import DbsrAdapterConfig, DbsrFeatureAdapter
from owac.raw.rawvla_sequence import MeasurementWindow, collate_measurement_windows
from owac.runtime.n01_full_policy_preflight import _adapter_encode_batch, _window_unique_counts

CAMERA = "base_0_rgb"
HEIGHT = WIDTH = 8


def _window(
    frame_ids,
    *,
    episodes=None,
    new_measurement=None,
    camera=CAMERA,
    seed=0,
):
    """One real ``MeasurementWindow`` with the declared identity arrays."""
    frames = len(frame_ids)
    if episodes is None:
        episodes = [0] * frames
    if new_measurement is None:
        flags = [True] + [bool(frame_ids[step] != frame_ids[step - 1]) for step in range(1, frames)]
    else:
        flags = list(new_measurement)
    generator = torch.Generator().manual_seed(seed)
    shape = (frames, 3, HEIGHT, WIDTH)
    return MeasurementWindow(
        values=torch.randn(*shape, generator=generator),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        bad_mask=torch.zeros(shape, dtype=torch.bool),
        saturated_mask=torch.zeros(shape, dtype=torch.bool),
        episode_ids=torch.tensor(episodes, dtype=torch.int64),
        frame_ids=torch.tensor(frame_ids, dtype=torch.int64),
        camera_id=camera,
        timestamp=torch.arange(frames, dtype=torch.float32),
        time_unit="frame",
        new_measurement=torch.tensor(flags, dtype=torch.bool),
        sample_xy=torch.zeros((3, 2), dtype=torch.int64),
    )


def test_repeated_arrivals_of_the_same_frame_count_once():
    batch = collate_measurement_windows([_window([5, 5, 5])])
    counts = _window_unique_counts(batch, 1)
    assert counts["identities"] == [1]
    assert counts["new_arrivals"] == [1]
    assert counts["identity_rule"] == "(episode_id, frame_id)"

    # A repeated transmission flagged True is still one identity and one arrival.
    repeated_flag = collate_measurement_windows(
        [_window([5, 5, 7], new_measurement=[True, True, True])]
    )
    assert _window_unique_counts(repeated_flag, 1)["new_arrivals"] == [2]

    # A skipped first occurrence is not an arrival, but a later flagged one is
    # (mirrors the adapter's own "flag and not already observed" rule).
    late = collate_measurement_windows([_window([5, 5], new_measurement=[False, True])])
    assert _window_unique_counts(late, 1)["new_arrivals"] == [1]
    assert _window_unique_counts(collate_measurement_windows([_window([5, 5])]), 1)[
        "new_arrivals"
    ] == [1]


def test_same_frame_id_in_two_episodes_counts_twice():
    # Inside one window: the episode axis changes, the frame id repeats.
    mixed = collate_measurement_windows([_window([5, 5], episodes=[0, 1])])
    counts = _window_unique_counts(mixed, 1)
    assert counts["identities"] == [2]
    # The default loader flag marks the second entry as a repeat; when both
    # entries are declared new arrivals, the two episodes give two arrivals.
    assert counts["new_arrivals"] == [1]
    flagged = collate_measurement_windows(
        [_window([5, 5], episodes=[0, 1], new_measurement=[True, True])]
    )
    assert _window_unique_counts(flagged, 1)["new_arrivals"] == [2]

    # Across the batch: member counts stay per member and are never merged.
    batch = collate_measurement_windows(
        [_window([5, 5], episodes=[0, 0], seed=1), _window([5, 5], episodes=[1, 1], seed=2)]
    )
    counts = _window_unique_counts(batch, 2)
    assert counts["identities"] == [1, 1]
    assert counts["new_arrivals"] == [1, 1]


def test_window_counts_do_not_depend_on_timestamps_or_metadata():
    """Two different timestamps for one identity are one frame; the rule is identity."""
    batch = collate_measurement_windows([_window([9, 9, 11])])
    counts = _window_unique_counts(batch, 1)
    assert counts["identities"] == [2] and counts["new_arrivals"] == [2]
    assert not hasattr(batch, "unique_frame_count"), (
        "the collated batch must not grow a single-window helper; the preflight rebuilds it"
    )


def _tiny_adapter() -> DbsrFeatureAdapter:
    return DbsrFeatureAdapter(
        DbsrAdapterConfig(
            input_mode="rawvla_rgb3",
            stem_width=4,
            residual_blocks=1,
            feature_width=8,
            projection_width=4,
            offset_width=8,
            max_history=4,
            pool_grid=2,
            tokens_per_camera=4,
            readout_width=16,
        )
    )


def test_adapter_encode_batch_matches_window_arrivals_end_to_end():
    """The exact call that failed on real data: no AttributeError, counts must match."""
    windows = [
        _window([3, 3, 4, 5], episodes=[0] * 4, seed=3),
        _window([3, 3, 4, 5], episodes=[1] * 4, seed=4),
    ]
    batch = {
        "windows": {CAMERA: collate_measurement_windows(windows)},
        "meta": [{"episode_key": "ep0"}, {"episode_key": "ep1"}],
    }
    adapter = _tiny_adapter()
    _state, output, feed = _adapter_encode_batch(adapter, batch, "cpu")
    entry = feed[CAMERA]
    assert entry["window_frames"] == 4
    assert entry["unique_frames_window"] == [3, 3]
    assert entry["unique_new_measurements_window"] == [3, 3]
    assert entry["adapter_unique_lifetime"] == [3, 3]
    assert entry["unique_matches_window"] == [True, True]
    assert entry["unique_frame_identity_rule"] == "(episode_id, frame_id)"
    assert entry["p2_motion_status"] == "identity_pending_p2"


def test_identity_count_and_arrival_count_are_reported_separately():
    """The episode axis matters: identities are 3, the flagged arrivals are 2."""
    window = _window([7, 7, 8], episodes=[0, 1, 1], seed=5)
    batch = {
        "windows": {CAMERA: collate_measurement_windows([window])},
        "meta": [{"episode_key": "ep0"}],
    }
    adapter = _tiny_adapter()
    _state, output, feed = _adapter_encode_batch(adapter, batch, "cpu")
    assert feed[CAMERA]["unique_frames_window"] == [3]
    assert feed[CAMERA]["unique_new_measurements_window"] == [2]
    assert feed[CAMERA]["unique_matches_window"] == [True]
    # The adapter's own diagnostics agree with the window array per member.
    assert output.diagnostics["unique_frames_lifetime"] == [2]


def test_window_counts_reject_a_batch_size_mismatch():
    batch = collate_measurement_windows([_window([1, 2])])
    with pytest.raises(ValueError, match="does not match batch_size"):
        _window_unique_counts(batch, 3)
