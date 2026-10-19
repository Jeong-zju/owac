"""Numeric tests for the RawVLA measurement loader.

The fixtures are *small synthetic NPZ files built in a tmp_path* that follow the
author's key schema exactly (camera keys, ``actions``, ``robot_states``,
``metadata_json``). They let the window indexing, the padding masks, the unit
conventions and the split manifest be checked against declared numbers without
the 714 GB server cache.

Anything that needs the real server data is skipped with a reason naming the
missing asset - the real files are **not** on this workstation (verified
2026-10-12: ``/workspace`` does not exist here and no RawVLA NPZ exists under
``/home/jeong``).
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from owac.raw import rawvla_sequence_n02 as raw

LIBERO_SUITES = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_long_no_noops_1.0.0_lerobot",
)
ROBOTWIN_TASKS = ("grab_roller", "lift_pot", "open_laptop")

#: Real cache layout recorded in the deployment dataset check.
SERVER_CACHE_ROOT = "/workspace/owac/rawvla"


# --------------------------------------------------------------------------------------
# fixture construction
# --------------------------------------------------------------------------------------


def write_libero_episode(
    cache_root,
    suite: str,
    episode: int,
    variant: int = 0,
    num_frames: int = 24,
    height: int = 8,
    width: int = 8,
) -> dict:
    """Write one author-schema LIBERO NPZ and return the arrays it stored.

    LIBERO raw frames are uint8 pseudo-RAW with the *author's* range (its own
    0..8 observed maximum recorded on the server, not 0..255), which is exactly
    why the loader must divide by 255 and must not rescale.
    """
    directory = cache_root / "data/libero/raw" / suite / f"episode_{episode:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"variant_{variant:02d}.npz"

    rng = np.random.default_rng(episode * 101 + variant + len(suite))
    agentview = rng.integers(0, 9, size=(num_frames, height, width, 3), dtype=np.uint8)
    wrist = rng.integers(0, 9, size=(num_frames, height, width, 3), dtype=np.uint8)
    actions = np.zeros((num_frames, 7), dtype=np.float32)
    actions[:, 0] = np.arange(num_frames, dtype=np.float32) / 100.0
    actions[:, 6] = np.linspace(-1.0, 1.0, num_frames, dtype=np.float32)
    robot_states = rng.standard_normal((num_frames, 8)).astype(np.float32)
    metadata = {"task_description": f"suite {suite} episode {episode}", "lighting_domain": "normal"}

    np.savez(
        path,
        agentview_raw_uint8=agentview,
        wrist_raw_uint8=wrist,
        actions=actions,
        robot_states=robot_states,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return {
        "path": path,
        "actions": actions,
        "robot_states": robot_states,
        "agentview": agentview,
        "wrist": wrist,
        "metadata": metadata,
    }


def write_robotwin_episode(
    cache_root,
    task: str,
    episode: int,
    num_frames: int = 40,
    height: int = 8,
    width: int = 8,
) -> dict:
    """Write one author-schema RoboTwin NPZ and return the arrays it stored.

    RoboTwin raw frames are float32 HDR-derived RAW with an observed range well
    below 1.0, so the loader must neither divide by 255 nor per-frame normalise.
    """
    directory = cache_root / "data/robotwin2/raw" / task
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{episode:06d}.npz"

    rng = np.random.default_rng(episode * 7919 + len(task))
    head = (rng.random((num_frames, height, width, 3)) * 0.4332).astype(np.float32)
    left = (rng.random((num_frames, height, width, 3)) * 0.3744).astype(np.float32)
    right = (0.0003 + rng.random((num_frames, height, width, 3)) * 0.3734).astype(np.float32)
    actions = np.zeros((num_frames, 14), dtype=np.float32)
    for column in range(14):
        actions[:, column] = np.linspace(-2.0, 2.0, num_frames, dtype=np.float32) + column * 0.1
    robot_states = rng.standard_normal((num_frames, 16)).astype(np.float32)
    metadata = {"task_name": f"{task}_demo", "lighting_domain": "normal"}

    np.savez(
        path,
        head_camera_raw_float32=head,
        left_camera_raw_float32=left,
        right_camera_raw_float32=right,
        actions=actions,
        robot_states=robot_states,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return {
        "path": path,
        "actions": actions,
        "robot_states": robot_states,
        "head": head,
        "left": left,
        "right": right,
        "metadata": metadata,
    }


@pytest.fixture
def libero_cache(tmp_path):
    root = tmp_path / "libero"
    stored = {
        suite: write_libero_episode(root, suite, episode=18, variant=variant)
        for variant, suite in enumerate(LIBERO_SUITES)
    }
    return root, stored


@pytest.fixture
def robotwin_cache(tmp_path):
    root = tmp_path / "robotwin"
    stored = {task: write_robotwin_episode(root, task, episode=18) for task in ROBOTWIN_TASKS}
    return root, stored


def libero_dataset(cache_root, **kwargs) -> raw.RawVLAWindowDataset:
    defaults = {
        "dataset_format": "libero",
        "window": raw.WindowOptions(observation_stride=1, rnn_frames=1, action_horizon=10),
        "split_seed": 0,
        "val_fraction": 0.5,
    }
    defaults.update(kwargs)
    return raw.RawVLAWindowDataset(cache_root, **defaults)


def assert_arrays_close(actual, expected, *, atol: float = 1e-6) -> None:
    """Compare arbitrarily nested numeric arrays elementwise."""
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    assert left.shape == right.shape, (left.shape, right.shape)
    assert np.allclose(left, right, atol=atol, rtol=0.0), float(np.abs(left - right).max())


def current_index(dataset: raw.RawVLAWindowDataset, index: int, num_frames: int) -> int:
    """Reproduce the loader's own frame draw so a test can predict the window."""
    _, _, rng = dataset.indices_for_index(index)
    return int(rng.integers(0, num_frames))


# --------------------------------------------------------------------------------------
# discovery and the no-data path
# --------------------------------------------------------------------------------------


def test_no_cache_reports_no_data_without_inventing_numbers(tmp_path) -> None:
    missing = tmp_path / "absent"
    report = raw.build_report(missing, "libero")
    assert report["status"] == "no_data_found"
    assert report["split"] is None
    assert report["inspection"]["npz_files"] == 0
    assert report["inspection"]["episodes"] == []
    assert report["inspection"]["search_pattern"] == "cache_root_missing"
    assert report["missing_assets"][0]["server_path"] == f"{SERVER_CACHE_ROOT}/data/libero/raw"


def test_cli_run_on_an_empty_cache_root_exits_cleanly(tmp_path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report_path = tmp_path / "report.json"
    exit_code = raw.main(
        [
            "--cache-root",
            str(empty),
            "--format",
            "libero",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    assert "no_data_found" in capsys.readouterr().out
    written = json.loads(report_path.read_text())
    assert written["status"] == "no_data_found"
    assert "episodes" not in written


def test_discovery_finds_episodes_and_reports_the_pattern(tmp_path, libero_cache) -> None:
    root, _ = libero_cache
    paths, pattern = raw.discover_npz_files(root, "libero")
    assert len(paths) == len(LIBERO_SUITES)
    assert pattern == "data/libero/raw/**/*.npz"
    assert paths == sorted(paths)


def test_robotwin_discovery_uses_its_own_marker(tmp_path, robotwin_cache) -> None:
    root, _ = robotwin_cache
    paths, pattern = raw.discover_npz_files(root, "robotwin")
    assert len(paths) == len(ROBOTWIN_TASKS)
    assert pattern == "data/robotwin2/raw/**/*.npz"


# --------------------------------------------------------------------------------------
# window indices, padding and identities
# --------------------------------------------------------------------------------------


def test_window_indices_respect_episode_boundaries(libero_cache) -> None:
    root, _ = libero_cache
    dataset = libero_dataset(
        root,
        window=raw.WindowOptions(
            observation_stride=3, rnn_frames=4, burst_frames=2, action_horizon=10
        ),
    )
    episode = dataset.episodes()[0]
    for index in range(0, 400, 7):
        sample = dataset[index]
        assert sample.burst_indices.min() >= 0
        assert sample.burst_indices.max() <= episode.num_frames - 1
        assert sample.supervision_indices.min() >= 0
        assert sample.supervision_indices.max() <= episode.num_frames - 1
        assert sample.observation_indices.shape == (4,)
        assert sample.burst_indices.shape == (4, 2)


def test_observation_stride_is_applied_to_the_outer_axis(libero_cache) -> None:
    root, _ = libero_cache
    dataset = libero_dataset(
        root,
        window=raw.WindowOptions(observation_stride=2, rnn_frames=4, action_horizon=10),
    )
    for index in (0, 11, 37):
        sample = dataset[index]
        steps = np.diff(sample.observation_indices)
        # A clipped observation repeats an earlier frame, which shows up either as a
        # zero step or as a smaller one; an unclipped window has the exact stride.
        assert set(steps.tolist()) <= {0, 1, 2}
        unclipped = (sample.observation_indices - 2 * np.arange(3, -1, -1) >= 0).all()
        if unclipped:
            assert steps.tolist() == [2, 2, 2]
            assert bool(sample.new_measurement.all())


def test_repeated_padding_frames_carry_new_measurement_false(libero_cache) -> None:
    """Only the first use of a source frame may be flagged as a new measurement."""
    root, _ = libero_cache
    dataset = libero_dataset(
        root,
        window=raw.WindowOptions(
            observation_stride=4, rnn_frames=5, burst_frames=3, action_horizon=10
        ),
    )
    seen_padding = 0
    for index in range(60):
        sample = dataset[index]
        flat = sample.burst_new_measurement.ravel(order="C")
        # A frame is new exactly once in the window's read order.
        assert sample.unique_measurement_count == int(flat.sum())
        indices = sample.burst_indices.ravel(order="C")
        first_seen: set[int] = set()
        for position, value in enumerate(indices.tolist()):
            expected = value not in first_seen
            assert bool(flat[position]) is expected
            first_seen.add(value)
        seen_padding += int((~flat).sum())
    assert seen_padding > 0, "no repeated padding frame appeared in 60 windows"


def test_every_frame_has_episode_camera_and_frame_identity(libero_cache) -> None:
    root, _ = libero_cache
    dataset = libero_dataset(
        root, window=raw.WindowOptions(rnn_frames=3, burst_frames=2, action_horizon=10)
    )
    sample = dataset[5]
    assert sample.camera_ids == ("libero_agentview", "libero_wrist")
    # frame_ids is [camera][rnn step][burst position]; burst_indices is shared by
    # all cameras because they are synchronised.
    assert len(sample.frame_ids) == 2
    assert sample.burst_indices.shape == (sample.observation_indices.size, 2)
    for camera_slot in range(2):
        assert len(sample.frame_ids[camera_slot]) == 3
        for row in sample.frame_ids[camera_slot]:
            assert len(row) == 2
            for frame_id in row:
                assert frame_id.startswith(sample.episode.episode_group_id)
                assert "#frame" in frame_id
        first_id = sample.frame_ids[camera_slot][0][0]
        index = int(first_id.rsplit("#frame", 1)[1])
        assert index == int(sample.burst_indices[0, 0])
    for camera_id in sample.camera_ids:
        payload = sample.measurement_payload(camera_id)
        assert payload["camera_id"] == camera_id
        assert len(payload["frame_ids"]) == 3
        assert len(payload["episode_ids"]) == 3
        assert payload["time_unit"] == "frame"
        assert payload["metadata"]["frame_index_unit"] == "frame"


# --------------------------------------------------------------------------------------
# action indexing and the supervision mask
# --------------------------------------------------------------------------------------


def test_action_indices_follow_the_declared_offset_and_horizon(libero_cache) -> None:
    root, stored = libero_cache
    dataset = libero_dataset(
        root,
        window=raw.WindowOptions(rnn_frames=1, action_offset=3, action_horizon=6),
    )
    checked = 0
    for index in range(40):
        sample = dataset[index]
        episode = sample.episode
        current = current_index(dataset, index, episode.num_frames)
        if current + 3 + 6 > episode.num_frames:
            continue  # this window runs past the action tail; another test covers padding
        checked += 1
        assert sample.supervision_indices.tolist() == list(range(current + 3, current + 9))
        assert bool(sample.supervision_mask.all())
        assert sample.padding_contribution == 0.0
        assert sample.supervision_actions.shape == (6, 7)
        assert sample.padding_contribution == 0.0

        # Without declared statistics the block must be the stored values
        # unchanged, which is what pins the indexing to the file's own numbers.
        actions = stored[episode.acquisition_source]["actions"]
        assert sample.normalisation["normalisation"] == "none"
        assert_arrays_close(sample.supervision_actions, actions[current + 3 : current + 9])
    assert checked >= 5, f"only {checked} unpadded windows were checked"


def test_action_tail_padding_is_masked_and_not_counted(libero_cache) -> None:
    """Padding must be marked so a loss cannot count it twice."""
    root, _ = libero_cache
    dataset = libero_dataset(
        root,
        # LIBERO's declared default offset is 0; state it explicitly here.
        window=raw.WindowOptions(rnn_frames=1, action_offset=0, action_horizon=10),
    )
    padded_samples = 0
    for index in range(120):
        sample = dataset[index]
        episode = sample.episode
        current = current_index(dataset, index, episode.num_frames)
        for offset in range(10):
            inside = current + offset <= episode.num_frames - 1
            assert bool(sample.supervision_mask[offset]) is inside
        if not sample.supervision_mask.all():
            padded_samples += 1
            assert sample.padding_contribution > 0.0
            assert sample.padding_contribution == pytest.approx(
                1.0 - sample.supervision_mask.mean()
            )
            # Every padding entry repeats the episode's last real frame.
            assert sample.supervision_indices[~sample.supervision_mask].tolist() == [
                episode.num_frames - 1
            ] * int((~sample.supervision_mask).sum())
    assert padded_samples > 0, "no window reached the action tail in 120 draws"


def test_supervision_mask_is_independent_of_the_measurement_flags(libero_cache) -> None:
    root, _ = libero_cache
    dataset = libero_dataset(
        root,
        window=raw.WindowOptions(
            observation_stride=3, rnn_frames=3, burst_frames=2, action_horizon=10
        ),
    )
    sample = dataset[40]
    assert sample.supervision_mask.shape == (10,)
    assert sample.new_measurement.shape == (3,)
    assert sample.burst_new_measurement.shape == (3, 2)


# --------------------------------------------------------------------------------------
# units: LIBERO uint8 vs RoboTwin float32
# --------------------------------------------------------------------------------------


def test_libero_uint8_is_divided_by_255(libero_cache) -> None:
    root, stored = libero_cache
    dataset = libero_dataset(root)
    sample = dataset[0]
    slot = sample.camera_ids.index("libero_agentview")
    stored_frames = stored[sample.episode.acquisition_source]["agentview"]
    source = stored_frames[int(sample.burst_indices[0, 0])]
    assert source.shape == (8, 8, 3)
    expected = source.astype(np.float32) / 255.0
    assert_arrays_close(sample.burst_values[slot][0, 0], expected.transpose(2, 0, 1), atol=0.0)
    unit = sample.provenance["cameras"][slot]["metadata"]
    assert unit["unit_divisor"] == 255.0
    assert unit["unit_scale_is_divide"] is True
    assert unit["raw_units"] == "uint8_pseudo_raw"
    assert unit["values_are_author_pseudo_raw"] is True
    assert "not a native Bayer CFA" in unit["note"]


def test_libero_values_stay_in_the_author_range(libero_cache) -> None:
    """The author's uint8 pseudo-RAW peaks at 8, so /255 must stay tiny."""
    root, _ = libero_cache
    dataset = libero_dataset(root)
    maxima = []
    for index in range(20):
        sample = dataset[index]
        for values in sample.burst_values:
            maxima.append(float(values.max()))
    assert max(maxima) <= 8.0 / 255.0 + 1e-6
    assert max(maxima) > 0.0


def test_robotwin_float32_is_not_divided_and_not_max_normalised(robotwin_cache) -> None:
    root, stored = robotwin_cache
    dataset = raw.RawVLAWindowDataset(
        root,
        dataset_format="robotwin",
        window=raw.WindowOptions(observation_stride=1, rnn_frames=1, action_horizon=50),
        split_seed=0,
        val_fraction=0.5,
    )
    sample = dataset[0]
    slot = sample.camera_ids.index("robotwin_head")
    frames = stored[sample.episode.acquisition_source]["head"]
    source = frames[int(sample.burst_indices[0, 0])]
    assert source.shape == (8, 8, 3)
    assert float(source.max()) <= 0.4332 + 1e-6
    # Exact pass-through: no /255, no per-frame normalisation.
    assert_arrays_close(
        sample.burst_values[slot][0, 0], source.transpose(2, 0, 1).astype(np.float32), atol=0.0
    )
    unit = sample.provenance["cameras"][slot]["metadata"]
    assert unit["unit_divisor"] == 1.0
    assert unit["unit_scale_is_divide"] is False
    assert unit["raw_units"] == "float32_hdr_derived_raw"
    assert unit["value_conversion"] == "none"


def test_observed_range_evidence_is_measured_not_assumed(libero_cache, robotwin_cache) -> None:
    libero_root, stored_libero = libero_cache
    dataset = libero_dataset(libero_root)
    sample = dataset[0]
    evidence = sample.range_evidence[0]
    frames = stored_libero[sample.episode.acquisition_source]["agentview"]
    assert evidence["dtype"] == "uint8"
    assert evidence["min"] == float(frames.min())
    assert evidence["max"] == float(frames.max())
    assert evidence["units"] == "cache units (as stored in the NPZ)"

    robotwin_root, stored_robotwin = robotwin_cache
    robotwin = raw.RawVLAWindowDataset(
        robotwin_root,
        dataset_format="robotwin",
        window=raw.WindowOptions(rnn_frames=1, action_horizon=50),
        split_seed=0,
        val_fraction=0.5,
    )
    robotwin_sample = robotwin[0]
    head = stored_robotwin[robotwin_sample.episode.acquisition_source]["head"]
    assert robotwin_sample.range_evidence[0]["dtype"] == "float32"
    assert robotwin_sample.range_evidence[0]["max"] == pytest.approx(float(head.max()))


def test_libero_and_robotwin_units_really_differ(libero_cache, robotwin_cache) -> None:
    libero_root, _ = libero_cache
    robotwin_root, _ = robotwin_cache
    libero_sample = libero_dataset(libero_root)[0]
    robotwin_sample = raw.RawVLAWindowDataset(
        robotwin_root,
        dataset_format="robotwin",
        window=raw.WindowOptions(rnn_frames=1, action_horizon=50),
        split_seed=0,
        val_fraction=0.5,
    )[0]
    assert (
        libero_sample.provenance["cameras"][0]["metadata"]["unit_scale_is_divide"]
        is not robotwin_sample.provenance["cameras"][0]["metadata"]["unit_scale_is_divide"]
    )
    assert str(libero_sample.burst_values[0].dtype) == "float32"
    assert str(robotwin_sample.burst_values[0].dtype) == "float32"


# --------------------------------------------------------------------------------------
# measurement-frame payload
# --------------------------------------------------------------------------------------


def test_build_measurement_frame_reports_absence_truthfully(libero_cache) -> None:
    root, _ = libero_cache
    dataset = libero_dataset(
        root, window=raw.WindowOptions(rnn_frames=3, burst_frames=1, action_horizon=10)
    )
    from owac.core.encoder_contract_n02 import MeasurementFrame

    for index in range(0, 60, 5):
        sample = dataset[index]
        frame = raw.build_measurement_frame(sample, "libero_agentview")
        assert isinstance(frame, MeasurementFrame)
        assert frame.camera_id == "libero_agentview"
        assert frame.time_unit == "frame"
        assert frame.timestamp is None
        # Absent stays absent, never a plausible zero.
        assert frame.metadata_available["timestamp"] is False
        assert frame.metadata_available["executed_action"] is False
        assert frame.metadata_available["exposure"] is False
        assert frame.metadata_available["camera_intrinsics"] is False
        # The declared noise model is present, so the variance is available.
        assert frame.metadata_available["noise_model"] is False
        assert frame.metadata["variance"] is None
        assert frame.metadata["variance_is_estimate"] is True
        # `new_measurement` mirrors the loader's padding flags exactly.
        assert np.array_equal(np.asarray(frame.new_measurement), sample.new_measurement)
        assert len(frame.frame_ids) == sample.new_measurement.size
        assert len(frame.episode_ids) == sample.new_measurement.size
        assert frame.metadata["range_evidence"]["dtype"] == "uint8"
        assert frame.metadata["range_evidence"]["method"] == (
            "min/max over all elements of the loaded array"
        )


def test_declared_noise_model_produces_a_variance(libero_cache) -> None:
    root, _ = libero_cache
    noise = raw.NoiseModel(shot=2.0, read_var=4.0, declared=True, source="unit_test")
    dataset = libero_dataset(root, noise=noise, window=raw.WindowOptions(rnn_frames=1))
    sample = dataset[0]
    # LIBERO values were divided by 255, so DN coefficients are rescaled.
    expected_shot = 2.0 / 255.0
    expected_read = 4.0 / (255.0**2)
    values = sample.burst_values[0][0, 0]
    expected = expected_shot * np.maximum(values, 0.0) + expected_read
    assert sample.variance[0] is not None
    assert_arrays_close(sample.variance[0][0, 0], expected, atol=1e-9)
    assert sample.provenance["noise"]["noise_model_declared"] is True
    assert sample.provenance["noise"]["variance_is_estimate"] is False
    payload = sample.measurement_payload("libero_agentview")
    assert payload["variance"] is not None
    assert payload["variance_is_estimate"] is False


def test_undeclared_noise_returns_none_and_marks_the_estimate(libero_cache) -> None:
    root, _ = libero_cache
    sample = libero_dataset(root)[0]
    assert sample.variance[0] is None
    payload = sample.measurement_payload("libero_agentview")
    assert payload["variance"] is None
    assert payload["variance_is_estimate"] is True
    assert payload["metadata"]["noise"]["shot"] is None

    samples = np.linspace(-1.0, 1.0, 64)
    variance, evidence = raw.fitted_variance_estimate(samples)
    # Declared estimator is mean(residual**2), not the continuous-uniform 1/3.
    assert variance == pytest.approx(float(np.mean(samples**2)), rel=1e-12)
    assert evidence["variance_is_estimate"] is True
    assert evidence["calibrated_probability"] is False
    unavailable, reason = raw.fitted_variance_estimate(np.array([1.0, 2.0]))
    assert np.isnan(unavailable)
    assert reason["estimate_available"] is False


def test_executed_action_is_never_fabricated(libero_cache, robotwin_cache) -> None:
    """A replay cache supervises future actions; it is not an executed-action log."""
    libero_root, _ = libero_cache
    robotwin_root, _ = robotwin_cache
    libero_sample = libero_dataset(libero_root)[0]
    robotwin_sample = raw.RawVLAWindowDataset(
        robotwin_root,
        dataset_format="robotwin",
        window=raw.WindowOptions(rnn_frames=1, action_horizon=50),
        split_seed=0,
        val_fraction=0.5,
    )[0]
    for sample in (libero_sample, robotwin_sample):
        assert sample.executed_action is None
        assert sample.executed_action_available is False
        for camera_id in sample.camera_ids:
            payload = sample.measurement_payload(camera_id)
            assert payload["metadata_available"]["executed_action"] is False
    # LIBERO supervises from the current step; RoboTwin from the next one.
    assert libero_sample.provenance["window"]["action_offset"] == 0
    assert robotwin_sample.provenance["window"]["action_offset"] == 1
    # RoboTwin reuses actions[t] as the *current state*, which is a declared
    # protocol choice, and its supervision starts one step later.
    assert robotwin_sample.dataset_format == "robotwin"
    assert robotwin_sample.provenance["window"]["state_source_key"] == "actions"


# --------------------------------------------------------------------------------------
# normalisation semantics
# --------------------------------------------------------------------------------------


def make_libero_stats(**overrides) -> raw.NormalisationStats:
    width = 7
    state_width = 8
    # Non-identity statistics on purpose: q01=0 / q99=2 makes the declared min-max
    # mapping `2 * (x - 0) / 2 - 1 = x - 1`, which is easy to check by hand.
    payload = {
        "q01": [0.0] * width,
        "q99": [2.0] * width,
        "mean": [0.0] * width,
        "std": [1.0] * width,
        "_state": {
            "q01": [0.0] * state_width,
            "q99": [4.0] * state_width,
            "mean": [0.0] * state_width,
            "std": [1.0] * state_width,
        },
    }
    payload.update(overrides)
    state = payload.pop("_state")
    return raw.NormalisationStats(mode="quantile", action=payload, state=state, source_path="unit")


def test_quantile_normalisation_matches_the_declared_formula() -> None:
    stats = make_libero_stats()
    actions = np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, 1.0]], dtype=np.float32)
    state = np.ones(8, dtype=np.float32)
    normalised, _, metadata = raw.normalise_actions(actions, state, stats, dataset_format="libero")
    # Declared: 2 * (x - q01) / (q99 - q01) - 1, clipped to [-1, 1].
    # Declared x -> x - 1, then clipped to [-1, 1].
    assert float(normalised[0, 0]) == pytest.approx(-1.0)
    assert float(normalised[0, 1]) == pytest.approx(-0.5)
    assert float(normalised[0, 2]) == pytest.approx(-1.0)  # -1.5 clipped
    assert float(normalised[0, 3]) == pytest.approx(0.0)
    assert float(normalised[0, 4]) == pytest.approx(-1.0)  # -2.0 clipped
    assert float(normalised[0, 5]) == pytest.approx(-0.75)
    # The gripper column is (x + 1) / 2, not the min-max form.
    assert float(normalised[0, 6]) == pytest.approx(1.0)
    assert metadata["gripper_index"] == 6
    assert metadata["gripper_rule"] == "(action + 1) * 0.5"

    normalised_state, state_metadata = raw.normalise_state(state, stats)
    assert normalised_state is not None
    # Declared state map is 2 * x / 4 - 1 = x / 2 - 1, and x == 1 gives -0.5.
    assert np.asarray(normalised_state).tolist() == pytest.approx([-0.5] * 8)
    assert state_metadata["state_normalisation"] == "quantile"


def test_starvla_minmax_reproduces_the_authors_robotwin_recipe() -> None:
    width = 14
    stats = raw.NormalisationStats(
        mode="starvla_minmax",
        action={
            "min": [0.0] * width,
            "max": [1.0] * width,
        },
        state=None,
        source_path="unit",
    )
    actions = np.zeros((1, width), dtype=np.float32)
    actions[0, :12] = 0.5
    actions[0, 12] = 0.6  # gripper above the declared 0.49 threshold
    actions[0, 13] = 0.4
    normalised, _, metadata = raw.normalise_actions(actions, None, stats, dataset_format="robotwin")
    order = metadata["joint_order"]
    assert order == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    assert [float(normalised[0, index]) for index in range(11)] == pytest.approx([0.0] * 11)
    # After the reorder the 12 arm dims come first, so position 11 is
    # original index 12 (0.6 -> 0.2) and the grippers land at 12 and 13.
    assert float(normalised[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(normalised[0, 11]) == pytest.approx(0.2, abs=1e-6)
    assert normalised[0, 13] == pytest.approx(0.0)
    assert metadata["gripper_rule"] == "(value > 0.49) -> {0, 1}"


def test_normalisation_is_refused_when_statistics_are_absent() -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    normalised, state, metadata = raw.normalise_actions(
        actions, None, None, dataset_format="libero"
    )
    assert normalised.tolist() == actions.tolist()
    assert state is None
    assert metadata["normalisation"] == "none"
    assert "no declared statistics" in metadata["reason"]
    state_vector, state_metadata = raw.normalise_state(np.zeros(8), None)
    assert state_vector is None
    assert state_metadata["state_normalisation"] == "unavailable"


def test_starvla_minmax_is_rejected_for_libero() -> None:
    stats = raw.NormalisationStats(
        mode="starvla_minmax", action={"min": [0.0] * 14, "max": [1.0] * 14}
    )
    with pytest.raises(ValueError, match="RoboTwin"):
        raw.normalise_actions(
            np.zeros((1, 14), dtype=np.float32), None, stats, dataset_format="libero"
        )


def test_statistics_file_is_read_and_hashed(tmp_path) -> None:
    path = tmp_path / "norm_stats.json"
    payload = {
        "norm_stats": {
            "actions": {"q01": [0.0] * 7, "q99": [1.0] * 7},
            "state": {"q01": [0.0] * 8, "q99": [1.0] * 8},
        }
    }
    path.write_text(json.dumps(payload))
    stats = raw.NormalisationStats.from_json(path, mode="quantile")
    assert stats.action is not None
    assert stats.state is not None
    assert len(stats.sha256) == 64
    assert stats.source_path == str(path)

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"action": {"min": [0.0] * 14, "max": [1.0] * 14}}))
    bare_stats = raw.NormalisationStats.from_json(bare, mode="starvla_minmax")
    assert bare_stats.action is not None
    assert bare_stats.state is None

    with pytest.raises(FileNotFoundError):
        raw.NormalisationStats.from_json(tmp_path / "absent.json", mode="quantile")


# --------------------------------------------------------------------------------------
# split manifest
# --------------------------------------------------------------------------------------


def test_split_is_deterministic_across_runs(libero_cache, tmp_path) -> None:
    root, _ = libero_cache
    first = libero_dataset(root, val_fraction=0.5).split_manifest()
    second = libero_dataset(root, val_fraction=0.5).split_manifest()
    assert first.as_dict() == second.as_dict()
    assert first.check() == []

    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    digest_a = raw.write_split_manifest(first, path_a)
    digest_b = raw.write_split_manifest(second, path_b)
    assert digest_a == digest_b
    assert path_a.read_bytes() == path_b.read_bytes()


def test_split_keeps_a_source_on_one_side(libero_cache, robotwin_cache) -> None:
    for root, dataset_format, sources in (
        (libero_cache[0], "libero", LIBERO_SUITES),
        (robotwin_cache[0], "robotwin", ROBOTWIN_TASKS),
    ):
        dataset = raw.RawVLAWindowDataset(
            root,
            dataset_format=dataset_format,
            split_seed=1,
            val_fraction=0.5,
        )
        manifest = dataset.split_manifest()
        assert manifest.check() == []
        assert set(manifest.train_sources) & set(manifest.val_sources) == set()
        assert set(manifest.train_sources) | set(manifest.val_sources) == set(sources)
        assert len(manifest.train_episodes) + len(manifest.val_episodes) == len(sources)

        # The two sides must physically select disjoint files.
        dataset.split = "train"
        train_paths = {episode.path for episode in dataset.split_episodes()}
        dataset.split = "validation"
        validation_paths = {episode.path for episode in dataset.split_episodes()}
        assert train_paths & validation_paths == set()
        assert train_paths | validation_paths == set(dataset.paths)


def test_split_never_separates_variants_of_one_episode(tmp_path) -> None:
    """Every variant of an episode is one acquisition source and stays together."""
    root = tmp_path / "libero"
    for variant in range(4):
        write_libero_episode(root, LIBERO_SUITES[0], episode=3, variant=variant)
    write_libero_episode(root, LIBERO_SUITES[1], episode=3, variant=0)
    write_libero_episode(root, LIBERO_SUITES[1], episode=4, variant=0)
    manifest = libero_dataset(root, val_fraction=0.5).split_manifest()
    variants = sorted(
        episode["relative_path"]
        for episode in manifest.episodes
        if episode["acquisition_source"] == LIBERO_SUITES[0]
    )
    assert len(variants) == 4
    side = "train" if variants[0] in manifest.train_episodes else "validation"
    chosen = set(manifest.train_episodes if side == "train" else manifest.val_episodes)
    assert set(variants) <= chosen
    assert manifest.check() == []


def test_split_manifest_round_trips(libero_cache, tmp_path) -> None:
    root, _ = libero_cache
    manifest = libero_dataset(root, val_fraction=0.5).split_manifest()
    path = tmp_path / "split.json"
    raw.write_split_manifest(manifest, path)
    reloaded = raw.load_split_manifest(path)
    assert reloaded.as_dict() == manifest.as_dict()
    assert reloaded.check() == []


def test_dataset_index_is_deterministic_and_split_scoped(libero_cache) -> None:
    root, _ = libero_cache
    first = libero_dataset(root, val_fraction=0.5, seed=3)
    second = libero_dataset(root, val_fraction=0.5, seed=3)
    for index in (0, 1, 5):
        left = first[index]
        right = second[index]
        assert left.episode.relative_path == right.episode.relative_path
        assert np.array_equal(left.burst_indices, right.burst_indices)
        assert np.array_equal(left.supervision_indices, right.supervision_indices)
        assert np.array_equal(left.new_measurement, right.new_measurement)
        assert left.split == "train"


# --------------------------------------------------------------------------------------
# CFA path (a genuinely mosaic input)
# --------------------------------------------------------------------------------------


def write_cfa_episode(cache_root, suite: str, episode: int, num_frames: int = 12) -> dict:
    """Write a single-mosaic NPZ: one CFA photosite per element."""
    directory = cache_root / "data/libero/raw" / suite / f"episode_{episode:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "variant_00.npz"
    rng = np.random.default_rng(4242)
    mosaic = rng.integers(0, 256, size=(num_frames, 16, 16), dtype=np.uint8)
    actions = rng.standard_normal((num_frames, 7)).astype(np.float32)
    np.savez(
        path,
        raw_mosaic_uint8=mosaic,
        actions=actions,
        metadata_json=np.array(json.dumps({"task_description": "cfa"})),
    )
    return {"path": path, "mosaic": mosaic}


def test_cfa_planes_keep_their_true_offsets(tmp_path) -> None:
    root = tmp_path / "cfa"
    stored = write_cfa_episode(root, "cfa_suite", episode=0)
    write_cfa_episode(root, "cfa_other", episode=0)
    # The shipped caches are three-channel, so a genuinely mosaic input needs a
    # declared single-plane spec; the loader must refuse an RGB array as a mosaic.
    spec = replace(
        raw.FORMAT_SPECS["libero"],
        measurement_keys=("raw_mosaic_uint8",),
        camera_ids=("cfa",),
        state_source_key="actions",
        measurement_channels=1,
        default_action_horizon=4,
        default_action_offset=0,
    )
    dataset = raw.RawVLAWindowDataset(
        root,
        dataset_format="libero",
        split_seed=0,
        val_fraction=0.5,
        cfa_pattern="GBRG",
        format_spec=spec,
        window=raw.WindowOptions(rnn_frames=1, action_horizon=4, action_offset=0),
    )
    assert dataset.window.state_source_key == "actions"
    assert dataset.window.action_offset == 0
    sample = dataset[0]
    assert sample.burst_values[0].shape == (1, 1, 4, 8, 8)
    unit = sample.provenance["cameras"][0]["metadata"]
    assert unit["cfa_pattern"] == "GBRG"
    assert unit["planes_are_not_co_sited"] is True

    frame = dataset._load_cfa_frame(stored["mosaic"], int(sample.burst_indices[0, 0]))
    planes = np.asarray(sample.burst_values[0][0, 0])
    # No CFA calibration was declared, so the planes keep the cache's own integer
    # units: there is no implicit /255 on this path.
    assert planes.shape == (4, 8, 8)
    for plane_index, colour in enumerate(raw.PLANE_COLOURS):
        assert_arrays_close(planes[plane_index], frame[plane_index], atol=0.0), colour
    assert float(planes.max()) > 1.0, "uncorrected mosaics must stay in cache units"
    assert unit["value_conversion"] == "none"
    assert unit["unit_scale_is_divide"] is False
    # The four planes must not be four co-sited copies.
    assert float(np.abs(planes[0] - planes[3]).max()) > 0.0
    # And each plane must match the declared plane coordinates in the mosaic.
    positions = raw.cfa_plane_positions("GBRG")
    mosaic_frame = stored["mosaic"][int(sample.burst_indices[0, 0])]
    for plane_index in range(4):
        y, x = positions[plane_index]
        row, column = int(y - 0.5), int(x - 0.5)
        expected_row = mosaic_frame[row::2, column::2].astype(np.float32)
        assert_arrays_close(planes[plane_index], expected_row, atol=0.0)


def test_cfa_plane_positions_follow_the_declared_pattern() -> None:
    for pattern, tiles in raw.CFA_PATTERNS.items():
        positions = raw.cfa_plane_positions(pattern)
        assert positions.shape == (4, 2)
        for plane_index, name in enumerate(raw.PLANE_COLOURS):
            y, x = positions[plane_index]
            tile = (int(y - 0.5), int(x - 0.5))
            expected = "G" if name in {"G1", "G2"} else name
            assert tiles[tile[0] * 2 + tile[1]] == expected, (pattern, name)
        # No two planes share a coordinate.
        for first in range(4):
            for second in range(first + 1, 4):
                assert np.abs(positions[first] - positions[second]).max() >= 1.0


def test_cfa_calibration_keeps_missing_bad_and_saturated_separate() -> None:
    mosaic = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    calibration = raw.CfaCalibration(
        black_level=0.0,
        white_level=30.0,
        missing_mask=np.array([[True, False], [False, False]]),
        bad_mask=np.array([[False, True], [False, False]]),
        saturated_mask=np.array([[False, False], [False, True]]),
    )
    normalised, metadata = raw.apply_cfa_calibration(mosaic, calibration)
    assert np.isnan(normalised[0, 0])
    assert np.isnan(normalised[0, 1])
    assert normalised[1, 0] == pytest.approx(20.0 / 30.0)
    assert normalised[1, 1] == pytest.approx(1.0)
    assert metadata["missing_count"] == 1
    assert metadata["bad_count"] == 1
    assert metadata["saturated_count"] == 1
    assert metadata["analog_gain"] is None
    assert metadata["exposure_s"] is None
    assert "no sample was acquired" in metadata["missing_mask_semantics"]


def test_cfa_plane_extraction_matches_the_mosaic() -> None:
    mosaic = np.arange(64, dtype=np.uint8).reshape(8, 8)
    planes = raw.cfa_plane_values(mosaic, "RGGB")
    assert planes.values.shape == (4, 4, 4)
    assert planes.values[0].tolist() == mosaic[0::2, 0::2].astype(np.float32).tolist()
    assert planes.values[3].tolist() == mosaic[1::2, 1::2].astype(np.float32).tolist()
    coordinates = planes.coordinates()
    assert coordinates.shape == (4, 4, 4, 2)
    assert coordinates[0, 0, 0].tolist() == [0.5, 0.5]
    assert coordinates[1, 0, 0].tolist() == [0.5, 1.5]
    # A crop must move the recorded coordinates, not re-phase the pattern.
    shifted = raw.cfa_plane_values(mosaic, "RGGB", grid_offset=(2, 0))
    assert shifted.positions[0].tolist() == [2.5, 0.5]
    assert shifted.coordinates()[0, 0, 0].tolist() == [2.5, 0.5]


def test_cfa_calibration_rejects_a_degenerate_white_level() -> None:
    with pytest.raises(ValueError):
        raw.CfaCalibration(black_level=10.0, white_level=10.0)


# --------------------------------------------------------------------------------------
# inspection report
# --------------------------------------------------------------------------------------


def test_report_summarises_a_real_cache(libero_cache, tmp_path) -> None:
    root, _ = libero_cache
    report_path = tmp_path / "report.json"
    exit_code = raw.main(
        [
            "--cache-root",
            str(root),
            "--format",
            "libero",
            "--report",
            str(report_path),
            "--val-fraction",
            "0.5",
            "--action-horizon",
            "10",
            "--split-manifest",
            str(tmp_path / "split.json"),
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok"
    assert report["episodes"]["count"] == len(LIBERO_SUITES)
    assert report["episodes"]["total_frames"] == 24 * len(LIBERO_SUITES)
    assert report["split"]["problems"] == []
    assert report["split"]["train"]["frames"] + report["split"]["validation"]["frames"] == 24 * len(
        LIBERO_SUITES
    )
    assert report["probe_sample"]["train"]["burst_shape"]
    assert (tmp_path / "split.json").is_file()
    reread = raw.load_split_manifest(tmp_path / "split.json")
    assert reread.check() == []
    assert report["split_manifest"]["sha256"] == raw.write_split_manifest(
        reread, tmp_path / "again.json"
    )


def test_inspection_measures_shapes_and_ranges(libero_cache) -> None:
    root, stored = libero_cache
    evidence = raw.inspect_cache(root, "libero")
    assert evidence["status"] == "ok"
    assert evidence["npz_files"] == len(LIBERO_SUITES)
    assert evidence["problems"] == []
    record = evidence["episodes"][0]
    assert record["agentview_raw_uint8"]["dtype"] == "uint8"
    assert record["agentview_raw_uint8"]["shape"] == [24, 8, 8, 3]
    assert record["robot_states"]["shape"] == [24, 8]
    assert record["num_frames"] == 24
    assert len(record["sha256"]) == 64
    suite = record["relative_path"].split("/")[-3]
    assert record["agentview_raw_uint8"]["max"] == float(stored[suite]["agentview"].max())
    assert evidence["key_availability"]["actions"]["files"] == len(LIBERO_SUITES)


def test_inspection_reports_missing_keys(tmp_path) -> None:
    directory = tmp_path / "broken" / "data/libero/raw" / "suite_a" / "episode_000000"
    directory.mkdir(parents=True)
    np.savez(
        directory / "variant_00.npz",
        actions=np.zeros((4, 7), dtype=np.float32),
        metadata_json=np.array("{}"),
    )
    evidence = raw.inspect_cache(tmp_path / "broken", "libero")
    assert evidence["status"] == "ok"
    problems = " ".join(evidence["problems"])
    assert "agentview_raw_uint8" in problems
    assert "robot_states" in problems


# --------------------------------------------------------------------------------------
# server-only assets
# --------------------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Real RawVLA-Bench NPZ cache is not present on this workstation: "
        f"/workspace does not exist and no NPZ exists under /home/jeong. The real files "
        f"live on the deployed server at {SERVER_CACHE_ROOT}/data/libero/raw (LIBERO) and "
        f"{SERVER_CACHE_ROOT}/data/robotwin2/raw (RoboTwin), per "
        "research/experiments/results/2026-10-10_rawvla_pi05_deployment/dataset-check.json."
    )
)
def test_real_libero_cache_matches_the_recorded_evidence() -> None:
    """Server-only: read the real 1771-file LIBERO cache and check the ranges."""
    dataset = raw.RawVLAWindowDataset(
        f"{SERVER_CACHE_ROOT}/data/libero/raw", dataset_format="libero"
    )
    assert len(dataset.episodes()) == 1771
    evidence = raw.inspect_cache(f"{SERVER_CACHE_ROOT}/data/libero/raw", "libero")
    assert evidence["status"] == "ok"


@pytest.mark.skip(
    reason=(
        "Real RoboTwin RawVLA-Bench NPZ cache is not present on this workstation; it is "
        f"on the deployed server at {SERVER_CACHE_ROOT}/data/robotwin2/raw (632 files)."
    )
)
def test_real_robotwin_cache_matches_the_recorded_evidence() -> None:
    dataset = raw.RawVLAWindowDataset(
        f"{SERVER_CACHE_ROOT}/data/robotwin2/raw", dataset_format="robotwin"
    )
    assert len(dataset.episodes()) == 632
