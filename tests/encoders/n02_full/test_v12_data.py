"""v1.2 measurement-completion tests: real windows, three-state checks, real interventions.

These run on the contract fixture caches written by :mod:`conftest` (CPU only).  They check the
mechanisms that P1 got wrong: the frozen-anchor reproduction of the regenerated history
window, the causal frame order, the deduplicated measurement identity, one real three-state
return type with a separately counted invalid case, and interventions that really change RAW
pixels including the history rows.  They are *not* evidence about the real RawVLA cache; the
real-cache probes are recorded in the P2 report.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
import torch

from owac.frameworks.n02_full.checks import ResidualView, ViewingRule
from owac.frameworks.n02_full.checks_v12 import (
    CalibrationIsolationError,
    N02V12Measurement,
    WindowedCalibration,
    calibrate_windowed,
    summarise_results,
)
from owac.frameworks.n02_full.data import build_anchor_table, build_partition_manifest
from owac.frameworks.n02_full.data_v12 import (
    PARTITIONS,
    V12WindowBatch,
    V12WindowSource,
    anchor_signature,
)
from owac.frameworks.n02_full.interventions_v12 import (
    cue_swap,
    exposure_shift,
    no_information,
    noise_shift,
    translation_shift,
)
from owac.raw.rawvla_sequence_n02 import RawVLAWindowDataset

HEIGHT = 24
WIDTH = 32


def build_cache(writer, tmp_path, *, platform: str, episodes: int = 3, frames: int = 30):
    for episode in range(episodes):
        if platform == "robotwin":
            writer(tmp_path, "grab_roller", episode, num_frames=frames, height=HEIGHT, width=WIDTH)
        else:
            for variant in (0, 1):
                writer(
                    tmp_path,
                    "libero_spatial_no_noops_1.0.0_lerobot",
                    episode,
                    variant,
                    num_frames=frames,
                )
    loader = RawVLAWindowDataset(tmp_path, dataset_format=platform)
    manifest = build_partition_manifest(
        loader.episodes(), dataset_format=platform, cache_root=str(tmp_path), seed=1103
    )
    return manifest


def frozen_receipt(tmp_path, manifest, partition: str):
    table = build_anchor_table(manifest, partition)
    path = tmp_path / f"frozen-{manifest.dataset_format}-{partition}.json.gz"
    path.write_bytes(gzip.compress(table.to_json().encode("utf-8")))
    return table, path


@pytest.fixture()
def robotwin_cache(tmp_path, robotwin_writer):
    root = tmp_path / "robotwin"
    return root, build_cache(robotwin_writer, root, platform="robotwin")


@pytest.fixture()
def libero_cache(tmp_path, libero_writer):
    root = tmp_path / "libero"
    return root, build_cache(libero_writer, root, platform="libero")


def make_source(root, manifest, *, partition, history=8, anchor_path=None, **kwargs):
    return V12WindowSource(
        dataset_format=manifest.dataset_format,
        data_root=str(root),
        partition=partition,
        history=history,
        anchor_path=anchor_path,
        seed=1103,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# data_v12: frozen anchors, causal history, measurement identity
# --------------------------------------------------------------------------------------


def test_source_reproduces_frozen_anchors_and_regenerates_history(robotwin_cache):
    root, manifest = robotwin_cache
    partition = "train"
    frozen, anchor_path = frozen_receipt(root, manifest, partition)
    assert frozen.rnn_frames == 1
    source = make_source(root, manifest, partition=partition, anchor_path=anchor_path)
    identity = source.identity()
    assert identity["frozen_anchor_receipt"]["reproduced_from_manifest"] is True
    assert identity["frozen_anchor_identity_preserved"] is True
    assert identity["history"] == 8
    assert source.window_options.rnn_frames == 8
    # the regenerated anchors keep the frozen anchor frames and their supervised action bounds
    assert anchor_signature(source.anchor_table) == anchor_signature(frozen)
    for entry in source.anchor_table:
        expected = tuple(
            int(value)
            for value in np.clip(
                entry.current_frame_index - np.arange(7, -1, -1),
                0,
                manifest.entry_of(entry.relative_path).num_frames - 1,
            )
        )
        assert entry.history_indices == expected


def test_source_refuses_an_anchor_receipt_the_manifest_cannot_reproduce(robotwin_cache):
    import dataclasses

    root, manifest = robotwin_cache
    table, _ = frozen_receipt(root, manifest, "train")
    tampered = dataclasses.replace(table, entries=table.entries[1:])
    path = root / "tampered-anchors.json.gz"
    path.write_bytes(gzip.compress(tampered.to_json().encode("utf-8")))
    with pytest.raises(ValueError, match="does not reproduce the frozen anchors"):
        make_source(root, manifest, partition="train", anchor_path=path)


def test_source_accepts_a_receipt_manifest_and_serves_the_same_partition(robotwin_cache):
    root, manifest = robotwin_cache
    _, anchor_path = frozen_receipt(root, manifest, "train")
    path = root / "manifest.json"
    path.write_text(manifest.to_json())
    source = V12WindowSource(
        dataset_format="robotwin",
        data_root=str(root),
        partition="train",
        history=4,
        anchor_path=anchor_path,
        manifest_path=path,
    )
    assert source.identity()["manifest"]["source"]["kind"] == "receipt_file"
    assert source.identity()["frozen_anchor_identity_preserved"] is True


def test_batch_rows_are_causal_and_deduplicated(robotwin_cache):
    root, manifest = robotwin_cache
    source = make_source(root, manifest, partition="train", history=8)
    batch = source.window(10, 12)
    assert isinstance(batch, V12WindowBatch)
    assert batch.batch_size == 2
    assert batch.row_count == 16
    frame_index = np.asarray(batch.frame_index)
    anchors = batch.current_frame_indices()
    assert len(anchors) == 2
    for element in range(batch.batch_size):
        rows = frame_index[element * 8 : (element + 1) * 8]
        assert rows[-1] == anchors[element]
        assert np.all(rows <= anchors[element]), "a row must never read a future frame"
        assert np.all(np.diff(rows) >= 0), "history must be oldest first"
        expected = np.clip(anchors[element] - np.arange(7, -1, -1), 0, 29)
        assert np.array_equal(rows, expected)
    # repeated padding rows collapse to one measurement per camera
    first = source.window(0, 1)
    assert first.unique_measurement_count == len(first.observation.frames)
    assert first.unique_measurement_count < first.row_count * len(first.observation.frames)
    # the unique set is exactly (episode, frame, camera), counted as a plain set
    for episode_id, frame_id, camera_id in batch.measurement_ids:
        assert camera_id in batch.observation.frames
        assert frame_id.startswith(episode_id)
    assert batch.measurement_id_set() == set(batch.measurement_ids)
    assert len(batch.measurement_ids) == batch.unique_measurement_count


def _stats_file(root, dim: int, name: str = "norm_stats.json"):
    """Minimal declared statistics so the loader really normalises (nothing is inferred)."""
    payload = {
        "action": {"mean": [0.0] * dim, "std": [1.0] * dim},
        "state": {"mean": [0.0] * dim, "std": [1.0] * dim},
    }
    path = root / name
    path.write_text(json.dumps(payload))
    return path


def test_batch_context_keeps_one_proprio_row_per_causal_row(robotwin_cache):
    root, manifest = robotwin_cache
    source = make_source(
        root, manifest, partition="train", history=4, stats_path=_stats_file(root, 14)
    )
    batch = source.window(3, 4)
    proprio = np.asarray(batch.observation.context.proprio)
    sample = source.sample(3)
    assert proprio.shape == (batch.row_count, np.asarray(sample.proprio).shape[-1])
    # every causal row of one anchor carries that anchor's own current proprio
    assert np.allclose(proprio, np.asarray(sample.proprio, dtype=np.float32)[None, :])
    assert source.identity()["stats"]["mode"] == "zscore"
    targets = source.targets(3, 4)
    assert targets["action_dim"] == 14
    assert targets["action_horizon"] == 50
    assert targets["action_offset"] == 1
    assert targets["actions"].shape == (50, 14)
    assert bool(np.isfinite(targets["actions"]).all())


def test_partition_identity_is_not_shared_between_partitions(libero_cache):
    root, manifest = libero_cache
    for partition in PARTITIONS:
        assert manifest.partitions[partition], partition
    train_paths = set(manifest.partitions["train"])
    dev_paths = set(manifest.partitions["dev"])
    calibration_paths = set(manifest.partitions["calibration"])
    assert not train_paths & dev_paths
    assert not train_paths & calibration_paths
    assert not dev_paths & calibration_paths
    sources = {
        partition: make_source(root, manifest, partition=partition, history=4)
        for partition in PARTITIONS
    }
    digests = {}
    for partition, source in sources.items():
        batch = source.window(0, 1)
        for camera_id, frame in batch.observation.frames.items():
            digests[(partition, camera_id)] = hash(np.asarray(frame.values).tobytes())
    train_digests = {value for (partition, _), value in digests.items() if partition == "train"}
    other_digests = {value for (partition, _), value in digests.items() if partition != "train"}
    assert train_digests and other_digests
    assert not train_digests & other_digests


def test_source_rejects_unknown_partition_and_serves_identity(robotwin_cache):
    root, manifest = robotwin_cache
    with pytest.raises(ValueError, match="partition must be one of"):
        make_source(root, manifest, partition="test")
    source = make_source(root, manifest, partition="dev", history=3)
    identity = source.identity()
    assert identity["partition"] == "dev"
    assert identity["history"] == 3
    assert identity["manifest"]["content_hash"] == manifest.content_hash()
    assert identity["anchor_table"]["entries"] == source.anchor_count
    assert identity["partition_verification"]["violations"] == []
    assert identity["stats"] is None  # no statistics declared, so nothing is normalised
    assert len(source) == source.anchor_count  # batch_size 1


# --------------------------------------------------------------------------------------
# checks_v12: energy, coverage, real three-state return type
# --------------------------------------------------------------------------------------


def _rule() -> ViewingRule:
    return ViewingRule(scales=("raw_full",), frames=(0,), regions=("full",), window=2)


def _group(value: float, *, valid: bool = True) -> list[ResidualView]:
    mask = torch.full((4, 4), valid, dtype=torch.bool)
    return [
        ResidualView(
            standardised_residual=torch.full((4, 4), float(value)),
            valid=mask,
            scale="raw_full",
            frame_index=0,
            region_id="full",
        )
    ]


def test_energy_is_a_complete_objective_with_the_variance_term():
    measurement = N02V12Measurement()
    mean = torch.zeros(2, 3, 2, 2)
    target = torch.ones(2, 2, 2)
    masks = torch.ones(2, 2, 2, dtype=torch.bool)
    # residual is exactly 1 everywhere, so v = r^2 = 1 is the minimum of 0.5*(r^2/v + log v)
    at_minimum = measurement.energy(
        mean=mean, variance=torch.ones(2, 3, 2, 2), masks=masks, target=target
    )
    inflated = measurement.energy(
        mean=mean, variance=torch.full((2, 3, 2, 2), 4.0), masks=masks, target=target
    )
    overconfident = measurement.energy(
        mean=mean, variance=torch.full((2, 3, 2, 2), 1e-3), masks=masks, target=target
    )
    assert at_minimum.shape == inflated.shape == overconfident.shape == (2, 3)
    # past v = (y-mu)^2 the log term makes an inflated variance strictly worse, so variance
    # cannot run away for free; a far too small variance is worse as well
    assert bool((inflated > at_minimum).all())
    assert bool((overconfident > at_minimum).all())
    per_valid_scalar = float(at_minimum[0, 0]) / 4.0
    assert per_valid_scalar == pytest.approx(0.5 * (1.0 + float(np.log(2 * np.pi))), rel=1e-6)
    empirical = measurement.energy(mean=mean, variance=None, masks=masks, target=target)
    assert bool(torch.isfinite(empirical).all())
    _, info = measurement.objective(mean=mean, variance=None, masks=masks, target=target)
    assert info["kind"] == "empirical_energy"
    with pytest.raises(ValueError, match=r"\(B, K, \.\.\.\)"):
        measurement.energy(mean=torch.zeros(4), variance=None, masks=masks, target=target)


def test_energy_counts_every_valid_cell_and_reports_coverage():
    measurement = N02V12Measurement()
    masks = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    masks[0, 0, 0, 0] = False
    target = torch.ones(1, 4, 4)
    _, info = measurement.objective(
        mean=torch.zeros(1, 2, 4, 4), variance=torch.ones(1, 2, 4, 4), masks=masks, target=target
    )
    assert info["n_valid_per_candidate"] == [15.0, 15.0]
    assert info["sampled_all_valid_cells"] is True
    coverage = measurement.coverage(masks=masks, target=target)
    assert coverage["n_used_cells"] == coverage["n_valid_cells"] == 15
    assert coverage["uses_every_valid_cell"] is True
    assert coverage["used_mean_minus_full_sensor_mean"] == 0.0
    # a fixed-stride style subsample would show up as a partial fraction and a shifted mean
    weights = torch.zeros(1, 4, 4)
    weights[:, ::2, ::2] = 1.0
    strided = measurement.coverage(
        masks=masks, target=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4), weights=weights
    )
    assert strided["fraction_used_of_valid"] < 1.0
    assert strided["uses_every_valid_cell"] is False


def test_calibration_requires_the_independent_split_and_disjoint_ids():
    groups = [_group(0.1 + 0.01 * index) for index in range(20)]
    with pytest.raises(CalibrationIsolationError, match="independent calibration split"):
        calibrate_windowed(groups, partition="dev", rule=_rule())
    with pytest.raises(CalibrationIsolationError, match="reuses"):
        calibrate_windowed(
            groups,
            partition="calibration",
            rule=_rule(),
            calibration_ids=["a"],
            test_ids=["a", "b"],
        )
    calibration = calibrate_windowed(groups, partition="calibration", rule=_rule())
    assert isinstance(calibration, WindowedCalibration)
    assert calibration.threshold == pytest.approx(0.29)
    assert calibration.n_sequences == 20
    assert calibration.as_dict()["partition"] == "calibration"


def test_absolute_check_returns_real_states_and_never_a_none_string():
    measurement = N02V12Measurement()
    groups = [_group(0.1 + 0.01 * index) for index in range(20)]
    calibration = calibrate_windowed(groups, partition="calibration", rule=_rule())

    compatible = measurement.absolute_check(
        residuals=_group(0.15), calibrator=calibration, rule=_rule()
    )
    inadequate = measurement.absolute_check(
        residuals=_group(5.0), calibrator=calibration, rule=_rule()
    )
    insufficient = measurement.absolute_check(
        residuals=_group(0.1, valid=False), calibrator=calibration, rule=_rule()
    )
    invalid = measurement.absolute_check(residuals=_group(0.1), calibrator=None, rule=_rule())

    assert compatible.status == "compatible" and compatible.valid is True
    assert inadequate.status == "model_inadequate" and inadequate.valid is True
    assert insufficient.status == "insufficient_information" and insufficient.valid is True
    assert invalid.valid is False
    assert invalid.status in ("compatible", "insufficient_information", "model_inadequate")
    assert invalid.statistics["reason"] == "no_fitted_calibrator"
    payload = [item.as_dict() for item in (compatible, inadequate, insufficient, invalid)]
    assert not _contains_none_string(payload)
    assert compatible.explanation and _valid_explanations(payload)

    summary = summarise_results([compatible, inadequate, insufficient, invalid])
    assert summary["total"] == 4
    assert summary["valid"] == 3
    assert summary["invalid"] == 1
    assert summary["compatible"] == 1
    assert summary["insufficient_information"] == 1
    assert summary["model_inadequate"] == 1
    assert not _contains_none_string(summary)
    assert compatible.counted_as_detection is False and invalid.counted_as_false_positive is False


def _contains_none_string(payload) -> bool:
    """True when a value in the payload is the literal missing-value string 'None'."""
    if isinstance(payload, str):
        return payload == "None"
    if isinstance(payload, dict):
        return any(_contains_none_string(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_contains_none_string(value) for value in payload)
    return False


def _valid_explanations(payload) -> bool:
    """Every row states what its status obliges, and the invalid row says it is invalid."""
    by_valid = {item["valid"]: item for item in payload}
    return "invalid result" in by_valid[False]["explanation"] and all(
        "invalid result" not in item["explanation"] for item in payload if item["valid"]
    )


def test_absolute_check_refuses_to_reuse_calibration_measurements():
    measurement = N02V12Measurement()
    groups = [_group(0.1 + 0.01 * index) for index in range(20)]
    calibration = calibrate_windowed(groups, partition="calibration", rule=_rule())
    with pytest.raises(CalibrationIsolationError, match="reuses"):
        measurement.absolute_check(
            residuals=_group(0.2),
            calibrator=calibration,
            rule=_rule(),
            measurement_ids=["ep#frame000001"],
            calibration_ids=["ep#frame000001"],
        )


def test_absolute_check_requires_a_declared_prediction_version():
    measurement = N02V12Measurement()
    groups = [_group(0.1 + 0.01 * index) for index in range(20)]
    calibration = calibrate_windowed(groups, partition="calibration", rule=_rule())
    result = measurement.absolute_check(
        residuals=_group(0.2),
        calibrator=calibration,
        rule=_rule(),
        prediction_version="refit_on_current_frame",
    )
    assert result.valid is False
    assert result.statistics["reason"] == "invalid_prediction_version"


def test_rule_mismatch_is_an_explicit_failure():
    measurement = N02V12Measurement()
    groups = [_group(0.1 + 0.01 * index) for index in range(20)]
    calibration = calibrate_windowed(groups, partition="calibration", rule=_rule())
    other = ViewingRule(scales=("raw_full",), frames=(0,), regions=("full",), window=3)
    with pytest.raises(ValueError, match="identical viewing rule"):
        measurement.absolute_check(residuals=_group(0.2), calibrator=calibration, rule=other)


# --------------------------------------------------------------------------------------
# interventions_v12: the change happens in the real RAW pixels
# --------------------------------------------------------------------------------------


@pytest.fixture()
def served_batch(robotwin_cache):
    root, manifest = robotwin_cache
    source = make_source(root, manifest, partition="train", history=4)
    return source.window(5, 6)


def _boxes():
    return {"left": (0, HEIGHT, 0, WIDTH // 2), "right": (0, HEIGHT, WIDTH // 2, WIDTH)}


def test_cue_swap_changes_pixels_on_the_history_rows_too(served_batch):
    swapped, record = cue_swap(served_batch, eps=1.0, positions=_boxes())
    assert record["kind"] == "cue_swap"
    assert record["raw_diff_norm"] > 0.0
    assert record["positions"]["history_rows"] and record["positions"]["current_rows"]
    assert record["applied_to_history"] is True
    assert record["positions"]["measurement_ids"]
    assert swapped.measurement_ids == served_batch.measurement_ids
    for camera_id, frame in swapped.observation.frames.items():
        before = np.asarray(served_batch.observation.frames[camera_id].values)
        after = np.asarray(frame.values)
        for row in range(before.shape[0]):
            assert not np.array_equal(before[row], after[row]), (camera_id, row)
        # a pure exchange: the left half now carries the old right half
        assert np.allclose(after[:, :, :, : WIDTH // 2], before[:, :, :, WIDTH // 2 :])
        assert np.allclose(after[:, :, :, WIDTH // 2 :], before[:, :, :, : WIDTH // 2])
    half, half_record = cue_swap(served_batch, eps=0.5, positions=_boxes())
    assert half_record["raw_diff_norm"] == pytest.approx(0.5 * record["raw_diff_norm"])


def test_cue_swap_refuses_boxes_of_different_shape(served_batch):
    with pytest.raises(ValueError, match="equal shape"):
        cue_swap(served_batch, eps=1.0, positions={"a": (0, 4, 0, 4), "b": (0, 8, 0, 4)})


def test_no_information_replaces_the_measurement(served_batch):
    replaced, record = no_information(served_batch, seed=11)
    assert record["kind"] == "no_information"
    assert record["raw_diff_norm"] > 0.0
    assert record["per_camera_detail"]["robotwin_head"]["mode"] == "matched_noise"
    again, _ = no_information(served_batch, seed=11)
    other, _ = no_information(served_batch, seed=12)
    assert np.array_equal(
        np.asarray(replaced.observation.frames["robotwin_head"].values),
        np.asarray(again.observation.frames["robotwin_head"].values),
    )
    assert not np.array_equal(
        np.asarray(replaced.observation.frames["robotwin_head"].values),
        np.asarray(other.observation.frames["robotwin_head"].values),
    )
    flat, flat_record = no_information(served_batch, seed=3, mode="uniform_field", region="cues")
    assert flat_record["raw_diff_norm"] > 0.0
    values = np.asarray(flat.observation.frames["robotwin_head"].values)
    original = np.asarray(served_batch.observation.frames["robotwin_head"].values)
    rows, cols = slice(HEIGHT // 4, 3 * HEIGHT // 4), slice(WIDTH // 4, 3 * WIDTH // 4)
    box = values[:, :, rows, cols]
    assert np.allclose(box, box.mean(axis=(-2, -1), keepdims=True))  # a real flat field
    assert np.array_equal(values[:, :, : HEIGHT // 4, :], original[:, :, : HEIGHT // 4, :])


def test_exposure_and_noise_shifts_record_what_changed(served_batch):
    exposed, record = exposure_shift(served_batch, gain=1.75)
    detail = record["per_camera_detail"]["robotwin_head"]
    assert record["strength"] == 1.75
    assert detail["mean_after"] == pytest.approx(1.75 * detail["mean_before"])
    assert record["raw_diff_norm"] > 0.0
    assert detail["saturation_reevaluated"] is False
    unchanged, unchanged_record = exposure_shift(served_batch, gain=1.0)
    assert unchanged_record["no_effect"] is True and unchanged_record["raw_diff_norm"] == 0.0

    noisy, noise_record = noise_shift(served_batch, sigma_scale=3.0)
    noise_detail = noise_record["per_camera_detail"]["robotwin_head"]
    assert noise_record["raw_diff_norm"] > 0.0
    assert noise_detail["variance_is_estimate"] is True
    assert noise_detail["fitted_scale"] > 0.0
    assert noise_detail["sigma_mean"] > 0.0
    assert noise_record["positions"]["pixel_boxes"]["robotwin_head"] == [[0, HEIGHT, 0, WIDTH]]
    assert not np.array_equal(
        np.asarray(noisy.observation.frames["robotwin_head"].values),
        np.asarray(served_batch.observation.frames["robotwin_head"].values),
    )


def test_translation_shifts_content_without_wrap_and_marks_the_vacated_band(served_batch):
    shifted, record = translation_shift(served_batch, pixels=(3, -2))
    assert record["strength"] == [3, -2]
    assert record["raw_diff_norm"] > 0.0
    assert record["valid_cells_removed"] > 0
    before = np.asarray(served_batch.observation.frames["robotwin_head"].values)
    after = np.asarray(shifted.observation.frames["robotwin_head"].values)
    assert np.allclose(after[:, :, 3:, :-2], before[:, :, :-3, 2:])
    valid = np.asarray(shifted.observation.frames["robotwin_head"].valid_mask)
    assert valid.shape == before.shape[:1] + before.shape[-2:]
    assert not valid[:, :3, :].any()
    assert not valid[:, :, -2:].any()
    assert valid[:, 3:, :-2].all()
    assert record["per_camera_detail"]["robotwin_head"]["wrap_around"] is False
    with pytest.raises(ValueError, match="must move the content"):
        translation_shift(served_batch, pixels=(0, 0))


def test_interventions_do_not_touch_identity_and_stack_their_records(served_batch):
    first, first_record = exposure_shift(served_batch, gain=1.2)
    second, second_record = noise_shift(first, sigma_scale=1.0)
    assert second.measurement_ids == served_batch.measurement_ids
    assert np.array_equal(np.asarray(second.frame_index), np.asarray(served_batch.frame_index))
    assert second.partition == served_batch.partition
    records = second.provenance["interventions"]
    assert [item["kind"] for item in records] == ["exposure_shift", "noise_shift"]
    assert records[0]["raw_diff_norm"] == first_record["raw_diff_norm"]
    assert records[1]["raw_diff_norm"] == second_record["raw_diff_norm"]
    with pytest.raises(TypeError, match="V12WindowBatch"):
        cue_swap(object(), eps=1.0, positions=_boxes())
