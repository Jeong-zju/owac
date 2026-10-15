"""Behavioural properties of the P2 RAW motion training recipe.

Every assertion has an independent oracle or a property a plausible wrong
implementation would fail: the author loss is recomputed in NumPy from its
definition, corrupting an invalid cell cannot change the loss while excluding a
valid one must, the packed/sensor conversion is pinned against the generator's
declared sub-pixel token, the platform criteria are driven to failure one
condition at a time, and the resume test compares the next batch identity *and*
the next loss of a resumed run against a continuous one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.raw.n01_motion_data import VISIBILITY_KINDS, MotionDataGenerator
from owac.training.n01_motion import (
    RUN_SCALES,
    STEM_GROUPS,
    MotionDataConfig,
    MotionSampleSpec,
    MotionSampleStream,
    MotionScheduleConfig,
    MotionTrainer,
    MotionTrainingConfig,
    analytic_stem_delta,
    apply_selection,
    count_lr_drops,
    decide_platform,
    flow_target_from_supervision,
    load_motion_training_config,
    packed_to_sensor,
    run_fit_check,
    run_selection,
    run_stem_report,
    run_training,
    sensor_to_packed,
    sequence_loss,
)

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/encoders/n01_full/p2_motion_train.json"
RGB3_CONFIG_PATH = REPO_ROOT / "configs/encoders/n01_full/p2_motion_train_rgb3.json"
RGB3_FORMAL_CONFIG_PATH = REPO_ROOT / "configs/encoders/n01_full/p2_motion_train_rgb3_formal.json"
DATA_CONFIG = REPO_ROOT / "configs/encoders/n01_full/p2_motion_noise.json"
CASES = (
    "rotation",
    "layered_occlusion",
    "local_nonrigid",
    "repeated_texture",
    "exposure_gain",
    "motion_blur",
    "combined_hard",
)


def _region(prefix: str) -> dict:
    cases: dict = {"translation_subpixel": ["x+0.25", "x-0.25"]}
    for case in CASES:
        cases[case] = {"prefix": prefix, "count": 1}
    return {"cases": cases, "history_offsets": [1]}


def tiny_config(
    *,
    iters: int = 0,
    selection_iters: tuple[int, ...] = (0,),
    deep_geometry: bool = False,
    diagnostics: dict | None = None,
    schedule: dict | None = None,
) -> MotionTrainingConfig:
    """The shipped configuration reduced to a fast but real CPU run."""
    payload = json.loads(CONFIG_PATH.read_text())
    payload["model"].update({"depth": "resnet18", "iters": iters, "pretrained_checkpoint": None})
    payload["data"].update(
        {
            "config_path": str(DATA_CONFIG),
            "geometry": (
                {
                    "height": 256,
                    "width": 256,
                    "input_mode": "synthetic_cfa",
                    "cfa_pattern": "RGGB",
                    "origin_yx": [0, 0],
                    "channel_phase": [0, 1, 2, 3],
                    "value_units": "raw_dn_12bit",
                }
                if deep_geometry
                else None
            ),
            "recipe_scale": 4.0 if deep_geometry else 1.0,
            "batch_size": 1,
            "cache_entries": 8,
            "train": _region("tr"),
            "calibration": _region("ca"),
            "validation": _region("va"),
            "test": _region("te"),
        }
    )
    payload["selection"] = {"lrs": [0.00025, 5e-05], "iters": list(selection_iters), "updates": 1}
    payload["budget"] = {
        "updates": 8,
        "max_updates": 16,
        "validation_interval": 4,
        "plateau_checks": 8,
        "delta_val": 0.002,
        "monitor": "val_epe_sensor_px",
    }
    payload["schedule"] = schedule or {"drops": [[2, 0.1], [4, 0.01]], "min_drop_ratio": 0.1}
    payload["diagnostics"] = diagnostics or {
        "fit_updates": 1,
        "check_updates": 1,
        "fit_batch": 1,
        "fit_max_ratio": 0.999999,
        "fit_target": 1e9,
        "input_limit": 8.0,
    }
    payload["checkpoint"] = {"every": 2, "keep_last": 1}
    for key in ("max_calibration_pairs", "max_validation_pairs", "max_test_pairs"):
        payload["evaluation"][key] = 4
    payload["evaluation"]["latency_warmup"] = 1
    payload["evaluation"]["latency_repeats"] = 1
    return MotionTrainingConfig.from_dict(payload)


def tiny_rgb3_config() -> MotionTrainingConfig:
    """The shipped dev-scale RGB3 arm reduced to a fast but real CPU run."""
    payload = json.loads(RGB3_CONFIG_PATH.read_text())
    payload["model"].update({"depth": "resnet18", "pretrained_checkpoint": None})
    payload["data"].update(
        {
            "config_path": str(DATA_CONFIG),
            "batch_size": 1,
            "cache_entries": 4,
            "train": {
                "cases": {
                    "translation_subpixel": ["x+0.25"],
                    "rotation": {"prefix": "r", "count": 1},
                    "combined_hard": {"prefix": "c", "count": 1},
                },
                "history_offsets": [1],
            },
            "validation": {
                "cases": {"rotation": {"prefix": "v", "count": 1}},
                "history_offsets": [1],
            },
        }
    )
    payload["budget"] = {
        "updates": 8,
        "max_updates": 16,
        "validation_interval": 4,
        "plateau_checks": 8,
        "delta_val": 0.002,
        "monitor": "val_epe_sensor_px",
    }
    payload["schedule"] = {"drops": [[2, 0.1], [4, 0.01]], "min_drop_ratio": 0.1}
    payload["diagnostics"] = {
        "fit_updates": 1,
        "check_updates": 1,
        "fit_batch": 1,
        "fit_max_ratio": 0.999999,
        "fit_target": 1e9,
        "input_limit": 8.0,
    }
    payload["checkpoint"] = {"every": 1000, "keep_last": 1}
    return MotionTrainingConfig.from_dict(payload)


@pytest.fixture(scope="module")
def generator() -> MotionDataGenerator:
    return MotionDataGenerator.from_config_path(DATA_CONFIG)


# ---------------------------------------------------------------------------
# the author loss
# ---------------------------------------------------------------------------


def _synthetic_output(seed: int, predictions: int = 3, batch: int = 2, side: int = 5):
    rng = np.random.default_rng(seed)
    flow_gt = torch.tensor(rng.normal(size=(batch, 2, side, side)), dtype=torch.float32)
    valid = torch.ones(batch, side, side, dtype=torch.bool)
    valid[0, 0, 0] = False
    flow_gt[1, :, side - 1, side - 1] = 500.0
    nf = [
        torch.tensor(rng.normal(size=(batch, 2, side, side)), dtype=torch.float32)
        for _ in range(predictions)
    ]
    return {"flow": [None] * predictions, "nf": nf}, flow_gt, valid


def test_sequence_loss_matches_an_independent_numpy_definition():
    output, flow_gt, valid = _synthetic_output(seed=3)
    gamma = 0.8
    magnitude = np.sqrt((flow_gt.numpy() ** 2).sum(axis=1))
    legal = (valid.numpy() >= 0.5) & (magnitude < 400.0)
    expected = 0.0
    for index, nf in enumerate(output["nf"]):
        loss_i = nf.numpy()
        weight = gamma ** (len(output["nf"]) - index - 1)
        mask = (~np.isnan(loss_i)) & (~np.isinf(loss_i)) & legal[:, None]
        expected += weight * (mask * loss_i).sum() / mask.sum()
    got = sequence_loss(output, flow_gt, valid, gamma=gamma)
    assert float(got) == pytest.approx(expected, rel=1e-6)
    # the gamma weighting is real: a different gamma must change the value
    assert float(sequence_loss(output, flow_gt, valid, gamma=0.5)) != pytest.approx(
        float(got), rel=1e-6
    )


def test_sequence_loss_is_unchanged_by_an_invalid_cell_but_changes_when_a_valid_cell_leaves():
    output, flow_gt, valid = _synthetic_output(seed=7)
    base = float(sequence_loss(output, flow_gt, valid))

    corrupt_invalid = flow_gt.clone()
    corrupt_invalid[0, :, 0, 0] = 123.0  # this cell is invalid
    assert float(sequence_loss(output, corrupt_invalid, valid)) == base

    # pushing a valid cell beyond max_flow must remove it from the loss, exactly
    # like pre-masking that cell
    excluded = flow_gt.clone()
    excluded[1, :, 0, 1] = 500.0
    pre_masked = valid.clone()
    pre_masked[1, 0, 1] = False
    assert float(sequence_loss(output, excluded, valid)) == pytest.approx(
        float(sequence_loss(output, excluded, pre_masked)), rel=1e-6
    )
    assert float(sequence_loss(output, excluded, valid)) != pytest.approx(base, rel=1e-6)


def test_sequence_loss_refuses_a_test_mode_output_or_mismatched_predictions():
    output, flow_gt, valid = _synthetic_output(seed=11)
    with pytest.raises(ValueError, match="nf"):
        sequence_loss({"flow": [None], "nf": None}, flow_gt, valid)
    with pytest.raises(ValueError, match="nf predictions"):
        sequence_loss({"flow": [None, None], "nf": output["nf"]}, flow_gt, valid)
    with pytest.raises(ValueError, match="gamma"):
        sequence_loss(output, flow_gt, valid, gamma=0.0)
    with pytest.raises(ValueError, match=r"\[N, 2, H, W\]"):
        sequence_loss(output, flow_gt[0], valid)


# ---------------------------------------------------------------------------
# units and targets
# ---------------------------------------------------------------------------


def test_packed_sensor_conversion_is_exact_and_per_stem():
    for step in (1, 2):
        flow = torch.tensor([[0.25, -0.5], [1.5, 0.0]])
        assert torch.allclose(packed_to_sensor(flow, step), flow * step, atol=0.0)
        assert torch.allclose(sensor_to_packed(flow * step, step), flow, atol=0.0)
    assert packed_to_sensor(torch.tensor([1.0]), 2).item() == 2.0
    assert sensor_to_packed(torch.tensor([2.0]), 2).item() == 1.0
    # a wrong implementation that treats packed cells as sensor pixels is caught
    assert sensor_to_packed(torch.tensor([1.0]), 2).item() != 1.0
    with pytest.raises(ValueError, match="sensor_per_packed"):
        packed_to_sensor(torch.tensor([1.0]), 3)


def test_target_is_in_packed_cells_of_the_declared_subpixel_token(generator):
    geometry = generator.geometry()
    for token, expected_sensor in (("x+0.25", (0.25, 0.0)), ("y-0.5", (0.0, -0.5))):
        pair = generator.sample_pair(split="train", case="translation_subpixel", sequence=token)
        target = flow_target_from_supervision(pair.supervision, geometry, mode="cfa4")
        dx_sensor, dy_sensor = expected_sensor
        expected_packed = sensor_to_packed(
            torch.tensor([dy_sensor, dx_sensor]), target.sensor_per_packed
        )
        assert target.flow[:, 8, 8].tolist() == pytest.approx(expected_packed.tolist(), abs=1e-6)
        # the per-channel field of a rigid translation is the same in every phase,
        # and it *is* the sensor displacement halved (one packed step = two pixels)
        assert torch.allclose(
            target.channel_flow[:, :, 8, 8],
            target.flow[:, 8, 8].expand(4, 2),
            atol=1e-6,
        )
        assert packed_to_sensor(target.flow[:, 8, 8], target.sensor_per_packed).tolist() == (
            pytest.approx([dy_sensor, dx_sensor], abs=1e-6)
        )


def test_target_mask_excludes_occluded_and_out_of_bounds_cells(generator):
    geometry = generator.geometry()
    pair = generator.sample_pair(split="train", case="layered_occlusion", sequence="s0")
    supervision = pair.supervision
    target = flow_target_from_supervision(supervision, geometry, mode="cfa4")
    channel_valid = torch.stack(
        [supervision.valid[geometry.grid_indices_yx(channel)] for channel in range(4)]
    )
    assert torch.equal(target.channel_valid, channel_valid)
    assert torch.equal(target.valid, channel_valid.any(dim=0))
    # the corpus really does carry occluded cells here, and they are excluded
    assert supervision.visibility_counts()["occluded_in_source"] > 0
    occluded = supervision.visibility == VISIBILITY_KINDS.index("occluded_in_source")
    for channel in range(4):
        row_index, col_index = geometry.grid_indices_yx(channel)
        assert int((target.channel_valid[channel] & occluded[row_index, col_index]).sum()) == 0
    assert float(target.valid.float().mean()) < 1.0


# ---------------------------------------------------------------------------
# deterministic sampling and the LR schedule
# ---------------------------------------------------------------------------


def test_sample_stream_replays_the_same_order_from_its_state():
    specs = tuple(
        MotionSampleSpec(split="train", case="rotation", sequence=f"s{index}") for index in range(9)
    )
    first = MotionSampleStream(specs, seed=5)
    consumed = [first.next().identity() for _ in range(3)]
    state = first.state()
    tail = [first.next().identity() for _ in range(2)]

    second = MotionSampleStream(specs, seed=5)
    assert [second.next().identity() for _ in range(3)] == consumed
    second.load_state(state)
    assert [second.next().identity() for _ in range(2)] == tail
    assert first.state()["cursor"] == second.state()["cursor"]

    other = MotionSampleStream(specs, seed=6)
    assert [other.next().identity() for _ in range(5)] != consumed + tail

    with pytest.raises(ValueError, match="different number of specs"):
        MotionSampleStream(specs[:3], seed=5).load_state(state)


def test_lr_schedule_drops_at_absolute_updates_and_counts_two_effective_drops():
    schedule = MotionScheduleConfig(drops=((10, 0.1), (20, 0.01)))
    lrs = [schedule.lr_at(update, 1.0) for update in range(25)]
    assert lrs[0] == 1.0 and lrs[9] == 1.0 and lrs[10] == pytest.approx(0.1)
    # the two drops multiply: the final level is 0.1 * 0.01
    assert lrs[19] == pytest.approx(0.1) and lrs[20] == pytest.approx(0.001)
    assert count_lr_drops(lrs) == 2
    assert not schedule.is_final_phase(19) and schedule.is_final_phase(20)
    # the trajectory cannot depend on how long the run is
    assert [schedule.lr_at(update, 1.0) for update in range(25)] == lrs

    assert count_lr_drops([1.0] * 3 + [0.1] * 3) == 1
    assert count_lr_drops([1.0] * 6) == 0
    with pytest.raises(ValueError, match="two effective LR drops"):
        MotionScheduleConfig(drops=((10, 0.1),))
    with pytest.raises(ValueError, match="strictly increasing"):
        MotionScheduleConfig(drops=((30, 0.1), (10, 0.01)))
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        MotionScheduleConfig(drops=((10, 0.1), (20, 1.0)))


def test_config_refuses_unknown_keys_and_an_unreachable_final_drop():
    payload = json.loads(CONFIG_PATH.read_text())
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        MotionTrainingConfig.from_dict(payload)

    payload = json.loads(CONFIG_PATH.read_text())
    payload["schedule"] = {"drops": [[50_000, 0.1], [200_000, 0.01]], "min_drop_ratio": 0.1}
    with pytest.raises(ValueError, match="must fall inside the planned budget"):
        MotionTrainingConfig.from_dict(payload)

    payload = json.loads(CONFIG_PATH.read_text())
    payload["selection"]["lrs"] = [1e-4, 5e-5, 1e-5]
    with pytest.raises(ValueError, match="exactly two LR tiers"):
        MotionTrainingConfig.from_dict(payload)

    payload = json.loads(CONFIG_PATH.read_text())
    payload["budget"]["monitor"] = "val_accuracy"
    with pytest.raises(ValueError, match="budget.monitor"):
        MotionTrainingConfig.from_dict(payload)


# ---------------------------------------------------------------------------
# platform criteria
# ---------------------------------------------------------------------------


def _history(values, *, interval=1_000, start=1_000):
    return [
        {"update": start + index * interval, "val_epe_sensor_px": float(value)}
        for index, value in enumerate(values)
    ]


def _decide(history, **overrides):
    kwargs = dict(
        budget=100_000,
        interval=1_000,
        monitor="val_epe_sensor_px",
        delta_val=0.01,
        plateau_checks=8,
        lr_drops=2,
        final_phase=True,
        diagnostics_passed=True,
    )
    kwargs.update(overrides)
    return decide_platform(val_history=history, **kwargs)


def test_platform_plateau_requires_every_contract_condition():
    # strongly improving for ten validations, then flat within delta_val
    values = [10.0 - index for index in range(10)] + [1.0] * 10
    good = _decide(_history(values))
    assert good.status == "validation_plateau"
    assert good.details["conditions"] == {
        "real_improvement_over_run": True,
        "tail_within_delta_val_of_best": True,
        "two_effective_lr_drops": True,
        "final_low_lr_phase": True,
        "flat_tail_below_delta_val": True,
        "best_not_at_budget_end": True,
        "diagnostics_passed": True,
    }
    # every condition carries its own description, so an unmet reason is readable
    for name in good.details["conditions"]:
        assert name in good.details["condition_descriptions"]

    for override, expected in (
        ({"lr_drops": 1}, "two_effective_lr_drops"),
        ({"final_phase": False}, "final_low_lr_phase"),
        ({"diagnostics_passed": False}, "diagnostics_passed"),
    ):
        decision = _decide(_history(values), **override)
        assert decision.status == "budget_exhausted"
        assert any(expected in reason for reason in decision.reasons)

    # improvements of a full unit each are far above delta_val: not a plateau
    creeping = _decide(_history([10.0 - 1.0 * index for index in range(20)]))
    assert creeping.status == "budget_exhausted"
    assert any("flat_tail_below_delta_val" in reason for reason in creeping.reasons)

    # tiny improvements keep the best at the last window: with the budget ending
    # right behind it there is no plateau at a low LR, only budget exhaustion
    tail_best = _decide(_history([10.0 - 0.001 * index for index in range(20)]), budget=25_000)
    assert tail_best.status == "budget_exhausted"
    assert any("best_not_at_budget_end" in reason for reason in tail_best.reasons)

    # a single validation cannot claim anything
    short = _decide(_history([1.0]))
    assert short.status == "budget_exhausted"
    assert "fewer than two validation points" in short.reasons[0]


def test_platform_delta_val_boundary_is_strict():
    """D1: constant/degrading curves are never plateaus; slow improvement still is."""
    # a constant curve has an improvement of exactly zero, which is not an improvement
    strictly_flat = _decide(_history([5.0] * 12))
    assert strictly_flat.status == "budget_exhausted"
    assert strictly_flat.details["conditions"]["real_improvement_over_run"] is False
    assert strictly_flat.details["run_improvement"] == 0.0
    assert any("real_improvement_over_run" in reason for reason in strictly_flat.reasons)

    # a curve that only gets worse has no improvement either
    worsening = _decide(_history([1.0 + index for index in range(12)]))
    assert worsening.status == "budget_exhausted"
    assert worsening.details["conditions"]["real_improvement_over_run"] is False
    assert any("real_improvement_over_run" in reason for reason in worsening.reasons)

    head = [10.0 - 1.0 * index for index in range(10)]
    # improved once, then drifted away from its best: the plateau claim names the drift
    drifting = _decide(
        _history(head + [1.0 + 0.25 * step for step in range(1, 11)]), delta_val=0.25
    )
    assert drifting.details["conditions"]["real_improvement_over_run"] is True
    assert drifting.details["conditions"]["tail_within_delta_val_of_best"] is False
    assert drifting.details["tail_drift"] > 0.25
    assert drifting.reasons == ("platform condition failed: tail_within_delta_val_of_best",)

    # slow but persistent improvement below delta_val: still a plateau (literal contract)
    slow = [10.0 - 0.125 * index for index in range(12)]  # 0.5 * delta_val per validation
    slow_decision = _decide(_history(slow), delta_val=0.25)
    assert slow_decision.details["conditions"]["flat_tail_below_delta_val"] is True
    assert slow_decision.details["conditions"]["real_improvement_over_run"] is True
    assert slow_decision.details["conditions"]["tail_within_delta_val_of_best"] is True
    assert slow_decision.status == "validation_plateau"

    # tail improvements of exactly delta_val do not count (the test is strictly below)
    at_threshold = _decide(
        _history(head + [1.0 - 0.25 * step for step in range(1, 11)]), delta_val=0.25
    )
    assert at_threshold.details["conditions"]["flat_tail_below_delta_val"] is False
    assert any("flat_tail_below_delta_val" in reason for reason in at_threshold.reasons)
    # below the threshold they do count, and then the best sitting in the tail is what
    # stops the plateau claim when the budget ends right behind it
    below = _decide(
        _history(head + [1.0 - 0.125 * step for step in range(1, 11)]),
        delta_val=0.25,
        budget=25_000,
    )
    assert below.details["conditions"]["flat_tail_below_delta_val"] is True
    assert below.details["conditions"]["best_not_at_budget_end"] is False
    assert below.status == "budget_exhausted"


# ---------------------------------------------------------------------------
# training mechanics
# ---------------------------------------------------------------------------


def test_training_step_updates_parameters_and_records_real_gradients():
    config = tiny_config()
    trainer = MotionTrainer(config, device="cpu")
    for update in range(2):
        record = trainer.advance(update=update)
        assert np.isfinite(record["loss"])
    evidence = trainer.parameter_update_evidence()
    assert evidence["total_abs_update"] > 0.0
    for group in ("cnet", "init_conv", "upsample_weight", "flow_head"):
        assert evidence["per_group"][group] > 0.0
    assert all(value > 0.0 for value in trainer._gradient_groups().values() if value)
    assert trainer.batch_log and all(
        entry["split"] == "train" for batch in trainer.batch_log for entry in batch
    )
    assert len(trainer.lr_history) == 2


def test_diagnostics_restore_the_model_and_the_optimizer():
    config = tiny_config()
    trainer = MotionTrainer(config, device="cpu")
    before = {name: tensor.clone() for name, tensor in trainer.model.state_dict().items()}
    optimizer_before = json.dumps(
        {key: value for key, value in trainer.optimizer.state_dict().items() if key != "state"},
        default=str,
    )
    diagnostics = trainer.diagnose_fit(updates=1)
    after = trainer.model.state_dict()
    assert set(before) == set(after)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name
    assert (
        json.dumps(
            {key: value for key, value in trainer.optimizer.state_dict().items() if key != "state"},
            default=str,
        )
        == optimizer_before
    )
    assert diagnostics["checks"]["unit_round_trip_exact"] is True
    assert diagnostics["checks"]["inputs_finite"] is True
    assert diagnostics["checks"]["gradient_groups_nonzero"] is True
    assert diagnostics["unit_probe"]["sensor_per_packed"] == 2
    assert (
        diagnostics["unit_probe"]["round_trip_sensor_px"] == diagnostics["unit_probe"]["sensor_px"]
    )


def test_checkpoint_roundtrip_replays_the_next_batch_and_the_next_update(tmp_path):
    config = tiny_config()
    checkpoint = tmp_path / "ckpt.pt"

    continuous = MotionTrainer(config, device="cpu")
    continuous.diagnostics = {"passed": True}
    for update in range(2):
        continuous.advance(update=update)
    payload = continuous.checkpoint_payload(update=2)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "sampler", "train_samples"):
        assert key in payload
    for key in ("python", "numpy", "torch"):
        assert key in payload["rng"]
    continuous.save_checkpoint(checkpoint, update=2)
    for update in range(2, 4):
        continuous.advance(update=update)

    resumed = MotionTrainer(config, device="cpu")
    resumed.diagnostics = {"passed": True}
    start = resumed.load_checkpoint(checkpoint)
    assert start == 2
    assert (
        resumed.stream.receipt()["permutation_sha256"]
        == continuous.stream.receipt()["permutation_sha256"]
    )
    assert resumed.stream.receipt()["cursor"] == 2
    for update in range(2, 4):
        resumed.advance(update=update)

    assert resumed.batch_log == continuous.batch_log[2:]
    assert resumed.train_losses == continuous.train_losses[2:]
    continuous_state = continuous.model.state_dict()
    resumed_state = resumed.model.state_dict()
    assert set(continuous_state) == set(resumed_state)
    for name, tensor in continuous_state.items():
        assert torch.equal(tensor, resumed_state[name]), name


def test_resume_refuses_a_different_run_definition(tmp_path):
    config = tiny_config()
    trainer = MotionTrainer(config, device="cpu")
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path, update=1)
    other = MotionTrainer(tiny_config(schedule={"drops": [[3, 0.1], [5, 0.01]]}), device="cpu")
    with pytest.raises(ValueError, match="different run definition"):
        other.load_checkpoint(path)


def test_run_training_uses_only_the_train_region_and_reports_the_platform(tmp_path):
    config = tiny_config()
    receipt = run_training(config, output=tmp_path / "run", device="cpu", max_updates=2, seed=101)
    assert receipt["run_mode"] == "train"
    assert receipt["platform"]["status"] in ("validation_plateau", "budget_exhausted")
    assert receipt["optimizer_updates"] == 2
    assert receipt["samples_consumed"] == 3  # 2 diagnostic updates + 2 training updates
    assert receipt["diagnostic_samples"] == 1
    assert all(
        entry["split"] == "train" for batch in receipt["batch_identity_head"] for entry in batch
    )
    assert receipt["loss_curve_summary"]["updates"] == 2
    assert receipt["input_hashes"]["training/n01_motion.py"]
    assert (tmp_path / "run" / "receipt.json").is_file()
    assert (tmp_path / "run" / "checkpoints" / "last.pt").is_file()
    # the receipt never records a test-region identity
    assert "test" not in json.dumps(receipt["batch_identity_head"])


def test_fit_check_receipt_is_machine_readable_and_fails_on_an_impossible_target(tmp_path):
    config = tiny_config()
    receipt = run_fit_check(config, output=tmp_path / "fit", device="cpu", max_updates=2)
    assert receipt["passed"] == all(receipt["diagnostics"]["checks"].values())
    assert receipt["samples_consumed"] == 2
    assert receipt["parameter_hash_after"]
    assert (tmp_path / "fit" / "fit_check.json").is_file()

    strict = tiny_config(
        diagnostics={
            "fit_updates": 1,
            "check_updates": 1,
            "fit_batch": 1,
            "fit_max_ratio": 0.999999,
            "fit_target": -1.0,
            "input_limit": 8.0,
        }
    )
    failed = run_fit_check(strict, output=tmp_path / "fit_fail", device="cpu", max_updates=1)
    assert failed["passed"] is False
    assert failed["diagnostics"]["checks"]["small_set_fit_target"] is False


def test_selection_uses_validation_only_and_records_the_winner(tmp_path):
    config = tiny_config(iters=1, selection_iters=(0, 1), deep_geometry=True)
    receipt = run_selection(config, output=tmp_path / "select", device="cpu", max_updates=1)
    assert receipt["selection_split"] == "validation"
    assert receipt["fitting_split"] == "train"
    assert receipt["test_split_used"] is False
    assert len(receipt["candidates"]) == 4  # two LR tiers x two iteration tiers
    assert all(item["updates"] == 1 for item in receipt["candidates"])
    assert receipt["winner"]["lr"] in list(config.selection.lrs)
    assert receipt["winner"]["iters"] in list(config.selection.iters)
    assert (tmp_path / "select" / "winner.json").is_file()


def test_apply_selection_guard_refuses_test_derived_or_foreign_selections(tmp_path):
    config = tiny_config()
    good = {
        "winner": {"lr": 0.00025, "iters": 4},
        "selection_split": "validation",
        "test_split_used": False,
        "config_fingerprint": config.core_fingerprint,
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(good))
    assert apply_selection(config, path) == {"lr": 0.00025, "iters": 4}

    for broken in (
        {**good, "selection_split": "test"},
        {**good, "test_split_used": True},
        {**good, "config_fingerprint": "0" * 64},
        {"winner": {"lr": 0.1, "iters": 4}, "selection_split": "validation"},
    ):
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            apply_selection(config, path)


def test_rgb3_config_switches_only_the_stem_domain():
    cfa = load_motion_training_config(CONFIG_PATH)
    rgb3 = load_motion_training_config(RGB3_CONFIG_PATH)
    assert rgb3.scale == "dev" and rgb3.scale in RUN_SCALES
    assert cfa.scale == "formal"
    assert rgb3.model.mode == "rgb3" and cfa.model.mode == "cfa4"
    assert rgb3.data.input_mode == "rawvla_rgb3"
    # same budget tier, criteria, schedule, seeds and optimizer as the CFA arm
    assert rgb3.seeds == cfa.seeds
    assert rgb3.budget == cfa.budget
    assert rgb3.schedule == cfa.schedule
    assert rgb3.optimizer == cfa.optimizer
    assert rgb3.checkpoint == cfa.checkpoint
    assert rgb3.diagnostics == cfa.diagnostics
    assert rgb3.selection == cfa.selection
    assert rgb3.evaluation.buckets == cfa.evaluation.buckets
    assert rgb3.data.batch_size == cfa.data.batch_size
    assert rgb3.data.train.cases == cfa.data.train.cases

    data_config = rgb3.data.resolved_data_config()
    geometry = data_config.geometry_for(rgb3.data.input_mode)
    assert geometry.is_cfa is False
    assert geometry.cfa_pattern == "rgb3"
    assert geometry.channel_phase == ()
    assert geometry.packed_shape == (128, 128)  # rgba3 keeps full sensor resolution
    assert geometry.sensor_shape == (128, 128)
    assert geometry.input_mode == "rawvla_rgb3"

    trainer = MotionTrainer(tiny_rgb3_config(), device="cpu")
    assert trainer.mode == "rgb3"
    assert trainer.sensor_per_packed == 1
    assert trainer.geometry.channel_phase == ()


def test_rgb3_training_step_has_real_gradients_and_parameter_updates():
    trainer = MotionTrainer(tiny_rgb3_config(), device="cpu")
    for update in range(2):
        record = trainer.advance(update=update)
        assert np.isfinite(record["loss"])
    evidence = trainer.parameter_update_evidence()
    assert evidence["total_abs_update"] > 0.0
    for group in STEM_GROUPS:
        assert evidence["per_group"][group] > 0.0, group
    groups = trainer._gradient_groups()
    assert all(value > 0.0 for value in groups.values())
    assert trainer.provider.receipt()["packed_shape"] == [128, 128]
    assert trainer.provider.receipt()["sensor_per_packed"] == 1
    assert all(entry["split"] == "train" for batch in trainer.batch_log for entry in batch)


def test_stem_report_matches_the_analytic_stem_difference(tmp_path):
    cfa = load_motion_training_config(CONFIG_PATH)
    rgb3 = load_motion_training_config(RGB3_CONFIG_PATH)
    receipt = run_stem_report(
        cfa,
        rgb3,
        output=tmp_path / "stem",
        device="cpu",
        config_path=CONFIG_PATH,
        compare_config_path=RGB3_CONFIG_PATH,
    )
    # analytic: the two stems are Conv2d(C, 64, 7, stride=2, padding=3) and cnet sees 2C
    analytic = analytic_stem_delta(4, 3)
    assert analytic["fnet"] == -(64 * 7 * 7)
    assert analytic["cnet"] == -(2 * 64 * 7 * 7)
    assert sum(analytic.values()) == -9408
    for group in STEM_GROUPS:
        measured = receipt["groups"][group]
        assert measured["delta_params"] == analytic[group], group
        assert measured["analytic_delta_params"] == analytic[group], group
        assert measured["matches_analytic"] is True
    assert receipt["total_trainable"]["cfa4"] - receipt["total_trainable"]["rgb3"] == 9408
    assert receipt["total_trainable"]["delta"] == -9408
    assert receipt["channels"] == {"cfa4": 4, "rgb3": 3}
    assert receipt["stem_conv"]["cnet_input"] == [8, 6]
    assert receipt["differing_parameter_names"] == ["cnet.conv1.weight", "fnet.conv1.weight"]
    assert receipt["groups"]["fnet"]["differing_tensors"]["fnet.conv1.weight"] == {
        "cfa4": [64, 4, 7, 7],
        "rgb3": [64, 3, 7, 7],
    }
    assert receipt["only_stem_input_channels_differ"] is True
    assert receipt["verdict"]["analytic_matches_measured"] is True
    assert receipt["config_difference"]["matches_expected"] is True
    # only the stem domain, the label and the resolution difference are allowed
    roots = {path.split(".")[0] for path in receipt["config_difference"]["observed_paths"]}
    assert roots == {"scale", "stage", "notes", "model", "data"}
    assert "budget" not in roots and "schedule" not in roots
    assert (tmp_path / "stem" / "stem_report.json").is_file()


def _config_differences(left: dict, right: dict, prefix: str = "") -> set[str]:
    """Dotted paths whose values differ; a dict-vs-scalar difference stops at the parent."""
    diffs: set[str] = set()
    for key in set(left) | set(right):
        left_value, right_value = left.get(key), right.get(key)
        path = f"{prefix}{key}"
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            diffs |= _config_differences(left_value, right_value, prefix=f"{path}.")
        elif left_value != right_value:
            diffs.add(path)
    return diffs


def test_rgb3_formal_config_matches_cfa_except_the_input_domain():
    cfa = load_motion_training_config(CONFIG_PATH)
    formal = load_motion_training_config(RGB3_FORMAL_CONFIG_PATH)
    dev = load_motion_training_config(RGB3_CONFIG_PATH)

    # the formal arm keeps the formal label and does not inherit the dev marker
    assert cfa.scale == "formal" and formal.scale == "formal"
    assert dev.scale == "dev"
    assert formal.stage != dev.stage and dev.stage == "P2-dev-rgb3"
    assert _config_differences(dev.as_dict(), formal.as_dict()) == {"scale", "stage", "notes"}

    # exactly the declared input-domain and label differences against the CFA arm
    assert _config_differences(cfa.as_dict(), formal.as_dict()) == {
        "stage",
        "notes",
        "model.mode",
        "data.input_mode",
        "data.geometry",
        "data.recipe_scale",
    }
    assert "scale" not in _config_differences(cfa.as_dict(), formal.as_dict())

    # same formal budget, schedule, seeds, criteria and evaluation as the CFA arm
    assert formal.seeds == cfa.seeds == (101, 211, 307)
    assert formal.budget == cfa.budget
    assert formal.budget.updates == 100_000 and formal.budget.max_updates == 300_000
    assert formal.budget.validation_interval == 1_000
    assert formal.budget.delta_val == 0.002
    assert formal.budget.monitor == cfa.budget.monitor
    assert formal.schedule.drops == cfa.schedule.drops == ((50_000, 0.1), (80_000, 0.01))
    assert formal.optimizer == cfa.optimizer
    assert formal.checkpoint == cfa.checkpoint
    assert formal.diagnostics == cfa.diagnostics
    assert formal.selection == cfa.selection
    assert formal.evaluation.buckets == cfa.evaluation.buckets
    assert formal.data.batch_size == cfa.data.batch_size
    assert formal.data.cache_entries == cfa.data.cache_entries
    assert formal.data.exposure_correction == cfa.data.exposure_correction
    assert formal.data.train.cases == cfa.data.train.cases
    assert formal.data.validation.cases == cfa.data.validation.cases
    assert formal.data.test.cases == cfa.data.test.cases

    # the input domain is the pseudo-RAW three-channel one
    assert formal.model.mode == "rgb3"
    data_config = formal.data.resolved_data_config()
    geometry = data_config.geometry_for(formal.data.input_mode)
    assert formal.data.input_mode == "rawvla_rgb3"
    assert geometry.input_mode == "rawvla_rgb3"
    assert geometry.cfa_pattern == "rgb3" and geometry.channel_phase == ()
    assert geometry.packed_shape == (128, 128) and geometry.sensor_shape == (128, 128)
    assert formal.data.recipe_scale == 2.0 and cfa.data.recipe_scale == 4.0


def test_rgb3_formal_stem_report_matches_the_analytic_difference(tmp_path):
    cfa = load_motion_training_config(CONFIG_PATH)
    formal = load_motion_training_config(RGB3_FORMAL_CONFIG_PATH)
    receipt = run_stem_report(
        cfa,
        formal,
        output=tmp_path / "stem-formal",
        device="cpu",
        config_path=CONFIG_PATH,
        compare_config_path=RGB3_FORMAL_CONFIG_PATH,
    )
    assert receipt["total_trainable"]["delta"] == -9408
    assert receipt["total_trainable"]["matches_analytic"] is True
    assert receipt["scale"] == {"cfa4": "formal", "rgb3": "formal"}
    assert receipt["differing_parameter_names"] == ["cnet.conv1.weight", "fnet.conv1.weight"]
    assert receipt["only_stem_input_channels_differ"] is True
    assert receipt["verdict"]["analytic_matches_measured"] is True
    assert receipt["config_difference"]["matches_expected"] is True
    observed = set(receipt["config_difference"]["observed_paths"])
    assert "scale" not in observed  # both arms are formal: no label difference at all
    assert {"model.mode", "data.input_mode"} <= observed
    assert receipt["geometries"]["rgb3"]["channel_phase"] == [] and receipt["geometries"]["rgb3"][
        "packed_shape"
    ] == [128, 128]
    assert (tmp_path / "stem-formal" / "stem_report.json").is_file()


def test_stem_report_refuses_a_hidden_depth_difference(tmp_path):
    cfa = load_motion_training_config(CONFIG_PATH)
    rgb3 = tiny_rgb3_config()
    assert rgb3.model.depth != cfa.model.depth
    with pytest.raises(ValueError, match="hidden difference"):
        run_stem_report(cfa, rgb3, output=tmp_path / "stem", device="cpu")


def test_data_region_specs_declare_their_own_split():
    config = tiny_config()
    assert all(spec.split == "train" for spec in config.data.train.specs("train"))
    assert all(spec.split == "validation" for spec in config.data.validation.specs("validation"))
    assert all(spec.split == "test" for spec in config.data.test.specs("test"))
    assert isinstance(config.data, MotionDataConfig)
    assert load_motion_training_config(CONFIG_PATH).budget.updates == 100_000
