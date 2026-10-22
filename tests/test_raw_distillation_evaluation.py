"""Pure-student inference, metrics, controls and stage-0 parity fixtures."""

from __future__ import annotations

import pytest
import torch

from owac.baselines.raw_stem_siglip import HistoryContract
from owac.core.encoder_contract_n02 import VisualTokens
from owac.evaluation.raw_distillation import (
    FrozenBackend,
    MetricRecord,
    OutcomeLedger,
    ParityTolerance,
    ParityTrace,
    PureStudentRuntime,
    Stage0ParityRunner,
    StudentInferenceUnavailable,
    evaluate_windows,
    select_validation,
    shuffle_visual_sources,
    target_position_intervention,
    wilson_interval,
    zero_visual,
)
from owac.raw.distillation_pairs import CameraFOV, PhysicalCrop, RawFrameData, StudentSample
from owac.raw.rawvla_sequence_n02 import NormalisationStats
from tests.test_raw_distillation_training import _student_and_bridge


class Tokenizer:
    def tokenize(self, text, state=None):
        return [2, 3, 0], [True, True, False]


def _stats():
    return NormalisationStats(
        mode="quantile",
        state={
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
        },
        action={
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
        },
    )


def _sample():
    raw = RawFrameData(
        values=torch.rand(3, 8, 8),
        units="normalized_0_1",
        cfa_pattern="",
        physical_origin=(0, 0),
    )
    return StudentSample(
        frame_key="source/base/0/rawvla_rgb3",
        source_id="source",
        acquisition_id="acq",
        demo_id="demo",
        frame_id=0,
        timestamp=0.0,
        split="validation",
        camera_id="camera_0",
        variant="rawvla_rgb3",
        raw=raw,
        state=torch.zeros(8),
        state_raw=torch.zeros(8),
        language="put the cream cheese in the bowl",
        fov=CameraFOV(8, 8, 1.0, 1.0, 4.0, 4.0),
        crop=PhysicalCrop(0, 0, 8, 8),
        flip="none",
        checkpoint_sha256="a" * 64,
        isp_id="isp",
        preprocessing_id="pre",
        normalization_id="norm",
        augmentation_id="none",
    )


def test_pure_runtime_infers_full_and_first5_without_teacher_or_cache():
    student, bridge = _student_and_bridge()
    bridge.base.tokenizer = Tokenizer()
    backend = FrozenBackend(
        bridge=bridge,
        tokenizer=bridge.base.tokenizer,
        stats=_stats(),
        identity={"sha256": "b" * 64},
    )
    runtime = PureStudentRuntime(student=student, backend=backend)
    from owac.backends.rawvla_language import prepare_language_state

    sample = _sample()
    expected_tokens, expected_masks, expected_state = prepare_language_state(
        bridge, [sample.language], sample.state.numpy().reshape(1, -1), device="cpu"
    )
    actual_tokens, actual_masks, actual_state = runtime._language(sample, None)
    assert torch.equal(actual_tokens, expected_tokens)
    assert torch.equal(actual_masks, expected_masks)
    assert torch.equal(actual_state, expected_state)
    result = runtime.infer(sample)
    assert result.normalized_actions.shape == (1, 10, 32)
    assert result.physical_actions.shape == (1, 10, 7)
    assert result.first5_physical.shape == (1, 5, 7)
    assert result.flow_steps == 10
    assert result.cache_key is None
    assert runtime.dependency_audit()["pure"]
    assert not hasattr(runtime, "teacher")
    assert not hasattr(runtime, "isp")
    assert not hasattr(runtime, "cache")
    with pytest.raises(StudentInferenceUnavailable, match="state"):
        bad = _sample()
        object.__setattr__(bad, "state", None)
        runtime.infer(bad)
    temporal = HistoryContract.from_frames((0, 1), (0.0, 0.1))
    with pytest.raises(StudentInferenceUnavailable, match="history"):
        PureStudentRuntime(student=student, backend=backend, history=temporal)


def test_metrics_complete_horizon_groups_padding_and_validation_selection():
    prediction = torch.zeros(2, 10, 32)
    target = torch.zeros(2, 10, 32)
    target[:, :, 0] = 1.0
    target[:, :, 6] = 2.0
    target[:, :, 7] = 3.0
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 5:] = False
    record = evaluate_windows(prediction, target, mask, split="validation", step=500, control_dim=7)
    assert record.valid_tokens == 15
    assert record.step1_mae7 == pytest.approx(3.0 / 7.0)
    assert record.first5_mae7 == pytest.approx(3.0 / 7.0)
    assert record.full10_mae7 == pytest.approx(3.0 / 7.0)
    assert record.groups["translation"] == pytest.approx(1.0 / 3.0)
    assert record.groups["gripper"] == pytest.approx(2.0)
    assert record.padding_mae == pytest.approx(3.0 / 25.0)
    first = MetricRecord(
        "validation", 2000, 10, 0.5, 0.5, 0.5, {}, valid_step1=10, valid_first5=10, valid_full10=10
    )
    tie_earlier = MetricRecord(
        "validation", 500, 10, 0.4, 0.5, 0.5, {}, valid_step1=10, valid_first5=10, valid_full10=10
    )
    assert select_validation([first, tie_earlier]).step == 500
    with pytest.raises(ValueError):
        select_validation(
            [
                MetricRecord(
                    "test",
                    500,
                    10,
                    0.1,
                    0.1,
                    0.1,
                    {},
                    valid_step1=10,
                    valid_first5=10,
                    valid_full10=10,
                )
            ]
        )
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low <= high <= 1.0
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_outcome_ledger_complete_denominator_and_controls():
    ledger = OutcomeLedger()
    ledger.add(previous_success=True, current_success=False, video="a.mp4")
    ledger.add(previous_success=False, current_success=True)
    ledger.add(previous_success=None, current_success=None)
    report = ledger.report()
    assert report["total"] == 2 and report["attempted"] == 2
    assert report["planned"] == 3 and report["complete_denominator"] == 3
    assert report["fully_observed"] is False
    assert report["regressions"] == 1 and report["recoveries"] == 1
    assert report["missing"] == 1 and report["videos"] == ["a.mp4"]
    tokens = VisualTokens(
        values=torch.ones(2, 4, 6),
        mask=torch.ones(2, 4, dtype=torch.bool),
        camera_ids=("camera_0",),
        positions=torch.zeros(2, 4, 3),
    )
    zero = zero_visual(tokens)
    assert torch.count_nonzero(zero.values) == 0
    assert torch.equal(zero.mask, tokens.mask)
    assert torch.equal(zero.positions, tokens.positions)
    shuffled = shuffle_visual_sources(tokens, (1, 0), source_ids=("a", "b"))
    assert torch.equal(shuffled.values, tokens.values.flip(0))
    with pytest.raises(ValueError, match="new action labels"):
        target_position_intervention(
            tokens, target_mask=torch.ones(2, 4, dtype=torch.bool), changed_actions=None
        )
    changed, actions = target_position_intervention(
        tokens,
        target_mask=torch.ones(2, 4, dtype=torch.bool),
        changed_actions=torch.ones(2, 10, 7),
        label_provenance={"source": "intervention_fixture", "sha256": "d" * 64},
    )
    assert changed.positions[0, 0, 0] == 1.0
    assert actions.shape == (2, 10, 7)


def test_stage0_parity_freezes_native_repeat_before_bridge_and_localizes_mismatch():
    history = HistoryContract()

    def native(batch):
        return ParityTrace(
            name="native",
            tensors={"visual": torch.ones(1, 4, 6), "actions": torch.zeros(1, 10, 32)},
            history=history,
            reset=False,
            provenance={"call": "real-native-policy", "batch": batch},
        )

    def bridge(batch):
        return ParityTrace(
            name="bridge",
            tensors={"visual": torch.ones(1, 4, 6), "actions": torch.full((1, 10, 32), 0.01)},
            history=history,
            reset=False,
            provenance={"call": "projected-teacher-token-bridge", "batch": batch},
        )

    report = Stage0ParityRunner(
        native_call=native, bridge_call=bridge, tolerance=ParityTolerance(atol=1e-4)
    ).run("fixed")
    assert report.passed is False
    assert report.first_mismatch == "actions"

    def incomplete(batch):
        return ParityTrace(
            name="bridge",
            tensors={"visual": torch.ones(1, 4, 6)},
            history=history,
            reset=False,
            provenance={"call": "incomplete-bridge"},
        )

    missing = Stage0ParityRunner(native_call=native, bridge_call=incomplete).run("fixed")
    assert missing.passed is False
    assert missing.first_mismatch == "actions:missing_bridge"
    with pytest.raises(ValueError, match="distinct"):
        Stage0ParityRunner(native_call=native, bridge_call=native)


def test_aggregate_metrics_weighted_all_windows_not_single_minimum():
    from owac.evaluation.raw_distillation import MetricRecord, aggregate_metrics

    a = MetricRecord(
        "validation", 500, 10, 0.1, 0.2, 0.3, {}, valid_step1=10, valid_first5=10, valid_full10=10
    )
    b = MetricRecord(
        "validation", 500, 30, 0.4, 0.5, 0.6, {}, valid_step1=30, valid_first5=30, valid_full10=30
    )
    result = aggregate_metrics([a, b], split="validation", step=500)
    # Weighted by actual denominators (10/40 and 30/40), not min single window.
    assert result.first5_mae7 == pytest.approx((0.2 * 10 + 0.5 * 30) / 40)
    assert result.valid_first5 == 40
    assert result.valid_full10 == 40
    with pytest.raises(ValueError):
        aggregate_metrics([a, b], split="test", step=500)
