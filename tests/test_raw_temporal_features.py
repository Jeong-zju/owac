"""CPU tests for held-out projected-token validation and feature selection."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from owac.evaluation.raw_temporal_features import (
    FeatureSelectionError,
    FeatureSelector,
    FeatureSourceSummary,
    FeatureValidationRecord,
    TokenFeatureAccumulator,
    evaluate_feature_validation,
    feature_total,
    masked_feature_components,
    select_feature_candidate,
    validation_endpoints,
)
from owac.training.raw_distillation import token_distillation_loss


def _record(
    *,
    split: str = "validation",
    value: float = 1.0,
    eligible: bool = True,
    valid_tokens: int = 8,
):
    summary = FeatureSourceSummary(
        source_id="s",
        windows=1,
        valid_tokens=valid_tokens,
        scalar_denominator=float(valid_tokens * 2),
        feature_mse=float(value),
        feature_cosine=0.0,
        feature_total=float(value),
    )
    return FeatureValidationRecord(
        split=split,
        sources=1,
        trajectories=1,
        windows=1,
        episode_resets=1,
        valid_tokens=valid_tokens,
        scalar_denominator=float(valid_tokens * 2),
        feature_mse=float(value),
        feature_cosine=0.0,
        feature_total=float(value),
        cosine_weight=0.1,
        per_source={"s": summary},
        source_order=("s",),
        test_used=False,
        selection_eligible=eligible,
    )


def test_masked_reduction_matches_training_loss_and_scalar_denominator() -> None:
    torch.manual_seed(7)
    student = torch.randn(1, 5, 8)
    teacher = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, False, True, True, False]])
    mse, cosine, total, valid, width = masked_feature_components(
        student, teacher, mask, cosine_weight=0.1
    )
    reference = token_distillation_loss(student, teacher, mask, cosine_weight=0.1)
    assert valid == 3 and width == 8
    assert math.isclose(float(mse), float(reference.mse), rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(cosine), float(reference.cosine), rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(total), float(reference.total), rel_tol=0.0, abs_tol=1e-6)
    # MSE divisor is valid_tokens * width, not the token count alone.
    manual = ((student - teacher).square() * mask.unsqueeze(-1)).sum() / (3 * 8)
    assert math.isclose(float(mse), float(manual), rel_tol=0.0, abs_tol=1e-6)


def test_global_reduction_weights_by_scalar_and_valid_tokens() -> None:
    accumulator = TokenFeatureAccumulator(cosine_weight=0.1)
    accumulator.begin_source("a")
    # window 1: 2 valid tokens, width 2 -> scalar 4
    teacher = torch.zeros(1, 3, 2)
    student = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [9.0, 9.0]]])
    mask1 = torch.tensor([[True, True, False]])
    w1 = accumulator.add_window("a", 0, student, teacher, mask1)
    # window 2: 1 valid token, width 2 -> scalar 2
    student2 = torch.tensor([[[0.0, 2.0], [9.0, 9.0], [9.0, 9.0]]])
    mask2 = torch.tensor([[True, False, False]])
    w2 = accumulator.add_window("a", 5, student2, teacher, mask2)
    accumulator.end_source()
    record = accumulator.finish()
    expected_scalar = 4.0 + 2.0
    expected_mse = (w1.feature_mse * 4.0 + w2.feature_mse * 2.0) / expected_scalar
    expected_valid = 3
    expected_cos = (w1.feature_cosine * 2.0 + w2.feature_cosine * 1.0) / expected_valid
    assert record.scalar_denominator == pytest.approx(expected_scalar)
    assert record.valid_tokens == expected_valid
    assert record.feature_mse == pytest.approx(expected_mse)
    assert record.feature_cosine == pytest.approx(expected_cos)
    assert record.feature_total == pytest.approx(expected_mse + 0.1 * expected_cos)
    assert record.feature_total == pytest.approx(
        feature_total(record.feature_mse, record.feature_cosine)
    )


def test_accumulator_requires_zero_start_and_strict_increase() -> None:
    accumulator = TokenFeatureAccumulator()
    accumulator.begin_source("a")
    values = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(FeatureSelectionError, match="endpoint 0"):
        accumulator.add_window("a", 5, values, values, mask)
    accumulator.add_window("a", 0, values, values, mask)
    with pytest.raises(FeatureSelectionError, match="strictly increase"):
        accumulator.add_window("a", 0, values, values, mask)
    accumulator.add_window("a", 5, values, values, mask)
    with pytest.raises(FeatureSelectionError, match="before finishing"):
        accumulator.begin_source("b")
    summary = accumulator.end_source()
    assert summary.windows == 2
    with pytest.raises(FeatureSelectionError, match="no open source"):
        accumulator.end_source()
    record = accumulator.finish()
    assert record.episode_resets == 1


def test_new_source_must_reset_at_zero_and_cannot_repeat() -> None:
    accumulator = TokenFeatureAccumulator()
    values = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    accumulator.begin_source("a")
    accumulator.add_window("a", 0, values, values, mask)
    accumulator.end_source()
    with pytest.raises(FeatureSelectionError, match="appears twice"):
        accumulator.begin_source("a")
    accumulator.begin_source("b")
    with pytest.raises(FeatureSelectionError, match="endpoint 0"):
        accumulator.add_window("b", 5, values, values, mask)
    accumulator.add_window("b", 0, values, values, mask)
    accumulator.end_source()
    record = accumulator.finish()
    assert record.sources == 2
    assert record.episode_resets == 2
    assert tuple(record.per_source) == ("a", "b")


def test_accumulator_rejects_bad_masks_and_nonfinite_tokens() -> None:
    accumulator = TokenFeatureAccumulator()
    accumulator.begin_source("a")
    values = torch.zeros(1, 2, 2)
    bad = values.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(FeatureSelectionError, match="non-finite"):
        accumulator.add_window("a", 0, bad, values, torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(FeatureSelectionError, match="at least one"):
        accumulator.add_window("a", 0, values, values, torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(FeatureSelectionError, match=r"\[B,N\]"):
        accumulator.add_window("a", 0, values, values, torch.ones(1, 1, dtype=torch.bool))


def test_expected_window_and_source_counts_are_enforced() -> None:
    values = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    accumulator = TokenFeatureAccumulator(expected_sources=2, expected_windows=2)
    accumulator.begin_source("a")
    accumulator.add_window("a", 0, values, values, mask)
    accumulator.end_source()
    with pytest.raises(FeatureSelectionError, match="expected 2 validation sources"):
        accumulator.finish()
    another = TokenFeatureAccumulator(expected_sources=1, expected_windows=2)
    another.begin_source("a")
    another.add_window("a", 0, values, values, mask)
    another.end_source()
    with pytest.raises(FeatureSelectionError, match="expected 2 validation windows"):
        another.finish()


def test_selector_minimum_and_earlier_tie() -> None:
    selector = FeatureSelector()
    assert selector.consider(100, _record(value=1.0), checkpoint_sha256="a" * 64)
    assert selector.best_step == 100
    # Equal total at an earlier step wins the declared tie-break.
    assert selector.consider(50, _record(value=1.0))
    assert selector.best_step == 50 and selector.best_value == 1.0
    assert selector.best_record is not None
    assert selector.best_record["step"] == 50
    assert selector.best_record["feature_total"] == 1.0
    assert not selector.consider(200, _record(value=1.2))
    assert selector.best_step == 50
    assert selector.consider(25, _record(value=0.5))
    assert selector.best_step == 25 and selector.best_value == 0.5
    # Replaying the same step with a different total is a hard error.
    with pytest.raises(FeatureSelectionError, match="different value"):
        selector.consider(25, _record(value=0.6))
    with pytest.raises(FeatureSelectionError, match="finite"):
        selector.consider(300, _record(value=float("nan")))


def test_selector_rejects_test_split_and_ineligible_records() -> None:
    selector = FeatureSelector()
    test_record = _record(split="test", eligible=False)
    assert test_record.split == "test" and not test_record.selection_eligible
    with pytest.raises(FeatureSelectionError):
        selector.consider(10, test_record)
    with pytest.raises(FeatureSelectionError):
        selector.consider(10, _record(eligible=False))
    with pytest.raises(FeatureSelectionError):
        selector.consider(10, {"split": "test", "test_used": False, "feature_total": 1.0})


def test_action_diagnostic_never_changes_selection_or_stop() -> None:
    selector = FeatureSelector()
    benchmark = _record(value=0.75)
    selector.consider(250, benchmark, checkpoint_sha256="b" * 64)
    before = (selector.best_step, selector.best_value)
    diagnostic = selector.record_action_diagnostic(
        step=250,
        metric="physical_first5_mae7",
        value=0.99,
        checkpoint_sha256="b" * 64,
        timestamp="2026-10-21T00:00:00+00:00",
        source="offline",
    )
    assert diagnostic["used_for_selection"] is False
    assert diagnostic["used_for_stop"] is False
    assert (selector.best_step, selector.best_value) == before
    # A vastly worse action value at a later step cannot move the token selector.
    selector.record_action_diagnostic(
        step=500,
        metric="physical_first5_mae7",
        value=100.0,
        checkpoint_sha256="c" * 64,
        timestamp="2026-10-21T00:01:00+00:00",
        source="offline",
    )
    assert (selector.best_step, selector.best_value) == before
    sidecar = selector.sidecar()
    assert sidecar["selection_metric"] == "validation_token_total"
    assert sidecar["test_selection"] is False
    assert all(item["used_for_selection"] is False for item in sidecar["action_diagnostics"])


def test_select_feature_candidate_tie_and_errors() -> None:
    assert select_feature_candidate([(100, 1.0), (50, 1.0), (75, 0.5)]) == (75, 0.5)
    assert select_feature_candidate([(100, 1.0), (50, 1.0)]) == (50, 1.0)
    assert select_feature_candidate([]) is None
    with pytest.raises(FeatureSelectionError):
        select_feature_candidate([(1, float("inf"))])


def test_validation_endpoints_fixed_stride() -> None:
    assert validation_endpoints(1) == (0,)
    assert validation_endpoints(95) == tuple(range(0, 95, 5))
    assert len(validation_endpoints(95)) == 19
    assert len(validation_endpoints(113)) == 23


class _FakeManifest:
    def __init__(self, sources):
        self._sources = list(sources)

    def sources_for_split(self, split):
        assert split in {"validation", "test"}
        return tuple(self._sources)


class _FakeData:
    def __init__(self, sources):
        self.manifest = _FakeManifest(sources)


class _FakePacket(SimpleNamespace):
    pass


class _FakeDataset:
    def __init__(self):
        self.calls = []

    def packet_for_endpoint(self, source_id, endpoint):
        self.calls.append((source_id, endpoint))
        return _FakePacket(
            source_id=source_id,
            endpoint=endpoint,
            reset=endpoint == 0,
        )


class _FakeCache:
    def __init__(self):
        self.calls = []

    def read(self, source_id, endpoint):
        self.calls.append((source_id, endpoint))
        teacher = torch.zeros(2, 3, 4)
        mask = torch.ones(2, dtype=torch.bool)
        return SimpleNamespace(teacher_tokens=teacher, teacher_token_mask=mask)


class _FakeStudent:
    def __init__(self, *, lose_state_after_first: bool = False):
        self.calls = []
        self.lose_state_after_first = lose_state_after_first
        self.training = True
        self.mode_calls = []

    def eval(self):
        self.training = False
        self.mode_calls.append("eval")
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        self.mode_calls.append(("train", bool(mode)))
        return self

    def __call__(self, packet, state):
        self.calls.append((packet.source_id, packet.endpoint, state is None))
        if packet.endpoint == 0:
            assert state is None
        else:
            assert state is not None
        values = torch.full((1, 6, 4), float(packet.endpoint + 1))
        output = SimpleNamespace(
            projected_tokens=values,
            tokens=SimpleNamespace(values=values, mask=torch.ones(1, 6, dtype=torch.bool)),
        )
        new_state = (
            None
            if (self.lose_state_after_first and packet.endpoint > 0)
            else SimpleNamespace(last_endpoint=packet.endpoint)
        )
        return output, new_state


def test_evaluate_feature_validation_resets_history_per_source() -> None:
    sources = (
        SimpleNamespace(source_demo_id="a", num_frames=21),  # 0,5,10,15,20 -> 5
        SimpleNamespace(source_demo_id="b", num_frames=11),  # 0,5,10 -> 3
    )
    student = _FakeStudent()
    dataset = _FakeDataset()
    cache = _FakeCache()
    record = evaluate_feature_validation(
        student,
        dataset,
        cache,
        _FakeData(sources),
        expected_sources=2,
        expected_windows=8,
    )
    assert record.sources == 2
    assert record.windows == 8
    assert record.episode_resets == 2
    assert record.per_source["a"].windows == 5
    assert record.per_source["b"].windows == 3
    # Every call records whether it inherited state; only the first window of each
    # source may be state-free.
    first_flags = [(sid, endpoint, reset) for sid, endpoint, reset in student.calls if reset]
    assert first_flags == [("a", 0, True), ("b", 0, True)]
    assert [sid for sid, _endpoint, _reset in student.calls] == ["a"] * 5 + ["b"] * 3
    assert record.valid_tokens == 8 * 6
    assert record.scalar_denominator == 8 * 6 * 4
    assert record.test_used is False and record.selection_eligible is True


def test_evaluate_feature_validation_fails_on_lost_state() -> None:
    sources = (SimpleNamespace(source_demo_id="a", num_frames=21),)
    student = _FakeStudent(lose_state_after_first=True)
    with pytest.raises(FeatureSelectionError, match="lost the causal student state"):
        evaluate_feature_validation(
            student,
            _FakeDataset(),
            _FakeCache(),
            _FakeData(sources),
            expected_sources=1,
            expected_windows=5,
        )


def test_evaluate_feature_validation_enforces_102_windows_fixture() -> None:
    # Actual parent validation source frame counts: 95,113,95,90,112 -> 19+23+19+18+23=102.
    frames = (95, 113, 95, 90, 112)
    sources = tuple(
        SimpleNamespace(source_demo_id=f"v{index}", num_frames=value)
        for index, value in enumerate(frames)
    )
    record = evaluate_feature_validation(
        _FakeStudent(),
        _FakeDataset(),
        _FakeCache(),
        _FakeData(sources),
        expected_sources=5,
        expected_windows=102,
    )
    assert record.windows == 102
    assert record.sources == 5
    assert record.episode_resets == 5
    assert sum(item.windows for item in record.per_source.values()) == 102
    with pytest.raises(FeatureSelectionError, match="expected 5 validation sources"):
        evaluate_feature_validation(
            _FakeStudent(),
            _FakeDataset(),
            _FakeCache(),
            _FakeData(sources[:4]),
            expected_sources=5,
            expected_windows=102,
        )


def test_validation_uses_eval_and_restores_original_training_mode() -> None:
    sources = (SimpleNamespace(source_demo_id="a", num_frames=6),)  # 0,5 -> 2 windows
    student = _FakeStudent()
    student.training = True
    record = evaluate_feature_validation(
        student,
        _FakeDataset(),
        _FakeCache(),
        _FakeData(sources),
        expected_sources=1,
        expected_windows=2,
    )
    assert record.windows == 2
    assert student.mode_calls[0] == "eval"
    assert student.mode_calls[-1] == ("train", True)
    assert student.training is True

    already_eval = _FakeStudent()
    already_eval.training = False
    evaluate_feature_validation(
        already_eval,
        _FakeDataset(),
        _FakeCache(),
        _FakeData((SimpleNamespace(source_demo_id="a", num_frames=6),)),
        expected_sources=1,
        expected_windows=2,
    )
    assert already_eval.mode_calls[0] == "eval"
    assert already_eval.mode_calls[-1] == ("train", False)
    assert already_eval.training is False
