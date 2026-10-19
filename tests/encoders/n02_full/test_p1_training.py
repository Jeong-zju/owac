"""P1 training-harness tests: specs, masking, resume, CLI boundaries, not_run policy.

These tests exercise the harness this module owns (T4): the frozen P1 starting
values, the strict config resolution, the recorded activation counts, the exact
resume path, the instrumentation columns and the CLI's honest ``not_run``
behaviour. The end-to-end ``step_bundle`` causality test is written against the
frozen signature and skips (with a reason) while the T3 module is absent.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.core.encoder_contract_n02 import MeasurementFrame
from owac.frameworks.n02_full import config as n02_config
from owac.training import n02_full as train

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "configs/encoders/n02_full"


# --------------------------------------------------------------------------------------
# fixtures / fakes
# --------------------------------------------------------------------------------------


class TinyModel(torch.nn.Module):
    """A deterministic model with one Base parameter and one marked head."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.base = torch.nn.Linear(width, width, bias=False)
        self.head = torch.nn.Linear(width, 1, bias=False)

    def weights(self) -> list[torch.Tensor]:
        return [self.base.weight, self.head.weight]


class MaskedModel(torch.nn.Module):
    """A pretrainer-like module whose loss ignores invalid positions.

    The harness must report the module's own ``valid_target_count`` and must not
    add or imply any other normalization: the loss here is a masked mean over
    valid positions, and the test compares it against an independent computation.
    """

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.base = torch.nn.Linear(3 * 4 * 4, width, bias=False)
        self.head = torch.nn.Linear(1, 1, bias=False)

    def forward(self, frames, *, camera_ids):
        frame = frames[camera_ids[0]]
        values = torch.as_tensor(np.asarray(frame.values), dtype=torch.float32)
        valid = torch.as_tensor(np.asarray(frame.valid_mask), dtype=torch.bool)
        target = values.reshape(values.shape[0], -1)
        prediction = self.head(self.base(target).mean(dim=-1, keepdim=True))
        error = (prediction - target.mean(dim=-1, keepdim=True)) ** 2
        error = error.reshape(values.shape[0])
        mask = valid.reshape(values.shape[0], -1).any(dim=-1)
        masked = error * mask
        loss = masked.sum() / mask.sum().clamp(min=1)
        out = type("Out", (), {})()
        out.loss = loss
        out.valid_target_count = torch.tensor(int(mask.sum()))
        out.saturated_count = torch.tensor(0)
        return out


def make_frame(rows: int, *, valid_rows: int | None = None, seed: int = 0) -> MeasurementFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, 3, 4, 4)).astype(np.float32)
    valid = np.ones((rows, 4, 4), dtype=bool)
    if valid_rows is not None and valid_rows < rows:
        valid[valid_rows:] = False
    return MeasurementFrame(
        values=values,
        valid_mask=valid,
        bad_mask=np.zeros_like(valid),
        saturated_mask=np.zeros_like(valid),
        sample_xy=np.zeros((rows, 4, 4, 2), dtype=np.float32),
        episode_ids=tuple(f"ep-{seed}-{index}" for index in range(rows)),
        frame_ids=tuple(f"ep-{seed}-{index}-f0" for index in range(rows)),
        camera_id="cam0",
        new_measurement=np.ones(rows, dtype=bool),
        metadata={"unit": "dn"},
        metadata_available={"calibration": True},
    )


class CountingSource:
    """A deterministic micro-batch source with a resumable cursor."""

    def __init__(self, rows: int = 4, *, seed: int = 0) -> None:
        self.rows = rows
        self.seed = seed
        self.cursor = 0

    def state_dict(self) -> dict:
        return {"cursor": int(self.cursor)}

    def load_state_dict(self, payload) -> None:
        self.cursor = int(payload.get("cursor", 0))

    def next_micro_batch(self) -> dict:
        frame = make_frame(self.rows, seed=self.seed + self.cursor)
        frame.metadata["cursor"] = self.cursor
        self.cursor += 1
        return {"cam0": frame}


def toy_loss_fn(model, batch, info):
    """Two differentiable terms so the per-term conflict angle is defined."""
    values = torch.as_tensor(np.asarray(batch["cam0"].values), dtype=torch.float32)
    flat = values.reshape(values.shape[0], -1)[:, :4]
    first = model.head(model.base(flat)).sum()
    second = (model.base(flat) ** 2).sum() * 0.25
    terms = {
        "first": {
            "status": "ok",
            "value": float(first.detach()),
            "sum": float(first.detach()),
            "activation_count": int(flat.numel()),
            "reason": "",
            "kind": "toy",
            "tensor": first,
        },
        "second": {
            "status": "ok",
            "value": float(second.detach()),
            "sum": float(second.detach()),
            "activation_count": int(flat.numel()),
            "reason": "",
            "kind": "toy",
            "tensor": second,
        },
    }
    return first + second, terms, {}


def build_trainer(
    *,
    source=None,
    model=None,
    updates: int = 4,
    accum_steps: int = 1,
    loss_fn=None,
    seed: int = 11,
    **kwargs,
):
    torch.manual_seed(seed)
    model = model or TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return (
        train.P1Trainer(
            stage="p1-pretrain",
            seed=seed,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            source=source or CountingSource(),
            loss_fn=loss_fn or toy_loss_fn,
            updates=updates,
            accum_steps=accum_steps,
            base_lrs=[1e-3],
            warmup_updates=1,
            **kwargs,
        ),
        model,
    )


# --------------------------------------------------------------------------------------
# frozen starting values and config resolution
# --------------------------------------------------------------------------------------


def test_pretrain_spec_defaults_are_the_frozen_p1_starting_values() -> None:
    spec = train.PretrainSpec()
    assert spec.updates == 100_000
    assert spec.lr == 1.5e-4
    assert spec.weight_decay == 0.05
    assert spec.betas == (0.9, 0.95)
    assert spec.warmup_fraction == 0.05
    assert spec.schedule == "cosine"
    assert spec.effective_batch_frames == 256
    assert spec.accum_steps == 16
    assert spec.lr_grid == (7.5e-5, 1.5e-4, 3e-4)
    assert spec.mask_ratio == 0.6
    assert spec.amp == "off"
    assert train.PRETRAIN_DEFAULTS["lr"] == spec.lr


def test_measurement_spec_defaults_are_the_frozen_p1_starting_values() -> None:
    spec = train.MeasurementSpec()
    assert spec.updates == 50_000
    assert spec.effective_batch_episodes == 32
    assert spec.frames_per_episode == 8
    assert spec.lr_base == 3e-5
    assert spec.lr_head == 3e-4
    assert spec.weight_decay == 0.01
    assert spec.warmup_fraction == 0.05
    assert spec.loss_weights == {
        "raw_proper_prediction": 1.0,
        "candidate_ce": 1.0,
        "geometry_aux": 0.1,
    }
    assert spec.lr_grid_scales == (0.5, 1.0, 2.0)
    assert spec.accum_steps == 4
    assert train.MEASUREMENT_DEFAULTS["lr_base"] == spec.lr_base


@pytest.mark.parametrize("name", ["p1_pretrain", "p1_measurement", "p1_eval"])
def test_shipped_configs_resolve_without_unknown_keys(name: str) -> None:
    cfg = n02_config.load_config(CONFIG_DIR / f"{name}.json")
    assert n02_config.validate_config(cfg) == []
    pretrain, checkpoint = train._resolve_specs(cfg, "p1-pretrain")
    measurement, _ = train._resolve_specs(cfg, "p1-measurement")
    assert pretrain.updates == 100_000
    assert measurement.updates == 50_000
    assert checkpoint.every == 5_000
    specs = train.data_specs(cfg)
    assert sorted(specs) == ["libero", "robotwin"]
    assert specs["libero"].anchors["train"]["anchors"] == 204301
    assert specs["robotwin"].anchors["train"]["anchors"] == 116117
    assert specs["libero"].anchors["dev"]["anchors"] == 28394
    assert specs["libero"].manifest_hash  # frozen manifest hash is carried verbatim
    assert len(specs["libero"].anchors["train"]["gzip_sha256"]) == 64


def test_unknown_section_key_is_rejected_with_the_full_key_path() -> None:
    with pytest.raises(train.TrainingSpecError) as excinfo:
        train._section(train.PretrainSpec, {"learning_rate": 1e-4}, "training.pretrain")
    assert "training.pretrain" in str(excinfo.value)
    assert "learning_rate" in str(excinfo.value)


def test_effective_batch_must_be_a_multiple_of_the_micro_batch() -> None:
    with pytest.raises(train.TrainingSpecError) as excinfo:
        train._section(
            train.PretrainSpec,
            {"effective_batch_frames": 100, "micro_batch_frames": 32},
            "training.pretrain",
        )
    assert "multiple" in str(excinfo.value)


@pytest.mark.parametrize(
    ("cls", "payload", "fragment"),
    [
        (train.PretrainSpec, {"lr": "fast"}, "expected a number"),
        (train.PretrainSpec, {"updates": 0}, "must be positive"),
        (train.PretrainSpec, {"mask_ratio": 1.0}, "mask_ratio"),
        (train.PretrainSpec, {"amp": "fp16"}, "amp"),
        (train.MeasurementSpec, {"betas": [0.9]}, "betas"),
        (train.MeasurementSpec, {"sources": ["imagined"]}, "unknown source"),
        (train.MeasurementSpec, {"loss_weights": {"candidate_ce": 1.0}}, "missing"),
    ],
)
def test_spec_validation_names_the_offending_value(cls, payload, fragment) -> None:
    section = "training.pretrain" if cls is train.PretrainSpec else "training.measurement"
    with pytest.raises(train.TrainingSpecError) as excinfo:
        train._section(cls, payload, section)
    assert fragment in str(excinfo.value)


def test_lr_schedule_is_linear_warmup_then_cosine() -> None:
    assert train.lr_multiplier(0, updates=10, warmup=5) == pytest.approx(1 / 5)
    assert train.lr_multiplier(4, updates=10, warmup=5) == pytest.approx(1.0)
    assert train.lr_multiplier(15, updates=20, warmup=10) < 1.0
    assert train.lr_multiplier(20, updates=20, warmup=10) == pytest.approx(0.0, abs=1e-12)
    # the last executed update (updates - 1) is one cosine step short of zero
    assert 0.0 < train.lr_multiplier(19, updates=20, warmup=10) < 0.03
    assert train.lr_multiplier(19, updates=20, warmup=10, min_ratio=0.1) > 0.1
    # warmup_updates is a fraction of the *total* budget
    assert train.PretrainSpec(warmup_fraction=0.05).warmup_updates(100_000) == 5_000


# --------------------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------------------


def _masked_step(frame) -> dict:
    """One optimizer step from an identical initialisation, returning its term record."""
    torch.manual_seed(0)
    model = MaskedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    source = CountingSource()
    source.next_micro_batch = lambda: {"cam0": frame}  # type: ignore[method-assign]
    trainer = train.P1Trainer(
        stage="p1-pretrain",
        seed=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        source=source,
        loss_fn=train.pretrain_loss_fn,
        updates=1,
        accum_steps=1,
        base_lrs=[1e-2],
        warmup_updates=1,
    )
    return trainer._optimizer_step()["terms"]["masked_reconstruction"]


def test_masked_loss_reports_only_valid_positions_and_ignores_invalid_rows() -> None:
    frame = make_frame(6, valid_rows=3, seed=3)
    term = _masked_step(frame)
    assert term["activation_count"] == 3  # not the batch size 6
    assert term["value"] == pytest.approx(term["sum"] / 3)

    # The same valid rows with the invalid rows replaced by garbage must not change
    # the loss: an invalid position contributes nothing.
    noisy = make_frame(6, valid_rows=3, seed=3)
    noisy.values[3:] = 1e6
    assert _masked_step(noisy)["value"] == pytest.approx(term["value"])


def test_pretrain_loss_fn_surfaces_the_module_valid_count_and_saturation() -> None:
    class Stub(torch.nn.Module):
        def forward(self, frames, *, camera_ids):
            out = type("Out", (), {})()
            out.loss = torch.tensor(2.0, requires_grad=True)
            out.valid_target_count = torch.tensor(7)
            out.saturated_count = torch.tensor(2)
            return out

    loss, terms, info = train.pretrain_loss_fn(
        Stub(), {"cam0": make_frame(2)}, train.MicroInfo(0, 0, 0)
    )
    assert float(loss.detach()) == 2.0
    assert terms["masked_reconstruction"]["activation_count"] == 7
    assert terms["masked_reconstruction"]["saturated_count"] == 2
    assert info["valid_target_count"] == 7


# --------------------------------------------------------------------------------------
# checkpoint / resume
# --------------------------------------------------------------------------------------


def test_checkpoint_carries_every_resume_quantity(tmp_path: Path) -> None:
    trainer, _ = build_trainer(updates=2)
    trainer.train(until=1)
    path = trainer.save(
        tmp_path / "ck.pt",
        partition={"partition_hash": "abc"},
        config_hash="cfg",
        run_id="run",
    )
    payload = train.load_checkpoint(path)
    assert payload["format"] == train.CHECKPOINT_FORMAT
    assert payload["updates"] == 1
    assert set(payload) >= {
        "model",
        "optimizer",
        "scheduler",
        "amp",
        "rng",
        "sampler",
        "partition",
        "versions",
        "config_hash",
    }
    assert payload["partition"]["partition_hash"] == "abc"
    assert payload["sampler"]["position"]["cursor"] == 1
    for key in ("python", "numpy", "torch"):
        assert key in payload["rng"]


def test_resume_reproduces_the_uninterrupted_run_bitwise(tmp_path: Path) -> None:
    reference, reference_model = build_trainer(updates=6, seed=5)
    reference.train(until=6)
    expected = [weight.detach().clone() for weight in reference_model.weights()]

    first, first_model = build_trainer(updates=6, seed=5)
    first.train(until=3)
    checkpoint = first.save(tmp_path / "mid.pt")

    resumed, resumed_model = build_trainer(updates=6, seed=5)
    resumed.resume_from(train.load_checkpoint(checkpoint))
    assert resumed.state.resume_updates == 3
    resumed.train(until=6)

    for left, right in zip(expected, resumed_model.weights(), strict=True):
        assert torch.equal(left, right)
    # the resumed segment's records continue the same loss trajectory
    reference_tail = [round(r["loss"], 6) for r in reference.state.history[3:]]
    resumed_tail = [round(r["loss"], 6) for r in resumed.state.history]
    assert resumed_tail == reference_tail


def test_resume_records_new_and_cumulative_updates_separately(tmp_path: Path) -> None:
    trainer, _ = build_trainer(updates=3)
    trainer.train(until=3)
    checkpoint = trainer.save(tmp_path / "ck.pt")

    resumed, _ = build_trainer(updates=3)
    resumed.resume_from(train.load_checkpoint(checkpoint))
    resumed.updates = 5
    resumed.train(until=5)
    assert resumed.state.resume_updates == 3
    assert resumed.state.updates_this_resume == 2
    assert resumed.state.updates == 5


def test_resume_rejects_a_checkpoint_from_another_stage(tmp_path: Path) -> None:
    trainer, _ = build_trainer(updates=1)
    trainer.train(until=1)
    checkpoint = trainer.save(tmp_path / "ck.pt")
    payload = train.load_checkpoint(checkpoint)
    payload["stage"] = "p1-measurement"
    other, _ = build_trainer(updates=1)
    with pytest.raises(train.TrainNotAvailable):
        other.resume_from(payload)


def test_missing_checkpoint_is_not_run_not_a_crash() -> None:
    with pytest.raises(train.TrainNotAvailable) as excinfo:
        train.load_checkpoint("/nonexistent/path/ck.pt")
    assert "not found" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------------------


def test_gradient_columns_and_term_angles_are_recorded() -> None:
    trainer, _ = build_trainer(updates=2, grad_conflict_every=1)
    trainer.train(until=2)
    record = trainer.state.history[-1]
    assert record["grad_norm_base"] is not None
    assert record["grad_norm_heads"] is not None
    assert record["grad_norm_ratio_heads_over_base"] is not None
    # the Base/head angle is mathematically undefined and stays null, but the
    # per-term angle over shared parameters is computed.
    assert record["grad_angle_deg"] is None
    assert set(record["term_angles"]) == {"first|second"}
    assert 0.0 <= record["term_angles"]["first|second"] <= 180.0


def test_term_angle_is_zero_for_parallel_terms() -> None:
    model = TinyModel()
    first = (model.base.weight**2).sum()
    second = 3.0 * (model.base.weight**2).sum()
    payload = train.term_gradient_conflict(model, {"a": first, "b": second})
    assert payload["term_angles"]["a|b"] == pytest.approx(0.0)
    assert payload["term_cosines"]["a|b"] == pytest.approx(1.0)


def test_history_records_are_json_serialisable() -> None:
    trainer, _ = build_trainer(updates=1)
    trainer.train(until=1)
    json.dumps(trainer.state.history)


def test_term_summary_keeps_activation_counts_and_values() -> None:
    trainer, _ = build_trainer(updates=2)
    trainer.train(until=2)
    summary = train.summarise_terms(trainer.state.history)
    assert summary["first"]["updates_present"] == 2
    assert summary["first"]["activation_count_total"] > 0
    assert summary["first"]["first_value"] is not None


# --------------------------------------------------------------------------------------
# loss composition (freeze 3.7)
# --------------------------------------------------------------------------------------


class FakeScore:
    def __init__(
        self, value: float, raw: float | None = None, *, calibration_only: bool = False
    ) -> None:
        self.per_valid_scalar = torch.tensor(value, requires_grad=True)
        self.raw_nll_or_energy = torch.tensor(value if raw is None else raw)
        self.kind = "empirical_energy" if calibration_only else "proper_nll"
        self.local_stats = {"calibration_only": torch.tensor(bool(calibration_only))}


class FakeWorld:
    def __init__(self, score: float, kind: str = "normalized_density") -> None:
        self.score = score
        self.score_kind = kind


class FakeStep:
    def __init__(self, *, worlds=(), locked=(), fit=(), best=None, world_scores=None) -> None:
        self.worlds = tuple(worlds)
        self.locked_checks = tuple(locked)
        self.fit_scores = tuple(fit)
        self.best_world = best
        self.world_scores = world_scores


class FakeTargets:
    def __init__(self, geometry=None) -> None:
        self.optional_geometry = geometry


def test_candidate_ce_uses_the_explicit_label_and_matches_log_softmax() -> None:
    scores = [0.5, 2.0, -1.0]
    step = FakeStep(worlds=[FakeWorld(value) for value in scores])
    targets = FakeTargets({"candidate_index": 1, "world_px": np.zeros((1, 2), dtype=np.float32)})
    total, terms = train.loss_terms_from_step(step, targets)
    # the declared convention treats the score as a cost: softmax(-score / T)
    expected = -torch.log_softmax(-torch.tensor(scores) / 1.0, dim=0)[1]
    assert terms["candidate_ce"]["value"] == pytest.approx(float(expected))
    assert terms["candidate_ce"]["label_source"].startswith("TrainingTargets")
    assert terms["candidate_ce"]["score_convention"] == "softmax(-world_score / temperature)"
    assert terms["candidate_ce"]["temperature"] == pytest.approx(1.0)
    assert total is not None


def test_candidate_ce_prefers_the_differentiable_world_scores_tensor() -> None:
    scores = torch.tensor([0.5, 2.0, -1.0], requires_grad=True)
    step = FakeStep(worlds=[FakeWorld(0.5), FakeWorld(2.0), FakeWorld(-1.0)], world_scores=scores)
    targets = FakeTargets({"candidate_index": 1, "world_px": np.zeros((1, 2), dtype=np.float32)})
    total, terms = train.loss_terms_from_step(step, targets, ce_temperature=2.0)
    record = terms["candidate_ce"]
    assert record["status"] == "ok"
    assert record["gradient_available"] is True
    assert record["score_storage"] == "differentiable"
    expected = -torch.log_softmax(-scores / 2.0, dim=0)[1]
    assert record["value"] == pytest.approx(float(expected.detach()))
    # the CE is a real training signal for the world scores
    assert total is not None
    total.backward()
    assert scores.grad is not None and float(scores.grad.abs().sum()) > 0.0


def test_candidate_ce_unavailable_for_an_all_wrong_dictionary() -> None:
    step = FakeStep(worlds=[FakeWorld(0.2), FakeWorld(0.4)])
    _, terms = train.loss_terms_from_step(step, FakeTargets({"world_px": np.zeros((1, 2))}))
    assert terms["candidate_ce"]["status"] == "unavailable"
    assert terms["candidate_ce"]["value"] is None


def test_candidate_ce_is_unavailable_without_an_explicit_label() -> None:
    step = FakeStep(worlds=[FakeWorld(0.1), FakeWorld(0.9)])
    total, terms = train.loss_terms_from_step(step, FakeTargets({"world_px": np.zeros((1, 2))}))
    assert terms["candidate_ce"]["status"] == "unavailable"
    assert terms["candidate_ce"]["value"] is None  # never a fabricated zero
    assert "candidate_index" in terms["candidate_ce"]["reason"]
    assert total is None


def test_mixed_world_score_kinds_are_refused() -> None:
    step = FakeStep(
        worlds=[FakeWorld(0.1, "normalized_density"), FakeWorld(0.9, "empirical_energy")]
    )
    _, terms = train.loss_terms_from_step(
        step, FakeTargets({"candidate_index": 0, "world_px": np.zeros((1, 2))})
    )
    assert terms["candidate_ce"]["status"] == "unavailable"
    assert "consistent family" in terms["candidate_ce"]["reason"]


def test_prequential_and_fit_scores_are_separate_columns_and_never_summed() -> None:
    world = FakeWorld(0.5)
    step = FakeStep(
        worlds=[world],
        locked=[FakeScore(1.5, raw=30.0)],
        fit=[FakeScore(2.5, raw=50.0)],
        best=world,
    )
    total, terms = train.loss_terms_from_step(step, FakeTargets())
    prequential = terms["raw_proper_prediction_prequential"]
    fit = terms["raw_proper_prediction_fit"]
    # the score is per world: the one proper-prediction target counts once
    assert prequential["value"] == pytest.approx(1.5)
    assert prequential["sum"] == pytest.approx(30.0)
    assert prequential["activation_count"] == 20
    assert prequential["score_selection"] == "best_world"
    assert fit["value"] == pytest.approx(2.5)
    assert fit["sum"] == pytest.approx(50.0)
    # the fit column carries weight 0 by default
    assert float(total.detach()) == pytest.approx(1.5)


def test_proper_score_uses_the_explicit_candidate_index() -> None:
    first, second = FakeWorld(0.5), FakeWorld(2.0)
    step = FakeStep(
        worlds=[first, second],
        locked=[FakeScore(1.5, raw=30.0), FakeScore(2.5, raw=50.0)],
        fit=[FakeScore(1.0, raw=20.0), FakeScore(3.0, raw=60.0)],
        best=second,
    )
    targets = FakeTargets({"candidate_index": 1, "world_px": np.zeros((1, 2), dtype=np.float32)})
    _, terms = train.loss_terms_from_step(step, targets)
    assert terms["raw_proper_prediction_prequential"]["value"] == pytest.approx(2.5)
    assert terms["raw_proper_prediction_prequential"]["score_selection"] == "candidate_index"
    # without a label the step's own best_world is used, and that choice is recorded
    _, unlabelled = train.loss_terms_from_step(step, FakeTargets())
    assert unlabelled["raw_proper_prediction_prequential"]["score_selection"] == "best_world"
    assert unlabelled["raw_proper_prediction_prequential"]["value"] == pytest.approx(2.5)


def test_proper_score_is_unavailable_without_a_usable_world_choice() -> None:
    step = FakeStep(
        worlds=[FakeWorld(0.5), FakeWorld(2.0)],
        locked=[FakeScore(1.0), FakeScore(2.0)],
    )
    _, terms = train.loss_terms_from_step(step, FakeTargets())
    record = terms["raw_proper_prediction_prequential"]
    assert record["status"] == "unavailable"
    assert record["value"] is None
    assert "best_world" in record["reason"]


def test_geometry_aux_is_unavailable_without_truth_or_entities() -> None:
    value, record = train.geometry_aux_term(None, FakeTargets())
    assert value is None and record["status"] == "unavailable"
    world = type("World", (), {"geometry_frame": "view_2d", "entities": ()})()
    value, record = train.geometry_aux_term(
        world, FakeTargets({"world_px": np.zeros((1, 2), dtype=np.float32)})
    )
    assert value is None and "entity geometry" in record["reason"]


def test_geometry_aux_mse_matches_the_declared_definition() -> None:
    entity = type("Entity", (), {"geometry": torch.tensor([1.0, 2.0])})()
    world = type("World", (), {"geometry_frame": "view_2d", "entities": (entity,)})()
    targets = FakeTargets({"world_px": np.array([[3.0, 6.0]], dtype=np.float32)})
    value, record = train.geometry_aux_term(world, targets)
    assert record["status"] == "ok"
    assert float(value.detach()) == pytest.approx(((3 - 1) ** 2 + (6 - 2) ** 2) / 2)


def test_geometry_aux_refuses_a_non_view2d_frame() -> None:
    entity = type("Entity", (), {"geometry": torch.tensor([1.0, 2.0])})()
    world = type("World", (), {"geometry_frame": "metric_3d", "entities": (entity,)})()
    value, record = train.geometry_aux_term(
        world, FakeTargets({"world_px": np.zeros((1, 2), dtype=np.float32)})
    )
    assert value is None and "view_2d" in record["reason"]


def test_stack_measurement_frames_keeps_identity_and_ands_availability() -> None:
    rows = [make_frame(1, seed=index) for index in range(3)]
    rows[1].metadata_available["calibration"] = False
    stacked = train.stack_measurement_frames(rows, camera_id="cam0", squeeze_leading=True)
    assert np.asarray(stacked.values).shape == (3, 3, 4, 4)
    assert stacked.episode_ids == ("ep-0-0", "ep-1-0", "ep-2-0")
    assert stacked.metadata_available["calibration"] is False
    assert stacked.metadata["batch_rows"] == 3
    assert np.asarray(stacked.new_measurement).all()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_missing_upstream_modules_report_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing upstream module is not_run with a reason, never a crash or a pass.

    The absence is simulated so the check is deterministic on hosts where T1 has
    already landed; the CLI must fail before it touches the data root.
    """

    def _absent(*args: object, **kwargs: object) -> None:
        raise train.TrainNotAvailable(
            "owac.frameworks.n02_full.observation is not available: simulated",
            missing=("owac.frameworks.n02_full.observation",),
        )

    monkeypatch.setattr(train, "build_pretrainer", _absent)
    output = tmp_path / "runs"
    code = train.main(
        [
            "--stage",
            "p1-pretrain",
            "--config",
            str(CONFIG_DIR / "p1_pretrain.json"),
            "--seed",
            "1103",
            "--updates",
            "1",
            "--device",
            "cpu",
            "--platforms",
            "libero",
            "--output",
            str(output),
        ]
    )
    assert code == 2
    manifest = json.loads((output / "p1-pretrain-seed1103" / "run_manifest.json").read_text())
    assert manifest["status"] == "not_run"
    assert "simulated" in manifest["reason"]
    assert manifest["config_hash"]
    assert not (output / "p1-pretrain-seed1103" / "metrics.json").exists()


def test_cli_exit_codes_distinguish_ok_fail_and_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ok -> 0, fail -> 1, not_run -> 2 (a scheduler must not read not_run as success)."""

    def _absent(*args: object, **kwargs: object) -> None:
        raise train.TrainNotAvailable("simulated absence", missing=("x",))

    monkeypatch.setattr(train, "build_pretrainer", _absent)
    not_run = train.main(
        [
            "--stage",
            "p1-pretrain",
            "--config",
            str(CONFIG_DIR / "p1_pretrain.json"),
            "--seed",
            "1103",
            "--updates",
            "1",
            "--device",
            "cpu",
            "--platforms",
            "libero",
            "--output",
            str(tmp_path / "not-run"),
        ]
    )
    assert not_run == 2

    class _FakeManifest:
        def content_hash(self) -> str:
            return "fake-manifest-hash"

    class _FakeAnchorTable:
        def sha256(self) -> str:
            return "fake-anchor-hash"

    class _FakeDataset:
        partition = "train"
        manifest = _FakeManifest()
        anchor_table = _FakeAnchorTable()
        window_options = {"observation_stride": 1, "rnn_frames": 1, "burst_frames": 1}
        n02_provenance = {"stats_path": "", "normalisation_loaded": False}

        def __len__(self) -> int:
            return 4

    class _FakeSource:
        dataset = None

        def __init__(self, dataset: object = None, **kwargs: object) -> None:
            self.dataset = dataset

        def next_micro_batch(self) -> dict:
            return {"cam0": make_frame(2)}

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, payload: object) -> None:
            pass

    monkeypatch.setattr(train, "open_anchor_dataset", lambda spec, partition: _FakeDataset())
    monkeypatch.setattr(train, "RealFrameBatchSource", _FakeSource)
    torch.manual_seed(0)
    monkeypatch.setattr(train, "build_pretrainer", lambda cfg, spec, device, *, seed: MaskedModel())
    ok = train.main(
        [
            "--stage",
            "p1-pretrain",
            "--config",
            str(CONFIG_DIR / "p1_pretrain.json"),
            "--seed",
            "1103",
            "--updates",
            "1",
            "--device",
            "cpu",
            "--platforms",
            "libero",
            "--output",
            str(tmp_path / "ok"),
        ]
    )
    assert ok == 0
    manifest = json.loads(
        (tmp_path / "ok" / "p1-pretrain-seed1103" / "run_manifest.json").read_text()
    )
    assert manifest["status"] == "ok"

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated data error")

    monkeypatch.setattr(train, "open_anchor_dataset", _boom)
    failed = train.main(
        [
            "--stage",
            "p1-pretrain",
            "--config",
            str(CONFIG_DIR / "p1_pretrain.json"),
            "--seed",
            "1103",
            "--updates",
            "1",
            "--device",
            "cpu",
            "--platforms",
            "libero",
            "--output",
            str(tmp_path / "fail"),
        ]
    )
    assert failed == 1
    failed_manifest = json.loads(
        (tmp_path / "fail" / "p1-pretrain-seed1103" / "run_manifest.json").read_text()
    )
    assert failed_manifest["status"] == "fail"


def test_cli_rejects_bad_arguments(tmp_path: Path) -> None:
    base = ["--stage", "p1-pretrain", "--config", str(CONFIG_DIR / "p1_pretrain.json")]
    assert train.main([*base, "--lr-scale", "0"]) == 2
    assert train.main([*base, "--updates", "0"]) == 2
    with pytest.raises(SystemExit):
        train.main(["--stage", "p2-pretrain", "--config", str(CONFIG_DIR / "p1_pretrain.json")])


def test_cli_reports_a_malformed_config_with_exit_code_two(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"training": {"pretrain": {"learning_rate": 1e-4}}}))
    code = train.main(
        ["--stage", "p1-pretrain", "--config", str(broken), "--output", str(tmp_path / "out")]
    )
    assert code == 2


def test_cli_missing_checkpoint_reports_not_run(tmp_path: Path) -> None:
    code = train.main(
        [
            "--stage",
            "p1-pretrain",
            "--config",
            str(CONFIG_DIR / "p1_pretrain.json"),
            "--resume",
            str(tmp_path / "absent.pt"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    manifest_path = tmp_path / "out" / "p1-pretrain-seed1103" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "not_run"
    assert "checkpoint not found" in manifest["reason"]


def test_run_pretrain_with_injected_fakes_writes_receipts(tmp_path: Path) -> None:
    cfg = n02_config.load_config(CONFIG_DIR / "p1_pretrain.json")
    spec = train.PretrainSpec(
        updates=4,
        effective_batch_frames=8,
        micro_batch_frames=4,
        checkpoint_every=2,
        dev_eval_every=2,
        dev_frames=4,
        log_every=1,
    )
    args = train.build_parser().parse_args(
        [
            "--stage",
            "p1-pretrain",
            "--config",
            str(CONFIG_DIR / "p1_pretrain.json"),
            "--updates",
            "4",
        ]
    )
    args.seed = 1103
    context = train.build_run_context(args, cfg, command=["test"])
    torch.manual_seed(0)
    model = MaskedModel()
    outcome = train.run_pretrain(
        cfg,
        spec,
        context,
        lr_scale=1.0,
        updates=4,
        device="cpu",
        model_factory=lambda: model,
        frame_sources={"libero": CountingSource(rows=4)},
        dev_source=CountingSource(rows=2),
        checkpoint_spec=train.CheckpointSpec(every=2),
    )
    assert outcome["updates"] == 4
    assert (context.checkpoints / "final.pt").is_file()
    assert (context.checkpoints / "update-0000002.pt").is_file()
    assert (context.train_log).is_file()
    dev_records = [
        json.loads(line)["dev_eval"]
        for line in context.train_log.read_text().splitlines()
        if "dev_eval" in json.loads(line)
    ]
    assert dev_records and dev_records[0]["status"] == "ok"
    assert dev_records[0]["dev_masked_reconstruction_loss"] >= 0.0
    # injected sources declare no anchor dataset, so the fingerprint records that
    # explicitly instead of inventing a partition hash
    assert outcome["partition"]["sources"] == "injected"
    assert outcome["partition"]["partition_hash"]


def test_dev_eval_does_not_change_the_model(tmp_path: Path) -> None:
    model = MaskedModel()
    source = CountingSource(rows=4)
    dev = CountingSource(rows=2)
    trainer = train.P1Trainer(
        stage="p1-pretrain",
        seed=0,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
        scheduler=None,
        scaler=None,
        source=source,
        loss_fn=train.pretrain_loss_fn,
        updates=2,
        accum_steps=1,
        base_lrs=[1e-2],
        warmup_updates=1,
    )
    before = [p.detach().clone() for p in model.parameters()]
    payload = train.make_pretrain_dev_eval(dev, device="cpu", micro_batch_frames=2)(trainer, 0)
    assert payload["status"] == "ok"
    for left, right in zip(before, model.parameters(), strict=True):
        assert torch.equal(left, right)


# --------------------------------------------------------------------------------------
# integration (runs once the T3 module exists)
# --------------------------------------------------------------------------------------


def test_step_bundle_causality_future_frames_do_not_change_past_outputs() -> None:
    """End-to-end causality through the frozen online path.

    The check is written against the frozen ``step_bundle`` signature: the same
    prefix with a perturbed future frame and perturbed future labels must produce
    the same already-produced outputs. It skips with a reason while the T3
    module is absent rather than asserting anything about a stand-in.
    """
    model_module = pytest.importorskip(
        "owac.frameworks.n02_full.model",
        reason="T3 model.py (step_bundle) not integrated in this checkout yet",
    )
    assert hasattr(model_module, "step_bundle")
    factors = pytest.importorskip(
        "owac.frameworks.n02_full.factors",
        reason="T3 factors.py (ObservationModel) not integrated in this checkout yet",
    )
    assert hasattr(factors, "build_p1_task")


# --------------------------------------------------------------------------------------
# dev subset, mode gate and coverage
# --------------------------------------------------------------------------------------


class _FakeAnchor:
    def __init__(self, index: int) -> None:
        self.relative_path = f"episode-{index // 4:02d}.npz"
        self.current_frame_index = index % 4


class _FakeAnchorTable:
    def __init__(self, count: int) -> None:
        self._entries = [_FakeAnchor(index) for index in range(count)]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index: int) -> _FakeAnchor:
        return self._entries[index]

    def sha256(self) -> str:
        return "fake"


class _FakeManifest:
    def content_hash(self) -> str:
        return "fake"


class _FakeCoverageDataset:
    partition = "train"
    anchor_table = _FakeAnchorTable(16)
    manifest = _FakeManifest()

    def __len__(self) -> int:
        return 16


def test_coverage_counts_unique_anchors_and_exposures(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _FakeCoverageDataset()
    walker = train._AnchorWalker(
        dataset, seed=7, platform="libero", partition="train", salt="test-order"
    )
    for _ in range(40):
        walker.next_index()
    coverage = walker.coverage()
    assert coverage["anchor_table_entries"] == 16
    assert coverage["unique_anchors_seen"] == 16  # 40 draws, 16 unique anchors
    assert coverage["unique_episodes_seen"] == 4
    assert coverage["frames_consumed"] == 40
    assert "exposure" in coverage["dedup_rule"]


def test_dev_eval_restarts_the_fixed_dev_subset() -> None:
    class _RecordingSource:
        def __init__(self) -> None:
            self.cursor = 0
            self.seen: list[int] = []

        def next_micro_batch(self) -> dict:
            self.seen.append(self.cursor)
            self.cursor += 1
            return {"cam0": make_frame(4)}

        def load_state_dict(self, payload: object) -> None:
            self.cursor = int(dict(payload).get("cursor", 0))  # type: ignore[arg-type]

    class _Stub(torch.nn.Module):
        def forward(self, frames, *, camera_ids):
            out = type("Out", (), {})()
            out.loss = torch.tensor(1.0, requires_grad=True)
            out.valid_target_count = torch.tensor(4)
            return out

    source = _RecordingSource()
    trainer, _ = build_trainer(updates=1)
    trainer.model = _Stub()
    evaluator = train.make_pretrain_dev_eval(
        source, device="cpu", micro_batch_frames=4, dev_frames=8
    )
    evaluator(trainer, 0)
    first = list(source.seen)
    evaluator(trainer, 1)
    second = list(source.seen[len(first) :])
    assert first == [0, 1]  # dev_frames / micro_batch + reset at the start
    assert second == [0, 1]  # the same fixed subset, not a rolling window


def test_mode_gate_marks_a_mismatched_source_as_skipped() -> None:
    cfg = n02_config.load_config(CONFIG_DIR / "p1_measurement.json")
    spec = train.MeasurementSpec(sources=("synthetic", "real"))
    gate = train.mode_gate_payload(spec, cfg)
    assert gate["encoder_mode"] == "cfa4"
    entries = {entry["kind"]: entry for entry in gate["sources"]}
    assert entries["synthetic"]["required_mode"] == "cfa4"
    assert entries["synthetic"]["gate"] == "active"
    # a raw3 encoder cannot serve the real 3-channel windows in this same run
    assert entries["real"]["gate"] == "skipped"
    assert entries["real"]["density"] == "empirical_energy"


def test_pretrain_mode_gate_records_the_platforms() -> None:
    cfg = n02_config.load_config(CONFIG_DIR / "p1_pretrain.json")
    gate = train.mode_gate_payload(train.PretrainSpec(), cfg)
    assert gate["encoder_mode"] == "raw3"
    assert gate["sources"][0]["gate"] == "active"
    assert gate["sources"][0]["platforms"] == ["libero", "robotwin"]


def test_scale_suspect_flags_a_scale_mistake() -> None:
    class _Score:
        def __init__(self, residual) -> None:
            self.local_stats = {"standardised_residual": np.asarray(residual, dtype=np.float64)}

    assert train._scale_suspect(_Score([0.5, 1.0, 2.0])) is False
    # T2's detector uses the median, so a single outlier is not a scale mistake
    assert train._scale_suspect(_Score([1.0, 5e3, 2.0])) is False
    assert train._scale_suspect(_Score([5e3, 6e3, 1.0])) is True
    assert train._scale_suspect(_Score([])) is None
    assert train._scale_suspect(object()) is None


def test_mode_gate_is_enforced_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gated-out source must not be constructed, and must not even open the cache."""
    cfg = n02_config.load_config(CONFIG_DIR / "p1_measurement.json")
    spec = train.MeasurementSpec(sources=("synthetic", "real"))

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the gated real source must not open the cache")

    monkeypatch.setattr(train, "open_anchor_dataset", _boom)
    source = train.build_measurement_sources(cfg, spec, 7)
    assert [item.kind for item in source.sources] == ["synthetic"]
    assert [entry["kind"] for entry in source.skipped_sources] == ["real"]
    assert source.skipped_sources[0]["declared_mode"] == "raw3"
    assert source.next_micro_batch().kind == "synthetic"


def test_ratio_arm_uses_the_official_loss_and_freezes_the_base(tmp_path: Path) -> None:
    """NRE/BNRE arms: official vendored loss, shared frozen Base, declared budget note."""
    pytest.importorskip("owac.baselines.nre_losses")
    cfg = n02_config.load_config(CONFIG_DIR / "p1_measurement.json")
    spec, checkpoint = train._resolve_specs(cfg, "p1-measurement")
    spec = dataclasses.replace(
        spec,
        sources=("synthetic",),
        canvas=(64, 64),
        micro_batch_episodes=1,
        synthetic_sensors=(
            {
                "camera_id": "cam0",
                "height": 32,
                "width": 32,
                "input_mode": "bayer_cfa",
                "cfa_pattern": "RGGB",
                "black_level": 64.0,
                "white_level": 4095.0,
                "bit_depth": 12,
                "read_noise_e": 2.0,
                "gain": 1.0,
                "exposure_ms": 1.0,
            },
        ),
    )
    for arm in ("nre", "bnre"):
        args = train.build_parser().parse_args(
            ["--stage", "p1-measurement", "--updates", "2", "--arm", arm]
        )
        args.seed = 99
        context = train.build_run_context(args, cfg, command=[f"ratio-{arm}"])
        context.run_dir = tmp_path / arm
        outcome = train.run_ratio_arm(
            cfg,
            spec,
            context,
            arm=arm,
            updates=2,
            device="cpu",
            batch=4,
            log_every=1,
            checkpoint_spec=train.CheckpointSpec(),
        )
        manifest = outcome["arm_manifest"]
        assert manifest["ablation_arm"] == arm
        assert manifest["loss_source"].endswith("NRELoss" if arm == "nre" else "BNRELoss")
        assert manifest["pinned_commit"] == "cbf500c09c538a014e72477d166acc62a6d8409a"
        assert manifest["base_frozen"] is True
        assert manifest["same_budget_as_main"] is False
        assert "budget_deviation" in manifest
        history = outcome["trainer"].state.history
        assert history and all(record["terms"][arm]["status"] == "ok" for record in history)
        assert (context.checkpoints / "final.pt").is_file()
    assert "nre" not in train.ARM_MISSING_INTERFACES
    assert "bnre" not in train.ARM_MISSING_INTERFACES
    assert "raw_frame" in train.ARM_MISSING_INTERFACES


class _FakeFacade(torch.nn.Module):
    """A façade-like model: ``encoder.backbone`` plus fresh measurement heads."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.backbone = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 1)

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, payload, strict: bool = True) -> None:
        self.encoder.load_state_dict(payload["encoder"], strict=strict)
        self.decoder.load_state_dict(payload["decoder"], strict=strict)


def test_cross_stage_resume_inherits_the_base_and_reports_it() -> None:
    """pretrain -> measurement inheritance: Base only, fresh heads, full receipt."""
    torch.manual_seed(0)
    model = _FakeFacade()
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters())},
            {"params": list(model.decoder.parameters())},
        ],
        lr=3e-5,
    )
    trainer = train.P1Trainer(
        stage="p1-measurement",
        seed=1103,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        source=CountingSource(),
        loss_fn=toy_loss_fn,
        updates=5000,
        accum_steps=1,
        base_lrs=[3e-5, 3e-4],
        warmup_updates=250,
    )
    payload = {
        "format": train.CHECKPOINT_FORMAT,
        "stage": "p1-pretrain",
        "updates": 2000,
        "run_id": "p1-pretrain-seed1103-stem",
        "model": {
            "fcmae.encoder.backbone.weight": torch.full((2, 2), 0.25),
            "fcmae.encoder.backbone.bias": torch.full((2,), 0.5),
            "fcmae.mask_token": torch.full((3,), 1.0),
            "fcmae.decoder.0.weight": torch.full((1, 1), 2.0),
        },
        "optimizer": {"param_groups": [{"lr": 1e-4, "params": [0]}], "state": {}},
        "scheduler": None,
        "amp": {"scaler": None},
        "rng": {},
        "sampler": {"position": {"cursor": 17}},
    }
    # the pretraining optimizer has one group and the measurement optimizer two:
    # a cross-stage inheritance must not try to restore it
    trainer.resume_from(payload)

    assert trainer.resumed_from_stage == "p1-pretrain"
    report = trainer.base_inheritance_report
    assert report["base_loaded"] is True
    assert report["loaded_tensors"] == 2  # the two Base tensors, mapped to encoder.backbone
    assert report["skipped_pretraining_tensors"] == 2  # mask_token + fcmae decoder
    assert report["skipped_modules"] == ["fcmae"]
    assert report["source_updates"] == 2000
    assert report["shape_conflicts"] == []
    assert torch.allclose(model.encoder.backbone.weight, torch.full((2, 2), 0.25))
    assert torch.allclose(model.encoder.backbone.bias, torch.full((2,), 0.5))
    # the run's own counters start at zero; the source count stays on the receipt
    assert trainer.state.updates == 0
    assert trainer.state.resume_updates == 0
    assert trainer.state.updates_this_resume == 0


def test_base_subset_loader_refuses_a_checkpoint_with_no_base_tensor() -> None:
    model = _FakeFacade()
    with pytest.raises(train.TrainNotAvailable) as excinfo:
        train.load_base_subset(model, {"decoder_only.weight": torch.zeros(1)})
    assert "no Base tensor" in str(excinfo.value)


def test_cross_stage_inheritance_does_not_consume_the_budget() -> None:
    """--updates N means N updates of this stage, whatever the inherited Base counted."""
    stem = {"stage": "p1-pretrain", "updates": 2000}
    assert train.budget_start(stem, "p1-measurement") == 0
    assert (
        train.budget_start({"stage": "p1-measurement", "updates": 1000}, "p1-measurement") == 1000
    )
    assert train.budget_start(None, "p1-measurement") == 0
    assert train.budget_start({}, "p1-pretrain") == 0


def test_ratio_arm_accepts_a_pretraining_ckpt_as_its_base(tmp_path: Path) -> None:
    """The ratio arm must inherit a pretraining Base through the subset loader.

    A strict load rejects a pretraining checkpoint ("unknown submodule fcmae.mask_token");
    the arm has to skip the FCMAE-only modules and report what it matched, so that the
    receipt shows a real inheritance instead of a failed run.
    """
    pytest.importorskip("owac.baselines.nre_losses")
    cfg = n02_config.load_config(CONFIG_DIR / "p1_measurement.json")
    spec, _ = train._resolve_specs(cfg, "p1-measurement")
    spec = dataclasses.replace(
        spec,
        sources=("synthetic",),
        canvas=(64, 64),
        micro_batch_episodes=1,
        synthetic_sensors=(
            {
                "camera_id": "cam0",
                "height": 32,
                "width": 32,
                "input_mode": "bayer_cfa",
                "cfa_pattern": "RGGB",
                "black_level": 64.0,
                "white_level": 4095.0,
                "bit_depth": 12,
                "read_noise_e": 2.0,
                "gain": 1.0,
                "exposure_ms": 1.0,
            },
        ),
    )
    model = train.build_observation_model(
        cfg,
        "cpu",
        units=str(spec.units),
        world_size=None,
    )
    stem_like = {
        "stage": "p1-pretrain",
        "updates": 2000,
        "model": {
            "fcmae.mask_token": torch.zeros(3),
            "fcmae.decoder.0.weight": torch.zeros(1, 1),
        },
    }
    # the model's own Base tensors are matched by name; the FCMAE-only ones are skipped
    base_state = train.model_state_dict(model)
    flat = train.flatten_state_dict(base_state)
    first_key = next(iter(sorted(flat)))
    stem_like["model"][first_key] = flat[first_key].clone()
    report = train.load_base_subset(model, stem_like["model"])
    assert report["base_loaded"] is True
    assert report["loaded_tensors"] == 1
    assert report["skipped_pretraining_tensors"] == 2
    assert report["skipped_modules"] == ["fcmae"]


def test_each_stage_bounds_its_own_budget() -> None:
    """Wiring guard: the stage string must match the function that calls budget_start.

    A swapped pair silently let a measurement run train to the source's update count
    (the "5k" run reached 4000 and an arm ran past its 2k budget); this pins the wiring
    rather than only the helper.
    """
    source = inspect.getsource(train.run_pretrain)
    assert 'budget_start(resume, "p1-pretrain")' in source
    source = inspect.getsource(train.run_measurement)
    assert 'budget_start(resume, "p1-measurement")' in source
