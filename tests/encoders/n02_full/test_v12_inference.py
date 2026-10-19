"""N02 v1.2 inference tests on CPU with conforming stand-ins for the parallel-owned modules.

``encoder_v12`` / ``decoder_v12`` / ``readout_v12`` / ``checks_v12`` are written by other owners,
so these tests build tiny objects that satisfy exactly the frozen interfaces and exercise the loop
itself: joint candidates, the declared prior really changing ``log_p0``/``q``, the accept-only-if-
the-complete-objective-does-not-increase rule, unique measurement IDs, reset semantics, save/resume
replay, gradient paths and the explicit handling of a missing check verdict.

The real-module integration test lives at the end and skips while those modules are absent.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full import inference_v12 as v12
from owac.frameworks.n02_full.model_v12 import LockedAbsoluteCheck, locked_absolute_check
from owac.frameworks.n02_full.sensor import MeasurementScore
from owac.frameworks.n02_full.state import ObservationBundle, SequenceState

BATCH = 2
SIDE = 16
CAMERAS = ("cam_front", "cam_wrist")
D_MODEL = 16
SLOT_DIM = 32
N_SLOTS = 4
K = 4
ROUNDS = 4


# ---------------------------------------------------------------------------------------
# conforming stand-ins
# ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class FakeSpatial:
    levels: dict
    coords: dict
    valid: dict
    camera_ids: tuple
    level_ids: tuple
    level_dims: dict
    support_hash: str = "fake-support"

    def per_camera(self) -> dict:
        return {
            camera: {
                level_id: self.levels[f"{camera}::{level_id}"]
                for level_id in self.level_ids
                if f"{camera}::{level_id}" in self.levels
            }
            for camera in self.camera_ids
        }


class FakeEncoder(nn.Module):
    """Multi-scale features with real coordinates/validity, as ``RawEncoderV12`` declares."""

    def __init__(self, camera_ids=CAMERAS, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.camera_ids = tuple(camera_ids)
        self.d_model = d_model
        self.stem = nn.Conv2d(3, d_model, 3, stride=4, padding=1)
        self.extra = nn.Conv2d(d_model, d_model, 3, padding=1)

    @property
    def spatial_dims(self) -> dict[str, int]:
        return {level: self.d_model for level in ("stride4", "stride8", "stride16")}

    def forward(self, observation: ObservationBundle) -> FakeSpatial:
        levels: dict[str, torch.Tensor] = {}
        coords: dict[str, torch.Tensor] = {}
        valid: dict[str, torch.Tensor] = {}
        for camera in self.camera_ids:
            frame = observation.frames[camera]
            values = torch.as_tensor(frame.values).float()
            batch = values.shape[0]
            base = F.relu(self.extra(F.relu(self.stem(values))))
            mask = torch.as_tensor(frame.valid_mask).float().unsqueeze(1)
            for level_id, tensor, factor in (
                ("stride4", base, 4),
                ("stride8", F.avg_pool2d(base, 2), 8),
                ("stride16", F.avg_pool2d(base, 4), 16),
            ):
                height, width = tensor.shape[-2:]
                key = f"{camera}::{level_id}"
                levels[key] = tensor
                rows = torch.arange(height, dtype=torch.float32) * factor
                cols = torch.arange(width, dtype=torch.float32) * factor
                grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
                coords[key] = grid.unsqueeze(0).expand(batch, -1, -1, -1)
                pooled = F.avg_pool2d(mask, factor).reshape(batch, height, width)
                valid[key] = pooled > 0.99
        return FakeSpatial(
            levels=levels,
            coords=coords,
            valid=valid,
            camera_ids=self.camera_ids,
            level_ids=("stride4", "stride8", "stride16"),
            level_dims=self.spatial_dims,
        )


class FakeDecoder(nn.Module):
    """Camera/time conditioned prediction of the same sensor space (mean, variance, masks)."""

    def __init__(self, d_model: int = D_MODEL, slot_dim: int = SLOT_DIM) -> None:
        super().__init__()
        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.coord_proj = nn.Linear(4, d_model)
        self.out = nn.Linear(d_model, 1)
        self.log_var = nn.Linear(d_model, 1)

    def forward(
        self,
        *,
        features: FakeSpatial,
        slots: torch.Tensor,
        frame_index: torch.Tensor,
        query_coords: torch.Tensor,
        query_camera_index: torch.Tensor,
        query_channel: torch.Tensor | None = None,
    ):
        batch, candidates, n_slots, slot_dim = slots.shape
        device = query_coords.device
        candidate = self.slot_proj(slots.mean(dim=2))
        context = torch.zeros(
            (batch, query_coords.shape[1], D_MODEL), device=device, dtype=query_coords.dtype
        )
        per_camera: list[torch.Tensor] = []
        for camera in features.camera_ids:
            key = f"{camera}::stride16"
            pooled = features.levels[key].mean(dim=(-2, -1))
            per_camera.append(pooled)
        camera_features = torch.stack(per_camera, dim=1)  # (B, n_cam, D)
        selected = camera_features.gather(
            1, query_camera_index.unsqueeze(-1).expand(-1, -1, D_MODEL)
        )
        context = context + selected
        channel = torch.zeros_like(query_camera_index) if query_channel is None else query_channel
        context = context + self.coord_proj(
            torch.cat(
                [
                    query_coords / float(SIDE),
                    query_camera_index.unsqueeze(-1).to(query_coords.dtype),
                    channel.unsqueeze(-1).to(query_coords.dtype) / 3.0,
                ],
                dim=-1,
            )
        )
        logits = torch.einsum("bkd,bqd->bkq", candidate, context)
        mean = self.out(context).squeeze(-1).unsqueeze(1) + logits
        variance = F.softplus(self.log_var(context)).squeeze(-1).unsqueeze(1) + 1e-3
        variance = variance.expand(batch, candidates, -1).contiguous()
        flat = query_coords.shape[:2]
        valid_probability = torch.ones(*flat, device=device)
        masks = {
            "valid": valid_probability,
            "bad": torch.zeros(*flat, device=device),
            "missing": torch.zeros(*flat, device=device),
            "saturated": torch.zeros(*flat, device=device),
            "in_support": torch.ones(
                (batch, candidates, *flat[1:]), dtype=torch.bool, device=device
            ),
        }
        return mean, variance, masks


class FakeReadout(nn.Module):
    """Small Perceiver stand-in: direct spatial features + K states + support -> tokens."""

    N_QUERIES = 256
    OUT_DIM = 2048

    def __init__(self, camera_count: int, d_model: int = D_MODEL, slot_dim: int = SLOT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(d_model + slot_dim, slot_dim)

    def forward(self, *, features: FakeSpatial, state_tokens, support, masks=None):
        batch = state_tokens.shape[0]
        pooled = torch.stack(
            [
                features.levels[f"{camera}::stride16"].mean(dim=(-2, -1))
                for camera in features.camera_ids
            ],
            dim=1,
        ).mean(dim=1)
        weighted = state_tokens * support.unsqueeze(-1).unsqueeze(-1)
        state = weighted.reshape(batch, K, N_SLOTS, SLOT_DIM).mean(dim=(1, 2))
        tokens = self.project(torch.cat([pooled, state], dim=-1)).unsqueeze(1)
        tokens = tokens.expand(batch, 2, SLOT_DIM).contiguous()
        attention = torch.ones(batch, 4, dtype=torch.float32)
        return tokens, {"attention": attention, "latents": state}


@dataclass(slots=True)
class FakeCheck:
    status: str
    valid: bool
    statistics: dict
    calibration_version: str


class FakeMeasurement:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.calls = 0
        self.last_masks: dict | None = None

    def energy(self, *, mean, variance, masks, target) -> torch.Tensor:
        self.calls += 1
        self.last_masks = masks
        valid = masks.to(mean.dtype)
        residual = target.unsqueeze(1) - mean
        nll = 0.5 * (residual.square() / variance + variance.log())
        return self.scale * (nll * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)

    def absolute_check(self, *, residuals, calibrator=None, **_: object):
        views = [view for group in residuals for view in group]
        magnitude = max(
            (float(view.standardised_residual.abs().max().detach()) for view in views), default=0.0
        )
        return FakeCheck(
            status="compatible",
            valid=True,
            statistics={"mean_abs": magnitude},
            calibration_version="fake-calibration-v1",
        )


class NoVerdictCheck:
    """A malformed result: no status, exactly what P1 turned into the string "None"."""

    valid = True


# ---------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------


def make_frame(
    camera: str, *, filled: float, valid: bool = True, frame_id: str = "7"
) -> MeasurementFrame:
    values = torch.full((BATCH, 3, SIDE, SIDE), float(filled))
    values[:, 1] += torch.arange(SIDE).view(1, 1, SIDE).float() / 10.0
    mask = torch.full((BATCH, SIDE, SIDE), bool(valid))
    if not valid:
        mask[:, : SIDE // 2] = False
    return MeasurementFrame(
        values=values,
        valid_mask=mask,
        camera_id=camera,
        frame_ids=tuple(frame_id for _ in range(BATCH)),
        metadata={"units": "unit_test_sensor"},
    )


def make_input(
    *,
    filled: float = 1.0,
    prior=None,
    measurement_ids=None,
    valid: bool = True,
    frame_id: str = "7",
):
    frames = {
        camera: make_frame(camera, filled=filled, valid=valid, frame_id=frame_id)
        for camera in CAMERAS
    }
    bundle = ObservationBundle(
        frames=frames,
        context=EmbodimentContext(),
        episode_ids=tuple("episode-1" for _ in range(BATCH)),
        anchor_ids=tuple("anchor-1" for _ in range(BATCH)),
    )
    return v12.V12InferenceInput(
        observation=bundle,
        declared_prior=prior,
        frame_index=torch.tensor([int(frame_id)] * BATCH),
        measurement_ids=measurement_ids,
    )


def build_model(
    *, rounds: int = ROUNDS, measurement=None, strict_check: bool = True, tau: float = 1.0
):
    from owac.frameworks.n02_full.model_v12 import N02V12Model

    evidence_dim = D_MODEL * 3 * len(CAMERAS)
    torch.manual_seed(1103)
    return N02V12Model(
        encoder=FakeEncoder(),
        decoder=FakeDecoder(),
        readout=FakeReadout(len(CAMERAS)),
        measurement=measurement if measurement is not None else FakeMeasurement(),
        n_slots=N_SLOTS,
        slot_dim=SLOT_DIM,
        tau=tau,
        rounds=rounds,
        k=K,
        d_model=D_MODEL,
        evidence_dim=evidence_dim,
        strict_check=strict_check,
    )


# ---------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------


def test_step_shapes_support_and_history() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    output = model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state)

    assert output.q.shape == (BATCH, K)
    assert torch.allclose(output.q.sum(dim=-1), torch.ones(BATCH), atol=1e-5)
    expected = torch.softmax(output.log_p0 - output.energies / model.inference.tau, dim=-1)
    assert torch.allclose(output.q, expected, atol=1e-6)
    assert output.mean.shape == output.variance.shape == (BATCH, K, output.mean.shape[-1])
    assert output.tokens.shape == (BATCH, 2, SLOT_DIM)
    assert len(output.rounds) == ROUNDS
    history = output.state.geometry[0]
    assert isinstance(history, v12.V12RowHistory)
    assert len(history.entries) == 1
    assert history.tokens is not None and history.attention is not None
    assert output.state.existence.shape == (BATCH, K)
    assert output.diagnostics["differentiable_path"] == v12.DIFFERENTIABLE_PATH
    assert "candidate_argmax_selection" in output.diagnostics["nondifferentiable_path"]


def test_declared_prior_changes_support() -> None:
    """P1 defect B: two legal normalised priors must not produce identical output."""
    model = build_model()
    state = model.build_state(BATCH)
    head = 0.95
    tail = (1.0 - head) / (K - 1)
    first = torch.tensor([[head] + [tail] * (K - 1)] * BATCH)
    second = torch.tensor([[tail, head] + [tail] * (K - 2)] * BATCH)
    assert torch.allclose(first.sum(dim=-1), torch.ones(BATCH), atol=1e-6)
    assert torch.allclose(second.sum(dim=-1), torch.ones(BATCH), atol=1e-6)

    out_first = model.step(make_input(prior=first), state)
    out_second = model.step(make_input(prior=second), state)

    diff = (out_first.q - out_second.q).abs().max().item()
    assert diff > 1e-6, "the declared prior did not change the support weights"
    assert out_first.diagnostics["prior"]["source"] == "declared"
    assert torch.allclose(out_first.diagnostics["prior"]["mass"], torch.ones(BATCH), atol=1e-6)
    # the prior also reaches the variational objective that decides acceptance
    assert out_first.log_p0[0, 0] > out_first.log_p0[0, 1]
    assert out_second.log_p0[0, 1] > out_second.log_p0[0, 0]


def test_accept_rule_never_increases_objective() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    output = model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state)

    total = 0
    accepted = 0
    for entry in output.rounds:
        keep = entry["accepted"]
        if bool(keep.any()):
            before = entry["energy_before"][keep]
            after = entry["energy_after"][keep]
            assert bool((after <= before).all()), (
                "an accepted round increased the complete objective"
            )
        accepted += int(keep.sum().item())
        total += keep.numel()
    assert output.diagnostics["accepted_rounds"].shape == (BATCH,)
    assert total == BATCH * ROUNDS
    assert 0 <= accepted <= total


def test_duplicate_measurement_id_adds_no_evidence() -> None:
    shared = tuple(("cam-identical", "cam-identical") for _ in range(BATCH))
    measurement = FakeMeasurement()
    model = build_model(measurement=measurement)
    state = model.build_state(BATCH)
    output = model.step(
        make_input(prior=torch.full((BATCH, K), 1.0 / K), measurement_ids=shared), state
    )

    assert output.diagnostics["duplicates_dropped"] == (1,) * BATCH
    assert len(output.diagnostics["meas_ids"][0]) == len(CAMERAS) - 1
    masks = measurement.last_masks
    assert masks is not None
    per_camera = output.diagnostics["per_camera_query_count"][0]
    assert int(masks[0, 0, per_camera:].sum()) == 0, "the repeated ID still added evidence"


def test_reset_clears_selected_rows_only() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    output = model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state)
    before = output.state
    reset = model.reset(torch.tensor([True, False]), before)

    assert reset.version == before.version + 1
    assert reset.episode_ids[0] == "" and reset.episode_ids[1] == before.episode_ids[1]
    assert torch.equal(reset.slot_hidden[0], torch.zeros_like(reset.slot_hidden[0]))
    assert torch.equal(reset.existence[0], torch.zeros_like(reset.existence[0]))
    assert torch.equal(reset.slot_hidden[1], before.slot_hidden[1])
    assert torch.equal(reset.existence[1], before.existence[1])
    history = reset.geometry[0]
    assert isinstance(history, v12.V12RowHistory)
    assert history.entries == () and history.tokens is None and history.attention is None
    assert len(reset.geometry[1].entries) == 1
    no_op = model.reset(torch.tensor([False, False]), before)
    assert no_op.version == before.version
    assert torch.equal(no_op.slot_hidden, before.slot_hidden)


def test_save_resume_matches_after_same_suffix() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    prior = torch.full((BATCH, K), 1.0 / K)
    first = model.step(make_input(filled=1.0, prior=prior), state)

    buffer = io.BytesIO()
    torch.save(model.snapshot(first.state), buffer)
    buffer.seek(0)
    restored = model.restore(torch.load(buffer, weights_only=False))

    original_next = model.step(make_input(filled=1.5, prior=prior), first.state)
    restored_next = model.step(make_input(filled=1.5, prior=prior), restored)

    assert torch.equal(original_next.tokens, restored_next.tokens)
    assert torch.equal(original_next.mean, restored_next.mean)
    max_diff = (original_next.q - restored_next.q).abs().max().item()
    assert max_diff <= 1e-6, f"resumed support differs by {max_diff}"


def test_gradients_reach_backbone_corrector_and_readout() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    output = model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state)

    loss = (
        output.tokens.abs().mean() + (output.q * output.energies).sum() + output.mean.abs().mean()
    )
    assert loss.requires_grad
    report = model.gradient_report(loss)

    for role in ("encoder", "decoder", "readout", "corrector", "predictor", "proposal"):
        assert report[role]["tensors"] > 0, f"{role} has no parameter in the graph"
        assert report[role]["l1"] > 0.0, f"{role} received no gradient"
        assert report[role]["finite"] == 1.0
    assert report["encoder"]["l1"] > 0.0, "the new encoder must not be detached before the loss"


def test_illegal_priors_raise() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    with pytest.raises(v12.N02V12InferenceError):
        model.step(make_input(prior=torch.zeros(BATCH, K)), state)
    with pytest.raises(v12.N02V12InferenceError):
        model.step(make_input(prior=torch.tensor([[-1.0, 0.0, 1.0, 1.0]] * BATCH)), state)
    with pytest.raises(v12.N02V12InferenceError):
        model.step(make_input(prior=torch.full((BATCH, K - 1), 1.0 / (K - 1))), state)
    with pytest.raises(v12.N02V12InferenceError):
        model.step(make_input(prior=torch.tensor([[float("nan")] * K] * BATCH)), state)


def test_missing_verdict_is_explicit() -> None:
    model = build_model()
    state = model.build_state(BATCH)

    class Malformed(FakeMeasurement):
        def absolute_check(self, **_: object):
            return NoVerdictCheck()

    state = model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state).state
    model.inference.measurement = Malformed()
    with pytest.raises(v12.N02V12InferenceError):
        model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state)

    class UnknownStatus(FakeMeasurement):
        def absolute_check(self, **_: object):
            return FakeCheck(
                status="maybe_fine", valid=True, statistics={}, calibration_version="x"
            )

    model.inference.measurement = UnknownStatus()
    with pytest.raises(v12.N02V12InferenceError):
        model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state)


def test_nonfinite_prediction_is_reported() -> None:
    class Broken(FakeMeasurement):
        def energy(self, *, mean, variance, masks, target):
            return torch.full(mean.shape[:2], float("nan"), device=mean.device)

    model = build_model(measurement=Broken())
    state = model.build_state(BATCH)
    with pytest.raises(v12.N02V12NonFiniteError):
        model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), state)


def test_rebuild_recomputes_real_state_and_cache() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    prior = torch.full((BATCH, K), 1.0 / K)
    first = model.step(make_input(filled=1.0, prior=prior, frame_id="7"), state)
    second = model.step(make_input(filled=2.0, prior=prior, frame_id="8"), first.state)
    assert len(second.state.geometry[0].entries) == 2

    dropped_ids = second.diagnostics["meas_ids"][0]
    pruned = model.drop_measurements(second.state, dropped_ids)
    assert len(pruned.geometry[0].entries) == 1
    assert pruned.geometry[0].tokens is None, "the readout cache must be invalidated with the state"

    rebuilt, info = model.rebuild(make_input(filled=2.0, prior=prior, frame_id="8"), pruned)
    assert info["recomputed"] is True
    assert rebuilt.geometry[0].tokens is not None, (
        "the readout cache must be rewritten from real tensors"
    )
    assert not torch.equal(rebuilt.slot_hidden, pruned.slot_hidden)


def test_episode_change_only_clears_that_row() -> None:
    model = build_model()
    state = model.build_state(BATCH)
    prior = torch.full((BATCH, K), 1.0 / K)
    first = model.step(make_input(prior=prior), state)

    inp = make_input(prior=prior)
    inp.observation.episode_ids = ("episode-2", "episode-1")
    inp.observation.anchor_ids = ("anchor-2", "anchor-1")
    second = model.step(inp, first.state)
    assert len(second.state.geometry[0].entries) == 1  # new episode: history starts over
    assert len(second.state.geometry[1].entries) == 2  # same episode: history accumulates


def test_p1_state_is_rejected_by_the_v12_path() -> None:
    model = build_model()
    p1_state = SequenceState(
        episode_ids=("episode-1", "episode-1"),
        slot_hidden=torch.zeros(2, 4),
        existence=torch.zeros(2, 4),
        geometry=torch.zeros(2, 4),
    )
    with pytest.raises(v12.N02V12InferenceError):
        model.step(make_input(prior=torch.full((BATCH, K), 1.0 / K)), p1_state)


def test_locked_absolute_check_never_yields_none_string() -> None:
    score = MeasurementScore(
        relative_terms={},
        raw_nll_or_energy=torch.tensor(1.5),
        local_stats={},
        kind="proper_nll",
        per_valid_scalar=torch.tensor(0.5),
    )
    checked = locked_absolute_check(score=score, reason="no calibration set")
    assert isinstance(checked, MeasurementScore)
    assert isinstance(checked, LockedAbsoluteCheck)
    assert checked.kind == "proper_nll"
    assert checked.valid is False
    assert checked.verdict == "invalid" and checked.verdict != "None"
    assert float(checked.raw_nll_or_energy) == 1.5


def test_check_statistics_follow_the_result() -> None:
    check = FakeCheck(
        status="model_inadequate",
        valid=True,
        statistics={"threshold": 3.0},
        calibration_version="cal-7",
    )
    score = MeasurementScore(
        relative_terms={},
        raw_nll_or_energy=torch.tensor(0.25),
        local_stats={},
        kind="proper_nll",
        per_valid_scalar=torch.tensor(0.25),
    )
    checked = LockedAbsoluteCheck(score, check)
    assert checked.status == "model_inadequate"
    assert checked.verdict == "model_inadequate"
    assert checked.valid is True
    assert checked.calibration_version == "cal-7"
    assert checked.as_dict()["statistics"] == {"threshold": 3.0}


# ---------------------------------------------------------------------------------------
# real-module integration (runs once the parallel-owned modules and a GPU are present)
# ---------------------------------------------------------------------------------------


def _real_modules_available() -> bool:
    try:
        from owac.frameworks.n02_full import (  # noqa: F401
            checks_v12,
            decoder_v12,
            encoder_v12,
            readout_v12,
        )
    except Exception:  # noqa: BLE001 - not landed yet
        return False
    return True


@pytest.mark.skipif(not _real_modules_available(), reason="v12 modules are still being written")
def test_real_encoder_decoder_readout_loop_forward() -> None:
    """One real forward through the frozen modules on CPU-sized input (no dataset required)."""
    model = _build_real_model()
    values = torch.rand((1, 3, 64, 64))
    frames = {
        camera: MeasurementFrame(
            values=values,
            valid_mask=torch.ones(1, 64, 64, dtype=torch.bool),
            camera_id=camera,
            frame_ids=("0",),
        )
        for camera in _real_cameras()
    }
    inp = v12.V12InferenceInput(
        observation=ObservationBundle(
            frames=frames,
            context=EmbodimentContext(),
            episode_ids=("real-episode",),
            anchor_ids=("real-anchor",),
        ),
        declared_prior=torch.full((1, v12.DEFAULT_K), 1.0 / v12.DEFAULT_K),
        frame_index=torch.tensor([0]),
    )
    state = model.build_state(1)
    output = model.step(inp, state)
    assert bool(torch.isfinite(output.tokens).all())
    assert bool(torch.isfinite(output.energies).all())
    assert torch.allclose(output.q.sum(dim=-1), torch.ones(1), atol=1e-5)


def _real_cameras() -> tuple[str, ...]:
    return ("camera_0", "camera_1")


def _build_real_model():
    from owac.frameworks.n02_full.model_v12 import build_n02_v12_model

    return build_n02_v12_model(
        {
            "camera_ids": _real_cameras(),
            "d_model": 64,
            "n_slots": 4,
            "slot_dim": 64,
            "rounds": 2,
            "query_stride": 16,
            "strict_check": True,
        }
    )
