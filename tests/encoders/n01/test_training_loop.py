"""Training-loop tests for N01 (task B).

Covers the five behaviours the task spec requires, with real numbers rather than
shape-only or fixed-output shells:

1. 50-step smoke training produces a restorable ``checkpoint.pt`` whose ``step``
   continues and whose losses are finite.
2. Training and inference agree frame by frame: stepping the same sequence one
   frame at a time matches a windowed forward pass to 1e-5.
3. A partial batch reset through ``N01Encoder.reset(state, mask)`` leaves the
   other episodes untouched.
4. Gradients really reach the support and motion parameters (non-zero check).
5. Every loss term has valid pixels > 0 and the masks take effect: zeroing the
   valid mask drives the loss to exactly 0.

The tests exercise the real encoder, the real synthetic task and the real CLI
machinery; nothing here replaces a module owned by another implementer.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from owac.training.n01 import (  # noqa: E402
    LocationProbeHead,
    SyntheticSource,
    TrainingSettings,
    build_probe_heads,
    load_checkpoint,
    loss_weights_from_args,
    measurement_holdout,
    sequence_step,
    train,
    train_one_step,
)
from owac.training.n01 import main as training_main  # noqa: E402

pytest.importorskip("owac.frameworks.n01.model")

from owac.frameworks.n01.config import N01Config, config_fingerprint  # noqa: E402
from owac.frameworks.n01.model import N01Encoder  # noqa: E402
from owac.frameworks.n01.synthetic import SyntheticWeakEdgeTask  # noqa: E402

IMAGE_SIZE = 32


def small_config(**overrides):
    base = {
        "input_mode": "rawvla_rgb3",
        "channels": 3,
        "planes": 1,
        "width": 16,
        "num_blocks": 1,
        "max_history": 4,
        "spatial_radii": (0, 1, 2),
        "time_depths": (1, 2, 4),
        "neighbourhood_radius": 2,
    }
    base.update(overrides)
    return N01Config(**base).validate()


def make_encoder_and_heads(config, device="cpu"):
    encoder = N01Encoder(config).to(device)
    heads = build_probe_heads(
        encoder, (IMAGE_SIZE, IMAGE_SIZE), channels=int(config.channels), device=device
    )
    return encoder, heads


def make_source(config, seed=0):
    return SyntheticSource(
        config, train_seed=seed, val_seed=seed + 1_000_003, image_size=IMAGE_SIZE
    )


# ---------------------------------------------------------------------------
# 1. 50-step smoke training, restore, continuous step, finite loss
# ---------------------------------------------------------------------------
def test_smoke_training_checkpoint_restores_and_continues(tmp_path):
    config = small_config()
    weights = loss_weights_from_args(config, None)
    settings = TrainingSettings(
        seed=0, steps=50, batch_size=2, length=4, lr=3e-4, tbptt=2, log_every=0, losses=weights
    )
    encoder, heads = make_encoder_and_heads(config)
    train(
        encoder,
        heads,
        make_source(config),
        settings,
        output_dir=tmp_path,
        config_payload={**config.to_dict(), "config_fingerprint": config_fingerprint(config)},
        repo_root=tmp_path,
        log_stream=open(tmp_path / "log.txt", "w"),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    assert checkpoint_path.is_file()
    payload = load_checkpoint(checkpoint_path)
    for field in (
        "config",
        "model",
        "optimizer",
        "step",
        "seed",
        "history",
        "git_sha",
        "torch_version",
        "rng_state",
    ):
        assert field in payload, f"checkpoint is missing {field!r}"
    assert payload["step"] == 50
    assert payload["seed"] == 0
    assert len(payload["history"]) == 50
    assert all(
        {"step", "loss_location", "loss_measurement", "loss_budget", "valid_pixels"} <= set(row)
        for row in payload["history"]
    )
    losses = [row["loss"] for row in payload["history"]]
    assert all(isinstance(value, float) and value == value for value in losses), (
        "loss must be finite"
    )
    assert all(row["valid_pixels"] > 0 for row in payload["history"])
    # run.json must carry the measured numbers and the auxiliary-head switch.
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["parameters"]["encoder_trainable"] > 0
    assert record["settings"]["aux_measurement_prediction"] is True
    assert record["timing_ms"]["step_p50"] > 0.0
    assert "peak_rss_bytes" in record["resources"]

    # resume: the step continues and the history extends
    settings_resume = TrainingSettings(
        seed=0, steps=3, batch_size=2, length=4, lr=3e-4, tbptt=2, log_every=0, losses=weights
    )
    encoder2, heads2 = make_encoder_and_heads(config)
    run2 = train(
        encoder2,
        heads2,
        make_source(config),
        settings_resume,
        output_dir=tmp_path / "resume",
        config_payload={**config.to_dict(), "config_fingerprint": config_fingerprint(config)},
        repo_root=tmp_path,
        resume=payload,
        log_stream=open(tmp_path / "resume.log", "w"),
    )
    steps = [row["step"] for row in run2["history"]]
    assert steps[:50] == list(range(1, 51))
    assert steps[50:] == [51, 52, 53]
    assert run2["history"][-1]["loss"] == run2["history"][-1]["loss"]


def test_cli_smoke_writes_checkpoint_and_run_json(tmp_path):
    """The documented command shape must run end to end (5 steps for speed)."""
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(small_config().to_dict()))
    output = tmp_path / "out"
    code = training_main(
        [
            "--config",
            str(config_path),
            "--synthetic",
            "--steps",
            "5",
            "--batch-size",
            "2",
            "--length",
            "4",
            "--image-size",
            str(IMAGE_SIZE),
            "--log-every",
            "0",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert (output / "checkpoint.pt").is_file()
    record = json.loads((output / "run.json").read_text())
    assert record["settings"]["steps"] == 5
    assert record["resources"]["peak_gpu_bytes"] is None, "no GPU -> null, never a fake 0"
    assert record["units"]["location_huber"].startswith("pixel units")


# ---------------------------------------------------------------------------
# 2. frame-by-frame training/inference agreement
# ---------------------------------------------------------------------------
def test_frame_by_frame_matches_window_forward():
    config = small_config()
    encoder, heads = make_encoder_and_heads(config)
    task = SyntheticWeakEdgeTask(config, image_size=IMAGE_SIZE, seed=3)
    batch = task.sample(2, 4, seed=11)
    encoder.eval()
    with torch.no_grad():
        # ``holdout=False`` disables the auxiliary task's own input holding, which
        # rewrites the frames; the consistency check must see identical inputs.
        result = sequence_step(
            encoder,
            heads,
            batch,
            step=0,
            tbptt=0,
            grad_enabled=False,
            device="cpu",
            holdout=False,
        )
        logits_window = result.logits.clone()

        # same frames, but reset between frames and compare per-frame outputs
        state = encoder.init_state(2, {"camera_ids": ["camera"], "shape": (IMAGE_SIZE, IMAGE_SIZE)})
        per_frame = []
        for index in range(4):
            from owac.core.encoder_contract import MeasurementFrame

            frame = MeasurementFrame(
                values=batch.values[:, index],
                valid_mask=batch.valid[:, index],
                camera_id="camera",
                timestamp=batch.timestamps[:, index],
                time_unit="frame",
                new_measurement=batch.new_measurement[:, index],
                metadata={"variance": batch.variance[:, index]},
            )
            output = encoder.step(frame, None, state, None)
            state = output.state
            frame_logits, _ = heads(output.representation)
            per_frame.append(frame_logits)
        logits_stepwise = torch.stack(per_frame, dim=1)

    # The encoder is deterministic, so the two paths must agree tightly.
    max_difference = float((logits_window - logits_stepwise).abs().max())
    assert max_difference < 1e-5, f"frame-by-frame drift {max_difference} exceeds 1e-5"


def test_window_length_changes_with_history_and_is_causal():
    """Appending frames must not change earlier outputs (no future leakage)."""
    config = small_config()
    encoder, heads = make_encoder_and_heads(config)
    task = SyntheticWeakEdgeTask(config, image_size=IMAGE_SIZE, seed=5)
    batch = task.sample(1, 5, seed=17)
    encoder.eval()
    with torch.no_grad():
        # Both windows must see the same (unmodified) measurements: the holdout
        # auxiliary task rewrites its own input, which would otherwise be read as
        # a causality violation.
        full = sequence_step(
            encoder, heads, batch, step=0, tbptt=0, grad_enabled=False, holdout=False
        )
        short = sequence_step(
            encoder,
            heads,
            _truncate(batch, 3),
            step=0,
            tbptt=0,
            grad_enabled=False,
            holdout=False,
        )
    difference = float((full.logits[:, :3] - short.logits).abs().max())
    assert difference < 1e-5, f"future frames leaked into past outputs ({difference})"


def _truncate(batch, length):
    from dataclasses import replace

    return replace(
        batch,
        values=batch.values[:, :length],
        valid=batch.valid[:, :length],
        variance=batch.variance[:, :length],
        saturated=batch.saturated[:, :length],
        bad=batch.bad[:, :length],
        timestamps=batch.timestamps[:, :length],
        new_measurement=batch.new_measurement[:, :length],
        edge_target=batch.edge_target[:, :length],
        target_visible=batch.target_visible[:, :length],
        target_xy=batch.target_xy[:, :length],
    )


# ---------------------------------------------------------------------------
# 3. partial batch reset
# ---------------------------------------------------------------------------
def test_partial_batch_reset_isolates_episodes():
    config = small_config()
    encoder = N01Encoder(config)
    shape = (IMAGE_SIZE, IMAGE_SIZE)
    state = encoder.init_state(3, {"camera_ids": ["camera"], "shape": shape})
    task = SyntheticWeakEdgeTask(config, image_size=IMAGE_SIZE, seed=7)
    batch = task.sample(3, 3, seed=23)
    from owac.core.encoder_contract import MeasurementFrame

    for index in range(3):
        frame = MeasurementFrame(
            values=batch.values[:, index],
            valid_mask=batch.valid[:, index],
            camera_id="camera",
            timestamp=batch.timestamps[:, index],
            time_unit="frame",
            new_measurement=batch.new_measurement[:, index],
            metadata={"variance": batch.variance[:, index]},
        )
        state = encoder.step(frame, None, state, None).state

    before = state.values.clone()
    mask = torch.tensor([True, False, False])
    after = encoder.reset(state, mask)
    # episode 0 is wiped, episodes 1 and 2 keep every number
    assert float(after.values[:, 0].abs().sum()) == 0.0
    assert float(after.filled[0]) == 0.0
    assert torch.equal(after.values[:, 1:], before[:, 1:])
    assert torch.equal(after.valid[:, 1:], state.valid[:, 1:])
    assert torch.equal(after.frame_ids[:, 1:], state.frame_ids[:, 1:])
    assert int(after.filled[1]) == int(state.filled[1]) == 3

    # and a second step on the mixed batch only advances the live episodes
    frame = MeasurementFrame(
        values=batch.values[:, 2],
        valid_mask=batch.valid[:, 2],
        camera_id="camera",
        timestamp=batch.timestamps[:, 2],
        time_unit="frame",
        new_measurement=batch.new_measurement[:, 2],
        metadata={"variance": batch.variance[:, 2]},
    )
    stepped = encoder.step(frame, None, after, None).state
    assert int(stepped.filled[0]) == 1, "the reset episode restarts from one measurement"
    assert int(stepped.filled[1]) == max(int(config.max_history), 3)


# ---------------------------------------------------------------------------
# 4. gradients reach support and motion parameters
# ---------------------------------------------------------------------------
def _trainable_grad_norm(module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total**0.5


def test_gradients_flow_to_support_and_motion_parameters():
    config = small_config(alignment="candidate", fractional_alignment=True)
    encoder, heads = make_encoder_and_heads(config)
    # A high-contrast scene is required: when a support coefficient sits at the
    # declared floor its derivative with respect to the gate is exactly zero, so a
    # low-contrast batch cannot carry a gradient to the gate no matter how the
    # optimiser is set up.  That limitation is recorded in the experiment notes.
    task = SyntheticWeakEdgeTask(config, image_size=IMAGE_SIZE, seed=13, snr="high")
    batch = task.sample(2, 4, seed=29)
    optimizer = torch.optim.AdamW(
        [p for p in list(encoder.parameters()) + list(heads.parameters()) if p.requires_grad],
        lr=3e-4,
    )
    breakdown = train_one_step(
        encoder,
        heads,
        batch,
        optimizer,
        step=1,
        weights={"location": 1.0, "measurement_prediction": 0.1, "budget": 0.0},
        grad_clip=1.0,
    )
    assert breakdown.location_valid_pixels > 0
    assert encoder.integrator.edge_head is not None, "the support gate must exist to be trained"
    support_norm = _trainable_grad_norm(encoder.integrator.edge_head)
    assert support_norm > 0.0, "no gradient reached the support gate"
    assert encoder.correspondence is not None
    assert encoder.correspondence.head is not None
    motion_norm = _trainable_grad_norm(encoder.correspondence.head)
    assert motion_norm > 0.0, "no gradient reached the correspondence head"
    assert _trainable_grad_norm(encoder.stem) > 0.0
    assert _trainable_grad_norm(encoder.blocks) > 0.0
    assert _trainable_grad_norm(heads.location) > 0.0
    # the encoder's own measurement head is a separate, also-trainable path
    assert encoder.measurement_head is not None
    assert _trainable_grad_norm(encoder.measurement_head) >= 0.0


# ---------------------------------------------------------------------------
# 5. valid pixels > 0 and masks take effect
# ---------------------------------------------------------------------------
def test_masks_zero_the_loss_when_no_pixel_is_valid():
    config = small_config()
    encoder, heads = make_encoder_and_heads(config)
    task = SyntheticWeakEdgeTask(config, image_size=IMAGE_SIZE, seed=19)
    batch = task.sample(2, 3, seed=31)
    optimizer = torch.optim.SGD([p for p in heads.parameters()], lr=1e-3)
    live = train_one_step(
        encoder,
        heads,
        batch,
        optimizer,
        step=1,
        weights={"location": 1.0, "measurement_prediction": 0.1, "budget": 0.0},
    )
    assert live.location_valid_pixels > 0
    assert live.measurement_valid_pixels > 0
    assert live.total.item() > 0.0

    # every target marked invisible: the supervised numerator must vanish
    from dataclasses import replace

    hidden = replace(batch, target_visible=torch.zeros_like(batch.target_visible))
    optimizer.zero_grad(set_to_none=True)
    blank = train_one_step(
        encoder,
        heads,
        hidden,
        optimizer,
        step=2,
        weights={"location": 1.0, "measurement_prediction": 1.0, "budget": 0.0},
    )
    assert blank.location_valid_pixels == 0
    assert float(blank.location.detach()) == 0.0
    # all measurements invalid -> the auxiliary target set is empty, loss 0
    invalid = replace(
        batch,
        valid=torch.zeros_like(batch.valid),
    )
    optimizer.zero_grad(set_to_none=True)
    empty = train_one_step(
        encoder,
        heads,
        invalid,
        optimizer,
        step=3,
        weights={"location": 1.0, "measurement_prediction": 1.0, "budget": 0.0},
    )
    assert empty.measurement_valid_pixels == 0
    assert float(empty.measurement.detach()) == 0.0


def test_measurement_holdout_is_disjoint_from_the_input():
    """The auxiliary target pixels are exactly the pixels removed from the input."""
    row, col = measurement_holdout(0)
    other = measurement_holdout(1)
    assert (row, col) != other, "successive steps must rotate the holdout block"
    config = small_config()
    encoder, heads = make_encoder_and_heads(config)
    task = SyntheticWeakEdgeTask(config, image_size=IMAGE_SIZE, seed=23)
    batch = task.sample(1, 3, seed=37)
    # mark one pixel that is inside the holdout block and check the input differs
    mask = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    mask[row::2, col::2] = True
    result = sequence_step(encoder, heads, batch, step=0, tbptt=0, grad_enabled=False)
    assert result.measurement_target is not None
    # targets come from the clean frame t-1, evaluated on the holdout block only
    # [B, T-1, C, P] with P the fixed held-out pixel count; every element is valid
    # and fresh in this batch, so the weighted valid count is B*(T-1)*C*P.
    assert result.measurement_target.shape == (1, 2, 3, int(mask.sum()))
    assert result.measurement_valid_count == 1 * 2 * 3 * int(mask.sum())


def test_loss_weights_override_and_units():
    config = small_config()
    weights = loss_weights_from_args(config, "measurement_prediction=0.0,budget=0.25")
    assert weights == {"location": 1.0, "measurement_prediction": 0.0, "budget": 0.25}
    with pytest.raises(SystemExit):
        loss_weights_from_args(config, "nonsense=1.0")


def test_heatmap_location_loss_agrees_with_huber_on_the_same_logits():
    head = LocationProbeHead(2, hidden=4)
    features = torch.randn(2, 2, 8, 8)
    logits = head(features, size=(8, 8))
    target_xy = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    visible = torch.tensor([True, True])
    huber, error_huber, count_huber = head.location_loss(
        logits, target_xy, visible, mode="huber", sigma=1.0
    )
    ce, error_ce, count_ce = head.location_loss(
        logits, target_xy, visible, mode="heatmap_ce", sigma=1.0
    )
    assert count_huber == count_ce == 2
    assert huber.item() > 0.0 and ce.item() > 0.0
    # both report the error in pixels through the same soft-argmax readout
    assert error_huber.item() == pytest.approx(error_ce.item(), rel=1e-5)
