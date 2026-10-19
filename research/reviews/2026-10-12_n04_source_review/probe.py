"""Read-only N04 review probes; do not change training or benchmark code."""

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from owac.frameworks.n04.ego import EgoMotionContext
from owac.frameworks.n04.innovation import estimate_low_frequency_gain
from owac.frameworks.n04.model import N04Encoder
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import (
    _change_loss,
    _ego_warp_loss,
    _future_loss,
    build_context,
    change_gate_loss,
    config_from_json,
    load_checkpoint,
    location_probe_loss,
)

torch.set_num_threads(4)
torch.manual_seed(123)
cfg, _ = config_from_json(Path("configs/encoders/n04/tiny.json"))
checkpoint = Path("outputs/n04-acceptance/candidate/checkpoint.pt")
encoder = N04Encoder(cfg)
predictor = nn.Conv2d(cfg.model.width, cfg.channels, 3, padding=1)
payload = load_checkpoint(checkpoint, encoder=encoder, predictor_head=predictor)
report = {
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "checkpoint_updates": payload["step"],
    "scope": "source review and controlled probes; no retraining or benchmark mutation",
}

# Follow all four actual training objectives on a short moving-object sequence.
scene = SceneConfig(object_velocity=(1.0, 1.0), noise_sigma=0.0, seed=123)
sequence = SyntheticSequenceGenerator(scene).sequence(
    length=5, batch=1, condition="object_only", seed=124
)
state = encoder.init_state(1)[cfg.camera_ids[0]]
previous = None
losses = []
for t, frame in enumerate(sequence):
    output = encoder.step(
        frame.measured,
        build_context(frame, previous),
        state,
        measurement_valid=frame.valid_mask,
        saturation=frame.saturated_mask,
    )
    if t >= 2:
        weights = cfg.model.loss_weights
        losses.append(
            weights["location"]
            * location_probe_loss(encoder.location_probe, output.state.stable, frame)
            + weights["change"]
            * (
                _change_loss(output.state.change, frame)
                + change_gate_loss(output.innovation.gate_change, frame)
            )
            + weights["ego_warp"] * _ego_warp_loss(output.state.forward_flow, frame, cfg)
            + weights["future_measurement"]
            * _future_loss(predictor, output.state.prediction, previous, frame)
        )
    state, previous = output.state, frame
sum(losses).backward()

sigma_grad = encoder.propagator.head.weight.grad[2].abs().sum().item()
stability_grad = encoder.decomposer.head.weight.grad[4].abs().sum().item()
head = encoder.decomposer.head
report["disconnected_heads"] = {
    "flow_gradient_l1": encoder.propagator.head.weight.grad[:2].abs().sum().item(),
    "uncertainty_gradient_l1": sigma_grad,
    "uncertainty_checkpoint_weight_l1": encoder.propagator.head.weight[2].abs().sum().item(),
    "uncertainty_checkpoint_bias": encoder.propagator.head.bias[2].item(),
    "stability_gradient_l1": stability_grad,
    "stability_checkpoint_weight_l1": head.weight[4].abs().sum().item(),
    "ego_unknown_weights_exactly_equal": torch.equal(head.weight[0], head.weight[3]),
    "ego_unknown_bias_exactly_equal": torch.equal(head.bias[0], head.bias[3]),
    "ego_unknown_mix_max_diff": (output.innovation.mix[:, 0] - output.innovation.mix[:, 3])
    .abs()
    .max()
    .item(),
    "valid_uncertainty_unique_values": torch.unique(
        state.uncertainty[state.warp_validity > 0.999]
    ).tolist(),
}
assert sigma_grad == 0 and stability_grad == 0
assert torch.equal(head.weight[0], head.weight[3])
assert report["disconnected_heads"]["ego_unknown_mix_max_diff"] == 0

# Actual trained stem: scale unsaturated RAW by a known factor.
raw = torch.rand(1, 3, 64, 64) * 0.25 + 0.05
bright = raw * 1.5
with torch.no_grad():
    encoded_previous = encoder.stem(raw)
    encoded_current = encoder.stem(bright)
    gain, _ = estimate_low_frequency_gain(
        encoded_current, encoded_previous, torch.ones(1, 1, 64, 64)
    )
    linear_gain, _ = estimate_low_frequency_gain(bright, raw, torch.ones(1, 1, 64, 64))
report["illumination_domain"] = {
    "true_raw_gain": 1.5,
    "saturated_pixels": int((bright >= 1).sum()),
    "linear_domain_gain_median": linear_gain.median().item(),
    "trained_feature_domain_gain_median": gain.median().item(),
    "note": "feature-domain fixture uses the actual trained stem, not a scaled-feature stub",
}

# The scalar gain excludes unreliable pixels; the local gain does not.
predicted = torch.ones(1, 3, 12, 12)
current = predicted.clone()
weight = torch.ones(1, 1, 12, 12)
weight[..., 6:] = 0
before, _ = estimate_low_frequency_gain(current, predicted, weight)
current[..., 6:] = 10
after, _ = estimate_low_frequency_gain(current, predicted, weight)
scalar, _ = estimate_low_frequency_gain(current, predicted, weight, low_frequency=False)
leak = ((after - before).abs() * weight).max().item()
report["illumination_mask_leak"] = {
    "scalar_gain": scalar.median().item(),
    "max_gain_change_on_reliable_pixels": leak,
    "unreliable_region_input_change": "1 -> 10, weight remains zero",
}
assert leak > 1

# Controlled feature-domain fixture: exact self-motion warp with no new residual.
small_cfg = replace(
    cfg,
    model=replace(cfg.model, width=1, blocks=1, scales=1, hidden=8, motion_dim=7),
)
small = N04Encoder(small_cfg)
with torch.no_grad():
    small.propagator.head.bias[0] = cfg.geometry.max_flow * torch.atanh(
        torch.tensor(1.0 / cfg.geometry.max_flow)
    )
context = EgoMotionContext(
    proprio_delta=torch.zeros(1, 3),
    executed_action=torch.zeros(1, 3),
    dt=torch.ones(1),
)
initial = small.init_state(1)[small_cfg.camera_ids[0]]
initial.stable[..., 10, 10] = 1
initial.change[..., 10, 10] = 1
initial.history_valid = torch.ones(1, dtype=torch.bool)
initial.warp_validity = torch.ones(1, 1, 64, 64)
pred, valid, uncertainty, propagation = small.predict_before_observe(initial, context)
corrected, innovation = small.correct_after_observe(initial, pred, pred, valid, uncertainty)
report["change_coordinates"] = {
    "previous_stable_peak_flat_index": initial.stable.flatten().argmax().item(),
    "predicted_stable_peak_flat_index": pred.flatten().argmax().item(),
    "updated_change_peak_flat_index": corrected.change.flatten().argmax().item(),
    "residual_l1": innovation.residual.abs().sum().item(),
    "expected_change_peak_flat_index": pred.flatten().argmax().item(),
    "note": (
        "1px exact self-motion; old change should move with the same feature coordinate transform"
    ),
}
assert corrected.change.flatten().argmax() != pred.flatten().argmax()

# Uniform moving object: the overlapping interior has zero residual regardless of gate.
interior_model = N04Encoder(small_cfg)
with torch.no_grad():
    interior_model.decomposer.head.bias[1] = 20
prev = torch.zeros(1, 1, 64, 64)
prev[..., 20:30, 20:30] = 1
measured = torch.zeros_like(prev)
measured[..., 20:30, 21:31] = 1
overlap = (prev > 0) & (measured > 0)
initial = interior_model.init_state(1)[small_cfg.camera_ids[0]]
initial.stable = prev
initial.warp_validity = torch.ones_like(prev)
initial.history_valid = torch.ones(1, dtype=torch.bool)
corrected, innovation = interior_model.correct_after_observe(
    initial, prev, measured, torch.ones_like(prev), torch.zeros_like(prev)
)
frame = SimpleNamespace(
    measured=measured,
    object_mask=measured,
    true_object_translation=torch.tensor([[1.0, 0.0]]),
    condition="object_only",
    metadata={"object_moving": [True]},
    valid_mask=torch.ones_like(prev),
    saturated_mask=torch.zeros_like(prev),
)
magnitude = corrected.change.square().mean(1, keepdim=True).add(1e-8).sqrt()
interior_loss = (magnitude - measured).abs()[overlap].mean()
interior_loss.backward()
report["change_supervision_conflict"] = {
    "moving_object_pixels": int(measured.sum()),
    "zero_residual_interior_pixels": int(overlap.sum()),
    "gate_interior_min": innovation.gate_change[overlap].min().item(),
    "change_interior_abs_max": corrected.change[overlap].abs().max().item(),
    "change_magnitude_training_loss": _change_loss(corrected.change, frame).item(),
    "interior_loss": interior_loss.item(),
    "interior_loss_write_gradient": interior_model.change_write.grad.abs().item(),
    "interior_loss_gate_head_gradient_l1": interior_model.decomposer.head.weight.grad.abs()
    .sum()
    .item(),
    "note": "diagnostic encoded-feature fixture; not a re-estimation of the reported FNR",
}
assert report["change_supervision_conflict"]["change_interior_abs_max"] == 0
assert interior_model.change_write.grad.item() == 0

destination = Path("outputs/n04-source-review/evidence.json")
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
