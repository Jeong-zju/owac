"""Read-only final checkpoint audit and aggregation; never select or retrain a model."""

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from owac.frameworks.n04.geometry import backward_grid_from_translation
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
    decomposition_loss,
    load_checkpoint,
    location_probe_loss,
    propagation_uncertainty_loss,
)

ARCHIVE = Path("research/experiments/results/2026-10-13_n04_six_fixes")
ROOT = Path("outputs/n04-six-fixes-20261013")


def save(name, value):
    (ARCHIVE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def mean(values):
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def aggregate(rows):
    def condition(name, key):
        return mean(
            c.get(key, c["extra"].get(key))
            for r in rows
            for c in r["conditions"]
            if c["condition"] == name
        )

    return {
        "evaluation_seeds": [r["seed"] for r in rows],
        "location_mean_px": mean(r["location_mean_px"] for r in rows),
        "camera_vs_object_change": mean(r["contrast"]["camera_vs_object_change"] for r in rows),
        "illumination_response_ratio": mean(
            r["contrast"]["illumination_response_ratio"] for r in rows
        ),
        "camera_only_fpr": condition("camera_only", "external_false_positive_rate"),
        "illumination_fpr": condition("illumination", "external_false_positive_rate"),
        "object_only_fnr": condition("object_only", "external_false_negative_rate"),
        "camera_only_warp_error": condition("camera_only", "warp_error"),
        "condition_locations_px": {
            c["condition"]: condition(c["condition"], "location_error")
            for c in rows[0]["conditions"]
        },
        "mean_condition_latency_p50_ms": mean(
            c["latency_p50_ms"] for r in rows for c in r["conditions"]
        ),
        "mean_condition_latency_p95_ms": mean(
            c["latency_p95_ms"] for r in rows for c in r["conditions"]
        ),
        "state_bytes_batch2": rows[0]["conditions"][0]["state_bytes"],
    }


def main():
    torch.set_num_threads(4)
    torch.manual_seed(123)
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    assert manifest["complete"]
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
        for p, digest in manifest["source_sha256"].items()
    )
    groups = defaultdict(list)
    groups["before_candidate"] = json.loads((ARCHIVE / "before-matrix.json").read_text())
    matrix = json.loads((ARCHIVE / "after-matrix.json").read_text())
    assert len(matrix) == 36
    for row in matrix:
        checkpoint = ROOT / "after" / row["name"] / "checkpoint.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
        groups[row["name"]].append(row)
    assert all(len(rows) == 3 for rows in groups.values())
    save("summary.json", {name: aggregate(rows) for name, rows in groups.items()})

    cfg, _ = config_from_json(Path("configs/encoders/n04/tiny.json"))
    encoder = N04Encoder(cfg)
    predictor = nn.Conv2d(cfg.model.width, cfg.channels, 3, padding=1)
    checkpoint = ROOT / "after/candidate/checkpoint.pt"
    payload = load_checkpoint(checkpoint, encoder=encoder, predictor_head=predictor)
    assert payload["mechanism_revision"] == "n04-six-fixes-v1" and payload["step"] == 200
    scene = SceneConfig(object_velocity=(1, 1), noise_sigma=0, seed=123)
    sequence = SyntheticSequenceGenerator(scene).sequence(
        length=5, batch=1, condition="object_only", seed=124
    )
    state = encoder.init_state(1)[cfg.camera_ids[0]]
    previous, losses = None, []
    for t, frame in enumerate(sequence):
        out = encoder.step(
            frame.measured,
            build_context(frame, previous),
            state,
            measurement_valid=frame.valid_mask,
            saturation=frame.saturated_mask,
        )
        if t >= 2:
            w = cfg.model.loss_weights
            inv = out.innovation
            flow = out.state.forward_flow
            losses.append(
                w["location"] * location_probe_loss(encoder.location_probe, out.state.stable, frame)
                + w["change"]
                * (
                    _change_loss(inv.external_evidence, frame, residual=inv.residual)
                    + change_gate_loss(inv.gate_change, frame)
                    + 0.25 * decomposition_loss(inv, flow, frame, previous, cfg)
                )
                + w["ego_warp"]
                * (
                    _ego_warp_loss(flow, frame, cfg)
                    + propagation_uncertainty_loss(out.state.uncertainty, flow, inv, frame, cfg)
                )
                + w["future_measurement"]
                * _future_loss(predictor, out.state.prediction, previous, frame)
            )
        state, previous = out.state, frame
    sum(losses).backward()
    head = encoder.decomposer.head
    sigma_grad = encoder.propagator.head.weight.grad[2].abs().sum().item()
    stability_grad = head.weight.grad[4].abs().sum().item()
    assert sigma_grad > 0 and stability_grad > 0
    assert not torch.equal(head.weight[0], head.weight[3])
    valid_sigma = state.uncertainty[state.warp_validity > 0.999].detach()
    with torch.no_grad():
        raw = torch.rand(1, 3, 64, 64) * 0.25 + 0.05
        inv = encoder.decomposer(
            current=encoder.stem(raw * 1.5),
            predicted=encoder.stem(raw),
            validity=torch.ones(1, 1, 64, 64),
            uncertainty=torch.zeros(1, 1, 64, 64),
            linear_current=raw * 1.5,
            linear_predicted=raw,
            linear_validity=torch.ones(1, 1, 64, 64),
        )
        gain = inv.illumination_gain.median().item()
        previous_raw = torch.ones(1, 3, 12, 12)
        current = previous_raw.clone()
        mask = torch.ones(1, 1, 12, 12)
        mask[..., 6:] = 0
        initial, _ = estimate_low_frequency_gain(current, previous_raw, mask)
        current[..., 6:] = 10
        final, _ = estimate_low_frequency_gain(current, previous_raw, mask)
        leak = ((final - initial).abs() * mask).max().item()
        fixture = encoder.init_state(1)[cfg.camera_ids[0]]
        stable = fixture.stable[0]
        change = fixture.change[0]
        stable[..., 10, 10] = 1
        change[..., 10, 10] = 1
        grid = backward_grid_from_translation(encoder.spec, 1, 0, batch=1)
        predicted = torch.nn.functional.grid_sample(stable, grid, align_corners=False)
        corrected, _ = encoder.correct_after_observe(
            fixture,
            predicted,
            predicted,
            torch.ones(1, 1, 64, 64),
            torch.zeros(1, 1, 64, 64),
            backward_grid=grid,
        )
        change_peak = corrected.change[0][0, 0].argmax().item()
    assert abs(gain - 1.5) < 1e-3 and leak == 0 and change_peak == 651
    audit = {
        "source_hashes_match_frozen_run": True,
        "all_36_evaluation_checkpoint_hashes_match": True,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_updates": payload["step"],
        "uncertainty_gradient_l1": sigma_grad,
        "stability_gradient_l1": stability_grad,
        "uncertainty_checkpoint_weight_l1": encoder.propagator.head.weight[2].abs().sum().item(),
        "stability_checkpoint_weight_l1": head.weight[4].abs().sum().item(),
        "ego_unknown_weights_exactly_equal": torch.equal(head.weight[0], head.weight[3]),
        "ego_unknown_mix_max_diff": (out.innovation.mix[:, 0] - out.innovation.mix[:, 3])
        .abs()
        .max()
        .item(),
        "valid_uncertainty_min": valid_sigma.min().item(),
        "valid_uncertainty_max": valid_sigma.max().item(),
        "trained_decomposer_true_1_5_linear_gain_median": gain,
        "invalid_content_gain_leak_max": leak,
        "change_peak_after_1px_motion": change_peak,
        "note": "Controlled source probes and trained head learning; not mechanism/robot success.",
    }
    save("audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
