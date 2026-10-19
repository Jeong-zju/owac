"""Read-only diagnosis of frozen N04 weights; no training or acceptance changes."""

import hashlib
import json
import statistics
from pathlib import Path

import torch

from owac.frameworks.n04.ego import IdentityPropagator
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import (
    build_context,
    build_encoder,
    config_from_json,
    load_checkpoint,
    make_synthetic_batches,
)

ROOT = Path("research/reviews/2026-10-13_n04_mechanism_diagnosis")
WEIGHTS = Path("outputs/n04-event-semantics-20261013/candidate")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_alignment_probe():
    scene = SceneConfig(noise_sigma=0, object_colors=((0.8, 0.2, 0.1),))
    generator = SyntheticSequenceGenerator(scene)
    ys, xs = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")

    def centroids(mask):
        area = mask.sum((1, 2, 3))
        return torch.stack(
            ((mask * xs).sum((1, 2, 3)) / area, (mask * ys).sum((1, 2, 3)) / area), -1
        ).tolist()

    rows = []
    for batch in (1, 2, 4):
        base = torch.tensor([[[56.0, 56.0]]]).expand(batch, -1, -1).clone()
        motion = torch.tensor([[12.0, 0], [-12.0, 0], [0, 12.0], [0, -12.0]])[:batch]
        frame = generator.frame_batch(
            camera_translation=torch.zeros(batch, 2),
            camera_rotation=torch.zeros(batch),
            object_translation=motion,
            illumination=torch.ones(batch),
            base_object_centers=base,
            base_camera_offset=torch.zeros(batch, 2),
        )
        rendered = (frame.oracle_clean[:, :1] > 0.3).float()
        label = (frame.object_mask > 0.5).float()
        expected = (motion + 32).tolist()
        rows.append(
            {
                "batch": batch,
                "broadcast_centers_shape": list((base + motion).shape),
                "expected_image_centroids": expected,
                "actual_image_centroids": centroids(rendered),
                "label_centroids": centroids(label),
                "rendered_pixels": rendered.sum((1, 2, 3)).tolist(),
                "label_pixels": label.sum((1, 2, 3)).tolist(),
                "phantom_label_pixels": ((label - rendered) > 0).sum((1, 2, 3)).tolist(),
                "render_label_equal": torch.equal(rendered, label),
                "render_matches_requested_motion": centroids(rendered) == expected,
            }
        )
    assert rows[0]["render_label_equal"] and rows[0]["render_matches_requested_motion"]
    assert all(not row["render_label_equal"] for row in rows[1:])
    return rows


@torch.no_grad()
def static_probe(cfg, seed, mode):
    checkpoint = WEIGHTS / f"train-{seed}" / "checkpoint.pt"
    before = digest(checkpoint)
    encoder = build_encoder(cfg)
    load_checkpoint(checkpoint, encoder=encoder)
    encoder.eval()
    if mode != "learned":
        encoder.propagator = IdentityPropagator()
    hooks = []
    if mode == "oracle_zero_ego_without_block_correction":
        for block in encoder.blocks:
            hooks.append(block.register_forward_hook(lambda mod, args, output: args[0]))
    scene = SceneConfig(
        camera_translation=(0, 0),
        camera_rotation=(0, 0),
        object_velocity=(0, 0),
        object_colors=((0.4, 0.2, 0.1),),
        noise_sigma=0,
        seed=31000,
    )
    sequence = SyntheticSequenceGenerator(scene).sequence(
        length=32, batch=1, condition="camera_only", seed=31001
    )
    assert all(torch.equal(f.measured, sequence[0].measured) for f in sequence)
    assert all(f.saturated_mask.sum() == 0 for f in sequence)
    state = encoder.init_state(1)[cfg.camera_ids[0]]
    previous = None
    rows = []
    for index, frame in enumerate(sequence):
        out = encoder.step(
            frame.measured,
            build_context(frame, previous),
            state,
            saturation=frame.saturated_mask,
            measurement_valid=frame.valid_mask,
        )
        state, previous = out.state, frame
        if index >= 8:
            rows.append(
                {
                    "feature_innovation_l1": out.innovation.residual.abs().mean().item(),
                    "instant_external_l1": out.innovation.external_evidence.abs().mean().item(),
                    "persistent_change_l1": state.change[0].abs().mean().item(),
                    "external_gate_mean": out.innovation.gate_change.mean().item(),
                    "event_positive_fraction": (out.innovation.gate_change >= 0.5)
                    .float()
                    .mean()
                    .item(),
                    "stable_to_measurement_l1": (
                        state.stable[0] - encoder.encode_measurement(frame.measured)
                    )
                    .abs()
                    .mean()
                    .item(),
                }
            )
    for hook in hooks:
        hook.remove()
    assert digest(checkpoint) == before
    return {
        "initialization": seed,
        "mode": mode,
        "checkpoint_sha256": before,
        "decay": encoder.change_decay.sigmoid().item(),
        "write_scale": encoder.change_write.tanh().item(),
        "mean_frames_8_to_31": {key: mean(r[key] for r in rows) for key in rows[0]},
    }


def training_support(cfg, recipe):
    sequences = make_synthetic_batches(cfg=cfg, train_cfg=recipe, steps=9, seed=0)
    rows = []
    for sequence in sequences:
        positive = usable = changed = abandoned = overlap_frames = 0.0
        for previous, frame in zip(sequence[1:-1], sequence[2:], strict=True):
            moving = torch.as_tensor(frame.metadata["object_moving"])[:, None, None, None]
            target = frame.object_mask * moving
            valid = frame.valid_mask * (1 - frame.saturated_mask)
            positive += target.sum().item()
            usable += (target * valid).sum().item()
            clean_difference = (frame.oracle_clean - previous.oracle_clean).abs().amax(
                1, keepdim=True
            ) > 1e-6
            if frame.condition == "object_only":
                changed += (target * clean_difference).sum().item()
                abandoned += ((1 - frame.object_mask) * clean_difference).sum().item()
            illumination_changed = (
                frame.true_illumination - previous.true_illumination
            ).abs() > 1e-6
            overlap_frames += (moving.flatten() & illumination_changed).sum().item()
        rows.append(
            {
                "condition": sequence[0].condition,
                "moving_object_pixel_sum": positive,
                "moving_object_usable_pixel_sum": usable,
                "usable_positive_fraction": usable / positive if positive else None,
                "simultaneous_motion_illumination_sample_frames": overlap_frames,
                "object_only_positive_pixels_with_clean_frame_difference": changed
                if sequence[0].condition == "object_only"
                else None,
                "object_only_changed_background_pixels": abandoned
                if sequence[0].condition == "object_only"
                else None,
            }
        )
    return {
        "scope": "first complete nine-condition cycle of actual training data, frames 2..11",
        "conditions": rows,
    }


def inspect_existing_results():
    by_condition = {}
    ratios = []
    for init in (0, 1, 2):
        for seed in (20000, 20001, 20002):
            report = json.loads((WEIGHTS / f"train-{init}" / f"eval-{seed}.json").read_text())
            conditions = {c["condition"]: c for c in report["conditions"]}
            for name in ("camera_only", "object_only", "both", "illumination"):
                c = conditions[name]
                by_condition.setdefault(name, []).append(
                    {
                        k: c[k]
                        for k in (
                            "change_at_object",
                            "change_off_object",
                            "change_response",
                            "prediction_residual",
                        )
                    }
                )
            response = {
                name: mean(c["change_response"] for c in conditions[name]["trajectory"][2:8])
                for name in ("camera_only", "object_only", "illumination")
            }
            ratios.append(
                {
                    "initialization": init,
                    "eval_seed": seed,
                    "camera_over_object_frames_2_to_7": response["camera_only"]
                    / response["object_only"],
                    "illumination_over_object_frames_2_to_7": response["illumination"]
                    / response["object_only"],
                }
            )
    return {
        "condition_means": {
            name: {key: mean(row[key] for row in values) for key in values[0]}
            for name, values in by_condition.items()
        },
        "matched_moving_window_ratios": ratios,
        "note": "Post-hoc diagnostic only; original frozen acceptance numbers and thresholds "
        "remain unchanged. Batch-label defect invalidates their physical interpretation.",
    }


def main():
    torch.set_num_threads(4)
    cfg, recipe = config_from_json(Path("configs/encoders/n04/tiny.json"))
    result = {
        "protocol": {
            "no_training": True,
            "reviewed_code_commit": "57138c1bac916f3c65644de73b1855c78062a62e",
            "source_sha256": {
                str(path): digest(path)
                for path in (
                    Path("src/owac/frameworks/n04/synthetic.py"),
                    Path("src/owac/frameworks/n04/model.py"),
                    Path("src/owac/frameworks/n04/innovation.py"),
                    Path("src/owac/training/n04.py"),
                    Path("src/owac/evaluation/n04.py"),
                )
            },
            "new_static_scene_seeds": [31000, 31001],
            "frames": 32,
            "average_frames": [8, 31],
            "device": "cpu",
            "modes": ["learned", "oracle_zero_ego", "oracle_zero_ego_without_block_correction"],
            "caveat": "Oracle identity also changes propagation uncertainty; block bypass "
            "isolates correction under the same oracle identity. No task benefit is claimed. "
            "The inspected weights were trained with corrupted multi-sample object labels.",
        },
        "batch_alignment": batch_alignment_probe(),
        "training_support": training_support(cfg, recipe),
        "existing_results": inspect_existing_results(),
        "static_control": [
            static_probe(cfg, seed, mode)
            for seed in (0, 1, 2)
            for mode in ("learned", "oracle_zero_ego", "oracle_zero_ego_without_block_correction")
        ],
    }
    (ROOT / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
