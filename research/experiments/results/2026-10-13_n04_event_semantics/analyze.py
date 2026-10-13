"""Aggregate the frozen comparison and audit trained score semantics without tuning."""

import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import (
    build_context,
    build_encoder,
    change_gate_loss,
    config_from_json,
    decomposition_loss,
    load_checkpoint,
)

ARCHIVE = Path("research/experiments/results/2026-10-13_n04_event_semantics")
ROOT = Path("outputs/n04-event-semantics-20261013")


def save(name, value):
    (ARCHIVE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def mean(values):
    present = [x for x in values if x is not None]
    return statistics.fmean(present) if present else None


def metric(row, key):
    if key == "location_mean_px":
        return row[key]
    if key in ("camera_vs_object_change", "illumination_response_ratio"):
        return row["contrast"][key]
    condition, field = key.split("/", 1)
    return next(c[field] for c in row["conditions"] if c["condition"] == condition)


KEYS = (
    "location_mean_px",
    "camera_vs_object_change",
    "illumination_response_ratio",
    "camera_only/external_false_positive_rate",
    "illumination/external_false_positive_rate",
    "object_only/external_false_negative_rate",
    "both/external_false_negative_rate",
    "camera_only/warp_error",
)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["train_seed"]].append(row)
    result = {}
    for key in KEYS:
        values = {seed: mean(metric(row, key) for row in group) for seed, group in grouped.items()}
        present = [v for v in values.values() if v is not None]
        result[key] = {
            "mean": mean(present),
            "std_across_initializations": statistics.stdev(present) if len(present) > 1 else None,
            "per_initialization": values,
        }
    result["mean_condition_latency_p50_ms"] = mean(
        c["latency_p50_ms"] for row in rows for c in row["conditions"]
    )
    result["mean_condition_latency_p95_ms"] = mean(
        c["latency_p95_ms"] for row in rows for c in row["conditions"]
    )
    result["state_bytes_batch2"] = rows[0]["conditions"][0]["state_bytes"]
    result["recovery"] = {
        condition: [
            {
                "train_seed": row["train_seed"],
                "eval_seed": row["eval_seed"],
                "frames": c["recovery_frames"],
                "censored": c["recovery_censored"],
            }
            for row in rows
            for c in row["conditions"]
            if c["condition"] == condition
        ]
        for condition in ("object_only", "occlusion")
    }
    return result


def checkpoint_audit(seed):
    cfg, _ = config_from_json(Path("configs/encoders/n04/tiny.json"))
    encoder = build_encoder(cfg)
    checkpoint = ROOT / "candidate" / f"train-{seed}" / "checkpoint.pt"
    payload = load_checkpoint(checkpoint, encoder=encoder)
    assert payload["step"] == 200
    scene = SceneConfig(
        noise_sigma=0, object_velocity=(1, 1), object_colors=((0.4, 0.2, 0.1),), seed=321
    )
    sequence = SyntheticSequenceGenerator(scene).sequence(
        length=5, batch=1, condition="both", seed=322
    )
    state = encoder.init_state(1)[cfg.camera_ids[0]]
    previous = None
    for frame in sequence[:-1]:
        with torch.no_grad():
            out = encoder.step(
                frame.measured,
                build_context(frame, previous),
                state,
                saturation=frame.saturated_mask,
                measurement_valid=frame.valid_mask,
            )
        state, previous = out.state, frame
    captured = []
    hook = encoder.decomposer.head.register_forward_hook(
        lambda mod, args, value: captured.append(value)
    )
    frame = sequence[-1]
    out = encoder.step(
        frame.measured,
        build_context(frame, previous),
        state,
        saturation=frame.saturated_mask,
        measurement_valid=frame.valid_mask,
    )
    hook.remove()
    logits = captured[0]
    event = change_gate_loss(out.innovation.gate_change, frame)
    auxiliary = decomposition_loss(out.innovation, out.state.forward_flow, frame, previous, cfg)
    event_grad = torch.autograd.grad(event, logits, retain_graph=True)[0][:, 1:2]
    auxiliary_grad = torch.autograd.grad(auxiliary, logits)[0][:, 1:2]
    positive = (
        F.adaptive_avg_pool2d(
            frame.object_mask * frame.valid_mask * (1 - frame.saturated_mask), event_grad.shape[-2:]
        )
        > 0.5
    )
    assert auxiliary_grad.abs().max() == 0
    assert event_grad[positive].sum() < 0
    return {
        "train_seed": seed,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "revision": payload["mechanism_revision"],
        "auxiliary_gradient_on_current_external_logit_max": auxiliary_grad.abs().max().item(),
        "event_gradient_on_positive_external_logits_sum": event_grad[positive].sum().item(),
        "external_head_weight_l1": encoder.decomposer.head.weight[1].abs().sum().item(),
    }


def main():
    protocol = json.loads((ARCHIVE / "protocol.json").read_text())
    assert protocol["complete"] and protocol["evaluations"] == 252
    assert (
        hashlib.sha256(Path("configs/encoders/n04/tiny.json").read_bytes()).hexdigest()
        == (protocol["config_sha256"])
    )
    for path, expected in protocol["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    rows = json.loads((ARCHIVE / "matrix.json").read_text())
    assert len(rows) == 252
    names = ["previous_candidate", *protocol["models"]]
    expected_keys = {
        (name, seed, eval_seed)
        for name in names
        for seed in protocol["initialization_seeds"]
        for eval_seed in (*protocol["continuity_eval_seeds"], *protocol["fresh_eval_seeds"])
    }
    assert {(r["name"], r["train_seed"], r["eval_seed"]) for r in rows} == expected_keys
    training = json.loads((ARCHIVE / "training.json").read_text())
    assert len(training) == protocol["checkpoints"] == 42
    assert {(r["name"], r["train_seed"]) for r in training} == {
        (name, seed) for name in names for seed in protocol["initialization_seeds"]
    }
    assert all(r["steps"] == protocol["updates"] == 200 for r in training)
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
        assert all(p["invariant_verified"] for p in row["counterfactuals"])
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["split"]][row["name"]].append(row)
    summary = {
        split: {name: aggregate(group) for name, group in models.items()}
        for split, models in groups.items()
    }
    save("summary.json", summary)
    torch.set_num_threads(4)
    torch.manual_seed(123)
    audits = [checkpoint_audit(seed) for seed in (0, 1, 2)]
    commit = subprocess.check_output(["git", "rev-parse", "57138c1"], text=True).strip()
    for path, expected in protocol["source_sha256"].items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        assert hashlib.sha256(blob).hexdigest() == expected
    save(
        "audit.json",
        {
            "verified_source_commit": commit,
            "all_source_and_checkpoint_hashes_match": True,
            "all_expected_training_and_evaluation_identities_verified": True,
            "all_paired_counterfactual_invariants_verified": True,
            "trained_score_audits": audits,
            "note": "Zero direct auxiliary gradient proves removal of the second event target; "
            "shared representation can still receive multiple objectives.",
        },
    )
    print(json.dumps({split: summary[split]["candidate"] for split in summary}, indent=2))


if __name__ == "__main__":
    main()
