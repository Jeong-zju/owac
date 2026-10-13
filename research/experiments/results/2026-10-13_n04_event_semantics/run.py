"""Frozen semantic repair comparison; checkpoints and full trajectories stay in outputs/."""

import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import tarfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from owac.baselines.n04 import BASELINE_NAMES
from owac.evaluation.n04 import json_safe, run_mechanism_suite
from owac.frameworks.n04.synthetic import SceneConfig
from owac.training.n04 import (
    build_encoder,
    config_from_json,
    load_checkpoint,
    make_synthetic_batches,
    train,
)

ROOT = Path("outputs/n04-event-semantics-20261013")
ARCHIVE = Path("research/experiments/results/2026-10-13_n04_event_semantics")
PREVIOUS_COMMIT = "771b33a"
CONFIG = Path("configs/encoders/n04/tiny.json")
TRAIN_SEEDS = (0, 1, 2)
CONTINUITY_SEEDS = (10000, 10001, 10002)
FRESH_SEEDS = (20000, 20001, 20002)
EXPECTED_DATA_HASH = "a3ae0870bf20e356e20a66f8a99c50723a02e2fd22b5feb7df8d4fc33fcf0c0b"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batches_hash(batches):
    value = hashlib.sha256()
    for sequence in batches:
        for frame in sequence:
            for key in (
                "measured",
                "proprio",
                "action",
                "valid_mask",
                "saturated_mask",
                "object_mask",
                "oracle_camera_flow",
                "true_illumination",
            ):
                value.update(getattr(frame, key).cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def compact_training(name, train_seed, result, destination):
    payload = asdict(result)
    history = payload.pop("history")
    return {
        "name": name,
        "train_seed": train_seed,
        **payload,
        "first": history[0],
        "mean_first20": {k: statistics.fmean(row[k] for row in history[:20]) for k in history[0]},
        "mean_last20": {k: statistics.fmean(row[k] for row in history[-20:]) for k in history[0]},
        "full_history": str(destination / "history.json"),
    }


def evaluate(name, train_seed, checkpoint, *, baseline=None):
    cfg, _ = config_from_json(CONFIG)
    encoder = build_encoder(cfg, baseline=baseline)
    load_checkpoint(checkpoint, encoder=encoder)
    encoder.to("cuda").eval()
    rows = []
    for eval_seed in (*CONTINUITY_SEEDS, *FRESH_SEEDS):
        scene = SceneConfig(
            height=64,
            width=64,
            camera_translation=(0.6, 2.0),
            camera_rotation=(0, 0.7),
            object_velocity=(0.4, 1.8),
            illumination=(0.7, 1.3),
            calibration_scale=1.08,
            action_delay=2,
            seed=eval_seed,
        )
        report = run_mechanism_suite(encoder, base_scene=scene, seed=eval_seed).as_dict()
        save(ROOT / name / f"train-{train_seed}" / f"eval-{eval_seed}.json", report)
        conditions = []
        for item in report["conditions"]:
            conditions.append(
                {
                    k: v
                    for k, v in item.items()
                    if k
                    in (
                        "condition",
                        "location_error",
                        "change_response",
                        "warp_error",
                        "recovery_frames",
                        "latency_p50_ms",
                        "latency_p95_ms",
                        "state_bytes",
                    )
                }
                | {
                    k: v
                    for k, v in item["extra"].items()
                    if k
                    in (
                        "external_false_positive_rate",
                        "external_false_negative_rate",
                        "recovery_censored",
                        "target_iou",
                    )
                }
            )
        row = {
            "name": name,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "split": "continuity" if eval_seed in CONTINUITY_SEEDS else "fresh",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "location_mean_px": statistics.fmean(c["location_error"] for c in conditions),
            "contrast": report["contrast"],
            "conditions": conditions,
            "counterfactuals": report["counterfactuals"],
        }
        rows.append(row)
        print(name, train_seed, eval_seed, row["location_mean_px"], row["contrast"], flush=True)
    return rows


def previous_worker(train_seed):
    # Executed under the archived previous PYTHONPATH; never load old shares in new code.
    cfg, recipe = config_from_json(CONFIG)
    recipe = replace(recipe, device="cuda", seed=train_seed)
    torch.set_num_threads(4)
    batches = make_synthetic_batches(cfg=cfg, train_cfg=recipe, steps=200, seed=0, device="cuda")
    assert batches_hash(batches) == EXPECTED_DATA_HASH
    destination = ROOT / "previous_candidate" / f"train-{train_seed}"
    if train_seed == 0:
        checkpoint = Path("outputs/n04-six-fixes-20261013/after/candidate/checkpoint.pt")
        old_training = json.loads(
            Path("research/experiments/results/2026-10-13_n04_six_fixes/training.json").read_text()
        )[0]
        training = {**old_training, "train_seed": 0, "checkpoint_reused": True}
    else:
        result = train(cfg, recipe, batches=batches, output_dir=destination)
        training = compact_training("previous_candidate", train_seed, result, destination)
        checkpoint = destination / "checkpoint.pt"
    rows = evaluate("previous_candidate", train_seed, checkpoint)
    save(ROOT / f"previous-{train_seed}-summary.json", {"training": training, "rows": rows})


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("comparison requires the same CUDA runtime as the previous acceptance")
    torch.set_num_threads(4)
    cfg, recipe = config_from_json(CONFIG)
    recipe = replace(recipe, device="cuda")
    previous_sha = subprocess.check_output(["git", "rev-parse", PREVIOUS_COMMIT], text=True).strip()
    snapshot = ROOT / "previous-source"
    snapshot.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", previous_sha, "src/owac"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as content:
        content.extractall(snapshot, filter="data")
    source_files = [
        *Path("src/owac/frameworks/n04").glob("*.py"),
        Path("src/owac/training/n04.py"),
        Path("src/owac/baselines/n04.py"),
        Path("src/owac/evaluation/n04.py"),
    ]
    models = ["candidate", *BASELINE_NAMES, "frozen"]
    protocol = {
        "previous_commit": previous_sha,
        "source_sha256": {str(p): digest(p) for p in source_files},
        "config_sha256": digest(CONFIG),
        "training_data_seed": 0,
        "training_data_sha256": EXPECTED_DATA_HASH,
        "initialization_seeds": TRAIN_SEEDS,
        "continuity_eval_seeds": CONTINUITY_SEEDS,
        "fresh_eval_seeds": FRESH_SEEDS,
        "updates": 200,
        "train_recipe": asdict(recipe),
        "external_gate_threshold": 0.5,
        "models": models,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "python": sys.version,
        "generator_unchanged": digest(Path("src/owac/frameworks/n04/synthetic.py"))
        == digest(snapshot / "src/owac/frameworks/n04/synthetic.py"),
        "evaluation_unchanged": digest(Path("src/owac/evaluation/n04.py"))
        == digest(snapshot / "src/owac/evaluation/n04.py"),
        "complete": False,
    }
    assert protocol["generator_unchanged"] and protocol["evaluation_unchanged"]
    save(ARCHIVE / "protocol.json", protocol)  # frozen before any evaluation
    env = dict(os.environ)
    env["PYTHONPATH"] = str((snapshot / "src").resolve())
    matrix, training = [], []
    for train_seed in TRAIN_SEEDS:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "previous", str(train_seed)],
            env=env,
            check=True,
        )
        payload = json.loads((ROOT / f"previous-{train_seed}-summary.json").read_text())
        matrix.extend(payload["rows"])
        training.append({**payload["training"], "name": "previous_candidate"})
        save(ARCHIVE / "matrix.json", matrix)
        save(ARCHIVE / "training.json", training)
    batches = make_synthetic_batches(cfg=cfg, train_cfg=recipe, steps=200, seed=0, device="cuda")
    assert batches_hash(batches) == EXPECTED_DATA_HASH
    print("shared data hash verified; independent-score training starts", flush=True)
    for train_seed in TRAIN_SEEDS:
        for name in models:
            baseline = None if name in ("candidate", "frozen") else name
            destination = ROOT / name / f"train-{train_seed}"
            result = train(
                cfg,
                replace(recipe, seed=train_seed, freeze_propagation=name == "frozen"),
                batches=batches,
                output_dir=destination,
                baseline=baseline,
            )
            training.append(compact_training(name, train_seed, result, destination))
            matrix.extend(
                evaluate(name, train_seed, destination / "checkpoint.pt", baseline=baseline)
            )
            save(ARCHIVE / "matrix.json", matrix)
            save(ARCHIVE / "training.json", training)
    assert all(digest(Path(p)) == v for p, v in protocol["source_sha256"].items())
    protocol["complete"] = True
    protocol["checkpoints"] = len(training)
    protocol["evaluations"] = len(matrix)
    save(ARCHIVE / "protocol.json", protocol)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        previous_worker(int(sys.argv[2]))
    else:
        main()
