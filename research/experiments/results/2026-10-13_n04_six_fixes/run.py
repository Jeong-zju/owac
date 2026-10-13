"""One frozen-budget before/after sweep; large artifacts stay under outputs/."""

import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import tarfile
import time
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

ROOT = Path("outputs/n04-six-fixes-20261013")
ARCHIVE = Path("research/experiments/results/2026-10-13_n04_six_fixes")
OLD_COMMIT = "b4c26cc4b0de3a3df5e73bca4eb4eab43b849bd4"
SEEDS = (10000, 10001, 10002)
CONFIG = Path("configs/encoders/n04/tiny.json")
DEVICE = "cuda"


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_training(name, payload, destination):
    history = payload.pop("history")
    return {
        "name": name,
        **payload,
        "first": history[0],
        "mean_first20": {
            key: statistics.fmean(row[key] for row in history[:20]) for key in history[0]
        },
        "mean_last20": {
            key: statistics.fmean(row[key] for row in history[-20:]) for key in history[0]
        },
        "full_history": str(destination / "history.json"),
    }


def record(name, seed, report, checkpoint):
    conditions = [{k: v for k, v in c.items() if k != "trajectory"} for c in report["conditions"]]
    return {
        "name": name,
        "seed": seed,
        "checkpoint_sha256": digest(checkpoint),
        "location_mean_px": statistics.fmean(c["location_error"] for c in conditions),
        "contrast": report["contrast"],
        "conditions": conditions,
        "counterfactuals": report["counterfactuals"],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("frozen comparison expects CUDA, as in the previous acceptance")
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    old_source = ROOT / "before-source"
    old_source.mkdir(exist_ok=True)
    archive = subprocess.check_output(["git", "archive", OLD_COMMIT, "src/owac"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(old_source, filter="data")
    env = dict(os.environ)
    env["PYTHONPATH"] = str((old_source / "src").resolve())
    before = []
    old_checkpoint = Path("outputs/n04-acceptance/candidate/checkpoint.pt")
    for seed in SEEDS:
        destination = ROOT / "before-candidate" / str(seed)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "owac.evaluation.n04",
                "--config",
                str(CONFIG),
                "--checkpoint",
                str(old_checkpoint),
                "--seed",
                str(seed),
                "--device",
                DEVICE,
                "--output",
                str(destination),
            ],
            env=env,
            check=True,
        )
        payload = json.loads((destination / "mechanism.json").read_text())
        before.append(record("candidate", seed, payload["report"], old_checkpoint))
    save(ARCHIVE / "before-matrix.json", before)
    cfg, recipe = config_from_json(CONFIG)
    recipe = replace(recipe, device=DEVICE)
    started = time.perf_counter()
    batches = make_synthetic_batches(cfg=cfg, train_cfg=recipe, steps=200, seed=0, device=DEVICE)
    data_digest = hashlib.sha256()
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
                data_digest.update(getattr(frame, key).cpu().contiguous().numpy().tobytes())
    manifest = {
        "before_commit": OLD_COMMIT,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": sys.version,
        "training_seed": 0,
        "updates": 200,
        "evaluation_seeds": SEEDS,
        "training_data_sha256": data_digest.hexdigest(),
        "data_seconds": time.perf_counter() - started,
        "config_sha256": digest(CONFIG),
        "train_config": asdict(recipe),
        "source_sha256": {
            str(p): digest(p)
            for p in [
                *Path("src/owac/frameworks/n04").glob("*.py"),
                Path("src/owac/training/n04.py"),
                Path("src/owac/baselines/n04.py"),
                Path("src/owac/evaluation/n04.py"),
            ]
        },
        "generator_unchanged": digest(Path("src/owac/frameworks/n04/synthetic.py"))
        == digest(old_source / "src/owac/frameworks/n04/synthetic.py"),
    }
    assert manifest["generator_unchanged"]
    save(ARCHIVE / "manifest.json", manifest)
    print("shared data generated and hashed", flush=True)
    matrix, training = [], []
    for name in ["candidate", *BASELINE_NAMES, "frozen"]:
        baseline = None if name in ("candidate", "frozen") else name
        destination = ROOT / "after" / name
        result = train(
            cfg,
            replace(recipe, freeze_propagation=name == "frozen"),
            batches=batches,
            output_dir=destination,
            baseline=baseline,
        )
        training.append(compact_training(name, asdict(result), destination))
        save(ARCHIVE / "training.json", training)
        encoder = build_encoder(cfg, baseline=baseline)
        checkpoint = destination / "checkpoint.pt"
        load_checkpoint(checkpoint, encoder=encoder)
        encoder.to(DEVICE).eval()
        for seed in SEEDS:
            scene = SceneConfig(
                height=64,
                width=64,
                camera_translation=(0.6, 2.0),
                camera_rotation=(0, 0.7),
                object_velocity=(0.4, 1.8),
                illumination=(0.7, 1.3),
                calibration_scale=1.08,
                action_delay=2,
                seed=seed,
            )
            report = run_mechanism_suite(encoder, base_scene=scene, seed=seed).as_dict()
            save(destination / f"eval-{seed}.json", report)
            matrix.append(record(name, seed, report, checkpoint))
            save(ARCHIVE / "after-matrix.json", matrix)
            print(name, seed, matrix[-1]["location_mean_px"], report["contrast"], flush=True)
        del encoder
    manifest["complete"] = True
    manifest["models"] = [row["name"] for row in training]
    save(ARCHIVE / "manifest.json", manifest)


if __name__ == "__main__":
    main()
