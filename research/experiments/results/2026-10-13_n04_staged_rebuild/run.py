"""One-factor N04 stages; train paired runs on independently seeded corrected data."""

import argparse
import hashlib
import json
import statistics
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import torch

from owac.evaluation.n04 import json_safe, run_mechanism_suite
from owac.frameworks.n04.synthetic import SceneConfig
from owac.training.n04 import (
    build_encoder,
    config_from_json,
    load_checkpoint,
    make_synthetic_batches,
    train,
)

ARCHIVE = Path("research/experiments/results/2026-10-13_n04_staged_rebuild")
ROOT = Path("outputs/n04-staged-rebuild-20261013")
DATA_SEEDS = (41000, 41001, 41002)
DEV_SEEDS = (42000, 42001, 42002)
TEST_SEEDS = (45000, 45001, 45002)
STAGE1_MODELS = (
    "candidate",
    "motion_compensation",
    "predict_update_filter",
    "n04_no_change_branch",
    "n04_no_illumination",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_hash(batches):
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


def stage_config(stage):
    cfg, recipe = config_from_json(Path("configs/encoders/n04/tiny.json"))
    cfg_path = ARCHIVE / f"{stage}-config.json"
    if cfg_path.exists():
        payload = json.loads(cfg_path.read_text())
        cfg = cfg.with_model(**payload.get("model", {}))
        recipe = replace(recipe, synthetic={**recipe.synthetic, **payload.get("synthetic", {})})
    return cfg, replace(recipe, device="cuda")


def evaluate(encoder, *, name, run, checkpoint, seeds, stage):
    rows = []
    encoder.cuda().eval()
    for seed in seeds:
        scene = SceneConfig(
            camera_translation=(0.6, 2.0),
            camera_rotation=(0, 0.7),
            object_velocity=(0.4, 1.8),
            illumination=(0.7, 1.3),
            calibration_scale=1.08,
            action_delay=2,
            seed=seed,
        )
        report = run_mechanism_suite(encoder, base_scene=scene, seed=seed).as_dict()
        save(ROOT / stage / name / f"run-{run}" / f"eval-{seed}.json", report)
        row = dict(
            stage=stage,
            name=name,
            run=run,
            eval_seed=seed,
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            location_mean_px=statistics.fmean(c["location_error"] for c in report["conditions"]),
            report={
                **report,
                "conditions": [
                    {key: value for key, value in condition.items() if key != "trajectory"}
                    for condition in report["conditions"]
                ],
            },
        )
        rows.append(row)
        print(stage, name, run, seed, row["location_mean_px"], report["contrast"], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    stage = args.stage
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    cfg, recipe = stage_config(stage)
    models = args.models or list(STAGE1_MODELS if stage == "s1-data" else ("candidate",))
    source = [
        *Path("src/owac/frameworks/n04").glob("*.py"),
        Path("src/owac/training/n04.py"),
        Path("src/owac/baselines/n04.py"),
        Path("src/owac/evaluation/n04.py"),
    ]
    protocol = dict(
        stage=stage,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p): digest(p) for p in source},
        model_config=cfg.model.as_dict(),
        recipe=asdict(recipe),
        initialization_seeds=[0, 1, 2],
        training_data_seeds=DATA_SEEDS,
        development_seeds=DEV_SEEDS,
        final_test_seeds_reserved=TEST_SEEDS,
        models=models,
        updates=200,
        detector_threshold=0.5,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        data_revision="sample-independent-render-label-v1",
        data_sha256={},
        complete=False,
        note="Same data per paired model; three independent data sets and initializations. "
        "Final test remains unobserved until all stages are frozen. "
        "Equal updates, not capacity or FLOPs.",
    )
    save(ARCHIVE / f"{stage}-protocol.json", protocol)
    rows, training = [], []
    for run, data_seed in enumerate(DATA_SEEDS):
        batches = make_synthetic_batches(
            cfg=cfg, train_cfg=recipe, steps=200, seed=data_seed, device="cuda"
        )
        protocol["data_sha256"][str(run)] = data_hash(batches)
        save(ARCHIVE / f"{stage}-protocol.json", protocol)
        for name in models:
            destination = ROOT / stage / name / f"run-{run}"
            assert not (destination / "checkpoint.pt").exists(), (
                "Never overwrite frozen stage weights"
            )
            baseline = None if name == "candidate" else name
            result = train(
                cfg,
                replace(recipe, seed=run),
                batches=batches,
                output_dir=destination,
                baseline=baseline,
            )
            summary = asdict(result)
            history = summary.pop("history")
            training.append(
                dict(
                    name=name,
                    run=run,
                    data_seed=data_seed,
                    **summary,
                    mean_first20={
                        k: statistics.fmean(x[k] for x in history[:20]) for k in history[0]
                    },
                    mean_last20={
                        k: statistics.fmean(x[k] for x in history[-20:]) for k in history[0]
                    },
                )
            )
            encoder = build_encoder(cfg, baseline=baseline)
            checkpoint = destination / "checkpoint.pt"
            load_checkpoint(checkpoint, encoder=encoder)
            rows.extend(
                evaluate(
                    encoder, name=name, run=run, checkpoint=checkpoint, seeds=DEV_SEEDS, stage=stage
                )
            )
            save(ARCHIVE / f"{stage}-matrix.json", rows)
            save(ARCHIVE / f"{stage}-training.json", training)
        assert data_hash(batches) == protocol["data_sha256"][str(run)], (
            "Training mutated paired data"
        )
    assert all(digest(Path(p)) == h for p, h in protocol["source_sha256"].items())
    assert len(rows) == len(models) * 9 and len(training) == len(models) * 3
    protocol.update(complete=True, checkpoints=len(training), suites=len(rows))
    save(ARCHIVE / f"{stage}-protocol.json", protocol)


if __name__ == "__main__":
    main()
