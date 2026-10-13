"""Run the reserved test once after code/config freeze. Never selects hyperparameters."""

import subprocess
from pathlib import Path

import torch
from run import ARCHIVE, ROOT, TEST_SEEDS, digest, save, stage_config

from owac.evaluation.n04 import MECHANISM_CONDITIONS, run_mechanism_suite
from owac.frameworks.n04.synthetic import SceneConfig
from owac.training.n04 import build_encoder, load_checkpoint

# Chosen before opening held-out results: original reference + matched final controls.
TARGETS = (
    ("s1-data", "candidate"),
    ("s1-data", "predict_update_filter"),
    ("s3b-combined", "candidate"),
    ("s3b-combined", "predict_update_filter"),
    ("s3b-combined", "n04_no_change_branch"),
    ("s3c-combined-hint", "candidate"),
)
CONDITIONS = (
    *MECHANISM_CONDITIONS,
    "static",
    "camera_illumination",
    "object_illumination",
    "both_illumination",
)


def main():
    torch.set_num_threads(4)
    destination = ARCHIVE / "final-test-matrix.json"
    assert not destination.exists(), "Reserved tests cannot be silently overwritten"
    source = [
        *Path("src/owac/frameworks/n04").glob("*.py"),
        Path("src/owac/training/n04.py"),
        Path("src/owac/evaluation/n04.py"),
        Path("src/owac/baselines/n04.py"),
        Path(__file__),
    ]
    protocol = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p): digest(p) for p in source},
        targets=TARGETS,
        seeds=TEST_SEEDS,
        conditions=CONDITIONS,
        complete=False,
        expected_suites=len(TARGETS) * 9,
        thresholds=dict(event=0.5, iou=0.5, location_px=2),
    )
    save(ARCHIVE / "heldout-protocol.json", protocol)
    results = []
    for stage, name in TARGETS:
        cfg, _ = stage_config(stage)
        for run in (0, 1, 2):
            cp = ROOT / stage / name / f"run-{run}" / "checkpoint.pt"
            encoder = build_encoder(cfg, baseline=None if name == "candidate" else name)
            load_checkpoint(cp, encoder=encoder)
            encoder.cuda().eval()
            for seed in TEST_SEEDS:
                scene = SceneConfig(
                    canvas_margin=32,
                    camera_translation=(0.6, 2.0),
                    camera_rotation=(0, 0.7),
                    object_velocity=(0.4, 1.8),
                    illumination=(0.7, 1.3),
                    calibration_scale=1.08,
                    action_delay=2,
                    seed=seed,
                )
                report = run_mechanism_suite(
                    encoder, base_scene=scene, seed=seed, conditions=CONDITIONS
                ).as_dict()
                assert all(x["invariant_verified"] for x in report["counterfactuals"])
                save(ROOT / stage / name / f"run-{run}" / f"heldout-{seed}.json", report)
                for condition in report["conditions"]:
                    condition.pop("trajectory")
                results.append(
                    dict(
                        stage=stage,
                        name=name,
                        run=run,
                        eval_seed=seed,
                        checkpoint=str(cp),
                        checkpoint_sha256=digest(cp),
                        report=report,
                    )
                )
                save(destination, results)
            print("heldout done", stage, name, run, flush=True)
    assert len(results) == protocol["expected_suites"]
    assert all(digest(Path(p)) == sha for p, sha in protocol["source_sha256"].items())
    protocol["complete"] = True
    save(ARCHIVE / "heldout-protocol.json", protocol)


if __name__ == "__main__":
    main()
