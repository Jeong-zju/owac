"""Development-only combination transfer with matched stage configs and inputs."""

from pathlib import Path

import torch
from run import ARCHIVE, DEV_SEEDS, ROOT, digest, save, stage_config

from owac.evaluation.n04 import evaluate_condition
from owac.frameworks.n04.synthetic import SceneConfig
from owac.training.n04 import build_encoder, load_checkpoint

TARGETS = (
    ("s2b-support", "candidate"),
    ("s3a-imaging", "candidate"),
    ("s3b-combined", "candidate"),
    ("s3b-combined", "predict_update_filter"),
    ("s3b-combined", "n04_no_change_branch"),
    ("s3c-combined-hint", "candidate"),
)
CONDITIONS = ("static", "camera_illumination", "object_illumination", "both_illumination")


def main():
    torch.set_num_threads(4)
    results = []
    sources = [
        *Path("src/owac/frameworks/n04").glob("*.py"),
        Path("src/owac/evaluation/n04.py"),
        Path(__file__),
    ]
    hashes = {str(p): digest(p) for p in sources}
    for stage, name in TARGETS:
        cfg, _ = stage_config(stage)
        for run in (0, 1, 2):
            cp = ROOT / stage / name / f"run-{run}" / "checkpoint.pt"
            encoder = build_encoder(cfg, baseline=None if name == "candidate" else name)
            load_checkpoint(cp, encoder=encoder)
            encoder.cuda().eval()
            for seed in DEV_SEEDS:
                scene = SceneConfig(
                    camera_translation=(0.6, 2),
                    camera_rotation=(0, 0.7),
                    object_velocity=(0.4, 1.8),
                    illumination=(0.7, 1.3),
                    seed=seed,
                )
                for condition in CONDITIONS:
                    report = evaluate_condition(
                        encoder, condition=condition, scene=scene, seed=seed
                    ).as_dict()
                    save(
                        ROOT / stage / name / f"run-{run}" / f"combo-{condition}-{seed}.json",
                        report,
                    )
                    report.pop("trajectory")
                    results.append(
                        dict(
                            stage=stage,
                            name=name,
                            run=run,
                            eval_seed=seed,
                            checkpoint_sha256=digest(cp),
                            **report,
                        )
                    )
            print("combo done", stage, name, run, flush=True)
    assert all(digest(Path(p)) == sha for p, sha in hashes.items())
    save(ARCHIVE / "development-combinations.json", results)
    save(
        ARCHIVE / "combinations-provenance.json",
        dict(
            source_sha256=hashes,
            conditions=CONDITIONS,
            targets=TARGETS,
            seeds=DEV_SEEDS,
            records=len(results),
            complete=True,
        ),
    )


if __name__ == "__main__":
    main()
