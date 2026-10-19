"""Recompute r16 stage/horizon/episode residual diagnostics, zero updates."""

import json
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_full.action_learning import selected_splits
from owac.evaluation.n02_full.action_stage import action_stages, correction_direction
from owac.frameworks.n02_full.task_chain import action_metrics

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "outputs/n02-v12/learning-r15"
R16 = ROOT / "outputs/n02-v12/residual-r16"


def main():
    data = torch.load(PARENT / "dataset.pt", weights_only=False)
    config = json.loads((HERE.parent / "training_r11_20261017/task_config.json").read_text())
    std = json.loads(Path(config["data"]["stats_path"]).read_text())["norm_stats"]["actions"]["std"]
    selection = json.loads((HERE.parent / "learning_r15_20261019/selection.json").read_text())
    splits = selected_splits(data["records"], selection["train_groups"]["5"])
    labels = {}
    trajectories = []
    for group in sorted({r["group"] for r in data["records"]}):
        with np.load(Path(config["data"]["data_root"]) / group) as a:
            stages = action_stages(a["actions"], a["left_endpose"], a["right_endpose"])
        trajectories.append(
            dict(
                group=group,
                counts={k: int((stages == k).sum()) for k in ("approach", "grasp", "lift")},
            )
        )
        for i, r in enumerate(data["records"]):
            if r["group"] == group:
                labels[i] = str(stages[r["t"]])
    rows = []
    for file in sorted(R16.glob("*/report.json")):
        report = json.loads(file.read_text())
        with np.load(file.parent / "predictions.npz") as saved:
            for split, indices in splits.items():
                p = saved["1024_selected_" + split]
                b = saved["baseline_" + split]
                for stage in ("all", "approach", "grasp", "lift"):
                    for light in ("all", "Normal", "Over"):
                        local = [
                            j
                            for j, i in enumerate(indices)
                            if (stage == "all" or labels[i] == stage)
                            and (light == "all" or data["records"][i]["light"] == light)
                        ]
                        if not local:
                            continue
                        ix = np.asarray(indices)[local]
                        for horizon in (1, 10, 50):
                            valid = np.asarray(data["mask"])[ix].copy()
                            valid[:, horizon:] = False
                            target = np.asarray(data["actions"])[ix]
                            before = action_metrics(b[local], target, valid, std)
                            after = action_metrics(p[local], target, valid, std)
                            rows.append(
                                dict(
                                    run_id=report["run_id"],
                                    modality=report["modality"],
                                    seed=report["seed"],
                                    split=split,
                                    stage=stage,
                                    light=light,
                                    horizon=horizon,
                                    baseline=before,
                                    residual=after,
                                    direction=correction_direction(
                                        b[local], p[local], target, valid, std
                                    ),
                                    episodes=sorted({data["records"][i]["group"] for i in ix}),
                                )
                            )
    output = dict(
        stage_protocol=(
            "commanded closure0.95/0.2 plus bilateral measured EE rise2cm; "
            "anchor stage; horizon can cross stages; not physical-contact labels"
        ),
        trajectories=trajectories,
        rows=rows,
        updates=0,
    )
    (HERE / "stage_analysis.json").write_text(json.dumps(output, indent=2) + chr(10))
    for stage in ("approach", "grasp", "lift"):
        for h in (1, 10, 50):
            rows2 = [
                r
                for r in rows
                if r["modality"] == "raw"
                and r["split"] == "test"
                and r["stage"] == stage
                and r["light"] == "Normal"
                and r["horizon"] == h
            ]
            print(
                stage,
                h,
                [
                    (
                        r["baseline"]["windows"],
                        round(r["baseline"]["joint_rms_deg"], 2),
                        round(r["residual"]["joint_rms_deg"], 2),
                    )
                    for r in rows2
                ],
            )


if __name__ == "__main__":
    main()
