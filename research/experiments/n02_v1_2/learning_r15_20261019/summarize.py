"""Audit raw updates, matched controls, and saved action predictions for r15."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_full.action_learning import selected_splits
from owac.frameworks.n02_full.task_chain import action_metrics

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/n02-v12/learning-r15"


def main(partial=False):
    data = torch.load(OUT / "dataset.pt", weights_only=False)
    selection = json.loads((EVIDENCE / "selection.json").read_text())
    config = json.loads((EVIDENCE.parent / "training_r11_20261017/task_config.json").read_text())
    std = json.loads(Path(config["data"]["stats_path"]).read_text())["norm_stats"]["actions"]["std"]
    reports = []
    audits = []
    actual = 0
    inits = {}
    orders = {}
    for file in sorted(OUT.glob("*/report.json")):
        r = json.loads(file.read_text())
        name = r["run_id"]
        directory = file.parent
        if not r.get("completed"):
            raise ValueError("incomplete report")
        splits = selected_splits(data["records"], selection["train_groups"][str(r["demos"])])
        if splits["train"] != r["train_indices"]:
            raise ValueError("training indices changed")
        history = [
            json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()
        ]
        if [h["step"] for h in history] != list(range(1, 1025)):
            raise ValueError("missing or repeated actual updates")
        if not all(np.isfinite(h["loss"]) and np.isfinite(h["gradient_norm"]) for h in history):
            raise ValueError("nonfinite optimizer path")
        actual += len(history)
        order = json.loads((directory / "sample_order.json").read_text())
        if len(order) != 1024 or any(not set(b) <= set(splits["train"]) for b in order):
            raise ValueError("optimizer saw heldout groups")
        key = (r["demos"], r["seed"])
        orders.setdefault(key, order)
        if order != orders[key]:
            raise ValueError("unpaired modality sampling")
        if r["modality"] != "pi":
            inits.setdefault(r["seed"], (r["initial_sha"], r["parameters"]))
            if (r["initial_sha"], r["parameters"]) != inits[r["seed"]]:
                raise ValueError("head initialization/capacity differs")
        if r["modality"] == "pi" and r["raw_sha_before"] != r["raw_sha_after"]:
            raise ValueError("frozen encoder mutated")
        with np.load(directory / "predictions.npz") as predictions:
            for row in r["rows"]:
                for version in ("selected", "last"):
                    for split in ("train", "val", "test"):
                        indices = splits[split]
                        p = predictions[f"{row['budget']}_{version}_{split}"]
                        metric = [
                            action_metrics(a, data["actions"][indices], data["mask"][indices], std)
                            for a in p
                        ]
                        avg = {k: float(np.mean([m[k] for m in metric])) for k in metric[0]}
                        if avg != row[version][split]:
                            raise ValueError("prediction metric mismatch")
                    indices = splits["test"]
                    p = predictions[f"{row['budget']}_{version}_test"]
                    row[version]["test_by_light"] = {}
                    for light in sorted({data["records"][i]["light"] for i in indices}):
                        local = [
                            j for j, i in enumerate(indices) if data["records"][i]["light"] == light
                        ]
                        original = np.asarray(indices)[local]
                        metric = [
                            action_metrics(
                                a[local], data["actions"][original], data["mask"][original], std
                            )
                            for a in p
                        ]
                        row[version]["test_by_light"][light] = {
                            k: float(np.mean([m[k] for m in metric])) for k in metric[0]
                        }
                    row[version]["normal_test"] = row[version]["test_by_light"]["Normal"]
                candidates = [h for h in r["history"] if h["step"] <= row["budget"]]
                expected = min(candidates, key=lambda h: h["val"]["normalized_control_mse"])["step"]
                if expected != row["selected_step"]:
                    raise ValueError("non-validation checkpoint selection")
        audits.append(
            dict(
                run_id=name,
                actual_updates=len(history),
                paired_sample_order=True,
                all_saved_metrics_exact=True,
                selected_only_on_validation=True,
            )
        )
        reports.append(r)
    if not partial and (len(reports) != 14 or actual != 14336):
        raise ValueError("full predefined matrix incomplete")
    aggregated = []
    for demos in (3, 5):
        for mode in ("state", "raw", "native", "pi"):
            selected = [r for r in reports if r["demos"] == demos and r["modality"] == mode]
            if not selected:
                continue
            for budget in (256, 512, 1024):
                points = [next(x for x in r["rows"] if x["budget"] == budget) for r in selected]
                row = dict(
                    modality=mode,
                    demos=demos,
                    budget=budget,
                    seeds=[r["seed"] for r in selected],
                    selected_steps=[p["selected_step"] for p in points],
                )
                for version in ("selected", "last"):
                    row[version] = {}
                    for split in ("train", "val", "test", "normal_test"):
                        row[version][split] = {
                            k: float(np.mean([p[version][split][k] for p in points]))
                            for k in points[0][version][split]
                        }
                aggregated.append(row)
    result = dict(
        completed=not partial,
        actual_updates=actual,
        planned_updates=14336,
        closed_loops=0,
        rows=aggregated,
        raw_audits=audits,
        matched_initial_heads=True,
        paired_sampling=True,
        counts=selection["counts"],
        actual_test_lights=selection["test_lights"],
        primary_scope="Normal heldout; alltest separate; historical development data",
        raw_encoder_updates=0,
        readout_parameters=inits,
        rgb_reference="clean original-lighting joint-action replay plus pretrained native vision",
        limits=[
            "3vs5 small data increase",
            "two head seeds;one pi seed",
            "pretraining and acquisition differ;not isolated architecture/RAW-format causality",
            "no physical success claim",
        ],
    )
    # JSON cannot serialize tuple keys; seed keys are integers here.
    (EVIDENCE / ("partial_summary.json" if partial else "summary.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    public_reports = [
        {k: r[k] for k in ("run_id", "modality", "demos", "seed", "actual_updates", "rows")}
        for r in reports
    ]
    temporary = OUT / "progress.audit.tmp"
    temporary.write_text(
        json.dumps(
            dict(
                reports=public_reports,
                planned_runs=14,
                completed_runs=len(reports),
                audited=not partial,
            ),
            indent=2,
        )
        + "\n"
    )
    temporary.replace(OUT / "progress.json")
    print(
        json.dumps(
            dict(
                completed=not partial,
                actual_updates=actual,
                rows=[
                    dict(
                        mode=r["modality"],
                        demos=r["demos"],
                        budget=r["budget"],
                        train_deg=r["selected"]["train"]["joint_rms_deg"],
                        normal_deg=r["selected"]["normal_test"]["joint_rms_deg"],
                        all_deg=r["selected"]["test"]["joint_rms_deg"],
                        val_deg=r["selected"]["val"]["joint_rms_deg"],
                    )
                    for r in aggregated
                    if r["budget"] == 1024
                ],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    main(args.partial)
