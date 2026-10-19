"""Independently recompute every reported metric from saved full action blocks."""

import json
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.n02_full.task_chain import action_metrics, split_groups

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/n02-v12/task-chain-r14"


def state_baseline(data, splits, std):
    """Train-only closed-form ridge, selected on val; no extra optimizer updates."""
    train = splits["train"]
    state = data["state"].astype(np.float64)
    mean, scale = state[train].mean(0), state[train].std(0).clip(1e-4)
    x = np.concatenate(((state - mean) / scale, np.ones((len(state), 1))), axis=1)
    predictions = []
    scores = []
    alphas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    for alpha in alphas:
        p = np.empty(data["actions"].shape, dtype=np.float64)
        penalty = np.eye(x.shape[1]) * alpha
        penalty[-1, -1] = 0
        for step in range(50):
            rows = np.asarray(train)[data["mask"][train, step]]
            a = x[rows]
            weights = np.linalg.solve(a.T @ a + penalty, a.T @ data["actions"][rows, step])
            p[:, step] = x @ weights
        val = splits["val"]
        score = action_metrics(p[val], data["actions"][val], data["mask"][val], std)
        predictions.append(p)
        scores.append(score["normalized_control_mse"])
    chosen = int(np.argmin(scores))
    p = predictions[chosen]
    np.savez(OUT / "state-only-predictions.npz", prediction=p)
    return dict(
        kind="linear measured-state-only ridge",
        optimizer_updates=0,
        selected_alpha=alphas[chosen],
        candidate_alphas=alphas,
        validation_scores=scores,
        selection="val only; train-only state normalization",
        metrics={
            s: action_metrics(p[i], data["actions"][i], data["mask"][i], std)
            for s, i in splits.items()
        },
        limitation="lower capacity than nonlinear heads; not a matched-capacity ablation",
    )


def main():
    data = torch.load(OUT / "dataset.pt", weights_only=False)
    splits = split_groups(data["records"])
    config = json.loads((EVIDENCE.parent / "training_r11_20261017/task_config.json").read_text())
    std = json.loads(Path(config["data"]["stats_path"]).read_text())["norm_stats"]["actions"]["std"]
    rows = []
    audits = []
    actual = 0
    orders = []
    for branch in ("small", "pi"):
        report = json.loads((EVIDENCE / (branch + "_report.json")).read_text())
        if not report.get("completed"):
            raise ValueError("do not summarize incomplete training")
        actual += report["actual_updates"]
        orders.append(json.loads((OUT / branch / "sample_order.json").read_text()))
        for phase in report["stages"]:
            stage = phase["phase"]
            with np.load(OUT / branch / (stage + "-predictions.npz")) as predictions:
                for split in ("train", "val", "test"):
                    indices = splits[split]
                    per_seed = [
                        action_metrics(p, data["actions"][indices], data["mask"][indices], std)
                        for p in predictions[split]
                    ]
                    average = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
                    if average != phase[split] or per_seed != phase[split + "_by_seed"]:
                        raise ValueError("saved prediction metrics mismatch")
                    audits.append(
                        dict(
                            branch=branch, phase=stage, split=split, seeds=len(per_seed), exact=True
                        )
                    )
                test_indices = splits["test"]
                by_light = {}
                for light in sorted({data["records"][i]["light"] for i in test_indices}):
                    local = [
                        j
                        for j, i in enumerate(test_indices)
                        if data["records"][i]["light"] == light
                    ]
                    global_ = np.asarray(test_indices)[local]
                    per_seed = [
                        action_metrics(
                            p[local], data["actions"][global_], data["mask"][global_], std
                        )
                        for p in predictions["test"]
                    ]
                    by_light[light] = {
                        k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]
                    }
                per_window = []
                for local, i in enumerate(splits["train"]):
                    per_window.extend(
                        action_metrics(
                            p[local : local + 1],
                            data["actions"][i : i + 1],
                            data["mask"][i : i + 1],
                            std,
                        )
                        for p in predictions["train"]
                    )
                maxima = {
                    k: max(m[k] for m in per_window)
                    for k in (
                        "joint_rms_deg",
                        "first_joint_max_deg",
                        "gripper_mae",
                        "gripper_max_error",
                    )
                }
                old_limits = dict(
                    joint_rms_deg=3.0,
                    first_joint_max_deg=5.0,
                    gripper_mae=0.05,
                    gripper_max_error=0.15,
                )
                fit_gate = all(maxima[k] <= limit for k, limit in old_limits.items())
                row = dict(
                    branch=branch,
                    phase=stage,
                    selected_step=phase["selected_step"],
                    selected_train=phase["train"],
                    selected_val=phase["val"],
                    selected_test=phase["test"],
                    test_by_light=by_light,
                    final_train=phase["history"][-1]["train"],
                    final_val=phase["history"][-1]["val"],
                    test_zero_visual=phase["test_zero_visual"],
                    training_window_maxima=maxima,
                    prior_r12_absolute_fit_limits=old_limits,
                    prior_r12_absolute_fit_passed=fit_gate,
                    actual_updates=phase["actual_updates"],
                    elapsed_s=phase["elapsed_s"],
                    spatial_changed_at_selected=phase["spatial_sha_before"]
                    != phase["spatial_sha_selected"],
                    peak_memory_gib=report["peak_memory_gib"],
                    task_parameters=report["head_parameters"],
                )
                rows.append(row)
    if orders[0] != orders[1] or actual != 1024:
        raise ValueError("paired samples or actual update budget differ")
    extraction = json.loads((EVIDENCE / "extraction.json").read_text())
    baseline = state_baseline(data, splits, std)
    report = dict(
        completed=True,
        actual_updates=actual,
        pi_updates=512,
        small_head_updates=512,
        closed_loops=0,
        rows=rows,
        exact_metric_audits=audits,
        paired_sample_order_exact=True,
        common_representation=extraction,
        state_only_baseline=baseline,
        split_counts={s: len(v) for s, v in splits.items()},
        limits=[
            "Historical development episodes, not an untouched benchmark",
            "Only3 Normal training episodes and1 Normal validation episode",
            "Different objectives/capacities; cannot infer inherent pi incapacity",
            "Zero-visual intervention is out of distribution, not a trained state-only control",
            "Sample seed14019; head initialization RNG restored from exact RAW850 checkpoint",
            "No physical success or broad lighting robustness conclusion",
        ],
    )
    (EVIDENCE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    public = dict(
        rows=[
            dict(
                branch=r["branch"],
                phase=r["phase"],
                train_joint_deg=r["selected_train"]["joint_rms_deg"],
                test_joint_deg=r["selected_test"]["joint_rms_deg"],
                normal_joint_deg=r["test_by_light"]["Normal"]["joint_rms_deg"],
                gripper_mae=r["selected_test"]["gripper_mae"],
                fit_passed=r["prior_r12_absolute_fit_passed"],
                selected_step=r["selected_step"],
            )
            for r in rows
        ],
        state_only_test_deg=baseline["metrics"]["test"]["joint_rms_deg"],
        actual_updates=actual,
        closed_loops=0,
    )
    (OUT / "comparison.json").write_text(json.dumps(public, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                completed=True,
                actual_updates=actual,
                rows=[
                    dict(
                        branch=r["branch"],
                        phase=r["phase"],
                        selected_step=r["selected_step"],
                        train_mse=r["selected_train"]["normalized_control_mse"],
                        test_mse=r["selected_test"]["normalized_control_mse"],
                        normal_joints=r["test_by_light"]["Normal"]["joint_rms_deg"],
                        normal_gripper=r["test_by_light"]["Normal"]["gripper_mae"],
                        fit_gate=r["prior_r12_absolute_fit_passed"],
                    )
                    for r in rows
                ],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
