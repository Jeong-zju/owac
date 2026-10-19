"""Independently recompute saved predictions and visualize held-out decoding errors."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from owac.evaluation.n02_full.information_probe import metrics, split_indices

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
OUT = ROOT / "outputs/n02-v12/information-r13"
EVIDENCE = ROOT / "research/experiments/n02_v1_2/information_r13_20261019"
ARMS = ("r11-task850", "r12-one256")
STAGES = (
    "shallow_stride4",
    "deep_stride16",
    "spatial_fusion",
    "pre_pi_fixed_state",
    "pre_pi_actual",
)


def same_metrics(rebuilt, recorded):
    assert rebuilt == recorded, (rebuilt, recorded)


def scalar(metric, target):
    if target == "grippers":
        return np.mean([metric[s]["mean_cm"] for s in ("left_gripper", "right_gripper")])
    if target == "relative":
        return np.mean([metric[s]["mean_cm"] for s in ("object_minus_left", "object_minus_right")])
    if target == "openness":
        return metric["openness"]["balanced_accuracy"]
    return metric[target]["mean_cm"]


def comparison(arrays, records, stage1, stage2, panel, target):
    """Paired scene bootstrap of mean per-seed test-position errors, in cm."""
    ix = split_indices(records)["test"]

    def error(stage):
        preds = [arrays[f"{panel}-{stage}-mlp-{s}"][ix] for s in (13019, 13020, 13021)]
        y = arrays[panel + "_target"][ix]
        columns = {
            "object": [(0, 3)],
            "grippers": [(3, 6), (6, 9)],
            "relative": [(9, 12), (12, 15)],
        }
        if panel == "cached":
            columns["grippers"] = [(0, 3), (3, 6)]
        return np.mean(
            [
                np.linalg.norm(p[:, a:b] - y[:, a:b], axis=1) * 100
                for p in preds
                for a, b in columns[target]
            ],
            axis=0,
        )

    delta = error(stage2) - error(stage1)
    groups = np.array([r["group"] for r in records])[ix]
    group_means = np.array([delta[groups == g].mean() for g in np.unique(groups)])
    rng = np.random.default_rng(13019)
    boot = rng.choice(group_means, size=(2000, len(group_means)), replace=True).mean(1)
    return dict(
        from_stage=stage1,
        to_stage=stage2,
        target=target,
        mean_error_increase_cm=float(group_means.mean()),
        scene_bootstrap95_cm=np.quantile(boot, [0.025, 0.975]).tolist(),
        test_groups=len(group_means),
    )


def main():
    reports = {a: json.loads((EVIDENCE / (a + "_report.json")).read_text()) for a in ARMS}
    native = json.loads((EVIDENCE / "native_control_report.json").read_text())
    with np.load(OUT / "native_control_predictions.npz", allow_pickle=False) as saved:
        native_records = json.loads(str(saved["records"].item()))
        native_indices = split_indices(native_records)
        for row in native["rows"]:
            pred = saved[f"{row['head']}-{row['seed']}"]
            for split, ix in native_indices.items():
                same_metrics(
                    metrics(pred[ix], saved["target"][ix], "controlled"), row["metrics"][split]
                )
    checks, comparisons = {}, {}
    for arm, report in reports.items():
        assert report["policy_updates"] == 0 and report["closed_loops"] == 0
        assert report["frozen"]["model_sha_before"] == report["frozen"]["model_sha_after"]
        arrays = np.load(OUT / arm / "head_predictions.npz", allow_pickle=False)
        checks[arm] = dict(verified_rows=0, panels={})
        comparisons[arm] = {}
        for panel in ("controlled", "cached"):
            visual_stages = (*STAGES, "shallow_native", "deep_native")
            expected = {
                (stage, head, seed)
                for stage in visual_stages
                for head in ("mlp", "linear_bottleneck")
                for seed in (13019, 13020, 13021)
            }
            expected |= {(stage, "shuffled_mlp", 13019) for stage in visual_stages}
            expected |= {
                (stage, "mlp", seed)
                for stage in ("phase", "proprio")
                for seed in (13019, 13020, 13021)
            }
            expected.add(("training_mean", "constant", None))
            actual_rows = [r for r in report["rows"] if r["panel"] == panel]
            assert len(actual_rows) == len(expected)
            assert {(r["stage"], r["head"], r["seed"]) for r in actual_rows} == expected
            with np.load(OUT / arm / (panel + ".npz"), allow_pickle=False) as data:
                records = json.loads(str(data["records"].item()))
                y = data["y"]
            indices = split_indices(records)
            checks[arm]["panels"][panel] = {
                s: dict(samples=len(ix), groups=len({records[i]["group"] for i in ix}))
                for s, ix in indices.items()
            }
            for row in [r for r in report["rows"] if r["panel"] == panel]:
                if row["stage"] == "training_mean":
                    pred = np.broadcast_to(y[indices["train"]].mean(0), y.shape)
                else:
                    key = (
                        f"{panel}-{row['stage']}-shuffled"
                        if row["head"] == "shuffled_mlp"
                        else f"{panel}-{row['stage']}-{row['head']}-{row['seed']}"
                    )
                    pred = arrays[key]
                    assert row["updates"] == 600
                    assert row["selected_step"] in range(100, 601, 100)
                    assert row["val_normalized_mse"] == min(c["val_mse"] for c in row["curve"])
                for split, ix in indices.items():
                    rebuilt = metrics(pred[ix], y[ix], panel)
                    same_metrics(rebuilt, row["metrics"][split])
                checks[arm]["verified_rows"] += 1
            comparisons[arm][panel] = [
                comparison(arrays, records, st1, st2, panel, target)
                for st1, st2 in [
                    ("shallow_stride4", "deep_stride16"),
                    ("deep_stride16", "spatial_fusion"),
                    ("spatial_fusion", "pre_pi_fixed_state"),
                ]
                for target in (
                    ("object", "grippers", "relative") if panel == "controlled" else ("grippers",)
                )
            ]
        arrays.close()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    table = []
    for target, ax in zip(("object", "grippers", "relative", "openness"), axes.flat, strict=True):
        for arm, color, shift in zip(ARMS, ("#2469a4", "#c65e22"), (-0.07, 0.07), strict=True):
            report = reports[arm]
            means, stds = [], []
            for stage in STAGES:
                rows = [
                    r
                    for r in report["rows"]
                    if r["panel"] == "controlled" and r["stage"] == stage and r["head"] == "mlp"
                ]
                values = [scalar(r["metrics"]["test"], target) for r in rows]
                means.append(np.mean(values))
                stds.append(np.std(values))
                table.append(
                    dict(
                        arm=arm,
                        stage=stage,
                        target=target,
                        mean=float(np.mean(values)),
                        seed_std=float(np.std(values)),
                    )
                )
            ax.errorbar(
                np.arange(5) + shift,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                color=color,
                label=arm,
            )
            baseline = next(
                r
                for r in report["rows"]
                if r["panel"] == "controlled" and r["stage"] == "training_mean"
            )
            ax.axhline(scalar(baseline["metrics"]["test"], target), ls="--", color=color, alpha=0.5)
        native_values = [
            scalar(r["metrics"]["test"], target) for r in native["rows"] if r["head"] == "mlp"
        ]
        ax.axhline(np.mean(native_values), color="#397c35", ls="-.", label="Native RGB reference")
        ax.set_xticks(
            range(5), ["Shallow", "Deep", "Fusion", "Pre-pi\nfixed state", "Pre-pi\nactual state"]
        )
        ax.set_title(
            {
                "object": "Object position",
                "grippers": "Gripper TCP position",
                "relative": "Object-to-gripper displacement",
                "openness": "Gripper open/closed",
            }[target]
        )
        ax.set_ylabel("Balanced accuracy" if target == "openness" else "Mean 3D error (cm)")
        if target == "openness":
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Frozen RAW information probes: independent controlled test scenes\n"
        "Mean +/- SD across 3 small MLP seeds; dashed = training-mean predictor"
    )
    fig.savefig(EVIDENCE / "heldout_information.png", dpi=160)
    plt.close(fig)
    result = dict(
        policy_updates=0,
        closed_loops=0,
        independent_prediction_audit=checks,
        metric_recomputation="exact equality, same NumPy version as head fitting",
        numpy_version=np.__version__,
        heldout_controlled_mlp=table,
        paired_stage_comparisons=comparisons,
        native_rgb_control=[
            dict(
                target=t,
                mean=float(
                    np.mean(
                        [
                            scalar(r["metrics"]["test"], t)
                            for r in native["rows"]
                            if r["head"] == "mlp"
                        ]
                    )
                ),
            )
            for t in ("object", "grippers", "relative", "openness")
        ],
        interpretation=(
            "Small-readout recoverability; does not prove information absence or policy success"
        ),
    )
    (EVIDENCE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
