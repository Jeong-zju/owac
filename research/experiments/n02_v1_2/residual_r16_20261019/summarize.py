"""Independently verify saved weights, predictions, bounds, selection and sampling."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_full.action_learning import sample_order, selected_splits
from owac.evaluation.n02_full.action_residual import BoundedActionResidual
from owac.frameworks.n02_full.task_chain import SmallActionHead, action_metrics, state_digest

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/n02-v12/residual-r16"
PARENT = ROOT / "outputs/n02-v12/learning-r15"


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = torch.load(PARENT / "dataset.pt", weights_only=False)
    selection = json.loads((EVIDENCE.parent / "learning_r15_20261019/selection.json").read_text())
    splits = selected_splits(data["records"], selection["train_groups"]["5"])
    config = json.loads((EVIDENCE.parent / "training_r11_20261017/task_config.json").read_text())
    std = json.loads(Path(config["data"]["stats_path"]).read_text())["norm_stats"]["actions"]["std"]
    datasets = sha(PARENT / "dataset.pt")
    features_sha = {m: sha(PARENT / f"{m}_tokens.pt") for m in ("raw", "native")}
    reports, audits, initial = [], [], {}
    state = torch.as_tensor(data["state"]).float().cuda()
    for path in sorted(OUT.glob("*/report.json")):
        r = json.loads(path.read_text())
        name, mode, seed = r["run_id"], r["modality"], r["seed"]
        if (
            not r["completed"]
            or r["actual_updates"] != 1024
            or r["train_indices"] != splits["train"]
        ):
            raise ValueError("incomplete or incorrect data exposure")
        source = r["source"]
        assert source["dataset_sha256"] == datasets
        assert source["feature_sha256"] == features_sha["native" if mode == "native" else "raw"]
        assert sha(Path(r["baseline_source"]["checkpoint"])) == r["baseline_source"]["sha256"]
        assert r["baseline_sha"] == r["baseline_sha_after"]
        initial.setdefault(seed, (r["initial_sha"], r["parameters"]))
        assert initial[seed] == (r["initial_sha"], r["parameters"])
        hist = [json.loads(s) for s in (path.parent / "history.jsonl").read_text().splitlines()]
        assert [h["step"] for h in hist] == list(range(1, 1025))
        assert all(np.isfinite(h["loss"]) and np.isfinite(h["gradient_norm"]) for h in hist)
        assert json.loads((path.parent / "sample_order.json").read_text()) == sample_order(
            splits["train"], 1024, seed
        )
        x = (
            torch.load(
                PARENT / ("native_tokens.pt" if mode == "native" else "raw_tokens.pt"),
                weights_only=False,
            )
            .float()
            .cuda()
        )
        if mode == "state":
            x.zero_()
        head = BoundedActionResidual().cuda().eval()
        baseline = SmallActionHead().cuda().eval()
        saved_base = torch.load(r["baseline_source"]["checkpoint"], weights_only=False)
        baseline.load_state_dict(saved_base["selected"], strict=True)
        assert state_digest(baseline) == r["baseline_sha"]
        predictions = np.load(path.parent / "predictions.npz")
        old = np.load(PARENT / f"state-5-s{seed}/predictions.npz")

        def metric(p, indices):
            return action_metrics(
                p, np.asarray(data["actions"])[indices], np.asarray(data["mask"])[indices], std
            )

        def enrich(value, p):
            indices = splits["test"]
            for light, label in (("Normal", "normal_test"), ("Over", "over_test")):
                local = [j for j, i in enumerate(indices) if data["records"][i]["light"] == light]
                value[label] = metric(p[local], np.asarray(indices)[local])
            value["by_episode"] = {}
            for group in sorted({data["records"][i]["group"] for i in indices}):
                local = [j for j, i in enumerate(indices) if data["records"][i]["group"] == group]
                value["by_episode"][group] = metric(p[local], np.asarray(indices)[local])

        for split, indices in splits.items():
            b = predictions["baseline_" + split]
            np.testing.assert_array_equal(b, old["1024_selected_" + split][0])
            assert metric(b, indices) == r["baseline"][split]
            with torch.no_grad():
                live = torch.cat(
                    [
                        baseline(torch.zeros_like(x[c]), state[c])[..., :14].cpu()
                        for c in [indices[i : i + 8] for i in range(0, len(indices), 8)]
                    ]
                ).numpy()
            np.testing.assert_array_equal(live, b)
        enrich(r["baseline"], predictions["baseline_test"])
        for row in r["rows"]:
            candidates = [
                (h["step"], c["gate"], c["val"]["normalized_control_mse"])
                for h in r["history"]
                if h["step"] <= row["budget"]
                for c in h["candidates"]
            ]
            expected = min(candidates, key=lambda c: c[2])
            assert expected == (
                row["selected_step"],
                row["selected_gate"],
                row["selected_validation_mse"],
            )
            assert expected[2] <= r["baseline"]["val"]["normalized_control_mse"]
            saved = torch.load(
                path.parent / f"checkpoint_{row['budget']:04}.pt", weights_only=False
            )
            for version, gate in (("selected", row["selected_gate"]), ("last", 1.0)):
                head.load_state_dict(saved[version], strict=True)
                assert state_digest(head) == row[version + "_sha"]
                for split, indices in splits.items():
                    prefix = f"{row['budget']}_{version}_"
                    p, d = predictions[prefix + split], predictions[prefix + "delta_" + split]
                    assert np.isfinite(d).all() and np.max(np.abs(d)) <= 1
                    b = predictions["baseline_" + split]
                    np.testing.assert_array_equal(p, b if gate == 0 else b + gate * d)
                    assert metric(p, indices) == row[version][split]
                    with torch.no_grad():
                        live = torch.cat(
                            [
                                head(x[c], state[c])[..., :14].cpu()
                                for c in [indices[i : i + 8] for i in range(0, len(indices), 8)]
                            ]
                        ).numpy()
                    np.testing.assert_array_equal(live, d)
                enrich(row[version], predictions[f"{row['budget']}_{version}_test"])
        audits.append(
            dict(
                run_id=name,
                actual_updates=len(hist),
                exact_weight_reload=True,
                all_prediction_metrics_exact=True,
                baseline_r15_exact=True,
                bounds_valid=True,
                validation_only_checkpoint_and_gate=True,
                paired_training_order=True,
                feature_and_dataset_hashes_unchanged=True,
            )
        )
        reports.append(r)
        write(EVIDENCE / (name + "_report.json"), r)
        del x, head, baseline
        torch.cuda.empty_cache()
    assert len(reports) == 6 and sum(r["actual_updates"] for r in reports) == 6144
    aggregate = []
    for mode in ("state", "raw", "native"):
        rs = [r for r in reports if r["modality"] == mode]
        assert len(rs) == 2
        for budget in (256, 512, 1024):
            points = [next(p for p in r["rows"] if p["budget"] == budget) for r in rs]
            row = dict(
                modality=mode,
                budget=budget,
                seeds=[r["seed"] for r in rs],
                selected_steps=[p["selected_step"] for p in points],
                selected_gates=[p["selected_gate"] for p in points],
            )
            for version in ("baseline", "selected", "last"):
                values = (
                    [r["baseline"] for r in rs]
                    if version == "baseline"
                    else [p[version] for p in points]
                )
                row[version] = {
                    s: {k: float(np.mean([v[s][k] for v in values])) for k in values[0][s]}
                    for s in ("train", "val", "test", "normal_test", "over_test")
                }
            row["normal_joint_improvement_percent"] = 100 * (
                1
                - row["selected"]["normal_test"]["joint_rms_deg"]
                / row["baseline"]["normal_test"]["joint_rms_deg"]
            )
            aggregate.append(row)
    summary = dict(
        protocol="n02-state-baseline-visual-residual-r16-v1",
        audited=True,
        completed_runs=6,
        planned_runs=6,
        actual_updates=6144,
        encoder_updates=0,
        pi_updates=0,
        closed_loops=0,
        split_counts={k: len(v) for k, v in splits.items()},
        aggregate=aggregate,
        audits=audits,
        reports=reports,
    )
    write(EVIDENCE / "summary.json", summary)
    write(
        OUT / "progress.json", dict(reports=reports, completed_runs=6, planned_runs=6, audited=True)
    )
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "modality",
                        "budget",
                        "selected_steps",
                        "selected_gates",
                        "normal_joint_improvement_percent",
                    )
                }
                | dict(
                    normal=r["selected"]["normal_test"]["joint_rms_deg"],
                    over=r["selected"]["over_test"]["joint_rms_deg"],
                    train=r["selected"]["train"]["joint_rms_deg"],
                    val=r["selected"]["val"]["joint_rms_deg"],
                    last_train=r["last"]["train"]["joint_rms_deg"],
                )
                for r in aggregate
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
