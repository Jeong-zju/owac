"""Independent source alignment, train exposure, gate selection and weight reload audit."""

import json
import runpy
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_full.action_learning import sample_order, selected_splits
from owac.evaluation.n02_full.action_residual import BoundedActionResidual
from owac.frameworks.n02_full.task_chain import SmallActionHead, action_metrics, state_digest

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/n02-v12/residual-joint-r17"
HELPERS = runpy.run_path(str(HERE / "run.py"))


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = torch.load(OUT / "dataset.pt", weights_only=False)
    selection = json.loads((HERE / "selection.json").read_text())
    stats = HELPERS["norm_stats"]()
    std = np.asarray(stats["actions"]["std"][:14])
    mean = np.asarray(stats["actions"]["mean"][:14])
    sm = np.asarray(stats["state"]["mean"][:14])
    ss = np.asarray(stats["state"]["std"][:14])
    splits = selected_splits(data["records"], selection["train_groups"])
    paired_rgb = torch.load(OUT / "rgb.pt", weights_only=False).numpy()
    # Read original episodes independently from prediction reports and indices.
    for source in selection["sources"]:
        assert source["accepted"] and HELPERS["sha"](Path(source["path"])) == source["sha256"]
        with np.load(source["path"]) as a:
            q, raw, rgb = a["state"], a["raw"], a["rgb"]
            for i, r in enumerate(data["records"]):
                if r["scene"] != source["scene"]:
                    continue
                t = r["t"]
                n = min(50, len(q) - t - 1)
                np.testing.assert_array_equal(
                    data["state"][i], ((q[t] - sm) / (ss + 1e-8)).astype(np.float32)
                )
                np.testing.assert_array_equal(
                    data["actions"][i, :n],
                    ((q[t + 1 : t + 1 + n] - mean) / (std + 1e-8)).astype(np.float32),
                )
                assert (
                    data["mask"][i].sum() == n
                    and data["mask"][i, :n].all()
                    and not data["mask"][i, n:].any()
                )
                np.testing.assert_array_equal(data["raw"][i], raw[t].transpose(0, 3, 1, 2))
                np.testing.assert_array_equal(paired_rgb[i], rgb[t])
    states = torch.tensor(data["state"]).float().cuda()
    initial = {}
    reports = []
    audits = []
    for file in sorted(OUT.glob("*/report.json")):
        r = json.loads(file.read_text())
        mode = r["modality"]
        seed = r["seed"]
        steps = 1024 if mode == "baseline" else 512
        history = [json.loads(s) for s in (file.parent / "history.jsonl").read_text().splitlines()]
        assert [h["step"] for h in history] == list(range(1, steps + 1))
        assert all(np.isfinite(h["loss"]) for h in history)
        assert r["train_indices"] == splits["train"]
        assert json.loads((file.parent / "sample_order.json").read_text()) == sample_order(
            splits["train"], steps, seed
        )
        if mode != "baseline":
            initial.setdefault(seed, r["initial_sha"])
            assert initial[seed] == r["initial_sha"]
            assert r["baseline_sha"] == r["baseline_sha_after"]
        features = torch.load(
            OUT / ("native_tokens.pt" if mode == "native" else "raw_tokens.pt"), weights_only=False
        ).cuda()
        if mode in ("baseline", "state"):
            features.zero_()
        head = (SmallActionHead() if mode == "baseline" else BoundedActionResidual()).cuda().eval()
        spatial = None
        if mode == "joint":
            spatial = HELPERS["load_spatial"]("audit-" + r["run_id"])
            params = dict(spatial.named_parameters())
            assert (
                r["spatial_sha_before"] != r["spatial_sha_after"]
                and r["initial_cache_parity"] <= 1e-5
            )
        saved_predictions = np.load(file.parent / "predictions.npz")
        if mode != "baseline":
            r["bounded_oracle_diagnostic"] = {}
            for split, indices in splits.items():
                base = saved_predictions["baseline_" + split]
                target = np.asarray(data["actions"])[indices]
                valid = np.asarray(data["mask"])[indices]
                optimal = base + np.clip(target - base, -1, 1)
                r["bounded_oracle_diagnostic"][split] = dict(
                    metrics=action_metrics(optimal, target, valid, std),
                    cap_binding_fraction=float((np.abs(target - base)[valid] > 1).mean()),
                    scope="privileged diagnostic bound; never a learned or runtime policy",
                )
        for row in r["rows"]:
            saved = torch.load(
                file.parent / f"checkpoint_{row['budget']:04}.pt", weights_only=False
            )
            if mode == "baseline":
                eligible = [h for h in r["history"] if h["step"] <= row["budget"]]
                best = min(eligible, key=lambda h: h["val"]["normalized_control_mse"])
                assert best["step"] == row["selected_step"]
            else:
                eligible = [
                    (h["step"], c["gate"], c["val"]["normalized_control_mse"])
                    for h in r["history"]
                    if h["step"] <= row["budget"]
                    for c in h["candidates"]
                ]
                assert min(eligible, key=lambda c: c[2]) == (
                    row["selected_step"],
                    row["selected_gate"],
                    row["selected_validation_mse"],
                )
            for version in ("selected", "last"):
                head.load_state_dict(saved[version], strict=True)
                assert state_digest(head) == row[version + "_sha"]
                if spatial is not None:
                    with torch.no_grad():
                        for name, value in saved[version + "_features"].items():
                            params[name].copy_(value)
                for split, indices in splits.items():
                    prefix = f"{row['budget']}_{version}_"
                    p = saved_predictions[prefix + split]
                    original = p[0] if mode == "baseline" else p
                    metric = action_metrics(
                        original,
                        np.asarray(data["actions"])[indices],
                        np.asarray(data["mask"])[indices],
                        std,
                    )
                    assert metric == row[version][split]
                    by_scene = {}
                    for group in sorted({data["records"][i]["group"] for i in indices}):
                        local = [
                            j for j, i in enumerate(indices) if data["records"][i]["group"] == group
                        ]
                        ix = np.asarray(indices)[local]
                        by_scene[group] = action_metrics(
                            original[local],
                            np.asarray(data["actions"])[ix],
                            np.asarray(data["mask"])[ix],
                            std,
                        )
                    row[version].setdefault("by_scene", {})[split] = by_scene
                    home = [j for j, i in enumerate(indices) if data["records"][i]["t"] == 0]
                    if home:
                        ix = np.asarray(indices)[home]
                        row[version].setdefault("home_only", {})[split] = action_metrics(
                            original[home],
                            np.asarray(data["actions"])[ix],
                            np.asarray(data["mask"])[ix],
                            std,
                        )
                        if split == "test":
                            row[version]["home_test"] = row[version]["home_only"][split]
                    with torch.no_grad():
                        chunks = []
                        for i in range(0, len(indices), 8):
                            ix = indices[i : i + 8]
                            x = (
                                features[ix]
                                if spatial is None
                                else HELPERS["spatial_values"](
                                    spatial, data["raw"][ix], data["state"][ix], data["cameras"]
                                )
                            )
                            chunks.append(head(x, states[ix])[..., :14].cpu())
                        live = torch.cat(chunks).numpy()
                    if mode == "baseline":
                        np.testing.assert_array_equal(live, original)
                    else:
                        delta = saved_predictions[prefix + "delta_" + split]
                        np.testing.assert_array_equal(live, delta)
                        assert np.max(np.abs(delta)) <= 1
                        base = saved_predictions["baseline_" + split]
                        gate = row["selected_gate"] if version == "selected" else 1
                        np.testing.assert_array_equal(p, base if gate == 0 else base + gate * delta)
                        with np.load(OUT / f"baseline-s{seed}/predictions.npz") as b:
                            np.testing.assert_array_equal(base, b["1024_selected_" + split][0])
        audits.append(
            dict(
                run_id=r["run_id"],
                actual_updates=steps,
                source_alignment=True,
                paired_sampling=True,
                validation_only_selection=True,
                exact_weight_reload=True,
            )
        )
        reports.append(r)
        del spatial, head, features
        torch.cuda.empty_cache()
    assert len(reports) == 10 and sum(a["actual_updates"] for a in audits) == 6144
    rejected_updates = 0
    for seed in (17019, 17020):
        rejected = OUT / "rejected" / f"joint-s{seed}-a1-baseline-drift"
        rows = [json.loads(s) for s in (rejected / "history.jsonl").read_text().splitlines()]
        assert [h["step"] for h in rows] == list(range(1, 513))
        rejected_updates += len(rows)
    assert rejected_updates == 1024
    aggregate = []
    for mode in ("baseline", "state", "raw", "native", "joint"):
        runs = [r for r in reports if r["modality"] == mode]
        points = [r["rows"][-1] for r in runs]
        values = dict(
            modality=mode,
            seeds=[r["seed"] for r in runs],
            selected_steps=[r["selected_step"] for r in points],
            selected_gates=[r.get("selected_gate") for r in points],
        )
        for version in ("selected", "last"):
            values[version] = {
                split: {
                    key: float(np.mean([r[version][split][key] for r in points]))
                    for key in points[0][version][split]
                }
                for split in ("train", "val", "test")
            }
            values[version + "_home"] = {
                split: {
                    key: float(np.mean([r[version]["home_only"][split][key] for r in points]))
                    for key in points[0][version]["home_only"][split]
                }
                for split in ("train", "val", "test")
            }
        aggregate.append(values)
    output = dict(
        audited=True,
        completed_runs=10,
        actual_updates=7168,
        accepted_matrix_updates=6144,
        rejected_joint_updates=1024,
        scene_counts=selection["accepted_scene_counts"],
        attempted_scenes=20,
        excluded_scenes=selection["excluded_scenes"],
        audits=audits,
        aggregate=aggregate,
        reports=reports,
        learned_policy_closed_loops=0,
    )
    HELPERS["write"](HERE / "summary.json", output)
    HELPERS["write"](
        OUT / "progress.json",
        dict(reports=reports, completed_runs=10, planned_runs=10, audited=True),
    )


if __name__ == "__main__":
    main()
