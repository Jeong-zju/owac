"""Audit frozen matched-measurement evidence and export portable tensor caches."""

import json
import statistics
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

import __main__
from owac.baselines.n04_tracking import RKNPositionTracker
from owac.evaluation.n04_tracking import (
    CachedScene,
    digest,
    observation_inputs,
    save_json,
    tensor_hash,
)

ROOT = Path("outputs/n04-matched-tracking-20261013-a1")
RAW = Path("outputs/n04-independent-raw-diagnostics-20261013-b1")
ARCHIVE = Path("research/experiments/results/2026-10-13_n04_matched_tracking")


def main():
    torch.set_num_threads(1)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    freeze = json.loads((ROOT / "freeze.json").read_text())
    assert protocol["complete"] and protocol["dynamic_benefit"]
    for path, expected in protocol["source_sha256"].items():
        assert digest(Path(path)) == expected
    assert digest(Path(protocol["source_checkpoint"])) == protocol["source_checkpoint_sha256"]
    for seed, expected in freeze["rkn_checkpoint_sha256"].items():
        assert digest(ROOT / f"rkn-{seed}.pt") == expected
    # -m stored this dataclass as __main__.CachedScene; export standard dictionaries
    # for consumers so they never need this compatibility binding.
    __main__.CachedScene = CachedScene
    scenes = torch.load(ROOT / "final-cache.pt", weights_only=False)
    development = torch.load(ROOT / "development-cache.pt", weights_only=False)
    portable = {key: [vars(s) for s in value] for key, value in development.items()}
    portable["final"] = [vars(s) for s in scenes]
    torch.save(portable, ROOT / "portable-cache.pt")
    predictions = torch.load(ROOT / "predictions.pt", weights_only=True)
    rows = json.loads((ROOT / "final-matrix.json").read_text())
    hashes = {}
    for scene in scenes:
        for duration in (0, 2, 4, 6):
            z, valid = observation_inputs(scene, duration)
            key = (scene.seed, duration)
            expected = tensor_hash((z, valid))
            part = [r for r in rows if (r["scene_seed"], r["missing_frames"]) == key]
            assert len(part) == 5 and {r["input_sha256"] for r in part} == {expected}
            hashes[str(key)] = expected
            for row in part:
                output = next(
                    r for r in predictions[row["method"]] if (r["seed"], r["duration"]) == key
                )
                error = (output["position"] - scene.target).norm(dim=-1)
                assert torch.equal(error, output["error"])
                assert abs(float(error[2:].mean()) - row["mean_error_px"]) < 1e-7
    assert not any((~s.detected).any() for s in scenes)
    summary = json.loads((ROOT / "summary.json").read_text())
    rng = np.random.default_rng(695000)
    bootstrap = {}
    for method in summary:
        if method == "unfiltered_hold":
            continue
        differences = []
        for scene in scenes:
            if scene.kind not in ("object_cv", "mixed_cv"):
                continue
            reference = statistics.fmean(
                r["hidden_error_px"]
                for r in rows
                if r["scene_seed"] == scene.seed
                and r["method"] == "unfiltered_hold"
                and r["missing_frames"]
            )
            candidate = statistics.fmean(
                r["hidden_error_px"]
                for r in rows
                if r["scene_seed"] == scene.seed and r["method"] == method and r["missing_frames"]
            )
            differences.append(reference - candidate)
        values = np.asarray(differences)
        samples = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
        bootstrap[method] = dict(
            independent_scene_groups=len(values),
            positive_groups=int((values > 0).sum()),
            reduction_px_95pct_interval=np.quantile(samples, [0.025, 0.975]).tolist(),
        )
    latent = []
    variance = torch.tensor(protocol["measurement_variance_px2"])
    for seed in (0, 1, 2):
        model = RKNPositionTracker(variance)
        model.load_state_dict(torch.load(ROOT / f"rkn-{seed}.pt", weights_only=True))
        model.eval()
        for scene in scenes:
            if scene.kind not in ("object_cv", "mixed_cv"):
                continue
            z, valid = observation_inputs(scene, 4)
            with torch.no_grad():
                normal = model(z, valid).position
                index = 0

                def reset_latent(module, args):
                    nonlocal index
                    at_gap = index == 10
                    index += 1
                    if at_gap:
                        prior = args[0].clone()
                        prior[:, 2:] = 0
                        return (prior, *args[1:])

                hook = model.cell.register_forward_pre_hook(reset_latent)
                try:
                    reset = model(z, valid).position
                finally:
                    hook.remove()
            assert torch.equal(normal[:10], reset[:10])
            latent.append(
                dict(
                    run=seed,
                    scene_seed=scene.seed,
                    normal_hidden_px=float(
                        (normal[10:14] - scene.target[10:14]).norm(dim=-1).mean()
                    ),
                    reset_hidden_px=float((reset[10:14] - scene.target[10:14]).norm(dim=-1).mean()),
                    maximum_position_difference=float((normal[10:14] - reset[10:14]).abs().max()),
                )
            )
    motion = json.loads((RAW / "motion-final.json").read_text())
    motion_summary = {}
    for condition in ("static", "camera_only", "object_only", "both"):
        motion_summary[condition] = {}
        for method in ("learned_ego", "truth_ego", "truth_ego_object"):
            part = [r for r in motion if r["condition"] == condition and r["method"] == method]
            motion_summary[condition][method] = {
                key: statistics.fmean(r[key] for r in part)
                for key in ("foreground_rmse", "background_rmse", "normalized_residual")
            }
    evidence = dict(
        complete=True,
        source_hashes_verified=True,
        checkpoint_hashes_verified=True,
        paired_input_groups=len(hashes),
        methods_per_group=5,
        final_rows=len(rows),
        detector_missing=0,
        final_frames=sum(s.detected.numel() for s in scenes),
        source_versions={n: version(n) for n in ("torch", "numpy", "filterpy", "scipy")},
        paired_scene_bootstrap=bootstrap,
        latent_interventions=latent,
        note=(
            "Scene-group bootstrap, not independent frames/pixels; "
            "three RKN initializations share one frozen sensor."
        ),
    )
    save_json(ARCHIVE / "audit.json", evidence)
    save_json(ARCHIVE / "tracking-summary.json", summary)
    save_json(ARCHIVE / "noise-summary.json", json.loads((RAW / "noise-summary.json").read_text()))
    save_json(ARCHIVE / "motion-summary.json", motion_summary)
    save_json(ARCHIVE / "protocol.json", protocol)
    save_json(ARCHIVE / "raw-protocol.json", json.loads((RAW / "protocol.json").read_text()))
    for seed in (0, 1, 2):
        part = [r for r in latent if r["run"] == seed]
        print(
            "latent",
            seed,
            statistics.fmean(r["normal_hidden_px"] for r in part),
            statistics.fmean(r["reset_hidden_px"] for r in part),
        )
    print("verified", len(rows), "rows;", len(hashes), "identical input groups")


if __name__ == "__main__":
    main()
