"""Independent S1 effect audit. Reads frozen dev cache; never trains or reads final.

Run with the encoder Python; --root is the remote n04-v1 directory.
Training GT supplies only the explicitly labelled static baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def transform(grid):
    a, b = grid["sensor_to_backbone_input"], grid["backbone_input_to_feature"]
    ma, mb = np.array(a["matrix"]), np.array(b["matrix"])
    return mb @ ma, mb @ np.array(a["translation"]) + b["translation"]


def lattice(grid):
    h, w = grid["feature_size_hw"]
    y, x = np.mgrid[:h, :w]
    xy = np.stack([x, y], -1)
    m, t = transform(grid)
    return (xy - t) @ np.linalg.inv(m).T


def sample(a, xy):
    x, y = np.rint(xy[..., 0]).astype(int), np.rint(xy[..., 1]).astype(int)
    h, w = a.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return a[y.clip(0, h - 1), x.clip(0, w - 1)], inside


def summary(rows):
    if not rows:
        return {}
    r = np.concatenate([a["residual"] for a in rows])
    v = np.concatenate([a["variance"] for a in rows])
    s = np.sqrt(v)
    a = np.abs(r)
    out = {
        "frames": len(rows),
        "pixels": len(r),
        "mae_m": float(a.mean()),
        "median_m": float(np.median(a)),
        "rmse_m": float(np.sqrt(np.mean(r**2))),
        "p90_m": float(np.quantile(a, 0.90)),
        "p95_m": float(np.quantile(a, 0.95)),
        "coverage_1sigma": float(np.mean(a <= s)),
        "coverage_2sigma": float(np.mean(a <= 2 * s)),
        "sigma_mean_m": float(s.mean()),
        "nll_no_constant_m": float(np.mean(0.5 * (r * r / v + np.log(v)))),
        "within_5mm": float(np.mean(a <= 0.005)),
        "within_10mm": float(np.mean(a <= 0.01)),
        "within_20mm": float(np.mean(a <= 0.02)),
        "within_50mm": float(np.mean(a <= 0.05)),
        "sigma_below_training_floor_fraction": float(np.mean(v < (0.8361701369285583**2) * 1e-4)),
        "training_ratio_clipped_fraction": float(
            np.mean(r * r / np.maximum(v, (0.8361701369285583**2) * 1e-4) > 100)
        ),
    }
    for key in ("template", "constant", "swapped"):
        values = [row[key] for row in rows if key in row]
        if values:
            out[key + "_mae_m"] = float(np.concatenate(values).mean())
    return out


def baseline(root, grid_by_camera):
    pools = defaultdict(list)
    sources = defaultdict(set)
    for p in sorted((root / "data/s0-paired-v1-20261014").glob("episode-*/record.json")):
        rec = json.loads(p.read_text())
        if rec["split"] != "train":
            continue
        platform = rec["metadata"]["platform"]
        names = rec["metadata"]["camera_names"]
        for frame in rec["frames"]:
            with np.load(p.parent / frame["file"]) as z:
                d, support = z["gt_depth_m"], z["gt_visible_support"].astype(bool)
                if "gt_geometry_support" in z.files:
                    support &= z["gt_geometry_support"].astype(bool)
                for c, name in enumerate(names):
                    label = platform + "/" + name
                    if label not in grid_by_camera:
                        continue
                    xy = lattice(grid_by_camera[label])
                    target, inside = sample(d[c], xy)
                    ok, _ = sample(support[c], xy)
                    good = inside & ok & np.isfinite(target) & (target > 0)
                    pools[label].append(np.where(good, target, np.nan))
                    sources[label].add(rec["source_trajectory_id"])
    manifest = json.loads((root / "runs/s0/completion-factors-20261014/manifest.json").read_text())
    for entry in manifest["sequences"]:
        if entry["split"] != "train":
            continue
        with np.load(root / "runs/s0/completion-factors-20261014" / entry["file"]) as z:
            depth = z["gt_depth_m"]
            for c in range(2):
                label = f"rendered_factor/camera_{c}"
                if label not in grid_by_camera:
                    continue
                xy = lattice(grid_by_camera[label])
                for frame in depth:
                    target, inside = sample(frame[c], xy)
                    pools[label].append(np.where(inside & (target > 0), target, np.nan))
                sources[label].add(entry["source_trajectory_id"])
    out = {}
    for label, arrays in pools.items():
        a = np.stack(arrays)
        constant = float(np.nanmedian(a))
        with np.errstate(all="ignore"):
            template = np.nanmedian(a, axis=0)
        out[label] = (np.nan_to_num(template, nan=constant), constant)
    return out, {k: sorted(v) for k, v in sources.items()}


def correspondence(ea, eb, ma, mb, ga, gb):
    # Independent geometric construction: camera A -> world -> camera B.
    xy = ma["candidate_xy_sensor"][0].astype(np.float64)
    d, inside = sample(ga["depth"], xy)
    support, _ = sample(ga["support"], xy)
    rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(ga["K"]).T
    pa = rays * d[:, None]
    t = np.linalg.inv(gb["T_W_from_C"]) @ ga["T_W_from_C"]
    pb = pa @ t[:3, :3].T + t[:3, 3]
    uvh = pb @ gb["K"].T
    uv = uvh[:, :2] / np.maximum(pb[:, 2:3], 1e-8)
    db, inside_b = sample(gb["depth"], uv)
    sb, _ = sample(gb["support"], uv)
    keep = inside & inside_b & (support > 0) & (sb > 0) & (pb[:, 2] > 0)
    keep &= ma["candidate_valid"][0].astype(bool) & (np.abs(db - pb[:, 2]) <= 0.05)
    if keep.sum() < 2:
        return None
    grid = json.loads(str(mb["sample_grid_json"]))
    m, tr = transform(grid)
    fxy = uv[keep] @ m.T + tr
    h, w = grid["feature_size_hw"]
    domain = (fxy[:, 0] >= 0) & (fxy[:, 0] <= w - 1) & (fxy[:, 1] >= 0) & (fxy[:, 1] <= h - 1)
    indices = np.flatnonzero(keep)[domain]
    if len(indices) < 2:
        return None
    fxy = fxy[domain]
    g = torch.tensor(
        np.stack([2 * (fxy[:, 0] + 0.5) / w - 1, 2 * (fxy[:, 1] + 0.5) / h - 1], -1),
        dtype=torch.float32,
    )
    b = F.grid_sample(
        torch.tensor(mb["latent_mean_map"]),
        g[None, None],
        align_corners=False,
        padding_mode="zeros",
    )[0, :, 0].T
    a = torch.tensor(ma["candidate_descriptor"][0, indices])
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    sim = (a @ b.T).numpy()
    n = len(indices)
    diag = np.arange(n)
    best = sim.argmax(1)
    pixel_error = np.linalg.norm(uv[indices][best] - uv[indices], axis=1)
    return {
        "platform": ea["platform"],
        "source": ea["source_trajectory_id"],
        "variant": ea["variant"],
        "camera_pair": ea["camera_id"] + "->" + eb["camera_id"],
        "frame": ea["frame_index"],
        "pairs": n,
        "positive_cosine": float(sim[diag, diag].mean()),
        "cyclic_negative_cosine": float(sim[diag, (diag + 1) % n].mean()),
        "top1": float(np.mean(best == diag)),
        "chance": 1 / n,
        "pck_4px": float(np.mean(pixel_error <= 4)),
        "pck_8px": float(np.mean(pixel_error <= 8)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cache = args.root / "runs/s1/cache-final-step50000"
    index = json.loads((cache / "index.json").read_text())
    entries = index["entries"]
    loaded, rows, grids, groups, factors = {}, [], {}, defaultdict(list), defaultdict(list)
    for e in entries:
        mp, gp = cache / e["measurement_path"], cache / e["gt_path"]
        assert sha(mp) == e["measurement_sha256"]
        assert sha(gp) == e["gt_sha256"]
        z, g = load(mp), load(gp)
        assert json.loads(str(z["measurement_ids_json"])) == [e["measurement_id"]]
        grid = json.loads(str(z["sample_grid_json"]))
        label = e["platform"] + "/" + e["camera_id"]
        grids[label] = grid
        xy = lattice(grid)
        d, inside = sample(g["depth"], xy)
        support, _ = sample(g["support"], xy)
        mask = inside & (support > 0) & np.isfinite(d) & (d > 0) & z["valid"][0, 0].astype(bool)
        row = {
            "residual": (z["depth_mean"][0, 0] - d)[mask],
            "variance": z["depth_variance"][0, 0][mask],
            "mask": mask,
            "depth": d,
            "entry": e,
        }
        rows.append(row)
        groups[label].append(row)
        factors[e["platform"] + "/" + e["variant"]].append(row)
        # Keep only small depth/GT arrays plus correspondence frames 0,12,23.
        if e["frame_index"] not in (0, 12, 23):
            z = {k: v for k, v in z.items() if k in ("depth_mean",)}
        loaded[(e["source_path"], e["frame_index"], e["camera_id"])] = (e, z, g)
    print("cache hashes verified", len(entries), flush=True)
    base, train_sources = baseline(args.root, grids)
    for label, rr in groups.items():
        template, constant = base[label]
        for row in rr:
            row["template"] = np.abs(template - row["depth"])[row["mask"]]
            row["constant"] = np.abs(constant - row["depth"])[row["mask"]]
            e = row["entry"]
            other = next(
                (
                    r
                    for r in rr
                    if r["entry"]["source_trajectory_id"] != e["source_trajectory_id"]
                    and r["entry"]["variant"] == e["variant"]
                ),
                None,
            )
            if other:
                oe = other["entry"]
                pred = loaded[(oe["source_path"], oe["frame_index"], oe["camera_id"])][1][
                    "depth_mean"
                ][0, 0]
                row["swapped"] = np.abs(pred - row["depth"])[row["mask"]]
    pairs = {
        "libero": [("agentview", "robot0_eye_in_hand")],
        "robotwin2": [("front_camera", "head_camera"), ("left_camera", "front_camera")],
        "rendered_factor": [("camera_0", "camera_1")],
    }
    corrs = []
    for (path, frame, camera), (ea, ma, ga) in loaded.items():
        if frame not in (0, 12, 23):
            continue
        for ca, cb in pairs[ea["platform"]]:
            if camera != ca or (path, frame, cb) not in loaded:
                continue
            eb, mb, gb = loaded[(path, frame, cb)]
            result = correspondence(ea, eb, ma, mb, ga, gb)
            if result:
                corrs.append(result)
    cg = defaultdict(list)
    for c in corrs:
        cg[c["platform"] + "/" + c["camera_pair"]].append(c)
    corr_summary = {}
    for label, rr in cg.items():
        corr_summary[label] = {"frame_pairs": len(rr), "points": sum(r["pairs"] for r in rr)}
        for key in (
            "positive_cosine",
            "cyclic_negative_cosine",
            "top1",
            "chance",
            "pck_4px",
            "pck_8px",
        ):
            corr_summary[label][key] = float(
                np.average([r[key] for r in rr], weights=[r["pairs"] for r in rr])
            )
    history = json.loads((args.root / "runs/s1/train-seed17/metrics.json").read_text())
    report = {
        "checkpoint_step": index["checkpoint_step"],
        "weights_sha256": index["weights_sha256"],
        "cache_index_sha256": sha(cache / "index.json"),
        "cache_hash_files_checked": len(entries) * 2,
        "protocol": (
            "unique dev camera/frame; metrics from cached outputs; "
            "train-only static per-camera median baselines; "
            "GT depth + pose correspondences on frames 0/12/23, 5cm visibility tolerance; "
            "top1 among visible projected points, not dense retrieval"
        ),
        "overall_unique": summary(rows),
        "by_camera": {k: summary(v) for k, v in groups.items()},
        "by_factor": {k: summary(v) for k, v in factors.items()},
        "train_sources": train_sources,
        "dev_sources": sorted({e["source_trajectory_id"] for e in entries}),
        "correspondence_summary": corr_summary,
        "correspondence_rows": corrs,
        "curve": {
            "evals": len(history),
            "last_step": history[-1]["step"],
            "best": min(history, key=lambda r: r["depth_mae_m"]),
            "last10": [
                {k: r[k] for k in ("step", "depth_mae_m", "depth_nll", "coverage_1sigma")}
                for r in history[-10:]
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k in ("overall_unique", "by_camera", "correspondence_summary")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
