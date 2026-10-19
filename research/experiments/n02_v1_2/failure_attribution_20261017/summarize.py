"""Recalculate exposure and selected-checkpoint errors from existing receipts."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from owac.frameworks.n02_full.data import _action_mask_hash

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT.parent / "representation_r8_20261016/receipts"


def read(path):
    return json.loads(path.read_text())


def main():
    existing = read(ROOT / "existing_evidence.json")
    metadata = {m["relative_path"]: m for m in existing["cached_metadata"]}
    identity = read(RECEIPTS / "fresh/data_identity.json")
    count = identity["train"]["anchor_table"]["entries"]
    windows = read(RECEIPTS / "fixedfront-d1-narrow/dev_windows.json")["identities"]
    result = {
        "optimizer_updates": 0,
        "model_inferences": 0,
        "new_rollouts": 0,
        "train_anchor_count": count,
        "cache_lighting_counts": dict(Counter(m["lighting_domain"] for m in metadata.values())),
        "cache_noise_enabled_counts": dict(
            Counter(str(m["raw_sensor_noise_enabled"]) for m in metadata.values())
        ),
        "dev_windows": [],
        "arms": {},
    }
    for w in windows:
        meta = metadata[w["relative_path"]]
        indices = np.arange(w["action_start"], w["action_stop"])
        mask = (indices >= 0) & (indices < meta["num_frames"])
        digest = _action_mask_hash(
            w["relative_path"],
            action_start=w["action_start"],
            action_stop=w["action_stop"],
            num_frames=meta["num_frames"],
            mask=mask,
        )
        assert digest == w["action_mask_hash"]
        result["dev_windows"].append(
            {
                "episode": meta["episode"],
                "t": w["t"],
                "lighting": meta["lighting_domain"],
                "valid_actions": int(mask.sum()),
                "horizon": len(mask),
                "mask_hash_verified": True,
            }
        )
    valid = np.array([w["valid_actions"] for w in result["dev_windows"]])
    for arm, saved in existing["arms"].items():
        directory = RECEIPTS / f"fixedfront-d1-{arm}"
        baseline, final = (
            read(directory / "dev_baseline.json"),
            read(directory / "dev_sampled_latest.json"),
        )
        selected = saved["reference"]
        errors = {}
        for name, record in (("before", baseline), ("selected", selected), ("final", final)):
            per_window = np.array(record["sampled_per_window"])
            assert np.isclose(per_window.mean(), record["sampled_control_mse"])
            errors[name] = {
                "window_mean_mse": float(per_window.mean()),
                "per_valid_action_mse": float(np.average(per_window, weights=valid)),
                "initial_windows_mse": float(per_window[[0, 3]].mean()),
                "tail_windows_mse": float(per_window[[1, 2]].mean()),
                "per_window": per_window.tolist(),
                "velocity_mse_32": record["action_error"],
            }
        rows = [
            json.loads(line) for line in (directory / "optimization.jsonl").read_text().splitlines()
        ]
        exposure = {}
        for budget in sorted({25, 50, 100, saved["selected_step"]}):
            entries = [
                (anchor, path, t)
                for row in rows[:budget]
                for anchor, path, t in zip(
                    range(row["window"]["start"], row["window"]["stop"]),
                    row["window"]["anchors"],
                    row["window"]["current_frame_indices"],
                    strict=True,
                )
            ]
            exposure[budget] = {
                "anchors": len({x[0] for x in entries}),
                "equivalent_anchor_epochs": len(entries) / count,
                "episode_files": len({x[1] for x in entries}),
                "t0": sum(t == 0 for _, _, t in entries),
                "t_lt10": sum(t < 10 for _, _, t in entries),
                "lighting_anchors": dict(
                    Counter(metadata[p]["lighting_domain"] for _, p, _ in entries)
                ),
                "lighting_t_lt10": dict(
                    Counter(metadata[p]["lighting_domain"] for _, p, t in entries if t < 10)
                ),
            }
        result["arms"][arm] = {
            "selected_step": saved["selected_step"],
            "errors": errors,
            "exposure": exposure,
        }
    sources = [ROOT / "existing_evidence.json", RECEIPTS / "fresh/data_identity.json"]
    result["source_sha256"] = {
        str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    output = ROOT / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
