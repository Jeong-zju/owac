"""Compare cached RAW and live RAW units/lighting without model updates."""

import json
from pathlib import Path

import numpy as np


def stats(x):
    x = np.asarray(x)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "quantiles": np.quantile(x, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        "channel_mean": x.mean((0, 1)).tolist(),
    }


def main():
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    root = Path(
        "/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data/robotwin2/raw/grab_roller"
    )
    metadata = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            row = json.loads(str(data["metadata_json"]))
        metadata.append({"path": str(path), "metadata": row})
    normal = [row for row in metadata if row["metadata"]["lighting_domain"] == "Normal"][:3]
    selected = []
    for row in normal:
        with np.load(row["path"], allow_pickle=False) as data:
            row["first_frame"] = {
                name: stats(data[name + "_camera_raw_float32"][0])
                for name in ("head", "left", "right")
            }
            row["state0"] = data["actions"][0].tolist()
        selected.append(row)
    data = np.load(
        checkout / "outputs/n02-v12/policy-audit-20261016/original-normal/initial_observation.npz"
    )
    live = {
        "state": data["state"].tolist(),
        "cameras": {str(i): stats(data[f"raw_{i}"]) for i in range(3)},
    }
    result = {
        "selection": "first three Normal episodes in filename order; only first physical frame",
        "metadata_inventory": metadata,
        "selected": selected,
        "live": live,
        "optimizer_updates": 0,
    }
    path = checkout / "outputs/n02-v12/visual-gap-20261016/data_probe.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": [
                    {
                        "path": r["path"],
                        "ev": r["metadata"]["lighting_ev"],
                        "head": r["first_frame"]["head"],
                    }
                    for r in selected
                ],
                "live": live,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
