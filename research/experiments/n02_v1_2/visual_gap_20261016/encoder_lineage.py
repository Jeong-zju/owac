"""Encoder-only checkpoint lineage probe; no VLA backend, gradients or updates."""

import gc
import json
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.n02_full.model_v12 import build_n02_v12_model
from owac.frameworks.n02_full.sequence_v12 import window_inputs
from owac.runtime.n02_v12_robotwin import online_window
from owac.runtime.rawvla_full_eval import write_json
from owac.runtime.rawvla_robotwin import normalize_state


def main():
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    output = checkout / "outputs/n02-v12/visual-gap-20261016/encoder-lineage-a1.json"
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(
        (checkout / "configs/encoders/n02_v12/raw_repair_warm_r7_robotwin.json").read_text()
    )
    model = build_n02_v12_model(config["model"]).cuda().eval()
    encoder = model.inference.encoder
    data = np.load(
        checkout / "outputs/n02-v12/policy-audit-20261016/original-normal/initial_observation.npz"
    )
    stats = json.loads(Path(config["data"]["stats_path"]).read_text())["norm_stats"]["state"]
    raw = [data[f"raw_{i}"] for i in range(3)]
    bundles = {}
    for name, images in [("original", raw), ("horizontal_flip", [x[:, ::-1].copy() for x in raw])]:
        window = online_window(
            {
                "history": [{"step": 0, "image": images}],
                "step": 0,
                "case_id": "encoder-lineage",
                "lang": "Grab the light brown cylindrical roller on the table.",
                "normalized_state": normalize_state(data["state"], stats),
            },
            config["model"]["camera_ids"],
            history=8,
        )
        bundles[name] = window_inputs(window)[0].observation
    weights = {
        "r5_raw900": "raw-n02-robotwin-r5-s1103/checkpoint_0000900.pt",
        "r7_warm100": "raw-repair-warm-r7-s1103-a3/checkpoint_0000100.pt",
        "r7_fresh100": "raw-repair-fresh-r7-s1103-a3/checkpoint_0000100.pt",
    }
    results = {}
    with torch.inference_mode():
        for name, reference in weights.items():
            payload = torch.load(
                checkout / "outputs/n02-v12/P2" / reference, map_location="cpu", weights_only=False
            )
            prefix = "inference.encoder."
            encoder.load_state_dict(
                {k[len(prefix) :]: v for k, v in payload["model"].items() if k.startswith(prefix)},
                strict=True,
            )
            del payload
            gc.collect()
            tensors = {k: encoder(v) for k, v in bundles.items()}
            rows = {}
            for key, values in tensors["original"].levels.items():
                valid = tensors["original"].valid[key][0].bool()
                x = values[:, :, valid].float()
                y = tensors["horizontal_flip"].levels[key][:, :, valid].float()
                rows[key] = {
                    "valid_rms": float(x.square().mean().sqrt()),
                    "valid_spatial_centred_energy_fraction": float(
                        (x - x.mean(2, keepdim=True)).square().sum() / x.square().sum()
                    ),
                    "horizontal_flip_relative_l2": float((y - x).norm() / x.norm()),
                    "valid_cells": int(valid.sum()),
                }
            results[name] = rows
    write_json(
        output,
        {
            "optimizer_updates": 0,
            "encoder_only": True,
            "padding_excluded": True,
            "weights": weights,
            "results": results,
        },
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
