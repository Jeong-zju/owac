"""Locate image-dependent variation before/after the RAW spatial readout, no updates."""

import json
from pathlib import Path

import numpy as np
import torch

from owac.runtime.n02_v12_robotwin import Policy, online_window
from owac.runtime.rawvla_full_eval import write_json
from owac.runtime.rawvla_robotwin import normalize_state


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20))


def summary(x):
    x = x.float()
    centred = x - x.mean(1, keepdim=True)
    fraction = centred.square().sum() / x.square().sum().clamp_min(1e-20)
    return {
        "rms": float(x.square().mean().sqrt()),
        "spatial_centred_energy_fraction": float(fraction),
        "camera_mean_relative_difference": float(
            (x[0].mean(0) - x[-1].mean(0)).norm() / x[0].mean(0).norm().clamp_min(1e-20)
        )
        if x.shape[0] > 1
        else None,
    }


def main():
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    root = checkout / "outputs/n02-v12/visual-gap-20261016/layers-a1"
    root.mkdir(exist_ok=False)
    data = np.load(
        checkout / "outputs/n02-v12/policy-audit-20261016/original-normal/initial_observation.npz"
    )
    policy = Policy(
        checkout / "configs/encoders/n02_v12/raw_repair_warm_r7_robotwin.json",
        checkout / "outputs/n02-v12/P2/raw-repair-warm-r7-s1103-a3/checkpoint_0000100.pt",
        root / "loader",
    )
    trainer = policy.trainer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    readout = trainer.model.inference.readout
    capture = {"input": [], "output": []}

    def pre(module, args):
        capture["input"].append(args[0].detach().clone())

    def post(module, args, result):
        capture["output"].append(result.detach().clone())

    pre_handle = readout.spatial_skip.register_forward_pre_hook(pre)
    post_handle = readout.spatial_skip.register_forward_hook(post)
    original = [data[f"raw_{i}"] for i in range(3)]
    images = {
        "original": original,
        "horizontal_flip": [x[:, ::-1].copy() for x in original],
        "all_zero": [np.zeros_like(x) for x in original],
        "camera_swap_head_left": [original[1], original[0], original[2]],
    }
    results = {}
    baseline = None
    with torch.inference_mode():
        for name, views in images.items():
            capture["input"].clear()
            capture["output"].clear()
            example = {
                "history": [{"step": 0, "image": views}],
                "step": 0,
                "case_id": "layer-diagnostic",
                "lang": "Grab the light brown cylindrical roller on the table.",
                "normalized_state": normalize_state(data["state"], policy.stats["state"]),
            }
            window = online_window(
                example, policy.cameras, history=8, representation=policy.representation
            )
            trainer.bridge.reset_episodes()
            visual = trainer.token_source(window)
            output = trainer.token_source.last_outputs[-1]
            spatial = output.state.geometry[0].spatial
            arrays = {
                f"encoder/{k}": v.detach().flatten(2).transpose(1, 2)
                for k, v in spatial.levels.items()
            }
            arrays.update(
                {
                    "sampled_multi_scale": torch.cat(capture["input"], dim=0),
                    "spatial_projection_before_layernorm": torch.cat(capture["output"], dim=0),
                    "final_tokens": visual.values.reshape(3, 256, 2048),
                }
            )
            row = {"layers": {k: summary(v) for k, v in arrays.items()}}
            if baseline is None:
                baseline = {k: v.clone() for k, v in arrays.items()}
            else:
                row["relative_l2_vs_original"] = {
                    k: relative(v, baseline[k]) for k, v in arrays.items()
                }
            if name == "original":
                torch.save(
                    {
                        "sampled_multi_scale": arrays["sampled_multi_scale"].cpu(),
                        "spatial_projection_before_layernorm": arrays[
                            "spatial_projection_before_layernorm"
                        ].cpu(),
                    },
                    root / "sampled_layers.pt",
                )
            results[name] = row
    pre_handle.remove()
    post_handle.remove()
    write_json(
        root / "report.json",
        {
            "optimizer_updates": 0,
            "new_rollouts": 0,
            "interventions": "fixed measured state/language, only RAW images replaced",
            "results": results,
        },
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
