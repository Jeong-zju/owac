"""Offline hard-prefix inpainting baseline; not an implementation or claim of RTC."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.runtime.n02_v12_robotwin import Policy
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import anchor_action_targets


@torch.inference_mode()
def sample(bridge, visual, language, masks, state, noise, known=None):
    bridge.base.vlm.language_model.config._attn_implementation = "eager"
    prefix = bridge.embed_prefix_from_visual_tokens(visual, language, masks)
    cached = bridge.base.vlm.forward_prefix(
        inputs_embeds=prefix.embeds,
        attention_mask=bridge.mask_4d(bridge.mask_2d(prefix.pad_masks, prefix.att_masks)),
        position_ids=torch.cumsum(prefix.pad_masks, dim=1) - 1,
        past_key_values=None,
        use_cache=True,
    ).past_key_values
    x = noise.clone()
    dt = torch.tensor(-0.1, device=x.device, dtype=torch.float32)
    t = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    for _ in range(10):
        velocity = bridge.denoise_step(state, prefix.pad_masks, cached, x, t.expand(1))
        x = x + dt * velocity
        t += dt
        if known is not None:
            length = known.shape[1]
            # The time-t path for known clean actions, preserving the same noise.
            x[:, :length, :14] = (
                t.clamp_min(0) * noise[:, :length, :14] + (1 - t.clamp_min(0)) * known[..., :14]
            )
    return x


@torch.inference_mode()
def main(attempt):
    root = Path("/workspace/owac/worktrees/n02-measurement-completion")
    evidence = root / "research/experiments/n02_v1_2/sequence_r12_20261019"
    output = evidence / ("prefix_probe_" + attempt)
    output.mkdir(exist_ok=False)
    policy = Policy(
        evidence.parent / "training_r11_20261017/task_config.json",
        root / "outputs/n02-v12/P2/raw-task-r11-task-s1103-a1/checkpoint_0000850.pt",
        output / "loader",
    )
    trainer = policy.trainer
    windows = trainer.dev_batch()
    identities = trainer.task_dev_manifest["rows"]
    i = next(
        i for i, r in enumerate(identities) if r["light"] == "Normal" and r["phase"] == "opening"
    )
    j = next(
        j
        for j, r in enumerate(identities)
        if r["path"] == identities[i]["path"] and r["phase"] == "approach"
    )
    data = []
    for index in (i, j):
        batch, target, prompt = windows[index]
        language, masks, state = trainer.language(prompt, trainer._anchor_proprio(batch))
        trainer.bridge.reset_episodes()
        visual = trainer.bridge.visual_tokens(batch)
        data.append((visual, language, masks, state, target))
    noise = torch.randn(
        (1, 50, 32),
        device=trainer.device,
        generator=torch.Generator(device=trainer.device).manual_seed(188390828),
    )
    before = sample(trainer.bridge, *data[0][:4], noise)
    raw, valid = anchor_action_targets(data[1][4], 1)
    target = torch.tensor(raw, device=trainer.device)
    baseline = sample(trainer.bridge, *data[1][:4], noise)
    official = trainer.bridge.predict_action_from_visual_tokens(
        *data[1][:4], noise=noise, num_steps=10
    )["normalized_actions"]
    parity = float((baseline - torch.as_tensor(official, device=trainer.device)).abs().max())
    if parity > 1e-6:
        raise ValueError("counterfactual integrator differs from production")
    rows = []
    arrays = {"before": before.cpu().numpy(), "baseline": baseline.cpu().numpy()}
    for length in (3, 5):
        known = before[:, 10 : 10 + length]
        conditioned = sample(trainer.bridge, *data[1][:4], noise, known)
        mask = torch.tensor(valid, device=trainer.device)
        mask[:, :length] = False
        rows.append(
            {
                "prefix_length": length,
                "constrained_prefix_error_by_construction": float(
                    (conditioned[:, :length, :14] - known[..., :14]).abs().max()
                ),
                "unconstrained_tail_mse": float(
                    (conditioned[..., :14] - target)[mask].square().mean()
                ),
                "baseline_same_tail_mse": float(
                    (baseline[..., :14] - target)[mask].square().mean()
                ),
                "unconstrained_next5_mse": float(
                    (conditioned[:, length : length + 5, :14] - target[:, length : length + 5])
                    .square()
                    .mean()
                ),
                "baseline_next5_mse": float(
                    (baseline[:, length : length + 5, :14] - target[:, length : length + 5])
                    .square()
                    .mean()
                ),
            }
        )
        arrays[f"conditioned-{length}"] = conditioned.cpu().numpy()
    report = {
        "optimizer_updates": 0,
        "new_closed_loops": 0,
        "production_parity_max": parity,
        "scope": "offline only; prefix uses previous model predictions, never expert actions",
        "method": "hard-prefix flow inpainting baseline, not full RTC soft guidance",
        "reference": "https://www.physicalintelligence.company/download/real_time_chunking.pdf",
        "rows": rows,
    }
    np.savez_compressed(output / "actions.npz", **arrays)
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="a1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    main(args.attempt)
