"""Paired RGB500/700 visual-token and frozen-backend sensitivity, no updates."""

import argparse
import gc
from pathlib import Path
from types import MethodType

import torch
from gradient_probe import WINDOWS, legacy_language

from owac.core.encoder_contract import VisualTokens
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import (
    LogTail,
    N02V12Trainer,
    anchor_action_targets,
    build_parser,
    resolve_plan,
)
from owac.training.n02_v12_profile import forbid_optimizer_steps


def main(arm):
    name = "activations" if arm == "rgb" else "activations-raw"
    root = Path("outputs/n02-v12/aux-audit-20261016") / name
    root.mkdir(exist_ok=False)
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                f"configs/encoders/n02_v12/{arm}_n02_robotwin.json",
                "--output",
                str(root / "loader"),
                "--run-id",
                "readonly-activations",
            ]
        )
    )
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    trainer.language = MethodType(legacy_language, trainer)
    reports = []
    with forbid_optimizer_steps(trainer.optimizer):
        for step in (500, 700) if arm == "rgb" else (900,):
            payload = trainer.load_weights_for_probe(
                Path(f"outputs/n02-v12/P2/{arm}-n02-robotwin-r5-s1103/checkpoint_{step:07d}.pt")
            )
            del payload
            trainer.model.train()
            reference = None
            for number, (start, label) in enumerate(WINDOWS):
                torch.manual_seed(20261016 + number)
                batch, targets, prompts = trainer.prepare_microbatch(start, start + 2)
                with torch.no_grad():
                    visual = trainer.token_source(batch)
                    token = visual.values.detach()
                    flat = token.flatten()
                    stats = {
                        "token_rms": float(flat.square().mean().sqrt()),
                        "token_abs_max": float(flat.abs().max()),
                        "within_sample_camera_position_std_mean": float(token.std(dim=1).mean()),
                    }
                    spatial = trainer.token_source.last_outputs[0].state.geometry[0].spatial
                    stats["encoder_spatial_std_over_rms"] = {
                        key: float(
                            value.flatten(2).std(dim=2).mean()
                            / value.square().mean().sqrt().clamp_min(1e-12)
                        )
                        for key, value in spatial.levels.items()
                    }
                    quantized = token.to(torch.bfloat16).float()
                    stats["bf16_tokens_equal_first_position_fraction"] = float(
                        (quantized == quantized[:, :1]).all(dim=-1).float().mean()
                    )
                    if reference is None:
                        reference = token[0].clone()
                    left = reference.flatten().float()
                    right = token[0].flatten().float()
                    stats["cosine_to_first_window"] = float(
                        torch.nn.functional.cosine_similarity(left, right, dim=0)
                    )
                    stats["relative_l2_to_first_window"] = float(
                        (left - right).norm() / left.norm()
                    )
                    language, masks, state = trainer.language(
                        prompts, trainer._anchor_proprio(batch)
                    )
                    actions, valid = anchor_action_targets(targets, 2)
                    target = trainer.bridge.prepare_control_targets(actions, device=trainer.device)
                    generator = torch.Generator(device=trainer.device).manual_seed(1104 + number)
                    noise = torch.randn(target.shape, generator=generator, device=trainer.device)
                    time = (
                        torch.rand((2,), generator=generator, device=trainer.device).pow(1 / 1.5)
                        * 0.999
                        + 0.001
                    )
                leaf = token.clone().requires_grad_(True)
                result = trainer.bridge.forward_from_visual_tokens(
                    VisualTokens(leaf, visual.mask, visual.camera_ids),
                    language,
                    masks,
                    state,
                    target,
                    noise=noise,
                    time=time,
                    action_mask=torch.as_tensor(valid, device=trainer.device),
                    return_debug=True,
                )
                (gradient,) = torch.autograd.grad(result["action_loss"], leaf)
                row = {
                    "step": step,
                    "label": label,
                    "action_loss": float(result["action_loss"]),
                    "flow_time": time.cpu().tolist(),
                    "token_gradient_l2": float(gradient.norm()),
                    **stats,
                }
                reports.append(row)
                print(row, flush=True)
                del result, leaf, visual, gradient, batch
                gc.collect()
            del reference
    assert not trainer.optimizer.state
    write_json(
        root / "probe.json",
        {
            "optimizer_updates": 0,
            "arm": arm,
            "source": "same fixed windows and flow draws as gradient probe",
            "rows": reports,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("raw", "rgb"), default="rgb")
    main(parser.parse_args().arm)
