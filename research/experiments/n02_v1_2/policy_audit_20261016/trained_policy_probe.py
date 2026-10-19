"""Read-only r5 sampling/teacher-forcing/gradient audit; no optimizer updates."""

import gc
import json
from pathlib import Path

import numpy as np
import torch

from owac.backends.rawvla_language import prepare_language_state
from owac.core.encoder_contract import VisualTokens
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import (
    LogTail,
    N02V12Trainer,
    anchor_action_targets,
    build_parser,
    resolve_plan,
)


def mse(left, right, valid):
    error = (left - right).square().mean(-1)
    return float(error[valid].mean())


def audit(arm, step):
    root = Path("outputs/n02-v12/policy-audit-20261016") / f"trained-{arm}"
    root.mkdir(exist_ok=False)
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                f"configs/encoders/n02_v12/{arm}_n02_robotwin.json",
                "--output",
                str(root / "runtime"),
                "--run-id",
                f"readonly-{arm}",
            ]
        )
    )
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    _, _, payload = trainer.load_for_resume(
        Path(f"outputs/n02-v12/P2/{arm}-n02-robotwin-r5-s1103/checkpoint_{step:07d}.pt")
    )
    del payload
    trainer.model.eval()
    generator = torch.Generator(device="cuda").manual_seed(1120)
    noise = torch.randn((4, 50, 32), generator=generator, device="cuda")
    times = torch.rand((4,), generator=generator, device="cuda").pow(1 / 1.5) * 0.999 + 0.001
    records = []
    with torch.no_grad():
        for index, (batch, targets, prompts) in enumerate(trainer.dev_batch()):
            proprio = trainer._anchor_proprio(batch)
            native = prepare_language_state(trainer.bridge, prompts, proprio, device="cuda")
            legacy = trainer.language(prompts, proprio)
            actual, mask = anchor_action_targets(targets, 1)
            target = trainer.bridge.prepare_control_targets(actual, device="cuda")
            valid = torch.as_tensor(mask, device="cuda")
            visual = trainer.token_source(batch)
            record = {
                "anchor": list(batch.anchors),
                "token_rms": float(visual.values.float().square().mean().sqrt()),
                "active_tokens": {"legacy": int(legacy[1].sum()), "native": int(native[1].sum())},
            }
            for label, (language, language_mask, state) in (("legacy", legacy), ("native", native)):
                flow = trainer.bridge.forward_from_visual_tokens(
                    visual,
                    language,
                    language_mask,
                    state,
                    target,
                    noise=noise[index : index + 1],
                    time=times[index : index + 1],
                    action_mask=valid,
                    return_debug=True,
                )
                sample = trainer.bridge.predict_action_from_visual_tokens(
                    visual, language, language_mask, state, noise=noise[index : index + 1]
                )
                prediction = torch.as_tensor(sample["control_actions"], device="cuda")
                record[label] = {
                    "flow_time": float(times[index]),
                    "flow_mse_32": float(flow["action_loss"]),
                    "flow_mse_control14": mse(flow["v_t"][..., :14], flow["u_t"][..., :14], valid),
                    "sampled_action_mse_control14": mse(prediction, target[..., :14], valid),
                    "first5_sample_mse": mse(prediction[:, :5], target[:, :5, :14], valid[:, :5]),
                }
                if index == 0:
                    zero = VisualTokens(
                        torch.zeros_like(visual.values), visual.mask, visual.camera_ids
                    )
                    absent = trainer.bridge.predict_action_from_visual_tokens(
                        zero, language, language_mask, state, noise=noise[index : index + 1]
                    )["control_actions"]
                    record[label]["zero_visual_sample_action_delta_max"] = float(
                        np.max(np.abs(sample["control_actions"] - absent))
                    )
            records.append(record)
    # Inspect objective gradient magnitudes on one non-prefix-padding real training anchor.
    trainer.optimizer.zero_grad(set_to_none=True)
    result = trainer.microbatch(
        100, 101, generator=torch.Generator(device="cuda").manual_seed(1104)
    )
    parameters = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    norms, first = {}, None
    for name, term in result["loss_tensors"].items():
        if term is None or not term.requires_grad:
            continue
        gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        squared = sum(float(g.float().square().sum()) for g in gradients if g is not None)
        norms[name] = {"loss": float(term), "gradient_l2": squared**0.5}
        if name == "action":
            first = [g.detach().clone() if g is not None else None for g in gradients]
        elif first is not None:
            dot = sum(
                float((a.float() * b.float()).sum())
                for a, b in zip(first, gradients, strict=True)
                if a is not None and b is not None
            )
            denominator = norms["action"]["gradient_l2"] * norms[name]["gradient_l2"]
            norms[name]["cosine_with_action"] = dot / denominator if denominator else None
    report = {
        "arm": arm,
        "step": step,
        "optimizer_updates": 0,
        "dev_windows": records,
        "gradient_window": result["info"]["window"],
        "objective_gradients": norms,
    }
    write_json(root / "probe.json", report)
    print(json.dumps(report), flush=True)
    del trainer, result, parameters, first, gradients
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    audit("raw", 900)
    audit("rgb", 500)
