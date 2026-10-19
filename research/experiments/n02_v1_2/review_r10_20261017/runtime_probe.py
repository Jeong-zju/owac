"""Read-only r9 audit: coupled padded actions, cached-forward parity and N02 diagnostics.

All interventions are in-memory counterfactuals. They are not trained models or rollouts.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from owac.backends.low_rank import restore_adapters
from owac.core.encoder_contract import VisualTokens
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import (
    LogTail,
    N02V12Trainer,
    anchor_action_targets,
    build_parser,
    resolve_plan,
)


def tensor_rms(x):
    return float(x.float().square().mean().sqrt())


@torch.inference_mode()
def sample(bridge, visual, lang, lmask, state, noise, *, analytic_padding=False):
    prefix = bridge.embed_prefix_from_visual_tokens(visual, lang, lmask)
    output = bridge.base.vlm.forward_prefix(
        inputs_embeds=prefix.embeds,
        attention_mask=bridge.mask_4d(bridge.mask_2d(prefix.pad_masks, prefix.att_masks)),
        position_ids=torch.cumsum(prefix.pad_masks, dim=1) - 1,
        past_key_values=None,
        use_cache=True,
    )
    x = noise.clone()
    trace = []
    dt = torch.tensor(-1.0 / 10, dtype=torch.float32, device=x.device)
    time_value = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    for _step in range(10):
        t = time_value.expand(x.shape[0])
        velocity = bridge.denoise_step(state, prefix.pad_masks, output.past_key_values, x, t)
        trace.append(
            dict(
                t=float(t[0]),
                padding_velocity_rms=tensor_rms(velocity[..., 14:]),
                padding_analytic_velocity_error=tensor_rms(velocity[..., 14:] - noise[..., 14:]),
            )
        )
        x = x + dt * velocity
        time_value += dt
        if analytic_padding:
            x[..., 14:] = noise[..., 14:] * time_value.clamp_min(0)
    return x, trace


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                "configs/encoders/n02_v12/raw_task_r9_task_robotwin.json",
                "--output",
                str(root / "loader"),
            ]
        )
    )
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()

    def forbidden(*a, **kw):
        raise RuntimeError("optimizer updates are forbidden in this code review")

    trainer.optimizer.step = forbidden
    trainer.model.eval()
    windows = trainer.dev_batch()
    generator = torch.Generator(device=trainer.device).manual_seed(plan.seed + 17)
    noise = torch.randn((len(windows), 50, 32), generator=generator, device=trainer.device)
    times = (
        torch.rand((len(windows),), generator=generator, device=trainer.device).pow(1 / 1.5) * 0.999
        + 0.001
    )
    specs = [
        ("initial", plan.config["initialisation"]["checkpoint"], False),
        ("task48", "outputs/n02-v12/P2/raw-task-r9-task-s1103-a1/checkpoint_0000048.pt", True),
        (
            "control32",
            "outputs/n02-v12/P2/raw-task-r9-control-s1103-a1/checkpoint_0000032.pt",
            True,
        ),
    ]
    adapter_initial = {
        n: {k: v.detach().clone() for k, v in a.state_dict().items()}
        for n, a in trainer.backend_adapters.items()
    }
    report = dict(optimizer_updates=0, new_closed_loops=0, checkpoints={}, rows=[])
    arrays = {}
    # Three Normal opening states and one eight-distinct-frame approach window.
    for label, checkpoint, has_adapters in specs:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(payload["model"], strict=True)
        restore_adapters(
            trainer.backend_adapters,
            payload["backend_adapters"] if has_adapters else adapter_initial,
        )
        report["checkpoints"][label] = dict(
            path=checkpoint, sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        )
        del payload
        for index in (0, 1, 8, 12):
            batch, targets, prompts = windows[index]
            proprio = trainer._anchor_proprio(batch)
            lang, lmask, state = trainer.language(prompts, proprio)
            raw, mask = anchor_action_targets(targets, batch.batch_size)
            target = trainer.bridge.prepare_control_targets(raw, device=trainer.device)
            count = Counter()
            handles = [
                module.register_forward_hook(
                    lambda m, a, o, name=name, counts=count: counts.update([name])
                )
                for name, module in (
                    ("encoder", trainer.model.inference.encoder),
                    ("decoder", trainer.model.inference.decoder),
                    ("readout", trainer.model.inference.readout),
                )
            ]
            captured = {}

            def capture(module, positional, kwargs, storage=captured):
                storage.update(kwargs)

            handles.append(
                trainer.model.inference.readout.register_forward_pre_hook(capture, with_kwargs=True)
            )
            with torch.inference_mode():
                torch.cuda.synchronize()
                start = time.perf_counter()
                visual = trainer.bridge.visual_tokens(batch)
                torch.cuda.synchronize()
                frontend_seconds = time.perf_counter() - start
                for h in handles:
                    h.remove()
                output = trainer.token_source.last_output
                native = trainer.bridge.predict_action_from_visual_tokens(
                    visual,
                    lang,
                    lmask,
                    state,
                    noise=noise[index : index + 1],
                    num_steps=10,
                )["normalized_actions"]
                original = torch.tensor(native, device=trainer.device)
                replay, trace = sample(
                    trainer.bridge, visual, lang, lmask, state, noise[index : index + 1]
                )
                repeat = trainer.bridge.predict_action_from_visual_tokens(
                    visual,
                    lang,
                    lmask,
                    state,
                    noise=noise[index : index + 1],
                    num_steps=10,
                )["normalized_actions"]
                assert torch.equal(replay, original), (
                    "counterfactual solver differs from production"
                )
                analytic, _ = sample(
                    trainer.bridge,
                    visual,
                    lang,
                    lmask,
                    state,
                    noise[index : index + 1],
                    analytic_padding=True,
                )
                altered_noise = noise[index : index + 1].clone()
                altered_noise[..., 14:] = noise[
                    (index + 1) % len(windows) : (index + 1) % len(windows) + 1, :, 14:
                ]
                altered, _ = sample(trainer.bridge, visual, lang, lmask, state, altered_noise)
                full = trainer.bridge.forward_from_visual_tokens(
                    visual,
                    lang,
                    lmask,
                    state,
                    target,
                    noise=noise[index : index + 1],
                    time=times[index : index + 1],
                    return_debug=True,
                    action_mask=torch.tensor(mask, device=trainer.device),
                )
                prefix = full["prefix"]
                cache = trainer.bridge.base.vlm.forward_prefix(
                    inputs_embeds=prefix.embeds,
                    attention_mask=trainer.bridge.mask_4d(
                        trainer.bridge.mask_2d(prefix.pad_masks, prefix.att_masks)
                    ),
                    position_ids=torch.cumsum(prefix.pad_masks, dim=1) - 1,
                    past_key_values=None,
                    use_cache=True,
                ).past_key_values
                cached_v = trainer.bridge.denoise_step(
                    state, prefix.pad_masks, cache, full["x_t"], full["time"]
                )
                q = output.q
                row = dict(
                    label=label,
                    index=index,
                    identity=trainer.task_dev_manifest["rows"][index],
                    frontend_seconds=frontend_seconds,
                    calls=dict(count),
                    q=q.cpu().tolist(),
                    energy=output.energies.cpu().tolist(),
                    accepted_rounds=output.diagnostics["accepted_rounds"].cpu().tolist(),
                    causal_check_status=str(output.check.status),
                    causal_check_valid=bool(output.check.valid),
                    language=prompts,
                    padded_output_rms=tensor_rms(original[..., 14:]),
                    teacher_control_velocity_mse=tensor_rms(
                        full["v_t"][..., :14] - full["u_t"][..., :14]
                    )
                    ** 2,
                    teacher_padding_velocity_mse=tensor_rms(
                        full["v_t"][..., 14:] - full["u_t"][..., 14:]
                    )
                    ** 2,
                    manual_sample_max_difference=float((replay - original).abs().max()),
                    repeat_sample_max_difference=float(
                        (torch.tensor(repeat, device=original.device) - original).abs().max()
                    ),
                    cached_vs_training_velocity_max=float((cached_v - full["v_t"]).abs().max()),
                    cached_vs_training_velocity_rms=tensor_rms(cached_v - full["v_t"]),
                    analytic_padding_control_rms_delta=tensor_rms(
                        analytic[..., :14] - original[..., :14]
                    ),
                    resample_only_padding_control_rms_delta=tensor_rms(
                        altered[..., :14] - original[..., :14]
                    ),
                    standard_control_mse=tensor_rms(original[..., :14] - target[..., :14]) ** 2,
                    standard_first10_mse=tensor_rms(original[:, :10, :14] - target[:, :10, :14])
                    ** 2,
                    analytic_control_mse=tensor_rms(analytic[..., :14] - target[..., :14]) ** 2,
                    trace=trace,
                )
                local = trainer.model.inference.readout.spatial_output_tokens(captured["features"])
                local_visual = VisualTokens(local, visual.mask, visual.camera_ids)
                local_actions, _ = sample(
                    trainer.bridge, local_visual, lang, lmask, state, noise[index : index + 1]
                )
                live_lang, live_mask, live_state = trainer.language(
                    ["Grab the light brown cylindrical roller on the table."], proprio
                )
                live_actions, _ = sample(
                    trainer.bridge,
                    visual,
                    live_lang,
                    live_mask,
                    live_state,
                    noise[index : index + 1],
                )
                row["live_prompt_control_rms_delta"] = tensor_rms(
                    live_actions[..., :14] - original[..., :14]
                )
                row["live_prompt_first10_control_rms_delta"] = tensor_rms(
                    live_actions[:, :10, :14] - original[:, :10, :14]
                )
                row["remove_context_token_relative_delta"] = float(
                    (local - visual.values).norm() / visual.values.norm()
                )
                row["remove_context_control_rms_delta"] = tensor_rms(
                    local_actions[..., :14] - original[..., :14]
                )
                row["local_only_control_mse"] = (
                    tensor_rms(local_actions[..., :14] - target[..., :14]) ** 2
                )
                if index == 0:
                    uniform, _ = trainer.model.inference.readout(
                        features=captured["features"],
                        state_tokens=captured["state_tokens"],
                        support=torch.full_like(captured["support"], 0.25),
                        masks=None,
                    )
                    uniform_visual = VisualTokens(uniform, visual.mask, visual.camera_ids)
                    uniform_actions, _ = sample(
                        trainer.bridge, uniform_visual, lang, lmask, state, noise[index : index + 1]
                    )
                    row["uniform_q_token_relative_delta"] = float(
                        (uniform - visual.values).norm() / visual.values.norm()
                    )
                    row["uniform_q_control_rms_delta"] = tensor_rms(
                        uniform_actions[..., :14] - original[..., :14]
                    )
                for key, value in (
                    ("sample", original),
                    ("analytic", analytic),
                    ("target", target),
                ):
                    arrays[f"{label}-{index}-{key}"] = value.cpu().numpy()
                report["rows"].append(row)
                write_json(root / "runtime_probe.json", report)
                print(
                    json.dumps({k: v for k, v in row.items() if k not in ("trace", "identity")}),
                    flush=True,
                )
            del captured, visual, output, full, local_visual
    np.savez_compressed(root / "actions.npz", **arrays)
    trainer.status(state="completed", activity="代码审查零更新诊断完成", step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
