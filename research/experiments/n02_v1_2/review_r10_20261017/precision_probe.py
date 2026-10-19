"""Check native joint/cached π0.5 velocity agreement without any optimization."""

import argparse
from pathlib import Path

import torch

from owac.backends.low_rank import restore_adapters
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import (
    LogTail,
    N02V12Trainer,
    anchor_action_targets,
    build_parser,
    resolve_plan,
)


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
        raise RuntimeError("optimizer update forbidden in precision review")

    trainer.optimizer.step = forbidden
    # Match _sample_impl: the native prefix cache uses eager attention, not the
    # language model's construction-time SDPA default.
    trainer.bridge.base.vlm.language_model.config._attn_implementation = "eager"
    checkpoint = Path("outputs/n02-v12/P2/raw-task-r9-task-s1103-a1/checkpoint_0000048.pt")
    p = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(p["model"], strict=True)
    restore_adapters(trainer.backend_adapters, p["backend_adapters"])
    del p
    trainer.model.eval()
    windows = trainer.dev_batch()
    rng = torch.Generator(device=trainer.device).manual_seed(plan.seed + 17)
    noise = torch.randn((len(windows), 50, 32), generator=rng, device=trainer.device)
    times = (
        torch.rand((len(windows),), generator=rng, device=trainer.device).pow(1 / 1.5) * 0.999
        + 0.001
    )
    prepared = []
    with torch.inference_mode():
        for i in (0, 1):
            batch, targets, prompts = windows[i]
            visual = trainer.bridge.visual_tokens(batch)
            lang, lmask, state = trainer.language(prompts, trainer._anchor_proprio(batch))
            raw, mask = anchor_action_targets(targets, batch.batch_size)
            target = trainer.bridge.prepare_control_targets(raw, device=trainer.device)
            prepared.append((i, visual, lang, lmask, state, target, mask))
        rows = []
        for mode, tf32 in (("production", True), ("bf16_no_tf32", False), ("fp32_no_tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            if mode == "fp32_no_tf32":
                trainer.bridge.base.to(dtype=torch.float32)
            for i, visual, lang, lmask, state, target, mask in prepared:
                full = trainer.bridge.forward_from_visual_tokens(
                    visual,
                    lang,
                    lmask,
                    state,
                    target,
                    noise=noise[i : i + 1],
                    time=times[i : i + 1],
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
                cached = trainer.bridge.denoise_step(
                    state, prefix.pad_masks, cache, full["x_t"], full["time"]
                )
                delta = cached - full["v_t"]
                rows.append(
                    dict(
                        mode=mode,
                        index=i,
                        prefix_dtype=str(prefix.embeds.dtype),
                        max_abs=float(delta.abs().max()),
                        rms=float(delta.square().mean().sqrt()),
                        control_max_abs=float(delta[..., :14].abs().max()),
                    )
                )
                print(rows[-1], flush=True)
        write_json(root / "precision_probe.json", dict(optimizer_updates=0, rows=rows))
        trainer.status(state="completed", activity="精度与缓存路径复核完成", step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
