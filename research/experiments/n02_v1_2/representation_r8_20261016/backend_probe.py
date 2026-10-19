"""Real two-update backend integration pilot after the RAW representation gate."""

import argparse
import time
from pathlib import Path

import torch

from owac.backends.low_rank import adapter_state
from owac.core.encoder_contract import VisualTokens
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan
from owac.training.n02_v12_profile import parameter_digest


def sample(trainer, prepared):
    batch, _, prompts = prepared
    trainer.model.eval()
    with torch.no_grad():
        visual = trainer.token_source(batch)
        language, mask, state = trainer.language(prompts, trainer._anchor_proprio(batch))
        generator = torch.Generator(device=trainer.device).manual_seed(71103)
        noise = torch.randn((batch.batch_size, 50, 32), generator=generator, device=trainer.device)
        result = trainer.bridge.predict_action_from_visual_tokens(
            VisualTokens(visual.values, visual.mask, visual.camera_ids),
            language,
            mask,
            state,
            noise=noise,
            num_steps=10,
        )
    return torch.as_tensor(result["normalized_actions"]).cpu().clone()


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                args.config,
                "--output",
                str(root),
                "--batch-size",
                str(args.batch),
            ]
        )
    )
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    trainer.initialise_repair_weights()
    encoder_digest = parameter_digest(trainer.model.inference.encoder.parameters())
    frontend_digest = (
        parameter_digest(trainer.model.parameters())
        if plan.config["policy"].get("freeze_frontend", False)
        else None
    )
    frozen = [p for p in trainer.bridge.base.parameters() if not p.requires_grad]
    original_digest = parameter_digest(frozen)
    parameters = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in trainer.model.parameters() if p.requires_grad}
    expected.update(id(p) for p in trainer.masked.parameters() if p.requires_grad)
    expected.update(id(p) for p in trainer.bridge.base.parameters() if p.requires_grad)
    expected.update(
        id(p) for m in trainer.bridge.bridge_modules() for p in m.parameters() if p.requires_grad
    )
    assert expected == {id(p) for p in parameters} and len(parameters) == len(expected)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for step in range(2):
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.model.train()
        generator = torch.Generator(device=trainer.device).manual_seed(1104 + step)
        begin = time.perf_counter()
        result = trainer.microbatch(0, args.batch, generator=generator)
        result["loss"].backward()
        gradients = {
            n: {
                k: None if p.grad is None else float(p.grad.float().norm())
                for k, p in a.named_parameters()
            }
            for n, a in trainer.backend_adapters.items()
        }
        assert all(p.grad is None for p in frozen)
        trainer.clip_gradients()
        trainer.optimizer.step()
        torch.cuda.synchronize()
        rows.append(
            {
                "step": step + 1,
                "loss": float(result["loss"].detach()),
                "seconds": time.perf_counter() - begin,
                "adapter_gradients": gradients,
            }
        )
        del result
    prepared = trainer.prepare_microbatch(0, 1)
    before = sample(trainer, prepared)
    state_before = adapter_state(trainer.backend_adapters)
    checkpoint = trainer.save(2, cursor=2)
    with torch.no_grad():
        for adapter in trainer.backend_adapters.values():
            adapter.B.zero_()
    ablated = sample(trainer, prepared)
    trainer.load_for_resume(checkpoint)
    restored = sample(trainer, prepared)
    delta = float((before - restored).abs().max())
    assert delta == 0
    assert encoder_digest == parameter_digest(trainer.model.inference.encoder.parameters())
    assert original_digest == parameter_digest(frozen)
    if frontend_digest is not None:
        assert frontend_digest == parameter_digest(trainer.model.parameters())
    b_nonzero = sum(
        bool(row["adapter_gradients"][name]["B"])
        for row in rows
        for name in trainer.backend_adapters
    )
    if trainer.backend_adapters:
        assert b_nonzero > 0
        assert float((before - ablated).abs().max()) > 0
    write_json(
        root / "result.json",
        {
            "optimizer_updates": 2,
            "batch": args.batch,
            "rows": rows,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "backend_original_unchanged": True,
            "encoder_unchanged": True,
            "full_frontend_unchanged": True if frontend_digest is not None else None,
            "sampled_action_save_reload_max_abs_difference": delta,
            "adapter_ablation_action_max_abs_difference": float((before - ablated).abs().max()),
            "adapter_report": trainer.backend_adapter_report,
            "adapter_state_norms": {
                n: {k: float(v.norm()) for k, v in a.items()} for n, a in state_before.items()
            },
            "parameter_report": trainer.bridge.parameter_report(),
        },
    )
    trainer.status(state="completed", activity="2步后端接线与动作保存恢复验证完成", step=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=4)
    main(parser.parse_args())
