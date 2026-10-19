"""Bounded actual updates, task gradients, frozen weights and exact action reload."""

import argparse
import time
from pathlib import Path

import torch

from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan
from owac.training.n02_v12_profile import parameter_digest


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    trainer = N02V12Trainer(
        resolve_plan(
            build_parser().parse_args(
                ["--config", args.config, "--output", str(output), "--batch-size", str(args.batch)]
            )
        ),
        LogTail(),
    )
    trainer.build()
    trainer.initialise_repair_weights()
    parameters = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    expected = {id(p) for p in trainer.model.parameters() if p.requires_grad}
    expected.update(id(p) for p in trainer.bridge.base.parameters() if p.requires_grad)
    assert expected == {id(p) for p in parameters} and len(expected) == len(parameters)
    frozen = [p for p in trainer.model.parameters() if not p.requires_grad]
    frozen.extend(p for p in trainer.bridge.base.parameters() if not p.requires_grad)
    before = parameter_digest(frozen)
    changes = parameter_digest(parameters)
    generator = torch.Generator(device=trainer.device).manual_seed(1104)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        trainer.optimizer.zero_grad(set_to_none=True)
        start, stop = trainer.training_bounds(step)
        begin = time.monotonic()
        result = trainer.microbatch(start, stop, generator=generator)
        result["loss"].backward()
        norms = {
            n: sum(float(p.grad.float().norm()) for p in ps if p.grad is not None)
            for n, ps in trainer.parameter_groups.items()
        }
        assert all(p.grad is None for p in frozen)
        assert norms["backend_adapter"] > 0
        if trainer.plan.config["task_adaptation"]["train_deep_blocks"]:
            assert norms["encoder"] > 0 and norms["readout"] > 0
        trainer.clip_gradients()
        trainer.optimizer.step()
        torch.cuda.synchronize()
        rows.append(
            dict(
                step=step + 1,
                seconds=time.monotonic() - begin,
                loss=float(result["loss"].detach()),
                grad_norms=norms,
            )
        )
        del result
    assert before == parameter_digest(frozen)
    assert changes != parameter_digest(parameters)
    # A real training window and fixed flow seed; no teacher/policy actions in the input.
    batch, _, prompts = trainer.prepare_microbatch(0, 1)

    def sample():
        with torch.no_grad():
            visual = trainer.token_source(batch)
            lang, mask, state = trainer.language(prompts, trainer._anchor_proprio(batch))
            noise = torch.randn(
                (1, 50, 32),
                generator=torch.Generator(device=trainer.device).manual_seed(1103),
                device=trainer.device,
            )
            return (
                torch.as_tensor(
                    trainer.bridge.predict_action_from_visual_tokens(
                        visual, lang, mask, state, noise=noise
                    )["control_actions"]
                )
                .cpu()
                .clone()
            )

    actions = sample()
    path = trainer.save(3, cursor=3)
    trainer.load_for_resume(path)
    delta = float((sample() - actions).abs().max())
    assert delta == 0
    write_json(
        output / "pilot.json",
        dict(
            optimizer_updates=3,
            rows=rows,
            batch=args.batch,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            frozen_parameters_unchanged=True,
            trainable_parameters_changed=True,
            save_reload_action_max_abs=delta,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=8)
    main(p.parse_args())
