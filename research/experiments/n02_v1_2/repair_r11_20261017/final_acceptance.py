"""Freeze the final B12/native50 recipe; fit, evaluate and reload at zero updates."""

import argparse
from pathlib import Path

import torch

from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan
from owac.training.n02_v12_task import BalancedBlocks, select_repaired_checkpoint, source_records


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = resolve_plan(build_parser().parse_args(["--config", args.config, "--output", str(root)]))
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    trainer.initialise_repair_weights()

    def forbidden(*a, **kw):
        raise RuntimeError("optimizer updates forbidden in final recipe acceptance")

    trainer.optimizer.step = forbidden
    write_json(root / "run_config.json", plan.config)
    rows = source_records(trainer.sources["train"])
    sampler = BalancedBlocks(rows, plan.batch_size, plan.seed)
    assert len(sampler.covered) == sum(r["valid"] > 0 for r in rows) == 3630
    trainer.model.eval()
    windows = trainer.dev_batch()
    gen = torch.Generator(device=trainer.device).manual_seed(plan.seed + 17)
    noise = torch.randn((len(windows), 50, 32), generator=gen, device=trainer.device)
    times = (
        torch.rand((len(windows),), generator=gen, device=trainer.device).pow(1 / 1.5) * 0.999
        + 0.001
    )
    dev = trainer.evaluate_dev(windows, noise=noise, time_values=times)
    assert dev["sampled_metric"] == "balanced_sampled_control14_mse"
    assert dev["sampled_control_mse"] == dev["task_control"]["balanced_mse"]
    assert dev["online_checks"]["valid"] == dev["online_checks"]["expected_causal"] == 15
    write_json(root / "dev_baseline.json", dev)
    checkpoint = trainer.save(0, cursor=0, dev_reference=dev)
    digest = trainer.calibration_fingerprint()
    step, cursor, payload = trainer.load_for_resume(checkpoint)
    del payload
    assert step == cursor == 0 and trainer.calibration_fingerprint() == digest
    decision = select_repaired_checkpoint(root)
    assert decision["selected"] is None
    write_json(root / "selection.json", decision)
    write_json(
        root / "result.json",
        dict(
            optimizer_updates=0,
            new_closed_loops=0,
            final_recipe=True,
            batch_size=plan.batch_size,
            sampler=sampler.record(),
            dev=dev,
            resume_fingerprint_equal=True,
            physical_success=None,
        ),
    )
    trainer.status(
        state="completed", activity="最终配方零更新验收完成，未训练、RAW抓持未验证", step=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
