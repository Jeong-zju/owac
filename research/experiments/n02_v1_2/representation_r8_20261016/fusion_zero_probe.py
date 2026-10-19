"""Same accepted RAW features and saved readout, zero updates; change fusion normalization."""

import argparse
from pathlib import Path

import torch

from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = resolve_plan(build_parser().parse_args(["--config", args.config, "--output", str(root)]))
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    trainer.initialise_repair_weights()
    trainer.model.eval()
    windows = trainer.dev_batch()
    generator = torch.Generator(device=trainer.device).manual_seed(plan.seed + 17)
    noise = torch.randn((len(windows), 50, 32), generator=generator, device=trainer.device)
    times = (
        torch.rand((len(windows),), generator=generator, device=trainer.device).pow(1 / 1.5) * 0.999
        + 0.001
    )

    def forbidden(*args, **kwargs):
        raise RuntimeError("optimizer update forbidden in fusion zero-update comparison")

    trainer.optimizer.step = forbidden
    report = trainer.evaluate_dev(windows, noise=noise, time_values=times)
    write_json(
        root / "result.json", {"optimizer_updates": 0, "metrics": report, "config": plan.config}
    )
    trainer.status(state="completed", activity="融合零更新对照完成", step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
