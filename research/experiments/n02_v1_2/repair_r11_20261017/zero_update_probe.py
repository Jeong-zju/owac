"""Production-width, real-data repair acceptance. Optimizer updates are forbidden."""

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import torch

from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan
from owac.training.n02_v12_task import BalancedBlocks, LegacyBalancedBlocks, source_records


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = resolve_plan(build_parser().parse_args(["--config", args.config, "--output", str(root)]))
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    trainer.initialise_repair_weights()
    write_json(root / "run_config.json", plan.config)

    def forbidden(*a, **kw):
        raise RuntimeError("optimizer updates forbidden in r11 acceptance")

    trainer.optimizer.step = forbidden
    report = dict(optimizer_updates=0, closed_loops=0, config=plan.config)
    rows = source_records(trainer.sources["train"])
    old = LegacyBalancedBlocks(rows, plan.batch_size, plan.seed)
    new = BalancedBlocks(rows, plan.batch_size, plan.seed)
    valid = {i for i, row in enumerate(rows) if row["valid"]}
    previous = {i for blocks in old.groups.values() for a, b in blocks for i in range(a, b)}
    opening = {i for i, row in enumerate(rows) if row["t"] == 0}
    report["sampler"] = dict(
        valid=len(valid),
        legacy_unreachable=len(valid - previous),
        repaired_unreachable=len(valid - set(new.covered)),
        episodes=len(opening),
        legacy_unreachable_t0=len(opening - previous),
        repaired_unreachable_t0=len(opening - set(new.covered)),
        repaired=new.record(),
    )
    assert set(new.covered) == valid
    write_json(root / "sampler_acceptance.json", report["sampler"])
    trainer.model.train()
    generator = torch.Generator(device=trainer.device).manual_seed(plan.seed + 37)
    # Exercise a complete eight-frame, eight-anchor batch; no optimizer step.
    a, b = next(
        (a, b)
        for blocks in new.groups.values()
        for a, b in blocks
        if b - a == 8 and rows[a]["t"] >= 10
    )
    start = time.perf_counter()
    result = trainer.microbatch(a, b, generator=generator)
    result["loss"].backward()
    torch.cuda.synchronize()
    gradients = {}
    for role, parameters in trainer.parameter_groups.items():
        active = [p.grad for p in parameters if p.grad is not None]
        gradients[role] = dict(
            tensors=len(active),
            finite=all(bool(torch.isfinite(g).all()) for g in active),
            norm=float(torch.stack([g.float().square().sum() for g in active]).sum().sqrt())
            if active
            else 0,
        )
    assert all(v["finite"] for v in gradients.values())
    for role in ("encoder", "decoder", "readout", "proposals_corrector", "backend_adapter"):
        assert gradients[role]["norm"] > 0, (role, gradients[role])
    frozen = [
        n
        for n, p in trainer.bridge.base.named_parameters()
        if not p.requires_grad and p.grad is not None
    ]
    assert not frozen
    report["backward"] = dict(
        seconds=time.perf_counter() - start,
        bounds=[a, b],
        loss=result["terms"],
        gradients=gradients,
        frozen_parameters_with_grad=frozen,
        peak_bytes=torch.cuda.max_memory_allocated(),
        global_gain_gradient=float(trainer.model.inference.readout.global_gain.grad),
        scale_logits_gradient=trainer.model.inference.readout.scale_logits.grad.cpu().tolist(),
        info=result["info"],
    )
    write_json(root / "backward_acceptance.json", report["backward"])
    del result
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.token_source.last_output = None
    trainer.token_source.last_outputs = []
    trainer.model.inference._last_state = None
    gc.collect()
    torch.cuda.empty_cache()
    trainer.model.eval()
    windows = trainer.dev_batch()
    generator = torch.Generator(device=trainer.device).manual_seed(plan.seed + 17)
    noise = torch.randn((len(windows), 50, 32), generator=generator, device=trainer.device)
    times = (
        torch.rand((len(windows),), generator=generator, device=trainer.device).pow(1 / 1.5) * 0.999
        + 0.001
    )
    dev = trainer.evaluate_dev(windows, noise=noise, time_values=times)
    report["dev_baseline"] = dev
    write_json(root / "dev_baseline.json", dev)
    fingerprint = trainer.calibration_fingerprint()
    saved = trainer.save(0, cursor=0, dev_reference=dev)
    with torch.no_grad():
        batch = windows[12][0]
        parity = []
        for fast in (False, True):
            trainer.model.inference.fast_prefix = fast
            calls = Counter()
            handles = [
                m.register_forward_hook(lambda m, a, o, n=n, c=calls: c.update([n]))
                for n, m in (
                    ("readout", trainer.model.inference.readout),
                    ("decoder", trainer.model.inference.decoder),
                )
            ]
            torch.cuda.synchronize()
            started = time.perf_counter()
            visual = trainer.bridge.visual_tokens(batch)
            torch.cuda.synchronize()
            for handle in handles:
                handle.remove()
            output = trainer.token_source.last_output
            parity.append(
                dict(
                    tokens=visual.values.detach().cpu().clone(),
                    q=output.q.detach().cpu().clone(),
                    calls=dict(calls),
                    seconds=time.perf_counter() - started,
                    checks=output.diagnostics.get("absolute_check"),
                    revision=output.diagnostics.get("revision"),
                )
            )
        report["fast_prefix"] = dict(
            token_max_delta=float((parity[0]["tokens"] - parity[1]["tokens"]).abs().max()),
            q_max_delta=float((parity[0]["q"] - parity[1]["q"]).abs().max()),
            calls=[p["calls"] for p in parity],
            seconds=[p["seconds"] for p in parity],
            diagnostics=[{k: p[k] for k in ("checks", "revision")} for p in parity],
        )
        assert report["fast_prefix"]["token_max_delta"] == 0
        trainer.model.inference.fast_prefix = True
        expected = trainer.bridge.visual_tokens(batch).values.detach().cpu().clone()
        trainer.model.inference.readout.global_gain.add_(0.01)
        try:
            trainer.restore_fixed_calibration(trainer.calibration_payload)
        except ValueError:
            report["stale_calibration_rejected"] = True
        else:
            raise AssertionError("stale calibration accepted")
        step, cursor, payload = trainer.load_for_resume(saved)
        del payload
        actual = trainer.bridge.visual_tokens(batch).values.detach().cpu()
        report["reload"] = dict(
            step=step,
            cursor=cursor,
            weight_fingerprint_equal=trainer.calibration_fingerprint() == fingerprint,
            token_max_delta=float((expected - actual).abs().max()),
            calibration_sha256=trainer.calibration_payload["weights_sha256"],
        )
        assert (
            report["reload"]["token_max_delta"] == 0
            and report["reload"]["weight_fingerprint_equal"]
        )
    write_json(root / "result.json", report)
    trainer.status(
        state="completed", activity="r11修补零更新验收完成；未训练、未证明RAW抓持", step=0
    )
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("config", "dev_baseline")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
