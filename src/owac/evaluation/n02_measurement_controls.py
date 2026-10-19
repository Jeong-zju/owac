"""Bounded causal RAW representation controls after the plain retrained baseline.

All choices use development only. Select a single configuration before opening
a NEW holdout; never try alternate configurations against that holdout.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from owac.baselines.n02_measurement_adapters import MODES, AdaptedMeasurementScorer, adapt_raw
from owac.evaluation.n02_measurement_isolation import file_digest, tensor_digest, write_json
from owac.evaluation.n02_ratio import assess, evaluation_data
from owac.evaluation.n02_retrained import (
    allow_extension,
    measurement_acceptance,
    train_interval,
)
from owac.evaluation.n02_retrained import (
    protocol as plain_protocol,
)


def protocol(smoke=False):
    declaration = plain_protocol(smoke)
    declaration.update(
        {
            "experiment": "causal input-representation controls with same trainable official CNN",
            "modes": list(MODES),
            "input_data": "reuse plain baseline train/dev RAW byte-for-byte, including labels",
            "input_root": "outputs/n02-retrained-20261013/run",
            "development_seeds": list(range(23000, 23016)),
            "holdout_seeds": [24999] if smoke else list(range(25000, 25016)),
            "center": "first RAW-frame intensity centroid above calibrated5sigma=.05; integer roll",
            "innovation": "RAW_t minus measured RAW_0, same noisy reference; causal",
            "normalization": "training-only mean/std AFTER each declared adapter",
            "selection": "dev priority: innovation, center_innovation, center; "
            "no switch after holdout; 2/3 seeds required",
            "stop_when_qualified": "if dev qualifies at200, do not extend to800",
            "budget": "each mode 3x200, at most one gated extension per mode to800",
            "scope": "static-cue measurement baseline; raw innovation is not semantic refutation",
        }
    )
    return declaration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    declaration = protocol(args.smoke)
    write_json(args.output / "protocol.json", declaration)
    source_files = [
        Path("src/owac/baselines/n02_measurement.py"),
        Path("src/owac/baselines/n02_measurement_adapters.py"),
        Path("src/owac/baselines/sbi_embedding.py"),
        Path("src/owac/baselines/sbi_embedding.LICENSE.txt"),
        Path("src/owac/evaluation/n02_retrained.py"),
        Path("src/owac/evaluation/n02_ratio.py"),
        Path("src/owac/evaluation/n02_bottlenecks.py"),
        Path("src/owac/evaluation/n02_measurement_isolation.py"),
        Path(__file__).resolve().relative_to(Path.cwd()),
    ] + sorted(Path("src/owac/frameworks/n02").glob("*.py"))
    hashes = {str(p): file_digest(p) for p in source_files}
    for p in source_files:
        destination = args.output / "source_snapshot" / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    started = time.perf_counter()
    data_path = Path(declaration["input_root"]) / "data.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    train, dev = data["train"], data["development"]
    if args.smoke:
        train = {k: v[:8] for k, v in train.items()}
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashes,
        "input_path": str(data_path),
        "input_sha256": file_digest(data_path),
        "train_raw_sha256": tensor_digest(train["raw"]),
        "dev_raw_sha256": tensor_digest(dev["raw"]),
        "protocol_sha256": file_digest(args.output / "protocol.json"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": 2,
        },
        "modes": {},
    }
    rng = np.random.default_rng(declaration["sampler_seed"])
    schedule = np.stack(
        [
            rng.choice(len(train["raw"]) // 2, declaration["batch_backgrounds"], replace=False)
            for _ in range(declaration["maximum_updates"])
        ]
    )
    write_json(args.output / "schedule.json", schedule.tolist())
    developments, final_steps, choices = {}, {}, {}
    for mode in declaration["modes"]:
        adapted = adapt_raw(train["raw"], mode)
        mean = adapted.mean((0, 1, 3, 4))
        std = adapted.std((0, 1, 3, 4)).clamp_min(1e-6)
        manifest["modes"][mode] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "adapted_train_sha256": tensor_digest(adapted),
        }
        del adapted
        developments[mode] = {}
        for seed in declaration["model_seeds"]:
            torch.manual_seed(declaration["initialization_seed_offset"] + seed)
            model = AdaptedMeasurementScorer(mean, std, mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            output = args.output / mode / f"seed{seed}"
            output.mkdir(parents=True)
            snapshots = {}
            train_interval(
                model,
                optimizer,
                train,
                dev,
                schedule,
                0,
                declaration["initial_updates"],
                output,
                snapshots,
                declaration["snapshots"],
            )
            developments[mode][str(seed)] = snapshots
        already = (
            sum(
                row[str(declaration["initial_updates"])]["qualified"]
                for row in developments[mode].values()
            )
            >= 2
        )
        extension = allow_extension(developments[mode], args.smoke)
        extension["extend"] = extension["extend"] and not already
        extension["already_dev_qualified"] = already
        choices[mode] = extension
        write_json(args.output / mode / "extension_decision.json", extension)
        final_step = declaration["initial_updates"]
        if extension["extend"]:
            final_step = declaration["maximum_updates"]
            for seed in declaration["model_seeds"]:
                output = args.output / mode / f"seed{seed}"
                saved = torch.load(
                    output / "checkpoint200.pt", map_location="cpu", weights_only=False
                )
                model = AdaptedMeasurementScorer(mean, std, mode)
                model.load_state_dict(saved["model"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                optimizer.load_state_dict(saved["optimizer"])
                train_interval(
                    model,
                    optimizer,
                    train,
                    dev,
                    schedule,
                    200,
                    final_step,
                    output,
                    developments[mode][str(seed)],
                    declaration["snapshots"],
                )
        final_steps[mode] = final_step
        write_json(args.output / "manifest.json", manifest)
    selected = next(
        (
            mode
            for mode in ("innovation", "center_innovation", "center")
            if sum(
                rows[str(final_steps[mode])]["qualified"] for rows in developments[mode].values()
            )
            >= 2
        ),
        None,
    )
    # This decision is written before the new holdout is generated at all.
    write_json(
        args.output / "selection.json",
        {"selected": selected, "final_steps": final_steps, "extensions": choices},
    )
    holdout, acceptance = {}, {"allow_mechanism_comparison": False}
    if selected is not None and not args.smoke:
        held = evaluation_data(declaration["holdout_seeds"])
        torch.save(held, args.output / "holdout_data.pt")
        stats = manifest["modes"][selected]
        for seed in declaration["model_seeds"]:
            output = args.output / selected / f"seed{seed}"
            model = AdaptedMeasurementScorer(stats["mean"], stats["std"], selected).eval()
            saved = torch.load(
                output / f"checkpoint{final_steps[selected]}.pt",
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(saved["model"])
            holdout[str(seed)] = assess(model, held["raw"], held)
            write_json(output / "holdout.json", holdout[str(seed)])
        acceptance = measurement_acceptance(
            developments[selected], holdout, final_steps[selected], args.smoke
        )
    assert hashes == {str(p): file_digest(p) for p in source_files}
    summary = {
        "selected": selected,
        "final_steps": final_steps,
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "formal_updates": sum(final_steps.values()) * len(declaration["model_seeds"]),
        "source_unchanged": True,
        "production_encoder_updates": 0,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
