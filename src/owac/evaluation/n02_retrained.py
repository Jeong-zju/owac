"""Measurement-first N02 continuation with a retrained official CNN embedding.

One predeclared baseline, three seeds, behavior-gated 200->800 updates. The
existing production encoder is not trained or used as a hidden feature oracle.
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
import torch.nn.functional as F

from owac.baselines.n02_measurement import RetrainedMeasurementScorer
from owac.evaluation.n02_measurement_isolation import file_digest, tensor_digest, write_json
from owac.evaluation.n02_ratio import assess, evaluation_data
from owac.frameworks.n02.synthetic import SyntheticGenerator, SyntheticOptions


def protocol(smoke=False):
    return {
        "experiment": "retrained official CNN, balanced supervised posterior-ratio baseline",
        "smoke": smoke,
        "model_seeds": [0] if smoke else [0, 1, 2],
        "train_background_seeds": list(range(22000, 22004 if smoke else 22256)),
        "development_seeds": [22999] if smoke else list(range(23000, 23016)),
        "holdout_seeds": [23999] if smoke else list(range(24000, 24016)),
        "task": "opening_direction; two fixed physical explanations; static scene",
        "raw": "existing generator, 3-channel pseudo-RAW, 64x64, 8 frames, all pixels valid",
        "noise_sigma": 0.01,
        "train_identifiable_probability": 0.8,
        "training": "two counterfactual measured worlds per background, same noise; balanced CE",
        "embedding": "sbi0.27.0 CNNEmbedding [6,12], kernel5/pool2, FC32/32; random init trainable",
        "temporal": "torch GRU32, causal, reset every sequence, classifier32->2",
        "normalization": "per-channel mean/std from training RAW only, frozen buffers",
        "scores": "class logits - log(training_prior=0.5), prefix evidence replaces prior scores",
        "external_prior_in_scorer": False,
        "training_loss": "torch cross_entropy on all causal prefixes, physical binary labels only",
        "optimizer": "AdamW lr0.001 weight_decay0.01 clip1.0, FP32 CPU threads2",
        "batch_backgrounds": 2 if smoke else 8,
        "initial_updates": 2 if smoke else 200,
        "maximum_updates": 2 if smoke else 800,
        "snapshots": [0, 1, 2] if smoke else [0, 100, 200, 400, 800],
        "sampler_seed": 22000,
        "initialization_seed_offset": 22500,
        "extension_rule": "2/3 seeds: dev pairs>=8/16, no-cue>=12/16, train final accuracy>=0.75; "
        "aggregate behavior count nondecreasing 100->200; no holdout consulted",
        "acceptance_rule": "2/3 same seeds qualify on dev AND holdout: pairs>=12/16, "
        "no-cue>=12/16 at all prefixes for priors .05/.5/.95, "
        "95% wrong-prior corrected>=4/16 in each world direction",
        "reuse_scope": "official embedding and torch CE/GRU/AdamW; not full sbi reproduction",
        "data_boundary": "disjoint train/val geometry; dev and holdout new seeds, same val set",
        "second_stage": "reliable score required; shared scorer/information; absolute-fit checks",
        "exit": "reliable evidence but no repeatable local gain: archive current mechanism",
    }


def paired_training_data(seeds):
    raws, labels, identities = [], [], []
    for seed in seeds:
        generator = SyntheticGenerator(
            SyntheticOptions(height=64, width=64), seed=seed, split="train"
        )
        episode = generator.generate(seed=seed, episode_index=seed, prior_condition="missing")
        templates = np.stack([generator.measurement_templates(episode, i) for i in range(8)])
        for world in range(2):
            raw = torch.from_numpy((templates[:, world] + episode.noise).astype(np.float32))
            raws.append(raw)
            labels.append(world)
            identities.append(
                {
                    "background_seed": seed,
                    "world": world,
                    "identifiable": episode.identifiable,
                    "raw_sha256": tensor_digest(raw),
                }
            )
    return {"raw": torch.stack(raws), "truth": torch.tensor(labels), "identities": identities}


def behavior_count(report):
    return (
        report["pair_both_correct"]
        + report["no_cue_pass"]
        + sum(report["wrong_prior_corrected"].values())
    )


def allow_extension(development, smoke):
    if smoke:
        return {"extend": False, "reason": "smoke is not a selection experiment"}
    qualified = {}
    counts = {
        str(step): sum(behavior_count(row[str(step)]) for row in development.values())
        for step in (100, 200)
    }
    for seed, snapshots in development.items():
        row = snapshots["200"]
        qualified[seed] = bool(
            row["pair_both_correct"] >= 8
            and row["no_cue_pass"] >= 12
            and row["train_final_accuracy"] >= 0.75
        )
    extend = sum(qualified.values()) >= 2 and counts["200"] >= counts["100"]
    return {"extend": extend, "seed_progress": qualified, "behavior_counts": counts}


def measurement_acceptance(development, holdout, final_step, smoke):
    per_seed = {
        seed: bool(snapshots[str(final_step)]["qualified"] and holdout[seed]["qualified"])
        for seed, snapshots in development.items()
    }
    return {
        "same_seed_qualified": per_seed,
        "allow_mechanism_comparison": bool(not smoke and sum(per_seed.values()) >= 2),
    }


@torch.no_grad()
def training_accuracy(model, data):
    final = torch.cat([model(piece)[:, -1] for piece in data["raw"].split(16)])
    return float((final.argmax(-1) == data["truth"]).float().mean())


def train_interval(model, optimizer, train, dev, schedule, start, stop, output, snapshots, stages):
    parameters = tuple(model.parameters())
    branch_names = ("embedding.cnn_subnet", "embedding.linear_subnet", "temporal", "classifier")
    logs = []
    before = {name: tensor_digest(p) for name, p in model.named_parameters()}

    def snapshot(step):
        model.eval()
        snapshots[str(step)] = assess(model, dev["raw"], dev)
        snapshots[str(step)]["train_final_accuracy"] = training_accuracy(model, train)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            output / f"checkpoint{step}.pt",
        )
        write_json(output / "development.json", snapshots)
        model.train()

    if start == 0:
        snapshot(0)
    started = time.perf_counter()
    for index in range(start, stop):
        # A balanced pair gives both possible worlds the same nuisance background.
        backgrounds = schedule[index]
        indexes = torch.tensor(np.stack([2 * backgrounds, 2 * backgrounds + 1], 1).flatten())
        raw, labels = train["raw"][indexes], train["truth"][indexes]
        optimizer.zero_grad(set_to_none=True)
        logits = model(raw)
        loss = F.cross_entropy(logits.flatten(0, 1), labels[:, None].expand(-1, 8).reshape(-1))
        loss.backward()
        gradients = {
            branch: float(
                torch.stack(
                    [
                        p.grad.norm()
                        for name, p in model.named_parameters()
                        if name.startswith(branch) and p.grad is not None
                    ]
                ).norm()
            )
            for branch in branch_names
        }
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite objective/gradient at step{index + 1}")
        optimizer.step()
        logs.append(
            {
                "step": index + 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "branch_gradient_norm": gradients,
            }
        )
        if index + 1 in stages:
            snapshot(index + 1)
            row = snapshots[str(index + 1)]
            print(
                f"{output.name} step{index + 1}: train_acc={row['train_final_accuracy']:.3f} "
                f"pairs={row['pair_both_correct']} no_cue={row['no_cue_pass']} "
                f"corrected={row['wrong_prior_corrected']}",
                flush=True,
            )
    changed = {
        branch: any(
            tensor_digest(p) != before[name]
            for name, p in model.named_parameters()
            if name.startswith(branch)
        )
        for branch in branch_names
    }
    write_json(
        output / f"training_{start}_{stop}.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "logs": logs,
            "branch_parameters_changed": changed,
        },
    )
    if not all(changed.values()):
        raise AssertionError(f"trainable branch did not change: {changed}")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = protocol(args.smoke)
    write_json(args.output / "protocol.json", declaration)
    dependencies = sorted(Path("src/owac/frameworks/n02").glob("*.py")) + [
        Path("src/owac/baselines/n02_measurement.py"),
        Path("src/owac/baselines/sbi_embedding.py"),
        Path("src/owac/baselines/sbi_embedding.LICENSE.txt"),
        Path(__file__).resolve().relative_to(Path.cwd()),
        Path("src/owac/evaluation/n02_ratio.py"),
        Path("src/owac/evaluation/n02_bottlenecks.py"),
        Path("src/owac/evaluation/n02_measurement_isolation.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    hashes = {str(path): file_digest(path) for path in dependencies}
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashes,
        "protocol_sha256": file_digest(args.output / "protocol.json"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": 2,
        },
    }
    for path in dependencies:
        target = args.output / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (args.output / "tracked_changes.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"])
    )
    started = time.perf_counter()
    train = paired_training_data(declaration["train_background_seeds"])
    dev = evaluation_data(declaration["development_seeds"])
    torch.save({"train": train, "development": dev}, args.output / "data.pt")
    mean = train["raw"].mean((0, 1, 3, 4))
    std = train["raw"].std((0, 1, 3, 4)).clamp_min(1e-6)
    manifest["normalization"] = {"mean": mean.tolist(), "std": std.tolist()}
    manifest["train_class_counts"] = torch.bincount(train["truth"], minlength=2).tolist()
    manifest["raw_sha256"] = {
        "train": tensor_digest(train["raw"]),
        "development": tensor_digest(dev["raw"]),
    }
    rng = np.random.default_rng(declaration["sampler_seed"])
    schedule = np.stack(
        [
            rng.choice(len(train["raw"]) // 2, declaration["batch_backgrounds"], replace=False)
            for _ in range(declaration["maximum_updates"])
        ]
    )
    write_json(args.output / "schedule.json", schedule.tolist())
    write_json(args.output / "manifest.json", manifest)
    development = {}
    for seed in declaration["model_seeds"]:
        torch.manual_seed(declaration["initialization_seed_offset"] + seed)
        model = RetrainedMeasurementScorer(mean, std)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        output = args.output / f"seed{seed}"
        output.mkdir()
        snapshots = {}
        manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters())
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
        development[str(seed)] = snapshots
    extension = allow_extension(development, args.smoke)
    write_json(args.output / "extension_decision.json", extension)
    final_step = declaration["initial_updates"]
    if extension["extend"]:
        final_step = declaration["maximum_updates"]
        for seed in declaration["model_seeds"]:
            output = args.output / f"seed{seed}"
            saved = torch.load(output / "checkpoint200.pt", map_location="cpu", weights_only=False)
            model = RetrainedMeasurementScorer(mean, std)
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
                development[str(seed)],
                declaration["snapshots"],
            )
    write_json(
        args.output / "development_complete.json",
        {seed: snapshots[str(final_step)]["gates"] for seed, snapshots in development.items()},
    )
    # This dataset is first generated after training/extension choices are final.
    held = evaluation_data(declaration["holdout_seeds"])
    torch.save(held, args.output / "holdout_data.pt")
    holdout = {}
    for seed in declaration["model_seeds"]:
        output = args.output / f"seed{seed}"
        saved = torch.load(
            output / f"checkpoint{final_step}.pt", map_location="cpu", weights_only=False
        )
        model = RetrainedMeasurementScorer(mean, std).eval()
        model.load_state_dict(saved["model"])
        holdout[str(seed)] = assess(model, held["raw"], held)
        write_json(output / "holdout.json", holdout[str(seed)])
    assert hashes == {str(path): file_digest(path) for path in dependencies}
    write_json(args.output / "manifest.json", manifest)
    summary = {
        "final_step": final_step,
        "extension": extension,
        "acceptance": measurement_acceptance(development, holdout, final_step, args.smoke),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "production_encoder_updates": 0,
        "source_unchanged": True,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
