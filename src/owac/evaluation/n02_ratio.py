"""Predeclared NRE/BNRE frozen-tiny-stem comparison, with independent episode negatives.

Run from the checkout root. This stage tests static two-explanation measurement
evidence only; it never installs a global ratio as a local refutation map.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from owac.baselines.n02_ratio import PrefixRatioHead, posterior_from_prefix
from owac.baselines.nre_losses import BNRELoss, NRELoss
from owac.evaluation.n02_bottlenecks import paired_episodes
from owac.evaluation.n02_measurement_isolation import (
    file_digest,
    state_digest,
    tensor_digest,
    write_json,
)
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import SyntheticGenerator, SyntheticOptions, values_for_task


def protocol(smoke=False):
    return {
        "stage": "independent measurement evidence; conditional filter/N02 comparison pending",
        "smoke": smoke,
        "seeds": [0] if smoke else [0, 1, 2],
        "methods": ["nre", "bnre"],
        "updates": 2 if smoke else 200,
        "snapshot_steps": [0, 1, 2] if smoke else [0, 100, 200],
        "batch_episodes": 4 if smoke else 16,
        "train_episode_seeds": list(range(16000, 16008 if smoke else 16256)),
        "development_seeds": [17999] if smoke else list(range(18000, 18016)),
        "holdout_seeds": [18999] if smoke else list(range(19000, 19016)),
        "task": "opening_direction",
        "input": "same synthetic 3-channel pseudo-RAW, side64, 8 causal frames, pool16",
        "split": "train versus disjoint validation geometry; dev/holdout differ in seeds only",
        "hypotheses": "fixed opening-left/right (-1/+1); no inferred geometry",
        "train_identifiable_probability": 0.8,
        "noise_sigma": 0.01,
        "stem": "each seed's frozen original tiny-contracts checkpoint200 RAWStem",
        "head": "1x1 width32->4, spatial linear1024->32, GRU32, shared ratio MLP33->32->1",
        "history": "causal GRU32 state, reset every episode; prefix scores replace evidence",
        "negative_sampling": "roll theta across independent distinct episodes, never time",
        "bnre_lambda": 100.0,
        "optimizer": "AdamW lr3e-4 weight_decay0.01 clip1.0",
        "head_seed_offset": 17000,
        "sampler_seed": 17000,
        "matched_budget": "NRE and BNRE same initialization, inputs, prefixes and 200 updates",
        "old_budget_difference": "B16/T8 vs previous end-to-end B2/T8; frozen stem and new head",
        "supervision": "sampled binary world parameter only; no templates or quality masks in loss",
        "temperature_or_offset_calibration": "none",
        "criteria": {
            "pair_both_correct_min": 12,
            "denominator": 16,
            "no_cue_all_prefix_prior_drift_max": 0.15,
            "no_cue_sequences_min": 12,
            "tested_no_cue_priors_p1": [0.05, 0.5, 0.95],
            "wrong_prior_final_truth_min": 0.7,
            "wrong_prior_last_zero_prefix_truth_max": 0.25,
            "wrong_prior_success_min_per_world": 4,
            "qualified_seeds_min_each_split": 2,
            "aggregate_behavior_count_100_to_200_nondecreasing": True,
        },
        "second_stage": "only if same method qualifies on dev and untouched holdout",
        "production_revision_and_absolute_fit": "unchanged and not bypassed",
    }


def training_data(seeds):
    """Generate independent static episodes. No paired-world forcing in negatives."""
    raws, truths, identities = [], [], []
    for seed in seeds:
        gen = SyntheticGenerator(SyntheticOptions(height=64, width=64), seed=seed, split="train")
        episode = gen.generate(seed=seed, episode_index=seed, prior_condition="missing")
        assert len(set(episode.value_over_time)) == 1
        raw = torch.from_numpy(episode.frames.copy())
        truth = values_for_task(episode.task).index(episode.value)
        raws.append(raw)
        truths.append(truth)
        identities.append(
            {
                "seed": seed,
                "episode_id": episode.episode_id,
                "truth": truth,
                "identifiable": episode.identifiable,
                "raw_sha256": tensor_digest(raw),
            }
        )
    assert len({row["episode_id"] for row in identities}) == len(identities)
    return {"raw": torch.stack(raws), "truth": torch.tensor(truths), "identities": identities}


def evaluation_data(seeds):
    """Two same-background/noise worlds plus true no-cue sequences per scene."""
    raws, identities, oracle_scores, zero_prefixes = [], [], [], []
    for seed in seeds:
        gen, episodes = paired_episodes(seed)
        base = episodes["missing"]
        templates = np.stack([gen.measurement_templates(base, i) for i in range(8)])
        worlds = [
            (str(truth), (templates[:, truth] + base.noise).astype(np.float32), templates)
            for truth in range(2)
        ]
        no_cue = episodes["unidentifiable"]
        no_cue_templates = np.stack([gen.measurement_templates(no_cue, i) for i in range(8)])
        np.testing.assert_array_equal(no_cue_templates[:, 0], no_cue_templates[:, 1])
        worlds.append(("no_cue", no_cue.frames, no_cue_templates))
        for world, raw_array, template_array in worlds:
            raw = torch.from_numpy(raw_array.copy())
            observed = F.avg_pool2d(raw.double(), 4)
            template = F.avg_pool2d(torch.from_numpy(template_array).double().flatten(0, 1), 4)
            template = template.reshape(8, 2, 3, 16, 16)
            variance = gen.options.noise_sigma**2 / 16
            scores = (-0.5 * (observed[:, None] - template).square() / variance).sum((2, 3, 4))
            gap = (template[:, 1] - template[:, 0]).square().sum((1, 2, 3))
            zero_prefixes.append(gap.cumsum(0) == 0)
            oracle_scores.append(scores.cumsum(0))
            raws.append(raw)
            identities.append({"seed": seed, "world": world, "raw_sha256": tensor_digest(raw)})
    return {
        "raw": torch.stack(raws),
        "identities": identities,
        "oracle_scores": torch.stack(oracle_scores),
        "zero_prefix": torch.stack(zero_prefixes),
    }


@torch.no_grad()
def frozen_features(stem, raw):
    """Same all-valid RAW pooling and stem as tiny; no recurrent N02 state/prior."""
    batch, steps = raw.shape[:2]
    if not torch.isfinite(raw).all():
        raise ValueError("this synthetic protocol requires finite all-valid RAW")
    observed = F.avg_pool2d(raw.flatten(0, 1), 4)
    chunks = []
    for piece in observed.split(64):
        valid = torch.ones_like(piece, dtype=torch.bool)
        absent = torch.zeros_like(valid)
        chunks.append(stem(piece, valid, absent, absent, None))
    return torch.cat(chunks).reshape(batch, steps, -1, 16, 16).detach()


def summarize_scores(scores, data):
    """All gates use realized measurements; templates only identify empty prefixes."""
    scores = scores.double()
    if scores.shape != (len(data["identities"]), 8, 2) or not torch.isfinite(scores).all():
        raise ValueError("invalid complete score trajectory")
    gaps = scores[..., 1] - scores[..., 0]
    uniform = posterior_from_prefix(scores)
    rows, pair_rows, calibration = [], {}, []
    corrected = {"0": 0, "1": 0}
    no_cue_pass = 0
    for index, identity in enumerate(data["identities"]):
        row = dict(identity, log_bayes_factor=gaps[index].tolist(), p1=uniform[index].tolist())
        world = identity["world"]
        if world == "no_cue":
            drifts = {
                str(p): float((posterior_from_prefix(scores[index], p) - p).abs().max())
                for p in (0.05, 0.5, 0.95)
            }
            row["max_prefix_prior_drift"] = drifts
            row["passed"] = max(drifts.values()) <= 0.15
            no_cue_pass += int(row["passed"])
        else:
            truth = int(world)
            wrong_prior_p1 = 0.05 if truth else 0.95
            p1 = posterior_from_prefix(scores[index], wrong_prior_p1)
            mass = p1 if truth else 1 - p1
            zero = data["zero_prefix"][index]
            row["wrong_prior_truth_mass"] = mass.tolist()
            row["passed"] = bool(zero.any() and mass[zero][-1] <= 0.25 and mass[-1] >= 0.7)
            corrected[world] += int(row["passed"])
            pair_rows.setdefault(identity["seed"], {})[world] = bool(
                uniform[index, -1] > 0.5 if truth else uniform[index, -1] < 0.5
            )
            calibration.extend((float(p), truth) for p in uniform[index])
        rows.append(row)
    pairs = sum(all(worlds.values()) and len(worlds) == 2 for worlds in pair_rows.values())
    no_cue_gaps = gaps[[r["world"] == "no_cue" for r in data["identities"]]].abs()
    probabilities = torch.tensor([p for p, _ in calibration], dtype=torch.float64)
    labels = torch.tensor([y for _, y in calibration], dtype=torch.float64)
    ece = 0.0
    bins = (probabilities * 10).long().clamp(max=9)
    for bin_index in range(10):
        mask = bins == bin_index
        if mask.any():
            ece += float(
                mask.double().mean() * (probabilities[mask].mean() - labels[mask].mean()).abs()
            )
    gates = {
        "paired": pairs >= 12 and len(pair_rows) == 16,
        "no_cue": no_cue_pass >= 12 and len(pair_rows) == 16,
        "wrong_prior_world0": corrected["0"] >= 4 and len(pair_rows) == 16,
        "wrong_prior_world1": corrected["1"] >= 4 and len(pair_rows) == 16,
    }
    return {
        "pair_both_correct": pairs,
        "scene_count": len(pair_rows),
        "no_cue_pass": no_cue_pass,
        "wrong_prior_corrected": corrected,
        "no_cue_mean_abs_logbf": float(no_cue_gaps.mean()),
        "no_cue_p95_abs_logbf": float(no_cue_gaps.flatten().quantile(0.95)),
        "paired_prefix_brier": float((probabilities - labels).square().mean()),
        "paired_prefix_ece10": ece,
        "calibration_scope": "paired worlds, correlated prefixes; descriptive not IID trials",
        "gates": gates,
        "qualified": all(gates.values()),
        "records": rows,
    }


@torch.no_grad()
def assess(head, features, data):
    scores = torch.cat([head.scores(chunk) for chunk in features.split(16)])
    return summarize_scores(scores, data)


def cache_manifest(features, data):
    return {"feature_sha256": tensor_digest(features), "identities": data["identities"]}


def fit(initial, features, truth, dev_features, dev, schedule, method, output, declaration):
    head = copy.deepcopy(initial)
    objective = NRELoss(head) if method == "nre" else BNRELoss(head, declaration["bnre_lambda"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    logs, snapshots = [], {}

    def snapshot(step):
        head.eval()
        snapshots[str(step)] = assess(head, dev_features, dev)
        with torch.no_grad():
            training_scores = torch.cat([head.scores(piece) for piece in features.split(16)])
            margin = (training_scores[..., 1] - training_scores[..., 0]) * (truth[:, None] * 2 - 1)
            snapshots[str(step)]["train_evidence"] = {
                "final_accuracy": float((margin[:, -1] > 0).float().mean()),
                "mean_final_truth_logbf": float(margin[:, -1].mean()),
                "max_final_abs_logbf": float(margin[:, -1].abs().max()),
            }
        torch.save(
            {"head": head.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            output / f"{method}_step{step}.pt",
        )
        write_json(output / f"{method}_development.json", snapshots)
        head.train()

    snapshot(0)
    started = time.perf_counter()
    for step, indexes in enumerate(schedule, 1):
        # Every theta_i/x_i pair is one independent episode. The rolled theta
        # may legitimately equal theta_i; negatives are NOT forced wrong labels.
        if len(set(indexes.tolist())) != len(indexes):
            raise ValueError("negative-pair batch must contain distinct episodes")
        theta = (truth[indexes].float() * 2 - 1)[:, None]
        optimizer.zero_grad(set_to_none=True)
        loss = objective(theta, features[indexes])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f"{method} step{step}: nonfinite objective/gradient")
        optimizer.step()
        logs.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "same_class_negative_fraction": float((theta == theta.roll(1, 0)).float().mean()),
            }
        )
        if step in declaration["snapshot_steps"]:
            snapshot(step)
            print(f"{method} step {step}: {snapshots[str(step)]['gates']}", flush=True)
    elapsed = time.perf_counter() - started
    write_json(output / f"{method}_training.json", {"elapsed_seconds": elapsed, "logs": logs})
    return head.eval(), snapshots


def decision(development, holdout, declaration):
    result = {}
    for method in declaration["methods"]:
        last, middle = map(str, declaration["snapshot_steps"][-2:][::-1])
        counts, dev_qualified, held_qualified = {}, 0, 0
        for step in (middle, last):
            counts[step] = sum(
                row[method][step]["pair_both_correct"]
                + row[method][step]["no_cue_pass"]
                + sum(row[method][step]["wrong_prior_corrected"].values())
                for row in development.values()
            )
        dev_qualified = sum(row[method][last]["qualified"] for row in development.values())
        held_qualified = sum(row[method]["qualified"] for row in holdout.values())
        result[method] = {
            "development_qualified_seeds": dev_qualified,
            "holdout_qualified_seeds": held_qualified,
            "behavior_counts": counts,
            "allow_shared_scorer_comparison": bool(
                not declaration["smoke"]
                and dev_qualified >= 2
                and held_qualified >= 2
                and counts[last] >= counts[middle]
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs/n02-contracts-20261013")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = protocol(args.smoke)
    write_json(args.output / "protocol.json", declaration)
    source_paths = sorted(Path("src/owac/frameworks/n02").glob("*.py")) + [
        Path(__file__).resolve().relative_to(Path.cwd()),
        Path("src/owac/baselines/n02_ratio.py"),
        Path("src/owac/baselines/nre_losses.py"),
        Path("src/owac/evaluation/n02_bottlenecks.py"),
        Path("src/owac/evaluation/n02_measurement_isolation.py"),
        Path("uv.lock"),
    ]
    source_before = {str(p): file_digest(p) for p in source_paths}
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_before,
        "protocol_sha256": file_digest(args.output / "protocol.json"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": 1,
        },
        "checkpoints": {},
    }
    (args.output / "tracked_changes.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"])
    )
    for path in source_paths:
        destination = args.output / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    started = time.perf_counter()
    train = training_data(declaration["train_episode_seeds"])
    dev = evaluation_data(declaration["development_seeds"])
    torch.save({"train": train, "development": dev}, args.output / "data.pt")
    rng = np.random.default_rng(declaration["sampler_seed"])
    schedule = np.stack(
        [
            rng.choice(len(train["raw"]), declaration["batch_episodes"], replace=False)
            for _ in range(declaration["updates"])
        ]
    )
    write_json(args.output / "schedule.json", schedule.tolist())
    manifest["train_class_counts"] = torch.bincount(train["truth"], minlength=2).tolist()
    write_json(
        args.output / "reference_development.json",
        {
            "zero_evidence": summarize_scores(torch.zeros_like(dev["oracle_scores"]), dev),
            "known_template_oracle": summarize_scores(dev["oracle_scores"], dev),
            "oracle_extra_information": "exact templates, evaluator-only, not a learned baseline",
        },
    )
    development = {}
    for seed in declaration["seeds"]:
        output = args.output / f"seed{seed}"
        output.mkdir()
        checkpoint = args.checkpoint_root / f"seed{seed}" / "checkpoint200.pt"
        checkpoint_hash = file_digest(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = N02Config.from_dict(payload["config"])
        source = N02Encoder(config).eval()
        source.load_state_dict(payload["model"])
        source.requires_grad_(False)
        before = state_digest(source)
        stem = source.measurement.stem
        train_features = frozen_features(stem, train["raw"])
        dev_features = frozen_features(stem, dev["raw"])
        torch.save({"train": train_features, "development": dev_features}, output / "features.pt")
        write_json(
            output / "cache_manifest.json",
            {
                "train": cache_manifest(train_features, train),
                "development": cache_manifest(dev_features, dev),
            },
        )
        torch.manual_seed(declaration["head_seed_offset"] + seed)
        initial = PrefixRatioHead(config.stem_width)
        manifest["head_parameters"] = sum(p.numel() for p in initial.parameters())
        manifest["stem_parameters"] = sum(p.numel() for p in stem.parameters())
        manifest["source_encoder_parameters"] = sum(p.numel() for p in source.parameters())
        development[str(seed)] = {}
        for method in declaration["methods"]:
            _, snapshots = fit(
                initial,
                train_features,
                train["truth"],
                dev_features,
                dev,
                schedule,
                method,
                output,
                declaration,
            )
            development[str(seed)][method] = snapshots
        assert before == state_digest(source)
        assert checkpoint_hash == file_digest(checkpoint)
        manifest["checkpoints"][str(seed)] = {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "unchanged": True,
            "stem_state_sha256": state_digest(stem),
        }
        write_json(args.output / "manifest.json", manifest)
    # No holdout generation/evaluation or model choice until all predeclared runs end.
    write_json(
        args.output / "development_complete.json",
        {
            seed: {m: rows[m][str(declaration["updates"])]["gates"] for m in declaration["methods"]}
            for seed, rows in development.items()
        },
    )
    held = evaluation_data(declaration["holdout_seeds"])
    torch.save(held, args.output / "holdout_data.pt")
    holdout = {}
    for seed in declaration["seeds"]:
        output = args.output / f"seed{seed}"
        payload = torch.load(
            args.checkpoint_root / f"seed{seed}" / "checkpoint200.pt",
            map_location="cpu",
            weights_only=False,
        )
        source = N02Encoder(N02Config.from_dict(payload["config"])).eval()
        source.load_state_dict(payload["model"])
        features = frozen_features(source.measurement.stem, held["raw"])
        torch.save(features, output / "holdout_features.pt")
        holdout[str(seed)] = {}
        for method in declaration["methods"]:
            head = PrefixRatioHead().eval()
            checkpoint = torch.load(
                output / f"{method}_step{declaration['updates']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            head.load_state_dict(checkpoint["head"])
            holdout[str(seed)][method] = assess(head, features, held)
        write_json(output / "holdout.json", holdout[str(seed)])
    write_json(
        args.output / "reference_holdout.json",
        {
            "zero_evidence": summarize_scores(torch.zeros_like(held["oracle_scores"]), held),
            "known_template_oracle": summarize_scores(held["oracle_scores"], held),
        },
    )
    assert source_before == {str(p): file_digest(p) for p in source_paths}
    summary = {
        "decision": decision(development, holdout, declaration),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "n02_production_updates": 0,
        "source_unchanged": True,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
