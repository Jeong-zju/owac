"""Audit saved ratio trials without selecting models or changing the experiment.

Reconstruct gates with independent arithmetic, replay cached predictions, check
fresh RAW/stem identities, and reproduce seed0's existing 100->200 trajectory.
The latter duplicates 200 optimizer updates across two arms for verification;
it never exceeds step200 or creates a new trained candidate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from owac.baselines.n02_ratio import PrefixRatioHead
from owac.baselines.nre_losses import BNRELoss, NRELoss
from owac.evaluation.n02_measurement_isolation import file_digest, tensor_digest, write_json
from owac.evaluation.n02_ratio import evaluation_data, frozen_features, training_data
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder


def read(path):
    return json.loads(Path(path).read_text())


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -700, 700)))


def audit_report(report, zero_prefix):
    """Reconstruct gates from stored raw logBF trajectories, not stored booleans."""
    pairs, corrected, no_cue = {}, {"0": 0, "1": 0}, 0
    for index, row in enumerate(report["records"]):
        gap = np.array(row["log_bayes_factor"])
        np.testing.assert_allclose(sigmoid(gap), row["p1"], rtol=1e-12, atol=1e-12)
        world = row["world"]
        if world == "no_cue":
            drift = max(
                np.max(np.abs(sigmoid(gap + math.log(p / (1 - p))) - p)) for p in (0.05, 0.5, 0.95)
            )
            no_cue += int(drift <= 0.15)
            assert row["passed"] == (drift <= 0.15)
        else:
            truth = int(world)
            mass = sigmoid(gap * (2 * truth - 1) - math.log(19))
            mask = zero_prefix[index].numpy()
            passes = bool(mask.any() and mass[mask][-1] <= 0.25 and mass[-1] >= 0.7)
            np.testing.assert_allclose(mass, row["wrong_prior_truth_mass"], atol=1e-12)
            assert row["passed"] == passes
            corrected[world] += int(passes)
            pairs.setdefault(row["seed"], {})[world] = gap[-1]
    both = sum(int(worlds["0"] < 0 < worlds["1"]) for worlds in pairs.values())
    assert both == report["pair_both_correct"]
    assert corrected == report["wrong_prior_corrected"]
    assert no_cue == report["no_cue_pass"]
    qualified = len(pairs) == 16 and both >= 12 and no_cue >= 12 and min(corrected.values()) >= 4
    assert report["qualified"] == qualified
    changes = [worlds["1"] - worlds["0"] for worlds in pairs.values()]
    return {
        "pair_both_correct": both,
        "no_cue_pass": no_cue,
        "wrong_prior_corrected": corrected,
        "qualified": qualified,
        "swap_correct_direction_count": int(np.sum(np.array(changes) > 0)),
        "swap_logbf_change_median": float(np.median(changes)),
        "swap_logbf_change_range": [float(min(changes)), float(max(changes))],
    }


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            equal_tree(a, b)
    else:
        assert first == second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    declaration = read(root / "protocol.json")
    manifest = read(root / "manifest.json")
    assert not declaration["smoke"]
    for path, digest in manifest["source_sha256"].items():
        assert file_digest(path) == digest, path
    assert file_digest(root / "protocol.json") == manifest["protocol_sha256"]
    data = load(root / "data.pt")
    held = load(root / "holdout_data.pt")
    regenerated = {
        "train": training_data(declaration["train_episode_seeds"]),
        "development": evaluation_data(declaration["development_seeds"]),
        "holdout": evaluation_data(declaration["holdout_seeds"]),
    }
    equal_tree(data["train"], regenerated["train"])
    equal_tree(data["development"], regenerated["development"])
    equal_tree(held, regenerated["holdout"])
    schedule = np.array(read(root / "schedule.json"))
    rng = np.random.default_rng(declaration["sampler_seed"])
    schedule_expected = np.stack(
        [
            rng.choice(len(data["train"]["raw"]), declaration["batch_episodes"], replace=False)
            for _ in range(declaration["updates"])
        ]
    )
    np.testing.assert_array_equal(schedule, schedule_expected)
    results, replayed_predictions = {}, 0
    for seed in declaration["seeds"]:
        output = root / f"seed{seed}"
        cache = load(output / "features.pt")
        cache["holdout"] = load(output / "holdout_features.pt")
        identity = manifest["checkpoints"][str(seed)]
        assert file_digest(identity["path"]) == identity["sha256"]
        payload = load(identity["path"])
        source = N02Encoder(N02Config.from_dict(payload["config"])).eval()
        source.load_state_dict(payload["model"])
        source.requires_grad_(False)
        for split in ("train", "development", "holdout"):
            # Recompute a real prefix from RAW in each split, independently of saved features.
            actual = frozen_features(source.measurement.stem, regenerated[split]["raw"][:1])
            torch.testing.assert_close(actual, cache[split][:1], rtol=1e-5, atol=1e-6)
        cache_record = read(output / "cache_manifest.json")
        for split in ("train", "development"):
            assert tensor_digest(cache[split]) == cache_record[split]["feature_sha256"]
        results[str(seed)] = {}
        for method in declaration["methods"]:
            dev = read(output / f"{method}_development.json")
            final = read(output / "holdout.json")[method]
            results[str(seed)][method] = {}
            for step in declaration["snapshot_steps"]:
                head = PrefixRatioHead().eval()
                head.load_state_dict(load(output / f"{method}_step{step}.pt")["head"])
                splits = [("development", dev[str(step)], data["development"])]
                if step == 200:
                    splits.append(("holdout", final, held))
                for split, report, observations in splits:
                    metrics = audit_report(report, observations["zero_prefix"])
                    with torch.no_grad():
                        scores = torch.cat([head.scores(piece) for piece in cache[split].split(16)])
                    gaps = scores.double()[..., 1] - scores.double()[..., 0]
                    expected = torch.tensor(
                        [r["log_bayes_factor"] for r in report["records"]], dtype=torch.float64
                    )
                    torch.testing.assert_close(gaps, expected, rtol=0, atol=0)
                    replayed_predictions += gaps.numel()
                    results[str(seed)][method][f"{split}_{step}"] = metrics
        equal_tree(load(output / "nre_step0.pt")["head"], load(output / "bnre_step0.pt")["head"])
        contrast = cache["holdout"][1::3] - cache["holdout"][::3]
        results[str(seed)]["frozen_stem_pair_difference_rms"] = float(
            contrast.square().mean().sqrt()
        )
        if seed == 0:
            for method in declaration["methods"]:
                saved = load(output / f"{method}_step100.pt")
                head = PrefixRatioHead().train()
                head.load_state_dict(saved["head"])
                optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
                optimizer.load_state_dict(saved["optimizer"])
                objective = NRELoss(head) if method == "nre" else BNRELoss(head, 100)
                for indexes in schedule[100:]:
                    theta = (data["train"]["truth"][indexes].float() * 2 - 1)[:, None]
                    optimizer.zero_grad(set_to_none=True)
                    objective(theta, cache["train"][indexes]).backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                expected = load(output / f"{method}_step200.pt")
                equal_tree(head.state_dict(), expected["head"])
                equal_tree(optimizer.state_dict(), expected["optimizer"])
    receipt = {
        "passed": True,
        "raw_regeneration": "all training/dev/holdout bytes identical",
        "prefix_predictions_replayed": replayed_predictions,
        "checkpoint_source_input_hashes": "passed",
        "source_stem_replay": "one full sequence per split per seed",
        "seed0_both_methods_resume_100_to_200": "model and AdamW bitwise identical",
        "duplicated_audit_optimizer_updates": 200,
        "new_training_beyond_step200": 0,
        "results": results,
    }
    write_json(root / "audit.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
