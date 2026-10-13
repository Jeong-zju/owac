"""Recompute behavior decisions and verify frozen sources, inputs and replay."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_acceptance import evaluate
from owac.frameworks.n02.config import N02Config
from owac.training.n02 import synthetic_batch

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "outputs/n02-contracts-20261013"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(1)
    manifest, summary = (load(OUTPUT / name) for name in ("manifest.json", "summary.json"))
    assert all(digest(ROOT / p) == h for p, h in manifest["source_sha256"].items())
    assert digest(Path(__file__).with_name("protocol_before_run.md")) == manifest["protocol_sha256"]
    assert not set(manifest["development_seeds"]) & set(manifest["final_seeds"])
    decisions, progress = [], {100: 0, 200: 0}
    evaluation_hashes, input_sets = {}, {"dev": set(), "final": set()}
    record_count, duplicate_max = 0, 0.0
    for path in sorted(OUTPUT.glob("*.json")):
        if not (path.name.startswith("dev_seed") or path.name.startswith("final_seed")):
            continue
        report = load(path)
        evaluation_hashes[path.name] = digest(path)
        group = "dev" if path.name.startswith("dev_") else "final"
        records = report["records"]
        record_count += len(records)
        assert all(len(row["steps"]) == 8 for row in records)
        for condition, counts in report["gates"].items():
            rows = [row for row in records if row["condition"] == condition]
            assert counts == {"passed": sum(row["gate_passed"] for row in rows), "total": len(rows)}
        for seed in {row["seed"] for row in records}:
            identical_raw = [
                row
                for row in records
                if row["seed"] == seed and row["condition"] in {"correct", "misleading", "missing"}
            ]
            assert len({row["measurement_sha256"] for row in identical_raw}) == 1
        for row in records:
            input_sets[group].add(row["measurement_sha256"])
            duplicate_max = max(duplicate_max, row["duplicate_weight_delta"])
        pairs = report["counterfactual"]["records"]
        for row in pairs:
            input_sets[group].update(row[f"measurement_sha256_{truth}"] for truth in (0, 1))
        assert report["counterfactual"]["both_correct"] == sum(
            row["p1_when_0"] < 0.5 < row["p1_when_1"] for row in pairs
        )
    assert not input_sets["dev"] & input_sets["final"]
    for seed in (0, 1, 2):
        stages = summary["development"][str(seed)]
        counts = stages["200"]["gates"]
        decisions.append(
            all(
                counts[key]["passed"] >= threshold
                for key, threshold in (
                    ("correct", 12),
                    ("misleading", 4),
                    ("unidentifiable", 12),
                    ("stale", 4),
                )
            )
            and stages["200"]["counterfactual"]["both_correct"] >= 8
        )
        for step in (100, 200):
            progress[step] += sum(
                stages[str(step)]["gates"][key]["passed"]
                for key in ("misleading", "unidentifiable", "stale")
            )
    expected = sum(decisions) >= 2 and progress[200] >= progress[100]
    assert expected == summary["decision"]["extend_to_800"]
    training_input_hashes, conditions, clipped = set(), {}, []
    steps = summary["final_step"]
    for seed in (0, 1, 2):
        destination = OUTPUT / f"seed{seed}"
        payload = torch.load(
            destination / f"checkpoint{steps}.pt", map_location="cpu", weights_only=False
        )
        assert payload["step"] == steps and payload["seed"] == seed
        assert all(torch.isfinite(value).all() for value in payload["model"].values())
        logs = [
            json.loads(line) for line in (destination / "train_log.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in logs] == list(range(steps))
        assert all(math.isfinite(row[key]) for row in logs for key in ("loss", "grad_norm"))
        clipped.extend(row["grad_norm"] > 1 for row in logs)
        rng = np.random.default_rng(seed)
        config = N02Config.from_dict(payload["config"])
        for _ in range(steps):
            batch = synthetic_batch(
                config, rng, batch_size=2, num_frames=8, frame_size=64, seed=seed, split="train"
            )
            for index in range(2):
                raw = np.stack([frame[index].numpy() for frame in batch.values])
                training_input_hashes.add(hashlib.sha256(raw.tobytes()).hexdigest())
            for condition in batch.notes["prior_conditions"]:
                conditions[condition] = conditions.get(condition, 0) + 1
        assert rng.bit_generator.state == payload["rng"]["numpy_bit_generator"]
    assert not training_input_hashes & (input_sets["dev"] | input_sets["final"])
    replay = evaluate(OUTPUT / f"seed0/checkpoint{steps}.pt", seeds=[600])
    original = load(OUTPUT / "final_seed0.json")
    assert replay["records"] == [row for row in original["records"] if row["seed"] == 600]
    historical = load(OUTPUT / "historical_reference.json")
    identities = {
        (row["seed"], row["condition"]): row["measurement_sha256"] for row in original["records"]
    }
    assert identities == {
        (row["seed"], row["condition"]): row["measurement_sha256"] for row in historical["records"]
    }
    report = {
        "passed": True,
        "source_hashes_match": True,
        "protocol_hash_matches": True,
        "recomputed_extension": expected,
        "qualified_seeds": decisions,
        "progress": progress,
        "formal_optimizer_steps": 3 * steps,
        "ordinary_evaluation_records": record_count,
        "train_dev_final_measurement_overlap": 0,
        "regenerated_training_unique_sequences": len(training_input_hashes),
        "training_conditions": conditions,
        "gradient_clipped_fraction": float(np.mean(clipped)),
        "checkpoint_rng_matches_regenerated_input_stream": True,
        "seed0_final_seed600_records_exact_replay": True,
        "historical_reference_uses_identical_measurements": True,
        "duplicate_max_weight_delta": duplicate_max,
        "duplicate_scope": (
            "a learned repeated-ID solver may refine a fixed likelihood; "
            "fixed-field tests separately prove no double-counting"
        ),
        "artifact_hashes": evaluation_hashes,
        "diagnostic_records_sha256": digest(OUTPUT / "learned_diagnostic_records.json"),
        "training_input_manifest_sha256": hashlib.sha256(
            "\n".join(sorted(training_input_hashes)).encode()
        ).hexdigest(),
    }
    (Path(__file__).parent / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
