"""Equal-budget old-rule versus repaired-rule study on identical corrected RAW.

Run from the checkout with uv run --locked python <this file>. Large outputs
stay under outputs/. The legacy implementation is reconstructed from the audited
commit; only the shared renderer, input-quality labels and evaluator are replaced.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BASE = "7c567cc33bb6abc382489c4f75c4a3ba810e6d43"
ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/n02-repair-20261013"
SEEDS = tuple(range(100, 116))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe():
    import numpy as np

    from owac.frameworks.n02.config import N02Config
    from owac.training.n02 import synthetic_batch

    rng = np.random.default_rng(0)
    data = json.loads((ROOT / "configs/encoders/n02/tiny.json").read_text())
    if not hasattr(N02Config().revision, "propagation_threshold"):
        data["revision"].pop("propagation_threshold", None)
    config = N02Config.from_dict(data)
    combined = hashlib.sha256()
    priors, informative = [], 0
    for _ in range(200):
        batch = synthetic_batch(
            config, rng, batch_size=2, num_frames=8, frame_size=64, seed=0, split="train"
        )
        for frame in batch.values:
            combined.update(frame.numpy().tobytes())
        combined.update(batch.contexts[0].hypothesis_prior.numpy().tobytes())
        combined.update(batch.identifiable_now.numpy().tobytes())
        informative += int(batch.identifiable_now.sum())
        priors.extend(batch.contexts[0].hypothesis_prior.max(-1).values.tolist())
    return {
        "sha256_measurements_context_and_quality": combined.hexdigest(),
        "sequences": 400,
        "frames": 3200,
        "informative_frames": informative,
        "sequences_with_prior_slot_mass_ge_0_475": sum(p >= 0.475 - 1e-6 for p in priors),
        "config": config.as_dict(),
    }


def child_environment(source):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(source), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def run_child(command, source, log):
    with log.open("w") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment(source),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe()))
        return
    if args.evaluate:
        from owac.evaluation.n02_acceptance import evaluate

        report = evaluate(args.evaluate, seeds=SEEDS)
        args.evaluate.with_name("learned.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["gates"]))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = OUTPUT / "before"
    if not (before / "src").exists():
        data = subprocess.run(
            ["git", "archive", BASE, "src", "configs"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        before.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(before, filter="data")
    original_sensor = before / "synthetic_original.py"
    if not original_sensor.exists():
        original_sensor.write_bytes((before / "src/owac/frameworks/n02/synthetic.py").read_bytes())
    for relative in (
        "src/owac/frameworks/n02/synthetic.py",
        "src/owac/evaluation/n02_acceptance.py",
    ):
        (before / relative).write_bytes((ROOT / relative).read_bytes())
    path = before / "src/owac/training/n02.py"
    text = path.read_text().replace(
        "identifiable_now[:, index] = torch.from_numpy(episode.cue_strength > 0.0)",
        (
            "identifiable_now[:, index] = torch.tensor("
            'generator.measurement_quality(episode)["identifiable_now"])'
        ),
    )
    path.write_text(text)
    arms = (
        ("data_only", "candidate", before / "src", before / "configs/encoders/n02/tiny.json"),
        ("repaired", "candidate", ROOT / "src", ROOT / "configs/encoders/n02/tiny.json"),
        (
            "no_propagation",
            "no_dependency_propagation",
            ROOT / "src",
            ROOT / "configs/encoders/n02/tiny.json",
        ),
        (
            "no_reset",
            "no_contradiction_reset",
            ROOT / "src",
            ROOT / "configs/encoders/n02/tiny.json",
        ),
    )
    plan = {
        "base_commit": BASE,
        "training_seed": 0,
        "validation_seeds": SEEDS,
        "steps": 200,
        "batch_size": 2,
        "sequence_length": 8,
        "frame_size": 64,
        "learning_rate": 3e-4,
        "threads": 1,
        "arms": [a[0] for a in arms],
        "scope": "CPU binary semantic proxy smoke; no robot or superiority claim",
        "comparison": "same corrected renderer, strong priors and quality labels",
        "gates": "fixed thresholds; explicit cue/no cue; stale always changes",
    }
    (RESULTS / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    probes = {}
    for name, source in (("data_only", before / "src"), ("repaired", ROOT / "src")):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe"],
            cwd=ROOT,
            env=child_environment(source),
            capture_output=True,
            text=True,
            check=True,
        )
        probes[name] = json.loads(result.stdout)
    assert (
        probes["data_only"]["sha256_measurements_context_and_quality"]
        == probes["repaired"]["sha256_measurements_context_and_quality"]
    )
    (RESULTS / "data_comparison.json").write_text(json.dumps(probes, indent=2) + "\n")
    print("Matched 3200 RAW frames, prior context and quality labels", flush=True)
    receipts = {}
    for name, arm, source, config in arms:
        dest = OUTPUT / name
        dest.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "owac.training.n02",
            "--config",
            str(config),
            "--synthetic",
            "--steps",
            "200",
            "--seed",
            "0",
            "--output",
            str(dest),
            "--arm",
            arm,
            "--threads",
            "1",
            "--log-every",
            "50",
        ]
        print(f"Training {name}: 200 steps", flush=True)
        start = time.monotonic()
        run_child(command, source, dest / "process.log")
        run_child(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--evaluate",
                str(dest / "checkpoint.pt"),
            ],
            source,
            dest / "evaluation.log",
        )
        learned = json.loads((dest / "learned.json").read_text())
        summary = json.loads((dest / "summary.json").read_text())
        records = [json.loads(line) for line in (dest / "train_log.jsonl").read_text().splitlines()]
        ambiguous = [r for r in learned["records"] if r["condition"] == "unidentifiable"]
        drift = [abs(s["semantic_mass"][0] - 0.5) for r in ambiguous for s in r["steps"]]
        receipts[name] = {
            "arm": arm,
            "checkpoint_sha256": digest(dest / "checkpoint.pt"),
            "train_log_sha256": digest(dest / "train_log.jsonl"),
            "evaluation_sha256": digest(dest / "learned.json"),
            "gates": learned["gates"],
            "loss_first": records[0]["loss"],
            "loss_last": records[-1]["loss"],
            "train_summary": summary,
            "no_cue_mean_absolute_drift": sum(drift) / len(drift),
            "no_cue_max_absolute_drift": max(drift),
            "encoder_ms": learned["encoder_ms"],
            "wall_seconds_training_and_evaluation": time.monotonic() - start,
        }
        (RESULTS / "comparison.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(
            json.dumps(
                {name: {"gates": learned["gates"], "mean_no_cue_drift": sum(drift) / len(drift)}}
            ),
            flush=True,
        )
    # All independently trained arms must be scored on exactly the same RAW.
    reports = [json.loads((OUTPUT / a[0] / "learned.json").read_text()) for a in arms]
    identities = [
        [(r["seed"], r["condition"], r["measurement_sha256"]) for r in p["records"]]
        for p in reports
    ]
    assert all(rows == identities[0] for rows in identities)
    hashes = {}
    for name, source in (("data_only", before / "src"), ("repaired", ROOT / "src")):
        hashes[name] = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))}
    (RESULTS / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("All four arms evaluated on identical 80 sequences; comparison complete", flush=True)


if __name__ == "__main__":
    main()
