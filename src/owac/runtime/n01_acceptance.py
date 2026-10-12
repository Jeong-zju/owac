"""Separate, hashed N01 acceptance bundle and real two-platform smoke suite.

Never alter rawvla_full_eval.py's existing baseline freeze. This bundle includes
all OWAC Python dependencies, the feature bridge, configuration, and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = sorted([*source.glob("src/owac/**/*.py"), *source.glob("configs/encoders/n01/*.json")])
    manifest = {
        "format": "owac-n01-acceptance-v1",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=source)
        ).hexdigest(),
        "files": {str(path.relative_to(source)): digest(path) for path in files},
        "baseline_snapshot_modified": False,
    }
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(source)))
        archive.add(output / "source-manifest.json", arcname="source-manifest.json")
    return manifest


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"acceptance source hash mismatch: {name}")
    return manifest


def suite(root, output, steps):
    root, output = Path(root), Path(output)
    verify(Path(__file__).resolve().parents[3])
    tokenizer = root / "source/openpi_converted_protocol/paligemma_tokenizer.model"
    results = []
    for platform in ("libero", "robotwin2"):
        weights = root / (
            "source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors"
            if platform == "libero"
            else "checkpoints/pi05_robotwin/model.safetensors"
        )
        for method, learned, auxiliary in (
            ("n01", False, 0.0),
            ("n01", True, 0.0),
            ("n01", False, 0.1),
            ("n01", True, 0.1),
            ("c3_raw_cnn", False, 0.0),
            ("c4_fixed_patch", False, 0.0),
        ):
            name = f"{platform}-{method}-learned{int(learned)}-aux{auxiliary}"
            dataset_name = "libero" if platform == "libero" else "robotwin"
            command = [
                sys.executable,
                "-m",
                "owac.training.n01_action",
                "--suite",
                platform,
                "--config",
                "configs/encoders/n01/tiny.json",
                "--weights",
                str(weights),
                "--tokenizer",
                str(tokenizer),
                "--dataset-config",
                f"configs/encoders/n01/rawvla-{dataset_name}-smoke.json",
                "--steps",
                str(steps),
                "--window",
                "4",
                "--crop",
                "32",
                "--method",
                method,
                "--aux-weight",
                str(auxiliary),
                "--output",
                str(output / name),
            ]
            if learned:
                command.append("--learned-support")
            subprocess.run(command, check=True)
            record = json.loads((output / name / "run.json").read_text())
            results.append(
                {
                    "name": name,
                    "loss_first": record["rows"][0]["action_loss"],
                    "loss_last": record["rows"][-1]["action_loss"],
                    "steps": len(record["rows"]),
                    "peak_gpu_bytes": record["peak_gpu_bytes"],
                    "inference": record["inference"],
                    "encoder_parameter_update": record["encoder_parameter_update"],
                    "record_sha256": digest(output / name / "run.json"),
                }
            )
            (output / "suite.json").write_text(json.dumps(results, indent=2) + "\n")
    verify(Path(__file__).resolve().parents[3])
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--source", type=Path, default=Path.cwd())
    b.add_argument("--output", type=Path, required=True)
    v = commands.add_parser("verify")
    v.add_argument("--root", type=Path, required=True)
    s = commands.add_parser("suite")
    s.add_argument("--root", type=Path, default=Path("/workspace/owac/rawvla"))
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--steps", type=int, default=20)
    args = p.parse_args(argv)
    if args.command == "build":
        result = build(args.source, args.output)
    elif args.command == "verify":
        result = verify(args.root)
    else:
        result = suite(args.root, args.output, args.steps)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
