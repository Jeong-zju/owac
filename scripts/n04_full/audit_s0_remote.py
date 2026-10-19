"""Bounded, read-only-input S0 audit in a verified source snapshot; no training/services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = Path("/workspace/owac/research/n04-v1")
    frozen = Path("/workspace/owac/rawvla/source")
    encoder = str(base / "envs/encoder/bin/python")
    simulator = "/workspace/owac/rawvla/envs/starvla/bin/python"
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(source / "src"),
        CUDA_VISIBLE_DEVICES="",
    )
    report = dict(
        source=str(source), output=str(output), started=time.time(), commands=[], captures=[]
    )

    def save():
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command, environ=env):
        start = time.time()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=source,
                env=environ,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        report["commands"].append(
            dict(
                name=name, command=command, exit_code=result.returncode, seconds=time.time() - start
            )
        )
        save()
        print(name, result.returncode, flush=True)
        return result.returncode

    for module in (
        "owac.raw.n04_manifest",
        "owac.raw.n04_mechanism_data",
        "owac.evaluation.n04_contracts",
    ):
        run("help-" + module.rsplit(".", 1)[-1], [encoder, "-m", module, "--help"])
    config = "configs/n04/full/s0_data.json"
    run(
        "tests",
        [
            encoder,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/encoders/n04",
            "tests/encoders/n04_full",
        ],
    )
    for replica in ("inventory-a", "inventory-b"):
        run(
            replica,
            [
                encoder,
                "-m",
                "owac.raw.n04_manifest",
                "--config",
                config,
                "--reference-root",
                str(source),
                "--output",
                str(output / replica / "inventory.json"),
            ],
        )
    run(
        "contracts",
        [
            encoder,
            "-m",
            "owac.evaluation.n04_contracts",
            "--config",
            config,
            "--manifest",
            str(output / "inventory-a/splits.json"),
            "--output",
            str(output / "contracts.json"),
        ],
    )
    run(
        "mechanism",
        [
            encoder,
            "-m",
            "owac.raw.n04_mechanism_data",
            "--config",
            config,
            "--output",
            str(output / "mechanism"),
        ],
    )
    split = json.loads((output / "inventory-a/splits.json").read_text())["main"]
    tasks = {}
    for item in split["assignments"]:
        if item["platform"] == "libero" and item["split"] == "train":
            tasks.setdefault(item["task_key"], []).append(item)
    eligible = sorted(key for key, items in tasks.items() if len(items) >= 2)[:2]
    sim_env = dict(
        env,
        CUDA_VISIBLE_DEVICES="0",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        LIBERO_CONFIG_PATH=str(frozen / ".local/libero"),
        PYTHONPATH=":".join(
            [
                str(source / "src"),
                str(frozen / "experiments/simulation/libero"),
                str(frozen / "starVLA"),
                str(frozen / "third_party/LIBERO"),
            ]
        ),
    )
    index = 0
    for task in eligible:
        for item in tasks[task][:2]:
            npz = frozen / "benchmark_data/RawVLA-Bench/data/libero/raw" / item["relpaths"][0]
            mode = (
                "saved_before_step",
                "zeroed_each_step",
                "disabled_by_flag",
                "saved_before_step",
            )[index]
            target = output / f"capture-{index}"
            command = [
                simulator,
                "-c",
                "from owac.raw.n04_mechanism_data import d1_main; raise SystemExit(d1_main())",
                "--platform",
                "libero",
                "--npz",
                str(npz),
                "--output",
                str(target),
                "--trace-steps",
                "8",
                "--warmstart-mode",
                mode,
                "--capture-measurements",
            ]
            code = run(f"capture-{index}", command, sim_env)
            report["captures"].append(
                dict(
                    source_trajectory_id=item["source_trajectory_id"],
                    split="train",
                    task=task,
                    mode=mode,
                    output=str(target),
                    exit_code=code,
                )
            )
            save()
            index += 1
    # Independently rehash frozen protocol/weights. Never start or stop existing services.
    recorded = json.loads(
        (
            source / "research/experiments/results/2026-10-14_n04_s0_infra/frozen-assets.json"
        ).read_text()
    )
    files = recorded["files"]
    assets = files["report_dir"] + files["frozen_protocol_dir"] + list(files["weights"].values())
    report["frozen_assets"] = []
    for asset in assets:
        digest = hashlib.sha256()
        with Path(asset["path"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        report["frozen_assets"].append(
            dict(
                path=asset["path"],
                sha256=digest.hexdigest(),
                matches=digest.hexdigest() == asset["sha256"],
            )
        )
    report["finished"] = time.time()
    save()
    print("complete", flush=True)


if __name__ == "__main__":
    main()
