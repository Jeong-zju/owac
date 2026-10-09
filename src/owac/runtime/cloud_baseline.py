"""Run the complete collect/train/100-episode pipeline under a process supervisor."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from owac.evaluation.baseline_protocol import file_sha256, write_json


def run(args):
    root = Path(__file__).resolve().parents[3]
    sim_python = root / "configs/simulation/lw-libero-runtime/.venv/bin/python"
    policy_python = root / "configs/backends/pi05-runtime/.venv/bin/python"
    args.run_root.mkdir(parents=True, exist_ok=False)
    results = args.run_root / "results"
    results.mkdir()
    (results / "protocol.json").write_bytes(args.protocol.read_bytes())
    status = {"status": "starting", "started_unix": time.time()}
    provenance = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for pattern in (
                "src/owac/**/*.py",
                "tests/*.py",
                "configs/baselines/*",
                "configs/backends/pi05-runtime/uv.lock",
                "configs/simulation/lw-libero-runtime/requirements.freeze.txt",
            )
            for path in sorted(root.glob(pattern))
        },
        "protocol_sha256": file_sha256(args.protocol),
        "host": subprocess.check_output(["uname", "-a"]).decode().strip(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]
        )
        .decode()
        .strip(),
        "runtime_root": str(root),
        "training_runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "openpi", "numpy", "safetensors")
        },
        "commands": [],
    }
    (results / "working-tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    )
    (results / "git-status.txt").write_bytes(
        subprocess.check_output(["git", "status", "--short"], cwd=root)
    )
    env = os.environ | {"OMNI_KIT_ALLOW_ROOT": "1", "JAX_PLATFORMS": "cpu"}

    def stage(name, command):
        status.update(status="running", stage=name, stage_started_unix=time.time())
        provenance["commands"].append(command)
        write_json(results / "status.json", status)
        write_json(results / "provenance.json", provenance)
        print("BASELINE_STAGE", name, flush=True)
        with (args.run_root / (name + ".log")).open("w") as log:
            subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    server = None
    try:
        stage(
            "collect",
            [
                str(sim_python),
                "-u",
                "-m",
                "owac.simulation.pi05_benchmark",
                "collect",
                "--protocol",
                str(args.protocol),
                "--output",
                str(args.run_root / "data"),
            ],
        )
        for name in ("dataset.json", "camera.json"):
            source = args.run_root / "data" / name
            (results / name).write_bytes(source.read_bytes())
        dataset = json.loads((results / "dataset.json").read_text())
        if dataset["status"] != "complete":
            raise RuntimeError("Collection did not finish all prescribed episodes")
        stage(
            "train",
            [
                str(policy_python),
                "-u",
                "-m",
                "owac.training.pi05_sim",
                "train",
                "--protocol",
                str(args.protocol),
                "--weights",
                str(args.weights),
                "--data",
                str(args.run_root / "data"),
                "--output",
                str(args.run_root / "trained"),
            ],
        )
        for name in ("training.json", "curve.jsonl", "normalization.json"):
            source = args.run_root / "trained" / name
            (results / name).write_bytes(source.read_bytes())
        training = json.loads((results / "training.json").read_text())
        if training["status"] != "complete" or not training["convergence_observed"]:
            raise RuntimeError("Training needs diagnosis: prescribed plateau not observed")
        policy_command = [
            str(policy_python),
            "-u",
            "-m",
            "owac.training.pi05_sim",
            "serve",
            "--protocol",
            str(args.protocol),
            "--weights",
            str(args.run_root / "trained"),
            "--port",
            str(args.port),
        ]
        provenance["commands"].append(policy_command)
        with (args.run_root / "policy.log").open("w") as log:
            server = subprocess.Popen(
                policy_command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Policy server exited during startup")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/healthz", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Policy server startup timed out")
            stage(
                "evaluate",
                [
                    str(sim_python),
                    "-u",
                    "-m",
                    "owac.simulation.pi05_benchmark",
                    "evaluate",
                    "--protocol",
                    str(args.protocol),
                    "--output",
                    str(results / "evaluation"),
                    "--port",
                    str(args.port),
                ],
            )
        summary = json.loads((results / "evaluation/summary.json").read_text())
        if summary["episodes"] != 100 or summary["status"] != "complete":
            raise RuntimeError("Final evaluation must contain all 100 prescribed episodes")
        for source in (
            args.run_root / "data/dataset.json",
            args.run_root / "trained/training.json",
            args.run_root / "trained/curve.jsonl",
            args.run_root / "trained/normalization.json",
        ):
            (results / source.name).write_bytes(source.read_bytes())
        status.update(status="complete", finished_unix=time.time(), summary=summary)
    except Exception as error:
        status.update(status="failed", error=repr(error), finished_unix=time.time())
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        write_json(results / "status.json", status)
        write_json(results / "provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.weights = args.weights.resolve()
    args.run_root = args.run_root.resolve()
    run(args)


if __name__ == "__main__":
    main()
