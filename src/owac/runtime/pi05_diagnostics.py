"""Manage the development-only policy service and sequential simulator diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from owac.evaluation.baseline_protocol import file_sha256, write_json


def run(args):
    repo = Path(__file__).resolve().parents[3]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    policy_python = repo / "configs/backends/pi05-runtime/.venv/bin/python"
    sim_python = repo / "configs/simulation/lw-libero-runtime/.venv/bin/python"
    baseline = args.baseline
    source = json.loads((baseline / "results/provenance.json").read_text())["source_sha256"]
    assert all(file_sha256(repo / name) == sha for name, sha in source.items())
    protocol = repo / "configs/baselines/pi05-lw-mug-v1.json"
    service_command = [
        str(policy_python),
        "-u",
        "-m",
        "owac.evaluation.pi05_diagnostics",
        "serve",
        "--protocol",
        str(protocol),
        "--weights",
        str(baseline / "trained"),
        "--port",
        str(args.port),
    ]
    simulation_command = [
        str(sim_python),
        "-u",
        "-m",
        "owac.simulation.pi05_diagnostics",
        "--protocol",
        str(protocol),
        "--data",
        str(baseline / "data"),
        "--output",
        str(output / "rollouts"),
        "--port",
        str(args.port),
    ]
    report = {
        "status": "starting",
        "started_unix": time.time(),
        "baseline": str(baseline),
        "commands": [service_command, simulation_command],
        "source_sha256": {
            str(path.relative_to(repo)): file_sha256(path)
            for path in (
                repo / "src/owac/evaluation/pi05_diagnostics.py",
                repo / "src/owac/simulation/pi05_diagnostics.py",
                Path(__file__),
            )
        },
        "frozen_baseline_source_matches": True,
        "final_test_seeds_used": False,
        "checkpoints_downloaded": False,
    }
    write_json(output / "run.json", report)
    env = os.environ | {"JAX_PLATFORMS": "cpu", "OMNI_KIT_ALLOW_ROOT": "1"}
    server = None
    try:
        with (output / "policy.log").open("w") as log:
            server = subprocess.Popen(
                service_command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Diagnostic policy service failed at startup")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/healthz", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Diagnostic policy startup timed out")
            report.update(status="running", stage="development rollouts")
            write_json(output / "run.json", report)
            with (output / "simulator.log").open("w") as sim_log:
                subprocess.run(
                    simulation_command,
                    cwd=repo,
                    env=env,
                    stdout=sim_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        assert all(file_sha256(repo / name) == sha for name, sha in source.items())
        report.update(status="complete", finished_unix=time.time())
    except Exception as error:
        report.update(status="failed", error=repr(error), finished_unix=time.time())
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        write_json(output / "run.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
