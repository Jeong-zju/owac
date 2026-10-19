"""Finite original-policy reference probe; never starts or resumes training."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from owac.runtime.n01_v12_libero_dev import _report, merge_ledgers, patch_client_source
from owac.runtime.n01_v12_paired_probe import stop_group, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--deploy", type=Path, required=True)
    args = parser.parse_args()
    root, previous, deploy = args.root, args.previous, args.deploy
    folder = root / "author-reference"
    folder.mkdir(parents=True, exist_ok=True)
    entries = json.loads((previous / "paired-plan.json").read_text())["identities"]
    selected = [row for row in entries if row["lighting_domain"] == "Normal"]
    assert len(selected) == 4
    write_json(folder / "identities.json", {"identities": selected})
    (folder / "libero-manifest.json").write_bytes(
        (previous / "raw_n01/libero-manifest.json").read_bytes()
    )
    client = folder / "client.py"
    client.write_text(
        patch_client_source((previous / "raw_n01/runtime/libero_client.py").read_text())
    )
    source = deploy / "source"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        TOKENIZERS_PARALLELISM="false",
        LIBERO_CONFIG_PATH=str(source / ".local/libero"),
        OPENPI_CONVERTED_ROOT=str(source / "openpi_converted_protocol"),
        OWAC_EPISODE_ALLOWLIST=str(folder / "identities.json"),
        OWAC_TIMELINE_PATH=str(folder / "timeline.jsonl"),
    )
    env["PYTHONPATH"] = ":".join(
        [
            str(root / "source/src"),
            str(source / "starVLA"),
            str(source / "experiments/simulation/libero"),
            str(source / "starVLA/examples/simBenchmarks/LIBERO/eval_files/openpi"),
        ]
    )
    server = child = None
    try:
        command = [
            sys.executable,
            "-m",
            "owac.backends.rawvla",
            "serve-libero",
            "--checkpoint",
            str(
                source / "openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors"
            ),
            "--tokenizer",
            str(source / "openpi_converted_protocol/paligemma_tokenizer.model"),
            "--port",
            "18042",
        ]
        with (folder / "server.log").open("ab") as log:
            server = subprocess.Popen(
                command,
                cwd="/tmp",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 360
        while True:
            if server.poll() is not None:
                raise RuntimeError("original policy server exited; see server.log")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", 18042)) == 0:
                    break
            if time.monotonic() > deadline:
                raise TimeoutError("original policy service readiness")
            time.sleep(2)
        ledger = folder / "episodes.jsonl"
        exits = []
        for item in selected:
            suite = item["suite"]
            command = [
                sys.executable,
                str(client),
                "--model",
                "PI05",
                "--model-source",
                "openpi",
                "--host",
                "127.0.0.1",
                "--port",
                "18042",
                "--task-suite",
                suite,
                "--task-start",
                str(item["task_id"]),
                "--max-tasks",
                "1",
                "--image-mode",
                "rawvla_bench",
                "--rawvla-bench-representation",
                "default_isp",
                "--rawvla-bench-manifest",
                str(folder / "libero-manifest.json"),
                "--resume-path",
                str(ledger),
                "--result-json",
                str(folder / f"{suite}.json"),
                "--assets-checkpoint",
                str(deploy / "checkpoints/pi05_libero_jax"),
                "--resize-size",
                "256",
                "--no-log-action-stats",
            ]
            with (folder / f"{suite}.log").open("ab") as log:
                child = subprocess.Popen(
                    command,
                    cwd=source / "starVLA",
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = child.wait(timeout=900)
            exits.append(code)
            if code:
                raise RuntimeError(f"{suite} client exited {code}")
            print(
                json.dumps({"suite": suite, "completed": len(merge_ledgers([ledger]))}), flush=True
            )
        report = _report(merge_ledgers([ledger]), selected)
        report.update(
            client_exit_codes=exits,
            input=(
                "author frozen default ISP from same-condition RAW; "
                "original image tower and pretrained policy"
            ),
            training_started=False,
        )
        write_json(folder / "result.json", report)
        print(json.dumps(report), flush=True)
    finally:
        stop_group(child)
        stop_group(server)


if __name__ == "__main__":
    main()
