"""Finite paired LIBERO probe with checkpoint-boundary GPU handoff and training resume."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from owac.runtime.n01_v12_libero_dev import _report, merge_ledgers


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--deploy-root", type=Path, required=True)
    parser.add_argument("--role", choices=("raw", "rgb"), required=True)
    parser.add_argument("--checkpoint-wait", type=int, default=1500)
    args = parser.parse_args(argv)
    root, train, deploy = args.root, args.training_root, args.deploy_root
    role = args.role
    arm, gpu = f"{role}_n01", 0 if role == "raw" else 1
    port = 18032 + gpu
    program = f"owac-n01-v12-complete-{role}"
    worker = train / "monitor/workers" / f"libero-{arm}-seed101.json"
    folder, source = root / arm, root / "source"
    folder.mkdir(exist_ok=True)
    capture = next(
        item
        for item in json.loads((root / "captured-checkpoints.json").read_text())
        if item["arm"] == arm
    )
    state = {
        "role": role,
        "arm": arm,
        "checkpoint_step": capture["step"],
        "checkpoint": capture["checkpoint"],
        "selected": 12,
        "completed": 0,
        "successes": 0,
        "paused_training": False,
        "state": "waiting_checkpoint",
        "holdout_status": "development; content-level split identity not proven",
    }

    def publish(**changes):
        state.update(changes, updated_at=time.time())
        write_json(folder / "progress.json", state)
        write_json(train / "monitor" / f"probe20k-{role}.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    server = client = None
    paused = False
    error = None
    result = None
    publish()
    try:
        deadline = time.monotonic() + args.checkpoint_wait
        while True:
            current = json.loads(worker.read_text())
            pid = int(current["pid"])
            os.kill(pid, 0)
            step, saved = int(current["step"]), int(current.get("checkpoint_step") or 0)
            if saved > 0 and 0 <= step - saved <= 1:
                publish(state="loading", paused_training=True, resume_checkpoint_step=saved)
                paused = True
                subprocess.run(["supervisorctl", "stop", program], check=True)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("next durable training checkpoint was not reached")
            time.sleep(2)
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"probe port {port} is already occupied")
        config = source / "configs/encoders/n01_full/v1_2/complete" / f"libero-{arm}-seed101.json"
        env = dict(os.environ, N01_SOURCE=str(source), PYTHONPATH=str(source / "src"))
        wrapper = source / "configs/backends/n01-v12-runtime/run-on-gpu.sh"
        command = [
            "bash",
            str(wrapper),
            role,
            sys.executable,
            "-m",
            "owac.runtime.n01_v12_service",
            "serve",
            "--suite",
            "libero",
            "--weights",
            str(
                deploy
                / "source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors"
            ),
            "--tokenizer",
            str(deploy / "source/openpi_converted_protocol/paligemma_tokenizer.model"),
            "--config",
            str(config),
            "--checkpoint",
            capture["checkpoint"],
            "--device",
            "cuda:0",
            "--port",
            str(port),
            "--openpi-adapter",
            "--precision",
            "bf16",
            "--flow-batch-size",
            "8",
            "--no-stale",
            "--no-degrade",
        ]
        with (folder / "service.log").open("ab") as handle:
            server = subprocess.Popen(
                command,
                cwd="/tmp",
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 300
        while True:
            if server.poll() is not None:
                raise RuntimeError("checkpoint service exited; see service.log")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("checkpoint service did not listen within 300 seconds")
            time.sleep(2)
        command = [
            sys.executable,
            "-m",
            "owac.runtime.n01_v12_libero_dev",
            "--devloop-root",
            str(folder),
            "--repo-root",
            str(source),
            "--deploy-root",
            str(deploy),
            "--config",
            str(config),
            "--workers",
            "1",
            "--gpu-indices",
            str(gpu),
            "--ports",
            str(port),
            "--per-suite",
            "3",
            "--lighting-domains",
            "ExtremeLow,Low,Normal",
            "--resize-size",
            "256",
            "--label",
            "paired",
            "--record-timeline",
            str(folder / "timeline.jsonl"),
            "--timeout",
            "1800",
        ]
        with (folder / "client.log").open("ab") as handle:
            client = subprocess.Popen(
                command,
                cwd="/tmp",
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        publish(state="evaluating", service_pid=server.pid, launcher_pid=client.pid)
        deadline = time.monotonic() + 7200
        while client.poll() is None:
            plan = folder / "devloop-paired/worker0-identities.json"
            if plan.exists():
                selected = json.loads(plan.read_text())["identities"]
                merged = merge_ledgers([folder / "devloop-paired/worker0-episodes.jsonl"])
                partial = _report(merged, selected)
                if partial["completed_rollouts"] != state["completed"]:
                    publish(completed=partial["completed_rollouts"], successes=partial["successes"])
            if time.monotonic() > deadline:
                raise TimeoutError("finite paired probe exceeded two-hour wall budget")
            if server.poll() is not None:
                raise RuntimeError("checkpoint service crashed during the closed loop")
            time.sleep(3)
        if client.returncode:
            raise RuntimeError(f"development launcher exited {client.returncode}")
        result = json.loads((folder / "devloop-paired/report.json").read_text())
        write_json(folder / "result.json", {**result, "checkpoint_step": capture["step"]})
        if result["completed_rollouts"] != 12:
            raise RuntimeError(f"only {result['completed_rollouts']}/12 rollouts completed")
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        stop_group(client)
        stop_group(server)
        if paused:
            publish(state="resuming", error=error)
            try:
                subprocess.run(["supervisorctl", "start", program], check=True, timeout=90)
                publish(paused_training=False, training_resumed=True)
            except BaseException as caught:
                error = f"{error or ''}; training resume: {type(caught).__name__}: {caught}"
        publish(
            state="failed" if error else "completed",
            error=error,
            completed=state["completed"] if result is None else result["completed_rollouts"],
            successes=state["successes"] if result is None else result["successes"],
        )
    if error:
        raise RuntimeError(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
