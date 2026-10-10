"""Deploy and inspect the pinned official Arena DROID reference in an isolated root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

SOURCES = {
    "Arena": ("isaac-sim/IsaacLab-Arena", "2cc9e9766f1f62a500b85c92207192d8c60fcce5"),
    "IsaacLab": ("isaac-sim/IsaacLab", "91a0906f93802eb859192dcb0bca879f891e92d6"),
    "Isaac-GR00T": ("NVIDIA/Isaac-GR00T", "e29d8fc50b0e4745120ae3fb72447986fe638aa6"),
    "openpi": ("Physical-Intelligence/openpi", "c23745b5ad24e98f66967ea795a07b2588ed6c79"),
}

POLICY_CONFIG = "pi05_droid_jointpos_polaris"


def convert(args):
    """Use the official converter; preserve this checkpoint's normalization assets."""
    root = args.root
    openpi = root / "openpi"
    python = openpi / ".venv/bin/python"
    site = openpi / ".venv/lib/python3.11/site-packages/transformers"
    shutil.copytree(
        openpi / "src/openpi/models_pytorch/transformers_replace", site, dirs_exist_ok=True
    )
    destination = root / "checkpoints/pi05_droid_jointpos_pytorch"
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = root / "checkpoints/pi05_droid_jointpos_orbax"
    subprocess.run(
        [
            str(python),
            "examples/convert_jax_model_to_pytorch.py",
            "--checkpoint-dir",
            str(checkpoint),
            "--config-name",
            POLICY_CONFIG,
            "--output-path",
            str(destination),
            "--precision",
            "bfloat16",
        ],
        cwd=openpi,
        env=os.environ | {"JAX_PLATFORMS": "cpu"},
        check=True,
    )
    # The pinned upstream converter uses checkpoint.parent/assets; correct that source explicitly.
    shutil.copytree(checkpoint / "assets", destination / "assets", dirs_exist_ok=True)
    write_json(
        root / "deployment/conversion.json",
        {
            "config": POLICY_CONFIG,
            "precision": "bfloat16",
            "fine_tuning": False,
            "model_sha256": sha256(destination / "model.safetensors"),
            "normalization_sha256": sha256(destination / "assets/droid/norm_stats.json"),
            "source_normalization_sha256": sha256(checkpoint / "assets/droid/norm_stats.json"),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def serve(args):
    """Serve the matched official joint-position policy without changing weights."""
    import dataclasses

    import numpy as np
    import torch
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    official = config.get_config(POLICY_CONFIG)
    official = dataclasses.replace(
        official,
        model=dataclasses.replace(official.model, pytorch_compile_mode=None),
        data=dataclasses.replace(
            official.data,
            assets=dataclasses.replace(
                official.data.assets, assets_dir=str(args.weights / "assets")
            ),
        ),
    )
    policy = policy_config.create_trained_policy(official, args.weights)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy, host="127.0.0.1", port=args.port, metadata=policy.metadata
    )

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    finally:
        write_json(
            args.metrics,
            {
                "config": POLICY_CONFIG,
                "seed": args.seed,
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "compile": False,
            },
        )


def run(args):
    """Serialize fresh official environments and policy servers for the declared seeds."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.protocol, args.output / "protocol.json")
    environment = os.environ | {
        "JAX_PLATFORMS": "cpu",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMNI_KIT_ALLOW_ROOT": "1",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE": "1",
        "PYNPUT_BACKEND": "dummy",
        "HEADLESS": "1",
        "OWAC_ARENA_RAW": str(int(args.phase == "raw")),
    }
    completed = []
    if args.resume_from is not None:
        from owac.runtime.arena_droid_archive import completed_episode

        previous = json.loads((args.resume_from / "status.json").read_text())
        if previous.get("phase") != args.phase:
            raise ValueError("Cannot resume a different phase")
        if json.loads((args.resume_from / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol must be unchanged")
        for seed in previous["completed_seeds"]:
            if seed not in protocol["environment_seeds"]:
                raise ValueError("Resume seed is outside the declared protocol")
            source = args.resume_from / f"seed-{seed}"
            episode, _ = completed_episode(source, seed)
            if episode is None:
                raise ValueError("Cannot carry forward an incomplete or invalid episode")
            # Hard links keep completed RAW evidence without duplicating large arrays.
            # These carried-forward seed directories are never written by the new run.
            shutil.copytree(source, args.output / source.name, copy_function=os.link)
            completed.append(seed)
    started = time.monotonic()
    status = {
        "status": "running",
        "phase": args.phase,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "carried_forward_seeds": completed.copy(),
    }

    def update(**values):
        status.update(
            **values, updated_at_utc=datetime.now(UTC).isoformat(), completed_seeds=completed
        )
        write_json(args.output / "status.json", status)

    server = None
    try:
        for seed in protocol["environment_seeds"]:
            if seed in completed:
                continue
            seed_dir = args.output / f"seed-{seed}"
            seed_dir.mkdir(exist_ok=False)
            update(stage="load_official_policy", seed=seed)
            with (seed_dir / "server.log").open("w") as server_log:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "owac.runtime.arena_droid",
                        "serve",
                        "--weights",
                        str(args.root / "checkpoints/pi05_droid_jointpos_pytorch"),
                        "--port",
                        str(args.port),
                        "--seed",
                        str(protocol["inference_seed"]),
                        "--metrics",
                        str(seed_dir / "policy_metrics.json"),
                    ],
                    env=environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 300
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Official policy startup failed for seed {seed}")
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "Policy server startup exceeded 300 seconds"
                            ) from None
                        time.sleep(1)
                experiment = {
                    "runs": {
                        "droid_pnp_openpi": {
                            "environment": {
                                "type": protocol["environment"],
                                "enable_cameras": True,
                                "embodiment": protocol["embodiment"],
                                "pick_up_object": protocol["object"],
                                "destination_location": protocol["destination"],
                                "hdr": protocol["hdr"],
                            },
                            "environment_builder": {
                                "num_envs": 1,
                                "seed": seed,
                                "language_instruction": protocol["instruction"],
                            },
                            "policy": {
                                "type": (
                                    "isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy"
                                ),
                                "policy_variant": "pi05",
                                "policy_device": "cuda:0",
                                "remote_host": "127.0.0.1",
                                "remote_port": args.port,
                                "openpi_embodiment_adapter": "droid",
                            },
                            "rollout_limit": {"num_episodes": 1},
                        }
                    }
                }
                config = seed_dir / "experiment.yaml"
                write_json(config, experiment)
                environment["OWAC_ARENA_EVIDENCE"] = str(seed_dir / "evidence")
                update(stage="official_closed_loop", seed=seed)
                command = [
                    str(args.root / "Arena/.venv/bin/python"),
                    "-u",
                    "-m",
                    "owac.simulation.arena_droid_rollout",
                    "--experiment_config",
                    str(config),
                    "--experiment_output_directory",
                    str(seed_dir / "official"),
                    "--viz",
                    "none",
                    "--device",
                    "cuda:0",
                    "--experience",
                    str(
                        args.root / "Arena/.venv/lib/python3.12/site-packages/isaacsim/apps/"
                        "isaacsim.exp.base.python.kit"
                    ),
                ]
                write_json(
                    seed_dir / "command.json", {"argv": command, "cwd": str(args.root / "Arena")}
                )
                with (seed_dir / "evaluation.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=args.root / "Arena",
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            server.terminate()
            server.wait(timeout=30)
            server = None
            completed.append(seed)
            update(stage="episode_complete", seed=seed)
        update(status="complete", stage="acceptance_complete")
    except BaseException as error:
        update(status="infrastructure_failed", error=repr(error))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=30)
        update(wall_seconds=time.monotonic() - started)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def deploy(args):
    args.root.mkdir(parents=True, exist_ok=True)
    results = args.root / "deployment"
    results.mkdir(exist_ok=False)
    status = {"started_at_utc": datetime.now(UTC).isoformat(), "status": "running"}
    environment = os.environ | {
        "GIT_LFS_SKIP_SMUDGE": "1",
        "OMNI_KIT_ALLOW_ROOT": "1",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
    }
    commands = []
    downloads = None
    started = time.monotonic()

    def stage(name, command, cwd):
        status.update(stage=name, updated_at_utc=datetime.now(UTC).isoformat())
        write_json(results / "status.json", status)
        commands.append({"stage": name, "argv": command, "cwd": str(cwd)})
        write_json(results / "commands.json", commands)
        print("ARENA_STAGE", name, flush=True)
        with (results / f"{name}.log").open("w") as log:
            subprocess.run(
                command, cwd=cwd, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT
            )

    try:
        checkpoints = args.root / "checkpoints/pi05_droid_jointpos_orbax"
        checkpoints.mkdir(parents=True, exist_ok=False)
        with (results / "weights.log").open("w") as weight_log:
            weight_command = [
                "rclone",
                "copy",
                "--gcs-anonymous",
                "--transfers",
                "8",
                "--checkers",
                "8",
                "--stats",
                "30s",
                ":gcs:openpi-assets-simeval/pi05_droid_jointpos",
                str(checkpoints),
            ]
            commands.append({"stage": "download_official_weights", "argv": weight_command})
            downloads = subprocess.Popen(
                weight_command, env=environment, stdout=weight_log, stderr=subprocess.STDOUT
            )
            sources = {}
            for name, (repository, commit) in SOURCES.items():
                destination = (
                    args.root / "Arena/submodules" / name
                    if name in {"IsaacLab", "Isaac-GR00T"}
                    else args.root / name
                )
                archive = args.root / f"{name}-{commit}.tar.gz"
                stage(
                    f"source_{name}",
                    [
                        "curl",
                        "-fL",
                        "--retry",
                        "4",
                        "--connect-timeout",
                        "20",
                        "--max-time",
                        "600",
                        f"https://codeload.github.com/{repository}/tar.gz/{commit}",
                        "-o",
                        str(archive),
                    ],
                    args.root,
                )
                destination.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive) as bundle:
                    members = bundle.getmembers()
                    prefix = members[0].name.split("/")[0] + "/"
                    for member in members:
                        if member.name.startswith(prefix) and member.name != prefix:
                            member.name = member.name[len(prefix) :]
                            bundle.extract(member, destination, filter="data")
                sources[name] = {
                    "repository": repository,
                    "commit": commit,
                    "archive_sha256": sha256(archive),
                    "path": str(destination),
                }
                write_json(results / "sources.json", sources)
            stage(
                "install_arena",
                ["uv", "sync", "--frozen", "--python", "3.12"],
                args.root / "Arena",
            )
            stage(
                "install_openpi",
                ["uv", "sync", "--frozen", "--python", "3.11"],
                args.root / "openpi",
            )
            if downloads.wait() != 0:
                raise RuntimeError("Official checkpoint download failed; see weights.log")
        write_json(
            results / "checkpoint_manifest.json",
            {
                "source": "gs://openpi-assets-simeval/pi05_droid_jointpos",
                "sha256": {
                    str(p.relative_to(checkpoints)): sha256(p)
                    for p in sorted(checkpoints.rglob("*"))
                    if p.is_file()
                },
            },
        )
        status.update(status="complete", stage="deployed")
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        if downloads is not None and downloads.poll() is None:
            downloads.terminate()
            downloads.wait(timeout=30)
        status.update(
            updated_at_utc=datetime.now(UTC).isoformat(), wall_seconds=time.monotonic() - started
        )
        write_json(results / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    deployment = commands.add_parser("deploy")
    deployment.add_argument("--root", type=Path, required=True)
    conversion = commands.add_parser("convert")
    conversion.add_argument("--root", type=Path, required=True)
    policy = commands.add_parser("serve")
    policy.add_argument("--weights", type=Path, required=True)
    policy.add_argument("--port", type=int, default=18020)
    policy.add_argument("--seed", type=int, default=7)
    policy.add_argument("--metrics", type=Path, required=True)
    evaluation = commands.add_parser("run")
    evaluation.add_argument("--root", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--protocol", type=Path, required=True)
    evaluation.add_argument("--phase", choices=["rgb", "raw"], required=True)
    evaluation.add_argument("--resume-from", type=Path)
    evaluation.add_argument("--port", type=int, default=18020)
    args = parser.parse_args()
    {"deploy": deploy, "convert": convert, "serve": serve, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
