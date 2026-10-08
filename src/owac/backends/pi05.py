"""Local PyTorch deployment checks for the upstream pi0.5 RGB reference policy.

Heavy dependencies are imported only by explicit commands. The smoke input is synthetic
RGB; it provides deployment evidence, not RAW or robot task evidence.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_patch_files(source: Path, destination: Path, environment: Path) -> dict[str, str]:
    """Replace files atomically inside an environment, preserving shared cache inodes."""
    environment = environment.resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(environment):
        raise ValueError("Patch destination must be inside the selected virtual environment")
    files = sorted(source.rglob("*.py"))
    if not files:
        raise ValueError("No upstream Python patches found")
    targets = [(file, destination / file.relative_to(source)) for file in files]
    for _, target in targets:
        if not target.resolve().is_relative_to(environment):
            raise ValueError("Patch target escapes the selected virtual environment")
    hashes = {}
    for file, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copyfile(file, temporary_path)
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        hashes[str(file.relative_to(source))] = sha256_file(target)
    return hashes


def patch_transformers(upstream: Path) -> dict:
    if importlib.metadata.version("transformers") != "4.53.2":
        raise ValueError("The pinned upstream requires transformers==4.53.2")
    distribution = importlib.metadata.distribution("transformers")
    destination = Path(distribution.locate_file("transformers"))
    hashes = copy_patch_files(
        upstream / "src/openpi/models_pytorch/transformers_replace",
        destination,
        Path(sys.prefix),
    )
    return {"status": "patched", "destination": str(destination), "sha256": hashes}


def runtime_report() -> dict:
    import jax
    import torch
    from transformers.models.siglip import check

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local pi0.5 deployment check")
    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise RuntimeError("Upstream Transformers patches are missing")
    matrix = torch.eye(16, device="cuda")
    if not torch.equal(matrix @ matrix, matrix):
        raise RuntimeError("CUDA matrix multiplication check failed")
    torch.cuda.synchronize()
    return {
        "python": sys.version,
        "environment": sys.prefix,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "openpi",
                "torch",
                "torchvision",
                "transformers",
                "jax",
                "orbax-checkpoint",
            )
        },
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_matmul": "passed",
        "transformers_patches": "passed",
    }


def convert_checkpoint(checkpoint: Path, output: Path) -> None:
    """Use the pinned upstream converter and copy the checkpoint's own normalization assets."""
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets/droid").is_dir():
        raise ValueError("Expected a pi05_droid Orbax checkpoint with its DROID assets")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite conversion output: {output}")
    upstream = REPO_ROOT / "third_party/openpi"
    subprocess.run(
        [
            sys.executable,
            str(upstream / "examples/convert_jax_model_to_pytorch.py"),
            "--checkpoint-dir",
            str(checkpoint.resolve()),
            "--config-name",
            "pi05_droid",
            "--output-path",
            str(output.resolve()),
            "--precision",
            "bfloat16",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    # At this upstream revision the converter looks in checkpoint.parent / "assets".
    # Use checkpoint/assets explicitly so no neighboring checkpoint supplies statistics.
    shutil.copytree(checkpoint / "assets", output / "assets", dirs_exist_ok=True)
    record = {
        "upstream_commit": UPSTREAM_COMMIT,
        "source_checkpoint": str(checkpoint.resolve()),
        "config": "pi05_droid",
        "precision": "bfloat16",
        "model_sha256": sha256_file(output / "model.safetensors"),
        "norm_stats_sha256": sha256_file(output / "assets/droid/norm_stats.json"),
        "source_norm_stats_sha256": sha256_file(checkpoint / "assets/droid/norm_stats.json"),
    }
    (output / "owac_conversion.json").write_text(json.dumps(record, indent=2) + "\n")


def smoke(checkpoint: Path, output: Path, seed: int, repeats: int) -> dict:
    import numpy as np
    import torch
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError("A converted PyTorch checkpoint is required")
    if not (checkpoint / "assets/droid/norm_stats.json").is_file():
        raise ValueError("DROID normalization statistics are required")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "config": "pi05_droid",
        "observation_kind": "synthetic_rgb",
        "robot": None,
        "camera": None,
        "dataset": None,
        "seed": seed,
        "repeats": repeats,
        "warmup_runs": 1,
        "compile_mode": None,
        "num_flow_steps": 10,
        "checkpoint": str(checkpoint.resolve()),
        "upstream_commit": UPSTREAM_COMMIT,
        "runtime_lock_sha256": sha256_file(REPO_ROOT / "configs/backends/pi05-runtime/uv.lock"),
        "owac_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    (output / "source.diff").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO_ROOT, text=True)
    )
    # Keep the actual executing source even when the module is not yet tracked by Git.
    shutil.copyfile(Path(__file__), output / "pi05_source.py")
    try:
        record["runtime"] = runtime_report()
        record["checkpoint_sha256"] = sha256_file(checkpoint / "model.safetensors")
        record["norm_stats_sha256"] = sha256_file(checkpoint / "assets/droid/norm_stats.json")
        config = training_config.get_config("pi05_droid")
        config = dataclasses.replace(
            config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
        )
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        observation = {
            "observation/exterior_image_1_left": rng.integers(
                0, 256, size=(224, 224, 3), dtype=np.uint8
            ),
            "observation/wrist_image_left": rng.integers(
                0, 256, size=(224, 224, 3), dtype=np.uint8
            ),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "pick up the cup",
        }
        noise = rng.standard_normal((config.model.action_horizon, config.model.action_dim)).astype(
            np.float32
        )
        policy = policy_config.create_trained_policy(
            config, checkpoint, pytorch_device="cuda", sample_kwargs={"num_steps": 10}
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        policy.infer(observation, noise=noise)
        torch.cuda.synchronize()
        record["warmup_ms"] = (time.perf_counter() - start) * 1000
        durations = []
        reference = None
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            actions = policy.infer(observation, noise=noise)["actions"]
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
            if actions.shape != (config.model.action_horizon, 8) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid DROID action output: shape={actions.shape}")
            if reference is not None and not np.array_equal(reference, actions):
                raise RuntimeError(
                    "Repeated inference with fixed observations/noise changed output"
                )
            reference = actions.copy()
        record.update(
            status="passed",
            action_shape=list(reference.shape),
            finite_actions=True,
            fixed_input_repeatability="passed" if repeats > 1 else "not_tested",
            synchronized_policy_ms=durations,
            sample_p50_ms=float(np.percentile(durations, 50)),
            sample_p95_ms=float(np.percentile(durations, 95)),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            measurement_scope=(
                "policy.infer including transforms and CPU transfer; excludes load/I/O"
            ),
        )
        (output / "actions.json").write_text(json.dumps(reference.tolist()) + "\n")
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def serve(checkpoint: Path, host: str, port: int) -> None:
    """Serve the same RGB reference policy through the upstream WebSocket protocol."""
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as training_config

    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError("A converted PyTorch checkpoint is required")
    runtime_report()
    config = training_config.get_config("pi05_droid")
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(
        config, checkpoint, pytorch_device="cuda", sample_kwargs={"num_steps": 10}
    )
    metadata = {
        "config": "pi05_droid",
        "input_kind": "rgb_reference",
        "robot": None,
        "camera": None,
        "dataset": None,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    print(f"pi0.5 RGB reference server: ws://{host}:{port}", flush=True)
    WebsocketPolicyServer(policy, host=host, port=port, metadata=metadata).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    patch = subparsers.add_parser("patch", help="Apply upstream patches inside this environment")
    patch.add_argument("--upstream", type=Path, default=REPO_ROOT / "third_party/openpi")
    subparsers.add_parser("doctor", help="Check installed patches and actual CUDA execution")
    convert = subparsers.add_parser("convert", help="Convert a cached pi05_droid checkpoint")
    convert.add_argument("--checkpoint", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    inference = subparsers.add_parser("smoke", help="Run synthetic RGB inference without a robot")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--seed", type=int, default=0)
    inference.add_argument("--repeats", type=int, default=3)
    server = subparsers.add_parser("serve", help="Start the local RGB reference policy server")
    server.add_argument("--checkpoint", type=Path, required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.command == "patch":
        result = patch_transformers(args.upstream)
    elif args.command == "doctor":
        result = runtime_report()
    elif args.command == "convert":
        convert_checkpoint(args.checkpoint, args.output)
        result = {"status": "converted", "output": str(args.output)}
    elif args.command == "serve":
        serve(args.checkpoint, args.host, args.port)
        return
    else:
        result = smoke(args.checkpoint, args.output, args.seed, args.repeats)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
