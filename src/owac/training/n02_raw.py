"""Real NPZ -> N02 -> frozen StarVLA PI05 smoke training and inference.

Run in the pinned external Python 3.10 runtime via PYTHONPATH. The root package
is not installed into that runtime. Outputs are offline wiring evidence, never
closed-loop task scores. Large checkpoints remain in the specified output dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.backends.rawvla_features_n02 import (
    EncoderReadoutTokenSource,
    build_feature_policy,
    check_runtime_assets,
    load_checkpoint_state,
)
from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02.config import N02Config, load_config
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.readout import N02Readout
from owac.raw.rawvla_sequence_n02 import (
    NormalisationStats,
    RawVLAWindowDataset,
    WindowOptions,
    _acquisition_source,
    _split_side,
    discover_npz_files,
)


def sample_frames(sample: Any, step: int, device: str) -> dict[str, MeasurementFrame]:
    """The recurrent axis is time, never a batch of independent observations."""
    frames = {}
    for camera in sample.camera_ids:
        payload = sample.measurement_payload(camera)

        def tensor(value):
            return None if value is None else torch.as_tensor(value[step : step + 1], device=device)

        metadata = dict(payload["metadata"])
        if payload["variance"] is not None:
            metadata["variance"] = tensor(payload["variance"])
            metadata["variance_is_estimate"] = payload["variance_is_estimate"]
        frames[camera] = MeasurementFrame(
            values=tensor(payload["values"]),
            valid_mask=tensor(payload["valid_mask"]),
            bad_mask=tensor(payload["bad_mask"]),
            saturated_mask=tensor(payload["saturated_mask"]),
            frame_ids=(int(sample.observation_indices[step]),),
            episode_ids=(sample.episode.episode_group_id,),
            camera_id=camera,
            new_measurement=bool(sample.new_measurement[step]),
            metadata=metadata,
            metadata_available=payload["metadata_available"],
            time_unit="frame",
        )
    return frames


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def language_state(bridge: Any, sample: Any, device: str):
    if not sample.proprio_available or sample.proprio is None:
        raise ValueError("real smoke requires the declared normalized current proprio")
    state = torch.zeros(1, bridge.protocol.max_state_dim, device=device)
    state[0, : sample.proprio.size] = torch.as_tensor(sample.proprio, device=device)
    prompt = sample.provenance.get("task_description", "")
    if not prompt:
        raise ValueError("NPZ has no task instruction; refusing synthetic language for real smoke")
    token_state = state[0].cpu().numpy() if bridge.protocol.name == "robotwin" else None
    tokens, masks = bridge.base.tokenizer.tokenize(prompt, token_state)
    return (
        torch.as_tensor(tokens, device=device).unsqueeze(0),
        torch.as_tensor(masks, device=device).unsqueeze(0),
        state,
    )


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assets = check_runtime_assets(
        checkpoint=args.checkpoint, tokenizer=args.tokenizer, data_root=args.data_root
    )
    if not all(
        assets.get(key)
        for key in (
            "real_backend_available",
            "checkpoint_exists",
            "tokenizer_exists",
            "data_root_exists",
        )
    ):
        report = {"status": "unavailable", "assets": assets}
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    config = load_config(args.config)
    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    if resume:
        config = N02Config.from_dict(resume["config"])
        for name in (
            "suite",
            "data_root",
            "stats",
            "stats_key",
            "checkpoint",
            "tokenizer",
            "window",
            "seed",
            "lr",
        ):
            if str(getattr(args, name)) != str(resume["run"][name]):
                raise ValueError(f"resume must preserve {name}: {resume['run'][name]}")
        if args.steps <= int(resume["step"]):
            raise ValueError("--steps must exceed the resumed step count")
    stats = NormalisationStats.from_json(
        args.stats,
        mode="quantile" if args.suite == "libero" else "zscore",
        stats_key=args.stats_key,
    )
    paths, _ = discover_npz_files(args.data_root, args.suite)
    selected = {}
    for path in paths:
        source_id = _acquisition_source(str(Path(path).relative_to(args.data_root)))
        side = _split_side(source_id, args.seed, 0.25)
        selected.setdefault(side, path)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError(
            "smoke needs at least one train and one validation source; choose a broader data root"
        )
    dataset = RawVLAWindowDataset(
        args.data_root,
        dataset_format=args.suite,
        stats=stats,
        window=WindowOptions(rnn_frames=args.window),
        episode_paths=list(selected.values()),
        split_seed=args.seed,
        samples_per_episode=32,
        hash_episodes=True,
    )
    if len(dataset) == 0:
        raise ValueError("no train identities in the declared source split")
    manifest = dataset.split_manifest()
    if manifest.check():
        raise ValueError(f"split audit failed: {manifest.check()}")
    (output / "split.json").write_text(json.dumps(manifest.as_dict(), indent=2) + "\n")
    encoder = N02Encoder(config).to(args.device)
    readout = N02Readout(config).to(args.device)
    source = EncoderReadoutTokenSource(
        encoder,
        readout,
        token_budget=config.tokens_per_camera,
        max_updates=config.cycles,
        max_history=config.history,
    )
    sample = dataset[0]
    bridge = build_feature_policy(args.suite, args.tokenizer, source, image_keys=sample.camera_ids)
    weights = load_checkpoint_state(args.checkpoint)
    audit = bridge.load_audited_replaced_vision(weights)
    del weights
    bridge.base.to(args.device).eval()
    optimizer = torch.optim.AdamW([*encoder.parameters(), *readout.parameters()], lr=args.lr)
    if resume:
        encoder.load_state_dict(resume["encoder"])
        readout.load_state_dict(resume["readout"])
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state(resume["cuda_rng"])
    records = list(resume.get("records", [])) if resume else []
    skipped_windows = list(resume.get("skipped_windows", [])) if resume else []
    cursor = int(resume.get("dataset_cursor", resume["step"])) if resume else 0
    first_step = int(resume["step"]) if resume else 0
    start = time.perf_counter()
    for index in range(first_step, args.steps):
        batch_start = time.perf_counter()
        for _ in range(len(dataset)):
            sample_index = cursor % len(dataset)
            sample = dataset[sample_index]
            cursor += 1
            if bool(sample.supervision_mask.any()):
                break
            skipped_windows.append(
                {"dataset_index": sample_index, "reason": "all future actions are padding"}
            )
        else:
            raise ValueError("selected trajectory has no window with valid action supervision")
        source.reset()
        optimizer.zero_grad(set_to_none=True)
        lang, lang_mask, proprio = language_state(bridge, sample, args.device)
        # Burn-in carries causal state; loss/backward uses the newest window frame.
        for step in range(len(sample.observation_indices) - 1):
            with torch.no_grad():
                source({"frames": sample_frames(sample, step, args.device)})
            source.detach()
        observation = {
            "frames": sample_frames(sample, len(sample.observation_indices) - 1, args.device),
            "context": EmbodimentContext(proprio=proprio, proprio_available=True),
        }
        result = bridge.forward(
            observation=observation,
            lang_tokens=lang,
            lang_masks=lang_mask,
            state=proprio,
            actions=bridge.prepare_control_targets(sample.supervision_actions, device=args.device),
            action_mask=sample.supervision_mask[None],
        )
        loss = result["action_loss"]
        loss.backward()
        gradients = {
            "encoder": sum(
                float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None
            ),
            "readout": sum(
                float(p.grad.abs().sum()) for p in readout.parameters() if p.grad is not None
            ),
        }
        frozen_grad = [name for name, p in bridge.base.named_parameters() if p.grad is not None]
        if (
            not bool(torch.isfinite(loss))
            or not all(np.isfinite(value) and value > 0 for value in gradients.values())
            or frozen_grad
        ):
            raise RuntimeError(f"gradient acceptance failed: {gradients}, frozen={frozen_grad}")
        torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *readout.parameters()], 1.0)
        optimizer.step()
        source.detach()
        record = {
            "step": index,
            "dataset_index": sample_index,
            "action_loss": float(loss.detach()),
            "gradient_abs_sum": gradients,
            "frozen_parameters_with_grad": frozen_grad,
            "run_stats": bridge.last_run_stats,
            "episode": sample.episode.episode_group_id,
            "npz_sha256": sample.episode.sha256,
            "observation_indices": sample.observation_indices.tolist(),
            "supervision_indices": sample.supervision_indices.tolist(),
            "supervision_mask": sample.supervision_mask.tolist(),
            "batch_seconds": time.perf_counter() - batch_start,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "readout": readout.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.as_dict(),
                "step": index + 1,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if args.device.startswith("cuda") else None,
                "run": vars(args),
                "records": records,
                "skipped_windows": skipped_windows,
                "dataset_cursor": cursor,
            },
            output / "checkpoint.pt",
        )
    inference_samples = []
    for _ in range(args.inference_repeats):
        source.reset()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        prediction = bridge.predict_action(
            observation=observation,
            lang_tokens=lang,
            lang_masks=lang_mask,
            state=proprio,
            num_steps=10,
        )
        if not bool(torch.isfinite(torch.as_tensor(prediction["control_actions"])).all()):
            raise RuntimeError("nonfinite inference actions")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_samples.append(time.perf_counter() - inference_start)
    report = {
        "status": "passed",
        "scope": "real offline NPZ/PI05 gradients and inference; no closed-loop score",
        "suite": args.suite,
        "records": records,
        "skipped_windows": skipped_windows,
        "load_audit": audit,
        "parameters": bridge.parameter_report(),
        "inference_stats": bridge.last_run_stats,
        "actions_shape": list(prediction["control_actions"].shape),
        "seconds": time.perf_counter() - start,
        "inference_samples_seconds": inference_samples,
        "inference_ms": {
            "p50": float(np.percentile(inference_samples, 50) * 1000),
            "p95": float(np.percentile(inference_samples, 95) * 1000),
            "n": len(inference_samples),
        },
        "inference_timing_scope": "encoder/readout/10 denoise steps; excludes NPZ loading",
        "checkpoint_sha256": file_hash(output / "checkpoint.pt"),
        "backend_checkpoint_sha256": file_hash(args.checkpoint),
        "runner_sha256": file_hash(Path(__file__)),
        "source_files_sha256": {
            str(path.relative_to(Path(__file__).parents[1])): file_hash(path)
            for path in sorted(Path(__file__).parents[1].rglob("*.py"))
        },
        "config": config.as_dict(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        "command_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stats_sha256": stats.sha256,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()
        if args.device.startswith("cuda")
        else None,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", choices=("libero", "robotwin"), required=True)
    for name in ("checkpoint", "tokenizer", "data-root", "stats", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stats-key", default="")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--inference-repeats", type=int, default=10)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"]}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
