"""N01 v1.2 P3 real action training, arm matrix and on-server verification.

This is the formal entry point.  It deliberately reuses the plumbing that P0/P1
already accepted -- the real loader binding (``dataset_from_protocol``), the same
observation builder, the same π0.5 action-flow-matching loss and the same frozen
backend loader -- and replaces the parts that were only a prototype: the encoder
factory (now :mod:`owac.frameworks.n01_full.model`), the window length and the
training loop itself.  ``owac.training.n01_action``'s loop (20 steps, tiny crop,
old N01 factory) is never called here.

Sub-commands
------------
``matrix``  write the eight main-run configurations of one seed
``verify``  real-batch checks before any long run: fields, camera order, the
            same-condition RAW/RGB pair, that the gradient reaches the encoder
            and the bridge, and that the frozen backend accumulates none
``train``   the real optimizer loop with resume, checkpoints and progress events

Everything the loop reports is a real measurement.  There is no convergence gate
here: the run stops on its registered budget, on a step limit, or when the user
stops it, and the report of how far it got is the honest deliverable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from owac.frameworks.n01_full.model import (
    ARM_METHODS,
    PLATFORMS,
    N01V12Config,
    N01V12Model,
    arm_matrix,
    build_model,
)

__all__ = ["main", "train", "verify", "matrix", "run_directory_layout"]


DEFAULT_PROTOCOL = "configs/encoders/n01_full/protocol_v1.json"
DEFAULT_TOKENIZER = "source/openpi_converted_protocol/paligemma_tokenizer.model"


def _protocol_path(repo_root: Path, value: str | None) -> Path:
    path = Path(value) if value else repo_root / DEFAULT_PROTOCOL
    if not path.is_file():
        raise FileNotFoundError(f"protocol file not found: {path}")
    return path


def run_directory_layout(run_root: Path) -> dict[str, Path]:
    """``source/config/outputs/receipts`` plus the monitor directory."""

    return {
        "source": run_root / "source",
        "config": run_root / "config",
        "outputs": run_root / "outputs",
        "receipts": run_root / "receipts",
        "monitor": run_root / "monitor",
        "workers": run_root / "monitor" / "workers",
    }


def ensure_layout(run_root: Path) -> dict[str, Path]:
    layout = run_directory_layout(run_root)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


ProgressWriter = None  # replaced below by the single real writer


def _progress_writer(
    run_root: Path,
    role: str,
    *,
    config: N01V12Config,
    planned_steps: int | None,
    stage: str = "P3",
    gpu_index: int | None = None,
):
    """The one progress writer for every worker: the monitor service's own class.

    Training and evaluation must not each carry their own writer, or the
    reconciler would see two incompatible worker files for one run.  The import is
    deferred so this module still imports on a machine without the runtime
    package installed.
    """

    from owac.runtime.n01_v12_monitor import RunStateWriter

    spec = config.spec
    return RunStateWriter(
        run_root,
        role,
        method=config.arm,
        input_domain=spec.input_domain,
        data_kind=config.data_kind,
        platform=config.platform,
        seed=config.seed,
        planned_steps=planned_steps,
        stage=stage,
        gpu_index=gpu_index,
    )


def encode_windows_v12(
    model: N01V12Model, batch: Any, device: Any, *, tbptt: int | None = None, transform=None
):
    """Push one real window per camera through the complete v1.2 model.

    Frame construction, masks and camera handling are the accepted ones; what
    changes is that the compute budget comes from the v1.2 configuration (a
    32-frame bounded window) and the encoder is the v1.2 model rather than the
    prototype.  The prototype helper reads ``encoder.config.num_blocks``, which
    the v1.2 adapter configuration does not define, so it cannot be reused here.
    """

    import torch

    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame, VisualTokens

    tokens: list[Any] = []
    diagnostics: dict[str, Any] = {}
    window_frames = int(model.config.window_frames)
    budget = ComputeBudget(max_updates=window_frames, max_history=window_frames)
    for camera, window in batch["windows"].items():
        values = window.values.to(device)
        batch_size, length, _, height, width = values.shape
        state = model.init_state(
            batch_size, {"camera_ids": [camera], "shape": (height, width)}, device
        )
        state.time_unit = window.time_unit
        output = None
        for index in range(length):
            valid = window.valid_mask[:, index].to(device)
            metadata = {}
            if window.variance is not None:
                metadata["variance"] = window.variance[:, index].to(device)
            frame = MeasurementFrame(
                values=values[:, index],
                valid_mask=valid,
                bad_mask=window.bad_mask[:, index].to(device),
                saturated_mask=window.saturated_mask[:, index].to(device),
                frame_ids=window.frame_ids[:, index].to(device),
                camera_id=camera,
                episode_ids=[
                    str(item.get("episode_key", item.get("episode", "isolated-window")))
                    for item in batch.get("meta", [{}] * batch_size)
                ],
                timestamp=window.timestamp[:, index].to(device),
                time_unit=window.time_unit,
                new_measurement=window.new_measurement[:, index].to(device),
                sample_xy=window.sample_xy,
                metadata=metadata,
            )
            step_budget = budget
            if model.config.motion is not None and index + 1 < length:
                step_budget = ComputeBudget(max_updates=0, max_history=window_frames)
            output = model.step(frame, None, state, step_budget)
            state = output.state
            if (
                model.config.motion is None
                and tbptt
                and (index + 1) % tbptt == 0
                and index + 1 < length
            ):
                state = model.detach_state(state)
        if transform is not None:
            output = transform(camera, output)
        tokens.append(model.readout(output))
        diagnostics[camera] = {**output.diagnostics, "stage_cost": output.cost}
    combined = VisualTokens(
        torch.cat([entry.values for entry in tokens], 1),
        torch.cat([entry.mask for entry in tokens], 1),
        [camera for entry in tokens for camera in entry.camera_ids],
        torch.cat([entry.positions for entry in tokens], 1),
    )
    return combined, diagnostics


def _suite(platform: str) -> str:
    return "libero" if platform == "libero" else "robotwin"


def _weights_and_tokenizer(repo_root: Path, deploy_root: Path, platform: str) -> tuple[Path, Path]:
    converted = deploy_root / "source" / "openpi_converted_protocol"
    if platform == "libero":
        weights = converted / "pi05_libero_starvla" / "bfloat16" / "model.safetensors"
    else:
        weights = deploy_root / "checkpoints" / "pi05_robotwin" / "model.safetensors"
    tokenizer = converted / "paligemma_tokenizer.model"
    for path in (weights, tokenizer):
        if not path.is_file():
            raise FileNotFoundError(
                f"frozen backend asset missing: {path}; the P3 run cannot be started "
                "without the real π0.5 weights"
            )
    return weights, tokenizer


def _build_datasets(
    repo_root: Path, protocol_path: Path, platform: str, region: str, window: int, config=None
):
    """Bind the real loader with the arm's declared window length.

    ``RawVLASequenceDataset`` defaults to an 8-frame window.  Leaving that default
    in place while the configuration says 32 would mean the run only ever saw 8
    frames, so the window is passed explicitly and the effective value is echoed
    into every receipt.
    """

    from owac.raw.n01_full_manifest import dataset_from_protocol

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return dataset_from_protocol(
        protocol,
        platform,
        region,
        repo_root=str(repo_root),
        window=int(window),
        **(
            {"observation_stride": config.observation.history_stride}
            if config is not None and config.observation is not None
            else {}
        ),
    )


def matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Write the eight main-run configs and the seed/cost declaration."""

    run_root = Path(args.run_root)
    layout = ensure_layout(run_root)
    configs = arm_matrix(
        platforms=tuple(args.platforms),
        methods=tuple(args.methods),
        seed=int(args.seed),
        window_frames=int(args.window_frames),
    )
    written = []
    for config in configs:
        name = f"{config.platform}-{config.arm}-seed{config.seed}.json"
        path = layout["config"] / name
        _write_json_atomic(path, config.to_dict())
        written.append({"path": str(path), "platform": config.platform, "arm": config.arm})
    receipt = {
        "stage": "P3",
        "seed": int(args.seed),
        "platforms": list(args.platforms),
        "methods": list(args.methods),
        "runs": len(configs),
        "window_frames": int(args.window_frames),
        "single_seed_note": (
            "One training seed per platform and method. This is not evidence about "
            "training-seed variance and no stability claim is made from it."
        ),
        "seeds_not_run": [211, 307],
        "configs": written,
    }
    _write_json_atomic(layout["receipts"] / "matrix.json", receipt)
    return receipt


def verify(args: argparse.Namespace) -> dict[str, Any]:
    """Real-batch pre-run verification on the actual RawVLA-Bench data."""

    import numpy as np
    import torch

    from owac.raw.n01_paired_views import paired_views
    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.training.n01_action import (
        action_masked_loss,
        load_backend,
        observation_from_batch,
    )

    repo_root = Path(args.repo_root)
    deploy_root = Path(args.deploy_root)
    config = N01V12Config.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    protocol_path = _protocol_path(repo_root, args.protocol)
    train_set = _build_datasets(
        repo_root, protocol_path, config.platform, "train", config.window_frames, config
    )
    if getattr(args, "full_history", False):
        path = train_set.paths[0]
        arrays = train_set._load_episode(path)
        current = max(
            len(arrays["actions"]) - train_set.action_offset - train_set.action_horizon, 0
        )
        sample = train_set.build_sample(path, arrays, current)
    else:
        sample = train_set[0]
    batch = collate_sequence_samples([sample])
    requested_history = int(getattr(args, "history_frames", 0))
    if requested_history:
        from dataclasses import fields, replace

        from owac.raw.rawvla_sequence import MeasurementWindowBatch

        if not 1 <= requested_history <= config.window_frames:
            raise ValueError("verification history must fit the configured bound")
        for camera, window in list(batch["windows"].items()):
            updates = {}
            for entry in fields(MeasurementWindowBatch):
                value = getattr(window, entry.name)
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 2
                    and value.shape[1] == window.values.shape[1]
                ):
                    updates[entry.name] = value[:, -requested_history:]
            batch["windows"][camera] = replace(window, **updates)
    if not bool(batch["action_mask"].any()):
        raise ValueError("verification requires at least one genuine action target")
    cameras = list(batch["windows"].keys())

    report: dict[str, Any] = {
        "arm": config.arm,
        "platform": config.platform,
        "data_kind": config.data_kind,
        "seed": config.seed,
        "train_samples": len(train_set),
        "cameras_from_loader": cameras,
        "expected_cameras": list(config.cameras),
        "raw_field_shape": {},
        "same_condition_pair": {},
    }
    if cameras != list(config.cameras):
        raise ValueError(f"loader camera order {cameras} != config {list(config.cameras)}")

    # 1) The real RAW field, and the RGB view rendered from that same tensor.
    for camera in cameras:
        window = batch["windows"][camera]
        values = torch.as_tensor(window.values)
        report["raw_field_shape"][camera] = list(values.shape)
        report["raw_dtype"] = str(values.dtype)
        report["raw_range"] = [float(values.min()), float(values.max())]
        pair = paired_views(values, platform=config.platform)
        report["same_condition_pair"][camera] = {
            "raw_domain": pair.raw_domain,
            "rgb_domain": pair.rgb_domain,
            "identical_source": bool(torch.equal(pair.raw, values)),
            "rgb_is_not_identity": not torch.allclose(pair.rgb, values, atol=1e-3),
            "rgb_range": [float(pair.rgb.min()), float(pair.rgb.max())],
            "provenance_module": pair.provenance["module"],
            "cache_dtype": pair.provenance["cache_dtype"],
        }

    # 2) Finite code verification; parameter gradients are not optimizer updates.
    verification_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, device=device)
    model.train()
    model.encoder.profile_stages = True
    weights, tokenizer = _weights_and_tokenizer(repo_root, deploy_root, config.platform)
    policy, audit = load_backend(_suite(config.platform), weights, tokenizer, device)
    bridge = _v12_bridge(model, policy, config)
    phase = getattr(args, "backend_phase", "warmup")
    if config.training is not None:
        from owac.frameworks.n01_full.training_plan import BackendSchedule

        scope = BackendSchedule(model, policy, config.training)
        scope.apply(config.training.warmup_steps if phase == "adaptation" else 0)
        report["backend_phase"] = scope.report(
            config.training.warmup_steps if phase == "adaptation" else 0
        )
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    bridge_parameters = [p for p in model.bridge.parameters() if p.requires_grad]
    if not encoder_parameters or not bridge_parameters:
        raise RuntimeError("encoder or bridge exposes no trainable parameter")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_started = time.perf_counter()
    with torch.set_grad_enabled(not getattr(args, "forward_only", False)):
        tokens, diagnostics = encode_windows_v12(model, batch, device, tbptt=args.tbptt)
        observation = observation_from_batch(policy, batch, device)
        actions = batch["action"].to(device)
        actions = torch.nn.functional.pad(
            actions, (0, max(int(policy.action_dim) - int(actions.shape[-1]), 0))
        )
        result = _policy_forward(bridge, observation, tokens, actions, return_debug=True)
        loss = action_masked_loss(
            result["v_t"],
            result["u_t"],
            batch["action_mask"].to(device),
            train_set.protocol.action_dim,
        )
        if not getattr(args, "forward_only", False):
            loss.backward()
    encoder_grad = sum(
        float(p.grad.detach().abs().sum()) for p in encoder_parameters if p.grad is not None
    )
    bridge_grad = sum(
        float(p.grad.detach().abs().sum()) for p in bridge_parameters if p.grad is not None
    )
    backend_grad = sum(
        float(p.grad.detach().abs().sum()) for p in policy.parameters() if p.grad is not None
    )
    provider = getattr(model.encoder, "motion_provider", None)
    motion_grad = (
        sum(float(p.grad.detach().abs().sum()) for p in provider.parameters() if p.grad is not None)
        if provider is not None
        else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report["compute_forward_backward_seconds"] = time.perf_counter() - compute_started
    report["verification_seconds_including_weight_load"] = (
        time.perf_counter() - verification_started
    )
    report["peak_gpu_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    report["unique_frames"] = {
        camera: len(set(batch["windows"][camera].frame_ids[0].tolist())) for camera in cameras
    }
    report["action_target_mask_count"] = int(batch["action_mask"].sum())
    report["motion_auxiliary"] = None
    if getattr(args, "check_motion_auxiliary", False) and provider is not None:
        model.zero_grad(set_to_none=True)
        auxiliary = _motion_auxiliary(model, batch, device, 0)
        auxiliary.backward()
        report["motion_auxiliary"] = {
            "kind": "known_spatial_transform_on_real_own_domain_frame",
            "loss": float(auxiliary.detach()),
            "grad_abs_sum": sum(
                float(p.grad.detach().abs().sum())
                for p in provider.parameters()
                if p.grad is not None
            ),
            "optimizer_updates": 0,
        }
    report["optimizer_updates"] = 0
    report["forward_only"] = bool(getattr(args, "forward_only", False))
    report["motion_gradient_from_native_action_loss"] = motion_grad
    report["loss"] = {
        "train_action_loss": float(loss.detach()),
        "encoder_grad_abs_sum": encoder_grad,
        "bridge_grad_abs_sum": bridge_grad,
        "backend_grad_abs_sum": backend_grad,
        "backend_is_frozen": not any(p.requires_grad for p in policy.parameters()),
        "visual_tokens_shape": list(tokens.values.shape),
        "visual_token_mask_valid": int(tokens.mask.sum()),
    }
    report["support"] = {
        camera: {
            "applied": bool(entry.get("p3_support_applied")),
            "status": entry.get("p3_support_status"),
            "window_frames": entry.get("window_frames"),
            "stage_cost": entry.get("stage_cost"),
            "state_counts": (entry.get("support") or {}).get("state_counts"),
        }
        for camera, entry in diagnostics.items()
    }
    report["effective_window_frames"] = {
        camera: int(batch["windows"][camera].values.shape[1]) for camera in cameras
    }
    report["parameters"] = model.parameter_report()
    report["backend_audit"] = audit if isinstance(audit, dict) else str(audit)
    report["initialisation"] = (
        "shared init rule: identical module shapes across the four arms, so one seed "
        "gives a comparable starting point; domain-specific pretrained weights are "
        "recorded separately as a cost, not mixed into this controlled comparison"
    )
    out = Path(args.output) if args.output else None
    if out is not None:
        _write_json_atomic(out, report)
    return report


def _policy_forward(bridge, observation, tokens, actions, **kwargs):
    """BF16 policy kernels with FP32 trainable action-expert master parameters."""
    import torch

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=actions.device.type == "cuda"):
        return bridge.forward_from_observation_with_features(observation, tokens, actions, **kwargs)


def _v12_bridge(model: N01V12Model, policy: Any, config: N01V12Config):
    """Attach the frozen π0.5 backend to *this* model's encoder and bridge."""

    from owac.backends.rawvla_features import ACTION_PROTOCOLS, RawVLAFeaturePolicy

    suite = _suite(config.platform)
    bridge = RawVLAFeaturePolicy(
        policy,
        encoder=model.encoder,
        postprocessor=model.bridge,
        action_protocol=ACTION_PROTOCOLS[suite],
    )
    bridge.freeze_backend_parameters()
    return bridge


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch_save = __import__("torch").save
    torch_save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    torch = __import__("torch")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch without the flag
        return torch.load(path, map_location="cpu")


@dataclass
class StepReport:
    step: int
    micro: int
    loss: float
    seconds: float


def _sample_order(length: int, seed: int, epoch: int) -> list[int]:
    order = list(range(length))
    random.Random(seed * 1000003 + epoch).shuffle(order)
    return order


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])


@contextmanager
def _isolated_rng():
    """Validation must not change the training flow times/noise, even on error."""
    state = _rng_state()
    try:
        yield
    finally:
        _restore_rng(state)


@contextmanager
def _validation_state(model: Any):
    import torch

    was_training = model.training
    try:
        model.eval()
        with _isolated_rng(), torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _training_contract(args: argparse.Namespace, config: N01V12Config, dataset: Any) -> dict:
    """Resume identity includes the data order and observation/optimizer protocol."""
    return {
        "implementation": "p3-complete-code-20261017",
        "diagnostic": bool(args.diagnostic),
        "config": config.to_dict(),
        "dataset_length": len(dataset),
        "dataset_paths": list(dataset.paths),
        "accum": max(int(args.accum), 1),
        "batch_size": int(getattr(args, "batch_size", 1)),
        "precision": getattr(args, "precision", "fp32"),
        "flow_batch_size": int(getattr(args, "flow_batch_size", 1)),
        "backend_master_weights": bool(config.training is not None),
        "crop": int(args.crop),
        "tbptt": args.tbptt,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
    }


def _check_resume(payload: dict, contract: dict) -> None:
    if payload.get("resume_schema") != "n01-v12-training-v2":
        raise ValueError(
            "legacy checkpoint lacks sampler/CUDA RNG state; exact continuation cannot "
            "be claimed. Preserve it and use a separate diagnostic run or a new run."
        )
    if payload.get("training_contract") != contract:
        raise ValueError(
            "checkpoint training contract differs from requested data/config/optimizer"
        )


def _sample_position(length: int, consumed: int) -> tuple[int, int]:
    if length <= 0 or consumed < 0:
        raise ValueError("dataset must be nonempty and consumed samples nonnegative")
    return divmod(consumed, length)


def _require_training_scope(args: argparse.Namespace, model: Any) -> None:
    if args.diagnostic:
        return
    if int(args.crop):
        raise ValueError(
            "--crop is a top-left diagnostic crop, not a full-frame resize. "
            "Register matching train/serve geometry before formal training."
        )
    if getattr(model.encoder, "motion_provider", None) is None:
        raise ValueError(
            "P3 motion correspondence is not connected; identity_pending_p2 is an "
            "incomplete model. --diagnostic explicitly permits diagnostic training only."
        )
    if model.config.observation is None or model.config.training is None:
        raise ValueError("formal training requires the shared observation contract and staged plan")
    if model.config.spec.support_enabled and model.config.support.implementation != "gls_window":
        raise ValueError("the historical proxy is not the complete support implementation")


def _motion_auxiliary(model, batch, device, step):
    from owac.core.encoder_contract import MeasurementFrame
    from owac.frameworks.n01_full.learned_motion import known_transform_loss
    from owac.frameworks.n01_full.observation import prepare_frame

    losses = []
    shift = ((2, 0), (0, 2), (-2, 1))[step % 3]
    for camera, window in batch["windows"].items():
        frame = MeasurementFrame(
            window.values[:, -1].to(device),
            camera_id=camera,
            valid_mask=window.valid_mask[:, -1].to(device),
            bad_mask=window.bad_mask[:, -1].to(device),
            saturated_mask=window.saturated_mask[:, -1].to(device),
        )
        frame = prepare_frame(frame, model.config.observation, model.config.spec.input_domain)
        losses.append(
            known_transform_loss(
                model.encoder.motion_provider,
                frame.values,
                frame.valid_mask & ~frame.bad_mask & ~frame.saturated_mask,
                shift=shift,
            )
        )
    return sum(losses) / max(len(losses), 1)


def train(args: argparse.Namespace) -> dict[str, Any]:
    """The real training loop: real data, real optimizer steps, real resume."""

    import torch

    from owac.training.n01_action import (
        action_masked_loss,
        load_backend,
        observation_from_batch,
    )

    repo_root = Path(args.repo_root)
    from owac.training.n01_v12_guard import assert_training_allowed

    assert_training_allowed(repo_root, args.run_root)
    deploy_root = Path(args.deploy_root)
    run_root = Path(args.run_root)
    layout = ensure_layout(run_root)
    output = Path(args.output)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)

    config = N01V12Config.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    device = torch.device(args.device)
    plan = config.training
    if plan is not None:
        if args.lr is not None and float(args.lr) != plan.encoder_lr:
            raise ValueError("register LR changes in the effective training plan")
        if not args.max_steps:
            args.max_steps = plan.total_steps
    if args.lr is None:
        args.lr = plan.encoder_lr if plan is not None else 1e-5
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    import numpy as np

    np.random.seed(config.seed)

    role = f"{config.platform}-{config.arm}-seed{config.seed}"
    progress = _progress_writer(
        run_root,
        role,
        config=config,
        planned_steps=int(args.max_steps) or None,
        gpu_index=int(args.gpu_index) if args.gpu_index is not None else None,
    )
    started = time.perf_counter()
    import threading

    heartbeat_stop = threading.Event()

    def keep_alive():
        while not heartbeat_stop.wait(15):
            progress.heartbeat()

    heartbeat_thread = threading.Thread(target=keep_alive, daemon=True)
    heartbeat_thread.start()
    progress.update(phase="loading", activity="加载数据和原生动作后端")
    try:
        protocol_path = _protocol_path(repo_root, args.protocol)
        train_set = _build_datasets(
            repo_root, protocol_path, config.platform, "train", config.window_frames, config
        )
        steps_done = 0
        optimizer = None
        model = build_model(config, device=device)
        from owac.frameworks.n01_full.acceleration import configure_acceleration

        execution = configure_acceleration(
            model,
            device,
            getattr(args, "precision", "fp32"),
            pair_chunk=int(getattr(args, "flow_batch_size", 1)),
        )
        _require_training_scope(args, model)
        weights, tokenizer = _weights_and_tokenizer(repo_root, deploy_root, config.platform)
        policy, audit = load_backend(_suite(config.platform), weights, tokenizer, device)
        bridge = _v12_bridge(model, policy, config)
        parameters = [p for p in model.encoder.parameters() if p.requires_grad] + [
            p for p in model.bridge.parameters() if p.requires_grad
        ]
        schedule = None
        if plan is not None:
            from owac.frameworks.n01_full.training_plan import BackendSchedule

            schedule = BackendSchedule(model, policy, plan, master_weights=True)
            parameters = schedule.parameter_groups()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            fused=device.type == "cuda",
        )
        contract = _training_contract(args, config, train_set)
        samples_consumed = 0
        from owac.training.n01_v12_sampling import Coverage

        coverage = Coverage()

        resume_path = Path(args.resume) if args.resume else output / "checkpoints" / "latest.pt"
        if resume_path.is_file():
            payload = _load_checkpoint(resume_path)
            _check_resume(payload, contract)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            steps_done = int(payload["step"])
            if schedule is not None:
                schedule.restore(payload["backend_adaptation"])
            samples_consumed = int(payload["samples_consumed"])
            coverage = Coverage.from_dict(payload.get("coverage", {}))
            _restore_rng(payload)
            progress.update(activity=f"从 checkpoint 续训 step={steps_done}")
        progress.update(
            activity=f"真实动作训练 {config.platform}/{config.arm}",
            step=steps_done,
            backend_audit_ok=True,
            phase="encoder_warmup" if schedule is None else schedule.phase,
        )

        accum = max(int(args.accum), 1)
        micro_index = 0
        action_sum = auxiliary_sum = total_sum = 0.0
        from owac.training.n01_v12_input import PrefetchBatches

        batch_size = int(getattr(args, "batch_size", 1))
        batches = PrefetchBatches(
            train_set, config.seed, samples_consumed, batch_size, int(getattr(args, "prefetch", 0))
        )
        last_loss = float("nan")
        last_action_loss = float("nan")
        interval_started = time.perf_counter()
        interval_step = steps_done
        data_wait = 0.0
        optimizer.zero_grad(set_to_none=True)
        if int(getattr(args, "initial_validation", 0)) and steps_done == 0:
            progress.update(activity="测量训练前的验证误差")
            baseline = _validate(
                args, config, model, bridge, policy, train_set, device, repo_root, protocol_path
            )
            progress.metric(
                "val_action_loss", 0, baseline, unit="loss", label="固定验证 action loss"
            )
            progress.update(validation_loss=baseline, baseline_validation_loss=baseline)
            interval_started = time.perf_counter()
        while (not args.max_steps) or steps_done < int(args.max_steps):
            loading = time.perf_counter()
            sample_indices, samples, batch = next(batches)
            data_wait += time.perf_counter() - loading
            for sample_index, sample in zip(sample_indices, samples, strict=True):
                coverage.observe(sample_index, getattr(sample, "meta", {}))
            samples_consumed += len(samples)
            if args.crop:
                from owac.runtime.n01_full_ledger_check import _crop_batch

                _crop_batch(batch, int(args.crop))
            model.train()
            if schedule is not None:
                schedule.apply(steps_done, optimizer)
            tokens, diagnostics = encode_windows_v12(model, batch, device, tbptt=args.tbptt)
            observation = observation_from_batch(policy, batch, device)
            actions = batch["action"].to(device)
            actions = torch.nn.functional.pad(
                actions, (0, max(int(policy.action_dim) - int(actions.shape[-1]), 0))
            )
            result = _policy_forward(bridge, observation, tokens, actions, return_debug=True)
            loss = action_masked_loss(
                result["v_t"],
                result["u_t"],
                batch["action_mask"].to(device),
                train_set.protocol.action_dim,
            )
            action_loss = loss
            auxiliary = loss.new_zeros(())
            if plan is not None and plan.motion_aux_weight:
                auxiliary = _motion_auxiliary(model, batch, device, samples_consumed)
                loss = loss + plan.motion_aux_weight * auxiliary
            (loss / accum).backward()
            action_sum += float(action_loss.detach())
            auxiliary_sum += float(auxiliary.detach())
            total_sum += float(loss.detach())
            micro_index += 1
            if micro_index % accum:
                continue
            if plan is not None:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for group in optimizer.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ],
                    plan.gradient_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1
            last_loss, last_action_loss = total_sum / accum, action_sum / accum
            last_auxiliary = auxiliary_sum / accum
            action_sum = auxiliary_sum = total_sum = 0.0
            progress.update(
                step=steps_done,
                step_progress_at=time.time(),
                phase="encoder_warmup" if schedule is None else schedule.phase,
                activity=f"训练中 action loss={last_action_loss:.4f}",
            )
            if steps_done % int(args.log_every) == 0 or steps_done <= 2:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval_seconds = now - interval_started
                step_seconds = interval_seconds / max(steps_done - interval_step, 1)
                elapsed = now - started
                support_counts = [
                    entry.get("support", {}).get("state_counts", {})
                    for entry in diagnostics.values()
                    if entry.get("support")
                ]
                accepted = sum(x.get("accepted", 0) for x in support_counts)
                total = sum(sum(x.values()) for x in support_counts)
                progress.update(
                    step=steps_done,
                    activity=f"训练中 action loss={last_action_loss:.4f}",
                    step_seconds=step_seconds,
                    samples_per_second=batch_size * accum / step_seconds,
                    data_wait_fraction=data_wait / max(interval_seconds, 1e-6),
                    phase="encoder_warmup" if schedule is None else schedule.phase,
                    action_loss=last_action_loss,
                    motion_loss=last_auxiliary,
                    history_accepted_fraction=accepted / total if total else None,
                    execution={
                        **execution,
                        "batch_size": batch_size,
                        "accum": accum,
                        "effective_batch_size": batch_size * accum,
                    },
                    peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else None,
                )
                interval_started, interval_step, data_wait = now, steps_done, 0.0
                progress.metric(
                    "train_action_loss",
                    steps_done,
                    last_action_loss,
                    unit="loss",
                    label="训练 action loss",
                )
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "role": role,
                            "step": steps_done,
                            "loss": last_loss,
                            "action_loss": float(action_loss.detach()),
                            "motion_known_transform_loss": last_auxiliary,
                            "phase": "frozen_backend_diagnostic"
                            if schedule is None
                            else schedule.phase,
                            "elapsed_s": round(elapsed, 2),
                            # ``diagnostics`` is keyed by camera, so the flag has to be
                            # read per camera: an outer-level lookup would always be False.
                            "support_applied": {
                                camera: bool(entry.get("p3_support_applied"))
                                for camera, entry in diagnostics.items()
                            },
                            "support_status": {
                                camera: entry.get("p3_support_status")
                                for camera, entry in diagnostics.items()
                            },
                            "support_state_counts": {
                                camera: (entry.get("support") or {}).get("state_counts")
                                for camera, entry in diagnostics.items()
                            },
                            "motion_status": {
                                camera: entry.get("p2_motion_status")
                                for camera, entry in diagnostics.items()
                            },
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if int(args.val_every) and (
                steps_done % int(args.val_every) == 0
                or steps_done == int(getattr(args, "early_val_step", 0))
            ):
                progress.update(activity=f"正在验证 @第{steps_done}步")
                value = _validate(
                    args, config, model, bridge, policy, train_set, device, repo_root, protocol_path
                )
                progress.metric(
                    "val_action_loss",
                    steps_done,
                    value,
                    unit="loss",
                    label="固定验证 action loss",
                )
                if getattr(args, "random_val_samples", 0):
                    random_value = _validate(
                        args,
                        config,
                        model,
                        bridge,
                        policy,
                        train_set,
                        device,
                        repo_root,
                        protocol_path,
                        mode="random",
                        step=steps_done,
                    )
                    progress.metric(
                        "random_val_action_loss",
                        steps_done,
                        random_value,
                        unit="loss",
                        label="随机验证 action loss",
                    )
                _write_json_atomic(output / "coverage.json", coverage.report(len(train_set)))
                progress.update(
                    validation_loss=value,
                    validation_step=steps_done,
                    activity=f"验证 loss={value:.4f} @step{steps_done}",
                )
                interval_started = time.perf_counter()
                interval_step = steps_done
            if int(args.ckpt_every) and (steps_done % int(args.ckpt_every) == 0 or steps_done == 2):
                _save_checkpoint(
                    output / "checkpoints" / "latest.pt",
                    {
                        "step": steps_done,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "backend_adaptation": {} if schedule is None else schedule.checkpoint(),
                        "training_phase": None if schedule is None else schedule.report(steps_done),
                        **_rng_state(),
                        "resume_schema": "n01-v12-training-v2",
                        "training_contract": contract,
                        "samples_consumed": samples_consumed,
                        "coverage": coverage.to_dict(),
                        "config": config.to_dict(),
                        "arm": config.arm,
                        "platform": config.platform,
                        "seed": config.seed,
                    },
                )
                progress.update(
                    checkpoint_step=steps_done,
                    checkpoint_at=time.time(),
                    activity=f"已保存 checkpoint @step{steps_done}",
                )
                interval_started = time.perf_counter()
                interval_step = steps_done

        _save_checkpoint(
            output / "checkpoints" / "latest.pt",
            {
                "step": steps_done,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "backend_adaptation": {} if schedule is None else schedule.checkpoint(),
                "training_phase": None if schedule is None else schedule.report(steps_done),
                **_rng_state(),
                "resume_schema": "n01-v12-training-v2",
                "training_contract": contract,
                "samples_consumed": samples_consumed,
                "coverage": coverage.to_dict(),
                "config": config.to_dict(),
                "arm": config.arm,
                "platform": config.platform,
                "seed": config.seed,
            },
        )
        batches.close()
        receipt = {
            "stage": "P3",
            "run_kind": "diagnostic" if args.diagnostic else "formal",
            "role": role,
            "arm": config.arm,
            "platform": config.platform,
            "seed": config.seed,
            "optimizer_steps": steps_done,
            "samples_consumed": samples_consumed,
            "training_contract": contract,
            "max_steps": int(args.max_steps) or None,
            "final_train_total_loss": last_loss,
            "final_train_action_loss": last_action_loss,
            "coverage": coverage.report(len(train_set)),
            "planned_steps": int(args.max_steps) or None,
            "budget_note": (
                "the planning band is 30k-100k real action optimizer updates per platform, "
                "method and seed; the value actually reached is what this receipt records"
            ),
            "wall_seconds": time.perf_counter() - started,
            "config": config.to_dict(),
            "parameters": model.parameter_report(),
            "backend_audit": audit if isinstance(audit, dict) else str(audit),
            "config_path": str(args.config),
        }
        _write_json_atomic(layout["receipts"] / f"train-{role}.json", receipt)
        progress.finish("budget_reached" if args.max_steps else "stopped_by_request")
        return receipt
    except BaseException as error:  # a crashed run must not look alive
        progress.finish(f"{type(error).__name__}: {error}", state="failed")
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)
        if "batches" in locals():
            batches.close()


def _validate(
    args: argparse.Namespace,
    config: N01V12Config,
    model: N01V12Model,
    bridge: Any,
    policy: Any,
    train_set: Any,
    device: Any,
    repo_root: Path,
    protocol_path: Path,
    *,
    mode="fixed",
    step=0,
) -> float:
    """Fixed-order validation subset with a fixed RNG, so points are comparable.

    The RNG is reset per sample, so the flow time and noise of one validation
    sample do not drift between validation points.  Ordinary random validation on
    the full distribution is a different check and is not claimed here.
    """

    import torch

    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.training.n01_action import (
        action_masked_loss,
        observation_from_batch,
    )
    from owac.training.n01_v12_sampling import validation_indices

    validation_set = _build_datasets(
        repo_root, protocol_path, config.platform, "validation", config.window_frames, config
    )
    count = min(
        int(getattr(args, "random_val_samples", 0) if mode == "random" else args.val_samples),
        len(validation_set),
    )
    if count <= 0:
        raise ValueError("validation requires at least one real sample")
    losses = []
    indices = validation_indices(len(validation_set), count, config.seed, mode=mode, step=step)
    with _validation_state(model):
        for index in indices:
            torch.manual_seed(
                config.seed * 7919 + index + (step * 104729 if mode == "random" else 0)
            )
            batch = collate_sequence_samples([validation_set[index]])
            if not bool(batch["action_mask"].any()):
                continue
            if args.crop:
                from owac.runtime.n01_full_ledger_check import _crop_batch

                _crop_batch(batch, int(args.crop))
            tokens, _ = encode_windows_v12(model, batch, device, tbptt=None)
            observation = observation_from_batch(policy, batch, device)
            actions = batch["action"].to(device)
            actions = torch.nn.functional.pad(
                actions, (0, max(int(policy.action_dim) - int(actions.shape[-1]), 0))
            )
            result = _policy_forward(bridge, observation, tokens, actions, return_debug=True)
            loss = action_masked_loss(
                result["v_t"],
                result["u_t"],
                batch["action_mask"].to(device),
                train_set.protocol.action_dim,
            )
            losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("validation subset has no genuine action targets")
    return float(sum(losses) / len(losses))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(item: argparse.ArgumentParser) -> None:
        item.add_argument("--repo-root", default=".")
        item.add_argument("--deploy-root", default=os.environ.get("N01_DEPLOY_ROOT", "."))
        item.add_argument("--protocol", default=None)
        item.add_argument("--device", default="cuda:0")
        item.add_argument("--crop", type=int, default=0, help="0 keeps the native resolution")
        item.add_argument("--config", required=True)

    build = sub.add_parser("matrix", help="write the eight main-run configs")
    build.add_argument("--run-root", required=True)
    build.add_argument("--platforms", nargs="+", default=list(PLATFORMS))
    build.add_argument("--methods", nargs="+", default=list(ARM_METHODS))
    build.add_argument("--seed", type=int, default=101)
    build.add_argument("--window-frames", type=int, default=32)
    build.set_defaults(func=matrix)

    check = sub.add_parser("verify", help="real-batch pre-run verification")
    common(check)
    check.add_argument("--output", default=None)
    check.add_argument("--tbptt", type=int, default=8)
    check.add_argument("--full-history", action="store_true")
    check.add_argument("--history-frames", type=int, default=0)
    check.add_argument("--check-motion-auxiliary", action="store_true")
    check.add_argument("--forward-only", action="store_true")
    check.add_argument("--backend-phase", choices=("warmup", "adaptation"), default="warmup")
    check.set_defaults(func=verify)

    run = sub.add_parser("train", help="real action training")
    common(run)
    run.add_argument("--run-root", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--resume", default=None)
    run.add_argument(
        "--diagnostic",
        action="store_true",
        help="explicitly allow an incomplete/identity-motion model; never a formal P3 result",
    )
    run.add_argument("--lr", type=float, default=None)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--accum", type=int, default=4)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    run.add_argument("--prefetch", type=int, default=0)
    run.add_argument("--flow-batch-size", type=int, default=8)
    run.add_argument("--initial-validation", type=int, default=0)
    run.add_argument("--early-val-step", type=int, default=0)
    run.add_argument("--max-steps", type=int, default=0)
    run.add_argument("--log-every", type=int, default=10)
    run.add_argument("--val-every", type=int, default=0)
    run.add_argument("--val-samples", type=int, default=8)
    run.add_argument("--random-val-samples", type=int, default=8)
    run.add_argument("--ckpt-every", type=int, default=500)
    run.add_argument("--gpu-index", type=int, default=None)
    run.add_argument("--tbptt", type=int, default=8)
    run.add_argument("--html", default=None, help="unused here; kept out of the trainer")
    run.set_defaults(func=train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        from owac.training.n01_v12_no_update import forbid_optimizer_updates

        with forbid_optimizer_updates():
            result = args.func(args)
    else:
        result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
