"""CLI for the concrete temporal teacher/data/cache/training/pure-runtime path."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ACTIVE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
INACTIVE_IMAGE_KEY = "right_wrist_0_rgb"
CAMERA_ORDER = ("agentview", "wrist")
TASK_INSTRUCTION = "put the cream cheese in the bowl"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _emit(payload: dict[str, Any], *, exit_code: int = 0) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def _config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    return value


def _research_config_digest(config: Mapping[str, Any]) -> str:
    """Canonical recipe digest; authorization and output directory are excluded."""
    import copy
    import hashlib

    value = copy.deepcopy(dict(config))
    value.pop("authorization", None)
    assets = value.get("assets")
    if isinstance(assets, Mapping):
        assets.pop("output_root", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _asset_set(config: dict[str, Any], *, device: str | None = None) -> Any:
    from owac.raw.temporal_distillation import TemporalAssetSet

    assets = TemporalAssetSet.from_config(config, repo_root=REPO_ROOT)
    return assets if device is None else replace(assets, device=device)


def _output(path: str | None) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _pure_paths(config: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    assets = config.get("assets", {})
    return SimpleNamespace(
        checkpoint=Path(str(assets.get("checkpoint", ""))),
        tokenizer=Path(str(assets.get("tokenizer", ""))),
        statistics=Path(str(assets.get("statistics", ""))),
        sources_path=Path(str(assets.get("source_manifest", ""))),
        starvla_root=Path(str(assets.get("starvla_root", ""))),
    )


def _manifest_report(config: dict[str, Any]) -> dict[str, Any]:
    from owac.raw.teacher_data import load_source_manifest

    assets = _asset_set(config)
    manifest = load_source_manifest(
        assets.sources_path,
        repo_root=REPO_ROOT,
        strict=True,
        expected_splits={"train": 40, "validation": 5, "test": 5},
    )
    return {
        "sources": len(manifest.sources),
        "frames": sum(source.num_frames for source in manifest.sources),
        "splits": {
            split: len(manifest.sources_for_split(split))
            for split in ("train", "validation", "test")
        },
    }


def cmd_assets(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        assets = _asset_set(config)
        identity = assets.validate_identity()
        manifest = _manifest_report(config)
    except Exception as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    return _emit(
        {
            "status": "ready",
            "authorization": config.get("authorization", {}),
            "identity": identity,
            "manifest": manifest,
        }
    )


def _cache_for(config: dict[str, Any], assets: Any, cache_root: str | None):
    from owac.raw.teacher_data import TeacherData, load_source_manifest
    from owac.raw.temporal_distillation import TemporalCache, temporal_cache_identity

    manifest = load_source_manifest(
        assets.sources_path,
        repo_root=REPO_ROOT,
        strict=True,
        expected_splits={"train": 40, "validation": 5, "test": 5},
    )
    data = TeacherData(manifest, repo_root=REPO_ROOT)
    root = Path(cache_root) if cache_root else Path(str(config["assets"]["cache_root"]))
    # Cache contents are device-independent; bind the canonical CUDA device so
    # an authorized config using "cuda" can consume the same verified rows as
    # the production cache created with "cuda:0".
    canonical = replace(
        assets,
        device=("cuda:0" if str(assets.device).split(":")[0] == "cuda" else assets.device),
    )
    return TemporalCache(root, temporal_cache_identity(canonical, data)), data


def _path_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_matches(path: Path, *, updates: int, student: Any, arm: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return bool(
        int(payload.get("cumulative_updates", -1)) == int(updates)
        and payload.get("model_identity") == student.model_identity
        and payload.get("arm") == arm
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _host_rss_bytes() -> int | None:
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _metric_payload(record: Any) -> dict[str, Any]:
    if record is None:
        return {}
    return {
        "split": record.split,
        "step": int(record.step),
        "first1_mae7": record.step1_mae7,
        "first5_mae7": record.first5_mae7,
        "full10_mae7": record.full10_mae7,
        "groups": dict(record.groups),
        "padding_mae": record.padding_mae,
        "valid_step1": int(record.valid_step1),
        "valid_first5": int(record.valid_first5),
        "valid_full10": int(record.valid_full10),
    }


def _write_selection_sidecar(
    output_root: Path,
    trainer: Any,
    *,
    best_value: float,
    best_step: int,
    best_path: str,
    best_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checkpoint = Path(best_path)
    if not checkpoint.is_file():
        raise ValueError("selector best checkpoint is absent")
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(selected.get("cumulative_updates", -1)) != int(best_step):
        raise ValueError("selected checkpoint step differs")
    if selected.get("arm") != trainer.arm:
        raise ValueError("selected checkpoint arm differs")
    payload = {
        "schema": "owac.raw-stem-temporal.selection.v1",
        "split": "validation",
        "best_value": float(best_value),
        "best_step": int(best_step),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _path_sha256(checkpoint),
        "model_identity": selected["model_identity"],
        "arm": trainer.arm,
        "cumulative_updates": int(best_step),
        "run_cumulative_updates": int(trainer.cumulative_updates),
        "scope": getattr(trainer, "scope", getattr(trainer.student, "_scope", "unknown")),
        "run_identity": dict(trainer.run_identity),
        "metrics": dict(best_metrics or {}),
    }
    _write_json_atomic(output_root / "selection.json", payload)
    return payload


def _training_run_identity(
    config: Mapping[str, Any], bundle: Any, cache_key: str
) -> dict[str, Any]:
    return {
        "scope": "stem_plus_gate",
        "research_config_sha256": _research_config_digest(config),
        "cache_identity_key": cache_key,
        "teacher_artifact_sha256": bundle.identity["artifact_sha256"],
        "backend_sha256": bundle.identity["backend_sha256"],
        "tokenizer_sha256": bundle.identity["tokenizer_sha256"],
        "stats_sha256": bundle.identity["stats_sha256"],
        "sources_sha256": bundle.identity["sources_sha256"],
        "source_revision": bundle.identity.get("source_revision"),
    }


def _path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _checkpoint_estimate(student: Any) -> int:
    state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in student.state_dict().values()
    )
    optimizer_moments = sum(
        parameter.numel() * 4 * 2 for parameter in student.parameters() if parameter.requires_grad
    )
    metadata_rng_margin = 64 * 1024 * 1024
    return int(state_bytes + optimizer_moments + metadata_rng_margin)


def _expected_source_endpoints(source: Any) -> list[int]:
    return list(range(0, int(source.num_frames), 5))


def _cache_coverage(
    cache: Any, data: Any, *, arm: str
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict[str, Any]]:
    train_sources = list(data.manifest.sources_for_split("train"))
    if len(train_sources) != 40:
        raise ValueError(
            f"training requires exactly 40 declared train sources, got {len(train_sources)}"
        )
    missing: list[tuple[str, int]] = []
    candidates: list[tuple[str, int]] = []
    for source in train_sources:
        expected = _expected_source_endpoints(source)
        for endpoint in expected:
            if not cache.exists(source.source_demo_id, endpoint):
                missing.append((source.source_demo_id, endpoint))
        candidates.extend(
            (source.source_demo_id, endpoint) for endpoint in expected if endpoint >= 25
        )
    validation_missing: list[tuple[str, int]] = []
    if arm == "B":
        validation_sources = list(data.manifest.sources_for_split("validation"))
        if len(validation_sources) != 5:
            raise ValueError(
                f"B diagnostic validation requires exactly 5 sources, got {len(validation_sources)}"
            )
        for source in validation_sources:
            for endpoint in _expected_source_endpoints(source):
                if not cache.exists(source.source_demo_id, endpoint):
                    validation_missing.append((source.source_demo_id, endpoint))
    return (
        candidates,
        missing + validation_missing,
        {
            "train_sources": len(train_sources),
            "candidate_anchors": len(candidates),
            "missing_rows": len(missing) + len(validation_missing),
            "validation_required": arm == "B",
        },
    )


def cmd_prepare_cache(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        from owac.raw.temporal_distillation import build_teacher_bundle, prepare_temporal_cache

        assets = _asset_set(config, device=args.device)
        assets.validate_identity()
        bundle = build_teacher_bundle(assets, device=args.device)
        cache, _data = _cache_for(config, assets, args.cache_dir)
        report = prepare_temporal_cache(
            bundle,
            cache,
            source_ids=args.source_id,
            max_endpoint=args.max_endpoint,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    return _emit({"status": "prepared", "cache": str(cache.root), **report})


def _observation_from_row(bundle: Any, row: Any) -> Any:
    camera_rgb = {
        camera: ((row.teacher_rgb[index] + 1.0) / 2.0).to(
            bundle.teacher.frontend.model.parameters().__next__().device
        )
        for index, camera in enumerate(CAMERA_ORDER)
    }
    raw = row.state_raw.detach().cpu().numpy().reshape(8)
    q01 = torch.as_tensor(bundle.stats.state["q01"][:8], dtype=torch.float32).numpy()
    q99 = torch.as_tensor(bundle.stats.state["q99"][:8], dtype=torch.float32).numpy()
    deployment = ((raw - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype("float32")
    return bundle.evaluator._policy_observation(camera_rgb, deployment, TASK_INSTRUCTION)


def cmd_preflight(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        from owac.raw.temporal_distillation import (
            TemporalPacketDataset,
            build_teacher_bundle,
            build_temporal_student,
            prepare_temporal_cache,
        )

        assets = _asset_set(config, device=args.device)
        assets.validate_identity()
        bundle = build_teacher_bundle(assets, device=args.device)
        student, bridge, student_report = build_temporal_student(
            bundle,
            stem_seed=int(config.get("student", {}).get("stem_seed", 101)),
            input_mode="rgb3_control" if args.arm == "B" else "rawvla_rgb3",
        )
        cache, data = _cache_for(config, assets, args.cache_dir)
        source_id = args.source_id or next(
            source.source_demo_id for source in data.manifest.sources_for_split("train")
        )
        anchor = int(args.anchor_endpoint)
        if anchor < 25 or anchor % 5:
            raise ValueError("preflight anchor endpoint must be a multiple of 5 and >=25")
        if not cache.exists(source_id, anchor) or args.prepare:
            prepare_temporal_cache(bundle, cache, source_ids=[source_id], max_endpoint=anchor)
        dataset = TemporalPacketDataset(data=data, cache=cache, student=student, arm=args.arm)
        sample = dataset.anchor_sample(source_id, anchor)
        from owac.training.raw_temporal_distillation import (
            TemporalDistillationTrainer,
            TemporalTrainingConfig,
        )

        settings = TemporalTrainingConfig.from_mapping(config)
        activation_mode = "save_on_cpu" if args.save_on_cpu else settings.activation_memory_mode
        trainer = TemporalDistillationTrainer(
            student=student,
            bridge=bridge,
            config=settings,
            teacher_ready=False,
            source_split_ready=True,
            parity_ready=False,
            resource_ready=False,
        )
        start_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if activation_mode == "save_on_cpu"
            else contextlib.nullcontext()
        )
        token_started = time.perf_counter()
        with context:
            token_result = trainer.zero_update_gradient_diagnostic(
                sample, include_action=False, mode="feature"
            )
            student.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        token_only_seconds = time.perf_counter() - token_started
        action_started = time.perf_counter()
        if args.action_diagnostic:
            with context:
                action_result = trainer.zero_update_gradient_diagnostic(
                    sample,
                    include_action=True,
                    mode="action",
                    use_cached_pairs=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            action_only_seconds = time.perf_counter() - action_started
        else:
            action_result = None
            action_only_seconds = None
        profile_started = time.perf_counter()
        profile = _profile_preflight(student, bridge, bundle.stats, sample, args)
        profile_seconds = time.perf_counter() - profile_started
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        trainable_params = [
            parameter for parameter in student.parameters() if parameter.requires_grad
        ]
        total_params = sum(parameter.numel() for parameter in student.parameters())
        trainable_unique = sum(parameter.numel() for parameter in trainable_params)
        report = {
            "status": "preflight_ready",
            "passed": True,
            "optimizer_updates": 0,
            "teacher_artifact_sha256": bundle.identity["artifact_sha256"],
            "backend_sha256": bundle.identity["backend_sha256"],
            "sources_sha256": bundle.identity["sources_sha256"],
            "teacher_source_tree_sha256": bundle.identity.get("teacher_source_tree_sha256"),
            "research_config_sha256": _research_config_digest(config),
            "cache_identity_key": cache.identity_key,
            "protocol": "temporal-fullprefix-unroll6-stride5-v1",
            "authorization": config.get("authorization", {}),
            "updates": 0,
            "student": student_report,
            "source_id": source_id,
            "anchor_endpoint": anchor,
            "prefix_endpoints": len(sample.prefix_packets),
            "unroll_endpoints": len(sample.unroll_packets),
            "context_frames": sample.charged_context_frames,
            "token_gradient": token_result,
            "action_gradient": action_result,
            "profile": profile,
            "activation_memory_mode": activation_mode,
            "timing": {
                "token_only_seconds": token_only_seconds,
                "action_only_seconds": action_only_seconds,
                "profile_seconds": profile_seconds,
                "diagnostic_total_seconds": elapsed,
                "effective_batch": int(config.get("training", {}).get("effective_batch", 16)),
                "scope": "full_prefix_unroll6_shared_teacher_targets",
                "conservative_token_plus_action_seconds": (
                    token_only_seconds + (action_only_seconds or 0.0)
                ),
            },
            "parameters": {
                "unique_total": total_params,
                "unique_trainable": trainable_unique,
                "trainable_names": [
                    name
                    for name, parameter in student.named_parameters()
                    if parameter.requires_grad
                ],
            },
            "memory": {
                "start_allocated": start_memory,
                "peak_allocated": peak,
            },
            "resource_estimate": {
                "optimizer_updates": 0,
                "assumed_updates": int(
                    config.get("training", {}).get("max_cumulative_updates_per_arm", 5000)
                ),
                "assumed_effective_batch": int(
                    config.get("training", {}).get("effective_batch", 16)
                ),
                "conservative_seconds_per_anchor": token_only_seconds
                + (action_only_seconds or 0.0),
                "gpu_hours_formula": (
                    "conservative_token_plus_action_seconds * assumed_updates * "
                    "assumed_effective_batch / 3600; excludes validation/optimizer/caching; "
                    "not a measured D wall time"
                ),
            },
            "disk": {
                "cache_bytes": sum(
                    path.stat().st_size for path in cache.root.rglob("*") if path.is_file()
                ),
                "checkpoint_estimate_bytes": _checkpoint_estimate(student),
                "checkpoint_state_bytes": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in student.state_dict().values()
                ),
                "optimizer_moments_fp32_bytes": sum(
                    parameter.numel() * 4 * 2
                    for parameter in student.parameters()
                    if parameter.requires_grad
                ),
                "metadata_rng_margin_bytes": 64 * 1024 * 1024,
            },
        }
    except Exception as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    target = _output(args.output_dir)
    if target is not None:
        (target / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return _emit(report)


def _profile_preflight(
    student: Any,
    bridge: Any,
    stats: Any,
    sample: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.profile:
        return None
    from owac.evaluation.raw_temporal_distillation import TemporalPureRuntime

    was_training = student.training
    state = None
    with torch.no_grad():
        for packet in sample.prefix_packets:
            state = student.update_state(packet, state)
        for packet in sample.unroll_packets[:-1]:
            state = student.update_state(packet, state)
        if state is None:
            raise ValueError("profile requires a state before the final endpoint")
        state = student.detach_state(state)
    runtime = TemporalPureRuntime(student=student, bridge=bridge, stats=stats, freeze=False)
    runtime.student.eval()
    samples = []
    try:
        # One warmup is excluded from the reported percentiles.
        with torch.inference_mode():
            runtime.infer_packet(
                sample.unroll_packets[-1],
                student.state_from_snapshot(
                    state.snapshot(), device=next(student.parameters()).device
                ),
                state8=sample.target.state_raw,
                language=TASK_INSTRUCTION,
                flow_steps=10,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for _ in range(int(args.profile_repeats)):
                clone = student.state_from_snapshot(
                    state.snapshot(), device=next(student.parameters()).device
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                runtime.infer_packet(
                    sample.unroll_packets[-1],
                    clone,
                    state8=sample.target.state_raw,
                    language=TASK_INSTRUCTION,
                    flow_steps=10,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000.0)
    finally:
        runtime.student.train(was_training)
    return {
        "samples": len(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, int(0.95 * len(samples)) - 1)],
        "warmup_excluded": 1,
        "state_cloned_each_call": True,
        "scope": "full_pure_student_action_flow10",
    }


def cmd_parity(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        from owac.core.encoder_contract_n02 import VisualTokens
        from owac.evaluation.raw_distillation_trace import (
            BridgePolicyTracer,
            NativePolicyTracer,
            run_stage0_traces,
        )
        from owac.evaluation.rawvla_teacher import load_pinned_postprocessor
        from owac.raw.temporal_distillation import (
            build_teacher_bundle,
            build_temporal_student,
            prepare_temporal_cache,
        )
        from owac.raw.temporal_packets import positions_for_geometry

        assets = _asset_set(config, device=args.device)
        assets.validate_identity()
        bundle = build_teacher_bundle(assets, device=args.device)
        student, bridge, _ = build_temporal_student(bundle)
        cache, data = _cache_for(config, assets, args.cache_dir)
        source_ids = list(args.source_id or [])
        if not source_ids:
            source_ids = [
                source.source_demo_id for source in data.manifest.sources_for_split("validation")
            ][:2]
        endpoints = [int(item) for item in args.endpoint] or [0, 30]
        if any(endpoint < 0 or endpoint % 5 for endpoint in endpoints):
            raise ValueError("parity endpoints must be non-negative multiples of 5")
        native = NativePolicyTracer(
            bundle.teacher.policy,
            image_keys=(*ACTIVE_IMAGE_KEYS, INACTIVE_IMAGE_KEY),
            num_steps=10,
        )
        bridge_tracer = BridgePolicyTracer(bridge, num_steps=10)
        postprocess = load_pinned_postprocessor()
        rows = []
        for source_index, source_id in enumerate(source_ids):
            source_endpoints = endpoints if source_index == 0 else [0]
            if not all(cache.exists(source_id, endpoint) for endpoint in source_endpoints):
                prepare_temporal_cache(
                    bundle,
                    cache,
                    source_ids=[source_id],
                    max_endpoint=max(source_endpoints),
                )
            for endpoint in source_endpoints:
                row = cache.read(source_id, endpoint)
                # TEACHER post-projection tokens, not the untrained student.  The
                # bridge receives exact native aliases, teacher mask, physical 256
                # positions and the inactive right-wrist slot.
                bridge_device = next(bridge.base.parameters()).device
                per_camera = row.teacher_tokens.shape[1]
                values = torch.zeros(
                    1,
                    3 * per_camera,
                    row.teacher_tokens.shape[-1],
                    dtype=row.teacher_tokens.dtype,
                    device=bridge_device,
                )
                values[:, : 2 * per_camera] = row.teacher_tokens.reshape(
                    1, -1, row.teacher_tokens.shape[-1]
                ).to(bridge_device)
                mask = torch.zeros(1, 3 * per_camera, dtype=torch.bool, device=bridge_device)
                mask[:, : 2 * per_camera] = (
                    row.teacher_token_mask[:, None]
                    .expand(2, per_camera)
                    .reshape(1, -1)
                    .to(bridge_device)
                )
                positions = positions_for_geometry(
                    student.geometry,
                    cameras=(*ACTIVE_IMAGE_KEYS, INACTIVE_IMAGE_KEY),
                    backmap_256=True,
                )
                tokens = VisualTokens(
                    values=values,
                    mask=mask,
                    camera_ids=(*ACTIVE_IMAGE_KEYS, INACTIVE_IMAGE_KEY),
                    positions=positions.to(bridge_device),
                    scale=16.0,
                )
                observation = _observation_from_row(bundle, row)
                q01 = bundle.stats.action["q01"][:7]
                q99 = bundle.stats.action["q99"][:7]

                def client(actions: torch.Tensor, _row=row, _q01=q01, _q99=q99):
                    import numpy as np

                    physical = postprocess(
                        actions.detach().cpu().numpy(),
                        _row.state_raw.detach().cpu().numpy(),
                        {
                            "actions": {
                                "q01": np.asarray(_q01, dtype=np.float32),
                                "q99": np.asarray(_q99, dtype=np.float32),
                            }
                        },
                        model_name="PI05",
                        model_source="openpi",
                    )
                    return torch.from_numpy(np.asarray(physical, dtype=np.float32))

                report = run_stage0_traces(
                    native,
                    bridge_tracer,
                    observation=observation,
                    tokens=tokens,
                    lang_tokens=row.lang_tokens.to(bridge_device),
                    lang_masks=row.lang_masks.to(bridge_device),
                    state=row.state32.to(bridge_device),
                    noise=row.noise.to(bridge_device),
                    reset=endpoint == 0,
                    real_stats=bundle.stats,
                    client_postprocess=client,
                    camera_order=(*ACTIVE_IMAGE_KEYS, INACTIVE_IMAGE_KEY),
                    inactive_cameras=(INACTIVE_IMAGE_KEY,),
                )
                rows.append(
                    {
                        "source_id": source_id,
                        "endpoint": endpoint,
                        "reset": endpoint == 0,
                        "teacher_tokens_source": "cache.teacher_tokens",
                        **report.as_dict(),
                    }
                )
    except Exception as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    result = {
        "status": "parity_complete",
        "passed": all(row["passed"] for row in rows),
        "optimizer_updates": 0,
        "teacher_artifact_sha256": bundle.identity["artifact_sha256"],
        "backend_sha256": bundle.identity["backend_sha256"],
        "sources_sha256": bundle.identity["sources_sha256"],
        "teacher_source_tree_sha256": bundle.identity.get("teacher_source_tree_sha256"),
        "research_config_sha256": _research_config_digest(config),
        "cache_identity_key": cache.identity_key,
        "protocol": "stage0-teacher-tokens-through-bridge-v1",
        "sources": source_ids,
        "rows": rows,
        "all_passed": all(row["passed"] for row in rows),
    }
    target = _output(args.output_dir)
    if target is not None:
        (target / "parity.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return _emit(result)


def _stage_gate_ready(
    arm: str,
    *,
    parity_ready: bool,
    resource_ready: bool,
    phase1_ready: bool,
    matched_ready: bool,
) -> bool:
    if arm == "D":
        return bool(parity_ready and resource_ready and phase1_ready and matched_ready)
    return bool(parity_ready and resource_ready)


def _receipt_ready(
    path: str | None,
    *,
    artifact_sha256: str,
    backend_sha256: str,
    sources_sha256: str,
    research_config_sha256: str,
    cache_identity_key: str,
    protocol: str | None = None,
) -> bool:
    if not path or not Path(path).is_file():
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("passed") is not True or payload.get("optimizer_updates", 0) != 0:
        return False
    if payload.get("teacher_artifact_sha256") != artifact_sha256:
        return False
    if payload.get("backend_sha256") != backend_sha256:
        return False
    if payload.get("sources_sha256") != sources_sha256:
        return False
    if payload.get("research_config_sha256") != research_config_sha256:
        return False
    if payload.get("cache_identity_key") != cache_identity_key:
        return False
    if protocol is not None and payload.get("protocol") != protocol:
        return False
    return True


def _authority(config: dict[str, Any], path: str | None) -> Any:
    from owac.training.raw_temporal_distillation import TemporalTrainingAuthority

    if not path or not Path(path).is_file():
        return TemporalTrainingAuthority()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return TemporalTrainingAuthority.from_mapping({**value, "authority_path": path})


def cmd_train(args: argparse.Namespace) -> int:
    config = _config(args.config)
    authority = _authority(config, args.authority)
    if not authority.authorized:
        return _emit(
            {
                "status": "unavailable",
                "reason": "explicit actual temporal training authority receipt is absent",
                "optimizer_updates": 0,
                "authorization": config.get("authorization", {}),
            },
            exit_code=2,
        )
    if config.get("authorization", {}).get("optimizer_updates") is not True:
        return _emit(
            {
                "status": "unavailable",
                "reason": "config.authorization.optimizer_updates is false",
                "optimizer_updates": 0,
            },
            exit_code=2,
        )
    try:
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

        from owac.raw.temporal_distillation import (
            TemporalPacketDataset,
            build_teacher_bundle,
            build_temporal_student,
        )
        from owac.raw.temporal_packets import RawTemporalDataset
        from owac.training.raw_temporal_distillation import (
            TemporalAnchorSampler,
            TemporalDistillationTrainer,
            TemporalTrainingConfig,
        )

        settings = TemporalTrainingConfig.from_mapping(config)
        import random

        import numpy as np

        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        if authority.max_wall_seconds <= 0 or authority.max_output_bytes <= 0:
            return _emit(
                {
                    "status": "unavailable",
                    "reason": (
                        "authority must declare positive max_wall_seconds and max_output_bytes"
                    ),
                    "optimizer_updates": 0,
                },
                exit_code=2,
            )
        configured_cap = min(
            5000,
            settings.pilot_max_updates_per_arm,
            authority.max_updates,
            settings.max_cumulative_updates_per_arm,
        )
        requested_updates = int(args.max_updates) or configured_cap
        if requested_updates <= 0 or requested_updates > configured_cap:
            return _emit(
                {
                    "status": "unavailable",
                    "reason": "requested max-updates exceeds configured/authority cumulative cap",
                    "requested_updates": requested_updates,
                    "authorized_cap": configured_cap,
                    "optimizer_updates": 0,
                },
                exit_code=2,
            )
        if authority.scope != "raw-stem-temporal":
            return _emit(
                {
                    "status": "unavailable",
                    "reason": "authority scope differs from raw-stem-temporal",
                    "optimizer_updates": 0,
                },
                exit_code=2,
            )
        assets = _asset_set(config, device=args.device)
        identity = assets.validate_identity()
        cache, data = _cache_for(config, assets, args.cache_dir)
        research_config_sha256 = _research_config_digest(config)
        receipt_args = {
            "artifact_sha256": identity["artifact_sha256"],
            "backend_sha256": identity["backend_sha256"],
            "sources_sha256": identity["sources_sha256"],
            "research_config_sha256": research_config_sha256,
            "cache_identity_key": cache.identity_key,
        }
        parity_ready = _receipt_ready(
            args.parity_receipt,
            **receipt_args,
            protocol="stage0-teacher-tokens-through-bridge-v1",
        )
        resource_ready = _receipt_ready(
            args.resource_receipt,
            **receipt_args,
            protocol="temporal-fullprefix-unroll6-stride5-v1",
        )
        phase1_ready = _receipt_ready(
            args.phase1_receipt,
            **receipt_args,
            protocol="phase1-temporal-diagnosis-v1",
        )
        matched_ready = _receipt_ready(
            args.matched_cd_receipt,
            **receipt_args,
            protocol="matched-C-D-v1",
        )
        if not _stage_gate_ready(
            args.arm,
            parity_ready=parity_ready,
            resource_ready=resource_ready,
            phase1_ready=phase1_ready,
            matched_ready=matched_ready,
        ):
            return _emit(
                {
                    "status": "unavailable",
                    "reason": "required identity-bound readiness receipts are absent or invalid",
                    "optimizer_updates": 0,
                    "parity_ready": parity_ready,
                    "resource_ready": resource_ready,
                    "phase1_ready": phase1_ready,
                    "matched_cd_ready": matched_ready,
                    "research_config_sha256": research_config_sha256,
                    "cache_identity_key": cache.identity_key,
                },
                exit_code=2,
            )
        candidates, missing_rows, coverage = _cache_coverage(cache, data, arm=args.arm)
        if missing_rows:
            return _emit(
                {
                    "status": "unavailable",
                    "reason": "temporal cache is incomplete; refusing a training subset",
                    "coverage": coverage,
                    "missing_examples": [
                        f"{source}@{endpoint}" for source, endpoint in missing_rows[:20]
                    ],
                    "optimizer_updates": 0,
                },
                exit_code=2,
            )
        sampler = TemporalAnchorSampler(candidates, seed=101, batch_size=settings.effective_batch)
        bundle = build_teacher_bundle(assets, device=args.device)
        student, bridge, _ = build_temporal_student(
            bundle,
            stem_seed=int(config.get("student", {}).get("stem_seed", 101)),
            input_mode="rgb3_control" if args.arm == "B" else "rawvla_rgb3",
        )
        checkpoint_reserve = _checkpoint_estimate(student)
        run_started = time.perf_counter()
        trainer = TemporalDistillationTrainer(
            student=student,
            bridge=bridge,
            config=settings,
            authority=authority,
            teacher_ready=True,
            source_split_ready=True,
            parity_ready=parity_ready,
            resource_ready=resource_ready,
            phase1_diagnosis={"passed": phase1_ready},
            matched_cd_declaration={"declared": matched_ready},
            teacher=torch.nn.ModuleList([bundle.teacher.policy, bundle.teacher.frontend.model]),
            arm=args.arm,
            run_identity=_training_run_identity(config, bundle, cache.identity_key),
        )
        optimizer = AdamW(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(1, settings.warmup_updates)),
        )
        dataset = TemporalPacketDataset(data=data, cache=cache, student=student, arm=args.arm)
        validation_samples = []
        if args.arm != "B":
            validation_dataset = RawTemporalDataset(data, student, stats=bundle.stats, arm="C")
            for source in data.manifest.sources_for_split("validation"):
                validation_samples.extend(validation_dataset.source_samples(source.source_demo_id))
        from owac.evaluation.raw_temporal_distillation import (
            TemporalPureRuntime,
            evaluate_temporal_rows,
            evaluate_temporal_samples,
        )

        validation_rows = []
        validation_dataset = None
        if args.arm == "B":
            validation_dataset = TemporalPacketDataset(
                data=data, cache=cache, student=student, arm="B"
            )
            for source in data.manifest.sources_for_split("validation"):
                for endpoint in _expected_source_endpoints(source):
                    validation_rows.append(cache.read(source.source_demo_id, endpoint))

        def validate(step: int) -> Any:
            runtime = TemporalPureRuntime(
                student=student, bridge=bridge, stats=bundle.stats, freeze=False
            )
            if args.arm == "B":
                return evaluate_temporal_rows(
                    runtime,
                    validation_dataset,
                    validation_rows,
                    split="validation",
                    step=step,
                    noise_seed=settings.seed,
                )
            if not validation_samples:
                return None
            return evaluate_temporal_samples(
                runtime,
                validation_samples,
                split="validation",
                step=step,
                noise_seed=settings.seed,
            )

        output_root = Path(args.output_dir or config["assets"]["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        progress_path = output_root / "progress.jsonl"
        stop_requested = {"value": False}

        def _request_stop(signum: int, _frame: Any) -> None:
            stop_requested["value"] = True

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        best_metrics: dict[str, Any] = dict(trainer.selector_state.get("best_metrics") or {})
        best_step = trainer.selector_state.get("best_step")
        best_value = trainer.selector_state.get("best_value")
        best_path = trainer.selector_state.get("best_checkpoint_path")
        last_best_update = int(best_step) if best_step is not None else 0
        early_stopped = False
        validated_steps: set[int] = set()
        final_validation_skipped_reason = None

        def save_current(path: Path) -> None:
            trainer.run_elapsed_seconds = time.perf_counter() - run_started
            trainer.save_checkpoint(
                path,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler_state=sampler.state(),
            )

        def publish_selection() -> None:
            nonlocal best_metrics
            if best_value is None or best_step is None or not best_path:
                return
            sidecar = _write_selection_sidecar(
                output_root,
                trainer,
                best_value=float(best_value),
                best_step=int(best_step),
                best_path=str(best_path),
                best_metrics=best_metrics,
            )
            trainer.set_selector_state(
                best_value=float(sidecar["best_value"]),
                best_step=int(sidecar["best_step"]),
                best_checkpoint_path=str(sidecar["checkpoint_path"]),
                best_checkpoint_sha256=str(sidecar["checkpoint_sha256"]),
                best_metrics=best_metrics,
            )

        if args.resume:
            resume_payload = trainer.load_checkpoint(
                args.resume, optimizer=optimizer, scheduler=scheduler
            )
            if not isinstance(resume_payload.get("sampler"), Mapping):
                raise ValueError("resume checkpoint lacks sampler state")
            sampler.load_state(resume_payload["sampler"])
            best_metrics = dict(trainer.selector_state.get("best_metrics") or best_metrics)
            best_step = trainer.selector_state.get("best_step")
            best_value = trainer.selector_state.get("best_value")
            best_path = trainer.selector_state.get("best_checkpoint_path")
            last_best_update = int(best_step) if best_step is not None else 0
            sidecar_path = output_root / "selection.json"
            if sidecar_path.is_file():
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if sidecar.get("split") != "validation":
                    raise ValueError("selection sidecar split differs")
                for key, expected in (
                    ("best_step", best_step),
                    ("best_value", best_value),
                    ("checkpoint_path", best_path),
                ):
                    if sidecar.get(key) != expected:
                        raise ValueError(f"selection sidecar differs: {key}")
                if sidecar.get("arm") != trainer.arm:
                    raise ValueError("selection sidecar arm differs")
                if sidecar.get("scope") not in {
                    getattr(student, "_scope", sidecar.get("scope")),
                    "stem_plus_gate",
                }:
                    raise ValueError("selection sidecar scope differs")
                checkpoint = Path(str(sidecar["checkpoint_path"]))
                if not checkpoint.is_file() or _path_sha256(checkpoint) != sidecar.get(
                    "checkpoint_sha256"
                ):
                    raise ValueError("selection sidecar checkpoint hash differs")
                best_metrics = dict(sidecar.get("metrics") or best_metrics)
                trainer.set_selector_state(
                    best_value=float(best_value),
                    best_step=int(best_step),
                    best_checkpoint_path=str(best_path),
                    best_checkpoint_sha256=str(sidecar["checkpoint_sha256"]),
                    best_metrics=best_metrics,
                )
            elif best_path:
                checkpoint = Path(str(best_path))
                if checkpoint.is_file():
                    actual_sha = _path_sha256(checkpoint)
                    stored_sha = trainer.selector_state.get("best_checkpoint_sha256")
                    if stored_sha not in (None, actual_sha):
                        raise ValueError("predecessor best checkpoint hash differs")
                    trainer.set_selector_state(
                        best_value=float(best_value),
                        best_step=int(best_step),
                        best_checkpoint_path=str(best_path),
                        best_checkpoint_sha256=actual_sha,
                        best_metrics=best_metrics,
                    )
                else:
                    raise ValueError("predecessor best checkpoint is absent")
            run_started = time.perf_counter() - trainer.run_elapsed_seconds
            updates = trainer.cumulative_updates
            _append_jsonl(
                progress_path,
                {
                    "event": "resume",
                    "checkpoint": str(Path(args.resume)),
                    "checkpoint_sha256": _path_sha256(Path(args.resume)),
                    "cumulative_updates": updates,
                    "anchor_presentations": trainer.anchor_presentations,
                    "sampler_state": dict(resume_payload.get("sampler", {})),
                    "optimizer_state_present": resume_payload.get("optimizer") is not None,
                    "scheduler_state_present": resume_payload.get("scheduler") is not None,
                    "selector": dict(trainer.selector_state),
                },
            )
        else:
            updates = 0
            checkpoint0 = output_root / "checkpoint_0000.pt"
            if 0 in settings.checkpoint_updates:
                initial = validate(0)
                validated_steps.add(0)
                if initial is not None and initial.first5_mae7 is not None:
                    best_step = 0
                    best_value = float(initial.first5_mae7)
                    best_path = str(checkpoint0)
                    best_metrics = _metric_payload(initial)
                    trainer.set_selector_state(
                        best_value=best_value,
                        best_step=best_step,
                        best_checkpoint_path=best_path,
                        best_checkpoint_sha256=None,
                        best_metrics=best_metrics,
                    )
                save_current(checkpoint0)
                if best_path == str(checkpoint0):
                    publish_selection()
        target_updates = requested_updates

        resource_blocked = False
        while updates < target_updates:
            if stop_requested["value"]:
                resource_blocked = True
                break
            if time.perf_counter() - run_started > authority.max_wall_seconds:
                resource_blocked = True
                break
            if (
                _path_bytes(cache.root) + _path_bytes(output_root) + checkpoint_reserve
                > authority.max_output_bytes
            ):
                resource_blocked = True
                break
            batch_keys = sampler.next_batch()
            batch = [
                dataset.anchor_sample(source_id, endpoint) for source_id, endpoint in batch_keys
            ]
            update_result = trainer.run_accumulated_update(
                batch, optimizer=optimizer, scheduler=scheduler, include_action=args.arm == "D"
            )
            updates += 1
            _append_jsonl(
                progress_path,
                {
                    "event": "update",
                    "arm": args.arm,
                    "cumulative_updates": int(update_result["updates"]),
                    "anchor_presentations": int(update_result["anchors"]),
                    "sampled_source_endpoints": [
                        {"source_id": source_id, "endpoint": int(endpoint)}
                        for source_id, endpoint in batch_keys
                    ],
                    "effective_batch": int(settings.effective_batch),
                    "valid_tokens": int(update_result["token_denominator"]),
                    "valid_actions": int(update_result["action_denominator"]),
                    "feature_mse": float(update_result["feature_mse"]),
                    "feature_cosine": float(update_result["feature_cosine"]),
                    "feature_total": float(update_result["feature_total"]),
                    "gradient_summary": dict(update_result["gradient_summary"]),
                    "actual_updates": int(update_result["actual_updates"]),
                    "changed_trainable_count": int(update_result["changed_trainable_count"]),
                    "actual_changed_names": list(update_result["actual_changed_names"]),
                    "actual_max_abs_delta": float(update_result["actual_max_abs_delta"]),
                    "frozen_hashes": dict(update_result["frozen_hashes"]),
                    "learning_rate": float(update_result["learning_rate"]),
                    "elapsed_seconds": float(update_result["elapsed_seconds"]),
                    "gpu_allocated_bytes": (
                        int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None
                    ),
                    "gpu_reserved_bytes": (
                        int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else None
                    ),
                    "host_rss_bytes": _host_rss_bytes(),
                },
            )
            del batch
            if updates in settings.checkpoint_updates:
                checkpoint_path = output_root / f"checkpoint_{updates:04d}.pt"
                record = validate(updates)
                validated_steps.add(updates)
                if record is not None and record.first5_mae7 is not None:
                    candidate = (float(record.first5_mae7), int(updates))
                    historical = (
                        (float(best_value), int(best_step))
                        if best_value is not None and best_step is not None
                        else None
                    )
                    if historical is None or candidate < historical:
                        best_step = updates
                        best_value = candidate[0]
                        best_path = str(checkpoint_path)
                        best_metrics = _metric_payload(record)
                        trainer.set_selector_state(
                            best_value=best_value,
                            best_step=best_step,
                            best_checkpoint_path=best_path,
                            best_checkpoint_sha256=None,
                            best_metrics=best_metrics,
                        )
                        last_best_update = int(updates)
                    else:
                        if (
                            settings.earlystop_patience > 0
                            and updates - last_best_update >= settings.earlystop_patience
                        ):
                            early_stopped = True
                save_current(checkpoint_path)
                if best_path == str(checkpoint_path):
                    publish_selection()
                if early_stopped:
                    break
        final_path = output_root / f"checkpoint_final_{updates:06d}.pt"
        final_checkpoint = None
        final_validation = None
        can_save_final = final_path.is_file() or (
            _path_bytes(cache.root) + _path_bytes(output_root) + checkpoint_reserve
            <= authority.max_output_bytes
        )
        if (
            updates not in validated_steps
            and (validation_samples or validation_rows)
            and time.perf_counter() - run_started <= authority.max_wall_seconds
        ):
            final_validation = validate(updates)
            validated_steps.add(updates)
            if final_validation is not None and final_validation.first5_mae7 is not None:
                candidate = (float(final_validation.first5_mae7), int(updates))
                historical = (
                    (float(best_value), int(best_step))
                    if best_value is not None and best_step is not None
                    else None
                )
                if (historical is None or candidate < historical) and can_save_final:
                    best_step = updates
                    best_value = candidate[0]
                    best_path = str(final_path)
                    best_metrics = _metric_payload(final_validation)
                    trainer.set_selector_state(
                        best_value=best_value,
                        best_step=best_step,
                        best_checkpoint_path=best_path,
                        best_checkpoint_sha256=None,
                        best_metrics=best_metrics,
                    )
        elif (
            updates not in validated_steps
            and time.perf_counter() - run_started > authority.max_wall_seconds
        ):
            final_validation_skipped_reason = "wall_time_cap"

        if _checkpoint_matches(final_path, updates=updates, student=student, arm=args.arm):
            final_checkpoint = str(final_path)
        elif (
            _path_bytes(cache.root) + _path_bytes(output_root) + checkpoint_reserve
            <= authority.max_output_bytes
        ):
            save_current(final_path)
            final_checkpoint = str(final_path)
            if best_path == str(final_path):
                publish_selection()
        else:
            scheduled = output_root / f"checkpoint_{updates:04d}.pt"
            if _checkpoint_matches(scheduled, updates=updates, student=student, arm=args.arm):
                final_checkpoint = str(scheduled)
            if final_checkpoint is None:
                resource_blocked = True
        if best_path and Path(best_path).is_file():
            publish_selection()
        validation_metrics = dict(best_metrics) if best_metrics else None

    except Exception as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    # validation_metrics was captured from the best selector metric payload.
    validation_metrics = dict(best_metrics) if best_metrics else None
    return _emit(
        {
            "status": "resource_blocked" if resource_blocked else "trained",
            "updates": updates,
            "validation": validation_metrics,
            "selected_checkpoint_step": best_step,
            "selected_checkpoint_value": best_value,
            "selected_checkpoint_path": best_path,
            "selector": dict(trainer.selector_state),
            "final_checkpoint": final_checkpoint,
            "early_stopped": early_stopped,
            "resource_blocked": resource_blocked,
            "final_validation_skipped_reason": final_validation_skipped_reason,
            "checkpoint_reserve_bytes": checkpoint_reserve,
            "authorization": config.get("authorization", {}),
        },
        exit_code=2 if resource_blocked else 0,
    )


def cmd_export(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        from owac.baselines.raw_temporal_siglip import export_temporal_student_artifact
        from owac.raw.temporal_distillation import build_teacher_bundle, build_temporal_student
        from owac.training.raw_temporal_distillation import (
            TemporalDistillationTrainer,
            TemporalTrainingConfig,
        )

        if args.step != 0 and not args.checkpoint:
            raise ValueError("nonzero export step requires --checkpoint")
        assets = _asset_set(config, device=args.device)
        bundle = build_teacher_bundle(assets, device=args.device)
        student, bridge, _report = build_temporal_student(
            bundle,
            input_mode="rgb3_control" if args.arm == "B" else "rawvla_rgb3",
        )
        step = 0
        checkpoint_sha256 = None
        scope = getattr(student, "_scope", "unknown")
        if args.checkpoint:
            cache, _data = _cache_for(config, assets, None)
            trainer = TemporalDistillationTrainer(
                student=student,
                bridge=bridge,
                config=TemporalTrainingConfig.from_mapping(config),
                arm=args.arm,
                teacher=torch.nn.ModuleList([bundle.teacher.policy, bundle.teacher.frontend.model]),
                run_identity=_training_run_identity(config, bundle, cache.identity_key),
            )
            payload = trainer.load_checkpoint(args.checkpoint)
            step = int(payload["cumulative_updates"])
            if args.step not in (0, step):
                raise ValueError("requested export step differs from checkpoint cumulative step")
            scope = str(payload.get("scope", scope))
            if scope not in {getattr(student, "_scope", scope), "stem_plus_gate"}:
                raise ValueError("checkpoint export scope differs from student")
            checkpoint_sha256 = _path_sha256(Path(args.checkpoint))
        target = Path(args.output_dir or config["assets"]["output_root"]) / f"student-{step:04d}"
        manifest = export_temporal_student_artifact(
            student,
            target,
            backend_identity={
                "sha256": bundle.identity["backend_sha256"],
                "registry_artifact": bundle.identity["artifact_sha256"],
            },
            tokenizer_identity={"sha256": bundle.identity["tokenizer_sha256"]},
            normalizer_identity={"sha256": bundle.identity["stats_sha256"]},
            checkpoint_step=step,
            extra_metadata={
                "checkpoint_path": None if not args.checkpoint else str(Path(args.checkpoint)),
                "checkpoint_sha256": checkpoint_sha256,
                "scope": scope,
                "arm": args.arm,
                "cumulative_updates": step,
                "selector": dict(trainer.selector_state) if args.checkpoint else {},
                "pure_raw_deployment": args.arm != "B",
            },
        )
    except Exception as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    return _emit(
        {
            "status": "exported",
            "path": str(target),
            "checkpoint_step": step,
            "checkpoint": None if not args.checkpoint else str(Path(args.checkpoint)),
            "checkpoint_sha256": checkpoint_sha256,
            "pure_raw_deployment": args.arm != "B",
            "manifest": manifest,
        }
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _config(args.config)
    try:
        from owac.evaluation.raw_temporal_distillation import (
            TemporalPureRuntime,
            evaluate_temporal_samples,
            evaluate_temporal_visual_controls,
        )
        from owac.raw.distillation_pairs import NormalisationStats
        from owac.raw.teacher_data import TeacherData, load_source_manifest
        from owac.raw.temporal_packets import RawTemporalDataset

        if args.arm == "B":
            raise ValueError("B is a teacher-RGB diagnostic and is not a pure evaluator arm")
        assets = _pure_paths(config)
        for name, path in (
            ("checkpoint", assets.checkpoint),
            ("tokenizer", assets.tokenizer),
            ("statistics", assets.statistics),
            ("sources", assets.sources_path),
        ):
            if not Path(path).is_file():
                raise ValueError(f"{name} is absent")
        artifact = args.student_artifact
        if not artifact:
            raise ValueError("--student-artifact is required for pure evaluation")
        runtime = TemporalPureRuntime.from_artifact(
            artifact,
            checkpoint=assets.checkpoint,
            tokenizer=assets.tokenizer,
            statistics=assets.statistics,
            runtime_root=assets.starvla_root,
            device=args.device or "cpu",
        )
        manifest = load_source_manifest(
            assets.sources_path,
            repo_root=REPO_ROOT,
            strict=True,
            expected_splits={"train": 40, "validation": 5, "test": 5},
        )
        data = TeacherData(manifest, repo_root=REPO_ROOT)
        stats = NormalisationStats.from_json(assets.statistics, mode="quantile")
        dataset = RawTemporalDataset(data, runtime.student, stats=stats, arm=args.arm)
        samples = []
        for source in data.manifest.sources_for_split(args.split):
            samples.extend(dataset.source_samples(source.source_demo_id))
        record = evaluate_temporal_samples(
            runtime,
            samples,
            split=args.split,
            step=int(args.step),
            flow_steps=int(args.flow_steps),
            noise_seed=args.noise_seed,
        )
        visual_controls = (
            evaluate_temporal_visual_controls(
                runtime,
                samples,
                split=args.split,
                flow_steps=int(args.flow_steps),
                step=int(args.step),
            )
            if args.visual_controls
            else None
        )
    except Exception as exc:
        return _emit({"status": "unavailable", "reason": str(exc)}, exit_code=2)
    result = {
        "status": "evaluated",
        "split": args.split,
        "step": int(args.step),
        "metrics": {
            "first1_mae7": record.step1_mae7,
            "first5_mae7": record.first5_mae7,
            "full10_mae7": record.full10_mae7,
            "groups": dict(record.groups),
            "padding_mae": record.padding_mae,
            "valid_step1": record.valid_step1,
            "valid_first5": record.valid_first5,
            "valid_full10": record.valid_full10,
        },
        "visual_controls": visual_controls,
        "test_selection": False,
        "teacher_or_cache_used": False,
    }
    target = _output(args.output_dir)
    if target is not None:
        (target / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return _emit(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assets = sub.add_parser("assets")
    assets.add_argument("--config", required=True)
    assets.set_defaults(func=cmd_assets)

    prepare = sub.add_parser("prepare-cache")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--source-id", action="append", default=None)
    prepare.add_argument("--max-endpoint", type=int, default=None)
    prepare.add_argument("--cache-dir", default=None)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--device", default=None)
    prepare.set_defaults(func=cmd_prepare_cache)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--source-id", default=None)
    preflight.add_argument("--anchor-endpoint", type=int, default=25)
    preflight.add_argument("--arm", choices=("B", "C", "D"), default="C")
    preflight.add_argument("--cache-dir", default=None)
    preflight.add_argument("--output-dir", default=None)
    preflight.add_argument("--device", default=None)
    preflight.add_argument("--prepare", action="store_true")
    preflight.add_argument("--save-on-cpu", action="store_true")
    preflight.add_argument("--action-diagnostic", action="store_true")
    preflight.add_argument("--profile", action="store_true")
    preflight.add_argument("--profile-repeats", type=int, default=3)
    preflight.set_defaults(func=cmd_preflight)

    parity = sub.add_parser("parity")
    parity.add_argument("--config", required=True)
    parity.add_argument("--source-id", action="append", default=None)
    parity.add_argument("--endpoint", action="append", type=int, default=None)
    parity.add_argument("--cache-dir", default=None)
    parity.add_argument("--output-dir", default=None)
    parity.add_argument("--device", default=None)
    parity.set_defaults(func=cmd_parity)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--authority", default=None)
    train.add_argument("--arm", choices=("B", "C", "D"), default="C")
    train.add_argument("--cache-dir", default=None)
    train.add_argument("--output-dir", default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--max-updates", type=int, default=0)
    train.add_argument("--resume", default=None)
    train.add_argument("--parity-receipt", default=None)
    train.add_argument("--resource-receipt", default=None)
    train.add_argument("--phase1-receipt", default=None)
    train.add_argument("--matched-cd-receipt", default=None)
    train.set_defaults(func=cmd_train)

    export = sub.add_parser("export-student")
    export.add_argument("--config", required=True)
    export.add_argument("--output-dir", default=None)
    export.add_argument("--device", default=None)
    export.add_argument("--checkpoint", default=None)
    export.add_argument("--step", type=int, default=0)
    export.add_argument("--arm", choices=("B", "C", "D"), default="C")
    export.set_defaults(func=cmd_export)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--student-artifact", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--step", type=int, default=0)
    evaluate.add_argument("--flow-steps", type=int, default=10)
    evaluate.add_argument("--noise-seed", type=int, default=101)
    evaluate.add_argument("--arm", choices=("B", "C", "D"), default="C")
    evaluate.add_argument("--cache-dir", default=None)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--visual-controls", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
