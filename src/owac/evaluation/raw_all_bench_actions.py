"""Offline teacher/student action comparison on LIBERO validation RAW."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from owac.raw.all_bench_distillation import (
    DOMAIN_CAMERAS,
    AllBenchDataset,
    BenchSource,
    TargetStore,
    build_all_bench_student,
    build_vision_teacher,
    load_inventory,
    teacher_embed_rgb,
)

LIBERO_CAMERAS = DOMAIN_CAMERAS["libero"]


@dataclass(frozen=True, slots=True)
class ActionWindow:
    source_id: str
    suite: str
    lighting: str
    endpoint: int


def _metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata_json.npy"].item()))


def select_libero_sources(
    groups: Sequence[BenchSource], *, max_sources: int = 20
) -> list[BenchSource]:
    selected: dict[tuple[str, str], BenchSource] = {}
    for source in sorted(groups, key=lambda item: (item.suite, item.lighting, item.group_id)):
        if source.domain != "libero" or source.split != "validation":
            continue
        selected.setdefault((source.suite, source.lighting), source)
    return list(selected.values())[: int(max_sources)]


def selected_windows(source: BenchSource, *, actions: np.ndarray) -> list[int]:
    endpoints = list(source.endpoints)
    if not endpoints:
        return []
    max_valid = max(0, int(actions.shape[0]) - 10)
    valid = [endpoint for endpoint in endpoints if endpoint <= max_valid]
    if not valid:
        return []
    candidates = [0, 25, valid[len(valid) // 2], valid[-1]]
    return sorted({int(value) for value in candidates if value in valid})


def deployment_state8(raw_state: np.ndarray, stats: Any) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(8)
    q01 = np.asarray(stats.state["q01"], dtype=np.float32)[:8]
    q99 = np.asarray(stats.state["q99"], dtype=np.float32)[:8]
    return ((raw - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32)


def _student_tokens(student: Any, packet: Any, state: Any) -> tuple[Any, Any]:
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, state = student(packet, state)
    return output.tokens, state


def _teacher_tokens(
    teacher: Any, raw: torch.Tensor, states: dict, thetas: dict, camera: str
) -> torch.Tensor:
    with torch.no_grad():
        out = teacher.frontend.model(raw, state=states.get(camera), theta_prev=thetas.get(camera))
        states[camera] = out.state.detach()
        thetas[camera] = out.theta.detach()
        return teacher_embed_rgb(
            teacher, out.rgb.detach()[0].float().clamp(0, 1).unsqueeze(0)
        ).float()


def _token_view(tokens: torch.Tensor, cameras: Sequence[str]) -> Any:
    return SimpleNamespace(
        values=tokens.reshape(1, -1, tokens.shape[-1]),
        mask=torch.ones(1, len(cameras) * tokens.shape[1], dtype=torch.bool, device=tokens.device),
        camera_ids=tuple(cameras),
        positions=None,
    )


def _build_bridge(config: Mapping[str, Any], *, device: torch.device) -> tuple[Any, Any]:
    import sys

    from owac.backends.rawvla_features_n02 import (
        build_feature_policy,
        load_checkpoint_state,
    )
    from owac.raw.distillation_pairs import NormalisationStats

    assets = config["assets"]
    root = str(assets["starvla_root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    bridge = build_feature_policy(
        "libero",
        Path(assets["tokenizer"]),
        None,
        token_dim=2048,
        tokens_per_camera=256,
        image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        freeze_backend=True,
    )
    bridge.base.eval()
    bridge.load_audited_replaced_vision(load_checkpoint_state(str(assets["checkpoint"])))
    # The action bridge never executes the visual tower/projector; release them
    # after the audited load to avoid a second 8GB visual copy.
    bridge.release_replaced_vision()
    bridge.base.to(device)
    bridge.base.eval()
    stats = NormalisationStats.from_json(Path(assets["statistics"]), mode="quantile")
    return bridge, stats


def normalize_checkpoint_state(payload: Any) -> dict[str, Any]:
    """Extract a student state dict and strip a DDP ``module.`` prefix."""
    value = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(value, dict):
        raise ValueError("checkpoint model state is not a mapping")
    return {
        (key[len("module.") :] if str(key).startswith("module.") else str(key)): tensor
        for key, tensor in value.items()
    }


def prepare_shared_inputs(
    bridge: Any,
    state8: np.ndarray,
    language: str,
    device: torch.device,
    *,
    noise_seed: int = 101,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare language/state once and one deterministic noise sample."""
    lang_tokens, lang_masks, state32 = _prepare_action_inputs(bridge, state8, language, device)
    noise = torch.randn(
        (1, 10, 32),
        generator=torch.Generator().manual_seed(int(noise_seed)),
        dtype=torch.float32,
    ).to(device)
    return lang_tokens, lang_masks, state32, noise


def _prepare_action_inputs(
    bridge: Any, state8: np.ndarray, language: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from owac.backends.rawvla_language import prepare_language_state

    return prepare_language_state(
        bridge, [language], np.asarray(state8, dtype=np.float32)[None], device=device
    )


def _action_from_prepared(
    bridge: Any,
    stats: Any,
    tokens: Any,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state32: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    from owac.evaluation.raw_distillation import denormalize_actions

    prediction = bridge.predict_action_from_visual_tokens(
        tokens, lang_tokens, lang_masks, state32, noise=noise, num_steps=10
    )
    normalized = torch.as_tensor(prediction["normalized_actions"], dtype=torch.float32)
    physical = denormalize_actions(normalized[..., :7], stats)
    return normalized.cpu().numpy(), physical.cpu().numpy()


def _window_metrics(prediction: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    """True 7D metrics with fixed per-dimension denominator (no masking shortcuts)."""
    p = np.asarray(prediction, dtype=np.float64)[:, :7]
    e = np.asarray(expert, dtype=np.float64)[:, :7]
    count = min(len(p), len(e))
    if count == 0:
        raise ValueError("action window has no valid rows")
    diff = np.abs(p[:count] - e[:count])
    result: dict[str, Any] = {}
    for horizon in (1, 5, 10):
        h = min(horizon, count)
        d = diff[:h]
        result[f"mae_first{horizon}"] = float(d.mean())
        result[f"rmse_first{horizon}"] = float(np.sqrt((d * d).mean()))
    for name, sl in (
        ("translation", slice(0, 3)),
        ("rotation", slice(3, 6)),
        ("gripper", slice(6, 7)),
    ):
        result[f"mae_{name}"] = float(diff[:, sl].mean())
        result[f"rmse_{name}"] = float(np.sqrt((diff[:, sl] ** 2).mean()))
    return result


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"windows": 0}
    result: dict[str, Any] = {"windows": len(rows)}
    for key in rows[0]:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = (
            float(np.sqrt(np.mean(values**2)))
            if key.startswith("rmse_")
            else float(np.mean(values))
        )
    return result


def _group_aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    subset = [row[key] for row in rows]
    return _aggregate_rows(subset)


def run(
    config: Mapping[str, Any],
    inventory: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cuda:0",
    max_sources: int = 20,
) -> dict[str, Any]:
    started = time.monotonic()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"output already has manifest: {manifest_path}")
    groups, _ = load_inventory(inventory)
    sources = select_libero_sources(groups, max_sources=max_sources)
    if not sources:
        raise RuntimeError("no LIBERO validation sources selected")
    plans = []
    for source in sources:
        meta = _metadata(source.path)
        with np.load(source.path, allow_pickle=False) as archive:
            actions = np.asarray(archive["actions"], dtype=np.float32)
        windows = selected_windows(source, actions=actions)
        if windows:
            plans.append((source, meta, actions, windows))
    manifest = {
        "schema": "owac.raw-all-bench.actions.v1",
        "sources": [
            {
                "source_id": s.group_id,
                "suite": s.suite,
                "lighting": s.lighting,
                "num_frames": s.num_frames,
                "path": str(s.path),
                "endpoints": windows,
            }
            for s, _meta, _actions, windows in plans
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dev = torch.device(device)
    dataset = AllBenchDataset(sources, targets=TargetStore(output_path / "_unused_targets"))
    teacher = build_vision_teacher(config, device=device)
    bridge, stats = _build_bridge(config, device=dev)
    student = build_all_bench_student(config, device=device)
    ckpt = Path(checkpoint)
    model_file = ckpt / "train.pt" if ckpt.is_dir() else ckpt
    payload = torch.load(model_file, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_step = payload.get("global_step") if isinstance(payload, dict) else None
    state_dict = normalize_checkpoint_state(payload)
    student.load_state_dict(state_dict, strict=True)
    del payload, state_dict
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    teacher.frontend.eval()
    teacher.frontend.model.eval()
    teacher.vision_tower.eval()
    teacher.projector.eval()
    for module in (teacher.frontend, teacher.vision_tower, teacher.projector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    records: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "teacher_normalized",
            "student_normalized",
            "teacher_physical",
            "student_physical",
            "expert",
            "state",
            "state_raw",
            "noise",
        )
    }
    strings: dict[str, list[str]] = {"source_id": [], "language": [], "suite": [], "lighting": []}
    endpoints: list[int] = []
    progress_path = output_path / "progress.jsonl"
    for source, meta, actions, windows in plans:
        language = str(meta.get("task_description") or meta.get("language") or "").strip()
        if not language:
            raise RuntimeError(f"source has no language prompt: {source.group_id}")
        with np.load(source.path, allow_pickle=False) as archive:
            robot_states = np.asarray(archive["robot_states"], dtype=np.float32)
        episode = dataset.cache.get(source.group_id)
        student_state = None
        t_states = {camera: None for camera in LIBERO_CAMERAS}
        t_thetas = {camera: None for camera in LIBERO_CAMERAS}
        for endpoint in source.endpoints:
            packet = dataset.packet_for_endpoint(source.group_id, endpoint, dev)
            s_tokens, student_state = _student_tokens(student, packet, student_state)
            burst = episode.burst(endpoint).to(dev)
            t_parts = [
                _teacher_tokens(teacher, burst[i : i + 1], t_states, t_thetas, camera)
                for i, camera in enumerate(LIBERO_CAMERAS)
            ]
            t_tokens = torch.cat(t_parts, dim=0)
            if endpoint not in windows:
                continue
            state_raw = np.asarray(robot_states[endpoint], dtype=np.float32)[:8]
            state8 = deployment_state8(state_raw, stats)
            lang_tokens, lang_masks, state32, noise = prepare_shared_inputs(
                bridge, state8, language, dev
            )
            teacher_view = _token_view(t_tokens, LIBERO_CAMERAS)
            t_norm, t_phys = _action_from_prepared(
                bridge, stats, teacher_view, lang_tokens, lang_masks, state32, noise
            )
            s_norm, s_phys = _action_from_prepared(
                bridge, stats, s_tokens, lang_tokens, lang_masks, state32, noise.clone()
            )
            t_repeat, _ = _action_from_prepared(
                bridge, stats, teacher_view, lang_tokens, lang_masks, state32, noise.clone()
            )
            expert = actions[endpoint : endpoint + 10]
            padding_diff = np.abs(t_norm[0, :, 7:] - s_norm[0, :, 7:])
            row = {
                "source_id": source.group_id,
                "suite": source.suite,
                "lighting": source.lighting,
                "endpoint": int(endpoint),
                "language": language,
                "teacher_expert": _window_metrics(t_phys[0], expert),
                "student_expert": _window_metrics(s_phys[0], expert),
                "teacher_student": _window_metrics(t_phys[0], s_phys[0]),
                "teacher_student_normalized_7d": _window_metrics(
                    t_norm[0, :, :7], s_norm[0, :, :7]
                ),
                "padding": {
                    "mae": float(padding_diff.mean()),
                    "rmse": float(np.sqrt((padding_diff**2).mean())),
                    "scalar_count": int(padding_diff.size),
                },
                "teacher_repeat_max_abs": float(np.max(np.abs(t_norm - t_repeat))),
            }
            records.append(row)
            arrays["teacher_normalized"].append(t_norm[0])
            arrays["student_normalized"].append(s_norm[0])
            arrays["teacher_physical"].append(t_phys[0])
            arrays["student_physical"].append(s_phys[0])
            arrays["expert"].append(expert)
            arrays["state"].append(state8)
            arrays["state_raw"].append(state_raw)
            arrays["noise"].append(noise.cpu().numpy()[0])
            strings["source_id"].append(source.group_id)
            strings["language"].append(language)
            strings["suite"].append(source.suite)
            strings["lighting"].append(source.lighting)
            endpoints.append(int(endpoint))
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(
                    json.dumps(
                        {
                            "source_id": source.group_id,
                            "endpoint": int(endpoint),
                            "suite": source.suite,
                            "lighting": source.lighting,
                            "window": len(records),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
    if not records:
        raise RuntimeError("no action windows selected")
    np.savez_compressed(
        output_path / "actions.npz",
        teacher_normalized=np.stack(arrays["teacher_normalized"]),
        student_normalized=np.stack(arrays["student_normalized"]),
        teacher_physical=np.stack(arrays["teacher_physical"]),
        student_physical=np.stack(arrays["student_physical"]),
        expert=np.stack(arrays["expert"]),
        state=np.stack(arrays["state"]),
        state_raw=np.stack(arrays["state_raw"]),
        noise=np.stack(arrays["noise"]),
        source_id=np.asarray(strings["source_id"]),
        language=np.asarray(strings["language"]),
        suite=np.asarray(strings["suite"]),
        lighting=np.asarray(strings["lighting"]),
        endpoint=np.asarray(endpoints, dtype=np.int64),
    )
    by_suite: dict[str, Any] = {}
    for suite in sorted({row["suite"] for row in records}):
        subset = [row for row in records if row["suite"] == suite]
        by_suite[suite] = {
            name: _group_aggregate(subset, name)
            for name in ("teacher_expert", "student_expert", "teacher_student")
        }
    by_lighting: dict[str, Any] = {}
    for light in sorted({row["lighting"] for row in records}):
        subset = [row for row in records if row["lighting"] == light]
        by_lighting[light] = {
            name: _group_aggregate(subset, name)
            for name in ("teacher_expert", "student_expert", "teacher_student")
        }
    report = {
        "schema": "owac.raw-all-bench.actions.v1",
        "windows": len(records),
        "sources": len(sources),
        "flow_steps": 10,
        "noise_seed": 101,
        "checkpoint_global_step": checkpoint_step,
        "control_units": (
            "7D LIBERO normalized actions; physical labels use native LIBERO "
            "action units, not meters or radians"
        ),
        "frozen": True,
        "checkpoint": str(model_file),
        "elapsed_seconds": time.monotonic() - started,
        "closed_loop": "not_run",
        "per_window": records,
        "aggregate": {
            "teacher_expert": _aggregate_rows([r["teacher_expert"] for r in records]),
            "student_expert": _aggregate_rows([r["student_expert"] for r in records]),
            "teacher_student": _aggregate_rows([r["teacher_student"] for r in records]),
            "teacher_student_normalized_7d": _aggregate_rows(
                [r["teacher_student_normalized_7d"] for r in records]
            ),
        },
        "per_suite": by_suite,
        "per_lighting": by_lighting,
        "padding_teacher_student": {
            "teacher_student_mae": float(np.mean([r["padding"]["mae"] for r in records])),
            "teacher_student_rmse": float(
                np.sqrt(np.mean([r["padding"]["rmse"] ** 2 for r in records]))
            ),
        },
    }
    report["frozen_parameters"] = {
        "teacher": sum(
            p.numel()
            for m in (teacher.frontend, teacher.vision_tower, teacher.projector)
            for p in m.parameters()
        ),
        "student": sum(p.numel() for p in student.parameters()),
        "bridge_backend": sum(p.numel() for p in bridge.base.parameters()),
    }
    (output_path / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
