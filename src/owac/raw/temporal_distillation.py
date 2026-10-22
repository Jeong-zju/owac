"""Concrete causal teacher target, cache and RAW-packet bridge for ALIGN.

This module is the data-side integration for the temporal student.  It loads the
actual selected teacher artifact and frozen backend, prepares full-prefix causal
targets from the audited RAW sources, writes strict atomic cache rows, and turns
those rows into explicit ``TemporalPacket`` inputs.  It deliberately has no
optimizer and does not run a simulator or a policy episode.

Teacher RGB is cached only when it was actually produced by the teacher's
Neural-ISP path.  The paired clean-RGB tree is never used as a B target.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.baselines.raw_stem_siglip import (
    CheckpointIdentity,
    PolicyLoadEvidence,
    RawStemInput,
    build_raw_stem_student,
)
from owac.baselines.raw_temporal_siglip import (
    TEMPORAL_HISTORY_LENGTH,
    TemporalPacket,
    TemporalRawStemStudent,
    causal_frame_ids,
    causal_padding_mask,
)
from owac.raw.teacher_data import (
    CAMERA_ORDER,
    TeacherData,
    load_pi05_normalization_stats,
    load_source_manifest,
    make_window,
)
from owac.raw.temporal_packets import (
    physical_positions_224_to_256,
    resize_raw_256_to_224,
)

__all__ = [
    "ACTIVE_IMAGE_KEYS",
    "CAMERA_ALIASES",
    "INACTIVE_IMAGE_KEY",
    "TemporalAnchorSample",
    "TemporalAssetSet",
    "TemporalCache",
    "TemporalPacketDataset",
    "TemporalTargetRow",
    "TeacherBundle",
    "build_teacher_bundle",
    "build_temporal_student",
    "causal_frame_ids",
    "causal_padding_mask",
    "native_camera_ids",
    "physical_positions_224_to_256",
    "prepare_temporal_cache",
    "resize_raw_256_to_224",
    "temporal_cache_identity",
    "validate_temporal_row",
]

ACTIVE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
INACTIVE_IMAGE_KEY = "right_wrist_0_rgb"
CAMERA_ALIASES = {"agentview": "base_0_rgb", "wrist": "left_wrist_0_rgb"}
CACHE_SCHEMA = "owac.raw-stem-temporal.cache.v2"
ROW_SCHEMA = "owac.raw-stem-temporal.row.v1"
TEACHER_ARTIFACT_SCHEMA = "owac.rawvla-teacher.frontend-artifact.v1"
TASK_INSTRUCTION = "put the cream cheese in the bowl"


class TemporalDataError(RuntimeError):
    """The concrete temporal asset/data/cache path cannot proceed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _snapshot(value: Any) -> Any:
    """Serialize tensors/dataclasses/lists/maps for a trusted-local cache."""
    if isinstance(value, Mapping) and value.get("__tensor__") is True:
        # Rows read from disk already contain encoded theta fields. Encoding
        # them again must preserve the same payload and digest.
        tensor = value.get("data")
        if not isinstance(tensor, torch.Tensor):
            raise TemporalDataError("encoded snapshot data must be a tensor")
        if list(tensor.shape) != value.get("shape") or str(tensor.dtype).replace(
            "torch.", ""
        ) != value.get("dtype"):
            raise TemporalDataError("encoded snapshot tensor metadata differs")
        return {
            "__tensor__": True,
            "dtype": value["dtype"],
            "shape": list(tensor.shape),
            "data": tensor.detach().cpu().clone(),
        }
    if torch.is_tensor(value):
        return {
            "__tensor__": True,
            "dtype": str(value.dtype).replace("torch.", ""),
            "shape": list(value.shape),
            "data": value.detach().cpu().clone(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {item.name: _snapshot(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot snapshot {type(value).__name__}")


def _move_snapshot_to(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            setattr(value, item.name, _move_snapshot_to(getattr(value, item.name), device))
        return value
    if isinstance(value, Mapping):
        return {key: _move_snapshot_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_snapshot_to(item, device) for item in value]
    return value


def _restore_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("__tensor__") is True:
            dtype = getattr(torch, str(value["dtype"]), None)
            if dtype is None:
                raise TemporalDataError(f"unknown snapshot dtype {value['dtype']!r}")
            tensor = value["data"]
            if not isinstance(tensor, torch.Tensor):
                raise TemporalDataError("snapshot tensor data is not a tensor")
            return tensor.to(dtype=dtype)
        if "__dataclass__" in value:
            module_name, _, class_name = str(value["__dataclass__"]).rpartition(".")
            try:
                import importlib

                klass = getattr(importlib.import_module(module_name), class_name)
            except Exception as exc:
                raise TemporalDataError("cannot restore cached theta class") from exc
            fields_value = value.get("fields")
            if not isinstance(fields_value, Mapping):
                raise TemporalDataError("cached theta fields are absent")
            return klass(**{key: _restore_snapshot(item) for key, item in fields_value.items()})
        return {str(key): _restore_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_restore_snapshot(item) for item in value]
    return value


def native_camera_ids() -> tuple[str, str]:
    return tuple(CAMERA_ALIASES[camera] for camera in CAMERA_ORDER)  # type: ignore[return-value]


def temporal_cache_identity(
    assets: TemporalAssetSet, data: TeacherData | None = None
) -> dict[str, Any]:
    """Canonical cache identity; no path-existence-only fields."""
    identity = assets.validate_identity()
    identity.update(
        {
            "cache_schema": CACHE_SCHEMA,
            "camera_order": list(CAMERA_ORDER),
            "native_image_keys": [*ACTIVE_IMAGE_KEYS, INACTIVE_IMAGE_KEY],
            "active_image_keys": list(ACTIVE_IMAGE_KEYS),
            "preprocess_id": "raw256_fullview_bilinear_align_corners_false_224",
            "raw_burst_frames": TEMPORAL_HISTORY_LENGTH,
            "endpoint_stride": 5,
            "instruction": TASK_INSTRUCTION,
        }
    )
    if data is not None:
        identity["splits"] = {
            split: len(data.manifest.sources_for_split(split))
            for split in ("train", "validation", "test")
        }
    return identity


@dataclass(slots=True)
class TemporalAssetSet:
    registry_path: Path
    artifact_path: Path
    sources_path: Path
    author_root: Path
    starvla_root: Path
    checkpoint: Path
    tokenizer: Path
    statistics: Path
    config_paths: tuple[Path, ...]
    repo_root: Path
    device: str = "cpu"

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, repo_root: str | Path) -> TemporalAssetSet:
        assets = config.get("assets")
        teacher = config.get("teacher_requirement")
        if not isinstance(assets, Mapping) or not isinstance(teacher, Mapping):
            raise TemporalDataError("config requires assets and teacher_requirement")
        root = Path(repo_root)
        required = {
            "registry": assets.get("teacher_registry"),
            "artifact": assets.get("teacher_frontend_checkpoint"),
            "sources": assets.get("source_manifest"),
            "author_root": teacher.get("author_root") or assets.get("author_root"),
            "starvla_root": teacher.get("starvla_root") or assets.get("starvla_root"),
            "checkpoint": assets.get("checkpoint"),
            "tokenizer": assets.get("tokenizer"),
            "statistics": assets.get("statistics"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise TemporalDataError(f"config assets missing: {missing}")
        config_paths = tuple(Path(str(item)) for item in assets.get("teacher_config_paths", ()))
        return cls(
            registry_path=Path(str(required["registry"])),
            artifact_path=Path(str(required["artifact"])),
            sources_path=Path(str(required["sources"])),
            author_root=Path(str(required["author_root"])),
            starvla_root=Path(str(required["starvla_root"])),
            checkpoint=Path(str(required["checkpoint"])),
            tokenizer=Path(str(required["tokenizer"])),
            statistics=Path(str(required["statistics"])),
            config_paths=config_paths,
            repo_root=root,
            device=str(assets.get("device", "cpu")),
        )

    def validate_identity(self) -> dict[str, Any]:
        """Structural asset checks; no file/source hashing."""
        for name, path in (
            ("registry", self.registry_path),
            ("artifact", self.artifact_path),
            ("sources", self.sources_path),
            ("checkpoint", self.checkpoint),
            ("tokenizer", self.tokenizer),
            ("statistics", self.statistics),
        ):
            if not path.is_file():
                raise TemporalDataError(f"{name} is absent: {path}")
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if registry.get("schema") != "owac.raw-teacher.acceptance.v1":
            raise TemporalDataError("teacher registry schema differs")
        if registry.get("status") != "teacher_task6_normal_development_screen_passed":
            raise TemporalDataError("teacher registry is not the task6 development screen")
        identity = registry.get("identity")
        if not isinstance(identity, Mapping):
            raise TemporalDataError("teacher registry lacks identity")
        if not self.config_paths or not self.config_paths[-1].is_file():
            raise TemporalDataError("author child config path is absent")
        if not self.author_root.is_dir() or not self.starvla_root.is_dir():
            raise TemporalDataError("teacher source roots are absent")
        return {
            "artifact_identity": dict(identity),
            "source_revision": identity.get("source"),
            "starvla_revision": identity.get("starvla_revision"),
            "author_root": str(self.author_root.resolve()),
            "starvla_root": str(self.starvla_root.resolve()),
            "config_paths": [str(path.resolve()) for path in self.config_paths],
            "device": self.device,
        }


@dataclass(slots=True)
class TeacherBundle:
    assets: TemporalAssetSet
    teacher: Any
    data: TeacherData
    stats: Any
    evaluator: Any
    identity: Mapping[str, Any]

    def close(self) -> None:
        for module in (
            getattr(self.teacher, "frontend", None),
            getattr(self.teacher, "policy", None),
        ):
            if module is not None:
                module.eval()


def build_teacher_bundle(assets: TemporalAssetSet, *, device: str | None = None) -> TeacherBundle:
    """Build the actual teacher, strict-load the selected artifact, freeze all."""
    identity = assets.validate_identity()
    from owac.training.rawvla_teacher import TeacherAssets, build_author_teacher

    target_device = device or assets.device
    teacher_assets = TeacherAssets(
        author_root=assets.author_root,
        starvla_root=assets.starvla_root,
        checkpoint=assets.checkpoint,
        tokenizer=assets.tokenizer,
        statistics=assets.statistics,
        config_paths=assets.config_paths,
    )
    teacher = build_author_teacher(teacher_assets, device=target_device)
    from owac.evaluation.rawvla_teacher import (
        TeacherOfflineEvaluator,
        load_teacher_frontend_artifact,
    )

    registry = json.loads(assets.registry_path.read_text(encoding="utf-8"))
    expected_identity = registry["identity"]
    state = load_teacher_frontend_artifact(
        assets.artifact_path, expected_identity=expected_identity
    )
    teacher.frontend.load_state_dict(state, strict=True)
    for parameter in teacher.frontend.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.policy.parameters():
        parameter.requires_grad_(False)
    teacher.frontend.eval()
    teacher.policy.eval()
    if any(parameter.requires_grad for parameter in teacher.frontend.parameters()):
        raise TemporalDataError("teacher frontend was not fully frozen")
    manifest = load_source_manifest(
        assets.sources_path,
        repo_root=assets.repo_root,
        strict=True,
        expected_splits={"train": 40, "validation": 5, "test": 5},
    )
    data = TeacherData(manifest, repo_root=assets.repo_root, cache_entries=2)
    stats = load_pi05_normalization_stats(assets.statistics)
    evaluator = TeacherOfflineEvaluator(teacher=teacher, data=data, stats=stats)
    return TeacherBundle(
        assets=assets,
        teacher=teacher,
        data=data,
        stats=stats,
        evaluator=evaluator,
        identity={
            **identity,
            "frontend_tensor_count": len(state),
        },
    )


def build_temporal_student(
    bundle: TeacherBundle,
    *,
    stem_seed: int = 101,
    input_mode: str = "rawvla_rgb3",
) -> tuple[TemporalRawStemStudent, Any, dict[str, Any]]:
    """Build inherited student and the feature bridge from the same policy."""
    from owac.backends.raw_distillation_context import _hash_visual_state
    from owac.backends.rawvla_features_n02 import (
        RawVLAFeaturePolicy,
        StaticTokenSource,
        protocol_for,
    )

    policy = bundle.teacher.policy
    visual_hashes = dict(_hash_visual_state(policy))
    audit = dict(bundle.teacher.load_audit)
    checkpoint = CheckpointIdentity(
        path=str(bundle.assets.checkpoint),
        sha256=bundle.identity["backend_sha256"],
        verified=True,
        config_sha256=bundle.identity.get("config_sha256", ""),
        tensor_sha256=visual_hashes,
    )
    evidence = PolicyLoadEvidence(
        checkpoint_sha256=bundle.identity["backend_sha256"],
        policy_module=type(policy.vlm).__module__,
        loaded_keys=int(audit.get("loaded_keys", 0)),
        model_keys=int(audit.get("model_keys", 0)),
        verified=True,
        acceptance="accepted_real_policy",
        visual_weights_source="actual_policy_checkpoint",
        active_missing=tuple(audit.get("active_missing", ())),
        unexpected=tuple(audit.get("unexpected", ())),
        shape_mismatch=tuple(audit.get("shape_mismatch", ())),
        inactive_keys=tuple(audit.get("inactive_keys", ())),
        vision_keys_loaded=tuple(visual_hashes),
        temporal_teacher=True,
    )
    # Hash the actual strict-load evidence before exporting it. The typed
    # constructor alone does not populate the audit identity.
    evidence = PolicyLoadEvidence.from_mapping(evidence.as_dict())
    base = build_raw_stem_student(
        policy,
        input_mode=input_mode,
        checkpoint=checkpoint,
        load_audit=evidence,
        stem_seed=stem_seed,
        allow_temporal_teacher=True,
    )
    bridge = RawVLAFeaturePolicy(
        policy,
        StaticTokenSource(None),
        protocol=protocol_for("libero"),
        token_dim=base.geometry.projector_out,
        tokens_per_camera=base.geometry.num_positions,
        image_keys=(*ACTIVE_IMAGE_KEYS, INACTIVE_IMAGE_KEY),
        freeze_backend=True,
    )
    student = TemporalRawStemStudent(base)
    report = {
        "stem_seed": int(stem_seed),
        "input_mode": input_mode,
        "student_model_identity": student.model_identity,
        "stem_weight_fingerprint": student.stem_weight_fingerprint(),
        "inherited_identity": student.inherited_identity(),
        "trainable": student.trainable_report(),
        "bridge_projection": getattr(bridge, "projection", None),
        "image_keys": list(bridge.image_keys),
        "load_audit_sha256": evidence.audit_sha256,
        "load_audit": evidence.as_dict(),
    }
    return student, bridge, report


@dataclass(slots=True)
class TemporalTargetRow:
    source_id: str
    endpoint: int
    frame_ids: tuple[int, ...]
    camera_order: tuple[str, ...]
    raw_frame_indices: torch.Tensor  # [2,6] int64
    raw_burst_sha256: str
    prefix_sha256: str
    teacher_tokens: torch.Tensor  # [2,N,D] float32
    teacher_token_mask: torch.Tensor  # [2] bool
    teacher_rgb: torch.Tensor  # [2,3,224,224] float32 native preprocessed
    inactive_image_keys: tuple[str, ...]
    lang_tokens: torch.Tensor  # [1,L] int64
    lang_masks: torch.Tensor  # [1,L] bool
    state32: torch.Tensor  # [1,32]
    state8: torch.Tensor  # [8] normalized teacher-training state
    state_raw: torch.Tensor  # [8] canonical robot state
    actions32: torch.Tensor  # [10,32]
    actions7: torch.Tensor  # [10,7] real expert
    future_mask: torch.Tensor  # [10] bool
    noise: torch.Tensor  # [1,10,32]
    flow_time: torch.Tensor  # [1]
    state_before: Mapping[str, Any]  # per-camera snapshot; None is explicit
    state_after: Mapping[str, Any]
    theta_before: Mapping[str, Any]
    theta_after: Mapping[str, Any]
    teacher_artifact_sha256: str
    backend_sha256: str
    tokenizer_sha256: str
    stats_sha256: str
    source_raw_sha256: str
    code_revision: str
    preprocess_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": ROW_SCHEMA,
            "source_id": self.source_id,
            "endpoint": int(self.endpoint),
            "frame_ids": list(self.frame_ids),
            "camera_order": list(self.camera_order),
            "raw_frame_indices": self.raw_frame_indices,
            "raw_burst_sha256": self.raw_burst_sha256,
            "prefix_sha256": self.prefix_sha256,
            "teacher_tokens": self.teacher_tokens,
            "teacher_token_mask": self.teacher_token_mask,
            "teacher_rgb": self.teacher_rgb,
            "inactive_image_keys": list(self.inactive_image_keys),
            "lang_tokens": self.lang_tokens,
            "lang_masks": self.lang_masks,
            "state32": self.state32,
            "state8": self.state8,
            "state_raw": self.state_raw,
            "actions32": self.actions32,
            "actions7": self.actions7,
            "future_mask": self.future_mask,
            "noise": self.noise,
            "flow_time": self.flow_time,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "theta_before": _snapshot(self.theta_before),
            "theta_after": _snapshot(self.theta_after),
            "teacher_artifact_sha256": self.teacher_artifact_sha256,
            "backend_sha256": self.backend_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "stats_sha256": self.stats_sha256,
            "source_raw_sha256": self.source_raw_sha256,
            "code_revision": self.code_revision,
            "preprocess_id": self.preprocess_id,
            "metadata": _snapshot(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TemporalTargetRow:
        if not isinstance(payload, Mapping) or payload.get("schema") != ROW_SCHEMA:
            raise TemporalDataError("temporal row schema differs")
        required = {item.name for item in fields(cls)} | {"schema"}
        if set(payload) != required:
            raise TemporalDataError("temporal row has unexpected/missing fields")
        tensors = {
            "raw_frame_indices",
            "teacher_tokens",
            "teacher_token_mask",
            "teacher_rgb",
            "lang_tokens",
            "lang_masks",
            "state32",
            "state8",
            "state_raw",
            "actions32",
            "actions7",
            "future_mask",
            "noise",
            "flow_time",
        }
        values = dict(payload)
        for name in tensors:
            if not isinstance(values[name], torch.Tensor):
                raise TemporalDataError(f"temporal row {name} is not a tensor")
        row = cls(
            source_id=str(payload["source_id"]),
            endpoint=int(payload["endpoint"]),
            frame_ids=tuple(int(item) for item in payload["frame_ids"]),
            camera_order=tuple(str(item) for item in payload["camera_order"]),
            raw_frame_indices=payload["raw_frame_indices"],
            raw_burst_sha256=str(payload["raw_burst_sha256"]),
            prefix_sha256=str(payload["prefix_sha256"]),
            teacher_tokens=payload["teacher_tokens"],
            teacher_token_mask=payload["teacher_token_mask"],
            teacher_rgb=payload["teacher_rgb"],
            inactive_image_keys=tuple(str(item) for item in payload["inactive_image_keys"]),
            lang_tokens=payload["lang_tokens"],
            lang_masks=payload["lang_masks"],
            state32=payload["state32"],
            state8=payload["state8"],
            state_raw=payload["state_raw"],
            actions32=payload["actions32"],
            actions7=payload["actions7"],
            future_mask=payload["future_mask"],
            noise=payload["noise"],
            flow_time=payload["flow_time"],
            state_before=payload["state_before"],
            state_after=payload["state_after"],
            theta_before=payload["theta_before"],
            theta_after=payload["theta_after"],
            teacher_artifact_sha256=str(payload["teacher_artifact_sha256"]),
            backend_sha256=str(payload["backend_sha256"]),
            tokenizer_sha256=str(payload["tokenizer_sha256"]),
            stats_sha256=str(payload["stats_sha256"]),
            source_raw_sha256=str(payload["source_raw_sha256"]),
            code_revision=str(payload["code_revision"]),
            preprocess_id=str(payload["preprocess_id"]),
            metadata=payload["metadata"],
        )
        validate_temporal_row(row)
        return row


def _snapshot_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_snapshot_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_snapshot_finite(item) for item in value)
    return True


def validate_temporal_row(row: TemporalTargetRow) -> None:
    if not row.source_id or row.endpoint < 0:
        raise TemporalDataError("temporal row source/endpoint is invalid")
    if tuple(row.frame_ids) != causal_frame_ids(row.endpoint):
        raise TemporalDataError("temporal row frame_ids are not the exact causal burst")
    if row.camera_order != CAMERA_ORDER:
        raise TemporalDataError("temporal row camera order differs")
    if tuple(row.raw_frame_indices.shape) != (2, TEMPORAL_HISTORY_LENGTH):
        raise TemporalDataError("raw_frame_indices must be [2,6]")
    if row.raw_frame_indices.dtype != torch.int64:
        raise TemporalDataError("raw_frame_indices must be int64")
    expected_indices = torch.tensor([row.frame_ids for _ in CAMERA_ORDER], dtype=torch.int64)
    if not torch.equal(row.raw_frame_indices, expected_indices):
        raise TemporalDataError("raw_frame_indices must equal actual causal frame ids")
    if row.teacher_tokens.ndim != 3 or row.teacher_tokens.shape[0] != 2:
        raise TemporalDataError("teacher_tokens must be [2,N,D]")
    if row.teacher_tokens.dtype != torch.float32 or not torch.isfinite(row.teacher_tokens).all():
        raise TemporalDataError("teacher_tokens must be finite float32")
    if tuple(row.teacher_token_mask.shape) != (2,) or row.teacher_token_mask.dtype != torch.bool:
        raise TemporalDataError("teacher_token_mask must be bool [2]")
    if tuple(row.teacher_rgb.shape[0:2]) != (2, 3) or tuple(row.teacher_rgb.shape[-2:]) != (
        224,
        224,
    ):
        raise TemporalDataError("teacher_rgb must be [2,3,224,224]")
    if row.teacher_rgb.dtype != torch.float32 or not torch.isfinite(row.teacher_rgb).all():
        raise TemporalDataError("teacher_rgb must be finite float32")
    if float(row.teacher_rgb.min()) < -1.0 - 1e-5 or float(row.teacher_rgb.max()) > 1.0 + 1e-5:
        raise TemporalDataError("teacher_rgb must be native preprocessed [-1,1]")
    if tuple(row.lang_tokens.shape) != tuple(row.lang_masks.shape) or row.lang_tokens.ndim != 2:
        raise TemporalDataError("lang tokens/masks must share [1,L]")
    if row.lang_tokens.dtype != torch.int64 or row.lang_masks.dtype != torch.bool:
        raise TemporalDataError("lang token/mask dtypes differ")
    # Native PI05 observations retain a singleton observation-time axis;
    # the feature bridge accepts both layouts without changing any values.
    if tuple(row.state32.shape) not in {(1, 32), (1, 1, 32)} or tuple(row.state8.shape) != (8,):
        raise TemporalDataError("state shape differs")
    if tuple(row.state_raw.shape) != (8,):
        raise TemporalDataError("state_raw must be [8]")
    if tuple(row.actions32.shape) != (10, 32) or tuple(row.actions7.shape) != (10, 7):
        raise TemporalDataError("action shapes differ")
    if tuple(row.future_mask.shape) != (10,) or row.future_mask.dtype != torch.bool:
        raise TemporalDataError("future_mask must be [10] bool")
    mask = row.future_mask.tolist()
    if not any(mask) or any((not mask[index - 1]) and mask[index] for index in range(1, 10)):
        raise TemporalDataError("future_mask must be a nonempty causal prefix")
    if tuple(row.noise.shape) != (1, 10, 32) or tuple(row.flow_time.shape) != (1,):
        raise TemporalDataError("noise/time shapes differ")
    for name in (
        "state32",
        "state8",
        "state_raw",
        "actions32",
        "actions7",
        "noise",
        "flow_time",
    ):
        value = getattr(row, name)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise TemporalDataError(f"{name} contains non-finite values")
    if float(row.flow_time.min()) < 0.0 or float(row.flow_time.max()) > 1.0:
        raise TemporalDataError("flow_time must be in [0,1]")
    for label, snapshot in (("before", row.state_before), ("after", row.state_after)):
        if not isinstance(snapshot, Mapping) or set(snapshot) != set(CAMERA_ORDER):
            raise TemporalDataError(f"teacher state {label} must have both cameras")
        for camera in CAMERA_ORDER:
            value = snapshot[camera]
            if value is not None and (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (1, 128)
                or not torch.isfinite(value).all()
            ):
                raise TemporalDataError(
                    f"teacher state {label}/{camera} must be None or finite [1,128]"
                )
            if label == "after" and value is None:
                raise TemporalDataError(f"teacher state after/{camera} cannot be None")
    if row.endpoint == 0 and any(value is not None for value in row.state_before.values()):
        raise TemporalDataError("first endpoint state_before must be explicit None")
    if row.endpoint == 0 and any(
        value is not None for value in row.theta_before["cameras"].values()
    ):
        raise TemporalDataError("first endpoint theta_before must be explicit None")
    for label, snapshot in (("before", row.theta_before), ("after", row.theta_after)):
        if not isinstance(snapshot, Mapping) or set(snapshot.get("cameras", {})) != set(
            CAMERA_ORDER
        ):
            raise TemporalDataError(f"teacher theta {label} must have both cameras")
    if not _snapshot_finite(row.theta_before) or not _snapshot_finite(row.theta_after):
        raise TemporalDataError("teacher theta snapshots contain non-finite tensors")
    if not row.inactive_image_keys:
        raise TemporalDataError("inactive native image-key metadata is required")


class TemporalCache:
    """Atomic, digest-checked cache of full-prefix teacher rows."""

    def __init__(self, root: str | Path, identity: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.identity = dict(identity)
        requested = str(self.identity.get("namespace") or "default")
        if len(requested) > 80:
            raise TemporalDataError("cache namespace is too long")
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Existing 64-hex directory is an opaque legacy namespace.
            self.identity_key = str(existing["identity_key"])
            self.identity = dict(existing.get("identity") or self.identity)
        else:
            self.identity_key = requested
            self._write_json_exclusive(
                manifest_path,
                {
                    "schema": CACHE_SCHEMA,
                    "identity": self.identity,
                    "identity_key": self.identity_key,
                },
            )

    def _write_json_exclusive(self, path: Path, value: Mapping[str, Any]) -> None:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _source_slug(source_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)

    def row_path(self, source_id: str, endpoint: int) -> Path:
        return self.root / self.identity_key / self._source_slug(source_id) / f"{endpoint:06d}.pt"

    def exists(self, source_id: str, endpoint: int) -> bool:
        return self.row_path(source_id, endpoint).is_file()

    def write(self, row: TemporalTargetRow) -> Path:
        validate_temporal_row(row)
        target = self.row_path(row.source_id, row.endpoint)
        if target.exists():
            return target
        payload = {
            "schema": CACHE_SCHEMA,
            "identity_key": self.identity_key,
            "source_id": row.source_id,
            "endpoint": row.endpoint,
            "row": row.to_payload(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        torch.save(payload, temp)
        try:
            os.link(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def read(self, source_id: str, endpoint: int) -> TemporalTargetRow:
        path = self.row_path(source_id, endpoint)
        if not path.is_file():
            raise TemporalDataError(f"temporal cache row is absent: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("schema") != CACHE_SCHEMA:
            raise TemporalDataError("temporal cache row schema differs")
        if payload.get("identity_key") != self.identity_key:
            raise TemporalDataError("temporal cache identity differs")
        if str(payload.get("source_id")) != source_id or int(payload.get("endpoint", -1)) != int(
            endpoint
        ):
            raise TemporalDataError("temporal cache row key/semantics differ")
        row = TemporalTargetRow.from_payload(payload.get("row"))
        if row.source_id != source_id or row.endpoint != endpoint:
            raise TemporalDataError("temporal cache row identity differs")
        return row

    def read_all(self, source_id: str, endpoints: Iterable[int]) -> list[TemporalTargetRow]:
        return [self.read(source_id, int(endpoint)) for endpoint in endpoints]

    def cached_endpoints(self, source_id: str) -> tuple[int, ...]:
        directory = self.root / self.identity_key / self._source_slug(source_id)
        if not directory.is_dir():
            return ()
        values = []
        for path in sorted(directory.glob("*.pt")):
            try:
                values.append(int(path.stem))
            except ValueError:
                continue
        return tuple(values)


def _theta_snapshot(theta: Any) -> Mapping[str, Any]:
    value = _snapshot(theta)
    if not isinstance(value, Mapping):
        raise TemporalDataError("teacher theta snapshot is not a mapping")
    return value


def prepare_temporal_cache(
    bundle: TeacherBundle,
    cache: TemporalCache,
    *,
    source_ids: Sequence[str] | None = None,
    max_endpoint: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare actual full-prefix teacher rows and write the strict cache."""
    sources = list(bundle.data.manifest.sources)
    if source_ids is None:
        selected = sources
    else:
        by_id = {source.source_demo_id: source for source in sources}
        missing = [source_id for source_id in source_ids if source_id not in by_id]
        if missing:
            raise TemporalDataError(f"unknown source ids: {missing}")
        selected = [by_id[source_id] for source_id in source_ids]
    report: dict[str, Any] = {"sources": {}, "rows": 0, "cache_identity_key": cache.identity_key}
    from owac.evaluation.rawvla_teacher import deterministic_action_noise

    teacher = bundle.teacher
    teacher.frontend.eval()
    teacher.policy.eval()
    for source in selected:
        if source.split not in {"train", "validation", "test"}:
            raise TemporalDataError("source split is invalid")
        episode = bundle.data.episode_raw(source.source_demo_id)
        endpoints = list(range(0, episode.num_frames, 5))
        if max_endpoint is not None:
            endpoints = [endpoint for endpoint in endpoints if endpoint <= int(max_endpoint)]
        if not endpoints:
            continue
        states: dict[str, Any] = {}
        thetas: dict[str, Any] = {}
        prefix = f"{source.source_demo_id}:{source.raw_sha256}"
        rows = 0
        frontend_device = next(teacher.frontend.parameters()).device
        for endpoint in endpoints:
            frame_ids = causal_frame_ids(endpoint)
            burst_raw: dict[str, np.ndarray] = {}
            for camera in CAMERA_ORDER:
                burst_raw[camera] = np.asarray(episode.raw[camera][list(frame_ids)], dtype=np.uint8)
            raw_burst_sha = f"{source.source_demo_id}:{endpoint}:raw_burst"
            state_before = {
                camera: None if states.get(camera) is None else states[camera].detach()
                for camera in CAMERA_ORDER
            }
            theta_before = {
                camera: None if thetas.get(camera) is None else thetas[camera]
                for camera in CAMERA_ORDER
            }
            expected_prefix = f"{prefix}:{endpoint}:prefix"
            if cache.exists(source.source_demo_id, endpoint) and not overwrite:
                cached = cache.read(source.source_demo_id, endpoint)
                states = {
                    camera: cached.state_after[camera].to(frontend_device)
                    for camera in CAMERA_ORDER
                }
                thetas = {
                    camera: _move_snapshot_to(
                        _restore_snapshot(cached.theta_after["cameras"][camera]),
                        frontend_device,
                    )
                    for camera in CAMERA_ORDER
                }
                prefix = expected_prefix
                rows += 1
                report["rows"] += 1
                continue
            if overwrite and cache.exists(source.source_demo_id, endpoint):
                cache.row_path(source.source_demo_id, endpoint).unlink()
            camera_rgb: dict[str, torch.Tensor] = {}
            state_after: dict[str, Any] = {}
            theta_after: dict[str, Any] = {}
            for camera in CAMERA_ORDER:
                raw = (
                    torch.from_numpy(burst_raw[camera]).permute(0, 3, 1, 2).float().unsqueeze(0)
                    / 255.0
                )
                raw = raw.to(frontend_device)
                with torch.no_grad():
                    output = teacher.frontend.model(
                        raw, state=states.get(camera), theta_prev=thetas.get(camera)
                    )
                states[camera] = output.state.detach()
                thetas[camera] = output.theta.detach()
                camera_rgb[camera] = output.rgb.detach()[0].float().clamp(0.0, 1.0)
                state_after[camera] = output.state.detach().cpu().clone()
                theta_after[camera] = output.theta.detach()
            # Canonical source state and teacher-training normalisation.
            window = make_window(episode, endpoint)
            normalized = window.normalized(bundle.stats)
            raw_state8 = np.asarray(window.state, dtype=np.float32).reshape(8)
            q01 = np.asarray(bundle.stats.state["q01"], dtype=np.float32)[:8]
            q99 = np.asarray(bundle.stats.state["q99"], dtype=np.float32)[:8]
            deployment_state8 = ((raw_state8 - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32)
            observation = bundle.evaluator._policy_observation(
                camera_rgb,
                deployment_state8,
                episode.source.metadata.get("language", TASK_INSTRUCTION),
            )
            with torch.no_grad():
                token_rows = []
                token_masks = []
                rgb_rows = []
                for image_key in ACTIVE_IMAGE_KEYS:
                    image = observation.images[image_key]
                    active = bool(observation.image_masks[image_key][0])
                    embedded = teacher.policy.vlm.embed_image(image).detach().clone()
                    token_rows.append(embedded)
                    token_masks.append(torch.tensor(active, dtype=torch.bool))
                    rgb_rows.append(image.detach().clone())
                teacher_tokens = torch.cat(token_rows, dim=0)
                teacher_token_mask = torch.stack(token_masks)
                teacher_rgb = torch.stack([row[0] for row in rgb_rows], dim=0)
            prefix = expected_prefix
            row = TemporalTargetRow(
                source_id=source.source_demo_id,
                endpoint=endpoint,
                frame_ids=frame_ids,
                camera_order=CAMERA_ORDER,
                raw_frame_indices=torch.tensor(
                    [frame_ids for _ in CAMERA_ORDER], dtype=torch.int64
                ),
                raw_burst_sha256=raw_burst_sha,
                prefix_sha256=prefix,
                teacher_tokens=teacher_tokens.to(torch.float32),
                teacher_token_mask=teacher_token_mask,
                teacher_rgb=teacher_rgb.to(torch.float32),
                inactive_image_keys=(INACTIVE_IMAGE_KEY,),
                lang_tokens=observation.tokenized_prompt.detach().clone(),
                lang_masks=observation.tokenized_prompt_mask.detach().clone(),
                state32=observation.state.detach().clone(),
                state8=torch.from_numpy(np.ascontiguousarray(normalized["state"])).float(),
                state_raw=torch.from_numpy(raw_state8.copy()).float(),
                actions32=torch.from_numpy(normalized["padded_actions"]).float(),
                actions7=torch.from_numpy(window.actions).float(),
                future_mask=torch.from_numpy(window.action_mask).bool(),
                noise=deterministic_action_noise(
                    seed=101, source_id=source.source_demo_id, frame=endpoint
                ).float(),
                flow_time=torch.tensor([0.5], dtype=torch.float32),
                state_before={
                    camera: None
                    if state_before[camera] is None
                    else state_before[camera].detach().cpu().clone()
                    for camera in CAMERA_ORDER
                },
                state_after={camera: state_after[camera] for camera in CAMERA_ORDER},
                theta_before={"cameras": {camera: theta_before[camera] for camera in CAMERA_ORDER}},
                theta_after={"cameras": {camera: theta_after[camera] for camera in CAMERA_ORDER}},
                teacher_artifact_sha256=bundle.identity["artifact_sha256"],
                backend_sha256=bundle.identity["backend_sha256"],
                tokenizer_sha256=bundle.identity["tokenizer_sha256"],
                stats_sha256=bundle.identity["stats_sha256"],
                source_raw_sha256=source.raw_sha256,
                code_revision=bundle.identity["source_revision"],
                preprocess_id="raw256_fullview_bilinear_align_corners_false_224",
                metadata={
                    "lighting": source.lighting,
                    "source_generator_lighting": source.metadata.get("generator_lighting"),
                    "measured_sensor_exposure": None,
                    "measured_sensor_noise_variance": None,
                    "measured_sensor_gain": None,
                    "camera_aliases": dict(CAMERA_ALIASES),
                    "inactive_image_key": INACTIVE_IMAGE_KEY,
                    "inactive_image_mask": False,
                    "initial_reset": endpoint == 0,
                    "full_prefix_endpoint": endpoint,
                    "teacher_rgb_source": "actual NeuralISP output through native observation",
                    "paired_clean_rgb_used": False,
                    "exposure": "unavailable_not_in_manifest",
                    "noise": "unavailable_not_in_manifest",
                    "gain": "unavailable_not_in_manifest",
                    "timestamps": "unavailable_not_in_manifest",
                },
            )
            validate_temporal_row(row)
            path = cache.write(row)
            report["rows"] += 1
            rows += 1
            del path
        report["sources"][source.source_demo_id] = {
            "split": source.split,
            "rows": rows,
            "endpoints": endpoints,
        }
    return report


def _packet_frame(
    values: torch.Tensor,
    *,
    camera_ids: Sequence[str],
    input_mode: str,
    units: str,
    timestamp: float | None = None,
) -> RawStemInput:
    return RawStemInput(
        values=values,
        input_mode=input_mode,
        camera_ids=tuple(camera_ids),
        units=units,
    )


class TemporalPacketDataset:
    """Load actual RAW episodes and turn cached rows into explicit packets."""

    def __init__(
        self,
        *,
        data: TeacherData,
        cache: TemporalCache,
        student: TemporalRawStemStudent,
        arm: str = "C",
    ) -> None:
        if arm not in {"B", "C", "D"}:
            raise TemporalDataError("arm must be B, C or D")
        self.data = data
        self.cache = cache
        self.student = student
        self.arm = arm

    def packet_for_endpoint(
        self, source_id: str, endpoint: int, *, row: TemporalTargetRow | None = None
    ) -> TemporalPacket:
        episode = self.data.episode_raw(source_id)
        frame_ids = causal_frame_ids(endpoint)
        timestamps = None
        if episode.timestamps is not None:
            timestamps = tuple(float(episode.timestamps[index]) for index in frame_ids)
        device = next(self.student.parameters()).device
        if self.arm == "B":
            if row is None:
                row = self.cache.read(source_id, endpoint)
            if row.endpoint != endpoint:
                raise TemporalDataError("B row endpoint differs")
            return TemporalPacket(
                frames=(
                    _packet_frame(
                        row.teacher_rgb.unsqueeze(0).to(device),
                        camera_ids=ACTIVE_IMAGE_KEYS,
                        input_mode="rgb3_control",
                        units="teacher_rgb_minus1_1",
                    ),
                ),
                source_demo_id=source_id,
                endpoint=endpoint,
                frame_ids=(endpoint,),
                padding_mask=(False,),
                reset=endpoint == 0,
                timestamps=None if timestamps is None else (timestamps[-1],),
                position_source_size=256,
                position_target_size=224,
                position_backmap=True,
                preprocess_id="teacher_actual_256_fullview_to_native_224",
            )
        frames = []
        for frame_id in frame_ids:
            raw = np.stack([episode.raw[camera][frame_id] for camera in CAMERA_ORDER], axis=0)
            resized = resize_raw_256_to_224(raw).to(device)
            frames.append(
                _packet_frame(
                    resized.unsqueeze(0),
                    camera_ids=native_camera_ids(),
                    input_mode="rawvla_rgb3",
                    units="normalized_0_1",
                )
            )
        return TemporalPacket(
            frames=tuple(frames),
            source_demo_id=source_id,
            endpoint=endpoint,
            frame_ids=frame_ids,
            padding_mask=causal_padding_mask(endpoint),
            reset=endpoint == 0,
            timestamps=timestamps,
            position_source_size=256,
            position_target_size=224,
            position_backmap=True,
            preprocess_id="raw256_fullview_bilinear_align_corners_false_224",
        )

    def anchor_sample(self, source_id: str, endpoint: int) -> TemporalAnchorSample:
        if endpoint < 25 or endpoint % 5:
            raise TemporalDataError("training anchor must be a 0/5 endpoint >=25")
        unroll_endpoints = tuple(range(endpoint - 25, endpoint + 1, 5))
        prefix_endpoints = tuple(range(0, unroll_endpoints[0], 5))
        target = self.cache.read(source_id, endpoint)
        rows = tuple(self.cache.read(source_id, value) for value in unroll_endpoints)
        packets = tuple(
            self.packet_for_endpoint(source_id, value, row=target if value == endpoint else None)
            for value in unroll_endpoints
        )
        prefix_packets = tuple(
            self.packet_for_endpoint(source_id, value) for value in prefix_endpoints
        )
        return TemporalAnchorSample(
            source_id=source_id,
            endpoint=endpoint,
            target=target,
            unroll_rows=rows,
            unroll_packets=packets,
            prefix_packets=prefix_packets,
            charged_context_frames=TEMPORAL_HISTORY_LENGTH * len(unroll_endpoints),
            charged_prefix_endpoints=len(prefix_endpoints),
        )


@dataclass(frozen=True, slots=True)
class TemporalAnchorSample:
    source_id: str
    endpoint: int
    target: TemporalTargetRow
    unroll_rows: tuple[TemporalTargetRow, ...]
    unroll_packets: tuple[TemporalPacket, ...]
    prefix_packets: tuple[TemporalPacket, ...]
    charged_context_frames: int
    charged_prefix_endpoints: int
