"""N02 v1.2 P2 evaluation entry: real dev action error, real three-state counts, real
interventions, and a small JSON receipt.

Declared conventions (each one exists for a reason, and the reason is checkable):

* A **dev sample** is one fixed anchor of the frozen ``dev`` partition, replayed causally: the
  anchor's history rows are fed one at a time with the carried state, so the decision time sees
  its real causal history instead of a batch of unrelated rows. The chosen anchors are written
  to ``dev_samples.json`` on the first run and reused afterwards; no run re-samples silently.
* The **absolute check** uses a prefix-only prediction, chosen using prior state support.
  Its viewing rule uses relative frame zero without changing decoder source time. Calibration
  and evaluation share this exact computation at fixed weights on isolated partitions.
  A missing prefix or fitted threshold is counted as invalid.
* The **action error** uses the model's own tokens through the frozen pi0.5 feature bridge, on
  the same fixed dev samples, with a fixed noise sample and a fixed flow time.
* An **intervention** is reported only with the real ``raw_diff_norm`` it produced on the real
  RAW pixels; a no-op is reported as a no-op, never as a pass. Interventions that cannot be
  applied are written as ``not_run`` with the reason.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["SUBCOMMANDS", "main", "resolve_settings", "build_check_rule", "locked_residual_groups"]

SUBCOMMANDS = ("all", "dev", "check", "interventions")

#: Assets of the deployed cache. Every one of them is overridable from the CLI and every path
#: actually used is written into the receipt.
DEFAULT_DATA_ROOT = "/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench"
DEFAULT_STATS = {
    "robotwin": "/workspace/owac/rawvla/checkpoints/pi05_robotwin/policy_norm_stats.json",
    "libero": (
        "/workspace/owac/rawvla/checkpoints/pi05_libero_jax/assets/physical-intelligence/libero/"
        "norm_stats.json"
    ),
}
DEFAULT_TOKENIZER = (
    "/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model"
)
DEFAULT_BACKEND_CHECKPOINT = {
    "robotwin": "/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors",
    "libero": (
        "/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/"
        "model.safetensors"
    ),
}
DEFAULT_MANIFEST_DIR = "outputs/n02-v12/P2/measurement-completion"
DEFAULT_ANCHOR_DIR = "outputs/n02-full/p0-independent-audit/frozen-data"

#: Model keys the v1.2 builder understands; collected from anywhere in the config tree so a
#: nested hand-written config and a flat one behave the same.
MODEL_KEYS = (
    "camera_ids",
    "level_ids",
    "level_dims",
    "dims",
    "depths",
    "in_chans",
    "input_mode",
    "d_model",
    "alignment",
    "channel_reduction",
    "decoder_hidden",
    "n_latents",
    "n_queries",
    "out_dim",
    "measurement_channels",
    "saturation_level",
    "units",
    "min_variance",
    "max_variance",
    "n_slots",
    "slot_dim",
    "tau",
    "rounds",
    "k",
    "action_dim",
    "history",
    "query_stride",
    "max_queries_per_camera",
    "batch_cameras",
    "group_camera_queries",
    "checkpoint_decoder",
    "init_variance",
    "strict_check",
)

INTERVENTIONS = ("cue_swap", "no_information", "exposure_shift", "noise_shift", "translation_shift")


class EvalError(RuntimeError):
    """A declared failure of the evaluation entry point."""


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _floats(value: Any, limit: int = 64) -> Any:
    """A compact, JSON-safe view of a tensor/list; long runs are summarised, not dropped."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().to(torch.float64).cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        items = [
            float(item) if not isinstance(item, (list, tuple)) else _floats(item) for item in value
        ]
        if len(items) > limit:
            return {"count": len(items), "first": items[:8], "last": items[-8:]}
        return items
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else None


def _count(value: Any) -> int | None:
    """Loaded-key counts arrive either as an int or as a list, depending on the loader."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return len(value)
    except TypeError:
        return None


def _first(value: Any, limit: int) -> list:
    """The first entries of a small list; a scalar is reported as itself."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in list(value)[:limit]]
    return [str(value)]


def _round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, digits) if math.isfinite(number) else None


def _jsonable(value: Any, *, limit: int = 64) -> Any:
    """Recursively make a payload JSON-safe: tensors -> lists, non-finite floats -> None."""
    if isinstance(value, torch.Tensor):
        return _jsonable(_floats(value, limit), limit=limit)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), limit=limit)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), limit=limit)
    if isinstance(value, float):
        return _round(value) if math.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, limit=limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > limit and all(
            isinstance(item, (int, float, np.floating, np.integer)) for item in value
        ):
            return {
                "count": len(value),
                "min": _round(min(value)),
                "median": _round(float(np.median(np.asarray(value, dtype=np.float64)))),
                "max": _round(max(value)),
                "first": _jsonable(list(value)[:8], limit=limit),
            }
        return [_jsonable(item, limit=limit) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value), limit=limit)
    return str(value)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=1, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _deep_get(mapping: Any, key: str, default: Any = None) -> Any:
    """First value of ``key`` anywhere in a nested mapping (depth <= 4), else ``default``."""
    seen = 0
    stack = [mapping]
    while stack and seen < 512:
        item = stack.pop(0)
        seen += 1
        if isinstance(item, Mapping):
            if key in item:
                return item[key]
            stack.extend(value for value in item.values() if isinstance(value, Mapping))
        elif isinstance(item, (list, tuple)):
            stack.extend(value for value in item if isinstance(value, Mapping))
    return default


def _path_get(mapping: Any, dotted: str, default: Any = None) -> Any:
    """Value at an explicit dotted path (``"policy.checkpoint"``), never a wildcard search.

    The config carries two different checkpoints: the N02 model's own weights and the pinned
    pi0.5 backend's ``policy.checkpoint``. A blind search for ``"checkpoint"`` would hand the
    backend's safetensors file to ``torch.load`` as if it were the model checkpoint.
    """
    current = mapping
    for part in str(dotted).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def load_config(path: Any) -> dict:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise EvalError(f"config {candidate} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalError(f"config {candidate} must be a JSON object, got {type(payload).__name__}")
    return payload


# ---------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class EvalSettings:
    config_path: str | None = None
    config_present: bool = False
    config_sha256: str = ""
    dataset_format: str = "robotwin"
    data_root: str = DEFAULT_DATA_ROOT
    stats_path: str | None = None
    manifest_path: str | None = None
    anchor_dir: str | None = None
    anchor_pattern: str = "{platform}-{partition}-anchors.json.gz"
    output: str = "outputs/n02-v12/P2/eval"
    run_dir: str = ""
    preview_dir: str = ""
    device: str = ""
    seed: int = 1103
    checkpoint: str | None = None
    allow_legacy_weights_probe: bool = False
    calibration_samples: int = 20
    dev_samples: int = 4
    intervention_samples: int = 2
    quantile: float = 0.95
    window: int = 4
    check_label: int = 0
    figures: bool = True
    publish_status: bool = True
    interventions_enabled: bool = True
    action_check: str = "auto"
    backend_checkpoint: str | None = None
    tokenizer: str | None = None
    lang_max_len: int = 48
    representation: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    argv: list = field(default_factory=list)

    def camera_ids(self) -> tuple[str, ...]:
        declared = self.model.get("camera_ids")
        if declared:
            return tuple(str(item) for item in declared)
        return _default_camera_ids(self.dataset_format)

    @property
    def query_stride(self) -> int:
        return int(self.model.get("query_stride", 1))

    @property
    def history(self) -> int:
        return int(self.model.get("history", 8))

    def anchor_path(self, partition: str) -> str | None:
        if not self.anchor_dir:
            return None
        name = self.anchor_pattern.format(platform=self.dataset_format, partition=partition)
        candidate = Path(self.anchor_dir) / name
        return str(candidate) if candidate.is_file() else None

    def as_record(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["camera_ids"] = list(self.camera_ids())
        payload["anchor_paths"] = {
            partition: self.anchor_path(partition) for partition in ("calibration", "dev")
        }
        return payload


def _default_camera_ids(dataset_format: str) -> tuple[str, ...]:
    try:
        from owac.raw.rawvla_sequence_n02 import FORMAT_SPECS

        return tuple(FORMAT_SPECS[dataset_format].camera_ids)
    except Exception:  # noqa: BLE001 - the loader is frozen but must not be required to import
        return ("camera_0",)


def resolve_settings(args: Any) -> EvalSettings:
    config = load_config(getattr(args, "config", None))
    settings = EvalSettings(config_path=getattr(args, "config", None))
    settings.config_present = bool(config)
    settings.representation = dict(config.get("representation", {}))
    if settings.config_path and Path(settings.config_path).is_file():
        settings.config_sha256 = _sha256_file(settings.config_path)

    declared_platform = (
        getattr(args, "dataset_format", None)
        or _deep_get(config, "dataset_format")
        or _deep_get(config, "data_format")
        or _deep_get(config, "platform")
        or "robotwin"
    )
    if str(declared_platform) in ("robotwin", "libero"):
        settings.dataset_format = str(declared_platform)
    else:
        settings.notes.append(
            f"declared platform {declared_platform!r} is not a known cache format; using robotwin"
        )
        settings.dataset_format = "robotwin"

    settings.data_root = str(
        getattr(args, "data_root", None)
        or _deep_get(config, "data_root")
        or _deep_get(config, "cache_root")
        or DEFAULT_DATA_ROOT
    )
    settings.stats_path = (
        getattr(args, "stats", None)
        or _deep_get(config, "stats_path")
        or _deep_get(config, "stats")
        or DEFAULT_STATS.get(settings.dataset_format)
    )
    settings.manifest_path = getattr(args, "manifest", None) or _deep_get(config, "manifest_path")
    if settings.manifest_path is None:
        candidate = Path(DEFAULT_MANIFEST_DIR) / f"manifest-{settings.dataset_format}.json"
        settings.manifest_path = str(candidate) if candidate.is_file() else None
    settings.anchor_dir = (
        getattr(args, "anchors_dir", None) or _deep_get(config, "anchor_dir") or DEFAULT_ANCHOR_DIR
    )
    if settings.anchor_dir and not Path(settings.anchor_dir).is_dir():
        settings.notes.append(f"anchor dir {settings.anchor_dir} is absent; anchors rebuild")
        settings.anchor_dir = None
    settings.anchor_pattern = str(
        getattr(args, "anchors_pattern", None)
        or _deep_get(config, "anchor_pattern")
        or "{platform}-{partition}-anchors.json.gz"
    )

    settings.output = str(getattr(args, "output", None) or settings.output)
    settings.run_dir = _resolve_run_dir(args, settings)
    settings.preview_dir = _resolve_preview_dir(args, settings)
    settings.device = str(
        getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    settings.seed = int(getattr(args, "seed", None) or _deep_get(config, "seed") or 1103)
    settings.checkpoint = (
        getattr(args, "checkpoint", None)
        or _path_get(config, "model.checkpoint")
        or _path_get(config, "paths.checkpoint")
    )
    settings.allow_legacy_weights_probe = bool(getattr(args, "allow_legacy_weights_probe", False))
    settings.calibration_samples = int(
        getattr(args, "calibration_samples", None) or settings.calibration_samples
    )
    settings.dev_samples = int(getattr(args, "dev_samples", None) or settings.dev_samples)
    settings.intervention_samples = int(
        getattr(args, "intervention_samples", None) or settings.intervention_samples
    )
    settings.quantile = float(getattr(args, "quantile", None) or settings.quantile)
    settings.window = int(getattr(args, "window", None) or settings.window)
    settings.check_label = int(getattr(args, "check_label", None) or 0)
    settings.figures = not bool(getattr(args, "no_figures", False))
    settings.publish_status = not bool(getattr(args, "no_publish_status", False))
    settings.interventions_enabled = not bool(getattr(args, "no_interventions", False))
    settings.action_check = str(getattr(args, "action_check", None) or settings.action_check)
    settings.tokenizer = getattr(args, "tokenizer", None) or DEFAULT_TOKENIZER
    settings.lang_max_len = int(getattr(args, "lang_max_len", None) or settings.lang_max_len)
    settings.backend_checkpoint = (
        getattr(args, "backend_checkpoint", None)
        or _path_get(config, "policy.checkpoint")
        or DEFAULT_BACKEND_CHECKPOINT.get(settings.dataset_format)
    )

    model: dict[str, Any] = {}
    for key in MODEL_KEYS:
        value = _deep_get(config, key)
        if value is not None:
            model[key] = value
    model.setdefault("camera_ids", list(_default_camera_ids(settings.dataset_format)))
    model.setdefault("history", settings.history)
    model.setdefault("query_stride", settings.query_stride)
    for key in (
        "k",
        "rounds",
        "d_model",
        "n_slots",
        "slot_dim",
        "n_latents",
        "n_queries",
        "out_dim",
        "decoder_hidden",
        "dims",
        "depths",
        "input_mode",
        "measurement_channels",
        "alignment",
    ):
        override = getattr(args, key, None)
        if override is not None:
            model[key] = override
    declared_cameras = getattr(args, "camera_ids", None)
    if declared_cameras:
        model["camera_ids"] = [
            item.strip() for item in str(declared_cameras).split(",") if item.strip()
        ]
    settings.model = model
    return settings


def _resolve_run_dir(args: Any, settings: EvalSettings) -> str:
    declared = getattr(args, "run_dir", None) or os.environ.get("N02_V12_RUN_DIR")
    if declared:
        return str(declared)
    output = Path(settings.output)
    if output.name == "eval" or output.name.startswith("eval-"):
        return str(output.parent)
    return str(output)


def _resolve_preview_dir(args: Any, settings: EvalSettings) -> str:
    declared = getattr(args, "preview_dir", None)
    if declared:
        return str(declared)
    run_dir = Path(settings.run_dir)
    if run_dir != Path(settings.output):
        return str(run_dir / "preview")
    return str(Path(settings.output) / "preview")


# ---------------------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------------------


def build_source(settings: EvalSettings, partition: str, *, max_windows: int | None = None):
    from owac.frameworks.n02_full.data_v12 import V12WindowSource

    source = V12WindowSource(
        dataset_format=settings.dataset_format,
        data_root=settings.data_root,
        partition=partition,
        history=settings.history,
        stats_path=settings.stats_path,
        anchor_path=settings.anchor_path(partition),
        manifest_path=settings.manifest_path,
        seed=settings.seed,
        batch_size=1,
        max_windows=max_windows,
    )
    from owac.frameworks.n02_full.data_v12 import RepresentedWindowSource

    return RepresentedWindowSource(source, settings.representation)


def anchor_indices(source: Any, count: int) -> list[int]:
    """Evenly spaced, deterministic anchor positions inside one frozen partition."""
    total = int(source.anchor_count)
    if total <= 0:
        raise EvalError("the partition has no anchors")
    count = max(1, min(int(count), total))
    if count == 1:
        return [total // 2]
    return sorted({(index + 1) * total // (count + 1) for index in range(count)})


def sample_identity(source: Any, index: int) -> dict:
    record = source.anchor(index)
    payload = record.as_dict() if hasattr(record, "as_dict") else dict(record)
    payload["anchor_index"] = int(index)
    return payload


def fixed_sample_identities(
    source: Any, count: int, *, path: Path, partition: str, frozen: list[dict] | None = None
) -> tuple[list[int], dict]:
    """Reuse the frozen dev sample list when it is present; otherwise freeze it now."""
    receipt: dict[str, Any] = {"partition": partition, "source": "frozen_receipt"}
    if frozen:
        indices = [int(item["anchor_index"]) for item in frozen]
        identity = [dict(item) for item in frozen]
        receipt["file"] = str(path)
        if path.is_file():
            receipt["sha256"] = _sha256_file(path)
        return indices, {"indices": indices, "identities": identity, **receipt}

    indices = anchor_indices(source, count)
    identity = [sample_identity(source, index) for index in indices]
    payload = {
        "partition": partition,
        "created_at": _utc_now(),
        "count": len(indices),
        "identities": identity,
    }
    write_json(path, payload)
    return indices, {
        "indices": indices,
        "identities": identity,
        "file": str(path),
        "sha256": _sha256_file(path),
        "source": "frozen_now",
    }


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------


def build_model(settings: EvalSettings, *, device: str):
    from owac.frameworks.n02_full.model_v12 import build_n02_v12_model, configure_v12_runtime

    configure_v12_runtime(load_config(settings.config_path), device)

    torch.manual_seed(settings.seed)
    model = build_n02_v12_model(dict(settings.model))
    model.to(torch.device(device))
    model.eval()
    return model


def load_checkpoint_into(
    model: torch.nn.Module, path: Any, *, allow_legacy_weights_probe=False
) -> dict:
    receipt: dict[str, Any] = {"path": str(path) if path else None, "loaded": False}
    if not path:
        receipt["reason"] = "no checkpoint given: the weights are the freshly built ones"
        return receipt
    candidate = Path(path)
    if not candidate.is_file():
        raise EvalError(f"requested checkpoint is missing: {candidate}")
    if candidate.suffix not in (".pt", ".pth", ".bin", ".ckpt"):
        raise EvalError(f"{candidate} is not a torch checkpoint")
    try:
        payload = torch.load(str(candidate), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a requested file must never fall back to random
        raise EvalError(f"cannot load requested checkpoint {candidate}: {exc}") from exc
    metadata = payload
    from owac.frameworks.n02_full.sequence_v12 import EXECUTION_REVISION

    saved_revision = metadata.get("execution_revision")
    if saved_revision and saved_revision != EXECUTION_REVISION and not allow_legacy_weights_probe:
        raise EvalError(
            "legacy checkpoint requires --allow-legacy-weights-probe; policy inputs changed"
        )
    state = payload.get("model", payload.get("state_dict", payload))
    step = metadata.get(
        "step", metadata.get("global_step", state.get("step", state.get("global_step")))
    )
    state = {
        key: value
        for key, value in state.items()
        if key not in ("step", "global_step", "updates", "update")
    }
    model.load_state_dict(state, strict=True)
    receipt.update(
        loaded=True,
        sha256=_sha256_file(candidate),
        training_step_recorded_in_file=step,
        execution_revision=metadata.get("execution_revision", "legacy_p2"),
        runtime_execution_revision=EXECUTION_REVISION,
        weights_only_probe=bool(saved_revision != EXECUTION_REVISION),
        tensors_in_file=len(state),
        loaded_keys=len(state),
        missing_keys=[],
        n_missing_keys=0,
        unexpected_keys=[],
        n_unexpected_keys=0,
    )
    return receipt


def resolve_model_step(checkpoint_step: Any, declared_step: Any = None) -> Any:
    """The model step a figure labels.

    A step recorded in the checkpoint file wins over the CLI's ``--model-step``. With neither,
    the caller gets ``None`` and the label stays ``unknown`` instead of inventing a number.
    """
    for candidate in (checkpoint_step, declared_step):
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return int(candidate)
    return None


def parameter_count(model: torch.nn.Module) -> dict:
    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    total = sum(item.numel() for item in model.parameters())
    return {"total": int(total), "trainable": int(trainable)}


# ---------------------------------------------------------------------------------------
# one decision time
# ---------------------------------------------------------------------------------------


def _declared_measurement_ids(observation: Any, rows: Sequence[int]) -> tuple:
    """One ``(episode, camera, frame)`` triple per camera of every selected row.

    The frozen inference accepts exactly this shape for one decision time; the measurement set of
    a whole window is deliberately not handed over, because declared ids name the measurements of
    one decision time, not of a batch.
    """
    episode_ids = tuple(observation.episode_ids)
    identifiers: list[tuple[str, str, str]] = []
    for position, row in enumerate(rows):
        episode = str(episode_ids[position]) if position < len(episode_ids) else ""
        for camera_id, frame in observation.frames.items():
            frame_ids = tuple(getattr(frame, "frame_ids", ()) or ())
            frame_id = frame_ids[position] if position < len(frame_ids) else str(row)
            identifiers.append((episode, str(camera_id), str(frame_id)))
    return tuple(identifiers)


def inference_input(
    batch: Any,
    *,
    frame_label: int | None = None,
    observation: Any = None,
    rows: Sequence[int] | None = None,
):
    """The declared input of one decision time; ``frame_label`` fixes the window label."""
    from owac.frameworks.n02_full.inference_v12 import V12InferenceInput

    observation = batch.observation if observation is None else observation
    frame_index = None
    if rows is not None:
        source = np.asarray(batch.frame_index).reshape(-1)
        selected = [int(row) for row in rows]
        frame_index = torch.tensor([int(source[row]) for row in selected], dtype=torch.long)
    return V12InferenceInput(
        observation=observation,
        declared_prior=batch.declared_prior,
        frame_index=frame_index,
        measurement_ids=(None if rows is None else _declared_measurement_ids(observation, rows)),
    )


def row_observation(observation: Any, row: int) -> Any:
    """One causal row of a served window batch as its own observation bundle."""
    frames: dict[str, Any] = {}
    for camera_id, frame in observation.frames.items():
        updates: dict[str, Any] = {}
        for name in (
            "values",
            "valid_mask",
            "bad_mask",
            "missing_mask",
            "saturated_mask",
            "new_measurement",
        ):
            value = getattr(frame, name, None)
            if value is None:
                continue
            tensor = value if torch.is_tensor(value) else np.asarray(value)
            if tensor.ndim >= 1 and int(tensor.shape[0]) > row:
                updates[name] = tensor[row : row + 1]
        for name in ("frame_ids", "episode_ids"):
            value = getattr(frame, name, None)
            if value is not None and len(value) > row:
                updates[name] = (value[row],)
        frames[str(camera_id)] = dataclasses.replace(frame, **updates)

    context = observation.context
    proprio = getattr(context, "proprio", None)
    if proprio is not None:
        tensor = proprio if torch.is_tensor(proprio) else np.asarray(proprio)
        if tensor.ndim >= 2 and int(tensor.shape[0]) > row:
            context = dataclasses.replace(context, proprio=tensor[row : row + 1])
    episode_ids = tuple(observation.episode_ids) if observation.episode_ids else ()
    new_mask = getattr(observation, "new_time_mask", None)
    return dataclasses.replace(
        observation,
        frames=frames,
        context=context,
        episode_ids=(episode_ids[row],) if len(episode_ids) > row else episode_ids,
        anchor_ids=(
            (observation.anchor_ids[row],)
            if getattr(observation, "anchor_ids", None) and len(observation.anchor_ids) > row
            else getattr(observation, "anchor_ids", None)
        ),
        new_time_mask=(
            None
            if new_mask is None
            else (new_mask[row : row + 1] if int(np.asarray(new_mask).size) > row else new_mask)
        ),
    )


def replay_window(
    model: torch.nn.Module,
    batch: Any,
    *,
    device: str,
    check_label: int,
    check: bool = True,
) -> dict:
    """Replay distinct causal acquisitions through the same executor as training."""
    from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

    replay = replay_inputs(model, window_inputs(batch))
    replay.update(
        check_backend=replay["rows"][-1],
        check_input=replay["inputs"][-1],
        check_state=replay["pre_states"][-1],
        row_checks=[output.check for output in replay["rows"]] if check else [],
    )
    return replay


# ---------------------------------------------------------------------------------------
# absolute check plumbing
# ---------------------------------------------------------------------------------------


def build_check_rule(settings: EvalSettings):
    from owac.frameworks.n02_full.checks_v12 import v12_viewing_rule

    return v12_viewing_rule(settings.query_stride, settings.camera_ids(), settings.window)


def locked_residual_groups(inference: Any, inp: Any, state: Any, q: torch.Tensor) -> dict:
    """Return the core's prefix-only residuals; q from the current image is not used."""
    from owac.frameworks.n02_full.inference_v12 import _frame_index_tensor

    prepared = inference._prepare_state(inp, state)
    batch = int(prepared.slot_hidden.shape[0])
    queries = inference.build_queries(inp, batch=batch)
    frame_index = _frame_index_tensor(inp, observation=inp.observation, batch=batch)
    with torch.no_grad():
        causal = inference.causal_prediction(prepared, queries, frame_index)
    if causal is None:
        raise EvalError("no distinct causal prefix: absolute check is uncalibrated")
    chosen = causal["chosen"]
    index = chosen.view(-1, 1, 1)
    residuals, valid = causal["residuals"], causal["valid"]
    supported = torch.gather(residuals, 1, index.expand(-1, 1, residuals.shape[-1])).squeeze(1)
    supported_valid = torch.gather(valid, 1, index.expand(-1, 1, valid.shape[-1])).squeeze(1)
    groups = inference._residual_views(supported, supported_valid, queries, frame_index)
    return {
        "groups": groups,
        "frame_index": frame_index.tolist(),
        "measurement_ids": [item for row in queries.ids for item in row],
        "labels": [
            [(view.scale, int(view.frame_index), view.region_id) for view in group]
            for group in groups
        ],
        "per_camera_query_count": list(queries.per_camera_query_count),
        "coverage": list(queries.coverage),
        "chosen": chosen.tolist(),
        "locked_residual_mean_abs": float(supported.abs().mean()),
        "prediction_version": "causal_pre_locked",
    }


def check_record(result: Any) -> dict:
    payload = result.as_dict() if hasattr(result, "as_dict") else {"status": str(result)}
    return {
        "status": str(payload.get("status")),
        "valid": bool(payload.get("valid", False)),
        "calibration_version": str(payload.get("calibration_version", "")),
        "viewing_rule_hash": str(payload.get("viewing_rule_hash", "")),
        "n_sequences": int(payload.get("n_sequences", 0) or 0),
        "statistics": {
            str(key): _round(value) for key, value in dict(payload.get("statistics") or {}).items()
        },
        "explanation": str(payload.get("explanation", "")),
    }


# ---------------------------------------------------------------------------------------
# calibration pass
# ---------------------------------------------------------------------------------------


def calibration_pass(model: torch.nn.Module, settings: EvalSettings, *, device: str) -> dict:
    from owac.frameworks.n02_full.checks import sequence_statistic
    from owac.frameworks.n02_full.checks_v12 import calibrate_windowed

    source = build_source(settings, "calibration", max_windows=None)
    indices = anchor_indices(source, settings.calibration_samples)
    rule = build_check_rule(settings)
    groups: list[Any] = []
    calibration_ids: list[str] = []
    per_anchor: list[dict[str, Any]] = []
    started = time.time()
    for index in indices:
        batch = source.window(index, index + 1)
        replay = replay_window(
            model, batch, device=device, check_label=settings.check_label, check=True
        )
        locked = locked_residual_groups(
            model.inference, replay["check_input"], replay["check_state"], replay["check_backend"].q
        )
        groups.extend(locked["groups"])
        calibration_ids.extend(locked["measurement_ids"])
        per_anchor.append(
            {
                "anchor_index": int(index),
                "identity": sample_identity(source, index),
                "rows": int(np.asarray(batch.frame_index).size),
                "frame_index": [int(item) for item in np.asarray(batch.frame_index).tolist()],
                "view_labels": locked["labels"][0],
                "locked_residual_mean_abs": _round(locked["locked_residual_mean_abs"]),
                "check_status": str(replay["check_backend"].check.status),
                "check_valid": bool(replay["check_backend"].check.valid),
            }
        )
    frozen = calibrate_windowed(
        groups,
        partition="calibration",
        rule=rule,
        quantile=settings.quantile,
        calibration_ids=calibration_ids,
        test_ids=[],
    )
    statistics = [float(sequence_statistic(group, rule=rule)) for group in groups]
    finite = [item for item in statistics if math.isfinite(item)]
    return {
        "status": "calibrated",
        "fitted_on": (
            "the causal prefix prediction residuals of the anchor row, "
            "using actual source time and a separate relative diagnostic axis; the anchors' "
            "earlier causal rows were replayed with their real frame indices before it"
        ),
        "source_identity": source.identity(),
        "anchor_indices": [int(item) for item in indices],
        "anchors": per_anchor,
        "n_sequences": len(groups),
        "n_anchors": len(indices),
        "calibration_ids": len(set(calibration_ids)),
        "statistics": {
            "finite": len(finite),
            "min": _round(min(finite)) if finite else None,
            "median": _round(float(np.median(finite))) if finite else None,
            "max": _round(max(finite)) if finite else None,
        },
        "calibration": frozen.as_dict(),
        "rule": rule.as_dict(),
        "rule_hash": rule.rule_hash(),
        "elapsed_seconds": round(time.time() - started, 2),
        "_calibration": frozen,
    }


# ---------------------------------------------------------------------------------------
# dev pass: three-state counts + diagnostics
# ---------------------------------------------------------------------------------------


def dev_pass(
    model: torch.nn.Module, settings: EvalSettings, *, device: str, calibration: Any
) -> dict:
    from owac.frameworks.n02_full.checks_v12 import summarise_results

    source = build_source(settings, "dev", max_windows=None)
    sample_path = Path(settings.output) / "dev_samples.json"
    frozen = None
    if sample_path.is_file():
        try:
            frozen = json.loads(sample_path.read_text(encoding="utf-8")).get("identities")
        except ValueError:
            frozen = None
    indices, identity = fixed_sample_identities(
        source, settings.dev_samples, path=sample_path, partition="dev", frozen=frozen
    )

    model.inference.calibrator = calibration
    records: list[dict[str, Any]] = []
    anchor_checks: list[Any] = []
    row_checks: list[Any] = []
    started = time.time()
    for index in indices:
        batch = source.window(index, index + 1)
        replay = replay_window(model, batch, device=device, check_label=settings.check_label)
        rows = replay["rows"]
        anchor = rows[-1]
        check_output = replay["check_backend"]
        anchor_checks.append(check_output.check)
        row_checks.extend(replay["row_checks"])
        targets = source.targets(index, index + 1)
        records.append(
            {
                "anchor_index": int(index),
                "identity": sample_identity(source, index),
                "rows": len(rows),
                "frame_index": [int(item) for item in np.asarray(batch.frame_index).tolist()],
                "measurement_ids": [list(item) for item in batch.measurement_ids][:8],
                "n_measurement_ids": int(len(batch.measurement_ids)),
                "energies": _round(anchor.energies[0]),
                "log_p0": _round(anchor.log_p0[0]),
                "support": _round(anchor.q[0]),
                "support_entropy": _round(
                    float(
                        -(anchor.q[0].clamp_min(1e-12) * anchor.q[0].clamp_min(1e-12).log()).sum()
                    )
                ),
                "accepted_rounds": _floats(anchor.diagnostics.get("accepted_rounds")),
                "round_energy_before": _floats([item["energy_before"] for item in anchor.rounds]),
                "round_energy_after": _floats([item["energy_after"] for item in anchor.rounds]),
                "round_accepted": [
                    bool(item.item()) if torch.is_tensor(item) else bool(item)
                    for item in [rounds["accepted"] for rounds in anchor.rounds]
                ],
                "check": check_record(check_output.check),
                "check_real_frame_label": check_record(anchor.check),
                "locked_residual_mean_abs": _round(
                    anchor.diagnostics.get("locked_residual_mean_abs")
                ),
                "fit_residual_mean_abs": _round(anchor.diagnostics.get("fit_residual_mean_abs")),
                "target_action_dim": int(targets["action_dim"]),
                "target_action_horizon": int(targets["action_horizon"]),
                "target_action_offset": int(targets["action_offset"]),
            }
        )
    payload = {
        "status": "ran",
        "samples": identity,
        "counting_contract": (
            "primary counts are the anchor row's own check under the declared constant window "
            "label; rows whose check produced no usable verdict are counted as invalid and are "
            "excluded from every status count - never rendered as 'None' and never as zero"
        ),
        "check_summary": summarise_results(anchor_checks),
        "check_summary_all_rows": summarise_results(row_checks),
        "invalid_reasons": sorted(
            {
                str(item.statistics.get("reason"))
                for item in [*anchor_checks, *row_checks]
                if not bool(getattr(item, "valid", False))
            }
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "records": records,
    }
    return payload


# ---------------------------------------------------------------------------------------
# action pass (real pi0.5 flow-matching error on the same fixed dev samples)
# ---------------------------------------------------------------------------------------


def _runtime_assets(settings: EvalSettings) -> dict:
    from owac.backends.rawvla_features_n02 import check_runtime_assets

    return check_runtime_assets(
        checkpoint=settings.backend_checkpoint,
        tokenizer=settings.tokenizer,
        data_root=settings.data_root,
    )


def official_prompts(settings: EvalSettings, identities: Sequence[Mapping]) -> list[str]:
    from owac.frameworks.n02_full.data_v12 import read_task_prompts

    return read_task_prompts(
        [str(item["relative_path"]) for item in identities], settings.data_root
    )


def _language_batch(policy: Any, texts: Sequence[str], proprio: Any, settings: EvalSettings):
    """The pinned runtime's own prompt format, tokenised with the pinned sentencepiece model."""
    from owac.backends.rawvla_language import POLICY_INPUT_CONTRACT, prepare_language_state

    try:
        tokens, masks, _ = prepare_language_state(policy, texts, proprio, device="cpu")
    except ValueError as error:
        raise EvalError(str(error)) from error
    return (
        tokens,
        masks,
        {
            "source": "official_policy_tokenizer",
            "text": list(texts),
            "policy_input_contract": POLICY_INPUT_CONTRACT,
        },
    )


def load_backend(settings: EvalSettings, *, device: str) -> dict:
    """Build the real pi0.5 feature bridge once; every action-consuming pass shares it."""
    from owac.backends.rawvla_features_n02 import (
        StaticTokenSource,
        build_feature_policy,
        load_checkpoint_state,
        protocol_for,
    )
    from owac.core.encoder_contract_n02 import VisualTokens

    if settings.action_check == "skip":
        return {"status": "not_run", "reason": "--action-check skip"}
    assets = _runtime_assets(settings)
    if not assets.get("real_backend_available") or not assets.get("tokenizer_exists"):
        return {
            "status": "not_run",
            "reason": "the external pi0.5 runtime or the pinned tokenizer is unavailable",
            "assets": assets,
        }
    if not assets.get("checkpoint_exists"):
        return {
            "status": "not_run",
            "reason": f"backend checkpoint {settings.backend_checkpoint} is missing",
            "assets": assets,
        }
    camera_ids = settings.camera_ids()
    tokens_per_camera = int(settings.model.get("n_queries", 256))
    policy = build_feature_policy(
        settings.dataset_format,
        Path(settings.tokenizer),
        StaticTokenSource(
            VisualTokens(
                values=torch.zeros((1, len(camera_ids) * tokens_per_camera, 2048)),
                mask=torch.ones((1, len(camera_ids) * tokens_per_camera), dtype=torch.bool),
                camera_ids=camera_ids,
                positions=torch.zeros((1, len(camera_ids) * tokens_per_camera, 2)),
            )
        ),
        token_dim=2048,
        tokens_per_camera=tokens_per_camera,
        image_keys=camera_ids,
    )
    audit = policy.load_audited_replaced_vision(load_checkpoint_state(settings.backend_checkpoint))
    policy.base.to(device)
    for module in policy.bridge_modules():
        module.to(device)
    return {
        "status": "ready",
        "policy": policy,
        "camera_ids": camera_ids,
        "tokens_per_camera": tokens_per_camera,
        "protocol": protocol_for(settings.dataset_format),
        "audit": {
            "loaded_keys": _count(audit.get("loaded_keys")),
            "unexpected": _first(audit.get("unexpected"), 8),
            "active_missing": _first(audit.get("active_missing"), 8),
        },
        "assets": assets,
    }


def action_pass(
    model: torch.nn.Module,
    settings: EvalSettings,
    *,
    device: str,
    dev_records: Sequence[Mapping[str, Any]],
    backend: Mapping[str, Any],
) -> dict:
    from owac.core.encoder_contract_n02 import VisualTokens

    if backend.get("status") != "ready":
        return {
            "status": "not_run",
            "reason": backend.get("reason", "backend unavailable"),
            "assets": backend.get("assets"),
        }
    policy = backend["policy"]
    camera_ids = backend["camera_ids"]
    protocol = backend["protocol"]

    source = build_source(settings, "dev", max_windows=None)

    records: list[dict[str, Any]] = []
    losses: list[float] = []
    started = time.time()
    for item in dev_records:
        index = int(item["anchor_index"])
        batch = source.window(index, index + 1)
        replay = replay_window(
            model, batch, device=device, check_label=settings.check_label, check=False
        )
        anchor = replay["rows"][-1]
        tokens = anchor.tokens[-1:].detach().clone()
        visual = VisualTokens(
            values=tokens,
            mask=torch.ones(tokens.shape[:2], dtype=torch.bool),
            camera_ids=camera_ids,
            positions=torch.zeros(tokens.shape[:2] + (2,)),
        )
        targets = source.targets(index, index + 1)
        actions = policy.prepare_control_targets(
            torch.as_tensor(targets["actions"], dtype=torch.float32)
        )
        action_mask = torch.as_tensor(targets["action_mask"]).reshape(-1)[: actions.shape[1]]
        action_mask = action_mask.unsqueeze(0).expand(actions.shape[0], -1)
        proprio = _anchor_proprio(batch, policy)
        texts = official_prompts(settings, [item["identity"]])
        lang_tokens, lang_masks, lang_info = _language_batch(
            policy, texts, _anchor_proprio(batch, policy, pad=False), settings
        )
        lang_tokens = lang_tokens.to(device)
        lang_masks = lang_masks.to(device)
        state = torch.as_tensor(proprio, dtype=torch.float32, device=device)
        generator = torch.Generator().manual_seed(settings.seed)
        noise = torch.randn(actions.shape, generator=generator, dtype=torch.float32).to(device)
        time_tensor = torch.full((actions.shape[0],), 0.5, dtype=torch.float32, device=device)
        out = policy.forward_from_visual_tokens(
            visual,
            lang_tokens,
            lang_masks,
            state,
            actions.to(device),
            noise=noise,
            time=time_tensor,
            action_mask=action_mask.to(device),
            return_debug=True,
        )
        loss = float(out["action_loss"].detach())
        element = float(
            torch.nn.functional.mse_loss(
                out["v_t"].detach(), out["u_t"].detach(), reduction="none"
            ).mean()
        )
        losses.append(loss)
        records.append(
            {
                "anchor_index": index,
                "identity": item["identity"],
                "flow_matching_action_error": _round(loss),
                "elementwise_mse_check": _round(element),
                "tokens_shape": [int(size) for size in tokens.shape],
                "actions_shape": [int(size) for size in actions.shape],
                "action_mask_true": int(action_mask.sum()),
                "control_dim": int(protocol.control_dim),
                "action_horizon": int(protocol.action_horizon),
                "language": lang_info,
                "backend_run_stats": out.get("run_stats", {}),
            }
        )
    payload = {
        "status": "ran",
        "backend_checkpoint": settings.backend_checkpoint,
        "tokenizer": settings.tokenizer,
        "audit": dict(backend.get("audit") or {}),
        "protocol": protocol.as_dict(),
        "conditions": {
            "noise": (
                f"torch.Generator(seed={settings.seed}) normal sample, reused for every sample"
            ),
            "flow_time": 0.5,
            "samples": "the same fixed dev anchors as the check",
        },
        "loss": {
            "mean": _round(float(np.mean(losses))),
            "median": _round(float(np.median(losses))),
            "values": _round(losses),
        },
        "records": records,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    return payload


def _anchor_proprio(batch: Any, policy: Any, *, pad: bool = True) -> np.ndarray:
    """The anchor row's declared proprio, padded to the action head's state width when asked."""
    context = batch.observation.context
    proprio = getattr(context, "proprio", None)
    if proprio is None:
        values = np.zeros((1, int(policy._max_state_dim())), dtype=np.float32)
    else:
        values = np.asarray(proprio, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
    row = values[-1:]
    if not pad:
        return row
    width = int(policy._max_state_dim())
    if row.shape[-1] >= width:
        return row[:, :width]
    return np.concatenate(
        [row, np.zeros((row.shape[0], width - row.shape[-1]), np.float32)], axis=1
    )


# ---------------------------------------------------------------------------------------
# interventions
# ---------------------------------------------------------------------------------------


def _cue_positions(batch: Any) -> tuple[list[int], list[int]]:
    shapes = []
    for frame in batch.observation.frames.values():
        tensor = np.asarray(frame.values)
        shapes.append((int(tensor.shape[-2]), int(tensor.shape[-1])))
    height = min(item[0] for item in shapes)
    width = min(item[1] for item in shapes)
    half_row, half_col = height // 4, width // 4
    box_a = [height // 8, height // 8 + half_row, width // 8, width // 8 + half_col]
    box_b = [height // 2, height // 2 + half_row, width // 2, width // 2 + half_col]
    return box_a, box_b


def intervention_pass(
    model: torch.nn.Module,
    settings: EvalSettings,
    *,
    device: str,
    dev_records: Sequence[Mapping[str, Any]],
) -> dict:
    from owac.frameworks.n02_full import interventions_v12 as inter

    if not settings.interventions_enabled:
        return {"status": "not_run", "reason": "--no-interventions"}
    source = build_source(settings, "dev", max_windows=None)
    indices = [int(item["anchor_index"]) for item in dev_records][
        : max(1, settings.intervention_samples)
    ]
    plan = {
        "cue_swap": lambda current: inter.cue_swap(
            current, eps=1.0, positions=_cue_positions(current)
        ),
        "no_information": lambda current: inter.no_information(
            current, seed=settings.seed, region="full", mode="matched_noise"
        ),
        "exposure_shift": lambda current: inter.exposure_shift(current, gain=2.0),
        "noise_shift": lambda current: inter.noise_shift(
            current, sigma_scale=1.0, seed=settings.seed
        ),
        "translation_shift": lambda current: inter.translation_shift(current, pixels=(8, -10)),
    }
    report: dict[str, Any] = {
        "status": "ran",
        "anchors": indices,
        "kinds": {},
        "strength": {
            "cue_swap": {"eps": 1.0},
            "no_information": {"mode": "matched_noise", "region": "full"},
            "exposure_shift": {"gain": 2.0},
            "noise_shift": {"sigma_scale": 1.0},
            "translation_shift": {"pixels": [8, -10]},
        },
    }
    started = time.time()
    for name in INTERVENTIONS:
        entries: list[dict[str, Any]] = []
        transform = plan[name]
        for index in indices:
            raw_batch = source.raw_window(index, index + 1)
            batch = source.represent(raw_batch)
            baseline = replay_window(
                model, batch, device=device, check_label=settings.check_label, check=True
            )
            try:
                changed, record = transform(raw_batch)
                changed = source.represent(changed)
            except Exception as exc:  # noqa: BLE001 - a refused intervention is reported
                entries.append(
                    {
                        "anchor_index": int(index),
                        "applied": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            intervened = replay_window(
                model, changed, device=device, check_label=settings.check_label, check=True
            )
            base_anchor = baseline["rows"][-1]
            new_anchor = intervened["rows"][-1]
            base_check = baseline["check_backend"].check
            new_check = intervened["check_backend"].check
            support_delta = float((new_anchor.q - base_anchor.q).abs().sum())
            energy_delta = float((new_anchor.energies - base_anchor.energies).mean())
            entries.append(
                {
                    "anchor_index": int(index),
                    "identity": sample_identity(source, index),
                    "applied": not bool(record.get("no_effect")),
                    "raw_diff_norm": _round(record.get("raw_diff_norm")),
                    "per_camera_diff_norm": {
                        str(key): _round(value)
                        for key, value in dict(record.get("per_camera_diff_norm") or {}).items()
                    },
                    "pixels_changed": int(record.get("pixels_changed", 0)),
                    "valid_cells_removed": int(record.get("valid_cells_removed", 0)),
                    "applied_to_history": bool(record.get("applied_to_history")),
                    "positions": _jsonable(record.get("positions")),
                    "energy_delta_mean": _round(energy_delta),
                    "support_delta_l1": _round(support_delta),
                    "baseline_support": _round(base_anchor.q[0]),
                    "intervened_support": _round(new_anchor.q[0]),
                    "baseline_check": check_record(base_check),
                    "intervened_check": check_record(new_check),
                    "status_changed": bool(
                        str(base_check.status) != str(new_check.status)
                        or bool(base_check.valid) != bool(new_check.valid)
                    ),
                }
            )
        diffs = [item["raw_diff_norm"] for item in entries if item.get("raw_diff_norm") is not None]
        applied = [item for item in entries if item.get("applied")]
        report["kinds"][name] = {
            "attempted": len(entries),
            "applied": len(applied),
            "no_effect": len(entries) - len(applied) - sum(1 for i in entries if "reason" in i),
            "failed": sum(1 for item in entries if "reason" in item),
            "raw_diff_norm": _jsonable(diffs),
            "status": (
                "ran"
                if applied
                else ("not_run" if all("reason" in item for item in entries) else "no_effect")
            ),
            "entries": entries,
        }
    report["elapsed_seconds"] = round(time.time() - started, 2)
    return report


# ---------------------------------------------------------------------------------------
# figures / status publishing
# ---------------------------------------------------------------------------------------


def figures_pass(
    model: torch.nn.Module,
    settings: EvalSettings,
    *,
    device: str,
    backend: Mapping[str, Any],
    model_step: Any = None,
) -> dict:
    if not settings.figures:
        return {"status": "not_run", "reason": "--no-figures"}
    try:
        from owac.evaluation.n02_full import v12_viz
    except Exception as exc:  # noqa: BLE001 - a missing viz module must not stop the numbers
        return {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}"}
    try:
        # a failed figure names its own traceback instead of hiding it in a log
        return v12_viz.render_run(
            model,
            settings,
            device=device,
            backend=backend,
            model_step=model_step,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1200:],
        }


def publish_status(settings: EvalSettings, *, preview_files: Any, report: dict) -> dict:
    if not settings.publish_status:
        return {"status": "skipped", "reason": "--no-publish-status"}
    try:
        from owac.runtime.n02_v12_status import update_status
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    run_dir = Path(settings.run_dir)
    if not run_dir.is_dir():
        return {"status": "skipped", "reason": f"run dir {run_dir} does not exist"}
    passes = report.get("passes") or {}
    action = ((passes.get("dev_action_check") or {}).get("loss")) or {}
    summary = report.get("summary") or {}
    interventions = {
        name: payload.get("raw_diff_norm")
        for name, payload in (summary.get("interventions") or {}).items()
    }
    update_status(
        run_dir,
        run_id=run_dir.name,
        preview_files=preview_files,
        dev_action_error=action.get("mean"),
        eval_report=str(Path(settings.output) / "eval_v12.json"),
        eval_state=report.get("status"),
        eval_check_counts=summary.get("check_counts"),
        eval_interventions=interventions,
        activity="评测：dev 动作误差/三态检查/干预已更新",
    )
    return {"status": "published", "run_dir": str(run_dir), "preview_files": len(preview_files)}


# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------


BULKY_KEYS = ("records", "entries", "anchors", "per_anchor", "identities")


def _strip_private(payload: Any) -> Any:
    """Drop private keys and the bulky per-sample lists that live in their own files."""
    if isinstance(payload, Mapping):
        return {
            key: _strip_private(item)
            for key, item in payload.items()
            if not str(key).startswith("_") and str(key) not in BULKY_KEYS
        }
    if isinstance(payload, (list, tuple)):
        return [_strip_private(item) for item in payload]
    return payload


def build_report(settings: EvalSettings, *, status: str, passes: dict, extra: dict) -> dict:
    report = {
        "module": "owac.evaluation.n02_full.v12",
        "module_file": str(Path(__file__).resolve()),
        "stage": "P2",
        "status": status,
        "created_at": _utc_now(),
        "command": list(settings.argv or ()),
        "dataset_format": settings.dataset_format,
        "data_root": settings.data_root,
        "stats_path": settings.stats_path,
        "manifest_path": settings.manifest_path,
        "anchors": {
            "dir": settings.anchor_dir,
            "calibration": settings.anchor_path("calibration"),
            "dev": settings.anchor_path("dev"),
        },
        "config": {
            "path": settings.config_path,
            "present": settings.config_present,
            "sha256": settings.config_sha256,
        },
        "checkpoint": passes.get("checkpoint"),
        "model": passes.get("model"),
        "passes": {key: _strip_private(value) for key, value in passes.items()},
        "notes": list(settings.notes),
        "not_run": [],
        "failures": [],
    }
    report.update(extra)
    report["summary"] = summarise_report(report)
    return report


def _model_record(settings: EvalSettings, model: torch.nn.Module, device: str) -> dict:
    readout = getattr(model.inference, "readout", None)
    return {
        "camera_ids": list(settings.camera_ids()),
        "history": settings.history,
        "query_stride": settings.query_stride,
        "k": int(model.inference.k),
        "rounds": int(model.inference.rounds),
        "d_model": int(model.inference.d_model),
        "n_slots": int(model.inference.n_slots),
        "slot_dim": int(model.inference.slot_dim),
        "n_candidates": int(model.inference.k),
        "readout": (readout.config_summary() if hasattr(readout, "config_summary") else None),
        "parameters": parameter_count(model),
        "device": device,
        "settings": _jsonable(settings.model),
        "calibrator": str(type(model.inference.calibrator).__name__),
        "strict_check": bool(model.inference.strict_check),
    }


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m owac.evaluation.n02_full.v12",
        description=(
            "N02 v1.2 evaluation: real dev flow-matching action error, real three-state "
            "absolute-check counts against a threshold frozen on the independent calibration "
            "partition, and real RAW interventions with their measured raw_diff_norm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=SUBCOMMANDS,
        help="all | dev (action error + check) | check (three-state counts) | interventions",
    )
    parser.add_argument("--config", default="configs/encoders/n02_v12/raw_n02_robotwin.json")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint to evaluate")
    parser.add_argument(
        "--allow-legacy-weights-probe",
        action="store_true",
        help="explicitly label a cross-revision weights-only diagnostic",
    )
    parser.add_argument("--output", default="outputs/n02-v12/P2/eval")
    parser.add_argument("--run-dir", default=None, help="monitor run dir (status.json/preview)")
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--dataset-format", default=None, choices=["robotwin", "libero"])
    parser.add_argument("--camera-ids", default=None, help="comma-separated camera ids")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stats", default=None, help="loader normalisation statistics")
    parser.add_argument("--manifest", default=None, help="verified partition manifest receipt")
    parser.add_argument("--anchors-dir", default=None)
    parser.add_argument("--anchors-pattern", default=None)
    parser.add_argument("--calibration-samples", type=int, default=20, help="calibration anchors")
    parser.add_argument("--dev-samples", type=int, default=4, help="fixed dev anchors")
    parser.add_argument("--intervention-samples", type=int, default=2)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--window", type=int, default=4, help="viewing-rule window")
    parser.add_argument("--check-label", type=int, default=0, help="declared window frame label")
    parser.add_argument("--lang-max-len", type=int, default=48)
    parser.add_argument("--backend-checkpoint", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--action-check", default="auto", choices=["auto", "skip"])
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-interventions", action="store_true")
    parser.add_argument("--no-publish-status", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--d-model", dest="d_model", type=int, default=None)
    parser.add_argument("--n-slots", dest="n_slots", type=int, default=None)
    parser.add_argument("--slot-dim", dest="slot_dim", type=int, default=None)
    parser.add_argument("--n-queries", dest="n_queries", type=int, default=None)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--query-stride", dest="query_stride", type=int, default=None)
    parser.add_argument(
        "--model-step",
        type=int,
        default=None,
        help=(
            "model step to label the figures with when the checkpoint file records none; a step "
            "recorded in the file wins"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = resolve_settings(args)
    settings.argv = ["python", "-m", "owac.evaluation.n02_full.v12", *(argv or [])]
    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    passes: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    notes = settings.notes

    try:
        source = build_source(settings, "calibration", max_windows=1)
    except Exception as exc:  # noqa: BLE001 - a missing dataset is a declared "not run"
        extra["not_run"] = [
            {
                "what": "everything",
                "reason": f"the real cache/partition is unavailable: {type(exc).__name__}: {exc}",
            }
        ]
        report = build_report(settings, status="not_run", passes=passes, extra=extra)
        write_json(output / "eval_v12.json", report)
        print(json.dumps({"status": report["status"], "reason": extra["not_run"][0]["reason"]}))
        return 2
    passes["data_identity"] = source.identity()

    model = build_model(settings, device=settings.device)
    passes["model"] = _model_record(settings, model, settings.device)
    passes["checkpoint"] = load_checkpoint_into(
        model, settings.checkpoint, allow_legacy_weights_probe=settings.allow_legacy_weights_probe
    )
    model_step = resolve_model_step(
        (passes.get("checkpoint") or {}).get("training_step_recorded_in_file"),
        args.model_step,
    )
    calibration = None
    backend: dict[str, Any] = {"status": "not_run", "reason": "backend not needed"}
    if args.command in ("all", "dev"):
        try:
            backend = load_backend(settings, device=settings.device)
        except Exception as exc:  # noqa: BLE001 - an absent backend is a declared "not run"
            backend = {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}"}
        passes["backend"] = {key: value for key, value in backend.items() if key != "policy"}

    if args.command in ("all", "dev", "check"):
        try:
            calibration_pass_result = calibration_pass(model, settings, device=settings.device)
            calibration = calibration_pass_result.pop("_calibration")
            passes["calibration"] = calibration_pass_result
            write_json(output / "calibration.json", calibration_pass_result)
        except Exception as exc:  # noqa: BLE001 - an unfitted check must not stop the run
            calibration = None
            passes["calibration"] = {
                "status": "not_calibrated",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            notes.append("no fitted calibrator: the check reports explicit invalid rows")
        try:
            dev = dev_pass(model, settings, device=settings.device, calibration=calibration)
        except Exception as exc:  # noqa: BLE001
            dev = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "records": []}
        passes["dev"] = dev
        if dev.get("records"):
            write_json(output / "records_dev.json", dev["records"])
            try:
                passes["dev_action_check"] = action_pass(
                    model,
                    settings,
                    device=settings.device,
                    dev_records=dev["records"],
                    backend=backend,
                )
            except Exception as exc:  # noqa: BLE001 - a failed action head must not lose the rest
                passes["dev_action_check"] = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            write_json(output / "action_error.json", passes["dev_action_check"])
    if args.command in ("all", "interventions"):
        try:
            records = (passes.get("dev") or {}).get("records") or []
            if not records:
                source_dev = build_source(settings, "dev", max_windows=None)
                indices, identity = fixed_sample_identities(
                    source_dev,
                    settings.dev_samples,
                    path=output / "dev_samples.json",
                    partition="dev",
                    frozen=None,
                )
                records = [
                    {"anchor_index": index, "identity": item}
                    for index, item in zip(indices, identity["identities"], strict=False)
                ]
            passes["interventions"] = intervention_pass(
                model, settings, device=settings.device, dev_records=records
            )
            write_json(output / "interventions.json", passes["interventions"])
        except Exception as exc:  # noqa: BLE001
            passes["interventions"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    if args.command == "all" and settings.figures:
        passes["figures"] = figures_pass(
            model,
            settings,
            device=settings.device,
            backend=backend,
            model_step=model_step,
        )

    for name, payload in list(passes.items()):
        if isinstance(payload, Mapping) and payload.get("status") in ("not_run", "failed"):
            extra.setdefault("not_run", []).append(
                {"what": name, "reason": payload.get("reason", "not run")}
            )
    extra["preview"] = {
        "dir": settings.preview_dir,
        "files": _preview_files(settings.preview_dir),
    }
    report = build_report(settings, status="ran", passes=passes, extra=extra)
    page_files = (passes.get("figures") or {}).get("page_preview_files") or extra["preview"][
        "files"
    ]
    extra["status_publish"] = publish_status(settings, preview_files=page_files, report=report)
    report = build_report(settings, status="ran", passes=passes, extra=extra)
    write_json(output / "eval_v12.json", report)
    print(json.dumps(_jsonable(report["summary"]), ensure_ascii=False))
    return 0


def summarise_report(report: Mapping[str, Any]) -> dict:
    """The few numbers a reader needs first; the full receipts stay in the report file."""
    passes = dict(report.get("passes") or {})
    dev = dict(passes.get("dev") or {})
    calibration = dict(passes.get("calibration") or {})
    action = dict(passes.get("dev_action_check") or {})
    interventions = dict(passes.get("interventions") or {})
    return {
        "status": report.get("status"),
        "check_counts": dev.get("check_summary"),
        "check_counts_all_rows": dev.get("check_summary_all_rows"),
        "invalid_reasons": dev.get("invalid_reasons"),
        "calibration": {
            "status": calibration.get("status"),
            "n_sequences": calibration.get("n_sequences"),
            "threshold": (calibration.get("calibration") or {}).get("threshold"),
            "statistics": calibration.get("statistics"),
        },
        "action_error": action.get("loss"),
        "interventions": {
            name: {
                "status": payload.get("status"),
                "applied": payload.get("applied"),
                "raw_diff_norm": payload.get("raw_diff_norm"),
            }
            for name, payload in dict(interventions.get("kinds") or {}).items()
        },
        "preview_files": (report.get("preview") or {}).get("files"),
        "not_run": report.get("not_run"),
    }


def _preview_files(directory: Any) -> list[str]:
    root = Path(directory)
    if not root.is_dir():
        return []
    run_dir = root.parent
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp"}:
            try:
                out.append(str(path.relative_to(run_dir)))
            except ValueError:
                out.append(str(path))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
