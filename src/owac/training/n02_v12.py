"""N02 v1.2 P2 real training entry: official RAW windows -> v1.2 chain -> pi0.5 flow matching.

What this file is
-----------------

The P2 main path, end to end on real data, in one readable loop::

    data_v12.V12WindowSource (official RoboTwin/LIBERO RAW cache, frozen P0 partition)
      -> RawEncoderV12 (ConvNeXt V2-Base, stride 4/8/16)
      -> N02V12Inference (K joint multi-camera explanations, RAW decode, support, accepted
         correction; SAVi-sourced corrector/predictor; Perceiver readout)
      -> n02_v12_tokens.N02V12TokenSource (the current-row tokens of one decision moment)
      -> RawVLAFeaturePolicy.forward (frozen pi0.5 backend, native flow-matching action loss)
      + lambda_raw  * L_masked_raw          (FCMAE masked RAW reconstruction, mask-aware)
      + lambda_pred * L_causal_raw_prediction (prediction locked before the target frame)
      + lambda_noise * L_noise_calibration   (only with a real declared noise target)

Trained and frozen
------------------

Frozen (``freeze_backend=True``): original pi0.5 checkpoint parameters. Explicitly declared
``policy.backend_adapter`` may add trainable LoRA weights; original weights stay frozen.
``freeze_encoder`` freezes the RAW CNN, while ``freeze_frontend`` freezes the entire RAW/N02
model and masked branch. Optimizer groups and reports disclose the actual trainable subset.
Every frozen backend parameter is asserted to receive no gradient.

Default frontend training includes: ``RawEncoderV12`` (ConvNeXt V2-Base raw, initialised from
scratch - no RGB weights), the
1x1 level projections, ``N02V12Inference`` (candidate seeds, evidence/prior/residual projections,
SAVi corrector/predictor), ``RawMeasurementDecoderV12``, ``N02ReadoutV12``, the FCMAE masked
branch (mask token + decoder + pixel head, whose *encoder* tensors are the shared RAW backbone),
and the bridge projection when one exists.  The hard candidate argmax and the accept/reject
decision carry no gradient; that boundary is reported, never hidden.

Units and protocol (no second normalisation)
--------------------------------------------

Actions and proprio leave the frozen loader already normalised by the platform's official
statistics; ``prepare_control_targets`` only pads to 32 dims and ``control_actions`` only slices
back.  This entry never normalises or de-normalises: that boundary belongs to the online sampling
path, where the output is de-normalised once before ``take_action``.  RoboTwin stays float32 raw
scale (no /255, no per-frame max), LIBERO is uint8/255 as the loader declares, the horizon/offset
are the loader's (RoboTwin 50/1, LIBERO 10/0) and the per-action padding mask is the loader's own.

CLI
---

``--help`` is the contract::

    cd <repo>/src
    PYTHONPATH=<repo>/src:/workspace/owac/rawvla/source/starVLA \\
      CUDA_VISIBLE_DEVICES=0 <repo>/outputs/n02-full/runtime/py310-cu128/bin/python \\
      -m owac.training.n02_v12 --config <repo>/configs/encoders/n02_v12/raw_n02_robotwin.json \\
      --seed 1103 --max-updates 1000 \\
      --output <repo>/outputs/n02-v12/P2/raw-n02-robotwin-s1103

A previous sync created a shadow package at the checkout root; it has been archived.
The import identity guard detects any recurrence before running a job.
:func:`package_identity_error` checks the imported package identity before anything runs
and refuses to run shadowed, printing the working alternative.

Status output
-------------

``status.json`` / ``history.jsonl`` are written through ``owac.runtime.n02_v12_status`` with the
frozen field set: at least every 10 optimizer updates or 5 seconds, plus a 5-second heartbeat
thread during long steps, plus an immediate write on every dev pass, save and failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.frameworks.n02_full.data_v12 import apply_declared_isp as apply_declared_isp
from owac.frameworks.n02_full.data_v12 import read_task_prompts as read_task_prompts
from owac.frameworks.n02_full.sequence_v12 import EXECUTION_REVISION
from owac.frameworks.n02_full.state import ObservationBundle

#: Status/history schema written by this entry (frozen by owac.runtime.n02_v12_status).
STATUS_VERSION = "n02-v12-p2-status-v1"
CHECKPOINT_FORMAT = "n02-v12-p2-checkpoint-v1"

STAGE = "P2"
MAX_STATE_DIM = 32
ACTION_DIM_PADDED = 32
DEFAULT_AUX = {"lambda_raw": 1.0, "lambda_pred": 1.0, "lambda_noise": 0.0}
STATUS_FIELDS = (
    "run_id",
    "stage",
    "arm",
    "platform",
    "seed",
    "state",
    "activity",
    "gpu",
    "step",
    "target_steps",
    "updated_at",
    "started_at",
    "step_seconds",
    "loss",
    "dev_action_error",
    "checkpoint",
    "eval",
    "preview_files",
    "log_tail",
)


# --------------------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------------------- #
class LogTail:
    """stdout logging that keeps the last lines for ``status.json['log_tail']``."""

    def __init__(self, keep: int = 30) -> None:
        self.keep = int(keep)
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S', time.localtime())}] {message}"
        with self._lock:
            print(line, flush=True)
            self.lines.append(line)
            if len(self.lines) > self.keep:
                del self.lines[: len(self.lines) - self.keep]

    def tail(self) -> list[str]:
        with self._lock:
            return list(self.lines)


class Heartbeat:
    """Refresh ``status.json['updated_at']`` every few seconds while a long step runs."""

    def __init__(self, run_dir: Path, activity: str, interval: float = 5.0) -> None:
        self.run_dir = Path(run_dir)
        self.activity = str(activity)
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Heartbeat:
        from owac.runtime.n02_v12_status import heartbeat

        def loop() -> None:
            while not self._stop.wait(self.interval):
                try:
                    heartbeat(self.run_dir, activity=self.activity)
                except Exception:  # noqa: BLE001 - a heartbeat never kills a run
                    return

        self._thread = threading.Thread(target=loop, name="n02-v12-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------------------- #
# import identity guard
# --------------------------------------------------------------------------------------- #
def _repo_root() -> Path:
    """``<repo>`` of this file (``<repo>/src/owac/training/n02_v12.py``)."""
    file = Path(__file__).resolve()
    for parent in file.parents:
        if (parent / "src" / "owac").is_dir():
            return parent
    return file.parents[2]


def package_identity_error() -> str | None:
    """The shadowing message, or ``None`` when ``owac`` really is ``<repo>/src/owac``."""
    import owac

    expected = (_repo_root() / "src" / "owac").resolve()
    actual = Path(owac.__file__).resolve()
    if actual == expected or expected in actual.parents:
        return None
    return (
        "N02 v1.2 training refuses to run: the imported 'owac' package is not this checkout's "
        "source tree.\n"
        f"  imported from : {actual}\n"
        f"  expected      : {expected}\n"
        "The checkout root carries a stale untracked copy named 'owac/' which shadows 'src/owac' "
        "for 'python -m' (python searches the working directory first).\n"
        "Working alternative, nothing moved or deleted:\n"
        f"  cd {expected.parent}\n"
        f"  PYTHONPATH={expected.parent}:<starvla root> <python> -m owac.training.n02_v12 "
        "--config <config> --output <run dir>\n"
        "Or keep the working directory and run the source file by path:\n"
        "  PYTHONPATH={src}:<starvla root> <python> {file}".format(
            src=expected.parent, file=expected / "training" / "n02_v12.py"
        ).strip()
    )


# --------------------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------------------- #
def deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in dict(extra).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config {path} must be a JSON object")
    missing = [key for key in ("platform", "model", "data", "policy", "loop") if key not in payload]
    if missing:
        raise ValueError(f"config {path} is missing required sections {missing}")
    return payload


@dataclass
class RunPlan:
    """Resolved run plan: config file plus CLI overrides (CLI wins)."""

    config: dict
    config_path: Path
    output: Path
    run_id: str
    arm: str
    platform: str
    seed: int
    max_updates: int
    profile_updates: int
    batch_size: int
    grad_accum: int
    validate_every: int
    save_every: int
    dev_windows: int
    calibration_windows: int
    device: str
    threads: int
    previews: int
    history: int
    resume: Path | None = None
    notes: list[str] = field(default_factory=list)


def resolve_plan(args: argparse.Namespace) -> RunPlan:
    config = load_config(args.config)
    declared_revision = config.get("execution_revision")
    if declared_revision is not None and declared_revision != EXECUTION_REVISION:
        raise ValueError("config execution revision differs from the current policy input contract")
    loop = dict(config.get("loop", {}))
    arm = str(config.get("arm", "raw_n02"))
    platform = str(config.get("platform", config.get("data", {}).get("dataset_format", "")))
    if not platform:
        raise ValueError("config must declare 'platform' (or data.dataset_format)")
    seed = int(args.seed if args.seed is not None else loop.get("seed", 1103))
    run_id = str(
        args.run_id or config.get("run_id") or f"{arm.replace('_', '-')}-{platform}-s{seed}"
    )
    if args.output:
        output = Path(args.output)
    else:
        root = str(config.get("paths", {}).get("output_root", "outputs/n02-v12"))
        output = Path(f"{root}/{STAGE}/{run_id}")
    notes: list[str] = []
    max_updates = int(args.max_updates if args.max_updates is not None else loop["max_updates"])
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    batch_size = int(args.batch_size if args.batch_size is not None else loop.get("batch_size", 1))
    grad_accum = int(args.grad_accum if args.grad_accum is not None else loop.get("grad_accum", 1))
    for name, value in (("batch_size", batch_size), ("grad_accum", grad_accum)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    profile_updates = int(
        args.profile_updates
        if args.profile_updates is not None
        else loop.get("profile_updates", 200)
    )
    if profile_updates > max_updates:
        notes.append(
            f"profile_updates={profile_updates} exceeds max_updates={max_updates}; the profile "
            "covers the whole run"
        )
    model = dict(config.get("model", {}))
    data = dict(config.get("data", {}))
    return RunPlan(
        config=config,
        config_path=Path(args.config),
        output=output,
        run_id=run_id,
        arm=arm,
        platform=platform,
        seed=seed,
        max_updates=max_updates,
        profile_updates=profile_updates,
        batch_size=batch_size,
        grad_accum=grad_accum,
        validate_every=int(
            args.validate_every
            if args.validate_every is not None
            else loop.get("validate_every", 50)
        ),
        save_every=int(
            args.save_every if args.save_every is not None else loop.get("save_every", 1000)
        ),
        dev_windows=int(loop.get("dev_windows", 4)),
        calibration_windows=int(loop.get("calibration_windows", 4)),
        device=str(args.device or config.get("runtime", {}).get("device", "cuda")),
        threads=int(config.get("runtime", {}).get("threads", 4)),
        previews=int(args.previews if args.previews is not None else loop.get("previews", 1)),
        history=int(model.get("history", data.get("history", 8))),
        resume=Path(args.resume) if args.resume else None,
        notes=notes,
    )


def training_contract(plan: RunPlan, identity: dict | None = None) -> dict:
    """Resume must preserve the objective, representation, data and update protocol."""
    sections = (
        "arm",
        "platform",
        "representation",
        "model",
        "data",
        "policy",
        "aux",
        "optim",
        "initialisation",
    )
    content = {key: plan.config.get(key) for key in sections}
    content.update(
        execution_revision=EXECUTION_REVISION,
        policy_input_contract="pi05-native-language-state-v1",
        seed=plan.seed,
        batch_size=plan.batch_size,
        grad_accum=plan.grad_accum,
        history=plan.history,
        bptt_steps=int(plan.config["loop"].get("bptt_steps", 2)),
        performance=plan.config.get("runtime", {}).get("performance", {}),
    )
    if plan.config.get("task_adaptation"):
        content["task_adaptation"] = plan.config["task_adaptation"]
    for key in ("execution", "online_checks"):
        if key in plan.config:
            content[key] = plan.config[key]
    if identity is not None:
        content["data_identity"] = identity
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return {"sha256": hashlib.sha256(canonical.encode()).hexdigest(), "content": content}


def validate_run_directory(plan: RunPlan, *, profile_only: bool = False) -> None:
    """Reject accidental replacement of prior evidence before constructing the runtime."""
    if profile_only and plan.resume is not None:
        raise ValueError("a fresh zero-update profile cannot resume a checkpoint")
    if plan.resume is None:
        if plan.output.exists() and any(plan.output.iterdir()):
            raise FileExistsError(
                f"run directory is not empty: {plan.output}; choose a new --output"
            )
        return
    payload = torch.load(plan.resume, map_location="cpu", weights_only=False)
    expected = training_contract(plan)
    saved = payload.get("training_contract", {})
    saved_config = dict(saved.get("content", {}))
    saved_config.pop("data_identity", None)
    if saved_config != expected["content"]:
        raise ValueError(
            "resume training contract mismatch (revision/arm/model/data/seed/optimizer)"
        )
    if int(payload["step"]) >= plan.max_updates:
        raise ValueError("--max-updates must exceed the resumed checkpoint step")
    if plan.config.get("task_adaptation", {}).get("protocol") == "n02-task-repair-r11-v1":
        if plan.output.resolve() != Path(plan.resume).resolve().parent:
            raise ValueError("r11 resume requires the original run directory and baseline")
        for filename in ("run_config.json", "dev_baseline.json", "task_dev_manifest.json"):
            if not (plan.output / filename).is_file():
                raise ValueError(f"r11 resume is missing its frozen {filename}")


# --------------------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------------------- #
def build_source(config: dict, partition: str, *, history: int, batch_size: int):
    """One frozen-partition window source (manifest receipt + frozen anchors when present)."""
    from owac.frameworks.n02_full.data_v12 import V12WindowSource

    data = dict(config["data"])
    anchor_path = None
    if data.get("anchor_dir"):
        candidate = (
            Path(data["anchor_dir"]) / f"{data['dataset_format']}-{partition}-anchors.json.gz"
        )
        if candidate.exists():
            anchor_path = candidate
    manifest_path = None
    if data.get("manifest_dir"):
        candidate = Path(data["manifest_dir"]) / f"manifest-{data['dataset_format']}.json"
        if candidate.exists():
            manifest_path = candidate
    return V12WindowSource(
        dataset_format=str(data["dataset_format"]),
        data_root=data["data_root"],
        partition=partition,
        history=int(history),
        stats_path=data.get("stats_path"),
        anchor_path=anchor_path,
        manifest_path=manifest_path,
        seed=int(data.get("manifest_seed", 1103)),
        batch_size=int(batch_size),
        task_names=data.get("task_names"),
        exclude_episode_ids=data.get("exclude_episode_ids", ()),
    )


def anchor_action_targets(targets: dict, anchors: int) -> tuple[np.ndarray, np.ndarray]:
    """One ``[anchors, h, control_dim]`` block and its ``[anchors, h]`` padding mask.

    ``data_v12.V12WindowSource.targets()`` concatenates the loader's own per-sample arrays, whose
    mask is one row per anchor, so the reshape restores the anchor axis without touching a value.
    """
    raw_actions = np.asarray(targets["actions"], dtype=np.float32)
    raw_mask = np.asarray(targets["action_mask"], dtype=bool)
    actions = raw_actions.reshape(int(anchors), -1, int(raw_actions.shape[-1]))
    mask = raw_mask.reshape(int(anchors), -1)
    if tuple(mask.shape[:2]) != tuple(actions.shape[:2]):
        raise ValueError(
            f"action mask {tuple(mask.shape)} does not match actions {tuple(actions.shape)}"
        )
    return actions, mask


def subset_spatial(spatial: Any, rows: torch.Tensor):
    """``SpatialV12`` restricted to ``rows`` (per-row convolutions, so slicing is exact)."""
    from owac.frameworks.n02_full.encoder_v12 import SpatialV12

    def take(mapping: dict, pick) -> dict:
        return {key: pick(value) for key, value in mapping.items()}

    return SpatialV12(
        levels=take(spatial.levels, lambda value: value.index_select(0, rows.to(value.device))),
        coords=take(spatial.coords, lambda value: value.index_select(0, rows.to(value.device))),
        valid=take(spatial.valid, lambda value: value.index_select(0, rows.to(value.device))),
        camera_ids=spatial.camera_ids,
        level_ids=spatial.level_ids,
        level_dims=dict(spatial.level_dims),
        support_hash=f"{spatial.support_hash}:subset{int(rows.numel())}",
        transforms=dict(getattr(spatial, "transforms", {})),
        support_ids=dict(getattr(spatial, "support_ids", {})),
        input_mode=str(getattr(spatial, "input_mode", "RAW3")),
    )


def subset_queries(queries: Any, rows: torch.Tensor):
    """``QuerySet`` restricted to ``rows`` (same measurement space, fewer decision rows)."""
    from owac.frameworks.n02_full.inference_v12 import QuerySet

    def take_ids(value: Any) -> Any:
        if not value:
            return value
        index = rows.tolist()
        return tuple(value[int(item)] for item in index)

    return QuerySet(
        coords=queries.coords.index_select(0, rows),
        camera_index=queries.camera_index.index_select(0, rows),
        query_channel=queries.query_channel.index_select(0, rows),
        target=queries.target.index_select(0, rows),
        masks={name: value.index_select(0, rows) for name, value in queries.masks.items()},
        ids=take_ids(queries.ids),
        camera_ids=queries.camera_ids,
        sampled_per_camera=queries.sampled_per_camera,
        coverage=queries.coverage,
        duplicates_dropped=queries.duplicates_dropped,
        channel_reduction=queries.channel_reduction,
        per_camera_query_count=queries.per_camera_query_count,
    )


def subset_frames(bundle: Any, rows: torch.Tensor) -> dict:
    """Per-camera ``MeasurementFrame`` restricted to ``rows`` (unit metadata kept)."""
    import dataclasses

    index = rows.tolist()
    out: dict[str, Any] = {}
    for camera_id, frame in bundle.frames.items():
        values = torch.as_tensor(frame.values)
        # the loader serves CPU frames and the chain moves them to the device itself, so the
        # row index always follows the tensor it indexes
        local = rows.to(values.device)
        updates: dict[str, Any] = {"values": values.index_select(0, local)}
        for name in ("valid_mask", "bad_mask", "saturated_mask", "missing_mask"):
            mask = getattr(frame, name, None)
            if mask is None:
                continue
            mask = torch.as_tensor(mask)
            if mask.ndim >= 3 and int(mask.shape[0]) == int(values.shape[0]):
                updates[name] = mask.index_select(0, rows.to(mask.device))
        for name in ("episode_ids", "frame_ids"):
            payload = getattr(frame, name, None)
            if payload is not None and len(payload) == int(values.shape[0]):
                updates[name] = tuple(payload[int(item)] for item in index)
        flags = getattr(frame, "new_measurement", None)
        if flags is not None:
            array = np.asarray(flags)
            if array.shape and int(array.shape[0]) == int(values.shape[0]):
                updates["new_measurement"] = array[index]
        out[camera_id] = dataclasses.replace(frame, **updates)
    return out


# --------------------------------------------------------------------------------------- #
# masked RAW branch (FCMAE semantics on the *shared* RAW backbone)
# --------------------------------------------------------------------------------------- #
def bind_shared_encoder(masked_encoder: Any, dense_backbone: Any) -> dict:
    """Point the mask-aware encoder at the dense backbone's parameters (shared tensors).

    The mask-aware FCMAE encoder is the accepted P0 port: convolutions run on the zero-filled
    support while LayerNorm/GRN statistics are computed over valid positions only, so a masked
    target pixel cannot be read through the unmasked current frame.  It declares the same
    parameter names and shapes as the dense ConvNeXt V2 stages (checked here at run time);
    rebinding makes the masked objective train *the* action-branch backbone instead of a copy.
    """
    dense = dict(dense_backbone.named_parameters())
    bound, unbound = [], []
    for name, parameter in list(masked_encoder.named_parameters()):
        target = dense.get(name)
        if target is None or tuple(target.shape) != tuple(parameter.shape):
            unbound.append(name)
            continue
        parent, _, leaf = name.rpartition(".")
        owner = masked_encoder.get_submodule(parent) if parent else masked_encoder
        owner._parameters[leaf] = target  # same tensor: one backward, one optimizer entry
        bound.append(name)
    if not bound:
        raise ValueError("the mask-aware encoder shares no parameter name with the backbone")
    return {"bound": len(bound), "unbound": unbound}


class MaskedRawBranch(torch.nn.Module):
    """FCMAE masked RAW reconstruction over the shared backbone (``components/fcmae.py``)."""

    def __init__(self, encoder: Any, config: dict, *, camera_ids: Any) -> None:
        super().__init__()
        from owac.frameworks.n02_full.components.fcmae import MaskAwareDenseConvNeXtV2Encoder
        from owac.frameworks.n02_full.observation import RawFCMAEPretrainer

        model = dict(config["model"])
        aux = dict(config.get("aux", {}))
        self.camera_ids = tuple(str(item) for item in camera_ids)
        mask_aware = MaskAwareDenseConvNeXtV2Encoder(
            in_chans=int(model.get("in_chans", 3)),
            depths=tuple(model.get("depths", (3, 3, 27, 3))),
            dims=tuple(model.get("dims", (128, 256, 512, 1024))),
            grn_scope=str(aux.get("grn_scope", "sparse")),
        )
        self.sharing = bind_shared_encoder(mask_aware, encoder.backbone)
        self.pretrainer = RawFCMAEPretrainer(
            input_mode=str(model.get("input_mode", "RAW3")).lower(),
            in_chans=int(model.get("in_chans", 3)),
            depths=tuple(model.get("depths", (3, 3, 27, 3))),
            dims=tuple(model.get("dims", (128, 256, 512, 1024))),
            patch_size=int(aux.get("patch_size", 32)),
            mask_ratio=float(aux.get("mask_ratio", 0.6)),
            decoder_embed_dim=int(aux.get("decoder_embed_dim", 512)),
            decoder_depth=int(aux.get("decoder_depth", 1)),
            img_size=tuple(model.get("input_size", (256, 256))),
            alignment=int(model.get("alignment", 32)),
            encoder=mask_aware,
        )

    def own_parameters(self) -> list:
        """Parameters that exist only in this branch (mask token, decoder, pixel head)."""
        shared = {id(parameter) for parameter in self.pretrainer.fcmae.encoder.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in shared and parameter.requires_grad
        ]

    def forward(self, frames: dict, *, camera_ids: Any = None):
        return self.pretrainer(frames, camera_ids=tuple(camera_ids or self.camera_ids))


# --------------------------------------------------------------------------------------- #
# the trainer
# --------------------------------------------------------------------------------------- #
class N02V12Trainer:
    """Build, check and run the P2 loop; everything it reports comes from real tensors."""

    def __init__(self, plan: RunPlan, log: LogTail) -> None:
        from owac.runtime.n02_v12_status import update_status

        self.plan = plan
        self.log = log
        self.update_status = update_status
        self.run_dir = Path(plan.output)
        self.checks: dict[str, Any] = {}
        self.overview: dict = {}
        self.final: dict = {}
        self.frozen_with_grad: list[str] = []
        self.parameter_groups: dict[str, list] = {}

    # -- construction -------------------------------------------------------------------

    def build(self) -> None:
        from owac.backends.n02_v12_tokens import N02V12TokenSource
        from owac.frameworks.n02_full.model_v12 import build_n02_v12_model, configure_v12_runtime

        plan = self.plan
        from owac.training.n02_v12_task import validate_repair_config

        validate_repair_config(plan.config)
        torch.set_num_threads(plan.threads)
        torch.manual_seed(plan.seed)
        np.random.seed(plan.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(plan.device)
        performance = plan.config.get("runtime", {}).get("performance", {})
        configure_v12_runtime(plan.config, self.device)
        self.representation = dict(plan.config.get("representation", {}) or {})
        self.log(
            "representation arm: {} - {}".format(
                self.representation.get("kind", "raw3"),
                self.representation.get("declared", "loader RAW3 in loader units"),
            )
        )
        self.initialise_status(activity="构建数据源")
        self.log("building window sources (frozen P0 partitions, frozen anchors)")
        self.sources = {
            name: build_source(plan.config, name, history=plan.history, batch_size=plan.batch_size)
            for name in ("train", "dev", "calibration")
        }
        self.log(
            "sources: "
            + ", ".join(
                f"{name}={source.anchor_count} anchors" for name, source in self.sources.items()
            )
        )
        self.identity = {name: source.identity() for name, source in self.sources.items()}
        self.status(state="initialising", activity="构建 v1.2 模型")
        self.model = build_n02_v12_model(plan.config["model"]).to(self.device)
        self.model.inference.encoder.batch_cameras = bool(performance.get("batch_cameras", False))
        self.model.inference.decoder.group_camera_queries = bool(
            performance.get("group_camera_queries", False)
        )
        self.masked = MaskedRawBranch(
            self.model.inference.encoder, plan.config, camera_ids=plan.config["model"]["camera_ids"]
        ).to(self.device)
        self.log(
            f"model: {sum(p.numel() for p in self.model.parameters())} parameters; mask-aware "
            f"branch shares {self.masked.sharing['bound']} backbone tensors "
            f"(unbound: {len(self.masked.sharing['unbound'])})"
        )
        self.status(state="initialising", activity="加载 π0.5 后端与权重")
        self._build_policy(N02V12TokenSource)

    def _build_policy(self, token_source_cls: Any) -> None:
        from owac.backends.rawvla_features_n02 import (
            build_feature_policy,
            check_runtime_assets,
            load_checkpoint_state,
        )

        plan = self.plan
        policy = plan.config["policy"]
        if not bool(policy.get("freeze_backend", True)):
            raise ValueError(
                "P2 optimizer supports only the frozen backend; freeze_backend must be true"
            )
        camera_ids = tuple(plan.config["model"]["camera_ids"])
        self.assets = check_runtime_assets(
            checkpoint=policy["checkpoint"],
            tokenizer=policy["tokenizer"],
            data_root=plan.config["data"]["data_root"],
        )
        if not all(
            self.assets.get(key)
            for key in (
                "real_backend_available",
                "checkpoint_exists",
                "tokenizer_exists",
                "data_root_exists",
            )
        ):
            raise RuntimeError(f"real π0.5 assets are incomplete: {self.assets}")
        self.token_source = token_source_cls(
            self.model,
            camera_ids=camera_ids,
            history=plan.history,
            device=self.device,
            bptt_steps=int(plan.config["loop"].get("bptt_steps", 2)),
        )
        self.bridge = build_feature_policy(
            str(policy["suite"]),
            Path(policy["tokenizer"]),
            self.token_source,
            token_dim=int(policy.get("token_dim", 2048)),
            tokens_per_camera=int(policy.get("tokens_per_camera", 256)),
            image_keys=camera_ids,
            freeze_backend=bool(policy.get("freeze_backend", True)),
        )
        weights = load_checkpoint_state(policy["checkpoint"])
        self.load_audit = self.bridge.load_audited_replaced_vision(weights)
        del weights
        self.bridge.base.to(self.device).eval()
        from owac.backends.low_rank import install_low_rank

        self.backend_adapters, self.backend_adapter_report = install_low_rank(
            self.bridge.base, policy.get("backend_adapter")
        )
        self.bridge.frozen_parameter_count = sum(
            p.numel() for p in self.bridge.base.parameters() if not p.requires_grad
        )
        if policy.get("freeze_encoder", False):
            self.model.inference.encoder.requires_grad_(False)
        if policy.get("freeze_frontend", False):
            self.model.requires_grad_(False)
            self.masked.requires_grad_(False)
            if not self.backend_adapters:
                raise ValueError("a fully frozen frontend requires declared backend adapters")
        task = self.plan.config.get("task_adaptation", {})
        if task and (
            self.plan.platform != "robotwin"
            or task.get("objective") not in ("control_flow_14", "native_flow_32")
        ):
            raise ValueError("unknown task adaptation objective/platform")
        if task.get("train_deep_blocks", 0):
            if policy.get("freeze_encoder") or policy.get("freeze_frontend"):
                raise ValueError("task-supervised encoder scope conflicts with full freeze flags")
            if self.plan.platform != "robotwin" or not policy.get("freeze_backend", True):
                raise ValueError("r9 task adaptation requires the frozen RoboTwin backend")
            self.model.requires_grad_(False)
            self.masked.requires_grad_(False)
            encoder = self.model.inference.encoder
            count = int(task["train_deep_blocks"])
            if not 1 <= count <= len(encoder.backbone.stages[2]):
                raise ValueError("invalid number of task-supervised deep encoder blocks")
            encoder.backbone.stages[2][-count:].requires_grad_(True)
            encoder.proj.requires_grad_(True)
            self.model.inference.readout.requires_grad_(True)
            if task.get("train_state_modules", False):
                for name, module in self.model.inference.named_children():
                    if name not in ("encoder", "readout"):
                        module.requires_grad_(True)
                self.model.inference.slot_latents.requires_grad_(True)
        if task:
            self._write_json(
                "task_trainable.json",
                {
                    "frontend": [n for n, p in self.model.named_parameters() if p.requires_grad],
                    "backend": self.backend_adapter_report,
                    "objective": task.get("objective"),
                },
            )
        self._write_json("backend_adapter.json", self.backend_adapter_report)
        self.log(
            "π0.5 backend loaded: frozen={frozen} frozen_parameters={count} "
            "active_missing={missing} replaced_vision_present={vision}".format(
                frozen=bool(policy.get("freeze_backend", True)),
                count=self.bridge.frozen_parameter_count,
                missing=len(self.load_audit.get("active_missing", [])),
                vision=len(self.load_audit.get("replaced_vision_present", [])),
            )
        )
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        optim = dict(self.plan.config.get("optim", {}))
        groups = {
            "encoder": list(self.model.inference.encoder.parameters()),
            "decoder": list(self.model.inference.decoder.parameters()),
            "readout": list(self.model.inference.readout.parameters()),
            "proposals_corrector": [
                parameter
                for name, parameter in self.model.inference.named_parameters()
                if not name.startswith(("encoder.", "decoder.", "readout."))
            ],
            "masked_branch": self.masked.own_parameters(),
            "bridge_projection": [
                parameter
                for module in self.bridge.bridge_modules()
                for parameter in module.parameters()
            ],
            "backend_adapter": [
                parameter
                for adapter in getattr(self, "backend_adapters", {}).values()
                for parameter in adapter.parameters()
            ],
        }
        groups = {
            name: [p for p in parameters if p.requires_grad] for name, parameters in groups.items()
        }
        self.parameter_groups = groups
        base_lr = float(optim.get("lr", 1e-4))
        head_lr = float(optim.get("lr_head", base_lr))
        parameter_groups = []
        for name, parameters in groups.items():
            if not parameters:
                continue
            lr = base_lr if name in ("encoder", "masked_branch") else head_lr
            if name == "backend_adapter":
                lr = float(optim.get("lr_backend_adapter", head_lr))
            parameter_groups.append({"params": parameters, "lr": lr, "name": name})
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=base_lr,
            betas=tuple(optim.get("betas", (0.9, 0.95))),
            weight_decay=float(optim.get("weight_decay", 0.01)),
            fused=self.device.type == "cuda"
            and bool(
                self.plan.config.get("runtime", {}).get("performance", {}).get("fused_adam", False)
            ),
        )
        self.grad_clip = float(optim.get("grad_clip", 1.0))
        self.warmup = int(optim.get("warmup_updates", 20))
        self.base_lrs = [group["lr"] for group in parameter_groups]

    def clip_gradients(self):
        scope = self.plan.config.get("optim", {}).get("clip_scope", "global")
        if scope not in ("global", "parameter_group"):
            raise ValueError(f"unknown gradient clip scope: {scope}")
        groups = (
            self.optimizer.param_groups
            if scope == "parameter_group"
            else [
                {
                    "name": "global",
                    "params": [p for g in self.optimizer.param_groups for p in g["params"]],
                }
            ]
        )
        norms = {
            group.get("name", str(index)): torch.nn.utils.clip_grad_norm_(
                group["params"], self.grad_clip, error_if_nonfinite=True
            )
            for index, group in enumerate(groups)
        }
        self.last_clip_norms = {name: float(value) for name, value in norms.items()}
        # clip_grad_norm_ returns a CPU zero for a group without gradients.
        # Disabled auxiliary branches therefore must not be stacked with CUDA norms.
        return math.sqrt(sum(value * value for value in self.last_clip_norms.values()))

    # -- language / targets -------------------------------------------------------------

    def language(
        self, prompts: list[str], proprio: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The native prompt includes the padded state, not only the measured joints."""
        from owac.backends.rawvla_language import prepare_language_state

        return prepare_language_state(self.bridge, prompts, proprio, device=self.device)

    def window(self, source: Any, start: int, stop: int) -> Any:
        """Load one real window and apply this arm's declared representation."""
        loaded = source.window(start, stop)
        kind = str(self.representation.get("kind", "raw3"))
        if kind in ("raw3", ""):
            return loaded
        if kind != "raw3_fixed_isp":
            raise ValueError(f"unknown representation kind {kind!r}")
        return dataclasses.replace(
            loaded, observation=apply_declared_isp(loaded.observation, self.representation)
        )

    def anchor_bundle(self, batch: Any, prior: Any = None) -> Any:
        """Current-row input for the isolated prior probe, never the action training path."""
        from owac.backends.n02_v12_tokens import anchor_row_indices
        from owac.frameworks.n02_full.inference_v12 import V12InferenceInput

        rows_total = int(batch.observation.context.proprio.shape[0])
        index = anchor_row_indices(rows_total, int(batch.history))
        rows = torch.as_tensor(index, device=self.device, dtype=torch.long)
        observation = batch.observation
        new_context = dataclasses.replace(
            observation.context, proprio=np.asarray(observation.context.proprio)[index]
        )
        if prior is not None:
            # the frozen context channel for a declared prior; ``_log_prior`` reads it when no
            # explicit ``declared_prior`` is attached to the inference input
            as_tensor = torch.as_tensor(prior, dtype=torch.float32)
            if as_tensor.ndim == 1:
                as_tensor = as_tensor.unsqueeze(0).expand(len(index), -1)
            elif as_tensor.ndim == 2 and int(as_tensor.shape[0]) == 1:
                as_tensor = as_tensor.expand(len(index), -1)
            if as_tensor.ndim != 2 or int(as_tensor.shape[0]) != len(index):
                raise ValueError(
                    f"declared prior {tuple(as_tensor.shape)} does not cover {len(index)} anchors"
                )
            new_context = dataclasses.replace(
                new_context,
                hypothesis_prior=as_tensor,
                hypothesis_prior_available=True,
            )
        episode_ids = (
            tuple(observation.episode_ids[item] for item in index)
            if len(observation.episode_ids) == rows_total
            else observation.episode_ids
        )
        anchor_ids = (
            tuple(observation.anchor_ids[item] for item in index)
            if len(observation.anchor_ids) == rows_total
            else observation.anchor_ids
        )
        new_time = observation.new_time_mask
        if new_time is not None:
            new_time = np.asarray(new_time)[index]
        bundle = ObservationBundle(
            frames=subset_frames(observation, rows),
            context=new_context,
            episode_ids=episode_ids,
            anchor_ids=anchor_ids,
            new_time_mask=new_time,
        )
        # the real causal frame number of each current row drives the decoder's time
        # conditioning; without it the frozen fallback reads the row's frame-id tuple and lands
        # on 0, which would hide the actual acquisition time from the RAW decoder
        raw_index = getattr(batch, "frame_index", None)
        # a plain list of ints: the frozen reader accepts a tensor/list/tuple or a scalar, and a
        # 1-D numpy array falls through to its scalar branch
        frame_index = (
            None if raw_index is None else [int(item) for item in np.asarray(raw_index)[index]]
        )
        # the per-decision query set carries one measurement unit per camera: the window-level
        # (episode, frame, camera) triples describe the whole history, not the current rows, and
        # the frozen reader would map them to the wrong row count
        return V12InferenceInput(
            observation=bundle,
            frame_index=frame_index,
            measurement_ids=None,
        )

    def encode_window(self, batch: Any) -> Any:
        """One explicit encoder pass over the whole causal window (history features)."""
        return self.model.inference.encoder(batch.observation)

    def decision_features(self, batch: Any, spatial: Any) -> Any:
        """Anchor-row features of one window: a slice of the single full-window encode pass."""
        from owac.backends.n02_v12_tokens import anchor_row_indices

        rows_total = int(batch.observation.context.proprio.shape[0])
        index = anchor_row_indices(rows_total, int(batch.history))
        rows = torch.as_tensor(index, device=self.device, dtype=torch.long)
        return subset_spatial(spatial, rows)

    @staticmethod
    def bundle_rows(decision: Any) -> int:
        """Number of decision rows of one inference input (one current row per anchor)."""
        bundle = getattr(decision, "observation", decision)
        values = np.asarray(bundle.context.proprio)
        if values.ndim >= 1 and values.shape[0] > 0:
            return int(values.shape[0])
        return len(bundle.episode_ids)

    def _anchor_proprio(self, batch: Any) -> np.ndarray:
        values = np.asarray(batch.observation.context.proprio)
        return np.stack(
            [
                values[int(index) * batch.history + batch.history - 1]
                for index in range(batch.batch_size)
            ]
        )

    def _aux_weights(self) -> dict:
        aux = deep_update(DEFAULT_AUX, self.plan.config.get("aux", {}))
        return {
            "masked_raw": float(aux["lambda_raw"]),
            "causal_raw": float(aux["lambda_pred"]),
            "noise": float(aux["lambda_noise"]),
        }

    def prepare_microbatch(self, start: int, stop: int) -> tuple:
        """CPU-only preparation, used by the single bounded prefetch worker."""
        source = self.sources["train"]
        batch = self.window(source, start, stop)
        targets = source.targets(start, stop)
        prompts = self.task_prompts(
            [str(anchor["relative_path"]) for anchor in batch.anchors],
            str(self.plan.config["data"]["data_root"]),
        )
        return batch, targets, prompts

    def task_prompts(self, paths, data_root):
        instructions = self.plan.config.get("policy", {}).get("task_instructions")
        return (
            read_task_prompts(paths, data_root, instructions=instructions)
            if instructions
            else read_task_prompts(paths, data_root)
        )

    def training_bounds(self, cursor):
        source = self.sources["train"]
        if self.plan.config.get("task_adaptation", {}).get("balanced_sampling", False):
            from owac.training.n02_v12_task import (
                BalancedBlocks,
                LegacyBalancedBlocks,
                is_repair,
                source_records,
            )

            if not hasattr(self, "task_sampler"):
                sampler = BalancedBlocks if is_repair(self.plan.config) else LegacyBalancedBlocks
                self.task_sampler = sampler(
                    source_records(source), self.plan.batch_size, self.plan.seed
                )
                self._write_json("task_sampler.json", self.task_sampler.record())
            return self.task_sampler.bounds(cursor)
        from owac.frameworks.n02_full.sequence_v12 import training_anchor_order

        blocks = (source.anchor_count + self.plan.batch_size - 1) // self.plan.batch_size
        start = training_anchor_order(blocks, self.plan.seed, cursor)[0] * self.plan.batch_size
        return start, min(source.anchor_count, start + self.plan.batch_size)

    def microbatch(
        self, start: int, stop: int, *, generator: torch.Generator, prepared=None
    ) -> dict:
        """One real microbatch: official window -> chain -> frozen pi0.5 flow-matching loss."""
        from owac.backends.n02_v12_tokens import anchor_row_indices

        # Calibration is a fixed-weight object. No optimizer step may consume
        # a threshold fitted before earlier updates.
        if self.plan.config.get("online_checks", {}).get("enabled"):
            self.model.inference.calibrator = None
            self.calibration_payload = None

        batch, targets, prompts = (
            self.prepare_microbatch(start, stop) if prepared is None else prepared
        )
        proprio = self._anchor_proprio(batch)
        anchors = int(batch.batch_size)
        rows = int(batch.observation.context.proprio.shape[0])
        if rows != anchors * int(batch.history):
            raise ValueError(f"window rows {rows} != anchors {anchors} x history {batch.history}")
        anchor_rows = torch.as_tensor(
            anchor_row_indices(rows, int(batch.history)), device=self.device
        )
        lang_tokens, lang_masks, state = self.language(prompts, proprio)
        actions, mask = anchor_action_targets(targets, anchors)
        if int(mask.sum()) == 0:
            # the loader's own action mask marks an episode tail whose whole horizon is padding;
            # that is a real property of the cache, so the window is skipped and counted, never
            # turned into a fabricated target
            return None
        padded = self.bridge.prepare_control_targets(actions, device=self.device)
        noise = torch.randn(padded.shape, generator=generator, device=self.device)
        flow_time = torch.rand((anchors,), generator=generator, device=self.device)
        flow_time = (flow_time.pow(1 / 1.5) * 0.999 + 0.001).to(torch.float32)
        result = self.bridge.forward(
            observation=batch,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=padded,
            noise=noise,
            time=flow_time,
            action_mask=torch.as_tensor(mask, device=self.device),
            return_debug=bool(self.plan.config.get("task_adaptation")),
        )
        output = self.token_source.last_output
        if output is None:
            raise RuntimeError("the token source ran no v1.2 step for this microbatch")
        action_loss = result["action_loss"]
        if self.plan.config.get("task_adaptation", {}).get("objective") == "control_flow_14":
            from owac.training.n02_v12_task import control_flow_loss

            action_loss = control_flow_loss(result["v_t"], result["u_t"], mask, actions.shape[-1])
        elif self.plan.config.get("task_adaptation", {}).get("objective") == "native_flow_32":
            from owac.training.n02_v12_task import control_flow_loss

            action_loss = control_flow_loss(result["v_t"], result["u_t"], mask, padded.shape[-1])
        terms: dict[str, Any] = {"action": action_loss}
        weights = self._aux_weights()
        if weights["masked_raw"] != 0 or self.plan.config.get("aux", {}).get(
            "audit_disabled_terms", True
        ):
            terms["masked_raw"], masked_info = self._masked_raw_loss(batch, anchor_rows)
        else:
            terms["masked_raw"], masked_info = None, {"enabled": False, "reason": "weight zero"}
        if weights["causal_raw"] != 0 or self.plan.config.get("aux", {}).get(
            "audit_disabled_terms", True
        ):
            terms["causal_raw"], causal_info = self._causal_loss(
                batch, anchor_rows, anchor_rows, output, None
            )
        else:
            terms["causal_raw"], causal_info = None, {"enabled": False, "reason": "weight zero"}
        terms["noise"], noise_info = self._noise_loss(batch)
        weights = self._aux_weights()
        total = terms["action"]
        for name in ("masked_raw", "causal_raw", "noise"):
            if terms[name] is not None and weights[name] != 0.0:
                total = total + weights[name] * terms[name]
        info = {
            "weights": weights,
            "action_mask_valid": int(mask.sum()),
            "action_shape": list(actions.shape),
            "action_dim_internal": int(padded.shape[-1]),
            "flow_time": [round(float(value), 6) for value in flow_time.tolist()],
            "masked_raw": masked_info,
            "causal_raw": causal_info,
            "noise": noise_info,
            "window": {
                "start": int(start),
                "stop": int(stop),
                "rows": rows,
                "partition": batch.partition,
                "current_frame_indices": list(batch.current_frame_indices()),
                "unique_measurements": int(len(batch.measurement_ids)),
                "anchors": [str(anchor["relative_path"]) for anchor in batch.anchors],
            },
            "run_stats": result["run_stats"],
            "total": float(total.detach()),
        }
        return {
            "loss": total,
            "loss_tensors": terms,
            "terms": {
                name: (None if value is None else float(value.detach()))
                for name, value in terms.items()
            },
            "output": output,
            "batch": batch,
            "info": info,
        }

    def _masked_raw_loss(self, batch: Any, anchor_rows: torch.Tensor):
        """FCMAE masked reconstruction of the current frame; the target pixels are labels.

        The masked branch is a separate forward of the *shared* backbone with the mask applied
        inside the encoder (support bookkeeping, valid-only statistics), so no unmasked path of
        the current frame can reach the masked target.
        """
        frames = subset_frames(batch.observation, anchor_rows)
        output = self.masked(frames)
        return output.loss, {
            "scope": "masked RAW pixels only (contributing, non-saturated)",
            "mask_ratio": float(self.masked.pretrainer.mask_ratio),
            "patch_size": int(self.masked.pretrainer.patch_size),
            "valid_target_count": int(output.valid_target_count),
            "saturated_count": int(output.saturated_count),
            "shared_backbone_tensors": self.masked.sharing["bound"],
        }

    def _causal_loss(
        self,
        batch: Any,
        anchor_rows: torch.Tensor,
        current_rows: torch.Tensor,
        output: Any,
        spatial: Any,
    ):
        """Prediction of the current frame from features locked before it was observed."""
        losses, samples = [], 0
        for anchor_output in self.token_source.last_outputs:
            causal = anchor_output.causal
            if causal is None:
                continue
            energy, _ = self.model.inference.measurement.objective(
                mean=causal["mean"],
                variance=causal["variance"],
                masks=causal["valid"],
                target=causal["queries"].target,
            )
            raw_mass = causal["valid"].sum(dim=-1)
            if not bool(raw_mass.any()):
                continue
            mass = raw_mass.clamp_min(1).to(energy.dtype)
            # Candidate weights are locked before observing this target; no current-q leakage.
            candidate_loss = energy / mass
            support = causal["support"].to(energy).detach()
            support = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            losses.append((candidate_loss * support).sum(dim=-1).mean())
            samples += int(causal["valid"].sum())
        if not losses:
            return None, {"enabled": False, "reason": "no distinct causal prefix"}
        return torch.stack(losses).mean(), {
            "enabled": True,
            "scored_samples": samples,
            "leak_guard": "previous spatial features/state only; current values are labels",
            "execution_revision": EXECUTION_REVISION,
            "gradient_boundary": "truncated recurrent graph, including last prefix corrector",
            "candidate_aggregation": "previous support, detached before observing target",
        }

    def _noise_loss(self, batch: Any):
        """Noise calibration, enabled only when the cache really declares a noise model."""
        declared = False
        reason = "the official cache declares no sensor noise model"
        for frame in batch.observation.frames.values():
            available = dict(getattr(frame, "metadata_available", {}) or {})
            if available.get("noise_model"):
                declared = True
            if dict(getattr(frame, "metadata", {}) or {}).get("variance_is_estimate"):
                reason = "only a fitted variance estimate exists, not a declared noise model"
        weight = float(self._aux_weights()["noise"])
        if not declared:
            return None, {"enabled": False, "weight": weight, "reason": reason}
        raise NotImplementedError(
            "a declared noise model appeared in the cache; implement the censored noise term "
            "before enabling lambda_noise"
        )

    # -- correctness checks -------------------------------------------------------------

    def correctness_checks(self) -> dict:
        """Only the checks that decide correctness: units, finiteness, causality, isolation."""
        plan = self.plan
        train = self.sources["train"]
        targets = train.targets(0, plan.batch_size)
        batch = self.window(train, 0, plan.batch_size)
        spec = train.window_options
        units = {}
        for camera_id, frame in batch.observation.frames.items():
            values = np.asarray(frame.values)
            units[camera_id] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "min": float(values.min()),
                "max": float(values.max()),
                "finite": bool(np.isfinite(values).all()),
                "scale_note": "loader units kept; this entry does not rescale RAW values",
            }
        actions = np.asarray(targets["actions"])
        mask = np.asarray(targets["action_mask"])
        causal: list[bool] = []
        for source in self.sources.values():
            for anchor in range(min(4, source.anchor_count)):
                entry = source.anchor(anchor)
                indices = np.asarray(entry.history_indices)
                causal.append(bool(indices[-1] == entry.current_frame_index))
                causal.append(bool(np.all(indices <= entry.current_frame_index)))
                causal.append(bool(np.all(np.diff(indices) >= 0)))
        checks: dict[str, Any] = {
            "units_and_shapes": {
                "dataset_format": plan.config["data"]["dataset_format"],
                "partition": batch.partition,
                "cameras": list(batch.observation.frames),
                "camera_count": len(batch.observation.frames),
                "rows": batch.row_count,
                "history": int(batch.history),
                "anchors": int(batch.batch_size),
                "frame_index": [int(value) for value in np.asarray(batch.frame_index).tolist()],
                "current_frame_indices": list(batch.current_frame_indices()),
                "channel_values": units,
                "action_shape_unpadded": list(actions.shape),
                "action_control_dim": int(actions.shape[-1]),
                "action_offset": int(spec.action_offset),
                "action_mask_valid": int(mask.sum()),
                "action_mask_shape": list(mask.shape),
                "normalisation": (
                    "official loader statistics only; the training entry never normalises or "
                    "de-normalises (prepare_control_targets pads to 32 dims)"
                ),
                "finite": all(item["finite"] for item in units.values()),
            },
            "causality": {
                "windows_checked": len(causal) // 3,
                "all_checks_passed": bool(all(causal)),
                "meaning": (
                    "history rows are earlier-or-equal real frame indices of their anchor and "
                    "never a future frame"
                ),
            },
        }
        train_ids = set(self.window(train, 0, 6).measurement_ids)
        dev_ids = set(self.window(self.sources["dev"], 0, 6).measurement_ids)
        calibration_ids = set(self.window(self.sources["calibration"], 0, 6).measurement_ids)
        checks["partition_isolation"] = {
            "train_dev_overlap": len(train_ids & dev_ids),
            "train_calibration_overlap": len(train_ids & calibration_ids),
            "dev_calibration_overlap": len(dev_ids & calibration_ids),
            "train_measurements": len(train_ids),
            "dev_measurements": len(dev_ids),
            "calibration_measurements": len(calibration_ids),
        }
        checks["episode_reset"] = self._check_episode_reset()
        return checks

    def _check_episode_reset(self) -> dict:
        """Real reset: selected rows lose their state, unselected rows stay bit-identical."""
        from owac.frameworks.n02_full.inference_v12 import empty_v12_state

        rows = 2 * self.plan.history
        state = empty_v12_state(
            batch=rows,
            k=self.model.inference.k,
            n_slots=self.model.inference.n_slots,
            slot_dim=self.model.inference.slot_dim,
            device=self.device,
        )
        state.slot_hidden = state.slot_hidden + torch.arange(rows, device=self.device).view(
            rows, 1, 1, 1
        )
        state.existence = state.existence + 1.0
        before = state.slot_hidden.clone()
        reset = self.model.reset([True] + [False] * (rows - 1), state)
        return {
            "selected_rows_zeroed": bool((reset.slot_hidden[0] == 0).all()),
            "unselected_rows_bit_identical": bool(torch.equal(reset.slot_hidden[1:], before[1:])),
            "version_incremented": int(reset.version) > int(state.version),
        }

    def prior_wiring_check(self) -> dict:
        """The declared prior must change ``log_p0`` and ``q`` on the same real measurement."""
        train = self.sources["train"]
        k = self.model.inference.k
        uniform = torch.full((1, k), 1.0 / k)
        biased = torch.full((1, k), 0.1 / (k - 1))
        biased[0, 0] = 0.9
        results = {}
        for name, prior in (("uniform", uniform), ("biased", biased)):
            batch = self.window(train, 0, 1)
            decision = self.anchor_bundle(batch, prior=prior)
            state = self.model.build_state(batch=self.bundle_rows(decision), device=self.device)
            with torch.no_grad():
                output = self.model.step(decision, state)
            results[name] = {
                "log_p0": [round(float(value), 6) for value in output.log_p0[0].tolist()],
                "q": [round(float(value), 6) for value in output.q[0].tolist()],
                "check_status": str(getattr(output.check, "status", None)),
                "check_valid": bool(getattr(output.check, "valid", False)),
                "calibration_version": str(getattr(output.check, "calibration_version", None)),
            }
        delta_p0 = float(
            np.abs(
                np.array(results["uniform"]["log_p0"]) - np.array(results["biased"]["log_p0"])
            ).max()
        )
        delta_q = float(
            np.abs(np.array(results["uniform"]["q"]) - np.array(results["biased"]["q"])).max()
        )
        return {
            "max_abs_log_p0_difference": delta_p0,
            "max_abs_q_difference": delta_q,
            "prior_reaches_log_p0": bool(delta_p0 > 0.0),
            "prior_changes_support": bool(delta_q > 0.0),
            "note": "same real window, same weights; only the declared prior differs",
            "runs": results,
        }

    def role_gradient_l1(self) -> dict:
        """Per-role gradient L1 after backward (finite, non-zero counts)."""
        stats: dict[str, Any] = {}
        for role, parameters in self.parameter_groups.items():
            if not parameters:
                continue
            values = [
                parameter.grad.detach().abs()
                for parameter in parameters
                if parameter.grad is not None
            ]
            total = float(sum(float(value.sum()) for value in values)) if values else 0.0
            stats[role] = {
                "l1": total,
                "finite": bool(all(bool(torch.isfinite(value).all()) for value in values)),
                "tensors_with_grad": len(values),
                "tensors": len(parameters),
            }
        return stats

    def gradient_wiring_check(self, stats: dict) -> dict:
        reached = [role for role, item in stats.items() if item["l1"] > 0.0]
        return {
            "roles": stats,
            "roles_with_nonzero_gradient": reached,
            "frozen_parameters_with_grad": sorted(self.frozen_with_grad),
            "hard_paths": self.model.gradient_paths(),
        }

    # -- calibration --------------------------------------------------------------------

    def calibration_fingerprint(self):
        from owac.frameworks.n02_full.calibration_v12 import parameter_fingerprint

        return parameter_fingerprint(
            self.model, getattr(self, "backend_adapters", {}).values(), self.bridge.bridge_modules()
        )

    def restore_fixed_calibration(self, payload):
        from owac.frameworks.n02_full.calibration_v12 import restore_calibration

        self.model.inference.calibrator = None
        self.calibration_payload = None
        if payload is None:
            raise ValueError("r11 runtime requires fixed-weight calibration in the checkpoint")
        calibration = restore_calibration(
            payload,
            fingerprint=self.calibration_fingerprint(),
            source_identity=self.identity["calibration"],
        )
        self.model.inference.calibrator = calibration
        self.calibration_payload = payload

    @torch.no_grad()
    def fit_fixed_calibration(self):
        from owac.frameworks.n02_full.calibration_v12 import pack_calibration
        from owac.frameworks.n02_full.checks_v12 import calibrate_windowed, v12_viewing_rule

        inference = self.model.inference
        inference.calibrator = None
        self.calibration_payload = None
        source = self.sources["calibration"]
        candidates = [
            i for i in range(source.anchor_count) if source.anchor(i).current_frame_index >= 1
        ]
        count = min(self.plan.calibration_windows, len(candidates))
        if count < 2:
            raise ValueError("fixed-weight calibration requires two causal calibration windows")
        indices = [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]
        rule = v12_viewing_rule(inference.query_stride, inference.encoder.camera_ids)
        was_training = self.model.training
        self.model.eval()
        try:
            groups = [self._residual_views_of_window(source, i, rule) for i in indices]
            calibration = calibrate_windowed(
                groups,
                partition="calibration",
                rule=rule,
                calibration_ids=[
                    m for i in indices for m in source.window(i, i + 1).measurement_ids
                ],
                test_ids=self.sources["dev"].window(0, 1).measurement_ids,
                version="n02-fixed-weight-r11-v1",
            )
            self.calibration_payload = pack_calibration(
                calibration,
                fingerprint=self.calibration_fingerprint(),
                source_identity=self.identity["calibration"],
            )
            inference.calibrator = calibration
            self._write_json("fixed_weight_calibration.json", self.calibration_payload)
            return self.calibration_payload
        finally:
            self.model.train(was_training)

    def calibration_pass(self) -> dict:
        """Fit the windowed absolute check on the independent calibration partition."""
        from owac.frameworks.n02_full.checks_v12 import calibrate_windowed, v12_viewing_rule

        source = self.sources["calibration"]
        windows = min(self.plan.calibration_windows, len(source.anchor_table))
        if windows < 2:
            return {"status": "not_run", "reason": "fewer than two calibration windows"}
        inference = self.model.inference
        # The frozen rule names exactly the (scale, frame, region) views a judged sequence must
        # contain, so the calibration sequences are built with the frame axis fixed at the
        # sequence origin: one view per camera per decision moment, the same rule for every
        # window.  The real source frame index of each row stays in the step diagnostics.
        rule = v12_viewing_rule(inference.query_stride, inference.encoder.camera_ids)
        from owac.evaluation.n02_full.v12 import anchor_indices

        groups, calibration_ids = [], []
        with torch.no_grad():
            for index in anchor_indices(source, windows):
                views = self._residual_views_of_window(source, int(index), rule)
                if views:
                    groups.append(views)
                    calibration_ids.extend(
                        source.window(int(index), int(index) + 1).measurement_ids
                    )
        if len(groups) < 2 or not groups[0]:
            return {"status": "not_run", "reason": "the calibration residual views are empty"}
        calibration = calibrate_windowed(
            groups,
            partition="calibration",
            rule=rule,
            calibration_ids=calibration_ids,
            test_ids=self.sources["dev"].window(0, self.plan.dev_windows).measurement_ids,
        )
        # Training changes the predictor weights: this threshold is a diagnostic snapshot,
        # not a valid threshold for subsequent optimizer updates. Eval fits at fixed weights.
        return {
            "status": "fitted",
            "sequences": len(groups),
            "threshold": float(calibration.calibrator.threshold()),
            "quantile": float(calibration.quantile),
            "version": str(calibration.version),
            "partition": str(calibration.partition),
            "viewing_rule_hash": str(calibration.viewing_rule_hash),
            "frame_axis_note": "relative check axis 0; actual decoder source time is unchanged",
            "prediction_version": "causal_pre_locked",
            "execution_revision": EXECUTION_REVISION,
            "scale": rule.scales[0],
            "regions": list(rule.regions),
            "verdicts": self._calibration_verdicts(calibration, groups, rule),
        }

    def _calibration_verdicts(self, calibration: Any, groups: list, rule: Any) -> dict:
        """Real three-state verdicts from the fitted threshold on calibration and dev views."""
        source = self.sources["dev"]
        from owac.evaluation.n02_full.v12 import anchor_indices

        dev_groups = []
        with torch.no_grad():
            for index in anchor_indices(source, self.plan.dev_windows):
                dev_groups.append(self._residual_views_of_window(source, index, rule))
        verdicts: dict[str, Any] = {}
        for name, items in (("calibration", groups), ("dev", dev_groups)):
            tally: dict[str, int] = {}
            for group in items:
                if not group:
                    continue
                verdict = calibration.assess_views(group, rule=rule)
                status = str(getattr(verdict, "status", "missing"))
                tally[status] = tally.get(status, 0) + 1
            verdicts[name] = {
                "sequences": len(items),
                "by_status": tally,
                "invalid_no_prefix": sum(not group for group in items),
                "meaning": (
                    "three-state verdicts of the frozen calibrator; 'compatible' means no "
                    "residual window exceeded the frozen threshold, not that the model is right"
                ),
            }
        verdicts["per_step_check"] = (
            "uncalibrated on purpose: see the comment in calibration_pass; an uncalibrated check "
            "returns AbsoluteCheckResult.invalid and is never counted as a zero"
        )
        return verdicts

    def _residual_views_of_window(self, source: Any, index: int, rule: Any) -> list:
        """Locked residual views of one window's current frame, in the rule's own frame axis."""
        from owac.evaluation.n02_full.v12 import locked_residual_groups
        from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

        batch = self.window(source, index, index + 1)
        replay = replay_inputs(self.model, window_inputs(batch))
        if replay["rows"][-1].causal is None:
            return []
        return locked_residual_groups(
            self.model.inference,
            replay["inputs"][-1],
            replay["pre_states"][-1],
            replay["rows"][-1].q,
        )["groups"][0]

    # -- save / resume ------------------------------------------------------------------

    def checkpoint_path(self, step: int) -> Path:
        return self.run_dir / f"checkpoint_{int(step):07d}.pt"

    def save(self, step: int, *, cursor: int, dev_reference: dict | None = None) -> Path:
        from owac.backends.low_rank import adapter_state

        path = self.checkpoint_path(step)
        if self.plan.config.get("online_checks", {}).get("enabled"):
            if getattr(self, "calibration_payload", None) is None:
                self.fit_fixed_calibration()
            # Refuse stale externally modified weights even outside the main loop.
            self.restore_fixed_calibration(self.calibration_payload)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "execution_revision": EXECUTION_REVISION,
                "training_contract": training_contract(self.plan, self.identity),
                "task_protocol": self.plan.config.get("task_adaptation", {}).get("protocol"),
                "training_generator": getattr(self, "train_generator", None).get_state()
                if hasattr(self, "train_generator")
                else None,
                "step": int(step),
                "cursor": int(cursor),
                "stage": STAGE,
                "run_id": self.plan.run_id,
                "config": self.plan.config,
                "config_path": str(self.plan.config_path),
                "seed": int(self.plan.seed),
                "model": self.model.state_dict(),
                "masked_branch": {
                    name: value
                    for name, value in self.masked.state_dict().items()
                    if not name.startswith("pretrainer.fcmae.encoder.")
                },
                "bridge_projection": [
                    module.state_dict() for module in self.bridge.bridge_modules()
                ],
                "backend_adapters": adapter_state(getattr(self, "backend_adapters", {})),
                "optimizer": self.optimizer.state_dict(),
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if self.device.type == "cuda" else None,
                    "numpy": np.random.get_state(),
                },
                "identity": self.identity,
                "load_audit": self.load_audit,
                "dev_reference": dev_reference,
                "fixed_weight_calibration": getattr(self, "calibration_payload", None),
            },
            path.with_suffix(".pt.tmp"),
        )
        os.replace(path.with_suffix(".pt.tmp"), path)
        if dev_reference is not None:
            self._write_json(
                path.stem + ".metrics.json",
                {
                    "step": int(step),
                    "action_error": dev_reference.get("action_error"),
                    "sampled_control_mse": dev_reference.get("sampled_control_mse"),
                    **{
                        key: dev_reference[key]
                        for key in (
                            "sampled_full_control_mse",
                            "sampled_metric",
                            "executed_prefix_proxy",
                            "execution_contract",
                            "calibration_weights_sha256",
                            "online_checks",
                        )
                        if key in dev_reference
                    },
                    **(
                        {
                            "task_control": dev_reference["task_control"],
                            "task_dev_manifest_sha256": dev_reference["task_dev_manifest_sha256"],
                        }
                        if "task_control" in dev_reference
                        else {}
                    ),
                },
            )
        self.log(f"saved checkpoint {path} ({path.stat().st_size / 1e6:.1f} MB) at step {step}")
        return path

    def load_for_resume(self, path: Path) -> tuple[int, int, dict]:
        from owac.backends.low_rank import restore_adapters

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("format")) != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        if payload.get("execution_revision") != EXECUTION_REVISION:
            raise ValueError(
                "legacy execution path: use weights only for a labelled probe; "
                "do not resume as corrected training"
            )
        if payload.get("training_contract") != training_contract(self.plan, self.identity):
            raise ValueError(
                "resume training contract/data identity mismatch; weights were not loaded"
            )
        self.model.load_state_dict(payload["model"])
        current = self.masked.state_dict()
        current.update(payload["masked_branch"])
        self.masked.load_state_dict(current)
        modules = self.bridge.bridge_modules()
        saved = payload.get("bridge_projection") or []
        if len(modules) != len(saved):
            raise ValueError(
                f"checkpoint carries {len(saved)} bridge projection(s) but this run builds "
                f"{len(modules)}"
            )
        for module, state in zip(modules, saved, strict=True):
            module.load_state_dict(state)
        restore_adapters(getattr(self, "backend_adapters", {}), payload.get("backend_adapters", {}))
        if self.plan.config.get("online_checks", {}).get("enabled"):
            self.restore_fixed_calibration(payload.get("fixed_weight_calibration"))
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng"]["torch"])
        if self.device.type == "cuda" and payload["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(payload["rng"]["cuda"])
        np.random.set_state(payload["rng"]["numpy"])
        return int(payload["step"]), int(payload.get("cursor", 0)), payload

    def load_weights_for_probe(self, path: Path) -> dict:
        """Explicit read-only cross-revision probe; never restores optimizer or RNG.

        r5 weights were trained with unpadded state text. They can be inspected
        with the repaired input contract, but cannot resume corrected training.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("execution_revision") not in (
            "n02-v12-p2-train-r5",
            EXECUTION_REVISION,
        ):
            raise ValueError("only r5/current checkpoints are supported by this labelled probe")
        if payload.get("identity") != self.identity or payload.get("seed") != self.plan.seed:
            raise ValueError("probe data identity/seed mismatch")
        saved_config = payload["config"]
        for key in ("model", "representation", "data"):
            if saved_config.get(key) != self.plan.config.get(key):
                raise ValueError(f"probe {key} contract mismatch")
        for key in (
            "suite",
            "checkpoint",
            "tokenizer",
            "token_dim",
            "tokens_per_camera",
            "freeze_backend",
            "backend_adapter",
            "freeze_encoder",
            "freeze_frontend",
        ):
            if saved_config["policy"].get(key) != self.plan.config["policy"].get(key):
                raise ValueError(f"probe backend {key} mismatch")
        self.model.load_state_dict(payload["model"], strict=True)
        masked = self.masked.state_dict()
        masked.update(payload["masked_branch"])
        self.masked.load_state_dict(masked, strict=True)
        modules, saved = self.bridge.bridge_modules(), payload.get("bridge_projection") or []
        if len(modules) != len(saved):
            raise ValueError("probe bridge projection count mismatch")
        for module, state in zip(modules, saved, strict=True):
            module.load_state_dict(state, strict=True)
        from owac.backends.low_rank import restore_adapters

        restore_adapters(getattr(self, "backend_adapters", {}), payload.get("backend_adapters", {}))
        return payload

    # -- evaluation ---------------------------------------------------------------------

    def initialise_repair_weights(self) -> dict | None:
        """Explicit transfer from failed RAW r5; reset readout, optimizer and training step."""
        settings = self.plan.config.get("initialisation")
        if not settings:
            return None
        if settings.get("kind") in ("raw_representation_r8", "task_repair_r11"):
            return self.initialise_representation_weights(settings)
        path = Path(settings["checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("execution_revision") != "n02-v12-p2-train-r5"
        ):
            raise ValueError("repair transfer only accepts declared r5 weights")
        for key in ("representation", "platform"):
            if payload["config"].get(key) != self.plan.config.get(key):
                raise ValueError(f"repair transfer {key} mismatch")
        for key in ("dataset_format", "data_root", "stats_path", "manifest_seed"):
            if payload["config"]["data"].get(key) != self.plan.config["data"].get(key):
                raise ValueError(f"repair transfer data {key} mismatch")
        current, saved = self.model.state_dict(), payload["model"]
        selected = {k: v for k, v in saved.items() if not k.startswith("inference.readout.")}
        expected = {k for k in current if not k.startswith("inference.readout.")}
        if set(selected) != expected:
            raise ValueError("repair transfer non-readout parameter keys differ")
        current.update(selected)
        self.model.load_state_dict(current, strict=True)
        if self.optimizer.state:
            raise ValueError("repair transfer must start with empty optimizer state")
        receipt = {
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_revision": payload["execution_revision"],
            "source_step": payload["step"],
            "transferred_tensors": len(selected),
            "readout": "fresh Perceiver + new local spatial skip",
            "optimizer": "fresh",
            "training_step": 0,
            "historical_exposure": "r5 trained the full train partition; not an unseen-data claim",
        }
        self._write_json("initialisation.json", receipt)
        self.log("repair transfer: " + json.dumps(receipt))
        return receipt

    def initialise_representation_weights(self, settings):
        """Only a separately accepted RAW representation may enter backend adaptation."""
        from owac.training.n02_v12_representation import FORMAT

        path = Path(settings["representation_checkpoint"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != settings["representation_sha256"]:
            raise ValueError("representation checkpoint hash changed")
        representation = torch.load(path, map_location="cpu", weights_only=False)
        if representation.get("format") != FORMAT or not representation["dev"].get(
            "representation_gate_passed", False
        ):
            raise ValueError("RAW representation gate has not passed")
        for partition in ("train", "dev"):
            saved_identity = dict(representation["data_identity"][partition])
            current_identity = dict(self.identity[partition])
            # Frame-pool loading uses one anchor; action batches group the same anchors.
            # All split, file-content, anchor, history and normalization identities stay exact.
            saved_identity.pop("batch_size", None)
            current_identity.pop("batch_size", None)
            if saved_identity != current_identity:
                raise ValueError(
                    f"representation and action-training {partition} partitions differ"
                )
        source = Path(settings["checkpoint"])
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_sha != settings["checkpoint_sha256"]:
            raise ValueError("representation source checkpoint hash changed")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload.get("execution_revision") != EXECUTION_REVISION:
            raise ValueError("representation repair requires the declared causal-execution source")
        fresh_readout = (
            {k: v.detach().clone() for k, v in self.model.inference.readout.state_dict().items()}
            if settings.get("reset_readout", False)
            else None
        )
        if settings.get("kind") == "task_repair_r11":
            current = self.model.state_dict()
            missing = set(current) - set(payload["model"])
            allowed = {
                "inference.readout." + n for n in ("scale_logits", "local_gain", "global_gain")
            }
            if missing != allowed or set(payload["model"]) - set(current):
                raise ValueError("undeclared r11 transfer parameter changes")
            current.update(payload["model"])
            self.model.load_state_dict(current, strict=True)
        else:
            self.model.load_state_dict(payload["model"], strict=True)
        self.model.inference.encoder.load_state_dict(representation["encoder"], strict=True)
        if fresh_readout is not None:
            self.model.inference.readout.load_state_dict(fresh_readout, strict=True)
        if self.optimizer.state:
            raise ValueError("backend adaptation must start with fresh optimizer state")
        if settings.get("continue_data_order", False):
            if "cursor" not in payload or payload.get("training_generator") is None:
                raise ValueError("continuing data order requires the saved cursor and flow RNG")
            self.initial_data_cursor = int(payload["cursor"])
            self.initial_training_generator = payload["training_generator"]
        receipt = {
            "kind": settings.get("kind", "raw_representation_r8"),
            "source": str(source),
            "source_sha256": source_sha,
            "representation": str(path),
            "representation_sha256": actual,
            "representation_updates": representation["step"],
            "representation_dev": representation["dev"],
            "encoder_frozen": self.plan.config["policy"].get("freeze_encoder", False),
            "backend_adapter": self.backend_adapter_report,
            "readout": "fresh weights with r8 balanced spatial conditioning"
            if fresh_readout is not None
            else "declared source spatial weights with r8 balanced spatial conditioning",
            "optimizer": "fresh",
            "training_step": 0,
            "initial_data_cursor": getattr(self, "initial_data_cursor", 0),
            "continue_flow_rng": hasattr(self, "initial_training_generator"),
        }
        self._write_json("initialisation.json", receipt)
        return receipt

    def dev_batch(self) -> list[tuple[Any, Any, list[str]]]:
        """The fixed development windows, read once and reused by every measurement."""
        source = self.sources["dev"]
        from owac.evaluation.n02_full.v12 import anchor_indices, sample_identity

        windows = []
        selected, skipped = [], []
        if self.plan.config.get("task_adaptation"):
            from owac.training.n02_v12_task import (
                PROTOCOL,
                REPAIR_PROTOCOL,
                dev_selection,
                is_repair,
                source_records,
            )

            self.task_dev_manifest = dev_selection(
                source_records(source),
                protocol=REPAIR_PROTOCOL if is_repair(self.plan.config) else PROTOCOL,
            )
            self._write_json("task_dev_manifest.json", self.task_dev_manifest)
            starts = [r["index"] for r in self.task_dev_manifest["rows"]]
        else:
            starts = anchor_indices(source, self.plan.dev_windows)
        for start in starts:
            for offset in range(source.anchor_count):
                index = (start + offset) % source.anchor_count
                if index in selected:
                    continue
                targets = source.targets(index, index + 1)
                if np.asarray(targets["action_mask"], dtype=bool).any():
                    break
                skipped.append(index)
            else:
                raise ValueError("dev partition has too few windows with valid action targets")
            selected.append(index)
            batch = self.window(source, index, index + 1)
            windows.append(
                (
                    batch,
                    targets,
                    self.task_prompts(
                        [str(anchor["relative_path"]) for anchor in batch.anchors],
                        str(self.plan.config["data"]["data_root"]),
                    ),
                )
            )
        self._write_json(
            "dev_windows.json",
            {
                "selection": "evenly spaced anchors; advance past all-padding targets",
                "indices": selected,
                "skipped_all_padding": sorted(set(skipped)),
                "identities": [sample_identity(source, index) for index in selected],
            },
        )
        return windows

    @torch.no_grad()
    def evaluate_dev(
        self,
        windows: list,
        *,
        noise: torch.Tensor,
        time_values: torch.Tensor,
        keep_prediction: bool = False,
    ) -> dict:
        """Teacher-forced velocity MSE, not sampled control-action error or success."""
        errors, predictions, samples, diversity = [], [], [], []
        task_rows = []
        executed_rows = []
        check_rows = []
        was_training = self.model.training
        self.model.eval()
        if self.plan.config.get("online_checks", {}).get("enabled"):
            self.fit_fixed_calibration()
        for index, (batch, targets, prompts) in enumerate(windows):
            proprio = self._anchor_proprio(batch)
            lang_tokens, lang_masks, state = self.language(prompts, proprio)
            raw_actions, raw_mask = anchor_action_targets(
                targets, len(targets["padding_contribution"])
            )
            actions = self.bridge.prepare_control_targets(raw_actions, device=self.device)
            mask = torch.as_tensor(raw_mask, device=self.device)
            if not bool(mask.any()):
                raise ValueError(
                    "fixed dev window has no valid action target; cannot score zero error"
                )
            result = self.bridge.forward(
                observation=batch,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                actions=actions,
                noise=noise[index : index + 1],
                time=time_values[index : index + 1],
                action_mask=mask,
                return_debug=True,
            )
            if self.plan.config.get("online_checks", {}).get("enabled"):
                output = self.token_source.last_output
                check_rows.append(
                    {
                        "window": index,
                        "expected_causal": int(batch.current_frame_indices()[0]) > 0,
                        "status": output.check.status,
                        "valid": output.check.valid,
                        "explanation": output.check.explanation,
                        "revision": output.diagnostics.get("revision"),
                    }
                )
            errors.append(float(result["action_loss"]))
            if self.plan.config.get("loop", {}).get("sampled_dev", False):
                from owac.core.encoder_contract_n02 import VisualTokens

                values = torch.cat([o.tokens for o in self.token_source.last_outputs])
                visual = VisualTokens(
                    values,
                    torch.ones(values.shape[:2], dtype=torch.bool, device=values.device),
                    tuple(self.plan.config["model"]["camera_ids"]),
                )
                sample = self.bridge.predict_action_from_visual_tokens(
                    visual, lang_tokens, lang_masks, state, noise=noise[index : index + 1]
                )["control_actions"]
                sample = torch.as_tensor(sample, device=self.device)
                error = (sample - actions[..., : sample.shape[-1]]).square().mean(-1)
                samples.append(float(error[mask].mean()))
                if self.plan.config.get("task_adaptation"):
                    from owac.training.n02_v12_task import control_metrics

                    task_rows.append(
                        control_metrics(
                            sample.float().cpu().numpy(),
                            actions[..., : sample.shape[-1]].float().cpu().numpy(),
                            mask.cpu().numpy(),
                            self.task_dev_manifest["rows"][index],
                        )
                    )
                    if self.plan.config.get("execution"):
                        from owac.runtime.n02_control import ControlProtocol
                        from owac.training.n02_v12_task import execution_proxy_metrics

                        if not hasattr(self, "control_stats"):
                            self.control_stats = json.loads(
                                Path(self.plan.config["data"]["stats_path"]).read_text()
                            )["norm_stats"]
                        executed_rows.append(
                            execution_proxy_metrics(
                                sample.float().cpu().numpy(),
                                raw_actions,
                                raw_mask,
                                self.task_dev_manifest["rows"][index],
                                proprio,
                                self.control_stats,
                                ControlProtocol.parse(self.plan.config["execution"]),
                            )
                        )
                diversity.append(
                    float(
                        values.float().std(dim=1).mean()
                        / values.float().square().mean().sqrt().clamp_min(1e-12)
                    )
                )
            if keep_prediction:
                predictions.append(result["v_t"].detach().float().cpu())
        record = {
            "action_error": float(np.mean(errors)),
            "metric": "teacher_forced_flow_velocity_mse_32",
            "capability_evidence": False,
            "windows": len(errors),
            "per_window": [round(value, 6) for value in errors],
            "anchor_identities": [list(batch.anchors) for batch, _, _ in windows],
            "noise": "fixed normal draw (declared seed), reused by every dev measurement",
            "flow_time": "fixed beta(1.5,1) draw with the same seed",
        }
        if keep_prediction:
            record["prediction"] = torch.stack(predictions)
        if check_rows:
            record["online_checks"] = {
                "rows": check_rows,
                "valid": sum(row["valid"] for row in check_rows),
                "invalid": sum(not row["valid"] for row in check_rows),
                "expected_causal": sum(row["expected_causal"] for row in check_rows),
                "revisions": sum(
                    bool(row["revision"] and row["revision"]["triggered"]) for row in check_rows
                ),
            }
        if task_rows:
            from owac.training.n02_v12_task import aggregate_control

            metrics = aggregate_control(task_rows)
            record.update(
                task_control=metrics,
                task_per_window=task_rows,
                task_dev_manifest_sha256=self.task_dev_manifest["sha256"],
            )
        if executed_rows:
            record.update(
                executed_prefix_proxy=aggregate_control(executed_rows),
                execution_contract=self.plan.config["execution"],
                calibration_weights_sha256=self.calibration_payload["weights_sha256"],
                sampled_full_control_mse=metrics["balanced_mse"],
                sampled_metric="balanced_sampled_prefix14_mse"
                if self.plan.config["execution"]["replan_interval"] == 10
                else "balanced_sampled_control14_mse",
            )
        if samples:
            record.update(
                sampled_control_mse=(
                    metrics["balanced_prefix_mse"]
                    if executed_rows and self.plan.config["execution"]["replan_interval"] == 10
                    else metrics["balanced_mse"]
                )
                if task_rows
                else float(np.mean(samples)),
                sampled_per_window=samples,
                spatial_diversity_per_window=diversity,
            )
            self._write_json(
                "dev_sampled_latest.json", {k: v for k, v in record.items() if k != "prediction"}
            )
            self.log(
                f"sampled dev control MSE={record['sampled_control_mse']:.5f}; "
                f"spatial diversity={np.mean(diversity):.4f}"
            )
        self.model.train(was_training)
        return record

    # -- status -------------------------------------------------------------------------

    def initialise_status(self, *, activity: str) -> None:
        """Write the complete frozen field set once, so the page never reads a half schema."""
        self.status(
            state="initialising",
            activity=activity,
            step=0,
            step_seconds=None,
            loss=None,
            dev_action_error=None,
            checkpoint=None,
            preview_files=None,
            metric_kind="sampled_control"
            if self.plan.config.get("loop", {}).get("monitor_metric") == "sampled_control_mse"
            else "action_proxy",
            metric_label="前10步动作误差"
            if self.plan.config.get("execution", {}).get("replan_interval") == 10
            else "完整采样动作误差"
            if self.plan.config.get("loop", {}).get("monitor_metric") == "sampled_control_mse"
            else "训练代理误差",
        )

    def status(self, **fields: Any) -> None:
        """Write the frozen status fields; only the identity block gets defaults.

        Data fields (step, loss, checkpoint, previews) are written only when a caller passes
        them, so a later heartbeat-style update cannot erase what an earlier one recorded.
        """
        fields.setdefault("run_id", self.plan.run_id)
        fields.setdefault("stage", STAGE)
        fields.setdefault("arm", self.plan.arm)
        fields.setdefault("platform", self.plan.platform)
        fields.setdefault("seed", int(self.plan.seed))
        fields.setdefault("target_steps", int(self.plan.max_updates))
        fields.setdefault("gpu", self._gpu_field())
        fields.setdefault(
            "eval",
            {
                "planned": 100,
                "completed": 0,
                "success": 0,
                "task_failure": 0,
                "infrastructure_error": 0,
            },
        )
        fields.setdefault("log_tail", self.log.tail())
        fields.setdefault("state", "initialising")
        fields.setdefault("activity", "")
        self.update_status(self.run_dir, **fields)

    def _gpu_field(self) -> dict:
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": str(self.device),
            "name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
        }

    # -- previews -----------------------------------------------------------------------

    def write_previews(self, batch: Any, output: Any) -> dict:
        """Real RAW display frame plus the real encoder attention overlay."""
        if self.plan.previews <= 0:
            return {"status": "disabled"}
        from owac.evaluation.n02_full import v12_viz
        from owac.frameworks.n02_full.sequence_v12 import window_inputs

        files = []
        try:
            inp = window_inputs(batch)[-1]
            camera = next(iter(inp.observation.frames))
            frame = inp.observation.frames[camera]
            values = np.asarray(frame.values)
            if not hasattr(self, "display_white_level"):
                self.display_white_level = max(float(np.quantile(values, 0.99)), 1e-8)
            labels = (
                f"episode={inp.observation.episode_ids[0]}; camera={camera}; "
                f"frame={inp.frame_index[0]}; iteration={self.model.inference.rounds}; "
                f"model step={getattr(self, 'current_step', 0)}; {EXECUTION_REVISION}"
            )
            capture = {
                "model": self.model,
                "output": output,
                "spatial": output.state.geometry[0].spatial,
                "aux": output.diagnostics["readout_aux"],
                "queries": self.model.inference.build_queries(inp, batch=1),
                "stride": self.model.inference.query_stride,
            }
            attention = v12_viz._attention_maps(capture, None)
            directory = self.run_dir / "preview"
            directory.mkdir(exist_ok=True)
            figures = {
                "raw": v12_viz.figure_raw_display(
                    frame,
                    camera_id=camera,
                    labels=labels,
                    title="fixed display from first preview, same-source RGB",
                    gamma=2.2,
                    gains=[1.0, 1.0, 1.0],
                    zoom_box=(0, values.shape[-2] // 2, 0, values.shape[-1] // 2),
                    white_level=self.display_white_level,
                ),
                "attention": v12_viz.figure_attention(
                    frame,
                    attention,
                    camera_id=camera,
                    labels=labels,
                    title="real Perceiver cross-attention",
                    white_level=self.display_white_level,
                ),
                "rounds": v12_viz.figure_rounds(
                    output,
                    labels=labels,
                    title="support and corrections",
                    rounds=self.model.inference.rounds,
                ),
            }
            for name, svg in figures.items():
                path = directory / f"{name}_{camera}.svg"
                path.write_bytes(svg)
                files.append(str(path.relative_to(self.run_dir)))
            return {
                "status": "written",
                "files": files,
                "display_white_level": self.display_white_level,
            }
        except Exception as error:  # noqa: BLE001 - preserve the error and checkpoint
            return {"status": "failed", "reason": f"{type(error).__name__}: {error}"}

    # -- main loop ----------------------------------------------------------------------

    def train(self) -> dict:
        from owac.runtime.n02_v12_status import append_history

        plan = self.plan
        log = self.log
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        dev_windows = self.dev_batch()
        dev_generator = torch.Generator(device=self.device).manual_seed(int(plan.seed) + 17)
        action_horizon = int(plan.config["model"].get("action_horizon", 50))
        dev_noise = torch.randn(
            (max(len(dev_windows), 1), action_horizon, ACTION_DIM_PADDED),
            generator=dev_generator,
            device=self.device,
        )
        dev_time = (
            torch.rand(
                (max(len(dev_windows), 1),), generator=dev_generator, device=self.device
            ).pow(1 / 1.5)
            * 0.999
            + 0.001
        )
        # These weights change on the next update; no fitted threshold stays installed.
        self.checks["calibration"] = (
            {"status": "fitted_during_each_fixed_weight_dev_pass"}
            if plan.config.get("online_checks", {}).get("enabled")
            else self.calibration_pass()
        )
        log(f"calibration: {json.dumps(self.checks['calibration'], ensure_ascii=False)}")
        self.checks["prior_wiring"] = self.prior_wiring_check()
        log(
            "prior wiring: max |dlog_p0| = "
            f"{self.checks['prior_wiring']['max_abs_log_p0_difference']:.4g}, max |dq| = "
            f"{self.checks['prior_wiring']['max_abs_q_difference']:.4g}, check status="
            f"{self.checks['prior_wiring']['runs']['biased']['check_status']}"
        )
        step = 0
        cursor = int(getattr(self, "initial_data_cursor", 0))
        if plan.resume is not None:
            step, cursor, payload = self.load_for_resume(Path(plan.resume))
            log(f"resumed {plan.resume} at step {step} (dataset cursor {cursor})")
            reference = payload.get("dev_reference")
            if reference is not None and "prediction" in reference:
                after = self.evaluate_dev(
                    dev_windows, noise=dev_noise, time_values=dev_time, keep_prediction=True
                )
                delta = float((after["prediction"] - reference["prediction"]).abs().max())
                self.checks["save_resume"] = {
                    "checkpoint": str(plan.resume),
                    "resumed_step": int(step),
                    "prediction_max_abs_difference": delta,
                    "dev_action_error_at_save": float(reference.get("action_error")),
                    "dev_action_error_after_resume": float(after["action_error"]),
                    "measurement": (
                        "same fixed dev windows, same fixed flow noise/time, same precision; "
                        "the action chunk is compared elementwise"
                    ),
                }
                log(
                    f"save/resume: dev action max |diff| = {delta:.3e}; dev action error "
                    f"{reference.get('action_error'):.6f} -> {after['action_error']:.6f}"
                )
        generator = torch.Generator(device=self.device).manual_seed(int(plan.seed) + 1)
        if plan.resume is not None and payload.get("training_generator") is not None:
            generator.set_state(payload["training_generator"])
        elif hasattr(self, "initial_training_generator"):
            generator.set_state(self.initial_training_generator)
        self.train_generator = generator
        if step == 0:
            self.status(activity="测量训练前验证误差")
            baseline = self.evaluate_dev(dev_windows, noise=dev_noise, time_values=dev_time)
            self._write_json("dev_baseline.json", baseline)
            self.status(dev_action_error=baseline["action_error"])
            append_history(
                self.run_dir,
                step=0,
                wall_time=0.0,
                train_loss=None,
                dev_action_error=baseline["action_error"],
                sampled_control_mse=baseline.get("sampled_control_mse"),
            )
            log(f"initial fixed dev action error: {baseline['action_error']:.6f}")
        budget = int(plan.max_updates)
        log(
            f"training {budget - step} more updates: microbatch={plan.batch_size} anchors, "
            f"grad_accum={plan.grad_accum} (effective {plan.batch_size * plan.grad_accum}), "
            f"history={plan.history}, action horizon {action_horizon}"
        )
        step_times: list[float] = []
        step_anchors: list[int] = []
        losses: list[float] = []
        skipped: list[int] = []
        consecutive_padding = 0
        profile: dict[str, Any] = {}
        last_status = 0.0
        last_terms: dict[str, Any] = {}
        from owac.training.n02_v12_prefetch import WindowPrefetch

        depth = int(plan.config.get("runtime", {}).get("performance", {}).get("prefetch", 0))
        with (
            WindowPrefetch(self, cursor, depth=depth) as prefetch,
            Heartbeat(self.run_dir, "训练中"),
        ):
            self.status(state="running", activity=f"训练更新 {step}/{budget}", step=int(step))
            while step < budget:
                update_start = time.perf_counter()
                self._set_lr(step)
                self.optimizer.zero_grad(set_to_none=True)
                collected: dict[str, list[float]] = {}
                info: dict[str, Any] = {}
                last: dict[str, Any] = {}
                update_windows = []
                actual_anchors = 0
                micro = 0
                while micro < plan.grad_accum:
                    # Shuffle contiguous anchor blocks; every anchor, including the final
                    # partial block, occurs once per epoch. Windows remain independent.
                    start, stop = self.training_bounds(cursor)
                    prepared = prefetch.get(cursor, start, stop)
                    result = self.microbatch(start, stop, generator=generator, prepared=prepared)
                    cursor += 1
                    if result is None:
                        skipped.append(start)
                        consecutive_padding += 1
                        if consecutive_padding > 256:
                            raise RuntimeError(
                                f"{consecutive_padding} consecutive padding windows; "
                                "the mask path is suspect, refusing to keep skipping"
                            )
                        continue
                    consecutive_padding = 0
                    if not bool(torch.isfinite(result["loss"])):
                        self._write_json(
                            "failure_context.json",
                            {
                                "step": step + 1,
                                "terms": result["terms"],
                                "window": result["info"]["window"],
                            },
                        )
                        raise RuntimeError(f"non-finite loss before backward at update {step + 1}")
                    (result["loss"] / plan.grad_accum).backward()
                    for name, value in result["terms"].items():
                        if value is not None:
                            collected.setdefault(name, []).append(value)
                    info = result["info"]
                    update_windows.append(info["window"])
                    actual_anchors += stop - start
                    last = result
                    micro += 1
                stats_every = int(
                    plan.config.get("runtime", {})
                    .get("performance", {})
                    .get("gradient_stats_every", 1)
                )
                detailed = step == 0 or step % max(1, stats_every) == 0 or self.grad_clip <= 0
                gradient_stats = self.role_gradient_l1() if detailed else {}
                if detailed:
                    self.frozen_with_grad = [
                        name
                        for name, parameter in self.bridge.base.named_parameters()
                        if parameter.grad is not None and not parameter.requires_grad
                    ]
                if "roles" not in self.checks.get("gradient_wiring", {}) and gradient_stats:
                    self.checks["gradient_wiring"] = self.gradient_wiring_check(gradient_stats)
                    log(
                        "gradient wiring (first real backward): "
                        + json.dumps(
                            {
                                role: round(item["l1"], 4)
                                for role, item in self.checks["gradient_wiring"]["roles"].items()
                            },
                            ensure_ascii=False,
                        )
                    )
                if self.frozen_with_grad:
                    raise RuntimeError(
                        "frozen π0.5 parameters received gradients: "
                        f"{self.frozen_with_grad[:5]} (count {len(self.frozen_with_grad)})"
                    )
                if not all(item["finite"] for item in gradient_stats.values()):
                    raise RuntimeError(f"non-finite gradients before optimizer update {step + 1}")
                grad_norm = None
                if self.grad_clip > 0:
                    grad_norm = self.clip_gradients()
                self.optimizer.step()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                step += 1
                self.current_step = step
                elapsed = time.perf_counter() - update_start
                step_times.append(elapsed)
                step_anchors.append(actual_anchors)
                means = {name: float(np.mean(values)) for name, values in collected.items()}
                weights = self._aux_weights()
                total_loss = float(means.get("action", float("nan")))
                for name, key in (
                    ("masked_raw", "masked_raw"),
                    ("causal_raw", "causal_raw"),
                    ("noise", "noise"),
                ):
                    if key in means:
                        total_loss += weights[name] * means[key]
                if not math.isfinite(total_loss):
                    raise RuntimeError(f"non-finite training loss at update {step}")
                with (self.run_dir / "optimization.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "terms": means,
                                "gradient_clip_scope": plan.config.get("optim", {}).get(
                                    "clip_scope", "global"
                                ),
                                "gradient_norms_by_clip_group": getattr(
                                    self, "last_clip_norms", {}
                                ),
                                "grad_norm_before_clip": None
                                if grad_norm is None
                                else float(grad_norm),
                                "window": info.get("window"),
                                "windows": update_windows,
                                "actual_anchors": actual_anchors,
                                "execution_revision": EXECUTION_REVISION,
                            }
                        )
                        + "\n"
                    )
                losses.append(total_loss)
                last_terms = {name: round(value, 6) for name, value in means.items()}
                dev_record = None
                if plan.validate_every and step % max(plan.validate_every, 1) == 0:
                    dev_record = self.evaluate_dev(
                        dev_windows,
                        noise=dev_noise,
                        time_values=dev_time,
                        keep_prediction=bool(plan.save_every and step % plan.save_every == 0),
                    )
                    log(
                        f"dev at step {step}: action error {dev_record['action_error']:.6f} over "
                        f"{dev_record['windows']} official windows"
                    )
                now = time.perf_counter()
                if step % 10 == 0 or (now - last_status) >= 5.0 or dev_record:
                    self.status(
                        state="running",
                        activity=f"训练更新 {step}/{budget}",
                        step=int(step),
                        step_seconds=float(np.mean(step_times[-10:])),
                        anchors_per_second=sum(step_anchors[-10:]) / sum(step_times[-10:]),
                        loss=total_loss,
                        **({"dev_action_error": dev_record["action_error"]} if dev_record else {}),
                    )
                    append_history(
                        self.run_dir,
                        step=int(step),
                        wall_time=float(np.sum(step_times)),
                        train_loss=total_loss,
                        dev_action_error=None if dev_record is None else dev_record["action_error"],
                        sampled_control_mse=None
                        if dev_record is None
                        else dev_record.get("sampled_control_mse"),
                    )
                    last_status = now
                if plan.profile_updates and step <= plan.profile_updates:
                    profile = {
                        "updates": int(step),
                        "wall_seconds": float(np.sum(step_times)),
                        "step_seconds_mean": float(np.mean(step_times)),
                        "step_seconds_p50": float(np.percentile(step_times, 50)),
                        "step_seconds_p95": float(np.percentile(step_times, 95)),
                        "microbatch_anchors": int(plan.batch_size),
                        "grad_accum": int(plan.grad_accum),
                        "effective_anchors": int(plan.batch_size * plan.grad_accum),
                        "actual_anchors": sum(step_anchors),
                        "anchors_per_second": sum(step_anchors) / sum(step_times),
                        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())
                        if self.device.type == "cuda"
                        else 0,
                        "loss": total_loss,
                        "terms": last_terms,
                        "run_stats": info.get("run_stats", {}),
                    }
                    if step == plan.profile_updates:
                        log(f"profile over {plan.profile_updates} updates: {json.dumps(profile)}")
                        self.checks["profile"] = profile
                        self._write_json("profile.json", profile)
                if plan.save_every and step % plan.save_every == 0 and step < budget:
                    self._save_point(
                        last, step, cursor, dev_windows, dev_noise, dev_time, reference=dev_record
                    )
                    log(f"update {step}: loss {total_loss:.6f} terms {json.dumps(last_terms)}")
        self.overview = {
            "updates_this_run": int(len(step_times)),
            "final_step": int(step),
            "wall_seconds": float(np.sum(step_times)),
            "step_seconds_mean": float(np.mean(step_times)) if step_times else 0.0,
            "step_seconds_p50": float(np.percentile(step_times, 50)) if step_times else 0.0,
            "step_seconds_p95": float(np.percentile(step_times, 95)) if step_times else 0.0,
            "last_terms": last_terms,
            "final_loss_mean_last10": float(np.mean(losses[-10:])) if losses else None,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())
            if self.device.type == "cuda"
            else 0,
            "skipped_all_padding_anchors": len(skipped),
            "loss_is_not_a_capability_score": True,
        }
        reference = self.evaluate_dev(
            dev_windows, noise=dev_noise, time_values=dev_time, keep_prediction=True
        )
        if step_times:
            path = self.save(step, cursor=cursor, dev_reference=reference)
            previews = self.write_previews(last["batch"], last["output"])
        else:
            path = Path(plan.resume) if plan.resume is not None else self.checkpoint_path(step)
            previews = {"status": "skipped", "reason": "no update ran in this invocation"}
        self.final = {
            "checkpoint": str(path),
            "checkpoint_bytes": int(path.stat().st_size),
            "dev_action_error": float(reference["action_error"]),
            "previews": previews,
        }
        self.status(
            state="completed",
            activity=f"训练完成 {step} updates",
            step=int(step),
            checkpoint=str(path),
            loss=self.overview["final_loss_mean_last10"],
            dev_action_error=reference["action_error"],
            step_seconds=self.overview["step_seconds_mean"],
            preview_files=previews.get("files") if previews.get("status") == "written" else None,
        )
        append_history(
            self.run_dir,
            step=int(step),
            wall_time=float(np.sum(step_times)),
            train_loss=self.overview["final_loss_mean_last10"],
            dev_action_error=float(reference["action_error"]),
            sampled_control_mse=reference.get("sampled_control_mse"),
        )
        return {"overview": self.overview, "final": self.final}

    def _save_point(
        self,
        last: dict,
        step: int,
        cursor: int,
        dev_windows: list,
        dev_noise: torch.Tensor,
        dev_time: torch.Tensor,
        reference: dict | None = None,
    ) -> None:
        if reference is None:
            reference = self.evaluate_dev(
                dev_windows, noise=dev_noise, time_values=dev_time, keep_prediction=True
            )
        path = self.save(step, cursor=cursor, dev_reference=reference)
        previews = self.write_previews(last["batch"], last["output"])
        self.checks.setdefault("previews", []).append(previews)
        self.status(
            state="running",
            activity=f"已保存 checkpoint step {step}",
            step=int(step),
            checkpoint=str(path),
            dev_action_error=float(reference["action_error"]),
            preview_files=previews.get("files") if previews.get("status") == "written" else None,
        )

    def _set_lr(self, step: int) -> None:
        scale = 1.0 if self.warmup <= 0 else min(1.0, float(step + 1) / self.warmup)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * scale

    def _write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        return path

    # -- single real step (checks only) --------------------------------------------------

    def single_step(self) -> dict:
        """One real forward+backward for the acceptance record, with real gradient numbers."""
        generator = torch.Generator(device=self.device).manual_seed(int(self.plan.seed) + 1)
        self.optimizer.zero_grad(set_to_none=True)
        result = self.microbatch(0, self.plan.batch_size, generator=generator)
        loss = result["loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"the first real loss is not finite: {float(loss)}")
        loss.backward()
        stats = self.role_gradient_l1()
        self.frozen_with_grad = [
            name
            for name, parameter in self.bridge.base.named_parameters()
            if parameter.grad is not None and not parameter.requires_grad
        ]
        self.checks["gradient_wiring"] = self.gradient_wiring_check(stats)
        self.checks["first_step"] = {
            "loss": float(loss.detach()),
            "terms": result["terms"],
            "info": result["info"],
            "gradients": self.gradient_wiring_check(stats),
            "actions_control_dim": int(result["info"]["action_shape"][-1]),
            "action_shape_after_padding": [int(item) for item in result["info"]["action_shape"]],
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())
            if self.device.type == "cuda"
            else 0,
        }
        self.log(
            "first real step: loss={loss:.6f} terms={terms} gradients={grads}".format(
                loss=float(loss.detach()),
                terms=json.dumps(result["terms"]),
                grads=json.dumps(
                    {role: round(item["l1"], 4) for role, item in stats.items()}, ensure_ascii=False
                ),
            )
        )
        self.optimizer.zero_grad(set_to_none=True)
        return self.checks["first_step"]


# --------------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="owac.training.n02_v12",
        description=(
            "N02 v1.2 P2 real training: official RawVLA-Bench RAW windows -> the v1.2 chain -> "
            "frozen pi0.5 flow-matching action loss plus the RAW masked/causal auxiliary terms."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="platform/model config JSON")
    parser.add_argument(
        "--seed", type=int, default=None, help="single training seed (config default)"
    )
    parser.add_argument("--max-updates", type=int, default=None, help="optimizer updates to run")
    parser.add_argument(
        "--output", type=Path, default=None, help="run directory (status.json lives here)"
    )
    parser.add_argument("--run-id", default=None, help="run id reported to the monitor page")
    parser.add_argument(
        "--profile-updates",
        type=int,
        default=None,
        help="leading updates whose timing/GPU profile is recorded (0 disables)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument("--validate-every", type=int, default=None, help="dev pass every N updates")
    parser.add_argument("--save-every", type=int, default=None, help="checkpoint every N updates")
    parser.add_argument("--batch-size", type=int, default=None, help="anchors per microbatch")
    parser.add_argument(
        "--grad-accum", type=int, default=None, help="microbatches per optimizer update"
    )
    parser.add_argument(
        "--previews", type=int, default=None, help="write RAW/attention previews (0 disables)"
    )
    parser.add_argument("--device", default=None, help="torch device (config default)")
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="run the correctness checks plus one real forward/backward, then stop",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="fresh forward/backward profile; ZERO optimizer updates",
    )
    parser.add_argument(
        "--profile-batches",
        type=int,
        default=3,
        help="distinct train windows in the zero-update profile",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    shadowed = package_identity_error()
    if shadowed:
        print(shadowed, file=sys.stderr, flush=True)
        return 3
    log = LogTail()
    plan = resolve_plan(args)
    validate_run_directory(plan, profile_only=args.profile_only)
    if plan.resume is None:
        # Atomic reservation prevents two fresh jobs from sharing the same output directory.
        plan.output.mkdir(parents=True, exist_ok=False)
    for note in plan.notes:
        log(f"note: {note}")
    log(
        f"run {plan.run_id} (stage {STAGE}, arm {plan.arm}, platform {plan.platform}, seed "
        f"{plan.seed}), output {plan.output}"
    )
    trainer = N02V12Trainer(plan, log)
    try:
        trainer.build()
        if plan.config.get("task_adaptation", {}).get("protocol") == "n02-task-repair-r11-v1":
            trainer._write_json("run_config.json", plan.config)
        if args.profile_only and plan.config.get("initialisation"):
            raise ValueError("fresh profile cannot use repair transfer weights")
        if plan.resume is None:
            trainer.initialise_repair_weights()
        if args.profile_only:
            from owac.training.n02_v12_profile import profile_fresh

            profile_fresh(trainer, batches=args.profile_batches)
            return 0
        trainer.checks = trainer.correctness_checks()
        log(
            "correctness checks: "
            + json.dumps(
                {
                    "units": trainer.checks["units_and_shapes"],
                    "causality": trainer.checks["causality"],
                    "isolation": trainer.checks["partition_isolation"],
                    "episode_reset": trainer.checks["episode_reset"],
                },
                ensure_ascii=False,
            )
        )
        if args.checks_only:
            trainer.single_step()
            trainer._write_json("checks.json", trainer.checks)
            return 0
        trainer.train()
        if plan.config.get("task_adaptation", {}).get("protocol") == "n02-task-repair-r11-v1":
            from owac.training.n02_v12_task import select_repaired_checkpoint

            trainer._write_json("selection.json", select_repaired_checkpoint(trainer.run_dir))
        trainer.checks["final"] = trainer.final
        trainer._write_json("checks.json", trainer.checks)
        trainer._write_json(
            "report.json",
            {
                "run_id": plan.run_id,
                "stage": STAGE,
                "execution_revision": EXECUTION_REVISION,
                "task_protocol": plan.config.get("task_adaptation", {}).get("protocol"),
                "arm": plan.arm,
                "platform": plan.platform,
                "seed": int(plan.seed),
                "config_path": str(plan.config_path),
                "config": plan.config,
                "assets": trainer.assets,
                "parameters": trainer.bridge.parameter_report(),
                "trained_parameters": {
                    name: sum(int(parameter.numel()) for parameter in parameters)
                    for name, parameters in trainer.parameter_groups.items()
                },
                "frozen": {
                    "backend_parameters": int(trainer.bridge.frozen_parameter_count),
                    "backend_parameters_with_grad": trainer.frozen_with_grad,
                },
                "overview": trainer.overview,
                "final": trainer.final,
                "checks": trainer.checks,
                "eval": {
                    "planned_closed_loop": 100,
                    "completed": 0,
                    "note": "the 100-episode closed loop belongs to P4; nothing ran here",
                },
            },
        )
        return 0
    except BaseException as error:  # noqa: BLE001 - the page must show why a run stopped
        message = f"{type(error).__name__}: {error}"
        log(f"run failed: {message}")
        try:
            if trainer.run_dir.is_dir():
                trainer.status(state="failed", activity=f"异常停止：{message}")
        except Exception:  # noqa: BLE001
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
