"""Reusable experiment context for RAW-stem distillation training/evaluation.

The normal factory loads the actual StarVLA/PI05 policy, full audited checkpoint,
student frontend, N02 bridge, paired dataset, canonical statistics and optional
learned RAW teacher.  A registered fixture factory is available for CPU tests
only and is always labelled ``fixture_contract``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CameraAnchor",
    "ContextUnavailable",
    "ExperimentContext",
    "build_real_context",
    "camera_anchors",
    "load_experiment_context",
    "register_fixture_context_factory",
]


class ContextUnavailable(RuntimeError):
    """A real experiment asset or native adapter is absent/unusable."""


@dataclass(frozen=True, slots=True)
class CameraAnchor:
    source_id: str
    frame_id: int
    timestamp: float
    variant: str
    camera_order: tuple[str, ...]
    frame_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "variant": self.variant,
            "camera_order": list(self.camera_order),
            "frame_keys": list(self.frame_keys),
        }


@dataclass(slots=True)
class ExperimentContext:
    config_path: Path
    config: Mapping[str, Any]
    evidence_level: str
    policy: Any = None
    student: Any = None
    bridge: Any = None
    dataset: Any = None
    tokenizer: Any = None
    stats: Any = None
    checkpoint: Any = None
    load_audit: Any = None
    history: Any = None
    teacher_adapter: Any = None
    student_artifact: str | None = None
    pure_load: bool = False
    checkpoint_step: int = 0
    parity_runner: Any = None
    profile_runner: Any = None
    optimizer_factory: Any = None
    scheduler_factory: Any = None
    validation_callback: Any = None
    backend_identity: Mapping[str, Any] = field(default_factory=dict)
    tokenizer_identity: Mapping[str, Any] = field(default_factory=dict)
    stats_identity: Mapping[str, Any] = field(default_factory=dict)
    source_split: Mapping[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    @property
    def fixture_contract(self) -> bool:
        return self.evidence_level == "fixture_contract"

    @property
    def real_ready(self) -> bool:
        return (
            self.evidence_level == "real_policy"
            and self.student is not None
            and self.bridge is not None
            and self.checkpoint is not None
            and self.load_audit is not None
        )

    @property
    def student_ready(self) -> bool:
        return self.student is not None and self.bridge is not None

    @property
    def teacher_ready(self) -> bool:
        if self.teacher_adapter is None:
            return False
        if self.fixture_contract:
            return True
        ready = getattr(self.teacher_adapter, "real_ready", None)
        if ready is None:
            receipt = getattr(self.teacher_adapter, "receipt", None)
            ready = getattr(receipt, "real_ready", False)
        return ready is True

    def camera_anchors(self) -> tuple[CameraAnchor, ...]:
        if self.dataset is None:
            raise ContextUnavailable("paired dataset is unavailable")
        return camera_anchors(self.dataset.manifest)

    def preflight(self) -> dict[str, Any]:
        anchors = 0
        try:
            anchors = len(self.camera_anchors())
        except ContextUnavailable:
            pass
        return {
            "schema": "owac.raw-distillation.context.v1",
            "evidence_level": self.evidence_level,
            "fixture_contract": self.fixture_contract,
            "real_ready": self.real_ready,
            "student_ready": self.student_ready,
            "teacher_ready": self.teacher_ready,
            "dataset_ready": self.dataset is not None,
            "camera_anchor_count": anchors,
            "source_split_sizes": _split_sizes(self.source_split),
            "limitations": list(self.limitations),
        }


def _split_sizes(split: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in split.values():
        counts[str(value)] += 1
    return dict(counts)


def camera_anchors(manifest: Any) -> tuple[CameraAnchor, ...]:
    """Group camera records into one control anchor per source/frame/time.

    Each manifest source declares its camera order.  A frame group with a
    missing or duplicate camera is rejected; no camera is treated as a separate
    control window.
    """
    source_order = {source.source_id: tuple(source.camera_ids) for source in manifest.sources}
    groups: dict[tuple[str, int, float, str], dict[str, Any]] = {}
    for frame in manifest.frames:
        key = (frame.source_id, frame.frame_id, frame.timestamp, frame.variant)
        groups.setdefault(key, {})[frame.camera_id] = frame
    anchors = []
    for (source_id, frame_id, timestamp, variant), frames in sorted(groups.items()):
        order = source_order[source_id]
        missing = [camera for camera in order if camera not in frames]
        extra = [camera for camera in frames if camera not in order]
        if missing or extra:
            raise ContextUnavailable(
                f"camera anchor {source_id}/{frame_id} missing={missing} extra={extra}"
            )
        anchors.append(
            CameraAnchor(
                source_id=source_id,
                frame_id=frame_id,
                timestamp=timestamp,
                variant=variant,
                camera_order=order,
                frame_keys=tuple(frames[camera].frame_key for camera in order),
            )
        )
    return tuple(anchors)


_FIXTURE_FACTORIES: dict[str, Callable[..., ExperimentContext]] = {}


def register_fixture_context_factory(name: str, factory: Callable[..., ExperimentContext]) -> None:
    if not name or not callable(factory):
        raise ValueError("fixture context factory name/callable required")
    _FIXTURE_FACTORIES[name] = factory


def load_experiment_context(
    config_path: str | Path,
    *,
    evidence_level: str = "real_policy",
    fixture_factory: str | Callable[..., ExperimentContext] | None = None,
) -> ExperimentContext:
    source = Path(config_path)
    if not source.is_file():
        raise ContextUnavailable(f"experiment config not found: {source}")
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ContextUnavailable("experiment config root must be an object")
    if fixture_factory is not None:
        if evidence_level != "fixture_contract":
            raise ContextUnavailable("fixture factory requires explicit fixture_contract evidence")
        factory = (
            _FIXTURE_FACTORIES.get(fixture_factory)
            if isinstance(fixture_factory, str)
            else fixture_factory
        )
        if factory is None:
            raise ContextUnavailable(f"fixture context factory not registered: {fixture_factory}")
        context = factory(config_path=source, config=config)
        if not isinstance(context, ExperimentContext) or not context.fixture_contract:
            raise ContextUnavailable("fixture factory must return a fixture_contract context")
        return context
    if evidence_level != "real_policy":
        raise ContextUnavailable("real context loading must use real_policy evidence")
    return build_real_context(source, config)


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    assets = config.get("assets", {})
    value = assets.get(key)
    if not value:
        raise ContextUnavailable(f"config.assets.{key} is absent")
    path = Path(str(value))
    if not path.exists():
        raise ContextUnavailable(f"config.assets.{key} does not exist: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous().reshape(-1)
    payload = b"" if value.numel() == 0 else value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _canonical(name: str) -> str:
    for marker in ("multi_modal_projector.", "vision_tower."):
        index = name.find(marker)
        if index >= 0:
            return name[index:]
    if name.startswith("projector."):
        return "multi_modal_projector." + name[len("projector.") :]
    return name


def _hash_visual_state(module: Any) -> dict[str, str]:
    return {
        _canonical(name): _tensor_sha(parameter)
        for name, parameter in module.named_parameters()
        if "vision_tower" in name or "multi_modal_projector" in name
    }


def _validate_load_counts(receipt: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Separate conversion receipt counts from serialized full reload counts."""
    receipt_loaded = int(receipt.get("loaded_keys", 0))
    receipt_model = int(receipt.get("model_keys", 0))
    actual_loaded = int(actual.get("loaded_keys", -1))
    actual_model = int(actual.get("model_keys", -1))
    if actual_loaded < 0 or actual_model < 0:
        raise ContextUnavailable("actual load audit lacks key counts")
    for key in ("active_missing", "unexpected", "shape_mismatch"):
        if actual.get(key):
            raise ContextUnavailable(f"actual load audit has nonempty {key}")
    if actual_loaded == actual_model:
        inactive = list(receipt.get("inactive_keys", ()))
        declared = {"action_head.gemma_expert.lm_head.weight"}
        if receipt_loaded != actual_model - 1 or set(inactive) != declared:
            raise ContextUnavailable(
                "serialized full reload count is explained only by the declared inactive head"
            )
    elif (
        actual_loaded != receipt_loaded
        or actual_model != receipt_model
        or list(actual.get("inactive_keys", ())) != list(receipt.get("inactive_keys", ()))
    ):
        raise ContextUnavailable("load audit count difference is not explained")


def _validate_real_identities(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    stats_path: Path,
    load_audit_path: Path,
) -> tuple[dict[str, Any], str]:
    assets = config.get("assets", {})
    expected = str(assets.get("checkpoint_sha256") or "")
    actual = _sha256_file(checkpoint_path)
    if len(expected) != 64 or actual != expected:
        raise ContextUnavailable("checkpoint SHA differs from config.assets.checkpoint_sha256")
    payload = json.loads(load_audit_path.read_text(encoding="utf-8"))
    if payload.get("model_sha256") != actual:
        raise ContextUnavailable("load audit model_sha256 differs from checkpoint")
    if int(payload.get("extra_training_steps", 0)) != 0:
        raise ContextUnavailable("load audit declares extra training steps")
    for key in ("active_missing", "unexpected", "shape_mismatch"):
        if payload.get(key):
            raise ContextUnavailable(f"load audit has nonempty {key}")
    if int(payload.get("loaded_keys", 0)) <= 0:
        raise ContextUnavailable("load audit contains no loaded keys")
    if int(payload.get("model_keys", 0)) < int(payload.get("loaded_keys", 0)):
        raise ContextUnavailable("load audit key counts are inconsistent")
    if payload.get("tokenizer_sha256") != _sha256_file(tokenizer_path):
        raise ContextUnavailable("load audit tokenizer identity differs")
    if payload.get("original_stats_sha256") != _sha256_file(stats_path):
        raise ContextUnavailable("load audit statistics identity differs")
    converter_expected = assets.get("converter_sha256")
    if converter_expected is not None and payload.get("converter_sha256") != converter_expected:
        raise ContextUnavailable("load audit converter identity differs from config")
    return payload, actual


def build_real_context(config_path: Path, config: Mapping[str, Any]) -> ExperimentContext:
    """Construct the actual policy/student/bridge/dataset from config assets.

    Imports stay command-local.  Missing StarVLA/checkpoint/tokenizer/statistics
    or a null asset entry raises ``ContextUnavailable``; no fixture fallback.
    """
    from owac.backends.rawvla import load_audited, new_model
    from owac.backends.rawvla_features_n02 import (
        RawVLAFeaturePolicy,
        StaticTokenSource,
        load_checkpoint_state,
        protocol_for,
    )
    from owac.baselines.raw_stem_siglip import (
        CheckpointIdentity,
        PolicyLoadEvidence,
        build_raw_stem_student,
    )
    from owac.raw.distillation_pairs import NormalisationStats, load_paired_manifest

    checkpoint_path = _required_path(config, "checkpoint")
    tokenizer_path = _required_path(config, "tokenizer")
    stats_path = _required_path(config, "statistics")
    manifest_path = _required_path(config, "paired_manifest")
    load_audit_path = _required_path(config, "load_audit")
    runtime_root = config.get("assets", {}).get("runtime")
    if runtime_root and Path(str(runtime_root)).exists():
        sys.path.insert(0, str(runtime_root))
    try:
        base = new_model("libero", tokenizer_path)
    except Exception as exc:
        raise ContextUnavailable(f"actual StarVLA policy cannot be constructed: {exc}") from exc
    audit_payload, checkpoint_sha = _validate_real_identities(
        config=config,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        stats_path=stats_path,
        load_audit_path=load_audit_path,
    )
    state = load_checkpoint_state(checkpoint_path)
    actual_audit = load_audited(base, state)
    _validate_load_counts(audit_payload, actual_audit)
    visual_hashes = _hash_visual_state(base)
    checkpoint = CheckpointIdentity(
        path=str(checkpoint_path),
        sha256=checkpoint_sha,
        verified=True,
        config_sha256=_sha256_file(config_path),
        tensor_sha256=visual_hashes,
    )
    evidence = PolicyLoadEvidence(
        checkpoint_sha256=checkpoint_sha,
        policy_module=type(base.vlm).__module__,
        loaded_keys=int(audit_payload.get("loaded_keys", len(state))),
        model_keys=int(audit_payload.get("model_keys", len(base.state_dict()))),
        verified=True,
        acceptance="accepted_real_policy",
        visual_weights_source="actual_policy_checkpoint",
        active_missing=tuple(audit_payload.get("active_missing", ())),
        unexpected=tuple(audit_payload.get("unexpected", ())),
        shape_mismatch=tuple(audit_payload.get("shape_mismatch", ())),
        inactive_keys=tuple(audit_payload.get("inactive_keys", ())),
        vision_keys_loaded=tuple(visual_hashes),
        temporal_teacher=False,
    )
    input_mode = str(config.get("task", {}).get("input_regime", "rawvla_rgb3"))
    student = build_raw_stem_student(
        base,
        input_mode=input_mode,
        checkpoint=checkpoint,
        load_audit=evidence,
    )
    protocol = protocol_for("libero")
    bridge = RawVLAFeaturePolicy(
        base,
        StaticTokenSource(None),
        protocol=protocol,
        token_dim=student.geometry.projector_out,
        tokens_per_camera=student.geometry.num_positions,
        image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        freeze_backend=True,
    )
    dataset = load_paired_manifest(manifest_path)
    stats = NormalisationStats.from_json(
        stats_path, mode="quantile", stats_key=str(config.get("assets", {}).get("stats_key", ""))
    )
    history_path = config.get("assets", {}).get("teacher_history_contract")
    history = _load_history(history_path)
    teacher = _load_raw_teacher(config, base, history)
    split = dataset.source_splits
    limitations = []
    if teacher is None:
        limitations.append("learned RawVLA teacher checkpoint/adapter unavailable")
    context = ExperimentContext(
        config_path=config_path,
        config=config,
        evidence_level="real_policy",
        policy=base,
        student=student,
        bridge=bridge,
        dataset=dataset,
        tokenizer=getattr(base, "tokenizer", None),
        stats=stats,
        checkpoint=checkpoint,
        load_audit=evidence,
        history=history,
        teacher_adapter=teacher,
        backend_identity={"sha256": checkpoint_sha, "load_audit": audit_payload},
        tokenizer_identity={"path": str(tokenizer_path), "sha256": _sha256_file(tokenizer_path)},
        stats_identity={"path": str(stats_path), "sha256": stats.sha256},
        source_split=split,
        limitations=tuple(limitations),
    )
    context.parity_runner = ContextParityRunner(context)
    context.profile_runner = ContextProfileRunner(context)
    return context


def build_pure_context(
    config_path: Path,
    config: Mapping[str, Any],
    student_artifact: str | Path,
    *,
    checkpoint_step: int | None = None,
    allow_fixture: bool = False,
) -> ExperimentContext:
    """Load an exported student + frozen backend without any teacher/ISP path."""
    from owac.backends.rawvla import load_audited, new_model
    from owac.backends.rawvla_features_n02 import (
        RawVLAFeaturePolicy,
        StaticTokenSource,
        load_checkpoint_state,
        protocol_for,
    )
    from owac.baselines.raw_stem_siglip import load_student_artifact
    from owac.raw.distillation_pairs import (
        NormalisationStats,
        PairedDistillationDataset,
        load_paired_manifest,
    )

    checkpoint_path = _required_path(config, "checkpoint")
    tokenizer_path = _required_path(config, "tokenizer")
    stats_path = _required_path(config, "statistics")
    manifest_path = _required_path(config, "paired_manifest")
    load_audit_path = _required_path(config, "load_audit")
    audit_payload, checkpoint_sha = _validate_real_identities(
        config=config,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        stats_path=stats_path,
        load_audit_path=load_audit_path,
    )
    runtime_root = config.get("assets", {}).get("runtime")
    if runtime_root and Path(str(runtime_root)).exists():
        sys.path.insert(0, str(runtime_root))
    try:
        base = new_model("libero", tokenizer_path)
        state = load_checkpoint_state(checkpoint_path)
        actual_audit = load_audited(base, state)
        for key in ("active_missing", "unexpected", "shape_mismatch"):
            if actual_audit.get(key):
                raise ContextUnavailable(f"actual load audit has nonempty {key}")
        if int(actual_audit.get("loaded_keys", -1)) != int(audit_payload.get("loaded_keys", -2)):
            raise ContextUnavailable("actual load audit key count differs from receipt")
    except Exception as exc:
        raise ContextUnavailable(f"actual frozen backend cannot be loaded: {exc}") from exc
    artifact_path = Path(student_artifact)
    deployment = load_student_artifact(
        artifact_path,
        expected_checkpoint_sha256=checkpoint_sha,
    )
    meta_step = int(dict(getattr(deployment, "metadata", {})).get("step", 0))
    if checkpoint_step is not None and int(checkpoint_step) != meta_step:
        raise ContextUnavailable(
            f"requested checkpoint step {checkpoint_step} differs from artifact step {meta_step}"
        )
    config_image_keys = list(
        getattr(getattr(getattr(base, "config", None), "framework", None), "image_keys", ())
        or ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    )
    bridge = RawVLAFeaturePolicy(
        base,
        StaticTokenSource(None),
        protocol=protocol_for("libero"),
        token_dim=deployment.geometry.projector_out,
        tokens_per_camera=deployment.geometry.num_positions,
        image_keys=tuple(config_image_keys),
        freeze_backend=True,
    )
    # The student is already an independent deep copy.  Release the unused
    # original visual modules from the backend only after that construction.
    core = getattr(getattr(getattr(base, "vlm", None), "model", None), "model", None)
    if core is not None:
        for attribute in ("vision_tower", "multi_modal_projector"):
            if hasattr(core, attribute):
                delattr(core, attribute)
    dataset = PairedDistillationDataset(
        load_paired_manifest(
            manifest_path,
            allow_fixture=allow_fixture,
            student_only=True,
        ),
        student_only=True,
    )
    stats = NormalisationStats.from_json(
        stats_path,
        mode=str(config.get("assets", {}).get("stats_mode", "quantile")),
        stats_key=str(config.get("assets", {}).get("stats_key", "")),
    )
    history = _load_history(config.get("assets", {}).get("teacher_history_contract"))
    context = ExperimentContext(
        config_path=config_path,
        config=config,
        evidence_level="real_policy",
        policy=base,
        student=deployment,
        bridge=bridge,
        dataset=dataset,
        tokenizer=getattr(base, "tokenizer", None),
        stats=stats,
        checkpoint=type("ArtifactCheckpoint", (), {"sha256": checkpoint_sha})(),
        load_audit={"audit": audit_payload, "checkpoint_sha256": checkpoint_sha},
        history=history,
        teacher_adapter=None,
        student_artifact=str(artifact_path),
        pure_load=True,
        checkpoint_step=meta_step,
        backend_identity={"sha256": checkpoint_sha, "load_audit": audit_payload},
        tokenizer_identity={"path": str(tokenizer_path), "sha256": _sha256_file(tokenizer_path)},
        stats_identity={"path": str(stats_path), "sha256": _sha256_file(stats_path)},
        source_split=dataset.source_splits,
        limitations=("pure artifact load: no teacher/ISP/cache",),
    )
    return context


def _load_history(path: Any) -> Any:
    from owac.baselines.raw_stem_siglip import HistoryContract

    if not path:
        return HistoryContract()
    payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    return HistoryContract(
        frame_ids=tuple(payload["frame_ids"]),
        timestamps=tuple(payload["timestamps"]),
        stride=int(payload.get("stride", 1)),
        camera_order=tuple(payload.get("camera_order", ())),
        burst_frames=int(payload.get("burst_frames", len(payload["frame_ids"]))),
        rnn_frames=int(payload.get("rnn_frames", len(payload["frame_ids"]))),
        reset=bool(payload.get("reset", False)),
        max_history=payload.get("max_history"),
    )


def _load_raw_teacher(config: Mapping[str, Any], base: Any, history: Any) -> Any:
    from owac.training.raw_distillation import (
        RawTeacherCheckpoint,
        RawVLATeacherAdapter,
        TeacherPreparationUnavailable,
        register_raw_teacher_adapter,
    )

    assets = config.get("assets", {})
    frontend_path = assets.get("teacher_frontend_checkpoint")
    if not frontend_path:
        return None
    frontend_config = assets.get("teacher_frontend_config")
    if not frontend_config:
        raise ContextUnavailable("teacher frontend checkpoint present but config is absent")
    config_path = Path(str(frontend_config))
    if not config_path.is_file():
        raise ContextUnavailable(f"teacher frontend config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = Path(str(config.get("teacher_requirement", {}).get("source_root", "")))
    if not source_root.exists():
        raise ContextUnavailable("author RawVLA source_root is absent")
    sys.path.insert(0, str(source_root))
    try:
        from baselines.rawvla.rawvla import RAWVLA
    except Exception as exc:
        raise ContextUnavailable(f"author RAWVLA implementation cannot be imported: {exc}") from exc
    try:
        frontend = RAWVLA(**payload)
    except Exception as exc:
        raise ContextUnavailable(f"author RAWVLA frontend construction failed: {exc}") from exc
    adapter_name = str(config.get("teacher_requirement", {}).get("adapter_name", ""))
    if adapter_name:
        register_raw_teacher_adapter(adapter_name, RAWVLA)
    checkpoint = RawTeacherCheckpoint(
        frontend_checkpoint_sha256=str(assets.get("teacher_frontend_checkpoint_sha256", "")),
        frontend_config_sha256=_sha256_file(config_path),
        code_revision=str(config.get("teacher_requirement", {}).get("code_revision", "")),
        adapter_name=adapter_name,
        source="author rawvla learned frontend",
        verified=True,
        state_path=str(frontend_path),
        state_dict_prefix=str(assets.get("teacher_state_dict_prefix", "")),
    )
    try:
        return RawVLATeacherAdapter(
            policy=base,
            frontend=frontend,
            checkpoint=checkpoint,
            history=history,
            evidence_level="real_policy",
            state_path=frontend_path,
        )
    except TeacherPreparationUnavailable as exc:
        raise ContextUnavailable(str(exc)) from exc


class ContextParityRunner:
    """Build a typed native/bridge stage-0 trace from the real context."""

    def __init__(self, context: ExperimentContext) -> None:
        self.context = context

    def _client_postprocess(self):
        context = self.context
        source = (
            Path(str(context.config.get("teacher_requirement", {}).get("source_root", "")))
            / "examples/simBenchmarks/LIBERO/eval_files/openpi/pi_libero_common.py"
        )
        if not source.is_file():
            return None, None
        import importlib.util

        spec = importlib.util.spec_from_file_location("owac_pinned_client_postprocess", source)
        if spec is None or spec.loader is None:
            return None, None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, "postprocess_openpi_actions", None)
        if not callable(function) or context.stats.action is None or context.dataset is None:
            return None, None
        import numpy as np

        raw_state = None
        for frame_key in context.dataset.frame_keys:
            sample = context.dataset.sample(frame_key)
            raw_state = sample.state_raw
            break
        if raw_state is None:
            return None, None
        norm_stats = {
            "actions": {
                "q01": np.asarray(context.stats.action["q01"], dtype=np.float32),
                "q99": np.asarray(context.stats.action["q99"], dtype=np.float32),
            }
        }

        def callback(actions):
            import torch

            output = function(
                actions.detach().cpu().numpy(),
                raw_state.detach().cpu().numpy(),
                norm_stats,
                model_name="PI05",
                model_source="openpi",
            )
            return torch.from_numpy(np.asarray(output, dtype=np.float32))

        return callback, _sha256_file(source)

    def run(self) -> Any:
        import torch

        from owac.evaluation.raw_distillation_trace import (
            BridgePolicyTracer,
            NativePolicyTracer,
            run_stage0_traces,
            typed_prepared_observation,
        )

        context = self.context
        if context.teacher_adapter is None or context.dataset is None:
            raise ContextUnavailable("teacher adapter or dataset is unavailable for parity")
        anchors = context.camera_anchors()
        if not anchors:
            raise ContextUnavailable("no paired camera anchor is available")
        anchor = anchors[0]
        native_keys = tuple(
            getattr(
                getattr(getattr(context.policy, "config", None), "framework", None),
                "image_keys",
                (),
            )
            or ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        )
        prepared_rgbs = []
        token_groups = []
        state = None
        language = None
        for frame_key in anchor.frame_keys:
            sample = context.dataset.sample(frame_key)
            raw = sample.raw.as_stem_input(camera_id=sample.camera_id)
            prepared = context.teacher_adapter.prepare(
                raw.values.unsqueeze(1), history=context.history
            )
            rgb = prepared.prepared_rgb
            if rgb.ndim == 4:
                rgb = rgb[:, None]
            if rgb.ndim != 5 or rgb.shape[1] != 1:
                raise ContextUnavailable("teacher prepared RGB must be [B,1,3,H,W]")
            prepared_rgbs.append(rgb)
            tokens = prepared.tokens
            if tokens.ndim == 4:
                tokens = tokens[:, 0]
            token_groups.append(tokens)
            state = sample.state if state is None else state
            language = sample.language if language is None else language
        prepared_rgb = torch.cat(prepared_rgbs, dim=1)
        from owac.backends.rawvla_language import prepare_language_state

        lang_tokens, lang_masks, padded_state = prepare_language_state(
            context.bridge,
            [language],
            state.detach().cpu().numpy().reshape(1, -1),
            device=prepared_rgb.device,
        )
        # Bridge token order must be the native three-camera order. Missing
        # physical cameras are explicit zero tokens with mask false.
        per_camera = token_groups[0].shape[1] if token_groups else 1
        width = token_groups[0].shape[-1] if token_groups else 0
        token_by_camera = dict(zip(anchor.camera_order, token_groups, strict=True))
        ordered_tokens = []
        mask_parts = []
        missing = []
        for camera in native_keys:
            if camera in token_by_camera:
                ordered_tokens.append(token_by_camera[camera])
                mask_parts.append(
                    torch.ones(
                        token_by_camera[camera].shape[:2],
                        dtype=torch.bool,
                        device=token_by_camera[camera].device,
                    )
                )
            else:
                missing.append(camera)
                ordered_tokens.append(
                    torch.zeros(
                        token_groups[0].shape[0],
                        per_camera,
                        width,
                        device=token_groups[0].device,
                    )
                )
                mask_parts.append(
                    torch.zeros(
                        token_groups[0].shape[0],
                        per_camera,
                        dtype=torch.bool,
                        device=token_groups[0].device,
                    )
                )
        tokens = torch.cat(ordered_tokens, dim=1)
        token_mask = torch.cat(mask_parts, dim=1)
        from owac.core.encoder_contract_n02 import VisualTokens

        visual = VisualTokens(
            values=tokens,
            mask=token_mask,
            camera_ids=native_keys,
            positions=torch.zeros(tokens.shape[0], tokens.shape[1], 3, device=tokens.device),
        )
        observation = typed_prepared_observation(
            prepared_rgb=prepared_rgb,
            camera_order=anchor.camera_order,
            native_image_keys=native_keys,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=padded_state,
        )
        client_callback, client_sha = self._client_postprocess()
        native = NativePolicyTracer(context.policy, image_keys=native_keys)
        bridge = BridgePolicyTracer(context.bridge)
        report = run_stage0_traces(
            native,
            bridge,
            observation=observation,
            tokens=visual,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=padded_state,
            noise=torch.zeros(1, 10, 32, device=prepared_rgb.device),
            history=context.history,
            reset=True,
            real_stats=context.stats,
            client_postprocess=client_callback,
            camera_order=native_keys,
            inactive_cameras=tuple(missing),
        )
        report.native.metadata["inactive_cameras"] = tuple(missing)
        report.bridge.metadata["inactive_cameras"] = tuple(missing)
        report.native.metadata["client_postprocess_sha256"] = client_sha
        report.bridge.metadata["client_postprocess_sha256"] = client_sha
        return report


class ContextProfileRunner:
    """Real zero-update train-step and pure-inference profiler from the context."""

    def __init__(self, context: ExperimentContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        from owac.evaluation.raw_distillation import FrozenBackend, PureStudentRuntime
        from owac.training.raw_distillation import AnchorBatchIterator, RawDistillationTrainer
        from owac.training.raw_distillation_profile import profile_inference, profile_train_step

        context = self.context
        if context.student is None or context.bridge is None or context.teacher_adapter is None:
            raise ContextUnavailable("student/bridge/teacher adapter is unavailable for profiling")
        iterator = AnchorBatchIterator(context, split="train")
        batches = []
        for batch in iterator:
            batches.append(batch)
            if len(batches) >= int(context.config.get("training", {}).get("effective_batch", 16)):
                break
        if not batches:
            raise ContextUnavailable("no real paired training anchor is available")
        trainer = RawDistillationTrainer(
            student=context.student,
            bridge=context.bridge,
            config=context.config,
            history=context.history,
            teacher_ready=context.teacher_ready,
            stage0_parity_ready=bool(context.config.get("assets", {}).get("stage0_parity")),
            source_split_ready=True,
            resource_budget_ready=True,
            teacher=context.teacher_adapter,
        )
        result = profile_train_step(
            trainer,
            batches,
            evidence_level=context.evidence_level,
            update_cap=min(
                5000,
                int(context.config.get("training", {}).get("max_cumulative_updates_per_arm", 5000)),
            ),
            effective_batch=int(context.config.get("training", {}).get("effective_batch", 16)),
        )
        report = result.as_dict()
        if context.dataset is not None:
            views = [
                context.dataset.student_view(frame.frame_key)
                for frame in context.dataset.manifest.frames
                if frame.variant == "rawvla_rgb3"
            ][:2]
            if views:
                backend = FrozenBackend(
                    bridge=context.bridge,
                    tokenizer=context.tokenizer,
                    stats=context.stats,
                    identity=context.backend_identity,
                    tokenizer_sha256=str(context.tokenizer_identity.get("sha256", "")),
                    stats_sha256=str(context.stats_identity.get("sha256", "")),
                )
                runtime = PureStudentRuntime(
                    student=context.student, backend=backend, history=context.history
                )
                report["inference"] = profile_inference(
                    runtime,
                    views,
                    evidence_level=context.evidence_level,
                )
        return report
