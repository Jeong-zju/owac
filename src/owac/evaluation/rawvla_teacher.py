"""Correct deployment-matched offline evaluation for the author RawVLA teacher.

Evaluation consumes only RAW/action/state arrays.  It never opens the paired
clean RGB files.  Each camera owns an independent persistent GRU/theta stream;
the native policy helpers build the three-key observation from Neural-ISP RGB,
and the pinned client postprocessor is called exactly once for physical output.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.raw.teacher_data import (
    ACTION_HORIZON,
    CAMERA_ORDER,
    TeacherData,
    TeacherEpisode,
    make_window,
)

K_BURST = 6
RECURRENT_ENDPOINTS = 6
REPLAN_STRIDE = 5
PI_LIBERO_COMMON = Path(
    "outputs/raw-distillation-20261020-sol/starvla-pinned-source/examples/"
    "simBenchmarks/LIBERO/eval_files/openpi/pi_libero_common.py"
)


class TeacherEvaluationError(RuntimeError):
    pass


def load_pinned_postprocessor():
    source = PI_LIBERO_COMMON
    if not source.is_file():
        raise TeacherEvaluationError(f"pinned client postprocessor absent: {source}")
    root = source.parents[5]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("owac_pinned_libero_common", source)
    if spec is None or spec.loader is None:
        raise TeacherEvaluationError("cannot load pinned client postprocessor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.postprocess_openpi_actions


def deterministic_action_noise(*, seed: int, source_id: str, frame: int) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:{source_id}:{frame}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
    return torch.randn((1, ACTION_HORIZON, 32), generator=generator, dtype=torch.float32)


def ring_indices(current: int, *, burst: int = K_BURST) -> np.ndarray:
    if current < 0:
        raise TeacherEvaluationError("negative current frame")
    return np.clip(
        current - np.arange(burst - 1, -1, -1, dtype=np.int64),
        0,
        current,
    )


def load_validation_catalog(path: str | Path, manifest: Any) -> list[tuple[str, int]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "owac.raw-teacher.validation-catalog.v1":
        raise TeacherEvaluationError("validation catalog schema differs")
    if payload.get("split") != "validation" or payload.get("source_demonstrations") != 5:
        raise TeacherEvaluationError("validation catalog is not the frozen five-source catalog")
    if (
        payload.get("filter_by_performance") is not False
        or payload.get("filter_by_noop") is not False
    ):
        raise TeacherEvaluationError("validation catalog was performance/no-op filtered")
    if payload.get("anchor_stride_environment_steps") != REPLAN_STRIDE:
        raise TeacherEvaluationError("validation catalog stride differs")
    if payload.get("flow_steps") != 10 or payload.get("action_horizon") != 10:
        raise TeacherEvaluationError("validation catalog flow/horizon differs")
    source_order = [source.source_demo_id for source in manifest.sources_for_split("validation")]
    expected = set(source_order)
    anchors = []
    for item in payload.get("anchors", []):
        source_id = str(item["source_demo_id"])
        frame = int(item["frame"])
        if source_id not in expected or frame < 0 or frame % REPLAN_STRIDE != 0:
            raise TeacherEvaluationError("catalog anchor source/frame is invalid")
        anchors.append((source_id, frame))
    if not anchors:
        raise TeacherEvaluationError("validation catalog has no anchors")
    catalog_order = list(dict.fromkeys(source for source, _ in anchors))
    if catalog_order != source_order:
        raise TeacherEvaluationError("validation catalog source order differs from manifest")
    return anchors


@dataclass(frozen=True, slots=True)
class FramePrediction:
    source_demo_id: str
    current_frame: int
    normalized_actions: np.ndarray
    physical_actions: np.ndarray
    target_actions7: np.ndarray
    target_normalized32: np.ndarray
    action_mask: np.ndarray
    source_frame_indices: np.ndarray


class TeacherOfflineEvaluator:
    def __init__(
        self,
        *,
        teacher: Any,
        data: TeacherData,
        stats: Any,
        catalog_path: str | Path | None = None,
    ) -> None:
        self.teacher = teacher
        self.data = data
        self.stats = stats
        self.catalog = (
            load_validation_catalog(catalog_path, data.manifest)
            if catalog_path is not None
            else None
        )
        if getattr(teacher, "frontend", None) is None:
            raise TeacherEvaluationError("teacher has no author frontend")
        if getattr(teacher, "policy", None) is None:
            raise TeacherEvaluationError("teacher has no frozen policy")

    def _ring(self, episode: TeacherEpisode, camera: str, current: int) -> np.ndarray:
        return np.asarray(episode.raw[camera][ring_indices(current)], dtype=np.uint8)

    def _policy_observation(
        self, camera_rgb: dict[str, torch.Tensor], normalized_state: np.ndarray, language: str
    ):
        # Use the exact native helpers. They consume RGB and do not call the RAW
        # frontend a second time. The helpers build native keys/batch/masks.
        batch = self.teacher.policy._prepare_examples(
            [
                {
                    "image": [camera_rgb["agentview"], camera_rgb["wrist"]],
                    "lang": language,
                    # Native helper pads state8 according to effective config.
                    # Actual LIBERO is discrete_state_input=false (0.0 pad, no
                    # state token); never prepad a 32-vector here.
                    "state": np.asarray(normalized_state, dtype=np.float32).reshape(8),
                }
            ],
            include_actions=False,
        )
        return self.teacher.policy._build_observation_from_batch(batch)

    def _predict(self, episode: TeacherEpisode, current: int) -> FramePrediction:
        device = next(self.teacher.frontend.model.parameters()).device
        camera_rgb = {}
        states = getattr(self, "_camera_state", {})
        thetas = getattr(self, "_camera_theta", {})
        for camera in CAMERA_ORDER:
            burst = self._ring(episode, camera, current)
            raw = torch.from_numpy(burst).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0
            raw = raw.to(device)
            with torch.no_grad():
                out = self.teacher.frontend.model(
                    raw, state=states.get(camera), theta_prev=thetas.get(camera)
                )
            states[camera] = out.state.detach()
            thetas[camera] = out.theta.detach()
            camera_rgb[camera] = out.rgb.detach()[0].float().clamp(0.0, 1.0)
        self._camera_state = states
        self._camera_theta = thetas
        window = make_window(episode, current)
        normalized = window.normalized(self.stats)
        # The public training loader clips quantile inputs.  The actual native
        # deployment client does not; retain that deliberate train/eval rule.
        raw_state = np.asarray(window.state, dtype=np.float32)
        state_stats = self.stats.state
        q01 = np.asarray(state_stats["q01"], dtype=np.float32)[: raw_state.size]
        q99 = np.asarray(state_stats["q99"], dtype=np.float32)[: raw_state.size]
        deployment_state = ((raw_state - q01) / (q99 - q01 + 1e-6) * 2 - 1).astype(np.float32)
        observation = self._policy_observation(
            camera_rgb,
            deployment_state,
            episode.source.metadata.get("language", "put the cream cheese in the bowl"),
        )
        noise = deterministic_action_noise(
            seed=101, source_id=episode.source.source_demo_id, frame=current
        ).to(device)
        with torch.no_grad():
            normalized_actions = self.teacher.policy.sample_actions(
                observation, noise=noise, num_steps=10
            )
        normalized_np = np.asarray(normalized_actions.detach().cpu(), dtype=np.float32)[0]
        # GT stays raw7. Only the explicit normalized target is built from stats.
        target7 = np.asarray(window.actions, dtype=np.float32).copy()
        target32 = np.asarray(normalized["padded_actions"], dtype=np.float32).copy()
        try:
            postprocess = load_pinned_postprocessor()
            physical = np.asarray(
                postprocess(
                    normalized_np,
                    np.asarray(episode.robot_states[current], dtype=np.float32),
                    {
                        "actions": {
                            "q01": np.asarray(self.stats.action["q01"], dtype=np.float32),
                            "q99": np.asarray(self.stats.action["q99"], dtype=np.float32),
                        }
                    },
                    model_name="PI05",
                    model_source="openpi",
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            raise TeacherEvaluationError(
                f"pinned physical/client postprocessing failed: {exc}"
            ) from exc
        return FramePrediction(
            source_demo_id=episode.source.source_demo_id,
            current_frame=int(current),
            normalized_actions=normalized_np,
            physical_actions=physical,
            target_actions7=target7,
            target_normalized32=target32,
            action_mask=np.asarray(window.action_mask, dtype=bool),
            source_frame_indices=window.source_frame_indices,
        )

    def evaluate(self, *, split: str = "validation", step: int = 0) -> dict[str, Any]:
        if split != "validation":
            raise TeacherEvaluationError("offline teacher evaluation may select only validation")
        if self.catalog is None:
            raise TeacherEvaluationError("frozen validation catalog is required")
        sources = self.data.manifest.sources_for_split("validation")
        if len(sources) != 5:
            raise TeacherEvaluationError("validation catalog must contain exactly five sources")
        # Preserve global RNG and module modes across the rollout scan.
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        frontend_training = getattr(self.teacher.frontend, "training", None)
        policy_training = getattr(self.teacher.policy, "training", None)
        self.teacher.frontend.eval()
        self.teacher.policy.eval()
        try:
            predictions = []
            normalized_records = []
            physical_records = []
            for source in sources:
                # RAW-only path: never open source.rgb_path.
                episode = self.data.episode_raw(source.source_demo_id)
                self._camera_state = {}
                self._camera_theta = {}
                catalog_frames = (
                    [
                        frame
                        for source_id, frame in self.catalog
                        if source_id == source.source_demo_id
                    ]
                    if self.catalog is not None
                    else list(range(0, episode.num_frames, REPLAN_STRIDE))
                )
                for current in catalog_frames:
                    if current >= episode.num_frames:
                        raise TeacherEvaluationError("catalog anchor exceeds source frames")
                    prediction = self._predict(episode, current)
                    predictions.append(prediction)
                    from owac.evaluation.raw_distillation import evaluate_windows

                    target32 = torch.from_numpy(prediction.target_normalized32).unsqueeze(0)
                    mask = torch.from_numpy(prediction.action_mask).unsqueeze(0)
                    normalized_records.append(
                        evaluate_windows(
                            torch.from_numpy(prediction.normalized_actions).unsqueeze(0),
                            target32,
                            mask,
                            split="validation",
                            step=int(step),
                            control_dim=7,
                            normalized=True,
                        )
                    )
                    # Physical prediction is already postprocessed once; compare
                    # directly with raw GT7, never denormalize the GT again.
                    physical_records.append(
                        evaluate_windows(
                            torch.from_numpy(prediction.physical_actions).unsqueeze(0),
                            torch.from_numpy(prediction.target_actions7).unsqueeze(0),
                            mask,
                            split="validation",
                            step=int(step),
                            control_dim=7,
                            normalized=False,
                        )
                    )
            from owac.evaluation.raw_distillation import aggregate_metrics

            normalized = aggregate_metrics(normalized_records, split="validation", step=step)
            physical = aggregate_metrics(physical_records, split="validation", step=step)
            return {
                "schema": "owac.rawvla-teacher.offline-eval.v1",
                "split": split,
                "step": int(step),
                "sources": [source.source_demo_id for source in sources],
                "predictions": len(predictions),
                "first1_mae7": physical.step1_mae7,
                "first5_mae7": physical.first5_mae7,
                "full10_mae7": physical.full10_mae7,
                "physical": asdict(physical),
                "normalized": asdict(normalized),
                "records": [asdict(record) for record in physical_records],
                "normalized_records": [asdict(record) for record in normalized_records],
                "predictions_data": predictions,
                "client_postprocess": (
                    "pinned examples/simBenchmarks/LIBERO/eval_files/openpi/"
                    "pi_libero_common.py:postprocess_openpi_actions"
                ),
            }
        finally:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            if frontend_training is True:
                self.teacher.frontend.train(True)
            if policy_training is True:
                self.teacher.policy.train(True)


def select_teacher_checkpoint(records: list[Mapping[str, Any]]) -> Any:
    """Physical real7 first5 minimum; earlier cumulative step on ties."""
    candidates = [
        record
        for record in records
        if record.get("split") == "validation"
        and record.get("first5_mae7") is not None
        and np.isfinite(float(record["first5_mae7"]))
    ]
    if not candidates:
        raise TeacherEvaluationError("no finite validation first5 records")
    return min(candidates, key=lambda record: (float(record["first5_mae7"]), int(record["step"])))


__all__ = [
    "FramePrediction",
    "K_BURST",
    "REPLAN_STRIDE",
    "TeacherEvaluationError",
    "TeacherOfflineEvaluator",
    "deterministic_action_noise",
    "load_pinned_postprocessor",
    "ring_indices",
    "select_teacher_checkpoint",
]


def save_teacher_frontend_artifact(
    teacher: Any,
    path: str | Path,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    state = {name: tensor.detach().cpu() for name, tensor in teacher.frontend.state_dict().items()}
    if len(state) != 83 or any(not name.startswith("model.") for name in state):
        raise TeacherEvaluationError("frontend artifact must contain the 83 wrapper model.* keys")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    payload = {
        "schema": "owac.rawvla-teacher.frontend-artifact.v1",
        "frontend": state,
        "frontend_key_names": sorted(state),
        "frontend_state_sha256": digest.hexdigest(),
        "identity": dict(identity),
    }
    target = Path(path)
    if target.exists():
        raise TeacherEvaluationError(f"refusing to overwrite teacher artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_teacher_frontend_artifact(
    path: str | Path, *, expected_identity: Mapping[str, Any]
) -> Mapping[str, torch.Tensor]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "owac.rawvla-teacher.frontend-artifact.v1":
        raise TeacherEvaluationError("teacher artifact schema differs")
    identity = payload.get("identity")
    required = {
        "backend",
        "tokenizer",
        "stats",
        "source",
        "burst_frames",
        "recurrent_endpoints",
        "cameras",
        "cumulative_updates",
        "input_provenance",
    }
    if not isinstance(identity, Mapping) or not required.issubset(identity):
        raise TeacherEvaluationError("teacher artifact identity schema is incomplete")
    if identity != dict(expected_identity):
        raise TeacherEvaluationError(
            "teacher artifact backend/source/tokenizer/stats identity differs"
        )
    state = payload.get("frontend")
    if not isinstance(state, Mapping) or len(state) != 83:
        raise TeacherEvaluationError("teacher artifact does not contain frontend83 keys")
    if sorted(state) != payload.get("frontend_key_names") or any(
        not name.startswith("model.") for name in state
    ):
        raise TeacherEvaluationError(
            "teacher artifact key names differ from wrapper model.* schema"
        )
    if _state_digest(state) != payload.get("frontend_state_sha256"):
        raise TeacherEvaluationError("teacher artifact frontend state digest differs")
    return state


def save_evaluation_result(result: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Save prediction arrays to NPZ and return a JSON-safe scalar result."""
    import json

    import numpy as np

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = result.pop("predictions_data", [])
    arrays: dict[str, np.ndarray] = {}
    for index, prediction in enumerate(predictions):
        for field in (
            "current_frame",
            "normalized_actions",
            "physical_actions",
            "target_actions7",
            "target_normalized32",
            "action_mask",
            "source_frame_indices",
        ):
            arrays[f"{index}_{field}"] = np.asarray(getattr(prediction, field))
    arrays["sources"] = np.asarray([prediction.source_demo_id for prediction in predictions])
    np.savez(output / "predictions.npz", **arrays)
    safe = json.loads(json.dumps(result))
    safe["prediction_npz"] = str(output / "predictions.npz")
    (output / "metrics.json").write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n")
    return safe
