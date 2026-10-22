"""Pure temporal student inference and offline paired evaluation.

The runtime loads only a temporal student artifact, the frozen policy backend,
tokenizer and statistics.  It has no teacher, Neural-ISP or target-cache import
on its construction path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.baselines.raw_temporal_siglip import (
    TemporalPacket,
    TemporalRawStemStudent,
    TemporalState,
    load_temporal_student_artifact,
)
from owac.evaluation.raw_distillation import (
    MetricRecord,
    aggregate_metrics,
    denormalize_actions,
    evaluate_windows,
    select_validation,
)
from owac.raw.temporal_packets import RawTemporalSample

__all__ = [
    "TemporalInferenceResult",
    "TemporalPureRuntime",
    "evaluate_temporal_rows",
    "evaluate_temporal_samples",
    "evaluate_temporal_visual_controls",
    "select_validation",
    "shuffle_visual_tokens",
    "zero_visual_tokens",
]


class TemporalPureUnavailable(RuntimeError):
    """A pure temporal artifact/backend dependency is absent."""


@dataclass(frozen=True, slots=True)
class TemporalInferenceResult:
    normalized_actions: torch.Tensor
    physical_actions: torch.Tensor
    first5_physical: torch.Tensor
    state: TemporalState | None
    state8_raw: torch.Tensor
    flow_steps: int
    run_stats: Mapping[str, Any] = field(default_factory=dict)


class TemporalPureRuntime:
    """Frozen backend + temporal student; no teacher/cache/ISP object reachable."""

    def __init__(
        self,
        *,
        student: TemporalRawStemStudent,
        bridge: Any,
        stats: Any,
        tokenizer: Any = None,
        freeze: bool = True,
    ) -> None:
        if getattr(student, "bridge_projection", None) is not None:
            raise TemporalPureUnavailable("temporal student bridge projection must be None")
        if getattr(bridge, "projection", None) is not None:
            raise TemporalPureUnavailable("backend bridge projection must be None")
        self.student = student
        self.bridge = bridge
        self.stats = stats
        self.tokenizer = tokenizer
        self.student.eval()
        self.bridge.base.eval()
        if freeze:
            for parameter in self.student.parameters():
                parameter.requires_grad_(False)
            for parameter in bridge.base.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        checkpoint: str | Path,
        tokenizer: str | Path,
        statistics: str | Path,
        runtime_root: str | Path | None = None,
        frontend_factory: Any = None,
        device: str | torch.device = "cpu",
    ) -> TemporalPureRuntime:
        from owac.backends.rawvla import load_audited, new_model
        from owac.backends.rawvla_features_n02 import (
            RawVLAFeaturePolicy,
            StaticTokenSource,
            load_checkpoint_state,
            protocol_for,
        )
        from owac.raw.distillation_pairs import NormalisationStats

        if runtime_root is not None:
            import sys

            root = str(runtime_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        student = load_temporal_student_artifact(
            artifact_path,
            frontend_factory=frontend_factory,
        )
        if student.input_mode == "rgb3_control":
            raise TemporalPureUnavailable("RGB diagnostic artifact is not a pure RAW deployment")
        base = new_model("libero", str(tokenizer))
        state = load_checkpoint_state(str(checkpoint))
        audit = load_audited(base, state)
        for key in ("active_missing", "unexpected", "shape_mismatch"):
            if audit.get(key):
                raise TemporalPureUnavailable(f"frozen backend load audit has {key}")
        if getattr(base, "raw_frontend", None) is not None:
            raise TemporalPureUnavailable("pure backend still has a reachable raw_frontend")
        bridge = RawVLAFeaturePolicy(
            base,
            StaticTokenSource(None),
            protocol=protocol_for("libero"),
            token_dim=student.geometry.projector_out,
            tokens_per_camera=student.geometry.num_positions,
            image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            freeze_backend=True,
        )
        # The student owns its inherited deep copy.  Release the now-unused
        # backend visual path only after the bridge and student are built.
        core = getattr(getattr(getattr(base, "vlm", None), "model", None), "model", None)
        if core is not None:
            for attribute in ("vision_tower", "multi_modal_projector"):
                if hasattr(core, attribute):
                    delattr(core, attribute)
        base.to(device)
        student.to(device)
        stats = NormalisationStats.from_json(statistics, mode="quantile")
        return cls(
            student=student, bridge=bridge, stats=stats, tokenizer=getattr(base, "tokenizer", None)
        )

    def _device(self) -> torch.device:
        return next(self.student.parameters()).device

    def reset(self) -> None:
        # State is explicit; callers own it.  This method exists for lifecycle
        # parity and does not retain a hidden global state.
        return None

    def dependency_audit(self) -> dict[str, bool]:
        forbidden = []
        for name, _value in vars(self).items():
            if name.lower() in {"teacher", "isp", "cache", "teacher_cache"}:
                forbidden.append(name)
        return {
            "teacher": False,
            "isp": False,
            "cache": False,
            "forbidden_attributes": bool(forbidden),
        }

    def infer_visual_tokens(
        self,
        tokens: Any,
        *,
        state8: torch.Tensor | np.ndarray,
        language: str,
        noise: torch.Tensor | None = None,
        flow_steps: int = 10,
    ) -> TemporalInferenceResult:
        """Run the frozen backend on already projected student visual tokens."""
        raw_state = torch.as_tensor(state8, dtype=torch.float32, device=self._device()).reshape(
            1, -1
        )
        if raw_state.shape[1] != 8:
            raise TemporalPureUnavailable("state8 must be canonical robot observation8")
        from owac.backends.rawvla_language import prepare_language_state

        lang_tokens, lang_masks, state32 = prepare_language_state(
            self.bridge,
            [language],
            raw_state.cpu().numpy(),
            device=self._device(),
        )
        with torch.inference_mode():
            prediction = self.bridge.predict_action_from_visual_tokens(
                tokens,
                lang_tokens,
                lang_masks,
                state32,
                noise=noise,
                num_steps=int(flow_steps),
            )
        normalized = torch.as_tensor(
            prediction["normalized_actions"], dtype=torch.float32, device=self._device()
        )
        physical = denormalize_actions(normalized[..., :7], self.stats)
        return TemporalInferenceResult(
            normalized_actions=normalized,
            physical_actions=physical,
            first5_physical=physical[:, :5],
            state=None,
            state8_raw=raw_state.detach().clone(),
            flow_steps=int(flow_steps),
            run_stats=prediction.get("run_stats", {}),
        )

    def infer_packet(
        self,
        packet: TemporalPacket,
        state: TemporalState | None,
        *,
        state8: torch.Tensor | np.ndarray,
        language: str,
        noise: torch.Tensor | None = None,
        flow_steps: int = 10,
    ) -> tuple[TemporalInferenceResult, TemporalState]:
        if not isinstance(packet, TemporalPacket):
            raise TemporalPureUnavailable("infer_packet requires a TemporalPacket")
        with torch.inference_mode():
            output, new_state = self.student(packet, state)
        result = self.infer_visual_tokens(
            output.tokens,
            state8=state8,
            language=language,
            noise=noise,
            flow_steps=flow_steps,
        )
        return replace(result, state=new_state), new_state


def _aggregate_physical_and_padding(
    records: Sequence[MetricRecord], *, split: str, step: int
) -> MetricRecord:
    physical = aggregate_metrics(
        [record for record in records if not record.normalized], split=split, step=step
    )
    normalized = aggregate_metrics(
        [record for record in records if record.normalized], split=split, step=step
    )
    # Physical 7D errors select the checkpoint; padding remains a separately
    # labelled normalized-space diagnostic and never dilutes that metric.
    return replace(physical, padding_mae=normalized.padding_mae)


def _identity_seed(identity: str) -> int:
    value = 0
    for index, character in enumerate(identity):
        value = (value + (index + 1) * ord(character)) & 0xFFFFFFFF
    return value


def _validation_noise(
    source_id: str, endpoint: int, seed: int, device: torch.device
) -> torch.Tensor:
    identity = f"{source_id}:{int(endpoint)}"
    seed_value = int(seed) + _identity_seed(identity)
    generator = torch.Generator(device="cpu").manual_seed(seed_value)
    return torch.randn((1, 10, 32), generator=generator, dtype=torch.float32).to(device)


def evaluate_temporal_samples(
    runtime: TemporalPureRuntime,
    samples: Sequence[RawTemporalSample],
    *,
    split: str,
    step: int = 0,
    flow_steps: int = 10,
    noise_seed: int | None = None,
) -> MetricRecord:
    """Pure RAW/label evaluation; no teacher/cache/target-row object is read."""
    if split not in {"validation", "test"}:
        raise TemporalPureUnavailable("offline evaluation accepts validation/test only")
    if any(sample.split != split for sample in samples):
        raise TemporalPureUnavailable("sample split differs from requested split")
    was_training = runtime.student.training
    runtime.student.eval()
    records = []
    state: TemporalState | None = None
    current_source = None
    try:
        for sample in samples:
            if sample.source_id != current_source:
                current_source = sample.source_id
                state = None
            noise = (
                None
                if noise_seed is None
                else _validation_noise(
                    sample.source_id,
                    sample.endpoint,
                    int(noise_seed),
                    runtime._device(),
                )
            )
            result, state = runtime.infer_packet(
                sample.packet,
                state,
                state8=sample.state8_raw,
                language=sample.language,
                noise=noise,
                flow_steps=flow_steps,
            )
            prediction = result.normalized_actions
            target = sample.actions32.unsqueeze(0).to(prediction.device)
            mask = sample.future_mask.unsqueeze(0).to(prediction.device)
            records.append(
                evaluate_windows(
                    prediction,
                    target,
                    mask,
                    split=split,
                    step=int(step),
                    control_dim=7,
                    normalized=True,
                )
            )
            physical = result.physical_actions
            physical_target = sample.actions7.unsqueeze(0).to(physical.device)
            records.append(
                evaluate_windows(
                    physical,
                    physical_target,
                    mask,
                    split=split,
                    step=int(step),
                    control_dim=7,
                    normalized=False,
                )
            )
    finally:
        runtime.student.train(was_training)
    if not records:
        raise TemporalPureUnavailable("no temporal samples to evaluate")
    return _aggregate_physical_and_padding(records, split=split, step=step)


def evaluate_temporal_rows(
    runtime: TemporalPureRuntime,
    dataset: Any,
    rows: Sequence[Any],
    *,
    split: str,
    step: int = 0,
    flow_steps: int = 10,
    noise_seed: int | None = None,
    state8_by_row: Mapping[tuple[str, int], torch.Tensor] | None = None,
    language: str = "put the cream cheese in the bowl",
) -> MetricRecord:
    """Cached-row compatibility path; pure path uses ``evaluate_temporal_samples``."""
    from owac.raw.temporal_distillation import TemporalPacketDataset, TemporalTargetRow

    if not isinstance(dataset, TemporalPacketDataset):
        raise TemporalPureUnavailable("cached evaluation requires TemporalPacketDataset")
    if not all(isinstance(row, TemporalTargetRow) for row in rows):
        raise TemporalPureUnavailable("cached evaluation requires TemporalTargetRow rows")
    if split not in {"validation", "test"}:
        raise TemporalPureUnavailable("offline evaluation accepts validation/test only")
    records = []
    state: TemporalState | None = None
    current_source = None
    for row in rows:
        if row.source_id != current_source:
            current_source = row.source_id
            state = None
        packet = dataset.packet_for_endpoint(row.source_id, row.endpoint, row=row)
        if state8_by_row is not None:
            state8 = state8_by_row[(row.source_id, row.endpoint)]
        else:
            state8 = row.state_raw
        noise = (
            None
            if noise_seed is None
            else _validation_noise(
                row.source_id,
                row.endpoint,
                int(noise_seed),
                runtime._device(),
            )
        )
        result, state = runtime.infer_packet(
            packet,
            state,
            state8=state8,
            language=language,
            noise=noise,
            flow_steps=flow_steps,
        )
        prediction = result.normalized_actions
        target = row.actions32.unsqueeze(0).to(prediction.device)
        mask = row.future_mask.unsqueeze(0).to(prediction.device)
        records.append(
            evaluate_windows(
                prediction,
                target,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=True,
            )
        )
        physical = result.physical_actions
        physical_target = row.actions7.unsqueeze(0).to(physical.device)
        records.append(
            evaluate_windows(
                physical,
                physical_target,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=False,
            )
        )
    if not records:
        raise TemporalPureUnavailable("no temporal rows to evaluate")
    return _aggregate_physical_and_padding(records, split=split, step=step)


def _visual_metric_dict(record: MetricRecord) -> dict[str, Any]:
    return {
        "first1_mae7": record.step1_mae7,
        "first5_mae7": record.first5_mae7,
        "full10_mae7": record.full10_mae7,
        "translation_mae": record.groups.get("translation"),
        "rotation_mae": record.groups.get("rotation"),
        "gripper_mae": record.groups.get("gripper"),
        "padding_mae": record.padding_mae,
        "valid_step1": record.valid_step1,
        "valid_first5": record.valid_first5,
        "valid_full10": record.valid_full10,
    }


def _diagnostic_action_delta(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor, *, dims: int = 7
) -> float:
    difference = (left[..., :dims] - right[..., :dims]).abs()
    selected = difference[mask.bool()]
    if selected.numel() == 0:
        raise TemporalPureUnavailable("control comparison has no valid action denominator")
    return float(selected.mean().detach().cpu())


def _control_tokens(target: Any, donor: Any) -> Any:
    if tuple(target.camera_ids) != tuple(donor.camera_ids):
        raise TemporalPureUnavailable("donor camera identities/order differ from target")
    if tuple(target.values.shape) != tuple(donor.values.shape):
        raise TemporalPureUnavailable("donor grid/token shape differs from target")
    if (
        target.positions is None
        or donor.positions is None
        or not torch.equal(target.positions, donor.positions)
    ):
        raise TemporalPureUnavailable("donor positions/grid differ from target")
    if (
        target.mask is None
        or donor.mask is None
        or tuple(target.mask.shape) != tuple(donor.mask.shape)
    ):
        raise TemporalPureUnavailable("donor mask shape differs from target")
    if not torch.equal(target.mask.to(torch.bool), donor.mask.to(torch.bool)):
        raise TemporalPureUnavailable("donor mask differs from target; refusing to swap semantics")
    # Donor supplies only projected values. Camera identities, physical
    # positions, scale and mask remain exactly the target's.
    return replace(
        target,
        values=donor.values,
        mask=target.mask,
        camera_ids=target.camera_ids,
        positions=target.positions,
        scale=target.scale,
        validity=target.validity,
        uncertainty=target.uncertainty,
    )


def evaluate_temporal_visual_controls(
    runtime: TemporalPureRuntime,
    samples: Sequence[RawTemporalSample],
    *,
    split: str,
    endpoints: Sequence[int] | None = None,
    flow_steps: int = 10,
    seed: int = 101,
    step: int = 0,
) -> dict[str, Any]:
    """Pure cross-demo visual-control diagnostic; never selects a checkpoint.

    The target's state8, language, actions, action mask and camera semantics are
    identical in baseline, zero-visual and donor-visual conditions. The donor is
    the cyclic next different source in supplied order at the exact same endpoint.
    """
    if split not in {"validation", "test"}:
        raise TemporalPureUnavailable("visual controls accept validation/test only")
    values = list(samples)
    if not values:
        raise TemporalPureUnavailable("visual controls require samples")
    grouped: dict[str, list[RawTemporalSample]] = {}
    source_order: list[str] = []
    last_source: str | None = None
    seen_endpoints: dict[str, set[int]] = {}
    camera_ids: tuple[str, ...] | None = None
    for sample in values:
        if sample.split != split:
            raise TemporalPureUnavailable("visual-control sample split differs")
        source = sample.source_id
        if source not in grouped:
            if source in source_order:
                raise TemporalPureUnavailable("visual-control source blocks are interleaved")
            grouped[source] = []
            source_order.append(source)
            seen_endpoints[source] = set()
        elif last_source != source:
            raise TemporalPureUnavailable("visual-control source blocks are out of order")
        if sample.endpoint < 0 or sample.endpoint % 5:
            raise TemporalPureUnavailable("visual-control endpoint must be nonnegative/5-aligned")
        if sample.endpoint in seen_endpoints[source]:
            raise TemporalPureUnavailable("visual-control duplicate source/endpoint")
        if grouped[source] and sample.endpoint <= grouped[source][-1].endpoint:
            raise TemporalPureUnavailable("visual-control endpoints are out of order")
        if sample.endpoint != (0 if not grouped[source] else grouped[source][-1].endpoint + 5):
            raise TemporalPureUnavailable(
                "visual-control source endpoints must be contiguous from 0"
            )
        if sample.packet.source_demo_id != source or sample.packet.endpoint != sample.endpoint:
            raise TemporalPureUnavailable("visual-control packet/source identity differs")
        if tuple(sample.frame_ids) != tuple(sample.packet.frame_ids):
            raise TemporalPureUnavailable("visual-control sample frame ids differ")
        frame_cameras = tuple(sample.packet.frames[-1].camera_ids)
        if camera_ids is None:
            camera_ids = frame_cameras
        elif camera_ids != frame_cameras:
            raise TemporalPureUnavailable("visual-control camera count/order differs")
        grouped[source].append(sample)
        seen_endpoints[source].add(sample.endpoint)
        last_source = source
    if len(source_order) < 2:
        raise TemporalPureUnavailable("visual controls require at least two sources")
    for source in source_order:
        if grouped[source][0].endpoint != 0:
            raise TemporalPureUnavailable("visual-control source must start at endpoint 0")
    requested = set(int(value) for value in endpoints) if endpoints is not None else None
    if requested is not None:
        for endpoint in requested:
            if endpoint < 0 or endpoint % 5:
                raise TemporalPureUnavailable("selected visual-control endpoint is invalid")

    was_training = runtime.student.training
    runtime.student.eval()
    token_map: dict[tuple[str, int], Any] = {}
    try:
        with torch.inference_mode():
            for source in source_order:
                state: TemporalState | None = None
                for sample in grouped[source]:
                    output, state = runtime.student(sample.packet, state)
                    token_map[(source, sample.endpoint)] = output.tokens
    finally:
        runtime.student.train(was_training)

    supplied_total = len(values)
    excluded: list[dict[str, Any]] = []
    donor_mapping: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    baseline_physical = []
    baseline_normalized = []
    zero_physical = []
    zero_normalized = []
    donor_physical = []
    donor_normalized = []
    paired = {"zero_visual": [], "cross_demo": []}
    action_denominator = 0
    selected_total = 0
    for source_index, source in enumerate(source_order):
        donor_source = source_order[(source_index + 1) % len(source_order)]
        if donor_source == source:
            continue
        donor_endpoints = {item.endpoint for item in grouped[donor_source]}
        for sample in grouped[source]:
            if requested is not None and sample.endpoint not in requested:
                continue
            selected_total += 1
            anchor = {
                "source_id": source,
                "endpoint": sample.endpoint,
                "donor_source_id": donor_source,
            }
            donor_mapping.append(anchor)
            if sample.endpoint not in donor_endpoints:
                excluded.append({**anchor, "reason": "donor_missing_endpoint"})
                continue
            target_tokens = token_map[(source, sample.endpoint)]
            donor_tokens = token_map[(donor_source, sample.endpoint)]
            try:
                donor_control = _control_tokens(target_tokens, donor_tokens)
            except TemporalPureUnavailable as exc:
                excluded.append({**anchor, "reason": str(exc)})
                continue
            identity = f"{split}:{source}:{sample.endpoint}:{donor_source}"
            noise_seed = int(seed) + _identity_seed(identity)
            generator = torch.Generator(device="cpu").manual_seed(noise_seed)
            noise = torch.randn((1, 10, 32), generator=generator, dtype=torch.float32)
            noise = noise.to(target_tokens.values.device)
            baseline = runtime.infer_visual_tokens(
                target_tokens,
                state8=sample.state8_raw,
                language=sample.language,
                noise=noise,
                flow_steps=flow_steps,
            )
            zero = runtime.infer_visual_tokens(
                zero_visual_tokens(target_tokens),
                state8=sample.state8_raw,
                language=sample.language,
                noise=noise,
                flow_steps=flow_steps,
            )
            donor = runtime.infer_visual_tokens(
                donor_control,
                state8=sample.state8_raw,
                language=sample.language,
                noise=noise,
                flow_steps=flow_steps,
            )
            target32 = sample.actions32.unsqueeze(0).to(baseline.normalized_actions.device)
            target7 = sample.actions7.unsqueeze(0).to(baseline.physical_actions.device)
            mask = sample.future_mask.unsqueeze(0).to(baseline.physical_actions.device)
            action_denominator += int(mask.sum())
            base_record = evaluate_windows(
                baseline.physical_actions,
                target7,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=False,
            )
            zero_record = evaluate_windows(
                zero.physical_actions,
                target7,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=False,
            )
            donor_record = evaluate_windows(
                donor.physical_actions,
                target7,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=False,
            )
            base_norm = evaluate_windows(
                baseline.normalized_actions,
                target32,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=True,
            )
            zero_norm = evaluate_windows(
                zero.normalized_actions,
                target32,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=True,
            )
            donor_norm = evaluate_windows(
                donor.normalized_actions,
                target32,
                mask,
                split=split,
                step=int(step),
                control_dim=7,
                normalized=True,
            )
            baseline_physical.append(base_record)
            baseline_normalized.append(base_norm)
            zero_physical.append(zero_record)
            zero_normalized.append(zero_norm)
            donor_physical.append(donor_record)
            donor_normalized.append(donor_norm)
            paired["zero_visual"].append(
                _diagnostic_action_delta(zero.physical_actions, baseline.physical_actions, mask)
            )
            paired["cross_demo"].append(
                _diagnostic_action_delta(donor.physical_actions, baseline.physical_actions, mask)
            )
            matched.append(
                {
                    **anchor,
                    "noise_seed": noise_seed,
                    "valid_actions": int(mask.sum()),
                    "baseline": _visual_metric_dict(base_record),
                    "zero_visual": _visual_metric_dict(zero_record),
                    "cross_demo": _visual_metric_dict(donor_record),
                    "baseline_padding": base_norm.padding_mae,
                    "zero_visual_padding": zero_norm.padding_mae,
                    "cross_demo_padding": donor_norm.padding_mae,
                    "paired_action_delta": {
                        "zero_visual": paired["zero_visual"][-1],
                        "cross_demo": paired["cross_demo"][-1],
                    },
                    "camera_ids": list(target_tokens.camera_ids),
                    "token_shape": list(target_tokens.values.shape),
                    "mask_strategy": "target_mask_preserved",
                    "donor_mask_used": False,
                }
            )

    def aggregate(
        physical: Sequence[MetricRecord], normalized: Sequence[MetricRecord]
    ) -> dict[str, Any]:
        if not physical:
            return {}
        record = _aggregate_physical_and_padding(
            list(physical) + list(normalized), split=split, step=int(step)
        )
        return _visual_metric_dict(record)

    if not matched:
        raise TemporalPureUnavailable("no matched target/donor visual-control anchors")
    return {
        "status": "visual_controls_complete",
        "schema": "owac.temporal.visual-controls.v1",
        "split": split,
        "step": int(step),
        "flow_steps": int(flow_steps),
        "supplied_total": supplied_total,
        "selected_total": selected_total,
        "matched_count": len(matched),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "donor_mapping": donor_mapping,
        "matched": matched,
        "conditions": {
            "baseline": aggregate(baseline_physical, baseline_normalized),
            "zero_visual": aggregate(zero_physical, zero_normalized),
            "cross_demo": aggregate(donor_physical, donor_normalized),
        },
        "paired_action_delta": {
            "zero_visual": float(np.mean(paired["zero_visual"])),
            "cross_demo": float(np.mean(paired["cross_demo"])),
        },
        "denominators": {
            "valid_actions": action_denominator,
            "matched_anchors": len(matched),
        },
        "provenance": {
            "source_order": source_order,
            "camera_ids": list(camera_ids or ()),
            "donor_rule": "cyclic_next_different_source_same_split_exact_endpoint",
            "noise_rule": "independent_torch_generator_seed_plus_stable_anchor_identity",
            "mask_strategy": "target_camera_ids_positions_mask_preserved; donor values only",
            "zero_visual": "values_zero_and_mask_zero",
            "teacher_or_cache_used": False,
            "target_position_intervention": "not_run",
            "checkpoint_selection": False,
            "test_selection": False,
        },
    }


def zero_visual_tokens(tokens: Any) -> Any:
    values = torch.zeros_like(tokens.values)
    mask = torch.zeros_like(tokens.mask, dtype=torch.bool) if tokens.mask is not None else None
    return type(tokens)(
        values=values,
        mask=mask,
        camera_ids=tokens.camera_ids,
        positions=tokens.positions,
        scale=getattr(tokens, "scale", None),
        validity=getattr(tokens, "validity", None),
    )


def shuffle_visual_tokens(tokens: Any, permutation: Sequence[int]) -> Any:
    if len(permutation) != len(tokens.camera_ids):
        raise TemporalPureUnavailable("permutation must match camera groups")
    per_camera = tokens.values.shape[1] // len(tokens.camera_ids)
    groups = [
        tokens.values[:, index * per_camera : (index + 1) * per_camera]
        for index in range(len(tokens.camera_ids))
    ]
    values = torch.cat([groups[index] for index in permutation], dim=1)
    mask = None
    if tokens.mask is not None:
        masks = [
            tokens.mask[:, index * per_camera : (index + 1) * per_camera]
            for index in range(len(tokens.camera_ids))
        ]
        mask = torch.cat([masks[index] for index in permutation], dim=1)
    return type(tokens)(
        values=values,
        mask=mask,
        camera_ids=tuple(tokens.camera_ids[index] for index in permutation),
        positions=tokens.positions,
        scale=getattr(tokens, "scale", None),
        validity=getattr(tokens, "validity", None),
    )
