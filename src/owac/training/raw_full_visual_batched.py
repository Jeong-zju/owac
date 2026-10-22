"""Vectorized full-visual temporal forward for per-rank sample batches.

The helper is deliberately a forward/loss primitive, not a second trainer.  It
keeps the current temporal-state semantics: each anchor burns its own prefix
under the current weights with no grad, then only that anchor's six unroll
endpoints participate in the differentiable graph.  Batch dimensions are used
for the stem, recurrence and the inherited SigLIP frontend; no sample can read
another sample's hidden state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from owac.baselines.raw_stem_siglip import (
    RawStemInput,
    _forward_frontend,
    _patch_pool,
    _prepare_measurement,
    _stem_patch_tokens,
)

BATCHED_HISTORY_LENGTH = 6


class BatchedTemporalError(ValueError):
    """The supplied samples cannot be packed without changing semantics."""


@dataclass(slots=True)
class BatchedTemporalRequest:
    samples: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise BatchedTemporalError("batched request requires at least one sample")
        object.__setattr__(self, "samples", tuple(self.samples))


@dataclass(slots=True)
class BatchedForwardResult:
    projected_tokens: torch.Tensor
    token_mask: torch.Tensor
    support_fraction: torch.Tensor
    hidden: torch.Tensor
    source_ids: tuple[str, ...]
    endpoints: tuple[int, ...]


def _as_batched_mask(
    value: Any, batch: int, cameras: int, height: int, width: int, fill: bool
) -> Any:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.bool)
    if tensor.ndim == 4:
        if tensor.shape[0] == 1:
            tensor = tensor.expand(batch, -1, -1, -1)
        if tuple(tensor.shape) != (batch, cameras, height, width):
            raise BatchedTemporalError("mask shape cannot be packed across samples")
        return tensor
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.expand(batch, -1, -1)
    if tuple(tensor.shape) != (batch, cameras, height, width):
        raise BatchedTemporalError("mask shape cannot be packed across samples")
    return tensor


def _stack_inputs(inputs: Sequence[RawStemInput]) -> RawStemInput:
    if not inputs:
        raise BatchedTemporalError("cannot stack an empty frame batch")
    first = inputs[0]
    for item in inputs[1:]:
        if (
            item.input_mode != first.input_mode
            or item.camera_ids != first.camera_ids
            or item.units != first.units
            or item.cfa_pattern != first.cfa_pattern
            or tuple(item.values.shape[1:]) != tuple(first.values.shape[1:])
        ):
            raise BatchedTemporalError("raw inputs differ in camera/geometry/units")
        if item.physical_origins != first.physical_origins:
            raise BatchedTemporalError("physical origins differ across batched samples")
    values = torch.cat([item.values for item in inputs], dim=0)

    def masks_for(name: str, fill: bool) -> Any:
        values_out = []
        any_mask = False
        for item in inputs:
            mask = getattr(item, name)
            if mask is not None:
                any_mask = True
            if mask is None:
                shape = item.values.shape
                if name == "camera_present":
                    shape = (shape[0], shape[1])
                else:
                    shape = (shape[0], shape[1], shape[3], shape[4])
                mask = torch.full(shape, bool(fill), dtype=torch.bool, device=item.values.device)
            values_out.append(torch.as_tensor(mask, dtype=torch.bool))
        if not any_mask:
            return None
        return torch.cat(values_out, dim=0)

    camera_present = masks_for("camera_present", True)
    valid_mask = masks_for("valid_mask", True)
    bad_mask = masks_for("bad_mask", False)
    saturated_mask = masks_for("saturated_mask", False)
    missing_mask = masks_for("missing_mask", False)
    for name in ("black_level", "white_level", "exposure", "noise"):
        if any(getattr(item, name) is not None for item in inputs):
            raise BatchedTemporalError(
                f"non-None {name} is not supported in the active batched RAW path"
            )
    return RawStemInput(
        values=values,
        input_mode=first.input_mode,
        camera_ids=first.camera_ids,
        units=first.units,
        valid_mask=valid_mask,
        bad_mask=bad_mask,
        saturated_mask=saturated_mask,
        missing_mask=missing_mask,
        cfa_pattern=first.cfa_pattern,
        physical_origins=first.physical_origins,
        camera_present=camera_present,
    )


def move_raw_input_to_device(value: RawStemInput, device: torch.device) -> RawStemInput:
    """Move only one stacked current burst; prefix packets stay on CPU."""
    if value.values.device == device:
        return value
    updates = {"values": value.values.to(device)}
    for name in ("valid_mask", "bad_mask", "saturated_mask", "missing_mask", "camera_present"):
        item = getattr(value, name)
        if torch.is_tensor(item):
            updates[name] = item.to(device)
    return replace(value, **updates)


def flatten_teacher_target(
    teacher_tokens: torch.Tensor, teacher_token_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match native [camera,patch,hidden]/[camera] expansion exactly."""
    if teacher_tokens.ndim != 3:
        raise BatchedTemporalError("teacher_tokens must be [camera,patch,hidden]")
    if teacher_token_mask.ndim != 1:
        raise BatchedTemporalError("teacher_token_mask must be [camera]")
    if int(teacher_token_mask.shape[0]) != int(teacher_tokens.shape[0]):
        raise BatchedTemporalError("teacher mask camera axis differs from teacher tokens")
    teacher_flat = teacher_tokens.reshape(1, -1, teacher_tokens.shape[-1])
    mask_flat = (
        teacher_token_mask[:, None]
        .expand(int(teacher_tokens.shape[0]), int(teacher_tokens.shape[1]))
        .reshape(1, -1)
    )
    return teacher_flat, mask_flat


def stack_teacher_targets(samples: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack native [camera,patch,hidden] targets and expand the camera masks."""
    teachers = []
    masks = []
    for sample in samples:
        teacher = sample.target.teacher_tokens
        mask = sample.target.teacher_token_mask
        if not torch.is_tensor(teacher) or teacher.ndim != 3:
            raise BatchedTemporalError("teacher_tokens must be [camera,patch,hidden]")
        if not torch.is_tensor(mask) or mask.ndim != 1:
            raise BatchedTemporalError("teacher_token_mask must be [camera]")
        if int(mask.shape[0]) != int(teacher.shape[0]):
            raise BatchedTemporalError("teacher mask camera axis differs from teacher tokens")
        teachers.append(teacher)
        masks.append(mask)
    teacher = torch.stack(teachers, dim=0)
    mask = torch.stack(masks, dim=0)
    batch, cameras, patches, width = teacher.shape
    teacher_flat = teacher.reshape(batch, cameras * patches, width)
    mask_flat = mask[:, :, None].expand(batch, cameras, patches).reshape(batch, cameras * patches)
    return teacher_flat, mask_flat


def _masked_mean(tokens: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = masks.sum(dim=1)
    weighted = (tokens * masks.unsqueeze(-1)).sum(dim=1)
    mean = weighted / count.clamp(min=1).unsqueeze(-1)
    has_support = count > 0
    mean = torch.where(has_support.unsqueeze(-1), mean, torch.zeros_like(mean))
    return mean, has_support


def _stack_packets(packets: Sequence[Any]) -> tuple[RawStemInput, torch.Tensor]:
    if any(len(packet.frames) != BATCHED_HISTORY_LENGTH for packet in packets):
        raise BatchedTemporalError("batched RAW path requires six causal frames per packet")
    frames = [frame for packet in packets for frame in packet.frames]
    if any(int(frame.batch_size) != 1 for frame in frames):
        raise BatchedTemporalError("each packet frame must have batch size 1")
    padding = torch.tensor(
        [list(packet.padding_mask) for packet in packets],
        dtype=torch.bool,
    )
    return _stack_inputs(frames), padding


def _burst_components(
    student: Any, packets: Sequence[Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RawStemInput]:
    batch = len(packets)
    stacked, padding = _stack_packets(packets)
    device = student.gate_logit.device
    stacked = move_raw_input_to_device(stacked, device)
    padding = padding.to(device)
    prepared = _prepare_measurement(stacked, student.geometry)
    patch_tokens, _ = _stem_patch_tokens(
        student.base.vision_tower,
        student.geometry,
        stacked,
        prepared=prepared,
    )
    cameras = len(stacked.camera_ids)
    num_patches = int(student.geometry.num_patches)
    hidden_size = int(student.geometry.hidden_size)
    patch_tokens = patch_tokens.reshape(
        batch, BATCHED_HISTORY_LENGTH, cameras, num_patches, hidden_size
    )
    stride = student.geometry.packed_patch_stride(stacked.input_mode)
    usable = prepared["usable"].flatten(0, 1).to(torch.float32).unsqueeze(1)
    support = _patch_pool(usable, stride, "avg")[:, 0].reshape(
        batch, BATCHED_HISTORY_LENGTH, cameras, num_patches
    )
    present = prepared["present"].reshape(batch, BATCHED_HISTORY_LENGTH, cameras)
    current_present = present[:, -1]
    # Match native _patch_burst: every burst frame is masked by its OWN present
    # flag; only the recurrence gate uses the current endpoint's present flag.
    masks = (support >= 1.0 - 1e-6) & present[:, :, :, None]
    masks = masks & ~padding.to(support.device)[:, :, None, None]
    mean, has_support = _masked_mean(patch_tokens, masks)
    current_input = move_raw_input_to_device(
        _stack_inputs([packet.frames[-1] for packet in packets]), device
    )
    return mean, has_support, current_present, current_input, support


def _batched_burst_step(
    student: Any,
    packets: Sequence[Any],
    previous: torch.Tensor | None,
    has_previous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RawStemInput, torch.Tensor]:
    mean, has_support, current_present, current_input, support = _burst_components(student, packets)
    if previous is None:
        previous = torch.zeros_like(mean)
    elif previous.dtype != mean.dtype:
        raise BatchedTemporalError(
            "prefix hidden dtype differs from current stem dtype; refusing to change numerics"
        )
    reset = torch.tensor(
        [bool(packet.reset) for packet in packets],
        dtype=torch.bool,
        device=previous.device,
    )
    use_previous = has_previous & ~reset
    alpha = student.gate_alpha.reshape(1, 1, 1, -1)
    advanced = (1.0 - alpha) * mean + alpha * previous
    recurrent = torch.where(has_support.unsqueeze(-1), advanced, previous)
    recurrent = torch.where(current_present[:, :, None, None], recurrent, previous)
    fresh = torch.where(has_support.unsqueeze(-1), mean, torch.zeros_like(mean))
    hidden = torch.where(use_previous[:, None, None, None], recurrent, fresh)
    return hidden, has_support, current_present, current_input, support


def _validate_samples(student: Any, samples: Sequence[Any]) -> None:
    if not samples:
        raise BatchedTemporalError("batched forward requires samples")
    stride = int(student.endpoint_stride)
    camera_order = tuple(samples[0].unroll_packets[0].frames[0].camera_ids)
    source = str(samples[0].source_id)
    for sample in samples:
        if str(sample.source_id) != source:
            # Different sources are allowed in a packed rank batch; only the
            # sequence within each sample is constrained below.
            pass
        packets = list(sample.prefix_packets) + list(sample.unroll_packets)
        if len(sample.unroll_packets) != BATCHED_HISTORY_LENGTH:
            raise BatchedTemporalError("each sample needs exactly six unroll endpoints")
        previous: int | None = None
        first = True
        for packet in packets:
            if str(packet.source_demo_id) != str(sample.source_id):
                raise BatchedTemporalError("prefix/unroll source differs from sample source")
            if tuple(packet.frames[0].camera_ids) != camera_order:
                raise BatchedTemporalError("camera order differs across batched samples")
            if packet.reset:
                if int(packet.endpoint) != 0:
                    raise BatchedTemporalError("reset packet must be endpoint zero")
                previous = 0
            elif not first and previous is not None:
                if int(packet.endpoint) != previous + stride:
                    raise BatchedTemporalError(
                        "prefix/unroll endpoint sequence skips or repeats a causal stride"
                    )
                previous = int(packet.endpoint)
            else:
                previous = int(packet.endpoint)
            first = False
        if str(sample.unroll_packets[-1].source_demo_id) != str(sample.source_id):
            raise BatchedTemporalError("sample last unroll source differs")


def _prefix_hidden(
    student: Any, samples: Sequence[Any]
) -> tuple[torch.Tensor | None, torch.Tensor]:
    max_length = max((len(sample.prefix_packets) for sample in samples), default=0)
    batch = len(samples)
    cameras = len(samples[0].unroll_packets[0].frames[0].camera_ids)
    num_patches = int(student.geometry.num_patches)
    hidden_size = int(student.geometry.hidden_size)
    hidden_by: list[torch.Tensor | None] = [None] * batch
    has_by = torch.zeros((batch,), dtype=torch.bool, device=student.gate_logit.device)
    with torch.no_grad():
        for step in range(max_length):
            active = [
                index for index, sample in enumerate(samples) if len(sample.prefix_packets) > step
            ]
            if not active:
                continue
            packets = [samples[index].prefix_packets[step] for index in active]
            existing = [hidden_by[index] for index in active]
            if any(item is not None for item in existing):
                dtype = next(item.dtype for item in existing if item is not None)
                if any(item is not None and item.dtype != dtype for item in existing):
                    raise BatchedTemporalError("prefix hidden dtypes differ across samples")
                previous = torch.zeros(
                    (len(active), cameras, num_patches, hidden_size),
                    dtype=dtype,
                    device=student.gate_logit.device,
                )
                has_previous = has_by[active].clone()
                for row, item in enumerate(existing):
                    if item is not None:
                        previous[row] = item
            else:
                previous = None
                has_previous = has_by[active].clone()
            next_hidden, _, _, _, _ = _batched_burst_step(student, packets, previous, has_previous)
            for row, index in enumerate(active):
                hidden_by[index] = next_hidden[row].detach()
            has_by[active] = True
    if not any(item is not None for item in hidden_by):
        return None, has_by
    dtype = next(item.dtype for item in hidden_by if item is not None)
    previous = torch.zeros(
        (batch, cameras, num_patches, hidden_size),
        dtype=dtype,
        device=student.gate_logit.device,
    )
    for index, item in enumerate(hidden_by):
        if item is not None:
            previous[index] = item
    return previous, has_by


def batched_anchor_forward(student: Any, samples: Sequence[Any]) -> BatchedForwardResult:
    """Run one rank-local sample batch through the six-endpoint temporal path."""
    samples = tuple(samples)
    if not samples:
        raise BatchedTemporalError("batched forward requires samples")
    if len(samples) > 8:
        raise BatchedTemporalError("active implementation supports at most 8 samples per rank")
    _validate_samples(student, samples)
    camera_order = tuple(samples[0].unroll_packets[0].frames[0].camera_ids)
    batch = len(samples)
    previous, has_previous = _prefix_hidden(student, samples)
    final_hidden = previous
    current_input: RawStemInput | None = None
    final_mask = None
    for step in range(BATCHED_HISTORY_LENGTH):
        packets = [sample.unroll_packets[step] for sample in samples]
        hidden, has_support, current_present, current_input, _ = _batched_burst_step(
            student, packets, previous, has_previous
        )
        previous = hidden
        has_previous = torch.ones_like(has_previous)
        final_hidden = hidden
        if step == BATCHED_HISTORY_LENGTH - 1:
            cameras = len(camera_order)
            num_patches = int(student.geometry.num_patches)
            hidden_size = int(student.geometry.hidden_size)
            flat_hidden = hidden.reshape(batch, cameras * num_patches, hidden_size)
            flat_hidden = torch.where(
                has_support.reshape(batch, cameras * num_patches, 1),
                flat_hidden,
                torch.zeros_like(flat_hidden),
            )
            output = _forward_frontend(
                student.base.vision_tower,
                student.base.projector,
                student.geometry,
                current_input,
                patch_tokens=flat_hidden,
            )
            final_mask = output.tokens.mask.reshape(batch, -1).to(dtype=torch.bool)
    if current_input is None or final_mask is None:
        raise BatchedTemporalError("batched temporal forward produced no final endpoint")
    return BatchedForwardResult(
        projected_tokens=output.projected_tokens.reshape(
            batch, -1, output.projected_tokens.shape[-1]
        ),
        token_mask=final_mask,
        support_fraction=output.support_fraction.reshape(batch, -1),
        hidden=final_hidden,
        source_ids=tuple(str(sample.source_id) for sample in samples),
        endpoints=tuple(int(sample.endpoint) for sample in samples),
    )
