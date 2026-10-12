"""N01 state: per-camera causal ring buffer of unique measurements.

The first reference implementation keeps the measurements and their validity,
so a later support-domain split can rebuild every sub-domain statistic from the
original samples instead of guessing how to break apart an already mixed mean.
Statistics are always recomputed from real measurements; feature updates never
write back into this buffer.

Layout: slot 0 is always the newest unique frame, so temporal depth ``D`` reads
slots ``0..min(D-1, filled-1)``.  A repeated frame (``new_measurement=False``)
advances the clock, updates the age of existing slots and its own measurement,
but never occupies a new unique slot, so it cannot be counted as independent
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import N01Config

__all__ = ["N01State", "PlaneSupport", "init_state", "push", "reset", "detach", "state_bytes"]


@dataclass
class PlaneSupport:
    """One-hop local support graph on a single CFA plane (or the single plane).

    ``edges`` is ``[B, P, 2, H, W]``: ``P`` is the CFA plane (1 when the
    measurement channels share one grid), channel 0 is the (left) neighbour at
    column offset -1 and channel 1 is the (up) neighbour at row offset -1.  Both
    neighbours are in the same CFA plane, so on a 2x2 mosaic they sit two mosaic
    columns/rows apart at their true sampling positions.
    """

    edges: Any
    seen: Any  # [B, P, 2, H, W] bool: the edge was measured at least once
    support_age: Any  # [B, P, 2, H, W] FP32: consecutive measured updates of the pair
    last_seen: Any  # [B, H, W] FP32: timestamp of the last valid measurement
    merge_count: Any  # [B] int64 cumulative
    split_count: Any  # [B] int64 cumulative


@dataclass
class N01State:
    """Causal per-camera measurement state for N01."""

    config: N01Config
    camera_ids: list[str]
    values: Any  # [N,B,C,H,W] FP32 calibrated measurement
    variance: Any  # [N,B,C,H,W] FP32 non-negative noise variance
    valid: Any  # [N,B,C,H,W] bool usable measurement (bad/saturated removed)
    observed: Any  # [N,B] bool: the slot holds a real unique measurement
    frame_ids: Any  # [N,B] int64, -1 when empty
    timestamps: Any  # [N,B] FP32 seconds
    age_steps: Any  # [N,B] int64 steps since that slot was written
    now: Any  # [B] FP32 current time in the frame's unit
    filled: Any  # [B] int64 unique measurements written (capped by max_history)
    step_index: Any  # [B] int64 number of pushed steps
    rng_index: Any  # [B] int64 write cursor (== filled % max_history)
    support: dict[str, PlaneSupport] = field(default_factory=dict)
    time_unit: str = "frame"
    shape: tuple[int, int] = (0, 0)

    # -- derived ------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        # The buffer layout is [slot, batch, channel, height, width]; the batch is
        # dimension 1 and must never be read from a derived 1-D tensor.
        return int(self.values.shape[1])

    @property
    def num_slots(self) -> int:
        return int(self.values.shape[0])

    @property
    def any_observed(self) -> Any:
        return self.observed.any(dim=0)

    def plane_support(self, camera_id: str) -> PlaneSupport:
        if camera_id not in self.support:
            raise KeyError(f"No support state for camera {camera_id!r}")
        return self.support[camera_id]

    def camera_index(self, camera_id: str) -> int:
        return self.camera_ids.index(camera_id)


def _zeros(reference, *shape):
    import torch

    return torch.zeros(shape, dtype=reference.dtype, device=reference.device)


def init_state(
    config: N01Config,
    batch_size: int,
    camera_spec: Any,
    device: Any = "cpu",
) -> N01State:
    """Create an empty state for ``batch_size`` independent episodes.

    ``camera_spec`` is either a camera id string, a list of camera ids, or a
    mapping with ``camera_ids`` / ``shape`` / ``channels`` keys.  Every camera
    keeps its own support graph so cross-camera information cannot leak through
    the measurement buffer.
    """
    import torch

    config.validate()
    if isinstance(camera_spec, str):
        camera_ids = [camera_spec]
        shape = None
    elif isinstance(camera_spec, dict):
        camera_ids = list(camera_spec.get("camera_ids", ["camera"]))
        shape = camera_spec.get("shape")
    else:
        camera_ids = list(camera_spec)
        shape = None
    if not camera_ids:
        raise ValueError("camera_spec must name at least one camera")
    if shape is None:
        raise ValueError("camera_spec must declare the measurement shape (H, W)")
    height, width = int(shape[0]), int(shape[1])
    need = 2 * int(max(config.spatial_radii)) + 1
    if height < need or width < need:
        raise ValueError(
            f"measurement shape {(height, width)} is smaller than the support window {need}"
        )

    slots = int(config.max_history)
    dtype = torch.float32
    state = N01State(
        config=config,
        camera_ids=camera_ids,
        values=torch.zeros(
            (slots, batch_size, config.channels, height, width), dtype=dtype, device=device
        ),
        variance=torch.ones(
            (slots, batch_size, config.channels, height, width), dtype=dtype, device=device
        ),
        valid=torch.zeros(
            (slots, batch_size, config.channels, height, width), dtype=torch.bool, device=device
        ),
        observed=torch.zeros((slots, batch_size), dtype=torch.bool, device=device),
        frame_ids=torch.full((slots, batch_size), -1, dtype=torch.int64, device=device),
        timestamps=torch.zeros((slots, batch_size), dtype=dtype, device=device),
        age_steps=torch.zeros((slots, batch_size), dtype=torch.int64, device=device),
        now=torch.zeros((batch_size,), dtype=dtype, device=device),
        filled=torch.zeros((batch_size,), dtype=torch.int64, device=device),
        step_index=torch.zeros((batch_size,), dtype=torch.int64, device=device),
        rng_index=torch.zeros((batch_size,), dtype=torch.int64, device=device),
        time_unit="frame",
        shape=(height, width),
    )
    for camera_id in camera_ids:
        state.support[camera_id] = PlaneSupport(
            edges=torch.ones(
                (batch_size, config.planes, 2, height, width), dtype=dtype, device=device
            ),
            seen=torch.zeros(
                (batch_size, config.planes, 2, height, width), dtype=torch.bool, device=device
            ),
            support_age=torch.zeros(
                (batch_size, config.planes, 2, height, width), dtype=dtype, device=device
            ),
            last_seen=torch.full(
                (batch_size, height, width), float("nan"), dtype=dtype, device=device
            ),
            merge_count=torch.zeros((batch_size,), dtype=torch.int64, device=device),
            split_count=torch.zeros((batch_size,), dtype=torch.int64, device=device),
        )
    return state


def _as_batch(value: Any, batch_size: int, device, dtype, name: str):
    import torch

    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    if tensor.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have leading dimension {batch_size}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


def push(state: N01State, frame: Any) -> N01State:
    """Advance the causal buffer by one frame for one camera.

    Returns a new state; ``state`` itself is treated as read-only.  The written
    slot keeps ``values/variance/valid/observed`` semantics so that repeated
    frames do not create extra independent observations.
    """
    import torch

    values = frame.values
    if values.ndim != 4:
        raise ValueError(f"MeasurementFrame.values must be [B,C,H,W], got {tuple(values.shape)}")
    batch_size, channels, height, width = values.shape
    config = state.config
    if channels != config.channels:
        raise ValueError(f"values has {channels} channels, config expects {config.channels}")
    if (height, width) != state.shape:
        raise ValueError(f"values shape {(height, width)} does not match state {state.shape}")
    if batch_size != state.batch_size:
        raise ValueError(f"frame batch {batch_size} does not match state batch {state.batch_size}")
    device = state.values.device

    # The measurement itself is carried with its graph: the encoder must be
    # trainable end to end from the raw measurement through the buffer.  Only
    # ``detach`` at a TBPTT boundary cuts it.
    new_values = values.to(device=device, dtype=torch.float32)
    valid = frame.valid_mask
    if valid is None:
        valid = torch.ones_like(new_values, dtype=torch.bool)
    else:
        valid = valid.to(device=device)
        if valid.dtype != torch.bool:
            valid = valid > 0.5
        valid = valid.expand(batch_size, channels, height, width)
    for mask in (frame.bad_mask, frame.saturated_mask):
        if mask is not None:
            mask = mask.to(device=device)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            valid = valid & ~mask.expand(batch_size, channels, height, width)

    variance = frame.metadata.get("variance")
    if variance is None:
        variance = config.noise_read + config.noise_shot * torch.clamp(new_values, min=0.0)
    variance = torch.as_tensor(variance, device=device, dtype=torch.float32)
    # Declared per-pixel variance may arrive as [C,H,W] (one frame), as [B,C,H,W]
    # (the documented MeasurementFrame shape, and what the real loaders emit) or as
    # a per-batch tensor that broadcasts to [B,C,H,W].  A mismatched batch
    # dimension is an error, never silently broadcast: silently reinterpreting
    # batch as slots would attach another episode's variance to this measurement.
    if variance.ndim == 5:
        if variance.shape[1] != 1 or variance.shape[3:] != (height, width):
            raise ValueError(f"variance shape {tuple(variance.shape)} is not a batch broadcast")
        variance = variance[:, 0]
    if variance.ndim == 3:
        variance = variance.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elif variance.ndim == 1:
        variance = variance.reshape(batch_size, 1, 1, 1).expand_as(new_values)
    if variance.shape != new_values.shape:
        if variance.shape[0] != batch_size:
            raise ValueError(
                f"variance batch {variance.shape[0]} does not match the frame batch {batch_size}"
            )
        variance = variance.expand_as(new_values)
    variance = torch.clamp(variance, min=1e-12)

    new_measurement = _as_batch(
        frame.new_measurement, batch_size, device, torch.bool, "new_measurement"
    )
    if new_measurement is None:
        new_measurement = torch.ones((batch_size,), dtype=torch.bool, device=device)

    frame_ids = _as_batch(frame.frame_ids, batch_size, device, torch.int64, "frame_ids")
    timestamps = _as_batch(frame.timestamp, batch_size, device, torch.float32, "timestamp")
    frame_time_unit = frame.time_unit or state.time_unit
    if frame_time_unit != state.time_unit:
        raise ValueError(f"time_unit changed from {state.time_unit!r} to {frame_time_unit!r}")
    if timestamps is None:
        timestamps = state.now + 1.0
    elif frame_time_unit == "frame":
        timestamps = torch.round(timestamps)

    history_values = state.values
    history_variance = state.variance
    history_valid = state.valid
    history_observed = state.observed
    history_frame_ids = state.frame_ids
    history_timestamps = state.timestamps

    # Shift every slot one step into the past, then write the newest measurement.
    write = new_measurement[None, :]
    next_values = torch.cat([new_values.unsqueeze(0), history_values[:-1]], dim=0)
    next_variance = torch.cat([variance.unsqueeze(0), history_variance[:-1]], dim=0)
    next_valid = torch.cat([valid.unsqueeze(0), history_valid[:-1]], dim=0)
    next_observed = torch.cat([write, history_observed[:-1]], dim=0)
    previous_frame_id = torch.where(
        history_observed[0], history_frame_ids[0], torch.full_like(history_frame_ids[0], -1)
    )
    incoming_frame_id = frame_ids if frame_ids is not None else state.step_index
    next_frame_ids = torch.cat(
        [
            torch.where(write, incoming_frame_id[None, :], previous_frame_id[None, :]),
            history_frame_ids[:-1],
        ],
        dim=0,
    )
    next_timestamps = torch.cat([timestamps.unsqueeze(0), history_timestamps[:-1]], dim=0)

    # Slot ages: 0 for a slot written by this step, previous age + 1 for a slot
    # shifted one step back, 0 for a slot that never held a measurement.  A
    # repeated frame therefore cannot refresh the age of an old observation.
    aged = torch.where(history_observed, state.age_steps + 1, torch.zeros_like(state.age_steps))
    next_age = torch.cat([torch.zeros_like(aged[:1]), aged[:-1]], dim=0)

    filled = torch.clamp(
        state.filled.reshape(-1) + write.reshape(-1).to(torch.int64), max=state.num_slots
    )

    return N01State(
        config=config,
        camera_ids=list(state.camera_ids),
        values=next_values,
        variance=next_variance,
        valid=next_valid,
        observed=next_observed,
        frame_ids=next_frame_ids,
        timestamps=next_timestamps,
        age_steps=next_age,
        now=timestamps,
        filled=filled,
        step_index=state.step_index + 1,
        rng_index=(state.step_index + 1) % state.num_slots,
        support={key: _copy_support(value) for key, value in state.support.items()},
        time_unit=state.time_unit,
        shape=state.shape,
    )


def _copy_support(support: PlaneSupport) -> PlaneSupport:
    return PlaneSupport(
        edges=support.edges,
        seen=support.seen,
        support_age=support.support_age,
        last_seen=support.last_seen,
        merge_count=support.merge_count,
        split_count=support.split_count,
    )


def measurement_age(state: N01State) -> Any:
    """Per-slot measurement age in the frame's time unit: ``now - t_slot``."""
    import torch

    observed = state.observed
    if state.time_unit == "frame":
        return torch.clamp(state.now[None, :] - state.timestamps, min=0.0)
    delta = (state.now[None, :] - state.timestamps).clamp(min=0.0)
    return torch.where(observed, delta, torch.full_like(delta, float("inf")))


def reset(state: N01State, batch_mask: Any) -> N01State:
    """Reset only the selected batch members; other episodes keep their state."""
    import torch

    mask = torch.as_tensor(batch_mask, device=state.values.device, dtype=torch.bool)
    if mask.ndim == 0:
        mask = mask.expand(state.batch_size)
    if mask.shape[0] != state.batch_size:
        raise ValueError(f"batch_mask must have length {state.batch_size}")
    keep = ~mask

    def wipe(tensor, fill=0.0):
        # Buffer tensors are [slot, batch, ...] while the per-camera support
        # tensors are [batch, ...].  The batch axis is the first axis whose size
        # equals the batch size; every other axis is broadcast.
        batch_dim = 0
        for axis, size in enumerate(tensor.shape):
            if size == keep.numel():
                batch_dim = axis
                break
        shape = [1] * tensor.ndim
        shape[batch_dim] = keep.numel()
        selector = keep.reshape(shape)
        for axis, size in enumerate(tensor.shape):
            if axis != batch_dim and size != 1:
                selector = selector.expand(
                    *[
                        size if index == axis else selector.shape[index]
                        for index in range(tensor.ndim)
                    ]
                )
        return torch.where(
            selector.expand(tuple(tensor.shape)), tensor, torch.full_like(tensor, fill)
        )

    values = wipe(state.values)
    variance = wipe(state.variance, 1.0)
    valid = wipe(state.valid, 0.0).to(torch.bool)
    observed = wipe(state.observed, 0.0).to(torch.bool)
    frame_ids = wipe(state.frame_ids, -1.0).to(torch.int64)
    timestamps = wipe(state.timestamps)
    age_steps = wipe(state.age_steps.to(torch.float32), 0.0).to(torch.int64)
    support = {}
    for key, plane in state.support.items():
        support[key] = PlaneSupport(
            edges=wipe(plane.edges, 1.0),
            seen=wipe(plane.seen, 0.0).to(torch.bool),
            support_age=wipe(plane.support_age),
            last_seen=wipe(plane.last_seen, float("nan")),
            merge_count=wipe(plane.merge_count.to(torch.float32), 0.0).to(torch.int64),
            split_count=wipe(plane.split_count.to(torch.float32), 0.0).to(torch.int64),
        )
    return N01State(
        config=state.config,
        camera_ids=list(state.camera_ids),
        values=values,
        variance=variance,
        valid=valid,
        observed=observed,
        frame_ids=frame_ids,
        timestamps=timestamps,
        age_steps=age_steps,
        now=wipe(state.now),
        filled=wipe(state.filled.to(torch.float32), 0.0).to(torch.int64),
        step_index=wipe(state.step_index.to(torch.float32), 0.0).to(torch.int64),
        rng_index=wipe(state.rng_index.to(torch.float32), 0.0).to(torch.int64),
        support=support,
        time_unit=state.time_unit,
        shape=state.shape,
    )


def detach(state: N01State) -> N01State:
    """Detach every tensor at a TBPTT boundary without changing the values."""

    def strip(tensor):
        return tensor.detach() if hasattr(tensor, "detach") else tensor

    return N01State(
        config=state.config,
        camera_ids=list(state.camera_ids),
        values=strip(state.values),
        variance=strip(state.variance),
        valid=strip(state.valid),
        observed=strip(state.observed),
        frame_ids=strip(state.frame_ids),
        timestamps=strip(state.timestamps),
        age_steps=strip(state.age_steps),
        now=strip(state.now),
        filled=strip(state.filled),
        step_index=strip(state.step_index),
        rng_index=strip(state.rng_index),
        support={
            key: PlaneSupport(
                edges=strip(plane.edges),
                seen=strip(plane.seen),
                support_age=strip(plane.support_age),
                last_seen=strip(plane.last_seen),
                merge_count=strip(plane.merge_count),
                split_count=strip(plane.split_count),
            )
            for key, plane in state.support.items()
        },
        time_unit=state.time_unit,
        shape=state.shape,
    )


def state_bytes(state: N01State) -> int:
    """Exact byte count of the persistent state (measured, not estimated)."""
    total = 0
    for name in (
        "values",
        "variance",
        "valid",
        "observed",
        "frame_ids",
        "timestamps",
        "age_steps",
        "now",
        "filled",
        "step_index",
        "rng_index",
    ):
        tensor = getattr(state, name)
        total += int(tensor.numel()) * int(tensor.element_size())
    seen: set[int] = set()
    for plane in state.support.values():
        for tensor in (
            plane.edges,
            plane.seen,
            plane.support_age,
            plane.last_seen,
            plane.merge_count,
            plane.split_count,
        ):
            pointer = tensor.data_ptr() if tensor.numel() else id(tensor)
            if pointer in seen:
                continue
            seen.add(pointer)
            total += int(tensor.numel()) * int(tensor.element_size())
    return total
