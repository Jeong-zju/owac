"""N01 v1.2 P3 runtime service: deadline readout on the complete model.

This is the online side of the v1.2 route.  It keeps the payload contract of the
accepted :mod:`owac.runtime.n01_feature_service` -- the pinned StarVLA server
dispatches ``predict_action(**payload)`` and expects ``normalized_actions`` back --
so the complete N01 model can replace the prototype without the platform clients
learning a new interface.

What the deadline semantics are, and what they are not
-----------------------------------------------------
The service guarantees an **engineering** property: it either returns a legal
snapshot, or it returns an explicit *unavailable* answer.  It never fabricates a
zero action to look successful, and it never returns an action computed from a
frame that did not arrive.

* A snapshot is published only after the source state update completes, and the
  publish is atomic: readers see the whole snapshot or the previous one.
* When the deadline passes before a *new* legal snapshot exists, the service may
  answer with the most recent legal snapshot **together with its source time**, or
  with a single-frame degradation when the caller allows it.
* When there is no legal input at all, the answer is ``unavailable`` with a reason.

It does **not** promise GPU hard real-time worst-case latency, and it does not
change the replanning rule or action-chunk interruption rule: changing those is a
separate control experiment and is not done here.

Stage latencies are recorded separately (queue, transfer, support, recompute,
flow, backend, commit) because the task requires the arrival rhythm, the backend
inference rhythm and the chunk execution rhythm to be reported apart rather than
collapsed into one number.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.frameworks.n01_full.model import N01V12Config, N01V12Model
from owac.runtime.n01_v12_monitor import RunStateWriter  # noqa: F401  (re-exported for callers)

__all__ = [
    "LATENCY_STAGES",
    "OPENPI_ADAPTER_INTERFACE",
    "OpenPIAdapterConfig",
    "N01V12OpenPIAdapter",
    "DeadlinePolicy",
    "LatencyLedger",
    "Snapshot",
    "N01V12Service",
    "main",
]

LATENCY_STAGES = ("queue", "transfer", "support", "recompute", "flow", "backend", "commit", "total")

UNAVAILABLE_NO_INPUT = "no_legal_snapshot"
UNAVAILABLE_NO_FRAME = "no_legal_frame"


@dataclass(frozen=True)
class DeadlinePolicy:
    """Turn-taking rules for the readout.

    ``deadline_ms=None`` means "no deadline": the service always computes a fresh
    snapshot and the stale path is never taken.  ``allow_single_frame_degradation``
    permits answering from the newest single frame when the window cannot be fused
    in time, which is a real degradation and is reported as such.
    """

    deadline_ms: float | None = None
    allow_stale_snapshot: bool = True
    allow_single_frame_degradation: bool = True
    max_snapshot_age_ms: float | None = None

    def __post_init__(self) -> None:
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive when set")
        if self.max_snapshot_age_ms is not None and self.max_snapshot_age_ms <= 0:
            raise ValueError("max_snapshot_age_ms must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadline_ms": self.deadline_ms,
            "allow_stale_snapshot": self.allow_stale_snapshot,
            "allow_single_frame_degradation": self.allow_single_frame_degradation,
            "max_snapshot_age_ms": self.max_snapshot_age_ms,
            "guarantee": (
                "engineering semantics only: a legal snapshot or an explicit unavailable "
                "answer; no GPU hard real-time bound is claimed"
            ),
        }


class LatencyLedger:
    """Per-stage timing with percentiles; no composite score is derived from it."""

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = int(capacity)
        self._samples: dict[str, list[float]] = {stage: [] for stage in LATENCY_STAGES}

    def record(self, stage: str, seconds: float) -> None:
        if stage not in self._samples:
            raise KeyError(f"unknown latency stage {stage!r}")
        values = self._samples[stage]
        values.append(float(seconds) * 1000.0)
        if len(values) > self.capacity:
            del values[: len(values) - self.capacity]

    def stage_summary(self, stage: str) -> dict[str, Any]:
        values = sorted(self._samples[stage])
        if not values:
            return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}

        def pick(fraction: float) -> float:
            index = min(int(round(fraction * (len(values) - 1))), len(values) - 1)
            return round(values[index], 3)

        return {
            "n": len(values),
            "p50_ms": pick(0.50),
            "p95_ms": pick(0.95),
            "p99_ms": pick(0.99),
            "max_ms": round(values[-1], 3),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "unit": "milliseconds, wall clock",
            "note": (
                "stages are reported separately on purpose: arrival rhythm, backend inference "
                "rhythm and chunk execution rhythm are different quantities.  No composite "
                "score is computed from them."
            ),
            "stages": {stage: self.stage_summary(stage) for stage in LATENCY_STAGES},
        }


@dataclass
class Snapshot:
    """One atomically published readout."""

    actions: np.ndarray
    kind: str  # fresh | stale | single_frame_degraded
    created_at: float
    source_time: float | None
    unique_frames: dict[str, Any]
    cost: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def describe(self, now: float, policy: DeadlinePolicy) -> dict[str, Any]:
        age_ms = (now - self.created_at) * 1000.0
        expired = policy.max_snapshot_age_ms is not None and age_ms > float(
            policy.max_snapshot_age_ms
        )
        return {
            "kind": self.kind,
            "age_ms": round(age_ms, 3),
            "source_time": self.source_time,
            "expired_by_age": bool(expired),
            "unique_frames": self.unique_frames,
        }


class N01V12Service:
    """The complete v1.2 model behind the accepted payload contract."""

    def __init__(
        self,
        model: N01V12Model,
        policy: Any,
        bridge: Any,
        *,
        device: Any = "cuda",
        deadline: DeadlinePolicy | None = None,
    ) -> None:
        self.model = model
        if getattr(model, "encoder", None) is not None:
            model.encoder.profile_stages = True
        self.policy = policy
        self.bridge = bridge
        self.device = torch.device(device)
        self.deadline = deadline or DeadlinePolicy()
        self.ledger = LatencyLedger()
        self.camera_order = list(model.config.cameras)
        self.states: dict[str, Any] = {}
        self.episodes: dict[str, list[Any]] = {}
        self.snapshots: dict[str, Snapshot] = {}
        self._lock = threading.Lock()
        self.counters = {
            "requests": 0,
            "fresh": 0,
            "stale": 0,
            "single_frame_degraded": 0,
            "deadline_missed": 0,
            "unavailable": 0,
            "reset": 0,
        }
        if int(policy.action_horizon) != int(bridge.protocol.action_horizon):
            raise ValueError("policy action_horizon does not match the selected suite protocol")
        if int(policy.action_dim) != int(bridge.protocol.model_action_dim):
            raise ValueError("policy action_dim does not match the selected suite protocol")

    # -- state -------------------------------------------------------------------

    def reset(self, batch_mask: Any = None) -> dict[str, Any]:
        with self._lock:
            self.counters["reset"] += 1
            if batch_mask is None:
                self.states.clear()
                self.episodes.clear()
                self.snapshots.clear()
            else:
                mask = torch.as_tensor(batch_mask, device=self.device, dtype=torch.bool)
                if bool(mask.any()):
                    self.snapshots.clear()
                for camera, state in list(self.states.items()):
                    self.states[camera] = self.model.reset(state, mask)
                    for index in mask.nonzero().flatten().tolist():
                        self.episodes[camera][index] = None
                        self.snapshots.pop(f"{camera}:{index}", None)
            self.bridge.action_cache.reset(batch_mask)
        return {"reset": True, "batch_mask": None if batch_mask is None else list(batch_mask)}

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "interface": "owac-n01-v1.2-deadline-readout",
                "arm": self.model.config.arm,
                "platform": self.model.config.platform,
                "cameras": self.camera_order,
                "support_enabled": bool(self.model.support is not None),
                "deadline": self.deadline.to_dict(),
                "counters": dict(self.counters),
                "latency": self.ledger.summary(),
                "legal_state": {
                    camera: len(self.episodes.get(camera, [])) for camera in self.camera_order
                },
            }

    # -- readout -----------------------------------------------------------------

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        """The pinned StarVLA server dispatches this signature."""

        if payload.pop("reset", False):
            self.reset(payload.pop("reset_batch_mask", None))
        if payload.pop("health", False):
            return self.health()
        return self.infer(payload, noise=payload.pop("noise", None))

    def _fresh_snapshot(self, data: dict[str, Any], noise: Any = None) -> Snapshot | None:
        # State and action-cache changes commit together only after all cameras and
        # the action readout succeed. A malformed second camera cannot half-reset one.
        old_states, old_episodes, old_snapshots = self.states, self.episodes, self.snapshots
        old_cache = self.bridge.action_cache
        self.states = dict(old_states)
        self.episodes = {camera: list(ids) for camera, ids in old_episodes.items()}
        self.snapshots = dict(old_snapshots)
        self.bridge.action_cache = copy.copy(old_cache)
        if hasattr(old_cache, "actions"):
            self.bridge.action_cache.actions = list(old_cache.actions)
            self.bridge.action_cache.chunk_start = list(old_cache.chunk_start)
        try:
            with torch.no_grad():
                return self._fresh_snapshot_impl(data, noise=noise)
        except BaseException:
            self.states, self.episodes, self.snapshots = old_states, old_episodes, old_snapshots
            self.bridge.action_cache = old_cache
            raise

    def _fresh_snapshot_impl(self, data: dict[str, Any], noise: Any = None) -> Snapshot | None:
        """Compute a new legal snapshot, or None when there is no legal input.

        This is the part that touches the model; it is a separate method so the
        deadline and fallback behaviour can be checked without a GPU.
        """

        from owac.backends.rawvla_features import ActionChunkCache
        from owac.core.encoder_contract import MeasurementFrame, VisualTokens
        from owac.training.n01_action import observation_from_batch

        frames = data.get("frames") or {}
        if not frames:
            return None

        tokens, counts, costs, camera_diagnostics = [], {}, {}, {}
        newest_source_time = None
        for camera, payload in frames.items():
            if camera not in self.camera_order:
                raise ValueError(f"unknown camera {camera!r}; expected {self.camera_order}")
            if payload.get("value_units") != "normalized_raw_measurement":
                raise ValueError("explicit calibrated RAW units are required")
            transfer_started = time.perf_counter()
            values = torch.as_tensor(payload["values"], device=self.device, dtype=torch.float32)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.ledger.record("transfer", time.perf_counter() - transfer_started)
            if values.ndim == 4:
                values = values[:, None]
            if values.ndim != 5:
                raise ValueError("frame values must be BCHW or chronological BTCHW")
            batch, length, _, height, width = values.shape
            episodes = list(payload["episode_ids"])
            if len(episodes) != batch:
                raise ValueError("one episode ID is required per batch member")
            if not self.states:
                self.bridge.action_cache = ActionChunkCache(
                    self.bridge.protocol.replan_steps, batch
                )
            if camera not in self.states:
                self.states[camera] = self.model.init_state(
                    batch, {"camera_ids": [camera], "shape": (height, width)}, self.device
                )
                self.episodes[camera] = episodes
                self.states[camera].time_unit = payload.get("time_unit", "frame")
            else:
                changed = torch.tensor(
                    [old != new for old, new in zip(self.episodes[camera], episodes, strict=True)],
                    device=self.device,
                )
                if bool(changed.any()):
                    self.states[camera] = self.model.reset(self.states[camera], changed)
                    self.bridge.action_cache.reset(changed)
                    # Never serve actions from the previous episode as a deadline fallback.
                    self.snapshots.pop("latest", None)
                    for index in changed.nonzero().flatten().tolist():
                        self.snapshots.pop(f"{camera}:{index}", None)
                self.episodes[camera] = episodes

            def optional(key: str, dtype: Any, payload: dict[str, Any] = payload) -> Any:
                value = payload.get(key)
                if value is None:
                    return None
                return torch.as_tensor(value, device=self.device, dtype=dtype)

            support_started = time.perf_counter()
            output = None
            ids_all = optional("frame_ids", torch.long)
            stamp_all = optional("timestamp", torch.float32)
            if ids_all is None:
                raise ValueError("real frame identities are required for every arrival")
            if ids_all.ndim == 1:
                ids_all = ids_all[:, None]
            if tuple(ids_all.shape) != (batch, length):
                raise ValueError("frame_ids must be B,T aligned with chronological observations")
            if stamp_all is not None and stamp_all.ndim == 1:
                stamp_all = stamp_all[:, None]

            def at_time(key, dtype, index, length=length):
                item = optional(key, dtype)
                if item is None:
                    return None
                if (
                    item.ndim >= 2
                    and item.shape[1] == length
                    and (item.ndim == 2 or item.ndim == 5)
                ):
                    return item[:, index]
                return item

            for index in range(length):
                current_state = self.states[camera]
                buffer = current_state.cameras[camera]
                frame_ids = ids_all[:, index]
                repeated = ((buffer.frame_ids == frame_ids[None]) & buffer.observed).any(0)
                # Old transport overlap may already have been evicted. It must not
                # backfill the retained window as if it were a newly arrived exposure.
                stale = buffer.any_observed & (frame_ids < buffer.frame_ids[0])
                if bool((repeated | stale).all()) and index + 1 < length:
                    continue
                new = at_time("new_measurement", torch.bool, index)
                if new is None:
                    new = torch.ones(batch, dtype=torch.bool, device=self.device)
                new = new & ~repeated & ~stale
                stamp = frame_ids.to(torch.float32) if stamp_all is None else stamp_all[:, index]
                stamp = torch.where(new, stamp, torch.maximum(stamp, buffer.now))
                from owac.core.encoder_contract import ComputeBudget

                output = self.model.step(
                    MeasurementFrame(
                        values[:, index],
                        valid_mask=at_time("valid_mask", torch.bool, index),
                        bad_mask=at_time("bad_mask", torch.bool, index),
                        saturated_mask=at_time("saturated_mask", torch.bool, index),
                        frame_ids=frame_ids,
                        episode_ids=episodes,
                        camera_id=camera,
                        timestamp=stamp,
                        time_unit=payload.get("time_unit", "frame"),
                        new_measurement=new,
                        metadata={"variance": at_time("variance", torch.float32, index)},
                    ),
                    state=current_state,
                    budget=ComputeBudget(
                        max_updates=int(index + 1 == length),
                        max_history=self.model.config.window_frames,
                    ),
                )
                self.states[camera] = output.state
            if output is None:
                raise ValueError("transport contains no usable readout event")
            self.states[camera] = output.state
            tokens.append(self.model.readout(output))
            # ``output.state`` is the v1.2 ``FullState``; the unique-measurement
            # buffer (and its ``filled`` counter) lives per camera underneath it.
            counts[camera] = int(output.state.cameras[camera].filled.max())
            costs[camera] = output.cost
            camera_diagnostics[camera] = self._last_diagnostics()
            if "motion_seconds" in output.cost:
                self.ledger.record("flow", output.cost["motion_seconds"])
            stamp = payload.get("timestamp")
            if stamp is not None:
                candidate = float(np.max(np.asarray(stamp, dtype=np.float64)))
                newest_source_time = (
                    candidate if newest_source_time is None else max(newest_source_time, candidate)
                )
            if output.cost.get("stage_timing_kind") == "synchronized_wall":
                self.ledger.record("support", output.cost["support_seconds"])
                self.ledger.record("recompute", output.cost["content_seconds"])
            else:
                self.ledger.record("recompute", time.perf_counter() - support_started)

        if not tokens:
            return None
        visual = VisualTokens(
            torch.cat([entry.values for entry in tokens], 1),
            torch.cat([entry.mask for entry in tokens], 1),
            [camera for entry in tokens for camera in entry.camera_ids],
            torch.cat([entry.positions for entry in tokens], 1),
        )
        if not bool(visual.mask.any()):
            return None
        language = data.get("language")
        if isinstance(language, str):
            language = [language] * visual.values.shape[0]
        if not language:
            raise ValueError("task language is required; do not fabricate a prompt")
        observation = observation_from_batch(
            self.policy,
            {
                "state": torch.as_tensor(np.asarray(data["normalized_state"])),
                "meta": [{"task_description": text} for text in language],
            },
            self.device,
        )
        flow_started = time.perf_counter()
        if noise is not None:
            flow_noise = torch.as_tensor(np.asarray(noise), device=self.device, dtype=torch.float32)
            if flow_noise.ndim == 2:
                flow_noise = flow_noise[None]
        else:
            flow_noise = None
        actions = self.bridge.sample_actions_with_features(
            observation, visual, noise=flow_noise, num_steps=10
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.ledger.record("backend", time.perf_counter() - flow_started)
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("non-finite normalized action chunk")
        commit_started = time.perf_counter()
        committed_actions = actions.detach().cpu().numpy()
        self.ledger.record("commit", time.perf_counter() - commit_started)
        return Snapshot(
            actions=committed_actions,
            kind="fresh",
            created_at=time.perf_counter(),
            source_time=newest_source_time,
            unique_frames=counts,
            cost=costs,
            diagnostics={**self._last_diagnostics(), "cameras": camera_diagnostics},
        )

    def _last_diagnostics(self) -> dict[str, Any]:
        """Compact state of the newest inference, for the synchronized timeline.

        The support decision and the fusion weights are computed on the way to the action
        and are otherwise discarded, so a timeline figure could only reconstruct them.
        Recording them here means the figure can show what the model actually saw rather
        than a replay that may differ from the weights that produced the action.
        """

        # These live on the DBSR feature adapter, which the model exposes as ``encoder``;
        # reading them off the top-level model silently yielded an empty block on every
        # reply, because getattr with a default cannot tell "missing" from "absent here".
        encoder = getattr(self.model, "encoder", None) or self.model
        support = getattr(encoder, "last_support", None)
        fusion = getattr(encoder, "last_fusion", None)
        summary: dict[str, Any] = {}
        if support is not None:
            usable = getattr(support, "source_usable", None)
            if usable is None:
                summary["support"] = {"usable_cells": None, "total_cells": None}
            else:
                mask = usable.detach().to("cpu").to(torch.bool)
                # The mask is [T-1, H, W]; at 31x128x128 that is ~508k entries and it was
                # shipped whole on every reply while the far smaller fusion weights were
                # already cut to a bounded sample.  The counts carry the meaning and a
                # strided sample keeps a map available for the figures.
                summary["support"] = {
                    "usable_cells": int(mask.sum()),
                    "total_cells": int(mask.numel()),
                    "source_usable_sample": _small(mask.to(torch.float32), limit=4096),
                }
            summary["support"]["change_logodds"] = _small(getattr(support, "change_logodds", None))
            summary["support"]["region_residual"] = _small(
                getattr(support, "region_residual", None)
            )
        if fusion is not None:
            summary["fusion"] = {
                "camera_id": fusion.get("camera_id"),
                "frame_ids": fusion.get("frame_ids"),
                "weights": _small(fusion.get("weights")),
                "pull_vectors": N01V12Service._pull_vectors(fusion),
            }
        return summary

    @staticmethod
    def _pull_vectors(fusion):
        offsets = fusion.get("offsets_xy")
        if offsets is None:
            return None
        height, width = offsets.shape[-2:]
        ys = torch.linspace(0, height - 1, 4, device=offsets.device).long()
        xs = torch.linspace(0, width - 1, 4, device=offsets.device).long()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        query = torch.stack([xx.flatten(), yy.flatten()], -1)
        values = offsets[0, :, :, yy.flatten(), xx.flatten()].permute(0, 2, 1)
        return {
            "query_xy": query.cpu().tolist(),
            "pull_xy": (values + query[None]).cpu().tolist(),
            "frame_ids": fusion.get("frame_ids"),
            "grid_shape": [height, width],
            "unit": "current model-grid pixels; proxy correspondence, not GT",
        }

    def _degrade_to_single_frame(
        self, data: dict[str, Any], key: str, noise: Any = None
    ) -> Snapshot | None:
        """Answer from the newest single legal frame when the window cannot be fused.

        The result is labelled ``single_frame_degraded``; the previous buffer is not
        destroyed, so the next on-time request can still fuse the full window.
        """

        frames = data.get("frames") or {}
        if not frames:
            return None
        saved = {camera: self.states.get(camera) for camera in self.camera_order}
        saved_episodes = {camera: self.episodes.get(camera) for camera in self.camera_order}
        saved_cache = self.bridge.action_cache
        saved_snapshots = dict(self.snapshots)
        # A burst fallback must actually use only the newest arrival, not replay T frames.
        single_frames = {}
        for camera, payload in frames.items():
            item = dict(payload)
            values = np.asarray(payload["values"])
            if values.ndim == 5:
                length = values.shape[1]
                item["values"] = values[:, -1]
                for field in (
                    "frame_ids",
                    "timestamp",
                    "new_measurement",
                    "valid_mask",
                    "bad_mask",
                    "saturated_mask",
                    "variance",
                ):
                    if payload.get(field) is not None:
                        value = np.asarray(payload[field])
                        if value.ndim >= 2 and value.shape[1] == length and value.ndim in (2, 5):
                            item[field] = value[:, -1]
            single_frames[camera] = item
        try:
            for camera in list(frames):
                self.states.pop(camera, None)
            # Dropping the per-camera state above already makes this a single-frame
            # readout: the buffer is rebuilt and holds only the frame just delivered, so
            # the fusion has no history.  The previous version additionally sliced
            # ``[:, -1:]``, which indexes the CHANNEL axis of a BCHW payload and would
            # have produced a one-channel tensor the adapter rejects.
            snapshot = self._fresh_snapshot({**data, "frames": single_frames}, noise=noise)
        finally:
            self.bridge.action_cache = saved_cache
            self.snapshots = saved_snapshots
            for camera in list(frames):
                if saved.get(camera) is not None:
                    self.states[camera] = saved[camera]
                if saved_episodes.get(camera) is not None:
                    self.episodes[camera] = saved_episodes[camera]
        if snapshot is None:
            return None
        snapshot.kind = "single_frame_degraded"
        del key
        return snapshot

    def infer(self, data: dict[str, Any], noise: Any = None) -> dict[str, Any]:
        request_started = time.perf_counter()
        with self._lock:
            self.ledger.record("queue", time.perf_counter() - request_started)
            self.counters["requests"] += 1
            deadline = self.deadline
            try:
                snapshot = self._fresh_snapshot(data, noise=noise)
            except Exception:
                # A failed computation must not be papered over with a stale answer
                # unless the caller explicitly allowed stale output.
                raise
            now = time.perf_counter()

            previous = self.snapshots.get("latest")
            elapsed_ms = (now - request_started) * 1000.0
            budget_ms = deadline.deadline_ms
            missed = budget_ms is not None and elapsed_ms > float(budget_ms)

            if snapshot is not None:
                if missed:
                    self.counters["deadline_missed"] += 1
                    # A late fresh answer is only preferred when no timely alternative is
                    # both permitted and still within its own age bound.
                    usable_stale = (
                        previous is not None
                        and deadline.allow_stale_snapshot
                        and not self._expired(previous, now)
                    )
                    if usable_stale:
                        previous.kind = "stale"
                        self.counters["stale"] += 1
                        return self._response(
                            previous,
                            now,
                            "stale",
                            request_started,
                            deadline_met=False,
                            note=f"fresh computation took {elapsed_ms:.1f} ms > {budget_ms} ms",
                        )
                    if deadline.allow_single_frame_degradation:
                        degraded = self._degrade_to_single_frame(data, "deadline", noise=noise)
                        if degraded is not None:
                            self.counters["single_frame_degraded"] += 1
                            return self._response(
                                degraded,
                                now,
                                "single_frame_degraded",
                                request_started,
                                deadline_met=False,
                                note=f"fresh computation took {elapsed_ms:.1f} ms",
                            )
                    # Nothing timely exists: say so rather than implying the deadline held.
                    self.snapshots["latest"] = snapshot
                    self.counters["fresh"] += 1
                    return self._response(
                        snapshot,
                        now,
                        "fresh",
                        request_started,
                        deadline_met=False,
                        note=f"no timely fallback available; returned a late fresh answer "
                        f"({elapsed_ms:.1f} ms > {budget_ms} ms)",
                    )
                self.snapshots["latest"] = snapshot
                self.counters["fresh"] += 1
                return self._response(snapshot, now, "fresh", request_started, deadline_met=True)

            if (
                previous is not None
                and deadline.allow_stale_snapshot
                and not self._expired(previous, now)
            ):
                previous.kind = "stale"
                self.counters["stale"] += 1
                return self._response(
                    previous, now, "stale", request_started, deadline_met=not missed
                )

            self.counters["unavailable"] += 1
            self.ledger.record("total", time.perf_counter() - request_started)
            return {
                "unavailable": UNAVAILABLE_NO_FRAME
                if not (data.get("frames") or {})
                else UNAVAILABLE_NO_INPUT,
                "reason": (
                    "no legal frame arrived for this request and no previous legal snapshot "
                    "exists; returning an explicit unavailable answer instead of a fabricated "
                    "zero action"
                ),
                "legal_snapshot": None,
                "latency_ms": {"total": round((now - request_started) * 1000.0, 3)},
                "deadline": deadline.to_dict(),
            }

    def _expired(self, snapshot: Snapshot, now: float) -> bool:
        """Whether a snapshot has outlived the policy's age bound.

        The bound used to be reported as ``expired_by_age`` and then ignored, so an
        arbitrarily old snapshot could still be served under a policy that declared a
        maximum age.
        """

        bound = self.deadline.max_snapshot_age_ms
        if bound is None:
            return False
        return (now - snapshot.created_at) * 1000.0 > float(bound)

    def _response(
        self,
        snapshot: Snapshot,
        now: float,
        kind: str,
        request_started: float,
        *,
        deadline_met: bool | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        now = time.perf_counter()
        self.ledger.record("total", now - request_started)
        return {
            "normalized_actions": snapshot.actions,
            "snapshot": snapshot.describe(now, self.deadline),
            "unique_frames": snapshot.unique_frames,
            "cost": snapshot.cost,
            "diagnostics": snapshot.diagnostics,
            "kind": kind,
            "latency_ms": {"total": round((now - request_started) * 1000.0, 3)},
            "deadline": self.deadline.to_dict(),
            "deadline_met": deadline_met,
            "deadline_elapsed_ms": round((now - request_started) * 1000.0, 3),
            "note": note,
            "guarantee": (
                "the deadline is checked against the measured elapsed time after the "
                "computation, not enforced by preemption: a fresh answer within the budget "
                "is returned as fresh; past the budget the service prefers a permitted, "
                "non-expired stale snapshot, then a labelled single-frame degradation, and "
                "otherwise says the deadline was missed rather than implying it held"
            ),
        }


def _small(value: Any, limit: int = 4096) -> Any:
    """Downsample a tensor to a JSON-safe list, preserving the coarse structure.

    A full 31x256 weight matrix would bloat every reply, and the figure only needs the
    shape of the distribution, so the values are flattened and strided to a bounded list.
    """

    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().to("cpu").to(torch.float32).reshape(-1)
    total = int(tensor.numel())
    if total == 0:
        return []
    if total <= limit:
        return [round(float(v), 6) for v in tensor]
    step = max(total // limit, 1)
    return [round(float(v), 6) for v in tensor[::step][:limit]]


def _parse_deadline(args: argparse.Namespace) -> DeadlinePolicy:
    return DeadlinePolicy(
        deadline_ms=args.deadline_ms,
        allow_stale_snapshot=not args.no_stale,
        allow_single_frame_degradation=not args.no_degrade,
        max_snapshot_age_ms=args.max_snapshot_age_ms,
    )


OPENPI_ADAPTER_INTERFACE = "owac-n01-v1.2-openpi-adapter"


@dataclass(frozen=True)
class OpenPIAdapterConfig:
    """How to read the author openpi payload.

    ``cameras`` must be the client's image order; it is required rather than
    guessed, because feeding the wrist view as the agentview view would silently
    lower the success rate with no error.  ``image_source`` selects which field of
    the example carries the measurement: ``raw_burst`` takes the newest element of
    ``raw_burst_float32`` (the rawvla_bench RAW path, no resize applied on the
    server), ``image`` takes ``example["image"]`` (which the client has already
    resized to ``--resize-size``; recorded as a limitation, not hidden).
    """

    cameras: tuple[str, ...]
    image_source: str = "raw_burst"
    raw_key: str = "raw_burst_float32"
    value_units: str = "normalized_raw_measurement"

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("cameras must list the client image order")
        if self.image_source not in {"raw_burst", "image"}:
            raise ValueError(f"unknown image_source {self.image_source!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameras": list(self.cameras),
            "image_source": self.image_source,
            "raw_key": self.raw_key,
            "value_units": self.value_units,
        }


class N01V12OpenPIAdapter:
    """Serve the complete v1.2 model to the author's openpi-style LIBERO client.

    The author client (``eval_starvla_openpi_client.predict_env_actions``) sends
    ``{"examples": [{"image","lang","state"}], "noise": ...}`` and reads
    ``response["data"]["normalized_actions"][0]``, raising when ``response["ok"]``
    is false.  The author server calls ``predict_action(**payload)``, so this class
    exposes that signature and always answers with an ``ok`` envelope.

    ``unavailable`` answers are still returned as ``ok: False`` with a reason: the
    client must be told that no legal snapshot existed rather than being handed a
    fabricated zero action.
    """

    def __init__(self, service: N01V12Service, config: OpenPIAdapterConfig) -> None:
        self.service = service
        self.config = config
        if tuple(service.camera_order) != tuple(config.cameras):
            raise ValueError(
                f"adapter camera order {config.cameras} does not match the arm's "
                f"{tuple(service.camera_order)}; refusing to guess the mapping"
            )
        self._frame_index = 0
        self._episodes: dict[str, str] = {}

    # -- payload mapping ----------------------------------------------------------

    def _to_bchw(self, array: Any, camera: str) -> Any:
        value = np.asarray(array)
        if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
            value = value.transpose(2, 0, 1)
        if value.ndim == 3:
            value = value[None]
        if value.ndim != 4:
            raise ValueError(f"camera {camera}: cannot read a [{value.shape}] measurement")
        if value.dtype == np.uint8:
            value = value.astype(np.float32) / 255.0
        return np.ascontiguousarray(value, dtype=np.float32)

    @staticmethod
    def _newest_frame(entry: Any) -> Any:
        """Take the newest single frame out of a history stack.

        The author client sends channel-last stacks: one entry per camera shaped
        ``[rnn, burst, H, W, C]``, where ``C`` is 3.  Treating such a stack as BCHW
        would read the height as the channel count -- which is exactly the failure
        this method removes.  Reducing the leading axes with ``[-1]`` also works for
        a channel-first ``[T, C, H, W]`` stack, because it lands on ``[C, H, W]``.
        """

        value = np.asarray(entry)
        while value.ndim > 3:
            value = value[-1]
        return value

    def _measurement(self, example: dict[str, Any], camera: str, index: int) -> Any:
        if self.config.image_source == "raw_burst":
            burst = example.get(self.config.raw_key)
            if burst is None:
                raise ValueError(
                    f"example has no {self.config.raw_key!r}; the client must run with "
                    "--image-mode rawvla_bench so the RAW burst is present"
                )
            return self._to_bchw(self._newest_frame(burst[index]), camera)
        images = example.get("image")
        if images is None:
            raise ValueError("example has no 'image' field")
        return self._to_bchw(np.asarray(images[index]), camera)

    def to_query(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Map one openpi ``examples`` entry onto the v1.2 ``infer`` payload."""

        if len(examples) != 1:
            raise ValueError(f"the adapter serves batch 1, got {len(examples)} examples")
        example = examples[0]
        language = example.get("lang")
        if isinstance(language, (list, tuple)):
            language = list(language)
        if not language:
            raise ValueError("task language is required; do not fabricate a prompt")
        frames = {}
        bound_model = getattr(self.service, "model", None)
        strict = (
            bound_model is not None and getattr(bound_model.config, "observation", None) is not None
        )
        for index, camera in enumerate(self.config.cameras):
            if strict:
                if self.config.image_source != "raw_burst":
                    raise ValueError("the complete model requires native RAW transport")
                burst = example.get(self.config.raw_key)
                identities = example.get("raw_frame_ids")
                if burst is None or identities is None or not example.get("episode_id"):
                    raise ValueError("native burst, real raw_frame_ids and episode_id are required")
                array = np.asarray(burst[index])
                if array.ndim < 3 or array.shape[-1] != 3:
                    raise ValueError(
                        "author native burst must be chronological channel-last frames"
                    )
                array = array.reshape(-1, *array.shape[-3:])
                ids = np.asarray(identities[index], dtype=np.int64).reshape(-1)
                if len(ids) != len(array) or (np.diff(ids) < 0).any():
                    raise ValueError("raw frame identities must align with chronological burst")
                if array.dtype == np.uint8:
                    array = array.astype(np.float32) / 255.0
                frames[camera] = {
                    "values": np.ascontiguousarray(
                        array.transpose(0, 3, 1, 2)[None], dtype=np.float32
                    ),
                    "value_units": self.config.value_units,
                    "episode_ids": [str(example["episode_id"])],
                    "frame_ids": ids[None],
                    "timestamp": ids[None].astype(np.float32),
                    "time_unit": "frame",
                    "new_measurement": np.ones((1, len(ids)), dtype=bool),
                }
            else:
                values = self._measurement(example, camera, index)
                batch = int(values.shape[0])
                episode = self._episodes.get(camera, "openpi")
                frames[camera] = {
                    "values": values,
                    "value_units": self.config.value_units,
                    "episode_ids": [episode] * batch,
                    "frame_ids": np.full((batch,), self._frame_index, dtype=np.int64),
                    "timestamp": np.full((batch,), float(self._frame_index), dtype=np.float32),
                    "time_unit": "frame",
                    "new_measurement": np.ones((batch,), dtype=bool),
                }
        state = example.get("state")
        if state is None:
            raise ValueError("normalized state is required")
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None]
        return {
            "frames": frames,
            "normalized_state": state,
            "language": language if isinstance(language, list) else [language],
        }

    # -- the server's entry points ------------------------------------------------

    def predict_action(self, **payload: Any) -> dict[str, Any]:
        """Return the reply *data*; raise on failure.

        The author ``WebsocketPolicyServer`` already wraps whatever this returns as
        ``{"status","ok","type","data"}`` and converts a raised exception into
        ``status="error"`` carrying the message.  Returning our own ``{"ok","data"}``
        envelope here nests it, and the client reads
        ``response["data"]["normalized_actions"]`` -- so a successful call would look
        like a missing key.  Success therefore returns the data dict directly and
        failure raises.
        """

        if payload.pop("reset", False):
            self.reset()
        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("payload must carry a non-empty 'examples' list")
        query = self.to_query(examples)
        noise = payload.get("noise")
        result = self.service.infer(query, noise=noise)
        self._frame_index += 1
        if "unavailable" in result:
            raise RuntimeError(f"no legal snapshot: {result.get('reason', result['unavailable'])}")
        actions = np.asarray(result["normalized_actions"], dtype=np.float32)
        # The client reads normalized_actions[0] as [horizon, action_dim].
        # Pass the whole envelope through rather than re-packing a subset: an earlier
        # version listed four keys by hand, which silently dropped every field added to
        # _response afterwards -- the synchronized-timeline diagnostics were lost that way
        # while the code that produced them looked correct in isolation.
        return {
            "normalized_actions": actions,
            **{key: value for key, value in result.items() if key != "normalized_actions"},
        }

    def reset(self, batch_mask: Any = None) -> dict[str, Any]:
        self._frame_index = 0
        self._episodes = {}
        return self.service.reset(batch_mask)

    def set_episode(self, episode: str, batch_mask: Any = None) -> None:
        """Mark a new episode; the client calls reset() itself, this is explicit."""

        self._episodes = {camera: episode for camera in self.config.cameras}
        self.service.reset(batch_mask)

    def health(self) -> dict[str, Any]:
        health = self.service.health()
        health["interface"] = OPENPI_ADAPTER_INTERFACE
        health["adapter"] = self.config.to_dict()
        return health


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Drive the running service with the author's own websocket client.

    Frames are sent one arrival at a time on purpose: that is the online rhythm the
    service was built for, so the measurement covers buffer push, unique-frame
    dedup, support revision and the backend, not a single batched forward pass.
    """

    import numpy as np
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    from owac.raw.n01_full_manifest import dataset_from_protocol
    from owac.raw.rawvla_sequence import collate_sequence_samples

    repo_root = Path(args.repo_root)
    protocol_path = (
        Path(args.protocol)
        if args.protocol
        else repo_root / "configs/encoders/n01_full/protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = N01V12Config.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    platform = config.platform
    dataset = dataset_from_protocol(
        protocol, platform, args.region, repo_root=str(repo_root), window=config.window_frames
    )
    sample = dataset[int(args.index)]
    batch = collate_sequence_samples([sample])
    language = batch["meta"][0]["task_description"]
    state = np.asarray(sample.state, dtype=np.float32)
    if state.ndim == 1:
        state = state[None]

    client = WebsocketClientPolicy(host=args.host, port=int(args.port))
    metadata = client.get_server_metadata()
    arrivals = []
    responses = []
    windows = {camera: batch["windows"][camera] for camera in sorted(batch["windows"])}
    total_frames = min(int(args.frames), int(next(iter(windows.values())).values.shape[1]))
    for index in range(total_frames):
        payload_frames = {}
        for camera, window in windows.items():
            payload_frames[camera] = {
                # One arrival is a single frame as BCHW; the buffer accumulates
                # across calls, which is the online rhythm this service serves.
                "values": np.asarray(window.values[:, index], dtype=np.float32),
                "value_units": "normalized_raw_measurement",
                "episode_ids": [str(batch["meta"][0].get("episode_key", "smoke"))],
                "frame_ids": np.asarray(window.frame_ids[:, index], dtype=np.int64),
                "timestamp": np.asarray(window.timestamp[:, index], dtype=np.float32),
                "time_unit": window.time_unit,
                "new_measurement": np.asarray(window.new_measurement[:, index], dtype=bool),
            }
        if args.payload == "openpi":
            # Mirror the author client: a RAW burst history per camera plus the
            # client's own flow noise, wrapped in an "examples" list.
            def burst(camera: str, upto: int) -> Any:
                window = windows[camera]
                start = max(0, upto - int(args.burst) + 1)
                stack = np.asarray(window.values[:, start : upto + 1], dtype=np.float32)
                return stack[0]

            payload_frames = {}
            query = {
                "examples": [
                    {
                        "raw_burst_float32": [burst(camera, index) for camera in sorted(windows)],
                        "lang": language,
                        "state": state,
                    }
                ],
                "noise": np.zeros((int(args.noise_horizon), int(args.noise_dim)), dtype=np.float32),
            }
        else:
            query = {
                "frames": payload_frames,
                "normalized_state": state,
                "language": [language],
            }
        started = time.perf_counter()
        envelope = client.predict_action(query)
        arrivals.append((time.perf_counter() - started) * 1000.0)
        # The author server wraps every reply as
        # {"status","ok","type","request_id","data"|"error"}, and reports a policy
        # exception as status="error" rather than raising at the client.  Reading the
        # top level would therefore both miss the payload and silently swallow a real
        # inference failure.
        if envelope.get("status") != "ok":
            raise RuntimeError(
                "inference failed through the websocket server: "
                f"{json.dumps(envelope.get('error'), ensure_ascii=False)}"
            )
        response = envelope.get("data") or {}
        # The openpi adapter answers with its own {"ok","data"} envelope, which the
        # author server then wraps again; unwrap it so the reported fields are the
        # real ones and an adapter refusal is not mistaken for a normal reply.
        if "ok" in response:
            if not response.get("ok"):
                raise RuntimeError(
                    "adapter refused the request: "
                    f"{json.dumps(response.get('error'), ensure_ascii=False)}"
                )
            response = response.get("data") or {}
        snapshot = response.get("snapshot") or {}
        responses.append(
            {
                # The adapter reports the kind inside the snapshot; the raw service
                # also mirrors it at the top level.
                "kind": response.get("kind") or snapshot.get("kind"),
                "unavailable": response.get("unavailable"),
                "snapshot": response.get("snapshot"),
                "unique_frames": response.get("unique_frames"),
                "actions_shape": list(np.asarray(response["normalized_actions"]).shape)
                if "normalized_actions" in response
                else None,
            }
        )
    kinds: dict[str, int] = {}
    for entry in responses:
        key = entry["kind"] or entry["unavailable"] or "unknown"
        kinds[key] = kinds.get(key, 0) + 1
    ordered = sorted(arrivals)

    def pick(fraction: float) -> float:
        return round(ordered[min(int(fraction * (len(ordered) - 1)), len(ordered) - 1)], 3)

    return {
        "stage": "P3",
        "command": "smoke",
        "arm": config.arm,
        "platform": platform,
        "server_metadata": metadata,
        "client": "deployment.model_server.tools.websocket_policy_client.WebsocketClientPolicy",
        "arrivals": total_frames,
        "arrival_kinds": kinds,
        "end_to_end_ms": {
            "p50": pick(0.50),
            "p95": pick(0.95),
            "p99": pick(0.99),
            "max": round(ordered[-1], 3),
        },
        "last_response": responses[-1] if responses else None,
        "note": (
            "this is the request/response latency of one arrival through the real "
            "websocket path, including buffer push, dedup, support and the backend; it "
            "is not a hard real-time bound and does not cover rendering"
        ),
    }


def _load_arm(weights: Path, tokenizer: Path, config_path: Path, suite: str, device: str):
    """Build one arm: the frozen π0.5 backend plus this arm's encoder and bridge."""

    from owac.backends.rawvla_features import ACTION_PROTOCOLS, RawVLAFeaturePolicy
    from owac.training.n01_action import load_backend

    config = N01V12Config.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))
    model = N01V12Model(config)
    policy, audit = load_backend(suite, weights, tokenizer, device)
    bridge = RawVLAFeaturePolicy(
        policy,
        encoder=model.encoder,
        postprocessor=model.bridge,
        action_protocol=ACTION_PROTOCOLS[suite],
    )
    bridge.freeze_backend_parameters()
    return model, policy, bridge, audit


def load_arm_checkpoint(
    model: N01V12Model, checkpoint_path: Path, device: Any, policy: Any = None
) -> dict[str, Any]:
    """Restore a trained arm produced by ``owac.training.n01_v12 train``.

    The training receipt stores the whole ``N01V12Model`` state dict, so loading it
    also proves the serving path is the arm that was trained -- not a prototype with
    the same name.
    """

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"{checkpoint_path} has no 'model' entry; it is not a v1.2 arm checkpoint")
    recorded = payload.get("config", {})
    if recorded.get("arm") and recorded["arm"] != model.config.arm:
        raise ValueError(
            f"checkpoint arm {recorded['arm']!r} does not match the serving arm "
            f"{model.config.arm!r}"
        )
    if int(recorded.get("window_frames", model.config.window_frames)) != int(
        model.config.window_frames
    ):
        raise ValueError("checkpoint window_frames does not match the serving configuration")
    if recorded and N01V12Config.from_dict(recorded).to_dict() != model.config.to_dict():
        raise ValueError("checkpoint effective model/observation/support configuration differs")
    backend = payload.get("backend_adaptation", {})
    if backend:
        if policy is None or model.config.training is None:
            raise ValueError(
                "adapted backend checkpoint requires its declared policy and training scope"
            )
        from owac.frameworks.n01_full.training_plan import BackendSchedule

        schedule = BackendSchedule(model, policy, model.config.training)
        schedule.restore(backend)
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
    elif (
        model.config.training is not None
        and payload.get("step", 0) >= model.config.training.warmup_steps
    ):
        raise ValueError("adaptation-phase checkpoint is missing backend state")
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return {
        "step": int(payload.get("step", -1)),
        "arm": recorded.get("arm"),
        "platform": recorded.get("platform"),
        "seed": recorded.get("seed"),
    }


def _serve(argv: list[str] | None = None) -> int:
    """Serve the payload contract over the author's websocket server.

    The author's ``server_policy.py`` hard-codes its own wrapper and offers no way
    to inject a different policy.  Rather than editing frozen upstream code, this
    entry point imports the author's :class:`WebsocketPolicyServer` and passes this
    service as its policy -- the same pattern the accepted
    :mod:`owac.runtime.n01_feature_service` already uses, and it leaves the author
    tree untouched.
    """

    parser = argparse.ArgumentParser(prog="owac.runtime.n01_v12_service serve")
    parser.add_argument("--suite", choices=("libero", "robotwin2"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--flow-batch-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=18020)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--deadline-ms", type=float, default=None)
    parser.add_argument("--max-snapshot-age-ms", type=float, default=None)
    parser.add_argument("--no-stale", action="store_true")
    parser.add_argument("--no-degrade", action="store_true")
    parser.add_argument("--health", action="store_true", help="print the health record and exit")
    parser.add_argument(
        "--openpi-adapter",
        action="store_true",
        help="accept the author client payload shape (examples + noise) instead of frames",
    )
    parser.add_argument(
        "--raw-frontend-burst-frames",
        type=int,
        default=1,
        help=(
            "advertised RAW burst length. The author client only builds "
            "example['raw_burst_float32'] when the server's handshake metadata sets "
            "raw_frontend='rawvla', so this is what makes the RAW path live."
        ),
    )
    parser.add_argument(
        "--adapter-image-source",
        choices=("raw_burst", "image"),
        default="raw_burst",
        help="where the measurement lives in the openpi example",
    )
    args = parser.parse_args(argv)

    declared = N01V12Config.from_dict(json.loads(args.config.read_text()))
    if declared.training is not None and args.checkpoint is None and not args.allow_untrained:
        raise ValueError("complete service requires --checkpoint or explicit --allow-untrained")
    model, policy, bridge, audit = _load_arm(
        args.weights, args.tokenizer, args.config, args.suite, args.device
    )
    restored = None
    if args.checkpoint is not None:
        restored = load_arm_checkpoint(model, args.checkpoint, args.device, policy=policy)
    from owac.frameworks.n01_full.acceleration import configure_acceleration

    execution = configure_acceleration(
        model, torch.device(args.device), args.precision, pair_chunk=args.flow_batch_size
    )
    service = N01V12Service(
        model, policy, bridge, device=args.device, deadline=_parse_deadline(args)
    )
    policy_endpoint: Any = service
    if args.openpi_adapter:
        policy_endpoint = N01V12OpenPIAdapter(
            service,
            OpenPIAdapterConfig(
                cameras=tuple(model.config.cameras), image_source=args.adapter_image_source
            ),
        )
    health = policy_endpoint.health()
    health["execution"] = execution
    health["backend_audit"] = audit if isinstance(audit, dict) else str(audit)
    health["checkpoint"] = restored or {
        "step": None,
        "note": "no --checkpoint given: serving the freshly initialised arm",
    }
    if restored is not None:
        health["metadata"] = {
            "interface": "owac-n01-v1.2-deadline-readout",
            "suite": args.suite,
            "actions": "normalized",
            "action_horizon": int(policy.action_horizon),
            "action_dim": int(policy.action_dim),
            "replan_steps": int(bridge.protocol.replan_steps),
            "cameras": list(model.config.cameras),
            "trained_step": restored["step"],
        }
    print(json.dumps(health, ensure_ascii=False, indent=2, default=str), flush=True)
    if args.health:
        return 0

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    metadata = dict(
        health.get("metadata")
        or {
            "interface": "owac-n01-v1.2-deadline-readout",
            "suite": args.suite,
            "cameras": list(model.config.cameras),
        }
    )
    # The author client gates `example["raw_burst_float32"]` on this handshake
    # field.  Without it the client never produces the RAW burst and the adapter
    # would (correctly) refuse every request, so it is part of the wiring rather
    # than a cosmetic detail.
    if args.openpi_adapter and args.adapter_image_source == "raw_burst":
        metadata["raw_frontend"] = "rawvla"
        # Carry every arrival between policy queries. Actual IDs deduplicate overlap;
        # repeated transport slots do not become new evidence.
        metadata["raw_frontend_rnn_frames"] = 1
        metadata["raw_frontend_burst_frames"] = (
            model.config.window_frames
            if model.config.observation is not None
            else int(args.raw_frontend_burst_frames)
        )
        metadata["owac_native_transport"] = model.config.observation is not None
    else:
        metadata.setdefault("raw_frontend", "none")
    WebsocketPolicyServer(
        policy=policy_endpoint,
        host=args.host,
        port=args.port,
        idle_timeout=-1,
        metadata=metadata,
    ).serve_forever()
    return 0


def _smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="owac.runtime.n01_v12_service smoke")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--region", default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18020)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--payload", choices=("frames", "openpi"), default="frames")
    parser.add_argument("--burst", type=int, default=4, help="openpi RAW burst length")
    parser.add_argument("--noise-horizon", type=int, default=10)
    parser.add_argument("--noise-dim", type=int, default=32)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """``serve`` (default) or ``smoke`` the v1.2 readout."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "smoke":
        args = _smoke_parser().parse_args(arguments[1:])
        result = smoke(args)
        if args.output:
            _write_json(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
        return 0
    if arguments and arguments[0] == "serve":
        arguments = arguments[1:]
    return _serve(arguments)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
