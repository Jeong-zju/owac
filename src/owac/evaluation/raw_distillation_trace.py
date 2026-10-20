"""Concrete native-StarVLA and projected-token bridge stage-0 tracers.

The tracers temporarily wrap the original instance methods, capture their
inputs/outputs in computational order, and restore every method in ``finally``.
No parameter is mutated.  The native path calls the pinned original PI0 methods;
the bridge path calls ``RawVLAFeaturePolicy`` replacement methods.  They are
never the same callable wrapped twice.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import types
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "BoundaryTrace",
    "BridgePolicyTracer",
    "NativePolicyTracer",
    "Stage0TraceReport",
    "ToleranceReceipt",
    "compare_boundary_traces",
    "run_stage0_traces",
    "typed_prepared_observation",
]


@dataclass(slots=True)
class BoundaryTrace:
    name: str
    tensors: OrderedDict[str, torch.Tensor]
    history: Any
    reset: bool
    camera_order: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unavailable: Mapping[str, str] = field(default_factory=dict)

    def digest(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in self.tensors.items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            if value.dtype == torch.bool:
                digest.update(value.numpy().astype("uint8").tobytes())
            elif value.dtype in (torch.int32, torch.int64):
                digest.update(value.numpy().astype("<i8").tobytes())
            else:
                digest.update(value.float().numpy().astype("<f4").tobytes())
        return digest.hexdigest()


@contextlib.contextmanager
def _temporary_method(target: Any, name: str, replacement: Callable[..., Any]):
    original = getattr(target, name)
    setattr(target, name, types.MethodType(replacement, target))
    try:
        yield original
    finally:
        setattr(target, name, original)


def _capture_tensor(value: Any, name: str, store: OrderedDict[str, torch.Tensor]) -> Any:
    if torch.is_tensor(value):
        store[name] = value.detach().clone()
    return value


class NativePolicyTracer:
    """Trace the original PI0 embed/prefix/denoise implementation."""

    def __init__(self, policy: Any, *, image_keys: Sequence[str], num_steps: int = 10):
        self.policy = policy
        self.image_keys = tuple(image_keys)
        self.num_steps = int(num_steps)

    def capture(
        self,
        observation: Any,
        *,
        noise: torch.Tensor | None = None,
        history: Any = None,
        reset: bool = False,
        real_stats: Any = None,
        client_postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> BoundaryTrace:
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        unavailable: dict[str, str] = {}
        denoise_times: list[torch.Tensor] = []
        denoise_velocities: list[torch.Tensor] = []
        visual_index = {"value": 0}
        head_calls: list[dict[str, Any]] = []
        action_head = getattr(self.policy, "action_head", None)
        head_handle = None
        if isinstance(action_head, torch.nn.Module):

            def head_pre_hook(module, args, kwargs):
                head_calls.append({key: value for key, value in kwargs.items()})

            head_handle = action_head.register_forward_pre_hook(head_pre_hook, with_kwargs=True)

        def embed_image(module, image):
            original = self.policy.vlm.__dict__.get("__trace_original_embed_image")
            result = original(image)
            camera = self.image_keys[min(visual_index["value"], len(self.image_keys) - 1)]
            tensors[f"visual.{camera}"] = result.detach().clone()
            visual_index["value"] += 1
            return result

        def embed_language(module, tokens):
            original = self.policy.vlm.__dict__["__trace_original_embed_language"]
            result = original(tokens)
            tensors["language.tokens"] = tokens.detach().clone()
            tensors["language.embedding"] = result.detach().clone()
            return result

        def embed_prefix(module, images, img_masks, lang_tokens, lang_masks):
            original = self.policy.__dict__["__trace_original_embed_prefix"]
            embs, pad, att = original(images, img_masks, lang_tokens, lang_masks)
            tensors["prefix.embeds"] = embs.detach().clone()
            tensors["prefix.pad_masks"] = pad.detach().clone()
            tensors["prefix.att_masks"] = att.detach().clone()
            tensors["input.img_masks"] = (
                torch.stack([torch.as_tensor(mask) for mask in img_masks], dim=1)
                if img_masks
                else torch.empty(0)
            )
            tensors["input.lang_masks"] = lang_masks.detach().clone()
            return embs, pad, att

        def forward_prefix(module, **kwargs):
            original = self.policy.vlm.__dict__["__trace_original_forward_prefix"]
            out = original(**kwargs)
            if "attention_mask" in kwargs:
                tensors["forward.attention_mask"] = kwargs["attention_mask"].detach().clone()
            if "position_ids" in kwargs:
                tensors["forward.position_ids"] = kwargs["position_ids"].detach().clone()
            return out

        def denoise_step(module, state, prefix_masks, past, x_t, timestep):
            original = self.policy.__dict__["__trace_original_denoise"]
            velocity = original(state, prefix_masks, past, x_t, timestep)
            index = len(denoise_velocities)
            denoise_times.append(timestep.detach().clone())
            denoise_velocities.append(velocity.detach().clone())
            tensors[f"denoise.{index:02d}.time"] = timestep.detach().clone()
            tensors[f"denoise.{index:02d}.noisy_action"] = x_t.detach().clone()
            tensors[f"denoise.{index:02d}.attention_mask"] = prefix_masks.detach().clone()
            tensors[f"denoise.{index:02d}.position_ids"] = (
                (torch.cumsum(prefix_masks, dim=1) - 1).detach().clone()
            )
            tensors[f"denoise.{index:02d}.velocity"] = velocity.detach().clone()
            return velocity

        original_embed_image = self.policy.vlm.embed_image
        original_embed_language = self.policy.vlm.embed_language_tokens
        original_embed_prefix = self.policy.embed_prefix
        original_forward_prefix = self.policy.vlm.forward_prefix
        original_denoise = self.policy.denoise_step
        self.policy.vlm.__dict__["__trace_original_embed_image"] = original_embed_image
        self.policy.vlm.__dict__["__trace_original_embed_language"] = original_embed_language
        self.policy.__dict__["__trace_original_embed_prefix"] = original_embed_prefix
        self.policy.vlm.__dict__["__trace_original_forward_prefix"] = original_forward_prefix
        self.policy.__dict__["__trace_original_denoise"] = original_denoise
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_temporary_method(self.policy.vlm, "embed_image", embed_image))
                stack.enter_context(
                    _temporary_method(self.policy.vlm, "embed_language_tokens", embed_language)
                )
                stack.enter_context(_temporary_method(self.policy, "embed_prefix", embed_prefix))
                stack.enter_context(
                    _temporary_method(self.policy.vlm, "forward_prefix", forward_prefix)
                )
                stack.enter_context(_temporary_method(self.policy, "denoise_step", denoise_step))
                actions = self.policy.sample_actions(
                    observation, noise=noise, num_steps=self.num_steps
                )
        finally:
            if head_handle is not None:
                head_handle.remove()
            for module, name in (
                (self.policy.vlm, "__trace_original_embed_image"),
                (self.policy.vlm, "__trace_original_embed_language"),
                (self.policy, "__trace_original_embed_prefix"),
                (self.policy.vlm, "__trace_original_forward_prefix"),
                (self.policy, "__trace_original_denoise"),
            ):
                module.__dict__.pop(name, None)
        if getattr(observation, "state", None) is not None:
            tensors["state"] = torch.as_tensor(observation.state).detach().clone()
        for index, kwargs in enumerate(head_calls[: len(denoise_velocities)]):
            if "attention_mask" in kwargs:
                tensors[f"denoise.{index:02d}.suffix_attention_mask"] = (
                    kwargs["attention_mask"].detach().clone()
                )
            if "position_ids" in kwargs:
                tensors[f"denoise.{index:02d}.suffix_position_ids"] = (
                    kwargs["position_ids"].detach().clone()
                )
        tensors["actions.normalized"] = actions.detach().clone()
        if real_stats is not None:
            from owac.evaluation.raw_distillation import denormalize_actions

            tensors["actions.real7"] = denormalize_actions(actions[..., :7], real_stats)
        else:
            unavailable["actions.real7"] = "action statistics unavailable"
        if client_postprocess is not None:
            tensors["actions.executable_first5"] = (
                client_postprocess(actions)[:, :5].detach().clone()
            )
        else:
            unavailable["actions.executable_first5"] = "original client transform unavailable"
        if len(denoise_velocities) != 10:
            raise AssertionError(
                f"native tracer requires exactly 10 denoise steps, got {len(denoise_velocities)}"
            )
        tensors["denoise.count"] = torch.tensor(len(denoise_velocities), dtype=torch.int64)
        return BoundaryTrace(
            name="native-pi0",
            tensors=tensors,
            history=history,
            reset=reset,
            camera_order=self.image_keys,
            metadata={"implementation": "original PI0"},
            unavailable=unavailable,
        )


class BridgePolicyTracer:
    """Trace the RawVLAFeaturePolicy replacement implementation."""

    def __init__(self, bridge: Any, *, num_steps: int = 10):
        self.bridge = bridge
        self.num_steps = int(num_steps)

    def capture(
        self,
        tokens: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        history: Any = None,
        reset: bool = False,
        real_stats: Any = None,
        client_postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        camera_order: Sequence[str] = (),
    ) -> BoundaryTrace:
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        unavailable: dict[str, str] = {}
        denoise_times: list[torch.Tensor] = []
        denoise_velocities: list[torch.Tensor] = []
        head_calls: list[dict[str, Any]] = []
        base_policy = getattr(self.bridge, "base", None)
        action_head = getattr(base_policy, "action_head", None)
        head_handle = None
        if isinstance(action_head, torch.nn.Module):

            def bridge_head_hook(module, args, kwargs):
                head_calls.append({key: value for key, value in kwargs.items()})

            head_handle = action_head.register_forward_pre_hook(bridge_head_hook, with_kwargs=True)
        tensors["language.tokens"] = lang_tokens.detach().clone()
        tensors["input.lang_masks"] = lang_masks.detach().clone()
        tensors["state"] = state.detach().clone()
        camera_ids = tuple(getattr(tokens, "camera_ids", ()) or ())
        values = tokens.values
        if camera_ids:
            per_camera = int(values.shape[1] // len(camera_ids))
            for camera_index, camera in enumerate(camera_ids):
                start = camera_index * per_camera
                tensors[f"visual.{camera}"] = values[:, start : start + per_camera].detach().clone()
            if getattr(tokens, "mask", None) is not None:
                token_mask = tokens.mask.reshape(values.shape[0], len(camera_ids), per_camera)
                tensors["input.img_masks"] = token_mask.any(dim=-1).detach().clone()
        bridge_vlm = getattr(getattr(self.bridge, "base", None), "vlm", None)
        bridge_language = getattr(bridge_vlm, "embed_language_tokens", None)
        if callable(bridge_language):
            tensors["language.embedding"] = bridge_language(lang_tokens).detach().clone()
        else:
            unavailable["language.embedding"] = "bridge language embedding unavailable"

        def embed_prefix(module, tokens_arg, lang_arg, lang_mask_arg):
            original = self.bridge.__dict__["__trace_original_bridge_prefix"]
            prefix = original(tokens_arg, lang_arg, lang_mask_arg)
            tensors["prefix.embeds"] = prefix.embeds.detach().clone()
            tensors["prefix.pad_masks"] = prefix.pad_masks.detach().clone()
            tensors["prefix.att_masks"] = prefix.att_masks.detach().clone()
            tensors["forward.attention_mask"] = self.bridge.mask_4d(
                self.bridge.mask_2d(prefix.pad_masks, prefix.att_masks)
            )
            tensors["forward.position_ids"] = (
                (torch.cumsum(prefix.pad_masks, dim=1) - 1).detach().clone()
            )
            return prefix

        def denoise_step(module, *args, **kwargs):
            original = self.bridge.__dict__["__trace_original_bridge_denoise"]
            velocity = original(*args, **kwargs)
            index = len(denoise_velocities)
            timestep = kwargs.get("timestep", args[-1] if args else None)
            if timestep is not None:
                denoise_times.append(torch.as_tensor(timestep).detach().clone())
            denoise_velocities.append(velocity.detach().clone())
            tensors[f"denoise.{index:02d}.velocity"] = velocity.detach().clone()
            if timestep is not None:
                tensors[f"denoise.{index:02d}.time"] = torch.as_tensor(timestep).detach().clone()
            if len(args) >= 5:
                _, prefix_masks, _, x_t, _ = args[:5]
                tensors[f"denoise.{index:02d}.noisy_action"] = x_t.detach().clone()
                tensors[f"denoise.{index:02d}.attention_mask"] = prefix_masks.detach().clone()
                tensors[f"denoise.{index:02d}.position_ids"] = (
                    (torch.cumsum(prefix_masks, dim=1) - 1).detach().clone()
                )
            return velocity

        original_prefix = self.bridge.embed_prefix_from_visual_tokens
        original_denoise = self.bridge.denoise_step
        self.bridge.__dict__["__trace_original_bridge_prefix"] = original_prefix
        self.bridge.__dict__["__trace_original_bridge_denoise"] = original_denoise
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    _temporary_method(self.bridge, "embed_prefix_from_visual_tokens", embed_prefix)
                )
                stack.enter_context(_temporary_method(self.bridge, "denoise_step", denoise_step))
                actions = self.bridge.sample_actions_from_visual_tokens(
                    tokens,
                    lang_tokens,
                    lang_masks,
                    state,
                    noise=noise,
                    num_steps=self.num_steps,
                )
        finally:
            if head_handle is not None:
                head_handle.remove()
            self.bridge.__dict__.pop("__trace_original_bridge_prefix", None)
            self.bridge.__dict__.pop("__trace_original_bridge_denoise", None)
        actions = actions.detach()
        for index, kwargs in enumerate(head_calls[: len(denoise_velocities)]):
            if "attention_mask" in kwargs:
                tensors[f"denoise.{index:02d}.suffix_attention_mask"] = (
                    kwargs["attention_mask"].detach().clone()
                )
            if "position_ids" in kwargs:
                tensors[f"denoise.{index:02d}.suffix_position_ids"] = (
                    kwargs["position_ids"].detach().clone()
                )
        tensors["actions.normalized"] = actions.clone()
        if real_stats is not None:
            from owac.evaluation.raw_distillation import denormalize_actions

            tensors["actions.real7"] = denormalize_actions(actions[..., :7], real_stats)
        else:
            unavailable["actions.real7"] = "action statistics unavailable"
        if client_postprocess is not None:
            tensors["actions.executable_first5"] = (
                client_postprocess(actions)[:, :5].detach().clone()
            )
        else:
            unavailable["actions.executable_first5"] = "original client transform unavailable"
        if len(denoise_velocities) != 10:
            raise AssertionError(
                f"bridge tracer requires exactly 10 denoise steps, got {len(denoise_velocities)}"
            )
        tensors["denoise.count"] = torch.tensor(len(denoise_velocities), dtype=torch.int64)
        return BoundaryTrace(
            name="bridge-projected",
            tensors=tensors,
            history=history,
            reset=reset,
            camera_order=tuple(camera_order),
            metadata={"implementation": "RawVLAFeaturePolicy"},
            unavailable=unavailable,
        )


@dataclass(frozen=True, slots=True)
class ToleranceReceipt:
    repeat_max_error: Mapping[str, float]
    tolerances: Mapping[str, float]
    floor: float
    multiplier: float
    digest: str

    @classmethod
    def freeze(
        cls,
        first: BoundaryTrace,
        second: BoundaryTrace,
        *,
        floor: float = 1e-6,
        multiplier: float = 2.0,
        max_repeat_noise: float = 1e-3,
    ) -> ToleranceReceipt:
        if list(first.tensors) != list(second.tensors):
            raise AssertionError("native repeat boundary set/order changed")
        repeat: dict[str, float] = {}
        tolerances: dict[str, float] = {}
        for name in first.tensors:
            a, b = first.tensors[name], second.tensors[name]
            if a.shape != b.shape:
                raise AssertionError(f"native repeat shape changed at {name}")
            if a.dtype == torch.bool or a.dtype in (torch.int32, torch.int64):
                if not torch.equal(a, b):
                    raise AssertionError(f"native repeat changed exact boundary {name}")
                repeat[name] = 0.0
                tolerances[name] = 0.0
            else:
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise AssertionError(f"native repeat contains non-finite values at {name}")
                error = float((a.float() - b.float()).abs().max())
                if not math.isfinite(error) or error > max_repeat_noise:
                    raise AssertionError(
                        f"native repeat error {error} exceeds admissible {max_repeat_noise} "
                        f"at {name}"
                    )
                repeat[name] = error
                tolerances[name] = floor + multiplier * error
        digest = hashlib.sha256(
            json.dumps(
                {
                    "repeat": repeat,
                    "tolerances": tolerances,
                    "floor": floor,
                    "multiplier": multiplier,
                    "max_repeat_noise": max_repeat_noise,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return cls(
            repeat_max_error=repeat,
            tolerances=tolerances,
            floor=float(floor),
            multiplier=float(multiplier),
            digest=digest,
        )


@dataclass(frozen=True, slots=True)
class Stage0TraceReport:
    passed: bool
    first_mismatch: str | None
    native: BoundaryTrace
    bridge: BoundaryTrace
    tolerance: ToleranceReceipt
    max_errors: Mapping[str, float]
    missing_required: tuple[str, ...]
    extra_required: tuple[str, ...]
    inactive_errors: Mapping[str, float] = field(default_factory=dict)
    comparison_policy: str = ""
    raw_prefix_max_delta: float | None = None
    ignored_slot_count: int = 0
    inactive_camera_names: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "first_mismatch": self.first_mismatch,
            "max_errors": dict(self.max_errors),
            "missing_required": list(self.missing_required),
            "extra_required": list(self.extra_required),
            "inactive_errors": dict(self.inactive_errors),
            "comparison_policy": self.comparison_policy,
            "raw_prefix_max_delta": self.raw_prefix_max_delta,
            "ignored_slot_count": self.ignored_slot_count,
            "inactive_camera_names": list(self.inactive_camera_names),
            "tolerance_digest": self.tolerance.digest,
            "native_digest": self.native.digest(),
            "bridge_digest": self.bridge.digest(),
            "native_camera_order": list(self.native.camera_order),
            "bridge_camera_order": list(self.bridge.camera_order),
            "native_unavailable": dict(self.native.unavailable),
            "bridge_unavailable": dict(self.bridge.unavailable),
            "denoise_steps": int(self.native.tensors["denoise.count"].item()),
        }


_REQUIRED_ORDER = (
    "prefix.embeds",
    "prefix.pad_masks",
    "prefix.att_masks",
    "forward.attention_mask",
    "forward.position_ids",
    "actions.normalized",
    "denoise.count",
    "actions.real7",
    "actions.executable_first5",
)


def _is_exact(tensor: torch.Tensor) -> bool:
    return tensor.dtype == torch.bool or tensor.dtype in (torch.int32, torch.int64)


def _prefix_layout(
    trace: BoundaryTrace, order: Sequence[str]
) -> tuple[dict[str, tuple[int, int]], int]:
    prefix = trace.tensors.get("prefix.embeds")
    language = trace.tensors.get("language.tokens")
    if prefix is None or language is None:
        raise ValueError("prefix.embeds/language.tokens missing")
    if prefix.ndim != 3 or language.ndim != 2:
        raise ValueError("prefix/language shape is invalid")
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for camera in order:
        visual = trace.tensors.get(f"visual.{camera}")
        if visual is None or visual.ndim != 3:
            raise ValueError(f"visual.{camera} missing/invalid")
        count = int(visual.shape[1])
        ranges[camera] = (cursor, cursor + count)
        cursor += count
    language_count = int(language.shape[1])
    if cursor + language_count != int(prefix.shape[1]):
        raise ValueError("camera/language token counts do not sum to prefix length")
    return ranges, cursor


def compare_boundary_traces(
    native: BoundaryTrace,
    bridge: BoundaryTrace,
    tolerance: ToleranceReceipt,
) -> Stage0TraceReport:
    missing = tuple(
        name for name in _REQUIRED_ORDER if name not in native.tensors or name not in bridge.tensors
    )
    extra = tuple(
        name
        for name in list(native.tensors) + list(bridge.tensors)
        if name not in native.tensors or name not in bridge.tensors
    )
    if missing or extra:
        return Stage0TraceReport(
            passed=False,
            first_mismatch=(missing + extra)[0] if missing or extra else None,
            native=native,
            bridge=bridge,
            tolerance=tolerance,
            max_errors={},
            missing_required=missing,
            extra_required=extra,
            comparison_policy="fail-closed missing boundary",
        )
    first_bad = None
    errors: dict[str, float] = {}
    inactive_errors: dict[str, float] = {}
    inactive = set(native.metadata.get("inactive_cameras", ())) | set(
        bridge.metadata.get("inactive_cameras", ())
    )
    native_order = tuple(native.camera_order)
    bridge_order = tuple(bridge.camera_order)
    active_native = tuple(name for name in native_order if name not in inactive)
    active_bridge = tuple(name for name in bridge_order if name not in inactive)
    if active_native != active_bridge:
        first_bad = "metadata.camera_order"
    try:
        native_ranges, _ = _prefix_layout(native, native_order)
        bridge_ranges, _ = _prefix_layout(bridge, bridge_order)
    except ValueError as exc:
        first_bad = first_bad or f"metadata.prefix_layout:{exc}"
        native_ranges, bridge_ranges = {}, {}
    ignored_native: set[int] = set()
    ignored_bridge: set[int] = set()
    raw_prefix_delta: float | None = None
    if first_bad is None and inactive:
        for camera in sorted(inactive):
            if camera not in native_order or camera not in bridge_order:
                first_bad = f"metadata.inactive_camera_unknown:{camera}"
                break
            native_range = native_ranges.get(camera)
            bridge_range = bridge_ranges.get(camera)
            if native_range is None or bridge_range is None:
                first_bad = f"metadata.inactive_range:{camera}"
                break
            for trace, camera_range, label in (
                (native, native_range, "native"),
                (bridge, bridge_range, "bridge"),
            ):
                masks = trace.tensors.get("input.img_masks")
                pads = trace.tensors.get("prefix.pad_masks")
                if masks is None or pads is None:
                    first_bad = f"metadata.inactive_masks_missing:{label}:{camera}"
                    break
                if masks.ndim != 2 or int(masks.shape[1]) != len(trace.camera_order):
                    first_bad = f"metadata.inactive_mask_shape:{label}:{camera}"
                    break
                index = list(trace.camera_order).index(camera)
                start, end = camera_range
                if bool(masks[:, index].any()):
                    first_bad = f"metadata.inactive_camera_active:{label}:{camera}"
                    break
                if bool(pads[:, start:end].any()):
                    first_bad = f"metadata.inactive_prefix_active:{label}:{camera}"
                    break
            if first_bad is not None:
                break
            ignored_native.update(range(*native_range))
            ignored_bridge.update(range(*bridge_range))
    if first_bad is None:
        a = native.tensors["prefix.embeds"]
        b = bridge.tensors["prefix.embeds"]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            first_bad = "prefix.embeds:shape/nonfinite"
        else:
            raw_prefix_delta = float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
            native_keep = [i for i in range(a.shape[1]) if i not in ignored_native]
            bridge_keep = [i for i in range(b.shape[1]) if i not in ignored_bridge]
            if len(native_keep) != len(bridge_keep):
                first_bad = "prefix.embeds:active_layout"
            else:
                active_error = (
                    float((a[:, native_keep].float() - b[:, bridge_keep].float()).abs().max())
                    if native_keep
                    else 0.0
                )
                errors["prefix.embeds"] = active_error
                inactive_errors["prefix.embeds"] = raw_prefix_delta
                if active_error > tolerance.tolerances.get("prefix.embeds", tolerance.floor):
                    first_bad = "prefix.embeds"
    if native.reset != bridge.reset:
        first_bad = first_bad or "metadata.reset"
    if native.history != bridge.history:
        first_bad = first_bad or "metadata.history"
    ordered = list(native.tensors) + [name for name in bridge.tensors if name not in native.tensors]
    for name in ordered:
        if name == "prefix.embeds":
            continue
        if name not in bridge.tensors or name not in native.tensors:
            first_bad = first_bad or f"{name}:missing"
            errors[name] = float("inf")
            continue
        a, b = native.tensors[name], bridge.tensors[name]
        if a.shape != b.shape:
            first_bad = first_bad or f"{name}:shape"
            errors[name] = float("inf")
            continue
        if a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            first_bad = first_bad or f"{name}:nonfinite"
            errors[name] = float("inf")
            continue
        if name.startswith("visual.") and name.split(".", 1)[1] in inactive:
            inactive_errors[name] = float((a.float() - b.float()).abs().max())
            errors[name] = 0.0
            continue
        if _is_exact(a):
            if not torch.equal(a, b):
                first_bad = first_bad or f"{name}:exact"
                errors[name] = float("inf")
        else:
            error = float((a.float() - b.float()).abs().max())
            errors[name] = error
            if first_bad is None and error > tolerance.tolerances.get(name, tolerance.floor):
                first_bad = name
    policy = "active_cameras_and_all_language_slots; raw_prefix_full_delta_reported"
    return Stage0TraceReport(
        passed=first_bad is None,
        first_mismatch=first_bad,
        native=native,
        bridge=bridge,
        tolerance=tolerance,
        max_errors=errors,
        missing_required=(),
        extra_required=(),
        inactive_errors=inactive_errors,
        comparison_policy=policy,
        raw_prefix_max_delta=raw_prefix_delta,
        ignored_slot_count=len(ignored_native),
        inactive_camera_names=tuple(sorted(inactive)),
    )


def run_stage0_traces(
    native_tracer: NativePolicyTracer,
    bridge_tracer: BridgePolicyTracer,
    *,
    observation: Any,
    tokens: Any,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor | None = None,
    history: Any = None,
    reset: bool = False,
    real_stats: Any = None,
    client_postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    camera_order: Sequence[str] = (),
    inactive_cameras: Sequence[str] = (),
    tolerance_floor: float = 1e-6,
    tolerance_multiplier: float = 2.0,
) -> Stage0TraceReport:
    """Native repeat/tolerance is frozen before the bridge is ever called."""
    first = native_tracer.capture(
        observation,
        noise=noise,
        history=history,
        reset=reset,
        real_stats=real_stats,
        client_postprocess=client_postprocess,
    )
    second = native_tracer.capture(
        observation,
        noise=noise,
        history=history,
        reset=reset,
        real_stats=real_stats,
        client_postprocess=client_postprocess,
    )
    for trace in (first, second):
        trace.metadata["inactive_cameras"] = tuple(inactive_cameras)
    tolerance = ToleranceReceipt.freeze(
        first, second, floor=tolerance_floor, multiplier=tolerance_multiplier
    )
    bridge = bridge_tracer.capture(
        tokens,
        lang_tokens,
        lang_masks,
        state,
        noise=noise,
        history=history,
        reset=reset,
        real_stats=real_stats,
        client_postprocess=client_postprocess,
        camera_order=camera_order,
    )
    bridge.metadata["inactive_cameras"] = tuple(inactive_cameras)
    return compare_boundary_traces(first, bridge, tolerance)


def typed_prepared_observation(
    *,
    prepared_rgb: torch.Tensor,
    camera_order: Sequence[str],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    native_image_keys: Sequence[str] | None = None,
) -> Any:
    """Build the native Observation from already-original-preprocessed RGB.

    ``prepared_rgb`` is ``[B,C,3,H,W]`` in the manifest camera order.  The
    teacher receipt must bind that preparation; this helper does not resize or
    apply gamma/ISP.
    """
    if prepared_rgb.ndim != 5 or int(prepared_rgb.shape[1]) != len(camera_order):
        raise ValueError("prepared_rgb must be [B,C,3,H,W] in camera_order")
    try:
        from starVLA.model.framework.VLM4A.PI0 import Observation
    except ImportError as exc:
        raise RuntimeError(f"native StarVLA Observation is unavailable: {exc}") from exc
    images = {camera: prepared_rgb[:, index] for index, camera in enumerate(camera_order)}
    masks = {
        camera: torch.ones(prepared_rgb.shape[0], dtype=torch.bool, device=prepared_rgb.device)
        for camera in camera_order
    }
    return Observation(
        images=images,
        image_masks=masks,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        state=state,
    )
