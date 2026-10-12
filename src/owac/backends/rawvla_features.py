"""Feature bridge: an OWAC encoder's visual tokens replace the π0.5 image tower.

Run with the external RawVLA runtime (Python 3.10 + CUDA + the author checkpoint).
This module imports only ``torch`` at module scope and is written in Python
3.10-compatible syntax, so the isolated StarVLA runtime can import it with
``PYTHONPATH=<this worktree>/src:<isolated StarVLA>/``.

What this module is
-------------------

A thin wrapper around the pinned StarVLA ``PI05``/``PI0`` policy class. It does
not re-implement the backend: it reuses ``vlm.embed_language_tokens``,
``vlm.forward_prefix``, ``forward_with_action_expert``, the action head's
``prepare_flow_targets`` / ``embed_action_suffix`` / ``decode_action_velocity``
and cached denoise forward, the original flow solver, and the pinned block-causal
mask helpers. Only the point where ``vlm.embed_image(image)`` used to deliver the
visual prefix is replaced, so the OWAC tokens *are* the visual prefix.

Prefix layout produced (``FeaturePrefix``)
------------------------------------------

Let ``C = len(image_keys)``, ``N = tokens_per_camera``, ``L`` the language length
and ``W`` the language width. ``VisualTokens.values`` must be ``[B, C*N, D]`` with
one contiguous group of ``N`` tokens per camera named by
``VisualTokens.camera_ids``::

    index range        content                              pad mask
    [0, N)             image_keys[0] tokens                 tokens.mask group 0
    [N, 2N)            image_keys[1] tokens                 tokens.mask group 1
    ...                ...                                  ...
    [C*N, C*N+L)       language tokens * sqrt(W)            lang_masks

* the group order is always ``image_keys``; ``camera_ids`` identifies each group
  and may arrive in any order. A camera present in ``image_keys`` but absent from
  ``camera_ids`` contributes ``N`` zero tokens with pad mask ``False``, the same
  convention the pinned ``_prepare_examples`` uses for a missing camera;
* a camera in ``camera_ids`` that is not in ``image_keys`` is an error, never a
  silent drop;
* ``pad_masks`` is per token and is ``False`` for an inactive camera, for a token
  whose ``values`` are not finite, for a token whose ``positions`` are not finite
  and for a token masked by ``VisualTokens.mask``;
* ``att_masks`` is ``False`` for every prefix token, exactly as in
  ``PI0.embed_prefix``: the prefix is one bidirectional block and the action
  suffix attends to all of it;
* ``position_ids = cumsum(pad_masks, dim=1) - 1`` is derived exactly as in the
  pinned code, so a masked token does not consume a position id;
* masked token values are zeroed before projection, so a non-finite measurement
  can never poison the KV cache with NaN even though the additive mask gives it
  zero attention weight.

``FeaturePrefix`` carries ``positions``, ``token_camera``, ``camera_ids``,
``scale`` and ``time`` next to the tensors. They are consumed, not logged:
positions are validated (finite) and drive "unsupported position" masking, camera
ids drive the group layout, and the observation time rides on the prefix so a
caller can tell which measurement produced a cached prefix.

One observation runs the encoder exactly once
---------------------------------------------

The visual prefix is built once per forward call. The ten flow-denoising
iterations re-read it from the language-model KV cache (``past_key_values``),
never from the encoder. Every entry point records ``prefix_builds`` /
``encoder_calls`` / ``denoise_steps`` in ``last_run_stats`` and, with
``assert_single_encoder_call=True`` (the default), raises if the encoder hook or
the prefix build ran a number of times other than one.

Gradients and freezing
----------------------

``freeze_backend()`` sets ``requires_grad_(False)`` on every backend parameter. It
never wraps the training forward in ``no_grad``, so the original flow-matching
action loss still reaches the new encoder and its readout.
``torch.inference_mode()`` is used only by ``predict_action*``. The gradient path
is verified for real in ``tests/encoders/n02/test_rawvla_features.py`` and by
``check-contract``.

Loading with the vision tower replaced
--------------------------------------

``build_feature_policy()`` reuses ``owac.backends.rawvla.new_model`` unchanged,
which *does* construct the SigLIP vision tower and the multi-modal projector
before any checkpoint is read. ``load_audited_replaced_vision()`` therefore:

1. runs the unmodified ``owac.backends.rawvla.load_audited`` against a restricted
   view holding only language/action parameters, so those are still audited item
   by item (missing, unexpected and shape mismatches all raise);
2. reports vision keys as ``replaced_vision_missing`` (allowed: the tower is never
   executed) or ``replaced_vision_present`` (in the checkpoint, still not run);
3. optionally releases the tower and projector and reports the freed bytes.

``owac.backends.rawvla.load_audited`` is left untouched for the full C0 baseline.

Command line
------------

::

    python -m owac.backends.rawvla_features check-contract
    python -m owac.backends.rawvla_features check-assets --suite libero
    python -m owac.backends.rawvla_features single-batch --suite libero \\
        --tokenizer /path/paligemma_tokenizer.model \\
        --checkpoint /path/pi05_libero --instruction "pick up the cube"

``check-contract`` is a CPU self-check against the stub backend defined at the
bottom of this file. It needs no StarVLA, no checkpoint and no data, and it is
**not** a π0.5 acceptance: the stub is a structurally faithful miniature, not the
real π0.5. ``single-batch`` performs a real forward/backward through the real
weights and requires the external runtime; when an asset is missing it prints a
JSON ``unavailable`` report and exits 2 instead of raising a traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

__all__ = [
    "ACTION_PROTOCOLS",
    "ActionChunkCache",
    "ActionProtocol",
    "EncoderReadoutTokenSource",
    "FeaturePrefix",
    "IMAGE_KEYS",
    "LIBERO",
    "MissingAssetError",
    "ROBOTWIN",
    "RawVLAFeaturePolicy",
    "StaticTokenSource",
    "TokenSource",
    "VISION_KEY_PREFIXES",
    "build_feature_policy",
    "check_contract",
    "check_runtime_assets",
    "load_audited_replaced_vision",
    "protocol_for",
    "release_replaced_vision",
]

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# ``OpenPIPaliGemma`` nests the PaliGemma core as ``vlm.model.model`` and exposes
# ``vlm.model.model.vision_tower.*`` / ``vlm.model.model.multi_modal_projector.*``.
# Both are replaced: ``embed_image`` is ``multi_modal_projector(vision_tower(x))``.
VISION_KEY_PREFIXES = (
    "vlm.model.model.vision_tower.",
    "vlm.model.model.multi_modal_projector.",
)

LIBERO = "libero"
ROBOTWIN = "robotwin"


class MissingAssetError(RuntimeError):
    """An external asset (runtime, weights, data) is absent.

    This is an infrastructure state, not a bug in the bridge. Callers must report
    it and must never present a stub or a synthetic batch as a real π0.5 result.
    """


# --------------------------------------------------------------------------- #
# Mask helpers
# --------------------------------------------------------------------------- #
def _local_make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Faithful local copy of the pinned ``OpenPI_ActionHead`` mask builder."""
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError(f"Expected 2D masks, got {pad_masks.ndim=} {att_masks.ndim=}")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _local_make_att_4d_masks(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Faithful local copy of the pinned ``OpenPI_ActionHead`` mask builder."""
    return torch.where(att_2d_masks[:, None, :, :], 0.0, -2.3819763e38)


def _resolve_mask_helpers(base: Any) -> tuple:
    """Prefer the pinned mask builders from the backend's own module.

    The bridge must use exactly the block-causal convention of the pinned
    ``OpenPI_ActionHead``. When the backend action head comes from a module that
    defines ``make_att_2d_masks`` / ``make_att_4d_masks`` (the pinned StarVLA
    layout) those functions are used directly. With a stub whose module does not
    define them, the local copies above are used and the source is reported.
    """
    head = getattr(base, "action_head", None)
    module = sys.modules.get(type(head).__module__) if head is not None else None
    if module is not None:
        two = getattr(module, "make_att_2d_masks", None)
        four = getattr(module, "make_att_4d_masks", None)
        if callable(two) and callable(four):
            return two, four, f"{module.__name__} (pinned backend module)"
    return _local_make_att_2d_masks, _local_make_att_4d_masks, "local copy in this module"


def verify_mask_helpers(base: Any) -> dict:
    """Numerically compare the resolved mask helpers with the local copies."""
    two, four, source = _resolve_mask_helpers(base)
    if two is _local_make_att_2d_masks:
        return {
            "source": source,
            "matches_local_copy": None,
            "note": "no pinned backend module found; the local copy is in use",
        }
    generator = torch.Generator().manual_seed(0)
    pad = torch.rand((3, 11), generator=generator) > 0.25
    att = torch.zeros((3, 11), dtype=torch.bool)
    att[:, 7] = True
    pad[0, 0] = True
    same_2d = bool(torch.equal(two(pad, att), _local_make_att_2d_masks(pad, att)))
    same_4d = bool(
        torch.equal(
            four(two(pad, att)), _local_make_att_4d_masks(_local_make_att_2d_masks(pad, att))
        )
    )
    return {"source": source, "matches_local_copy": bool(same_2d and same_4d)}


# --------------------------------------------------------------------------- #
# Frozen action protocol
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActionProtocol:
    """Frozen action settings for one suite.

    ``action_dim`` is the *internal* padding width of the action expert and never
    the real control dimension. ``num_denoise_steps`` is a separate parameter: it
    is not derived from the horizon, and the RoboTwin replan cadence is declared
    here rather than inferred from ``action_horizon``.
    """

    name: str
    action_horizon: int
    replan_steps: int
    control_dim: int
    action_dim: int = 32
    max_state_dim: int = 32
    num_denoise_steps: int = 10
    state_note: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "action_horizon": self.action_horizon,
            "replan_steps": self.replan_steps,
            "control_dim": self.control_dim,
            "action_dim": self.action_dim,
            "max_state_dim": self.max_state_dim,
            "num_denoise_steps": self.num_denoise_steps,
            "state_note": self.state_note,
        }


ACTION_PROTOCOLS = {
    LIBERO: ActionProtocol(
        name=LIBERO,
        action_horizon=10,
        replan_steps=5,
        control_dim=7,
        state_note="8-dim robot state in, 7-dim end-effector control chunk out",
    ),
    ROBOTWIN: ActionProtocol(
        name=ROBOTWIN,
        action_horizon=50,
        replan_steps=50,
        control_dim=14,
        state_note=(
            "14-dim absolute joint targets; the replan cadence mirrors the frozen client and "
            "is declared here, never inferred from action_horizon"
        ),
    ),
}


def protocol_for(name: Any) -> ActionProtocol:
    if isinstance(name, ActionProtocol):
        return name
    try:
        return ACTION_PROTOCOLS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown action protocol {name!r}; known: {sorted(ACTION_PROTOCOLS)}"
        ) from exc


# --------------------------------------------------------------------------- #
# Prefix container
# --------------------------------------------------------------------------- #
@dataclass
class FeaturePrefix:
    """Visual prefix built from OWAC tokens, ready for the pinned backend.

    ``embeds``/``pad_masks``/``att_masks`` are exactly the tensors the pinned
    ``embed_prefix`` would have returned. ``positions``/``token_camera``/
    ``camera_ids``/``scale``/``time`` describe where each visual token came from
    and stay available to the caller (token probes, attribution, staleness
    checks); they are consumed fields, not decoration.
    """

    embeds: torch.Tensor
    pad_masks: torch.Tensor
    att_masks: torch.Tensor
    positions: torch.Tensor
    token_camera: torch.Tensor
    camera_ids: tuple
    scale: Any = None
    time: Any = None
    notes: tuple = ()
    missing_cameras: tuple = ()

    @property
    def n_visual(self) -> int:
        return int(self.positions.shape[1])

    @property
    def language_len(self) -> int:
        return int(self.embeds.shape[1]) - self.n_visual

    @property
    def visual_slice(self) -> slice:
        return slice(0, self.n_visual)

    @property
    def language_slice(self) -> slice:
        return slice(self.n_visual, int(self.embeds.shape[1]))

    def position_ids(self) -> torch.Tensor:
        """The pinned numbering: ``cumsum(pad_masks, dim=1) - 1``."""
        return torch.cumsum(self.pad_masks, dim=1) - 1

    def summary(self) -> dict:
        groups = max(len(self.camera_ids), 1)
        return {
            "embeds_shape": list(self.embeds.shape),
            "embeds_dtype": str(self.embeds.dtype),
            "n_visual": self.n_visual,
            "language_len": self.language_len,
            "camera_ids": list(self.camera_ids),
            "tokens_per_camera": self.n_visual // groups,
            "masked_tokens": int((~self.pad_masks).sum()),
            "missing_cameras": list(self.missing_cameras),
            "positions_shape": list(self.positions.shape),
            "att_masks_true": int(self.att_masks.sum()),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Token sources
# --------------------------------------------------------------------------- #
def _as_bool_list(mask: Any) -> list:
    if mask is None:
        return []
    if isinstance(mask, (list, tuple)):
        return [bool(value) for value in mask]
    if hasattr(mask, "tolist"):
        return [bool(value) for value in mask.tolist()]
    return [bool(mask)]


def _module_records(module: Any, prefix: str, *, trainable_only: bool = False) -> list:
    if not isinstance(module, nn.Module):
        return []
    records = []
    for name, param in module.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        pending = isinstance(param, torch.nn.parameter.UninitializedParameter)
        records.append(
            {
                "name": f"{prefix}.{name}" if prefix else name,
                "parameters": 0 if pending else int(param.numel()),
                "shape": None if pending else list(param.shape),
                "initialized": not pending,
                "dtype": str(param.dtype),
                "requires_grad": bool(param.requires_grad),
            }
        )
    return records


def _concat_visual_tokens(groups: Sequence[Any], camera_ids: Sequence[str]) -> Any:
    """Concatenate per-camera ``VisualTokens`` groups along the token axis."""
    from owac.core.encoder_contract import VisualTokens

    if not groups:
        raise ValueError("no per-camera VisualTokens groups to concatenate")
    if len(camera_ids) != len(groups):
        raise ValueError("camera_ids and per-camera VisualTokens groups must have equal length")
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError(f"per-camera groups must carry distinct camera ids, got {camera_ids}")
    shapes = {tuple(group.values.shape) for group in groups}
    if len(shapes) != 1:
        raise ValueError(f"per-camera VisualTokens must share [B, N, D], got {sorted(shapes)}")
    bsize, per_camera, _dim = next(iter(shapes))
    device = groups[0].values.device
    masks, positions = [], []
    for group in groups:
        if group.mask is None:
            masks.append(torch.ones((bsize, per_camera), dtype=torch.bool, device=device))
        else:
            mask = torch.as_tensor(group.mask, device=device)
            if tuple(mask.shape) != (bsize, per_camera):
                raise ValueError(f"VisualTokens.mask must be [B, N], got {tuple(mask.shape)}")
            masks.append(mask.to(torch.bool))
        if group.positions is None:
            positions.append(torch.full((bsize, per_camera, 2), float("nan"), device=device))
        else:
            value = torch.as_tensor(group.positions, device=device, dtype=torch.float32)
            if value.ndim == 2:
                value = value[:, :, None]
            if value.ndim != 3 or tuple(value.shape[:2]) != (bsize, per_camera):
                raise ValueError(
                    f"VisualTokens.positions must be [B, N] or [B, N, P], got {tuple(value.shape)}"
                )
            positions.append(value)
    width = max(int(value.shape[-1]) for value in positions)
    padded = []
    for value in positions:
        if int(value.shape[-1]) < width:
            fill = torch.full(
                (*value.shape[:2], width - int(value.shape[-1])), float("nan"), device=device
            )
            value = torch.cat([value, fill], dim=-1)
        padded.append(value)
    return VisualTokens(
        values=torch.cat([group.values for group in groups], dim=1),
        mask=torch.cat(masks, dim=1),
        camera_ids=tuple(camera_ids),
        positions=torch.cat(padded, dim=1),
    )


class TokenSource:
    """Callable that yields ``VisualTokens`` for one observation and counts calls.

    The bridge owns the *timing* of this call: exactly one call per forward and
    never one per denoising iteration. Every subclass must increment ``calls`` on
    each ``__call__`` so the bridge can assert that.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def __call__(self, observation: Any) -> Any:
        raise NotImplementedError

    def reset(self, batch_mask: Any = None) -> dict:
        self.resets += 1
        return {"reset": True, "episodes": _as_bool_list(batch_mask)}

    def trainable_records(self) -> list:
        return []


class StaticTokenSource(TokenSource):
    """Replay pre-computed ``VisualTokens`` (the encoder ran elsewhere).

    Used when the OWAC encoder cannot run in the same process as the backend, for
    example tokens dumped by the encoder host. The single-call contract still
    holds, and the token origin must be reported by the caller.
    """

    def __init__(self, tokens: Any) -> None:
        super().__init__()
        self.tokens = tokens

    def __call__(self, observation: Any = None) -> Any:
        self.calls += 1
        return self.tokens


class EncoderReadoutTokenSource(TokenSource):
    """Adapter from the v0 contract ``Encoder``/``Readout`` to a token source.

    One ``__call__`` runs ``encoder.step`` once per camera and ``readout`` once per
    camera, then concatenates the per-camera token groups. Per-camera encoder state
    lives here (the contract gives each camera its own state) and ``reset``
    forwards a per-episode mask to ``encoder.reset``.

    ``observation`` is a mapping with keys:

    ``frames``
        one ``MeasurementFrame`` per camera, or a mapping ``camera_id -> frame``
        (required);
    ``context``
        an ``EmbodimentContext`` (optional);
    ``budget``
        a ``ComputeBudget`` (optional, defaults to the source default).
    """

    def __init__(
        self,
        encoder: Any,
        readout: Any,
        *,
        token_budget: int = 256,
        max_updates: int = 4,
        max_history: int = 4,
        budget: Any = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.readout = readout
        self.token_budget = int(token_budget)
        self._budget = budget
        self.max_updates = int(max_updates)
        self.max_history = int(max_history)
        self.states: dict = {}
        self.step_calls = 0
        self.readout_calls = 0
        self.last_cost: Any = None

    def _resolve_budget(self, observation: Any) -> Any:
        budget = observation.get("budget") if isinstance(observation, dict) else None
        if budget is not None:
            return budget
        if self._budget is not None:
            return self._budget
        from owac.core.encoder_contract import ComputeBudget

        return ComputeBudget(max_updates=self.max_updates, max_history=self.max_history)

    def _resolve_context(self, observation: Any) -> Any:
        from owac.core.encoder_contract import EmbodimentContext

        if isinstance(observation, dict) and observation.get("context") is not None:
            return observation["context"]
        return EmbodimentContext()

    def _items(self, observation: Any) -> list:
        frames = observation.get("frames") if isinstance(observation, dict) else observation
        if frames is None:
            raise ValueError("observation must provide 'frames'")
        if isinstance(frames, dict):
            return [
                (getattr(frame, "camera_id", "") or key, frame) for key, frame in frames.items()
            ]
        if hasattr(frames, "camera_id"):
            return [(frames.camera_id, frames)]
        return [(frame.camera_id, frame) for frame in frames]

    def __call__(self, observation: Any) -> Any:
        items = self._items(observation)
        if not items:
            raise ValueError("observation provides no measurement frames")
        context = self._resolve_context(observation)
        budget = self._resolve_budget(observation)
        groups, camera_ids = [], []
        for camera_id, frame in items:
            if not camera_id:
                raise ValueError(
                    "every MeasurementFrame needs a camera_id: the backend prefix layout "
                    "cannot place tokens without camera identity"
                )
            output = self.encoder.step(frame, context, self.states.get(camera_id), budget)
            self.states[camera_id] = output.state
            self.step_calls += 1
            self.last_cost = output.cost
            groups.append(self.readout(output, self.token_budget))
            self.readout_calls += 1
            camera_ids.append(camera_id)
        self.calls += 1
        return _concat_visual_tokens(groups, camera_ids)

    def reset(self, batch_mask: Any = None) -> dict:
        reset_ids = []
        for camera_id, state in list(self.states.items()):
            self.states[camera_id] = self.encoder.reset(state, batch_mask)
            reset_ids.append(camera_id)
        self.resets += 1
        return {"reset_cameras": reset_ids, "episodes": _as_bool_list(batch_mask)}

    def detach(self) -> list:
        detached = []
        for camera_id, state in list(self.states.items()):
            self.states[camera_id] = self.encoder.detach_state(state)
            detached.append(camera_id)
        return detached

    def trainable_records(self) -> list:
        records = _module_records(self.encoder, "token_source.encoder", trainable_only=True)
        records += _module_records(self.readout, "token_source.readout", trainable_only=True)
        return records


# --------------------------------------------------------------------------- #
# Action chunk cache
# --------------------------------------------------------------------------- #
class ActionChunkCache:
    """Per-episode action-chunk cache mirroring the frozen replan protocol.

    The deployed runtime caches one chunk per process (``chunk_start`` in
    ``owac.runtime.rawvla_robotwin.ModelClient``). A batched rollout needs one slot
    per episode, and an episode that ends must be reset *together with* that
    episode's encoder state while the other episodes keep theirs.
    """

    def __init__(self, protocol: Any, *, replan_steps: Any = None) -> None:
        self.protocol = protocol_for(protocol)
        self.replan_steps = (
            int(replan_steps) if replan_steps is not None else self.protocol.replan_steps
        )
        if self.replan_steps < 1:
            raise ValueError("replan_steps must be >= 1")
        self._start: dict = {}
        self._chunks: dict = {}
        self.replans = 0
        self.resets = 0

    @property
    def episodes(self) -> list:
        return sorted(self._start)

    def needs_replan(self, slot: int, step: int) -> bool:
        start = self._start.get(slot)
        if start is None or self._chunks.get(slot) is None:
            return True
        return (int(step) - int(start)) >= self.replan_steps

    def store(self, slot: int, step: int, actions: Any) -> None:
        self._start[int(slot)] = int(step)
        self._chunks[int(slot)] = actions
        self.replans += 1

    def action(self, slot: int, step: int) -> Any:
        chunk = self._chunks.get(int(slot))
        if chunk is None:
            raise KeyError(f"episode slot {slot} has no cached action chunk")
        offset = int(step) - int(self._start[int(slot)])
        if offset < 0 or offset >= len(chunk):
            raise IndexError(f"episode slot {slot}: step {step} outside the cached chunk")
        return chunk[offset]

    def reset(self, batch_mask: Any = None) -> list:
        """Reset only the finished episodes; ``None`` resets every cached episode."""
        if batch_mask is None:
            slots = sorted(self._start)
        else:
            flags = _as_bool_list(batch_mask)
            slots = [slot for slot in sorted(self._start) if slot < len(flags) and flags[slot]]
        for slot in slots:
            self._start.pop(slot, None)
            self._chunks.pop(slot, None)
        self.resets += 1
        return slots

    def state_bytes(self) -> int:
        total = 0
        for chunk in self._chunks.values():
            element_size = getattr(chunk, "element_size", None)
            numel = getattr(chunk, "numel", None)
            if callable(element_size) and callable(numel):
                total += int(element_size()) * int(numel())
        return total


# --------------------------------------------------------------------------- #
# Policy width helpers
# --------------------------------------------------------------------------- #
def _policy_width(base: Any) -> int:
    language_model = getattr(getattr(base, "vlm", None), "language_model", None)
    width = getattr(getattr(language_model, "config", None), "hidden_size", None)
    if isinstance(width, int) and width > 0:
        return int(width)
    for param in getattr(base, "parameters", list)():
        if param.ndim == 2:
            return int(param.shape[-1])
    raise ValueError("cannot determine the language width of the backend policy")


def _policy_depth(base: Any) -> int:
    language_model = getattr(getattr(base, "vlm", None), "language_model", None)
    depth = getattr(getattr(language_model, "config", None), "num_hidden_layers", None)
    return int(depth) if isinstance(depth, int) else -1


# --------------------------------------------------------------------------- #
# The bridge
# --------------------------------------------------------------------------- #
class RawVLAFeaturePolicy:
    """Replace the π0.5 image tower with OWAC visual tokens.

    The wrapped ``base`` is the pinned ``PI05``/``PI0`` instance (or the
    structurally faithful stub at the bottom of this file). ``__getattr__``
    delegates unknown attributes to it, so ``load_audited`` and other existing
    helpers keep working on the wrapped policy.
    """

    def __init__(
        self,
        base: Any,
        token_source: Any,
        *,
        protocol: Any = LIBERO,
        token_dim: int = 2048,
        tokens_per_camera: int = 256,
        image_keys: Sequence[str] = IMAGE_KEYS,
        projection: Any = None,
        projection_dtype: torch.dtype = torch.float32,
        mask_policy: str = "mask_only",
        order_within_camera: str = "as_given",
        assert_single_encoder_call: bool = True,
        freeze_backend: bool = True,
        vision_key_prefixes: Sequence[str] = VISION_KEY_PREFIXES,
    ) -> None:
        if base is None:
            raise ValueError("RawVLAFeaturePolicy needs a backend policy instance")
        if mask_policy not in {"mask_only", "mask_and_validity"}:
            raise ValueError(f"unknown mask_policy: {mask_policy!r}")
        if order_within_camera not in {"as_given", "row_major"}:
            raise ValueError(f"unknown order_within_camera: {order_within_camera!r}")
        self.base = base
        self.protocol = protocol_for(protocol)
        self.image_keys = tuple(image_keys)
        if not self.image_keys:
            raise ValueError("image_keys must not be empty")
        self.token_dim = int(token_dim)
        self.tokens_per_camera = int(tokens_per_camera)
        if self.tokens_per_camera < 1:
            raise ValueError("tokens_per_camera must be >= 1")
        self.token_source = token_source
        self.mask_policy = mask_policy
        self.order_within_camera = order_within_camera
        self.assert_single_encoder_call = bool(assert_single_encoder_call)
        self.vision_key_prefixes = tuple(vision_key_prefixes)
        self.width = _policy_width(base)
        self.depth = _policy_depth(base)
        self.mask_2d, self.mask_4d, self.mask_helper_source = _resolve_mask_helpers(base)
        self._validate_action_settings()
        self.projection = self._build_projection(projection, projection_dtype)
        self.chunk_cache = ActionChunkCache(self.protocol)
        self._prefix_builds = 0
        self._encoder_calls = int(getattr(token_source, "calls", 0))
        self._run_start = None
        self.last_run_stats: dict = {}
        self.frozen_parameter_count = 0
        if freeze_backend:
            self.freeze_backend()

    # -- delegation ------------------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError as exc:  # only reachable before __init__ sets base
            raise AttributeError(name) from exc
        return getattr(base, name)

    # -- configuration ---------------------------------------------------- #
    def _validate_action_settings(self) -> None:
        head = getattr(self.base, "action_head", None)
        protocol = self.protocol
        horizon = getattr(head, "action_horizon", None)
        if horizon is not None and int(horizon) != protocol.action_horizon:
            raise ValueError(
                f"action protocol {protocol.name!r} declares horizon={protocol.action_horizon} "
                f"but the backend action head has action_horizon={horizon}"
            )
        action_dim = getattr(head, "action_dim", None)
        if action_dim is not None and int(action_dim) != protocol.action_dim:
            raise ValueError(
                f"action protocol {protocol.name!r} expects internal action_dim="
                f"{protocol.action_dim}, backend has {action_dim}"
            )
        if protocol.control_dim > protocol.action_dim:
            raise ValueError("control_dim cannot exceed the internal action_dim padding")
        max_state = getattr(head, "max_state_dim", None)
        if max_state is not None and int(max_state) != protocol.max_state_dim:
            raise ValueError(
                f"action protocol {protocol.name!r} expects max_state_dim="
                f"{protocol.max_state_dim}, backend has {max_state}"
            )

    def _build_projection(self, projection: Any, projection_dtype: torch.dtype) -> Any:
        if projection is not None:
            weight = getattr(projection, "weight", None)
            if weight is None or weight.ndim != 2:
                raise ValueError("projection must be a linear module with a 2-D weight")
            if int(weight.shape[1]) != self.token_dim or int(weight.shape[0]) != self.width:
                raise ValueError(
                    f"projection maps {tuple(weight.shape)} but the bridge needs "
                    f"[{self.width}, {self.token_dim}]"
                )
            return projection
        if self.token_dim == self.width:
            return None
        return nn.Linear(self.token_dim, self.width, bias=False).to(dtype=projection_dtype)

    def _language_dtype(self) -> torch.dtype:
        """Parameter dtype of the language stack, read exactly as the pin does."""
        language_model = getattr(self.base.vlm, "language_model", None)
        try:
            return language_model.layers[0].self_attn.q_proj.weight.dtype
        except (AttributeError, IndexError, TypeError):
            for param in getattr(self.base.vlm, "parameters", list)():
                return param.dtype
            return torch.float32

    def _max_state_dim(self) -> int:
        return int(getattr(self.base.action_head, "max_state_dim", self.protocol.max_state_dim))

    def _action_horizon(self) -> int:
        return int(getattr(self.base.action_head, "action_horizon", self.protocol.action_horizon))

    def _action_dim(self) -> int:
        return int(getattr(self.base.action_head, "action_dim", self.protocol.action_dim))

    def _device(self) -> torch.device:
        return self.base.action_head.action_in_proj.weight.device

    # -- parameters ------------------------------------------------------- #
    def freeze_backend(self) -> dict:
        """Freeze backend parameters with ``requires_grad_(False)``, never no_grad."""
        frozen = 0
        for param in self.base.parameters():
            param.requires_grad_(False)
            frozen += int(param.numel())
        self.frozen_parameter_count = frozen
        return self.parameter_report()

    def bridge_modules(self) -> list:
        return [self.projection] if self.projection is not None else []

    def trainable_records(self) -> list:
        """Name/count list of every bridge-added trainable parameter."""
        records = _module_records(self.projection, "bridge.projection")
        source_records = getattr(self.token_source, "trainable_records", None)
        if callable(source_records):
            records += list(source_records())
        else:
            records += _module_records(self.token_source, "token_source", trainable_only=True)
        return records

    def parameter_report(self) -> dict:
        added = _module_records(self.projection, "bridge.projection")
        source = getattr(self.token_source, "trainable_records", None)
        source_records = list(source()) if callable(source) else []
        frozen_records = _module_records(self.base, "backend")
        return {
            "bridge_added": {
                "tensors": len(added),
                "parameters": sum(record["parameters"] for record in added),
                "names": [record["name"] for record in added],
                "disclosure": (
                    "learned projection from the OWAC token width into the π0.5 prefix width; "
                    "it is new trainable capacity"
                    if self.projection is not None
                    else "no projection needed: the OWAC token width already equals the π0.5 "
                    "prefix width"
                ),
            },
            "token_source_trainable": {
                "tensors": len(source_records),
                "parameters": sum(record["parameters"] for record in source_records),
                "names": [record["name"] for record in source_records],
                "count_complete": all(record["initialized"] for record in source_records),
                "uninitialized_names": [
                    record["name"] for record in source_records if not record["initialized"]
                ],
            },
            "frozen_backend_parameters": sum(record["parameters"] for record in frozen_records),
            "frozen_backend_tensors": len(frozen_records),
        }

    # -- prefix construction ---------------------------------------------- #
    def _visual_layout(self, tokens: Any) -> dict:
        values = getattr(tokens, "values", None)
        if not torch.is_tensor(values) or values.ndim != 3:
            shape = tuple(values.shape) if torch.is_tensor(values) else type(values).__name__
            raise ValueError(f"VisualTokens.values must be [B, L, D], got {shape}")
        device = values.device
        bsize, length, dim = (int(size) for size in values.shape)
        if dim != self.token_dim:
            raise ValueError(
                f"VisualTokens descriptor_dim={dim} does not match the declared token_dim="
                f"{self.token_dim}"
            )
        camera_ids = tuple(getattr(tokens, "camera_ids", ()) or ())
        if not camera_ids:
            raise ValueError("VisualTokens.camera_ids is empty; one camera id per token group")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError(f"duplicate camera ids: {camera_ids}")
        unknown = [camera for camera in camera_ids if camera not in self.image_keys]
        if unknown:
            raise ValueError(
                f"tokens carry cameras outside image_keys {self.image_keys}: {unknown}"
            )
        if length % len(camera_ids):
            raise ValueError(
                f"VisualTokens length {length} is not a multiple of {len(camera_ids)} cameras"
            )
        per_camera = length // len(camera_ids)
        if per_camera != self.tokens_per_camera:
            raise ValueError(
                f"VisualTokens has {per_camera} tokens per camera but the readout budget is "
                f"{self.tokens_per_camera}"
            )

        notes = []
        raw_mask = getattr(tokens, "mask", None)
        if raw_mask is None:
            mask = torch.ones((bsize, length), dtype=torch.bool, device=device)
            notes.append("VisualTokens.mask absent: all supplied tokens treated as supported")
        else:
            mask = torch.as_tensor(raw_mask, device=device)
            if tuple(mask.shape) != (bsize, length):
                raise ValueError(f"VisualTokens.mask must be [B, L], got {tuple(mask.shape)}")
            mask = mask.to(torch.bool)

        finite_values = torch.isfinite(values).all(dim=-1)
        dropped_values = int((mask & ~finite_values).sum())
        mask = mask & finite_values
        if dropped_values:
            notes.append(f"{dropped_values} token(s) masked for non-finite values")

        positions = None
        dropped_positions = 0
        raw_positions = getattr(tokens, "positions", None)
        if raw_positions is None:
            notes.append("VisualTokens.positions absent: unsupported-position masking disabled")
        else:
            positions = torch.as_tensor(raw_positions, device=device, dtype=torch.float32)
            if positions.ndim == 2:
                positions = positions[:, :, None]
            if positions.ndim != 3 or tuple(positions.shape[:2]) != (bsize, length):
                raise ValueError(
                    f"VisualTokens.positions must be [B, L] or [B, L, P], got "
                    f"{tuple(positions.shape)}"
                )
            position_ok = torch.isfinite(positions).all(dim=-1)
            dropped_positions = int((mask & ~position_ok).sum())
            mask = mask & position_ok
            if dropped_positions:
                notes.append(f"{dropped_positions} token(s) masked for unsupported positions")

        if self.mask_policy == "mask_and_validity":
            validity = getattr(tokens, "validity", None)
            if validity is None:
                notes.append("mask_and_validity requested but VisualTokens.validity is None")
            elif (
                not torch.is_tensor(validity)
                or validity.dtype != torch.bool
                or tuple(validity.shape) != (bsize, length)
            ):
                raise ValueError(
                    "mask_policy='mask_and_validity' requires VisualTokens.validity to be a "
                    "bool tensor of shape [B, L]"
                )
            else:
                mask = mask & validity.to(device)

        if positions is None:
            positions = torch.full((bsize, length, 2), float("nan"), device=device)

        if self.order_within_camera == "row_major":
            # Lexicographic order by the position columns (last column fastest):
            # successive stable sorts from the last column to the first compose into
            # one permutation. Non-finite positions sort last and stay masked.
            order = torch.arange(length, device=device).expand(bsize, -1)
            for column in reversed(range(int(positions.shape[-1]))):
                values_column = positions[:, :, column]
                key = torch.where(
                    torch.isfinite(values_column),
                    values_column,
                    torch.full_like(values_column, float("inf")),
                )
                order = torch.gather(order, 1, torch.argsort(key, dim=1, stable=True))
            values = torch.gather(values, 1, order.unsqueeze(-1).expand_as(values))
            mask = torch.gather(mask, 1, order)
            positions = torch.gather(positions, 1, order.unsqueeze(-1).expand_as(positions))
            notes.append("tokens re-ordered row-major within each camera group by positions")

        return {
            "values": values,
            "mask": mask,
            "positions": positions,
            "camera_ids": camera_ids,
            "per_camera": per_camera,
            "notes": notes,
            "dropped_nonfinite_values": dropped_values,
            "dropped_unsupported_positions": dropped_positions,
        }

    def _project_visual_tokens(self, values: torch.Tensor) -> torch.Tensor:
        projected = values
        if self.projection is not None:
            projected = self.projection(values.to(self.projection.weight.dtype))
        # ``embed_image`` returned the vision module dtype; the replacement visual
        # stream enters the language stack at that same dtype.
        return projected.to(self._language_dtype())

    def embed_prefix_from_visual_tokens(
        self, tokens: Any, lang_tokens: torch.Tensor, lang_masks: torch.Tensor
    ) -> FeaturePrefix:
        """Build the pinned prefix tensors from OWAC tokens instead of images."""
        layout = self._visual_layout(tokens)
        values = layout["values"]
        mask = layout["mask"]
        positions = layout["positions"]
        device = values.device
        bsize = int(values.shape[0])
        per_camera = layout["per_camera"]

        # Zero masked tokens before projection so a bad measurement cannot poison
        # the KV cache with NaN even though the additive mask hides it.
        clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        clean = clean * mask.unsqueeze(-1).to(clean.dtype)

        groups = {camera: index for index, camera in enumerate(layout["camera_ids"])}
        value_parts, mask_parts, position_parts, missing = [], [], [], []
        for camera in self.image_keys:
            if camera in groups:
                start = groups[camera] * per_camera
                stop = start + per_camera
                value_parts.append(clean[:, start:stop])
                mask_parts.append(mask[:, start:stop])
                position_parts.append(positions[:, start:stop])
            else:
                missing.append(camera)
                value_parts.append(torch.zeros((bsize, per_camera, self.token_dim), device=device))
                mask_parts.append(torch.zeros((bsize, per_camera), dtype=torch.bool, device=device))
                position_parts.append(
                    torch.full(
                        (bsize, per_camera, int(positions.shape[-1])), float("nan"), device=device
                    )
                )
        visual_values = torch.cat(value_parts, dim=1)
        visual_mask = torch.cat(mask_parts, dim=1)
        visual_positions = torch.cat(position_parts, dim=1)

        notes = list(layout["notes"])
        if missing:
            notes.append(
                "cameras absent from VisualTokens.camera_ids contributed masked zero tokens: "
                + ", ".join(missing)
            )

        embeds = self._project_visual_tokens(visual_values)
        lang_emb = self.base.vlm.embed_language_tokens(lang_tokens)
        # Pinned convention: language embeddings are scaled by sqrt(hidden).
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        if tuple(lang_masks.shape) != (bsize, int(lang_emb.shape[1])):
            raise ValueError(
                f"lang_masks must be [B, {lang_emb.shape[1]}], got {tuple(lang_masks.shape)}"
            )
        embeds = torch.cat([embeds, lang_emb], dim=1)
        pad_masks = torch.cat([visual_mask, lang_masks.to(device=device, dtype=torch.bool)], dim=1)
        att_masks = torch.zeros(
            (bsize, int(embeds.shape[1])), dtype=torch.bool, device=pad_masks.device
        )
        token_camera = torch.tensor(
            [index for index in range(len(self.image_keys)) for _ in range(per_camera)],
            dtype=torch.long,
            device=device,
        )
        self._prefix_builds += 1
        return FeaturePrefix(
            embeds=embeds,
            pad_masks=pad_masks,
            att_masks=att_masks,
            positions=visual_positions,
            token_camera=token_camera,
            camera_ids=tuple(self.image_keys),
            scale=getattr(tokens, "scale", None),
            time=getattr(tokens, "time", None),
            notes=tuple(notes),
            missing_cameras=tuple(missing),
        )

    # -- tensor preparation ----------------------------------------------- #
    def prepare_control_targets(
        self, actions: Any, *, pad_value: float = 0.0, device: Any = None
    ) -> torch.Tensor:
        """Pad declared control targets ``[B, h, control_dim]`` to ``[B, H, action_dim]``.

        Mirrors the pinned ``_pad_array_2d`` convention (leading slice filled, the
        rest set to ``pad_value``). The real control dimension is never conflated
        with the internal ``action_dim`` padding.
        """
        tensor = torch.as_tensor(actions, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None]
        if tensor.ndim != 3:
            raise ValueError(
                f"control targets must be [B, h, control_dim], got {tuple(tensor.shape)}"
            )
        horizon = self._action_horizon()
        action_dim = self._action_dim()
        if int(tensor.shape[1]) > horizon or int(tensor.shape[2]) > action_dim:
            raise ValueError(
                f"control targets {tuple(tensor.shape)} exceed horizon={horizon}, "
                f"action_dim={action_dim}"
            )
        out = torch.full(
            (int(tensor.shape[0]), horizon, action_dim),
            float(pad_value),
            dtype=torch.float32,
            device=tensor.device,
        )
        out[:, : tensor.shape[1], : tensor.shape[2]] = tensor
        return out.to(device) if device is not None else out

    def control_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Slice the internal action padding back down to the real control dims."""
        if int(actions.shape[-1]) < self.protocol.control_dim:
            raise ValueError(
                f"action width {actions.shape[-1]} is below control_dim={self.protocol.control_dim}"
            )
        return actions[..., : self.protocol.control_dim]

    def _prepare_state(self, state: Any) -> torch.Tensor:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self._device())
        if tensor.ndim == 3 and int(tensor.shape[1]) == 1:
            tensor = tensor.squeeze(1)
        max_state_dim = self._max_state_dim()
        if tensor.ndim != 2 or int(tensor.shape[1]) != max_state_dim:
            raise ValueError(
                f"state must be [B, {max_state_dim}] (the pinned Observation carries "
                f"[B, 1, {max_state_dim}]); got {tuple(tensor.shape)}"
            )
        return tensor

    def _prepare_actions(self, actions: Any) -> torch.Tensor:
        tensor = torch.as_tensor(actions, dtype=torch.float32, device=self._device())
        horizon, action_dim = self._action_horizon(), self._action_dim()
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (horizon, action_dim):
            raise ValueError(
                f"actions must be [B, {horizon}, {action_dim}] including the internal padding; "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    # -- run bookkeeping -------------------------------------------------- #
    def _begin_run(self) -> None:
        self._run_start = (
            self._prefix_builds,
            self._encoder_calls,
            int(getattr(self.token_source, "calls", 0)),
        )

    def _end_run(self, *, expect_encoder_call: bool, denoise_steps: int = 0) -> dict:
        start = self._run_start or (self._prefix_builds, self._encoder_calls, 0)
        prefix_delta = self._prefix_builds - start[0]
        encoder_delta = self._encoder_calls - start[1]
        source_delta = int(getattr(self.token_source, "calls", 0)) - start[2]
        stats = {
            "prefix_builds": prefix_delta,
            "encoder_calls": encoder_delta,
            "token_source_calls": source_delta,
            "denoise_steps": int(denoise_steps),
        }
        self.last_run_stats = stats
        if self.assert_single_encoder_call:
            expected_encoder = 1 if expect_encoder_call else 0
            if prefix_delta != 1 or source_delta != expected_encoder:
                raise RuntimeError(
                    "one observation must run the OWAC encoder exactly once: observed "
                    f"prefix_builds={prefix_delta}, token_source_calls={source_delta} "
                    f"(expected prefix_builds=1, token_source_calls={expected_encoder})"
                )
        return stats

    def visual_tokens(self, observation: Any) -> Any:
        """Run the encoder hook exactly once and return its ``VisualTokens``."""
        counter = getattr(self.token_source, "calls", None)
        if counter is None:
            raise TypeError("token_source must expose a 'calls' counter")
        before = int(counter)
        tokens = self.token_source(observation)
        after = int(getattr(self.token_source, "calls", before))
        if after - before != 1:
            raise RuntimeError(
                "the OWAC encoder must run exactly once per observation: the token source "
                f"reported {after - before} calls; expected 1"
            )
        self._encoder_calls += after - before
        return tokens

    # -- training forward ------------------------------------------------- #
    def _forward_impl(
        self,
        tokens: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Any,
        actions: Any,
        *,
        noise: Any,
        time: Any,
        return_debug: bool,
        expect_encoder_call: bool,
        action_mask: Any = None,
    ) -> dict:
        prefix = self.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)
        state_tensor = self._prepare_state(state)
        action_tensor = self._prepare_actions(actions)
        x_t, u_t, noise, flow_time = self.base.action_head.prepare_flow_targets(
            action_tensor, noise=noise, time=time
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.base.action_head.embed_action_suffix(state_tensor, x_t, flow_time)
        )
        prefix_embs = prefix.embeds
        if self._language_dtype() == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)
        pad_masks = torch.cat([prefix.pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix.att_masks, suffix_att_masks], dim=1)
        attention_mask = self.mask_4d(self.mask_2d(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        suffix_out = self.base.forward_with_action_expert(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
        )
        v_t = self.base.action_head.decode_action_velocity(suffix_out)
        element_loss = torch.nn.functional.mse_loss(u_t, v_t, reduction="none")
        if action_mask is None:
            loss = element_loss.mean()
        else:
            valid = torch.as_tensor(action_mask, device=v_t.device, dtype=torch.bool)
            if tuple(valid.shape) != tuple(v_t.shape[:2]) or not bool(valid.any()):
                raise ValueError(
                    f"action_mask must be nonempty [B,H]={tuple(v_t.shape[:2])}; "
                    f"got shape={tuple(valid.shape)}, valid={int(valid.sum())}"
                )
            loss = (element_loss * valid.unsqueeze(-1)).sum() / (valid.sum() * v_t.shape[-1])
        out = {
            "v_t": v_t,
            "action_loss": loss,
            "prefix": prefix,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if return_debug:
            out.update({"u_t": u_t, "x_t": x_t, "time": flow_time, "noise": noise})
        out["run_stats"] = self._end_run(expect_encoder_call=expect_encoder_call)
        return out

    def forward_from_visual_tokens(
        self,
        tokens: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Any,
        actions: Any,
        *,
        noise: Any = None,
        time: Any = None,
        return_debug: bool = False,
        action_mask: Any = None,
    ) -> dict:
        """Training forward over an OWAC visual prefix (no ``no_grad`` anywhere)."""
        self._begin_run()
        return self._forward_impl(
            tokens,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise=noise,
            time=time,
            return_debug=return_debug,
            expect_encoder_call=False,
            action_mask=action_mask,
        )

    def forward(
        self,
        *,
        observation: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Any,
        actions: Any,
        noise: Any = None,
        time: Any = None,
        return_debug: bool = False,
        action_mask: Any = None,
    ) -> dict:
        """Training forward that runs the encoder hook once for this observation."""
        self._begin_run()
        tokens = self.visual_tokens(observation)
        return self._forward_impl(
            tokens,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise=noise,
            time=time,
            return_debug=return_debug,
            expect_encoder_call=True,
            action_mask=action_mask,
        )

    # -- inference forward ------------------------------------------------ #
    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """One cached denoise step; the prefix comes from ``past_key_values``."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.base.action_head.embed_action_suffix(state, x_t, timestep)
        )
        suffix_len = int(suffix_pad_masks.shape[1])
        bsize, prefix_len = (int(size) for size in prefix_pad_masks.shape)
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = self.mask_2d(suffix_pad_masks, suffix_att_masks)
        attention_mask = self.mask_4d(torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2))
        position_ids = (
            torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        )
        if self._language_dtype() == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)
        return self.base.action_head(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            suffix_embs=suffix_embs,
            adarms_cond=adarms_cond,
        )

    def _sample_impl(
        self,
        tokens: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Any,
        *,
        noise: Any,
        num_steps: Any,
        expect_encoder_call: bool,
    ) -> torch.Tensor:
        num_steps = self.protocol.num_denoise_steps if num_steps is None else int(num_steps)
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        prefix = self.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)
        device = self._device()
        state_tensor = self._prepare_state(state)
        prefix_att_2d_masks_4d = self.mask_4d(self.mask_2d(prefix.pad_masks, prefix.att_masks))
        prefix_position_ids = torch.cumsum(prefix.pad_masks, dim=1) - 1
        self.base.vlm.language_model.config._attn_implementation = "eager"
        prefix_output = self.base.vlm.forward_prefix(
            inputs_embeds=prefix.embeds,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = prefix_output.past_key_values
        bsize = int(state_tensor.shape[0])
        shape = (bsize, self._action_horizon(), self._action_dim())
        if noise is None:
            x_t = self.base.action_head.sample_action_noise(shape, device)
        else:
            x_t = torch.as_tensor(noise, device=device, dtype=torch.float32)
            if x_t.ndim == 2:
                x_t = x_t[None]
        dt = torch.tensor(-1.0 / max(num_steps, 1), dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        steps = 0
        while time >= -dt / 2:
            v_t = self.denoise_step(
                state_tensor, prefix.pad_masks, past_key_values, x_t, time.expand(bsize)
            )
            x_t = x_t + dt * v_t
            time += dt
            steps += 1
            if steps > 100000:
                raise RuntimeError("flow solver did not terminate on a finite step budget")
        stats = self._end_run(expect_encoder_call=expect_encoder_call, denoise_steps=steps)
        stats["num_steps"] = num_steps
        stats["prefix"] = prefix.summary()
        return x_t

    def sample_actions_from_visual_tokens(
        self,
        tokens: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Any,
        *,
        noise: Any = None,
        num_steps: Any = None,
    ) -> torch.Tensor:
        """Flow solver over a visual prefix that is encoded once and cached."""
        self._begin_run()
        return self._sample_impl(
            tokens,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=num_steps,
            expect_encoder_call=False,
        )

    @torch.inference_mode()
    def predict_action_from_visual_tokens(
        self,
        tokens: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Any,
        *,
        noise: Any = None,
        num_steps: Any = None,
    ) -> dict:
        """Inference forward; the only place ``inference_mode`` is used."""
        self._begin_run()
        actions = self._sample_impl(
            tokens,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=num_steps,
            expect_encoder_call=False,
        )
        return self._prediction(actions)

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        observation: Any,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Any,
        noise: Any = None,
        num_steps: Any = None,
    ) -> dict:
        """Inference forward that runs the encoder hook once for this observation."""
        self._begin_run()
        tokens = self.visual_tokens(observation)
        actions = self._sample_impl(
            tokens,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=num_steps,
            expect_encoder_call=True,
        )
        return self._prediction(actions)

    def _prediction(self, actions: torch.Tensor) -> dict:
        control = self.control_actions(actions)
        return {
            "normalized_actions": actions.detach().cpu().numpy(),
            "control_actions": control.detach().cpu().numpy(),
            "protocol": self.protocol.name,
            "control_dim": self.protocol.control_dim,
            "action_dim": self._action_dim(),
            "num_denoise_steps": int(self.last_run_stats.get("num_steps", 0)),
            "run_stats": dict(self.last_run_stats),
        }

    # -- state lifecycle -------------------------------------------------- #
    def reset_episodes(self, batch_mask: Any = None) -> dict:
        """Reset finished episodes: encoder state and action cache together."""
        source_reset = self.token_source.reset(batch_mask)
        cache_slots = self.chunk_cache.reset(batch_mask)
        return {
            "episodes": _as_bool_list(batch_mask),
            "token_source": source_reset,
            "action_cache_slots": cache_slots,
            "action_cache_episodes": self.chunk_cache.episodes,
        }

    def detach_state(self) -> list:
        detach = getattr(self.token_source, "detach", None)
        return list(detach()) if callable(detach) else []

    def state_bytes(self) -> int:
        from owac.core.encoder_contract import state_tensor_bytes

        total = 0
        for state in getattr(self.token_source, "states", {}).values():
            total += state_tensor_bytes(state)
        return total + self.chunk_cache.state_bytes()

    # -- loading ---------------------------------------------------------- #
    def load_audited_replaced_vision(self, state: dict, *, release_vision: bool = True) -> dict:
        """Audited load that explicitly allows the replaced vision weights."""
        return load_audited_replaced_vision(
            self,
            state,
            release_vision=release_vision,
            vision_key_prefixes=self.vision_key_prefixes,
        )

    def release_replaced_vision(self) -> dict:
        return release_replaced_vision(self, vision_key_prefixes=self.vision_key_prefixes)

    def describe(self) -> dict:
        return {
            "backend": type(self.base).__name__,
            "language_width": self.width,
            "language_depth": self.depth,
            "token_dim": self.token_dim,
            "tokens_per_camera": self.tokens_per_camera,
            "image_keys": list(self.image_keys),
            "projection": "none" if self.projection is None else f"{self.width}x{self.token_dim}",
            "mask_policy": self.mask_policy,
            "order_within_camera": self.order_within_camera,
            "mask_helper_source": self.mask_helper_source,
            "language_dtype": str(self._language_dtype()),
            "protocol": self.protocol.as_dict(),
            "prefix_layout": (
                "[0,C*N) per-camera OWAC tokens in image_keys order, then [C*N,C*N+L) "
                "language tokens * sqrt(W); att_masks all False; "
                "position_ids = cumsum(pad_masks, 1) - 1"
            ),
        }


# --------------------------------------------------------------------------- #
# Loading with replaced vision weights
# --------------------------------------------------------------------------- #
class _NonVisionView:
    """Restrict ``load_audited`` to language/action parameters, unmodified.

    ``load_audited`` is reused verbatim: this view only hides the replaced vision
    parameters from ``state_dict()`` so the untouched audit still checks every
    language/action tensor (missing, unexpected, shape mismatch) item by item.
    """

    def __init__(self, base: Any, vision_key_prefixes: Sequence[str]) -> None:
        self._base = base
        self._prefixes = tuple(vision_key_prefixes)
        self._vision_keys = {key for key in base.state_dict() if key.startswith(self._prefixes)}

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_base"), name)

    def state_dict(self) -> dict:
        return {
            key: value
            for key, value in self._base.state_dict().items()
            if key not in self._vision_keys
        }

    def load_state_dict(self, state: dict, strict: bool = False) -> Any:
        current = self.state_dict()
        unexpected = sorted(set(state) - set(current))
        mismatches = sorted(
            key for key in set(state) & set(current) if state[key].shape != current[key].shape
        )
        if mismatches:
            raise ValueError(f"shape mismatch in audited load: {mismatches}")
        self._base.load_state_dict(dict(state), strict=False)
        return SimpleNamespace(
            missing_keys=sorted(set(current) - set(state)), unexpected_keys=unexpected
        )


def _parameter_bytes(module: Any) -> int:
    total = 0
    for param in module.parameters():
        total += int(param.numel()) * int(param.element_size())
    return total


def load_audited_replaced_vision(
    policy: Any,
    state: dict,
    *,
    release_vision: bool = True,
    vision_key_prefixes: Sequence[str] = VISION_KEY_PREFIXES,
) -> dict:
    """Load a checkpoint whose visual tower may be absent or unused.

    Language/action parameters are audited by the untouched
    ``owac.backends.rawvla.load_audited``. Vision parameters are reported and, by
    default, released: ``new_model`` constructs the tower even though this path
    never executes it.
    """
    from owac.backends.rawvla import load_audited

    base = getattr(policy, "base", policy)
    prefixes = tuple(vision_key_prefixes)
    normalized = base._normalize_state_dict(state)
    vision_in_state = sorted(key for key in normalized if key.startswith(prefixes))
    vision_in_state_set = set(vision_in_state)
    language_state = {
        key: value for key, value in normalized.items() if key not in vision_in_state_set
    }
    view = _NonVisionView(base, prefixes)
    vision_keys = sorted(view._vision_keys)
    audit = load_audited(view, language_state)
    result = {
        "loaded_keys": audit["loaded_keys"],
        "model_keys_non_vision": audit["model_keys"],
        "inactive_keys": audit["inactive_keys"],
        "active_missing": audit["active_missing"],
        "unexpected": audit["unexpected"],
        "shape_mismatch": audit["shape_mismatch"],
        "replaced_vision_missing": sorted(set(vision_keys) - set(vision_in_state)),
        "replaced_vision_present": vision_in_state,
        "vision_constructed_at_load": bool(vision_keys),
        "released": None,
    }
    if release_vision:
        result["released"] = release_replaced_vision(policy, vision_key_prefixes=prefixes)
    return result


def release_replaced_vision(
    policy: Any, *, vision_key_prefixes: Sequence[str] = VISION_KEY_PREFIXES
) -> dict:
    """Drop the vision tower and multi-modal projector, reporting freed bytes."""
    base = getattr(policy, "base", policy)
    vlm = getattr(base, "vlm", None)
    core = getattr(getattr(vlm, "model", None), "model", None)
    module_names = []
    for prefix in vision_key_prefixes:
        parts = [part for part in str(prefix).split(".") if part]
        if parts and parts[-1] not in module_names:
            module_names.append(parts[-1])
    freed = 0
    released = []
    for name in module_names:
        holder = core if hasattr(core, name) else vlm
        module = getattr(holder, name, None)
        if module is None:
            continue
        freed += _parameter_bytes(module)
        setattr(holder, name, None)
        released.append(name)
    return {
        "released_modules": released,
        "freed_bytes": int(freed),
        "released": bool(released),
        "note": (
            "new_model constructs the SigLIP tower before the checkpoint is read; it is not "
            "executed on this path, and these parameters are dropped here"
        ),
    }


def build_feature_policy(
    suite: str,
    tokenizer: Path,
    token_source: Any,
    *,
    token_dim: int = 2048,
    tokens_per_camera: int = 256,
    image_keys: Sequence[str] = IMAGE_KEYS,
    projection: Any = None,
    freeze_backend: bool = True,
    assert_single_encoder_call: bool = True,
) -> RawVLAFeaturePolicy:
    """Build a real π0.5 policy wrapped by the feature bridge.

    Reuses ``owac.backends.rawvla.new_model`` unchanged; requires the external
    RawVLA/StarVLA runtime and raises ``MissingAssetError`` when it is absent.
    """
    protocol = protocol_for(suite)
    from owac.backends.rawvla import new_model

    if not Path(tokenizer).exists():
        raise MissingAssetError(
            f"π0.5 tokenizer is absent: {tokenizer}. The feature bridge cannot tokenize a "
            "language prompt without the pinned paligemma tokenizer."
        )
    try:
        base = new_model(suite, Path(tokenizer))
    except ImportError as exc:
        raise MissingAssetError(
            "the external StarVLA/RawVLA runtime is not importable, so the real π0.5 backend "
            f"cannot be constructed: {exc}"
        ) from exc
    return RawVLAFeaturePolicy(
        base,
        token_source,
        protocol=protocol,
        token_dim=token_dim,
        tokens_per_camera=tokens_per_camera,
        image_keys=image_keys,
        projection=projection,
        freeze_backend=freeze_backend,
        assert_single_encoder_call=assert_single_encoder_call,
    )


def check_runtime_assets(
    *, checkpoint: Any = None, tokenizer: Any = None, data_root: Any = None
) -> dict:
    """Report which external assets exist. Never raises for absent assets."""
    import importlib.util

    def exists(path: Any) -> Any:
        return None if path is None else bool(Path(path).exists())

    report = {
        "starvla_importable": importlib.util.find_spec("starVLA") is not None,
        "omegaconf_importable": importlib.util.find_spec("omegaconf") is not None,
        "safetensors_importable": importlib.util.find_spec("safetensors") is not None,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_exists": exists(checkpoint),
        "tokenizer": str(tokenizer) if tokenizer is not None else None,
        "tokenizer_exists": exists(tokenizer),
        "data_root": str(data_root) if data_root is not None else None,
        "data_root_exists": exists(data_root),
    }
    report["real_backend_available"] = bool(
        report["starvla_importable"] and report["omegaconf_importable"]
    )
    return report


def load_checkpoint_state(checkpoint: Any) -> dict:
    """Read a checkpoint state dict, or raise ``MissingAssetError`` explaining why not."""
    path = Path(checkpoint)
    if path.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin", "model.pt"):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
        else:
            raise MissingAssetError(f"no supported checkpoint file under {checkpoint}")
    if not path.exists():
        raise MissingAssetError(f"checkpoint is absent: {path}")
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise MissingAssetError(
                f"safetensors is not importable, so {path} cannot be read: {exc}"
            ) from exc
        return load_file(str(path))
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "module", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


# --------------------------------------------------------------------------- #
# Stub backend (self-check only)
# --------------------------------------------------------------------------- #
class _StubKVCache:
    """Minimal ``past_key_values`` substitute holding per-layer prefix K/V."""

    def __init__(self, keys: list, values: list) -> None:
        self.keys = keys
        self.values = values


class _StubAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)


class _StubLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _StubAttention(width)


def _stub_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Any, heads: int
) -> torch.Tensor:
    bsize, q_len, width = (int(size) for size in query.shape)
    k_len = int(key.shape[1])
    head_dim = width // heads
    q = query.view(bsize, q_len, heads, head_dim).transpose(1, 2)
    k = key.view(bsize, k_len, heads, head_dim).transpose(1, 2)
    v = value.view(bsize, k_len, heads, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask[..., :q_len, :k_len].to(scores.dtype)
    probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    return (probs @ v).transpose(1, 2).reshape(bsize, q_len, width)


class _StubLanguageModel(nn.Module):
    """Structurally faithful miniature of the pinned Gemma language model."""

    def __init__(self, width: int, depth: int, heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_StubLayer(width) for _ in range(depth)])
        self.mlp = nn.Linear(width, width, bias=False)
        self.config = SimpleNamespace(
            hidden_size=width,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            _attn_implementation="eager",
        )
        self.prefix_calls = 0
        self.grad_enabled_during_prefix: list = []
        self.last_prefix_attention_mask: Any = None
        self.last_prefix_position_ids: Any = None

    def forward_prefix(self, **kwargs: Any) -> Any:
        self.prefix_calls += 1
        self.grad_enabled_during_prefix.append(bool(torch.is_grad_enabled()))
        embeds = kwargs["inputs_embeds"]
        self.last_prefix_attention_mask = kwargs.get("attention_mask")
        self.last_prefix_position_ids = kwargs.get("position_ids")
        if kwargs.get("use_cache") is not True:
            raise AssertionError("the pinned inference path requests use_cache=True")
        if kwargs.get("past_key_values") is not None:
            raise AssertionError("the prefix must be encoded with an empty cache")
        keys = [layer.self_attn.k_proj(embeds) for layer in self.layers]
        values = [layer.self_attn.v_proj(embeds) for layer in self.layers]
        return SimpleNamespace(last_hidden_state=embeds, past_key_values=_StubKVCache(keys, values))


class _StubCore(nn.Module):
    def __init__(self, width: int, depth: int, heads: int) -> None:
        super().__init__()
        self.vision_tower = nn.Linear(3, width, bias=False)
        self.multi_modal_projector = nn.Linear(width, width, bias=False)
        self.language_model = _StubLanguageModel(width, depth, heads)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.multi_modal_projector(self.vision_tower(pixel_values))


class _StubContainer(nn.Module):
    def __init__(self, width: int, depth: int, heads: int, vocab: int) -> None:
        super().__init__()
        self.model = _StubCore(width, depth, heads)
        self.lm_head = nn.Linear(width, vocab, bias=False)


class _StubVLM(nn.Module):
    """Stub PaliGemma with the pinned ``vlm.model.model`` nesting and key names.

    ``embed_image`` increments ``image_calls`` and then raises: the feature bridge
    must never reach the replaced image tower, and this stub turns that into a
    hard, observable failure.
    """

    def __init__(self, width: int, depth: int, heads: int, vocab: int) -> None:
        super().__init__()
        self.model = _StubContainer(width, depth, heads, vocab)
        self.embed_tokens = nn.Embedding(vocab, width)
        self.image_calls = 0

    @property
    def config(self) -> Any:
        return SimpleNamespace(text_config=self.model.model.language_model.config)

    @property
    def language_model(self) -> Any:
        return self.model.model.language_model

    @property
    def vision_tower(self) -> Any:
        return self.model.model.vision_tower

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        self.image_calls += 1
        raise AssertionError(
            "the stub has no image tower: the feature bridge must build its prefix from OWAC "
            "tokens instead of calling vlm.embed_image"
        )

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(tokens)

    def forward_prefix(self, **kwargs: Any) -> Any:
        return self.model.model.language_model.forward_prefix(**kwargs)


class _StubActionHead(nn.Module):
    """Stub π0.5 action expert: pinned flow-matching math, trivial attention."""

    def __init__(
        self,
        width: int,
        *,
        action_dim: int,
        action_horizon: int,
        max_state_dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.max_state_dim = int(max_state_dim)
        self.width = int(width)
        self.heads = int(heads)
        self.action_in_proj = nn.Linear(self.action_dim, width)
        self.action_out_proj = nn.Linear(width, self.action_dim)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)
        self.denoise_calls = 0
        self.cache_tokens: list = []
        self.grad_enabled_during_denoise: list = []
        self.last_adarms_cond_shape: Any = None

    def set_gemma_expert_eager_attention(self) -> None:
        return None

    def sample_action_noise(self, shape: Any, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_flow_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        alpha = torch.tensor(1.5, device=device)
        beta = torch.tensor(1.0, device=device)
        return (torch.distributions.Beta(alpha, beta).sample((bsize,)) * 0.999 + 0.001).to(
            torch.float32
        )

    def prepare_flow_targets(self, actions: torch.Tensor, noise: Any = None, time: Any = None):
        actions = actions.to(device=self.action_in_proj.weight.device, dtype=torch.float32)
        if noise is None:
            noise = self.sample_action_noise(actions.shape, actions.device)
        else:
            noise = noise.to(actions.device, torch.float32)
        if time is None:
            time = self.sample_flow_time(actions.shape[0], actions.device)
        else:
            time = time.to(actions.device, torch.float32)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        return x_t, u_t, noise, time

    def embed_action_suffix(self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: Any):
        if int(state.shape[-1]) != self.max_state_dim:
            raise ValueError(f"state must be [B, {self.max_state_dim}], got {tuple(state.shape)}")
        time_emb = self._sinusoidal(timestep).to(self.time_mlp_in.weight.dtype)
        action_time_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        adarms_cond = torch.nn.functional.silu(
            self.time_mlp_out(torch.nn.functional.silu(self.time_mlp_in(time_emb)))
        )
        pad_masks = torch.ones(
            action_time_emb.shape[:2], dtype=torch.bool, device=action_time_emb.device
        )
        att_masks = torch.tensor(
            [True] + [False] * (self.action_horizon - 1),
            dtype=torch.bool,
            device=action_time_emb.device,
        )
        return (
            action_time_emb,
            pad_masks,
            att_masks[None, :].expand(action_time_emb.shape[0], -1),
            adarms_cond,
        )

    def _sinusoidal(self, time: torch.Tensor) -> torch.Tensor:
        dimension = self.width
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=time.device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling = 1.0 / period * 2 * math.pi
        angle = scaling[None, :] * time.to(torch.float32)[:, None]
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)

    def decode_action_velocity(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = action_hidden_states[:, -self.action_horizon :].to(
            self.action_out_proj.weight.dtype
        )
        return self.action_out_proj(hidden).to(torch.float32)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        suffix_embs: torch.Tensor,
        adarms_cond: Any,
        force_bfloat16: bool = False,
    ) -> torch.Tensor:
        if force_bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)
        self.denoise_calls += 1
        self.cache_tokens.append(id(past_key_values))
        self.grad_enabled_during_denoise.append(bool(torch.is_grad_enabled()))
        self.last_adarms_cond_shape = (
            tuple(adarms_cond.shape) if torch.is_tensor(adarms_cond) else None
        )
        if past_key_values is None:
            raise AssertionError("the cached denoise path must receive past_key_values")
        if not past_key_values.keys:
            raise AssertionError("empty prefix cache")
        prefix_k = past_key_values.keys[-1]
        prefix_v = past_key_values.values[-1]
        query = self.time_mlp_in(suffix_embs)
        suffix_k = self.time_mlp_in(suffix_embs)
        suffix_v = self.time_mlp_out(suffix_embs)
        key = torch.cat([prefix_k, suffix_k], dim=1)
        value = torch.cat([prefix_v, suffix_v], dim=1)
        hidden = _stub_attention(query, key, value, attention_mask, self.heads)
        hidden = hidden + suffix_embs
        hidden = hidden + self.time_mlp_out(torch.nn.functional.silu(hidden))
        return self.decode_action_velocity(hidden)


class _StubPolicy(nn.Module):
    """Self-check stub backend — **not** π0.5.

    It mirrors the pinned module nesting, state-dict naming, mask/position
    conventions and flow-matching math the bridge needs, with trivial attention
    math. It exists so ``check-contract`` and the contract test can exercise the
    prefix layout, the single-encoder-call guarantee and the gradient path on CPU
    without StarVLA. Its numbers carry no meaning, and no result from it may be
    reported as a π0.5 acceptance.
    """

    def __init__(
        self,
        *,
        width: int = 32,
        depth: int = 2,
        heads: int = 2,
        vocab: int = 64,
        action_dim: int = 32,
        action_horizon: int = 10,
        max_state_dim: int = 32,
        precision: str = "float32",
    ) -> None:
        super().__init__()
        self.vlm = _StubVLM(width, depth, heads, vocab)
        self.action_head = _StubActionHead(
            width,
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            heads=heads,
        )
        self.train_forward_calls = 0
        self.grad_enabled_during_train: list = []
        self.last_train_attention_mask: Any = None
        self.last_train_position_ids: Any = None
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)

    def forward_with_action_expert(
        self,
        *,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: Any,
    ) -> torch.Tensor:
        self.train_forward_calls += 1
        self.grad_enabled_during_train.append(bool(torch.is_grad_enabled()))
        self.last_train_attention_mask = attention_mask
        self.last_train_position_ids = position_ids
        hidden = torch.cat([prefix_embs, suffix_embs], dim=1)
        layers = self.vlm.language_model.layers
        query = layers[0].self_attn.q_proj(hidden)
        key = layers[0].self_attn.k_proj(hidden)
        value = layers[0].self_attn.v_proj(hidden)
        heads = int(self.vlm.language_model.config.num_attention_heads)
        out = _stub_attention(query, key, value, attention_mask, heads)
        hidden = out + hidden
        hidden = hidden + self.vlm.language_model.mlp(hidden)
        return hidden[:, int(hidden.shape[1]) - int(suffix_embs.shape[1]) :]

    def _normalize_state_dict(self, state_dict: dict) -> dict:
        return dict(state_dict)

    def _is_expected_openpi_missing_key(self, key: str) -> bool:
        return False


class _StubEncoder(nn.Module):
    """Contract-level stub encoder: one learnable linear, per-slot state."""

    def __init__(self, in_channels: int, out_width: int) -> None:
        super().__init__()
        self.stem = nn.Linear(in_channels, out_width, bias=False)
        self.step_calls = 0
        self.reset_masks: list = []

    def init_state(self, batch_size: int, camera_spec: Any, device: Any) -> torch.Tensor:
        return torch.zeros(batch_size, device=device)

    def step(self, frame: Any, context: Any, state: Any, budget: Any) -> Any:
        from owac.core.encoder_contract import CostRecord, EncoderOutput, Validity

        self.step_calls += 1
        values = torch.as_tensor(frame.values, dtype=torch.float32)
        if values.ndim != 4:
            raise ValueError(f"_StubEncoder expects [B, C, H, W], got {tuple(values.shape)}")
        tokens = values.flatten(2).transpose(1, 2)
        valid = None
        if frame.valid_mask is not None:
            valid = torch.as_tensor(frame.valid_mask, device=values.device).flatten(2).all(dim=1)
        return EncoderOutput(
            representation=self.stem(tokens),
            state=state,
            validity=Validity(valid_mask=valid),
            diagnostics={"stub": True},
            cost=CostRecord(updates=int(getattr(budget, "max_updates", 0))),
        )

    def reset(self, state: Any, batch_mask: Any) -> Any:
        self.reset_masks.append(_as_bool_list(batch_mask))
        if state is None:
            return state
        flags = _as_bool_list(batch_mask)
        if not flags:
            return torch.zeros_like(state)
        keep = torch.ones(state.shape[-1], dtype=state.dtype, device=state.device)
        for slot, flag in enumerate(flags[: state.shape[-1]]):
            if flag:
                keep[slot] = 0.0
        return state * keep

    def detach_state(self, state: Any) -> Any:
        return state.detach() if torch.is_tensor(state) else state


class _StubReadout:
    """Contract-level stub readout: one token per spatial position."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, output: Any, token_budget: int) -> Any:
        from owac.core.encoder_contract import VisualTokens

        self.calls += 1
        values = output.representation
        bsize, count = (int(size) for size in values.shape[:2])
        mask = torch.ones((bsize, count), dtype=torch.bool, device=values.device)
        if output.validity is not None and torch.is_tensor(output.validity.valid_mask):
            mask = mask & output.validity.valid_mask.to(values.device, torch.bool)
        index = torch.arange(count, device=values.device, dtype=torch.float32)
        positions = index.view(1, count, 1).expand(bsize, count, 2).contiguous()
        return VisualTokens(values=values, mask=mask, camera_ids=(), positions=positions)


def build_stub_bridge(
    *,
    width: int = 32,
    token_dim: int = 32,
    tokens_per_camera: int = 4,
    channels: int = 4,
    frame_pixels: int = 2,
    protocol: Any = LIBERO,
    precision: str = "float32",
    action_horizon: int = 10,
    cameras: Sequence[str] = ("stub",),
    mask_policy: str = "mask_only",
    order_within_camera: str = "as_given",
    freeze_backend: bool = True,
    with_frame: bool = True,
    batch_size: int = 1,
) -> dict:
    """Build a stub-backed bridge with a real contract encoder/readout chain.

    ``frame_pixels ** 2`` must equal ``tokens_per_camera``: the stub readout emits
    one token per spatial position of the stub measurement frame.
    """
    from owac.core.encoder_contract import MeasurementFrame

    if frame_pixels * frame_pixels != tokens_per_camera:
        raise ValueError(
            "the stub readout emits one token per pixel, so frame_pixels ** 2 must equal "
            "tokens_per_camera"
        )
    policy = _StubPolicy(
        width=width,
        action_horizon=action_horizon,
        precision=precision,
    )
    encoder = _StubEncoder(channels, token_dim)
    readout = _StubReadout()
    source = EncoderReadoutTokenSource(encoder, readout, token_budget=tokens_per_camera)
    bridge = RawVLAFeaturePolicy(
        policy,
        source,
        protocol=protocol,
        token_dim=token_dim,
        tokens_per_camera=tokens_per_camera,
        image_keys=cameras,
        mask_policy=mask_policy,
        order_within_camera=order_within_camera,
        freeze_backend=freeze_backend,
    )
    observation = None
    if with_frame:
        frames = {
            camera: MeasurementFrame(
                values=torch.randn(batch_size, channels, frame_pixels, frame_pixels),
                valid_mask=torch.ones(
                    batch_size, channels, frame_pixels, frame_pixels, dtype=torch.bool
                ),
                camera_id=camera,
            )
            for camera in cameras
        }
        observation = {"frames": frames}
    return {
        "bridge": bridge,
        "policy": policy,
        "encoder": encoder,
        "readout": readout,
        "source": source,
        "observation": observation,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def check_contract() -> dict:
    """CPU contract self-check against the stub backend. Not a π0.5 acceptance."""
    from owac.core.encoder_contract import ComputeBudget, VisualTokens

    torch.manual_seed(0)  # the self-check is a fixed-seed CPU example
    checks: list = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": _jsonable(detail)})

    batch, width, per_camera = 2, 32, 4
    built = build_stub_bridge(width=width, tokens_per_camera=per_camera, batch_size=batch)
    bridge = built["bridge"]
    encoder = built["encoder"]
    readout = built["readout"]
    source = built["source"]

    lang_len = 5
    lang_tokens = torch.arange(lang_len)[None].expand(batch, -1).contiguous()
    lang_masks = torch.ones((batch, lang_len), dtype=torch.bool)

    # --- prefix layout, masks and position numbering ---------------------- #
    mask = torch.ones((batch, per_camera), dtype=torch.bool)
    mask[1, 2] = False
    positions = torch.arange(per_camera).float()[None, :, None].expand(batch, -1, 2).clone()
    positions[0, 3] = float("nan")
    tokens = VisualTokens(
        values=torch.randn(batch, per_camera, width),
        mask=mask,
        camera_ids=("stub",),
        positions=positions,
    )
    prefix = bridge.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)
    expected_mask = torch.tensor([[True, True, True, False], [True, True, False, True]])
    record(
        "prefix_layout",
        tuple(prefix.embeds.shape) == (batch, per_camera + lang_len, width),
        prefix.summary(),
    )
    record(
        "token_level_mask",
        torch.equal(prefix.pad_masks[:, :per_camera], expected_mask),
        {
            "observed": prefix.pad_masks[:, :per_camera].tolist(),
            "expected": expected_mask.tolist(),
            "note": "row 0 token 3 has a non-finite position; row 1 token 2 is masked by mask",
        },
    )
    record(
        "prefix_att_masks_all_false",
        int(prefix.att_masks.sum()) == 0,
        {"att_masks_true": int(prefix.att_masks.sum())},
    )
    record(
        "position_ids_from_pad_masks",
        torch.equal(prefix.position_ids(), torch.cumsum(prefix.pad_masks, dim=1) - 1),
        {
            "first_language_position_id": int(prefix.position_ids()[0, per_camera]),
            "note": "a masked token does not consume a position id",
        },
    )
    lang_expected = bridge.base.vlm.embed_language_tokens(lang_tokens) * math.sqrt(width)
    record(
        "language_scaling_sqrt_hidden",
        torch.allclose(prefix.embeds[:, per_camera:], lang_expected, atol=1e-5),
        {"scale": math.sqrt(width)},
    )
    del tokens

    # --- training forward: real backward, gradient reaches the encoder ---- #
    state = torch.zeros((batch, 32))
    actions = bridge.prepare_control_targets(torch.randn(batch, 10, 7))
    out = bridge.forward(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state,
        actions=actions,
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    out["action_loss"].backward()
    encoder_grad = (
        float(encoder.stem.weight.grad.abs().sum()) if encoder.stem.weight.grad is not None else 0.0
    )
    frozen_with_grad = sorted(
        name
        for name, param in bridge.base.named_parameters()
        if param.grad is not None and float(param.grad.abs().sum()) > 0
    )
    record(
        "gradient_reaches_owac_encoder",
        encoder_grad > 0.0,
        {
            "encoder_stem_grad_abs_sum": encoder_grad,
            "action_loss": float(out["action_loss"].detach()),
        },
    )
    record(
        "frozen_backend_gets_no_gradient",
        not frozen_with_grad,
        {"frozen_parameters_with_grad": frozen_with_grad},
    )
    record(
        "training_forward_does_not_call_image_tower",
        bridge.base.vlm.image_calls == 0,
        {"embed_image_calls": bridge.base.vlm.image_calls},
    )
    record(
        "training_prefix_built_once",
        out["run_stats"]["prefix_builds"] == 1
        and out["run_stats"]["token_source_calls"] == 1
        and encoder.step_calls == 1,
        {**out["run_stats"], "encoder_step_calls": encoder.step_calls},
    )
    record(
        "grad_enabled_during_training",
        all(bridge.base.grad_enabled_during_train),
        {"grad_enabled": bridge.base.grad_enabled_during_train},
    )

    # --- inference: one prefix, ten cached denoise steps ------------------ #
    encoder.stem.weight.grad = None
    steps_before_inference = encoder.step_calls
    inference = bridge.predict_action(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        noise=torch.zeros((batch, 10, 32)),
        num_steps=10,
    )
    stats = inference["run_stats"]
    record(
        "inference_one_prefix_ten_denoise_steps",
        stats["prefix_builds"] == 1
        and stats["denoise_steps"] == 10
        and stats["token_source_calls"] == 1,
        stats,
    )
    record(
        "encoder_ran_once_for_ten_denoise_steps",
        bridge.base.action_head.denoise_calls == 10
        and encoder.step_calls - steps_before_inference == 1,
        {
            "denoise_forward_calls": bridge.base.action_head.denoise_calls,
            "encoder_step_calls_in_this_inference": encoder.step_calls - steps_before_inference,
        },
    )
    record(
        "visual_prefix_encoded_once_from_cache",
        bridge.base.vlm.language_model.prefix_calls == 1
        and len(set(bridge.base.action_head.cache_tokens)) == 1,
        {
            "forward_prefix_calls": bridge.base.vlm.language_model.prefix_calls,
            "distinct_kv_caches": len(set(bridge.base.action_head.cache_tokens)),
        },
    )
    record(
        "inference_mode_only_in_inference_path",
        not any(bridge.base.action_head.grad_enabled_during_denoise),
        {"grad_enabled_during_denoise": bridge.base.action_head.grad_enabled_during_denoise},
    )
    record(
        "control_actions_sliced_to_control_dim",
        tuple(inference["control_actions"].shape) == (batch, 10, 7)
        and tuple(inference["normalized_actions"].shape) == (batch, 10, 32),
        {
            "control_actions": list(inference["control_actions"].shape),
            "normalized_actions": list(inference["normalized_actions"].shape),
        },
    )
    record("readout_called_once_per_camera", readout.calls == 2, {"readout_calls": readout.calls})

    # --- partial episode reset ------------------------------------------- #
    bridge.chunk_cache.store(0, 0, torch.zeros(5, 32))
    bridge.chunk_cache.store(1, 0, torch.zeros(5, 32))
    source.states["stub"] = torch.ones(2)
    reset = bridge.reset_episodes([True, False])
    record(
        "partial_episode_reset",
        reset["action_cache_slots"] == [0]
        and bridge.chunk_cache.episodes == [1]
        and torch.equal(source.states["stub"], torch.tensor([0.0, 1.0])),
        {**reset, "encoder_state": source.states["stub"].tolist()},
    )

    # --- projection disclosure ------------------------------------------- #
    projected = build_stub_bridge(
        width=32, token_dim=16, tokens_per_camera=per_camera, batch_size=batch
    )
    report = projected["bridge"].parameter_report()
    projected_out = projected["bridge"].forward(
        observation=projected["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state,
        actions=actions,
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    projected_out["action_loss"].backward()
    projection_grad = float(projected["bridge"].projection.weight.grad.abs().sum())
    record(
        "projection_is_disclosed_and_trainable",
        report["bridge_added"]["tensors"] == 1
        and report["bridge_added"]["parameters"] == 32 * 16
        and projection_grad > 0.0,
        {"report": report["bridge_added"], "projection_grad_abs_sum": projection_grad},
    )

    # --- loading audit with the vision tower replaced -------------------- #
    full_state = {key: value.clone() for key, value in bridge.base.state_dict().items()}
    expected_vision = sorted(key for key in full_state if key.startswith(VISION_KEY_PREFIXES))
    without_vision = {
        key: value for key, value in full_state.items() if not key.startswith(VISION_KEY_PREFIXES)
    }
    audit = bridge.load_audited_replaced_vision(without_vision, release_vision=False)
    record(
        "audit_allows_absent_vision",
        not audit["active_missing"]
        and not audit["unexpected"]
        and not audit["shape_mismatch"]
        and audit["replaced_vision_missing"] == expected_vision
        and audit["vision_constructed_at_load"],
        {
            "expected_vision_keys": expected_vision,
            "replaced_vision_missing": audit["replaced_vision_missing"],
            "vision_constructed_at_load": audit["vision_constructed_at_load"],
        },
    )
    tampered = {key: value.clone() for key, value in without_vision.items()}
    tampered["vlm.embed_tokens.weight"] = torch.zeros(1)
    rejected = False
    try:
        bridge.load_audited_replaced_vision(tampered, release_vision=False)
    except ValueError:
        rejected = True
    record(
        "audit_rejects_language_shape_mismatch",
        rejected,
        {"note": "a tampered language tensor must not load silently"},
    )
    released = bridge.release_replaced_vision()
    record(
        "vision_released_after_load",
        released["released"] and released["freed_bytes"] > 0,
        released,
    )

    # --- frozen action protocol ------------------------------------------ #
    protocols = {name: protocol.as_dict() for name, protocol in ACTION_PROTOCOLS.items()}
    record(
        "frozen_action_protocols",
        protocols["libero"]["action_horizon"] == 10
        and protocols["libero"]["replan_steps"] == 5
        and protocols["libero"]["control_dim"] == 7
        and protocols["robotwin"]["action_horizon"] == 50
        and protocols["robotwin"]["control_dim"] == 14
        and protocols["robotwin"]["action_dim"] == 32,
        protocols,
    )
    padded = bridge.prepare_control_targets(torch.ones((2, 10, 7)))
    record(
        "control_padding_kept_separate",
        tuple(padded.shape) == (2, 10, 32) and float(padded[:, :, 7:].abs().sum()) == 0.0,
        {
            "padded_shape": list(padded.shape),
            "control_dim": bridge.protocol.control_dim,
            "action_dim": bridge.protocol.action_dim,
        },
    )
    chunk_cache = ActionChunkCache(LIBERO)
    chunk_cache.store(0, 0, torch.zeros(4, 7))
    replan = [chunk_cache.needs_replan(0, step) for step in (0, 4, 5)]
    record(
        "replan_cadence",
        replan == [False, False, True],
        {"needs_replan_at_steps_0_4_5": replan, "replan_steps": chunk_cache.replan_steps},
    )
    mask_helpers = verify_mask_helpers(bridge.base)
    record("mask_helpers", mask_helpers.get("matches_local_copy") is not False, mask_helpers)
    record("budget_is_deterministic", ComputeBudget(max_updates=4).max_updates == 4, {"ok": True})

    report = {
        "check": "contract",
        "note": (
            "CPU contract self-check against the stub backend in this module. This is NOT a "
            "π0.5 acceptance: the stub is a structurally faithful miniature, not the real "
            "backend, and its numbers carry no meaning."
        ),
        "bridge": bridge.describe(),
        "parameters": bridge.parameter_report(),
        "checks": checks,
        "passed": all(check["ok"] for check in checks),
    }
    return report


def _build_real_bridge(args: Any) -> tuple:
    """Construct the real π0.5 policy, load audited weights, wrap it."""
    tokens = _load_or_synthesize_tokens(args)
    source = StaticTokenSource(tokens)
    bridge = build_feature_policy(
        args.suite,
        args.tokenizer,
        source,
        token_dim=int(tokens.values.shape[-1]),
        tokens_per_camera=int(tokens.values.shape[1] // len(tokens.camera_ids)),
    )
    state = load_checkpoint_state(args.checkpoint)
    audit = bridge.load_audited_replaced_vision(state)
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bridge.base.to(device)
    return bridge, {"audit": audit, "device": device, "token_source": source}


def _load_or_synthesize_tokens(args: Any) -> Any:
    from owac.core.encoder_contract import VisualTokens

    if args.tokens_npz:
        import numpy as np

        payload = np.load(args.tokens_npz)
        return VisualTokens(
            values=torch.as_tensor(payload["values"], dtype=torch.float32),
            mask=torch.as_tensor(payload["mask"], dtype=torch.bool),
            camera_ids=tuple(str(name) for name in payload["camera_ids"]),
            positions=torch.as_tensor(payload["positions"], dtype=torch.float32),
        )
    cameras = tuple(args.image_keys)
    count = len(cameras) * int(args.tokens_per_camera)
    return VisualTokens(
        values=torch.randn(int(args.batch_size), count, int(args.token_dim)),
        mask=torch.ones((int(args.batch_size), count), dtype=torch.bool),
        camera_ids=cameras,
        positions=torch.zeros((int(args.batch_size), count, 2), dtype=torch.float32),
    )


def single_batch(args: Any) -> int:
    """Real forward/backward through the real π0.5 backend on one batch."""
    assets = check_runtime_assets(
        checkpoint=args.checkpoint, tokenizer=args.tokenizer, data_root=args.data_root
    )
    if not assets["real_backend_available"] or not assets["tokenizer_exists"]:
        print(
            json.dumps(
                {
                    "check": "single-batch",
                    "status": "unavailable",
                    "reason": (
                        "the external π0.5 runtime is not available here: StarVLA/omegaconf is "
                        "not importable or the pinned tokenizer path is missing. Nothing was run "
                        "and no π0.5 result is claimed."
                    ),
                    "assets": assets,
                },
                indent=2,
            )
        )
        return 2
    try:
        bridge, info = _build_real_bridge(args)
    except MissingAssetError as exc:
        print(
            json.dumps(
                {
                    "check": "single-batch",
                    "status": "unavailable",
                    "reason": str(exc),
                    "assets": assets,
                },
                indent=2,
            )
        )
        return 2
    torch.manual_seed(int(args.seed))
    protocol = bridge.protocol
    batch = int(args.batch_size)
    lang_tokens = torch.zeros((batch, 8), dtype=torch.long)
    lang_masks = torch.ones((batch, 8), dtype=torch.bool)
    state = torch.zeros((batch, bridge._max_state_dim()))
    actions = bridge.prepare_control_targets(
        torch.randn(batch, protocol.action_horizon, protocol.control_dim)
    )
    losses = []
    for _ in range(int(args.steps)):
        bridge.base.zero_grad(set_to_none=True)
        out = bridge.forward_from_visual_tokens(
            info["token_source"].tokens, lang_tokens, lang_masks, state, actions
        )
        out["action_loss"].backward()
        losses.append(float(out["action_loss"].detach()))
    prediction = bridge.predict_action_from_visual_tokens(
        info["token_source"].tokens,
        lang_tokens,
        lang_masks,
        state,
        num_steps=(
            protocol.num_denoise_steps
            if args.num_denoise_steps is None
            else int(args.num_denoise_steps)
        ),
    )
    report = {
        "check": "single-batch",
        "status": "ran",
        "suite": args.suite,
        "checkpoint": str(args.checkpoint),
        "device": info["device"],
        "observation_source": (
            f"pre-computed OWAC tokens from {args.tokens_npz}"
            if args.tokens_npz
            else "SYNTHETIC random tokens (wiring smoke test only, not a measurement)"
        ),
        "load_audit": info["audit"],
        "losses": losses,
        "last_action_loss": losses[-1] if losses else None,
        "run_stats": bridge.last_run_stats,
        "trainable": bridge.trainable_records(),
        "parameter_report": bridge.parameter_report(),
        "control_actions_shape": list(prediction["control_actions"].shape),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "note": (
            "real backend weights and a real forward/backward; the observation is synthetic "
            "unless --tokens-npz was given, so this is not a benchmark result and not a π0.5 "
            "acceptance of any encoder"
        ),
    }
    print(json.dumps(_jsonable(report), indent=2))
    return 0


def _parse_args(argv: Any) -> Any:
    aliases = {
        "--check-contract": "check-contract",
        "--check-assets": "check-assets",
        "--single-batch": "single-batch",
    }
    if argv and argv[0] in aliases:
        argv[0] = aliases[argv[0]]
    parser = argparse.ArgumentParser(
        prog="owac.backends.rawvla_features",
        description=(
            "Feature bridge that feeds OWAC visual tokens into the pinned π0.5 backend. "
            "check-contract is a CPU stub self-check; single-batch needs the external runtime."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser(
        "check-contract", help="CPU contract self-check with the stub backend (not π0.5)"
    )
    contract.add_argument("--output", type=Path, default=None, help="also write the JSON report")

    assets = commands.add_parser("check-assets", help="report which external π0.5 assets exist")
    assets.add_argument("--checkpoint", type=Path, default=None)
    assets.add_argument("--tokenizer", type=Path, default=None)
    assets.add_argument("--data-root", type=Path, default=None)

    single = commands.add_parser(
        "single-batch",
        help="real π0.5 forward/backward; requires the external runtime, weights and tokenizer",
    )
    single.add_argument("--suite", default=LIBERO, choices=sorted(ACTION_PROTOCOLS))
    single.add_argument("--checkpoint", type=Path, required=True)
    single.add_argument("--tokenizer", type=Path, required=True)
    single.add_argument("--data-root", type=Path, default=None)
    single.add_argument(
        "--tokens-npz",
        type=Path,
        default=None,
        help="npz with real OWAC tokens: values, mask, camera_ids, positions",
    )
    single.add_argument("--token-dim", type=int, default=2048)
    single.add_argument("--tokens-per-camera", type=int, default=256)
    single.add_argument(
        "--image-keys", nargs="+", default=list(IMAGE_KEYS), help="camera order of the prefix"
    )
    single.add_argument("--batch-size", type=int, default=1)
    single.add_argument("--steps", type=int, default=1)
    single.add_argument("--num-denoise-steps", type=int, default=None)
    single.add_argument("--device", default=None)
    single.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Any = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "check-contract":
        report = check_contract()
        text = json.dumps(_jsonable(report), indent=2)
        print(text)
        if args.output is not None:
            args.output.write_text(text + "\n")
        return 0 if report["passed"] else 1
    if args.command == "check-assets":
        print(
            json.dumps(
                check_runtime_assets(
                    checkpoint=args.checkpoint, tokenizer=args.tokenizer, data_root=args.data_root
                ),
                indent=2,
            )
        )
        return 0
    return single_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
