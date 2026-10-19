"""OWAC encoder readout -> pinned StarVLA pi0.5 language-width feature entry (task D).

Corresponds to "实际 π0.5 接入位置" in ``docs/prompts/DeepSeek_N01_implementation.md`` against the
frozen StarVLA PI0/PI05 framework (RawVLA ``ba9c56b0``, starVLA ``57d508cb``):

* :class:`VisualTokenPostprocessor`: ``VisualTokens`` ``[B, L, D]`` -> language-width prefix
  (camera / 2-D spatial / optional temporal position encoding, pad+att masks, valid-token scaling).
* :class:`RawVLAFeaturePolicy`: five feature entry points replacing the ``vlm.embed_image`` loop;
  one observation -> one readout, and the flow denoising loop only re-reads its prefix.
* :class:`AuditedVisualReplacementLoader`: load a checkpoint while explicitly replacing (never
  loading) the vision tower; ``owac.backends.rawvla.load_audited`` keeps serving the image baseline.

Boundaries: no RGB/sRGB reconstruction, no encoder call inside the denoising loop, no device
initialization at import time (torch/starVLA are lazy, so importing this module works without them).
"""

from __future__ import annotations

import ast
import gc
import hashlib
import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ACTION_PROTOCOLS",
    "CONTROLLED_CONFIG_NAME",
    "MINI_CONFIG_NAME",
    "PI0_IMAGE_KEYS",
    "PI05_LANGUAGE_WIDTH",
    "ActionChunkCache",
    "ActionProtocol",
    "AuditedVisualReplacementLoader",
    "MissingAsset",
    "RawVLAFeaturePolicy",
    "TokenView",
    "VisualPrefix",
    "VisualPrefixConfig",
    "VisualTokenPostprocessor",
    "MiniVisualTokenPostprocessor",
    "describe_missing_assets",
    "load_action_protocols",
    "load_audited_with_replaced_vision",
    "vision_tower_policy_kwargs",
    "mask_helper_source",
    "parameter_report",
]

_TORCH_REQUIRED = (
    "owac.backends.rawvla_features needs torch for tensor operations. Run it inside the "
    "pinned RawVLA Python 3.10 runtime (uv sync --locked --extra encoder, or the deployed "
    "envs/starvla). Importing this module is still safe without torch."
)

try:  # heavy dependency: imported once when available, never at device-init time
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised by the no-torch import test
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if nn is not None:
    _ModuleBase: Any = nn.Module
else:

    class _ModuleBase:  # type: ignore[no-redef]
        """Stand-in for ``torch.nn.Module`` when torch is not installed.

        Only exists so that this module stays importable (and its signatures inspectable)
        in a torch-free environment.  Instantiating any subclass raises.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(_TORCH_REQUIRED)


def _inference_mode(method):
    """Decorate an inference entry point with ``torch.inference_mode()`` (no-op without torch)."""
    if torch is None:
        return method
    return torch.inference_mode()(method)


# --------------------------------------------------------------------------------------
# Frozen protocol constants (docs/decisions/0012-rawvla-pi05-protocol-freeze.md)
# --------------------------------------------------------------------------------------

PI05_LANGUAGE_WIDTH = 2048
"""pi0.5 language width (PI05.py:31 paligemma_config.width)."""

PI0_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
"""Camera order used by ``PI0.embed_prefix``'s image loop (PI0.py:36)."""

CONTROLLED_CONFIG_NAME = "owac-n01-256x2048"
"""First controlled readout configuration: 256 tokens per valid camera, 2048 wide."""

MINI_CONFIG_NAME = "owac-n01-mini-cpu-test-only"
"""Small CPU configuration; explicitly *not* the deployed backend configuration."""

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROBOTWIN_CLIENT = _REPO_ROOT / "src/owac/runtime/rawvla_robotwin.py"
_FROZEN_PROTOCOL = _REPO_ROOT / "src/owac/runtime/rawvla_full_eval.py"


@dataclass(frozen=True)
class ActionProtocol:
    """Frozen action settings; ``replan_steps`` is read from a client, never inferred.

    ``model_action_dim`` is the internal pi0.5 padding width (PI0DefaultConfig.action_dim=32)
    and stays separate from ``control_dim``.  ``num_denoise_steps`` is a third, independent
    parameter.
    """

    suite: str
    action_horizon: int
    replan_steps: int
    control_dim: int
    state_dim: int
    model_action_dim: int = 32
    num_denoise_steps: int = 10
    provenance: dict[str, Any] = field(default_factory=dict)


ACTION_PROTOCOLS: dict[str, ActionProtocol] = {
    "libero": ActionProtocol(
        suite="libero",
        action_horizon=10,
        replan_steps=5,
        control_dim=7,
        state_dim=8,
        provenance={
            "action_horizon": "frozen protocol record src/owac/runtime/rawvla_full_eval.py",
            "replan_steps": "frozen protocol record src/owac/runtime/rawvla_full_eval.py",
            "note": (
                "The LIBERO runtime client is generated from the external RawVLA source tree "
                "(eval_starvla_openpi_client.py) and is not present in this checkout; the "
                "cadence is taken from the frozen protocol record, not from the horizon."
            ),
        },
    ),
    "robotwin": ActionProtocol(
        suite="robotwin",
        action_horizon=50,
        replan_steps=50,
        control_dim=14,
        state_dim=14,
        provenance={
            "action_horizon": "metadata action_chunk_size in src/owac/runtime/rawvla_robotwin.py",
            "replan_steps": "client source src/owac/runtime/rawvla_robotwin.py:127",
            "note": "RoboTwin is a 14-D absolute joint target with a 50-step executed chunk.",
        },
    ),
}

FROZEN_PROTOCOL_VALUES: dict[str, dict[str, int]] = {
    "libero": {"action_horizon": 10, "replan_steps": 5, "control_dim": 7, "state_dim": 8},
    "robotwin": {"action_horizon": 50, "replan_steps": 50, "control_dim": 14, "state_dim": 14},
}
"""Single source of truth for the frozen numbers; parsed sources and the table must agree."""


def _compare_cadence_node(node: ast.AST) -> int | None:
    """Return the literal in ``step - self.chunk_start >= <literal>`` (rawvla_robotwin.py:127)."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    if not isinstance(node.ops[0], ast.GtE):
        return None
    left = node.left
    if not isinstance(left, ast.BinOp) or not isinstance(left.op, ast.Sub):
        return None
    right = left.right
    if not isinstance(right, ast.Attribute) or right.attr != "chunk_start":
        return None
    value = node.comparators[0]
    if isinstance(value, ast.Constant) and isinstance(value.value, int):
        return int(value.value)
    return None


def _dict_int_literal(tree: ast.AST, key: str) -> list[tuple[int, int, str]]:
    """Find ``{key: <int>}`` entries; return ``(line, value, unparsed)`` triples."""
    found: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for map_key, map_value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(map_key, ast.Constant)
                and map_key.value == key
                and isinstance(map_value, ast.Constant)
                and isinstance(map_value.value, int)
            ):
                found.append((map_value.lineno, int(map_value.value), ast.unparse(map_value)))
    return sorted(found)


def read_replan_cadence(client_path: Path) -> dict[str, Any]:
    """Read the replan cadence out of an existing client source file.

    Matches the exact guard shape of ``ModelClient.step`` (``step - self.chunk_start >= N``)
    and cross-checks ``action_chunk_size`` metadata.  Raises when the anchor disappears, so an
    upstream change cannot silently change the cadence.
    """
    client_path = Path(client_path)
    if not client_path.is_file():
        raise FileNotFoundError(f"client source not found: {client_path}")
    source = client_path.read_text()
    tree = ast.parse(source)
    matches = [
        (node.lineno, value)
        for node in ast.walk(tree)
        if (value := _compare_cadence_node(node)) is not None
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one replan guard in {client_path}, found {matches}")
    line, cadence = matches[0]
    chunk_sizes = _dict_int_literal(tree, "action_chunk_size")
    return {
        "replan_steps": cadence,
        "source": str(client_path),
        "line": line,
        "anchor": f"step - self.chunk_start >= {cadence}",
        "action_chunk_size": chunk_sizes[0][1] if chunk_sizes else None,
        "action_chunk_size_line": chunk_sizes[0][0] if chunk_sizes else None,
    }


def read_frozen_protocol_record(protocol_path: Path) -> dict[str, Any]:
    """Read horizon / replan / dimensions from the frozen protocol record."""
    protocol_path = Path(protocol_path)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"frozen protocol record not found: {protocol_path}")
    tree = ast.parse(protocol_path.read_text())
    out: dict[str, Any] = {"source": str(protocol_path)}
    for key in ("action_horizon", "replan_steps", "action_dim"):
        hits = _dict_int_literal(tree, key)
        if not hits:
            raise ValueError(f"{key} not found in {protocol_path}")
        out[key] = {"value": hits[0][1], "line": hits[0][0], "entries": [h[1] for h in hits]}
    return out


def load_action_protocols(
    client_path: Path | None = None, protocol_path: Path | None = None
) -> dict[str, ActionProtocol]:
    """Build the protocol table from the real client/record and verify it.

    RoboTwin's cadence comes from the only client present in this checkout
    (``src/owac/runtime/rawvla_robotwin.py``); LIBERO's comes from the frozen protocol record.
    Both are checked against :data:`FROZEN_PROTOCOL_VALUES`.
    """
    client_path = Path(client_path) if client_path else _ROBOTWIN_CLIENT
    protocol_path = Path(protocol_path) if protocol_path else _FROZEN_PROTOCOL
    for suite, expected in FROZEN_PROTOCOL_VALUES.items():
        protocol = ACTION_PROTOCOLS[suite]
        for name, value in expected.items():
            actual = getattr(protocol, name)
            if actual != value:
                raise ValueError(
                    f"{suite}.{name} drifted from the frozen table: {actual} != {value}"
                )
    cadence = read_replan_cadence(client_path)
    record = read_frozen_protocol_record(protocol_path)

    robotwin = ACTION_PROTOCOLS["robotwin"]
    if cadence["replan_steps"] != robotwin.replan_steps:
        raise ValueError(f"RoboTwin cadence changed: {cadence}")
    if cadence["action_chunk_size"] not in (None, robotwin.action_horizon):
        raise ValueError(f"RoboTwin chunk metadata contradicts horizon: {cadence}")

    libero = ACTION_PROTOCOLS["libero"]
    if record["replan_steps"]["value"] != libero.replan_steps:
        raise ValueError(f"LIBERO cadence changed: {record['replan_steps']}")
    if record["action_horizon"]["value"] != libero.action_horizon:
        raise ValueError(f"LIBERO horizon changed: {record['action_horizon']}")

    libero_prov = dict(libero.provenance)
    libero_prov.update(
        source=str(protocol_path),
        line=record["replan_steps"]["line"],
        derived_from_horizon=False,
    )
    robotwin_prov = dict(robotwin.provenance)
    robotwin_prov.update(
        source=str(client_path),
        line=cadence["line"],
        anchor=cadence["anchor"],
        derived_from_horizon=False,
    )
    return {
        "libero": ActionProtocol(**{**libero.__dict__, "provenance": libero_prov}),
        "robotwin": ActionProtocol(**{**robotwin.__dict__, "provenance": robotwin_prov}),
    }


class ActionChunkCache:
    """Per-episode action chunk cache mirroring ``ModelClient.step`` (rawvla_robotwin.py:126).

    Replanning happens when ``step - chunk_start >= replan_steps``; the cadence is injected
    (read from the client source), never inferred from the action horizon.
    """

    def __init__(self, replan_steps: int, batch_size: int = 1):
        if replan_steps < 1:
            raise ValueError(f"replan_steps must be >= 1, got {replan_steps}")
        self.replan_steps = int(replan_steps)
        self.batch_size = int(batch_size)
        self.actions: list[Any] = [None] * self.batch_size
        self.chunk_start: list[int] = [0] * self.batch_size

    def should_replan(self, step: int, batch_index: int = 0) -> bool:
        return self.actions[batch_index] is None or step - self.chunk_start[batch_index] >= int(
            self.replan_steps
        )

    def record(self, actions: Any, step: int, batch_index: int = 0) -> None:
        self.actions[batch_index] = actions
        self.chunk_start[batch_index] = int(step)

    def select(self, step: int, batch_index: int = 0) -> Any:
        actions = self.actions[batch_index]
        if actions is None:
            raise RuntimeError("no action chunk cached; call should_replan/record first")
        offset = int(step) - self.chunk_start[batch_index]
        if offset < 0 or offset >= len(actions):
            raise IndexError(f"step {step} is outside the cached chunk (offset {offset})")
        return actions[offset]

    def step(self, step: int, batch_index: int = 0) -> Any:
        """Client-parity accessor for batch size 1 (``ModelClient.step`` semantics)."""
        return self.select(step, batch_index)

    def reset(self, batch_mask: Any = None) -> dict[str, Any]:
        """Clear whole batch or only the episodes selected by ``batch_mask``."""
        cleared = _mask_indices(batch_mask, self.batch_size)
        for index in cleared:
            self.actions[index] = None
            self.chunk_start[index] = 0
        return {
            "slots": len(cleared),
            "indices": cleared,
            "partial": batch_mask is not None,
            "replan_steps": self.replan_steps,
        }


# --------------------------------------------------------------------------------------
# VisualTokens -> language-width prefix
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualPrefixConfig:
    """Configuration of the readout-to-prefix projection.

    ``scale_mode`` records the pi0.5 scale convention: ``embed_prefix`` multiplies only the
    *language* embedding by ``sqrt(dim)`` (PI0.py:392) and feeds ``vlm.embed_image`` output
    unscaled, so the default ``"none"`` keeps the visual prefix unscaled and states the
    normalization that is applied instead (LayerNorm + learned per-camera gain, then invalid
    rows zeroed).

    ``spatial_mode="auto"`` registers both the index branch and the 2-D Fourier branch; which one
    is applied per call is reported in ``VisualPrefix.diagnostics["spatial_branch"]`` (the unused
    branch is still allocated and counted, so pin ``"index"`` or ``"positions"`` to avoid it).
    """

    width: int = PI05_LANGUAGE_WIDTH
    in_dim: int = PI05_LANGUAGE_WIDTH
    tokens_per_camera: int = 256
    camera_order: tuple[str, ...] = PI0_IMAGE_KEYS
    spatial_mode: str = "auto"  # "auto" | "positions" | "index"
    num_fourier_bands: int = 4
    time_mode: str = "none"  # "none" | "learned"
    max_time_slots: int = 1
    normalize: str = "layernorm"  # "layernorm" | "none"
    camera_gain: bool = True
    scale_mode: str = "none"  # "none" | "sqrt_dim"
    name: str = CONTROLLED_CONFIG_NAME

    def __post_init__(self) -> None:
        if self.width < 1 or self.in_dim < 1 or self.tokens_per_camera < 1:
            raise ValueError(f"invalid VisualPrefixConfig sizes: {self}")
        if not self.camera_order:
            raise ValueError("camera_order must list at least one camera key")
        if self.spatial_mode not in {"auto", "positions", "index"}:
            raise ValueError(f"unknown spatial_mode {self.spatial_mode!r}")
        if self.time_mode not in {"none", "learned"}:
            raise ValueError(f"unknown time_mode {self.time_mode!r}")
        if self.normalize not in {"layernorm", "none"}:
            raise ValueError(f"unknown normalize {self.normalize!r}")
        if self.scale_mode not in {"none", "sqrt_dim"}:
            raise ValueError(f"unknown scale_mode {self.scale_mode!r}")
        if self.time_mode == "learned" and self.max_time_slots < 1:
            raise ValueError("learned time embedding needs max_time_slots >= 1")

    @classmethod
    def from_readout_config(
        cls,
        readout_config: Any,
        *,
        camera_order: Sequence[str] | None = None,
        in_dim: int | None = None,
        name: str = CONTROLLED_CONFIG_NAME,
    ) -> VisualPrefixConfig:
        """Build from a readout configuration exposing ``readout_tokens``/``readout_width``.

        Duck-typed on purpose: the N01 configuration module is owned by the framework task, so
        this module never imports it.  ``readout_tokens`` is the per-camera token budget and
        ``readout_width`` the encoder readout width (contract defaults 256 / 2048).
        """
        tokens = getattr(readout_config, "readout_tokens", None)
        width = getattr(readout_config, "readout_width", None)
        if tokens is None or width is None:
            raise ValueError(
                "readout_config must expose readout_tokens and readout_width, got "
                f"{type(readout_config).__name__}"
            )
        return cls(
            width=int(width),
            in_dim=int(in_dim if in_dim is not None else width),
            tokens_per_camera=int(tokens),
            camera_order=tuple(camera_order) if camera_order else PI0_IMAGE_KEYS,
            name=name,
        )


@dataclass
class VisualPrefix:
    """Language-width prefix built from OWAC visual tokens."""

    embs: Any
    pad_masks: Any
    att_masks: Any
    camera_ids: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenView:
    """Normalized view over ``VisualTokens`` (dataclass, tuple or bare tensor)."""

    values: Any
    mask: Any
    camera_ids: tuple[str, ...] = ()
    positions: Any | None = None


def _mask_indices(batch_mask: Any, size: int) -> list[int]:
    """Accept a bool tensor, a bool list or explicit indices; return affected indices."""
    if batch_mask is None:
        return list(range(size))
    if isinstance(batch_mask, (list, tuple)):
        if batch_mask and all(isinstance(item, bool) for item in batch_mask):
            return [index for index, keep in enumerate(batch_mask) if keep]
        return [int(index) for index in batch_mask]
    return [int(i) for i in batch_mask.detach().cpu().reshape(-1).nonzero().flatten().tolist()]


def _unpack_visual_tokens(source: Any) -> TokenView:
    """Accept ``VisualTokens``, ``(values, mask[, camera_ids[, positions]])`` or a tensor."""
    if isinstance(source, TokenView):
        return source
    values = getattr(source, "values", None)
    if values is not None:
        mask = getattr(source, "mask", None)
        camera_ids = tuple(getattr(source, "camera_ids", None) or ())
        positions = getattr(source, "positions", None)
        return TokenView(values=values, mask=mask, camera_ids=camera_ids, positions=positions)
    if isinstance(source, (tuple, list)):
        if not 1 <= len(source) <= 4:
            raise ValueError(
                f"expected (values, mask[, camera_ids[, positions]]), got {len(source)} entries"
            )
        padded = list(source) + [None] * (4 - len(source))
        return TokenView(
            values=padded[0],
            mask=padded[1],
            camera_ids=tuple(padded[2] or ()),
            positions=padded[3],
        )
    return TokenView(values=source, mask=None)


def _fourier_coords(coords: Any, bands: int) -> Any:
    """Deterministic 2-D Fourier features of ``coords`` ``[..., 2]`` in normalized units."""
    freqs = (2.0 ** torch.arange(bands, dtype=coords.dtype, device=coords.device)) * math.pi
    scaled = coords.unsqueeze(-1) * freqs  # [..., 2, bands]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)


class VisualTokenPostprocessor(_ModuleBase):
    """Turn ``VisualTokens`` ``[B, L, D]`` into a pi0.5 language-width prefix.

    Layout follows ``PI0.embed_prefix`` (PI0.py:383-398): one contiguous block per camera in
    ``camera_order``, then the language tokens appended by the caller; every prefix token gets
    ``att_mask = 0`` and a ``pad_mask`` from the token validity.  Blocks for cameras that are
    absent from the input are emitted as zeros with an all-False pad mask, so the prefix length
    is constant and a KV cache stays valid across steps.
    """

    def __init__(self, config: VisualPrefixConfig | None = None) -> None:
        super().__init__()
        if torch is None:  # pragma: no cover - guarded by _ModuleBase too
            raise RuntimeError(_TORCH_REQUIRED)
        self.config = config or VisualPrefixConfig()
        cfg = self.config
        self.width = int(cfg.width)
        self.tokens_per_camera = int(cfg.tokens_per_camera)
        self.camera_order = tuple(cfg.camera_order)
        # LayerNorm right after the projection makes a projection bias redundant (a uniform
        # per-row shift cancels in the normalisation), so it is only allocated when no
        # normalisation follows: no dead parameter is reported as trainable.
        self.in_proj = nn.Linear(int(cfg.in_dim), self.width, bias=cfg.normalize != "layernorm")
        self.norm = nn.LayerNorm(self.width, eps=1e-6) if cfg.normalize == "layernorm" else None
        self.camera_gain = (
            nn.Parameter(torch.ones(len(self.camera_order))) if cfg.camera_gain else None
        )
        self.camera_embed = nn.Embedding(len(self.camera_order), self.width)
        nn.init.normal_(self.camera_embed.weight, std=0.02)
        self.spatial_index = None
        self.spatial_fourier = None
        if cfg.spatial_mode in {"auto", "index"}:
            self.spatial_index = nn.Embedding(self.tokens_per_camera, self.width)
            nn.init.normal_(self.spatial_index.weight, std=0.02)
        if cfg.spatial_mode in {"auto", "positions"}:
            self.spatial_fourier = nn.Linear(4 * int(cfg.num_fourier_bands), self.width)
        self.time_embed = None
        if cfg.time_mode == "learned":
            self.time_embed = nn.Embedding(int(cfg.max_time_slots), self.width)
            nn.init.normal_(self.time_embed.weight, std=0.02)

    # -- introspection ----------------------------------------------------------------

    def parameter_report(self) -> dict[str, Any]:
        return parameter_report(self)

    def describe(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "name": cfg.name,
            "width": cfg.width,
            "in_dim": cfg.in_dim,
            "tokens_per_camera": cfg.tokens_per_camera,
            "camera_order": list(cfg.camera_order),
            "prefix_tokens": cfg.tokens_per_camera * len(cfg.camera_order),
            "spatial_mode": cfg.spatial_mode,
            "time_mode": cfg.time_mode,
            "max_time_slots": cfg.max_time_slots,
            "normalize": cfg.normalize,
            "scale_mode": cfg.scale_mode,
            "parameters": parameter_report(self),
        }

    # -- forward ----------------------------------------------------------------------

    def forward(
        self,
        visual_tokens: Any,
        *,
        time_slots: Any = None,
        camera_order: Sequence[str] | None = None,
    ) -> VisualPrefix:
        if torch is None:  # pragma: no cover
            raise RuntimeError(_TORCH_REQUIRED)
        view = _unpack_visual_tokens(visual_tokens)
        values = view.values
        if not torch.is_tensor(values) or values.ndim != 3:
            raise ValueError(f"visual token values must be a [B, L, D] tensor, got {values!r}")
        batch, length, dim = values.shape
        if dim != int(self.config.in_dim):
            raise ValueError(
                f"visual token width {dim} does not match configured in_dim {self.config.in_dim}"
            )
        order = tuple(camera_order) if camera_order else self.camera_order
        mask = view.mask
        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=values.device)
        else:
            mask = mask.to(device=values.device, dtype=torch.bool)
            if mask.shape != (batch, length):
                raise ValueError(f"mask shape {tuple(mask.shape)} != values {(batch, length)}")
        positions = view.positions
        if positions is not None and positions.shape[:2] != (batch, length):
            raise ValueError(
                f"positions shape {tuple(positions.shape)} incompatible with {(batch, length)}"
            )

        input_cameras = self._resolve_input_cameras(view.camera_ids, order, length)
        time_index = self._resolve_time_slot(time_slots)
        blocks: list[Any] = []
        pad_blocks: list[Any] = []
        valid_counts: dict[str, int] = {}
        synthetic: list[str] = []
        spatial_branch: set[str] = set()
        for camera_index, camera_id in enumerate(order):
            source_index = input_cameras.index(camera_id) if camera_id in input_cameras else None
            if source_index is None:
                synthetic.append(camera_id)
                blocks.append(
                    torch.zeros(
                        (batch, self.tokens_per_camera, self.width),
                        dtype=self.in_proj.weight.dtype,
                        device=values.device,
                    )
                )
                pad_blocks.append(
                    torch.zeros(
                        (batch, self.tokens_per_camera), dtype=torch.bool, device=values.device
                    )
                )
                valid_counts[camera_id] = 0
                continue
            start = source_index * self.tokens_per_camera
            stop = start + self.tokens_per_camera
            block = self._encode_block(
                values[:, start:stop],
                mask[:, start:stop],
                camera_index=camera_index,
                positions=None if positions is None else positions[:, start:stop],
                time_index=time_index,
                spatial_branch=spatial_branch,
            )
            blocks.append(block)
            pad_blocks.append(mask[:, start:stop])
            valid_counts[camera_id] = int(mask[:, start:stop].sum().item())

        embs = torch.cat(blocks, dim=1)
        pad_masks = torch.cat(pad_blocks, dim=1)
        att_masks = torch.zeros(pad_masks.shape[1], dtype=torch.bool, device=pad_masks.device)
        scale = 1.0 if self.config.scale_mode == "none" else math.sqrt(self.width)
        return VisualPrefix(
            embs=embs * scale,
            pad_masks=pad_masks,
            att_masks=att_masks,
            camera_ids=tuple(order),
            diagnostics={
                "config_name": self.config.name,
                "spatial_branch": sorted(spatial_branch),
                "synthetic_camera_blocks": synthetic,
                "valid_tokens_per_camera_batch_sum": valid_counts,
                "valid_tokens_total": int(pad_masks.sum().item()),
                "prefix_tokens": int(pad_masks.shape[1]),
                "language_width": self.width,
                "visual_prefix_scale": self.config.scale_mode,
                "scale_value": scale,
                "time_slot": time_index,
                "dtype": str(embs.dtype),
            },
        )

    # -- internals --------------------------------------------------------------------

    def _resolve_input_cameras(
        self, camera_ids: Sequence[str], order: Sequence[str], length: int
    ) -> list[str]:
        if camera_ids:
            if len(camera_ids) * self.tokens_per_camera != length:
                raise ValueError(
                    f"token count {length} != tokens_per_camera {self.tokens_per_camera} * "
                    f"cameras {len(camera_ids)}"
                )
            unknown = [cid for cid in camera_ids if cid not in order]
            if unknown:
                raise ValueError(f"unknown camera ids {unknown}; configured order is {list(order)}")
            if len(set(camera_ids)) != len(camera_ids):
                raise ValueError(f"duplicate camera ids in visual tokens: {list(camera_ids)}")
            return list(camera_ids)
        if length % self.tokens_per_camera:
            raise ValueError(
                f"token count {length} is not a multiple of tokens_per_camera "
                f"{self.tokens_per_camera}"
            )
        count = length // self.tokens_per_camera
        if count > len(order):
            raise ValueError(f"{count} camera blocks exceed configured cameras {list(order)}")
        return list(order[:count])

    def _resolve_time_slot(self, time_slots: Any) -> int:
        if self.time_embed is None:
            return -1
        slot = 0
        if time_slots is not None:
            slot = int(time_slots) if not torch.is_tensor(time_slots) else int(time_slots.item())
        return max(0, min(slot, int(self.config.max_time_slots) - 1))

    def _encode_block(
        self,
        values: Any,
        mask: Any,
        *,
        camera_index: int,
        positions: Any | None,
        time_index: int,
        spatial_branch: set[str],
    ) -> Any:
        hidden = self.in_proj(values)
        if self.norm is not None:
            hidden = self.norm(hidden)
        if self.camera_gain is not None:
            hidden = hidden * self.camera_gain[camera_index]
        hidden = hidden + self.camera_embed.weight[camera_index]
        if positions is not None and self.spatial_fourier is not None:
            spatial_branch.add("positions")
            coords = _fourier_coords(positions.to(hidden.dtype), int(self.config.num_fourier_bands))
            hidden = hidden + self.spatial_fourier(coords)
        elif self.spatial_index is not None:
            spatial_branch.add("index")
            slots = torch.arange(self.tokens_per_camera, device=values.device)
            hidden = hidden + self.spatial_index(slots)[None, :, :]
        if self.time_embed is not None and time_index >= 0:
            hidden = hidden + self.time_embed.weight[time_index]
        # torch.where (not multiplication) keeps padding slots exactly zero even if the
        # encoder emitted NaN there; invalid rows must never enter the prefix.
        zeros = torch.zeros_like(hidden)
        valid = mask.unsqueeze(-1).expand_as(hidden)
        return torch.where(valid, hidden, zeros)


class MiniVisualTokenPostprocessor(VisualTokenPostprocessor):
    """CPU-test-only postprocessor (4 tokens per camera, width 16).

    Explicitly *not* the deployed configuration (:data:`CONTROLLED_CONFIG_NAME`); the name is
    recorded in diagnostics and in every audit dict that uses it.
    """

    def __init__(self, config: VisualPrefixConfig | None = None) -> None:
        super().__init__(
            config
            or VisualPrefixConfig(
                width=16,
                in_dim=8,
                tokens_per_camera=4,
                camera_order=("base_0_rgb", "left_wrist_0_rgb"),
                spatial_mode="index",
                time_mode="learned",
                max_time_slots=4,
                name=MINI_CONFIG_NAME,
            )
        )


def parameter_report(module: Any) -> dict[str, Any]:
    """Exact learned-parameter accounting for a readout module (required by task D item 1)."""
    named = [(name, param) for name, param in module.named_parameters()]
    return {
        "total": int(sum(param.numel() for _, param in named)),
        "trainable": int(sum(p.numel() for _, p in named if p.requires_grad)),
        "by_name": {name: int(param.numel()) for name, param in named},
    }


# --------------------------------------------------------------------------------------
# Flow-matching helpers reused from the pinned action head
# --------------------------------------------------------------------------------------


def _make_att_2d_masks_local(pad_masks: Any, att_masks: Any) -> Any:
    """Local replica of ``OpenPI_ActionHead.make_att_2d_masks`` (lines 49-55)."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _make_att_4d_masks_local(att_2d_masks: Any) -> Any:
    """Local replica of ``OpenPI_ActionHead.make_att_4d_masks`` (lines 58-59)."""
    return torch.where(att_2d_masks[:, None, :, :], 0.0, -2.3819763e38)


_MASK_HELPERS: tuple[Any, Any, str] | None = None


def _mask_helpers() -> tuple[Any, Any, str]:
    """Return ``(make_att_2d_masks, make_att_4d_masks, source)``.

    Prefers the pinned StarVLA implementation; falls back to the documented local replica when
    starVLA is not importable (CPU tests, library use).  The replica is checked numerically
    against the frozen source in the test suite.
    """
    global _MASK_HELPERS
    if _MASK_HELPERS is None:
        try:
            from starVLA.model.modules.action_model.OpenPI_ActionHead import (  # noqa: PLC0415
                make_att_2d_masks,
                make_att_4d_masks,
            )

            _MASK_HELPERS = (
                make_att_2d_masks,
                make_att_4d_masks,
                "starVLA.model.modules.action_model.OpenPI_ActionHead",
            )
        except ImportError:
            _MASK_HELPERS = (
                _make_att_2d_masks_local,
                _make_att_4d_masks_local,
                "local replica of OpenPI_ActionHead.py:49-59",
            )
    return _MASK_HELPERS


def mask_helper_source() -> str:
    """Which ``make_att_*_masks`` implementation is active (recorded in debug output)."""
    return _mask_helpers()[2]


# --------------------------------------------------------------------------------------
# Feature policy wrapper
# --------------------------------------------------------------------------------------


class RawVLAFeaturePolicy:
    """Feature-entry wrapper around a pinned StarVLA ``PI0``/``PI05`` policy.

    ``policy`` is not modified structurally: this class only calls the interface that
    ``PI0`` already exposes (``vlm.embed_language_tokens``, ``vlm.forward_prefix``,
    ``forward_with_action_expert``, ``denoise_step`` and the action-head methods), replacing
    the ``self.vlm.embed_image(image)`` loop with an OWAC readout prefix.

    One observation runs the encoder once; the flow denoising loop re-reads that prefix.
    """

    def __init__(
        self,
        policy: Any,
        postprocessor: VisualTokenPostprocessor | None = None,
        *,
        encoder: Any = None,
        action_protocol: ActionProtocol | None = None,
        action_cache: ActionChunkCache | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError(_TORCH_REQUIRED)
        self.policy = policy
        self.postprocessor = postprocessor or VisualTokenPostprocessor()
        self.encoder = encoder
        protocol = (
            action_protocol
            or ACTION_PROTOCOLS[
                "libero" if int(getattr(policy, "action_horizon", 10)) == 10 else "robotwin"
            ]
        )
        self.protocol = protocol
        self.action_cache = action_cache or ActionChunkCache(protocol.replan_steps)
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else getattr(policy, "num_inference_steps", protocol.num_denoise_steps)
        )
        self.stats: dict[str, int] = {
            "encoder_calls": 0,
            "prefix_builds": 0,
            "language_embeds": 0,
            "training_forwards": 0,
            "denoise_steps": 0,
        }
        self._readout_state: Any = None
        self._cached_visual_tokens: Any = None
        self._prefix_source: str | None = None
        self._time_slot = 0

    # -- interface helpers -------------------------------------------------------------

    @property
    def action_head(self) -> Any:
        return self.policy.action_head

    @property
    def vlm(self) -> Any:
        return self.policy.vlm

    @property
    def action_horizon(self) -> int:
        return int(getattr(self.policy, "action_horizon", self.protocol.action_horizon))

    @property
    def action_dim(self) -> int:
        return int(getattr(self.policy, "action_dim", self.protocol.model_action_dim))

    @property
    def encoder_calls(self) -> int:
        return self.stats["encoder_calls"]

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0

    def advance_time_slot(self, slot: int | None = None) -> int:
        """Advance the temporal position index used by the readout prefix."""
        self._time_slot = self._time_slot + 1 if slot is None else int(slot)
        return self._time_slot

    def describe(self) -> dict[str, Any]:
        return {
            "protocol": {
                "suite": self.protocol.suite,
                "action_horizon": self.protocol.action_horizon,
                "replan_steps": self.protocol.replan_steps,
                "control_dim": self.protocol.control_dim,
                "model_action_dim": self.protocol.model_action_dim,
                "num_denoise_steps": self.num_inference_steps,
                "provenance": dict(self.protocol.provenance),
            },
            "postprocessor": self.postprocessor.describe(),
            "mask_helpers": mask_helper_source(),
            "stats": dict(self.stats),
        }

    # -- entry point 0: the single counted encoder readout ------------------------------

    def encode_visual_tokens(self, encoder_output: Any, token_budget: Any = None) -> Any:
        """Run the readout of ``self.encoder`` exactly once for one observation."""
        if self.encoder is None:
            raise RuntimeError(
                "no encoder bound: pass encoder=<owac encoder with readout()> to "
                "RawVLAFeaturePolicy before using the encoder-driven entry points"
            )
        if not hasattr(self.encoder, "readout"):
            raise TypeError("encoder must expose readout(output, token_budget) -> VisualTokens")
        tokens = self.encoder.readout(encoder_output, token_budget)
        self.stats["encoder_calls"] += 1
        state = getattr(encoder_output, "state", None)
        if state is not None:
            self._readout_state = state
        self._cached_visual_tokens = tokens
        return tokens

    def visual_tokens_from(self, source: Any, token_budget: Any = None) -> Any:
        """Resolve a ``VisualTokens`` value or an encoder output into readout tokens."""
        if source is None:
            raise ValueError("visual tokens source must not be None")
        if hasattr(source, "values") and hasattr(source, "mask"):
            self._cached_visual_tokens = source
            return source
        if isinstance(source, (tuple, list)) and source and torch.is_tensor(source[0]):
            self._cached_visual_tokens = source
            return source
        if torch.is_tensor(source):
            self._cached_visual_tokens = source
            return source
        return self.encode_visual_tokens(source, token_budget)

    # -- entry point 1: embed_prefix replacement ---------------------------------------

    def embed_prefix_from_visual_tokens(
        self, visual_tokens: Any, lang_tokens: Any, lang_masks: Any
    ) -> tuple[Any, Any, Any]:
        """Replace ``PI0.embed_prefix``'s ``vlm.embed_image`` loop (PI0.py:383-398).

        Keeps the original conventions: one camera block per ``image_keys`` entry, then
        ``lang_emb * sqrt(dim)``, ``pad_masks`` concatenation, all-zero ``att_masks`` for the
        prefix, expanded to ``[B, L]``.
        """
        self._check_language_batch(lang_tokens, lang_masks)
        prefix, lang_emb = self._build_prefix(visual_tokens, lang_tokens)
        return self._concat_prefix(prefix, lang_emb, lang_masks)

    def _check_language_batch(self, lang_tokens: Any, lang_masks: Any) -> None:
        if not torch.is_tensor(lang_tokens) or not torch.is_tensor(lang_masks):
            raise TypeError("lang_tokens and lang_masks must be tensors")
        if lang_tokens.ndim != 2 or lang_masks.shape != lang_tokens.shape:
            raise ValueError(
                f"lang_masks shape {tuple(lang_masks.shape)} != lang_tokens "
                f"{tuple(lang_tokens.shape)}"
            )

    def _concat_prefix(
        self, prefix: VisualPrefix, lang_emb: Any, lang_masks: Any
    ) -> tuple[Any, Any, Any]:
        """Validate widths and concatenate [visual prefix | scaled language] exactly once."""
        land = lang_emb.shape[-1]
        if land != self.postprocessor.width:
            raise ValueError(
                f"language embedding width {land} does not match readout prefix width "
                f"{self.postprocessor.width}"
            )
        if prefix.embs.shape[-1] != land:
            raise ValueError(
                f"visual prefix width {prefix.embs.shape[-1]} != language width {land}"
            )
        if prefix.embs.shape[0] != lang_emb.shape[0]:
            raise ValueError(
                f"batch {prefix.embs.shape[0]} of visual tokens != batch {lang_emb.shape[0]} of "
                "language tokens"
            )
        visual = prefix.embs
        if visual.dtype != lang_emb.dtype:
            # The frozen language side owns the model dtype; a fp32 OWAC readout is cast to it
            # (differentiably) instead of forcing the backend to change precision.
            prefix.diagnostics["dtype_cast"] = f"{visual.dtype} -> {lang_emb.dtype}"
            visual = visual.to(lang_emb.dtype)
        embs = torch.cat([visual, lang_emb * math.sqrt(land)], dim=1)
        pad_masks = torch.cat([prefix.pad_masks, lang_masks.to(torch.bool)], dim=1)
        att_masks = torch.cat(
            [
                prefix.att_masks,
                torch.zeros(lang_emb.shape[1], dtype=torch.bool, device=pad_masks.device),
            ]
        )
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)
        return embs, pad_masks, att_masks

    def _build_prefix(self, visual_tokens: Any, lang_tokens: Any) -> tuple[VisualPrefix, Any]:
        prefix = self.postprocessor(visual_tokens, time_slots=self._time_slot)
        lang_emb = self.vlm.embed_language_tokens(lang_tokens)
        self.stats["prefix_builds"] += 1
        self.stats["language_embeds"] += 1
        self._prefix_source = "visual_tokens"
        if lang_emb.ndim != 3 or lang_emb.shape[1] != lang_tokens.shape[1]:
            raise ValueError(
                f"embed_language_tokens returned {tuple(lang_emb.shape)} for "
                f"{tuple(lang_tokens.shape)}"
            )
        return prefix, lang_emb

    # -- entry point 2: training forward ----------------------------------------------

    def forward_from_observation_with_features(
        self,
        observation: Any,
        visual_tokens: Any,
        actions: Any,
        noise: Any = None,
        time: Any = None,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        """Training forward; mirrors ``PI0.forward_from_observation`` (PI0.py:474-497).

        Reuses ``action_head.prepare_flow_targets`` / ``embed_action_suffix`` and
        ``forward_with_action_expert`` with the original flow-matching objective.  Autograd is
        never disabled here: freezing is done with ``requires_grad_(False)`` so the action loss
        still reaches the OWAC encoder.
        """
        if torch.is_inference_mode_enabled():  # pragma: no cover - defensive
            raise RuntimeError("training forward must not run inside torch.inference_mode()")
        self._check_language_batch(observation.tokenized_prompt, observation.tokenized_prompt_mask)
        prefix, lang_emb = self._build_prefix(visual_tokens, observation.tokenized_prompt)
        out = self._flow_forward(
            prefix,
            lang_emb,
            observation,
            actions,
            noise=noise,
            time=time,
            return_debug=return_debug,
        )
        self.stats["training_forwards"] += 1
        return out

    def _flow_forward(
        self,
        prefix: VisualPrefix,
        lang_emb: Any,
        observation: Any,
        actions: Any,
        *,
        noise: Any,
        time: Any,
        return_debug: bool,
    ) -> dict[str, Any]:
        embs, pad_masks, att_masks = self._concat_prefix(
            prefix, lang_emb, observation.tokenized_prompt_mask
        )
        make_att_2d_masks, make_att_4d_masks, _source = _mask_helpers()
        state = observation.state
        x_t, u_t, noise, time = self.action_head.prepare_flow_targets(
            actions, noise=noise, time=time
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.action_head.embed_action_suffix(state.squeeze(1), x_t, time)
        )
        prefix_embs, suffix_embs = self._cast_like_language_model(embs, suffix_embs)
        full_pad = torch.cat([pad_masks, suffix_pad_masks], dim=1)
        full_att = torch.cat([att_masks, suffix_att_masks], dim=1)
        attention_mask = make_att_4d_masks(make_att_2d_masks(full_pad, full_att))
        position_ids = torch.cumsum(full_pad, dim=1) - 1
        suffix_out = self.policy.forward_with_action_expert(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
        )
        v_t = self.action_head.decode_action_velocity(suffix_out)
        out: dict[str, Any] = {
            "v_t": v_t,
            "action_loss": F.mse_loss(u_t, v_t, reduction="none").mean(),
        }
        if return_debug:
            out.update(
                {
                    "u_t": u_t,
                    "x_t": x_t,
                    "time": time,
                    "noise": noise,
                    "prefix_embs": prefix_embs,
                    "pad_masks": full_pad,
                    "att_masks": full_att,
                    "position_ids": position_ids,
                    "visual_prefix_tokens": int(prefix.pad_masks.shape[1]),
                    "language_tokens": int(lang_emb.shape[1]),
                    "visual_prefix_diagnostics": prefix.diagnostics,
                    "mask_helper_source": _source,
                    "prefix_source": self._prefix_source,
                }
            )
        return out

    def _cast_like_language_model(self, prefix_embs: Any, suffix_embs: Any) -> tuple[Any, Any]:
        """Replicate PI0.py:479-481 (cast to bfloat16 when the language model is bf16)."""
        dtype = None
        try:
            weight = self.vlm.language_model.layers[0].self_attn.q_proj.weight
            dtype = weight.dtype
        except (AttributeError, IndexError, TypeError):
            dtype = None
        if dtype == torch.bfloat16:
            return prefix_embs.to(torch.bfloat16), suffix_embs.to(torch.bfloat16)
        return prefix_embs, suffix_embs

    # -- entry point 3/4: inference ----------------------------------------------------

    @_inference_mode
    def sample_actions_with_features(
        self,
        observation: Any,
        visual_tokens: Any,
        noise: Any = None,
        num_steps: int = 10,
    ) -> Any:
        """Inference sampling; mirrors ``PI0.sample_actions`` (PI0.py:534-560).

        The prefix is built once and cached in ``past_key_values``; the denoising loop calls
        :meth:`denoise_step_with_features`, which cannot reach the image or encoder path.
        """
        make_att_2d_masks, make_att_4d_masks, _source = _mask_helpers()
        prefix, lang_emb = self._build_prefix(visual_tokens, observation.tokenized_prompt)
        embs, pad_masks, att_masks = self._concat_prefix(
            prefix, lang_emb, observation.tokenized_prompt_mask
        )
        prefix_att_2d_masks_4d = make_att_4d_masks(make_att_2d_masks(pad_masks, att_masks))
        prefix_position_ids = torch.cumsum(pad_masks, dim=1) - 1
        self._set_eager_attention()
        prefix_output = self.vlm.forward_prefix(
            inputs_embeds=self._cast_prefix_for_cache(embs),
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = prefix_output.past_key_values
        state = observation.state
        bsize = state.shape[0]
        shape = (bsize, self.action_horizon, self.action_dim)
        if noise is None:
            x_t = self.action_head.sample_action_noise(shape, self._device())
        else:
            x_t = torch.as_tensor(noise, device=self._device(), dtype=torch.float32)
        if x_t.ndim == 2:
            x_t = x_t[None]
        steps = max(int(num_steps), 1)
        dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=self._device())
        time = torch.tensor(1.0, dtype=torch.float32, device=self._device())
        while time >= -dt / 2:
            v_t = self.denoise_step_with_features(
                state.squeeze(1), pad_masks, past_key_values, x_t, time.expand(bsize)
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @_inference_mode
    def predict_action_with_features(
        self,
        observation: Any,
        visual_tokens: Any = None,
        noise: Any = None,
        num_steps: int | None = None,
        token_budget: Any = None,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        """Inference entry point; one encoder readout, then one flow solve.

        ``visual_tokens`` may be a ``VisualTokens`` value (already read out) or an encoder
        output object (``.representation`` / ``.state``), in which case the bound encoder's
        ``readout`` runs exactly once.
        """
        tokens = self.visual_tokens_from(visual_tokens, token_budget)
        steps = int(num_steps if num_steps is not None else self.num_inference_steps)
        calls_before = self.stats["encoder_calls"]
        actions = self.sample_actions_with_features(
            observation, tokens, noise=noise, num_steps=steps
        )
        if self.stats["encoder_calls"] != calls_before:
            raise AssertionError("encoder ran inside the denoising loop")
        out: dict[str, Any] = {
            "normalized_actions": actions.detach().cpu().numpy(),
            "actions": actions,
            "num_denoise_steps": steps,
            "encoder_calls": self.stats["encoder_calls"],
        }
        if return_debug:
            out["stats"] = dict(self.stats)
            out["postprocessor"] = self.postprocessor.describe()
        return out

    def denoise_step_with_features(
        self, state: Any, prefix_pad_masks: Any, past_key_values: Any, x_t: Any, timestep: Any
    ) -> Any:
        """One denoising step; delegates to ``PI0.denoise_step`` (PI0.py:562-577).

        Refuses to run unless the cached prefix came from OWAC visual tokens, so the flow loop
        cannot silently fall back to the original image prefix.
        """
        if self._prefix_source != "visual_tokens":
            raise RuntimeError(
                "denoise_step_with_features requires a prefix built by "
                "embed_prefix_from_visual_tokens/_build_prefix"
            )
        self.stats["denoise_steps"] += 1
        return self.policy.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)

    def _cast_prefix_for_cache(self, prefix_embs: Any) -> Any:
        cast, _ = self._cast_like_language_model(prefix_embs, prefix_embs)
        return cast

    def _set_eager_attention(self) -> None:
        try:
            self.vlm.language_model.config._attn_implementation = "eager"
        except AttributeError:
            pass

    def _device(self) -> Any:
        try:
            return self.action_head.action_in_proj.weight.device
        except AttributeError:  # pragma: no cover - stubs always expose it
            return torch.device("cpu")

    # -- entry point 5: gradient flow ---------------------------------------------------

    def encoder_side_parameters(self) -> list[Any]:
        """Parameters on the OWAC side: the readout postprocessor plus the bound encoder."""
        params = [p for p in self.postprocessor.parameters()]
        if self.encoder is not None and hasattr(self.encoder, "parameters"):
            params.extend(p for p in self.encoder.parameters())
        return params

    def freeze_backend_parameters(self) -> dict[str, Any]:
        """Freeze the wrapped pi0.5 backend; keep OWAC-side parameters trainable.

        Uses ``requires_grad_(False)`` (not ``no_grad``) so the action loss keeps flowing to the
        OWAC encoder during the training forward.
        """
        frozen = 0
        if hasattr(self.policy, "parameters"):
            for param in self.policy.parameters():
                if param.requires_grad:
                    param.requires_grad_(False)
                    frozen += 1
        unfrozen = 0
        for module in (self.postprocessor, self.encoder):
            if module is None or not hasattr(module, "parameters"):
                continue
            for param in module.parameters():
                # Respect caller-owned frozen projections and support ablations.
                unfrozen += int(param.requires_grad)
        return {
            "frozen_backend_tensors": frozen,
            "trainable_encoder_tensors": unfrozen,
            "backend_requires_grad": self._count_backend_requires_grad(),
        }

    def _count_backend_requires_grad(self) -> int:
        if not hasattr(self.policy, "parameters"):
            return 0
        return int(sum(1 for p in self.policy.parameters() if p.requires_grad))

    def gradient_flow_check(
        self,
        observation: Any,
        visual_tokens: Any = None,
        actions: Any = None,
        noise: Any = None,
        time: Any = None,
        freeze_backend: bool = True,
        encoder_output: Any = None,
    ) -> dict[str, Any]:
        """Really check that the action loss reaches the OWAC readout.

        Verifies non-zero gradient (a) at the encoder output tensor, (b) at the language-width
        prefix produced by the postprocessor and (c) on OWAC-side parameters, while every
        frozen language/action parameter has ``grad is None``.  Returns an audit dict.

        Two boundary modes:

        * ``visual_tokens`` given: the readout tensor is detached and re-attached as a leaf, so
          ``d loss / d(readout tokens)`` is measured exactly.  Encoder *parameters* are then
          outside this graph and are reported in ``encoder_params_without_graph``.
        * ``encoder_output`` given (an encoder is bound): the readout runs inside the graph, so
          the bound encoder's own parameters receive gradient as well.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError("gradient_flow_check must not run inside torch.inference_mode()")
        if visual_tokens is None and encoder_output is None:
            raise ValueError("pass visual_tokens or encoder_output")
        freeze_report = (
            self.freeze_backend_parameters()
            if freeze_backend
            else {"frozen_backend_tensors": 0, "skipped": True}
        )
        if encoder_output is not None:
            tokens = self.encode_visual_tokens(encoder_output)
            view = _unpack_visual_tokens(tokens)
            probe = view.values
        else:
            view = _unpack_visual_tokens(visual_tokens)
            probe = view.values.detach().clone().requires_grad_(True)
        probe_view = TokenView(
            values=probe, mask=view.mask, camera_ids=view.camera_ids, positions=view.positions
        )
        prefix, lang_emb = self._build_prefix(probe_view, observation.tokenized_prompt)
        out = self._flow_forward(
            prefix,
            lang_emb,
            observation,
            actions,
            noise=noise,
            time=time,
            return_debug=False,
        )
        loss = out["action_loss"]
        graph_retained = prefix.embs.grad_fn is not None and loss.grad_fn is not None
        token_grad, prefix_grad = torch.autograd.grad(
            loss, [probe, prefix.embs], retain_graph=True, allow_unused=True
        )
        named_encoder_params = [
            (f"postprocessor.{name}", param)
            for name, param in self.postprocessor.named_parameters()
        ]
        if self.encoder is not None and hasattr(self.encoder, "named_parameters"):
            named_encoder_params.extend(
                (f"encoder.{name}", param) for name, param in self.encoder.named_parameters()
            )
        trainable = [(name, param) for name, param in named_encoder_params if param.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable OWAC-side parameters: nothing to check")
        param_grads = torch.autograd.grad(
            loss, [param for _name, param in trainable], retain_graph=False, allow_unused=True
        )
        nonzero = [
            (name, float(grad.norm().item()))
            for (name, _param), grad in zip(trainable, param_grads, strict=True)
            if grad is not None and float(grad.norm().item()) > 0.0
        ]
        without_graph = [
            name
            for (name, _param), grad in zip(trainable, param_grads, strict=True)
            if grad is None
        ]
        frozen_with_grad = []
        if hasattr(self.policy, "named_parameters"):
            frozen_with_grad = [
                name
                for name, param in self.policy.named_parameters()
                if param.grad is not None and float(param.grad.norm().item()) > 0.0
            ]
        return {
            "action_loss": float(loss.detach().item()),
            "action_loss_finite": bool(torch.isfinite(loss).item()),
            "encoder_output_grad_norm": float(token_grad.norm().item())
            if token_grad is not None
            else 0.0,
            "postprocessor_output_grad_norm": float(prefix_grad.norm().item())
            if prefix_grad is not None
            else 0.0,
            "encoder_param_grad_norm": float(
                sum(float(g.norm().item()) for g in param_grads if g is not None)
            ),
            "encoder_params_with_nonzero_grad": len(nonzero),
            "encoder_params_checked": len(trainable),
            "encoder_params_without_graph": without_graph,
            "nonzero_param_names": [name for name, _value in nonzero],
            "frozen_backend_params_with_grad": frozen_with_grad,
            "graph_retained": bool(graph_retained),
            "grad_enabled_during_forward": bool(torch.is_grad_enabled()),
            "inference_mode": bool(torch.is_inference_mode_enabled()),
            "freeze_report": freeze_report,
            "mask_helper_source": mask_helper_source(),
        }

    # -- entry point 6: state lifecycle -------------------------------------------------

    def reset_encoder_state(self, batch_mask: Any = None) -> dict[str, Any]:
        """Reset readout/encoder state and the action chunk cache *together*.

        ``batch_mask`` selects the episodes that ended inside a batch (bool tensor or list);
        ``None`` resets the whole batch.  The readout state and the action cache move together
        so a new episode can never read a prefix cached for the previous one.

        The cached readout tokens are *dropped*, never mutated in place: a readout produced by
        the inference path is a ``torch.inference_mode()`` tensor that PyTorch forbids updating
        outside inference mode, and the policy never reuses a readout across observations.
        """
        report: dict[str, Any] = {
            "partial": batch_mask is not None,
            "encoder_reset": "no_encoder_bound",
            "encoder_calls": self.stats["encoder_calls"],
            "notes": [],
        }
        encoder = self.encoder
        if encoder is not None and hasattr(encoder, "reset") and self._readout_state is not None:
            self._readout_state = encoder.reset(self._readout_state, batch_mask)
            report["encoder_reset"] = "delegated_to_encoder"
        elif encoder is not None:
            report["encoder_reset"] = "encoder_bound_without_state"
        cached = self._cached_visual_tokens
        if cached is not None:
            values = getattr(cached, "values", None)
            tokens = int(values.shape[0]) if values is not None else 0
            report["readout_tokens_dropped"] = tokens
            report["readout_tokens_cleared"] = (
                "all" if batch_mask is None else _mask_indices(batch_mask, tokens)
            )
        self._cached_visual_tokens = None
        self._prefix_source = None
        report["action_cache"] = self.action_cache.reset(batch_mask)
        return report


# --------------------------------------------------------------------------------------
# Audited loading with an explicitly replaced vision tower
# --------------------------------------------------------------------------------------

VISION_KEY_PREFIXES: tuple[str, ...] = (
    "vlm.model.model.vision_tower.",
    "vlm.model.vision_tower.",
    "vlm.vision_tower.",
)
"""Checkpoint prefixes of the original SigLIP tower replaced by the OWAC readout."""

VISION_TOWER_ATTR_PATHS: tuple[tuple[str, ...], ...] = (
    ("vlm", "model", "vision_tower"),
    ("vlm", "model", "model", "vision_tower"),
    ("vlm", "vision_tower"),
)

LANGUAGE_ACTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("language_model", ("vlm.model.model.language_model.", "vlm.model.language_model.")),
    (
        "language_projection",
        ("vlm.model.lm_head.", "vlm.model.model.embed_tokens.", "vlm.lm_head."),
    ),
    ("action_expert", ("action_head.",)),
)


@dataclass(frozen=True)
class MissingAsset:
    """One asset required for real pi0.5 acceptance, with its expected location."""

    path: str
    kind: str
    required_by: str
    present: bool


def describe_missing_assets(root: str = "/workspace/owac/rawvla") -> list[MissingAsset]:
    """Precise list of what real pi0.5 feature acceptance still needs (checked on disk)."""
    base = Path(root)
    libero_weights = base / (
        "source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors"
    )
    robotwin = base / "checkpoints/pi05_robotwin"
    entries = [
        MissingAsset(root, "runtime_root", "RawVLA deployment root", base.is_dir()),
        MissingAsset(
            f"{root}/source/starVLA",
            "source_tree",
            "importable StarVLA @57d508cb (PI0/PI05/OpenPI_ActionHead)",
            (base / "source/starVLA").is_dir(),
        ),
        MissingAsset(
            f"{root}/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors",
            "weights",
            "LIBERO pi0.5 policy weights",
            libero_weights.is_file(),
        ),
        MissingAsset(
            f"{root}/checkpoints/pi05_robotwin/model.safetensors",
            "weights",
            "RoboTwin pi0.5 policy weights",
            (robotwin / "model.safetensors").is_file(),
        ),
        MissingAsset(
            f"{root}/checkpoints/pi05_robotwin/policy_norm_stats.json",
            "statistics",
            "RoboTwin MEAN_STD normalisation",
            (robotwin / "policy_norm_stats.json").is_file(),
        ),
        MissingAsset(
            f"{root}/source/assets/physical-intelligence/libero/norm_stats.json",
            "statistics",
            "LIBERO quantile statistics (owac/backends/rawvla.py:152)",
            (base / "source/assets/physical-intelligence/libero/norm_stats.json").is_file(),
        ),
        MissingAsset(
            f"{root}/source/paligemma_tokenizer.model",
            "tokenizer",
            "LazyPaliGemmaTokenizer.model_path; passed via --tokenizer "
            "(openpi convention: <checkpoint>/assets/paligemma_tokenizer.model)",
            (base / "source/paligemma_tokenizer.model").is_file(),
        ),
        MissingAsset(
            f"{root}/envs/starvla/bin/python",
            "environment",
            "Python 3.10 StarVLA runtime (torch/transformers/sentencepiece/safetensors)",
            (base / "envs/starvla/bin/python").is_file(),
        ),
        MissingAsset(
            f"{root}/envs/robotwin/bin/python",
            "environment",
            "Python 3.10 RoboTwin runtime (sapien + starVLA)",
            (base / "envs/robotwin/bin/python").is_file(),
        ),
    ]
    return entries


def _vision_param_bytes(model: Any) -> int:
    if not hasattr(model, "named_parameters"):
        return 0
    total = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in VISION_KEY_PREFIXES):
            total += int(param.numel()) * int(param.element_size())
    return total


def _release_vision_tower(
    model: Any, attr_paths: Sequence[Sequence[str]], *, free: bool = True
) -> tuple[dict[str, Any], Any]:
    """Detach the constructed vision tower and report what was released.

    The tower is always detached from the module tree (the unchanged baseline audit in
    ``load_audited`` must see a vision-free model); ``free`` additionally drops the storage.
    With ``free=False`` the detached module is returned to the caller instead of collected.
    """
    report: dict[str, Any] = {
        "vision_tower_detected": False,
        "vision_param_bytes_before": _vision_param_bytes(model),
        "release_method": None,
        "verified_removed": False,
        "cuda_cache_cleared": False,
        "detached_reference_kept": False,
    }
    detached: Any = None
    for path in attr_paths:
        target = model
        for part in path[:-1]:
            target = getattr(target, part, None)
            if target is None:
                break
        if target is None or not hasattr(target, path[-1]):
            continue
        tower = getattr(target, path[-1], None)
        if tower is None:
            continue
        report["vision_tower_detected"] = True
        report["release_path"] = ".".join(path)
        try:
            nn.Module.__setattr__(target, path[-1], None)
        except (AttributeError, TypeError) as exc:  # pragma: no cover - non-module containers
            report["release_method"] = f"failed: {exc}"
            continue
        detached = tower
        if free:
            report["release_method"] = "nn.Module.__setattr__(target, 'vision_tower', None) + gc"
            del tower
            detached = None
            gc.collect()
            if torch is not None and torch.cuda.is_initialized():  # never initialise CUDA here
                torch.cuda.empty_cache()
                report["cuda_cache_cleared"] = True
        else:
            report["release_method"] = (
                "nn.Module.__setattr__(target, 'vision_tower', None); storage kept by caller"
            )
            report["detached_reference_kept"] = True
        break
    report["vision_param_bytes_after"] = _vision_param_bytes(model)
    report["bytes_freed"] = report["vision_param_bytes_before"] - report["vision_param_bytes_after"]
    report["verified_removed"] = report["vision_param_bytes_after"] == 0
    return report, detached


class AuditedVisualReplacementLoader:
    """Audited checkpoint loading that *replaces* the vision tower instead of loading it.

    ``owac.backends.rawvla.load_audited`` keeps serving the complete image baseline; this loader
    filters the checkpoint's vision keys out first and then delegates the remaining item-by-item
    audit (tied lm_head, expected-inactive adaRMS scale, unexpected keys, shape mismatches,
    parameter count) to that unchanged baseline function.

    ``classify`` runs the model's own ``_normalize_state_dict`` and the delegated baseline runs it
    once more on the filtered dict; that normalisation only renames keys and is idempotent for the
    pinned prefixes (``paligemma_with_expert.*`` -> ``vlm.*``).
    """

    def __init__(
        self,
        *,
        vision_prefixes: Sequence[str] = VISION_KEY_PREFIXES,
        groups: Sequence[tuple[str, Sequence[str]]] = LANGUAGE_ACTION_GROUPS,
        release_vision_tower: bool = True,
        vision_tower_attr_paths: Sequence[Sequence[str]] = VISION_TOWER_ATTR_PATHS,
    ) -> None:
        self.vision_prefixes = tuple(vision_prefixes)
        self.groups = tuple((name, tuple(prefixes)) for name, prefixes in groups)
        self.release_vision_tower = bool(release_vision_tower)
        self.vision_tower_attr_paths = tuple(tuple(p) for p in vision_tower_attr_paths)

    def classify(self, model: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Split a checkpoint state dict into replaced-vision and loadable parts."""
        normalizer = getattr(model, "_normalize_state_dict", None)
        normalized = normalizer(state) if callable(normalizer) else dict(state)
        vision_keys = sorted(
            key
            for key in normalized
            if any(key.startswith(prefix) for prefix in self.vision_prefixes)
        )
        return {
            "normalized": normalized,
            "vision_keys": vision_keys,
            "loadable": {k: v for k, v in normalized.items() if k not in set(vision_keys)},
        }

    def group_report(self, model: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Item-by-item check of the language/action parameter groups kept from the checkpoint."""
        current = model.state_dict()
        report: dict[str, Any] = {}
        for name, prefixes in self.groups:
            model_keys = sorted(key for key in current if any(key.startswith(p) for p in prefixes))
            missing = [key for key in model_keys if key not in state]
            mismatched = [
                key
                for key in model_keys
                if key in state and tuple(state[key].shape) != tuple(current[key].shape)
            ]
            report[name] = {
                "prefixes": list(prefixes),
                "model_tensors": len(model_keys),
                "checked_tensors": len(model_keys) - len(missing),
                "missing": missing,
                "shape_mismatch": mismatched,
                "parameters": int(
                    sum(
                        current[key].numel() for key in model_keys if hasattr(current[key], "numel")
                    )
                ),
            }
        return report

    def load(self, model: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Audit, load the non-vision weights, then release the replaced vision tower."""
        from owac.backends.rawvla import load_audited

        classified = self.classify(model, state)
        groups = self.group_report(model, classified["normalized"])
        broken = {
            name: {"missing": info["missing"], "shape_mismatch": info["shape_mismatch"]}
            for name, info in groups.items()
            if info["missing"] or info["shape_mismatch"]
        }
        if broken:
            raise ValueError(
                f"checkpoint does not match the pinned language/action parameters: {broken}"
            )
        vision_digest = hashlib.sha256("\n".join(classified["vision_keys"]).encode()).hexdigest()
        # The replaced tower must leave the module tree before the unchanged baseline audit
        # runs, otherwise its (intentionally unloaded) weights look like missing parameters.
        release, detached = _release_vision_tower(
            model, self.vision_tower_attr_paths, free=self.release_vision_tower
        )
        audit = load_audited(model, classified["loadable"])
        if detached is not None:
            audit["detached_vision_tower"] = detached
        audit.update(
            replaced_vision_keys=len(classified["vision_keys"]),
            replaced_vision_key_sha256=vision_digest,
            replaced_vision_key_sample=classified["vision_keys"][:10],
            parameter_groups=groups,
            vision_release=release,
            notes=[
                "vision tower weights from the checkpoint are intentionally not loaded; the OWAC "
                "readout replaces the pi0.5 vision prefix",
                "language/action parameters were checked item by item before loading",
            ],
        )
        return audit


def vision_tower_policy_kwargs(builder: Any) -> dict[str, Any]:
    """Return ``{"build_vision_tower": False}`` when ``builder`` accepts the patch flag.

    ``third_party/patches/rawvla_features/vision_tower_optional.patch`` lets the pinned
    ``OpenPIPaliGemma`` skip constructing the replaced SigLIP tower.  Without the patch this probe
    returns ``{}`` and the caller falls back to construct-then-release
    (:class:`AuditedVisualReplacementLoader`); nothing is assumed about the unpatched API.
    """
    try:
        parameters = inspect.signature(builder).parameters
    except (TypeError, ValueError):  # pragma: no cover - non-callables / builtins
        return {}
    return {"build_vision_tower": False} if "build_vision_tower" in parameters else {}


def load_audited_with_replaced_vision(
    model: Any,
    state: dict[str, Any],
    *,
    release_vision_tower: bool = True,
    loader: AuditedVisualReplacementLoader | None = None,
) -> dict[str, Any]:
    """Function form of :class:`AuditedVisualReplacementLoader` (``load_audited`` untouched)."""
    active = loader or AuditedVisualReplacementLoader(release_vision_tower=release_vision_tower)
    return active.load(model, state)
