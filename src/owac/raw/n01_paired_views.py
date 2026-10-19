"""Same-condition RAW -> frozen-ISP RGB pairing for the v1.2 four-arm protocol.

The v1.2 route needs an RGB arm that shares the *physical state, camera,
exposure/noise condition, time and action label* of the RAW arm.  The only
legitimate way to get it is to read the real RAW measurement ``X`` of a source
window and render ``Y = ISP_fixed(X)`` with the author's frozen ISP.  This module
owns that pairing; it never reads the author's paired-RGB cache, which may be a
normal-light reference captured under different conditions.

Two entry points are kept explicitly apart:

``RAW_VIEW_DOMAIN = "rawvla_rgb3"``
    The legal three-channel HDR-derived / pseudo-RAW measurement entry.  It is
    *not* a native Bayer CFA, and it is not called "raw sensor counts".

``RENDERED_RGB_DOMAIN = "rendered_rgb3"``
    An RGB tensor produced by this module from that RAW view.  Marking it as
    ``rawvla_rgb3`` would let the RGB arm claim RAW measurement support.

Frozen upstream sources (read, not guessed; see ``isp_provenance``):

* LIBERO, fixed ``57d508cb180d0510ac9060358009c0f6ffaac420`` of RawVLA:
  ``starVLA/examples/simBenchmarks/LIBERO/eval_files/raw_rgb10_unprocessing.py``
  (``FIXED_RAW_RGB10_PARAMS``, ``_raw10_to_rgb8``) and the call site
  ``experiments/simulation/libero/rawvla_bench/raw.py::make_rawvla_observation``.
* RoboTwin, same RawVLA checkout:
  ``third_party/FastWAM/third_party/RoboTwin/envs/camera/raw_rgb10.py``
  (``raw10_to_rgb8``, identical arithmetic).

Both platforms store the RAW view as ``rint(raw10 * 255 / 1023)`` and the policy
loader divides by 255 only for the uint8 dtype, so a cached float in ``[0, 1]``
is ``raw10 / 1023``.  RoboTwin's cache is float32 (no extra quantisation);
LIBERO's is uint8, so ~2 bits of the 10-bit measurement are already gone before
any P3 model sees it.  That loss is a property of the cache, is recorded in
``provenance`` and must be reported, not repaired by inventing information.

The ISP arithmetic is written twice on purpose: :func:`raw_unit_to_srgb_u8` is a
NumPy transcription of the upstream function used as an oracle, and
:func:`raw_unit_to_srgb` is the torch path used in training/service.  A test
asserts they agree, so the training path cannot silently drift from the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = [
    "RAW_VIEW_DOMAIN",
    "RENDERED_RGB_DOMAIN",
    "RawRGB10Params",
    "FIXED_RAW_RGB10_PARAMS",
    "isp_provenance",
    "raw10_to_view8",
    "raw_unit_to_srgb_u8",
    "raw_unit_to_srgb",
    "paired_views",
    "PairedViews",
]


RAW_VIEW_DOMAIN = "rawvla_rgb3"
RENDERED_RGB_DOMAIN = "rendered_rgb3"

RAWVLA_SOURCE_COMMIT = "57d508cb180d0510ac9060358009c0f6ffaac420"


@dataclass(frozen=True)
class RawRGB10Params:
    """Fixed per-channel gains of the author pseudo-RAW chain."""

    red_gain: float
    green_gain: float
    blue_gain: float
    inverse_global_gain: float

    def gains(self) -> tuple[float, float, float]:
        return (self.red_gain, self.green_gain, self.blue_gain)


FIXED_RAW_RGB10_PARAMS = RawRGB10Params(
    red_gain=1.8,
    green_gain=1.0,
    blue_gain=1.7,
    inverse_global_gain=1.0,
)


_LIBERO_MODULE = "starVLA/examples/simBenchmarks/LIBERO/eval_files/raw_rgb10_unprocessing.py"
_ROBOTWIN_MODULE = "third_party/FastWAM/third_party/RoboTwin/envs/camera/raw_rgb10.py"


def isp_provenance(platform: str) -> dict[str, Any]:
    """Frozen author ISP; complete service uses native RAW before shared resize.

    Historical legacy clients resized before ISP. Complete models reject those
    frames by their configured native shape and use the native transport patch.
    """

    key = str(platform).lower()
    if key not in {"libero", "robotwin", "robotwin2"}:
        raise ValueError(f"unknown platform {platform!r}; expected 'libero' or 'robotwin'")
    robotwin = key.startswith("robotwin")
    return {
        "platform": "robotwin2" if robotwin else "libero",
        "rawvla_commit": RAWVLA_SOURCE_COMMIT,
        "module": _ROBOTWIN_MODULE if robotwin else _LIBERO_MODULE,
        "symbol": "raw10_to_rgb8" if robotwin else "_raw10_to_rgb8",
        "params": {
            "red_gain": FIXED_RAW_RGB10_PARAMS.red_gain,
            "green_gain": FIXED_RAW_RGB10_PARAMS.green_gain,
            "blue_gain": FIXED_RAW_RGB10_PARAMS.blue_gain,
            "inverse_global_gain": FIXED_RAW_RGB10_PARAMS.inverse_global_gain,
        },
        "tone_curve": "smoothstep(linear_to_srgb(clip(raw10/1023 * gains, 0, 1)))",
        "cache_dtype": "float32" if robotwin else "uint8",
        "raw_domain_tag": RAW_VIEW_DOMAIN,
        "rendered_domain_tag": RENDERED_RGB_DOMAIN,
        "raw_unit": (
            "linear_pseudo_raw_view_over_255" if not robotwin else "linear_pseudo_raw_float32"
        ),
        "quantisation": (
            "float32 cache keeps raw10/1023 at full precision"
            if robotwin
            else "uint8 cache stores rint(raw10*255/1023): the ISP input is quantised to 1/255 "
            "before any P3 model sees it, so ~2 bits of the 10-bit measurement are already lost"
        ),
        "resize_crop_order": (
            "RAW cache resolution -> frozen ISP on the full frame -> policy resize/crop"
        ),
        "serving_order_note": (
            "Complete v1.2 configurations require native transport and apply this ISP "
            "inside the shared observation operator before optional full-field resize. "
            "Historical legacy clients resized RAW before ISP; those receipts remain diagnostics."
        ),
        "not_a_cfa": True,
        "note": (
            "author three-channel HDR-derived / pseudo-RAW; not a native Bayer CFA and not "
            "the ColorChecker ISP in finetune/calibration/default_isp.py"
        ),
    }


def raw10_to_view8(raw10: np.ndarray) -> np.ndarray:
    """Upstream ``raw10_to_view8``: the uint8 view the LIBERO cache stores."""

    return np.rint(np.asarray(raw10, dtype=np.float32) * (255.0 / 1023.0)).astype(np.uint8)


def raw_unit_to_srgb_u8(
    raw_unit: np.ndarray, params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS
) -> np.ndarray:
    """NumPy oracle: ``[0, 1]`` linear pseudo-RAW view -> uint8 sRGB-like display."""

    raw = np.asarray(raw_unit)
    if raw.ndim < 3 or raw.shape[-1] != 3:
        raise ValueError(f"expected HWC three-channel input, got shape={raw.shape}")
    gains = np.asarray(params.gains(), dtype=np.float32)
    linear = np.clip(raw.astype(np.float32) * gains / params.inverse_global_gain, 0.0, 1.0)
    display = _smoothstep_np(_linear_to_srgb_np(linear))
    return np.rint(display * 255.0).astype(np.uint8)


def _linear_to_srgb_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _smoothstep_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def raw_unit_to_srgb(
    values: torch.Tensor, params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS
) -> torch.Tensor:
    """Torch path: ``[..., 3, H, W]`` or ``[..., H, W, 3]`` linear pseudo-RAW -> ``[0, 1]`` RGB.

    Channel-last input is accepted and returned channel-last so previews and
    numpy oracles can share the arithmetic; the training path is channel-first.
    """

    if not torch.is_tensor(values):
        raise TypeError(f"expected a torch tensor, got {type(values).__name__}")
    channel_last = values.shape[-1] == 3 and values.shape[-3] != 3
    raw = values if channel_last else values.movedim(-3, -1)
    if raw.shape[-1] != 3:
        raise ValueError(f"expected a three-channel input, got shape={tuple(values.shape)}")
    gains = torch.as_tensor(params.gains(), dtype=raw.dtype, device=raw.device)
    linear = torch.clamp(raw * gains / params.inverse_global_gain, 0.0, 1.0)
    display = _smoothstep(_linear_to_srgb(linear))
    return display if channel_last else display.movedim(-1, -3)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class PairedViews:
    """One same-condition pair plus its provenance.

    ``raw`` and ``rgb`` come from the *same* tensor element by element, so the
    two arms share physical state, camera, exposure/noise condition and time by
    construction.  ``rgb`` is float32 in ``[0, 1]``; ``rgb_u8`` keeps the
    upstream integer output for visual comparison against the author chain.
    """

    raw: torch.Tensor
    rgb: torch.Tensor
    rgb_u8: torch.Tensor
    raw_domain: str
    rgb_domain: str
    provenance: dict[str, Any]


def paired_views(
    values: torch.Tensor,
    *,
    platform: str,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> PairedViews:
    """Render the frozen-ISP RGB of an already loaded RAW window.

    ``values`` is the loader's real field in its real layout: ``[B, T, C, H, W]``
    channel-first (accepted also as ``[B, T, H, W, C]``), dtype float32 in
    ``[0, 1]`` for the pseudo-RAW view.  No resampling, no per-frame
    normalisation and no extra noise are applied here: the RGB view must differ
    from the RAW view by the ISP alone.
    """

    if not torch.is_tensor(values):
        raise TypeError(f"expected a torch tensor, got {type(values).__name__}")
    if values.ndim < 3:
        raise ValueError(f"expected at least [C, H, W] or [H, W, C], got {tuple(values.shape)}")
    provenance = isp_provenance(platform)
    channel_last = values.shape[-1] == 3 and values.shape[-3] != 3
    raw = values if not channel_last else values.movedim(-1, -3)
    rgb = raw_unit_to_srgb(raw, params=params)
    rgb_u8 = torch.round(rgb * 255.0).to(torch.uint8)
    provenance = dict(provenance)
    provenance["pairs"] = "same tensor element: identical physical state, camera and condition"
    provenance["noise"] = "no noise is drawn here; the RAW arm's measurement is the only source"
    return PairedViews(
        raw=raw,
        rgb=rgb,
        rgb_u8=rgb_u8,
        raw_domain=RAW_VIEW_DOMAIN,
        rgb_domain=RENDERED_RGB_DOMAIN,
        provenance=provenance,
    )
