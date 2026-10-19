"""Cross-check the P3 frozen-ISP transcription against the real author source.

This is an integration check, not a unit test: it needs the fixed RawVLA checkout
that the author chain lives in.  Set ``OWAC_RAWVLA_SOURCE`` to its root (the
directory containing ``starVLA/`` and ``third_party/``); when the variable is
unset or the files are absent the test skips, so a CPU-only machine still runs
the rest of the suite.

It is deliberately one small check with a real oracle rather than a hashing
system: it proves that ``raw_unit_to_srgb`` is the author's arithmetic and not a
re-invented tone curve, which is what the same-condition RAW/RGB contract needs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from owac.raw.n01_paired_views import (
    FIXED_RAW_RGB10_PARAMS,
    isp_provenance,
    raw_unit_to_srgb_u8,
)

LIBERO_RELATIVE = "starVLA/examples/simBenchmarks/LIBERO/eval_files/raw_rgb10_unprocessing.py"
ROBOTWIN_RELATIVE = "third_party/FastWAM/third_party/RoboTwin/envs/camera/raw_rgb10.py"


def _source_root() -> Path | None:
    raw = os.environ.get("OWAC_RAWVLA_SOURCE")
    if not raw:
        return None
    root = Path(raw)
    return root if root.is_dir() else None


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cached_view(raw10: np.ndarray) -> np.ndarray:
    """The LIBERO cache convention: ``rint(raw10 * 255 / 1023)`` stored as uint8.

    This is lossy by construction, which is why it is *not* used for the
    transcription check below: that check isolates our arithmetic from the
    cache's own quantisation.
    """

    view8 = np.rint(raw10.astype(np.float32) * (255.0 / 1023.0)).astype(np.uint8)
    return view8.astype(np.float32) / 255.0


@pytest.mark.parametrize(
    ("platform", "relative", "symbol"),
    [
        ("libero", LIBERO_RELATIVE, "_raw10_to_rgb8"),
        ("robotwin2", ROBOTWIN_RELATIVE, "raw10_to_rgb8"),
    ],
)
def test_transcription_matches_upstream(platform: str, relative: str, symbol: str) -> None:
    root = _source_root()
    if root is None:
        pytest.skip("OWAC_RAWVLA_SOURCE is not set; author checkout unavailable")
    path = root / relative
    if not path.is_file():
        pytest.skip(f"author module not found at {path}")

    module = _load(path, f"owac_upstream_isp_{platform}")
    upstream = getattr(module, symbol)
    upstream_params = module.FIXED_RAW_RGB10_PARAMS
    assert upstream_params.red_gain == FIXED_RAW_RGB10_PARAMS.red_gain
    assert upstream_params.green_gain == FIXED_RAW_RGB10_PARAMS.green_gain
    assert upstream_params.blue_gain == FIXED_RAW_RGB10_PARAMS.blue_gain
    assert upstream_params.inverse_global_gain == FIXED_RAW_RGB10_PARAMS.inverse_global_gain

    generator = np.random.default_rng(20261015)
    raw10 = generator.integers(0, 1024, size=(8, 9, 3), dtype=np.uint16)
    expected = upstream(raw10, upstream_params)
    # Exact ten-bit domain, i.e. what the float32 cache stores, so the only thing
    # under test is our transcription of the upstream arithmetic.
    got = raw_unit_to_srgb_u8(raw10.astype(np.float32) / 1023.0)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected), np.abs(got.astype(int) - expected.astype(int)).max()


def test_uint8_cache_quantisation_is_bounded_and_reported() -> None:
    """The LIBERO cache shifts the ISP input by up to half a 1/255 step.

    The shift is a property of the stored cache, applies identically to the RAW
    and RGB arms (both read the same field), and is recorded in the provenance.
    Reporting it matters: the RGB arm must not be described as a lossless render
    of a ten-bit measurement.
    """

    root = _source_root()
    if root is None:
        pytest.skip("OWAC_RAWVLA_SOURCE is not set; author checkout unavailable")
    path = root / LIBERO_RELATIVE
    if not path.is_file():
        pytest.skip(f"author module not found at {path}")
    module = _load(path, "owac_upstream_isp_quantisation")

    generator = np.random.default_rng(7)
    raw10 = generator.integers(0, 1024, size=(16, 16, 3), dtype=np.uint16)
    exact = module._raw10_to_rgb8(raw10, module.FIXED_RAW_RGB10_PARAMS)
    cached = raw_unit_to_srgb_u8(_cached_view(raw10))
    delta = np.abs(cached.astype(int) - exact.astype(int))
    assert delta.max() <= 8, delta.max()
    assert delta.mean() < 1.0

    provenance = isp_provenance("libero")
    assert "quantised" in provenance["quantisation"]
    assert provenance["cache_dtype"] == "uint8"
    assert isp_provenance("robotwin")["cache_dtype"] == "float32"


def test_author_cache_quantisation_is_not_recoverable() -> None:
    """A uint8 cache cannot return the ten-bit measurement; that loss is reported, not hidden."""

    raw10 = np.array([[[0, 1, 2], [500, 501, 502], [1021, 1022, 1023]]], dtype=np.uint16)
    cached = _cached_view(raw10)
    recovered = np.rint(cached * 1023.0)
    assert recovered.max() <= 1023
    error = np.abs(recovered - raw10.astype(np.float64))
    # Two ten-bit steps are routinely indistinguishable after the 1/255 grid.
    assert error.max() >= 1.0
