"""N01 v1.2 P3: same-condition RAW/RGB pairing.

The point of these tests is the causal contract, not a score: the RGB arm must be
a deterministic function of the *same* RAW element, tagged as a rendered view,
and the torch training path must reproduce the NumPy transcription of the
upstream ISP exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from owac.raw.n01_paired_views import (
    FIXED_RAW_RGB10_PARAMS,
    RAW_VIEW_DOMAIN,
    RENDERED_RGB_DOMAIN,
    isp_provenance,
    paired_views,
    raw10_to_view8,
    raw_unit_to_srgb,
    raw_unit_to_srgb_u8,
)


def _window(seed: int = 0, batch: int = 2, frames: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((batch, frames, 3, 16, 24), generator=generator, dtype=torch.float32)


def test_torch_path_matches_numpy_oracle_exactly() -> None:
    values = _window()
    torch_rgb = raw_unit_to_srgb(values)
    oracle = raw_unit_to_srgb_u8(values.movedim(-3, -1).numpy())
    as_u8 = torch.round(torch_rgb.movedim(-3, -1) * 255.0).to(torch.uint8).numpy()
    assert np.array_equal(as_u8, oracle)


def test_oracle_matches_upstream_formula_on_integer_raw10() -> None:
    """The oracle is the upstream arithmetic, not a reinvented tone curve."""

    raw10 = np.array([[[0, 1, 1023], [512, 200, 900]]], dtype=np.uint16)
    params = FIXED_RAW_RGB10_PARAMS
    gains = np.asarray([params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32)
    linear = np.clip(raw10.astype(np.float32) / 1023.0 * gains, 0.0, 1.0)
    display = np.clip(linear, 0.0, 1.0)
    srgb = np.where(display <= 0.0031308, 12.92 * display, 1.055 * display ** (1.0 / 2.4) - 0.055)
    smooth = np.clip(srgb, 0.0, 1.0)
    smooth = smooth * smooth * (3.0 - 2.0 * smooth)
    expected = np.rint(smooth * 255.0).astype(np.uint8)

    # The cached view is uint8 / 255, and raw10/1023 is what that stores.
    cached = raw10.astype(np.float32) / 1023.0
    assert np.array_equal(raw_unit_to_srgb_u8(cached), expected)


def test_integer_roundtrip_reproduces_the_cached_view() -> None:
    raw10 = np.arange(0, 1024, dtype=np.uint16).reshape(1, 1, -1)
    raw10 = np.repeat(raw10, 3, axis=-1)
    view8 = raw10_to_view8(raw10)
    assert view8.dtype == np.uint8
    assert int(view8.max()) == 255
    # The cache is raw10/1023 quantised to 1/255, so the ISP input is already lossy.
    recovered = view8.astype(np.float32) / 255.0
    assert np.abs(recovered - raw10.astype(np.float32) / 1023.0).max() <= 0.5 / 255.0


def test_rgb_is_a_deterministic_function_of_the_same_element() -> None:
    values = _window(seed=3)
    first = paired_views(values, platform="libero")
    second = paired_views(values, platform="libero")
    assert torch.equal(first.rgb, second.rgb)

    perturbed = values.clone()
    perturbed[:, 1, 2, 5, 7] = torch.clamp(perturbed[:, 1, 2, 5, 7] + 0.25, 0.0, 1.0)
    after = paired_views(perturbed, platform="libero")
    changed = (after.rgb != first.rgb).any(dim=-3)
    assert bool(changed[:, 1, 5, 7].all())
    # Untouched frame slots keep the RAW arm's own measurement.
    assert torch.equal(after.rgb[:, 0], first.rgb[:, 0])


def test_domain_tags_keep_rendered_rgb_out_of_the_raw_entry() -> None:
    pair = paired_views(_window(), platform="robotwin")
    assert pair.raw_domain == RAW_VIEW_DOMAIN
    assert pair.rgb_domain == RENDERED_RGB_DOMAIN
    assert pair.raw_domain != pair.rgb_domain
    assert pair.provenance["raw_domain_tag"] == RAW_VIEW_DOMAIN
    assert pair.provenance["rendered_domain_tag"] == RENDERED_RGB_DOMAIN
    assert pair.provenance["not_a_cfa"] is True


def test_channel_last_layout_agrees_with_channel_first() -> None:
    values = _window(seed=5)
    channel_first = paired_views(values, platform="libero")
    channel_last = paired_views(values.movedim(-3, -1), platform="libero")
    # A shape equality followed by a tensor equality adds nothing (the tensor equality
    # already implies it), so only the equality is kept.
    assert torch.equal(channel_last.raw, channel_first.raw)
    assert torch.equal(channel_last.rgb, channel_first.rgb)


def test_provenance_records_the_real_upstream_identity() -> None:
    libero = isp_provenance("libero")
    assert libero["symbol"] == "_raw10_to_rgb8"
    assert "raw_rgb10_unprocessing.py" in libero["module"]
    assert libero["cache_dtype"] == "uint8"
    assert "quantised" in libero["quantisation"]

    robotwin = isp_provenance("robotwin2")
    assert robotwin["symbol"] == "raw10_to_rgb8"
    assert "RoboTwin" in robotwin["module"]
    assert robotwin["cache_dtype"] == "float32"

    with pytest.raises(ValueError):
        isp_provenance("droid")


def test_isp_is_not_identity_but_preserves_monotonic_order() -> None:
    ramp = torch.linspace(0.0, 1.0, 32, dtype=torch.float32)
    values = ramp.reshape(1, 1, 1, 1, 32).repeat(1, 1, 3, 1, 1)
    rgb = raw_unit_to_srgb(values)
    # One channel only: flattening the three identical channels would compare the
    # end of one ramp against the start of the next.
    flat = rgb[0, 0, 0].reshape(-1)
    source = values[0, 0, 0].reshape(-1)
    assert not torch.allclose(flat, source)
    assert bool((flat[1:] >= flat[:-1]).all())
    assert float(flat[0]) == 0.0
    assert float(flat[-1]) == pytest.approx(1.0, abs=1e-6)
