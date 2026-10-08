from pathlib import Path

import numpy as np
import pytest

from owac.simulation.raw_camera import RawCameraProfile, audit_cfa, read_buffer


def sensor_fixture():
    profile = RawCameraProfile(width=8, height=8)
    hdr = np.zeros((8, 8, 4), dtype=np.float16)
    levels = (np.arange(64).reshape(8, 8) + 1) / 256
    hdr[..., 0] = levels
    hdr[..., 1] = levels + 0.25
    hdr[..., 2] = levels + 0.5
    hdr[..., 3] = 1
    # A known GRBG measurement, independent of the audit routine.
    mosaic = hdr[..., 1].copy()
    mosaic[0::2, 1::2] = hdr[0::2, 1::2, 0]
    mosaic[1::2, 0::2] = hdr[1::2, 0::2, 2]
    cfa = (mosaic.astype(np.float32) * 16777215).astype(np.uint32)
    return profile, hdr, cfa


def test_grbg_measurements_remain_integer_and_full_resolution():
    profile, hdr, cfa = sensor_fixture()
    result = audit_cfa(hdr, cfa, profile)
    assert result["cfa_quantization_maximum_error_dn"] == 0
    assert result["unique_values"] > 16
    assert cfa.dtype == np.uint32


def test_shifted_cfa_phase_is_rejected():
    profile, hdr, cfa = sensor_fixture()
    with pytest.raises(ValueError, match="does not match"):
        audit_cfa(hdr, np.roll(cfa, 1, axis=1), profile)


def test_black_sensor_output_is_rejected():
    profile, hdr, cfa = sensor_fixture()
    with pytest.raises(ValueError, match="no useful scene signal"):
        audit_cfa(hdr, np.zeros_like(cfa), profile)


def test_truncated_capture_is_rejected(tmp_path: Path):
    path = tmp_path / "truncated.bin"
    path.write_bytes(b"\x00" * 15)
    with pytest.raises(ValueError, match="Incomplete buffer"):
        read_buffer(path, "<u4", (2, 2))


def test_profile_cannot_silently_change_bit_depth_or_phase():
    with pytest.raises(ValueError):
        RawCameraProfile(bit_depth=12)
    with pytest.raises(ValueError):
        RawCameraProfile(cfa="RGGB")
