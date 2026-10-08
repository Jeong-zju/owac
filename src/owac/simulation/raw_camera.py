"""Capture native Isaac camera CFA buffers before demosaicing.

The simulator is a synthetic sensor, not a calibrated physical RAW source.
Only the constructor imports Isaac. NumPy is an optional data dependency.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawCameraProfile:
    width: int = 320
    height: int = 240
    bit_depth: int = 24
    cfa: str = "GRBG"
    response_scale: float = 1.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("CFA dimensions must be positive and even")
        if self.bit_depth != 24 or self.cfa != "GRBG" or self.response_scale != 1.0:
            raise ValueError("This validated profile uses 24-bit GRBG with unit response")

    @property
    def white_level(self) -> int:
        return (1 << self.bit_depth) - 1


def read_buffer(path: Path, dtype: str, shape: tuple[int, ...]):
    """Reject incomplete frames instead of padding, truncating, or normalizing them."""
    import numpy as np

    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Incomplete buffer {path}: expected {expected} bytes")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def audit_cfa(hdr, cfa, profile: RawCameraProfile) -> dict:
    """Check native quantization against the corresponding linear HDR samples."""
    import numpy as np

    if cfa.dtype != np.dtype("uint32") or cfa.shape != (profile.height, profile.width):
        raise ValueError("CFA must be a full-resolution uint32 mosaic")
    if hdr.shape != (profile.height, profile.width, 4) or not np.isfinite(hdr).all():
        raise ValueError("HDR must contain a finite RGBA frame")
    if int(cfa.max()) > profile.white_level or np.unique(cfa).size < 16:
        raise ValueError("CFA is out of range or contains no useful scene signal")
    expected = np.empty_like(cfa)
    for row, col, channel in [(0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 1)]:
        values = np.clip(hdr[row::2, col::2, channel].astype(np.float32), 0, 1)
        expected[row::2, col::2] = (values * profile.white_level).astype(np.uint32)
    error = np.abs(cfa.astype(np.int64) - expected.astype(np.int64))
    maximum = int(error.max())
    if maximum > 1:
        raise ValueError(f"CFA does not match native HDR sampling: maximum error {maximum} DN")
    return {
        "dtype": str(cfa.dtype),
        "shape": list(cfa.shape),
        "minimum_dn": int(cfa.min()),
        "maximum_dn": int(cfa.max()),
        "unique_values": int(np.unique(cfa).size),
        "hdr_rgb_maximum": float(hdr[..., :3].max()),
        "cfa_quantization_maximum_error_dn": maximum,
        "black_fraction": float(np.mean(cfa == 0)),
        "saturated_fraction": float(np.mean(cfa == profile.white_level)),
    }


class NativeRawCamera:
    """Native renderer HDR -> CFA -> noise / companding, exported without an ISP decode."""

    def __init__(self, directory: Path, profile: RawCameraProfile, eye, target):
        import numpy as np
        import omni.kit.app
        from isaacsim.core.experimental.utils.app import enable_extension

        enable_extension("isaacsim.sensors.experimental.rtx")
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from pxr import Gf

        self.directory = directory
        self.profile = profile
        directory.mkdir(parents=True, exist_ok=False)
        enable_extension("omni.sensors.nv.camera")
        ext = (
            omni.kit.app.get_app()
            .get_extension_manager()
            .get_extension_path_by_module("omni.sensors.nv.camera")
        )
        # The native implementation initializes its ISP even for pre-ISP requests.
        # Its bundled program is auxiliary; CFA data is read from stage 2, before it.
        program = (Path(ext) / "bin/isp_program_example_3848_2168_r16.ispprg").read_bytes()
        attributes = {
            "focalLength": 18.0,
            "omni:sensor:modelName": "CameraCore",
            "omni:sensor:modelVendor": "NVIDIA",
            "omni:sensor:marketName": "Generic",
            "omni:sensor:core:colorCorrectionBlack": 0.0,
            "omni:sensor:core:colorCorrectionFullwellBlack": 1.0,
            "omni:sensor:core:colorCorrectionSensorResponseScale": profile.response_scale,
            "omni:sensor:core:colorCorrectionWhiteBalance": [1.0, 1.0, 1.0],
            "omni:sensor:core:colorCorrectionRedBlueSwap": False,
            "omni:sensor:core:colorCorrectionOutputFloat16": True,
            "omni:sensor:core:colorCorrectionApplySaturation": True,
            "omni:sensor:core:colorFilterArrayCfaSemantic": profile.cfa,
            "omni:sensor:core:colorFilterArrayCfaCf00": [0.0, 1.0, 0.0],
            "omni:sensor:core:colorFilterArrayCfaCf01": [1.0, 0.0, 0.0],
            "omni:sensor:core:colorFilterArrayCfaCf10": [0.0, 0.0, 1.0],
            "omni:sensor:core:colorFilterArrayCfaCf11": [0.0, 1.0, 0.0],
            "omni:sensor:core:colorFilterArrayMaximalValue": profile.white_level,
            "omni:sensor:core:colorFilterArrayOutputDataTypeFormat": "UINT32",
            "omni:sensor:core:ispSmodelCameraProgram": base64.b64encode(program).decode("ascii"),
            "omni:sensor:core:introspectionOutputFile": True,
            "omni:sensor:core:introspectionOutputFileDirectory": str(directory.resolve()),
            "omni:sensor:core:introspectionOutputFileEachFrameOneFile": True,
            "omni:sensor:core:introspectionOutputFileOnlyLastFrame": False,
        }
        for row, a in enumerate("RGB"):
            for col, b in enumerate("rgb"):
                attributes[f"omni:sensor:core:colorCorrectionMatrix{a}{b}"] = float(row == col)
        rotation = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
            .GetInverse()
            .ExtractRotationQuat()
        )
        self.camera = RtxCamera(
            "/World/OWACRawCamera",
            tick_rate=0,
            schemas=["OmniSensorGenericCameraCoreAPI"],
            attributes=attributes,
            positions=np.asarray(eye, dtype=np.float64),
            orientations=np.asarray([rotation.GetReal(), *rotation.GetImaginary()]),
        )
        self.sensor = CameraSensor(
            self.camera,
            resolution=(profile.height, profile.width),
            annotators=["rgb"],
            render_vars=["HdrColor", "OmniCameraSensorPreIsp", "OmniCameraSensorPostIsp"],
        )
        self.last_frame = -1
        self.metadata = {
            **asdict(profile),
            "source": "Isaac native CameraCore ISP introspection, stage 2-cfa",
            "signal_kind": "synthetic_sensor_cfa_noiseless",
            "container_dtype": "uint32",
            "byte_order": "little",
            "black_level": 0,
            "white_level": profile.white_level,
            "cfa_origin_row_col": [0, 0],
            "pixel_coordinates": "full frame, row and column starting at zero; no resize/crop",
            "exposure_seconds": None,
            "physical_sensor_gain": None,
            "physical_camera": None,
            "hdr_source_dtype": "float16",
            "effective_precision": "limited by renderer FP16 HDR; not 24 physical measurement bits",
            "bad_pixel_mask": None,
            "missing_pixels": "none; incomplete buffers fail capture",
            "saturation_mask": "CFA equals white_level; separately saved per frame",
            "native_isp_program_sha256": hashlib.sha256(program).hexdigest(),
            "camera_eye_xyz": list(eye),
            "camera_target_xyz": list(target),
            "usd_focal_length": 18.0,
            "processing": ["renderer linear HDR", "identity color correction", "native GRBG ADC"],
            "noise_variant": "3-noise, separate file; native default uncalibrated noise",
        }

    def capture(self):
        """Return one new complete native frame, its noise variant, and audit evidence."""
        candidates = []
        for path in self.directory.glob("2-cfa*.bin"):
            match = re.fullmatch(r"2-cfa(\d+)\.bin", path.name)
            if match and int(match[1]) > self.last_frame:
                candidates.append(int(match[1]))
        for frame in sorted(candidates, reverse=True):
            hdr_path = self.directory / f"0-texread{frame}.bin"
            cfa_path = self.directory / f"2-cfa{frame}.bin"
            noise_path = self.directory / f"3-noise{frame}.bin"
            if not hdr_path.exists() or not noise_path.exists():
                continue
            shape = (self.profile.height, self.profile.width)
            hdr = read_buffer(hdr_path, "<f2", (*shape, 4))
            cfa = read_buffer(cfa_path, "<u4", shape)
            noisy = read_buffer(noise_path, "<u4", shape)
            audit = audit_cfa(hdr, cfa, self.profile)
            self.last_frame = frame
            return cfa, noisy, {"native_frame_id": frame, **audit}
        raise RuntimeError("No new complete native RAW frame was produced")
