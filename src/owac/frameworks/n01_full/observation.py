"""One observation transform for training, replay and serving.

The buffer stores this domain's observations. RGB is formed at native resolution,
then both arms apply the same optional full-field resize. No client crop is accepted
as that operation. A repeated preprocessed frame is checked rather than transformed twice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch.nn import functional as F

from owac.core.encoder_contract import MeasurementFrame
from owac.raw.n01_paired_views import raw_unit_to_srgb


@dataclass(frozen=True)
class ObservationConfig:
    native_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    history_stride: int = 1
    shot: float = 0.0
    read_variance: float = 1e-4
    noise_provenance: str = "declared_same_domain_proxy_unfitted"
    version: str = "native-isp-full-field-v1"

    def __post_init__(self):
        if self.history_stride != 1:
            raise ValueError("native complete transport currently collects every actual arrival")
        if self.shot < 0 or self.read_variance <= 0:
            raise ValueError(
                "domain variance must have nonnegative shot and positive read variance"
            )
        for shape in (self.native_shape, self.output_shape):
            if len(shape) != 2 or any(int(x) < 0 for x in shape) or (0 in shape and any(shape)):
                raise ValueError("observation shapes must be (0,0) or two positive dimensions")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("native_shape", "output_shape"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


def _mask(value: Any, values: torch.Tensor, default: bool) -> torch.Tensor:
    if value is None:
        return torch.full_like(values, default, dtype=torch.bool)
    value = torch.as_tensor(value, device=values.device, dtype=torch.bool)
    if value.ndim == 1:
        value = value[:, None, None, None]
    if value.ndim == 3:
        value = value[:, None]
    return torch.broadcast_to(value, values.shape)


def prepare_frame(
    frame: MeasurementFrame, config: ObservationConfig, domain: str
) -> MeasurementFrame:
    if domain not in ("RAW", "RGB"):
        raise ValueError("observation domain must be RAW or RGB")
    values = torch.as_tensor(frame.values).float()
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("v1.2 observations require native BCHW, three co-located channels")
    tag = frame.metadata.get("observation_contract")
    contract = {**config.to_dict(), "domain": domain}
    if tag is not None:
        # JSON transports turn tuples into lists.
        if {
            **tag,
            "native_shape": tuple(tag["native_shape"]),
            "output_shape": tuple(tag["output_shape"]),
        } != contract:
            raise ValueError("preprocessed observation contract differs from the model")
        return frame
    native = tuple(values.shape[-2:])
    if any(config.native_shape) and native != config.native_shape:
        raise ValueError(
            f"native observation {native} != {config.native_shape}; client resize/crop forbidden"
        )
    valid = _mask(frame.valid_mask, values, True) & torch.isfinite(values)
    bad = _mask(frame.bad_mask, values, False)
    saturated = _mask(frame.saturated_mask, values, False)
    usable = valid & ~bad & ~saturated
    values = values.masked_fill(~usable, 0)
    if domain == "RGB":
        values = raw_unit_to_srgb(values)
    # Identical declared domain model in train and serve. RGB never receives RAW
    # variance/flow/support, including when the loader happens to supply it.
    variance = config.read_variance + config.shot * values.clamp_min(0)
    output = config.output_shape if any(config.output_shape) else native
    if output != native:

        def resize(value):
            return F.interpolate(value.float(), size=output, mode="bilinear", align_corners=False)

        values = resize(values)
        valid = resize(valid.float()) >= 1 - 1e-6
        bad = resize(bad.float()) > 1e-6
        saturated = resize(saturated.float()) > 1e-6
        if variance is not None:
            # A conservative bound for unknown correlation of interpolation taps.
            variance = resize(variance.clamp_min(0).sqrt()).square()
    metadata = {
        **frame.metadata,
        "observation_contract": contract,
        "native_shape": native,
        "grid_to_sensor_xy": (native[1] / output[1], native[0] / output[0]),
        "variance": variance,
        "variance_resampling": "unknown_correlation_bound",
    }
    metadata["noise_provenance"] = config.noise_provenance
    return replace(
        frame,
        values=values,
        valid_mask=valid,
        bad_mask=bad,
        saturated_mask=saturated,
        metadata=metadata,
    )
