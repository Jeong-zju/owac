"""C3 RAW CNN and C4 fixed RAW patch projection with the SAME spatial readout."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from owac.core.encoder_contract import EncoderOutput
from owac.frameworks.n01.model import MeasurementStem, SpatialReadout
from owac.frameworks.n01.state import detach, init_state, push, reset, state_bytes


class RAWControlEncoder(nn.Module):
    def __init__(self, config, method, projection_seed=7):
        super().__init__()
        if method not in {"c3_raw_cnn", "c4_fixed_patch"}:
            raise ValueError(method)
        self.config, self.method = config, method
        if method == "c3_raw_cnn":
            self.stem = MeasurementStem(config)
        else:
            with torch.random.fork_rng():
                torch.manual_seed(projection_seed)
                self.stem = nn.Conv2d(config.channels, config.width, 2, stride=2, bias=False)
            self.stem.requires_grad_(False)
        self.readout_layer = SpatialReadout(config)
        self.measurement_head = (
            nn.Conv2d(config.width, config.channels, 1)
            if config.aux_measurement_prediction
            else None
        )
        self.register_buffer("calibration_count", torch.tensor(1))

    def init_state(self, batch_size, camera_spec, device="cpu"):
        return init_state(self.config, batch_size, camera_spec, device)

    def reset(self, state, mask):
        return reset(state, mask)

    def detach_state(self, state):
        return detach(state)

    def step(self, frame, context=None, state=None, budget=None):
        del context, budget
        state = push(state, frame)
        values = state.values[0].masked_fill(~state.valid[0], 0.0)
        if self.method == "c3_raw_cnn":
            features = self.stem(values, state.valid[0], state.variance[0])
            valid = state.valid[0].any(1)
        else:
            features = self.stem(values)
            valid = F.avg_pool2d(state.valid[0].float().mean(1, keepdim=True), 2, 2)[:, 0] == 1
        state.last_representation = features
        return EncoderOutput(
            features,
            state,
            valid,
            {"method": self.method, "input_frames": 1},
            {"updates": 1, "state_bytes": state_bytes(state)},
        )

    def readout(self, output, token_budget=None):
        tokens = self.readout_layer(output.representation, output.validity, token_budget)
        tokens.camera_ids = output.state.camera_ids
        return tokens

    def predict_measurement(self, output):
        prediction = self.measurement_head(output.representation)
        return F.interpolate(
            prediction, size=output.state.shape, mode="bilinear", align_corners=False
        )
