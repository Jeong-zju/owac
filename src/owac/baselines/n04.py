"""Trainable direct N04 controls, sharing measurements, masks and motion context."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..frameworks.n04.ego import build_propagator
from ..frameworks.n04.geometry import GridSpec
from ..frameworks.n04.model import MeasurementStem, N04Encoder, N04Readout, StepOutput, TokenReadout
from ..frameworks.n04.state import new_state

BASELINE_NAMES = (
    "single_frame",
    "fixed_raw_patch",
    "convgru",
    "motion_compensation",
    "predict_update_filter",
    "action_conditioned_recurrent",
    "n04_no_ego",
    "n04_no_change_branch",
    "n04_all_residual_noise",
    "n04_time_shifted_context",
)


class ConvGRUCell(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.gates = nn.Conv2d(channels + hidden, 2 * hidden, 3, padding=1)
        self.candidate = nn.Conv2d(channels + hidden, hidden, 3, padding=1)

    def forward(self, x, hidden):
        reset, update = torch.sigmoid(self.gates(torch.cat((x, hidden), 1))).chunk(2, 1)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * hidden), 1)))
        return (1 - update) * hidden + update * candidate


class FixedRawPatch(nn.Module):
    """C4: seed-0 row-normalized fixed RAW projection; only readout/probes learn."""

    def __init__(self, cfg):
        super().__init__()
        weights = torch.randn(
            cfg.model.width, cfg.channels, 3, 3, generator=torch.Generator().manual_seed(0)
        )
        weights = weights / weights.flatten(1).norm(dim=1)[:, None, None, None]
        self.register_buffer("weights", weights)
        self.stride = int(cfg.geometry.pixel_stride)

    def forward(self, values):
        return F.conv2d(values, self.weights, stride=self.stride, padding=1)


class ContextRecurrentCell(nn.Module):
    """Action/context-conditioned transition then observation correction; no GRU gates."""

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.transition = nn.Conv2d(width + 8, width, 3, padding=1)
        self.correction = nn.Conv2d(2 * width, width, 3, padding=1)

    def forward(self, inputs, hidden):
        measured, context = inputs[:, : self.width], inputs[:, self.width :]
        predicted = torch.tanh(self.transition(torch.cat((hidden, context), 1)))
        return predicted + 0.5 * torch.tanh(
            self.correction(torch.cat((measured - predicted, predicted), 1))
        )


class DirectBaseline(nn.Module):
    def __init__(self, cfg, name):
        super().__init__()
        self.cfg, self.name = cfg, name
        self.width = cfg.model.width
        self.stem = MeasurementStem(
            channels=cfg.channels,
            width=self.width,
            input_mode=cfg.input_mode,
            cfa_pattern=cfg.cfa_pattern,
            pixel_stride=cfg.geometry.pixel_stride,
        )
        if name == "fixed_raw_patch":
            if cfg.input_mode != "rawvla_rgb3":
                raise ValueError("C4 prototype supports three-channel RawVLA measurements")
            self.stem = FixedRawPatch(cfg)
        self.readout = N04Readout(
            in_channels=self.width,
            tokens=cfg.model.readout_tokens,
            dim=cfg.model.readout_dim,
            cameras=len(cfg.camera_ids),
        )
        self.spec = GridSpec(*cfg.geometry.feature_size, pixel_stride=cfg.geometry.pixel_stride)
        self.propagator = None
        if name in ("motion_compensation", "predict_update_filter"):
            self.propagator = build_propagator(
                "learned_proprio",
                model_config=cfg.model,
                geometry=cfg.geometry,
                motion_dim=cfg.model.motion_dim,
            )
        self.motion_proj = (
            nn.LazyLinear(8) if name in ("convgru", "action_conditioned_recurrent") else None
        )
        self.cell = ConvGRUCell(self.width + 8, self.width) if self.motion_proj else None
        if name == "action_conditioned_recurrent":
            self.cell = ContextRecurrentCell(self.width)
        self.gain = nn.Parameter(torch.tensor(0.0)) if name == "predict_update_filter" else None

    def init_state(self, batch_size, camera_ids=None, device="cpu", dtype=torch.float32):
        return {
            cam: new_state(
                batch_size,
                self.width,
                *self.cfg.geometry.feature_size,
                camera_id=cam,
                device=device,
                dtype=dtype,
            )
            for cam in (camera_ids or self.cfg.camera_ids)
        }

    def reset(self, states, batch_mask):
        return {cam: state.masked_reset(batch_mask) for cam, state in states.items()}

    @staticmethod
    def detach_state(states):
        return {cam: state.detach() for cam, state in states.items()}

    def ensure_motion_dim_from_state_dict(self, state):
        if self.propagator is not None:
            weight = state.get("propagator.motion_encoder.0.weight")
            if weight is not None:
                self.propagator.ensure_motion_dim(
                    weight.shape[1] - 3 - self.cfg.model.camera_embedding
                )

    def encode_measurement(self, values):
        return self.stem(values)

    def step(
        self,
        values,
        context,
        state,
        *,
        camera_id=None,
        measurement_valid=None,
        saturation=None,
        new_measurement=True,
        **kwargs,
    ):
        b = values.shape[0]
        valid = torch.ones_like(values[:, :1]) if measurement_valid is None else measurement_valid
        if saturation is not None:
            valid = valid * (1 - saturation)
        fresh_samples = torch.as_tensor(new_measurement, device=values.device).expand(b)
        valid = valid * fresh_samples[:, None, None, None]
        measured = self.stem(values * valid)
        valid = F.adaptive_avg_pool2d(valid, measured.shape[-2:])
        history = torch.as_tensor(state.history_valid, device=values.device).expand(b)
        history = history[:, None, None, None].to(measured)
        hidden = state.stable * history
        predicted = hidden
        flow = None
        previous_support = state.warp_validity
        if previous_support is None:
            previous_support = torch.zeros_like(valid)
        prior_support = previous_support * history
        if self.propagator is not None:
            propagation = self.propagator(hidden, self.spec, context)
            predicted = propagation.predicted
            flow = propagation.forward_flow
            prior_support = (
                propagation.validity
                * history
                * F.grid_sample(
                    previous_support,
                    propagation.backward_grid,
                    align_corners=False,
                    padding_mode="zeros",
                )
            )
        if self.name in ("single_frame", "fixed_raw_patch"):
            combined = measured
            prior_support = torch.zeros_like(valid)
        elif self.cell is not None:
            vector, _ = context.motion_vector()
            flags = torch.stack(
                [
                    torch.as_tensor(flag, device=vector.device, dtype=vector.dtype)
                    .reshape(-1)
                    .expand(b)
                    for flag in (
                        context.proprio_delta_available,
                        context.executed_action_available,
                        context.dt_available,
                    )
                ],
                -1,
            )
            motion = self.motion_proj(torch.cat((vector, flags), -1))
            ctx = motion[:, :, None, None].expand(b, 8, *measured.shape[-2:])
            combined = self.cell(torch.cat((measured, ctx), 1), hidden)
        else:
            gain = torch.sigmoid(self.gain) if self.gain is not None else 0.4
            combined = predicted + gain * (measured - predicted)
        combined = torch.where(prior_support <= 0, measured, combined)
        combined = torch.where(valid > 0, combined, predicted)
        support = torch.maximum(prior_support, valid)
        fresh = new_state(
            b,
            self.width,
            *measured.shape[-2:],
            camera_id=camera_id or state.camera_id,
            device=measured.device,
            dtype=measured.dtype,
        )
        fresh.stable, fresh.measured = combined, measured
        fresh.prediction = predicted
        fresh.forward_flow = flow
        fresh.warp_validity = support
        fresh.history_valid = support.flatten(1).any(-1)
        fresh.updates = state.updates + 1
        fresh.change = None
        return StepOutput(
            representation=combined,
            state=fresh,
            validity=support,
            diagnostics={"baseline": self.name},
        )

    def readout_tokens(self, output, token_budget=None):
        values, mask = self.readout(output.representation, output.validity)
        if token_budget is not None:
            values, mask = values[:, :token_budget], mask[:, :token_budget]
        return TokenReadout(values, mask)


class N04Ablation(nn.Module):
    def __init__(self, cfg, *, mode):
        super().__init__()
        self.mode = mode
        if mode == "n04_no_ego":
            cfg = cfg.with_model(predictor="identity")
        elif mode == "n04_no_change_branch":
            cfg = cfg.with_model(change_branch=False)
        elif mode == "n04_all_residual_noise":
            cfg = cfg.with_model(change_branch=False, illumination_branch=False)
        elif mode != "n04_time_shifted_context":
            raise ValueError(mode)
        self.encoder = N04Encoder(cfg)
        self._previous = {}

    def init_state(self, *args, **kwargs):
        self._previous.clear()
        return self.encoder.init_state(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._previous.clear()
        return self.encoder.reset(*args, **kwargs)

    def detach_state(self, *args, **kwargs):
        return self.encoder.detach_state(*args, **kwargs)

    def readout_tokens(self, *args, **kwargs):
        return self.encoder.readout_tokens(*args, **kwargs)

    def encode_measurement(self, *args, **kwargs):
        return self.encoder.encode_measurement(*args, **kwargs)

    def predict_before_observe(self, *args, **kwargs):
        return self.encoder.predict_before_observe(*args, **kwargs)

    def correct_after_observe(self, *args, **kwargs):
        return self.encoder.correct_after_observe(*args, **kwargs)

    def step(self, values, context, state, **kwargs):
        if self.mode == "n04_time_shifted_context":
            camera = kwargs.get("camera_id", state.camera_id)
            used = self._previous.get(camera, context)
            self._previous[camera] = context
            context = used
        return self.encoder.step(values, context, state, **kwargs)


def build_baseline(name, cfg):
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline: {name}")
    return N04Ablation(cfg, mode=name) if name.startswith("n04_") else DirectBaseline(cfg, name)
