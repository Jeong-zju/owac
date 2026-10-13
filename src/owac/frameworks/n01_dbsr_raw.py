"""Original N01 RAW-domain integration as an extra DBSR feature node.

The mature backbone and fusion are unchanged. Integration acts on distinct
original measurements before encoding, separately for each CFA plane. Cached
sufficient statistics are the deliberate no-recompute control; new graphs
cannot withdraw the old spatial constituents of those cached sums.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from owac.baselines.burst_features import CFA_PHASES, align_features
from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.state import init_state, push
from owac.frameworks.n01.support import SupportDomainIntegrator

MODES = (
    "baseline",
    "duplicate_current",
    "all_open_integrated",
    "stock_dynamic",
    "calibrated_static",
    "calibrated_stale",
    "calibrated_recompute",
)


def config_for(mode, calibrated):
    return N01Config.tiny(
        input_mode="bayer_cfa",
        channels=4,
        planes=4,
        learnable_gate=False,
        spatial_radii=(2,),
        time_depths=(8,),
        aux_measurement_prediction=False,
        **(calibrated if mode.startswith("calibrated") else {}),
    )


def align_planes(values, valid, offsets):
    b, t, p, h, w = values.shape
    data = values.permute(0, 2, 1, 3, 4).reshape(b * p, t, 1, h, w)
    mask = valid.permute(0, 2, 1, 3, 4).reshape_as(data)
    field = offsets.repeat_interleave(p, 0)
    result, usable, _ = align_features(data, mask, field)
    return (
        result.reshape(b, p, t, h, w).permute(0, 2, 1, 3, 4),
        usable.reshape(b, p, t, h, w).permute(0, 2, 1, 3, 4),
    )


class RawDomainHistory:
    def __init__(self, config, batch, shape, freeze_frame=8):
        self.config = config
        self.integrator = SupportDomainIntegrator(config)
        self.single = SupportDomainIntegrator(config.replace(time_depths=(1,)))
        self.state = init_state(config, batch, {"camera_ids": ["burst"], "shape": shape})
        self.freeze_frame = freeze_frame
        self.fixed = None
        self.cached = []

    def step(self, frame, mode, offsets=None):
        if mode not in MODES:
            raise ValueError("unknown RAW support intervention")
        valid = frame.valid_mask & torch.isfinite(frame.values)
        frame = replace(frame, values=frame.values.masked_fill(~valid, 0), valid_mask=valid)
        self.state = push(self.state, frame)
        # The original buffer folds bad/saturated masks into validity. Missing
        # pairs provide no refutation; retain the core's gradual forgetting.
        valid = self.state.valid[0]
        plane = self.state.plane_support("burst")
        evidence, _ = self.integrator.edge_field(frame, self.state)
        measured = torch.stack(
            [
                valid & F.pad(valid[..., :, :-1], (1, 0)),
                valid & F.pad(valid[..., :-1, :], (0, 0, 1, 0)),
            ],
            2,
        )
        strength, age, merges, splits = self.integrator.update_edges(
            plane.edges, plane.support_age, evidence, measured, plane.open_edges
        )
        opened = self.integrator.last_open_edges
        self.state.support["burst"] = replace(
            plane,
            edges=strength,
            open_edges=opened,
            support_age=age,
            merge_count=plane.merge_count + merges,
            split_count=plane.split_count + splits,
        )
        edges = torch.where(opened, strength, 0)
        now = int(self.state.now[0])
        if mode in {"baseline", "duplicate_current", "all_open_integrated"}:
            edges = torch.ones_like(edges)
        elif mode == "calibrated_static":
            if now <= self.freeze_frame:
                self.fixed = edges.clone()
            edges = self.fixed
        state = self.state
        if offsets is not None:
            values, aligned_valid = align_planes(
                state.values.transpose(0, 1), state.valid.transpose(0, 1), offsets
            )
            variance, _ = align_planes(
                state.variance.transpose(0, 1), state.valid.transpose(0, 1), offsets
            )
            state = replace(
                state,
                values=values.transpose(0, 1),
                variance=variance.transpose(0, 1),
                valid=aligned_valid.transpose(0, 1),
            )
        if mode == "calibrated_stale":
            # Each arrival contributes one immutable spatial sufficient sum.
            # Never treat a pooled mean as an extra independent measurement.
            single, _ = self.single.domain_statistics(self.state, edges, depths=(1,))
            weight = single.support_count[0]
            denominator = weight.clamp_min(1e-6)
            self.cached.append(
                {
                    "S": single.mu[0] * denominator,
                    "W": weight,
                    "Z": single.variance_proxy[0] * (denominator.square() + 1e-6),
                    "time": now,
                }
            )
            rows = self.cached[-self.config.max_history :][::-1]
            sums = {key: torch.stack([r[key] for r in rows], 1) for key in ["S", "W", "Z"]}
            if offsets is not None:
                usable = sums["W"] > 0
                for key, data in sums.items():
                    sums[key], _ = align_planes(data, usable, offsets[:, : len(rows)])
            total = sums["W"].sum(1)
            denominator = total.clamp_min(1e-6)
            times = total.new_tensor([now - r["time"] for r in rows])[None, :, None, None, None]
            mu = sums["S"].sum(1) / denominator
            variance = sums["Z"].sum(1) / (denominator.square() + 1e-6)
            age_field = (sums["W"] * times).sum(1) / denominator
            available = total > 0
        else:
            statistics, _ = self.integrator.domain_statistics(state, edges)
            mu, variance, age_field = (
                statistics.mu[0],
                statistics.variance_proxy[0],
                statistics.age[0],
            )
            available = statistics.support_count[0] > 0
        return {
            "mu": mu,
            "variance": variance,
            "age": age_field,
            "valid": available,
            "merges": merges.tolist(),
            "splits": splits.tolist(),
            "plane_open_fraction": float(opened.float().mean()),
            "unique_source_ids": self.state.frame_ids.T.tolist(),
            "state": self.state,
        }


def sequence_features(model, batch, mode, protocol):
    device = next(model.parameters()).device
    b, t, _, h, w = batch.values.shape
    cfg = config_for(mode, protocol["calibrated_support"])
    domain = RawDomainHistory(cfg, b, (h, w), protocol["freeze_frame"])
    for now in range(t):
        frame = MeasurementFrame(
            values=batch.values[:, now],
            valid_mask=batch.valid[:, now],
            sample_xy=CFA_PHASES,
            camera_id="burst",
            frame_ids=torch.full((b,), now),
            timestamp=torch.full((b,), float(now)),
            metadata={"variance": batch.variance[:, now]},
        )
        # Source IDs refer only to current/past observations. Empty slots have
        # no usable measurement and are masked by the original N01 buffer.
        count = min(now + 1, cfg.max_history)
        offsets = torch.zeros(b, cfg.max_history, 2, h, w)
        if batch.condition == "translation_oracle_alignment":
            ids = torch.arange(now, now - cfg.max_history, -1).clamp_min(0)
            delta = batch.translations[:, ids] - batch.translations[:, now : now + 1]
            offsets = delta.flip(-1)[:, :, :, None, None].expand(-1, -1, -1, h, w)
        with torch.no_grad():
            stats = domain.step(
                frame, mode, offsets if batch.condition == "translation_oracle_alignment" else None
            )
        state = stats["state"]
        raw = state.values[:count].transpose(0, 1).to(device)
        valid = state.valid[:count].transpose(0, 1).to(device)
        features, centers = model.encoder.encode_frames(raw, valid)
        features, centers, field = align_features(features, centers, offsets[:, :count].to(device))
        if mode == "baseline":
            represented, weights = model.encoder.fusion(features, centers, field)
        else:
            extra = raw[:, :1] if mode == "duplicate_current" else stats["mu"].to(device)[:, None]
            extra_valid = (
                valid[:, :1] if mode == "duplicate_current" else stats["valid"].to(device)[:, None]
            )
            encoded, available = model.encoder.encode_frames(extra, extra_valid)
            all_features = torch.cat([features, encoded], 1)
            all_valid = torch.cat([centers, available], 1)
            all_fields = torch.cat([field, field.new_zeros(b, 1, 2, h, w)], 1)
            _, weights = model.encoder.fusion(all_features, all_valid, all_fields)
            # Original N01 variance/age cost form, kept separate from content.
            # This is a bounded heuristic prior, not a calibrated probability.
            cost = (
                protocol["reliability_noise_weight"] * stats["variance"] / batch.variance[:, now]
                + protocol["reliability_age_weight"] * stats["age"] / cfg.max_history
            )
            quality = (-cost).exp().amin(1, keepdim=True).to(device)
            prior = torch.ones_like(weights)
            prior[:, -1] = quality
            weights = weights * prior
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
            represented = (all_features * weights).sum(1)
        yield represented, weights, {k: v for k, v in stats.items() if k != "state"}
