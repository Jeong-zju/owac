"""Differentiable, same-domain SEA-RAFT on the actual causal observation window.

The accepted P2 estimator is an inference adapter and intentionally uses no_grad.
This module reuses its full vendored network, not its inference-only wrapper, so
native action loss can adapt flow parameters. No RGB teacher or cross-arm cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from owac.backends.raw_correspondence import CorrespondenceEstimator
from owac.frameworks.n01_full.motion_builders import WindowMotion


@dataclass(frozen=True)
class MotionConfig:
    depth: str = "resnet34"
    iters: int = 4
    checkpoint: str | None = None
    checkpoint_domain: str | None = None
    trainable: bool = True
    pair_chunk: int = 1
    max_cycle_error: float = 3.0
    cycle_check: bool = True
    activation_checkpointing: bool = True
    version: str = "same-domain-sea-raft-action-v1"

    def __post_init__(self):
        if self.depth not in ("resnet18", "resnet34") or self.iters < 0 or self.pair_chunk < 1:
            raise ValueError("invalid SEA-RAFT architecture/iteration/chunk declaration")
        if self.checkpoint and self.checkpoint_domain not in ("RAW", "RGB"):
            raise ValueError("pretrained motion weights require their explicit input domain")
        if self.max_cycle_error <= 0:
            raise ValueError("cycle-error proxy threshold must be positive")

    def to_dict(self):
        return asdict(self)


class DomainWindowMotion(nn.Module):
    def __init__(self, config: MotionConfig, domain: str, network: nn.Module | None = None):
        super().__init__()
        self.config, self.domain = config, domain
        if config.checkpoint and config.checkpoint_domain != domain:
            raise ValueError("motion checkpoint belongs to a different input domain")
        if config.checkpoint and not Path(config.checkpoint).is_file():
            raise FileNotFoundError(config.checkpoint)
        if network is None:
            estimator = CorrespondenceEstimator(
                mode="rgb3",
                depth=config.depth,
                iters=config.iters,
                pretrained_checkpoint=config.checkpoint,
            )
            report = estimator.pretrained_report
            if report and (report["missing_key_count"] or report["unexpected_key_count"]):
                raise ValueError("motion checkpoint must load the complete declared architecture")
            network = estimator.model
            self.initialization = dict(report) if report else {"kind": "random", "trained": False}
        else:
            self.initialization = {"kind": "injected_test_network", "trained": False}
        self.network = network
        self.network.requires_grad_(config.trainable)
        self.network.eval()  # fixed BN statistics, shared by Base/N01 and train/serve

    def train(self, mode: bool = True):
        super().train(mode)
        self.network.eval()
        return self

    def _pair(self, reference, source):
        from owac.frameworks.n01_full.acceleration import neural_autocast

        with neural_autocast(self, reference):
            return self._pair_neural(reference, source)

    def _pair_neural(self, reference, source):
        height, width = reference.shape[-2:]
        # Full correlation pyramid needs 128 cells; padding is never a source observation.
        pad_h, pad_w = max(128 - height, 0), max(128 - width, 0)
        reference = F.pad(reference, (0, pad_w, 0, pad_h), mode="replicate")
        source = F.pad(source, (0, pad_w, 0, pad_h), mode="replicate")
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            def forward_pair(a, b):
                result = self.network(a, b, iters=self.config.iters, test_mode=True)
                return result["final"] if isinstance(result, dict) else result

            result = checkpoint(forward_pair, reference, source, use_reentrant=False)
        else:
            result = self.network(reference, source, iters=self.config.iters, test_mode=True)
        flow = result["final"] if isinstance(result, dict) else result
        if isinstance(flow, (list, tuple)):
            flow = flow[-1]
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError("SEA-RAFT must return BCHW pull flow with two channels")
        return flow[..., :height, :width].float()

    def forward(self, *, frame: Any, state: Any, camera_id=None, window=None):
        from owac.baselines.burst_features import align_features

        count = int(state.filled.max()) if window is None else int(window)
        values = state.values[:count].transpose(0, 1).float()
        legal = state.valid[:count].transpose(0, 1) & torch.isfinite(values)
        values = torch.where(legal, values, torch.zeros_like(values))
        batch, _, _, height, width = values.shape
        observed = state.observed[:count].T
        pair_valid = observed.clone()
        zero_flow = values.new_zeros(batch, 1, 2, height, width)
        zero = values.new_zeros(batch, 1, height, width)
        fields, risk = zero_flow, zero
        visible = torch.ones_like(zero, dtype=torch.bool)
        if count > 1:
            # Batch independent (current, historical source) pairs. Fixed BN
            # statistics mean this does not mix observations across sources.
            reference = (
                values[:, :1].expand(-1, count - 1, -1, -1, -1).reshape(-1, *values.shape[2:])
            )
            sources = values[:, 1:].reshape(-1, *values.shape[2:])
            chunk = getattr(self, "pair_chunk", self.config.pair_chunk)
            pieces, cycles = [], []
            for start in range(0, len(sources), chunk):
                ref, src = reference[start : start + chunk], sources[start : start + chunk]
                flow = self._pair(ref, src)
                pieces.append(flow)
                if self.config.cycle_check:
                    with torch.no_grad():
                        reverse = self._pair(src, ref)
                        pair = torch.stack([torch.zeros_like(reverse), reverse], 1)
                        pull = torch.stack([torch.zeros_like(flow), flow.detach()], 1)
                        sampled, valid, _ = align_features(
                            pair, torch.ones_like(pair[:, :, :1], dtype=torch.bool), pull
                        )
                        cycle = (flow.detach() + sampled[:, 1]).norm(dim=1)
                        cycles.append(cycle.masked_fill(~valid[:, 1, 0], float("inf")))
            field = torch.cat(pieces).reshape(batch, count - 1, 2, height, width)
            finite = torch.isfinite(field).all(2)
            field = torch.nan_to_num(field)
            cycles = (
                torch.cat(cycles).reshape(batch, count - 1, height, width)
                if cycles
                else values.new_zeros(batch, count - 1, height, width)
            )
            pair_valid[:, 1:] &= finite.flatten(2).any(2)
            fields = torch.cat([zero_flow, field], 1)
            risk = torch.cat([zero, cycles], 1)
            visible = torch.cat([visible, finite & (cycles <= self.config.max_cycle_error)], 1)
        return WindowMotion(
            offsets_xy=fields,
            valid=pair_valid,
            mapping_version=self.config.version,
            decision_version=self.config.version,
            risk_kind="forward_backward_cycle_proxy_not_calibrated",
            risk=risk,
            visibility=visible,
            source_frame_ids=tuple(state.frame_ids[:count, 0].tolist()),
            sensor_per_packed=1,
            query_shape=(height, width),
        )


def known_transform_loss(provider: DomainWindowMotion, image, valid=None, shift=(2, 1)):
    """Optional known-transform supervision on a REAL current-domain frame.

    This labels only the registered synthetic spatial transform, never natural
    scene motion, exposure differences or future actions. It needs no RGB teacher.
    """
    if valid is not None:
        image = torch.where(valid, image, torch.zeros_like(image))
    dx, dy = map(int, shift)
    source = torch.zeros_like(image)
    height, width = image.shape[-2:]
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError("known transform exceeds the available frame")
    x0, x1, y0, y1 = max(0, -dx), min(width, width - dx), max(0, -dy), min(height, height - dy)
    source[..., y0 + dy : y1 + dy, x0 + dx : x1 + dx] = image[..., y0:y1, x0:x1]
    flow = provider._pair(image, source)
    target = flow.new_tensor([dx, dy])[None, :, None, None]
    mask = torch.zeros_like(flow[:, :1], dtype=torch.bool)
    mask[..., y0:y1, x0:x1] = True
    if valid is not None:
        mask &= valid.all(1, keepdim=True)
    count = mask.sum() * 2
    if not bool(count):
        return flow.sum() * 0
    return ((flow - target).abs() * mask).sum() / count
