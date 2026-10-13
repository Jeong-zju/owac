"""N01 spatial-support intervention on the full, unchanged DBSR parameter set.

Only transport through existing convolutions changes. No learned network is
added. Recompute rebuilds arrived RAW features under the latest support graph;
the stale-cache control keeps the graph used when each source arrived. These
are spatial support interventions, not temporal history selection.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from owac.baselines._vendor.dbsr.blocks import ResBlock
from owac.baselines.burst_features import CFA_PHASES, align_features
from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.state import init_state, push
from owac.frameworks.n01.support import SupportDomainIntegrator

MODES = ("none", "static", "dynamic_stale", "dynamic_recompute")


def graph_sequence(values, valid, variance, config):
    """Reuse N01's measured pair gates and hysteresis, once per unique arrival.

    Inputs are detached RAW measurements in acquisition coordinates. Caller owns
    sequence reset; each member has independent state. No target/visibility/event
    metadata is accepted. Original N01 state and support source remain unchanged.
    """
    if config.learnable_gate:
        raise ValueError("this isolated mechanism uses analytic support, with no extra network")
    b, t, c, h, w = values.shape
    if c != 4 or valid.shape != values.shape or variance.shape != values.shape:
        raise ValueError("four-plane values, validity and variance must match")
    if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
        raise ValueError("declared variance must be finite and positive")
    integrator = SupportDomainIntegrator(config)
    state = init_state(config, b, {"camera_ids": ["burst"], "shape": (h, w)}, values.device)
    graphs, diagnostics = [], []
    with torch.no_grad():
        for now in range(t):
            frame = MeasurementFrame(
                values=values[:, now].detach(),
                valid_mask=valid[:, now],
                sample_xy=CFA_PHASES,
                camera_id="burst",
                frame_ids=torch.full((b,), now, device=values.device),
                timestamp=torch.full((b,), float(now), device=values.device),
                metadata={"variance": variance[:, now]},
            )
            state = push(state, frame)
            plane = state.plane_support("burst")
            evidence, _ = integrator.edge_field(frame, state)
            current = state.valid[0]
            measured = torch.stack(
                [
                    current & F.pad(current[..., :, :-1], (1, 0)),
                    current & F.pad(current[..., :-1, :], (0, 0, 1, 0)),
                ],
                2,
            )
            strength, age, merges, splits = integrator.update_edges(
                plane.edges, plane.support_age, evidence, measured, plane.open_edges
            )
            opened = integrator.last_open_edges
            state.support["burst"] = replace(
                plane,
                edges=strength,
                open_edges=opened,
                support_age=age,
                merge_count=plane.merge_count + merges,
                split_count=plane.split_count + splits,
            )
            graphs.append(opened.detach().clone())
            diagnostics.append(
                {
                    "merge_count_per_member": merges.cpu().tolist(),
                    "split_count_per_member": splits.cpu().tolist(),
                    "plane_open_fraction": float(opened.float().mean()),
                    "shared_open_fraction": float(opened.all(1).float().mean()),
                    "unique_source_ids": state.frame_ids.T.cpu().tolist(),
                }
            )
    return torch.stack(graphs, 1), diagnostics


def kernel_support(graph, integrator):
    """N01's original local path rule, intersected across CFA phases.

    Intermediate DBSR channels mix all four phases; the shared graph therefore
    permits a transport only if every phase permits the path. This conservative
    projection can reduce usable context and is explicitly part of this port.
    """
    if graph.ndim != 5 or graph.shape[2] != 2 or graph.dtype != torch.bool:
        raise ValueError("boolean graph [B,P,2,H,W] required")
    strength = graph.float()
    return torch.stack(
        [
            integrator.open_path(strength, row, col).all(1)
            for row in (-1, 0, 1)
            for col in (-1, 0, 1)
        ],
        1,
    )


def supported_forward(layer, values, gates):
    """Execute original DBSR layers, masking each 3x3 convolution's input taps.

    Weights/biases/residuals/activations are the published modules themselves.
    No renormalization or replacement parameters are introduced. Spatial/global
    normalization is rejected because it would reintroduce cross-graph mixing.
    """
    if isinstance(layer, nn.Conv2d):
        if layer.kernel_size == (1, 1):
            return layer(values)
        if (
            layer.kernel_size != (3, 3)
            or layer.padding != (1, 1)
            or layer.stride != (1, 1)
            or layer.dilation != (1, 1)
            or layer.groups != 1
            or layer.padding_mode != "zeros"
        ):
            raise ValueError("support adapter requires the original dense DBSR 3x3 convolutions")
        b, c, h, w = values.shape
        if gates.shape != (b, 9, h, w):
            raise ValueError("kernel support must match the input grid")
        patches = F.unfold(values, 3, padding=1).reshape(b, c, 9, h * w)
        patches = patches * gates.reshape(b, 1, 9, h * w).to(values.dtype)
        patches = patches.reshape(b, c * 9, h * w).transpose(1, 2)
        result = F.linear(patches, layer.weight.flatten(1), layer.bias)
        return result.transpose(1, 2).reshape(b, layer.out_channels, h, w)
    if isinstance(layer, nn.Sequential):
        for child in layer:
            values = supported_forward(child, values, gates)
        return values
    if isinstance(layer, ResBlock):
        if layer.downsample is not None or layer.attention is not None:
            raise ValueError("unsupported residual block variant")
        residual = supported_forward(layer.conv1, values, gates)
        residual = supported_forward(layer.conv2, residual, gates)
        return layer.activation(residual + values)
    if isinstance(layer, (nn.ReLU, nn.Identity)):
        return layer(values)
    raise TypeError(f"layer {type(layer).__name__} has no audited support-constrained execution")


def supported_backbone(backbone, values, gates):
    layers = backbone.upstream
    result = supported_forward(layers.init_layer, values, gates)
    result = supported_forward(layers.res_layers, result, gates)
    return supported_forward(layers.out_layer, result, gates)


def supported_fusion(fusion, features, valid, offsets, gates):
    """Apply support to weight prediction as well as feature formation."""
    b, t, c, h, w = features.shape
    source_gates = gates[:, None].expand(-1, t, -1, -1, -1).flatten(0, 1)
    features = features.masked_fill(~valid, 0)
    projected = (
        supported_forward(fusion.feat_project_layer, features.flatten(0, 1), source_gates)
        .reshape(b, t, -1, h, w)
        .masked_fill(~valid, 0)
    )
    consensus = projected.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
    base = (
        torch.where(valid[:, :1], projected[:, :1], consensus)
        if fusion.use_base_frame
        else consensus
    )
    difference = (projected - base).masked_fill(~valid, 0)
    offset_features = supported_forward(
        fusion.offset_feat_extractor,
        offsets.remainder(fusion.offset_modulo).flatten(0, 1),
        source_gates,
    )
    weight_input = torch.cat(
        [base.expand(-1, t, -1, -1, -1).flatten(0, 1), difference.flatten(0, 1), offset_features], 1
    )
    logits = supported_forward(fusion.weight_predictor, weight_input, source_gates).reshape(
        b, t, c, h, w
    )
    available = valid.any(1, keepdim=True)
    logits = torch.where(
        available, logits.masked_fill(~valid, -torch.inf), torch.zeros_like(logits)
    )
    weights = logits.softmax(1).masked_fill(~valid, 0)
    return (features * weights).sum(1), weights


def support_config(overrides=None):
    return N01Config.tiny(
        input_mode="bayer_cfa",
        channels=4,
        planes=4,
        learnable_gate=False,
        selection="analytic",
        spatial_radii=(0, 1, 2),
        time_depths=(1, 4, 8),
        aux_measurement_prediction=False,
        **(overrides or {}),
    )


def encode_window(model, values, valid, gates, *, chunk=16):
    """Rebuild features from RAW using the current graph, with bounded batches."""
    b, t, c, h, w = values.shape
    flat = values.masked_fill(~valid, 0).flatten(0, 1)
    shared = gates[:, None].expand(-1, t, -1, -1, -1).flatten(0, 1)
    rows = [
        supported_backbone(
            model.encoder.backbone, flat[start : start + chunk], shared[start : start + chunk]
        )
        for start in range(0, b * t, chunk)
    ]
    centers = valid.all(2, keepdim=True)
    return torch.cat(rows).reshape(b, t, -1, h, w).masked_fill(~centers, 0), centers


def sequence_features(model, batch, mode, config, *, freeze_frame=8, encode_chunk=16):
    """Yield causal representations and diagnostics for an independent episode.

    Oracle displacement aligns RAW before feature formation in ALL intervention
    arms. Ordinary conditions have no displacement. Thus the unmodified arm is
    exactly the ordinary baseline, while the oracle has a disclosed alignment-
    order control and must not be compared numerically with the previous oracle.
    """
    if mode not in MODES:
        raise ValueError("unknown support intervention")
    if model.encoder.fusion is None:
        raise ValueError("the intervention requires the mature DBSR fusion predictor")
    device = next(model.parameters()).device
    values, valid = batch.values.to(device), batch.valid.to(device)
    b, t, _, h, w = values.shape
    graphs, records = graph_sequence(batch.values, batch.valid, batch.variance, config)
    integrator = SupportDomainIntegrator(config)
    # The graph is a detached measurement decision. Compute all local path
    # masks on its acquisition device before moving them to the GPU, avoiding
    # device synchronization in the original path walk for every convolution.
    kernels = kernel_support(graphs.flatten(0, 1), integrator).reshape(b, t, 9, h, w)
    kernels = kernels.to(device)
    cached = []
    for now in range(t):
        graph = graphs[:, now]
        if mode == "none":
            graph = torch.ones_like(graph)
        elif mode == "static":
            graph = graphs[:, min(now, freeze_frame)]
        gates = kernels[:, min(now, freeze_frame) if mode == "static" else now]
        start = max(0, now + 1 - model.encoder.config.max_history)
        raw, source_valid = values[:, start : now + 1].flip(1), valid[:, start : now + 1].flip(1)
        offsets = raw.new_zeros(b, raw.shape[1], 2, h, w)
        if batch.condition == "translation_oracle_alignment":
            translations = batch.translations.to(device)
            delta = translations[:, start : now + 1].flip(1) - translations[:, now : now + 1]
            offsets = delta.flip(-1)[:, :, :, None, None].expand(-1, -1, -1, h, w)
            raw, _, _ = align_features(raw, source_valid.all(2, keepdim=True), offsets)
            _, aligned_centers, _ = align_features(
                source_valid.float(), source_valid.all(2, keepdim=True), offsets
            )
            source_valid = aligned_centers.expand_as(raw)
        if mode == "none":
            features, centers = model.encoder.encode_frames(raw, source_valid)
            representation, _, weights = model.encoder.fuse_encoded(features, centers)
        elif mode == "dynamic_stale":
            # Arrival encoding uses only that frame's graph. New splits cannot
            # remove dependencies already embedded in an older cached feature.
            current, _ = encode_window(
                model, values[:, now : now + 1], valid[:, now : now + 1], gates, chunk=encode_chunk
            )
            cached.append(current[:, 0])
            features = torch.stack(cached[start : now + 1][::-1], 1)
            original_valid = valid[:, start : now + 1].flip(1).all(2, keepdim=True)
            features, centers, _ = align_features(features, original_valid, offsets)
            representation, weights = supported_fusion(
                model.encoder.fusion, features, centers, offsets, gates
            )
        else:
            features, centers = encode_window(model, raw, source_valid, gates, chunk=encode_chunk)
            representation, weights = supported_fusion(
                model.encoder.fusion, features, centers, offsets, gates
            )
        yield representation, weights, records[now], graph
