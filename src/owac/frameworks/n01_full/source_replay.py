"""Whole-frame deletion and frozen-decision readout on an already bounded window."""

from __future__ import annotations

from dataclasses import replace

import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01_full.adapters import CameraTokens
from owac.frameworks.n01_full.tensor_gls import conservative_source_mask


def delete_sources(model, state, camera, source_ids, selected=None):
    buffer = state.cameras[camera]
    batch = buffer.batch_size
    selected = (
        torch.ones(batch, dtype=torch.bool, device=buffer.values.device)
        if selected is None
        else torch.as_tensor(selected, device=buffer.values.device, dtype=torch.bool)
    )
    ids = torch.as_tensor(sorted(set(int(x) for x in source_ids)), device=buffer.values.device)
    removed = (
        (buffer.frame_ids[..., None] == ids).any(-1)
        if ids.numel()
        else torch.zeros_like(buffer.observed)
    )
    keep = buffer.observed & ~(removed & selected[None])
    updates = {}
    for key in ("values", "variance", "valid", "observed", "frame_ids", "timestamps", "age_steps"):
        value = getattr(buffer, key)
        members = []
        for member in range(batch):
            retained = value[:, member][keep[:, member]]
            fill = -1 if key == "frame_ids" else 0
            tail = torch.full(
                (buffer.num_slots - len(retained), *retained.shape[1:]),
                fill,
                dtype=value.dtype,
                device=value.device,
            )
            members.append(torch.cat([retained, tail], 0))
        updates[key] = torch.stack(members, 1)
    updates["filled"] = keep.sum(0)
    updates["rng_index"] = updates["filled"] % buffer.num_slots
    cleaned = replace(buffer, **updates)
    clean_state = replace(
        state,
        cameras={**state.cameras, camera: cleaned},
        tokens={k: v for k, v in state.tokens.items() if k != camera},
        support={k: v for k, v in state.support.items() if k != camera},
        correspondence={k: v for k, v in state.correspondence.items() if k != camera},
    )
    if not bool(cleaned.observed.any()):
        from owac.core.encoder_contract import EncoderOutput

        height, width = cleaned.shape
        features = cleaned.values.new_zeros(batch, model.config.feature_width, height, width)
        available = torch.zeros(
            batch, 1, height, width, dtype=torch.bool, device=cleaned.values.device
        )
        pooled, present, positions = model.encoder._masked_pool(features, available)
        entry = CameraTokens(pooled, present, positions, 0, cleaned.frame_ids[0])
        clean_state = replace(clean_state, tokens={**clean_state.tokens, camera: entry})
        return EncoderOutput(
            representation={camera: pooled},
            state=clean_state,
            validity={camera: present},
            diagnostics={
                "recompute_scope": (
                    "counterfactual_all_dependencies_from_surviving_retained_sources"
                ),
                "deleted_source_ids": sorted(set(source_ids)),
                "new_independent_observations": 0,
                "unavailable_camera": camera,
            },
        )
    if model.support is not None and not hasattr(model.support, "replay_window"):
        from owac.frameworks.n01 import state as measurement_state

        blank = measurement_state.init_state(
            model.encoder.buffer_config,
            batch,
            {"camera_ids": [camera], "shape": cleaned.shape},
            cleaned.values.device,
        )
        replay = replace(
            clean_state,
            cameras={**clean_state.cameras, camera: blank},
            arrivals={
                **state.arrivals,
                camera: torch.zeros(batch, dtype=torch.int64, device=cleaned.values.device),
            },
            unique_arrivals={
                **state.unique_arrivals,
                camera: torch.zeros(batch, dtype=torch.int64, device=cleaned.values.device),
            },
            step_index=0,
        )
        for slot in range(int(cleaned.filled.max()) - 1, -1, -1):
            frame = MeasurementFrame(
                cleaned.values[slot],
                valid_mask=cleaned.valid[slot],
                frame_ids=cleaned.frame_ids[slot],
                camera_id=camera,
                timestamp=cleaned.timestamps[slot],
                new_measurement=cleaned.observed[slot],
                metadata={"variance": cleaned.variance[slot]},
            )
            output = model.encoder.step(
                frame,
                state=replay,
                budget=ComputeBudget(max_updates=1, max_history=model.config.window_frames),
            )
            replay = output.state
        if bool(cleaned.observed.any()):
            output.diagnostics["recompute_scope"] = (
                "historical_sequential_replay_of_retained_survivors"
            )
            return output
    # The readout request is not another arrival. Source timestamps/values remain
    # the physically retained samples; no older sample exists here to backfill.
    metadata = {}
    if model.config.observation is not None:
        metadata["observation_contract"] = {
            **model.config.observation.to_dict(),
            "domain": model.config.spec.input_domain,
        }
    metadata["variance"] = cleaned.variance[0]
    frame = MeasurementFrame(
        cleaned.values[0],
        valid_mask=cleaned.valid[0],
        camera_id=camera,
        frame_ids=cleaned.frame_ids[0],
        timestamp=cleaned.now,
        new_measurement=torch.zeros(batch, dtype=torch.bool, device=cleaned.values.device),
        metadata=metadata,
    )
    output = model.encoder.step(
        frame,
        state=clean_state,
        budget=ComputeBudget(max_updates=1, max_history=model.config.window_frames),
    )
    output.state = replace(
        output.state,
        arrivals=state.arrivals,
        unique_arrivals=state.unique_arrivals,
        step_index=state.step_index,
    )
    output.diagnostics.update(
        recompute_scope="counterfactual_all_dependencies_from_surviving_retained_sources",
        deleted_source_ids=sorted(set(source_ids)),
        new_independent_observations=0,
    )
    return output


def fixed_decision_readout(model, state, camera, source_ids):
    buffer = state.cameras[camera]
    length = int(buffer.filled.max())
    values, mask = model.encoder._window(buffer, length)
    offsets = state.correspondence[camera].offsets_xy
    reference_mask = None
    if camera in state.support and hasattr(model.support, "replay_window"):
        decision = state.support[camera]["state"]
        valid = buffer.observed[:length].T
        usable = model.support.expand_to_cells(decision, valid[:, 1:], *values.shape[-2:])
        reference_mask = mask.clone()
        reference_mask[:, 1:] &= usable[:, :, None]
        mask = conservative_source_mask(reference_mask, mask, offsets)
    ids = torch.as_tensor(sorted(set(source_ids)), device=buffer.values.device)
    dropped = (
        (buffer.frame_ids[:length, :, None] == ids).any(-1).T
        if ids.numel()
        else torch.zeros(values.shape[:2], device=values.device, dtype=torch.bool)
    )
    mask = mask & ~dropped[:, :, None, None, None]
    if reference_mask is not None:
        reference_mask = reference_mask & ~dropped[:, :, None, None, None]
    fused, available, _ = model.encoder.encode_window(
        values, mask, offsets, reference_mask=reference_mask
    )
    pooled, present, positions = model.encoder._masked_pool(fused, available)
    tokens = CameraTokens(pooled, present, positions, length, buffer.frame_ids[0])
    fixed_state = replace(state, tokens={**state.tokens, camera: tokens})
    from owac.core.encoder_contract import EncoderOutput

    return EncoderOutput(
        representation={camera: pooled},
        state=fixed_state,
        validity={camera: present},
        diagnostics={"recompute_scope": "fixed_motion_noise_support_decisions_direct_content_only"},
    )
