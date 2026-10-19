"""Shared causal-window execution for training and evaluation.

Each row is one time step, not an independent batch example. Training retains
a configurable truncated recurrent graph; evaluation replays the same values.
No future frame enters the prefix state.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

EXECUTION_REVISION = "n02-v12-p2-spatial-r7"


def snapshot(value: Any) -> Any:
    """Detached, owned state, including dataclasses with tensor payloads."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value, **{f.name: snapshot(getattr(value, f.name)) for f in dataclasses.fields(value)}
        )
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


def select_observation(observation: Any, rows: list[int]) -> Any:
    total = int(next(iter(observation.frames.values())).values.shape[0])

    def select(value: Any) -> Any:
        if torch.is_tensor(value) and value.ndim and value.shape[0] == total:
            return value[torch.tensor(rows, device=value.device)]
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == total:
            return value[rows]
        if isinstance(value, (tuple, list)) and len(value) == total:
            return type(value)(value[row] for row in rows)
        return value

    def record(value: Any) -> Any:
        return dataclasses.replace(
            value, **{f.name: select(getattr(value, f.name)) for f in dataclasses.fields(value)}
        )

    return dataclasses.replace(
        observation,
        frames={camera: record(frame) for camera, frame in observation.frames.items()},
        context=record(observation.context),
        episode_ids=select(observation.episode_ids),
        anchor_ids=select(observation.anchor_ids),
        new_time_mask=select(observation.new_time_mask),
    )


def select_spatial(spatial: Any, row: int) -> Any:
    updates = {
        name: {key: value[row : row + 1] for key, value in getattr(spatial, name).items()}
        for name in ("levels", "coords", "valid")
    }
    if hasattr(spatial, "support_ids"):
        updates["support_ids"] = {k: v[row : row + 1] for k, v in spatial.support_ids.items()}
    return dataclasses.replace(spatial, **updates)


def window_inputs(batch: Any, anchor: int = 0) -> list[Any]:
    from owac.frameworks.n02_full.inference_v12 import V12InferenceInput

    history = int(getattr(batch, "history", len(batch.frame_index)))
    indices = np.asarray(batch.frame_index).reshape(-1)
    if history < 1 or len(indices) % history or not 0 <= anchor < len(indices) // history:
        raise ValueError("invalid causal window history/anchor dimensions")
    window_indices = indices[anchor * history : (anchor + 1) * history]
    if np.any(np.diff(window_indices) < 0):
        raise ValueError("causal window frame indices must be nondecreasing")
    prior = getattr(batch, "declared_prior", None)
    result = []
    seen = set()
    for row in range(anchor * history, (anchor + 1) * history):
        obs = select_observation(batch.observation, [row])
        identity = tuple(
            (c, tuple(f.episode_ids), str(f.frame_ids), int(indices[row]))
            for c, f in obs.frames.items()
        )
        if identity in seen:
            continue  # left-padding is one acquisition, not several observations
        seen.add(identity)
        selected_prior = prior
        if prior is not None and torch.as_tensor(prior).ndim == 2:
            selected_prior = prior[anchor : anchor + 1]
        result.append(V12InferenceInput(obs, selected_prior, [int(indices[row])], None))
    return result


def replay_inputs(
    model: Any, inputs: list[Any], *, grad_anchor: bool = False, bptt_steps: int = 2
) -> dict:
    if not inputs:
        raise ValueError("a causal window must contain a measurement")
    if bptt_steps < 1:
        raise ValueError("bptt_steps must be positive")
    first_grad = max(0, len(inputs) - bptt_steps)
    device = next(model.parameters()).device
    state = model.build_state(batch=1, device=device)
    outputs, pre_states = [], []
    for index, inp in enumerate(inputs):
        pre_states.append(state)
        with torch.set_grad_enabled(grad_anchor and index >= first_grad):
            inference = getattr(model, "inference", None)
            prefix_only = index < len(inputs) - 1 and getattr(
                inference, "revision_anchor_only", False
            )
            # Calibration and deployment lock the same unrevised causal prefix.
            # Only the current decision may consume its verdict and revise.
            if prefix_only and getattr(inference, "fast_prefix", False):
                output = inference.step(inp, state, emit_readout=False, compute_causal=False)
            else:
                output = inference.step(inp, state) if prefix_only else model.step(inp, state)
        outputs.append(output)
        state = snapshot(output.state) if index < first_grad else output.state
    return {"rows": outputs, "pre_states": pre_states, "inputs": inputs, "state": state}


def subset_queries(queries: Any, rows: torch.Tensor):
    """``QuerySet`` restricted to ``rows`` (same measurement space, fewer decision rows)."""
    from owac.frameworks.n02_full.inference_v12 import QuerySet

    def take_ids(value: Any) -> Any:
        if not value:
            return value
        index = rows.tolist()
        return tuple(value[int(item)] for item in index)

    return QuerySet(
        coords=queries.coords.index_select(0, rows),
        camera_index=queries.camera_index.index_select(0, rows),
        query_channel=queries.query_channel.index_select(0, rows),
        target=queries.target.index_select(0, rows),
        masks={name: value.index_select(0, rows) for name, value in queries.masks.items()},
        ids=take_ids(queries.ids),
        camera_ids=queries.camera_ids,
        sampled_per_camera=queries.sampled_per_camera,
        coverage=queries.coverage,
        duplicates_dropped=queries.duplicates_dropped,
        channel_reduction=queries.channel_reduction,
        per_camera_query_count=queries.per_camera_query_count,
    )


def training_anchor_order(count: int, seed: int, cursor: int, batch_size: int = 1) -> list[int]:
    """Seeded epoch permutation; cursor counts consumed anchors, including padding skips."""
    selected = []
    for offset in range(batch_size):
        absolute = cursor * batch_size + offset
        epoch, position = divmod(absolute, count)
        selected.append(
            int(
                np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)[
                    position
                ]
            )
        )
    return selected
