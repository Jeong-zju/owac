"""Bridge adapter: the frozen N02 v1.2 chain as one pi0.5 ``TokenSource``.

``N02V12TokenSource`` is the only seam between the v1.2 chain
(``RawEncoderV12`` -> ``N02V12Inference`` -> ``N02ReadoutV12``, composed by
``model_v12.N02V12Model``) and the existing feature bridge
(``owac.backends.rawvla_features.RawVLAFeaturePolicy``).

Contract, copied from :class:`owac.backends.rawvla_features.TokenSource`:

* exactly one ``__call__`` per policy forward - never one per denoising step.  One call runs
  :meth:`N02V12Inference.step` **once** for the whole joint multi-camera decision moment and
  returns ``(rows, n_cameras * 256, 2048)``; it never loops per camera and never picks a
  per-camera MAP candidate, so the K candidate explanations stay the joint multi-camera ones
  the chain produced.
* ``calls``/``resets`` counters, ``reset(batch_mask)``, ``detach()``, ``states`` and
  ``trainable_records()`` behave as the bridge expects.
* **no detach happens before the loss**: the returned tokens are the readout output of the
  recorded step and keep their autograd history, so the action objective reaches the encoder,
  the candidates, the decoder and the readout.  ``detach()`` is only for callers that
  deliberately cut the recurrent state between steps (an online rollout), never for training.

Window batches execute chronologically, carrying state across distinct observations.
Training keeps the last two distinct recurrent steps differentiable by default.
A bridge call still builds one language/action prefix regardless of denoising iterations.
"""

from __future__ import annotations

from typing import Any

import torch

from owac.backends.rawvla_features import TokenSource

__all__ = ["N02V12TokenSource", "anchor_row_indices"]


def anchor_row_indices(rows: int, history: int) -> list[int]:
    """Row indices of the current measurement, one per anchor, in batch order.

    ``rows`` is the number of causal rows of one loader window (``anchors * history``).  A
    bundle that is not an exact multiple of ``history`` is not a loader window: it is treated
    as one decision moment per row and said so by returning every index, instead of silently
    dropping rows that no anchor claims.
    """
    rows = int(rows)
    history = int(history)
    if rows <= 0:
        raise ValueError(f"rows must be >= 1, got {rows}")
    if history <= 1 or rows % history:
        return list(range(rows))
    return [anchor * history + history - 1 for anchor in range(rows // history)]


def _frame_rows(bundle: Any) -> int:
    """Number of batch rows of an ``ObservationBundle`` (row count, not camera count)."""
    episode_ids = getattr(bundle, "episode_ids", ())
    if episode_ids:
        return int(len(episode_ids))
    frames = getattr(bundle, "frames", None) or {}
    for frame in frames.values():
        values = torch.as_tensor(frame.values)
        if values.ndim >= 4:
            return int(values.shape[0])
    if frames:
        return 1
    raise ValueError("observation bundle carries no camera frames")


def _split_observation(observation: Any) -> tuple[Any, Any, Any, Any]:
    """``(bundle, declared_prior, frame_index, measurement_ids)`` from the accepted shapes."""
    from owac.frameworks.n02_full.inference_v12 import V12InferenceInput
    from owac.frameworks.n02_full.state import InferenceInput, ObservationBundle

    if isinstance(observation, V12InferenceInput):
        return (
            observation.observation,
            observation.declared_prior,
            observation.frame_index,
            observation.measurement_ids,
        )
    if isinstance(observation, InferenceInput):
        return observation.bundle, observation.declared_prior, None, None
    if isinstance(observation, ObservationBundle):
        return observation, None, None, None
    if isinstance(observation, dict):
        for key in ("observation", "bundle"):
            if key in observation:
                return _split_observation(observation[key])
        frames = observation.get("frames")
        if frames is None:
            raise ValueError("observation dict carries neither 'observation' nor 'frames'")
        context = observation.get("context")
        bundle = ObservationBundle(
            frames=dict(frames) if isinstance(frames, dict) else frames,
            context=context,
        )
        return bundle, observation.get("declared_prior"), observation.get("frame_index"), None
    # ``data_v12.V12WindowBatch`` and anything else exposing an ObservationBundle.
    bundle = getattr(observation, "observation", None)
    if isinstance(bundle, ObservationBundle):
        return (
            bundle,
            getattr(observation, "declared_prior", None),
            getattr(observation, "frame_index", None),
            getattr(observation, "measurement_ids", None),
        )
    raise TypeError(
        "N02V12TokenSource needs an ObservationBundle, a V12InferenceInput, one decision "
        f"moment batch, or a dict carrying one of them; got {type(observation).__name__}"
    )


class N02V12TokenSource(TokenSource):
    """One ``N02V12Model.step`` per call, returning the current-row tokens for pi0.5.

    Args:
        model: the composed v1.2 model (``model_v12.N02V12Model``).
        camera_ids: camera identity of each token group, in the chain's own camera order.  The
            bridge places groups by this tuple, so it must equal the loader/encoder order.
        history: causal rows per decision moment used to pick the current row.
        state: optional starting ``SequenceState`` for online current-observation bundles.
            Explicit loader windows always start fresh and replay their own causal history.
    """

    def __init__(
        self,
        model: Any,
        *,
        camera_ids: Any,
        history: int = 8,
        state: Any = None,
        device: Any = None,
        strict: bool = True,
        bundle_is_current: bool = True,
        bptt_steps: int = 2,
    ) -> None:
        super().__init__()
        self.model = model
        self.camera_ids = tuple(str(item) for item in camera_ids)
        if not self.camera_ids:
            raise ValueError("camera_ids must name at least one camera")
        self.history = int(history)
        self.device_hint = device
        self.strict = bool(strict)
        # Bare bundles contain current independent rows. Explicit WindowBatch carries history.
        # False retains compatibility only for callers explicitly requesting legacy slicing.
        self.bundle_is_current = bool(bundle_is_current)
        self.state = state
        self.bptt_steps = int(bptt_steps)
        if self.bptt_steps < 1:
            raise ValueError("bptt_steps must be positive")
        self.states: dict[str, Any] = {}
        self.last_output: Any = None
        self.last_outputs: list[Any] = []
        self.last_anchor_rows: list[int] = []
        self.last_batch_rows = 0

    # -- TokenSource ---------------------------------------------------------------------

    def __call__(self, observation: Any) -> Any:
        from owac.core.encoder_contract import VisualTokens
        from owac.frameworks.n02_full.inference_v12 import V12InferenceInput

        if hasattr(observation, "history") and hasattr(observation, "frame_index"):
            from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

            anchors = len(observation.frame_index) // int(observation.history)
            replays = [
                replay_inputs(
                    self.model,
                    window_inputs(observation, anchor),
                    grad_anchor=torch.is_grad_enabled(),
                    bptt_steps=self.bptt_steps,
                )
                for anchor in range(anchors)
            ]
            self.last_outputs = [replay["rows"][-1] for replay in replays]
            self.last_output = self.last_outputs[0]
            current = torch.cat([output.tokens for output in self.last_outputs])
            self.last_batch_rows = len(observation.frame_index)
            self.last_anchor_rows = anchor_row_indices(self.last_batch_rows, self.history)
            self.states = {str(i): output.state for i, output in enumerate(self.last_outputs)}
            self.state = None  # independent windows have independent episode state
            self.calls += 1
            expected = (anchors, len(self.camera_ids) * 256, 2048)
            if tuple(current.shape) != expected or not bool(torch.isfinite(current).all()):
                raise ValueError(
                    f"invalid window tokens {tuple(current.shape)}, expected {expected}"
                )
            return VisualTokens(
                current,
                torch.ones(current.shape[:2], dtype=torch.bool, device=current.device),
                self.camera_ids,
            )

        bundle, declared_prior, frame_index, measurement_ids = _split_observation(observation)
        rows = _frame_rows(bundle)
        state = self.state
        if state is None:
            state = self.model.build_state(batch=rows, device=self._resolve_device())
        inference_input = V12InferenceInput(
            observation=bundle,
            declared_prior=declared_prior,
            frame_index=frame_index,
            measurement_ids=measurement_ids,
        )
        output = self.model.step(inference_input, state)
        indices = (
            list(range(rows)) if self.bundle_is_current else anchor_row_indices(rows, self.history)
        )
        tokens = output.tokens
        if tokens.ndim != 3 or int(tokens.shape[0]) != rows:
            raise ValueError(
                "the v1.2 readout must return one token group per batch row: got "
                f"{tuple(tokens.shape)} for {rows} rows"
            )
        current = tokens[torch.as_tensor(indices, device=tokens.device)]
        expected = (len(indices), len(self.camera_ids) * 256, 2048)
        if tuple(current.shape) != expected:
            raise ValueError(
                f"visual tokens must be {expected} (decisions x cameras*256 x 2048); got "
                f"{tuple(current.shape)}"
            )
        if self.strict and not bool(torch.isfinite(current).all()):
            raise ValueError("the v1.2 readout produced non-finite tokens")

        self.calls += 1
        self.last_output = output
        self.last_outputs = [output]
        self.last_anchor_rows = indices
        self.last_batch_rows = rows
        self.states = {"latest": output.state}
        self.state = output.state  # explicit windows reset above; online bundles retain state
        return VisualTokens(
            values=current,
            mask=torch.ones(
                (len(indices), current.shape[1]), dtype=torch.bool, device=current.device
            ),
            camera_ids=self.camera_ids,
            positions=None,
        )

    def reset(self, batch_mask: Any = None) -> dict:
        """Row-wise episode reset; without a state yet this only records the intent."""
        self.resets += 1
        state = self.state if self.state is not None else self.states.get("latest")
        episodes = [] if batch_mask is None else [bool(item) for item in batch_mask]
        if state is None:
            return {"reset": False, "episodes": episodes, "reason": "no v1.2 state yet"}
        self.state = self.model.reset(batch_mask, state)
        self.states = {"latest": self.state}
        self.last_output = None
        return {"reset": True, "episodes": episodes}

    def detach(self) -> list:
        """Detach the recurrent state from the autograd graph (online rollouts only)."""
        detached: list[str] = []
        state = self.state if self.state is not None else self.states.get("latest")
        if state is not None:
            state = state.detach() if hasattr(state, "detach") else _detach_state(state)
            self.state = state
            self.states = {"latest": state}
            detached.append("latest")
        self.last_output = None
        return detached

    def trainable_records(self) -> list:
        """One record per trainable parameter, in the bridge's own record schema.

        ``RawVLAFeaturePolicy.parameter_report`` reads ``parameters``/``shape``/``initialized``/
        ``dtype``/``requires_grad`` from these records, so the schema is copied exactly instead of
        summarised per role.
        """
        records = []
        for role, parameters in self.model.parameter_groups().items():
            trainable = [parameter for parameter in parameters if parameter.requires_grad]
            if not trainable:
                continue
            records.append(
                {
                    "name": f"token_source.model.{role}",
                    "parameters": sum(int(parameter.numel()) for parameter in trainable),
                    "shape": None,  # a role is a set of tensors, not one
                    "initialized": True,
                    "dtype": str(trainable[0].dtype),
                    "requires_grad": True,
                    "tensors": len(trainable),
                    "frozen_tensors": len(parameters) - len(trainable),
                }
            )
        return records

    # -- helpers -------------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        if self.device_hint is not None:
            return self.device_hint
        for parameter in self.model.parameters():
            return parameter.device
        return torch.device("cpu")


def _detach_state(state: Any) -> Any:
    """``SequenceState.detach()`` replacement for a state object without that method."""
    import dataclasses

    if dataclasses.is_dataclass(state):
        return dataclasses.replace(
            state,
            slot_hidden=state.slot_hidden.detach(),
            existence=state.existence.detach(),
        )
    return state
