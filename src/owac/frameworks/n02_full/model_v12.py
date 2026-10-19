"""N02 v1.2 model entry: one traceable object that owns the whole P2 chain.

``N02V12Model`` is the only main entry of the v1.2 line. It composes the frozen cross-module
interfaces (``encoder_v12`` / ``decoder_v12`` / ``readout_v12`` / ``checks_v12``, owned by the
parallel agents) with :class:`inference_v12.N02V12Inference` and exposes:

* ``step(input, state)``  one decision time, returning ``N02V12StepOutput``;
* ``reset(mask)``         row-wise episode reset;
* ``tokens(output)``      the ``(B, n_cameras*256, 2048)`` tokens the pi0.5 bridge consumes;
* ``rebuild``/``drop_measurements``/``snapshot``/``restore`` for state reconstruction and replay;
* ``parameter_groups()``  per-role parameters so a training run can report honest gradient paths.

This module also hosts the P1 defect-A fix used by ``model.py``: :class:`LockedAbsoluteCheck`
unifies the real three-state verdict (``checks_v12.AbsoluteCheckResult``) with the score object the
P1 step already produced, so a consumer that reads ``.verdict`` can never see the string ``None``
and a missing verdict is explicitly marked invalid instead of being counted as 0 false alarms /
0 detections.

New layers introduced here (disclosed, all initialized inside their own module): the candidate
latent seeds, evidence/prior/residual projections and the renewal projection live in
``inference_v12.N02V12Inference``; the backbone, RAW decoder, Perceiver readout, SAVi
corrector/predictor and the measurement/check layer are the modules owned by the other agents and
the P0 ports. No unsourced small network replaces a proven backbone.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from owac.frameworks.n02_full.inference_v12 import (
    CHECK_STATUSES,
    DEFAULT_K,
    DEFAULT_ROUNDS,
    DIFFERENTIABLE_PATH,
    INVALID_VERDICT,
    MAX_CAMERAS,
    NON_DIFFERENTIABLE_PATH,
    N02V12Inference,
    N02V12StepOutput,
    empty_v12_state,
)
from owac.frameworks.n02_full.sensor import MeasurementScore
from owac.frameworks.n02_full.state import SequenceState

__all__ = [
    "DEFAULT_ENTRY",
    "LockedAbsoluteCheck",
    "N02V12Model",
    "build_default_v12_model",
    "build_n02_v12_model",
    "invalid_check_result",
    "locked_absolute_check",
    "prediction_residuals",
]

DEFAULT_ENTRY = "n02_v12"


@dataclass(slots=True)
class _MissingCheck:
    """Placeholder result used only when ``checks_v12`` cannot be imported yet."""

    status: str
    valid: bool
    statistics: dict
    calibration_version: str


class LockedAbsoluteCheck(MeasurementScore):
    """A P1 ``MeasurementScore`` carrying a real three-state verdict.

    Subclassing keeps the historical P0/P1 consumers working (``isinstance(..., MeasurementScore)``,
    ``.kind``, ``.raw_nll_or_energy``) while ``.verdict``/``.status``/``.valid`` are the real check
    result. ``valid=False`` marks a missing or unusable check so statistics count it separately.
    """

    _SCORE_FIELDS = (
        "relative_terms",
        "raw_nll_or_energy",
        "local_stats",
        "kind",
        "per_valid_scalar",
    )

    def __init__(self, score: Any, result: Any) -> None:
        # Copy the score's fields in place instead of re-running MeasurementScore.__post_init__:
        # the P1 step accepts duck-typed scores (its own tests pass shapes this class would
        # re-validate differently), and the check must not change what the score already held.
        for name in self._SCORE_FIELDS:
            object.__setattr__(self, name, getattr(score, name))
        self.status = str(result.status)
        self.valid = bool(result.valid)
        self.statistics = dict(getattr(result, "statistics", {}) or {})
        self.calibration_version = str(getattr(result, "calibration_version", "uncalibrated"))
        self.check_kind = type(result).__name__

    @property
    def verdict(self) -> str:
        """Never ``"None"``: the real status when valid, an explicit ``invalid`` otherwise."""
        return self.status if self.valid else INVALID_VERDICT

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "status": self.status,
            "valid": self.valid,
            "calibration_version": self.calibration_version,
            "score_kind": self.kind,
            "statistics": {
                key: (value.detach() if torch.is_tensor(value) else value)
                for key, value in self.statistics.items()
            },
        }


def prediction_residuals(prediction: Any, raw: Any) -> torch.Tensor | None:
    """Standardised residuals ``(y - mean) / sqrt(variance)`` of one locked prediction.

    ``model.score`` already declares that a ``RawQuery`` carries the same units as the prediction,
    so the residual uses exactly those declared units. If the query and the prediction do not
    describe the same query set, this returns ``None`` and the check is marked invalid instead of
    inventing a residual.
    """
    mean = getattr(prediction, "mean", None)
    variance = getattr(prediction, "variance", None)
    values = getattr(raw, "values", None)
    if mean is None or variance is None or values is None:
        return None
    mean = torch.as_tensor(mean).reshape(-1)
    variance = torch.as_tensor(variance).reshape(-1)
    values = torch.as_tensor(values).reshape(-1)
    if mean.numel() != values.numel() or variance.numel() != values.numel():
        return None
    if mean.numel() == 0:
        return None
    return (values - mean) / variance.clamp_min(1e-12).sqrt()


def invalid_check_result(reason: str, *, calibration_version: str = "uncalibrated") -> Any:
    """An explicitly invalid ``AbsoluteCheckResult``: visible, never scored as a real verdict."""
    result_type = _absolute_check_result_type()
    return result_type(
        status="insufficient_information",
        valid=False,
        statistics={"reason": reason, "verdict": INVALID_VERDICT},
        calibration_version=calibration_version,
    )


def locked_absolute_check(
    *,
    score: MeasurementScore,
    residuals: Any = None,
    masks: Any = None,
    calibrator: Any = None,
    calibration_version: str = "uncalibrated",
    query_ids: Any = None,
    reason: str | None = None,
) -> LockedAbsoluteCheck:
    """Attach a real absolute check to a P1 score, or mark it explicitly invalid.

    A missing verdict must never reach a statistic as ``"None"`` or as 0 false alarms: whenever the
    check cannot be computed (no ``checks_v12``, no residuals, no calibrator) the wrapped result
    carries ``valid=False`` and a machine-readable reason.
    """
    if reason is not None:
        return LockedAbsoluteCheck(
            score, invalid_check_result(reason, calibration_version=calibration_version)
        )
    checker = _measurement_checker()
    if checker is None:
        return LockedAbsoluteCheck(
            score,
            invalid_check_result(
                "checks_v12 is unavailable", calibration_version=calibration_version
            ),
        )
    if residuals is None:
        return LockedAbsoluteCheck(
            score,
            invalid_check_result(
                "no residual field on the prediction", calibration_version=calibration_version
            ),
        )
    try:
        from owac.frameworks.n02_full.inference_v12 import _call_with_supported_kwargs

        result = _call_with_supported_kwargs(
            checker,
            "absolute_check",
            {
                "residuals": residuals,
                "masks": masks,
                "calibrator": calibrator,
                "query_ids": query_ids,
            },
        )
    except Exception as exc:  # noqa: BLE001 - report the failure, never hide it as a verdict
        return LockedAbsoluteCheck(
            score,
            invalid_check_result(
                f"absolute_check raised {type(exc).__name__}: {exc}",
                calibration_version=calibration_version,
            ),
        )
    status = getattr(result, "status", None)
    valid = getattr(result, "valid", None)
    if status is None or valid is None:
        return LockedAbsoluteCheck(
            score,
            invalid_check_result(
                f"absolute_check returned {type(result).__name__} without status/valid",
                calibration_version=calibration_version,
            ),
        )
    if str(status) not in CHECK_STATUSES:
        return LockedAbsoluteCheck(
            score,
            invalid_check_result(
                f"absolute_check returned unknown status {status!r}",
                calibration_version=calibration_version,
            ),
        )
    return LockedAbsoluteCheck(score, result)


def _absolute_check_result_type() -> Any:
    try:
        from owac.frameworks.n02_full import checks_v12

        return checks_v12.AbsoluteCheckResult
    except Exception:  # noqa: BLE001 - the parallel owner may not have landed yet
        return _MissingCheck


def _measurement_checker() -> Any:
    """A measurement object providing ``absolute_check``; checks_v12's module-level default."""
    try:
        from owac.frameworks.n02_full import checks_v12

        return getattr(checks_v12, "absolute_check", None)
    except Exception:  # noqa: BLE001 - not landed yet
        return None


class N02V12Model(nn.Module):
    """The single traceable v1.2 entry: joint candidates -> RAW objective -> support -> tokens."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        readout: nn.Module,
        measurement: Any,
        n_slots: int = 16,
        slot_dim: int = 512,
        tau: float = 1.0,
        rounds: int = DEFAULT_ROUNDS,
        k: int = DEFAULT_K,
        d_model: int = 512,
        evidence_dim: int | None = None,
        action_dim: int = 0,
        history: int = 8,
        query_stride: int = 1,
        max_queries_per_camera: int = 0,
        calibrator: Any = None,
        strict_check: bool = True,
    ) -> None:
        super().__init__()
        self.inference = N02V12Inference(
            encoder=encoder,
            decoder=decoder,
            readout=readout,
            measurement=measurement,
            n_slots=n_slots,
            slot_dim=slot_dim,
            tau=tau,
            rounds=rounds,
            k=k,
            d_model=d_model,
            evidence_dim=evidence_dim,
            action_dim=action_dim,
            history=history,
            query_stride=query_stride,
            max_queries_per_camera=max_queries_per_camera,
            calibrator=calibrator,
            strict_check=strict_check,
        )

    # -- main entry -----------------------------------------------------------------------

    def step(self, inference_input: Any, state: SequenceState) -> N02V12StepOutput:
        output = self.inference.step(inference_input, state)
        if getattr(self.inference, "revision_policy", "disabled") == "reset_explanations":
            output = self.inference.revise_if_contradicted(inference_input, state, output)
        return output

    def reset(self, mask: Any, state: SequenceState | None = None) -> SequenceState:
        return self.inference.reset(mask, state)

    def build_state(self, batch: int = 1, *, device: Any = None) -> SequenceState:
        return empty_v12_state(
            batch=batch,
            k=self.inference.k,
            n_slots=self.inference.n_slots,
            slot_dim=self.inference.slot_dim,
            device=device,
        )

    # -- bridge / state helpers -----------------------------------------------------------

    @staticmethod
    def tokens(output: N02V12StepOutput) -> torch.Tensor:
        """The ``(B, n_cameras*256, 2048)`` visual tokens for the pi0.5 feature bridge."""
        return output.tokens

    @staticmethod
    def support(output: N02V12StepOutput) -> torch.Tensor:
        """``(B,K)`` support weights; not a calibrated posterior."""
        return output.q

    def add_measurement(self, inference_input: Any, state: SequenceState) -> N02V12StepOutput:
        """Alias of :meth:`step` for the online loop: execute, then feed the next observation."""
        return self.step(inference_input, state)

    def rebuild(self, inference_input: Any, state: SequenceState, *, rows: Any = None):
        return self.inference.rebuild(inference_input, state, rows=rows)

    def drop_measurements(self, state: SequenceState, measurement_ids: Any, *, rows: Any = None):
        return self.inference.drop_measurements(state, measurement_ids, rows=rows)

    def snapshot(self, state: SequenceState) -> dict[str, Any]:
        return self.inference.snapshot(state)

    def restore(self, payload: Mapping[str, Any]) -> SequenceState:
        return self.inference.restore(dict(payload))

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return self.inference.parameter_groups()

    def gradient_report(self, loss: torch.Tensor) -> dict[str, dict[str, float]]:
        """Backward once and report per-role gradient L1 (finite, zero, non-zero counts)."""
        if loss.requires_grad:
            loss.backward()
        report: dict[str, dict[str, float]] = {}
        for role, parameters in self.parameter_groups().items():
            values = [
                parameter.grad.detach().abs()
                for parameter in parameters
                if parameter.grad is not None
            ]
            if values:
                flat = torch.cat([value.reshape(-1) for value in values])
                report[role] = {
                    "l1": float(flat.sum()),
                    "max": float(flat.max()),
                    "finite": float(torch.isfinite(flat).all().to(torch.float32)),
                    "tensors": float(len(values)),
                }
            else:
                report[role] = {"l1": 0.0, "max": 0.0, "finite": 1.0, "tensors": 0.0}
        return report

    def gradient_paths(self) -> dict[str, tuple[str, ...]]:
        """Honest gradient boundary of one decision time."""
        return {
            "differentiable": DIFFERENTIABLE_PATH,
            "hard_choice": NON_DIFFERENTIABLE_PATH,
            "note": (
                "the discrete candidate selection and the accept/reject decision are hard; their "
                "gradient is not the gradient of the search, only of the selected values"
            ),
        }


def configure_v12_runtime(config: dict, device: Any) -> None:
    """Shared precision policy for training and independently rebuilt evaluation."""
    if torch.device(device).type == "cuda":
        performance = config.get("runtime", {}).get("performance", {})
        torch.backends.cuda.matmul.allow_tf32 = bool(performance.get("tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(performance.get("tf32", True))
        torch.backends.cudnn.benchmark = bool(performance.get("cudnn_benchmark", False))


def build_n02_v12_model(config: Any = None, **overrides: Any) -> N02V12Model:
    """Build the v1.2 model from a config mapping/object using the frozen cross-module interfaces.

    The four components owned by the parallel agents are imported here, at call time, so importing
    ``model`` stays possible while those modules are still being written. Every keyword below is the
    constructor name the owning module actually declares.
    """
    settings = _settings(config)
    settings.update(overrides)
    from owac.frameworks.n02_full import checks_v12, decoder_v12, encoder_v12, readout_v12

    camera_ids = tuple(settings["camera_ids"])
    d_model = int(settings.get("d_model", 512))
    n_slots = int(settings.get("n_slots", 16))
    slot_dim = int(settings.get("slot_dim", 512))
    k = int(settings.get("k", DEFAULT_K))
    encoder = encoder_v12.RawEncoderV12(
        camera_ids=camera_ids,
        dims=tuple(settings.get("dims", (128, 256, 512, 1024))),
        depths=tuple(settings.get("depths", (3, 3, 27, 3))),
        in_chans=int(settings.get("in_chans", 3)),
        input_mode=str(settings.get("input_mode", "RAW3")),
        d_model=d_model,
        alignment=int(settings.get("alignment", 32)),
    )
    level_dims = dict(settings.get("level_dims") or encoder.spatial_dims)
    encoder.batch_cameras = bool(settings.get("batch_cameras", False))
    decoder = decoder_v12.RawMeasurementDecoderV12(
        in_dims=level_dims,
        slot_dim=slot_dim,
        n_slots=n_slots,
        n_candidates=k,
        hidden=int(settings.get("decoder_hidden", 256)),
        measurement_channels=int(settings.get("measurement_channels", 3)),
        channel_reduction=str(settings.get("channel_reduction", "select")),
        saturation_level=settings.get("saturation_level"),
        init_variance=float(settings.get("init_variance", 1.0)),
        units=str(settings.get("units", "sensor_units")),
        max_cameras=max(MAX_CAMERAS, len(camera_ids)),
    )
    readout = readout_v12.N02ReadoutV12(
        spatial_dims=level_dims,
        latent_dim=d_model,
        n_latents=int(settings.get("n_latents", 128)),
        n_queries=int(settings.get("n_queries", readout_v12.N02ReadoutV12.N_QUERIES)),
        out_dim=int(settings.get("out_dim", readout_v12.N02ReadoutV12.OUT_DIM)),
        state_dim=slot_dim,
        max_cameras=max(MAX_CAMERAS, len(camera_ids)),
        preserve_spatial_tokens=bool(settings.get("preserve_spatial_tokens", False)),
        output_token_rms=float(settings.get("output_token_rms", 4.0)),
        context_gate_init=float(settings.get("context_gate_init", 0.1)),
        scale_balance=bool(settings.get("scale_balance", False)),
        scale_balance_mode=str(settings.get("scale_balance_mode", "contrast")),
        learned_global_fusion=bool(settings.get("learned_global_fusion", False)),
    )
    decoder.group_camera_queries = bool(settings.get("group_camera_queries", False))
    measurement = checks_v12.N02V12Measurement(
        min_variance=settings.get("min_variance"),
        max_variance=settings.get("max_variance"),
    )
    calibrator = settings.get("calibrator")
    if calibrator is None:
        calibrator = getattr(measurement, "calibrator", None)
    model = N02V12Model(
        encoder=encoder,
        decoder=decoder,
        readout=readout,
        measurement=measurement,
        n_slots=n_slots,
        slot_dim=slot_dim,
        tau=float(settings.get("tau", 1.0)),
        rounds=int(settings.get("rounds", DEFAULT_ROUNDS)),
        k=k,
        d_model=d_model,
        evidence_dim=sum(level_dims.values()) * len(camera_ids),
        action_dim=int(settings.get("action_dim", 0)),
        history=int(settings.get("history", 8)),
        query_stride=int(settings.get("query_stride", 1)),
        max_queries_per_camera=int(settings.get("max_queries_per_camera", 0)),
        calibrator=calibrator,
        strict_check=bool(settings.get("strict_check", True)),
    )
    model.inference.checkpoint_decoder = bool(settings.get("checkpoint_decoder", False))
    revision = settings.get("revision_policy", "disabled")
    if revision not in ("disabled", "reset_explanations"):
        raise ValueError("unsupported N02 revision policy")
    model.inference.revision_policy = revision
    model.inference.revision_anchor_only = revision != "disabled"
    model.inference.fast_prefix = bool(settings.get("fast_prefix", False))
    return model


def build_default_v12_model(config: Any = None, **overrides: Any) -> N02V12Model:
    """Default model path of this checkout: the v1.2 chain, never the synthetic P1 path."""
    return build_n02_v12_model(config, **overrides)


def _settings(config: Any) -> dict[str, Any]:
    if config is None:
        return {"camera_ids": ("camera_0",)}
    if isinstance(config, Mapping):
        settings = dict(config)
    elif hasattr(config, "as_dict"):
        settings = dict(config.as_dict())
    else:
        settings = {
            name: getattr(config, name)
            for name in (
                "camera_ids",
                "level_ids",
                "level_dims",
                "dims",
                "depths",
                "in_chans",
                "input_mode",
                "d_model",
                "alignment",
                "calibrator",
                "channel_reduction",
                "decoder_hidden",
                "n_latents",
                "n_queries",
                "out_dim",
                "measurement_channels",
                "saturation_level",
                "units",
                "min_variance",
                "max_variance",
                "n_slots",
                "slot_dim",
                "tau",
                "rounds",
                "k",
                "action_dim",
                "history",
                "query_stride",
                "strict_check",
            )
            if hasattr(config, name)
        }
    cameras = settings.get("camera_ids")
    if cameras is None:
        settings["camera_ids"] = ("camera_0",)
    elif isinstance(cameras, str):
        settings["camera_ids"] = (cameras,)
    return settings


def snapshot_copy(state: SequenceState) -> SequenceState:
    """Deep copy of a state for replay comparisons (used by the save/resume check)."""
    return SequenceState(
        episode_ids=tuple(state.episode_ids),
        slot_hidden=copy.deepcopy(state.slot_hidden),
        existence=copy.deepcopy(state.existence),
        geometry=copy.deepcopy(state.geometry),
        version=state.version,
    )
