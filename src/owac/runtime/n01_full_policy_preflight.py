"""N01 v1.0 early policy preflight, ``mode=contract`` (P1) and ``mode=raw_adapter`` (P2).

``docs/prompts/n01_v1_0/EARLY_POLICY_PREFLIGHT.md`` defines two modes:

* ``contract`` (P1): reuse the already-accepted reference encoder and the real
  feature bridge on real RawVLA NPZ batches, and verify that the frozen pi0.5
  backend plus the trainable bridge/encoder are connected -- gradients reach the
  bridge and the encoder, the backend accumulates none, the padded action steps
  are excluded, and a validation batch is checked without an update.  It also
  records the four-region path binding and the new ``SourceKey`` mapping with a
  real unique-source count, because the fixed protocol paths are what the later
  phases will consume.
* ``raw_adapter`` (P2): the real small-batch train/inference hand-off check for
  :class:`~owac.frameworks.n01_full.adapters.DbsrFeatureAdapter`.  It feeds real
  windows through the adapter's ``init_state/step/readout/reset/detach_state``
  interface on both platforms, then through the shared ``VisualTokenPostprocessor``
  (``in_dim=512``, ``width=2048``, ``tokens_per_camera=256``) and the frozen
  pi0.5 feature bridge, and records repeat-frame, episode-change, partial-reset,
  missing-camera and ten-step-denoise behaviour, the image-tower call counters,
  and the unique-source accounting.  ``--smoke`` adds a local CPU-only interface
  smoke (synthetic frames plus a stub PI0 surface) so the mode is invocable
  without the server assets; a smoke receipt is explicitly *not* P2 acceptance.

The module must run inside the pinned StarVLA runtime (Python 3.10, torch
2.7.1+cu128) with ``PYTHONPATH=<snapshot>/src:<RawVLA>/starVLA``; the local CPU
lock is never allowed to overwrite that environment.

``configs/encoders/n01_full/policy_contract_encoder.json`` is a verbatim copy of
the accepted ``configs/encoders/n01/small.json`` reference-encoder config, moved
into the P1 bundle directory so one source snapshot is self-contained; the report
records both hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from owac.runtime.n01_full_ledger_check import _crop_batch, ledger_source_check

MODES = ("contract", "raw_adapter")
DEFAULT_ENCODER_CONFIG = "configs/encoders/n01_full/policy_contract_encoder.json"


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="owac-n01-full-policy-preflight", description=__doc__)
    top.add_argument("--mode", choices=MODES, required=True)
    top.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/encoders/n01_full/protocol_v1.json"),
        help="frozen protocol that binds data paths, statistics and manifests",
    )
    top.add_argument("--repo-root", type=Path, default=Path("."))
    top.add_argument(
        "--platform",
        action="append",
        choices=("libero", "robotwin2"),
        help="platform(s) to check; default both",
    )
    top.add_argument("--steps", type=int, default=2, help="distinct train batches per platform")
    top.add_argument("--validation-batches", type=int, default=1)
    top.add_argument("--lr", type=float, default=1e-4)
    top.add_argument("--crop", type=int, default=32, help="top-left crop; 0 keeps the full window")
    top.add_argument("--device", default="cuda")
    top.add_argument("--method", default="n01", help="reference encoder used for the bridge")
    top.add_argument("--encoder-config", type=Path, default=Path(DEFAULT_ENCODER_CONFIG))
    top.add_argument("--weights", type=Path, default=None, help="override the protocol checkpoint")
    top.add_argument("--tokenizer", type=Path, default=None, help="override the protocol tokenizer")
    top.add_argument("--output", type=Path, default=Path("outputs/n01-full-policy-preflight"))
    top.add_argument("--seed", type=int, default=101)
    top.add_argument(
        "--ledger-batches",
        type=int,
        default=2,
        help="real batches fed through the new ledger for the unique-source count",
    )
    top.add_argument(
        "--input-mode",
        default=None,
        choices=("rawvla_rgb3", "bayer_cfa", "synthetic_cfa"),
        help="override the protocol's model_input_mode for the P2 adapter stem",
    )
    top.add_argument(
        "--adapter-history",
        type=int,
        default=8,
        help="unique frames the P2 DBSR adapter fuses (max 8 in the first version)",
    )
    top.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "local CPU interface smoke for --mode raw_adapter: synthetic frames and a stub "
            "PI0 surface; never P2 acceptance"
        ),
    )
    top.add_argument(
        "--motion-checkpoint",
        type=Path,
        default=None,
        help=(
            "P2 motion checkpoint: when set, a real CorrespondenceEstimator + "
            "WindowMotionProvider drives the adapter's causal-window offsets (otherwise identity)"
        ),
    )
    top.add_argument(
        "--motion-iters",
        type=int,
        default=4,
        help="iterative RAFT refinement steps of the P2 motion estimator",
    )
    top.add_argument(
        "--motion-calibration-version",
        default="unversioned",
        help="declared calibration version handed to the motion RawSensorFrames",
    )
    return top


def tensor_hash(tensor: Any) -> str:
    """Hash actual tensor bytes, including BF16, plus dtype and shape."""
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def gradient_receipt(parameters: Any) -> dict[str, Any]:
    import torch

    gradients = [p.grad for p in parameters if p.grad is not None]
    return {
        "total": sum(float(g.norm()) for g in gradients),
        "finite": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
        "tensors": len(gradients),
    }


def batch_hashes(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "windows": {
            camera: {
                name: tensor_hash(getattr(window, name))
                for name in (
                    "values",
                    "valid_mask",
                    "bad_mask",
                    "saturated_mask",
                    "frame_ids",
                    "new_measurement",
                )
            }
            for camera, window in batch["windows"].items()
        },
        "actions": tensor_hash(batch["action"]),
        "action_mask": tensor_hash(batch["action_mask"]),
        "state": tensor_hash(batch["state"]),
    }


# ---------------------------------------------------------------------------------------
# mode=raw_adapter (P2): DBSR feature adapter through the shared bridge
# ---------------------------------------------------------------------------------------


def _measurement_frame(
    window: Any,
    step: int,
    camera: str,
    episode_keys: Sequence[str],
    device: Any,
    *,
    sample_xy: bool = True,
):
    """One loader time step as the external ``MeasurementFrame`` (no conversion hidden)."""
    from owac.core.encoder_contract import MeasurementFrame

    metadata: dict[str, Any] = {}
    if window.variance is not None:
        metadata["variance"] = window.variance[:, step].to(device)
    return MeasurementFrame(
        values=window.values[:, step].to(device),
        valid_mask=window.valid_mask[:, step].to(device),
        bad_mask=window.bad_mask[:, step].to(device),
        saturated_mask=window.saturated_mask[:, step].to(device),
        frame_ids=window.frame_ids[:, step].to(device),
        episode_ids=list(episode_keys),
        camera_id=camera,
        timestamp=None if window.timestamp is None else window.timestamp[:, step].to(device),
        time_unit=window.time_unit,
        new_measurement=(
            None if window.new_measurement is None else window.new_measurement[:, step].to(device)
        ),
        sample_xy=window.sample_xy if sample_xy else None,
        metadata=metadata,
    )


def _window_unique_counts(window: Any, batch_size: int) -> dict[str, list[int]]:
    """Per-member unique-frame counts of a collated ``MeasurementWindowBatch``.

    The batch type in ``owac.raw.rawvla_sequence`` has **no**
    ``unique_frame_count()`` helper (only the single-window ``MeasurementWindow``
    does), so the count is rebuilt from the identity arrays it really exposes:
    ``episode_ids``/``frame_ids``/``new_measurement`` with shape ``[B, T]``.

    Two explicit numbers are reported:

    * ``identities``: distinct ``(episode_id, frame_id)`` pairs across the window.
      A re-sent exposure counts once and the same ``frame_id`` in another episode
      counts twice, because identity is the pair, never the timestamp.
    * ``new_arrivals``: the same rule the adapter's ``step`` applies -- the first
      occurrence of an identity is registered only when ``new_measurement`` is
      True (absent flag means True).  This is what ``FullState.unique_arrivals``
      counts, so a match check compares like with like.

    Without identity arrays the count falls back to the declared
    ``new_measurement`` flags and the fallback is recorded by the caller.
    """
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    values = getattr(window, "values", None)
    if values is None or int(values.shape[0]) != int(batch_size):
        raise ValueError(
            f"window batch {None if values is None else int(values.shape[0])} does not match "
            f"batch_size {int(batch_size)}"
        )
    length = int(values.shape[1])
    episodes = getattr(window, "episode_ids", None)
    frame_ids = getattr(window, "frame_ids", None)
    new_measurement = getattr(window, "new_measurement", None)
    flags: list[list[bool]]
    if new_measurement is None:
        flags = [[True] * length for _ in range(batch_size)]
    else:
        reshaped = new_measurement.reshape(batch_size, length)
        flags = [[bool(value) for value in row] for row in reshaped.tolist()]
    if episodes is None or frame_ids is None:
        fallback = [sum(1 for value in row if value) for row in flags]
        return {
            "identities": fallback,
            "new_arrivals": fallback,
            "identity_rule": "new_measurement_fallback",
        }
    episode_rows = episodes.reshape(batch_size, length).tolist()
    frame_rows = frame_ids.reshape(batch_size, length).tolist()
    identities: list[int] = []
    new_arrivals: list[int] = []
    for member in range(batch_size):
        seen: set[tuple[str, Any]] = set()
        observed: set[tuple[str, Any]] = set()
        identity_count = 0
        arrival_count = 0
        for step in range(length):
            identity = (str(episode_rows[member][step]), frame_rows[member][step])
            if identity not in seen:
                seen.add(identity)
                identity_count += 1
            # Mirror the adapter's arrival rule exactly: register a flagged first
            # occurrence, never a duplicate of an already-registered identity.
            if flags[member][step] and identity not in observed:
                observed.add(identity)
                arrival_count += 1
        identities.append(identity_count)
        new_arrivals.append(arrival_count)
    return {
        "identities": identities,
        "new_arrivals": new_arrivals,
        "identity_rule": "(episode_id, frame_id)",
    }


def _adapter_encode_batch(adapter: Any, batch: dict[str, Any], device: Any, *, tbptt: int = 4):
    """Feed a loader batch through the adapter's external interface, one arrival per step."""
    from owac.core.encoder_contract import ComputeBudget

    windows = batch["windows"]
    camera_order = list(windows)
    episode_keys = [meta["episode_key"] for meta in batch["meta"]]
    first = next(iter(windows.values()))
    batch_size, length, _, height, width = first.values.shape
    state = adapter.init_state(
        batch_size, {"camera_ids": camera_order, "shape": (height, width)}, device
    )
    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    feed: dict[str, Any] = {}
    output = None
    for camera, window in windows.items():
        for step in range(int(length)):
            output = adapter.step(
                _measurement_frame(window, step, camera, episode_keys, device), None, state, budget
            )
            state = output.state
            if tbptt and (step + 1) % tbptt == 0 and step + 1 < length:
                state = adapter.detach_state(state)
        counts = _window_unique_counts(window, batch_size)
        adapter_unique = [int(value) for value in state.unique_arrivals[camera].tolist()]
        feed[camera] = {
            "window_frames": int(length),
            "unique_frames_window": counts["identities"],
            "unique_new_measurements_window": counts["new_arrivals"],
            "unique_frame_identity_rule": counts["identity_rule"],
            "adapter_arrivals_lifetime": [int(value) for value in state.arrivals[camera].tolist()],
            "adapter_unique_lifetime": adapter_unique,
            "unique_matches_window": [
                adapter_value == window_value
                for adapter_value, window_value in zip(
                    adapter_unique, counts["new_arrivals"], strict=True
                )
            ],
        }
        entry = state.correspondence.get(camera)
        feed[camera]["p2_motion_status"] = output.diagnostics.get("p2_motion_status")
        feed[camera]["correspondence"] = None if entry is None else entry.describe()
    return state, output, feed


def _vision_tower_probe(policy: Any) -> dict[str, Any]:
    """Count real image-tower entry points; a successful import alone is not evidence."""
    counts: dict[str, Any] = {
        "embed_image_present": False,
        "embed_image_calls": 0,
        "hooked_modules": [],
        "vision_module_forwards": 0,
        "vision_parameter_tensors": 0,
        "method": "wrapped vlm.embed_image + forward hooks on every module named *vision*",
    }
    vlm = getattr(policy, "vlm", None)
    embed_image = getattr(vlm, "embed_image", None) if vlm is not None else None
    if callable(embed_image):
        counts["embed_image_present"] = True

        def counting_embed_image(*args: Any, **kwargs: Any) -> Any:
            counts["embed_image_calls"] += 1
            return embed_image(*args, **kwargs)

        try:
            vlm.embed_image = counting_embed_image
        except (AttributeError, TypeError):  # pragma: no cover - exotic module trees
            counts["embed_image_present"] = False

    def counting_hook(*_args: Any, **_kwargs: Any) -> None:
        counts["vision_module_forwards"] += 1

    for name, module in policy.named_modules():
        if name and "vision" in name:
            counts["hooked_modules"].append(name)
            module.register_forward_hook(counting_hook)
    counts["vision_parameter_tensors"] = sum(
        1 for name, _ in policy.named_parameters() if "vision" in name
    )
    return counts


def _repeat_frame_check(
    adapter: Any, sample: Any, device: Any, crop: int, camera_order: Sequence[str]
):
    """Re-send the newest frame: the buffer must not gain evidence and tokens must not move."""
    import torch

    from owac.core.encoder_contract import ComputeBudget
    from owac.raw.rawvla_sequence import collate_sequence_samples

    batch = collate_sequence_samples([sample])
    _crop_batch(batch, crop)
    camera = camera_order[0]
    window = batch["windows"][camera]
    episode_keys = [meta["episode_key"] for meta in batch["meta"]]
    length = int(window.values.shape[1])
    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    with torch.no_grad():
        state, output, feed = _adapter_encode_batch(adapter, batch, device)
        tokens_once = adapter.readout(output)
        repeat = adapter.step(
            _measurement_frame(window, length - 1, camera, episode_keys, device),
            None,
            state,
            budget,
        )
        tokens_repeat = adapter.readout(repeat)
        tokens_again = adapter.readout(repeat)
    unique_before = feed[camera]["adapter_unique_lifetime"]
    arrivals_before = feed[camera]["adapter_arrivals_lifetime"]
    unique_after = [int(value) for value in repeat.state.unique_arrivals[camera].tolist()]
    arrivals_after = [int(value) for value in repeat.state.arrivals[camera].tolist()]
    return {
        "camera": camera,
        "unique_before": unique_before,
        "unique_after": unique_after,
        "arrivals_before": arrivals_before,
        "arrivals_after": arrivals_after,
        "unique_unchanged": unique_before == unique_after,
        "one_arrival_per_step": all(
            after == before + 1
            for before, after in zip(arrivals_before, arrivals_after, strict=True)
        ),
        "tokens_identical_after_repeat": bool(
            torch.equal(tokens_repeat.values, tokens_once.values)
        ),
        "mask_identical_after_repeat": bool(torch.equal(tokens_repeat.mask, tokens_once.mask)),
        "readout_repeat_identical": bool(torch.equal(tokens_again.values, tokens_repeat.values)),
        "repeated_state_unchanged_by_readout": bool(
            torch.equal(tokens_again.mask, tokens_repeat.mask)
        ),
        "tokens_sha256": tensor_hash(tokens_once.values),
        "window_frames_after_repeat": int(repeat.diagnostics["window_frames"]),
        "repeated_this_step": repeat.diagnostics["repeated_this_step"],
    }


def _analytic_variance_check(sample: Any, camera: str, crop: int, *, repeats: int = 10):
    """Repeated arrivals must not shrink the FP64 analytic variance of the unique sources."""
    import numpy as np
    import torch

    from owac.frameworks.n01_full.ledger import MeasurementLedger
    from owac.frameworks.n01_full.reference import dense_gls
    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.runtime.n01_full_ledger_check import _sample_grid

    batch = collate_sequence_samples([sample])
    _crop_batch(batch, crop)
    window = batch["windows"][camera]
    episode_keys = [meta["episode_key"] for meta in batch["meta"]]
    cpu = torch.device("cpu")

    def ledger_frame(step: int):
        frame = _measurement_frame(window, step, camera, episode_keys, cpu)
        if window.sample_xy is not None:
            # The ledger keeps real per-sample coordinates: [B,C,H,W] -> [B,C,H,W,2].
            frame.sample_xy = _sample_grid(frame.values, window.sample_xy)
        return frame

    ledger = MeasurementLedger("n01-full-policy-preflight", value_units="pseudo_raw")
    for step in range(int(window.values.shape[1])):
        ledger.observe(ledger_frame(step))
    unique = int(ledger.unique_frame_count(camera_id=camera))
    first = dense_gls(np.ones((unique, 1)), np.zeros(unique))
    for _ in range(int(repeats)):
        ledger.observe(ledger_frame(0))
    unique_after = int(ledger.unique_frame_count(camera_id=camera))
    second = dense_gls(np.ones((unique_after, 1)), np.zeros(unique_after))
    return {
        "unique_sources": unique,
        "unique_sources_after_repeats": unique_after,
        "posterior_variance_unique": float(first.covariance[0, 0]),
        "posterior_variance_after_repeats": float(second.covariance[0, 0]),
        "covariance_identical": bool(np.array_equal(first.covariance, second.covariance)),
        "repeats_sent": int(repeats),
        "estimator": "unit-noise FP64 dense GLS over registered unique sources",
    }


def _episode_change_check(
    adapter: Any, samples: Sequence[Any], device: Any, crop: int, camera_order: Sequence[str]
):
    """A first arrival of a new episode resets that member before the frame is pushed."""

    from owac.raw.rawvla_sequence import collate_sequence_samples

    camera = camera_order[0]
    first = collate_sequence_samples([samples[0]])
    _crop_batch(first, crop)
    second = collate_sequence_samples([samples[1]])
    _crop_batch(second, crop)
    _, first_output, first_feed = _adapter_encode_batch(adapter, first, device)
    second_window = second["windows"][camera]
    second_episode = [meta["episode_key"] for meta in second["meta"]]
    from owac.core.encoder_contract import ComputeBudget

    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    output = adapter.step(
        _measurement_frame(second_window, 0, camera, second_episode, device),
        None,
        first_output.state,
        budget,
    )
    return {
        "camera": camera,
        "unique_before_switch": first_feed[camera]["adapter_unique_lifetime"],
        "unique_after_switch": [
            int(value) for value in output.state.unique_arrivals[camera].tolist()
        ],
        "window_frames_after_switch": int(output.diagnostics["window_frames"]),
        "episode_before": first["meta"][0]["episode_key"],
        "episode_after": second["meta"][0]["episode_key"],
        "source_frame_ids_after_switch": [
            int(value) for value in output.diagnostics["source_frame_ids"][0].tolist()
        ],
    }


def _partial_reset_check(
    adapter: Any, samples: Sequence[Any], device: Any, crop: int, camera_order: Sequence[str]
):
    """Reset one member of a real batch and prove the other member kept its buffer."""
    from dataclasses import replace as dataclass_replace

    import torch

    from owac.raw.rawvla_sequence import collate_sequence_samples

    batch = collate_sequence_samples(list(samples))
    _crop_batch(batch, crop)
    camera = camera_order[0]
    state, output, _feed = _adapter_encode_batch(adapter, batch, device)
    per_camera = int(adapter.config.tokens_per_camera)
    before = {
        "frame_ids": state.cameras[camera].frame_ids.detach().clone(),
        "filled": state.cameras[camera].filled.detach().clone(),
        "tokens": adapter.readout(output).values.detach().clone(),
    }
    state = adapter.reset(state, torch.tensor([True, False]))
    after_frame_ids = state.cameras[camera].frame_ids.detach().clone()
    after_filled = state.cameras[camera].filled.detach().clone()
    reset_tokens = adapter.readout(dataclass_replace(output, state=state))
    return {
        "camera": camera,
        "member_reset_filled": int(after_filled[0]),
        "member_kept_filled": int(after_filled[1]),
        "member_kept_filled_unchanged": int(after_filled[1]) == int(before["filled"][1]),
        "member_reset_frame_ids_cleared": bool((after_frame_ids[:, 0] == -1).all()),
        "member_kept_frame_ids_unchanged": bool(
            torch.equal(after_frame_ids[:, 1], before["frame_ids"][:, 1])
        ),
        "member_kept_tokens_unchanged": bool(
            torch.equal(reset_tokens.values[:, per_camera : 2 * per_camera], before["tokens"][1:2])
        ),
        "reset_member_token_mask_false": bool(~reset_tokens.mask[0].any()),
    }


def contract(args: argparse.Namespace) -> dict[str, Any]:

    import numpy as np
    import torch
    import torch.nn.functional as F

    from owac.backends.rawvla import sha256_file
    from owac.frameworks.n01.config import load_config
    from owac.raw.n01_full_manifest import dataset_from_protocol
    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.training.n01_action import (
        action_masked_loss,
        build_bridge,
        encode_windows,
        load_backend,
        observation_from_batch,
    )

    if not 2 <= args.steps <= 200:
        raise ValueError("P1 contract requires 2..200 train updates per platform")
    if args.validation_batches < 1:
        raise ValueError("P1 contract requires a validation batch")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    protocol = json.loads(args.protocol.read_text())
    backend = protocol["backend_protocol"]
    platforms = args.platform or ["libero", "robotwin2"]
    report: dict[str, Any] = {
        "mode": "contract",
        "reference_encoder": {
            "note": "the already-accepted three-channel reference encoder; the P1 analytic "
            "coefficients are not a neural front end",
            "config": str(args.encoder_config),
            "method": args.method,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
        },
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "encoder_config_sha256": sha256_file(args.encoder_config),
        "runtime_sha256": sha256_file(Path(__file__)),
        "platforms": {},
    }
    for name in platforms:
        weights = args.weights or Path(backend["weights"][name]["path"])
        tokenizer = args.tokenizer or Path(backend["tokenizer"]["path"])
        entry: dict[str, Any] = {
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "tokenizer": str(tokenizer),
            "tokenizer_sha256": sha256_file(tokenizer),
        }
        if entry["weights_sha256"] != backend["weights"][name]["sha256"]:
            raise ValueError("checkpoint hash differs from the frozen protocol")
        if entry["tokenizer_sha256"] != backend["tokenizer"]["sha256"]:
            raise ValueError("tokenizer hash differs from the frozen protocol")
        train_set = dataset_from_protocol(protocol, name, "train", repo_root=args.repo_root)
        validation_set = dataset_from_protocol(
            protocol, name, "validation", repo_root=args.repo_root
        )
        entry["regions"] = {
            "train_samples": len(train_set),
            "validation_samples": len(validation_set),
            "train_first_path": str(train_set.paths[0]),
            "validation_first_path": str(validation_set.paths[0]),
        }
        policy, audit = load_backend(name, weights, tokenizer, args.device)
        config = load_config(args.encoder_config)
        first = train_set[0]
        camera_order = list(first.windows)
        encoder, bridge = build_bridge(policy, config, camera_order, args.device, args.method, name)
        encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
        bridge_parameters = [p for p in bridge.postprocessor.parameters() if p.requires_grad]
        parameters = encoder_parameters + bridge_parameters
        backend_before = model_hash(policy)
        entry["initial_model_hashes"] = {
            "encoder": model_hash(encoder),
            "bridge": model_hash(bridge.postprocessor),
            "backend": backend_before,
        }
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
        before = {name_: p.detach().clone() for name_, p in encoder.named_parameters()}
        bridge_before = {
            key: p.detach().clone() for key, p in bridge.postprocessor.named_parameters()
        }
        rows = []
        episodes = set()
        if args.crop > 0:
            entry["crop"] = args.crop
        # Spread the requested real batches across the train region so two steps are
        # genuinely different data, not two windows of the same episode.
        stride = max(len(train_set) // max(args.steps, 1), 1)
        for step in range(max(args.steps, 1)):
            sample = train_set[(step * stride) % len(train_set)]
            batch = collate_sequence_samples([sample])
            _crop_batch(batch, args.crop)
            episode = batch["meta"][0]["episode_key"]
            if episode in episodes:
                raise ValueError("contract requires distinct train episodes")
            episodes.add(episode)
            update_before = [p.detach().clone() for p in parameters]
            tick = time.perf_counter()
            encoder.train()
            optimizer.zero_grad(set_to_none=True)
            tokens, _, diagnostics = encode_windows(encoder, batch, args.device)
            observation = observation_from_batch(policy, batch, args.device)
            actions = batch["action"].to(args.device)
            actions = F.pad(actions, (0, policy.action_dim - actions.shape[-1]))
            result = bridge.forward_from_observation_with_features(
                observation, tokens, actions, return_debug=True
            )
            loss = action_masked_loss(
                result["v_t"],
                result["u_t"],
                batch["action_mask"].to(args.device),
                train_set.protocol.action_dim,
            )
            loss.backward()
            encoder_grads = gradient_receipt(encoder_parameters)
            bridge_grads = gradient_receipt(bridge_parameters)
            if bridge_grads["total"] <= 0 or not bridge_grads["finite"]:
                raise RuntimeError(f"invalid bridge gradient: {bridge_grads}")
            backend_gradients = sum(1 for p in policy.parameters() if p.grad is not None)
            if backend_gradients:
                raise RuntimeError("the frozen backend accumulated a parameter gradient")
            if encoder_grads["total"] <= 0.0 or not encoder_grads["finite"]:
                raise RuntimeError(f"invalid encoder gradient: {encoder_grads}")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            deltas = [
                float((p - old).abs().sum())
                for p, old in zip(parameters, update_before, strict=True)
            ]
            update = {
                "encoder": sum(deltas[: len(encoder_parameters)]),
                "bridge": sum(deltas[len(encoder_parameters) :]),
            }
            if not all(value > 0 for value in update.values()):
                raise RuntimeError(f"encoder/bridge did not both update: {update}")
            count = result["visual_prefix_tokens"]
            prefix_shape = list(result["prefix_embs"][:, :count].shape)
            prefix_mask = result["pad_masks"][:, :count]
            if prefix_shape != [1, len(camera_order) * 256, 2048]:
                raise RuntimeError(f"unexpected actual visual prefix: {prefix_shape}")
            if not bool(torch.equal(prefix_mask, tokens.mask)):
                raise RuntimeError("visual prefix mask differs from encoder mask")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            rows.append(
                {
                    "step": step,
                    "episode": batch["meta"][0]["episode_key"],
                    "cameras": list(batch["windows"]),
                    "action_loss": float(loss.detach()),
                    "valid_actions": int(batch["action_mask"].sum()),
                    "encoder_gradient": encoder_grads,
                    "bridge_gradient": bridge_grads,
                    "parameter_updates_l1": update,
                    "input_hashes": batch_hashes(batch),
                    "actual_visual_prefix_shape": prefix_shape,
                    "actual_visual_prefix_dtype": str(result["prefix_embs"].dtype),
                    "actual_visual_prefix_mask_sha256": tensor_hash(prefix_mask),
                    "backend_gradients": backend_gradients,
                    "wall_ms": (time.perf_counter() - tick) * 1000.0,
                    "prefix_tokens": list(tokens.values.shape),
                    "prefix_mask_valid": int(tokens.mask.sum()),
                    "diagnostics": {
                        key: value
                        for key, value in (diagnostics or {}).items()
                        if isinstance(value, (int, float, str, bool))
                    },
                }
            )
        entry["train_updates"] = rows
        entry["token_contract"] = {
            "values_shape": list(tokens.values.shape),
            "mask_shape": list(tokens.mask.shape),
            "camera_ids": list(tokens.camera_ids),
            "frozen_backend_parameters": sum(1 for _ in policy.parameters()),
        }
        # Validation batches: forward only, and the parameters must not move.
        validation_snapshot = [p.detach().clone() for p in parameters]
        validation_hashes = {
            "encoder": model_hash(encoder),
            "bridge": model_hash(bridge.postprocessor),
        }
        validation_rows = []
        encoder.eval()
        # ``no_grad`` keeps the bridge's training forward available (it refuses
        # ``torch.inference_mode()``); no update happens here and the parameter
        # drift below proves it.
        with torch.no_grad():
            for step in range(max(args.validation_batches, 1)):
                sample = validation_set[step % len(validation_set)]
                batch = collate_sequence_samples([sample])
                _crop_batch(batch, args.crop)
                tokens, _, _ = encode_windows(encoder, batch, args.device)
                observation = observation_from_batch(policy, batch, args.device)
                actions = batch["action"].to(args.device)
                actions = F.pad(actions, (0, policy.action_dim - actions.shape[-1]))
                result = bridge.forward_from_observation_with_features(
                    observation, tokens, actions, return_debug=True
                )
                loss = action_masked_loss(
                    result["v_t"],
                    result["u_t"],
                    batch["action_mask"].to(args.device),
                    validation_set.protocol.action_dim,
                )
                validation_rows.append(
                    {
                        "episode": batch["meta"][0]["episode_key"],
                        "action_loss": float(loss.detach()),
                        "updated": False,
                    }
                )
        parameter_drift = sum(
            float((p - old).abs().sum())
            for p, old in zip(parameters, validation_snapshot, strict=True)
        )
        validation_after = {
            "encoder": model_hash(encoder),
            "bridge": model_hash(bridge.postprocessor),
        }
        if parameter_drift != 0 or validation_hashes != validation_after:
            raise RuntimeError("validation changed encoder/bridge state")
        entry["validation"] = {
            "rows": validation_rows,
            "parameter_drift_after_validation": parameter_drift,
            "before_hashes": validation_hashes,
            "after_hashes": validation_after,
        }
        entry["encoder_parameter_update"] = float(
            sum(
                float((p - before[name_]).abs().sum())
                for name_, p in encoder.named_parameters()
                if name_ in before
            )
        )
        entry["bridge_parameter_update"] = sum(
            float((p - bridge_before[key]).abs().sum())
            for key, p in bridge.postprocessor.named_parameters()
        )
        backend_after = model_hash(policy)
        entry["backend_parameter_hashes"] = {"before": backend_before, "after": backend_after}
        if backend_before != backend_after:
            raise RuntimeError("frozen backend changed")
        entry["load_audit"] = audit if isinstance(audit, (dict, list, str)) else str(type(audit))
        entry["ledger_source_check"] = ledger_source_check(
            train_set, name, batches=args.ledger_batches, crop=args.crop
        )
        report["platforms"][name] = entry
        del policy, bridge, encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return report


def _missing_camera_check(
    adapter: Any,
    postprocessor: Any,
    sample: Any,
    device: Any,
    crop: int,
    camera_order: Sequence[str],
):
    """Only arrived cameras get token blocks; absent ones stay zero and masked out."""
    import torch

    from owac.core.encoder_contract import ComputeBudget
    from owac.raw.rawvla_sequence import collate_sequence_samples

    present = camera_order[0]
    batch = collate_sequence_samples([sample])
    _crop_batch(batch, crop)
    window = batch["windows"][present]
    episode_keys = [meta["episode_key"] for meta in batch["meta"]]
    height, width = (int(size) for size in window.values.shape[-2:])
    state = adapter.init_state(
        1, {"camera_ids": list(camera_order), "shape": (height, width)}, device
    )
    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    output = None
    for step in range(int(window.values.shape[1])):
        output = adapter.step(
            _measurement_frame(window, step, present, episode_keys, device), None, state, budget
        )
        state = output.state
    tokens = adapter.readout(output)
    prefix = postprocessor(tokens)
    per_camera = int(adapter.config.tokens_per_camera)
    missing_slice = slice(per_camera, len(camera_order) * per_camera)
    return {
        "present_camera": present,
        "missing_cameras": [camera for camera in camera_order if camera != present],
        "token_camera_ids": list(tokens.camera_ids),
        "token_shape": list(tokens.values.shape),
        "prefix_shape": list(prefix.embs.shape),
        "prefix_len_is_fixed": int(prefix.pad_masks.shape[1]) == len(camera_order) * per_camera,
        "missing_block_zero": float(prefix.embs[:, missing_slice].abs().sum()) == 0.0,
        "missing_block_mask_false": bool(not prefix.pad_masks[:, missing_slice].any()),
        "present_block_mask_matches_tokens": bool(
            torch.equal(prefix.pad_masks[:, :per_camera], tokens.mask)
        ),
        "synthetic_blocks": prefix.diagnostics["synthetic_camera_blocks"],
    }


def _denoise_check(
    bridge: Any, policy: Any, batch: dict[str, Any], tokens: Any, state: Any, device: Any
):
    """Ten flow steps twice on the cached prefix: no re-observation, no re-readout."""
    import torch

    from owac.core.encoder_contract import VisualTokens
    from owac.training.n01_action import observation_from_batch

    observation = observation_from_batch(policy, batch, device)
    batch_size = len(batch["meta"])
    noise = torch.randn(
        batch_size,
        int(policy.action_horizon),
        int(policy.action_dim),
        generator=torch.Generator().manual_seed(11),
    )
    # Detached on purpose: the prefix is data for the flow loop, and a repeated solve
    # must not run the encoder, the adapter or the vision tower again.
    frozen = VisualTokens(
        values=tokens.values.detach(),
        mask=tokens.mask.detach(),
        camera_ids=list(tokens.camera_ids),
        positions=None if tokens.positions is None else tokens.positions.detach(),
    )
    arrivals_before = int(sum(int(value.sum()) for value in state.arrivals.values()))
    calls_before = bridge.stats["encoder_calls"]
    steps_before = bridge.stats["denoise_steps"]
    builds_before = bridge.stats["prefix_builds"]
    first = bridge.sample_actions_with_features(
        observation, frozen, noise=noise.clone(), num_steps=10
    )
    second = bridge.sample_actions_with_features(
        observation, frozen, noise=noise.clone(), num_steps=10
    )
    arrivals_after = int(sum(int(value.sum()) for value in state.arrivals.values()))
    return {
        "num_denoise_steps": 10,
        "shape": list(first.shape),
        "finite": bool(torch.isfinite(first).all()),
        "repeat_max_abs": float((first - second).abs().max()),
        "tolerance": 1e-3,
        "encoder_calls_during_denoise": bridge.stats["encoder_calls"] - calls_before,
        "arrivals_during_denoise": arrivals_after - arrivals_before,
        "denoise_steps_two_calls": bridge.stats["denoise_steps"] - steps_before,
        "prefix_builds_two_calls": bridge.stats["prefix_builds"] - builds_before,
        "prefix_source": bridge._prefix_source,
    }


def _distinct_samples(dataset: Any, crop: int, count: int) -> list[Any]:
    """First ``count`` train samples from distinct source episodes (no adjacent-frame leak)."""
    from owac.raw.rawvla_sequence import collate_sequence_samples

    samples: list[Any] = []
    seen: set[str] = set()
    for index in range(min(len(dataset), 64)):
        sample = dataset[index]
        batch = collate_sequence_samples([sample])
        _crop_batch(batch, crop)
        key = str(batch["meta"][0]["episode_key"])
        if key in seen:
            continue
        seen.add(key)
        samples.append(sample)
        if len(samples) >= count:
            return samples
    raise ValueError(f"raw_adapter needs {count} distinct train episodes, found {len(samples)}")


#: ``(fnet, cnet)`` first-convolution input channels of the two motion stems.
MOTION_STEM_CHANNELS = {"cfa4": (4, 8), "rgb3": (3, 6)}


def _motion_checkpoint_stem(checkpoint: Any) -> dict[str, Any]:
    """Infer a motion checkpoint's stem from its ``fnet``/``cnet`` input channels.

    The motion core owns checkpoint loading (its guarded loader accepts the
    versioned P2 checkpoint schema and a plain tensor state dict), so this reads
    the state dict through that loader instead of re-implementing it, then reads
    the two first-convolution input widths the protocol pairs as ``C`` and ``2C``.
    """
    import torch

    from owac.backends import raw_correspondence as motion_core

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"motion checkpoint not found: {path}")
    loader = getattr(motion_core, "_load_pretrained_payload", None)
    extractor = getattr(motion_core, "_pretrained_state_dict", None)
    if loader is None or extractor is None:  # pragma: no cover - interface drift
        raise RuntimeError(
            "this checkout's motion core does not expose the guarded checkpoint loader "
            "(_load_pretrained_payload/_pretrained_state_dict); cannot inspect the stem"
        )
    raw = path.read_bytes()
    payload, weights_only_fallback = loader(raw, path)
    state, schema, declared = extractor(payload, path)

    def first_conv_channels(key: str) -> int:
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or value.dim() < 2:
            raise ValueError(
                f"motion checkpoint {path} has no usable {key!r} tensor; cannot infer its stem"
            )
        return int(value.shape[1])

    fnet = first_conv_channels("fnet.conv1.weight")
    cnet = first_conv_channels("cnet.conv1.weight")
    inferred = next(
        (name for name, pair in MOTION_STEM_CHANNELS.items() if pair == (fnet, cnet)), None
    )
    declared_mode = declared.get("mode") if isinstance(declared, Mapping) else None
    if inferred is None:
        raise ValueError(
            f"motion checkpoint {path} has fnet input {fnet} and cnet input {cnet} channels; "
            f"the two accepted stems are cfa4 fnet/cnet {MOTION_STEM_CHANNELS['cfa4']} and rgb3 "
            f"{MOTION_STEM_CHANNELS['rgb3']}"
        )
    if declared_mode is not None and str(declared_mode) != inferred:
        raise ValueError(
            f"motion checkpoint {path} declares mode {declared_mode!r} but its tensors carry the "
            f"{inferred} stem (fnet {fnet} / cnet {cnet}); the declaration and the weights disagree"
        )
    return {
        "checkpoint": str(path),
        "schema": schema,
        "weights_only_fallback": bool(weights_only_fallback),
        "declared_mode": None if declared_mode is None else str(declared_mode),
        "fnet_channels": fnet,
        "cnet_channels": cnet,
        "inferred_mode": inferred,
        "expected_pairs": {name: list(pair) for name, pair in MOTION_STEM_CHANNELS.items()},
    }


def _require_matching_motion_stem(
    checkpoint: Any, adapter_config: Any, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse a checkpoint whose stem differs from the windows this run feeds.

    The measurement channels come from the adapter/window configuration, never
    from the checkpoint: the protocol pairs a ``C``-channel measurement with a
    ``C``/``2C`` motion stem, and a mismatch is reported with both sides and the
    rule instead of torch's raw ``load_state_dict`` size error.  Nothing is
    switched or adapted implicitly.
    """
    actual_mode = "cfa4" if adapter_config.is_cfa else "rgb3"
    actual_channels = int(adapter_config.input_channels)
    stem = _motion_checkpoint_stem(checkpoint)
    rule = str(protocol.get("coordinate_units", {}).get("input_channels_multiple_rule", ""))
    if stem["inferred_mode"] != actual_mode or stem["cnet_channels"] != 2 * actual_channels:
        recommendation = "cfa4 (4/8)" if actual_mode == "cfa4" else "rawvla_rgb3 (3/6)"
        raise ValueError(
            "motion checkpoint stem mismatch: "
            f"{stem['checkpoint']} is a {stem['inferred_mode']} checkpoint "
            f"(fnet {stem['fnet_channels']} input channels, cnet {stem['cnet_channels']} "
            f"channels), but this run feeds {actual_mode} windows with {actual_channels} "
            f"measured channels (a matching cnet needs {2 * actual_channels}). "
            f"Protocol rule: {rule or 'n/a'}. Use a {recommendation} motion checkpoint for this "
            "input mode; the stem is never switched or adapted implicitly."
        )
    return stem


def _raw_adapter_real(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from owac.backends.rawvla import sha256_file
    from owac.backends.rawvla_features import ACTION_PROTOCOLS, RawVLAFeaturePolicy
    from owac.frameworks.n01_full.adapters import (
        P3_SUPPORT_ENABLED,
        DbsrAdapterConfig,
        DbsrFeatureAdapter,
    )
    from owac.raw.n01_full_manifest import dataset_from_protocol
    from owac.raw.rawvla_sequence import collate_sequence_samples
    from owac.training.n01_action import (
        action_masked_loss,
        load_backend,
        observation_from_batch,
    )

    if not 1 <= args.steps <= 200:
        raise ValueError("raw_adapter requires 1..200 train updates per platform")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    protocol = json.loads(args.protocol.read_text())
    backend = protocol["backend_protocol"]
    platforms = args.platform or ["libero", "robotwin2"]
    report: dict[str, Any] = {
        "mode": "raw_adapter",
        "acceptance": "P2_end_early_policy_preflight",
        "p2_acceptance_note": "interface/gradient connectivity only; not P7 action quality",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
        },
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "runtime_sha256": sha256_file(Path(__file__)),
        "p3_support_enabled": P3_SUPPORT_ENABLED,
        "aux_raw_loss": {
            "enabled": False,
            "config": "aux_measurement_prediction=False",
            "note": "auxiliary RAW-domain loss stays off unless a later phase turns it on",
        },
        "platforms": {},
    }
    for name in platforms:
        weights = args.weights or Path(backend["weights"][name]["path"])
        tokenizer = args.tokenizer or Path(backend["tokenizer"]["path"])
        for label, path in (("weights", weights), ("tokenizer", tokenizer)):
            if not path.is_file():
                raise FileNotFoundError(
                    f"raw_adapter needs the server {label} asset ({path} is absent). Run this "
                    "mode inside the pinned RawVLA runtime on the GPU host; the local CPU lock "
                    "does not carry the frozen backend."
                )
        entry: dict[str, Any] = {
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "tokenizer": str(tokenizer),
            "tokenizer_sha256": sha256_file(tokenizer),
        }
        if entry["weights_sha256"] != backend["weights"][name]["sha256"]:
            raise ValueError("checkpoint hash differs from the frozen protocol")
        if entry["tokenizer_sha256"] != backend["tokenizer"]["sha256"]:
            raise ValueError("tokenizer hash differs from the frozen protocol")
        train_set = dataset_from_protocol(protocol, name, "train", repo_root=args.repo_root)
        validation_set = dataset_from_protocol(
            protocol, name, "validation", repo_root=args.repo_root
        )
        entry["regions"] = {
            "train_samples": len(train_set),
            "validation_samples": len(validation_set),
            "train_first_path": str(train_set.paths[0]),
            "validation_first_path": str(validation_set.paths[0]),
        }
        policy, audit = load_backend(name, weights, tokenizer, args.device)
        probe = _vision_tower_probe(policy)
        first = train_set[0]
        camera_order = list(first.windows)
        input_mode = args.input_mode or protocol["data"]["platforms"][name].get(
            "model_input_mode", "rawvla_rgb3"
        )
        adapter_config = DbsrAdapterConfig(
            input_mode=input_mode, max_history=int(args.adapter_history)
        )
        adapter = DbsrFeatureAdapter(adapter_config).to(args.device)
        post = adapter.build_postprocessor(camera_order).to(args.device)
        entry["motion"] = None
        if args.motion_checkpoint is not None:
            from owac.backends.raw_correspondence import CorrespondenceEstimator
            from owac.frameworks.n01_full.motion_builders import WindowMotionProvider

            # Refuse a stem mismatch with the protocol rule before torch's raw
            # ``load_state_dict`` error can surface; the mode is never switched.
            stem_report = _require_matching_motion_stem(
                args.motion_checkpoint, adapter_config, protocol
            )
            mode = "cfa4" if adapter_config.is_cfa else "rgb3"
            cfa_pattern = adapter_config.cfa_pattern if adapter_config.is_cfa else "rgb3"
            channel_phase = (0, 1, 2, 3) if adapter_config.is_cfa else ()
            estimator = CorrespondenceEstimator(
                mode=mode,
                pretrained_checkpoint=args.motion_checkpoint,
                iters=int(args.motion_iters),
                device=args.device,
            )
            adapter.set_motion_provider(
                WindowMotionProvider(
                    estimator,
                    dataset_id=f"rawvla_bench:{name}",
                    cfa_pattern=cfa_pattern,
                    channel_phase=channel_phase,
                    calibration_version=str(args.motion_calibration_version),
                    iters=int(args.motion_iters),
                )
            )
            entry["motion"] = {
                "checkpoint": str(args.motion_checkpoint),
                "checkpoint_sha256": sha256_file(args.motion_checkpoint),
                "mode": mode,
                "checkpoint_stem": stem_report,
                "iters": int(args.motion_iters),
                "calibration_version": str(args.motion_calibration_version),
                "mapping_version": estimator.mapping_version,
                "pretrained_report": (
                    None
                    if estimator.pretrained_report is None
                    else dict(estimator.pretrained_report)
                ),
            }
        bridge = RawVLAFeaturePolicy(
            policy,
            encoder=adapter,
            postprocessor=post,
            action_protocol=ACTION_PROTOCOLS["libero" if name == "libero" else "robotwin"],
        )
        freeze = bridge.freeze_backend_parameters()
        adapter_parameters = [p for p in adapter.parameters() if p.requires_grad]
        bridge_parameters = [p for p in post.parameters() if p.requires_grad]
        parameters = adapter_parameters + bridge_parameters
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
        entry["input_mode"] = input_mode
        entry["camera_order"] = camera_order
        entry["adapter"] = adapter_config.to_dict()
        entry["stem"] = adapter.stem_report()
        entry["parameters"] = {
            "adapter": adapter.parameter_report(),
            "postprocessor": post.parameter_report(),
            "freeze": freeze,
            "no_hidden_readout_projection": (
                "the only 512 -> 2048 map is postprocessor.in_proj; no (2048, 2048) parameter "
                "exists in either module"
            ),
        }
        entry["initial_model_hashes"] = {
            "adapter": model_hash(adapter),
            "postprocessor": model_hash(post),
            "backend": model_hash(policy),
        }
        rows = []
        episodes: set[str] = set()
        samples_used: list[Any] = []
        last_batch: dict[str, Any] = {}
        last_state: Any = None
        last_tokens: Any = None
        stride = max(len(train_set) // max(args.steps, 1), 1)
        for step in range(max(args.steps, 1)):
            sample = train_set[(step * stride) % len(train_set)]
            samples_used.append(sample)
            batch = collate_sequence_samples([sample])
            _crop_batch(batch, args.crop)
            episode = str(batch["meta"][0]["episode_key"])
            if episode in episodes:
                raise ValueError("raw_adapter requires distinct train episodes")
            episodes.add(episode)
            update_before = [p.detach().clone() for p in parameters]
            tick = time.perf_counter()
            adapter.train()
            post.train()
            optimizer.zero_grad(set_to_none=True)
            state, output, feed = _adapter_encode_batch(adapter, batch, args.device, tbptt=4)
            # The bridge's single counted readout path, not a side call to readout().
            tokens = bridge.encode_visual_tokens(output)
            observation = observation_from_batch(policy, batch, args.device)
            actions = batch["action"].to(args.device)
            actions = F.pad(actions, (0, policy.action_dim - actions.shape[-1]))
            result = bridge.forward_from_observation_with_features(
                observation, tokens, actions, return_debug=True
            )
            loss = action_masked_loss(
                result["v_t"],
                result["u_t"],
                batch["action_mask"].to(args.device),
                train_set.protocol.action_dim,
            )
            loss.backward()
            adapter_grads = gradient_receipt(adapter_parameters)
            bridge_grads = gradient_receipt(bridge_parameters)
            if adapter_grads["total"] <= 0 or not adapter_grads["finite"]:
                raise RuntimeError(f"invalid adapter gradient: {adapter_grads}")
            if bridge_grads["total"] <= 0 or not bridge_grads["finite"]:
                raise RuntimeError(f"invalid bridge gradient: {bridge_grads}")
            backend_gradients = sum(1 for p in policy.parameters() if p.grad is not None)
            if backend_gradients:
                raise RuntimeError("the frozen backend accumulated a parameter gradient")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            deltas = [
                float((p - old).detach().abs().sum())
                for p, old in zip(parameters, update_before, strict=True)
            ]
            update = {
                "adapter": sum(deltas[: len(adapter_parameters)]),
                "postprocessor": sum(deltas[len(adapter_parameters) :]),
            }
            if not all(value > 0 for value in update.values()):
                raise RuntimeError(f"adapter/postprocessor did not both update: {update}")
            count = result["visual_prefix_tokens"]
            prefix_shape = list(result["prefix_embs"][:, :count].shape)
            prefix_mask = result["pad_masks"][:, :count]
            expected_shape = [
                len(batch["meta"]),
                len(camera_order) * adapter_config.tokens_per_camera,
                adapter_config.readout_width,
            ]
            if prefix_shape != expected_shape:
                raise RuntimeError(f"unexpected visual prefix {prefix_shape} != {expected_shape}")
            if not torch.equal(prefix_mask, tokens.mask):
                raise RuntimeError("visual prefix mask differs from the adapter mask")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            rows.append(
                {
                    "step": step,
                    "episode": episode,
                    "action_loss": float(loss.detach()),
                    "valid_actions": int(batch["action_mask"].sum()),
                    "encoder_gradient": adapter_grads,
                    "bridge_gradient": bridge_grads,
                    "parameter_updates_l1": update,
                    "input_hashes": batch_hashes(batch),
                    "actual_visual_prefix_shape": prefix_shape,
                    "actual_visual_prefix_dtype": str(result["prefix_embs"].dtype),
                    "actual_visual_prefix_mask_sha256": tensor_hash(prefix_mask),
                    "visual_prefix_tokens": int(count),
                    "prefix_valid_slots": int(tokens.mask.sum()),
                    "token_shape": list(tokens.values.shape),
                    "bridge_encoder_calls": bridge.stats["encoder_calls"],
                    "backend_gradients": backend_gradients,
                    "wall_ms": (time.perf_counter() - tick) * 1000.0,
                    "source_observe_counts": feed,
                    "prefix_diagnostics": {
                        key: value
                        for key, value in (result["visual_prefix_diagnostics"] or {}).items()
                        if isinstance(value, (int, float, str, bool, list))
                    },
                    "p3_support_applied": False,
                    "aux_raw_loss": False,
                }
            )
            last_batch, last_state, last_tokens = batch, state, tokens
        entry["train_updates"] = rows
        entry["distinct_train_episodes"] = len(episodes)
        # The policy-level reset must delegate to the adapter's state (partial member reset).
        entry["policy_reset_check"] = bridge.reset_encoder_state(
            torch.ones(len(batch["meta"]), dtype=torch.bool)
        )
        # Semantics checks on real windows, without any optimizer step.
        adapter.eval()
        post.eval()
        samples = _distinct_samples(train_set, args.crop, 2)
        entry["repeat_frame_check"] = _repeat_frame_check(
            adapter, samples[0], args.device, args.crop, camera_order
        )
        entry["analytic_variance_check"] = _analytic_variance_check(
            samples[0], camera_order[0], args.crop
        )
        entry["episode_change_check"] = _episode_change_check(
            adapter, samples, args.device, args.crop, camera_order
        )
        entry["partial_reset_check"] = _partial_reset_check(
            adapter, samples, args.device, args.crop, camera_order
        )
        entry["missing_camera_check"] = _missing_camera_check(
            adapter, post, samples[0], args.device, args.crop, camera_order
        )
        entry["denoise_check"] = _denoise_check(
            bridge, policy, last_batch, last_tokens, last_state, args.device
        )
        entry["vision_tower"] = {
            "probe": probe,
            "audit_vision_release": (
                audit.get("vision_release") if isinstance(audit, dict) else None
            ),
            "replaced_vision_keys": (
                audit.get("replaced_vision_keys") if isinstance(audit, dict) else None
            ),
            "zero_entry_calls": probe["embed_image_calls"] == 0
            and probe["vision_module_forwards"] == 0,
            "zero_vision_parameter_tensors": probe["vision_parameter_tensors"] == 0,
            "evidence": "wrapped vlm.embed_image + forward hooks installed before the batches",
        }
        if (
            probe["embed_image_calls"]
            or probe["vision_module_forwards"]
            or probe["vision_parameter_tensors"]
        ):
            raise RuntimeError(f"the image tower ran on the raw_adapter batch: {probe}")
        # Validation: forward only, and the parameters must not move.
        validation_snapshot = [p.detach().clone() for p in parameters]
        validation_hashes = {"adapter": model_hash(adapter), "postprocessor": model_hash(post)}
        validation_rows = []
        with torch.no_grad():
            for step in range(max(args.validation_batches, 1)):
                sample = validation_set[step % len(validation_set)]
                batch = collate_sequence_samples([sample])
                _crop_batch(batch, args.crop)
                state, output, feed = _adapter_encode_batch(adapter, batch, args.device, tbptt=0)
                tokens = bridge.encode_visual_tokens(output)
                observation = observation_from_batch(policy, batch, args.device)
                actions = batch["action"].to(args.device)
                actions = F.pad(actions, (0, policy.action_dim - actions.shape[-1]))
                result = bridge.forward_from_observation_with_features(
                    observation, tokens, actions, return_debug=True
                )
                loss = action_masked_loss(
                    result["v_t"],
                    result["u_t"],
                    batch["action_mask"].to(args.device),
                    validation_set.protocol.action_dim,
                )
                validation_rows.append(
                    {
                        "episode": batch["meta"][0]["episode_key"],
                        "action_loss": float(loss.detach()),
                        "updated": False,
                        "source_observe_counts": feed,
                    }
                )
        parameter_drift = sum(
            float((p - old).abs().sum())
            for p, old in zip(parameters, validation_snapshot, strict=True)
        )
        validation_after = {"adapter": model_hash(adapter), "postprocessor": model_hash(post)}
        if parameter_drift != 0 or validation_hashes != validation_after:
            raise RuntimeError("validation changed adapter/postprocessor state")
        entry["validation"] = {
            "rows": validation_rows,
            "parameter_drift_after_validation": parameter_drift,
            "before_hashes": validation_hashes,
            "after_hashes": validation_after,
        }
        backend_after = model_hash(policy)
        entry["backend_parameter_hashes"] = {
            "before": entry["initial_model_hashes"]["backend"],
            "after": backend_after,
        }
        if entry["initial_model_hashes"]["backend"] != backend_after:
            raise RuntimeError("frozen backend changed")
        entry["load_audit"] = audit if isinstance(audit, (dict, list, str)) else str(type(audit))
        entry["ledger_source_check"] = ledger_source_check(
            train_set, name, batches=args.ledger_batches, crop=args.crop
        )
        report["platforms"][name] = entry
        del policy, bridge, adapter, post
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return report


def _smoke_policy(
    width: int, action_dim: int, action_horizon: int, max_state_dim: int, vocab: int = 32
):
    """A tiny interface replica of the PI0/PI05 surface used by ``RawVLAFeaturePolicy``.

    It is a *stub*: it has none of pi0.5's structure and exists only so the local CPU
    smoke can exercise the real adapter, the real postprocessor and the real bridge
    plumbing.  ``embed_image`` raises, so an accidental image-tower call fails the run.
    """
    from types import SimpleNamespace

    import torch
    from torch import nn

    class SmokeVLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, width)
            self.image_proj = nn.Linear(3, width)
            self.embed_image_calls = 0

        def embed_language_tokens(self, tokens):
            return self.embed(tokens)

        def forward_prefix(
            self, *, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache=True
        ):
            return SimpleNamespace(past_key_values=SimpleNamespace(prefix_embs=inputs_embeds))

        def embed_image(self, image):
            self.embed_image_calls += 1
            raise RuntimeError("the raw_adapter path must never run the image tower")

    class SmokeActionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_in = nn.Linear(action_dim, width)
            self.hidden = nn.Linear(width, width)
            self.out = nn.Linear(width, action_dim)

        def sample_action_noise(self, shape, device):
            return torch.zeros(shape, device=device)

        def prepare_flow_targets(self, actions, noise=None, time=None):
            time = (
                torch.full((actions.shape[0],), 0.5, device=actions.device)
                if time is None
                else time
            )
            noise = torch.zeros_like(actions) if noise is None else noise
            x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
            return x_t, noise - actions, noise, time

        def embed_action_suffix(self, state, x_t, timestep):
            embs = self.action_in(x_t)
            pad = torch.ones(embs.shape[:2], dtype=torch.bool, device=embs.device)
            att = torch.zeros_like(pad)
            att[:, :1] = True
            return embs, pad, att, None

        def decode_action_velocity(self, hidden):
            return self.out(hidden)

    class SmokePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm = SmokeVLM()
            self.action_head = SmokeActionHead()
            self.action_dim = action_dim
            self.action_horizon = action_horizon
            self.max_state_dim = max_state_dim
            self.num_inference_steps = 10
            self.state_pad_value = 0.0
            self.discrete_state_input = False

        def forward_with_action_expert(
            self, *, prefix_embs, suffix_embs, attention_mask, position_ids, adarms_cond
        ):
            context = prefix_embs.mean(1, keepdim=True)
            return self.action_head.hidden(suffix_embs) + context

        def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            context = past_key_values.prefix_embs.mean(1, keepdim=True)
            hidden = self.action_head.hidden(self.action_head.action_in(x_t)) + context
            return self.action_head.decode_action_velocity(hidden)

    return SmokePolicy()


def raw_adapter_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Local CPU interface smoke for ``mode=raw_adapter`` (never P2 acceptance)."""
    from types import SimpleNamespace

    import numpy as np
    import torch

    from owac.backends.rawvla_features import ACTION_PROTOCOLS, RawVLAFeaturePolicy
    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame, VisualTokens
    from owac.frameworks.n01_full.adapters import (
        P3_SUPPORT_ENABLED,
        DbsrAdapterConfig,
        DbsrFeatureAdapter,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    adapter_config = DbsrAdapterConfig(
        input_mode="rawvla_rgb3",
        stem_width=16,
        residual_blocks=1,
        feature_width=32,
        projection_width=8,
        offset_width=16,
        max_history=4,
        pool_grid=4,
        tokens_per_camera=16,
        readout_width=64,
    )
    adapter = DbsrFeatureAdapter(adapter_config)
    camera_order = ("base_0_rgb", "left_wrist_0_rgb")
    post = adapter.build_postprocessor(camera_order, name="owac-n01-raw-adapter-smoke-cpu")

    class _SmokeFlow(torch.nn.Module):
        """Tiny ``PackedFlowModel``: one constant packed pull field for every pair."""

        def __init__(self, packed_shape):
            super().__init__()
            self.packed_shape = packed_shape
            self.calls = 0

        def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
            self.calls += 1
            batch = int(image1.shape[0])
            flow = torch.zeros(batch, 2, *self.packed_shape, dtype=torch.float32)
            flow[:, 0] = 0.5
            return {"final": flow}

    from owac.backends.raw_correspondence import CorrespondenceEstimator
    from owac.frameworks.n01_full.motion_builders import WindowMotionProvider

    smoke_flow = _SmokeFlow((32, 32))
    adapter.set_motion_provider(
        WindowMotionProvider(
            CorrespondenceEstimator(mode="rgb3", model=smoke_flow, iters=1, device="cpu"),
            dataset_id="raw-adapter-smoke",
            cfa_pattern="rgb3",
        )
    )
    policy = _smoke_policy(
        width=adapter_config.readout_width, action_dim=4, action_horizon=3, max_state_dim=4
    )
    bridge = RawVLAFeaturePolicy(
        policy, encoder=adapter, postprocessor=post, action_protocol=ACTION_PROTOCOLS["libero"]
    )
    freeze = bridge.freeze_backend_parameters()
    probe = _vision_tower_probe(policy)
    budget = ComputeBudget(max_updates=1, max_history=adapter_config.max_history)

    def synthetic_frame(camera, step, episode_ids, *, seed, invalid_corner, frame_id=None):
        generator = torch.Generator().manual_seed(seed * 100 + step)
        values = torch.randn(2, 3, 32, 32, generator=generator)
        valid = torch.ones(2, 3, 32, 32, dtype=torch.bool)
        if invalid_corner:
            valid[0, :, :8, :8] = False
        identity = step if frame_id is None else frame_id
        return MeasurementFrame(
            values=values,
            valid_mask=valid,
            frame_ids=torch.tensor([identity, identity]),
            episode_ids=list(episode_ids),
            camera_id=camera,
            timestamp=torch.tensor([float(step), float(step)]),
            new_measurement=torch.ones(2, dtype=torch.bool),
        )

    def feed(state, camera, frames, episode_ids, *, seed, invalid_corner=False):
        output = None
        last = None
        for step in range(frames):
            last = synthetic_frame(
                camera, step, episode_ids, seed=seed, invalid_corner=invalid_corner
            )
            output = adapter.step(last, None, state, budget)
            state = output.state
        return state, output, last

    state = adapter.init_state(2, {"camera_ids": list(camera_order), "shape": (32, 32)}, device)
    state, _, _ = feed(
        state, camera_order[0], 4, ("smoke-a", "smoke-b"), seed=1, invalid_corner=True
    )
    state, output, last_frame = feed(state, camera_order[1], 4, ("smoke-a", "smoke-b"), seed=2)
    tokens = adapter.readout(output)
    prefix = post(tokens)
    per_camera = int(adapter_config.tokens_per_camera)
    with torch.no_grad():
        repeat = adapter.step(last_frame, None, state, budget)
        tokens_repeat = adapter.readout(repeat)
        # ``last_frame`` belongs to the second camera; its counters are the ones that move.
        repeated_camera = camera_order[1]
        arrivals_before = [int(value) for value in state.arrivals[repeated_camera].tolist()]
        arrivals_after = [int(value) for value in repeat.state.arrivals[repeated_camera].tolist()]
        unique_before = [int(value) for value in state.unique_arrivals[repeated_camera].tolist()]
        unique_after = [
            int(value) for value in repeat.state.unique_arrivals[repeated_camera].tolist()
        ]
        new_episode = synthetic_frame(
            camera_order[0],
            4,
            ("smoke-new", "smoke-b"),
            seed=9,
            invalid_corner=True,
            frame_id=3,  # member 1 repeats; member 0 is reset and takes it as its first frame
        )
        switched = adapter.step(new_episode, None, repeat.state, budget)
        partial = adapter.reset(repeat.state, [True, False])
        missing_state = adapter.init_state(
            1, {"camera_ids": list(camera_order), "shape": (32, 32)}, device
        )
        missing_output = None
        for step in range(4):
            frame = synthetic_frame(
                camera_order[0], step, ("smoke-single",), seed=5, invalid_corner=False
            )
            frame.values = frame.values[:1]
            frame.valid_mask = frame.valid_mask[:1]
            frame.frame_ids = frame.frame_ids[:1]
            frame.timestamp = frame.timestamp[:1]
            frame.new_measurement = frame.new_measurement[:1]
            missing_output = adapter.step(frame, None, missing_state, budget)
            missing_state = missing_output.state
        missing_tokens = adapter.readout(missing_output)
        missing_prefix = post(missing_tokens)
        missing_slice = slice(per_camera, len(camera_order) * per_camera)
    observation = SimpleNamespace(
        state=torch.zeros(2, 1, 4),
        tokenized_prompt=torch.randint(1, 32, (2, 5), generator=torch.Generator().manual_seed(3)),
        tokenized_prompt_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    actions = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(4))
    adapter.train()
    post.train()
    adapter_trainable = [p for p in adapter.parameters() if p.requires_grad]
    post_trainable = [p for p in post.parameters() if p.requires_grad]
    parameters = adapter_trainable + post_trainable
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    update_before = [p.detach().clone() for p in parameters]
    result = bridge.forward_from_observation_with_features(
        observation, tokens, actions, return_debug=True
    )
    loss = result["action_loss"]
    loss.backward()
    adapter_grads = gradient_receipt(adapter_trainable)
    post_grads = gradient_receipt(post_trainable)
    backend_grads = sum(1 for p in policy.parameters() if p.grad is not None)
    optimizer.step()
    deltas = [
        float((p - old).detach().abs().sum())
        for p, old in zip(parameters, update_before, strict=True)
    ]
    calls_before = bridge.stats["encoder_calls"]
    steps_before = bridge.stats["denoise_steps"]
    noise = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(6))
    frozen = VisualTokens(
        values=tokens.values.detach(),
        mask=tokens.mask.detach(),
        camera_ids=list(tokens.camera_ids),
        positions=None if tokens.positions is None else tokens.positions.detach(),
    )
    sampled = bridge.sample_actions_with_features(
        observation, frozen, noise=noise.clone(), num_steps=10
    )
    sampled_again = bridge.sample_actions_with_features(
        observation, frozen, noise=noise.clone(), num_steps=10
    )
    checks = {
        "adapter_stem": adapter.stem_report(),
        "adapter_parameters": adapter.parameter_report(),
        "postprocessor_parameters": post.parameter_report(),
        "freeze": freeze,
        "token_shape": list(tokens.values.shape),
        "token_mask_shape": list(tokens.mask.shape),
        "token_camera_ids": list(tokens.camera_ids),
        "invalid_corner_cell_mask_false": bool(not tokens.mask[0, 0]),
        "prefix_shape": list(prefix.embs.shape),
        "prefix_len_is_fixed": int(prefix.pad_masks.shape[1]) == len(camera_order) * per_camera,
        "prefix_mask_matches_tokens": bool(torch.equal(prefix.pad_masks, tokens.mask)),
        "visual_prefix_scale": prefix.diagnostics["visual_prefix_scale"],
        "repeat_unique_unchanged": unique_before == unique_after,
        "repeat_one_arrival": all(
            after == before + 1
            for before, after in zip(arrivals_before, arrivals_after, strict=True)
        ),
        "repeat_tokens_identical": bool(torch.equal(tokens_repeat.values, tokens.values)),
        "episode_switch_resets_member": [
            int(value) for value in switched.state.unique_arrivals[camera_order[0]].tolist()
        ],
        "partial_reset_member0_filled": int(partial.cameras[camera_order[0]].filled[0]),
        "partial_reset_member1_filled": int(partial.cameras[camera_order[0]].filled[1]),
        "missing_camera_token_ids": list(missing_tokens.camera_ids),
        "missing_camera_block_zero": float(missing_prefix.embs[:, missing_slice].abs().sum())
        == 0.0,
        "missing_camera_block_mask_false": bool(
            not missing_prefix.pad_masks[:, missing_slice].any()
        ),
        "missing_camera_synthetic_blocks": missing_prefix.diagnostics["synthetic_camera_blocks"],
        "action_loss": float(loss.detach()),
        "adapter_gradient": adapter_grads,
        "postprocessor_gradient": post_grads,
        "backend_gradients": backend_grads,
        "parameter_update_l1": sum(deltas),
        "adapter_update_l1": sum(deltas[: len(adapter_trainable)]),
        "postprocessor_update_l1": sum(deltas[len(adapter_trainable) :]),
        "denoise_shape": list(sampled.shape),
        "denoise_finite": bool(torch.isfinite(sampled).all()),
        "denoise_repeat_max_abs": float((sampled - sampled_again).abs().max()),
        "encoder_calls_during_denoise": bridge.stats["encoder_calls"] - calls_before,
        "denoise_steps_two_calls": bridge.stats["denoise_steps"] - steps_before,
        "vision_probe": probe,
        "vision_entry_calls": probe["embed_image_calls"] + probe["vision_module_forwards"],
        "p3_support_enabled": P3_SUPPORT_ENABLED,
        "aux_raw_loss_enabled": False,
        "motion": {
            "status": output.diagnostics.get("p2_motion_status"),
            "entry": (
                None
                if output.state.correspondence.get(camera_order[1]) is None
                else output.state.correspondence[camera_order[1]].describe()
            ),
            "flow_model_calls": int(smoke_flow.calls),
            "window_frames": int(output.diagnostics["window_frames"]),
        },
    }
    failures = []
    if not checks["prefix_mask_matches_tokens"] or not checks["prefix_len_is_fixed"]:
        failures.append("prefix/mask contract")
    if not checks["repeat_unique_unchanged"] or not checks["repeat_tokens_identical"]:
        failures.append("repeat-frame semantics")
    if checks["episode_switch_resets_member"] != [1, 4]:
        failures.append("episode-change reset")
    if checks["partial_reset_member0_filled"] != 0 or checks["partial_reset_member1_filled"] < 1:
        failures.append("partial reset")
    if not checks["missing_camera_block_zero"] or not checks["missing_camera_block_mask_false"]:
        failures.append("missing-camera padding")
    if adapter_grads["total"] <= 0 or post_grads["total"] <= 0 or backend_grads:
        failures.append("gradient connectivity")
    if checks["encoder_calls_during_denoise"] != 0:
        failures.append("encoder ran inside the denoising loop")
    if checks["denoise_steps_two_calls"] != 20:
        failures.append("ten-step denoise count")
    if checks["vision_entry_calls"] != 0:
        failures.append("image tower ran")
    motion = checks["motion"]
    if motion["status"] != "motion_provider":
        failures.append("motion provider not attached")
    if motion["entry"] is None or motion["entry"]["window_frames"] != motion["window_frames"]:
        failures.append("motion window mismatch")
    if motion["entry"] is not None and motion["entry"]["valid_pairs_per_member"] != [4, 4]:
        failures.append("motion pairs not accepted")
    if motion["flow_model_calls"] <= 0:
        failures.append("motion estimator never ran")
    if failures:
        raise RuntimeError(f"raw_adapter smoke failed: {failures}: {checks}")
    return {
        "mode": "raw_adapter",
        "acceptance": False,
        "backend": "stub_interface_replica_local_cpu",
        "data": "synthetic_local_cpu",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "note": (
            "local CPU interface smoke only: the server run with the frozen backend and real "
            "NPZ windows is the P2 acceptance receipt"
        ),
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": checks,
        "failures": failures,
    }


def raw_adapter(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        return raw_adapter_smoke(args)
    return _raw_adapter_real(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = contract(args) if args.mode == "contract" else raw_adapter(args)
    if args.mode == "contract":
        receipt_name = "policy-preflight.json"
    elif args.smoke:
        receipt_name = "raw-adapter-smoke.json"
    else:
        receipt_name = "raw-adapter-preflight.json"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / receipt_name).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "platforms": sorted(report.get("platforms", {})),
                "acceptance": report.get("acceptance"),
            },
            indent=2,
        )
    )
    print(f"receipt: {args.output / receipt_name}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
