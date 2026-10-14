"""N01 v1.0 early policy preflight, ``mode=contract`` (the P1 hand-off check).

``docs/prompts/n01_v1_0/EARLY_POLICY_PREFLIGHT.md`` defines two modes and this
entry point implements the one P1 owes:

* ``contract`` (P1): reuse the already-accepted reference encoder and the real
  feature bridge on real RawVLA NPZ batches, and verify that the frozen pi0.5
  backend plus the trainable bridge/encoder are connected -- gradients reach the
  bridge and the encoder, the backend accumulates none, the padded action steps
  are excluded, and a validation batch is checked without an update.  It also
  records the four-region path binding and the new ``SourceKey`` mapping with a
  real unique-source count, because the fixed protocol paths are what the later
  phases will consume.
* ``raw_adapter`` (P2): not implemented here; it needs the three-channel DBSR
  adapter and the causal window, so the mode is refused instead of faked.

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
import json
import platform
import sys
import time
from collections.abc import Sequence
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
    return top


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
        parameters = [p for p in encoder.parameters() if p.requires_grad]
        parameters += [p for p in bridge.postprocessor.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
        before = {name_: p.detach().clone() for name_, p in encoder.named_parameters()}
        rows = []
        if args.crop > 0:
            entry["crop"] = args.crop
        # Spread the requested real batches across the train region so two steps are
        # genuinely different data, not two windows of the same episode.
        stride = max(len(train_set) // max(args.steps, 1), 1)
        for step in range(max(args.steps, 1)):
            sample = train_set[(step * stride) % len(train_set)]
            batch = collate_sequence_samples([sample])
            _crop_batch(batch, args.crop)
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
            encoder_grads = {
                "total": float(
                    sum(float(p.grad.detach().norm()) for p in parameters if p.grad is not None)
                ),
                "finite": all(
                    bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None
                ),
            }
            backend_gradients = sum(1 for p in policy.parameters() if p.grad is not None)
            if backend_gradients:
                raise RuntimeError("the frozen backend accumulated a parameter gradient")
            if encoder_grads["total"] <= 0.0 or not encoder_grads["finite"]:
                raise RuntimeError(f"invalid encoder gradient: {encoder_grads}")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
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
        validation_snapshot = {
            key: value.detach().clone()
            for key, value in encoder.named_parameters()
            if "stem" in key
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
        parameter_drift = float(
            sum(
                float((p - validation_snapshot[name_]).abs().sum())
                for name_, p in encoder.named_parameters()
                if name_ in validation_snapshot
            )
        )
        entry["validation"] = {
            "rows": validation_rows,
            "parameter_drift_after_validation": parameter_drift,
        }
        entry["encoder_parameter_update"] = float(
            sum(
                float((p - before[name_]).abs().sum())
                for name_, p in encoder.named_parameters()
                if name_ in before
            )
        )
        entry["load_audit"] = audit if isinstance(audit, (dict, list, str)) else str(type(audit))
        entry["ledger_source_check"] = ledger_source_check(
            train_set, name, batches=args.ledger_batches, crop=args.crop
        )
        report["platforms"][name] = entry
        del policy, bridge, encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "raw_adapter":
        print(
            "mode=raw_adapter is the P2 hand-off: it needs the three-channel DBSR adapter "
            "and the causal window, which do not exist yet. Refusing instead of faking it.",
            file=sys.stderr,
        )
        return 2
    report = contract(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "policy-preflight.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(json.dumps({"platforms": sorted(report["platforms"])}, indent=2))
    print(f"receipt: {args.output / 'policy-preflight.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
