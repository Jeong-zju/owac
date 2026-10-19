# ruff: noqa: E501  (verbatim device-side probe receipt; not part of the package)
#!/usr/bin/env python3
"""LIBERO: real forward/backward + one real sample through the real pi0.5 backend, plus the
declared protocol checked against the frozen loader. Evidence only; writes a small JSON."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

W = Path("/workspace/owac/worktrees/n02-measurement-completion")
ROOT = "/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench"
STATS = (
    "/workspace/owac/rawvla/checkpoints/pi05_libero_jax/assets/physical-intelligence/libero/"
    "norm_stats.json"
)
MANIFEST = W / "outputs/n02-v12/P2/measurement-completion/manifest-libero.json"
ANCHORS = W / "outputs/n02-full/p0-independent-audit/frozen-data"
CKPT = "/workspace/owac/rawvla/source/openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors"
TOKENIZER = "/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model"
OUT = W / "outputs/n02-v12/P2/probes/libero_protocol_probe.json"

from owac.backends.rawvla_features import (  # noqa: E402
    ActionChunkCache,
    StaticTokenSource,
    build_feature_policy,
    load_checkpoint_state,
    protocol_for,
)
from owac.core.encoder_contract import VisualTokens  # noqa: E402
from owac.evaluation.n02_full import v12 as v12_eval  # noqa: E402
from owac.frameworks.n02_full.data_v12 import V12WindowSource  # noqa: E402
from owac.frameworks.n02_full.inference_v12 import V12InferenceInput  # noqa: E402
from owac.frameworks.n02_full.model_v12 import build_n02_v12_model  # noqa: E402
from owac.raw.rawvla_sequence import FORMAT_SPECS  # noqa: E402


def main() -> int:
    receipt: dict = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = FORMAT_SPECS["libero"]
    cameras = tuple(spec.camera_ids)
    source = V12WindowSource(
        dataset_format="libero",
        data_root=ROOT,
        partition="dev",
        history=8,
        stats_path=STATS,
        manifest_path=str(MANIFEST) if MANIFEST.is_file() else None,
        anchor_path=str(ANCHORS / "libero-dev-anchors.json.gz"),
        batch_size=1,
    )
    index = source.anchor_count // 2
    batch = source.window(index, index + 1)
    sample = source.sample(index)
    targets = source.targets(index, index + 1)

    frames = {c: np.asarray(batch.observation.frames[c].values) for c in cameras}
    receipt["data"] = {
        "anchor_index": int(index),
        "anchor": source.anchor(index).as_dict(),
        "rows": int(np.asarray(batch.frame_index).size),
        "frame_index": [int(item) for item in np.asarray(batch.frame_index).tolist()],
        "cameras": list(cameras),
        "measurement_keys": list(spec.measurement_keys),
        "raw_units": "declared by the loader; observed below",
        "per_camera": {
            c: {
                "shape": list(frames[c].shape),
                "dtype": str(frames[c].dtype),
                "min": float(frames[c].min()),
                "max": float(frames[c].max()),
            }
            for c in cameras
        },
        "proprio_shape": list(np.asarray(sample.proprio).shape)
        if sample.proprio is not None
        else None,
        "proprio_available": bool(sample.proprio_available),
        "supervision_actions_shape": list(np.asarray(sample.supervision_actions).shape),
        "supervision_indices": [int(i) for i in np.asarray(sample.supervision_indices)[:12]],
        "action_shape_from_targets": list(np.asarray(targets["actions"]).shape),
        "action_dim": int(targets["action_dim"]),
        "action_horizon": int(targets["action_horizon"]),
        "action_offset": int(targets["action_offset"]),
        "window_options": dataclasses.asdict(source.window_options),
        "task": str(sample.provenance.get("task_description", ""))[:120],
        "sample_fields": sorted(f.name for f in dataclasses.fields(sample)),
    }
    protocol = protocol_for("libero")
    cache = ActionChunkCache(protocol)
    cache.store(0, 0, torch.zeros(4, 7))
    receipt["protocol"] = {
        "declared": protocol.as_dict(),
        "needs_replan_steps_0_4_5": [bool(cache.needs_replan(0, s)) for s in (0, 4, 5)],
        "camera_ids": list(cameras),
        "n_cameras": len(cameras),
        "loader_state_source_key": spec.state_source_key,
        "loader_state_notes": spec.state_notes,
        "loader_state_is_joint_vector": bool(spec.state_is_joint_vector),
        "raw_uint8_over_255": "see data.per_camera min/max (already divided by 255 by the loader)"
        if max(float(frames[c].max()) for c in cameras) <= 1.5
        else "NOT in [0,1]: inspect before claiming uint8/255",
    }

    settings = v12_eval.resolve_settings(
        v12_eval.build_parser().parse_args(
            [
                "--dataset-format",
                "libero",
                "--camera-ids",
                ",".join(cameras),
                "--history",
                "8",
                "--query-stride",
                "8",
                "--d-model",
                "512",
                "--k",
                "4",
                "--n-slots",
                "16",
                "--slot-dim",
                "512",
                "--n-queries",
                "256",
            ]
        )
    )
    settings.model.update(
        {
            "camera_ids": list(cameras),
            "dims": (128, 256, 512, 1024),
            "depths": (3, 3, 27, 3),
            "out_dim": 2048,
            "n_latents": 128,
            "decoder_hidden": 256,
            "measurement_channels": 3,
            "input_mode": "RAW3",
            "alignment": 32,
        }
    )
    torch.manual_seed(1103)
    model = build_n02_v12_model(dict(settings.model)).to(device)
    model.eval()
    receipt["model"] = {
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "camera_ids": list(cameras),
        "settings": {k: v for k, v in settings.model.items() if k != "camera_ids"},
    }

    protocol_denoise = int(protocol.num_denoise_steps)
    tokens_per_camera = 256
    policy = build_feature_policy(
        "libero",
        Path(TOKENIZER),
        StaticTokenSource(
            VisualTokens(
                values=torch.zeros((1, len(cameras) * tokens_per_camera, 2048)),
                mask=torch.ones((1, len(cameras) * tokens_per_camera), dtype=torch.bool),
                camera_ids=cameras,
                positions=torch.zeros((1, len(cameras) * tokens_per_camera, 2)),
            )
        ),
        token_dim=2048,
        tokens_per_camera=tokens_per_camera,
        image_keys=cameras,
    )
    audit = policy.load_audited_replaced_vision(load_checkpoint_state(CKPT))
    policy.base.to(device)
    for module in policy.bridge_modules():
        module.to(device)
    receipt["backend"] = {
        "checkpoint": CKPT,
        "loaded_keys": audit.get("loaded_keys"),
        "unexpected": audit.get("unexpected"),
        "active_missing": audit.get("active_missing"),
        "max_state_dim": int(policy._max_state_dim()),
        "action_dim_internal": int(policy._action_dim()),
        "action_horizon_internal": int(policy._action_horizon()),
    }

    row = int(np.asarray(batch.frame_index).size) - 1
    observation = v12_eval.row_observation(batch.observation, row)
    inp = V12InferenceInput(
        observation=observation,
        declared_prior=None,
        frame_index=torch.tensor([int(np.asarray(batch.frame_index)[row])]),
        measurement_ids=v12_eval._declared_measurement_ids(observation, [row]),
    )
    state = model.build_state(batch=1, device=device)
    out = model.step(inp, state)
    tokens = out.tokens
    receipt["api_shapes"] = {
        "tokens": list(tokens.shape),
        "q": list(out.q.shape),
        "energies": list(out.energies.shape),
        "energy_mean": float(out.energies.mean()),
        "tokens_finite": bool(torch.isfinite(tokens).all()),
        "queries_per_camera": [int(i) for i in out.diagnostics["per_camera_query_count"]],
        "coverage": [round(float(i), 4) for i in out.diagnostics["coverage"]],
    }

    visual = VisualTokens(
        values=tokens.detach().clone().requires_grad_(True),
        mask=torch.ones(tokens.shape[:2], dtype=torch.bool, device=torch.device(device)),
        camera_ids=cameras,
        positions=torch.zeros(tokens.shape[:2] + (2,), device=torch.device(device)),
    )
    proprio = v12_eval._anchor_proprio(batch, policy)
    texts = [str(sample.provenance.get("task_description", ""))]
    lang_tokens, lang_masks, lang_info = v12_eval._language_batch(
        policy, texts, v12_eval._anchor_proprio(batch, policy, pad=False), settings
    )
    actions = policy.prepare_control_targets(
        torch.as_tensor(targets["actions"], dtype=torch.float32)
    )
    action_mask = torch.as_tensor(targets["action_mask"]).reshape(1, -1)
    generator = torch.Generator().manual_seed(1103)
    noise = torch.randn(actions.shape, generator=generator, dtype=torch.float32)
    time_tensor = torch.full((actions.shape[0],), 0.5, dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    forward_started = time.time()
    result = policy.forward_from_visual_tokens(
        visual,
        lang_tokens.to(device),
        lang_masks.to(device),
        torch.as_tensor(proprio, dtype=torch.float32, device=device),
        actions.to(device),
        noise=noise.to(device),
        time=time_tensor.to(device),
        action_mask=action_mask.to(device),
        return_debug=True,
    )
    loss = result["action_loss"]
    grads = model.gradient_report(loss)
    receipt["forward_backward"] = {
        "status": "ran",
        "action_loss_flow_matching": float(loss.detach()),
        "elementwise_mse_check": float(
            torch.nn.functional.mse_loss(
                result["v_t"].detach(), result["u_t"].detach(), reduction="none"
            ).mean()
        ),
        "loss_requires_grad": bool(loss.requires_grad),
        "language": {k: v for k, v in lang_info.items() if k != "text"},
        "per_role_gradient_l1": {
            role: {k: round(float(v), 6) for k, v in payload.items()}
            for role, payload in grads.items()
        },
        "seconds": round(time.time() - forward_started, 2),
        "gradient_paths": model.gradient_paths(),
    }
    # --- gradient path: the SAME forward, but with the model's own graph kept ---------------
    model.zero_grad(set_to_none=True)
    graph_out = model.step(inp, state)
    graph_visual = VisualTokens(
        values=graph_out.tokens,
        mask=torch.ones(graph_out.tokens.shape[:2], dtype=torch.bool, device=torch.device(device)),
        camera_ids=cameras,
        positions=torch.zeros(graph_out.tokens.shape[:2] + (2,), device=torch.device(device)),
    )
    graph_result = policy.forward_from_visual_tokens(
        graph_visual,
        lang_tokens.to(device),
        lang_masks.to(device),
        torch.as_tensor(proprio, dtype=torch.float32, device=device),
        actions.to(device),
        noise=noise.to(device),
        time=time_tensor.to(device),
        action_mask=action_mask.to(device),
        return_debug=True,
    )
    graph_loss = graph_result["action_loss"]
    graph_grads = model.gradient_report(graph_loss)
    receipt["gradient_path_from_tokens"] = {
        "status": "ran",
        "action_loss": float(graph_loss.detach()),
        "tokens_require_grad": bool(graph_out.tokens.requires_grad),
        "tokens_have_grad_fn": bool(graph_out.tokens.grad_fn is not None),
        "per_role_gradient_l1": {
            role: {k: round(float(v), 6) for k, v in payload.items()}
            for role, payload in graph_grads.items()
        },
        "note": (
            "this forward keeps the N02 graph (no detach): the loss is computed on the very "
            "tokens the model produced, so a non-zero encoder L1 is real gradient flow. The "
            "earlier 'forward_backward' section deliberately detached the tokens first and only "
            "exercises the frozen bridge."
        ),
    }

    sample_started = time.time()
    prediction = policy.predict_action_from_visual_tokens(
        visual,
        lang_tokens.to(device),
        lang_masks.to(device),
        torch.as_tensor(proprio, dtype=torch.float32, device=device),
        num_steps=protocol_denoise,
    )
    control = np.asarray(prediction["control_actions"])
    receipt["sampling"] = {
        "status": "ran",
        "control_actions_shape": list(control.shape),
        "expected_shape": [1, protocol.action_horizon, protocol.control_dim],
        "shape_matches_expected": list(control.shape)
        == [1, protocol.action_horizon, protocol.control_dim],
        "num_denoise_steps": int(prediction["num_denoise_steps"]),
        "finite": bool(np.isfinite(control).all()),
        "first_chunk_row": [round(float(v), 5) for v in control[0, 0]],
        "reference_first_row": [
            round(float(v), 5) for v in np.atleast_2d(np.asarray(targets["actions"]))[0]
        ],
        "seconds": round(time.time() - sample_started, 2),
    }
    if torch.cuda.is_available():
        receipt["memory"] = {
            "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "device": torch.cuda.get_device_name(0),
        }
    receipt["elapsed_seconds"] = round(time.time() - started, 1)
    OUT.write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("data", "model")}, indent=1, default=str
        )[:3000]
    )
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
