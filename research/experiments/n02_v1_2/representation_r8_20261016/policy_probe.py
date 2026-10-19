"""Same fixed scene, frozen r8 policy: image interventions and flow-noise diagnosis."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from owac.backends.rawvla_features import RawVLAFeaturePolicy
from owac.core.encoder_contract import VisualTokens
from owac.runtime.rawvla_full_eval import write_json
from owac.runtime.rawvla_robotwin import normalize_state, unnormalize_actions


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def token_stats(values):
    x = values.float().reshape(3, 256, 2048)
    centred = x - x.mean(1, keepdim=True)
    samples = x[:, ::8]
    unit = torch.nn.functional.normalize(samples, dim=-1)
    cosine = unit @ unit.transpose(-1, -2)
    mask = ~torch.eye(32, device=x.device, dtype=torch.bool)
    singular = torch.linalg.svdvals(centred.reshape(768, 2048))
    power = singular.square()
    fraction = power / power.sum().clamp_min(1e-20)
    return {
        "shape": list(values.shape),
        "rms": float(x.square().mean().sqrt()),
        "abs_max": float(x.abs().max()),
        "camera_rms": x.square().mean((1, 2)).sqrt().tolist(),
        "spatial_centred_energy_fraction": float(centred.square().sum() / x.square().sum()),
        "per_camera_position_std_mean": x.std(1).mean(-1).tolist(),
        "per_camera_position_pair_cosine_mean": cosine[:, mask].mean(-1).tolist(),
        "spatial_effective_rank": float(
            torch.exp(-(fraction * fraction.clamp_min(1e-20).log()).sum())
        ),
        "bf16_equal_first_position_fraction": float(
            (x.bfloat16() == x[:, :1].bfloat16()).all(-1).float().mean()
        ),
    }


def main(mode, attempt, config_path, checkpoint_path):
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    root = checkout / f"outputs/n02-v12/representation-r8-probes/{attempt}/{mode}"
    root.mkdir(parents=True, exist_ok=False)
    previous = checkout / "outputs/n02-v12/policy-audit-20261016"
    input_path = previous / "original-normal/initial_observation.npz"
    data = np.load(input_path)
    record = json.loads((previous / "original-policy/inference.jsonl").open().readline())
    language_text = "Grab the light brown cylindrical roller on the table."
    state_original = data["state"]
    report = {
        "mode": mode,
        "optimizer_updates": 0,
        "new_rollouts": 0,
        "input_file": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "flow_seed": record["flow_seed"],
        "language": language_text,
        "raw_sha256": [hashlib.sha256(data[f"raw_{i}"].tobytes()).hexdigest() for i in range(3)],
    }
    image_visuals = {}
    if mode == "native":
        from owac.runtime.rawvla_robotwin import Policy

        policy = Policy(
            Path("/workspace/owac/rawvla/checkpoints/pi05_robotwin"),
            Path(
                "/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model"
            ),
            root / "loader",
        )
        base = policy.model
        bridge = RawVLAFeaturePolicy(
            base, SimpleNamespace(calls=0), protocol="robotwin", image_keys=tuple(base.image_keys)
        )
        stats = policy.stats
        example = {
            "image": [data[f"rgb_{i}"] for i in range(3)],
            "lang": language_text,
            "state": normalize_state(state_original, stats["state"])[None],
        }
        with torch.inference_mode():
            batch = base._prepare_examples([example], include_actions=False)
            observation = base._build_observation_from_batch(batch)
            images, masks, language, language_mask, state = base._preprocess_observation(
                observation
            )
            prefix = base.embed_prefix(images, masks, language, language_mask)
            visual = VisualTokens(prefix[0][:, :768], prefix[1][:, :768], tuple(base.image_keys))
            state = state[:, 0]
        report["processed_image_shapes"] = [list(image.shape) for image in images]
        report["native_modules"] = [
            {
                "name": n,
                "class": type(m).__name__,
                "parameters": sum(p.numel() for p in m.parameters()),
            }
            for n, m in base.named_modules()
            if n.endswith(("vision_tower", "multi_modal_projector"))
        ]
        for n, m in base.named_modules():
            if n.endswith("vision_tower") and hasattr(m, "config"):
                report["vision_config"] = m.config.to_dict()
        np.savez_compressed(
            root / "processed_images.npz",
            **{f"image_{i}": x.cpu().float().numpy() for i, x in enumerate(images)},
        )
    else:
        from owac.runtime.n02_v12_robotwin import Policy, online_window

        checkpoint = Path(checkpoint_path)
        policy = Policy(Path(config_path), checkpoint, root / "loader")
        trainer = policy.trainer
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def forbidden_step(*args, **kwargs):
            raise RuntimeError("optimizer step forbidden in visual diagnosis")

        trainer.optimizer.step = forbidden_step
        bridge, stats = trainer.bridge, policy.stats
        normalized = normalize_state(state_original, stats["state"])
        example = {
            "history": [{"step": 0, "image": [data[f"raw_{i}"] for i in range(3)]}],
            "step": 0,
            "case_id": "same-initial-scene",
            "lang": language_text,
            "normalized_state": normalized,
        }
        window = online_window(
            example, policy.cameras, history=policy.history, representation=policy.representation
        )
        with torch.inference_mode():
            language, language_mask, state = trainer.language([language_text], normalized[None])
            bridge.reset_episodes()
            visual = trainer.token_source(window)
            output = trainer.token_source.last_outputs[-1]
            readout = trainer.model.inference.readout
            spatial = output.state.geometry[0].spatial
            local = readout.spatial_output_tokens(spatial)
            report["readout"] = {
                "local_rms": float(local.square().mean().sqrt()),
                "context_contribution_rms": float((visual.values - local).square().mean().sqrt()),
                "tanh_context_gate": float(readout.context_gate.tanh()),
                "q": output.q.cpu().tolist(),
                "encoder_spatial_std_over_rms": {
                    k: float(v.flatten(2).std(2).mean() / v.square().mean().sqrt())
                    for k, v in spatial.levels.items()
                },
            }
        with torch.inference_mode():
            for label, operation in (
                ("raw_zero", np.zeros_like),
                ("raw_horizontal_flip", lambda x: x[:, ::-1].copy()),
            ):
                changed = dict(example)
                changed["history"] = [
                    {"step": 0, "image": [operation(data[f"raw_{i}"]) for i in range(3)]}
                ]
                changed_window = online_window(
                    changed,
                    policy.cameras,
                    history=policy.history,
                    representation=policy.representation,
                )
                bridge.reset_episodes()
                altered = trainer.token_source(changed_window)
                image_visuals[label] = altered.values.clone()
        report["raw_image_token_relative_l2"] = {
            label: float((values - visual.values).norm() / visual.values.norm())
            for label, values in image_visuals.items()
        }
        report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report["optimizer_state_loaded_without_updates"] = bool(trainer.optimizer.state)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["language_ids_sha256"] = digest_tensor(language)
    report["language_mask_sha256"] = digest_tensor(language_mask)
    report["state_sha256"] = digest_tensor(state)
    report["active_language_tokens"] = int(language_mask.sum())
    report["tokens"] = token_stats(visual.values)
    torch.save(
        {"tokens": visual.values.detach().cpu(), "mask": visual.mask.cpu()}, root / "tokens.pt"
    )
    x = visual.values.reshape(1, 3, 256, 2048)
    alternatives = {
        "all_zero": torch.zeros_like(visual.values),
        "spatial_mean": x.mean(2, keepdim=True).expand_as(x).reshape_as(visual.values).contiguous(),
        "camera_swap_head_left": x[:, [1, 0, 2]].reshape_as(visual.values).contiguous(),
        "spatial_reverse": x.flip(2).reshape_as(visual.values).contiguous(),
    }
    alternatives.update(image_visuals)
    predictions = {}
    seed = int(report["flow_seed"])
    with torch.inference_mode():
        for offset in (0, 50, 100, 150, 200):
            generator = torch.Generator(device="cuda").manual_seed((seed + offset) % 2**32)
            noise = torch.randn((1, 50, 32), generator=generator, device="cuda")
            prediction = bridge.predict_action_from_visual_tokens(
                visual, language, language_mask, state, noise=noise, num_steps=10
            )
            predictions[f"noise_{offset}"] = np.asarray(prediction["normalized_actions"])[0]
        for name, values in alternatives.items():
            generator = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn((1, 50, 32), generator=generator, device="cuda")
            replacement = VisualTokens(values, visual.mask, visual.camera_ids)
            prediction = bridge.predict_action_from_visual_tokens(
                replacement, language, language_mask, state, noise=noise, num_steps=10
            )
            predictions[name] = np.asarray(prediction["normalized_actions"])[0]
    physical = {k: unnormalize_actions(v, stats["actions"]) for k, v in predictions.items()}
    joint_ids = [*range(6), *range(7, 13)]
    baseline = physical["noise_0"]
    report["action_baseline"] = {
        "first_target_vs_current_joint_max_rad": float(
            np.abs(baseline[0, joint_ids] - state_original[joint_ids]).max()
        ),
        "within_chunk_max_joint_target_delta_rad": float(
            np.abs(np.diff(baseline[:, joint_ids], axis=0)).max()
        ),
    }
    report["counterfactuals"] = {
        k: {
            "joint_target_delta_rms_rad": float(
                np.sqrt(np.mean((v[:, joint_ids] - baseline[:, joint_ids]) ** 2))
            ),
            "first_target_delta_max_rad": float(
                np.abs(v[0, joint_ids] - baseline[0, joint_ids]).max()
            ),
            "all_target_delta_max_rad": float(
                np.abs(v[:, joint_ids] - baseline[:, joint_ids]).max()
            ),
        }
        for k, v in physical.items()
        if k != "noise_0"
    }
    np.savez_compressed(root / "predictions.npz", **physical)
    write_json(root / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw-r8",), default="raw-r8")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--attempt", default="a1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    main(args.mode, args.attempt, args.config, args.checkpoint)
