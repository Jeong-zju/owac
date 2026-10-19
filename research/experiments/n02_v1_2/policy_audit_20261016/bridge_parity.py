"""Real-model native-prefix, flow-forward and sampler parity, plus r5 language ablation."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from owac.backends.rawvla_features import RawVLAFeaturePolicy
from owac.core.encoder_contract import VisualTokens
from owac.evaluation.n02_full.v12 import _language_batch
from owac.runtime.rawvla_full_eval import paired_seed, write_json
from owac.runtime.rawvla_robotwin import Policy, normalize_state, unnormalize_actions
from owac.training.n02_v12 import N02V12Trainer


def main():
    root = Path("outputs/n02-v12/policy-audit-20261016")
    evidence = root / "bridge-parity"
    evidence.mkdir(exist_ok=False)
    policy = Policy(
        Path("/workspace/owac/rawvla/checkpoints/pi05_robotwin"),
        Path("/workspace/owac/rawvla/source/openpi_converted_protocol/paligemma_tokenizer.model"),
        evidence / "native-load",
    )
    base = policy.model
    input_path = root / "original-normal/initial_observation.npz"
    deadline = time.monotonic() + 240
    while not input_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("reference simulator did not provide a real initial observation")
        time.sleep(1)
    data = np.load(input_path)
    demo = Path("outputs/n02-v12/demos/r5-lighting-20261016/raw/episodes.jsonl")
    record = next(
        json.loads(line)
        for line in demo.read_text().splitlines()
        if json.loads(line)["entry"]["lighting_domain"] == "Normal"
    )
    example = {
        "image": [data[f"rgb_{i}"] for i in range(3)],
        "lang": record["language"],
        "state": normalize_state(data["state"], policy.stats["state"])[None],
    }
    bridge = RawVLAFeaturePolicy(
        base, SimpleNamespace(calls=0), protocol="robotwin", image_keys=tuple(base.image_keys)
    )
    trainer = object.__new__(N02V12Trainer)
    trainer.device, trainer.bridge = torch.device("cuda"), bridge
    report = {"optimizer_updates": 0, "input": str(input_path), "instruction": example["lang"]}
    with torch.inference_mode():
        batch = base._prepare_examples([example], include_actions=False)
        observation = base._build_observation_from_batch(batch)
        images, masks, language, language_mask, state = base._preprocess_observation(observation)
        prefix = base.embed_prefix(images, masks, language, language_mask)
        length = len(base.image_keys) * 256
        tokens = VisualTokens(prefix[0][:, :length], prefix[1][:, :length], tuple(base.image_keys))
        feature_prefix = bridge.embed_prefix_from_visual_tokens(tokens, language, language_mask)
        report["prefix"] = {
            "embedding_max_abs_difference": float((prefix[0] - feature_prefix.embeds).abs().max()),
            "padding_masks_identical": bool(torch.equal(prefix[1], feature_prefix.pad_masks)),
            "attention_masks_identical": bool(torch.equal(prefix[2], feature_prefix.att_masks)),
            "shape": list(prefix[0].shape),
            "visual_rms": float(tokens.values.float().square().mean().sqrt()),
        }
        entry = record["entry"]
        seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn((1, 50, 32), generator=generator, device="cuda")
        native = base.sample_actions(observation, noise=noise, num_steps=10)
        replacement = bridge.predict_action_from_visual_tokens(
            tokens, language, language_mask, state[:, 0], noise=noise
        )["normalized_actions"]
        replacement = torch.as_tensor(replacement, device="cuda")
        report["sampler_max_abs_difference"] = float((native - replacement).abs().max())
        old_tokens, old_mask, old_state = trainer.language([example["lang"]], example["state"])
        eval_tokens, eval_mask, _ = _language_batch(
            bridge, [example["lang"]], example["state"], None
        )
        ablated = bridge.predict_action_from_visual_tokens(
            tokens, old_tokens, old_mask, old_state, noise=noise
        )["normalized_actions"]
        report["language"] = {
            "native_state_width": int(state.shape[-1]),
            "r5_state_width_passed_to_tokenizer": 14,
            "native_active_tokens": int(language_mask.sum()),
            "r5_active_tokens": int(old_mask.sum()),
            "native_and_r5_token_ids_identical": bool(torch.equal(language, old_tokens)),
            "r5_train_and_eval_token_ids_identical": bool(
                torch.equal(old_tokens.cpu(), eval_tokens)
            ),
            "r5_train_and_eval_masks_identical": bool(torch.equal(old_mask.cpu(), eval_mask)),
            "native_text": base.tokenizer._load().decode(
                language[0, language_mask[0]].cpu().tolist()
            ),
            "r5_text": base.tokenizer._load().decode(old_tokens[0, old_mask[0]].cpu().tolist()),
            "padded_action_expert_state_identical": bool(torch.equal(state[:, 0], old_state)),
        }
        report["state_token_only_ablation"] = {
            "normalized_action_max_abs_difference": float(
                np.max(np.abs(native.cpu().numpy() - ablated))
            ),
            "physical_action_max_abs_difference": float(
                np.max(
                    np.abs(
                        unnormalize_actions(native.cpu().numpy()[0], policy.stats["actions"])
                        - unnormalize_actions(ablated[0], policy.stats["actions"])
                    )
                )
            ),
        }
        actions = torch.zeros_like(noise)
        flow_time = torch.tensor([0.5], device="cuda")
        official = base.forward_from_observation(
            observation, actions, noise=noise, time=flow_time, return_debug=True
        )
        migrated = bridge.forward_from_visual_tokens(
            tokens,
            language,
            language_mask,
            state[:, 0],
            actions,
            noise=noise,
            time=flow_time,
            return_debug=True,
        )
        report["flow_forward"] = {
            "velocity_max_abs_difference": float((official["v_t"] - migrated["v_t"]).abs().max()),
            "native_loss": float(official["action_loss"]),
            "bridge_loss": float(migrated["action_loss"]),
        }
    write_json(evidence / "parity.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
