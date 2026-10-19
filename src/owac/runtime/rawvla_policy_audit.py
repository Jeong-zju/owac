"""Original π0.5 versus the feature bridge with original vision; no learned replacement."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from owac.runtime.rawvla_robotwin import get_model as get_model
from owac.runtime.rawvla_robotwin import reset_model as reset_model


def eval(task, model, observation):
    from owac.runtime.rawvla_robotwin import eval as original_eval

    if task.take_action_cnt == 0:
        cameras = ("head_camera", "left_camera", "right_camera")
        np.savez_compressed(
            Path(os.environ["ROBOTWIN_POLICY_AUDIT_DIR"]) / "initial_observation.npz",
            state=observation["joint_action"]["vector"],
            **{
                f"raw_{i}": observation["observation"][camera]["raw_linear"]
                for i, camera in enumerate(cameras)
            },
            **{
                f"rgb_{i}": observation["observation"][camera]["render_rgb"]
                for i, camera in enumerate(cameras)
            },
        )
    original_eval(task, model, observation)


def bridge_prediction(base, bridge, examples, *, state_width=None, **kwargs):
    """Preserve official preprocessing and extract real SigLIP+projector outputs."""
    import torch

    from owac.core.encoder_contract import VisualTokens

    batch = base._prepare_examples(examples, include_actions=False)
    observation = base._build_observation_from_batch(batch)
    images, masks, language, language_mask, state = base._preprocess_observation(observation)
    if state_width is not None:
        encoded = [
            base.tokenizer.tokenize(prompt, row[:state_width])
            for prompt, row in zip(batch["lang"], state[:, 0].cpu().numpy(), strict=True)
        ]
        language = torch.as_tensor(np.array([row[0] for row in encoded]), device=state.device)
        language_mask = torch.as_tensor(np.array([row[1] for row in encoded]), device=state.device)
    features = [base.vlm.embed_image(image) for image in images]
    tokens = VisualTokens(
        values=torch.cat(features, dim=1),
        mask=torch.cat(
            [
                mask[:, None].expand(-1, feature.shape[1])
                for mask, feature in zip(masks, features, strict=True)
            ],
            dim=1,
        ),
        camera_ids=tuple(base.image_keys),
    )
    return bridge.predict_action_from_visual_tokens(
        tokens,
        language,
        language_mask,
        state.squeeze(1),
        noise=kwargs.get("noise"),
        num_steps=int(kwargs.get("num_steps", 10)),
    )


def main():
    import torch
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    from owac.backends.rawvla_features import RawVLAFeaturePolicy
    from owac.runtime.rawvla_robotwin import Policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "bridge", "bridge-unpadded"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy = Policy(args.checkpoint, args.tokenizer, args.evidence)
    if args.mode in ("bridge", "bridge-unpadded"):
        base = policy.model
        bridge = RawVLAFeaturePolicy(
            base, SimpleNamespace(calls=0), protocol="robotwin", image_keys=tuple(base.image_keys)
        )

        @torch.inference_mode()
        def predict(examples, **kwargs):
            width = 14 if args.mode == "bridge-unpadded" else None
            return bridge_prediction(base, bridge, examples, state_width=width, **kwargs)

        base.predict_action = predict
    WebsocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=args.port,
        idle_timeout=-1,
        metadata={"model": f"PI05-policy-audit-{args.mode}"},
    ).serve_forever()


if __name__ == "__main__":
    main()
