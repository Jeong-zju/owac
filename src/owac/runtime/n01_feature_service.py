"""Independent loopback policy service for calibrated RAW -> N01 -> pi0.5.

Wire contract: normalized current state [B,D], language strings, and frames
mapping camera names to calibrated FP32 BCHW values with episode/frame IDs.
Returns normalized native action chunks; the client owns audited action
unnormalization and physical execution. This service never runs the image tower.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from owac.backends.rawvla_features import ActionChunkCache
from owac.core.encoder_contract import MeasurementFrame, VisualTokens
from owac.frameworks.n01.config import N01Config
from owac.training.n01_action import build_bridge, load_backend, observation_from_batch


class N01FeatureService:
    def __init__(self, policy, encoder, bridge, *, device="cuda"):
        self.policy, self.encoder, self.bridge = policy, encoder, bridge
        self.device = device
        if (
            policy.action_horizon != bridge.protocol.action_horizon
            or policy.action_dim != bridge.protocol.model_action_dim
        ):
            raise ValueError("policy dimensions do not match the selected suite action protocol")
        self.states = {}
        self.episodes = {}
        self.metadata = {
            "interface": "owac-n01-calibrated-raw-v1",
            "suite": bridge.protocol.suite,
            "actions": "normalized",
            "action_horizon": policy.action_horizon,
            "action_dim": policy.action_dim,
            "control_dim": bridge.protocol.control_dim,
            "replan_steps": bridge.protocol.replan_steps,
            "visual_tokens_per_camera": encoder.config.readout_tokens,
            "cameras": list(bridge.postprocessor.camera_order),
        }

    def reset(self, batch_mask=None):
        if batch_mask is None:
            self.states.clear()
            self.episodes.clear()
        else:
            for camera, state in self.states.items():
                mask = torch.as_tensor(batch_mask, device=self.device, dtype=torch.bool)
                self.states[camera] = self.encoder.reset(state, mask)
                for index in mask.nonzero().flatten().tolist():
                    self.episodes[camera][index] = None
        self.bridge.reset_encoder_state(batch_mask)

    def predict_action(self, **payload):
        # The pinned StarVLA WebSocket server dispatches predict_action(**payload).
        if payload.pop("reset", False):
            self.reset(payload.pop("reset_batch_mask", None))
        return self.infer(payload)

    @torch.inference_mode()
    def infer(self, data):
        tokens, counts, costs = [], {}, {}
        for camera, payload in data["frames"].items():
            if camera not in self.bridge.postprocessor.camera_order:
                raise ValueError(f"unknown camera {camera!r}")
            if payload.get("value_units") != "normalized_raw_measurement":
                raise ValueError("explicit calibrated RAW units are required")
            values = torch.as_tensor(payload["values"], device=self.device, dtype=torch.float32)
            if values.ndim != 4:
                raise ValueError("frame values must be BCHW, not PIL/RGB")
            b, _, h, w = values.shape
            episodes = list(payload["episode_ids"])
            if len(episodes) != b:
                raise ValueError("one episode ID is required per batch member")
            if not self.states:
                self.bridge.action_cache = ActionChunkCache(self.bridge.protocol.replan_steps, b)
            if camera not in self.states:
                self.states[camera] = self.encoder.init_state(
                    b, {"camera_ids": [camera], "shape": (h, w)}, self.device
                )
                self.episodes[camera] = episodes
                self.states[camera].time_unit = payload.get("time_unit", "frame")
            else:
                changed = torch.tensor(
                    [old != new for old, new in zip(self.episodes[camera], episodes, strict=True)],
                    device=self.device,
                )
                if bool(changed.any()):
                    self.states[camera] = self.encoder.reset(self.states[camera], changed)
                    self.bridge.action_cache.reset(changed)
                self.episodes[camera] = episodes

            def optional(key, dtype, payload=payload):
                value = payload.get(key)
                return (
                    None
                    if value is None
                    else torch.as_tensor(value, device=self.device, dtype=dtype)
                )

            output = self.encoder.step(
                MeasurementFrame(
                    values,
                    valid_mask=optional("valid_mask", torch.bool),
                    bad_mask=optional("bad_mask", torch.bool),
                    saturated_mask=optional("saturated_mask", torch.bool),
                    frame_ids=optional("frame_ids", torch.long),
                    camera_id=camera,
                    timestamp=optional("timestamp", torch.float32),
                    time_unit=payload.get("time_unit", "frame"),
                    new_measurement=optional("new_measurement", torch.bool),
                    metadata={"variance": optional("variance", torch.float32)},
                ),
                state=self.states[camera],
            )
            self.states[camera] = output.state
            tokens.append(self.encoder.readout(output))
            counts[camera] = output.state.filled.cpu().tolist()
            costs[camera] = output.cost
        if not tokens:
            raise ValueError("at least one calibrated camera frame is required")
        visual = VisualTokens(
            torch.cat([t.values for t in tokens], 1),
            torch.cat([t.mask for t in tokens], 1),
            [c for t in tokens for c in t.camera_ids],
            torch.cat([t.positions for t in tokens], 1),
        )
        language = data["language"]
        if isinstance(language, str):
            language = [language] * visual.values.shape[0]
        observation = observation_from_batch(
            self.policy,
            {
                "state": torch.as_tensor(np.asarray(data["normalized_state"])),
                "meta": [{"task_description": text} for text in language],
            },
            self.device,
        )
        actions = self.bridge.sample_actions_with_features(observation, visual, num_steps=10)
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("non-finite normalized action chunk")
        return {
            "normalized_actions": actions.cpu().numpy(),
            "unique_frames": counts,
            "cost": costs,
            "encoder_steps": len(tokens),
            "denoising_steps": 10,
        }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("libero", "robotwin2"), required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=18120)
    args = p.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get("suite") != args.suite:
        raise ValueError("checkpoint suite must match the requested service suite")
    config = N01Config.from_dict(checkpoint["config"])
    policy, _audit = load_backend(args.suite, args.weights, args.tokenizer, args.device)
    encoder, bridge = build_bridge(
        policy,
        config,
        checkpoint["camera_order"],
        args.device,
        checkpoint.get("method", "n01"),
        suite=args.suite,
    )
    encoder.load_state_dict(checkpoint["encoder"])
    bridge.postprocessor.load_state_dict(checkpoint["postprocessor"])
    encoder.eval()
    service = N01FeatureService(policy, encoder, bridge, device=args.device)
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    WebsocketPolicyServer(
        policy=service, host="127.0.0.1", port=args.port, idle_timeout=-1, metadata=service.metadata
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
