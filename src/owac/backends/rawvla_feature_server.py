"""Independent loopback HTTP policy service using the frozen StarVLA feature path.

POST /infer accepts cameras: {camera_id: HWC RAW values}, current proprio,
task, episode_id, frame_id, and optional executed_action (already executed).
POST /reset clears all temporal state. No simulator or robot is controlled.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class OnlineFeaturePolicy:
    def __init__(
        self,
        *,
        suite,
        weights,
        stats,
        tokenizer,
        device="cuda",
        baseline=None,
        encoder_checkpoint=None,
    ):
        import torch
        from safetensors.torch import load_file

        from owac.backends.rawvla import new_model
        from owac.backends.rawvla_features import load_feature_audited

        self.torch, self.suite, self.device = torch, suite, device
        self.model = new_model(suite, Path(tokenizer))
        self.audit = load_feature_audited(self.model, load_file(weights))
        self.model.to(device).eval().requires_grad_(False)
        statistics = json.loads(Path(stats).read_text())
        self.stats = (
            next(iter(statistics.values())) if suite == "libero" else statistics["norm_stats"]
        )
        self.baseline, self.encoder_checkpoint = baseline, encoder_checkpoint
        self.encoder = self.bridge = self.states = self.previous = None
        self.episode = None

    def normalize(self, value, key, inverse=False):
        import numpy as np

        value = np.asarray(value, dtype=np.float32)
        item = self.stats[key]
        if self.suite == "libero":
            low, high = np.asarray(item["q01"]), np.asarray(item["q99"])
            return (
                (value + 1) / 2 * (high - low + 1e-6) + low
                if inverse
                else 2 * (value - low) / (high - low + 1e-6) - 1
            )
        mean, std = np.asarray(item["mean"]), np.asarray(item["std"])
        return value * (std + 1e-8) + mean if inverse else (value - mean) / (std + 1e-8)

    def reset(self):
        self.states = self.previous = None
        self.episode = None

    def predict_action(self, **payload):
        import numpy as np

        from owac.backends.rawvla_features import (
            FeatureBridgeConfig,
            RawVLAFeaturePolicy,
            _FeatureBackendView,
            tokenize_backend_task,
        )
        from owac.core.encoder_contract import MeasurementFrame, VisualTokens
        from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
        from owac.training.n04 import (
            RawVLAFrame,
            build_encoder,
            build_real_context,
            load_checkpoint,
        )

        torch = self.torch
        cameras = payload["cameras"]
        names = tuple(cameras)
        episode = payload["episode_id"]
        height, width = np.asarray(cameras[names[0]]).shape[:2]
        if self.encoder is None:
            cfg = N04Config(
                measurement_size=(height, width),
                camera_ids=names,
                geometry=N04GeometryConfig(feature_size=(height // 4, width // 4), pixel_stride=4),
                model=N04ModelConfig(readout_tokens=256, readout_dim=2048),
            )
            self.encoder = build_encoder(cfg, baseline=self.baseline).to(self.device).eval()
            if self.encoder_checkpoint:
                load_checkpoint(self.encoder_checkpoint, encoder=self.encoder)
            self.bridge = RawVLAFeaturePolicy(
                self.encoder,
                self.model,
                FeatureBridgeConfig(
                    control_dim=7 if self.suite == "libero" else 14,
                    action_horizon=10 if self.suite == "libero" else 50,
                ),
            )
        if self.states is None or episode != self.episode:
            self.states = self.encoder.init_state(1, camera_ids=names, device=self.device)
            self.previous, self.episode = None, episode
        if names != tuple(self.states):
            raise ValueError("camera set changed; reset service with a consistent camera set")
        proprio = torch.tensor(payload["proprio"], device=self.device, dtype=torch.float32)[None]
        expected = 8 if self.suite == "libero" else 14
        if proprio.shape != (1, expected):
            raise ValueError(f"current proprio must have {expected} dimensions")
        executed = payload.get("executed_action")
        control_dim = 7 if self.suite == "libero" else 14
        action = (
            torch.zeros(1, control_dim, device=self.device)
            if executed is None
            else torch.tensor(executed, device=self.device, dtype=torch.float32)[None]
        )
        if action.shape != (1, control_dim):
            raise ValueError(f"executed_action must have {control_dim} dimensions")
        frame = RawVLAFrame(
            measured=None,
            valid_mask=None,
            saturated_mask=None,
            proprio=proprio,
            action=action,
            metadata={"suite": self.suite, "executed_action_available": executed is not None},
        )
        values = []
        with torch.no_grad():
            for i, name in enumerate(names):
                raw = torch.tensor(
                    np.asarray(cameras[name]), device=self.device, dtype=torch.float32
                )
                if raw.shape != (height, width, 3):
                    raise ValueError(
                        "all cameras must supply equal HWC three-channel RAW measurements"
                    )
                raw = raw.permute(2, 0, 1)[None]
                white = payload.get("raw_white_level", 255 if self.suite == "libero" else None)
                saturated = (
                    torch.zeros_like(raw[:, :1])
                    if white is None
                    else (raw >= white).any(1, keepdim=True).float()
                )
                measurement = MeasurementFrame(
                    values=raw / 255 if self.suite == "libero" else raw,
                    valid_mask=torch.ones_like(raw[:, :1]),
                    saturated_mask=saturated,
                    camera_id=name,
                    episode_ids=(episode,),
                    frame_ids=(payload["frame_id"],),
                )
                context = build_real_context(
                    frame, self.previous, camera_index=torch.tensor([i], device=self.device)
                )
                context.camera_id = name
                visual, self.states[name] = self.bridge.encode_observation(
                    measurement, context, self.states[name]
                )
                values.append(visual)
            backend_state = torch.tensor(
                self.normalize(payload["proprio"], "state"), device=self.device, dtype=torch.float32
            )[None]
            task, mask = tokenize_backend_task(self.model, payload["task"], backend_state[0])
            task, mask = (
                torch.as_tensor(task, device=self.device)[None],
                torch.as_tensor(mask, device=self.device)[None],
            )
            visual = VisualTokens(
                torch.cat([v.values for v in values], 1), torch.cat([v.mask for v in values], 1)
            )
            prefix = self.bridge.embed_prefix_from_visual_tokens(visual, task, mask)
            view = _FeatureBackendView(self.model, prefix)
            observation = self.bridge._observation(context, task, mask, backend_state)
            sampled = type(self.model).sample_actions(view, observation, num_steps=10)
        self.previous = frame
        normalized = sampled[0, :, : expected if self.suite == "robotwin" else 7].cpu().numpy()
        denormalized = self.normalize(
            normalized, "action" if self.suite == "libero" else "actions", inverse=True
        )
        return {
            "normalized_actions": normalized.tolist(),
            "actions": denormalized.tolist(),
            "finite": bool(np.isfinite(denormalized).all()),
            "visual_shape": list(visual.values.shape),
            "updates_per_camera": {c: s.updates for c, s in self.states.items()},
            "encoder_checkpoint": self.encoder_checkpoint,
            "baseline": self.baseline,
            "note": "Action chunk only; independent service smoke is not closed-loop success.",
        }


def make_server(policy, host="127.0.0.1", port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/reset":
                    policy.reset()
                    result = {"reset": True}
                elif self.path == "/infer":
                    result = policy.predict_action(**body)
                else:
                    raise ValueError("supported endpoints: POST /reset, POST /infer")
                encoded = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
            except Exception as exc:
                encoded = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return HTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("libero", "robotwin"))
    for name in ("weights", "stats", "tokenizer"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline", choices=("single_frame", "fixed_raw_patch"))
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=10095, type=int)
    parser.add_argument("--smoke-npz")
    parser.add_argument("--output", default="service-smoke.json")
    args = parser.parse_args(argv)
    policy = OnlineFeaturePolicy(
        suite=args.suite,
        weights=args.weights,
        stats=args.stats,
        tokenizer=args.tokenizer,
        device=args.device,
        baseline=args.baseline,
        encoder_checkpoint=args.encoder_checkpoint,
    )
    server = make_server(policy, args.host, 0 if args.smoke_npz else args.port)
    if not args.smoke_npz:
        print(f"N04 feature service http://{args.host}:{server.server_port}", flush=True)
        server.serve_forever()
        return 0
    import threading
    import time
    import urllib.request
    from urllib.error import HTTPError

    from owac.backends.rawvla import sha256_file
    from owac.raw.rawvla_sequence import RawVLASequenceLoader

    sequence = RawVLASequenceLoader(suite=args.suite).load(args.smoke_npz)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    responses = []
    try:
        for t in range(2):
            payload = {
                "cameras": {
                    c: sequence.measurements[c][t].permute(1, 2, 0).numpy().tolist()
                    for c in sequence.camera_ids
                },
                "proprio": sequence.proprio[t].tolist(),
                "task": sequence.metadata.get(
                    "task_description", sequence.metadata.get("task_name", "")
                ),
                "episode_id": sequence.episode_id,
                "frame_id": t,
                "raw_white_level": sequence.metadata.get(
                    "raw_white_level", 255 if args.suite == "libero" else None
                ),
            }
            started = time.perf_counter()
            request = urllib.request.Request(
                f"http://{args.host}:{server.server_port}/infer",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
            except HTTPError as exc:
                raise RuntimeError(exc.read().decode()) from exc
            result["http_roundtrip_ms"] = (time.perf_counter() - started) * 1000
            assert result["finite"] and set(result["updates_per_camera"].values()) == {t + 1}
            responses.append(result)
        request = urllib.request.Request(
            f"http://{args.host}:{server.server_port}/reset", data=b"{}"
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["reset"]
        report = {
            "suite": args.suite,
            "responses": responses,
            "reset_verified": policy.states is None,
            "pretrained_weights": True,
            "random_encoder": args.encoder_checkpoint is None,
            "hashes": {
                key: sha256_file(Path(path))
                for key, path in {
                    "weights": args.weights,
                    "stats": args.stats,
                    "tokenizer": args.tokenizer,
                    "npz": args.smoke_npz,
                }.items()
            },
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps({"output": args.output, "reset_verified": report["reset_verified"]}))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
