"""RoboTwin closed-loop adapter with canonical policy inputs (no training).

The simulator collects the last eight control observations. Each decision replays
that bounded window through the training path, then samples one 50-action chunk.
RGB receives the saved arm's fixed ISP applied to the same float32 RAW source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")


def online_window(example, camera_ids, *, history=8, representation=None):
    import torch

    from owac.core.encoder_contract import EmbodimentContext, MeasurementFrame
    from owac.frameworks.n02_full.data_v12 import V12WindowBatch, apply_declared_isp
    from owac.frameworks.n02_full.state import ObservationBundle

    observations = list(example["history"])
    if not observations or len(observations) > history:
        raise ValueError("online history must contain 1..history actual observations")
    indices = [int(row["step"]) for row in observations]
    if any(b <= a for a, b in zip(indices, indices[1:], strict=False)):
        raise ValueError("online history contains duplicate or future/out-of-order observations")
    if indices[-1] != int(example["step"]):
        raise ValueError("online history does not end at the decision step")
    observations = [observations[0]] * (history - len(observations)) + observations
    indices = np.array([int(row["step"]) for row in observations], dtype=np.int64)
    fresh = np.concatenate(([True], np.diff(indices) != 0))
    episode = str(example["case_id"])
    frames = {}
    for slot, camera_id in enumerate(camera_ids):
        values = np.stack([np.asarray(row["image"][slot]) for row in observations])
        if values.dtype != np.float32 or values.shape[1:] != (240, 320, 3):
            raise ValueError(
                f"expected unscaled float32 RAW3 240x320, got {values.dtype} {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("nonfinite online RAW input")
        values = np.ascontiguousarray(values.transpose(0, 3, 1, 2))
        valid = np.isfinite(values).all(axis=1)
        frames[camera_id] = MeasurementFrame(
            values=torch.from_numpy(values),
            valid_mask=valid,
            bad_mask=~valid,
            # Match the frozen float32 loader, which declares no saturation mask.
            saturated_mask=np.zeros(valid.shape, dtype=bool),
            episode_ids=(episode,) * history,
            frame_ids=tuple(f"{episode}/{camera_id}/{step}" for step in indices),
            camera_id=camera_id,
            new_measurement=fresh,
            metadata={
                "source": "RoboTwin HdrColor/raw_linear",
                "variance": None,
                "frame_index_unit": "control_step",
                "native_bayer": False,
            },
            metadata_available={"timestamp": False, "noise_model": False},
        )
    proprio = np.asarray(example["normalized_state"], dtype=np.float32)
    if proprio.shape != (14,) or not np.isfinite(proprio).all():
        raise ValueError("expected one normalized 14-dimensional current joint vector")
    context = EmbodimentContext(
        # The training loader broadcasts anchor proprio across its window too.
        proprio=np.broadcast_to(proprio, (history, 14)).copy(),
        proprio_available=True,
        executed_action_available=False,
        camera_transform_available=False,
        task_tokens=str(example["lang"]),
        task_mask=True,
        hypothesis_prior_available=False,
    )
    bundle = ObservationBundle(
        frames=frames,
        context=context,
        episode_ids=(episode,) * history,
        anchor_ids=(f"{episode}/decision{indices[-1]}",) * history,
        new_time_mask=fresh,
    )
    bundle = apply_declared_isp(bundle, representation or {"kind": "raw3"})
    return V12WindowBatch(
        observation=bundle,
        frame_index=indices,
        history=history,
        batch_size=1,
        partition="online_smoke",
        measurement_ids=tuple(
            (episode, str(step), camera) for step in sorted(set(indices)) for camera in camera_ids
        ),
    )


class Policy:
    def __init__(self, config, checkpoint, evidence, *, legacy_weights_probe=False):
        import torch

        from owac.frameworks.n02_full.sequence_v12 import EXECUTION_REVISION
        from owac.runtime.rawvla_full_eval import digest, write_json
        from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan

        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        plan = resolve_plan(
            build_parser().parse_args(
                [
                    "--config",
                    str(config),
                    "--output",
                    str(self.evidence / "loader"),
                    "--run-id",
                    "closed-loop-readonly",
                    "--max-updates",
                    "1000",
                ]
            )
        )
        self.trainer = N02V12Trainer(plan, LogTail())
        self.trainer.build()
        if legacy_weights_probe:
            payload = self.trainer.load_weights_for_probe(Path(checkpoint))
            step = int(payload["step"])
        else:
            step, _, payload = self.trainer.load_for_resume(Path(checkpoint))
        self.trainer.model.eval()
        self.stats = json.loads(Path(plan.config["data"]["stats_path"]).read_text())["norm_stats"]
        self.cameras = tuple(plan.config["model"]["camera_ids"])
        self.representation = plan.config["representation"]
        self.history = plan.history
        from owac.runtime.n02_control import ControlProtocol

        self.execution = (
            ControlProtocol.parse(plan.config["execution"])
            if plan.config.get("execution")
            else None
        )
        if self.execution and os.environ.get("N02_EXECUTION_PROTOCOL"):
            if ControlProtocol.parse(os.environ["N02_EXECUTION_PROTOCOL"]) != self.execution:
                raise ValueError("server/client execution contract differs")
        self.seen = set()
        write_json(
            self.evidence / "load_audit.json",
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "step": step,
                "execution_revision": payload["execution_revision"],
                "runtime_execution_revision": EXECUTION_REVISION,
                "task_protocol": plan.config.get("task_adaptation", {}).get("protocol"),
                "execution_contract": self.execution.as_dict() if self.execution else None,
                "legacy_weights_probe": bool(legacy_weights_probe),
                "policy_input_contract": "pi05-native-language-state-v1",
                "training_contract_sha256": payload["training_contract"]["sha256"],
                "representation": self.representation,
                "history": self.history,
                "action_horizon": 50,
                "replan_interval": int(os.environ.get("N02_REPLAN_INTERVAL", "50")),
                "client_max_joint_delta_rad_per_command": os.environ.get("N02_MAX_JOINT_DELTA"),
                "denoise_steps": 10,
                "flow_noise_mode": os.environ.get("N02_FLOW_NOISE_MODE", "independent"),
                "optimizer_updates": 0,
                "restore": "strict model/data/backend weights only; no optimizer or RNG"
                if legacy_weights_probe
                else "strict training contract and state_dict",
                "observation": "live float32 raw_linear; 3 cameras; 8 most recent control frames",
                "proprio_context": "current normalized joint vector, broadcast as during training",
                "sensor_metadata": "same undeclared noise/float saturation fields as frozen loader",
                "torch_version": torch.__version__,
            },
        )
        del payload

    def predict_action(self, examples):
        import torch

        from owac.runtime.rawvla_robotwin import normalize_state, unnormalize_actions

        if len(examples) != 1:
            raise ValueError("closed-loop RPC requires a single decision")
        example = dict(examples[0])
        instructions = self.trainer.plan.config["policy"].get("task_instructions")
        if instructions:
            from owac.backends.rawvla_language import canonical_task_prompt

            example["lang"] = canonical_task_prompt(
                example.get("task_name"), example.get("lang"), instructions
            )
        example["normalized_state"] = normalize_state(example["state"], self.stats["state"])
        window = online_window(
            example, self.cameras, history=self.history, representation=self.representation
        )
        torch.manual_seed(int(example["flow_seed"]))
        start = time.perf_counter()
        tokens, masks, state = self.trainer.language(
            [str(example["lang"])], example["normalized_state"][None]
        )
        self.trainer.bridge.reset_episodes()
        torch.cuda.synchronize()
        with torch.inference_mode():
            prediction = self.trainer.bridge.predict_action(
                observation=window, lang_tokens=tokens, lang_masks=masks, state=state, num_steps=10
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        actions = unnormalize_actions(prediction["control_actions"][0], self.stats["actions"])
        case_id = str(example["case_id"])
        if case_id not in self.seen:
            self.seen.add(case_id)
            np.savez_compressed(
                self.evidence / f"{case_id}_first_input.npz",
                state=example["state"],
                **{f"raw_{i}": example["history"][-1]["image"][i] for i in range(3)},
            )
        record = {
            "case_id": case_id,
            "step": int(example["step"]),
            "language": example["lang"],
            "history_steps": window.frame_index.tolist(),
            "flow_seed": int(example["flow_seed"]),
            "inference_seconds": seconds,
            "actions": actions.tolist(),
            "shape": list(actions.shape),
            "finite": bool(np.isfinite(actions).all()),
            "run_stats": prediction["run_stats"],
            "execution_contract": self.execution.as_dict() if self.execution else None,
            "check": {
                "status": self.trainer.token_source.last_output.check.status,
                "valid": self.trainer.token_source.last_output.check.valid,
                "revision": self.trainer.token_source.last_output.diagnostics.get("revision"),
            },
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "q": self.trainer.token_source.last_output.q.detach().cpu().tolist(),
            "raw_input_sha256": [
                hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                for image in example["history"][-1]["image"]
            ],
        }
        with (self.evidence / "inference.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"decision {case_id} step={record['step']} {seconds:.2f}s finite={record['finite']}",
            flush=True,
        )
        return {
            "actions": actions,
            "execution_contract": self.execution.as_dict() if self.execution else None,
        }


class ModelClient:
    def __init__(self, client, *, replan_interval=50, max_joint_delta=None, protocol=None):
        from owac.runtime.n02_control import ActionExecution, ControlProtocol

        self.client = client
        self.execution = ActionExecution(ControlProtocol.parse(protocol)) if protocol else None
        if self.execution:
            replan_interval = self.execution.protocol.replan_interval
            max_joint_delta = self.execution.protocol.max_joint_delta
        self.replan_interval = int(replan_interval)
        if not 1 <= self.replan_interval <= 50:
            raise ValueError("replan interval must be between 1 and 50")
        self.max_joint_delta = max_joint_delta
        if max_joint_delta is not None and (
            not np.isfinite(max_joint_delta) or max_joint_delta <= 0
        ):
            raise ValueError("joint delta bound must be finite and positive")
        self.last_action = None
        self.history = deque(maxlen=8)
        self.actions = None
        self.chunk_start = 0

    def step(self, example, step):
        from owac.runtime.n02_control import decision_flow_seed

        # Cameras may reuse their acquisition buffer. Causal history must own
        # its pixels, or a later capture silently rewrites an earlier frame.
        self.history.append(
            {"step": int(step), "image": [np.array(image, copy=True) for image in example["image"]]}
        )
        if self.actions is None or step - self.chunk_start >= self.replan_interval:
            request = dict(
                example,
                history=list(self.history),
                step=int(step),
                case_id=os.environ["RAWVLA_CASE_ID"],
                flow_seed=decision_flow_seed(
                    int(os.environ["RAWVLA_FLOW_SEED"]),
                    step,
                    os.environ.get("N02_FLOW_NOISE_MODE", "independent"),
                ),
            )
            response = self.client.predict_action({"examples": [request]})
            if not response.get("ok"):
                raise RuntimeError(f"N02 policy inference failed: {response}")
            expected_contract = self.execution.protocol.as_dict() if self.execution else None
            if response["data"].get("execution_contract") != expected_contract:
                raise ValueError("policy response execution contract mismatch")
            actions = np.asarray(response["data"]["actions"], dtype=np.float32)
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise ValueError("N02 returned an invalid control chunk")
            self.actions, self.chunk_start = actions, int(step)
        requested = self.actions[step - self.chunk_start].copy()
        action = requested.copy()
        if self.execution:
            action = self.execution.step(requested, example["state"])
        elif self.max_joint_delta is not None:
            if self.last_action is None:
                self.last_action = np.asarray(example["state"], dtype=np.float32).copy()
            if self.last_action.shape != (14,) or not np.isfinite(self.last_action).all():
                raise ValueError("bounded execution requires a finite 14-dimensional state")
            limits = np.full(14, self.max_joint_delta, dtype=np.float32)
            limits[[6, 13]] = 0.05
            action = self.last_action + np.clip(action - self.last_action, -limits, limits)
            action[[6, 13]] = action[[6, 13]].clip(0, 1)
        path = os.environ.get("N02_CONTROL_AUDIT")
        if path:
            with Path(path).open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": int(step),
                            "requested": requested.tolist(),
                            "executed": action.tolist(),
                            "max_joint_delta_rad_per_command": self.max_joint_delta,
                            "replan_interval": self.replan_interval,
                            "flow_noise_mode": os.environ.get("N02_FLOW_NOISE_MODE", "independent"),
                            "execution_contract": self.execution.protocol.as_dict()
                            if self.execution
                            else None,
                        }
                    )
                    + "\n"
                )
        self.last_action = action.copy()
        return action


def get_model(arguments):
    from owac.runtime.rawvla_robotwin import get_model as base_client

    limit = os.environ.get("N02_MAX_JOINT_DELTA")
    return ModelClient(
        base_client(arguments).client,
        replan_interval=int(os.environ.get("N02_REPLAN_INTERVAL", "50")),
        max_joint_delta=float(limit) if limit is not None else None,
        protocol=os.environ.get("N02_EXECUTION_PROTOCOL"),
    )


def reset_model(model):
    model.history.clear()
    model.actions = None
    model.chunk_start = 0
    model.last_action = None
    if model.execution:
        model.execution.reset()


def eval(task, model, observation):
    # Reuse original lighting assertions, action trace, state order and take_action.
    # Policy observations switch only to float32 RAW3; no display image enters the model.
    from owac.runtime.rawvla_robotwin import eval as base_eval

    base_eval(task, model, observation)


def main():
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--legacy-weights-probe",
        action="store_true",
        help="labelled r5 weights-only probe; no optimizer/RNG restore",
    )
    args = parser.parse_args()
    policy = Policy(
        args.config, args.checkpoint, args.evidence, legacy_weights_probe=args.legacy_weights_probe
    )
    WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=args.port,
        idle_timeout=-1,
        metadata={
            "model": "N02-RAW-policy",
            "scale_balance_mode": policy.trainer.plan.config["model"].get(
                "scale_balance_mode", "contrast"
            ),
            "legacy_weights_probe": args.legacy_weights_probe,
            "action_chunk_size": 50,
            "state_dim": 14,
            "norm": "MEAN_STD",
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
