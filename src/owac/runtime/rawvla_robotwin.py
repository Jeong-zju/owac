"""RoboTwin policy plugin and server for the author's SidneyXie/pi05_robotwin.

Use the pinned RawVLA StarVLA implementation, original LeRobot safetensors and
normalizers. The generic Qwen RoboTwin client reorders joints and uses min/max;
this checkpoint instead requires left-arm/gripper/right-arm/gripper and z-score.
Image preprocessing remains the author StarVLA PI05 implementation.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def normalize_state(state, stats):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("π0.5 RoboTwin requires 14 finite joint/gripper values")
    return (state - np.asarray(stats["mean"], dtype=np.float32)) / (
        np.asarray(stats["std"], dtype=np.float32) + np.float32(1e-8)
    )


def unnormalize_actions(actions, stats):
    actions = np.asarray(actions, dtype=np.float32)[..., :14]
    result = actions * np.asarray(stats["std"], dtype=np.float32) + np.asarray(
        stats["mean"], dtype=np.float32
    )
    if result.shape != (50, 14) or not np.isfinite(result).all():
        raise ValueError(f"Invalid π0.5 action chunk: {result.shape}")
    return result


class Policy:
    def __init__(self, checkpoint: Path, tokenizer: Path, evidence: Path):
        import torch
        from safetensors.torch import load_file

        from owac.backends.rawvla import load_audited, new_model, sha256_file

        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.stats = json.loads((checkpoint / "policy_norm_stats.json").read_text())["norm_stats"]
        self.model = new_model("robotwin", tokenizer)
        audit = load_audited(self.model, load_file(str(checkpoint / "model.safetensors")))
        self.model = self.model.to("cuda").eval()
        self.evidence = evidence
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.input_index = len(list(self.evidence.glob("input_*.npz")))
        audit.update(
            checkpoint_sha256=sha256_file(checkpoint / "model.safetensors"),
            stats_sha256=sha256_file(checkpoint / "policy_norm_stats.json"),
            state_order="left6,left_gripper,right6,right_gripper",
            normalization="MEAN_STD",
            state_padding=0.0,
            image_preprocessing="pinned RawVLA StarVLA PI05",
            extra_training_steps=0,
        )
        (self.evidence / "load-audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    def predict_action(self, examples):
        import torch

        if len(examples) != 1 or len(examples[0]["image"]) != 3:
            raise ValueError("RoboTwin deployment uses batch=1 and head/left/right cameras")
        example = dict(examples[0])
        flow_seed = example.get("flow_seed")
        if flow_seed is not None:
            torch.manual_seed(int(flow_seed))
        index = self.input_index
        arrays = {
            "state": np.asarray(example["state"], dtype=np.float32),
            **{f"camera_{i}": np.asarray(image) for i, image in enumerate(example["image"])},
        }
        np.savez_compressed(self.evidence / f"input_{index:05d}.npz", **arrays)
        self.input_index += 1
        example["state"] = normalize_state(example["state"], self.stats["state"])[None]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model.predict_action([example])["normalized_actions"][0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        actions = unnormalize_actions(prediction, self.stats["actions"])
        record = {
            "case_id": example.get("case_id"),
            "flow_seed": flow_seed,
            "actions": actions.tolist(),
            "inference_seconds": seconds,
            "shape": list(actions.shape),
            "finite": bool(np.isfinite(actions).all()),
            "action_min": float(actions.min()),
            "action_max": float(actions.max()),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "input": f"input_{index:05d}.npz",
            "image_stats": [
                {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "min": float(image.min()),
                    "max": float(image.max()),
                    "mean": float(image.mean()),
                    "std": float(image.std()),
                }
                for image in arrays.values()
                if image.ndim == 3
            ],
        }
        with (self.evidence / "inference.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        from owac.runtime.n02_control import ControlProtocol

        protocol = os.environ.get("N02_EXECUTION_PROTOCOL")
        return {
            "actions": actions,
            "execution_contract": ControlProtocol.parse(protocol).as_dict() if protocol else None,
        }


class ModelClient:
    def __init__(self, args):
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(args.get("host", "127.0.0.1"), args.get("port", 18005))
        self.actions = None
        self.chunk_start = 0
        from owac.runtime.n02_control import ActionExecution, ControlProtocol

        setting = os.environ.get("N02_EXECUTION_PROTOCOL")
        self.execution = ActionExecution(ControlProtocol.parse(setting)) if setting else None

    def step(self, example, step):
        interval = self.execution.protocol.replan_interval if self.execution else 50
        if self.actions is None or step - self.chunk_start >= interval:
            if os.environ.get("RAWVLA_FLOW_SEED") is not None:
                example = dict(example)
                example["flow_seed"] = (int(os.environ["RAWVLA_FLOW_SEED"]) + step) % (2**32)
                example["case_id"] = os.environ["RAWVLA_CASE_ID"]
            response = self.client.predict_action({"examples": [example]})
            if not response.get("ok"):
                raise RuntimeError(f"π0.5 inference failed: {response}")
            if (
                self.execution
                and response["data"].get("execution_contract") != self.execution.protocol.as_dict()
            ):
                raise ValueError("native policy execution contract mismatch")
            self.actions = np.asarray(response["data"]["actions"], dtype=np.float32)
            self.chunk_start = step
        requested = self.actions[step - self.chunk_start].copy()
        return self.execution.step(requested, example["state"]) if self.execution else requested


def get_model(usr_args):
    denoiser = os.environ.get("RAWVLA_ROBOTWIN_DENOISER")
    if denoiser:
        import sapien

        if denoiser not in {"optix", "none", "oidn"}:
            raise ValueError(f"Unknown SAPIEN denoiser: {denoiser}")
        original = sapien.render.set_ray_tracing_denoiser

        def set_denoiser(name):
            original(denoiser if name == "oidn" else name)

        # SAPIEN 3.0b1's bundled OIDN cannot run on this Blackwell GPU. Keep
        # the pinned scene and ray tracer; explicitly choose a supported backend.
        sapien.render.set_ray_tracing_denoiser = set_denoiser
    return ModelClient(usr_args)


def reset_model(model):
    model.actions = None
    model.chunk_start = 0
    if getattr(model, "execution", None):
        model.execution.reset()


def eval(TASK_ENV, model, observation):
    expected = os.environ.get("RAWVLA_EXPECTED_ENTRY")
    if expected and TASK_ENV.take_action_cnt == 0:
        entry = json.loads(expected)
        actual = TASK_ENV.rawvla_lighting_metadata
        for field in (
            "lighting_ev",
            "lighting_scale",
            "sensor_saturation_ev",
            "raw_white_level",
            "raw_sensor_shot_noise",
            "raw_sensor_read_noise",
            "noise_seed",
        ):
            if not np.isclose(float(actual[field]), float(entry[field]), rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"Manifest lighting mismatch: {field}: {actual[field]} != {entry[field]}"
                )
        evidence = Path(os.environ["ROBOTWIN_POLICY_AUDIT_DIR"])
        (evidence / "lighting.json").write_text(json.dumps(actual, indent=2) + "\n")
    key = os.environ.get("ROBOTWIN_IMAGE_OBS_KEY", "rgb")
    cameras = ("head_camera", "left_camera", "right_camera")
    example = {
        "task_name": str(TASK_ENV.task_name),
        "lang": str(TASK_ENV.get_instruction()),
        "image": [observation["observation"][camera][key] for camera in cameras],
        "state": observation["joint_action"]["vector"],
    }
    action = model.step(example, TASK_ENV.take_action_cnt)
    audit = os.environ.get("RAWVLA_STEP_AUDIT")
    if audit:
        with Path(audit).open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "step": int(TASK_ENV.take_action_cnt),
                        "state": np.asarray(example["state"]).tolist(),
                        "action": action.tolist(),
                    }
                )
                + "\n"
            )
    if hasattr(TASK_ENV, "policy_predicted_actions"):
        TASK_ENV.policy_predicted_actions.append(action.copy())
    TASK_ENV.take_action(action)


def main():
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--port", default=18005, type=int)
    args = parser.parse_args()
    policy = Policy(args.checkpoint, args.tokenizer, args.evidence)
    WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=args.port,
        idle_timeout=-1,
        metadata={"model": "PI05", "action_chunk_size": 50, "state_dim": 14, "norm": "MEAN_STD"},
    ).serve_forever()


if __name__ == "__main__":
    main()
