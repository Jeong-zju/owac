"""RoboTwin policy plugin and server for the author's SidneyXie/pi05_robotwin.

Use the pinned RawVLA StarVLA implementation, original LeRobot safetensors and
normalizers.

可选协议扩展（N04 v1.1 契约，2026-10-15）：``Policy.predict_action`` 接受可选 payload
``proprio_history``（``[3,14]``、旧→新、含当前帧、原始物理量），校验后不转发给 π0.5 模型；
不带该字段时行为与扩展前逐位相同。客户端侧同一开关：设 ``RAWVLA_PROPRIO_HISTORY=1`` 后
``ModelClient`` 会按同一契约把最近 3 帧真实观测本体放进 payload（每 episode 重置、不足 3 帧
不发）；**默认关闭**，关闭时一个字节都不多发。

The generic Qwen RoboTwin client reorders joints and uses min/max;
this checkpoint instead requires left-arm/gripper/right-arm/gripper and z-score.
Image preprocessing remains the author StarVLA PI05 implementation.
"""

import argparse
import collections
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

#: 客户端精确本体历史开关：**默认关闭**，只有显式设为 1/true/yes/on 时 `ModelClient` 才会在
#: payload 里多发 ``proprio_history``（服务端 metadata 的 ``n04.proprio_history_source`` 随之
#: 变成 ``observation_frames``）；关闭时一个字节都不多发。
PROPRIO_HISTORY_ENV = "RAWVLA_PROPRIO_HISTORY"

#: N04 v1.1 契约：payload 里的精确本体历史 = ``proprio_history+1`` 帧，旧→新、含当前帧、原始物理量。
ROBOTWIN_PROPRIO_HISTORY_FRAMES = 3
ROBOTWIN_STATE_DIM = 14


def validate_proprio_history(proprio_history):
    """校验可选的 payload ``proprio_history``；``None`` 原样返回（官方 π0.5 路径逐位不变）。

    契约与 N04 v1.1 策略服务一致：``[3, 14]``（允许前置批维 ``[1,3,14]``）、旧→新、含当前帧、
    原始物理量；长度/维度/有限性不符直接报错，不截断也不补零。**官方 π0.5 模型不消费这个字段**
    （它没有状态估计器），这里只做协议兼容与显式校验，真正用它的是 N04 v1.1 服务端。

    注意（避免空对照）：编码器在 episode 首帧走 ``state_stack.reset()``（spawn），本体条件只
    在**非首帧**的状态预测里起作用——所以"比较每个 episode 的首个动作块"区分不出带不带历史。
    """
    if proprio_history is None:
        return None
    array = np.asarray(proprio_history, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    expected = (ROBOTWIN_PROPRIO_HISTORY_FRAMES, ROBOTWIN_STATE_DIM)
    if array.ndim != 2 or array.shape != expected:
        raise ValueError(
            f"proprio_history 需要 {expected} = [proprio_history+1, 平台本体维]"
            f"（旧→新、含当前帧），实际 {array.shape}；不截断也不补零"
        )
    if not np.isfinite(array).all():
        raise ValueError("proprio_history 含 NaN/Inf")
    return array


def proprio_history_enabled():
    """开关默认关闭（未设置 / 空串 / 0 / false 都算关闭）。"""
    return os.environ.get(PROPRIO_HISTORY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


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

    def predict_action(self, examples, proprio_history=None):
        import torch

        if len(examples) != 1 or len(examples[0]["image"]) != 3:
            raise ValueError("RoboTwin deployment uses batch=1 and head/left/right cameras")
        # 可选：客户端按 N04 v1.1 契约带上的真实连续本体历史。只校验、不转发给 π0.5 模型，
        # 所以"不带历史"的路径与改动前逐位相同（官方 π0.5 闭环依赖这一点）。
        validate_proprio_history(proprio_history)
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
        return {"actions": actions}


class ModelClient:
    def __init__(self, args):
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(args.get("host", "127.0.0.1"), args.get("port", 18005))
        self.actions = None
        self.chunk_start = 0
        from owac.runtime.n02_control import ActionExecution, ControlProtocol

        setting = os.environ.get("N02_EXECUTION_PROTOCOL")
        self.execution = ActionExecution(ControlProtocol.parse(setting)) if setting else None
        #: 最近 ``ROBOTWIN_PROPRIO_HISTORY_FRAMES`` 帧真实观测本体（旧→新，含当前帧）。
        self.proprio_history = collections.deque(maxlen=ROBOTWIN_PROPRIO_HISTORY_FRAMES)

    def _history(self):
        """本体历史 deque（旧→新）；懒建，替身对象（测试里 ``__new__`` 出来的）也能用。"""
        history = getattr(self, "proprio_history", None)
        if history is None:
            history = collections.deque(maxlen=ROBOTWIN_PROPRIO_HISTORY_FRAMES)
            self.proprio_history = history
        return history

    def _proprio_history_payload(self):
        """开关打开且已攒够 3 帧真实观测时给出 ``(3,14)`` 历史；否则 ``None``（不发该字段）。

        不足 3 帧时**不发**，而不是重复最早帧：训练侧的补齐口径是"重复最早真实帧"，与本客户端
        "只发真实观测"的语义不同，宁可让服务端退回 ``policy_calls_approx``（metadata 里可区分）。
        坏值（维度不对 / 非有限）也不会被上送。
        """
        history = self._history()
        if not proprio_history_enabled():
            return None
        if len(history) < ROBOTWIN_PROPRIO_HISTORY_FRAMES:
            return None
        try:
            return validate_proprio_history(np.stack(history))
        except ValueError as error:  # 坏输入不上送，只在本机日志里报出来
            LOGGER.warning("跳过 proprio_history（未上送）：%s", error)
            return None

    def _native_replan_due(self, step):
        """Keep the frozen native cadence explicit for source-based protocol audits."""
        if self.actions is None or step - self.chunk_start >= 50:
            return True
        return False

    def step(self, example, step):
        history = self._history()
        if self.actions is None:  # episode 首次推理（作者 harness 每 episode 调 reset_model）
            history.clear()
        if proprio_history_enabled():
            state = np.asarray(example["state"], dtype=np.float32).reshape(-1)
            if state.shape == (ROBOTWIN_STATE_DIM,) and np.isfinite(state).all():
                history.append(state)
        execution = getattr(self, "execution", None)
        should_replan = (
            self.actions is None or step - self.chunk_start >= execution.protocol.replan_interval
            if execution
            else self._native_replan_due(step)
        )
        if should_replan:
            if os.environ.get("RAWVLA_FLOW_SEED") is not None:
                example = dict(example)
                example["flow_seed"] = (int(os.environ["RAWVLA_FLOW_SEED"]) + step) % (2**32)
                example["case_id"] = os.environ["RAWVLA_CASE_ID"]
            payload = {"examples": [example]}
            history_payload = self._proprio_history_payload()
            if history_payload is not None:
                payload["proprio_history"] = history_payload
            response = self.client.predict_action(payload)
            if not response.get("ok"):
                raise RuntimeError(f"π0.5 inference failed: {response}")
            if (
                execution
                and response["data"].get("execution_contract") != execution.protocol.as_dict()
            ):
                raise ValueError("native policy execution contract mismatch")
            self.actions = np.asarray(response["data"]["actions"], dtype=np.float32)
            self.chunk_start = step
        requested = self.actions[step - self.chunk_start].copy()
        return execution.step(requested, example["state"]) if execution else requested


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
    history = getattr(model, "proprio_history", None)
    if history is not None:
        history.clear()  # episode 边界：本体历史绝不跨 episode 拼接


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
