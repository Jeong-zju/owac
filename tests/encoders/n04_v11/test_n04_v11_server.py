"""N04 v1.1 策略服务：协议、形状、因果（episode 状态）与错误路径。

用一个极小替身桥（``_FakeBridge``）替换编码器／π0.5 后端，这样测的是**服务自己的**逻辑：
官方 payload 解析、图像缩放、本体历史、episode 边界、只融合一次的计数器、metadata 与实际
错误回报。真实编码器／后端由 WS-C 与训练侧测试覆盖；真实联调见任务报告。
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.backends import n04_v11_server as server
from owac.raw.n04_v11_data import PLATFORM_SPECS, ActionNormalizer

STATE_Q01 = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
STATE_Q99 = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
ACTION_Q01 = [-1.0] * 7
ACTION_Q99 = [1.0] * 7


class _FakeTokenizer:
    def tokenize(self, text, state=None):
        length = max(2, min(8, len(str(text).split()) + 1))
        return np.zeros(length, dtype=np.int32), np.ones(length, dtype=bool)


class _FakeBackend:
    """只为 tokenize_backend_task 提供的最小属性面。"""

    max_state_dim = 32
    state_pad_value = 0.0
    discrete_state_input = False

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


class _FakeEncoder:
    def __init__(self, proprio_dim: int = 24) -> None:
        self.config = SimpleNamespace(proprio_dim=proprio_dim)
        self.resets: list[object] = []

    def reset(self, episode=None, mask=None) -> None:
        self.resets.append(episode)


class _FakeBridge:
    """动作 = 本体向量前 control_dim 维（可复现，能检查"状态推进→动作变化"）。"""

    def __init__(
        self,
        encoder: _FakeEncoder,
        *,
        horizon: int = 10,
        action_dim: int = 32,
        control_dim: int = 7,
        nan: bool = False,
        fusion: int = 1,
    ) -> None:
        self.encoder = encoder
        self.config = SimpleNamespace(
            action_dim=action_dim,
            action_horizon=horizon,
            control_dim=control_dim,
            tokens_per_camera=256,
        )
        self.counter = SimpleNamespace(calls=0)
        self.nan = nan
        self.fusion = fusion
        self.calls: list[dict] = []
        self.saturation_shapes: list[list[tuple[int, ...]]] = []

    def sample_actions(
        self,
        frame,
        context,
        state,
        task_tokens,
        task_mask,
        *,
        backend_state=None,
        num_denoise_steps=None,
        noise=None,
    ):
        self.counter.calls += self.fusion
        proprio = context.proprio.detach().clone()
        self.saturation_shapes.append([tuple(mask.shape) for mask in (frame.saturated_mask or [])])
        self.calls.append(
            {
                "proprio": proprio,
                "state": state,
                "backend_state": None if backend_state is None else backend_state.clone(),
                "frame_ids": tuple(frame.frame_ids),
                "episode_ids": tuple(frame.episode_ids),
                "camera_id": list(frame.camera_id),
                "sizes": [tuple(value.shape) for value in frame.values],
                "task_tokens": int(task_tokens.shape[-1]),
                "noise": None if noise is None else tuple(noise.shape),
                "metadata": dict(context.metadata),
                "saturation_shapes": [tuple(mask.shape) for mask in (frame.saturated_mask or [])],
                "denoise": num_denoise_steps,
            }
        )
        actions = torch.zeros(1, self.config.action_horizon, self.config.action_dim)
        head = proprio[: self.config.control_dim].reshape(1, -1)
        actions[..., : self.config.control_dim] = head.expand(
            1, self.config.action_horizon, self.config.control_dim
        )
        if self.nan:
            actions[0, 0, 0] = float("nan")
        step = 1 if state is None else int(state.step) + 1
        return {
            "actions": actions,
            "state": SimpleNamespace(step=step, proprio=proprio),
            "encoder_calls": 1,
        }


def _normalizer(platform: str = "libero") -> ActionNormalizer:
    if platform == "libero":
        stats = {
            "state": {"q01": STATE_Q01, "q99": STATE_Q99},
            "action": {"q01": ACTION_Q01, "q99": ACTION_Q99},
        }
    else:
        stats = {
            "state": {"mean": [0.0] * 14, "std": [1.0] * 14},
            "actions": {"mean": [0.0] * 14, "std": [1.0] * 14},
        }
    return ActionNormalizer(platform, stats)


def _policy(
    *, platform: str = "libero", bridge: _FakeBridge | None = None, proprio_history: int = 2
) -> tuple[server.N04V11Policy, _FakeBridge]:
    spec = PLATFORM_SPECS[platform]
    encoder = _FakeEncoder(proprio_dim=(proprio_history + 1) * spec.proprio_dim)
    bridge = bridge or _FakeBridge(
        encoder, control_dim=spec.action_dim, horizon=spec.action_horizon
    )
    policy = server.N04V11Policy(
        bridge,
        _FakeBackend(),
        platform_spec=spec,
        normalizer=_normalizer(platform),
        device="cpu",
        proprio_history=proprio_history,
        num_denoise_steps=10,
        amp_dtype="fp32",
    )
    return policy, bridge


def _images(seed: int, size: int = 64) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(2)]


def _payload(
    *,
    state: np.ndarray | None = None,
    lang: str = "pick up the black bowl",
    episode_id: str | None = "episode-1",
    seed: int = 0,
    images: list[np.ndarray] | None = None,
) -> dict:
    example: dict = {
        "image": images if images is not None else _images(seed),
        "lang": lang,
        "state": np.zeros((1, 8), dtype=np.float32) if state is None else state.reshape(1, -1),
    }
    if episode_id is not None:
        example["episode_id"] = episode_id
    return {"examples": [example]}


# --------------------------------------------------------------------------------------
# metadata：如实声明 N04 v1.1（与官方 π0.5 可区分）
# --------------------------------------------------------------------------------------


def test_metadata_declares_n04_v11(tmp_path: Path) -> None:
    spec = PLATFORM_SPECS["libero"]
    checkpoint = tmp_path / "checkpoint-step0005000.pt"
    checkpoint.write_bytes(b"stub")
    metadata = server.build_metadata(
        spec=server.ServingSpec(checkpoint=checkpoint, platform="libero"),
        config={
            "stage": "S2",
            "arm": "D",
            "seed": 101,
            "model": {"input_mode": "rawvla_raw3", "tokens_per_camera": 256, "token_dim": 2048},
            "data": {"proprio_history": 2},
            "backend": {"action_dim": 32, "num_denoise_steps": 10},
        },
        encoder=_FakeEncoder(),
        platform_spec=spec,
        checkpoint=checkpoint,
        pi05_checkpoint=Path("/ws/checkpoints/starvla_pi05_libero/bfloat16/model.safetensors"),
        payload={"stage": "S2", "arm": "D", "seed": 101, "step": 5000},
        stats_path=Path("/ws/checkpoints/starvla_pi05_libero/dataset_statistics.json"),
    )
    assert metadata["env"] == "n04_v11_server"
    assert metadata["model"] == "N04-V1.1-PI05"
    # 官方客户端用 checkpoint 推导 norm stats：必须指向真正加载的冻结 π0.5 文件所在目录。
    assert metadata["checkpoint"].endswith("starvla_pi05_libero/bfloat16/model.safetensors")
    assert metadata["action_horizon"] == 10 and metadata["action_dim"] == 32
    assert metadata["norm"] == "quantile"
    assert metadata["training_obs_image_size"] == [256, 256]
    assert metadata["n04"]["input_domain"] == "raw_counts_0_255"
    assert metadata["n04"]["stage"] == "S2"
    assert metadata["n04"]["arm"] == "D"
    assert metadata["n04"]["seed"] == 101
    assert metadata["n04"]["step"] == 5000
    assert metadata["n04"]["platform"] == "libero"
    assert metadata["n04"]["checkpoint"] == str(checkpoint)
    assert metadata["n04"]["num_denoise_steps"] == 10
    assert metadata["n04_checkpoint"] == str(checkpoint)
    assert metadata["n04_step"] == 5000
    # 元数据必须可 JSON 序列化（写日志用）。
    json.dumps(metadata, default=str)


def test_metadata_robotwin_omits_unverified_image_size(tmp_path: Path) -> None:
    assert server.training_image_hw("libero") == (256, 256)
    assert server.training_image_hw("robotwin") is None


# --------------------------------------------------------------------------------------
# 归一化逆变换
# --------------------------------------------------------------------------------------


def test_denormalize_state_inverts_normalizer() -> None:
    normalizer = _normalizer()
    raw = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]])
    normalized = normalizer.normalize_state(raw)
    restored = server.denormalize_state(normalizer, normalized.reshape(-1))
    assert torch.allclose(restored.reshape(1, -1), raw, atol=1e-4), (restored, raw)


# --------------------------------------------------------------------------------------
# 一帧 → 动作块
# --------------------------------------------------------------------------------------


def test_infer_returns_action_block_not_zeros() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {"env": server.PROTOCOL_ENV})
    normalized = policy.normalizer.normalize_state(torch.tensor(STATE_Q01, dtype=torch.float32))
    response = service.handle_message(
        {"type": "infer", "payload": _payload(state=normalized.numpy())}
    )
    assert response["ok"] is True, response
    actions = response["data"]["normalized_actions"]
    assert actions.shape == (1, 10, 32), actions.shape
    assert np.isfinite(actions).all()
    assert float(np.abs(actions[..., :7]).max()) > 0.0  # 不是零动作
    assert float(np.abs(actions[..., :7]).max()) < 50.0  # 归一化域内
    assert bridge.counter.calls == 1  # 一次观测只融合一次
    call = bridge.calls[0]
    assert call["camera_id"] == ["agentview", "wrist"]
    assert call["sizes"] == [(3, 64, 64)] * 2
    assert call["episode_ids"] == ("episode-1",)
    assert call["frame_ids"] == (1,)
    assert call["denoise"] == 10
    assert call["task_tokens"] >= 2


def test_uint8_frames_stay_in_raw_counts() -> None:
    """uint8 只做 dtype 归一，数值域保持 0..255（平台缩放只由编码器 InputScale 施加一处）。"""
    policy, _ = _policy()
    tensor = policy._image_tensor(np.full((8, 8, 3), 255, dtype=np.uint8), "agentview")
    assert torch.allclose(tensor, torch.full((3, 8, 8), 255.0))
    # 与训练侧 ``_saturation`` 同款：saturation_mask 吃单帧 [C,H,W]，按 channel 归约成 [1,H,W]，
    # 服务端把同一输出直接放进 frame.saturated_mask（不再自己 unsqueeze）。
    mask = policy._saturation(tensor)
    assert mask.shape == (1, 8, 8) and bool(mask.all())
    # 饱和阈值在原始计数域：255 才算饱和（10 不算）
    assert not bool(policy._saturation(torch.full((3, 8, 8), 10.0)).any())
    partial = policy._saturation(
        torch.tensor(
            [
                [[0.0, 255.0], [0.0, 0.0]],
                [[0.0, 0.0], [255.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
    )
    assert partial.tolist() == [[[False, True], [True, False]]]


def test_saturation_reaches_frame_as_single_channel_plane() -> None:
    """新契约 [1,H,W] 经 ``_merge_saturation`` 的 ``any(dim=0)`` 仍是每像素一平面。"""
    policy, bridge = _policy()
    policy.predict(_payload())
    frame = bridge.calls[-1]
    assert frame["camera_id"] == ["agentview", "wrist"]
    assert bridge.saturation_shapes[-1] == [(1, 64, 64), (1, 64, 64)]


def test_flat_infer_payload_accepts_raw_state() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    images = _images(3)
    response = service.handle_message(
        {
            "type": "infer",
            "payload": {
                "image": images,
                "prompt": "open the drawer",
                "state": np.asarray(STATE_Q99, dtype=np.float32),
            },
        }
    )
    assert response["ok"] is True, response
    # 扁平路径的 state 是物理量：服务端要归一化后交给 π0.5 后端。
    expected = policy.normalizer.normalize_state(torch.tensor(STATE_Q99, dtype=torch.float32))
    assert torch.allclose(bridge.calls[-1]["backend_state"][0], expected, atol=1e-5)


def test_noise_passes_through_with_expected_shape() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    payload = _payload()
    payload["noise"] = np.zeros((10, 32), dtype=np.float32)
    response = service.handle_message({"type": "infer", "payload": payload})
    assert response["ok"] is True, response
    assert bridge.calls[-1]["noise"] == (1, 10, 32)


# --------------------------------------------------------------------------------------
# 因果：episode 内状态推进 / 跨 episode reset
# --------------------------------------------------------------------------------------


def test_state_advances_within_episode() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    states = [
        np.asarray([0.2, -0.4, 0.6, -0.8, 0.1, -0.3, 0.5, -0.7], dtype=np.float32),
        np.asarray([-0.1, 0.3, -0.5, 0.7, -0.2, 0.4, -0.6, 0.8], dtype=np.float32),
        np.asarray([0.9, -0.9, 0.4, -0.4, 0.2, -0.2, 0.0, 0.0], dtype=np.float32),
    ]
    normalized = [policy.normalizer.normalize_state(torch.tensor(s)).numpy() for s in states]
    for index, state in enumerate(normalized):
        response = service.handle_message(
            {
                "type": "infer",
                "payload": _payload(state=state, seed=index, images=_images(10 + index)),
            }
        )
        assert response["ok"] is True, response
    assert [call["state"].step if call["state"] else None for call in bridge.calls] == [None, 1, 2]
    # 本体历史按 新→旧 拼接，窗口开头用最早的真实帧补齐（与训练 proprio_history_tensor 一致）。
    raw = [server.denormalize_state(policy.normalizer, torch.tensor(state)) for state in normalized]
    expected_history = [
        torch.cat([raw[0], raw[0], raw[0]]),
        torch.cat([raw[1], raw[0], raw[0]]),
        torch.cat([raw[2], raw[1], raw[0]]),
    ]
    for call, expected in zip(bridge.calls, expected_history, strict=True):
        assert torch.allclose(call["proprio"], expected, atol=1e-5), (call["proprio"], expected)
    assert [call["metadata"]["padded_frames"] for call in bridge.calls] == [2, 1, 0]
    assert [call["metadata"]["ordering"] for call in bridge.calls] == ["newest_first"] * 3


def test_reset_message_clears_carried_state() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    policy.predict(_payload())
    policy.predict(_payload(seed=1, images=_images(1)))
    assert bridge.calls[-1]["metadata"]["padded_frames"] == 1
    response = service.handle_message({"type": "reset", "payload": {"episode_id": "episode-1"}})
    assert response["ok"] is True and response["type"] == "reset"
    assert policy.stats["resets"] == 1
    assert bridge.encoder.resets == ["episode-1"]
    policy.predict(_payload(seed=2, images=_images(2)))
    assert bridge.calls[-1]["metadata"]["padded_frames"] == 2
    assert bridge.calls[-1]["frame_ids"] == (1,)
    assert bridge.calls[-1]["state"] is None


def test_episode_id_change_resets_and_task_text_change_resets() -> None:
    policy, bridge = _policy()
    policy.predict(_payload(episode_id="ep-1"))
    policy.predict(_payload(episode_id="ep-1", seed=1, images=_images(1)))
    assert bridge.calls[-1]["metadata"]["padded_frames"] == 1
    policy.predict(_payload(episode_id="ep-2", seed=2, images=_images(2)))
    assert bridge.calls[-1]["metadata"]["padded_frames"] == 2
    assert policy.stats["resets"] == 1
    policy.predict(_payload(episode_id="ep-2", lang="different task", seed=3, images=_images(3)))
    assert bridge.calls[-1]["metadata"]["padded_frames"] == 2
    assert policy.stats["resets"] == 2


def test_duplicate_payload_is_counted_not_silently_refused() -> None:
    policy, bridge = _policy()
    payload = _payload()
    policy.predict(payload)
    policy.predict(payload)
    assert policy.stats["duplicate_requests"] == 1
    assert bridge.counter.calls == 2
    assert bridge.calls[-1]["frame_ids"] == (2,)  # 仍然是新的一帧测量，不是复用缓存


# --------------------------------------------------------------------------------------
# 错误路径：明确报错，不返回零动作
# --------------------------------------------------------------------------------------


def _error_message(response: dict) -> str:
    assert response["ok"] is False, response
    assert "data" not in response
    return response["error"]["message"]


def test_wrong_image_shape_errors() -> None:
    policy, _ = _policy()
    service = server.N04V11Service(policy, {})
    images = [np.zeros((64, 64), dtype=np.uint8), np.zeros((64, 64), dtype=np.uint8)]
    message = _error_message(
        service.handle_message({"type": "infer", "payload": _payload(images=images)})
    )
    assert "[H,W,3]" in message


def test_wrong_camera_count_errors() -> None:
    policy, _ = _policy()
    service = server.N04V11Service(policy, {})
    images = [np.zeros((64, 64, 3), dtype=np.uint8)]
    message = _error_message(
        service.handle_message({"type": "infer", "payload": _payload(images=images)})
    )
    assert "相机数不符" in message


def test_wrong_state_width_errors() -> None:
    policy, _ = _policy()
    service = server.N04V11Service(policy, {})
    payload = _payload()
    payload["examples"][0]["state"] = np.zeros((1, 5), dtype=np.float32)
    assert "本体状态需要 8 维" in _error_message(
        service.handle_message({"type": "infer", "payload": payload})
    )


def test_missing_language_errors() -> None:
    policy, _ = _policy()
    service = server.N04V11Service(policy, {})
    payload = _payload(lang="   ")
    assert "语言指令" in _error_message(
        service.handle_message({"type": "infer", "payload": payload})
    )


def test_nan_actions_error_instead_of_zeros() -> None:
    spec = PLATFORM_SPECS["libero"]
    encoder = _FakeEncoder(proprio_dim=24)
    bridge = _FakeBridge(encoder, nan=True)
    policy, _ = _policy(bridge=bridge)
    service = server.N04V11Service(policy, {})
    message = _error_message(service.handle_message({"type": "infer", "payload": _payload()}))
    assert "NaN/Inf" in message
    assert spec.name == "libero"


def test_normalized_float_frames_are_rejected_without_fusing() -> None:
    """[0,1] 归一化浮点：明确报错，且一帧都不融合（计数不变）——不静默乘回 255。"""
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    before_calls = bridge.counter.calls
    before_requests = policy.stats["requests"]

    left = np.full((8, 8, 3), 255, dtype=np.uint8)
    normalized = np.full((8, 8, 3), 0.5, dtype=np.float32)
    response = service.handle_message(
        {"type": "infer", "payload": _payload(images=[left, normalized])}
    )
    message = _error_message(response)
    assert "归一化浮点" in message and "uint8" in message and "255" in message
    assert bridge.counter.calls == before_calls  # 坏输入不融合
    assert policy.stats["requests"] == before_requests

    # 全部为 [0,1] 浮点同样报错
    both = [np.zeros((8, 8, 3), dtype=np.float32), np.ones((8, 8, 3), dtype=np.float32)]
    assert "归一化浮点" in _error_message(
        service.handle_message({"type": "infer", "payload": _payload(images=both)})
    )
    assert bridge.counter.calls == before_calls


def test_raw_domain_float_frames_pass() -> None:
    """原始域浮点照收：LIBERO 计数域（>1）通过；RoboTwin 线性原值（含 ≤1 的暗帧、>1）通过。"""
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    raw_counts = [
        np.full((8, 8, 3), 200.0, dtype=np.float32),
        np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3),
    ]
    response = service.handle_message({"type": "infer", "payload": _payload(images=raw_counts)})
    assert response["ok"] is True, response
    assert bridge.counter.calls == 1
    # 原始域数值原样进编码器（不在服务端缩放）
    assert float(bridge.calls[-1]["sizes"][0][0]) == 3.0

    robotwin, robotwin_bridge = _policy(platform="robotwin")
    robotwin_service = server.N04V11Service(robotwin, {})
    for value in (0.0, 0.75, 3.5, 40.0):  # identity 域：暗帧与 >1 都不能误判
        payload = {
            "examples": [
                {
                    "image": [np.full((16, 16, 3), value, dtype=np.float32)] * 3,
                    "lang": "place the block",
                    "state": np.zeros(14, dtype=np.float32),
                }
            ]
        }
        reply = robotwin_service.handle_message({"type": "infer", "payload": payload})
        assert reply["ok"] is True, (value, reply)
    assert robotwin_bridge.counter.calls == 4


def test_fusion_count_not_one_errors() -> None:
    encoder = _FakeEncoder(proprio_dim=24)
    bridge = _FakeBridge(encoder, fusion=2)
    policy, _ = _policy(bridge=bridge)
    service = server.N04V11Service(policy, {})
    message = _error_message(service.handle_message({"type": "infer", "payload": _payload()}))
    assert "融合了 2 次" in message


def test_unknown_message_type_errors() -> None:
    policy, _ = _policy()
    service = server.N04V11Service(policy, {})
    response = service.handle_message({"type": "teleport"})
    assert response["ok"] is False and response["type"] == "unknown"


def test_ping_reports_counters() -> None:
    policy, _ = _policy()
    service = server.N04V11Service(policy, {})
    policy.predict(_payload())
    response = service.handle_message({"type": "ping"})
    assert response["ok"] is True
    assert response["stats"]["requests"] == 1
    assert response["stats"]["encoder_calls"] == 1


def test_batch_examples_rejected() -> None:
    policy, _ = _policy()
    service = server.N04V11Service(policy, {})
    payload = _payload()
    payload["examples"].append(payload["examples"][0])
    assert "只能 1 条" in _error_message(
        service.handle_message({"type": "infer", "payload": payload})
    )


# --------------------------------------------------------------------------------------
# 真实 websocket 往返（需要 websockets + msgpack；远端 n04v11 环境具备）
# --------------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"服务未在 {timeout}s 内监听 127.0.0.1:{port}")


def test_robotwin_protocol_b_returns_environment_actions() -> None:
    """协议 B（owac.runtime.rawvla_robotwin）：原始本体入、关节量纲 [50,14] 出。"""
    policy, bridge = _policy(platform="robotwin")
    service = server.N04V11Service(policy, {})
    raw_state = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    images = [np.zeros((64, 64, 3), dtype=np.float32) + 0.25 for _ in range(3)]
    response = service.handle_message(
        {
            "type": "infer",
            "payload": {
                "examples": [
                    {
                        "image": images,
                        "lang": "place the block",
                        "state": raw_state,
                        "flow_seed": 1234,
                    }
                ]
            },
        }
    )
    assert response["ok"] is True, response
    actions = response["data"]["actions"]
    assert actions.shape == (50, 14), actions.shape
    assert np.isfinite(actions).all()
    # 协议 B：本体会原样归一化给后端（而不是把原始关节量当已归一化值）。
    expected = policy.normalizer.normalize_state(torch.tensor(raw_state))
    assert torch.allclose(bridge.calls[-1]["backend_state"][0], expected, atol=1e-5)
    assert bridge.calls[-1]["camera_id"] == ["head", "left", "right"]


def test_environment_actions_stats_live_on_action_device() -> None:
    """协议 B 的反归一化统计必须跟动作同设备（GPU 上真踩过 cuda:0 vs cpu 的 mismatch）。

    用 meta 设备当替身：统计建错设备时这里会直接 RuntimeError，不需要 GPU。
    """
    policy, _ = _policy(platform="robotwin")
    value = torch.zeros(1, 50, 32, device="meta")
    result = policy._environment_actions_value(value)
    assert result.device.type == "meta"
    assert tuple(result.shape) == (50, 14)
    libero_policy, _ = _policy(platform="libero")
    libero_value = torch.zeros(1, 10, 32, device="meta")
    assert tuple(libero_policy._environment_actions_value(libero_value).shape) == (10, 7)


def test_robotwin_metadata_declares_chunk_and_state_dim() -> None:
    spec = PLATFORM_SPECS["robotwin"]
    metadata = server.build_metadata(
        spec=server.ServingSpec(checkpoint=Path("/tmp/ckpt.pt"), platform="robotwin"),
        config={
            "stage": "S2",
            "arm": "D",
            "seed": 101,
            "model": {"input_mode": "rawvla_raw3"},
            "data": {"proprio_history": 2},
            "backend": {"action_dim": 32, "num_denoise_steps": 10},
        },
        encoder=_FakeEncoder(proprio_dim=42),
        platform_spec=spec,
        checkpoint=Path("/tmp/ckpt.pt"),
        pi05_checkpoint=Path("/ws/pi05_robotwin/model.safetensors"),
        payload={"stage": "S2", "arm": "D", "seed": 101, "step": 100},
        stats_path=Path("/ws/pi05_robotwin/policy_norm_stats.json"),
    )
    assert metadata["norm"] == "MEAN_STD"
    assert metadata["n04"]["input_domain"] == "raw_linear_float"
    assert metadata["action_chunk_size"] == 50
    assert metadata["state_dim"] == 14
    assert "training_obs_image_size" not in metadata  # RoboTwin 帧尺寸未核实，不声张
    assert metadata["n04"]["input_cameras"] == ["head", "left", "right"]


# --------------------------------------------------------------------------------------
# 本体历史来源：客户端真实连续帧（优先） vs 最近 N 次策略调用（近似）
# --------------------------------------------------------------------------------------


def _history(seed: int = 0, value: float | None = None) -> np.ndarray:
    """payload 契约：``[proprio_history+1, 平台本体维]``、**旧→新**（含当前帧）、原始物理量。"""
    if value is not None:
        return np.full((3, 8), value, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (3, 8)).astype(np.float32)


def test_exact_history_is_used_verbatim_and_declared() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {"n04": {}})
    history = _history(seed=3)
    payload = _payload()
    payload["proprio_history"] = history
    response = service.handle_message({"type": "infer", "payload": payload})
    assert response["ok"] is True, response
    assert response["data"]["proprio_history_source"] == server.PROPRIO_SOURCE_EXACT
    call = bridge.calls[-1]
    # payload 是旧→新；编码器拿到训练同款"新→旧"拼接（逐值等于反向拼接，不补零）。
    assert torch.allclose(
        call["proprio"], torch.flip(torch.as_tensor(history), dims=[0]).reshape(-1), atol=1e-6
    ), (call["proprio"], history)
    assert call["metadata"]["padded_frames"] == 0
    assert call["metadata"]["ordering"] == "newest_first"
    # 同一 episode 再发一次：整段替换而不是再追加一帧（否则会变成 4 帧历史）。
    second = _history(seed=4)
    payload2 = _payload(seed=1, images=_images(1))
    payload2["proprio_history"] = second
    service.handle_message({"type": "infer", "payload": payload2})
    assert torch.allclose(
        bridge.calls[-1]["proprio"],
        torch.flip(torch.as_tensor(second), dims=[0]).reshape(-1),
        atol=1e-6,
    )
    assert service.metadata["n04"]["proprio_history_source"] == server.PROPRIO_SOURCE_EXACT
    assert policy.stats["proprio_history_source"] == server.PROPRIO_SOURCE_EXACT


def test_example_level_history_is_accepted() -> None:
    """协议 B（RoboTwin）把字段放在 example 里，也要认。"""
    policy, bridge = _policy(platform="robotwin")
    service = server.N04V11Service(policy, {})
    payload = {
        "examples": [
            {
                "image": [np.zeros((32, 32, 3), dtype=np.float32)] * 3,
                "lang": "place the block",
                "state": np.zeros(14, dtype=np.float32),
                "proprio_history": np.full((3, 14), 0.25, dtype=np.float32),
            }
        ]
    }
    response = service.handle_message({"type": "infer", "payload": payload})
    assert response["ok"] is True, response
    assert response["data"]["proprio_history_source"] == server.PROPRIO_SOURCE_EXACT
    assert torch.allclose(bridge.calls[-1]["proprio"], torch.full((42,), 0.25), atol=1e-6)


def test_missing_history_falls_back_to_policy_calls_approx() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    response = service.handle_message({"type": "infer", "payload": _payload()})
    assert response["ok"] is True, response
    assert response["data"]["proprio_history_source"] == server.PROPRIO_SOURCE_APPROX
    assert bridge.calls[-1]["metadata"]["padded_frames"] == 2  # 只有当前帧，补到 3 帧
    assert policy.stats["proprio_history_source"] == server.PROPRIO_SOURCE_APPROX


def test_bad_history_shape_errors_instead_of_truncating() -> None:
    policy, bridge = _policy()
    service = server.N04V11Service(policy, {})
    before = len(bridge.calls)

    short = _payload()
    short["proprio_history"] = _history()[:2]  # 少一帧
    message = _error_message(service.handle_message({"type": "infer", "payload": short}))
    assert "proprio_history" in message and "不截断也不补零" in message

    narrow = _payload()
    narrow["proprio_history"] = np.zeros((3, 5), dtype=np.float32)  # 维度不对
    assert "proprio_history" in _error_message(
        service.handle_message({"type": "infer", "payload": narrow})
    )

    bad = _payload()
    bad["proprio_history"] = np.full((3, 8), np.nan, dtype=np.float32)
    assert "NaN/Inf" in _error_message(service.handle_message({"type": "infer", "payload": bad}))
    assert len(bridge.calls) == before  # 报错的请求一帧都没融合


def test_metadata_declares_proprio_history_contract(tmp_path: Path) -> None:
    spec = PLATFORM_SPECS["libero"]
    checkpoint = tmp_path / "checkpoint-step0000100.pt"
    checkpoint.write_bytes(b"stub")
    metadata = server.build_metadata(
        spec=server.ServingSpec(checkpoint=checkpoint, platform="libero"),
        config={
            "stage": "S2",
            "arm": "D",
            "seed": 101,
            "model": {"input_mode": "rawvla_raw3"},
            "data": {"proprio_history": 2},
            "backend": {"action_dim": 32, "num_denoise_steps": 10},
        },
        encoder=_FakeEncoder(),
        platform_spec=spec,
        checkpoint=checkpoint,
        pi05_checkpoint=Path("/ws/pi05/model.safetensors"),
        payload={"stage": "S2", "arm": "D", "seed": 101, "step": 100},
        stats_path=Path("/ws/dataset_statistics.json"),
    )
    n04 = metadata["n04"]
    assert n04["proprio_history_source"] == server.PROPRIO_SOURCE_APPROX  # 尚未收到请求
    assert n04["proprio_history_modes"] == [
        server.PROPRIO_SOURCE_EXACT,
        server.PROPRIO_SOURCE_APPROX,
    ]
    assert n04["proprio_history_key"] == "proprio_history"
    assert n04["proprio_history_order"] == "oldest_to_newest"
    assert n04["proprio_history_units"] == "raw_physical"
    assert n04["proprio_history_frames"] == 3
    assert n04["proprio_frame_dim"] == 8
    assert n04["proprio_history"] == 2


def test_websocket_round_trip_with_metadata() -> None:
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack")
    if not _protocol_available():
        pytest.skip("需要 msgpack_numpy（starVLA deployment.model_server.tools 或 pip 包）")
    protocol = server.load_protocol_module()
    policy, bridge = _policy()
    metadata = {"env": server.PROTOCOL_ENV, "model": server.PROTOCOL_MODEL, "action_horizon": 10}
    service = server.N04V11Service(policy, metadata)
    port = _free_port()
    thread = threading.Thread(
        target=server.serve,
        kwargs={"service": service, "host": "127.0.0.1", "port": port},
        daemon=True,
    )
    thread.start()
    _wait_port(port)
    packer = protocol.Packer()
    ws_connect = pytest.importorskip("websockets.sync.client").connect
    connection = ws_connect(
        f"ws://127.0.0.1:{port}", compression=None, max_size=None, open_timeout=30
    )
    try:
        handshake = protocol.unpackb(connection.recv())
        assert handshake["env"] == server.PROTOCOL_ENV
        assert handshake["model"] == server.PROTOCOL_MODEL
        connection.send(packer.pack({"type": "infer", "payload": _payload(seed=7)}))
        response = protocol.unpackb(connection.recv())
        assert response["ok"] is True, response
        actions = response["data"]["normalized_actions"]
        assert actions.shape == (1, 10, 32)
        connection.send(packer.pack({"type": "reset", "payload": {"episode_id": "next"}}))
        reset_response = protocol.unpackb(connection.recv())
        assert reset_response["ok"] is True and reset_response["type"] == "reset"
        bad = _payload(images=[np.zeros((64, 64, 3), dtype=np.uint8)])
        connection.send(packer.pack({"type": "infer", "payload": bad}))
        error = protocol.unpackb(connection.recv())
        assert error["ok"] is False and "相机" in error["error"]["message"]
        connection.send(packer.pack({"type": "ping"}))
        pong = protocol.unpackb(connection.recv())
        assert pong["ok"] is True and pong["stats"]["requests"] == 1
    finally:
        connection.close()
    assert bridge.counter.calls == 1


def _protocol_available() -> bool:
    try:
        server.load_protocol_module()
    except ImportError:
        return False
    return True
