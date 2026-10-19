"""Numerical and episode-boundary checks for the external RawVLA runtime."""

import collections
from types import SimpleNamespace

import numpy as np
import pytest

from owac.backends.rawvla import load_audited
from owac.runtime.rawvla_robotwin import (
    ModelClient,
    normalize_state,
    proprio_history_enabled,
    reset_model,
    unnormalize_actions,
    validate_proprio_history,
)


class _StubRobotwinModel:
    """替身：不加载真权重，只回一个形状正确的动作块。"""

    def predict_action(self, examples):
        return {"normalized_actions": [np.zeros((50, 14), dtype=np.float32)]}


def _stub_policy(module, tmp_path):
    policy = module.Policy.__new__(module.Policy)  # 绕开真 checkpoint／CUDA，只测协议面
    policy.model = _StubRobotwinModel()
    policy.stats = {
        "state": {"mean": np.zeros(14, np.float32), "std": np.ones(14, np.float32)},
        "actions": {"mean": np.zeros(14, np.float32), "std": np.ones(14, np.float32)},
    }
    policy.evidence = tmp_path
    policy.input_index = 0
    return policy


def test_robotwin_policy_accepts_optional_proprio_history_without_changing_pi05_path(
    tmp_path, monkeypatch
):
    """带/不带 ``proprio_history`` 两条路径：都要通过，且 π0.5 输出逐位不变。

    该字段只做协议兼容与校验（真正的消费者是 N04 v1.1 服务端），所以带历史不能改变
    官方 π0.5 的动作输出；不带历史必须是扩展前的原行为。
    """
    import torch

    from owac.runtime import rawvla_robotwin as module

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    policy = _stub_policy(module, tmp_path)
    example = {
        "lang": "place the block",
        "image": [np.zeros((8, 8, 3), np.float32)] * 3,
        "state": np.zeros(14, np.float32),
    }
    without = policy.predict_action([example])
    with_history = policy.predict_action(
        [dict(example)], proprio_history=np.zeros((3, 14), np.float32)
    )
    np.testing.assert_array_equal(without["actions"], with_history["actions"])
    assert without["actions"].shape == (50, 14)

    # 前置批维也认（客户端可能整批堆叠）
    assert validate_proprio_history(np.zeros((1, 3, 14), np.float32)).shape == (3, 14)
    for bad in (
        np.zeros((2, 14), np.float32),  # 帧数不足
        np.zeros((3, 13), np.float32),  # 维度不对
        np.full((3, 14), np.nan, np.float32),  # 非有限
    ):
        with pytest.raises(ValueError):
            validate_proprio_history(bad)
        with pytest.raises(ValueError):
            policy.predict_action([dict(example)], proprio_history=bad)


class _CapturePolicyClient:
    """替身 websocket 客户端：只记下 payload，不发网络。"""

    def __init__(self):
        self.payloads = []

    def predict_action(self, payload):
        self.payloads.append(payload)
        return {"ok": True, "data": {"actions": np.zeros((50, 14), dtype=np.float32)}}


def _stub_client_model(module):
    model = module.ModelClient.__new__(module.ModelClient)  # 绕开 websockets 依赖
    model.client = _CapturePolicyClient()
    model.actions = None
    model.chunk_start = 0
    model.proprio_history = collections.deque(maxlen=module.ROBOTWIN_PROPRIO_HISTORY_FRAMES)
    return model


def _robotwin_example(value: float) -> dict:
    return {
        "lang": "place the block",
        "image": [np.zeros((8, 8, 3), np.float32)] * 3,
        "state": np.full(14, value, dtype=np.float32),
    }


def test_robotwin_client_switch_defaults_off_and_omits_the_field(monkeypatch):
    """默认关闭：payload 只有 ``examples``，与改动前逐位一致（一个字节都不多发）。"""
    from owac.runtime import rawvla_robotwin as module

    monkeypatch.delenv(module.PROPRIO_HISTORY_ENV, raising=False)
    assert proprio_history_enabled() is False
    for value in ("", "0", "false", "no"):
        monkeypatch.setenv(module.PROPRIO_HISTORY_ENV, value)
        assert proprio_history_enabled() is False
    monkeypatch.delenv(module.PROPRIO_HISTORY_ENV, raising=False)

    model = _stub_client_model(module)
    example = _robotwin_example(0.0)
    model.step(example, 0)
    payload = model.client.payloads[-1]
    assert set(payload) == {"examples"}
    assert payload["examples"][0] is example  # 原对象直通，未被改写或增字段


def test_robotwin_client_sends_exact_history_only_when_enabled(monkeypatch):
    from owac.runtime import rawvla_robotwin as module

    monkeypatch.setenv(module.PROPRIO_HISTORY_ENV, "1")
    assert proprio_history_enabled() is True
    for value in ("true", "YES", "on"):
        monkeypatch.setenv(module.PROPRIO_HISTORY_ENV, value)
        assert proprio_history_enabled() is True
    monkeypatch.setenv(module.PROPRIO_HISTORY_ENV, "1")

    model = _stub_client_model(module)
    states = [np.full(14, float(step), np.float32) for step in range(51)]
    for step, state in enumerate(states):
        model.step(
            {"lang": "x", "image": [np.zeros((8, 8, 3), np.float32)] * 3, "state": state}, step
        )
    first, second = model.client.payloads[0], model.client.payloads[1]
    # 首帧只有 1 帧真实观测：不足 3 帧就不发（服务端退回 policy_calls_approx）
    assert "proprio_history" not in first
    history = second["proprio_history"]
    assert history.shape == (3, 14) and history.dtype == np.float32
    np.testing.assert_allclose(history, np.stack(states[48:51]))  # 旧→新、含当前帧

    # 关掉开关：即使历史仍满，也不再多发任何字段
    monkeypatch.delenv(module.PROPRIO_HISTORY_ENV, raising=False)
    model.actions = None
    model.chunk_start = 0
    model.step(_robotwin_example(99.0), 51)
    assert set(model.client.payloads[-1]) == {"examples"}


def test_robotwin_client_history_resets_per_episode_and_drops_bad_values(monkeypatch):
    from owac.runtime import rawvla_robotwin as module

    monkeypatch.setenv(module.PROPRIO_HISTORY_ENV, "1")
    model = _stub_client_model(module)
    for step in range(3):
        model.step(_robotwin_example(float(step)), step)
    assert len(model.proprio_history) == 3

    reset_model(model)  # 作者 harness 每个 episode 调一次
    assert model.actions is None and len(model.proprio_history) == 0
    model.step(_robotwin_example(7.0), 3)
    assert "proprio_history" not in model.client.payloads[-1]
    assert len(model.proprio_history) == 1  # 只含新 episode 的真实帧

    # 坏值不上送：维度错 / 非有限都让 _proprio_history_payload 返回 None
    model.proprio_history.extend([np.zeros(14, np.float32)] * 2)
    model.proprio_history[-1] = np.full(14, np.nan, dtype=np.float32)
    assert model._proprio_history_payload() is None
    model.proprio_history.clear()
    model.proprio_history.extend(
        [np.zeros(13, np.float32), np.zeros(14, np.float32), np.zeros(14, np.float32)]
    )
    assert model._proprio_history_payload() is None


def test_robotwin_joint_order_and_zscore_preserve_gripper_values():
    mean = np.arange(14, dtype=np.float32)
    std = np.full(14, 0.25, dtype=np.float32)
    state = mean + np.arange(14, dtype=np.float32) / 10
    actual = normalize_state(state, {"mean": mean, "std": std})
    np.testing.assert_allclose(actual, (state - mean) / (std + 1e-8))
    assert actual[6] > 1  # No Qwen-style joint reorder, clipping or binarization.


def test_robotwin_action_unnormalization_keeps_left_and_right_grippers():
    mean = np.arange(14, dtype=np.float32)
    std = np.arange(1, 15, dtype=np.float32)
    normalized = np.ones((50, 32), dtype=np.float32)
    actual = unnormalize_actions(normalized, {"mean": mean, "std": std})
    np.testing.assert_allclose(actual, np.broadcast_to(mean + std, (50, 14)))
    assert actual[0, 6] == 13 and actual[0, 13] == 27


def test_robotwin_rejects_endpose_as_joint_state():
    with pytest.raises(ValueError, match="14 finite"):
        normalize_state(np.zeros(16), {"mean": np.zeros(14), "std": np.ones(14)})


def test_action_chunk_starts_at_actual_first_step_and_resets_between_episodes():
    class Client:
        calls = 0

        def predict_action(self, query):
            self.calls += 1
            actions = np.broadcast_to(np.arange(50)[:, None], (50, 14)).copy()
            return {"ok": True, "data": {"actions": actions}}

    model = ModelClient.__new__(ModelClient)
    model.client = Client()
    reset_model(model)
    assert model.step({}, 3)[0] == 0
    assert model.step({}, 52)[0] == 49
    assert model.step({}, 53)[0] == 0
    assert model.client.calls == 2
    reset_model(model)
    assert model.step({}, 0)[0] == 0
    assert model.client.calls == 3


class _Model:
    def state_dict(self):
        return {"action_head.action_out_proj.weight": np.zeros((14, 4))}

    def _normalize_state_dict(self, state):
        return state

    def _is_expected_openpi_missing_key(self, key):
        return False

    def load_state_dict(self, state, strict):
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"action_head.action_out_proj.weight": np.zeros((4, 14))},
        {"action_head.action_out_proj.weight": np.zeros((14, 4)), "unknown": np.zeros(1)},
    ],
)
def test_checkpoint_audit_rejects_missing_transposed_and_unknown_weights(weights):
    with pytest.raises(ValueError, match="does not match"):
        load_audited(_Model(), weights)


def test_checkpoint_audit_accepts_matching_readout():
    audit = load_audited(_Model(), {"action_head.action_out_proj.weight": np.zeros((14, 4))})
    assert audit["active_missing"] == audit["unexpected"] == audit["shape_mismatch"] == []
