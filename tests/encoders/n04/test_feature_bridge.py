"""N04 ⇄ π0.5 特征接入的形状、调用次数与梯度贯通测试。

**这不是真实 π0.5 验收。** 本机没有 StarVLA / π0.5 权重（数据与权重部署在记录中的
服务器 ``/workspace/owac/rawvla/``），因此这里用一个按 PI0 文档接口实现的极小 mock
后端，验证桥本身的语义：

* ``embed_prefix_from_visual_tokens`` 的形状、语言嵌入缩放、prefix mask、位置编号。
* 一个观测只运行**一次** encoder；10 次去噪迭代不产生 10 次新测量。
* 冻结语言/动作参数时动作损失仍回传到新编码器（真实梯度贯通，不是形状检查）。
* 内部 ``action_dim=32`` 与真实控制维度分开；horizon 与去噪次数是两个参数。

真实 π0.5 上的验收状态在实验记录中标注为“未做”。
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from owac.backends.rawvla_features import (
    CONTROLLED_TOKEN_DIM,
    CONTROLLED_TOKENS_PER_CAMERA,
    FeatureBridgeConfig,
    RawVLAFeaturePolicy,
    verify_gradient_path,
)
from owac.core.encoder_contract import (
    CameraSpec,
    ComputeBudget,
    EmbodimentContext,
    MeasurementFrame,
    VisualTokens,
)
from owac.frameworks.n04.config import N04Config, N04GeometryConfig, N04ModelConfig
from owac.frameworks.n04.ego import EgoMotionContext
from owac.frameworks.n04.model import N04Encoder

TOKENS = 4
DIM = 16
LANG = 5


class _MockVLM(nn.Module):
    """按 PI0 文档接口实现的最小 VLM：语言嵌入 + 缩放系数。"""

    def __init__(self, dim: int, language_embedding_scale: float = 0.7) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, dim)
        self.language_embedding_scale = language_embedding_scale

    def embed_language(self, task_tokens: torch.Tensor, task_mask: torch.Tensor) -> torch.Tensor:
        return self.embed(task_tokens.clamp(0, 63))


class _MockActionHead(nn.Module):
    """复用原 flow-matching 目标形式的占位头（``prepare_flow_targets``）。"""

    def __init__(self, dim: int, action_dim: int, control_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.control_dim = control_dim
        self.proj = nn.Linear(dim * 2, 1)
        self.flow_proj = nn.Linear(dim, action_dim)

    def prepare_flow_targets(self, actions: torch.Tensor, *, control_dim: int) -> dict:
        if actions.shape[-1] != control_dim:
            raise ValueError(f"真实控制维度应为 {control_dim}，实际 {actions.shape[-1]}")
        padded = torch.zeros(*actions.shape[:-1], self.action_dim, dtype=actions.dtype)
        padded[..., :control_dim] = actions
        noise = torch.zeros_like(padded)
        return {"actions": padded, "noise": noise, "target": padded - noise}


class _MockBackend(nn.Module):
    """镜像 PI0 的 ``embed_prefix`` / ``forward_from_observation`` / ``sample_actions`` 结构。"""

    def __init__(
        self, dim: int, action_dim: int, control_dim: int, language_embedding_scale: float = 0.7
    ) -> None:
        super().__init__()
        self.vlm = _MockVLM(dim, language_embedding_scale)
        self.action_head = _MockActionHead(dim, action_dim, control_dim)
        self.suffix = nn.Linear(action_dim, dim)

    def embed_action_suffix(self, actions: torch.Tensor) -> torch.Tensor:
        return self.suffix(actions)

    def forward_with_action_expert(
        self, prefix: torch.Tensor, suffix: torch.Tensor, attention_mask=None, **kwargs
    ) -> torch.Tensor:
        pooled = torch.cat(
            (prefix.mean(dim=1, keepdim=True).expand(-1, suffix.shape[1], -1), suffix), dim=-1
        )
        return self.action_head.proj(pooled).mean()

    def sample_flow(self, prefix: torch.Tensor, attention_mask=None, steps: int = 10, **kwargs):
        """模拟 N 次去噪迭代：只用 prefix，不重新编码测量。"""
        state = torch.zeros(prefix.shape[0], self.action_head.action_dim)
        velocity = self.action_head.flow_proj(prefix.mean(dim=1))
        for _ in range(steps):
            state = state + 0.01 * velocity
        return state


def _tiny_encoder(tokens: int = TOKENS, dim: int = DIM) -> N04Encoder:
    return N04Encoder(
        N04Config(
            input_mode="rawvla_rgb3",
            measurement_size=(16, 16),
            channels=3,
            camera_ids=("head",),
            geometry=N04GeometryConfig(feature_size=(16, 16), pixel_stride=1.0),
            model=N04ModelConfig(
                name="n04-bridge-test",
                width=8,
                blocks=1,
                scales=1,
                scale_strides=(1,),
                hidden=16,
                readout_tokens=tokens,
                readout_dim=dim,
                predictor="learned_proprio",
            ),
        )
    )


def _context(batch: int = 1) -> EgoMotionContext:
    return EgoMotionContext(
        proprio_delta=torch.zeros(batch, 4),
        proprio_delta_available=True,
        executed_action=torch.zeros(batch, 3),
        executed_action_available=True,
        dt=torch.ones(batch),
        dt_available=False,
        camera_index=torch.zeros(batch, dtype=torch.long),
        camera_id="head",
    )


def _bridge() -> tuple[RawVLAFeaturePolicy, N04Encoder, _MockBackend]:
    encoder = _tiny_encoder()
    backend = _MockBackend(DIM, action_dim=32, control_dim=7)
    cfg = FeatureBridgeConfig(
        action_dim=32,
        control_dim=7,
        action_horizon=10,
        num_denoise_steps=10,
        tokens_per_camera=TOKENS,
        token_dim=DIM,
        name="n04-tiny-bridge-test",
    )
    return RawVLAFeaturePolicy(encoder, backend, cfg), encoder, backend


def test_contract_types_are_lightweight_and_importable():
    """契约模块不依赖 torch，也不初始化模型或设备。"""
    import owac.core.encoder_contract as contract

    source = inspect.getsource(contract)
    assert "import torch" not in source
    assert "import numpy" not in source
    spec = CameraSpec(camera_id="head", height=16, width=16, channels=3)
    frame = MeasurementFrame(values=None, camera_id="head", frame_ids=(0,), episode_ids=("e",))
    assert spec.input_mode == "rawvla_rgb3"
    assert frame.availability()["valid_mask"] is False
    with pytest.raises(ValueError):
        CameraSpec(camera_id="c", height=8, width=8, channels=3, input_mode="bayer_cfa")
    assert EmbodimentContext(executed_action_available=False).executed_action_available is False
    assert ComputeBudget(max_updates=2, max_history=4).deadline_ms is None


def test_readout_shapes_match_the_controlled_config_shape():
    """读出形状：每相机固定 token 数、固定宽度；小尺寸配置必须单独命名。"""
    policy, encoder, _ = _bridge()
    state = encoder.init_state(1)
    frame = MeasurementFrame(
        values=torch.rand(1, 3, 16, 16), camera_id="head", frame_ids=(0,), episode_ids=("e",)
    )
    tokens, new_state = policy.encode_observation(frame, _context(), state["head"])
    #: 读出对象与契约的 `VisualTokens` 字段同名（values/mask），可互换使用。
    assert hasattr(tokens, "values") and hasattr(tokens, "mask")
    assert isinstance(VisualTokens, type) and not isinstance(tokens, MeasurementFrame)
    assert tokens.values.shape == (1, TOKENS, DIM)
    assert tokens.mask.shape == (1, TOKENS)
    assert tokens.mask.dtype == torch.bool
    assert policy.counter.calls == 1
    #: 真实受控配置常量本身不应被测试配置冒充
    assert policy.config.tokens_per_camera != CONTROLLED_TOKENS_PER_CAMERA
    assert policy.config.describe()["is_controlled_backend_config"] is False
    #: 受控配置常量与提示词一致
    assert CONTROLLED_TOKENS_PER_CAMERA == 256 and CONTROLLED_TOKEN_DIM == 2048
    assert new_state.measured is not None


def test_embed_prefix_preserves_language_scale_mask_and_positions():
    policy, encoder, backend = _bridge()
    state = encoder.init_state(1)
    frame = MeasurementFrame(
        values=torch.rand(1, 3, 16, 16), camera_id="head", frame_ids=(0,), episode_ids=("e",)
    )
    tokens, _ = policy.encode_observation(frame, _context(), state["head"])
    task_tokens = torch.randint(0, 64, (1, LANG))
    task_mask = torch.ones(1, LANG, dtype=torch.bool)
    task_mask[0, -1] = False
    prefix = policy.embed_prefix_from_visual_tokens(tokens, task_tokens, task_mask)
    assert prefix["prefix"].shape == (1, TOKENS + LANG, DIM)
    assert prefix["prefix_mask"].shape == (1, TOKENS + LANG)
    assert prefix["num_visual_tokens"] == TOKENS
    assert prefix["num_language_tokens"] == LANG
    assert prefix["embedding_scale"] == pytest.approx(backend.vlm.language_embedding_scale)
    assert prefix["positions"].tolist() == list(range(TOKENS + LANG))
    #: 语言嵌入确实乘了缩放系数
    raw = backend.vlm.embed_language(task_tokens, task_mask)
    expected = raw * backend.vlm.language_embedding_scale
    assert torch.allclose(prefix["prefix"][:, TOKENS:], expected, atol=1e-6)
    #: 视觉部分逐位等于读出结果
    assert torch.allclose(prefix["prefix"][:, :TOKENS], tokens.values, atol=1e-6)


def test_one_observation_runs_encoder_once_for_ten_denoise_steps():
    """10 次去噪迭代只运行一次 encoder：不把一次测量重复当成 10 次新测量。"""
    policy, encoder, _ = _bridge()
    state = encoder.init_state(1)
    frame = MeasurementFrame(
        values=torch.rand(1, 3, 16, 16), camera_id="head", frame_ids=(0,), episode_ids=("e",)
    )
    out = policy.sample_actions(
        frame,
        _context(),
        state["head"],
        torch.randint(0, 64, (1, LANG)),
        torch.ones(1, LANG, dtype=torch.bool),
        num_denoise_steps=10,
    )
    assert out["num_denoise_steps"] == 10
    assert out["encoder_calls"] == 1
    assert policy.counter.calls == 1
    assert out["actions"].shape == (1, 32)


def test_training_forward_keeps_autograd_to_the_new_encoder():
    """冻结语言/动作参数时，动作损失仍必须回传到新编码器（真实梯度检查）。"""
    policy, encoder, backend = _bridge()
    for parameter in backend.vlm.parameters():
        parameter.requires_grad_(False)
    for parameter in backend.action_head.parameters():
        parameter.requires_grad_(False)
    for parameter in backend.suffix.parameters():
        parameter.requires_grad_(False)
    state = encoder.init_state(2)
    frame = MeasurementFrame(
        values=torch.rand(2, 3, 16, 16),
        camera_id="head",
        frame_ids=(0, 0),
        episode_ids=("e0", "e1"),
    )
    task_tokens = torch.randint(0, 64, (2, LANG))
    task_mask = torch.ones(2, LANG, dtype=torch.bool)
    actions = torch.randn(2, 10, 7)  # 真实控制维度 7，horizon 10
    out = policy.forward_from_observation(
        frame, _context(2), state["head"], actions, task_tokens, task_mask
    )
    assert out["loss"].requires_grad
    out["loss"].backward()
    report = verify_gradient_path(
        policy,
        encoder_trainable_parameters=list(encoder.named_parameters()),
        forbidden_parameters=list(backend.named_parameters()),
    )
    assert report["ok"], report
    assert report["encoder_parameters_with_gradient"] > 0
    #: 读出层的梯度必须非零（读出属于编码器的可训练部分）
    assert report["gradient_norms"]["readout.proj.weight"] > 0
    assert not report["frozen_parameter_violations"]


def test_gradient_path_detects_a_detached_encoder():
    """反例：如果把编码器输出 detach，梯度检查必须报失败而不是静默通过。"""
    policy, encoder, backend = _bridge()
    original = policy.encode_observation

    def detached(frame, context, state):
        tokens, new_state = original(frame, context, state)
        #: 用 detached 的 values 重建读出对象（NamedTuple 不可就地改字段）。
        return type(tokens)(values=tokens.values.detach(), mask=tokens.mask), new_state

    policy.encode_observation = detached  # type: ignore[assignment]
    state = encoder.init_state(1)
    frame = MeasurementFrame(
        values=torch.rand(1, 3, 16, 16), camera_id="head", frame_ids=(0,), episode_ids=("e",)
    )
    out = policy.forward_from_observation(
        frame,
        _context(),
        state["head"],
        torch.randn(1, 10, 7),
        torch.randint(0, 64, (1, LANG)),
        torch.ones(1, LANG, dtype=torch.bool),
    )
    out["loss"].backward()
    report = verify_gradient_path(
        policy,
        encoder_trainable_parameters=list(encoder.named_parameters()),
        forbidden_parameters=[],
    )
    assert report["ok"] is False
    assert report["encoder_parameters_with_gradient"] == 0


def test_action_dim_padding_is_separate_from_control_dim():
    """内部 action_dim=32 的 padding 与真实控制维度分开；horizon 与去噪次数是两个参数。"""
    policy, encoder, backend = _bridge()
    with pytest.raises(ValueError):
        backend.action_head.prepare_flow_targets(torch.randn(1, 10, 14), control_dim=7)
    targets = backend.action_head.prepare_flow_targets(torch.randn(1, 10, 7), control_dim=7)
    assert targets["actions"].shape == (1, 10, 32)
    assert targets["actions"][..., 7:].abs().sum().item() == 0.0
    described = policy.config.describe()
    assert described["action_dim"] == 32 and described["control_dim"] == 7
    assert described["action_horizon"] == 10 and described["num_denoise_steps"] == 10
    assert described["tokens_per_camera"] == TOKENS and described["token_dim"] == DIM
