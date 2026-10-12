"""π0.5（StarVLA / RawVLA）特征接入桥。

保持 **Python 3.10 可执行语法**：已部署的 RawVLA 运行时是 3.10，桥代码及其传递依赖
必须能在 ``PYTHONPATH=<owac>/src:<隔离的 StarVLA>/`` 下导入。因此本模块不使用
3.11+ 专有语法（``except*``、``Self``、``typing.override``、PEP 695 泛型等）。

接入位置（对应 PI0.py 的原有数据流）：

* ``embed_prefix``：原来逐相机 ``self.vlm.embed_image(image)`` 再拼语言嵌入。这里改用
  ``embed_prefix_from_visual_tokens``，**保留**原代码的语言嵌入缩放、prefix attention
  mask 与位置编号约定。
* ``forward_from_observation``：训练前向。复用 ``action_head.prepare_flow_targets``、
  ``embed_action_suffix``、``forward_with_action_expert`` 与原 flow-matching 目标。
* ``sample_actions`` / ``predict_action``：推理前向。复用原 flow 求解、缓存与后处理。
  **一个观测只运行一次 encoder**，其视觉前缀在 10 次去噪迭代中重复读取。

冻结语义：语言/动作参数用 ``requires_grad_(False)`` 冻结，但训练前向**不**用
``no_grad`` 包住整个后端，动作损失仍要回传到新编码器；推理才用 inference mode。
``verify_gradient_path`` 提供真实的梯度贯通验证。
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RawVLAFeaturePolicy",
    "FeatureBridgeConfig",
    "FeatureEncoderCallCounter",
    "verify_gradient_path",
]

#: 首个受控配置：每个有效相机 256 个 2048 维 token。小尺寸配置必须单独命名，
#: 不得冒称真实后端配置。
CONTROLLED_TOKENS_PER_CAMERA = 256
CONTROLLED_TOKEN_DIM = 2048


@dataclass
class FeatureBridgeConfig:
    """桥接配置。动作维度与 horizon 沿用冻结协议，不从 horizon 反推重规划节奏。"""

    #: 模型内部动作维度（π0.5 padding 到 32），与真实控制维度分开。
    action_dim: int = 32
    #: 真实控制维度：LIBERO 7、RoboTwin 14。
    control_dim: int = 7
    #: 动作 horizon：LIBERO 10、RoboTwin 50。
    action_horizon: int = 10
    #: 去噪次数是**另一个**参数，不等于 horizon。
    num_denoise_steps: int = 10
    tokens_per_camera: int = CONTROLLED_TOKENS_PER_CAMERA
    token_dim: int = CONTROLLED_TOKEN_DIM
    #: 标记小尺寸配置，避免冒充真实后端。
    name: str = "owac-n04-controlled"
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "control_dim": self.control_dim,
            "action_horizon": self.action_horizon,
            "num_denoise_steps": self.num_denoise_steps,
            "tokens_per_camera": self.tokens_per_camera,
            "token_dim": self.token_dim,
            "is_controlled_backend_config": (
                self.tokens_per_camera == CONTROLLED_TOKENS_PER_CAMERA
                and self.token_dim == CONTROLLED_TOKEN_DIM
            ),
        }


class FeatureEncoderCallCounter:
    """统计一次 ``sample_actions`` 里跑了几次 encoder。

    要求是 1：10 次去噪迭代只读同一份视觉前缀，不产生 10 次新测量。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.observations: list[Any] = []

    def record(self, observation_key: Any) -> None:
        self.calls += 1
        self.observations.append(observation_key)

    def reset(self) -> None:
        self.calls = 0
        self.observations = []


class RawVLAFeaturePolicy:
    """把 OWAC 编码器的 ``VisualTokens`` 接到 StarVLA/PI0 特征入口。

    通过明确的 wrapper 接入，不直接改正在评测的冻结目录。``backend`` 需要提供以下接口
    （StarVLA 的 ``PI0``/``PI05`` 子类具备）：

    * ``vlm.embed_language(task_tokens, task_mask)`` -> 语言嵌入 ``[B, L_lang, D]``
    * ``vlm.language_embedding_scale``：语言嵌入缩放系数（原代码约定）
    * ``action_head.prepare_flow_targets(actions, ...)``
    * ``embed_action_suffix(actions)`` -> 动作后缀嵌入
    * ``forward_with_action_expert(prefix, suffix, attention_mask=None)``
    * ``sample_flow(prefix, attention_mask=None, steps=...)``
    """

    def __init__(
        self,
        encoder: Any,
        backend: Any,
        config: FeatureBridgeConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.backend = backend
        self.config = config or FeatureBridgeConfig()
        self.counter = FeatureEncoderCallCounter()
        #: 视觉投影已被新读出替代：记录新增参数，便于报告。
        self.readout_parameters = _count_parameters(getattr(encoder, "readout", None))

    # -- 前缀 ---------------------------------------------------------------------

    def embed_prefix_from_visual_tokens(
        self,
        visual_tokens: Any,
        task_tokens: Any,
        task_mask: Any,
    ) -> dict[str, Any]:
        """用 OWAC 的 tokens/mask 构造 prefix，语义与 ``embed_prefix`` 保持一致。

        保留原代码的三条约定：语言嵌入缩放、prefix attention mask、位置编号。
        """
        language = self.backend.vlm.embed_language(task_tokens, task_mask)
        scale = getattr(self.backend.vlm, "language_embedding_scale", 1.0)
        language = language * scale
        values = visual_tokens.values
        mask = visual_tokens.mask
        if _shape(mask, 0) != _shape(values, 0):
            raise ValueError("视觉 token mask 与 values 的 batch 不一致")
        if _shape(values, 1) % self.config.tokens_per_camera != 0:
            raise ValueError(
                "视觉 token 数必须是每相机 token 数的整数倍；"
                f"实际 L={_shape(values, 1)}，每相机 {self.config.tokens_per_camera}"
            )
        prefix = _concat((values, language), dim=1)
        visual_mask = mask
        language_mask = _ones_like(task_mask)
        prefix_mask = _concat((visual_mask, language_mask), dim=1)
        positions = _arange(_shape(prefix, 1))
        return {
            "prefix": prefix,
            "prefix_mask": prefix_mask,
            "positions": positions,
            "embedding_scale": scale,
            "num_visual_tokens": _shape(values, 1),
            "num_language_tokens": _shape(language, 1),
        }

    # -- 一个观测只跑一次 encoder --------------------------------------------------

    def encode_observation(self, frame: Any, context: Any, state: Any) -> tuple[Any, Any]:
        """跑一次编码器并记录调用次数。返回 ``(visual_tokens, new_state)``。

        接受 `owac.core.encoder_contract.MeasurementFrame`，也接受直接的测量张量：
        契约帧会在这里被适配成编码器所需的张量，测量数值与掩码原样传递；相机身份等
        帧侧元数据并入运动上下文（不覆盖调用方已有字段）。
        """
        measured, extra_context = _adapt_measurement(frame)
        self.counter.record(_observation_key(frame))
        merged = _merge_context(context, extra_context)
        output = self.encoder.step(measured, merged, state)
        #: 契约的 ``readout(output, token_budget) -> VisualTokens``；命名一致的读出对象。
        tokens = self.encoder.readout_tokens(output, self.config.tokens_per_camera)
        return tokens, output.state

    def forward_from_observation(
        self,
        frame: Any,
        context: Any,
        state: Any,
        actions: Any,
        task_tokens: Any,
        task_mask: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """训练前向：测量直接走 OWAC loader/encoder，语言/状态/动作处理沿用原路径。

        函数体内**不出现** ``torch.no_grad()``：动作损失必须能回传到新编码器。
        """
        visual_tokens, new_state = self.encode_observation(frame, context, state)
        prefix = self.embed_prefix_from_visual_tokens(visual_tokens, task_tokens, task_mask)
        targets = self.backend.action_head.prepare_flow_targets(
            actions, control_dim=self.config.control_dim
        )
        suffix = self.backend.embed_action_suffix(targets["actions"])
        loss = self.backend.forward_with_action_expert(
            prefix["prefix"],
            suffix,
            attention_mask=prefix["prefix_mask"],
            **kwargs,
        )
        return {
            "loss": loss,
            "state": new_state,
            "visual_tokens": visual_tokens,
            "num_visual_tokens": prefix["num_visual_tokens"],
            "num_language_tokens": prefix["num_language_tokens"],
            "encoder_calls": self.counter.calls,
        }

    def sample_actions(
        self,
        frame: Any,
        context: Any,
        state: Any,
        task_tokens: Any,
        task_mask: Any,
        *,
        num_denoise_steps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """推理前向：一次 encoder + N 次去噪迭代，复用同一视觉前缀。

        推理使用 inference mode；这与训练前向保留 autograd 图形成明确分工。
        """
        import torch  # 延迟导入：本模块在无 torch 环境下也要能 import

        steps = self.config.num_denoise_steps if num_denoise_steps is None else num_denoise_steps
        before = self.counter.calls
        #: 编码与状态更新在 inference_mode 之外完成：这样返回的新状态仍是普通张量，
        #: 可以进入后续的训练/推理流程；只有去噪求解本身在 inference mode 内运行。
        visual_tokens, new_state = self.encode_observation(frame, context, state)
        prefix = self.embed_prefix_from_visual_tokens(visual_tokens, task_tokens, task_mask)
        with torch.inference_mode():
            actions = self.backend.sample_flow(
                prefix["prefix"], attention_mask=prefix["prefix_mask"], steps=steps, **kwargs
            )
        encoder_calls = self.counter.calls - before
        if encoder_calls != 1:
            raise AssertionError(f"一次采样应当只运行一次 encoder，实际 {encoder_calls} 次")
        return {
            "actions": actions,
            "state": new_state,
            "encoder_calls": encoder_calls,
            "num_denoise_steps": steps,
        }

    def predict_action(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """``sample_actions`` 的别名，保留原后端命名。"""
        return self.sample_actions(*args, **kwargs)

    @contextlib.contextmanager
    def inference_mode(self):
        """显式推理上下文，便于调用方在边界处使用。"""
        import torch

        with torch.inference_mode():
            yield


# --------------------------------------------------------------------------------------
# 冻结参数下的梯度贯通验证
# --------------------------------------------------------------------------------------


def verify_gradient_path(
    policy: Any,
    *,
    encoder_trainable_parameters: Sequence[Any],
    forbidden_parameters: Sequence[Any] = (),
) -> dict[str, Any]:
    """真实梯度贯通验证：冻结语言/动作参数时，损失仍须回传到新编码器。

    检查三件事：

    1. 至少一个编码器参数拿到非零梯度（否则说明被无意 detach 或整个后端被 no_grad 包住）。
    2. 被冻结的后端参数**没有**梯度（或梯度恒为零）。
    3. 返回每个编码器模块的梯度范数，便于定位是哪一段断了。
    """
    norms: dict[str, float] = {}
    nonzero = 0
    for name, parameter in _named(encoder_trainable_parameters):
        grad = getattr(parameter, "grad", None)
        norm = 0.0
        if grad is not None:
            norm = float(grad.detach().abs().sum().item())
        norms[name] = norm
        if norm > 0:
            nonzero += 1
    frozen_violations = []
    for name, parameter in _named(forbidden_parameters):
        grad = getattr(parameter, "grad", None)
        if grad is not None and float(grad.detach().abs().sum().item()) > 0:
            frozen_violations.append(name)
    return {
        "encoder_parameters_with_gradient": nonzero,
        "encoder_parameters_total": len(norms),
        "gradient_norms": norms,
        "frozen_parameter_violations": frozen_violations,
        "ok": nonzero > 0 and not frozen_violations,
    }


# --------------------------------------------------------------------------------------
# 极小的张量工具（避免在模块层依赖 torch，保持 3.10 与无 torch 环境可用）
# --------------------------------------------------------------------------------------


def _shape(tensor: Any, dim: int) -> int:
    return int(tensor.shape[dim])


def _concat(tensors: Sequence[Any], dim: int) -> Any:
    import torch

    return torch.cat(list(tensors), dim=dim)


def _ones_like(mask: Any) -> Any:
    import torch

    return torch.ones_like(mask, dtype=torch.bool if mask.dtype == torch.bool else mask.dtype)


def _arange(length: int) -> Any:
    import torch

    return torch.arange(length)


def _adapt_measurement(frame: Any) -> tuple[Any, Any | None]:
    """把契约帧适配成编码器输入。张量输入直接透传。"""
    values = getattr(frame, "values", frame)
    context = None
    camera_id = getattr(frame, "camera_id", None)
    if camera_id is not None:
        context = {"camera_id": camera_id}
    return values, context


def _merge_context(context: Any, extra: Any | None) -> Any:
    """把帧侧的相机身份合进运动上下文（不覆盖调用方已有的字段）。"""
    if not extra:
        return context
    for key, value in extra.items():
        if getattr(context, key, None) in (None, ""):
            try:
                setattr(context, key, value)
            except AttributeError:  # pragma: no cover - 冻结 dataclass 的兜底
                pass
    return context


def _observation_key(frame: Any) -> Any:
    frame_ids = getattr(frame, "frame_ids", None)
    camera = getattr(frame, "camera_id", "")
    return (camera, tuple(frame_ids) if frame_ids is not None else None)


def _count_parameters(module: Any) -> int:
    if module is None or not hasattr(module, "parameters"):
        return 0
    return int(sum(p.numel() for p in module.parameters()))


def _named(parameters: Sequence[Any]):
    for index, entry in enumerate(parameters):
        if isinstance(entry, tuple) and len(entry) == 2:
            yield entry[0], entry[1]
        else:
            yield f"param[{index}]", entry
