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


class _FeatureBackendView:
    """Call pinned PI0 methods with an external prefix, without patching a shared model."""

    def __init__(self, backend, prefix):
        self.backend, self.prefix = backend, prefix

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        return self.prefix["prefix"], self.prefix["prefix_mask"], self.prefix["attention_blocks"]

    @staticmethod
    def _preprocess_observation(observation):
        return (
            (),
            (),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )


def tokenize_backend_task(backend, text, normalized_state):
    """Match PI0._prepare_batch: pad state BEFORE discrete-state tokenization."""
    import torch

    state = torch.as_tensor(normalized_state, dtype=torch.float32).reshape(-1)
    if state.numel() > backend.max_state_dim:
        raise ValueError("current state exceeds backend max_state_dim")
    fill = float(getattr(backend, "state_pad_value", 0.0))
    state = torch.nn.functional.pad(state, (0, backend.max_state_dim - state.numel()), value=fill)
    return backend.tokenizer.tokenize(
        text, state.cpu().numpy() if backend.discrete_state_input else None
    )


class RawVLAFeaturePolicy:
    """Use the actual pinned PI0 training and cached denoise methods.

    Language scaling, flow targets, adaRMS time conditioning and velocity decoding
    are inherited from the backend. No RGB preprocessing or visual tower is called.
    Backend state and actions must already use the frozen protocol's normalization.
    """

    def __init__(self, encoder, backend, config=None):
        self.encoder, self.backend = encoder, backend
        self.config = config or FeatureBridgeConfig()
        self.counter = FeatureEncoderCallCounter()
        self.readout_parameters = _count_parameters(getattr(encoder, "readout", None))
        for name in ("forward_from_observation", "sample_actions"):
            if not callable(getattr(type(backend), name, None)):
                raise TypeError(f"pinned StarVLA backend missing {name}")

    def embed_prefix_from_visual_tokens(self, visual_tokens, task_tokens, task_mask):
        import math

        import torch

        language = self.backend.vlm.embed_language_tokens(task_tokens)
        scale = math.sqrt(language.shape[-1])
        values = visual_tokens.values.to(language)
        mask = visual_tokens.mask.to(device=language.device, dtype=torch.bool)
        if mask.shape != values.shape[:2] or values.shape[-1] != language.shape[-1]:
            raise ValueError("visual values/mask/language width mismatch")
        if values.shape[1] % self.config.tokens_per_camera:
            raise ValueError("visual token count is not a camera budget multiple")
        prefix = torch.cat((values, language * scale), 1)
        pad = torch.cat((mask, task_mask.to(device=mask.device, dtype=torch.bool)), 1)
        return {
            "prefix": prefix,
            "prefix_mask": pad,
            "attention_blocks": torch.zeros_like(pad),
            "positions": pad.long().cumsum(1) - 1,
            "embedding_scale": scale,
            "num_visual_tokens": values.shape[1],
            "num_language_tokens": language.shape[1],
        }

    def encode_observation(self, frame, context, state):
        import torch

        camera = getattr(frame, "camera_id", getattr(context, "camera_id", None))
        if hasattr(frame, "values"):
            values = frame.values
            valid = getattr(frame, "valid_mask", None)
            bad = getattr(frame, "bad_mask", None)
            if bad is not None:
                valid = (torch.ones_like(bad) if valid is None else valid) * (1 - bad.to(values))
            ids = getattr(frame, "frame_ids", None)
            episodes = getattr(frame, "episode_ids", ())
            key = None if ids is None else (tuple(episodes), tuple(int(v) for v in ids))
            kwargs = dict(
                camera_id=camera,
                frame_key=key,
                measurement_valid=valid,
                saturation=getattr(frame, "saturated_mask", None),
                new_measurement=getattr(frame, "new_measurement", True),
            )
        else:
            values, kwargs = frame, dict(camera_id=camera)
        self.counter.record(_observation_key(frame))
        output = self.encoder.step(values, context, state, **kwargs)
        tokens = self.encoder.readout_tokens(output, self.config.tokens_per_camera)
        return tokens, output.state

    def _observation(self, context, task_tokens, task_mask, backend_state):
        from types import SimpleNamespace

        import torch

        if backend_state is None:
            backend_state = getattr(context, "proprio", None)
        if backend_state is None:
            raise ValueError("provide normalized current backend_state; proprio delta is not state")
        backend_state = torch.as_tensor(
            backend_state, device=task_tokens.device, dtype=torch.float32
        )
        if backend_state.ndim == 2:
            backend_state = backend_state[:, None]
        dim = self.backend.max_state_dim
        if backend_state.shape[-1] > dim:
            raise ValueError("backend state exceeds max_state_dim")
        backend_state = torch.nn.functional.pad(backend_state, (0, dim - backend_state.shape[-1]))
        return SimpleNamespace(
            images={},
            image_masks={},
            tokenized_prompt=task_tokens,
            tokenized_prompt_mask=task_mask,
            state=backend_state,
        )

    def _view(self, frame, context, state, task_tokens, task_mask, backend_state):
        visual, new_state = self.encode_observation(frame, context, state)
        prefix = self.embed_prefix_from_visual_tokens(visual, task_tokens, task_mask)
        observation = self._observation(context, task_tokens, task_mask, backend_state)
        return _FeatureBackendView(self.backend, prefix), observation, visual, new_state

    def forward_from_observation(
        self,
        frame,
        context,
        state,
        actions,
        task_tokens,
        task_mask,
        *,
        backend_state=None,
        action_mask=None,
        noise=None,
        time=None,
    ):
        import torch

        view, observation, visual, new_state = self._view(
            frame, context, state, task_tokens, task_mask, backend_state
        )
        if (
            actions.shape[-1] != self.config.control_dim
            or actions.shape[1] != self.config.action_horizon
        ):
            raise ValueError("control_dim/action_horizon mismatch")
        actions = torch.nn.functional.pad(
            actions, (0, self.config.action_dim - self.config.control_dim)
        )
        result = type(self.backend).forward_from_observation(
            view, observation, actions, noise=noise, time=time, return_debug=True
        )
        # Exclude tail padding and padded control dimensions from supervision.
        errors = (
            result["v_t"][..., : self.config.control_dim]
            - result["u_t"][..., : self.config.control_dim]
        ).square()
        mask = torch.ones_like(errors[..., 0]) if action_mask is None else action_mask.to(errors)
        result["loss"] = (errors * mask[..., None]).sum() / (
            mask.sum() * self.config.control_dim
        ).clamp_min(1)
        result.update(
            state=new_state,
            visual_tokens=visual,
            encoder_calls=1,
            num_visual_tokens=visual.values.shape[1],
            num_language_tokens=task_tokens.shape[1],
        )
        return result

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
        import torch

        steps = self.config.num_denoise_steps if num_denoise_steps is None else num_denoise_steps
        if steps < 1:
            raise ValueError("denoise steps must be positive")
        with torch.no_grad():
            view, observation, visual, new_state = self._view(
                frame, context, state, task_tokens, task_mask, backend_state
            )
            actions = type(self.backend).sample_actions(
                view, observation, noise=noise, num_steps=steps
            )
        return {
            "actions": actions,
            "control_actions": actions[..., : self.config.control_dim],
            "state": new_state,
            "encoder_calls": 1,
            "num_denoise_steps": steps,
        }

    def predict_action(self, *args, **kwargs):
        return self.sample_actions(*args, **kwargs)

    def reset(self, states, batch_mask):
        return self.encoder.reset(states, batch_mask)

    @contextlib.contextmanager
    def inference_mode(self):
        import torch

        with torch.no_grad():
            yield


def load_feature_audited(model, state):
    """Replace only the vision tower/projector; audit all language/action tensors."""
    from .rawvla import load_audited

    prefixes = ("vlm.model.model.vision_tower.", "vlm.model.model.multi_modal_projector.")
    normalized = model._normalize_state_dict(state)
    removed = sorted(k for k in normalized if k.startswith(prefixes))
    core = model.vlm.model.model
    core.vision_tower = None
    core.multi_modal_projector = None
    audit = load_audited(model, {k: v for k, v in normalized.items() if not k.startswith(prefixes)})
    audit["replaced_visual_keys"] = removed
    return audit


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
