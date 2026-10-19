"""N04 v1.1 策略推理服务：N04 编码器 + 冻结 π0.5 后端，官方 starVLA/OpenPI websocket 协议。

正式 100 次闭环只能用"当前训练模型"作答；官方服务只暴露 π0.5 自己的视觉塔。本模块把
**N04 v1.1 编码器 + π0.5 后端**按同一协议暴露出去，让既有闭环客户端（作者 LIBERO 客户端／
WS-F 执行器）不改一行就能接上我们的模型。

协议（与 ``deployment/model_server/tools/websocket_policy_server.py`` 同形）：

* 连接建立后服务端先发 metadata（msgpack + numpy 数组）；
* 客户端每步发 ``{"type": "infer", "payload": {"examples": [{"image", "lang", "state"}],
  "noise"}}``（扁平 dict 也接受，``payload`` 缺省时整条消息当 payload）；
* 服务端回 ``{"status": "ok", "ok": true, "type": "inference_result",
  "data": {"normalized_actions": [1, horizon, action_dim]}}``；客户端自己用同一份
  q01/q99 统计反归一化。

与官方 π0.5 服务的差异（metadata 里如实声明，不冒充）：

1. 视觉前缀来自 N04 v1.1 编码器（RAW 测量 → 支持状态 → 读出），不是 π0.5 的视觉塔；
2. 服务**持有 episode 内状态**（每相机 128 支持 + 双模式 RKN），官方服务无状态；
3. 因此需要 episode 边界信号：``{"type": "reset"}`` 消息，或 payload 里的 ``episode_id``；
   两者都没有时只在任务文本变化时自动 reset；
4. 输入按客户端发送的 uint8 HWC 帧处理，服务端按平台固定缩放转成训练同款张量
   （libero：/255，与 ``raw/n04_v11_data.scaled_raw`` 一致）；RAW 还是 default_isp 由
   客户端的 ``--rawvla-bench-representation`` 决定，N04 臂 D 需要 ``raw``；
5. 客户端只发当前本体状态，服务端按"最近 N 次策略调用的状态"拼本体历史（训练用连续帧），
   这一处近似在 metadata 里显式写出。

数值错误一律暴露（NaN/Inf、形状不符、相机数不符、非法取值域），不返回零动作。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("owac.n04_v11_server")
_LOG_LEVEL = "INFO"


def configure_logging(level: str | None = None) -> None:
    """把本模块的日志固定在自己的 handler 上。

    starVLA／transformers 那侧的 import 会触发 ``logging.config.dictConfig`` 一类全局重配，
    把**先创建**的 logger 直接禁掉：实测服务端在模型构造之后的 metadata／每帧证据行会整段
    消失（只剩 websockets 自己的日志）。这里显式持有 handler 并且不 propagate，随后每次
    重活之后再调用一次，保证 100 次闭环的日志证据不会静默丢。
    """
    global _LOG_LEVEL
    if level is not None:
        _LOG_LEVEL = str(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOGGER.handlers = [handler]
    LOGGER.setLevel(getattr(logging, _LOG_LEVEL.upper(), logging.INFO))
    LOGGER.propagate = False
    LOGGER.disabled = False


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18000
DEFAULT_ROBOTWIN_PORT = 18005
DEFAULT_ACTION_DIM = 32
#: 与 ``ActionNormalizer.normalize`` 的 ``+1e-6`` 同一常量，逆变换要减掉同一量纲。
AMPLITUDE_EPS = 1e-6

#: 服务接受的输入域：LIBERO 是 uint8 计数域 0..255（浮点同域），RoboTwin 是 float32 线性原值
#: （identity，白电平已除，允许 ≤1 的暗帧）。LIBERO 的 [0,1] 归一化浮点一律报错——它与
#: "真正偏暗的原始计数帧"数值上不可区分，静默 ×255 会把错误藏起来。
INPUT_DOMAINS = {"libero": "raw_counts_0_255", "robotwin": "raw_linear_float"}
#: 平台固定缩放只在编码器 ``InputScale`` 施加一处；这里只做 dtype 归一（``scaled_raw``）。
INPUT_DOMAIN_NOTE = (
    "uint8(0..255) 或原始域浮点（libero=计数域 0..255、robotwin=线性原值）；"
    "libero 的 [0,1] 归一化浮点直接报错，不静默乘回 255"
)

#: 本体历史的两种来源：客户端发来的真实连续帧 / 服务端按最近 N 次策略调用近似。
PROPRIO_SOURCE_EXACT = "observation_frames"
PROPRIO_SOURCE_APPROX = "policy_calls_approx"
#: 客户端携带精确本体历史的 payload 键（顶层或 example 内），形状 [proprio_history+1, 平台本体维]。
PROPRIO_HISTORY_KEY = "proprio_history"
#: payload 里的时间顺序（客户端自然堆叠）：旧→新、含当前帧、原始物理量。
#: 服务端内部转成训练一致的"新→旧"再喂编码器。
PROPRIO_HISTORY_ORDER = "oldest_to_newest"
PROPRIO_HISTORY_UNITS = "raw_physical"

#: 如实标记本服务不是官方 π0.5 服务；客户端与报告据此区分。
PROTOCOL_ENV = "n04_v11_server"
PROTOCOL_MODEL = "N04-V1.1-PI05"


class N04V11ServingError(RuntimeError):
    """输入或服务状态不满足 N04 v1.1 服务契约；明确报错，不返回零动作。"""


# --------------------------------------------------------------------------------------
# 服务面配置
# --------------------------------------------------------------------------------------


@dataclass
class ServingSpec:
    """一次服务进程的入口参数；平台、checkpoint、端口、设备、GPU 都可配。"""

    checkpoint: Path
    platform: str | None = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str = "cuda:0"
    pi05_checkpoint: Path | None = None
    tokenizer: Path | None = None
    norm_stats: Path | None = None
    #: 客户端发的图像表示（raw / default_isp）；只用于如实声明与报告，服务端不做 ISP。
    input_representation: str = "raw"
    num_denoise_steps: int | None = None
    amp_dtype: str = "bfloat16"
    log_level: str = "INFO"


@dataclass
class _ServingFrame:
    """与 ``RawVLAFeaturePolicy.encode_observation`` 读取的字段同名的单帧多相机输入。"""

    values: list[Any]
    camera_id: list[str]
    frame_ids: tuple[int, ...]
    episode_ids: tuple[str, ...]
    saturated_mask: list[Any] | None = None
    valid_mask: None = None
    bad_mask: None = None
    new_measurement: bool = True


# --------------------------------------------------------------------------------------
# 归一化逆变换：ActionNormalizer 只有正向，这里给出同一统计文件的精确逆
# --------------------------------------------------------------------------------------


def denormalize_state(normalizer: Any, value: Any) -> Any:
    """``ActionNormalizer.normalize_state`` 的逆，用于还原训练同款物理量本体。

    训练时编码器读的是**物理量**本体历史（``window.proprio`` 未归一化），只有 π0.5 后端
    读归一化状态。客户端只发归一化状态，所以服务端必须还原后再喂编码器。
    """
    import torch

    if "state" not in normalizer.stats:
        raise N04V11ServingError(
            f"{normalizer.platform} 统计缺少 'state'；实际键 {sorted(normalizer.stats)}"
        )
    item = normalizer.stats["state"]
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    width = tensor.numel()
    if normalizer.platform == "libero":
        for field in ("q01", "q99"):
            if field not in item:
                raise N04V11ServingError(f"LIBERO state 统计缺少 {field!r}")
        low = torch.as_tensor(item["q01"], dtype=torch.float32)[:width]
        high = torch.as_tensor(item["q99"], dtype=torch.float32)[:width]
        return (tensor + 1.0) / 2.0 * (high - low + AMPLITUDE_EPS) + low
    for field in ("mean", "std"):
        if field not in item:
            raise N04V11ServingError(f"RoboTwin state 统计缺少 {field!r}")
    mean = torch.as_tensor(item["mean"], dtype=torch.float32)[:width]
    std = torch.as_tensor(item["std"], dtype=torch.float32)[:width]
    return tensor * (std + 1e-8) + mean


# --------------------------------------------------------------------------------------
# 策略：一次请求 = 一帧观测 → 动作块
# --------------------------------------------------------------------------------------


class N04V11Policy:
    """把 ``RawVLAFeaturePolicy`` 包成"单帧在线 + episode 携带状态"的服务端策略。

    一次请求只融合一次当前观测（编码器计数器增量必须恰好为 1），10 次去噪复用同一 prefix。
    episode 边界由 ``reset`` 驱动；``predict`` 里还会检查 ``episode_id`` 变化与任务文本变化。
    """

    def __init__(
        self,
        bridge: Any,
        backend: Any,
        *,
        platform_spec: Any,
        normalizer: Any,
        device: str = "cuda:0",
        proprio_history: int = 2,
        num_denoise_steps: int = 10,
        amp_dtype: str = "bfloat16",
        input_representation: str = "raw",
        max_state_dim: int | None = None,
    ) -> None:
        import torch

        self.bridge = bridge
        self.backend = backend
        self.platform_spec = platform_spec
        self.platform = str(platform_spec.name)
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.proprio_history = max(1, int(proprio_history))
        self.num_denoise_steps = int(num_denoise_steps)
        self.amp_dtype = str(amp_dtype)
        self.input_representation = str(input_representation)
        self.max_state_dim = int(
            max_state_dim if max_state_dim is not None else getattr(backend, "max_state_dim", 32)
        )
        self.cameras = tuple(platform_spec.camera_ids)
        self.control_dim = int(platform_spec.action_dim)
        self.action_horizon = int(platform_spec.action_horizon)
        self.action_dim = int(getattr(bridge.config, "action_dim", DEFAULT_ACTION_DIM))
        encoder = getattr(bridge, "encoder", None)
        encoder_dim = int(getattr(getattr(encoder, "config", None), "proprio_dim", 0) or 0)
        required = (self.proprio_history + 1) * self._platform_proprio_dim()
        if encoder_dim and encoder_dim != required:
            raise N04V11ServingError(
                f"encoder.proprio_dim={encoder_dim} 与 (proprio_history+1)×平台本体维 "
                f"{self.proprio_history + 1}×{self._platform_proprio_dim()}={required} 不一致；"
                "本体历史长度必须与训练配置一致"
            )
        self._state: Any = None
        self.proprio_source = PROPRIO_SOURCE_APPROX
        self._proprio: deque = deque(maxlen=self.proprio_history + 1)
        self._episode_id: str | None = None
        self._task_text: str | None = None
        self._frame_index = 0
        self._last_request_digest: str | None = None
        self._task_token_cache: tuple[str, Any] | None = None
        self.stats: dict[str, Any] = {
            "requests": 0,
            "resets": 0,
            "proprio_history_source": PROPRIO_SOURCE_APPROX,
            "duplicate_requests": 0,
            "encoder_calls": 0,
            "denoise_steps": 0,
            "last_actions": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # -- 生命周期 ------------------------------------------------------------------

    def _platform_proprio_dim(self) -> int:
        return int(self.platform_spec.proprio_dim)

    def reset(self, *, episode_id: str | None = None, reason: str = "") -> None:
        """episode 边界：清空携带状态、本体历史与帧计数，不跨 episode 继承。"""
        encoder = getattr(self.bridge, "encoder", None)
        if encoder is not None:
            encoder.reset(episode_id or "n04-v11-serving")
        self._state = None
        self._proprio.clear()
        self._episode_id = episode_id
        self._frame_index = 0
        self._task_token_cache = None
        self._last_request_digest = None
        self.stats["resets"] += 1
        LOGGER.info("episode reset: episode_id=%s reason=%s", episode_id, reason or "unspecified")

    # -- 输入解析 ------------------------------------------------------------------

    @staticmethod
    def _as_images(value: Any, cameras: Sequence[str]) -> list[Any]:
        images = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(images) != len(cameras):
            raise N04V11ServingError(
                f"相机数不符：服务端按平台相机集 {list(cameras)} 期望 {len(cameras)} 帧，"
                f"收到 {len(images)} 帧"
            )
        return images

    def _image_tensor(self, image: Any, camera: str) -> Any:
        """客户端帧 → 训练同款张量：uint8 HWC（或 float HWC）→ [C,H,W] float32 + 平台缩放。"""
        import numpy as np
        import torch

        from ..raw.n04_v11_data import scaled_raw

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise N04V11ServingError(
                f"{self.platform}/{camera} 帧需要 [H,W,3]，实际 {tuple(array.shape)} {array.dtype}"
            )
        if min(array.shape[0], array.shape[1]) <= 1:
            raise N04V11ServingError(f"{self.platform}/{camera} 帧尺寸非法 {tuple(array.shape)}")
        tensor = torch.as_tensor(
            np.ascontiguousarray(array.transpose(2, 0, 1)), dtype=torch.float32
        )
        if array.dtype == np.uint8:
            values = scaled_raw(tensor, self.platform)  # dtype 归一：0..255 原样进编码器
        elif np.issubdtype(array.dtype, np.floating):
            if not bool(torch.isfinite(tensor).all()):
                raise N04V11ServingError(f"{self.platform}/{camera} 帧含 NaN/Inf")
            if self.platform == "libero" and float(tensor.max()) <= 1.0 + 1e-6:
                raise N04V11ServingError(
                    f"libero/{camera} 帧看起来是 [0,1] 归一化浮点"
                    f"（max={float(tensor.max()):.4f}）：本服务只接受 uint8(0..255) 或"
                    "原始计数域浮点；请发原始域数值或改发 uint8，不静默乘回 255"
                )
            values = tensor
        else:
            raise N04V11ServingError(
                f"{self.platform}/{camera} 帧 dtype {array.dtype} 不支持（uint8 / 浮点）"
            )
        return values.to(self.device)

    def _saturation(self, values: Any) -> Any:
        """饱和标记。RoboTwin 没有记录白电平时返回 None（与训练侧 ``_saturation`` 同规则）。"""
        from ..raw.n04_v11_data import saturation_mask

        mask = saturation_mask(values.detach().cpu(), self.platform)
        return None if mask is None else mask.to(self.device)

    # -- 请求解析与执行 -------------------------------------------------------------

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """官方 ``predict_action`` 语义：``examples`` 列表（单条）或扁平 infer 字典。

        状态语义按平台事实分：LIBERO 的官方 openpi 客户端**已经**把本体归一化后放进
        ``example["state"]``；RoboTwin 协议 B（``owac.runtime.rawvla_robotwin``）发的是原始
        14 维关节量。两条路径都不再二次转换错。
        """
        import numpy as np
        import torch

        if not isinstance(payload, Mapping):
            raise N04V11ServingError(f"payload 需要 dict，实际 {type(payload).__name__}")

        if "examples" in payload:
            examples = payload["examples"]
            if not isinstance(examples, (list, tuple)) or not examples:
                raise N04V11ServingError("examples 需要非空列表")
            if len(examples) != 1:
                raise N04V11ServingError(
                    f"N04 v1.1 是单序列在线服务，examples 只能 1 条，实际 {len(examples)} 条"
                )
            example = examples[0]
            images = self._as_images(example.get("image"), self.cameras)
            text = str(example.get("lang", example.get("prompt", "")))
            # msgpack 解出来的 numpy 只读：显式拷一份，避免 torch 的只读缓冲告警。
            state = torch.as_tensor(np.array(example.get("state"), dtype=np.float32)).reshape(-1)
            state_is_normalized = self.platform == "libero"
            flow_seed = example.get("flow_seed", payload.get("flow_seed"))
        else:
            images = payload.get("image") or [
                payload.get(f"observation/{camera}") for camera in self.cameras
            ]
            images = self._as_images(images, self.cameras)
            text = str(payload.get("prompt", payload.get("lang", "")))
            raw = payload.get("observation/state", payload.get("state"))
            if raw is None:
                raise N04V11ServingError("缺少本体状态（observation/state 或 state）")
            state = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
            state_is_normalized = False
            flow_seed = payload.get("flow_seed")

        if not text.strip():
            raise N04V11ServingError("缺少语言指令（lang / prompt）")
        expected = self._platform_proprio_dim()
        if state.numel() != expected:
            raise N04V11ServingError(
                f"{self.platform} 本体状态需要 {expected} 维，实际 {state.numel()} 维"
            )
        if not bool(torch.isfinite(state).all()):
            raise N04V11ServingError("本体状态含 NaN/Inf")

        episode_id = payload.get("episode_id", payload.get("episode"))
        if episode_id is None and "examples" in payload:
            episode_id = examples[0].get("episode_id")
        if episode_id is not None:
            episode_id = str(episode_id)

        self._maybe_reset(episode_id, text)
        self._check_duplicate(images, text, state)

        normalized = state.unsqueeze(0) if state_is_normalized else self._normalize_state(state)
        frames = [
            self._image_tensor(image, camera)
            for image, camera in zip(images, self.cameras, strict=True)
        ]
        saturation = [self._saturation(frame) for frame in frames]
        if any(mask is None for mask in saturation):
            saturation = None
        exact_history = self._history_from_payload(
            payload, example if "examples" in payload else None
        )
        if exact_history is not None:
            # 客户端给的是**旧→新**的真实连续帧（含当前帧）：整段替换，不再追加当前帧。
            # deque 本身就是旧→新，所以按原序入队；读出的"新→旧拼接"由 _context 完成。
            self._proprio.clear()
            for row in range(exact_history.shape[0]):
                self._proprio.append(exact_history[row])
            self.proprio_source = PROPRIO_SOURCE_EXACT
        else:
            self._proprio.append(denormalize_state(self.normalizer, normalized[0]))
            self.proprio_source = PROPRIO_SOURCE_APPROX
        context = self._context()
        task_tokens, task_mask = self._task_tokens(normalized)
        noise = payload.get("noise")
        noise_tensor = None
        if noise is not None:
            noise_tensor = torch.as_tensor(np.array(noise, dtype=np.float32), device=self.device)
            if noise_tensor.dim() == 2:
                noise_tensor = noise_tensor.unsqueeze(0)
            if tuple(noise_tensor.shape) != (1, self.action_horizon, self.action_dim):
                raise N04V11ServingError(
                    f"noise 需要 [1,{self.action_horizon},{self.action_dim}]，"
                    f"实际 {tuple(noise_tensor.shape)}"
                )

        self._frame_index += 1
        frame = _ServingFrame(
            values=frames,
            camera_id=list(self.cameras),
            frame_ids=(self._frame_index,),
            episode_ids=(self._episode_id or "n04-v11-serving",),
            saturated_mask=saturation,
        )
        before = self._encoder_calls()
        if flow_seed is not None and noise_tensor is None:
            # RoboTwin 协议 B 的复现方式：按 flow_seed 播种，让后端自己的噪声采样可复现。
            torch.manual_seed(int(flow_seed))
        with self._autocast():
            result = self.bridge.sample_actions(
                frame,
                context,
                self._state,
                task_tokens,
                task_mask,
                backend_state=normalized,
                num_denoise_steps=self.num_denoise_steps,
                noise=noise_tensor,
            )
        fused = self._encoder_calls() - before
        if fused != 1:
            raise N04V11ServingError(
                f"一次请求融合了 {fused} 次观测；契约要求每观测只融合一次"
                "（10 次去噪复用同一 prefix）"
            )
        actions = result["actions"]
        if not bool(torch.isfinite(actions).all()):
            raise N04V11ServingError(
                "动作块含 NaN/Inf：拒绝返回（不退回零动作），请检查 checkpoint 与输入取值域"
            )
        shape = tuple(actions.shape)
        expected_shape = (1, self.action_horizon, self.action_dim)
        if shape != expected_shape:
            raise N04V11ServingError(f"动作块形状 {shape} 与期望 {expected_shape} 不一致")

        self._state = result.get("state")
        self.stats["requests"] += 1
        self.stats["proprio_history_source"] = self.proprio_source
        self.stats["encoder_calls"] += fused
        self.stats["denoise_steps"] += int(self.num_denoise_steps)
        control = actions[..., : self.control_dim].to(torch.float32)
        self.stats["last_actions"] = {
            "min": float(control.min()),
            "max": float(control.max()),
            "mean": float(control.mean()),
            "shape": list(shape),
            "camera_sizes": [list(frame.shape[-2:]) for frame in frames],
            "frame_index": self._frame_index,
        }
        if self.stats["requests"] == 1 or self._frame_index == 1:
            LOGGER.info(
                "episode %s frame %d: cameras=%s size=%s actions[%s] proprio=%s"
                " min=%.4f max=%.4f mean=%.4f",
                self._episode_id,
                self._frame_index,
                list(self.cameras),
                [tuple(frame.shape[-2:]) for frame in frames],
                shape,
                self.proprio_source,
                control.min(),
                control.max(),
                control.mean(),
            )
        if self.platform == "libero":
            # 官方 openpi 客户端自己用同一份 q01/q99 反归一化。
            return {
                "normalized_actions": actions.to(torch.float32).cpu().numpy(),
                "proprio_history_source": self.proprio_source,
            }
        # RoboTwin 协议 B（owac.runtime.rawvla_robotwin）：客户端不再反归一化，
        # 服务端直接给关节量纲 [horizon, control_dim]。
        return {
            "actions": self._environment_actions(actions),
            "proprio_history_source": self.proprio_source,
        }

    def _environment_actions(self, actions: Any) -> Any:
        """协议 B 的响应体：关节量纲、批维去掉后转 numpy。"""
        return self._environment_actions_value(actions).cpu().numpy()

    def _environment_actions_value(self, actions: Any) -> Any:
        """把归一化动作还原成平台关节量纲（与协议 B 的逆变换一致）。

        统计张量必须建在**动作所在设备**上：RoboTwin 冒烟实测到 cuda:0 动作配 cpu 统计会
        RuntimeError（第一版就踩了，协议 B 形状冒烟抓到）。
        """
        import torch

        item = self.normalizer.stats[self.normalizer.action_key]
        value = actions.to(torch.float32)[..., : self.control_dim]
        device = value.device
        if self.platform == "libero":
            low = torch.as_tensor(item["q01"], dtype=torch.float32, device=device)[
                : self.control_dim
            ]
            high = torch.as_tensor(item["q99"], dtype=torch.float32, device=device)[
                : self.control_dim
            ]
            result = (value + 1.0) / 2.0 * (high - low + AMPLITUDE_EPS) + low
        else:
            mean = torch.as_tensor(item["mean"], dtype=torch.float32, device=device)[
                : self.control_dim
            ]
            std = torch.as_tensor(item["std"], dtype=torch.float32, device=device)[
                : self.control_dim
            ]
            result = value * (std + 1e-8) + mean
        # 协议 B 的客户端按 [horizon, control_dim] 取用（batch=1 时去掉批维）。
        if result.shape[0] == 1:
            result = result[0]
        return result

    def _history_from_payload(
        self, payload: Mapping[str, Any], example: Mapping[str, Any] | None
    ) -> Any:
        """读客户端发来的**真实连续**本体历史；缺失返回 None（不回退到猜测）。

        契约：``proprio_history`` 形状 ``[proprio_history+1, 平台本体维]``（允许带一个前置
        批维 ``[1,H+1,D]``），**旧→新**（含当前帧，客户端自然堆叠顺序）、**原始物理量**。
        服务端内部转成训练一致的"新→旧"拼接后喂编码器。长度或维度不符直接报错，
        不静默截断、不补零。
        """
        import numpy as np
        import torch

        raw = payload.get(PROPRIO_HISTORY_KEY)
        if raw is None and example is not None:
            raw = example.get(PROPRIO_HISTORY_KEY)
        if raw is None:
            return None
        tensor = torch.as_tensor(np.array(raw, dtype=np.float32))
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        expected = (self.proprio_history + 1, self._platform_proprio_dim())
        if tensor.dim() != 2 or tuple(tensor.shape) != expected:
            raise N04V11ServingError(
                f"{PROPRIO_HISTORY_KEY} 需要 {expected} = [proprio_history+1, 平台本体维]"
                f"（新→旧、含当前帧），实际 {tuple(tensor.shape)}；不截断也不补零"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise N04V11ServingError(f"{PROPRIO_HISTORY_KEY} 含 NaN/Inf")
        return tensor

    def _maybe_reset(self, episode_id: str | None, text: str) -> None:
        if self._episode_id is None and self._task_text is None:
            # 进程内第一帧：携带状态本来就是空的，记 reset 只会污染计数。
            self._episode_id = episode_id or f"{self.platform}:unlabelled"
            self._task_text = text
            return
        if episode_id is not None and episode_id != self._episode_id:
            self.reset(episode_id=episode_id, reason="episode_id 变化")
            self._task_text = text
            return
        if self._task_text is not None and text != self._task_text:
            self.reset(episode_id=episode_id, reason="任务文本变化")
            self._task_text = text

    def _check_duplicate(self, images: Sequence[Any], text: str, state: Any) -> None:
        """同一 payload 重复出现只计数并告警，不静默当成新测量。"""
        import numpy as np

        digest = hashlib.sha256()
        digest.update(self.proprio_source.encode("utf-8"))
        digest.update(text.encode("utf-8"))
        digest.update(np.ascontiguousarray(state.numpy(), dtype=np.float32).tobytes())
        for image in images:
            digest.update(np.ascontiguousarray(image).tobytes())
        value = digest.hexdigest()
        if value == self._last_request_digest:
            self.stats["duplicate_requests"] += 1
            LOGGER.warning(
                "第 %d 帧收到与上一帧逐字节相同的观测（episode=%s）；已按新测量融合，"
                "该计数用于让调用方看见重复投喂",
                self._frame_index + 1,
                self._episode_id,
            )
        self._last_request_digest = value

    # -- 训练同款的上下文与 token ----------------------------------------------------

    def _normalize_state(self, state: Any) -> Any:
        return self.normalizer.normalize_state(state.reshape(1, -1))

    def _context(self) -> Any:
        import torch

        from ..core.encoder_contract import EmbodimentContext

        width = self.proprio_history + 1
        rows = list(self._proprio)
        padded = width - len(rows)
        rows = [rows[0]] * padded + rows
        history = torch.stack(rows[::-1]).reshape(-1).to(self.device)
        return EmbodimentContext(
            proprio=history,
            proprio_available=True,
            executed_action=None,
            executed_action_available=False,
            time_unit="frame",
            metadata={
                "platform": self.platform,
                "proprio_expr": getattr(self.platform_spec, "proprio_expr", ""),
                "history_frames": width,
                "ordering": "newest_first",
                "padded_frames": padded,
                "future_action_excluded": True,
                "source": "N04V11Policy: 客户端每次策略调用只发当前状态，历史按最近调用拼",
            },
        )

    def _task_tokens(self, normalized_state: Any) -> tuple[Any, Any]:
        import torch

        from .rawvla_features import tokenize_backend_task

        text = self._task_text or ""
        discrete = bool(getattr(self.backend, "discrete_state_input", False))
        if (
            not discrete
            and self._task_token_cache is not None
            and self._task_token_cache[0] == text
        ):
            return self._task_token_cache[1]
        tokens, mask = tokenize_backend_task(
            self.backend, text, normalized_state[0].detach().to("cpu").numpy()
        )
        result = (
            torch.as_tensor(tokens, device=self.device).to(torch.long).reshape(1, -1),
            torch.as_tensor(mask, device=self.device).to(torch.bool).reshape(1, -1),
        )
        if not discrete:
            self._task_token_cache = (text, result)
        return result

    def _encoder_calls(self) -> int:
        return int(getattr(getattr(self.bridge, "counter", None), "calls", 0))

    @contextlib.contextmanager
    def _autocast(self):
        import torch

        if self.device.type != "cuda" or self.amp_dtype in ("fp32", "float32", "none"):
            yield
            return
        dtype = torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=dtype):
            yield


# --------------------------------------------------------------------------------------
# websocket 服务：官方路由语义 + N04 的 reset 扩展
# --------------------------------------------------------------------------------------


class N04V11Service:
    """元数据 + 消息路由；``handle_message`` 是不依赖 websockets 的纯协议层。"""

    def __init__(self, policy: N04V11Policy, metadata: Mapping[str, Any]) -> None:
        self.policy = policy
        self.metadata = dict(metadata)

    def handle_message(self, message: Any) -> dict[str, Any]:
        if not isinstance(message, Mapping):
            return _error("default", f"消息需要 dict，实际 {type(message).__name__}")
        request_id = str(message.get("request_id", "default"))
        message_type = str(message.get("type", "infer"))
        payload = message.get("payload", message)
        if message_type == "ping":
            return {
                "status": "ok",
                "ok": True,
                "type": "ping",
                "request_id": request_id,
                "stats": dict(self.policy.stats),
            }
        if message_type in ("reset", "episode_reset"):
            payload = payload if isinstance(payload, Mapping) else {}
            self.policy.reset(
                episode_id=payload.get("episode_id"),
                reason=str(payload.get("reason", "客户端 reset")),
            )
            return {
                "status": "ok",
                "ok": True,
                "type": "reset",
                "request_id": request_id,
                "stats": dict(self.policy.stats),
            }
        if message_type in ("infer", "predict_action"):
            if not isinstance(payload, Mapping):
                return _error(
                    request_id,
                    f"payload 需要 dict，实际 {type(payload).__name__}",
                )
            try:
                data = self.policy.predict(payload)
            except Exception as error:  # noqa: BLE001 - 协议层把失败如实回给客户端
                LOGGER.exception("推理失败 request_id=%s", request_id)
                return _error(request_id, f"{type(error).__name__}: {error}")
            # metadata 是连接建立时发的快照；这里把"本次实际用的历史来源"同步进去，
            # 让随后写日志/报告的读者看到的是真实值而不是初始默认值。
            self.metadata.setdefault("n04", {})["proprio_history_source"] = (
                self.policy.proprio_source
            )
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": request_id,
                "data": data,
            }
        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": request_id,
            "error": {"message": f"Unsupported message type {message_type!r}"},
        }


def _error(request_id: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "ok": False,
        "type": "inference_result",
        "request_id": request_id,
        "error": {"message": message},
    }


def load_protocol_module() -> Any:
    """官方 msgpack+numpy 编解码：优先 starVLA 内实现的同一份代码，其次 pip 包。"""
    try:
        from deployment.model_server.tools import msgpack_numpy

        return msgpack_numpy
    except ImportError:
        pass
    try:
        import msgpack_numpy

        return msgpack_numpy
    except ImportError as error:  # pragma: no cover - 远端环境已具备
        raise ImportError(
            "websocket 服务需要 msgpack + numpy 编解码（starVLA deployment.model_server.tools."
            "msgpack_numpy 或 pip msgpack_numpy）；把 starVLA 根目录放进 PYTHONPATH"
        ) from error


def serve(service: N04V11Service, *, host: str, port: int) -> None:  # pragma: no cover - IO
    """按官方 handler 语义服务：先发 metadata，逐条消息回一条响应。"""
    import asyncio
    import traceback

    # websockets 15 的顶层延迟导入在本部署里取不到子模块属性（实测 attr 不存在但
    # 显式子模块导入可用）；官方服务写法 ``websockets.asyncio.server`` 会 AttributeError。
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import CloseCode

    configure_logging()  # websockets/msgpack 的 import 之后再钉一次，见 configure_logging
    protocol = load_protocol_module()
    packer = protocol.Packer()

    async def handler(websocket: Any) -> None:
        LOGGER.info("连接建立：%s", getattr(websocket, "remote_address", "?"))
        await websocket.send(packer.pack(service.metadata))
        while True:
            try:
                message = protocol.unpackb(await websocket.recv())
                response = service.handle_message(message)
                await websocket.send(packer.pack(response))
            except ConnectionClosed:
                LOGGER.info("连接关闭")
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def run() -> None:
        async with websocket_serve(handler, host, port, compression=None, max_size=None):
            LOGGER.info(
                "server listening on %s:%d metadata=%s",
                host,
                port,
                json.dumps(service.metadata, default=str),
            )
            print(f"READY {host}:{port} {PROTOCOL_ENV} {PROTOCOL_MODEL}", flush=True)
            await asyncio.Future()

    asyncio.run(run())


# --------------------------------------------------------------------------------------
# 构造：从 N04 v1.1 checkpoint 起服务
# --------------------------------------------------------------------------------------


def _readout_summary(encoder: Any) -> dict[str, Any] | None:
    import dataclasses

    config = getattr(encoder, "config", None)
    if config is None or not hasattr(config, "resolve_readout"):
        return None
    try:
        return dataclasses.asdict(config.resolve_readout())
    except Exception:  # noqa: BLE001 - 元数据不能因为读出配置读不出来就崩
        return None


def _load_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """读训练 checkpoint。优先 mmap：正式 checkpoint 含 π0.5 后端与优化器状态（数十 GB），
    一次性读进内存既慢又浪费；mmap 只在真正取用时落页。"""
    import torch

    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError, ValueError) as error:  # pragma: no cover - 老 torch 回退
        LOGGER.warning("mmap 读 checkpoint 失败（%s），回退普通加载", error)
        return torch.load(checkpoint, map_location="cpu", weights_only=False)


def build_service(spec: ServingSpec) -> tuple[N04V11Service, N04V11Policy]:
    """加载 N04 v1.1 checkpoint + 冻结 π0.5 后端，构造可服务的策略与元数据。"""
    import torch

    from ..backends.rawvla_features import FeatureBridgeConfig, RawVLAFeaturePolicy
    from ..raw.n04_v11_data import PLATFORM_SPECS, ActionNormalizer
    from ..training.n04_v11 import apply_lora, build_pi05_backend, build_v11_encoder

    checkpoint = Path(spec.checkpoint)
    if not checkpoint.is_file():
        raise N04V11ServingError(
            f"找不到 N04 v1.1 checkpoint {checkpoint}；--checkpoint 要传带 encoder/backend 的 "
            "checkpoint-step*.pt 文件本身"
        )
    payload = _load_checkpoint(checkpoint)
    from ..frameworks.n04.compatibility import require_current_implementation

    require_current_implementation(payload)
    for key in ("config", "encoder"):
        if key not in payload:
            raise N04V11ServingError(f"checkpoint 缺少 {key!r}；这不是 N04 v1.1 训练 checkpoint")
    config = dict(payload["config"])
    platform = str(spec.platform or config.get("platform", ""))
    if platform not in PLATFORM_SPECS:
        raise N04V11ServingError(
            f"未知平台 {platform!r}；支持 {sorted(PLATFORM_SPECS)}（用 --platform 指定）"
        )
    platform_spec = PLATFORM_SPECS[platform]
    device = torch.device(spec.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise N04V11ServingError(f"--device {spec.device} 需要 CUDA，但当前不可用")
    fraction = os.environ.get("N04V11_GPU_MEMORY_FRACTION")
    if fraction and device.type == "cuda":
        # 与训练共存时的自我保护：超预算先让自己 OOM，不去挤训练。
        torch.cuda.set_per_process_memory_fraction(float(fraction), device)
        LOGGER.info("CUDA 显存上限设为设备总量的 %.1f%%", float(fraction) * 100)

    model_cfg = dict(config["model"])
    model_cfg.pop("builder", None)
    arm = str(config.get("arm", "D"))
    encoder = build_v11_encoder(model_cfg, platform=platform, arm=arm)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.to(device).eval()

    backend_cfg = dict(config["backend"])
    backend_cfg.pop("builder", None)
    if spec.pi05_checkpoint is not None:
        backend_cfg.setdefault("checkpoints", {})
        backend_cfg["checkpoints"][platform] = str(spec.pi05_checkpoint)
    if spec.tokenizer is not None:
        backend_cfg["tokenizer"] = str(spec.tokenizer)
    backend = build_pi05_backend(backend_cfg, platform=platform)
    adapter = dict(config["backend"].get("adapter", {}) or {})
    lora = dict(adapter.get("lora", {}) or {})
    if lora.get("enabled", False):
        apply_lora(
            backend,
            rank=int(lora.get("rank", 16)),
            alpha=float(lora.get("alpha", 16.0)),
            targets=tuple(lora.get("targets", ()))
            or (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ),
        )
    backend_state = payload.get("backend")
    if backend_state:
        backend.load_state_dict(backend_state, strict=True)
    else:
        LOGGER.warning("checkpoint 没有 backend 权重，退回冻结 π0.5 权重 + 零初始化 LoRA")
    backend.to(device).eval()

    bridge_config = FeatureBridgeConfig(
        action_dim=int(backend_cfg.get("action_dim", DEFAULT_ACTION_DIM)),
        control_dim=int(platform_spec.action_dim),
        action_horizon=int(platform_spec.action_horizon),
        num_denoise_steps=int(spec.num_denoise_steps or backend_cfg.get("num_denoise_steps", 10)),
        tokens_per_camera=int(model_cfg.get("tokens_per_camera", 256)),
        token_dim=int(model_cfg.get("token_dim", 2048)),
        name=str(model_cfg.get("name", "owac-n04-v1.1")),
    )
    bridge = RawVLAFeaturePolicy(encoder, backend, bridge_config)

    stats_path = spec.norm_stats or Path(config["data"]["norm_stats"][platform])
    normalizer = ActionNormalizer.from_file(platform, stats_path)
    pi05_checkpoint = Path(
        spec.pi05_checkpoint or config.get("backend", {}).get("checkpoints", {}).get(platform, "")
    )

    configure_logging()  # starVLA／transformers import 之后重新接管本模块日志
    policy = N04V11Policy(
        bridge,
        backend,
        platform_spec=platform_spec,
        normalizer=normalizer,
        device=str(device),
        proprio_history=int(config["data"].get("proprio_history", 2)),
        num_denoise_steps=bridge_config.num_denoise_steps,
        amp_dtype=spec.amp_dtype if bool(config["optim"].get("amp", True)) else "fp32",
        input_representation=spec.input_representation,
    )
    metadata = build_metadata(
        spec=spec,
        config=config,
        encoder=encoder,
        platform_spec=platform_spec,
        checkpoint=checkpoint,
        pi05_checkpoint=pi05_checkpoint,
        payload=payload,
        stats_path=Path(stats_path),
    )
    return N04V11Service(policy, metadata), policy


def build_metadata(
    *,
    spec: ServingSpec,
    config: Mapping[str, Any],
    encoder: Any,
    platform_spec: Any,
    checkpoint: Path,
    pi05_checkpoint: Path,
    payload: Mapping[str, Any],
    stats_path: Path,
) -> dict[str, Any]:
    """如实声明这是 N04 v1.1；``checkpoint`` 保持官方语义（客户端据此解析 norm stats）。"""
    readout = _readout_summary(encoder)
    metadata = {
        # 官方客户端读这几个键；checkpoint 是**实际加载的冻结 π0.5 后端权重文件**，
        # 客户端据此找到同一份 dataset_statistics.json（N04 训练用的也是这一份）。
        "env": PROTOCOL_ENV,
        "model": PROTOCOL_MODEL,
        "checkpoint": str(pi05_checkpoint),
        "precision": spec.amp_dtype,
        "model_source": "openpi",
        "action_horizon": int(platform_spec.action_horizon),
        "action_dim": int(config.get("backend", {}).get("action_dim", DEFAULT_ACTION_DIM)),
        "norm": "quantile" if platform_spec.name == "libero" else "MEAN_STD",
        # RoboTwin 协议 B 的客户端读这两个键（LIBERO 官方客户端不读，额外键无害）。
        "action_chunk_size": int(platform_spec.action_horizon),
        "state_dim": int(platform_spec.proprio_dim),
        # 客户端不改发送内容（它发什么我们就按什么缩放），所以这里声明 none；
        # N04 的 RAW/ISP 选择由客户端 --rawvla-bench-representation 决定。
        "raw_frontend": "none",
        "raw_frontend_source_key": "image",
        "raw_frontend_burst_frames": 1,
        "raw_frontend_rnn_frames": 1,
        # N04 v1.1 如实声明（与官方 π0.5 服务区分：stage/arm/seed/step/readout/输入域）
        "n04": {
            "policy": "n04_v11",
            "stage": str(payload.get("stage", config.get("stage", "S2"))),
            "arm": str(payload.get("arm", config.get("arm", "D"))),
            "seed": payload.get("seed", config.get("seed")),
            "step": int(payload.get("step", 0)),
            "platform": platform_spec.name,
            "checkpoint": str(checkpoint),
            "pi05_checkpoint": str(pi05_checkpoint),
            "input_mode": str(config.get("model", {}).get("input_mode", "rawvla_raw3")),
            "input_representation": spec.input_representation,
            "input_cameras": list(platform_spec.camera_ids),
            "input_scale": platform_spec.value_scale,
            "input_domain": INPUT_DOMAINS[platform_spec.name],
            "input_domain_note": INPUT_DOMAIN_NOTE,
            "norm_stats": str(stats_path),
            "readout": readout,
            "supports_per_camera": int(getattr(encoder.config, "supports_per_camera", 0)),
            "obs_dim": int(getattr(encoder.config, "obs_dim", 0)),
            "proprio_dim": int(getattr(encoder.config, "proprio_dim", 0)),
            "proprio_history": int(config.get("data", {}).get("proprio_history", 2)),
            #: 运行时按每次请求更新：observation_frames（客户端发真实连续帧）或
            #: policy_calls_approx（退回"最近 N 次策略调用"近似）。
            "proprio_history_source": PROPRIO_SOURCE_APPROX,
            "proprio_history_modes": [PROPRIO_SOURCE_EXACT, PROPRIO_SOURCE_APPROX],
            "proprio_history_key": PROPRIO_HISTORY_KEY,
            "proprio_history_order": PROPRIO_HISTORY_ORDER,
            "proprio_history_units": PROPRIO_HISTORY_UNITS,
            "proprio_history_frames": int(config.get("data", {}).get("proprio_history", 2)) + 1,
            "proprio_frame_dim": int(platform_spec.proprio_dim),
            "proprio_source": (
                "优先用 payload.proprio_history（旧→新、含当前帧、原始物理量，服务端转新→旧）；"
                "缺失时退回最近 N 次策略调用拼接"
            ),
            "num_denoise_steps": int(
                spec.num_denoise_steps or config.get("backend", {}).get("num_denoise_steps", 10)
            ),
            "state_mode": "dual_mode_rkn",
            "episode_reset": '{"type":"reset"} 消息或 payload.episode_id；任务文本变化也会 reset',
            "tokens_per_camera": int(config.get("model", {}).get("tokens_per_camera", 256)),
            "token_dim": int(config.get("model", {}).get("token_dim", 2048)),
        },
        "n04_checkpoint": str(checkpoint),
        "n04_step": int(payload.get("step", 0)),
    }
    size = training_image_hw(platform_spec.name)
    if size is not None:
        metadata["training_obs_image_size"] = list(size)
    return metadata


def training_image_hw(platform: str) -> tuple[int, int] | None:
    """官方训练帧尺寸（客户端用它核对 eval 侧是否 resize 到同一分辨率）。"""
    return {"libero": (256, 256)}.get(platform)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="N04 v1.1 checkpoint-step*.pt 文件")
    parser.add_argument("--platform", default=None, choices=("libero", "robotwin"))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pi05-checkpoint", default=None, help="覆盖配置里的冻结 π0.5 权重")
    parser.add_argument("--tokenizer", default=None, help="覆盖配置里的 PaliGemma tokenizer")
    parser.add_argument("--norm-stats", default=None, help="覆盖配置里的动作／本体统计文件")
    parser.add_argument(
        "--input-representation",
        default="raw",
        choices=("raw", "default_isp"),
        help="如实声明客户端发的表示（N04 臂 D 需要 raw）",
    )
    parser.add_argument(
        "--num-denoise-steps", type=int, default=None, help="去噪次数（默认取配置）"
    )
    parser.add_argument("--amp-dtype", default="bfloat16", choices=("bfloat16", "float16", "fp32"))
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    configure_logging(str(args.log_level))
    port = (
        args.port
        if args.port is not None
        else (DEFAULT_ROBOTWIN_PORT if args.platform == "robotwin" else DEFAULT_PORT)
    )
    spec = ServingSpec(
        checkpoint=Path(args.checkpoint),
        platform=args.platform,
        host=str(args.host),
        port=int(port),
        device=str(args.device),
        pi05_checkpoint=Path(args.pi05_checkpoint) if args.pi05_checkpoint else None,
        tokenizer=Path(args.tokenizer) if args.tokenizer else None,
        norm_stats=Path(args.norm_stats) if args.norm_stats else None,
        input_representation=str(args.input_representation),
        num_denoise_steps=args.num_denoise_steps,
        amp_dtype=str(args.amp_dtype),
        log_level=str(args.log_level),
    )
    service, policy = build_service(spec)
    LOGGER.info("metadata: %s", json.dumps(service.metadata, ensure_ascii=False, default=str))
    LOGGER.info("policy ready: device=%s cameras=%s", policy.device, list(policy.cameras))
    serve(service, host=spec.host, port=spec.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
