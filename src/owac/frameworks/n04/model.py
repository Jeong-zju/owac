"""N04 编码器：测量 stem、组合式传播/修正 block、predict/correct 分离接口与读出。

执行顺序与在线完全一致（`step` 逐帧调用）::

    stem:      f_t = MeasurementStem(raw_t, metadata_t)
    predict:   pred_t, warp_valid, uncertainty = propagate(state_{t-1}, context_t)
    decompose: innovation = InnovationDecomposer(f_t, pred_t, warp_valid, uncertainty, ...)
    update:    stable_t, change_t = update(pred_t, f_t, innovation)
    readout:   [stable_t, change_t, measured_feature_t, validity] -> tokens

`predict_before_observe()` 与 `correct_after_observe()` 是两个独立函数，便于检查
“当前图像不影响纯 predict 输出”：前者签名里没有图像参数。

同一帧调用多次（例如去噪迭代重复读 prefix）只更新一次状态：`step` 用 ``frame_key``
去重，重复调用直接返回缓存输出。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N04Config, N04ModelConfig
from .ego import EgoMotionContext, build_propagator
from .geometry import GridSpec
from .innovation import Innovation, InnovationDecomposer
from .state import N04State, new_state

__all__ = [
    "MeasurementStem",
    "N04Encoder",
    "N04Readout",
    "StepOutput",
    "TokenReadout",
    "measurement_saturation",
]


class TokenReadout(NamedTuple):
    """读出结果。字段名与 `owac.core.encoder_contract.VisualTokens` 一致，
    便于在不引入重依赖的情况下互换使用。"""

    values: torch.Tensor
    mask: torch.Tensor


# --------------------------------------------------------------------------------------
# 测量 stem
# --------------------------------------------------------------------------------------


def measurement_saturation(values: torch.Tensor, *, near_max: float = 0.99) -> torch.Tensor:
    """按**相对量程**给出饱和掩码。

    三通道 RawVLA 数据与 CFA 单测量面的量程不同，调用方需给出该数据已归一化到的
    单位量程；这里只在 ``values`` 已归一到 ``[0, 1]`` 时按 0.99 判定，并明确标注
    这是“近似饱和”，真实饱和位深由数据侧 ``saturated_mask`` 提供。
    """
    return (values >= near_max).to(values.dtype).amax(dim=1, keepdim=True)


def linear_measurement_planes(values, valid, input_mode):
    """Keep linear sensor samples; CFA planes retain their sampling offsets."""
    if input_mode != "bayer_cfa":
        return torch.where(valid > 0, values.detach(), 0), valid
    planes = [values[:, :, dy::2, dx::2] for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))]
    masks = [valid[:, :, dy::2, dx::2] for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))]
    support = torch.cat(masks, 1)
    return torch.where(support > 0, torch.cat(planes, 1).detach(), 0), support


def _resize_sampling_grid(grid, size):
    # Resize displacement, not absolute coordinates: identity remains exact at borders.
    b, h, w, _ = grid.shape
    origin = GridSpec(h, w).identity_grid(b, grid.device, grid.dtype)
    displacement = F.interpolate(
        (grid - origin).permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    return GridSpec(*size).identity_grid(b, grid.device, grid.dtype) + displacement


def measurement_from_linear_planes(planes, input_mode):
    """Restore sensor sampling positions without demosaicing or changing CFA colors."""
    if input_mode != "bayer_cfa":
        return planes
    b, c, h, w = planes.shape
    if c != 4:
        raise ValueError("CFA prediction requires four separate sampling planes")
    samples = planes.new_zeros(b, 1, 2 * h, 2 * w)
    for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        samples[:, :, dy::2, dx::2] = planes[:, i : i + 1]
    return samples


def warp_linear_measurement(values, valid, grid, input_mode):
    """Warp each linear plane in its own coordinates; normalize missing support."""
    if grid is None:
        return values, valid
    if input_mode == "bayer_cfa":
        h, w = values.shape[-2:]
        full_grid = _resize_sampling_grid(grid, (2 * h, 2 * w))
        warped, supports = [], []
        for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            sub_grid = full_grid[:, dy::2, dx::2] + full_grid.new_tensor(
                [(1 - 2 * dx) / (2 * w), (1 - 2 * dy) / (2 * h)]
            )
            support = F.grid_sample(valid[:, i : i + 1], sub_grid, align_corners=False)
            sample = F.grid_sample(
                values[:, i : i + 1] * valid[:, i : i + 1], sub_grid, align_corners=False
            )
            warped.append(sample / support.clamp_min(1e-6))
            supports.append(support)
        return torch.cat(warped, 1), torch.cat(supports, 1)
    grid = _resize_sampling_grid(grid, values.shape[-2:])
    support = F.grid_sample(valid, grid, align_corners=False)
    sample = F.grid_sample(values * valid, grid, align_corners=False)
    return sample / support.clamp_min(1e-6), support


class MeasurementStem(nn.Module):
    """把测量编码成特征场。

    主表征不在测量域做几何插值。带采样坐标的 CFA 测量先按平面编码，再在特征坐标下传播
    （`ego`/`geometry` 只作用于特征场）。CFA 的四个子平面**不是**四个共址颜色通道：
    每个子平面在 2x2 周期内各自有固定的行列采样相位，本模块显式保留该相位。
    线性测量的辅助对齐仅用于照明统计，保持滤色平面分离，不生成中间 RGB。
    """

    def __init__(
        self,
        *,
        channels: int,
        width: int,
        input_mode: str = "rawvla_rgb3",
        cfa_pattern: str = "RGGB",
        pixel_stride: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.input_mode = input_mode
        self.cfa_pattern = cfa_pattern
        self.pixel_stride = float(pixel_stride)
        if input_mode == "bayer_cfa":
            #: 每个 CFA 相位一个可学习平面嵌入；共享同一 3x3 局部算子。
            self.plane_embed = nn.Parameter(torch.zeros(4, width))
            nn.init.normal_(self.plane_embed, std=0.02)
            self.conv = nn.Conv2d(1, width, 3, padding=1)
            self.phase_offset = self._phase_offsets(cfa_pattern)
        else:
            stride = max(1, int(round(pixel_stride)))
            self.conv = nn.Sequential(
                nn.Conv2d(channels, width, 3, stride=stride, padding=1),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1),
            )
            self.phase_offset = None
        self.norm = nn.GroupNorm(1, width)

    @staticmethod
    def _phase_offsets(pattern: str) -> list[tuple[int, int]]:
        """CFA 四个子平面在 2x2 周期内的实际采样偏移 (row, col)（RGGB/BGGR/GRBG/GBRG）。"""
        table = {
            "RGGB": [("R", 0, 0), ("Gr", 0, 1), ("Gb", 1, 0), ("B", 1, 1)],
            "BGGR": [("B", 0, 0), ("Gb", 0, 1), ("Gr", 1, 0), ("R", 1, 1)],
            "GRBG": [("Gr", 0, 0), ("R", 0, 1), ("B", 1, 0), ("Gb", 1, 1)],
            "GBRG": [("Gb", 0, 0), ("B", 0, 1), ("R", 1, 0), ("Gr", 1, 1)],
        }[pattern]
        #: 顺序固定为 (00, 01, 10, 11) 的 (row, col) 偏移。
        return [(r, c) for _, r, c in table]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "bayer_cfa":
            return self._forward_cfa(values)
        out = self.conv(values)
        return self.norm(out)

    def _forward_cfa(self, values: torch.Tensor) -> torch.Tensor:
        """按 CFA 相位分成四个子平面编码，保留每平面的采样偏移与缺失位置。

        缺失/坏点位置不插值：先做掩码归一化的 3x3 卷积，再把该位置的结果置零，并在
        特征域保留“此处无测量”的信息（由调用方通过 validity 通道携带）。
        """
        b, c, h, w = values.shape
        if c != 1:
            raise ValueError("bayer_cfa 输入必须是单测量面 [B,1,H,W]")
        if h % 2 or w % 2:
            raise ValueError("CFA 测量网格的边长必须是偶数")
        planes = torch.cat(
            (
                values[:, :, 0::2, 0::2],
                values[:, :, 0::2, 1::2],
                values[:, :, 1::2, 0::2],
                values[:, :, 1::2, 1::2],
            ),
            dim=1,
        )  # [B,4,h/2,w/2]
        #: 每个平面单独卷积，再按平面嵌入混合；通道维混合发生在平面内部，
        #: 因此不会把四个不同采样位置的测量当成共址颜色。
        plane_feat = self.conv(planes.reshape(b * 4, 1, h // 2, w // 2))
        plane_feat = plane_feat.reshape(b, 4, self.width, h // 2, w // 2)
        semantics = {
            "RGGB": (0, 1, 2, 3),
            "BGGR": (3, 2, 1, 0),
            "GRBG": (1, 0, 3, 2),
            "GBRG": (2, 3, 0, 1),
        }[self.cfa_pattern]
        plane_feat = plane_feat + self.plane_embed[list(semantics)][None, :, :, None, None]
        #: 放回 2x2 相位网格：像素 (2i+r, 2j+c) 只携带来自平面 (r,c) 的测量。
        feat = values.new_zeros(b, self.width, h, w)
        feat[:, :, 0::2, 0::2] = plane_feat[:, 0]
        feat[:, :, 0::2, 1::2] = plane_feat[:, 1]
        feat[:, :, 1::2, 0::2] = plane_feat[:, 2]
        feat[:, :, 1::2, 1::2] = plane_feat[:, 3]
        return self.norm(feat)


# --------------------------------------------------------------------------------------
# 组合式 block 与更新规则
# --------------------------------------------------------------------------------------


class N04Block(nn.Module):
    """一个可组合的传播/修正 block。

    同一套 predict/correct 规则在不同尺度上复用（提示词要求“核心传播/更新应能组合为
    多个 block，并用一维移动信号例子复用其 predict/correct 规则”）。block 只做特征
    混合，真实的状态转移由 `N04Encoder` 的传播算子统一给出。
    """

    def __init__(self, *, width: int, hidden: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(3 * width, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, width, 1),
        )
        self.norm = nn.GroupNorm(1, width)
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(
        self, stable: torch.Tensor, change: torch.Tensor, measured: torch.Tensor
    ) -> torch.Tensor:
        correction = self.mix(torch.cat((measured - stable, change, measured), dim=1))
        return stable + 0.1 * torch.tanh(correction)


# --------------------------------------------------------------------------------------
# 读出
# --------------------------------------------------------------------------------------


class N04Readout(nn.Module):
    """把 N04 表示变成 π0.5 视觉前缀形状的 token。

    第一个受控配置取每个有效相机 ``readout_tokens`` 个、``readout_dim`` 维 token。
    真实 π0.5 前缀为 2048；小尺寸配置（如 128）**单独命名**，不冒充真实后端配置。
    相机/空间编码由本层承担，学习参数计入编码器。
    """

    def __init__(self, *, in_channels: int, tokens: int, dim: int, cameras: int) -> None:
        super().__init__()
        self.tokens = int(tokens)
        self.dim = int(dim)
        self.grid = self._grid_for_tokens(self.tokens)
        self.proj = nn.Conv2d(in_channels, self.dim, 1)
        self.position = nn.Parameter(torch.zeros(1, self.tokens, self.dim))
        nn.init.normal_(self.position, std=0.02)
        self.camera = nn.Embedding(max(1, cameras), self.dim)
        nn.init.normal_(self.camera.weight, std=0.02)

    @staticmethod
    def _grid_for_tokens(tokens: int) -> tuple[int, int]:
        side = int(tokens**0.5)
        while side > 1 and tokens % side:
            side -= 1
        return side, max(1, tokens // side)

    def forward(
        self,
        representation: torch.Tensor,
        validity: torch.Tensor,
        camera_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = representation.shape[0]
        gh, gw = self.grid
        pooled = F.adaptive_avg_pool2d(representation, (gh, gw))
        pooled_valid = F.adaptive_avg_pool2d(validity, (gh, gw))
        tokens = self.proj(pooled).flatten(2).transpose(1, 2)  # [B, L, dim]
        tokens = tokens + self.position[:, : tokens.shape[1]]
        if camera_index is not None:
            idx = camera_index if camera_index.dim() == 1 else camera_index.reshape(b)
            tokens = tokens + self.camera(idx.clamp(0, self.camera.num_embeddings - 1))[:, None, :]
        mask = (pooled_valid.flatten(2).transpose(1, 2) > 0.5).squeeze(-1)
        return tokens, mask


# --------------------------------------------------------------------------------------
# 编码器
# --------------------------------------------------------------------------------------


@dataclass
class StepOutput:
    """`N04Encoder.step` 的返回。"""

    representation: torch.Tensor | list[torch.Tensor]
    state: N04State
    validity: torch.Tensor
    innovation: Innovation | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    updated: bool = True
    #: 缓存命中的帧键（同一帧重复调用时为非空）。
    cache_key: Any = None


class N04Encoder(nn.Module):
    """N04 编码器。每个相机维护独立状态（`init_state` 返回按相机分组的 dict）。"""

    def __init__(self, config: N04Config) -> None:
        super().__init__()
        self.config = config
        m: N04ModelConfig = config.model
        self.motion_dim = _motion_dim(config)
        self.spec = GridSpec(
            config.geometry.feature_size[0],
            config.geometry.feature_size[1],
            align_corners=False,
            padding_mode="zeros",
            pixel_stride=float(config.geometry.pixel_stride),
        )
        self.stem = MeasurementStem(
            channels=config.channels,
            width=m.width,
            input_mode=config.input_mode,
            cfa_pattern=config.cfa_pattern,
            pixel_stride=config.geometry.pixel_stride,
        )
        self.propagator = build_propagator(
            m.predictor, model_config=m, geometry=config.geometry, motion_dim=self.motion_dim
        )
        self.decomposer = InnovationDecomposer(
            feature_channels=m.width,
            hidden=m.hidden,
            sigma_floor=m.sigma_floor,
            use_illumination_branch=m.illumination_branch,
            use_change_branch=m.change_branch,
        )
        strides = m.scale_strides_for()
        self.blocks = nn.ModuleList(
            [N04Block(width=m.width, hidden=m.hidden) for _ in range(max(1, m.blocks))]
        )
        #: 多尺度：高尺度由最高分辨率场平均池化得到，规则可组合。
        self.scale_strides = tuple(strides)
        self.downsample = nn.ModuleList(
            [
                nn.Conv2d(m.width, m.width, 3, stride=2, padding=1)
                for _ in range(max(0, len(strides) - 1))
            ]
        )
        self.readout = N04Readout(
            in_channels=3 * m.width,
            tokens=m.readout_tokens,
            dim=m.readout_dim,
            cameras=max(1, len(config.camera_ids)),
        )
        #: 变化 cell 的写入强度先验（可学习，初始 0.1，避免首步就把变化写满）。
        self.change_write = nn.Parameter(torch.tensor(0.1))
        #: 变化 cell 的保留率 logit（可学习，初始 sigmoid(1.0)≈0.73）。
        self.change_decay = nn.Parameter(torch.tensor(1.0))
        self._cache: dict[Any, StepOutput] = {}

    def ensure_motion_dim_from_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """按 checkpoint 的参数形状预先建出懒初始化层。

        运动层需要恢复宽度；分解层构造时已预建：

        * ``learned_proprio`` 输入宽度 =
          ``motion_dim + 3(availability) + camera_embedding``，
          而 ``motion_dim`` 默认由数据侧在首次 ``step`` 时确定；
        * 分解器输入宽度固定为 4*C+3，恢复时显式检查兼容性。

        恢复 checkpoint 时还没跑过任何一步，必须先按 checkpoint 记录的真实宽度建层，
        否则 ``load_state_dict`` 会把训练好的参数判为 unexpected keys。
        """
        weight = state_dict.get("propagator.motion_encoder.0.weight")
        if weight is not None:
            ensure = getattr(self.propagator, "ensure_motion_dim", None)
            if ensure is not None:
                ensure(int(weight.shape[1]) - 3 - int(self.config.model.camera_embedding))
        trunk = state_dict.get("decomposer.trunk.0.weight")
        if trunk is not None:
            ensure_decomposer = getattr(self.decomposer, "ensure_in_channels", None)
            if ensure_decomposer is not None:
                ensure_decomposer(int(trunk.shape[1]))

    # -- 生命周期 -----------------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        camera_ids: tuple[str, ...] | list[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, N04State]:
        """每个相机一份独立状态。"""
        cameras = tuple(camera_ids) if camera_ids is not None else self.config.camera_ids
        self._cache.clear()
        h, w = self.config.geometry.feature_size
        return {
            cam: new_state(
                batch_size,
                self.config.model.width,
                h,
                w,
                camera_id=cam,
                device=device,
                dtype=dtype,
                scales=len(self.scale_strides),
            )
            for cam in cameras
        }

    def reset(self, state: dict[str, N04State], batch_mask: torch.Tensor) -> dict[str, N04State]:
        """部分 batch reset：只重置 ``batch_mask`` 为 True 的样本。"""
        self._cache.clear()
        return {cam: s.masked_reset(batch_mask) for cam, s in state.items()}

    @staticmethod
    def detach_state(state: dict[str, N04State]) -> dict[str, N04State]:
        """TBPTT 边界：切断图，保留值。"""
        return {cam: s.detach() for cam, s in state.items()}

    # -- predict / correct --------------------------------------------------------

    def predict_before_observe(
        self,
        state: N04State,
        context: EgoMotionContext,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor, torch.Tensor, Any]:
        """纯预测。**签名里没有测量参数**：当前图像不可能影响输出。

        只使用上一状态与截至当前已实际发生的运动上下文，返回
        ``(pred, validity, uncertainty, propagation_result)``。
        """
        prev = state.stable
        had_history = state.history_valid
        result = self.propagator(prev, self.spec, context)
        # Discrete support is detached; learned uncertainty retains its training path.
        validity = result.validity.detach()
        if state.warp_validity is not None and getattr(result, "backward_grid", None) is not None:
            validity = (
                validity
                * F.grid_sample(
                    state.warp_validity,
                    result.backward_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).detach()
            )
        uncertainty = result.uncertainty
        if torch.is_tensor(had_history):
            keep = had_history.reshape(-1, 1, 1, 1).to(validity.dtype)
            validity = validity * keep
        elif not had_history:
            validity = torch.zeros_like(validity)
        return result.predicted, validity, uncertainty, result

    def correct_after_observe(
        self,
        state: N04State,
        predicted: torch.Tensor | list[torch.Tensor],
        measured: torch.Tensor | list[torch.Tensor],
        validity: torch.Tensor,
        uncertainty: torch.Tensor,
        *,
        saturation: torch.Tensor | None = None,
        availability: torch.Tensor | bool = True,
        measurement_valid: torch.Tensor | None = None,
        backward_grid: torch.Tensor | None = None,
        linear_current: torch.Tensor | None = None,
        linear_predicted: torch.Tensor | None = None,
        linear_validity: torch.Tensor | None = None,
        current_linear_support: torch.Tensor | None = None,
    ) -> tuple[N04State, Innovation | None]:
        """以新测量修正状态。

        首版规则：``stable = (1 - g_update) * pred + g_update * f_t``，并维护独立 change
        cell。无历史区域直接用有效当前测量初始化。

        ``measurement_valid`` 是**测量侧**的有效性（遮挡/缺失像素为 0）：没有测量支持的
        像素既不能推动更新，也不能被当成新证据初始化，只能保留传播预测。与 ``validity``
        区分：后者是传播（warp）的越界/遮挡有效性。
        """
        pred_list = predicted if isinstance(predicted, list) else [predicted]
        meas_list = measured if isinstance(measured, list) else [measured]
        pred0, meas0 = pred_list[0], meas_list[0]
        b, c, h, w = meas0.shape
        avail = availability
        if torch.is_tensor(avail):
            avail = avail.reshape(b, 1, 1, 1).to(meas0.dtype)
        else:
            avail = torch.full(
                (b, 1, 1, 1), 1.0 if avail else 0.0, device=meas0.device, dtype=meas0.dtype
            )
        if measurement_valid is not None:
            mask = measurement_valid.reshape(b, 1, h, w).to(meas0.dtype)
            avail = avail * (mask > 0.5).to(meas0.dtype)
        if saturation is None:
            saturation = torch.zeros(b, 1, h, w, device=meas0.device, dtype=meas0.dtype)
        else:
            saturation = saturation.reshape(b, 1, h, w)
        avail = avail * (1.0 - saturation).clamp(0, 1)
        reference = pred0
        innovation_validity = validity.detach()
        if self.config.model.innovation_reference == "sensor_history":
            if linear_predicted is None:
                reference = torch.zeros_like(meas0)
                innovation_validity = torch.zeros_like(validity)
            else:
                observation_prediction = linear_predicted
                if self.config.model.match_observation_support:
                    if current_linear_support is None:
                        raise ValueError(
                            "matched observation support requires current sensor support"
                        )
                    observation_prediction = observation_prediction * current_linear_support
                reference = self.predict_observation(observation_prediction)
                if linear_validity is not None:
                    support = measurement_from_linear_planes(
                        linear_validity, self.config.input_mode
                    ).amin(1, keepdim=True)
                    support = F.adaptive_avg_pool2d(support, (h, w))
                    innovation_validity = innovation_validity * support
        innovation = self.decomposer(
            current=meas0,
            predicted=reference,
            validity=innovation_validity,
            uncertainty=uncertainty,
            saturation=saturation.detach(),
            linear_current=linear_current,
            linear_predicted=linear_predicted,
            linear_validity=linear_validity,
        )
        innovation.diagnostics["innovation_reference"] = self.config.model.innovation_reference
        #: 无测量时（new_measurement=False）只传播不更新。
        g_update = innovation.g_update * avail
        new_stable = (1.0 - g_update) * pred0 + g_update * meas0
        #: 变化 cell：只写“自运动解释不了、且空间连贯”的残余，且不被稳定分支抹平。
        #: 另有 ``change_decay`` 衰减：没有持续证据支持的变化会退掉，否则一次瞬时
        #: 残差会被永久保留，物体走后仍报“变化”。衰减与写入都由测量证据驱动。
        #: 上一状态的变化 cell：**无条件**取出。关闭变化分支时下面的多尺度循环仍要
        #: 引用它（此前只在 change_branch=True 分支里赋值，导致消融配置直接崩）。
        old_change = state.change if state.change is not None else torch.zeros_like(new_stable)
        if self.config.model.change_branch:
            #: 变化写入同样只能来自有测量支持的区域：否则被遮挡像素的残差会被写进
            #: change cell，等于用不存在的内容制造“外部变化”。
            external_support = (
                innovation.propagation_validity
                if self.config.model.innovation_reference == "sensor_history"
                else 1.0
            )
            change_delta = innovation.gate_change * innovation.residual * avail * external_support
            innovation.external_evidence = change_delta
            old_change0 = old_change[0] if isinstance(old_change, list) else old_change
            if backward_grid is not None:
                old_change0 = (
                    F.grid_sample(
                        old_change0, backward_grid, padding_mode="zeros", align_corners=False
                    )
                    * validity
                )
            decay = torch.sigmoid(self.change_decay)
            new_change0 = decay * old_change0 + torch.tanh(self.change_write) * change_delta
        else:
            new_change0 = torch.zeros_like(new_stable)
        #: 没有历史的区域用**有效**当前测量直接初始化（不把 pred=0 当成历史证据）；
        #: 没有新测量（``new_measurement=False``）时不写入，避免把重复帧当成新证据。
        init_mask = (1.0 - validity) * avail
        new_stable = new_stable * (1.0 - init_mask) + meas0 * init_mask
        new_change0 = new_change0 * (1.0 - init_mask)
        fields = [new_stable]
        changes = [new_change0]
        for i in range(1, len(pred_list)):
            fields.append(pred_list[i])
            changes.append(
                old_change[i]
                if isinstance(old_change, list) and i < len(old_change)
                else torch.zeros_like(pred_list[i])
            )
        #: 组合 block：多尺度特征在同一套规则下混合，不改变 predict/correct 语义。
        #: 送进 block 的测量必须按可用性（含遮挡）屏蔽：否则被遮挡像素的内容会绕过
        #: 更新门控直接从 block 泄漏进状态。
        block_measurement = meas0 * avail
        for block in self.blocks:
            mixed = block(fields[0], changes[0], block_measurement)
            fields[0] = torch.where(avail > 0, mixed, fields[0])
        for i in range(1, len(fields)):
            fields[i] = self.downsample[i - 1](fields[i - 1])
            changes[i] = F.adaptive_avg_pool2d(changes[0], fields[i].shape[-2:])
        support = torch.maximum(validity, avail)
        new_state = N04State(
            stable=fields if isinstance(state.stable, list) else fields[0],
            change=changes if isinstance(state.stable, list) else changes[0],
            prediction=predicted,
            warp_validity=support,
            history_valid=support.flatten(1).any(-1),
            available=torch.as_tensor(availability, device=meas0.device).bool().expand(b),
            uncertainty=uncertainty,
            decomposition={
                "gate_ego": innovation.gate_ego,
                "gate_change": innovation.gate_change,
                "gate_uncertain": innovation.gate_uncertain,
                "unknown": innovation.unknown,
                "g_update": g_update,
            },
            illumination=innovation.illumination_gain,
            measured=measured,
            linear_measurement=state.linear_measurement,
            linear_validity=state.linear_validity,
            forward_flow=state.forward_flow,
            camera_id=state.camera_id,
            frame_id=state.frame_id,
            step_index=state.step_index,
            timestamp=state.timestamp,
            time_unit=state.time_unit,
            age=(state.age + avail.flatten(1).any(-1).long()) if state.age is not None else None,
            updates=state.updates + int(bool(avail.any())),
            observed_frame=state.observed_frame,
            extra=dict(state.extra),
        )
        return new_state, innovation

    # -- 一步 ---------------------------------------------------------------------

    def encode_measurement(self, values: torch.Tensor) -> torch.Tensor:
        """把测量编码成特征场（不修改编码器的状态缓存）。"""
        return self.stem(values)

    def predict_observation(self, propagated_linear_history: torch.Tensor) -> torch.Tensor:
        """Predict observation features from propagated past sensor samples only.

        Task-state block corrections do not enter this reference. The original current
        measurement and task-state prediction remain separate update/readout inputs.
        """
        values = measurement_from_linear_planes(propagated_linear_history, self.config.input_mode)
        return self.stem(values)

    def step(
        self,
        values: torch.Tensor,
        context: EgoMotionContext,
        state: N04State,
        *,
        camera_id: str | None = None,
        frame_key: Any = None,
        new_measurement: bool = True,
        saturation: torch.Tensor | None = None,
        measurement_valid: torch.Tensor | None = None,
        cached_only: bool = False,
    ) -> StepOutput:
        """逐帧调用，与在线执行完全一致。

        ``frame_key`` 非空时同一次调用被去重：同一帧的多次调用（例如 10 次去噪迭代
        反复读 prefix）只更新一次状态并复用缓存输出。

        ``saturation`` / ``measurement_valid`` 是**测量侧**的逐像素标记（饱和、遮挡、
        缺失）。不传时饱和由测量值推断、有效性视为全 1：调用方若确实知道遮挡，必须传进来，
        否则被遮挡像素会被当成新证据（此前合成遮挡条件就是这样被静默忽略的）。
        """
        key = (camera_id or state.camera_id, frame_key) if frame_key is not None else None
        if key is not None and key in self._cache:
            cached = self._cache[key]
            return StepOutput(
                representation=cached.representation,
                state=state,
                validity=cached.validity,
                innovation=cached.innovation,
                diagnostics={**cached.diagnostics, "cache_hit": True},
                updated=False,
                cache_key=frame_key,
            )
        if cached_only:
            raise KeyError(f"缓存中没有帧 {key!r}")
        sat_raw = saturation if saturation is not None else measurement_saturation(values)
        raw_valid = torch.ones_like(values[:, :1])
        if measurement_valid is not None:
            raw_valid = raw_valid * measurement_valid.to(values)
        available = torch.as_tensor(new_measurement, device=values.device).expand(values.shape[0])
        raw_valid = raw_valid * (1 - sat_raw) * available.reshape(-1, 1, 1, 1)
        linear_current, linear_support = linear_measurement_planes(
            values, raw_valid, self.config.input_mode
        )
        if measurement_valid is not None:
            values = values * measurement_valid.to(values)
        if saturation is not None:
            values = values * (1.0 - saturation.to(values))
        available = torch.as_tensor(new_measurement, device=values.device).expand(values.shape[0])
        values = values * available.reshape(-1, 1, 1, 1).to(values)
        measured = self.stem(values)
        pred, validity, uncertainty, propagation = self.predict_before_observe(state, context)
        linear_predicted = prior_support = None
        if state.linear_measurement is not None:
            linear_predicted, prior_support = warp_linear_measurement(
                state.linear_measurement,
                state.linear_validity,
                propagation.backward_grid,
                self.config.input_mode,
            )
        state = N04State(
            stable=state.stable,
            change=state.change,
            prediction=state.prediction,
            warp_validity=state.warp_validity,
            history_valid=state.history_valid,
            available=torch.as_tensor(new_measurement, device=measured.device)
            .bool()
            .expand(measured.shape[0]),
            uncertainty=state.uncertainty,
            decomposition=state.decomposition,
            illumination=state.illumination,
            measured=state.measured,
            linear_measurement=state.linear_measurement,
            linear_validity=state.linear_validity,
            forward_flow=getattr(propagation, "forward_flow", None),
            camera_id=camera_id or state.camera_id,
            frame_id=state.frame_id,
            step_index=state.step_index,
            timestamp=state.timestamp,
            time_unit=state.time_unit,
            age=state.age,
            updates=state.updates,
            observed_frame=state.observed_frame,
            extra=state.extra,
        )
        sat = sat_raw
        if sat.shape[-2:] != measured.shape[-2:]:
            sat = F.adaptive_max_pool2d(sat, measured.shape[-2:])
        if measurement_valid is not None and measurement_valid.shape[-2:] != measured.shape[-2:]:
            measurement_valid = F.adaptive_avg_pool2d(measurement_valid, measured.shape[-2:])
        new_state, innovation = self.correct_after_observe(
            state,
            pred,
            measured,
            validity,
            uncertainty,
            saturation=sat,
            availability=new_measurement,
            measurement_valid=measurement_valid,
            backward_grid=propagation.backward_grid,
            linear_current=linear_current,
            linear_predicted=linear_predicted,
            linear_validity=linear_support * prior_support if prior_support is not None else None,
            current_linear_support=linear_support,
        )
        # New evidence replaces supported pixels; otherwise keep propagated history.
        if linear_predicted is None:
            new_state.linear_measurement = linear_current
            new_state.linear_validity = linear_support
        else:
            retained_support = prior_support * (1 - linear_support)
            total_support = linear_support + retained_support
            new_state.linear_measurement = (
                (linear_current * linear_support + linear_predicted * retained_support)
                / total_support.clamp_min(1e-6)
            ).detach()
            new_state.linear_validity = total_support.detach()
        new_state.forward_flow = getattr(propagation, "forward_flow", None)
        representation = torch.cat(
            (
                _first(new_state.stable),
                _first(new_state.change)
                if new_state.change is not None
                else torch.zeros_like(_first(new_state.stable)),
                measured * new_state.warp_validity * available.reshape(-1, 1, 1, 1),
            ),
            dim=1,
        )
        validity_full = new_state.warp_validity
        out = StepOutput(
            representation=representation,
            state=new_state,
            validity=validity_full,
            innovation=innovation,
            diagnostics={
                "geometry_mode": getattr(propagation, "geometry_mode", "none"),
                "motion_available": getattr(propagation, "motion_available", False),
                **getattr(propagation, "diagnostics", {}),
                **({} if innovation is None else innovation.diagnostics),
                "cache_hit": False,
            },
            updated=bool(available.any()),
            cache_key=key,
        )
        if key is not None:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != key[0]}
            self._cache[key] = out
        return out

    def readout_tokens(
        self,
        output: StepOutput,
        token_budget: int | None = None,
    ) -> TokenReadout:
        """把表示读成 π0.5 视觉前缀形状的 token（``values`` / ``mask`` 两个命名域）。"""
        tokens, mask = self.readout(output.representation, output.validity)
        if token_budget is not None and token_budget < tokens.shape[1]:
            tokens = tokens[:, :token_budget]
            mask = mask[:, :token_budget]
        return TokenReadout(values=tokens, mask=mask)


def _first(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    return fuse_scales(value)


def fuse_scales(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Expose lower-scale fields to losses and readout, rather than storing unused tensors."""
    if not isinstance(value, list):
        return value
    top = value[0]
    if len(value) == 1:
        return top
    return top + 0.25 * sum(
        F.interpolate(field, size=top.shape[-2:], mode="bilinear", align_corners=False)
        for field in value[1:]
    ) / (len(value) - 1)


def _motion_dim(config: N04Config) -> int | None:
    """运动上下文向量维度；``None`` 表示运行期由实际上下文推断。"""
    return None if config.model.motion_dim is None else int(config.model.motion_dim)
