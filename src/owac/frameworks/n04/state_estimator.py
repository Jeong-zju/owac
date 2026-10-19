"""N04 v1.1 持续支持与双模式 RKN 条件预测（契约 §4）。

每个相机维护 ``N=128`` 个持续支持。每个支持保存稳定 ID、当前 xy（原始像素）、
年龄、有效性、起始标记，以及两个动态模式各自的 RKN 状态。协方差沿用作者 RKN 的
结构化表示：每个模式 = ``2*obs_dim`` 维均值 + 三个 ``obs_dim`` 维对角块
``(upper, lower, side)``，不生成 512×512 稠密矩阵。

时间顺序（与作者 cell 的 ``forward`` 不同，不能把它的 update→predict 当当前 prior）：

.. code-block:: text

    t = 0:   state = reset(...)                      # 首帧覆盖网格出生（fresh）
             post  = update(state, y0, R0)           # 首个观测直接初始化，见 evidence_update
    t >= 1:  prior = predict(post, proprio_t)        # 只读 t-1 posterior 与截至 t 的本体
             state = advance_positions(prior, flow)   # xy_t = xy_{t-1} + flow(xy_{t-1})
             post  = update(state, y_t, R_t)          # 当前测量只在这一步进入
             state = birth(post, candidates, ...)     # 只在未被覆盖的新区域占用空槽

``predict`` 不接收当前 RAW、当前特征或当前 flow；本体条件按 Ac-RKN 的加性控制结构
注入：``m_j^- = RKN_predict_j(mix_j) + B(proprio)``，控制只进均值、不进协方差。
修订模式（下标 1）在 ``_predict`` 之后再叠加一个非负额外过程噪声，且加在可影响观测
预测的 upper 块上，否则模式似然无法区分两者。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from owac.baselines._vendor.rkn.cell import RKNCell

from .measurement_geometry import PixelPoints, in_bounds_xy, normalized_from_xy
from .rkn_banded import band_layout, predict_banded, predict_dense_bounded

__all__ = [
    "DEFAULT_TRANSITION_INIT",
    "SupportState",
    "StateEstimator",
    "birth_covariance",
    "covering_grid",
    "covering_radius_px",
    "flow_at_points",
    "occlusion_at_points",
    "require_finite",
    "stabilise_block_covariance",
]

# 行归一化模式转移矩阵的初始化（延续 0 / 修订 1）。
DEFAULT_TRANSITION_INIT = ((0.95, 0.05), (0.3, 0.7))
_COVARIANCE_FLOOR = 1e-8


def require_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """发现 NaN/Inf 时直接报出首个位置，不用 ``nan_to_num`` 掩盖。"""
    if not bool(torch.isfinite(tensor).all()):
        first = torch.nonzero(torch.isfinite(tensor).logical_not(), as_tuple=True)
        where = tuple(int(v[0]) for v in first)
        raise ValueError(f"{name} 出现 NaN/Inf，首个位置 {where}")
    return tensor


def stabilise_block_covariance(
    upper: torch.Tensor,
    lower: torch.Tensor,
    side: torch.Tensor,
    *,
    floor: float = _COVARIANCE_FLOOR,
):
    """把 RKN 对角块协方差限制在结构允许域内：``cu, cl >= floor`` 且 ``|cs| <= sqrt(cu*cl)``。

    这是契约 §7.4 的行列式条件在结构表示下的唯一必要保护；不做上限截断、不加对角抖动。
    """
    upper = upper.clamp_min(floor)
    lower = lower.clamp_min(floor)
    limit = (upper * lower).sqrt()
    return upper, lower, torch.clamp(side, -limit, limit)


def covering_grid(num_supports: int, sensor_size_hw: tuple[int, int]) -> torch.Tensor:
    """首帧覆盖网格：``[N,2]`` 原始像素，按行列最接近画幅宽高比的均匀分区取中心。"""
    height, width = int(sensor_size_hw[0]), int(sensor_size_hw[1])
    if num_supports <= 0 or height <= 0 or width <= 0:
        raise ValueError("num_supports 与画幅尺寸必须为正")
    rows, cols = _factor_pair(num_supports, aspect=height / width)
    ys = (torch.arange(rows, dtype=torch.float32) + 0.5) * height / rows
    xs = (torch.arange(cols, dtype=torch.float32) + 0.5) * width / cols
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1).contiguous()


def covering_radius_px(num_supports: int, sensor_size_hw: tuple[int, int]) -> float:
    """覆盖半径：半个最小网格间距。`birth` 用它判断候选点是否落在“新区域”。"""
    height, width = int(sensor_size_hw[0]), int(sensor_size_hw[1])
    rows, cols = _factor_pair(num_supports, aspect=height / width)
    return 0.5 * min(height / rows, width / cols)


def _factor_pair(count: int, *, aspect: float) -> tuple[int, int]:
    best: tuple[int, int] | None = None
    best_error = math.inf
    for rows in range(1, count + 1):
        if count % rows:
            continue
        cols = count // rows
        error = abs(rows / cols - aspect)
        if error < best_error:
            best, best_error = (rows, cols), error
    assert best is not None
    return best


def birth_covariance(obs_dim: int, *, device, dtype, initial_variance: float = 1.0):
    """出生时的结构化协方差：upper/lower 为初始方差，side 为 0。"""
    shape = (1, 1, 2, obs_dim)
    upper = torch.full(shape, float(initial_variance), device=device, dtype=dtype)
    return upper, upper.clone(), torch.zeros(shape, device=device, dtype=dtype)


def flow_at_points(flow_px: torch.Tensor, xy_px: torch.Tensor) -> torch.Tensor:
    """在 ``xy_px [B,N,2]``（原始像素）处双线性采样 ``flow_px [B,2,H,W]`` -> ``[B,N,2]``。

    采样约定与位置推进一致：``flow(previous_xy)`` 给出该支持从上一帧到当前帧的位移，
    即 ``xy_t = xy_{t-1} + flow_at_points(flow, xy_{t-1})``。flow 是总图像运动，
    这里不解释成相机或物体运动。
    """
    if flow_px.dim() != 4 or flow_px.shape[1] != 2:
        raise ValueError(f"flow_px 需要 [B,2,H,W]，实际 {tuple(flow_px.shape)}")
    if xy_px.shape[-1] != 2 or xy_px.shape[0] != flow_px.shape[0]:
        raise ValueError("xy_px 需要 [B,N,2] 且 batch 与 flow_px 一致")
    points = PixelPoints(xy=xy_px.to(dtype=torch.float32), unit="original_px")
    normalized = normalized_from_xy(points, tuple(flow_px.shape[-2:])).unsqueeze(1)
    sampled = F.grid_sample(
        flow_px.to(torch.float32),
        normalized,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return sampled.squeeze(2).permute(0, 2, 1).contiguous()


def occlusion_at_points(occlusion: torch.Tensor, xy_px: torch.Tensor) -> torch.Tensor:
    """在 ``xy_px`` 处最近邻采样遮挡/有效平面，返回 ``[B,N]`` bool（True = 该点被遮挡）。"""
    if occlusion.dim() == 4:
        if occlusion.shape[1] != 1:
            raise ValueError("遮挡平面需要 [B,H,W] 或 [B,1,H,W]")
        occlusion = occlusion[:, 0]
    if occlusion.dim() != 3 or xy_px.shape[0] != occlusion.shape[0]:
        raise ValueError("遮挡平面需要 [B,H,W] 且 batch 与 xy_px 一致")
    points = PixelPoints(xy=xy_px.to(dtype=torch.float32), unit="original_px")
    normalized = normalized_from_xy(points, tuple(occlusion.shape[-2:])).unsqueeze(1)
    sampled = F.grid_sample(
        occlusion.to(torch.float32).unsqueeze(1),
        normalized,
        mode="nearest",
        align_corners=False,
        padding_mode="border",
    )
    return sampled.squeeze(2)[:, 0] > 0.5


@dataclass(frozen=True)
class SupportState:
    """每相机一份的持续支持状态；所有张量共享 ``[B,N,...]`` 布局（B = batch×相机）。

    ``xy``          当前关联位置，原始像素 ``[B,N,2]``；
    ``ids``         稳定 ID ``[B,N]`` long，只在出生时改变；
    ``age``         出生后的记录帧数 ``[B,N]`` long；
    ``valid``       本帧该支持是否可以进入更新（越界/遮挡后为 False）；
    ``fresh``       已出生但还没有被任何观测初始化；
    ``mode_mean``   ``[B,N,2,2*obs_dim]``，前 obs_dim 维是可观测方向；
    ``mode_cov``    ``(upper, lower, side)``，各 ``[B,N,2,obs_dim]``；
    ``mode_weights`` ``[B,N,2]``，t-1 后验权重或当前 prior 权重。
    """

    xy: torch.Tensor
    ids: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor
    fresh: torch.Tensor
    mode_mean: torch.Tensor
    mode_cov: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    mode_weights: torch.Tensor
    sensor_size_hw: tuple[int, int]

    @property
    def batch_size(self) -> int:
        return int(self.xy.shape[0])

    @property
    def num_supports(self) -> int:
        return int(self.xy.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.mode_mean.shape[-1]) // 2


def _default_cell_config():
    config = RKNCell.get_default_config()
    config.finalize_modifying()
    return config


class StateEstimator(nn.Module):
    """双模式条件预测器：共享作者 RKN cell 的两个分量 + Ac-RKN 加性控制。"""

    def __init__(
        self,
        *,
        num_supports: int = 128,
        obs_dim: int = 256,
        proprio_dim: int = 16,
        control_hidden: int = 128,
        cell_config=None,
        revision_variance_init: float = 0.1,
        initial_variance: float = 1.0,
        dtype: torch.dtype = torch.float32,
        checkpoint_predictions: bool = True,
        prediction_backend: str = "banded",
        transition_init: str = "author",
        transition_norm_limit: float | None = None,
    ) -> None:
        super().__init__()
        if num_supports <= 0 or obs_dim <= 0 or proprio_dim <= 0:
            raise ValueError("num_supports / obs_dim / proprio_dim 必须为正")
        self.checkpoint_predictions = bool(checkpoint_predictions)
        if prediction_backend not in ("banded", "dense"):
            raise ValueError("prediction_backend must be banded or dense")
        self.prediction_backend = prediction_backend
        self.banded_kernel = predict_banded
        self.num_supports = int(num_supports)
        self.obs_dim = int(obs_dim)
        self.state_dim = 2 * self.obs_dim
        self.proprio_dim = int(proprio_dim)
        self.initial_variance = float(initial_variance)
        if transition_norm_limit is not None and not 0 < transition_norm_limit <= 1:
            raise ValueError("transition_norm_limit must be in (0,1]")
        self.transition_norm_limit = transition_norm_limit
        config = cell_config if cell_config is not None else _default_cell_config()
        self.cells = nn.ModuleList(
            [RKNCell(self.obs_dim, config, dtype=dtype), RKNCell(self.obs_dim, config, dtype=dtype)]
        )
        if transition_init not in ("author", "constant_velocity", "identity"):
            raise ValueError("transition_init must be author, constant_velocity or identity")
        self.transition_init = transition_init
        if transition_init == "constant_velocity":
            # The author's +/-0.2 off-diagonal blocks rotate a nonzero static feature
            # into the initially-zero hidden state. Here the hidden half starts as a
            # velocity: a constant observation has zero initial acceleration. Keep
            # the upper-right coupling learnable and retain every RKN parameter.
            with torch.no_grad():
                for cell in self.cells:
                    cell._tm_21_basis.zero_()
        if transition_init == "identity":
            with torch.no_grad():
                for cell in self.cells:
                    for basis in (
                        cell._tm_11_basis,
                        cell._tm_12_basis,
                        cell._tm_21_basis,
                        cell._tm_22_basis,
                    ):
                        basis.zero_()
        for name, tensor in zip(
            ("_band_lookup", "_band_columns", "_band_valid"),
            band_layout(self.cells[0]),
            strict=True,
        ):
            self.register_buffer(name, tensor, persistent=False)
        self.control = nn.Sequential(
            nn.Linear(self.proprio_dim, control_hidden),
            nn.GELU(),
            nn.Linear(control_hidden, control_hidden),
            nn.GELU(),
            nn.Linear(control_hidden, self.state_dim),
        )
        # 控制项从 0 开始：初始 prior 就是纯 RKN 预测，本体条件由训练学出来。
        nn.init.zeros_(self.control[-1].weight)
        nn.init.zeros_(self.control[-1].bias)
        # 修订模式的额外非负过程噪声（softplus 参数化，逐观测通道）。
        self.revision_q_raw = nn.Parameter(
            torch.full((self.obs_dim,), math.log(math.expm1(float(revision_variance_init))))
        )
        # 模式转移矩阵：softmax 参数化、行归一化，初始化约 [[0.95,0.05],[0.3,0.7]]。
        init = torch.tensor(DEFAULT_TRANSITION_INIT, dtype=dtype)
        self.mode_transition_logits = nn.Parameter(torch.log(init))

    # -- 参数化与常数 ---------------------------------------------------------------
    def mode_transition(self) -> torch.Tensor:
        """``[2,2]`` 行归一化转移矩阵 ``T[i,j] = P(mode_j at t | mode_i at t-1)``。"""
        return torch.softmax(self.mode_transition_logits, dim=-1)

    def revision_extra_q(self) -> torch.Tensor:
        """``[obs_dim]`` 非负额外过程噪声（softplus），只加到修订模式的 upper 块。"""
        return F.softplus(self.revision_q_raw)

    def mode_prior(self, weights: torch.Tensor) -> torch.Tensor:
        """``[B,N,2]`` 权重 -> 模式先验：``prior_j = Σ_i w_i T[i,j]``（只看旧权重与 T）。"""
        return torch.einsum("bni,ij->bnj", weights, self.mode_transition())

    def birth_weights(self) -> torch.Tensor:
        """出生权重 = 转移矩阵的平稳分布（detach，不把出生常量接进 T 的梯度）。"""
        transition = self.mode_transition().detach().to(dtype=torch.float32)
        p0 = float(transition[0, 0])
        p1 = float(transition[1, 1])
        denominator = 2.0 - p0 - p1
        if abs(denominator) < 1e-6:
            return torch.full((2,), 0.5, device=transition.device, dtype=torch.float32)
        pi0 = (1.0 - p1) / denominator
        return torch.stack((torch.as_tensor(pi0), torch.as_tensor(1.0 - pi0))).to(
            device=transition.device, dtype=torch.float32
        )

    # -- 生命周期 -------------------------------------------------------------------
    def reset(
        self,
        *,
        batch_size: int,
        sensor_size_hw: tuple[int, int],
        device: torch.device | None = None,
    ) -> SupportState:
        """首帧覆盖网格出生：所有支持 fresh、age 0、均值 0、协方差为出生值。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        if device is None:
            device = next(self.parameters()).device
        dtype = torch.float32
        size = (int(sensor_size_hw[0]), int(sensor_size_hw[1]))
        grid = covering_grid(self.num_supports, size).to(device=device, dtype=dtype)
        xy = grid.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        ids = (
            torch.arange(self.num_supports, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        upper, lower, side = birth_covariance(
            self.obs_dim, device=device, dtype=dtype, initial_variance=self.initial_variance
        )
        weights = self.birth_weights().to(device=device)
        return SupportState(
            xy=xy,
            ids=ids,
            age=torch.zeros(batch_size, self.num_supports, device=device, dtype=torch.long),
            valid=torch.ones(batch_size, self.num_supports, device=device, dtype=torch.bool),
            fresh=torch.ones(batch_size, self.num_supports, device=device, dtype=torch.bool),
            mode_mean=torch.zeros(
                batch_size, self.num_supports, 2, self.state_dim, device=device, dtype=dtype
            ),
            mode_cov=(
                upper.expand(batch_size, self.num_supports, -1, -1).clone(),
                lower.expand(batch_size, self.num_supports, -1, -1).clone(),
                side.expand(batch_size, self.num_supports, -1, -1).clone(),
            ),
            mode_weights=(
                weights.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, self.num_supports, -1)
                .contiguous()
            ),
            sensor_size_hw=size,
        )

    def advance_positions(
        self,
        state: SupportState,
        flow_px: torch.Tensor,
        *,
        occlusion: torch.Tensor | None = None,
    ) -> SupportState:
        """``xy_t = xy_{t-1} + flow_at_points(flow_px, xy_{t-1})``；越界/遮挡只置 ``valid=False``。

        flow 与遮挡都定义在上一帧坐标，遮挡在推进前的位置采样；状态分布、权重与 ID 不变
        （不按每帧置信度重排、不产生相机/物体标签）。``age`` 在每个新记录帧加一。
        """
        require_finite(flow_px, "flow_px")
        xy = state.xy + flow_at_points(flow_px, state.xy)
        require_finite(xy, "支持点 xy")
        in_bounds = in_bounds_xy(PixelPoints(xy=xy, unit="original_px"), state.sensor_size_hw)
        valid = in_bounds
        if occlusion is not None:
            valid = valid & ~occlusion_at_points(occlusion, state.xy)
        age = torch.where(state.fresh, state.age, state.age + 1)
        return replace(state, xy=xy, valid=valid, age=age)

    def birth(
        self,
        state: SupportState,
        candidate_xy: torch.Tensor,
        candidate_valid: torch.Tensor,
        *,
        radius_px: float,
    ) -> SupportState:
        """在未被覆盖的新区域出生：只占用越界空槽，按槽位顺序取候选，不重排已有支持。

        ``candidate_xy [B,M,2]`` 由调用方按“当前可见/新进入画幅”的区域给出；
        ``candidate_valid [B,M]`` 标出合法候选。候选点必须落在画幅内、且与所有仍在
        画幅内的支持距离都大于 ``radius_px``（即覆盖半径之外的新区域）。
        """
        if candidate_xy.dim() != 3 or candidate_xy.shape[-1] != 2:
            raise ValueError(f"candidate_xy 需要 [B,M,2]，实际 {tuple(candidate_xy.shape)}")
        if candidate_xy.shape[0] != state.batch_size:
            raise ValueError("candidate_xy 与 state 的 batch 不一致")
        if candidate_valid.shape != candidate_xy.shape[:2]:
            raise ValueError("candidate_valid 需要 [B,M]")
        xy = candidate_xy.to(device=state.xy.device, dtype=torch.float32)
        candidate_valid = candidate_valid.to(device=state.xy.device, dtype=torch.bool)
        live = in_bounds_xy(PixelPoints(xy=state.xy, unit="original_px"), state.sensor_size_hw)
        vacant = ~live
        distance = torch.cdist(xy, state.xy)
        distance = distance.masked_fill(~live[:, None, :].expand_as(distance), math.inf)
        uncovered = (distance > float(radius_px)).all(dim=-1)
        candidate_ok = candidate_valid & uncovered
        candidate_ok = candidate_ok & in_bounds_xy(
            PixelPoints(xy=xy, unit="original_px"), state.sensor_size_hw
        )

        new_xy = state.xy.clone()
        new_ids = state.ids.clone()
        new_age = state.age.clone()
        new_valid = state.valid.clone()
        new_fresh = state.fresh.clone()
        new_mean = state.mode_mean.clone()
        new_cov = [block.clone() for block in state.mode_cov]
        new_weights = state.mode_weights.clone()
        birth_upper, birth_lower, birth_side = birth_covariance(
            self.obs_dim,
            device=state.xy.device,
            dtype=state.mode_mean.dtype,
            initial_variance=self.initial_variance,
        )
        birth_weight = self.birth_weights().to(
            device=state.xy.device, dtype=state.mode_weights.dtype
        )
        base_ids = state.ids.max(dim=-1).values + 1
        for batch in range(state.batch_size):
            slots = torch.nonzero(vacant[batch], as_tuple=False).flatten()
            chosen = torch.nonzero(candidate_ok[batch], as_tuple=False).flatten()
            count = min(int(slots.numel()), int(chosen.numel()))
            if count == 0:
                continue
            slots, chosen = slots[:count], chosen[:count]
            new_xy[batch, slots] = xy[batch, chosen]
            new_ids[batch, slots] = base_ids[batch] + torch.arange(
                count, device=state.xy.device, dtype=state.ids.dtype
            )
            new_age[batch, slots] = 0
            new_valid[batch, slots] = True
            new_fresh[batch, slots] = True
            new_mean[batch, slots] = 0.0
            new_cov[0][batch, slots] = birth_upper.to(new_cov[0].dtype)
            new_cov[1][batch, slots] = birth_lower.to(new_cov[1].dtype)
            new_cov[2][batch, slots] = birth_side.to(new_cov[2].dtype)
            new_weights[batch, slots] = birth_weight.to(new_weights.dtype)
        return replace(
            state,
            xy=new_xy,
            ids=new_ids,
            age=new_age,
            valid=new_valid,
            fresh=new_fresh,
            mode_mean=new_mean,
            mode_cov=(new_cov[0], new_cov[1], new_cov[2]),
            mode_weights=new_weights,
        )

    # -- 条件预测 -------------------------------------------------------------------
    def predict(
        self,
        state: SupportState,
        proprio_context: torch.Tensor,
        *,
        proprio_valid: torch.Tensor | None = None,
    ) -> SupportState:
        """当前 prior：IMM 混合 -> 作者 RKN ``_predict`` -> 加性控制 -> 修订模式额外 Q。

        只读 ``state``（t-1 后验）与截至当前时刻的本体上下文；当前图像、当前特征、
        当前 flow 都不在参数里。fresh 支持还没有历史，原样保留（不被 predict 破坏）。
        """
        if proprio_context.dim() != 2 or proprio_context.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio_context 需要 [B,{self.proprio_dim}]，实际 {tuple(proprio_context.shape)}"
            )
        if proprio_context.shape[0] != state.batch_size:
            raise ValueError("proprio_context 与 state 的 batch 不一致")
        require_finite(proprio_context, "proprio_context")
        return self._predict_impl(state, proprio_context, proprio_valid=proprio_valid)

    def _predict_impl(self, state, proprio_context, *, proprio_valid=None):
        """Pure prediction math; can be compiled without online observation side effects."""
        batch, supports = state.batch_size, state.num_supports
        dtype = state.mode_mean.dtype

        transition = self.mode_transition().to(dtype=dtype)
        prior_weights = torch.einsum("bni,ij->bnj", state.mode_weights, transition)
        mixed_mean, mixed_cov = self._mix_components(state, transition)

        means = mixed_mean.reshape(-1, 2, self.state_dim)
        cov_blocks = [block.reshape(-1, 2, self.obs_dim) for block in mixed_cov]
        control = self.control(proprio_context.to(dtype=dtype))
        if proprio_valid is not None:
            if proprio_valid.shape != (batch,):
                raise ValueError("proprio_valid 需要 [B]")
            control = control * proprio_valid.to(dtype=dtype).unsqueeze(-1)
        control = (
            control[:, None, :]
            .expand(batch, supports, self.state_dim)
            .reshape(-1, 1, self.state_dim)
        )

        predicted_mean, predicted_blocks = [], [[], [], []]
        for index, cell in enumerate(self.cells):
            cov_in = [block[:, index] for block in cov_blocks]

            # RKN builds dense transition blocks per support. Saving every block for
            # every forecast roll exhausts memory; recompute only this pure cell math.
            # The online encoder, RAW measurement, support births and counters never rerun.
            def pure_predict(mean, upper, lower, side, cell=cell):
                if self.prediction_backend == "banded":
                    return self.banded_kernel(
                        cell,
                        mean,
                        upper,
                        lower,
                        side,
                        self._band_lookup,
                        self._band_columns,
                        self._band_valid,
                        max_operator_norm=self.transition_norm_limit,
                    )
                if self.transition_norm_limit is not None:
                    return predict_dense_bounded(
                        cell, mean, upper, lower, side, self.transition_norm_limit
                    )
                result_mean, result_cov = cell._predict(mean, [upper, lower, side])
                return result_mean, *result_cov

            if self.checkpoint_predictions and torch.is_grad_enabled():
                result = checkpoint(pure_predict, means[:, index], *cov_in, use_reentrant=False)
                mean_j, cov_j = result[0], list(result[1:])
            else:
                result = pure_predict(means[:, index], *cov_in)
                mean_j, cov_j = result[0], list(result[1:])
            mean_j = mean_j + control[:, 0]
            if index == 1:
                # 额外非负 Q 只加在 upper（可观测）块：等价于在可影响观测的方向上加过程噪声。
                cov_j = [cov_j[0] + self.revision_extra_q().to(dtype=dtype), cov_j[1], cov_j[2]]
            stabilised = stabilise_block_covariance(*cov_j)
            predicted_mean.append(mean_j)
            for slot, block in enumerate(stabilised):
                predicted_blocks[slot].append(block)

        shape = (batch, supports, 2, self.state_dim)
        fresh_mean = state.mode_mean
        next_mean = torch.where(
            state.fresh[..., None, None],
            fresh_mean,
            torch.stack(predicted_mean, dim=1).reshape(shape),
        )
        next_cov = []
        for slot in range(3):
            block = torch.stack(predicted_blocks[slot], dim=1)
            block = block.reshape(batch, supports, 2, self.obs_dim)
            next_cov.append(torch.where(state.fresh[..., None, None], state.mode_cov[slot], block))
        return replace(
            state,
            mode_mean=next_mean,
            mode_cov=(next_cov[0], next_cov[1], next_cov[2]),
            mode_weights=torch.where(state.fresh[..., None], state.mode_weights, prior_weights),
        )

    def _mix_components(self, state: SupportState, transition: torch.Tensor):
        """IMM 条件混合，按 RKN 对角块结构近似，保留分量间均值分歧。

        稠密形式的混合协方差是 ``Σ_i μ_{i|j} (P_i + Δ_i Δ_i^T)``；本实现保留
        ``(upper,upper)``、``(lower,lower)``、``(upper,lower)`` 三个对角块，
        ``Δ`` 项照常进入，因此两个分量不一致时协方差会相应变大，而不是被平均掉。
        """
        weights = state.mode_weights.to(dtype=torch.float32)
        transition = transition.to(dtype=torch.float32)
        mix = weights[:, :, :, None] * transition[None, None, :, :]  # [B,N,i,j]
        prior = mix.sum(dim=2)  # [B,N,j]
        mix = mix / prior[:, :, None, :].clamp_min(torch.finfo(torch.float32).tiny)
        mean = state.mode_mean.to(dtype=torch.float32)
        mixed_mean = torch.einsum("bnij,bnid->bnjd", mix, mean)
        delta = mean[:, :, :, None, :] - mixed_mean[:, :, None, :, :]  # [B,N,i,j,D]
        upper = state.mode_cov[0].to(dtype=torch.float32)
        lower = state.mode_cov[1].to(dtype=torch.float32)
        side = state.mode_cov[2].to(dtype=torch.float32)
        weight = mix.unsqueeze(-1)  # [B,N,i,j,1]
        delta_upper = delta[..., : self.obs_dim]
        delta_lower = delta[..., self.obs_dim :]
        mixed_upper = (weight * (upper[:, :, :, None, :] + delta_upper.square())).sum(dim=2)
        mixed_lower = (weight * (lower[:, :, :, None, :] + delta_lower.square())).sum(dim=2)
        mixed_side = (weight * (side[:, :, :, None, :] + delta_upper * delta_lower)).sum(dim=2)
        stabilised = stabilise_block_covariance(mixed_upper, mixed_lower, mixed_side)
        return mixed_mean.to(device=state.xy.device), tuple(
            block.contiguous() for block in stabilised
        )

    # -- 便捷入口 -------------------------------------------------------------------
    def step(
        self,
        state: SupportState,
        proprio_context: torch.Tensor,
        flow_px: torch.Tensor,
        *,
        occlusion: torch.Tensor | None = None,
        proprio_valid: torch.Tensor | None = None,
    ) -> SupportState:
        """``predict`` + ``advance_positions``（t>=1 使用；当前测量随后由 update 注入）。"""
        prior = self.predict(state, proprio_context, proprio_valid=proprio_valid)
        return self.advance_positions(prior, flow_px, occlusion=occlusion)
