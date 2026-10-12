"""N04 合成二维可控测量场。

独立控制：相机平移/旋转、目标运动、全局照明、噪声、遮挡、标定误差，再经 CFA 采样得到
观测。生成器同时给出各变化的**真值标签**，供训练（只在与真值对应的 loss 上）与评测
（分解准确率、定位误差）使用。

信息边界：encoder 只读取配置明确声明的测量、本体及标定。本模块另外提供
``oracle_depth`` / ``object_mask`` 等仅供**诊断**的量，调用方必须显式区分，不得把它们
当作在线输入放进正常策略。

CFA：``bayer_cfa`` 模式按 2x2 排列从**连续场景**直接采样（每个像素只保留其滤色片上
的读数），不是把 RGB 重新马赛克后冒充原生 RAW；``rawvla_rgb3`` 模式输出三通道
HDR 派生测量。两条路径共用同一场景与运动学，便于对照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .config import CFA_PATTERNS

__all__ = ["SceneConfig", "SyntheticFrame", "SyntheticSequenceGenerator", "cfa_plane_offsets"]

#: 背景画布只取决于 (尺寸, 种子, 背景电平)：跨生成器实例复用，避免每步重复生成。
_BACKGROUND_CACHE: dict[tuple[int, int, int, float], torch.Tensor] = {}

CFA_PLANE_OFFSETS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "RGGB": (("R", 0, 0), ("Gr", 0, 1), ("Gb", 1, 0), ("B", 1, 1)),
    "BGGR": (("B", 0, 0), ("Gb", 0, 1), ("Gr", 1, 0), ("R", 1, 1)),
    "GRBG": (("Gr", 0, 0), ("R", 0, 1), ("B", 1, 0), ("Gb", 1, 1)),
    "GBRG": (("Gb", 0, 0), ("B", 0, 1), ("R", 1, 0), ("Gr", 1, 1)),
}


def cfa_plane_offsets(pattern: str) -> tuple[tuple[str, int, int], ...]:
    """返回 CFA 四个子平面的 (滤色片, 行偏移, 列偏移)，顺序固定为 (00, 01, 10, 11)。"""
    if pattern not in CFA_PATTERNS:
        raise ValueError(f"未知 CFA 排列 {pattern!r}")
    return CFA_PLANE_OFFSETS[pattern]


@dataclass
class SceneConfig:
    """一个可控场景的完整参数。"""

    height: int = 64
    width: int = 64
    #: 场景画布比视野大，便于让相机在画布上平移而不立即越界。
    canvas_margin: int = 24
    #: 相机平移速度范围（像素/帧）与旋转速度范围（度/帧）。
    camera_translation: tuple[float, float] = 1.5
    camera_rotation: tuple[float, float] = 0.0
    #: 目标运动速度范围（像素/帧）。目标沿一条直线匀速运动。
    object_velocity: tuple[float, float] = 0.0
    #: 全局照明增益（1.0 = 不变）。可给 (lo, hi) 让每帧随机采样。
    illumination: float | tuple[float, float] = 1.0
    #: 高斯读出噪声标准差（在 0..1 线性强度域）。
    noise_sigma: float = 0.005
    #: 遮挡：在被遮挡区域加一段固定图案（模拟前景挡板）。
    occluder: bool = False
    occluder_rect: tuple[int, int, int, int] = (20, 20, 40, 40)
    #: 标定误差：相机平移/旋转的已知量被错报的比例（1.05 = 报大 5%）。
    calibration_scale: float = 1.0
    calibration_offset: tuple[float, float] = (0.0, 0.0)
    #: 动作延迟（帧）：上报给 encoder 的动作/本体比实际观测滞后多少帧。
    action_delay: int = 0
    #: 输入模式。
    input_mode: Literal["rawvla_rgb3", "bayer_cfa"] = "rawvla_rgb3"
    cfa_pattern: str = "RGGB"
    #: 目标数量与颜色（颜色用于“颜色决定目标选择”的反例）。
    object_colors: tuple[tuple[float, float, float], ...] = ((1.0, 0.25, 0.2),)
    object_radius: float = 6.0
    #: 背景结构：低频纹理，避免全黑背景下增益不可识别。
    background_level: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if self.height % 2 or self.width % 2:
            raise ValueError("合成测量网格的边长必须是偶数（CFA 2x2 需要）")
        if self.input_mode == "bayer_cfa" and self.cfa_pattern not in CFA_PATTERNS:
            raise ValueError(f"未知 CFA 排列 {self.cfa_pattern!r}")


@dataclass
class SyntheticFrame:
    """一帧的全部可用量与真值。

    ``measured``/``valid_mask``/``saturated_mask``/``proprio``/``action`` 是在线可用量；
    其余带 ``oracle_`` 前缀或明确命名的字段**仅供诊断与监督**，不得进入在线输入。
    """

    measured: torch.Tensor  # [B,C,H,W]
    valid_mask: torch.Tensor  # [B,1,H,W]
    saturated_mask: torch.Tensor  # [B,1,H,W]
    #: 本体状态与**已执行**动作（相机自身运动，非未来监督）。
    proprio: torch.Tensor  # [B,D]
    action: torch.Tensor  # [B,A]
    #: 仅诊断/监督：相机与目标的真实二维运动。
    true_camera_translation: torch.Tensor  # [B,2]
    true_object_translation: torch.Tensor  # [B,2]
    true_camera_rotation: torch.Tensor  # [B]
    true_illumination: torch.Tensor  # [B]
    #: 仅诊断：无噪声、无照明的干净观测与目标掩码。
    oracle_clean: torch.Tensor  # [B,C,H,W]
    object_mask: torch.Tensor  # [B,1,H,W]
    #: 相机自运动造成的真值 forward flow（特征像素单位），用于 L_ego_warp。
    oracle_camera_flow: torch.Tensor  # [B,H,W,2]
    #: 场景类别标签，仅用于评测分组。
    condition: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SyntheticSequenceGenerator:
    """可控二维测量场生成器。

    场景 = 低频背景 + 若干彩色圆盘。每帧的观测由相机位姿（平移 + 绕图像中心的旋转）
    与目标运动共同决定。目标运动与相机运动**独立**控制，因此可以分别生成
    “相机动/物体不动”“相机不动/物体动”“二者都动”“仅照明变化”等条件。
    """

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or SceneConfig()
        self.canvas_h = self.config.height + 2 * self.config.canvas_margin
        self.canvas_w = self.config.width + 2 * self.config.canvas_margin
        self._background = self._make_background()

    # -- 场景构造 -----------------------------------------------------------------

    def _make_background(self) -> torch.Tensor:
        """低频背景纹理（在画布上一次性生成，之后各帧共用）。"""
        key = (self.canvas_h, self.canvas_w, self.config.seed, self.config.background_level)
        cached = _BACKGROUND_CACHE.get(key)
        if cached is not None:
            return cached
        gen = torch.Generator().manual_seed(self.config.seed)
        low_h = max(2, self.canvas_h // 16)
        low_w = max(2, self.canvas_w // 16)
        low = torch.rand(1, 3, low_h, low_w, generator=gen)
        bg = torch.nn.functional.interpolate(
            low, size=(self.canvas_h, self.canvas_w), mode="bilinear", align_corners=False
        )
        out = bg * self.config.background_level
        _BACKGROUND_CACHE[key] = out
        return out

    def _paint_objects(self, object_centers: torch.Tensor) -> torch.Tensor:
        """在画布上绘制目标圆盘，返回 [B,3,canvas_h,canvas_w]。"""
        b = object_centers.shape[0]
        cfg = self.config
        canvas = self._background.expand(b, -1, -1, -1).clone()
        ys = torch.arange(self.canvas_h, dtype=torch.float32).reshape(1, -1, 1)
        xs = torch.arange(self.canvas_w, dtype=torch.float32).reshape(1, 1, -1)
        for i, color in enumerate(cfg.object_colors):
            if i >= object_centers.shape[1]:
                break
            cy = object_centers[:, i, 1].reshape(b, 1, 1)
            cx = object_centers[:, i, 0].reshape(b, 1, 1)
            disc = ((ys - cy) ** 2 + (xs - cx) ** 2 <= cfg.object_radius**2).to(canvas.dtype)
            for ch in range(3):
                canvas[:, ch] = canvas[:, ch] * (1.0 - disc) + color[ch] * disc
        return canvas

    def _object_mask(self, object_centers: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """目标在**视野**内的掩码（仅诊断/监督用）。"""
        b = object_centers.shape[0]
        ys = torch.arange(h, dtype=torch.float32).reshape(1, -1, 1)
        xs = torch.arange(w, dtype=torch.float32).reshape(1, 1, -1)
        mask = torch.zeros(b, 1, h, w)
        for i in range(object_centers.shape[1]):
            cy = object_centers[:, i, 1].reshape(b, 1, 1)
            cx = object_centers[:, i, 0].reshape(b, 1, 1)
            disc = ((ys - cy) ** 2 + (xs - cx) ** 2 <= self.config.object_radius**2).to(mask.dtype)
            mask = mask + disc.unsqueeze(1)
        return mask.clamp(max=1.0)

    # -- 采样 ---------------------------------------------------------------------

    def _crop(self, canvas: torch.Tensor, top: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        """按（可微的）整数裁剪取视野。"""
        b, c = canvas.shape[0], canvas.shape[1]
        h, w = self.config.height, self.config.width
        out = torch.zeros(b, c, h, w, dtype=canvas.dtype, device=canvas.device)
        for bi in range(b):
            top_i = int(top[bi].item())
            left_i = int(left[bi].item())
            out[bi] = canvas[bi, :, top_i : top_i + h, left_i : left_i + w]
        return out

    def _rotate_batch(self, image: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        """绕图像中心旋转（用于相机 roll）。"""
        if bool((degrees.abs() < 1e-6).all()):
            return image
        b = image.shape[0]
        theta = degrees * torch.pi / 180.0
        cos, sin = torch.cos(theta), torch.sin(theta)
        h, w = image.shape[-2:]
        ys = torch.arange(h, dtype=image.dtype) - (h - 1) / 2.0
        xs = torch.arange(w, dtype=image.dtype) - (w - 1) / 2.0
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        #: 逆变换采样：目标坐标 -> 源坐标
        src_x = cos.reshape(b, 1, 1) * gx + sin.reshape(b, 1, 1) * gy
        src_y = -sin.reshape(b, 1, 1) * gx + cos.reshape(b, 1, 1) * gy
        norm_x = (src_x + (w - 1) / 2.0) / (w - 1) * 2 - 1
        norm_y = (src_y + (h - 1) / 2.0) / (h - 1) * 2 - 1
        grid = torch.stack((norm_x, norm_y), dim=-1)
        return torch.nn.functional.grid_sample(
            image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

    def _sample_cfa(self, rgb: torch.Tensor) -> torch.Tensor:
        """从连续三通道场景按 CFA 排列采样**单测量面**（每像素只保留其滤色片读数）。

        注意：这是从连续场景直接采样，不是把三通道图像重新马赛克。四个子平面各自保留
        自己的行列采样偏移与滤色片语义（见 ``cfa_plane_offsets``）。
        """
        cfg = self.config
        planes = cfa_plane_offsets(cfg.cfa_pattern)
        channel_of = {"R": 0, "Gr": 1, "Gb": 1, "B": 2}
        b, _, h, w = rgb.shape
        out = torch.zeros(b, 1, h, w, dtype=rgb.dtype, device=rgb.device)
        for name, dy, dx in planes:
            out[:, :, dy::2, dx::2] = rgb[:, channel_of[name] : channel_of[name] + 1, dy::2, dx::2]
        return out

    def frame_batch(
        self,
        *,
        camera_translation: torch.Tensor,
        camera_rotation: torch.Tensor,
        object_translation: torch.Tensor,
        illumination: torch.Tensor,
        base_object_centers: torch.Tensor,
        base_camera_offset: torch.Tensor,
        occlusion: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        camera_step: torch.Tensor | None = None,
    ) -> SyntheticFrame:
        """生成一个 batch 的观测与真值。

        ``camera_translation``/``object_translation`` 是**相对初始帧的累计量**（用于摆位），
        ``camera_step`` 是本帧相对上一帧的**位移增量**。``oracle_camera_flow`` 必须用增量：
        它是“传播算子要预测的单步 ego warp”，不是从序列开头累计的位移。未提供时退化为
        累计量，并在 ``metadata`` 中标注，避免误用。
        """
        cfg = self.config
        b = camera_translation.shape[0]
        margin = cfg.canvas_margin
        centers = base_object_centers + object_translation
        canvas = self._paint_objects(centers)
        top = (margin + base_camera_offset[:, 1] + camera_translation[:, 1]).round().long()
        left = (margin + base_camera_offset[:, 0] + camera_translation[:, 0]).round().long()
        top = top.clamp(0, self.canvas_h - cfg.height)
        left = left.clamp(0, self.canvas_w - cfg.width)
        view = self._crop(canvas, top, left)
        view = self._rotate_batch(view, camera_rotation)
        if cfg.occluder:
            y0, x0, y1, x1 = cfg.occluder_rect
            view[:, :, y0:y1, x0:x1] = 0.35
        clean_rgb = (view * illumination.reshape(b, 1, 1, 1)).clamp(0.0, 1.0)
        noisy = clean_rgb
        if cfg.noise_sigma > 0:
            noise = torch.randn(clean_rgb.shape, generator=generator) * cfg.noise_sigma
            noisy = (clean_rgb + noise).clamp(0.0, 1.0)
        if cfg.input_mode == "bayer_cfa":
            measured = self._sample_cfa(noisy)
        else:
            measured = noisy
        saturated = (noisy >= 0.999).to(noisy.dtype)[:, :1]
        valid = torch.ones(b, 1, cfg.height, cfg.width, dtype=noisy.dtype)
        if occlusion is not None:
            #: 遮挡区域没有测量支持：validity=0，而不是靠补零伪造内容。
            valid = valid * (1.0 - occlusion.reshape(b, 1, cfg.height, cfg.width))
        #: 目标掩码必须换算到**视野坐标**：画布坐标减去实际裁剪原点。
        view_centers = centers.clone()
        view_centers[:, :, 0] = view_centers[:, :, 0] - left.reshape(b, 1).to(centers.dtype)
        view_centers[:, :, 1] = view_centers[:, :, 1] - top.reshape(b, 1).to(centers.dtype)
        object_mask = self._object_mask(view_centers, cfg.height, cfg.width).clone()
        #: 真值单步 ego flow：相机本帧位移的负方向（内容在图像中向相反方向移动）。
        step = camera_translation if camera_step is None else camera_step
        flow = torch.zeros(b, cfg.height, cfg.width, 2, dtype=noisy.dtype)
        flow[..., 0] = -step[:, 0].reshape(b, 1, 1)
        flow[..., 1] = -step[:, 1].reshape(b, 1, 1)
        proprio = torch.cat(
            (
                camera_translation,
                camera_rotation.reshape(b, 1),
            ),
            dim=-1,
        )
        action = torch.cat((camera_translation, camera_rotation.reshape(b, 1)), dim=-1)
        return SyntheticFrame(
            measured=measured,
            valid_mask=valid,
            saturated_mask=saturated,
            proprio=proprio,
            action=action,
            true_camera_translation=camera_translation,
            true_object_translation=object_translation,
            true_camera_rotation=camera_rotation,
            true_illumination=illumination,
            oracle_clean=clean_rgb,
            object_mask=object_mask,
            oracle_camera_flow=flow,
            metadata={
                "input_mode": cfg.input_mode,
                "cfa_pattern": cfg.cfa_pattern,
                "oracle_flow_is_per_step": camera_step is not None,
            },
        )

    # -- 序列 ---------------------------------------------------------------------

    def sequence(
        self,
        *,
        batch: int = 1,
        length: int = 12,
        condition: str = "camera_only",
        seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> list[SyntheticFrame]:
        """生成一条可控条件的序列。

        ``condition`` 决定哪些量随时间变化：

        ``camera_only`` 相机平移/旋转，目标静止
        ``object_only`` 相机静止，目标运动
        ``both``        二者独立运动
        ``illumination`` 几何全静止，只有全局照明变化
        ``same_direction`` 相机与目标同向近同速（不可识别的困难条件）
        ``occlusion``   相机运动 + 周期遮挡，用于检查遮挡后重新暴露
        ``calibration_error`` 相机运动，但上报的标定被故意错报
        ``action_delay`` 相机运动，但上报的动作滞后若干帧
        ``exposure_step`` 相机运动 + 中途曝光突变与饱和
        """
        cfg = self.config
        gen = torch.Generator().manual_seed(cfg.seed if seed is None else seed)
        rng = torch.Generator().manual_seed((cfg.seed if seed is None else seed) + 1)
        cam_speed = _as_pair(cfg.camera_translation)
        rot_speed = _as_pair(cfg.camera_rotation)
        obj_speed = _as_pair(cfg.object_velocity)
        illum_range = _illumination_range(cfg.illumination)
        #: 场景声明里 ``(lo, hi)`` 是**每样本速率范围**（不是 x/y 分量范围）；
        #: 方向由 ``cam_dir``/``obj_dir`` 承担。
        cam_speed_lo, cam_speed_hi = float(cam_speed.min()), float(cam_speed.max())
        obj_speed_lo, obj_speed_hi = float(obj_speed.min()), float(obj_speed.max())

        #: 目标初始位置放在视野中心对应的画布坐标上。
        center_y = self.config.canvas_margin + self.config.height / 2.0
        center_x = self.config.canvas_margin + self.config.width / 2.0
        base_centers = (
            torch.tensor([[[center_x, center_y]]], dtype=torch.float32)
            .expand(batch, -1, -1)
            .clone()
        )
        base_offset = torch.zeros(batch, 2)
        rotation_sign = torch.rand(batch, generator=rng) * 2 - 1
        object_sign = torch.rand(batch, generator=rng) * 2 - 1
        #: 平动/旋转速率逐样本采样（[B]），方向由 cam_dir 承担。
        cam_rate = torch.empty(batch).uniform_(cam_speed_lo, cam_speed_hi, generator=rng)
        rot_rate = torch.empty(batch).uniform_(
            min(float(rot_speed[0]), float(rot_speed[1])),
            max(float(rot_speed[0]), float(rot_speed[1])),
            generator=rng,
        )
        obj_rate = torch.empty(batch).uniform_(obj_speed_lo, obj_speed_hi, generator=rng)
        cam_dir = torch.randn(batch, 2, generator=rng)
        cam_dir = cam_dir / cam_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        obj_dir = torch.randn(batch, 2, generator=rng)
        obj_dir = obj_dir / obj_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        camera_on = condition in {
            "camera_only",
            "both",
            "occlusion",
            "calibration_error",
            "action_delay",
            "exposure_step",
            "same_direction",
        }
        object_on = condition in {"object_only", "both", "same_direction"}
        illumination_on = condition == "illumination"
        frames: list[SyntheticFrame] = []
        previous_cam_t = torch.zeros(batch, 2)
        for t in range(length):
            step = float(t)
            cam_t = (
                cam_dir * cam_rate.reshape(batch, 1) * step if camera_on else torch.zeros(batch, 2)
            )
            cam_r = rotation_sign * rot_rate * step if camera_on else torch.zeros(batch)
            if condition == "same_direction":
                obj_t = cam_dir * obj_rate.reshape(batch, 1) * step * 0.95
            elif object_on:
                obj_t = obj_dir * obj_rate.reshape(batch, 1) * step * object_sign.reshape(batch, 1)
            else:
                obj_t = torch.zeros(batch, 2)
            illum = torch.ones(batch)
            if illumination_on:
                illum = torch.empty(batch).uniform_(*illum_range, generator=gen)
            elif condition == "exposure_step" and t >= length // 2:
                illum = torch.full((batch,), 2.5)
            #: 标定误差只影响“上报给 encoder 的运动”，真值观测不变。
            cam_reported = cam_t
            rot_reported = cam_r
            if condition == "calibration_error":
                cam_reported = cam_t * cfg.calibration_scale + torch.tensor(cfg.calibration_offset)
                rot_reported = cam_r * cfg.calibration_scale
            if condition == "action_delay" and cfg.action_delay > 0:
                cam_reported = cam_t - cam_dir * cam_rate.reshape(batch, 1) * cfg.action_delay
                rot_reported = cam_r - rotation_sign * rot_rate * cfg.action_delay
            occlusion = None
            if condition == "occlusion":
                occlusion = torch.zeros(batch, cfg.height, cfg.width)
                if (t // 3) % 2 == 1:
                    occlusion[:, cfg.height // 4 : cfg.height // 2, :] = 1.0
            frame = self.frame_batch(
                camera_translation=cam_t,
                camera_rotation=cam_r,
                object_translation=obj_t,
                illumination=illum,
                base_object_centers=base_centers,
                base_camera_offset=base_offset,
                occlusion=occlusion,
                generator=gen,
                camera_step=cam_t - previous_cam_t,
            )
            #: 本体与已执行动作都只含**自身运动**（相机平移+旋转）。此前 proprio 的最后一维
            #: 放的是物体位移，等于把外部运动泄漏进 ego 预测输入：模型可以走捷径而不是
            #: 从影像估计自运动，直接破坏“自运动 vs 外部变化”的可识别性。
            frame.proprio = torch.cat((cam_reported, rot_reported.reshape(batch, 1)), dim=-1)
            frame.action = torch.cat((cam_reported, rot_reported.reshape(batch, 1)), dim=-1)
            frame.condition = condition
            #: ``metadata`` 保持 JSON 可序列化：运动量以嵌套 list 记录，便于写出清单。
            frame.metadata.update(
                {
                    "t": t,
                    "camera_motion_reported": cam_reported.detach().tolist(),
                    "object_motion_true": obj_t.detach().tolist(),
                    "illumination_true": illum.detach().tolist(),
                }
            )
            frames.append(frame)
            previous_cam_t = cam_t
        return frames


def _as_pair(value) -> torch.Tensor:
    if isinstance(value, (int, float)):
        return torch.tensor([float(value), float(value)])
    if len(value) != 2:
        raise ValueError("速度范围必须是标量或 (lo, hi)")
    return torch.tensor([float(value[0]), float(value[1])])


def _illumination_range(value) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    return (float(value[0]), float(value[1]))
