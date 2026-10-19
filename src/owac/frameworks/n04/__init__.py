"""N04 · 自运动预测与外部变化分离编码器。

研究问题：利用实际可得的本体状态、已执行运动与测量坐标预测自身运动造成的观测
变化，再以新测量修正状态，并保留对外部真实变化的敏感性。

本包只实现 N04 方向的机制与它需要的必要公共层，不把 N01/N02 的机制串进来。
"""

from .config import N04Config, N04GeometryConfig, N04ModelConfig, N04TrainConfig
from .ego import (
    CalibratedGeometryPropagator,
    LearnedProprioPropagator,
    PropagationResult,
    build_propagator,
)
from .geometry import (
    AffineFlow,
    FlowToAffineFit,
    GridSpec,
    TranslationFlow,
    WarpResult,
    backward_grid_from_affine,
    backward_grid_from_translation,
    compose_affine,
    forward_project_points,
    forward_splat,
    homography_from_rt,
    invert_affine,
    invert_rigid_2d,
    pixel_to_normalized,
    project_points,
    unproject_pixels,
    warp_feature_field,
)
from .innovation import Innovation, InnovationDecomposer
from .model import N04Encoder, N04Readout
from .state import N04State, new_state

__all__ = [
    "AffineFlow",
    "CalibratedGeometryPropagator",
    "FlowToAffineFit",
    "GridSpec",
    "Innovation",
    "InnovationDecomposer",
    "LearnedProprioPropagator",
    "N04Config",
    "N04Encoder",
    "N04GeometryConfig",
    "N04ModelConfig",
    "N04Readout",
    "N04State",
    "N04TrainConfig",
    "PropagationResult",
    "TranslationFlow",
    "WarpResult",
    "backward_grid_from_affine",
    "backward_grid_from_translation",
    "build_propagator",
    "compose_affine",
    "forward_project_points",
    "forward_splat",
    "homography_from_rt",
    "invert_affine",
    "invert_rigid_2d",
    "new_state",
    "pixel_to_normalized",
    "project_points",
    "unproject_pixels",
    "warp_feature_field",
]
