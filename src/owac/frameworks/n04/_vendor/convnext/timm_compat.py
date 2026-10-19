"""timm 两个小工具在本仓库的独立等价实现。

上游 ``models/convnext.py`` 依赖 ``timm.models.layers.{DropPath, trunc_normal_}``。
N04 encoder 环境不引入 timm，本模块按这两个名字的公开语义独立实现：

* `trunc_normal_`：在 ``[a, b]`` 截断的正态初始化（逆 CDF 构造），签名与截断规则
  与 timm 一致（``a`` / ``b`` 作用在按 ``std`` 缩放前的标准正态上）。
* `DropPath`：stochastic depth，训练时按样本丢弃整条残差路径并可选按 keep 概率放大。

数值实现为本仓库自写，不复制 timm 源码；timm 为 Apache-2.0，此处不构成其再分发。
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import nn


def _no_grad_trunc_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    if mean - 2 * std > b or mean + 2 * std < a:
        warnings.warn(
            f"trunc_normal_ 的 mean={mean} / std={std} 与截断区间 [{a}, {b}] 几乎不重叠",
            stacklevel=2,
        )
    with torch.no_grad():
        lower = 0.5 * (1.0 + math.erf((a - mean) / (std * math.sqrt(2.0))))
        upper = 0.5 * (1.0 + math.erf((b - mean) / (std * math.sqrt(2.0))))
        tensor.uniform_(2.0 * lower - 1.0, 2.0 * upper - 1.0)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """把 ``tensor`` 原地填成 ``[a, b]`` 截断的正态样本并返回它。"""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    """按样本随机丢弃残差分支（stochastic depth）。

    ``drop_prob=0`` 或 ``self.training=False`` 时为恒等；否则每个 batch 元素独立采样
    ``keep_prob = 1 - drop_prob``，保留时按 ``1 / keep_prob`` 放大（``scale_by_keep``）。
    """

    def __init__(self, drop_prob: float | None = None, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob or 0.0)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 - nn.Module 约定
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:  # noqa: D102 - nn.Module 约定
        return f"drop_prob={self.drop_prob:g}, scale_by_keep={self.scale_by_keep}"
