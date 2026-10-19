"""N04 v1.1 读出：OpenFlamingo 完整 Perceiver Resampler + 每相机媒体 token 组装。

上游登记（结构与来源，不是逐行复制）
------------------------------------

* 仓库：``https://github.com/mlfoundations/open_flamingo``
* commit：``655f693fbfa04cd6e9a987d960654624d48917cf``
* 许可：上游仓库为 MIT；本文件不复制上游文件正文，只按其结构重写。
* 结构：``PerceiverResampler`` = 可学习 ``latents``（``num_latents × dim``）+ ``depth`` 个
  ``PerceiverAttentionBlock``（内含 ``PerceiverAttention`` 与 ``FeedForward``）+ 末尾
  ``LayerNorm``；``PerceiverAttention`` 自带 ``norm_media`` / ``norm_latents``、``to_q`` /
  ``to_kv`` / ``to_out``，``inner_dim = dim_head * heads``；``FeedForward`` 为
  ``LayerNorm -> Linear(dim, dim*mult) -> GELU -> Linear(-> dim)``。每个 block 都是
  ``x = x + attn(x, media)`` 与 ``x = x + ff(x)`` 的残差形式。
* 本项目改动（登记在案）：① ``PerceiverAttention`` 增加可选 media mask（上游按定长媒体
  token 处理，无 mask）；② 下游 ``N04V11Readout`` 的媒体 token 组装（当前稠密特征直通、
  支持点双模式状态/权重、位置、年龄、相机嵌入）与输出到 2048 维的 ``to_prefix`` 投影是
  本项目新增；③ 层数/宽度是配置项，项目配置取 6 blocks / 每相机 256 queries。
* 2026-10-16 回源检查上述固定 commit 的 helpers.py，补齐规范化 latents 进入 K/V 的路径。
  此文件按上游结构实现，项目媒体组装／mask／SDPA 路径独立登记。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "UPSTREAM_PROVENANCE",
    "FeedForward",
    "N04V11Readout",
    "PerceiverAttention",
    "PerceiverAttentionBlock",
    "PerceiverResampler",
    "ReadoutConfig",
]

UPSTREAM_PROVENANCE: dict[str, Any] = {
    "repository": "https://github.com/mlfoundations/open_flamingo",
    "commit": "655f693fbfa04cd6e9a987d960654624d48917cf",
    "license": "MIT",
    "module": "PerceiverResampler / PerceiverAttentionBlock / PerceiverAttention",
    "fetched": "2026-10-16",
    "structure_only": True,
    "changes": [
        "PerceiverAttention 增加可选 media mask（本仓库新增；上游无 mask 分支）",
        "N04V11Readout 的媒体 token 组装与 to_prefix(2048) 投影为本项目新增",
        "输入未使用上游的 einops，改用 reshape/permute 等价实现",
    ],
    "source": "https://github.com/mlfoundations/open_flamingo/blob/"
    "655f693fbfa04cd6e9a987d960654624d48917cf/open_flamingo/src/helpers.py",
}


class PerceiverAttention(nn.Module):
    """OpenFlamingo ``PerceiverAttention``：latents 交叉注意媒体 token。"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = float(dim_head) ** -0.5
        inner_dim = self.dim_head * self.heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        latents: torch.Tensor,
        media: torch.Tensor,
        media_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``latents [B,L,dim]`` 对 ``media [B,M,dim]`` 做交叉注意。"""
        latents = self.norm_latents(latents)
        media = self.norm_media(media)
        query = self._split_heads(self.to_q(latents))
        # OpenFlamingo attends to media AND normalized latent queries as keys/values.
        key, value = self.to_kv(torch.cat((media, latents), dim=1)).chunk(2, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)

        keep = None
        if media_mask is not None:
            if media_mask.shape != media.shape[:2]:
                raise ValueError("media_mask 需与 media 的 [B,M] 一致")
            latent_mask = torch.ones(latents.shape[:2], dtype=torch.bool, device=latents.device)
            keep = torch.cat((media_mask.to(torch.bool), latent_mask), dim=1)[:, None, None, :]
        if not return_attention:
            # 不需要热图时用融合 attention，避免为每层保存 [B,H,256,媒体数] 矩阵。
            # 权重/媒体/softmax 的数学含义不变；诊断路径仍返回实际读取分布。
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=keep)
            batch, _, length, _ = out.shape
            out = self.to_out(out.transpose(1, 2).reshape(batch, length, -1))
            return out, None
        logits = torch.einsum("bhid,bhjd->bhij", query, key) * self.scale
        if keep is not None:
            logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
        attention = logits.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attention, value)
        batch, _, length, _ = out.shape
        out = self.to_out(out.transpose(1, 2).reshape(batch, length, -1))
        summary = attention.mean(dim=1) if return_attention else None
        return out, summary


class FeedForward(nn.Module):
    """OpenFlamingo ``FeedForward``：pre-norm 的两层 GELU MLP。"""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        hidden = int(dim * mult)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class PerceiverAttentionBlock(nn.Module):
    """OpenFlamingo ``PerceiverAttentionBlock``：交叉注意 + FFN，各自残差。"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, ff_mult: int = 4) -> None:
        super().__init__()
        self.attn = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        latents: torch.Tensor,
        media: torch.Tensor,
        media_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, attention = self.attn(
            latents, media, media_mask, return_attention=return_attention
        )
        latents = latents + attended
        latents = latents + self.ff(latents)
        return latents, attention


class PerceiverResampler(nn.Module):
    """OpenFlamingo ``PerceiverResampler``：定长 queries 把媒体 token 压成固定长度。"""

    def __init__(
        self,
        *,
        dim: int = 1024,
        depth: int = 6,
        dim_context: int | None = None,
        num_latents: int = 256,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_latents = int(num_latents)
        dim_context = self.dim if dim_context is None else int(dim_context)
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.dim) * 0.02)
        self.input_proj: nn.Module = (
            nn.Linear(dim_context, self.dim) if dim_context != self.dim else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                PerceiverAttentionBlock(
                    dim=self.dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        media: torch.Tensor,
        media_mask: torch.Tensor | None = None,
        *,
        latents: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``media [B,M,dim_context]`` -> ``[B,num_latents,dim]``（可选末块 attention）。"""
        if media.dim() != 3:
            raise ValueError(f"media 需要 [B,M,dim]，实际 {tuple(media.shape)}")
        media = self.input_proj(media)
        if latents is None:
            latents = self.latents.unsqueeze(0).expand(media.shape[0], -1, -1)
        else:
            latents = latents.to(media)
        attention = None
        for index, block in enumerate(self.blocks):
            last = index == len(self.blocks) - 1
            latents, attention = block(
                latents,
                media,
                media_mask,
                return_attention=bool(return_attention and last),
            )
        return self.norm(latents), attention


@dataclass
class ReadoutConfig:
    """读出的结构配置（层数/宽度是项目配置，不冒称作者最优设置）。"""

    latent_channels: int = 256
    state_dim: int = 512
    num_latents: int = 256
    depth: int = 6
    dim: int = 1024
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    prefix_dim: int = 2048
    num_cameras: int = 2
    #: 年龄归一化尺度（记录步，不是秒）。
    age_scale: float = 32.0
    #: 是否返回末块 attention 供可视化；训练时可关掉省显存。
    collect_attention: bool = True
    #: Preserve small observation differences before the downstream BF16 cast.
    prefix_fp32: bool = False


class N04V11Readout(nn.Module):
    """每相机读出：当前稠密特征直通 + 支持点记忆分量 -> ``[B,256,2048]``。

    媒体 token 由三部分组成，缺一不可：

    1. **当前稠密特征直通**：``latent_mean_map [B,256,h,w]`` 全图展平（拼接 2 维归一化坐标后
       1×1 投影）。不能只读 128 个支持点而丢掉当前帧其余区域。
    2. **记忆**：每个支持的**两个模式分量**（``[B,N,2,state_dim]`` 后验均值，不先平均）与
       模式权重 ``[B,N,2]``。
    3. **位置与年龄**：``xy_px [B,N,2]``（原始像素，按帧尺寸归一化）与年龄 ``[B,N]``
       （记录步，按 ``age_scale`` 归一化）。

    相机身份用 ``camera_embedding`` 加到该相机所有媒体 token 上（本项目新增；上游单相机塔
    不需要）。``forward`` 处理单相机，``forward_cameras`` 按相机顺序拼接。
    """

    def __init__(self, config: ReadoutConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReadoutConfig()
        cfg = self.config
        self.dense_proj = nn.Conv2d(cfg.latent_channels + 2, cfg.dim, kernel_size=1)
        self.support_proj = nn.Linear(2 * cfg.state_dim + 2 + 2 + 1, cfg.dim)
        self.camera_embedding = nn.Parameter(torch.zeros(cfg.num_cameras, cfg.dim))
        self.resampler = PerceiverResampler(
            dim=cfg.dim,
            depth=cfg.depth,
            dim_context=cfg.dim,
            num_latents=cfg.num_latents,
            dim_head=cfg.dim_head,
            heads=cfg.heads,
            ff_mult=cfg.ff_mult,
        )
        self.to_prefix = nn.Linear(cfg.dim, cfg.prefix_dim)

    @property
    def num_latents(self) -> int:
        return int(self.config.num_latents)

    def _dense_tokens(self, latent_map: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = latent_map.shape
        device, dtype = latent_map.device, latent_map.dtype
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        nx = 2.0 * (grid_x + 0.5) / float(width) - 1.0
        ny = 2.0 * (grid_y + 0.5) / float(height) - 1.0
        coords = torch.stack((nx, ny), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        fused = torch.cat((latent_map, coords.to(latent_map)), dim=1)
        return self.dense_proj(fused).flatten(2).transpose(1, 2)

    def _support_tokens(
        self,
        support_state: torch.Tensor,
        mode_weights: torch.Tensor,
        xy_px: torch.Tensor,
        age: torch.Tensor,
        size_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch, num_support, modes, state_dim = support_state.shape
        if modes != 2:
            raise ValueError(f"读出需要两个记忆分量，实际 {modes}")
        height, width = size_hw
        # 位置／年龄是坐标，恒 FP32（不随 autocast 变 BF16）：契约 §6 的"坐标与统计"边界。
        xy_px = xy_px.to(torch.float32)
        age = age.to(torch.float32)
        xy_norm = torch.stack(
            (
                2.0 * (xy_px[..., 0] + 0.5) / float(width) - 1.0,
                2.0 * (xy_px[..., 1] + 0.5) / float(height) - 1.0,
            ),
            dim=-1,
        )
        payload = torch.cat(
            (
                support_state.reshape(batch, num_support, modes * state_dim),
                mode_weights.reshape(batch, num_support, modes),
                xy_norm,
                (age / float(self.config.age_scale)).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.support_proj(payload)

    def forward(
        self,
        latent_map: torch.Tensor,
        *,
        camera_index: int = 0,
        support_state: torch.Tensor | None = None,
        mode_weights: torch.Tensor | None = None,
        xy_px: torch.Tensor | None = None,
        age: torch.Tensor | None = None,
        size_hw: tuple[int, int] | None = None,
        support_valid: torch.Tensor | None = None,
        dense_valid: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, int]]:
        """单相机读出 -> ``([B,256,prefix_dim], attention | None, media 计数)``。

        ``support_state`` 为 ``None`` 时只走当前稠密特征直通（消融臂 C）。``attention`` 是
        末块对全部媒体 token 的注意力（跨头平均），与**当前帧**的稠密网格前 ``h*w`` 列对齐。
        """
        cfg = self.config
        if not 0 <= camera_index < cfg.num_cameras:
            raise ValueError(f"camera_index {camera_index} 超出 {cfg.num_cameras} 个相机")
        dense = self._dense_tokens(latent_map)
        media = [dense]
        if (
            dense_valid is not None
            and dense_valid.shape != latent_map.shape[:1] + latent_map.shape[-2:]
        ):
            raise ValueError("dense_valid must match latent [B,H,W]")
        mask = [
            torch.ones_like(dense[..., 0], dtype=torch.bool)
            if dense_valid is None
            else dense_valid.flatten(1).bool()
        ]
        if support_state is not None:
            for value, name in (
                (mode_weights, "mode_weights"),
                (xy_px, "xy_px"),
                (age, "age"),
                (size_hw, "size_hw"),
            ):
                if value is None:
                    raise ValueError(f"支持点读出缺少 {name}")
            assert size_hw is not None
            support = self._support_tokens(support_state, mode_weights, xy_px, age, size_hw)
            media.append(support)
            if support_valid is None:
                mask.append(torch.ones_like(support[..., 0], dtype=torch.bool))
            else:
                if support_valid.shape != support.shape[:2]:
                    raise ValueError("support_valid 需与支持 token 的 [B,N] 一致")
                mask.append(support_valid.to(torch.bool))
        tokens = torch.cat(media, dim=1)
        token_mask = torch.cat(mask, dim=1)
        tokens = tokens + self.camera_embedding[camera_index].to(tokens)
        if tokens.shape[-1] != cfg.dim:
            raise ValueError(f"媒体 token 宽度 {tokens.shape[-1]} != {cfg.dim}")
        want_attention = (
            cfg.collect_attention if return_attention is None else bool(return_attention)
        )
        latents, attention = self.resampler(tokens, token_mask, return_attention=want_attention)
        counts = {
            "dense_tokens": int(dense.shape[1]),
            "support_tokens": int(tokens.shape[1]) - int(dense.shape[1]),
            "media_tokens": int(tokens.shape[1]),
            "latent_tokens": int(self.num_latents),
            "attention_keys": int(tokens.shape[1] + self.num_latents),
        }
        if cfg.prefix_fp32:
            with torch.autocast(device_type=latents.device.type, enabled=False):
                prefix = self.to_prefix(latents.float())
        else:
            prefix = self.to_prefix(latents)
        return prefix, attention, counts
