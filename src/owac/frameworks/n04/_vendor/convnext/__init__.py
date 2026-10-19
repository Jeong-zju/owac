"""ConvNeXt 官方实现的固定快照（MIT）。

上游：https://github.com/facebookresearch/ConvNeXt
commit：048efcea897d999aed302f2639b6270aedf8d4c8
来源与逐文件 hash、修改说明见同目录 ``provenance.json``。

本包只导出网络定义；N04 的测量模块在它之上做多尺度读出与几何头，
不修改这里的层结构。
"""

from .convnext import Block, ConvNeXt, LayerNorm, convnext_base, convnext_large, convnext_small

__all__ = [
    "Block",
    "ConvNeXt",
    "LayerNorm",
    "convnext_base",
    "convnext_large",
    "convnext_small",
]
