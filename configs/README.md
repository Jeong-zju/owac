# 配置目录

保存可审查、可版本化的运行参数。当前 `backends/pi05-runtime/` 提供 π0.5 的独立环境、锁文件、来源清单与环境补丁，运行命令见[部署说明](../docs/pi05-deployment.md)。候选训练尚未开始，没有默认候选训练配置。

| 子目录 | 预留用途 |
| --- | --- |
| `candidates/` | 具体候选的状态、规模、学习/求解与预算参数 |
| `baselines/` | 最近机制及 C0—C4 系统基线配置 |
| `data/` | 采集、校准、清单、拆分和数据路径引用 |
| `experiments/` | 组合候选/基线、数据、后端、种子与输出位置 |
| `backends/pi05-runtime/` | 已部署的 π0.5 PyTorch 环境、版本锁与源码来源 |

配置格式和加载器尚未冻结。选择后记录默认值、类型校验和完整解析配置；不要在配置中嵌入密钥。机器私有路径放在被忽略的 `configs/local/`，实验记录保留可重建的路径映射说明。

## N02 编码器配置

`encoders/n02/` 保存 N02 测量约束循环补全的 tiny/small/medium 三档配置，由
`owac.frameworks.n02.config.N02Config.from_json` 直接加载（未知键报错，不静默忽略）。

| 配置 | K | D | 循环 | 历史 | stem 宽度 |
| --- | --- | --- | --- | --- | --- |
| `tiny.json` | 4 | 32 | 4 | 4 | 32 |
| `small.json` | 4 | 64 | 6 | 4 | 48 |
| `medium.json` | 8 | 96 | 8 | 4 | 64 |

三档都显式写出 `measurement`、`revision`、`loss_weights` 三个块。
`measurement.input_mode=rawvla_rgb3` 对应三通道伪 RAW 测量头（合成生成器的默认形状）；
`loss_weights` 取提示词起点 `1.0 / 1.0 / 0.1 / 0.01`，每一项的归一化定义写在
`src/owac/training/n02.py` 的模块文档里并逐项记录。K 是候选解释数，不是额外观测数；
三档只用于分别消融 K、宽度与循环，不把三项同时增长解释为单项收益。

加载示例：`N02Config.from_json(Path("configs/encoders/n02/tiny.json"))`；命令行入口见
`python -m owac.training.n02 --config configs/encoders/n02/tiny.json --synthetic --steps 200`。
