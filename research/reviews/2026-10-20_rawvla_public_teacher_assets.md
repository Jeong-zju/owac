# RawVLA 的公开模型与 RAW 教师资产核查

核查日期：2026-10-20。OWAC-D1.1 要求教师策略对外接收 RAW，可使用因果 burst 与递归状态。作者公开了模型实现和训练入口；本轮未在核查的官方入口找到可直接下载的 RawVLA 训练后 Neural ISP／完整 RAW 教师 checkpoint。这个结论是“尚未找到公开下载”，不等于断言作者从未提供权重。

## 已确认的公开资源

| 资源 | 核查结果 | 对教师构建的意义 |
| --- | --- | --- |
| [作者项目页](https://shuhongll.github.io/rawvla/) | 描述 RAW burst、递归成像前端与冻结 VLA | 教师完整策略可以接收 RAW，内部 ISP 输出图像不改变输入模态 |
| [官方源码](https://github.com/ShuhongLL/RawVLA) | 公开 RAW 前端、StarVLA 接入、训练与仿真评测组件 | 有实现可复用，不表示已有本地可用教师 |
| [前端模型与用法](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/baselines/rawvla/README.md) | 支持 burst 和跨步状态 | 无需为了旧 prompt 的单帧限制而删去作者时序能力 |
| [共享训练配置](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/base/rawvla.yaml) | `burst_frames=6`、`rnn_frames=6`；只保存可训练部分 | 区分 burst 长度与训练展开长度；不能假定两者等同于独立帧数 |
| [π0.5 LIBERO 配置](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/libero/pi05.yaml) | 引用本地已转换 π0.5 权重路径 | 这是后端配置，不是公开完整 RAW 教师权重链接 |
| [checkpoint 下载脚本](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/tools/checkpoints/download_starvla_libero.py) | 下载 StarVLA Qwen 系列策略／基础模型 | 不能当作 RawVLA Neural ISP 或所需 π0.5 完整教师权重 |
| [作者 Hugging Face 账号](https://huggingface.co/ToferFish) | 核查时显示公开 models 为 0 | 未从该账号获得训练后教师权重 |
| [RawVLA-Bench](https://huggingface.co/datasets/ToferFish/RawVLA-Bench) | 有 RAW／RGB 配对训练轨迹 | 数据集不是模型 checkpoint |

GitHub 默认分支本轮定位为 `ba9c56b04da2c4931b70d8844710f5928d43775b`。GitHub API 的该提交递归文件清单未截断，没有发现 `.pt`、`.pth`、`.safetensors` 或 `.ckpt` 权重文件；公开 releases 列表为空。作者前端 README 提及历史检查点，不构成公开下载地址。Hugging Face 全站搜索 API 本轮网络连接失败，不据此声称全站不存在其他权重。

## 复现与使用边界

作者 [RoboTwin 训练说明](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/robotwin/README.md)明确当前配置从头初始化 RAW 前端，历史论文权重早于当前前端更改，不能把单次新训练说成逐位复现论文权重。

优先寻找有来源、配置和可加载证据的作者 RAW 前端权重，再组合相匹配的策略后端、tokenizer 和统计量；若只有从零初始化的模型，先交付训练教师的独立计划与预算，按新的实验授权执行。可以由用户提供作者权重；本轮没有联系作者、下载大模型或训练教师。

验收对象是完整 `RAW 历史＋可用状态／语言→前端→VLA→动作` 系统，需核对真实输入路径、权重／配置、状态 reset 和实际任务表现。单独 RGB 策略、随机 RAW 前端、函数能 forward 或论文成绩都不能替代这一验收。
