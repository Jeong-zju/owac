# N01 v1.2 P3 — 同源冻结 ISP 说明

> 2026-10-17：完整配置使用原生RAW传输，由共享 `prepare_frame` 在原分辨率先做冻结ISP，再做双方相同的可选全视野resize；旧客户端先resize的观测差异仅适用于历史实现。ISP公式和作者源码逐元素检查不变。

本文件是"同源 ISP"这一项交付物的**独立说明**。`P3_report.md` §1.1 有同样内容的一段，
但分散在报告里不易单独复核；这里把它收敛成一份可自足的文档，
并补充后来才发现的**服务顺序偏差**。

---

## 1. "同源"指什么，以及它为什么是四臂设计的前提

四臂设计的第一个问题是"同样的机制，换成 ISP 渲染的输入会怎样"（`raw_n01` vs `rgb_n01`）。
只要 RGB 与 RAW 来自**同一张测量张量**、由**确定性函数**产生、**没有另采噪声、没有正常光参考误配**，
这个比较才成立。

本项目用两条显式表示标记把它写进数据里，而不是靠约定：

| 标记 | 含义 |
| --- | --- |
| `rawvla_rgb3` | 作者的三通道 HDR 派生 / pseudo-RAW 视图，即**真实测量**本身 |
| `rendered_rgb3` | 同一测量经**冻结 ISP** 之后的视图 |

**必须说清楚的边界**：这里的 "RAW" **不是原生 Bayer CFA**，而是作者已经过三通道派生处理的
pseudo-RAW。`isp_provenance` 里以 `not_a_cfa: True` 与 `note` 字段显式记录，
不得据此声称恢复了真实传感器的原始动态范围。

---

## 2. 冻结的上游实现（先检索作者源码再封装）

| 平台 | 上游入口 | 符号 |
| --- | --- | --- |
| LIBERO | `starVLA/examples/simBenchmarks/LIBERO/eval_files/raw_rgb10_unprocessing.py` | `_raw10_to_rgb8` |
| RoboTwin | `third_party/FastWAM/third_party/RoboTwin/envs/camera/raw_rgb10.py` | `raw10_to_rgb8` |

作者的调用点为 `experiments/simulation/libero/rawvla_bench/raw.py::make_rawvla_observation`。
本项目转写在 `src/owac/raw/n01_paired_views.py::paired_views`，
`isp_provenance(platform)` 记录上游 commit（LIBERO 侧实测为 `57d508cb180d0510ac9060358009c0f6ffaac420`）。

**一个必须区分的同名陷阱**：作者仓库里 "default_isp" 指两个不同东西——

- `finetune/calibration/default_isp.py::apply_default_isp`：ColourChecker 版，**只有
  `agentview`/`wrist` 两个 profile**；
- RawVLA-Bench 的 `representation="default_isp"`：即上面的 `_raw10_to_rgb8`。

本项目用**后者**，并在 provenance 里写明，避免二者混为一谈。

---

## 3. 映射与参数

固定参数（`FIXED_RAW_RGB10_PARAMS`，来自作者冻结源码，不是本项目拟合）：

```
red_gain = 1.8, green_gain = 1.0, blue_gain = 1.7, inverse_global_gain = 1.0
```

处理的形状为：

```
smoothstep(linear_to_srgb(clip(raw10 / 1023 * gains, 0, 1)))
```

即：按 10 bit 归一 → 逐通道增益 → 裁剪到 [0,1] → sRGB 线性化 → smoothstep 音调曲线。
`paired_views` **不做重采样、不做逐帧归一化、不额外加噪**：RGB 与 RAW 只差 ISP 本身。

---

## 4. 已验证的内容

远程实跑（`OWAC_RAWVLA_SOURCE` 指向固定 checkout）：

- 本项目的 **NumPy 转写**与作者 `_raw10_to_rgb8` / `raw10_to_rgb8` 在真实 uint16 raw10 上
  **逐元素相等**；
- **torch 训练路径**与 NumPy oracle 逐元素相等（uint8）；
- 每条记录都带 `rawvla_rgb3` / `rendered_rgb3` 标记，**RGB 不会被标成 RAW**。

远程真实 batch 复核（`receipts/verify-libero-raw_n01-seed101.json`）：

该 receipt 的实际键路径是 `same_condition_pair.agentview`（**不是** `pair[agentview]`——
本文件初稿写错过，已按 receipt 实际内容更正）：

```json
"same_condition_pair": {"agentview": {
  "raw_domain": "rawvla_rgb3", "rgb_domain": "rendered_rgb3",
  "identical_source": true, "rgb_is_not_identity": true,
  "rgb_range": [0.0, 0.9316818118095398],
  "provenance_module": "starVLA/examples/simBenchmarks/LIBERO/eval_files/raw_rgb10_unprocessing.py",
  "cache_dtype": "uint8"}}
```

`identical_source: true` 是"同源同条件"的直接证据；`rgb_is_not_identity: true` 说明 ISP 确实改变了张量。

---

## 5. 已记录的信息损失（不掩盖）

**LIBERO 缓存是 uint8**：`rint(raw10 * 255 / 1023)`，即 ISP 的输入在任何 P3 模型看到之前
**已经被量化到 1/255**，约 2 bit 的 10-bit 测量**不可恢复**。
这一点以 `quantisation` 字段写进 provenance，并在报告中按"信息已丢失"而非"模型不好"记录。

**RoboTwin 是 float32**：无此量化损失。两个平台的信息条件因此不同，跨平台比较必须带上这一条。

---

## 6. 服务顺序偏差（后来才实测发现，必须写在这里）

本项目原先记录的顺序是：

```
RAW cache resolution -> frozen ISP on the full frame -> policy resize/crop
```

**但服务路径上的实际顺序相反**：冻结的客户端在**发送前**就把帧缩放到 128×128，
因此 ISP 看到的是**已经缩放过的**帧。实际顺序是：

```
RAW cache -> client resize -> frozen ISP
```

这不是实现疏漏——客户端是作者冻结代码，不改它就无法调换顺序——但它是一个**真实的
train/eval 观测顺序差异**，且作者自己的警告表明这类不一致会静默降低成功率。
`isp_provenance` 现已包含 `serving_order_note` 字段如实记录该偏差。

**作者源码里的权威依据**（本轮补上，使这条不是我的推断）：
`source/starVLA/examples/simBenchmarks/LIBERO/eval_files/eval_libero.py:431-433` 的注释原文是

```
# Correct deployment order: simulator/render resolution
# -> RAW + frame noise -> downstream policy resize.
```

即作者明确把正确顺序写成"渲染分辨率 → RAW 与帧噪声 → 下游策略缩放"，
**ISP 属于 RAW 这一步的一部分**，因此应当发生在缩放之前。
本项目的服务把自带的冻结 ISP 放在**收到已缩放帧之后**，与该顺序相反。
这条注释同时说明：作者对顺序敏感是有意为之，而不是可以忽略的实现细节。

**补充说明本项目的实际链路**（与 README 的对照）：客户端在 `rawvla_make_observation`
（RAW 转换 + 帧噪声）**之后**才 resize，然后才发送；服务收到的是 128×128 的 RAW 帧，
再施加本项目的冻结 ISP。所以链路是 `render → RAW → client resize → ISP`，
而不是作者的 `render → RAW(+ISP) → resize`。

---

## 7. 未验证的部分（如实列出）

- 第 4 节的**逐元素相等验证需要** `OWAC_RAWVLA_SOURCE` 指向
  **`starVLA/` 与 `third_party/` 两者的父目录**（测试按 `<root>/starVLA/...` 与
  `<root>/third_party/...` 拼路径）。未设置时该交叉检查会被**跳过**。
  **2026-10-16 复跑（远端）**：`OWAC_RAWVLA_SOURCE=/workspace/owac/rawvla/source`
  → `test_n01_v12_frozen_isp.py` **4 passed，0 skip**，故本节第 4 节的结论是**当前可复现**的。
  **注意**：若误把该变量设成 `.../starVLA` 本身，两个平台都会 "author module not found" 而 skip——
  **命令成功退出并不代表检查执行了**；
- 本说明**不**声称 ISP 的数值选择（增益、音调曲线）对本项目任务是"最优"的——
  它是作者冻结实现，本项目只保证**同源且一致**；
- 本说明**不**提供 ISP 对下游动作影响的证据：那需要对照臂，属 P4 安排。

---

## 8. 复核方式

```bash
# 逐元素交叉检查（需要作者 checkout）
OWAC_RAWVLA_SOURCE=<starVLA checkout> uv run --locked pytest \
    tests/integration/test_n01_v12_frozen_isp.py -q

# 同源性与标记的实跑证据
python -c "from owac.raw.n01_paired_views import isp_provenance; import json; \
print(json.dumps(isp_provenance('libero'), ensure_ascii=False, indent=2))"
```

远程同样内容的 receipt：`receipts/verify-libero-raw_n01-seed101.json`
（含上面 `same_condition_pair.agentview` 的字段）。
