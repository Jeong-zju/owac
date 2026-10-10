# 官方 π0.5 / 原版 LIBERO RGB 直接使用验收

**后续用途纠正（2026-10-10）：用户要求开箱即用与仿真器原生 CFA/RAW 同时满足，见 [0008](../../docs/decisions/0008-pretrained-and-native-cfa-required.md)。本实验保留为有效 RGB 参考，未通过主 benchmark 的 RAW 准入条件；10/10 与原始运行记录不变。**

目的：按用户最新要求更换失败的 benchmark/backbone 组合，确认已有官方策略可直接完成原版任务，避免继续自定义 backbone 的微调。决策见 [0007](../../docs/decisions/0007-pretrained-benchmark-first.md)。

## 固定身份

- 协议：[pi05-libero-rgb-v1.json](../../configs/baselines/pi05-libero-rgb-v1.json)。
- 模型：官方 `pi05_libero`，额外训练 0 步；未使用红杯微调后的权重。
- openpi：`15a9616a00943ada6c20a0f158e3adb39df2ccac`；原版 LIBERO gitlink：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。
- 模型 SHA256：`4bcdb2aacab3e215211f9a5733c62d8701633e5aef757192cfc0bf7d85f5f47d`；原归一化统计 SHA256：`b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`。
- 官方评测入口 SHA256：`d53748c95804ada559e3ca867305b8f755d8651c4c7f4e2723ef80aea55a3943`。使用其原 `eval_libero` 循环，没有改写相机、动作、控制器或任务成功规则。
- LIBERO-Goal 的全部 10 项任务，各 1 条，种子 7，使用每任务原初始状态文件的索引 0，等待 10 步、最多执行 300 个策略控制步，动作重规划间隔 5。
- 模拟器 Python 3.8.20、robosuite 1.4.1、MuJoCo 3.2.3，独立 runtime；推理复用已有 Python 3.11 / PyTorch 2.7.1+cu128 环境。完整冻结依赖另留档。
- GPU：租用的 RTX PRO 6000 Blackwell Max-Q，云端路径 `/workspace/owac/reference/pi05-libero-rgb-v1/`。旧运行目录及权重保留。

## 可观测性与运行变化

本仓库包装器调用官方评测函数，仅记录任务、初始状态哈希、完整 episode 成功/异常、动作步数和推理往返时延；关闭 MP4 写入并在任务切换时关闭前一环境。额外种下 PyTorch 推理随机种子 7；关闭 `torch.compile`，不修改预训练计算、模型结构或权重。策略服务只绑定云端 localhost，并由 supervisor 父流程收尾。

`acceptance-01` 在控制循环开始前因数字记录器导入 Python 3.11 的 `datetime.UTC` 而退出；`acceptance-02` 因未加入官方要求的 LIBERO 根目录 `PYTHONPATH` 而退出。两次均没有策略 episode，不计为 0% 分数。修复记录器 Python 3.8 时间兼容与运行路径后，先验收全部依赖导入和官方 10 项任务/50 个初始状态，再在新目录 `acceptance-03` 重跑；原目录保留。没有改变模型或仿真任务。

## 当前结果

`acceptance-03` 于 2026-10-10 02:46:43 UTC 完成：LIBERO-Goal 全部 10 个任务各 1 条，**成功 10/10，基础设施异常 0，额外训练 0 步**。包括开抽屉、放碗、放酒瓶、抽屉内放物、推盘、奶酪入碗、开炉灶与杯盘放置等任务，没有筛掉失败任务或替换初始状态。10 条证据的任务 ID / 初始状态索引唯一且完整，逐条成功计数与摘要一致，哈希已核验。

推理往返时延 p50 / p95 为 **116.586 / 120.984 ms**。10/10 的汇总 Wilson 95% 区间为 72.247%–100%，仅为这个小样本的二项近似，不证明完整任务分布达到 100%；本次每任务只有 1 条，不作为官方 98.0% 的完整复现。模型峰值和实际耗时见数字留档。

含加载在内总耗时 **82.632 s**，PyTorch 峰值分配 / 保留显存为 **7,644,988,416 / 7,767,851,008 bytes**。记录器 CPU 边界测试、`make check` 与全部 22 项测试通过，历史页面脚本通过 Node 语法检查。HTML 为无服务的离线结果页；本轮未作新页面的浏览器视觉验收。验收 supervisor 自然退出，策略 localhost 服务已收尾，租用实例和官方权重保留。后续运行入口加入端口空闲检查，避免误连已有策略；本次运行的实际源码哈希保存在 provenance，不重写历史哈希。

结果说明：已有官方策略在对应原版环境直接成功，满足当前“先要一个可以直接用的 backbone/bench”验收目标。当前无需继续红杯模型训练或修补。原版 RGB 的后续更大规模评测，以及 RAW/特征替换，是独立工作。

数字清单：[results/2026-10-10_pi05_official_libero_reference](results/2026-10-10_pi05_official_libero_reference)。逐条结果 SHA256：`a15734d6f0fa7396ffffc448d8f4b0b455ebb079d6f071c23fd2593896565015`。

同步数字并生成固定离线 HTML：

```bash
uv run --locked python -m owac.runtime.libero_reference archive \
  --run-root /workspace/owac/reference/pi05-libero-rgb-v1/acceptance-03 \
  --destination research/experiments/results/2026-10-10_pi05_official_libero_reference
```

新进度入口：[官方 LIBERO RGB 基线](../../outputs/official-libero-baseline/progress.html)。定时任务维持关闭；仅手动同步。RGB 参考不声明原生 CFA/RAW 能力。
