# RawVLA-Bench π0.5 本机参数与结果归档

2026-10-14 已将本轮训练声明、全部有效评测参数、原始清单、正式与异常数字记录、正常化统计、环境版本及实际代码快照整理到本机。主清单 13,250 条及独立原条件 632 条均完整；LIBERO 58.20%、RoboTwin 15.63%、原条件 17.09%。额外训练为 0 步，上游检查点训练配方和作者 ISP 参考配方独立标记，未公开或未记录的项明确保留未知。

- [归档说明与参数索引](../../outputs/rawvla-archives/2026-10-14_rawvla_pi05_baseline/README.md)：查看每类设置、结果和有效覆盖关系。
- [有效评测参数](../../outputs/rawvla-archives/2026-10-14_rawvla_pi05_baseline/parameters/effective-evaluation-parameters.json)：两平台模型配置、40 个 LIBERO client 参数、13 个 RoboTwin 步数上限和服务参数白名单。
- [训练声明与上游来源](../../outputs/rawvla-archives/2026-10-14_rawvla_pi05_baseline/parameters/training-provenance.json)。
- [完整归档包](../../outputs/rawvla-archives/owac-rawvla-pi05-baseline-20261014-v1.tar.gz)：约 13.39 MiB，解压后约 98.08 MiB，669 个文件。
- [机器可读归档收据](2026-10-14_rawvla_pi05_baseline.json)：包含包 SHA256、文件清单哈希、比较条件指纹及独立解压校验结果。
- [未来算法比较模板](../templates/rawvla-comparison.json)：为新实验填写实际 method、训练、评测、成本和逐 episode 结果，不覆盖本基线。

归档根目录为 `/home/jeong/zeno/owac/repo/outputs/rawvla-archives/2026-10-14_rawvla_pi05_baseline/`。大归档放在被忽略的 `outputs/`，本页、收据和比较模板保留在研究目录。权重、图像和大动作数组继续留在服务器；本机保存其来源、哈希与已完成审计，归档不依赖云端即可核对数字和设置，仿真重跑仍需要原数据、资产和检查点。

全部 668 个文件哈希、13,882 个正式身份、配对 flow seed/语言和固定成绩已在独立解压目录校验通过。包 SHA256：`a36e13d34b238e22584e7a90e7f44e05a98fc3071155f223aa199c7e55ccbd5b`。比较条件指纹：`3e11d3280a5709788de14a5f1dc66d45863a8f5fc6e7b0f4c14038a910c9e1d9`。两个继承 LIBERO 日志没有原始启动头，实际参数已由冻结源重建并明确区分；上游训练配方不是本轮训练执行证据。完整分母已取得，论文数值仍未复现，边界见[完整评测报告](../experiments/2026-10-14_rawvla_pi05_complete_evaluation.md)。

```bash
cd /home/jeong/zeno/owac/repo/outputs/rawvla-archives/2026-10-14_rawvla_pi05_baseline
python3 verify_archive.py
```
