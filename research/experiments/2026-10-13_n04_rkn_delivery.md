# N04 RKN 独立交付

用户选择基于作者 RKN 继续开发，FilterPy 实现、开发调参、测试和依赖另存 Git stash。核心 N04 未修改。

本提交保留作者 RKN（MIT 许可与固定来源见 vendor provenance）、固定当前帧测量、因果训练及缺失标记、无滤波对照和测试。RKN 三个初始化缺失误差 0.2637/0.2402/0.2527 px，均值 0.2522±0.0117；无滤波 1.2029 px。连续观测 RKN 0.1796±0.0081，无滤波 0.1695。停止/反向误差 0.3280/0.6455，修订滞后仍待解决。

原始实验使用共享训练测量头、64 训练场景、30 开发场景和30最终场景，RKN 每初始化1200步，开发集选模后冻结。完整历史对照含 FilterPy 的成绩作为真实历史保留，不能宣称 RKN 超过它；本提交仅选定后续开发基础。独立 RAW 诊断留在工作树，不纳入本次算法提交。

小型原始证据见 results/2026-10-13_n04_matched_tracking/{protocol.json,tracking-summary.json,rkn-vendor-audit.json}。协议中的源码哈希指向拆分前实验，拆分未重新训练，不将旧哈希冒充新提交。权重和大缓存仍位于 ignored outputs/n04-matched-tracking-20261013-a1。

复跑 RKN/无滤波：

```bash
uv sync --locked --extra encoder
uv run --locked python -m owac.evaluation.n04_tracking --output-dir outputs/n04-rkn-reproduction --rkn-steps 1200
```

该入口不运行 FilterPy，因此新结果行数不同于历史完整矩阵。新输出目录不覆盖历史证据。合成局部动态收益不代表 N04 分解收益、真实 RAW 或 π0.5 闭环成功。
