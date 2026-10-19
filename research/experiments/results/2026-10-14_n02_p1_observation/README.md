# N02 v1.1 P1 结果清单（2026-10-14）

本目录是 P1 阶段的紧凑清单；大产物（检查点、逐场景记录、日志、曲线）在远端 `outputs/n02-full/` 下。

| 文件 | 内容 |
| --- | --- |
| `stage_decision.json` | 四栏判定与证据路径、代码哈希、预算与裁定 |
| `metrics.json` | 实测计数、成本、尺度证据、行为基线 |
| `bundle_receipts.py`（旧机）/ 本文件 | 清单生成与说明 |

**当前状态（交付时）**：implementation=pass、numerical=pass、behavior=incomplete（CFA4 测量链在飞）、closed_loop=not_run。
远端主机（迁移后）：`ssh -p 16724 root@154.59.156.22`，worktree `/workspace/owac/worktrees/n02-measurement-completion`。

**恢复/复现入口**：
1. 目标测试：`N02_PY=$PWD/outputs/n02-full/runtime/py310-cu128/bin/python; PYTHONPATH=$PWD/src $N02_PY -m pytest tests/encoders/n02_full -q`
2. P0 parity：`PYTHONPATH=$PWD/src $N02_PY -m owac.frameworks.n02_full.components.parity_suite --fixtures outputs/n02-full/p0/parity/fixtures --output outputs/n02-full/p0/parity`
3. dev 网格结论：`outputs/n02-full/p1-dev-grid/SELECTION_DECISION.md`
4. 正式链（GPU0）：`outputs/n02-full/p1-formal/chain.sh`（watcher 等 stem → interim 1k → 验收 → 续 5k → 四臂）
5. raw3 臂（GPU1）：`outputs/n02-full/p1-formal/start_raw3_new.sh`（检查点每 5000，`--resume` 续训）
6. 验收：`PYTHONPATH=$PWD/src $N02_PY -m owac.evaluation.n02_full.p1 {oracle,behavior,report} --config configs/encoders/n02_full/p1_eval.json --measurement-config configs/encoders/n02_full/p1_measurement.json --checkpoint <ckpt> --interim-budget <n>`
