# N04 修复前源码审查档案

`review.json` 和 `probe.py` 记录 `b4c26cc4b0de3a3df5e73bca4eb4eab43b849bd4` 的六项问题。探针中的断言预期旧缺陷存在，重放时须使用该版本的源码与 review 中指定的旧 checkpoint；它不是当前实现的验收脚本。

当前修复、回归和对比见[六项修复记录](../../experiments/2026-10-13_n04_six_fixes.md)，当前权重的只读审计见[analyze.py](../../experiments/results/2026-10-13_n04_six_fixes/analyze.py)。旧源码副本及旧权重保持在被忽略的 `outputs/`，原 review 数字与断言保留。
