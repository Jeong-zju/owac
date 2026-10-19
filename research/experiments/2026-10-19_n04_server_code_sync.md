# N04 服务器代码同步（2026-10-19）

按用户要求只同步代码及必要配置；未下载检查点、训练数据、视频或模型依赖权重。未启动或恢复训练。服务器查询时两项 v5 训练均为 STOPPED，监控仍在运行。

## 本地保存与校验

全部保存于 `outputs/n04-server-rescue-20261019/`（本 checkout 内，被 Git 忽略）。

- `n04-code-rescue-20261019.zip`：94 版源码快照，29,279 个路径条目，628 份不同内容文件，6,831,296 字节。按内容去重；排除 research、data、outputs、缓存、虚拟环境和 .git。历史文档/测试/依赖声明包含在代码包内。
- SHA-256：`3df8f090351968713c3f747e56847f3378f84c089068e5d3d8a05fb7b57de5fe`；与服务器一致，全部内容 blob 已逐份重新校验。
- `latest-source/`：最新正式源码 `n04-v11-codex-stability-v5-deterministic-dev-20261019` 的代码目录，已解包便于直接查看。
- `verify_restore.py`：重跑归档和逐文件校验并恢复最新代码；历史版本按 `code-manifest.json` 中的路径和 blob 哈希恢复。
- `supervisor/`：11 份 N04 服务配置；`control-and-monitor/`：监控代码与控制脚本/运行说明。11 份配置与 2 份线上监控源码均与服务器 SHA-256 一致。
- `remote-config-and-links.txt` 保存源码软链接映射（外部目标未下载）；`PROVENANCE_N04.md` 与 `upstream-state.txt` 保存 StarVLA 来源与本地修改状态。上游只有来源说明为未跟踪文件，无已跟踪代码差异。
- `local-before-sync.diff` 保存同步前本地已跟踪文件的未提交修改。未覆盖、重置或提交本地工作树。

## 差异结果

最新快照与工作树比较：333 文件相同；5 处不同均为文档（AGENTS.md、README.md、docs/status.md、docs/n04_v11_s2_interfaces.md、docs/prompts/n04_v1_1/README.md），本地含更新的 v5 交付说明，保留本地版本。4 个服务器独有文件均为来源/校验清单，已保存在 latest-source 中。线上监控 py/html 与本地 src/owac/runtime 文件完全相同。实现代码无待合并差异。

## 边界

这是 N04 代码备份与同步，不是完整实验迁移。按用户指令，检查点、数据集、视频、环境二进制和外部模型权重仍未下载。服务器被回收后，仅凭本代码包不能恢复这些资产。小型校验收据见 verification.json 与 config-verification.json。
