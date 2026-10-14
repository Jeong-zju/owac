# RawVLA-Bench π0.5 全量仿真基线

目的：为 OWAC 后续算法建立现成 Default π0.5 的完整仿真参照。**2026-10-14 已补齐 33 条主 RoboTwin 基础设施缺口；主清单 13,250/13,250 完成，身份审计异常 0 个。** 之前的部署验收成绩独立保留，不复用为本次正式结果。

做了什么：保留作者清单、检查点、正常化与固定种子，在独立专家修复覆盖中补跑原缺失身份；策略 setup 前恢复原任务方法。原 3,217 条有效 RoboTwin 结果字节不变，旧异常留档，正式 ledger 仅追加通过核验的 33 条。

效果：LIBERO 成功 5820/10000，58.20%；主 RoboTwin 成功 508/3250，15.63%；独立原条件核查成功 108/632，17.09%。逐任务、照明域、SOG/Long/Avg、全部 RoboTwin 实际动作及模型调用成本见[完整评测报告](2026-10-14_rawvla_pi05_complete_evaluation.md)。

结论：完整 manifest 的本机固定实现基线已经取得，可用于同协议算法配对比较。论文数值尚未复现；OptiX、StarVLA/原 LeRobot 数值等价与论文 RNG 边界仍须披露，不能把本轮与论文的分数差解释为算法改进。

## 最新运行快照

完整结果导出时间：2026-10-14 10:44:24（北京时间）。主结果 complete、13,250/13,250；三个补跑队列全部 EXITED，未补齐基础设施身份 0 个。

| 评测 | 有效完成 | 成功数 | 正常策略失败 | 完整成功率 |
| --- | --- | --- | --- | --- |
| LIBERO | 10,000/10,000 | 5820 | 4180 | 58.20% |
| RoboTwin 主清单 | 3,250/3,250 | 508 | 2742 | 15.63% |
| 原条件核查（独立计分） | 632/632 | 108 | 524 | 17.09% |

主 RoboTwin 原 85 次异常和本次 2 次运行阻塞均保留；附加原条件历史 18 次异常保留。两个运行阻塞通过同身份重试完成，卡点前分别 1,002/668 个完整步骤的状态/动作逐值一致。完整身份、真实动作与输入审计、追加证明和校验和见[完整结果目录](results/2026-10-14_rawvla_pi05_complete_evaluation/complete-summary.json)。此前 632 条的五个专家异常修复见[原记录](2026-10-13_rawvla_robotwin_infrastructure_repair.md)。原发布快照和 `owac-rawvla-pi05-v0.1.0` 标签不变；下文保留原部署和运行协议记录。

## 来源与范围

- [论文 §5 与 Appendix B](https://arxiv.org/html/2609.37530v1)：四套 LIBERO 共 40 个任务、每任务 50 个初态、五个照明域，10,000 条；RoboTwin ALOHA-AgileX clean 的 13 个任务、每任务 50 个 dataset seed、五域，3,250 条。
- [作者 LIBERO 清单](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json)、[RoboTwin 清单](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/robotwin/manifests/robotwin2_test_manifest_50_seeds_3250_rollouts.json)。保留完整原清单及 SHA256，不自行用 0–49 代替实际 dataset seed。
- [作者 π0.5 LIBERO 配置](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/configs/libero/pi05.yaml)明确覆盖共享配置为十步 horizon；[RoboTwin backbone 指南](https://github.com/ShuhongLL/RawVLA/blob/ba9c56b04da2c4931b70d8844710f5928d43775b/experiments/simulation/robotwin/backbones/README.md)指定原始 LeRobot PyTorch 权重和统计。
- 本轮执行论文主仿真对比中的 **Default π0.5**：没有训练 backbone 或学习 ISP；不包含全部其他 ISP 方法、附录的全部干扰/消融矩阵或实体机器人实验。后续算法相对本机固定基线比较，不直接将差值宣称为超过论文方法。

## 冻结协议

| 项目 | LIBERO | RoboTwin |
| --- | --- | --- |
| 输入 | 作者 RAWVLA-Bench 生成链路的 `default_isp` | SAPIEN `render_rgb`（Color） |
| 任务 | Spatial/Object/Goal/LIBERO-10 各 10 个 | 作者 13 个任务 |
| 每照明域分母 | 2,000 | 650 |
| 状态/动作 | 原本体状态，七维环境动作 | 实测十四维关节与夹爪，十四维绝对关节目标 |
| 正常化 | 原 π0.5 quantile | 原检查点 MEAN_STD、epsilon 1e-8 |
| 动作 horizon/重新规划 | 10 / 5 | 50 / 50 |
| 预测与成功 | 原作者 client、原仿真成功判据 | 原作者 eval loop、原任务成功判据 |
| 随机性 | 每初态独立 action-noise seed | 每 dataset seed 独立 flow seed，chunk seed 加实际步数 |
| 渲染 | 原 MuJoCo EGL | 原 SAPIEN RT，显式 OptiX 去噪 |

论文 Table 2 的 Default 明确是 SAPIEN tone-mapped RGB。之前部署用 `default_isp`，因此本次 RoboTwin 正式任务改用 `render_rgb`，不把这两种成像入口混为一谈。RAW 候选可以访问对应的 HDR-derived RAW；独立 Default 基线访问 Color。两者由同一场景、照明、状态和相机协议产生。

action-noise seed 使用 `owac-pi05-full-v1/platform/task/initial-state` 的 SHA256 前四字节，小端 uint32。同一初态的五域使用相同随机流；每 episode 重置，避免恢复调度或上一条轨迹长度改变后续噪声。RoboTwin 的语言候选来自作者任务生成器，Python random 与 NumPy 均按同一初态独立重置；不向策略传入专家动作、未来状态或仿真物体真值。独立审计会拒绝跨照明域改变配对语言的结果。

权重、正常化、数据与环境来源见 [部署验收](2026-10-10_rawvla_pi05_deployment.md)。运行 source 为 RawVLA `ba9c56b04da2c4931b70d8844710f5928d43775b`；本地分支 `lzh/pi05-sim-baseline`，基础 HEAD `0fc31025f2effeaa1b1fd49d6b175b84ef624265`，本轮改动未提交。固定协议文件 SHA256：`3ac9626ef71eab4719be540393f51b66b767d7348dc4cac0ce5f5455f5b52e0e`。

## 实现、结果与异常

调度器 [rawvla_full_eval.py](../../src/owac/runtime/rawvla_full_eval.py)保存作者脚本的独立快照，添加每 episode 随机流重置、有限动作检查、实际动作/步数/墙钟记录及固定种子的异常退出。原作者 source 不被改写。[策略插件](../../src/owac/runtime/rawvla_robotwin.py)核对每个案例的真实 lighting metadata，保存完整模型调用输入、预测 action chunk 与实际控制步的状态/动作。[独立审计](../../src/owac/runtime/rawvla_eval_audit.py)按原始 entry 内容散列检查身份和分母，全部条目完成前不输出正式成功率。

- LIBERO：每任务 250 条，采用作者 episode ledger 恢复；所有应用动作单独追加保存，每任务保留首个实际输入/正常化/noise/action 的 NPZ 审计。动作日志在中断后可能含未完成 episode 的前缀，只有 episode ledger 的完成记录计分。
- RoboTwin：逐条独立进程、固定原 seed，先按原清单执行 13 个任务的首 seed/Normal，再执行余下条目；此顺序预先固定，不选成功任务。每条最多两次本轮基础设施尝试，独立目录保留日志。策略未达到原步数上限前的成功条件计正常失败，不重试求成功。
- 模拟器、加载或非有限推理异常保留在 attempts 清单；异常不替换 seed，不写成成功，也不从正式分母中删除。若存在尚未解决条目，最终状态是 incomplete。
- copied RoboTwin 脚本首次运行的相对相机路径未连接，出现 missing task config，在模型动作产生前失败。已添加指向作者原 task_config 的软链接，无相机参数修改；原失败尝试保留，同一条目在 attempt 3 等新目录重跑。首成功记录因此是 attempt 3。更早的 schema/脚本准备失败也在独立目录保留，未产生可计分 episode。
- v3 已在 `prepare()` 自动生成该软链接，修正 RoboTwin 指令生成还会调用 Python random 的问题。v2 全部 RoboTwin 记录与旧原条件附加结果保留，v3 的这两套 RoboTwin 评测重新执行；不复用旧版成绩。LIBERO 使用字节相同的 client、完全相同的 manifest、权重与配对随机流，继续全部有效记录。[deployment-audit.json](results/2026-10-10_rawvla_pi05_full_baseline/deployment-audit.json)补充相机、服务配置、submodule 和审计器哈希。

本轮数值/协议/进程检查 17 项通过，本任务文件静态与格式检查、最终全仓 `make check`（129 个 Python 文件）通过。期间并行新增的其他任务文件曾阻挡全仓检查，其后复核已通过；未改写该文件。它们验证状态顺序、正常化、动作缓存重置、权重加载拒绝、manifest 唯一性/五域完整性、冻结文件修改拒绝与并发追加日志边界；完整云端评测尚未完成。

版本切换时，v2 的独立仿真子进程曾在父调度器退出后继续运行，形成 13 个重复身份（231 行、218 个身份；重复结果没有成功/失败分歧）。独立审计拒绝计分，完整原始 ledger 留档；两个执行均移出计分，原身份全部重新执行，不按结果择优。恢复后保留 205 个唯一有效条目。[恢复记录](results/2026-10-10_rawvla_pi05_full_baseline/libero-duplicate-recovery.json)保存原始/恢复哈希和重跑身份。新增 [进程 guard](../../src/owac/runtime/rawvla_job_guard.py)将 supervisor 信号传给所属仿真进程组，并为每次启动保留独立输入 NPZ；实际测试确认所属子进程停止、无关并行进程继续运行。

## 运行入口

服务器 `107.209.104.125:44327`，RTX PRO 6000 Blackwell 96 GB。云端正式结果根（v3；LIBERO 有效记录通过链接继续同一固定任务）：

```text
/workspace/owac/rawvla/records/pi05-full-20261010-v3/
  protocol.json
  deployment-audit.json
  libero-manifest.json / robotwin-manifest.json
  frozen/src/owac/          # 策略/调度器字节快照
  runtime/                  # 作者 client / eval 的带审计快照
  libero/<suite>/task_*.episodes.jsonl
  robotwin/episodes.jsonl / attempts.jsonl
  robotwin/robotwin-*/attempt*/
  robotwin-paper/            # 632 条原缓存条件，单独计分
  robotwin-paper-conditions.json
  recovery/                  # 完整重复记录与恢复说明
  # 模型调用输入/chunk 在 v2/robotwin-policy，case_id 含 v3 运行前缀
  audit-status.json          # 每 30 秒更新；任务结束后停止审计进程
```

```bash
ssh -p 44327 root@107.209.104.125
supervisorctl status owac-rawvla-full-libero owac-rawvla-full-robotwin owac-rawvla-paper-robotwin owac-rawvla-full-audit
cat /workspace/owac/rawvla/records/pi05-full-20261010-v3/audit-status.json
```

策略服务 `owac-rawvla-full-libero-policy`（18010）与 `owac-rawvla-full-robotwin-policy`（18015）仅监听回环地址。全部任务由 supervisor 独立运行，退出 SSH/聊天不会停止；没有创建定时任务，也没有变更旧 Arena/PolaRiS 服务器任务。若实例停止/重启，需先确认策略服务加载完成，再恢复 full 调度器；它们没有无限自动重启。服务器 filesystem 不是持久挂载卷，不 recycle/destroy；小型配置、哈希和进度同步在本地结果目录，大数组保留云端。

## RoboTwin 论文表格口径核查

论文 Appendix B 写完整 3,250 条 manifest，但 Table 2 Default PI0.5 的五域百分比对应 22/127、65/127、90/127、71/126、48/125；整体 296/632 = 46.84%。实际读取全部 632 个公开 RAW NPZ 的 metadata 后，五域计数也精确为 127/127/127/126/125。这说明表格很可能使用成功配对缓存原先分配的照明条件；**这仍是推断，作者未公开对应逐条策略评测 ledger，不能宣称已确认论文内部口径。**

632 个 dataset seed/照明域身份均在 3,250 条清单中，但全部 632 个 EV、lighting scale 与 noise seed 都不同。因此不能从完整清单结果简单筛选得到原条件分数。新增 [原条件评测](../../src/owac/runtime/rawvla_paper_eval.py)，按全部 632 个原 metadata 重设作者照明 sample 与传感器噪声 seed，单独运行和计分，不向模型输入缓存动作或专家未来状态。物理 seed、checkpoint、状态/动作正常化、图像入口和配对 flow/语言随机流与主基线一致。附加任务由 `owac-rawvla-paper-robotwin` 管理，共实际计划 **13,882 次正式执行**（主清单 13,250 + 核查 632），两套 RoboTwin 的结果分别报告。

原条件选择只依据公开元信息，不依据本机策略成绩。[条件清单](results/2026-10-10_rawvla_pi05_full_baseline/robotwin-paper-conditions.json)保留 metadata/hash、原父清单身份与差异计数；它不是论文成绩的直接复现证明。

## 比较边界与完成核验

这是作者完整 manifest 覆盖下、本机固定实现的 Default π0.5 基线。**不是已经复现论文表格数值。** Blackwell 使用 torch 2.7.1/cu128 与 OptiX，作者默认 OIDN 无法可靠运行；RoboTwin 使用原检查点零填充及作者 StarVLA 图像预处理，原 LeRobot 全量数值等价尚未建立。论文公开 RNG 序列不明，本轮采用上述明确的逐 episode 配对随机流。后续比较必须保持这些运行边界，或另建版本重新评测所有方法，不能把更换后端得到的变化归因于新 encoder。

2026-10-14 已完成全部有效唯一条目、逐任务/逐照明域/SOG/Long/Avg 汇总及成本审计，完整结论见[报告](2026-10-14_rawvla_pi05_complete_evaluation.md)。同步 model forward 与 episode 墙钟分别报告，共享 GPU 的运行成本不当作独占设备的算法成本。
