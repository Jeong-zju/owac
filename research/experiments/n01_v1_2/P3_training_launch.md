# N01 v1.2 P3：完整模型训练与简洁监控

2026-10-17。用户最新授权“接入训练……充分利用Pro 6000”取代本会话此前的禁止训练要求。旧审计锁与旧检查点保留，新运行独立从初始化开始。工作树基于 `d3dec908d78d0a86a761abc985b83113d836c8cf` 加完整未提交修改；训练源码以远端 `SOURCE_MANIFEST.json` 的逐文件SHA256为准。

## 运行和观察

- 用户入口：http://127.0.0.1:18080/ ，每5秒更新。两张卡片分别显示RAW和同源RGB的状态、步数、秒/步、预计剩余时间、固定验证误差及GPU工作强度/显存/功耗。曲线按 `(name, role)` 分开。新运行根没有旧实验曲线。
- 新运行根：`/workspace/owac/rawvla/research/n01-v1/runs/v1-2-p3-complete`。`source/` 冻结源码、配置、数据拆分清单；`outputs/` 独立权重；`monitor/` 当前状态和带role指标；`receipts/` 完成收据。
- GPU0：LIBERO `raw_n01` → RoboTwin `raw_n01`；GPU1：LIBERO `rgb_n01` → RoboTwin `rgb_n01`。seed101，每run计划50,000个真实动作优化器步；Base沿用既有决定留至P4。50k是预算，不是收敛判据。
- 32帧、原生LIBERO256×256/RoboTwin240×320、完整SEA-RAFT与DBSR；不通过缩图/缩历史获得速度。神经网络BF16，物理观测/支持数值FP32，动作后端可训练范围使用FP32主权重以保存1e-6量级更新。
- 每卡物理batch2、累积2次，有效batch4；独立运动对batch8；内容分块4、两级activation checkpoint、CPU预取2批、CUDA fused AdamW。编码器/桥/同域已知变换辅助项先训练，2000步后按配置开放 `action_head.`。
- 首先测8个固定验证样本作为第0步起点，第10步及每250步复测；另有4个可复现随机验证样本。每100步保存 `latest.pt`，启动第2步额外保存；恢复包含优化器、RNG、已消费采样位置和阶段。重启最多重算最近100步以内；没有承诺零重算。

监控中的验证误差是固定离线小子集的动作flow-matching损失，越低越好；它不是成功率，也不替代全分布验证或机器人闭环。随机运动可能使历史全部被拒，已知变换辅助监督不等于自然运动GT。风险校准、完整训练效果、四臂增量和新闭环结论仍待实证；旧“0/96留出成立”的撤回不变。

## 加速实测与显存

两张卡均为RTX PRO 6000 Blackwell Server Edition，97887 MiB/600W。性能探针在独立目录，真实原生32帧、全部相机、真实后端和动作损失，包含后端适配阶段的反向。批量显存探针重复同一个真实样本，只用于计算/容量，不作为学习结果。

| 平台 | 优化前FP32 batch1预热后 | 优化后FP32 batch1 | 优化后BF16 batch1 | 优化后BF16 batch2 | batch2全内容+Adam峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LIBERO | 7.875 s | 1.763 s | 1.651 s | 2.412 s | 72.82 GiB |
| RoboTwin | 11.864 s | 2.512 s | 2.576 s | 3.635 s | 83.75 GiB |

前四列是单次前向/反向，不含数据读取和正式梯度累积；不能当作正式秒/优化器步。优化包括把独立历史对应成批计算、跳过对融合完全不可用来源的内容CNN，保持reference fallback/输出/梯度语义；所有32帧仍参与运动与支持。初始对应无用时跳过内容较多，随训练接受更多历史，耗时/显存会变化。全内容压力路径强制编码全部来源，连续2次真实Adam更新，梯度有限；LIBERO/RoboTwin被测后端FP32参数最大改变分别约1.999e-6/1.998e-6。两轮探针共6次隔离更新，全部丢弃，不迁移到正式权重。

首轮发现BF16本身对RoboTwin不提速，主要瓶颈是逐对运动调用和无效内容编码，故采用上述结构等价优化；最终BF16 batch2提供更高每样本吞吐，并为全内容最坏情况保留约12GiB显存。启动后的实际速度、数据等待比例、显存和效果以JSON启动收据及实时页面为准。

## 持久运行与故障边界

`owac-n01-v12-complete-raw/rgb` 由supervisor运行，`autostart=true`，有限队列 `autorestart=false`，避免失败后无休止耗时重试。单项失败继续下一平台，最终非零；正常重启时按匹配完成收据跳过已完成run，其余从各自 `latest.pt` 续训。`N01_SOURCE` 显式指向冻结快照，工作目录 `/tmp`，避免旧根包副本遮蔽源码。数据/权重预检覆盖两平台train/validation；不宣称未来真实handover已发生。

监控服务 `owac-n01-v12-p3-monitor` 只绑定服务器loopback18080，自动重启。用户机 `owac-n01-monitor-tunnel.service` 开机/断线自动重连，user lingering已启用；无登录也能运行。关闭网页、退出SSH或本机断网均不会停训练。没有新增定时训练、消息推送或P4全量评测。

第一次启动因快照遗漏4.7MB数据拆分清单，在模型加载/优化器更新前失败，4个worker状态和第一版manifest保留于 `launch-attempt-1-no-updates/`，日志原样保留。补齐清单、加强预检后生成部署revision2再启动，当前309个源/配置/清单文件可逐一复核；后续开发不覆盖运行源码。

## 验证与复核

- `make check` 通过；本地全仓 **879 passed / 6 skipped**（本机CUDA不可用、作者ISP源码不在本机）；远端定向 **15 passed / 0 skipped**，包含CUDA断点一致性和作者ISP逐元素4项。额外运动batch等价/梯度及重启完成跳过测试通过。
- 真实端到端正式训练的更新、检查点和第0/10步验证由 `P3_training_launch_receipt.json` 记录；离线测试绿灯不是其替代品。
- 原始探针：`.../runs/v1-2-p3-launch-20261017/{libero,robotwin2}-benchmark.json` 及旧版before-pair-batching文件。小型摘要已写入本目录JSON，原始产物保留于被忽略的本地outputs。

```bash
ssh -p 34302 root@154.59.156.21 \
  'supervisorctl status owac-n01-v12-complete-raw owac-n01-v12-complete-rgb owac-n01-v12-p3-monitor'
curl -fsS http://127.0.0.1:18080/api/status
curl -fsS http://127.0.0.1:18080/api/metrics
```

如需人工停止，只停止上述两个complete训练program；不要停止服务器管理supervisor、Caddy、portal或tunnel_manager。旧 `owac-n01-v12-p3-raw/rgb` 继续STOPPED、autostart=false，旧main audit hold保留。

## 正式启动快照

- `libero-raw_n01-seed101`：实到15步，最近约8.53秒/优化器步；第10步固定验证 0.649071 → 0.625444（下降3.64%），第2步完整检查点已验证。
- `libero-rgb_n01-seed101`：实到22步，最近约6.20秒/优化器步；第10步固定验证 0.601568 → 0.598710（下降0.47%），第2步完整检查点已验证。

两臂第2步各已消费8个真实训练样本，检查点约3.04GB，341组优化器状态；对同seed初始化逐项比较，两臂内容编码器各56个参数张量改变、桥分别6/7个、运动分别276/274个。运动变化包含已知变换辅助项，不能归因于自然对应能力。40秒GPU采样均值79%/69%，数据等待低于1%；不是恒定满载保证，也不是最终吞吐分布。上述为启动快照，后续进度以页面为准。

## 2026-10-17 13时监控链接修复

服务器训练和监控进程持续运行，两个训练PID未变；本机SSH转发在13:17:16被远端关闭，3次历史断线均有systemd日志。本次本机转发已恢复，额外缩短SSH建连超时至10秒、keepalive为15秒×2、重试间隔3秒；网页请求增加15秒AbortController超时，finally释放刷新锁并终止未结束的并行请求，后续5秒周期可继续重试。只修改本机隧道与可变监控HTML，未重启训练或覆盖冻结训练源码。HTTP与浏览器显示实时状态均已复核。

链路复测发现94KB曲线数据经SSH转发约20.14秒、服务器内仅0.017秒。开启SSH Compression并使用IPQoS=none后，转发实测约4.76秒；因此采用15秒网页超时，避免把正常慢链路当作断开。浏览器仍每5秒尝试刷新、避免重叠请求；网络不保证固定延迟。
