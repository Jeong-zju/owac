# N01 按用户要求停止训练

2026-10-19 15:46（北京时间），用户明确要求“停止训练”。已停止两个 `owac-n01-v12-complete-raw/rgb` supervisor program及其训练进程组；两队列均STOPPED，后续RoboTwin不启动。取消autostart（autorestart原为false），已reread/update对应program；运行根增加持久 `P3_AUDIT_HOLD.json` 和 `USER_TRAINING_STOP.json`。只有用户重新授权后才能解除停止状态。

实际检查 `/proc` 无N01训练或评测控制器进程，`nvidia-smi --query-compute-apps` 为空。既有管理服务和监控服务保留，不修改已冻结的训练/评测源码。

最后观测训练进度RAW 20,974、RGB 21,342步；这是停止前worker最后上报值，不等于最新已落盘检查点的步数。两份 `latest.pt` 各6,484,845,221字节仍保留；停止未触发额外checkpoint，尚未保存的更新不作可恢复承诺。已固定用于小测的20,800/21,100步权重、24次闭环原始结果和视频全部保留。

停止前worker状态及supervisor配置备份在远端运行根 `receipts/user-stop-20261019/`；机器可读核查见 [停止收据](P3_training_stopped_20261019_receipt.json)。运行根为 `/workspace/owac/rawvla/research/n01-v1/runs/v1-2-p3-complete`。

监控worker状态以停止后外部进程核查为依据标为 `stopped`，原始停止前快照已备份；状态不表示训练预算完成。监控页明确显示“训练已停止”，不再把最后训练速度显示为当前速度。原先闭环收据中的“已恢复训练”是当时事实，后续已由本次用户停止取代。
