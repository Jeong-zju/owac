# resume notes (2026-10-16T04:10:33Z)

- resumed from: `/workspace/owac/research/n04-v1/runs/v1.1/codex-s2/libero-D-seed101/checkpoint-step0000010.pt` at step 10
- loss_definition: `v1_1_s2_live_forecast_observation_vicreg_v3`
- loss 定义未变
- implementation_version: `n04-v1.1-s2-repair-v3`
- forecast 源端活梯度、未来目标停梯度；辅助项包含所有梯度帧，按实际有效项归约。
- 表征方差在固定 query 上跨观测计算；v3 与历史损失曲线不可直接比较。
- memory: `{"gradient_checkpointing": false, "raft_resize_hw": [128, 128], "batch": 8, "window": {"burn_in": 16, "gradient": 32, "stride": 16, "supervision_points": 4}, "forecast_anchor_stride": 1, "per_supervision_point_backward": false, "per_segment_term_gradients": true, "max_forecast_delay_frames": 4, "micro_batch": 1}`
