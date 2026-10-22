# OWAC-D1.1 RAW教师与RAW学生：实际框架与单臂蒸馏方向

2026-10-21 最新授权：用户明确“可以开始充分蒸馏，并给我一个简单的监控页面来实时查看进度和效果。”本阶段仅C单臂RAW→RAW投影后视觉token蒸馏，从真实C50续训，累计最多5000更新／80000 anchors；新阶段最多64 GPU小时／20GiB。优化器、scheduler、RNG、sampler及累计50步必须恢复；只更新现有stem＋gate，继承视觉与后端冻结。按固定验证集token MSE＋0.1cosine选型，动作误差仅诊断；无B/D扩展、视觉解冻、闭环或弱光授权。当前C51真实首步已独立验收，连续蒸馏已启动并真实恢复到后续更新；实时进度见[本地监控](http://127.0.0.1:8878/)，每5秒刷新。验证token损失每250步检查；后台每20分钟核查，正常无异常时保持安静。闭环0，控制恢复未证明。派工与运行收据：`/home/jeong/zeno/owac/repo/outputs/raw-distillation-c-full-20261021-123313`。此条取代此前“暂不启动”的当前状态，历史结果不变。

2026-10-21。用户明确：先充分蒸馏RAW学生，不再开展多轮B/D对照；随后明确“暂不启动，先看教师和学生框架”。本文件说明当前实际实现，**没有启动新训练或修改冻结范围**。此前B/C50 pilot结果保留。

## 总体定义

同次采集、同源示范、同相机、同一端点及其因果RAW历史分别进入教师与学生。两者自己的递归状态不要求数值相同，但历史来源／reset语义必须对应。学习目标是教师经过原多模态投影、实际送入π0.5后端的视觉tokens，不是RGB重建或教师动作。

```text
                     同一RAW观测与因果历史
                      /                \
                     /                  \
     RAW教师视觉前端＋原SigLIP       RAW学生前端＋继承SigLIP
                 |                         |
       教师投影后视觉tokens Z_T     学生投影后视觉tokens Z_S
                 |                         |
                 +---- 蒸馏：Z_S逼近Z_T ----+
                 |                         |
       冻结π0.5语言／动作后端       相同权重／语义的冻结后端
                 |                         |
              教师动作                   学生动作
```

这里的后端是π0.5视觉编码之后的语言／动作部分；视觉tokens与语言指令、机器人状态共同参与动作生成。不是将tokens直接当作7维动作。

## RAW教师：允许内部成像

```text
每相机的6帧因果RAW burst ＋ 该相机教师历史状态
  → 作者RawVLA RAW时序处理／Neural ISP
      RAW去噪、场景条件、递归ISP状态与成像参数
  → 教师内部RGB（当前端点）
  → 原策略RGB预处理、resize至224×224
  → 原SigLIP patch embedding（14×14 / stride14）
  → 原位置编码
  → 原SigLIP 27个Transformer blocks ＋ 原末端归一化
  → 原多模态投影：1152 → 2048
  → 教师视觉tokens Z_T
  → 冻结π0.5语言／动作后端（同语言、状态、采样协议）
  → 动作
```

教师的外部输入始终是RAW。内部RGB是其计算中间结果，不是从配对RGB数据直接替代RAW输入。教师固定eval、停止梯度；缓存的Z_T来自这条真实RAW路径，缓存键绑定RAW因果历史、教师状态、checkpoint、相机、预处理与端点。

当前实际选用的教师是**本地作者架构初始化0前端＋实际LIBERO π0.5 checkpoint**，前阶段20/20仅正常光单任务开发筛查；不是作者公开训练后前端权重，不将未选中的155次教师更新称为能力收益。

## RAW学生：直接产生非图像表征

```text
同一相机、同端点的6帧因果RAW ＋ 学生历史状态
  → RAW校准／有效性处理、同视野256→224
  → 对每帧应用共享RAW stem：
      Conv3×3：3 → 64，stride1
      GELU
      Conv14×14：64 → 1152，stride14
      得到16×16个1152维patch embeddings
  → 六帧对应patch按有效mask求均值 m_t
  → 按相机独立的门控历史融合：
      h_t = (1 − sigmoid(g)) · m_t + sigmoid(g) · h_(t−1)
      首窗口以有效输入初始化；新episode必须reset
  → 原位置编码
  → 继承当前策略的SigLIP 27个blocks ＋ 原末端归一化
  → 继承的原多模态投影：1152 → 2048
  → 学生视觉tokens Z_S
  → 相同权重／语义的冻结π0.5语言／动作后端
  → 动作
```

这是**当前已实现的时序模块**，不是任意attention／GRU网络。学生不生成供SigLIP重新编码的RGB图像；stem输出已经是patch embeddings。相机状态独立存放，1152维gate参数按通道共享使用。端点stride5；训练用当前权重burn-in完整源前缀，然后展开6端点，仅最终端点监督。burst长度和展开长度两个6不等于36个独立观测。

当前数据为三通道LIBERO逆处理pseudo-RAW，学生输入3通道；真实Bayer校准／四通道打包是独立入口，不代表本轮已经验证原生RAW。

## 蒸馏位置、尺寸与冻结

| 项目 | 当前实际设置 |
| --- | --- |
| 活动相机 | base_0、left_wrist，固定顺序 |
| 每相机tokens | 16×16＝256 |
| 投影前宽度 | 1152 |
| 投影后宽度 | 2048 |
| 活动视觉接口 | 每样本512×2048，附原相机／位置／mask语义 |
| 教师 | 全冻结、eval、无梯度 |
| 学生可训练 | RAW stem四个张量＋门控g，共14,454,784参数 |
| 学生继承部分 | 同实际checkpoint的SigLIP blocks／位置／归一化／原投影，首轮冻结 |
| 语言／动作后端 | 冻结；需要时保留对学生输入的梯度通路 |
| 学生部署依赖 | 学生artifact、冻结后端、tokenizer、统计量；无需教师、ISP或教师缓存 |

失活right_wrist保持原策略mask语义；上述512是两个活动相机的token数。逐位置监督对应同物理视野，学生224网格的位置回映至原256视野；不随意插值教师tokens。

```text
L_distill = masked_FP32_MSE(Z_S, stopgrad(Z_T))
          + 0.1 × masked_cosine_loss(Z_S, stopgrad(Z_T))
```

MSE按有效token数×2048归约，cosine按有效token数归约；不做逐样本幅值归一化来隐藏尺度差异。当前没有RGB重建损失，没有动作监督损失。真实专家动作误差是下游观察指标，不是这个蒸馏目标。

## 后续只做C单臂充分蒸馏

停止把B时序消融、B/D多轮对照作为下一步默认工作。保留历史对照证据，后续主任务仅C：同RAW输入、同教师目标、现有冻结范围、同40/5/5源拆分，从实际C checkpoint50恢复optimizer／scheduler／RNG／sampler。

“充分”不能由50更新或训练批次loss下降判定。下一阶段应连续训练，在固定留出集评估投影后token MSE／cosine，以验证token损失选择蒸馏checkpoint；动作MAE仍记录，但不因为其短期变差而转方案、提前挑回checkpoint0。原native动作选型记录不重写，新阶段需独立记录选型语义。当前CLI仍按动作选型，**尚未把后续token选型实现冒称已完成**。

候选上限累计5000更新（历史50计入）、最多80000 anchor呈现；不保证这个上限即收敛。按C已测50步2014秒的更新耗时外推，余下4950步约55.4GPU小时，连同验证／加载预计56–60小时；建议新阶段硬上限64GPU小时、新输出20GiB。此为预算提案，**用户本次明确暂不启动**，没有签发新authority，没有训练／闭环／视觉解冻。

## 对应实现与证据

- 教师构建：`src/owac/training/rawvla_teacher.py::build_author_teacher`；作者Neural ISP位于实际固定源码的`baselines/rawvla/rawvla.py`。
- 学生局部stem：`src/owac/baselines/raw_stem_siglip.py::LocalConvStem`。
- 学生时序：`src/owac/baselines/raw_temporal_siglip.py::TemporalRawStemStudent`，`_patch_burst`与`_advance`。
- 同RAW配对与缓存：`src/owac/raw/temporal_distillation.py`。
- 实际pilot独立证据：[B/C50独立验收](../../research/experiments/raw_stem_siglip_20261020/pilot-20261021-independent-review.md)。
