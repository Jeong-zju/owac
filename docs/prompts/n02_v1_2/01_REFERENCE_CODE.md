# N02 v1.2 论文与实现导航

<!-- BEGIN DEEPSEEK KEY ROLES -->
> **DeepSeek 编程会话、权限与 key 规则（2026-10-21，适用于本文全部启动命令）：**
> 编程默认完全权限 `danger-full-access`，审批策略 `never`：已授权任务内的文件、命令和工具操作直接执行，不要求用户逐项人工审批。新建与续接后核对实际权限；已有会话仍为受限模式时，由执行者在该可见会话发送 `/permission danger-full-access` 或操作 Permissions 切换并确认生效，不交给用户点击批准。完全权限保持既定研究范围、训练预算和停止点。
> 编程必须使用用户可见的 Harness Web 专用对话：先打开页面、核对工作区／模型／coding 凭据，再在该对话发送工作单；回复、工具调用与返工都在同一可见会话中进行。禁止 headless、stdin／JSON 一次性后台编程派工或只交日志；正文旧启动模板如有冲突，以本规则为准。
> 编程／Harness 使用 `DEEPSEEK_CODING_API_KEY`；闭环在线推理／规划／决策使用 `DEEPSEEK_CLOSED_LOOP_API_KEY`。编程代理启动闭环子进程时，子进程必须切换到 closed-loop 角色。
> 启动前使用 `owac.runtime.deepseek_credentials --role coding -- <编程命令>` 或 `--role closed-loop -- <在线闭环命令>`；Python 调度使用 `environment_for_role(...)` 注入子进程环境。私有 key 只从本机对应角色文件读取，不回退或混用。
> 具体绝对路径与命令见 [凭据使用说明](../../deepseek-credentials.md)。明文 key 不写入提示词、源码、配置、日志、网页或 Git；本规则不恢复历史阶段或扩大实验授权。
<!-- END DEEPSEEK KEY ROLES -->

## 直接参考论文与实现

先读当前实现再读对应原模块。已有P0移植和许可证保留，直接复用；需要适配时解释差异，不另写无来源的小网络，也不因上游主分支更新重写已验收组件。

- ConvNeXt V2：<https://arxiv.org/abs/2301.00808>；官方Base骨干、GRN与FCMAE：<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/convnextv2.py>、<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py>、<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/fcmae.py>；训练循环<https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/engine_pretrain.py>。读取Base depths/dims、GRN、mask与loss，不用RGB权重做隐形教师。
- SAVi / SAVi++：<https://arxiv.org/abs/2111.12594>、<https://arxiv.org/abs/2206.07764>；官方predict/correct与slot注意力：<https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/video.py>、<https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/attention.py>；decoder <https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/decoders.py>。复用P0 PyTorch移植，不为了运行SAVi改成JAX训练全系统；对象槽不保证自然获得语义分割。
- Perceiver IO：<https://arxiv.org/abs/2107.14795>；官方attention/latent/decoder <https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/perceiver/perceiver.py>，位置编码同目录`position_encoding.py`；本仓`src/owac/backends/rawvla_features.py`是已存在策略桥；`src/owac/frameworks/n02/readout.py`是旧token readout，并非Perceiver。请在`n02_full/readout_v12.py`中移植官方Perceiver核心注意力/latent/query读出，默认512宽、128 latents、每相机256查询，再投影2048维；只对新增移植做一次必要数值/梯度对照，保留许可证，不重做P0全套。
- RawVLA论文、benchmark和官方固定实现：<https://arxiv.org/html/2609.37530v1>、<https://github.com/ShuhongLL/RawVLA/tree/ba9c56b04da2c4931b70d8844710f5928d43775b>、<https://huggingface.co/datasets/ToferFish/RawVLA-Bench/tree/8586c594fff9ce38580e29745f074e82e96e74f7>。主源码`experiments/simulation/{libero,robotwin}`，同源ISP查该提交实际实现。
- StarVLA NPZ loader和π0.5：<https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/dataloader/rawvla_npz_datasets.py>、<https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/framework/VLM4A/PI05.py>、<https://github.com/ShuhongLL/RawVLA/blob/57d508cb180d0510ac9060358009c0f6ffaac420/starVLA/model/modules/action_model/OpenPI_ActionHead.py>。关注prefix、flow-matching、padding、动作和normalizer，不经过PIL把RAW量化成RGB。
- 官方OpenPI固定PyTorch：<https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py>；π0.5论文<https://arxiv.org/abs/2504.16054>；RoboTwin原权重<https://huggingface.co/SidneyXie/pi05_robotwin/tree/e49e2ab6c11f07511573b67261bd129e88d0a416>。
- RAW噪声/处理：Unprocessing <https://arxiv.org/abs/1811.11127>；RAW-Adapter <https://arxiv.org/abs/2408.14802>；Dr.RAW <https://papers.neurips.cc/paper_files/paper/2025/hash/20eadb6338a53114a6371f102c5ded95-Abstract-Conference.html>。它们界定已有工作，不证明当前模块有RAW能力；Dr.RAW当前来源不能冒称可直接执行的完整代码。
- 多假设强近邻SWB <https://proceedings.mlr.press/v139/singh21a.html>；广义更新理论<https://arxiv.org/abs/1306.6430>；校准条件<https://arxiv.org/abs/2202.13415>；attention解释边界<https://aclanthology.org/N19-1357/>。
- 可选连续优化参考Theseus：<https://github.com/facebookresearch/theseus/blob/c8583de41824613fb135ee9bad0e930ded6be404/examples/simple_example.py>；状态重执行Gen <https://www.gen.dev/docs/stable/>。只参考必要机制，不把安装两套求解框架当P2/P3/P4前置。
- 本地组件来源：`src/owac/frameworks/n02_full/components/README.md`、`configs/encoders/n02_full/third_party.lock.json`。更多固定文件：`docs/prompts/n02_v1_1/02_REFERENCES.md`；只读源码导航，不执行该历史包的阶段要求。

