# RawVLA 作者模型与权重核查

2026-10-20，用户明确要求 RAW 输入的已训练 RawVLA 教师，并撤销固定单帧限制。此前 Default ISP＋pi05 复现保留为历史诊断基线，不替代神经 ISP 教师。

官方 [RawVLA 仓库](https://github.com/ShuhongLL/RawVLA) main 当前 commit `ba9c56b04da2c4931b70d8844710f5928d43775b`。GitHub API 完整递归 tree 为153项，非截断；没有前端模型权重文件，Releases、tags、issues 均空。模型 [baselines/rawvla/rawvla.py](https://github.com/ShuhongLL/RawVLA/blob/main/baselines/rawvla/rawvla.py) 已公开；本地参考源码与官方 Git blob `7601fb8c86eebd693ae99e21a2000c23da5b4627` 完全相同（SHA256 `404daa204da7c81b6f508bd41f68980e29be98cccdcd0f367e453ad7ae11ebb0`）。这证明源码来源，不证明训练结果。

[checkpoint 工具](https://github.com/ShuhongLL/RawVLA/blob/main/tools/checkpoints/download_starvla_libero.py) 下载 StarVLA/Qwen2.5、Qwen3骨干及基础VLM，没有 RawVLA 神经 ISP checkpoint；LIBERO/pi05 配置引用冻结官方pi05骨干，不能当作训练好的完整 RawVLA。作者 [Hugging Face 主页](https://huggingface.co/ToferFish) 当前公开models数量为0，RawVLA-Bench是数据集。直接HF API在宿主网络连接重置，网页核查成功；不能把网络失败当不存在。结论是当前公开入口中未找到所需前端权重，不断言私人或未索引渠道中不存在。

作者默认 [rawvla.yaml](https://github.com/ShuhongLL/RawVLA/blob/main/configs/base/rawvla.yaml) 为burst_frames6、rnn_frames6；模型含GRU状态和显式ISP状态。用户允许时序教师与学生，无需为单帧而强改配置；实际历史跨度还取决于采样stride，不能把两个6简单等同为6个相同观测。未选定有效权重时，不伪造可用教师。后续蒸馏必须声明历史、因果性、时间相机对齐及reset；历史范围不同须单独报告。

宿主正在按原成功基线相同generation和SHA256恢复官方pi05后端对象（总12,439,085,481 bytes），这是后端资产恢复，不是RawVLA教师准备完成。未训练RawVLA前端，未蒸馏，未新增闭环；教师训练复现仍需具体新授权。

原始公开查询收据：`outputs/raw-distillation-20261020-sol/rawvla-author-release-probe.json`；完整基线账本复核：`outputs/raw-distillation-20261020-sol/baseline-reference-audit.json`。


## 2026-10-20 后续取得核查

用户本轮要求先取得有效教师，无权重时另行制定并授权复现。已补查6个公开分支、main32次历史提交，均未见训练后的RawVLA前端；RoboTwin子模块的empty_lang_embed.pt是语言常量，不能冒充教师。项目页、论文、作者HF主页及数据集顶层附件也未提供所需前端下载。HF宿主API访问失败与网页成功分别记录，不以网络错误证明不存在，也未穷尽所有嵌套数据目录。[完整收据v2](teacher-public-search-v2.json)。

已下载完整作者main源码（2811790bytes，SHA256 `57bc54e8a695ae07c2ffee9ebb2549e9efbb48536a97df14c5c8f63c52f3e012`），原本地参考snapshot缺base配置的判断已撤回。实际官方config_loader合并证实burst6／rec6／stride5、horizon10、flow10；公开配置action系数.15、brightness.01、chroma.001、weight decay1e-8，与论文D1/D3有部分差异。源码保留六步BPTT，不强改单帧。

上文“正在恢复”的同源pi05后端现已下载、转换、重载验收，详见[恢复收据](backend-recovery.json)。**完整已训练RAW教师仍未取得；没有新增训练或闭环。** 已形成[单任务教师复现提案](../../../docs/plans/RAWVLA_TEACHER_REPRODUCTION_PROPOSAL_2026-10-20.md)，执行仍待用户明确授权。
