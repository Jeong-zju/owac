# N01 v1.2 面向 VLA 的 RAW 时空证据编码

日期：2026-10-15。状态：按用户要求正式修订技术路线。适用分支：`lzh/n01-evidence-integration`。当前P0—P2资产保留，v1.2 P3待实施；本次更新方案、阅读版、控制台和编程交接材料。

N01研究的是：**如何利用RAW测量中被具体RGB成像管线削弱的行动信息，在空间细节、噪声降低与响应速度之间组织时空支持，直接形成VLA可用的非图像表征。** 来源撤回与状态重算是维护这些证据的手段，任务成功和动作恢复才是最终结果。

执行入口：[v1.2完整阶段提示词](prompts/n01_v1_2/README.md)。P3一次建立完整模型、真实数据训练和每轮100次开发闭环；P4冻结全量验证；P5按已知限制开展原生RAW、跨域或架构扩展。每平台、每核心方法只训练1个种子，默认101，不自动追加211/307。

## 1 为什么从 v1.1 修订

v1.1把支持修订、来源删除、有限状态和成熟融合整体纳入P3，解决了局部代理实验长期替代完整模型的问题。但N/B0/B1均使用RAW，主要回答“支持修订是否超过成熟融合或标准reset”，不能单独归因RAW相对RGB的信息收益；删除一致性本身也适用于RGB。

v1.2保留完整建模和已有资产，把主问题明确成一条可检查的因果链：

```text
同源采集观测中实际保留了什么
    → 固定ISP改变了什么信息或统计结构
    → 编码器保留/积累了什么行动相关证据
    → VLA动作、任务成功和事件后恢复怎样变化
```

这不是要求算法只能用于RAW。算法具有通用性可以保留；论文必须区分输入信息、通用时序机制、预训练与训练机会各自的影响。v1.0/v1.1方案和P0—P2结果作为历史保留；新路线不回写原失败，也不宣称已有RAW信息增益或闭环能力。

## 2 核心假设与相关工作边界

### 2.1 RAW 优势必须具体到测量条件

| 实际条件 | N01需要利用的东西 | 对行动的可检验作用 |
| --- | --- | --- |
| 经校准的近线性信号、信号相关噪声 | 测量尺度、可靠性与跨帧协方差，而非仅特征相似度 | 暗处弱边缘/小目标判断是否更稳定 |
| 显示处理前仍存在的亮暗差异 | 避免过早裁剪、量化、去噪或归一化抹除任务差异 | 逆光、暗亮交界任务是否保留有效信息 |
| 多帧中独立或部分互补的观测 | 在正确对应下积累；变化后停止不再支持当前结构的历史 | 降噪收益是否抵消运动偏差与反应滞后 |
| 原生CFA真实采样与相位，限确有此输入时 | 在实际采样位置形成测量支持，避免把插值值视为独立测量 | 细结构/采样互补是否改善定位和行动 |

RAW不是额外的深度/光谱传感器；更高位深不自动等于更高有效动态范围。RGB可用高精度保存，部分ISP变换也可逆；优势只针对明确的处理链和数据条件。传感器已饱和、量化前已丢失或根本未采集的内容，不能由编码器保证恢复。[R34–R35]

### 2.2 拟新增的规则

完整编码器在成熟内容通路内，用测量噪声、实际坐标、对齐风险与沿轨迹的变化证据组织区域/时间支持，让噪声降低与陈旧偏差的取舍真正改变内容融合和任务读出。动作监督学习有用的内容与读出；支持选择须有实际梯度路径、校准依据或可检查的解析规则，不能只把action loss加在网络末端就宣布支持已按行动优化。

当前拟贡献是这套规则在RAW信息利用及行动后果上的可检验增量，不是“首次RAW进VLA”“首次动作监督”“首次多帧RAW融合”或“首次可撤回状态”。DBSR及Google多帧RAW工作已研究成熟融合/采样；RawVLA已使用动作监督流式ISP；RAW-Adapter、Dr. RAW已研究RAW高层视觉与适配。[R01、R06、R25、R36–R37]

### 2.3 三种可能结果都应保留

RAW优于同机制RGB，且支持修订改善行动时，可支持相应条件下的RAW信息利用与机制价值。机制在RAW/RGB上同样有用时，通用机制贡献与RAW输入收益分别表述。RAW不优于充分训练RGB、或改进仅来自更多容量/历史/预训练时，按实际证据收窄或否定主张；不设必须提升某百分比的通过线。

## 3 数据与同源 RGB 构造

### 3.1 两平台的证据角色

主体仍使用已部署RawVLA-Bench。RoboTwin的HDR派生浮点输入优先用于检查显示处理前的信息利用；LIBERO的RGB8逆处理及量化缓存用于固定基准兼容、实际弱信号条件和一般行为，不据此声称恢复了真实传感器的原始动态范围。两者均无原生Bayer采样。[R25]

本地协议 `src/owac/raw/rawvla_sequence.py`：LIBERO为三通道`uint8`线性视图、按既定尺度转FP32；RoboTwin三通道float32沿作者单位使用。缓存未普遍提供曝光、增益、相机内外参、坏点图和实测执行动作；缺失项不可凭名称补真值。物理噪声参数有标定才称标定，没有则明确估计/proxy，不能把程序生成系数直接冒称自然传感器校准。

| 来源轨迹划分 | train | calibration | validation | test |
| --- | ---: | ---: | ---: | ---: |
| LIBERO | 1228 | 175 | 206 | 162 |
| RoboTwin | 450 | 48 | 58 | 76 |

继承 [四区索引](../data/manifests/n01-full-v1/n01-full-v1-index.json)，同源轨迹的相机、重叠窗口和成像变体不得跨区。任务模板跨区共享，不声称未见任务泛化。源轨迹划分与正式rollout身份是不同层次；作者池的任务/初态重叠按原协议披露，不默改原分母。

### 3.2 RGB 必须来自同一观测

离线配对定义为 `X_t = 本次合法RAW测量`，`Y_t = ISP_fixed(X_t)`。先在原测量上完成本条件已有的噪声/曝光处理，再由冻结ISP渲染RGB；四臂共享同一源帧、相机、采集时间、历史长度和动作标签。固定ISP复用冻结作者Default链，登记代码位置、参数、输出dtype/量化、resize与裁剪顺序。它是独立RGB对照的输入构造，不进入RAW主方法，不作RGB教师。

**不能直接取缓存配对RGB充当受损RAW的RGB版本。** 作者配对RGB可能来自正常照明；拿正常光RGB对暗光RAW混入了采集条件差异。只有逐项确认同源同条件才可复用缓存。现成冻结Default成绩只有在输入/控制/源码协议确实相同时才能复用。[R25]

图像路径优先写拟新增 `src/owac/raw/n01_paired_views.py`，输出同源`raw_view/rgb_view`与小型处理记录；原缓存只读。量化、裁剪、tone mapping、色彩、噪声相关性各按实际记录，不把所有RGB图都称有损，也不故意选极差ISP证明RAW。

### 3.3 在线公平比较

四臂按相同任务/初态/环境seed和采集/控制机会独立闭环。动作不同后物理轨迹会分叉，后续帧不再逐像素配对；“相同观测”只用于离线相同状态或首个相同状态，不能把整段独立rollout假称同帧同状态。在线RGB从该方法自己当前状态的RAW接口经同一固定ISP得到，不能离线偷读另一方法的未来画面。

原生CFA适配从P3保持可执行规格；能取得合法真实RAW时，用少量真实案例核对相位/噪声/时序行为，作为同一完整模型的输入诊断，不重开长期局部阶段。若未测真实CFA，相应能力保持未验证；P5扩展传感器/会话和行动数据，不用CV视频替代机器人能力。

## 4 完整编码器与 RAW 信息通路

### 4.1 实现起点与规模

复用 `frameworks/n01_full/`、`raw/rawvla_sequence.py`、`backends/raw_correspondence.py`、`backends/rawvla_features.py`。内容通路默认完整DBSR：宽64、9残差块、512维特征，现有成熟`MaskedDbsrFusion`；对应默认已适配完整SEA-RAFT；读出使用现有π0.5前缀桥。P2运动模型是有局限的初始化/对照，真实大运动与风险外推问题在完整任务中处理。[R01、R07]

采用新网络必须有作者模块、规模、许可证和实际适配说明。BSRT-L/Burstormer作为有证据的成熟替换，Mamba/长记忆按瓶颈需要引入；不为名称新颖而添加手搓小网络。[R02、R04、R13]

### 4.2 信息怎样真正进入表示

```text
合法RAW及采样/噪声条件
  ├─ 测量通路：FP32校准、噪声/相关性、真实坐标、缺失/饱和
  └─ 内容通路：逐来源成熟特征编码、因果运动对应
                 ↓
      测量相容性 + 变化证据 → 区域/时间支持
                 ↓
      支持改变内容、融合参考、权重归一化和必要的重算
                 ↓
      保留任务细节的非图像读出 → π0.5语言/动作后端
```

共同测量输入、合法metadata和校准机会对同域Base/N01共享；不能让Base少看额外通道而把信息输入差异全算成修订收益。内容特征与可靠性估计分别处理，attention不解释成物理精度或后验概率。来源按实际episode/camera/frame/phase/坐标去重，重叠窗口和策略多次denoising不是新观测。

RAW不做逐帧最大值归一化以抹除绝对测量尺度；所有增益变换同步处理噪声尺度。保留真实相位/颜色，不先平均成亮度再声称完整RAW利用。若没有相应metadata，可用有来源的估计并说明局限，不设置等待全部物理标定才允许训练的新关卡。

### 4.3 空间细节、运动与时间支持

在真实映射后的采样位置使用局部常数/仿射GLS及域级相容性；避免A–B、B–C连通就无限合并整域。沿对应轨迹使用作者BOCPD/鲁棒BOCPD参考，区分稳定测量、对齐不确定、单帧异常、持续结构变化、缺失/坏点/饱和。速度不是删除依据，残差也不自动表示物体变化。[R16–R17]

噪声减少必须与空间边缘混合、错配和历史滞后一起考虑。支持选择实际作用于特征内容、融合参考、logit/normalization及其感受野；不能只改最后mask。区域级实现可以使用保守感受野重算，声明精确范围，不假称任意像素级独立。

### 4.4 读出不能再次丢掉任务细节

保留现有512→2048、每相机256个16×16带mask token作为可运行起点。正式训练检查窄边/小目标信息到达读出后是否仍影响动作；仅输入端保留RAW而池化后完全失去细节，不能算RAW信息已被利用。需要改读出时复用成熟resampler/多尺度特征方法并同步四臂读出机会，不自行堆弱化网络；没有必要则保留已有桥。

语言/当前合法任务上下文可用于读出，未来专家动作和隐藏目标真值不作输入。RAW主方法替换RGB图像塔，无RGB/sRGB/XYZ中间图像，无RGB教师/重建；独立RGB四臂对照中的RGB输入不违反主方法边界。

### 4.5 有界记忆、撤回与期限

默认每episode/camera最多32个唯一实际到达帧，完整有限窗口重算优先。BOCPD和数值状态从同一窗口固定初值重建，不能夹带更老后验；特征缓存不跨optimizer参数更新。删除窗口内源后不偷偷从窗口外补更老帧。

结构支持退出允许保留“曾发生变化”的上下文，不等于整个输出与该源独立。完整来源删除必须重建所有内容、对应、支持、参考、归一化、状态和读出依赖；固定决策干预与完整反事实分别报告。删除检查用于保证证据通路可信，不作为独立任务性能贡献。

流式reset、缺帧、重复请求、同帧多次policy读取与批量参考语义一致。原子提交合法读出；期限到达返回标时的最近合法快照或预定有效降级，不能用合法但陈旧快照掩盖反应滞后。包含采集/队列/搬运/flow/支持/重算/后端/动作提交成本，报告实测p50/p95/p99，不承诺GPU硬实时。

## 5 四臂核心比较与成本

### 5.1 一次核心矩阵

| 方法ID | 输入 | 编码/融合规则 | 作用 |
| --- | --- | --- | --- |
| `raw_base` | 合法RAW | 完整成熟内容/融合与同域校准 | RAW上的强机制基线 |
| `raw_n01` | 同样RAW | 完整N01测量支持修订 | 主方法 |
| `rgb_base` | 同源RAW经固定ISP的RGB | 匹配成熟内容/融合 | 控制历史、容量、训练机会的RGB基线 |
| `rgb_n01` | 同样RGB | 对RGB统计合理适配的支持修订 | 检查机制是否主要是通用时序收益 |

两个平台为LIBERO、RoboTwin，四个ID是方法，不是四个平台。每平台每臂1个训练种子，默认101，主体共8个训练run。由v1.1六个run改为八个，是以四臂归因矩阵替代三臂全RAW主矩阵；标准检测/reset原B1移为有需要的补充，不默认再训练第五臂。多种子为后续选择，不作本阶段门槛。

### 5.2 RGB 对照的合法性

RGB臂使用同规模成熟内容、token预算、历史、任务上下文和π0.5适配机会；RGB统计在RGB域估计，不能机械套RAW泊松噪声。每个臂的motion/支持只根据该臂可见输入计算；不能把RAW计算的flow、逐像素可靠性或特征暗中提供给RGB。同一来源标识和公开相机metadata可以共享，额外RAW内容不行。

主矩阵尽量共享合规初始化规则和搜索机会；域专用预训练差异明列并计预算。原预训练RGB塔的Default/RawVLA系统继续作为实用强参照，不与受控矩阵混作同初始化单变量实验；RAW不必凭从头少量训练击败大规模RGB预训练后才允许继续研究。

### 5.3 怎样解释差异

分别看同机制RAW与RGB差异、同输入N01与Base差异，以及两种差异在任务/成像条件下怎样变化。四格结果分别保留；若计算常规配对差异或交互量，只用于解释，不发明“RAW增益总分”或硬阈值。只赢`rgb_base`不能自动证明超过同机制`rgb_n01`；机制在RGB上也有效不否定其通用价值。

优先沿真实案例追踪暗部边缘、暗亮交界、微小目标和运动后恢复。固定ISP的信息限制、噪声条件去除、支持/读出干预形成有来源的消融；冻结干预与匹配重训分开解释，输入扰动导致分布外退化不独立证明RAW物理信息收益。

## 6 理论和创新证据

### 6.1 信息可用性的条件性基础

令RGB为同源RAW经固定映射`Y=T(X)`，任务上下文相同。任何只依赖Y的决策在信息层面可由拥有X的一方复合T来实现，因此在不限制函数类、最优求解等假设下，额外保留X不会降低最优可达决策能力；有损映射可能使不等关系严格。可逆且保留任务信息的ISP不保证出现RAW优势。

这是已有信息处理原理的应用，不是新定理，也不证明受限N01架构能实现该最优解。RAW主链禁止以RGB中间图供下游使用，不可据上述理论假定本架构包含所有RGB策略；实际表征、预训练、有限训练和算力会改变结果。

### 6.2 积累收益与陈旧偏差

固定设计与权重、`y_i=z_i+eps_i`、零均值噪声、权重和为1时，目标为当前`z_t`，有：

```text
MSE = (sum_i w_i (z_i - z_t))² + wᵀΣw
```

第一项含运动/结构/成像不相容带来的偏差，第二项含测量及插值相关噪声。它说明为什么更多历史并不总更好，也把RAW的可校准测量统计与支持选择连接起来。自适应权重存在数据选择偏差，不能靠`detach`消除；未知协方差只给proxy或有条件保守界。RGB臂不假装拥有同样物理模型。[R06、R34]

### 6.3 删除一致性和顺序证据的角色

既有P1的依赖/回放参考用于主链删除正确性；在完整神经算子上说明适用粒度和感受野。BOCPD后验、经验校准与time-uniform错误保证分开；只有满足可预测性/条件矩等假设才引用e-process或CS，不强制编译新统计库作为训练前置。[R16–R18]

主链理论文件拟为 `research/theory/n01_raw_evidence_contract_v1_2.md`。已有结果、标准原理、本候选未证命题与经验现象分开；研究假设未获支持可以交付负结果，不以夹具数量或自造“定理PASS”替代任务证据。

## 7 训练与有意义的验证

完整模型从第一轮正式主体训练即存在。真实RAW/RGB同源数据、完整32帧、原生π0.5 action flow matching共同训练；不得把原型20步/小裁剪或RawVLA训练ISP的2k步直接称替换视觉塔后的充分训练。

先短fit确认真实参数更新、视觉/历史确有作用和有效恢复，再按四臂同等机会冻结/开放后端范围。每平台/方法约30k—100k action updates仅作待完整profiling的初始规划，可从50k安排并按已登记预算扩展，不是必须达到的“收敛合格线”。明确搜索、预训练、失败/续训、验证和评测总成本。RoboTwin保持已审计`MEAN_STD`与14维顺序，LIBERO用其7维独立统计。

训练充分性用真实源轨迹/条件覆盖、有效优化更新、固定噪声/flow时间验证、普通随机验证与开发闭环曲线说明。若仍改善按合理预算延长，欠拟合先查对接/优化；不得设置连续八次平台、两次降LR、loss必须下降X%等门槛，也不自动无限续训。

当前单种子只说明固定训练seed下的结果。按任务/episode做配对统计和区间，不估计不存在的跨训练seed方差；rollout任务/初态seed仍按清单覆盖，不能只评一条轨迹。

每次普通修改只运行受影响的正确性、固定真实案例和必要开发验证。只有改变模型/训练配置或影响比较公平性时，才补受影响的训练臂；不默认重跑整个8-run矩阵。没有严格AST/逐tensor/每batch哈希准入，没有无来源综合分；保留防泄漏、因果、实际更新、恢复、来源删除和原评测分母这些实质检查。

## 8 同步可视化和控制台

每个固定案例展示：同源RAW固定预览与ISP-RGB、局部差异/裁剪/量化说明、真实融合热图、对应点与支持历史、来源/信息限制干预、动作及闭环结果。先回答“哪项信息不同、模型利用了什么、行动怎样改变”，再给实现日志。

RAW预览也经过显示映射，不能只凭预览看起来更亮就证明保留信息；图例与数值域明确，同帧同色标，不对每方法独立调亮。DBSR通道均值图标平均融合权重；测量可靠性、风险、干预影响分别画。无自然flow真值不画所谓flow error，无运动学标定不画假真值三维轨迹。案例包含分层随机、事件和失败，不只挑成功。

本机研究统筹台更新N01 v1.2路线、四臂矩阵、数据边界、P3当前入口及可复制完整包；只读显示计划/证据，不把文档更新写成训练进度。新v1.2 HTML与Markdown同源，提供示意架构和文档导航。

DeepSeek执行时调用前端专家做简单实时监控：原生单页、有限轮询、实际活动/step/rollout、方法ID与RAW/RGB、平台、单种子、数据来源层级、曲线、固定案例、日志/错误；编码未知总量不编造百分比。监控独立于训练，可断连恢复，无数据库/复杂登录/任意shell按钮。准备时远端8080由Jupyter占用，监控默认loopback18080经SSH转发；运行前重查。

## 9 阶段与交付

### 9.1 P3 完整模型与 RAW 信息归因

同一阶段完成同源RAW/RGB入口、完整N01及成熟Base、四臂真实单种子主体训练、离线/案例归因、按本轮总100次预算的开发闭环、理论说明和同步案例；开发闭环不要求全四臂或两平台都覆盖。内部按依赖调试，不重新拆成局部研究关卡。RoboTwin优先检查HDR到显示输入的信息作用，LIBERO继续完整研究其实际可用条件。

按用户最新补充，**开发期每轮默认总预算为100次闭环，优先当前模型；不是每平台100次，也不是四臂各100次。** 默认先评raw_n01；两平台均已就绪时可50＋50，只研究一个平台时可该平台100次，不为覆盖另一平台阻塞当前开发。若本轮主要比较两臂/四臂，可在这100次内分配同身份子集，逐组报告实际成功数/分母，不假称每组100次。

本轮预算按一次开发交付累计，不是每个checkpoint或每次验证点各100次。默认先用离线验证选当前checkpoint，再跑100次闭环；若在预算内测试多个checkpoint，按各自身份分别报成功数/分母，不能混成同一模型100次。

完成本轮100次即报告成功率、正常失败/超时、少量代表性案例及成本，无需开发期跑完所有任务、成像条件、四臂或原13,250清单；不因成功率低、区间宽、没有正收益而自动追加。可比较项优先共享初态子集，尚未闭环的臂标未运行，只作已实际完成的离线/案例结论，不把完整四臂闭环作为P3交付硬门槛。基础设施缺口单列，有限恢复仍有缺口则如实报告，不能无限重跑到100个成功。

四臂是研究比较规格与一次主体训练安排；100次开发闭环只支持当前覆盖和固定训练seed下的初步结果，不宣称完整任务覆盖或训练种子稳定。开发身份与P4正式身份分开，P4完整清单须后续单独安排，不由开发脚本自动扩张。当前P3尚未实施。

### 9.2 P4 冻结全量验证

冻结四臂每平台checkpoint、同源ISP、选模规则、历史/相机/后端/控制协议和正式清单。每方法完成LIBERO10000＋RoboTwin3250，四方法单种子核心为`4×13,250=53,000` rollouts。附加632原条件核查单列，不加入13,250主分母；补充B1/新ISP/额外种子若执行另记矩阵，不默认加入。

正常失败/策略超时按原规则计原分母，基础设施缺口按固定同身份恢复；不换seed挑最好尝试，不删除困难条件。正式测试不选checkpoint/阈值；测试后修改模型按新迭代披露。分平台/条件/任务结果、事件后恢复、完整时延与失败保留，单种子不外推训练稳定性。

### 9.3 P5 依据限制扩展

按P3/P4限制选择真实CFA/跨传感器会话、第二成熟骨干/读出、不同ISP或长记忆/成本优化之一；涉及输入/机制主张时保持对应RAW/RGB归因。真实RAW视频可验证输入/表示，不冒充VLA动作数据；实际机器人需要相应数据与执行授权。额外曝光/主动采集另计采集和控制机会。

不把P3必须回答的RAW/RGB信息归因延期到P5。P5也默认单种子，不强制所有扩展串行。32帧外状态或局部缓存改变删除范围时重述保证与实测成本，不假称压缩状态可精确恢复任意源。

## 10 远程运行、实现位置与迁移

旧服务器已停用，新服务器沿用户提供 `ssh -p 34302 root@154.59.156.21 -L 8080:localhost:8080` 连接。准备时8080是Jupyter，实时监控转发应使用 `ssh -N -p 34302 root@154.59.156.21 -L 8080:127.0.0.1:18080`；本机端口冲突可改本机18080。运行前完整读 `/etc/vast-agents-guide.md`，查supervisor/GPU/磁盘/端口，不停止其他服务。

数据根 `/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench/data`；encoder Python `/workspace/owac/rawvla/envs/encoder/bin/python`、policy Python `/workspace/owac/rawvla/envs/starvla/bin/python`。独立新run建议 `/workspace/owac/rawvla/research/n01-v1/runs/v1-2-<stage>-<run-id>/`，source/config/outputs/receipts分开，保留真实源码版本与diff，不建立逐文件哈希门槛。新服务器的环境与资产实际部署情况见[部署记录](N01_new_server_deployment_2026-10-15.md)及[新机快照](prompts/n01_v1_2/remote_snapshot.json)，运行前重查。源码快照在`/workspace/owac/rawvla/owac`，原小型基准归档另存`records/baseline-archive/`；旧P2正式运动checkpoint不在本机归档内，不虚称已恢复。

**新服务器有两张RTX PRO 6000 Blackwell Server Edition，每张约96GiB。** 默认GPU0运行RAW队列、GPU1运行RGB队列，各卡Base/N01依次执行，可并行训练，也可用独立策略端口并行验证；进程内设备号与物理GPU号分别记录，检查渲染设备实际归属。开发闭环先建每轮总100条清单，再按不重复episode身份分给两卡，不能变成每卡100条。并行不增加训练种子、主体run数或验证范围，不为此引入DDP/复杂调度。

拟新增训练/评估/服务入口采用`n01_v12`命名；这是未来实施位置，不是已运行CLI。现有v1.1的同名拟议文件未实现时不重复造两套；若实际已有实现，则复用并登记迁移。新配置放 `configs/encoders/n01_full/v1_2/`。具体文件职责和命令见完整提示词。

长任务用独立supervisor program，保存model/optimizer/scheduler/scaler/RNG/sampler以恢复；新GPU依赖不升级冻结基线环境。端到端训练不能经HTTP截断autograd，HTTP仅闭环推理。重要文件及时同步，`/workspace`不默认具备回收/销毁持久性。

v1.1及旧提示词保留并标历史；当前入口、研究卡、规则、状态与控制台统一指向v1.2。P0—P2运行收据、旧方案原文及原Default基准不因新版本改写。全部阶段完成后分别交接，不自动越阶段；本次没有新模型训练、远程监控部署或正式评测。

## 11 参考文献与作者实现

既有33项来源沿用v1.1审查记录，新增R34—R37补充RAW信息、成像处理和高层视觉边界。论文/代码是选型依据，未运行的新模块不称复现完成；固定版本用于定位作者实现，不建立hash准入工程。逐文件导航见[v1.2参考实现](prompts/n01_v1_2/01_REFERENCE_CODE.md)。

- **[R01]** Bhat et al. *Deep Burst Super-Resolution*. CVPR 2021。[论文](https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.html)；[固定作者代码](https://github.com/goutamgmb/deep-burst-sr/tree/16f158e0553b6702cc1693c43f733c2830973278)。现有内容与融合基线。
- **[R02]** Luo et al. *BSRT: Improving Burst Super-Resolution with Swin Transformer and Flow-Guided Deformable Alignment*. CVPRW 2022。[论文](https://arxiv.org/abs/2204.08332)；[作者代码](https://github.com/Algolzw/BSRT)。强骨干替换候选；完整 RAW 合规适配待做。
- **[R03]** Dudhane et al. *Burst Image Restoration and Enhancement*. CVPR 2022。[论文](https://arxiv.org/abs/2110.03680)；[BIPNet](https://github.com/akshaydudhane16/BIPNet)。隐式对齐与跨帧通信近邻。
- **[R04]** Dudhane et al. *Burstormer: Burst Image Restoration and Enhancement Transformer*. CVPR 2023。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.html)；[作者代码](https://github.com/akshaydudhane16/Burstormer)。已查部分特征/对齐/融合源码，未运行新模型。
- **[R05]** Mildenhall et al. *Burst Denoising with Kernel Prediction Networks*. CVPR 2018。[作者论文与代码页](https://bmild.github.io/kpn/)。来源利用和优化参考。
- **[R06]** Wronski et al. *Handheld Multi-Frame Super-Resolution*. ACM TOG 2019。[作者机构页](https://research.google/pubs/handheld-multi-frame-super-resolution/)。RAW 采样与鲁棒融合依据。
- **[R07]** Wang, Lipson and Deng. *SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow*. ECCV 2024。[论文](https://arxiv.org/html/2405.14793v1)；[作者代码](https://github.com/princeton-vl/SEA-RAFT)。已复用完整模块，P2 风险/泛化边界按验收保留。
- **[R08]** *WAFT: Warping-Alone Field Transforms for Optical Flow*. 2025 预印本。[论文](https://arxiv.org/abs/2506.21526)；[作者代码](https://github.com/princeton-vl/WAFT)。运动替换近邻，未做 RAW 运行验收。
- **[R09]** *RViDeformer: Efficient Raw Video Denoising Transformer with a Larger Benchmark Dataset*. TCSVT 2025（早期预印本 2023）。[论文](https://arxiv.org/abs/2305.00767)；[作者代码/数据](https://github.com/cao-cong/RViDeformer)。ReCRVD 显示屏重拍范围须披露。
- **[R10]** *AIM 2025 Low-light RAW Video Denoising Challenge: Dataset, Methods and Results*. 2025。[论文](https://arxiv.org/abs/2508.16830)。多传感器 RAW 数据与评测参考，未取得/运行新数据。
- **[R11]** Di et al. *QMambaBSR: Burst Image Super-Resolution with Query State Space Model*. CVPR 2025。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Di_QMambaBSR_Burst_Image_Super-Resolution_with_Query_State_Space_Model_CVPR_2025_paper.html)。查询状态空间融合近邻。
- **[R12]** Zhu et al. *Trajectory-aware Shifted State Space Models for Online Video Super-Resolution*. ICLR 2026。[论文](https://arxiv.org/html/2508.10453v2)；[TS-Mamba 作者代码](https://github.com/QZ1-boy/TS-Mamba)。已查方法与仓库，README 的训练配置路径访问失败，不能声称开箱可训练或 RAW 适配完成。
- **[R13]** Shi et al. *MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation*. ICLR 2026。[论文](https://arxiv.org/html/2508.19236v2)；[作者代码](https://github.com/shihao1895/MemoryVLA)。检索、融合与记忆合并近邻。
- **[R14]** Shi et al. *MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models*. 2026 预印本。[论文](https://arxiv.org/abs/2606.09827)。记忆/想象边界，不因相关而把世界模型加入 N01。
- **[R15]** Jiang et al. *RoboTTT: Context Scaling for Robot Policies*. 2026 预印本。[论文](https://arxiv.org/abs/2607.15275)；[作者机构页](https://research.nvidia.com/labs/gear/robottt/)。长上下文和序列训练参考，不声称 fast weights 可直接删源。
- **[R16]** Adams and MacKay. *Bayesian Online Changepoint Detection*. 2007。[论文](https://arxiv.org/abs/0710.3742)。标准变化检测对照。
- **[R17]** Altamirano, Briol and Knoblauch. *Robust and Scalable Bayesian Online Changepoint Detection*. ICML 2023。[论文](https://proceedings.mlr.press/v202/altamirano23a.html)；[DSM-bocd 作者代码](https://github.com/maltamiranomontero/DSM-bocd)。鲁棒参考，条件保证不混为频率学误报界。
- **[R18]** Howard et al. *Time-uniform, Nonparametric, Nonasymptotic Confidence Sequences*. Annals of Statistics 2021。[论文](https://arxiv.org/abs/1810.08240)；[confseq](https://github.com/gostevehoward/confseq)。条件适用的顺序推断。
- **[R19]** Shin, Ramdas and Rinaldo. *E-detectors: A Nonparametric Framework for Sequential Change Detection*。[论文](https://arxiv.org/abs/2203.03532)；[期刊原文](https://nejsds.nestat.org/journal/NEJSDS/article/59/text)。平均误报间隔与检测延迟参考，不能换成全程误报概率结论。
- **[R20]** Sünderhauf and Protzel. *Switchable Constraints for Robust Pose Graph SLAM*. IROS 2012。[作者论文](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf)。错误约束退出的已有机制。
- **[R21]** Kaess et al. *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. IJRR 2012。[作者页](https://people.csail.mit.edu/kaess/pub/Kaess12ijrr.html)；[GTSAM](https://github.com/borglab/gtsam)。依赖重求解参考，不强制搬成 GPU 主线。
- **[R22]** Cui and Harada. *RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images*. ECCV 2024。[论文](https://arxiv.org/abs/2408.14802)。适配近邻，ISP 路线不是本主链。
- **[R23]** Huang et al. *Dr. RAW: Towards General High-Level Vision from RAW with Efficient Task Conditioning*. NeurIPS 2025。[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20eadb6338a53114a6371f102c5ded95-Abstract-Conference.html)。传感器/任务适配近邻，预处理与重马赛克另审。
- **[R24]** Li et al. *Towards RAW Object Detection in Diverse Conditions*. CVPR 2025。[论文](https://arxiv.org/abs/2411.15678)；[AODRaw 作者代码/数据](https://github.com/lzyhha/AODRaw)。RAW 预训练与数据不足的依据；RGB 教师蒸馏不进入本主训练链。
- **[R25]** Liu et al. *RawVLA: Embodied Neural Image Signal Processor for Robotic Manipulation*. 2026 预印本。[原文及训练协议](https://arxiv.org/html/2609.37530v1)；[作者项目](https://shuhongll.github.io/rawvla/)。直接竞争边界，实际输入以固定数据审计为准。
- **[R26]** Physical Intelligence et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization*. 2025。[论文](https://arxiv.org/abs/2504.16054)；[openpi](https://github.com/Physical-Intelligence/openpi)。执行仍用冻结 StarVLA/RawVLA 接口与权重。
- **[R27]** Lehtinen et al. *Noise2Noise: Learning Image Restoration without Clean Data*. ICML 2018。[论文](https://proceedings.mlr.press/v80/lehtinen18a.html)。辅助监督的条件依据。
- **[R28]** Batson and Royer. *Noise2Self: Blind Denoising by Self-Supervision*. ICML 2019。[论文](https://proceedings.mlr.press/v97/batson19a.html)。噪声独立与 J-invariance 的适用边界。
- **[R29]** Jain and Wallace. *Attention is not Explanation*. NAACL 2019。[论文](https://aclanthology.org/N19-1357/)。原实验在 NLP，不直接当作视觉注意力无效定理。
- **[R30]** Wiegreffe and Pinter. *Attention is not not Explanation*. EMNLP 2019。[论文](https://aclanthology.org/D19-1002.pdf)。解释性需要具体干预/对照实验。
- **[R31]** Jaegle et al. *Perceiver IO: A General Architecture for Structured Inputs & Outputs*. ICLR 2022。[论文](https://arxiv.org/abs/2107.14795)；[官方实现](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)。按需读出候选。
- **[R32]** Gu and Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM 2024。[论文](https://arxiv.org/abs/2312.00752)；[官方实现](https://github.com/state-spaces/mamba)。状态候选，整体稳定/删除保证需另证。
- **[R33]** Revay, Wang and Manchester. *Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness*. IEEE TAC 2023。[论文](https://arxiv.org/abs/2104.05942)；[作者代码](https://github.com/imanchester/REN)。稳定结构参考，不将其保证转移给任意 Mamba 组合。

## 14. 版本记录与当前实施状态

| 版本 | 日期 | 路线与证据状态 |
| --- | --- | --- |
| v1.0 | 2026-10-14 | 来源级支持、T1—T6、P0—P8；原方案及全部实验保留 |
| P2 独立验收 | 2026-10-15 | 实现已修补，六组仅一组满足原平台条件，分布外运动/风险失效；未续训、未进入 P3 |
| **v1.1** | **2026-10-15** | **用户认可后正式采用：P3 完整模型＋真实数据训练＋开发闭环与可视化，P4 冻结全量验证，P5 迁移扩展；本次仅更新文档，P3 尚未实施** |

后续每轮研究交付必须回答：**完整编码器在真实任务上改变了什么，哪些旧观测导致了这个变化，机器人因此做得更好还是更差。**

- **[R34]** Brooks et al. *Unprocessing Images for Learned Raw Denoising*. CVPR2019。[论文](https://arxiv.org/abs/1811.11127)；[作者代码](https://github.com/timothybrooks/unprocessing)。成像与噪声域参考，逆处理不恢复源RGB已丢失信息。
- **[R35]** Chen et al. *Learning to See in the Dark*. CVPR2018。[论文](https://arxiv.org/abs/1805.01934)；[作者代码](https://github.com/cchen156/Learning-to-See-in-the-Dark)。短曝光RAW研究背景，RGB恢复目标不进入主链。
- **[R36]** Cui and Harada. *RAW-Adapter*. ECCV2024。[论文](https://arxiv.org/abs/2408.14802)；[作者代码](https://github.com/cuiziteng/ECCV_RAW_Adapter)。已有RAW高层视觉适配。
- **[R37]** Huang et al. *Dr. RAW*. NeurIPS2025。[论文](https://papers.neurips.cc/paper_files/paper/2025/hash/20eadb6338a53114a6371f102c5ded95-Abstract-Conference.html)。已有传感器条件与任务适配，不借其结果保证N01。
