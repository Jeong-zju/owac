# 最小补丁：让原视觉塔可跳过构造（OWAC π0.5 特征接入）

任务 D 需要从 projector **之后**的语言宽度接口进入 π0.5，主配置不保留原视觉塔执行路径
（见 `docs/prompts/DeepSeek_N01_implementation.md`「实际 π0.5 接入位置」）。冻结的
StarVLA `OpenPIPaliGemma` 无条件构造 SigLIP 视觉塔，因此这里保存一个**默认行为完全不变**
的最小补丁，供 OWAC 特征分支使用。冻结目录本身不改。

## 固定来源（本机已核对的参考副本）

| 项 | 值 |
| --- | --- |
| RawVLA 固定 commit | `ba9c56b04da2c4931b70d8844710f5928d43775b` |
| starVLA 子模块 commit | `57d508cb180d0510ac9060358009c0f6ffaac420` |
| 本机参考副本 | `repo/outputs/encoder-handoff-references/starvla/starVLA/`（只读参考，未安装） |
| 补丁基线文件 sha256 | `OpenPIPaliGemma.py` `b770f93c5c4297d3ec9eac951ca131dc96719784e5609795f1c38eea9f89ffe6`；`PI0.py` `286936ef8cf921e44d5718b1e617e9d91505326ec4be17ca88363523ebcb1617` |

## 文件

- `vision_tower_optional.patch`：2 个文件、6 处改动，全部由 `diff -ruN` 生成，路径前缀为
  `a/`、`b/`。

改动逐项：

1. `OpenPIPaliGemmaCore.__init__` 新增仅关键字参数 `build_vision_tower: bool = True`；为
   `False` 时 `self.vision_tower = None`。
2. `OpenPIPaliGemmaCore.get_image_features` 在视觉塔缺失时给出明确错误，而不是
   `NoneType is not callable`。
3. `OpenPIPaliGemmaContainer.__init__` 透传同名参数。
4. `OpenPIPaliGemma.__init__` 透传同名参数。
5. `PI0DefaultConfig` 新增字段 `build_vision_tower: bool = True`（`PI05DefaultConfig`
   继承，无需另改）。
6. `PI0._build_model_modules` 用 `self.config.framework.get("build_vision_tower", True)`
   读取该字段（与 `PI0.py:218` 已有的 `framework.get("raw_frontend", None)` 同一写法）。

## 应用

在 **starVLA 树根**（即包含 `starVLA/` 包的那一层，仓库根为 `<rawvla>/source/starVLA`）
执行其一：

```bash
git apply -p1 /path/to/vision_tower_optional.patch
patch -p1 -i /path/to/vision_tower_optional.patch
```

随后在加载配置里设置：

```yaml
framework:
  build_vision_tower: false
```

不设置时默认 `true`，与已发布基线逐字节一致；`owac.backends.rawvla.load_audited` 与
冻结评测路径不受影响。

## 理由

- 新读出层取代视觉投影，原 SigLIP 塔的权重在 `load_audited_with_replaced_vision` 里被
  明确列为“被替换”，加载时跳过；
- 若不补此丁，加载过程仍会临时构造视觉塔再释放，多一次大参数分配与释放（本机未实测具体
  耗时/显存，不写成收益数字）；
- 补丁保持默认路径不变，因此 C0/C2 基线回归不受影响。

## 验证状态（如实）

已在本机执行：

- 对冻结参考副本做 `git apply --check -p1` 与 `patch -p1 --dry-run`：均通过；
- 打补丁后 `OpenPIPaliGemma.py`、`PI0.py` 通过
  `ast.parse(source, feature_version=(3, 10))`（部署 runtime 为 Python 3.10）。

**未验证**：本机没有可导入的 starVLA 与 π0.5 权重，补丁未在真实 runtime 执行，因此
`framework.build_vision_tower: false` 经 `merge_framework_config` 传入的实际效果、
以及跳过视觉塔带来的参数量/显存/加载时间差异都没有实测。`merge_framework_config` 的
实现不在本机参考副本内。没有补丁时，OWAC 侧走
`AuditedVisualReplacementLoader` 的“构造后释放”回退路径（记录在返回的 audit 字典）。
