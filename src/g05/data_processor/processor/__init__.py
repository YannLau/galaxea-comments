# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/processor/__init__.py — 数据处理器的“总装车间”包入口
# =============================================================================
#
# 【这个文件是什么】
#   它是 Python 包 ``g05.data_processor.processor`` 的 ``__init__.py``，
#   作用是把 ``processor/`` 这个目录标记成一个可被导入的包。
#
#   注意：这个文件里**故意一行代码都没有**（只有你正在读的这段注释和下面的中文
#   docstring）。它不 import 任何 Processor、不定义任何类、也没有 ``__all__``，
#   原因见下面【为什么这个 __init__ 是空的】。
#   真正的实现全部在同目录的 5 个模块里，并且都以“全路径”方式被导入使用。
#
# 【这个包负责什么（一句话）】
#   把数据集 ``__getitem__`` 取出来的**原始样本**（原始 action/state 张量、相机画面、
#   任务文本），加工成 DataLoader 可以直接拼批（collate）的**样本字典**：
#   图像 resize/张量化、动作/状态变换（相对动作、旋转表示……）、归一化、
#   按语义分组的多键合并、指令语言选择、CoT 模板槽位填充。
#   与它平级的 ``transforms/`` 包提供“被它调用的可逆算子”，本包负责“编排顺序”。
#
# 【它在整条数据链路中的位置】
#
#   configs/data/<task>.yaml（数据侧）
#   configs/model/g05.yaml 里的 processor: 段（模型侧，优先级更高）
#        │
#        │  Hydra 按 ``_target_`` 反射实例化，例如
#        │    _target_: g05.data_processor.processor.galaxea_cot_processor.GalaxeaCoTProcessor
#        │  实例化入口：g05/utils/data/processor_utils.py 的 build_processors()
#        │    · cfg.data.processors 不存在 → 直接 instantiate(cfg.model.processor)
#        │    · 存在（多 embodiment 混合）→ 逐 embodiment 合并配置后再包成 MixtureProcessor
#        ▼
#   g05.data_processor.processor           ← 本文件（包入口，无代码）
#   ├─ base_processor.py                   ← 核心：BaseProcessor / ActionProcessor / FullProcessor
#   ├─ galaxea_cot_processor.py            ← 配置默认使用的 GalaxeaCoTProcessor（带 CoT 样本模板）
#   ├─ galaxea_action_processor.py         ← 具身相关的动作过滤器（R1LiteJointActionFilter）
#   ├─ samples_builder.py                  ← RoboVQA 样本模板类族（由 _target_ 选择）
#   └─ mixture_processor.py                ← MixtureProcessor：按 embodiment_type 分发
#        │
#        ▼
#   数据集 __getitem__ 末尾调用 processor.preprocess(sample)
#        │
#        ▼
#   DataLoader 拼批 → 模型侧 g05.models.g05.io.input_preprocessor.InputPreprocessor
#        （token 化属于模型层，不在本包内完成）
#
# 【类继承关系一览（新手先看这张“家谱”）】
#
#   BaseProcessor(ABC)                        base_processor.py
#     │  · 最底层骨架：解析/校验 shape_meta、持有 normalizer、
#     │    管理 train()/eval() 状态、解析图像 shape
#     │  · 提供 action_state_transform()（先断言维度，再依次 forward 每个变换）
#     │  · 提供 augment_instruction()（指令采样 / 中英文选择）
#     │
#     ├── ActionProcessor(BaseProcessor)      base_processor.py
#     │     · 只做动作/状态，不碰图像与文本：适合动作 tokenizer 训练等场景
#     │     · preprocess() 只产出 gt_action / action / proprio 等数值字段
#     │
#     └── FullProcessor(ActionProcessor)      base_processor.py
#           · 在 ActionProcessor 之上增加：图像变换 + pixel_values 组装、
#             指令增强、tokenizer 元信息（pad_token_id / image_token_index /
#             max_text_tokens）、build_pixel_values()（多分辨率 / 补零 / 选取）
#           │
#           └── GalaxeaCoTProcessor(FullProcessor)   galaxea_cot_processor.py
#                 · 当前配置默认使用的处理器（见 configs/model/g05.yaml 的 processor: 段）
#                 · 两阶段 preprocess：
#                     阶段 1 _process_tensors()：张量加工，并留下临时字段
#                         _instructions（指令文本）、_vlm_action（给模板用的动作副本）
#                     阶段 2 samples_builder.build(data, sample)：填 CoT 模板槽位
#                 · 具体模板由配置 ``samples_builder._target_`` 决定，
#                   默认为 BaseSamplesBuilder（无 CoT）
#
#   MixtureProcessor(ABC)                     mixture_processor.py
#     · 不是上面那条继承链的子孙，而是一个“注册表 / 分发器”：
#       构造参数是 {embodiment_type: BaseProcessor 实例}
#     · train() / eval() / set_normalizer_from_stats() 会转发给每个子 processor
#     · 自身不实现 preprocess() / postprocess()（调用会抛 NotImplementedError），
#       因为“该用哪个 processor”要由样本所属的 embodiment 决定
#     · 支持 processor["r1pro"] 这种下标取用（__getitem__ 按 embodiment_type 查表）
#
#   R1LiteJointActionFilter(BaseActionFilter)  galaxea_action_processor.py
#     · 具身相关的“逐维动作过滤器”：按维判断该动作维是否有效 / 是否运动，
#       不属于 processor 继承链，而是被 processor 当作 action_filter 使用
#
# 【一个样本在 preprocess() 里走过的固定流水线】
#   （顺序很重要：改动会直接影响归一化统计量的含义）
#     action_filter.forward           过滤 / 屏蔽无效动作维
#       → action_state_transform     相对动作、旋转表示等（transforms/ 里的算子）
#       → normalizer.forward         用数据集统计量做归一化
#       → action_state_merger.forward 把多个 key 合并成一条固定维度向量
#   反向过程在 postprocess() 里按相反顺序执行（merger → normalizer → transforms 逆序），
#   供推理 / 评估把模型输出还原回物理量。
#
# 【为什么这个 __init__ 是空的】
#   和父包 ``src/g05/data_processor/__init__.py`` 同样的理由，而且更严格：
#   一旦在这里 import 具体 Processor，就会连锁触发
#   ``base_processor → g05.utils.data.normalizer → g05.data_processor`` 这类回路，
#   在 Python 里表现为“部分初始化的模块（partially initialized module）”报错；
#   同时也会让只想拿到包名的轻量脚本被迫加载 torch / transformers。
#   所以约定是：**本包不提供包级导出，所有使用者都写全路径导入**，例如
#     · ``from g05.data_processor.processor.base_processor import BaseProcessor``
#     · ``from g05.data_processor.processor.mixture_processor import MixtureProcessor``
#   （仓库里的 scripts/serve_policy.py、scripts/eval_open_loop.py、
#     src/g05/data/base_lerobot_dataset.py 都是这样写的。）
#
# 【新手最容易踩的坑】
#   1) 别在本文件加 import：会出现循环导入，或让这个包变得很“重”。
#      真要加导出，先确认依赖方向不会成环。
#   2) 别把“本包的 processor”和另外两个同名的东西搞混：
#        · 配置里的 ``processor:`` 段（configs/model/g05.yaml）——只是配置块的名字；
#        · 模型侧的 ``InputPreprocessor``（g05/models/g05/io/input_preprocessor.py）
#          ——负责 token 化，跑在模型里，不是数据处理器。
#   3) 别以为 ``MixtureProcessor`` 能直接处理样本：它只负责按 embodiment 分发，
#      单样本的 preprocess 必须落到某个具体 processor 上。
#   4) 新增 Processor 时不要在本文件“注册”：配置里的 ``_target_`` 写全路径即可，
#      Hydra 会自己反射导入。
#
# 【相关文件与文档】
#   src/g05/data_processor/transforms/               本包调用的可逆算子（forward/backward）
#   src/g05/utils/data/processor_utils.py            build_processors()：配置 → 实例
#   src/g05/utils/data/normalizer.py                 LinearNormalizer（归一化实现）
#   src/g05/data/base_lerobot_dataset.py             数据集侧调用 processor.preprocess 的位置
#   configs/model/g05.yaml                           processor: 段完整示例（含各参数含义）
#   configs/data/parts_meta/*.yaml                   action_state_merger 的分组定义
#   docs/data/samples_builders_zh.md                 samples_builder 类族总览
#   docs/architecture/g05_io_zh.md                   Dataset → Collate → 模型的张量形状
# =============================================================================

"""数据处理器：负责 shape_meta 解析、归一化与样本组装。

本模块（包入口）刻意不导入任何具体类，请直接从各自的子模块导入，例如：

- ``base_processor.py``           ``BaseProcessor`` / ``ActionProcessor`` / ``FullProcessor``
- ``galaxea_cot_processor.py``    ``GalaxeaCoTProcessor``（配置默认使用的处理器）
- ``galaxea_action_processor.py`` ``R1LiteJointActionFilter`` 等具身相关的动作过滤器
- ``samples_builder.py``          RoboVQA 样本模板类族（由配置 ``_target_`` 选择）
- ``mixture_processor.py``        ``MixtureProcessor``（按 embodiment_type 分发）

这样做可以避免与 utils / transforms 之间产生循环导入，并让包入口保持足够轻量。
"""
