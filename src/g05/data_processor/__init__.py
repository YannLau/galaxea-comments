# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/__init__.py — 数据处理层（data_processor 包）的入口文件
# =============================================================================
#
# 【这个文件是什么】
#   它是 Python 包 ``g05.data_processor`` 的 ``__init__.py``，也就是“进这个包的第一道门”。
#   Python 里，只要一个目录下有 ``__init__.py``，它就是一个包；在
#   ``import g05.data_processor``（或任何 ``from g05.data_processor.xxx import ...``）
#   被执行时，Python 一定会先运行本文件。所以这里写什么，直接决定了
#   “别人一进这个包，能看到哪些东西、会额外触发多少导入”。
#
#   它本身只做两件事（正文只有 2 行有效代码）：
#     1) 从 ``transforms/base.py`` 把基类 ``BaseActionStateTransform`` 提上来，
#        实现一次“再导出（re-export）”；
#     2) 用 ``__all__`` 显式声明“本包对外只推荐用这一个名字”。
#   真正的功能实现都在两个子包里，不在本文件。
#
# 【它在整条数据链路中的位置】
#
#   configs/data/*.yaml + configs/model/g05.yaml 里的 processor: 段
#        │  （Hydra 按 ``_target_`` 字段反射实例化，串成流水线）
#        ▼
#   g05.data_processor                ← 本文件（包入口）
#   ├─ processor/                     数据 → 批量张量的“加工车间”
#   │   · base_processor.py           BaseProcessor / FullProcessor：
#   │                                 preprocess() 归一化、合并动作/状态、拼样本
#   │   · galaxea_cot_processor.py    GalaxeaCoTProcessor：带思维链样本模板的处理类
#   │   · galaxea_action_processor.py R1LiteJointActionFilter 等具身相关的动作过滤器
#   │   · samples_builder.py          把张量 + 指令组装成 VLM 训练样本模板
#   │   · mixture_processor.py        多 embodiment 混合时的“按具身分发”包装器
#   └─ transforms/                    ``_target_`` 驱动的“可逆数据变换”
#       · base.py                     BaseActionStateTransform（本文件导出的那一个）
#       · image.py / rotation.py / relative_action.py / action_filter.py /
#         action_state_merger.py / misc.py —— 全部继承上面这个基类
#        │
#        ▼
#   数据集 ``__getitem__`` 末尾调用 ``processor.preprocess(sample)``
#        │
#        ▼
#   DataLoader 组装成 batch
#        │
#        ▼
#   模型侧 g05.models.g05.io.input_preprocessor.InputPreprocessor
#        （把文本/图像/动作 token 化，这一步已经属于模型，不属于数据层）
#
# 【两个子包各自负责什么】
#   processor/
#     按 embodiment（具身/机器人型号）解析 shape_meta、做归一化、组装样本，
#     由 YAML 里 ``processor:`` 这一段配置实例化。
#     典型入口：``BaseProcessor.preprocess()``；开关在 ``configs/model/g05.yaml``。
#   transforms/
#     配置驱动（``_target_``）的“可逆数据变换”：相对动作、旋转表示、图像张量化、
#     动作/状态字典合并等；每个变换都必须实现 ``forward()``，可逆的还要实现
#     ``backward()``（详见 transforms/base.py）。
#
# 【为什么这个文件必须写得这么“空”】
#   关键在导入顺序和循环导入（circular import）：
#     · ``g05/utils/data/normalizer.py``（归一化器）需要
#       ``from g05.data_processor import BaseActionStateTransform``；
#     · 而 ``transforms/*.py``、``processor/base_processor.py`` 又都需要归一化器。
#   如果本文件顺手把子模块也 import 进来，就会形成
#       data_processor/__init__ → 子模块 → normalizer → data_processor（只加载了一半）
#   的回路，Python 拿到的是“半成品模块”，直接抛 ImportError 或拿到不完整的类。
#   所以这里的约定是：**本文件只导出不依赖任何重逻辑的基类，保持导入尽量轻**。
#
# 【新手最容易踩的 3 个坑】
#   1) 想“图方便”在本文件加 ``from .processor.base_processor import BaseProcessor``：
#      运行到一半就会报循环导入 / 部分初始化模块的错误。要用具体类，
#      请直接写全路径导入，例如
#      ``from g05.data_processor.processor.base_processor import BaseProcessor``。
#   2) 在 YAML 的 ``_target_`` 里写“包级”的类名（例如
#      ``g05.data_processor.ToTensor``）：本包级只承诺 ``BaseActionStateTransform``
#      这一个名字，其余类都藏在子模块里，必须写全路径，例如
#      ``_target_: g05.data_processor.transforms.image.ToTensor``。
#   3) 以为 ``__all__`` 是“对外可见名单”的强制开关：它实际只影响
#      ``from g05.data_processor import *`` 这种写法；显式写名字的导入
#      （``from g05.data_processor import 任意名字``）不受它限制，
#      所以真正避免循环导入的仍是“本文件不写重导入”这条纪律。
#
# 【和模型侧 input_preprocessor 的区别（别混用）】
#   · 本包（数据层）：Dataset → batch，产出的是**数值张量**
#     （归一化后的 action/state、uint8/float 图像等）。
#   · ``g05.models.g05.io.input_preprocessor``（模型层）：
#     batch → 模型输入，负责文本/图像/动作的 **token 化**，
#     在 collate 之后、前向之前执行。
#   两者职责不重叠，所以这里刻意不做任何跨层导入。
#
# 【相关文件与文档】
#   src/g05/data_processor/processor/base_processor.py    processor 主流程（preprocess/postprocess）
#   src/g05/data_processor/transforms/base.py             BaseActionStateTransform 定义处
#   src/g05/utils/data/normalizer.py                      LinearNormalizer，本基类的下游使用者
#   src/g05/data/base_lerobot_dataset.py                  数据集侧调用 processor 的位置
#   src/g05/models/g05/io/input_preprocessor.py           模型侧 token 化（与本包区分）
#   docs/data/schema_zh.md                                shape_meta 字段语义
#   docs/architecture/g05_io_zh.md                        Dataset → Collate → 模型的张量形状
#   configs/data/_transforms.yaml                         transforms 的配置写法示例
# =============================================================================

"""数据处理层：负责“数据集（dataset）→ 批量（batch）”之间的全部变换。

包内包含两个子包：
- ``processor/``：按具身（embodiment）解析 shape_meta、做归一化，并组装样本；
  这些类由配置里的 ``processor:`` 段落实例化。
- ``transforms/``：由配置 ``_target_`` 驱动的可逆数据变换，
  所有变换都继承自 ``BaseActionStateTransform``。

本模块刻意与 ``g05.models.g05.io.input_preprocessor`` 区分开：那个模块负责
模型侧的文本/图像/动作 token 化，不属于数据层。

本 ``__init__`` 只再导出轻量的基类。具体 processor / transform 类请直接从
各自的子模块导入，配置里的 ``_target_`` 也应指向子模块，
这样才能避免与 utils 之间产生循环导入。
"""

# 唯一的再导出：让下游可以写 ``from g05.data_processor import BaseActionStateTransform``。
# 只导入 transforms/base.py 这一个“零重依赖”的模块（它仅依赖 abc / torch），
# 从而保证本包入口足够轻，不会牵出 normalizer、processor 等重模块。
from .transforms.base import BaseActionStateTransform

# ``__all__`` 声明“from g05.data_processor import *”时对外暴露的名字，
# 同时也是给使用者和静态检查工具的提示：本包公开的只有这一个基类。
# 注意：具体的 Processor / Transform 类不在这里，请从子模块直接导入，
# 或者直接在 configs 的 ``_target_`` 中写子模块路径。
__all__ = ["BaseActionStateTransform"]
