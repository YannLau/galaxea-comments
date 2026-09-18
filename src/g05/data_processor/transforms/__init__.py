# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/transforms/__init__.py — 数据变换子包的轻量入口
# =============================================================================
#
# 【这个文件是什么】
#   本文件是 ``g05.data_processor.transforms`` 子包的入口。Python 第一次执行
#   ``import g05.data_processor.transforms``，或导入它下面的任意模块（例如
#   ``g05.data_processor.transforms.relative_action``）时，都会先执行这里的代码。
#
#   它有意只做两件很小的事：
#     1) 从 ``base.py`` 导入数据变换的公共基类 ``BaseActionStateTransform``；
#     2) 通过 ``__all__`` 声明该子包入口正式导出的名字。
#
#   这里不实现任何变换算法，也不把所有具体变换集中导入。相对动作、旋转转换、
#   图像增强、动作过滤和张量合并等实现，都应从各自的子模块直接使用。
#
# 【它在数据处理流程中的位置】
#
#   Dataset 读出一条原始样本
#        │
#        ▼
#   BaseProcessor.preprocess()
#        │  按 YAML 配置顺序调用各变换的 forward()
#        │
#        ├─ action_filter.py        标记可操作的动作维度
#        ├─ relative_action.py      绝对动作 ↔ 相对/增量动作
#        ├─ rotation.py             Euler / 四元数 / 旋转矩阵等表示转换
#        ├─ misc.py                 拼接、角度环绕、二值化、数值压缩等小变换
#        ├─ action_state_merger.py  按部件拼接并补齐 action/state
#        └─ image.py                图像张量化、裁剪、缩放和颜色增强
#        │
#        ▼
#   模型使用整理后的 batch
#
#   推理得到动作后，``BaseProcessor.postprocess()`` 会对动作/状态级变换按相反顺序
#   调用 ``backward()``，把模型空间中的结果还原成机器人可执行的物理量。是否支持
#   反向恢复由每个 ``BaseActionStateTransform`` 子类的 ``invertible`` 属性决定。
#
# 【两类“Transform”不要混淆】
#   1) 动作/状态级变换
#      ``relative_action.py``、``rotation.py``、``misc.py``、``action_filter.py`` 和
#      ``action_state_merger.py`` 中的主要类继承 ``BaseActionStateTransform``。它们接收
#      包含 ``action``、``state`` 等字段的 batch 字典，并遵守 forward/backward 契约。
#
#   2) 图像级变换
#      ``image.py`` 中的类继承 ``torch.nn.Module``，接收的是单个图像张量，通常形如
#      ``[T, C, H, W]``。它们不继承本文件导出的基类，也没有统一的 backward 契约。
#
# 【配置驱动是怎样工作的】
#   仓库使用 Hydra 的 ``_target_`` 写完整类路径，例如：
#
#       action_state_transforms:
#         - _target_: g05.data_processor.transforms.relative_action.RelativeJointTransform
#           keys: [left_arm, right_arm]
#
#       transforms:
#         - _target_: g05.data_processor.transforms.image.ToTensor
#
#   Hydra 会按字符串中的模块路径导入具体子模块，再实例化目标类。因此，无需为了让
#   配置“找得到类”而把具体类写进本文件；完整的 ``_target_`` 路径本身就是入口。
#
# 【为什么不在这里导出所有具体类】
#   看似方便的集中导入，例如：
#
#       from .relative_action import RelativeJointTransform
#       from .image import ToTensor
#
#   会让任何一次 ``import g05.data_processor.transforms`` 都立即加载 torch、torchvision、
#   旋转工具和项目内其它依赖，不仅增加启动成本，还可能触发循环导入。仓库中多个具体
#   变换通过 ``from g05.data_processor import BaseActionStateTransform`` 获取基类，而
#   ``g05.data_processor.__init__`` 又会进入本子包；若这里继续导入这些具体模块，Python
#   可能在模块尚未初始化完成时再次访问它，最终得到“partially initialized module”错误。
#
#   所以本入口必须保持轻量：只导入依赖很少的 ``base.py``。需要具体类时应写：
#
#       from g05.data_processor.transforms.relative_action import RelativeJointTransform
#
#   而不要假设它能从 ``g05.data_processor.transforms`` 直接取得。
#
# 【下面两行代码分别做什么】
#   · ``from .base import BaseActionStateTransform`` 是“再导出”：类仍定义在 base.py，
#     这里只给它增加一条更短的公共导入路径。
#   · ``__all__`` 主要影响 ``from g05.data_processor.transforms import *``，并向文档工具、
#     IDE 和读者表明推荐的公共接口。它不是访问控制机制，也不会自动导入其它模块。
#
# 【新手常见误区】
#   1) 在 ``__all__`` 中添加类名并不会让类自动可用；必须先真实地导入那个名字。
#   2) 不要为缩短 YAML 路径而在这里批量导入具体类，完整子模块路径是仓库的既定写法。
#   3) ``BaseActionStateTransform`` 只适用于 batch 字典变换；写图像增强时应继承
#      ``torch.nn.Module``，并参考 ``image.py`` 的张量形状约定。
#   4) 继承基类后必须实现 ``forward()``；若保持 ``invertible=True``，还必须实现
#      ``backward()``。不可逆变换应显式声明 ``invertible = False``。
#
# 【相关文件】
#   src/g05/data_processor/__init__.py                   上一级包入口与循环导入说明
#   src/g05/data_processor/transforms/base.py            基类契约及往返测试工具
#   src/g05/data_processor/processor/base_processor.py   变换流水线的实际调用方
#   configs/data/_transforms.yaml                        图像变换配置示例
#   configs/data/r1pro.yaml                              动作/状态变换配置示例
# =============================================================================

"""配置驱动的数据变换子包。

本入口只再导出轻量的 :class:`BaseActionStateTransform` 基类。具体变换类请直接从
``relative_action``、``rotation``、``image``、``misc``、``action_filter`` 或
``action_state_merger`` 等子模块导入，以避免不必要的依赖加载与循环导入。
"""

# 将定义在 base.py 中的基类提升为子包级公共接口，使调用方可以写：
# ``from g05.data_processor.transforms import BaseActionStateTransform``。
# 这里只导入基类；不要顺手加入具体变换类，原因见文件头的循环导入说明。
from .base import BaseActionStateTransform

# 明确 ``import *`` 时只导出公共基类；具体变换必须从其所属子模块显式导入。
__all__ = ["BaseActionStateTransform"]
