# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""动作有效维过滤器的基础实现。

这里的“过滤”并不是删除动作张量中的列，而是为每个动作部件生成一份布尔掩码
``action_op_mask``，描述当前样本中哪些动作维度属于“有效操作维”。例如：

    batch["action"] = {"left_arm": Tensor[H, 6], "left_gripper": Tensor[H, 1]}
    batch["action_op_mask"] = {
        "left_arm": BoolTensor[6], "left_gripper": BoolTensor[1]
    }

``H`` 是动作序列长度，而掩码没有时间维。基础实现把所有维度都标成 ``True``；真正根据
动作幅度和阈值判断的逻辑位于 ``processor/galaxea_action_processor.py`` 中的
``R1LiteJointActionFilter``。

它处于数据流水线的最前面：原始动作字典 → 过滤器 → action/state 变换 → 归一化 →
merger。merger 随后会把动作和掩码分别按相同顺序拼接、补齐，再交给模型和 tokenizer。
``action_op_mask`` 与 ``action_is_pad``、``action_dim_is_pad`` 不同：前者标记真实动作维中
哪些维度执行了操作；后两者标记时间轴和特征轴上为尺寸对齐而补出的无效位置。
"""

import torch

# 所有 action/state 变换都遵循这个接口。过滤器虽然也继承它，但一般无法仅根据动作值
# 逆推出先前生成的有效维掩码，因此基础过滤器按不可逆变换处理。
from g05.data_processor import BaseActionStateTransform


class BaseActionFilter(BaseActionStateTransform):
    """动作有效维过滤器的基类，同时也是“全部有效”的默认实现。

    ``forward`` 不改变 ``batch["action"]`` 的数值，只为 ``shape_meta`` 中登记的每个动作
    部件生成长度为 ``D`` 的全 ``True`` 掩码。子类可复用构造参数和元数据处理，并覆盖
    ``forward``，根据动作变化幅度生成真正的有效维掩码。

    一般的过滤器会从一段动作中计算并附加掩码，而原动作数值不足以恢复这份判定结果，
    所以这里声明为不可逆。基础类的 ``backward`` 只是原样返回输入；确实只生成全 True
    掩码的 :class:`DummyActionFilter` 会把 ``invertible`` 改回 ``True``。

    构造参数：
        joint_threshold: 普通关节动作的默认阈值。
        gripper_threshold: 夹爪动作的默认阈值。
        velocity_threshold: 速度类动作（例如底盘、躯干）的默认阈值。
        eef_threshold: 末端执行器位姿动作的默认阈值，默认 ``1e-3``。
        dim_thresholds: 按动作 key 配置的逐维阈值字典，通常优先于上述分类阈值。

    本基类只保存阈值，不消费它们；当前由 ``R1LiteJointActionFilter._resolve_threshold``
    解释这些配置。将参数放在基类中，可以让不同过滤器共享一致的 Hydra 配置接口。
    """

    # 这是过滤器这一类算子的通用契约，并不表示下面的全 True 实现修改了动作数值。
    invertible = False

    def __init__(
        self,
        joint_threshold: float | None = None,
        gripper_threshold: float | None = None,
        velocity_threshold: float | None = None,
        eef_threshold: float | None = 1e-3,
        dim_thresholds: dict | None = None,
    ):
        # None 表示该分类没有显式配置阈值，具体子类可自行决定兜底值。
        self.joint_threshold = joint_threshold
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold
        self.eef_threshold = eef_threshold

        # 确保子类始终能安全执行 ``key in self.dim_thresholds``。传入非空字典时不作复制。
        self.dim_thresholds = dim_thresholds or {}

    def set_shape_meta(self, shape_meta):
        """记录当前 embodiment 的动作/状态部件元数据。

        ``shape_meta`` 由 processor 根据 parts 配置构造，结构类似：

            {"action": [{"key": "left_arm", "raw_shape": 6, "shape": 6}, ...],
             "state":  [{"key": "left_arm", "raw_shape": 6, "shape": 6}, ...]}

        ``key is None`` 的条目是统一布局中的空槽位，不对应 batch 中的真实张量，必须排除。
        列表顺序会保留，后续 merger 正是按这套顺序拼接各部件。

        此方法通常由 ``BaseProcessor.__init__`` 调用；必须先调用它再执行 ``forward``，
        否则实例上还没有 ``action_meta`` 和 ``state_meta``。
        """
        processed_action_meta, processed_state_meta = [], []

        # 只保留能在 batch["action"] 中按 key 找到的真实动作部件。
        for meta in shape_meta["action"]:
            if meta["key"] is not None:
                processed_action_meta.append(meta)

        # 基类目前不使用状态元数据，但统一保存它，方便子类扩展，并与其他变换接口一致。
        for meta in shape_meta["state"]:
            if meta["key"] is not None:
                processed_state_meta.append(meta)
        self.action_meta = processed_action_meta
        self.state_meta = processed_state_meta

    def forward(self, batch):
        """为每个真实动作部件生成全 ``True`` 的逐维掩码。

        输入动作结构是 ``batch["action"][key] = Tensor[..., D]``。本方法只读取最后一维
        ``D``，因此前面是时间维还是 batch + 时间维都不影响结果。它会原地增加或覆盖
        ``batch["action_op_mask"] = {key: BoolTensor[D]}``，并返回同一个字典对象。

        若输入没有 ``"action"``（例如只处理观测的推理路径），则完全不修改输入。
        本实现按动作张量的实际宽度创建掩码；具体的 R1Lite 子类还会校验它是否等于元数据
        中的 ``raw_shape``。
        """
        # 纯观测样本没有动作，也就没有需要标记的动作维；不要凭空创建空掩码字段。
        if "action" not in batch:
            return batch

        action_op_mask = {}
        for meta in self.action_meta:
            # raw_shape 在本实现中未参与计算；与 key 一并取出是为了保留元数据语义，具体子类
            # 通常会用它检查输入宽度（R1LiteJointActionFilter 正是如此）。
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["action"][k].shape[-1]

            # 每个布尔值对应动作最后一维的一列。基类不做静止检测，所以所有真实维均有效。
            # 沿用原行为在 CPU 创建；数据加载器之后会把整理好的 batch 统一搬到训练设备。
            flag = torch.ones(actual_shape, dtype=torch.bool)
            action_op_mask[k] = flag

        # 整体赋值：重复调用会重建并覆盖旧掩码，不会混入上次调用遗留的 key。
        batch["action_op_mask"] = action_op_mask
        return batch

    def backward(self, batch):
        """不修改数据，直接返回输入。

        通用过滤判定不能从动作值中逆推，所以基类既不重建/删除掩码，也不修改动作。需要在
        反向阶段按掩码清零无效维的实现，应像 ``R1LiteJointActionFilter`` 一样覆盖本方法。
        """
        return batch


class DummyActionFilter(BaseActionFilter):
    """把所有动作维标为有效的空操作过滤器。

    它复用基类的 ``forward``，生成全 ``True`` 的 ``action_op_mask``，即不按动作幅度屏蔽
    任何真实维。从结果上看，这等价于所有阈值均设为 0 的
    ``R1LiteJointActionFilter``（该实现使用 ``>=`` 与阈值比较）。

    前向和反向都不改变动作数值，掩码也不依赖输入内容，所以可声明为可逆。反向仍保留
    ``action_op_mask``，供流水线后续组件继续使用。
    """

    invertible = True

    def forward(self, batch):
        """调用基类实现，为全部动作部件生成全 ``True`` 掩码。"""
        return super().forward(batch)

    def backward(self, batch):
        """空操作反向变换：不改动作与掩码，原样返回同一个 batch。"""
        return batch
