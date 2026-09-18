# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from typing import Dict, List, Optional

import torch
from torch.nn.functional import pad

from g05.data_processor import BaseActionStateTransform
from g05.tokenizer.utils.parts_meta_utils import compute_grouped_layout, compute_grouped_dims


class ConcatLeftAlign(BaseActionStateTransform):
    """
    将按部件保存的动作/状态字典拼接为一个张量，并在最右侧补零到目标宽度。

    数据进入模型前通常长这样：``{"left_arm": [T, 6], "gripper": [T, 1]}``。
    模型更适合接收定宽张量，因此 ``forward`` 会严格按照 ``shape_meta`` 中的顺序
    拼接各部件，再补成 ``[T, target_dim]``。真实数据始终位于左侧，所以称为“左对齐”。

    ``backward`` 执行相反过程：先去掉右侧补位，再依据每个部件的原始维度切回字典。
    因而只要中间结果未被破坏，这个变换是可逆的。

    约定：
        - ``T``：时间步（action horizon）；``D``：特征维度。
        - ``action_dim_is_pad`` / ``proprio_dim_is_pad`` 中 ``True`` 表示补位维度。
        - ``action_op_mask`` 中 ``True`` 通常表示该动作维度可参与训练/执行；它会与
          action 使用完全相同的拼接顺序，但补位掩码本身不会被返回。
    """

    invertible = True

    def __init__(
        self,
        action_target_dim: int | None = None,
        state_target_dim: int | None = None,
        **kwargs,
    ):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim

    def set_shape_meta(self, shape_meta):
        """保存当前机器人形态的元数据；列表顺序就是之后的拼接/拆分顺序。"""
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        """将 batch 内的动作/状态字典拼接并补零为定宽张量。"""
        if "action" in batch:
            # 例如 [T, 6] + [T, 1] -> [T, 7]；meta 决定部件先后顺序。
            batch["action"] = self._concat(batch["action"], self.action_meta)
            if "action_op_mask" in batch:
                batch["action_op_mask"] = self._concat(batch["action_op_mask"], self.action_meta)
                batch["action_op_mask"], _ = self._pad(
                    batch["action_op_mask"], self.action_target_dim
                )

            # 右侧补到配置要求的宽度，并生成一维的“哪些维度是补位”掩码。
            batch["action"], batch["action_dim_is_pad"] = self._pad(
                batch["action"], self.action_target_dim
            )

        # state（本体状态/proprioception）与 action 使用同一套规则。
        if "state" in batch:
            batch["state"] = self._concat(batch["state"], self.state_meta)
            batch["state"], batch["proprio_dim_is_pad"] = self._pad(
                batch["state"], self.state_target_dim
            )

        return batch

    def backward(self, batch):
        """去除补位并按当前形态的元数据把张量还原为字典。"""
        # 若配置了定宽，先检查模型输出宽度，避免静默地按错误布局拆分。
        if self.state_target_dim is not None:
            assert batch["state"].shape[-1] == self.state_target_dim
        batch["state"] = self._crop(batch["state"], self.state_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)

        # action 的处理与 state 相同。
        if self.action_target_dim is not None:
            assert batch["action"].shape[-1] == self.action_target_dim
        batch["action"] = self._crop(batch["action"], self.action_meta)
        batch["action"] = self._split(batch["action"], self.action_meta)

        # 操作掩码必须与 action 同步裁剪、拆分，才能重新对应到各部件。
        if "action_op_mask" in batch:
            batch["action_op_mask"] = self._crop(batch["action_op_mask"], self.action_meta)
            batch["action_op_mask"] = self._split(batch["action_op_mask"], self.action_meta)

        return batch

    @staticmethod
    def _pad(x: torch.Tensor, dim: int):
        """在最后一维右侧补零，并返回补位后的张量和一维补位掩码。

        支持 action/state 的二维 ``[T, D]``，也支持 action_op_mask 的一维 ``[D]``。
        若 ``dim`` 为 ``None``，目标宽度就是当前宽度，相当于不补位。
        """
        if dim is None:
            dim = x.shape[-1]

        assert x.ndim in (1, 2) and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim))
        if x.ndim == 2:
            mask = torch.zeros_like(x[0]).bool()
            mask = pad(mask, (0, pad_dim), value=True)
        else:
            mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
            mask = pad(mask, (0, pad_dim), value=True)
        return x_padded, mask

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        """按照 meta 中各部件维度之和，裁掉张量最后一维右侧的补位。

        反向阶段的数据已经成批，因此支持 action 的 ``[B, T, D]`` 和
        action_op_mask 的 ``[B, D]``。这里的 ``meta`` 实际是元数据列表。
        """
        assert x.ndim in (2, 3)
        dim = sum([m["shape"] for m in meta])
        if x.ndim == 3:
            x = x[:, :, :dim]
        else:
            x = x[:, :dim]
        return x

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        """按 meta 的列表顺序取出各 key，并沿最后一维拼接。

        支持二维 action/state ``[T, D]`` 与一维 action_op_mask ``[D]``。
        """
        x = torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim in (1, 2)
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        """按 meta 中记录的 key 和 shape，从左至右把张量切回字典。

        支持三维 action/state ``[B, T, D]`` 与二维 action_op_mask ``[B, D]``。
        """
        assert x.ndim in (2, 3)
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            if x.ndim == 3:
                y[key] = x[:, :, idx : idx + dim]
            else:
                y[key] = x[:, idx : idx + dim]
            idx += dim

        return y


class DummyActionStateMerger(BaseActionStateTransform):
    """
    占位用合并器：不修改任何数据。

    它让不需要对齐/拼接的配置仍能使用统一的 processor 接口。``forward`` 和
    ``backward`` 都原样返回传入的 batch；恒等变换天然可逆。下方几个静态方法仅为
    保持与其他合并器接口一致而保留，正常流程不会调用。
    """

    invertible = True

    def __init__(self, action_target_dim: int | None = None, state_target_dim: int | None = None):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim

    def set_shape_meta(self, shape_meta):
        """接收并保存元数据以保持接口一致；恒等变换本身不会使用它。"""
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        """恒等操作：直接返回原 batch。"""
        return batch

    def backward(self, batch):
        """恒等操作：直接返回原 batch。"""
        return batch

    # 以下方法只为接口兼容而保留，DummyActionStateMerger 自身不会调用。
    @staticmethod
    def _pad(x: torch.Tensor, dim: int):
        """（未使用）把最后一维补到指定宽度。"""
        if dim is None:
            dim = x.shape[-1]

        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim))
        mask = torch.zeros_like(x[0]).bool()
        mask = pad(mask, (0, pad_dim), value=True)
        return x_padded, mask

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        """（未使用）裁掉右侧补位。"""
        assert x.ndim == 3
        dim = sum([m["shape"] for m in meta])
        x = x[:, :, :dim]
        return x

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        """（未使用）拼接字典中的张量。"""
        x = torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        """（未使用）把张量切回字典。"""
        assert x.ndim == 3
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            y[key] = x[:, :, idx : idx + dim]
            idx += dim

        return y


class PaddingActionMerger(BaseActionStateTransform):
    """
    将不同机器人形态（embodiment）的 action/state 字典对齐，便于混合成一个 batch。

    不同机器人可能缺少某些部件，或同名部件自由度不同。例如一台机器人有 6 维
    ``left_arm``，另一台有 7 维。``max_*_shape_meta`` 定义整个训练集合的统一键集合
    和每个键的最大宽度；前向时会补零、截断或创建虚拟部件。若 ``merge=True``，还会
    按该字典的插入顺序拼成一个扁平张量。反向时则依据 ``set_shape_meta`` 收到的当前
    机器人元数据，丢弃虚拟部件并裁回原始维度。

    注意：当原始维度大于配置的最大维度时会发生截断，这部分信息无法恢复。因此，
    “可逆”成立的前提是 ``max_*_shape_meta`` 至少覆盖所有真实部件的维度。
    """

    invertible = True

    def __init__(
        self,
        max_action_shape_meta: Dict[str, int] | None = None,
        max_state_shape_meta: Dict[str, int] | None = None,
        merge: bool = False,
        **kwargs,
    ):
        # 格式示例：{"left_arm": 6, "right_arm": 6, "gripper": 1, ...}。
        # Python 字典保持插入顺序；merge=True 时该顺序也就是扁平张量的字段布局。
        self.max_action_shape_meta = max_action_shape_meta
        self.max_state_shape_meta = max_state_shape_meta
        self.merge = merge

    def set_shape_meta(self, shape_meta):
        """保存当前机器人形态的原始 key/维度，供 backward 精确还原。"""
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        """对齐到全局最大布局；按 ``merge`` 配置决定是否继续拼成张量。"""
        if self.max_action_shape_meta is not None:
            if "action" in batch:
                # action 与 action_op_mask 必须使用相同的 key 布局。
                has_op_mask = "action_op_mask" in batch
                batch["action"], aligned_op_mask, action_padding_info = self._align_dict(
                    batch["action"], batch.get("action_op_mask", {}), self.max_action_shape_meta
                )
                if has_op_mask:
                    batch["action_op_mask"] = aligned_op_mask

                # merge=True 时把已对齐的字典变成模型可直接消费的单个张量。
                if self.merge:
                    batch["action"], batch["action_dim_is_pad"] = self._concat_aligned_dict(
                        batch["action"], action_padding_info, self.max_action_shape_meta
                    )
                    # 操作掩码也按完全相同的 key 顺序拼接。
                    if has_op_mask:
                        batch["action_op_mask"], _ = self._concat_aligned_dict(
                            batch["action_op_mask"], {}, self.max_action_shape_meta
                        )

            if "gt_action" in batch:
                # gt_action 是训练监督目标，也必须采用和 action 相同的布局。
                batch["gt_action"], _, gt_action_padding_info = self._align_dict(
                    batch["gt_action"], {}, self.max_action_shape_meta
                )

                # 与输入动作一致，可选地进一步拼成扁平张量。
                if self.merge:
                    batch["gt_action"], _ = self._concat_aligned_dict(
                        batch["gt_action"], gt_action_padding_info, self.max_action_shape_meta
                    )

        if self.max_state_shape_meta is not None and "state" in batch:
            # 对本体状态做同样的跨形态对齐。
            batch["state"], _, state_padding_info = self._align_dict(
                batch["state"], {}, self.max_state_shape_meta
            )

            # proprio_dim_is_pad 告诉模型哪些状态维度只是对齐用的补位。
            if self.merge:
                batch["state"], batch["proprio_dim_is_pad"] = self._concat_aligned_dict(
                    batch["state"], state_padding_info, self.max_state_shape_meta
                )

        return batch

    def backward(self, batch):
        """将统一布局的数据恢复为当前机器人原有的 key 与维度。"""
        if self.max_action_shape_meta is not None:
            if "action" in batch:
                # 若前向曾拼接，需先按全局布局切回“已对齐字典”。
                if self.merge:
                    batch["action"] = self._split_aligned_dict(
                        batch["action"], self.max_action_shape_meta
                    )

                # 再删除虚拟 key，并把每个真实 key 裁回当前形态的原始宽度。
                batch["action"] = self._restore_dict(batch["action"], self.action_meta)

            if "action_op_mask" in batch:
                # action_op_mask 使用同一套拆分与还原步骤。
                if self.merge:
                    batch["action_op_mask"] = self._split_aligned_dict(
                        batch["action_op_mask"], self.max_action_shape_meta
                    )

                # 最终掩码 key/维度须与还原后的 action 一一对应。
                batch["action_op_mask"] = self._restore_dict(
                    batch["action_op_mask"], self.action_meta
                )

            if "gt_action" in batch:
                if self.merge:
                    batch["gt_action"] = self._split_aligned_dict(
                        batch["gt_action"], self.max_action_shape_meta
                    )
                batch["gt_action"] = self._restore_dict(batch["gt_action"], self.action_meta)

        if self.max_state_shape_meta is not None and "state" in batch:
            if self.merge:
                batch["state"] = self._split_aligned_dict(batch["state"], self.max_state_shape_meta)
            batch["state"] = self._restore_dict(batch["state"], self.state_meta)

        return batch

    def _align_dict(
        self,
        data_dict: Dict[str, torch.Tensor],
        mask_dict: Dict[str, torch.Tensor],
        max_shape_meta: Dict[str, int],
    ):
        """
        将一个部件字典的 key 集合及各 key 宽度对齐到 ``max_shape_meta``。

        处理规则：
            1. key 存在但维度不足：在右侧补零到 ``target_dim``；
            2. key 存在但维度过大：从右侧截断到 ``target_dim``；
            3. key 不存在：创建全零的虚拟数据和全 False 的操作掩码。

        返回：
            aligned_data: 对齐后的数据字典，各值形状为 ``[T, target_dim]``。
            aligned_mask: 对齐后的 ``action_op_mask`` 字典，各值形状为 ``[target_dim]``。
            padding_info: 每个 key 的补位标记；``True`` 表示该维没有真实数据。

        ``aligned_mask`` 与 ``padding_info`` 容易混淆：前者表示动作维是否可操作，后者
        表示维度是否由跨形态对齐产生。真实但被禁用的动作维可能同时满足
        ``aligned_mask=False``、``padding_info=False``。
        """
        if not data_dict:
            return data_dict, mask_dict, {}

        # 以第一个真实部件为模板，获得时间长度、设备和数据类型。
        h = next(iter(data_dict.values())).shape[0]
        device = next(iter(data_dict.values())).device
        dtype = next(iter(data_dict.values())).dtype

        aligned_data = {}
        aligned_mask = {}
        padding_info = {}

        for key, target_dim in max_shape_meta.items():
            if key in data_dict:
                # key 存在：根据当前宽度与全局目标宽度的关系进行补齐/截断。
                current_data = data_dict[key]  # (h, current_dim)
                current_dim = current_data.shape[-1]

                if current_dim < target_dim:
                    # 当前形态自由度较少，在右侧补零。
                    pad_size = target_dim - current_dim
                    aligned_data[key] = torch.nn.functional.pad(current_data, (0, pad_size))

                    # 原始维为 False，新增补位为 True。
                    padding_info[key] = torch.cat(
                        [
                            torch.zeros(current_dim, dtype=torch.bool, device=device),
                            torch.ones(pad_size, dtype=torch.bool, device=device),
                        ]
                    )

                    # 新增维对当前机器人并不存在，因此操作掩码必须补 False。
                    if key in mask_dict:
                        current_mask = mask_dict[key]  # (current_dim,)
                        aligned_mask[key] = torch.nn.functional.pad(
                            current_mask, (0, pad_size), value=False
                        )
                    else:
                        # 未提供操作掩码时，默认所有真实维均可用。
                        aligned_mask[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

                elif current_dim > target_dim:
                    # 当前宽度超过统一布局；数据和掩码都从右侧截断。
                    aligned_data[key] = current_data[..., :target_dim]

                    # 这里没有新增维度，因此补位标记全部为 False。
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)

                    # 操作掩码必须与数据同步截断。
                    if key in mask_dict:
                        current_mask = mask_dict[key]  # (current_dim,)
                        aligned_mask[key] = current_mask[..., :target_dim]
                    else:
                        aligned_mask[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

                else:
                    # 宽度已匹配，无需复制或补零数据。
                    aligned_data[key] = current_data

                    # 所有维度均来自真实部件。
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)

                    # 数据宽度相同不代表外部传入的 mask 一定相同，故独立校正。
                    if key in mask_dict:
                        current_mask = mask_dict[key]  # (mask_dim,)
                        mask_dim = current_mask.shape[-1]

                        if mask_dim < target_dim:
                            # mask 较短时补 False，避免不存在的维度被误判为可操作。
                            aligned_mask[key] = torch.nn.functional.pad(
                                current_mask, (0, target_dim - mask_dim), value=False
                            )
                        elif mask_dim > target_dim:
                            # mask 较长时同步截断。
                            aligned_mask[key] = current_mask[..., :target_dim]
                        else:
                            # mask 宽度也匹配，可直接使用。
                            aligned_mask[key] = current_mask
                    else:
                        aligned_mask[key] = torch.ones(target_dim, dtype=torch.bool, device=device)
            else:
                # 当前机器人没有此部件：创建形状正确的全零“虚拟部件”。
                # 操作掩码全 False，保证下游不会把它当作可执行动作。
                aligned_data[key] = torch.zeros((h, target_dim), dtype=dtype, device=device)
                aligned_mask[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)

                # 虚拟部件的所有维度都属于补位。
                padding_info[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

        return aligned_data, aligned_mask, padding_info

    def _restore_dict(self, aligned_dict: Dict[str, torch.Tensor], meta: List[Dict]):
        """按当前形态 meta 保留真实 key、裁回原宽度，并丢弃补位和虚拟 key。"""
        restored = {}
        for m in meta:
            key = m["key"]
            original_dim = m["shape"]
            if key in aligned_dict:
                # 使用省略号兼容 [T, D]、[B, T, D] 和掩码 [B, D]。
                restored[key] = aligned_dict[key][..., :original_dim]
        return restored

    def _concat_aligned_dict(
        self,
        aligned_dict: Dict[str, torch.Tensor],
        padding_info: Dict[str, torch.Tensor],
        max_shape_meta: Dict[str, int],
    ):
        """
        按 ``max_shape_meta`` 的 key 顺序，把已对齐字典拼成一个张量。

        参数：
            aligned_dict: key 和宽度已经统一的数据字典。
            padding_info: 对齐过程产生的逐维补位标记。
            max_shape_meta: 同时定义拼接顺序和每个 key 的目标宽度。

        返回：
            concatenated_tensor: ``[T, sum(max_shape_meta.values())]`` 的单个张量。
            dim_is_pad: ``[sum(...)]`` 的布尔掩码；True 表示虚拟 key 或扩展维，
                False 表示来自当前机器人的真实维度。
        """
        if not aligned_dict:
            return aligned_dict, None

        # 遍历配置字典而不是 aligned_dict，确保不同样本的字段顺序完全一致。
        tensors = []
        padding_masks = []
        for key in max_shape_meta.keys():
            assert key in aligned_dict, f"Key '{key}' missing from aligned_dict"
            tensors.append(aligned_dict[key])

            # 普通 action/state 会携带 padding_info。
            if key in padding_info:
                padding_masks.append(padding_info[key])
            else:
                # action_op_mask 等调用可能不传 padding_info，此时默认无补位。
                dim = aligned_dict[key].shape[-1]
                device = aligned_dict[key].device
                padding_masks.append(torch.zeros(dim, dtype=torch.bool, device=device))

        # 始终沿特征维拼接，时间维保持不变。
        concatenated = torch.cat(tensors, dim=-1)  # [T, sum(dims)]

        # 补位掩码是一维布局描述，不随时间步重复。
        dim_is_pad = torch.cat(padding_masks, dim=-1)  # [sum(dims)]，True 表示补位

        return concatenated, dim_is_pad

    def _split_aligned_dict(
        self, concatenated_tensor: torch.Tensor, max_shape_meta: Dict[str, int]
    ):
        """
        按统一布局的顺序和宽度，将拼接张量切回“已对齐字典”。

        参数：
            concatenated_tensor: ``[B, T, sum(dims)]`` 或 ``[T, sum(dims)]``。
            max_shape_meta: 定义切分顺序和每段宽度。

        返回：
            aligned_dict: key 与 ``max_shape_meta`` 相同、但仍包含补位/虚拟 key 的字典。
        """
        # 兼容未组 batch 的二维数据和模型输出的三维数据。
        assert concatenated_tensor.ndim in (2, 3)

        aligned_dict = {}
        idx = 0
        for key, dim in max_shape_meta.items():
            aligned_dict[key] = concatenated_tensor[..., idx : idx + dim]
            idx += dim

        return aligned_dict


class GroupedPaddingMerger(BaseActionStateTransform):
    """
    使用 ``merge_spec`` 将互斥部件复用同一槽位，得到更紧凑的定宽张量。

    ``merge_spec`` 格式为 ``{槽位名: [候选原始 key, ...]}``。同一机器人形态通常只会
    拥有一组候选中的一个 key（例如关节控制与末端位姿控制二选一）。前向时，每个槽位
    按候选列表顺序选择第一个真实存在的 key，补到该组候选的最大宽度，再与其他槽位
    拼接。没有出现在任何组中的 key 称为“残余 key”，按 ``parts_meta`` 顺序追加在末尾。

    例如双臂布局可以是：
        - ``left_control``：``left_arm(8)`` 或 ``left_ee_pose(9)``，共用 9 维；
        - ``left_gripper``：``left_gripper``，占 1 维；
        - ``right_control``：``right_arm(8)`` 或 ``right_ee_pose(9)``，共用 9 维；
        - ``right_gripper``：``right_gripper``，占 1 维。
        最终只需 20 维，而不必为两套互斥控制表示同时预留空间。

    未配置 ``merge_spec`` 时不做互斥分组，所有 key 仅按 parts_meta 顺序拼接。

    前向：``Dict[raw_key, Tensor] -> Tensor [T, total_grouped_dim]``。
    反向：根据当前形态的 ``shape_meta`` 判断每个槽位属于哪个原始 key，并恢复维度。

    在每个分组对当前形态至多有一个真实候选、且最大维度配置正确时，该变换可逆。
    """

    invertible = True

    def __init__(
        self,
        max_action_shape_meta: Optional[Dict[str, int]] = None,
        max_state_shape_meta: Optional[Dict[str, int]] = None,
        merge_spec: Optional[Dict] = None,
        merge: bool = True,
        **kwargs,
    ):
        # 保留未分组的原始 parts_meta；反向恢复原 key 时仍然需要它。
        self._raw_max_action_shape_meta = max_action_shape_meta
        self._raw_max_state_shape_meta = max_state_shape_meta
        self.merge_spec = merge_spec
        self.merge = merge

        # layout 描述分组槽位；residual_keys 是未被任何分组消费的普通部件。
        self._action_layout, self._action_residual_keys = self._precompute(
            max_action_shape_meta, merge_spec
        )
        self._state_layout, self._state_residual_keys = self._precompute(
            max_state_shape_meta, merge_spec
        )

        if max_action_shape_meta is not None and merge_spec is not None:
            # 对外暴露的是分组后的紧凑维度，供模型/样本构建器计算输入输出宽度。
            self.max_action_shape_meta = compute_grouped_dims(max_action_shape_meta, merge_spec)
        else:
            self.max_action_shape_meta = max_action_shape_meta

        if max_state_shape_meta is not None and merge_spec is not None:
            self.max_state_shape_meta = compute_grouped_dims(max_state_shape_meta, merge_spec)
        else:
            self.max_state_shape_meta = max_state_shape_meta

    @staticmethod
    def _precompute(shape_meta, merge_spec):
        """把配置解析成稳定的分组布局，并找出未参与分组的残余 key。"""
        if shape_meta is None or merge_spec is None:
            return None, []
        layout = compute_grouped_layout(shape_meta, merge_spec)
        merged_keys = set()
        for group in layout:
            merged_keys.update(group.part_names)
        residual_keys = [k for k in shape_meta if k not in merged_keys]
        return layout, residual_keys

    def set_shape_meta(self, shape_meta):
        """记录当前机器人真实拥有的 action/state key 及其原始宽度。"""
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        """分别将 action、监督动作和 state 映射到各自的分组定宽布局。"""
        if self._raw_max_action_shape_meta is not None:
            if "action" in batch:
                has_op_mask = "action_op_mask" in batch
                batch["action"], batch["action_dim_is_pad"] = self._forward_one(
                    batch["action"],
                    self._raw_max_action_shape_meta,
                    self._action_layout,
                    self._action_residual_keys,
                )
                if has_op_mask:
                    batch["action_op_mask"], _ = self._forward_one(
                        batch["action_op_mask"],
                        self._raw_max_action_shape_meta,
                        self._action_layout,
                        self._action_residual_keys,
                    )
            if "gt_action" in batch:
                batch["gt_action"], _ = self._forward_one(
                    batch["gt_action"],
                    self._raw_max_action_shape_meta,
                    self._action_layout,
                    self._action_residual_keys,
                )

        if self._raw_max_state_shape_meta is not None and "state" in batch:
            batch["state"], batch["proprio_dim_is_pad"] = self._forward_one(
                batch["state"],
                self._raw_max_state_shape_meta,
                self._state_layout,
                self._state_residual_keys,
            )

        return batch

    def backward(self, batch):
        """根据当前形态元数据，把分组张量还原成原始部件字典。"""
        if self._raw_max_action_shape_meta is not None:
            if "action" in batch:
                batch["action"] = self._backward_one(
                    batch["action"],
                    self._action_layout,
                    self._action_residual_keys,
                    self.action_meta,
                    self._raw_max_action_shape_meta,
                )
            if "action_op_mask" in batch:
                batch["action_op_mask"] = self._backward_one(
                    batch["action_op_mask"],
                    self._action_layout,
                    self._action_residual_keys,
                    self.action_meta,
                    self._raw_max_action_shape_meta,
                )
            if "gt_action" in batch:
                batch["gt_action"] = self._backward_one(
                    batch["gt_action"],
                    self._action_layout,
                    self._action_residual_keys,
                    self.action_meta,
                    self._raw_max_action_shape_meta,
                )

        if self._raw_max_state_shape_meta is not None and "state" in batch:
            batch["state"] = self._backward_one(
                batch["state"],
                self._state_layout,
                self._state_residual_keys,
                self.state_meta,
                self._raw_max_state_shape_meta,
            )

        return batch

    def _align_per_raw_key(
        self,
        data_dict: Dict[str, torch.Tensor],
        max_shape_meta: Dict[str, int],
    ):
        """逐原始 key 对齐宽度：补零、截断，或为缺失 key 创建全零虚拟值。

        同时支持 action/state 的二维值 ``[T, D]`` 与 action_op_mask 的一维值 ``[D]``；
        虚拟 key 会沿用输入值的维数、dtype 与 device。返回的 ``padding_info`` 中
        ``True`` 表示该维度不是当前机器人的真实数据。
        """
        if not data_dict:
            return {}, {}

        sample_val = next(iter(data_dict.values()))
        device, dtype = sample_val.device, sample_val.dtype
        # action_op_mask 是一维；action/state 则以首维作为时间长度 T。
        is_1d = sample_val.ndim == 1
        if not is_1d:
            h = sample_val.shape[0]

        aligned, padding_info = {}, {}
        for key, target_dim in max_shape_meta.items():
            if key in data_dict:
                t = data_dict[key]
                d = t.shape[-1]
                if d < target_dim:
                    # 右补零，并仅把新增的尾部维度标为 padding。
                    pad_size = target_dim - d
                    aligned[key] = pad(t, (0, pad_size))
                    padding_info[key] = torch.cat(
                        [
                            torch.zeros(d, dtype=torch.bool, device=device),
                            torch.ones(pad_size, dtype=torch.bool, device=device),
                        ]
                    )
                elif d > target_dim:
                    # 超宽时保留左侧 target_dim 维；正确配置下通常不应触发。
                    aligned[key] = t[..., :target_dim]
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)
                else:
                    # 宽度已经匹配，保留原张量且所有维度均为真实数据。
                    aligned[key] = t
                    padding_info[key] = torch.zeros(target_dim, dtype=torch.bool, device=device)
            else:
                # 缺失部件占据统一布局中的槽位，但其值全零、所有维均标记为 padding。
                if is_1d:
                    aligned[key] = torch.zeros(target_dim, dtype=dtype, device=device)
                else:
                    aligned[key] = torch.zeros((h, target_dim), dtype=dtype, device=device)
                padding_info[key] = torch.ones(target_dim, dtype=torch.bool, device=device)

        return aligned, padding_info

    def _forward_one(self, data_dict, max_shape_meta, layout, residual_keys):
        """对齐原始 key，应用互斥分组，再沿最后一维拼成扁平张量。"""
        aligned, padding_info = self._align_per_raw_key(data_dict, max_shape_meta)
        device = next(iter(aligned.values())).device

        tensors, pad_masks = [], []

        if layout is None:
            # 无 merge_spec：退化为按 parts_meta 顺序直接拼接所有原始 key。
            for key in max_shape_meta:
                tensors.append(aligned[key])
                pad_masks.append(padding_info[key])
        else:
            for group in layout:
                # 选择首个非虚拟候选；候选在 merge_spec 中的顺序即优先级。
                # 若全都缺失，则使用第一个候选的全零虚拟值来占住该槽位。
                chosen = next(
                    (p for p in group.part_names if not padding_info[p].all()),
                    group.part_names[0],
                )
                t = aligned[chosen]
                is_pad = padding_info[chosen].clone()

                # 被选 key 的全局宽度仍可能小于组内最大宽度，需要再补齐槽位。
                chosen_max_dim = max_shape_meta[chosen]
                if chosen_max_dim < group.max_dim:
                    extra = group.max_dim - chosen_max_dim
                    t = pad(t, (0, extra))
                    is_pad = torch.cat(
                        [
                            is_pad,
                            torch.ones(extra, dtype=torch.bool, device=device),
                        ]
                    )

                tensors.append(t)
                pad_masks.append(is_pad)

        for key in residual_keys:
            # 未参与互斥分组的部件原样追加在所有分组槽位之后。
            tensors.append(aligned[key])
            pad_masks.append(padding_info[key])

        flat = torch.cat(tensors, dim=-1)
        dim_is_pad = torch.cat(pad_masks)
        return flat, dim_is_pad

    def _backward_one(self, flat_tensor, layout, residual_keys, embodiment_meta, max_shape_meta):
        """将分组扁平张量切片，并恢复为当前机器人形态的原始 key 字典。

        ``embodiment_meta`` 是关键：同一槽位可能代表多个候选 key，只有它能说明当前
        机器人实际使用哪一个。每段最终还会裁到该 key 的原始 ``shape``。
        """
        assert flat_tensor.ndim in (2, 3)
        result = {}
        idx = 0

        for group in layout:
            # 先按组的最大宽度取出整个共享槽位。
            slot_t = flat_tensor[..., idx : idx + group.max_dim]
            idx += group.max_dim
            for m in embodiment_meta:
                if m["key"] in group.part_names:
                    # 当前形态在该组中的真实 key 胜出，并裁掉组内补齐的尾部维度。
                    result[m["key"]] = slot_t[..., : m["shape"]]
                    break

        for key in residual_keys:
            # 残余 key 的槽位宽度直接来自原始全局 parts_meta。
            dim = max_shape_meta[key]
            t = flat_tensor[..., idx : idx + dim]
            idx += dim
            for m in embodiment_meta:
                if m["key"] == key:
                    # 当前形态不存在的残余 key 不会写入 result，等价于丢弃虚拟部件。
                    result[key] = t[..., : m["shape"]]
                    break

        return result
