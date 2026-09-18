# 版权所有 (c) Meta Platforms, Inc. 及其关联公司。
# 保留所有权利。
#
# 本文件中的旋转转换代码改编自 PyTorch3D 的旋转工具。
# 上游项目所采用的 BSD 风格许可证声明请参阅 THIRD_PARTY_NOTICES.md。

from typing import Literal, List
import torch
import torch.nn.functional as F

from g05.utils.data.rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_rotation_9d,
    rotation_9d_to_matrix,
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    quaternion_to_rotation_6d_jit,
    quaternion_to_rotation_9d_jit,
    rotation_6d_to_quaternion_jit,
)
from g05.data_processor import BaseActionStateTransform
from copy import deepcopy


class PoseRotationTransform(BaseActionStateTransform):
    """
    在四元数/欧拉角与适合神经网络学习的 6D/9D 旋转表示之间转换位姿。

    一个位姿由“位置 + 姿态”组成。本类始终保留前三个位置分量，只转换后面的
    姿态分量。前向变换通常用于模型输入或训练标签的预处理，反向变换则把模型空间
    中的结果还原成数据集原本使用的姿态格式：

    - 前向：``input_rotation_type`` → ``rotation_6d``/``rotation_9d``；
    - 反向：``rotation_6d``/``rotation_9d`` → ``input_rotation_type``。

    输入位姿最后一维的布局：

    - 7 维四元数位姿：``[x, y, z, qx, qy, qz, qw]``；
    - 6 维欧拉角位姿：``[x, y, z, ex, ey, ez]``。

    输出位姿最后一维的布局：

    - 9 维 6D 位姿：``[x, y, z, rotation_6d(6个数)]``；
    - 12 维 9D 位姿：``[x, y, z, rotation_9d(9个数)]``。

    这里的“6D/9D”仅指旋转部分的维数，不包含位置的 3 维。6D 表示取旋转矩阵
    的前两行，恢复时用 Gram-Schmidt 正交化补出第三行；9D 表示保存完整旋转矩阵
    的 9 个元素，恢复时用 SVD 投影到合法旋转矩阵。它们避免了四元数正负二义性
    和欧拉角跳变，因而更适合作为神经网络连续回归的目标。

    注意：夹爪开合量不属于空间位姿，应保存在独立字段（例如 ``left_gripper``），
    不要拼进 ``ee_pose``。本变换声明为可逆；但欧拉角本身存在周期性，同一姿态在
    数值上不一定恢复为完全相同的一组角度。
    """

    invertible = True

    def __init__(
        self,
        rotation_type: Literal["quaternion", "rotation_6d", "rotation_9d"],
        category_keys: List[str],
        input_rotation_type: Literal["quaternion", "euler"] = "quaternion",
        euler_convention: str = "XYZ",
        fast_forward: bool = True,
    ):
        # 目标旋转格式，也是 forward 输出、backward 输入所采用的格式。
        self.rotation_type = rotation_type
        # 形如 {"state": ["left_ee_pose"], "action": [...]}，决定要处理哪些字段。
        self.category_keys = category_keys
        # 原始数据使用的旋转格式，也是 forward 输入、backward 输出的格式。
        self.input_rotation_type = input_rotation_type
        # 欧拉角的轴旋转顺序；仅在输入/输出格式为 euler 时生效。
        self.euler_convention = euler_convention
        # 四元数路径可调用融合后的 TorchScript 函数，减少中间张量和 Python 开销。
        self.fast_forward = fast_forward

    def forward(self, batch):
        """复制批数据，并把配置字段从原始旋转格式转换成模型使用的格式。"""
        # 变换不应原地污染数据集返回的 batch；后续变换可以安全复用原始数据。
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            # 闭环评测时 batch 可能只有观测状态、没有待训练的 action。
            if cat == "action" and "action" not in out_batch:
                continue

            for k in ks:
                if self.fast_forward:
                    out_batch[cat][k] = self._forward_fast(out_batch[cat][k])
                else:
                    out_batch[cat][k] = self._forward(out_batch[cat][k])

                if (
                    cat == "action"
                    and "action_op_mask" in out_batch
                    and k in out_batch["action_op_mask"]
                ):
                    # 位姿维数变化后，动作有效位掩码也必须具有对应的维数。
                    out_batch["action_op_mask"][k] = self._forward_mask(
                        out_batch["action_op_mask"][k]
                    )

        return out_batch

    def backward(self, batch):
        """复制批数据，并把模型格式的位姿还原成数据集原始格式。"""
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            for k in ks:
                # 快速反向函数主要服务单条/单序列推理。高维批数据沿用通用矩阵路径，
                # 既与现有流水线保持兼容，也避免 JIT 实现对复杂批维的额外约束。
                if self.fast_forward and out_batch[cat][k].ndim <= 2:
                    out_batch[cat][k] = self._backward_fast(out_batch[cat][k])
                else:
                    out_batch[cat][k] = self._backward(out_batch[cat][k])

                if (
                    cat == "action"
                    and "action_op_mask" in out_batch
                    and k in out_batch["action_op_mask"]
                ):
                    out_batch["action_op_mask"][k] = self._backward_mask(
                        out_batch["action_op_mask"][k]
                    )

        return out_batch

    # 每种表示对应 (完整位姿维数, 旋转部分维数)；完整位姿还包含前三维位置。
    _ROT_DIMS = {
        "quaternion": (7, 4),
        "euler": (6, 3),
        "rotation_6d": (9, 6),
        "rotation_9d": (12, 9),
    }

    def _remap_mask(self, mask: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        """
        在两种旋转表示之间重映射动作有效位掩码 ``op_mask``。

        位置的三个掩码逐项保留。旋转表示之间没有可靠的一一维度对应关系，例如一个
        四元数分量变化会影响旋转矩阵中的多个元素，因此先用 ``any`` 把源旋转掩码
        合并成“该旋转整体是否有效”，再把这个布尔值扩展到目标旋转的全部维度。

        某些数据源的掩码长度可能不规范：过短时在末尾补 ``False``，过长时截断，
        从而先对齐源格式的标准位姿长度。
        """
        if src_type == dst_type:
            return mask

        in_dim, _ = self._ROT_DIMS[src_type]
        out_dim, out_rot_dim = self._ROT_DIMS[dst_type]

        if mask.shape[-1] < in_dim:
            mask = F.pad(mask, (0, in_dim - mask.shape[-1]), value=False)
        elif mask.shape[-1] > in_dim:
            mask = mask[..., :in_dim]

        pos_mask = mask[..., :3]  # x/y/z 的有效性可以直接逐项继承。
        rot_active = mask[..., 3:].any(dim=-1, keepdim=True)
        rot_out = rot_active.expand(*rot_active.shape[:-1], out_rot_dim)
        out = torch.cat([pos_mask, rot_out], dim=-1)
        assert out.shape[-1] == out_dim
        return out

    def _forward_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """令动作掩码的维数跟随前向位姿转换。"""
        return self._remap_mask(mask, self.input_rotation_type, self.rotation_type)

    def _backward_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """令动作掩码的维数跟随反向位姿转换。"""
        return self._remap_mask(mask, self.rotation_type, self.input_rotation_type)

    def _forward(self, pose):
        """
        用通用的“原表示 → 旋转矩阵 → 目标表示”路径转换位姿。

        参数:
            pose: 形状为 ``(..., 6)`` 或 ``(..., 7)`` 的张量。前导的 ``...``
                可以是任意批次维或时间维，函数只解释最后一维：

                - 欧拉角：``[x, y, z, ex, ey, ez]``；
                - 四元数：``[x, y, z, qx, qy, qz, qw]``。

        返回:
            转换后的位姿。旋转为 6D 时形状为 ``(..., 9)``，为 9D 时形状为
            ``(..., 12)``；前三维位置保持不变。
        """
        # 尽早检查最后一维，可在数据配置错误时给出比矩阵运算更直观的提示。
        if self.input_rotation_type == "quaternion":
            assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        elif self.input_rotation_type == "euler":
            assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"

        # 输入与目标都是四元数时无需转换，且不会改变张量值或形状。
        if self.rotation_type == "quaternion":
            return pose

        # 姿态转换与平移无关，先保存位置，最后再拼接回来。
        position = pose[..., :3]

        if self.input_rotation_type == "quaternion":
            # 数据集位姿使用 (qx,qy,qz,qw)，底层旋转工具使用 (qw,qx,qy,qz)。
            quaternion = pose[..., [6, 3, 4, 5]]  # (x,y,z,w) → (w,x,y,z)
            matrix = quaternion_to_matrix(quaternion)
        elif self.input_rotation_type == "euler":
            # 角度单位由底层接口约定为弧度，旋转顺序由 euler_convention 指定。
            euler_angles = pose[..., 3:6]
            matrix = euler_angles_to_matrix(euler_angles, self.euler_convention)
        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

        if self.rotation_type == "rotation_6d":
            rotation = matrix_to_rotation_6d(matrix)
        elif self.rotation_type == "rotation_9d":
            rotation = matrix_to_rotation_9d(matrix)
        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")

        return torch.cat([position, rotation], dim=-1)

    def _backward(self, pose: torch.Tensor):
        """
        前向变换的逆过程：经旋转矩阵把模型格式还原为数据集格式。

        参数:
            pose: 6D 旋转位姿的形状为 ``(..., 9)``，9D 旋转位姿的形状为
                ``(..., 12)``。

        返回:
            原始格式的位姿：欧拉角为 ``(..., 6)``，四元数为 ``(..., 7)``。
        """
        if self.rotation_type == "quaternion":
            return pose

        if self.rotation_type == "rotation_6d":
            rot_dim = 6
            expected_dim = 9  # 3 维位置 + 6 维旋转。
        elif self.rotation_type == "rotation_9d":
            rot_dim = 9
            expected_dim = 12  # 3 维位置 + 9 维旋转。
        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")

        assert pose.shape[-1] == expected_dim, f"Expected {expected_dim}D pose, got {pose.shape}"

        position = pose[..., :3]
        rotation = pose[..., 3 : 3 + rot_dim]

        if self.rotation_type == "rotation_6d":
            matrix = rotation_6d_to_matrix(rotation)
        elif self.rotation_type == "rotation_9d":
            matrix = rotation_9d_to_matrix(rotation)

        if self.input_rotation_type == "quaternion":
            quaternion = matrix_to_quaternion(matrix)
            # 底层工具返回 wxyz；写回数据集前恢复为位姿字段约定的 xyzw。
            quaternion = quaternion[..., [1, 2, 3, 0]]  # (w,x,y,z) → (x,y,z,w)
            return torch.cat([position, quaternion], dim=-1)
        elif self.input_rotation_type == "euler":
            euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
            return torch.cat([position, euler_angles], dim=-1)
        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

    def add_noise(self, pose: torch.Tensor, std_position=0.05, std_angle=0.05):
        """
        给 7 维四元数位姿添加高斯噪声，用于数据增强。

        位置直接叠加标准差为 ``std_position`` 的逐轴噪声。旋转先由四元数转换为
        轴角向量，再叠加标准差为 ``std_angle`` 的噪声，最后转回单位四元数；这样
        比直接扰动四元数四个分量更符合旋转几何，也能保证输出仍表示合法旋转。
        ``std_angle`` 的单位是弧度。
        """
        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        position = pose[..., 0:3]
        quaternion = pose[..., [6, 3, 4, 5]]
        axis_angles = quaternion_to_axis_angle(quaternion)
        position = position + std_position * torch.randn_like(position)
        axis_angles = axis_angles + std_angle * torch.randn_like(axis_angles)
        quaternion = axis_angle_to_quaternion(axis_angles)
        quaternion = quaternion[..., [1, 2, 3, 0]]
        return torch.cat([position, quaternion], dim=-1)

    def _forward_fast(self, pose: torch.Tensor):
        """
        使用 TorchScript 融合旋转转换的优化前向路径。

        四元数输入可直接生成 6D/9D 表示，省去显式构造中间旋转矩阵。欧拉角输入
        没有对应的融合函数，因此仍走“欧拉角 → 矩阵 → 6D/9D”的通用路径。
        """
        if self.input_rotation_type == "quaternion":
            assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        elif self.input_rotation_type == "euler":
            assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"

        if self.rotation_type == "quaternion":
            return pose

        position = pose[..., :3]

        if self.input_rotation_type == "quaternion":
            quaternion = pose[..., [6, 3, 4, 5]]  # 数据集 xyzw → 底层工具 wxyz。

            if self.rotation_type == "rotation_6d":
                rotation = quaternion_to_rotation_6d_jit(quaternion)
            elif self.rotation_type == "rotation_9d":
                rotation = quaternion_to_rotation_9d_jit(quaternion)
            else:
                raise NotImplementedError

            return torch.cat([position, rotation], dim=-1)

        elif self.input_rotation_type == "euler":
            euler_angles = pose[..., 3:6]
            matrix = euler_angles_to_matrix(euler_angles, self.euler_convention)

            if self.rotation_type == "rotation_6d":
                rotation = matrix_to_rotation_6d(matrix)
            elif self.rotation_type == "rotation_9d":
                rotation = matrix_to_rotation_9d(matrix)
            else:
                raise NotImplementedError

            return torch.cat([position, rotation], dim=-1)

        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

    def _backward_fast(self, pose: torch.Tensor):
        """
        使用 TorchScript 融合转换的优化反向路径。

        6D 旋转可由融合函数直接恢复为四元数；若最终需要欧拉角，再经旋转矩阵
        转换。9D 路径仍需先用 SVD 投影得到合法旋转矩阵。
        """
        if self.rotation_type == "quaternion":
            return pose

        if self.rotation_type == "rotation_6d":
            rot_dim = 6
            expected_dim = 9
        elif self.rotation_type == "rotation_9d":
            rot_dim = 9
            expected_dim = 12
        else:
            raise NotImplementedError

        assert pose.shape[-1] == expected_dim, f"Expected {expected_dim}D pose, got {pose.shape}"

        position = pose[..., :3]
        rotation = pose[..., 3 : 3 + rot_dim]

        if self.rotation_type == "rotation_6d":
            quaternion = rotation_6d_to_quaternion_jit(rotation)

            if self.input_rotation_type == "quaternion":
                quaternion = quaternion[..., [1, 2, 3, 0]]  # 底层工具 wxyz → 数据集 xyzw。
                return torch.cat([position, quaternion], dim=-1)
            elif self.input_rotation_type == "euler":
                matrix = quaternion_to_matrix(quaternion)
                euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
                return torch.cat([position, euler_angles], dim=-1)

        elif self.rotation_type == "rotation_9d":
            matrix = rotation_9d_to_matrix(rotation)

            if self.input_rotation_type == "quaternion":
                quaternion = matrix_to_quaternion(matrix)
                quaternion = quaternion[..., [1, 2, 3, 0]]  # 底层工具 wxyz → 数据集 xyzw。
                return torch.cat([position, quaternion], dim=-1)
            elif self.input_rotation_type == "euler":
                euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
                return torch.cat([position, euler_angles], dim=-1)

        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")


class EulerPoseToQuaternionTransform(BaseActionStateTransform):
    """
    在 ``RelativePoseTransform`` 之前，把欧拉角位姿转换为四元数位姿。

    前向转换：
        ``[..., 6] [x, y, z, ex, ey, ez]``
        → ``[..., 7] [x, y, z, qx, qy, qz, qw]``。

    反向转换：
        ``[..., 7]`` → ``[..., 6]``。

    典型用途是处理使用欧拉角的 OXE 数据集。相对位姿并不是两个欧拉角向量的
    逐元素之差；正确做法是先转换到四元数/旋转矩阵空间，再进行刚体变换的复合与
    求逆。因此该变换应放在 ``RelativePoseTransform`` 之前，保证相对姿态计算
    符合三维旋转几何。

    本变换可逆，但由于欧拉角的周期性和奇异点，反向得到的角度数值可能与输入不同，
    它们所表示的空间姿态仍然等价。
    """

    invertible = True

    def __init__(self, category_keys: dict, euler_convention: str = "XYZ"):
        # category_keys 指定 batch 中需要转换的 state/action 子字段。
        self.category_keys = category_keys
        # 例如 "XYZ" 表示按照 X、Y、Z 轴的约定解释三个欧拉角。
        self.euler_convention = euler_convention

    def forward(self, batch: dict) -> dict:
        """复制 batch，并将存在的目标字段由欧拉角位姿转换为四元数位姿。"""
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            if cat == "action" and "action" not in out_batch:
                continue
            for k in ks:
                if k in out_batch.get(cat, {}):
                    out_batch[cat][k] = self._euler_to_quat(out_batch[cat][k])
        return out_batch

    def backward(self, batch: dict) -> dict:
        """复制 batch，并将存在的目标字段由四元数位姿还原为欧拉角位姿。"""
        out_batch = deepcopy(batch)
        for cat, ks in self.category_keys.items():
            for k in ks:
                if k in out_batch.get(cat, {}):
                    out_batch[cat][k] = self._quat_to_euler(out_batch[cat][k])
        return out_batch

    def _euler_to_quat(self, pose: torch.Tensor) -> torch.Tensor:
        """将 ``[x,y,z,ex,ey,ez]`` 转为 ``[x,y,z,qx,qy,qz,qw]``。"""
        assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"
        position = pose[..., :3]
        euler = pose[..., 3:6]
        matrix = euler_angles_to_matrix(euler, self.euler_convention)
        # PyTorch3D 风格的底层函数返回 wxyz，而 RelativePoseTransform 接收 xyzw。
        quat_wxyz = matrix_to_quaternion(matrix)  # 底层格式：(w,x,y,z)。
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]  # 相对位姿格式：(x,y,z,w)。
        return torch.cat([position, quat_xyzw], dim=-1)

    def _quat_to_euler(self, pose: torch.Tensor) -> torch.Tensor:
        """将 ``[x,y,z,qx,qy,qz,qw]`` 转为 ``[x,y,z,ex,ey,ez]``。"""
        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        position = pose[..., :3]
        quat_xyzw = pose[..., 3:7]
        # 调用底层矩阵转换前，把数据集采用的 xyzw 重排为工具函数采用的 wxyz。
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]  # (x,y,z,w) → (w,x,y,z)。
        matrix = quaternion_to_matrix(quat_wxyz)
        euler = matrix_to_euler_angles(matrix, self.euler_convention)
        return torch.cat([position, euler], dim=-1)
