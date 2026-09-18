# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from typing import List, Dict

import torch
from .rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)
from copy import deepcopy
from g05.data_processor import BaseActionStateTransform

# =============================================================================
# 本模块：把机器人动作改写成“相对于当前状态的变化量”
# =============================================================================
#
# 数据处理器传入的 batch 通常形如：
#
#   batch = {
#       "action": {key: Tensor[..., T_action, D]},  # 未来 T_action 步动作
#       "state":  {key: Tensor[..., T_state, D]},   # 历史 T_state 帧状态
#       ...                                          # 其它元数据原样透传
#   }
#
# 相对动作的共同基准是最新观测状态 ``state[..., -1:, :]``。保留长度为 1 的
# 时间维而不是取成 ``state[..., -1, :]``，是为了让 PyTorch 自动把它广播到全部
# T_action 个未来动作上。训练前调用 forward() 把绝对量变为更容易学习的相对量；
# 推理得到模型输出后，处理器按相反顺序调用 backward()，还原机器人真正要执行的
# 绝对动作。基类契约和流水线顺序详见 transforms/base.py。
#
# 本文件包含几类略有不同的变换：
#   · RelativePoseTransform：完整的刚体相对位姿，平移也会随基准坐标系旋转；
#   · DeltaPoseTransform：相邻时间步逐元素作差，不处理刚体坐标系；
#   · RelativeJointTransform：每个未来关节位置减去同一个最新关节状态；
#   · ReorderLowerBodyTransform / PartialRelativeTransform：下半身数据适配工具；
#   · BehaviorPerKeyTransform：把上述若干步骤组合起来适配 b1k behavior 数据。
#
# 术语及表示约定：
#   pose：位姿。四元数模式为 (x, y, z, i, j, k, r)，欧拉角模式为
#         (x, y, z, angle_1, angle_2, angle_3)。这里 r 是四元数实部，也常写作 w。
#   matrix：4×4 齐次变换矩阵，上左 3×3 是旋转，最右列前三项是平移。
#   position / pos：三维位置 (x, y, z)。
#   quaternion / quat：本文件对外采用 (i, j, k, r)，即常见的 (x, y, z, w)。
#   ``...``：任意数量的批次维；倒数第二维通常是时间维，最后一维是特征维。
# =============================================================================


# 这三个纯张量函数用 TorchScript 预编译，供快速路径复用，避免构造 4×4 矩阵。
@torch.jit.script
def quaternion_conjugate_jit(q: torch.Tensor) -> torch.Tensor:
    """求 ``(i, j, k, r)`` 顺序四元数的共轭；单位四元数的共轭即其逆。"""
    # 向量部取反、实部不变：(v, r)* = (-v, r)。
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


@torch.jit.script
def quaternion_multiply_jit(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """计算两个 ``(i, j, k, r)`` 四元数的 Hamilton 积 ``q1 * q2``。"""
    i1, j1, k1, r1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    i2, j2, k2, r2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Hamilton 积不满足交换律；乘法顺序对应旋转的复合顺序，不能互换 q1/q2。
    r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
    i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
    j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
    k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2

    return torch.stack([i, j, k, r], dim=-1)


@torch.jit.script
def quaternion_rotate_vector_jit(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """用 ``(i, j, k, r)`` 单位四元数 ``q`` 旋转三维向量 ``v``。"""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    # 展开式来自 v' = v + q.r * t + cross(q.xyz, t)，其中 t = 2*cross(q.xyz, v)。
    # 显式展开叉积可避免临时拼接纯四元数，也无需先生成旋转矩阵。
    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)

    # v' = v + q.r * t + cross(q.xyz, t)
    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)

    return torch.stack([rx, ry, rz], dim=-1)


class RelativePoseTransform(BaseActionStateTransform):
    """
    以最后一帧状态为基准，在绝对位姿与相对位姿之间转换。

    这里不是简单地逐元素相减。位姿属于刚体变换 SE(3)，必须同时考虑基准姿态
    对平移向量的影响。若 ``T_base`` 是最新状态位姿，``T_target`` 是动作目标：

        forward : T_relative = inverse(T_base) @ T_target
        backward: T_target   = T_base @ T_relative

    直观上，forward 回答“从机器人当前坐标系看，目标位于哪里”；backward 则把
    这个局部目标重新放回全局/父坐标系。两条路径支持：
      · quaternion：每个位姿 7 维 ``(x, y, z, i, j, k, r)``；
      · euler：每个位姿 6 维 ``(x, y, z, 三个欧拉角)``。

    参数：
        keys: 要转换的 action/state 子键，例如 ``["left_ee", "right_ee"]``。
            这些键必须同时存在，并具有可 stack 的相同形状。
        input_rotation_type: ``"euler"`` 时走欧拉角路径；其它值按四元数处理。
        euler_convention: 欧拉角轴顺序，直接传给 rotation.py 中的转换函数。
        fast_forward: True 时使用四元数直接公式；False 时经 4×4 齐次矩阵计算。
            两者数学含义相同，慢路径更直观，适合对照调试。

    该变换可逆；数值上默认假定输入四元数已经归一化。对于单位四元数，
    ``conjugate(q) == inverse(q)``，快速路径正是利用了这一点。
    """

    invertible = True

    def __init__(
        self,
        keys: List[str],
        input_rotation_type: str = "quaternion",
        euler_convention: str = "XYZ",
        fast_forward: bool = True,
    ):
        self.keys = keys
        self.input_rotation_type = input_rotation_type
        self.euler_convention = euler_convention
        self.fast_forward = fast_forward

    def forward(self, batch: Dict):
        """把指定键的绝对目标位姿转换为相对最新状态的位姿。"""
        # 闭环评估的某些阶段只预处理观测，还没有 action；此时无需转换。
        if "action" not in batch:
            return batch

        # 把 K 个身体部件沿新的第 0 维堆叠：每项 (..., T, D) → (K, ..., T, D)。
        # 后续函数把 K 也视作批次维，一次完成全部键的并行计算。
        action_stacked = torch.stack([batch["action"][k] for k in self.keys])
        state_stacked = torch.stack([batch["state"][k] for k in self.keys])

        # 只取状态历史的最后一帧，形状保留为 (K, ..., 1, D)，从而广播到动作
        # 的 T 个未来时间步，每一步都以同一个“当前时刻”作为参考系。
        if self.fast_forward:
            result = self._forward_fast(action_stacked, state_stacked[..., -1:, :])
        else:
            result = self._forward(action_stacked, state_stacked[..., -1:, :])

        # 位姿版本使用深拷贝，确保调用者传入的嵌套字典和原张量不被原地污染。
        out_batch = deepcopy(batch)
        # 拆掉最前面的 K 维，按原 keys 顺序写回 action；state 与元信息保持原样。
        for i, k in enumerate(self.keys):
            out_batch["action"][k] = result[i]

        return out_batch

    def backward(self, batch: Dict):
        """把模型输出的相对位姿还原成绝对目标位姿。"""
        action_stacked = torch.stack([batch["action"][k] for k in self.keys])
        state_stacked = torch.stack([batch["state"][k] for k in self.keys])

        if self.fast_forward:
            result = self._backward_fast(action_stacked, state_stacked[..., -1:, :])
        else:
            result = self._backward(action_stacked, state_stacked[..., -1:, :])

        out_batch = deepcopy(batch)
        for i, k in enumerate(self.keys):
            out_batch["action"][k] = result[i]

        return out_batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """可读性优先的正向实现：先转齐次矩阵，再做坐标系变换。"""
        if self.input_rotation_type == "euler":
            return self._forward_euler(pose, base_pose)

        # pose 与 base_pose 均为 (..., x, y, z, i, j, k, r)，最后一维必须为 7。
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )
        # T_base^{-1} @ T_target 同时完成“减去基准平移”和“旋回基准局部坐标系”。
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose

    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """可读性优先的反向实现：用基准齐次矩阵左乘相对位姿。"""
        if self.input_rotation_type == "euler":
            return self._backward_euler(pose, base_pose)

        # pose 与 base_pose 均为 (..., x, y, z, i, j, k, r)。
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose

    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        """把 ``(..., 7)`` 位姿组装为 ``(..., 4, 4)`` 齐次变换矩阵。"""
        position = pose[..., 0:3]
        # rotation.py 的工具采用实部在前的 (r, i, j, k)，这里先调整分量顺序。
        quaternion = pose[..., [6, 3, 4, 5]]  # (i, j, k, r) → (r, i, j, k)
        rotation = quaternion_to_matrix(quaternion)
        # 最后一行设为 [0, 0, 0, 1]，由此可统一用矩阵乘法复合旋转和平移。
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0:3, 0:3] = rotation
        matrix[..., 0:3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        """把 ``(..., 4, 4)`` 齐次矩阵还原为 ``(..., 7)`` 位姿。"""
        # 除以齐次尺度可兼容最后元素不是 1 的合法齐次矩阵；本模块生成的矩阵中它为 1。
        position = matrix[..., 0:3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0:3, 0:3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]]  # (r, i, j, k) → (i, j, k, r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose

    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        """把父/世界坐标系中的绝对位姿表达为 base 局部坐标系中的位姿。"""
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        """把 base 局部坐标系中的相对位姿放回父/世界坐标系。"""
        return base_pose_matrix @ pose_matrix

    @staticmethod
    def _quaternion_conjugate(q: torch.Tensor):
        """求 ``(i, j, k, r)`` 四元数的共轭（非 JIT 对照实现）。"""
        return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

    @staticmethod
    def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor):
        """计算 ``(i, j, k, r)`` 四元数的 Hamilton 积（非 JIT 对照实现）。"""
        i1, j1, k1, r1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        i2, j2, k2, r2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
        i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
        j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
        k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2

        return torch.stack([i, j, k, r], dim=-1)

    @staticmethod
    def _quaternion_rotate_vector(q: torch.Tensor, v: torch.Tensor):
        """用单位四元数旋转三维向量（非 JIT 对照实现）。"""
        i, j, k, r = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        x, y, z = v[..., 0], v[..., 1], v[..., 2]

        # 使用等价的四元数旋转展开式，比“先转旋转矩阵再乘向量”少一些开销。
        qx, qy, qz = i, j, k
        qw = r

        # t = 2 * cross(q.xyz, v)
        tx = 2 * (qy * z - qz * y)
        ty = 2 * (qz * x - qx * z)
        tz = 2 * (qx * y - qy * x)

        # v' = v + q.r * t + cross(q.xyz, t)
        rx = x + qw * tx + (qy * tz - qz * ty)
        ry = y + qw * ty + (qz * tx - qx * tz)
        rz = z + qw * tz + (qx * ty - qy * tx)

        return torch.stack([rx, ry, rz], dim=-1)

    def _forward_fast(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """快速正向实现：直接做四元数运算，不构造/求逆 4×4 矩阵。"""
        if self.input_rotation_type == "euler":
            return self._forward_euler(pose, base_pose)

        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )

        # 分离目标的平移和四元数；四元数顺序始终是 (i, j, k, r)。
        p_target = pose[..., :3]
        q_target = pose[..., 3:7]

        p_base = base_pose[..., :3]
        q_base = base_pose[..., 3:7]

        # 相对平移不能只做 p_target - p_base：这个差仍表达在父坐标系中。
        # 再乘 R_base^T，才会把它旋转到 base 的局部坐标系：
        #     p_rel = R_base^T * (p_target - p_base)
        # 单位四元数的共轭表示逆旋转，因此用 q_base_conj 旋转 delta_p。
        q_base_conj = quaternion_conjugate_jit(q_base)
        delta_p = p_target - p_base
        p_rel = quaternion_rotate_vector_jit(q_base_conj, delta_p)

        # 相对旋转是“先撤销 base 的旋转，再应用 target 的旋转”：
        #     q_rel = inverse(q_base) * q_target
        q_rel = quaternion_multiply_jit(q_base_conj, q_target)

        return torch.cat([p_rel, q_rel], dim=-1)

    def _backward_fast(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """快速反向实现：直接把相对平移/旋转复合到基准位姿上。"""
        if self.input_rotation_type == "euler":
            return self._backward_euler(pose, base_pose)

        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, (
            f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        )

        # 此处 pose 已是相对于 base 坐标系的 (p_rel, q_rel)。
        p_rel = pose[..., :3]
        q_rel = pose[..., 3:7]

        p_base = base_pose[..., :3]
        q_base = base_pose[..., 3:7]

        # 先把局部位移旋到父坐标系，再加基准位置：
        #     p_target = R_base * p_rel + p_base
        p_target = quaternion_rotate_vector_jit(q_base, p_rel) + p_base

        # 旋转复合顺序与齐次矩阵 T_base @ T_rel 一致。
        q_target = quaternion_multiply_jit(q_base, q_rel)

        return torch.cat([p_target, q_target], dim=-1)

    def _forward_euler(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """欧拉角正向实现：借助 3×3 旋转矩阵保证相对旋转的几何正确性。"""
        assert pose.shape[-1] == 6, f"Pose shape must be (..., 6), but got {pose.shape}"
        assert base_pose.shape[-1] == 6, (
            f"Base pose shape must be (..., 6), but got {base_pose.shape}"
        )

        p_target = pose[..., :3]
        e_target = pose[..., 3:6]
        p_base = base_pose[..., :3]
        e_base = base_pose[..., 3:6]

        # 欧拉角不能直接相减来表示一般的三维相对旋转；先转矩阵再复合。
        r_target = euler_angles_to_matrix(e_target, self.euler_convention)
        r_base = euler_angles_to_matrix(e_base, self.euler_convention)
        r_base_t = r_base.transpose(-1, -2)

        # 旋转矩阵是正交矩阵，所以 inverse(R_base) = transpose(R_base)。
        # unsqueeze 把 (..., 3) 变成列向量 (..., 3, 1)，乘完再 squeeze 回去。
        delta_p = (r_base_t @ (p_target - p_base).unsqueeze(-1)).squeeze(-1)
        r_rel = r_base_t @ r_target
        e_rel = matrix_to_euler_angles(r_rel, self.euler_convention)
        return torch.cat([delta_p, e_rel], dim=-1)

    def _backward_euler(self, pose: torch.Tensor, base_pose: torch.Tensor):
        """欧拉角反向实现：把局部相对位姿重新复合到基准位姿。"""
        assert pose.shape[-1] == 6, f"Pose shape must be (..., 6), but got {pose.shape}"
        assert base_pose.shape[-1] == 6, (
            f"Base pose shape must be (..., 6), but got {base_pose.shape}"
        )

        p_rel = pose[..., :3]
        e_rel = pose[..., 3:6]
        p_base = base_pose[..., :3]
        e_base = base_pose[..., 3:6]

        r_rel = euler_angles_to_matrix(e_rel, self.euler_convention)
        r_base = euler_angles_to_matrix(e_base, self.euler_convention)

        # 分别还原 p_target = R_base*p_rel+p_base 和 R_target = R_base*R_rel。
        p_target = (r_base @ p_rel.unsqueeze(-1)).squeeze(-1) + p_base
        r_target = r_base @ r_rel
        e_target = matrix_to_euler_angles(r_target, self.euler_convention)
        return torch.cat([p_target, e_target], dim=-1)


class DeltaPoseTransform(BaseActionStateTransform):
    """
    把绝对动作序列变成“相邻帧逐元素差分”，并可用累加恢复。

    对动作时间维 t：

        forward : delta[0] = action[0] - state[-1]
                  delta[t] = action[t] - action[t-1], t >= 1
        backward: action[0] = delta[0] + state[-1]
                  action[t] = action[0] + sum(delta[1:t+1])

    注意：这是所有特征维上的普通减法，并不是 RelativePoseTransform 那样的 SE(3)
    坐标系运算。如果特征里包含四元数，直接相减只具有数值差分含义，并不等价于
    相对旋转。该类适用于数据定义本来就允许逐元素差分的向量。

    ``keys`` 指定要处理的子键。键只在 action 中存在、却不在 state 中存在时会原样
    保留；不在 keys 中的 action 键同样透传。该变换可逆。
    """

    invertible = True

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        """沿动作时间维生成首帧对状态、后续帧对前一动作的差分。"""
        if "action" not in batch:
            return batch

        # 新建两层字典，但复用 state、元数据和未变换张量的引用；计算出的 delta 是新张量。
        out_batch = {
            "action": {},
            "state": batch["state"],
        }
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                action = batch["action"][k]
                state_last = batch["state"][k][..., -1:, :]
                # 保留时间维长度 1，以便与后续差分沿 dim=-2 拼接。
                first_delta = action[..., 0:1, :] - state_last
                if action.shape[-2] > 1:
                    # action[..., 1:, :] 与 action[..., :-1, :] 对齐相邻的后帧/前帧。
                    rest_delta = action[..., 1:, :] - action[..., :-1, :]
                    out_batch["action"][k] = torch.cat([first_delta, rest_delta], dim=-2)
                else:
                    out_batch["action"][k] = first_delta
            elif k in batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        # 补回不在 keys 中（或此前未覆盖到）的 action 项，保证字典结构完整。
        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch

    def backward(self, batch: Dict):
        """从首帧基准和后续差分累加还原绝对动作序列。"""
        if "action" not in batch:
            return batch

        out_batch = {
            "action": {},
            "state": batch["state"],
        }
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                delta = batch["action"][k]
                state_last = batch["state"][k][..., -1:, :]
                first_action = delta[..., 0:1, :] + state_last
                if delta.shape[-2] > 1:
                    # cumsum(delta[1:]) 得到相对于 action[0] 的累计变化；再整体加 first_action。
                    rest_cumsum = torch.cumsum(delta[..., 1:, :], dim=-2)
                    out_batch["action"][k] = torch.cat(
                        [first_action, first_action + rest_cumsum], dim=-2
                    )
                else:
                    out_batch["action"][k] = first_action
            elif k in batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch


class RelativeJointTransform(BaseActionStateTransform):
    """
    把绝对关节目标转换为相对于最后一帧状态的关节增量。

        forward : action_relative = action_absolute - state_last
        backward: action_absolute = action_relative + state_last

    与 DeltaPoseTransform 的区别是：这里 action 的每个未来时间步都减同一个
    ``state_last``，而不是让第 t 步减第 t-1 步。因此输出表示“每个目标相对当前
    机器人状态要走多远”，不是“动作轨迹中相邻两点的变化”。

    参数：
        keys: 要处理的关节组键，例如 ``["left_arm", "right_arm"]``。
        fast_forward: True 使用逐键运算和浅层容器复制；False 先拼接所有键、统一
            运算后再切分，并 deepcopy 整个 batch。正常配置默认使用快速路径。

    action 的典型形状为 ``(..., T_action, D)``，state 为
    ``(..., T_state, D)``；``state[..., -1:, :]`` 会广播到全部动作步。该变换可逆。
    """

    invertible = True

    def __init__(self, keys: List[str], fast_forward: bool = True):
        self.keys = keys
        self.fast_forward = fast_forward

    def forward(self, batch: Dict):
        """对指定关节键执行 ``action -= 最新 state``。"""
        # 闭环评估可能只有 state；缺任一顶层字段都没有足够数据做相对化。
        if "action" not in batch or "state" not in batch:
            return batch

        if self.fast_forward:
            return self._forward_fast(batch)
        else:
            return self._forward(batch)

    def _forward(self, batch: Dict):
        """原始实现：拼接各键统一相减，并深拷贝输出 batch。"""
        # 各关节组 D 可能不同，不能 stack；沿最后一维 cat 后记录每段宽度以便 split。
        action_parts = [batch["action"][k] for k in self.keys if k in batch["action"]]
        state_parts = [batch["state"][k] for k in self.keys if k in batch["state"]]
        dims = [t.shape[-1] for t in action_parts]

        if len(action_parts) == 0 or len(state_parts) == 0:
            return batch

        action_merged = torch.cat(action_parts, dim=-1)
        state_merged = torch.cat(state_parts, dim=-1)

        # 状态的最后一帧保留时间维，利用广播一次作用到整个动作 chunk。
        result = action_merged - state_merged[..., -1:, :]

        out_batch = deepcopy(batch)
        for k, t in zip(self.keys, result.split(dims, dim=-1)):
            out_batch["action"][k] = t

        return out_batch

    def _forward_fast(self, batch: Dict):
        """快速正向实现：避免 deepcopy，直接逐键产生新的相对动作张量。"""
        # 只新建顶层字典和 action 子字典；state 及其它字段仍与输入共享对象引用。
        # 本函数不修改这些共享对象，因此可安全省去昂贵的 tensor 深拷贝。
        out_batch = {
            "action": {},
            "state": batch["state"],  # 共享 state 引用，本函数不会修改它
        }

        # 观测、任务文本、索引等其它字段原样透传。
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        # 只有 action 和 state 中同时存在的键才能计算相对量。
        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                # 非原地减法会产生新张量，且省去了 cat/split 的临时大张量。
                out_batch["action"][k] = batch["action"][k] - batch["state"][k][..., -1:, :]
            elif k in batch["action"]:
                # 缺少对应状态时无法相对化，保留原动作，而不是静默删除该键。
                out_batch["action"][k] = batch["action"][k]

        # 把未配置在 self.keys 里的其余 action 键也完整透传。
        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch

    def backward(self, batch: Dict):
        """对指定关节键执行 ``action += 最新 state``，还原绝对目标。"""
        if "action" not in batch or "state" not in batch:
            return batch

        if self.fast_forward:
            return self._backward_fast(batch)
        else:
            return self._backward(batch)

    def _backward(self, batch: Dict):
        """原始反向实现：拼接各键统一相加，并深拷贝输出 batch。"""
        action_parts = [batch["action"][k] for k in self.keys if k in batch["action"]]
        state_parts = [batch["state"][k] for k in self.keys if k in batch["state"]]
        dims = [t.shape[-1] for t in action_parts]

        if len(action_parts) == 0 or len(state_parts) == 0:
            return batch

        action_merged = torch.cat(action_parts, dim=-1)
        state_merged = torch.cat(state_parts, dim=-1)

        result = action_merged + state_merged[..., -1:, :]

        out_batch = deepcopy(batch)
        for k, t in zip(self.keys, result.split(dims, dim=-1)):
            out_batch["action"][k] = t

        return out_batch

    def _backward_fast(self, batch: Dict):
        """快速反向实现：逐键加回状态，并复用所有未修改对象。"""
        # 与快速 forward 对称：新建 action 容器，state 仍共享引用。
        out_batch = {
            "action": {},
            "state": batch["state"],  # 共享 state 引用
        }

        # 其它顶层字段无需参与计算，直接透传。
        for k in batch:
            if k not in ("action", "state"):
                out_batch[k] = batch[k]

        # 加回每个键的最新状态，把相对关节量恢复为绝对关节目标。
        for k in self.keys:
            if k in batch["action"] and k in batch["state"]:
                # 普通加法产生新张量，不会原地改模型输出。
                out_batch["action"][k] = batch["action"][k] + batch["state"][k][..., -1:, :]
            elif k in batch["action"]:
                # 缺对应 state 的键从未被 forward 修改，这里也保持原值。
                out_batch["action"][k] = batch["action"][k]

        # 补回不属于本变换处理范围的 action 键。
        for k in batch["action"]:
            if k not in out_batch["action"]:
                out_batch["action"][k] = batch["action"][k]

        return out_batch


class ReorderLowerBodyTransform(BaseActionStateTransform):
    """调整 ``lower_body`` 最后一维中躯干与底盘数据的排列顺序。

    b1k 原始动作采用 ``[base_qvel(3), trunk_qpos(4)]``：前三维是底盘速度，
    后四维是躯干关节位置；WBC（全身控制器）及其 Cholesky 矩阵采用相反的
    ``[trunk_qpos(4), base_qvel(3)]``。forward 把 action 与 state 都对齐到 WBC
    顺序，backward 再恢复原始顺序。

    ``trunk_dim`` 与 ``base_dim`` 是两段宽度；默认总宽度为 7。实现只重排最后
    一维，因此前面的批次维和时间维都不受影响。此类会原地改 batch 字典中的张量引用。
    """

    invertible = True

    def __init__(self, key: str = "lower_body", trunk_dim: int = 4, base_dim: int = 3):
        self.key = key
        self.trunk_dim = trunk_dim
        self.base_dim = base_dim

    def forward(self, batch: Dict):
        """将 action/state 中目标键从“底盘在前”改为“躯干在前”。"""
        for domain in ("action", "state"):
            if domain in batch and self.key in batch[domain]:
                t = batch[domain][self.key]
                # 从 base_dim 处分段并交换：[底盘速度, 躯干位置] → [躯干位置, 底盘速度]。
                batch[domain][self.key] = torch.cat(
                    [t[..., self.base_dim :], t[..., : self.base_dim]], dim=-1
                )
        return batch

    def backward(self, batch: Dict):
        """将 action/state 中目标键恢复为原始的“底盘在前”顺序。"""
        for domain in ("action", "state"):
            if domain in batch and self.key in batch[domain]:
                t = batch[domain][self.key]
                # 从 trunk_dim 处分段并交换：[躯干位置, 底盘速度] → [底盘速度, 躯干位置]。
                batch[domain][self.key] = torch.cat(
                    [t[..., self.trunk_dim :], t[..., : self.trunk_dim]], dim=-1
                )
        return batch


class PartialRelativeTransform(BaseActionStateTransform):
    """只把一个键中指定的特征维转换为相对量，其余维度保持绝对值。

    典型用途是 ``trunk_qpos``：前三维需要表示相对于当前状态的增量，而第 4 维
    仍保持绝对目标。forward 对 ``relative_dims`` 逐维执行 ``action -= state``，
    backward 逐维加回，因此该变换可逆。

    状态既可能带时间维 ``(..., T_state, D)``，也可能已经是单帧 ``(..., D)``；
    当 state 与 action 的维数相同时，代码取 state 的最后一帧并保留长度为 1 的
    时间维，以便广播到全部动作步。

    注意：这里通过 ``action[..., d] = ...`` 原地修改 action 张量。如果调用方还需
    保留未变换动作，应在进入本变换前自行复制。
    """

    invertible = True

    def __init__(self, key: str, relative_dims: List[int]):
        self.key = key
        self.relative_dims = relative_dims

    def forward(self, batch: Dict):
        """从指定动作维减去对应的最新状态维。"""
        if "action" not in batch or "state" not in batch:
            return batch
        if self.key not in batch["action"] or self.key not in batch["state"]:
            return batch
        action = batch["action"][self.key]
        state = batch["state"][self.key]
        # 兼容带时间维与不带时间维的 state；常规同维场景明确取最后一帧。
        state_last = state[..., -1:, :] if state.ndim > action.ndim - 1 else state
        if state.ndim == action.ndim:
            state_last = state[..., -1:, :]
        # 逐维赋值只改变选中的维度，其它维度保持绝对值。
        for d in self.relative_dims:
            action[..., d] = action[..., d] - state_last[..., d]
        batch["action"][self.key] = action
        return batch

    def backward(self, batch: Dict):
        """给指定动作维加回对应的最新状态维。"""
        if "action" not in batch or "state" not in batch:
            return batch
        if self.key not in batch["action"] or self.key not in batch["state"]:
            return batch
        action = batch["action"][self.key]
        state = batch["state"][self.key]
        state_last = state[..., -1:, :] if state.ndim > action.ndim - 1 else state
        if state.ndim == action.ndim:
            state_last = state[..., -1:, :]
        for d in self.relative_dims:
            action[..., d] = action[..., d] + state_last[..., d]
        batch["action"][self.key] = action
        return batch


class BehaviorPerKeyTransform(BaseActionStateTransform):
    """面向 b1k behavior 数据、兼容 per-key ``shape_meta`` 的复合变换。

    预期输入键大致为：
      action: left_arm(7)、left_gripper(1/2)、right_arm(7)、right_gripper(1/2)，
              以及 lower_body(7)，或可合并的 trunk_qpos(4) + base_qvel(3)
      state:  left_arm(7)、left_gripper(2)、right_arm(7)、right_gripper(2)、
              trunk_qpos(4)、base_qvel(3)

    forward 按顺序完成：
      1. 若夹爪为 2 维，把两指开度求和成总宽度，再线性映射到 [-1, 1]；
      2. 若 action 的 trunk_qpos/base_qvel 分开存储，拼成
         lower_body = [trunk_qpos(4), base_qvel(3)]；
      3. 左右臂动作减去各自最新状态，变为相对关节目标；
      4. lower_body 中 ``trunk_relative_dims`` 指定的躯干维减去最新躯干状态；
      5. state 的 trunk_qpos/base_qvel 也拼成 lower_body，并删除原来的两个键。

    输出由此统一为逐部件结构：
      action: left_arm(7)、left_gripper(1)、right_arm(7)、right_gripper(1)、lower_body(7)
      state:  left_arm(7)、left_gripper(1)、right_arm(7)、right_gripper(1)、lower_body(7)

    ``alters_key_structure = True`` 会通知 processor：本变换会增删/合并键，不应按
    普通数值变换的规则强制检查键结构完全不变。

    重要限制：二维夹爪压成一维时只保留两指开度之和，原先每根手指各自的值已经
    丢失。backward 只能假设两指各占总宽度的一半。因此类属性沿用流水线需要的
    ``invertible = True``，但夹爪这一步并非严格的一一可逆；其余相对量与键合并步骤
    都有明确逆操作。此外，本类直接修改 batch 内的字典和部分张量。
    """

    invertible = True
    alters_key_structure = True

    def __init__(self, gripper_max_width: float = 0.1, trunk_relative_dims: List[int] = None):
        self.gripper_max_width = gripper_max_width
        self.trunk_relative_dims = trunk_relative_dims or [0, 1, 2]

    def forward(self, batch: Dict):
        """把原始 behavior 动作/状态原地整理成模型使用的逐键相对表示。"""
        if "state" not in batch:
            return batch

        # 取出的都是 batch 内部字典的引用，下面的赋值和 del 会直接反映到输入 batch。
        state = batch["state"]
        action = batch.get("action", {})

        # 1) 夹爪：两指开度相加得到总宽度，再由 [0, max_width] 映射到 [-1, 1]。
        #    若输入已经是 1 维，则视为已处理，避免重复缩放。
        for gk in ("left_gripper", "right_gripper"):
            for domain in (state, action):
                if gk in domain and domain[gk].shape[-1] == 2:
                    width = domain[gk].sum(dim=-1, keepdim=True)
                    domain[gk] = 2.0 * (width / self.gripper_max_width) - 1.0

        # 2) 动作下半身：仅在两个分离键都存在时合并为“躯干在前、底盘在后”。
        #    如果上游已经提供 lower_body，则保持现有值，不做重复合并。
        if "trunk_qpos" in action and "base_qvel" in action:
            action["lower_body"] = torch.cat([action["trunk_qpos"], action["base_qvel"]], dim=-1)
            del action["trunk_qpos"]
            del action["base_qvel"]

        # 3) 左右臂相对化。state/action 同维通常表示二者都有时间维，此时只取
        #    state 最后一帧；少一维则认为 state 已是一帧，让 PyTorch 直接广播。
        for k in ("left_arm", "right_arm"):
            if k in action and k in state:
                state_last = state[k][..., -1:, :] if state[k].ndim == action[k].ndim else state[k]
                action[k] = action[k] - state_last

        # 4) 只相对化 lower_body 中配置的躯干维。默认 [0,1,2]，第 4 个躯干维
        #    以及后面的 base_qvel 仍保持绝对值/原有语义。
        if "lower_body" in action and "trunk_qpos" in state:
            trunk_state = state["trunk_qpos"]
            trunk_state_last = (
                trunk_state[..., -1:, :]
                if trunk_state.ndim == action["lower_body"].ndim
                else trunk_state
            )
            for d in self.trunk_relative_dims:
                action["lower_body"][..., d] = (
                    action["lower_body"][..., d] - trunk_state_last[..., d]
                )

        # 5) 状态也统一成 lower_body。合并后删除来源键，防止同一信息保存两份并
        #    导致 shape_meta 与实际键结构不一致。
        if "trunk_qpos" in state and "base_qvel" in state:
            state["lower_body"] = torch.cat([state["trunk_qpos"], state["base_qvel"]], dim=-1)
            del state["trunk_qpos"]
            del state["base_qvel"]

        batch["state"] = state
        if action:
            batch["action"] = action
        return batch

    def backward(self, batch: Dict):
        """按 forward 的逆序，把模型空间中的结果恢复到原始 behavior 键结构。"""
        state = batch["state"]
        action = batch.get("action", {})

        # 逆步骤 5：先拆 state lower_body。暂时保留 lower_body，供后续动作还原使用；
        # 函数末尾再删除它。
        if "lower_body" in state:
            state["trunk_qpos"] = state["lower_body"][..., :4]
            state["base_qvel"] = state["lower_body"][..., 4:]

        # 逆步骤 4：给配置的躯干动作维加回最新 trunk_qpos 状态。
        if "lower_body" in action and "trunk_qpos" in state:
            trunk_state = state["trunk_qpos"]
            trunk_state_last = (
                trunk_state[..., -1:, :]
                if trunk_state.ndim == action["lower_body"].ndim
                else trunk_state
            )
            for d in self.trunk_relative_dims:
                action["lower_body"][..., d] = (
                    action["lower_body"][..., d] + trunk_state_last[..., d]
                )

        # 还原原始动作约定：[trunk(4), base(3)] → [base(3), trunk(4)]。
        # 注意 forward 只在动作以两个分离键输入时显式合并；此处统一输出原始 b1k
        # 所需的 lower_body 排列，而不会重新拆成两个 action 键。
        if "lower_body" in action:
            lb = action["lower_body"]
            action["lower_body"] = torch.cat([lb[..., 4:], lb[..., :4]], dim=-1)

        # 逆步骤 3：左右臂动作加回最新状态，恢复绝对关节目标。
        for k in ("left_arm", "right_arm"):
            if k in action and k in state:
                state_last = state[k][..., -1:, :] if state[k].ndim == action[k].ndim else state[k]
                action[k] = action[k] + state_last

        # 逆步骤 1：把归一化值先映射回总宽度，再平均分给两指。
        # 由于 forward 只保存了两指之和，这只是基于“两指等宽”的约定性重建，
        # 无法恢复输入中可能存在的左右指差异。
        for gk in ("left_gripper", "right_gripper"):
            if gk in state and state[gk].shape[-1] == 1:
                width = (state[gk] + 1.0) * self.gripper_max_width / 2.0
                state[gk] = torch.cat([width / 2, width / 2], dim=-1)

        # state 已拆回原来的两个键，删除中间态 lower_body，恢复原始键结构。
        if "lower_body" in state and "trunk_qpos" in state:
            del state["lower_body"]

        batch["state"] = state
        if action:
            batch["action"] = action
        return batch
