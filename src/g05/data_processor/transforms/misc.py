# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# misc.py —— 动作 / 状态数据的通用小型变换
# =============================================================================
#
# 【这个文件在做什么】
#   本文件收纳不适合归入“相对动作”“旋转表示”等专门模块的通用算子。它们都继承
#   BaseActionStateTransform，因此遵循同一套接口：
#
#       原始 batch ── forward() ──> 训练 / 模型使用的表示
#       执行用 batch <── backward() ── 模型输出的表示（仅可逆算子）
#
#   BaseProcessor 会按 YAML 中 action_state_transforms 的书写顺序调用 forward()，
#   后处理时再按相反顺序调用 backward()。所以每个类只负责“一步变换”，流水线的
#   编排由 processor 完成。
#
# 【batch 的核心结构】
#   这些算子接收的不是单个 Tensor，而是嵌套字典，例如：
#
#       batch = {
#           "action": {"left_arm": Tensor[..., T, 7], ...},
#           "state": {"left_arm": Tensor[..., T_obs, 7], ...},
#           "action_op_mask": {"left_arm": Tensor[..., T, 7], ...},
#           "gt_action": {"left_arm": Tensor[..., T, 7], ...},
#           # task、idx、图像等其它字段由本文件原样透传
#       }
#
#   最后一维始终是特征维 D；前面的维度可以是批次维、时间维等。因此，本文件需要
#   拼接或检查特征宽度时统一操作 shape[-1]，而不假定前面究竟有多少维。
#
# 【五个算子的职责】
#   · ConcatKeysTransform：把多个身体部件的 key 沿特征维拼成一个 key，并可拆回。
#   · WrapStateAngle：把状态角度环绕到 [-pi, pi]，消除跨圈后的大数值。
#   · BinarizeTransform：用严格的 “> 阈值” 把连续值离散为 high / low 两档。
#   · ArcSinhTransform：用 arcsinh 平滑压缩动作中的大幅值，且能够精确求逆。
#   · LinearTailTransform：保留常见区间内的原值，只对分布两端做对数压缩。
#
# 【实现时为何经常先 dict(...)】
#   ``dict(batch)`` 只复制最外层字典，嵌套字典和 Tensor 仍共享引用；随后再复制
#   ``batch["action"]``，便能只替换 action 下的目标 Tensor，而不复制图像等大型数据。
#   这是一种“按需复制（copy-on-write）”风格。例外是 WrapStateAngle：它会直接改写
#   ``batch["state"]``，调用者不能假设输入 batch 保持不变。
#
# 【与仓库其它代码的关系】
#   · 基类契约：src/g05/data_processor/transforms/base.py
#   · 调用顺序：src/g05/data_processor/processor/base_processor.py
#   · 配置示例：configs/data/bridge.yaml（BinarizeTransform）
#   · 拼接类的 WBC 配置提示：configs/data/r1pro_wbc.yaml
# =============================================================================

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

import torch

from g05.data_processor import BaseActionStateTransform


class ConcatKeysTransform(BaseActionStateTransform):
    """
    沿最后一个维度，把多个 key 对应的张量拼成一个新 key。

    该操作会同步作用于 batch 中存在的 ``action``、``state``、``action_op_mask``
    和 ``gt_action`` 子字典，避免“动作拼过了、掩码却没拼”造成维度错位。
    WBC（全身控制）配置中的典型用途是：
    ``chassis_velocities(3D) + torso(4D) → lower_body(7D)``。

    ``invertible=True`` 表示它可逆：``backward`` 会根据 ``input_sizes``，把
    ``output_key`` 沿最后一维拆回原来的 ``input_keys``。

    参数：
        input_keys: 要拼接的源 key，顺序同时决定各片段在输出中的排列顺序。
        input_sizes: 每个源 key 的特征宽度，供反向拆分使用；必须与 input_keys 等长。
        output_key: 拼接后写入的新 key。

    注意：forward 允许某些 input_key 缺失，并只拼接实际存在的 key；但 backward
    仍会严格按完整的 input_sizes 拆分。因此，要做可靠的往返变换时，应保证所有
    input_keys 同时存在。这个宽松的 forward 行为主要用于兼容字段不完整的 batch。
    """

    invertible = True

    def __init__(self, input_keys: List[str], input_sizes: List[int], output_key: str):
        # 列表必须一一对应，否则反向时无法知道每个 key 应取多宽的切片。
        assert len(input_keys) == len(input_sizes), (
            "input_keys and input_sizes must have same length"
        )
        self.input_keys = input_keys
        self.input_sizes = input_sizes
        self.output_key = output_key

    def _concat(self, d: Dict) -> Dict:
        """拼接一个子字典（如 action），不直接改写传入的 d。"""
        # 保持 input_keys 声明的顺序，并忽略当前子字典里不存在的 key。
        present = [(k, d[k]) for k in self.input_keys if k in d]
        if not present:
            # 一个目标 key 都没有时直接复用原字典，避免没有意义的复制。
            return d
        out = dict(d)
        # dim=-1 表示只合并特征维；批次维、时间维等前导维度必须彼此一致。
        out[self.output_key] = torch.cat([v for _, v in present], dim=-1)
        # 拼接后删除源 key，确保同一份信息不会以两套 key 重复存在。
        for k, _ in present:
            del out[k]
        return out

    def _split(self, d: Dict) -> Dict:
        """按配置的特征宽度拆分 output_key，恢复所有 input_keys。"""
        if self.output_key not in d:
            return d
        out = dict(d)
        # Tensor.split 会校验各段宽度之和是否等于最后一维，配置不一致会立即报错。
        parts = out.pop(self.output_key).split(self.input_sizes, dim=-1)
        for k, t in zip(self.input_keys, parts):
            out[k] = t
        return out

    def forward(self, batch: Dict) -> Dict:
        """对 batch 中所有需要保持结构同步的字段执行拼接。"""
        batch = dict(batch)
        for field in ("action", "state", "action_op_mask", "gt_action"):
            if field in batch:
                batch[field] = self._concat(batch[field])
        return batch

    def backward(self, batch: Dict) -> Dict:
        """以与 forward 相同的字段范围，把合并后的 key 拆回去。"""
        batch = dict(batch)
        for field in ("action", "state", "action_op_mask", "gt_action"):
            if field in batch:
                batch[field] = self._split(batch[field])
        return batch


class WrapStateAngle(BaseActionStateTransform):
    """
    使用 ``atan2(sin(x), cos(x))`` 把指定状态角度环绕到 ``[-pi, pi]``。

    三角函数只保留角度在单位圆上的位置。例如 ``0``、``2*pi`` 和 ``-2*pi`` 会得到
    相同结果，原角度转过了多少整圈（winding number）已经丢失，所以该变换不可逆。

    参数：
        keys: ``batch["state"]`` 中需要环绕的 key。每个 key 对应张量的所有元素都会
            被视为弧度值；这里不做“哪些维度是角度”的细粒度筛选。

    注意：该类会原地替换 ``batch["state"][key]``。此外，它假定 ``state`` 与所有
    配置的 key 都存在；配置错误会直接抛 KeyError，以便尽早暴露数据 schema 问题。
    """

    invertible = False

    def __init__(self, keys: List[str]):
        self.keys = keys

    @staticmethod
    def _wrap(x):
        # 相比取模写法，这一公式天然适用于 Tensor，并在 ±pi 附近保持周期语义。
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, batch):
        """原地环绕指定 state key，并返回同一个 batch 对象。"""
        for k in self.keys:
            batch["state"][k] = self._wrap(batch["state"][k])
        return batch

    def backward(self, batch):
        # 信息已丢失，无法恢复；保留此方法是为了清楚表达调用时的透传行为。
        return batch


class BinarizeTransform(BaseActionStateTransform):
    """
    使用严格阈值比较，把指定 action / state key 二值化。

    规则是 ``x > threshold`` 时输出 ``max``，否则输出 ``min``。特别注意，恰好等于
    threshold 的值会落到 min 一侧。默认作用于 action 和 state；也可以用 fields
    改成其它同结构字段。

    thresholds / max / min 支持三类写法：
        1. 标量：所有 key、所有特征维共用一个值，例如 ``thresholds=0.5``；
        2. 按 key 的字典，字典值仍为标量，例如 ``{"gripper": 0.5}``；
        3. 按 key 的字典，字典值为一维列表，例如 ``{"hand": [0.2, 0.8]}``，
           从而为最后一维的每个通道设置不同参数。

    广播发生在最后一维：标量或单元素列表可广播到任意特征宽度；多元素列表的长度
    必须等于 ``x.shape[-1]``。参数张量会被搬到 x 所在设备并转换成 x 的 dtype，
    避免 CPU/GPU 或 float32/float64 不一致。

    这是不可逆变换：一个区间内的许多连续输入会坍缩成同一个离散值，backward
    无法知道原值，因此只能透传。
    """

    invertible = False

    def __init__(
        self,
        keys: List[str],
        thresholds: Any = 0.5,
        max: Any = 1.0,
        min: Any = 0.0,
        fields: Optional[List[str]] = None,
    ):
        self.keys = keys
        # 未显式指定时同时处理动作与状态；tuple 可防止外部意外修改字段列表。
        self.fields = tuple(fields or ("action", "state"))
        # 初始化时先检查按 key 配置是否完整，运行到训练中途才报错会更难排查。
        self._thresholds = self._normalize_param("thresholds", thresholds)
        self._max = self._normalize_param("max", max)
        self._min = self._normalize_param("min", min)

    def _normalize_param(self, name: str, value: Any) -> Any:
        """校验字典参数覆盖了全部目标 key，并复制一份以隔离外部修改。"""
        if isinstance(value, Mapping):
            missing = [key for key in self.keys if key not in value]
            if missing:
                raise ValueError(f"BinarizeTransform: {name} missing keys {missing}")
            return dict(value)
        return value

    @staticmethod
    def _is_sequence(value: Any) -> bool:
        """判断是否是可转为逐维参数的序列；字符串不能被当成数值列表。"""
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    def _value_for_key(self, name: str, params: Any, key: str, x: torch.Tensor) -> torch.Tensor:
        """取出某个 key 的参数，并规范成可与 x 广播运算的 Tensor。"""
        # 字典配置按 key 查找；非字典配置则由所有 key 共用。
        value = params[key] if isinstance(params, Mapping) else params
        if torch.is_tensor(value):
            # .to 同时对齐设备和 dtype；不会无条件复制已经匹配的 Tensor。
            tensor = value.to(device=x.device, dtype=x.dtype)
        elif self._is_sequence(value):
            tensor = torch.tensor(list(value), device=x.device, dtype=x.dtype)
        else:
            tensor = torch.tensor(value, device=x.device, dtype=x.dtype)

        # 只接受标量（0 维）或沿特征维广播的一维参数，不接受矩阵等含糊写法。
        if tensor.ndim > 1:
            raise ValueError(
                f"BinarizeTransform: {name}[{key!r}] must be a scalar or 1D list, "
                f"got shape {tuple(tensor.shape)}"
            )
        # 单元素向量可广播；否则必须与输入的最后一维完全对齐。
        if tensor.ndim == 1 and tensor.numel() not in (1, x.shape[-1]):
            raise ValueError(
                f"BinarizeTransform: {name}[{key!r}] last dim {tensor.numel()} "
                f"does not match input last dim {x.shape[-1]}"
            )
        return tensor

    def _apply_dict(self, values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """在一个 action/state 风格的子字典中二值化目标 key。"""
        out = dict(values)
        for key in self.keys:
            if key not in out:
                # 允许不同数据源缺少某个可选部件；缺失项不创建、不报错。
                continue
            x = out[key]
            threshold = self._value_for_key("thresholds", self._thresholds, key, x)
            high = self._value_for_key("max", self._max, key, x)
            low = self._value_for_key("min", self._min, key, x)
            # torch.where 不会原地修改 x；条件为 False（包括等于阈值）时选择 low。
            out[key] = torch.where(x > threshold, high, low)
        return out

    def forward(self, batch: Dict) -> Dict:
        """复制待修改的字典层，并对配置的 fields 逐一应用二值化。"""
        out_batch = dict(batch)
        for field in self.fields:
            if field in batch:
                out_batch[field] = self._apply_dict(batch[field])
        return out_batch

    def backward(self, batch: Dict) -> Dict:
        # 二值化已经丢失连续值信息，不存在真正的逆运算。
        return batch


class ArcSinhTransform(BaseActionStateTransform):
    """
    对指定 action key 应用反双曲正弦（arcsinh）压缩。

    对 bounds 中的每个 key，正向公式为：

        y = asinh(x / s)，其中 s = 1.5 * bound

    反向公式为：

        x = s * sinh(y)

    asinh 是奇函数，因此会保留正负号；在零附近近似线性，在绝对值较大时近似对数，
    可以温和压缩离群动作，又不会像截断那样丢失信息。s 决定进入明显压缩区间的尺度。

    参数：
        bounds: ``{action_key: [每个特征维的正数尺度, ...]}``。列表长度必须等于
            对应动作张量的最后一维。所有 bound 必须严格大于 0，避免除零和符号翻转。

    该算子只处理 ``batch["action"]``，bounds 中未出现在当前 batch 的 key 会跳过。
    类本身不是 ``torch.nn.Module``，所以这些常量不会随 ``module.to(device)`` 自动迁移，
    必须在每次处理张量时显式搬到相同设备。
    """

    invertible = True

    def __init__(self, bounds: Dict[str, List[float]]):
        # s 会作为除数，因此每一维的 bound 都必须为正。
        for key, bound_list in bounds.items():
            for v in bound_list:
                if v <= 0:
                    raise ValueError(f"bounds must be positive, got bounds[{key!r}] = {bound_list}")
        # 配置只在初始化时转一次 Tensor；默认留在 CPU，使用时再迁移设备。
        self._bounds: Dict[str, torch.Tensor] = {
            key: torch.tensor(bound_list, dtype=torch.float32) for key, bound_list in bounds.items()
        }

    def forward(self, batch: Dict) -> Dict:
        """压缩动作值，并保持 batch 的其它字段和未命中 key 不变。"""
        out_batch = dict(batch)
        # 复制 action 子字典后再替换 Tensor，避免改写调用者的 action 映射。
        out_batch["action"] = dict(batch["action"])
        for key, bound in self._bounds.items():
            if key not in out_batch["action"]:
                continue
            x = out_batch["action"][key]
            # 每个 bound 对应最后一维的一个动作通道，必须逐维对齐。
            if x.shape[-1] != bound.shape[0]:
                raise ValueError(
                    f"ArcSinhTransform: action[{key!r}] has shape {tuple(x.shape)}, "
                    f"but bounds has {bound.shape[0]} elements"
                )
            # 本类不是 nn.Module，常量不会自动随模型迁移设备，因此每批都需显式对齐。
            s = 1.5 * bound.to(x.device)
            out_batch["action"][key] = torch.asinh(x / s)
        return out_batch

    def backward(self, batch: Dict) -> Dict:
        """用 sinh 精确撤销 forward 的 asinh 压缩。"""
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for key, bound in self._bounds.items():
            if key not in out_batch["action"]:
                continue
            x = out_batch["action"][key]
            if x.shape[-1] != bound.shape[0]:
                raise ValueError(
                    f"ArcSinhTransform: action[{key!r}] has shape {tuple(x.shape)}, "
                    f"but bounds has {bound.shape[0]} elements"
                )
            # 与 forward 使用完全相同的 s，才能保证数值上的往返一致性。
            s = 1.5 * bound.to(x.device)
            out_batch["action"][key] = s * torch.sinh(x)
        return out_batch


class LinearTailTransform(BaseActionStateTransform):
    """
    分段变换：``[q01, q99]`` 区间内保持原值，区间外用 ``log1p`` 压缩极端值。

    正向变换：
      x > q99[i]:  f = q99[i]  + c_pos[i] * log1p((x - q99[i])  / c_pos[i])
      q01[i] ≤ x ≤ q99[i]: f = x
      x < q01[i]:  f = q01[i]  - c_neg[i] * log1p((q01[i] - x)  / c_neg[i])

    其中：
      c_pos[i] = tail_scale * (q99[i] - mean[i])    # 第 i 维正侧压缩系数
      c_neg[i] = tail_scale * (mean[i] - q01[i])    # 第 i 维负侧压缩系数

    反向变换（正向变换的精确逆函数）：
      y > q99[i]:  x = q99[i]  + c_pos[i] * expm1((y - q99[i])  / c_pos[i])
      q01[i] ≤ y ≤ q99[i]: x = y
      y < q01[i]:  x = q01[i]  - c_neg[i] * expm1((q01[i] - y)  / c_neg[i])

    ``log1p(z)`` 即 ``log(1 + z)``，在 z 接近 0 时比直接计算更稳定；``expm1``
    同理用于稳定计算 ``exp(z) - 1``。边界处两段公式都等于 q01 / q99，且尾部公式
    在边界处斜率为 1，所以函数连续、过渡平滑，不会给常见区间内的数据引入失真。

    tail_scale 越小，尾部压缩越强。以正侧为例，参考效果如下（默认值 0.075）：
      0.075：mean + 2*sigma → mean + 1.20*sigma
      0.100：mean + 2*sigma → mean + 1.24*sigma
      0.050：mean + 2*sigma → mean + 1.15*sigma

    参数：
        mean: 每个动作 key、每个特征维的均值。
        q01: 每维的低分位边界（名称通常表示 1% 分位数）。
        q99: 每维的高分位边界（名称通常表示 99% 分位数）。
        tail_scale: 尾部压缩强度系数，必须严格大于 0。

    mean、q01、q99 必须包含完全相同的 key；每一维还必须满足
    ``q01 < mean < q99``。该类只处理 action，当前 batch 中缺失的已配置 key 会跳过。
    """

    invertible = True

    def __init__(
        self,
        mean: Dict[str, List[float]],
        q01: Dict[str, List[float]],
        q99: Dict[str, List[float]],
        tail_scale: float = 0.075,
    ):
        # 非正值会令压缩系数为零或反号，使公式失去单调可逆性。
        if tail_scale <= 0:
            raise ValueError(f"tail_scale must be > 0, got {tail_scale}")

        # 这些统计量按 action key 保存，每个 Tensor 的长度等于该 key 的特征宽度。
        self._q01: Dict[str, torch.Tensor] = {}
        self._q99: Dict[str, torch.Tensor] = {}
        self._c_pos: Dict[str, torch.Tensor] = {}
        self._c_neg: Dict[str, torch.Tensor] = {}

        # 使用并集遍历，能明确指出“只出现在某一份统计量中”的遗漏 key。
        keys = set(mean) | set(q01) | set(q99)
        for key in keys:
            if key not in mean or key not in q01 or key not in q99:
                raise ValueError(
                    f"LinearTailTransform: key {key!r} must appear in all of mean/q01/q99"
                )
            m = torch.tensor(mean[key], dtype=torch.float32)
            lo = torch.tensor(q01[key], dtype=torch.float32)
            hi = torch.tensor(q99[key], dtype=torch.float32)
            # 每一维都要有非空的线性区间。
            if not (lo < hi).all():
                raise ValueError(
                    f"LinearTailTransform: q01 must be < q99 for key {key!r}, "
                    f"got q01={lo.tolist()}, q99={hi.tolist()}"
                )
            # mean 必须严格位于区间内部，才能保证两侧压缩系数均为正。
            if not ((m > lo) & (m < hi)).all():
                raise ValueError(
                    f"LinearTailTransform: mean must be strictly inside (q01, q99) for key {key!r}, "
                    f"got mean={m.tolist()}, q01={lo.tolist()}, q99={hi.tolist()}"
                )
            # 这里的 sigma_pos / sigma_neg 是“均值到分位边界的距离”，并不一定是
            # 统计学定义中的标准差；变量名用于表达正、负两侧各自的典型尺度。
            sigma_pos = hi - m  # 严格大于 0
            sigma_neg = m - lo  # 严格大于 0
            self._q01[key] = lo
            self._q99[key] = hi
            self._c_pos[key] = tail_scale * sigma_pos
            self._c_neg[key] = tail_scale * sigma_neg

    def _apply_forward(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """对单个 action Tensor 应用逐元素的分段正向公式。"""
        # 本类不是 nn.Module，构造时保存在 CPU 的常量需要按输入设备迁移。
        q01 = self._q01[key].to(x.device)
        q99 = self._q99[key].to(x.device)
        c_pos = self._c_pos[key].to(x.device)
        c_neg = self._c_neg[key].to(x.device)

        # clamp 让两个候选尾部公式在其非适用区域也保持定义良好。虽然 torch.where
        # 最终只选中对应分支，但 PyTorch 会先计算两个候选表达式。
        pos_tail = q99 + c_pos * torch.log1p(torch.clamp((x - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.log1p(torch.clamp((q01 - x) / c_neg, min=0.0))

        # 优先选正尾，再选负尾；落在闭区间 [q01, q99] 内时直接保留 x。
        return torch.where(x > q99, pos_tail, torch.where(x < q01, neg_tail, x))

    def _apply_backward(self, y: torch.Tensor, key: str) -> torch.Tensor:
        """用 expm1 对单个 action Tensor 撤销尾部的 log1p 压缩。"""
        q01 = self._q01[key].to(y.device)
        q99 = self._q99[key].to(y.device)
        c_pos = self._c_pos[key].to(y.device)
        c_neg = self._c_neg[key].to(y.device)

        # 与正向相同，clamp 保障未被选择的候选分支也不会接收错误符号的距离。
        pos_tail = q99 + c_pos * torch.expm1(torch.clamp((y - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.expm1(torch.clamp((q01 - y) / c_neg, min=0.0))

        return torch.where(y > q99, pos_tail, torch.where(y < q01, neg_tail, y))

    def forward(self, batch: Dict) -> Dict:
        """仅复制并更新 action 子字典，对其它 batch 字段零拷贝透传。"""
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for key in self._q99:
            if key not in out_batch["action"]:
                # 允许同一份变换配置服务于只含部分动作 key 的数据样本。
                continue
            x = out_batch["action"][key]
            # 统计量按特征维定义，长度必须与动作最后一维一致。
            if x.shape[-1] != self._q99[key].shape[0]:
                raise ValueError(
                    f"LinearTailTransform: action[{key!r}] last dim {x.shape[-1]} "
                    f"!= bounds shape {self._q99[key].shape[0]}"
                )
            out_batch["action"][key] = self._apply_forward(x, key)
        return out_batch

    def backward(self, batch: Dict) -> Dict:
        """对已压缩的 action 应用精确逆函数，恢复到原动作数值空间。"""
        out_batch = dict(batch)
        out_batch["action"] = dict(batch["action"])
        for key in self._q99:
            if key not in out_batch["action"]:
                continue
            x = out_batch["action"][key]
            if x.shape[-1] != self._q99[key].shape[0]:
                raise ValueError(
                    f"LinearTailTransform: action[{key!r}] last dim {x.shape[-1]} "
                    f"!= bounds shape {self._q99[key].shape[0]}"
                )
            out_batch["action"][key] = self._apply_backward(x, key)
        return out_batch
