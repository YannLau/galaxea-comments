# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/transforms/base.py — 所有“数据变换”的抽象基类
#                                        （BaseActionStateTransform）
# =============================================================================
#
# 【这个文件是什么】
#   它定义了抽象基类 BaseActionStateTransform，即 data_processor 里“变换算子”这一族的
#   共同祖先。所谓变换算子，就是一个“吃进 sample/batch 字典、吐出一个（可能被改写的）
#   sample/batch 字典”的可组合运算：相对动作、旋转表示转换、动作/状态合并、归一化……
#   全都是它的子类。
#
#   基类本身不含任何数学逻辑，只规定 4 件事：
#     1) forward()        —— 必填。怎么把数据“变换过去”，是抽象方法，子类必须实现。
#     2) backward()       —— 选填。怎么把数据“变回来”，只有 invertible=True 时才需要实现；
#                            invertible=False 的子类直接继承默认实现（原样返回输入）。
#     3) invertible / rtol / atol
#                         —— 类属性，声明“这个变换是否可逆”“往返测试用什么容差”。
#     4) test_roundtrip() —— 自检工具：跑一遍 forward → backward，检查数据是否被还原。
#                            它依赖两个内部小工具：_deep_copy_batch()（深拷贝 batch）
#                            与 _compare_nested_dicts()（递归比较嵌套字典里的张量）。
#
# 【它在整条数据链路中的位置】
#
#   configs/data/*.yaml 里的 action_state_transforms 列表（每个元素带 _target_）
#        │  Hydra 反射实例化，按“列表顺序”拼成一条流水线
#        │  入口：src/g05/utils/data/processor_utils.py → BaseProcessor 构造参数
#        ▼
#   transforms/ 下的各个子类                     ← 本文件是它们的共同基类
#     · relative_action.py     相对位姿 / 相对关节角 / 增量动作（可逆，配套 backward）
#     · rotation.py            四元数 ↔ euler ↔ rotation_6d/9d 表示转换（可逆）
#     · action_state_merger.py 把多个 key 拼接 / 补零成一条定宽向量（可逆）
#     · action_filter.py       标记“哪些动作维是有效操作维”（不可逆，invertible=False）
#     · misc.py                其它小工具（arcsinh 压缩、角度环绕、二值化等）
#     · g05/utils/data/normalizer.py 里的 LinearNormalizer / SingleFieldLinearNormalizer
#                              同样继承本基类（虽然它在 utils 层，但契约一致）
#        │
#        ▼
#   由 BaseProcessor 驱动（src/g05/data_processor/processor/base_processor.py）：
#     · preprocess() ：action_filter.forward → transforms 正序 forward
#                      → normalizer.forward → action_state_merger.forward
#     · postprocess()：action_state_merger.backward → normalizer.backward
#                      → transforms 逆序 backward → action_filter.backward
#   也就是说：“正序 forward、逆序 backward”这条纪律由调用方（processor）保证；
#   而“同一算子的 forward 与 backward 必须互为逆运算”这条纪律由本基类 + 各子类保证。
#
# 【核心契约：batch 长什么样】
#   所有 forward / backward 的入参与返回值都是同一种“样本字典”：
#
#     batch = {
#       "action": { "left_arm": Tensor[..., T, 7],        # 动作 chunk：T 步，每步 7 维
#                   "left_gripper": Tensor[..., T, 1], ... },
#       "state":  { "left_arm": Tensor[..., T_obs, 7],    # 观测状态；相对动作以它的
#                   ... },                                # “最后一步 state[..., -1:, :]”为基准
#       "action_op_mask": {...},  # 动作过滤器产出的“有效操作维”掩码（部分变换会同步改它）
#       "task" / "idx" / "embodiment" / ...  # 透传的元信息（本层既不读也不改）
#     }
#   约定：
#     · 最后一维 = 该 part 的特征宽度 D；action 的倒数第二维 = 时间/动作步维 T。
#     · 变换只应改动自己关心的 key，其余内容原样透传（元信息尤其重要：
#       下游要靠 "task" / "idx" 组装训练样本、跑评估）。
#     · 允许“增删 key”的变换会额外声明 alters_key_structure = True
#       （例如把 trunk_qpos + base_qvel 合并成 lower_body 的 BehaviorPerKeyTransform）；
#       processor 见到这个标记后，会放宽“变换后维度断言”。
#     · 闭环评估时 batch 里可能根本没有 "action"（没有动作可执行），
#       所以子类的 forward 通常要先判断 "action" 是否存在，不存在就原样返回。
#
# 【边界：谁不属于这条“batch 字典”契约】
#   transforms/image.py 以及 configs/data/_transforms.yaml 里的那些类
#   （ToTensor、FlipChannels、torchvision.transforms.*）是 nn.Module，
#   接口是“单张图像张量 in → 单张图像张量 out”，既不是本基类的子类，
#   也不遵守 forward/backward 成对的可逆约定。
#   真正继承本基类的是“动作/状态级”算子：relative_action / rotation /
#   action_state_merger / action_filter / misc，以及 utils 里的 normalizer。
#
# 【为什么这个文件必须保持“零重依赖”】
#   src/g05/data_processor/__init__.py 只做一件事：把本文件的 BaseActionStateTransform
#   再导出给全仓库使用。原因是有循环导入的风险：
#     · g05/utils/data/normalizer.py 需要 ``from g05.data_processor import BaseActionStateTransform``；
#     · 而 transforms/* 、processor/* 又需要 normalizer。
#   如果把重量级依赖（Hydra、数据集、processor 等）搬进本文件，
#   就会形成 data_processor → normalizer → data_processor 的循环导入。
#   因此本文件只 import abc / typing / torch，且**永远不要在这里 import 本包内的其它模块**。
#
# 【新手最容易踩的 5 个坑】
#   1) 只写 forward 就以为完事：若子类保持 invertible=True（基类默认值），
#      推理/评估链路调用 backward() 时会直接抛 NotImplementedError（这是刻意报错，
#      避免“悄悄用错误空间里的动作去执行”）。
#      规则：不可逆就显式写 ``invertible = False``；声明可逆就必须把 backward() 补齐。
#   2) 以为 backward() 一定返回新对象：当 invertible=False 时，默认实现是
#      “原封不动返回传进来的那个 batch”（同一个字典对象、同一批张量引用），
#      而部分子类（如 BehaviorPerKeyTransform）是原地改字典的。
#      所以不要依赖“调用前后 batch 的内容不变”。
#   3) 在 forward 里原地改输入张量（如 ``x -= y`` 或 ``x[..., 0] = ...``）：
#      test_roundtrip 之所以先深拷贝一份原始数据，正是为了防住这种写法把“留底数据”
#      一起改掉、让往返测试永远自洽，却在 processor 里污染真实样本。
#      要改请先 ``clone()``，或返回新的张量。
#   4) 把 test_roundtrip 当成严格断言：它有三处明确局限——
#      · 只比较“原始 batch 里出现过的 key”（多出来的 key 不报错；少了的 key 记为 inf）；
#      · 只比较 Tensor 与嵌套 dict，其它类型（str / int / tuple / list）会被跳过；
#      · 阈值判断目前只用了绝对误差 atol，rtol 只是声明、尚未参与判定。
#   5) 以为变换顺序无所谓：action_state_transforms 是“列表顺序 = 正向执行顺序”，
#      反向由 processor 用 reversed() 回放，因此两个算子的先后会改变数值结果
#      （例如必须“先把 euler 姿态转成四元数，再做相对位姿”，否则相对位姿的几何意义是错的）。
#      调整顺序时，请同步核对同一份 yaml 里的 raw_shape / shape。
#
# 【相关文件与文档】
#   src/g05/data_processor/__init__.py                       本基类的再导出处（含循环导入说明）
#   src/g05/data_processor/processor/base_processor.py       action_state_transform() / postprocess()
#   src/g05/data_processor/transforms/relative_action.py     相对位姿 / 相对关节角 / 增量动作
#   src/g05/data_processor/transforms/rotation.py            旋转表示转换
#   src/g05/data_processor/transforms/action_state_merger.py 动作/状态拼接与补零
#   src/g05/data_processor/transforms/action_filter.py       invertible=False 的动作过滤器
#   src/g05/utils/data/normalizer.py                         LinearNormalizer（同样继承本基类）
#   configs/data/r1pro.yaml                                  动作/状态变换的真实配置示例
#   configs/data/_transforms.yaml                            图像变换的配置写法
#   docs/data/schema_zh.md                                   shape_meta 字段语义（raw_shape / shape）
# =============================================================================

"""数据变换的基类。

只依赖 abc / torch，刻意保持足够轻量，好让 ``g05.utils.data.normalizer`` 这类上游模块
能够安全导入，而不会与 processor 之间形成循环导入。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch


class BaseActionStateTransform(ABC):
    """
    所有数据变换的统一基类（抽象类，不能直接实例化）。

    子类必须实现 forward()；声明为可逆（invertible=True）的子类还必须实现 backward()，
    使 ``backward(forward(batch))`` 能把数据还原回 batch。

    典型用法是配置驱动的，在 configs/data/*.yaml 里写：

        action_state_transforms:
          - _target_: g05.data_processor.transforms.relative_action.RelativeJointTransform
            keys: [left_arm, right_arm]

    processor 会按列表顺序依次调用 forward()，反向时用 reversed() 逆序调用 backward()。

    类属性（写在类体里，子类按需覆盖）：
        invertible: 该变换是否可逆。为 True 时，backward() 必须能把 forward() 的输出
            还原成输入；为 False 时，backward() 直接原样返回输入。
            默认值是 True，因此“不可逆”的子类必须显式写 ``invertible = False``。
        rtol/atol: 往返测试（test_roundtrip）的误差容差。
            当前实现只用 atol 做绝对误差判定；rtol 是为将来加入相对误差判定预留的。
    """

    # 子类可覆盖此默认值；不可逆的子类必须显式改成 False（见 action_filter.py）。
    invertible: bool = True
    rtol: float = 1e-5
    atol: float = 1e-5

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """正向变换：把 batch 变换成“训练 / 推理需要的表示”。

        抽象方法，子类必须实现。这里的 raise NotImplementedError 只是形式上的兜底——
        @abstractmethod 已经保证“没实现 forward 的子类无法被实例化”。

        参数：
            batch: 样本字典，至少包含 "action" / "state" 之一，形如
                ``{"action": {key: Tensor[..., T, D]}, "state": {key: Tensor[..., T_obs, D]}}``。
                闭环评估时可能没有 "action"，实现里应先判断再返回。

        返回：
            变换后的样本字典。既可以返回新字典，也可以原地改完再返回同一个字典
            （processor 统一接收返回值，见 base_processor.action_state_transform）。
        """
        raise NotImplementedError

    def backward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """反向变换：forward() 的逆运算，把数据从“变换后的表示”还原回物理量 / 原始表示。

        行为按 invertible 分两种：
          · invertible=False：默认实现直接原样返回传进来的 batch（什么也不做）。
            注意返回的是同一个对象（引用相同），而不是副本。
          · invertible=True ：子类必须覆盖本方法；没覆盖就直接抛 NotImplementedError，
            避免“该还原时悄悄跳过”，让错误空间里的动作进入部署链路。

        调用方：processor.postprocess() 会用 ``reversed(action_state_transforms)`` 逆序调用，
        所以这里只需保证“本算子自己的逆运算正确”，不必关心流水线顺序。

        参数：
            batch: forward() 的输出（或与之等价的表示）。

        返回：
            还原后的样本字典。
        """
        if not self.invertible:
            # 不可逆变换：没有逆运算可做，原样返回（同一个对象，不拷贝）。
            return batch
        # 声明了可逆却没实现 backward：立刻报错，不要静默跳过。
        raise NotImplementedError(
            f"{self.__class__.__name__} is marked as invertible but backward() is not implemented"
        )

    def test_roundtrip(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        往返自检：forward → backward，检查原数据能否被还原。仅在 invertible=True 时有意义。

        典型用途：新增 / 修改一个可逆变换后，先拿一条真实样本跑一遍，确认还原误差在容差内，
        再把它接进训练配置。

        参数：
            batch: 待测试的样本字典。

        返回：
            测试结果字典：
            - passed: bool            # 全部通过（error_keys 为空）才为 True
            - max_error: float        # 所有被比较张量中的最大绝对误差
            - error_keys: List[str]   # 超出阈值的 key（嵌套字典写成 "外层.内层"）
            若 invertible=False，直接返回 ``{"passed": True, "reason": "not_invertible"}``，
            因为不可逆变换本来就无法（也不应该）还原。

        判定细节（新手请留意）：
          · 只比较“原始 batch 里存在的 key”：还原结果里多出的 key 不报错；
            原始 key 在还原结果里缺失则记为误差 inf，并计入 error_keys。
          · 只比较 Tensor（shape 必须一致，否则误差记 inf）和嵌套 dict（递归比较）；
            其它类型（str / int / tuple / list）直接跳过。
          · 阈值判断目前只用绝对误差 ``|orig - rec| > atol``；rtol 尚未参与判定。
        """
        if not self.invertible:
            return {"passed": True, "reason": "not_invertible"}

        # 先深拷贝一份原始数据“留底”：不少子类会原地修改输入（改字典或改张量），
        # 不先留底的话，forward 之后就没有“变换前”的数据可用于比较了。
        original = self._deep_copy_batch(batch)

        # 正向 → 反向。注意 forward 可能直接改写 batch，所以上面必须先拷贝。
        transformed = self.forward(batch)
        recovered = self.backward(transformed)

        # 逐个 key 比较误差：errors 记录每个 key 的误差值，error_keys 只记录超阈值的 key。
        errors = {}
        error_keys = []

        for k in original:
            # 还原结果里缺了这个 key：无法比较，按无穷大误差处理（视为失败）。
            if k not in recovered:
                errors[k] = float("inf")
                error_keys.append(k)
                continue

            orig_v, rec_v = original[k], recovered[k]

            if isinstance(orig_v, torch.Tensor) and isinstance(rec_v, torch.Tensor):
                if orig_v.shape != rec_v.shape:
                    # 形状都对不上，说明变换 / 还原改变了维度，直接记无穷大误差。
                    errors[k] = float("inf")
                    error_keys.append(k)
                else:
                    # 元素级最大绝对误差（标量化成 Python float，便于汇总）。
                    max_err = (orig_v - rec_v).abs().max().item()
                    errors[k] = max_err
                    if max_err > self.atol:
                        error_keys.append(k)
            elif isinstance(orig_v, dict) and isinstance(rec_v, dict):
                # 嵌套字典（如 batch 里的 "action" / "state"）：递归比较，
                # key 名会带上前缀，形如 "action.left_arm"。
                nested_result = self._compare_nested_dicts(orig_v, rec_v, k)
                errors.update(nested_result["errors"])
                error_keys.extend(nested_result["error_keys"])

        # 没有任何可比较项时（errors 为空）约定最大误差为 0.0，即“空测试视为通过”。
        max_error = max(errors.values()) if errors else 0.0

        return {
            "passed": len(error_keys) == 0,
            "max_error": max_error,
            "error_keys": error_keys,
        }

    def _deep_copy_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """深拷贝 batch，避免原地修改污染原始数据。

        只处理 batch 里实际会出现的 4 类值：
          · torch.Tensor → ``clone()``（复制数值，不与原张量共享存储）
          · dict         → 递归深拷贝（"action" / "state" / "action_op_mask" 都是这一层）
          · list         → 逐元素 clone 张量，非张量元素原样保留
          · 其它（str / int / float / None 等）→ 按引用透传（不可变对象，无需拷贝）

        局限：不处理 tuple，也不处理“列表里再嵌列表 / 字典”的情况。
        若某个变换要在 batch 里塞入这类结构，请自行扩展本方法。
        """
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.clone()
            elif isinstance(v, dict):
                # 递归下去，保证 action / state 里的每个张量都被复制。
                result[k] = self._deep_copy_batch(v)
            elif isinstance(v, list):
                result[k] = [item.clone() if isinstance(item, torch.Tensor) else item for item in v]
            else:
                # 字符串、数字等不可变对象共享引用即可。
                result[k] = v
        return result

    def _compare_nested_dicts(
        self, orig: Dict[str, Any], rec: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """递归比较两个嵌套字典中的张量，返回逐 key 的误差明细。

        参数：
            orig:   原始（变换前）的字典。
            rec:    还原之后得到的字典。
            prefix: key 前缀，用于把嵌套 key 拼成 "action.left_arm" 这类可读名字；
                    顶层调用留空即可（内部递归时会自动带上）。

        返回：
            ``{"errors": {full_key: max_abs_error}, "error_keys": [超过 atol 的 full_key]}``
        """
        errors = {}
        error_keys = []

        for k in orig:
            # 拼接完整 key 名：顶层是 "left_arm"，嵌套层是 "action.left_arm"。
            full_key = f"{prefix}.{k}" if prefix else k

            if k not in rec:
                # 还原结果里缺失该 key：无法比较，按无穷大误差处理。
                errors[full_key] = float("inf")
                error_keys.append(full_key)
                continue

            orig_v, rec_v = orig[k], rec[k]

            if isinstance(orig_v, torch.Tensor) and isinstance(rec_v, torch.Tensor):
                if orig_v.shape != rec_v.shape:
                    # 形状不一致说明还原过程改变了维度，直接记无穷大误差。
                    errors[full_key] = float("inf")
                    error_keys.append(full_key)
                else:
                    # 元素级最大绝对误差，超过 atol 才算失败。
                    max_err = (orig_v - rec_v).abs().max().item()
                    errors[full_key] = max_err
                    if max_err > self.atol:
                        error_keys.append(full_key)
            elif isinstance(orig_v, dict) and isinstance(rec_v, dict):
                # 更深的嵌套：继续递归，把 full_key 当作新的前缀往下传。
                nested = self._compare_nested_dicts(orig_v, rec_v, full_key)
                errors.update(nested["errors"])
                error_keys.extend(nested["error_keys"])

        return {"errors": errors, "error_keys": error_keys}
