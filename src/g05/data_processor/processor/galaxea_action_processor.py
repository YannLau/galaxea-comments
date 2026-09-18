# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/processor/galaxea_action_processor.py
#      —— 动作过滤器 R1LiteJointActionFilter：逐 key、逐维判断“这一维到底动没动”
# =============================================================================
#
# 【这个文件是什么】
#   本文件只定义一个类：``R1LiteJointActionFilter``。名字里的 R1Lite 来自 Galaxea R1 Lite
#   机器人，但 droid / libero 等数据配置也在复用同一个类。它是动作过滤器基类
#   ``BaseActionFilter``（transforms/action_filter.py）的子类，属于数据流水线里**第一步**
#   要执行的东西（在“相对动作变换 / 归一化 / 合并”之前）。
#
#   一句话职责：给定一段“原始动作块”，逐 key、逐维地判断这一维在这段时间里有没有真的在动，
#   产出一个布尔掩码 ``action_op_mask``：
#
#       True  = 该维“有效 / 有操作”（这段动作里动过，应该由模型预测）
#       False = 该维“静止 / 未使用 / 本具身根本没有这个部件”
#
#   注意它的输出不是新动作，而是**标记**。真正消费这个标记的是下游三处：
#     · 动作 tokenizer（encode）：``dropout_noop_parts`` 打开时，整段都判定为静止的 part
#       （例如“一直没动的那条胳膊”）会被直接从 token 序列里丢掉，模型只为动起来的部件生成
#       token —— 这正是 G0.5 论文里 “Only active motion groups are emitted during action
#       generation” 所说的那件事（README 的 overview 第 2 点）；
#     · 评估脚本：scripts/utils/metric.py 用 ``~action_op_mask`` 把“本来就不动的维”排除在
#       L1 / 二值精度之外，避免“模型没有瞎动”反而被算成误差，同时统计
#       “实际预测了哪些 part / 应该预测哪些 part”的比例；
#     · 反向流程：本文件的 ``backward()`` 会在还原后的物理动作里把这些不动的维写成 0。
#
# 【先搞清楚 action_op_mask 的形状与生命周期】（新手最容易在这里迷路）
#   本文件 forward() 产出的是**字典**：{key: bool 张量 (D,)}，一个 key 一个向量。
#   它离开本文件后会被“动作/状态合并器”搬运并改写形状：
#
#     本文件 forward 产出          {left_arm: (6,), left_gripper: (1,), ...}
#       │  action_state_merger.forward()
#       │    —— 按 max_action_shape_meta 对齐（多出来的维度补 False、缺失的 key 给全 False）
#       │       再拼成一条定长向量
#       ▼
#     样本字段 sample["action_op_mask"]   (D_action,)    ← dict 变张量，padding 位是 False
#       │  DataLoader collate
#       ▼
#     模型/评估侧 batch["action_op_mask"] (B, D_action)（旧缓存里可能出现 (B, 1, D_action)，
#                                           scripts/utils/metric.py 会兼容地 squeeze 掉中间维）
#       │  action_state_merger.backward() —— 反向走一遍，张量再切回每个 key
#       ▼
#     本文件 backward() 拿到的又变回字典：{key: (D,)}；推理路径上可能带 batch 维 (B, D)。
#       （(D,) 与 (B, D) 都能靠广播作用在动作张量的最后一维上，所以这里不必区分。）
#
# 【它在整条数据链路中的位置】
#
#   configs/data/<task>.yaml 的 processor.action_filter:
#       _target_: g05.data_processor.processor.galaxea_action_processor.R1LiteJointActionFilter
#       joint_threshold / gripper_threshold / velocity_threshold / eef_threshold / dim_thresholds
#        │  Hydra 按 _target_ 反射实例化；入口是 g05/utils/data/processor_utils.py 的
#        │  build_processors()（它还要把 task 层 model.processor 合并进来）
#        │  ⚠️ 合并语义是“task 层覆盖数据层”：configs/task/r1lite.yaml、r1pro.yaml、
#        │     r1pro_wbc.yaml 在 model.processor 下把 action_filter 的 _target_ 换成了
#        │     DummyActionFilter（= 所有维都算有效）。由于 OmegaConf 是**递归合并**，
#        │     数据层遗留的阈值参数（joint_threshold / dim_thresholds …）仍会一并透传过去
#        │     （父类 __init__ 能接收，于是被静默忽略）—— 总之这几个任务里本类根本不会被实例化。
#        ▼
#   BaseProcessor.__init__
#       self.action_filter = action_filter
#       self.action_filter.set_shape_meta(self.shape_meta)
#     （set_shape_meta 由父类 BaseActionFilter 实现，把 shape_meta["action"] 存进
#       self.action_meta —— 本文件的 forward / backward 就靠它决定要遍历哪些 key）
#        ▼
#   训练：dataset.__getitem__ 末尾 → processor.preprocess(sample)
#       base_processor.preprocess_action_state() 的前四步：
#           data = self.action_filter.forward(data)        ← 本文件的 forward
#           data = self.action_state_transform(data)        （相对动作 / 旋转表示等可逆变换）
#           data = self.normalizer.forward(data)            （按数据集统计量归一化）
#           data = self.action_state_merger.forward(data)   （对齐 + 拼成定长向量，mask 跟着走）
#       ⇒ 关键推论：本文件看到的是**未变换、未归一化的原始动作**（这就是下面断言用
#         raw_shape、以及“与块首帧作差”能成立的原因）
#        ▼
#   推理 / 离线评估：processor.postprocess(...) 的反向链
#           merger.backward → normalizer.backward → transforms 逆序 → action_filter.backward
#                                                                    ↑ 本文件的 backward（置 0）
#       最后再按 [:, num_obs_steps-1:, :] 切掉作为条件的观测前缀，只留预测区间。
#
# 【forward 的判定规则：一句话 + 一个例子】
#   规则：对每个 key，只看动作块**前一半**的帧；只要某一维在其中任意一帧越过阈值，就把这一维
#   标成 True（OR 语义，比较宽松 —— “动过一次”就算有效维）。
#   例：r1lite 的 ``left_arm``，动作块形状 (H=32, D=6)，配置里给了 6 个逐维阈值：
#         half      = 32 // 2 = 16                       # 只检查前 16 帧
#         deviation = |action[0:16] - action[0:1]|       # 与本块第 1 帧作差，形状 (16, 6)
#         flag      = (deviation >= [0.00756, ...]).any(dim=0)   # 形状 (6,)
#   为什么与第 1 帧作差而不是与 0 比：绝对位置控制的量纲里 0 往往不是机器人的静止位置，
#   而“相对块首的差”无论上层给的是绝对量还是相对量，都能正确刻画“这段里到底动了没动”。
#
#   ⚠️ “只看前一半”是本实现的一个硬编码选择（``half = action.shape[0] // 2``，
#      r1lite 即 32 → 16）。源码没有写明原因，从数据链路看，两种解释都说得通：
#        (a) 动作块后半段接近 episode 末尾时会被 clamp 到边界帧并标记 action_is_pad，
#            信息量更低、更不可靠；
#        (b) 闭环部署时通常只执行 chunk 的前一段就重新规划，前一半才代表“接下来真会执行的动作”。
#      改动这一行会直接改变 mask 的含义，若要做请先跑 tests/test_dataloader_batch.py 复核可视化。
#
#   三类 key 走三条分支（顺序很重要，见 forward 内注释）：
#     1) key 名里含 "hand"（灵巧手）     → 直接全 True，不参与“静止”判定；
#     2) key 名里含 "torso" / "chassis"  → 速度语义：|值| 本身就有意义，直接与阈值比较；
#     3) 其余（arm / eef / gripper ...） → 位置语义：与块首帧作差后再比阈值。
#
# 【阈值是怎么定的：先查逐维表，再退到类型级标量】
#   _resolve_threshold() 的优先级：
#     ① dim_thresholds[key]  → 逐维向量；长度必须等于该 key 的实际维度，否则直接报错
#     ② 类型级兜底（按 key 名里含什么子串来选）：
#          "gripper" → gripper_threshold
#          "eef" / "ee_pose" → eef_threshold（父类默认 1e-3）
#          "torso" / "chassis" → velocity_threshold
#          其余（关节等）→ joint_threshold
#     ③ 兜底值本身是 None 时按 0.0 处理（``val or 0.0``）
#   ⚠️ 阈值 0 的语义是“**这一维永远算有效**”（任何非负偏差都 ≥ 0），所以“不想要任何过滤”
#      就是给 0 或 None —— configs/data/libero.yaml 正是这么写的。
#
# 【谁在用这个类（看一眼现网配置，避免误判）】
#   · configs/data/droid.yaml  ：joint_threshold 0.003、gripper_threshold 0.01
#                                → 真的在做 idle 维过滤（droid 文档里“Idle 帧过滤”指的就是它）
#   · configs/data/libero.yaml ：joint/gripper/eef_threshold 全 0 → 等价于不过滤
#                                （保留类、关掉行为，便于以后按需打开）
#   · configs/data/r1lite.yaml ：给了 velocity_threshold 0.1 与 left_arm / right_arm 的逐维阈值
#   · 但 r1lite / r1pro / r1pro_wbc 的 task 层把 action_filter 的 _target_ 换成了
#     DummyActionFilter（task 层 model.processor 覆盖数据层），所以这些任务训练时实际生效的
#     是“全 True”，本类不会被实例化。
#   · scripts/serve_policy.py 在部署时会主动把 action_filter 换成 no-op 的 BaseActionFilter，
#     原因见下面【新手最容易踩的坑】第 5 条。
#
# 【新手最容易踩的坑】
#   1) 以为“阈值没配就等于不过滤”是对的（None → 0.0 → 恒 True），但反过来：
#      **想真正过滤就必须显式给正阈值**，否则 mask 永远全 True。
#   2) ``val or 0.0``：阈值写 0.0 与写 None 完全等价（都落到 0.0），因为 0.0 是假值。
#   3) 类型判定用的是**子串**而不是精确 key：``torso.velocities``（6 维）会命中 "torso"，
#      这是合理的（它本来就是速度量，见 configs/data/parts_meta/gripper_wbc.yaml）；
#      但反过来说，“位置型”的 ``torso``（4 维）也会走速度分支（拿绝对值比阈值），
#      语义上是有偏差的。当前 r1pro_wbc 训练走 DummyActionFilter，所以这条偏差不会触发。
#   4) 断言比较的是 ``meta["raw_shape"]`` 而不是 ``meta["shape"]``：过滤器跑在“相对动作 /
#      旋转表示”变换之前，此处维度还是原始列宽度（raw_shape）。配置里写错 raw_shape，
#      会在这里 forward 时就报错。
#   5) ``backward()`` 依赖同一 batch 里先跑过 forward：mask 必须与动作配套。
#      部署时喂进来的是全 0 的 dummy 动作（scripts/serve_policy.py 为了生成正确的
#      action_dim_is_pad 而构造的），算出来的 mask 几乎全是 False（只有 "hand" 类键例外），
#      于是 backward 会把整个预测动作清零 —— 这就是部署脚本必须把它换成 BaseActionFilter
#      的原因。
#   6) mask 为 False 的维会在**物理空间**里被写成 0（反向链已把速度和归一化都还原回来）：
#      这是一个“占位输出”，不等于“让关节去 0 位”；这些维在下游指标里本来就被排除，
#      训练时也不会进入 token 序列。要真正命令机器人“保持不动”，看的是部署客户端如何使用
#      mask / 缺席 part，而不是这里的 0 值。
#   7) ``invertible = False``（父类写死）：mask 无法从动作数值反推，所以“只有动作、没有 mask”
#      的调用会直接 assert 失败；也不要在需要可逆的变换链里指望它能被还原。
#
# 【相关文件与文档】
#   src/g05/data_processor/transforms/action_filter.py
#       BaseActionFilter / DummyActionFilter（本类的父类与“全 True”版本）
#   src/g05/data_processor/transforms/action_state_merger.py
#       action_op_mask 的对齐 / 拼接 / 切分规则
#   src/g05/data_processor/processor/base_processor.py
#       preprocess_action_state / postprocess 调用本类
#   src/g05/data_processor/processor/__init__.py
#       本文件在数据流水线里的位置一览
#   src/g05/tokenizer/interface/vq_base.py
#       _derive_noop_keys()：mask → 丢弃 noop part
#   src/g05/utils/training/train_utils.py
#       action_op_mask 传入 tokenizer.encode
#   scripts/utils/metric.py
#       noop_dim_mask：评估时排除“不动的维”
#   scripts/serve_policy.py
#       部署时把 action_filter 换成 no-op 版
#   configs/data/{droid,libero,r1lite}.yaml
#       本类的实际参数
# =============================================================================

# typing：仅用于类型标注。本文件当前的真实依赖只有 torch 与父类；
# Dict / Any / Optional / Literal / List 属于历史遗留导入（本文件的 forward / backward 都没有
# 写类型标注），保留是为了不和上游代码产生无意义的 diff —— 若你要补类型标注，可直接使用它们。
from typing import Dict, Any, Optional, Literal, List

# torch：阈值构造（torch.tensor）、逐维比较、布尔掩码，以及 backward 里的 torch.where 都靠它。
import torch

# BaseActionFilter：本类的父类，提供三样东西：
#   · __init__ 的五个阈值参数（joint / gripper / velocity / eef_threshold、dim_thresholds）
#   · set_shape_meta()：把 shape_meta["action"] 存进 self.action_meta，供下列遍历 key
#   · invertible = False 的声明（动作过滤器不可逆：mask 无法从动作数值反推）
from g05.data_processor.transforms.action_filter import BaseActionFilter


class R1LiteJointActionFilter(BaseActionFilter):
    """
    Galaxea 机器人的“逐维”动作过滤器（R1 Lite 版本，droid / libero 等配置也在复用）。

    它把每个动作维单独拿出来，各自用自己的阈值判断这一维在这段动作块里是
    “有效操作”还是“静止”，结果写进 batch["action_op_mask"] = {key: (D,) bool}。

    阈值解析优先级（先查逐维表，再退到类型级标量）：
        1. dim_thresholds[key] → 逐维向量（长度必须等于该 key 的维度）
        2. 类型级兜底（按 key 名里的子串选，取标量并广播到每一维）：
             "gripper"           → gripper_threshold
             "eef" / "ee_pose"   → eef_threshold（父类默认 1e-3）
             "torso" / "chassis" → velocity_threshold
             其余                 → joint_threshold
        兜底值为 None 时按 0.0 处理；而 0 阈值意味着“该维永远算有效”。

    控制方案假设（决定用“与块首帧作差”还是“直接取绝对值”）：
        - arm / eef / gripper：相对本块第 1 帧的偏差（绝对位置控制，用块首当参考）
        - torso / chassis    ：速度量，绝对值本身就有意义（速度控制）
    """

    # 速度语义的 key 判据（子串匹配！）：名字里含 "torso" 或 "chassis" 的 key 都按速度处理，
    # 因此 ``torso.velocities`` / ``chassis.velocities`` 这类键也能被正确识别。
    # 副作用见文件头【新手最容易踩的坑】第 3 条。
    _VELOCITY_KEYS = ("torso", "chassis")

    def forward(self, batch):
        """标记每个动作 key 的每一维是否“有效操作”，产出 action_op_mask。

        参数：
            batch: 至少包含 "action" 的样本字典，
                   "action" 是 {key: 张量 (H, D)} 的字典（H = 动作块长度，如 r1lite 的 32）。
                   若没有 "action" 就原样返回（推理时可能只给观测，不产出 mask）。

        返回：
            同一个 batch，额外带上
                batch["action_op_mask"] = {key: bool 张量 (D,)}
                    True  = 该维在这段动作里动过（有效操作维）
                    False = 静止 / 未使用
            注意形状是 **1 维 (D,)**：此处还没有 batch 维，也还没被合并器对齐 / padding，
            每个 key 的长度就是该 key 的原始维度。
        """
        # 没有动作就无事可做：直接返回，不产出 action_op_mask
        # （下游对 action_op_mask 都是“有则用、没有就跳过”的存在性判断）。
        if "action" not in batch:
            return batch

        # 逐个动作 key 计算掩码，最后整体写进 batch["action_op_mask"]。
        # 注意是**整体覆盖**：同一 batch 上重复调用 forward 会重算并覆盖旧 mask。
        action_op_mask = {}
        # self.action_meta 由父类 set_shape_meta() 在 processor.__init__ 时填好，
        # 就是 shape_meta["action"] 里 key 非空的那些条目（即本具身真正拥有的动作部件）。
        for meta in self.action_meta:
            # k：内部 key（如 left_arm）；meta_shape：该 part 的**原始**宽度（raw_shape）
            k, meta_shape = meta["key"], meta["raw_shape"]
            action = batch["action"][k]  # (H, D) —— H = 动作块长度，D = 该 part 的维度
            actual_shape = action.shape[-1]  # D：当前张量的实际维度，用来校验与解析阈值
            # 只检查动作块的**前一半**帧（r1lite：32 → 16）。动机见文件头说明：
            # 后半段更接近 episode 末尾的 clamp/padding 区，或超出闭环执行范围，参考价值更低。
            half = action.shape[0] // 2

            # 取本 key 的阈值：可能是逐维向量 (D,)，也可能是类型级标量（广播到每一维）。
            # 传 actual_shape 是为了在逐维表长度对不上时立刻报错。
            threshold = self._resolve_threshold(k, actual_shape, action.device)

            # ---- 分支 1：灵巧手（key 名里含 "hand"，如 left_hand / right_hand）----
            # 一律标记为“有效维”。原因：灵巧手维度多（见 configs/data/parts_meta/10k_wbc.yaml
            # 的 left_hand: 12 / right_hand: 12），且“有没有动”很难用单一阈值刻画，
            # 因此不参与静止判定，直接全 True（dtype/device 与动作张量对齐，便于后续拼接）。
            if "hand" in k:
                flag = torch.ones(actual_shape, dtype=torch.bool, device=action.device)
            # ---- 分支 2：速度语义 key（torso / chassis）----
            # 这类键的值本身就是速度，0 附近才是“静止”，所以直接取绝对值与阈值比较，
            # 不需要（也不能）与块首帧作差。
            elif any(vk in k for vk in self._VELOCITY_KEYS):
                # 速度语义：这类键的值本身就是速度，绝对值就有意义（不需要与块首帧作差）
                deviation = torch.abs(action[:half])  # (half, D)
                # any(dim=0)：前 half 帧里只要有一帧越阈，就认为该维“有效”
                flag = (deviation >= threshold).any(dim=0)  # (D,)
            # ---- 分支 3：其余（arm / eef / gripper，位置语义）----
            # 用“本块第 1 帧”作为参考帧，衡量每一维在后继帧里偏离了多少：
            # action[:1] 形状 (1, D) 会广播到 (half, D)，因此这里不需要额外 squeeze。
            else:
                # arm / eef / gripper：相对块首帧的偏差（绝对位置控制下的“动了多少”）
                deviation = torch.abs(action[:half] - action[:1])  # (half, D)
                flag = (deviation >= threshold).any(dim=0)  # (D,)

            # 每个 key 一条 (D,) 的 bool 向量；后续由 action_state_merger 拼成长向量
            action_op_mask[k] = flag  # (D,)
            # 维度自检：这里必须等于 **raw_shape**（而不是 shape）—— 因为过滤器跑在
            # “相对动作 / 旋转表示”等变换之前，此刻还是数据集的原始列宽度。
            # 断言放在计算之后只是代码顺序问题（先算再校验），语义上属于“输入校验”。
            assert actual_shape == meta_shape, (
                f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
            )

        batch["action_op_mask"] = action_op_mask
        return batch

    def backward(self, batch):
        """把“静止维”置 0：mask 为 False 的维度在动作张量里直接清零。

        调用时机：postprocess 反向链的**最后一步**（在 merger.backward →
        normalizer.backward → transforms 逆序之后），所以此时动作已经还原回数据集的物理量纲，
        mask 也已经被 merger.backward 从长向量切回 {key: (D,)} 的字典。

        参数：
            batch: 需要同时具备
                "action_op_mask" —— forward 产出的掩码（缺了就什么都不做，直接返回）；
                "action"          —— {key: 张量}，最后一维与掩码一一对应；
                                     既支持 (D,) 掩码对 (H, D)/(B, H, D) 动作做广播。

        返回：
            同一个 batch，其中被标记为 False 的维已被写成 0（占位输出，见文件头坑点 6）。

        注意：mask 与动作必须配套。若上游（例如部署脚本）没有先跑 forward、或喂的是
        全 0 的 dummy 动作，这里会 assert 失败 / 把整段预测清零。
        """
        # 没有 mask 就无从下手：说明上游没走过滤器（例如部署时换成了 no-op 过滤器），
        # 这种情况按“不过滤”处理，原样返回。
        if "action_op_mask" not in batch:
            return batch
        # 有 mask 但没有动作也没得改（只处理观测的推理场景），原样返回。
        if "action" in batch:
            for meta in self.action_meta:
                k = meta["key"]
                # 断言动作与掩码两侧的 key 必须齐全：缺任何一个都说明调用方没有成对使用
                # forward / backward（例如 mask 来自另一个 batch，或 merger 没把 mask 切回来）。
                assert k in batch["action"], f"Missing action key in backward filter: {k}"
                assert k in batch["action_op_mask"], (
                    f"Missing action_op_mask key in backward filter: {k}"
                )
                # torch.where：mask 为 True 的维保留原值，为 False 的维取标量 0。
                # 掩码 (D,) / (B, D) 会自动广播到动作的最后一维，因此 2 维、3 维动作都适用。
                batch["action"][k] = torch.where(batch["action_op_mask"][k], batch["action"][k], 0)
        return batch

    def _resolve_threshold(self, key: str, dim: int, device) -> torch.Tensor:
        """取出某个 key 的阈值：优先逐维向量，其次类型级标量。

        参数：
            key:    动作 part 的 key（如 left_arm / right_gripper / torso.velocities）
            dim:    该 key 的实际维度 D（用于校验逐维表长度）
            device: 目标设备，保证阈值与动作张量同设备，比较时不会触发跨设备错误

        返回：
            float32 张量。逐维表命中时形状 (D,)（逐维比较）；
            否则是 0 维标量张量（等价于广播到每一维）。

        异常：
            dim_thresholds[key] 的长度与 dim 不一致时抛 ValueError —— 这是配置写错的
            常见信号（例如把 6 维的 left_arm 阈值写成了 8 个值）。
        """
        # ① 逐维阈值表命中：配置里给该 key 单独写了一个“每维一个值”的列表。
        if key in self.dim_thresholds:
            t = torch.tensor(self.dim_thresholds[key], dtype=torch.float32, device=device)
            if t.shape[0] != dim:
                raise ValueError(f"dim_thresholds['{key}'] has {t.shape[0]} values, expected {dim}")
            return t

        # ② 类型级兜底（type-level fallback）：按 key 名里的子串判断这属于哪类部件。
        #    注意判断顺序：gripper 在最前，eef/ee_pose 其次，再速度类，最后才是普通关节。
        if "gripper" in key:
            val = self.gripper_threshold
        elif "eef" in key or "ee_pose" in key:
            val = self.eef_threshold
        elif any(vk in key for vk in self._VELOCITY_KEYS):
            val = self.velocity_threshold
        else:
            val = self.joint_threshold

        # ③ ``val or 0.0``：阈值是 None（未配置）时退回 0.0。
        #    0 阈值 ⇒ 任何非负偏差都 ≥ 0 ⇒ 该维恒为 True，也就是“不做过滤”。
        return torch.tensor(val or 0.0, dtype=torch.float32, device=device)
