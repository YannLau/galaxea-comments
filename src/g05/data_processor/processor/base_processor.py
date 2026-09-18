# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/processor/base_processor.py — 数据处理器三大基类
#                          （BaseProcessor / ActionProcessor / FullProcessor）
# =============================================================================
#
# 【这个文件是什么】
#   它定义了“数据集取出的原始样本 → 模型可直接使用的张量样本”这条流水线的骨架类。
#   一句话概括职责：把 shape_meta 声明的“维度契约”落实到真实数据上，并按固定顺序
#   执行   过滤 → 变换 → 归一化 → 合并   四个阶段的动作/状态加工，同时负责图像的
#   resize / 张量化，以及指令文本的采样与中英文选择。
#
#   文件里一共三个类，构成一条继承链（本文件是整条链的公共基座）：
#     · BaseProcessor   —— 抽象骨架：解析并校验 shape_meta、持有 normalizer、
#                          维护 train()/eval() 模式、按 camera_size_config 解析图像尺寸
#                          并自动注入 T.Resize；提供 action_state_transform() 与
#                          augment_instruction() 两个通用能力。
#     · ActionProcessor —— 只做动作/状态，不碰图像与文本（动作 tokenizer 训练走这条）。
#     · FullProcessor   —— 在 ActionProcessor 之上增加多分辨率图像处理与（tokenizer 相关的）
#                          元信息（pad_token_id / image_token_index / max_text_tokens），
#                          VLM 训练与推理走这条；galaxea_cot_processor.py 再继承它。
#
# 【它在整条数据链路中的位置】
#
#   configs/data/*.yaml + configs/model/g05.yaml 里的 processor: 段
#        │  Hydra 按 _target_ 反射实例化
#        │  入口：src/g05/utils/data/processor_utils.py 的 build_processors()
#        │    · cfg.data.processors 不存在 → 直接 instantiate(cfg.model.processor)
#        │    · 存在（多 embodiment 混合）→ 每个具身各建一个，再包进 MixtureProcessor
#        ▼
#   BaseProcessor / ActionProcessor / FullProcessor      ← 本文件
#        │  被数据集与推理器调用：
#        │    · 训练：src/g05/data/base_lerobot_dataset.py 的 __getitem__ 末尾执行
#        │            sample = self.processor.preprocess(sample)
#        │            （放在 __getitem__ 而不是 collate 阶段，是为了让 DataLoader
#        │             的多个 worker 并行分摊这部分开销）
#        │    · 推理：src/g05/models/g05/inferencer.py
#        │            preprocess(obs) → 模型前向 → postprocess(item_batch)["action"]
#        │    · 离线评估：scripts/eval_open_loop.py 直接用 postprocess() 做反归一化
#        ▼
#   DataLoader 拼批 → 模型侧 g05.models.g05.io.input_preprocessor.InputPreprocessor
#        （token 化属于模型层，不在本文件完成；本文件只负责产出数值张量）
#
# 【先记住这 4 个概念】
#
#   (1) shape_meta —— 写在 config 里的“维度契约”，每个 part 一条记录：
#         key          内部 key（训练侧用的名字，如 left_arm / head_rgb）
#         raw_shape    变换前的维度（从 parquet 原始列切出来时的宽度）
#         shape        变换后的维度（action_state_transforms 跑完之后的目标宽度）
#         camera_type  仅图像需要：head / wrist_left / wrist_right ...
#                      用来查 camera_size_config 得到 [H, W]
#       processor 在 __init__ 里逐条校验 key 非空、且不含历史遗留的“语义化字段”
#       （见 _FORBIDDEN_META_KEYS），避免新旧切分语义混用导致静默取错维度。
#
#   (2) 四个加工阶段（顺序固定，改动会直接改变归一化统计量的含义）：
#         action_filter.forward            按维标记/过滤无效动作（产出 action_op_mask）
#           → action_state_transform       相对动作、旋转表示等可逆变换（transforms/ 里的算子）
#           → normalizer.forward           用数据集统计量归一化
#           → action_state_merger.forward  把多个 key 合并成一条固定维度向量并补 padding
#       postprocess() 按相反顺序还原：
#         merger.backward → normalizer.backward → transforms 逆序 → action_filter.backward
#
#   (3) train() / eval() —— 决定“这次用哪套图像变换”。
#       self._is_train 未设置时读取 processor.is_train 会直接抛 ValueError，这是刻意设计：
#       训练与验证的图像增强不同（train_transforms vs val_transforms），必须显式声明模式，
#       否则极易在验证时误用训练增强、或反之。
#
#   (4) camera_size_config —— 相机分辨率的“唯一真源”，形如 {camera_type: [H, W]}。
#       __init__ 会据此改写 shape_meta 里每个图像条目的 shape=[C, H, W]，并把 T.Resize
#       插到每个相机变换列表的最前面（见 _resolve_image_shapes / _patch_resize_per_camera）。
#       因此在 embodiment 配置里不要再手写 T.Resize，否则会与自动注入的缩放叠加。
#
# 【一个样本在 preprocess() 里会发生什么】
#   BaseProcessor.preprocess（基类版本：所有相机拼成一个大张量）
#     1. 逐相机做变换 → torch.cat 成 pixel_values [N, C, H, W]，不足 num_output_cameras 则补零；
#     2. gt_action = 合并后的“原始动作”深拷贝（归一化之前，供开环评估做对照）；
#     3. action_filter → action_state_transform → normalizer → action_state_merger；
#     4. 输出 action / action_is_pad / action_dim_is_pad / proprio / proprio_is_pad ...
#   ActionProcessor.preprocess（无图像）
#     流程相同，只是不做图像；gt_action 取自“变换后、归一化前”的动作（供 tokenizer 评估）。
#   FullProcessor.preprocess（支持多分辨率）
#     图像按 camera_type 分别保留为 dict（不做 torch.cat），其余与 ActionProcessor 相同。
#
# 【输出契约（preprocess 的返回值，节选）】
#     {
#       "pixel_values":       Tensor[N, C, H, W]  或  Dict[camera_type, Tensor[T, C, H, W]]
#                                                   # N = 相机数 × 观测帧数
#       "proprio":            Tensor[T, D_state],      # 合并后的 state（注意名字是 proprio）
#       "proprio_is_pad":     Tensor[T]  (bool),       # 时间维 padding 掩码
#       "proprio_dim_is_pad": Tensor[D_state] (bool),  # 维度维 padding 掩码
#       "action":             Optional Tensor[H, D_action],
#       "action_is_pad":      Optional Tensor[H]  (bool),
#       "action_dim_is_pad":  Optional Tensor[D_action] (bool),
#       "action_op_mask":     Optional，过滤器产出，标记哪些动作维是“有效操作维”
#       "gt_action":          Optional，归一化之前的动作副本（开环评估对照）
#       "idx" / "task" / "embodiment" / "dataset_locator": 透传的元信息
#     }
#   注意两点：字段名是 proprio（不是 state）、proprio_is_pad（不是 state_is_pad）。
#   postprocess() 反向还原时：双路（AR/FM）分支按 "state" 优先、"proprio" 兜底取值；
#   旧版单路分支则会把 proprio 改名为 state 后再走反向流程。
#
# 【postprocess() 的三条路径（推理 / 评估侧）】
#     · action_fm：Flow-Matching 路径输出的动作（z-score 等归一化空间）→ 需要反归一化
#     · action_ar：自回归路径输出的动作（已是物理量 / dummy 空间）→ 同样走完整反向流程
#     · action   ：兼容旧代码的键，按 action_fm 优先、否则 action_ar 的顺序取值
#   两条路径都执行：merger.backward → normalizer.backward → transforms 逆序 →
#   action_filter.backward，最后用 [:, num_obs_steps-1:, :] 把动作对齐到“最后一个观测步”。
#   当 action_fm / action_ar 都不存在时，退回旧版单路逻辑（要求 data 中已有
#   "action" 与 "proprio"，或调用方按约定自行提供）。
#
# 【新手最容易踩的坑】
#   1) 忘记调用 train()/eval()：直接读 processor.is_train 会抛 ValueError。
#   2) 忘记 set_normalizer_from_stats()：直接读 processor.normalizer 会抛 ValueError
#      （MixtureProcessor 会按 embodiment 逐个转发统计量）。
#   3) 同时在 embodiment yaml 里写 shape 与 camera_size_config 且两者不一致：
#      会打 warning，并以 camera_size_config 为准（yaml 里的 shape 成为“陈旧信息”）。
#   4) 在 train/val transforms 里再手写一次 T.Resize：会与自动注入的 Resize 叠加。
#   5) 以为 raw_shape 和 shape 一定相同：做相对动作 / 旋转表示的变换会改变维度，
#      此时 action_state_transform() 会先按 raw_shape 断言输入、再按 shape 断言输出。
#   6) B1K 具身（BehaviorPerKeyTransform）会合并/删除 key：这类变换置
#      alters_key_structure=True，此后“变换后断言”会跳过已被合并掉的 key。
#   7) 归一化后出现 NaN/Inf：_warn_nan_in_batch() 会在变换后与归一化后各扫一遍并打
#      warning（常见来源是四元数/欧拉角变换在退化姿态下产生除零）。
#   8) 不要在 g05/data_processor/__init__.py 里 import 本文件：会触发循环导入
#      （normalizer → g05.data_processor 之间互相依赖）。
#
# 【相关文件与文档】
#   src/g05/data_processor/transforms/base.py                   BaseActionStateTransform（forward/backward 契约）
#   src/g05/data_processor/transforms/action_filter.py          BaseActionFilter（本文件的 action_filter 基类）
#   src/g05/data_processor/transforms/action_state_merger.py    各合并器实现（ConcatLeftAlign / GroupedPaddingMerger ...）
#   src/g05/utils/data/normalizer.py                            LinearNormalizer 与 NormMode 定义
#   src/g05/utils/data/processor_utils.py                       build_processors()：配置 → 实例
#   src/g05/data_processor/processor/mixture_processor.py       多 embodiment 分发（MixtureProcessor）
#   src/g05/data_processor/processor/galaxea_cot_processor.py   当前配置默认使用的子类（两阶段 preprocess）
#   src/g05/data/base_lerobot_dataset.py                        preprocess 的调用方（__getitem__ 末尾）
#   src/g05/models/g05/inferencer.py                            preprocess / postprocess 的推理侧调用方
#   configs/model/g05.yaml                                      processor: 段完整参数示例
#   docs/data/schema_zh.md                                      shape_meta 字段语义
#   docs/architecture/g05_io_zh.md                              Dataset → Collate → 模型的张量形状
# =============================================================================

# ABC / abstractmethod：ABC 让 BaseProcessor 成为抽象基类；
# abstractmethod 在本文件中没有被直接使用（子类实现的是约定方法），保留导入以便后续扩展。
from abc import ABC, abstractmethod
# typing：仅用于类型标注。Literal 目前在本文件中未使用，保留以便后续扩展。
from typing import Dict, Any, Optional, List, Literal, Tuple
# logging：模块级 logger，用于 shape_meta 校验提示与 NaN/Inf 告警。
import logging

# OmegaConf：构造时把 Hydra 传入的 DictConfig 转成普通 dict，避免 struct 模式挡住写操作。
from omegaconf import OmegaConf
import torch
import numpy as np
# deepcopy：给 gt_action 留一份“原始动作”的深拷贝，避免后续原地变换污染对照数据。
from copy import deepcopy
# LinearNormalizer：按 shape_meta + 数据集统计量做逐 part 归一化；NormMode 是归一化模式联合类型。
from g05.utils.data.normalizer import LinearNormalizer, NormMode
# dict_apply：对嵌套 dict 里的每个张量施加同一个函数（postprocess 里按时间切片就靠它）。
from g05.utils.common.pytorch_utils import dict_apply

# BaseActionFilter：动作过滤器基类；processor 在流水线第一步调用它的 forward。
from g05.data_processor.transforms.action_filter import BaseActionFilter

# 本模块 logger：命名与包路径一致，便于在训练日志里定位到本文件。
logger = logging.getLogger(__name__)


def _warn_nan_in_batch(data: Dict[str, Any], stage: str) -> bool:
    """
    扫描样本里的 action / state 字典，发现 NaN 或 Inf 就打一条 warning。

    这是“体检”而不是“治疗”：函数不修改数据（normalizer 内部有自己的 nan_to_num 逻辑），
    只在关键节点上暴露数据问题，方便定位是哪个具身、哪个数据集、第几个样本出的错。

    参数：
        data:  样本字典，期望其中 "action" / "state" 的值是 {key: Tensor}
        stage: 调用阶段标签（如 "post_transform" / "post_norm"），会原样打进日志

    返回：
        bool，只要发现过任意一个 NaN/Inf 就返回 True（全部干净则返回 False）。
    """
    found = False
    # 这三个字段是“定位信息”：分别来自具身类型、数据集路径与样本下标。
    # 取不到时用占位符，保证日志始终能打出完整一行。
    embodiment = data.get("embodiment", "unknown")
    dataset_locator = data.get("dataset_locator", "unknown")
    idx = data.get("idx", "?")
    # 只检查 action / state 两支；其余键（task、images 等）不在此函数职责内。
    for split in ("action", "state"):
        # 缺少该分支，或该分支不是 {key: Tensor} 的字典结构（例如已经被合并成单个张量），就直接跳过。
        if split not in data or not isinstance(data[split], dict):
            continue
        for key, val in data[split].items():
            # 非张量（如字符串、None）无法做 NaN 判断，跳过。
            if not isinstance(val, torch.Tensor):
                continue
            # 分别统计 NaN 与 ±Inf 的个数；int() 把张量标量转成 Python 整数。
            n_nan = int(torch.isnan(val).sum())
            n_inf = int(torch.isinf(val).sum())
            if n_nan > 0 or n_inf > 0:
                # 一条 warning 里带齐“阶段 / 分支 / key / 数量 / 定位信息”，
                # 便于在海量日志里快速筛出问题样本。
                logger.warning(
                    "[NaN/Inf] stage=%s split=%s key=%s nan=%d inf=%d | "
                    "embodiment=%s dataset=%s idx=%s",
                    stage,
                    split,
                    key,
                    n_nan,
                    n_inf,
                    embodiment,
                    dataset_locator,
                    idx,
                )
                found = True
    return found


# 语言模型损失里代表“该位置不计算 loss”的忽略索引（PyTorch CrossEntropyLoss 的默认 ignore_index）。
# 本文件只定义这个常量，真正使用它的是模型侧
# src/g05/models/g05/io/input_preprocessor.py（它从这里 import，作为 padding / 条件 token 的标签值）。
IGNORE_INDEX = -100

# 旧版配置里出现过的“语义化字段”黑名单。
# 现在的约定是：所有切分都必须用 start_index + raw_shape 显式描述，不允许再写
# source / target_key / resolved_lerobot_key 这类“隐式推断”字段——一旦出现就立刻报错，
# 避免新旧语义混用导致静默取错维度的数据。
# 与 src/g05/data/base_lerobot_dataset.py 中的同名集合保持一致。
_FORBIDDEN_META_KEYS = {
    "source",
    "target_key",
    "target_offset",
    "target_from",
    "semantic_key",
    "resolved_lerobot_key",
    "resolved_start_index",
}


class BaseProcessor(ABC):
    """
    数据处理器抽象基类：负责“原始样本 → 张量样本”的公共骨架。

    它给出了最基础的 preprocess() / postprocess() 实现：
      · preprocess():  所有相机 torch.cat 成一个大张量 + 完整的动作/状态流水线；
      · postprocess(): 单路（只有 "action"）的反向还原。
    子类在此之上分叉：
      · ActionProcessor：去掉图像，并把流水线拆成 preprocess_action_state() /
        build_action_sample() 两个可复用步骤，postprocess() 增加 AR/FM 双路支持；
      · FullProcessor：图像改为按 camera_type 分别保留的字典（支持多分辨率）；
      · GalaxeaCoTProcessor（同目录另一个文件）：在 FullProcessor 之上再叠加
        SamplesBuilder 的 CoT 模板构造。

    此外，所有子类都要用的公共设施都准备在这里：
      · 解析、校验并持有 shape_meta（维度契约）；
      · 解析图像尺寸，并为每个相机列表自动注入 T.Resize；
      · 持有 normalizer（由 set_normalizer_from_stats() 用数据集统计量创建）；
      · 维护 train()/eval() 模式开关；
      · 提供 action_state_transform()（带输入/输出维度断言的可逆变换执行器）；
      · 提供 augment_instruction()（指令采样与中英文选择）。

    生命周期约定（顺序不能颠倒）：
        1. 由 Hydra 按配置实例化（__init__ 会做全部静态校验）；
        2. 调用 train() 或 eval() 声明本次是训练还是评估；
        3. 调用 set_normalizer_from_stats(dataset_stats) 注入归一化统计量；
        4. 之后才能在 __getitem__ / inferencer 中调用 preprocess() / postprocess()。

    补充说明：本类虽然继承 ABC，但当前没有标注 @abstractmethod 的方法
    （abstractmethod 只是为将来扩展保留的导入），所以它其实可以被直接实例化；
    实际项目里请按需求选择 ActionProcessor / FullProcessor / GalaxeaCoTProcessor。
    """

    def __init__(
        self,
        # ---------------- 维度契约与观测设置 ----------------
        # shape_meta：核心维度声明，含 action / state / images 三张表，每张表是“每条 part 一个 dict”的列表。
        #             本类会原地校验并改写其中的图像条目（见 _resolve_image_shapes）。
        shape_meta: Dict[str, Any],
        # num_obs_steps：一次送进模型的观测帧数 T。1 = 单帧；>1 = 多帧历史（MEM 设计）。
        #                必须与数据侧 obs_size、模型侧 cond_steps 保持一致。
        num_obs_steps: int,
        # num_output_cameras：输出张量预留的“相机帧”槽位数 = 相机数 × 观测帧数。
        #                     实际数量少于它时补零张量，多于它时不做处理（由子类/上层保证匹配）。
        num_output_cameras: int,
        # action_state_transforms：作用于 action/state 的可逆变换列表（相对动作、旋转表示等）。
        #                          允许为 None 表示不做任何变换；顺序会被 forward/backward 双向使用。
        action_state_transforms: Optional[List[Any]],
        # ---------------- action & state 归一化 ----------------
        # 是否按“逐步（stepwise）”使用归一化统计量：动作 chunk 内各时间步分布不同，逐步统计更精确。
        # 需要数据集统计量文件里存在 stepwise_* 字段与之配套。
        use_stepwise_action_norm: bool,
        # 默认归一化模式（dummy / min/max / q01/q99 / z-score / tanh ...，见 normalizer.NormMode）。
        norm_default_mode: NormMode,
        # 动作/状态合并器：把 {key: Tensor} 合并成一条固定维度向量，并负责反向拆分。
        # 实例由配置提供（如 GroupedPaddingMerger / ConcatLeftAlign），本类只负责调用与转发 shape_meta。
        action_state_merger,
        # 动作过滤器：按维标记/过滤无效动作（产出 action_op_mask），流水线的第一步。
        action_filter: BaseActionFilter,
        # ---------------- 图像变换 ----------------
        # 训练 / 验证两套变换列表：{相机 key: [transform, ...]}。
        # 由 train() / eval() 决定用哪一套；未列出的相机 key 会在下面被补成空列表。
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None,
        # ---------------- 指令（语言）处理 ----------------
        # 丢弃“高层指令”的概率：1.0 = 只给 [Low]；0.0 = 总是给 [High] + [Low]（用于训练鲁棒性）。
        drop_high_level_prob: float,
        # 指令语言开关：Galaxea 数据的 task 字段形如 "中文@English"，True 取中文，False 取英文。
        use_zh_instruction: bool,
        # ---------------- 相机分辨率配置 ----------------
        # {camera_type: [H, W]}，相机分辨率的“唯一真源”：用于解析 shape_meta 中图像的
        # shape=[C,H,W]，并为每个相机的变换列表注入 T.Resize。
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        # ---------------- 归一化的可选覆盖项 ----------------
        # 逐 key 覆盖归一化模式：{"action": {key: mode}, "state": {key: mode}}；
        # None（或某个 key 缺失）表示该 key 使用 norm_default_mode。
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # 给 VLM 输入用的“第二条归一化通道”，与训练动作归一化相互独立：
        # 用于把动作填进文本模板时使用另一套缩放，避免模板里的数值与训练目标互相干扰。
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # dummy 归一化模式的裁剪范围（仅当 norm_mode == "dummy" 时生效）：
        # dummy 模式不改变数值，只做 clamp，用于“离散动作 tokenizer 自己处理数值”的场景。
        dummy_clip_default: Tuple[float, float] = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        # 尾部压缩系数：仅当 norm_mode 为 "z-score-tail" / "q01/q99-tail" 时生效，
        # 用来抑制长尾离群点（越大压缩越强，默认 0.075）。
        norm_tail_scale: float = 0.075,
    ):
        """构造处理器：完成全部静态校验与组件装配。

        参数含义按分组标注在签名与下方正文注释中，这里只说三件关键副作用：
          1) shape_meta 会被校验并原地改写（图像条目的 shape、每个相机的变换列表）；
          2) action_state_merger / action_filter 在此刻拿到 shape_meta；
          3) 归一化器仍为 None —— 训练前必须调用 set_normalizer_from_stats() 注入统计量。
        """
        # 把 OmegaConf 容器转成普通 Python dict/list，这样后续的原地改写
        # （例如 _resolve_image_shapes 写回 img_meta["shape"]）才能无条件生效：
        # OmegaConf 默认的 struct 模式会拦截“写新键”，而且它的容器不是真正的 dict/list。
        # 注意：resolve=True 会把配置里的插值 ${...} 一并求值，避免残留未解析对象。
        from omegaconf import OmegaConf, DictConfig

        if isinstance(shape_meta, DictConfig):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)

        # shape_meta 必须是“完全显式”的：这里逐条校验 action / state 的 key 合法，
        # 并拒绝历史遗留的语义化字段。此处的处理是“校验后原样保留”，
        # 不会静默丢弃占位条目——因为丢弃会让下游维度对不上，问题被推迟到更难查的地方才暴露。
        processed_action_meta, processed_state_meta = [], []
        for meta in shape_meta["action"]:
            key = meta.get("key")
            # key 必须是非空字符串（None、空串、纯空白都视为配置错误）。
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"action meta key must be a non-empty string, got {key!r}.")
            # 命中黑名单字段就直接报错，提示用户改写为 start_index + raw_shape 的显式写法。
            unexpected = [field for field in _FORBIDDEN_META_KEYS if field in meta]
            if unexpected:
                raise ValueError(
                    f"action meta for key={key!r} contains forbidden fields: {unexpected}."
                )
            processed_action_meta.append(meta)
        # state 表做同样的校验（注意与 action 的报错文案区分，便于定位）。
        for meta in shape_meta["state"]:
            key = meta.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"state meta key must be a non-empty string, got {key!r}.")
            unexpected = [field for field in _FORBIDDEN_META_KEYS if field in meta]
            if unexpected:
                raise ValueError(
                    f"state meta for key={key!r} contains forbidden fields: {unexpected}."
                )
            processed_state_meta.append(meta)
        shape_meta["action"] = processed_action_meta
        shape_meta["state"] = processed_state_meta

        # 保存三份最基础的配置：维度契约、观测帧数、相机槽位数。
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras

        # 指令相关配置：一个控制“是否给高层指令”的采样概率，一个控制中英文选择。
        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        # 图像变换：先原样存下 train/val 两套列表，随后会做“补键 + 注入 Resize”两步加工。
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        # 根据 camera_size_config 解析每张图像的 [C, H, W]（会原地改写 shape_meta 中的 shape 字段），
        # 并在下面把对应的 T.Resize 注入到每个相机的变换列表开头。
        self._resolve_image_shapes(self.shape_meta, camera_size_config)
        # 建立 {相机 key: [H, W]} 映射，供后续补键与 Resize 注入使用。
        # 只收录已经有 shape 的图像条目；shape[1:] 即 (H, W)，shape[0] 是通道数 C。
        camera_key_to_size = {
            img_meta["key"]: img_meta["shape"][1:]
            for img_meta in (self.shape_meta.get("images") or [])
            if img_meta.get("shape") is not None
        }
        # 为 train/val 变换里“缺失的相机 key”补一个空列表：
        # 这样数据集侧补零注入的图像槽位（例如 DROID 的 dummy_wrist_right）也能走通，
        # 而不必强迫每个 embodiment 的 yaml 为这些占位相机重复写一份变换栈。
        self.train_transforms = self._ensure_transforms_for_keys(
            self.train_transforms, camera_key_to_size.keys()
        )
        self.val_transforms = self._ensure_transforms_for_keys(
            self.val_transforms, camera_key_to_size.keys()
        )
        # 把 T.Resize([H, W]) 插到每个相机变换列表的最前面，保证所有变换都在统一分辨率下进行。
        self.train_transforms = self._patch_resize_per_camera(
            self.train_transforms, camera_key_to_size
        )
        self.val_transforms = self._patch_resize_per_camera(self.val_transforms, camera_key_to_size)

        # 训练/评估模式开关：三态（None = 未声明，True = 训练，False = 评估）。
        # 保持三态而不是默认 False，是为了让“忘记设置模式”变成显式报错而不是静默走错分支。
        self._is_train = None

        # 可逆变换列表（相对动作、旋转表示等）。可以为 None，表示不加任何变换。
        self.action_state_transforms = action_state_transforms
        # 标记“本组变换是否会改变 key 结构”（会合并/删除 key）。
        # 目前只有 B1K 具身使用的 BehaviorPerKeyTransform 属于这种情况：
        #   · B1K 的 proprio 字段在原始列里并不连续——trunk_qpos[236:240] 与
        #     base_qvel[253:256] 必须作为两个独立的 shape_meta key 读出来，
        #     再由变换合并成 lower_body(7D) = trunk_qpos(4) + base_qvel(3)；
        #   · 夹爪同理：proprio 里存的是 2D 双指 qpos，需要压平成 1D 宽度。
        # 该标记为 True 时，“变换后维度断言”会跳过已经被合并消耗掉的 key；
        # 其它具身（r1lite 等）不改变 key 结构，标记保持 False 以保留严格断言。
        self._transforms_alter_keys = any(
            getattr(t, "alters_key_structure", False) for t in (action_state_transforms or [])
        )
        # 合并器与过滤器都是“配置注入的算子”，需要拿到 shape_meta 才知道各 part 的维度与顺序；
        # 这里在构造期一次性注入，避免每次 forward 重复传递。
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.action_filter = action_filter
        self.action_filter.set_shape_meta(self.shape_meta)

        # 归一化相关配置先存下来，真正的 LinearNormalizer 要等 set_normalizer_from_stats()
        # 拿到数据集统计量之后才创建（因此这里保持为 None）。
        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer = None

        # VLM 输入动作归一化通道：默认不启用（default_mode 为 None 时不创建第二个归一化器）。
        self.vlm_input_action_norm_default_mode = vlm_input_action_norm_default_mode
        self.vlm_input_action_norm_exception_mode = vlm_input_action_norm_exception_mode
        self._vlm_input_action_normalizer = None

        # 供 set_normalizer_from_stats() 透传给 LinearNormalizer 的 dummy 裁剪范围与尾部压缩系数。
        self.dummy_clip_default = dummy_clip_default
        self.dummy_clip_exception = dummy_clip_exception
        self.norm_tail_scale = norm_tail_scale

    @property
    def is_train(self):
        """当前是否处于训练模式（由 train() / eval() 设置）。

        故意做成“未设置就抛错”的属性：训练与验证使用不同的图像变换栈，
        如果默默给一个默认值，很容易在验证时误用训练增强（或反之）而难以察觉。
        """
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        """归一化器（惰性创建）。

        必须先调用 set_normalizer_from_stats() 把数据集统计量灌进来，
        否则这里直接抛 ValueError —— 因为“没有统计量的归一化器”在语义上没有意义。
        """
        if self._normalizer is None:
            raise ValueError(
                "normalizer has not been set. Please call set_normalizer_from_stats() first."
            )
        return self._normalizer

    def train(self):
        """切换到训练模式，并返回 self 以便链式调用（如 ds.set_processor(p.train())）。"""
        self._is_train = True
        return self

    def eval(self):
        """切换到评估模式，并返回 self（链式写法同上）。"""
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        """用数据集统计量构造归一化器（训练开始前必须调用一次）。

        参数：
            dataset_stats: 单个 embodiment 的统计量字典，形如
                {"action": {key: {...}}, "state": {key: {...}}}；
                由 MixtureProcessor 按 embodiment 逐个转发进来。
                允许为 None（此时归一化器按各 mode 的定义退化处理）。

        副作用：
            · 创建 self._normalizer；
            · 若配置了 vlm_input_action_norm_default_mode，再额外创建一个独立的
              self._vlm_input_action_normalizer（给 VLM 文本模板里的动作数值使用）。
        """
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
            dummy_clip_default=self.dummy_clip_default,
            dummy_clip_exception=self.dummy_clip_exception,
            # 统计量里缺少某个 key 时只打 warning 不报错，便于混合数据集逐步补齐统计量。
            missing_key_mode="warn",
            tail_scale=self.norm_tail_scale,
        )

        # 第二条归一化通道（VLM 输入专用）：只有显式配置了默认模式才创建。
        if self.vlm_input_action_norm_default_mode is not None:
            self._vlm_input_action_normalizer = LinearNormalizer(
                use_stepwise_action_norm=self.use_stepwise_action_norm,
                shape_meta=self.shape_meta,
                default_mode=self.vlm_input_action_norm_default_mode,
                # 逐 key 覆盖表的回退顺序：优先用 VLM 专属的 exception_mode；
                # 没配就退回训练归一化的 exception_mode，保持逐 key 覆盖行为一致。
                exception_mode=self.vlm_input_action_norm_exception_mode
                if self.vlm_input_action_norm_exception_mode is not None
                else self.norm_exception_mode,
                stats=dataset_stats,
                dummy_clip_default=self.dummy_clip_default,
                dummy_clip_exception=self.dummy_clip_exception,
                missing_key_mode="warn",
                tail_scale=self.norm_tail_scale,
            )

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """
        从样本字段中采样出本次训练用的指令文本（基类实现）。

        规则（与 Galaxea 数据的字段约定绑定）：
          · "coarse_task"（若存在）作为高层指令，"task" 作为低层指令；
          · "task" 形如 "中文@English" 时，按 self.use_zh_instruction 选其中一边；
          · 以 drop_high_level_prob 的概率丢弃高层指令，只留 "[Low]: ..."；
            否则输出 "[High]: ..., [Low]: ..." 两段式指令。

        参数：
            data: 数据集 __getitem__ 产出的原始样本（LeRobot / mcap 语义），
                  其中可能含 "task" / "coarse_task" 等文本字段。

        返回：
            str，拼好的指令文本。提示：类型标注沿用了历史写法 List[str]，
            但基类实现实际返回单个字符串；子类 GalaxeaCoTProcessor 会覆盖本方法
            （改用 hardcode_instruction 或直接返回 data["task"]），使用方应以实际返回值为准。
        """
        # 高层指令（coarse_task）可选：缺失时用空串占位，保证后续拼接格式稳定。
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        # 连低层指令（task）都没有时，只能返回高层指令。
        # 注意这里用的是小写 "[high]"，与下面的两段式 "[High]" 拼写不同（历史行为），
        # 下游若按字符串匹配前缀，需要把这两种写法都考虑进去。
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea 的 LeRobot 数据用 "@" 分隔中英文指令：前半段中文，后半段英文。
        # 按 use_zh_instruction 选择要使用的语言；这里保持与原实现一致的 split 行为。
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        # 按概率决定是否丢掉高层指令；两种格式都保留显式层级前缀，便于模型学会区分。
        if np.random.rand() < self.drop_high_level_prob:
            instruction = f"[Low]: {low_level_instruction}"
        else:
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"

        return instruction

    def action_state_transform(self, batch):
        """执行 action/state 的可逆变换，并在变换前后各做一次维度断言。

        为什么要“前后各断言一次”：
          · 变换前按 raw_shape 断言 → 尽早发现“数据集切出来的宽度”与 shape_meta 不一致；
          · 变换后按 shape 断言     → 尽早发现某个变换算子把维度改错了。
        这样出错时能直接定位到是“数据源”还是“变换算子”，
        而不是等到归一化 / 拼批阶段才以更难排查的形式崩溃。

        参数：
            batch: 样本字典，其中 "action" / "state" 形如 {key: Tensor[..., D]}。

        返回：
            变换后的 batch（算子既有原地修改风格也有返回新对象风格，这里统一接返回值）。
        """
        # ---- 变换前：按 raw_shape 校验输入宽度 ----
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                # 取最后一维作为“该 part 的宽度”；前面的时间/帧维度不在本断言范围内。
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, (
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
                )

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, (
                    f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape{meta_shape}."
                )

        # ---- 依次执行变换（顺序即配置里的书写顺序；反向时按逆序回放）----
        if self.action_state_transforms is not None:
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)

        # ---- 变换后：按 shape 校验输出宽度 ----
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                # B1K 等会合并 key 的变换：该 key 已被合并消耗掉，跳过它的形状断言。
                if self._transforms_alter_keys and k not in batch["action"]:
                    continue
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, (
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."
                )

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            if self._transforms_alter_keys and k not in batch["state"]:
                continue
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, (
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."
            )

        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        把数据集样本加工成“可直接 collate 的数值张量样本”（基类实现，含图像）。

        处理顺序（不可随意调整，顺序变了归一化统计量的含义也会变）：
            1) 图像：逐相机做变换 → 所有相机沿“帧/相机”维拼接成一个大张量；
            2) gt_action：在归一化之前留一份“合并后的原始动作”深拷贝，供开环评估对照；
            3) action/state：action_filter → action_state_transform → normalizer → merger；
            4) 组装输出字段（action / proprio 及其掩码）。

        参数：
            data: 由数据集 __getitem__ 产出的原始样本（LeRobot / mcap 语义）：
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state":  Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad":  torch.Tensor -> [num_obs_steps,]
                - "image_is_pad":  torch.Tensor -> [num_obs_steps,]
                - "idx": int，样本下标

        返回：
            sample: 可直接 collate 的样本字典（节选）：
                - "pixel_values":     torch.Tensor -> [num_input_cameras, C, H, W]
                - "proprio":          torch.Tensor -> [num_obs_steps, proprio_dim]
                - "proprio_is_pad":   torch.Tensor -> [num_obs_steps,]
                - "proprio_dim_is_pad": torch.Tensor -> [proprio_dim,]
                - "action":           Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad":    Optional, torch.Tensor -> [action_horizon,]
                - "action_dim_is_pad":Optional, torch.Tensor -> [action_dim,]
                - "gt_action":        Optional，输入动作的深拷贝，供开环评估对照（不被归一化改写）
                - "idx": int，样本下标
            说明：旧版 docstring 里写的 "input_ids" / "attention_mask" / "state_is_pad"
            并不由本方法产出（token 化在模型侧；实际的 padding 字段名是 proprio_is_pad）。
        """
        sample = {}

        # ---------------- 步骤 1：图像 ----------------
        # 逐相机处理，处理完的每张张量形状为 [num_obs_steps, C, H, W]，暂存到列表里。
        processed_images = []
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]
            # 统一约定：进入处理器的图像必须是 4 维（帧、通道、高、宽）。
            # 维度不对通常是数据集侧取帧/拼接出了问题，这里尽早报错。
            assert image.ndim == 4, (
                f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"
            )

            # 在“已合并多帧的批”上做变换，避免逐帧调用的开销。
            # 用哪套变换由 train() / eval() 决定（训练用 train_transforms，评估用 val_transforms）。
            transforms = self.train_transforms if self.is_train else self.val_transforms
            for trans in transforms[key]:
                image = trans(image)

            # 变换后必须恰好是 [num_obs_steps, C, H, W]，其中 [C, H, W] 来自 shape_meta（已由 camera_size_config 解析）。
            meta_shape = tuple([self.num_obs_steps] + shape)
            assert image.shape == meta_shape, (
                f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"
            )

            processed_images.append(image)

        # 沿第 0 维拼接所有相机：结果前 num_obs_steps 个槽位是第 1 个相机的 T 帧，
        # 接着是第 2 个相机的 T 帧……因此总长度是“相机数 × 观测帧数”。
        pixel_values = torch.cat(processed_images, dim=0)  # [num_input_cameras, C, H, W]
        # 相机帧数不足 num_output_cameras 时补零张量（占位，模型侧会配合掩码忽略它们）。
        # 注意：这里只在“少于”时补零；多于时原样返回，由子类 / 上层保证不会发生。
        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros(
                (self.num_output_cameras,) + pixel_values.shape[1:],
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            out[0 : pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        else:
            sample["pixel_values"] = pixel_values

        # ---------------- 步骤 2：原始动作对照（开环评估用）----------------
        # 在 action_filter / 变换 / 归一化之前，用 deepcopy 另存一份“合并后的原始动作”。
        # 关键点：这里的 merger.forward 只做“拼接 + padding”，不做归一化，
        # 因此 gt_action 与数据集里记录的物理动作量纲一致，可直接与预测动作对比。
        # deepcopy 是必需的：后面的变换可能原地改写 data，不能污染这份对照数据。
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # ---------------- 步骤 3：动作 / 状态流水线 ----------------
        # 顺序固定：过滤 → 变换 → 归一化 → 合并。反向（postprocess）按逆序执行。
        data = self.action_filter.forward(data)
        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        # ---------------- 步骤 4：组装输出字段 ----------------
        # 动作是可选的：推理时可能只给观测；训练时由数据集保证存在。
        if "action" in data:
            sample["action"] = data["action"]  # [action_horizon, action_dim]
            # action_is_pad：时间维掩码（越界 / 子任务切换的步不算 loss）
            sample["action_is_pad"] = data["action_is_pad"]  # [action_horizon,]
            # action_dim_is_pad：维度维掩码（合并器为对齐而补出来的维度不算 loss）
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]  # [action_dim,]
            # action_op_mask：哪些动作维属于“有效操作维”（由 action_filter 产出），
            # 不是所有过滤器都会写这个字段，因此按存在性透传。
            if "action_op_mask" in data:
                sample["action_op_mask"] = data["action_op_mask"]

        # state 在输出侧统一改名为 proprio（模型侧沿用 proprio 这一命名）；
        # 两个掩码分别对应“时间维”与“维度维”的 padding。
        sample["proprio"] = data["state"]  # [num_obs_steps, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"]  # [num_obs_steps,]
        sample["proprio_dim_is_pad"] = data["proprio_dim_is_pad"]  # [proprio_dim,]

        # 透传定位 / 统计用元信息：idx 必存在；其余字段视数据集是否提供而定。
        sample["idx"] = data["idx"]
        for meta_key in ("task", "embodiment", "dataset_locator"):
            if meta_key in data:
                sample[meta_key] = data[meta_key]

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        把模型输出（归一化空间）还原回物理量（基类单路版本）。

        这是 preprocess 的逆过程，按流水线的相反顺序执行：
            action_state_merger.backward → normalizer.backward → transforms 逆序
        （注意：本基类版本不调用 action_filter.backward，因为过滤器不可逆；
          ActionProcessor.postprocess 会补上这一步。）

        参数：
            data: 至少包含 "action"（归一化后的模型输出）与 "proprio"（合并后的状态）。

        返回：
            反归一化 / 反合并后的 data，其中 "action" 已是逐 part 的物理量字典。

        注意：需要“双路（AR + FM）”推理时请使用 ActionProcessor.postprocess，
        它额外支持 action_fm / action_ar 两个键。
        """
        # 动作是唯一必需的输入：没有动作就没什么可还原的。
        assert "action" in data, "Action is required in postprocess"
        # 把 preprocess 阶段改名的 proprio 改回 state，供合并器的 backward 识别。
        data["state"] = data.pop("proprio")
        # 反向三步：先拆开合并的向量，再反归一化，最后回放变换的逆运算。
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)

        # 动作 chunk 是对齐到“最后一个观测步”的：这里把前面作为条件的前缀切掉，
        # 得到从当前控制步开始的 num_obs_steps 之后的动作（形如 x[:, start:, :]）。
        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data

    @staticmethod
    def _resolve_image_shapes(
        shape_meta: Dict, camera_size_config: Optional[Dict[str, List[int]]] = None
    ) -> None:
        """为 shape_meta["images"] 里的每个条目原地解析出 [C, H, W]。

        解析优先级（camera_size_config 是分辨率的唯一真源，简写为 csc）：
          1. 图像有 camera_type 且 csc 里存在该 camera_type：
             用 csc[camera_type] 作为 (H, W)。
             若 img_meta.shape 也已显式写出但与 csc 不一致，则 csc 覆盖它并打 warning
             （此时 yaml 里的 shape 属于“陈旧信息”，建议直接删掉）。
             通道数 C 取自显式 shape[0]，否则取 raw_shape[0]，再兜底为 3。
          2. csc 里没有对应条目、但 img_meta.shape 已显式写出：直接用这个 shape 作兜底。
          3. 两者都缺失：抛出 ValueError（无法推断尺寸，属于配置错误）。

        注意：本方法是原地修改（会写回 img_meta["shape"]）。
        对同一个 shape_meta 重复调用是幂等的——因为第一次调用已把 shape 改成 csc 的值，
        第二次就不会再出现“显式 shape 与 csc 不一致”的警告。

        参数：
            shape_meta: 数据维度契约，要求其中 "images" 是条目列表。
            camera_size_config: {camera_type: [H, W]}，可为 None（表示完全依赖显式 shape）。
        """
        images = shape_meta.get("images", [])
        for img_meta in images:
            # key 仅用于日志/报错定位；camera_type 用于查分辨率；shape 是可能已存在的显式声明。
            key = img_meta.get("key")
            cam_type = img_meta.get("camera_type")
            explicit_shape = img_meta.get("shape")

            # 分支 1（优先）：用 camera_size_config 覆盖 / 解析分辨率。
            if cam_type and camera_size_config and cam_type in camera_size_config:
                H, W = camera_size_config[cam_type]
                if explicit_shape is not None:
                    # 已有显式 shape：保留它的通道数 C，但如果 (H, W) 与 csc 不一致就告警。
                    C = explicit_shape[0]
                    if list(explicit_shape[-2:]) != [H, W]:
                        logger.warning(
                            f"[shape_meta] Image '{key}' shape {list(explicit_shape)} "
                            f"is overridden by camera_size_config[{cam_type}]={[H, W]} -> "
                            f"[{C}, {H}, {W}]. The shape field in the embodiment yaml is stale; "
                            f"consider removing it."
                        )
                else:
                    # 没有显式 shape：通道数从 raw_shape 推（图像 raw_shape 形如 [C, H_raw, W_raw]）。
                    # raw_shape 不是序列时（例如只写了标量）退化为 3 通道。
                    raw_shape = img_meta.get("raw_shape", [3])
                    C = raw_shape[0] if hasattr(raw_shape, "__len__") else 3
                # 把解析结果写回 shape_meta；后续所有断言都以这个值为准。
                img_meta["shape"] = [C, H, W]
                continue

            # 分支 2：csc 里没有该相机，但 yaml 写了显式 shape —— 直接采用，不做修改。
            if explicit_shape is not None:
                continue

            # 分支 3：既没有可用的 csc，也没有显式 shape —— 无法推断，报错并给出两种修法。
            if cam_type is None:
                raise ValueError(
                    f"Image '{key}' needs either explicit 'shape: [C, H, W]' "
                    f"or 'camera_type' + processor.camera_size_config to resolve shape."
                )
            # 有 camera_type 但 csc 里查不到：列出当前可用的类型，帮助用户快速发现拼写错误。
            if camera_size_config is None or cam_type not in camera_size_config:
                available = list(camera_size_config.keys()) if camera_size_config else []
                raise ValueError(
                    f"Image '{key}' has 'camera_type: {cam_type}' but no matching "
                    f"camera_size_config entry. Either add 'shape: [C, H, W]' to the "
                    f"image config, or set 'processor.camera_size_config.{cam_type}: [H, W]'."
                    + (f" (available types: {available})" if available else "")
                )

            # 以下三行实际上不可达：能执行到这里就必须满足
            # “cam_type 非空 + csc 非空 + cam_type 在 csc 里”，而那种情况已被
            # 开头的分支 1 处理并 continue 掉了。保留此段仅为逻辑兜底，
            # 语义与分支 1 中“无显式 shape”的写法一致（C 取 raw_shape[0]，缺省 3）。
            H, W = camera_size_config[cam_type]
            raw_shape = img_meta.get("raw_shape", [3])
            C = raw_shape[0] if hasattr(raw_shape, "__len__") else 3
            img_meta["shape"] = [C, H, W]

    @staticmethod
    def _ensure_transforms_for_keys(transforms_dict, keys):
        """为缺失的相机 key 补一个空变换列表，返回新的字典。

        用途：数据集侧存在“补零注入”的图像槽位（例如 DROID 的 dummy_wrist_right），
        它们在 shape_meta.images 里已声明，但 embodiment 的 train/val transforms 里没有条目。
        本方法保证这类 key 也有一个（空）变换列表，从而不会在 process_images 里 KeyError；
        随后的 T.Resize 注入仍会对所有 key 生效。

        参数：
            transforms_dict: 原始变换字典，允许为 None。
            keys:           需要保证存在的相机 key 集合。

        返回：
            补全后的新字典（不修改入参）；transforms_dict 为 None 时返回 None。
        """
        if transforms_dict is None:
            return None
        # 复制一份再补键，避免原地改写配置对象（同一份配置可能被其他实例复用）。
        result = dict(transforms_dict)
        for key in keys:
            result.setdefault(key, [])
        return result

    @staticmethod
    def _patch_resize_per_camera(transforms_dict, camera_key_to_size):
        """按 camera_key_to_size 给每个相机的变换列表“最前面”插入一个 T.Resize。

        为什么插在最前面：后续变换（如 ToTensor、Normalize）都假定图像已是目标分辨率，
        先 resize 可以避免在大分辨率上做无谓计算。

        参数：
            transforms_dict:    {相机 key: [transform, ...]}
            camera_key_to_size: {相机 key: [H, W]}，只有出现在这里的 key 才会被插 Resize。

        返回：
            新的变换字典（不修改入参）；transforms_dict 为 None 时返回 None。

        前提假设：启用 camera_size_config 时，embodiment 配置里不应再自行写 T.Resize，
        否则会出现两次缩放（先自动注入一次，再按配置缩放一次）。
        """
        # 局部导入：torchvision 较重，只有真正需要注入 Resize 时才引入。
        import torchvision.transforms as T

        if transforms_dict is None:
            return None
        result = {}
        for key, ts in transforms_dict.items():
            # 不在映射里的相机保持原样（例如数据集补零注入、不需要缩放的占位相机）。
            if key not in camera_key_to_size:
                result[key] = ts
                continue
            # 列表拼接得到新的列表，避免原地修改原始列表对象。
            result[key] = [T.Resize(list(camera_key_to_size[key]))] + list(ts)
        return result


class ActionProcessor(BaseProcessor):
    """
    只处理动作/状态、不涉及 VLM 输入（图像、token 化）的处理器。

    适用场景：只需要动作/状态数值的流水线，例如动作 tokenizer 的训练 / 评估。

    它把公共流水线拆成可复用的两半，方便子类与上层组合：
      · preprocess_action_state(): action_filter → action_state_transform → normalizer → merger
      · build_action_sample():     把处理好的动作/状态写进输出样本（含掩码与 parts 元信息）

    四个加工组件（都从配置注入，含义见 BaseProcessor.__init__）：
      - action_filter:         按维标记/过滤无效动作
      - action_state_transform: 动作/状态变换（例如相对关节量）
      - normalizer:            数值归一化
      - action_state_merger:   多 key 合并成单条固定维度张量（并支持反向拆分）

    注意：图像相关的 train_transforms / val_transforms / camera_size_config 参数仍然保留
    （直接透传给 BaseProcessor），因为子类 FullProcessor 需要它们；
    但本类的 preprocess() 不会读取图像，也不会产出 pixel_values。
    """

    def __init__(
        self,
        # ---------------- 维度契约与观测设置（与 BaseProcessor 相同）----------------
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_state_transforms: Optional[List[Any]],
        # ---------------- action & state 归一化 ----------------
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        action_state_merger,
        action_filter: BaseActionFilter,
        # ---------------- 图像变换（可选）----------------
        # 本类的 preprocess() 不使用图像，这里保留参数只是为了透传给 BaseProcessor，
        # 以及让 FullProcessor / GalaxeaCoTProcessor 复用同一套构造签名。
        train_transforms: Dict[str, List[Any]] | None = None,
        val_transforms: Dict[str, List[Any]] | None = None,
        # 相机分辨率配置（可选，直接透传给 BaseProcessor）：
        # 即使本类不处理图像，也需要它来解析 shape_meta 中图像条目的 shape。
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        # ---------------- 指令（语言）处理（可选）----------------
        # 本类 preprocess() 不产出指令文本，默认值保持“不丢弃高层指令、用英文”的中性行为，
        # 仅当子类需要 augment_instruction 时才真正起作用。
        drop_high_level_prob: float = 0.0,
        use_zh_instruction: bool = False,
        # ---------------- token 化相关（仅用于兼容）----------------
        # MixtureProcessor 会在构造时统一读取各子处理器的 pad_token_id 并断言一致，
        # 因此即使是纯动作处理器，也要保留这个字段。
        pad_token_id: int = 0,
        # ---------------- 归一化可选覆盖项（语义同 BaseProcessor）----------------
        # 逐 key 覆盖归一化模式；None（或缺某个 key）表示该 key 用 norm_default_mode。
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # VLM 输入动作归一化（独立于训练动作归一化）：用于填进文本模板的动作数值。
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # dummy 模式的裁剪范围（仅 norm_mode == "dummy" 时生效）。
        dummy_clip_default: Tuple[float, float] = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        # 尾部压缩系数（仅 norm_mode 为 z-score-tail / q01/q99-tail 时生效）。
        norm_tail_scale: float = 0.075,
    ):
        """构造“只做动作/状态”的处理器，参数全部透传给 BaseProcessor。"""
        # 本类没有额外逻辑，全部直接交给 BaseProcessor：
        # shape_meta 校验、图像尺寸解析、normalizer / merger / filter 的装配都在基类完成。
        super().__init__(
            shape_meta=shape_meta,
            num_obs_steps=num_obs_steps,
            num_output_cameras=num_output_cameras,
            action_state_transforms=action_state_transforms,
            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,
            action_state_merger=action_state_merger,
            action_filter=action_filter,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            camera_size_config=camera_size_config,
            drop_high_level_prob=drop_high_level_prob,
            use_zh_instruction=use_zh_instruction,
            vlm_input_action_norm_default_mode=vlm_input_action_norm_default_mode,
            vlm_input_action_norm_exception_mode=vlm_input_action_norm_exception_mode,
            dummy_clip_default=dummy_clip_default,
            dummy_clip_exception=dummy_clip_exception,
            norm_tail_scale=norm_tail_scale,
        )

        # MixtureProcessor 会断言所有子处理器的 pad_token_id 一致（见 mixture_processor.py）；
        # 纯动作处理器不真正使用它，但必须提供一个值以通过该断言。
        self.pad_token_id = pad_token_id

    def preprocess_action_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行动作/状态加工流水线：
            action_filter → action_state_transform → normalizer → action_state_merger

        这是“只做张量加工、不组装最终样本”的中间步骤，供本类的 preprocess()
        与 FullProcessor / GalaxeaCoTProcessor 的子类流程复用。

        参数：
            data: 原始样本字典（含 "action" / "state" 等键）。

        返回：
            处理后的 data；若配置了 VLM 输入归一化通道，会额外带上 "vlm_action"。
        """
        # 第 1 步：动作过滤（顺带产出 / 更新 action_op_mask）。
        data = self.action_filter.forward(data)
        # 第 2 步：可逆变换（相对动作、旋转表示等），内部自带 raw_shape / shape 断言。
        data = self.action_state_transform(data)
        # 在进入归一化之前先体检一次：旋转变换（如 quaternion_to_matrix、
        # matrix_to_euler_angles）在退化姿态下可能产生 NaN/Inf，
        # 若不在此处暴露，归一化会把这些坏值“静默地”带进后续所有环节。
        _warn_nan_in_batch(data, stage="post_transform")

        # 如果配置了 VLM 输入动作归一化器，就另外归一化一份“动作副本”给 VLM 用。
        # 注意：这份副本只经过 VLM 归一化器 + 合并器，不参与训练动作的归一化，
        # 因此两套数值空间互不影响（模板里填的是 vlm_action，训练目标仍是 data["action"]）。
        vlm_action = None
        if self._vlm_input_action_normalizer is not None and "action" in data:
            # 用 deepcopy 隔离：VLM 归一化器是原地修改的，不能污染训练用的动作数据。
            vlm_action_data = {"action": deepcopy(data["action"])}
            self._vlm_input_action_normalizer.forward(vlm_action_data)
            # 合并器同样需要 action_is_pad 才能产出正确的维度/时间掩码，这里从主数据借用。
            vlm_action_data = self.action_state_merger.forward(
                {
                    "action": vlm_action_data["action"],
                    "action_is_pad": data["action_is_pad"],
                }
            )
            vlm_action = vlm_action_data["action"]

        # 第 3 步：训练用的归一化 + 合并。
        data = self.normalizer.forward(data)
        # 再体检一次：理论上 normalizer 内部的 nan_to_num 已把坏值处理掉（此处应为 0），
        # 但仍然记录日志——一旦出现就说明处理逻辑与预期不符，值得排查。
        _warn_nan_in_batch(data, stage="post_norm")
        data = self.action_state_merger.forward(data)

        # 把 VLM 专用的动作副本挂到返回结果上（下游 SamplesBuilder 会取走这个字段）。
        if vlm_action is not None:
            data["vlm_action"] = vlm_action

        return data

    def build_action_sample(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        从加工后的 data 中抽取动作/状态相关字段，写进输出样本 sample。

        必须在 preprocess_action_state() 之后调用（它依赖后者产出的
        action / action_is_pad / action_dim_is_pad / state / proprio_dim_is_pad 等字段）。

        参数：
            data:   已加工完成的样本字典。
            sample: 待补充的输出样本字典（原地更新并返回，便于链式组装）。

        返回：
            补充了 action / proprio 及各类掩码的 sample。
        """
        # 动作相关字段：仅在样本确实带动作（训练 / 开环评估）时写入。
        if "action" in data:
            sample["action"] = data["action"]
            # 时间维掩码：哪些时间步不参与 loss。
            sample["action_is_pad"] = data["action_is_pad"]
            # 维度维掩码：合并器为对齐而补出的维度不参与 loss。
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]
            # 部分合并器（如 GroupedPaddingMerger）会记录各 part 的原始维度布局；
            # 训练侧据此把扁平动作向量还原成各 part，做分组 loss / 分组指标。
            # 用 getattr 做兼容判断：并非所有合并器都提供 max_action_shape_meta。
            if getattr(self.action_state_merger, "max_action_shape_meta", None) is not None:
                sample["action_parts_meta"] = dict(self.action_state_merger.max_action_shape_meta)
            # 有效操作维掩码（可选字段）。
            if "action_op_mask" in data:
                sample["action_op_mask"] = data["action_op_mask"]

        # 状态统一以 proprio 命名输出（与模型侧字段一致）；
        # state_is_pad / proprio_dim_is_pad 分别是时间维、维度维的 padding 掩码。
        sample["proprio"] = data["state"]
        sample["proprio_is_pad"] = data["state_is_pad"]
        sample["proprio_dim_is_pad"] = data["proprio_dim_is_pad"]

        # 控制频率（Hz）可选：动作 tokenizer / 模型侧需要它把“动作步”换算成真实时间跨度。
        if "frequency" in data:
            sample["frequency"] = data["frequency"]

        return sample

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        把原始样本加工成“只有动作/状态”的张量样本（不产出图像与文本）。

        与 BaseProcessor.preprocess 的区别：
          · 完全不处理 images，输出里没有 pixel_values；
          · gt_action 取自“变换之后、归一化之前”（BaseProcessor 则在变换之前取），
            这样动作已经过相对动作 / 旋转表示等变换，可直接用于 tokenizer 评估对照。

        参数：
            data: 原始样本，至少包含 "action" / "state" / "idx"。

        返回：
            含 action / proprio 及各类掩码的样本（见 build_action_sample）。
        """
        sample = {}

        # 先做过滤 + 可逆变换（这两步不涉及统计量，可复用于推理侧）。
        data = self.action_filter.forward(data)
        data = self.action_state_transform(data)

        # 在归一化之前留一份动作副本作为 gt_action（供 tokenizer 评估做对照）。
        # 这里用 merger.forward 只做“拼接 + padding”，保持物理量纲不变；
        # deepcopy 用于隔离后续归一化对原数据的改写。
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # 再做归一化 + 合并。
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        # 把数值字段与掩码搬进输出样本，并透传样本下标。
        sample = self.build_action_sample(data, sample)

        sample["idx"] = data["idx"]

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        把模型输出（归一化空间）还原回物理量，支持“双路（AR + FM）”推理输出。

        输入里的三类动作键：
          - action_fm: Flow-Matching 路径输出的动作（z-score 等归一化空间）
                       → 需要经过 self.normalizer 反归一化；
          - action_ar: 自回归路径输出的动作（dummy 空间 / 已是物理量）
                       → 数值上不需要缩放，但仍要走一遍反向流程做拆解与变换逆运算；
          - action:    向后兼容键，最终按其取值优先级解析为 action_fm > action_ar。

        两条路径都会执行完整的反向流程：
            merger.backward → normalizer.backward → transforms 逆序 → action_filter.backward
        最后统一按 [:, num_obs_steps-1:, :] 切掉条件前缀，只保留预测区间。

        注意：这里显式调用了 action_filter.backward()（基类版本没有），
        因为 ActionProcessor 处理的是可能带过滤/掩码的完整流程。
        """
        # 局部导入 logging：让本方法在不想引入模块级 logger 的调用场景下也能单独工作。
        import logging

        logger = logging.getLogger(__name__)

        def _action_stats(action_val):
            """给日志用的动作数值摘要，兼容 dict（逐 part）与 tensor 两种形态。"""
            if isinstance(action_val, dict):
                # 逐 part 的 dict：打印有哪些 key（形状/量纲由 part 决定，不做数值统计）。
                all_tensors = [
                    v.flatten() for v in action_val.values() if isinstance(v, torch.Tensor)
                ]
                if all_tensors:
                    return f"dict with {list(action_val.keys())}"
                return "dict (no tensors)"
            elif isinstance(action_val, torch.Tensor):
                # 单个张量：打印 min/max，便于快速判断量纲是否落在合理区间。
                return f"min={action_val.min().item():.4f}, max={action_val.max().item():.4f}"
            return str(type(action_val))

        # 先判断这次是双路输出还是旧版单路输出。
        has_fm = "action_fm" in data
        has_ar = "action_ar" in data

        # 下面三行 debug 日志用于排查“推理结果量纲不对”类问题：
        # 打印走的是哪条路径、以及当前 processor 的归一化配置。
        logger.debug(f"[postprocess] has_fm={has_fm}, has_ar={has_ar}")
        logger.debug(
            f"[postprocess] processor normalizer: norm_default_mode={getattr(self, 'norm_default_mode', 'N/A')}"
        )
        logger.debug(
            f"[postprocess] processor vlm_input_action_norm_default_mode={getattr(self, 'vlm_input_action_norm_default_mode', 'N/A')}"
        )

        # ---------------- FM 路径：需要反归一化 ----------------
        if has_fm:
            logger.debug(f"[postprocess] BEFORE FM denorm: {_action_stats(data['action_fm'])}")
            # 只挑出反向流程需要的字段；state 取 "state"（评估脚本传入）或 "proprio"（样本字段名）。
            fm_data = {
                "action": data["action_fm"],
                "state": data.get("state", data.get("proprio")),
            }
            # 反向四步：拆解合并向量 → 反归一化 → 回放变换逆运算 → 恢复过滤器语义。
            fm_data = self.action_state_merger.backward(fm_data)
            fm_data = self.normalizer.backward(fm_data)
            if self.action_state_transforms is not None:
                for trans in reversed(self.action_state_transforms):
                    fm_data = trans.backward(fm_data)
            fm_data = self.action_filter.backward(fm_data)
            # 切掉前 num_obs_steps-1 步（那是作为条件的观测前缀），只保留预测区间。
            start_obs_step = self.num_obs_steps - 1
            data["action_fm"] = dict_apply(fm_data["action"], lambda x: x[:, start_obs_step:, :])
            logger.debug(f"[postprocess] AFTER FM denorm: {_action_stats(data['action_fm'])}")

        # ---------------- AR 路径：同样走完整反向流程 ----------------
        if has_ar:
            logger.debug(f"[postprocess] BEFORE AR denorm: {_action_stats(data['action_ar'])}")
            ar_data = {
                "action": data["action_ar"],
                "state": data.get("state", data.get("proprio")),
            }
            ar_data = self.action_state_merger.backward(ar_data)
            ar_data = self.normalizer.backward(ar_data)
            if self.action_state_transforms is not None:
                for trans in reversed(self.action_state_transforms):
                    ar_data = trans.backward(ar_data)
            ar_data = self.action_filter.backward(ar_data)
            start_obs_step = self.num_obs_steps - 1
            data["action_ar"] = dict_apply(ar_data["action"], lambda x: x[:, start_obs_step:, :])
            logger.debug(f"[postprocess] AFTER AR denorm: {_action_stats(data['action_ar'])}")

        # ---------------- 旧版单路回退 ----------------
        # action_fm / action_ar 都不存在时，认为调用方走的是历史单路接口：
        # 要求 data 里已有 "action" 与 "proprio"，处理完毕后直接返回。
        if not has_fm and not has_ar:
            assert "action" in data, "Action is required in postprocess"
            logger.debug(f"[postprocess] Fallback (legacy): {_action_stats(data['action'])}")
            data["state"] = data.pop("proprio")
            data = self.action_state_merger.backward(data)
            data = self.normalizer.backward(data)
            if self.action_state_transforms is not None:
                for trans in reversed(self.action_state_transforms):
                    data = trans.backward(data)
            data = self.action_filter.backward(data)
            start_obs_step = self.num_obs_steps - 1
            data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
            logger.debug(f"[postprocess] AFTER fallback denorm: {_action_stats(data['action'])}")
            return data

        data["action"] = data["action_fm"] if has_fm else data["action_ar"]
        logger.debug(f"[postprocess] Final 'action' key set: {_action_stats(data['action'])}")
        return data


class FullProcessor(ActionProcessor):
    """
    处理完整 VLM 输入（图像 + 动作/状态）的处理器。

    在 ActionProcessor 的基础上扩展出：
      - 图像处理：逐相机做变换，并按 camera_type 保留为 dict（支持多分辨率）；
      - tokenizer 元信息：pad_token_id / image_token_index / max_text_tokens，
        以及可选的 tokenizer 实例（真正的 token 化仍在模型侧 InputPreprocessor 完成）；
      - VLM 输入动作归一化：给文本模板里的动作数值准备独立的一套缩放。

    适用场景：带图像与语言指令的 VLM 训练 / 推理。
    当前配置默认使用的 GalaxeaCoTProcessor 就是本类的子类
    （它在 preprocess 里再叠加 SamplesBuilder 的 CoT 模板构造）。
    """

    def __init__(
        self,
        # ---------------- 维度契约与观测设置（同 BaseProcessor）----------------
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_state_transforms: Optional[List[Any]],
        # ---------------- action & state 归一化 ----------------
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        action_state_merger,
        action_filter: BaseActionFilter,
        # ---------------- 图像变换（本类必需）----------------
        # {相机 key: [transform, ...]}；BaseProcessor 会补键并在列表最前面注入 T.Resize。
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None,
        # ---------------- 指令（语言）处理 ----------------
        drop_high_level_prob: float,
        use_zh_instruction: bool,
        # ---------------- token 化元信息 ----------------
        # 以下四个值必须与模型侧配置（configs 里的 model.model_arch / processor）保持一致：
        pad_token_id: int,
        # 图像占位 token 的 id：模板文本里用它占位，由模型侧替换为视觉特征。
        image_token_index: int,
        # 指令文本的最大 token 数（真正做截断的是模型侧 InputPreprocessor，这里只是透传）。
        max_text_tokens: int,
        # 输入的相机帧总数（= 相机数 × 观测帧数），用于与像素张量的槽位数对齐。
        num_input_cameras: int,
        # 相机分辨率配置：{camera_type: [H, W]}，用于从 camera_type 自动解析 shape。
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        # tokenizer 加载参数（HF 路径 + 认证 token 等），为 None 时不加载 tokenizer。
        tokenizer_params: Optional[Dict[str, Any]] = None,
        # ---------------- 归一化可选覆盖项（语义同 BaseProcessor）----------------
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # dummy 模式的裁剪范围（仅 norm_mode == "dummy" 时生效）。
        dummy_clip_default: Tuple[float, float] = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ):
        """构造带图像处理的处理器，参数全部透传给 ActionProcessor / BaseProcessor。

        额外副作用：保存 tokenizer 相关元信息，并在提供了 tokenizer_params 时加载 tokenizer。
        """
        # 注意：norm_tail_scale 没有出现在本类签名中（沿用 ActionProcessor 的默认值 0.075）。
        super().__init__(
            shape_meta=shape_meta,
            num_obs_steps=num_obs_steps,
            num_output_cameras=num_output_cameras,
            action_state_transforms=action_state_transforms,
            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,
            action_state_merger=action_state_merger,
            action_filter=action_filter,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            camera_size_config=camera_size_config,
            drop_high_level_prob=drop_high_level_prob,
            use_zh_instruction=use_zh_instruction,
            vlm_input_action_norm_default_mode=vlm_input_action_norm_default_mode,
            vlm_input_action_norm_exception_mode=vlm_input_action_norm_exception_mode,
            dummy_clip_default=dummy_clip_default,
            dummy_clip_exception=dummy_clip_exception,
        )

        # MixtureProcessor 要求各子处理器的 pad_token_id 一致，这里保存供其断言。
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        # 只有显式给了 tokenizer_params 才真正加载 tokenizer：
        # 训练/评估路径通常不需要它（token 化在模型侧），可省掉一次加载开销。
        self.tokenizer = (
            self._load_tokenizer(tokenizer_params) if tokenizer_params is not None else None
        )
        self.max_text_tokens = max_text_tokens
        # 注意属性名与构造参数不同：对内存的是 num_input_images 而不是 num_input_cameras。
        # 该值会继续传给 SamplesBuilder（用于模板槽位计数）。
        self.num_input_images = num_input_cameras

    def _load_tokenizer(self, tokenizer_params):
        """加载 tokenizer（默认走 transformers 的 AutoProcessor）。子类可覆盖以适配自定义加载方式（如 Qwen3.5）。"""
        from transformers import AutoProcessor

        # 用 AutoProcessor 是因为部分多模态模型只在 processor 里暴露 tokenizer；
        # 这里只取其中的 .tokenizer，图像处理仍由本文件负责。
        processor = AutoProcessor.from_pretrained(**tokenizer_params)
        return processor.tokenizer

    def process_images(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """对每个相机做变换，并“按相机分别保留”，而不是拼成一个大张量。

        与 BaseProcessor.preprocess 的差异：那里用 torch.cat 把所有相机拼成一个
        [N, C, H, W] 张量，要求所有相机分辨率一致；这里改用 dict，允许不同相机
        使用不同的 (H, W)，即支持多分辨率输入。

        参数：
            data: 原始样本，要求 data["images"] 是 {相机 key: Tensor[T, C, H, W]}。

        返回：
            pixel_values: Dict[str, Tensor[num_obs_steps, C, H_k, W_k]]
                键是 camera_type（不是 shape_meta 里的 key），值是该相机自己的张量。
        """
        result: Dict[str, torch.Tensor] = {}
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]
            # 与基类相同：约定输入为 4 维（帧、通道、高、宽）。
            assert image.ndim == 4, (
                f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"
            )

            # 训练 / 评估各用一套变换；transform 列表里已包含自动注入的 T.Resize。
            transforms = self.train_transforms if self.is_train else self.val_transforms
            for trans in transforms[key]:
                image = trans(image)

            # 变换结果必须严格等于 [num_obs_steps, C, H, W]（H、W 来自本相机自己的 shape）。
            meta_shape = tuple([self.num_obs_steps] + shape)
            assert image.shape == meta_shape, (
                f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"
            )

            # 用 camera_type 作为输出键：同一相机类型在多 embodiment 间语义一致，
            # 便于模型侧按 camera_type 绑定对应的视觉输入槽位。
            result[meta["camera_type"]] = image  # [num_obs_steps, C, H_k, W_k]

        return result

    def build_pixel_values(self, pixel_values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """把 pixel_values 的“相机帧总数”调整到 num_output_cameras。

        两种情形：
          · 多分辨率（各相机的 H/W 不同）：
              要求帧数总和恰好等于 num_output_cameras，满足时原样返回；
              不满足则直接报错 —— 不同分辨率的张量无法在同一个张量里补零/挑选。
          · 同分辨率（所有相机的 H/W 相同）：
              允许通过“补零”或“前 N 帧 / 随机选取”把数量对齐到 num_output_cameras。

        参数：
            pixel_values: {camera_type: Tensor[T, C, H, W]}，来自 process_images()。

        返回：
            调整后的同结构 dict（键与顺序保持不变）。
        """
        # 统计总帧数（每个相机的第 0 维之和）。
        n = sum(v.shape[0] for v in pixel_values.values())
        # 数量刚好匹配是理想情况：直接返回，不引入任何额外改动。
        if n == self.num_output_cameras:
            return pixel_values

        # 判断所有相机的空间形状是否一致（即是否同分辨率）。
        shapes = [tuple(v.shape[1:]) for v in pixel_values.values()]
        if len(set(shapes)) > 1:
            # 多分辨率无法跨相机补零（张量形状不同），只能要求数量严格匹配。
            raise ValueError(
                f"Multi-resolution cameras require n_frames == num_output_cameras "
                f"({n} != {self.num_output_cameras}). Padding/selection not supported "
                "across cameras of different resolutions."
            )
        else:
            # 同分辨率：允许补零/挑选，但这里会打一条 warning，
            # 因为“数量不匹配”通常意味着配置或数据有问题，值得被注意到。
            import logging

            logging.warning(
                f"All cameras share the same resolution {shapes[0]}. "
                f"Applying padding/selection to match num_output_cameras={self.num_output_cameras}."
            )

        # 同分辨率路径：先摊平成 [n, C, H, W]，做补零/挑选，再按 key 重新切开。
        keys = list(pixel_values.keys())
        flat = torch.cat(list(pixel_values.values()), dim=0)  # [n, C, H, W]
        if n < self.num_output_cameras:
            # 数量不足：补零张量（零值区由模型侧的掩码/槽位约定负责忽略）。
            out = torch.zeros(
                (self.num_output_cameras,) + flat.shape[1:],
                device=flat.device,
                dtype=flat.dtype,
            )
            out[:n] = flat
            flat = out
        else:  # n > num_output_cameras
            if self.is_train:
                # 训练：随机选 num_output_cameras 帧（再排序保持时间顺序稳定），
                # 相当于一种对“多余帧”的随机采样增强。
                indices = torch.randperm(n, device=flat.device)[: self.num_output_cameras]
                indices = indices.sort().values
            else:
                # 评估：固定取前 num_output_cameras 帧，保证结果可复现。
                indices = torch.arange(self.num_output_cameras, device=flat.device)
            flat = flat[indices]

        # 重新分发：把调整后的帧“均匀”切回各个相机 key。
        # 注意这里不是按原相机帧数还原，而是平均分——因此只在同分辨率、
        # 且各相机帧数相等（通常就是同一 num_obs_steps）时才语义正确。
        n_out = flat.shape[0]
        frames_per_key = n_out // len(keys)
        return {k: flat[i * frames_per_key : (i + 1) * frames_per_key] for i, k in enumerate(keys)}

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        把原始样本加工成“图像 + 动作/状态”的张量样本（FullProcessor 版本）。

        与 ActionProcessor.preprocess 的区别：
          · 增加了图像处理，pixel_values 是 {camera_type: Tensor} 的字典（支持多分辨率）；
          · 动作/状态部分复用 preprocess_action_state()，语义与基类一致；
          · gt_action 取自“变换 / 归一化之前”的原始动作（与 BaseProcessor 的取法一致）。

        参数：
            data: 由数据集 __getitem__ 产出的原始样本（LeRobot / mcap 语义）：
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state":  Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad":  torch.Tensor -> [num_obs_steps,]
                - "image_is_pad":  torch.Tensor -> [num_obs_steps,]
                - "idx": int，样本下标

        返回：
            sample: 可直接 collate 的样本字典（节选）：
                - "pixel_values":     Dict[str, Tensor[num_obs_steps, C, H_k, W_k]]
                - "proprio":          torch.Tensor -> [num_obs_steps, proprio_dim]
                - "proprio_is_pad":   torch.Tensor -> [num_obs_steps,]
                - "proprio_dim_is_pad": torch.Tensor -> [proprio_dim,]
                - "action":           Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad":    Optional, torch.Tensor -> [action_horizon,]
                - "action_dim_is_pad":Optional, torch.Tensor -> [action_dim,]
                - "gt_action":        Optional，归一化前的原始动作副本（开环评估对照）
                - "idx": int，样本下标
            说明：旧版 docstring 里写的 "input_ids" / "attention_mask" / "state_is_pad"
            不由本方法产出（token 化在模型侧；实际的 padding 字段名是 proprio_is_pad）。
        """
        sample = {}

        # ---------------- 步骤 1：图像 ----------------
        # 先按相机分别变换，再把帧数对齐到 num_output_cameras（多分辨率下要求严格相等）。
        pixel_values = self.process_images(data)
        sample["pixel_values"] = self.build_pixel_values(pixel_values)

        # ---------------- 步骤 2：原始动作对照（开环评估用）----------------
        # 与 BaseProcessor 相同：在过滤 / 变换 / 归一化之前深拷贝一份合并后的原始动作。
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # ---------------- 步骤 3：动作 / 状态流水线 ----------------
        # 复用 ActionProcessor 的实现：过滤 → 变换（含 NaN 体检）→ 归一化 → 合并，
        # 若有 VLM 归一化通道，还会额外产出 data["vlm_action"]。
        data = self.preprocess_action_state(data)
        sample = self.build_action_sample(data, sample)

        sample["idx"] = data["idx"]

        return sample
