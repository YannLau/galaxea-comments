# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/processor/galaxea_cot_processor.py
#     —— GalaxeaCoTProcessor：VLM（视觉-语言-动作）训练 / 推理用的“两阶段”数据处理器
# =============================================================================
#
# 【这个文件是什么】
#   它定义了 GalaxeaCoTProcessor —— 当前仓库**默认使用**的数据处理器
#   （configs/model/g05.yaml 里 processor._target_ 就指向本类）。
#   一句话概括职责：把数据集 __getitem__ 取出的**原始样本**（原始 action/state 张量、
#   相机画面、任务文本）加工成“数值张量 + 模板化文本样本”，交给 DataLoader 拼批，
#   最后由模型侧完成 token 化。
#
#   它是 FullProcessor 的子类（继承链见下），在本仓库里额外承担三件事：
#     1) **两阶段 preprocess**：先做张量加工（阶段 1 `_process_tensors`），
#        再用 SamplesBuilder 组装 CoT 样本模板（阶段 2 `samples_builder.build`）；
#     2) 收拢一批“训练 / 调试开关”：指令覆盖（hardcode）、VLM 专用动作归一化、
#        embodiment 同步、离散动作与合并器的兼容性校验；
#     3) 为中英双语指令字段做统一的语言选择（"中文@English" → 取其中一边）。
#   注：历史上这些开关曾由一个单独的子类 GalaxeaCoTProcessorHardcode 提供，
#       现在已合并进本类的构造参数（见 __init__ 里 hardcode_* 三个参数）。
#
# 【职责三分——先弄清“谁负责什么”，否则很容易改错文件】
#   · GalaxeaCoTProcessor（本文件）—— 负责**张量加工**
#       图像变换与像素张量组装、动作/状态变换+归一化+合并、指令文本选择，
#       并在 sample 上准备好两个供下一阶段消费的临时字段（_instructions / _vlm_action）。
#   · SamplesBuilder（同目录 samples_builder.py）—— 负责**文本模板组装**
#       决定“这一帧用哪种 CoT 格式”（有没有 CoT、CoT 写在模板哪里、指令槽填什么），
#       产出 RoboVQA 风格的 samples dict（template + command + proprio + action + 各 CoT 槽位）。
#   · InputPreprocessor（模型侧 src/g05/models/g05/io/input_preprocessor.py）—— 负责**token 化**
#       解析模板里的占位符、分词、拼出 input_ids / labels / attention_mask。
#   ⚠️ 结论：**本文件不与 tokenizer 打交道**，也不产出 input_ids。
#      构造参数里的 tokenizer_params 只是历史兼容，本类甚至不会把它传下去。
#
# 【继承链（新手先看这张“家谱”）】
#
#   BaseProcessor(ABC)                         base_processor.py
#     │  公共骨架：解析/校验 shape_meta、持有 normalizer、train()/eval()、图像 shape 解析
#     └── ActionProcessor(BaseProcessor)        base_processor.py
#           │  只做动作/状态：action_filter → transform → normalizer → merger
#           └── FullProcessor(ActionProcessor)  base_processor.py
#                 │  增加多分辨率图像处理 + tokenizer 元信息
#                 └── GalaxeaCoTProcessor        ← 本文件（当前配置默认使用）
#                       在 preprocess 里再叠加 SamplesBuilder 的模板构造，
#                       并把 preprocess 拆成“张量阶段 / 模板阶段”两段。
#
# 【它在整条数据链路中的位置】
#
#   LeRobotDataset.__getitem__                   src/g05/data/lerobot/lerobot_dataset_v3.py
#     │  · 读 parquet 数值列 + mp4 画面；把 *_index 翻回文本（task/atomic_task/bbox/memory/...）
#     │  · 末尾调用 sample = processor.preprocess(sample)
#     ▼
#   GalaxeaCoTProcessor.preprocess               ← 本文件（两阶段）
#     │  阶段 1  _process_tensors(data)            ：图像 / 动作 / 状态 / 指令 → 张量
#     │  阶段 2  samples_builder.build(data, sample)：CoT 模板 + 槽位文本 → samples dict
#     ▼
#   DataLoader 拼批（collate）
#     ▼
#   InputPreprocessor                            src/g05/models/g05/io/input_preprocessor.py
#     │  解析模板占位符、分词、拼 input_ids / labels / attention_mask
#     ▼
#   模型前向（训练：算 CE loss；推理：先自回归生成 CoT 文本 → 再生成/解码动作 token）
#
#   另外两个调用方（都在推理侧）：
#     · src/g05/models/g05/inferencer.py      preprocess(obs) → 模型前向 → postprocess(...)["action"]
#     · scripts/eval_open_loop.py             用 gt_action 做开环对照
#
# 【两阶段到底各做了什么（本文件的核心）】
#
#   阶段 1：_process_tensors(data)   —— 只产出数值张量与“待填进模板”的内容
#     ① 双语文本规范化：把 task / atomic_task / future_task / high_level_instruction /
#        action_hint 这些形如 "中文@English" 的字段按 use_zh_instruction 取其中一边；
#     ② 指令：augment_instruction(data)（本类覆盖了基类实现，见方法注释）；
#     ③ 图像：process_images()（逐相机变换，按 camera_type 保留为 dict）
#            → build_pixel_values()（把相机帧数对齐到 num_output_cameras）；
#     ④ gt_action：在过滤/变换/归一化**之前**深拷贝一份“合并后的原始动作”，
#        专供开环评估与 tokenizer 评估做对照（train_utils.py 会优先用 vlm_action，
#        取不到才回退到它）；
#     ⑤ 动作/状态流水线：preprocess_action_state()，即
#        action_filter → action_state_transform → normalizer → action_state_merger；
#        若配置了 VLM 专用归一化通道，这一步还会额外产出 data["vlm_action"]；
#     ⑥ 组装输出：build_action_sample() + idx + 少量元信息透传；
#     ⑦ 留下两个**下划线开头的临时字段**（阶段 2 会取走，最终样本里不存在）：
#          sample["_instructions"] —— 指令文本（模板 command 槽的来源）
#          sample["_vlm_action"]   —— VLM 视角的动作副本，可能为 None
#
#   阶段 2：samples_builder.build(data, sample)  —— 组装 RoboVQA 样本
#     由配置 processor.samples_builder._target_ 决定使用哪个 Builder（默认无 CoT）。
#     Builder 会 pop 掉上面两个临时字段，返回 samples dict：
#          samples["template"]  —— 带占位符的模板字符串
#          samples["command"]   —— 填进 <command_text_!_200> 的指令
#          samples["proprio"] / samples["action"] —— 状态 / 动作（含各种掩码与 parts_meta）
#          samples["<CoT 槽>"]  —— 该 Builder 自己引入的 CoT 文本槽
#     具体模板语法、有哪些 Builder、怎么新增一个，见 samples_builder.py 的文件头。
#
# 【输出契约（preprocess 的返回值）】
#     {
#       "pixel_values":        Dict[camera_type, Tensor[T, C, H, W]]  # FullProcessor 的多分辨率字典形式
#       "proprio":             Tensor[T, D_state]        # 注：字段名是 proprio，不是 state
#       "proprio_is_pad":      Tensor[T] (bool)          # 时间维 padding 掩码
#       "proprio_dim_is_pad":  Tensor[D_state] (bool)    # 维度维 padding 掩码
#       "action":              Optional Tensor[H, D_action]
#       "action_is_pad":       Optional Tensor[H] (bool)
#       "action_dim_is_pad":   Optional Tensor[D_action] (bool)
#       "action_parts_meta":   Optional，各 part 的维度布局（部分合并器才提供）
#       "action_op_mask":      Optional，动作过滤器标出的“有效操作维”
#       "gt_action":           Optional，深拷贝的原始动作（开环评估对照）
#       "frequency":           Optional，控制频率 Hz（动作 tokenizer 需要）
#       "samples":             阶段 2 的产物（模板 + 各槽位文本），交给 InputPreprocessor
#       "idx":                 int，样本下标
#       "task" / "embodiment" / "dataset_locator": 透传的元信息（便于排查坏样本）
#     }
#   注意："_instructions" / "_vlm_action" 这两个临时字段**不会**出现在返回值里
#   （已被 SamplesBuilder pop 掉；若哪天它们残留下来，collate 会因为 “str 无法 stack” 而报错）。
#
# 【配置长什么样（摘自 configs/model/g05.yaml，节选）】
#
#   processor:
#     _target_: g05.data_processor.processor.galaxea_cot_processor.GalaxeaCoTProcessor
#     num_obs_steps: 1
#     discrete_action: ${model.model_arch.discrete_action}
#     use_stepwise_action_norm: true
#     num_output_cameras: 3
#     use_zh_instruction: false
#     drop_high_level_prob: 1.0
#     pad_token_id: ${model.model_arch.pad_token_id}
#     image_token_index: ${model.model_arch.image_token_index}
#     tokenizer_params: {...}          # 仅兼容旧配置；本类不加载 tokenizer
#     max_text_tokens: ${model.model_arch.max_text_tokens}
#     num_input_cameras: 3
#     camera_size_config: {exterior: [256, 256], wrist_left: [256, 256], wrist_right: [256, 256]}
#     action_state_merger: {_target_: ...GroupedPaddingMerger, ...}
#     # samples_builder 不写时 = BaseSamplesBuilder（无 CoT）；
#     # 想开 CoT 就按 docs/data/samples_builders_zh.md 第 8 节写 _target_ + _partial_: true
#
#   配置到实例的转换：src/g05/utils/data/processor_utils.py 的 build_processors()
#     · data 配置里没有 processors      → 直接 instantiate(cfg.model.processor)
#     · 有（多 embodiment 混合）        → 每个具身各建一个，再包进 MixtureProcessor
#   processor.samples_builder / action_state_merger 这类嵌套对象由 Hydra 递归实例化；
#   Builder 通常写成 `_partial_: true` + `_recursive_: false`，得到一个“可调用对象”，
#   再由本类补上运行时才知道的参数（num_input_images / image_sizes / embodiment_type ...）。
#
# 【生命周期（顺序不能颠倒）】
#     1) Hydra 实例化（__init__ 会做全部静态校验，包括离散动作与合并器的兼容性检查）；
#     2) ds.set_processor(p) → MixtureLerobotDataset 会把 p.embodiment_type 回写到本类
#        （本类的 setter 再同步给 samples_builder，模板里的 <embodiment_text_!> 靠它填值）；
#     3) 显式调用 train() 或 eval()（否则读 processor.is_train 会直接抛 ValueError）；
#     4) set_normalizer_from_stats(dataset_stats)（否则读 processor.normalizer 会抛 ValueError）；
#     5) 之后才能在 __getitem__ / inferencer 里调用 preprocess() / postprocess()。
#
# 【新手最容易踩的坑】
#   1) 忘记 train()/eval()：阶段 1 的 process_images() 会读 self.is_train 选变换栈，直接报错。
#   2) 忘记 set_normalizer_from_stats()：normalizer 属性未初始化就抛 ValueError。
#   3) 把 _instructions / _vlm_action 留下来：它们分别是 str / 可选张量，残留会污染 collate
#      与下游校验；正确的交接方式是交给 SamplesBuilder 的 build() 去 pop。
#   4) 以为本类会加载 tokenizer：tokenizer_params 只是兼容旧配置，本类**不传给父类**，
#      因此 self.tokenizer 恒为 None。真正的 token 化在模型侧 InputPreprocessor。
#   5) 配 discrete_action=True 却把 action_state_merger 写成 ConcatLeftAlign：
#      __init__ 会直接 ValueError —— 离散动作 tokenizer 需要“逐 part 对齐填充”的掩码布局，
#      扁平拼接会丢掉 part 边界（改用 PaddingActionMerger / GroupedPaddingMerger）。
#   6) 以为 drop_high_level_prob / coarse_task 的 “[High]/[Low]” 拼接还生效：
#      那是基类 augment_instruction 的行为，本类已覆盖该方法（只返回 task 或固定文本）。
#   7) 以为 vlm_action 和 action 是一回事：前者只用于“填进模板文本的动作数值”，
#      其归一化口径由 vlm_input_action_norm_* 单独控制，两者互不影响。
#   8) 在没有 action 的纯观测推理样本上调 preprocess：此时没有 action / gt_action /
#      _vlm_action（均为 None 或缺失），模板里的 <action_action> 槽会缺值 —— 属预期行为。
#
# 【相关文件与文档】
#   src/g05/data_processor/processor/base_processor.py       父类与三段流水线实现
#   src/g05/data_processor/processor/samples_builder.py      阶段 2：模板与 CoT 槽位
#   src/g05/data_processor/processor/mixture_processor.py    多 embodiment 分发
#   src/g05/data_processor/transforms/action_state_merger.py 合并器（含本例的离散动作约束）
#   src/g05/data_processor/transforms/action_filter.py       action_op_mask 的来源
#   src/g05/utils/data/normalizer.py                         LinearNormalizer 与 NormMode
#   src/g05/utils/data/processor_utils.py                    build_processors()：配置 → 实例
#   src/g05/data/base_lerobot_dataset.py                     preprocess 的调用方（__getitem__ 末尾）
#   src/g05/data/mixture_lerobot_dataset.py                  set_processor()：回写 embodiment_type
#   src/g05/models/g05/io/input_preprocessor.py              占位符解析与 token 化
#   src/g05/models/g05/inferencer.py                         推理侧 preprocess / postprocess
#   src/g05/utils/training/train_utils.py                    tokenizer 评估里 vlm_action / gt_action 的用法
#   configs/model/g05.yaml                                   processor: 段完整参数示例
#   docs/data/samples_builders_zh.md                         Builder 总览与选型
#   docs/data/schema_zh.md                                   数据字段来源（parquet 列 → item 字段）
#   docs/architecture/g05_io_zh.md                           Dataset → Collate → 模型的张量形状
# =============================================================================

# typing：只用于类型标注（Dict / Any / Optional / List）。
from typing import Dict, Any, Optional, List

# FullProcessor：本类的父类（base_processor.py），提供图像处理 + 动作/状态流水线。
from .base_processor import FullProcessor
# BaseSamplesBuilder：默认的样本模板构建器（无 CoT）。配置里可用 _target_ 换成别的 Builder，
# 例如 MixedSamplesBuilder（多种 CoT 格式加权采样）。
from .samples_builder import BaseSamplesBuilder
# deepcopy：给 gt_action 留一份不受后续原地修改影响的拷贝。
from copy import deepcopy
# NormMode：归一化模式的联合类型（z-score / z-score-tail / q01/q99 / dummy ...），仅用于类型标注。
from g05.utils.data.normalizer import NormMode
# BaseActionFilter：动作过滤器基类，仅用于构造参数的类型标注
# （真正的实例由 Hydra 按配置注入，例如 R1LiteJointActionFilter / DummyActionFilter）。
from g05.data_processor.transforms.action_filter import BaseActionFilter

import logging

# 模块级 logger：本文件只在 __init__ 里打一条 debug（记录 embodiment_type），
# 排查“模板里的具身名不对”时可以把日志级别调到 DEBUG 看这行。
logger = logging.getLogger()


class GalaxeaCoTProcessor(FullProcessor):
    """
    VLM（视觉-语言-动作）训练 / 推理用的数据处理器，支持 Chain-of-Thought（CoT）样本模板。

    职责拆分（改代码前先确认该改哪一层）：
    - GalaxeaCoTProcessor（本类）：张量加工（图像、动作/状态变换与归一化、指令文本选择）
    - SamplesBuilder（samples_builder.py）：RoboVQA samples dict + 可组合的模板构造
    - 模型侧 InputPreprocessor：token 化（本类不参与）

    相比父类 FullProcessor 的扩展：
    - 两阶段 preprocess：_process_tensors() → SamplesBuilder.build()；
    - VLM 输入动作归一化（给模板文本里出现的动作数值单独一套归一化口径）；
    - Hardcode 模式（指令 / 本体状态掩码 / 动作掩码的固定化，用于调试与消融实验）。

    生命周期提醒：必须先 train()/eval() 并 set_normalizer_from_stats()，再调用 preprocess()。
    """

    def __init__(
        self,
        # ---------------- 维度契约与观测设置（透传父类）----------------
        shape_meta: Dict[str, Any],          # 每个 part 的 key / raw_shape / shape / camera_type
        num_obs_steps: int,                  # 每次取几帧观测（T），需与数据侧 obs_size、模型侧 cond_steps 一致
        num_output_cameras: int,             # 输出给模型的相机帧槽位总数（相机数 × 观测帧数）
        action_state_transforms: Optional[List[Any]],   # 可逆变换列表（相对动作、旋转表示……），无则 None
        # ---------------- 动作 & 状态归一化（透传父类）----------------
        use_stepwise_action_norm: bool,      # True：动作分块内逐时间步用各自的统计量；False：整个 chunk 共用
        norm_default_mode: NormMode,         # 所有 action/state 键的默认归一化模式
        action_state_merger,                 # 把按语义分组的 dict 对齐并合并成一条扁平向量（也负责反向还原）
        action_filter: BaseActionFilter,     # 动作过滤器：按维标记“有效操作维”，产出 action_op_mask
        # ---------------- 图像变换（透传父类）----------------
        train_transforms: Optional[Dict[str, List[Any]]],  # {相机 key: [变换, ...]}，训练用
        val_transforms: Optional[Dict[str, List[Any]]],    # {相机 key: [变换, ...]}，评估用
        # ---------------- 指令（语言）处理（透传父类）----------------
        drop_high_level_prob: float,         # 基类 augment_instruction 的“丢高层指令”概率；
                                             # 注意本类覆盖了该方法，所以此参数在本类里实际不生效（仅签名兼容）
        use_zh_instruction: bool,            # task 形如 "中文@English" 时：True 取中文、False 取英文
        # ---------------- token 化元信息（透传父类，本类本身不做 token 化）----------------
        pad_token_id: int,                   # padding token id（MixtureProcessor 会断言各子处理器一致）
        image_token_index: int,              # 图像占位 token id（与模型侧配置保持一致）
        max_text_tokens: int,                # 指令文本最大 token 数（真正截断在模型侧）
        num_input_cameras: int,              # 输入相机帧总数（= 相机数 × 观测帧数），用于对齐像素槽位
        # 相机尺寸配置：{camera_type: [H, W]}；用于按 camera_type 自动解析 shape 并注入 Resize
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        # 为兼容配置而接收，**本类不会传给父类、也不会用它加载 tokenizer**
        # （token 化由模型侧的 InputPreprocessor 完成）——因此 self.tokenizer 恒为 None。
        tokenizer_params: Optional[Dict[str, Any]] = None,
        # use_cot：为兼容配置而接收，本类不使用（是否启用 CoT 由 samples_builder 决定）
        use_cot: bool = False,
        # cot_steps：已废弃，忽略 —— 请改用配置里的 samples_builder._target_
        cot_steps: Optional[List[str]] = None,
        # 可选的逐键归一化覆盖：{action|state: {键名: 模式}}；None = 所有键都用 norm_default_mode
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # 是否按离散动作训练：决定动作是否走 tokenizer（详见下方 ConcatLeftAlign 兼容性检查）
        discrete_action: bool = True,
        # VLM 输入动作归一化（与训练动作归一化相互独立）：给模板文本里的动作数值单独一套口径
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # hardcode 模式（历史子类 GalaxeaCoTProcessorHardcode 的参数，现已合并到本类）
        hardcode_instruction: Optional[str] = None,       # 非 None：指令槽永远用这句固定文本
        hardcode_proprio_pad_zeros: bool = False,         # True：本体状态掩码全 0（= 所有维度都算有效）
        hardcode_action_pad_ones: bool = False,           # True：动作掩码全 1（= 所有 part 都当作 noop）
        # embodiment_type：机器人形态名（r1 / r1pro ...）。既可在此由配置写死，
        # 也可由 MixtureLerobotDataset.set_processor() 在运行时回写（多具身混训时走后者）。
        embodiment_type: Optional[str] = None,
        # dummy 模式的裁剪范围（仅当某个键的 norm_mode == "dummy" 时生效）
        dummy_clip_default: tuple = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, tuple]]] = None,
        # SamplesBuilder：由配置的 `_target_` 指定具体 Builder 类。
        # 默认 BaseSamplesBuilder（无 CoT）；可切换为 SubtaskCoTBuilder / MixedSamplesBuilder 等。
        samples_builder: Optional[Any] = None,
        # image_resize：历史预留参数，**本类当前完全不使用**（None = 不做任何事，行为保持不变）。
        # 图像缩放仍由 camera_size_config → 自动注入的 T.Resize 负责。
        image_resize: Optional[List[int]] = None,
    ):
        """构造处理器：先保存本类独有的开关，再把公共参数透传给 FullProcessor 链。

        参数按主题分为五组（语义与 base_processor.py 里的同名参数一致，这里只补充本类的差异点）：

        ① 维度契约与观测设置：shape_meta / num_obs_steps / num_output_cameras /
           action_state_transforms。
        ② 动作与状态流水线：use_stepwise_action_norm / norm_default_mode /
           norm_exception_mode / action_state_merger / action_filter。
        ③ 图像与语言：train_transforms / val_transforms / camera_size_config /
           num_input_cameras / use_zh_instruction；以及只影响基类方法的 drop_high_level_prob
           （本类覆盖了 augment_instruction，因此它在本类里不生效）。
        ④ token 化元信息：pad_token_id / image_token_index / max_text_tokens /
           tokenizer_params（⚠️ tokenizer_params 被刻意丢弃，本类不加载 tokenizer）。
        ⑤ 本类独有的开关：
             · VLM 输入动作归一化：vlm_input_action_norm_default_mode / _exception_mode；
             · 调试硬编码：hardcode_instruction / hardcode_proprio_pad_zeros /
               hardcode_action_pad_ones；
             · 形态标签：embodiment_type（通过 property setter 同步到 samples_builder）；
             · dummy 裁剪：dummy_clip_default / dummy_clip_exception；
             · 模板构建器：samples_builder（通常由配置写成 _partial_，这里再补运行时参数）；
             · 历史遗留且**不生效**的参数：use_cot / cot_steps / image_resize。

        副作用：
          会创建 self.samples_builder（模板构建器实例），并校验“离散动作 + 合并器”的兼容性。
        """
        # ---------------- 本类独有的状态：先存下来，供后面（及父类）使用 ----------------
        # pad_token_id / image_token_index：父类也会再存一份（值相同），这里先存是为了
        # 在 super().__init__ 之前就能用（并让 MixtureProcessor 的断言随时可读）。
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        # 离散动作开关：true = 动作会经离散 tokenizer（VQ/BAR）编码成 token。
        self.discrete_action = discrete_action
        # 形态名先存到私有字段：下面用 property setter 同步给 samples_builder 时，
        # 需要保证 samples_builder 已经存在（所以这里先写 _embodiment_type）。
        self._embodiment_type = embodiment_type

        # 只在 DEBUG 级别打一行：确认这条 processor 到底绑定了哪个具身
        # （多具身混训时，模板 / tokenizer 都依赖它，排查问题先看这行）。
        if self._embodiment_type is not None:
            logger.debug(f"[GalaxeaCoTProcessor] Embodiment type: {self._embodiment_type}")

        # ---------------- 把公共参数透传给父类 ----------------
        # 父类链会完成：shape_meta 解析与校验、Resize 自动注入、normalizer 占位、
        # tokenizer 元信息保存（num_input_images = num_input_cameras）等。
        # ⚠️ 注意这里**没有传 tokenizer_params**：本类不需要 tokenizer，self.tokenizer 保持 None。
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
            drop_high_level_prob=drop_high_level_prob,
            use_zh_instruction=use_zh_instruction,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
            max_text_tokens=max_text_tokens,
            num_input_cameras=num_input_cameras,
            camera_size_config=camera_size_config,
            vlm_input_action_norm_default_mode=vlm_input_action_norm_default_mode,
            vlm_input_action_norm_exception_mode=vlm_input_action_norm_exception_mode,
            dummy_clip_default=dummy_clip_default,
            dummy_clip_exception=dummy_clip_exception,
        )

        # ---------------- 静态校验：离散动作与合并器必须兼容 ----------------
        # 为什么必须检查：离散动作 tokenizer 按“part（部件）”切分动作向量，
        # 需要 merger 为每个 part 保留对齐后的维度布局与 padding 掩码；
        # ConcatLeftAlign 是“扁平左对齐拼接”，会抹掉 part 边界，使 tokenizer 无法还原分组。
        # 局部 import 的原因：避免模块顶层引入 transforms 依赖（也避免潜在的循环导入）。
        if self.discrete_action:
            from g05.data_processor.transforms.action_state_merger import ConcatLeftAlign

            if isinstance(self.action_state_merger, ConcatLeftAlign):
                # 报错信息（英文，保持仓库一致的异常文案）大意：
                #   "ConcatLeftAlign 与 discrete_action=True 不兼容。
                #    请改用 PaddingActionMerger —— 离散 tokenizer 需要逐 part 对齐的
                #    padding，而不是扁平拼接。"
                raise ValueError(
                    "ConcatLeftAlign is incompatible with discrete_action=True. "
                    "Use PaddingActionMerger instead — discrete tokenizers require "
                    "per-part aligned padding, not flat concatenation."
                )

        # ---------------- Hardcode 模式：指令固定 ----------------
        # 非 None 时，augment_instruction() 会无条件返回这句话（调试 / 消融实验用）。
        # 注意：另外两个 hardcode_* 开关是直接透传给 SamplesBuilder 的（见下方构造），
        # 因为“掩码怎么造”属于模板构建阶段的知识。
        self.hardcode_instruction = hardcode_instruction

        # ---------------- 汇总相机尺寸，供模板里的图像占位符使用 ----------------
        # shape_meta["images"] 的每一项形如：
        #   {"key": "head_rgb", "shape": [C, H, W], "camera_type": "exterior", ...}
        # 这里只取 (H, W)：模板占位符 <image{i}_image_!> 只需要尺寸信息，
        # 真正的像素张量在 _process_tensors 里走 sample["pixel_values"] 另一条路。
        # shape 已由父类按 camera_size_config 解析过，所以这里拿到的是**最终 resize 后**的尺寸。
        image_sizes = {
            meta["key"]: (meta["shape"][1], meta["shape"][2])
            for meta in (self.shape_meta.get("images") or [])
        }
        # 构建器类：配置给了 samples_builder 就用它，否则用默认的“无 CoT”基类。
        # config 里通常写成 `_target_: ...MixedSamplesBuilder` + `_partial_: true`
        # （见 docs/data/samples_builders_zh.md 第 8 节），Hydra 会先造出一个可调用对象，
        # 这里再补上“只有运行时才知道”的参数：
        #   num_input_images —— 模板里放几个 <image{i}> 占位符
        #   image_sizes      —— {相机 key: (H, W)}，键的字典序决定 <image0>/<image1> 对应哪路相机
        #   embodiment_type  —— 填进 <embodiment_text_!> 的形态名
        #   hardcode_*       —— 调试用的掩码覆盖开关
        # 注意：hardcode_instruction 不在这里透传（指令固定是在 augment_instruction 里做的），
        # 该形参留给“直接实例化 Builder”的场景（冒烟测试、离线脚本等）。
        builder_cls = samples_builder or BaseSamplesBuilder
        self.samples_builder = builder_cls(
            num_input_images=self.num_input_images,
            image_sizes=image_sizes,
            embodiment_type=embodiment_type,
            hardcode_proprio_pad_zeros=hardcode_proprio_pad_zeros,
            hardcode_action_pad_ones=hardcode_action_pad_ones,
        )

    # ------------------------------------------------------------------ #
    #  train / eval —— 把模式同步给 samples_builder                        #
    # ------------------------------------------------------------------ #

    def train(self):
        """切到训练模式：父类切换图像变换栈，本类顺带通知模板构建器。

        为什么要通知 Builder：像 MixedSamplesBuilder 这样的“多格式采样器”需要知道
        当前是训练还是推理 —— 训练时按权重随机抽一条 CoT 模板，推理时固定用 eval_builder。
        没实现 set_training 的 Builder（大多数）用 hasattr 跳过即可，互不影响。
        """
        super().train()
        if hasattr(self.samples_builder, "set_training"):
            self.samples_builder.set_training(True)

    def eval(self):
        """切到评估模式：父类切换图像变换栈，本类顺带通知模板构建器（语义同 train()）。"""
        super().eval()
        if hasattr(self.samples_builder, "set_training"):
            self.samples_builder.set_training(False)

    # ------------------------------------------------------------------ #
    #  embodiment_type 属性 —— 改动时自动同步给 samples_builder            #
    # ------------------------------------------------------------------ #

    @property
    def embodiment_type(self) -> Optional[str]:
        """当前绑定的机器人形态名（r1 / r1pro ...），未设置时为 None。

        调用方：MixtureLerobotDataset.set_processor() 会写 `p.embodiment_type = emb_type`，
        从而让模板里的 <embodiment_text_!> 填到正确的形态名。
        """
        return self._embodiment_type

    @embodiment_type.setter
    def embodiment_type(self, value: Optional[str]):
        """写入形态名，并同步给 samples_builder（模板渲染时要用）。

        用 hasattr 做保护：__init__ 里构造 builder 之前也可能被赋值，
        此时只需更新私有字段，等 builder 建好时会在构造函数里拿到该值。
        """
        self._embodiment_type = value
        if hasattr(self, "samples_builder"):
            self.samples_builder.embodiment_type = value

    # ------------------------------------------------------------------ #
    #  指令覆盖（本类覆盖了基类实现）                                       #
    # ------------------------------------------------------------------ #

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """产出“这一帧要用哪句指令”——配了 hardcode_instruction 就固定用它，否则用 data["task"]。

        与基类 BaseProcessor.augment_instruction 的差别（重要）：
          · 基类：会把 coarse_task（高层）与 task（低层）拼成
                  "[High]: ..., [Low]: ..."，并以 drop_high_level_prob 的概率丢掉高层；
          · 本类：直接返回**单句** task，不做高低层拼接，也不用 drop_high_level_prob。
            原因是 Galaxea 的 CoT 数据里 task 已是细粒度指令，再拼 [High]/[Low] 反而会把
            高层语义重复注入到模板的指令槽里。
        另外：本方法不做中英文选择 —— 那一步已在 _process_tensors() 开头对 data["task"]
              做过（"中文@English" → 取其中一边），所以这里拿到的已经是目标语言文本。

        参数：
            data: 数据集样本（原始字段字典）；必须含 "task"（除非配了 hardcode_instruction）。

        返回：
            指令字符串。类型标注里的 List[str] 是沿用基类签名的历史写法，
            实际返回 str（它会被 SamplesBuilder 填进 <command_text_!_200> 槽）。
        """
        # hardcode_instruction 优先级最高：调试 / 消融时让模型只看到同一句指令。
        if self.hardcode_instruction is not None:
            return self.hardcode_instruction
        # 常规路径：直接取（已做过语言选择的）task 文本。
        return data["task"]

    # ------------------------------------------------------------------ #
    #  阶段 1：张量加工                                                    #
    # ------------------------------------------------------------------ #

    def _process_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """把原始样本加工成张量样本，但**不**构建 RoboVQA samples（那属于阶段 2）。

        处理顺序（顺序不可随意调整，改动会直接影响归一化统计量的含义）：
          1) 双语文本字段规范化（"中文@English" → 按 use_zh_instruction 取其中一边）；
          2) 指令：augment_instruction()；
          3) 图像：process_images() → build_pixel_values()；
          4) gt_action：过滤 / 变换 / 归一化之前的“合并后原始动作”深拷贝；
          5) 动作 / 状态流水线：preprocess_action_state()（含可选的 vlm_action 通道）；
          6) 组装输出字段与元信息。

        参数：
            data: 数据集 __getitem__ 产出的原始样本（LeRobot / mcap 语义）。

        返回：
            样本字典，除常规张量字段外还携带两个**临时字段**（由 SamplesBuilder 消费）：
            - "_instructions"：指令文本，供 SamplesBuilder 填 command 槽
            - "_vlm_action"  ：VLM 视角的动作副本（可能为 None）

        注意：
            · 双语规范化与动作/状态流水线都是**原地修改**传入的 dict（transforms /
              normalizer / merger 的实现都返回同一个对象），因此阶段 2 拿到的 data 里，
              action/state 已是“合并 + 归一化后”的形式，而 task/atomic_task 等文本字段
              保持原样（只做了语言选择）。
        """
        sample = {}

        # ---------------- 步骤 0：双语文本字段的语言选择 ----------------
        # Galaxea 的数据约定用 "中文@English" 同时存两种语言（见 configs/data/*.yaml）。
        # 这里统一处理 5 个可能出现在模板里的文本字段，避免各 Builder 各切一次。
        # 用 split("@", 1) 只切第一个 "@"：文本内部若含 "@" 也不会被切坏。
        # 注意：这里是**在原 dict 上原地赋值**，所以阶段 2 的 SamplesBuilder 看到的也是目标语言文本。
        for key in ("task", "atomic_task", "future_task", "high_level_instruction", "action_hint"):
            if key in data and isinstance(data[key], str) and "@" in data[key]:
                zh, eng = data[key].split("@", 1)
                data[key] = zh if self.use_zh_instruction else eng

        # ---------------- 步骤 1：指令 ----------------
        # 本类覆盖了 augment_instruction（返回单句 task 或 hardcode 文本），
        # 结果暂存到临时字段，等阶段 2 由 SamplesBuilder 填进模板的指令槽并 pop 掉。
        instructions = self.augment_instruction(data)
        sample["_instructions"] = instructions

        # ---------------- 步骤 2：图像 ----------------
        # process_images()：逐相机做变换（train/val 两套变换栈），并按 camera_type 保留为 dict，
        #                   从而支持不同相机使用不同分辨率；
        # build_pixel_values()：把“相机帧总数”对齐到 num_output_cameras
        #                   （多分辨率时必须严格相等；同分辨率下可补零 / 选帧，细节见父类）。
        pixel_values = self.process_images(data)
        sample["pixel_values"] = self.build_pixel_values(pixel_values)

        # ---------------- 步骤 3：原始动作副本（开环评估用）----------------
        # 在 action_filter / transform / normalizer 之前，先用 merger 只做“合并 + padding”
        # 得到一条扁平的动作向量并深拷贝下来，保持物理量纲不变。
        # 用途：开环评估做对照；以及训练首个 batch 的 tokenizer 评估
        #       （train_utils.py 优先用 vlm_action，取不到才回退到 gt_action）。
        # deepcopy 很关键：后续归一化会原地改写 data，不拷贝就会把对照数据一起改掉。
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # ---------------- 步骤 4：动作 / 状态流水线 ----------------
        # 父类实现：action_filter → action_state_transform（含 NaN 体检）
        #           → normalizer → action_state_merger；
        # 若配置了 vlm_input_action_norm_default_mode，还会额外产出 data["vlm_action"]。
        data = self.preprocess_action_state(data)

        # VLM 输入动作：一份**独立归一化**的动作副本，专门用于填进模板文本里的动作数值。
        # 详见 base_processor.py 的 preprocess_action_state()：
        # “喂给 VLM 上下文的动作”与“模型要预测 / tokenizer 要训练的动作”可以用不同的
        # 归一化口径（配置项 vlm_input_action_norm_*）。
        # 若未配置该通道则取到 None，此时 SamplesBuilder 会回退使用训练动作 sample["action"]。
        sample["_vlm_action"] = data.pop("vlm_action", None)

        # ---------------- 步骤 5：组装输出字段 ----------------
        # build_action_sample() 负责把 action / action_is_pad / action_dim_is_pad /
        # action_parts_meta / action_op_mask / proprio / proprio_is_pad /
        # proprio_dim_is_pad / frequency 等字段搬进 sample。
        sample = self.build_action_sample(data, sample)
        # 样本下标（下游按它做调试定位、坏样本统计）。
        sample["idx"] = data["idx"]
        # 透传元信息：排查“哪个具身 / 哪个数据集 / 哪句指令出的问题”时非常有用；
        # collate 阶段对 str 类型字段有专门处理（不会被 stack 成张量）。
        for meta_key in ("task", "embodiment", "dataset_locator"):
            if meta_key in data:
                sample[meta_key] = data[meta_key]

        return sample

    # ------------------------------------------------------------------ #
    #  主入口                                                             #
    # ------------------------------------------------------------------ #

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        把数据集样本加工成“可直接 collate 的样本字典”（策略模型的输入）。

        两阶段流水线：
        1. _process_tensors()：动作/状态变换与归一化、图像变换与像素张量组装，
           并留下 _instructions / _vlm_action 两个临时字段；
        2. samples_builder.build()：RoboVQA samples dict + 可组合的模板构造
           （决定这一帧用哪种 CoT 格式、指令填哪个槽、CoT 文本放哪里）。

        为什么拆成两段：张量加工与“文本模板格式”是两件独立的事。
        张量加工对所有数据一致，而模板格式随数据集标注能力变化
        （有的只有 task，有的还有 atomic_task / bbox / memory ...），
        因此把后者交给可替换的 SamplesBuilder，用配置切换而不必改处理器代码。

        参数：
            data: Dict[str, Any]，数据集 __getitem__ 取出的原始样本（LeRobot 语义）：
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int，样本下标

        返回：
            sample: Dict[str, Any]，可直接 collate：
                - "pixel_values": Dict[camera_type, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                                  （由父类的图像链路透传，见 base_processor.py）
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "proprio_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action": Optional，输入动作的深拷贝，供开环评估做对照
                - "action_op_mask": Optional, torch.BoolTensor -> [action_dim]
                - "samples": RoboVQA 格式 dict（模板 + 各槽位文本），供模型侧多模态处理
                - "idx": int，样本下标
        """
        # 阶段 1：张量加工（图像 / 动作 / 状态 / 指令），并留下两个临时字段。
        sample = self._process_tensors(data)

        # 阶段 2：RoboVQA 样本构造（委托给 SamplesBuilder）。
        # 注意这里传的是**原始 data**（它已被阶段 1 原地改写：action/state 已归一化 + 合并），
        # Builder 会从 data 里读 CoT 标注字段（task / atomic_task / bbox ...），
        # 并从 sample 里读张量字段、pop 掉 _instructions / _vlm_action。
        sample["samples"] = self.samples_builder.build(data, sample)

        return sample
