# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/processor/samples_builder.py — 训练样本模板（CoT）构建器
# =============================================================================
#
# 【这个文件是什么】
#   它是“一帧数据 → 一条模型输入样本”的组装规则集合。
#   数据集（LeRobotDataset.__getitem__）给出的是**原始字段**：
#       data["task"] / data["atomic_task"] / data["bbox"] / data["action_hint"] / data["action"] ...
#   模型侧（InputPreprocessor）需要的是**带占位符的模板 + 每个占位符要填的值**：
#       samples["template"]    = "<image0_image_!>...Embodiment: <embodiment_text_!>; Task: ..."
#       samples["command"]     = "把毛巾叠好"             （填进 <command_text_!_200>）
#       samples["atomic_task"] = "Subtask: 抓起蓝色毛巾"   （填进 <atomic_task_text>）
#   本文件就是中间这层“翻译官”：把前者组装成后者。
#   一个 Builder 子类 = 一种模板格式（也就是一种 CoT 输出格式 / 一种指令注入策略）；
#   换模板只要在 YAML 里换 `samples_builder._target_`，不需要改代码。
#
# 【它在整条数据链路中的位置】
#
#   LeRobotDataset.__getitem__                src/g05/data/lerobot/lerobot_dataset_v3.py
#     │  · 读 parquet 数值列 + mp4 画面；把 *_index 翻回文本（task/atomic_task/bbox/memory/plan ...）
#     ▼
#   GalaxeaCoTProcessor.preprocess            src/g05/data_processor/processor/galaxea_cot_processor.py
#     │  ① _process_tensors()：图像张量化、动作/状态归一化、掩码，并准备 _vlm_action 副本
#     │  ② SamplesBuilder.build(data, sample)      ← 本文件
#     ▼
#   samples dict（template + command + proprio + action + 各 CoT slot 文本）
#     │
#     ▼
#   InputPreprocessor                         src/g05/models/g05/io/input_preprocessor.py
#     │  · _parse_template()：把模板切成“静态文本段 / 动态占位符段 / 控制标签”
#     │  · 按占位符去 samples 里取值、逐段 token 化，拼成 input_ids / labels / attention_mask
#     ▼
#   模型前向（训练：算 loss；推理：自回归生成 CoT 文本 → 再生成/解码动作 token）
#
# 【必须先看懂的占位符语法（本文件的核心）】
#   模板字符串里的 <...> 都是占位符，由 InputPreprocessor._parse_template() 解释：
#
#       <sampleKey_processorKey>       取 samples["sampleKey"]，交给 processorKey 对应的处理器，
#                                      **参与 loss**（这就是模型要学会输出的内容）
#       <sampleKey_processorKey_!>     同上，但 **不参与 loss**（只作为条件输入，label 置为忽略）
#       <sampleKey_processorKey_!_N>   不参与 loss，且最多 N 个 token（超出截断并告警）
#       <sampleKey_processorKey_N>     参与 loss，且最多 N 个 token
#       <EOC> / <EOV>                  控制标签（切分用，见下），本身不产生 token
#       <bos> / <eos> / <chat_*>       由 SpecialTokenManager.resolve_template() 按模型类型替换
#
#   例：`<command_text_!_200>` = 取 samples["command"]，用 text 处理器分词，**不算 loss**
#       （指令是“条件”而不是“答案”），最多 200 个 token；
#       `<atomic_task_text>`    = 取 samples["atomic_task"]，**算 loss**（这是模型要学的 CoT）。
#
#   处理器（processorKey）常见取值：text（文本）/ proprio（本体状态）/ action（动作）/ image（图像）。
#   占位符里的 sampleKey 就是 samples 的键名，所以 `_populate_extra_samples()` 写进去的键
#   必须与模板里的 `<key>_text` 一一对应，否则 InputPreprocessor 会抛 KeyError。
#
# 【两个控制标签的含义（模型侧的切分规则）】
#   <EOC> = End Of Context（“上下文到此为止”）
#       它把模板切成两半：EOC 之前是给模型看的条件（推理时照常输入），
#       EOC 之后是 CoT 目标文本（推理时**被丢掉**，由模型自己生成）。
#       所以“要模型学会输出的 CoT 文本”必须写在 <EOC> 之后。
#       另外：EOC 之后的静态文本会被自动转成“可学习片段”，也会计入 loss。
#   <EOV> = 推理链与动作的分界（“文本到此为止，接下来是动作”）
#       训练时：EOV 之前为 prefix、之后为 suffix，分别编码后拼接（encode_train）；
#       推理时：模型先生成 EOV 之前的文本（含 CoT），到 EOV 后转入动作解码。
#       所以模板尾部总是：`Action: <EOV><action_action>|<eos>`。
#
# 【所有 Builder 的差别其实只有三点】
#   1) `template` 属性            —— CoT 区段写什么、指令槽填什么；
#   2) `_populate_extra_samples()` —— 往 samples 里塞模板占位符需要的键；
#   3) `_override_command()`       —— 可选：覆盖指令槽文本
#      （例如改用 atomic_task / high_level_instruction 顶掉原本的 task）。
#   另外还有两个字段声明，决定“这条模板能被哪些数据命中”：
#   · `required_fields`      —— 训练路径判据（can_handle）
#   · `eval_required_fields` —— 推理路径判据（can_handle_for_eval）
#
# 【新增一个 Builder 只要 3 步】
#
#   1) 继承 BaseSamplesBuilder，把 `template` 属性写成一个完整模板字符串：
#
#      class MyBuilder(BaseSamplesBuilder):
#          @property
#          def template(self):
#              return (
#                  f"{self._images}<bos>Embodiment: <embodiment_text_!>; "
#                  f"Task: <command_text_!_200> MyField: <myfield_text> State: <proprio_proprio_!>;\n"
#                  f"Action: <EOV><EOC><action_action>|<eos>"
#              )
#
#   2) 实现 `_populate_extra_samples(data, samples)`，把数据搬进 samples：
#
#      def _populate_extra_samples(self, data, samples):
#          samples["myfield"] = data["my_raw_field"]
#
#   3) 在配置里用 `_target_` 选中它：
#
#      processor:
#        samples_builder:
#          _target_: g05.data_processor.processor.samples_builder.MyBuilder
#
#   `build()` 会自动把 command / proprio / action / embodiment / image 填好，
#   只有“这个 Builder 新引入的字段”需要自己在 `_populate_extra_samples` 里处理。
#
# 【train / eval 两条路径（以 MixedSamplesBuilder 为例）】
#   · 训练：在所有 `can_handle(data) == True` 的候选里按权重随机抽一个
#     ——“这条数据有什么标注，就练什么 CoT”；一个都不匹配就退回 BaseSamplesBuilder（无 CoT）。
#   · 推理：固定用 eval_builder，判据是 `can_handle_for_eval(data)`
#     —— CoT 目标由模型自己生成、不需要标注，所以只看推理时真正要喂进去的输入字段。
#     没配 eval_builder 同样退回 BaseSamplesBuilder。
#
# 【⚠️ 新手最容易踩的 4 个坑】
#   坑 1：模板里加了占位符，却没在 `_populate_extra_samples` 里写对应键
#        → InputPreprocessor 抛
#          `KeyError: Template placeholder <xxx_text> is missing from the sample.`。
#          记住：模板里每个 `<a_b>` 都要求 samples 里有键 "a"。
#   坑 2：把 CoT 文本写在了 <EOC> 之前
#        → 这段文本被当成“条件输入”（推理时依然喂给模型），模型永远学不会生成它。
#   坑 3：FM-only（`batchify_action=false`，动作不走离散 tokenizer）时仍用带
#        `<action_action>` 的模板
#        → 动作字典被 `str()` 成 Python repr 当作标签训练，模型会学着输出
#          `"[ 1.1111, -0.6111, ..."` 这类张量字面量。此时应改用
#          `SubtaskCoTBuilderFMOnly`（模板去掉 `<action_action>`），详见该类 docstring。
#   坑 4：标注里的脏值（None / NaN / -1 / "null" / "none" / "nan" / "" / 空 JSON `{}`）
#        被误认为“有标注”，CoT 标签变成空壳（例如 `"Subtask: "`）。
#        判据统一收敛在 `_check_fields()`，新 Builder 请复用它。
#
# 【自测（冒烟测试）】
#   python -m g05.data_processor.processor.samples_builder
#   会打印每个 Builder 在 PaliGemma / Qwen3.5-base / Qwen3.5-instruct 三种 token map 下的
#   模板渲染结果，并跑一批断言（含 “Slot-content Semantics” 回归断言，
#   专门防“atomic_task 槽被错填成 task 文本”这类隐性 bug）。
#   字段来源与数据覆盖范围见 docs/data/samples_builders_zh.md。
#
# 【相关文件与文档】
#   src/g05/data_processor/processor/galaxea_cot_processor.py  调用方（preprocess 第二阶段）
#   src/g05/models/g05/io/input_preprocessor.py                _parse_template / 占位符解析
#   src/g05/utils/common/special_tokens.py                     <bos>/<eos>/<chat_*> 的模型相关替换
#   src/g05/data/lerobot/lerobot_dataset_v3.py                 data 里各 CoT 字段的来源与解码
#   src/g05/data_processor/processor/base_processor.py         动作/状态归一化、_vlm_action 的来源
#   docs/data/samples_builders_zh.md                           全部 Builder 的字段依赖与模板速查
#   docs/data/schema_zh.md                                     数据字段来源（parquet 列 → item 字段）
# =============================================================================

from typing import Dict, Any, List, Optional
import logging

import torch

# 顶层 logger：本模块只在 MixedSamplesBuilder 里打 debug 日志
# （记录“这一帧最终选了哪个 Builder”），排查“为什么 CoT 没生效”时很有用。
logger = logging.getLogger(__name__)


class BaseSamplesBuilder:
    """最小 VLA 模板：没有 CoT、没有额外区段（“只看指令与状态，直接预测动作”）。

    它既是所有其它 Builder 的基类，也是兜底默认值
    （config 里不写 processor.samples_builder 时，GalaxeaCoTProcessor 就用它）。

    ── 模板渲染出来长这样（embodiment=r1、2 个相机、离散动作）──────────────
        PaliGemma（图像 token 由模型侧按图像数量展开，模板里只写占位符）：
            <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
            Action: <EOV><EOC><action_action>|<eos>
        Qwen3.5-instruct:
            <|im_start|>user\\n<image0_image_!><image1_image_!>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;<|im_end|>\\n<|im_start|>robot\\n
            Action: <EOV><EOC><action_action>|<|endoftext|>

    ── 每个占位符最终填什么值 ──────────────────────────────────────────────
      <image{i}_image_!>     samples[f"image{i}"]：本 Builder 只放 (H, W) 尺寸；
                             图像像素另走 sample["pixel_values"]。占位符的作用是告诉
                             InputPreprocessor“这个位置要插入第 i 路相机的图像 token”。
      <embodiment_text_!>    samples["embodiment"]（机器人形态名，如 r1 / r1pro）
      <command_text_!_200>   samples["command"]（任务指令，最多 200 个 token）
      <proprio_proprio_!>    samples["proprio"]["value"]（本体状态 / 关节量）
      <action_action>        samples["action"]["value"]（动作，**参与 loss**，这才是训练目标）
      规律：带 `_!` 的都是“条件输入”，不参与 loss；只有动作（以及 EOC 之后的 CoT 文本）
      才是模型要学会输出的“答案”。

    ── 五个特殊字符串由谁替换 ─────────────────────────────────────────────
      <bos> / <eos> / <chat_user_prefix> / <chat_user_suffix> / <chat_assistant_prefix>
      由 InputPreprocessor._parse_template() → SpecialTokenManager.resolve_template()
      按模型类型替换（旧注释里的 SpecialTokenMap.resolve() 就是同一个功能）：
        · PaliGemma: <bos>=<bos>、<eos>=<eos>、<chat_user_suffix> 换成一个换行，其余为空串
        · Qwen3.5-instruct: <bos>=""、<eos>=<|endoftext|>，对话格式由 chat_* 注入
      所以写模板时不必关心模型差异，交给占位符就行。
    """

    # 子类声明“我这条模板需要哪些标注字段”。can_handle() 会逐个检查这些字段存在且非空
    # （判据见 _check_fields：None / NaN / -1 / "null" / "none" / "nan" / "" 都算“没有”）；
    # 任一字段不满足就跳过本 Builder（MixedSamplesBuilder 会去挑别的候选）。
    required_fields: tuple = ()
    # 推理（eval）时**真正需要**的字段集合：EOC 之后的 CoT 目标是模型自己生成的，
    # 不需要标注，所以这里通常比 required_fields 少。
    # MixedSamplesBuilder 在推理路径上用它挑 eval_builder（见 can_handle_for_eval）。
    eval_required_fields: tuple = ()

    def __init__(
        self,
        num_input_images: int,                       # 模板里放几个 <image{i}_image_!> 占位符
        image_sizes: Dict[str, Any],                 # {相机 key: (H, W)}，来自 shape_meta
        embodiment_type: Optional[str] = None,       # 形态名（r1 / r1pro / ...），填进 embodiment 槽
        # ↓ 以下三个是“硬编码 / 调参调试”开关；当前仓库 configs 都没有开启，
        #   即保持默认值，行为与上游一致。
        hardcode_proprio_pad_zeros: bool = False,    # True：本体状态掩码全 0（= 所有维度都算有效）
        hardcode_action_pad_ones: bool = False,      # True：动作掩码全 1（= 所有 part 都当作 noop）
        hardcode_instruction: Optional[str] = None,  # 非空：指令槽永远用这句固定文本
    ):
        # 至少要有一路相机：否则模板里的图像占位符无从取值。
        assert len(image_sizes) > 0, "image_sizes must be non-empty."
        # {相机 key: (H, W)} —— 每个相机 resize 后的目标尺寸，来自数据集的 shape_meta。
        # 历史帧（num_obs_steps > 1）时：同一个相机 key 在所有观测步上共用同一尺寸，
        # 下标按 i % num_cameras 回环取（见 build() 末尾的循环）。
        self._image_sizes: Dict[str, Any] = dict(image_sizes)
        # 相机 key 的固定顺序：字典序决定 <image0>/<image1> 分别对应哪一路相机。
        self._image_keys: List[str] = list(image_sizes.keys())
        self.num_input_images = num_input_images
        self._embodiment_type = embodiment_type
        self.hardcode_proprio_pad_zeros = hardcode_proprio_pad_zeros
        self.hardcode_action_pad_ones = hardcode_action_pad_ones
        # 注意：当前 GalaxeaCoTProcessor 只透传 hardcode_proprio_pad_zeros /
        # hardcode_action_pad_ones，并没有传 hardcode_instruction
        # （指令固定是在 processor.augment_instruction() 里做的）。
        # 这个形参留给“直接实例化 Builder”的场景（冒烟测试、离线脚本等）。
        self.hardcode_instruction = hardcode_instruction

        # 图像占位符串，例如 num_input_images=2 → "<image0_image_!><image1_image_!>"，
        # 供子类在 template 里直接 f-string 拼进去。
        self._images = "".join(f"<image{i}_image_!>" for i in range(num_input_images))

    # 形态名（r1 / r1pro / ...）。注意 MixedSamplesBuilder 覆盖了 setter：
    # 一旦通过它设置，会把值同步给所有候选 Builder 与 eval_builder（见该类的实现）。
    @property
    def embodiment_type(self) -> Optional[str]:
        return self._embodiment_type

    @embodiment_type.setter
    def embodiment_type(self, value: Optional[str]) -> None:
        self._embodiment_type = value

    # “字符串形态的无效值”黑名单，三种来源：
    #   · tasks 表（LeRobot 的 meta/tasks.parquet）里表示“空”的占位串：null / none / nan
    #   · pandas 把缺失值写成字符串后的产物（"nan" / "None" / "" ...）
    #   · 少数数据集用 "-1" 表示“无标注”
    # 比较前统一做 strip + 小写，避免各 Builder 各写一套判断。
    _INVALID_STRINGS: frozenset = frozenset({"", "null", "none", "nan", "-1"})

    def _check_fields(self, data: Dict[str, Any], fields: tuple) -> bool:
        # 局部 import math：模块顶层只保留 typing / logging / torch，
        # 只有真正走到校验逻辑时才需要 math，保持导入轻量。
        import math

        for k in fields:
            v = data.get(k)
            # 字段缺失，或数据集显式给了 None。
            if v is None:
                return False
            # float('nan')：pandas 读出来的缺失值。注意 NaN != NaN，不能用 == 判断。
            if isinstance(v, float) and math.isnan(v):
                return False
            # -1：部分数据集约定的“无标注”整数值。
            if isinstance(v, int) and v == -1:
                return False
            # 字符串脏值："" / "null" / "none" / "nan" / "-1"（大小写与前后空格都不敏感）。
            # 注意：这里只过滤“整串就是脏值”，空 JSON "{}" 由子类自己再加严
            # （见 BBoxCoTBuilder.can_handle）。
            if isinstance(v, str) and v.strip().lower() in self._INVALID_STRINGS:
                return False
        return True

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """这条样本是否具备本 Builder 所需的全部**标注**字段（训练路径判据）。"""
        return self._check_fields(data, self.required_fields)

    def can_handle_for_eval(self, data: Dict[str, Any]) -> bool:
        """这条样本是否具备推理时**真正需要**的输入字段（eval 路径判据）。

        EOC 之后的 CoT 目标字段是模型在自回归推理时自己生成的，所以这里不检查它们。
        这样即便某些数据集没有 CoT 标注，也不会影响它在推理时匹配到正确的 eval builder。
        """
        return self._check_fields(data, self.eval_required_fields)

    @property
    def template(self) -> str:
        """完整模板；子类覆盖这个属性即可（基类给的是最小 VLA 模板）。

        解析前会先做一次特殊字符串替换（SpecialTokenManager.resolve_template()）：
        <bos>/<eos>/<chat_user_prefix>/<chat_user_suffix>/<chat_assistant_prefix>
        按模型类型换成各自的真实 token：
            · PaliGemma: <bos>=<bos>、<eos>=<eos>、<chat_user_suffix> 换成一个换行，其余为空串
            · Qwen3.5-instruct: <bos>=""、<eos>=<|endoftext|>，对话格式由 chat_* 注入
        """
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "Action: <EOV><EOC><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        """子类钩子：把本 Builder 新引入的字段搬进 samples。

        基类没有额外字段，所以这里什么都不做。子类写法示例：
            samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"
        写进去的键名必须与模板里的 `<键名>_text` 占位符一致，否则会 KeyError（文件头坑 1）。
        """
        pass

    def _override_command(self, data: Dict[str, Any]) -> Optional[str]:
        """子类钩子：返回非 None 就替换指令槽文本，返回 None 表示不干预。

        指令槽 = 模板里 <command_text_!_200> 实际填进去的值。
        默认返回 None，即保留 sample["_instructions"]（一般是 data["task"]，
        也可能已被 processor 的 hardcode_instruction 覆盖）。
        典型用法：AtomicTaskBaseSamplesBuilder 返回 atomic_task；
        HighLevelAtomicTaskCoTBuilder 返回 high_level_instruction。
        """
        return None

    def build(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        """通用组装流程；子类一般**不需要**重写它。

        输入：
            data   —— 数据集这一帧的原始字段（task / atomic_task / bbox / action / idx ...）
            sample —— GalaxeaCoTProcessor 已经处理好的张量样本
                      （含两个下划线开头的临时字段：_instructions / _vlm_action）
        输出：
            samples —— 交给 InputPreprocessor 的字典：template + command + proprio + action
                       + 各 Builder 自己塞进去的 CoT 文本槽

        子类只需要重写三个钩子：template / _populate_extra_samples / _override_command。
        """
        # sample 里两个下划线开头的字段是 GalaxeaCoTProcessor 临时挂上来的：
        #   _instructions —— 指令文本（默认注入 command 槽），这里 pop 出来，
        #                    因为 samples 里不允许残留它（会污染下游校验/拼装）。
        #   _vlm_action   —— 只给 VLM 看的动作副本，可能为 None（见下方说明）。
        instructions = sample.pop("_instructions")
        # 硬编码指令优先级最高：配了 hardcode_instruction 就无条件覆盖指令槽。
        if self.hardcode_instruction is not None:
            instructions = self.hardcode_instruction
        # 子类钩子（例如用 atomic_task / high_level_instruction 顶掉 task）。
        cmd_override = self._override_command(data)
        if cmd_override is not None:
            instructions = cmd_override
        # VLM 输入动作：一份“独立归一化”的动作副本，优先填进模板的 <action_action> 槽；
        # 为 None 时回退到训练动作 sample["action"]。
        # 为什么要两份动作？见 base_processor.py 的 preprocess_action_state()：
        # 让“喂给 VLM 上下文的动作”与“模型要预测/tokenizer 要训练的动作”可以用不同
        # 归一化口径（vlm_input_action_norm_* 配置项）。
        vlm_action = sample.pop("_vlm_action")

        samples: Dict[str, Any] = {}
        tpl = self.template

        # 先让子类填自己的额外字段（例如 samples["atomic_task"] = f"Subtask: ..."），
        # 因为 template 里的占位符需要它们存在。
        self._populate_extra_samples(data, samples)

        # 模板本体 + 指令文本。
        samples["template"] = tpl
        samples["command"] = instructions

        # 本体状态：value 是 [proprio_dim] 张量，proprio_dim_is_pad 标记“哪些维度是凑数补出来的”。
        # 跨形态合并（MixtureLerobotDataset）时，某些维度是别的具身才有的，会被 pad 出来，
        # 编码/送入 MLP 前会按这个掩码丢掉或置零（见 ProprioEncoder / proprio_helper.py）。
        proprio_pad_mask = (
            torch.zeros_like(sample["proprio_dim_is_pad"])
            if self.hardcode_proprio_pad_zeros
            else sample["proprio_dim_is_pad"]
        )
        samples["proprio"] = {
            "value": sample["proprio"],
            "proprio_dim_is_pad": proprio_pad_mask,
        }

        # 形态名填进 <embodiment_text_!>；没设置就写 "unknown"。
        samples["embodiment"] = (
            self.embodiment_type if self.embodiment_type is not None else "unknown"
        )

        # 只有当模板里真的有 <action_action> 占位符、并且样本里有动作时才填。
        # （FM-only 的模板故意不带这个占位符，于是动作完全交给 FM 头，不走 tokenizer。）
        if "<action_action" in tpl and "action" in sample:
            # action_dim_is_pad：True = 该维是 padding（跨形态补齐出来的维度）。
            # 动作 tokenizer 用它在解码时判定“这个 part 在当前形态下是 noop”，
            # 全 1 就等于“所有 part 都是 noop”（hardcode_action_pad_ones 调试模式）。
            action_pad_mask = (
                torch.ones_like(sample["action_dim_is_pad"])
                if self.hardcode_action_pad_ones
                else sample["action_dim_is_pad"]
            )
            samples["action"] = {
                # 优先用 VLM 动作副本，没有就用训练动作。
                "value": vlm_action if vlm_action is not None else sample["action"],
                "action_dim_is_pad": action_pad_mask,
                # action_op_mask：哪些维度/part 在本帧真的有操作（全 False 的 part 视为 noop）。
                "action_op_mask": sample.get("action_op_mask"),
                # parts_meta：{part 名: 维度数}，动作 tokenizer 用它把动作向量切回各个 part
                # （来源见 base_processor.py：action_state_merger.max_action_shape_meta）。
                "parts_meta": sample.get("action_parts_meta"),
            }

        # 控制频率（Hz）：动作 tokenizer 编码/解码时需要（多频率数据用它区分时间尺度）。
        if "frequency" in data:
            samples["frequency"] = data["frequency"]

        # 逐路相机的尺寸：只有模板里出现了 <image{i}_image 占位符才需要填。
        # 注意填进去的只是 (H, W) 尺寸（InputPreprocessor 的图像处理器要求 data=(H, W)），
        # 真正的像素在 sample["pixel_values"] 里。
        num_cameras = len(self._image_keys)
        for i in range(self.num_input_images):
            if f"<image{i}_image" in tpl:
                # 历史帧回环：相机数少于图像占位符数时，i % num_cameras 复用同一路相机。
                cam_key = self._image_keys[i % num_cameras]
                samples[f"image{i}"] = self._image_sizes[cam_key]

        return samples


class BaseActionSamplesBuilderFMOnly(BaseSamplesBuilder):
    """FM-only 的最小 VLA 模板：没有 CoT、没有额外区段，而且**不带动作 token 槽**。

    与 BaseSamplesBuilder 的唯一区别：模板里没有 `<action_action>`，即
    动作完全不进 tokenizer，只由 FM（连续动作）分支回归。
    适合 `batchify_action=false` 的 FM-only 训练（详见 SubtaskCoTBuilderFMOnly
    里对“为什么必须去掉 <action_action>”的完整解释）。

    模板（渲染后）：
        <image0_image_!><image1_image_!><bos>Embodiment: ...; Task: ... State: <proprio>;
        Action: <EOV><EOC><eos>
    """

    @property
    def template(self) -> str:
        # 注意这里没有用 <chat_user_prefix> 等占位符（历史写法，模板自带换行）。
        return (
            f"{self._images}<bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;\n"
            f"Action: <EOV><EOC><eos>"
        )


class MemorySamplesBuilder(BaseSamplesBuilder):
    """带“记忆输入”的 VLA 模板：memory 只作为**条件输入**（不参与 loss）。

    data["memory"] 来自 prev_memory_index（上一帧/历史的记忆文本，由 lerobot_dataset_v3.py 解码）。
    模板里用 `<memory_text_!>`，注意那个 `_!`：它标记“这段是输入、不算 loss”。
    想让它既做输入又要求模型输出更新后的记忆，请用 MemoryCoTBuilder。

    模板（embodiment=r1、2 个相机、离散动作）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Memory: <memory_text_!> State: <proprio_proprio_!>;
        Action: <EOV><EOC><action_action>|<eos>
    """

    # 训练/推理都需要 memory（推理时它就是模型输入的一部分）。
    required_fields = ("memory",)
    eval_required_fields = ("memory",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Memory: <memory_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "Action: <EOV><EOC><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        # 用 .get(..., "") 兜底：即便 memory 缺失也不会 KeyError（此时填空串）。
        samples["memory"] = data.get("memory", "")


class SubtaskCoTBuilder(BaseSamplesBuilder):
    """在动作之前先输出一段子任务文本 CoT（`Subtask: ...`）。

    这是最常用的 CoT 模板：模型先“想一步”（当前该做的子任务），再输出动作。

    CoT 文本来源：data["atomic_task"]，由 atomic_task_index 解码得到
    （数据侧 r1lite/r1pro 的 `_merged_final_v30` 目录才有这份标注）。
    atomic_task 是**严格必需**字段：没有这份标注的数据集 can_handle() 直接返回 False。

    模板（embodiment=r1、2 个相机、离散动作）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!><EOC><atomic_task_text>|
        Action: <EOV><action_action>|<eos>

    逐段解释：
      · `<prompt_text_!>` 是提示语（"predict subtask"），属于输入，不参与 loss；
      · `<EOC>` 之后才是模型要生成的 CoT（`<atomic_task_text>`，参与 loss）；
      · 结尾的 `|` 和 `Action: ` 是 EOC 之后的静态文本，同样参与 loss；
      · `<EOV>` 之后是动作 token（`<action_action>`，参与 loss）。

    ⚠️ 提醒：本模板要求动作区段能被正确离散化
       （batchify_action=True，或者 action_tokenizer 能处理原始动作 dict）。
       如果是 FM-only（连续动作、动作不过 tokenizer），请改用 SubtaskCoTBuilderFMOnly，
       否则会把 Python repr 字符串当成标签训练（详见该类的解释）。
    """

    # 严格要求 atomic_task；eval_required_fields=() 表示推理时不需要任何 CoT 标注。
    required_fields = ("atomic_task",)
    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><atomic_task_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        # 模板里的 <atomic_task_text> 取这个键；CoT 文本前面固定加 "Subtask: " 前缀。
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"
        # 模板里的 <prompt_text_!> 取这个键（给模型的任务提示，不算 loss）。
        samples["prompt"] = "predict subtask"


class TaskAsSubtaskCoTBuilder(SubtaskCoTBuilder):
    """用**当前帧的 task 文本**当 CoT 目标（而不是 atomic_task）。

    适用场景：foldbench 这类 hardcode_instruction 配置——指令槽被固定成一句高层描述，
    而 data["task"] 本身保留着逐帧的细粒度标签，可以直接拿来当 CoT 目标。

    ⚠️ required_fields=("task",) 意味着“任何数据集都能匹配上”（task 人人都有），
       所以它只应该用在 task 确实是细粒度标注的数据上，否则等于拿粗粒度指令当 CoT 标签。
    """

    required_fields = ("task",)
    eval_required_fields = ()

    def _populate_extra_samples(self, data, samples):
        # 与父类唯一差别：CoT 目标取 data["task"] 而不是 data["atomic_task"]。
        samples["atomic_task"] = f"Subtask: {data.get('task', '')}"
        samples["prompt"] = "predict subtask"


class SubtaskCoTBuilderFMOnly(SubtaskCoTBuilder):
    """Subtask CoT 的 FM-only 版本：自回归头只学生成子任务文本，动作交给 FM 分支。

    相比 SubtaskCoTBuilder，模板里**去掉了 `<action_action>` 占位符**：
        <image0_image_!><image1_image_!><bos>Embodiment: ...; Task: ... State: <proprio>;
        <prompt_text_!><EOC><atomic_task_text>|
        Action: <EOV><eos>

    ── 为什么需要这个变体（重要）────────────────────────────────────────
      FM-only 训练通常设置 `batchify_action=false`，动作不经过离散 tokenizer。
      而父类 SubtaskCoTBuilder 的模板里仍然有 `<action_action>` 占位符，
      这会触发 BuiltinActionProcessor.process(raw_dict)，即拿“原始动作 dict”去序列化。
      当动作模板直接写 `{action}` 时，Python 的 str() 会把原始 dict 变成一个普通字符串，例如
      `{'value': tensor([[...]]), 'action_dim_is_pad': ...}`；
      这个字符串随后被切成 AR 标签 —— 于是模型被训练去预测“Python 张量的文本表示”，
      推理时的 CoT 就可能吐出一长串 `"[ 1.1111, -0.6111, ..."` 这样的畸形文本。
      看起来像“没训好”，实际上是训练目标本身错了。

      本类去掉动作占位符后，自回归头只在子任务文本上计算 loss，
      错误信号从源头消失；与此同时 FM 分支照常回归连续动作，训练不受影响。
    """

    # 继承父类的 required_fields=("atomic_task",)；推理同样不需要 CoT 标注。
    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><atomic_task_text>|"
            "Action: <EOV><eos>"
        )


class FutureSubtaskCoTBuilder(SubtaskCoTBuilder):
    """用“未来第 N 帧”的任务文本作为 CoT 目标（让模型学会前瞻规划）。

    数据集侧已经把它预解码成 data["future_task"]。与 SubtaskCoTBuilder 的区别：
    目标是**未来帧**的子任务描述，而不是当前帧的，从而赋予模型 look-ahead 规划能力。

    当目标位置越过 episode 末尾时，数据集会 clamp 到最后一帧（直接复制），
    与动作 chunk 的补齐方式一致，因此这种情况仍视为合法训练信号。

    前置条件（都已具备，这里只是说明依赖）：
      · 数据集必须注册 delta_timestamps["task_index"]
        —— GalaxeaLerobotDataset._build_delta_timestamps 里已注册；
      · 并在 __getitem__ 里解码 future_task
        —— 实现在 lerobot_dataset_v3.py。
      偏移量默认 16 帧（注意：galaxea_lerobot_dataset.py 的 future_task_offset
      目前传不下去，底层读到的恒为默认值，详见该文件注释）。
    """

    # 严格要求 future_task；推理时不需要 CoT 标注。
    required_fields = ("future_task",)
    eval_required_fields = ()

    def _populate_extra_samples(self, data, samples):
        # 注意：这里仍然复用 <atomic_task_text> 这个槽名，只是填的是未来帧的任务文本。
        future_task = data.get("future_task", "")
        samples["atomic_task"] = f"Subtask: {future_task}"
        samples["prompt"] = "predict future subtask"


class BBoxCoTBuilder(BaseSamplesBuilder):
    """BBox CoT：模型在输出动作之前，先把目标物体的框（bbox）说出来。

    这条模板训练的是“空间 grounding”能力：先定位“要操作的东西在哪”，再决定怎么动。

    数据来源（lerobot_dataset_v3.py）：
        data["bbox"] ← bbox_index 查 tasks 表，得到一个 JSON 字符串
                       {"obj_name": [x1,y1,x2,y2], ...}
        坐标已归一化到 0~1，且同一帧可能包含多个物体。

    模板（embodiment=r1、2 个相机、离散动作）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>\n<bbox_text>|
        Action: <EOV><EOC><action_action>|<eos>
    注意本模板的 <EOC> 出现在 <bbox_text> **之前**：
    也就是说 bbox 就是模型要生成的**第一段** CoT，之后才轮到动作 token。
    （对比：SubtaskCoTBuilder 的 <EOC> 写在提示语之后、子任务之前。）
    """

    # 训练需要 bbox；推理（eval）也要求 bbox 存在，因为它是推理时真正要喂进去的输入之一。
    required_fields = ("bbox",)
    eval_required_fields = ("bbox",)

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """在父类判据之上，额外要求 bbox 是**非空** JSON。

        原因：部分批次（例如 warning_lite）里 bbox 字段存在、内容却是 "{}"，
        这种“有键无内容”的情况不该被当成有效标注。
        """
        if not super().can_handle(data):
            return False
        # 局部 import：只有走到这个类才需要 json。
        import json

        try:
            # bbox 可能是 JSON 字符串，也可能已经是 dict（取决于上游解码方式）。
            bbox_data = json.loads(data["bbox"]) if isinstance(data["bbox"], str) else data["bbox"]
            return bool(bbox_data)
        except (json.JSONDecodeError, TypeError):
            # JSON 解析失败、或类型不对 → 视为没有可用标注。
            return False

    def can_handle_for_eval(self, data: Dict[str, Any]) -> bool:
        """eval 路径：同样保留“bbox JSON 非空”的检查。

        空的 JSON '{}' 对推理没有任何意义，必须排除。
        BBoxSubtaskCoTBuilder（eval_required_fields=()）会跳过这项检查——
        因为 bbox 不在它的 eval_required_fields 里，下面会提前 return True。
        """
        if not super().can_handle_for_eval(data):
            return False
        # 这一段其实是给子类留的“开关”：
        # BBoxSubtaskCoTBuilder 继承了本方法，但它的 eval_required_fields=()，
        # 于是 bbox 不在其中 → 跳过“非空 JSON”检查。
        # 约定：如果某个子类推理时不需要 bbox 作为输入，就必须把 bbox
        # 从 eval_required_fields 里去掉。
        if "bbox" not in self.eval_required_fields:
            return True
        import json

        try:
            bbox_data = json.loads(data["bbox"]) if isinstance(data["bbox"], str) else data["bbox"]
            return bool(bbox_data)
        except (json.JSONDecodeError, TypeError):
            return False

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><bbox_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    @staticmethod
    def _format_bbox_json(bbox_json: str) -> str:
        """把 {"obj": [x1,y1,x2,y2], ...} 格式化成 CoT 文本。

        输出形如：`BBox: towel <loc0490><loc0116><loc0705><loc0287>`
        其中 <locXXXX> 是 PaliGemma 系的“位置 token”（编号 0~1023，形如 <loc0000>），
        需要模型侧开启 add_loc_tokens 才会被当成单个 token 处理。
        坐标顺序是 **y1 x1 y2 x2**（PaliGemma 约定：先 y 后 x），并 clamp 到 0~1023。

        解析失败或内容为空时返回空串（调用方据此得到“没有 bbox CoT”）。
        """
        # 局部 import：只有真正格式化时才需要 json。
        import json

        def _loc(v):
            # 归一化坐标 [0,1] → loc token 编号 [0,1023]，不足 4 位补零。
            return f"<loc{max(0, min(1023, round(v * 1024))):04d}>"

        def _fmt(name, bbox):
            x1, y1, x2, y2 = bbox
            # 按 PaliGemma 的 yxyx 顺序拼 4 个 loc token。
            locs = "".join(_loc(v) for v in [y1, x1, y2, x2])
            return f"{name} {locs}"

        try:
            bbox_data = json.loads(bbox_json) if isinstance(bbox_json, str) else bbox_json
        except (json.JSONDecodeError, TypeError):
            return ""
        if not bbox_data:
            return ""
        # coords 为空（例如 null）的物体直接跳过；多个物体之间用 "; " 连接。
        parts = [_fmt(name, coords) for name, coords in bbox_data.items() if coords]
        return "BBox: " + "; ".join(parts) if parts else ""

    def _populate_extra_samples(self, data, samples):
        # 模板里 <bbox_text> 取这个键；缺字段时给 "{}"，最终会格式化成空串。
        bbox_text = self._format_bbox_json(data.get("bbox", "{}"))
        samples["bbox"] = bbox_text
        samples["prompt"] = "predict bbox"


class BBoxSubtaskCoTBuilder(BBoxCoTBuilder):
    """bbox + subtask 联合 CoT：先输出 bbox，再输出子任务，最后输出动作。

    相当于把 BBoxCoTBuilder 与 SubtaskCoTBuilder 串起来，形成两级 CoT：
    “看哪里（bbox）”→“做什么（subtask）”→“怎么动（action）”。

    模板（embodiment=r1、2 个相机、离散动作）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>\n<EOC><bbox_text>|<atomic_task_text>|
        Action: <EOV><action_action>|<eos>
    """

    # 两个 CoT 目标都严格必需；推理不需要任何标注（eval_required_fields=()），
    # 因此也会自动跳过父类那条“bbox 必须非空”的检查（见 BBoxCoTBuilder.can_handle_for_eval）。
    required_fields = ("bbox", "atomic_task")
    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><bbox_text>|<atomic_task_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        # 先复用父类逻辑：填好 samples["bbox"] 与 samples["prompt"]。
        super()._populate_extra_samples(data, samples)  # 设置 samples["bbox"]、samples["prompt"]
        # 再补上第二级 CoT：子任务文本。
        # ⚠️ 这里必须用 atomic_task，不能退回 task——
        #    否则就会重现 “Slot-content Semantics” 断言专门防住的那类隐性 bug。
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"
        samples["prompt"] = "predict bbox, subtask and action"


class Trace2DCoTBuilder(BaseSamplesBuilder):
    """2D 轨迹 CoT：模型在输出动作之前，先说出左右夹爪在画面中的 2D 落点。

    训练的是“夹爪落点 grounding”：让模型用图像坐标把“要摸到哪里”讲清楚。
    好处是推理时的 CoT 可以直接可视化，便于检查模型是不是真的看懂了画面。

    数据来源（lerobot_dataset_v3.py）：
        data["trace_2d"] ← 2d_trace_index 查 tasks 表，得到 JSON 字符串
            {"uv_left": [u,v]|null, "visb_left": bool,
             "uv_right": [u,v]|null, "visb_right": bool, ...}
        坐标归一化到 0~1，表示当前帧左/右臂夹爪投影到画面上的位置。

    模板（embodiment=r1、2 个相机、离散动作）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>
        <EOC><trace_2d_text>|
        Action: <EOV><action_action>|<eos>

    CoT 文本示例：
        Trace: Left <loc0543><loc0436>; Right None
    """

    # 训练需要 trace_2d 标注；推理不需要（CoT 由模型自己生成）。
    required_fields = ("trace_2d",)
    eval_required_fields = ()

    def can_handle(self, data: Dict[str, Any]) -> bool:
        if not super().can_handle(data):
            return False
        import json

        try:
            td = (
                json.loads(data["trace_2d"])
                if isinstance(data["trace_2d"], str)
                else data["trace_2d"]
            )
            # 只有至少一只手臂可见时这条 trace 才有意义（两只都不可见 = 没有可用监督）。
            return bool(td.get("visb_left") or td.get("visb_right"))
        except (json.JSONDecodeError, TypeError, AttributeError):
            # JSON 坏掉、或 td 不是 dict（AttributeError）→ 视为无标注。
            return False

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><trace_2d_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    @staticmethod
    def _format_trace_2d_json(trace_json: str) -> str:
        """把 trace_2d JSON 格式化成 CoT 文本，UV 坐标用 <loc> token 编码。

        输出示例：`Trace: Left <loc0543><loc0436>; Right None`
        左/右臂各输出一段：可见且坐标存在 → 两个 loc token（按 u、v 顺序）；
        否则输出 `Xxx None`，让模型也能学会表达“这只手这帧看不见”。
        """
        # 局部 import：只有真正格式化时才需要 json。
        import json

        def _loc(v):
            # 归一化坐标 [0,1] → loc token 编号 [0,1023]，不足 4 位补零。
            return f"<loc{max(0, min(1023, round(v * 1024))):04d}>"

        try:
            data = json.loads(trace_json) if isinstance(trace_json, str) else trace_json
        except (json.JSONDecodeError, TypeError):
            return ""

        parts = []
        uv_left = data.get("uv_left")
        # 左臂：可见且坐标不为 None 才写坐标，否则写 None。
        if data.get("visb_left") and uv_left is not None:
            parts.append(f"Left {_loc(uv_left[0])}{_loc(uv_left[1])}")
        else:
            parts.append("Left None")

        uv_right = data.get("uv_right")
        # 右臂同理。
        if data.get("visb_right") and uv_right is not None:
            parts.append(f"Right {_loc(uv_right[0])}{_loc(uv_right[1])}")
        else:
            parts.append("Right None")

        return "Trace: " + "; ".join(parts)

    def _populate_extra_samples(self, data, samples):
        # 模板里 <trace_2d_text> 取这个键；缺字段时给 "{}"，
        # 最终格式化成 "Trace: Left None; Right None"。
        trace_text = self._format_trace_2d_json(data.get("trace_2d", "{}"))
        samples["trace_2d"] = trace_text
        samples["prompt"] = "predict 2d trace of gripper"


class SubtaskActionHintCoTBuilder(SubtaskCoTBuilder):
    """subtask + action_hint 联合 CoT：先预测子任务，再预测当前帧的夹爪动作描述。

    比 SubtaskCoTBuilder 多一级更细的“动作意图”文本，等于让模型把
    “做什么（subtask）”→“这只手具体怎么动（action_hint）”都说清楚，最后才输出动作。

    数据来源（lerobot_dataset_v3.py）：
        data["atomic_task"]  ← 由 atomic_task_index 解码
                               （r1lite/r1pro 的 _merged_final_v30）
        data["action_hint"]  ← action_hint_index 查 tasks 表
                               （逐帧的夹爪动作描述），例如：
                               "Right gripper moves forward right down, rotates roll positive pitch positive, closes."

    模板（embodiment=r1、2 个相机、离散动作）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>
        <EOC><atomic_task_text>|<action_hint_text>|
        Action: <EOV><action_action>|<eos>
    """

    # 严格同时要求 atomic_task，是为了贴合数据事实：action_hint 只出现在
    # _merged_final_v30 里，而那份数据中 atomic_task 一定同时存在。
    required_fields = ("atomic_task", "action_hint")
    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><atomic_task_text>|<action_hint_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        # 先复用父类：填好 samples["atomic_task"] 与 samples["prompt"]。
        super()._populate_extra_samples(
            data, samples
        )  # 设置 samples["atomic_task"]、samples["prompt"]
        # 再补上第二级 CoT：动作提示文本。`or ""` 用来把 None 归一成空串。
        action_hint = data.get("action_hint") or ""
        samples["action_hint"] = f"ActionHint: {action_hint}"
        samples["prompt"] = "predict subtask and action hint"


class HighLevelAtomicTaskCoTBuilder(SubtaskCoTBuilder):
    """用 high_level_instruction 当指令，CoT 仍输出 atomic_task，最后输出动作。

    这是“高层指令 → 子任务 → 动作”的分层训练：指令槽给的是粗粒度目标，
    模型要自己想出与当前帧匹配的细粒度子任务，再动。

    数据来源（r1lite/r1pro 的 _merged_final_v30）：
        data["high_level_instruction"] ← 由 high_level_instruction_index 解码
                                          （注入到 command 槽）
        data["atomic_task"]            ← 由 atomic_task_index 解码（CoT 目标）

    模板与 SubtaskCoTBuilder 完全一致，只有注入到指令槽的取值换成了
    high_level_instruction（靠 _override_command 实现）。
    """

    # 训练要两者都有；推理时只需要 high_level_instruction（CoT 由模型生成）。
    required_fields = ("high_level_instruction", "atomic_task")
    eval_required_fields = ("high_level_instruction",)

    def _override_command(self, data):
        # 指令槽改用高层指令（默认应该是 task）。
        return data.get("high_level_instruction")

    def _populate_extra_samples(self, data, samples):
        # 父类已经写过一次 atomic_task，这里再显式写一遍（内容相同，属于冗余但无害）。
        super()._populate_extra_samples(
            data, samples
        )  # 设置 samples["atomic_task"]、samples["prompt"]
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"


class AtomicTaskBaseSamplesBuilder(BaseSamplesBuilder):
    """模板与 BaseSamplesBuilder 相同，但把 atomic_task 注入到指令槽。

    它**没有 CoT**，看完指令直接输出动作。
    用途：只想训练 `细粒度子任务 → 动作` 的映射（而不是 `粗粒度 task → 动作`），
    又不想引入任何额外文本生成时使用。

    数据来源：
        data["atomic_task"] ← 由 atomic_task_index 解码
                              （r1lite/r1pro 的 _merged_final_v30）
    """

    # 训练与推理都需要 atomic_task（推理时它是输入，不是输出）。
    required_fields = ("atomic_task",)
    eval_required_fields = ("atomic_task",)

    def _override_command(self, data):
        # 关键点：用 atomic_task 顶掉默认的 task 作为指令。
        return data.get("atomic_task")


class MemoryCoTBuilder(MemorySamplesBuilder):
    """记忆输入 + 记忆更新 CoT 输出：不仅读记忆，还要求模型写出“更新后的记忆”。

    数据来源（lerobot_dataset_v3.py）：
        data["memory"]        ← prev_memory_index 查 tasks 表
                                （上一帧的记忆，作为**输入**，不参与 loss）
        data["memory_update"] ← memory_index 查 tasks 表
                                （当前帧更新后的记忆，作为 **CoT 输出目标**）

    模板（embodiment=r1、2 个相机）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Memory: <memory_text_!> State: <proprio_proprio_!>;
        <cotprefix_text_!><EOC><memory_update_text>|
        Action: <EOV><action_action>|<eos>
    """

    # 训练要 memory + memory_update；推理只需要 memory（更新后的记忆由模型生成）。
    required_fields = ("memory", "memory_update")
    eval_required_fields = ("memory",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            "Memory: <memory_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<cotprefix_text_!><EOC><memory_update_text>|\n"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        # 父类：samples["memory"] = data["memory"]。
        super()._populate_extra_samples(data, samples)  # 设置 samples["memory"] = data["memory"]
        memory_update = data.get("memory_update")
        # 有更新内容才写 CoT 文本；否则填空串（等价于“这帧不更新记忆”）。
        samples["memory_update"] = f"Updated Memory: {memory_update}" if memory_update else ""
        # <cotprefix_text_!> 是给模型的提示语（输入，不算 loss）。
        samples["cotprefix"] = "Please output the updated memory:"


class MemorySubtaskCoTBuilder(MemorySamplesBuilder):
    """记忆输入 + 子任务 CoT + 记忆更新 CoT：一次输出两段 CoT。

    字段来源：
        data["memory"]        ← prev_memory_index（输入，不参与 loss）
        data["atomic_task"]   ← atomic_task_index（CoT 输出）
        data["memory_update"] ← memory_index（CoT 输出）
    注意虽然继承自 MemorySamplesBuilder，但 required_fields 里**没有** memory_update，
    也就是说这份 CoT 的第二段允许为空。

    模板：
        <images><bos>Embodiment: ...; Task: ... Memory: <memory_text_!> State: ...;
        <cotprefix_text_!><EOC><atomic_task_text>|<memory_update_text>|
        Action: <EOV><action_action>|<eos>
    """

    # 训练要 memory + atomic_task；推理只要 memory。
    required_fields = ("memory", "atomic_task")
    eval_required_fields = ("memory",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            "Memory: <memory_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<cotprefix_text_!><EOC><atomic_task_text>|<memory_update_text>|\n"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        super()._populate_extra_samples(data, samples)  # 设置 samples["memory"]
        # 第一段 CoT：子任务。
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"
        # 第二段 CoT：更新后的记忆（可能为空）。
        memory_update = data.get("memory_update")
        samples["memory_update"] = f"Updated Memory: {memory_update}" if memory_update else ""
        samples["cotprefix"] = "Please output the subtask and updated memory:"


class PlanStepCoTBuilder(BaseSamplesBuilder):
    """计划（输入，不参与 loss）+ 当前计划步 CoT 输出。

    训练模型“按计划做事”：计划全文作为条件输入，模型需要判断当前执行到第几步。

    数据来源（lerobot_dataset_v3.py）：
        data["plan"]      ← plan_index 查 tasks 表
                             （完整计划文本，作为输入，不参与 loss）
        data["plan_step"] ← parquet 里的原始整数字段（当前步号，从 0 开始）

    计划文本格式示例：
        "1. Pick up the towel\\n2. Move to the table\\n3. Fold the towel"

    模板（embodiment=r1、2 个相机）：
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Plan: <plan_text_!> State: <proprio_proprio_!>;
        <cotprefix_text_!><EOC><plan_step_text>|
        Action: <EOV><action_action>|<eos>
    说明：计划全文用 `<plan_text_!>`（输入、不算 loss），
    当前步文本用 `<plan_step_text>`（在 EOC 之后，算 loss）。
    """

    # 训练需要 plan + plan_step；推理只需要 plan（当前步由模型生成）。
    required_fields = ("plan", "plan_step")
    eval_required_fields = ("plan",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            "Plan: <plan_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<cotprefix_text_!><EOC><plan_step_text>|\n"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        plan = data.get("plan", "")
        plan_step_raw = data.get("plan_step")
        # plan_step 可能来自 parquet 的 numpy/torch 标量（带 .item()），
        # 也可能是普通 int；两种情况都要兼容，拿不到就当第 0 步。
        if hasattr(plan_step_raw, "item"):
            step_num = plan_step_raw.item()
        else:
            step_num = int(plan_step_raw) if plan_step_raw is not None else 0
        # 从计划全文里按行找出 "{step_num}. <文本>" 那一行，取其中的文本部分。
        # 找不到就留空串（此时 <plan_step_text> 为 "Step: "，属于无监督的退化情况）。
        current_step = ""
        for line in plan.split("\n"):
            # 逐行去掉首尾空白后再匹配，避免缩进/空格导致匹配失败。
            line = line.strip()
            if line.startswith(f"{step_num}."):
                # 去掉 "N." 前缀，只保留步骤描述文本。
                current_step = line[len(f"{step_num}.") :].strip()
                break
        samples["plan"] = plan
        samples["plan_step"] = f"Step: {current_step}" if current_step else ""
        samples["cotprefix"] = "Please output the current plan step:"


class MixedSamplesBuilder(BaseSamplesBuilder):
    """混合 Builder：训练时按权重随机挑一个“有标注的候选 Builder”，推理时固定用一个。

    ── 训练路径 ────────────────────────────────────────────────────────
      在所有 `can_handle(data) == True` 的候选里按 weight 加权随机抽一个
      ——“这条数据有什么标注，就练什么 CoT”；
      一个候选都不匹配（这帧没有任何 CoT 标注）→ 退回 BaseSamplesBuilder（无 CoT）。
    ── 推理路径 ────────────────────────────────────────────────────────
      固定用 eval_builder，判据是 `can_handle_for_eval(data)`
      （CoT 目标由模型生成，不依赖标注）；
      没配 eval_builder，或它也不匹配 → 同样退回 BaseSamplesBuilder。

    典型用法：一个训练集里同时混着“有 atomic_task 的目录”“有 bbox 的目录”“只有 task 的目录”，
    用 candidates 把它们全列出来，数据命中谁就练谁的 CoT，谁也不命中就退化成纯动作监督。

    配置示例（注意 `_recursive_: false`）：
        samples_builder:
          _target_: g05.data_processor.processor.samples_builder.MixedSamplesBuilder
          _partial_: true
          _recursive_: false        # 让 Hydra 不要把 candidates 里的 _target_ 也递归实例化
          candidates:
            - _target_: g05.data_processor.processor.samples_builder.SubtaskCoTBuilder
              weight: 1.0
            - _target_: g05.data_processor.processor.samples_builder.BBoxCoTBuilder
              weight: 0.5
          eval_builder:             # 推理固定用这个 Builder；写 null 表示推理不要 CoT
            _target_: g05.data_processor.processor.samples_builder.SubtaskCoTBuilder
    """

    def __init__(
        self,
        num_input_images: int,
        image_sizes: Dict[str, Any],
        embodiment_type: Optional[str] = None,
        candidates: Optional[List[Any]] = None,   # [(候选 Builder 配置, weight), ...]
        eval_builder: Optional[Any] = None,       # 推理时固定使用的 Builder 配置（可为 None）
        **kwargs,
    ):
        # 先按父类初始化（同时把 kwargs 里的 hardcode_* 收进属性）。
        super().__init__(num_input_images, image_sizes, embodiment_type, **kwargs)
        # 默认按“训练”模式；由 GalaxeaCoTProcessor.train() / eval() 通过
        # set_training() 切换。注意：这个标志决定走“随机选候选”还是“固定 eval_builder”。
        self._training: bool = True

        # 这三个参数要原样传给每个候选 Builder，让它们与整体行为保持一致。
        # 其余 kwargs（未知参数）不透传，避免子 Builder 收到不认识的形参。
        _builder_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in ("hardcode_proprio_pad_zeros", "hardcode_action_pad_ones", "hardcode_instruction")
        }

        # 候选列表：[(Builder 实例, 权重), ...]
        self._candidates: List[tuple] = []
        for cand in candidates or []:
            cls, weight = self._resolve_candidate(cand)
            # 权重必须为正：random.choices 遇到 <=0 的权重会报错或退化成“永远只选一个”。
            if weight <= 0:
                raise ValueError(
                    f"MixedSamplesBuilder candidate {cls.__name__} has non-positive weight={weight}; "
                    f"weights must be > 0 or random.choices may raise ValueError or degenerate."
                )
            # 逐个实例化候选 Builder（它们都是 BaseSamplesBuilder 的子类，
            # 构造参数和父类一致，所以这里显式把公共参数传下去）。
            builder = cls(
                num_input_images=num_input_images,
                image_sizes=image_sizes,
                embodiment_type=embodiment_type,
                **_builder_kwargs,
            )
            self._candidates.append((builder, weight))

        # 推理专用 Builder：为 None 时推理退回 BaseSamplesBuilder（无 CoT）。
        self._eval_builder: Optional[BaseSamplesBuilder] = None
        if eval_builder is not None:
            # 第二个返回值（权重）在 eval 场景没有意义，直接丢掉。
            cls, _ = self._resolve_candidate(eval_builder)
            self._eval_builder = cls(
                num_input_images=num_input_images,
                image_sizes=image_sizes,
                embodiment_type=embodiment_type,
                **_builder_kwargs,
            )

    @staticmethod
    def _resolve_candidate(cand: Any):
        """把候选配置统一解析成 (类, 权重) 二元组。

        支持三种写法：
          · Hydra 的 DictConfig（配置里最常见）→ 先转成普通 dict 再解析；
          · 普通 dict：{"_target_": "完整类路径", "weight": 1.0}；
          · 直接给类对象（写脚本/测试时用），此时权重默认 1.0。
        """
        # 局部 import：只有解析配置时才需要 locate / omegaconf，
        # 也让本模块在没装 omegaconf 的环境下仍能 import（例如纯推理环境）。
        from pydoc import locate

        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(cand, DictConfig):
                # resolve=True：把 ${...} 插值也一并展开。
                cand = OmegaConf.to_container(cand, resolve=True)
        except ImportError:
            # 没装 omegaconf 也无所谓：下面按 dict / class 处理。
            pass
        if isinstance(cand, dict):
            # _target_ 是 Hydra 的“反射实例化”约定：一个完整的类路径字符串。
            cls = locate(cand["_target_"])
            if cls is None:
                raise ImportError(f"Cannot locate class: {cand['_target_']}")
            return cls, float(cand.get("weight", 1.0))
        # 既不是 dict 也不是 DictConfig：当成类本身，权重 1.0。
        return cand, 1.0

    # 覆盖父类的 setter：形态名变化时要同步给所有候选 Builder 与 eval_builder，
    # 否则它们仍然拿着旧的 embodiment 值去填模板。
    @BaseSamplesBuilder.embodiment_type.setter
    def embodiment_type(self, value: Optional[str]) -> None:
        self._embodiment_type = value
        for builder, _ in self._candidates:
            builder.embodiment_type = value
        if self._eval_builder is not None:
            self._eval_builder.embodiment_type = value

    def set_training(self, training: bool) -> None:
        """切换训练/推理模式（由 GalaxeaCoTProcessor.train()/eval() 调用）。"""
        self._training = training
        # 递归下发：候选中可能嵌套着另一个 MixedSamplesBuilder（多层混合），
        # 用 hasattr 判断是因为普通 Builder 没有这个方法，也不必接收这个标志。
        for builder, _ in self._candidates:
            if hasattr(builder, "set_training"):
                builder.set_training(training)
        if self._eval_builder is not None and hasattr(self._eval_builder, "set_training"):
            self._eval_builder.set_training(training)

    def build(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        """按模式分发：训练=加权随机抽候选；推理=固定 eval_builder（都可回退到无 CoT 基类）。"""
        # 局部 import：只有这里需要随机数。
        import random

        # ── 推理路径 ─────────────────────────────────────────────
        if not self._training:
            if self._eval_builder is not None and self._eval_builder.can_handle_for_eval(data):
                logger.debug("[MixedSamplesBuilder] eval → %s", type(self._eval_builder).__name__)
                return self._eval_builder.build(data, sample)
            # 没配 eval_builder，或这帧不满足它的输入要求 → 退回父类（无 CoT）。
            logger.debug("[MixedSamplesBuilder] eval → BaseSamplesBuilder (fallback, no CoT)")
            return super().build(data, sample)

        # ── 训练路径 ─────────────────────────────────────────────
        # 筛出“这条数据的标注能满足”的候选（can_handle 里会过滤 None/NaN/-1/空 JSON 等）。
        applicable = [(b, w) for b, w in self._candidates if b.can_handle(data)]
        if not applicable:
            # 一个都不命中：这帧没有可用 CoT 标注，退化成纯动作监督。
            logger.debug("[MixedSamplesBuilder] train → BaseSamplesBuilder (no annotation matched)")
            return super().build(data, sample)
        # zip(*pairs)：把 (builder, weight) 列表拆成两个元组，供 random.choices 使用。
        builders, weights = zip(*applicable)
        # 加权随机抽一个：weight 越大，这种 CoT 格式被抽中的概率越高。
        chosen = random.choices(builders, weights=list(weights), k=1)[0]
        logger.debug(
            "[MixedSamplesBuilder] train → %s (from %d applicable)",
            type(chosen).__name__,
            len(applicable),
        )
        return chosen.build(data, sample)


# ====================================================================== #
#  冒烟测试（Smoke test）                                                #
#  运行：python -m g05.data_processor.processor.samples_builder         #
# ====================================================================== #

if __name__ == "__main__":
    # ================================================================= #
    #  冒烟测试（Smoke Test）                                           #
    #                                                                   #
    #  运行方式：python -m g05.data_processor.processor.samples_builder #
    #                                                                   #
    #  它做两件事：                                                     #
    #    1) 把每个 Builder 的模板在三种模型形态下渲染出来（看模板长啥样）；#
    #    2) 跑一批断言，覆盖：                                          #
    #       · MixedSamplesBuilder 的候选筛选（can_handle 过滤脏值）      #
    #       · set_training / embodiment_type 的传播                     #
    #       · “Slot-content Semantics”回归断言（防 atomic_task 槽错填）  #
    #       · can_handle_for_eval 的字段范围                            #
    #  改了 Builder 之后建议先跑一遍；新增 Builder 时请按
    #  docs/data/samples_builders_zh.md 第 9 节的清单补充对应断言。
    # ================================================================= #
    # 注意：这里没有用 pytest，纯靠 assert + print，方便直接看输出。
    # import sys 是从上游带过来的历史遗留（本段代码并没有用到 sys），保留不影响运行。
    import sys

    # types.SimpleNamespace 只用来伪造一个“极简 tokenizer”，
    # 目的仅仅是让 SpecialTokenManager 能识别出 instruct 模型的 bos/eos。
    import types as _types
    from g05.utils.common.special_tokens import SpecialTokenManager

    # 三种 token map，对应三种模板渲染结果：
    #   · paligemma       —— 无对话包装，<bos>/<eos> 原样使用
    #   · qwen35-base     —— 无对话包装
    #   · qwen35-instruct —— 注入 <|im_start|>user ... <|im_end|> 等对话格式
    _paligemma = SpecialTokenManager.for_model("paligemma")
    _qwen35_base = SpecialTokenManager.for_model("qwen35")
    _mock_inst = _types.SimpleNamespace(
        bos_token="<|im_start|>",
        eos_token="<|endoftext|>",
        pad_token=None,
    )
    _qwen35_inst = SpecialTokenManager.for_model("qwen35", tokenizer=_mock_inst)
    TOKEN_MAPS = [
        ("paligemma", _paligemma),
        ("qwen35-base", _qwen35_base),
        ("qwen35-instruct", _qwen35_inst),
    ]

    # 要渲染模板的 Builder 清单：(名字, 类, 额外构造参数)。
    # 注意 TaskAsSubtaskCoTBuilder / SubtaskCoTBuilderFMOnly / FutureSubtaskCoTBuilder /
    # AtomicTaskBaseSamplesBuilder 等没有列在这里，它们的覆盖在下面的断言区。
    CASES = [
        ("BaseSamplesBuilder", BaseSamplesBuilder, {}),
        ("MemorySamplesBuilder", MemorySamplesBuilder, {"embodiment_type": "r1"}),
        ("SubtaskCoTBuilder", SubtaskCoTBuilder, {"embodiment_type": "r1"}),
        ("BBoxCoTBuilder", BBoxCoTBuilder, {"embodiment_type": "r1"}),
        ("BBoxSubtaskCoTBuilder", BBoxSubtaskCoTBuilder, {"embodiment_type": "r1"}),
        ("Trace2DCoTBuilder", Trace2DCoTBuilder, {"embodiment_type": "r1"}),
        ("SubtaskActionHintCoTBuilder", SubtaskActionHintCoTBuilder, {"embodiment_type": "r1"}),
        ("MemoryCoTBuilder", MemoryCoTBuilder, {"embodiment_type": "r1"}),
        ("PlanStepCoTBuilder", PlanStepCoTBuilder, {"embodiment_type": "r1"}),
    ]

    # 统一用 2 路相机、256x256（模拟 r1 的 head + wrist 两路相机）。
    DEFAULTS = dict(
        num_input_images=2, image_sizes={"image": (256, 256), "wrist_image": (256, 256)}
    )

    print("=" * 70)
    print("SamplesBuilder Smoke Test")
    print("=" * 70)

    # 逐个 Builder：先打印原始模板（repr 形式，方便看 \n 等转义），
    # 再打印三种模型形态下替换特殊 token 之后的模板。
    for name, cls, kwargs in CASES:
        builder = cls(**{**DEFAULTS, **kwargs})
        print(f"\n--- {name} ---")
        print(f"  raw template:")
        for line in builder.template.split("\n"):
            print(f"    {repr(line)}")
        for model_name, token_map in TOKEN_MAPS:
            resolved = token_map.resolve_template(builder.template)
            print(f"  [{model_name}]:")
            for line in resolved.split("\n"):
                print(f"    {line}")

    # ---- MixedSamplesBuilder ----
    print("\n" + "=" * 70)
    print("MixedSamplesBuilder Tests")
    print("=" * 70)

    # 构造一个“Subtask（权重 1.0）+ BBox（权重 0.5）”的混合 Builder，
    # 并指定推理时固定用 SubtaskCoTBuilder。
    # 这里故意用 dict 形式写 _target_（而不是 Hydra 的 DictConfig），
    # 顺便验证 _resolve_candidate 对 dict 的解析路径。
    mixed = MixedSamplesBuilder(
        **DEFAULTS,
        embodiment_type="r1",
        candidates=[
            {
                "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                "weight": 1.0,
            },
            {
                "_target_": "g05.data_processor.processor.samples_builder.BBoxCoTBuilder",
                "weight": 0.5,
            },
        ],
        eval_builder={"_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder"},
    )
    # 两个候选都被实例化；eval_builder 也被建出来了。
    assert len(mixed._candidates) == 2, "Expected 2 candidates"
    assert mixed._eval_builder is not None, "eval_builder should be set"

    # 用三份“假数据”验证 can_handle 的分流：
    #   有 atomic_task 的 → 只该命中 SubtaskCoTBuilder
    #   有 bbox 的        → 只该命中 BBoxCoTBuilder
    #   什么都没有的      → 谁都不该命中（训练时会退回无 CoT 的基类）
    data_with_subtask = {"atomic_task": "pick up cup"}
    data_with_bbox = {"bbox": '{"towel": [0.113, 0.479, 0.28, 0.688]}'}
    data_empty = {}

    applicable_subtask = [b for b, _ in mixed._candidates if b.can_handle(data_with_subtask)]
    applicable_bbox = [b for b, _ in mixed._candidates if b.can_handle(data_with_bbox)]
    applicable_empty = [b for b, _ in mixed._candidates if b.can_handle(data_empty)]

    assert (
        len(applicable_subtask) == 1 and type(applicable_subtask[0]).__name__ == "SubtaskCoTBuilder"
    ), f"data with atomic_task should only match SubtaskCoTBuilder, got {applicable_subtask}"
    assert len(applicable_bbox) == 1 and type(applicable_bbox[0]).__name__ == "BBoxCoTBuilder", (
        f"data with bbox should only match BBoxCoTBuilder, got {applicable_bbox}"
    )
    assert len(applicable_empty) == 0, "data without annotations should match no candidates"

    # 脏值 / 哨兵值过滤：NaN、-1、"null"、"none"、空 JSON '{}' 都不算“有标注”。
    subtask_builder = applicable_subtask[0]
    bbox_builder = applicable_bbox[0]
    import math

    assert not subtask_builder.can_handle({"task": "null"}), "string 'null' should be rejected"
    assert not subtask_builder.can_handle({"task": "none"}), "string 'none' should be rejected"
    assert not subtask_builder.can_handle({"task": "None"}), "string 'None' should be rejected"
    assert not subtask_builder.can_handle({"task": float("nan")}), "float NaN should be rejected"
    assert not subtask_builder.can_handle({"task": -1}), "int -1 should be rejected"
    assert not bbox_builder.can_handle({"bbox": "{}"}), "empty JSON '{}' should be rejected"
    assert not bbox_builder.can_handle({"bbox": "null"}), "bbox 'null' string should be rejected"
    print("  ✓ can_handle() filtering correct (incl. NaN/null/sentinel/-1/empty-JSON)")

    # 切换到推理模式：此后 build() 会固定走 eval_builder（而不是随机抽候选）。
    mixed.set_training(False)
    assert not mixed._training, "set_training(False) should set _training=False"
    print("  ✓ set_training(False) OK")

    # 权重保护：weight=0 必须在构造期就报错（否则 random.choices 行为不可控）。
    try:
        MixedSamplesBuilder(
            **DEFAULTS,
            embodiment_type="r1",
            candidates=[
                {
                    "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                    "weight": 0,
                }
            ],
        )
        raise AssertionError("weight=0 should raise ValueError")
    except ValueError as e:
        assert "non-positive weight" in str(e), f"unexpected error: {e}"
    print("  ✓ weight <= 0 raises ValueError")

    # set_training 的递归传播：验证“外层 MixedSamplesBuilder 切换模式时，
    # 内层（被当作 eval_builder 的）MixedSamplesBuilder 也会跟着切”。
    inner_mixed = MixedSamplesBuilder(
        **DEFAULTS,
        embodiment_type="r1",
        candidates=[
            {
                "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                "weight": 1.0,
            }
        ],
    )
    # 外层 Builder：这里先用一份同样的配置造出来，
    # 下面再手工把内层塞进它的 _eval_builder，间接验证嵌套场景也支持 set_training。
    outer = MixedSamplesBuilder(
        **DEFAULTS,
        embodiment_type="r1",
        candidates=[
            {
                "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                "weight": 1.0,
            }
        ],
    )
    # 手工构造嵌套关系（配置里直接用 _target_ 嵌套比较绕，这里模拟真实场景即可）。
    outer._eval_builder = inner_mixed
    # 切到推理：内层也应变成 False。
    outer.set_training(False)
    assert inner_mixed._training is False, (
        "Nested MixedSamplesBuilder._training should be propagated as False by the outer builder"
    )
    # 切回训练：内层同样应变成 True。
    outer.set_training(True)
    assert inner_mixed._training is True, (
        "Nested MixedSamplesBuilder._training should be propagated as True by the outer builder"
    )
    print("  ✓ nested MixedSamplesBuilder set_training propagates correctly")

    # embodiment_type 的传播：改外层的形态名，候选与 eval_builder 都要同步更新
    # （否则模板里的 Embodiment 槽会填成旧值）。
    mixed.embodiment_type = "r1pro"
    assert mixed._eval_builder.embodiment_type == "r1pro"
    for b, _ in mixed._candidates:
        assert b.embodiment_type == "r1pro"
    print("  ✓ embodiment_type propagation OK")

    # ---- Slot-content semantics (regression guards) ----
    # 槽内容语义（回归防线）：
    #   当数据里**同时**有 atomic_task（细粒度）和 task（粗粒度）时，
    #   samples["atomic_task"] 必须填 atomic_task 的文本，不能退化成 task 文本。
    #   这类 bug 很隐蔽（模板能跑、loss 也有，只是学错了目标），
    #   所以这里用断言把它钉住——新增/修改 Builder 时请同步补断言。
    print("\n" + "=" * 70)
    print("Slot-content Semantics")
    print("=" * 70)

    # 造一份“什么标注都有”的数据，下面反复复用。
    ATOMIC_TXT = "reach for and grasp the blue towel"
    TASK_TXT = "Fetch towel"
    HL_TXT = "Tidy the basket"
    HINT_TXT = "Right gripper closes."
    BBOX_TXT = '{"towel": [0.1, 0.2, 0.3, 0.4]}'
    MEM_TXT = "previous memory state"
    MEM_UPDATE_TXT = "new memory state"

    data_full = {
        "task": TASK_TXT,
        "atomic_task": ATOMIC_TXT,
        "high_level_instruction": HL_TXT,
        "action_hint": HINT_TXT,
        "bbox": BBOX_TXT,
        "memory": MEM_TXT,
        "memory_update": MEM_UPDATE_TXT,
    }
    # 同一份数据去掉 atomic_task：用来验证“严格必需字段”的 Builder 会拒收。
    data_no_atomic = {k: v for k, v in data_full.items() if k != "atomic_task"}

    def _slot(builder, data):
        """只跑 _populate_extra_samples，直接看它往 samples 里写了什么。"""
        out = {}
        builder._populate_extra_samples(data, out)
        return out

    # 1) 只要数据里有 atomic_task，atomic_task 槽就必须填 atomic_task 文本。
    atomic_slot_cases = [
        ("SubtaskCoTBuilder", SubtaskCoTBuilder),
        ("BBoxSubtaskCoTBuilder", BBoxSubtaskCoTBuilder),
        ("SubtaskActionHintCoTBuilder", SubtaskActionHintCoTBuilder),
        ("MemorySubtaskCoTBuilder", MemorySubtaskCoTBuilder),
        ("HighLevelAtomicTaskCoTBuilder", HighLevelAtomicTaskCoTBuilder),
    ]
    for name, cls in atomic_slot_cases:
        b = cls(**{**DEFAULTS, "embodiment_type": "r1"})
        # 先确认这份数据能让该 Builder 命中（否则后面断言没有意义）。
        assert b.can_handle(data_full), f"{name} should can_handle(data_full)"
        out = _slot(b, data_full)
        slot = out.get("atomic_task", "")
        # 必须包含细粒度文本……
        assert ATOMIC_TXT in slot, (
            f"{name}: atomic_task slot should contain atomic_task text, got {slot!r}; "
            f"fallback may have incorrectly used task text"
        )
        # ……且不能混进粗粒度 task 文本。
        assert TASK_TXT not in slot, (
            f"{name}: coarse task text should not appear when data has atomic_task, got {slot!r}"
        )
    print(f"  ✓ atomic_task slot prefers atomic_task across {len(atomic_slot_cases)} builders")

    # 2) TaskAsSubtaskCoTBuilder 是**故意**用 task 当 CoT 目标的特例
    #    （foldbench 那类 hardcode_instruction 场景：指令被固定，task 保留细粒度标签）。
    #    所以这里断言的是反面：没有 atomic_task 也必须能命中，且槽里放的是 task 文本。
    b = TaskAsSubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert b.can_handle(data_no_atomic), (
        "TaskAsSubtaskCoTBuilder should still can_handle when atomic_task is missing via task fallback"
    )
    slot = _slot(b, data_no_atomic).get("atomic_task", "")
    assert TASK_TXT in slot, f"TaskAsSubtaskCoTBuilder should use task as the CoT target, got {slot!r}"
    print("  ✓ TaskAsSubtaskCoTBuilder uses task text as the CoT target")

    # 3) 严格必需字段：这些 Builder 少了 atomic_task 就必须 can_handle=False
    #    （否则会在缺标注的数据上产出 "Subtask: " 这种空壳标签）。
    for name, cls in [
        ("SubtaskCoTBuilder", SubtaskCoTBuilder),
        ("BBoxSubtaskCoTBuilder", BBoxSubtaskCoTBuilder),
        ("SubtaskActionHintCoTBuilder", SubtaskActionHintCoTBuilder),
        ("MemorySubtaskCoTBuilder", MemorySubtaskCoTBuilder),
        ("HighLevelAtomicTaskCoTBuilder", HighLevelAtomicTaskCoTBuilder),
        ("AtomicTaskBaseSamplesBuilder", AtomicTaskBaseSamplesBuilder),
    ]:
        b = cls(**{**DEFAULTS, "embodiment_type": "r1"})
        assert not b.can_handle(data_no_atomic), (
            f"{name} must return can_handle=False when atomic_task is missing because it is required"
        )
    print(
        "  ✓ Subtask/BBoxSubtask/SubtaskActionHint/MemorySubtask/HighLevelAtomicTask/AtomicTaskBase strictly require atomic_task"
    )

    # 4) _override_command 钩子：AtomicTaskBase 用 atomic_task 顶替指令，
    #    HighLevel 用 high_level_instruction 顶替指令。
    atb = AtomicTaskBaseSamplesBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert atb._override_command(data_full) == ATOMIC_TXT, (
        f"AtomicTaskBaseSamplesBuilder._override_command should return atomic_task, got {atb._override_command(data_full)!r}"
    )
    hl = HighLevelAtomicTaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert hl._override_command(data_full) == HL_TXT, (
        f"HighLevelAtomicTaskCoTBuilder._override_command should return high_level_instruction, got {hl._override_command(data_full)!r}"
    )
    # 其它 Builder 默认不覆盖指令：返回 None，表示沿用上游给过来的 instructions。
    for cls in (BaseSamplesBuilder, SubtaskCoTBuilder, BBoxCoTBuilder):
        b = cls(**{**DEFAULTS, "embodiment_type": "r1"})
        assert b._override_command(data_full) is None, f"{cls.__name__} should not override command"
    print(
        "  ✓ _override_command: AtomicTaskBase=atomic_task, HighLevel=high_level_instruction, others=None"
    )

    # ---- can_handle_for_eval tests ----
    # 推理路径的字段判据：只检查“推理时真正要喂给模型的输入”，
    # CoT 目标字段（模型自己生成）一律不检查。
    # 下面逐个验证“有输入就能过、没输入就不过、但训练判据仍然更严”。
    print("\n" + "=" * 70)
    print("can_handle_for_eval Tests")
    print("=" * 70)

    # SubtaskCoTBuilder：推理不需要任何标注（空 dict 也放行），
    # 但训练判据 can_handle 仍要求 atomic_task。
    stcb = SubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert stcb.can_handle_for_eval({}), "SubtaskCoTBuilder.can_handle_for_eval({}) should be True"
    assert not stcb.can_handle({}), "SubtaskCoTBuilder.can_handle({}) must be False"
    # 顺便验证缺字段时 _populate_extra_samples 不会炸（退化成 "Subtask: "）。
    _out = {}
    stcb._populate_extra_samples({}, _out)
    assert _out["atomic_task"] == "Subtask: ", (
        f"SubtaskCoTBuilder._populate_extra_samples with missing key: {_out['atomic_task']!r}"
    )
    print(
        "  ✓ SubtaskCoTBuilder: can_handle_for_eval(no-annot)=True, can_handle(no-annot)=False, populate safe"
    )

    # BBoxCoTBuilder：推理时 bbox 是真实输入，所以 eval 判据同样要求“非空 bbox JSON”。
    bcb = BBoxCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert bcb.can_handle_for_eval({"bbox": '{"towel": [0.1, 0.2, 0.3, 0.4]}'}), (
        "BBoxCoTBuilder.can_handle_for_eval with valid bbox should be True"
    )
    assert not bcb.can_handle_for_eval({"bbox": "{}"}), (
        "BBoxCoTBuilder.can_handle_for_eval with empty bbox '{}' should be False"
    )
    assert not bcb.can_handle_for_eval({}), (
        "BBoxCoTBuilder.can_handle_for_eval without bbox should be False"
    )
    print("  ✓ BBoxCoTBuilder: can_handle_for_eval preserves non-empty JSON check")

    # BBoxSubtaskCoTBuilder：因为它把 bbox 从 eval_required_fields 里移除了，
    # 推理时不再要求 bbox（bbox 变成模型生成的 CoT），所以空 dict 也能命中。
    bscb = BBoxSubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert bscb.can_handle_for_eval({}), (
        "BBoxSubtaskCoTBuilder.can_handle_for_eval({}) should be True"
    )
    assert not bscb.can_handle({}), "BBoxSubtaskCoTBuilder.can_handle({}) must be False"
    _out2 = {}
    bscb._populate_extra_samples({}, _out2)
    assert _out2["atomic_task"] == "Subtask: ", (
        f"BBoxSubtaskCoTBuilder._populate_extra_samples missing keys: {_out2['atomic_task']!r}"
    )
    print("  ✓ BBoxSubtaskCoTBuilder: can_handle_for_eval(no-annot)=True, populate safe")

    # HighLevelAtomicTaskCoTBuilder：推理只要 high_level_instruction，
    # 但训练判据还额外要求 atomic_task。
    hlb = HighLevelAtomicTaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert hlb.can_handle_for_eval({"high_level_instruction": "Tidy"}), (
        "HighLevelAtomicTaskCoTBuilder.can_handle_for_eval with hl_inst should be True"
    )
    assert not hlb.can_handle_for_eval({}), (
        "HighLevelAtomicTaskCoTBuilder.can_handle_for_eval without hl_inst should be False"
    )
    assert not hlb.can_handle({"high_level_instruction": "Tidy"}), (
        "HighLevelAtomicTaskCoTBuilder.can_handle requires atomic_task too"
    )
    print(
        "  ✓ HighLevelAtomicTaskCoTBuilder: can_handle_for_eval checks only high_level_instruction"
    )

    # MemoryCoTBuilder：推理只要 memory（memory_update 由模型生成）。
    mcb = MemoryCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert mcb.can_handle_for_eval({"memory": "prev"}), (
        "MemoryCoTBuilder.can_handle_for_eval with memory should be True"
    )
    assert not mcb.can_handle_for_eval({}), (
        "MemoryCoTBuilder.can_handle_for_eval without memory should be False"
    )
    assert not mcb.can_handle({"memory": "prev"}), (
        "MemoryCoTBuilder.can_handle requires memory_update too"
    )
    print("  ✓ MemoryCoTBuilder: can_handle_for_eval checks only memory")

    # MemorySubtaskCoTBuilder：缺 atomic_task 时 populate 也不能炸。
    mscb = MemorySubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    _out3 = {}
    mscb._populate_extra_samples({"memory": "M"}, _out3)
    assert _out3["atomic_task"] == "Subtask: ", (
        f"MemorySubtaskCoTBuilder._populate_extra_samples missing atomic_task: {_out3['atomic_task']!r}"
    )
    print("  ✓ MemorySubtaskCoTBuilder._populate_extra_samples safe with missing atomic_task")

    # PlanStepCoTBuilder：推理只要 plan（当前步由模型生成）。
    psb = PlanStepCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert psb.can_handle_for_eval({"plan": "1. Pick\n2. Fold"}), (
        "PlanStepCoTBuilder.can_handle_for_eval with plan should be True"
    )
    assert not psb.can_handle_for_eval({}), (
        "PlanStepCoTBuilder.can_handle_for_eval without plan should be False"
    )
    assert not psb.can_handle({"plan": "1. Pick"}), (
        "PlanStepCoTBuilder.can_handle requires plan_step too"
    )
    print("  ✓ PlanStepCoTBuilder: can_handle_for_eval checks only plan")

    # 全部断言通过（期间任何一条失败都会抛 AssertionError 并中断）。
    print("\n" + "=" * 70)
    print("All cases OK.")
    print("=" * 70)
