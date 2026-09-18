# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# =============================================================================
# src/g05/data_processor/processor/mixture_processor.py — 多具身混合的“调度台”
#                                                       （MixtureProcessor）
# =============================================================================
#
# 【这个文件是什么】
#   它只定义一个类：``MixtureProcessor``，一个**注册表 + 分发器**，自己完全不处理样本。
#
#   一句话职责：把“每个具身（embodiment）各一个”的具体 processor 收集到字典里，然后对外
#     (1) 把 train() / eval() / set_normalizer_from_stats() 这类**全局动作**广播给每个子项；
#     (2) 用 ``processor["galaxea_r1pro"]`` 这种下标接口提供**按具身查表**的能力，
#         让调用方（数据集、推理器、部署脚本）自己决定“这个样本该用哪个 processor”。
#
#   两个必须先建立的认知：
#     · **组合，不是继承**：MixtureProcessor 不是 BaseProcessor 的子孙，
#       它内部持有若干个 BaseProcessor 实例。所以拿到它之后，
#       ``processor.shape_meta`` / ``processor.num_obs_steps`` 这类属性**并不存在**，
#       必须先取出子 processor 再访问（推理侧封装在 inferencer.resolve_processor()）。
#       判类型时也因此用 ``isinstance(processor, MixtureProcessor)`` 分支，
#       而不是判断“是不是 BaseProcessor”。
#     · **只分发，不加工**：它没有 preprocess() / postprocess() 实现，
#       调用这两个方法会直接抛 NotImplementedError（见文件中部两处 raise）。
#
# 【最重要的概念：embodiment_type ≠ 数据集别名】
#   注册表的 key 必须是**逻辑具身类型**（仓库里实际的取值见 configs/data/*.yaml 的
#   ``embodiment_type``：galaxea_r1pro / galaxea_r1lite / robotwin / libero / so100 /
#   Bridge / Droid_Franka ...），而不是外层数据源的名字（``embodiment_datasets`` 的
#   外层 key，即“别名”）。别名由使用者自由命名，只用于定位与日志。
#   两者之所以不能混用，是因为存在“多对一”：
#
#       数据源（configs/data/*.yaml）           embodiment_type（逻辑形态，决定流水线）
#       r1pro.yaml      ─┐
#       r1pro_wbc.yaml  ─┼──────►  "galaxea_r1pro"  → 一个 processor / 一套归一化参数
#       其他同形态数据源 ─┘
#
#   同一类型下的所有数据源必须共享同一套 processor 与归一化统计量（统计量在
#   mixture_lerobot_dataset.get_dataset_stats() 里同样按 embodiment_type 聚合），
#   所以路由键只能用它。build_processors() 还会显式校验 ``cfg.data.processors``
#   的 key 集合与 ``embodiment_datasets[*].embodiment_type`` 完全相等，
#   名字写错或漏一个都会当场报 ValueError（见 utils/data/processor_utils.py）。
#
# 【它在整条数据链路中的位置】
#
#   configs/data/<task>.yaml 的 embodiment_datasets + processors（每个具身一段配置）
#        │
#        │  入口：src/g05/utils/data/processor_utils.py 的 build_processors()
#        │    · cfg.data.processors 不存在 → 没有混合，直接 instantiate，
#        │                                  返回单个 BaseProcessor（不经过本类）
#        │    · 存在 → 逐 embodiment_type 合并配置、实例化、按类型放进 result 字典，
#        │              最后 ``return MixtureProcessor(result)``  ← 本类在这里诞生
#        ▼
#   MixtureProcessor（本文件）—— 只保存 {embodiment_type: BaseProcessor}，不处理样本
#        │
#        ├─ dataset.set_processor(mp)      把子 processor 下发给各子数据集
#        │    （mixture_lerobot_dataset.set_processor()：别名 → 类型 → mp[类型]）
#        ├─ mp.set_normalizer_from_stats(stats)  逐具身灌入统计量（stats 也按类型分键）
#        ├─ mp.train() / mp.eval()              逐具身切换增强模式（广播）
#        │
#        └─ 运行期取子 processor：
#             · 推理：inferencer.resolve_processor(mp, data)
#                      （兼容两种键名：raw_obs 的 "embodiment_type"、obs_dict 的 "embodiment"）
#             · 部署：scripts/serve_policy*.py、experiments/.../deploy_policy.py
#             · 评估：scripts/eval_open_loop.py 里 ``mixture_processor[emb_type]``
#             · 测试：tests/test_dataloader_batch.py 的 print_processor_structure()
#        ▼
#   子 processor.preprocess(sample)   ← 真正干活的地方
#        （base_processor.py / galaxea_cot_processor.py，见各自文件头的流水线说明）
#
# 【方法速查：哪些是广播，哪些是分发，哪些是取用】
#
#   __init__                     构造时读取各子项的 pad_token_id 并断言一致（拼批需要）
#   train() / eval()             广播：逐个调用子项的 train() / eval()（**不返回 self**）
#   set_normalizer_from_stats()  分发：按 embodiment_type 把对应统计量转交给子项
#   __getitem__(emb)             取用：``processor["galaxea_r1pro"]``，带友好报错的查表
#   preprocess() / postprocess() 占位：直接抛 NotImplementedError
#   processors                   公开属性 {embodiment_type: BaseProcessor}；调用方常直接用
#                                （例如 scripts/serve_policy.py 用 len(processor.processors)
#                                  判断是否为“多具身混合”，据此决定要不要 embodiment_type 字段）
#
# 【典型用法（照抄即可）】
#
#   # 1) 训练 / 评估脚本里构造与初始化
#   processor = build_processors(cfg)                 # 可能返回 MixtureProcessor
#   processor.set_normalizer_from_stats(dataset_stats)  # 逐具身灌统计量
#   processor.eval()                                  # 推理/评估侧（训练侧用 train()）
#   dataset.set_processor(processor)                  # 数据集按“别名 → 类型”取子 processor
#
#   # 2) 需要某个具身的维度契约 / 元信息时：先取子 processor
#   p = processor["galaxea_r1pro"] if isinstance(processor, MixtureProcessor) else processor
#   shape_meta, num_obs_steps = p.shape_meta, p.num_obs_steps
#
#   # 3) 单样本路由：由调用方决定用哪个子 processor
#   emb = sample["embodiment"]                      # 数据集/服务端填好的 embodiment_type
#   sample = processor[emb].preprocess(sample)      # 而不是 processor.preprocess(sample)
#
# 【新手最容易踩的坑】
#   1) 传了空字典：__init__ 里 ``next(iter({}))`` 会抛 StopIteration（不是友好的报错）。
#      正常路径不会发生——build_processors() 至少会塞进一个具身。
#   2) 以为能链式调用：BaseProcessor.train()/eval() 会 ``return self``，本类的同名方法
#      返回 None，``mp.train().eval()`` 会 AttributeError。
#   3) 拿 MixtureProcessor 当单处理器用：``mp.shape_meta`` / ``mp.num_obs_steps`` /
#      ``mp.normalizer`` 都会 AttributeError（属性在子 processor 上，本类只转发方法）。
#      正确姿势是 ``processor[emb]`` 或 inferencer.resolve_processor()。
#   4) 拿数据集别名去查表：别名若与 embodiment_type 不同名（例如别名写成 "r1pro_wbc"
#      而 type 是 "galaxea_r1pro"），``processor["r1pro_wbc"]`` 会 KeyError；
#      key 只能是 embodiment_type（见上文“多对一”示意）。
#   5) 以为 set_normalizer_from_stats() 会现算统计量：它只做**转发**；统计量必须先由
#      ``MixtureLerobotDataset.get_dataset_stats(processor)`` 算好，键也必须是
#      embodiment_type，否则报 KeyError 并打印当前可用的键列表。
#   6) 各具身 pad_token_id 配得不一致：构造时 AssertionError。该约束来自推理侧拼批
#      （整批共用一个 padding id），改 tokenizer 配置时别只改一个具身。
#   7) 忘了调用 train()/eval()：子 processor 的 ``is_train`` 是“未设置就抛 ValueError”
#      的刻意设计，混合模式下记得整体广播一次。
#   8) 文档口径：docs/deployment/serve_policy*.md 里“MixtureProcessor.preprocess() 按
#      data['embodiment'] 路由”属于**旧描述**；当前实现把路由责任交给调用方，
#      所以请按上面第 3、4 条的姿势自己取子 processor。
#
# 【相关文件与文档】
#   src/g05/data_processor/processor/__init__.py                 包入口（刻意不导入任何类）
#   src/g05/data_processor/processor/base_processor.py           BaseProcessor / ActionProcessor / FullProcessor
#   src/g05/data_processor/processor/galaxea_cot_processor.py    配置默认使用的子 processor
#   src/g05/utils/data/processor_utils.py                        build_processors()：配置 → 实例 + 打包成本类
#   src/g05/data/mixture_lerobot_dataset.py                      按别名→类型下发 processor、按类型聚合统计量
#   src/g05/models/g05/inferencer.py                             resolve_processor()：推理侧路由
#   scripts/serve_policy.py / scripts/serve_policy_mem.py        部署侧：分桶、逐请求 postprocess
#   scripts/finetune.py / scripts/eval_open_loop.py              训练 / 离线评估侧的调用姿势
#   experiments/robotwin/galaxeafm_policy/deploy_policy.py       按固定 key（"robotwin"）取子 processor
#   tests/test_dataloader_batch.py                               print_processor_structure() 遍历 processors
#   docs/deployment/serve_policy_zh.md                           mixture 部署流程（含上述过时描述）
# =============================================================================

# ABC：abc 模块提供的“抽象基类”标记。
#   注意本文件**没有**用 @abstractmethod —— 也就是说在 Python 语言层面这个类并不抽象，
#   真正阻止“直接拿它处理样本”的，是下面 preprocess()/postprocess() 里主动抛出的
#   NotImplementedError。这里继承 ABC 更多是表达“这是抽象层/基类接口”的语义。
from abc import ABC
# typing：仅用于类型标注。
#   Dict[str, BaseProcessor] 描述“字典 key 是具身名(str)，value 是某个具体 processor 实例”；
#   Any 用于 preprocess()/postprocess() 的宽松签名（它们只是占位，不会真正收到数据）。
from typing import Any, Dict

# BaseProcessor：所有具体处理器的基类。这里只用来做类型标注，
# 并提醒读者“子项是什么东西”——本类持有它、调用它的方法，但**不继承**它。
from g05.data_processor.processor.base_processor import BaseProcessor


class MixtureProcessor(ABC):
    """多具身（embodiment）processor 的注册表 / 分发器。

    它不是预处理流水线的一环：

    · **不处理样本**：preprocess() / postprocess() 只抛 NotImplementedError；
    · **只做两件事**：把全局动作广播给子 processor，以及按 embodiment_type 查表取子项。

    换句话说，它回答的问题是“这个样本该交给谁处理”，而“怎么处理”写在
    base_processor.py / galaxea_cot_processor.py 里的具体 processor 中。

    属性：
        processors (Dict[str, BaseProcessor]):
            {embodiment_type: 具体 processor 实例}，本类唯一的状态。
            外部代码可以直接读它来遍历全部具身（测试脚本与部署脚本都这么用）。
        pad_token_id (int):
            全体子 processor 约定的同一个 padding token id，
            取自其中一个子项并要求其余子项与之相等（构造时断言）。
            推理侧把多个请求拼成一个 batch 时，整批只用一个 padding id，
            因此各具身必须一致，否则 mask 语义会错。

    异常约定：
        传空字典 → StopIteration（``next(iter({}))`` 取不到元素，见文件头“坑 1”）；
        pad_token_id 不一致 → AssertionError。
    """

    def __init__(
        self,
        embodiment_processors: Dict[str, BaseProcessor],
    ):
        """构造注册表（通常由 build_processors() 调用，业务代码一般不自己 new）。

        参数：
            embodiment_processors: {embodiment_type: 已实例化的具体 processor}。
                key 必须是**逻辑具身类型**（"galaxea_r1pro" / "galaxea_r1lite" / "robotwin" ...）；
                由 src/g05/utils/data/processor_utils.py 的 build_processors() 组装：
                它按 cfg.data.embodiment_datasets 里的 embodiment_type 去
                cfg.data.processors 取配置，逐具身实例化（含 key 集合一致性校验）后塞进来。

        异常：
            StopIteration: 空字典（等于“一个具身都没有”），报错信息不友好，见文件头坑 1。
            AssertionError: 各子 processor 的 pad_token_id 不一致。
        """
        # 以“逻辑具身类型（embodiment_type）”为 key 保存子 processor。
        # 外层数据集名（别名，embodiment_datasets 的外层 key）只是**数据源的名字**，
        # 绝不能拿来做 processor 路由：同一具身类型往往对应多个数据源，
        # 而这些数据源必须共享同一套 processor / 归一化参数
        # （统计量也按 embodiment_type 聚合，见 mixture_lerobot_dataset.get_dataset_stats）。
        self.processors = embodiment_processors
        # 取“第一个子 processor”的 pad_token_id 作为全批约定值。
        # 为什么取第一个就够：紧接着的循环会断言所有子项都相同，
        # 因此“第一个”与“任意一个”等价；这样写只是免得再多一次字典查找。
        # （若传空字典，这里的 next(iter({})) 会抛 StopIteration，见文件头坑 1。）
        self.pad_token_id = embodiment_processors[next(iter(embodiment_processors))].pad_token_id
        # 逐个断言 pad_token_id 一致，并把出问题的具身名写进报错信息，便于定位。
        # 写成显式 for 循环而不是 all(...)：出错时能直接指出是哪个 embodiment_type 不一致。
        # 该约束的来源见文件头“坑 6”：推理侧拼批时整批共用一个 padding id。
        for emb_type, processor in self.processors.items():
            assert processor.pad_token_id == self.pad_token_id, (
                f"Pad token id mismatch for embodiment_type {emb_type}."
            )

    def train(self):
        """把所有子 processor 切到训练模式（广播；无返回值）。

        与 BaseProcessor.train() 的差异要留意：
          · BaseProcessor.train() 会 ``return self``，支持 ``ds.set_processor(p.train())`` 这类链式写法；
          · 本方法不返回任何东西（None），所以 ``mp.train().eval()`` 会直接 AttributeError。

        为什么必须逐具身切换：每个具身都有自己的图像增强栈（train_transforms / val_transforms），
        让它们一起切才能保证金标准一致；否则同一个 batch 里不同具身的预处理口径会不一样。
        """
        # 逐个转发；顺序无关（各子 processor 互相独立）。
        for processor in self.processors.values():
            processor.train()

    def eval(self):
        """把所有子 processor 切到评估模式（广播；无返回值，注意事项同 train()）。

        调用时机（参见 scripts/eval_open_loop.py、scripts/serve_policy.py）：
        ``processor = build_processors(cfg)`` 之后、``set_processor`` / 首次取数之前，
        推理与离线评估都先调用本方法一次。
        """
        # 逐个转发；顺序无关。
        for processor in self.processors.values():
            processor.eval()

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """未实现：本类不做单样本预处理。

        调用方必须先按样本所属具身取出子 processor，例如
        ``processor[sample["embodiment"]].preprocess(sample)``；
        推理侧已把这个动作封装成
        :func:`g05.models.g05.inferencer.resolve_processor`。

        （文档提示：docs/deployment/serve_policy*.md 里“MixtureProcessor.preprocess()
        按 data['embodiment'] 路由到子 processor”是旧描述，当前实现把路由责任交给调用方，
        本方法只有下面这句异常。）
        """
        raise NotImplementedError(
            "MixtureProcessor is a registry/manager only. "
            "Bind or call a concrete per-embodiment processor instead."
        )

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """未实现：本类不做单样本后处理（反归一化 / 逆变换）。

        与 preprocess() 同理，必须先把模型输出交回**当初处理该样本的那个子 processor**，
        例如动态拼批场景：每个请求记住自己的 sub_processor，forward 结束后逐个调
        ``sub_processor.postprocess(item)``（见 inferencer.py 的 _postprocess_single()；
        部署脚本 scripts/serve_policy*.py 自己也不直接调 postprocess，而是整段交给
        PolicyInferencer）。
        这一点在混合批里尤其重要：不同具身的归一化参数与变换链不同，用错 processor
        会把动作还原成物理量之外的值，而且不一定报错、只是结果离谱。
        """
        raise NotImplementedError(
            "MixtureProcessor is a registry/manager only. "
            "Bind or call a concrete per-embodiment processor instead."
        )

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any]):
        """把统计量按具身分发给每个子 processor（训练/推理前必须调用一次）。

        参数：
            dataset_stats: 形如 ``{embodiment_type: {"action": {...}, "state": {...}}}``，
                由 MixtureLerobotDataset.get_dataset_stats() 计算（各数据源按权重聚合后得到）。
                注意 key 同样是 embodiment_type，不是数据源别名。
                子 processor 的 set_normalizer_from_stats() 接受的正是“单个具身的那一份”。

        异常：
            KeyError: 某个具身缺少统计量；报错信息里附带当前可用的 key 列表，
                方便对照拼写（常见原因：换数据集后统计量文件没重算 / key 用了别名）。
            AssertionError: 某个子 processor 在校验统计量时断言失败；
                这里会补上 ``Embodiment type '<名字>':`` 前缀再抛出，
                避免多具身一起训练时分不清是谁炸的。

        副作用：每个子 processor 内部据此创建各自的 LinearNormalizer（见 base_processor.py）。
        """
        # 逐个具身转发：数据集统计量是“按类型”组织的，本方法就是别名无关的那一层胶水。
        for emb_type, processor in self.processors.items():
            # 缺统计量直接失败：子 processor 拿不到对应 key 的统计量时，
            # 归一化会静默退化或用错参数，与其训练一阵子才发现，不如在初始化阶段就报出来。
            if emb_type not in dataset_stats:
                raise KeyError(
                    f"No stats found for embodiment_type '{emb_type}'. "
                    f"Available stats keys: {list(dataset_stats.keys())}"
                )
            stats = dataset_stats[emb_type]
            try:
                # 注意：这里只做转发，不负责计算统计量，也不改 key 的名字。
                processor.set_normalizer_from_stats(stats)
            except AssertionError as ex:
                # 补上具身名再抛出：子 processor 的断言信息本身不带具身上下文，
                # 混合训练时很难判断是哪个形态出的问题。
                # ``from ex`` 保留原始异常链，traceback 里仍能看到最底层的原因。
                raise AssertionError(f"Embodiment type '{emb_type}': {ex}") from ex

    def __getitem__(self, emb: str):
        """按 embodiment_type 取子 processor：``processor["galaxea_r1pro"]``。

        这是本类最常用的接口：功能上等价于 ``processor.processors[emb]``，
        但多了一层“查不到就列出可用名字”的友好报错，拼错具身名时能立刻看出问题。

        参数：
            emb: 逻辑具身类型（不是数据源别名）。

        返回：
            对应的具体 processor 实例（BaseProcessor 子类）。

        异常：
            KeyError: 该具身名不在注册表中（消息里带排序后的可用 key 列表）。
        """
        if emb in self.processors:
            return self.processors[emb]
        # 主动抛出而非让 dict 自己报错，是为了把“当前有哪些具身可用”一并打印出来——
        # 在多具身混合的配置里，这个提示能省掉不少排查时间。
        raise KeyError(
            f"embodiment_type '{emb}' not found in processors. "
            f"Available: {sorted(self.processors.keys())}"
        )
