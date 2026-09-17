# =============================================================================
# src/g05/data/mixture_lerobot_dataset.py — 混合数据集 MixtureLerobotDataset
# =============================================================================
#
# 【这个文件是什么】
#   “混合数据集” = 把 1..N 个 embodiment（机器人形态）× 每个形态下 1..N 个
#   LeRobot 数据集目录，组装成一个普通的 torch.utils.data.Dataset。
#   它自己不解码 parquet / mp4，只做“上层调度与聚合”，负责 5 件事：
#     1) 建子数据集：按 configs/data/<task>.yaml 的 embodiment_datasets，为每个
#        (embodiment, dataset_group) 实例化一个 BaseLerobotDataset 子类；
#     2) 管索引    ：对外只暴露一条连续下标空间，内部两级映射到
#                    “哪个子数据集、第几条样本”；
#     3) 管长度    ：真实长度(actual_lengths) → 逻辑长度(effective_lengths)，
#                    受“加权采样 / overfit / 跨节点分片”三处影响；
#     4) 补标签    ：给样本补 embodiment / embodiment_type / dataset_locator；
#     5) 算统计量  ：聚合各子数据集的归一化统计量（min/max/mean/std/分位数）。
#
# 【它在数据链路中的位置】
#
#   configs/data/<task>.yaml          （声明数据是什么、在哪、权重多少）
#        │  instantiate_dataset()      src/g05/utils/data/processor_utils.py
#        ▼
#   MixtureLerobotDataset ← 本文件
#        │  · 逐 embodiment、逐 dataset_group new 出子数据集
#        │  · 汇总权重 / 长度 / 统计量
#        ▼
#   BaseLerobotDataset 子类（一个子数据集 = 一个实例，对应一个 group 的一组目录）
#     ├─ GalaxeaLerobotDataset           src/g05/data/galaxea_lerobot_dataset.py
#     ├─ BaseLerobotDatasetV3            src/g05/data/base_lerobot_datasetV3.py
#     ├─ DroidLerobotDataset             src/g05/data/droid/droid_lerobot_dataset.py
#     └─ SO100CanonicalLerobotDatasetV3  src/g05/data/so100_canonical_dataset.py
#        │  · 按 shape_meta 决定读哪些帧、怎么切维、归一化、拼 samples 模板
#        ▼
#   MultiLeRobotDataset               src/g05/data/lerobot/lerobot_dataset{,_v3}.py
#        │  · 把同一 group 里的多个目录“首尾拼接”成一个索引空间
#        ▼
#   parquet（action / state 等数值列） + mp4 / png（相机画面）
#
#   训练侧的使用顺序（scripts/finetune.py）：
#     instantiate_dataset(cfg, is_training_set=True/False)      # 构造本类
#       → get_dataset_stats(...) → set_processor(...)           # 统计量 + 处理器
#       → (可选) enable_overfit(...)                            # 小样本过拟合自检
#       → 每个 epoch 调 set_epoch(epoch)                        # 加权下采样换子集用
#       → DataLoader 调 __getitem__(idx)                        # 真正取数
#     另外两个方法 sync_weights_for_sharding() / cap_length_for_sharding() 属于
#     “按节点分片数据”的实验路径（对应配置项 shard_datasets_by_node，默认 false），
#     当前训练脚本里没有调用点，读代码时知道它们存在即可。
#
# 【新手先记住这 4 个概念】
#
#   (1) 别名 vs embodiment_type
#         别名（embodiment_datasets 的外层 key，例如 galaxea_r1pro）只是本文件内的
#         “数据源名字”，用于日志与定位；
#         embodiment_type 才是“机器人形态”标签，决定用哪套 processor、模型看到什么形态。
#         两者不是一回事，但别名的 type 必须能在 processors 里找到同名条目
#         （build_processors 会校验，写错直接报错）。
#   (2) dataset_group 与 weight
#         一个 embodiment 下可以有多个 group，每个 group = {weight, dataset_dirs}。
#         同一 group 内的多个目录被顺序拼接（视作同一批数据）；
#         group 之间靠 weight 表达“采样偏好”与“统计量聚合权重”。
#   (3) 三种“长度”
#         actual_lengths      子数据集真实样本数 len(ds)（已含训练/验证切分的结果）
#         effective_lengths   逻辑采样长度，默认 == actual_lengths；
#                             use_weight_for_sampling / 分片 cap 会改写它
#         _overfit_len        overfit 模式下的长度，优先级最高（直接覆盖 __len__）
#         ⚠️ 三处改长度的逻辑互相覆盖，顺序敏感：先权重归一化 → 再 overfit → 最后分片 cap。
#   (4) 两级下标
#         __getitem__(idx) 的 idx 是“混合数据集全局下标”，先定位到某个子数据集
#         (dataset_idx)，再在该子数据集内部取 local_idx；
#         开 use_weight_for_sampling 时，local_idx 还要经 _map_weighted_local_index()
#         映射成子数据集里的真实下标。
#
#       全局下标 idx 的解析示意（lengths = [100, 50, 30]）
#         effective_starts = [  0, 100, 150]
#         effective_ends   = [100, 150, 180]   # 每个子数据集的右边界（开区间）
#         dataset_idx = searchsorted(ends, idx, side="right")
#         local_idx   = idx - starts[dataset_idx]
#
# 【相关文件】
#   configs/data/r1pro.yaml                 实例：一个 embodiment、多个 group 的写法
#   configs/data/parts_meta/*.yaml          各 part 的宽度定义（merger 用）
#   src/g05/utils/data/processor_utils.py   instantiate_dataset / build_processors
#   src/g05/utils/data/normalizer.py        统计量落盘与加载 dataset_stats.json
#   src/g05/data/base_lerobot_dataset.py    子数据集基类（读帧、切维、归一化）
#   docs/data/schema_zh.md                  shape_meta 字段说明
#   docs/data/samples_builders_zh.md        samples 模板与 CoT slot
# =============================================================================

# 标准库导入。实际用到的是：math（互质步长）、os / sys（进度条开关的环境判断）、
# logging / defaultdict（日志与统计量分组）；gc 与 re 目前在本文件里没有使用
# （历史遗留，保留不影响行为），初读时不用在这里纠结。
import gc
import logging
import math
import os
import sys
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set

# numpy：这里主要用它的前缀和/二分查找（cumsum / searchsorted / concatenate）来做下标定位。
import numpy as np

# 每个模块自带一个 logger 是 Python 的惯例：日志里能看到消息来自哪个文件（__name__）。
logger = logging.getLogger(__name__)

import torch
# tqdm.auto：在 Jupyter / 终端里都能显示进度条，加载大集群数据时很关键。
from tqdm.auto import tqdm
# OmegaConf：读别的 YAML 配置（load_embodiment_config 里的 config 继承）。
from omegaconf import OmegaConf
# Hydra 会把进程的工作目录切到输出目录，相对路径必须先转成“启动时的工作目录”下的绝对路径。
from hydra.utils import to_absolute_path

import re

# 导入 g05.data 这个包本身会连带加载包内的数据集实现（__all__ 是它的导出清单）。
# 本行在本文件里没有直接使用 __all__，作用是保留这条包级依赖关系。
from g05.data import __all__
# 按“模块路径字符串”动态 import 并取属性，用来实例化配置里写的 type。
from g05.utils.common.import_utils import get_obj_from_str
# 子数据集基类：用于类型标注与静态方法里的能力判断。
from g05.data.base_lerobot_dataset import BaseLerobotDataset
# 混合处理器：按 embodiment_type 管理各形态的 processor，本类只负责路由。
from g05.data_processor.processor.mixture_processor import MixtureProcessor


def load_embodiment_config(emb_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    读取并合并一个 embodiment 的配置（支持从另一个 data 配置继承）。

    为什么需要它：不同的 embodiment 配置往往只差几个键（例如 dataset_dirs、
    embodiment_type），其他像 shape_meta / transforms / shape 声明完全一样。
    与其复制粘贴（改一处忘一处），不如在 YAML 里写一行
        config: configs/data/xxx.yaml
    表示“先继承这份配置，再用本地写的键覆盖它”。

    合并规则：
        * emb_cfg 里没有 "config" → 原样返回（最常见的情况，不读任何文件）；
        * 有 "config"             → 读该 YAML 作为 base，再叠加 emb_cfg（emb_cfg 优先，
                                   同键覆盖、base 里多余的同名键被覆盖掉）；
        * 子数据集类由 `type` 指定；若 emb_cfg 没写 `type`，则取 base 配置里的
          `_target_`（Hydra 风格配置里用 `_target_` 声明实现类）作为 type。

    Args:
        emb_cfg: 来自 configs/data/<task>.yaml 里 `embodiment_datasets.<别名>` 的配置。

    Returns:
        合并后的普通 dict，可以直接作为子数据集的构造参数（**kwargs）使用。
    """
    # 先复制一份：后面会 pop() 掉 config / _target_ 等键，
    # 不能就地修改调用方传进来的配置对象（Hydra 的 DictConfig 尤其不能乱改）。
    emb_cfg = dict(emb_cfg)

    if "config" not in emb_cfg:
        # 没有继承声明：原样返回，零额外开销。
        return emb_cfg

    # 用 pop 而不是 get：合并后的配置要能直接当构造参数用，不能残留 "config" 这个键。
    config_path = emb_cfg.pop("config")
    # Convert to absolute path relative to Hydra's original working directory
    # （Hydra 会切换 cwd，所以相对路径必须按“启动时的工作目录”解析成绝对路径。）
    config_path = to_absolute_path(config_path)
    base_cfg = OmegaConf.load(config_path)
    # resolve=True：把 YAML 里的 ${...} 插值全部展开，得到纯 Python 容器
    # （否则后面做 dict 合并时拿到的还是 OmegaConf 节点，行为不一致）。
    base_cfg = OmegaConf.to_container(base_cfg, resolve=True)

    # Extract type from _target_
    # 子数据集类名：本地没写 type 就从 base 的 `_target_` 继承；
    # 无论走哪条分支，都要把 `_target_` 从 base 里摘掉（它不是构造参数）。
    if "type" not in emb_cfg and "_target_" in base_cfg:
        emb_cfg["type"] = base_cfg.pop("_target_")
    else:
        base_cfg.pop("_target_", None)

    # Merge: base_cfg first, then emb_cfg overrides
    # 浅合并：base 在前、本地在后 → 同键以本地为准（这就是“本地覆盖继承”）。
    merged = {**base_cfg, **emb_cfg}
    return merged


def normalize_dataset_weights(lengths: List[int], given_weights: List[float]) -> List[float]:
    """把 dataset_group 的“原始相对权重”换算成“保持总样本量不变”的缩放因子。

    原始权重表达的是“相对偏好”（例如 1:3），但它的绝对大小会顺手把总长度也放大/缩小。
    本函数求一个统一系数 k，使得

        sum_i lengths[i] * normalized_weights[i] == sum_i lengths[i]

    也就是：加权后的总逻辑长度与未加权时一致，只有 group 之间的“比例”变了。
    这样采样长度与统计量聚合的权重口径统一，且不会因为改权重而意外改变一个 epoch 的步数。

    Example:
        lengths=[100, 100], raw=[1, 3] -> normalized=[0.5, 1.5]
        有效长度从 [100, 100] 变成 [50, 150]，总长仍是 200，采样比变成 1:3。

    Args:
        lengths: 各 dataset_group 的真实样本数（子数据集 len(ds)）。
        given_weights: 对应的原始权重（YAML 里写的 weight）。

    Returns:
        归一化后的权重列表（与输入等长）。注意它是“缩放因子”而不是概率，
        所以单个值可能大于 1、也可能小于 1。
    """
    if len(lengths) != len(given_weights):
        raise ValueError("lengths and given_weights must have the same length")
    total_len = sum(lengths)
    # denom = Σ(len_i × w_i) 是“按原始权重的加权总长”，把它的尺度拉回 total_len 即可。
    denom = sum(l * w for l, w in zip(lengths, given_weights))
    if denom <= 0:
        # denom ≤ 0 说明权重全为 0/负数（或数据全空），此时无法定义归一化系数，
        # 直接报错比后面出现 NaN 权重、长度算成 0 更容易定位。
        raise ValueError("Invalid weight normalization denominator (<= 0)")
    k = total_len / denom
    return [k * w for w in given_weights]


def _should_show_loading_bar() -> bool:
    """判断当前进程要不要显示 tqdm 加载进度条。

    只有“主进程 + 交互式终端”才显示：
      * RANK / LOCAL_RANK 不是 0（或 None）→ 说明是 DDP 的其他 worker，
        每个 rank 都画进度条会把日志刷得没法看，所以直接关掉；
      * sys.stderr 不是 tty（被重定向到日志文件、或被集群调度器托管）→ 进度条靠
        \\r 原地刷新，写进文件会变成一坨转义内容，也关掉。
    """
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    return (rank in (None, "0")) and (local_rank in (None, "0")) and sys.stderr.isatty()


def _stats_write(message: str, color: Optional[str] = None) -> None:
    """输出统计量相关的信息行，可选着色，并且不会破坏进度条。

    用 tqdm.write 而不是 print / logger.info：
    tqdm.write 会先擦掉当前进度条、打印内容、再把进度条画回来，
    否则统计日志会和进度条互相覆盖、终端输出变成乱码。
    termcolor 没装（或着色失败）时退化成纯文本，不影响功能。
    """
    try:
        from termcolor import colored

        output = colored(message, color) if color else message
    except Exception:
        # 着色只是锦上添花：任何异常（未安装 termcolor 等）都不能影响统计流程。
        output = message

    tqdm.write(output)


class MixtureLerobotDataset(torch.utils.data.Dataset):
    """混合数据集：把多个 embodiment、多个数据目录合成一个 Dataset。

    本类只做“调度与聚合”，不关心 parquet / 视频具体怎么读（那是每个子数据集的活）。
    对外就是一个普通的 torch Dataset：
        len(ds)                  逻辑样本数（受加权采样 / overfit 影响）
        ds[i]                    一条已经过 processor 预处理、可直接 collate 的样本
        ds.get_item_with_meta(i) 同上，并额外返回 (dataset_idx, local_idx) 便于定位

    构造方式（正常都走 Hydra，不要手动 new）：
        cfg.data._target_ = g05.data.mixture_lerobot_dataset.MixtureLerobotDataset
        train_dataset = instantiate_dataset(cfg, is_training_set=True)
        eval_dataset  = instantiate_dataset(cfg, is_training_set=False)

    关键内部状态（调试时最常看的几个）：
        self.embodiments      List[str]   每个子数据集的别名（同一 embodiment 有多个
                                          group 时会重复出现，与 datasets 一一对应）
        self.datasets         List[Base]  子数据集实例
        self.embodiments2types Dict[str, str]  别名 → embodiment_type
        self.weights          List[float] 采样 / 统计量聚合用的权重（可能已归一化）
        self.actual_lengths   List[int]   各子数据集真实样本数
        self.effective_lengths List[int]  逻辑采样长度（默认等于 actual_lengths）
        self.effective_starts / effective_ends  np.ndarray 逻辑长度的前缀和索引表

    生命周期（谁在什么时候调用本类的哪些方法，见 scripts/finetune.py）：
        __init__ → (分片对齐) → get_dataset_stats → set_processor
                 → (enable_overfit) → 每 epoch set_epoch → __getitem__
    """

    def __init__(
        self,
        embodiment_datasets: Dict[str, Any],
        use_weight_normalization: bool,
        action_size: int,
        past_action_size: int,
        val_set_proportion: float,
        is_training_set: bool,
        obs_size: int = 1,
        obs_stride_second: float = 0.0,
        use_weight_for_sampling: bool = False,
        n_datasets: Optional[int] = None,
        load_images: Optional[bool] = None,
        in_memory: bool = False,
    ):
        """构造混合数据集：把配置里的所有子数据集都实例化出来。

        参数（对应 configs/data/<task>.yaml 的顶层字段）：
            embodiment_datasets   Dict[别名, 子数据集配置]。每份子配置里的键分三类：
                                    · 数据来源：type / config / dataset_groups /
                                                lerobot_ds_version
                                    · 形态与布局：embodiment_type / shape_meta
                                    · 可选覆盖：action_size / obs_size /
                                                val_set_proportion / override_fps ...
            use_weight_normalization
                                  是否把 weight 归一化成“保持总样本量”的因子，
                                  见 normalize_dataset_weights()。
            action_size           动作 chunk 长度 H（预测未来多少步），下发给子数据集。
            past_action_size      历史动作步数；子数据集当前断言必须为 0。
            val_set_proportion    验证集比例（每个子数据集按自己的 episode 数切分）。
            is_training_set       True=取训练切分，False=取验证切分；同一份配置两种用法。
            obs_size              观测帧数 T（state 与 image 共用同一帧数）。
            obs_stride_second     相邻观测帧的时间间隔（秒），仅 obs_size > 1 时有意义。
            use_weight_for_sampling
                                  True 时用 weight 缩放“逻辑采样长度”
                                  （weight < 1 相当于对该 group 下采样），
                                  见 _compute_effective_lengths()。
            n_datasets            调试开关：每个 group 最多取前 n 个数据目录
                                  （等价 scripts/run/finetune.sh --max_datasets N 在配置层做的
                                    同样截断，只是这里发生在构造数据集时）。
            load_images           统一覆盖子数据集的 load_images（None = 由各自配置决定）；
                                  显式 False 可跳过视频解码（例如只算统计量时）。
            in_memory             True 时让子数据集把数值列常驻内存：取数更快、更吃内存。

        注意 action_size / past_action_size / obs_size / obs_stride_second /
        val_set_proportion 在本类是“全局默认值”：子数据集配置里若写了同名字段，
        以子数据集自己的为准（见下面 ds_* = emb_ds_cfg.pop(..., 全局默认值) 几行）。
        """
        # ---- 0) 参数校验：n_datasets 表示“每个 group 取前 N 个目录”，必须为正整数 ----
        if n_datasets is not None:
            n_datasets = int(n_datasets)
            if n_datasets <= 0:
                raise ValueError(f"n_datasets must be a positive integer or None, got {n_datasets}")

        # ---- 1) 内部状态初始化（主循环里逐步填充）----
        self.n_datasets = n_datasets
        self.embodiments = []  # 每个子数据集的别名（同一 embodiment 多个 group 会重复出现）
        self.weights = []  # 与 self.datasets 一一对应的权重
        self.datasets: List[BaseLerobotDataset] = []  # 子数据集实例
        self.embodiments2types = {}  # 别名 → embodiment_type（processor / 模型标签用它）

        # Pre-load all embodiment configs once (avoids double YAML I/O for configs with "config" key)
        # ---- 2) 预读所有 embodiment 配置 ----
        # 为什么要单独先跑一遍？因为进度条的总数 total_groups 需要提前知道；
        # 而 load_embodiment_config() 可能真的要读 YAML（config 继承），
        # 先读一次缓存到 emb_configs，主循环里就不必重复做 I/O。
        emb_configs: Dict[str, Any] = {}
        total_groups = 0
        for emb in embodiment_datasets:
            emb_ds_cfg = load_embodiment_config(embodiment_datasets[emb])
            emb_configs[emb] = emb_ds_cfg
            dataset_groups = emb_ds_cfg.get("dataset_groups")
            if dataset_groups is not None:
                # 进度条按“dataset_group 个数”计数（一个 group = 一个子数据集）。
                total_groups += len(dataset_groups)

        # 只在主进程 + 终端里画进度条（多卡训练时其他 rank 会刷屏，见函数注释）。
        show_loading_bar = _should_show_loading_bar()

        # ---- 3) 主循环：逐 embodiment → 逐 dataset_group → 实例化子数据集 ----
        # 循环结束时，self.embodiments / self.weights / self.datasets 三个列表等长，
        # 第 i 项共同描述第 i 个子数据集。
        with tqdm(
            total=total_groups,
            desc="Loading datasets",
            dynamic_ncols=True,
            leave=False,
            disable=not show_loading_bar,
        ) as progress:
            for emb in embodiment_datasets:
                # 浅拷贝出本轮要改的配置：下面会连续 pop() 掉 dataset_groups / type /
                # embodiment_type 等键，必须复制，否则第二次用到同一 embodiment 时缓存已被污染。
                # （浅拷贝足够：pop 只删顶层键，不会改到嵌套对象。）
                emb_ds_cfg = dict(emb_configs[emb])  # shallow copy; .pop() below mutates it
                # dataset_groups：本 embodiment 的数据来源列表，每个元素 = {weight, dirs}。
                # type：子数据集实现类的完整路径，交给 get_obj_from_str 动态导入。
                dataset_groups = emb_ds_cfg.pop("dataset_groups")
                dataset_type = emb_ds_cfg.pop("type")
                if dataset_groups is None:
                    # 配置里该 embodiment 没有 group = 本次不提供数据（例如临时停用），跳过。
                    continue

                # embodiment_type 是“机器人形态”标签：决定用哪套 processor、
                # 模型侧如何区分形态。这里统一转成 str，保证标签类型稳定
                # （YAML 里手写成数字/布尔时也不会传出奇怪的类型）。
                emb_type_raw = emb_ds_cfg.pop("embodiment_type")
                if not isinstance(emb_type_raw, str):
                    emb_type_raw = str(emb_type_raw)
                self.embodiments2types[emb] = emb_type_raw

                # Per-dataset params with global defaults
                # 子数据集私有参数优先；没写就用本类构造参数（“顶层字段 = 全局默认值”就在这里生效）。
                # 注意用 pop 取走：这些键后面会以显式关键字参数传给子数据集构造函数，
                # 若同时留在 **emb_ds_cfg 里会触发 “got multiple values for keyword argument”。
                ds_action_size = emb_ds_cfg.pop("action_size", action_size)
                ds_past_action_size = emb_ds_cfg.pop("past_action_size", past_action_size)
                ds_obs_size = emb_ds_cfg.pop("obs_size", obs_size)
                ds_obs_stride_second = emb_ds_cfg.pop("obs_stride_second", obs_stride_second)
                ds_val_set_proportion = emb_ds_cfg.pop("val_set_proportion", val_set_proportion)
                # override_fps：模型控制频率，只影响 sample["frequency"]（供 action tokenizer
                # 做时间编码），不改变“读哪一帧”的计算；None = 用数据集自带的 data_fps。
                ds_override_fps = emb_ds_cfg.pop("override_fps", None)

                # 一个 embodiment 可以有多个 group（例如“主力数据 + 少量补充数据”）；
                # 每个 group 单独实例化一个子数据集，并各自记录一条权重。
                for group_idx, group in enumerate(dataset_groups):
                    # group 是 OmegaConf 节点：weight 是相对权重，dataset_dirs 是目录列表。
                    weight = group.weight
                    dataset_dirs = list(group.dataset_dirs)  # 转成普通 list，便于切片
                    original_n_dirs = len(dataset_dirs)
                    if self.n_datasets is not None:
                        # 调试用：每个 group 只保留前 n 个目录，能显著缩短启动时间
                        # （配合 finetune.sh --max_datasets N 使用）。
                        if len(dataset_dirs) > self.n_datasets:
                            logger.info(
                                "Limiting dataset_dirs for %s group %s: %s -> %s",
                                emb,
                                group_idx,
                                len(dataset_dirs),
                                self.n_datasets,
                            )
                        dataset_dirs = dataset_dirs[: self.n_datasets]

                    # 进度条模式显示简短后置信息；非进度条模式打一条完整日志，便于事后排查。
                    if show_loading_bar:
                        progress.set_postfix_str(
                            f"{emb} g{group_idx} dirs={len(dataset_dirs)}",
                            refresh=False,
                        )
                    else:
                        logger.info(
                            "Loading dataset for embodiment %s group %s with %s dirs%s",
                            emb,
                            group_idx,
                            len(dataset_dirs),
                            (
                                f" (from {original_n_dirs})"
                                if len(dataset_dirs) != original_n_dirs
                                else ""
                            ),
                        )

                    # 把“本次 group 要用的目录”写回配置，并做可选的全局覆盖。
                    emb_ds_cfg["dataset_dirs"] = dataset_dirs
                    # Global load_images override: mixture-level setting wins over per-embodiment config
                    # （mixture 层显式指定的 load_images 优先级高于 embodiment 自己的配置。）
                    if load_images is not None:
                        emb_ds_cfg["load_images"] = load_images
                    if in_memory:
                        # 把数值列常驻内存：省掉反复读 parquet 的开销，代价是内存占用。
                        emb_ds_cfg["in_memory"] = True
                        logger.info(f"[in_memory] Injecting in_memory=True for {emb}")
                    # 动态实例化子数据集：
                    #   dataset_type 是类路径字符串（如 g05.data.galaxea_lerobot_dataset
                    #   .GalaxeaLerobotDataset），get_obj_from_str 负责 import + 取类；
                    #   **emb_ds_cfg 里带着 shape_meta / lerobot_ds_version / 其余子类参数，
                    #   显式列出的那些（action_size 等）已在上面 pop 走，不会重复传参。
                    dataset = get_obj_from_str(dataset_type)(
                        **emb_ds_cfg,
                        action_size=ds_action_size,
                        past_action_size=ds_past_action_size,
                        obs_size=ds_obs_size,
                        obs_stride_second=ds_obs_stride_second,
                        val_set_proportion=ds_val_set_proportion,
                        override_fps=ds_override_fps,
                        is_training_set=is_training_set,
                    )
                    self.embodiments.append(emb)
                    self.weights.append(weight)
                    self.datasets.append(dataset)
                    progress.update(1)

        # ---- 4) 收尾：长度、权重、索引表 ----
        self.is_training_set = bool(is_training_set)
        self.in_memory = in_memory
        self.use_weight_for_sampling = bool(use_weight_for_sampling)
        # sampling_epoch 供“加权下采样换子集”用，由 set_epoch() 在每个 epoch 更新。
        self.sampling_epoch = 0
        # 真实样本数：子数据集内部已经做完了训练/验证切分，所以 len(ds) 就是最终可用条数。
        self.actual_lengths = [len(ds) for ds in self.datasets]

        if in_memory:
            # in_memory 常见死法是 OOM：这里打一条内存体检日志，便于对上机器的 RAM 余量。
            total_rows = sum(self.actual_lengths)
            import psutil
            rss_gb = psutil.Process().memory_info().rss / 1024**3
            logger.info(
                f"[in_memory] MixtureLerobotDataset summary: "
                f"{len(self.datasets)} sub-datasets, {total_rows:,} total rows, "
                f"process RSS={rss_gb:.1f} GB"
            )
        # 原始权重留档：sync_weights_for_sharding() 需要拿“未归一化”的权重按全局比例重算，
        # 而 self.weights 马上就要被替换成归一化后的值。
        self._raw_weights = list(self.weights)

        if use_weight_normalization:
            # 归一化只改变 group 之间的相对比例，不改变加权后的总逻辑长度（见函数注释）。
            self.weights = normalize_dataset_weights(self.actual_lengths, self.weights)

        # 逻辑长度 + 索引表：这两步之后 len(self) 与 __getitem__ 才能正常工作。
        # ⚠️ 之后若调用 cap_length_for_sharding()，会再次重算这两项。
        self.effective_lengths = self._compute_effective_lengths()
        self._set_effective_offsets(self.effective_lengths)

    def _set_effective_offsets(self, lengths: List[int]) -> None:
        """按逻辑长度建立“前缀和”索引表，供 __getitem__ 定位样本。

        举例（lengths = [100, 50, 30]）：
            effective_ends   = [100, 150, 180]   # 每个子数据集的右边界（开区间）
            effective_starts = [  0, 100, 150]   # 每个子数据集的左边界（闭区间）

        有了这两张表：
            dataset_idx = np.searchsorted(effective_ends, idx, side="right")
            local_idx   = idx - effective_starts[dataset_idx]
        即“先二分找到落在哪个长度区间，再算区间内偏移”。

        同一个函数既服务常规长度（effective_*），也服务 overfit 长度
        （_set_overfit_offsets() 用同样的算法建 _overfit_effective_* 那套镜像表）。
        """
        self.effective_ends = np.cumsum(lengths)
        self.effective_starts = np.concatenate([[0], self.effective_ends[:-1]])

    def _compute_effective_lengths(self) -> List[int]:
        """Return logical sampling lengths for all inner dataset_groups."""
        # 逻辑长度 = 采样器认为“这个子数据集有多少条样本”。分三种情况：
        #
        #   1) 验证集，或没开 use_weight_for_sampling
        #          → 逻辑长度 = 真实长度。最直观、最容易复现，推荐默认这样。
        #
        #   2) 训练集 + use_weight_for_sampling = True
        #          → 逻辑长度 = max(1, int(真实长度 × weight))
        #            weight < 1：下采样。数据集本身仍是全量，但一个 epoch 内只取其中一部分
        #                         （具体取哪些由 _map_weighted_local_index() 决定，
        #                          且会随 epoch 变化，避免每次都看同一段数据）。
        #            weight > 1：上采样。超出的部分靠“取模重复取样”实现。
        #
        #   3) overfit / 跨节点分片
        #          → 不在这里处理：overfit 走 enable_overfit()，
        #            分片对齐走 cap_length_for_sharding()，它们各自改写索引表。
        if not (self.is_training_set and self.use_weight_for_sampling):
            # 只对“训练集 + 开启加权采样”生效；返回副本，避免调用方原地改到 self.actual_lengths。
            return self.actual_lengths.copy()

        effective_lengths = []
        for dataset_idx, (actual_len, weight) in enumerate(zip(self.actual_lengths, self.weights)):
            if weight <= 0:
                # weight ≤ 0 的 group 在逻辑上没有意义（采样概率/长度无法定义），
                # 大概率是配置写错，直接报错而不是静默当成 0 条。
                raise ValueError(
                    "use_weight_for_sampling=True requires positive dataset_group weights; "
                    f"dataset_idx={dataset_idx} has weight={weight}."
                )
            # max(1, ...)：权重极小时也至少保留 1 条，避免出现长度为 0 的空数据集
            # （长度为 0 会让某些采样器/断言直接崩掉）。
            effective_lengths.append(max(1, int(actual_len * weight)))
        return effective_lengths

    def set_epoch(self, epoch: int) -> None:
        """Set epoch used by deterministic undersampling permutation."""
        # 只有 use_weight_for_sampling=True 的下采样路径会用到它：
        # 如果不把 epoch 混进偏移量，整个训练都会只看数据集里固定的那一小段
        # （1/weight 的数据），等于人为制造分布偏置。
        # 调用方：scripts/finetune.py 在每个 epoch 调完 sampler.set_epoch(epoch) 后紧接着调用。
        self.sampling_epoch = int(epoch)

    def _map_weighted_local_index(self, dataset_idx: int, local_idx: int) -> int:
        """Map a group-local effective index to a valid real index."""
        # 把“逻辑下标”翻译成子数据集里的“真实下标”。三种情况：
        #   · 非加权采样（默认）                        → 逻辑下标 == 真实下标；
        #   · effective_len >= actual_len（上采样/等长）→ 取模循环复用（local_idx % actual_len）；
        #   · effective_len <  actual_len（下采样）     → 用“近似黄金比例”的步长在真实下标
        #     空间里跳着取，保证一个 epoch 取到的子集散布均匀、不扎堆。
        actual_len = int(self.actual_lengths[dataset_idx])
        effective_len = int(self.effective_lengths[dataset_idx])
        if actual_len <= 0:
            raise ValueError(f"dataset_idx={dataset_idx} has non-positive actual length: {actual_len}")

        if not (self.is_training_set and self.use_weight_for_sampling):
            # 默认路径：不做任何映射，逻辑下标就是真实下标。
            return local_idx
        if effective_len >= actual_len:
            # 上采样（或等长）：绕圈重复取样，取模即可。
            return local_idx % actual_len

        # Undersampling must not always take prefix [0, effective_len).
        # Example: actual_len=10, effective_len=4, stride=7, offset=3 maps
        # local_idx 0..3 -> actual_idx [3, 0, 7, 4], a deterministic subset
        # spread across the real dataset. Offset changes with epoch so the
        # subset changes after Trainer advances sampler epoch.
        #
        # 为什么不能直接取前缀 [0, effective_len)？
        #   那样每个 epoch 只看数据集最前面的一小段：同一场景、同一动作阶段会被反复强化，
        #   后续阶段永远学不到，等于给模型喂了有偏数据。
        # 做法（下面几行）：
        #   stride = 与 actual_len 互质、且接近 actual_len × 0.618 的整数
        #   offset = (epoch × 1000003 + dataset_idx) % actual_len   # 随 epoch 换起点
        #   real_idx = (local_idx × stride + offset) % actual_len
        # 互质是关键：沿步长走一圈能遍历全部下标（若 stride 与 actual_len 有公因数，
        # 只能走到一部分下标，采样集合会明显偏小）。
        golden_ratio_conjugate = 0.61803398875
        epoch_offset_multiplier = 1000003
        stride = max(1, int(actual_len * golden_ratio_conjugate))
        # 把 stride 调到与 actual_len 互质：直接 +1 往上找，找到就停；
        # 万一加到 actual_len 都没有互质值（几乎不可能，actual_len=1 时会走到这里），
        # 退回 stride=1（退化为顺序取，但至少不会死循环）。
        while math.gcd(stride, actual_len) != 1:
            stride += 1
            if stride >= actual_len:
                stride = 1
                break
        # 起点随 epoch 与 dataset_idx 变化：不同 group、不同 epoch 取到的子集都不同，
        # 但同一 (epoch, dataset_idx) 下完全确定（可复现）。
        offset = (self.sampling_epoch * epoch_offset_multiplier + dataset_idx) % actual_len
        return int((local_idx * stride + offset) % actual_len)

    @staticmethod
    def _clear_inner_overfit(dataset: BaseLerobotDataset) -> None:
        """清掉某个子数据集上的 overfit 状态（幂等：没有这些属性时什么都不做）。

        子数据集用 _overfit_len / _overfit_indices 两个属性表示“我现在处于 overfit 模式”
        （见 base_lerobot_dataset.enable_overfit）。这里直接把属性删掉即退出该模式 ——
        用删除而不是置 0，是因为子数据集内部也是用 hasattr 判断是否处于 overfit。
        """
        for attr in ("_overfit_len", "_overfit_indices"):
            if hasattr(dataset, attr):
                delattr(dataset, attr)

    def _set_overfit_offsets(self, lengths: List[int]) -> None:
        """为 overfit 模式建立一套独立的索引表（不覆盖常规的 effective_* 表）。

        overfit 时 __len__ / _resolve_index 会改用这组 _overfit_effective_*，
        这样关闭 overfit（_disable_overfit）后能立刻回到常规索引表，无需重建数据集。
        """
        self._overfit_effective_lengths = lengths
        self._overfit_effective_ends = np.cumsum(lengths)
        self._overfit_effective_starts = np.concatenate([[0], self._overfit_effective_ends[:-1]])

    def _disable_overfit(self) -> None:
        """关闭 overfit：同时清掉子数据集与本类上的 overfit 状态。

        本类需要一起删的属性：
            _overfit_len                    __len__ 的覆盖值
            _overfit_effective_lengths/…    overfit 专用索引表
            _overfit_mode                   global / per_dataset（仅用于日志与调试）
        """
        # 先让子数据集退出 overfit（否则它们还是“固定样本列表”的行为）。
        for dataset in self.datasets:
            self._clear_inner_overfit(dataset)

        # 再删本类自己的标记属性 —— 删掉之后 hasattr(self, "_overfit_len") 为 False，
        # __len__ 与 _resolve_index 自动回到常规路径。
        for attr in (
            "_overfit_len",
            "_overfit_effective_lengths",
            "_overfit_effective_ends",
            "_overfit_effective_starts",
            "_overfit_mode",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def all_embodiment_types(self) -> List[str]:
        """Return sorted list of unique embodiment types in this mixture."""
        # 用途：模型/评测/日志侧需要知道“本次训练涉及哪些机器人形态”，
        # 例如给 action tokenizer 准备 embodiment 词表、打印形态清单做核对。
        # 返回去重 + 排序后的列表：顺序稳定，便于复现与跨次对比。
        vals = set(self.embodiments2types.values())
        for v in vals:
            if not isinstance(v, str):
                # 兜底检查：正常配置里 embodiment_type 都应是 str。
                # 出现数字/None（YAML 类型写错）时先 warning 提示，下面仍统一 str() 转换，
                # 保证返回值类型稳定，不会因为一个坏配置就让整条链路崩掉。
                logger.warning(
                    "embodiment_type value %r (type=%s) is not a str! Full set: %s",
                    v, type(v).__name__, vals, exc_info=True,
                )
        return sorted(str(v) for v in set(self.embodiments2types.values()))

    def enable_overfit(self, n_samples: int, mode: str = "global"):
        """Pin overfit mode to a stable deterministic subset across datasets.

        “小样本过拟合”自检：把混合数据集缩到固定的 n 条样本上反复训练。
        如果连这几条都学不会（loss 不降），说明问题在模型/数据管线本身，
        而不是“数据不够多”——这是排查训练链路最有效的开关之一。

        Args:
            n_samples: 期望的稳定样本数。
                · mode="global"      → 整个混合数据集合计 n 条；
                · mode="per_dataset" → 每个子数据集各 n 条。
            mode: ``global`` or ``per_dataset``.

        两个模式的区别（3 个子数据集、n=8 为例）：
            global      → 一共挑 8 条，按顺序从各子数据集“先到先得”
                          （前面的子数据集可能被挑满，后面的分不到）。
            per_dataset → 每个子数据集各挑 8 条 → 共 24 条，保证每种形态都被看到。

        实现要点：
          · 先 _disable_overfit() 复位，保证重复调用是幂等的；
          · “取多少条”由本类按逻辑长度/真实长度夹取，真正“挑哪几条”交给子数据集的
            enable_overfit(take)（它会在训练集里筛掉读不出来、质量不合格的样本）；
          · 挑不满只 warning 不报错 —— 小数据集上这是正常现象。
        """
        target_samples = int(n_samples)
        if target_samples <= 0:
            # n_samples ≤ 0 视为“关掉 overfit”，回到常规数据集。
            self._disable_overfit()
            return

        if mode not in {"global", "per_dataset"}:
            raise ValueError(f"Unsupported overfit mode: {mode}")

        # 先复位：清理上一次可能残留的 overfit 状态（含子数据集内部的固定样本列表）。
        self._disable_overfit()

        overfit_lengths = []
        selected = 0

        if mode == "per_dataset":
            # per_dataset：每个子数据集都尽量挑 n 条，保证每种形态都有代表样本。
            for dataset_idx, dataset in enumerate(self.datasets):
                effective_len = int(self.effective_lengths[dataset_idx])
                actual_len = int(self.actual_lengths[dataset_idx])
                # 取三者最小值：请求数 / 逻辑长度（可能被下采样缩过）/ 真实长度。
                take = min(target_samples, effective_len, actual_len)

                if take > 0 and hasattr(dataset, "enable_overfit"):
                    # 真正的挑选在子数据集里做（会扫描数据筛掉坏样本，仅在 overfit 时发生）。
                    dataset.enable_overfit(take)
                else:
                    # take==0（该数据集没样本可用）或子类不支持 overfit：
                    # 确保它不残留旧状态，避免“只有一部分数据集进了 overfit 模式”。
                    self._clear_inner_overfit(dataset)

                overfit_lengths.append(take)
                selected += take
        else:
            # global：整份全局预算按顺序分配，前面的子数据集先拿满。
            # 先按逻辑总长度夹一次，避免请求数超过数据集规模。
            target_samples = min(target_samples, int(self.effective_ends[-1]))
            remaining = target_samples

            for dataset_idx, dataset in enumerate(self.datasets):
                effective_len = int(self.effective_lengths[dataset_idx])
                actual_len = int(self.actual_lengths[dataset_idx])
                # 该子数据集能贡献的条数 = min(剩余预算, 逻辑长度, 真实长度)。
                take = min(remaining, effective_len, actual_len)

                if take > 0 and hasattr(dataset, "enable_overfit"):
                    dataset.enable_overfit(take)
                else:
                    self._clear_inner_overfit(dataset)

                overfit_lengths.append(take)
                remaining -= take
                selected += take
                if remaining <= 0:
                    # 预算用完：剩下的子数据集补 0 并立刻结束循环。
                    # ⚠️ 必须补齐长度，overfit_lengths 要与 self.datasets 等长
                    # （_set_overfit_offsets 会拿它做前缀和索引）。
                    overfit_lengths.extend([0] * (len(self.datasets) - dataset_idx - 1))
                    break

        if selected < target_samples and mode == "global":
            # 挑不满：数据集本身样本太少，或大量样本被质量标注筛掉。只提示不报错。
            logger.warning(
                "Mixture overfit requested %s samples but only pinned %s stable samples.",
                target_samples,
                selected,
            )
        elif mode == "per_dataset":
            # per_dataset 模式下“期望总数”= 每数据集 n 条 × 数据集个数。
            expected_samples = target_samples * len(self.datasets)
            if selected < expected_samples:
                logger.warning(
                    "Mixture overfit requested %s stable samples per dataset (%s total) but only pinned %s.",
                    target_samples,
                    expected_samples,
                    selected,
                )

        # 收尾：记录模式、索引表与覆盖后的长度。
        # 这三个属性一写出，之后 __len__ / _resolve_index / __getitem__ 就都走 overfit 路径。
        self._overfit_mode = mode
        self._set_overfit_offsets(overfit_lengths)
        self._overfit_len = selected

    def __len__(self):
        """逻辑样本数：overfit 模式下是挑出来的稳定样本数，否则是各子数据集逻辑长度之和。"""
        if hasattr(self, "_overfit_len"):
            # 用 hasattr 判断而不是标志位：overfit 关闭时该属性被删掉了（见 _disable_overfit）。
            return self._overfit_len
        return int(self.effective_ends[-1])

    def _resolve_index(self, idx: int):
        """把“混合数据集全局下标”解析成 (dataset_idx, 子数据集内部下标)。

        两级映射：
          1) dataset_idx：二分查找 idx 落在哪个逻辑长度区间
                          （overfit 时改用 _overfit_effective_* 那套索引表）；
          2) local_idx  ：区间内偏移 = idx - starts[dataset_idx]，
                          非 overfit 时还要经 _map_weighted_local_index() 做加权下采样映射。

        返回值中的 local_idx 已经可以直接喂给 self.datasets[dataset_idx]。
        """
        overfit_active = hasattr(self, "_overfit_len")
        # 两套索引表二选一：overfit 用 _overfit_effective_*，常规用 effective_*。
        active_ends = self._overfit_effective_ends if overfit_active else self.effective_ends
        active_starts = self._overfit_effective_starts if overfit_active else self.effective_starts
        # 二分查找用 "right"：ends 是右开区间，正好落在边界上的 idx 归到下一个数据集，
        # 这样每个 idx 只会命中唯一一个区间。
        dataset_idx = int(np.searchsorted(active_ends, idx, side="right"))
        local_idx = int(idx - active_starts[dataset_idx])
        if overfit_active:
            # overfit 的样本是子数据集内部预先筛好的稳定列表，下标直接透传即可
            # （不能做加权映射，否则会指到没被选中的样本上）。
            return dataset_idx, local_idx
        return dataset_idx, self._map_weighted_local_index(dataset_idx, local_idx)

    def get_item_with_meta(self, idx: int):
        """取一条样本，并额外返回来源信息 (dataset_idx, local_idx)。

        与 __getitem__ 的唯一区别就是把 dataset_idx / local_idx 一起返回，便于定位：
        例如 scripts/finetune.py 打印每个 embodiment 的首条样本、
        eval 脚本记录样本来自哪个数据集目录。

        副作用：会往样本里补三个键
            embodiment / embodiment_type  该样本属于哪个机器人形态（processor / 模型要用）
            dataset_locator              人类可读的定位串，排查坏样本时直接用

        Returns:
            (sample, dataset_idx, local_idx)
        """
        # int(idx)：DataLoader / 采样器可能传 numpy 整数，统一转成 Python int
        # （便于日志打印，也避免下游对类型做各种兼容）。
        dataset_idx, local_idx = self._resolve_index(int(idx))
        # 真正的取数在子数据集里：它负责读帧、切维、补掩码，并在最后调用 processor.preprocess。
        sample = self.datasets[dataset_idx][local_idx]
        emb_type = self.embodiments2types[self.embodiments[dataset_idx]]
        # 混合数据集是所有形态、所有来源的公共上层，所以“形态标签”由这里统一补：
        # 子数据集自己并不知道（也不该知道）自己被混进了哪个形态标签下。
        sample["embodiment"] = emb_type
        sample["embodiment_type"] = emb_type
        # 定位串优先用子数据集自己提供的（它知道真实分片）；没有时才走下面的兜底近似。
        existing_locator = sample.get("dataset_locator")
        if existing_locator is None:
            # 兜底近似：如果子数据集没提供精确定位信息，就用“采样所属组的第一个目录
            # + 子数据集内部下标”近似。
            # 单目录的 group 是精确的；多目录 group 仍可能差一个分片（真实目录需要再算一遍），
            # 但作为日志定位已经足够，不参与任何训练逻辑。
            dataset_dirs = getattr(self.datasets[dataset_idx], "dataset_dirs", None)
            dataset_name = dataset_dirs[0] if dataset_dirs else f"dataset_group_{dataset_idx}"
            # Approximation fallback: if the inner dataset did not provide its own precise
            # locator, fall back to the first dataset_dir of the sampled group plus local_idx.
            # This is exact for single-dir groups and usually close for grouped mixtures, but
            # multi-dir groups can still differ by one inner shard.
            sample["dataset_locator"] = f"dataset_dir={dataset_name}, local_idx={local_idx}"

        return sample, dataset_idx, local_idx

    def __getitem__(self, idx):
        """取一条样本（torch Dataset 的标准入口，DataLoader 的各 worker 直接调它）。

        返回的 sample 已经是“最终形态”：子数据集完成了 shape_meta 取帧 + 切维 +
        processor 预处理（归一化、相对动作、模板 slots），这里只补跨数据集公共字段。
        """
        sample, dataset_idx, _local_idx = self.get_item_with_meta(int(idx))
        emb_type = self.embodiments2types[self.embodiments[dataset_idx]]

        # 把形态标签也塞进 action / proprio 两个 slot 的内容 dict 里。
        # slot 内容是一份带元信息的 dict（value / *_dim_is_pad / parts_meta / ...），
        # 这里额外挂上 embodiment，供按 slot 读取形态的实现使用；
        # 主路径编码 embodiment token 时读的是顶层 samples["embodiment"]
        # （由 samples_builder 填写，见 docs/data/samples_builders_zh.md）。
        if "samples" in sample and "action" in sample["samples"] and "proprio" in sample["samples"]:
            sample["samples"]["action"]["embodiment"] = emb_type
            sample["samples"]["proprio"]["embodiment"] = emb_type

        return sample

    def get_dataset_stats(
        self,
        processor: MixtureProcessor,
        only_keys: Optional[Dict[str, Dict[str, Set[str]]]] = None,
        embodiments: Optional[Set[str]] = None,
    ):
        """
        Compute dataset statistics for normalization（计算归一化统计量）。

        把各子数据集的统计量按 embodiment_type 分组、按 weight 聚合，得到
            {embodiment_type: {"action": {...}, "state": {...}}}
        交给 MixtureProcessor.set_normalizer_from_stats() 做归一化。

        ⚠️ 聚合的键是 embodiment_type，不是数据源别名：
        同一形态（例如两个数据集的 galaxea_r1pro）必须共享一套归一化参数，
        这与 processor 的注册方式（processors[embodiment_type]）保持一致。

        Args:
            processor: MixtureProcessor，按 embodiment_type 提供子 processor。
                统计前要先把 action/state 过一遍 processor 的变换（例如相对动作），
                所以这里是“按形态传 processor”，而不是在本类里自己算。
            only_keys: 只统计指定键，格式（外层 key 是数据源别名）：
                {"galaxea_r1lite": {"action": {"left_arm"}, "state": {"torso"}}}
                None = 全量统计。用于加速：只算真正会用到的 part。
            embodiments: 只返回这些数据源别名所对应的 type 的统计量。
                注意：为了保持“按 type 聚合”的语义，与请求别名同 type 的其他别名
                仍会参与计算（只过滤输出，不改变聚合口径）。

        Returns:
            Dict with structure: {embodiment_type: {"action": {...}, "state": {...}}}
        """
        # ---- 1) 解析过滤条件：别名 → type ----
        # 请求里给的是别名（用户看得懂的名字），但聚合与输出都按 type 组织，
        # 所以先做一次“别名 → type”的展开，并顺手校验别名拼写。
        requested_embodiments = None
        requested_types = None
        if embodiments is not None:
            requested_embodiments = set(embodiments)
            unknown_embodiments = requested_embodiments - set(self.embodiments2types.keys())
            if unknown_embodiments:
                # 拼错别名时立刻报错，避免静默算出空统计量、后面才在归一化阶段炸掉。
                raise ValueError(
                    f"Unknown embodiments requested for stats computation: {sorted(unknown_embodiments)}"
                )
            requested_types = {self.embodiments2types[emb] for emb in requested_embodiments}

        # ---- 2) 按 type 收集：每个 type 一份统计量列表 + 一份对应的权重列表 ----
        # 用 defaultdict(list) 是因为同一 type 下通常有多个子数据集（多个别名/多个 group），
        # 它们要合并成一份统计量（见 _aggregate_weighted_stats）。
        stats_by_type = defaultdict(list)
        weights_by_type = defaultdict(list)

        _stats_write(
            f"📊 [Stats] Mixture stats requested for {len(processor.processors)} embodiment processors",
            "cyan",
        )

        # ---- 3) 逐子数据集算统计量（耗时大头：要扫一遍全部 episode）----
        for emb, w, ds in zip(self.embodiments, self.weights, self.datasets):
            emb_type = self.embodiments2types[emb]
            if requested_types is not None and emb_type not in requested_types:
                # 本次不需要这个 type（例如只统计评测用到的形态）→ 跳过，省时间。
                continue

            # 下面两行只为日志：让“这个数据集有多大”在终端上一眼可见。
            # num_frames / num_episodes 是底层 LeRobot 数据集(每个目录)的属性，
            # 走 multi_dataset._datasets 取是最直接的读法（辅助信息，缺失也不影响统计）。
            ds_frames = sum(
                int(getattr(inner_ds, "num_frames", 0)) for inner_ds in ds.multi_dataset._datasets
            )
            ds_episodes = sum(
                int(getattr(inner_ds, "num_episodes", 0)) for inner_ds in ds.multi_dataset._datasets
            )
            _stats_write(
                "🚚 [Stats] Computing {} (type={}, weight={:.4f}, len={}, frames={}, episodes={})".format(
                    emb,
                    emb_type,
                    w,
                    len(ds),
                    ds_frames,
                    ds_episodes,
                ),
                "cyan",
            )
            # 把“当前在处理哪个别名/形态”写进子数据集：v3 的统计实现会读取这两个属性，
            # 在日志里打印 “🪪 [Stats] Dataset label=<别名> (type=<形态>)”，
            # 方便在一长串统计日志里对上“现在算的是哪个数据源”。
            setattr(ds, "_stats_debug_name", emb)
            setattr(ds, "_stats_debug_type", emb_type)

            # only_keys 是按别名给的，这里取出该别名对应的子集（None = 该别名全量统计）。
            emb_only_keys = only_keys.get(emb) if only_keys else None
            # 真正的统计在子数据集里做：它按 type 对应的 processor 做变换后统计分布。
            stats = ds.get_dataset_stats(processor[emb_type], only_keys=emb_only_keys)

            stats_by_type[emb_type].append(stats)
            weights_by_type[emb_type].append(w)

        # Aggregate stats for each type
        # ---- 4) 同 type 合并：多个数据源的统计量按 weight 加权成一份 ----
        aggregated_stats_by_type = {}
        for emb_type in stats_by_type:
            aggregated_stats_by_type[emb_type] = self._aggregate_weighted_stats(
                weights_by_type[emb_type], stats_by_type[emb_type]
            )

        return aggregated_stats_by_type

    def sync_weights_for_sharding(self):
        """Recompute normalized weights using global actual_lengths across all nodes.

        （跨节点分片时，用全局真实长度重算归一化权重。）

        场景：开启按节点分片（shard_datasets_by_node）后，同一个 (embodiment, group)
        的目录被分摊到不同节点，各节点看到的 actual_lengths 不同。
        而 __init__ 里的 normalize_dataset_weights() 用的系数
            k = sum(L) / sum(L*w)
        依赖本地长度 → 各节点算出的权重不一致 → 采样比例在节点之间漂移。

        做法：把两个标量 sum(L_i) 与 sum(L_i * raw_w_i) 做一次 all-reduce 求和，
        得到全局系数 k_global = ΣL / Σ(L·w)，再作用到原始权重 _raw_weights 上。

        为什么只 reduce 两个标量：各节点持有的 (embodiment, group) 条目集合和向量长度
        都可能不同，逐元素对齐后再 reduce 很麻烦；而这两个标量已足够算出所需比值。
        （同节点的多张卡共享同一份数据集，all-reduce 结果会多乘一个 gpus_per_node，
          但分子分母同时放大，比值里约掉，不影响最终权重。）

        When dataset sharding splits dirs across nodes, each node sees different
        actual_lengths for the same embodiment/group.  ``normalize_dataset_weights``
        (called in ``__init__``) therefore produces divergent weights per node
        because the normalization factor ``k = sum(L) / sum(L*w)`` depends on
        local lengths.

        This method computes a *global* normalization factor by all-reducing the
        partial sums ``sum(L_i)`` and ``sum(L_i * raw_w_i)`` across all ranks,
        then re-applies it to the raw (pre-normalization) weights.  Because each
        node has a different subset of (embodiment, group) entries (and thus
        different-length weight vectors), we only reduce two scalars — not the
        per-dataset vectors — so alignment is not needed.

        GPUs on the same node share the same dataset, so the all-reduce sum is
        ``global_value * gpus_per_node``, but the factor cancels in the ratio.

        Must be called after ``__init__`` and before any sampling takes place,
        only when ``shard_datasets_by_node`` is ``True``.

        ⚠️ 当前仓库里没有调用点（保留给按节点分片的实验路径；
        configs/task/*.yaml 的 shard_datasets_by_node 默认都是 false）。
        """
        # 延迟 import：只有分片场景才需要分布式库，单机训练不必在这里引入。
        import torch.distributed as dist

        if not dist.is_initialized():
            # 非分布式环境（单卡/调试脚本）没有“跨节点”概念，直接跳过。
            return

        device = torch.cuda.current_device()

        # Partial sums from this node's datasets
        # 本节点（严格说是本进程可见的）两个关键标量：真实总长、按原始权重的加权总长。
        local_total_len = float(sum(self.actual_lengths))
        local_weighted_sum = float(
            sum(l * w for l, w in zip(self.actual_lengths, self._raw_weights))
        )

        # All-reduce to get global sums (factor of gpus_per_node cancels)
        # 用 float64 累加：大规模数据下总帧数可达千万级，float32 相加大数会掉精度。
        # 一次性把两个标量放进同一个 tensor 做 all_reduce，省一次通信。
        pair = torch.tensor(
            [local_total_len, local_weighted_sum], device=device, dtype=torch.float64
        )
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        global_total, global_weighted = pair.tolist()

        if global_weighted == 0:
            # 全局权重和为 0（没有数据/权重全 0）：保持原权重，避免除零。
            return

        k_global = global_total / global_weighted
        # 注意是作用在 _raw_weights（原始权重）上，而不是已被本地系数乘过的 self.weights，
        # 否则会把本地的偏差再叠加一次。
        self.weights = [k_global * w for w in self._raw_weights]

        # Recompute effective lengths with the globally-consistent weights
        # 权重变了，逻辑长度与索引表必须一起重算，否则下标映射会对不上。
        self.effective_lengths = self._compute_effective_lengths()
        self._set_effective_offsets(self.effective_lengths)

    def cap_length_for_sharding(self):
        """Equalize effective dataset length across nodes by capping to the global min.

        （把各节点的逻辑长度对齐到全局最小值，避免节点间 epoch 进度漂移。）

        场景：按节点分片后，各节点拿到的数据量可能差很多（例如某个节点分到一个
        2700 万帧的目录，其他节点各约 2000 万帧）。这会导致“epoch 长度不平衡”：
        数据多的节点 epoch 更长，其他节点只能反复绕圈重看数据，训练进度漂移。

        做法：all-reduce(MIN) 求出所有节点里最小的 len(dataset)，然后按比例缩小本节点
        各子数据集的 effective_lengths（超出的部分在 epoch 内随机取样，机制与
        use_weight_for_sampling 相同）。

        After node-level sharding, different nodes may have very different total
        dataset sizes (e.g. one node gets a giant 27M-frame dir while others have
        ~20M each).  This causes epoch-length imbalance: the overloaded node has
        longer epochs and other nodes must wrap and re-see data.

        This method all-reduces to find the minimum ``len(dataset)`` across all
        nodes, then scales down each local dataset's effective_lengths
        proportionally so that ``len(self)`` matches the global min.  Datasets
        whose effective_length is reduced will randomly sample from their full
        actual_length each epoch (same mechanism as ``use_weight_for_sampling``).

        Must be called after ``sync_weights_for_sharding()`` and before sampler
        creation, only when ``shard_datasets_by_node`` is ``True``.

        ⚠️ 注意它与 use_weight_for_sampling 的耦合：只有该开关为 True 时，
        缩短后的逻辑下标才会经 _map_weighted_local_index() 映射回“真实下标空间”，
        从而在整段数据上取样；若开关为 False，逻辑下标会被直接当成真实下标，
        实际效果退化为“每个 epoch 只看数据集的前缀”。

        ⚠️ 当前仓库里没有调用点（同 sync_weights_for_sharding，属分片实验路径）。
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        device = torch.cuda.current_device()
        # 各节点上报自己的逻辑长度，取全局最小值作为目标长度。
        local_len = torch.tensor([len(self)], device=device, dtype=torch.long)
        dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
        target_len = int(local_len.item())
        current_len = len(self)

        if target_len >= current_len:
            # 本节点本来就是最短的（或刚好相等）：无需缩小，直接返回。
            return  # this node is already at or below the min

        # Different shard bins can still have different logical epoch lengths
        # after dir assignment and weighted sampling. Cap longer bins to the
        # shortest bin so all ranks finish each epoch together. This only shrinks
        # the logical sampling space; it does not unload dirs or reduce init memory.
        # 只缩小“逻辑采样空间”，不会卸载数据目录、也不减少初始化时的内存占用 ——
        # 目标仅仅是让所有 rank 每个 epoch 进度一致。
        scale = target_len / current_len
        logger.info(
            f"[Dataset Sharding] Capping dataset length: {current_len} -> {target_len} "
            f"(scale={scale:.4f})"
        )
        # 等比缩放：每个子数据集的逻辑长度都乘同一个 scale。
        # max(1, ...) 防止极小数据集被缩成 0 条（长度 0 会让采样器出错）。
        self.effective_lengths = [max(1, int(l * scale)) for l in self.effective_lengths]
        self._set_effective_offsets(self.effective_lengths)

    def set_processor(self, processor: MixtureProcessor):
        """把处理器下发到每个子数据集（本类不自己处理样本，只做路由）。

        分工：MixtureProcessor 按 embodiment_type 存放各形态的 processor；
        本类按别名找到对应 type，从 processor 里取出子 processor 塞给子数据集。
        之后子数据集的 __getitem__ 会执行 processor.preprocess（归一化、相对动作、
        模板拼装 samples），所以这一步必须在第一次取数之前完成。

        顺带把 embodiment_type 回写到 processor 上：processor 内部不少部件
        （例如 samples_builder 的模板 slot）需要知道自己是哪个形态才能拼出正确样本。
        """
        for emb, ds in zip(self.embodiments, self.datasets):
            emb_type = self.embodiments2types[emb]
            # 同一个 type 的多个子数据集从 processor 里取到同一个实例，标签自然一致；
            # build_processors() 会为每个 embodiment 深拷贝独立实例，因此就地改属性
            # 不会串到别的形态上。
            p = processor[emb_type]
            p.embodiment_type = emb_type
            ds.set_processor(p)

    def get_invalid_sample_report(self, reset: bool = False) -> List[Dict[str, Any]]:
        """汇总所有子数据集的“坏样本”报告（读失败、质量不合格等）。

        子数据集在 __getitem__ 里遇到坏样本会跳过并登记（多进程共享的容器，
        见 base_lerobot_dataset），训练/评测结束后调用本方法即可一次性拿到整份清单。
        这里额外补上 embodiment / embodiment_type，便于按形态统计坏样本比例。

        Args:
            reset: True 表示读取后清空登记（例如每个 epoch 取一次增量）。

        Returns:
            List[Dict]，每条记录含阶段(stage)、请求下标、实际下标、错误信息、来源等。
        """
        report = []
        for emb_name, dataset in zip(self.embodiments, self.datasets):
            # 能力探测：不是所有子类都实现了这份报告，缺失时跳过即可。
            if not hasattr(dataset, "get_invalid_sample_report"):
                continue
            for record in dataset.get_invalid_sample_report(reset=reset):
                # 复制一份再补字段，避免就地改到子数据集内部维护的记录结构。
                item = dict(record)
                item["embodiment"] = emb_name
                item["embodiment_type"] = self.embodiments2types.get(emb_name)
                report.append(item)
        return report

    def clear_invalid_sample_report(self) -> None:
        """清空所有子数据集的坏样本登记（例如每个 epoch 开始前重置计数）。"""
        for dataset in self.datasets:
            if hasattr(dataset, "clear_invalid_sample_report"):
                dataset.clear_invalid_sample_report()

    @staticmethod
    def _aggregate_weighted_stats(weights: List, stats: List):
        """
        Aggregate multiple dataset stats with the given weights
        （把同一 embodiment_type 下多个子数据集的统计量按权重合并成一份）。

        为什么需要“按权重”：同一形态可能有多个数据源（多个别名 / 多个 group），
        它们各自算出自己的 action/state 统计量；训练时各组的出现频率就是 weight，
        所以合并后的均值/方差应按 weight 加权，否则统计量会偏向“数据多但采样少”的组。

        合并规则：
            · min/max    → 取极值（amin/amax），保证覆盖所有子数据集的取值范围；
            · 分位数     → 用加权平均近似。严格做法是把原始数据合到一起重算分位数，
                           成本太高；归一化只需要一个稳健的粗略范围。
            · mean/std   → 用“合并方差公式”精确加权合并（见 _weighted_mean_std）；
            · key 不一致 → 取并集，只被部分子数据集标注的 key 也会保留。

        Args:
            weights: 每个子数据集的权重（与本类 self.weights 一致，可能已归一化）。
            stats: 每个子数据集的统计量，结构
                {"action": {key: {...}}, "state": {key: {...}}}，
                由子数据集的 get_dataset_stats() 产出。
        """
        assert len(weights) == len(stats), "weights and stats must have the same length"
        assert len(weights) > 0, "weights cannot be empty"

        def _weight_view(example: torch.Tensor) -> torch.Tensor:
            """Reshape weights for broadcasting."""
            # 把权重整理成可广播的形状：前面插一维长度 N，后面按张量维度补 1。
            # 例如 N=3、张量形状 [H, D] → view_shape=[3, 1, 1]，可直接与 [3, H, D] 相乘。
            w = torch.as_tensor(weights, dtype=example.dtype, device=example.device)
            total = w.sum()
            if total.item() == 0:
                raise ValueError("Sum of weights must be greater than zero.")
            # 归一化：只保留相对比例（和为 1）；后面的加权和/方差公式都基于这个前提。
            w = w / total
            view_shape = [len(weights)] + [1] * (example.dim())
            return w.view(view_shape)

        def _weighted_mean_std(means_list, std_list):
            """加权合并均值与标准差（组内方差 + 组间偏移）。

            合并方差的标准公式：
                Var = Σ_k w_k · ( Var_k + (mean_k − mean)² )
            其中 w_k 是归一化权重。
            第二项 (mean_k − mean)² 是“组间方差”：少了它，当各组均值差异较大时
            （例如两个数据集的关节角分布明显不同），合并后的 std 会偏小。
            """
            means = torch.stack(means_list)
            # 只有方差满足加权合并公式：先 std → var，最后再 sqrt() 变回 std。
            vars = torch.stack([s**2 for s in std_list])
            w_view = _weight_view(means[0])
            weighted_mean = (means * w_view).sum(dim=0)
            weighted_var = (vars + (means - weighted_mean) ** 2) * w_view
            weighted_var = weighted_var.sum(dim=0)
            return weighted_mean, weighted_var.sqrt()

        def _weighted_avg(tensor_list):
            """按权重求平均：用于分位数这类无法精确合并、只能近似的量。"""
            stacked = torch.stack(tensor_list)
            w_view = _weight_view(stacked[0])
            return (stacked * w_view).sum(dim=0)

        # 输出结构与单个数据集的统计量一致：外层 field，内层 key（part 名）。
        aggregated_stats = {"state": defaultdict(dict), "action": defaultdict(dict)}

        for field in ["state", "action"]:
            # Collect all keys across datasets for this field
            # 取所有子数据集 key 的并集：允许某些数据集只标注了部分 part。
            keys = set()
            for s in stats:
                keys.update(s[field].keys())

            for key in keys:
                # field_stats[k] = 第 k 个子数据集在“该 field / 该 key”下的统计量字典。
                field_stats = [s[field][key] for s in stats]

                # Stepwise min/max: take the extreme values across datasets
                # 取极值而不是加权平均：min/max 必须覆盖所有子数据集的取值范围，
                # 否则归一化后可能把某些数据集的真实值裁掉。
                stepwise_min = torch.stack([fs["stepwise_min"] for fs in field_stats]).amin(dim=0)
                stepwise_max = torch.stack([fs["stepwise_max"] for fs in field_stats]).amax(dim=0)

                # Global min/max: same approach as stepwise
                global_min = torch.stack([fs["global_min"] for fs in field_stats]).amin(dim=0)
                global_max = torch.stack([fs["global_max"] for fs in field_stats]).amax(dim=0)

                # Quantiles: approximate with weighted average
                # 分位数用加权平均近似（见函数开头说明）。三组不同“极端程度”的分位数
                # 供不同归一化/裁剪策略选用：q01/q99 是常用稳健范围，
                # q001/q999 … q00001/q99999 是逐级放宽的兜底范围。
                stepwise_q01 = _weighted_avg([fs["stepwise_q01"] for fs in field_stats])
                stepwise_q99 = _weighted_avg([fs["stepwise_q99"] for fs in field_stats])
                global_q01 = _weighted_avg([fs["global_q01"] for fs in field_stats])
                global_q99 = _weighted_avg([fs["global_q99"] for fs in field_stats])

                stepwise_q001 = _weighted_avg([fs["stepwise_q001"] for fs in field_stats])
                stepwise_q999 = _weighted_avg([fs["stepwise_q999"] for fs in field_stats])
                global_q001 = _weighted_avg([fs["global_q001"] for fs in field_stats])
                global_q999 = _weighted_avg([fs["global_q999"] for fs in field_stats])

                stepwise_q0001 = _weighted_avg([fs["stepwise_q0001"] for fs in field_stats])
                stepwise_q9999 = _weighted_avg([fs["stepwise_q9999"] for fs in field_stats])
                global_q0001 = _weighted_avg([fs["global_q0001"] for fs in field_stats])
                global_q9999 = _weighted_avg([fs["global_q9999"] for fs in field_stats])

                stepwise_q00001 = _weighted_avg([fs["stepwise_q00001"] for fs in field_stats])
                stepwise_q99999 = _weighted_avg([fs["stepwise_q99999"] for fs in field_stats])
                global_q00001 = _weighted_avg([fs["global_q00001"] for fs in field_stats])
                global_q99999 = _weighted_avg([fs["global_q99999"] for fs in field_stats])

                # Means/stds: weighted aggregation
                # 均值/标准差用“合并方差公式”精确加权合并，而不是简单平均。
                stepwise_mean, stepwise_std = _weighted_mean_std(
                    [fs["stepwise_mean"] for fs in field_stats],
                    [fs["stepwise_std"] for fs in field_stats],
                )
                global_mean, global_std = _weighted_mean_std(
                    [fs["global_mean"] for fs in field_stats],
                    [fs["global_std"] for fs in field_stats],
                )

                # 逐项写回。注意两套粒度都保留：
                #   stepwise_* = 动作块内“第 t 步”各自的统计量（逐步归一化）
                #   global_*   = 整个动作块共用一套统计量
                # 下游按配置里的 use_stepwise_action_norm 选用其中一套
                # （见 configs/data/r1pro.yaml 的说明）。
                aggregated_stats[field][key]["stepwise_min"] = stepwise_min
                aggregated_stats[field][key]["stepwise_max"] = stepwise_max
                aggregated_stats[field][key]["global_min"] = global_min
                aggregated_stats[field][key]["global_max"] = global_max
                aggregated_stats[field][key]["stepwise_q01"] = stepwise_q01
                aggregated_stats[field][key]["stepwise_q99"] = stepwise_q99
                aggregated_stats[field][key]["global_q01"] = global_q01
                aggregated_stats[field][key]["global_q99"] = global_q99
                aggregated_stats[field][key]["stepwise_q001"] = stepwise_q001
                aggregated_stats[field][key]["stepwise_q999"] = stepwise_q999
                aggregated_stats[field][key]["global_q001"] = global_q001
                aggregated_stats[field][key]["global_q999"] = global_q999
                aggregated_stats[field][key]["stepwise_q0001"] = stepwise_q0001
                aggregated_stats[field][key]["stepwise_q9999"] = stepwise_q9999
                aggregated_stats[field][key]["global_q0001"] = global_q0001
                aggregated_stats[field][key]["global_q9999"] = global_q9999
                aggregated_stats[field][key]["stepwise_q00001"] = stepwise_q00001
                aggregated_stats[field][key]["stepwise_q99999"] = stepwise_q99999
                aggregated_stats[field][key]["global_q00001"] = global_q00001
                aggregated_stats[field][key]["global_q99999"] = global_q99999
                aggregated_stats[field][key]["stepwise_mean"] = stepwise_mean
                aggregated_stats[field][key]["stepwise_std"] = stepwise_std
                aggregated_stats[field][key]["global_mean"] = global_mean
                aggregated_stats[field][key]["global_std"] = global_std

        return aggregated_stats
