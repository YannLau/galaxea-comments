# =============================================================================
# src/g05/data/base_lerobot_dataset.py — LeRobot 数据集基类（v2.1 / v3.0 共用）
# =============================================================================
#
# 【这个文件是什么】
#   它是整个训练数据链路的“总装车间”：把磁盘上若干个 LeRobot 格式数据集目录，
#   按配置（shape_meta）组装成 PyTorch 可以按样本索引取数的 Dataset。
#   具体负责 4 件事：
#     1) 读配置：把 YAML 里的 shape_meta（action/state/images 的维度声明）解析成
#        “内部 key → parquet 原始列”的映射，并做严格校验；
#     2) 读数据：决定“每个样本要读哪些帧”（delta_timestamps），交给底层
#        MultiLeRobotDataset 去读 parquet 里的数值列与 mp4 里的图像帧；
#     3) 切数据：把底层读回来的“一大块原始列”按 start_index/raw_shape 切成
#        每个 part（left_arm / right_arm / gripper ...）各自的张量；
#     4) 补数据：生成 action_is_pad / state_is_pad 掩码、注入 dummy 相机、
#        合并多数据集、切分训练/验证集、算归一化统计量、错误样本自动重采样。
#
# 【它在数据链路中的位置】
#
#   configs/data/<task>.yaml
#        │  (Hydra 实例化)
#        ▼
#   MixtureLerobotDataset                     src/g05/data/mixture_lerobot_dataset.py
#      │  · 按 embodiment 逐个 new 出本类实例（可含多个数据集目录）
#      │  · 把多个 embodiment 按权重混合、汇总归一化统计量
#      ▼
#   BaseLerobotDataset ← 本文件（= 一个 embodiment 的数据集）
#   ├─ BaseLerobotDatasetV3                   src/g05/data/base_lerobot_datasetV3.py
#   │     子类：只重写 __init__ 与统计相关的三个方法（_get_parquet_columns /
#   │           _fast_get_dataset_stats / get_dataset_stats，走 pyarrow 批量读，快 10~50×），
#   │           其余全部逻辑（本文件）原样复用；v3.0 数据集配置走这条路径。
#   ├─ GalaxeaLerobotDataset                  src/g05/data/galaxea_lerobot_dataset.py
#   ├─ DroidLerobotDataset                    src/g05/data/droid/droid_lerobot_dataset.py
#   └─ SO100CanonicalLerobotDatasetV3         src/g05/data/so100_canonical_dataset.py
#        │  (子类钩子：_get_additional_data / _resample_random_idx / _build_delta_timestamps)
#        ▼
#   MultiLeRobotDataset                       src/g05/data/lerobot/lerobot_dataset{,_v3}.py
#      │  · 把多个数据集目录“首尾拼接”成一个索引空间
#      │  · 按 delta_timestamps 精确读取帧（数值列走 parquet，图像走 mp4 解码）
#      ▼
#   parquet（数值列） + mp4 / png（相机画面）
#
# 【新手先记住这 3 个概念】
#
#   (1) shape_meta —— “声明式维度契约”，写在各 config 里，每个 part 一条记录：
#         key          内部 key（训练侧用的名字，如 left_arm / head_rgb）
#         lerobot_key  原始 parquet 列名（如 action.left_arm / observation.state）
#         start_index  从这个原始列的第几维开始切
#         raw_shape    切出来的宽度（int）或形状（list，图像用 [C,H,W]）
#         shape        经过 processor 变换后的目标形状（通常与 raw_shape 相同）
#         time_offset  时间偏移（0 = 当前帧；1 = 下一帧 t+1，用于 state-as-action）
#         camera_type  仅图像需要：head / wrist_left / wrist_right ...
#         dummy: true  仅图像：YAML 里占位但数据里不存在的相机（补零张量）
#       详见 docs/data/schema_zh.md 与 configs/data/parts_meta/*.yaml。
#
#   (2) delta_timestamps —— “这一次要读第几帧”的查询计划，形如
#         {"action.left_arm":    [0.0, 1/fps, ..., (H-1)/fps],   # 未来 H 步 = action chunk
#          "observation.state":  [0.0]}                          # 当前 1 帧（单帧配置）
#       单位是“秒”，由 _build_delta_timestamps() 从 fps + action_size + obs_size 推出
#       （例：fps=30、action_size=32 → 0.0, 0.0333, ..., 1.0333）。
#       同一原始列被多个 part 共用时，本文件会去重合并成一个列表（见 _append_offsets）。
#
#   (3) query_positions —— “读回来的那一大块，怎么切回各自的 part”。
#       底层会为每个 lerobot_key 返回 [查询帧数, 原始维度] 的张量；
#       query_positions[i] 告诉第 i 个 part“你要的帧在上面的第几个位置”，
#       于是 index_select 就能切出这个 part 专属的 [帧数, 该 part 维度]。
#
# 【一次训练取数（__getitem__）发生了什么】
#   1. 把 DataLoader 给的“局部下标 idx”换算成全局样本下标（+ _start_idx）；
#      overfit 模式下直接查预先筛好的稳定样本列表 _overfit_indices；
#   2. 调 multi_dataset[sample_idx]：底层按 delta_timestamps 读一帧窗口，
#      返回 {"action.left_arm": [H,6], "observation.state": [1,12], ...,
#            "action.left_arm_is_pad": [H], "task": "pick up the cup", ...}；
#   3. 逐 part 切片（_get_state / _get_action / _get_image）→ 组装成嵌套 dict；
#   4. 合并 is_pad 掩码（越界帧 / 子任务切换帧 / 未达标样本）；
#   5. 透传额外字段（task / coarse_task / atomic_task ...，供 CoT 模板使用）；
#   6. processor.preprocess(sample)：归一化 + padding 合并 + 构造 samples 模板；
#   7. 出错就记一条 invalid 记录，换一个随机下标重试（最多 MAX_GETITEM_ATTEMPT 次）。
#
# 【输出契约（发给 DataLoader 的一个样本）】
#   与 docs/architecture/g05_io_zh.md 保持一致，processor 未设置时大致为：
#     {
#       "idx":             int,                        # 全局样本下标（调试用）
#       "task":            str,                        # 语言指令
#       "dataset_locator": str,                        # "dataset_dir=..., local_idx=..." 便于定位
#       "frequency":       int,                        # 模型控制频率 → action tokenizer
#       "action":  {key: Tensor[H, D_raw]},            # H = action_size 的 raw 值
#       "state":   {key: Tensor[T_obs, D_raw]},        # T_obs = obs_size
#       "images":  {key: Tensor[T_obs, 3, H, W]},      # uint8
#       "action_is_pad": Tensor[H]   (bool),           # True = 该步不算 loss
#       "state_is_pad":  Tensor[T_obs] (bool),
#       "image_is_pad":  Tensor[...] (bool),           # 仅当存在图像
#       ... 其余 lerobot 原始字段（task_index / coarse_task / atomic_task / bbox_index ...）
#     }
#   设置 processor 后，返回的是 processor.preprocess() 的结果（多了 pixel_values 等）。
#
# 【常用术语对照】
#   data_fps     数据集真实录制频率（读自 meta/info.json），决定“一帧 = 多少秒”
#   model_fps    模型控制频率（override_fps），只影响 action tokenizer 的编解码，不影响读数
#   obs_size     观测帧数（image 与 state 共用），1 = 单帧；>1 = MEM 多帧历史
#   obs_stride   相邻观测帧间隔的“步数”（由 obs_stride_second × data_fps 得到）
#   action_size  action chunk 长度（未来多少步），H
#   tolerance_s  图像查帧时间容差（默认 0.4/data_fps ≈ 13ms @30fps），
#                用来吸收 ffmpeg H.264 重编码带来的时间戳抖动
#   *_is_pad     底层的越界掩码：查询帧跨出 episode 边界时该位置为 True
#   query_positions  见上文 (3)
#
# 【常见坑 / 排查清单】
#   · 报 KeyError “meta ... is missing fields”：shape_meta 少了 key/lerobot_key/
#     start_index/raw_shape/shape（图像还要 camera_type）。
#   · 报 “contains forbidden fields”：写了 _FORBIDDEN_META_KEYS 里的字段
#     （历史版本的语义字段已废弃，切分一律用 start_index 显式声明）。
#   · 报 “Missing pad key ..._is_pad”：该 lerobot_key 没有被注册进 delta_timestamps，
#     底层就不会产出对应掩码（子类 _build_delta_timestamps 覆盖时容易漏）。
#   · 图像全是黑色：相机 key 是 dummy: true，本来就注入零张量。
#   · 取数一直重试并伴随 warning：样本被判定为不达标（step_is_qualified=False）
#     或读帧失败，可用 get_invalid_sample_report() 拿到汇总清单。
#   · 训练/验证集怎么切：episode 级切分，val 取“后面 val_set_proportion 比例的 episode”，
#     保证验证集是完整轨迹，不会出现同一 episode 一半训练一半验证。
#
# 【相关文件与文档】
#   src/g05/data/mixture_lerobot_dataset.py            多 embodiment 混合 + 权重 + 统计量汇总
#   src/g05/data/lerobot/lerobot_dataset.py            v2.1 底层实现（delta_timestamps 的实际语义）
#   src/g05/data/lerobot/lerobot_dataset_v3.py         v3.0 底层实现
#   src/g05/data/base_lerobot_datasetV3.py             本类的 v3 快速版子类
#   src/g05/data_processor/processor/base_processor.py processor.preprocess / action_state_transform
#   docs/data/schema_zh.md                             shape_meta 字段与语义
#   docs/architecture/g05_io_zh.md                     Dataset → Collate → 模型的张量形状
#   docs/architecture/qwen_mem_overview_zh.md          obs_size / obs_stride_second 的多帧设计
#   configs/data/r1lite.yaml, configs/data/r1pro.yaml  可直接对照的完整配置示例
# =============================================================================

# bisect：用于把全局样本下标 O(log N) 映射回“第几个数据集目录 + 目录内第几帧”
import bisect
# multiprocessing：只用来创建一个跨进程共享的 Manager，保存“坏样本”记录
import multiprocessing as mp
# traceback：底层 MultiLeRobotDataset 构造失败时会打印堆栈（这里保留导入以便调试）
import traceback
# 线程池：get_dataset_stats() 并行遍历 episode 算统计量
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Union

import numpy as np
import torch
from tqdm import tqdm

from g05.data_processor.processor.base_processor import BaseProcessor
from g05.utils.logging.logging_config import get_logger

logger = get_logger(__name__)

# 单个样本最多尝试多少次（“读样本 → 组装”两个阶段合计的上限）。
# 触顶说明数据集本身有系统性问题（列缺失、时间戳错乱等），直接抛错而不是无限重试。
MAX_GETITEM_ATTEMPT = 10000
# 旧版配置里出现过的“语义化字段”黑名单。
# 现在的约定是：所有切分都必须用 start_index + raw_shape 显式描述，
# 不允许再写 source/target_key/... 这类“隐式推断”字段——一旦出现就立刻报错，
# 避免新旧语义混用导致静默取错维度的数据。
_FORBIDDEN_META_KEYS = {
    "source",
    "target_key",
    "target_offset",
    "target_from",
    "semantic_key",
    "resolved_lerobot_key",
    "resolved_start_index",
}


def _to_plain(obj):
    """Recursively convert OmegaConf containers to plain Python objects."""
    # Hydra/OmegaConf 传给类的 shape_meta 往往是 DictConfig / ListConfig（ΩConf 容器），
    # 它们虽然“看起来像” dict/list，但 isinstance(x, (list, tuple)) 之类的判断会失败，
    # 而且里面可能含有未解析的插值 ${...}。
    # 这里统一递归转换成原生 dict/list（resolve=True 会就地展开插值），
    # 之后本文件的读写、切片、索引都按普通 Python 对象处理，行为可预期。
    # 未安装 omegaconf（例如脱离 Hydra 单独实例化）时直接原样返回，保持兼容。
    if obj is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


def shift_sequence_with_replication(data: torch.Tensor, offset: int) -> torch.Tensor:
    """   本质就是 “state-as-action” 构造
    Shift a single-episode sequence by `offset`.

    Args:
        data: Episode-level tensor with shape `(T, D)` or `(T,)`.
        offset: Temporal offset applied to the first dimension. Positive values
            mean "look into the future"; negative values mean "look into the past".

    Returns:
        Tensor with the same shape as `data`. Indices that would cross the
        episode boundary are clamped to the first/last valid frame, so this
        function never crosses episodes and naturally implements the padding
        policy used for target construction.
    """
    # 【用途】把一条完整 episode 的序列整体平移 offset 帧，用于 time_offset != 0 的场景：
    #   例：time_offset = 1 表示“用 t+1 帧当监督目标”（state-as-action 构造）。
    # 【为什么用 clamp 而不是丢帧/补零】越界的位置复制首帧/末帧，好处有三：
    #   1) 输出长度与输入完全一致（下游按 episode 长度对齐时不需特殊处理）；
    #   2) 永远不会跨到相邻 episode 去取帧（episode 之间没有时间连续性，串帧=错标签）；
    #   3) 与底层 _get_query_indices 的 padding 策略一致（越界即 clamp，同时置 is_pad）。
    # 默认按第 0 维平移；一维输入先补一个特征维，保证后面统一按 (T, D) 处理。
    if data.ndim == 1:
        data = data.unsqueeze(-1)
    length = data.shape[0]
    # [0,1,...,T-1] + offset，再把结果夹到 [0, T-1]（clamp_ 原地操作，省一次拷贝）
    indices = torch.arange(length, device=data.device) + offset
    indices = indices.clamp_(0, length - 1)
    # 花式索引实现整体平移：out[i] = data[clamp(i + offset)]
    return data[indices]


class BaseLerobotDataset(torch.utils.data.Dataset):
    """一个 embodiment 的数据集：多个 LeRobot 目录 → 可直接喂给 DataLoader 的样本。

    职责边界（新手最容易混淆的点）：
      · 本类不做“混合多个 embodiment”的事（那是 MixtureLerobotDataset 的活），
        但允许一个实例里包含多个数据集目录（dataset_dirs），它们会被底层首尾拼接；
      · 本类不做数值归一化 / padding 合并（那是 processor 的活），
        但负责在 __getitem__ 里调用 processor.preprocess()；
      · 本类不直接读 parquet/mp4（那是 MultiLeRobotDataset 的活），
        只负责“告诉它读哪些帧、并把结果切成 part”。

    生命周期（与 scripts/finetune.py 的调用顺序一致）：
        ds = BaseLerobotDataset(...)      # 建索引、拼多数据集、切分 train/val
        stats = ds.get_dataset_stats(...) # 可选：统计 action/state 分布（归一化用）
        ds.set_processor(processor)       # 注入处理器（训练/评估模式）
        DataLoader(ds, num_workers=N)     # 多进程取数，每步调用 ds[idx]
        ds.enable_overfit(n)              # 可选：小样本过拟合调试
        ds.get_invalid_sample_report()    # 可选：查看被跳过的坏样本
    """
    # 类级（进程内共享）的 MultiLeRobotDataset 缓存：
    # 同一个进程里 MixtureLerobotDataset 可能为多个 embodiment 建实例，
    # 若它们的 (目录列表, delta_timestamps, tolerance, load_images, in_memory) 完全相同，
    # 就直接复用已有的底层数据集，避免重复读 parquet 元数据。
    _multi_dataset_cache = {}
    # 懒创建的 multiprocessing.Manager：它提供的 dict/list 可以在 DataLoader 的
    # 多进程 worker 之间共享，用来汇总“哪些样本坏了、坏了多少次”。
    _shared_manager = None
    _cache_hit_count = 0  # tracks how many datasets were reused from cache (for summary log)

    @staticmethod
    def _make_cache_key(dataset_dirs, delta_timestamps, tolerances_s, load_images, in_memory=False):
        # 把“会影响底层数据集内容与读法”的全部入参压成一个可哈希的元组：
        # 目录（排序后比较，与传入顺序无关）、每列的查询偏移（转 tuple 才可哈希）、
        # 容差、是否读图像、是否常驻内存。五项一致 ⇒ 底层样本完全一致 ⇒ 可安全复用。
        ds_key = tuple(sorted(dataset_dirs))
        dt_key = tuple(sorted((k, tuple(v)) for k, v in delta_timestamps.items()))
        tol_key = tuple(sorted(tolerances_s.items()))
        return (ds_key, dt_key, tol_key, bool(load_images), bool(in_memory))

    @classmethod
    def clear_cache(cls):
        # 训练结束时由 scripts/finetune.py 调用，释放被缓存占用的底层数据集对象。
        cls._multi_dataset_cache.clear()

    @classmethod
    def _get_shared_manager(cls):
        # 整个进程只创建一个 Manager（创建它会额外 fork 一个子进程，代价不低），
        # 并且必须“用到才建”，避免模块导入期就起子进程。
        if cls._shared_manager is None:
            cls._shared_manager = mp.Manager()
        return cls._shared_manager

    def __init__(
        self,
        dataset_dirs: List[str],
        # shapes
        shape_meta: Dict[str, Any],
        action_size: int,
        past_action_size: int = 0,  # Excludes the current frame.
        obs_size: int = 1,
        obs_stride_second: float = 0.0,
        # train vs val
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        # lerobot_ds_version
        lerobot_ds_version: Optional[Literal["2.1", "3.0"]] = "2.1",
        # tolerance
        tolerance_s: Optional[float] = None,
        # fps override
        override_fps: Optional[int] = None,
        load_images: Optional[bool] = None,
        in_memory: bool = False,
        **kwargs,
    ):
        """构造一个 embodiment 数据集（全部耗时工作都在这里一次性做完）。

        参数（按“作用域”分组，与上面签名里的分节注释对应）：
          dataset_dirs          数据集根目录列表，每个目录是一个标准 LeRobot 数据集
                                （内含 meta/info.json、data/*.parquet、videos/*.mp4）。
                                多个目录会被底层首尾拼接成同一个索引空间，
                                且会先检查它们存在公共列（没有公共列直接报错）。
          shape_meta            action/state/images 的维度声明，见文件头“3 个概念 (1)”。
          action_size           action chunk 长度 H（预测未来多少步）。
          past_action_size      历史动作步数；当前实现强制为 0（动作只向前看）。
          obs_size              观测帧数（state 与 image 共用，1 = 单帧）。
          obs_stride_second     相邻观测帧间隔（秒）；0 = 每帧都取（stride=1 步）。
                                用“秒”而不是“步”是为了让 15fps 与 30fps 的数据集
                                在时间尺度上取到同样长的历史窗口。
          val_set_proportion    验证集占比（按 episode 数切分），1e-6 以下视为不切分。
          is_training_set       True=训练集（取前 (1-prop) 的 episode），False=验证集。
          lerobot_ds_version    底层实现版本："2.1" / "3.0"（见文件头链路图）。
          tolerance_s           图像查帧时间容差；None = 自动 0.4/data_fps。
          override_fps          模型控制频率；None = 用数据集的 data_fps。
                                它只进入 sample["frequency"] 供 action tokenizer 使用，
                                不改变任何“读哪一帧”的计算。
          load_images           是否读图像；None = 有图像 meta 就自动读；
                                显式 False 可跳过视频解码（例如只为算统计量时）。
          in_memory             True 时把数值列常驻内存（v3 路径生效），换取更快取数。
          **kwargs              吸收配置里的其他键（例如 embodiment_type、
                                future_task_offset 等由子类消费的字段），保持前向兼容。
        """
        # 至少要有一个数据集目录，否则后续索引运算没有意义。
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        # 当前实现刻意不支持 past action：动作查询固定从当前帧开始。
        # 若将来要支持，需要同时改 _build_delta_timestamps 的 action 偏移与下游 shape。
        assert past_action_size == 0

        # ---------- 1. 保存基础超参 ----------
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = int(obs_size)
        assert self.obs_size >= 1, f"obs_size must be >= 1, got {self.obs_size}"
        self.obs_stride_second = float(obs_stride_second)
        self.processor = None  # Will be set externally
        # ---------- 2. 创建跨进程共享的“坏样本”登记簿 ----------
        # dict：key = "阶段|样本下标|位置" → 记录（次数、错误类型、原始 idx）；list：保序，便于稳定输出报告。
        # 用 Manager 的容器而非普通 dict/list，是因为 DataLoader 的多个 worker 进程都要往里写，
        # 汇总报告时（get_invalid_sample_report）主进程才能看到全部 worker 的记录。
        manager = self._get_shared_manager()
        self._invalid_sample_records = manager.dict()
        self._invalid_sample_order = manager.list()
        metas = []
        self.lerobot_ds_version = lerobot_ds_version
        # ---------- 3. 按版本选择底层实现 ----------
        # 两个版本对外接口一致（LeRobotDatasetMetadata / MultiLeRobotDataset 同名），
        # 差别只在内部读法：v2.1 逐帧 select，v3.0 批量化 + 支持 in_memory。
        # 这里刻意“按需 import”：避免 train 时同时把两套读实现（及各自的依赖）加载进内存。
        if lerobot_ds_version == "2.1":
            from g05.data.lerobot.lerobot_dataset import (
                LeRobotDatasetMetadata,
                MultiLeRobotDataset,
            )

            # v2.1 是历史路径，新数据集建议用 v3.0，所以打一条 warning 提醒。
            logger.warning(
                f"[lerobot_ds_version=2.1] {dataset_dirs[0]} "
                f"(+{len(dataset_dirs) - 1} more dirs) — using legacy v2 dataset path"
            )
        elif lerobot_ds_version == "3.0":
            from g05.data.lerobot.lerobot_dataset_v3 import (
                LeRobotDatasetMetadata,
                MultiLeRobotDataset,
            )
        else:
            raise ValueError(f"Unsupported lerobot_ds_version: {lerobot_ds_version}")

        # ---------- 4. 读取每个目录的元数据（轻量：只读 meta/info.json） ----------
        # repo_id 直接用本地目录路径（这套代码不联 HuggingFace Hub，全部本地读盘）。
        # meta 里包含：fps、features（列名与形状）、total_episodes、tasks 等。
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)
        try:
            # data_fps: actual recording frequency read from meta/info.json.
            # All data-read fields, including delta_timestamps and tolerance_s, use it.
            data_fps = meta.fps
        except Exception as e:
            # 元数据缺 fps 时不要直接崩（有些内部数据集 info.json 不完整），
            # 退回 15fps 这个常见默认值，同时打 warning 让使用者知道精度可能受影响。
            logger.warning(f"Failed to read fps from dataset meta, falling back to 15: {e}")
            data_fps = 15
        self.fps = data_fps

        # ---------- 5. 把“秒”换算成“步” ----------
        # obs_stride: convert time-based stride (seconds) to step count using actual fps
        if self.obs_stride_second > 0:
            # 例：obs_stride_second=1.0，fps=30 → stride=30 步（相邻观测帧相隔 1 秒）。
            # max(1, ...) 保证至少隔 1 步，避免 round 到 0 导致所有帧都取同一帧。
            self.obs_stride = max(1, round(self.obs_stride_second * data_fps))
        else:
            self.obs_stride = 1

        if self.obs_size > 1:
            # 多帧（MEM）时才打印，单帧日志没有信息量。
            # history_sec = 从最早一帧到当前帧的时间跨度 = (obs_size-1) × stride / fps。
            history_sec = (self.obs_size - 1) * self.obs_stride / data_fps
            logger.info(
                f"[obs] {dataset_dirs[0]}: obs_size={self.obs_size}, "
                f"obs_stride_second={self.obs_stride_second}s → stride_steps={self.obs_stride} "
                f"(fps={data_fps}), history_window={history_sec:.1f}s"
            )

        # ---------- 6. data_fps vs model_fps（两个频率别混用） ----------
        # override_fps: model control frequency, i.e. the FPS at which the model runs.
        # It is unrelated to data reads and is only passed to the action tokenizer
        # through the "frequency" field. Defaults to data_fps when unset.
        self.model_fps = override_fps if override_fps is not None else data_fps
        if override_fps is not None:
            logger.info(f"model_fps={self.model_fps} (override), data_fps={data_fps} (from meta)")

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        # ---------- 7. 解析 shape_meta（拆成 image / state / action 三组） ----------
        # Convert meta lists to plain Python so OmegaConf ListConfig/DictConfig
        # don't leak into downstream isinstance(x, (list, tuple)) checks.
        # Filter out dummy cameras: they have no lerobot_key in YAML, so the base
        # class neither validates nor loads them. Entries with dummy: true are
        # injected as zero tensors by _inject_dummy_images in __getitem__.
        all_image_meta = _to_plain(shape_meta.get("images", None)) or []
        # dummy 相机（数据集里真实不存在，但模型/processor 需要占位）单独存放：
        # 它们没有 lerobot_key，所以不参与校验、不参与读盘，只在组装样本时补零。
        self._dummy_image_meta = [m for m in all_image_meta if m.get("dummy", False)]
        # 真实相机；空列表统一转成 None，便于下游用 `if self.image_meta:` 判断“有没有图像”。
        self.image_meta = [m for m in all_image_meta if not m.get("dummy", False)] or None
        self.state_meta = _to_plain(shape_meta["state"])
        self.action_meta = _to_plain(shape_meta["action"])
        # load_images 的三态语义：
        #   None（未指定）→ 有图像 meta 就读；
        #   True/False（显式指定）→ 以调用方为准，但最后再与“是否真的有图像 meta”相与，
        #   即：没有图像 meta 时绝不会去读图（避免底层白白解码视频）。
        default_load_images = bool(self.image_meta)
        self.load_images = default_load_images if load_images is None else bool(load_images)
        self.load_images = self.load_images and default_load_images
        # return_images 是“对外是否返回 images 字段”的开关，正常等于 load_images；
        # 子类可用 _set_return_images(False) 临时关掉图像（例如只取若干帧的位置做统计），
        # 此时 load_images 与 return_images 会同步变化，避免出现“不返回却仍在解码视频”的浪费。
        self.return_images = self.load_images
        self.in_memory = in_memory
        if in_memory:
            logger.info(
                f"[in_memory] BaseLerobotDataset: in_memory=True, "
                f"version={lerobot_ds_version}, dirs={len(dataset_dirs)}, "
                f"first_dir={dataset_dirs[0]}"
            )

        # 校验并规范化每个 meta 的显式切片字段（缺字段/写了废弃字段都会在这里报错）。
        # Validate explicit raw-layout fields provided by config.
        self._setup_meta_lerobot_keys()

        # ---------- 8. 生成“读哪些帧”的查询计划 ----------
        # Build delta_timestamps for querying frames from parquet.
        # data_fps is the actual recording frequency and determines the interval
        # between adjacent query points: 1/data_fps seconds per step.
        delta_timestamps = self._build_delta_timestamps(data_fps, past_action_size, action_size)

        # ---------- 9. 训练/验证集切分（episode 级） ----------
        # 按 episode 而非“帧”切分：验证集是一整条条轨迹，避免同一 episode 的相邻帧
        # 一半进训练一半进验证，造成信息泄漏（相邻帧高度相似）。
        episodes = {}
        if val_set_proportion < 1e-6:
            # 不切分：每个数据集目录的全部 episode 都归本实例。
            for meta in metas:
                episodes.update({meta.repo_id: list(range(meta.total_episodes))})
        else:
            for meta in metas:
                # split_idx 之前的 episode 归训练，之后的归验证。
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                if self.is_training_set:
                    episodes.update({meta.repo_id: list(range(split_idx))})
                else:
                    episodes.update({meta.repo_id: list(range(split_idx, meta.total_episodes))})

        if lerobot_ds_version == "3.0":
            # v3 的 MultiLeRobotDataset 自行处理 episode 选择（走 _start_idx/_end_idx 切分），
            # 这里传 None 表示“底下先加载全部 episode”，由本类在后面按帧区间收窄。
            episodes = None

        # 单独 import time 而不是放文件顶部：极小化改动面（局部使用）。
        import time

        start_time = time.time()
        # ---------- 10. 容差与缓存键 ----------
        # tolerance_s: parquet timestamp matching tolerance, based on the actual
        # recording frequency with frame spacing 1/data_fps.
        # 0.4/data_fps is about 13.3 ms at 30 fps, accommodating 0.6-4 ms timestamp
        # drift caused by ffmpeg H.264 re-encoding.
        self.tolerance_s = tolerance_s if tolerance_s is not None else 0.4 / data_fps
        logger.info(
            f"tolerance_s={'user-specified' if tolerance_s is not None else 'auto (0.4/data_fps)'}: {self.tolerance_s:.6f}s (data_fps={data_fps}, model_fps={self.model_fps})"
        )
        # 每个目录用同一个容差（dict.fromkeys 让所有 key 指向同一个值）。
        tolerances_s = dict.fromkeys(self.dataset_dirs, self.tolerance_s)
        cache_key = self._make_cache_key(
            self.dataset_dirs,
            delta_timestamps,
            tolerances_s,
            self.load_images,
            self.in_memory,
        )
        if cache_key in BaseLerobotDataset._multi_dataset_cache and lerobot_ds_version == "3.0":
            # 命中缓存：直接复用已构建的底层数据集（省掉一次 parquet 元数据扫描）。
            # 注意：只有 v3.0 会读取缓存；v2.1 只写不读（历史行为，保持兼容不改动）。
            self.multi_dataset = BaseLerobotDataset._multi_dataset_cache[cache_key]
            BaseLerobotDataset._cache_hit_count += 1
            logger.debug(f"Reusing cached MultiLeRobotDataset (cache hit)")
        else:
            # 走真正的构造：这一步会读各目录的 meta、parquet schema，建立 episode 索引，
            # 是 __init__ 里最耗时的一步，因此日志里单独打印耗时。
            self.multi_dataset = MultiLeRobotDataset(
                dataset_dirs=self.dataset_dirs,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                tolerances_s=tolerances_s,
                load_images=self.load_images,
                in_memory=self.in_memory,
            )
            BaseLerobotDataset._multi_dataset_cache[cache_key] = self.multi_dataset
            logger.debug(f"MultiLeRobotDataset initialized in {time.time() - start_time:.2f}s")
        # 无论命中缓存还是新建，都显式同步一次图像开关，
        # 避免复用到的实例保留了上一个使用者的 load_images 状态。
        self.multi_dataset.set_load_images(self.load_images)

        # ---------- 11. 建立“全局帧下标 → 第几个数据集目录”的查询表 ----------
        # Build cumulative frame boundaries for O(log N) sample-to-dataset lookup
        # 例：三个目录各有 100/50/80 帧 → _ds_cumulative_frames = [100, 150, 230]，
        # 全局下标 120 落在第二个目录（本地下标 20）。_locate_sample() 用 bisect 做这件事。
        cumulative = 0
        self._ds_cumulative_frames = []
        for ds in self.multi_dataset._datasets:
            cumulative += ds.num_frames
            self._ds_cumulative_frames.append(cumulative)

        # ---------- 12. 拼接全局 episode 索引 ----------
        # 每个子数据集里的 episode_data_index 是“局部的”（帧下标从 0 开始），
        # 这里按前面累加的 end_index 平移成“全局帧下标”，拼成一张全量索引表：
        #   episode_data_index["from"][e] ~ ["to"][e] 就是第 e 条（全局）episode 的帧区间 [from, to)。
        # 下游按 episode 遍历（统计量、初始化位置等）都依赖这张表。
        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }
        # ---------- 13. 用帧区间表达“本实例负责的数据范围” ----------
        # 对外长度 = _end_idx - _start_idx，取样本时再把局部 idx 平移 + _start_idx。
        # dataset range: [self._start_idx, self._end_idx)
        if lerobot_ds_version == "3.0" and val_set_proportion > 1e-6:
            # v3 路径下 episodes=None，无法在建底层实例时就限定 episode，
            # 因此在这里按 episode 边界把帧区间算出来（而不是在 episode 粒度的列表里）。
            total_episodes = len(self.episode_data_index["from"])
            split_ep = int(total_episodes * (1 - val_set_proportion))
            # at least one episode for training and validation to avoid empty dataset
            # 兜底：数据量很小时防止切出空集合（训练/验证各至少 1 条 episode）。
            split_ep = max(1, split_ep)
            val_start_ep = min(split_ep, total_episodes - 1)
            if self.is_training_set:
                # 训练集：从第 0 帧到第 (split_ep-1) 条 episode 的结束帧。
                self._start_idx = 0
                self._end_idx = self.episode_data_index["to"][split_ep - 1].item()
            else:
                # 验证集：从第 val_start_ep 条 episode 的起始帧，到最后一条 episode 的结束帧。
                self._start_idx = self.episode_data_index["from"][val_start_ep].item()
                self._end_idx = self.episode_data_index["to"][-1].item()
        else:
            # v2.1（或不需要切分）：底层已在构造时按 episodes 过滤，
            # 它的 num_frames 就是本实例的全部样本数。
            self._start_idx = 0
            self._end_idx = self.multi_dataset.num_frames

    def _setup_meta_lerobot_keys(self):
        """Configs must provide the explicit raw layout for every meta."""
        # 【这一步干什么】把 YAML 里的 shape_meta 逐条“体检 + 规范化”，
        # 任何配置错误都在构造阶段就炸掉（fail fast），而不是等到训练跑几千步才报错。
        #
        # 三类 meta 的必填字段（required）不同：
        #   state / action : key, lerobot_key, start_index, raw_shape, shape
        #   images         : 上面 5 个 + camera_type（processor 按它决定图像增强与用途）
        # 一次性列出缺了哪些字段，方便使用者一次改完。
        #
        # 校验通过后还会做三件“规范化”：
        #   start_index  强制转 int（YAML 里可能写成字符串或浮点）
        #   time_offset  缺省补 0，并强制 int
        #   query_positions 先置 None，稍后由 _build_delta_timestamps 填成具体位置下标
        # 注意：这里直接原地修改 self.image_meta / state_meta / action_meta 里的 dict，
        # 后续所有方法读到的都是这份“已规范化”的 meta。
        for group_name, metas in (
            ("images", self.image_meta or []),
            ("state", self.state_meta),
            ("action", self.action_meta),
        ):
            for meta in metas:
                required = ["key", "lerobot_key", "start_index", "raw_shape", "shape"]
                required += ["camera_type"] if group_name == "images" else []
                missing = [field for field in required if field not in meta]
                if missing:
                    raise KeyError(
                        f"{group_name} meta for key={meta.get('key')!r} is missing fields: {missing}."
                    )
                # 禁止使用已废弃的语义化字段（见文件顶部 _FORBIDDEN_META_KEYS 说明），
                # 因为它们会与 start_index/raw_shape 的显式切分语义冲突。
                unexpected = [field for field in _FORBIDDEN_META_KEYS if field in meta]
                if unexpected:
                    raise ValueError(
                        f"{group_name} meta for key={meta.get('key')!r} contains forbidden fields: {unexpected}."
                    )
                key = meta["key"]
                # key 是训练侧唯一的 part 名（processor、模型都按它取数），空值会造成难以定位的错误。
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(
                        f"{group_name} meta key must be a non-empty string, got {key!r}."
                    )
                meta["start_index"] = int(meta["start_index"])
                if "time_offset" in meta:
                    meta["time_offset"] = int(meta["time_offset"])
                else:
                    meta["time_offset"] = 0
                meta["query_positions"] = None

    @staticmethod
    def _append_offsets(
        delta_timestamps: Dict[str, List[float]], lerobot_key: str, offsets: List[float]
    ) -> List[int]:
        """
        Merge requested offsets into one lerobot query list for a raw key.

        Args:
            delta_timestamps: Global query plan being built for
                `MultiLeRobotDataset`. Each raw lerobot key maps to a list of
                time offsets in seconds.
            lerobot_key: Raw lerobot column to update, e.g.
                `observation.state.left_arm`.
            offsets: Offsets requested by one meta for that raw key.

        Returns:
            A list of integer positions. Each position tells the caller where
            its requested offset lands inside `delta_timestamps[lerobot_key]`
            after de-duplication.
        """
        # 【为什么需要去重】同一个原始列经常被多个 part 共用，典型有两种情况：
        #   1) state 与 action 都指向 observation.state（state-as-action 训练方式）；
        #   2) 多个 part 来自同一列的不同维度段（left_arm 取 [0:6]，gripper 取 [6:7]）。
        # 若不去重，底层会把同一帧读很多遍（I/O 与显存都浪费）。
        #
        # 做法：对每个请求偏移，先查是否已存在（list.index 即“相等则复用”），
        # 不存在才 append。返回的 positions 是“该请求落在合并列表中的下标”，
        # 所以调用方拿到的是一组位置编号，而不是偏移本身。
        #
        # 例：先请求 state 的 [-1/30, 0] → 列表 [−0.0333, 0.0]，返回 [0, 1]；
        #     再请求 action 的 [0, 1/30] → 0.0 已存在（返回 1），1/30 新增（返回 2），
        #     列表变为 [-0.0333, 0.0, 0.0333]，第二次返回 [1, 2]。
        # 之后读回的张量形状是 [3, D]，state 取第 [0,1] 行，action 取第 [1,2] 行。
        existing = delta_timestamps.setdefault(lerobot_key, [])
        positions = []
        for offset in offsets:
            try:
                pos = existing.index(offset)
            except ValueError:
                existing.append(offset)
                pos = len(existing) - 1
            positions.append(pos)
        return positions

    def _build_delta_timestamps(self, fps, past_action_size, action_size) -> Dict[str, list]:
        """   这个方法根本不关心数据有多长，它只负责"相对当前帧，往前/往后取哪几帧"。
        真正的边界处理，是底层数据集在拿到"当前帧索引"之后才做的。
        
        Build the raw lerobot query plan used by `MultiLeRobotDataset`.

        Args:
            fps: Dataset frequency. Integer frame offsets are converted into
                seconds by dividing with `fps`.
            past_action_size: Number of past action steps. Current dataset
                assumes `0`.
            action_size: Number of action target steps returned to training.

        Returns:
            Dict mapping raw lerobot keys to the list of offsets (in seconds)
            that should be fetched for each key.

        Side effects:
            - Fills `query_positions` for every state meta.
            - Fills `query_positions` for every action meta.

        Important detail:
            State keys and action keys may point to the same raw lerobot column.
            `_append_offsets()` merges those requests into one offset list and
            records which positions belong to which meta, so later
            `__getitem__` can slice the shared query result back into per-key
            tensors.

        obs_size / obs_stride:
            obs_size frames, stride obs_stride steps apart (both image and state).
            Offsets: [-(obs_size-1)*obs_stride/fps, ..., -obs_stride/fps, 0]
        """
        # 【产出的东西】{"<raw 列名>": [偏移秒, ...]}，直接交给底层 MultiLeRobotDataset。
        # 【三组 meta 的取帧策略】
        #   images / state：向后看历史，取 obs_size 帧、相邻相隔 obs_stride 步，
        #                   偏移形如 [-(K-1)*stride, ..., -stride, 0]（最后一个元素总是 0 = 当前帧）。
        #                   单帧配置（obs_size=1）时就是 [0]。
        #   action：向前看未来，从当前帧起取 action_size 步（past_action_size 固定 0），
        #           偏移形如 [0, 1/fps, ..., (H-1)/fps]。
        #   time_offset 会整体平移这组偏移（例如 state 的 time_offset=1 → 取 t+1 帧）。
        #
        # 【一个关键区别】
        #   images 走的是“直接赋值”，因为一个相机 key 只会被它自己查询；
        #   state / action 走 _append_offsets_for_meta()，因为它们可能与他人共用同一 raw 列，
        #   必须去重合并并记录 query_positions（见 _append_offsets 的说明与例子）。
                # 因为它们可能共用同一原始列。比如 state 和 action 都指向同一列（state-as-action 场景）。这时候：
                # 两个 meta 的偏移要合并去重成一条请求，底层只取一次。
                # 合并后要记录每个 meta 需要的是合并结果里的哪几行，这就是 query_positions。
        query_offsets_by_key = {}

        obs_size = self.obs_size
        obs_stride = self.obs_stride

        # Images: sample obs_size frames spaced obs_stride steps apart.
        if self.image_meta is not None:
            for meta in self.image_meta:
                # reversed(range(0, -K*stride, -stride)) 生成 [-(K-1)*stride, ..., -stride, 0]：
                # 按“由旧到新”的时间顺序排列，因此最后一位永远是当前帧；
                # 后续图像张量的第 0 维就是这个时间顺序（旧 → 新），模型侧按同样约定解读。
                image_offsets = [
                    (meta["time_offset"] + step) / fps
                    for step in reversed(range(0, -obs_size * obs_stride, -obs_stride))
                ]
                query_offsets_by_key[meta["lerobot_key"]] = image_offsets

        # States: sample obs_size steps spaced obs_stride steps apart.
        for meta in self.state_meta:
            # 与图像完全相同的取帧规则，保证 state 与 image 在时间轴上严格对齐。
            state_offsets = [
                (meta["time_offset"] + step) / fps
                for step in reversed(range(0, -obs_size * obs_stride, -obs_stride))
            ]
            # 合并进全局计划，并记录“我需要的是合并后张量的哪几行”。
            query_positions = self._append_offsets_for_meta(
                query_offsets_by_key,
                meta["lerobot_key"],
                state_offsets,
            )
            meta["query_positions"] = query_positions

        # Actions query the raw source declared by their own meta.
        for meta in self.action_meta:
            # range(-past_action_size, action_size) 目前等价于 range(0, action_size) = [0..H-1]：
            # 第 0 个元素是当前帧（不包含历史帧），第 H-1 个元素是未来第 H-1 步。
            # 因为是“未来”，不参与去重合并逻辑的复用优化场景较少，但仍走同一套 positions 机制，
            # 以便 action 与 state 指向同一列（state-as-action）时也能正确切分。
            action_offsets = [
                (meta["time_offset"] + step) / fps for step in range(-past_action_size, action_size)
            ]
            query_positions = self._append_offsets_for_meta(
                query_offsets_by_key,
                meta["lerobot_key"],
                action_offsets,
            )
            meta["query_positions"] = query_positions

        return query_offsets_by_key

    @staticmethod
    def _is_multi_source_meta(meta: Dict[str, Any]) -> bool:
        # 【多源 meta】少数 part 需要把多个原始列“横向拼接”成一个 part，
        # 例如把 action.left_arm_end_pose（6 维）与 action.left_gripper（1 维）
        # 拼成 7 维的 left_arm。这类 meta 的 lerobot_key 写成 list：
        #   {key: left_arm, lerobot_key: [action.left_arm_end_pose, action.left_gripper],
        #    source_start_indices: [0, 0], source_raw_shapes: [6, 1], raw_shape: 7, shape: 7}
        # 见 _slice_meta_feature 的拼接分支。
        return isinstance(meta.get("lerobot_key"), (list, tuple))

    @staticmethod
    def _slice_single_meta_feature(data: torch.Tensor, start: int, width: int) -> torch.Tensor:
        # 从“原始列张量”里切出本 part 的那一段维度（最后一维）。
        # 先统一形状，保证末尾维度就是“特征维”：
        #   ndim == 0：标量（例如某些布尔/计数列），没有特征维可切，原样返回；
        #   ndim == 1：[C] → [C, 1]（单维特征，如夹爪），这样 [..., start:start+width] 才有意义；
        #   ndim > 2 ：把 [帧数, D1, D2] 之类的多维列先压平成 [帧数, D1*D2]，
        #              使 raw_shape 能按“宽”统一切分（图像不走这里，图像在 _get_image 处理）。
        if data.ndim == 0:
            return data
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        elif data.ndim > 2:
            data = data.reshape(*data.shape[:-2], -1)
        return data[..., start : start + width]

    @staticmethod
    def _slice_meta_feature(data: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        # 【统一入口】按 meta 声明把“原始数据”切成这个 part 应有的张量。
        # data 可能是：单源张量，或（多源时）张量列表；两种情况的处理路径不同。
        if BaseLerobotDataset._is_multi_source_meta(meta):
            # 多源：逐源按各自的起点/宽度切片，再在特征维（最后一维）上拼接。
            # 例：pose(6 维) + gripper(1 维) → (T, 7)，与 raw_shape=7 对应。
            starts = meta.get("source_start_indices")
            widths = meta.get("source_raw_shapes")
            if starts is None or widths is None:
                raise KeyError(
                    "Multi-source meta requires 'source_start_indices' and 'source_raw_shapes'."
                )
            if not isinstance(data, (list, tuple)):
                raise TypeError(
                    f"Multi-source meta expects a list/tuple of tensors, got {type(data)!r}."
                )
            if not (len(data) == len(starts) == len(widths)):
                raise ValueError("Multi-source meta lengths must match between sources and slices.")

            # zip(..., strict=True)：Python 3.10+ 的长度校验，长度不一致直接抛错，
            # 避免悄悄少拼一段导致维度对不上却查不出原因。
            parts = [
                BaseLerobotDataset._slice_single_meta_feature(source, start, width)
                for source, start, width in zip(data, starts, widths, strict=True)
            ]
            return torch.cat(parts, dim=-1)

        # 单源路径：从原始列的第 start_index 维开始取 raw_shape 宽的一段。
        start = meta["start_index"]
        width = meta["raw_shape"]
        # 图像（raw_shape 是 list，如 [3, 720, 1280]）不走数值切分：
        # 图像整帧由 _get_image 处理（它只做 dtype 转换，不做维度切片）。
        if not isinstance(width, int):
            return data
        return BaseLerobotDataset._slice_single_meta_feature(data, start, width)

    def _append_offsets_for_meta(
        self,
        query_offsets_by_key: Dict[str, List[float]],
        lerobot_key,
        offsets: List[float],
    ):
        # 按 meta 的类型分派：
        #   单源 meta（lerobot_key 是字符串）→ 直接返回一组 positions；
        #   多源 meta（lerobot_key 是列表）  → 每个源各返回一组 positions，
        #   于是 query_positions 变成“positions 的列表”，与源一一对应（见 _get_sample_*_tensor）。
        # 注意：actions 共享同一组 offsets（同一时间轴），所以可以复用同一份 offsets 列表。
        if isinstance(lerobot_key, (list, tuple)):
            return [self._append_offsets(query_offsets_by_key, key, offsets) for key in lerobot_key]
        return self._append_offsets(query_offsets_by_key, lerobot_key, offsets)

    @staticmethod
    def _get_meta_source_data(container: Dict[str, Any], meta: Dict[str, Any]):
        # 从“某个容器”（sample 或 episode 数据）里按 meta 的声明取出原始数据：
        #   单源 → 一个张量；多源 → 张量列表（长度与 lerobot_key 列表一致）。
        # 与 _slice_query_tensor 的区别：这里只做“取”，不做“挑帧”，
        # 供 get_episode_data 这类“整条 episode 已经读进来”的场景使用。
        lerobot_key = meta["lerobot_key"]
        if isinstance(lerobot_key, (list, tuple)):
            return [container[key] for key in lerobot_key]
        return container[lerobot_key]

    @staticmethod
    def _slice_query_tensor(data: torch.Tensor, positions: Optional[List[int]]) -> torch.Tensor:
        """
        Select the positions belonging to one meta from a shared query tensor.

        Args:
            data: Tensor returned by `MultiLeRobotDataset` for one raw lerobot
                key. The first dimension enumerates queried offsets.
            positions: Integer positions previously produced by
                `_append_offsets()`. `None` means "use the whole tensor".

        Returns:
            Tensor restricted to the requested positions, preserving the input
            dtype/device.
        """
        # 把“共享查询结果”按本 meta 需要的行挑出来。
        # 例：底层为 observation.state 返回 [3, D]（因为 state 与 action 共用，
        # 合并后的偏移有 3 个），而本 state meta 的 query_positions = [0, 1]，
        # 则 index_select(0, [0,1]) 得到 [2, D] —— 恰好是它要的两帧。
        # positions=None 表示“没登记过查询计划”（例如 get_episode_data 的整段数据），
        # 这时原样返回整块张量。
        # 用 index_select 而不是 data[positions] 是为了保持 dtype/device 不变并支持 GPU 上的索引张量。
        if positions is None:
            return data
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected tensor query result, got {type(data)!r}")
        pos = torch.as_tensor(positions, device=data.device, dtype=torch.long)
        return data.index_select(0, pos)

    def _get_sample_action_tensor(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one action key from the shared lerobot sample.

        Args:
            meta: One action meta after `_build_delta_timestamps()`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.

        Returns:
            Tensor for this action key with time dimension already sliced to the
            offsets specified by `query_positions`.
        """
        # 【三步走】挑帧 → 拼多源 → 切维度：
        #   1) _slice_query_tensor：从“共享查询结果”里挑出本 action 要的那几帧；
        #   2) 多源时对每个源各挑一次，得到张量列表；
        #   3) _slice_meta_feature：按 start_index/raw_shape 切出本 part 的维度
        #      （多源则先各自切片再在特征维拼接）。
        # 结果形状：[H, raw_shape]，H = action_size（未来 H 步）。
        if self._is_multi_source_meta(meta):
            action = [
                self._slice_query_tensor(lerobot_sample[key], positions)
                for key, positions in zip(meta["lerobot_key"], meta["query_positions"], strict=True)
            ]
        else:
            action = self._slice_query_tensor(
                lerobot_sample[meta["lerobot_key"]], meta["query_positions"]
            )
        return self._slice_meta_feature(action, meta)

    def _get_sample_state_tensor(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one state key from the shared lerobot sample.

        Args:
            meta: One state meta after `_build_delta_timestamps()`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.

        Returns:
            Tensor for this state key with time dimension already sliced to the
            offsets specified by `query_positions`.
        """
        # 与 _get_sample_action_tensor 逻辑完全一致，只是取的是 state 的帧窗口。
        # 结果形状：[T_obs, raw_shape]，T_obs = obs_size（当前帧在最后一行）。
        if self._is_multi_source_meta(meta):
            state = [
                self._slice_query_tensor(lerobot_sample[key], positions)
                for key, positions in zip(meta["lerobot_key"], meta["query_positions"], strict=True)
            ]
        else:
            state = self._slice_query_tensor(
                lerobot_sample[meta["lerobot_key"]], meta["query_positions"]
            )
        return self._slice_meta_feature(state, meta)

    def _get_pad_mask(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract the pad mask corresponding to one state/action query.

        Args:
            meta: State or action meta with resolved query information.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.

        Returns:
            Boolean tensor aligned with the query tensor produced for this meta.
            Pad mask aligned with this meta's declared raw source.
        """
        # 【掩码从哪来】底层读帧时发现“请求的帧落在本 episode 之外”就会 clamp 到边界帧，
        # 同时把该位置记成 True 放进 "{raw_key}_is_pad"（见 lerobot_dataset.py 的
        # _get_query_indices）。这里的任务就是把它对齐到本 meta 的帧顺序上。
        #
        # 例：episode 只有 40 帧，却请求未来 32 步 action，则末尾若干步越界，
        # action_is_pad 对应位置为 True —— 训练侧据此把这些步排除在 loss 之外。
        #
        # 多源 meta 只取第一个源的掩码：因为同一 meta 的各个源共用同一组时间偏移，
        # 越界与否只取决于时间轴，与源无关。
        query_key = (
            meta["lerobot_key"][0] if self._is_multi_source_meta(meta) else meta["lerobot_key"]
        )
        pad_key = f"{query_key}_is_pad"
        if pad_key not in lerobot_sample:
            # 常见原因：子类重写 _build_delta_timestamps 时漏注册某个 lerobot_key，
            # 底层就不会为它生成掩码（见文件头“常见坑”）。
            raise KeyError(f"Missing pad key {pad_key!r} in lerobot sample.")
        positions = (
            meta["query_positions"][0]
            if self._is_multi_source_meta(meta)
            else meta["query_positions"]
        )
        pad = self._slice_query_tensor(lerobot_sample[pad_key], positions)
        # 某些数据格式会把标量掩码压成 0 维，这里补一个维度，保证与目标张量的时间维对齐。
        if pad.ndim == 0:
            pad = pad.unsqueeze(0)
        # 转成 bool，方便后续用 `|`（按位或）合并多个 meta 的掩码。
        return pad.bool()

    def _get_episode_state_sequence(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Read one full-episode state sequence for stats / episode utilities.

        Args:
            meta: One state meta.
            lerobot_sample: Episode-level sample returned by
                `MultiLeRobotDataset.get_episode_data()`.

        Returns:
            Tensor with shape `(T, D)` for the requested state key.
        """
        # 【样本级 vs episode 级】本方法处理的是“整条 episode 一次读进来”的数据
        # （get_episode_data 的返回），因此不需要挑帧，直接切片 + 按需时间平移。
        # 与 _get_state 的区别：那个返回 [T_obs, D] 的观测窗口，这个返回 [T, D] 的全轨迹。
        # 用途：get_dataset_stats() 统计整条轨迹的分布、子类算初始化位置等。
        state = self._slice_meta_feature(self._get_meta_source_data(lerobot_sample, meta), meta)
        # 单维 part（如夹爪）补特征维，统一成 (T, D)。
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        # time_offset != 0 时整条序列平移（clamp 边界，不跨 episode）。
        if meta["time_offset"] != 0:
            state = shift_sequence_with_replication(state, meta["time_offset"])
        # 兜底断言：切出来的宽度必须与配置声明一致，否则说明 shape_meta 与实际数据不符。
        assert state.shape[-1] == meta["raw_shape"], (
            f"State '{meta['key']}' shape {state.shape[-1]} mismatch with meta {meta['raw_shape']}."
        )
        return state

    def _get_episode_action_sequence(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Build one full-episode raw action target sequence.

        Args:
            meta: One action meta.
            lerobot_sample: Episode-level sample returned by
                `MultiLeRobotDataset.get_episode_data()`.

        Returns:
            Tensor with shape `(T, D)` representing the raw supervision target
            for this action key before sliding-window expansion.
        """
        # 与 _get_episode_state_sequence 对称：返回整条 episode 的原始 action 序列 (T, D)。
        # 注意这里还只是“逐帧的单步动作”，真正训练用的 [T, H, D] 动作块
        # 是在 _get_episode_data 里用 sliding_window_with_replication 展开的。
        action = self._slice_meta_feature(self._get_meta_source_data(lerobot_sample, meta), meta)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if meta["time_offset"] != 0:
            action = shift_sequence_with_replication(action, meta["time_offset"])
        assert action.shape[-1] == meta["raw_shape"], (
            f"Action '{meta['key']}' shape {action.shape[-1]} mismatch with meta {meta['raw_shape']}."
        )
        return action

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one action key from a sample-level lerobot query result.

        Args:
            meta: One action meta after initialization has filled
                `query_positions`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.
                For each queried raw lerobot key, the typical tensor shape is
                `[num_requested_offsets, raw_shape]` (or `[num_requested_offsets]`
                for scalar keys before unsqueeze).

        Returns:
            Tensor with shape `[action_size, raw_shape]` for the current action
            key.
        """
        # 【本方法的定位】_get_sample_action_tensor 负责“挑帧 + 切维度”，
        # 这里再补上两个收尾动作：单维 part 补特征维、断言宽度与配置一致。
        # 之所以要断言：shape_meta 写错（例如 raw_shape 写 7 而实际列只有 6 维）时，
        # 后面的归一化/合并会以“形状对不上”的模糊报错炸掉，不如在这里直接指出是哪个 part。
        key, raw_shape = meta["key"], meta["raw_shape"]
        action: torch.Tensor = self._get_sample_action_tensor(meta, lerobot_sample)
        if action.ndim == 1:  # for shape of 1, like gripper
            # 单维 part（如夹爪）在底层读出来是 [H]，统一补成 [H, 1]，
            # 使下游所有 part 都是“二维 (时间, 特征)”的一致结构。
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, (
            f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        )
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one state key from a sample-level lerobot query result.

        Args:
            meta: One state meta after initialization has filled
                `query_positions`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.
                For each queried raw lerobot key, the typical tensor shape is
                `[num_requested_offsets, raw_shape]` (or `[num_requested_offsets]`
                for scalar keys before unsqueeze).

        Returns:
            Tensor with shape `[obs_size, raw_shape]` for the current state key.
            In the current dataset assumptions `obs_size == 1`, so the typical
            result is `[1, raw_shape]`.
        """
        # 与 _get_action 完全对称：挑帧（obs_size 帧）→ 补维 → 断言。
        # 多帧（MEM）时返回 [obs_size, raw_shape]，行顺序是“由旧到新”，最后一行为当前帧。
        key, raw_shape = meta["key"], meta["raw_shape"]
        state: torch.Tensor = self._get_sample_state_tensor(meta, lerobot_sample)
        if state.ndim == 1:  # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        assert state.shape[-1] == raw_shape, (
            f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        )
        return state

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        """取一个相机 key 的图像窗口，统一成 uint8 的 [T_obs, 3, H, W]。"""
        lerobot_key, raw_shape = meta["lerobot_key"], meta["raw_shape"]
        if lerobot_key == "__dummy__":
            # 历史遗留写法：YAML 里把不存在的相机写成 lerobot_key: "__dummy__"，
            # 这里直接返回全 0 图（同 _inject_dummy_images，只是入口不同）。
            C, H, W = raw_shape[0], raw_shape[1], raw_shape[2]
            return torch.zeros(1, C, H, W, dtype=torch.uint8)
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3:  # time dim will lost when obs_size is 1
            # obs_size == 1 时底层把时间维压掉了，只返回 [3, H, W]；
            # 这里补回时间维，让单帧与多帧的返回结构完全一致（[1, 3, H, W]）。
            image = image.unsqueeze(0)
        # 底层解码出来是 [0,1] 的 float32，训练侧约定用 uint8（0-255）表示像素，
        # 乘 255 再截断为整数即可；此处传下去的张量通道顺序是 CHW。
        image = (image * 255).to(torch.uint8)  # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        # 故意不做形状断言：raw_shape 通常写的是“原始采集分辨率”（如 [3,720,1280]），
        # 而实际的缩放/裁剪交给 processor 里的 image transform（shape 字段才是变换后的目标形状）。
        return image

    def _inject_dummy_images(self, sample: dict) -> dict:
        """Inject zero tensors for image entries marked ``dummy: true`` in shape_meta."""
        # 【为什么要有 dummy 相机】不同数据集/embodiment 的相机数量不一致，
        # 但模型输入需要固定的相机槽位；于是把缺失的相机在 YAML 里标 dummy: true，
        # 取数时补一张全 0 黑图占位，保证 batch 内所有样本的图像 key 完全一致
        # （否则 DataLoader 的默认 collate 拼不出来）。
        if not self._dummy_image_meta:
            return sample
        images = sample.setdefault("images", {})
        # 时间维 T 从“已有的第一张真实图像”推断，保证 dummy 图与真实图帧数一致；
        # 若一张真实图都没有（纯 dummy 配置），退化用 obs_size。
        T = None
        for v in images.values():
            if hasattr(v, "shape") and v.ndim >= 1:
                T = int(v.shape[0])
                break
        if T is None:
            T = self.obs_size
        for meta in self._dummy_image_meta:
            key = meta["key"]
            # 已经由 _get_image 填过的 key 不覆盖（避免把真实图像盖成黑图）。
            if key in images:
                continue
            # dummy meta 只需要 key/camera_type/shape；不带 lerobot_key、也不做切片校验。
            shape = meta.get("shape") or meta.get("raw_shape")
            if shape is None or len(shape) != 3:
                raise ValueError(f"dummy image meta {key!r} requires shape=[C,H,W], got {shape!r}")
            C, H, W = shape
            images[key] = torch.zeros(T, C, H, W, dtype=torch.uint8)
        sample["images"] = images
        return sample

    def _get_episode_data(self, episode_idx):
        """读整条 episode，并按“训练用的时间结构”整理好（统计量与子类工具都用它）。

        与 __getitem__ 的区别：__getitem__ 只取“一个样本窗口”（state 的 obs_size 帧 +
        action 的 action_size 步），而本方法取整条轨迹，供遍历式统计使用。

        返回（键名与 processor.action_state_transform 的入参约定一致）：
            {"state":  {key: Tensor[T, 1, D]},   # 中间那维 1 = 单步观测（与 batch 维对齐）
             "action": {key: Tensor[T, H, D]}}   # 每条帧位置都展开成未来 H 步的动作块
        """
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_episode_state_sequence(meta, lerobot_sample)
            # unsqueeze(1)：把 (T, D) 变成 (T, 1, D)，即“T 个样本、每个 1 步观测”，
            # 这样后面 processor 里按 (B, T_obs, D) 统一处理的代码无需分支。
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_episode_action_sequence(meta, lerobot_sample)
            # 把逐帧单步动作 (T, D) 展开成动作块 (T, H, D)：
            # 第 t 行的 H 步 = 原始序列的 [t, t+1, ..., t+H-1]，越界处复制末帧。
            # 这样每个时间点都能直接作为“以该帧为起点的 action chunk”监督信号。
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        """运行期开关图像：同时同步“是否返回 / 是否解码 / 是否处于训练态”。

        典型用法见 GalaxeaLerobotDataset.get_init_positions()：只为读若干帧的数值状态，
        不需要图像，就把图像关掉以省掉视频解码开销。
        """
        self.return_images = flag
        # 即使调用方要求 True，若配置里根本没有图像 meta 也保持 False。
        self.load_images = bool(flag) and bool(self.image_meta)
        self.multi_dataset.set_load_images(self.load_images)
        # during_training 也一起置位：底层据此决定是否走视频解码分支
        # （v2.1 的 LeRobotDataset 只在 during_training and load_images 时才解码视频）。
        self.multi_dataset.set_during_training(flag)

    def enable_overfit(self, n_samples: int):
        """Pin overfit mode to a stable subset of qualified samples."""
        # 【用途】小样本过拟合自检：把数据集“缩小”到固定的 n 个样本，
        # 让模型反复看同样几条数据 —— 若 loss 下不去，说明模型/数据管线本身有问题。
        #
        # 注意“stable（稳定）”二字：索引集合在这里一次性选好并保存（_overfit_indices），
        # 之后每次取数都取同一批样本，不会因为随机重采样而漂移。
        #
        # 由 scripts/finetune.py 的 --test/overfit 开关调用，见 MixtureLerobotDataset.enable_overfit()。
        target_samples = min(int(n_samples), self._end_idx - self._start_idx)
        self._overfit_indices = self._collect_overfit_indices(target_samples)
        self._overfit_len = len(self._overfit_indices)

    def _collect_overfit_indices(self, n_samples: int) -> List[int]:
        """挑选 n 个“可用且稳定”的样本下标（全局下标）。"""
        if n_samples <= 0:
            return []

        if not self.is_training_set:
            # 验证集不做“合格性”筛查，按顺序取前 n 个即可。
            return list(range(self._start_idx, self._start_idx + n_samples))

        qualified_indices = []
        skipped_unqualified = 0

        for sample_idx in range(self._start_idx, self._end_idx):
            try:
                # 真实读一次样本：只有能读出来、且被标注为合格的样本才会入选。
                # 这里的读取代价是一次全量扫描，但只在开启 overfit 模式时发生。
                lerobot_sample = self.multi_dataset[sample_idx]
            except Exception as err:
                # 读不出来的样本（视频缺失、时间戳错乱等）直接跳过，不中断整个挑选过程。
                location = self._locate_sample(sample_idx)
                logger.warning(
                    f"Skipping corrupted overfit candidate {sample_idx} ({location}). Error: {err}"
                )
                continue

            if lerobot_sample.get("step_is_qualified", True):
                # 合格样本：入选。默认 True 表示“该数据集没有质量标注”，一律视为合格。
                qualified_indices.append(sample_idx)
                if len(qualified_indices) >= n_samples:
                    break
            else:
                skipped_unqualified += 1

        if not qualified_indices:
            # 一个都没挑出来说明数据范围选错了（例如给定目录里全是未标注合格的帧）。
            raise RuntimeError(
                "Overfit mode could not find any qualified samples in the selected dataset range."
            )

        if len(qualified_indices) < n_samples:
            # 只是警告而非报错：合格样本不够时用现有的，仍然能起到过拟合自检的作用。
            logger.warning(
                "Overfit mode requested %s samples but only found %s qualified samples.",
                n_samples,
                len(qualified_indices),
            )

        if skipped_unqualified > 0:
            # 记录跳过了多少，便于判断“训练一直重采样”的根因是否在数据质量标注上。
            logger.info(
                "Overfit mode skipped %s unqualified samples before selecting %s stable samples.",
                skipped_unqualified,
                len(qualified_indices),
            )

        return qualified_indices

    def __len__(self):
        # overfit 模式下长度就是被挑中的样本数；否则是“本实例负责的帧区间长度”。
        # DataLoader 据此决定每个 epoch 迭代多少个样本。
        if hasattr(self, "_overfit_len"):
            return self._overfit_len
        return self._end_idx - self._start_idx

    def _locate_sample(self, sample_idx: int) -> str:
        """Map a global sample index back to its dataset directory and local index (O(log N) bisect)."""
        # 【为什么需要】报错信息里若只有“全局下标 12345”，使用者几乎无法在多个数据集目录中定位问题。
        # 本方法把它翻译成 “dataset_dir=/path/to/ds, local_idx=42” 这种可直接去 parquet 里查的形式。
        #
        # bisect_right(cumulative, idx) 返回“第一个大于 idx 的位置”，正好是目标目录的下标 i
        # （例如 cumulative=[100,150,230]，idx=120 → i=1，即第二个目录）。
        i = bisect.bisect_right(self._ds_cumulative_frames, sample_idx)
        if i < len(self._ds_cumulative_frames):
            ds = self.multi_dataset._datasets[i]
            # 本地下标 = 全局下标 − 前面所有目录的帧数之和。
            local_idx = sample_idx - (self._ds_cumulative_frames[i - 1] if i > 0 else 0)
            # 目录路径优先用 root，取不到再退回 repo_id（不同版本属性名不同）。
            ds_dir = getattr(ds, "root", getattr(ds, "repo_id", f"dataset[{i}]"))
            return f"dataset_dir={ds_dir}, local_idx={local_idx}"
        return f"sample_idx={sample_idx} (out of bounds)"

    def _get_additional_data(self, sample, lerobot_sample):
        # 【子类扩展点】默认原样返回。子类可以在这里把额外字段挂进 sample，
        # 例如 GalaxeaLerobotDataset 会附上 coarse_task（高层任务描述）供 CoT 模板使用。
        return sample

    def _record_invalid_sample(
        self,
        *,
        stage: str,
        requested_idx: int,
        sample_idx: int,
        error: Optional[Exception] = None,
    ) -> None:
        """登记一条“坏样本”记录（跨进程可见，用于训练结束后的汇总报告）。

        stage 取值（对应 __getitem__ 里的三个阶段）：
            "unqualified"  样本被质量标注判为不合格（step_is_qualified=False）
            "load_error"   底层读样本抛异常
            "build_error"  读到了但组装/预处理阶段抛异常
        """
        # 同一个 (阶段, 样本, 位置) 会被反复命中，因此用复合 key 聚合并计数，
        # 而不是每条错误都追加一条（否则坏数据会瞬间刷爆日志与内存）。
        location = self._locate_sample(sample_idx)
        key = f"{stage}|{sample_idx}|{location}"
        record = dict(self._invalid_sample_records.get(key, {}))
        if not record:
            # 首次出现：建记录并记下顺序（order 列表保证报告的输出顺序稳定可控）。
            record = {
                "stage": stage,
                "requested_idx": int(requested_idx),
                "sample_idx": int(sample_idx),
                "location": location,
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
                "count": 0,
            }
            self._invalid_sample_order.append(key)
        record["count"] = int(record.get("count", 0)) + 1
        self._invalid_sample_records[key] = record

    def get_invalid_sample_report(self, reset: bool = False) -> List[Dict[str, Any]]:
        # 按登记顺序导出全部记录（MixtureLerobotDataset 会把各 embodiment 的报告汇总打印）。
        # reset=True 时顺便清空，便于“训练前清一次、训练后再看”的用法。
        report = [
            dict(self._invalid_sample_records[key]) for key in list(self._invalid_sample_order)
        ]
        if reset:
            self.clear_invalid_sample_report()
        return report

    def clear_invalid_sample_report(self) -> None:
        self._invalid_sample_records.clear()
        # Manager.list 不支持 clear()（没有该接口），必须用切片赋空来就地清空。
        self._invalid_sample_order[:] = []

    def _resample_random_idx(self) -> int:
        """Random absolute sample_idx for retry fallback.

        Subclasses with index filtering (e.g. DroidLerobotDataset's
        _valid_local_indices) override to ensure retry stays within valid frames.
        """
        # 遇到坏样本时“换一个随机样本顶上”，保证 batch 形状/数量稳定（少一个样本会拖慢训练）。
        # 只在 [self._start_idx, self._end_idx) 内取值，因此不会跨出本实例负责的帧区间。
        # 子类可重写以进一步限制范围（例如 DROID 里有有效帧白名单）。
        return np.random.randint(self._start_idx, self._end_idx)

    def __getitem__(self, idx):
        """取一个训练样本（DataLoader 的核心入口，性能与稳定性都关键）。

        整体分两个阶段、共用同一个 attempt 预算（MAX_GETITEM_ATTEMPT）：
          阶段 A（load）  : 从底层读出一个可用的 lerobot_sample，必要时换随机下标重试；
          阶段 B（build） : 把原始样本组装/切分/加掩码/预处理，同样支持失败重试。

        为什么要有重试机制：真实数据里总有一部分帧是坏的（视频缺帧、时间戳越界、
        被质量标注判为不合格……）。与其让训练崩在某个坏样本上，不如跳过它换一个样本，
        同时把问题登记到 invalid 报告里，训练结束后统一排查。
        """
        # overfit 模式会临时把数据集“变小”，长度也随之变化，这里按 len() 校验下标。
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        # requested_idx 保存“调用方原本想要的下标”，出错日志里同时给出
        # “请求下标”和“实际使用的下标”，方便定位是哪个位置的数据有问题。
        requested_idx = idx
        overfit_active = hasattr(self, "_overfit_indices")
        if overfit_active:
            # overfit：idx 是“稳定样本列表”的下标，取其对应的全局下标。
            sample_idx = int(self._overfit_indices[idx])
        else:
            # 常规：把实例内的局部下标平移成底层多数据集空间里的全局下标。
            sample_idx = idx + self._start_idx
        attempt = 0
        last_exception: Optional[Exception] = None
        # ---------------- 阶段 A：读出可用的 lerobot_sample ----------------
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                if self.is_training_set and not lerobot_sample.get("step_is_qualified", True):
                    # 质量标注不合格：训练集里直接换样本（验证/评估集保留，以免验证集被掏空）。
                    # 默认 get(..., True) 表示“没有该字段 = 视为合格”，兼容未打标的旧数据集。
                    if overfit_active:
                        # overfit 模式下样本是预先筛好的，出现不合格说明数据集本身在变
                        # （例如标注文件被改写），这时必须报错而不是悄悄换样本，
                        # 否则“固定样本”的语义被破坏，过拟合实验的结论就不可信了。
                        location = self._locate_sample(sample_idx)
                        raise RuntimeError(
                            "Overfit sample became unqualified after selection: "
                            f"index={sample_idx} ({location})"
                        )
                    self._record_invalid_sample(
                        stage="unqualified",
                        requested_idx=requested_idx,
                        sample_idx=sample_idx,
                    )
                    attempt += 1
                    sample_idx = self._resample_random_idx()
                    continue
                break
            except Exception as err:
                if overfit_active:
                    # 同上：overfit 只允许“指定的那批样本”，读不出来就应立刻暴露问题。
                    location = self._locate_sample(sample_idx)
                    raise RuntimeError(
                        f"Failed to load preselected overfit sample {sample_idx} ({location})."
                    ) from err
                attempt += 1
                last_exception = err
                self._record_invalid_sample(
                    stage="load_error",
                    requested_idx=requested_idx,
                    sample_idx=sample_idx,
                    error=err,
                )
                location = self._locate_sample(sample_idx)
                # 每次换样本都打一条 warning：训练日志里能看到错误率，便于判断数据质量。
                logger.warning(
                    f"Error loading sample {sample_idx} ({location}) "
                    f"(attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                sample_idx = self._resample_random_idx()
        else:
            # while-else：循环因 attempt 触顶而正常结束（不是 break 出来的）才会执行。
            # 触顶说明坏样本比例极高（或某类错误必然发生），必须让训练停下来。
            last_location = self._locate_sample(sample_idx)
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {requested_idx}. Last sample: {sample_idx} ({last_location})."
            ) from last_exception

        # ---------------- 阶段 B：组装样本 ----------------
        # 注意 attempt 是复用的：阶段 A 用掉的尝试次数会占用阶段 B 的预算，
        # 因此两种重试合计不超过 MAX_GETITEM_ATTEMPT 次（避免无限重试打满 CPU）。
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                # Get data from lerobot, organized in nested dict
                # action:
                #   left_arm: torch.Tensor
                #   right_arm: torch.Tensor
                # state:
                #   left_arm: torch.Tensor
                #   right_arm: torch.Tensor
                # images:
                #   head_rgb: torch.Tensor
                sample = {
                    # idx：本样本的“全局下标”，调试时用它反查 parquet 行（与 dataset_locator 配合）。
                    "idx": sample_idx,
                    # task：语言指令（底层已把 task_index 解码成文本），CoT 模板的 command 来源。
                    "task": lerobot_sample["task"],
                    # dataset_locator：人类可读的定位串，出错/可视化时不用再自己算。
                    "dataset_locator": self._locate_sample(sample_idx),
                    "action": {},
                    "state": {},
                    "images": {},
                    # frequency：模型控制频率，processor 会透传到 samples["frequency"]，
                    # 最终交给 action tokenizer 做“每步多少秒”的时间编码/解码换算。
                    "frequency": self.model_fps,  # Model control frequency (override_fps), used by action tokenizer encode/decode.
                }
                # 逐 part 取 state/action，并把各 part 的越界掩码按“或”合并成一份全局掩码：
                # 只要某个 part 在这一帧越界，整帧就视为无效（因为一个样本被当作一个整体使用）。
                state_is_pad = None
                for meta in self.state_meta:
                    sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
                    cur_state_is_pad = self._get_pad_mask(meta, lerobot_sample)
                    state_is_pad = (
                        # 第一个 part 直接作为初值……
                        cur_state_is_pad
                        if state_is_pad is None
                        # ……后续 part 与它按位或：任一为 True 则结果 True。
                        else (state_is_pad | cur_state_is_pad)
                    )

                action_is_pad = None
                for meta in self.action_meta:
                    sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)
                    cur_action_is_pad = self._get_pad_mask(meta, lerobot_sample)
                    action_is_pad = (
                        cur_action_is_pad
                        if action_is_pad is None
                        else (action_is_pad | cur_action_is_pad)
                    )

                if self.image_meta:
                    # 图像：逐相机取窗口（[T_obs, 3, H, W] uint8）。
                    for meta in self.image_meta:
                        sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
                    # 图像掩码直接复用第一个相机 key 的掩码：所有相机共用同一时间轴，
                    # 因此越界位置完全一致，不需要逐相机取再合并。
                    sample["image_is_pad"] = lerobot_sample[
                        f"{self.image_meta[0]['lerobot_key']}_is_pad"
                    ]
                # dummy 相机（YAML 里 dummy: true）在这里补全 0 张量。
                self._inject_dummy_images(sample)

                sample["action_is_pad"] = action_is_pad
                sample["state_is_pad"] = state_is_pad

                # ---------- 子任务（subtask）边界打掩码 ----------
                # chunked_task_index 是“这 H 步各自属于哪个子任务”的下标序列（由 task_index 查询得到）。
                # 动作块如果跨越了子任务切换点，后面的动作就不再是“当前指令”下的动作，
                # 直接当监督目标会污染训练，所以：
                #   1) 把这些步在 action_is_pad 里标 True（不计 loss）；
                #   2) 同时把它们的动作值替换成切换前最后一个有效动作（保持张量内容自洽，
                #      避免下游若绕过掩码直接取用时读到跨子任务的错误值）。
                if (
                    "chunked_task_index" in lerobot_sample
                    and lerobot_sample["chunked_task_index"] is not None
                ):
                    chunked_task_index = lerobot_sample["chunked_task_index"]
                    mask = (
                        chunked_task_index != chunked_task_index[0]
                    )  # (T,), True for steps with different subtask
                    # 形状必须与 action_is_pad 一致（都是 [H]），否则说明 delta_timestamps
                    # 里 task_index 与 action 注册的偏移数量不一致。
                    assert mask.shape == sample["action_is_pad"].shape, (
                        f"Mask shape {mask.shape} does not match action_is_pad shape {sample['action_is_pad'].shape}"
                    )
                    sample["action_is_pad"] = sample["action_is_pad"] | mask
                    if mask.any():
                        # 最后一个“仍与当前帧同子任务”的步（即切换点前一步）。
                        last_valid = (~mask).nonzero(as_tuple=False)[-1].item()
                        for meta in self.action_meta:
                            # 用 clone() 避免写入时污染原张量的共享内存视图。
                            sample["action"][meta["key"]][mask] = sample["action"][meta["key"]][
                                last_valid
                            ].clone()

                # 子类扩展点（例如附加 coarse_task 等额外字段）。
                sample = self._get_additional_data(sample, lerobot_sample)

                # 把底层样本里“尚未被消费”的字段原样透传给下游（task_index / coarse_task /
                # atomic_task / bbox_index / high_level_instruction ... 由 SamplesBuilder 使用）。
                # 过滤规则：
                #   · 已经在本 sample 里的 key 不覆盖（本类整理过的版本更权威）；
                #   · 名字里含 "observation"/"action" 的原始列不再透传：
                #     这些大数据张量已经被切成 part 放进 sample["state"]/["action"]，
                #     再留一份完整副本既占内存，又容易让下游误用未切分的原始形状。
                for key in lerobot_sample:
                    if key not in sample and "observation" not in key and "action" not in key:
                        sample[key] = lerobot_sample[key]

                # Preprocess the sample using the processor
                # for quick data loading
                # 归一化 + padding 合并 + 构造 samples 模板都在 processor 里完成；
                # 在 __getitem__ 阶段做（而不是 collate 阶段）是为了让多 worker 并行分摊这部分开销。
                # 未设置 processor 时（例如只做数据检查、评测脚本）直接返回原始样本。
                if self.processor is not None:
                    sample = self.processor.preprocess(sample)

                return sample

            except Exception as err:
                # overfit 模式不允许“换样本”，否则就失去了“固定样本反复看”的意义。
                if overfit_active:
                    location = self._locate_sample(sample_idx)
                    raise RuntimeError(
                        f"Failed to build preselected overfit sample {sample_idx} ({location})."
                    ) from err

                attempt += 1
                last_exception = err
                self._record_invalid_sample(
                    stage="build_error",
                    requested_idx=requested_idx,
                    sample_idx=sample_idx,
                    error=err,
                )
                location = self._locate_sample(sample_idx)
                logger.warning(
                    f"Error building sample {sample_idx} ({location}) "
                    f"(attempt {attempt}). Retrying with a random index. Error: {err}"
                )
                # 换随机下标重来。注意：这里的多数据集读取没有包在 try 里，
                # 若读新样本也失败，异常会直接向上抛出（由上层/DataLoader 报错）。
                sample_idx = self._resample_random_idx()
                lerobot_sample = self.multi_dataset[sample_idx]

        # 阶段 B 同样用 while-else 语义：只有尝试次数耗尽还没成功才会走到这里。
        last_location = self._locate_sample(sample_idx)
        raise RuntimeError(
            f"Failed to build a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
            f"for index {requested_idx}. Last sample: {sample_idx} ({last_location})."
        ) from last_exception

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        # 注入处理器（MixtureLerobotDataset.set_processor 会为每个 embodiment 分别调用）。
        # 这里顺带切换 train/eval 模式：训练模式下 processor 会启用图像增强等随机变换，
        # 评估模式下关闭随机性，保证多次推理结果一致。
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        # 返回 self 支持链式写法：ds.set_processor(p).enable_overfit(8)
        return self

    def get_dataset_stats(
        self,
        preprocessor: BaseProcessor,
        only_keys: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Compute dataset statistics for normalization.

        Args:
            preprocessor: Processor to transform action/state data
            only_keys: If provided, only compute stats for specified keys.
                       Format: {"action": {"left_arm", "right_arm"}, "state": {"torso"}}
                       If None, compute stats for all keys.

        Returns:
            Dict with structure: {"action": {key: {...stats}}, "state": {key: {...stats}}}
        """
        # 【为什么要算统计量】训练要先把 action/state 归一化到同一量纲
        # （z-score / min-max / quantile），否则不同 part 的数值尺度差异会让优化极不稳定。
        # 本方法遍历整个数据集，算出每个 key 的 min/max/mean/std 与多组分位数，
        # 最终由 scripts/finetune.py 落盘成 dataset_stats.json 供训练与部署共用。
        #
        # 【两个粒度】下游归一化有两种模式（见 configs/data/r1pro.yaml 的
        # use_stepwise_action_norm 说明），所以这里同时产出两套：
        #   stepwise_*：动作块内“第 t 步”各自的统计量（先按 episode 聚合，再按步对齐）
        #   global_*  ：整个动作块共用一套统计量
        # 【多组分位数】q01/q99 用于稳健归一化（裁掉离群点）；更极端的 q0.001~q0.99999
        # 用于需要更宽覆盖范围时的兜底（例如相对动作变换的裁剪区间）。
        # 每个统计量都用 DefaultDict(list) 收集：key = part 名，list 里每个 episode 一个张量。
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)
        state_q001 = DefaultDict(list)
        state_q999 = DefaultDict(list)
        state_q0001 = DefaultDict(list)
        state_q9999 = DefaultDict(list)
        state_q00001 = DefaultDict(list)
        state_q99999 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)
        action_q001 = DefaultDict(list)
        action_q999 = DefaultDict(list)
        action_q0001 = DefaultDict(list)
        action_q9999 = DefaultDict(list)
        action_q00001 = DefaultDict(list)
        action_q99999 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes

        # 需要统计的 key 集合：None 表示“全部 part”；否则只算调用方指定的那几个
        # （MixtureLerobotDataset 会按 embodiment 传入 only_keys，避免算用不到的 part）。
        state_keys_to_compute = (
            set(m["key"] for m in self.state_meta)
            if only_keys is None
            else only_keys.get("state", set())
        )
        action_keys_to_compute = (
            set(m["key"] for m in self.action_meta)
            if only_keys is None
            else only_keys.get("action", set())
        )

        def process_episode(episode_idx):
            # 读整条 episode → 过一遍 processor 的 action_state_transform。
            # 这里必须先做变换再统计：统计量要描述“模型实际看到的数值分布”，
            # 例如相对动作变换（RelativeJointTransform）会改变数值范围，统计原始值就错位了。
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        # 线程池并行遍历 episode：主要瓶颈是磁盘/parquet 读取与解码，
        # 属于 I/O 密集，用多线程即可（GIL 在 I/O 期间会释放）。
        with ThreadPoolExecutor() as executor:
            # 一次性提交所有 episode 的任务；as_completed 保证“谁先完成先统计”，
            # 配合 tqdm 显示进度（统计大集群数据时这一步可能几分钟到几十分钟）。
            futures = [executor.submit(process_episode, num) for num in range(episodes_num)]

            for future in tqdm(
                as_completed(futures),
                total=episodes_num,
                desc="Iterating dataset to get normalization",
            ):
                try:
                    batch = future.result()
                    for meta in self.state_meta:
                        key = meta["key"]
                        if key not in state_keys_to_compute:
                            continue
                        # 形状说明：(B, T, dim) 里的 B 就是这条 episode 的帧数 T_episode，
                        # T=1（_get_episode_data 里 unsqueeze(1) 的“单步观测”），dim = part 维度。
                        # 所有统计都沿第 0 维（时间）归约，得到 (1, dim)：即“该 episode 内
                        # 每个 part 维度的整体分布”，稍后再跨 episode 聚合。
                        cur_state: torch.Tensor = batch["state"][key]  # (B, T, dim)
                        state_min[key].append(cur_state.amin(0))
                        state_max[key].append(cur_state.amax(0))
                        state_mean[key].append(cur_state.mean(0))
                        state_var[key].append(cur_state.var(0))
                        # 分位数：quantile 无法像 mean 那样“先算局部再合并”，
                        # 因此这里是逐 episode 算分位数，最后用 amin/amax 跨 episode 合并
                        # （即取“最保守”的下界/上界），见后文 stats 组装部分。
                        state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                        state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                        state_q001[key].append(
                            torch.quantile(cur_state, 0.001, dim=0, keepdim=False)
                        )
                        state_q999[key].append(
                            torch.quantile(cur_state, 0.999, dim=0, keepdim=False)
                        )
                        state_q0001[key].append(
                            torch.quantile(cur_state, 0.0001, dim=0, keepdim=False)
                        )
                        state_q9999[key].append(
                            torch.quantile(cur_state, 0.9999, dim=0, keepdim=False)
                        )
                        state_q00001[key].append(
                            torch.quantile(cur_state, 0.00001, dim=0, keepdim=False)
                        )
                        state_q99999[key].append(
                            torch.quantile(cur_state, 0.99999, dim=0, keepdim=False)
                        )

                    for meta in self.action_meta:
                        key = meta["key"]
                        if key not in action_keys_to_compute:
                            continue
                        # action 的形状是 (T_episode, H, dim)：第 1 维是“动作块内的第 t 步”。
                        # amin/amax/mean/var 沿第 0 维（episode 时间）归约后得到 (H, dim)，
                        # 因此每个“未来第 t 步”都保留了自己的统计量 —— 这正是 stepwise 归一化所需的。
                        cur_action: torch.Tensor = batch["action"][key]  # (B, T, dim)
                        action_min[key].append(cur_action.amin(0))
                        action_max[key].append(cur_action.amax(0))
                        action_mean[key].append(cur_action.mean(0))
                        action_var[key].append(cur_action.var(0))
                        action_q01[key].append(
                            torch.quantile(cur_action, 0.01, dim=0, keepdim=False)
                        )
                        action_q99[key].append(
                            torch.quantile(cur_action, 0.99, dim=0, keepdim=False)
                        )
                        action_q001[key].append(
                            torch.quantile(cur_action, 0.001, dim=0, keepdim=False)
                        )
                        action_q999[key].append(
                            torch.quantile(cur_action, 0.999, dim=0, keepdim=False)
                        )
                        action_q0001[key].append(
                            torch.quantile(cur_action, 0.0001, dim=0, keepdim=False)
                        )
                        action_q9999[key].append(
                            torch.quantile(cur_action, 0.9999, dim=0, keepdim=False)
                        )
                        action_q00001[key].append(
                            torch.quantile(cur_action, 0.00001, dim=0, keepdim=False)
                        )
                        action_q99999[key].append(
                            torch.quantile(cur_action, 0.99999, dim=0, keepdim=False)
                        )

                except Exception as e:
                    # 单条 episode 出错不影响整体统计：记 error 后继续处理其他 episode。
                    # （坏 episode 的分布信息缺失，但比整个训练因为一条脏数据而中断要好。）
                    logger.error(f"Error processing episode: {e}")

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            """合并“每 episode 的均值/方差”为 stepwise 与 global 两套 (mean, std)。

            用到总方差分解（law of total variance）：
                E[Var] + Var[E]  =  组内平均方差 + 组间均值差异
            即“某个维度在整个数据集上的方差” = 各 episode 内方差的平均
                                            + 各 episode 均值相对总体均值的波动。
            只对 per-episode 的 mean/var 求平均是不够的（会漏掉第二项）。

            形状约定：means/vars 都是 [(E, 1, dim) 或 (E, H, dim)] 的列表，
            堆叠后为 (E, ...)，其中 E = episode 数。
              stepwise：沿 E 归约 → 保留动作步维，每个未来步一套统计量
              global  ：沿 (E, 步) 全部归约 → 整套动作块共用一套统计量
            """
            means = torch.stack(means)
            vars = torch.stack(vars)
            # 逐时间步的均值（state 的 T=1，因此结果形状与单帧一致）。
            stepwise_mean = means.mean(0)
            # 逐时间步的 std：组内方差 + 组间均值波动（见上面公式）。
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            # 全局均值/标准差：把所有 episode、所有动作步一起看。
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict)}
        # ---------- 跨 episode 聚合 state 统计量 ----------
        # 记法：每个 list 里有 E 个张量，元素形状为 (1, dim)。
        #   stepwise_* = stack 后沿 E 维归约 → (1, dim)，即“这一维在整个数据集上的取值范围”
        #   global_*   = 再把 stepwise 的结果沿剩余的时间维压平 → (dim,)
        # min 用 amin、max 用 amax：跨 episode 取“最保守”的边界，保证归一化区间覆盖全部数据。
        # 分位数按“保守合并”处理：q01 取各 episode 的最小值、q99 取最大值
        # （因为分位数本身不能再平均，这样合并能确保区间不窄于真实分布）。
        for meta in self.state_meta:
            key = meta["key"]
            if key not in state_keys_to_compute:
                continue
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            stats["state"][key]["stepwise_q001"] = torch.stack(state_q001[key]).amin(0)
            stats["state"][key]["stepwise_q999"] = torch.stack(state_q999[key]).amax(0)
            stats["state"][key]["global_q001"] = stats["state"][key]["stepwise_q001"].amin(0)
            stats["state"][key]["global_q999"] = stats["state"][key]["stepwise_q999"].amax(0)
            stats["state"][key]["stepwise_q0001"] = torch.stack(state_q0001[key]).amin(0)
            stats["state"][key]["stepwise_q9999"] = torch.stack(state_q9999[key]).amax(0)
            stats["state"][key]["global_q0001"] = stats["state"][key]["stepwise_q0001"].amin(0)
            stats["state"][key]["global_q9999"] = stats["state"][key]["stepwise_q9999"].amax(0)
            stats["state"][key]["stepwise_q00001"] = torch.stack(state_q00001[key]).amin(0)
            stats["state"][key]["stepwise_q99999"] = torch.stack(state_q99999[key]).amax(0)
            stats["state"][key]["global_q00001"] = stats["state"][key]["stepwise_q00001"].amin(0)
            stats["state"][key]["global_q99999"] = stats["state"][key]["stepwise_q99999"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        # ---------- 跨 episode 聚合 action 统计量 ----------
        # 与 state 的差别只有一个：这里的元素形状是 (H, dim)（动作块），
        # 因此 stepwise_* 天然带“未来第几步”这一维，供 use_stepwise_action_norm=True 使用。
        for meta in self.action_meta:
            key = meta["key"]
            if key not in action_keys_to_compute:
                continue
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            stats["action"][key]["stepwise_q001"] = torch.stack(action_q001[key]).amin(0)
            stats["action"][key]["stepwise_q999"] = torch.stack(action_q999[key]).amax(0)
            stats["action"][key]["global_q001"] = stats["action"][key]["stepwise_q001"].amin(0)
            stats["action"][key]["global_q999"] = stats["action"][key]["stepwise_q999"].amax(0)
            stats["action"][key]["stepwise_q0001"] = torch.stack(action_q0001[key]).amin(0)
            stats["action"][key]["stepwise_q9999"] = torch.stack(action_q9999[key]).amax(0)
            stats["action"][key]["global_q0001"] = stats["action"][key]["stepwise_q0001"].amin(0)
            stats["action"][key]["global_q9999"] = stats["action"][key]["stepwise_q9999"].amax(0)
            stats["action"][key]["stepwise_q00001"] = torch.stack(action_q00001[key]).amin(0)
            stats["action"][key]["stepwise_q99999"] = torch.stack(action_q99999[key]).amax(0)
            stats["action"][key]["global_q00001"] = stats["action"][key]["stepwise_q00001"].amin(0)
            stats["action"][key]["global_q99999"] = stats["action"][key]["stepwise_q99999"].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].

    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)

    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window

    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    # 【用途】把逐帧序列展开成“动作块”：第 i 行 = 从第 i 帧开始的 window_size 步。
    # 用于 get_dataset_stats / _get_episode_data：让每个时间点都能作为一个
    # “以该帧为起点、未来 H 步”的训练样本，从而按 episode 一次性统计/检查数据。
    #
    # 【例】N=4, window_size=3, D=1，输入 [a,b,c,d]：
    #   out[0] = [a, b, c]
    #   out[1] = [b, c, d]
    #   out[2] = [c, d, d]   ← 越界处复制末行（d）
    #   out[3] = [d, d, d]
    # 越界用“复制末帧”而不是补零，与数据集里 *_is_pad 的语义保持一致：
    # 越界位置只是被标记为无效（不计 loss），而不是引入 0 这种会污染统计的假值。
    assert x.dim() == 2
    assert window_size > 0

    N, D = x.shape

    # 用广播一次性构造“所有起点 × 所有窗口内偏移”的下标矩阵：
    #   i_indices[i, 0] = i        （起点）
    #   j_indices[0, j] = j        （窗口内第 j 步）
    #   indices[i, j]   = i + j    （该样本该步对应的原始帧下标）
    i_indices = torch.arange(N).unsqueeze(1)  # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices  # [N, window_size]

    # 越界处理：把 i + j >= N 的位置夹到最后一帧 N-1（等价于复制末行），
    # 从而输出长度与输入一致、且不会越界取到别的数据。
    # （下界 0 在当前实现里不会触发，写上是为了函数自身语义完整、也可被负偏移复用。）
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # 按行列取数：out[i, j, :] = x[clamped_indices[i, j], :]，结果形状 [N, window_size, D]
    # （clamped_indices 形状 [N, window_size]，x 形状 [N, D]）
    out = x[clamped_indices]  # [N, window_size, D]

    return out
