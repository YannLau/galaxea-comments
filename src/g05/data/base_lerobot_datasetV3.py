# =============================================================================
# src/g05/data/base_lerobot_datasetV3.py — 数据集基类 V3（快速统计量版）
# =============================================================================
#
# 【这个文件是什么】
#   它是 src/g05/data/base_lerobot_dataset.py 里 BaseLerobotDataset 的**子类**，
#   只重写“计算归一化统计量”这一件事：
#
#     父类（v2 做法）：for episode in 数据集: 读整条 episode → 过 processor 变换
#                      → 逐 episode 算 min/max/mean/var/8 组分位数 → 跨 episode 聚合
#     本类（v3 做法）：一次扫 parquet（只挑真正需要的那些帧）→ 拼成一个大张量
#                      → 用 np.partition / torch.sort 在 GPU 或 CPU 上一次算完
#
#   除统计量以外的全部训练语义——shape_meta 解析、delta_timestamps 查询计划、
#   __getitem__ 取数、is_pad 掩码、训练/验证集切分、坏样本重采样——都原样继承父类，
#   本文件一行都不改（这就是为什么本文件只有“统计”相关的函数）。
#
# 【为什么需要它：旧路径慢在哪】
#   统计量要描述“模型实际看到的数值分布”，所以必须把全量数据扫一遍。旧做法有两个
#   瓶颈，在千万帧级数据上会把几分钟拖成几十分钟：
#     1) 逐 episode 走 __getitem__ 式的取数：每条 episode 都要重新定位 parquet 行、
#        解码图像、构造 dict，python 层开销占了绝大部分时间；
#     2) 分位数用 torch.quantile / np.quantile：复杂度 O(N log N)，而且每个 key、
#        每个动作步都要单独排一次序。
#   本文件的两条优化正对应这两点：
#     · 按“帧下标集合”用 pyarrow 批量取行（_load_selected_rows_from_parquet），
#       一次顺序扫描就读完所有需要的帧，不碰图像、不构造 dict；
#     · 分位数改用 np.partition（O(N) 的“第 k 小”选择算法）或 GPU 上的 sort，
#       并把多个 key 拼成一条宽列一次算完（_merge_dict_to_tensor / _split_stats_by_key）。
#
# 【在数据链路中的位置】
#
#   configs/data/<task>.yaml                    （哪些数据集、哪些 part、什么形状）
#        │  (Hydra 实例化)
#        ▼
#   MixtureLerobotDataset                      src/g05/data/mixture_lerobot_dataset.py
#      │  · 按 embodiment 逐个 new 出下面这个类
#      │  · 统计阶段：ds.get_dataset_stats(processor) ← 调用本文件重写的方法
#      │  · 再把多个数据源的统计量按权重合并成“按 embodiment_type 一份”
#      ▼
#   BaseLerobotDatasetV3  ← 本文件
#      │  · 继承 BaseLerobotDataset 的全部取数逻辑
#      │  · 只把 get_dataset_stats 换成 _fast_get_dataset_stats（本文件的核心）
#      ▼
#   BaseLerobotDataset                         src/g05/data/base_lerobot_dataset.py
#      ▼
#   MultiLeRobotDataset / LeRobotDataset       src/g05/data/lerobot/lerobot_dataset{,_v3}.py
#      ▼
#   parquet（数值列） + mp4 / png（相机画面）
#
#   注意：本文件只读 parquet 里的**数值列**（action / observation.state 等），
#   图像完全不参与统计量计算，所以统计阶段比训练取数快得多。
#
# 【新手先记住的 6 个概念】
#
#   (1) “统计量”是什么、给谁用
#       训练前要把 action/state 归一化到同一量纲，否则各 part 的数值尺度差异会让优化
#       极不稳定。归一化需要每个 key 的 min/max/mean/std 与几组极端分位数，这些数就是
#       “统计量”：调用入口是 scripts/finetune.py（ds.get_dataset_stats(...)），
#       算完由它落盘成 run_dir/dataset_stats.json，训练 / 评测 / 部署共用同一份
#       （见 scripts/eval_open_loop.py 的读取逻辑）。本文件负责“算”，不负责“存”。
#       每个 key 同时产出两套粒度（下游按配置二选一，见 configs/data/r1pro.yaml 的
#       use_stepwise_action_norm）：
#         stepwise_*  动作块内“第 t 步”各自的统计量，形状 (H, dim)，H = action_size
#         global_*    整个动作块共用一套统计量，形状 (dim,)
#       state 没有“动作块”概念，所以它的 stepwise_* 退化成 (1, dim)。
#
#   (2) meta 的显式 raw 布局（与父类完全一致）
#       state_meta / action_meta 里每条记录声明“从哪个原始列的第几维开始、取多宽”：
#         key          内部 key（如 left_arm，训练侧用的名字）
#         lerobot_key  原始 parquet 列名（如 action.left_arm / observation.state）
#         start_index  从该列第几维开始切
#         raw_shape    切出来的宽度
#         time_offset  时间偏移（0 = 当前帧；1 = 下一帧，用于 state-as-action）
#       本文件用 _get_meta_source_data + _slice_meta_feature 复用父类同一套切分逻辑，
#       保证“统计看到的数值”与“训练看到的数值”定义完全一致。
#
#   (3) 抽样锚点 sampled_base_idx 与 stats_downsample_rate
#       统计不需要每一帧都参与：默认每 stats_downsample_rate 帧取一个“锚点”
#       （GalaxeaLerobotDataset 用 10，本类默认 1 = 全量）。
#       每个锚点还要顺带取它未来 H 步（以及带 time_offset 的 state 帧），
#       这些帧的并集才是真正要读的行 → required_indices。
#       降采样只影响“分布估计的精度”，不影响归一化公式本身；数据量大时能省下大量 I/O。
#
#   (4) 两套下标：required_indices（原始帧号）与 local 下标（读回来后的行号）
#       required_indices 是“整个数据集里的全局帧号”，排序去重后用它去 parquet 取行；
#       但取回来的张量只有 len(required_indices) 行，所以再用 torch.searchsorted 把
#       每个锚点、每个 key 的帧号翻译成“第几行” → *_local_indices_by_key。
#       这一步是 O(M log M) 的向量化查找，替代了早期用 python dict 建“帧号→行号”
#       映射的写法（千万级条目会把内存吃爆）。
#
#   (5) episode 边界必须夹紧，不能跨 episode 取帧
#       动作块取未来 H 步时，如果越过 episode 末尾，取到的会是**下一条 episode 的第 0 帧**
#       ——两条 episode 之间通常没有连续性，这会污染统计量。因此所有“取未来帧/过去帧”
#       的地方都统一夹到本 episode 的最后一帧/第一帧（clamp），与训练时 is_pad 的
#       语义一致（越界帧被当成 padding）。
#
#   (6) 分位数实现：np.partition / torch.sort，而不是 np.quantile
#       np.quantile 会完整排序（O(N log N)）；而“取第 k 小”只需要 O(N) 的
#       np.partition。做法是把 q 换算成 k = int(q * (N - 1))，一次 partition 就能
#       拿回全部 8 个分位点。GPU 上则先 torch.sort 再按 k 取行。
#       代价是分位数无法精确合并：跨数据源合并时只能取加权平均（见
#       mixture_lerobot_dataset.py 的 _aggregate_weighted_stats）。
#
# 【一次统计（_fast_get_dataset_stats）发生了什么】
#   1. 收集 parquet 文件清单，打印数据规模（帧数 / episode 数 / 文件数）；
#   2. 从 meta/episodes/*.parquet 读每条 episode 的帧区间
#      （dataset_from_index / dataset_to_index）——比读整列 episode_index 省内存；
#      读不到就退回“读 episode_index 列”的老办法（_get_episode_metadata_from_parquet）；
#   3. 按 stats_downsample_rate 生成锚点，算出所有需要读的帧号 required_indices，
#      同时记录“每个 key 的每一帧落在 required_indices 的哪个位置”
#      （_build_stats_sampling_plan）；
#   4. 用 pyarrow 顺序扫一遍 parquet，只取 required_indices 这些行
#      （_load_selected_rows_from_parquet，带进度条）；
#   5. 按 meta 把原始列切成每个 part 的张量（复用父类的 _get_meta_source_data /
#      _slice_meta_feature），再按第 4 步的下标取出“锚点 / 动作步”对应的行；
#   6. 计算 state 统计量（compute_state_stats_with_transforms）与
#      action 统计量（compute_action_stats_with_transforms）——两者都会先过一遍
#      processor 的 action_state_transforms，保证统计的是“模型真正看到的数值”
#      （相对动作变换会改变数值范围，统计原始值就错位了）；
#   7. 按 only_keys 过滤输出，返回 {"state": {...}, "action": {...}}。
#
# 【输出契约（返回值结构）】
#   {
#     "state":  {key: {stepwise_min/max/mean/std:      (1, dim),
#                      stepwise_q01/q99/q001/q999/q0001/q9999/q00001/q99999: (1, dim),
#                      global_min/max/mean/std:        (dim,),
#                      global_q01/...:                 (dim,)}},
#     "action": {key: {同上，但 stepwise_* 形状是 (H, dim)，H = action_size}}
#   }
#   所有统计量固定为 float32（下游归一化的统一约定）。
#
# 【常见坑 / 排查清单】
#   · 统计很慢 / 内存很高：先看 stats_downsample_rate（1 = 全量最慢），再看日志里是否
#     走到了“元数据路径”（会打印 “Using episode metadata for boundary computation”）。
#   · 日志出现 “falling back to full scan”：数据集缺 meta/episodes/*.parquet
#     （旧格式，或转换数据时没写元数据），会退回读整列 episode_index，内存明显上升。
#   · 统计量形状不对（例如 action 的 stepwise_* 少了一维）：统计量是按
#     _split_stats_by_key 从合并张量切回来的，形状与 meta 的 raw_shape 绑定；
#     改了 shape_meta 必须重算统计量（缓存的签名也会随之失效）。
#   · 相对动作（Relative*Transform）的统计结果异常：变换只在
#     action_state_transforms 存在时生效；只统计 state 时，仅作用于 action 的变换
#     会退化成空操作（见 compute_state_stats_with_transforms）。
#   · 只想要部分 part：传 only_keys={"action": {...}, "state": {...}}；
#     注意内部仍然全量计算（合并张量路径更省时），只是在返回时过滤。
#   · 分位数合并是近似：多数据源合并时 min/max 取极值、分位数取加权平均，
#     所以“多个数据源合起来的分位数”不等于对全量数据直接算出的分位数。
#
# 【本文件里“保留但当前主路径未调用”的函数】
#   · sliding_window_with_episode_boundary / shift_sequence_with_episode_boundary
#   · _collect_required_frame_indices
#   · compute_action_stats_columnwise / compute_state_stats
#   · fast_quantile_parallel（仅被文件末尾 __main__ 的性能对比脚本调用）
#   它们要么是早期实现，要么是“逐 key 版”的备选实现，读代码时可以先跳过；
#   真正跑在训练流程里的是 compute_action_stats_with_transforms /
#   compute_state_stats_with_transforms 这条链路。
#
# 【相关文件与文档】
#   src/g05/data/base_lerobot_dataset.py        父类：shape_meta 解析 + 取数 + 旧统计路径
#   src/g05/data/mixture_lerobot_dataset.py     多数据源混合 + 统计量加权合并
#   src/g05/data/galaxea_lerobot_dataset.py     R1 系列子类（stats_downsample_rate 默认 10）
#   src/g05/data/lerobot/lerobot_dataset_v3.py  底层读盘层（parquet / mp4 语义）
#   docs/data/schema_zh.md                      shape_meta 字段与 v2.1/v3.0 版本差异
#   docs/architecture/g05_io_zh.md              Dataset → Collate → 模型的张量形状
#   configs/data/r1pro.yaml                     use_stepwise_action_norm 与统计量缓存说明
#   scripts/finetune.py                         调用 get_dataset_stats 并落盘 dataset_stats.json
# =============================================================================

"""LeRobot 数据集基类 V3：为 LeRobot 数据集提供“快速统计量”实现。

中文说明：
    本模块提供 BaseLerobotDatasetV3，用“一次 parquet 扫描 + 向量化/GPU 计算”的方式
    计算归一化统计量，比父类 BaseLerobotDataset 的逐 episode 版本快 10~50 倍。
    state/action 的 meta 使用统一的“显式 raw 布局”：
      - lerobot_key：原始 parquet 列名
      - start_index / raw_shape：该 key 在原始列里的切片起点与宽度
      - time_offset：时间偏移
    这些字段的语义与父类完全一致（详见 base_lerobot_dataset.py 顶部说明）。
"""

# gc：在统计的各阶段主动触发垃圾回收，让几 GB 的中间张量尽早还给系统
import gc
# logging / time：日志与计时（统计过程会打印每一步的耗时，方便定位瓶颈）
import logging
import time
# ThreadPoolExecutor：只被“逐 key 备选实现”（fast_quantile_parallel）使用，
# 主路径是向量化计算，用不到线程池。
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import torch
# tqdm.auto：终端和 notebook 都能正确显示进度条
from tqdm.auto import tqdm

# 父类：本文件全部取数/切分逻辑都复用它，这里只重写统计量计算相关的三个方法
from g05.data.base_lerobot_dataset import BaseLerobotDataset

logger = logging.getLogger(__name__)


def _format_count(value: int) -> str:
    """把整数格式化成带千分位的字符串（日志里读大数字更直观，如 1,234,567）。"""
    return f"{int(value):,}"


def _format_ratio(kept: int, total: int) -> str:
    """把“占比”格式化成百分比字符串，用于“只需要读全量 x% 的帧”这类日志。"""
    if total <= 0:
        # total=0 说明数据集为空，直接给 0.00%，避免除零。
        return "0.00%"
    return f"{100.0 * kept / total:.2f}%"


def _should_show_stats_pbar() -> bool:
    """只有主进程才显示进度条，避免多卡训练时每个 rank 都往同一个终端刷屏。

    读环境变量 RANK / LOCAL_RANK：未设置（单进程）或为 "0"（主进程）时才显示。
    """
    import os

    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    return (rank in (None, "0")) and (local_rank in (None, "0"))


def _stats_write(message: str, color: Optional[str] = None) -> None:
    """带颜色地向终端输出一条统计日志。

    为什么不用 print：统计过程里同时有 tqdm 进度条，print 会把进度条冲花；
    tqdm.write 会正确地把这行输出到进度条上方。termcolor 缺失时退化成无色文本。
    """
    try:
        from termcolor import colored

        output = colored(message, color) if color else message
    except Exception:
        # termcolor 未安装或当前终端不支持颜色时，静默退化成普通文本。
        output = message

    tqdm.write(output)


def fast_quantile_parallel(
    data: np.ndarray, q_values: List[float] = [0.01, 0.99], num_workers: int = 8
) -> Dict[float, np.ndarray]:
    """
    用 np.partition + 线程并行计算分位数，比 np.quantile 快约 15~20 倍。

    ⚠️ 本函数属于“备选/验证用”的实现：主统计路径（compute_action_stats_*）已经
    内联了同样的 np.partition 思路，不再调用它；只有文件末尾的 __main__ 性能对比
    脚本（_compare_quantile_methods）会用它来证明优化有效。读主流程时可以跳过。

    为什么能快：np.quantile 要对每个维度完整排序（O(N log N)）；而“取第 k 小”只需
    O(N) 的 np.partition。这里再把 action_size 个动作步分给多个线程并行。

    Args:
        data: 形状 (N, action_size, dim) 的动作数据；N = 采样帧数，
            action_size = 动作块长度（动作步数），dim = 该 key 的特征维度。
        q_values: 要计算的分位数列表。
        num_workers: 并行线程数。

    Returns:
        dict: {分位数: 结果张量}，其中结果形状为 (action_size, dim)，
        即“每个动作步、每个维度”各一个分位点。
    """
    N, action_size, dim = data.shape

    def compute_for_action_step(i: int) -> Tuple[int, dict]:
        """计算“第 i 个动作步”在所有维度上的分位数（在线程池里并行执行）。"""
        # 只取出该步的那一层：(N, action_size, dim) → (N, dim)
        slice_data = data[:, i, :]  # (N, dim)
        result = {}
        for q in q_values:
            # 把分位数换算成“第 k 小”的下标（与 np.quantile 的线性插值口径略有近似，
            # 因此两边的结果会有极小的数值差异，见 _compare_quantile_methods 的误差打印）。
            k = int(q * (N - 1))
            # np.partition 会把第 k 小的元素放到位置 k 上，所以取 partitioned[k] 就是该分位数。
            # 这里每个 q 单独调一次，读起来最直观；主路径为了效率改成“把多个 k 合并成
            # 一次 partition 调用”（见 compute_action_stats_columnwise 的 ks 列表）。
            partitioned = np.partition(slice_data, k, axis=0)
            result[q] = partitioned[k, :]  # (dim,)，即该动作步、该分位数的取值
        return i, result

    # 预先开好输出缓冲：{分位数: (action_size, dim) 的零张量}
    results = {q: np.zeros((action_size, dim), dtype=data.dtype) for q in q_values}

    # 线程池并行处理各个动作步。这里属于纯 numpy 计算（大部分时间在 C 层），
    # 但 numpy 的 partition 在排队时仍会释放 GIL，所以多线程能拿到实际加速。
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list(executor.map(compute_for_action_step, range(action_size)))
        for i, q_dict in futures:
            for q, vals in q_dict.items():
                # 回填到第 i 个动作步那一行
                results[q][i, :] = vals

    return results


def sliding_window_with_episode_boundary(
    data: torch.Tensor, ep_indices: torch.Tensor, window_size: int
) -> torch.Tensor:
    """
    向量化滑窗：跨到 episode 边界时“复制最后一帧”，而不是串到上一条 episode 的第一帧。

    ⚠️ 当前主统计路径没有调用本函数（compute_action_stats_with_transforms 里用的是
    更省内存的“逐列 gather”写法：每个动作步只保留 (N, D) 一份数据，而不是一次性
    展开成 (N, window_size, D)）。这里保留它作为参考实现。

    为什么要“复制最后一帧”：滑窗取未来帧时，episode 末尾之后没有真实数据；
    如果直接按行号 +j 取，会取到下一条 episode 的第 0 帧 —— 两条 episode 之间通常
    不连续，这会让统计量被无关数据污染。复制最后一帧相当于“停在原地”，
    与训练时把越界帧标成 padding 的语义一致。

    Args:
        data: 输入数据，形状 (N, D)；N = 总帧数（所有 episode 首尾相接）。
        ep_indices: 每帧所属的 episode 下标，形状 (N,)。
        window_size: 滑窗长度（= 动作块长度 H）。

    Returns:
        滑窗结果，形状 (N, window_size, D)：result[i, j] = data[第 i 帧往后第 j 帧]。
    """
    N, D = data.shape

    # 先找出每条 episode 的边界：ep_indices 变化的位置 +1 就是“新 episode 的第 0 帧”。
    # 例：ep_indices = [0,0,0,1,1] → ep_changes = [3] → 第 0 条 episode 是 [0,3)。
    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    # ep_ends 是“开区间终点”（不含），末尾补上 N 作为最后一条 episode 的终点。
    ep_ends = torch.cat([ep_changes, torch.tensor([N])])
    ep_starts = torch.cat([torch.tensor([0]), ep_changes])

    # 为每一帧记录“它所属 episode 的最后一帧下标”，即上面说的 clamp 上界。
    frame_ep_end = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        frame_ep_end[start:end] = end - 1  # 最后一帧的下标（end 是开区间终点）

    # 逐列构建滑窗结果（内存换简单：完整展开成 (N, window_size, D)）。
    result = torch.zeros(N, window_size, D, dtype=data.dtype)
    for j in range(window_size):
        # 第 j 列要取的下标：每帧自身的下标 + j
        target_idx = torch.arange(N) + j
        # 越过本条 episode 末尾时，夹到本 episode 的最后一帧（复制而不是跨界）
        target_idx = torch.minimum(target_idx, frame_ep_end)
        # 再兜一层：越过整个数据集的最后一帧时夹到 N-1（防止下标越界）
        target_idx = torch.minimum(target_idx, torch.tensor(N - 1))
        result[:, j] = data[target_idx]

    return result


def shift_sequence_with_episode_boundary(
    data: torch.Tensor, ep_indices: torch.Tensor, offset: int
) -> torch.Tensor:
    """
    在 episode 边界内整体平移一个“逐帧张量”（把 t 帧的数据换成 t+offset 帧的）。

    ⚠️ 与 sliding_window_with_episode_boundary 一样，当前主统计路径未调用本函数
    （主路径用 _shift_indices_with_episode_boundary 只平移“下标”，再统一 gather，
    省掉重复的数据拷贝）。保留作为语义参考。

    典型用途：state 的 time_offset（例如 time_offset=1 表示“用下一帧的 state 作为
    动作目标”，即 state-as-action）。

    Args:
        data: 形状 (N, D) 或 (N,) 的张量；N = 所有 episode 的总帧数。
        ep_indices: 每帧所属的 episode 下标，形状 (N,)。
        offset: 时间偏移，作用在第 0 维。正数 = 取未来帧，负数 = 取过去帧。

    Returns:
        与输入同形状的张量。越界（跨出 episode）的位置会被夹到该 episode 的第一帧/
        最后一帧，因此任何取值都不会来自别的 episode。
    """
    if data.ndim == 1:
        # 统一成 (N, 1) 处理，逻辑只写一份；返回时形状不变（外面按原维度用）。
        data = data.unsqueeze(-1)

    N, _ = data.shape
    if N == 0:
        # 空数据集：没有帧可平移，直接返回（也避免下面 cat 出空张量）。
        return data

    # 与滑窗函数相同：先算每条 episode 的 [start, end) 区间。
    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    ep_ends = torch.cat([ep_changes, torch.tensor([N], dtype=torch.long)])
    ep_starts = torch.cat([torch.tensor([0], dtype=torch.long), ep_changes])

    # 每帧所属 episode 的首帧 / 末帧下标，作为 clamp 的上下界。
    frame_ep_start = torch.zeros(N, dtype=torch.long)
    frame_ep_end = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        frame_ep_start[start:end] = start
        frame_ep_end[start:end] = end - 1

    # 平移后先夹下界（不能早于本 episode 第 0 帧），再夹上界（不能晚于最后一帧）。
    indices = torch.arange(N, dtype=torch.long) + offset
    indices = torch.maximum(indices, frame_ep_start)
    indices = torch.minimum(indices, frame_ep_end)
    return data[indices]


def _compute_frame_ep_end(ep_indices: torch.Tensor, N: int) -> torch.Tensor:
    """
    预计算“每帧所属 episode 的最后一帧下标”，供滑窗/clamp 反复使用。

    为什么要预计算：统计阶段每个动作步（H 次）、每个 state/action key 都要做一次
    “未来帧不能越界”的夹紧；如果每次都重新扫一遍 ep_indices 找边界，就是 H×K 次
    多余的 O(N) 扫描。算一次 (N,) 的表，后面全是 O(1) 的查表。

    Args:
        ep_indices: 每帧所属的 episode 下标，形状 (N,)
        N: 总帧数

    Returns:
        形状 (N,) 的张量，frame_ep_end[i] = 第 i 帧所在 episode 的最后一帧下标。
    """
    # 相邻帧 episode 变化处 +1 = 新 episode 的起始行（参见 sliding_window 里的说明）。
    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    ep_ends = torch.cat([ep_changes, torch.tensor([N])])
    ep_starts = torch.cat([torch.tensor([0]), ep_changes])
    frame_ep_end = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        # 区间 [start, end) 内的所有帧，其“所属 episode 末帧”都是 end-1
        frame_ep_end[start:end] = end - 1
    return frame_ep_end


def _compute_frame_ep_start(ep_indices: torch.Tensor, N: int) -> torch.Tensor:
    """与 _compute_frame_ep_end 对称：返回 frame_ep_start[i] = 第 i 帧所属 episode 的第 0 帧下标。

    只在“回退路径”（读整列 episode_index 建表）里使用，作为负向 time_offset 的夹紧下界。
    """
    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    ep_starts = torch.cat([torch.tensor([0], dtype=torch.long), ep_changes])
    ep_ends = torch.cat([ep_changes, torch.tensor([N], dtype=torch.long)])
    frame_ep_start = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        frame_ep_start[start:end] = start
    return frame_ep_start


def _shift_indices_with_episode_boundary(
    indices: torch.Tensor,
    offset: int,
    frame_ep_start: torch.Tensor,
    frame_ep_end: torch.Tensor,
) -> torch.Tensor:
    """把“帧下标”平移 offset，并把结果夹在本 episode 的 [首帧, 末帧] 之内。

    与 shift_sequence_with_episode_boundary 的区别：这里只算下标、不动数据，
    因此可以先把所有 key、所有动作步的下标算完再一次性 gather（更省内存、更快）。

    Args:
        indices: 待平移的帧下标，形状 (M,)。
        offset: 时间偏移，正数取未来帧、负数取过去帧。
        frame_ep_start: 每个帧下标所属 episode 的首帧（用 indices 去查）。
        frame_ep_end: 每个帧下标所属 episode 的末帧。

    Returns:
        平移并夹紧后的帧下标，形状 (M,)。
    """
    shifted = indices + offset
    # 注意：上下界是用**原始下标** indices 查出来的，即“这些帧本来属于哪条 episode”。
    shifted = torch.maximum(shifted, frame_ep_start[indices])
    shifted = torch.minimum(shifted, frame_ep_end[indices])
    return shifted


def _get_episode_metadata_from_parquet(
    parquet_files: List[Any],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    从 meta/episodes/*.parquet 直接读 episode 边界，避免把整列 episode_index 读进内存。

    为什么需要它：统计阶段要回答“某帧属于哪条 episode、这条 episode 的首帧/末帧是几号”，
    而这件事有两种做法：
      ① 读数据表里的 episode_index 整列（N 个 int64）→ 自己找边界：简单但要额外
         读一遍全部帧的索引，千万帧数据上就是几十 MB 的内存和一次全表扫描；
      ② 读元数据表 meta/episodes：每条 episode 一行，直接给出帧区间：几乎零成本。
    本函数就是做法 ②，并在读不到时返回 None 让调用方回退到做法 ①。

    LeRobot v3 的 meta/episodes/*.parquet 里，每条 episode 存的是：
      - dataset_from_index：该 episode 的起始帧号（闭区间起点）
      - dataset_to_index：该 episode 的结束帧号（**开区间**终点，即最后一帧 + 1）
    注意这是“整个 dataset 的全局帧号”，不是 episode 内的偏移。

    Args:
        parquet_files: 该 subset 下所有帧级 parquet 文件的路径列表
            （形如 <dataset_root>/data/chunk-XXX/file-YYY.parquet）。
            一个 subset 里可能包含多个数据集目录（多目录首尾拼接），所以这里按 root 分组处理。

    Returns:
        ep_starts: (num_episodes,) 每条 episode 的起始帧号；失败时返回 None。
        ep_ends: (num_episodes,) 每条 episode 的结束帧号（开区间）；失败时返回 None。
        total_frames: 总帧数（由各目录元数据累加得到）。
        约定：只要有任何一个目录读不到元数据，就整体返回 (None, None, 0)，
              让调用方统一回退到“读 episode_index 列”的全扫描路径。
    """
    from pathlib import Path

    import pyarrow.parquet as pq

    if not parquet_files:
        # 没有任何数据文件（空目录/路径配错），直接当作“没有元数据可用”。
        return None, None, 0

    # 数据文件路径形如 <dataset_root>/data/chunk-XXX/file.parquet。
    # 从文件往上走三层（file → chunk → data → dataset_root）就得到该目录的 root；
    # 这里按出现顺序去重，得到“本 subset 涉及哪几个数据集目录”。
    #
    # 历史 bug（已修）：早期实现只看 parquet_files[0] 所在目录的元数据，把第一个
    # 数据集的帧数当成整个混合数据集的 N；于是后面生成的抽样锚点 sampled_base_idx
    # 只覆盖到第一个数据集，其余数据集的统计量被完全漏掉。
    # 现在改成“逐目录累加 + 坐标平移到全局帧号”，见下面的 offset 逻辑。
    roots: List[Path] = []
    seen = set()
    for p in parquet_files:
        root = Path(p).parent.parent.parent
        if root not in seen:
            seen.add(root)
            roots.append(root)

    # 每个目录的 episode 边界先攒起来，最后拼成一份全局表。
    # offset 记录“前面几个目录一共有多少帧”，用于把各目录的局部帧号平移到全局帧号。
    all_starts: List[torch.Tensor] = []
    all_ends: List[torch.Tensor] = []
    offset = 0
    for root in roots:
        meta_dir = root / "meta" / "episodes"
        if not meta_dir.exists():
            # 没有 episode 元数据（旧格式数据集），让调用方回退到全扫描路径。
            logger.debug(
                f"[Stats] meta/episodes not found at {meta_dir}, falling back to full scan"
            )
            return None, None, 0
        # rglob：episode 元数据可能按 chunk 分文件存放，全部收集起来一起读。
        episode_files = sorted(meta_dir.rglob("*.parquet"))
        if not episode_files:
            logger.debug(f"[Stats] no parquet files in {meta_dir}, falling back to full scan")
            return None, None, 0
        try:
            # 只读需要的那两列，避免把 episode 元数据里的其他列（视频时间戳等）也读进来。
            ep_table = pq.read_table(
                episode_files, columns=["dataset_from_index", "dataset_to_index"]
            )
        except Exception as e:
            # 元数据文件损坏/列名不符：同样回退，不让统计流程整体失败。
            logger.debug(f"[Stats] failed to read {meta_dir}: {e}, falling back to full scan")
            return None, None, 0

        starts = torch.tensor(ep_table["dataset_from_index"].to_numpy(), dtype=torch.long)
        ends = torch.tensor(ep_table["dataset_to_index"].to_numpy(), dtype=torch.long)
        if len(ends) == 0:
            # 该目录没有任何 episode（空数据集）：跳过，offset 不变。
            continue
        # 本目录在“局部坐标系”下的帧区间是 [starts[0], ends[-1])。
        # 平移到全局坐标系：先减掉本目录的起点（对齐到 0），再加上前面目录的总帧数。
        # 这样 ep_starts/ep_ends 与 FrameDataset/MultiLeRobotDataset 的全局帧下标一致。
        local_start = int(starts[0].item())
        local_end = int(ends[-1].item())
        shift = offset - local_start
        all_starts.append(starts + shift)
        all_ends.append(ends + shift)
        # 累加本目录的帧数，作为下一个目录的起点偏移。
        offset += local_end - local_start

    if not all_ends:
        # 所有目录都是空的 → 视作没有元数据可用。
        return None, None, 0

    return torch.cat(all_starts), torch.cat(all_ends), offset


def _get_ep_indices_for_frame_indices(
    frame_indices: torch.Tensor,
    ep_starts: torch.Tensor,
    ep_ends: torch.Tensor,
) -> torch.Tensor:
    """
    查表：给定一批帧号，返回每帧属于第几条 episode。

    用途：走“元数据路径”时，我们只读了 episode 的帧区间、没读 episode_index 列，
    但后面 compute_action_stats_with_transforms 需要一个“逐帧 episode 下标”来夹紧边界，
    于是用二分查找现推一份出来（成本 O(M log E)，比重读一列 N 个整数便宜得多）。

    实现要点：ep_ends 是**开区间**终点（末帧 + 1），且按 episode 顺序严格递增，
    所以“最后一个 ep_end <= 帧号”的位置正好等于该帧所在 episode 的下标；
    side="right" 表示相等时取右边 → 恰好落在开区间边界的那一帧归入后一条 episode，
    与“末帧 = ep_end - 1”的定义一致。

    Args:
        frame_indices: 要查询的帧号，形状 (M,)。
        ep_starts: 每条 episode 的起始帧号，形状 (E,)。
            （本函数只用到 ep_ends；ep_starts 保留在签名里是为了与相邻工具函数保持对称，
              调用方无需再额外过滤参数。）
        ep_ends: 每条 episode 的结束帧号（开区间），形状 (E,)，必须递增。

    Returns:
        episode_index: 每帧对应的 episode 下标，形状 (M,)。
    """
    ep_idx = torch.searchsorted(ep_ends, frame_indices, side="right")
    return ep_idx


def _get_frame_ep_boundaries_for_indices(
    frame_indices: torch.Tensor,
    ep_starts: torch.Tensor,
    ep_ends: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    查表：给定一批帧号，返回“这些帧所属 episode 的首帧 / 末帧”。

    与 _get_ep_indices_for_frame_indices 的关系：先二分求出 episode 下标，再用它去
    ep_starts / ep_ends 里取值。返回值直接就是 clamp 用的上下界，因为统计阶段需要的是
    “不能越过的边界帧号”，而不是 episode 编号本身。

    注意返回的末帧是**闭区间**（ep_ends 减 1），与 _compute_frame_ep_end 的口径一致。

    Args:
        frame_indices: 要查询的帧号，形状 (M,)。
        ep_starts: 每条 episode 的起始帧号，形状 (E,)。
        ep_ends: 每条 episode 的结束帧号（开区间），形状 (E,)。

    Returns:
        frame_ep_start: 每帧所属 episode 的首帧，形状 (M,)。
        frame_ep_end: 每帧所属 episode 的末帧（闭区间），形状 (M,)。
    """
    ep_idx = torch.searchsorted(ep_ends, frame_indices, side="right")
    frame_ep_start = ep_starts[ep_idx]
    # 开区间终点 - 1 = 闭区间末帧（末尾几帧的 clamp 上界）
    frame_ep_end = ep_ends[ep_idx] - 1
    return frame_ep_start, frame_ep_end


def _collect_required_frame_indices(
    sampled_base_idx: torch.Tensor,
    frame_ep_start: torch.Tensor,
    frame_ep_end: torch.Tensor,
    action_size: int,
    state_meta: List[Dict[str, Any]],
    action_meta: List[Dict[str, Any]],
) -> torch.Tensor:
    """算出“统计一共需要读哪些帧号”（去重、排序后的全局帧号）。

    ⚠️ 已被 _build_stats_sampling_plan 取代：新版本除了这批帧号，还会顺带返回
    “每个 state key 的帧、每个 action key 的每一步动作落在哪个位置”，
    而本函数只返回帧号集合，调用方还得自己再定位一次。当前主路径不调用它，
    这里保留是为了让“要读哪些帧”这件事的语义可以单独阅读。

    需要的帧一共来自三处：
      1) 抽样锚点本身（每个锚点的观测帧）；
      2) 带 time_offset 的 state（例如 time_offset=1 表示统计 t+1 帧的 state）；
      3) 每个锚点之后 action_size 步的 action（动作块），同样要支持 time_offset。
    所有“往后取/往前取”都通过 _shift_indices_with_episode_boundary 夹在 episode 内。

    Args:
        sampled_base_idx: 抽样锚点的帧号，形状 (M,)。
        frame_ep_start / frame_ep_end: 每帧所属 episode 的首帧/末帧（长度 N），
            由 _compute_frame_ep_start / _compute_frame_ep_end 预计算。
        action_size: 动作块长度 H。
        state_meta / action_meta: 父类解析好的 state/action meta 列表。

    Returns:
        torch.unique(sorted=True) 之后的帧号张量 —— 有序是后续用 searchsorted
        定位的前提（见 _fast_get_dataset_stats 里的 sampled_local_idx）。
    """
    # 用列表把“各路来源的下标”攒起来，最后一次性 cat + unique，比边算边合并快。
    required = [sampled_base_idx]

    # ① state：只有 time_offset != 0 时才需要额外取帧（否则就是锚点本身）。
    for meta in state_meta:
        offset = int(meta.get("time_offset", 0))
        if offset != 0:
            required.append(
                _shift_indices_with_episode_boundary(
                    sampled_base_idx,
                    offset,
                    frame_ep_start,
                    frame_ep_end,
                )
            )

    # ② action：每个锚点要取未来 H 步。这里逐 step 展开（而不是展开成 (M, H) 的大张量），
    #    因为后面会用 searchsorted 找位置，用一维列表更好拼接。
    for meta in action_meta:
        offset = int(meta.get("time_offset", 0))
        for j in range(action_size):
            # 先做“动作块不能跨 episode”的夹紧：锚点 + j 超过本条 episode 末帧就停在末帧。
            step_idx = torch.minimum(sampled_base_idx + j, frame_ep_end[sampled_base_idx])
            if offset != 0:
                # time_offset 用于 state-as-action 这类场景：再整体平移一次并夹紧。
                step_idx = _shift_indices_with_episode_boundary(
                    step_idx,
                    offset,
                    frame_ep_start,
                    frame_ep_end,
                )
            required.append(step_idx)

    # unique(sorted=True)：去重 + 升序，方便后面用二分查找定位行号。
    return torch.unique(torch.cat(required), sorted=True)


def _build_stats_sampling_plan(
    sampled_base_idx: torch.Tensor,
    action_size: int,
    state_meta: List[Dict[str, Any]],
    action_meta: List[Dict[str, Any]],
    ep_starts: Optional[torch.Tensor] = None,
    ep_ends: Optional[torch.Tensor] = None,
    frame_ep_start: Optional[torch.Tensor] = None,
    frame_ep_end: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
    """
    构造“统计抽样计划”：告诉后续步骤“要读哪些帧”以及“每个 key 该取哪些帧”。

    这是统计流程的枢纽函数，输出三样东西（都是**原始帧号**，还没变成行号）：
      1) required_indices：所有需要读的帧号（去重升序）→ 交给 pyarrow 精确取行；
      2) state_indices_by_key：每个 state key 要取哪些帧（通常就是锚点本身；
         若该 state 的 time_offset != 0，则是平移后的那批帧）；
      3) action_indices_by_key：每个 action key 的“每一步动作”分别要取哪些帧，
         形状是 {key: [第 0 步的帧号数组, 第 1 步的帧号数组, ...]}，长度为 action_size。
    为什么要把“取哪几帧”和“读哪些帧”拆开：一次只需要读 required_indices 这批行
    （可能远少于 N），读回来之后再用下标把它们还原成各个 key 的 (帧, 维度) 视图。

    支持两种边界的表达方式（调用方按数据来源二选一）：
      模式 1（省内存，推荐）：只给 ep_starts/ep_ends（每条 episode 首/末帧，长度 E），
            需要边界时用二分现查 → 无需构造长度 N 的表；
      模式 2（兼容旧路径）：给预计算好的 frame_ep_start/frame_ep_end（长度 N），
            查表 O(1)。

    Args:
        sampled_base_idx: 抽样锚点的帧号，形状 (M,)。
        action_size: 动作块长度 H。
        state_meta / action_meta: 父类解析好的 state/action meta 列表。
        ep_starts: 每条 episode 的起始帧号，形状 (E,)；模式 1 使用。
        ep_ends: 每条 episode 的结束帧号（开区间），形状 (E,)；模式 1 使用。
        frame_ep_start: 每帧所属 episode 的首帧，形状 (N,)；模式 2 使用。
        frame_ep_end: 每帧所属 episode 的末帧（闭区间），形状 (N,)；模式 2 使用。

    Returns:
        required_indices: 需要读取的全部帧号（unique + 升序）。
        state_indices_by_key: {state key: 该 key 要取的帧号}。
        action_indices_by_key: {action key: [第 0..H-1 步各自要取的帧号]}。
    """
    # 逐 key 记录“要取哪些帧”，与上面命名一致，均为原始帧号。
    state_indices_by_key: Dict[str, torch.Tensor] = {}
    action_indices_by_key: Dict[str, List[torch.Tensor]] = {}
    # 所有帧号的并集来源；锚点本身一定要读（观测帧）。
    required = [sampled_base_idx]

    # 只要给了 episode 元数据，就走模式 1（按需二分）；否则退化为模式 2（查预计算表）。
    use_ep_meta = ep_starts is not None and ep_ends is not None

    def get_boundaries_for_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """取这批帧号所属 episode 的首帧/末帧（两种模式统一入口）。"""
        if frame_ep_start is not None and frame_ep_end is not None:
            # 模式 2：直接查长度为 N 的表
            return frame_ep_start[indices], frame_ep_end[indices]
        else:
            # 模式 1：用 episode 边界二分现算
            return _get_frame_ep_boundaries_for_indices(indices, ep_starts, ep_ends)

    # ── state：一般只取锚点帧；带 time_offset 时取平移后的帧 ──
    for meta in state_meta:
        key = meta["key"]
        state_idx = sampled_base_idx
        offset = int(meta.get("time_offset", 0))
        if offset != 0:
            # 两种模式只是“边界怎么拿”不同，夹紧逻辑完全一致：
            # 目标帧 = 当前帧 + offset，再夹到本 episode 的 [首帧, 末帧]。
            if use_ep_meta:
                fep_start, fep_end = get_boundaries_for_indices(state_idx)
                shifted = state_idx + offset
                shifted = torch.maximum(shifted, fep_start)
                shifted = torch.minimum(shifted, fep_end)
                state_idx = shifted
            else:
                state_idx = _shift_indices_with_episode_boundary(
                    state_idx, offset, frame_ep_start, frame_ep_end
                )
        state_indices_by_key[key] = state_idx
        required.append(state_idx)

    # ── action：每个锚点展开成 H 步的动作块，每一步都可能被夹紧/平移 ──
    for meta in action_meta:
        key = meta["key"]
        offset = int(meta.get("time_offset", 0))
        # per_step_indices[j] = 该 key 第 j 步要取的帧号（长度都是 M）
        per_step_indices = []

        # 动作块的上界只跟“锚点所在 episode 的末帧”有关，循环外先算一次复用
        # （否则每个 step 都要查一次表，白白多花 H 倍开销）。
        if use_ep_meta:
            _, fep_end_for_base = get_boundaries_for_indices(sampled_base_idx)
        else:
            fep_end_for_base = frame_ep_end[sampled_base_idx]

        for j in range(action_size):
            # 先夹一次：锚点 + j 越过 episode 末尾就停在末帧（与训练时 is_pad 口径一致）
            step_idx = torch.minimum(sampled_base_idx + j, fep_end_for_base)
            if offset != 0:
                # 再按 time_offset 平移一次，并以“平移前这些帧所属的 episode”为界夹紧
                if use_ep_meta:
                    fep_start, fep_end = get_boundaries_for_indices(step_idx)
                    shifted = step_idx + offset
                    shifted = torch.maximum(shifted, fep_start)
                    shifted = torch.minimum(shifted, fep_end)
                    step_idx = shifted
                else:
                    step_idx = _shift_indices_with_episode_boundary(
                        step_idx, offset, frame_ep_start, frame_ep_end
                    )
            per_step_indices.append(step_idx)
            required.append(step_idx)
        action_indices_by_key[key] = per_step_indices

    # 把各路帧号拼起来去重排序：得到的这批就是真正要从 parquet 读的行。
    # 必须升序，后面才能用 searchsorted 把“帧号”翻译成“读回来的第几行”。
    required_indices = torch.unique(torch.cat(required), sorted=True)
    return required_indices, state_indices_by_key, action_indices_by_key


def _load_selected_rows_from_parquet(
    parquet_files: List[Any],
    columns: List[str],
    required_indices: torch.Tensor,
    desc: str,
) -> Dict[str, torch.Tensor]:
    """从 parquet 里“只挑出需要的那几行、那几列”，返回 {列名: (行数, 维度) 张量}。

    这是整个统计流程里唯一大量读磁盘的地方，也是快慢的关键：逐 episode 读（父类做法）
    会反复定位文件、反复解码；这里改成“顺序扫一遍 + 精确取行”，且完全不碰图像。

    算法（流式归并，前提：required_indices 已升序去重）：
      · pyarrow 的 scanner 按批次（batch）顺序吐行，global_offset 记录“当前批次第一行
        在整个数据里的行号”；
      · 因为 required_indices 是升序的，只需要维护一个游标 next_required，
        每来一个批次就把“落在本批次内的那些帧号”全部取出来（batch.take），
        本批次不需要的帧直接跳过，不做任何转换；
      · 局部行号 = 全局帧号 - global_offset；
      · 全部取够（next_required 追上总数）就提前 break，不再读后面的文件。
    这样每个 parquet 文件最多被顺序读一遍，且只对需要的行做 Arrow → numpy 转换。

    为什么用 pyarrow.dataset 而不是 pandas：dataset 能跨多个文件/多个 chunk 统一编号，
    列裁剪（columns=）和行裁剪（take）都发生在 C++ 层，几乎没有 python 循环开销。

    Args:
        parquet_files: 要扫描的帧级 parquet 文件（一个 subset 下全部文件）。
        columns: 需要读取的原始列名（来自 meta 的 lerobot_key，已去重）。
        required_indices: 需要读取的全局帧号，必须升序（由 torch.unique(sorted=True) 保证）。
        desc: 进度条标题（例如 "📥 Loading sampled stats rows"）。

    Returns:
        {列名: 形状 (行数, 维度) 的 float32 张量}，行数 = len(required_indices)，
        行序与 required_indices 一一对应（后续用 searchsorted 得到的行号来索引）。
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    # 帧号转成 numpy（C++ 侧按 int64 数组取行更快，且避免每行一次 python 索引）。
    required_np = required_indices.cpu().numpy()
    # next_required：下一个还没取到的帧号在 required_np 里的位置（遍历游标）。
    next_required = 0
    total_required = int(required_np.shape[0])
    # global_offset：当前批次第一行在整个数据集里的全局行号。
    global_offset = 0
    # 逐列的缓冲列表（每个批次取出来的片段先攒着，最后统一 concatenate）。
    arrays_by_column: Dict[str, List[np.ndarray]] = {col: [] for col in columns}

    # 把多个 parquet 文件当成一张大表来扫描（跨文件行号连续），并做列裁剪。
    dataset = ds.dataset(parquet_files, format="parquet")
    scanner = dataset.scanner(columns=columns)
    # 只有主进程显示进度条，避免多卡训练时日志互相刷屏。
    show_pbar = _should_show_stats_pbar()

    with tqdm(
        total=total_required,
        desc=desc,
        dynamic_ncols=True,
        leave=False,  # 跑完就擦掉进度条，不干扰后面的日志
        disable=not show_pbar,
    ) as progress:
        for batch in scanner.to_batches():
            batch_size = batch.num_rows
            # 本批次的全局行号范围是 [global_offset, batch_end)
            batch_end = global_offset + batch_size

            # 找出“帧号仍落在本批次内”的那一段 required_np[next_required:take_end)。
            # 因为 required_np 升序，这个 while 一旦遇到 >= batch_end 就可以停。
            take_end = next_required
            while take_end < total_required and required_np[take_end] < batch_end:
                take_end += 1

            if take_end > next_required:
                # 把全局帧号换成“本批次内的局部行号”，一次性取出来。
                local_indices = required_np[next_required:take_end] - global_offset
                taken = batch.take(pa.array(local_indices, type=pa.int64()))
                for col in columns:
                    # zero_copy_only=False：允许 Arrow → numpy 时复制（有些类型无法零拷贝）。
                    col_np = taken.column(col).to_numpy(zero_copy_only=False)
                    arrays_by_column[col].append(col_np)
                progress.update(take_end - next_required)
                next_required = take_end

            global_offset = batch_end
            if next_required >= total_required:
                # 需要的行都拿齐了，后面的批次/文件直接不看（这是“只读需要的部分”的关键）。
                break

    # 逐列把各批次的片段拼成完整数组，并统一成 (行数, 维度) 的 float32 张量。
    selected: Dict[str, torch.Tensor] = {}
    for col in columns:
        if arrays_by_column[col]:
            col_np = np.concatenate(arrays_by_column[col], axis=0)
        else:
            # 一帧都没取到（例如 required_indices 为空）→ 建空数组占位，形状约定保持一致。
            col_np = np.empty((0,), dtype=np.float32)
        if col_np.dtype == object:
            # 列里存的是“变长列表/数组”时，Arrow 会给 object 数组，
            # 需要 stack 成规则的二维数组才能做后续的维度切片。
            col_np = np.stack(col_np)
        if col_np.ndim == 1:
            # 单维特征（如夹爪）补一列，让后续统一按 [..., start:start+width] 切片。
            col_np = col_np[:, np.newaxis]
        # 统一转 float32：统计量与归一化全程用 float32，避免 float64 白占一倍内存。
        selected[col] = torch.tensor(col_np, dtype=torch.float32)

    return selected


def compute_action_stats_columnwise(
    data: torch.Tensor,
    ep_indices: torch.Tensor,
    action_size: int,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    frame_ep_end: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    逐列计算 action 统计量：不展开 (N, action_size, D) 的滑窗大张量。

    ⚠️ 这是“单 key 版”的实现，当前主路径不走这里（主路径用
    compute_action_stats_with_transforms，它把多个 key 拼成一条宽列一次算完）。
    保留它的价值在于：它把“滑窗 + 分位数”的做法写得最直白，适合对照理解。

    核心技巧：与其先构造滑窗张量（N × H × D，动辄几十 GB），不如逐 step 处理——
    对第 j 步，把“每一帧往后第 j 帧”的那一列 (N, D) 取出来，直接在这一列上算
    min/max/mean/std/分位数。峰值内存从 O(N × H × D) 降到 O(N × D)。

    Args:
        data: 单个 key 的 action 数据，形状 (N, D)。
        ep_indices: 每帧所属 episode 下标，形状 (N,)。
        action_size: 滑窗长度 H（动作块长度）。
        quantile_low: 下分位数（默认 0.01）。
        quantile_high: 上分位数（默认 0.99）。
        frame_ep_end: 预先算好的“每帧所属 episode 末帧”；传 None 时内部现算。

    Returns:
        统计量字典，含逐 step 的 (H, D) 张量与全局的 (D,) 张量。
    """
    if data.ndim == 1:
        # 单维特征统一补成 (N, 1)，让后面的 amin(0)/分位数逻辑只有一份。
        data = data.unsqueeze(-1)
    # 统一 float32：统计与归一化全程 float32，避免 float64 白占一倍内存。
    data = data.float()
    N, D = data.shape

    if frame_ep_end is None:
        # 没传就现算：滑窗第 j 步要夹到“本 episode 的最后一帧”，避免取到别的 episode。
        frame_ep_end = _compute_frame_ep_end(ep_indices, N)

    # 8 组分位数：q01/q99 是常用稳健范围（配合裁掉离群点的归一化），
    # 更极端的 q001…q00001 是逐级放宽的兜底范围（相对动作裁剪区间等场景会用到）。
    quantile_spec = [
        ("q01", quantile_low),
        ("q99", quantile_high),
        ("q001", 0.001),
        ("q999", 0.999),
        ("q0001", 0.0001),
        ("q9999", 0.9999),
        ("q00001", 0.00001),
        ("q99999", 0.99999),
    ]

    # 输出缓冲：每个动作步一行，每行 D 个维度。
    stepwise_min = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_max = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_mean = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_std = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_quantiles = {
        name: torch.empty(action_size, D, dtype=torch.float32) for name, _ in quantile_spec
    }

    # 所有帧的自身下标，以及“全数据集最后一帧”的下标（第二道兜底夹紧）。
    base_idx = torch.arange(N, dtype=torch.long)
    N_limit = torch.tensor(N - 1, dtype=torch.long)

    # 预先把 8 个分位数换算成“第 k 小”的下标，供 np.partition 一次调用复用。
    # 夹到 [0, N-1]：极端 q（如 0.99999）在很小的数据集上可能算出越界的 k。
    ks = [max(0, min(int(q_val * (N - 1)), N - 1)) for _, q_val in quantile_spec]

    for j in range(action_size):
        # 第 j 步要取“每帧往后第 j 帧”，先夹在 episode 内，再夹在全数据集内。
        target_idx = base_idx + j
        target_idx = torch.minimum(target_idx, frame_ep_end)
        target_idx = torch.minimum(target_idx, N_limit)
        col_data = data[target_idx]  # (N, D) — 只有一列数据的内存占用

        # 四类基础统计量：沿第 0 维（所有帧）归约，得到该动作步的 (D,) 结果。
        stepwise_min[j] = col_data.amin(0)
        stepwise_max[j] = col_data.amax(0)
        stepwise_mean[j] = col_data.mean(0)
        stepwise_std[j] = col_data.std(0)

        # 一次 np.partition 传入全部 k，拿回所有分位点（比逐个 q 调 np.quantile 快得多）。
        # 说明：这里把每行的分位数都算出来了，是因为 np.partition 的 k 是“位置”，
        # 传入多个 k 只做一次划分，摊薄了排序成本。
        col_np = col_data.numpy()
        partitioned = np.partition(col_np, ks, axis=0)
        for (q_name, _), k in zip(quantile_spec, ks):
            # copy()：partitioned 是共享内存的视图，转 torch 前要先拷贝，避免被后续复用覆盖。
            stepwise_quantiles[q_name][j] = torch.from_numpy(partitioned[k].copy()).float()

        # 显式释放本步的中间张量，避免在循环里堆积内存。
        del col_data, col_np, partitioned

    # 组装最终的统计量字典。
    # global_std 用“总方差分解”（law of total variance）：
    #   全局方差 = 各动作步方差的平均（组内）+ 各动作步均值的方差（组间）
    # 只把各步的 std 求平均是错的：那样会漏掉“不同动作步均值不同”带来的波动。
    if stepwise_mean.shape[0] > 1:
        _global_std = (stepwise_std.pow(2).mean(0) + stepwise_mean.var(0)).sqrt()
    else:
        # 只有 1 个动作步时 var(0) 无定义（样本数为 1），直接取该步的 std。
        _global_std = stepwise_std.squeeze(0)
    stats = {
        "stepwise_min": stepwise_min,
        "stepwise_max": stepwise_max,
        "stepwise_mean": stepwise_mean,
        "stepwise_std": stepwise_std,
        # 全局量 = 再沿“动作步”维度归约一次：min 取最小、max 取最大，保证覆盖全部取值。
        "global_min": stepwise_min.amin(0),
        "global_max": stepwise_max.amax(0),
        "global_mean": stepwise_mean.mean(0),
        "global_std": _global_std,
    }

    for q_name, _ in quantile_spec:
        stats[f"stepwise_{q_name}"] = stepwise_quantiles[q_name]

    # 全局分位数按“保守合并”处理：下分位取各步最小值、上分位取各步最大值。
    # （分位数本身不能再求平均；取极值能保证区间不窄于真实分布。）
    for q_name in ["q01", "q001", "q0001", "q00001"]:
        stats[f"global_{q_name}"] = stats[f"stepwise_{q_name}"].amin(0)
    for q_name in ["q99", "q999", "q9999", "q99999"]:
        stats[f"global_{q_name}"] = stats[f"stepwise_{q_name}"].amax(0)

    return stats


def compute_state_stats(
    data: torch.Tensor,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
) -> Dict[str, torch.Tensor]:
    """
    计算单个 state key 的统计量（state 不需要滑窗，逻辑比 action 简单得多）。

    ⚠️ 这是“单 key 版”的实现，当前主路径走 compute_state_stats_with_transforms
    （多 key 拼成一条宽列一次算完，并且会先过 processor 的变换）。保留本函数
    主要用于对照阅读“不做变换时的最朴素的统计做法”。

    与 action 的唯一结构差异：state 没有“动作块”这一维，所以 stepwise_* 只是
    为了与 action 的输出契约对齐而保留的 (1, D) 形状，语义上等于 global_*。

    Args:
        data: 单个 key 的 state 数据，形状 (N, D) 或 (N,)。
        quantile_low: 下分位数（默认 0.01）。
        quantile_high: 上分位数（默认 0.99）。

    Returns:
        统计量字典：stepwise_* 形状 (1, D)，global_* 形状 (D,)。
    """
    if data.ndim == 1:
        # 单维特征补成 (N, 1)，统一按二维处理。
        data = data.unsqueeze(-1)
    data = data.float()

    stats = {}
    # 基础统计量：沿“帧”这一维归约 → (D,)；stepwise 版本额外 unsqueeze(0) 保证是 (1, D)。
    stats["stepwise_min"] = data.amin(0).unsqueeze(0)
    stats["stepwise_max"] = data.amax(0).unsqueeze(0)
    stats["global_min"] = data.amin(0)
    stats["global_max"] = data.amax(0)
    stats["stepwise_mean"] = data.mean(0).unsqueeze(0)
    stats["stepwise_std"] = data.std(0).unsqueeze(0)
    stats["global_mean"] = data.mean(0)
    stats["global_std"] = data.std(0)

    data_np = data.numpy()
    N = data_np.shape[0]

    # 8 组分位数：q01/q99 常用稳健范围；其余是逐级放宽的兜底范围。
    quantile_spec = [
        ("q01", quantile_low),
        ("q99", quantile_high),
        ("q001", 0.001),
        ("q999", 0.999),
        ("q0001", 0.0001),
        ("q9999", 0.9999),
        ("q00001", 0.00001),
        ("q99999", 0.99999),
    ]

    # 用 np.partition（O(N) 的“取第 k 小”）替代 np.quantile（O(N log N) 的完整排序）。
    # 同样把 k 夹到 [0, N-1]，避免小数据集上极端分位数算出越界下标。
    ks = [max(0, min(int(q_val * (N - 1)), N - 1)) for _, q_val in quantile_spec]
    partitioned = np.partition(data_np, ks, axis=0)
    for (q_name, _), k in zip(quantile_spec, ks):
        # copy()：partitioned 是共享内存的视图，转 torch 前先拷贝一份再交给 torch。
        q_result = partitioned[k].copy()
        # state 的 stepwise 与 global 数值相同，只是形状不同（(1, D) 与 (D,)）。
        stats[f"stepwise_{q_name}"] = torch.from_numpy(q_result).unsqueeze(0).float()
        stats[f"global_{q_name}"] = torch.from_numpy(q_result).float()

    # 显式释放 numpy 中间张量（大 key 上这些缓冲可能是几百 MB）。
    del data_np, partitioned
    return stats


def _get_quantile_spec(quantile_low: float = 0.01, quantile_high: float = 0.99):
    """返回标准的 8 组分位数定义：(名字, 分位值)。

    为什么固定 8 组：下游不同归一化/裁剪策略需要的“覆盖范围”不同——
      q01/q99        常用稳健范围（裁掉 1% 离群点，z-score / min-max 都用它）
      q001/q999      放宽一档
      q0001/q9999    再放宽
      q00001/q99999  最宽（相对动作变换的裁剪区间等场景）
    名字直接当统计量字典的 key（如 "stepwise_q01"），所以这里用字符串而不是数值命名。
    """
    return [
        ("q01", quantile_low),
        ("q99", quantile_high),
        ("q001", 0.001),
        ("q999", 0.999),
        ("q0001", 0.0001),
        ("q9999", 0.9999),
        ("q00001", 0.00001),
        ("q99999", 0.99999),
    ]


def _merge_dict_to_tensor(
    data_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List[str], List[int]]:
    """把 {key: (N, D_key)} 沿特征维拼成一个 (N, D_total) 的大张量。

    为什么要拼：统计的开销主要与“列数”相关（每列都要 mean/std/分位数）。
    把 left_arm(7) + left_gripper(1) + torso(6)… 拼成一条宽列后，
    amin/amax/mean/std 和 np.partition 都能一次算完，然后按 dims 切回各自的 key
    （见 _split_stats_by_key）。这会显著减少 python 层循环次数。

    注意 key 的顺序会被记录下来：merged 的列区间 [offset, offset+dims[i]) 属于 keys[i]，
    切回时必须用同一份 keys/dims，所以这里把三者一起返回。

    Args:
        data_dict: {key: 形状 (N, D_key) 或 (N,) 的张量}。

    Returns:
        merged: (N, D_total) 的 float32 张量。
        keys: 列区间的顺序（= list(data_dict.keys())）。
        dims: 每个 key 的宽度（与 keys 一一对应）。
    """
    keys = list(data_dict.keys())
    dims = []
    tensors = []
    for k in keys:
        # 统一成 float32 + 二维，保证 cat 时维度一致（单维 key 补成 (N, 1)）。
        t = data_dict[k].float()
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        dims.append(t.shape[-1])
        tensors.append(t)
    # dim=-1：沿特征维拼接（各 key 的帧数 N 必须相同，因为它们来自同一批抽样行）。
    merged = torch.cat(tensors, dim=-1)
    return merged, keys, dims


def _split_stats_by_key(
    keys: List[str],
    dims: List[int],
    stepwise_min: torch.Tensor,
    stepwise_max: torch.Tensor,
    stepwise_mean: torch.Tensor,
    stepwise_std: torch.Tensor,
    stepwise_quantiles: torch.Tensor,
    quantile_spec: List[Tuple[str, float]],
    is_action: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """把“合并张量算出来的统计量”按列区间切回每个 key 的统计量字典。

    与 _merge_dict_to_tensor 互为逆操作：merged 的第 i 段列 [offset, offset+dims[i])
    对应 keys[i]，因此这里用 slice 切出该 key 的所有统计量。

    形状约定：
        stepwise_min/max/mean/std 传入时形状是 (H, D_total)（H = action_size）；
        stepwise_quantiles 是 (num_q, H, D_total)；
        返回的每个 key 里 stepwise_* 为 (H, dim_key)，global_* 为 (dim_key,)。

    Args:
        keys / dims: 与 _merge_dict_to_tensor 的输出一致（列区间的顺序与宽度）。
        stepwise_min / max / mean / std: 合并后的逐 step 统计量，形状 (H, D_total)。
        stepwise_quantiles: 合并后的逐 step 分位数，形状 (num_q, H, D_total)。
        quantile_spec: 分位数定义列表（决定返回字典里的 key 名）。
        is_action: True = action（stepwise 保留 H 行）；False = state
            （state 没有动作步这一维，stepwise_* 只保留第 0 行 → (1, dim)）。

    Returns:
        {key: {统计量名: 张量}}。
    """
    result = {}
    offset = 0
    for key, dim in zip(keys, dims):
        # 本 key 在合并张量里占据的列区间。
        s = slice(offset, offset + dim)
        sw_mean = stepwise_mean[:, s]
        sw_std = stepwise_std[:, s]
        # global_std 用总方差分解计算：全局方差 = 各步方差的平均 + 各步均值的方差。
        # 只对各步的 std 求平均会漏掉“不同动作步均值不同”带来的波动。
        # 特例：只有 1 个动作步（state）时 var(0) 无定义（样本数为 1），直接取该步的 std。
        if sw_mean.shape[0] > 1:
            global_std = (sw_std.pow(2).mean(0) + sw_mean.var(0)).sqrt()
        else:
            global_std = sw_std.squeeze(0)
        ks_dict = {
            "stepwise_min": stepwise_min[:, s],
            "stepwise_max": stepwise_max[:, s],
            "stepwise_mean": sw_mean,
            "stepwise_std": sw_std,
            # 全局量：再沿“动作步”维归约一次（min 取最小、max 取最大 → 覆盖全部取值）
            "global_min": stepwise_min[:, s].amin(0),
            "global_max": stepwise_max[:, s].amax(0),
            "global_mean": sw_mean.mean(0),
            "global_std": global_std,
        }
        # 分位数：stepwise_quantiles 的第 0 维是“第几组分位数”，所以用 qi 取出该组。
        for qi, (qname, _) in enumerate(quantile_spec):
            ks_dict[f"stepwise_{qname}"] = stepwise_quantiles[qi, :, s]
        # 全局分位数按“保守合并”：下分位取各步最小值、上分位取各步最大值。
        for qname in ["q01", "q001", "q0001", "q00001"]:
            ks_dict[f"global_{qname}"] = ks_dict[f"stepwise_{qname}"].amin(0)
        for qname in ["q99", "q999", "q9999", "q99999"]:
            ks_dict[f"global_{qname}"] = ks_dict[f"stepwise_{qname}"].amax(0)

        if not is_action:
            # state：stepwise_* 只保留第 0 行，形状从 (H, dim) 变成 (1, dim)，
            # 与父类（v2）产出的 state 统计量形状保持一致，下游不需要区分来源。
            # （state 的合并张量里 H 其实就是 1，这里显式截断只是把契约写死。）
            for stat_name in list(ks_dict.keys()):
                if stat_name.startswith("stepwise_"):
                    ks_dict[stat_name] = ks_dict[stat_name][:1]  # 只留第一行 → (1, dim)

        result[key] = ks_dict
        # 移动到下一个 key 的列区间
        offset += dim
    return result


def compute_action_stats_merged(
    action_dict: Dict[str, torch.Tensor],
    ep_indices: torch.Tensor,
    action_size: int,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    frame_ep_end: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    把全部 action key 合并成一条宽列，一次算完统计量（“无变换”时的快速路径）。

    流程：所有 key 沿特征维拼成 (N, D_total) → 逐动作步取列（带 episode 夹紧）
    → min/max/mean/std + 8 组分位数 → 再按 dims 切回每个 key。
    分位数实现按环境自动选择：有 GPU 就用 torch.sort 按 k 取行（不落回 CPU），
    否则用 CPU 上的 np.partition。

    内存提示：GPU 峰值约为 3 × N × D_total × 4 字节（merged 副本 + 当前列 + 排序副本）。
    算完立刻 del + empty_cache 把显存还给系统；万一显存不够（OutOfMemoryError），
    会落回 CPU 路径重算——注意 CPU 上的 merged 一直留着，所以回退是安全的。

    注意：本函数不应用 processor 变换。需要“先变换再统计”（相对动作等）时走
    compute_action_stats_with_transforms，它会在无变换时才调用本函数。

    Args:
        action_dict: {key: (N, D_key)} 的 action 数据。
        ep_indices: 每帧所属 episode 下标，形状 (N,)（仅用于算 frame_ep_end）。
        action_size: 滑窗长度 H。
        quantile_low/high: 分位数的下/上界（默认 0.01 / 0.99）。
        frame_ep_end: 预先算好的“每帧所属 episode 末帧”；None 时内部现算。

    Returns:
        {key: {统计量名: 张量}}，形状约定见 _split_stats_by_key。
    """
    if not action_dict:
        # 没有 action key（例如只统计 state 的调用）→ 直接返回空字典。
        return {}

    merged, keys, dims = _merge_dict_to_tensor(action_dict)
    N, D_total = merged.shape

    if frame_ep_end is None:
        # 滑窗夹紧需要的“每帧所属 episode 末帧”。
        frame_ep_end = _compute_frame_ep_end(ep_indices, N)

    quantile_spec = _get_quantile_spec(quantile_low, quantile_high)
    num_q = len(quantile_spec)
    # 分位数 → “第 k 小”的下标，一次 np.partition / sort 全部拿回。
    ks = [max(0, min(int(qv * (N - 1)), N - 1)) for _, qv in quantile_spec]

    # 有 CUDA 就优先用 GPU（排序快很多）；computed 标记是否已成功算完，
    # 用于在显存不足时安全回退到 CPU 分支。
    use_gpu = torch.cuda.is_available()
    computed = False

    if use_gpu:
        try:
            device = torch.device("cuda")
            # 把合并张量与 episode 末帧表搬到 GPU，后续全部在显存里完成。
            merged_dev = merged.to(device)
            fep_dev = frame_ep_end.to(device)
            base_idx = torch.arange(N, dtype=torch.long, device=device)
            N_limit = torch.tensor(N - 1, dtype=torch.long, device=device)

            # 输出缓冲（在 GPU 上）：4 个基础统计量 + 分位数立方体。
            sw_min = torch.empty(action_size, D_total, device=device)
            sw_max = torch.empty(action_size, D_total, device=device)
            sw_mean = torch.empty(action_size, D_total, device=device)
            sw_std = torch.empty(action_size, D_total, device=device)
            sw_q = torch.empty(num_q, action_size, D_total, device=device)

            for j in range(action_size):
                # 第 j 步：取“每帧往后第 j 帧”，并做两道夹紧（episode 末帧 / 数据集末帧）。
                tidx = torch.minimum(base_idx + j, fep_dev)
                tidx = torch.minimum(tidx, N_limit)
                col = merged_dev[tidx]

                sw_min[j] = col.amin(0)
                sw_max[j] = col.amax(0)
                sw_mean[j] = col.mean(0)
                sw_std[j] = col.std(0)

                # GPU 上取分位数：整列排序后按 k 取行（O(N log N) 但常数极小；
                # GPU 上通常仍比 CPU 的 partition 快，而且避免了 H×num_q 次单独计算）。
                sorted_col, _ = col.sort(dim=0)
                for qi in range(num_q):
                    sw_q[qi, j] = sorted_col[ks[qi]]
                # 释放本步的临时张量，避免显存随动作步数线性增长。
                del col, sorted_col

            # 一次性搬回 CPU（只同步一次，比逐步 .cpu() 快很多）
            stepwise_min = sw_min.cpu()
            stepwise_max = sw_max.cpu()
            stepwise_mean = sw_mean.cpu()
            stepwise_std = sw_std.cpu()
            stepwise_quantiles = sw_q.cpu()

            # 显式释放显存：统计阶段可能与训练共用一张卡，早点还回去避免影响后续。
            del merged, merged_dev, fep_dev, base_idx, N_limit
            del sw_min, sw_max, sw_mean, sw_std, sw_q
            torch.cuda.empty_cache()
            computed = True
        except torch.cuda.OutOfMemoryError:
            # 显存不够：CPU 上的 merged 一直没被释放，直接走下面的 CPU 分支重算。
            torch.cuda.empty_cache()

    if not computed:
        # CPU 回退路径：np.partition（O(N) 取第 k 小），逻辑与 GPU 分支一一对应。
        if not merged.is_contiguous():
            # np.partition 需要连续内存（否则 numpy 会先隐式拷贝，反而更慢）。
            merged = merged.contiguous()
        base_idx = torch.arange(N, dtype=torch.long)
        N_limit = torch.tensor(N - 1, dtype=torch.long)

        stepwise_min = torch.empty(action_size, D_total)
        stepwise_max = torch.empty(action_size, D_total)
        stepwise_mean = torch.empty(action_size, D_total)
        stepwise_std = torch.empty(action_size, D_total)
        stepwise_quantiles = torch.empty(num_q, action_size, D_total)

        for j in range(action_size):
            # 与 GPU 分支相同的取列 + 夹紧逻辑（只是设备是 CPU）。
            tidx = torch.minimum(base_idx + j, frame_ep_end)
            tidx = torch.minimum(tidx, N_limit)
            col = merged[tidx]

            stepwise_min[j] = col.amin(0)
            stepwise_max[j] = col.amax(0)
            stepwise_mean[j] = col.mean(0)
            stepwise_std[j] = col.std(0)

            # 一次 partition 传入全部 k，拿回所有 8 组分位数。
            col_np = col.numpy()
            partitioned = np.partition(col_np, ks, axis=0)
            for qi in range(num_q):
                # copy()：partitioned 是共享内存视图，转 torch 前先拷贝。
                stepwise_quantiles[qi, j] = torch.from_numpy(partitioned[ks[qi]].copy()).float()
            # 循环内及时释放，峰值只有 1×N×D_total。
            del col, col_np, partitioned
        del merged

    return _split_stats_by_key(
        keys,
        dims,
        stepwise_min,
        stepwise_max,
        stepwise_mean,
        stepwise_std,
        stepwise_quantiles,
        quantile_spec,
        is_action=True,
    )


def compute_action_stats_with_transforms(
    action_dict: Dict[str, torch.Tensor],
    state_dict: Dict[str, torch.Tensor],
    ep_indices: torch.Tensor,
    action_size: int,
    transforms: Optional[list] = None,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    frame_ep_end: Optional[torch.Tensor] = None,
    quantile_method: Literal["numpy_partition", "torch_quantile", "torch_sort"] = "numpy_partition",
    downsample_rate: int = 1,
    show_progress: bool = False,
    action_step_indices: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    计算 action 统计量，并保证“每个滑窗列都正确地应用了 processor 变换”。

    为什么不能先变换再统计（本函数存在的理由）：
        相对动作变换（RelativeJointTransform / RelativePoseTransform）会用
        ``state[..., -1:, :]`` 取“基座状态”来算差值。如果直接把扁平 (N, D) 的
        state 丢进去，``[..., -1:, :]`` 取到的会是**整个数据集最后一帧**的 state
        ——完全错误的基座。正确做法是先 unsqueeze 成 (N, 1, D)，
        这样 ``[..., -1:, :]`` 正好是“该帧自己的 state”（每个样本一个基座）。

    处理顺序（每个动作步 j）：
        取列（含 episode 夹紧）→ 组 batch {"action": (N,1,D), "state": (N,1,D)}
        → 依次过 transforms → 取输出的最后一帧 (N, dim) → 算 min/max/mean/std/分位数。

    快速路径：transforms 为空（None 或 []）且没有预定位下标时，直接交给
    compute_action_stats_merged()（省掉逐列组装 batch 的开销）。

    Args:
        action_dict: {key: (N, D_key)} 的原始 action 数据。
        state_dict: {key: (N, D_key)} 的原始 state 数据（相对动作变换的基座）。
        ep_indices: 每帧所属 episode 下标，形状 (N,)。
        action_size: 滑窗长度 H。
        transforms: 变换对象列表（RelativeJointTransform / RelativePoseTransform /
            PoseRotationTransform 等）。None 或空列表 = 不做变换。
        quantile_low/high: 分位数的下/上界。
        frame_ep_end: 预先算好的“每帧所属 episode 末帧”；None 时内部现算。
        quantile_method: 分位数的计算方式，三选一：
            "numpy_partition"（默认，CPU 上 O(N) 最快）、
            "torch_quantile"（实现最直观，最慢）、
            "torch_sort"（GPU 上最合适）。
        downsample_rate: 统计前对“帧”再做一次等间隔抽样（>1 时每 rate 帧取一帧）。
            主统计路径传 1，因为上游已经按 stats_downsample_rate 抽过锚点了。
        show_progress: 是否显示逐动作步的进度条（与日志二选一）。
        action_step_indices: {key: [第 0..H-1 步各自的帧号]} —— 上游按“锚点 + episode
            夹紧”预先算好的取帧计划。给了它就不再自己算滑窗（这是主路径的用法）。

    Returns:
        {key: {统计量名: 张量}}；key 与维度可能和输入不同——
        若某个变换改变了维度（如 PoseRotationTransform 7→9），输出的 key/dim 以
        变换后的结果为准。
    """
    transforms = transforms or []

    # 快速路径：没有变换、也没有“预定位的动作步下标”
    if not transforms and action_step_indices is None:
        return compute_action_stats_merged(
            action_dict,
            ep_indices,
            action_size,
            quantile_low,
            quantile_high,
            frame_ep_end,
        )

    # 帧数 N 的取法：优先用“动作步下标”张量的长度（它是抽样后的锚点数），
    # 否则用 state / action 数据的第 0 维长度。
    if action_step_indices is not None:
        first_key = next(iter(action_step_indices))
        N = action_step_indices[first_key][0].shape[0]
    elif state_dict:
        N = next(iter(state_dict.values())).shape[0]
    else:
        N = next(iter(action_dict.values())).shape[0]

    # 只有在走“滑窗 + 夹紧”那条路时才需要 episode 末帧表；
    # 走 action_step_indices 时，帧号早就由上游算好并夹紧过了。
    if frame_ep_end is None and action_step_indices is None:
        frame_ep_end = _compute_frame_ep_end(ep_indices, N)

    # ── 试算一次，确定“变换之后的输出 key 与维度” ──
    # 为什么要试算：相对动作变换等会改变数值甚至维度（例如 PoseRotationTransform 7→9），
    # 输出缓冲的大小必须在循环前就知道。这里只取 1 帧、1 个动作步走一遍变换，
    # 拿它的输出形状作为最终统计量的形状（成本可以忽略）。
    if action_step_indices is None:
        trial_action = {k: v[0:1].unsqueeze(1).float() for k, v in action_dict.items()}  # (1,1,D)
    else:
        trial_action = {
            k: v[action_step_indices[k][0][:1]].unsqueeze(1).float() for k, v in action_dict.items()
        }
    trial_state = {k: v[0:1].unsqueeze(1).float() for k, v in state_dict.items()}  # (1,1,D)
    trial_batch = {"action": trial_action, "state": trial_state}
    for trans in transforms:
        trial_batch = trans.forward(trial_batch)
    out_keys = list(trial_batch["action"].keys())
    out_dims = [trial_batch["action"][k].shape[-1] for k in out_keys]
    D_total = sum(out_dims)
    del trial_action, trial_state, trial_batch

    # ── 把 state 统一成 float32，供每个动作步复用 ──
    # 相对动作变换要从 state 里取“基座状态”，所以每个动作步都要用到 state；
    # 先转一次 float 避免在循环里反复转换。
    state_float = {k: v.float() for k, v in state_dict.items()}

    quantile_spec = _get_quantile_spec(quantile_low, quantile_high)
    num_q = len(quantile_spec)

    # 有 CUDA 就优先用 GPU；computed 用于在显存不足时安全回退到 CPU 分支。
    use_gpu = torch.cuda.is_available()
    computed = False

    if use_gpu:
        try:
            device = torch.device("cuda")
            # state 只搬一次显存：每个动作步的变换都要用它当基座，反复搬运很浪费。
            state_gpu = {k: v.to(device) for k, v in state_float.items()}
            # action 也只搬一次：后面按帧号在显存里做花式索引，不再回 CPU。
            action_gpu = {k: v.to(device).float() for k, v in action_dict.items()}
            if action_step_indices is None:
                # 自己算滑窗时要用的三件套：episode 末帧表 + 帧下标 + 数据集末帧（兜底夹紧）。
                fep_dev = frame_ep_end.to(device)
                base_idx = torch.arange(N, dtype=torch.long, device=device)
                N_limit = torch.tensor(N - 1, dtype=torch.long, device=device)
            else:
                # 主路径：帧号计划已经由上游算好并夹紧，这里只把它搬到显存直接用。
                fep_dev = None
                base_idx = None
                N_limit = None
                action_step_indices_dev = {
                    key: [idx.to(device) for idx in step_indices]
                    for key, step_indices in action_step_indices.items()
                }

            # 输出缓冲（显存）：4 个基础统计量 + 分位数立方体 (num_q, H, D_total)。
            sw_min = torch.empty(action_size, D_total, device=device)
            sw_max = torch.empty(action_size, D_total, device=device)
            sw_mean = torch.empty(action_size, D_total, device=device)
            sw_std = torch.empty(action_size, D_total, device=device)
            sw_q = torch.empty(num_q, action_size, D_total, device=device)

            # 进度条只在 show_progress 时显示（否则改用 logger.info 打印每步耗时）。
            step_iter = tqdm(
                range(action_size),
                desc="🎬 Action stats (GPU)",
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
            for j in step_iter:
                if show_progress:
                    step_iter.set_postfix_str(f"step={j + 1}/{action_size}", refresh=False)

                # 组装第 j 步的“动作块前缀”：每个样本要取它后续 0..j 步的动作。
                if action_step_indices is None:
                    # 自算滑窗：用广播一次性构造 (N, j+1) 的帧号矩阵，再两重夹紧。
                    # all_tidx[i, s] = 第 i 个样本往后第 s 步的帧号（越界则停在边界帧）。
                    steps = torch.arange(j + 1, device=device)
                    all_tidx = base_idx.unsqueeze(0) + steps.unsqueeze(1)
                    all_tidx = torch.minimum(all_tidx, fep_dev.unsqueeze(0))
                    all_tidx = torch.minimum(all_tidx, N_limit)
                    # permute(1, 0, 2)：把 (N, j+1, D) 变成 (j+1, N, D)，
                    # 与下面 action_step_indices 分支的 stack(dim=1) 形状保持一致。
                    action_col_3d = {k: v[all_tidx].permute(1, 0, 2) for k, v in action_gpu.items()}
                else:
                    # 主路径：直接按上游给的帧号 gather，再沿第 1 维（时间）堆成 (j+1, N, D)。
                    action_col_3d = {
                        k: torch.stack(
                            [v[action_step_indices_dev[k][step]] for step in range(j + 1)], dim=1
                        )
                        for k, v in action_gpu.items()
                    }
                # state 升一维成 (N, 1, D_k)：这样变换里的 [..., -1:, :] 取到的就是
                # “该帧自己的 state”，而不是整个数据集的最后一帧（见函数文档字符串）。
                state_3d = {k: v.unsqueeze(1) for k, v in state_gpu.items()}

                # 应用 processor 变换；state 用浅拷贝，避免变换内部原地修改影响后续步骤。
                batch = {"action": action_col_3d, "state": dict(state_3d)}
                action_trans_start = time.perf_counter()
                for trans in transforms:
                    batch = trans.forward(batch)
                if not show_progress:
                    # 没开进度条时用日志报告“本步变换耗时”，便于定位是哪一步慢。
                    logger.info(
                        "🧩 Applied transforms for action step %s/%s in %.2fs",
                        j + 1,
                        action_size,
                        time.perf_counter() - action_trans_start,
                    )

                # 变换后的动作块是 (j+1, N, dim)：统计只需要“最新的一步”，即最后那一帧。
                # 取 [-1, :, :] → (N, dim)，再按 out_keys 顺序拼成一条宽列 (N, D_total)。
                col_parts = [batch["action"][k][:, -1, :] for k in out_keys]
                if downsample_rate > 1:
                    # 额外的等间隔抽样（主路径传 1，不会走到这里）。
                    col_parts = [part[::downsample_rate] for part in col_parts]
                col = torch.cat(col_parts, dim=-1)

                # 基础统计量（沿帧维归约）。
                sw_min[j] = col.amin(0)
                sw_max[j] = col.amax(0)
                sw_mean[j] = col.mean(0)
                sw_std[j] = col.std(0)

                action_before_quantile = time.perf_counter()

                # 抽样后样本数变了，分位点对应的“第 k 小”下标要重新算（否则会取错位置）。
                N_sampled = col.shape[0]
                ks_q_sampled = [
                    max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
                ]

                # 三种分位数实现，结果数值会有极小差异（后两者精确、前者是 O(N) 选择）：
                if quantile_method == "numpy_partition":
                    # 默认方式：搬回 CPU 用 np.partition（O(N) 取第 k 小，大 N 时最快）。
                    col_np = col.cpu().numpy()
                    partitioned = np.partition(col_np, ks_q_sampled, axis=0)
                    for qi in range(num_q):
                        sw_q[qi, j] = torch.from_numpy(partitioned[ks_q_sampled[qi]]).to(device)
                    del col_np, partitioned
                elif quantile_method == "torch_quantile":
                    # 实现最直观（一次算完 8 个分位数），但内部需要排序，最慢。
                    quantile_values = torch.tensor([qv for _, qv in quantile_spec], device=device)
                    sw_q[:, j] = torch.quantile(col, quantile_values, dim=0)
                elif quantile_method == "torch_sort":
                    # GPU 上最合适：整列排一次序，然后按 k 取 8 行。
                    sorted_col, _ = col.sort(dim=0)
                    for qi in range(num_q):
                        sw_q[qi, j] = sorted_col[ks_q_sampled[qi]]
                    del sorted_col
                else:
                    raise ValueError(f"Unknown quantile_method: {quantile_method}")

                if not show_progress:
                    logger.info(
                        "📐 Computed action quantiles (%s) for step %s/%s in %.2fs",
                        quantile_method,
                        j + 1,
                        action_size,
                        time.perf_counter() - action_before_quantile,
                    )
                # 及时释放本步的临时张量：否则显存会随动作步数 j 线性增长。
                del col, action_col_3d, state_3d, batch, col_parts

            # 一次性搬回 CPU（只同步一次；逐项 .cpu() 会触发多次同步，明显更慢）
            stepwise_min = sw_min.cpu()
            stepwise_max = sw_max.cpu()
            stepwise_mean = sw_mean.cpu()
            stepwise_std = sw_std.cpu()
            stepwise_quantiles = sw_q.cpu()

            del action_gpu, state_gpu
            if fep_dev is not None:
                del fep_dev, base_idx, N_limit
            if action_step_indices is not None:
                del action_step_indices_dev
            # 显式释放显存：统计常与训练共用一张卡，早点还回去避免影响后续前向/反向。
            del sw_min, sw_max, sw_mean, sw_std, sw_q
            torch.cuda.empty_cache()
            computed = True
        except torch.cuda.OutOfMemoryError:
            # 显存不足：CPU 上的原始张量都还在，直接落到下面的 CPU 分支重算。
            torch.cuda.empty_cache()

    if not computed:
        # ── CPU 回退路径：与 GPU 分支逐行对应，只是不再有设备搬运 ──
        if action_step_indices is None:
            # 自算滑窗需要帧下标与兜底上界（主路径下这两个都不需要）。
            base_idx = torch.arange(N, dtype=torch.long)
            N_limit = torch.tensor(N - 1, dtype=torch.long)
        else:
            base_idx = None
            N_limit = None

        # 输出缓冲（CPU）。
        stepwise_min = torch.empty(action_size, D_total)
        stepwise_max = torch.empty(action_size, D_total)
        stepwise_mean = torch.empty(action_size, D_total)
        stepwise_std = torch.empty(action_size, D_total)
        stepwise_quantiles = torch.empty(num_q, action_size, D_total)

        # 变换要求浮点输入，先统一转一次 float（避免在循环里反复转换）。
        action_float = {k: v.float() for k, v in action_dict.items()}

        step_iter = tqdm(
            range(action_size),
            desc="🎬 Action stats (CPU)",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for j in step_iter:
            if show_progress:
                step_iter.set_postfix_str(f"step={j + 1}/{action_size}", refresh=False)
            # 与 GPU 分支完全相同的取列逻辑：自算滑窗（夹紧）或按预定位帧号 gather。
            if action_step_indices is None:
                steps = torch.arange(j + 1)
                all_tidx = base_idx.unsqueeze(0) + steps.unsqueeze(1)
                all_tidx = torch.minimum(all_tidx, frame_ep_end.unsqueeze(0))
                all_tidx = torch.minimum(all_tidx, N_limit)
                action_col_3d = {k: v[all_tidx].permute(1, 0, 2) for k, v in action_float.items()}
            else:
                action_col_3d = {
                    k: torch.stack(
                        [v[action_step_indices[k][step]] for step in range(j + 1)], dim=1
                    )
                    for k, v in action_float.items()
                }
            # state 升维成 (N, 1, D_k)：保证变换里的 [..., -1:, :] 取到的是“该帧自己”。
            state_3d = {k: v.unsqueeze(1) for k, v in state_float.items()}

            # 过变换（浅拷贝 state 字典，避免变换内部原地改到 state_float）。
            batch = {"action": action_col_3d, "state": dict(state_3d)}
            for trans in transforms:
                batch = trans.forward(batch)

            # 取变换后动作块的“最后一帧”，按 out_keys 拼成一条宽列 (N, D_total)。
            col_parts = [batch["action"][k][:, -1, :] for k in out_keys]
            if downsample_rate > 1:
                col_parts = [part[::downsample_rate] for part in col_parts]
            col = torch.cat(col_parts, dim=-1)

            # 基础统计量（沿帧维归约）。
            stepwise_min[j] = col.amin(0)
            stepwise_max[j] = col.amax(0)
            stepwise_mean[j] = col.mean(0)
            stepwise_std[j] = col.std(0)

            # 抽样后样本数变了，分位点对应的“第 k 小”下标要重算。
            N_sampled = col.shape[0]
            ks_q_sampled = [
                max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
            ]

            # 三种分位数实现（与 GPU 分支一致，只是都在 CPU 上做）：
            if quantile_method == "numpy_partition":
                # 默认方式：np.partition 一次取回全部 8 个分位点（O(N)）。
                col_np = col.numpy()
                partitioned = np.partition(col_np, ks_q_sampled, axis=0)
                for qi in range(num_q):
                    # copy()：partitioned 是共享内存视图，交给 torch 前先拷贝。
                    stepwise_quantiles[qi, j] = torch.from_numpy(
                        partitioned[ks_q_sampled[qi]].copy()
                    ).float()
                del col_np, partitioned
            elif quantile_method == "torch_quantile":
                # 最直观但最慢（内部要排序）。
                quantile_values = torch.tensor([qv for _, qv in quantile_spec])
                stepwise_quantiles[:, j] = torch.quantile(col, quantile_values, dim=0)
            elif quantile_method == "torch_sort":
                # 排一次序后按 k 取 8 行。
                sorted_col, _ = col.sort(dim=0)
                for qi in range(num_q):
                    stepwise_quantiles[qi, j] = sorted_col[ks_q_sampled[qi]]
                del sorted_col
            else:
                raise ValueError(f"Unknown quantile_method: {quantile_method}")

            # 及时释放本步的中间张量，让峰值内存保持在 1×N×D_total 级别。
            del col, action_col_3d, state_3d, batch, col_parts

    # 把合并张量的统计量按 out_keys/out_dims 切回每个 key（is_action=True：保留 H 行）。
    return _split_stats_by_key(
        out_keys,
        out_dims,
        stepwise_min,
        stepwise_max,
        stepwise_mean,
        stepwise_std,
        stepwise_quantiles,
        quantile_spec,
        is_action=True,
    )


def compute_state_stats_merged(
    state_dict: Dict[str, torch.Tensor],
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    quantile_method: Literal["numpy_partition", "torch_quantile", "torch_sort"] = "numpy_partition",
    downsample_rate: int = 10,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    把全部 state key 合并成一条宽列，一次算完统计量（“无变换”时的快速路径）。

    与 action 的差别：state 没有“动作块”这个概念，不需要滑窗、也不需要夹到 episode
    末帧（只统计“这一帧自己的观测值”）。所以这里就是：合并 → 降采样 → 一次归约 +
    分位数 → 切回各 key。

    Args:
        state_dict: {key: (N, D_key)} 的 state 数据。
        quantile_low/high: 分位数的下/上界。
        quantile_method: "numpy_partition" / "torch_quantile" / "torch_sort"，
            语义与 compute_action_stats_with_transforms 里的同名参数一致。
        downsample_rate: 统计前对“帧”再做一次等间隔抽样（每 rate 帧取一帧）。
            默认 10（state 的帧间高度相关，抽 1/10 对分布估计几乎没有影响，但快 10 倍）；
            主统计路径会显式传 1，因为上游已经抽过锚点了。

    Returns:
        {key: {统计量名: 张量}}；state 的 stepwise_* 形状是 (1, dim)。
    """
    if not state_dict:
        # 没有 state key → 返回空字典。
        return {}

    merged, keys, dims = _merge_dict_to_tensor(state_dict)
    N, D_total = merged.shape

    quantile_spec = _get_quantile_spec(quantile_low, quantile_high)

    # 有 CUDA 就优先用 GPU；computed 用于在显存不足时安全回退到 CPU 分支。
    use_gpu = torch.cuda.is_available()
    computed = False

    if use_gpu:
        try:
            device = torch.device("cuda")
            merged_dev = merged.to(device)

            # 等间隔降采样（主路径传 1 时不生效）。注意是“抽帧”，不是“抽维度”。
            if downsample_rate > 1:
                merged_dev = merged_dev[::downsample_rate]

            # 基础统计量：沿帧维归约，得到 (D_total,)。
            s_min = merged_dev.amin(0)
            s_max = merged_dev.amax(0)
            s_mean = merged_dev.mean(0)
            s_std = merged_dev.std(0)

            # 抽样后样本数变了，分位点对应的“第 k 小”下标必须重算。
            N_sampled = merged_dev.shape[0]
            ks_sampled = [
                max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
            ]

            # 三种分位数实现（与 action 侧一致）：
            if quantile_method == "numpy_partition":
                # 默认方式：搬回 CPU 用 np.partition，一次取回 8 个分位点再搬回显存。
                merged_np = merged_dev.cpu().numpy()
                partitioned = np.partition(merged_np, ks_sampled, axis=0)
                sw_q = torch.stack([torch.from_numpy(partitioned[k]) for k in ks_sampled]).to(
                    device
                )
                del merged_np, partitioned, merged_dev
            elif quantile_method == "torch_quantile":
                # 最直观但最慢（内部需要排序）。
                quantile_values = torch.tensor([qv for _, qv in quantile_spec], device=device)
                sw_q = torch.quantile(merged_dev, quantile_values, dim=0)
                del merged_dev
            elif quantile_method == "torch_sort":
                # GPU 上最合适：排一次序后按 k 取行。
                sorted_dev, _ = merged_dev.sort(dim=0)
                sw_q = torch.stack([sorted_dev[k] for k in ks_sampled])
                del merged_dev, sorted_dev
            else:
                raise ValueError(f"Unknown quantile_method: {quantile_method}")

            # 搬回 CPU，并把基础统计量补上“动作步”那一维（state 恒为 1 行）。
            stepwise_min = s_min.cpu().unsqueeze(0)
            stepwise_max = s_max.cpu().unsqueeze(0)
            stepwise_mean = s_mean.cpu().unsqueeze(0)
            stepwise_std = s_std.cpu().unsqueeze(0)
            stepwise_quantiles = sw_q.cpu().unsqueeze(1)  # (num_q, 1, D_total)

            # 显式释放显存（统计常与训练共用一张卡）。
            del merged, s_min, s_max, s_mean, s_std, sw_q
            torch.cuda.empty_cache()
            computed = True
        except torch.cuda.OutOfMemoryError:
            # 显存不足：CPU 上的 merged 一直没释放，走下面的 CPU 分支重算。
            torch.cuda.empty_cache()

    if not computed:
        # ── CPU 回退路径 ──
        # 等间隔降采样（与 GPU 分支同一语义）。
        if downsample_rate > 1:
            merged = merged[::downsample_rate]

        # 基础统计量 + 补一维（state 的 stepwise 恒为 (1, D_total)）。
        stepwise_min = merged.amin(0).unsqueeze(0)
        stepwise_max = merged.amax(0).unsqueeze(0)
        stepwise_mean = merged.mean(0).unsqueeze(0)
        stepwise_std = merged.std(0).unsqueeze(0)

        # 抽样后样本数变了，分位点下标要重算。
        N_sampled = merged.shape[0]
        ks_sampled = [
            max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
        ]

        # 三种分位数实现（都在 CPU 上），最后统一整理成 (num_q, 1, D_total)。
        if quantile_method == "numpy_partition":
            # 默认方式：np.partition 一次拿回 8 个分位点。
            merged_np = merged.numpy()
            partitioned = np.partition(merged_np, ks_sampled, axis=0)
            # copy()：partitioned 是共享内存视图，转 torch 前先拷贝。
            q_rows = [torch.from_numpy(partitioned[k].copy()).float() for k in ks_sampled]
            stepwise_quantiles = torch.stack(q_rows).unsqueeze(1)  # (num_q, 1, D_total)
            del merged_np, partitioned
        elif quantile_method == "torch_quantile":
            # 最直观但最慢。
            quantile_values = torch.tensor([qv for _, qv in quantile_spec])
            sw_q = torch.quantile(merged, quantile_values, dim=0)
            stepwise_quantiles = sw_q.unsqueeze(1)
        elif quantile_method == "torch_sort":
            # 排一次序后按 k 取行。
            sorted_merged, _ = merged.sort(dim=0)
            q_rows = [sorted_merged[k] for k in ks_sampled]
            stepwise_quantiles = torch.stack(q_rows).unsqueeze(1)
            del sorted_merged
        else:
            raise ValueError(f"Unknown quantile_method: {quantile_method}")

        del merged

    # 按 keys/dims 把合并列切回每个 key 的统计量（is_action=False：stepwise 只留 1 行）。
    return _split_stats_by_key(
        keys,
        dims,
        stepwise_min,
        stepwise_max,
        stepwise_mean,
        stepwise_std,
        stepwise_quantiles,
        quantile_spec,
        is_action=False,
    )


def compute_state_stats_with_transforms(
    state_dict: Dict[str, torch.Tensor],
    transforms: Optional[list] = None,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    quantile_method: Literal["numpy_partition", "torch_quantile", "torch_sort"] = "numpy_partition",
    downsample_rate: int = 10,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    先对 state 应用 processor 变换，再计算统计量（主路径用的 state 统计入口）。

    为什么必须先变换再统计：这样得到的才是“模型实际看到的数值分布”。
    与旧路径保持一致——旧的统计实现是先跑 preprocessor.action_state_transform，
    再对 state/action 一起收统计量。

    两种变换在本函数里的行为：
      · 只作用于 action 的变换（RelativeJointTransform / RelativePoseTransform）：
        batch 里只有 "state" 时它们退化成空操作，state 数值与维度都不变；
      · 会改 state 的变换（例如作用在 ee_pose 上的 PoseRotationTransform）：
        输出维度可能变化（7→9），因此最终统计量的 key/维度必须以变换后的结果为准，
        这也是内部会重新调 compute_state_stats_merged() 的原因。

    Args:
        state_dict: {key: (N, D_key)} 的原始 state 数据。
        transforms: 变换对象列表；None 或空列表表示不做变换（直接走快速路径）。
        quantile_low/high: 分位数的下/上界。
        quantile_method: 分位数实现方式（透传）。
        downsample_rate: 降采样率（透传；主路径传 1）。

    Returns:
        {key: {统计量名: 张量}}；key 与维度以“变换之后”的结果为准。
    """
    transforms = transforms or []

    if not state_dict:
        # 没有 state key → 无事可做。
        return {}

    if not transforms:
        # 无变换：直接走“合并张量一次算完”的快速路径（省掉组 batch 的开销）。
        return compute_state_stats_merged(
            state_dict,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            quantile_method=quantile_method,
            downsample_rate=downsample_rate,
        )

    # 组 batch 时 state 升一维成 (N, 1, D)：与 action 侧保持一致的形状约定，
    # 使变换内部的 [..., -1:, :] 取到的是“该帧自己的 state”。
    state_batch = {"state": {k: v.unsqueeze(1).float() for k, v in state_dict.items()}}
    for trans in transforms:
        state_batch = trans.forward(state_batch)

    # 变换完成后再压回 (N, D)，交给合并路径统计（此时维度可能是变换后的新维度）。
    transformed_state_dict = {k: v.squeeze(1) for k, v in state_batch["state"].items()}
    return compute_state_stats_merged(
        transformed_state_dict,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        quantile_method=quantile_method,
        downsample_rate=downsample_rate,
    )


class BaseLerobotDatasetV3(BaseLerobotDataset):
    """
    LeRobot 数据集基类 V3（快速统计量版）。

    它在 BaseLerobotDataset 的基础上只做一件事：把 get_dataset_stats 换成
    “一次扫 parquet、把需要的行读出来、向量化/GPU 算完统计量”的实现
    （_fast_get_dataset_stats）。取数、切分、训练/验证集划分等逻辑全部继承父类。

    本类使用的 state/action meta 布局（与父类一致，只是这里显式写出来）：
      - lerobot_key：原始 parquet 列名
      - start_index / raw_shape：该 key 在原始列里的切片起点与宽度
      - time_offset：时间偏移

    典型用法（Hydra 配置里直接写类路径，见 configs/data/libero.yaml）：
        type: g05.data.base_lerobot_datasetV3.BaseLerobotDatasetV3
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int,
        past_action_size: int = 0,
        obs_size: int = 1,
        obs_stride_second: float = 0.0,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        lerobot_ds_version: Optional[Literal["2.1", "3.0"]] = "3.0",
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        tolerance_s: Optional[float] = None,
        fast_stats_computation: bool = True,
        stats_downsample_rate: int = 1,
        **kwargs,
    ):
        """构造数据集对象：父类参数原样透传，这里只处理“统计量”相关的三个参数。

        Args（只列与父类不同的部分，其余含义见 base_lerobot_dataset.py）：
            quantile_low / quantile_high: 统计量里“下/上分位数”的取值（默认 0.01 / 0.99）。
                其余 6 组分位数固定为 0.001/0.999 … 0.00001/0.99999，不受这两个参数影响。
            fast_stats_computation: 是否启用本文件的快速统计实现。
                False 时保留父类的逐 episode 版本（用于对比排查统计量差异）。
            stats_downsample_rate: 统计时的抽样步长（每多少帧取一个锚点），必须 >= 1。
                1 = 全量（最准也最慢）；数据量大时用 10 通常对分布估计几乎没有影响。
                注意：它只影响“统计量”，不影响训练时 __getitem__ 取到的数据。
        """
        # 父类负责：解析 shape_meta、构造 MultiLeRobotDataset、切分训练/验证集、
        # 生成 delta_timestamps 查询计划等（本文件不重复实现）。
        super().__init__(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            action_size=action_size,
            past_action_size=past_action_size,
            obs_size=obs_size,
            obs_stride_second=obs_stride_second,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            lerobot_ds_version=lerobot_ds_version,
            tolerance_s=tolerance_s,
            **kwargs,
        )
        # 分位数边界与抽样率：只在统计量计算时用到，存下来供 _fast_get_dataset_stats 读取。
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.stats_downsample_rate = int(stats_downsample_rate)
        if self.stats_downsample_rate <= 0:
            # 步长为 0 会让 torch.arange(0, N, 0) 直接报错，这里提前给出清晰的错误信息。
            raise ValueError(
                f"stats_downsample_rate must be a positive integer, got {stats_downsample_rate}"
            )

        if fast_stats_computation:
            # 用实例属性覆盖类方法：之后 ds.get_dataset_stats(...) 直接走到快速实现。
            # （这是“本类只替换统计逻辑、不改动父类其它行为”的关键一行。
            #   同时意味着：若子类又重写了 get_dataset_stats，会被这里覆盖掉，
            #   子类想自定义统计路径时应当把 fast_stats_computation 设为 False。）
            self.get_dataset_stats = self._fast_get_dataset_stats

    def _get_parquet_columns(self, only_keys: Optional[Dict[str, Set[str]]] = None) -> List[str]:
        """收集 state/action meta 引用到的“原始 parquet 列名”，去重后返回。

        用途：统计阶段只需要这些列，pyarrow 扫描时可以按列裁剪（columns=...），
        不读的列完全不会从磁盘解析出来——这是“快”的一半来源。

        Args:
            only_keys: {"state": {...}, "action": {...}}，只收集指定 key 用到的列；
                None = 收集全部 key 用到的列。

        Returns:
            去重后的原始列名列表（保持首次出现的顺序）。
        """
        columns = []
        # 遍历两类 meta：state 与 action；逻辑完全一致，所以共用一段代码。
        for category, metas in (("state", self.state_meta), ("action", self.action_meta)):
            # 本类别允许的 key 集合；only_keys 里没写这一类时取空集合（即不统计这类）。
            allowed_keys = None if only_keys is None else only_keys.get(category, set())
            for meta in metas:
                if allowed_keys is not None and meta["key"] not in allowed_keys:
                    # 调用方明确说“不需要这个 part”→ 它用到的列也不用读。
                    continue
                # 单个 key 可能对应多个原始列（多源 meta，如 pose + gripper 拼成一个 part）。
                col_names = meta["lerobot_key"]
                if not isinstance(col_names, (list, tuple)):
                    col_names = [col_names]
                for col_name in col_names:
                    if col_name not in columns:
                        # 手动去重（而不是用 set）是为了让返回顺序稳定，日志与调试更好读。
                        columns.append(col_name)
        return columns

    def _fast_get_dataset_stats(
        self,
        _preprocessor=None,
        apply_processor: bool = True,
        use_fast_quantile: bool = True,
        num_workers: int = 32,
        only_keys: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        快速统计量的主实现：一次 parquet 扫描 + 合并张量 + GPU/CPU 向量化计算。

        与父类（v2）的旧统计路径相比，差别有两处：
          · 读数阶段：不逐 episode 走取数逻辑，而是按帧号精确取行（pyarrow 批量读，不碰图像）；
          · 计算阶段：不做“逐 episode 统计再跨 episode 聚合”，而是把整个数据集拼成
            一条宽列一次算完（state → compute_state_stats_merged，
            action → compute_action_stats_with_transforms）。
        统计量的最终定义（stepwise/global 两套粒度、8 组分位数、总方差分解）与旧路径一致，
        因此新旧两条路径产出的 dataset_stats.json 可以直接互换。

        Args:
            _preprocessor: 可选的 processor，用来取 action_state_transforms。
                名字带下划线是因为调用方按位置传参（MixtureLerobotDataset 里写的是
                ds.get_dataset_stats(processor, only_keys=...)）。
            apply_processor: 是否应用 processor 的 action_state_transforms，默认 True。
                （相对动作等变换会改变数值范围，关掉它统计的就是“原始动作”的分布。）
            use_fast_quantile: 未使用，保留它是为了和旧 API 兼容（调用方仍按老签名传参）。
            num_workers: 未使用，同上（快速路径是向量化的，不需要线程池）。
            only_keys: 只返回指定 key 的统计量，格式
                       {"action": {"left_arm", "right_arm"}, "state": {"torso"}}。
                       注意内部仍然全量计算（合并张量路径比“只算几个 key”更快），
                       过滤只发生在返回之前。

        Returns:
            {"state": {key: {...统计量...}}, "action": {key: {...统计量...}}}
        """
        # 整个统计流程的总计时，最后会打印总耗时。
        total_start = time.time()
        # 这两个属性由 MixtureLerobotDataset 在调用前 setattr 注入（别名 / 形态），
        # 只为让日志能对上“现在算的是哪个数据源”；单数据集直接用时为 None。
        dataset_label = getattr(self, "_stats_debug_name", None)
        dataset_type = getattr(self, "_stats_debug_type", None)
        if dataset_label is not None:
            if dataset_type is not None:
                _stats_write(
                    f"🪪 [Stats] Dataset label={dataset_label} (type={dataset_type})",
                    "cyan",
                )
            else:
                _stats_write(f"🪪 [Stats] Dataset label={dataset_label}", "cyan")
        _stats_write("🚀 [Stats] Starting online stats computation...", "cyan")
        _stats_write(f"🔧 [Stats] apply_processor={apply_processor}", "cyan")

        # 需要统计的 key 集合：None = 全部 key；否则只算调用方点名的那些 key。
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

        # 把 meta 列表过滤成“真正要参与本次统计”的那些条目。
        state_meta_to_use = [m for m in self.state_meta if m["key"] in state_keys_to_compute]
        action_meta_to_use = [m for m in self.action_meta if m["key"] in action_keys_to_compute]

        # 只收集这些 part 引用到的原始列，后面 pyarrow 按列裁剪时用。
        columns = self._get_parquet_columns(only_keys=only_keys)

        # 收集所有数据目录下的帧级 parquet 文件，并顺手统计规模信息（仅用于日志）。
        # self.multi_dataset 是父类构造的 MultiLeRobotDataset（N 个数据集目录首尾拼接）。
        parquet_files = []
        total_frames = 0
        total_episodes = 0
        for dataset in self.multi_dataset._datasets:
            # 每个数据集目录的帧级数据都在 <root>/data/ 下（v3 会再按 chunk 分子目录，
            # 所以用 rglob 递归找；sorted 保证遍历顺序稳定、日志可复现）。
            data_dir = dataset.root / "data"
            parquet_files.extend(sorted(data_dir.rglob("*.parquet")))
            # num_frames / num_episodes 是底层 LeRobotDataset 的属性（元数据里读出来的），
            # getattr 兜底是为了兼容没有这两个属性的老版本实现。
            total_frames += int(getattr(dataset, "num_frames", 0))
            total_episodes += int(getattr(dataset, "num_episodes", 0))

        # 打印“这次要处理多大规模”：帧数、episode 数、文件数、以及 len(dataset)。
        # len(self) 是训练侧的样本数（已按训练/验证集切分），与 total_frames 不一定相等。
        _stats_write(
            "📊 [Stats] Dataset has {} frames, {} episodes, {} parquet files, len(dataset)={}".format(
                _format_count(total_frames),
                _format_count(total_episodes),
                _format_count(len(parquet_files)),
                _format_count(len(self)),
            ),
            "cyan",
        )
        # 打印抽样后的锚点规模：向上取整的 ceil(total_frames / rate)。
        _stats_write(
            "🎯 [Stats] stats_downsample_rate={} -> base anchors ~{}".format(
                self.stats_downsample_rate,
                _format_count(
                    (total_frames + self.stats_downsample_rate - 1) // self.stats_downsample_rate
                ),
            ),
            "yellow",
        )

        # 从这里开始计时“读数 + 建表”阶段。
        read_start = time.time()

        # 优先从 meta/episodes 读 episode 边界（省内存）。返回 None 表示读不到，
        # 需要退回到“读整列 episode_index”的老路径（见下面的 else 分支）。
        ep_starts, ep_ends, N_from_meta = _get_episode_metadata_from_parquet(parquet_files)

        if ep_starts is not None and N_from_meta > 0:
            # ── 路径 A（推荐）：用 episode 元数据算边界，不把 episode_index 整列读进来 ──
            _stats_write(
                "📥 [Stats] Using episode metadata for boundary computation (memory efficient)",
                "blue",
            )
            N = N_from_meta
            # 抽样锚点：0, rate, 2*rate, ... 直到 N（步长 = stats_downsample_rate）。
            sampled_base_idx = torch.arange(0, N, self.stats_downsample_rate, dtype=torch.long)

            # 构造抽样计划（要读哪些帧 + 每个 key 每步取哪帧），边界按需二分查找。
            required_indices, state_indices_by_key, action_indices_by_key = (
                _build_stats_sampling_plan(
                    sampled_base_idx,
                    self.action_size,
                    state_meta_to_use,
                    action_meta_to_use,
                    ep_starts=ep_starts,
                    ep_ends=ep_ends,
                )
            )
            # 日志：锚点数 → 需要读的原始帧数 → 占全量数据的比例（越小越省 I/O）。
            _stats_write(
                "🧮 [Stats] Sample plan: {} base anchors -> {} required raw frames ({} of full data)".format(
                    _format_count(sampled_base_idx.numel()),
                    _format_count(required_indices.numel()),
                    _format_ratio(required_indices.numel(), N),
                ),
                "yellow",
            )

            _stats_write(
                f"📦 [Stats] Loading only required raw rows for {len(columns)} columns...",
                "blue",
            )
            process_start = time.time()
            # 顺序扫 parquet，只取 required_indices 这些行、columns 这些列。
            raw_column_cache = _load_selected_rows_from_parquet(
                parquet_files,
                columns,
                required_indices,
                desc="📥 Loading sampled stats rows",
            )
            # 从 episode 边界反推“每帧属于哪条 episode”：
            # 后面 action 侧的变换需要它来保证动作块不跨 episode（虽然这里的帧号
            # 在 _build_stats_sampling_plan 里已经夹紧过，这里提供的是供变换使用的口径）。
            selected_ep_indices = _get_ep_indices_for_frame_indices(
                required_indices, ep_starts, ep_ends
            )
        else:
            # ── 路径 B（回退）：读整列 episode_index，自己算 episode 边界 ──
            # 适用场景：旧格式数据没有 meta/episodes，或元数据读取失败。
            # 代价：要额外读 N 个 int64，并构造两张长度为 N 的表（内存占用随帧数线性增长）。
            _stats_write(
                "📥 [Stats] Reading episode_index column to build sampling plan...", "blue"
            )
            import pyarrow.dataset as ds

            # to_table：把 episode_index 整列读进内存（这条路径为什么费内存就在这里）。
            dataset = ds.dataset(parquet_files, format="parquet")
            ep_table = dataset.to_table(columns=["episode_index"])
            all_ep_indices = torch.tensor(ep_table["episode_index"].to_numpy(), dtype=torch.long)
            N = int(all_ep_indices.shape[0])
            # 由逐帧 episode 下标推出两张长度为 N 的表：每帧所属 episode 的首帧 / 末帧。
            frame_ep_start = _compute_frame_ep_start(all_ep_indices, N)
            frame_ep_end = _compute_frame_ep_end(all_ep_indices, N)
            sampled_base_idx = torch.arange(0, N, self.stats_downsample_rate, dtype=torch.long)

            # 抽样计划：这里走的是“模式 2”（传预计算的 frame_ep_start/end）。
            required_indices, state_indices_by_key, action_indices_by_key = (
                _build_stats_sampling_plan(
                    sampled_base_idx,
                    self.action_size,
                    state_meta_to_use,
                    action_meta_to_use,
                    frame_ep_start=frame_ep_start,
                    frame_ep_end=frame_ep_end,
                )
            )
            _stats_write(
                "🧮 [Stats] Sample plan: {} base anchors -> {} required raw frames ({} of full data)".format(
                    _format_count(sampled_base_idx.numel()),
                    _format_count(required_indices.numel()),
                    _format_ratio(required_indices.numel(), N),
                ),
                "yellow",
            )

            _stats_write(
                f"📦 [Stats] Loading only required raw rows for {len(columns)} columns...",
                "blue",
            )
            process_start = time.time()
            raw_column_cache = _load_selected_rows_from_parquet(
                parquet_files,
                columns,
                required_indices,
                desc="📥 Loading sampled stats rows",
            )
            # 直接从整列 episode_index 里取需要的帧（O(1) 取值，不用再二分）。
            selected_ep_indices = all_ep_indices[required_indices]
            # 这三个中间结果只在上面用过，及时释放（大 N 时各占几十 MB 到几百 MB）。
            del all_ep_indices, frame_ep_start, frame_ep_end
            gc.collect()
        # ── 把“原始帧号”翻译成“读回来的行号” ──
        # required_indices 由 torch.unique(sorted=True) 生成，天然升序，
        # 因此可以用 searchsorted 二分查找“某帧号落在第几行”。
        #
        # 为什么用 searchsorted 而不是 python dict {帧号: 行号}：
        # 千万帧规模下这种 dict 会占用几个 GB，构造它的过程本身就会 OOM；
        # 而 searchsorted 是 O(M log M) 的向量化操作，内存只有几个下标数组。
        sampled_local_idx = torch.searchsorted(required_indices, sampled_base_idx)
        # 每个 state key 要取的行（通常 = 锚点行；带 time_offset 时是平移后的帧对应的行）。
        state_local_indices_by_key = {
            key: torch.searchsorted(required_indices, raw_indices)
            for key, raw_indices in state_indices_by_key.items()
        }
        # 每个 action key、每个动作步要取的行（嵌套结构：key → [第 0..H-1 步的行号]）。
        action_local_indices_by_key = {
            key: [
                torch.searchsorted(required_indices, raw_indices)
                for raw_indices in per_step_indices
            ]
            for key, per_step_indices in action_indices_by_key.items()
        }
        gc.collect()
        # 打印本阶段的耗时拆分：读计划（算下标）vs 取行（真正读盘）。
        _stats_write(
            "✅ [Stats] Built sampled tensors in {:.2f}s (read plan {:.2f}s, row materialization {:.2f}s)".format(
                time.time() - read_start,
                process_start - read_start,
                time.time() - process_start,
            ),
            "green",
        )

        # ── 按 meta 把原始列切成每个 part 的张量，并取出对应行 ──
        state_dict = {}
        _stats_write("🧠 [Stats] Building sampled state/action tensors...", "blue")
        for meta in state_meta_to_use:
            key = meta["key"]
            if key is None:
                # key=None 是历史写法里的“占位、不输出”，这里直接跳过。
                continue
            # _get_meta_source_data：从 raw_column_cache 里按 lerobot_key 取原始列
            # （多源 meta 会返回张量列表）；_slice_meta_feature：按 start_index/raw_shape
            # 切出本 part 的维度；最后按行号取出锚点对应的那些帧。
            # 这两个方法都来自父类，保证与训练取数的切分口径完全一致。
            state = self._slice_meta_feature(
                self._get_meta_source_data(raw_column_cache, meta), meta
            )
            state_dict[key] = state[state_local_indices_by_key[key]]

        # action 不做行索引：后面 compute_action_stats_with_transforms 会按
        # action_step_indices（每个动作步各自的帧号）逐列取行，这样只读一次就能
        # 复用到 H 个动作步上。
        action_dict = {}
        for meta in action_meta_to_use:
            key = meta["key"]
            if key is None:
                continue
            action = self._slice_meta_feature(
                self._get_meta_source_data(raw_column_cache, meta), meta
            )
            action_dict[key] = action

        # ── 统计量计算阶段（本文件的优化重点）─────────────────────────────────
        # 说明：raw_column_cache 里只有数值列（图像不参与统计），到这一步已经不再需要，
        # 但函数返回后它自然会被回收，所以这里不额外 del。
        stats_start = time.time()

        # 从 processor 里取“动作/状态变换”。统计必须建立在变换后的数值上，
        # 否则相对动作等变换会让统计范围与实际训练输入不一致。
        transforms = None
        if apply_processor and _preprocessor is not None:
            if (
                hasattr(_preprocessor, "action_state_transforms")
                and _preprocessor.action_state_transforms
            ):
                transforms = _preprocessor.action_state_transforms
                _stats_write(
                    f"🧩 [Stats] Will apply {len(transforms)} action_state_transforms",
                    "magenta",
                )
            else:
                _stats_write("🧩 [Stats] No action_state_transforms to apply", "magenta")

        # ── state：多 key 合并成一条宽列一次算完（必要时先过变换）──
        state_stats_start = time.time()
        _stats_write(
            "📐 [Stats] Computing state stats for {} keys on {} sampled rows...".format(
                len(state_dict),
                _format_count(sampled_local_idx.numel()),
            ),
            "blue",
        )
        # downsample_rate=1：上面已经按 stats_downsample_rate 抽过锚点了，
        # 这里再抽一次会让统计口径比配置预期的更稀疏，所以固定传 1。
        state_stats = compute_state_stats_with_transforms(
            state_dict,
            transforms=transforms,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            downsample_rate=1,
        )
        _stats_write(
            f"✅ [Stats] Computed state stats in {time.time() - state_stats_start:.2f}s",
            "green",
        )

        # ── action：逐动作步取列 + 应用变换（相对动作需要 state 当基座）──
        action_stats_start = time.time()
        _stats_write(
            "🎬 [Stats] Computing action stats for {} keys, action_size={}, sampled anchors={}...".format(
                len(action_dict),
                self.action_size,
                _format_count(sampled_local_idx.numel()),
            ),
            "blue",
        )
        # 参数说明（按位置）：
        #   action_dict / state_dict：动作数据与作为变换基座的 state；
        #   selected_ep_indices：每帧所属 episode（用于夹紧滑窗，走 action_step_indices
        #       时只是备用口径，因为帧号已在上游算好）；
        #   action_step_indices：每个动作步要取的行号（见上文“把帧号翻译成行号”）；
        #   show_progress：只有主进程显示进度条。
        action_stats = compute_action_stats_with_transforms(
            action_dict,
            state_dict,
            selected_ep_indices,
            self.action_size,
            transforms=transforms,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            downsample_rate=1,
            show_progress=_should_show_stats_pbar(),
            action_step_indices=action_local_indices_by_key,
        )
        _stats_write(
            f"✅ [Stats] Computed action stats in {time.time() - action_stats_start:.2f}s",
            "green",
        )

        stats = {"state": state_stats, "action": action_stats}

        # only_keys 过滤：内部是全量算的（合并路径更快），这里只挑出调用方要的 key。
        # MixtureLerobotDataset 会按 embodiment 传 only_keys，避免上层拿到用不到的 part。
        if only_keys is not None:
            filtered_stats = {"state": {}, "action": {}}
            for category in ["action", "state"]:
                keys_to_keep = only_keys.get(category, set())
                for key in keys_to_keep:
                    if key in stats[category]:
                        filtered_stats[category][key] = stats[category][key]
            stats = filtered_stats

        # 收尾：主动 GC + 打印统计阶段与总耗时（便于对比不同 downsample_rate 的效果）。
        gc.collect()
        _stats_write(f"🏁 [Stats] Computed all stats in {time.time() - stats_start:.2f}s", "green")
        _stats_write(
            f"🏁 [Stats] Total get_dataset_stats time: {time.time() - total_start:.2f}s",
            "green",
        )
        return stats

    def get_dataset_stats(
        self,
        _preprocessor=None,
        apply_processor: bool = True,
        use_fast_quantile: bool = True,
        num_workers: int = 32,
        only_keys: Optional[Dict[str, Set[str]]] = None,
    ):
        """统计量入口：与父类同名同签名，内部直接转发到快速实现。

        为什么要多这一层包装：父类的签名是 get_dataset_stats(preprocessor, only_keys=...)，
        而 MixtureLerobotDataset 也是按这个签名调用的；保留同样的参数顺序，
        才能在不改任何调用方代码的情况下把统计实现替换掉。

        Args:
            _preprocessor: 处理器（提供 action_state_transforms）；
                名字带下划线表示“按位置传入”，见 _fast_get_dataset_stats 的说明。
            apply_processor: 是否应用处理器里的 action/state 变换。
            use_fast_quantile / num_workers: 仅为兼容旧签名而保留，当前未使用。
            only_keys: 只返回这些 key 的统计量，格式见 _fast_get_dataset_stats。

        Returns:
            {"state": {key: {...}}, "action": {key: {...}}}
        """
        return self._fast_get_dataset_stats(
            _preprocessor=_preprocessor,
            apply_processor=apply_processor,
            use_fast_quantile=use_fast_quantile,
            num_workers=num_workers,
            only_keys=only_keys,
        )


# 分位数实现的性能/精度对比脚本（只用于开发期验证，训练流程不会调用）
def _compare_quantile_methods(
    N: int = 1_000_000,
    action_size: int = 32,
    dim: int = 7,
    num_workers: int = 8,
    repeat: int = 3,
):
    """
    对比 np.quantile（基准）与 fast_quantile_parallel（优化版）的速度与精度。

    结论（在本仓库常用规模下）：优化版快 15~20 倍；两者的分位数只有极小的数值差异
    （来自“取第 k 小”与 np.quantile 的线性插值两种定义），对归一化区间没有实际影响。

    Args:
        N: 数据点数量（模拟的采样帧数）。
        action_size: 动作块长度。
        dim: 动作维度。
        num_workers: 并行线程数。
        repeat: 重复次数（取平均，抵消冷启动/缓存波动）。
    """
    import time

    print("=" * 70)
    print(f"Quantile method comparison (N={N:,}, action_size={action_size}, dim={dim})")
    print("=" * 70)

    # 造一份 [0, 1] 区间内的合成“机器人动作数据”，固定随机种子保证可复现。
    np.random.seed(42)
    data = np.random.rand(N, action_size, dim).astype(np.float32)
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")

    # 基准实现：np.quantile（完整排序，O(N log N)）。
    print("\n[Original method] np.quantile ...")
    times_orig = []
    for i in range(repeat):
        start = time.perf_counter()
        orig_q01 = np.quantile(data, 0.01, axis=0)
        orig_q99 = np.quantile(data, 0.99, axis=0)
        elapsed = time.perf_counter() - start
        times_orig.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
    avg_orig = np.mean(times_orig)
    print(f"  Average time: {avg_orig:.3f}s")

    # 优化实现：np.partition + 多线程（O(N) 选择算法）。
    print(f"\n[Fast method] fast_quantile_parallel (workers={num_workers}) ...")
    times_fast = []
    for i in range(repeat):
        start = time.perf_counter()
        fast_results = fast_quantile_parallel(data, [0.01, 0.99], num_workers)
        fast_q01 = fast_results[0.01]
        fast_q99 = fast_results[0.99]
        elapsed = time.perf_counter() - start
        times_fast.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
    avg_fast = np.mean(times_fast)
    print(f"  Average time: {avg_fast:.3f}s")

    # 计算两者的差异：绝对值 + 相对百分比（用于确认“优化不改变统计结论”）。
    q01_diff = np.abs(orig_q01 - fast_q01)
    q99_diff = np.abs(orig_q99 - fast_q99)
    # 分母接近 0 时相对误差没有意义，直接记 0（np.where 避免除零告警）。
    q01_pct = np.where(np.abs(orig_q01) > 1e-10, q01_diff / np.abs(orig_q01) * 100, 0)
    q99_pct = np.where(np.abs(orig_q99) > 1e-10, q99_diff / np.abs(orig_q99) * 100, 0)

    # 结果汇总：加速比 + 两种分位数的误差统计。
    print("\n" + "=" * 70)
    print("Result summary")
    print("=" * 70)
    print(f"Speedup: {avg_orig / avg_fast:.1f}x")
    print("\nq01 accuracy:")
    print(f"  Max absolute difference: {q01_diff.max():.2e}")
    print(f"  Mean absolute difference: {q01_diff.mean():.2e}")
    print(f"  Max percentage error: {q01_pct.max():.4f}%")
    print(f"  Mean percentage error: {q01_pct.mean():.4f}%")
    print("\nq99 accuracy:")
    print(f"  Max absolute difference: {q99_diff.max():.2e}")
    print(f"  Mean absolute difference: {q99_diff.mean():.2e}")
    print(f"  Max percentage error: {q99_pct.max():.4f}%")
    print(f"  Mean percentage error: {q99_pct.mean():.4f}%")
    print(f"\nOutput shape: q01={fast_q01.shape}, q99={fast_q99.shape}")


if __name__ == "__main__":
    # 直接 `python -m g05.data.base_lerobot_datasetV3` 就会跑一次上面的性能对比。
    _compare_quantile_methods(
        N=1_000_000,
        action_size=32,
        dim=7,
        num_workers=8,
        repeat=3,
    )
