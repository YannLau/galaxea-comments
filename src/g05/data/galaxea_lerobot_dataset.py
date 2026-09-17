# =============================================================================
# src/g05/data/galaxea_lerobot_dataset.py — 银河通用（Galaxea）数据适配层
# =============================================================================
#
# 【这个文件是什么】
#   GalaxeaLerobotDataset 是 src/g05/data/base_lerobot_datasetV3.py 里
#   BaseLerobotDatasetV3 的**子类**，专门服务银河通用 R1 Lite / R1 Pro 这类机器人数据集。
#   “通用取数”全部由基类负责（按 shape_meta 决定读哪些帧、怎么切维、归一化、
#   坏样本重试……），本文件只额外加 4 件 Galaxea 特有的事：
#
#     1) 注册 task_index 的时间查询（_build_delta_timestamps）
#        让底层一次取回“当前帧往后 H 帧的任务文本”，从而能解出 future_task
#        （未来若干帧后的子任务描述），供 FutureSubtaskCoTBuilder 训练“预测下一个子任务”。
#
#     2) 可选地裁掉每条 episode 开头的“静止帧”（ee_start_moving_thresh）
#        录制数据时，机械臂常常先停几秒才开始动，这些帧的动作为 0、信息量很低。
#        打开这个开关后，本类会找出“末端开始运动”的那一帧，把该帧之前的部分裁掉，
#        并把数据集长度与下标映射（get_original_index）一起改成裁剪后的版本。
#
#     3) 给样本补 coarse_task（_get_additional_data）
#        粗粒度/高层任务描述（例如“把盘子放到架子上”），processor 会把它当
#        high-level instruction 用（见 src/g05/data_processor/processor/base_processor.py）。
#
#     4) get_init_positions()：统计各 episode 首帧状态的平均值
#        （遗留工具方法，当前仓库没有调用点，详见下方“⚠️ 坑 3”）。
#
# 【它在数据链路中的位置】
#
#   configs/data/r1lite.yaml / configs/data/r1pro.yaml
#        │  embodiment_datasets.<别名>.type = g05.data.galaxea_lerobot_dataset.GalaxeaLerobotDataset
#        │  （实例化入口：src/g05/utils/data/processor_utils.py）
#        ▼
#   MixtureLerobotDataset                 src/g05/data/mixture_lerobot_dataset.py
#        │  · 多 embodiment 混合、权重采样、统计量聚合
#        ▼
#   GalaxeaLerobotDataset  ← 本文件
#        │  · 只加“Galaxea 特有”的 4 件事，其余全交给基类
#        ▼
#   BaseLerobotDatasetV3 → BaseLerobotDataset
#        │  · 取数、切维、is_pad 掩码、训练/验证切分、坏样本重试、归一化统计量
#        ▼
#   MultiLeRobotDataset → LeRobotDataset(v2.1) / LeRobotDataset(v3.0)
#        │  · 按 delta_timestamps 取帧；把 task_index 翻回文本字符串
#        ▼
#   parquet（action / state 等数值列） + mp4 / png（相机画面）
#
#   消费方：scripts/finetune.py（训练）、scripts/eval_open_loop.py（离线评测）、
#           scripts/serve_policy*.py（真机部署；部署时 coarse_task 由客户端直接给）。
#
# 【新手先记住的 5 个概念】
#
#   (1) 帧下标有三层，不要混用
#       · 全局帧号      episode_data_index["from"][e] ~ ["to"][e] 是**全局帧号**，
#                       左闭右开 [from, to)：把同一组里的多个数据集目录首尾拼接后得到，
#                       由基类在 __init__ 里算好。
#       · 实例内局部下标 本类对外只暴露 [0, len(ds)) 的连续下标，基类 __getitem__ 会
#                       加 self._start_idx 变回全局帧号（训练/验证集切分就体现在这里）。
#       · 裁剪后的压缩下标（可选） 开了 ee_start_moving_thresh 之后，下标空间会被
#                       “压紧”（去掉每条 episode 开头的静止帧），再由 get_original_index()
#                       还原成全局帧号。多出来的这一层**只在本文件启用裁剪时存在**。
#
#   (2) “末端速度” = 动作块里相邻两步的位姿之差
#       动作是动作块：每一帧位置对应 (T, H, D) 的 H 步未来动作。
#       对 ee_pose（末端位姿，前 3 维 = xyz 位置）来说：
#           actions[:, 0, :3]  = 当前帧（第 0 步）末端位置
#           actions[:, 1, :3]  = 下一帧（第 1 步）末端位置
#       两者相减取模长 = 这一帧的末端位移（≈ 速度）。超过阈值就认为“机械臂开始动了”。
#       注意：用的是 action（指令意图）而不是 state（实际状态）；并且只看前 3 维位置，
#       因为姿态分量的数值尺度与位置完全不同，直接拼在一起取范数没有物理意义。
#
#   (3) from_moving_step：每条 episode 的“真正起点”（全局帧号）
#       裁剪后，第 e 条 episode 的有效帧数是 to[e] − from_moving_step[e]，
#       所有 episode 的有效帧数之和就是新的 len(ds)。详见 _get_ee_start_moving_step()。
#
#   (4) future_task 与 task_index
#       LeRobot 的 meta/tasks.parquet 是一张“通用字符串表”（index = 文本，列 task_index = 整数），
#       task / coarse_task / plan / bbox / memory 等 CoT 文本全都存在这张表里。
#       把 task_index 注册进 delta_timestamps 后，底层一次就会取回
#       [当前帧, …, 当前帧 + H − 1] 这 H 个任务索引（= chunked_task_index）；
#       再取其中第 future_task_offset 个翻回文本，就是 future_task。
#       完整链路：本文件注册查询 → lerobot_dataset_v3.py 解码 → samples_builder.py 的
#       FutureSubtaskCoTBuilder 填进 CoT 模板（“Subtask: <未来子任务>”，见 docs/data/samples_builders_zh.md）。
#
#   (5) coarse_task：粗粒度任务描述
#       由底层从 coarse_task_index 翻成文本，本文件在 _get_additional_data 里挂到
#       sample["coarse_task"]；processor 把它当 high-level instruction 用
#       （drop_high_level_prob 控制训练时以一定概率丢弃）。真机部署时客户端也可以直接传
#       coarse_task（见 scripts/serve_policy.py 对 raw_obs["coarse_task"] 的处理）。
#
# 【⚠️ 读代码前先知道的 4 个坑】
#
#   坑 1：裁剪静止帧 + 训练/验证切分不能同时开
#       _get_ee_start_moving_step() 是**对全部 episode**（不区分本实例是训练集还是验证集）
#       重算 dataset_len 与累计长度表的，而基类 __getitem__ 拿到帧号后还会再加 self._start_idx。
#       两者同时生效时下标会错位。所以该功能只应在 val_set_proportion < 1e-6
#       （不切分，_start_idx == 0）时使用。
#       当前仓库所有 configs 都没有设置 ee_start_moving_thresh（默认 0.0），
#       也就是说这段裁剪逻辑在现有配置下**完全不会执行**。
#
#   坑 2：future_task_offset 目前“传不下去”
#       本类只把它存成 self._future_task_offset，并没有下发给底层 LeRobotDataset 实例，
#       所以 lerobot_dataset_v3.py 里的 getattr(self, "_future_task_offset", 16)
#       永远拿到默认值 16。要真正改偏移量，得让上层把它设到子数据集对象上。
#
#   坑 3：get_init_positions() 是死代码
#       它依赖 self.init_state_meta，但基类从未定义这个属性（全仓库只有本方法引用它），
#       也没有任何调用点 —— 直接调用会 AttributeError。属于上游遗留工具方法，
#       读到时理解意图即可（“每个 part 在各 episode 首帧的平均状态”）。
#
#   坑 4：overfit 模式会绕过裁剪
#       __getitem__ 里若处于 overfit 模式（存在 self._overfit_indices）就直接走
#       super().__getitem__(idx)，不做压缩下标映射。这是刻意的：overfit 的样本集合
#       本来就是用原始下标预先筛出来的，再映射一次就取错样本了。
#
# 【相关文件与文档】
#   src/g05/data/base_lerobot_dataset.py          基类：读写帧计划、切维、归一化、坏样本重试
#   src/g05/data/base_lerobot_datasetV3.py        上方基类：只重写“统计量计算”
#   src/g05/data/mixture_lerobot_dataset.py       多 embodiment 混合与索引调度
#   src/g05/data/lerobot/lerobot_dataset_v3.py    底层实现：delta 查询、future_task / coarse_task 解码
#   src/g05/data_processor/processor/samples_builder.py  FutureSubtaskCoTBuilder 等 CoT 模板
#   configs/data/r1lite.yaml, configs/data/r1pro.yaml    实际使用本类的配置
#   docs/data/samples_builders_zh.md              future_task 与 CoT 模板说明
# =============================================================================

import torch
import numpy as np
from typing import List, Literal, Dict, Optional, Any, DefaultDict
from tqdm import tqdm

# 单独把并发工具 import 出来：用于“逐 episode 扫一遍是否有末端运动”这一步。
# （as_completed：谁先算完先处理，配合 tqdm 显示进度。）
from concurrent.futures import ThreadPoolExecutor, as_completed

# 基类：负责全部通用取数逻辑；本文件只做 Galaxea 特有的扩展。
from g05.data.base_lerobot_datasetV3 import BaseLerobotDatasetV3


class GalaxeaLerobotDataset(BaseLerobotDatasetV3):
    """银河通用（R1 Lite / R1 Pro）数据集适配层。

    相比基类只多做 4 件事（细节见文件头说明）：
      1) 注册 task_index 的 delta 查询，让样本里带上 future_task；
      2) 可选地裁掉每条 episode 开头的静止帧（ee_start_moving_thresh > 0 时生效）；
      3) 把 coarse_task 挂到样本上；
      4) 提供 get_init_positions()（遗留方法，当前无调用点）。

    其余一切（shape_meta 解析、读帧、切维、掩码、统计量、坏样本重试）都继承自基类，
    因此本文件里几乎看不到“真正取数”的代码。
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        # 形状声明（每个 part 从哪一列的哪几维取多少宽），详见基类 _setup_meta_lerobot_keys
        shape_meta: Dict[str, Any],
        action_size: int,
        past_action_size: int = 0,
        obs_size: int = 1,
        obs_stride_second: float = 0.0,
        # 信号处理：末端开始运动的判定阈值（0 = 不做裁剪，保持基类行为）
        ee_start_moving_thresh: float = 0.0,
        # 未来帧任务的偏移量（future task offset），供 FutureSubtaskCoTBuilder 使用
        # （0 = 就是当前帧）；
        # ⚠️ 当前实现里它只被本类保存，没有下发到底层，实际生效值恒为 16，见文件头“坑 2”。
        future_task_offset: int = 16,
        # 训练集 vs 验证集
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        # 底层 LeRobot 数据集格式版本：'2.1' / '3.0'（必须与磁盘上 meta/info.json 一致）
        lerobot_ds_version: Optional[Literal["2.1", "3.0"]] = "2.1",
        # 归一化统计量用的分位数上下界（稳健归一化，裁掉极端离群点）
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        stats_downsample_rate: int = 10,
        # 图像查帧时间容差（None = 自动用 0.4/data_fps 吸收 H.264 重编码的时间戳抖动）
        tolerance_s: Optional[float] = None,
        **kwargs,
    ):
        # 先把通用参数原样交给基类：基类在这一步里完成绝大部分耗时工作
        # （读 meta、建 MultiLeRobotDataset、算训练/验证帧区间、注册全局 episode 索引……）。
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
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            stats_downsample_rate=stats_downsample_rate,
            tolerance_s=tolerance_s,
            **kwargs,
        )

        # _future_task_offset 存的是“取未来第几帧的任务文本”的偏移量。
        # 注意上层（SamplesBuilder）并不会读这个属性，真正的解码发生在底层
        # lerobot_dataset_v3.py 里，且那里读的是“底层实例自己的属性”，所以当前恒为 16。
        self._future_task_offset = future_task_offset
        # 末端开始运动的判定阈值：> 1e-6 才启用裁剪静止帧的功能（默认 0 = 关闭）。
        self.ee_start_moving_thresh = ee_start_moving_thresh
        if self.ee_start_moving_thresh > 1e-6:
            # 只有需要判运动时，才去 action_meta 里筛出“末端位姿”这类 part。
            # 判定依据是 ee_pose 的位置分量，所以配置里必须真的有带 "ee_pose" 的 action part。
            self.ee_pose_action_meta = [
                meta for meta in self.action_meta if "ee_pose" in meta["key"]
            ]
            # 配了阈值却没有 ee_pose 可读 → 直接在这里报错，而不是等到训练时才发现没裁掉静止帧。
            assert len(self.ee_pose_action_meta) > 0, (
                "ee_start_moving_thresh is set but ee_pose is not in action_meta"
            )
            # 立刻扫描全部 episode，算出每条 episode 的“开始运动帧”，
            # 并把数据集长度 / 下标映射切换成裁剪后的版本。
            # max_workers=1：串行执行（见该方法里的说明），构造期行为保持可预测。
            self._get_ee_start_moving_step(max_workers=1)

    def _get_ee_start_moving_step_of_episode(self, episode_idx: int) -> int:
        """求第 episode_idx 条 episode“末端开始运动”的帧号（**全局帧号**）。

        判定方法：把该 episode 的动作块按帧展开，比较“第 0 步位置”和“第 1 步位置”的
        距离（≈ 末端位移/速度），第一个超过阈值 ee_start_moving_thresh 的帧就是起点。
        多个 ee_pose part（例如左臂 + 右臂）的位移会先相加，
        也就是“任意一只手臂开始动”就算开始（数据里通常只有一条 ee_pose，相加即不变）。

        返回值的三种情况：
          · 找到运动帧 → 该帧的全局帧号 = episode 起点 + 帧内相对下标；
          · 全程都没超过阈值（一直不动）→ 该 episode 的**最后一帧**（to − 1），
            相当于把整条 episode 压成 1 帧，避免产生越界下标；
          · 该 episode 里没有 ee_pose 这类列（.get 返回 None）→ 同上，返回最后一帧。
        """
        # 读整条 episode 的**数值列**（不解码视频），并按训练用的时间结构整理好：
        #   {"action": {key: (T, H, D)}, "state": {key: (T, 1, D)}}
        # 这里关心的是 action：每条帧位置都已经展开成未来 H 步的动作块。
        episode_data: Dict[str, Any] = self._get_episode_data(episode_idx)
        # 现场再筛一次 ee_pose part，而不是复用 __init__ 里的 self.ee_pose_action_meta：
        # 这样本方法被单独调用（没有走过 __init__ 的那个分支）时也能正常工作。
        ee_pose_action_meta: List[Dict[str, Any]] = [
            meta for meta in self.action_meta if "ee_pose" in meta.get("key", "")
        ]

        # 收集每个 ee_pose part 的逐帧位移（长度 T 的一维张量）。
        all_movement_distances = []
        for meta in ee_pose_action_meta:
            key = meta["key"]

            # 取该 part 的动作块；若这条 episode 没有这个 part（例如单臂数据），
            # .get 返回 None，跳过即可（不会报错）。
            actions: torch.Tensor = episode_data["action"].get(key)

            if actions is None:
                continue

            # 形状说明：actions 是动作块 (T, H, D) —— T 帧、每帧 H 步未来动作、每步 D 维。
            # 只取前 3 维（xyz 位置），姿态分量不参与“是否运动”的判定。
            # 两者形状都是 (T, 3)：
            #   position_a = 第 0 步 = 当前帧的末端位置
            #   position_b = 第 1 步 = 下一帧的末端位置（H=1 时是复制出来的同一位置，位移恒为 0）
            position_a = actions[:, 0, :3]
            position_b = actions[:, 1, :3]

            # 相邻两步的位置差 → 取模长 → 得到逐帧的位移量（单位与数据集的长度单位一致）。
            difference_vector = position_a - position_b
            movement_distance = torch.linalg.norm(difference_vector, dim=1)

            all_movement_distances.append(movement_distance)

        # 一个 ee_pose 都没取到（该 episode 缺列）：退回“最后一帧”，等价于不裁剪。
        if not all_movement_distances:
            return self.episode_data_index["to"][episode_idx] - 1

        # 多个 ee_pose part 相加：只要任意一条手臂的位移超过阈值，就认为整个机器人开始运动了。
        total_movement = torch.stack(all_movement_distances).sum(dim=0)

        # 找出所有超过阈值的帧号（nonzero 返回二维下标，squeeze(1) 压成一维帧号列表）。
        indices_of_movement = torch.nonzero(
            total_movement > self.ee_start_moving_thresh, as_tuple=False
        ).squeeze(1)

        if indices_of_movement.numel() > 0:
            # 取第一个运动帧，并把“episode 内相对下标”换算成“全局帧号”。
            first_movement_relative_index = indices_of_movement[0].item()
            absolute_index = (
                self.episode_data_index["from"][episode_idx] + first_movement_relative_index
            )
            return absolute_index
        else:
            # 整条 episode 都没动：返回最后一帧（to 是右开边界，所以要减 1）。
            return self.episode_data_index["to"][episode_idx] - 1

    def _get_ee_start_moving_step(self, max_workers: Optional[int] = None):
        """为所有 episode 计算 from_moving_step，并把数据集长度改成“裁剪后”的版本。

        产出：
          self.episode_data_index["from_moving_step"]  每条 episode 的裁剪起点（全局帧号）
          self.dataset_len                             裁剪后的总样本数
          self._episode_lengths / self._cumulative_lengths
                       裁剪后各 episode 的长度与累计长度（_cumulative_lengths 开头补一个 0）

        这些量在别处这样被用到：
          __len__()             → 返回 dataset_len（覆盖基类的 _end_idx − _start_idx）
          __getitem__(idx)      → 先 get_original_index(idx) 把压缩下标还原成全局帧号
          get_original_index()  → 用 _cumulative_lengths 二分定位 episode

        为什么用线程池：每个 episode 都要独立读一遍 parquet 数值列（I/O 密集），
        结构上按“可并行”写；但调用处（__init__）固定传 max_workers=1，
        也就是**实际串行执行**——因为所有任务共用同一个底层数据集对象
        （文件句柄 / 缓存），并发读同一实例并不安全；保留该参数是为了将来需要时再放开。

        ⚠️ 注意本方法遍历的是**全部 episode**，与基类的训练/验证切分（_start_idx/_end_idx）
        不联动，所以两者不要同时启用（见文件头“坑 1”）。
        """
        # 任务列表 = 所有 episode 的下标（不是帧下标）。
        tasks = list(range(len(self.episode_data_index["from"])))

        # 预分配结果数组：先占位 None，算完再按 episode 下标回填，
        # 这样即使任务乱序完成，结果顺序也始终与 episode 顺序一致。
        from_moving_step_idxs = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有 episode 的计算任务，并记录“这个 future 对应哪条 episode”。
            future_to_episode = {
                executor.submit(self._get_ee_start_moving_step_of_episode, episode_idx): episode_idx
                for episode_idx in tasks
            }

            # 谁先算完先处理；tqdm 的 desc 是进度条上的文字（构造大数据集时这一步可能比较慢）。
            for future in tqdm(
                as_completed(future_to_episode),
                total=len(future_to_episode),
                desc="Calculating from_moving_step indices",
            ):
                episode_idx = future_to_episode[future]
                try:
                    result = future.result()
                    from_moving_step_idxs[episode_idx] = result
                except Exception as exc:
                    # 单条 episode 出问题（例如文件损坏）不应该让整个数据集构造失败：
                    # 打印告警，并退回“该 episode 的起点”，即这条 episode 不做裁剪。
                    print(f"Episode {episode_idx} generated an exception: {exc}")
                    from_moving_step_idxs[episode_idx] = self.episode_data_index["from"][
                        episode_idx
                    ]

        # 存成张量（python int 列表 → LongTensor），后续按下标取值都在张量上做。
        self.episode_data_index["from_moving_step"] = torch.tensor(from_moving_step_idxs)

        # 重新计算总长度：每条 episode 的有效帧数 = to − from_moving_step，
        # 累加起来就是裁剪后的样本总数（也就是新的 len(ds)）。
        total_length = 0
        for i in range(len(self.episode_data_index["from"])):
            from_moving_idx = self.episode_data_index["from_moving_step"][i]
            to_idx = self.episode_data_index["to"][i]
            total_length += to_idx - from_moving_idx
        self.dataset_len = total_length

        # 每条 episode 裁剪后的长度；to 是右开边界，所以 to − from_moving_step 正好是帧数。
        self._episode_lengths = (
            self.episode_data_index["to"] - self.episode_data_index["from_moving_step"]
        )
        # 累计长度表：开头补一个 0，于是第 e 条 episode 占据 [cum[e], cum[e+1]) 这段压缩下标，
        # 用 torch.searchsorted 就能 O(log n) 地把压缩下标翻译成“第几条 episode + 帧内偏移”。
        self._cumulative_lengths = torch.cat(
            [torch.tensor([0]), torch.cumsum(self._episode_lengths, dim=0)]
        )

    def get_original_index(self, idx):
        """把“裁剪后的压缩下标”还原成“全局帧号”。

        没启用裁剪（没有 from_moving_step 字段或还没算好累计长度）时是恒等映射，直接返回 idx。

        启用裁剪后：_cumulative_lengths = [0, L0, L0+L1, …]，第 e 条 episode 覆盖
        [cum[e], cum[e+1]) 这段压缩下标，因此：
          · searchsorted(..., right=True) − 1 → 定位到 episode 下标 e
            （开头补的那个 0 正好让这个减一成立；
             例：cum=[0,5,11]，idx=0/4→0，idx=5/10→1，idx=11→2）
          · idx − cum[e]                     → 在该 episode 内的偏移
          · from_moving_step[e] + 偏移        → 原始全局帧号
        返回时统一 .item() 成 python int，交给基类 __getitem__ 使用。
        """
        # 未启用裁剪：下标本身就是帧号，直接透传。
        if not "from_moving_step" in self.episode_data_index:
            return idx

        # 裁剪起点算过、但累计长度还没建好（理论上不会发生）也按透传处理，避免崩。
        if not hasattr(self, "_episode_lengths"):
            return idx

        # 二分定位压缩下标落在哪条 episode 上。
        episode_idx = torch.searchsorted(self._cumulative_lengths, idx, right=True) - 1

        # 帧内偏移 + 该 episode 的裁剪起点 = 原始全局帧号。
        offset = idx - self._cumulative_lengths[episode_idx]
        original_idx = self.episode_data_index["from_moving_step"][episode_idx] + offset

        return original_idx.item()

    def __len__(self):
        """数据集长度，三种情况按优先级返回：

        1) overfit 模式（存在 _overfit_len）→ 固定的小样本数量（基类 enable_overfit 设置）；
        2) 启用裁剪静止帧           → 裁剪后的 dataset_len；
        3) 默认                    → 基类的帧区间长度 _end_idx − _start_idx。
        """
        if hasattr(self, "_overfit_len"):
            return self._overfit_len
        if "from_moving_step" in self.episode_data_index:
            return self.dataset_len
        else:
            return self._end_idx - self._start_idx

    def _get_additional_data(self, sample, lerobot_sample):
        """子类扩展点：把粗粒度任务文本挂到样本上。

        基类此方法默认原样返回 sample；本类利用它补上 coarse_task
        （由底层从 coarse_task_index 翻成文本），processor 会把它当 high-level instruction。
        """
        sample["coarse_task"] = lerobot_sample["coarse_task"]
        return sample

    def __getitem__(self, idx):
        """取一个训练样本（DataLoader 的入口）。

        与基类的唯一区别：如果启用了“裁剪静止帧”，需要先把压缩下标 idx
        翻译回原始全局帧号，再交给基类按常规流程取数（基类内部还会加 _start_idx、
        处理 is_pad 掩码、坏样本重试等）。
        """
        # 越界检查（长度可能是 overfit 长度或裁剪后的长度，所以用 len(self) 判断）。
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        # overfit 模式下基类会把 idx 解释为“稳定样本列表”的下标，
        # 这里不能做压缩映射，直接透传（见文件头“坑 4”）。
        if hasattr(self, "_overfit_indices"):
            return super().__getitem__(idx)

        # 压缩下标 → 原始全局帧号 → 基类取数。
        original_idx = self.get_original_index(idx)
        sample = super().__getitem__(original_idx)

        return sample

    def get_init_positions(self):
        """统计“各 episode 首帧状态的平均值”，每个 state part 得到一条向量。

        ⚠️ 当前仓库里没有任何调用点，而且 self.init_state_meta 在基类中并不存在
        （全仓库只有本方法引用它），直接调用会 AttributeError。属于上游遗留工具方法，
        保留是为了将来需要“固定初始参考位姿”时能接回来（例如给相对/绝对动作提供参考起点）。

        实现要点：
          1) _set_return_images(False) 临时关掉图像 —— 这里只用数值状态，
             关掉后底层就不会去解码视频，省掉大量 I/O；
          2) 遍历每条 episode 的**首帧全局帧号**（episode_data_index["from"]），
             取该帧的 state 张量（形状 [查询帧数, D]），第 0 行即该 episode 的第一帧；
          3) 跨 episode 求平均（np.mean(axis=0)），得到一个“典型初始状态”；
          4) 结束后把图像开关恢复，避免影响后续正常的训练取数。
        """
        # 关掉图像：只读数值列，避免不必要的视频解码。
        self._set_return_images(False)
        init_positions = {}
        for meta in self.init_state_meta:
            init_positions[meta["key"]] = []
        # 各 episode 的首帧全局帧号（转成 python list 以便遍历）。
        first_frame_indices = self.episode_data_index["from"].numpy().tolist()
        for index in tqdm(
            first_frame_indices, desc="Processing first frames for init positions", leave=False
        ):
            for meta in self.init_state_meta:
                key = meta["key"]
                # 直接向底层多数据集取该帧（返回 delta 查询结果，形状 [查询帧数, D]），
                # 再按 meta 声明切出本 part 的维度。
                state = self._slice_meta_feature(
                    self.multi_dataset[index][meta["lerobot_key"]], meta
                )
                # 取第 0 行 = 该查询窗口的第一帧 = 该 episode 的首帧状态。
                init_positions[key].append(state[0])
        # 跨 episode 取平均，得到每个 part 的“典型初始状态”。
        for key, val in init_positions.items():
            init_positions[key] = np.mean(val, axis=0)
        # 恢复图像开关，避免影响后续取数。
        self._set_return_images(True)
        return init_positions

    def _build_delta_timestamps(
        self, fps, past_action_size, action_size
    ) -> Dict[str, list]:
        """先委托基类，再追加 task_index 的查询。

        做法：先让基类按 shape_meta 生成常规查询计划（images / state / action），
        再追加一项 task_index 查询 —— 这样每个样本都会额外带上“未来 H 帧的任务索引”，
        底层据此解出 future_task（见文件头概念 4）。
        返回格式：{"<原始列名>": [秒偏移, ...]}，直接交给 MultiLeRobotDataset。
        """
        delta_timestamps = super()._build_delta_timestamps(fps, past_action_size, action_size)
        # 注意：quality_index（数据质量标注）暂时**不注册**查询（对应原英文注释
        # “quality_index disabled — has multiple bugs”），原因有三个：
        #   1) 会导致无限循环；2) 把 torch 张量当作 pandas 的 iloc 用；3) 标注列缺失时直接崩。
        # 将来要重新启用，得先修好 lerobot_dataset_v3.py 的 __getitem__ 与
        # base_lerobot_dataset.py 里的重试（attempt）逻辑，再放开下面这行。
        # delta_timestamps["quality_index"] = [t / fps for t in range(-past_action_size, action_size)]
        # 与 action 使用完全相同的偏移序列：第 t 个元素 = “当前帧往后第 t 步”的任务索引，
        # 因此下标 future_task_offset 处就是“未来第 future_task_offset 帧的任务文本”。
        delta_timestamps["task_index"] = [t / fps for t in range(-past_action_size, action_size)]
        return delta_timestamps
