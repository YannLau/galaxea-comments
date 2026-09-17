#!/usr/bin/env python
# Adapted from Hugging Face LeRobot dataset utilities.
# Upstream: https://github.com/huggingface/lerobot

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# src/g05/data/lerobot/lerobot_dataset.py — LeRobot v2.1 数据集实现（== 本文件）
# 上游来源：https://github.com/huggingface/lerobot （Apache-2.0，见上方 license）
# =============================================================================
#
# 【这个文件是什么】
#   它是整个训练数据链路的**最底层读取实现**：打开一个 LeRobot 格式的数据集目录，
#   把 parquet 里的数值列和 mp4 里的相机画面按“时间戳”读出来。上游代码按本仓库的
#   目录结构做了移植：所有 `lerobot.*` 导入改成 `g05.data.lerobot.*`，并去掉了
#   训练用不到的部分（例如图像异步写入 AsyncImageWriter）。
#
# 【它在数据链路中的位置】
#
#   configs/data/*.yaml                    ← 声明数据集目录 / shape_meta / fps …
#        │
#   MixtureLerobotDataset                  src/g05/data/mixture_lerobot_dataset.py
#        │  按权重混合多个 embodiment
#   BaseLerobotDataset                     src/g05/data/base_lerobot_dataset.py
#        │  · 按 shape_meta 把原始列切成各 part（left_arm / head_rgb …）
#        │  · 拼观测窗口、生成 action chunk、算归一化统计量
#        │  · lerobot_ds_version == "2.1" 时 import 本文件，== "3.0" 时改用
#        │    lerobot_dataset_v3.py（接口同名，内部读法不同）
#        ▼
#   MultiLeRobotDataset                    ← 本文件，见「类 3/3」
#        │  把多个数据集目录“首尾相接”成一个索引空间，并补 dataset_index
#        ▼
#   LeRobotDataset                         ← 本文件，见「类 2/3」
#        │  单个数据集目录的读取：parquet 数值列 + mp4 相机帧
#        ▼
#   LeRobotDatasetMetadata                 ← 本文件，见「类 1/3」
#          只读 meta/*.json(l)，不读任何真实数据（最轻量）
#
# 【两点必须知道的前提】
#   1）v2.1 这条路径**只从本地磁盘读**：repo_id 直接就是目录路径，基本不走
#      HuggingFace Hub。下面出现的 pull_from_repo / push_to_hub 只在“录制新数据集”
#      或“确实需要下载”时才被触发。
#   2）v2.1 是历史兼容路径，新数据集建议用 v3.0（lerobot_dataset_v3.py，官方口径
#      快 10~50×）。BaseLerobotDataset 选到 v2.1 时会打一条 warning 提醒。
#
# 【磁盘上的目录结构（一个 LeRobot 数据集长什么样）】
#
#   <dataset_root>/
#   ├── meta/
#   │   ├── info.json             # 全局元信息：fps、features（列名/形状/dtype）、总帧数…
#   │   ├── episodes.jsonl        # 每行一条 episode：长度、涉及哪些 task
#   │   ├── episodes_stats.jsonl  # 每行一条 episode 的统计量（min/max/mean/std/count）
#   │   ├── stats.json            # 全数据集聚合统计量（仅 v2.0 老格式才有）
#   │   ├── tasks.jsonl           # task 字符串 ↔ task_index 的字典
#   │   └── annotations/*.jsonl   # 可选：子任务/场景/夹爪等标注（本仓库扩展）
#   ├── data/
#   │   └── chunk-000/episode_000000.parquet   # 所有非视觉列（state/action/timestamp…）
#   ├── videos/                   # 仅当 features 里存在 dtype=="video" 的相机列
#   │   └── chunk-000/observation.images.<cam>/episode_000000.mp4
#   └── images/                   # 录制期的临时图像目录，episode 保存后会被删除
#
#   · chunk：episode 的分组，默认 1000 条一组（DEFAULT_CHUNK_SIZE），
#     避免单个目录里文件数爆炸。
#   · 一帧 = parquet 里的一行；同一帧的所有相机图像在各自 mp4 里处于同一时间戳。
#   · parquet 只存非视觉列；视觉列在 info["features"] 里声明为 dtype=image/video，
#     读的时候由 _query_videos() 到 mp4 里按时间戳解码。
#
# 【新手先记住这 6 个概念】
#
#   (1) feature / dtype：info["features"] 是列的声明表，形如
#         {"observation.state": {"dtype": "float32", "shape": (14,), "names": [...]}}
#       dtype 取 float32/int64/bool/string/image/video，后两者是视觉列；
#       另有 8 个底层自带列（见 DEFAULT_FEATURES）：timestamp / frame_index /
#       episode_index / index / coarse_task_index / task_index /
#       coarse_quality_index / quality_index。
#
#   (2) episode_data_index：把“第几个 episode”映射到“全局帧下标区间”的索引表，
#         episode_data_index["from"][e] ~ ["to"][e] 即第 e 条 episode 的区间 [from, to)。
#       __getitem__(idx) 里的 idx 就是全局帧下标（所有 episode 首尾相接后的下标）。
#
#   (3) delta_timestamps：用“秒”表达的相对采样时刻，例如
#         {"action": [-1.0, 0.0], "observation.state": [0.0]}
#       表示“取当前帧前 1 秒的 action + 当前帧的 state”。底层乘 fps 换算出整数
#       步长 delta_indices；单位是秒而不是帧，是为了对不同 fps 的数据集保持同一语义。
#
#   (4) is_pad：请求的帧若落在 episode 之外，底层**不报错**，而是把帧号 clamp 到
#       边界帧，同时把该位置标记为 True（键名 "{key}_is_pad"）。训练侧据此把这些步
#       排除在 loss 之外，避免拿“复制出来的边界帧”当真值。
#
#   (5) stats：每个 feature 的 min/max/mean/std/count，供 normalizer 归一化使用。
#       v2.1 逐 episode 存在 meta/episodes_stats.jsonl，加载时再聚合成 self.stats。
#
#   (6) load_images / during_training：两个开关都只影响**读**，不影响写。
#         · load_images=False：不返回视觉列、也不解码视频（只要 state/action 的场景
#           用它，能明显省时间）；
#         · during_training=False：只在 v2.1 的 __getitem__ 里生效，临时关掉视频解码。
#
# 【建议的阅读顺序】
#   1. LeRobotDatasetMetadata：meta 从哪来，info/features/stats/tasks 各是什么；
#   2. LeRobotDataset.__init__：root/revision/是否需要下载/时间戳一致性校验；
#   3. LeRobotDataset.__getitem__ 及配套的 _get_query_*：一次取样完整的数据流；
#   4. add_frame / save_episode：录制时数据如何落盘（纯训练场景用不到）；
#   5. MultiLeRobotDataset：多数据集拼接，以及样本里 dataset_index 的来历。
#
# 【常见坑 / 排查清单】
#   · 报 “Timestamp ... not in sync”（check_timestamps_sync 失败）：parquet 里的
#     timestamp 列与 info.json 的 fps 不一致，通常是重编码/清洗数据时改过 fps，
#     或采样有丢帧。容差由 tolerance_s 决定，BaseLerobotDataset 默认 0.4/fps。
#   · 报 “The given NumPy array is not writable”：只是告警，来自 pyarrow 的
#     zero-copy 数组转 torch（MultiLeRobotDataset.get_episode_data 里已屏蔽）。
#   · 取到的 action chunk 末尾几帧“看着重复”：那是越界 clamp 的边界帧，
#     请用对应的 {key}_is_pad 掩码把它们排除，而不是丢掉整个样本。
#   · 指定了 episodes 子集后索引对不上：注意有两套下标 —— 真实 episode_index
#     （存在 parquet 列里）和“在选中列表中的位置”（episode_data_index 用），
#     __getitem__ 里是靠 self.episodes.index(ep_idx) 做换算的。
#   · load_images=False 却仍想用 video 列：不会解码，视频列根本不会出现在 item 里。
#   · 出现 “One or more datasets have no keys common to all of them”：多数据集目录的
#     features 交集为空，先检查是否混进了 schema 完全不同的目录。
#   · MultiLeRobotDataset 静默少了一个数据源：某个目录构造失败时只打日志并跳过
#     （见 __init__ 里的 try/except），务必查看日志里的 traceback。
#
# 【上游遗留、本次未修的行为（读到别惊讶）】
#   · save_episode(episode_data=...) 传参时会因 episode_buffer 未赋值而 NameError；
#   · add_frame 的断言文案写的是 “two elements”，实际要求 task 长度为 4；
#   · MultiLeRobotDataset.repo_index_to_id 迭代 dict 时少了 .items()，调用会抛错；
#   · v2.0 老格式的逐 episode 统计量只是把全量 stats 复制了一份（近似值）。
#
# 【相关文件与文档】
#   src/g05/data/lerobot/lerobot_dataset_v3.py     v3.0 实现（接口同名，批量读，推荐新数据集使用）
#   src/g05/data/base_lerobot_dataset.py           上层：shape_meta / delta_timestamps / 统计量
#   src/g05/data/lerobot/datasets/utils.py         meta 读写、feature 校验、时间戳校验
#   src/g05/data/lerobot/datasets/video_utils.py   mp4 编解码
#   src/g05/data/lerobot/datasets/compute_stats.py episode 统计量与聚合
#   docs/data/schema_zh.md                         shape_meta 与数据集 schema
# =============================================================================
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Callable, List, Literal
import warnings

# 第三方依赖：
#   datasets         —— HuggingFace datasets，按 features 定义读 parquet / 解码图像
#   pyarrow.parquet  —— 绕过 datasets 直接读 parquet（MultiLeRobotDataset.get_episode_data）
#   huggingface_hub  —— 下载/上传数据集（只有非本地场景才会真的用到）
import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
import pyarrow.parquet as pq
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

# 本仓库把上游的 lerobot.* 导入改成了 g05.data.lerobot.*。
# HF_LEROBOT_HOME 是“不传 root 参数时”的默认落盘位置：
#   环境变量 HF_LEROBOT_HOME 优先，否则 ~/.cache/huggingface/lerobot
from g05.data.lerobot.constants import HF_LEROBOT_HOME
from g05.data.lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

# 注意：下面这个导入已被注释掉 —— 图像异步写入（AsyncImageWriter / write_image）在本
# 仓库未启用，录制路径目前是同步写盘的（add_frame 里对应的 _save_image 也一并注释掉了）。
# from g05.data.lerobot.datasets.image_writer import AsyncImageWriter, write_image

# datasets/utils.py 里的文件级工具：路径常量、meta 的读写、feature 校验、时间戳校验等。
# 其中几个常量后面会频繁出现：
#   INFO_PATH          = "meta/info.json"
#   TASKS_PATH         = "meta/tasks.jsonl"
#   DEFAULT_IMAGE_PATH = "images/{image_key}/episode_XXXXXX/frame_XXXXXX.jpeg"
#   DEFAULT_FEATURES   = 上文中那 8 个底层自有列
from g05.data.lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    _validate_feature_names,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    check_timestamps_sync,
    # check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_hf_features_from_features,
    # get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    load_annotations,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)

# 视频工具：
#   VideoFrame            datasets 的 feature 类型，声明“这一列要从 mp4 解码”
#   decode_video_frames   按时间戳列表从 mp4 解码出对应帧
#   encode_video_frames   调 ffmpeg 把临时 png 帧压成 mp4（录制路径用）
#   get_video_info        读 mp4 的分辨率/编码信息，写进 info["features"][key]["info"]
#   get_safe_default_codec 选择本机可用的解码后端（torchcodec / pyav / video_reader）
from g05.data.lerobot.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
import traceback

# 本文件实现的是**数据集格式**版本 v2.1（不是本仓库的版本号）。
# 它会写进 info["codebase_version"]，加载时用来判断走新格式还是老格式分支
# （例如 load_metadata 里判断是否要读 stats.json 做向后兼容）。
# v2.1 与 v3.0 的差别：v2.1 逐帧/逐 episode 读 parquet，v3.0 批量读并支持
# in_memory 缓存，官方口径快 10~50×。
CODEBASE_VERSION = "v2.1"


# =============================================================================
# 类 1/3：LeRobotDatasetMetadata —— 只读元信息，不读任何真实数据
# =============================================================================
# 【职责】
#   打开 <root>/meta/*.json(l)，把全局元信息解析成一批属性：
#     info            meta/info.json 的原始 dict（fps / features / 总数 / 路径模板 …）
#     tasks           {task_index: task 字符串}
#     task_to_task_index  上面那个的反向表（字符串 → 下标）
#     episodes        {episode_index: {"length":…, "tasks":…}}
#     episodes_stats  {episode_index: {feature: {min/max/mean/std/count}}}
#     stats           全数据集的聚合统计量（v2.1 从 episodes_stats 聚合而来）
#     annotations     可选，meta 同级 annotations/*.jsonl（本仓库扩展）
#
# 【为什么单独拆一个类】
#   下游（BaseLerobotDataset、MixtureLerobotDataset）经常只需要 fps / features /
#   total_episodes 这些信息就能把配置和 shape_meta 对上，完全不需要打开 parquet 或
#   解码视频。这个类就是那个“轻量入口”，构造代价极低。
# =============================================================================
class LeRobotDatasetMetadata:
    # -------------------------------------------------------------------------
    # 构造：定位本地目录 → 尝试读 meta → 读不到就从 HuggingFace Hub 拉 meta/
    # -------------------------------------------------------------------------
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        # repo_id 在本仓库里通常就是**本地数据集目录路径**（BaseLerobotDataset 里
        # 直接传 ds_dir 当 repo_id），此时 root 也等于同一个路径。
        self.repo_id = repo_id
        # revision 只在走 Hub 时才有意义；不传就用格式版本号当默认 revision。
        self.revision = revision if revision else CODEBASE_VERSION
        # 没有显式给 root 时，按 HF 缓存约定拼出 <HF_LEROBOT_HOME>/<repo_id>。
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        # 【读 meta 的两条路】
        #   force_cache_sync=True 时故意抛 FileNotFoundError，强制走“从 Hub 重新拉
        #   meta/”的分支（用于确保本地 meta 与远端一致）。
        #   否则先尝试直接读本地 meta；本地不存在（FileNotFoundError /
        #   NotADirectoryError）时才去 Hub 拉。
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            # 下面这几行是上游的“版本兼容/安全版本”逻辑，本仓库已停用（也没有 Hub
            # 场景），保留注释以备将来需要时恢复。
            # if is_valid_version(self.revision):
            #     self.revision = get_safe_version(self.repo_id, self.revision)

            # 只拉 meta/ 下的文件（体积很小），不碰 data/ 和 videos/。
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        """把 meta/ 下的各类文件读进内存（构造期调用，读盘但很轻）。"""
        # info.json：全局元信息。load_info 顺手把每个 feature 的 shape 从 list 转成 tuple。
        self.info = load_info(self.root)
        # 上游的版本兼容性检查在本仓库被注释掉了（TODO 也保留着）：
        # 目前不校验 codebase_version 与本地代码版本是否匹配，靠 __init__ 的分支兜底。
        # TODO add new check
        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        # tasks.jsonl → {task_index: task} 以及反向表 {task: task_index}。
        # 这两个表是 __getitem__ 里把 task_index 翻译回自然语言指令的依据。
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        # annotations/ 是本仓库扩展（子任务、场景、夹爪动作等标注），数据集里可能没有，
        # 所以读之前先判断目录是否存在。
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        # episodes.jsonl → {episode_index: {"length":…, "tasks":…}}，"length" 很重要：
        # 它决定 episode_data_index 里每条 episode 占多少帧。
        self.episodes = load_episodes(self.root)
        # 【两代格式的统计量】
        #   v2.0：只有 meta/stats.json（全数据集一份），逐 episode 的统计量用同一份
        #         复制过去（backward_compatible_episodes_stats）——只是为了让下游
        #         “按 episode 取 stats”的代码不改动，精度上是近似的。
        #   v2.1：meta/episodes_stats.jsonl 逐 episode 记录，加载后聚合成 self.stats。
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """从 HuggingFace Hub 把数据集（或部分文件）下载到 self.root，已存在的文件不重复下载。"""
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        # 数据集**创建时**用的格式版本（如 v2.1 / v3.0），不是当前代码的版本。
        # 用 packaging 解析成可比较对象，便于写 `self._version < packaging.version.parse("v2.1")`。
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        """第 ep_index 条 episode 的 parquet 相对路径。

        路径模板存在 info["data_path"] 里，形如
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"，
        用 episode_index 先算出它属于哪个 chunk，再把两个占位符填上。
        返回的是相对路径（相对 self.root），需要拼 self.root 才能读。
        """
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """第 ep_index 条 episode 中相机 vid_key 的 mp4 相对路径。"""
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(
            episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index
        )
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        # 整除即分组：chunks_size=1000 时，episode 0~999 → chunk-000，1000~1999 → chunk-001……
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        # 带占位符的路径模板（不是真实路径），真实路径由 get_data_file_path 生成。
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        # 纯 image 数据集里该字段为 None（创建时 use_videos=False）。
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        # 采集时的真实频率。delta_timestamps（秒）换算成步长、时间戳容差都用它。
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        # 列的声明表：{列名: {"dtype":…, "shape":…, "names":…, "info":…}}。
        # BaseLerobotDataset 的 shape_meta 就是靠它校验 lerobot_key 与维度是否对得上。
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        # 视觉列的两种存储方式：image（图像直接嵌在 parquet 里）与 video（单独 mp4）。
        # 下面的 image_keys / video_keys / camera_keys 就是对它们的三种筛选方式。
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        # 统一转成 tuple，方便和 shape_meta 里的 raw_shape 直接比较。
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        # 下面这一组都是全局计数，写在 info.json 里，save_episode 时会同步更新。
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        # 注意这里查询的是 task_to_task_index（含所有历史 task 的字典）。
        # 本仓库的“task”字段其实打包了 4 个字符串：
        #   [coarse_task, task, coarse_quality, quality]
        # 所以调用处会对 task[0..3] 分别查下标，分别写进
        # coarse_task_index / task_index / coarse_quality_index / quality_index 四列。
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        # 新任务追加到 tasks.jsonl 末尾，下标取当前 total_tasks。
        # 注意落盘范围：这里只把新任务**追加一行**到 meta/tasks.jsonl，并更新内存里的
        # info["total_tasks"]；info.json 要等到 save_episode → write_info 才真正写回。
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        raw_file_name: str | None = None,
    ) -> None:
        """把一条 episode 的元信息与统计量写进 meta/（由 LeRobotDataset.save_episode 调用）。

        写入顺序：info.json（更新总数）→ episodes.jsonl → episodes_stats.jsonl。
        注意本方法只更新 meta，不写 parquet/mp4。
        """
        # 更新全局计数：episode 条数、总帧数。
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        # 若这条 episode 落进了新的 chunk 目录，就把 total_chunks 加一
        # （get_episode_chunk 是按 chunks_size 整除算的，所以只有跨组时才会 +1）。
        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        # splits 用于 HF datasets 的切分描述，"0:{n}" 表示前 n 条都算 train。
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        # 每条 episode 会为每个视频列产生一个 mp4，所以 total_videos 按相机数累加。
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            # 从第一条 episode 的 mp4 里读出分辨率/编码信息，缓存进 features[*]["info"]。
            self.update_video_info()

        # info.json 必须在这里落盘：下次加载时 total_episodes 等计数靠它。
        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        # raw_file_name：本仓库扩展，记录这条 episode 来自原始数据的哪个文件，
        # 便于回溯（上游格式里没有这一项）。
        if raw_file_name is not None:
            episode_dict["raw_file_name"] = raw_file_name

        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        # episodes_stats 逐 episode 记录；stats 是全数据集聚合值，
        # 这里用 aggregate_stats([旧聚合, 新episode]) 增量更新，避免每次重算全量。
        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        # 取第 0 条 episode 的每个视频列，读出分辨率/编码信息写进 features[key]["info"]。
        # 之所以能这么省事：同一次采集里所有视频的编码参数一致（同样是 fps、同样的
        # 分辨率），所以读一条就够；代价是**必须已经有第 0 条 episode 存在**。
        # 已经写过 info 的列会跳过（`if not ...get("info")`），避免重复读 mp4。
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        # 【空数据集工厂】不走 __init__（避免去读还不存在的 meta），
        # 而是 __new__ 出一个空壳，手工把各字段摆好，供录制流程使用。
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        # exist_ok=False：目录已存在就直接报错，防止误覆盖已有数据集。
        obj.root.mkdir(parents=True, exist_ok=False)

        # DEFAULT_FEATURES 是底层 8 个自有列（timestamp/frame_index/...）。
        # 这里把业务 features 与它们合并：调用方只声明业务列即可。
        # TODO(aliberts, rcadene): implement sanity check for features
        features = {**features, **DEFAULT_FEATURES}
        # 目前只校验列名里不能含 "/"（datasets 的 feature 名限制）。
        _validate_feature_names(features)

        # 任务表/统计量/episode 表都从空的开始，后续 add_task / save_episode 往里填。
        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        # info.json 的初始骨架（total_* 全为 0，路径模板按 use_videos 决定）。
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION, fps, features, use_videos, robot_type
        )
        # 声明了 video 列却说不存视频 → 矛盾，直接报错（上游也是这么做的，没写错误信息）。
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        # 还没推过 Hub，没有 revision 概念。
        obj.revision = None
        return obj


# =============================================================================
# 类 2/3：LeRobotDataset —— 单个数据集目录的读取（训练时真正被用到的那个）
# =============================================================================
# 【职责】对外表现为一个 PyTorch Dataset：
#     len(ds)      = 选中 episode 的总帧数（全局帧下标空间）
#     ds[idx]      = 第 idx 帧的样本 dict
# 内部把两件事缝在一起：
#     · hf_dataset（datasets.Dataset）→ 读 parquet 里的数值/文本列
#     · videos（mp4）                → 按时间戳解码相机帧，与 parquet 行严格对齐
#
# 【样本（单帧）里都有什么】
#   parquet 自带列：timestamp / frame_index / episode_index / index /
#                   coarse_task_index / task_index / coarse_quality_index / quality_index
#   业务列        ：action、observation.state、observation.images.*（image 列直接是张量）
#   delta 查询列  ：delta_timestamps 里声明的列会额外返回一个「帧窗口」张量
#                   （形状 [查询帧数, 原始维度]），例如 action: [H, 7]
#   掩码          ：{key}_is_pad，越界帧为 True（见文件头概念 (4)）
#   视频解码列    ：video 列在 __getitem__ 里解码后写进 item（形状 [C,H,W] float32）
#   语言字段      ：task / coarse_task / atomic_task / high_level_instruction /
#                   operating_hand（由 *_index 翻译回自然语言字符串）
#
# 【与上游的差异（本仓库改动点，看到时不用怀疑）】
#   · 无需 Hub：root/repo_id 都是本地路径，下载分支基本不会触发；
#   · 新增 load_images 开关：False 时不返回视觉列、也不解码视频；
#     配套的 _hf_dataset_without_images / _get_hf_dataset_for_reads 负责“去掉图像列”的视图；
#   · 新增 during_training 开关：False 时 __getitem__ 跳过视频解码；
#   · __getitem__ 里额外翻译 coarse_task / atomic_task / high_level_instruction /
#     operating_hand 等 CoT 需要的语言字段；
#   · _query_hf_dataset_fast：同一批帧下标会被多个列复用，只 select 一次（性能优化）；
#   · get_episode_data：整条 episode 一次读出来（供统计量/初始化位置等使用）；
#   · is_compute_episode_stats_image：算 episode 统计量时是否包含图像列。
# =============================================================================
class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        load_images: bool = True,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1",
        is_compute_episode_stats_image: bool = True,
    ):
        """
        【中文导读】上游原文档见下方英文；本仓库的实际用法是“本地目录 + 少量 episode 子集”：
          · DS = LeRobotDataset(repo_id=<目录路径>, root=<同一路径>, episodes=None,
                                delta_timestamps={"action": [...], "observation.state": [...]})
            —— root 和 repo_id 指向同一个本地目录（v2.1 只读本地盘）；
          · delta_timestamps 由 BaseLerobotDataset._build_delta_timestamps() 生成，
            单位为秒（如 action 取未来 H 步 → [0, 1/fps, ..., (H-1)/fps]）；
          · 本类只负责“按时间戳把帧读出来”，不做维度切分/归一化（那是上一层的活）。

        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        # ---- 1) 保存构造参数（这些参数决定了后续取数的行为） ----
        # repo_id：本仓库里通常就是数据集目录路径；root 缺省时按 <HF_LEROBOT_HOME>/<repo_id> 拼。
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        # image_transforms：对**所有**视觉列统一施加的图像变换（torchvision.transforms.v2），
        # 只影响读出来的图，不改磁盘上的数据。
        self.image_transforms = image_transforms
        # delta_timestamps：用秒表达的相对采样计划；下面会换算成整数步长 self.delta_indices。
        self.delta_timestamps = delta_timestamps
        # episodes：只加载这些 episode_index；None = 全部加载。
        # 注意它会影响三处：读哪些 parquet 文件、episode_data_index 的范围、
        # 以及 __getitem__ 里“episode_index → 在选中列表中的位置”的换算。
        self.episodes = episodes
        # tolerance_s：时间戳容差（秒）。同一条 episode 内相邻帧的时间差应当是 1/fps±tolerance_s；
        # 视频解码后按时间戳找帧也用这个容差。容差太小 → 因数值/编码误差误报；
        # 太大 → 可能匹配到相邻帧。BaseLerobotDataset 默认按 0.4/fps 传入。
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        # load_images：只读 state/action 的场景（统计量、索引校验）把它设为 False，
        # 既省掉图像列的反序列化，也省掉视频解码。
        self.load_images = load_images
        # 视频解码后端：不指定就选本机可用的（torchcodec → pyav → video_reader）。
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        # 录制时编码 mp4 用的编码器（默认 libsvtav1，体积小但编码慢）。
        self.video_codec = video_codec
        # 算 episode 统计量时是否把图像列也算进去（False 可省大量时间）。
        self.is_compute_episode_stats_image = is_compute_episode_stats_image
        # delta_indices 在下面“Setup delta_indices”处填充；None 表示不做 delta 查询，
        # 此时 __getitem__ 只返回单帧数据。
        self.delta_indices = None
        # during_training 只对本类的 __getitem__ 生效：False 时不解码视频。
        # MixtureLerobotDataset.set_during_training() 会统一设置它
        # （推理/评估场景常关掉视频解码以省时间）。
        self.during_training = True
        # load_images=False 时用的“去掉图像列”的 hf_dataset 视图（懒构造，见 _refresh_hf_dataset_views）。
        self._hf_dataset_without_images = None

        # 下面两个属性是为**录制路径**（add_frame/save_episode）准备的状态：
        #   image_writer   —— 上游的异步图像写入器，本仓库未启用，恒为 None；
        #   episode_buffer —— 录制时逐帧累积的 buffer，训练时不用。
        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        # 目录不存在就建出来（读不了数据时 download_episodes 会往这里写）。
        self.root.mkdir(exist_ok=True, parents=True)

        # ---- 2) 读元信息（info/features/tasks/episodes/stats），此时还没碰任何真实数据 ----
        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # image 列（图像嵌在 parquet 里）单独记一份：load_images=False 时需要把它们摘掉。
        # 注意 video 列不在这里 —— video 列本来就不在 parquet 里。
        self._image_columns = [key for key, ft in self.features.items() if ft["dtype"] == "image"]
        # 只选了部分 episode 时，统计量要按这些 episode 重新聚合（否则会带上没选的 episode）。
        # 注意：只有「指定了 episodes 且格式是 v2.1」时才会给本对象挂 self.stats；
        # 其余情况（episodes=None，或 v2.0 老格式）本类**不**设置 self.stats ——
        # 训练侧用的是 MultiLeRobotDataset.stats（它统一聚合各子数据集的 meta.stats），
        # 所以这里不要误以为 self.stats 总会存在。
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # ---- 3) 读真实数据：本地文件齐全就直接读，否则先下载再读 ----
        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            # 逐个检查“本实例需要的每个 parquet/mp4 文件”是否存在——注意用的是
            # get_episodes_file_paths()（受 self.episodes 与 load_images 影响），
            # 所以只要需要的文件齐了，就不会触发任何下载。
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            # load_hf_dataset 只扫 parquet（不读内容），把 transform 设成 hf_transform_to_torch。
            self.hf_dataset = self.load_hf_dataset()
            # 按 load_images 决定是否再造一个“无图像列”的视图。
            self._refresh_hf_dataset_views()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            # 文件缺失（或 force_cache_sync）→ 从 Hub 拉一遍再读。本仓库的本地场景几乎
            # 不会走到这里；真的走到说明数据集目录不全，会直接报网络/路径错误。
            # self.revision = get_safe_version(self.repo_id, self.revision)
            self.revision = CODEBASE_VERSION
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
            self._refresh_hf_dataset_views()

        # ---- 4) 建立 episode ↔ 全局帧下标的索引表 ----
        # 传 self.episodes 时，表里只包含选中的 episode（顺序与 self.episodes 一致），
        # 因此“第 i 条选中的 episode”对应索引 i —— __getitem__ 里的
        # self.episodes.index(ep_idx) 就是在做这层换算。
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # ---- 5) 校验时间戳 ----
        # 把 parquet 的 timestamp / episode_index 两列整列取出（stack 成 [N] 张量），
        # 检查同一条 episode 内相邻帧间隔是否都是 1/fps（允许 tolerance_s 误差）。
        # 这一步是“数据可用性”的最后一道闸：时间戳乱了，后面按时间戳解码视频就会错帧。
        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        # 校验函数要的是 numpy 数组，这里把 from/to 两个张量转过去。
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(
            timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
        )

        # ---- 6) 把 delta_timestamps（秒）换算成 delta_indices（整数步长） ----
        # 校验：每个偏移必须是 1/fps 的整数倍（在容差内），否则换算成步长会有歧义。
        # 换算：get_delta_indices 用 round(秒 * fps) 得到步长，例如
        #       0.0 → 0、1/30 → 1、-0.5（fps=30）→ -15。
        # 之后 __getitem__ 用 idx + delta 得到要取的帧号。
        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    # -------------------------------------------------------------------------
    # 上传 / 下载（本仓库几乎不用，读代码时可先略过）
    # -------------------------------------------------------------------------
    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        """把整个数据集目录传到 HuggingFace Hub（录制完数据集后使用）。"""
        # images/ 是录制期的临时图像目录，永远不上传。
        ignore_patterns = ["images/"]
        if not push_videos:
            # push_videos=False 时连 videos/ 也不传（例如只想传 parquet + meta 做索引）。
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        # exist_ok=True：仓库已存在就复用，不报错。
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            # 指定分支时先从当前 revision 建出分支，再往分支上推。
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        # 文件数非常多时用 upload_large_folder（会分片并发上传，但需要 git-lfs 支持）。
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        # 没有 README（dataset card）时自动生成一个，里面带上 info.json 的元信息。
        if not hub_api.file_exists(
            self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch
        ):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        # 给这次上传打一个与格式版本同名的 tag（例如 v2.1），方便按版本号取数据。
        # 先删旧 tag 再建（旧 tag 不存在时 RevisionNotFoundError 会被 suppress 掉）。
        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(
                self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset"
            )

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """从 Hub 下载到 self.root；已存在的文件会被 snapshot_download 自动跳过。"""
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # 注意这个方法只在“本地文件不全”时才会被 __init__ 调到。
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        # files=None → 下载整个仓库；指定 episodes 时只下载这些 episode 的 parquet/mp4。
        files = None
        # 只有在“要视频 且 要图像”时才下载 videos/；否则忽略整个 videos/ 目录。
        ignore_patterns = None if (download_videos and self.load_images) else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        """列出本实例需要的所有文件（相对路径）：每个 episode 的 parquet +（按需）每个相机的 mp4。

        被两处使用：
          · __init__ 里判断“本地文件是否齐全，要不要下载”；
          · download_episodes 里作为 allow_patterns。
        注意返回的是相对 self.root 的路径（Path 对象），不是绝对路径。
        """
        # 没指定 episodes 就是全量；指定了就只有那几条。
        episodes = (
            self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        )
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # 只有“存在 video 列”且“本次需要图像”时才需要 mp4 文件。
        # load_images=False 的场景不会解码视频，所以别把 mp4 也算进“必需文件”。
        if len(self.meta.video_keys) > 0 and self.load_images:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        # 【hf_dataset 是什么】datasets.Dataset 对象：把 parquet 当成一张大表，
        # 索引 idx 就是全局帧下标（所有 episode 首尾相接后的下标），和 __getitem__ 的
        # idx 语义一致。它只做“读 + 类型转换”，不做任何维度切分。
        # 【两条加载路径】不指定 episodes → 整个 data/ 目录；指定 episodes → 只列那些文件。
        # 注意这里都是懒加载（只读 schema/元数据，真正取数在 __getitem__ 时发生）。
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # 【关键一步】设置 transform：每个样本从 pyarrow 出来时都会被 hf_transform_to_torch
        # 处理一遍 —— 数值列 → torch.Tensor，image 列（PIL/bytes）→ [C,H,W] float32（0~1），
        # 字符串列保持 str。所以 __getitem__ 里拿到的 item 已经是张量了。
        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        """录制新数据集时创建一个空的 hf_dataset（只有 schema，没有数据）。"""
        # get_hf_features_from_features 把 info["features"] 翻译成 datasets.Features
        # （数值 → Value/Sequence，图像 → datasets.Image，video 列会被跳过，因为它们存在
        #  mp4 里、不进 parquet）。
        features = get_hf_features_from_features(self.features)
        # 建一个“列齐全但每个列表都为空”的 Dataset，之后 save_episode 会往上 append。
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _refresh_hf_dataset_views(self) -> None:
        """按 load_images 重新生成“无图像列视图”。

        为什么需要它：load_images=False 时我们不想让 datasets 去反序列化图像列
        （图像列的解码/转 tensor 很贵），于是做一个删掉这些列的 Dataset 视图，
        读数据时统一走 _get_hf_dataset_for_reads()。
        注意 remove_columns 返回的是**新对象**，不会改动 self.hf_dataset。
        """
        self._hf_dataset_without_images = None
        if self.hf_dataset is None:
            return
        # 只有真的存在于当前 parquet 里的 image 列才需要摘（video 列不在 parquet 里）。
        removable_columns = [
            col for col in self._image_columns if col in self.hf_dataset.column_names
        ]
        if not removable_columns:
            # 没有图像列 → 视图与原数据集等价，直接共用同一个对象。
            self._hf_dataset_without_images = self.hf_dataset
            return
        self._hf_dataset_without_images = self.hf_dataset.remove_columns(removable_columns)
        # remove_columns 之后 transform 会丢，需要重新设一次。
        self._hf_dataset_without_images.set_transform(hf_transform_to_torch)

    def _is_visual_key(self, key: str) -> bool:
        # 视觉列 = image 或 video（两种存储方式都算）。
        feature = self.features.get(key)
        return feature is not None and feature["dtype"] in {"image", "video"}

    def _get_hf_dataset_for_reads(self) -> datasets.Dataset:
        """读数据时该用哪个 hf_dataset：要图像 → 原数据集；不要图像 → 无图像视图。"""
        if self.load_images or self._hf_dataset_without_images is None:
            return self.hf_dataset
        return self._hf_dataset_without_images

    def _episode_read_columns(self) -> list[str]:
        """整条 episode 读取时需要的列：排除 video 列（在 mp4 里），再按 load_images 决定是否排除 image 列。"""
        return [
            key
            for key, ft in self.features.items()
            if ft["dtype"] != "video" and (self.load_images or ft["dtype"] != "image")
        ]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        # 选中 episode 的总帧数 = len(hf_dataset)（hf_dataset 本身就只包含选中的 episode）。
        # hf_dataset 还没建时退回 meta 里的全量帧数。
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        # 已经建好 hf_dataset 就用它自带的 Features（含 datasets.Image 等类型信息）；
        # 录制路径下 hf_dataset 可能是空的，则按 features 现算一份。
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    # -------------------------------------------------------------------------
    # 取样相关的核心方法（__getitem__ 会依次调用它们）
    # -------------------------------------------------------------------------
    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """把「当前帧 + delta 步长」换算成真正要读的帧下标，并生成越界掩码。

        Args:
            idx: 当前帧的**全局**下标（就是 __getitem__ 收到的 idx）。
            ep_idx: 当前帧所属 episode 在 episode_data_index 里的**位置**
                （指定了 episodes 子集时，它是“在选中列表里的第几条”，不是真实 episode_index）。

        Returns:
            (query_indices, padding)
            query_indices: {列名: [要读的帧下标, ...]}，长度 = 该列的 delta 个数；
            padding:       {"{列名}_is_pad": BoolTensor[同长度]}。

        关键点（clamp + pad 的配合）：
          · 越界（idx+delta 落在 [ep_start, ep_end) 之外）时**不报错**，而是 clamp 到
            [ep_start, ep_end-1]（区间的第一帧 / 最后一帧），保证读到的帧一定合法；
          · 同时把该位置记成 True 放进 is_pad，训练侧据此把它们从 loss 里剔除，
            所以 clamp 出来的“假帧”不会影响训练目标。
          · 之所以 clamp 而不是报错：episode 末尾请求未来 N 步 action 是常态
            （episode 只剩 3 帧却要 32 步），不这么做就没法构造训练样本。

        例：episode 占全局帧 [1000, 1040)，当前 idx=1038，delta=[0,1,2,3]：
            query_indices → [1038, 1039, 1039, 1039]
            is_pad        → [False, False, True, True]
        """
        # 当前 episode 的帧区间 [from, to)（张量标量，用 .item() 取 Python int）。
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        # 逐列展开 delta 步长；max/min 就是上面的 clamp。
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        # 掩码的依据是**未 clamp 的**原始帧号是否越界（< ep_start 或 >= ep_end）。
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """计算每个视频列要解码的**时间戳**列表（秒）。

        mp4 解码是按时间戳找帧的，所以不能直接用帧下标，得先查 parquet 的 timestamp 列。
        · 该视频列也在 delta_timestamps 里（query_indices 有它）→ 逐帧查它的 timestamp；
        · 否则 → 只解码当前帧（列表长度为 1）。
        返回的列表长度决定了 _query_videos 解出几帧：长度 1（单帧）时后面会把帧维 squeeze 掉。
        """
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                # 注意用的是 self.hf_dataset（含全部列），不是无图像视图：
                # 只要 timestamp 一列，和图像开关无关。
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """（逐列 select 的慢版本，目前只在需要时保留）按帧下标批量取回各列的窗口张量。

        对每个列单独 select 一次，同一批下标在不同列之间会重复扫表；
        实际 __getitem__ 走的是下面的 _query_hf_dataset_fast。
        """
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
        """按帧下标取回各列的窗口张量（带去重缓存）。

        相比 _query_hf_dataset 的两点优化：
          1. 相同下标序列只 select 一次；不同列若共享同一组 delta（如 state 与 action
             用同一时间轴），可以直接复用上一次 select 的结果，避免重复扫 parquet；
          2. 跳过不需要的列：video 列本来就不在 parquet 里；load_images=False 时
             image 列也一并跳过（顺带避免解码图像）。

        返回 {列名: Tensor[查询帧数, 原始维度]}，注意返回的是**原始维度**，
        真正按 shape_meta 切维度是上一层 BaseLerobotDataset 的事。
        """
        result = {}
        # 用 tuple(下标列表) 做缓存键（list 不可哈希）。
        processed_indices = set()
        index_to_selected = {}
        source_dataset = self._get_hf_dataset_for_reads()
        for key, q_idx in query_indices.items():
            # 视频列在 mp4 里，这里跳过；不要图像时不取 image 列。
            if key in self.meta.video_keys or (not self.load_images and self._is_visual_key(key)):
                continue
            q_idx_tuple = tuple(q_idx)
            if q_idx_tuple not in processed_indices:
                # 这一批下标（一次 select 拿到所有列）缓存下来，供后面共享同一批下标的列复用。
                selected_data = source_dataset.select(q_idx)
                index_to_selected[q_idx_tuple] = selected_data
                processed_indices.add(q_idx_tuple)
            else:
                selected_data = index_to_selected[q_idx_tuple]
            result[key] = torch.stack(selected_data[key])
        return result

    # -------------------------------------------------------------------------
    # 整条 episode 读取（供上一层做统计量 / 初始化位置等用）
    # -------------------------------------------------------------------------
    # no videos
    def get_episode_data(self, episode_id: int) -> dict:
        """一次把第 episode_id 条 episode 的**全部帧**读成一个 dict（不含视频列）。

        Args:
            episode_id: episode 在 episode_data_index 里的位置（0 起，非 episode_index 本身）。

        Returns:
            {列名: Tensor[T, D]}，T = 该 episode 的帧数；video 列不返回（注释里的 "no videos"），
            load_images=False 时 image 列也不返回。

        与 __getitem__ 的区别：__getitem__ 是“训练用的一帧 + 时间窗口”，
        本方法是“整段轨迹”，调用方（BaseLerobotDataset._get_episode_data）会再按
        sliding window 展开成训练样本。
        """
        # [from, to) 半开区间 → 用 range 生成该 episode 的全部全局帧下标。
        ep_start = self.episode_data_index["from"][episode_id].item()
        ep_end = self.episode_data_index["to"][episode_id].item()
        q_idx = list(range(ep_start, ep_end))
        # 一次性 select 整段（比 __getitem__ 逐帧取快得多），再逐列 stack 成 [T, D]。
        selected_data = self._get_hf_dataset_for_reads()[q_idx]
        res = {key: torch.stack(selected_data[key]) for key in self._episode_read_columns()}
        return res

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        # 【上游警告的中文说明】不要在**主进程**里调用本方法（例如为了调试再建一个
        # num_workers=0 的 DataLoader）。视频解码器会持有内存引用，主进程先建引用、
        # 子进程再访问时可能直接 Segmentation Fault。要调试就单独写脚本读。
        #
        # 入参 query_timestamps：{视频列: [时间戳秒, ...]}，由 _get_query_timestamps 给出。
        # decode_video_frames 返回 [T, C, H, W] float32(0~1)；每个时间戳解出一帧：
        #   · 只查 1 个时间戳（T=1）→ squeeze(0) 去掉帧维，得到 [C, H, W]；
        #   · 查多个时间戳（T>1）→ squeeze 不改变形状，仍是 [T, C, H, W]。
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            # 精确到“时间戳 + 容差”地找帧：容差不够会取到相邻帧或直接报错。
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        """把 {key}_is_pad 掩码写进样本（保持与上层约定的键名）。"""
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        # 长度 = 选中 episode 的总帧数，因此合法 idx ∈ [0, num_frames)。
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        """取第 idx 帧的样本（idx 是全局帧下标，见文件头概念 (2)）。

        执行顺序（对新手最重要的一段代码）：
          1. 从 hf_dataset 取当前帧的所有列（数值列已是张量，image 列已是 [C,H,W] float32）；
          2. 若配置了 delta_timestamps：算出要读的帧下标 + 越界掩码，取回窗口张量
             （例如 action: [H, D]），并把掩码写进 item；
          3. 若存在视频列且需要图像：按时间戳从 mp4 解码相机帧，写进 item；
          4. 对视觉列施加 image_transforms；
          5. 把 *_index 翻译成自然语言字段（task / coarse_task / atomic_task /
             high_level_instruction / operating_hand）。
        返回的 item 是“原始粒度”的一帧，维度切分/归一化由上层 BaseLerobotDataset 完成。
        """
        # 取当前帧：load_images=False 时这里走的是“无图像列视图”，不会解码图像。
        item = self._get_hf_dataset_for_reads()[idx]
        # 当前帧属于哪条 episode（真实 episode_index，不是列表位置）。
        ep_idx = item["episode_index"].item()

        # ---- 步骤 2：delta 查询（action chunk / 多帧观测都靠这里） ----
        # delta_indices=None 表示没配置时间窗口，此时 query_indices 为 None，
        # 下面视频解码就只取当前帧（与 _get_query_timestamps 的行为对应）。
        query_indices = None
        if self.delta_indices is not None:
            # episode_data_index 是按“选中列表的位置”建的，所以指定了 episodes 子集时
            # 要先把真实 episode_index 换算成列表位置。
            current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
            # query_indices: {列名: [帧下标…]}；padding: {列名_is_pad: BoolTensor}
            query_indices, padding = self._get_query_indices(idx, current_ep_idx)
            # 一次性把各列的窗口读回来（相同下标序列会复用同一次 select）。
            query_result = self._query_hf_dataset_fast(query_indices)
            # 先把掩码并进 item，再覆盖上窗口张量：于是 item[key] 是 [帧数, D] 的窗口，
            # item[f"{key}_is_pad"] 是与它逐帧对齐的掩码。
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # ---- 步骤 3：视频解码 ----
        # 三个条件同时成立才解码：有视频列、允许在训练路径上解码、本次需要图像。
        # 解码用的是 parquet 里的真实时间戳（而非帧号），保证与数值列严格对齐。
        if len(self.meta.video_keys) > 0 and self.during_training and self.load_images:
            current_ts = item["timestamp"].item()
            # 每个视频列要解码的时间戳列表（不在 delta 计划里的列只取当前帧）。
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            # 写成 {**video_frames, **item}：video 列本来就不在 parquet 里，
            # 所以这等价于“把解出的帧并进 item”；若真有同名键，item 里的值优先。
            item = {**video_frames, **item}

        # ---- 步骤 4：统一图像变换 ----
        # 对 image + video 两种视觉列都生效（camera_keys 是二者的并集）。
        # 变换后 dtype/取值范围由 transforms 决定，上层 processor 一般不再重复处理图像。
        if self.image_transforms is not None and self.load_images:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # ---- 步骤 5：语言字段翻译（*_index → 字符串） ----
        # 本仓库把“语言指令”存在 tasks.jsonl 里，parquet 只存下标（省空间）。
        # 上层 BaseLerobotDataset 直接用这些字符串字段填 CoT 模板 / 语言 token。
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        # 下面几个字段是本仓库的扩展列，不是每个数据集都有，所以逐个判断存在性；
        # 取下标时统一 .item() 转成 Python int 再查表。
        if "coarse_task_index" in item:
            coarse_task_index = item["coarse_task_index"].item()
            item["coarse_task"] = self.meta.tasks[coarse_task_index]

        # 操作手（左/右/双手）也复用 tasks 表：录制时把 "left"/"right" 这类字符串
        # 当普通 task 注册进 tasks.jsonl，这里再翻译回来。
        if "operating_hand_index" in item:
            operating_hand_index = item["operating_hand_index"].item()
            item["operating_hand"] = self.meta.tasks[operating_hand_index]

        # 原子子任务（atomic_task）与高级指令（high_level_instruction）同理；
        # 这两列允许为空（缺失或 None 时跳过），因此判断里带了 `is not None`。
        # robocoin subtask_annotation alias was removed on 2026-05-08 and can be
        # restored after further investigation.
        # atomic_task_index decodes into the standalone item["atomic_task"] field.
        if "atomic_task_index" in item and item["atomic_task_index"] is not None:
            atomic_task_index = item["atomic_task_index"].item()
            # int(...) 是因为某些数据源把下标存成了浮点（如 3.0），查表会失败。
            item["atomic_task"] = self.meta.tasks[int(atomic_task_index)]

        # to support high level instruction
        if (
            "high_level_instruction_index" in item
            and item["high_level_instruction_index"] is not None
        ):
            high_level_instruction_index = item["high_level_instruction_index"].item()
            item["high_level_instruction"] = self.meta.tasks[int(high_level_instruction_index)]

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    # -------------------------------------------------------------------------
    # 录制路径（采集数据时用；纯训练场景不会调用这一组方法）
    # 流程：create() 建空数据集 → 逐帧 add_frame() 攒进 episode_buffer
    #       → save_episode() 一次性落盘（parquet + mp4 + meta）
    # -------------------------------------------------------------------------
    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        """建一个空的 episode 累加器：{列名: []}，逐帧往里 append。

        两个特殊键不在 features 里：
          · "size"：已累积多少帧（同时充当下一个 frame_index）；
          · "task"：逐帧的 4 元组语言标注列表（见 add_frame 的说明）。
        """
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            # episode_index 是整条 episode 共用的常量，先填好；其余列都是逐帧追加的列表。
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        """录制期临时图像的路径（images/<key>/episode_XXXXXX/frame_XXXXXX.jpeg）。

        这些 png/jpeg 只是“编码 mp4 之前的中间产物”，save_episode 末尾会把整个 images/
        目录删掉，所以它不会出现在最终数据集里。
        """
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    # def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image | bytes, fpath: Path) -> None:
    #     if self.image_writer is None:
    #         if isinstance(image, torch.Tensor):
    #             image = image.cpu().numpy()
    #         write_image(image, fpath)
    #     else:
    #         self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict, task: List[str], timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        """（中文补充）本方法只把一帧追加进内存里的 episode_buffer，磁盘上什么都不写：
          视觉列只记一个“临时图像路径”字符串（真正的落盘在 save_episode → 编码 mp4）；
          数值列直接 append。攒完一条 episode 后必须调用 save_episode() 才真正落盘。

        Args:
            frame: {列名: 值}，必须覆盖 features 里除 DEFAULT_FEATURES 之外的所有业务列。
            task: 长度恰好为 4 的字符串列表，本仓库约定为
                  [coarse_task, task, coarse_quality, quality]。
            timestamp: 该帧时间戳（秒）；不传就按 frame_index / fps 自动推算。
        """
        # Convert torch to numpy if needed
        # 注意这里的断言信息（"must be of two elements"）与实际条件（长度 4）不一致，
        # 是历史遗留的文案问题，不要被它误导。
        assert len(task) == 4, "Task frame must be of two elements"
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                # 统一转 numpy：后续 validate 与 parquet 写盘都按 numpy 处理。
                frame[name] = frame[name].numpy()

        # 校验“列齐全、dtype/shape 与 features 声明一致”，提前暴露录制侧的错误。
        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # frame_index 直接取自缓冲区当前大小，因此调用顺序即帧顺序。
        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            # 默认时间戳：第 n 帧 = n / fps，保证 check_timestamps_sync 一定通过。
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # 逐列写入缓冲区
        # Add frame features to episode_buffer
        for key in frame:
            # 传了 features 里没有的列 → 直接报错（拼错列名是最常见的坑）。
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            # 视觉列只记“临时图像路径”字符串（图像的真正写盘在 _save_image，本仓库已注释掉，
            # 也就是说当前这份代码不会把帧写成 png，需要外部先把图写好后传路径）。
            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"],
                    image_key=key,
                    frame_index=frame_index,
                )
                if frame_index == 0:
                    # 第一帧时顺手建目录（images/<key>/episode_XXXXXX/）。
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                # 对应上游的 _save_image(frame[key], img_path)：把图像写到 img_path。
                # 本仓库禁用了异步图像写入，若将来启用录制需自行恢复。
                # self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        # 帧计数 +1（下一帧的 frame_index 就是它）。
        self.episode_buffer["size"] += 1

    def save_episode(
        self, episode_data: dict | None = None, raw_file_name: str | None = None
    ) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        """（中文补充）一条 episode 的完整落盘流程：
          1. 校验缓冲区 → 取出 size/task → 补 index / episode_index 两列；
          2. 把新出现的 task 字符串注册进 tasks.jsonl，并把 4 元语言标注翻译成 4 个 *_index 列；
          3. 逐列 stack 成 numpy，写 parquet（_save_episode_table）；
          4. 算 episode 统计量（compute_episode_stats）；
          5. 把临时图像编码成每个相机的 mp4（encode_episode_videos）；
          6. 写 meta（episodes.jsonl / episodes_stats.jsonl / info.json）；
          7. 一致性检查（时间戳同步、文件数量），删掉临时 images/，重置缓冲区。
        """
        # 【已知脆弱点】只有 episode_data=None 时才会给 episode_buffer 赋值，
        # 因此传了 episode_data 又走到 validate_episode_buffer 会因未定义变量而 NameError。
        # 实际调用方（录制脚本）都走默认值，所以上游一直没修。
        if not episode_data:
            episode_buffer = self.episode_buffer

        # 校验：必须存在（有帧）、且列/dtype 与 features 一致。
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size / task 是缓冲区的记账字段，不属于 features，取出来后就 pop 掉，
        # 免得后面被当成列写进 parquet。
        #   episode_length：本条 episode 的帧数
        #   tasks：逐帧的 4 元组列表（长度 = episode_length）
        #   episode_tasks：把所有帧的 4 个字符串摊平去重 —— 需要注册进 tasks 表的全集
        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set([item for sublist in tasks for item in sublist]))
        episode_index = episode_buffer["episode_index"]

        # index：全局帧下标，从 total_frames 往后连续排（__getitem__ 的 idx 就是它）。
        # episode_index：整条 episode 共用的常量，展开成长度 episode_length 的数组。
        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # 新任务才注册（get_task_index 返回 None 说明 tasks.jsonl 里还没有这个字符串）。
        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # 把逐帧的 4 元语言标注翻译成 4 个下标列（列名与 DEFAULT_FEATURES 对应）。
        # 训练时 __getitem__ 再把这些下标翻译回字符串。
        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["coarse_task_index"] = np.array(
            [self.meta.get_task_index(task[0]) for task in tasks]
        )
        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task[1]) for task in tasks]
        )
        episode_buffer["coarse_quality_index"] = np.array(
            [self.meta.get_task_index(task[2]) for task in tasks]
        )
        episode_buffer["quality_index"] = np.array(
            [self.meta.get_task_index(task[3]) for task in tasks]
        )

        # 其余列逐列 stack 成 [T, …] 的 numpy 数组，供 parquet 写盘使用。
        for key, ft in self.features.items():
            # 跳过：上面已处理的下标列，以及视觉列（它们通过 mp4 + 图像路径间接处理）。
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in [
                "index",
                "episode_index",
                "coarse_task_index",
                "task_index",
                "coarse_quality_index",
                "quality_index",
            ] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # 3) 落 parquet：把 buffer 里需要的列拼成 datasets.Dataset 再写 .parquet。
        #    （_wait_image_writer 是上游异步写图的等待，本仓库 image_writer 恒为 None，等于空操作。）
        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        # 4) 统计量：min/max/mean/std/count，供归一化使用。
        #    is_compute_episode_stats_image=False 时跳过图像列（省大量时间）。
        ep_stats = compute_episode_stats(
            episode_buffer, self.features, self.is_compute_episode_stats_image
        )

        # 5) 编码视频：把 images/<key>/episode_XXXXXX/ 下的一串帧压成 mp4，
        #    并把生成的视频路径写回 episode_buffer（save_episode 之后的日志/检查会用到）。
        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # 6) 写 meta。注释里的顺序很重要：必须在视频编码**之后**调用，
        #    因为 meta.save_episode → update_video_info 要去读第 0 条的 mp4 拿分辨率信息。
        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, raw_file_name
        )

        # 7) 一致性自查：
        #    a) 时间戳间隔是否与 fps 一致（episode_data_index 只算刚存的这一条）；
        #    b) 磁盘上的 mp4 / parquet 文件个数是否与 episode 数吻合 —— 这条断言能抓出
        #       “视频漏编码 / parquet 漏写”之类的低级错误，非常值钱。
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        # 文件数断言：mp4 数 = episode 数 × 相机数；parquet 数 = episode 数。
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # 清理录制期的临时图像目录（最终数据集里只保留 parquet + mp4 + meta）。
        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        # 重置缓冲区，准备录下一条 episode。
        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        """把一条 episode 的数值列写成 <root>/data/chunk-XXX/episode_XXXXXX.parquet。"""
        # 只挑 hf_features 里声明的列（这样与 parquet 的 schema 严格一致）。
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(
            episode_dict, features=self.hf_features, split="train"
        )
        # 图像列在 datasets 里是 PIL/bytes；embed_images 把它们编码成 parquet 内嵌的字节。
        ep_dataset = embed_images(ep_dataset)
        # 同步更新内存里的全量视图（录制过程中数据集会不断变长）。
        # 注意 concatenate_datasets 会产生新对象，所以后面要重设一次 transform。
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        # chunk 目录可能还不存在，先建父目录。
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        """丢弃当前未保存的 episode（连临时图像一起清掉），并把缓冲区重置为空。"""
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            # 只有启用异步图像写入时才会在磁盘上留下临时图，需要清理。
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    # def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
    #     if isinstance(self.image_writer, AsyncImageWriter):
    #         logging.warning(
    #             "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
    #         )

    #     self.image_writer = AsyncImageWriter(
    #         num_processes=num_processes,
    #         num_threads=num_threads,
    #     )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be picklable and parallelized.
        """
        # 上游的“异步图像写入器”会持有线程/进程句柄，无法被 pickle，
        # 所以要在塞进 DataLoader（多 worker）之前调一次本方法。
        # 本仓库 image_writer 恒为 None，所以这里等于空操作，numpy/torch 对象本身可 pickle。
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        """（中文补充）批量把**所有** episode 的临时帧编码成 mp4（重编码已有数据时用）。"""
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        """（中文补充）把一条 episode 的临时图像编码成每个相机一个 mp4。

        Returns:
            {视频列名: mp4 的**绝对路径字符串**}（save_episode 会把它写回 buffer）。
        注意 ffmpeg 调用是阻塞的，但它内部已多线程，改异步并不会更快。
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            # 先把路径返回给调用方（哪怕本次因为已存在而跳过编码）。
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # 已编码过就跳过：录制中断后续跑时常用得上。
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # 临时帧目录：images/<key>/episode_XXXXXX/（encode_video_frames 按文件名顺序拼帧）。
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(
                img_dir, video_path, self.fps, overwrite=True, vcodec=self.video_codec
            )

        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1",
        is_compute_episode_stats_image=True,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        """（中文补充）不读任何已有数据，直接在 root 下新建一个空数据集骨架：
        meta/info.json（空计数）+ 空的 hf_dataset。之后用 add_frame/save_episode 往里灌数据。
        与 __init__ 的区别：这里不走“读 meta / 校验时间戳 / 建 episode_data_index”那套流程。
        """
        obj = cls.__new__(cls)
        # 先建元信息骨架（会 mkdir，目录已存在则报错，避免误覆盖）。
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        # 本仓库不启用异步图像写入，恒为 None（见文件头的注释）。
        obj.image_writer = None

        # 上游的异步写图器开关，本仓库已停用。
        # if image_writer_processes or image_writer_threads:
        #     obj.start_image_writer(image_writer_processes, image_writer_threads)

        # 逐帧累积用的缓冲区（add_frame 往里写，save_episode 落盘后清空）。
        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        # 录制模式下这些读路径属性都不存在/为空，保持与 __init__ 的接口一致即可。
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()  # 空表（只有 schema）
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.video_codec = video_codec
        obj.is_compute_episode_stats_image = is_compute_episode_stats_image
        return obj


# =============================================================================
# 类 3/3：MultiLeRobotDataset —— 把多个 LeRobotDataset 目录拼成一个
# =============================================================================
# 【为什么需要它】一次训练往往要混合多个数据源（不同机器人/不同批次采集）。
# 本类把 N 个 LeRobotDataset “首尾相接”成一个大索引空间：
#
#   全局 idx:  0 ── ds0.num_frames-1 │ ds0.num_frames ── ... │  最后一个 ds
#              └──── 数据集 0 ───────┘└──── 数据集 1 ────┘
#
#   __getitem__(idx) 先用累加长度定位到某个子数据集，再用**局部下标**取样本，
#   并额外塞一个 item["dataset_index"]（第几个数据集，0 起）—— 上层
#   BaseLerobotDataset 会根据它来选对应的 embodiment 配置/统计量。
#
# 【注意：上游的“共同列”规则在本仓库被放松了】
#   上游会在 __init__ 里求出各子数据集 features 的交集，并把非共同列从样本里删掉
#   （disabled_features）。本仓库把那段“删列”逻辑注释掉了，只保留“交集为空就报错”
#   这一条硬检查。所以：
#     · disabled_features 恒为空集 → __getitem__ 不删任何列；
#     · features 属性返回的是各子数据集的**并集**；
#     · 每个样本只带「它所属那个子数据集」真正有的列，列集合的差异由上层
#       BaseLerobotDataset 结合 dataset_index 与 shape_meta 处理。
# =============================================================================
class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        load_images: bool = True,
        video_backend: str | None = None,
        in_memory: bool = False,  # accepted for API compat with v3; ignored in v2
    ):
        # 注意 in_memory：v2.1 不支持内存缓存，这里只为了与 v3.0 的签名兼容而收下，
        # 传入后不会生效（BaseLerobotDataset 会把同一个参数同时传给两个版本）。
        super().__init__()
        self.dataset_dirs = dataset_dirs
        # v2.1 里“目录路径”同时充当 repo_id 和 root（见 LeRobotDataset.__init__）。
        ds_roots = [Path(ds_dir) for ds_dir in dataset_dirs]
        ds_names = [ds_dir for ds_dir in dataset_dirs]
        self.ds_names = ds_names
        self.ds_roots = ds_roots
        # 每个子数据集一个时间戳容差；缺省 1e-4（BaseLerobotDataset 会传自己的值进来）。
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(ds_names, 0.0001)
        self.load_images = load_images
        # 【逐个子数据集构造】image_transforms / delta_timestamps 是“统一口径”，
        # 所有子数据集共用同一套（所以不在这里区分）。
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = []
        for ds_root, ds_name in zip(ds_roots, ds_names, strict=True):
            try:
                _dataset = LeRobotDataset(
                    ds_name,
                    root=ds_root,
                    # episodes 是 {目录名: [episode_index, ...]}，用于只加载部分 episode。
                    episodes=episodes[ds_name] if episodes else None,
                    image_transforms=image_transforms,
                    delta_timestamps=delta_timestamps,
                    tolerance_s=self.tolerances_s[ds_name],
                    download_videos=download_videos,
                    load_images=load_images,
                    video_backend=video_backend,
                )
                self._datasets.append(_dataset)
            except Exception as e:
                # 【容错设计】某个目录读失败（路径不存在、meta 损坏、时间戳不同步…）时
                # 只打日志并跳过，不中断整体构造 —— 目的是让“一批数据里有坏目录”也能训练。
                # 代价：失败会静默降级（少了一个数据源），所以别忽略这里打印的 traceback。
                # logging.error(e)
                logging.error(f"Exception while process ds_root: {ds_root}, ds_name: {ds_name}")
                traceback.print_exc()
                continue

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        # 【共同列检查】先求所有子数据集 features 的交集：
        #   · 交集为空 → 直接报错（说明这些目录根本不是同一种机器人/同一套 schema）；
        #   · 交集非空 → 上游会把“非共同列”登记进 disabled_features，再由 __getitem__ 删掉
        #     （原因：PyTorch 默认 collate_fn 要求同一 batch 各样本的键集合一致）。
        # 本仓库把下面那段“登记非共同列”的循环注释掉了，所以 disabled_features 恒为空集、
        # 实际不删任何列；列集合的差异由上层 BaseLerobotDataset 结合 dataset_index 处理。
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        # 【本仓库把下面这段“禁用非共同列”的逻辑注释掉了】因此 disabled_features 恒为空集，
        # 也就是说 features 属性会把各子数据集的列**并集**都报出来（后面的 features
        # property 就是并集写法），但取样本时每个样本只带自己数据集真正有的列。
        # 上层（BaseLerobotDataset）靠 shape_meta + dataset_index 自己处理这种差异，
        # 所以这里不需要再删列。若将来遇到“缺列的样本导致 collate 失败”，
        # 可以恢复这段逻辑来变成“交集语义”。
        # Disable non-common features
        # for ds_name, ds in zip(self.ds_names, self._datasets, strict=True):
        #     # Disable warning of empty extra keys do
        #     extra_keys = set(ds.features).difference(intersection_features)
        #     if extra_keys:
        #         logging.warning(
        #             f"keys {extra_keys} of {ds_name} were disabled as they are not contained in all the "
        #             "other datasets."
        #         )
        #     self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # 【统计量】把所有子数据集的 meta.stats 聚合成一份。
        # TODO 里指出的隐患：不同机器人状态/动作范围差别很大时，合并归一化会互相“拉偏”，
        # 更合理的做法是每个 robot 一套归一化参数（属于上游遗留问题）。
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    def set_during_training(self, during_training: bool):
        """统一设置所有子数据集的 during_training（决定 __getitem__ 是否解码视频）。"""
        for dataset in self._datasets:
            dataset.during_training = during_training

    def set_load_images(self, load_images: bool):
        """统一开关图像读取：BaseLerobotDataset 每次拿到（可能被缓存的）实例后都会调一次，
        保证复用的实例不会残留上一个使用者的开关状态。
        """
        self.load_images = load_images
        for dataset in self._datasets:
            dataset.load_images = load_images

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        # 目录名 → 子数据集下标（__getitem__ 里塞进 item["dataset_index"] 的就是它）。
        return {repo_id: i for i, repo_id in enumerate(self.ds_names)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        # 上游笔误：dict 推导里迭代 dict 得到的是 key，这里应写成 self.repo_id_to_index.items()，
        # 现在调用会抛 ValueError（解包 key）。本仓库上游没有调用它，故保持原样不改。
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        # 只取第一个子数据集的 fps：上游假设所有子数据集 fps 相同（构造时并未强制校验，
        # 所以混合不同 fps 的数据集时要自己小心——delta_timestamps 是按秒换算的，
        # 混合不同 fps 会让“同一秒”对应不同步数）。
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        """所有子数据集 features 的**并集**（同名取最后一个子数据集的定义）。

        注意与 __init__ 里的 intersection_features 不同：那个只用于判断“有没有共同列”，
        这里给上游调用方（BaseLerobotDataset 也用它来校验 shape_meta）看的是完整视野。
        """
        features = {}
        for dataset in self._datasets:
            features.update(
                {k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features}
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        # datasets.Image（parquet 内嵌图像）与 VideoFrame（mp4 视频）都算相机列。
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        # 全局帧数 = 各子数据集帧数之和，也就是 __getitem__ 接受的下标上界。
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 【与构造函数参数的区别】这个属性不是“配置用的容差”，而是“一帧的时长上界”
        # （1/fps - 1e-4）：用于判断“解出来的帧时间戳是否离目标太远”。
        # 构造时传入的 tolerances_s（dict）才是逐子数据集的匹配容差。
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def get_episode_data(self, episode_idx: int) -> dict:
        """按**全局** episode 下标读一整条 episode（返回 {列名: 张量/数组}，不含视频列）。

        逻辑：从头依次减去每个子数据集的 episode 数，找到它属于哪个子数据集和局部下标
        （核心片段见下面注释），然后用 pyarrow 直接读那一个 parquet 文件。
        与 LeRobotDataset.get_episode_data 的差别：这里刻意走 pyarrow 而不是 datasets，
        因为只要“整段列”，直接 read_table 最省事也最快。
        """
        # 逐个子数据集比较：episode_idx < num_episodes 说明落在当前子数据集里，
        # 否则减掉它的数量继续往后找（这是“全局 episode → 子数据集 + 局部下标”的换算）。
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                # dataset.episodes[episode_idx]：局部下标 → 真实 episode_index
                # （只加载了部分 episode 时两者不同）。
                # 注意这里假定 dataset.episodes 是个 list —— BaseLerobotDataset 构造时会
                # 显式传入完整的 episode 列表，所以一定成立；若有人直接以 episodes=None
                # 构造底层实例再来调本方法，会在这一行抛 TypeError。
                file = str(
                    dataset.root / dataset.meta.get_data_file_path(dataset.episodes[episode_idx])
                )
                # 只读需要的列（_episode_read_columns 已排除 video，必要时排除 image）。
                table = pq.read_table(str(file), columns=dataset._episode_read_columns())

                result_dict = {}
                for col_name in table.column_names:
                    col = table[col_name]
                    try:
                        # zero_copy_only=True：尽量不复制内存（列表类型做不到时会抛异常）。
                        np_arr = col.to_numpy(zero_copy_only=True)
                    except Exception:
                        # 走到这里通常是“列的值本身是数组”（如 (14,) 的 state）：
                        # to_numpy 得到 object 数组，需要逐元素 stack 成 [T, 14]。
                        raw = col.to_numpy()
                        np_arr = np.stack(raw) if raw.dtype == object else raw
                    with warnings.catch_warnings():
                        # 【为什么会有这个警告】zero-copy 出来的 numpy 数组是只读的，
                        # torch.from_numpy 会告警“array is not writable”。这里只是看一眼数据，
                        # 不会原地改它，所以把这类警告屏蔽掉，避免刷屏。
                        warnings.filterwarnings(
                            "ignore",
                            message="The given NumPy array is not writable",
                            category=UserWarning,
                        )
                        # 【字符串列特殊处理】parquet 里的 task 字符串不能塞进张量，
                        # 原样返回 numpy 数组（dtype==object 或 str_ 两类都算）。
                        # deal with string in parquet file
                        if np_arr.dtype == "O":
                            result_dict[col_name] = np_arr
                        elif np.issubdtype(np_arr.dtype, np.str_):
                            result_dict[col_name] = np_arr
                        else:
                            # 数值列 → torch.Tensor，与 __getitem__ 的返回类型保持一致。
                            result_dict[col_name] = torch.from_numpy(np_arr)
                return result_dict
            else:
                # 不在当前子数据集里：扣掉它的 episode 数，继续检查下一个。
                episode_idx -= dataset.num_episodes
        # 所有子数据集都扣完还没找到 → 下标越界。
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def __len__(self):
        # 全局帧数（所有子数据集帧数之和）。
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """按**全局**帧下标取样本：先定位子数据集，再用局部下标取，最后补 dataset_index。"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # 【定位算法】沿子数据集累加 num_frames，直到 idx 落在当前区间里。
        # start_idx 记录该子数据集在全局下标空间里的起点，用于换算成本地下标
        # （dataset_num_frames 很小时也可以换成 weights 前缀和 + bisect，
        #  BaseLerobotDataset 就是自己维护了一份 _ds_cumulative_frames 做 O(log N) 查找）。
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            # for-else：循环没 break 就说明 idx 越界（上面已提前检查，这里只是兜底）。
            raise AssertionError(
                "We expect the loop to break out as long as the index is within bounds."
            )
        # idx - start_idx 把全局下标换成子数据集内部下标；子数据集内部仍按自己的
        # episode_data_index / delta 逻辑取数（见 LeRobotDataset.__getitem__）。
        item = self._datasets[dataset_idx][idx - start_idx]
        # 【关键补充字段】dataset_index：告诉上层“这个样本来自第几个目录”，
        # 上层据此选择对应的 shape_meta / 统计量 / embodiment 配置。
        item["dataset_index"] = torch.tensor(dataset_idx)
        # disabled_features 在本仓库恒为空（见 __init__ 里的说明），循环因此不做事。
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Dataset Names: '{self.ds_names}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
