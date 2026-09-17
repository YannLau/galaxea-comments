#!/usr/bin/env python
# =============================================================================
# src/g05/data/lerobot/lerobot_dataset_v3.py — LeRobot v3.0 底层数据集（读盘层）
# =============================================================================
#
# 【这个文件是什么】
#   整个训练数据链路里“离磁盘最近”的一层：把 LeRobot v3.0 格式的数据集目录，
#   按“第几帧 + 时间偏移”读成 PyTorch 张量。它不认识 shape_meta / action chunk /
#   归一化这些训练语义（那些在 base_lerobot_dataset.py 里做），
#   它只保证一件事：给我帧下标和时间偏移，我给出那张帧的原始数值和画面。
#
#   文件里有 3 个类，从下到上分三层：
#
#     LeRobotDatasetMetadata   只读 meta/ 目录（info.json / tasks.parquet /
#                              episodes/*.parquet / stats.json），不碰数据本身
#       ▲
#     LeRobotDataset           一个数据集目录：读 parquet 数值列 + mp4 画面，
#                              同时保留了上游 LeRobot 的“录制/写盘”能力
#       ▲
#     MultiLeRobotDataset      把 N 个数据集目录“首尾拼接”成一条全局帧下标空间
#
# 【在数据链路中的位置】
#
#   configs/data/<task>.yaml
#     │  (Hydra 实例化)
#     ▼
#   MixtureLerobotDataset                 src/g05/data/mixture_lerobot_dataset.py
#     ▼
#   BaseLerobotDataset 子类               src/g05/data/base_lerobot_dataset.py
#     ▼
#   MultiLeRobotDataset  ← 本文件（顶层：多目录拼接 + 全局帧下标）
#     ▼
#   LeRobotDataset       ← 本文件（单目录：帧下标 + 时间偏移 → 张量）
#     ▼
#   LeRobotDatasetMetadata ← 本文件（元数据：有哪些 episode、哪些 feature）
#     ▼
#   parquet（数值列） + mp4 / png（相机画面） + meta/*.json
#
#   上游按版本二选一 import（见 base_lerobot_dataset.py 里的 lerobot_ds_version）：
#     lerobot_ds_version="2.1" → lerobot_dataset.py  （旧路径：逐帧 select）
#     lerobot_ds_version="3.0" → 本文件              （新路径：批量读 + in_memory）
#   两个文件对外接口同名同形（LeRobotDatasetMetadata / LeRobotDataset /
#   MultiLeRobotDataset），所以上游可以无感切换；差别只在内部读法。
#
# 【v3.0 的目录结构】（一个数据集目录 = 一个 repo_id / root）
#
#   <root>/
#   ├── meta/
#   │   ├── info.json        fps、features（列名/形状/dtype）、total_episodes、
#   │   │                    chunks_size、data_path / video_path 模板 ...
#   │   ├── tasks.parquet    字符串表：index = 文本，列 task_index = 整数
#   │   ├── episodes/        每条 episode 一行：帧区间 dataset_from_index / to_index、
#   │   │                    data/chunk_index + file_index、videos/<key>/from_timestamp ...
#   │   ├── stats.json       归一化统计量（min/max/mean/std/分位数）
#   │   └── annotations/     可选：subtask / scene / gripper 等文本标注
#   ├── data/chunk-000/file-000.parquet              帧级数值列（每帧一行）
#   └── videos/<camera_key>/chunk-000/file-000.mp4   相机画面（多条 episode 顺序拼在一个 mp4 里）
#
# 【新手先记住的 5 个概念】
#
#   (1) 四种“下标”，别混
#         frame_index    帧在“本 episode 内”的序号（0,1,2,...）
#         episode_index  第几条 episode（在数据集目录内唯一）
#         index          帧在“整个 dataset 内”的全局序号，是 parquet 里的主键列，
#                        也是 MultiLeRobotDataset 那条一维下标空间的基础
#         ⚠️ 本文件里 episodes 表里的 dataset_from_index / dataset_to_index 是
#            “整个 dataset 的全局帧号区间”（v3 语义），不是 episode 内的偏移
#   (2) delta_timestamps / delta_indices
#         上层只声明“相对当前帧要取哪些时间偏移（秒）”，例如 action chunk 取
#         [0, 1/30, ..., 31/30]；本文件在 __init__ 里用 fps 把它换成整数帧偏移
#         delta_indices（get_delta_indices）；取数时 current_idx + delta = 要读的帧号。
#         越界（跨出本条 episode）的帧不会报错，而是被“夹到边界 + 打 _is_pad 掩码”。
#   (3) 三条读取路径（性能差异很大，读代码时先分清走的是哪条）
#         ① 普通路径：datasets.Dataset 的 mmap / Arrow（_query_hf_dataset）
#         ② select 快路径：按“唯一的帧下标组合”做一次 .select()，多个 key 复用（_query_hf_dataset_fast）
#         ③ in_memory 纯 numpy 路径：启动时把用到的列搬进 RAM，取数退化成
#            numpy 花式索引（_materialize_numpy / _query_numpy_fast）——
#            为“NAS 随机读很慢”的场景准备，代价是启动时间和内存
#   (4) *_is_pad 掩码
#         每个 delta 查询 key 都会额外产出一份布尔掩码 {key}_is_pad，
#         True = 该帧越过了 episode 边界（上层据此把它排除在 loss 之外）。
#   (5) tasks.parquet 被当成“通用字符串表”用
#         除 task 文本外，coarse_task / plan / memory / bbox / action_hint /
#         2d_trace / high_level_instruction 等 CoT 文本也都存在这张表里，
#         列名统一叫 task_index；__getitem__ 负责把整数索引翻回文本字符串，
#         供 samples_builder 填进训练模板（这是本仓库相对上游 LeRobot 的扩展）。
#
# 【⚠️ 关于“写路径”（录制 / 转换数据）】
#   本文件保留了上游 LeRobot 的写入能力：create / add_frame / save_episode /
#   _save_episode_data / _save_episode_video / _batch_save_episode_video 等。
#   但本项目训练时**只用读路径**；而且这些写函数里引用的若干名字
#   （DEFAULT_EPISODES_PATH、update_chunk_file_indices、write_tasks、write_image、
#   AsyncImageWriter、get_file_size_in_mb、concatenate_video_files、
#   get_video_duration_in_s）并没有 import 进来，直接调用会 NameError。
#   新手读到这里可以直接跳过写路径：它们是“上游代码原样保留”，不是本仓库的使用方式。
#
# 【常见坑 / 排查清单】
#   · FileNotFoundError: Provided directory does not contain any parquet file：
#     root 指向的目录里没有 data/chunk-*/file-*.parquet（路径写错，或数据还没同步过来）。
#   · KeyError / IndexError 找不到 episode：请求的 episode 不在 meta/episodes 里，
#     或者 self.episodes 传入了越界的下标列表。
#   · 图像全黑：相机 key 在 shape_meta 里标了 dummy（上层注入零张量），本来就没读盘。
#   · 画面与数值对不齐：tolerance_s 给得太小（H.264 重编码会带来毫秒级时间戳抖动），
#     上层默认用 0.4/data_fps 来吸收。
#   · DataLoader num_workers>0 时开视频会 Segfault：见 _query_videos 的说明。
#   · in_memory=True 时 len(dataset) 可能偏大：hf_dataset 被释放后 num_frames 会
#     退回 meta.total_frames（本项目 v3 路径固定 episodes=None，两者相等，不受影响）。
#   · 传了 episodes 子集时要小心“两套下标”的错位：过滤后 hf_dataset 的行号是
#     “过滤后的局部行号”，而 episodes 表里的 dataset_from_index / to_index 是全局帧号；
#     当请求的 episode 不是从第 0 帧开始时，_get_query_indices 里的夹紧/掩码会算偏。
#     本项目 v3 路径固定 episodes=None（上层用帧区间做训练/验证切分），所以不会遇到。
#   · future_task 的偏移量在本仓库里恒为 16：GalaxeaLerobotDataset 把 future_task_offset
#     存在“上层数据集对象”上（self._future_task_offset），并没有下发到这里的
#     LeRobotDataset 实例，因此 getattr(self, "_future_task_offset", 16) 永远拿到默认值。
#
# 【相关文件与文档】
#   src/g05/data/lerobot/lerobot_dataset.py            v2.1 底层实现（接口一一对应）
#   src/g05/data/lerobot/datasets/util_v3.py           v3 的元数据 / 特征 / 统计等工具函数
#   src/g05/data/lerobot/datasets/video_utils.py       mp4 解码与编码（decode_video_frames）
#   src/g05/data/lerobot/datasets/compute_stats.py     episode 级统计量
#   src/g05/data/base_lerobot_dataset.py               上层：读哪些帧、怎么切维、归一化
#   src/g05/data/mixture_lerobot_dataset.py            上层：多 embodiment 混合
#   src/g05/data_processor/processor/samples_builder.py 消费 future_task / bbox / ... 等 CoT 字段
#   docs/data/schema_zh.md                             shape_meta 与数据集字段说明
# =============================================================================

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
# 标准库部分：
#   contextlib  —— push_to_hub 里用 contextlib.suppress 忽略“删除旧 tag 失败”
#   logging     —— 用 logging.getLogger(__name__) 打日志（in_memory 统计、跳过数据集等）
#   shutil / tempfile —— 写路径：临时目录里编码 mp4，再搬进数据集目录
#   warnings    —— get_episode_data 里屏蔽 “numpy 数组只读” 告警
#   Callable    —— image_transforms 的类型标注；Path —— 所有 root / 文件路径统一用 Path
#   traceback   —— 构造子数据集失败时打印堆栈（上游调试用）
import contextlib
import logging
import shutil
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
import traceback

# 第三方库部分：
#   datasets  —— HuggingFace datasets：本文件的数据容器。读的时候是 Arrow 表（mmap 或内存），
#                写的时候是临时 Dataset；MultiLeRobotDataset 的 features/camera_keys 也依赖它的类型
#   numpy     —— in_memory 模式下的张量缓存（_np_columns），以及写路径的数组拼接
#   packaging.version —— 解析 / 比较 info.json 里的 codebase_version
#   pandas    —— 读 tasks.parquet（字符串表）与 episodes parquet 元数据
#   PIL.Image —— hf_transform_to_torch 靠它判断“这一列是不是图像”，读出来是 PIL 对象
#   pyarrow   —— 写路径直接写 parquet；读路径用它做“Arrow 列 → numpy”的零拷贝转换
#   torch     —— 返回张量；torch.utils.data.Dataset 是基类
#   HfApi / snapshot_download —— 上传 / 下载数据集（本项目全部本地读盘，只在 push/download 分支用到）
import datasets
import numpy as np
import packaging.version
import pandas as pd
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

# g05 内部依赖：
#   统计量：aggregate_stats 汇总多条 episode 的统计；compute_episode_stats 算单条 episode 的统计
from g05.data.lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from g05.data.lerobot.datasets.util_v3 import (
    # —— 读路径真正会用到的 ——
    DEFAULT_FEATURES,          # 每个数据集必备的列（timestamp / frame_index / episode_index / index / task_index ...）
    DEFAULT_IMAGE_PATH,        # 写路径：临时图片的落盘模板（编码成 mp4 之前先落成 jpeg）
    INFO_PATH,                 # "meta/info.json" 的相对路径
    _validate_feature_names,   # 写路径：校验 feature 名里不能带 "/"
    check_delta_timestamps,    # 写路径：校验秒偏移是否对齐 1/fps（本文件里已被注释掉，改由上层校验）
    create_empty_dataset_info, # 写路径：造一份空的 info.json 模板（LeRobotDataset.create）
    create_lerobot_dataset_card,  # 写路径：生成 HuggingFace dataset card（push_to_hub）
    embed_images,              # 写路径：把图片内联进 Arrow 表
    flatten_dict,              # 写路径：把嵌套 stats 字典拍平成 "stats/xxx" 这样的列名
    get_delta_indices,         # 秒偏移 → 帧偏移（delta_timestamps → delta_indices）
    get_hf_features_from_features,  # LeRobot 的 features 描述 → datasets.Features（决定 parquet 怎么解析）
    hf_transform_to_torch,     # 行级 transform：把 PIL 图 / 列表 / 标量统一转成 torch.Tensor
    load_episodes,             # 读 meta/episodes/**（每条 episode 的帧区间与文件位置）
    load_info,                 # 读 meta/info.json
    load_stats,                # 读 meta/stats.json
    load_tasks,                # 读 meta/tasks.parquet（那张“整数 → 文本”的字符串表）
    load_annotations,          # 读 meta/annotations/**（可选，缺失时跳过）
    validate_episode_buffer,   # 写路径：写一条 episode 前校验缓冲区
    validate_frame,            # 写路径：写一帧前校验
    write_info,                # 写路径：写 meta/info.json
    write_json,                # 写路径：写 json（create 时写 info.json）
    write_stats,               # 写路径：写 meta/stats.json
    load_nested_dataset,       # 扫 data/chunk-*/file-*.parquet → datasets.Dataset（支持按 episode 过滤）
)
from g05.data.lerobot.datasets.video_utils import (
    VideoFrame,             # “视频帧”这个 feature 类型的类对象；用来判断哪些 key 需要解码
    decode_video_frames,    # 按时间戳从 mp4 取帧（读路径的核心）
    encode_video_frames,    # png/jpeg 序列 → mp4（写路径）
    get_safe_default_codec, # 自动挑解码后端：有 torchcodec 就用它，否则回退 pyav
    get_video_info,         # 读 mp4 的元信息（分辨率 / fps），写路径写进 info.json
)
# 数据集的默认缓存根目录：root=None 且 repo_id 不是绝对路径时，root = HF_LEROBOT_HOME / repo_id
from g05.data.lerobot.constants import HF_LEROBOT_HOME

# 本文件实现的是 v3.0 格式；info.json 里的 codebase_version 也是这个值。
# revision 参数缺省时就用它当选定的数据集版本（本地读盘时基本不生效）。
CODEBASE_VERSION = "v3.0"


# =============================================================================
# 第 1 层：LeRobotDatasetMetadata —— 只读 meta/ 的“目录说明书”
# =============================================================================
# 作用：在真正读数据之前，先把“这个数据集长什么样”读进来：
#   · info.json      → fps / features（列名、形状、dtype）/ total_episodes / ...
#   · tasks.parquet  → 那张“整数索引 → 文本”的字符串表（任务名、CoT 文本都放这里）
#   · episodes/**    → 每条 episode 的帧区间与文件位置（读数据时按它定位）
#   · stats.json     → 归一化统计量
# 它不读 parquet 数据本身、不解码视频，所以构造非常快（上游会先建它来探数据）。
#
# 上层怎么用（见 base_lerobot_dataset.py / mixture_lerobot_dataset.py）：
#     meta = LeRobotDatasetMetadata(repo_id=ds_dir, root=Path(ds_dir))
#     meta.total_episodes / meta.fps / meta.features / meta.tasks ...
# 之后 MultiLeRobotDataset 内部还会再建一份（每个子目录一份）。
class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        metadata_buffer_size: int = 10,
    ):
        # repo_id：数据集的“名字”。在本项目里它通常就是数据集的本地目录路径
        #          （base_lerobot_dataset.py 里直接写 repo_id = ds_dir），
        #          root 也一并显式传入，因此不会去查 HuggingFace Hub。
        # revision：数据集版本（分支 / tag / commit）；缺省 = 当前代码库版本 v3.0。
        # force_cache_sync：True 时先强制刷新本地缓存（本项目未启用，见下面被注释的代码）。
        # metadata_buffer_size：episode 元数据攒多少条再一起写 parquet（写路径用）。
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        # root 缺省时退化成 HuggingFace 缓存目录下的 <repo_id>，只在上传/下载场景才有意义。
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        # 下面三个属性服务于“写路径”：parquet writer、最近一条 episode、元数据缓冲。
        # 只读场景下它们会一直保持 None / 空列表，可以无视。
        self.writer = None
        self.latest_episode = None
        self.metadata_buffer: list[dict] = []
        self.metadata_buffer_size = metadata_buffer_size

        # ↓↓↓ 上游会在这里判断“本地缺文件就自动从 Hub 下载 meta/”。
        # 本仓库全部本地读盘（repo_id 就是本地路径），所以这段被整段注释掉，
        # 直接 load_metadata()：读不到就抛错，不会静默去联网。
        # try:
        #    if force_cache_sync:
        #        raise FileNotFoundError
        self.load_metadata()
        """
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()
        """

    # -------------------------------------------------------------------------
    # 写路径：episode 元数据的缓冲与落盘（只读训练数据时不涉及）
    # -------------------------------------------------------------------------
    def _flush_metadata_buffer(self) -> None:
        """Write all buffered episode metadata to parquet file.

        中文说明：把 metadata_buffer 里攒下的若干条 episode 元数据合并成一张 Arrow 表，
        追加写入 meta/episodes/chunk-xxx/file-xxx.parquet，并记住最新写过的这条 episode。
        攒批写一次比逐条写省掉大量 I/O（每条 episode 都要写一行元数据）。
        """
        # 没有缓冲内容（或还没初始化）就直接返回——只读路径调用到这里会立刻退出。
        if not hasattr(self, "metadata_buffer") or len(self.metadata_buffer) == 0:
            return

        # 目标结构：{列名: [该批次每条 episode 在该列上的取值]}，正好是 Arrow 表的列式布局。
        combined_dict = {}
        for episode_dict in self.metadata_buffer:
            for key, value in episode_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                # Extract value and serialize numpy arrays
                # because PyArrow's from_pydict function doesn't support numpy arrays
                # 中文：_save_episode_metadata 会把每个值包成单元素列表（如 [123]），
                # 这里剥掉那层列表；numpy 数组要转回 python list，pyarrow 才认识。
                val = value[0] if isinstance(value, list) else value
                combined_dict[key].append(val.tolist() if isinstance(val, np.ndarray) else val)

        # 这批 episode 写进哪个文件，由第一条 episode 的 chunk/file 下标决定。
        first_ep = self.metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        table = pa.Table.from_pydict(combined_dict)

        # 首批写入时才创建 ParquetWriter：schema 取自这批数据；
        # snappy 压缩 + 字典编码是上游的存储优化（字符串列压缩效果尤其好）。
        if not self.writer:
            path = Path(
                self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )

        self.writer.write_table(table)

        # latest_episode 记住“最后写过的那条”：下一条 episode 要靠它推算
        # dataset_from_index / dataset_to_index（全局帧号区间）。
        self.latest_episode = self.metadata_buffer[-1]
        self.metadata_buffer.clear()

    def _close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        # 先把缓冲区里的内容落盘，再关闭 writer。
        # 注意：parquet 的 footer（文件元数据）只在 close() 时才写入，
        # 忘记关闭得到的文件是读不出来的。
        self._flush_metadata_buffer()

        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
            self.writer = None

    def __del__(self):
        """
        Trust the user to call .finalize() but as an added safety check call the parquet writer to stop when calling the destructor
        """
        # 对象被回收时兜底关一次 writer，避免忘记调用 finalize() 把 parquet 留成坏文件。
        self._close_writer()

    # -------------------------------------------------------------------------
    # 读路径：一次性把全部元数据加载进内存
    # -------------------------------------------------------------------------
    def load_metadata(self):
        # info.json：fps / features / total_episodes / 文件路径模板等（下面所有 property 都读它）。
        self.info = load_info(self.root)
        # 上游会在这里校验 codebase_version 与当前代码是否兼容；本仓库注释掉了，
        # 因为线上有 v2.1 → v3.0 转换出来的多种数据，宽松一点更实用。
        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        # tasks.parquet：pandas.DataFrame，index = 文本，列 task_index = 整数。
        self.tasks = load_tasks(self.root)
        # annotations/ 是可选目录（subtask / scene / gripper 等标注），不存在就不加载。
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        # episodes/**：HF Dataset，一行一条 episode（帧区间 + 数据/视频文件下标）。
        # load_episodes 会剔掉 "stats/" 开头的列，让这张表保持轻量。
        self.episodes = load_episodes(self.root)
        # stats.json：归一化统计量（读路径只负责透传，真正的合并/计算在上层）。
        self.stats = load_stats(self.root)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        # 从 HuggingFace Hub 把文件拉到 self.root（allow/ignore 可只下拉 meta/ 或跳过视频）。
        # 本项目里 repo_id 是本地路径，正常不会走到这里。
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def url_root(self) -> str:
        """该数据集在 HuggingFace Hub 上的地址前缀（本项目基本用不到）。"""
        return f"hf://datasets/{self.repo_id}"

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        # 把 info.json 里的版本字符串解析成可比较的版本号（"3.0" → Version("3.0")）。
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        """给定 episode 下标，返回它所在的 parquet 文件路径。

        v3.0 里多条 episode 共用一个 parquet 文件，所以拿到的文件里可能还有
        别的 episode；要定位某一帧，靠的是 episodes 表里的
        dataset_from_index / dataset_to_index（“整个 dataset 的全局帧号”）。
        """
        if self.episodes is None:
            # 懒加载：某些路径（例如 LeRobotDataset.create 造出来的对象）还没读过 episodes 表。
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        # episodes 表里存的是 chunk/file 下标，文件名模板存在 info.json 的 data_path 里。
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        # data_path 形如 "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """给定 episode 下标 + 相机 key，返回它所在 mp4 文件路径。

        与数据文件同理：多条 episode 顺序拼在同一个 mp4 里，真正定位靠
        episodes 表里的 videos/<key>/from_timestamp（该 episode 在这个 mp4 中的起始秒数）。
        """
        if self.episodes is None:
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        # 每个相机 key 各自有独立的 chunk/file 分片（videos/<key>/chunk-xxx/file-xxx.mp4）。
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        fpath = self.video_path.format(
            video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx
        )
        return Path(fpath)

    # -------------------------------------------------------------------------
    # 只读属性：把 info.json / features 里的字段包装成好用的 Python 属性
    # （全是“读 self.info[...]”的糖，没有额外逻辑，可以快速扫过）
    # -------------------------------------------------------------------------
    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        # 形如 "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"，用 .format() 填下标。
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        # 形如 "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"；
        # 纯图片数据集（use_videos=False）这里是 None。
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        # ⚠️ 这是数据集“录制时”的真实频率（data_fps），不是模型的控制频率（model_fps）。
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        # {列名: {"dtype": "float32"/"image"/"video", "shape": [...], "names": ...}}，
        # 是判断“哪些列是相机、形状多大”的唯一依据。
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        # dtype == "image"：以 jpeg/png 形式内联在 parquet 里的相机。
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        # dtype == "video"：以 mp4 形式存放的相机（需要 decode_video_frames 解码）。
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        # 两种存储方式合起来：上层只关心“哪些 key 是画面”，不关心存成什么。
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        # 逐维度的语义名（例如关节名），一般只有部分数据集会填。
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        # 与 features 同构，但 shape 转成 tuple（便于与张量形状直接比较）。
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    # 下面这些是 info.json 里的“汇总计数 / 分片设置”，读路径基本只用到 total_episodes。
    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        # 全部 episode 的帧数之和；当只取一部分 episode 时，别用它当长度（用 LeRobotDataset.num_frames）。
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def chunks_size(self) -> int:
        """Max number of files per chunk."""
        # 一个 chunk 目录最多放多少个文件（默认 1000），写数据时用来决定何时开新 chunk。
        return self.info["chunks_size"]

    @property
    def data_files_size_in_mb(self) -> int:
        """Max size of data file in mega bytes."""
        # 单个 parquet 文件的大小上限（默认 100MB）：写路径据此切分新文件。
        return self.info["data_files_size_in_mb"]

    @property
    def video_files_size_in_mb(self) -> int:
        """Max size of video file in mega bytes."""
        # 单个 mp4 文件的大小上限（默认 500MB）。
        return self.info["video_files_size_in_mb"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.

        中文：任务文本 → 整数下标。这张表（tasks.parquet）就是“整数 ↔ 文本”的双向字典：
          · 正向：task 文本 → task_index（本方法 / __getitem__ 里查表用）
          · 反向：task_index → task 文本（self.tasks.iloc[idx].name）
        """
        # self.tasks 的 index 是文本，列 task_index 是整数，所以查表就是"按 index 取列值"。
        if task in self.tasks.index:
            return int(self.tasks.loc[task].task_index)
        else:
            return None

    def save_episode_tasks(self, tasks: list[str]):
        """写路径：把本条 episode 用到的任务文本登记进 tasks.parquet（已存在的复用旧下标）。"""
        # 一条 episode 里的任务去重后必须唯一（同一文本不重复登记）。
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Tasks are not unique: {tasks}")

        # 第一次写：直接按顺序建表，index = 文本，task_index = 0..N-1。
        if self.tasks is None:
            new_tasks = tasks
            task_indices = range(len(tasks))
            self.tasks = pd.DataFrame({"task_index": task_indices}, index=tasks)
        else:
            # 后续写：只登记新出现的文本，下标从当前表尾继续递增。
            new_tasks = [task for task in tasks if task not in self.tasks.index]
            new_task_indices = range(len(self.tasks), len(self.tasks) + len(new_tasks))
            for task_idx, task in zip(new_task_indices, new_tasks, strict=False):
                self.tasks.loc[task] = task_idx

        if len(new_tasks) > 0:
            # Update on disk
            # 中文：有新任务才写盘，避免每条 episode 都重写一次 tasks.parquet。
            write_tasks(self.tasks, self.root)

    def _save_episode_metadata(self, episode_dict: dict) -> None:
        """Buffer episode metadata and write to parquet in batches for efficiency.

        This function accumulates episode metadata in a buffer and flushes it when the buffer
        reaches the configured size. This reduces I/O overhead by writing multiple episodes
        at once instead of one row at a time.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.

        中文说明（写路径）：为“这一条 episode”生成元数据行，核心是算三个东西——
          · 它在哪个 parquet 文件里（meta/episodes/chunk_index + file_index）
          · 它占的全局帧区间（dataset_from_index ~ dataset_to_index，左闭右开）
          · 附加的 videos/<key>/... 字段（由 _save_episode_video 构造后并进来）
        算完先塞进 metadata_buffer，攒够 metadata_buffer_size 条再一次性调
        _flush_metadata_buffer() 写盘。
        """
        # Convert to list format for each value
        # 中文：先把每个值包成单元素列表，后面统一按“列”处理（一列对本条 episode 只有一个值）。
        episode_dict = {key: [value] for key, value in episode_dict.items()}
        num_frames = episode_dict["length"][0]

        if self.latest_episode is None:
            # Initialize indices and frame count for a new dataset made of the first episode data
            # 中文：还没有任何 episode 写过（新数据集的第一条），从 chunk-000/file-000 开始，
            # 全局帧号从 0 开始。
            chunk_idx, file_idx = 0, 0
            if self.episodes is not None and len(self.episodes) > 0:
                # It means we are resuming recording, so we need to load the latest episode
                # Update the indices to avoid overwriting the latest episode
                # 中文：本地已经存在 episodes 表 → 这次是“续录”。
                # 全局帧号接着上一条的 dataset_to_index 往下排，文件下标也往后挪一个，
                # 避免把已经存在的 episode 覆盖掉。
                chunk_idx = self.episodes[-1]["meta/episodes/chunk_index"]
                file_idx = self.episodes[-1]["meta/episodes/file_index"]
                latest_num_frames = self.episodes[-1]["dataset_to_index"]
                episode_dict["dataset_from_index"] = [latest_num_frames]
                episode_dict["dataset_to_index"] = [latest_num_frames + num_frames]

                # When resuming, move to the next file
                chunk_idx, file_idx = update_chunk_file_indices(
                    chunk_idx, file_idx, self.chunks_size
                )
            else:
                episode_dict["dataset_from_index"] = [0]
                episode_dict["dataset_to_index"] = [num_frames]

            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
        else:
            # 中文：已经写过至少一条 episode（正常录制流程），沿用上一条的文件下标，
            # 并检查“再加这一条是否会超过单文件大小上限”。
            chunk_idx = self.latest_episode["meta/episodes/chunk_index"][0]
            file_idx = self.latest_episode["meta/episodes/file_index"][0]

            # 正在写的 writer 身上能直接问到当前文件路径（已 flush 的情况则按模板拼出来）。
            latest_path = (
                self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
                if self.writer is None
                else self.writer.where
            )

            if Path(latest_path).exists():
                latest_size_in_mb = get_file_size_in_mb(Path(latest_path))
                # ⚠️ 上游这里取的是 episode_index（条数）而不是 length（帧数），
                # 严格说是估算偏差；但它只用于“要不要换新文件”的粗略判断，不影响读数。
                latest_num_frames = self.latest_episode["episode_index"][0]

                av_size_per_frame = (
                    latest_size_in_mb / latest_num_frames if latest_num_frames > 0 else 0.0
                )

                if latest_size_in_mb + av_size_per_frame * num_frames >= self.data_files_size_in_mb:
                    # Size limit is reached, flush buffer and prepare new parquet file
                    # 中文：文件快满了 → 先把缓冲落盘，再切到下一个 chunk/file，并关掉旧 writer。
                    self._flush_metadata_buffer()
                    chunk_idx, file_idx = update_chunk_file_indices(
                        chunk_idx, file_idx, self.chunks_size
                    )
                    self._close_writer()

            # Update the existing pandas dataframe with new row
            # 中文：全局帧区间 = 紧接着上一条的结尾（左闭右开）。
            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
            episode_dict["dataset_from_index"] = [self.latest_episode["dataset_to_index"][0]]
            episode_dict["dataset_to_index"] = [
                self.latest_episode["dataset_to_index"][0] + num_frames
            ]

        # Add to buffer
        # 中文：入缓冲；同时把它记为 latest_episode（下一条要用它算起始帧号）。
        self.metadata_buffer.append(episode_dict)
        self.latest_episode = episode_dict

        # 攒够一批就写盘。
        if len(self.metadata_buffer) >= self.metadata_buffer_size:
            self._flush_metadata_buffer()

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:
        """写路径：登记一条 episode 的元数据 + 更新 info.json / stats.json。

        episode_metadata 里带着 _save_episode_data / _save_episode_video 算出的
        文件下标与视频时间戳，这里只负责拼成一行元数据并落盘。
        """
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        # episode_metadata：数据文件位置（data/chunk_index、dataset_from_index ...）
        # 与视频位置（videos/<key>/from_timestamp ...）。
        episode_dict.update(episode_metadata)
        # stats 是嵌套字典，拍平成 "stats/action/..." 这样的列名后塞进同一行。
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)

        # Update info
        # 中文：同步更新 info.json 里的计数；splits 固定写成 "train": "0:N"（HF 数据集的分片约定）。
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}

        write_info(self.info, self.root)

        # stats.json 是“增量聚合”：用上一条的统计量与这条合并，而不是重算全量。
        self.stats = (
            aggregate_stats([self.stats, episode_stats])
            if self.stats is not None
            else episode_stats
        )
        write_stats(self.stats, self.root)

    def update_video_info(self, video_key: str | None = None) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.

        中文（写路径）：把 mp4 的元信息（分辨率 / fps）写进 info.json 的 features[key]["info"]，
        只从第 1 条 episode 的视频里读一次，因此要求所有视频用同样参数编码。
        """
        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        for key in video_keys:
            if not self.features[key].get("info", None):
                # 只在该 feature 还没有 info 字段时补写，避免覆盖已有信息。
                video_path = self.root / self.video_path.format(
                    video_key=key, chunk_index=0, file_index=0
                )
                self.info["features"][key]["info"] = get_video_info(video_path)

    def update_chunk_settings(
        self,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> None:
        """Update chunk and file size settings after dataset creation.

        This allows users to customize storage organization without modifying the constructor.
        These settings control how episodes are chunked and how large files can grow before
        creating new ones.

        Args:
            chunks_size: Maximum number of files per chunk directory. If None, keeps current value.
            data_files_size_in_mb: Maximum size for data parquet files in MB. If None, keeps current value.
            video_files_size_in_mb: Maximum size for video files in MB. If None, keeps current value.
        """
        # 中文（写路径）：数据集建好之后也能调整“分片策略”，并存回 info.json。
        # 三个参数传 None 表示保持原值；传非正数直接报错。
        # chunks_size：一个 chunk 目录最多放多少个文件。
        if chunks_size is not None:
            if chunks_size <= 0:
                raise ValueError(f"chunks_size must be positive, got {chunks_size}")
            self.info["chunks_size"] = chunks_size

        # data_files_size_in_mb：单个数据 parquet 文件的大小上限（MB）。
        if data_files_size_in_mb is not None:
            if data_files_size_in_mb <= 0:
                raise ValueError(
                    f"data_files_size_in_mb must be positive, got {data_files_size_in_mb}"
                )
            self.info["data_files_size_in_mb"] = data_files_size_in_mb

        # video_files_size_in_mb：单个 mp4 文件的大小上限（MB）。
        if video_files_size_in_mb is not None:
            if video_files_size_in_mb <= 0:
                raise ValueError(
                    f"video_files_size_in_mb must be positive, got {video_files_size_in_mb}"
                )
            self.info["video_files_size_in_mb"] = video_files_size_in_mb

        # Update the info file on disk
        # 改完立刻落盘，保证下次加载读到的是新设置。
        write_info(self.info, self.root)

    def get_chunk_settings(self) -> dict[str, int]:
        """Get current chunk and file size settings.

        Returns:
            Dict containing chunks_size, data_files_size_in_mb, and video_files_size_in_mb.
        """
        # 把三个分片设置打包返回（update_chunk_settings 的读取侧）。
        return {
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
        }

    def __repr__(self):
        # 打印数据集时能看到的信息：是谁、多少条 episode、多少帧、有哪些 feature。
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
        metadata_buffer_size: int = 10,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset.

        中文（写路径）：造一个“空数据集”的元数据层。注意它不是 __init__，
        而是用 cls.__new__(cls) 绕开构造函数手工搭对象——因为这里还没有 meta/ 目录可读，
        要反过来“先把目录建好、把 info.json 写出来”。
        调用链：LeRobotDataset.create(...) → LeRobotDatasetMetadata.create(...)
        """
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        # exist_ok=False：目标目录已存在就直接报错，避免把已有数据集覆盖掉。
        obj.root.mkdir(parents=True, exist_ok=False)

        # 补上每个数据集都有的标准列（timestamp / frame_index / episode_index / index / task_index ...），
        # 并校验列名里不能出现 "/"（会与 stats 的 "stats/xxx" 扁平列名冲突）。
        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        # 空数据集：还没有 episode / 统计量 / 任务表。
        obj.tasks = None
        obj.episodes = None
        obj.stats = None
        # 用上游模板生成 info.json 的初始内容（total_episodes=0、data_path/video_path 模板、features ...）。
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION,
            fps,
            features,
            use_videos,
            robot_type,
            chunks_size,
            data_files_size_in_mb,
            video_files_size_in_mb,
        )
        if len(obj.video_keys) > 0 and not use_videos:
            # 声明了 video 类型的 feature 却又说不用视频，属于自相矛盾的配置。
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        # 其余运行时字段与 __init__ 保持一致，让这个对象用起来和正常加载出来的没区别。
        obj.revision = None
        obj.writer = None
        obj.latest_episode = None
        obj.metadata_buffer = []
        obj.metadata_buffer_size = metadata_buffer_size
        return obj


# =============================================================================
# 第 2 层：LeRobotDataset —— 一个数据集目录的 Dataset
# =============================================================================
# 这是上层 MultiLeRobotDataset 内部真正干活的类：一个实例 = 磁盘上的一个数据集目录。
# 它同时具备两种身份：
#   · 读（本项目训练时唯一用到的能力）：__getitem__(idx) 按“帧下标 + delta 时间偏移”
#     读出 parquet 数值列与（可选的）mp4 解码画面，返回一个普通 dict；
#   · 写（录制 / 转换数据用的上游能力）：add_frame / save_episode / create ...
#     —— 本项目不涉及，见文件顶部“写路径”说明。
#
# 读路径最值得先看的两处：
#   · __init__  ：决定“加载哪些 episode、要不要 in_memory、delta 偏移换算成整数帧”
#   · __getitem__：一次取数的完整流程（含 *_is_pad 掩码、视频解码、CoT 文本解码）
#
# 参数速览（上游 base_lerobot_dataset.py 的传参）：
#   episodes         只加载这些 episode（None = 全部）。本项目 v3 路径传 None，
#                    训练/验证切分改由上层按帧区间完成。
#   delta_timestamps "相对当前帧取哪些时间偏移(秒)"，形如 {"action": [0, 1/30, ...]}。
#   tolerance_s      查视频帧的时间容差：H.264 重编码会让时间戳有毫秒级抖动，
#                    容差太小会报 "more than tolerance_s"；上层默认给 0.4/data_fps。
#   load_images      False 时完全不碰图像列（省 I/O，便于只跑数值列的场景）。
#   in_memory        True 时把用到的列搬进 RAM（NAS 随机读慢时用），见 _materialize_numpy。
class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        load_images: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        in_memory: bool = False,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v3.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v2.1 to v3.0, which you can find at
              lerobot/datasets/v30/convert_dataset_v21_to_v30.py.


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
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes
        │   │   ├── chunk-000
        │   │   │   ├── file-000.parquet
        │   │   │   ├── file-001.parquet
        │   │   │   └── ...
        │   │   ├── chunk-001
        │   │   │   └── ...
        │   │   └── ...
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.parquet
        └── videos
            ├── observation.images.laptop
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            ├── observation.images.phone
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. Multiple episodes are
        consolidated into chunked files which improves storage efficiency and loading performance. The
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
            force_cache_sync (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
            batch_encoding_size (int, optional): Number of episodes to accumulate before batch encoding videos.
                Set to 1 for immediate encoding (default), or higher for batched encoding. Defaults to 1.
        """
        # 中文小结：上游那段 docstring 讲了两种用法——
        #   (a) 数据集已存在（本地 root 或 Hub 上的 repo_id）→ 直接加载；
        #   (b) 数据集还不存在 → 用类方法 create() 造一个空数据集，然后 add_frame 逐帧写入。
        # 本项目（训练）用的是 (a)，且全部走本地目录：repo_id 就是目录路径，root 也显式传入；
        # 磁盘结构可以对照 __init__ docstring 里那棵目录树。
        super().__init__()
        # 基础属性：数据在哪、怎么变换、取哪些帧、图像开关、解码后端。
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms       # 可选：对图像做 torchvision 变换
        self.delta_timestamps = delta_timestamps       # 秒为单位的取帧计划（None = 只取当前帧）
        self.episodes = episodes                       # 要加载的 episode 列表（None = 全部）
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.load_images = load_images                 # False = 跳过全部图像列
        self.in_memory = in_memory                     # True = 把用到的列搬成常驻 numpy 数组
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None                      # 下面由 get_delta_indices 填：帧偏移（整数）
        self.batch_encoding_size = batch_encoding_size # 写路径：几条 episode 一起编码视频
        self.episodes_since_last_encoding = 0
        self.during_training = True                    # True = 训练态（会解码视频帧）
        self._hf_dataset_without_images = None         # 去掉图像列后的视图（load_images=False 时用）

        # Unused attributes
        # 下面这几个属性只有写路径（录制）才用；只读加载时一直是 None，读代码可以跳过。
        self.image_writer = None
        self.episode_buffer = None
        self.writer = None
        self.latest_episode = None
        self._current_file_start_frame = (
            None  # Track the starting frame index of the current parquet file
        )

        # 确保 root 存在（下载/写入时要用；只读场景下它本来就在）。
        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        # 先建元数据层：info.json / tasks / episodes / stats 全部读进来，
        # 后面的 features、total_episodes、episodes 表都从这里取。
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # 图像列清单（dtype == "image"，即内联在 parquet 里的图）；
        # 抛开图像读时，_refresh_hf_dataset_views 会按它把列移除。
        self._image_columns = [key for key, ft in self.features.items() if ft["dtype"] == "image"]

        # Track dataset state for efficient incremental writing
        # 写路径的状态位：是否处于“懒加载”（写盘后没重读 HF 数据集）、
        # 已记录帧数、是否为了读而关闭了 writer。只读加载时它们就停在初始值。
        self._lazy_loading = False
        self._recorded_frames = self.meta.total_frames
        self._writer_closed_for_reading = False
        # Load actual data
        self._np_columns = None  # initialized here; populated by _materialize_numpy if in_memory
        try:
            # 本地已经有完整数据 → 直接加载（正常路径）。
            # force_cache_sync=True 时故意抛错，强制走下面的“下载/刷新”分支。
            if force_cache_sync:
                raise FileNotFoundError
            self.hf_dataset = self.load_hf_dataset()
            self._refresh_hf_dataset_views()
            # 缓存里缺 episode（或缺对应的 mp4）→ 抛错，交给下一段去补齐。
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            # 兜底分支：从 HuggingFace Hub 补齐文件（本项目里 repo_id 是本地路径，
            # 正常不会走到；真走到通常意味着数据目录不完整，会直接报下载失败）。
            self.download(download_videos)
            self.hf_dataset = self.load_hf_dataset()
            self._refresh_hf_dataset_views()

        if self.in_memory:
            # 把用到的列转成常驻 numpy 数组（换来 O(1) 随机访问），
            # 之后彻底释放 Arrow/HF 数据集，避免同一份数据占两遍内存。
            self._materialize_numpy()
            # Free Arrow table — numpy arrays are now the sole data source
            self.hf_dataset = None
            self._hf_dataset_without_images = None

        # episodes 为 None 表示“全部”：
        if self.episodes is None:
            self.episodes = list(range(self.meta.total_episodes))
        # 只取有效的那些 episode（防止 meta 表里多出来的行）。
        valid_episode_len = self.meta.total_episodes

            # Read the raw list from metadata and truncate it to the valid length.
            # 中文：从 episodes 表里取出每条 episode 的全局帧区间。
        from_list = self.meta.episodes["dataset_from_index"][:valid_episode_len]
        to_list = self.meta.episodes["dataset_to_index"][:valid_episode_len]

            # Build a dict matching get_episode_data_index output format.
            # 中文：整理成 {"from": [...], "to": [...]}（左闭右开），
            # 与 v2.1 里 get_episode_data_index() 的输出格式保持一致，
            # 让上层代码不用区分版本。上半部分是按 episode 取数的索引表。
        self.episode_data_index = {
            "from": torch.LongTensor(from_list),
            "to": torch.LongTensor(to_list),
        }
        # 注意：上游原实现（含 episodes 过滤）在这里被本仓库替换成了上面的写法，
        # 因为 v3 的 dataset_from_index / to_index 已经是“全局帧号”，可以直接用。
        # self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            # 秒偏移 → 整数帧偏移，例如 {"action": [0.0, 0.0333, ...]} → {"action": [0, 1, ...]}。
            # 上游本来还会在这里 check_delta_timestamps 校验“偏移是否对齐 1/fps”，
            # 本仓库注释掉了：容差/合法性统一由上层（base_lerobot_dataset.py）负责。
            # check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        # 写路径：关闭 parquet writer（close 时才会写 footer）。
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
            self.writer = None

    def __del__(self):
        """
        Trust the user to call .finalize() but as an added safety check call the parquet writer to stop when calling the destructor
        """
        # 对象回收时的兜底：确保 writer 被关掉。
        self._close_writer()

    # -------------------------------------------------------------------------
    # HuggingFace Hub 交互（本项目全部本地读盘，这一组只在“上传/下载”场景用）
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
        """把本地数据集目录上传到 HuggingFace Hub，并按需打上版本 tag。"""
        # 本地临时图片目录（images/）永远不传；不带视频时把 videos/ 也排除。
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        # 建仓库（已存在就复用）→ 需要的话建分支 → 上传目录 → 生成并推送 dataset card。
        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
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
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            # 先用 delete_tag 删掉旧的同名 tag（不存在会报 RevisionNotFoundError，被 suppress 吃掉），
            # 再在当前分支上新建 tag，让 tag 永远指向最新上传的内容。
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
        # 与 LeRobotDatasetMetadata.pull_from_repo 相同：按模式过滤地从 Hub 拉文件到 root。
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # 中文：__init__ 里发现本地数据不全时会调用它。
        # 已经有了的文件不会被重复下载（snapshot_download 自身的行为）。
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        # 不使用图像时连视频都不下（省带宽/时间）；指定了 episodes 就只下这些 episode 的文件。
        ignore_patterns = None if (download_videos and self.load_images) else "videos/"
        files = None
        if self.episodes is not None:
            files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        """列出需要下载/读取的文件：这些 episode 的 parquet 文件 + 对应的 mp4 文件。"""
        episodes = (
            self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        )
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0 and self.load_images:
            # 每个相机 key × 每条 episode 的 mp4 路径。
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        # 中文：多条 episode 常常共用同一个 parquet/mp4，去重后返回（set 不保证顺序，仅用于下载）。
        fpaths = list(set(fpaths))
        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc.

        中文说明：这是读路径的“建索引”步骤——
          ① 把 info.json 里的 features 描述翻译成 datasets.Features（决定每列怎么解析，
             image → datasets.Image()、shape=(1,) → Value、shape=(K,) → Sequence ...）；
          ② 扫 data/chunk-*/file-*.parquet 得到 datasets.Dataset（默认内存映射，不占 RAM；
             in_memory=True 时整表读进内存，并按 episode 过滤）；
          ③ 给数据集挂上 set_transform(hf_transform_to_torch)，这样以后 row = ds[i]
             拿到的就是 torch.Tensor / PIL 图，而不是 python list。
        """

        # load_images=False 时把图像列整体摘掉（省 I/O：图像列是 parquet 里最重的部分）。
        features_map = dict(self.features)
        if not self.load_images:
            features_map = {key: ft for key, ft in features_map.items() if ft["dtype"] != "image"}
        features = get_hf_features_from_features(features_map)
        columns = list(features)

        # In-memory mode: only load columns we'll actually materialize to numpy
        # 中文：in_memory 模式下只读“真正会用到”的列——
        #   · _meta：__getitem__ / _query_* 要用到的元数据列（下面 _materialize_numpy 里同款清单）
        #   · delta_timestamps 的 key：action / state / 相机等查询列
        # 其余列（例如一堆没被 shape_meta 引用的 CoT 索引）根本不读进来，省内存。
        if self.in_memory and self.delta_timestamps is not None:
            _meta = {
                'episode_index', 'frame_index', 'timestamp', 'task_index', 'index',
                'coarse_task_index', 'operating_hand_index', 'subtask_annotation',
                'atomic_task_index', 'plan_index', 'memory_index', 'prev_memory_index',
                'scene_annotation',
                'bbox_index', 'action_hint_index', '2d_trace_index',
            }
            _needed = _meta | set(self.delta_timestamps.keys())
            columns = [c for c in columns if c in _needed]

        # data 目录下的 parquet → HF Dataset；episodes 参数会用 pyarrow 谓词下推只读指定 episode。
        # 输入根目录写成 self.root / "data"：对应 v3 布局里的 data/chunk-xxx/file-xxx.parquet。
        hf_dataset = load_nested_dataset(
            self.root / "data",
            features=features,
            episodes=self.episodes,
            columns=columns,
            in_memory=self.in_memory,
        )

        # 行级 transform：取值时把 python list / PIL 图统一转成 torch.Tensor。
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _materialize_numpy(self):
        """Convert HF dataset columns to numpy arrays for O(1) random access.

        Called once after load_hf_dataset() when in_memory=True.
        Only materializes columns actually used during training:
        - Columns referenced by delta_timestamps (action/state data)
        - Metadata columns needed by __getitem__ (episode_index, timestamp, etc.)

        中文说明（本仓库扩展，in_memory 模式的实现）：
          · 目的：数据放在 NAS 上时，Arrow 内存映射每次随机读都要走一次磁盘；
            训练每条样本要读几十帧、几十列，随机 I/O 会拖垮吞吐。
            把用到的列一次性读进 RAM（numpy 数组），__getitem__ 就退化成纯内存索引。
          · 代价：启动变慢、常驻内存变大（日志里会打印总共多少 MB）。
          · 只搬数值列：video 相机由 _query_videos 解码 mp4，不经过这里；
            image 类型的相机列也不在需要的列清单里（除非 delta_timestamps 显式点名它）。
          · 搬完后 self._np_columns = {列名: np.ndarray}，__getitem__ 走 _query_numpy_fast。
        """
        import numpy as np
        # 直接拿到底层 pyarrow 表（HF Dataset 的内部结构），按列转 numpy。
        table = self.hf_dataset._data.table  # underlying pyarrow.Table

        # Determine which columns are actually needed
        # Core metadata + all optional scalar columns referenced by __getitem__
        # 中文：这些是 __getitem__ 里会被读到的元数据列。
        # ⚠️ 与 load_hf_dataset 里那份 _meta 相比，这里少了 '2d_trace_index'（历史遗留）：
        # in_memory 模式下该列不会被搬进 numpy，于是 __getitem__ 的 numpy 分支里
        # "2d_trace_index" in item 为 False，trace_2d（2D 轨迹 CoT）会静默缺席。
        _META_COLS = {
            'episode_index', 'frame_index', 'timestamp', 'task_index', 'index',
            'coarse_task_index', 'operating_hand_index', 'subtask_annotation',
            'atomic_task_index', 'plan_index', 'memory_index', 'prev_memory_index',
            'scene_annotation',
            'bbox_index', 'action_hint_index',
        }
        needed = set(_META_COLS)
        if self.delta_timestamps is not None:
            # 加上所有被查询的列（action / state / 相机键 / task_index ...）。
            needed |= set(self.delta_timestamps.keys())
        # Only materialize columns that exist in the table AND are needed
        # 中文：只搬“表里真有、而且确实需要”的列；skipped_total 用于日志说明省掉了多少列。
        target_cols = [c for c in table.column_names if c in needed]
        skipped_total = len(table.column_names) - len(target_cols)

        self._np_columns = {}
        slow_cols = []   # 走“慢路径”转换的列名（嵌套 list 等），只用于日志
        total_bytes = 0
        n_rows = len(table)
        for col_name in target_cols:
            col = table.column(col_name)
            # 列 → numpy：能走 Arrow 缓冲区的走快路径，嵌套结构退回 to_pylist。
            arr = self._arrow_col_to_numpy(col, n_rows, col_name, slow_cols)
            if arr is not None:
                self._np_columns[col_name] = arr
                total_bytes += arr.nbytes
        if slow_cols:
            # 慢路径列会明显拖慢启动，打条日志方便排查（不影响正确性）。
            logging.getLogger(__name__).info(
                f"[in_memory] {len(slow_cols)} cols used slow path: {slow_cols}"
            )
        # 一行汇总日志：搬了多少列 / 跳过了多少列 / 占多少内存 / 多少帧。
        logging.getLogger(__name__).info(
            f"[in_memory] Materialized numpy: root={self.root}, "
            f"{len(self._np_columns)}/{len(table.column_names)} cols "
            f"({skipped_total} unused skipped), "
            f"{total_bytes / 1024**2:.1f} MB, {n_rows} rows"
        )

    @staticmethod
    def _arrow_col_to_numpy(col, n_rows, col_name, slow_cols):
        """Convert a single Arrow column to a numpy array.

        Handles: scalar (int/float), list<T>, fixed_size_list<T>[K], list<list<T>>.
        Returns None only if conversion completely fails.

        中文说明：把 Arrow 的一列转成 numpy 数组，尽量走“零拷贝/缓冲区直读”的快路径。
        参数 slow_cols 是“回传参数”——凡是退回慢路径的列名都会被追加进去，
        最后由 _materialize_numpy 汇总打日志（只是为了可观测性，不影响正确性）。

        parquet 里可能出现的三种列形态（对应 LeRobot 的 features shape）：
          · 标量列      shape=(1,)         → 一维数组 [N]
          · list<T>     shape=(K,)         → 二维数组 [N, K]（多数 action/state 就是这种）
          · list<list<T>>（嵌套）          → 把每行拍平后拼成 [N, 各行长度之和]
        """
        import numpy as np

        # Arrow 的 list 类型有 value_type 属性（list / fixed_size_list 都有），标量类型没有。
        col_type = col.type
        is_list_like = hasattr(col_type, 'value_type')

        # --- Scalar columns (int64, float32, etc.) ---
        # 标量列：直接 to_numpy。object 类型（字符串/混合）尝试转 float64，
        # 转不动就原样返回（例如 task 文本列、bbox JSON 字符串列）。
        if not is_list_like:
            arr = col.to_numpy(zero_copy_only=False)
            if arr.dtype == np.object_:
                try:
                    return arr.astype(np.float64)
                except (ValueError, TypeError):
                    slow_cols.append(col_name)
                    return arr
            return arr

        # --- List-like columns: try fast Arrow buffer path ---
        # 定长列表列（[N, K]）走快路径：Arrow 内部本来就是“N*K 的连续缓冲区”，
        # combine_chunks 把多个 row group 合并成一整块，直接 reshape 就得到 [N, K]，
        # 不经过 python 对象，因此比逐行 to_pylist 快很多。
        inner_type = col_type.value_type
        is_nested = hasattr(inner_type, 'value_type')  # list<list<T>>

        if not is_nested:
            try:
                combined = col.combine_chunks()
                flat = combined.values.to_numpy(zero_copy_only=False)
                # 用第一行的长度推断“每行多少个数”（定长列表才有这个前提）。
                first_val = col[0].as_py()
                list_len = len(first_val) if first_val is not None else 1
                return flat.reshape(n_rows, list_len).astype(np.float32, copy=False)
            except Exception:
                pass  # fall through to slow path

        # --- Slow path: to_pylist + stack (nested lists, edge cases) ---
        # 慢路径：先转成 python 列表再手工填 numpy。只在嵌套结构或上面的快路径失败时走。
        try:
            pylist = col.to_pylist()
            first = pylist[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (int, float)):
                # 一维数值列表 → [N, K]
                arr = np.empty((n_rows, len(first)), dtype=np.float32)
                for i, row in enumerate(pylist):
                    arr[i] = row
            elif isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
                # Nested list<list<float>>: flatten each row
                # 中文：二维嵌套（例如 [n_obj, n_coord] 的 bbox 列）把每行拍平成一维。
                flat_len = sum(len(sub) for sub in first)
                arr = np.empty((n_rows, flat_len), dtype=np.float32)
                for i, row in enumerate(pylist):
                    arr[i] = [v for sub in row for v in sub]
            else:
                # 兜底：直接交给 numpy 推断（可能是字符串列等）。
                arr = np.array(pylist)
            slow_cols.append(col_name)
            return arr
        except Exception:
            # 彻底转不了就返回 None，调用方会跳过这一列（下游取不到该字段）。
            slow_cols.append(f"{col_name}(FAILED)")
            return None

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files.

        中文说明：__init__ 里读完本地数据后的“体检”：
          ① episode 是否齐全（hf 数据里出现过的 episode_index 集合是否覆盖请求集合）；
          ② 需要的 mp4 文件是否都在磁盘上（少一个文件就判缓存不合格，去走下载/报错分支）。
        返回 False 时 __init__ 会抛 FileNotFoundError 并转到下载逻辑。
        """
        # 一条数据都没读出来 → 缓存肯定不合格（目录为空 / parquet 缺失）。
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        # Get available episode indices from cached dataset
        # 中文：unique("episode_index") 返回去重后的 episode 下标（可能是 tensor，统一成 int）。
        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        # Determine requested episodes
        # 请求集合：没指定 episodes 就是“全部 episode”，否则就是用户/上层给的那个列表。
        if self.episodes is None:
            requested_episodes = set(range(self.meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        # Check if all requested episodes are available in cached data
        # 请求集合必须是实际数据的子集，否则说明还有 episode 没同步下来。
        if not requested_episodes.issubset(available_episodes):
            return False

        # Check if all required video files exist
        # 中文：逐个 episode × 每个相机 key 检查 mp4 文件是否存在。
        # （只检查文件是否存在，不校验内容完整性——那是解码时报错的职责。）
        if len(self.meta.video_keys) > 0 and self.load_images:
            for ep_idx in requested_episodes:
                for vid_key in self.meta.video_keys:
                    video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def create_hf_dataset(self) -> datasets.Dataset:
        """写路径：造一个“空骨架”HF Dataset（只有 schema 没有行），供录制时往里塞帧。"""
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    # -------------------------------------------------------------------------
    # 下面 4 个是“读的时候到底用哪份数据/哪些列”的辅助方法
    # -------------------------------------------------------------------------
    def _refresh_hf_dataset_views(self) -> None:
        """维护一个“去掉图像列”的视图，供 load_images=False 时读取。

        为什么要单独存一份视图：本仓库要求 load_images 可以随时切换（上层会调
        MultiLeRobotDataset.set_load_images 同步），但 hf_dataset 的列不能原地删，
        所以这里预先算好 remove_columns 之后的视图，切换时直接换用。
        """
        self._hf_dataset_without_images = None
        if self.hf_dataset is None:
            # in_memory 模式下 hf_dataset 已被释放（numpy 是唯一数据源），无需维护视图。
            return
        removable_columns = [
            col for col in self._image_columns if col in self.hf_dataset.column_names
        ]
        if not removable_columns:
            # 没有图像列可去 → 两个视图指向同一个对象。
            self._hf_dataset_without_images = self.hf_dataset
            return
        self._hf_dataset_without_images = self.hf_dataset.remove_columns(removable_columns)
        self._hf_dataset_without_images.set_transform(hf_transform_to_torch)

    def _is_visual_key(self, key: str) -> bool:
        """这个 key 是不是画面（image 或 video）？——用来在不用图像时跳过它。"""
        feature = self.features.get(key)
        return feature is not None and feature["dtype"] in {"image", "video"}

    def _get_hf_dataset_for_reads(self) -> datasets.Dataset:
        """返回本次读取应该用的数据集视图（要不要带图像列，取决于 load_images）。"""
        if self.load_images or self._hf_dataset_without_images is None:
            return self.hf_dataset
        return self._hf_dataset_without_images

    def _episode_read_columns(self) -> list[str]:
        """读整条 episode（MultiLeRobotDataset.get_episode_data）时要读哪些列。

        规则：video 列永远不在这里（视频要解码，不走 parquet）；image 列看 load_images。
        """
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
        """Number of frames in selected episodes.

        Note: When episodes a subset of the full dataset is requested, we must return the
        actual loaded data length (len(self.hf_dataset)) rather than metadata total_frames.
        self.meta.total_frames is the total number of frames in the full dataset.

        中文：只取部分 episode 时，真正的样本数 = 实际加载进来的行数（len(hf_dataset)），
        而不是 info.json 里的 total_frames。注意 in_memory=True 时 hf_dataset 已被释放，
        这个分支会退回 total_frames —— 本项目 v3 路径固定 episodes=None，两者相等，因此无影响。
        """
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self.meta.total_frames

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
        if self.hf_dataset is not None:
            # 已经加载好数据 → 直接用它的 schema（最准确）。
            return self.hf_dataset.features
        else:
            # in_memory 模式下 hf_dataset 已释放 → 从 features 描述现推一份等价的 schema。
            return get_hf_features_from_features(self.features)

    # =========================================================================
    # 取数核心（三）：把“当前帧 + delta 偏移”翻译成“真正要读的帧号 + 越界掩码”
    # =========================================================================
    # 下标体系提醒：这里的 idx 是“整个 dataset 的全局帧号”，
    # episode 表里的 dataset_from_index / dataset_to_index 也是全局帧号（左闭右开）。
    # 例：某 episode 占全局帧 [1000, 1100)，当前帧 idx=1099，action 要取未来 32 帧
    #     → 1099+1 起就越界了，越界位置会被夹到 1099 并标 _is_pad=True。
    # =========================================================================
    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """算出一帧样本要读的所有帧号，以及每个位置的越界掩码。

        返回两个字典（都以“查询 key”为键）：
          query_indices: {key: [帧号, ...]}     —— 要读哪些帧（已夹在 episode 范围内）
          padding      : {"{key}_is_pad": [bool,...]} —— 对应位置是否越界（True = 该步不算 loss）

        clamp 规则 max(ep_start, min(ep_end - 1, idx + delta))：
          · 越到 episode 之前 → 夹到第一帧 ep_start
          · 越到 episode 之后 → 夹到最后一帧 ep_end - 1
        也就是说“越界处复制边界帧”，同时用 _is_pad 标记无效，
        上层（base_lerobot_dataset）读回后把无效步的 loss 权重清零。
        """
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        # 对每个查询 key 的每个偏移算一遍“夹紧后的帧号”。
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        # 同时记录“原始帧号是否越界”，作为 is_pad 掩码（shape 与查询结果的第一维一致）。
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """给每个相机 key 算出“要解码哪些时间戳（秒）”。

        为什么要单独算：视频文件是按时间戳解码的，而数值列是按行号取的。
        做法是从 hf 数据里直接读这些帧的 timestamp 列（比用 fps 现推更准，
        因为真实录制存在时间戳抖动）。
        current_ts 是当前帧的时间戳，query_indices 为 None 时（单帧配置）就只取它。
        """
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                # self.hf_dataset[list_of_idx] 返回的是按列组织的 dict，取其中的 timestamp 列。
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    # =========================================================================
    # 取数核心（四）：三条读取路径的实现
    #   ① _query_hf_dataset       —— 最朴素的逐 key 读（v2.1 风格；本文件里已被 fast 版取代）
    #   ② _query_hf_dataset_fast  —— 对“相同帧号列表”只 select 一次，多个列复用（默认走这条）
    #   ③ _query_numpy_fast       —— in_memory 模式的纯 numpy 花式索引（最快）
    # =========================================================================
    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.

        Tries column-first [key][indices] for speed, falls back to row-first.

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)

        中文说明：先试“列优先” self.hf_dataset[key][q_idx]（只取这一列，最快），
        某些 HF 版本/列类型不支持就退回“行优先” self.hf_dataset[q_idx][key]。
        ⚠️ 本文件里 __getitem__ 调的是 _query_hf_dataset_fast，这个函数是上游保留版本。
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                # 视频列不进 parquet 读取路径（它是 mp4 + 时间戳，由 _query_videos 处理）。
                continue
            try:
                result[key] = torch.stack(self.hf_dataset[key][q_idx])
            except (KeyError, TypeError, IndexError):
                # 行优先兜底：先按行取，再取该行里的这一列，最后堆成张量。
                result[key] = torch.stack(self.hf_dataset[q_idx][key])
        return result

    def _query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
        """本文件默认使用的读取入口：按“帧号元组”去重复用 select 结果。

        为什么要去重：delta_timestamps 里多个列经常共享同一份帧号列表，
        例如 state 与 action 都指向同一原始列、或一次查询里多个 part 用同样的偏移。
        去重后同一种帧号列表只 select 一次，多个 key 复用这份数据。
        """
        # Fast path: numpy arrays pre-materialized in RAM
        # in_memory 模式：完全绕开 Arrow，走 numpy 花式索引。
        if self._np_columns is not None:
            return self._query_numpy_fast(query_indices)

        # Original HF datasets path (mmap / non-in_memory)
        result = {}
        processed_indices = set()      # 已经 select 过的帧号元组（去重集合）
        index_to_selected = {}         # 帧号元组 → select 出来的子数据集（复用）
        source_dataset = self._get_hf_dataset_for_reads()
        for key, q_idx in query_indices.items():
            # 视频列不在这里读；不用图像时，图像列也直接跳过。
            if key in self.meta.video_keys or (not self.load_images and self._is_visual_key(key)):
                continue
            q_idx_tuple = tuple(q_idx)
            if q_idx_tuple not in processed_indices:
                # 第一次见到这组帧号：真正做一次 select（这是最贵的一步）。
                selected_data = source_dataset.select(q_idx)
                index_to_selected[q_idx_tuple] = selected_data
                processed_indices.add(q_idx_tuple)
            else:
                # 已经选过：直接复用，省掉一次 select。
                selected_data = index_to_selected[q_idx_tuple]
            # 取出该列并堆成张量，形状通常是 [查询帧数, 维度]（标量列则是 [查询帧数]）。
            result[key] = torch.stack(selected_data[key])
        return result

    def _query_numpy_fast(self, query_indices: dict[str, list[int]]) -> dict:
        """Numpy fast path: direct array indexing, no Arrow/HF overhead.

        中文：in_memory 模式下的取数实现——数据已经在 self._np_columns 里，
        取数就等于“numpy 花式索引”：arr[np.array(q_idx)]。
        这是整条读路径里最快的形态（纯内存、无 Arrow/HF/python 对象开销）。
        """
        import numpy as np
        result = {}
        idx_cache = {}  # dedup: same q_idx → same np index array
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys or (not self.load_images and self._is_visual_key(key)):
                continue
            if key not in self._np_columns:
                # 该列在 materialize 时被跳过了（不在需要的列清单里）→ 干脆不返回，
                # 下游取不到这个字段就当作“本数据集没有提供”。
                continue
            q_tuple = tuple(q_idx)
            if q_tuple not in idx_cache:
                # 同一组帧号只构造一次索引数组（np.array 的构造也有开销，顺手缓存）。
                idx_cache[q_tuple] = np.array(q_idx, dtype=np.intp)
            arr = self._np_columns[key][idx_cache[q_tuple]]  # fancy indexing → [128, dim] or [128,]
            if arr.dtype.kind in ('f', 'i', 'u'):
                # 数值类型：from_numpy 零拷贝（ascontiguousarray 保证内存连续，
                # 否则 torch.from_numpy 会报错或隐含复制）。
                result[key] = torch.from_numpy(np.ascontiguousarray(arr))
            else:
                # 字符串/object 等：只能走 list 转换（慢路径，一般只有文本列会到这里）。
                result[key] = torch.tensor(arr.tolist())

        # Skipped columns are simply absent — downstream will not see them
        return result

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.

        中文说明：按时间戳从 mp4 里解码画面。
          · 一个 mp4 里顺序存放了多条 episode，所以“请求的时间戳”要先加上本条 episode
            在该 mp4 中的起始时间 from_timestamp，才是文件内的绝对时间戳；
          · decode_video_frames 会返回 [1, 帧数, C, H, W]，squeeze(0) 去掉多余的第 0 维，
            得到 [帧数, C, H, W]（数值范围 [0,1]，float32）；
          · tolerance_s 用来容忍时间戳抖动，解码器取最接近的帧。

        ⚠️ 上面那段英文警告很重要：不要在“已经开着 DataLoader worker 的主进程”里
        再建一个 num_workers=0 的 DataLoader 来预热这个函数，会让视频后端句柄跨进程，直接段错误。
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Episodes are stored sequentially on a single mp4 to reduce the number of files.
            # Thus we load the start timestamp of the episode on this mp4 and,
            # shift the query timestamp accordingly.
            # 中文：episode 内时间 → 文件内绝对时间。
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            # 真正解码：一次调用解码整组时间戳（比逐帧调用高效）。
            frames = decode_video_frames(
                video_path, shifted_query_ts, self.tolerance_s, self.video_backend
            )
            # 去掉 batch 维，得到 [查询帧数, C, H, W]。
            item[vid_key] = frames.squeeze(0)

        return item

    def _ensure_hf_dataset_loaded(self):
        """Lazy load the HF dataset only when needed for reading.

        中文说明：写路径（录制）过程中会不断往 parquet 追加帧，
        每次追加完就重读整个数据集代价太大，所以 _save_episode_data 只置
        _lazy_loading = True；等真的要读（__getitem__）时再在这里重建一次。
        · 重建前先把 writer 关掉：parquet 必须写完 footer 才能被读；
        · _writer_closed_for_reading 记录下来，写路径下次会另开一个新文件继续追加。
        """
        if self._lazy_loading or self.hf_dataset is None:
            # Close the writer before loading to ensure parquet file is properly finalized
            if self.writer is not None:
                self._close_writer()
                self._writer_closed_for_reading = True
            self.hf_dataset = self.load_hf_dataset()
            self._refresh_hf_dataset_views()
            self._lazy_loading = False

    def __len__(self):
        # 样本数 = 帧数（每个样本对应数据集里的一帧）。
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        """取一条样本 —— 本文件最重要的函数，一次训练取数的完整流程。

        入参 idx：全局帧号（0 ~ num_frames-1）。
        返回  ：一个普通 dict，包含
                  · 该帧的所有元数据列（episode_index / frame_index / timestamp / index / task_index ...）
                  · delta 查询结果：action / state / 相机等张量，形状 [查询帧数, 维度]
                  · 对应的 {key}_is_pad 布尔掩码
                  · 视频列：解码后的画面 [查询帧数, C, H, W]
                  · 文本字段：task / coarse_task / atomic_task / plan / memory / bbox / ...（见函数末尾）
        上层（base_lerobot_dataset.py）拿到这个 dict 后，才按 shape_meta 切片、归一化、拼 samples 模板。

        流程总览：① 取“当前帧”这一行 → ② 按 delta_indices 取一窗帧 + 掩码
                 → ③ 解码视频帧 → ④ 图像变换 → ⑤ 把各种“索引”翻成文本。
        """
        # Fast path: numpy arrays pre-materialized (skip HF dataset entirely)
        # ①-a in_memory 模式：直接从 numpy 缓存里取“当前帧”那一行，完全不碰 HF/Arrow。
        if self._np_columns is not None:
            import numpy as np
            item = {}
            for key, arr in self._np_columns.items():
                # arr 形状 [N] 或 [N, K]；arr[idx] 的结果可能是标量、数组或字符串。
                val = arr[idx]
                if isinstance(val, np.ndarray) and val.dtype.kind in ('f', 'i', 'u'):
                    # np.array(val) 复制一份，避免张量与常驻缓存共享内存被下游改写。
                    item[key] = torch.from_numpy(np.array(val))
                elif isinstance(val, np.ndarray):
                    item[key] = torch.tensor(val.tolist())
                elif isinstance(val, (str, bytes)):
                    # 字符串列（如 subtask_annotation）保持原样，不转张量。
                    item[key] = val
                else:
                    item[key] = torch.tensor(val)
            ep_idx = int(item["episode_index"].item())
        else:
            # ①-b 常规模式：懒加载确保数据可读，然后取第 idx 行。
            # （hf_transform_to_torch 已经把每列转成 torch.Tensor / PIL 图。）
            self._ensure_hf_dataset_loaded()
            item = self._get_hf_dataset_for_reads()[idx]
            ep_idx = item["episode_index"].item()

        # ② delta 查询：算出要读的帧号与越界掩码，再批量读出这些帧。
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset_fast(query_indices)
            # 掩码先并进来（键名是 "{key}_is_pad"）。
            item = {**item, **padding}
            for key, val in query_result.items():
                # 查询结果覆盖同名的单帧值：例如 action 从“当前帧”变成 [H, D] 的动作块。
                item[key] = val

        if len(self.meta.video_keys) > 0 and self.during_training and self.load_images:
            # ③ 视频相机：按时间戳解码。
            #    during_training=True 时才解码（录制/写数据时不做无用功）；
            #    load_images=False 时整段跳过。
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            # 视频结果放前面、item 放后面：键名冲突时以 item 里的数值列为准（保持原有语义）。
            item = {**video_frames, **item}

        if self.image_transforms is not None and self.load_images:
            # ④ 可选的图像增强：上层默认不传（None），真正的归一化在 processor 里做。
            #    注意遍历的是 meta.camera_keys，因此 image 与 video 两种相机都会被处理；
            #    此时图像形状是 [T, C, H, W]，transform 需要自己能接受带时间维的输入。
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # ⑤ 文本解码：把 parquet 里的“整数索引”翻回 tasks.parquet 里的文本。
        #    这是本仓库对上游 LeRobot 的扩展：tasks.parquet 被当成通用字符串表，
        #    每行是（index = 文本, task_index = 整数），不同语义的文本共用这一张表。
        #    因此下面每一段都是“索引 → 文本”同一个套路。
        #
        # 先处理当前任务文本：
        # Add task as a string
        if item["task_index"].dim() == 0:
            # 单帧配置：task_index 是标量，直接按它取文本。
            task_idx = item["task_index"].item()
            item["chunked_task_index"] = None
        else:
            # delta 里包含 task_index → 拿到的是整窗任务索引（chunked_task_index）；
            # 当前帧任务取第 0 个，原始张量原样保留给上层（子任务切换时用来打 pad 掩码）。
            task_idx = item["task_index"][0].item()
            item["chunked_task_index"] = item["task_index"]
        item["task"] = self.meta.tasks.iloc[task_idx].name

        # Decode task text for the future N-th frame, used by FutureSubtaskCoTBuilder.
        # At episode-end overflow, the dataset has already clamped to the last frame,
        # matching action chunk behavior, so it is a valid value.
        # 中文：future_task = “未来第 N 帧的任务文本”，给 FutureSubtaskCoTBuilder 训练“预测下一个子任务”。
        # N 默认 16（可用 self._future_task_offset 覆盖）；越界位置在 _get_query_indices 里
        # 已被夹到 episode 最后一帧，所以这里拿到的永远是合法文本（与 action chunk 的行为一致）。
        # ⚠️ 注意 self 是“这里的 LeRobotDataset 实例”，而 GalaxeaLerobotDataset 把
        # future_task_offset 存在它自己（上层数据集）身上，没有下发到本实例，
        # 所以实际取值恒为默认的 16；要改偏移得让上层把它设到子数据集上。
        if item.get("chunked_task_index") is not None:
            ct = item["chunked_task_index"]
            future_offset = getattr(self, "_future_task_offset", 16)
            if future_offset < len(ct):
                fidx = ct[future_offset].item()
                item["future_task"] = self.meta.tasks.iloc[fidx].name

        if "coarse_task_index" in item:
            # 粗粒度任务文本（例如“把盘子放到架子上”），在 CoT 模板里当 high-level 指令用。
            coarse_task_index = item["coarse_task_index"].item()
            # print(f"Get coarse task index: {coarse_task_index}")
            item["coarse_task"] = self.meta.tasks.iloc[coarse_task_index].name
            # print(f"Get coarse task: {item['coarse_task']}")

        if "operating_hand_index" in item:
            # 操作手（left / right / both）：部分数据集会带这个字段。
            operating_hand_index = item["operating_hand_index"].item()
            item["operating_hand"] = self.meta.tasks.iloc[operating_hand_index].name

        # Only handle atomic_task_index -> item["atomic_task"] for
        # r1lite/r1pro _merged_final_v30 here. robocoin's
        # subtask_annotation -> atomic_task decoding lives in the
        # RobocoinLerobotDatasetV3 subclass
        # (src/g05/data/robocoin/robocoin_lerobot_dataset.py), not in the generic
        # LeRobot loader.
        # 中文：原子任务（最小可执行动作，例如“抓起杯子”）。上游 robocoin 数据的
        # subtask_annotation → atomic_task 解码在自己的子类里，不走这里。
        # （那段注释提到的 RobocoinLerobotDatasetV3 在本仓库当前代码里并不存在，属上游遗留说明。）
        # 查表用的是“按列反查”：tasks 表的 index 是文本、列 task_index 是整数，
        # 所以 tasks[tasks["task_index"] == i].index[0] 就是把整数翻回文本。
        # 查不到时返回空字符串（而不是报错），保证训练不中断。
        if "atomic_task_index" in item and item["atomic_task_index"] is not None:
            atomic_task_index = item["atomic_task_index"].item()
            filtered = self.meta.tasks[self.meta.tasks["task_index"] == int(atomic_task_index)]
            item["atomic_task"] = filtered.index[0] if len(filtered) > 0 else ""

        # to support plan input (plan_index → full plan string from tasks_new.jsonl)
        if "plan_index" in item and item["plan_index"] is not None:
            # 中文：plan_index → 计划文本（“先抓取、再移动、最后放置”这类多步计划），
            # 供计划类 CoT 模板使用。这里多做一层 pd.isna 判断：
            # 该列可能整列是 NaN（数据里没标注），NaN 不能 int()。
            import pandas as pd

            plan_idx_raw = item["plan_index"]
            if hasattr(plan_idx_raw, "item"):
                # 可能是 0 维张量 → 取出 python 标量。
                plan_idx_raw = plan_idx_raw.item()
            if plan_idx_raw is not None and not pd.isna(plan_idx_raw):
                plan_idx = int(plan_idx_raw)
                filtered = self.meta.tasks[self.meta.tasks["task_index"] == plan_idx]
                item["plan"] = filtered.index[0] if len(filtered) > 0 else ""

        # to support memory-based VLM training (memory_index / prev_memory_index from v21 data)
        # 中文（记忆类 CoT，来自 v2.1 数据改造）：
        #   memory_update → 当前应该“写入记忆”的内容
        #   memory        → 上一时刻的“记忆状态”（prev_memory_index）
        if "memory_index" in item and item["memory_index"] is not None:
            memory_idx = item["memory_index"].item()
            filtered = self.meta.tasks[self.meta.tasks["task_index"] == int(memory_idx)]
            item["memory_update"] = filtered.index[0] if len(filtered) > 0 else ""

        if "prev_memory_index" in item and item["prev_memory_index"] is not None:
            prev_mem_idx = item["prev_memory_index"].item()
            filtered = self.meta.tasks[self.meta.tasks["task_index"] == int(prev_mem_idx)]
            item["memory"] = filtered.index[0] if len(filtered) > 0 else ""

        # to support bbox CoT (bbox_index → JSON string {"obj_name": [x1,y1,x2,y2]})
        # 中文：bbox 类 CoT。值本身是一段 JSON 文本
        # （{"obj_name": [x1,y1,x2,y2], ...}，坐标已归一化），这里只负责取出来，
        # 解析与拼接模板由 samples_builder 负责（见 samples_builder.py 的 BBoxCoTBuilder）。
        # 取不到时给 None（区别于其他字段给 ""），上层据此判断“本样本没有 bbox”。
        if "bbox_index" in item and item["bbox_index"] is not None:
            import pandas as _pd
            bbox_idx_raw = item["bbox_index"]
            if hasattr(bbox_idx_raw, "item"):
                bbox_idx_raw = bbox_idx_raw.item()
            if bbox_idx_raw is not None and not (isinstance(bbox_idx_raw, float) and _pd.isna(bbox_idx_raw)):
                bbox_idx = int(bbox_idx_raw)
                filtered = self.meta.tasks[self.meta.tasks["task_index"] == bbox_idx]
                item["bbox"] = filtered.index[0] if len(filtered) > 0 else None

        # to support action hint CoT (action_hint_index → natural language gripper motion text)
        # 中文：动作提示 CoT —— 一句自然语言的夹爪运动描述（如 “gripper close → move up”）。
        if "action_hint_index" in item and item["action_hint_index"] is not None:
            import pandas as _pd
            ah_idx_raw = item["action_hint_index"]
            if hasattr(ah_idx_raw, "item"):
                ah_idx_raw = ah_idx_raw.item()
            if ah_idx_raw is not None and not (isinstance(ah_idx_raw, float) and _pd.isna(ah_idx_raw)):
                ah_idx = int(ah_idx_raw)
                filtered = self.meta.tasks[self.meta.tasks["task_index"] == ah_idx]
                item["action_hint"] = filtered.index[0] if len(filtered) > 0 else None

        # to support 2D trace CoT (2d_trace_index → JSON string with uv_left/uv_right gripper positions)
        # 中文：2D 轨迹 CoT —— 一段 JSON，内含夹爪在图像平面上的 uv 坐标（uv_left / uv_right）。
        if "2d_trace_index" in item and item["2d_trace_index"] is not None:
            import pandas as _pd
            trace_idx_raw = item["2d_trace_index"]
            if hasattr(trace_idx_raw, "item"):
                trace_idx_raw = trace_idx_raw.item()
            if trace_idx_raw is not None and not (isinstance(trace_idx_raw, float) and _pd.isna(trace_idx_raw)):
                trace_idx = int(trace_idx_raw)
                filtered = self.meta.tasks[self.meta.tasks["task_index"] == trace_idx]
                item["trace_2d"] = filtered.index[0] if len(filtered) > 0 else None

        # to support high level instruction
        # 中文：高层指令文本（比 task 更抽象的目标，用于分层训练）。
        if (
            "high_level_instruction_index" in item
            and item["high_level_instruction_index"] is not None
        ):
            high_level_instruction_index = item["high_level_instruction_index"].item()
            filtered = self.meta.tasks[
                self.meta.tasks["task_index"] == int(high_level_instruction_index)
            ]
            item["high_level_instruction"] = filtered.index[0] if len(filtered) > 0 else ""

        # quality_index disabled — always treat as qualified
        # 中文：上游用 quality_index 标记“这一步是否达到质量标准”，本项目停用了这项过滤，
        # 统一认为样本合格（下面的 step_is_qualified 恒为 True，上层不用再判空）。
        # if "quality_index" in item and item["quality_index"] is not None:
        #     quality_index = item["quality_index"].item()
        #     quality_value = self.meta.tasks.iloc[quality_index].index.tolist()
        #     item["step_is_qualified"] = all([value == "qualified" for value in quality_value])
        # else:
        item["step_is_qualified"] = True
        # 返回给上层：一个普通 dict（不是 Batch），后续由 base_lerobot_dataset 按
        # shape_meta 切片、归一化、拼 samples 模板（见 docs/architecture/g05_io_zh.md）。
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

    def finalize(self):
        """
        Close the parquet writers. This function needs to be called after data collection/conversion, else footer metadata won't be written to the parquet files.
        The dataset won't be valid and can't be loaded as ds = LeRobotDataset(repo_id=repo, root=HF_LEROBOT_HOME.joinpath(repo))
        """
        # 写路径专属：录制/转换结束后必须调用，把 parquet footer 写出去；
        # 否则文件不完整，之后用 LeRobotDataset(...) 也加载不了。
        # （读路径不需要调用。）
        self._close_writer()
        self.meta._close_writer()

    # =========================================================================
    # 以下全部是“写路径”（录制 / 把已有数据转成 LeRobot 格式）的实现。
    # 本项目训练时不会走到，读代码时可以先跳过这一大段；
    # 若真要按顺序读，建议顺序：create_episode_buffer → add_frame → save_episode
    #   → _save_episode_data（写 parquet）/ _save_episode_video（写 mp4）
    #   → meta.save_episode（写 episode 元数据 + info + stats）。
    # =========================================================================
    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        """造一条 episode 的空缓冲区（一个“按列组织的 list 字典”）。

        约定：episode_buffer[key] 是一个 list，每调用一次 add_frame 就往每个 list 追加一项；
        size / task 是特殊字段（不在 features 里），分别记“当前帧数”和“本帧任务文本”。
        """
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            # episode_index 是“整条 episode 共用一个值”，所以直接填标量；其余列先给空 list。
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        """临时图片的落盘路径，例如 images/<camera>/episode-000123/frame-000045.jpeg。"""
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _get_image_file_dir(self, episode_index: int, image_key: str) -> Path:
        """该 episode 该相机的图片目录（编码 mp4 时按目录顺序读帧）。"""
        return self._get_image_file_path(episode_index, image_key, frame_index=0).parent

    def _save_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image | bytes, fpath: Path
    ) -> None:
        """落一张图：开了异步 image_writer 就交给它（并行写盘），否则同步写。"""
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                # torch 张量先搬到 CPU 再转 numpy（write_image 只认 numpy/PIL/bytes）。
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # 中文：把一帧塞进内存缓冲（episode_buffer）。除了图像会立刻落到临时目录之外，
        # 数值列都还在内存里；真正写盘要等 save_episode()。
        # frame 的形状：{"observation.state": np.ndarray/torch.Tensor, "cam": HWC uint8, "task": str, ...}
        # Convert torch to numpy if needed
        # 统一成 numpy：后面的校验与落盘都按 numpy 处理。
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        # 校验这一帧的字段名/形状/dtype 是否与 features 声明一致（写错立刻报错）。
        validate_frame(frame, self.features)

        # 第一次调用时自动创建缓冲（也可以由 create() 预先建好）。
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        # 帧号 / 时间戳由数据集自己填：frame_index 就是当前缓冲里已有多少帧，
        # timestamp 优先用调用方传入的，没传就按帧号 / fps 推算。
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        # task 是“每帧都要带”的文本，pop 出来单独存（不参与下面的 features 遍历）。
        self.episode_buffer["task"].append(
            frame.pop("task")
        )  # Remove task from frame after processing

        # Add frame features to episode_buffer
        # 其余字段逐个追加。图像/视频帧：先落到临时图片目录，缓冲里存的是“图片路径”；
        # 等 save_episode / _save_episode_video 时再把这些图片编码成 mp4 并删掉临时目录。
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"],
                    image_key=key,
                    frame_index=frame_index,
                )
                if frame_index == 0:
                    # 第一帧时创建目录（一个 episode 一个目录）。
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        # 帧数 +1（下一帧的 frame_index 就是它）。
        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Video encoding is handled automatically based on batch_encoding_size:
        - If batch_encoding_size == 1: Videos are encoded immediately after each episode
        - If batch_encoding_size > 1: Videos are encoded in batches.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        # 中文流程（写路径的主流程，按顺序看这 7 步就够）：
        #   ① 校验缓冲 → ② 生成 index / episode_index / task_index 三列
        #   → ③ 把各列 list 堆成数组 → ④ 计算统计量 → ⑤ 写 parquet（_save_episode_data）
        #   → ⑥ 编码视频（_save_episode_video，可批量）→ ⑦ 写 episode 元数据 + info + stats
        #   最后清空缓冲，准备下一条 episode。
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        # 中文：size / task 是缓冲的“内部字段”，不属于 parquet 的列，先取出来再删掉。
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))          # 本条 episode 用到的任务文本（去重）
        episode_index = episode_buffer["episode_index"]

        # index：全局帧号，接着已有帧数往下排（这是 v3 的主键列）。
        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        # episode_index：整条 episode 共用一个值，展开成与帧数等长的数组。
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        # 先把任务文本登记进 tasks.parquet（新任务会分配新的整数下标）。
        self.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        # 再把“逐帧的任务文本”翻译成“逐帧的任务下标”写进 task_index 列。
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                # 上面三列已经处理完；图像/视频列存的是“文件路径 + 帧信息”，
                # 在 _save_episode_data / _save_episode_video 里单独处理，这里跳过。
                continue
            # 把“逐帧的 list”堆成 [帧数, 维度] 的数组。
            episode_buffer[key] = np.stack(episode_buffer[key])

        # Wait for image writer to end, so that episode stats over images can be computed
        # 等异步图片写盘结束，保证后面算统计量/编码视频时图片都已落盘。
        self._wait_image_writer()
        # 计算本条 episode 的统计量（min/max/mean/std/分位数），随后要并进 meta.stats。
        # is_compute_episode_stats_image=False：图像不在这里统计（上层另有处理）。
        ep_stats = compute_episode_stats(
            episode_buffer, self.features, is_compute_episode_stats_image=False
        )

        # ⑤ 写数值列 parquet，返回 {"data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"}。
        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        # ⑥-a 非批量模式：本条 episode 立刻编码 mp4，并把视频位置信息并进元数据。
        if has_video_keys and not use_batched_encoding:
            for video_key in self.meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` need to be executed after encoding the videos
        # ⑦ 写 episode 元数据行（必须先有上面的视频时间戳信息）。
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # Check if we should trigger batch encoding
            # ⑥-b 批量模式：攒够 batch_encoding_size 条 episode 才一起编码视频（省启动开销）。
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                # 编码最近这批 episode 的 mp4，并把视频信息回写进 episodes 表。
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # Reset episode buffer and clean up temporary images (if not already deleted during video encoding)
            # 复用同一实例连续录制时清空缓冲；传入 episode_data 的一次性写入则不动状态。
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None) -> None:
        """
        Batch save videos for multiple episodes.

        Args:
            start_episode: Starting episode index (inclusive)
            end_episode: Ending episode index (exclusive). If None, encodes all episodes from start_episode to the current episode.
        """
        # 中文：把 [start_episode, end_episode) 这段 episode 的临时图片一起编码成 mp4，
        # 然后把视频位置信息（chunk/file/from_timestamp/to_timestamp）写回这些 episode 的元数据行。
        # 之所以要“批量”，是因为每条 episode 单独开一次编码会话开销不小。
        if end_episode is None:
            end_episode = self.num_episodes

        logging.info(
            f"Batch encoding {self.batch_encoding_size} videos for episodes {start_episode} to {end_episode - 1}"
        )

        chunk_idx = self.meta.episodes[start_episode]["data/chunk_index"]
        file_idx = self.meta.episodes[start_episode]["data/file_index"]
        # 直接读这个 episode 元数据 parquet 成 pandas，逐条改完再写回。
        episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        episode_df = pd.read_parquet(episode_df_path)

        for ep_idx in range(start_episode, end_episode):
            logging.info(f"Encoding videos for episode {ep_idx}")

            if (
                self.meta.episodes[ep_idx]["data/chunk_index"] != chunk_idx
                or self.meta.episodes[ep_idx]["data/file_index"] != file_idx
            ):
                # The current episode is in a new chunk or file.
                # Save previous episode dataframe and update the Hugging Face dataset by reloading it.
                # 中文：跨到新的 chunk/file 了 → 先把上一份改好的元数据落盘，
                # 再重新加载 episodes 表、换到新文件上继续改。
                episode_df.to_parquet(episode_df_path)
                self.meta.episodes = load_episodes(self.root)

                # Load new episode dataframe
                chunk_idx = self.meta.episodes[ep_idx]["data/chunk_index"]
                file_idx = self.meta.episodes[ep_idx]["data/file_index"]
                episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
                episode_df = pd.read_parquet(episode_df_path)

            # Save the current episode's video metadata to the dataframe
            video_ep_metadata = {}
            for video_key in self.meta.video_keys:
                # 每个相机 key 各自编码一个 mp4，并返回它在该文件里的时间区间。
                video_ep_metadata.update(self._save_episode_video(video_key, ep_idx))
            # episode_index 已经在这一行里了（作为索引），不必重复写成列。
            video_ep_metadata.pop("episode_index")
            video_ep_df = pd.DataFrame(video_ep_metadata, index=[ep_idx]).convert_dtypes(
                dtype_backend="pyarrow"
            )  # allows NaN values along with integers

            # combine_first：用新算出的视频列去补（合并）这一行已有的列，其他列保持不变。
            episode_df = episode_df.combine_first(video_ep_df)
            episode_df.to_parquet(episode_df_path)
            # 改完立刻重载，保证 self.meta.episodes 里的内容和磁盘一致。
            self.meta.episodes = load_episodes(self.root)

    def _save_episode_data(self, episode_buffer: dict) -> dict:
        """Save episode data to a parquet file and update the Hugging Face dataset of frames data.

        This function processes episodes data from a buffer, converts it into a Hugging Face dataset,
        and saves it as a parquet file. It handles both the creation of new parquet files and the
        updating of existing ones based on size constraints. After saving the data, it reloads
        the Hugging Face dataset to ensure it is up-to-date.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.

        中文说明（写路径）：把一条 episode 的逐帧数据写进 parquet，并返回四个元数据字段
        （chunk/file 下标 + 全局帧区间）。要决定的只有一件事：
        “接着往当前 parquet 文件追加，还是另开一个新文件？”
        判据 = 当前文件大小 + （本 episode 帧数 × 该文件平均每帧大小）是否超过上限；
        另外还有一种情况必须换文件：文件已经被“为读而关闭”过（关掉的 parquet 无法再追加）。
        注意这里写的是值经过 embed_images 处理（图像内联进 Arrow）后的数据。
        """
        # Convert buffer into HF Dataset
        # 只取 features 里声明的列（缓冲里多余的内部字段不要）。
        ep_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        ep_num_frames = len(ep_dataset)
        # 把图片真正嵌进 Arrow 表（避免只存引用）。
        ep_dataset = embed_images(ep_dataset)
        if self.latest_episode is None:
            # Initialize indices and frame count for a new dataset made of the first episode data
            # 中文：本次进程内还没写过任何 episode。
            chunk_idx, file_idx = 0, 0
            global_frame_index = 0
            self._current_file_start_frame = 0
            # However, if the episodes already exists
            # It means we are resuming recording, so we need to load the latest episode
            # Update the indices to avoid overwriting the latest episode
            # 中文：但磁盘上已有 episode 表 → 续录。全局帧号接着最后一条的结尾，
            # 文件下标往后挪一格（不覆盖已有文件）。
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                latest_ep = self.meta.episodes[-1]
                global_frame_index = latest_ep["dataset_to_index"]
                chunk_idx = latest_ep["data/chunk_index"]
                file_idx = latest_ep["data/file_index"]

                # When resuming, move to the next file
                chunk_idx, file_idx = update_chunk_file_indices(
                    chunk_idx, file_idx, self.meta.chunks_size
                )
                self._current_file_start_frame = global_frame_index
        else:
            # Retrieve information from the latest parquet file
            # 中文：接着上一次写入的文件继续追加，并估算“再加这一条会不会超限”。
            latest_ep = self.latest_episode
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]
            # 全局帧号 = 上一个文件里最后一帧的 index + 1。
            global_frame_index = latest_ep["index"][-1] + 1

            latest_path = self.root / self.meta.data_path.format(
                chunk_index=chunk_idx, file_index=file_idx
            )
            latest_size_in_mb = get_file_size_in_mb(latest_path)

            # 平均每帧占多少 MB：注意分母是“本文件里实际有多少帧”，
            # 而不是整个数据集的总帧数（_current_file_start_frame 就是本文件的起始全局帧号）。
            frames_in_current_file = global_frame_index - self._current_file_start_frame
            av_size_per_frame = (
                latest_size_in_mb / frames_in_current_file if frames_in_current_file > 0 else 0
            )

            # Determine if a new parquet file is needed
            if (
                latest_size_in_mb + av_size_per_frame * ep_num_frames
                >= self.meta.data_files_size_in_mb
                or self._writer_closed_for_reading
            ):
                # Size limit is reached or writer was closed for reading, prepare new parquet file
                # 中文：超限，或文件已经为“读”而关闭过（关掉的 parquet 不能继续追加）
                # → 换到下一个 chunk/file 重新开始写。
                chunk_idx, file_idx = update_chunk_file_indices(
                    chunk_idx, file_idx, self.meta.chunks_size
                )
                self._close_writer()
                self._writer_closed_for_reading = False
                self._current_file_start_frame = global_frame_index

        # 这两列会被写进 episode 元数据行（指向本条 episode 所在的数据文件）。
        ep_dict["data/chunk_index"] = chunk_idx
        ep_dict["data/file_index"] = file_idx

        # Write the resulting dataframe from RAM to disk
        path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 取出底层 Arrow 表，追加写入（writer 首次使用时按当前 schema 创建）。
        table = ep_dataset.with_format("arrow")[:]
        if not self.writer:
            self.writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )
        self.writer.write_table(table)

        metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_frame_index,
            "dataset_to_index": global_frame_index + ep_num_frames,
        }

        # Store metadata with episode data for next episode
        # 中文：记住这条 episode 的信息，下一条要靠它续算文件位置与全局帧号。
        self.latest_episode = {**ep_dict, **metadata}

        # Mark that the HF dataset needs reloading (lazy loading approach)
        # This avoids expensive reloading during sequential recording
        # 中文：写盘后不立刻重读整个数据集（连续录制时那样会非常慢），
        # 只打个标记；等真的读数据时由 _ensure_hf_dataset_loaded 重建。
        self._lazy_loading = True
        # Update recorded frames count for efficient length tracking
        self._recorded_frames += ep_num_frames

        # 返回给 save_episode，用于写 episode 元数据行。
        return metadata

    def _save_episode_video(self, video_key: str, episode_index: int) -> dict:
        """写路径：把一条 episode 某个相机的图片序列编码成 mp4，并返回它在该 mp4 中的时间区间。

        返回 {"episode_index", "videos/<key>/chunk_index", "videos/<key>/file_index",
              "videos/<key>/from_timestamp", "videos/<key>/to_timestamp"}
        —— add_frame 时图片是逐帧落盘的，这里才真正压成视频；
        from/to_timestamp 是“本条 episode 在合并后的大 mp4 里的起止秒数”，
        读取时 _query_videos 就靠这个偏移把 episode 内时间换算成文件内绝对时间。
        """
        # Encode episode frames into a temporary video
        # ① 先把图片序列编码成一个临时 mp4（放在 root 下的临时目录里）。
        ep_path = self._encode_temporary_episode_video(video_key, episode_index)
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        # ② 决定这个临时 mp4 是“新开一个正式视频文件”，还是“并进已有文件”。
        if (
            episode_index == 0
            or self.meta.latest_episode is None
            or f"videos/{video_key}/chunk_index" not in self.meta.latest_episode
        ):
            # Initialize indices for a new dataset made of the first episode data
            # 中文：第一条 episode（或该相机还没有任何视频文件）→ 从 chunk-000/file-000 起，
            # 直接把临时文件搬过去当正式文件，起始时间戳为 0。
            chunk_idx, file_idx = 0, 0
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                # It means we are resuming recording, so we need to load the latest episode
                # Update the indices to avoid overwriting the latest episode
                # 续录：接着上一条 episode 的视频文件下标往后挪一格。
                old_chunk_idx = self.meta.episodes[-1][f"videos/{video_key}/chunk_index"]
                old_file_idx = self.meta.episodes[-1][f"videos/{video_key}/file_index"]
                chunk_idx, file_idx = update_chunk_file_indices(
                    old_chunk_idx, old_file_idx, self.meta.chunks_size
                )
            latest_duration_in_s = 0.0
            new_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep_path), str(new_path))
        else:
            # Retrieve information from the latest updated video file using latest_episode
            # 中文：该相机已有视频文件 → 看“并进去会不会超过文件大小上限”。
            latest_ep = self.meta.latest_episode
            chunk_idx = latest_ep[f"videos/{video_key}/chunk_index"][0]
            file_idx = latest_ep[f"videos/{video_key}/file_index"][0]

            latest_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            latest_size_in_mb = get_file_size_in_mb(latest_path)
            # 已有文件当前的时长 → 本条 episode 的起始时间戳（from_timestamp）。
            latest_duration_in_s = latest_ep[f"videos/{video_key}/to_timestamp"][0]

            if latest_size_in_mb + ep_size_in_mb >= self.meta.video_files_size_in_mb:
                # Move temporary episode video to a new video file in the dataset
                # 中文：超限 → 另开一个文件（本条 episode 的起始时间戳归零）。
                chunk_idx, file_idx = update_chunk_file_indices(
                    chunk_idx, file_idx, self.meta.chunks_size
                )
                new_path = self.root / self.meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep_path), str(new_path))
                latest_duration_in_s = 0.0
            else:
                # Update latest video file
                # 中文：装得下 → 把临时 mp4 追加到已有文件尾部（保持一个文件放多条 episode 的设计）。
                concatenate_video_files(
                    [latest_path, ep_path],
                    latest_path,
                )

        # Remove temporary directory
        # 中文：正式文件已经就位，删掉临时目录（连里面的临时 mp4 一起）。
        shutil.rmtree(str(ep_path.parent))

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if episode_index == 0:
            # 只用第 0 条 episode 的视频去写 info.json 里的视频元信息（后续假定编码参数一致）。
            self.meta.update_video_info(video_key)
            write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

        # 交给 meta.save_episode 写进 episode 元数据行：
        # 读取时先按 ep_idx 找到文件，再加上 from_timestamp 就是文件内的绝对时间。
        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": latest_duration_in_s,
            f"videos/{video_key}/to_timestamp": latest_duration_in_s + ep_duration_in_s,
        }
        return metadata

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        # Clean up image files for the current episode buffer
        # Reset the buffer
        # 中文：重建一个空缓冲，准备录下一条 episode。
        # （参数 delete_images 在当前实现里没有被使用；临时图片由视频编码流程负责删除。）
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        """启动异步图片写盘器（录制时把落图任务丢给线程/进程池，避免拖慢采样循环）。"""
        if isinstance(self.image_writer, AsyncImageWriter):
            # 注意：这里引用的 AsyncImageWriter 在本文件里并未 import（上游代码原样保留）。
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        # 中文：AsyncImageWriter 不能被 pickle，所以在把数据集交给 DataLoader
        # （num_workers>0，需要 pickle 到子进程）之前必须先把它停掉。
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        # 阻塞等待所有异步写图任务完成（算统计量 / 编码视频前必须保证图片已落盘）。
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        # 中文：在 root 下开一个临时目录，把该 episode 该相机的整目录图片编码成一个 mp4；
        # 编码完成后删掉图片目录（图片只是中间产物）。
        temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        img_dir = self._get_image_file_dir(episode_index, video_key)
        encode_video_frames(img_dir, temp_path, self.fps, overwrite=True)
        shutil.rmtree(img_dir)
        return temp_path

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
        batch_encoding_size: int = 1,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data.

        中文（写路径入口）：从零造一个空数据集，用来录制新数据。
        注意它同样用 cls.__new__(cls) 手工搭对象（因为 __init__ 是“加载已有数据”的语义）：
          · 先让 LeRobotDatasetMetadata.create 建目录、写空 info.json；
          · 再补齐读路径需要的那些属性（hf_dataset 用 create_hf_dataset 造空骨架，
            delta_timestamps / episodes 等留空），使对象“看起来”和正常加载出来的一样。
        之后典型用法：obj.add_frame(frame) 逐帧写 → obj.save_episode() → obj.finalize()。
        """
        obj = cls.__new__(cls)
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
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            # 指定了进程/线程数就启用异步写图（录制时更快）。
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        # 建一个 episode 空缓冲；之后每次 add_frame 都往里追加。
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        # 空骨架数据集：有 schema、没有数据行。
        obj.hf_dataset = obj.create_hf_dataset()
        # 录制场景不需要这些（它们是为读服务配置的）。
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.writer = None
        obj.latest_episode = None
        obj._current_file_start_frame = None
        # Initialize tracking for incremental recording
        obj._lazy_loading = False
        obj._recorded_frames = 0
        obj._writer_closed_for_reading = False
        return obj


# =============================================================================
# 第 3 层：MultiLeRobotDataset —— N 个数据集目录“首尾拼接”
# =============================================================================
# 这是上层 BaseLerobotDataset 直接持有的对象（self.multi_dataset），职责很单一：
#   · 为每个目录建一个 LeRobotDataset（一个目录 = 一个 embodiment 的一份数据）；
#   · 对外暴露“一条连续的一维下标空间”= 各子数据集帧数之和；
#   · __getitem__(idx) 先定位是“第几个目录 + 目录内第几帧”，再转发给对应子数据集，
#     并额外塞一个 dataset_index 字段标出这条样本来自哪个目录。
# 拼接示意（3 个目录，各 100 / 50 / 30 帧）：
#   全局 idx:  0        99 | 100      149 | 150      179
#   目录:      目录 0        | 目录 1         | 目录 2
#   本地下标:  0..99         | 0..49          | 0..29
# 注意：这里“拼接”的是帧下标，不是 episode（episode 编号是各目录内部自己的编号），
# 所以跨目录的 episode 统计/切分由上层用 dataset_index 或帧区间来处理。
class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        load_images: bool = True,
        video_backend: str | None = None,
        in_memory: bool = False,
    ):
        """构造多个子数据集。

        参数（上层 base_lerobot_dataset.py 的传法）：
          dataset_dirs  数据集目录列表（每个目录一个 LeRobotDataset）。
          episodes      {目录名: [episode 下标]}；本项目 v3 路径传 None（= 每目录全部 episode）。
          tolerances_s  {目录名: 容差秒数}；上层统一给 0.4/data_fps。
          in_memory     True 时透传给每个子数据集（把用到的列搬进 RAM）。
        """
        super().__init__()
        self.dataset_dirs = dataset_dirs
        # root 只是“这些目录的父级/默认缓存位置”，每个子数据集实际用的是 ds_roots[i]。
        self.root = Path(root) if root else HF_LEROBOT_HOME
        ds_roots = [Path(ds_dir) for ds_dir in dataset_dirs]
        ds_names = [ds_dir for ds_dir in dataset_dirs]
        self.ds_names = ds_names
        self.ds_roots = ds_roots
        # 没传容差时给 0.0001（近乎严格）；真实训练走上层传进来的 0.4/data_fps。
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(ds_names, 0.0001)
        self.load_images = load_images
        self.in_memory = in_memory
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        # 中文：上面这句上游注释其实有点过时——delta_timestamps 明明也一起透传给了子数据集
        # （见下面 LeRobotDataset(...) 的参数）。真正由本类统一持有的，是
        # “图像变换 / load_images 开关的解释权”，所以 set_load_images 要向下同步。

        self._datasets = []
        for ds_root, ds_name in zip(ds_roots, ds_names, strict=True):
            try:
                # 一个目录一个子数据集；episodes 参数按目录名取（本项目是 None → 全量）。
                _dataset = LeRobotDataset(
                    ds_name,
                    root=ds_root,
                    episodes=episodes[ds_name] if episodes else None,
                    image_transforms=image_transforms,
                    delta_timestamps=delta_timestamps,
                    tolerance_s=self.tolerances_s[ds_name],
                    download_videos=download_videos,
                    load_images=load_images,
                    video_backend=video_backend,
                    in_memory=self.in_memory,
                )
                self._datasets.append(_dataset)
            except Exception as e:
                # logging.error(e)
                # 中文：某个目录坏了（缺 parquet / 缺 mp4 / 路径不对）不会让整个训练崩掉，
                # 而是打印错误与堆栈后“跳过这个目录”。排障时请留意这条日志，
                # 因为跳过会让实际样本数少于预期（上层不会感知）。
                logging.error(f"Exception while process ds_root: {ds_root}, ds_name: {ds_name}")
                traceback.print_exc()
                continue

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        # 中文：求所有子数据集 features 的交集（只有所有目录都有的列才是安全可用的，
        # 否则 DataLoader 的默认 collate 会因列不一致而报错）。
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        # 上游原本会把“非交集列”加进 disabled_features 并在 __getitem__ 里删掉；
        # 本仓库把这段注掉了（disabled_features 恒为空集），即不做自动裁剪，
        # 由上层 shape_meta 明确声明需要哪些列来保证一致性。
        # for ds_name, ds in zip(self.ds_names, self._datasets, strict=True):
        #     # extra_keys = set(ds.features).difference(intersection_features)
        #     logging.warning(
        #         f"keys {extra_keys} of {ds_name} were disabled as they are not contained in all the "
        #         "other datasets."
        #     )
        #     self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        # 中文：跨机器人形态直接聚合统计量在语义上不严谨（不同关节范围不可比），
        # 因此这里不算 stats，交给上层按 embodiment 分别计算/聚合。
        # self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
        self.stats = None

    def set_during_training(self, during_training: bool):
        """统一切换训练态：during_training=False 时子数据集不再解码视频帧（录制/只读数值时省算力）。"""
        for dataset in self._datasets:
            dataset.during_training = during_training

    def set_load_images(self, load_images: bool):
        """统一开关图像：

        上层（base_lerobot_dataset.py）在复用缓存的 MultiLeRobotDataset 之后会调它，
        确保所有子数据集与本实例的 load_images 状态一致，避免复用实例带着上一个使用者的开关。
        """
        self.load_images = load_images
        for dataset in self._datasets:
            dataset.load_images = load_images

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.

        中文：目录名 → 序号（即 __getitem__ 里塞进 item["dataset_index"] 的那个编号）。
        ⚠️ v3 这里写的是 self.repo_ids，但本文件的 __init__ 只定义了 self.ds_names
        （v2.1 版本用的是 ds_names），所以这个属性一调用就会 AttributeError。
        本项目没有用到它，读到时知道是上游遗留的笔误即可，不要照抄这种写法。
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        # 取第 0 个子数据集的 fps 作为整体 fps（前提是所有子集 fps 一致，上层会保证）。
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        # info.json 里没有 "video" 字段时默认 False（纯图片数据集）。
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        """所有子数据集 features 的并集（键相同时后面的子集覆盖前面的）。"""
        features = {}
        for dataset in self._datasets:
            features.update(
                {k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features}
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        # 这里用 HF features 的类型对象判断：datasets.Image（内联图片）或 VideoFrame（需要解码的视频）。
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image | VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        # 只挑“视频型”的相机（读取时要解码），是 camera_keys 的子集。
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        # 一维下标空间的长度 = 所有子数据集帧数之和。
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""

        # 所有子数据集的 episode 数之和（注意 episode 编号在子数据集内部各自从 0 开始）。
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        # 中文：整体容差的兜底值 ≈ 一帧的时长（1/fps）再留 1e-4 的数值误差余量。
        # 注意它和 __init__ 里 tolerances_s 的 1e-4 是两回事：那个是“查帧时间容差”，
        # 这个是本属性的默认值；实际训练用的是上层传入的 0.4/data_fps。
        return 1 / self.fps - 1e-4

    def get_episode_data(self, episode_idx: int) -> dict:
        """整条读取某条 episode 的全部数值列（不做 delta 查询、不解码视频）。

        用途：上层算数据集统计量 / 抽查数据时，需要“一次性拿到整条 episode 的所有帧”，
        __getitem__ 那种逐帧 delta 取法不适合，所以单独提供这个批量读接口。

        episode_idx 是“跨目录拼接后”的编号：依次尝试各子数据集，超出就减去该目录的
        episode 数继续往后找（这也是循环里的 episode_idx -= dataset.num_episodes 的作用）。
        """
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                # 定位到“这个目录的第 episode_idx 条 episode 所在的 parquet 文件”。
                # 注意 dataset.episodes[episode_idx] 是“目录内的 episode 下标”（前面已做过平移）。
                file = str(
                    dataset.root / dataset.meta.get_data_file_path(dataset.episodes[episode_idx])
                )
                # 只读该数据集该读的列（video 列除外，image 列看 load_images）。
                table = pq.read_table(str(file), columns=dataset._episode_read_columns())
                result_dict = {}
                for col_name in table.column_names:
                    col = table[col_name]
                    try:
                        # 走零拷贝（要求列布局允许，列表列常常不允许 → 落到下面的兜底）。
                        np_arr = col.to_numpy(zero_copy_only=True)
                    except Exception:
                        # 兜底：先转成 python 对象数组，再按需要 stack 成二维数组。
                        raw = col.to_numpy()
                        np_arr = np.stack(raw) if raw.dtype == object else raw
                    with warnings.catch_warnings():
                        # torch.from_numpy 会对只读数组发告警（NumPy 较新版本），这里静音。
                        warnings.filterwarnings(
                            "ignore",
                            message="The given NumPy array is not writable",
                            category=UserWarning,
                        )
                        # deal with string in parquet file
                        # 中文：object 数组（文本列）直接原样返回 numpy，其余转成 torch.Tensor。
                        # 注意：整个 episode 的所有帧一起返回，第一维是帧（不做窗口切分）。
                        if np_arr.dtype == "O":
                            result_dict[col_name] = np_arr
                        else:
                            result_dict[col_name] = torch.from_numpy(np_arr)
                return result_dict
            else:
                # 不在这个子数据集里 → 跳到下一个目录，把它当作“新的第 episode_idx 条”。
                episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def __len__(self):
        # 拼接后的总样本数（所有目录的帧数之和）。
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """把全局帧下标 idx 映射到“第几个子数据集 + 目录内第几帧”，然后转发取数。

        例：子数据集帧数 [100, 50, 30]，idx=120 → 落在第 1 个目录，本地下标 20
            → self._datasets[1][20]，并在返回的 dict 里加上 dataset_index=1。
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        # 中文：顺序遍历各目录累加帧数，找到 idx 落在哪个目录里（目录数量通常很小，
        # 因此这里用线性扫描而不是二分；上层 BaseLerobotDataset 会另建一张二分查找表）。
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            # for-else：循环没有 break（说明下标越界）才会走到这里。
            raise AssertionError(
                "We expect the loop to break out as long as the index is within bounds."
            )
        # 转发给对应子数据集；本地下标 = 全局下标 - 该目录的起始帧号。
        item = self._datasets[dataset_idx][idx - start_idx]
        # 标出这条样本来自哪个目录（上层用于定位/分 embodiment 处理）。
        item["dataset_index"] = torch.tensor(dataset_idx)
        # disabled_features 恒为空集（见 __init__ 的说明），这段目前不会删任何字段。
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        # 概览打印：目录、样本数、episode 数、视频/图片、fps、相机键。
        # ⚠️ 这里同样引用了未定义的 self.repo_ids（应为 self.ds_names），
        # 所以 v3 版本 print() 这个数据集对象会抛 AttributeError（本项目未使用）。
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
