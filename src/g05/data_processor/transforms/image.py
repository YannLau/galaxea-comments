# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""项目自定义的图像预处理与数据增强算子。

本模块中的类都继承 :class:`torch.nn.Module`，因此既可以像普通函数一样调用，
也可以在 ``configs/data/_transforms.yaml`` 中通过 Hydra 的 ``_target_`` 动态实例化。
处理器会按配置顺序逐个执行这些算子（参见 ``BaseProcessor.preprocess``）。

仓库中的图像张量通常采用 ``[T, C, H, W]`` 排列，其中 ``T`` 是观测帧数；
单张图片也可以是 ``[C, H, W]``。本文件除 ``Pad`` 外的算子都只从末尾定位
``C/H/W``，所以能够保留 ``T`` 等任意前缀维度。输入像素通常是 ``uint8``、
范围 ``[0, 255]``，先经 :class:`ToTensor` 转成 ``float32`` 的 ``[0, 1]``，
之后才进入颜色增强和 ``torchvision.transforms.Normalize``。

注意：相机目标分辨率通常由 ``BaseProcessor`` 根据 ``camera_size_config`` 自动
注入的 ``torchvision.transforms.Resize`` 负责；这里的裁剪类会在裁剪后恢复到
输入尺寸，因此不会破坏处理器对最终形状的校验。
"""

import math

import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF


class ToTensor(nn.Module):
    """把 ``uint8`` 图像转换为归一化到 ``[0, 1]`` 的 ``float32`` 张量。

    这里与 torchvision 中同名变换的目标相似，但输入已经必须是 Tensor，而非
    PIL 图片或 NumPy 数组。严格检查 ``uint8`` 可以及早发现重复缩放：如果已经
    是浮点图像却再次除以 255，数值会错误地缩小。

    输入与输出形状完全相同；典型形状为 ``[T, C, H, W]``。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # 数据集解码出的原始图像应是 8 位无符号整数，像素范围为 [0, 255]。
        assert x.dtype == torch.uint8
        # 先转换类型再除法，避免 uint8 整数运算丢失小数；结果范围变为 [0, 1]。
        x = x.to(torch.float32) / 255.0
        return x


class FlipChannels(nn.Module):
    """反转通道维，将 BGR 与 RGB 两种颜色顺序互相转换。

    输入形状约定为 ``[..., C, H, W]``。通道维使用倒数第 3 维定位，因此既
    支持单图 ``[C, H, W]``，也支持多帧/批量 ``[N, C, H, W]``。本算子只是
    倒序排列全部通道；仓库中用于三通道图像时，效果正是 BGR↔RGB。
    """

    def forward(self, x: torch.Tensor):
        # 只翻转 C 维，空间方向 H/W 以及所有前缀维度均保持不变。
        return x.flip(-3)


class DummyImageTransform(nn.Module):
    """恒等变换：原样返回输入，不复制张量也不改变数值。

    它主要作为配置占位符使用，例如某条流水线不需要任何图像处理时，仍可提供
    一个合法、可调用的 ``nn.Module``，从而避免在处理器中加入特殊分支。
    """

    def forward(self, x: torch.Tensor):
        return x


class SmartResize(nn.Module):
    """保持宽高比的自适应缩放，并让输出边长对齐到 ``factor`` 的整数倍。

    参数：
        max_pixels: 输出面积 ``H×W`` 的上限，常用于控制视觉 token 数量。
        min_pixels: 输出面积的期望下限；传入 0 时取 ``factor²``，保证至少能
            形成一个对齐网格。
        factor: 输出高度和宽度都必须是该值的整数倍。默认 32 通常对应
            ``patch_size × merge_size`` 的空间对齐要求。
        antialias: 缩小时是否启用抗锯齿，默认开启。

    支持任意前缀维度，例如 ``[C, H, W]`` 或 ``[T, C, H, W]``；只会缩放
    最后两个空间维度。算法先按像素预算计算等比例缩放系数，再把两条边分别
    四舍五入到 ``factor`` 的倍数。由于两条边独立取整，最终宽高比可能有轻微
    误差；若取整后面积超过上限，则改用向下取整。

    在 ``_transforms.yaml`` 中的配置示例::

        - _target_: g05.data_processor.transforms.image.SmartResize
          max_pixels: 40960    # 256×160 的面积预算；实际比例由输入分辨率决定
    """

    def __init__(
        self,
        max_pixels: int,
        min_pixels: int = 0,
        factor: int = 32,
        antialias: bool = True,
    ):
        super().__init__()
        self.max_pixels = max_pixels
        # 使用 ``or`` 保留原实现语义：0 表示采用一个 factor×factor 网格的默认下限。
        self.min_pixels = min_pixels or factor * factor
        self.factor = factor
        self.antialias = antialias

    @staticmethod
    def target_size(
        h: int, w: int, max_pixels: int, min_pixels: int, factor: int
    ) -> tuple[int, int]:
        """计算兼顾宽高比、像素预算和倍数对齐的目标 ``(高度, 宽度)``。

        面积超上限时等比缩小，低于下限时等比放大，位于区间内则保持原尺度；
        随后把高度与宽度对齐到 ``factor``。返回值的每一边至少为 ``factor``。

        >>> SmartResize.target_size(720, 1280, 512*288, 32*32, 32)
        (288, 512)
        >>> SmartResize.target_size(480, 640, 256*256, 32*32, 32)
        (224, 288)
        """
        # 面积决定整体缩放率；对面积比开平方，才能得到边长的缩放率。
        area = h * w
        if area > max_pixels:
            scale = math.sqrt(max_pixels / area)
        elif area < min_pixels:
            scale = math.sqrt(min_pixels / area)
        else:
            scale = 1.0
        # round 选择离理想尺寸最近的对齐网格，通常比一律向下取整更保真。
        h_out = round(h * scale / factor) * factor
        w_out = round(w * scale / factor) * factor
        # 四舍五入可能使面积略超预算；这种情况下两条边都改为向下对齐。
        if h_out * w_out > max_pixels:
            h_out = math.floor(h * scale / factor) * factor
            w_out = math.floor(w * scale / factor) * factor
        # 防止极端长宽比或很小的输入让某一边向下取整为 0。
        return max(h_out, factor), max(w_out, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        h_out, w_out = self.target_size(h, w, self.max_pixels, self.min_pixels, self.factor)
        # 尺寸未变化时直接返回原张量，避免一次无意义的插值及其数值误差。
        if h_out == h and w_out == w:
            return x
        # torchvision 会保留所有前缀维度，只对最后的 H/W 做插值。
        return TFF.resize(x, [h_out, w_out], antialias=self.antialias)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_pixels={self.max_pixels}, "
            f"min_pixels={self.min_pixels}, "
            f"factor={self.factor})"
        )


class RandomScaleCrop(nn.Module):
    """随机裁剪图像的一块区域，再缩放回原尺寸，用作训练期数据增强。

    ``scale`` 是每条边可保留比例的下界。每次调用会从 ``[scale, 1]`` 均匀采样
    一个比例，并对高度和宽度使用同一比例，因此裁剪区域与原图宽高比基本一致；
    再随机采样左上角位置。若输入为 ``[T, C, H, W]``，同一组裁剪参数会应用
    到全部 ``T`` 帧，避免时序中画面窗口随机跳动。

    注意这里采样的是“边长比例”，不是严格的面积比例；实际保留面积约为该比例
    的平方。输出形状始终与输入相同。
    """

    def __init__(self, scale: float = 0.95):
        super().__init__()
        assert 0 < scale <= 1.0, f"scale must be in (0, 1], got {scale}"
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        # rand() ∈ [0, 1)，所以保留比例位于 [scale, 1)；scale=1 时恒为 1。
        area_ratio = self.scale + (1.0 - self.scale) * torch.rand(1).item()
        # int 会向下截断到合法像素数；由于 scale>0，正常图像上裁剪尺寸为正数。
        crop_h = int(h * area_ratio)
        crop_w = int(w * area_ratio)
        # randint 的上界不包含在采样范围内，故需 +1 才能取到贴住下/右边界的位置。
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        # 省略所有前缀维度，只切最后的空间维；多帧由此共享同一个裁剪窗口。
        cropped = x[..., top : top + crop_h, left : left + crop_w]
        # 恢复到裁剪前尺寸，使后续 shape_meta 形状断言仍然成立。
        return TFF.resize(cropped, [h, w], antialias=True)


class CenterScaleCrop(nn.Module):
    """按固定比例做中心裁剪，再缩放回原尺寸。

    这是 :class:`RandomScaleCrop` 的确定性对应版本，适合验证/推理阶段：给定相同
    输入总会产生相同输出。``scale`` 表示高度和宽度各自保留的比例；所有前缀帧
    共享同一中心窗口，输出形状与输入一致。
    """

    def __init__(self, scale: float = 0.95):
        super().__init__()
        assert 0 < scale <= 1.0, f"scale must be in (0, 1], got {scale}"
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        # 向下截断为整数像素；scale=1 时裁剪窗口就是完整图像。
        crop_h = int(h * self.scale)
        crop_w = int(w * self.scale)
        # 若两侧无法完全均分多出的像素，整除会让窗口向上/左偏最多 1 个像素。
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped = x[..., top : top + crop_h, left : left + crop_w]
        # 恢复原始 H/W；抗锯齿可减轻插值产生的高频伪影。
        return TFF.resize(cropped, [h, w], antialias=True)


class Pad(nn.Module):
    """对四维图像张量做边缘填充，是 ``torchvision.transforms.Pad`` 的薄封装。

    参数含义与 torchvision 一致：``padding`` 可指定各边宽度，``fill`` 是常量
    填充模式下的填充值，``padding_mode`` 可为 ``constant``、``edge``、
    ``reflect`` 或 ``symmetric``。仓库约定输入必须为 ``[N, C, H, W]``；这里
    显式限制为四维，以免把单图或额外批维误当成合法输入。填充只改变 H/W。
    """

    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()
        # 保存原始参数，便于调试/检查模块；实际计算委托给 torchvision 实现。
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        # 配置系统可能传入 list，而 torchvision 接受 tuple；在此统一类型。
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        # 本仓库在处理器中以 [观测帧数, C, H, W] 的四维形式批量处理图像。
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)
