import numpy as np
from numpy import random

# type: ignore is required because the stub for mmcv is not shipped with the repo.
import mmcv
import mmengine.fileio as fileio
from mmcv.transforms import BaseTransform  # type: ignore[attr-defined]
from mmcv.transforms import LoadImageFromFile
import pickle
from pathlib import Path
from typing import Optional

import cv2

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromNpyFile(LoadImageFromFile):
    """Load either ``.npy``/``.npz`` arrays or normal image files."""

    def transform(self, results: dict) -> Optional[dict]:
        filename = results['img_path']

        try:
            if Path(filename).suffix in ['.npy', '.npz']:
                img = np.load(filename)
            else:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            raise e

        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class MFNetRandomEdgeCrop(BaseTransform):
    """Randomly shave the image borders by a percentage before resizing.

    This replicates the random cropping augmentation used in the original
    MFNet PyTorch implementation, where a random portion (up to ``crop_rate``)
    of each border is removed before the image (and corresponding labels) are
    resized back to the training size.

    Args:
        crop_rate (float): The maximum fraction of the height/width that may
            be removed from each side. Defaults to ``0.1`` which matches the
            reference project (10%).
        prob (float): Probability of applying the transform. Defaults to 1.0.
    """

    def __init__(self, crop_rate: float = 0.1, prob: float = 1.0) -> None:
        assert 0 <= crop_rate < 0.5, 'crop_rate should be in [0, 0.5).'
        assert 0.0 <= prob <= 1.0, 'prob must be in [0, 1].'
        self.crop_rate = crop_rate
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if random.rand() >= self.prob:
            return results

        img = results['img']
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return results

        max_delta_h = int(round(h * self.crop_rate))
        max_delta_w = int(round(w * self.crop_rate))

        top = random.randint(0, max_delta_h + 1) if max_delta_h > 0 else 0
        left = random.randint(0, max_delta_w + 1) if max_delta_w > 0 else 0
        bottom = random.randint(h - max_delta_h, h + 1) if max_delta_h > 0 else h
        right = random.randint(w - max_delta_w, w + 1) if max_delta_w > 0 else w

        bottom = np.clip(bottom, top + 1, h)
        right = np.clip(right, left + 1, w)

        results['img'] = img[top:bottom, left:right, ...]
        results['img_shape'] = results['img'].shape[:2]
        results['pad_shape'] = results['img_shape']

        for key in results.get('seg_fields', []):
            results[key] = results[key][top:bottom, left:right]

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(crop_rate={self.crop_rate}, '
                f'prob={self.prob})')


@TRANSFORMS.register_module()
class UAVMToRGBThermal(BaseTransform):
    """Convert 13-channel UAVM tensors into 4-channel RGB + thermal inputs.

    MFNet expects a 4-channel image where the first three channels are RGB and
    the last channel corresponds to a single thermal/infrared map. UAVM
    provides 13 channels (RGB + 10 hyperspectral bands). This transform keeps
    the RGB channels and aggregates the specified ``thermal_indices`` into one
    channel via a reduction method (``mean`` by default).

    Args:
        rgb_indices (tuple[int]): Indices of the RGB channels. Defaults to
            ``(0, 1, 2)`` which matches the dataset ordering.
        thermal_indices (tuple[int]): Indices used to build the thermal
            channel. Defaults to ``(3, ..., 12)`` to include all NIR bands.
        reduction (str): Aggregation method for the thermal channel. Supported
            options are ``'mean'``, ``'max'`` and ``'median'``.
    """

    def __init__(
            self,
            rgb_indices=(0, 1, 2),
            thermal_indices=tuple(range(3, 13)),
            reduction='mean') -> None:
        assert len(rgb_indices) == 3, 'rgb_indices must contain 3 channels.'
        assert len(thermal_indices) > 0, 'thermal_indices cannot be empty.'
        valid_reductions = {'mean', 'max', 'median'}
        if reduction not in valid_reductions:
            raise ValueError(
                f'Reduction must be one of {valid_reductions}, got {reduction}.')
        self.rgb_indices = tuple(rgb_indices)
        self.thermal_indices = tuple(thermal_indices)
        self.reduction = reduction

    def _reduce(self, thermal_stack: np.ndarray) -> np.ndarray:
        if self.reduction == 'mean':
            return thermal_stack.mean(axis=2, keepdims=True)
        if self.reduction == 'max':
            return thermal_stack.max(axis=2, keepdims=True)
        return np.median(thermal_stack, axis=2, keepdims=True)

    def transform(self, results: dict) -> dict:
        img = results['img']
        if img.ndim != 3:
            raise ValueError('UAVMToRGBThermal expects an HWC tensor.')
        max_index = max(self.rgb_indices + self.thermal_indices)
        if img.shape[2] <= max_index:
            raise ValueError(
                f'Image has {img.shape[2]} channels, but index {max_index} is requested.')

        rgb = img[..., list(self.rgb_indices)]
        thermal_stack = img[..., list(self.thermal_indices)]
        thermal = self._reduce(thermal_stack)
        if thermal.dtype != img.dtype:
            thermal = thermal.astype(img.dtype)

        fused = np.concatenate([rgb, thermal], axis=2)
        results['img'] = fused
        results['img_shape'] = fused.shape[:2]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(rgb_indices={self.rgb_indices}, '
                f'thermal_indices={self.thermal_indices}, '
                f'reduction={self.reduction!r})')


@TRANSFORMS.register_module()
class SelectUAVMChannels(BaseTransform):
    """Slice the loaded UAVM tensor to a specific set of channels.

    This utility is shared by ablation configs that need RGB-only or
    NIR-only inputs without touching the dataset files.

    Args:
        indices (Sequence[int]): Channel indices to keep, in the order they
            should appear in the output. Defaults to ``(0, 1, 2)``.
    """

    def __init__(self, indices=(0, 1, 2)) -> None:
        if isinstance(indices, int):
            indices = (indices,)
        if len(indices) == 0:
            raise ValueError('indices cannot be empty for SelectUAVMChannels.')
        self.indices = tuple(int(i) for i in indices)

    def transform(self, results: dict) -> dict:
        img = results['img']
        if img.ndim != 3:
            raise ValueError('SelectUAVMChannels expects an HWC tensor.')
        max_index = max(self.indices)
        if img.shape[2] <= max_index:
            raise ValueError(
                f'Image has {img.shape[2]} channels, but index {max_index} was requested.')
        results['img'] = img[..., list(self.indices)]
        results['img_shape'] = results['img'].shape[:2]
        results['pad_shape'] = results['img_shape']
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(indices={self.indices})'


@TRANSFORMS.register_module()
class AppendSAMMasksFromCache(BaseTransform):
    """Append SAM masks from an offline cache to UAVM images.

    该变换假设你已经使用 ``tools/generate_uavm_sam_cache.py``
    为每个 UAVM 样本生成了若干个 SAM 掩码，并将其保存为一个
    ``.pkl`` 文件，格式大致为::

        {
            'image_stem': np.ndarray (K, H, W),  # K 个二值掩码
            ...
        }

    在数据流水线中，本模块会：
    1. 根据 ``results['img_path']`` 的文件名查找对应的掩码；
    2. 将前 ``use_mask_num`` 个掩码 resize 到图像分辨率；
    3. 在通道维度上将它们拼接到 ``results['img']`` 后面；
    4. 更新 ``img_shape`` 与 ``pad_shape``。

    Args:
        cache_file (str): SAM 掩码缓存文件路径（`.pkl`）。
        use_mask_num (int): 实际拼接到图像上的掩码数量。
    """

    def __init__(self, cache_file: str, use_mask_num: int = 4) -> None:
        if not cache_file:
            raise ValueError('cache_file must be a valid path.')
        self.cache_file = str(cache_file)
        self.use_mask_num = int(use_mask_num)

        cache_path = Path(self.cache_file)
        if not cache_path.is_file():
            raise FileNotFoundError(f'SAM cache file not found: {cache_path}')

        with cache_path.open('rb') as f:
            self.mask_cache = pickle.load(f)

    def transform(self, results: dict) -> dict:
        img = results.get('img', None)
        img_path = results.get('img_path', None)
        if img is None or img_path is None:
            return results

        if img.ndim != 3:
            # 只支持 HWC 格式
            return results

        key = Path(img_path).stem
        masks = self.mask_cache.get(key, None)
        if masks is None or len(masks) == 0:
            # 没有对应掩码则直接返回
            return results

        # masks: (K, Hm, Wm)
        masks = np.asarray(masks)
        k = min(self.use_mask_num, masks.shape[0])
        masks = masks[:k]

        h, w = img.shape[:2]
        resized_masks = []
        for m in masks:
            if m.shape[0] != h or m.shape[1] != w:
                m_resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                m_resized = m
            resized_masks.append(m_resized.astype(img.dtype))

        if not resized_masks:
            return results

        mask_stack = np.stack(resized_masks, axis=-1)  # (H, W, K)

        # 拼接到原图像通道之后
        fused = np.concatenate([img, mask_stack], axis=2)
        results['img'] = fused
        results['img_shape'] = fused.shape[:2]
        results['pad_shape'] = fused.shape[:2]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(cache_file={self.cache_file!r}, '
                f'use_mask_num={self.use_mask_num})')
