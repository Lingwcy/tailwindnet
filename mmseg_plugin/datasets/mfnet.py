from __future__ import annotations

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


MFNET_METAINFO = dict(
    classes=(
        'background',
        'car',
        'person',
        'bike',
        'curve',
        'car stop',
        'guardrail',
        'color cone',
        'bump',
    ),
    palette=[
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
    ],
)


@DATASETS.register_module()
class MFNetDataset(BaseSegDataset):
    """MFNet RGB-T semantic segmentation dataset."""

    METAINFO = MFNET_METAINFO

    def __init__(
        self,
        img_suffix: str = '.png',
        seg_map_suffix: str = '.png',
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs,
        )
