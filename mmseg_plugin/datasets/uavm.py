from __future__ import annotations

import os.path as osp
from typing import Dict, List, Optional

import mmengine
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


PRUNED_METAINFO = dict(
    classes=(
        'background',
        'canadian_goldenrod',
        'road',
        'pine_tree',
        'shrub',
        'reed',
    ),
    palette=[
        [0, 0, 0],
        [235, 235, 90],
        [128, 64, 128],
        [34, 139, 34],
        [107, 142, 35],
        [0, 255, 127],
    ],
)


PRUNED_LABEL_MAP: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 0,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
}


@DATASETS.register_module()
class UAVMDataset(BaseSegDataset):
    """MPS/UAVM multispectral semantic segmentation dataset."""

    METAINFO = PRUNED_METAINFO

    def __init__(
        self,
        img_suffix: str = '.npy',
        seg_map_suffix: str = '.png',
        label_map: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> None:
        if label_map is None:
            label_map = PRUNED_LABEL_MAP
        self._uavm_label_map = label_map
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs,
        )
        self.label_map = label_map

    def load_data_list(self) -> List[dict]:
        img_dir = self.data_prefix.get('img_path', None)
        seg_dir = self.data_prefix.get('seg_map_path', None)
        data_list: List[dict] = []

        if self.ann_file and osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                if not img_name:
                    continue

                stem, ext = osp.splitext(img_name)
                if ext:
                    img_filename = img_name
                else:
                    stem = img_name
                    img_filename = f'{img_name}{self.img_suffix}'

                data_info = dict(
                    img_path=osp.join(img_dir, img_filename),
                    label_map=self._uavm_label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                )

                if seg_dir is not None:
                    data_info['seg_map_path'] = osp.join(
                        seg_dir, f'{stem}{self.seg_map_suffix}')

                data_list.append(data_info)
            return data_list

        data_list = super().load_data_list()
        for item in data_list:
            item['label_map'] = self._uavm_label_map
        return data_list
