from __future__ import annotations

from typing import Optional

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList
from torch import Tensor

from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class TailwindEncoderDecoder(EncoderDecoder):
    """EncoderDecoder variant that appends Tailwind-specific auxiliary losses."""

    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = super().loss(inputs, data_samples)

        if hasattr(self.backbone, 'get_fusion_alignment_loss'):
            alignment_loss = self.backbone.get_fusion_alignment_loss()
            if alignment_loss is not None:
                losses['loss_alignment'] = alignment_loss

        return losses
