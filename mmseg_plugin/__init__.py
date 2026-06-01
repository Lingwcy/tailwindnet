# Copyright (c) TailwindNet authors. All rights reserved.
"""TailwindNet: paper implementation for the TailwindV6 multimodal segmentor.

This package is an mmsegmentation plugin. Importing ``mmseg_plugin`` registers
the paper's Tailwind backbone and segmentor into mmseg's ``MODELS`` registry.
"""

from .models import backbones, segmentors  # noqa: F401
from .utils import custom_modules  # noqa: F401

__version__ = '0.1.0'
