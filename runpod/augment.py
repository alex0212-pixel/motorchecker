"""Image augmentation utilities for motor sticker finetuning.

Goals
- Make the model robust to:
  - rotation / upside-down
  - perspective / shear / mild warp
  - motion blur / noise / illumination

We keep augmentations *label-preserving*: since the label is the sticker's
number+color, aggressive crops are avoided.

This module is imported by runpod/train_qwen2vl_lora.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class AugmentConfig:
    enabled: bool = True
    p: float = 0.85
    # Allow full 180/360-style rotations so the model learns upside-down digits
    max_rotate: int = 180


def _build_albu(cfg: AugmentConfig):
    import albumentations as A

    # NOTE: keep crop out to not lose the sticker region.
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Rotate(limit=cfg.max_rotate, border_mode=0, value=(0, 0, 0), p=1.0),
                    A.Perspective(scale=(0.02, 0.08), keep_size=True, pad_mode=0, pad_val=(0, 0, 0), p=1.0),
                    A.Affine(
                        rotate=(-cfg.max_rotate, cfg.max_rotate),
                        shear=(-12, 12),
                        translate_percent=(-0.02, 0.02),
                        mode=0,
                        cval=(0, 0, 0),
                        p=1.0,
                    ),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                ],
                p=0.25,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.35),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=8, p=0.25),
        ]
    )


def augment_pil(img: Image.Image, cfg: Optional[AugmentConfig] = None) -> Image.Image:
    """Apply stochastic augmentation to PIL image.

    The augmentation runs on CPU and returns a new PIL.Image.
    """

    if cfg is None:
        cfg = AugmentConfig()
    if not cfg.enabled:
        return img

    # lazy import to keep base environment light
    import random

    if random.random() > cfg.p:
        return img

    albu = _build_albu(cfg)
    arr = np.array(img)
    out = albu(image=arr)["image"]
    return Image.fromarray(out)

