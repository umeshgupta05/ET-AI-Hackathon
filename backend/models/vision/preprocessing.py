"""Shared preprocessing and forensic-safe augmentation for currency vision models."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from timm import get_pretrained_cfg
from timm.data.transforms_factory import create_transform


@dataclass(frozen=True)
class VisionDataConfig:
    input_size: tuple[int, int, int]
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": list(self.input_size),
            "interpolation": self.interpolation,
            "mean": list(self.mean),
            "std": list(self.std),
            "crop_pct": self.crop_pct,
        }


def get_base_data_config(timm_name: str, input_size: int = 224) -> VisionDataConfig:
    """Resolve preprocessing from the exact timm pretrained configuration."""
    cfg = get_pretrained_cfg(timm_name)
    if cfg is None:
        return VisionDataConfig(
            input_size=(3, input_size, input_size),
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0,
        )

    resolved_size = tuple(cfg.input_size or (3, input_size, input_size))
    if len(resolved_size) != 3:
        resolved_size = (3, input_size, input_size)

    return VisionDataConfig(
        input_size=(int(resolved_size[0]), int(resolved_size[1]), int(resolved_size[2])),
        interpolation=str(cfg.interpolation or "bicubic"),
        mean=tuple(float(v) for v in (cfg.mean or (0.485, 0.456, 0.406))),
        std=tuple(float(v) for v in (cfg.std or (0.229, 0.224, 0.225))),
        crop_pct=float(cfg.crop_pct or 1.0),
    )


def get_region_tensor_transform(timm_name: str, input_size: int = 224):
    """Return deterministic region preprocessing with no random geometry."""
    cfg = get_base_data_config(timm_name, input_size)
    return create_transform(
        input_size=cfg.input_size,
        interpolation=cfg.interpolation,
        mean=cfg.mean,
        std=cfg.std,
        crop_pct=cfg.crop_pct,
        is_training=False,
    )


class RandomJPEGCompression:
    """Apply mild JPEG compression while preserving the full note geometry."""

    def __init__(self, probability: float = 0.25, quality_range: tuple[int, int] = (82, 98)):
        self.probability = probability
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomSensorNoise:
    """Apply low-amplitude camera noise without moving security features."""

    def __init__(self, probability: float = 0.25, sigma_range: tuple[float, float] = (1.0, 4.0)):
        self.probability = probability
        self.sigma_range = sigma_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        sigma = random.uniform(*self.sigma_range)
        array += np.random.normal(0.0, sigma, array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


class RandomMildBlur:
    def __init__(self, probability: float = 0.15, radius_range: tuple[float, float] = (0.1, 0.7)):
        self.probability = probability
        self.radius_range = radius_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return image.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius_range)))


def get_forensic_safe_geometric_transform():
    """One shared whole-note transform applied before semantic region extraction."""
    return transforms.Compose(
        [
            transforms.RandomRotation(
                degrees=3,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=255,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.06,
                p=0.25,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=255,
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.08,
                hue=0.01,
            ),
            RandomMildBlur(probability=0.12),
            RandomJPEGCompression(probability=0.25),
            RandomSensorNoise(probability=0.20),
        ]
    )


def get_simclr_geometric_transform():
    """Currency-aware SSL transform: stronger capture variation, no mirroring."""
    return transforms.Compose(
        [
            transforms.RandomRotation(
                degrees=3,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=255,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.10,
                p=0.45,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=255,
            ),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.12,
                hue=0.02,
            ),
            RandomMildBlur(probability=0.20, radius_range=(0.1, 1.0)),
            RandomJPEGCompression(probability=0.35, quality_range=(75, 96)),
            RandomSensorNoise(probability=0.30, sigma_range=(1.0, 6.0)),
        ]
    )