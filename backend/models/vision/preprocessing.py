import torch
from torchvision import transforms
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

def get_base_data_config(timm_name: str, input_size: int = 224):
    """
    Resolve model-specific data configuration directly from the timm registry.
    This guarantees that the preprocessing normalization, interpolation, and crop
    percentages match the exact pretrained weights.
    """
    return resolve_model_data_config({"architecture": timm_name, "input_size": (3, input_size, input_size)})

def get_region_tensor_transform(timm_name: str, input_size: int = 224, is_training: bool = False):
    """
    Returns the *non-geometric* region-level tensor preprocessing.
    Geometric transformations must occur on the full note BEFORE region extraction.
    """
    config = get_base_data_config(timm_name, input_size)
    
    # We use timm's create_transform which handles mean/std normalization and resizing.
    # We enforce is_training=False even during our pipeline's training because 
    # we apply our forensic-safe augmentations to the full image manually beforehand.
    return create_transform(
        input_size=config["input_size"],
        interpolation=config["interpolation"],
        mean=config["mean"],
        std=config["std"],
        crop_pct=config["crop_pct"],
        is_training=False  # Do not let timm apply random flips or crops to our regions!
    )

def get_forensic_safe_geometric_transform():
    """
    A single, consistent whole-note augmentation pipeline for training.
    Does NOT contain flips. Rotation is limited to ±3 degrees.
    Contains mild perspective, noise, blur, and color variation.
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=3, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        # Mild scale variation without losing content
        transforms.RandomResizedCrop(
            size=None, # Will be resized later, just defining scale
            scale=(0.95, 1.0),
            ratio=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR
        ) if False else transforms.Lambda(lambda x: x), # Disable random crop to keep full note intact, just rely on perspective
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        # Mild JPEG compression / noise simulation could be added here
    ])

def get_simclr_geometric_transform():
    """
    More aggressive augmentations for SSL (SimCLR).
    Still avoids flips and large rotations to preserve currency topologies.
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=3, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])
