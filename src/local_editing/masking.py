"""Mask utilities for real local-editing experiments."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageOps


def box_mask(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(bbox, fill=255)
    return mask


def dilate_mask(mask: Image.Image, radius: int = 8) -> Image.Image:
    return mask.convert("L").filter(ImageFilter.MaxFilter(radius * 2 + 1))


def full_image_mask(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 255)


def select_real_mask(
    strategy: str,
    image: Image.Image,
    sam_mask: Image.Image,
    bbox: tuple[int, int, int, int],
    dilation_radius: int = 8,
    oracle_mask: Image.Image | None = None,
) -> Image.Image:
    sam_mask = resize_mask(sam_mask, image.size)
    if oracle_mask is not None:
        oracle_mask = resize_mask(oracle_mask, image.size)

    if strategy == "box":
        return box_mask(image.size, bbox)
    if strategy == "sam":
        return sam_mask.convert("L")
    if strategy == "dilated_sam":
        return dilate_mask(sam_mask, radius=dilation_radius)
    if strategy == "oracle":
        if oracle_mask is None:
            raise ValueError("oracle mask strategy requires target_mask/mask_path in metadata")
        return oracle_mask.convert("L")
    if strategy == "dilated_oracle":
        if oracle_mask is None:
            raise ValueError("dilated_oracle mask strategy requires target_mask/mask_path in metadata")
        return dilate_mask(oracle_mask, radius=dilation_radius)
    if strategy == "full":
        return full_image_mask(image.size)
    raise ValueError(f"Unknown mask strategy: {strategy}")


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    mask = mask.convert("L")
    if mask.size == size:
        return mask
    return mask.resize(size, Image.Resampling.NEAREST)


def invert_mask(mask: Image.Image) -> Image.Image:
    return ImageOps.invert(mask.convert("L"))
