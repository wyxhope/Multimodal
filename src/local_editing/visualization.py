"""Visualization helpers for real-model editing outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .metrics import pixel_change


def overlay_mask(image: Image.Image, mask: Image.Image, color=(255, 40, 40), alpha=0.38) -> Image.Image:
    base = image.convert("RGB")
    overlay = Image.new("RGB", base.size, color)
    mask_alpha = mask.convert("L")
    if mask_alpha.size != base.size:
        mask_alpha = mask_alpha.resize(base.size, Image.Resampling.NEAREST)
    mask_alpha = mask_alpha.point(lambda v: int(v * alpha))
    blended = Image.composite(overlay, base, mask_alpha)
    return blended


def draw_box(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, outline=(255, 35, 35), width=3)
    return out


def difference_map(original: Image.Image, edited: Image.Image) -> Image.Image:
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    diff = pixel_change(original, edited)
    diff = (255 * diff / (diff.max() + 1e-8)).astype(np.uint8)
    return Image.fromarray(diff, mode="L").convert("RGB")


def save_case_grid(
    original: Image.Image,
    mask: Image.Image,
    bbox_image: Image.Image,
    edited: Image.Image,
    output_path: Path,
    title: str,
) -> None:
    panels = [
        ("Original", original),
        ("Detected box", bbox_image),
        ("Mask", overlay_mask(original, mask)),
        ("Edited", edited),
        ("Difference", difference_map(original, edited)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3))
    for ax, (label, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
