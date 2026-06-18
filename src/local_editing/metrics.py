"""Metrics for instruction effect and non-target preservation."""

from __future__ import annotations

import numpy as np
from PIL import Image


def to_float(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


def pixel_change(original: Image.Image, edited: Image.Image) -> np.ndarray:
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    return np.mean(np.abs(to_float(original) - to_float(edited)), axis=2)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def masked_mse(original: Image.Image, edited: Image.Image, mask: np.ndarray) -> float:
    diff = (to_float(original) - to_float(edited)) ** 2
    return masked_mean(np.mean(diff, axis=2), mask)


def masked_psnr(original: Image.Image, edited: Image.Image, mask: np.ndarray) -> float:
    mse = masked_mse(original, edited, mask)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def simple_ssim(original: Image.Image, edited: Image.Image, mask: np.ndarray) -> float:
    """A compact masked SSIM approximation over selected pixels."""

    x = to_float(original)[mask]
    y = to_float(edited)[mask]
    if len(x) < 2:
        return 1.0
    c1, c2 = 0.01**2, 0.03**2
    mux, muy = x.mean(axis=0), y.mean(axis=0)
    vx, vy = x.var(axis=0), y.var(axis=0)
    cov = ((x - mux) * (y - muy)).mean(axis=0)
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))
    return float(np.clip(score.mean(), -1.0, 1.0))


def evaluate(original: Image.Image, edited: Image.Image, target_mask: Image.Image) -> dict[str, float]:
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    if target_mask.size != original.size:
        target_mask = target_mask.resize(original.size, Image.Resampling.NEAREST)
    target = np.array(target_mask.convert("L")) > 127
    background = ~target
    change = pixel_change(original, edited)
    inside = masked_mean(change, target)
    outside = masked_mean(change, background)
    return {
        "change_inside": inside,
        "change_outside": outside,
        "locality_score": inside / (inside + outside + 1e-8),
        "background_mse": masked_mse(original, edited, background),
        "background_psnr": masked_psnr(original, edited, background),
        "background_ssim": simple_ssim(original, edited, background),
    }
