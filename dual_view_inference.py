# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Pure dual-view inference operations; no locator or production side effects."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from PIL import Image


def _expanded_crop(image: Image.Image, box: Sequence[float], expansion: float) -> Image.Image:
    if len(box) != 4 or any(not math.isfinite(float(value)) for value in box):
        raise ValueError("each bird box must contain four finite xyxy coordinates")
    left, top, right, bottom = map(float, box)
    if not (0 <= expansion <= 1) or right <= left or bottom <= top:
        raise ValueError("invalid bird box or crop expansion")
    width, height = image.size
    box_width, box_height = right - left, bottom - top
    crop_left = max(0, math.floor(left - box_width * expansion))
    crop_top = max(0, math.floor(top - box_height * expansion))
    crop_right = min(width, math.ceil(right + box_width * expansion))
    crop_bottom = min(height, math.ceil(bottom + box_height * expansion))
    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError("bird box does not overlap the image")
    return image.crop((crop_left, crop_top, crop_right, crop_bottom))


def build_view_batch(
    image: Image.Image,
    ordered_boxes: Sequence[Sequence[float]],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    max_views: int,
    crop_expansion: float = 0.15,
) -> tuple[torch.Tensor, int]:
    """Return one full-image view followed by bounded locator-ordered crops."""
    if max_views < 1:
        raise ValueError("max_views must be at least one")
    crop_limit = max_views - 1
    selected = ordered_boxes[:crop_limit]
    views = [transform(image)]
    views.extend(transform(_expanded_crop(image, box, crop_expansion)) for box in selected)
    shape = tuple(views[0].shape)
    if any(tuple(view.shape) != shape for view in views):
        raise ValueError("all transformed views must have the same tensor shape")
    return torch.stack(views, dim=0), len(selected)


def fuse_parent_logits(
    flat_logits: torch.Tensor,
    crop_count: int,
    *,
    default_full_alpha: float = 0.325,
    many_box_threshold: int = 2,
    many_box_full_alpha: float = 0.1,
) -> torch.Tensor:
    if flat_logits.ndim != 2 or flat_logits.shape[0] != crop_count + 1:
        raise ValueError("flat logits and crop count disagree")
    if crop_count == 0:
        return flat_logits[0]
    alpha = many_box_full_alpha if crop_count > many_box_threshold else default_full_alpha
    if not 0 <= alpha <= 1:
        raise ValueError("fusion alpha must be between zero and one")
    crop_mean = flat_logits[1:].mean(dim=0)
    return alpha * flat_logits[0] + (1 - alpha) * crop_mean


def classify_parent(
    classifier,
    image: Image.Image,
    ordered_boxes: Sequence[Sequence[float]],
    transform,
    device: torch.device,
    *,
    max_views: int,
    crop_expansion: float = 0.15,
    default_full_alpha: float = 0.325,
    many_box_threshold: int = 2,
    many_box_full_alpha: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Run the shared classifier exactly once and return post-fusion probabilities."""
    batch, crop_count = build_view_batch(
        image, ordered_boxes, transform,
        max_views=max_views, crop_expansion=crop_expansion,
    )
    with torch.no_grad():
        flat_logits = classifier(batch.to(device))
        fused = fuse_parent_logits(
            flat_logits,
            crop_count,
            default_full_alpha=default_full_alpha,
            many_box_threshold=many_box_threshold,
            many_box_full_alpha=many_box_full_alpha,
        )
        probabilities = torch.softmax(fused, dim=0)
    return probabilities, crop_count
