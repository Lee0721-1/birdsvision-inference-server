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


def classify_parent_v2(
    classifier,
    image: Image.Image,
    ordered_boxes: Sequence[Sequence[float]],
    transform,
    device: torch.device,
    *,
    max_views: int,
    crop_expansion: float,
    full_weight: float,
    many_box_threshold: int,
    many_box_full_weight: float,
    appended_class_boundary: int,
    appended_logit_offset: float,
) -> tuple[torch.Tensor, int, int]:
    """Frozen v2 decision for one parent; the caller supplies all release settings."""
    if type(max_views) is not int or max_views < 1:
        raise ValueError("max_views must be a positive integer")
    if type(many_box_threshold) is not int or many_box_threshold < 1:
        raise ValueError("many_box_threshold must be a positive integer")
    if (type(appended_logit_offset) not in (int, float)
            or not math.isfinite(appended_logit_offset)):
        raise ValueError("appended_logit_offset must be finite")
    for name, value in (("full_weight", full_weight),
                        ("many_box_full_weight", many_box_full_weight)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite in [0, 1]")
    batch, crop_count = build_view_batch(
        image, ordered_boxes, transform,
        max_views=max_views, crop_expansion=crop_expansion,
    )
    truncated_count = max(0, len(ordered_boxes) - crop_count)
    with torch.no_grad():
        logits = classifier(batch.to(device))
        if (logits.ndim != 2 or logits.shape[0] != crop_count + 1
                or type(appended_class_boundary) is not int
                or not 0 < appended_class_boundary < logits.shape[1]):
            raise ValueError("classifier output and release class partition disagree")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("classifier returned non-finite logits")
        if crop_count == 0:
            fused = logits[0]
        else:
            alpha = (many_box_full_weight if crop_count > many_box_threshold
                     else full_weight)
            fused = alpha * logits[0] + (1 - alpha) * logits[1:].mean(dim=0)
        decision = fused.clone()
        decision[appended_class_boundary:] += appended_logit_offset
        probabilities = torch.softmax(decision, dim=0)
    return probabilities, crop_count, truncated_count


def locate_and_classify_parent_v2(
    locator,
    classifier,
    image: Image.Image,
    transform,
    device: torch.device,
    **decision_settings,
) -> tuple[torch.Tensor, int, int]:
    """In-memory orchestration; locator errors propagate instead of becoming zero boxes."""
    ordered_boxes = locator(image)
    if ordered_boxes is None:
        raise ValueError("locator returned no result")
    return classify_parent_v2(
        classifier, image, ordered_boxes, transform, device, **decision_settings,
    )
