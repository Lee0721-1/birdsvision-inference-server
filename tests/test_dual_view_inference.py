# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import torch
from PIL import Image

import dual_view_inference as dual


def transform(image):
    return torch.full((3, 2, 2), float(image.size[0] * image.size[1]))


def test_one_shared_forward_contains_full_and_all_selected_crops():
    calls = []

    def classifier(batch):
        calls.append(batch.clone())
        return torch.tensor([[4.0, 0.0], [0.0, 2.0], [0.0, 4.0]])

    probabilities, crop_count = dual.classify_parent(
        classifier,
        Image.new("RGB", (100, 80)),
        [(10, 10, 30, 30), (50, 20, 90, 60)],
        transform,
        torch.device("cpu"),
        max_views=8,
    )
    assert len(calls) == 1
    assert calls[0].shape == (3, 3, 2, 2)
    assert crop_count == 2
    expected_logits = 0.325 * torch.tensor([4.0, 0.0]) + 0.675 * torch.tensor([0.0, 3.0])
    assert torch.allclose(probabilities, torch.softmax(expected_logits, dim=0))


def test_zero_box_is_exact_full_only():
    full_logits = torch.tensor([[1.0, 3.0]])
    fused = dual.fuse_parent_logits(full_logits, 0)
    assert fused.data_ptr() == full_logits[0].data_ptr()


def test_more_than_two_crops_uses_many_box_alpha_and_equal_mean():
    logits = torch.tensor([
        [10.0, 0.0],
        [0.0, 3.0],
        [0.0, 6.0],
        [0.0, 9.0],
    ])
    assert torch.allclose(
        dual.fuse_parent_logits(logits, 3),
        torch.tensor([1.0, 5.4]),
    )


def test_view_limit_is_deterministic_and_includes_full_view():
    batch, crop_count = dual.build_view_batch(
        Image.new("RGB", (100, 80)),
        [(0, 0, 10, 10), (10, 10, 30, 30), (30, 30, 60, 60)],
        transform,
        max_views=3,
    )
    assert batch.shape[0] == 3
    assert crop_count == 2


def test_invalid_or_non_overlapping_box_is_rejected():
    for box in ((20, 20, 10, 30), (200, 200, 220, 220)):
        try:
            dual.build_view_batch(Image.new("RGB", (100, 80)), [box], transform, max_views=2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid box was accepted")
