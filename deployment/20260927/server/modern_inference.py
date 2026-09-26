"""Versioned classifier path; SOYOL runs in a separate local process."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import timm
import torch

import inference
import locator_client
from dual_view_inference import classify_parent_v2


MODEL_ENV = "BIRDSVISION_1983_MODEL_PATH"
LABELS_ENV = "BIRDSVISION_1983_LABELS_PATH"
LOCATOR_URL_ENV = "BIRDSVISION_SOYOL_LOCATOR_URL"
_model = None
_transform = None
_device = None
_labels = None


def configured() -> bool:
    values = [os.getenv(name) for name in (MODEL_ENV, LABELS_ENV, LOCATOR_URL_ENV)]
    if any(values) and not all(values):
        raise ValueError("1.0.2 classifier paths and locator URL must be set together")
    if all(values):
        locator_client.validate_url(os.environ[LOCATOR_URL_ENV])
    return all(values)


def init_model() -> None:
    global _model, _transform, _device, _labels
    if not configured() or _model is not None:
        return
    labels_path = Path(os.environ[LABELS_ENV])
    loaded_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if len(loaded_labels) != 1983 or any(
        row.get("class_id") != index or not row.get("scientific_names")
        for index, row in enumerate(loaded_labels)
    ):
        raise ValueError("1983 class labels or scientific names are incomplete")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inference.StableCUBModel(num_classes=1983)
    state = torch.load(os.environ[MODEL_ENV], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False,
    )
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224, device=device))
    _model, _transform, _device = model, transform, device
    _labels = loaded_labels


def is_ready() -> bool:
    return _model is not None and bool(os.getenv(LOCATOR_URL_ENV))


def _manual_pixels(box: tuple[float, float, float, float], image) -> list[float]:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("INVALID_BIRD_BOX")
    left, top, right, bottom = box
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("INVALID_BIRD_BOX")
    return [left * image.width, top * image.height,
            right * image.width, bottom * image.height]


def predict(image_bytes: bytes, top_k: int = 3,
            manual_box: tuple[float, float, float, float] | None = None):
    if not is_ready():
        raise RuntimeError("MODEL_NOT_READY")
    image = inference.decode_image(image_bytes)
    started = time.time()
    boxes = ([_manual_pixels(manual_box, image)] if manual_box is not None
             else locator_client.locate(
                 os.environ[LOCATOR_URL_ENV], image_bytes, image.size,
             ))
    probabilities, _, _ = classify_parent_v2(
        _model, image, boxes, _transform, _device,
        max_views=11, crop_expansion=0.15,
        full_weight=0.3, many_box_threshold=3, many_box_full_weight=0.3,
        appended_class_boundary=1224,
        appended_logit_offset=3.1824216842651367,
    )
    values, indices = torch.topk(probabilities, max(1, min(int(top_k), 10)))
    results = []
    for rank, (confidence, class_id) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        label = _labels[class_id]
        results.append({
            "rank": rank, "class_id": class_id, "class_key": label["class_key"],
            "chinese_name": label["chinese_name"], "english_name": label["english_name"],
            "folk_name": label.get("folk_name", ""),
            "scientific_names": label["scientific_names"],
            "confidence": round(float(confidence), 4),
        })
    return results, int((time.time() - started) * 1000)
