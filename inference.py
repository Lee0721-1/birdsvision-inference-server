# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Optional ConvNeXt-Tiny backend; no model weight is bundled."""

from __future__ import annotations

import io
import time
import warnings

import timm
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

import config
import labels as labels_mod


class StableClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        if config.MODEL_NAME != "convnext_tiny":
            raise ValueError("only convnext_tiny is supported by the reference backend")
        self.backbone = models.convnext_tiny(weights=None)
        features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(features, num_classes)

    def forward(self, inputs):
        return self.backbone(inputs)


_model = None
_transform = None
_device = None


def decode_image(image_bytes):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.verify()
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
                image.load()
                return image
    except Exception as exc:
        raise ValueError("INVALID_IMAGE") from exc


def init_model():
    global _model, _transform, _device
    if _model is not None:
        return
    if not config.MODEL_PATH.is_file():
        raise RuntimeError(
            "MODEL_NOT_CONFIGURED: set BIRDSVISION_MODEL_PATH to a compatible private weight"
        )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableClassifier(config.NUM_CLASSES)
    payload = torch.load(config.MODEL_PATH, map_location=_device, weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.to(_device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    _transform = timm.data.create_transform(**data_config, is_training=False)
    _model = model


def is_ready():
    return _model is not None


def predict(image_bytes, top_k=3):
    if not is_ready():
        raise RuntimeError("MODEL_NOT_READY")
    image = decode_image(image_bytes)
    started = time.perf_counter()
    batch = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        probabilities = _model(batch)[0].softmax(dim=0)
        count = max(1, min(int(top_k), 10, config.NUM_CLASSES))
        values, indices = probabilities.topk(count)
    results = []
    for rank, (confidence, class_id) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        label = labels_mod.get_label(class_id)
        results.append({
            "rank": rank, "class_id": class_id,
            "class_key": label["class_key"],
            "chinese_name": label["chinese_name"],
            "english_name": label["english_name"],
            "folk_name": label.get("folk_name", ""),
            "confidence": round(float(confidence), 4),
        })
    return results, int((time.perf_counter() - started) * 1000)
