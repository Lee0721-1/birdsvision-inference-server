# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import io

import pytest
from PIL import Image

import inference


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return output.getvalue()


def test_decode_image_accepts_valid_pixels_and_rejects_invalid_bytes():
    assert inference.decode_image(image_bytes()).mode == "RGB"
    with pytest.raises(ValueError, match="INVALID_IMAGE"):
        inference.decode_image(b"not an image")


def test_predict_requires_initialized_private_model(monkeypatch):
    monkeypatch.setattr(inference, "_model", None)
    with pytest.raises(RuntimeError, match="MODEL_NOT_READY"):
        inference.predict(image_bytes())
