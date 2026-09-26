# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import io

import httpx
import pytest
from PIL import Image
import torch

from birdsvision_server.recognition import locator_client
from birdsvision_server.recognition import modern_inference


def sample_image():
    output = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(output, format="PNG")
    return output.getvalue()


def test_classifier_accepts_only_loopback_locator_and_validated_boxes(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/locate"
        assert request.content == sample_image()
        return httpx.Response(200, json={"width": 40, "height": 20,
                                         "boxes": [[2.0, 3.0, 30.0, 18.0]]})

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(locator_client.httpx, "Client",
                        lambda **kwargs: original_client(transport=transport, **kwargs))
    assert locator_client.locate("http://127.0.0.1:8001/v1/locate",
                                 sample_image(), (40, 20)) == [[2.0, 3.0, 30.0, 18.0]]
    with pytest.raises(ValueError):
        locator_client.validate_url("http://example.com/v1/locate")


@pytest.mark.parametrize("payload", [
    {"width": 41, "height": 20, "boxes": []},
    {"width": 40, "height": 20, "boxes": [[2, 3, 41, 18]]},
    {"width": 40, "height": 20, "boxes": [[2, 3, "NaN", 18]]},
])
def test_classifier_fails_closed_on_bad_locator_response(monkeypatch, payload):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    original_client = httpx.Client
    monkeypatch.setattr(locator_client.httpx, "Client",
                        lambda **kwargs: original_client(transport=transport, **kwargs))
    with pytest.raises(RuntimeError, match="SOYOL_LOCATOR_INVALID_RESPONSE"):
        locator_client.locate("http://127.0.0.1:8001/v1/locate", b"image", (40, 20))


def test_manual_box_bypasses_locator_process(monkeypatch):
    captured = []

    def classify(_model, image, boxes, _transform, _device, **kwargs):
        captured.append((image.size, boxes))
        return torch.tensor([0.9, 0.1]), 1, 0

    monkeypatch.setattr(modern_inference, "_model", object())
    monkeypatch.setattr(modern_inference, "_labels", [
        {"class_key": "first", "chinese_name": "一", "english_name": "first",
         "scientific_names": ["Example first"]},
        {"class_key": "second", "chinese_name": "二", "english_name": "second",
         "scientific_names": ["Example second"]},
    ])
    monkeypatch.setenv(modern_inference.LOCATOR_URL_ENV,
                       "http://127.0.0.1:8001/v1/locate")
    monkeypatch.setattr(modern_inference, "classify_parent_v2", classify)
    monkeypatch.setattr(locator_client, "locate",
                        lambda *_: (_ for _ in ()).throw(AssertionError("locator called")))
    results, _ = modern_inference.predict(sample_image(), top_k=1,
                                          manual_box=(0.1, 0.2, 0.8, 0.9))
    assert captured == [((40, 20), [[4.0, 4.0, 32.0, 18.0]])]
    assert results[0]["class_key"] == "first"


def test_automatic_route_uses_locator_boxes(monkeypatch):
    captured = []

    def classify(_model, image, boxes, _transform, _device, **kwargs):
        captured.append((image.size, boxes))
        return torch.tensor([0.9, 0.1]), 1, 0

    monkeypatch.setattr(modern_inference, "_model", object())
    monkeypatch.setattr(modern_inference, "_labels", [
        {"class_key": "first", "chinese_name": "一", "english_name": "first",
         "scientific_names": ["Example first"]},
        {"class_key": "second", "chinese_name": "二", "english_name": "second",
         "scientific_names": ["Example second"]},
    ])
    monkeypatch.setenv(modern_inference.LOCATOR_URL_ENV,
                       "http://127.0.0.1:8001/v1/locate")
    monkeypatch.setattr(modern_inference, "classify_parent_v2", classify)

    def fake_locate(url, data, size):
        assert url == "http://127.0.0.1:8001/v1/locate"
        assert data == sample_image()
        assert size == (40, 20)
        return [[2.0, 3.0, 30.0, 18.0]]

    monkeypatch.setattr(locator_client, "locate", fake_locate)
    modern_inference.predict(sample_image(), top_k=1)
    assert captured == [((40, 20), [[2.0, 3.0, 30.0, 18.0]])]
