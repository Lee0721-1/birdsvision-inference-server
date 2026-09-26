# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
from birdsvision_server.api import app as app_module


def test_unversioned_and_101_requests_keep_legacy_response(client, auth_headers, monkeypatch):
    def old_predict(data, top_k):
        return [{"rank": 1, "class_id": 1, "scientific_names": ["Aves"],
                 "confidence": 0.9}], 1

    monkeypatch.setattr(app_module.inference, "predict", old_predict)
    for version in (None, "1.0.1"):
        headers = auth_headers(b"bird-image")
        if version:
            headers["X-BirdsVision-App-Version"] = version
        response = client.post(
            "/api/identify?top_k=3",
            files={"file": ("bird.jpg", b"bird-image", "image/jpeg")},
            headers=headers,
        )
        assert response.status_code == 200
        assert "scientific_names" not in response.json()["results"][0]


def test_102_uses_modern_route_and_manual_box(client, auth_headers, monkeypatch):
    received = []

    def modern_predict(data, top_k, manual_box=None):
        received.append((data, top_k, manual_box))
        return [{"rank": 1, "class_id": 1224,
                 "scientific_names": ["Aves example"], "confidence": 0.8}], 2

    monkeypatch.setattr(app_module.modern_inference, "is_ready", lambda: True)
    monkeypatch.setattr(app_module.modern_inference, "predict", modern_predict)
    headers = auth_headers(b"bird-image", app_version="1.0.2", bird_box="0.1,0.2,0.8,0.9")
    headers["X-BirdsVision-App-Version"] = "1.0.2"
    response = client.post(
        "/api/identify?top_k=3&bird_box=0.1,0.2,0.8,0.9",
        files={"file": ("bird.jpg", b"bird-image", "image/jpeg")},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["scientific_names"] == ["Aves example"]
    assert received == [(b"bird-image", 3, (0.1, 0.2, 0.8, 0.9))]


def test_102_does_not_fall_back_when_modern_model_missing(client, auth_headers, monkeypatch):
    monkeypatch.setattr(app_module.modern_inference, "is_ready", lambda: False)
    headers = auth_headers(b"bird-image", app_version="1.0.2")
    headers["X-BirdsVision-App-Version"] = "1.0.2"
    response = client.post(
        "/api/identify?top_k=3",
        files={"file": ("bird.jpg", b"bird-image", "image/jpeg")},
        headers=headers,
    )
    assert response.status_code == 503
    assert response.json()["error_code"] == "MODEL_NOT_READY"


def test_102_manual_box_is_bound_to_request_signature(client, auth_headers, monkeypatch):
    monkeypatch.setattr(app_module.modern_inference, "is_ready", lambda: True)
    headers = auth_headers(b"bird-image", app_version="1.0.2", bird_box="0.1,0.2,0.8,0.9")
    headers["X-BirdsVision-App-Version"] = "1.0.2"
    response = client.post(
        "/api/identify?top_k=3&bird_box=0.1,0.2,0.7,0.9",
        files={"file": ("bird.jpg", b"bird-image", "image/jpeg")},
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json()["error_code"] == "INVALID_AUTH"


def test_102_version_is_bound_to_request_signature(client, auth_headers, monkeypatch):
    monkeypatch.setattr(app_module.modern_inference, "is_ready", lambda: True)
    headers = auth_headers(b"bird-image", app_version="1.0.2")
    headers["X-BirdsVision-App-Version"] = "1.0.3"
    response = client.post(
        "/api/identify?top_k=3",
        files={"file": ("bird.jpg", b"bird-image", "image/jpeg")},
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json()["error_code"] == "INVALID_AUTH"
