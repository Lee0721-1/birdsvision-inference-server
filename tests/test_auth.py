# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import time

import app as app_module


def assert_error(response, status_code, error_code):
    assert response.status_code == status_code
    body = response.json()
    assert body["success"] is False
    assert body["error_code"] == error_code


def test_challenge_can_only_be_exchanged_once(client):
    response = client.get("/api/auth/challenge")
    assert response.status_code == 200
    challenge = response.json()
    payload = {
        "challenge_id": challenge["challenge_id"],
        "challenge": challenge["challenge"],
    }

    first = client.post("/api/auth/token", json=payload)
    second = client.post("/api/auth/token", json=payload)

    assert first.status_code == 200
    assert first.json()["token_type"] == "Bearer"
    assert first.json()["expires_in"] == app_module.config.AUTH_TOKEN_TTL_SECONDS
    assert len(first.json()["signing_key"]) == 64
    assert_error(second, 401, "INVALID_CHALLENGE")


def test_identify_rejects_missing_authentication(client):
    response = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"image-bytes", "image/jpeg")},
    )

    assert_error(response, 401, "AUTH_REQUIRED")


def test_identify_rejects_replayed_nonce(client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        app_module.inference,
        "predict",
        lambda image_bytes, top_k: ([], 1),
    )
    headers = auth_headers(b"image-bytes", nonce="fixed_nonce_123456789")

    first = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"image-bytes", "image/jpeg")},
        headers=headers,
    )
    second = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"image-bytes", "image/jpeg")},
        headers=headers,
    )

    assert first.status_code == 200
    assert_error(second, 409, "REPLAY_DETECTED")


def test_identify_rejects_expired_timestamp(client, auth_headers):
    response = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"image-bytes", "image/jpeg")},
        headers=auth_headers(
            b"image-bytes",
            timestamp=int(time.time())
            - app_module.config.AUTH_TIMESTAMP_SKEW_SECONDS
            - 1,
        ),
    )

    assert_error(response, 401, "INVALID_AUTH")


def test_identify_rejects_tampered_image(client, auth_headers):
    response = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"tampered", "image/jpeg")},
        headers=auth_headers(b"original"),
    )

    assert_error(response, 401, "INVALID_AUTH")


def test_identify_applies_token_rate_limit(
    client,
    auth_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module.inference,
        "predict",
        lambda image_bytes, top_k: ([], 1),
    )
    client.app.state.auth_manager.token_limiter.limit = 1

    first = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"first", "image/jpeg")},
        headers=auth_headers(b"first"),
    )
    second = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"second", "image/jpeg")},
        headers=auth_headers(b"second"),
    )

    assert first.status_code == 200
    assert_error(second, 429, "RATE_LIMITED")


def test_identify_applies_ip_rate_limit(
    client,
    auth_headers,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module.inference,
        "predict",
        lambda image_bytes, top_k: ([], 1),
    )
    client.app.state.auth_manager.ip_limiter.limit = 1

    first = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"first", "image/jpeg")},
        headers=auth_headers(b"first"),
    )
    second = client.post(
        "/api/identify",
        files={"file": ("bird.jpg", b"second", "image/jpeg")},
        headers=auth_headers(b"second"),
    )

    assert first.status_code == 200
    assert_error(second, 429, "RATE_LIMITED")
