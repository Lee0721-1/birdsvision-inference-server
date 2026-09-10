# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only

import hashlib
import hmac
import secrets
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import app as app_module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.inference, "init_model", lambda: None)
    monkeypatch.setattr(app_module.inference, "is_ready", lambda: True)
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def auth_session(client):
    challenge = client.get("/api/auth/challenge").json()
    response = client.post(
        "/api/auth/token",
        json={
            "challenge_id": challenge["challenge_id"],
            "challenge": challenge["challenge"],
        },
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture
def auth_headers(auth_session):
    signing_key = bytes.fromhex(auth_session["signing_key"])

    def build(image_bytes, top_k=3, nonce=None, timestamp=None):
        request_timestamp = int(time.time()) if timestamp is None else timestamp
        request_nonce = nonce or secrets.token_urlsafe(18)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        canonical = app_module.api_auth.AuthManager.canonical_identify_request(
            request_timestamp, request_nonce, image_sha256, top_k
        )
        signature = hmac.new(signing_key, canonical, hashlib.sha256).hexdigest()
        return {
            "Authorization": f"Bearer {auth_session['access_token']}",
            "X-BirdsVision-Timestamp": str(request_timestamp),
            "X-BirdsVision-Nonce": request_nonce,
            "X-BirdsVision-Content-SHA256": image_sha256,
            "X-BirdsVision-Signature": signature,
        }

    return build
