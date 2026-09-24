# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
from fastapi.testclient import TestClient

import app as server_app
import inference


def test_health_and_identify_contract_with_fake_backend(monkeypatch):
    monkeypatch.setattr(inference, "init_model", lambda: None)
    monkeypatch.setattr(inference, "is_ready", lambda: True)
    monkeypatch.setattr(
        inference,
        "predict",
        lambda _content, top_k: ([{
            "rank": 1, "class_id": 0, "class_key": "example:bird-0",
            "chinese_name": "示例鸟类零", "english_name": "Example bird zero",
            "folk_name": "", "confidence": 1.0,
        }], 1),
    )
    monkeypatch.setattr(server_app.config, "AUTH_REQUIRED", False)
    with TestClient(server_app.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        response = client.post(
            "/api/identify?top_k=1",
            files={"file": ("bird.png", b"synthetic", "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["results"][0]["class_key"] == "example:bird-0"


def test_source_revision_requires_published_commit(client, monkeypatch):
    monkeypatch.setattr(server_app.config, "SOURCE_REPOSITORY_URL", "")
    monkeypatch.setattr(server_app.config, "SOURCE_COMMIT", "")
    assert client.get("/api/source").status_code == 503
    monkeypatch.setattr(server_app.config, "SOURCE_REPOSITORY_URL", "https://github.com/example/server")
    monkeypatch.setattr(server_app.config, "SOURCE_COMMIT", "a" * 40)
    response = client.get("/api/source")
    assert response.status_code == 200
    assert response.json()["source_url"] == "https://github.com/example/server/tree/" + "a" * 40
