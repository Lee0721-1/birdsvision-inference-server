# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import importlib

import pytest

from birdsvision_server import config


def test_public_defaults_are_local_and_use_example_labels():
    assert config.HOST == "127.0.0.1"
    assert config.NUM_CLASSES == 3
    assert config.LABELS_PATH.name == "labels.example.json"


def test_invalid_integer_is_rejected(monkeypatch):
    monkeypatch.setenv("BIRDSVISION_PORT", "not-an-integer")
    with pytest.raises(ValueError, match="BIRDSVISION_PORT"):
        importlib.reload(config)
    monkeypatch.delenv("BIRDSVISION_PORT")
    importlib.reload(config)


def test_source_identity_requires_url_and_full_commit(monkeypatch):
    monkeypatch.setenv("BIRDSVISION_SOURCE_REPOSITORY_URL", "https://github.com/example/server")
    with pytest.raises(ValueError, match="configured together"):
        importlib.reload(config)
    monkeypatch.setenv("BIRDSVISION_SOURCE_COMMIT", "short")
    with pytest.raises(ValueError, match="full Git commit"):
        importlib.reload(config)
    monkeypatch.delenv("BIRDSVISION_SOURCE_REPOSITORY_URL")
    monkeypatch.delenv("BIRDSVISION_SOURCE_COMMIT")
    importlib.reload(config)
