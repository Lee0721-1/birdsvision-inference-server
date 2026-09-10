# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import importlib

import pytest

import config


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
