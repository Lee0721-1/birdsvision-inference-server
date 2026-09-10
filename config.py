# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Environment-only public server configuration."""

from __future__ import annotations

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent


def read_int(name, default, minimum, maximum):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def read_bool(name, default):
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def read_path(name, default):
    path = Path(os.path.expanduser(os.getenv(name, default)))
    return (SERVER_DIR / path).resolve() if not path.is_absolute() else path.resolve()


MODEL_PATH = read_path("BIRDSVISION_MODEL_PATH", "private/model.pth")
LABELS_PATH = read_path("BIRDSVISION_LABELS_PATH", "examples/labels.example.json")
NUM_CLASSES = read_int("BIRDSVISION_NUM_CLASSES", 3, 1, 100000)
MODEL_NAME = os.getenv("BIRDSVISION_MODEL_NAME", "convnext_tiny").strip()
MAX_IMAGE_MB = read_int("BIRDSVISION_MAX_IMAGE_MB", 10, 1, 1024)
INFERENCE_CONCURRENCY = read_int("BIRDSVISION_INFERENCE_CONCURRENCY", 1, 1, 16)
INFERENCE_QUEUE_CAPACITY = read_int("BIRDSVISION_INFERENCE_QUEUE_CAPACITY", 4, 0, 100)
AUTH_REQUIRED = read_bool("BIRDSVISION_AUTH_REQUIRED", True)
AUTH_CHALLENGE_TTL_SECONDS = read_int("BIRDSVISION_AUTH_CHALLENGE_TTL_SECONDS", 60, 10, 300)
AUTH_TOKEN_TTL_SECONDS = read_int("BIRDSVISION_AUTH_TOKEN_TTL_SECONDS", 300, 60, 3600)
AUTH_TIMESTAMP_SKEW_SECONDS = read_int("BIRDSVISION_AUTH_TIMESTAMP_SKEW_SECONDS", 30, 5, 300)
AUTH_CHALLENGES_PER_MINUTE = read_int("BIRDSVISION_AUTH_CHALLENGES_PER_MINUTE", 10, 1, 600)
AUTH_IDENTIFY_PER_IP_PER_MINUTE = read_int("BIRDSVISION_AUTH_IDENTIFY_PER_IP_PER_MINUTE", 30, 1, 6000)
AUTH_IDENTIFY_PER_TOKEN_PER_MINUTE = read_int("BIRDSVISION_AUTH_IDENTIFY_PER_TOKEN_PER_MINUTE", 12, 1, 6000)
DEFAULT_TOP_K = read_int("BIRDSVISION_DEFAULT_TOP_K", 3, 1, 10)
APP_LATEST_VERSION = os.getenv("BIRDSVISION_APP_LATEST_VERSION", "0.0.0").strip()
APP_MINIMUM_SUPPORTED_VERSION = os.getenv("BIRDSVISION_APP_MINIMUM_SUPPORTED_VERSION", "0.0.0").strip()
APP_UPDATE_URL = os.getenv("BIRDSVISION_APP_UPDATE_URL", "https://example.invalid/").strip()
HOST = os.getenv("BIRDSVISION_HOST", "127.0.0.1").strip()
PORT = read_int("BIRDSVISION_PORT", 8000, 1, 65535)
