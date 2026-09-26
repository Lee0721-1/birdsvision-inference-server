# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded local HTTP contract for the separate SOYOL locator process."""

from __future__ import annotations

import math
from urllib.parse import urlsplit

import httpx


def validate_url(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "::1"}
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path != "/v1/locate" or parts.port is None):
        raise ValueError("BIRDSVISION_SOYOL_LOCATOR_URL must be a loopback /v1/locate URL")
    return url


def locate(url: str, image_bytes: bytes, image_size: tuple[int, int]) -> list[list[float]]:
    validate_url(url)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            response = client.post(url, content=image_bytes,
                                   headers={"content-type": "application/octet-stream"})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError("MODEL_NOT_READY") from exc
    if not isinstance(payload, dict) or set(payload) != {"width", "height", "boxes"}:
        raise RuntimeError("SOYOL_LOCATOR_INVALID_RESPONSE")
    width, height = image_size
    if (type(payload["width"]) is not int or type(payload["height"]) is not int
            or (payload["width"], payload["height"]) != (width, height)):
        raise RuntimeError("SOYOL_LOCATOR_INVALID_RESPONSE")
    boxes = payload["boxes"]
    if not isinstance(boxes, list) or len(boxes) > 10:
        raise RuntimeError("SOYOL_LOCATOR_INVALID_RESPONSE")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in box)
                or not (0 <= box[0] < box[2] <= width
                        and 0 <= box[1] < box[3] <= height)):
            raise RuntimeError("SOYOL_LOCATOR_INVALID_RESPONSE")
    return boxes
