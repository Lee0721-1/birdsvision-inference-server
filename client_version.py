# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Select the identification contract from the explicit app version header.

Unversioned requests are the deployed 1.0.1 contract.  Version parsing is
strict so a malformed request cannot accidentally enter the new model route.
"""

import re


APP_VERSION_HEADER = "x-birdsvision-app-version"
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*\Z")
_NEW_ROUTE = (1, 0, 2)


def route_for_version(raw: str | None) -> str:
    if raw is None:
        return "legacy"
    if len(raw) > 64 or _VERSION.fullmatch(raw) is None:
        raise ValueError("INVALID_APP_VERSION")
    parts = tuple(int(part) for part in raw.split("."))
    width = max(len(parts), len(_NEW_ROUTE))
    padded = parts + (0,) * (width - len(parts))
    threshold = _NEW_ROUTE + (0,) * (width - len(_NEW_ROUTE))
    return "modern" if padded >= threshold else "legacy"


def legacy_results(results: list[dict]) -> list[dict]:
    """Keep the 1.0.1 response shape even when labels contain new metadata."""
    return [{key: value for key, value in result.items() if key != "scientific_names"}
            for result in results]
