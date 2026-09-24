# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from client_version import legacy_results, route_for_version


@pytest.mark.parametrize("version,expected", [
    (None, "legacy"), ("1.0.1", "legacy"), ("1.0.2", "modern"),
    ("1.0.2.0", "modern"), ("1.1", "modern"), ("2.0", "modern"),
])
def test_route_for_version(version, expected):
    assert route_for_version(version) == expected


@pytest.mark.parametrize("version", ["", "1.0.2 beta", "v1.0.2", "1..2", "-1.0.2"])
def test_invalid_version_rejected(version):
    with pytest.raises(ValueError, match="INVALID_APP_VERSION"):
        route_for_version(version)


def test_legacy_shape_excludes_scientific_names_without_mutating_results():
    results = [{"class_id": 4, "scientific_names": ["Aves example"], "rank": 1}]
    assert legacy_results(results) == [{"class_id": 4, "rank": 1}]
    assert "scientific_names" in results[0]
