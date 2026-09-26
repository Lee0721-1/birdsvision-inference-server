# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Check vendored Ultralytics package files against its installed wheel RECORD."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import zipfile
from pathlib import Path


def verify(vendor: Path, wheel: Path | None = None) -> dict:
    vendor = vendor.resolve(strict=True)
    metadata = vendor / "ultralytics-8.4.126.dist-info"
    if not metadata.is_dir():
        raise ValueError("Ultralytics 8.4.126 distribution metadata is missing")
    record = metadata / "RECORD"
    checked = set()
    with record.open(encoding="utf-8", newline="") as stream:
        for name, hash_field, size_field in csv.reader(stream):
            relative = Path(name)
            if (not relative.parts or relative.parts[0] != "ultralytics"
                    or "__pycache__" in relative.parts):
                continue
            path = (vendor / relative).resolve(strict=True)
            if vendor not in path.parents or not path.is_file():
                raise ValueError(f"invalid vendor path: {name}")
            algorithm, separator, encoded = hash_field.partition("=")
            if algorithm != "sha256" or not separator or not size_field:
                raise ValueError(f"missing package hash or size: {name}")
            actual = path.read_bytes()
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if len(actual) != int(size_field) or hashlib.sha256(actual).digest() != expected:
                raise ValueError(f"vendor file differs from installed RECORD: {name}")
            checked.add(relative.as_posix())
    package_files = {
        path.relative_to(vendor).as_posix()
        for path in (vendor / "ultralytics").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if len(checked) < 100 or package_files != checked:
        raise ValueError("vendored package file list differs from installed RECORD")
    result = {"status": "verified_against_installed_record",
              "distribution": "ultralytics==8.4.126", "package_files": len(checked)}
    if wheel is not None:
        wheel = wheel.resolve(strict=True)
        if wheel.name != "ultralytics-8.4.126-py3-none-any.whl":
            raise ValueError("unexpected Ultralytics wheel filename")
        with zipfile.ZipFile(wheel) as archive:
            wheel_files = {
                name for name in archive.namelist()
                if name.startswith("ultralytics/") and not name.endswith("/")
            }
            if wheel_files != checked:
                raise ValueError("wheel package file list differs from deployed vendor")
            for name in wheel_files:
                if archive.read(name) != (vendor / name).read_bytes():
                    raise ValueError(f"wheel file differs from deployed vendor: {name}")
        result["status"] = "verified_against_pypi_wheel_and_installed_record"
        result["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.vendor, args.wheel)))
