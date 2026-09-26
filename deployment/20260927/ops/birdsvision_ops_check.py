#!/usr/bin/env python3
"""BirdsVision 本机运行状态检查；输出 JSON，并为 systemd 返回明确状态。"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


APP_LOG_PATTERN = re.compile(
    r"identify request_id=\S+ status=(?P<status>\d+) \S+ "
    r"image_bytes=\d+ queue_ms=(?P<queue_ms>\d+) "
    r"inference_ms=\d+ total_ms=(?P<total_ms>\d+)"
)
NGINX_IDENTIFY_LOCATION_PATTERN = re.compile(
    r"location = /api/identify\s*\{(?P<body>.*?)\n\s*\}",
    re.DOTALL,
)
NGINX_ERROR_TIME_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
)
NGINX_TEMP_FILE_WARNING = "a client request body is buffered to a temporary file"
SOYOL_SPLIT_CONFIG = Path(
    "/etc/systemd/system/birdsvision.service.d/20-soyol-split.conf"
)


def read_int_env(name: str, default: int, minimum: int = 0) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return value


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return ordered[index]


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def systemd_state(unit: str, action: str = "is-active") -> str:
    completed = subprocess.run(
        ["systemctl", action, unit],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return (completed.stdout or completed.stderr).strip()


def systemd_properties(unit: str, names: Iterable[str]) -> dict[str, str]:
    command = ["systemctl", "show", unit]
    for name in names:
        command.extend(["-p", name])
    values: dict[str, str] = {}
    for line in run_command(command).splitlines():
        name, separator, value = line.partition("=")
        if separator:
            values[name] = value
    return values


def fetch_health(url: str, timeout_seconds: int) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BirdsVision-Ops-Check/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise RuntimeError(f"health_http_status={response.status}")
        return json.loads(response.read().decode("utf-8"))


def read_memory_metrics() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            key, raw_value = line.split(":", 1)
            value_parts = raw_value.strip().split()
            if value_parts:
                values[key] = int(value_parts[0])

    return {
        "available_mb": values.get("MemAvailable", 0) // 1024,
        "swap_total_mb": values.get("SwapTotal", 0) // 1024,
        "swap_used_mb": (
            values.get("SwapTotal", 0) - values.get("SwapFree", 0)
        )
        // 1024,
    }


def parse_proc_stat_cpu_times(line: str) -> tuple[int, int]:
    parts = line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        raise ValueError("proc_stat_cpu_line_invalid")
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError("proc_stat_cpu_line_invalid") from exc
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


def calculate_cpu_usage_percent(
    before: tuple[int, int],
    after: tuple[int, int],
) -> float:
    total_delta = after[0] - before[0]
    idle_delta = after[1] - before[1]
    if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
        raise ValueError("proc_stat_cpu_delta_invalid")
    return round((total_delta - idle_delta) * 100 / total_delta, 2)


def parse_loadavg_metrics(
    text: str,
    logical_cpu_count: int,
) -> dict[str, float | int]:
    if logical_cpu_count < 1:
        raise ValueError("logical_cpu_count_invalid")
    parts = text.split()
    if len(parts) < 3:
        raise ValueError("proc_loadavg_invalid")
    try:
        load_1m, load_5m, load_15m = (float(value) for value in parts[:3])
    except ValueError as exc:
        raise ValueError("proc_loadavg_invalid") from exc
    return {
        "logical_cpu_count": logical_cpu_count,
        "load_1m": load_1m,
        "load_5m": load_5m,
        "load_15m": load_15m,
        "load_1m_percent": round(load_1m * 100 / logical_cpu_count, 2),
        "load_5m_percent": round(load_5m * 100 / logical_cpu_count, 2),
        "load_15m_percent": round(load_15m * 100 / logical_cpu_count, 2),
    }


def read_cpu_metrics(sample_seconds: float = 0.25) -> dict[str, float | int]:
    with open("/proc/stat", encoding="utf-8") as proc_stat:
        before = parse_proc_stat_cpu_times(proc_stat.readline())
    time.sleep(sample_seconds)
    with open("/proc/stat", encoding="utf-8") as proc_stat:
        after = parse_proc_stat_cpu_times(proc_stat.readline())
    with open("/proc/loadavg", encoding="utf-8") as loadavg:
        load_metrics = parse_loadavg_metrics(
            loadavg.read(),
            os.cpu_count() or 1,
        )
    return {
        "sample_usage_percent": calculate_cpu_usage_percent(before, after),
        **load_metrics,
    }


def read_certificate_expiry(cert_path: str) -> datetime:
    output = run_command(
        ["openssl", "x509", "-enddate", "-noout", "-in", cert_path]
    )
    prefix = "notAfter="
    if not output.startswith(prefix):
        raise RuntimeError("certificate_expiry_unreadable")
    parsed = datetime.strptime(output[len(prefix):], "%b %d %H:%M:%S %Y %Z")
    return parsed.replace(tzinfo=timezone.utc)


def tail_lines(path: Path, max_bytes: int = 4 * 1024 * 1024) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - max_bytes), os.SEEK_SET)
        data = source.read()
    if size > max_bytes:
        data = data.split(b"\n", 1)[-1]
    return data.decode("utf-8", errors="replace").splitlines()


def parse_nginx_records(
    lines: Iterable[str],
    since: datetime,
) -> dict[str, int | float | None]:
    statuses: list[int] = []
    request_times: list[float] = []

    for line in lines:
        try:
            record = json.loads(line)
            recorded_at = datetime.fromisoformat(record["time"])
            if recorded_at < since:
                continue
            statuses.append(int(record["status"]))
            request_times.append(float(record["request_time"]) * 1000)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    return {
        "requests": len(statuses),
        "http_429": sum(status == 429 for status in statuses),
        "http_5xx": sum(status >= 500 for status in statuses),
        "request_p95_ms": percentile(request_times, 0.95),
    }


def parse_app_records(lines: Iterable[str]) -> dict[str, int | float | None]:
    statuses: list[int] = []
    queue_times: list[float] = []
    total_times: list[float] = []

    for line in lines:
        match = APP_LOG_PATTERN.search(line)
        if match is None:
            continue
        statuses.append(int(match.group("status")))
        queue_times.append(float(match.group("queue_ms")))
        total_times.append(float(match.group("total_ms")))

    return {
        "requests": len(statuses),
        "http_429": sum(status == 429 for status in statuses),
        "http_5xx": sum(status >= 500 for status in statuses),
        "queue_p95_ms": percentile(queue_times, 0.95),
        "total_p95_ms": percentile(total_times, 0.95),
    }


def parse_nginx_memory_upload_config(config_text: str) -> dict[str, bool]:
    match = NGINX_IDENTIFY_LOCATION_PATTERN.search(config_text)
    location_body = match.group("body") if match is not None else ""
    return {
        "client_max_body_size_12m": "client_max_body_size 12m;" in config_text,
        "client_body_buffer_size_12m": (
            "client_body_buffer_size 12m;" in location_body
        ),
        "proxy_http_version_1_1": "proxy_http_version 1.1;" in location_body,
        "proxy_request_buffering_off": (
            "proxy_request_buffering off;" in location_body
        ),
    }


def count_recent_nginx_temp_file_warnings(
    lines: Iterable[str],
    since: datetime,
) -> int:
    local_timezone = datetime.now().astimezone().tzinfo
    warnings = 0
    for line in lines:
        if NGINX_TEMP_FILE_WARNING not in line:
            continue
        match = NGINX_ERROR_TIME_PATTERN.match(line)
        if match is None:
            continue
        recorded_at = datetime.strptime(
            match.group("timestamp"),
            "%Y/%m/%d %H:%M:%S",
        ).replace(tzinfo=local_timezone)
        if recorded_at >= since.astimezone(local_timezone):
            warnings += 1
    return warnings


def write_status(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as target:
        json.dump(report, target, ensure_ascii=False, separators=(",", ":"))
        target.write("\n")
        temp_path = Path(target.name)
    os.chmod(temp_path, 0o640)
    os.replace(temp_path, path)


def build_report() -> dict:
    now = datetime.now(timezone.utc)
    problems: list[str] = []
    metrics: dict[str, object] = {}

    expected_classes = read_int_env("BIRDSVISION_OPS_EXPECTED_CLASSES", 755, 1)
    timeout_seconds = read_int_env("BIRDSVISION_OPS_HTTP_TIMEOUT_SECONDS", 15, 1)
    lookback_minutes = read_int_env("BIRDSVISION_OPS_LOOKBACK_MINUTES", 10, 1)
    disk_min_free_mb = read_int_env("BIRDSVISION_OPS_DISK_MIN_FREE_MB", 5120, 1)
    memory_min_available_mb = read_int_env(
        "BIRDSVISION_OPS_MEMORY_MIN_AVAILABLE_MB", 512, 1
    )
    cpu_load_max_percent = read_int_env(
        "BIRDSVISION_OPS_CPU_LOAD_MAX_PERCENT", 90, 1
    )
    cert_warn_days = read_int_env("BIRDSVISION_OPS_CERT_WARN_DAYS", 30, 1)
    max_5xx = read_int_env("BIRDSVISION_OPS_MAX_5XX", 0, 0)

    soyol_split_active = SOYOL_SPLIT_CONFIG.is_file()
    services = {
        "birdsvision": systemd_state("birdsvision.service"),
        "nginx": systemd_state("nginx.service"),
        "certbot_timer": systemd_state("certbot.timer"),
        "certbot_timer_enabled": systemd_state("certbot.timer", "is-enabled"),
        "unattended_upgrades": systemd_state("unattended-upgrades.service"),
        "unattended_upgrades_enabled": systemd_state(
            "unattended-upgrades.service", "is-enabled"
        ),
    }
    if soyol_split_active:
        services["birdsvision_soyol_locator"] = systemd_state(
            "birdsvision-soyol-locator.service"
        )
    metrics["services"] = services
    for name, state in services.items():
        if state not in {"active", "enabled"}:
            problems.append(f"{name}={state or 'unknown'}")

    memory_only_services: dict[str, dict[str, str]] = {}
    try:
        units = ["birdsvision.service", "nginx.service"]
        if soyol_split_active:
            units.append("birdsvision-soyol-locator.service")
        for unit in units:
            properties = systemd_properties(
                unit,
                ("MemorySwapMax", "MemorySwapCurrent", "LimitCORE"),
            )
            memory_only_services[unit] = properties
            if properties.get("MemorySwapMax") != "0":
                problems.append(f"{unit}_swap_not_disabled")
            if properties.get("MemorySwapCurrent") != "0":
                problems.append(f"{unit}_swap_in_use")
            if properties.get("LimitCORE") != "0":
                problems.append(f"{unit}_core_dump_not_disabled")
        metrics["memory_only_services"] = memory_only_services
    except Exception as exc:
        metrics["memory_only_services_error"] = f"{type(exc).__name__}: {exc}"
        problems.append("memory_only_service_check_failed")

    nginx_site_config = Path(
        os.getenv(
            "BIRDSVISION_OPS_NGINX_SITE_CONFIG",
            "/etc/nginx/sites-available/birdsvision",
        ).strip()
    )
    try:
        nginx_memory_config = parse_nginx_memory_upload_config(
            nginx_site_config.read_text(encoding="utf-8")
        )
        metrics["nginx_memory_upload_config"] = nginx_memory_config
        if not all(nginx_memory_config.values()):
            problems.append("nginx_memory_upload_config_invalid")
    except Exception as exc:
        metrics["nginx_memory_upload_config_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        problems.append("nginx_memory_upload_config_check_failed")

    health_url = os.getenv(
        "BIRDSVISION_OPS_HEALTH_URL",
        "http://127.0.0.1:8000/api/health",
    ).strip()
    try:
        health = fetch_health(health_url, timeout_seconds)
        metrics["health"] = health
        if health.get("status") != "ok":
            problems.append("health_status_not_ok")
        if health.get("model_loaded") is not True:
            problems.append("health_model_not_loaded")
        if health.get("num_classes") != expected_classes:
            problems.append("health_num_classes_mismatch")
    except Exception as exc:
        metrics["health_error"] = f"{type(exc).__name__}: {exc}"
        problems.append("health_request_failed")

    if soyol_split_active:
        try:
            locator_health = fetch_health(
                "http://127.0.0.1:8001/health", timeout_seconds
            )
            metrics["soyol_locator_health"] = locator_health
            if locator_health.get("status") != "ok":
                problems.append("soyol_locator_health_status_not_ok")
            if locator_health.get("model_loaded") is not True:
                problems.append("soyol_locator_model_not_loaded")
        except Exception as exc:
            metrics["soyol_locator_health_error"] = f"{type(exc).__name__}: {exc}"
            problems.append("soyol_locator_health_request_failed")

    disk = shutil.disk_usage("/")
    disk_metrics = {
        "total_mb": disk.total // (1024 * 1024),
        "used_mb": disk.used // (1024 * 1024),
        "free_mb": disk.free // (1024 * 1024),
        "used_percent": round(disk.used * 100 / disk.total, 2),
    }
    metrics["disk"] = disk_metrics
    if disk_metrics["free_mb"] < disk_min_free_mb:
        problems.append("disk_free_below_threshold")

    memory_metrics = read_memory_metrics()
    metrics["memory"] = memory_metrics
    if memory_metrics["available_mb"] < memory_min_available_mb:
        problems.append("memory_available_below_threshold")

    try:
        cpu_metrics = read_cpu_metrics()
        metrics["cpu"] = cpu_metrics
        if cpu_metrics["load_5m_percent"] > cpu_load_max_percent:
            problems.append("cpu_sustained_load_above_threshold")
    except Exception as exc:
        metrics["cpu_error"] = f"{type(exc).__name__}: {exc}"
        problems.append("cpu_check_failed")

    cert_path = os.getenv(
        "BIRDSVISION_OPS_CERT_PATH",
        "/etc/letsencrypt/live/birdsvision.com.cn/fullchain.pem",
    ).strip()
    try:
        expiry = read_certificate_expiry(cert_path)
        cert_days = math.floor((expiry - now).total_seconds() / 86400)
        metrics["certificate"] = {
            "path": cert_path,
            "expires_at": expiry.isoformat(),
            "days_remaining": cert_days,
        }
        if cert_days < cert_warn_days:
            problems.append("certificate_near_expiry")
    except Exception as exc:
        metrics["certificate_error"] = f"{type(exc).__name__}: {exc}"
        problems.append("certificate_check_failed")

    since = now - timedelta(minutes=lookback_minutes)
    nginx_log = Path(
        os.getenv(
            "BIRDSVISION_OPS_NGINX_LOG",
            "/var/log/birdsvision/nginx-access.log",
        ).strip()
    )
    nginx_metrics = parse_nginx_records(tail_lines(nginx_log), since)
    metrics["nginx_window"] = nginx_metrics

    nginx_error_log = Path(
        os.getenv(
            "BIRDSVISION_OPS_NGINX_ERROR_LOG",
            "/var/log/birdsvision/nginx-error.log",
        ).strip()
    )
    nginx_temp_file_warnings = count_recent_nginx_temp_file_warnings(
        tail_lines(nginx_error_log),
        since,
    )
    metrics["nginx_temp_file_warnings"] = nginx_temp_file_warnings
    if nginx_temp_file_warnings > 0:
        problems.append("nginx_request_body_written_to_temp_file")

    try:
        journal_lines = run_command(
            [
                "journalctl",
                "-u",
                "birdsvision.service",
                f"--since=-{lookback_minutes}min",
                "--no-pager",
                "-o",
                "cat",
            ]
        ).splitlines()
        app_metrics = parse_app_records(journal_lines)
        metrics["app_window"] = app_metrics
    except Exception as exc:
        metrics["app_window_error"] = f"{type(exc).__name__}: {exc}"
        problems.append("app_journal_check_failed")

    observed_5xx = max(
        int(nginx_metrics["http_5xx"] or 0),
        int(metrics.get("app_window", {}).get("http_5xx", 0)),
    )
    if observed_5xx > max_5xx:
        problems.append("recent_http_5xx_above_threshold")

    return {
        "status": "ok" if not problems else "alert",
        "checked_at": now.isoformat(),
        "lookback_minutes": lookback_minutes,
        "metrics": metrics,
        "problems": problems,
    }


def main() -> int:
    status_path = Path(
        os.getenv(
            "BIRDSVISION_OPS_STATUS_PATH",
            "/var/lib/birdsvision/ops-status.json",
        ).strip()
    )
    try:
        report = build_report()
    except Exception as exc:
        report = {
            "status": "alert",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "problems": [f"ops_check_crashed:{type(exc).__name__}:{exc}"],
        }

    write_status(status_path, report)
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
