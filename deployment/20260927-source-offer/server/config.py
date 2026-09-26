import os
import re
from urllib.parse import urlsplit

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SERVER_DIR, ".."))

_APP_VERSION_PATTERN = re.compile(r"\d+(?:\.\d+)*\Z")
_MAX_APP_VERSION_LENGTH = 64
_MAX_UPDATE_URL_LENGTH = 2048
_OFFICIAL_UPDATE_ORIGIN = "https://birdsvision.com.cn"


def _read_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return value


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} 必须是 true/false、1/0、yes/no 或 on/off"
    )


def _read_path_env(name: str, default: str) -> str:
    value = os.path.expanduser(os.getenv(name, default))
    if not os.path.isabs(value):
        value = os.path.join(SERVER_DIR, value)
    return os.path.abspath(value)


def _read_app_version_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if (
        not value
        or len(value) > _MAX_APP_VERSION_LENGTH
        or _APP_VERSION_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{name} 必须是点分隔的纯数字版本号")
    return value


def _compare_app_versions(left: str, right: str) -> int:
    left_parts = left.split(".")
    right_parts = right.split(".")
    length = max(len(left_parts), len(right_parts))
    for index in range(length):
        left_value = left_parts[index] if index < len(left_parts) else "0"
        right_value = right_parts[index] if index < len(right_parts) else "0"
        left_part = left_value.lstrip("0") or "0"
        right_part = right_value.lstrip("0") or "0"
        if len(left_part) != len(right_part):
            return -1 if len(left_part) < len(right_part) else 1
        if left_part != right_part:
            return -1 if left_part < right_part else 1
    return 0


def _read_update_url_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if (
        not value
        or len(value) > _MAX_UPDATE_URL_LENGTH
        or not (
            value == _OFFICIAL_UPDATE_ORIGIN
            or value.startswith(f"{_OFFICIAL_UPDATE_ORIGIN}/")
        )
    ):
        raise ValueError(f"{name} 必须指向 BirdsVision 官方 HTTPS 网站")
    try:
        parsed = urlsplit(value)
        is_official = (
            parsed.scheme == "https"
            and parsed.hostname == "birdsvision.com.cn"
            and parsed.username is None
            and parsed.password is None
            and parsed.port is None
        )
    except ValueError as exc:
        raise ValueError(f"{name} 必须是有效的官网 HTTPS 地址") from exc
    if not is_official:
        raise ValueError(f"{name} 必须指向 BirdsVision 官方 HTTPS 网站")
    return value


# 后端自包含：部署时只需要整个 server/ 目录
MODEL_PATH = _read_path_env(
    "BIRDSVISION_MODEL_PATH",
    os.path.join(SERVER_DIR, "model", "ResNet_best.pth"),
)
LABELS_PATH = _read_path_env(
    "BIRDSVISION_LABELS_PATH",
    os.path.join(SERVER_DIR, "data", "labels.json"),
)

NUM_CLASSES = _read_int_env("BIRDSVISION_NUM_CLASSES", 755, 1, 100000)
MODEL_NAME = "convnext_tiny"
MAX_IMAGE_MB = _read_int_env("BIRDSVISION_MAX_IMAGE_MB", 10, 1, 1024)
INFERENCE_CONCURRENCY = _read_int_env(
    "BIRDSVISION_INFERENCE_CONCURRENCY",
    1,
    1,
    16,
)
INFERENCE_QUEUE_CAPACITY = _read_int_env(
    "BIRDSVISION_INFERENCE_QUEUE_CAPACITY",
    4,
    0,
    100,
)
AUTH_REQUIRED = _read_bool_env("BIRDSVISION_AUTH_REQUIRED", True)
AUTH_CHALLENGE_TTL_SECONDS = _read_int_env(
    "BIRDSVISION_AUTH_CHALLENGE_TTL_SECONDS",
    60,
    10,
    300,
)
AUTH_TOKEN_TTL_SECONDS = _read_int_env(
    "BIRDSVISION_AUTH_TOKEN_TTL_SECONDS",
    300,
    60,
    3600,
)
AUTH_TIMESTAMP_SKEW_SECONDS = _read_int_env(
    "BIRDSVISION_AUTH_TIMESTAMP_SKEW_SECONDS",
    30,
    5,
    300,
)
AUTH_CHALLENGES_PER_MINUTE = _read_int_env(
    "BIRDSVISION_AUTH_CHALLENGES_PER_MINUTE",
    10,
    1,
    600,
)
AUTH_IDENTIFY_PER_IP_PER_MINUTE = _read_int_env(
    "BIRDSVISION_AUTH_IDENTIFY_PER_IP_PER_MINUTE",
    30,
    1,
    6000,
)
AUTH_IDENTIFY_PER_TOKEN_PER_MINUTE = _read_int_env(
    "BIRDSVISION_AUTH_IDENTIFY_PER_TOKEN_PER_MINUTE",
    12,
    1,
    6000,
)
DEFAULT_TOP_K = 3
APP_LATEST_VERSION = _read_app_version_env(
    "BIRDSVISION_APP_LATEST_VERSION",
    "1.0.1",
)
APP_MINIMUM_SUPPORTED_VERSION = _read_app_version_env(
    "BIRDSVISION_APP_MINIMUM_SUPPORTED_VERSION",
    "1.0.1",
)
if _compare_app_versions(APP_MINIMUM_SUPPORTED_VERSION, APP_LATEST_VERSION) > 0:
    raise ValueError(
        "BIRDSVISION_APP_MINIMUM_SUPPORTED_VERSION 不得高于 "
        "BIRDSVISION_APP_LATEST_VERSION"
    )
APP_UPDATE_URL = _read_update_url_env(
    "BIRDSVISION_APP_UPDATE_URL",
    "https://birdsvision.com.cn/",
)
SOURCE_REPOSITORY_URL = os.getenv("BIRDSVISION_SOURCE_REPOSITORY_URL", "").strip()
SOURCE_COMMIT = os.getenv("BIRDSVISION_SOURCE_COMMIT", "").strip()
if bool(SOURCE_REPOSITORY_URL) != bool(SOURCE_COMMIT):
    raise ValueError("源码仓库地址与固定提交必须一起设置")
if SOURCE_REPOSITORY_URL:
    source_url = urlsplit(SOURCE_REPOSITORY_URL)
    if (source_url.scheme != "https" or not source_url.hostname
            or source_url.username or source_url.password
            or source_url.query or source_url.fragment
            or re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT) is None):
        raise ValueError("源码仓库必须是 HTTPS 地址并使用完整 Git 提交")
HOST = os.getenv("BIRDSVISION_HOST", "0.0.0.0").strip()
if HOST == "":
    raise ValueError("BIRDSVISION_HOST 不能为空")
PORT = _read_int_env("BIRDSVISION_PORT", 8000, 1, 65535)
