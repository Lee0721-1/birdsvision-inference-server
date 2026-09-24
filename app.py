# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import asyncio
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

import config
import inference
import modern_inference
from client_version import APP_VERSION_HEADER, legacy_results, route_for_version
import auth as api_auth


MULTIPART_OVERHEAD_ALLOWANCE_BYTES = 2 * 1024 * 1024
logger = logging.getLogger("uvicorn.error")


class InferenceQueueFull(RuntimeError):
    pass


class UploadTooLarge(ValueError):
    pass


class InvalidMultipartUpload(ValueError):
    pass


class AuthTokenRequest(BaseModel):
    challenge_id: str
    challenge: str


class InMemoryImageMultipartParser:
    """只在内存中提取 multipart 的唯一 file 文件字段。"""

    def __init__(self, max_image_bytes: int):
        self.max_image_bytes = max_image_bytes
        self.image_data = bytearray()
        self.file_found = False
        self.finished = False
        self._current_is_file = False
        self._current_headers: dict[bytes, bytes] = {}
        self._current_header_name = bytearray()
        self._current_header_value = bytearray()

    def on_part_begin(self) -> None:
        self._current_is_file = False
        self._current_headers = {}
        self._current_header_name.clear()
        self._current_header_value.clear()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._current_header_name.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._current_header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        name = bytes(self._current_header_name).lower()
        if not name:
            raise InvalidMultipartUpload("EMPTY_HEADER_NAME")
        self._current_headers[name] = bytes(self._current_header_value)
        self._current_header_name.clear()
        self._current_header_value.clear()

    def on_headers_finished(self) -> None:
        disposition, options = parse_options_header(
            self._current_headers.get(b"content-disposition")
        )
        if disposition != b"form-data":
            raise InvalidMultipartUpload("INVALID_CONTENT_DISPOSITION")

        is_file = (
            options.get(b"name") == b"file"
            and b"filename" in options
        )
        if not is_file:
            return
        if self.file_found:
            raise InvalidMultipartUpload("MULTIPLE_FILE_FIELDS")

        self.file_found = True
        self._current_is_file = True

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._current_is_file:
            return

        part = data[start:end]
        if len(self.image_data) + len(part) > self.max_image_bytes:
            raise UploadTooLarge("FILE_TOO_LARGE")
        self.image_data.extend(part)

    def on_end(self) -> None:
        self.finished = True

    def callbacks(self):
        return {
            "on_part_begin": self.on_part_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_end": self.on_end,
        }


class InferenceLimiter:
    def __init__(self, concurrency: int, queue_capacity: int):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.capacity = concurrency + queue_capacity
        self.admitted = 0

    def try_admit(self) -> bool:
        if self.admitted >= self.capacity:
            return False
        self.admitted += 1
        return True

    def release(self) -> None:
        if self.admitted <= 0:
            raise RuntimeError("INFERENCE_LIMITER_RELEASE_WITHOUT_ADMISSION")
        self.admitted -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_limiter = InferenceLimiter(
        config.INFERENCE_CONCURRENCY,
        config.INFERENCE_QUEUE_CAPACITY,
    )
    app.state.auth_manager = api_auth.AuthManager(
        config.AUTH_CHALLENGE_TTL_SECONDS,
        config.AUTH_TOKEN_TTL_SECONDS,
        config.AUTH_TIMESTAMP_SKEW_SECONDS,
        config.AUTH_CHALLENGES_PER_MINUTE,
        config.AUTH_IDENTIFY_PER_IP_PER_MINUTE,
        config.AUTH_IDENTIFY_PER_TOKEN_PER_MINUTE,
    )
    inference.init_model()
    modern_inference.init_model()
    yield


app = FastAPI(title="BirdsVision API", lifespan=lifespan)


def err(status: int, code: str, msg: str):
    return JSONResponse(
        status_code=status,
        content={"success": False, "error_code": code, "message": msg},
    )


def request_client_ip(request: Request) -> str:
    if request.client is None or not request.client.host:
        return "unknown"
    return request.client.host


def log_identify(
    request_id: str,
    status_code: int,
    total_ms: int,
    image_bytes: int = 0,
    queue_ms: int = 0,
    inference_ms: int = 0,
    error_code: str = "",
):
    log_method = logger.info if status_code < 400 else logger.warning
    log_method(
        "identify request_id=%s status=%d error_code=%s "
        "image_bytes=%d queue_ms=%d inference_ms=%d total_ms=%d",
        request_id,
        status_code,
        error_code or "-",
        image_bytes,
        queue_ms,
        inference_ms,
        total_ms,
    )


async def read_image_upload_in_memory(
    request: Request,
    max_image_bytes: int,
) -> bytes | None:
    content_type, options = parse_options_header(
        request.headers.get("content-type")
    )
    boundary = options.get(b"boundary")
    if content_type != b"multipart/form-data" or boundary is None:
        return None

    max_request_bytes = (
        max_image_bytes + MULTIPART_OVERHEAD_ALLOWANCE_BYTES
    )
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError as exc:
            raise InvalidMultipartUpload("INVALID_CONTENT_LENGTH") from exc
        if declared_bytes < 0:
            raise InvalidMultipartUpload("INVALID_CONTENT_LENGTH")
        if declared_bytes > max_request_bytes:
            raise UploadTooLarge("REQUEST_TOO_LARGE")

    state = InMemoryImageMultipartParser(max_image_bytes)
    parser = MultipartParser(
        boundary,
        state.callbacks(),
        max_size=max_request_bytes,
    )
    received_bytes = 0

    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            received_bytes += len(chunk)
            if received_bytes > max_request_bytes:
                raise UploadTooLarge("REQUEST_TOO_LARGE")
            if parser.write(chunk) != len(chunk):
                raise UploadTooLarge("REQUEST_TOO_LARGE")
        parser.finalize()
    except MultipartParseError as exc:
        raise InvalidMultipartUpload("MALFORMED_MULTIPART") from exc

    if not state.finished or not state.file_found:
        return None
    return bytes(state.image_data)


async def run_inference_limited(
    data: bytes,
    top_k: int,
    limiter: InferenceLimiter,
    predictor=None,
    manual_box=None,
):
    if not limiter.try_admit():
        raise InferenceQueueFull()

    waiting_started = time.perf_counter()
    try:
        async with limiter.semaphore:
            queue_ms = int((time.perf_counter() - waiting_started) * 1000)
            prediction_task = asyncio.create_task(
                asyncio.to_thread(
                    predictor or inference.predict,
                    data,
                    top_k,
                    *(() if manual_box is None else (manual_box,)),
                )
            )
            try:
                results, elapsed_ms = await asyncio.shield(prediction_task)
            except asyncio.CancelledError:
                with suppress(Exception):
                    await prediction_task
                raise
        return results, elapsed_ms, queue_ms
    finally:
        limiter.release()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": inference.is_ready(),
        "num_classes": config.NUM_CLASSES,
    }


@app.get("/api/version")
def app_version():
    return {
        "latest_version": config.APP_LATEST_VERSION,
        "minimum_supported_version": config.APP_MINIMUM_SUPPORTED_VERSION,
        "update_url": config.APP_UPDATE_URL,
    }


@app.get("/api/source")
def source_revision():
    if not config.SOURCE_REPOSITORY_URL:
        return err(503, "SOURCE_NOT_PUBLISHED", "Source revision has not been published")
    return {
        "repository_url": config.SOURCE_REPOSITORY_URL,
        "commit": config.SOURCE_COMMIT,
        "source_url": f"{config.SOURCE_REPOSITORY_URL.rstrip('/')}/tree/{config.SOURCE_COMMIT}",
    }


@app.get("/api/auth/challenge")
def auth_challenge(request: Request):
    try:
        return request.app.state.auth_manager.issue_challenge(
            request_client_ip(request)
        )
    except api_auth.AuthError as exc:
        return err(exc.status_code, exc.code, exc.message)


@app.post("/api/auth/token")
def auth_token(request: Request, payload: AuthTokenRequest):
    try:
        return request.app.state.auth_manager.exchange_challenge(
            request_client_ip(request),
            payload.challenge_id,
            payload.challenge,
        )
    except api_auth.AuthError as exc:
        return err(exc.status_code, exc.code, exc.message)


@app.post("/api/identify")
async def identify(
    request: Request,
    top_k: int = Query(config.DEFAULT_TOP_K),
    bird_box: str | None = Query(None),
):
    request_id = uuid.uuid4().hex
    request_started = time.perf_counter()

    try:
        route = route_for_version(request.headers.get(APP_VERSION_HEADER))
    except ValueError:
        return err(400, "INVALID_APP_VERSION", "应用版本号无效")

    expected_content_sha256: str | None = None
    if config.AUTH_REQUIRED:
        try:
            expected_content_sha256 = (
                request.app.state.auth_manager.authorize_identify(
                    request_client_ip(request),
                    request.headers.get("authorization", ""),
                    request.headers.get("x-birdsvision-timestamp", ""),
                    request.headers.get("x-birdsvision-nonce", ""),
                    request.headers.get("x-birdsvision-content-sha256", ""),
                    request.headers.get("x-birdsvision-signature", ""),
                    top_k,
                    app_version=request.headers.get(APP_VERSION_HEADER) if route == "modern" else None,
                    bird_box=bird_box if route == "modern" else None,
                )
            )
        except api_auth.AuthError as exc:
            total_ms = int((time.perf_counter() - request_started) * 1000)
            log_identify(
                request_id,
                exc.status_code,
                total_ms,
                error_code=exc.code,
            )
            return err(exc.status_code, exc.code, exc.message)

    manual_box = None
    if bird_box is not None:
        if route != "modern":
            return err(400, "INVALID_BIRD_BOX", "当前版本不支持手动画框")
        try:
            parts = tuple(float(value) for value in bird_box.split(","))
            if len(parts) != 4 or not all(math.isfinite(value) for value in parts):
                raise ValueError("INVALID_BIRD_BOX")
            left, top, right, bottom = parts
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("INVALID_BIRD_BOX")
            manual_box = parts
        except ValueError:
            return err(400, "INVALID_BIRD_BOX", "鸟框坐标无效")
    if not (modern_inference.is_ready() if route == "modern" else inference.is_ready()):
        total_ms = int((time.perf_counter() - request_started) * 1000)
        log_identify(request_id, 503, total_ms, error_code="MODEL_NOT_READY")
        return err(503, "MODEL_NOT_READY", "服务正在启动，请稍后重试")
    max_bytes = config.MAX_IMAGE_MB * 1024 * 1024
    try:
        data = await read_image_upload_in_memory(request, max_bytes)
    except UploadTooLarge:
        total_ms = int((time.perf_counter() - request_started) * 1000)
        log_identify(
            request_id,
            413,
            total_ms,
            image_bytes=max_bytes + 1,
            error_code="FILE_TOO_LARGE",
        )
        return err(413, "FILE_TOO_LARGE", "图片太大，请压缩后重试")
    except InvalidMultipartUpload:
        total_ms = int((time.perf_counter() - request_started) * 1000)
        log_identify(request_id, 400, total_ms, error_code="NO_FILE")
        return err(400, "NO_FILE", "未收到图片")

    if data is None:
        total_ms = int((time.perf_counter() - request_started) * 1000)
        log_identify(request_id, 400, total_ms, error_code="NO_FILE")
        return err(400, "NO_FILE", "未收到图片")

    if (
        expected_content_sha256 is not None
        and api_auth.content_sha256(data) != expected_content_sha256
    ):
        total_ms = int((time.perf_counter() - request_started) * 1000)
        log_identify(
            request_id,
            401,
            total_ms,
            image_bytes=len(data),
            error_code="INVALID_AUTH",
        )
        return err(401, "INVALID_AUTH", "请求验证失败，请重试")

    queue_ms = 0
    inference_ms = 0
    try:
        results, inference_ms, queue_ms = await run_inference_limited(
            data,
            top_k,
            request.app.state.inference_limiter,
            predictor=modern_inference.predict if route == "modern" else inference.predict,
            manual_box=manual_box,
        )
        if route == "legacy":
            results = legacy_results(results)
    except InferenceQueueFull:
        total_ms = int((time.perf_counter() - request_started) * 1000)
        log_identify(
            request_id,
            503,
            total_ms,
            image_bytes=len(data),
            error_code="SERVER_BUSY",
        )
        return err(503, "SERVER_BUSY", "当前请求较多，请稍后重试")
    except ValueError:
        total_ms = int((time.perf_counter() - request_started) * 1000)
        log_identify(
            request_id,
            400,
            total_ms,
            image_bytes=len(data),
            queue_ms=queue_ms,
            error_code="INVALID_IMAGE",
        )
        return err(400, "INVALID_IMAGE", "无法识别，请换一张清晰照片")
    except RuntimeError as exc:
        total_ms = int((time.perf_counter() - request_started) * 1000)
        if str(exc) == "MODEL_NOT_READY":
            log_identify(
                request_id,
                503,
                total_ms,
                image_bytes=len(data),
                queue_ms=queue_ms,
                error_code="MODEL_NOT_READY",
            )
            return err(503, "MODEL_NOT_READY", "服务正在启动，请稍后重试")
        logger.exception("identify_exception request_id=%s", request_id)
        log_identify(
            request_id,
            500,
            total_ms,
            image_bytes=len(data),
            queue_ms=queue_ms,
            error_code="INFERENCE_ERROR",
        )
        return err(500, "INFERENCE_ERROR", "服务暂时不可用，请稍后重试")
    except Exception:
        total_ms = int((time.perf_counter() - request_started) * 1000)
        logger.exception("identify_exception request_id=%s", request_id)
        log_identify(
            request_id,
            500,
            total_ms,
            image_bytes=len(data),
            queue_ms=queue_ms,
            error_code="INFERENCE_ERROR",
        )
        return err(500, "INFERENCE_ERROR", "服务暂时不可用，请稍后重试")

    total_ms = int((time.perf_counter() - request_started) * 1000)
    log_identify(
        request_id,
        200,
        total_ms,
        image_bytes=len(data),
        queue_ms=queue_ms,
        inference_ms=inference_ms,
    )
    return {"success": True, "results": results, "elapsed_ms": inference_ms}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )
