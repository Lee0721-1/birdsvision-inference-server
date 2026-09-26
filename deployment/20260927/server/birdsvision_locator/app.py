# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""SOYOL-only ASGI process. It never loads classifier code or weights."""

from __future__ import annotations

import io
import os
import threading
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
import torch


MODEL_ENV = "BIRDSVISION_SOYOL_MODEL_PATH"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
_model = None
_lock = threading.Lock()


def init_model() -> None:
    global _model
    if _model is not None:
        return
    model_path = os.getenv(MODEL_ENV)
    if not model_path:
        raise RuntimeError(f"set {MODEL_ENV} for the SOYOL locator process")
    from ultralytics import YOLO
    model = YOLO(model_path)
    if model.task != "detect" or model.names != {0: "bird"}:
        raise ValueError("SOYOL must be a single-class bird Detect model")
    model.model.end2end = False
    if model.model.end2end:
        raise ValueError("SOYOL NMS branch is not active")
    _model = model


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.verify()
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
                image.load()
                return image
    except Exception as exc:
        raise ValueError("INVALID_IMAGE") from exc


def locate(image_bytes: bytes) -> dict:
    if _model is None:
        raise RuntimeError("MODEL_NOT_READY")
    image = decode_image(image_bytes)
    with _lock:
        predictions = _model.predict(
            source=image, imgsz=640, conf=0.25, iou=0.7, max_det=10,
            device="cuda" if torch.cuda.is_available() else "cpu", verbose=False,
        )
    if len(predictions) != 1 or _model.model.end2end:
        raise RuntimeError("SOYOL returned an invalid prediction")
    result_boxes = predictions[0].boxes
    boxes = [] if result_boxes is None else result_boxes.xyxy.detach().cpu().tolist()
    return {"width": image.width, "height": image.height, "boxes": boxes}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_model()
    yield


app = FastAPI(title="BirdsVision SOYOL Locator", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "starting",
            "model_loaded": _model is not None}


@app.post("/v1/locate")
async def locate_endpoint(request: Request):
    if request.headers.get("content-type") != "application/octet-stream":
        return JSONResponse(status_code=415, content={"error": "UNSUPPORTED_MEDIA_TYPE"})
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > MAX_IMAGE_BYTES:
            return JSONResponse(status_code=413, content={"error": "FILE_TOO_LARGE"})
        chunks.extend(chunk)
    try:
        import asyncio
        return await asyncio.to_thread(locate, bytes(chunks))
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "INVALID_IMAGE"})
    except RuntimeError as exc:
        if str(exc) == "MODEL_NOT_READY":
            return JSONResponse(status_code=503, content={"error": "MODEL_NOT_READY"})
        raise
