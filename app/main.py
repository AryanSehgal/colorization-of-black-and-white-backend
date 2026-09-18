from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .colorizer import Colorizer, InvalidImage, decode_image, MAX_EDGE, MAX_PIXELS

MAX_FILE_BYTES = 12 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
colorizer = Colorizer(ROOT / "models")
# OpenCV Net is stateful. Serialize decode + inference and bound memory usage.
inference_lock = threading.Lock()
logger = logging.getLogger("chroma")


class BodyLimitMiddleware:
    """Bound the multipart body even when Content-Length is absent."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        total = 0
        exceeded = False
        async def limited_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > MAX_FILE_BYTES + 128 * 1024:
                    exceeded = True
                    raise HTTPException(413, "Upload exceeds the 12 MB limit.")
            return message
        async def limited_send(message):
            if exceeded:
                return
            await send(message)
        try:
            await self.app(scope, limited_receive, limited_send)
        except HTTPException:
            if not exceeded:
                raise
        if exceeded:
            await JSONResponse({"detail": "Upload exceeds the 12 MB limit."}, status_code=413)(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(colorizer.load)
    if colorizer.net is None:
        logger.warning("Model unavailable: %s", colorizer.error)
    yield


app = FastAPI(title="Chroma Studio", version="1.0.0", lifespan=lifespan)
app.add_middleware(BodyLimitMiddleware)
allowed_origins = [origin.strip().rstrip("/") for origin in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Image-Width", "X-Image-Height", "X-Processing-Seconds", "Content-Disposition"],
)


@app.get("/api/health")
def health():
    return {"ready": colorizer.net is not None, "model": "ECCV 2016 · OpenCV CPU",
            "message": colorizer.error, "max_file_mb": 12, "max_edge": MAX_EDGE, "max_pixels": MAX_PIXELS}


def run_inference(data: bytes, intensity: float):
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(429, "The model is busy. Wait a moment and retry this photo.")
    try:
        started = time.perf_counter()
        rgb = decode_image(data)
        result = colorizer.colorize(rgb, intensity)
        return result, rgb.shape[1], rgb.shape[0], time.perf_counter() - started
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Local inference failed")
        raise HTTPException(500, "Colorization failed. Check the backend terminal and retry.") from exc
    finally:
        inference_lock.release()


@app.post("/api/colorize", responses={200: {"content": {"image/png": {}}}})
async def colorize(file: UploadFile = File(...), intensity: float = Form(1.0, ge=0.0, le=1.5)):
    try:
        if colorizer.net is None:
            raise HTTPException(503, "Model unavailable. Run the model download script and restart the backend.")
        data = await file.read(MAX_FILE_BYTES + 1)
        if not data:
            raise HTTPException(422, "The uploaded file is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, "Each photo must be 12 MB or smaller.")
        result, width, height, seconds = await run_in_threadpool(run_inference, data, intensity)
        return Response(result, media_type="image/png", headers={
            "Content-Disposition": 'attachment; filename="colorized.png"',
            "Cache-Control": "no-store", "X-Image-Width": str(width),
            "X-Image-Height": str(height), "X-Processing-Seconds": f"{seconds:.2f}",
        })
    finally:
        await file.close()


