from io import BytesIO
from pathlib import Path
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.colorizer import Colorizer, InvalidImage, decode_image


def image_bytes(mode="RGB", size=(48, 32), color="gray", format="PNG"):
    stream = BytesIO()
    Image.new(mode, size, color).save(stream, format=format)
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch):
    # Request validation tests do not need a 129 MB model download.
    monkeypatch.setattr(main.colorizer, "load", lambda: None)
    monkeypatch.setattr(main.colorizer, "net", object())
    monkeypatch.setattr(main.colorizer, "colorize", lambda rgb, intensity: image_bytes(size=(rgb.shape[1], rgb.shape[0])))
    with TestClient(main.app) as test_client:
        yield test_client


def test_png_response(client):
    response = client.post("/api/colorize", files={"file": ("test.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert Image.open(BytesIO(response.content)).size == (48, 32)


@pytest.mark.parametrize("content", [b"", b"not an image", b"<svg></svg>"])
def test_invalid_uploads(client, content):
    response = client.post("/api/colorize", files={"file": ("pretend.png", content, "image/png")})
    assert response.status_code == 422


@pytest.mark.parametrize("intensity", ["-0.1", "1.6", "nan", "not-a-number"])
def test_intensity_validation(client, intensity):
    response = client.post("/api/colorize", files={"file": ("test.png", image_bytes())}, data={"intensity": intensity})
    assert response.status_code == 422


def test_oversize_file(client):
    response = client.post("/api/colorize", files={"file": ("huge.png", b"0" * (main.MAX_FILE_BYTES + 1))})
    assert response.status_code == 413


def test_oversize_request_body(client):
    response = client.post("/api/colorize", content=b"0" * (main.MAX_FILE_BYTES + 200_000), headers={"Content-Type": "multipart/form-data; boundary=test"})
    assert response.status_code == 413


def test_unavailable_model(client, monkeypatch):
    monkeypatch.setattr(main.colorizer, "net", None)
    assert client.get("/api/health").json()["ready"] is False
    assert client.post("/api/colorize", files={"file": ("x.png", image_bytes())}).status_code == 503


def test_busy_model(client):
    with main.inference_lock:
        assert client.post("/api/colorize", files={"file": ("x.png", image_bytes())}).status_code == 429


def test_lock_released_after_error(client):
    assert client.post("/api/colorize", files={"file": ("x.png", b"broken")}).status_code == 422
    assert client.post("/api/colorize", files={"file": ("x.png", image_bytes())}).status_code == 200


def test_transparency_and_resize():
    decoded = decode_image(image_bytes("RGBA", (3000, 1500), (0, 0, 0, 0)))
    assert decoded.shape == (1200, 2400, 3)
    assert np.all(decoded == 255)


def test_exif_orientation():
    source = Image.new("RGB", (40, 20), "gray")
    exif = source.getexif()
    exif[274] = 6
    stream = BytesIO()
    source.save(stream, format="JPEG", exif=exif)
    assert decode_image(stream.getvalue()).shape == (40, 20, 3)


def test_oversize_pixels():
    with pytest.raises(InvalidImage):
        decode_image(image_bytes(size=(5000, 5000)))


def test_animated_image():
    stream = BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (8, 8), "blue")], duration=100, loop=0)
    with pytest.raises(InvalidImage, match="Animated"):
        decode_image(stream.getvalue())


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1 after downloading weights")
def test_real_local_inference():
    model = Colorizer(Path(__file__).resolve().parents[1] / "models")
    model.load()
    assert model.net is not None, model.error
    gray = np.tile(np.linspace(25, 230, 96, dtype=np.uint8), (64, 1))
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    colored = np.asarray(Image.open(BytesIO(model.colorize(rgb, 1.0))))
    neutral = np.asarray(Image.open(BytesIO(model.colorize(rgb, 0.0))))
    assert colored.shape == rgb.shape
    assert np.abs(colored[:, :, 0].astype(float) - colored[:, :, 2]).mean() > 1
    assert np.abs(neutral[:, :, 0].astype(float) - neutral[:, :, 2]).max() <= 1


def test_cors_allows_local_frontend_and_exposes_image_metadata(client):
    response = client.post('/api/colorize', headers={'Origin': 'http://localhost:5173'},
                           files={'file': ('test.png', image_bytes(), 'image/png')})
    assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'
    assert 'X-Image-Width' in response.headers['access-control-expose-headers']


def test_cors_does_not_allow_an_unlisted_origin(client):
    response = client.get('/api/health', headers={'Origin': 'https://unlisted.example'})
    assert 'access-control-allow-origin' not in response.headers
