"""Local ECCV 2016 inference. This module never makes network requests."""
from io import BytesIO
from pathlib import Path
import warnings
import os

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", "24000000"))
MAX_EDGE = int(os.environ.get("MAX_IMAGE_EDGE", "2400"))
MODEL_INPUT_SIZE = int(os.environ.get("MODEL_INPUT_SIZE", "224"))
if MODEL_INPUT_SIZE not in {128, 160, 192, 224}:
    raise ValueError("MODEL_INPUT_SIZE must be 128, 160, 192, or 224.")
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("Please upload a JPEG, PNG, or WebP image.")
                if getattr(source, "is_animated", False):
                    raise InvalidImage("Animated images are not supported. Use a still photo.")
                if source.width * source.height > MAX_PIXELS:
                    raise InvalidImage(f"Image exceeds the {MAX_PIXELS / 1_000_000:g}-megapixel limit. Resize the photo and retry.")
                source.load()
                oriented = ImageOps.exif_transpose(source)
                # Resize before allocating RGBA/composite copies of large inputs.
                oriented.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
                # Composite transparency rather than turning transparent pixels black.
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
                image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
                return np.asarray(image)
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidImage("This file is damaged, unsupported, or too large to decode safely.") from exc


class Colorizer:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.net = None
        self.error = "Model is not loaded."

    def load(self):
        try:
            # Bound OpenCV's native thread pool on small CPU instances.
            cv2.setNumThreads(1)
            proto = self.model_dir / "colorization_deploy_v2.prototxt"
            weights = self.model_dir / "colorization_release_v2.caffemodel"
            centers = self.model_dir / "pts_in_hull.npy"
            if not all(p.is_file() for p in (proto, weights, centers)):
                raise FileNotFoundError("Run python scripts/download_model.py, then restart the backend.")
            net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
            points = np.load(centers, allow_pickle=False)
            if points.shape != (313, 2) or not np.isfinite(points).all():
                raise ValueError("Invalid color cluster centers. Download the model again.")
            net.getLayer(net.getLayerId("class8_ab")).blobs = [
                points.T.reshape(2, 313, 1, 1).astype(np.float32)
            ]
            net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
                np.full((1, 313), 2.606, dtype=np.float32)
            ]
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            # Winograd's transformed weights/workspace substantially increase
            # peak RAM for this model. Prefer lower memory use over throughput.
            net.enableWinograd(False)
            self.net = net
            self.error = ""
        except (OSError, ValueError, cv2.error) as exc:
            self.net = None
            self.error = str(exc)

    def colorize(self, rgb: np.ndarray, intensity: float) -> bytes:
        if self.net is None:
            raise RuntimeError(self.error)
        # Retain luminance; predict only the a/b chroma channels in CIE Lab.
        normalized = rgb.astype(np.float32) / 255.0
        # Copy just L so this view does not retain all three full-size Lab channels.
        lightness = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)[:, :, 0].copy()
        small = cv2.resize(normalized, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        del normalized
        centered = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[:, :, 0] - 50.0
        self.net.setInput(cv2.dnn.blobFromImage(centered))
        chroma = self.net.forward()[0].transpose(1, 2, 0)
        chroma = cv2.resize(chroma, (rgb.shape[1], rgb.shape[0])) * intensity
        lab = np.concatenate((lightness[:, :, None], chroma), axis=2)
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        result = (np.clip(output, 0, 1) * 255).round().astype(np.uint8)
        stream = BytesIO()
        Image.fromarray(result).save(stream, format="PNG")
        return stream.getvalue()
