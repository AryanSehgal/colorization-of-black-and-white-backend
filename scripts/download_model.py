"""One-time download only. The running app never downloads or uploads externally."""
from pathlib import Path
import hashlib
import shutil
import sys
import urllib.request
import urllib.error

DEST = Path(__file__).resolve().parents[1] / "models"
# SHA-1 references are published in OpenCV's official download_models.py.
ASSETS = [
    ("colorization_deploy_v2.prototxt",
     "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt",
     "f528334e386a69cbaaf237a7611d833bef8e5219"),
    ("colorization_release_v2.caffemodel",
     "https://dl.opencv.org/models/colorization_release_v2.caffemodel",
     "21e61293a3fa6747308171c11b6dd18a68a26e7f"),
    ("pts_in_hull.npy",
     "https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/pts_in_hull.npy", None),
    ("MODEL_LICENSE.txt",
     "https://raw.githubusercontent.com/richzhang/colorization/caffe/LICENSE", None),
]


def valid(path, checksum):
    if not path.is_file():
        return False
    if checksum:
        digest = hashlib.sha1()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == checksum
    if path.suffix == ".npy":
        import numpy as np
        try:
            points = np.load(path, allow_pickle=False)
            return points.shape == (313, 2) and bool(np.isfinite(points).all())
        except (OSError, ValueError):
            return False
    return path.stat().st_size > 100


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url, checksum in ASSETS:
        target = DEST / name
        if valid(target, checksum):
            print(f"Verified: {name}", flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {name}…", flush=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ChromaStudio/1.0"})
            try:
                response = urllib.request.urlopen(request, timeout=120)
            except (urllib.error.URLError, TimeoutError):
                if name != "colorization_release_v2.caffemodel":
                    raise
                # Community-hosted copy, accepted ONLY if it matches OpenCV's hash.
                print("Primary host unavailable; trying checksum-verified mirror…", flush=True)
                mirror = "https://www.dropbox.com/scl/fi/d8zffur3wmd4wet58dp9x/colorization_release_v2.caffemodel?rlkey=iippu6vtsrox3pxkeohcuh4oy&dl=1"
                response = urllib.request.urlopen(mirror, timeout=120)
            with response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not valid(partial, checksum if checksum else None):
                raise ValueError(f"Invalid download: {name}")
            partial.replace(target)
            if not valid(target, checksum):
                target.unlink(missing_ok=True)
                raise ValueError(f"Validation failed: {name}")
        finally:
            partial.unlink(missing_ok=True)
    print("Model ready. You can now run the backend offline.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"Download failed: {exc}. Check your connection and rerun this command.")
