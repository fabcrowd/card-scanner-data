"""Synthetic photo-like variants of clean catalog scans: the clean-scan to marketplace-photo gap the adapter learns.

Each effect fires with its own probability from a seeded generator, so the same (image, seed) gives the same pixels.
"""
import io

import numpy as np
from PIL import Image, ImageFilter


def _glare(arr, rng):
    h, w, _ = arr.shape
    cy, cx = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
    sigma = rng.uniform(0.1, 0.3) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    return arr + blob[..., None] * rng.uniform(60, 160)


def _sheen(arr, rng):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    angle = rng.uniform(0, np.pi)
    d = (xx * np.cos(angle) + yy * np.sin(angle)) / max(h, w)
    centre, width = rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.2)
    return arr + np.exp(-((d - centre) ** 2) / (2 * width**2))[..., None] * rng.uniform(25, 70)


def _warm(arr, rng):
    k = rng.uniform(0.04, 0.18)
    return arr * np.array([1 + k, 1 + k / 3, 1 - k], dtype=np.float32)


def _noise(arr, rng):
    return arr + rng.normal(0, rng.uniform(2, 8), arr.shape)


def _tilt(img, rng):
    w, h = img.size
    j = rng.uniform(-0.06, 0.06, 8) * np.array([w, h] * 4)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    src = [(x + j[2 * i], y + j[2 * i + 1]) for i, (x, y) in enumerate(corners)]
    # PIL maps output corners to the given source quad
    return img.transform((w, h), Image.QUAD, tuple(c for p in src for c in p), Image.BICUBIC, fillcolor=(128, 128, 128))


def photo_like(img, seed=0):
    """A PIL RGB image that looks like a phone photo of `img`; the input is never modified."""
    rng = np.random.default_rng(seed)
    out = img.convert("RGB")
    if rng.random() < 0.7:
        out = _tilt(out, rng)
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.8)))
    arr = np.asarray(out, dtype=np.float32)
    for effect, p in ((_glare, 0.5), (_sheen, 0.5), (_warm, 0.6), (_noise, 0.8)):
        if rng.random() < p:
            arr = effect(arr, rng)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=int(rng.integers(40, 85)))
    out = Image.open(io.BytesIO(buf.getvalue()))
    out.load()
    return out.convert("RGB")
