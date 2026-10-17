"""Embeds a card crop exactly like the extension (tools/extension-wasm/lib/embed.js, EMBEDDING_VERSION 2).

The crop is the rectified card; only the top 45% strip is used, as a centred square resized to 224 px.
Integer maths and PIL bicubic follow scripts/embed-card-back.py, the Python twin of lib/embed.js.
"""
import io

import numpy as np
from PIL import Image

# Contributions come from strangers: decoding is bounded before any pixels are read. These match the relay.
Image.MAX_IMAGE_PIXELS = 1_000_000
MAX_SIDE = 400
MIN_LONG_SIDE = 64
CLIP_SIZE = 224
ART_CROP_FRAC = 0.45
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def art_rect(width, height):
    """(sx, sy, side): the centred square of the top 45% strip, as lib/embed.js artSourceRect."""
    strip = max(1, int(height * ART_CROP_FRAC))
    side = min(width, strip)
    return (width - side) // 2, (strip - side) // 2, side


def preprocess(jpeg_bytes, trusted=False):
    """(1, 3, 224, 224) float32 CLIP input for a card JPEG.

    Untrusted crops are size-checked from the header before conversion, then decoded at reduced size (JPEG draft
    mode). `trusted` is for our own full-size reference images and skips only the size limits.
    """
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        width, height = img.size
        if not trusted and (max(width, height) > MAX_SIDE or max(width, height) < MIN_LONG_SIDE or width * height > Image.MAX_IMAGE_PIXELS):
            raise ValueError(f"crop is {width}x{height}")
        if not trusted:
            img.draft("RGB", (320, 320))
        card = img.convert("RGB")
    except Exception as err:  # PIL raises several types for bad data
        raise ValueError(f"cannot decode crop: {err}") from err
    sx, sy, side = art_rect(*card.size)
    art = card.crop((sx, sy, sx + side, sy + side)).resize((CLIP_SIZE, CLIP_SIZE), Image.BICUBIC)
    x = ((np.asarray(art, dtype=np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32)


class ClipEmbedder:
    """Callable: card JPEG bytes -> L2-normalised 512-float vector, using the pinned int8 CLIP visual model."""

    def __init__(self, model_path):
        import onnxruntime as ort

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def __call__(self, jpeg_bytes, trusted=False):
        v = self._session.run(None, {self._input: preprocess(jpeg_bytes, trusted)})[0].reshape(-1).astype(np.float32)
        return v / np.linalg.norm(v)
