"""Builders shared by the pipeline tests."""
import base64
import io
import json
import os

import _paths  # noqa: F401
import numpy as np
from PIL import Image

DIM = 512
CLIP = "0123456789abcdef"
CATALOG = "fedcba9876543210"


def jpeg_b64(seed=0, size=(100, 140)):
    img = Image.new("RGB", size, (seed % 256, (seed * 7) % 256, (seed * 13) % 256))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def photo_id(n, split):
    """A 16-hex photoId whose low 32 bits give the wanted split."""
    k = n * 5 + (0 if split == "exam" else 1 + n % 4)
    return f"{0xAB:08x}{k:08x}"


def install_id(n):
    """The relay stores a 16-hex HMAC in place of the install uuid."""
    return f"{n:016x}"


def card(card_id="base1-4", box=(0.1, 0.2, 0.3, 0.4), answer=None, seed=0, scanner_id="base1-5", conf=0.8, stripe="high"):
    return {
        "box": list(box),
        "crop": jpeg_b64(seed),
        "answer": answer if answer is not None else {"cardId": card_id},
        "scanner": {"cardId": scanner_id, "confidence": conf, "stripe": stripe},
        "source": "pick",
    }


def item(install=1, photo=None, split="teach", cards=None, day="2026-10-15", received="2026-10-15T12:00:00.000Z"):
    photo = photo or photo_id(1, split)
    return {
        "v": 1,
        "install": install_id(install),
        "split": split,
        "photoId": photo,
        "day": day,
        "received": received,
        "model": {"clip": CLIP, "crop": 2},
        "catalog": CATALOG,
        "cards": cards if cards is not None else [card()],
    }


def write_item(root, it, name):
    d = os.path.join(root, "contributions", it["received"][:10])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(it, f)


def unit(k, noise=0.0, seed=0):
    """A unit vector near basis direction k; noise > 0 moves it a little."""
    v = np.zeros(DIM, dtype=np.float32)
    v[k] = 1.0
    if noise:
        v += np.random.RandomState(seed).randn(DIM).astype(np.float32) * noise
    return v / np.linalg.norm(v)
