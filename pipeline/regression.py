"""The fixed, server-owned regression set: embeddings and catalog ids of labelled crops, no images.

Contributors cannot touch it. The pack must not lower top-1 accuracy on it (see manifest.is_improved).
"""
import hashlib
import json
import os

import numpy as np

DIM = 512
UNIT_TOLERANCE = 1e-3
DEFAULT_PINS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pins.json")


def tag_for(model_path, pins_path=None):
    """(model, crop) for the files the build scripts write: the first 16 hex of the recognizer file's sha256, and the
    crop version from pins.json (default: the template's own)."""
    digest = hashlib.sha256()
    with open(model_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    with open(pins_path or DEFAULT_PINS, encoding="utf-8") as f:
        return digest.hexdigest()[:16], json.load(f)["cropVersion"]


def save_regression(path, embeddings, card_ids, model=None, crop=None):
    """model (the recognizer's first 16 hex of sha256) and crop (the crop version) are stored when given, so the nightly
    job can tell which recognizer made the embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, DIM)
    if len(embeddings) != len(card_ids):
        raise ValueError("one card id per embedding is required")
    tag = {} if model is None or crop is None else {"model": np.array(str(model)), "crop": np.array(int(crop))}
    np.savez_compressed(path, embeddings=embeddings, cardIds=np.array(list(card_ids), dtype="U64"), **tag)


def load_regression(path):
    """(embeddings float32 (n, 512), card ids). Raises ValueError for a file that is not a regression set."""
    with np.load(path, allow_pickle=False) as raw:
        if sorted(raw.files) not in (["cardIds", "embeddings"], ["cardIds", "crop", "embeddings", "model"]):
            raise ValueError("regression file must hold exactly embeddings and cardIds (and optionally model and crop)")
        embeddings = np.asarray(raw["embeddings"], dtype=np.float32)
        ids = [str(i) for i in raw["cardIds"]]
    if embeddings.ndim != 2 or embeddings.shape[1] != DIM or len(embeddings) != len(ids):
        raise ValueError("regression embeddings and ids do not line up")
    if len(embeddings) and not np.all(np.abs(np.linalg.norm(embeddings, axis=1) - 1) <= UNIT_TOLERANCE):
        raise ValueError("regression embeddings must be unit length")
    return embeddings, ids
