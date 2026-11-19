"""The fixed, server-owned regression set: embeddings and labels of real crops, no images.

A row is labelled either with one catalog id or with a card name (names.NameTruth: the base name and the shipped ids that
have it). Contributors cannot touch the file. The pack must not lower top-1 accuracy on it (see manifest.is_improved).
"""
import hashlib
import json
import os

import numpy as np

from .names import MAX_ID_LEN, MAX_NAME_LEN, NameTruth

DIM = 512
UNIT_TOLERANCE = 1e-3
DEFAULT_PINS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pins.json")
ID_KEYS = (["cardIds", "embeddings"], ["cardIds", "crop", "embeddings", "model"])
NAME_KEYS = ["acceptIds", "acceptOffsets", "cardIds", "crop", "embeddings", "model", "names"]


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
    job can tell which recognizer made the embeddings. card_ids holds one label per embedding: a catalog id (str) or a
    names.NameTruth. A file with only ids keeps the old key set; one with a name row uses the name key set and must be tagged."""
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, DIM)
    if len(embeddings) != len(card_ids):
        raise ValueError("one card id per embedding is required")
    tag = {} if model is None or crop is None else {"model": np.array(str(model)), "crop": np.array(int(crop))}
    if not any(isinstance(t, NameTruth) for t in card_ids):
        np.savez_compressed(path, embeddings=embeddings, cardIds=np.array(list(card_ids), dtype="U64"), **tag)
        return
    if not tag:
        raise ValueError("a regression file with name rows must carry the model and crop tag")
    ids = [t if isinstance(t, str) else "" for t in card_ids]
    names = [t.name if isinstance(t, NameTruth) else "" for t in card_ids]
    accepts = [sorted(t.accept) if isinstance(t, NameTruth) else [] for t in card_ids]
    offsets = np.cumsum([0] + [len(a) for a in accepts]).astype(np.int32)
    flat = [i for a in accepts for i in a]
    if any(not i or len(i) > MAX_ID_LEN for i in flat) or any(len(i) > MAX_ID_LEN for i in ids):
        raise ValueError(f"regression ids must be 1 to {MAX_ID_LEN} characters")
    np.savez_compressed(
        path, embeddings=embeddings, cardIds=np.array(ids, dtype=f"U{MAX_ID_LEN}"), names=np.array(names, dtype=f"U{MAX_NAME_LEN}"),
        acceptIds=np.array(flat, dtype=f"U{MAX_ID_LEN}"), acceptOffsets=offsets, **tag,
    )


def _strings(array, limit, what):
    """The values of a 1-d unicode array, each at most `limit` characters."""
    if array.ndim != 1 or array.dtype.kind != "U":
        raise ValueError(f"regression {what} must be a 1-d array of strings")
    values = [str(v) for v in array]
    if any(len(v) > limit for v in values):
        raise ValueError(f"a regression {what} entry is longer than {limit} characters")
    return values


def _tag_is_valid(model, crop):
    return model.shape == () and model.dtype.kind == "U" and crop.shape == () and crop.dtype.kind in "iu"


def _name_truths(raw, n):
    """One label per row from the name key set. ValueError for anything the file builder would not have written."""
    ids = _strings(raw["cardIds"], MAX_ID_LEN, "card id")
    names = _strings(raw["names"], MAX_NAME_LEN, "name")
    accept_ids = _strings(raw["acceptIds"], MAX_ID_LEN, "accept id")
    offsets = np.asarray(raw["acceptOffsets"])
    if len(ids) != n or len(names) != n:
        raise ValueError("regression labels do not line up with the embeddings")
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != n + 1:
        raise ValueError("regression acceptOffsets must hold n + 1 integers")
    if offsets[0] != 0 or offsets[-1] != len(accept_ids) or np.any(np.diff(offsets.astype(np.int64)) < 0):
        raise ValueError("regression acceptOffsets must start at 0, never decrease and end at the number of accept ids")
    if any(not a for a in accept_ids):
        raise ValueError("a regression accept id is empty")
    truths = []
    for i in range(n):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if bool(ids[i]) == bool(names[i]):
            raise ValueError("every regression row needs exactly one of a card id and a name")
        if ids[i]:
            if hi != lo:
                raise ValueError("an id-labelled regression row cannot have accept ids")
            truths.append(ids[i])
        else:
            truths.append(NameTruth(names[i], frozenset(accept_ids[lo:hi])))
    return truths


def load_regression(path):
    """(embeddings float32 (n, 512), labels): one label per row, a card id (str) or a names.NameTruth. Raises ValueError for
    a file that is not a regression set."""
    with np.load(path, allow_pickle=False) as raw:
        keys = sorted(raw.files)
        if keys not in ID_KEYS and keys != NAME_KEYS:
            raise ValueError("regression file must hold exactly embeddings and cardIds (and optionally model and crop), or the name key set")
        embeddings = np.asarray(raw["embeddings"], dtype=np.float32)
        if keys == NAME_KEYS:
            if not _tag_is_valid(raw["model"], raw["crop"]):
                raise ValueError("regression model and crop must be one string and one integer")
            if embeddings.ndim != 2:
                raise ValueError("regression embeddings and ids do not line up")
            labels = _name_truths(raw, len(embeddings))
        else:
            labels = [str(i) for i in raw["cardIds"]]
    if embeddings.ndim != 2 or embeddings.shape[1] != DIM or len(embeddings) != len(labels):
        raise ValueError("regression embeddings and ids do not line up")
    if len(embeddings) and not np.all(np.abs(np.linalg.norm(embeddings, axis=1) - 1) <= UNIT_TOLERANCE):
        raise ValueError("regression embeddings must be unit length")
    return embeddings, labels
