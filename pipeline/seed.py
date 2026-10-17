"""Seed pairs (embedding, card id) for the adapter: labelled real crops plus synthetic photo-like catalog variants.

HiBid reviewed boxes are the fixed regression exam, and data/sealed-test and data/realworld are never to be tuned
against, so a real crop from any of them stops the build (ForbiddenSource) instead of being skipped quietly.
"""
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from .augment import photo_like

FORBIDDEN = ("hibid", "sealed-test", "realworld")
DIGITAL = re.compile(r"^(A\d|B\d|P-A)")  # TCG Pocket ids; the physical search drops them too
SKIP_TOKENS = {"ex", "gx", "v", "vmax", "vstar", "lv", "x", "star", "the"}
NUMBER = re.compile(r"(?<!\d)(\d{1,3})\s*/\s*\d{1,3}(?!\d)|#\s*(\d{1,3})(?!\d)")


class ForbiddenSource(ValueError):
    pass


@dataclass
class RealCrop:
    data: bytes  # card crop image bytes
    card_id: str
    origin: str  # where it came from, e.g. "variant:123_00_0.jpg"; checked against FORBIDDEN


@dataclass
class Sources:
    embedder: Callable  # (image bytes, trusted=False) -> unit float32[512], e.g. embed.ClipEmbedder
    real: list = field(default_factory=list)
    catalog_images: list = field(default_factory=list)  # [(image path, card id)]
    synthetic_per_card: int = 1
    max_synthetic: int = None  # catalog cards sampled for synthetic variants; None means all
    seed: int = 0


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("é", "e"))


def _name_tokens(card):
    return [t for t in _tokens(card.get("name", "")) if t not in SKIP_TOKENS and len(t) > 1]


def build_title_index(meta):
    """first name token -> physical cards, so labelling a title does not scan the whole catalog."""
    index = {}
    for card in meta:
        if DIGITAL.match(str(card.get("id", ""))):
            continue
        toks = _name_tokens(card)
        if toks:
            index.setdefault(toks[0], []).append(card)
    return index


def label_title(title, meta, index=None):
    """The one physical card id a listing title names, else None.

    The card's name tokens and its printed number (80/111 or #80) must be in the title; if several sets still
    match, the set name has to appear in the title and leave exactly one.
    """
    index = index if index is not None else build_title_index(meta)
    words = set(_tokens(title))
    numbers = {(a or b).lstrip("0") for a, b in NUMBER.findall(title)}
    if not numbers:
        return None
    cands = []
    for w in words:
        for card in index.get(w, []):
            if all(t in words for t in _name_tokens(card)) and str(card.get("number", "")).lstrip("0") in numbers:
                cands.append(card)
    if len(cands) > 1:
        cands = [c for c in cands if (st := [t for t in _tokens(c.get("set", "")) if len(t) > 1]) and all(t in words for t in st)]
    ids = {c["id"] for c in cands}
    return ids.pop() if len(ids) == 1 else None


def real_crops_from_repo(data_dir, meta):
    """Labelled non-HiBid crops: variant photos and edition crops whose listing title maps to exactly one card."""
    index = build_title_index(meta)
    titles = {}
    with open(os.path.join(data_dir, "variant-photos", "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            titles[r["listing"]] = r.get("title") or ""
    by_listing = {k: label_title(t, meta, index) for k, t in titles.items()}
    out = []
    for folder, tag in (("variant-photos/crops", "variant"), ("edition-crops", "edition")):
        d = os.path.join(data_dir, folder)
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            card_id = by_listing.get(name.split("_")[0])
            if card_id and name.lower().endswith((".jpg", ".jpeg", ".png")):
                with open(os.path.join(d, name), "rb") as f:
                    out.append(RealCrop(f.read(), card_id, f"{tag}:{name}"))
    return out


def _refuse_forbidden(origin):
    low = origin.replace("\\", "/").lower()
    if any(word in low for word in FORBIDDEN):
        raise ForbiddenSource(f"seed may not use {origin}")


def _jpeg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def build_seed(sources):
    """[(float32[512], card id)]: real crops first, then synthetic variants in sorted catalog order."""
    for crop in sources.real:
        _refuse_forbidden(crop.origin)
    pairs = []
    for crop in sources.real:
        try:
            pairs.append((np.asarray(sources.embedder(crop.data, trusted=True), dtype=np.float32), crop.card_id))
        except ValueError:
            continue
    images = sorted(sources.catalog_images, key=lambda x: x[1])
    if sources.max_synthetic is not None and sources.max_synthetic < len(images):
        pick = np.random.RandomState(sources.seed).choice(len(images), sources.max_synthetic, replace=False)
        images = [images[i] for i in sorted(pick)]
    for n, (path, card_id) in enumerate(images):
        try:
            with Image.open(path) as im:
                base = im.convert("RGB")
        except OSError:
            continue
        for k in range(sources.synthetic_per_card):
            variant = photo_like(base, seed=sources.seed * 1_000_003 + n * 31 + k)
            try:
                pairs.append((np.asarray(sources.embedder(_jpeg(variant), trusted=True), dtype=np.float32), card_id))
            except ValueError:
                continue
    return pairs
