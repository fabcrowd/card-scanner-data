"""Data side of the full-tower asymmetric-distillation pilot: split, view sampling, dataset."""
import json
import math
import os
import random
import re
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_synth_small as E  # noqa: E402
from pipeline.embed import CLIP_SIZE, MEAN, STD, art_rect  # noqa: E402

import io  # noqa: E402

LARGE_H = 320
JITTER = 0.06
BLUR_P = 0.35
BLUR_SIGMA = (0.3, 1.2)
JPEG_Q = (40, 85)
H_FOCUS = (70, 180)
H_WIDE = (48, 320)
FOCUS_FRAC = 0.6
FILTERS = {"area": Image.BOX, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}
# TCG Pocket (digital-only) ids: A1-, A1a-, A2b-, A3-, A4a-, B1-, B1a-, B2-, P-A- ... (sets with an uppercase code).
DIGITAL_RE = re.compile(r"^(?:[AB]\d+[a-z]?|P-[AB])-")


def is_digital(card_id):
    return bool(DIGITAL_RE.match(card_id))


def sample_height(rng):
    lo, hi = H_FOCUS if rng.random() < FOCUS_FRAC else H_WIDE
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def resize_to_height(img, h, resample):
    return img.resize((max(1, round(img.width * h / img.height)), h), resample)


def to_tensor(card):
    """Art crop (embed.py art_rect) -> 224 px PIL bicubic -> CLIP-normalised CHW float32."""
    sx, sy, side = art_rect(*card.size)
    art = card.crop((sx, sy, sx + side, sy + side)).resize((CLIP_SIZE, CLIP_SIZE), Image.BICUBIC)
    x = ((np.asarray(art, dtype=np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def jitter_box(img, rng):
    """Random +-6% per-side box (clamped to the image), then shrunk about its centre to the card's aspect."""
    w, h = img.size
    x0 = max(0.0, rng.uniform(-JITTER, JITTER) * w)
    x1 = min(float(w), w - rng.uniform(-JITTER, JITTER) * w)
    y0 = max(0.0, rng.uniform(-JITTER, JITTER) * h)
    y1 = min(float(h), h - rng.uniform(-JITTER, JITTER) * h)
    bw, bh, ar = x1 - x0, y1 - y0, w / h
    if bw / bh > ar:
        nw = bh * ar
        x0, x1 = x0 + (bw - nw) / 2, x0 + (bw - nw) / 2 + nw
    else:
        nh = bw / ar
        y0, y1 = y0 + (bh - nh) / 2, y0 + (bh - nh) / 2 + nh
    return img.crop((round(x0), round(y0), max(round(x0) + 1, round(x1)), max(round(y0) + 1, round(y1))))


def degrade_view(big, h, filt, blur_sigma, quality):
    """Optional blur, resize to height h, JPEG round trip; returns an RGB PIL card."""
    src = big.filter(ImageFilter.GaussianBlur(blur_sigma)) if blur_sigma else big
    small = resize_to_height(src, h, FILTERS[filt])
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def make_pair(img, rng):
    """One jittered big image -> (large tensor [teacher + student drift view], degraded tensor, H, big)."""
    big = jitter_box(img, rng)
    large = to_tensor(resize_to_height(big, LARGE_H, Image.BOX))
    h = sample_height(rng)
    filt = rng.choice(list(FILTERS))
    sigma = rng.uniform(*BLUR_SIGMA) if rng.random() < BLUR_P else 0.0
    deg = to_tensor(degrade_view(big, h, filt, sigma, rng.randint(*JPEG_Q)))
    return large, deg, h, big


def eval_rows(cat):
    have = [i for i, cid in enumerate(cat.ids) if E.image_path(cid)]
    return have, random.Random(E.SEED).sample(have, E.N)


def build_split(cat, name_of, out_path=None):
    have, ev = eval_rows(cat)
    ev_names = {name_of[r] for r in ev if name_of[r]}
    ev_set = set(ev)
    excl_eval, excl_name, excl_digital, train = [], [], [], []
    for r in have:
        if r in ev_set:
            excl_eval.append(r)
        elif name_of[r] and name_of[r] in ev_names:
            excl_name.append(r)
        elif is_digital(cat.ids[r]):
            excl_digital.append(r)
        else:
            train.append(r)
    assert not (set(train) & ev_set), "training and eval ids intersect"
    assert not any(name_of[r] in ev_names for r in train if name_of[r]), "same-name card in training"
    split = {
        "digital_rule": DIGITAL_RE.pattern,
        "counts": {"with_image": len(have), "eval": len(excl_eval), "excluded_same_name": len(excl_name),
                   "excluded_digital": len(excl_digital), "train": len(train),
                   "eval_digital": sum(is_digital(cat.ids[r]) for r in ev)},
        "excluded_eval_ids": [cat.ids[r] for r in excl_eval],
        "excluded_same_name_ids": [cat.ids[r] for r in excl_name],
        "excluded_digital_ids": [cat.ids[r] for r in excl_digital],
        "train_ids": [cat.ids[r] for r in train],
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split, f)
    return split, train, ev


class PairDataset(torch.utils.data.Dataset):
    """Item = (row, sample_seed) so persistent workers stay deterministic across epochs."""

    def __init__(self, paths_by_row):
        self.paths = paths_by_row

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        row, seed = item
        with Image.open(self.paths[row]) as im:
            img = im.convert("RGB")
        large, deg, h, _ = make_pair(img, random.Random(seed))
        return large, deg, row, h


class EpochBatches:
    """Batch sampler: fresh seeded permutation per epoch, drop-last, items carry their own seeds."""

    def __init__(self, rows, batch, seed):
        self.rows, self.batch, self.seed, self.epoch = list(rows), batch, seed, 0

    def __len__(self):
        return len(self.rows) // self.batch

    def __iter__(self):
        order = list(self.rows)
        random.Random(f"{self.seed}-perm-{self.epoch}").shuffle(order)
        for b in range(len(self)):
            chunk = order[b * self.batch:(b + 1) * self.batch]
            yield [(r, f"{self.seed}-{self.epoch}-{b * self.batch + k}") for k, r in enumerate(chunk)]
