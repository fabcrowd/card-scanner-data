"""Synthetic small-blurry-crop baseline against the frozen catalog. Synthetic only: NOT a ship gate."""
import io
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402

IMG_DIR = r"C:\repos\projects\auto card lister\data\card-catalog\images"
N = 1000
SEED = 9001
HEIGHTS = [60, 90, 120, 150, 200]
FILTERS = {"area": Image.BOX, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def image_path(card_id):
    for ext in (".webp", ".png"):
        p = os.path.join(IMG_DIR, card_id + ext)
        if os.path.exists(p):
            return p
    return None


def degrade(img, h, rng):
    """Resize to height h (random filter), JPEG at random quality 40..85; returns JPEG bytes."""
    w = max(1, round(img.width * h / img.height))
    small = img.resize((w, h), FILTERS[rng.choice(list(FILTERS))])
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=rng.randint(40, 85))
    return buf.getvalue()


def main():
    t0 = time.time()
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    emb = ClipEmbedder(os.path.join(ROOT, "assets", "clip_visual_int8.onnx"))
    have = [i for i, cid in enumerate(cat.ids) if image_path(cid)]
    rows = random.Random(SEED).sample(have, N)
    rng = random.Random(SEED + 1)
    fulls = []
    for r in rows:
        with Image.open(image_path(cat.ids[r])) as im:
            fulls.append(im.convert("RGB"))
    results = []
    print("SYNTHETIC baseline (resize + JPEG degradation of clean catalog art); NOT a ship gate.")
    print("height |    n | top-1 | top-5 | median cos to true row")
    for h in HEIGHTS:
        fresh = np.stack([emb(degrade(img, h, rng), trusted=True) for img in fulls])
        idx, _ = cat.top_k_batch(fresh, k=5)
        truth = np.array(rows)[:, None]
        top1 = float((idx[:, 0] == truth[:, 0]).mean())
        top5 = float((idx == truth).any(axis=1).mean())
        med = float(np.median(np.einsum("ij,ij->i", fresh, cat.vectors[rows])))
        results.append({"height": h, "n": N, "top1": top1, "top5": top5, "median_cos": med})
        print(f"{h:6d} | {N:4d} | {top1:.3f} | {top5:.3f} | {med:.4f}")
    out = os.path.join(ROOT, "data", "reports", "synth-baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "seed": SEED, "results": results}, f, indent=2)
    print(f"saved {out}; name-level accuracy skipped (id_map has ids only); runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
