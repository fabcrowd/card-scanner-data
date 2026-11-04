"""Checks that catalog.bin rows equal fresh int8-model embeddings of the catalog art (exact extension crop)."""
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402

IMG_DIR = r"C:\repos\projects\auto card lister\data\card-catalog\images"
N = 300
SEED = 9001


def image_path(card_id):
    for ext in (".webp", ".png"):
        p = os.path.join(IMG_DIR, card_id + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    t0 = time.time()
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    emb = ClipEmbedder(os.path.join(ROOT, "assets", "clip_visual_int8.onnx"))
    have = [i for i, cid in enumerate(cat.ids) if image_path(cid)]
    rows = random.Random(SEED).sample(have, N)
    fresh = []
    for r in rows:
        with open(image_path(cat.ids[r]), "rb") as f:
            fresh.append(emb(f.read(), trusted=True))
    fresh = np.stack(fresh)
    cos = np.einsum("ij,ij->i", fresh, cat.vectors[rows])
    nearest = (fresh @ cat.vectors.T).argmax(axis=1)
    self_match = float((nearest == np.array(rows)).mean())
    med = float(np.median(cos))
    print(f"cards with image: {len(have)} of {len(cat.ids)}; sampled {N}")
    print(f"cosine min={cos.min():.4f} p1={np.percentile(cos, 1):.4f} p5={np.percentile(cos, 5):.4f} median={med:.4f} mean={cos.mean():.4f}")
    print(f"top-1 self-match={self_match:.4f}  cos>=0.99: {(cos >= 0.99).mean():.4f}  cos>=0.98: {(cos >= 0.98).mean():.4f}")
    for k in np.argsort(cos)[:5]:
        print(f"worst: {cat.ids[rows[k]]} cos={cos[k]:.4f}")
    ok = med >= 0.99 and self_match >= 0.95
    print("PARITY OK" if ok else f"PARITY SUSPECT (median={med:.4f}, self-match={self_match:.4f})")
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
