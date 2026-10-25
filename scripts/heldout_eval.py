"""Held-out-set (H = newest sets) evaluation for fastEho experiment. CPU only. Tunes nothing.

Stages (all write to train/out/ho/):
  synth  <label>=<onnx> ...  per-card synthetic reads on H cards at heights 90/120/150/200 (eval_synth_small.degrade, fixed
                             seed, JPEGs cached once and shared by every model) and large crops (full/400/300, large_crop_check.degrade)
  xs     <label>=<onnx> ...  cross-source eval (cross_source_eval.main with MODELS/OUT redirected; data/reports untouched)
"""
import json
import os
import pickle
import random
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_synth_small as E  # noqa: E402
import large_crop_check as LC  # noqa: E402
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402
from run_real_photos import load_names  # noqa: E402

HO = os.path.join(ROOT, "train", "out", "ho")
HEIGHTS = [90, 120, 150, 200]
LARGE = [None, 400, 300]
SEED = 777


def load():
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    nm = load_names()
    ids = json.load(open(os.path.join(HO, "heldout.json"), encoding="utf-8"))["ids"]
    row_of = {c: i for i, c in enumerate(cat.ids)}
    return cat, [nm.get(c) for c in cat.ids], ids, [row_of[c] for c in ids]


def make_inputs(cat, rows):
    path = os.path.join(HO, "h_inputs.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    rng = random.Random(SEED)
    fulls = []
    for r in rows:
        with Image.open(E.image_path(cat.ids[r])) as im:
            fulls.append(im.convert("RGB"))
    inp = {"synth": {h: [E.degrade(img, h, rng) for img in fulls] for h in HEIGHTS}, "large": {}}
    for h in LARGE:
        inp["large"][h] = [LC.png_bytes_of(E.image_path(cat.ids[r])) if h is None else LC.degrade(img, h)
                           for r, img in zip(rows, fulls)]
    inp["full_h"] = [img.height for img in fulls]
    pickle.dump(inp, open(path, "wb"))
    return inp


def reads(emb, cat, name_of, datas, hs):
    out = []
    qs = np.stack([emb(d, trusted=True) for d in datas])
    idx, sims = cat.top_k_batch(qs, k=5)
    for q in range(len(datas)):
        ids = [cat.ids[j] for j in idx[q]]
        nms = [name_of[j] for j in idx[q]]
        c = classify([float(x) for x in sims[q]], ids, nms, hs[q])
        out.append({"ids": ids, "names": nms, "cos": [round(float(x), 4) for x in sims[q]], "label": c.label,
                    "accepted": bool(c.accepted), "chosen": c.id})
    return out


def stage_synth(models):
    cat, name_of, ids, rows = load()
    inp = make_inputs(cat, rows)
    for label, path in models.items():
        emb = ClipEmbedder(path)
        res = {"synth": {}, "large": {}}
        for h in HEIGHTS:
            res["synth"][h] = reads(emb, cat, name_of, inp["synth"][h], [h] * len(ids))
        for h in LARGE:
            res["large"][str(h or "full")] = reads(emb, cat, name_of, inp["large"][h], [h or fh for fh in inp["full_h"]])
        json.dump(res, open(os.path.join(HO, f"synth_{label}.json"), "w"))
        print("synth done", label, flush=True)


def stage_xs(models):
    import cross_source_eval as X

    X.MODELS = models
    X.OUT = os.path.join(HO, "xs")
    X.main()


if __name__ == "__main__":
    stage, models = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    {"synth": stage_synth, "xs": stage_xs}[stage](models)
