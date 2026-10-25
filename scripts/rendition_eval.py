"""Rendition study: recognition of labelled boxes on the original vs real CDN renditions vs our synthetic degradation.

Truth = v3 pseudo-label NAME (geometric, >=300 px boxes on the original). Boxes are scaled onto each rendition.
Path = run_real_photos.crop_card (stretch 750x1050) -> ClipEmbedder (art crop) -> catalog top-5 -> fusion_rule.classify.
Synthetic = the same original box crop resized to the rendition's box size (random PIL filter), JPEG q40-85, then the same path.
Run: .venv-gpu/Scripts/python.exe -I scripts/rendition_eval.py
"""
import io
import json
import os
import random
import sys
import zlib
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_real_photos as R  # noqa: E402
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402
from pseudo_label_v2 import group_name  # noqa: E402

H = os.path.join(ROOT, "data", "harvest-20261022")
REP = os.path.join(ROOT, "data", "reports", "rendition")
MODELS = {"shipped": os.path.join(ROOT, "assets", "clip_visual_int8.onnx"),
          "F": os.path.join(ROOT, "train", "out", "fastC_real", "epoch8-ema-int8.onnx")}
SITES = ["mercari", "hibid", "shopgoodwill", "craigslist", "poshmark"]
FILTERS = [Image.BOX, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
SYNTH_REPEATS = 3
HBINS = [("<60", 0, 60), ("60-99", 60, 100), ("100-149", 100, 150), ("150-199", 150, 200), ("200-299", 200, 300), (">=300", 300, 10**6)]
SKIP_TAGS = {"50x50c"}  # cropped square thumbnail, boxes cannot be mapped


def hbin(h):
    return next(n for n, lo, hi in HBINS if lo <= h < hi)


def jpeg_roundtrip(img, rng):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(40, 85))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    nm = R.load_names()
    name_of = [nm.get(c) for c in cat.ids]
    gname_of = [group_name(n) if n else None for n in name_of]
    is_dig = np.array([bool(R.DIGITAL_RE.match(c)) for c in cat.ids])
    embs = {k: ClipEmbedder(p) for k, p in MODELS.items()}
    labels = defaultdict(list)
    for line in open(os.path.join(REP, "labels_all.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["level"] in ("id", "name"):
            labels[(r["site"], r["listing"], r["photo_n"])].append(r)

    def read(emb, img, box, h):
        q = emb(R.to_png(R.crop_card(img, box)), trusted=True)
        idx, sims = cat.top_k_batch(q[None], k=R.FETCH_K)
        pairs = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if not is_dig[j]][:R.TOP]
        cls = classify([s for _, s in pairs], [cat.ids[j] for j, _ in pairs], [name_of[j] for j, _ in pairs], int(round(h)))
        return pairs, cls

    recs = []
    for site in SITES:
        mpath = os.path.join(H, "_renditions", site, "rendition-manifest.json")
        if not os.path.exists(mpath):
            continue
        for m in json.load(open(mpath, encoding="utf-8")):
            rows = labels.get((site, m["listing"], m["photo"]), [])
            if not rows:
                continue
            with Image.open(os.path.join(H, site, "train", m["listing"], f"photo-{m['photo']}.jpg")) as im:
                orig = im.convert("RGB")
            variants = [("orig", orig)]
            for g in m["renditions"]:
                if g.get("file") and g["tag"] not in SKIP_TAGS and not g.get("mismatch"):
                    with Image.open(os.path.join(H, "_renditions", site, g["file"])) as im:
                        variants.append((g["tag"], im.convert("RGB")))
            for r in rows:
                truth = group_name(r["name"])
                b = r["box"]
                for tag, vimg in variants:
                    sx, sy = vimg.width / orig.width, vimg.height / orig.height
                    vb = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                    h = vb[3] - vb[1]
                    kinds = [("real", vimg, vb)]
                    if tag != "orig":
                        tw, th = max(8, round((b[2] - b[0]) * sx)), max(8, round(h))
                        crop = orig.crop(tuple(round(v) for v in b))
                        for rep in range(SYNTH_REPEATS):
                            rng = random.Random(zlib.crc32(f"{site}|{m['listing']}|{m['photo']}|{b[0]}|{tag}|{rep}".encode()))
                            small = jpeg_roundtrip(crop.resize((tw, th), rng.choice(FILTERS)), rng)
                            kinds.append((f"synth{rep}", small, [0, 0, small.width, small.height]))
                    for kind, kimg, kb in kinds:
                        for mname, emb in embs.items():
                            pairs, cls = read(emb, kimg, kb, h)
                            names5 = [gname_of[j] for j, _ in pairs]
                            recs.append({"site": site, "listing": m["listing"], "photo": m["photo"], "tag": tag, "kind": kind[:5], "model": mname,
                                         "vw": vimg.width, "h": round(h, 1), "top1": bool(names5 and names5[0] == truth),
                                         "top5": truth in names5, "accepted": cls.accepted, "label": cls.label,
                                         "cos1": round(pairs[0][1], 4) if pairs else 0.0, "box": b})
        print(site, "done", len(recs), flush=True)
    with open(os.path.join(REP, "eval-records.json"), "w", encoding="utf-8") as fh:
        json.dump(recs, fh)
    summarize(recs)


def agg(rs):
    n = len(rs)
    if not n:
        return None
    acc = [r for r in rs if r["accepted"]]
    return {"n": n, "median_h": float(np.median([r["h"] for r in rs])), "top1": sum(r["top1"] for r in rs) / n,
            "top5": sum(r["top5"] for r in rs) / n, "accepted": len(acc) / n,
            "acc_wrong_of_all": sum(not r["top1"] for r in acc) / n,
            "sure_wrong_of_all": sum(r["label"] == "sure" and not r["top1"] for r in rs) / n}


def summarize(recs):
    out = {}
    for model in MODELS:
        for kind in ("real", "synth"):
            groups = defaultdict(list)
            for r in recs:
                if r["model"] == model and r["kind"] == kind:
                    groups[(r["site"], r["tag"])].append(r)
            out[f"{model}/{kind}/site_tag"] = {f"{s}|{t}": {**agg(v), "median_w": float(np.median([x["vw"] for x in v]))} for (s, t), v in groups.items()}
            bins = defaultdict(list)
            for r in recs:
                if r["model"] == model and r["kind"] == kind and r["tag"] != "orig":
                    bins[hbin(r["h"])].append(r)
            out[f"{model}/{kind}/height_bin_renditions"] = {k: agg(bins[k]) for k, _, _ in HBINS if bins[k]}
    with open(os.path.join(REP, "eval-summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        print("==", k)
        for kk, vv in v.items():
            print(f"{kk:24s} n={vv['n']:4d} medH={vv['median_h']:6.0f} top1={vv['top1']:.2f} top5={vv['top5']:.2f} acc={vv['accepted']:.2f} accWrong={vv['acc_wrong_of_all']:.3f}")


if __name__ == "__main__":
    main()
