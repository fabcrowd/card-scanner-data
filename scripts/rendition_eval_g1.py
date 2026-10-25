"""G1 vs E vs shipped on the REAL CDN renditions (harvest TRAIN split, never trained on by any of these models).

Same box mapping, crop path, truth and top-5/classify as scripts/rendition_eval.py (imported helpers; real kind only).
Writes train/out/fastG1_all/rendition-records.json and rendition-g1-vs-e.json (per-bucket top1/top5 + listing-cluster
bootstrap of G1 minus E). Does not touch data/reports/rendition. Run: .venv-gpu/Scripts/python.exe -I scripts/rendition_eval_g1.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rendition_eval as RE  # noqa: E402
import run_real_photos as R  # noqa: E402
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402
from pseudo_label_v2 import group_name  # noqa: E402

OUT = os.path.join(ROOT, "train", "out", "fastG1_all")
MODELS = {"shipped": os.path.join(ROOT, "assets", "clip_visual_int8.onnx"),
          "E": os.path.join(ROOT, "train", "out", "fastE_all", "epoch16-ema-int8.onnx"),
          "G1": os.path.join(OUT, "epoch16-ema-int8.onnx")}
BUCKETS = [("60-99", 60, 100), ("100-149", 100, 150), ("150-199", 150, 200), ("200-299", 200, 300), (">=300", 300, 10**6),
           ("100-199", 100, 200)]
N_BOOT = 5000


def collect():
    H, REP = RE.H, RE.REP
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
    recs = []
    for site in RE.SITES:
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
                if g.get("file") and g["tag"] not in RE.SKIP_TAGS and not g.get("mismatch"):
                    with Image.open(os.path.join(H, "_renditions", site, g["file"])) as im:
                        variants.append((g["tag"], im.convert("RGB")))
            for r in rows:
                truth, b = group_name(r["name"]), r["box"]
                for tag, vimg in variants:
                    sx, sy = vimg.width / orig.width, vimg.height / orig.height
                    vb = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                    h = vb[3] - vb[1]
                    data = R.to_png(R.crop_card(vimg, vb))
                    for mname, emb in embs.items():
                        idx, sims = cat.top_k_batch(emb(data, trusted=True)[None], k=R.FETCH_K)
                        pairs = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if not is_dig[j]][:R.TOP]
                        cls = classify([s for _, s in pairs], [cat.ids[j] for j, _ in pairs], [name_of[j] for j, _ in pairs], int(round(h)))
                        names5 = [gname_of[j] for j, _ in pairs]
                        recs.append({"site": site, "listing": m["listing"], "photo": m["photo"], "tag": tag, "model": mname,
                                     "h": round(h, 1), "top1": bool(names5 and names5[0] == truth), "top5": truth in names5,
                                     "top1_id": cat.ids[pairs[0][0]] if pairs else None, "accepted": cls.accepted,
                                     "label": cls.label, "box": b})
        print(site, "done", len(recs), flush=True)
    json.dump(recs, open(os.path.join(OUT, "rendition-records.json"), "w"))
    return recs


def analyse(recs):
    pr = defaultdict(dict)
    for r in recs:
        if r["tag"] != "orig":
            pr[(r["site"], r["listing"], r["photo"], tuple(r["box"]), r["tag"])][r["model"]] = r
    clusters = defaultdict(list)
    for k, v in pr.items():
        clusters[(k[0], k[1])].append(v)  # cluster = listing
    keys = list(clusters)
    rng = np.random.default_rng(20261023)
    idx = rng.integers(0, len(keys), (N_BOOT, len(keys)))
    out = {"n_clusters": len(keys), "n_boot": N_BOOT, "buckets": {}}
    diffs = {"G1-E top1": ("G1", "E", "top1"), "G1-E top5": ("G1", "E", "top5"), "G1-shipped top1": ("G1", "shipped", "top1"),
             "E-shipped top1": ("E", "shipped", "top1")}
    for name, lo, hi in BUCKETS:
        sel = [[v for v in clusters[k] if lo <= v["E"]["h"] < hi] for k in keys]  # height is model-independent
        n = np.array([len(s) for s in sel], float)
        cell = {"n": int(n.sum()), "n_listings_nonempty": int((n > 0).sum())}
        for m in MODELS:
            vs = [v for s in sel for v in s]
            ids = Counter(v[m]["top1_id"] for v in vs)
            wrong = Counter(v[m]["top1_id"] for v in vs if not v[m]["top1"])
            cell[m] = {"top1": sum(v[m]["top1"] for v in vs) / max(1, len(vs)), "top5": sum(v[m]["top5"] for v in vs) / max(1, len(vs)),
                       "hub_modal_top1_share": ids.most_common(1)[0][1] / len(vs) if vs else None,
                       "hub_modal_share_of_wrong": wrong.most_common(1)[0][1] / sum(wrong.values()) if wrong else None,
                       "sure_wrong_rate": sum(v[m]["label"] == "sure" and not v[m]["top1"] for v in vs) / max(1, len(vs)),
                       "accepted_wrong_rate": sum(v[m]["accepted"] and not v[m]["top1"] for v in vs) / max(1, len(vs))}
        for lab, (a, b, key) in diffs.items():
            d = np.array([sum(v[a][key] - v[b][key] for v in s) for s in sel], float)
            num, den = d[idx].sum(1), n[idx].sum(1)
            ok = den > 0
            bs = num[ok] / den[ok]
            cell[lab] = {"point": float(d.sum() / max(1, n.sum())), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        out["buckets"][name] = cell
    json.dump(out, open(os.path.join(OUT, "rendition-g1-vs-e.json"), "w"), indent=1)
    return out


def main():
    path = os.path.join(OUT, "rendition-records.json")
    recs = json.load(open(path)) if os.path.exists(path) else collect()
    out = analyse(recs)
    print("listings", out["n_clusters"])
    for name, c in out["buckets"].items():
        d = c["G1-E top1"]
        print(f"{name:8s} n={c['n']:4d} " + " ".join(f"{m} {c[m]['top1']:.3f}/{c[m]['top5']:.3f}" for m in MODELS)
              + f" | G1-E top1 {d['point']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
