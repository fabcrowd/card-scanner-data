"""Second synthetic eval (generator 2) for the same 1000 held-out cards, plus eval1/eval2/real realism measures.

Generator 2 is deliberately different from training (cv2 resize, motion/defocus blur at source resolution,
over-sharpening, ISO noise, tint, glare, double JPEG with block misalignment). Evaluation only: do not tune on it.
"""
import io
import json
import math
import os
import pickle
import random
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pilot_eval as P  # noqa: E402
from pipeline.embed import art_rect  # noqa: E402

D, E = P.D, P.E
OUT = P.OUT
SEED2 = 9003
INTERPS = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
P_BLUR, P_SHARPEN, P_NOISE, P_TINT, P_GLARE = 0.35, 0.4, 0.7, 0.3, 0.2
REAL_RESULTS = os.path.join(ROOT, "data", "reports", "real-baseline-shipped", "results.json")
REAL_PHOTOS = os.path.join(ROOT, "data", "fb-eval-pack", "user-saved")
BUCKETS = [("<100", 0, 100, 80), ("100-149", 100, 150, 125), ("150-199", 150, 200, 175),
           ("200-299", 200, 300, 250), (">=300", 300, 10 ** 9, 350)]


def _motion_kernel(rng, nprng):
    n = rng.randint(2, 5)
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1
    ang = nprng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), ang, 1.0)
    k = cv2.warpAffine(k, rot, (n, n))
    return k / max(k.sum(), 1e-6)


def _disk_kernel(nprng):
    r = nprng.uniform(0.5, 1.5)
    n = int(math.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    k = (np.sqrt(xx ** 2 + yy ** 2) <= r).astype(np.float32)
    return k / k.sum()


def _glare(arr, rng, nprng):
    """Blurred bright ellipse over 5-25% of the art square, added at 20-60% strength."""
    h, w = arr.shape[:2]
    sx, sy, side = art_rect(w, h)
    cover = rng.uniform(0.05, 0.25)
    ratio = nprng.uniform(0.5, 2.0)
    area = cover * side * side
    a = math.sqrt(area / math.pi * ratio)  # semi-axes with pi*a*b = area
    b = area / (math.pi * a)
    cx, cy = sx + nprng.uniform(0, side), sy + nprng.uniform(0, side)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (int(cx), int(cy)), (max(1, int(a)), max(1, int(b))), nprng.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, 0.25 * min(a, b)))
    strength = rng.uniform(0.2, 0.6)
    return arr + strength * 255.0 * mask[:, :, None]


def _shift(arr, dx, dy):
    if not (dx or dy):
        return arr
    padded = np.pad(arr, ((dy, 0), (dx, 0), (0, 0)), mode="edge")
    return padded[:arr.shape[0], :arr.shape[1]]


def _jpeg(arr, quality, subsampling=None):
    buf = io.BytesIO()
    kw = {"subsampling": subsampling} if subsampling is not None else {}
    Image.fromarray(arr).save(buf, "JPEG", quality=quality, **kw)
    return buf.getvalue()


def degrade2(img, h, rng, nprng):
    """img: PIL RGB catalog card. Returns JPEG bytes of the card at height h (generator 2)."""
    src = np.asarray(img, dtype=np.uint8)
    interp = rng.choice(INTERPS)
    if rng.random() < P_BLUR:
        kern = _motion_kernel(rng, nprng) if rng.random() < 0.5 else _disk_kernel(nprng)
        src = cv2.filter2D(src, -1, kern, borderType=cv2.BORDER_REPLICATE)
    w = max(1, round(img.width * h / img.height))
    x = cv2.resize(src, (w, h), interpolation=interp).astype(np.float32)
    if rng.random() < P_SHARPEN:
        amount = rng.uniform(0.3, 1.2)
        x = x + amount * (x - cv2.GaussianBlur(x, (0, 0), 1.0))
    if rng.random() < P_NOISE:
        luma = nprng.normal(0, rng.uniform(2, 10), x.shape[:2]).astype(np.float32)[:, :, None]
        chroma = nprng.normal(0, rng.uniform(1, 5), x.shape).astype(np.float32)
        x = x + luma + chroma
    if rng.random() < P_TINT:
        cast = np.array([1 + rng.uniform(-0.08, 0.0), 1 + rng.uniform(-0.04, 0.04), 1 + rng.uniform(0.0, 0.08)], np.float32)
        x = x * cast
    if rng.random() < P_GLARE:
        x = _glare(x, rng, nprng)
    x = np.clip(x, 0, 255).astype(np.uint8)
    first = np.asarray(Image.open(io.BytesIO(_jpeg(x, rng.randint(85, 95)))).convert("RGB"))
    shifted = _shift(first, rng.randint(0, 7), rng.randint(0, 7))
    return _jpeg(shifted, rng.randint(40, 85), subsampling=2)


def build_evalset2(seed=SEED2, rows=None, cat=None):
    """{height: [jpeg bytes per eval card]} for the pilot's 1000 held-out cards; cached at train/out/evalset2.pkl."""
    cache = os.path.join(OUT, "evalset2.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    if rows is None:
        cat = cat or P.Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
        _, rows = D.eval_rows(cat)
    cat = cat or P.Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    rng, nprng = random.Random(seed), np.random.default_rng(seed)
    fulls = []
    for r in rows:
        with Image.open(E.image_path(cat.ids[r])) as im:
            fulls.append(im.convert("RGB"))
    jp = {h: [degrade2(img, h, rng, nprng) for img in fulls] for h in E.HEIGHTS}
    os.makedirs(OUT, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(jp, f)
    return jp


def make_session(path):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def eval_set(onnx_path, ctx, jpegs):
    """Pilot metrics per height (batch 1) on the given eval set; also returns raw top-1 sims and fresh embeddings."""
    sess = make_session(onnx_path)
    heights, top1sim = {}, {}
    allm = np.ones(len(ctx.rows), dtype=bool)
    for h in E.HEIGHTS:
        fresh = P.embed_jpegs(sess, jpegs[h])
        heights[h] = {"all": P._height_stats(ctx, fresh, allm), "nodigital": P._height_stats(ctx, fresh, ~ctx.digital),
                      "median_cos": float(np.median(np.einsum("ij,ij->i", fresh, ctx.cat.vectors[ctx.rows])))}
        _, sims = ctx.cat.top_k_batch(fresh, k=1)
        top1sim[h] = float(np.median(sims[:, 0]))
        heights[h]["median_top1_sim"] = top1sim[h]
    return {"heights": heights}


def art224(card):
    sx, sy, side = art_rect(*card.size)
    return card.crop((sx, sy, sx + side, sy + side)).resize((224, 224), Image.BICUBIC)


def lapvar(img224):
    return float(cv2.Laplacian(cv2.cvtColor(np.asarray(img224), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def synth_lapvar(jpegs):
    return {h: float(np.median([lapvar(art224(Image.open(io.BytesIO(b)).convert("RGB"))) for b in jpegs[h]])) for h in jpegs}


def real_measures():
    """Per bucket: n, median top-1 cosine (stored in results.json) and median Laplacian var of the art crop as the
    real pipeline builds it (box crop stretched to 750x1050 bilinear, then art crop)."""
    with open(REAL_RESULTS, encoding="utf-8") as f:
        boxes = json.load(f)["boxes"]
    cache = {}
    per = {b[0]: {"sim": [], "lap": []} for b in BUCKETS}
    for r in boxes:
        label = next(b[0] for b in BUCKETS if b[1] <= r["height"] < b[2])
        if r["photo"] not in cache:
            with Image.open(os.path.join(REAL_PHOTOS, r["photo"])) as im:
                cache = {r["photo"]: im.convert("RGB")}
        img, box = cache[r["photo"]], r["box"]
        x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
        card = img.crop((x1, y1, round(box[2]), round(box[3]))).resize((750, 1050), Image.BILINEAR)
        per[label]["sim"].append(r["top5"][0]["cos"])
        per[label]["lap"].append(lapvar(art224(card)))
    return {k: {"n": len(v["sim"]), "median_top1_sim": float(np.median(v["sim"])) if v["sim"] else None,
                "median_lapvar": float(np.median(v["lap"])) if v["lap"] else None} for k, v in per.items()}


def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    ctx = P.Context()
    jp1 = ctx.jpegs
    jp2 = build_evalset2(rows=ctx.rows, cat=ctx.cat)
    print("evalset2 ready", {h: len(v) for h, v in jp2.items()}, flush=True)
    base2 = eval_set(P.SHIPPED, ctx, jp2)
    _w(os.path.join(OUT, "baseline2", "metrics.json"), base2)
    base1 = eval_set(P.SHIPPED, ctx, jp1)  # eval1 incl. median top-1 sim for realism (same numbers as baseline/metrics.json)
    _w(os.path.join(OUT, "baseline2", "eval1_recomputed.json"), base1)
    models = {}
    for arm in ("A", "B"):
        path = os.path.join(OUT, arm, "epoch4-int8.onnx")
        if os.path.exists(path):
            models[arm] = eval_set(path, ctx, jp2)
            _w(os.path.join(OUT, "baseline2", f"{arm}_epoch4_eval2.json"), models[arm])
        else:
            print("MISSING", path)
    real = real_measures()
    lap = {"eval1": synth_lapvar(jp1), "eval2": synth_lapvar(jp2), "real": real}
    _w(os.path.join(OUT, "baseline2", "realism.json"), {"base1": base1, "base2": base2, "real": real, "lap": {k: v for k, v in lap.items() if k != "real"}})

    g = lambda m, h, k: m["heights"][h]["all"][k]
    print("\nEVAL1 vs EVAL2 shipped: H | top1 e1/e2 | top5 e1/e2 | acc-WRONG e1/e2 | hub1 e1/e2 | hub5 e1/e2")
    for h in E.HEIGHTS:
        print(f"{h:4d} | {g(base1, h, 'top1'):.3f}/{g(base2, h, 'top1'):.3f} | {g(base1, h, 'top5'):.3f}/{g(base2, h, 'top5'):.3f} | "
              f"{g(base1, h, 'accepted_wrong')}/{g(base2, h, 'accepted_wrong')} (acc {g(base1, h, 'accepted')}/{g(base2, h, 'accepted')}) | "
              f"{g(base1, h, 'hub1_share'):.3f}/{g(base2, h, 'hub1_share'):.3f} | {g(base1, h, 'hub5_share'):.3f}/{g(base2, h, 'hub5_share'):.3f}")
    ep = {}
    for arm in ("A", "B"):
        with open(os.path.join(OUT, arm, "metrics.json"), encoding="utf-8") as f:
            ep[arm] = {int(k): v for k, v in json.load(f)["epochs"][-1]["heights"].items()}
    print("\nCOMPARE (shipped | A ep4 EMA | B ep4 EMA), eval1 then eval2")
    for h in E.HEIGHTS:
        row = [f"{h:4d}"]
        for lab, m1, m2 in (("shipped", base1["heights"][h]["all"], base2["heights"][h]["all"]),
                            *[(a, ep[a][h]["all"], models[a]["heights"][h]["all"] if a in models else None) for a in ("A", "B")]):
            row.append(f"{lab}: e1 t1 {m1['top1']:.3f} W {m1['accepted_wrong']} hub1 {m1['hub1_share']:.3f}"
                       + (f" | e2 t1 {m2['top1']:.3f} W {m2['accepted_wrong']} hub1 {m2['hub1_share']:.3f}" if m2 else " | e2 n/a"))
        print("  ".join(row))
    print("\nREALISM: H/bucket(mid) | median top-1 sim e1 | e2 | real (n) || Laplacian var e1 | e2 | real")
    for h in E.HEIGHTS:
        print(f"synth H={h}: sim {base1['heights'][h]['median_top1_sim']:.3f} | {base2['heights'][h]['median_top1_sim']:.3f}   lap {lap['eval1'][h]:.1f} | {lap['eval2'][h]:.1f}")
    for lab, _, _, mid in BUCKETS:
        r = real[lab]
        if r["n"]:
            print(f"real {lab} (mid {mid}) n={r['n']}: top-1 sim {r['median_top1_sim']:.3f}  lap {r['median_lapvar']:.1f}")


if __name__ == "__main__":
    main()
