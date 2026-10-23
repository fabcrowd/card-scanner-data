"""Pseudo-label cards in sharp real listing photos by matching full-res crops to catalog ARTWORK.

Per photo: shipped detector (detect_cards.py) -> keep boxes >= MIN_HEIGHT px in ORIGINAL pixels -> crop at full
resolution and stretch to 750x1050 (run_real_photos.crop_card) -> shipped int8 CLIP art-crop embedding (batch 1) ->
top-10 physical catalog candidates by cosine -> geometric verification with the project's own verifier
(auto card lister/tools/vellum-ai/vellum_ai/local_verify.py: SIFT 1500 + CLAHE at 400x560, ratio 0.75, RANSAC 8px,
inlier count), loaded by file path so its code is reused unchanged.

Label rule (high precision, coverage secondary):
  ACCEPT   best inliers >= 40 and best >= 2 x runner-up (best is in the CLIP top 10 by construction).
  NAME-ONLY  every candidate with inliers >= 40 shares one normalised name and there are >= 2 of them (reprints share
           art): accepted at name level, card_id null.
  else REJECT.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detect_cards as D  # noqa: E402
import run_real_photos as R  # noqa: E402
from score_real_labels import normalise  # noqa: E402

LOCAL_VERIFY = r"C:\repos\projects\auto card lister\tools\vellum-ai\vellum_ai\local_verify.py"
IMAGES_DIR = os.path.join(R.CARD_DIR, "images")
MIN_HEIGHT = 300
CLIP_TOP = 10
FETCH_K = 40
MIN_INLIERS = 40
LEAD_FACTOR = 2.0
SEED = 7
MAX_WORKERS = 8
PHOTO_EXT = R.PHOTO_EXT
OUT_DIR = os.path.join(ROOT, "data", "pseudo-labels")

_ctx = {}
Image.MAX_IMAGE_PIXELS = None


def _load_verifier():
    spec = importlib.util.spec_from_file_location("local_verify_ext", LOCAL_VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _one_thread_session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def init_ctx(model=None, catalog=None, detector=D.DEFAULT_MODEL, threads_one=True):
    """Load models into module state (once per process)."""
    from pipeline.catalog import Catalog
    from pipeline.embed import ClipEmbedder

    model = model or os.path.join(ROOT, "assets", "clip_visual_int8.onnx")
    catalog = catalog or os.path.join(ROOT, "assets", "catalog.bin.gz")
    cat = Catalog.load(catalog, os.path.join(ROOT, "assets", "id_map.json"))
    emb = ClipEmbedder.__new__(ClipEmbedder)
    emb._session = _one_thread_session(model) if threads_one else None
    if emb._session is None:
        emb = ClipEmbedder(model)
    emb._input = emb._session.get_inputs()[0].name
    if threads_one:
        D._sessions[detector] = _one_thread_session(detector)
    Image.MAX_IMAGE_PIXELS = None  # pipeline.embed lowers it for untrusted crops; these are our own photos
    nm = R.load_names()
    _ctx.update(cat=cat, emb=emb, name_of=nm, detector=detector, verify=_load_verifier(),
                is_dig=np.array([bool(R.DIGITAL_RE.match(c)) for c in cat.ids]))


def decide(cands):
    """cands: list of dicts with id, name, inliers (any order). Returns (accepted, name_only, best, runner_up)."""
    ranked = sorted(cands, key=lambda c: (c["inliers"], c["cos"]), reverse=True)
    best = ranked[0]["inliers"]
    second = ranked[1]["inliers"] if len(ranked) > 1 else 0
    strong = [c for c in ranked if c["inliers"] >= MIN_INLIERS]
    if len(strong) >= 2 and len({normalise(c["name"] or c["id"]) for c in strong}) == 1:
        return True, True, best, second
    if best >= MIN_INLIERS and best >= LEAD_FACTOR * second:
        return True, False, best, second
    return False, False, best, second


def label_box(img, box):
    """Label one box of an RGB PIL image. Returns the output record (without photo path)."""
    cat, emb, name_of, V = _ctx["cat"], _ctx["emb"], _ctx["name_of"], _ctx["verify"]
    crop = R.crop_card(img, box)
    q = emb(R.to_png(crop), trusted=True)
    idx, sims = cat.top_k_batch(q[None], k=FETCH_K)
    pairs = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if not _ctx["is_dig"][j]][:CLIP_TOP]
    bgr = np.asarray(crop)[:, :, ::-1].copy()
    query = V._features(bgr)
    cands = []
    for rank, (j, s) in enumerate(pairs, 1):
        cid = cat.ids[j]
        inl = V._inliers(query, V._catalog_features(IMAGES_DIR, cid))
        cands.append({"id": cid, "name": name_of.get(cid), "clip_rank": rank, "cos": round(s, 4), "inliers": int(inl)})
    accepted, name_only, best, second = decide(cands) if cands else (False, False, 0, 0)
    ranked = sorted(cands, key=lambda c: (c["inliers"], c["cos"]), reverse=True)
    top = ranked[0] if ranked else None
    return {
        "box": [round(float(v), 1) for v in box], "box_height": round(float(box[3] - box[1]), 1),
        "accepted": accepted, "name_only": name_only,
        "card_id": top["id"] if accepted and not name_only else None,
        "name": top["name"] if accepted else None,
        "clip_rank": top["clip_rank"] if accepted and not name_only else None,
        "equals_clip_top1": bool(accepted and not name_only and top["clip_rank"] == 1),
        "top1_cos": cands[0]["cos"] if cands else None,
        "inliers_best": best, "inliers_runner_up": second,
        "top3": ranked[:3],
        "clip_top1_id": cands[0]["id"] if cands else None,
    }


def process_image(img, photo):
    rows = []
    for d in D.detect_cards(img, _ctx["detector"]):
        b = d["box"]
        if b[3] - b[1] < MIN_HEIGHT:
            continue
        r = label_box(img, b)
        r["photo"] = photo
        r["det_score"] = round(d["score"], 3)
        rows.append(r)
    return rows


def _work(path):
    t = time.time()
    with Image.open(path) as im:
        img = im.convert("RGB")
    n_all = len(D.detect_cards(img, _ctx["detector"]))
    rows = process_image(img, path)
    return path, n_all, rows, time.time() - t


def list_photos(root):
    out = []
    for dp, _, fs in os.walk(root):
        out += [os.path.join(dp, f) for f in fs if f.lower().endswith(PHOTO_EXT)]
    return sorted(out)


def dist(vals):
    if not vals:
        return "n=0"
    a = np.array(vals)
    return f"n={len(a)} min={a.min():.0f} p25={np.percentile(a, 25):.0f} med={np.median(a):.0f} p75={np.percentile(a, 75):.0f} max={a.max():.0f}"


def summarize(rows, n_photos, n_all, secs):
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]
    idl = [r for r in acc if not r["name_only"]]
    lines = [f"photos {n_photos}; detected boxes (all heights) {n_all}; boxes >= {MIN_HEIGHT}px {len(rows)}",
             f"accepted {len(acc)} (id-level {len(idl)}, name_only {sum(r['name_only'] for r in acc)}); rejected {len(rej)}",
             f"inliers best  accepted: {dist([r['inliers_best'] for r in acc])}",
             f"inliers best  rejected: {dist([r['inliers_best'] for r in rej])}",
             f"inliers 2nd   accepted: {dist([r['inliers_runner_up'] for r in acc])}",
             f"inliers 2nd   rejected: {dist([r['inliers_runner_up'] for r in rej])}",
             f"accepted id-level label == CLIP top-1: {sum(r['equals_clip_top1'] for r in idl)}/{len(idl)}",
             f"runtime {secs:.0f}s total, {secs / max(n_photos, 1):.1f}s/photo (wall, {MAX_WORKERS} workers)"]
    return "\n".join(lines)


def _tile_font():
    return R._font(13)


def _catalog_image(card_id):
    for ext in (".webp", ".png"):
        p = os.path.join(IMAGES_DIR, f"{card_id}{ext}")
        if os.path.exists(p):
            with Image.open(p) as im:
                return im.convert("RGB")
    return None


def _tile(r, tw=200, th=280):
    with Image.open(r["photo"]) as im:
        crop = R.crop_card(im.convert("RGB"), r["box"]).resize((tw, th), Image.BILINEAR)
    ref_id = r["card_id"] if r["accepted"] and not r["name_only"] else (r["top3"][0]["id"] if r["accepted"] else r["clip_top1_id"])
    ref = _catalog_image(ref_id) if ref_id else None
    tile = Image.new("RGB", (2 * tw, th + 44), (255, 255, 255))
    tile.paste(crop, (0, 0))
    if ref is not None:
        tile.paste(ref.resize((tw, th), Image.BILINEAR), (tw, 0))
    status = ("NAME-ONLY" if r["name_only"] else "ACCEPT") if r["accepted"] else "REJECT"
    ref_name = r["name"] if r["accepted"] else next((c["name"] for c in r["top3"] if c["id"] == ref_id), "")
    dr = ImageDraw.Draw(tile)
    dr.text((3, th + 2), f"{status} {ref_id} inl {r['inliers_best']}/{r['inliers_runner_up']}", fill=(0, 0, 0), font=_tile_font())
    dr.text((3, th + 20), f"{(ref_name or '')[:48]} h={r['box_height']:.0f}", fill=(0, 0, 0), font=_tile_font())
    dr.rectangle([0, 0, 2 * tw - 1, th + 43], outline=(0, 160, 0) if r["accepted"] else (200, 0, 0))
    return tile


def build_audit(rows, out_dir, n_acc=40, n_rej=20, per_sheet=15, cols=3):
    rnd = random.Random(SEED)
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]
    pick = rnd.sample(acc, min(n_acc, len(acc))) + rnd.sample(rej, min(n_rej, len(rej)))
    rnd.shuffle(pick)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in range(0, len(pick), per_sheet):
        chunk = pick[s:s + per_sheet]
        tiles = [_tile(r) for r in chunk]
        tw, th = tiles[0].size
        nrows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, nrows * th), (255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
        p = os.path.join(out_dir, f"sheet-{s // per_sheet + 1}.jpg")
        sheet.save(p, "JPEG", quality=90)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default=os.path.join(ROOT, "data", "goodwill-pack"))
    ap.add_argument("--out", default=None, help="jsonl path (default data/pseudo-labels/<photos dir name>.jsonl)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(OUT_DIR, os.path.basename(os.path.normpath(a.photos)) + ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    photos = list_photos(a.photos)
    t0 = time.time()
    rows, n_all = [], 0
    workers = max(1, min(a.workers, MAX_WORKERS))
    with mp.Pool(workers, initializer=init_ctx) as pool:
        for k, (path, n, r, dt) in enumerate(pool.imap(_work, photos), 1):
            rows += r
            n_all += n
            print(f"[{k}/{len(photos)}] {os.path.relpath(path, a.photos)} boxes>=300: {len(r)} acc {sum(x['accepted'] for x in r)} {dt:.1f}s", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"photo": r["photo"], **{k: v for k, v in r.items() if k != "photo"}}) + "\n")
    secs = time.time() - t0
    print(summarize(rows, len(photos), n_all, secs))
    sheets = build_audit(rows, os.path.join(os.path.dirname(out), "audit"))
    print("audit sheets:", *sheets, sep="\n  ")
    print("jsonl:", out)


if __name__ == "__main__":
    mp.freeze_support()
    main()
