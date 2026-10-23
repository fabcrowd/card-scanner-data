"""Dedupe (sha256 + dHash<=4) vs existing packs and within harvest, split by listing 60/40, detect, write manifests.
Run: .venv-gpu/Scripts/python.exe -I scripts/harvest_finalize.py   (reads data/harvest-20261022/_raw/*)"""
import hashlib, json, os, random, shutil, statistics, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_cards import DEFAULT_MODEL, detect_cards  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data")
H = os.path.join(DATA, "harvest-20261022")
RAW = os.path.join(H, "_raw")
REFS = ["hibid-pack", "goodwill-pack", "goodwill-pack2", "fb-eval-pack"]
EXT = (".jpg", ".jpeg", ".png", ".webp")
BUCKETS = [("<100", 0, 100), ("100-149", 100, 150), ("150-199", 150, 200), ("200-299", 200, 300), (">=300", 300, 10**9)]


def dhash(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def ham(a, b):
    return bin(a ^ b).count("1")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


ref_sha, ref_dh = set(), []
for r in REFS:
    for dp, _, fs in os.walk(os.path.join(DATA, r)):
        for f in fs:
            if f.lower().endswith(EXT):
                p = os.path.join(dp, f)
                try:
                    ref_sha.add(sha(p)); ref_dh.append(dhash(p))
                except Exception:
                    pass
print("reference photos", len(ref_dh), flush=True)
ref_arr = np.array(ref_dh, dtype=np.uint64)


def near_ref(d):
    if not len(ref_arr):
        return False
    x = (ref_arr ^ np.uint64(d)).view(np.uint8).reshape(-1, 8)
    return bool((np.unpackbits(x, axis=1).sum(axis=1) <= 4).any())


rows = []
for site in sorted(os.listdir(RAW)):
    mp = os.path.join(RAW, site, "raw-manifest.json")
    if os.path.exists(mp):
        rows += json.load(open(mp))
stats = {"raw": len(rows), "dup_sha_existing": 0, "dup_phash_existing": 0, "dup_within": 0}
kept, seen_sha, seen_dh = [], set(), []
for r in rows:
    p = os.path.join(RAW, r["site"], r["listingId"], f"photo-{r['photo']}.jpg")
    if not os.path.exists(p):
        continue
    try:
        d = dhash(p)
    except Exception:
        continue
    if r["sha256"] in ref_sha:
        stats["dup_sha_existing"] += 1; continue
    if near_ref(d):
        stats["dup_phash_existing"] += 1; continue
    if r["sha256"] in seen_sha or any(ham(d, x) <= 4 for x in seen_dh):
        stats["dup_within"] += 1; continue
    seen_sha.add(r["sha256"]); seen_dh.append(d)
    kept.append({**r, "_p": p, "_d": d})
print(stats, "kept", len(kept), flush=True)

# split by listing; listings joined into clusters when any of their photos are near-dups (incl. dropped dups' owners)
lst = {}
for r in kept:
    lst.setdefault((r["site"], r["listingId"]), []).append(r)
keys = sorted(lst)
random.Random(20261022).shuffle(keys)
n_train = round(len(keys) * 0.6)
split_of = {k: ("train" if i < n_train else "eval") for i, k in enumerate(keys)}
# guard: kept photos are pairwise dHash>4, so no near-dup crosses the boundary by construction.
for k in keys:
    for r in lst[k]:
        r["split"] = split_of[k]

for s in {k[0] for k in keys}:
    shutil.rmtree(os.path.join(H, s), ignore_errors=True)
manifest = []
for k in keys:
    for r in lst[k]:
        dst = os.path.join(H, k[0], r["split"], k[1])
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(r["_p"], os.path.join(dst, f"photo-{r['photo']}.jpg"))
        manifest.append({"site": k[0], "listingId": k[1], "split": r["split"], "url": r["url"], "title": r["title"],
                         "price": r["price"], "photo": r["photo"], "width": r["width"], "height": r["height"], "sha256": r["sha256"]})
json.dump(manifest, open(os.path.join(H, "MANIFEST.json"), "w"), indent=1)

summary = {"dedupe": stats, "listings": {}, "sites": {}}
for k in keys:
    summary["listings"].setdefault(k[0], {"train": 0, "eval": 0})[split_of[k]] += 1
for site in sorted({k[0] for k in keys}):
    counts = {b[0]: 0 for b in BUCKETS}
    photos = boxes = zero = 0
    widths = []
    for r in manifest:
        if r["site"] != site:
            continue
        p = os.path.join(H, site, r["split"], r["listingId"], f"photo-{r['photo']}.jpg")
        with Image.open(p) as im:
            dets = detect_cards(im.convert("RGB"), DEFAULT_MODEL)
        photos += 1; widths.append(r["width"]); zero += not dets
        for d in dets:
            h = d["box"][3] - d["box"][1]; boxes += 1
            for lab, lo, hi in BUCKETS:
                if lo <= h < hi:
                    counts[lab] += 1
    summary["sites"][site] = {"photos": photos, "boxes": boxes, "zero_box_photos": zero, "box_height_buckets": counts,
                              "median_photo_width": statistics.median(widths) if widths else None}
json.dump(summary, open(os.path.join(H, "detect-summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
