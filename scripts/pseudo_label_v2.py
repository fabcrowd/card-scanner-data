"""Pseudo-labels v2: stricter rules from the Opus audit of v1 (see pseudo_label.py for the shared detect/CLIP/SIFT stage).

Per box (>= 300 px, CLIP top-10 physical candidates, SIFT verified like v1) every intermediate number is stored.
Rules, in order:
  energy        normalised name contains 'energy'                         -> REJECT 'energy'
  japanese      OCR tags the name bar as CJK/kana                         -> REJECT 'japanese' (no local OCR is installed
                in this venv, so lang is always 'unknown' and this rule never fires)
  geometry      best candidate: homography exists, det > 0, scale in [0.4, 2.5], |rotation| <= 25 deg, >= 25 inliers
                inside the catalog ART window (top-45% strip centred square)  -> else REJECT 'geometry'
  ID level      best >= 100 and best >= 2x best candidate outside its art group, name not gappy -> label = art group
  NAME level    best >= 40; candidates >= 0.5x best must all share the best's normalised name; for best 40-99 the CLIP
                top-1 name must equal the best's name -> else REJECT 'clip_disagree' / 'ambiguous_name'
  else          REJECT 'low_inliers'
Art groups (art_groups.json): same normalised name (owner possessive stripped) and (art-crop pHash Hamming <= 10 or CLIP
cosine >= 0.95), connected components. Gappy names (a catalog id with that name has no image) are capped at NAME level.
Trainer/Item/Stadium name-level-only rule is NOT applied: catalog meta has no supertype/category field.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pseudo_label as P  # noqa: E402
import run_real_photos as R  # noqa: E402
from score_real_labels import normalise  # noqa: E402

OUT_DIR = P.OUT_DIR
GROUPS_PATH = os.path.join(OUT_DIR, "art_groups.json")
MAX_WORKERS = 8
PHASH_MAX_DIST = 10
GROUP_COS = 0.95
ID_MIN_INLIERS = 100
ID_LEAD = 2.0
NAME_MIN_INLIERS = 40
NAME_CLIP_AGREE_BELOW = 100
NAME_NEAR_FRAC = 0.5
ART_MIN_INLIERS = 25
SCALE_MIN, SCALE_MAX = 0.4, 2.5
MAX_ROT_DEG = 25.0
SEED = 11
VERIFY_SIZE = (400, 560)  # local_verify._SIZE

_ctx2 = {}


# ---------------------------------------------------------------- art groups

def art_window(width=VERIFY_SIZE[0], height=VERIFY_SIZE[1]):
    """(x0, y0, x1, y1) of the extension art crop (top 45% strip, centred square) in a width x height image."""
    strip = max(1, int(height * 0.45))
    side = min(width, strip)
    sx, sy = (width - side) // 2, (strip - side) // 2
    return sx, sy, sx + side, sy + side


def group_name(name):
    """normalise() plus a leading trainer-owner possessive dropped ("Erika's Bellsprout" -> "bellsprout")."""
    s = (name or "").lower().replace("’", "'")
    s = re.sub(r"^[\w .\-]+?'s\s+", "", s, count=1)
    return normalise(s)


def phash64(img):
    """64-bit DCT pHash (int) of the extension art crop of a PIL image."""
    from scipy.fft import dctn

    sx, sy, x1, y1 = art_window(*img.size)
    art = img.convert("L").crop((sx, sy, x1, y1)).resize((32, 32), Image.BICUBIC)
    low = dctn(np.asarray(art, dtype=np.float64), norm="ortho")[:8, :8].reshape(-1)
    bits = low > np.median(low)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def build_groups(ids, names, hashes, vectors):
    """Connected components over pairs with equal group_name and (pHash <= 10 or CLIP cosine >= 0.95).

    ids/names/hashes parallel lists (hash may be None); vectors an (n, d) L2-normalised array. Returns {id: group_id}
    where group_id is the smallest member id (string order)."""
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets = {}
    for i, n in enumerate(names):
        buckets.setdefault(group_name(n), []).append(i)
    for members in buckets.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if find(i) == find(j):
                    continue
                near_hash = hashes[i] is not None and hashes[j] is not None and hamming(hashes[i], hashes[j]) <= PHASH_MAX_DIST
                if near_hash or float(vectors[i] @ vectors[j]) >= GROUP_COS:
                    parent[find(i)] = find(j)
    comp = {}
    for i in range(len(ids)):
        comp.setdefault(find(i), []).append(ids[i])
    out = {}
    for members in comp.values():
        gid = min(members)
        for m in members:
            out[m] = gid
    return out


def _hash_one(cid):
    for ext in (".webp", ".png"):
        p = os.path.join(P.IMAGES_DIR, cid + ext)
        if os.path.exists(p):
            try:
                with Image.open(p) as im:
                    return cid, phash64(im.convert("RGB"))
            except Exception:
                return cid, None
    return cid, None


def precompute_groups(path=GROUPS_PATH, workers=MAX_WORKERS):
    from pipeline.catalog import Catalog

    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    name_of = R.load_names()
    keep = [i for i, c in enumerate(cat.ids) if not R.DIGITAL_RE.match(c)]
    ids = [cat.ids[i] for i in keep]
    with mp.Pool(max(1, min(workers, MAX_WORKERS))) as pool:
        hmap = dict(pool.imap(_hash_one, ids, chunksize=64))
    names = [name_of.get(c) or c for c in ids]
    gmap = build_groups(ids, names, [hmap[c] for c in ids], cat.vectors[keep])
    members = {}
    for cid, g in gmap.items():
        members.setdefault(g, []).append(cid)
    members = {g: sorted(m) for g, m in members.items()}
    sizes = Counter(len(m) for m in members.values())
    data = {"params": {"phash_max_dist": PHASH_MAX_DIST, "clip_cos": GROUP_COS},
            "n_ids": len(ids), "n_groups": len(members),
            "size_distribution": {str(k): v for k, v in sorted(sizes.items())},
            "groups": {g: m for g, m in members.items() if len(m) > 1},
            "phash": {c: (f"{h:016x}" if h is not None else None) for c, h in hmap.items()}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def load_groups(path=GROUPS_PATH):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    id_to_group = {}
    for g, m in data["groups"].items():
        for c in m:
            id_to_group[c] = g
    return id_to_group, data["groups"], data


def gappy_names():
    """Normalised names of which any physical catalog id (meta.json or meta.full.json) has no image file."""
    have = {f.rsplit(".", 1)[0] for f in os.listdir(P.IMAGES_DIR)}
    gappy = set()
    for f in ("meta.json", "meta.full.json"):
        with open(os.path.join(R.CARD_DIR, f), encoding="utf-8") as fh:
            for m in json.load(fh):
                cid, nm = m.get("id"), m.get("name")
                if cid and nm and not R.DIGITAL_RE.match(cid) and cid not in have:
                    gappy.add(normalise(nm))
    return gappy


# ---------------------------------------------------------------- geometry and decision

def geometry_check(H, art_inliers):
    """(ok, reason, stats) for a query->catalog homography (400x560 both). H may be None."""
    if H is None:
        return False, "no_homography", {"det": None, "scale": None, "rot_deg": None}
    H = np.asarray(H, dtype=np.float64)
    if H[2, 2] != 0:
        H = H / H[2, 2]
    det = float(np.linalg.det(H[:2, :2]))
    scale = math.sqrt(abs(det))
    rot = math.degrees(math.atan2(H[1, 0], H[0, 0]))
    stats = {"det": round(det, 4), "scale": round(scale, 4), "rot_deg": round(rot, 2)}
    if not np.all(np.isfinite(H)) or det <= 0:
        return False, "bad_determinant", stats
    if not SCALE_MIN <= scale <= SCALE_MAX:
        return False, "bad_scale", stats
    if abs(rot) > MAX_ROT_DEG:
        return False, "bad_rotation", stats
    if art_inliers < ART_MIN_INLIERS:
        return False, "few_art_inliers", stats
    return True, "ok", stats


def decide_v2(cands, id_to_group, gappy, lang="unknown"):
    """cands: dicts id, name, clip_rank, cos, inliers, geom_ok (for the best only it matters), art_inliers.

    Returns {level: id|name|reject, reason, ids, name, best, outside_best, ...}."""
    base = {"level": "reject", "reason": None, "ids": [], "name": None, "best_id": None, "inliers_best": 0,
            "inliers_outside_group": 0, "art_inliers_best": 0, "clip_top1_name": None}
    if not cands:
        base["reason"] = "no_candidates"
        return base
    ranked = sorted(cands, key=lambda c: (c["inliers"], c["cos"]), reverse=True)
    best = ranked[0]
    top1 = min(cands, key=lambda c: c["clip_rank"])
    nbest = normalise(best["name"] or best["id"])
    gid = lambda c: id_to_group.get(c["id"], c["id"])  # noqa: E731
    outside = max([c["inliers"] for c in ranked if gid(c) != gid(best)] or [0])
    base.update(best_id=best["id"], inliers_best=best["inliers"], inliers_outside_group=outside,
                art_inliers_best=best.get("art_inliers", 0), clip_top1_name=top1["name"], name=best["name"])
    if "energy" in nbest:
        base["reason"] = "energy"
        return base
    if lang == "ja":
        base["reason"] = "japanese"
        return base
    if not best.get("geom_ok", False):
        base["reason"] = "geometry"
        base["geom_reason"] = best.get("geom_reason")
        return base
    if best["inliers"] >= ID_MIN_INLIERS and best["inliers"] >= ID_LEAD * outside and nbest not in gappy:
        g = gid(best)
        members = sorted({c for c, gg in id_to_group.items() if gg == g} | {best["id"]})
        base.update(level="id", reason="id_ok", ids=members)
        return base
    if best["inliers"] >= NAME_MIN_INLIERS:
        if best["inliers"] < NAME_CLIP_AGREE_BELOW and normalise(top1["name"] or top1["id"]) != nbest:
            base["reason"] = "clip_disagree"
            return base
        near = [c for c in ranked if c["inliers"] >= NAME_NEAR_FRAC * best["inliers"]]
        if all(normalise(c["name"] or c["id"]) == nbest for c in near):
            why = "gappy_name" if (best["inliers"] >= ID_MIN_INLIERS and nbest in gappy) else "name_ok"
            base.update(level="name", reason=why)
            return base
        base["reason"] = "ambiguous_name"
        return base
    base["reason"] = "low_inliers"
    return base


# ---------------------------------------------------------------- per-box work

def verify_candidate(V, query, cid):
    """Like local_verify._inliers (same matcher, ratio, RANSAC) but keeps the homography and the art-window count."""
    ck, cd = V._catalog_features(P.IMAGES_DIR, cid)
    qk, qd = query
    out = {"inliers": 0, "art_inliers": 0, "H": None}
    if qd is None or cd is None or len(qd) < 8 or len(cd) < 8:
        return out
    pairs = V._matcher.knnMatch(qd, cd, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < V._RATIO * n.distance]
    if len(good) < 8:
        return out
    src = np.float32([qk[m.queryIdx].pt for m in good])
    dst = np.float32([ck[m.trainIdx].pt for m in good])
    import cv2

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, V._RANSAC_PX)
    if mask is None:
        return out
    inl = mask.ravel().astype(bool)
    x0, y0, x1, y1 = art_window()
    d = dst[inl]
    out["inliers"] = int(inl.sum())
    out["art_inliers"] = int(((d[:, 0] >= x0) & (d[:, 0] < x1) & (d[:, 1] >= y0) & (d[:, 1] < y1)).sum())
    out["H"] = H
    return out


def init_ctx2():
    P.init_ctx()
    id_to_group, _, _ = load_groups()
    _ctx2.update(id_to_group=id_to_group, gappy=gappy_names())


def label_box_v2(img, box):
    cat, emb, name_of, V = P._ctx["cat"], P._ctx["emb"], P._ctx["name_of"], P._ctx["verify"]
    crop = R.crop_card(img, box)
    q = emb(R.to_png(crop), trusted=True)
    idx, sims = cat.top_k_batch(q[None], k=P.FETCH_K)
    pairs = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if not P._ctx["is_dig"][j]][:P.CLIP_TOP]
    query = V._features(np.asarray(crop)[:, :, ::-1].copy())
    cands = []
    for rank, (j, s) in enumerate(pairs, 1):
        cid = cat.ids[j]
        v = verify_candidate(V, query, cid)
        ok, why, st = geometry_check(v["H"], v["art_inliers"])
        cands.append({"id": cid, "name": name_of.get(cid), "clip_rank": rank, "cos": round(s, 4),
                      "inliers": v["inliers"], "art_inliers": v["art_inliers"], "geom_ok": ok, "geom_reason": why, **st})
    d = decide_v2(cands, _ctx2["id_to_group"], _ctx2["gappy"], lang="unknown")
    nb = normalise(d["name"] or "")
    d["gappy"] = bool(d["name"]) and nb in _ctx2["gappy"]
    return {"box": [round(float(v), 1) for v in box], "box_height": round(float(box[3] - box[1]), 1),
            "lang": "unknown", **d, "candidates": cands}


def _work(path):
    t = time.time()
    with Image.open(path) as im:
        img = im.convert("RGB")
    rows = []
    n_all = len(P.D.detect_cards(img, P._ctx["detector"]))
    for dct in P.D.detect_cards(img, P._ctx["detector"]):
        b = dct["box"]
        if b[3] - b[1] < P.MIN_HEIGHT:
            continue
        r = label_box_v2(img, b)
        r["photo"] = path
        r["det_score"] = round(dct["score"], 3)
        rows.append(r)
    return path, n_all, rows, time.time() - t


# ---------------------------------------------------------------- reporting

def v1_level(r):
    return ("name" if r["name_only"] else "id") if r["accepted"] else "reject"


def box_key(r):
    return (r["photo"], tuple(r["box"]))


def compare_v1(rows, v1_path):
    v1 = [json.loads(line) for line in open(v1_path, encoding="utf-8")]
    v2 = {box_key(r): r for r in rows}
    ch = Counter()
    for a in v1:
        b = v2.get(box_key(a))
        ch[(v1_level(a), b["level"] if b else "missing")] += 1
    return v1, v2, ch


WATCH = [("Rhydon", "base4-59"), ("Combusken", "ex1-28"), ("Koffing", "swsh2-112"), ("Lairon", "ex14-36"),
         ("Regirock", "ex9-99"), ("Diglett", "xy12-55"), ("Larvesta", "sm12-34"), ("Decidueye ex", "sv03-015"),
         ("Drapion V", "swsh11-118"), ("Cinccino", "swsh9-125"), ("Gardevoir", "TG05")]


def watch_report(v1, v2):
    lines = []
    for name, cid in WATCH:
        hits = [a for a in v1 if a["accepted"] and ((a.get("card_id") or "").endswith(cid) or (cid == "TG05" and "TG05" in str(a["top3"][0]["id"]) and a["name"] == name)
                                                    or (a["top3"] and a["top3"][0]["id"] == cid))]
        if not hits:
            lines.append(f"  {name} {cid}: no v1 box found")
        for a in hits:
            b = v2.get(box_key(a))
            lines.append(f"  {name} {cid} [{os.path.basename(os.path.dirname(a['photo']))}/{os.path.basename(a['photo'])} {a['box']}]: "
                         f"v1 {v1_level(a)} -> v2 {b['level']}/{b['reason']} best {b['best_id']} inl {b['inliers_best']}/{b['inliers_outside_group']} "
                         f"art {b['art_inliers_best']} ids {len(b['ids'])}")
    return "\n".join(lines)


def summarize(rows, n_photos, n_all, secs, ch, gdata):
    big = [r for r in rows if r["box_height"] >= P.MIN_HEIGHT]
    idl = [r for r in rows if r["level"] == "id"]
    nml = [r for r in rows if r["level"] == "name"]
    rej = Counter(r["reason"] for r in rows if r["level"] == "reject")
    gs = Counter(len(r["ids"]) for r in idl)
    lines = [f"photos {n_photos}; boxes (all heights) {n_all}; boxes >= {P.MIN_HEIGHT}px {len(big)}",
             f"id-level/art-group {len(idl)} (exact single id {gs.get(1, 0)}); name-level {len(nml)}; rejected {sum(rej.values())}",
             "rejected by reason: " + ", ".join(f"{k} {v}" for k, v in rej.most_common()),
             "id-label group sizes: " + ", ".join(f"{k}:{v}" for k, v in sorted(gs.items())),
             f"art groups: {gdata['n_groups']} over {gdata['n_ids']} ids; size distribution {gdata['size_distribution']}",
             "v1 -> v2 level transitions: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(ch.items())),
             "lang: all 'unknown' (no local OCR installed); trainer rule skipped (no supertype in meta)",
             f"runtime {secs:.0f}s"]
    return "\n".join(lines)


def _tile(r, tw=200, th=280):
    with Image.open(r["photo"]) as im:
        crop = R.crop_card(im.convert("RGB"), r["box"]).resize((tw, th), Image.BILINEAR)
    ref_id = r["best_id"] if r["level"] != "reject" else (r["candidates"][0]["id"] if r["candidates"] else None)
    ref = P._catalog_image(ref_id) if ref_id else None
    tile = Image.new("RGB", (2 * tw, th + 62), (255, 255, 255))
    tile.paste(crop, (0, 0))
    if ref is not None:
        tile.paste(ref.resize((tw, th), Image.BILINEAR), (tw, 0))
    f = P._tile_font()
    dr = ImageDraw.Draw(tile)
    ids = ",".join(r["ids"][:3]) + (f"+{len(r['ids']) - 3}" if len(r["ids"]) > 3 else "") if r["ids"] else (ref_id or "")
    dr.text((3, th + 2), f"{r['level'].upper()}/{r['reason']} {ids}"[:62], fill=(0, 0, 0), font=f)
    dr.text((3, th + 20), f"inl best {r['inliers_best']} outside-grp {r['inliers_outside_group']} art {r['art_inliers_best']}", fill=(0, 0, 0), font=f)
    dr.text((3, th + 38), f"{(r['name'] or '')[:40]} h={r['box_height']:.0f}", fill=(0, 0, 0), font=f)
    col = {"id": (0, 160, 0), "name": (220, 140, 0), "reject": (200, 0, 0)}[r["level"]]
    dr.rectangle([0, 0, 2 * tw - 1, th + 61], outline=col)
    return tile


def build_audit(rows, out_dir, n_id=45, n_name=30, n_rej=15, per_sheet=15, cols=3):
    rnd = random.Random(SEED)
    pools = [[r for r in rows if r["level"] == k] for k in ("id", "name", "reject")]
    pick = []
    for pool, n in zip(pools, (n_id, n_name, n_rej)):
        pick += rnd.sample(pool, min(n, len(pool)))
    rnd.shuffle(pick)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in range(0, len(pick), per_sheet):
        tiles = [_tile(r) for r in pick[s:s + per_sheet]]
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (cols * tw, ((len(tiles) + cols - 1) // cols) * th), (255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
        p = os.path.join(out_dir, f"sheet-{s // per_sheet + 1}.jpg")
        sheet.save(p, "JPEG", quality=90)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default=os.path.join(ROOT, "data", "goodwill-pack"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "goodwill-pack-v2.jsonl"))
    ap.add_argument("--v1", default=os.path.join(OUT_DIR, "goodwill-pack.jsonl"))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    a = ap.parse_args(argv)
    if not os.path.exists(GROUPS_PATH):
        t = time.time()
        precompute_groups(workers=a.workers)
        print(f"art groups built in {time.time() - t:.0f}s", flush=True)
    gdata = load_groups()[2]
    photos = P.list_photos(a.photos)
    t0 = time.time()
    rows, n_all = [], 0
    with mp.Pool(max(1, min(a.workers, MAX_WORKERS)), initializer=init_ctx2) as pool:
        for k, (path, n, r, dt) in enumerate(pool.imap(_work, photos), 1):
            rows += r
            n_all += n
            print(f"[{k}/{len(photos)}] boxes>=300: {len(r)} id {sum(x['level'] == 'id' for x in r)} name {sum(x['level'] == 'name' for x in r)} {dt:.1f}s", flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    v1, v2, ch = compare_v1(rows, a.v1)
    print(summarize(rows, len(photos), n_all, time.time() - t0, ch, gdata))
    print("watch list:\n" + watch_report(v1, v2))
    print("audit sheets:", *build_audit(rows, os.path.join(OUT_DIR, "audit-v2")), sep="\n  ")


if __name__ == "__main__":
    mp.freeze_support()
    main()
