"""Pseudo-labels v3: clique art groups, listing-level Japanese exclusion (decision rules and per-box work are v2's).

Changes from v2 (Opus audit: v2 groups chained/over-merged and missed identical prints):
  art groups   among ids with the same group_name only. Candidate pair = (art pHash Hamming <= 10 AND CLIP cosine >= 0.97)
               OR (SIFT between the two CATALOG images, local_verify inliers >= 100; only tried for pairs with cosine >= 0.90).
               Groups are greedy CLIQUES: every pair inside a group is a candidate pair (no chaining).
  label        v2's decide_v2 with the v3 groups. 'train_id' = best_id (id level), 'no_negative' = the other members of
               its group (must not be used as negatives for the training id).
  japanese     (a) Windows OCR has no Japanese pack on this machine, so no per-crop OCR; lang stays 'unknown'.
               (b) listing-level 'suspect_ja': title mentions japanese/jp/japan, OR >= 50% of the listing's >= 300 px boxes
               are geometry rejects AND >= 50% of its accepted labels come from NEW_SETS. All boxes of such listings are
               excluded from labels.
Eval folders (any path component 'eval') and data/sealed-test are refused.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pseudo_label as P  # noqa: E402
import pseudo_label_v2 as V2  # noqa: E402
import run_real_photos as R  # noqa: E402
from score_real_labels import normalise  # noqa: E402

OUT_DIR = P.OUT_DIR
GROUPS_V3_PATH = os.path.join(OUT_DIR, "art_groups_v3.json")
V2_GROUPS_PATH = V2.GROUPS_PATH  # holds the cached art-crop pHashes
OUT_V3 = os.path.join(OUT_DIR, "v3")
AUDIT_V3 = os.path.join(OUT_DIR, "audit-v3")
MAX_WORKERS = 8
PHASH_MAX_DIST = 10
PAIR_COS = 0.97
SIFT_COS_FLOOR = 0.90
SIFT_MIN_INLIERS = 100
JA_TITLE_RE = re.compile(r"japan(?:ese)?|\bjp\b", re.IGNORECASE)
JA_GEOM_FRAC = 0.5
JA_NEW_SET_FRAC = 0.5
# Newest sets in the catalog (no add-date field exists): the Mega Evolution tail the audit saw Japanese prints match.
NEW_SETS = frozenset({"me03", "me04", "me05"})
RECENT_SECS = 120
AUDIT_SEED = 13

_ctx3 = {}


# ---------------------------------------------------------------- safety

def assert_allowed_path(path):
    """Refuse eval folders and the sealed test set."""
    parts = [p.lower() for p in re.split(r"[\\/]+", os.path.abspath(path))]
    if "eval" in parts or "sealed-test" in parts:
        raise ValueError(f"refusing to read eval / sealed data: {path}")
    return path


# ---------------------------------------------------------------- art groups v3

def greedy_cliques(nodes, edges):
    """Partition nodes into cliques of the undirected graph `edges` (set of frozenset pairs). No chaining.

    Highest-degree nodes first (ties by node order), each joins the first group it is adjacent to in full."""
    adj = {n: set() for n in nodes}
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    pos = {n: k for k, n in enumerate(nodes)}
    order = sorted(nodes, key=lambda n: (-len(adj[n]), pos[n]))
    groups = []
    for n in order:
        for g in groups:
            if all(m in adj[n] for m in g):
                g.append(n)
                break
        else:
            groups.append([n])
    return groups


def bucket_edges(members, hashes, vectors, sift_fn):
    """Candidate-pair edges among `members` (indices into hashes/vectors). sift_fn(i, j) -> inliers (or None to skip)."""
    edges = set()
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            i, j = members[a], members[b]
            cos = float(vectors[i] @ vectors[j])
            near = hashes[i] is not None and hashes[j] is not None and V2.hamming(hashes[i], hashes[j]) <= PHASH_MAX_DIST
            if near and cos >= PAIR_COS:
                edges.add(frozenset((i, j)))
            elif cos >= SIFT_COS_FLOOR and sift_fn is not None and sift_fn(i, j) >= SIFT_MIN_INLIERS:
                edges.add(frozenset((i, j)))
    return edges


def name_buckets(names):
    buckets = {}
    for i, n in enumerate(names):
        buckets.setdefault(V2.group_name(n), []).append(i)
    return [m for m in buckets.values() if len(m) > 1]


def build_groups_v3(ids, names, hashes, vectors, sift_fn=None):
    """{id: group_id} (group_id = smallest member id). Sequential reference implementation (used by tests)."""
    out = {}
    for members in name_buckets(names):
        for g in greedy_cliques(members, bucket_edges(members, hashes, vectors, sift_fn)):
            gid = min(ids[i] for i in g)
            for i in g:
                out[ids[i]] = gid
    for c in ids:
        out.setdefault(c, c)
    return out


def _sift_pair(task):
    """Worker: edges for one name bucket using the project verifier on the two CATALOG images."""
    members, hashes, vecs = task
    if "V" not in _ctx3:
        _ctx3["V"] = P._load_verifier()
    V = _ctx3["V"]
    ids = [m[1] for m in members]
    idx = list(range(len(members)))
    cache = {}

    def feats(k):
        if k not in cache:
            cache[k] = V._catalog_features(P.IMAGES_DIR, ids[k])
        return cache[k]

    def sift(i, j):
        return V._inliers(feats(i), feats(j))

    edges = bucket_edges(idx, hashes, vecs, sift)
    gidx = [m[0] for m in members]
    return gidx, {frozenset((gidx[a], gidx[b])) for a, b in (tuple(e) for e in edges)}


def precompute_groups_v3(path=GROUPS_V3_PATH, workers=MAX_WORKERS):
    from pipeline.catalog import Catalog

    with open(V2_GROUPS_PATH, encoding="utf-8") as fh:
        ph = json.load(fh)["phash"]
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    name_of = R.load_names()
    keep = [i for i, c in enumerate(cat.ids) if not R.DIGITAL_RE.match(c)]
    ids = [cat.ids[i] for i in keep]
    names = [name_of.get(c) or c for c in ids]
    vectors = cat.vectors[keep]
    hashes = [int(ph[c], 16) if ph.get(c) else None for c in ids]
    buckets = name_buckets(names)
    tasks = [([(i, ids[i]) for i in b], [hashes[i] for i in b], vectors[b]) for b in buckets]
    tasks.sort(key=lambda t: -len(t[0]))  # biggest buckets first for load balance
    members_by_gid = {}
    t = time.time()
    with mp.Pool(max(1, min(workers, MAX_WORKERS))) as pool:
        for k, (gidx, local_edges) in enumerate(pool.imap_unordered(_sift_pair, tasks), 1):
            for g in greedy_cliques(gidx, local_edges):
                members_by_gid[min(ids[i] for i in g)] = sorted(ids[i] for i in g)
            if k % 500 == 0:
                print(f"  buckets {k}/{len(tasks)} {time.time() - t:.0f}s", flush=True)
    groups = {g: m for g, m in members_by_gid.items() if len(m) > 1}
    n_groups = len(ids) - sum(len(m) - 1 for m in groups.values())
    sizes = Counter(len(m) for m in groups.values())
    singles = n_groups - len(groups)
    dist = {"1": singles, **{str(k): v for k, v in sorted(sizes.items())}}
    data = {"params": {"phash_max_dist": PHASH_MAX_DIST, "pair_cos": PAIR_COS, "sift_cos_floor": SIFT_COS_FLOOR,
                       "sift_min_inliers": SIFT_MIN_INLIERS, "grouping": "greedy cliques, same group_name only"},
            "n_ids": len(ids), "n_groups": n_groups, "size_distribution": dist, "groups": groups}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def load_groups_v3(path=GROUPS_V3_PATH):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    id_to_group = {c: g for g, m in data["groups"].items() for c in m}
    return id_to_group, data["groups"], data


# ---------------------------------------------------------------- listing-level Japanese flag

def set_of(card_id):
    return re.sub(r"-[^-]+$", "", card_id or "")


def title_is_japanese(title):
    return bool(JA_TITLE_RE.search(title or ""))


def suspect_ja(title, rows):
    """(flag, why) for one listing from its title and v2-style row dicts (level, reason, best_id)."""
    if title_is_japanese(title):
        return True, "title"
    if not rows:
        return False, None
    geom = sum(1 for r in rows if r["level"] == "reject" and r["reason"] == "geometry")
    acc = [r for r in rows if r["level"] in ("id", "name")]
    if not acc:
        return False, None
    new = sum(1 for r in acc if set_of(r["best_id"]) in NEW_SETS)
    if geom >= JA_GEOM_FRAC * len(rows) and new >= JA_NEW_SET_FRAC * len(acc):
        return True, "geometry_new_sets"
    return False, None


def finalize_row(r, listing, flagged):
    """Add v3 fields; a suspect_ja listing's accepted boxes become rejects (pre-exclusion level kept)."""
    out = dict(r)
    out["listing"] = listing
    out["level_pre_exclusion"] = r["level"]
    out["train_id"] = r["best_id"] if r["level"] == "id" else None
    out["no_negative"] = [c for c in r["ids"] if c != r["best_id"]] if r["level"] == "id" else []
    out["suspect_ja"] = flagged
    if flagged and r["level"] in ("id", "name"):
        out.update(level="reject", reason="suspect_ja_listing", train_id=None, no_negative=[])
    return out


# ---------------------------------------------------------------- manifests and listing discovery

def load_manifest_titles(manifest_path, split=None):
    with open(manifest_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    return {str(e["itemId"]): e.get("title", "") for e in entries if split is None or e.get("split") == split}


def listing_folders(photos_dir, titles, now=None, recent_secs=RECENT_SECS):
    """(ready folders, skipped-recent folders, skipped-no-manifest folders), names only, sorted."""
    assert_allowed_path(photos_dir)
    now = time.time() if now is None else now
    ready, recent, nomf = [], [], []
    for name in sorted(os.listdir(photos_dir)):
        full = os.path.join(photos_dir, name)
        if not os.path.isdir(full) or name.startswith("_"):
            continue
        if name not in titles:
            nomf.append(name)
        elif now - os.path.getmtime(full) < recent_secs:
            recent.append(name)
        else:
            ready.append(name)
    return ready, recent, nomf


def photos_of(folder):
    assert_allowed_path(folder)
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(P.PHOTO_EXT))


# ---------------------------------------------------------------- per-box work

def init_ctx3():
    P.init_ctx()
    id_to_group, _, _ = load_groups_v3()
    V2._ctx2.update(id_to_group=id_to_group, gappy=V2.gappy_names())


def _work3(path):
    t = time.time()
    assert_allowed_path(path)
    with Image.open(path) as im:
        img = im.convert("RGB")
    dets = P.D.detect_cards(img, P._ctx["detector"])
    rows = []
    for dct in dets:
        b = dct["box"]
        if b[3] - b[1] < P.MIN_HEIGHT:
            continue
        r = V2.label_box_v2(img, b)
        r["photo"] = path
        r["det_score"] = round(dct["score"], 3)
        rows.append(r)
    return path, len(dets), rows, time.time() - t


# ---------------------------------------------------------------- run, summary, audit

def summarize_pack(name, n_photos, n_all, rows, flagged, secs):
    big = [r for r in rows if r["box_height"] >= P.MIN_HEIGHT]
    idl = [r for r in rows if r["level"] == "id"]
    nml = [r for r in rows if r["level"] == "name"]
    rej = Counter(r["reason"] for r in rows if r["level"] == "reject")
    return {"pack": name, "photos": n_photos, "boxes_all_heights": n_all, "boxes_ge_300": len(big),
            "id_level": len(idl), "name_level": len(nml), "rejected": sum(rej.values()),
            "rejected_by_reason": dict(rej.most_common()),
            "suspect_ja_listings": flagged, "runtime_s": round(secs)}


def _tile_path_ok(r):
    assert_allowed_path(r["photo"])
    return r


def build_audit3(rows, out_dir, n_id=40, n_name=20, per_sheet=15, cols=3):
    rnd = random.Random(AUDIT_SEED)
    pick = []
    for level, n in (("id", n_id), ("name", n_name)):
        pool = [r for r in rows if r["level"] == level]
        pick += rnd.sample(pool, min(n, len(pool)))
    rnd.shuffle(pick)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in range(0, len(pick), per_sheet):
        tiles = [V2._tile(_tile_path_ok(r)) for r in pick[s:s + per_sheet]]
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (cols * tw, ((len(tiles) + cols - 1) // cols) * th), (255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
        p = os.path.join(out_dir, f"sheet-{s // per_sheet + 1}.jpg")
        sheet.save(p, "JPEG", quality=90)
        paths.append(p)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump([{"sheet": i // per_sheet + 1, "pos": i % per_sheet, "photo": r["photo"], "box": r["box"],
                    "level": r["level"], "best_id": r["best_id"]} for i, r in enumerate(pick)], fh)
    return paths


def run_pack(name, photos_dir, manifest_path, split, workers):
    titles = load_manifest_titles(manifest_path, split)
    ready, recent, nomf = listing_folders(photos_dir, titles)
    print(f"{name}: {len(ready)} listing folders ready, {len(recent)} skipped (modified < {RECENT_SECS}s), "
          f"{len(nomf)} skipped (no manifest entry)", flush=True)
    photos = [p for f in ready for p in photos_of(os.path.join(photos_dir, f))]
    t0 = time.time()
    rows, n_all = [], 0
    with mp.Pool(max(1, min(workers, MAX_WORKERS)), initializer=init_ctx3) as pool:
        for k, (path, n, r, dt) in enumerate(pool.imap(_work3, photos), 1):
            rows += r
            n_all += n
            if k % 25 == 0 or k == len(photos):
                print(f"[{k}/{len(photos)}] rows so far {len(rows)} {dt:.1f}s", flush=True)
    by_listing = {}
    for r in rows:
        by_listing.setdefault(os.path.basename(os.path.dirname(r["photo"])), []).append(r)
    flagged, final = {}, []
    for lid in sorted(by_listing):
        flag, why = suspect_ja(titles.get(lid, ""), by_listing[lid])
        if flag:
            flagged[lid] = why
        final += [finalize_row(r, lid, flag) for r in by_listing[lid]]
    os.makedirs(OUT_V3, exist_ok=True)
    with open(os.path.join(OUT_V3, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r) + "\n")
    summ = summarize_pack(name, len(photos), n_all, final, flagged, time.time() - t0)
    summ["skipped_recent"] = recent
    summ["skipped_no_manifest"] = nomf
    with open(os.path.join(OUT_V3, f"{name}-summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))
    return final


def report_groups(data):
    print(f"v3 groups: {data['n_groups']} over {data['n_ids']} ids; size distribution {data['size_distribution']}")
    id_to_group = {c: g for g, m in data["groups"].items() for c in m}
    together = id_to_group.get("svp-196") is not None and id_to_group.get("svp-196") == id_to_group.get("sv03-125")
    print(f"Charizard ex svp-196 & sv03-125 together: {together}")
    print(f"Gabite sm11-113 vs swsh9-108 merged: {id_to_group.get('sm11-113') is not None and id_to_group.get('sm11-113') == id_to_group.get('swsh9-108')}")
    name_of = R.load_names()
    for nm in ("pikachu", "charmander", "swablu", "gabite"):
        sizes = sorted((len(m) for m in data["groups"].values() if V2.group_name(name_of.get(m[0]) or m[0]) == nm), reverse=True)
        print(f"  {nm}: groups>1 sizes {sizes[:8]} (max {sizes[0] if sizes else 1})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--groups-only", action="store_true")
    ap.add_argument("--rebuild-groups", action="store_true")
    a = ap.parse_args(argv)
    if a.rebuild_groups or not os.path.exists(GROUPS_V3_PATH):
        t = time.time()
        precompute_groups_v3(workers=a.workers)
        print(f"art groups v3 built in {time.time() - t:.0f}s", flush=True)
    report_groups(load_groups_v3()[2])
    if a.groups_only:
        return
    data = os.path.join(ROOT, "data")
    run_pack("goodwill-pack", os.path.join(data, "goodwill-pack"), os.path.join(data, "goodwill-pack", "MANIFEST.json"), None, a.workers)
    rows2 = run_pack("goodwill-pack2", os.path.join(data, "goodwill-pack2", "train"),
                     os.path.join(data, "goodwill-pack2", "MANIFEST.json"), "train", a.workers)
    print("audit sheets:", *build_audit3(rows2, AUDIT_V3), sep="\n  ")


if __name__ == "__main__":
    mp.freeze_support()
    main()
