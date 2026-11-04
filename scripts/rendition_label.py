"""Label harvest TRAIN photos (id/name-level geometric labels, >=300 px boxes) for the rendition study.
Candidate listings per site: >=2 photos, ordered by largest photo, first CAND listings, first PHOTOS photos each.
Output: data/reports/rendition/labels.jsonl (v3 rows)."""
import json, multiprocessing as mp, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import pseudo_label_v3 as V3  # noqa: E402
H = os.path.join(ROOT, "data", "harvest-20261022")
CAND, PHOTOS = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (18, 6)
SITES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mercari", "hibid", "shopgoodwill", "craigslist", "poshmark"]
OUTNAME = sys.argv[4] if len(sys.argv) > 4 else "labels.jsonl"

def pick():
    m = json.load(open(os.path.join(H, "MANIFEST.json"), encoding="utf-8"))
    out = []
    for s in SITES:
        d = collections.defaultdict(list)
        for e in m:
            if e["site"] == s and e["split"] == "train":
                d[e["listingId"]].append(e)
        multi = sorted((v for v in d.values() if len(v) >= 2), key=lambda v: (-max(max(e["width"], e["height"]) for e in v), v[0]["listingId"]))
        for v in multi[:CAND]:
            for e in sorted(v, key=lambda e: e["photo"])[:PHOTOS]:
                out.append((s, e["listingId"], e["photo"], os.path.join(H, s, "train", e["listingId"], f"photo-{e['photo']}.jpg")))
    return out

def main():
    items = pick()
    for it in items:
        V3.assert_allowed_path(it[3])
        assert os.path.exists(it[3]), it
    print(len(items), "photos", flush=True)
    rows = []
    with mp.Pool(8, initializer=V3.init_ctx3) as pool:
        for k, (path, n, r, dt) in enumerate(pool.imap(V3._work3, [i[3] for i in items]), 1):
            s, lid, ph, _ = items[k - 1]
            for x in r:
                x.update(site=s, listing=lid, photo_n=ph)
            rows += r
            if k % 20 == 0:
                print(k, len(rows), flush=True)
    with open(os.path.join(ROOT, "data", "reports", "rendition", OUTNAME), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(collections.Counter((r["site"], r["level"]) for r in rows))

if __name__ == "__main__":
    mp.freeze_support(); main()
