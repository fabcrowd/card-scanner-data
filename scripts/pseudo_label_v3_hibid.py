"""Run the v3 detect + pseudo-label pipeline on HiBid TRAIN photos only (data/hibid-pack/train).

HiBid MANIFEST uses 'lotId' (v3's load_manifest_titles reads 'itemId'); everything else is v3's code unchanged.
Also records every detection's height (the v3 rule still drops boxes < MIN_HEIGHT before labelling)."""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pseudo_label as P  # noqa: E402
import pseudo_label_v3 as V3  # noqa: E402
import pseudo_label_v2 as V2  # noqa: E402

ROOT = V3.ROOT
PACK = os.path.join(ROOT, "data", "hibid-pack")
TRAIN = os.path.join(PACK, "train")


def _work(path):
    t = time.time()
    V3.assert_allowed_path(path)
    with Image.open(path) as im:
        img = im.convert("RGB")
    dets = P.D.detect_cards(img, P._ctx["detector"])
    heights = [d["box"][3] - d["box"][1] for d in dets]
    rows = []
    for d in dets:
        b = d["box"]
        if b[3] - b[1] < P.MIN_HEIGHT:
            continue
        r = V2.label_box_v2(img, b)
        r["photo"] = path
        r["det_score"] = round(d["score"], 3)
        rows.append(r)
    return path, heights, rows, time.time() - t


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    with open(os.path.join(PACK, "MANIFEST.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    titles = {str(e["lotId"]): e.get("title", "") for e in man if e.get("split") == "train"}
    ready, recent, nomf = V3.listing_folders(TRAIN, titles)
    print(f"{len(ready)} ready, {len(recent)} recent, {len(nomf)} no-manifest", flush=True)
    photos = [p for f in ready for p in V3.photos_of(os.path.join(TRAIN, f))]
    t0 = time.time()
    rows, heights = [], []
    with mp.Pool(8, initializer=V3.init_ctx3) as pool:
        for k, (path, hs, r, dt) in enumerate(pool.imap(_work, photos), 1):
            rows += r
            heights += hs
            if k % 25 == 0 or k == len(photos):
                print(f"[{k}/{len(photos)}] rows {len(rows)}", flush=True)
    by = {}
    for r in rows:
        by.setdefault(os.path.basename(os.path.dirname(r["photo"])), []).append(r)
    flagged, final = {}, []
    for lid in sorted(by):
        flag, why = V3.suspect_ja(titles.get(lid, ""), by[lid])
        if flag:
            flagged[lid] = why
        final += [V3.finalize_row(r, lid, flag) for r in by[lid]]
    os.makedirs(V3.OUT_V3, exist_ok=True)
    with open(os.path.join(V3.OUT_V3, "hibid-pack-train.jsonl"), "w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r) + "\n")
    summ = V3.summarize_pack("hibid-pack-train", len(photos), len(heights), final, flagged, time.time() - t0)
    summ["skipped_recent"] = recent
    summ["skipped_no_manifest"] = nomf
    summ["detection_height_hist_all"] = {"<150": sum(h < 150 for h in heights),
                                          "150-299": sum(150 <= h < 300 for h in heights),
                                          ">=300": sum(h >= 300 for h in heights)}
    with open(os.path.join(V3.OUT_V3, "hibid-pack-train-summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))
    # input hygiene
    inp = {sha(p) for p in photos}
    ev = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(PACK, "eval")) for f in fs]
    fb = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ROOT, "data", "fb-eval-pack")) for f in fs]
    print("inputs", len(photos), "eval files", len(ev), "fb files", len(fb))
    print("eval overlap", len(inp & {sha(p) for p in ev}), "fb overlap", len(inp & {sha(p) for p in fb}))
    print("any input path with eval:", any("eval" in p.lower().split(os.sep) for p in photos))


if __name__ == "__main__":
    mp.freeze_support()
    main()
