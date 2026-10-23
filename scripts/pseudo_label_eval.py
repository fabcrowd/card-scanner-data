"""Label HELD-OUT eval photos (data/pseudo-labels/eval/photos.json) with the v3 detect + geometric pipeline, for EVALUATION only.

Same per-box code as pseudo_label_v3_hibid.py (V2.label_box_v2 with v3 groups, listing-level suspect_ja flag), but
without v3's eval-path refusal. Output: data/pseudo-labels/eval/<pack>.jsonl. Never read as training input."""
import json
import multiprocessing as mp
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pseudo_label as P  # noqa: E402
import pseudo_label_v2 as V2  # noqa: E402
import pseudo_label_v3 as V3  # noqa: E402

ROOT = V3.ROOT
OUT = os.path.join(ROOT, "data", "pseudo-labels", "eval")


def _work(rel):
    path = os.path.join(ROOT, rel)
    assert "sealed-test" not in path.lower()
    with Image.open(path) as im:
        img = im.convert("RGB")
    rows = []
    for d in P.D.detect_cards(img, P._ctx["detector"]):
        b = d["box"]
        if b[3] - b[1] < P.MIN_HEIGHT:
            continue
        r = V2.label_box_v2(img, b)
        r["photo"] = rel
        r["det_score"] = round(d["score"], 3)
        rows.append(r)
    return rel, rows


def main():
    info = json.load(open(os.path.join(OUT, "photos.json")))
    for src, e in info["sources"].items():
        photos = e["kept"]
        t0 = time.time()
        rows = []
        with mp.Pool(8, initializer=V3.init_ctx3) as pool:
            for k, (rel, r) in enumerate(pool.imap(_work, photos), 1):
                rows += r
                if k % 25 == 0 or k == len(photos):
                    print(src, f"[{k}/{len(photos)}] rows {len(rows)} {time.time()-t0:.0f}s", flush=True)
        by = {}
        for r in rows:
            by.setdefault(os.path.basename(os.path.dirname(r["photo"])), []).append(r)
        final = []
        for lid in sorted(by):
            flag, _ = V3.suspect_ja("", by[lid])  # no titles read: geometry-based flag only
            final += [V3.finalize_row(r, lid, flag) for r in by[lid]]
        with open(os.path.join(OUT, f"{src}.jsonl"), "w", encoding="utf-8") as fh:
            for r in final:
                fh.write(json.dumps(r) + "\n")
        from collections import Counter
        print(src, "photos", len(photos), "rows", len(final), Counter(r["level"] for r in final), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
