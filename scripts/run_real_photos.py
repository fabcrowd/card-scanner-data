"""Score real photos through detector -> art crop -> CLIP -> frozen catalog -> fusion rule. Reusable, any int8 model.

Per box: detect (detect_cards.py), crop the box and stretch to 750x1050 as rectify.js, art crop + normalise as
pipeline/embed.py (lib/embed.js), top-5 cosine against the catalog, scripts/fusion_rule.classify at the box height.
Digital-only (TCG Pocket) ids are skipped in the search (top-30 fetched, digital dropped, 5 kept); the extension's
own handling of them is not verified here.
"""
import argparse
import io
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cards import DEFAULT_MODEL, detect_cards, detect_tiled  # noqa: E402
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402

CARD_DIR = r"C:\repos\projects\auto card lister\data\card-catalog"
OUT_W, OUT_H = 750, 1050  # rectify.js
BUCKETS = [("<100", 0, 100), ("100-149", 100, 150), ("150-199", 150, 200), ("200-299", 200, 300), (">=300", 300, 10**9)]
GROUPS = [("<150", 0, 150), (">=150", 150, 10**9)]
FETCH_K = 30
TOP = 5
# Same rule as train/pilot_data.py DIGITAL_RE (TCG Pocket ids).
import re  # noqa: E402

DIGITAL_RE = re.compile(r"^(?:[AB]\d+[a-z]?|P-[AB])-")
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".webp")


def load_names():
    names = {}
    for f in ("meta.json", "meta.full.json"):  # full last = preferred
        with open(os.path.join(CARD_DIR, f), encoding="utf-8") as fh:
            for m in json.load(fh):
                if m.get("name"):
                    names[m["id"]] = m["name"]
    return names


def crop_card(img, box):
    x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
    x2, y2 = round(box[2]), round(box[3])
    return img.crop((x1, y1, x2, y2)).resize((OUT_W, OUT_H), Image.BILINEAR)


def to_png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def bucket_of(h, table):
    for label, lo, hi in table:
        if lo <= h < hi:
            return label
    return table[-1][0]


def score_photo(path, emb, cat, name_of, is_dig, tiles=False, detector=DEFAULT_MODEL):
    with Image.open(path) as im:
        img = im.convert("RGB")
    rows = []
    for d in (detect_tiled(img, detector) if tiles else detect_cards(img, detector)):
        box = d["box"]
        h = int(round(box[3] - box[1]))
        q = emb(to_png(crop_card(img, box)), trusted=True)
        idx, sims = cat.top_k_batch(q[None], k=FETCH_K)
        pairs = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if not is_dig[j]][:TOP]
        ids = [cat.ids[j] for j, _ in pairs]
        nms = [name_of[j] for j, _ in pairs]
        cos = [s for _, s in pairs]
        r = classify(cos, ids, nms, h)
        rows.append({"box": [round(v, 1) for v in box], "height": h, "det_score": round(d["score"], 3),
                     "top5": [{"id": i, "name": n, "cos": round(c, 4)} for i, n, c in zip(ids, nms, cos)],
                     "accepted": r.accepted, "label": r.label, "reason": r.reason, "chosen_id": r.id})
    return img, rows


def summarize(records, name_of=None):
    """records: flat list of per-box dicts (with 'height', 'top5', 'accepted', 'label')."""
    name_of = name_of or {}
    summary = {"photos": len({r["photo"] for r in records}), "boxes": len(records), "buckets": {}, "hubs": {}}
    for label, lo, hi in BUCKETS:
        sel = [r for r in records if lo <= r["height"] < hi]
        cos = [r["top5"][0]["cos"] for r in sel if r["top5"]]
        summary["buckets"][label] = {"boxes": len(sel), "accepted": sum(r["accepted"] for r in sel),
                                     "sure": sum(r["label"] == "sure" for r in sel),
                                     "mean_top1_cos": round(float(np.mean(cos)), 4) if cos else None}
    for label, lo, hi in GROUPS:
        top1 = [r["top5"][0]["id"] for r in records if r["top5"] and lo <= r["height"] < hi]
        cnt = Counter(top1)
        n = len(top1)
        common = cnt.most_common(5)
        summary["hubs"][label] = {
            "n_top1": n,
            "top1_share": round(common[0][1] / n, 3) if n else None,
            "top5_share": round(sum(c for _, c in common) / n, 3) if n else None,
            "top5_ids": [{"id": i, "name": name_of.get(i), "count": c} for i, c in common]}
    return summary


def _font(size):
    for p in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_sheet(img, rows, title, path):
    scale = 2 if max(img.size) <= 1200 else 1
    big = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    dr = ImageDraw.Draw(big)
    f = _font(max(14, 11 * scale))
    for n, r in enumerate(rows, 1):
        b = [v * scale for v in r["box"]]
        dr.rectangle(b, outline=(255, 0, 0), width=2)
        dr.rectangle([b[0], b[1], b[0] + 8 * scale + 8 * len(str(n)) * scale / 2, b[1] + 13 * scale], fill=(255, 0, 0))
        dr.text((b[0] + 2, b[1]), str(n), fill=(255, 255, 255), font=f)
    pf = _font(15)
    lines = [title] + [""]
    for n, r in enumerate(rows, 1):
        t = r["top5"][0] if r["top5"] else {"name": "-", "cos": 0}
        lines.append(f"{n:2d} h={r['height']:3d} {t['cos']:.3f} {r['label'][:5]:5s} {t['name']}")
    line_h = 20
    panel = Image.new("RGB", (620, max(big.height, line_h * (len(lines) + 1))), (255, 255, 255))
    pd = ImageDraw.Draw(panel)
    for i, ln in enumerate(lines):
        pd.text((8, 6 + i * line_h), ln, fill=(0, 0, 0), font=pf)
    sheet = Image.new("RGB", (big.width + panel.width, max(big.height, panel.height)), (255, 255, 255))
    sheet.paste(big, (0, 0))
    sheet.paste(panel, (big.width, 0))
    sheet.save(path, "JPEG", quality=92)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "assets", "clip_visual_int8.onnx"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog", default=os.path.join(ROOT, "assets", "catalog.bin.gz"))
    ap.add_argument("--detector", default=DEFAULT_MODEL, help="detector onnx (default: verified shipped v2.1)")
    ap.add_argument("--tiles", action="store_true", help="detect on 2x2+3x3 tiles (opt-in; the extension does this only for hero photos)")
    a = ap.parse_args(argv)
    cat = Catalog.load(a.catalog, os.path.join(ROOT, "assets", "id_map.json"))
    emb = ClipEmbedder(a.model)
    nm = load_names()
    name_of = [nm.get(c) for c in cat.ids]
    is_dig = np.array([bool(DIGITAL_RE.match(c)) for c in cat.ids])
    os.makedirs(os.path.join(a.out, "sheets"), exist_ok=True)
    files = sorted(f for f in os.listdir(a.photos) if f.lower().endswith(PHOTO_EXT))
    records = []
    for n, f in enumerate(files, 1):
        img, rows = score_photo(os.path.join(a.photos, f), emb, cat, name_of, is_dig, a.tiles, a.detector)
        for r in rows:
            r["photo"] = f
        records.extend(rows)
        draw_sheet(img, rows, f"sheet-{n}: {f[:24]}  boxes={len(rows)}", os.path.join(a.out, "sheets", f"sheet-{n}.jpg"))
        print(f"sheet-{n}: {f} {img.size} boxes={len(rows)}")
    summary = summarize(records, nm)
    summary["model"] = a.model
    summary["tiles"] = a.tiles
    summary["detector"] = a.detector
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "boxes": records}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
