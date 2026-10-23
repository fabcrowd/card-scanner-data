"""Compare real-photo results of student models against the shipped baseline; build a contact sheet of changed top-1s."""
import json
import os
import sys
from collections import defaultdict

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_real_photos import GROUPS, _font, bucket_of, crop_card, BUCKETS  # noqa: E402

ROOT = os.path.dirname(HERE)
REP = os.path.join(ROOT, "data", "reports")
PHOTOS = os.path.join(ROOT, "data", "fb-eval-pack", "user-saved")
IMG_DIR = r"C:\repos\projects\auto card lister\data\card-catalog\images"
MODELS = {"B_ema4": "real-student-B-ema4", "B_raw": "real-student-B-raw", "wise_0.5": "real-student-wise-0.5"}
BEST = "B_raw"
MAX_BOXES = 24
SCALE = 3


def load(d):
    return json.load(open(os.path.join(REP, d, "results.json"), encoding="utf-8"))


def key(r):
    return r["photo"], tuple(r["box"])


def art(i):
    p = os.path.join(IMG_DIR, f"{i}.webp")
    return Image.open(p).convert("RGB") if os.path.exists(p) else None


def main():
    base = load("real-baseline-shipped")
    bidx = {key(r): r for r in base["boxes"]}
    out = {"baseline_summary": base["summary"]}
    res = {n: load(d) for n, d in MODELS.items()}
    for n, rs in res.items():
        s = rs["summary"]
        cmp = {}
        for g, lo, hi in GROUPS:
            same = tot = 0
            for r in rs["boxes"]:
                b = bidx.get(key(r))
                if b is None or not (lo <= r["height"] < hi) or not r["top5"] or not b["top5"]:
                    continue
                tot += 1
                same += r["top5"][0]["id"] == b["top5"][0]["id"]
            cmp[g] = {"n": tot, "agree": round(same / tot, 3) if tot else None}
        missing = sum(key(r) not in bidx for r in rs["boxes"])
        out[n] = {"buckets": s["buckets"], "hubs": s["hubs"], "agreement": cmp, "unmatched_boxes": missing}
    with open(os.path.join(REP, "real-compare", "compare.json"), "w", encoding="utf-8") if os.makedirs(os.path.join(REP, "real-compare"), exist_ok=True) is None else None as f:
        json.dump(out, f, indent=1)
    # contact sheet
    files = sorted(f for f in os.listdir(PHOTOS) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    pref = {files[2]: 0, files[4]: 0}  # photos 3 and 5
    changed = []
    for r in res[BEST]["boxes"]:
        b = bidx.get(key(r))
        if b and r["top5"] and b["top5"] and r["top5"][0]["id"] != b["top5"][0]["id"]:
            changed.append((r, b))
    changed.sort(key=lambda t: (t[0]["height"] >= 150, t[0]["photo"] not in pref, t[0]["photo"], t[0]["box"]))
    sel = changed[:MAX_BOXES]
    font = _font(15)
    tiles = []
    imgs = {}
    for r, b in sel:
        if r["photo"] not in imgs:
            imgs[r["photo"]] = Image.open(os.path.join(PHOTOS, r["photo"])).convert("RGB")
        crop = crop_card(imgs[r["photo"]], r["box"])
        ch = r["height"] * SCALE
        cw = max(40, round(ch * 750 / 1050))
        crop = crop.resize((cw, ch), Image.BICUBIC)
        panel_h = 230
        a_b, a_s = art(b["top5"][0]["id"]), art(r["top5"][0]["id"])
        arts = []
        for a in (a_b, a_s):
            arts.append(a.resize((round(panel_h * 0.714), panel_h)) if a else Image.new("RGB", (round(panel_h * 0.714), panel_h), (200, 200, 200)))
        crop_disp = crop if crop.height <= panel_h else crop.resize((round(crop.width * panel_h / crop.height), panel_h))
        w = crop_disp.width + sum(x.width for x in arts) + 40
        tile = Image.new("RGB", (max(w, 560), panel_h + 50), (255, 255, 255))
        d = ImageDraw.Draw(tile)
        d.text((4, 2), f"base: {b['top5'][0]['name']} {b['top5'][0]['cos']:.3f} | student: {r['top5'][0]['name']} {r['top5'][0]['cos']:.3f}", fill=(0, 0, 0), font=font)
        d.text((4, 24), f"{r['photo'][:12]} h={r['height']} crop | base art | student art", fill=(90, 90, 90), font=font)
        x = 4
        for im in [crop_disp] + arts:
            tile.paste(im, (x, 46))
            x += im.width + 10
        tiles.append(tile)
    cols = 2
    cw_ = max(t.width for t in tiles)
    rows_h = [max(t.height for t in tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    sheet = Image.new("RGB", (cw_ * cols, sum(rows_h)), (255, 255, 255))
    y = 0
    for ri, i in enumerate(range(0, len(tiles), cols)):
        for ci, t in enumerate(tiles[i:i + cols]):
            sheet.paste(t, (ci * cw_, y))
        y += rows_h[ri]
    sheet.save(os.path.join(REP, "real-compare", "sheet-changed.jpg"), "JPEG", quality=90)
    out["sheet"] = {"changed_total": len(changed), "shown": len(sel), "shown_small": sum(r["height"] < 150 for r, _ in sel)}
    json.dump(out, open(os.path.join(REP, "real-compare", "compare.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
