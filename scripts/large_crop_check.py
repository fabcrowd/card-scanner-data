"""Large-crop regression check: shipped int8 vs candidate B raw. CPU only, ORT batch 1. Tasks 1 and 2."""
import io
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from detect_cards import detect_cards  # noqa: E402
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402
from run_real_photos import DIGITAL_RE, crop_card, load_names, to_png, _font  # noqa: E402
from score_real_labels import base_name  # noqa: E402
import eval_synth_small as E  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # pipeline.embed lowers it; local trusted photos
MODELS = {"ship": os.path.join(ROOT, "assets", "clip_visual_int8.onnx"),
          "B": os.path.join(ROOT, "train", "out", "B", "raw-final-int8.onnx")}
PACK = os.path.join(ROOT, "data", "goodwill-pack")
OUT = os.path.join(ROOT, "data", "reports", "large-crop-check")
N, SEED = 2000, 4242
HEIGHTS = [None, 400, 300]  # None = full clean image


def degrade(img, h):
    w = max(1, round(img.width * h / img.height))
    buf = io.BytesIO()
    img.resize((w, h), Image.BOX).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def png_bytes_of(path):
    with open(path, "rb") as f:
        return f.read()


def task1(cat, name_of, embs):
    have = [i for i, c in enumerate(cat.ids) if E.image_path(c)]
    excluded = set(random.Random(E.SEED).sample(have, E.N))
    pool = [i for i in have if i not in excluded]
    rows = random.Random(SEED).sample(pool, N)
    assert not set(rows) & excluded
    res = {}
    for h in HEIGHTS:
        qs = {m: [] for m in embs}
        for r in rows:
            p = E.image_path(cat.ids[r])
            if h is None:
                data = png_bytes_of(p)
            else:
                with Image.open(p) as im:
                    data = degrade(im.convert("RGB"), h)
            for m, emb in embs.items():
                qs[m].append(emb(data, trusted=True))
        for m in embs:
            idx, sims = cat.top_k_batch(np.stack(qs[m]), k=5)
            t1 = tb = acc = accr = accrb = 0
            for q, r in enumerate(rows):
                ids = [cat.ids[j] for j in idx[q]]
                nms = [name_of[j] for j in idx[q]]
                t1 += idx[q][0] == r
                tb += bool(nms[0]) and base_name(nms[0] or "") == base_name(name_of[r] or "")
                a = classify([float(x) for x in sims[q]], ids, nms, 0)
                if a.accepted:
                    acc += 1
                    accr += a.id == cat.ids[r]
                    accrb += base_name(name_of[idx[q][list(ids).index(a.id)]] or "") == base_name(name_of[r] or "")
            res[f"{m}@{h or 'full'}"] = {"top1_id": t1 / N, "top1_base": tb / N, "accept_rate": acc / N,
                                         "accepted_prec_id": accr / acc if acc else None,
                                         "accepted_prec_base": accrb / acc if acc else None}
            print(m, h, res[f"{m}@{h or 'full'}"], flush=True)
    return res


def task2(cat, name_of, is_dig, embs):
    photos = sorted(os.path.join(d, f) for d, _, fs in os.walk(PACK) for f in fs if f.lower().endswith(".jpg"))
    recs = []
    for p in photos:
        with Image.open(p) as im:
            img = im.convert("RGB")
        for d in detect_cards(img):
            b = d["box"]
            h = int(round(b[3] - b[1]))
            if h < 200:
                continue
            data = to_png(crop_card(img, b))
            rec = {"photo": os.path.relpath(p, PACK), "box": [round(v, 1) for v in b], "height": h}
            for m, emb in embs.items():
                idx, sims = cat.top_k_batch(emb(data, trusted=True)[None], k=30)
                pairs = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if not is_dig[j]][:5]
                rec[m] = {"ids": [cat.ids[j] for j, _ in pairs], "names": [name_of[j] for j, _ in pairs],
                          "cos": [s for _, s in pairs]}
            rec["ship_read"] = classify(rec["ship"]["cos"], rec["ship"]["ids"], rec["ship"]["names"], 0)
            recs.append(rec)
        print("photo", p, len(recs), flush=True)
    return recs


def sheet(items, path):
    f = _font(13)
    cw, ch, pad = 200, 280, 6
    cols = 3
    rows = (len(items) + cols - 1) // cols
    cell_w = cw * 3 + pad * 4
    out = Image.new("RGB", (cols * cell_w, rows * (ch + 40)), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    for n, it in enumerate(items):
        x0, y0 = (n % cols) * cell_w, (n // cols) * (ch + 40)
        with Image.open(it["_photo"]) as im:
            crop = crop_card(im.convert("RGB"), it["box"]).resize((cw, ch))
        out.paste(crop, (x0 + pad, y0 + 36))
        for k, key in enumerate(("ship", "raw")):
            p = E.image_path(it[key + "_id"])
            if p:
                with Image.open(p) as im:
                    out.paste(im.convert("RGB").resize((cw, ch)), (x0 + pad + (k + 1) * (cw + pad), y0 + 36))
        dr.text((x0 + pad, y0 + 2), f"ship: {it['ship_name'][:34]} {it['ship_cos']:.3f}", fill=(0, 0, 0), font=f)
        dr.text((x0 + pad, y0 + 18), f"raw: {it['raw_name'][:36]} {it['raw_cos']:.3f} (h={it['height']})", fill=(180, 0, 0), font=f)
    out.save(path, "JPEG", quality=88)


def main():
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    nm = load_names()
    name_of = [nm.get(c) for c in cat.ids]
    is_dig = np.array([bool(DIGITAL_RE.match(c)) for c in cat.ids])
    embs = {"ship": ClipEmbedder(MODELS["ship"]), "B": ClipEmbedder(MODELS["B"])}
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("1", "all"):
        r1 = task1(cat, name_of, embs)
        with open(os.path.join(OUT, "task1.json"), "w", encoding="utf-8") as f:
            json.dump(r1, f, indent=1)
    if which in ("2", "all"):
        recs = task2(cat, name_of, is_dig, embs)
        n = len(recs)
        high = [r for r in recs if r["ship_read"].high]
        agree = lambda r: base_name(r["B"]["names"][0] or "") == base_name(r["ship"]["names"][0] or "")  # noqa: E731
        dis = [r for r in high if not agree(r)]
        out = []
        for r in dis:
            out.append({"photo": r["photo"], "box": r["box"], "height": r["height"],
                        "ship_top1": r["ship"]["names"][0], "ship_cos": r["ship"]["cos"][0], "ship_id": r["ship"]["ids"][0],
                        "raw_top1": r["B"]["names"][0], "raw_cos": r["B"]["cos"][0], "raw_id": r["B"]["ids"][0]})
        with open(os.path.join(OUT, "disagreements.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        items = [{"_photo": os.path.join(PACK, o["photo"]), "box": o["box"], "height": o["height"],
                  "ship_name": o["ship_top1"] or "?", "ship_cos": o["ship_cos"], "ship_id": o["ship_id"],
                  "raw_name": o["raw_top1"] or "?", "raw_cos": o["raw_cos"], "raw_id": o["raw_id"]} for o in out[:30]]
        if items:
            sheet(items, os.path.join(OUT, "sheet-disagree.jpg"))
        summ = {"n_boxes_ge200": n, "n_high_proxy": len(high),
                "agree_high": (len(high) - len(dis)) / len(high) if high else None,
                "agree_all": sum(agree(r) for r in recs) / n if n else None, "n_disagree_high": len(dis),
                "photos": len({r["photo"] for r in recs})}
        with open(os.path.join(OUT, "task2.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=1)
        print(summ)


if __name__ == "__main__":
    main()
