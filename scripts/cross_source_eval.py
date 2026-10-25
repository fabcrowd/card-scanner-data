"""Cross-source held-out eval: shipped vs B vs F on pseudo-labelled eval crops at native and degraded heights.

Reads data/pseudo-labels/eval/<src>.jsonl (id-level rows = ground truth), embeds each crop as the extension does
(crop_card stretch to 750x1050 -> ClipEmbedder), top-5 against the shipped catalog (digital ids skipped), fusion_rule.classify.
Downscale = BOX resize of the box region to height h + JPEG q90 (large_crop_check.degrade), only when native height >= h.
Writes data/reports/cross-source/results.json (per-cell metrics) and boxes.json (per-box reads). Tunes nothing."""
import io
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402
from run_real_photos import DIGITAL_RE, crop_card, load_names, to_png  # noqa: E402
from score_real_labels import base_name  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MODELS = {"shipped": os.path.join(ROOT, "assets", "clip_visual_int8.onnx"),
          "B": os.path.join(ROOT, "train", "out", "B", "raw-final-int8.onnx"),
          "F": os.path.join(ROOT, "train", "out", "fastC_real", "epoch8-ema-int8.onnx"),
          "E": os.path.join(ROOT, "train", "out", "fastE_all", "epoch16-ema-int8.onnx"),
          "G1": os.path.join(ROOT, "train", "out", "fastG1_all", "epoch16-ema-int8.onnx")}
HEIGHTS = [120, 150, 200, 250]
SRCS = ["hibid", "goodwill-pack2"]
OUT = os.environ.get("CROSS_OUT") or os.path.join(ROOT, "data", "reports", "cross-source")  # G1 run sets CROSS_OUT
LABELS = os.path.join(ROOT, "data", "pseudo-labels", "eval")


def degrade_png(img, box, h):
    x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
    x2, y2 = round(box[2]), round(box[3])
    reg = img.crop((x1, y1, x2, y2))
    w = max(1, round(reg.width * h / reg.height))
    buf = io.BytesIO()
    reg.resize((w, h), Image.BOX).save(buf, "JPEG", quality=90)
    buf.seek(0)
    small = Image.open(buf).convert("RGB")
    return to_png(crop_card(small, [0, 0, small.width, small.height]))


def main():
    os.makedirs(OUT, exist_ok=True)
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    nm = load_names()
    name_of = [nm.get(c) for c in cat.ids]
    id_name = {c: n for c, n in zip(cat.ids, name_of)}
    is_dig = np.array([bool(DIGITAL_RE.match(c)) for c in cat.ids])
    _, groups, _ = __import__("pseudo_label_v3").load_groups_v3()
    id_to_group = {c: g for g, mem in groups.items() for c in mem}
    embs = {m: ClipEmbedder(p) for m, p in MODELS.items()}
    boxes = []
    for src in SRCS:
        rows = [json.loads(l) for l in open(os.path.join(LABELS, f"{src}.jsonl"), encoding="utf-8")]
        gt = [r for r in rows if r["level"] == "id"]
        by = {}
        for r in gt:
            by.setdefault(r["photo"], []).append(r)
        print(src, "photos", len({r['photo'] for r in rows}), "rows", len(rows), "id-level", len(gt), flush=True)
        for k, (photo, rs) in enumerate(sorted(by.items()), 1):
            with Image.open(os.path.join(ROOT, photo)) as im:
                img = im.convert("RGB")
            for r in rs:
                box, H = r["box"], int(round(r["box_height"]))
                gid = r["train_id"]
                for h in ["native"] + HEIGHTS:
                    if h != "native" and H < h:
                        continue
                    data = to_png(crop_card(img, box)) if h == "native" else degrade_png(img, box, h)
                    hh = H if h == "native" else h
                    rec = {"src": src, "photo": photo, "box": box, "native_h": H, "cell": h, "h": hh,
                           "gt_id": gid, "gt_name": id_name.get(gid), "reads": {}}
                    for m, emb in embs.items():
                        idx, sims = cat.top_k_batch(emb(data, trusted=True)[None], k=30)
                        pairs = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if not is_dig[j]][:5]
                        ids = [cat.ids[j] for j, _ in pairs]
                        nms = [name_of[j] for j, _ in pairs]
                        c = classify([s for _, s in pairs], ids, nms, hh)
                        rec["reads"][m] = {"ids": ids, "names": nms, "cos": [round(s, 4) for _, s in pairs],
                                           "label": c.label, "accepted": c.accepted, "chosen": c.id,
                                           "chosen_name": id_name.get(c.id), "reason": c.reason}
                    boxes.append(rec)
            if k % 20 == 0:
                print(src, k, "/", len(by), len(boxes), flush=True)
    json.dump(boxes, open(os.path.join(OUT, "boxes.json"), "w"))
    # metrics
    def nrm(n):
        return base_name(n or "")
    res = {}
    for src in SRCS + ["ALL"]:
        for m in MODELS:
            cells = {}
            for b in boxes:
                if src != "ALL" and b["src"] != src:
                    continue
                keys = [str(b["cell"])]
                if b["cell"] == "native":
                    keys = ["native<200" if b["native_h"] < 200 else "native>=200"]
                    keys.append("native_all")
                for key in keys:
                    cells.setdefault(key, []).append(b)
            for key, bs in cells.items():
                n = len(bs)
                t1 = t5 = acc = accr = sure = surew = accw_id = 0
                top1_ids = Counter()
                for b in bs:
                    rd = b["reads"][m]
                    g = nrm(b["gt_name"])
                    names = [nrm(x) for x in rd["names"]]
                    t1 += names[0] == g
                    t5 += g in names
                    top1_ids[rd["ids"][0]] += 1
                    if rd["accepted"]:
                        acc += 1
                        right = nrm(rd["chosen_name"]) == g
                        accr += right
                        accw_id += id_to_group.get(rd["chosen"], rd["chosen"]) != id_to_group.get(b["gt_id"], b["gt_id"])
                        if rd["label"] == "sure":
                            sure += 1
                            surew += not right
                common = top1_ids.most_common(5)
                res.setdefault(src, {}).setdefault(m, {})[key] = {
                    "n": n, "top1_name": t1, "top5_name": t5, "accepted": acc, "accepted_right": accr,
                    "accepted_wrong": acc - accr, "sure": sure, "sure_wrong": surew, "accepted_wrong_idlevel": accw_id,
                    "hub_top1_share": round(common[0][1] / n, 3), "hub_top5_share": round(sum(c for _, c in common) / n, 3)}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("done", len(boxes))


if __name__ == "__main__":
    main()
