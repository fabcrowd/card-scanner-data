"""Score the synthetic small-crop set through the fusion accept rule (views a and c). Synthetic only."""
import json
import os
import random
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_synth_small as E  # noqa: E402
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402

CARD_DIR = r"C:\repos\projects\auto card lister\data\card-catalog"
K = 5


def load_names():
    names = {}
    for f in ("meta.json", "meta.full.json"):  # full last = preferred
        with open(os.path.join(CARD_DIR, f), encoding="utf-8") as fh:
            for m in json.load(fh):
                if m.get("name"):
                    names[m["id"]] = m["name"]
    return names


def main():
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    emb = ClipEmbedder(os.path.join(ROOT, "assets", "clip_visual_int8.onnx"))
    nm = load_names()
    name_of = [nm.get(c) for c in cat.ids]
    have_names = sum(1 for n in name_of if n)
    have = [i for i, cid in enumerate(cat.ids) if E.image_path(cid)]
    rows = random.Random(E.SEED).sample(have, E.N)
    rng = random.Random(E.SEED + 1)
    fulls = []
    for r in rows:
        with Image.open(E.image_path(cat.ids[r])) as im:
            fulls.append(im.convert("RGB"))
    nameless = sum(1 for r in rows if not name_of[r])
    print(f"names: {have_names}/{len(cat.ids)} ids ({100 * have_names / len(cat.ids):.1f}%); eval cards without name: {nameless}/{E.N}")
    print("SYNTHETIC; view a = fuseUntiered accepted (no height); view c = capSmallSource at synth height")
    print("h | n | top1 | a:acc | a:right | a:WRONG | a:notacc | high-precap | c:ident (right/WRONG) | c:guess | abstain")
    out = []
    for h in E.HEIGHTS:
        fresh = np.stack([emb(E.degrade(img, h, rng), trusted=True) for img in fulls])
        idx, sims = cat.top_k_batch(fresh, k=K)
        t = {k: 0 for k in ("top1", "acc", "accR", "high", "idn", "idnR", "gs", "ab")}
        for q, r in enumerate(rows):
            ids = [cat.ids[j] for j in idx[q]]
            nms = [name_of[j] for j in idx[q]]
            s = [float(x) for x in sims[q]]
            a = classify(s, ids, nms, 0)
            c = classify(s, ids, nms, h)
            truth = cat.ids[r]
            t["top1"] += ids[0] == truth
            if a.accepted:
                t["acc"] += 1
                t["accR"] += a.id == truth
                t["high"] += a.high
            else:
                t["ab"] += 1
            if c.label == "sure":
                t["idn"] += 1
                t["idnR"] += c.id == truth
            elif c.label == "guess":
                t["gs"] += 1
        row = {"height": h, "n": E.N, "top1": t["top1"] / E.N, "accepted": t["acc"], "accepted_right": t["accR"],
               "accepted_wrong": t["acc"] - t["accR"], "not_accepted": t["ab"], "high_before_cap": t["high"],
               "c_identified": t["idn"], "c_identified_right": t["idnR"], "c_identified_wrong": t["idn"] - t["idnR"],
               "c_guess": t["gs"], "c_abstain": t["ab"]}
        out.append(row)
        print(f"{h:3d} | {E.N} | {row['top1']:.3f} | {t['acc']} | {t['accR']} | {row['accepted_wrong']} | {t['ab']} | "
              f"{t['high']} | {t['idn']} ({t['idnR']}/{row['c_identified_wrong']}) | {t['gs']} | {t['ab']}")
    path = os.path.join(ROOT, "data", "reports", "synth-fusion-baseline.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "seed": E.SEED, "names_covered": have_names, "ids": len(cat.ids),
                   "eval_cards_without_name": nameless, "results": out}, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
