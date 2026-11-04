"""int8 export + synthetic small-crop eval (eval_synth_small / score_synth_fusion logic) + drift for the pilot."""
import argparse
import collections
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_data as D  # noqa: E402
E = D.E
from fusion_rule import classify  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import preprocess  # noqa: E402
import score_synth_fusion as S  # noqa: E402

ROOT = D.ROOT
OUT = os.path.join(ROOT, "train", "out")
SHIPPED = os.path.join(ROOT, "assets", "clip_visual_int8.onnx")
PARITY_N, PARITY_SEED = 300, 9001
BATCH = 1  # dynamic int8 scales activations per batch; deployed path is batch 1
K = 5


class VisualEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.encode_image(x)
        return features / features.norm(dim=-1, keepdim=True)


def export_int8(model, fp32_path, int8_path):
    """export_clip_onnx.py settings (legacy exporter, opset 14, QUInt8 dynamic quantisation), as export_parity.py."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = model.cpu().float().eval()
    torch.onnx.export(
        VisualEncoder(model), torch.zeros(1, 3, 224, 224, dtype=torch.float32), fp32_path,
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=14, dynamo=False,
    )
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
    os.remove(fp32_path)


class Context:
    def __init__(self):
        self.cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
        nm = S.load_names()
        self.name_of = [nm.get(c) for c in self.cat.ids]
        self.have, self.rows = D.eval_rows(self.cat)
        self.digital = np.array([D.is_digital(self.cat.ids[r]) for r in self.rows])
        self.parity_rows = random.Random(PARITY_SEED).sample(self.have, PARITY_N)
        self.jpegs = self._evalset()

    def _evalset(self):
        """Degraded JPEGs per height; same rng stream as eval_synth_small.py / score_synth_fusion.py (seed 9001+1)."""
        cache = os.path.join(OUT, "evalset.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return pickle.load(f)
        rng = random.Random(E.SEED + 1)
        fulls = []
        for r in self.rows:
            with Image.open(E.image_path(self.cat.ids[r])) as im:
                fulls.append(im.convert("RGB"))
        jp = {h: [E.degrade(img, h, rng) for img in fulls] for h in E.HEIGHTS}
        os.makedirs(OUT, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(jp, f)
        return jp


def embed_jpegs(sess, jpegs):
    name = sess.get_inputs()[0].name
    out = []
    for s in range(0, len(jpegs), BATCH):
        x = np.concatenate([preprocess(b, trusted=True) for b in jpegs[s:s + BATCH]])
        v = sess.run(None, {name: x})[0].astype(np.float32)
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(out)


def _height_stats(ctx, fresh, mask):
    cat, rows = ctx.cat, np.array(ctx.rows)
    sel = np.flatnonzero(mask)
    idx, sims = cat.top_k_batch(fresh[sel], k=K)
    n = len(sel)
    truth = rows[sel]
    top1 = int((idx[:, 0] == truth).sum())
    top5 = int((idx == truth[:, None]).any(axis=1).sum())
    acc = accR = 0
    for q in range(n):
        ids = [cat.ids[j] for j in idx[q]]
        nms = [ctx.name_of[j] for j in idx[q]]
        a = classify([float(x) for x in sims[q]], ids, nms, 0)
        if a.accepted:
            acc += 1
            accR += a.id == cat.ids[truth[q]]
    counts = collections.Counter(idx[:, 0].tolist()).most_common(5)
    return {"n": n, "top1": top1 / n, "top5": top5 / n, "accepted": acc, "accepted_right": int(accR),
            "accepted_wrong": acc - int(accR), "hub1_share": counts[0][1] / n,
            "hub5_share": sum(c for _, c in counts) / n}


def eval_int8(onnx_path, ctx):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    heights = {}
    allm = np.ones(len(ctx.rows), dtype=bool)
    for h in E.HEIGHTS:
        fresh = embed_jpegs(sess, ctx.jpegs[h])
        heights[h] = {"all": _height_stats(ctx, fresh, allm), "nodigital": _height_stats(ctx, fresh, ~ctx.digital)}
        heights[h]["median_cos"] = float(np.median(np.einsum("ij,ij->i", fresh, ctx.cat.vectors[ctx.rows])))
    pj = []
    for r in ctx.parity_rows:
        with open(E.image_path(ctx.cat.ids[r]), "rb") as f:
            pj.append(f.read())
    fresh = embed_jpegs(sess, pj)
    cos = np.einsum("ij,ij->i", fresh, ctx.cat.vectors[ctx.parity_rows])
    nearest = (fresh @ ctx.cat.vectors.T).argmax(axis=1)
    drift = {"clean_median_cos": float(np.median(cos)),
             "self_match": float((nearest == np.array(ctx.parity_rows)).mean())}
    return {"heights": heights, "drift": drift}


def table(m, base=None):
    lines = ["  H | top1  top5 | acc  right WRONG | hub1  hub5 | nodig top1 acc WRONG hub1" + ("  || base top1 WRONG" if base else "")]
    for h in E.HEIGHTS:
        hm = m["heights"][h] if h in m["heights"] else m["heights"][str(h)]  # JSON round-trip makes keys str
        a, d = hm["all"], hm["nodigital"]
        s = (f"{h:4d}| {a['top1']:.3f} {a['top5']:.3f} | {a['accepted']:4d} {a['accepted_right']:4d} {a['accepted_wrong']:4d} | "
             f"{a['hub1_share']:.3f} {a['hub5_share']:.3f} | {d['top1']:.3f} {d['accepted']:4d} {d['accepted_wrong']:4d} {d['hub1_share']:.3f}")
        if base:
            b = base["heights"][str(h)]["all"] if str(h) in base["heights"] else base["heights"][h]["all"]
            s += f"  || {b['top1']:.3f} {b['accepted_wrong']:4d}"
        lines.append(s)
    dr = m["drift"]
    lines.append(f"  drift: clean median cos {dr['clean_median_cos']:.4f} | self-match {dr['self_match']:.4f}"
                 + (f" | cos(S,T) large {dr['cos_ST_large_median']:.4f}" if "cos_ST_large_median" in dr else ""))
    return "\n".join(lines)


def main():
    """Baseline: evaluate the shipped int8 asset through this exact harness."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=SHIPPED)
    ap.add_argument("--out", default=os.path.join(OUT, "baseline", "metrics.json"))
    a = ap.parse_args()
    t0 = time.time()
    ctx = Context()
    m = eval_int8(a.onnx, ctx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(table(m))
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
