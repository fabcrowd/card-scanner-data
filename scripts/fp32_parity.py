"""Isolates int8 quantisation loss from preprocessing: embeds catalog art like embed_all.py (fp32 open_clip + PIL)."""
import os
import random
import sys
import time

import numpy as np
import onnxruntime as ort
import open_clip
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.catalog import Catalog  # noqa: E402

IMG_DIR = r"C:\repos\projects\auto card lister\data\card-catalog\images"
N = 300
SEED = 9001
BATCH = 50


def image_path(card_id):
    for ext in (".webp", ".png"):
        p = os.path.join(IMG_DIR, card_id + ext)
        if os.path.exists(p):
            return p
    return None


def load_preprocess(preprocess, path):
    """embed_all.py exactly: RGB, top 45% crop at full width, open_clip transform."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    image = image.crop((0, 0, w, max(1, int(h * 0.45))))
    return preprocess(image)


def stats(a):
    return f"min={a.min():.4f} p1={np.percentile(a, 1):.4f} p5={np.percentile(a, 5):.4f} median={np.median(a):.4f} mean={a.mean():.4f}"


def main():
    t0 = time.time()
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    have = [i for i, cid in enumerate(cat.ids) if image_path(cid)]
    rows = random.Random(SEED).sample(have, N)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    model.to(device)
    tensors = torch.stack([load_preprocess(preprocess, image_path(cat.ids[r])) for r in rows])
    feats = []
    with torch.no_grad():
        for s in range(0, N, BATCH):
            f = model.encode_image(tensors[s:s + BATCH].to(device))
            feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    fp32 = np.concatenate(feats).astype(np.float32)
    cos = np.einsum("ij,ij->i", fp32, cat.vectors[rows])
    nearest = (fp32 @ cat.vectors.T).argmax(axis=1)
    self_match = float((nearest == np.array(rows)).mean())
    print(f"device={device} torch={torch.__version__} sampled {N} of {len(have)} with image; catalog rows {len(cat.ids)}")
    print(f"fp32 torch vs catalog: {stats(cos)}")
    print(f"frac cos>=0.999: {(cos >= 0.999).mean():.4f}  top-1 self-match: {self_match:.4f}")

    sess = ort.InferenceSession(os.path.join(ROOT, "assets", "clip_visual_int8.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    i8 = []
    for t in tensors.numpy():
        v = sess.run(None, {name: t[None].astype(np.float32)})[0].reshape(-1)
        i8.append(v / np.linalg.norm(v))
    i8 = np.stack(i8).astype(np.float32)
    c2 = np.einsum("ij,ij->i", i8, fp32)
    print(f"cos(int8, fp32 torch), same tensors: min={c2.min():.4f} median={np.median(c2):.4f} mean={c2.mean():.4f}")
    med = float(np.median(cos))
    print("CATALOG IS FP32 open_clip (median cos >= 0.999)" if med >= 0.999 else f"CATALOG NOT MATCHED by fp32 open_clip: median cos={med:.4f}, min={cos.min():.4f}, frac>=0.999={(cos >= 0.999).mean():.4f}")
    print(f"INT8 QUANT LOSS: median cos(int8, fp32) = {np.median(c2):.4f}")
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
