"""Re-runs export_clip_onnx.py's export + quantisation and compares to the shipped int8 asset."""
import os
import random
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import open_clip
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32_parity import IMG_DIR, N, SEED, image_path, load_preprocess  # noqa: E402,F401
from pipeline.catalog import Catalog  # noqa: E402

OUT = os.path.join(ROOT, "data", "export-test")
SHIPPED = os.path.join(ROOT, "assets", "clip_visual_int8.onnx")


class VisualEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.encode_image(x)
        return features / features.norm(dim=-1, keepdim=True)


def run_onnx(path, tensors):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for t in tensors:
        v = sess.run(None, {name: t[None].astype(np.float32)})[0].reshape(-1)
        out.append(v / np.linalg.norm(v))
    return np.stack(out).astype(np.float32)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    fp32_path = os.path.join(OUT, "clip_visual_fp32.onnx")
    int8_path = os.path.join(OUT, "clip_visual_int8.onnx")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    encoder = VisualEncoder(model)
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        encoder, dummy, fp32_path,
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=14, dynamo=False,
    )
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)

    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    have = [i for i, cid in enumerate(cat.ids) if image_path(cid)]
    rows = random.Random(SEED).sample(have, N)
    tensors = torch.stack([load_preprocess(preprocess, image_path(cat.ids[r])) for r in rows])
    with torch.no_grad():
        f = model.encode_image(tensors)
        fp32 = (f / f.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
    tn = tensors.numpy()
    new8 = run_onnx(int8_path, tn)
    old8 = run_onnx(SHIPPED, tn)
    c = np.einsum("ij,ij->i", new8, old8)
    c_fp = np.einsum("ij,ij->i", new8, fp32)
    print(f"cos(re-exported int8, shipped int8): min={c.min():.4f} median={np.median(c):.4f} mean={c.mean():.4f} frac>=0.999={(c >= 0.999).mean():.4f}")
    print(f"cos(re-exported int8, fp32 torch): median={np.median(c_fp):.4f}")
    print(f"sizes bytes: exported fp32={os.path.getsize(fp32_path)} exported int8={os.path.getsize(int8_path)} shipped int8={os.path.getsize(SHIPPED)} (expected 88999043)")
    print(f"versions: torch={torch.__version__} onnx={onnx.__version__} onnxruntime={ort.__version__} open_clip={open_clip.__version__}")
    print("EXPORT REPRODUCES SHIPPED ASSET" if np.median(c) >= 0.999 else f"EXPORT DIFFERS: median cos={np.median(c):.4f} min={c.min():.4f} frac>=0.999={(c >= 0.999).mean():.4f}")
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
