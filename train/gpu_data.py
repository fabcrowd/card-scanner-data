"""GPU-resident training data + augmentation for the full-tower distillation (replaces pilot_data's PIL DataLoader path).

All training cards live on the GPU as one uint8 tensor [N,3,320,229] (area-resized from the catalog image). Per step:
  1. coupled geometric jitter: +-6% per-side box (shrunk about its centre to the card aspect) via roi_align -> 'big'
  2. teacher input / student large view = art crop of big (embed.art_rect: top 144 rows, centred 144 px square) -> 224 bicubic
  3. student degraded view from the SAME big: height H from the 60/40 mixture, art square -> optional Gaussian blur
     -> s x s (s = max(8, round(0.45*H))) with a random resize mode -> JPEG round trip (q 40..85, binned by 5, batched
     nvJPEG on CUDA) -> 224 bicubic.

Differences from the PIL pipeline (pilot_data.py / eval_synth_small.degrade), deliberately not tuned away:
  - no true Lanczos on GPU: the resize mode is one of {bilinear+antialias, bicubic+antialias, bilinear without antialias}
    instead of {box, bilinear, bicubic, lanczos}; blur+resize are per-sample separable matrices (torch's antialias kernels;
    the final upscale is bicubic a=-0.75, PIL: -0.5); blur is not rounded to uint8 before the resize
  - the degradation is applied to the art square only (PIL degrades the whole card, then crops the art): JPEG block grid
    and chroma subsampling are aligned to the art square, blur edges use reflect padding
  - cards are first reduced to 320 px height; H>=250 views are downscaled from 320, not from the full-size original
  - JPEG quality is quantised to multiples of 5 (40,45,..,85) so each nvJPEG call is a single batch
  - roi_align (sampling_ratio=2) replaces PIL crop+BOX resize for the jitter
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_data as D  # noqa: E402
from pipeline.embed import CLIP_SIZE, MEAN, STD, art_rect  # noqa: E402

CARD_H, CARD_W = D.LARGE_H, 229
SX, SY, SIDE = art_rect(CARD_W, CARD_H)
JITTER, BLUR_P, BLUR_SIGMA, JPEG_Q = D.JITTER, D.BLUR_P, D.BLUR_SIGMA, D.JPEG_Q
H_FOCUS, H_WIDE, FOCUS_FRAC = D.H_FOCUS, D.H_WIDE, D.FOCUS_FRAC
MODES = ("bilinear_aa", "bicubic_aa", "bilinear_noaa")
CACHE = os.path.join(D.ROOT, "train", "out", "cache", "cards320.pt")
_USE_NVJPEG = {"on": True}  # flipped off if the CUDA encoder/decoder is unavailable


def norm_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower()) if name else ""


# ---------------------------------------------------------------- card cache
def _load_card(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").resize((CARD_W, CARD_H), Image.BOX), dtype=np.uint8)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def load_training_cards(cat, name_of, eval_rows, split_path, cache_path=CACHE):
    """-> (cards uint8 [N,3,320,229] CPU, catalog rows int64 [N], report). Builds the cache on first use."""
    with open(split_path, encoding="utf-8") as f:
        train_ids = json.load(f)["train_ids"]
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        assert blob["ids"] == train_ids, "cards cache does not match split.json; delete it"
        cards = blob["cards"]
    else:
        with ThreadPoolExecutor(12) as ex:
            cards = torch.stack(list(ex.map(_load_card, [D.E.image_path(c) for c in train_ids])))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"ids": train_ids, "cards": cards}, cache_path)
    row_of = {c: i for i, c in enumerate(cat.ids)}
    rows = np.array([row_of[c] for c in train_ids], dtype=np.int64)
    ev_norm = {norm_name(name_of[r]) for r in eval_rows if name_of[r]}
    keep = np.array([not (name_of[r] and norm_name(name_of[r]) in ev_norm) for r in rows])
    report = {"split_train": len(rows), "removed_normalised_name_match": int((~keep).sum()), "kept": int(keep.sum())}
    k = torch.from_numpy(np.flatnonzero(keep))
    return cards.index_select(0, k), rows[keep], report


# ---------------------------------------------------------------- parameter sampling (CPU, numpy)
def sample_heights(rng, n):
    focus = rng.random(n) < FOCUS_FRAC
    lo = np.where(focus, H_FOCUS[0], H_WIDE[0])
    hi = np.where(focus, H_FOCUS[1], H_WIDE[1])
    return np.minimum(np.rint(np.exp(rng.uniform(np.log(lo), np.log(hi)))).astype(int), CARD_H)


def sample_degradation(rng, n):
    return {"hs": sample_heights(rng, n), "modes": rng.integers(0, len(MODES), n),
            "sigmas": np.where(rng.random(n) < BLUR_P, rng.uniform(*BLUR_SIGMA, n), 0.0),
            "quals": rng.integers(JPEG_Q[0], JPEG_Q[1] + 1, n)}


def jitter_boxes(rng, n):
    """[n,4] x0,y0,x1,y1 in card pixels: +-6% per side (clamped), shrunk about the centre to the card aspect."""
    w, h = float(CARD_W), float(CARD_H)
    u = rng.uniform(-JITTER, JITTER, (n, 4))
    x0, x1 = np.maximum(0.0, u[:, 0] * w), np.minimum(w, w - u[:, 1] * w)
    y0, y1 = np.maximum(0.0, u[:, 2] * h), np.minimum(h, h - u[:, 3] * h)
    bw, bh, ar = x1 - x0, y1 - y0, w / h
    wide = bw / bh > ar
    nw, nh = bh * ar, bw / ar
    x0n = np.where(wide, x0 + (bw - nw) / 2, x0)
    x1n = np.where(wide, x0n + nw, x1)
    y0n = np.where(wide, y0, y0 + (bh - nh) / 2)
    y1n = np.where(wide, y1, y0n + nh)
    return np.stack([x0n, y0n, x1n, y1n], 1)


# ---------------------------------------------------------------- GPU ops
def _normalise(x255):
    mean = torch.tensor(MEAN, device=x255.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x255.device).view(1, 3, 1, 1)
    return (x255 / 255 - mean) / std


def jitter(cards_u8, boxes):
    """Batched crop + resize back to 320x229 -> float32 [B,3,320,229] in 0..255 ('big')."""
    b = cards_u8.shape[0]
    rois = torch.cat([torch.arange(b, dtype=torch.float32).view(-1, 1), torch.from_numpy(boxes).float()], 1)
    return roi_align(cards_u8.float(), rois.to(cards_u8.device), (CARD_H, CARD_W), 1.0, 2, True)


def art_view(big):
    """Teacher input / student large view: art square of big -> 224 bicubic -> CLIP-normalised."""
    art = big[:, :, SY:SY + SIDE, SX:SX + SIDE]
    return _normalise(F.interpolate(art, size=(CLIP_SIZE, CLIP_SIZE), mode="bicubic", align_corners=False).clamp(0, 255))


def _jpeg_roundtrip(items, quality, device):
    """List of uint8 [3,s,s] tensors -> JPEG encode/decode at one quality (nvJPEG on CUDA, CPU threads otherwise)."""
    from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

    if device.type == "cuda" and _USE_NVJPEG["on"]:
        try:
            # torchvision's nvJPEG calls are not ordered against a non-default current stream: fence both streams around them
            cur, default = torch.cuda.current_stream(device), torch.cuda.default_stream(device)
            cur.synchronize()
            enc = encode_jpeg(items, quality=quality)
            default.synchronize()
            host = [e.cpu() for e in enc]
            out = decode_jpeg(host, device=device, mode=ImageReadMode.RGB)
            default.synchronize()
            return out
        except Exception as err:  # noqa: BLE001 - torchvision raises RuntimeError/ValueError depending on build
            print(f"nvJPEG unavailable ({err}); falling back to CPU JPEG in a thread pool", flush=True)
            _USE_NVJPEG["on"] = False

    def one(t):
        out = decode_jpeg(encode_jpeg(t.cpu(), quality=quality), mode=ImageReadMode.RGB)
        return out.to(device)

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, items))


def _cubic(ax, a):
    return torch.where(ax < 1, (a + 2) * ax ** 3 - (a + 3) * ax ** 2 + 1,
                       torch.where(ax < 2, a * ax ** 3 - 5 * a * ax ** 2 + 8 * a * ax - 4 * a, torch.zeros_like(ax)))


def _down_matrix(s, modes, n=SIDE):
    """[B,n,n] per-sample row-resize matrices n -> s (rows >= s are zero); same kernels as F.interpolate (antialias)."""
    dev = s.device
    i = torch.arange(n, device=dev, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(n, device=dev, dtype=torch.float32).view(1, 1, -1)
    scale = (n / s).view(-1, 1, 1)
    center = (i + 0.5) * scale
    ax = ((j + 0.5 - center) / scale).abs()
    aa = torch.where(modes.view(-1, 1, 1) == 0, (1 - ax).clamp(min=0), _cubic(ax, -0.5))
    aa = aa / aa.sum(-1, keepdim=True).clamp(min=1e-8)
    src = center - 0.5  # bilinear without antialias: two taps, border-clamped
    i0 = torch.floor(src)
    f = src - i0
    noaa = (1 - f) * (j == i0.clamp(0, n - 1)).float() + f * (j == (i0 + 1).clamp(0, n - 1)).float()
    w = torch.where(modes.view(-1, 1, 1) == 2, noaa, aa)
    return w * (i < s.view(-1, 1, 1)).float()


def _blur_matrix(sigma, n=SIDE, radius=4):
    """[B,n,n] Gaussian blur matrices with reflect padding (sigma 0 -> identity); radius covers 3*max sigma."""
    dev = sigma.device
    k = torch.arange(-radius, radius + 1, device=dev, dtype=torch.float32).view(1, -1)
    sig = torch.where(sigma > 0, sigma, torch.ones_like(sigma)).view(-1, 1)
    g = torch.exp(-k ** 2 / (2 * sig ** 2)) * (k.abs() <= torch.ceil(3 * sigma).view(-1, 1)).float()
    g = g / g.sum(1, keepdim=True)
    i = torch.arange(n, device=dev)
    m = torch.zeros(len(sigma), n, n, device=dev)
    for t in range(2 * radius + 1):
        idx = (i + t - radius).abs()
        idx = torch.where(idx > n - 1, 2 * (n - 1) - idx, idx)
        m.scatter_add_(2, idx.view(1, n, 1).expand(len(sigma), n, 1), g[:, t].view(-1, 1, 1).expand(len(sigma), n, 1))
    return m


def _up_matrix(s, n_in=SIDE, n_out=CLIP_SIZE):
    """[B,n_out,n_in] bicubic (a=-0.75, border-clamped; as F.interpolate bicubic) upscale matrices s -> n_out."""
    dev = s.device
    o = torch.arange(n_out, device=dev, dtype=torch.float32).view(1, -1)
    src = (o + 0.5) * (s.view(-1, 1) / n_out) - 0.5
    i0 = torch.floor(src)
    m = torch.zeros(len(s), n_out, n_in, device=dev)
    for t in (-1, 0, 1, 2):
        idx = torch.minimum((i0 + t).clamp(min=0), s.view(-1, 1) - 1).long()
        m.scatter_add_(2, idx.unsqueeze(-1), _cubic((src - i0 - t).abs(), -0.75).unsqueeze(-1))
    return m


def degrade(big, p, device):
    """Degraded student view from big per the sampled parameters p -> CLIP-normalised float [B,3,224,224].

    Fully batched: blur + antialiased resize are one per-sample matrix product (rows/cols padded to 144), the JPEG
    round trip is nvJPEG per quality bin, the bicubic upscale is another per-sample matrix product.
    """
    n = big.shape[0]
    art = big[:, :, SY:SY + SIDE, SX:SX + SIDE]
    s_np = np.maximum(8, np.rint(0.45 * np.asarray(p["hs"])).astype(int))
    s = torch.from_numpy(s_np).to(device).float()
    r = torch.bmm(_down_matrix(s, torch.from_numpy(np.asarray(p["modes"])).to(device)),
                  _blur_matrix(torch.from_numpy(np.asarray(p["sigmas"])).float().to(device)))
    small = torch.matmul(torch.matmul(r.unsqueeze(1), art), r.transpose(1, 2).unsqueeze(1))
    small_u8 = small.round().clamp(0, 255).to(torch.uint8)
    padded = torch.zeros(n, 3, SIDE, SIDE, dtype=torch.uint8, device=device)
    qbin = (np.asarray(p["quals"]) // 5) * 5
    for q in np.unique(qbin):
        idx = np.flatnonzero(qbin == q)
        items = [small_u8[i, :, :s_np[i], :s_np[i]].contiguous() for i in idx]
        for i, out in zip(idx, _jpeg_roundtrip(items, int(q), device)):
            padded[i, :, :s_np[i], :s_np[i]] = out
    u = _up_matrix(s)
    up = torch.matmul(torch.matmul(u.unsqueeze(1), padded.float()), u.transpose(1, 2).unsqueeze(1))
    return _normalise(up.clamp(0, 255))


def make_batch(cards_u8, rng, device):
    """cards_u8 [B,3,320,229] on device -> (large, deg, hs, big). large and deg both derive from the same jittered big."""
    n = cards_u8.shape[0]
    big = jitter(cards_u8, jitter_boxes(rng, n))
    p = sample_degradation(rng, n)
    return art_view(big), degrade(big, p, device), p["hs"], big


def validate(device="cuda"):
    """Printed report: cache/split, height mixture, and GPU-degraded eval crops through the SHIPPED int8 ONNX (batch 1)."""
    import time

    import onnxruntime as ort
    import score_synth_fusion as S
    from concurrent.futures import ThreadPoolExecutor as TP
    from pipeline.catalog import Catalog

    import pilot_eval as V

    dev = torch.device(device)
    cat = Catalog.load(os.path.join(D.ROOT, "assets", "catalog.bin.gz"), os.path.join(D.ROOT, "assets", "id_map.json"))
    nm = S.load_names()
    name_of = [nm.get(c) for c in cat.ids]
    _, ev = D.eval_rows(cat)
    cards, rows, rep = load_training_cards(cat, name_of, ev, os.path.join(V.OUT, "split.json"))
    print("train cards:", rep, "| tensor", tuple(cards.shape), f"{cards.numel() / 2**30:.2f} GiB", flush=True)
    h = sample_heights(np.random.default_rng(0), 200_000)
    print(f"height draws: 70-180 {np.mean((h >= 70) & (h <= 180)):.3f} (focus component is 0.60 of draws), >=250 {np.mean(h >= 250):.3f}")
    with TP(12) as ex:
        evc = torch.stack(list(ex.map(_load_card, [D.E.image_path(cat.ids[r]) for r in ev]))).to(dev)
    sess = ort.InferenceSession(V.SHIPPED, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    base = json.load(open(os.path.join(V.OUT, "baseline", "metrics.json"), encoding="utf-8"))["heights"]
    rng = np.random.default_rng(9001)
    truth = np.array(ev)
    print("  H | GPU top1 top5 | CPU eval1 top1 top5 | d_top1")
    for hh in D.E.HEIGHTS:
        t0 = time.time()
        p = {"hs": np.full(len(ev), hh), "modes": rng.integers(0, len(MODES), len(ev)), "sigmas": np.zeros(len(ev)),
             "quals": rng.integers(JPEG_Q[0], JPEG_Q[1] + 1, len(ev))}
        x = torch.cat([degrade(evc[s:s + 100].float(), {k: v[s:s + 100] for k, v in p.items()}, dev) for s in range(0, len(ev), 100)]).cpu().numpy()
        fresh = []
        for i in range(len(x)):
            v = sess.run(None, {name: x[i:i + 1]})[0].astype(np.float32)
            fresh.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        idx, _ = cat.top_k_batch(np.concatenate(fresh), k=5)
        t1, t5 = float((idx[:, 0] == truth).mean()), float((idx == truth[:, None]).any(1).mean())
        b = base[str(hh)]["all"]
        print(f"{hh:4d}| {t1:.3f} {t5:.3f} | {b['top1']:.3f} {b['top5']:.3f} | {t1 - b['top1']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    print("nvJPEG used:", _USE_NVJPEG["on"])


if __name__ == "__main__":
    validate()
