"""gpu_data v2 (fastC): audit fixes + real-photo crops. Reuses gpu_data (jitter, art_view, matrices) unchanged.

Changes against gpu_data.py, each deliberate:
  1. nvJPEG failure RAISES on CUDA (no silent CPU fallback; a fallback would hide a 10x slowdown and a different pipeline).
  3. Blur sigma is in 320-px-card units: uniform 0.12-0.47 (p=0.35). The pilot blurred the whole ~825 px card with
     sigma 0.3-1.2; the GPU pipeline blurs the 144 px art square cut from the 320 px card, so the same strength is
     0.3*320/825 .. 1.2*320/825 = 0.116 .. 0.465 (rounded to 0.12-0.47).
  4. Only the art square is degraded (the extension embeds only the art crop). This deviates from design doc section 3e
     (degrade the whole card, then crop the art) ON PURPOSE: JPEG grid / chroma subsampling align to what is embedded.
  5. JPEG quality: 10 bins {40,45,..,85}, every bin equally likely (uniform on [40,90) floored to 5; the old integer draw
     40..85 gave the 85 bin 1/46 of the mass instead of 1/10).
  6. Height mixture: 50% log-uniform 70-180, 25% log-uniform 48-320, 25% log-uniform 250-320 (capped at 320).
Real crops: full-resolution detector boxes from the Goodwill packs, rectified as scripts/run_real_photos.crop_card
(box -> 750x1050 bilinear), reduced to 320x229 (area) and cached on the GPU like the catalog cards.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_data as G  # noqa: E402
import pilot_data as D  # noqa: E402
from pipeline.embed import CLIP_SIZE  # noqa: E402

SIDE, SX, SY, CARD_H, CARD_W = G.SIDE, G.SX, G.SY, G.CARD_H, G.CARD_W
BLUR_P = 0.35
BLUR_SIGMA = (0.12, 0.47)
QUAL_BINS = tuple(range(40, 90, 5))
H_COMPONENTS = ((0.50, 70, 180), (0.25, 48, 320), (0.25, 250, 320))
OUT_W, OUT_H = 750, 1050
PACKS = ("goodwill-pack.jsonl", "goodwill-pack2.jsonl")
LABELS_DIR = os.path.join(D.ROOT, "data", "pseudo-labels", "v3")
REAL_CACHE = os.path.join(D.ROOT, "train", "out", "cache", "real320.pt")
Image.MAX_IMAGE_PIXELS = None  # local trusted photos


# ---------------------------------------------------------------- parameter sampling
def sample_heights(rng, n):
    p = np.array([c[0] for c in H_COMPONENTS])
    comp = rng.choice(len(H_COMPONENTS), size=n, p=p)
    lo = np.array([c[1] for c in H_COMPONENTS], float)[comp]
    hi = np.array([c[2] for c in H_COMPONENTS], float)[comp]
    return np.minimum(np.rint(np.exp(rng.uniform(np.log(lo), np.log(hi)))).astype(int), CARD_H)


def sample_qualities(rng, n):
    return QUAL_BINS[0] + 5 * np.floor(rng.uniform(0, len(QUAL_BINS), n)).astype(int)


def sample_degradation(rng, n):
    return {"hs": sample_heights(rng, n), "modes": rng.integers(0, len(G.MODES), n),
            "sigmas": np.where(rng.random(n) < BLUR_P, rng.uniform(*BLUR_SIGMA, n), 0.0),
            "quals": sample_qualities(rng, n)}


# ---------------------------------------------------------------- GPU ops
def jpeg_roundtrip(items, quality, device):
    """List of uint8 [3,s,s] tensors -> JPEG encode/decode at one quality. CUDA: nvJPEG, any failure raises."""
    from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

    if device.type == "cuda":
        try:
            # torchvision's nvJPEG calls are not ordered against a non-default current stream: fence around them
            cur, default = torch.cuda.current_stream(device), torch.cuda.default_stream(device)
            cur.synchronize()
            enc = encode_jpeg(items, quality=quality)
            default.synchronize()
            host = [e.cpu() for e in enc]
            out = decode_jpeg(host, device=device, mode=ImageReadMode.RGB)
            default.synchronize()
            return out
        except Exception as err:  # noqa: BLE001 - re-raised with context; no CPU fallback by design
            raise RuntimeError(f"nvJPEG round trip failed (no CPU fallback): {type(err).__name__}: {err}") from err
    return [decode_jpeg(encode_jpeg(t.cpu(), quality=quality), mode=ImageReadMode.RGB).to(device) for t in items]


def degrade(big, p, device):
    """As gpu_data.degrade (art square only; see module docstring) with this module's JPEG round trip and exact bins."""
    n = big.shape[0]
    art = big[:, :, SY:SY + SIDE, SX:SX + SIDE]
    s_np = np.maximum(8, np.rint(0.45 * np.asarray(p["hs"])).astype(int))
    s = torch.from_numpy(s_np).to(device).float()
    r = torch.bmm(G._down_matrix(s, torch.from_numpy(np.asarray(p["modes"])).to(device)),
                  G._blur_matrix(torch.from_numpy(np.asarray(p["sigmas"])).float().to(device)))
    small = torch.matmul(torch.matmul(r.unsqueeze(1), art), r.transpose(1, 2).unsqueeze(1))
    small_u8 = small.round().clamp(0, 255).to(torch.uint8)
    padded = torch.zeros(n, 3, SIDE, SIDE, dtype=torch.uint8, device=device)
    quals = np.asarray(p["quals"]).astype(int)
    for q in np.unique(quals):
        idx = np.flatnonzero(quals == q)
        items = [small_u8[i, :, :s_np[i], :s_np[i]].contiguous() for i in idx]
        for i, out in zip(idx, jpeg_roundtrip(items, int(q), device)):
            padded[i, :, :s_np[i], :s_np[i]] = out
    u = G._up_matrix(s)
    up = torch.matmul(torch.matmul(u.unsqueeze(1), padded.float()), u.transpose(1, 2).unsqueeze(1))
    return G._normalise(up.clamp(0, 255))


def make_batch(cards_u8, rng, device):
    """cards_u8 [B,3,320,229] on device (catalog cards and/or real crops alike) -> (large, deg, hs, big).

    For a real crop 'big' is the (jittered) real crop: teacher input = student large view = art_view(big), and the
    degraded view comes from the same big with the same degrader.
    """
    n = cards_u8.shape[0]
    big = G.jitter(cards_u8, G.jitter_boxes(rng, n))
    p = sample_degradation(rng, n)
    return G.art_view(big), degrade(big, p, device), p["hs"], big


# ---------------------------------------------------------------- labels
def name_groups(name_of):
    """-> (group id per catalog row [C] int64 tensor, normalised name -> group id). Same ids as train_fast.norm_groups."""
    key_to_gid, ids = {}, []
    for i, nm in enumerate(name_of):
        key = G.norm_name(nm)
        ids.append(key_to_gid.setdefault(key, len(key_to_gid)) if key else -1 - i)
    return torch.tensor(ids), key_to_gid


def ce_group(s, gid, catv, grp, tau):
    """Multi-positive CE: positives = catalog rows whose group id == the sample's; every other row is a negative."""
    logits = s @ catv.T / tau
    mask = gid[:, None] == grp[None, :]
    return (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~mask, float("-inf")), 1)).mean()


# ---------------------------------------------------------------- real crops
def read_pack_rows(labels_dir=LABELS_DIR, packs=PACKS):
    """Rows with level id or name (rejects excluded) -> dicts {photo, box, level, train_id, name}."""
    rows = []
    for pack in packs:
        with open(os.path.join(labels_dir, pack), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("level") in ("id", "name"):
                    rows.append({"pack": pack, "photo": r["photo"], "box": r["box"], "level": r["level"],
                                 "train_id": r.get("train_id"), "name": r.get("name")})
    return rows


# fastD: hard eval-photo exclusion, applied to the rows BEFORE crops are cached (so excluded photos are never decoded).
def _sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exclude_eval_photos(rows, eval_dirs, dup_json=None):
    """Drop rows whose photo is listed in dup_json (paths relative to D.ROOT) or whose SHA-256 equals any file under
    eval_dirs (recursive). -> (kept rows, report incl. per-pack counts per reason, SHA-256 set of the eval files)."""
    norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731
    dups = set()
    if dup_json:
        with open(dup_json, encoding="utf-8") as f:
            dups = {norm(os.path.join(D.ROOT, p)) for p in json.load(f)}
    eval_sha = {_sha256(os.path.join(d, f)) for top in eval_dirs for d, _, fs in os.walk(top) for f in fs}
    sha_cache, kept = {}, []
    rep = {"eval_files_hashed": len(eval_sha), "dup_list_size": len(dups), "per_pack": {}}
    for r in rows:
        pk = rep["per_pack"].setdefault(r["pack"], {"candidates": 0, "excluded_dup_list": 0, "excluded_sha_other": 0, "kept": 0})
        pk["candidates"] += 1
        if norm(r["photo"]) in dups:
            pk["excluded_dup_list"] += 1
            continue
        if r["photo"] not in sha_cache:
            sha_cache[r["photo"]] = _sha256(r["photo"])
        if sha_cache[r["photo"]] in eval_sha:
            pk["excluded_sha_other"] += 1
            continue
        pk["kept"] += 1
        kept.append(r)
    rep["dup_list_photos_hit"] = len({norm(r["photo"]) for r in rows} & dups)
    return kept, rep, eval_sha


def rectify_320(img, box):
    """scripts/run_real_photos.crop_card (box -> 750x1050 bilinear), then 320x229 area -> uint8 [3,320,229]."""
    x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
    card = img.crop((x1, y1, round(box[2]), round(box[3]))).resize((OUT_W, OUT_H), Image.BILINEAR)
    arr = np.asarray(card.resize((CARD_W, CARD_H), Image.BOX), dtype=np.uint8)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def _crops_of_photo(job):
    photo, boxes = job
    with Image.open(photo) as im:
        img = im.convert("RGB")
    return [rectify_320(img, b) for b in boxes]


def load_real_crops(rows, cache_path=REAL_CACHE):
    """-> uint8 [N,3,320,229] aligned with rows. Builds the cache (all non-reject rows) on first use."""
    keys = [(r["photo"], [round(v, 1) for v in r["box"]]) for r in rows]
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        assert blob["keys"] == keys, "real crop cache does not match the pack rows; delete it"
        return blob["crops"]
    by_photo = {}
    for i, (photo, box) in enumerate(keys):
        by_photo.setdefault(photo, []).append((i, box))
    jobs = [(p, [b for _, b in v]) for p, v in by_photo.items()]
    out = [None] * len(rows)
    with ThreadPoolExecutor(8) as ex:
        for (p, v), crops in zip(by_photo.items(), ex.map(_crops_of_photo, jobs)):
            for (i, _), c in zip(v, crops):
                out[i] = c
    crops = torch.stack(out)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"keys": keys, "crops": crops}, cache_path)
    return crops


def label_names(rows, name_of, row_of):
    """Label name per pack row: id level -> catalog name of train_id (fallback the row's name); name level -> its name."""
    out = []
    for r in rows:
        if r["level"] == "id" and r["train_id"] in row_of:
            out.append(name_of[row_of[r["train_id"]]] or r["name"])
        else:
            out.append(r["name"])
    return out


def filter_real(names, row_names, ev_norm, key_to_gid):
    """Leakage rule (same as the catalog split): drop a crop whose normalised label name equals the normalised name of
    any synthetic eval card (checks the label name and, for id rows, the row's own name too). Also drops crops whose
    name matches no catalog row (no positive). -> (keep bool [N], gid [N] for the kept, report)."""
    keys = [G.norm_name(n) for n in names]
    row_keys = [G.norm_name(n) for n in row_names]
    leak = np.array([bool(k) and (k in ev_norm or (rk and rk in ev_norm)) for k, rk in zip(keys, row_keys)])
    nocat = np.array([(not k) or (k not in key_to_gid) for k in keys])
    keep = ~leak & ~nocat
    gid = np.array([key_to_gid.get(k, -1) for k in keys], dtype=np.int64)
    rep = {"candidates": len(keys), "dropped_leakage": int(leak.sum()),
           "dropped_no_catalog_name": int((nocat & ~leak).sum()), "used": int(keep.sum())}
    return keep, gid[keep], rep


class Cycle:
    """Endless shuffled stream over range(n): take(k) -> k indices (reshuffles when a pass is used up)."""

    def __init__(self, n, rng):
        self.n, self.rng, self.buf = n, rng, np.empty(0, dtype=np.int64)

    def take(self, k):
        while len(self.buf) < k:
            self.buf = np.concatenate([self.buf, self.rng.permutation(self.n)])
        out, self.buf = self.buf[:k], self.buf[k:]
        return out
