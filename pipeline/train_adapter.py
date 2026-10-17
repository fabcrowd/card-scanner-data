"""Photo adapter: e' = normalize(e + U(V^T e)), applied to queries and to catalog rows (adapter.bin v1).

    bytes 0..3   little-endian uint32 H, the length of the header
    bytes 4..4+H UTF-8 JSON {"v":1,"dim":512,"rank":r,"scale":s}
    then         U (512 x r int8, row-major) followed by V (512 x r int8); float = int8 * s

Trained with numpy only: softmax over the whole adapted catalog (so every other card is a negative, the hard ones
carry most of the gradient) with Adam. It ships only if it beats the identity map on a held-out 10% of the pairs.
"""
import json
import struct
import time

import numpy as np

DIM = 512
RANK = 32
TAU = 0.05
HOLDOUT = 0.1
MIN_PAIRS = 20
MAX_HEADER = 1 << 20


def pack_adapter(u, v):
    """adapter.bin bytes for float U, V of shape (512, r); one shared int8 scale."""
    u, v = np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32)
    rank = u.shape[1]
    scale = float(max(np.abs(u).max(), np.abs(v).max(), 1e-12) / 127)
    header = json.dumps({"v": 1, "dim": DIM, "rank": rank, "scale": scale}, separators=(",", ":")).encode()
    q = lambda m: np.clip(np.rint(m / scale), -127, 127).astype(np.int8).tobytes()
    return struct.pack("<I", len(header)) + header + q(u) + q(v)


def parse_adapter(data):
    """(U, V) float32 (512, r). Raises ValueError for anything that is not adapter.bin v1."""
    if len(data) < 4:
        raise ValueError("adapter too short")
    (n,) = struct.unpack("<I", data[:4])
    if n > MAX_HEADER or 4 + n > len(data):
        raise ValueError("bad header length")
    try:
        head = json.loads(data[4 : 4 + n].decode("utf-8"))
        rank, scale = int(head["rank"]), float(head["scale"])
    except (ValueError, KeyError, TypeError) as err:
        raise ValueError("bad header") from err
    if head.get("v") != 1 or head.get("dim") != DIM or rank < 1 or not scale > 0:
        raise ValueError("unsupported adapter")
    body = data[4 + n :]
    if len(body) != 2 * DIM * rank:
        raise ValueError("body size does not match rank")
    m = np.frombuffer(body, dtype=np.int8).astype(np.float32).reshape(2, DIM, rank) * scale
    return m[0], m[1]


def apply_adapter(adapter, x):
    """Adapted, re-normalised copy of x (n, 512) or (512,)."""
    u, v = adapter
    x = np.asarray(x, dtype=np.float32)
    y = x + (x @ v) @ u.T
    return y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)


def _forward(x, u, v):
    h = x @ v
    y = x + h @ u.T
    n = np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y / n, h, n


def _backward(dz, z, x, h, n, u, v):
    dy = (dz - z * np.sum(z * dz, axis=1, keepdims=True)) / n
    dh = dy @ u
    return dy.T @ h, x.T @ dh  # dU, dV


def _top1(cat_vecs, queries, targets):
    hits = 0
    for s in range(0, len(queries), 512):
        hits += int(np.sum(np.argmax(queries[s : s + 512] @ cat_vecs.T, axis=1) == targets[s : s + 512]))
    return hits / max(len(queries), 1)


def _heldout_top1(adapter, cat, queries, targets):
    return _top1(apply_adapter(adapter, cat), apply_adapter(adapter, queries), targets)


def train(pairs, catalog, rank=RANK, batch=128, max_epochs=8, patience=6, lr=1e-3, max_seconds=480, seed=0):
    """adapter.bin bytes, or None when there is too little data or it does not beat identity on the held-out split."""
    row = {cid: i for i, cid in enumerate(catalog.ids)}
    kept = [(e, row[c]) for e, c in pairs if c in row]
    if len(kept) < MIN_PAIRS:
        return None
    x = np.asarray(np.stack([e for e, _ in kept]), dtype=np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    t = np.array([r for _, r in kept])
    cat = catalog.vectors
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(x))
    n_val = max(1, int(len(x) * HOLDOUT))
    val, tr = order[:n_val], order[n_val:]
    base = _top1(cat, x[val], t[val])

    u = np.zeros((DIM, rank), dtype=np.float32)
    v = (rng.randn(DIM, rank) / np.sqrt(DIM)).astype(np.float32)
    m = [np.zeros_like(u), np.zeros_like(v)]
    s2 = [np.zeros_like(u), np.zeros_like(v)]
    step = 0
    best, best_uv, stale = base, None, 0
    started = time.monotonic()
    for _ in range(max_epochs):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            zq, hq, nq = _forward(x[b], u, v)
            zc, hc, nc = _forward(cat, u, v)
            logits = zq @ zc.T / TAU
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(b)), t[b]] -= 1
            g = p / (len(b) * TAU)
            gu1, gv1 = _backward(g @ zc, zq, x[b], hq, nq, u, v)
            gu2, gv2 = _backward(g.T @ zq, zc, cat, hc, nc, u, v)
            step += 1
            for k, (param, grad) in enumerate(((u, gu1 + gu2), (v, gv1 + gv2))):
                m[k] = 0.9 * m[k] + 0.1 * grad
                s2[k] = 0.999 * s2[k] + 0.001 * grad * grad
                param -= lr * (m[k] / (1 - 0.9**step)) / (np.sqrt(s2[k] / (1 - 0.999**step)) + 1e-8)
        # U V^T is invariant to rebalancing; equal scales keep the shared int8 grid fine for both
        c = np.sqrt((np.abs(v).max() + 1e-12) / (np.abs(u).max() + 1e-12))
        cand = pack_adapter(u * c, v / c)
        score = _heldout_top1(parse_adapter(cand), cat, x[val], t[val])
        if score > best:
            best, best_uv, stale = score, cand, 0
        else:
            stale += 1
        if stale >= patience or time.monotonic() - started > max_seconds:
            break
    return best_uv
