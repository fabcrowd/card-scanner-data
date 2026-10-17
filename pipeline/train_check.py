"""The "is this a card?" check: logistic regression on raw embeddings (check.bin v1).

    bytes 0..3   little-endian uint32 H, the length of the header
    bytes 4..4+H UTF-8 JSON {"v":1,"dim":512,"bias":b,"threshold":t}
    then         512 float32 LE weights

A crop is a card when sigmoid(w.e + bias) >= threshold. The threshold lets 99% of held-out real cards through, so the
check removes false boxes without costing cards.
"""
import json
import struct

import numpy as np

DIM = 512
MIN_NEGATIVES = 50
MIN_POSITIVES = 20
HOLDOUT = 0.2
MAX_HEADER = 1 << 20


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def parse_check(data):
    """{"weights", "bias", "threshold"}. Raises ValueError for anything that is not check.bin v1."""
    if len(data) < 4:
        raise ValueError("check too short")
    (n,) = struct.unpack("<I", data[:4])
    if n > MAX_HEADER or 4 + n > len(data):
        raise ValueError("bad header length")
    try:
        head = json.loads(data[4 : 4 + n].decode("utf-8"))
        bias, threshold = float(head["bias"]), float(head["threshold"])
    except (ValueError, KeyError, TypeError) as err:
        raise ValueError("bad header") from err
    if head.get("v") != 1 or head.get("dim") != DIM:
        raise ValueError("unsupported check")
    body = data[4 + n :]
    if len(body) != DIM * 4:
        raise ValueError("body size does not match dim")
    return {"weights": np.frombuffer(body, dtype="<f4").copy(), "bias": bias, "threshold": threshold}


def score_check(data, embeddings):
    """Probability each embedding (n, 512) is a card."""
    c = parse_check(data)
    return _sigmoid(np.asarray(embeddings, dtype=np.float32).reshape(-1, DIM) @ c["weights"] + c["bias"])


def _fit(x, y, steps=600, lr=0.1, l2=1e-3):
    w = np.zeros(DIM)
    b = 0.0
    wt = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))  # class balanced
    mw, vw = np.zeros(DIM), np.zeros(DIM)
    mb = vb = 0.0
    for t in range(1, steps + 1):
        r = (_sigmoid(x @ w + b) - y) * wt
        gw, gb = x.T @ r + l2 * w, r.sum()
        mw, vw = 0.9 * mw + 0.1 * gw, 0.999 * vw + 0.001 * gw * gw
        mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb * gb
        w -= lr * (mw / (1 - 0.9**t)) / (np.sqrt(vw / (1 - 0.999**t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9**t)) / (np.sqrt(vb / (1 - 0.999**t)) + 1e-8)
    return w, b


def train(pos, neg, seed=0):
    """check.bin bytes, or None with fewer than 50 negatives (or too few positives to hold any out)."""
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, DIM)
    neg = np.asarray(neg, dtype=np.float32).reshape(-1, DIM)
    if len(neg) < MIN_NEGATIVES or len(pos) < MIN_POSITIVES:
        return None
    rng = np.random.RandomState(seed)
    pp = rng.permutation(len(pos))
    n_val = max(1, int(len(pos) * HOLDOUT))
    held, pos_tr = pos[pp[:n_val]], pos[pp[n_val:]]
    x = np.vstack([pos_tr, neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg))])
    w, b = _fit(x, y)
    scores = np.sort(_sigmoid(held.astype(np.float64) @ w + b))
    threshold = float(scores[int(np.floor(0.01 * len(scores)))])  # at least 99% of held-out cards score >= this
    header = json.dumps({"v": 1, "dim": DIM, "bias": float(b), "threshold": threshold}, separators=(",", ":")).encode()
    return struct.pack("<I", len(header)) + header + w.astype("<f4").tobytes()
