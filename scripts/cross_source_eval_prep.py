"""Held-out eval photo inventory: lists data/hibid-pack/eval and data/goodwill-pack2/eval photos, drops any byte-identical
(SHA-256) to a training-split file or to data/fb-eval-pack/user-saved. Writes data/pseudo-labels/eval/photos.json."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
EXT = (".jpg", ".jpeg", ".png", ".webp")


def files(d):
    return sorted(os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs if f.lower().endswith(EXT))


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


train = files(os.path.join(D, "goodwill-pack")) + files(os.path.join(D, "goodwill-pack2", "train")) \
    + files(os.path.join(D, "hibid-pack", "train")) + files(os.path.join(D, "fb-eval-pack", "user-saved"))
tr = {sha(p) for p in train}
out = {"train_like_files": len(train), "sources": {}}
for src, d in (("hibid", os.path.join(D, "hibid-pack", "eval")), ("goodwill-pack2", os.path.join(D, "goodwill-pack2", "eval"))):
    fs = files(d)
    keep, dup = [], []
    for p in fs:
        (dup if sha(p) in tr else keep).append(os.path.relpath(p, ROOT).replace("\\", "/"))
    out["sources"][src] = {"found": len(fs), "dropped_identical": len(dup), "kept": keep, "dropped": dup}
    print(src, "found", len(fs), "dropped", len(dup), "kept", len(keep))
json.dump(out, open(os.path.join(D, "pseudo-labels", "eval", "photos.json"), "w"), indent=1)
