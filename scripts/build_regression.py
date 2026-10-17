#!/usr/bin/env python3
"""Builds regression.npz: embeddings and catalog ids of labelled, non-sealed crops. No images are stored.

    python scripts/build_regression.py --labels labels.json --root <folder with the crops> \
        --model clip_visual_int8.onnx --out regression.npz

labels.json is a list of {"image": "<path relative to --root>", "cardId": "<catalog id>"}. Use your labelled
HiBid development crops. The sealed test set is never used: any path containing a "sealed" folder is refused.
The owner commits the output to the data repo root; the nightly job reads it, contributors cannot change it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embed import ClipEmbedder, preprocess  # noqa: E402
from pipeline.regression import save_regression  # noqa: E402


def is_sealed(path):
    return any("sealed" in part.lower() for part in os.path.normpath(os.path.abspath(path)).split(os.sep))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for path in (args.labels, args.root):
        if is_sealed(path):
            sys.exit("refusing to read anything under a sealed test folder")
    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)

    embedder = ClipEmbedder(args.model)
    vectors, ids = [], []
    for entry in labels:
        path = os.path.join(args.root, entry["image"])
        if is_sealed(path):
            sys.exit(f"refusing {entry['image']}: sealed test data is never used")
        with open(path, "rb") as f:
            data = f.read()
        preprocess(data, trusted=True)  # fails early on an unreadable image
        vectors.append(embedder(data, trusted=True))
        ids.append(entry["cardId"])
    save_regression(args.out, vectors, ids)
    print(f"wrote {len(ids)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
