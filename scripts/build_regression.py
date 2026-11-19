#!/usr/bin/env python3
"""Builds regression.npz: embeddings and labels of real, non-sealed crops. No images are stored. The file also
carries the model (first 16 hex of the sha256 of --model) and the crop version (pins.json), and the nightly job ignores
it when they are not its own.

Image mode (rows labelled with an exact catalog id; embeds the crops with --model):

    python scripts/build_regression.py --labels labels.json --root <folder with the crops> \
        --model clip_visual_int8.onnx --out regression.npz

labels.json is a list of {"image": "<path relative to --root>", "cardId": "<catalog id>"}. Use your labelled
HiBid development crops.

Name mode (rows labelled with a card name; embeddings that already exist, so --model only supplies the tag):

    python scripts/build_regression.py --embeddings emb.npy --labels labels.json --catalog-names names.json \
        --id-map id_map.json --model clip_visual_int8.onnx --out regression.npz

emb.npy is float32 (n, 512) with unit rows. labels.json is a list of {"row": <index into emb.npy>, "name": "<label>"}.
names.json maps each shipped catalog id to its printed card name. id_map.json is the shipped catalog's id list. Each row
stores the label's base name and the shipped ids whose base name equals it (only ids in id_map.json), resolved here so
the nightly job needs no name map. A row with no such id is allowed: only a new card can make it right.

The sealed test set is never used: any path containing a "sealed" folder is refused, in both modes.
The owner commits the output to the data repo root; the nightly job reads it, contributors cannot change it.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embed import ClipEmbedder, preprocess  # noqa: E402
from pipeline.names import NameTruth, accept_lists, base_name  # noqa: E402
from pipeline.regression import DIM, UNIT_TOLERANCE, save_regression, tag_for  # noqa: E402


def is_sealed(path):
    return any("sealed" in part.lower() for part in os.path.normpath(os.path.abspath(path)).split(os.sep))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_image_rows(args):
    labels = read_json(args.labels)
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
    return vectors, ids


def build_name_rows(args):
    embeddings = np.load(args.embeddings, allow_pickle=False)
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or embeddings.shape[1] != DIM:
        sys.exit(f"{args.embeddings} must be float32 with shape (n, {DIM})")
    if not np.all(np.abs(np.linalg.norm(embeddings, axis=1) - 1) <= UNIT_TOLERANCE):
        sys.exit(f"{args.embeddings} must hold unit-length rows")
    labels = read_json(args.labels)
    catalog_names = read_json(args.catalog_names)
    shipped_ids = read_json(args.id_map)
    if not isinstance(catalog_names, dict) or not isinstance(shipped_ids, list):
        sys.exit("--catalog-names must be an object of id to name and --id-map a list of ids")
    accepts = accept_lists(catalog_names, shipped_ids)
    rows, truths = [], []
    for entry in labels:
        row, name = entry["row"], entry["name"]
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(embeddings):
            sys.exit(f"label row {row!r} is not an index into {args.embeddings} ({len(embeddings)} rows)")
        base = base_name(name)
        try:
            truths.append(NameTruth(base, frozenset(accepts.get(base, ()))))
        except ValueError:
            sys.exit(f"label {name!r} (row {row}) has no usable base name")
        rows.append(row)
    return embeddings[rows], truths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", help="image mode: the folder with the crops")
    ap.add_argument("--embeddings", help="name mode: float32 (n, 512) unit rows, .npy")
    ap.add_argument("--catalog-names", help="name mode: JSON object of shipped catalog id to printed card name")
    ap.add_argument("--id-map", help="name mode: the shipped catalog's id_map.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pins", default=None, help="pins.json whose cropVersion is stored in the file (default: the template's)")
    args = ap.parse_args()

    name_mode = args.embeddings is not None
    if name_mode and (args.root is not None or not args.catalog_names or not args.id_map):
        ap.error("name mode takes --embeddings, --labels, --catalog-names, --id-map, --model and --out, and no --root")
    if not name_mode and (args.root is None or args.catalog_names or args.id_map):
        ap.error("image mode takes --labels, --root, --model and --out")
    for path in (args.labels, args.root, args.embeddings, args.catalog_names, args.id_map, args.model, args.out, args.pins):
        if path and is_sealed(path):
            sys.exit("refusing to read anything under a sealed test folder")

    vectors, truths = build_name_rows(args) if name_mode else build_image_rows(args)
    model, crop = tag_for(args.model, args.pins)
    save_regression(args.out, vectors, truths, model=model, crop=crop)
    print(f"wrote {len(truths)} rows to {args.out}", file=sys.stderr)
    if name_mode:
        print(f"rows with an empty accept list: {sum(not t.accept for t in truths)}", file=sys.stderr)
        print(f"distinct base names: {len({t.name for t in truths})}", file=sys.stderr)


if __name__ == "__main__":
    main()
