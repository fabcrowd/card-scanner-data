#!/usr/bin/env python3
"""Builds seed.npz: embeddings and catalog ids of labelled real crops plus synthetic photo-like catalog variants.
No images are stored in the output. Run it on the machine that has the catalog art and the labelled crops.

    python scripts/build_seed.py --data-dir data --meta data/card-catalog/meta.json \
        --catalog-images data/card-catalog/images --model clip_visual_int8.onnx --out seed.npz

--data-dir holds variant-photos/ and edition-crops/ (see pipeline/seed.py). HiBid, sealed-test and realworld data
are the fixed exam and are never trained on: any input path containing "hibid", "sealed-test", "sealed" or
"realworld" is refused. The owner commits seed.npz to the data repo's main branch; the nightly loads it from there.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pipeline.seed import FORBIDDEN, Sources, build_seed, real_crops_from_repo  # noqa: E402

FORBIDDEN_PARTS = FORBIDDEN + ("sealed",)


def is_forbidden(path):
    return any(word in part.lower() for part in os.path.normpath(os.path.abspath(path)).split(os.sep) for word in FORBIDDEN_PARTS)


def save_seed(path, pairs):
    """seed.npz: float32 embeddings (n, 512) and card ids (n,) as plain unicode, loadable without pickle."""
    vectors = np.stack([np.asarray(e, dtype=np.float32) for e, _ in pairs]) if pairs else np.zeros((0, 512), dtype=np.float32)
    np.savez_compressed(path, embeddings=vectors, cardIds=np.array([c for _, c in pairs], dtype="<U64"))


def build(data_dir, meta_path, images_dir, embedder, per_card=1, max_synthetic=None):
    """[(embedding, card id)]: labelled real crops first, then synthetic variants of catalog images."""
    for path in (data_dir, meta_path, images_dir):
        if path and is_forbidden(path):
            raise ValueError(f"refusing {path}: HiBid, sealed and real-world data are never used for the seed")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    images = []
    for card in meta:
        for ext in (".webp", ".png", ".jpg"):
            candidate = os.path.join(images_dir, card["id"] + ext)
            if os.path.exists(candidate):
                images.append((candidate, card["id"]))
                break
    real = real_crops_from_repo(data_dir, meta)
    pairs = build_seed(Sources(embedder=embedder, real=real, catalog_images=images, synthetic_per_card=per_card, max_synthetic=max_synthetic))
    return pairs, len(real), len(images)


def main(argv=None, embedder=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--catalog-images", required=True)
    ap.add_argument("--model", help="clip_visual_int8.onnx (unused when an embedder is passed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-card", type=int, default=1, help="synthetic variants per catalog card")
    ap.add_argument("--max-synthetic", type=int, default=None, help="sample this many catalog cards for variants")
    args = ap.parse_args(argv)
    if embedder is None:
        from pipeline.embed import ClipEmbedder

        embedder = ClipEmbedder(args.model)
    try:
        pairs, n_real, n_images = build(args.data_dir, args.meta, args.catalog_images, embedder, args.per_card, args.max_synthetic)
    except ValueError as err:
        sys.exit(str(err))
    save_seed(args.out, pairs)
    print(f"seed rows {len(pairs)} (real crops {n_real}, catalog images {n_images}) -> {args.out}")
    return len(pairs)


if __name__ == "__main__":
    main()
