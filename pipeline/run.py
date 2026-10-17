"""Nightly pipeline: contributions/ -> the bundle parts + manifest.json (unsigned) in --out.

    python -m pipeline.run --data data --assets assets --out out --base-url <url ending in /> --seq <run number>

Parts: community-pack.bin (memory) always; adapter.bin, check.bin, catalog-extra.bin when their trainer or the new-sets
step produced one. The trainers, seed, gate and status modules are optional: while one is missing its part is simply
left out and the gate falls back to the memory-only rule, so the pipeline still builds a memory-only bundle.

This step holds no secrets and signs nothing: the separate `sign` job verifies the output and signs the manifest
(scripts/sign_release.py). --data is only read as JSON files; nothing in it is executed or imported.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
import shutil
import sys

import numpy as np

from .catalog import Catalog
from .consensus import promote
from .contributions import build_samples, load_contributions
from .embed import ClipEmbedder
from .exam import TOP_K, exam_cases, score_exam
from .manifest import ASSET_FILES, build_manifest, is_improved, manifest_bytes
from .pack import read_pack, write_pack
from .regression import load_regression

OPTIONAL_MODULES = ("train_adapter", "train_check", "gate", "status")
# The learned parts a night can change; catalogExtra is separate because it adds cards, not learning.
LEARNED = ("memory", "adapter")
MIN_NOT_A_CARD_INSTALLS = 2


def optional_modules():
    """{name: module} for each optional pipeline module that exists. A module that exists but fails to import is an error."""
    found = {}
    for name in OPTIONAL_MODULES:
        full = f"{__package__}.{name}"
        try:
            found[name] = importlib.import_module(full)
        except ModuleNotFoundError as err:
            if err.name != full:
                raise
    return found


def _embed(samples, embedder):
    """Samples whose crop embeds, with their vectors; the rest are dropped and counted."""
    kept, vectors = [], []
    for s in samples:
        try:
            vectors.append(np.asarray(embedder(s.crop), dtype=np.float32))
            kept.append(s)
        except ValueError:
            continue
    dim = vectors[0].shape[0] if vectors else 512
    return kept, (np.stack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)), len(samples) - len(kept)


def _not_a_card(samples, embeddings):
    """Embeddings of crops that two or more installs called "not a card" and nobody called a card. One install alone
    cannot teach the check what is not a card."""
    keep = [
        i for i, s in enumerate(samples)
        if not s.card_votes and sum(v.kind == "notACard" for v in s.votes.values()) >= MIN_NOT_A_CARD_INSTALLS
    ]
    return embeddings[keep] if keep else np.zeros((0, embeddings.shape[1]), dtype=np.float32)


def _scores(block):
    return {k: block[k] for k in ("before", "after", "n")}


def run_pipeline(data_dir, out_dir, pins, embedder, catalog, clip_hash, base_url, seq, issued_at, regression=None,
                 modules=None, live=None, seed_pairs=None, new_sets_fn=None, status_path=None, status_history=None):
    """Runs every stage and writes the parts and the unsigned manifest. Returns a summary dict.

    regression: (embeddings, card ids) of the fixed server-owned set, or None. Without it the manifest can never
        say improved.
    modules: {name: module} of the optional pipeline modules (default: those that exist).
    live: {"memory": bytes, "adapter": bytes} of the live bundle, which the gate compares against. Missing means
        the plain catalog.
    seed_pairs: [(embedding, card id)] from seed.npz (scripts/build_seed.py), or None for no seed.
    new_sets_fn: () -> new_sets.build result, or None to add no cards.
    status_path / status_history: where STATUS.md is written, and the earlier status.json to carry forward.
    """
    modules = optional_modules() if modules is None else modules
    live = live or {}
    items, dropped = load_contributions(data_dir)
    valid_ids = set(catalog.ids)

    teach, teach_emb, teach_failed = _embed(build_samples(items, valid_ids=valid_ids, split="teach"), embedder)
    if len(teach):
        idx, _ = catalog.top_k_batch(teach_emb, TOP_K)
        neighbours = [{catalog.ids[j] for j in row} for row in idx]
    else:
        neighbours = []
    promotion = promote(teach, teach_emb, neighbours)
    promoted_vectors = np.stack([r.embedding for r in promotion.rows]) if promotion.rows else np.zeros((0, 512), dtype=np.float32)
    pack_bytes = write_pack([r.card_id for r in promotion.rows], promoted_vectors)
    pack = read_pack(pack_bytes)
    parts = {"memory": pack_bytes}

    pairs = [(r.embedding, r.card_id) for r in promotion.rows] + list(seed_pairs or [])
    if "train_adapter" in modules:
        adapter = modules["train_adapter"].train(pairs, catalog)
        if adapter:
            parts["adapter"] = adapter
    if "train_check" in modules:
        check = modules["train_check"].train(promoted_vectors, _not_a_card(teach, teach_emb))
        if check:
            parts["check"] = check

    new_cards = new_sets_fn() if new_sets_fn else None
    if new_cards and new_cards.get("bytes"):
        parts["catalogExtra"] = new_cards["bytes"]

    cases = exam_cases(items, valid_ids)
    exam_samples, exam_emb, exam_failed = _embed([s for s, _ in cases], embedder)
    truths = {s.key: t for s, t in cases}
    exam_truths = [truths[s.key] for s in exam_samples]
    has_regression = regression is not None and len(regression[1]) > 0
    outcomes = None

    if "gate" in modules:
        empty = (np.zeros((0, 512), dtype=np.float32), [])
        verdict = modules["gate"].evaluate(
            {k: parts[k] for k in LEARNED if k in parts}, live, (exam_emb, exam_truths), regression if has_regression else empty, catalog=catalog,
        )
        exam = {**_scores(verdict["exam"]), "wrongAtHigh": verdict["wrongHigh"]}
        regression_result = _scores(verdict["regression"]) if has_regression else None
        improved = bool(verdict["improved"])
        outcomes = verdict.get("outcomes")
        # A night that only adds cards changes nothing the gate can measure. It may ship when what was learned is
        # byte-identical to the live bundle and the fixed regression set shows no drop.
        if not improved and new_cards and new_cards["added"] and regression_result and regression_result["n"] > 0:
            same = all(parts.get(k) == live.get(k) for k in LEARNED)
            improved = same and regression_result["after"] >= regression_result["before"]
    else:
        exam = score_exam(
            exam_emb, exam_truths, catalog, pack.card_ids, pack.dequantized(),
            ref_min=pins["reference"]["min"], ref_margin=pins["reference"]["margin"],
            high_sim=pins["highTier"]["sim"], high_gap=pins["highTier"]["gap"],
        )
        regression_result = None
        if has_regression:
            scored = score_exam(
                regression[0], regression[1], catalog, pack.card_ids, pack.dequantized(),
                ref_min=pins["reference"]["min"], ref_margin=pins["reference"]["margin"],
                high_sim=pins["highTier"]["sim"], high_gap=pins["highTier"]["gap"],
            )
            regression_result = _scores(scored)
        improved = is_improved(exam, regression_result)

    manifest = build_manifest(pins, clip_hash, parts, base_url, exam, regression_result, improved, seq, issued_at)
    raw = manifest_bytes(manifest)

    os.makedirs(out_dir, exist_ok=True)
    for name, data in parts.items():
        with open(os.path.join(out_dir, ASSET_FILES[name]), "wb") as f:
            f.write(data)
    with open(os.path.join(out_dir, "manifest.json"), "wb") as f:
        f.write(raw)

    summary = {
        "entries": len(pack.card_ids),
        "parts": sorted(parts),
        "improved": manifest["improved"],
        "exam": exam,
        "regression": regression_result,
        "outcomes": outcomes,
        "newCards": len(new_cards["added"]) if new_cards else 0,
        "newSets": new_cards["sets"] if new_cards else [],
        "contributionsDropped": dropped,
        "embedFailed": teach_failed + exam_failed,
        **promotion.stats,
    }
    if "status" in modules and status_path:
        if status_history and os.path.exists(status_history):
            shutil.copyfile(status_history, os.path.join(os.path.dirname(os.path.abspath(status_path)), "status.json"))
        modules["status"].write(status_path, {
            "date": issued_at[:10],
            "seq": seq,
            "contributors": len({it["install"] for it in items}),
            "promoted": len(promotion.rows),
            "newSets": summary["newSets"],
            "regression": regression_result and {k: regression_result[k] for k in ("before", "after")},
            "exam": {k: exam[k] for k in ("before", "after", "n")},
            "shipped": manifest["improved"],
        })
    return summary


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


MAX_SEED_ROWS = 400000


def load_seed(path):
    """[(float32[512] unit embedding, card id)] from seed.npz, or [] when the file is missing. The file comes from the
    data repo's main branch (owner-committed), is read without pickle, and a malformed one stops the run."""
    if not path or not os.path.exists(path):
        return []
    with np.load(path, allow_pickle=False) as z:
        vectors, ids = np.asarray(z["embeddings"], dtype=np.float32), z["cardIds"]
    if vectors.ndim != 2 or vectors.shape[1] != 512 or len(ids) != len(vectors) or len(ids) > MAX_SEED_ROWS or not np.all(np.isfinite(vectors)):
        raise ValueError(f"{path} is not a valid seed file")
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return [(vectors[i], str(ids[i])) for i in range(len(ids))]


def load_live(folder):
    """The live bundle's parts that exist in `folder` (its release assets), as {name: bytes}."""
    live = {}
    for name, filename in ASSET_FILES.items():
        path = os.path.join(folder, filename) if folder else None
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                live[name] = f.read()
    return live


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=".", help="data repo root (holds contributions/)")
    ap.add_argument("--assets", default="assets", help="folder written by scripts/fetch_assets.py")
    ap.add_argument("--out", default="out")
    ap.add_argument("--pins", default="pins.json")
    ap.add_argument("--base-url", required=True, help="where the extension will download the parts; ends with /")
    ap.add_argument("--seq", type=int, required=True, help="the workflow run number; only ever goes up")
    ap.add_argument("--regression", default="regression.npz", help="fixed regression set (scripts/build_regression.py); missing means never improved")
    ap.add_argument("--live-dir", default="live", help="the live release's assets, for the gate and to extend catalog-extra.bin; may be missing")
    ap.add_argument("--status-history", default="live-status/status.json", help="earlier status.json to carry forward; may be missing")
    ap.add_argument("--seed", default="seed.npz", help="seed embeddings from scripts/build_seed.py, committed to main; may be missing")
    ap.add_argument("--no-new-sets", action="store_true", help="skip the card-source requests")
    args = ap.parse_args(argv)
    if not args.base_url.endswith("/"):
        sys.exit("--base-url must end with /")

    with open(args.pins, encoding="utf-8") as f:
        pins = json.load(f)
    paths = {a["name"]: os.path.join(args.assets, a["name"]) for a in pins["assets"]}
    for a in pins["assets"]:
        if _sha256_file(paths[a["name"]]) != a["sha256"]:
            sys.exit(f"{a['name']} does not match its pinned sha256; refusing to run")
    hashes = {a["name"]: a["sha256"][:16] for a in pins["assets"]}

    embedder = ClipEmbedder(paths["clip_visual_int8.onnx"])
    catalog = Catalog.load(paths["catalog.bin.gz"], paths["id_map.json"])
    live = load_live(args.live_dir)
    new_sets_fn = None
    if not args.no_new_sets:
        from . import new_sets

        new_sets_fn = lambda: new_sets.build(catalog.ids, live.get("catalogExtra"), embedder)  # noqa: E731

    os.makedirs(args.out, exist_ok=True)
    summary = run_pipeline(
        data_dir=args.data, out_dir=args.out, pins=pins, embedder=embedder, catalog=catalog,
        clip_hash=hashes["clip_visual_int8.onnx"], base_url=args.base_url, seq=args.seq,
        issued_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        regression=load_regression(args.regression) if os.path.exists(args.regression) else None,
        live=live, new_sets_fn=new_sets_fn, seed_pairs=load_seed(args.seed),
        status_path=os.path.join(args.out, "STATUS.md"), status_history=args.status_history,
    )
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
