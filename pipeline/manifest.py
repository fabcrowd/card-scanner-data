"""The manifest the extension downloads first (v2): which bundle parts exist, how to check each, how well it did.

The bundle is keyed to the CLIP model and the crop version only, never to the catalog. Each part (memory, adapter,
check, catalogExtra) is optional on its own and carries its own sha256, so a client downloads only what changed.
"""
import hashlib
import json

from .signing import sign_bytes, verify_signature

MIN_EXAM_CASES = 30
# Release file name of each part, in the order they are listed.
ASSET_FILES = {
    "memory": "community-pack.bin",
    "adapter": "adapter.bin",
    "check": "check.bin",
    "catalogExtra": "catalog-extra.bin",
}

__all__ = ["ASSET_FILES", "build_manifest", "is_improved", "manifest_bytes", "sign_bytes", "verify_signature"]


def is_improved(exam, regression):
    """Memory-only rule, used when the pipeline has no gate: the pack helps only if it beat the plain catalog on at
    least 30 held-out exam cards AND did not lose top-1 accuracy on the fixed regression set. Without a regression
    result it is never improved."""
    if not regression or regression["n"] <= 0 or regression["after"] < regression["before"]:
        return False
    return exam["after"] > exam["before"] and exam["n"] >= MIN_EXAM_CASES


def build_manifest(pins, clip, assets, base_url, exam, regression, improved, seq, issued_at):
    """assets: {name: bytes} for the parts present (names in ASSET_FILES). base_url ends with "/"; each part is
    published at base_url + its file name.

    seq (the workflow run number) only goes up and issued_at is UTC; the extension refuses a lower seq or an old
    issuedAt, so a stale signed manifest cannot be replayed.
    """
    unknown = set(assets) - set(ASSET_FILES)
    if unknown:
        raise ValueError(f"unknown bundle parts: {sorted(unknown)}")
    return {
        "v": 2,
        "model": clip,
        "crop": pins["cropVersion"],
        "seq": seq,
        "issuedAt": issued_at,
        "assets": {
            name: {"url": base_url + ASSET_FILES[name], "sha256": hashlib.sha256(assets[name]).hexdigest(), "bytes": len(assets[name])}
            for name in ASSET_FILES if name in assets
        },
        "exam": exam,
        "regression": regression,
        "improved": bool(improved),
    }


def manifest_bytes(manifest):
    """The exact bytes that are published as manifest.json and signed."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
