#!/usr/bin/env python3
"""Verifies the build job's output and signs manifest.json. Runs in the `sign` job, the only place the key exists.

    PACK_SIGNING_KEY=<PKCS8 PEM> python scripts/sign_release.py --dir out --base-url <release download url, ends with /> --run-number <n>

Deliberately self-contained: it imports nothing but the standard library and `cryptography`, and none of the
pipeline code, so a change to the (unprivileged) pipeline cannot reach the key. It refuses to sign unless
  * the manifest (v2) has exactly the known fields and only known parts,
  * every listed part is a file in --dir whose sha256 and size match, whose url is this run's release asset
    (--base-url + the part's file name), and whose contents are well formed within size limits,
  * seq is this run's number and issuedAt is within an hour of now,
  * `improved` is not claimed without the numbers to back it: it may only be true when the regression set is
    present and non-empty with no drop, the exam shows no drop once it has 30 cards, and wrong "high" answers
    did not rise. (False is always honest and signable: it tells clients to keep what they have.)
"""
import argparse
import base64
import datetime
import hashlib
import json
import math
import os
import re
import struct
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

MAX_ENTRIES = 20000
MAX_HEADER = 32 * 1024 * 1024
MAX_RANK = 64
MIN_EXAM_CASES = 30
DIM = 512
FIELDS = {"v", "model", "crop", "seq", "issuedAt", "assets", "exam", "regression", "improved"}
ASSET_FIELDS = {"url", "sha256", "bytes"}
ASSET_FILES = {
    "memory": "community-pack.bin",
    "adapter": "adapter.bin",
    "check": "check.bin",
    "catalogExtra": "catalog-extra.bin",
}
HEX16 = re.compile(r"[0-9a-f]{16}")
HEX64 = re.compile(r"[0-9a-f]{64}")
FIELD_MAX = 200
IMAGE_MAX = 300


class Refused(Exception):
    pass


def _rate(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1


def _score_block(name, block, extra=()):
    if not isinstance(block, dict) or not {"before", "after", "n"} <= set(block) <= {"before", "after", "n", *extra}:
        raise Refused(f"{name} has the wrong fields")
    if not (_rate(block["before"]) and _rate(block["after"]) and isinstance(block["n"], int) and not isinstance(block["n"], bool) and block["n"] >= 0):
        raise Refused(f"{name} numbers are out of range")
    wrong = block.get("wrongAtHigh")
    if wrong is not None and not (isinstance(wrong, dict) and set(wrong) == {"before", "after"} and _rate(wrong["before"]) and _rate(wrong["after"])):
        raise Refused(f"{name} wrongAtHigh is malformed")


def _split(data, what):
    """(header dict, body offset) of a part: uint32 header length, JSON header."""
    if len(data) < 4:
        raise Refused(f"{what} too short")
    (n,) = struct.unpack("<I", data[:4])
    if n > MAX_HEADER or 4 + n > len(data):
        raise Refused(f"{what} header length is wrong")
    try:
        header = json.loads(data[4 : 4 + n].decode("utf-8"))
    except ValueError as err:
        raise Refused(f"{what} header is not JSON") from err
    if not isinstance(header, dict) or header.get("v") != 1 or header.get("dim") != DIM:
        raise Refused(f"{what} header fields are wrong")
    return header, 4 + n


def _check_rows(data, what, extra_cards=False):
    header, at = _split(data, what)
    ids = header.get("cardIds")
    if not isinstance(ids, list) or header.get("entries") != len(ids):
        raise Refused(f"{what} header fields are wrong")
    if len(ids) > MAX_ENTRIES:
        raise Refused(f"{what} has too many entries")
    if not all(isinstance(i, str) and i and len(i) <= 64 for i in ids):
        raise Refused(f"{what} has an empty or oversized card id")
    if len(data) != at + len(ids) * DIM:
        raise Refused(f"{what} size does not match its header")
    if extra_cards:
        cards = header.get("cards")
        if not isinstance(cards, list) or len(cards) != len(ids):
            raise Refused(f"{what} cards do not match its card ids")
        for card, card_id in zip(cards, ids):
            ok = (
                isinstance(card, dict) and set(card) == {"id", "name", "set", "setName", "number", "image"} and card["id"] == card_id
                and all(isinstance(card[k], str) and 0 < len(card[k]) <= FIELD_MAX for k in ("name", "set", "setName", "number"))
                and isinstance(card["image"], str) and card["image"].startswith("https://") and len(card["image"]) <= IMAGE_MAX
            )
            if not ok:
                raise Refused(f"{what} cards are malformed")


def _check_adapter(data):
    header, at = _split(data, "adapter")
    rank, scale = header.get("rank"), header.get("scale")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= MAX_RANK:
        raise Refused("adapter rank is out of range")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise Refused("adapter scale must be a positive number")
    if len(data) != at + 2 * DIM * rank:
        raise Refused("adapter size does not match its header")


def _check_check(data):
    header, at = _split(data, "check")
    bias, threshold = header.get("bias"), header.get("threshold")
    if not isinstance(bias, (int, float)) or isinstance(bias, bool) or not math.isfinite(bias):
        raise Refused("check bias must be a finite number")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 < threshold < 1:
        raise Refused("check threshold must be between 0 and 1")
    if len(data) != at + DIM * 4:
        raise Refused("check size does not match its header")
    if not all(math.isfinite(w) for w in struct.unpack(f"<{DIM}f", data[at:])):
        raise Refused("check weights must be finite")


CHECKS = {
    "memory": lambda data: _check_rows(data, "memory"),
    "adapter": _check_adapter,
    "check": _check_check,
    "catalogExtra": lambda data: _check_rows(data, "catalogExtra", extra_cards=True),
}


def _parse_time(text):
    if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise Refused("issuedAt is not a UTC timestamp")
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)


def _improved_is_supported(exam, regression):
    """The numbers must not contradict `improved: true`. The nightly's gate decides the rest (it can compare against
    the live bundle, which this job cannot see)."""
    if not regression or regression["n"] <= 0 or regression["after"] < regression["before"]:
        return False
    if exam["n"] >= MIN_EXAM_CASES and exam["after"] < exam["before"]:
        return False
    wrong = exam.get("wrongAtHigh")
    return not (wrong and wrong["after"] > wrong["before"])


def verify_release(manifest_bytes, files, *, base_url, run_number, now):
    """files: {part name: bytes} for the parts the manifest lists. Returns the parsed manifest, or raises Refused."""
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except ValueError as err:
        raise Refused("manifest is not JSON") from err
    if not isinstance(manifest, dict) or set(manifest) != FIELDS:
        raise Refused("manifest fields are not the expected set")
    if manifest["v"] != 2 or isinstance(manifest["v"], bool) or not isinstance(manifest["crop"], int) or isinstance(manifest["crop"], bool):
        raise Refused("manifest v or crop is wrong")
    if not (isinstance(manifest["model"], str) and HEX16.fullmatch(manifest["model"])):
        raise Refused("manifest model is not 16 hex digits")
    if not isinstance(manifest["seq"], int) or isinstance(manifest["seq"], bool) or manifest["seq"] != run_number:
        raise Refused("manifest seq is not this run's number")
    if abs((_parse_time(manifest["issuedAt"]) - now).total_seconds()) > 3600:
        raise Refused("manifest issuedAt is not within an hour of now")

    assets = manifest["assets"]
    if not isinstance(assets, dict) or not assets or not set(assets) <= set(ASSET_FILES):
        raise Refused("manifest assets are missing or unknown")
    for name, asset in assets.items():
        if not isinstance(asset, dict) or set(asset) != ASSET_FIELDS:
            raise Refused(f"{name} entry has the wrong fields")
        if name not in files:
            raise Refused(f"{name} file is missing")
        data = files[name]
        if asset["url"] != base_url + ASSET_FILES[name]:
            raise Refused(f"{name} url is not this run's release asset")
        if not (isinstance(asset["sha256"], str) and HEX64.fullmatch(asset["sha256"])) or hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise Refused(f"{name} sha256 does not match the manifest")
        if not isinstance(asset["bytes"], int) or isinstance(asset["bytes"], bool) or asset["bytes"] != len(data):
            raise Refused(f"{name} size does not match the manifest")
        CHECKS[name](data)

    exam, regression = manifest["exam"], manifest["regression"]
    _score_block("exam", exam, extra=("wrongAtHigh",))
    if regression is not None:
        _score_block("regression", regression)
    if not isinstance(manifest["improved"], bool):
        raise Refused("improved is not a boolean")
    if manifest["improved"] and not _improved_is_supported(exam, regression):
        raise Refused("improved does not follow from the exam and regression numbers")
    return manifest


def sign(data, pem):
    """Base64 raw r||s ECDSA P-256 / SHA-256 signature."""
    key = serialization.load_pem_private_key(pem.encode(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise Refused("signing key must be ECDSA P-256")
    r, s = decode_dss_signature(key.sign(data, ec.ECDSA(hashes.SHA256())))
    return base64.b64encode(r.to_bytes(32, "big") + s.to_bytes(32, "big")).decode()


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base-url", required=True, help="this run's release download address, ending with /")
    ap.add_argument("--run-number", type=int, required=True)
    ap.add_argument("--now", help="UTC time for tests, as 2026-10-15T03:30:00Z")
    args = ap.parse_args(argv)

    pem = os.environ.get("PACK_SIGNING_KEY", "")
    if not pem.strip():
        sys.exit("PACK_SIGNING_KEY is not set")
    manifest_bytes = _read(os.path.join(args.dir, "manifest.json"))
    files = {name: _read(os.path.join(args.dir, filename)) for name, filename in ASSET_FILES.items() if os.path.exists(os.path.join(args.dir, filename))}
    now = _parse_time(args.now) if args.now else datetime.datetime.now(datetime.timezone.utc)
    try:
        verify_release(manifest_bytes, files, base_url=args.base_url, run_number=args.run_number, now=now)
        signature = sign(manifest_bytes, pem)
    except Refused as err:
        sys.exit(f"refusing to sign: {err}")
    with open(os.path.join(args.dir, "manifest.sig"), "w", encoding="ascii") as f:
        f.write(signature)
    print("signed", file=sys.stderr)


if __name__ == "__main__":
    main()
