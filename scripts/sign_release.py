#!/usr/bin/env python3
"""Verifies the build job's output and signs manifest.json. Runs in the `sign` job, the only place the key exists.

    PACK_SIGNING_KEY=<PKCS8 PEM> python scripts/sign_release.py --dir out --base-url <release download url, ends with /> --run-number <n>

Deliberately self-contained: it imports nothing but the standard library and `cryptography`, and none of the
pipeline code, so a change to the (unprivileged) pipeline cannot reach the key. The one exception is
pipeline/image_hosts.py, a file of constants with no imports, shared so the build and the signer use one host list.

    ... --reissue-live-dir <fresh download of the latest release>   (a quiet night republishing the live bundle)

Then the live manifest.json must verify against manifest.sig with the public key of the signing key, and the manifest
being signed must equal it in every field except seq (higher, this run's number) and issuedAt (later, within an hour of
now), keep its urls, and pass every check below against the part files. The live directory must be downloaded by the
sign job itself, so an older signed manifest handed over by the build job cannot be re-issued.

It refuses to sign unless
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

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature

# The one list of picture hosts is shared with the build; the sign job checks out pipeline/image_hosts.py for it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.image_hosts import IMAGE_HOSTS  # noqa: E402

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
# Exit code of a refusal (any other failure exits 1), so the workflow can treat a refused re-issue as a recorded outcome.
REFUSED_EXIT = 3
# A re-issue may change only these manifest fields.
REISSUE_CHANGES = ("seq", "issuedAt")
# A refused re-issue only warns, so a live manifest this old (or of unknown age) fails the job instead: clients drop a
# manifest at 30 days (MAX_MANIFEST_AGE_MS in bundle.js), and nothing else would make nights of refusals visible.
REISSUE_FAIL_AFTER_DAYS = 21
FIELD_MAX = 200
IMAGE_MAX = 300


class Refused(Exception):
    pass


class KeyProblem(Exception):
    """The signing key itself is unusable. Not a refusal about the content: it exits 1 in every mode."""


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
                and isinstance(card["image"], str) and card["image"].startswith(IMAGE_HOSTS) and len(card["image"]) <= IMAGE_MAX
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


def _same_but_seq_and_time(manifest, live):
    return {k: v for k, v in manifest.items() if k not in REISSUE_CHANGES} == {k: v for k, v in live.items() if k not in REISSUE_CHANGES}


def verify_release(manifest_bytes, files, *, base_url, run_number, now, live=None):
    """files: {part name: bytes} for the parts the manifest lists. Returns the parsed manifest, or raises Refused.

    live: None for a normal release. For a re-issue, the parsed manifest that is live now (its signature already
    verified by verify_live). The manifest must then equal it in every field except seq and issuedAt, with a higher seq
    and a later issuedAt; its urls stay the live release's, so base_url is not used. Every other check still applies."""
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
    issued = _parse_time(manifest["issuedAt"])
    if abs((issued - now).total_seconds()) > 3600:
        raise Refused("manifest issuedAt is not within an hour of now")
    if live is not None:
        if not _same_but_seq_and_time(manifest, live):
            raise Refused("re-issue differs from the live manifest in more than seq and issuedAt")
        if manifest["improved"] is not True:
            raise Refused("re-issue of a manifest that is not improved")
        if not isinstance(live["seq"], int) or isinstance(live["seq"], bool) or manifest["seq"] <= live["seq"]:
            raise Refused("re-issue seq is not above the live seq")
        if issued <= _parse_time(live["issuedAt"]):
            raise Refused("re-issue issuedAt is not later than the live issuedAt")

    assets = manifest["assets"]
    if not isinstance(assets, dict) or not assets or not set(assets) <= set(ASSET_FILES):
        raise Refused("manifest assets are missing or unknown")
    for name, asset in assets.items():
        if not isinstance(asset, dict) or set(asset) != ASSET_FIELDS:
            raise Refused(f"{name} entry has the wrong fields")
        if name not in files:
            raise Refused(f"{name} file is missing")
        data = files[name]
        if live is None and asset["url"] != base_url + ASSET_FILES[name]:
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


def _load_key(pem):
    """The ECDSA P-256 private key in pem, or KeyProblem. The message never repeats the key text."""
    try:
        key = serialization.load_pem_private_key(pem.encode(), password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as err:
        raise KeyProblem(f"signing key could not be loaded ({type(err).__name__})") from None
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise KeyProblem("signing key must be ECDSA P-256")
    return key


def sign(data, pem):
    """Base64 raw r||s ECDSA P-256 / SHA-256 signature."""
    key = _load_key(pem)
    r, s = decode_dss_signature(key.sign(data, ec.ECDSA(hashes.SHA256())))
    return base64.b64encode(r.to_bytes(32, "big") + s.to_bytes(32, "big")).decode()


def verify_live(manifest_bytes, signature_b64, pem):
    """The parsed live manifest, once its signature verifies with the public key of the signing key. Raises Refused
    (KeyProblem if the key itself is unusable)."""
    key = _load_key(pem)
    try:
        raw = base64.b64decode(signature_b64.strip(), validate=True)
        if len(raw) != 64:
            raise ValueError("signature is not 64 bytes")
        key.public_key().verify(encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")), manifest_bytes, ec.ECDSA(hashes.SHA256()))
        live = json.loads(manifest_bytes.decode("utf-8"))
    except (InvalidSignature, ValueError) as err:
        raise Refused("live manifest signature does not verify with the pack public key") from err
    if not isinstance(live, dict):
        raise Refused("live manifest is not an object")
    return live


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _live_is_nearly_stale(live_dir, now):
    """True when the live manifest's issuedAt (the field clients check against their 30-day limit) is
    REISSUE_FAIL_AFTER_DAYS or more before now, or cannot be read."""
    try:
        issued = _parse_time(json.loads(_read(os.path.join(live_dir, "manifest.json")).decode("utf-8"))["issuedAt"])
    except (OSError, ValueError, KeyError, TypeError, Refused):
        return True
    return now - issued >= datetime.timedelta(days=REISSUE_FAIL_AFTER_DAYS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base-url", required=True, help="this run's release download address, ending with /")
    ap.add_argument("--run-number", type=int, required=True)
    ap.add_argument("--now", help="UTC time for tests, as 2026-10-15T03:30:00Z")
    ap.add_argument("--reissue-live-dir", help="a re-issue of the live bundle: a fresh download of the latest release (manifest.json and manifest.sig)")
    args = ap.parse_args(argv)

    pem = os.environ.get("PACK_SIGNING_KEY", "")
    if not pem.strip():
        sys.exit("PACK_SIGNING_KEY is not set")
    manifest_bytes = _read(os.path.join(args.dir, "manifest.json"))
    files = {name: _read(os.path.join(args.dir, filename)) for name, filename in ASSET_FILES.items() if os.path.exists(os.path.join(args.dir, filename))}
    now = _parse_time(args.now) if args.now else datetime.datetime.now(datetime.timezone.utc)
    try:
        live = None
        if args.reissue_live_dir:
            try:
                live_bytes = _read(os.path.join(args.reissue_live_dir, "manifest.json"))
                live_signature = _read(os.path.join(args.reissue_live_dir, "manifest.sig")).decode("ascii")
            except (OSError, ValueError) as err:
                raise Refused(f"live manifest or its signature could not be read: {err}") from err
            live = verify_live(live_bytes, live_signature, pem)
        verify_release(manifest_bytes, files, base_url=args.base_url, run_number=args.run_number, now=now, live=live)
        signature = sign(manifest_bytes, pem)
    except KeyProblem as err:
        sys.exit(f"cannot sign: {err}")
    except Refused as err:
        print(f"refusing to sign: {err}", file=sys.stderr)
        if args.reissue_live_dir and _live_is_nearly_stale(args.reissue_live_dir, now):
            sys.exit(f"the live bundle is {REISSUE_FAIL_AFTER_DAYS} or more days old, or its age is unknown, and clients stop accepting it at 30 days: failing the job so the refused re-issue is not missed")
        sys.exit(REFUSED_EXIT)
    with open(os.path.join(args.dir, "manifest.sig"), "w", encoding="ascii") as f:
        f.write(signature)
    print("signed", file=sys.stderr)


if __name__ == "__main__":
    main()
