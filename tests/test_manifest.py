import base64
import hashlib
import json
import unittest

import _paths  # noqa: F401
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from pipeline.manifest import ASSET_FILES, build_manifest, is_improved, manifest_bytes, sign_bytes, verify_signature
from pipeline.signing import generate_keypair

PINS = {"cropVersion": 2, "dim": 512}
EXAM = {"before": 0.5, "after": 0.6, "n": 40, "wrongAtHigh": {"before": 0.1, "after": 0.05}}
REGRESSION = {"before": 0.8, "after": 0.8, "n": 50}
BASE = "https://example.com/dl/"


def manifest(**over):
    args = dict(
        pins=PINS, clip="77538aade655907f", assets={"memory": b"pack"}, base_url=BASE, exam=EXAM,
        regression=REGRESSION, improved=True, seq=7, issued_at="2026-10-15T03:17:00Z",
    )
    args.update(over)
    return build_manifest(**args)


class Manifest(unittest.TestCase):
    def test_fields(self):
        m = manifest()
        self.assertEqual(set(m), {"v", "model", "crop", "seq", "issuedAt", "assets", "exam", "regression", "improved"})
        self.assertEqual((m["v"], m["model"], m["crop"]), (2, "77538aade655907f", 2))
        self.assertEqual(m["exam"]["n"], 40)
        self.assertIs(m["improved"], True)
        self.assertEqual((m["seq"], m["issuedAt"]), (7, "2026-10-15T03:17:00Z"))
        self.assertEqual(m["regression"], REGRESSION)
        self.assertNotIn("catalog", m)

    def test_every_part_has_url_sha256_and_size(self):
        parts = {"memory": b"m", "adapter": b"aa", "check": b"ccc", "catalogExtra": b"dddd"}
        m = manifest(assets=parts)
        self.assertEqual(list(m["assets"]), ["memory", "adapter", "check", "catalogExtra"])
        for name, data in parts.items():
            a = m["assets"][name]
            self.assertEqual(a["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(a["bytes"], len(data))
            self.assertEqual(a["url"], BASE + ASSET_FILES[name])

    def test_parts_are_optional_and_unknown_parts_are_refused(self):
        self.assertEqual(list(manifest(assets={"adapter": b"x"})["assets"]), ["adapter"])
        with self.assertRaises(ValueError):
            manifest(assets={"evil": b"x"})

    def test_improved_is_what_the_caller_decided(self):
        self.assertIs(manifest(improved=False)["improved"], False)

    def test_memory_only_rule(self):
        self.assertIs(is_improved({**EXAM, "n": 29}, REGRESSION), False)
        self.assertIs(is_improved({**EXAM, "after": 0.5}, REGRESSION), False)
        self.assertTrue(is_improved({"before": 0.0, "after": 0.1, "n": 30}, REGRESSION))
        self.assertIs(is_improved(EXAM, {"before": 0.8, "after": 0.79, "n": 50}), False)
        self.assertIs(is_improved(EXAM, {"before": 0.0, "after": 0.0, "n": 0}), False)
        self.assertIs(is_improved(EXAM, None), False)

    def test_manifest_bytes_are_stable_compact_json(self):
        a = manifest_bytes(manifest())
        self.assertEqual(a, manifest_bytes(manifest()))
        self.assertNotIn(b" ", a)
        self.assertEqual(json.loads(a)["seq"], 7)


class Signing(unittest.TestCase):
    def setUp(self):
        self.pem, self.spki_b64 = generate_keypair()

    def test_signature_is_64_raw_bytes_and_verifies_with_the_spki_public_key(self):
        data = manifest_bytes(manifest())
        sig = sign_bytes(data, self.pem)
        self.assertEqual(len(base64.b64decode(sig)), 64)
        self.assertTrue(verify_signature(data, sig, self.spki_b64))

    def test_tampered_manifest_or_wrong_key_fails(self):
        data = manifest_bytes(manifest())
        sig = sign_bytes(data, self.pem)
        self.assertFalse(verify_signature(data + b" ", sig, self.spki_b64))
        _, other = generate_keypair()
        self.assertFalse(verify_signature(data, sig, other))
        self.assertFalse(verify_signature(data, "not base64!!", self.spki_b64))
        self.assertFalse(verify_signature(data, base64.b64encode(b"short").decode(), self.spki_b64))

    def test_public_key_is_p256_spki_der(self):
        key = serialization.load_der_public_key(base64.b64decode(self.spki_b64))
        self.assertIsInstance(key.curve, ec.SECP256R1)

    def test_private_key_is_pkcs8_pem(self):
        self.assertTrue(self.pem.startswith(b"-----BEGIN PRIVATE KEY-----"))
        key = serialization.load_pem_private_key(self.pem, password=None)
        self.assertIsInstance(key.curve, ec.SECP256R1)

    def test_signing_without_a_key_fails_fast(self):
        with self.assertRaises(ValueError):
            sign_bytes(b"x", b"")


if __name__ == "__main__":
    unittest.main()
