"""The sign job's gate: it re-checks what the unprivileged build job produced before it signs anything."""
import base64
import datetime
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import helpers as h
from pipeline.manifest import ASSET_FILES, build_manifest, manifest_bytes
from pipeline.new_sets import write_extra
from pipeline.pack import write_pack
from pipeline.signing import generate_keypair, verify_signature

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sign_release.py")
spec = importlib.util.spec_from_file_location("sign_release", SCRIPT)
sign_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sign_release)

BASE = "https://github.com/fabcrowd/card-scanner-data/releases/download/pack-20261015-9/"
NOW = datetime.datetime(2026, 10, 15, 3, 30, tzinfo=datetime.timezone.utc)
PINS = {"cropVersion": 2, "dim": 512}
EXAM = {"before": 0.5, "after": 0.6, "n": 40, "wrongAtHigh": {"before": 0.1, "after": 0.05}}
REG = {"before": 0.8, "after": 0.8, "n": 20}
CARD = {"id": "sv9-1", "name": "Pikachu", "set": "sv9", "setName": "Journey Together", "number": "1", "image": "https://assets.tcgdex.net/en/sv/sv9/1/high.webp"}


def with_header(header, body):
    head = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(head)) + head + body


def adapter(rank=4, scale=0.01):
    return with_header({"v": 1, "dim": 512, "rank": rank, "scale": scale}, bytes(2 * 512 * rank))


def check(bias=-1.0, threshold=0.4, weights=None):
    w = np.zeros(512, dtype="<f4") if weights is None else weights
    return with_header({"v": 1, "dim": 512, "bias": bias, "threshold": threshold}, w.tobytes())


def parts(**over):
    out = {
        "memory": write_pack(["c-1", "c-2"], np.stack([h.unit(1), h.unit(2)])),
        "adapter": adapter(),
        "check": check(),
        "catalogExtra": write_extra([CARD], np.zeros((1, 512), dtype=np.int8)),
    }
    out.update(over)
    return {k: v for k, v in out.items() if v is not None}


def release(files=None, **over):
    files = parts() if files is None else files
    args = dict(pins=PINS, clip="77538aade655907f", assets=files, base_url=BASE, exam=EXAM, regression=REG, improved=True, seq=9, issued_at="2026-10-15T03:17:00Z")
    args.update(over)
    return manifest_bytes(build_manifest(**args)), files


def tamper(raw, edit):
    m = json.loads(raw)
    edit(m)
    return json.dumps(m, sort_keys=True, separators=(",", ":")).encode()


def verify(manifest, files, **over):
    kw = dict(base_url=BASE, run_number=9, now=NOW)
    kw.update(over)
    return sign_release.verify_release(manifest, files, **kw)


class Verify(unittest.TestCase):
    def refused(self, manifest, files, pattern, **over):
        with self.assertRaisesRegex(sign_release.Refused, pattern):
            verify(manifest, files, **over)

    def test_a_good_release_passes_with_every_part_and_with_any_one(self):
        manifest, files = release()
        self.assertEqual(sorted(verify(manifest, files)["assets"]), ["adapter", "catalogExtra", "check", "memory"])
        for name in ASSET_FILES:
            only = {name: parts()[name]}
            manifest, files = release(files=only)
            self.assertEqual(list(verify(manifest, files)["assets"]), [name])

    def test_a_file_that_does_not_match_its_entry_is_refused(self):
        manifest, files = release()
        self.refused(manifest, {**files, "adapter": files["adapter"] + b"\0"}, "adapter")
        self.refused(manifest, {k: v for k, v in files.items() if k != "check"}, "check file is missing")
        swapped = tamper(manifest, lambda m: m["assets"]["check"].update(bytes=3))
        self.refused(swapped, files, "check")

    def test_urls_seq_and_issue_time_must_belong_to_this_run(self):
        manifest, files = release()
        self.refused(manifest, files, "url", base_url=BASE.replace("pack-20261015-9", "pack-20261015-10"))
        self.refused(tamper(manifest, lambda m: m["assets"]["memory"].update(url="https://evil.test/community-pack.bin")), files, "url")
        self.refused(manifest, files, "seq", run_number=10)
        self.refused(manifest, files, "issuedAt", now=NOW + datetime.timedelta(hours=3))
        old, files_old = release(issued_at="2026-01-01T00:00:00Z")
        self.refused(old, files_old, "issuedAt")

    def test_improved_cannot_be_claimed_against_the_numbers(self):
        for name, over in {
            "no regression set": dict(regression=None),
            "regression dropped": dict(regression={"before": 0.8, "after": 0.7, "n": 20}),
            "empty regression": dict(regression={"before": 0.0, "after": 0.0, "n": 0}),
            "exam dropped with 30 cards": dict(exam={**EXAM, "after": 0.4}),
            "more wrong highs": dict(exam={**EXAM, "wrongAtHigh": {"before": 0.1, "after": 0.2}}),
        }.items():
            manifest, files = release(**over)
            with self.assertRaisesRegex(sign_release.Refused, "improved", msg=name):
                verify(manifest, files)

    def test_improved_with_a_small_exam_is_fine_and_honest_not_improved_is_always_signable(self):
        manifest, files = release(exam={**EXAM, "n": 5, "after": 0.1})
        verify(manifest, files)  # the exam is not trusted below 30 cards
        for over in (dict(regression=None), dict(regression={"before": 0.8, "after": 0.7, "n": 20}), dict(exam={**EXAM, "after": 0.4})):
            manifest, files = release(improved=False, **over)
            self.assertIs(verify(manifest, files)["improved"], False)

    def test_unknown_fields_parts_and_old_manifests_are_refused(self):
        manifest, files = release()
        self.refused(tamper(manifest, lambda m: m.update(evil=1)), files, "fields")
        self.refused(tamper(manifest, lambda m: m.update(catalog="0" * 16)), files, "fields")
        self.refused(tamper(manifest, lambda m: m.update(v=1)), files, "v or crop")
        self.refused(tamper(manifest, lambda m: m["assets"].update(evil=m["assets"]["memory"])), files, "unknown")
        self.refused(tamper(manifest, lambda m: m["assets"]["memory"].update(extra=1)), files, "wrong fields")
        self.refused(tamper(manifest, lambda m: m.update(assets={})), files, "assets")
        self.refused(b"not json", files, "manifest")

    def test_memory_and_extra_rows_are_validated(self):
        bad_ids = write_pack(["", "c-2"], np.stack([h.unit(1), h.unit(2)]))
        self.refused(*release(files=parts(memory=bad_ids)), "card id")
        big = [f"c-{i}" for i in range(sign_release.MAX_ENTRIES + 1)]
        self.refused(*release(files=parts(memory=write_pack(big, np.tile(h.unit(1), (len(big), 1))))), "too many")
        self.refused(*release(files=parts(memory=parts()["memory"] + b"x")), "size")

    def test_extra_cards_must_match_their_ids_and_be_well_formed(self):
        zero = np.zeros((1, 512), dtype=np.int8)
        for bad in (
            {**CARD, "name": ""},
            {**CARD, "image": "http://assets.tcgdex.net/x.webp"},
            {**CARD, "image": "https://x/" + "a" * 400},
            {**CARD, "extra": 1},
            {k: v for k, v in CARD.items() if k != "number"},
        ):
            self.refused(*release(files=parts(catalogExtra=write_extra([bad], zero))), "catalogExtra")
        no_cards = with_header({"v": 1, "dim": 512, "entries": 1, "cardIds": ["sv9-1"]}, bytes(512))
        self.refused(*release(files=parts(catalogExtra=no_cards)), "cards")

    def test_adapter_and_check_are_validated(self):
        self.refused(*release(files=parts(adapter=adapter(rank=65))), "rank")
        self.refused(*release(files=parts(adapter=adapter(rank=0))), "rank")
        self.refused(*release(files=parts(adapter=adapter(scale=-1))), "scale")
        self.refused(*release(files=parts(adapter=adapter()[:-1])), "size")
        self.refused(*release(files=parts(check=check(threshold=1.5))), "threshold")
        self.refused(*release(files=parts(check=check(bias=float("nan")))), "bias")
        self.refused(*release(files=parts(check=check(weights=np.full(512, np.inf, dtype="<f4")))), "finite")
        self.refused(*release(files=parts(check=check()[:-4])), "size")


class Signing(unittest.TestCase):
    def write(self, out, manifest, files):
        with open(os.path.join(out, "manifest.json"), "wb") as f:
            f.write(manifest)
        for name, data in files.items():
            with open(os.path.join(out, ASSET_FILES[name]), "wb") as f:
                f.write(data)

    def args(self, out):
        return [sys.executable, SCRIPT, "--dir", out, "--base-url", BASE, "--run-number", "9", "--now", "2026-10-15T03:30:00Z"]

    def test_main_signs_the_exact_manifest_bytes(self):
        pem, spki = generate_keypair()
        manifest, files = release()
        with tempfile.TemporaryDirectory() as out:
            self.write(out, manifest, files)
            r = subprocess.run(self.args(out), env={**os.environ, "PACK_SIGNING_KEY": pem.decode()}, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(out, "manifest.sig")) as f:
                sig = f.read()
            self.assertEqual(len(base64.b64decode(sig)), 64)
            self.assertTrue(verify_signature(manifest, sig, spki))
            self.assertFalse(verify_signature(manifest + b" ", sig, spki))

    def test_refused_release_writes_no_signature_and_a_missing_key_stops_early(self):
        pem, _ = generate_keypair()
        manifest, files = release()
        with tempfile.TemporaryDirectory() as out:
            self.write(out, manifest, {**files, "adapter": files["adapter"] + b"x"})
            r = subprocess.run(self.args(out), env={**os.environ, "PACK_SIGNING_KEY": pem.decode()}, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(os.path.exists(os.path.join(out, "manifest.sig")))
            env = {k: v for k, v in os.environ.items() if k != "PACK_SIGNING_KEY"}
            self.assertNotEqual(subprocess.run(self.args(out), env=env, capture_output=True, text=True).returncode, 0)

    def test_a_part_the_manifest_lists_but_the_folder_lacks_is_refused(self):
        pem, _ = generate_keypair()
        manifest, files = release()
        with tempfile.TemporaryDirectory() as out:
            self.write(out, manifest, {k: v for k, v in files.items() if k != "check"})
            r = subprocess.run(self.args(out), env={**os.environ, "PACK_SIGNING_KEY": pem.decode()}, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("check", r.stderr)

    def test_the_script_imports_only_the_standard_library_and_cryptography(self):
        with open(SCRIPT, encoding="utf-8") as f:
            text = f.read()
        imports = {line.split()[1].split(".")[0] for line in text.splitlines() if line.startswith(("import ", "from "))}
        self.assertTrue(imports <= {"argparse", "base64", "datetime", "hashlib", "json", "math", "os", "re", "struct", "sys", "cryptography"}, imports)


if __name__ == "__main__":
    unittest.main()
